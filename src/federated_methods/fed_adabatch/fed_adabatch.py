import math
import time
import torch
import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from client_selectors.const_weight import load_probabilities_from_json
from .fed_adabatch_client import FedAdaBatchClient
from .fed_adabatch_server import FedAdaBatchServer
from ..fedavg_one_sample.fedavg_os import FedAvgOS


class FedAdaBatch(FedAvgOS):
    ignore_bn_keywords = [
        "bn",
        "batchnorm",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ]

    def __init__(
        self,
        theta_norm,
        theta_inner,
        nu_ortho,
        checkpoint_path,
        ht_mk_k,
        training_type,
        local_iters,
        tests,
        recalc_every_round,
    ):
        super().__init__()
        self.theta_norm = theta_norm
        self.theta_inner = theta_inner
        self.nu_ortho = nu_ortho
        self.checkpoint_path = checkpoint_path
        self.training_type = training_type
        self.local_iters = local_iters
        self.tests = tests
        self.recalc_every_round = recalc_every_round
        self.ht_mk_k = ht_mk_k
        self.need_calcualte_pi_from_start = True
        self.eps = 1e-5
        print(f"\n[AdaBatch]: using tests: {self.tests}\n")

    def _init_server(self, cfg):
        self.server = FedAdaBatchServer(cfg, self.checkpoint_path)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedAdaBatchClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.append(self.training_type)
        self.client_args.append(self.local_iters)

    def aggregate(self):
        aggr_weights = super(FedAvgOS, self).aggregate()
        new_amount_cl = self.define_amount_clients()

        # 2 <= new amount clients <= all clients
        self.num_clients_subset = min(new_amount_cl, self.amount_of_clients)
        self.num_clients_subset = max(2, self.num_clients_subset)

        # Clear memory after aggregate
        self.server.client_gradients = [
            OrderedDict() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
        self.server.server_metrics = [
            pd.DataFrame() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
        return aggr_weights

    def define_amount_clients(self):
        if (self.cur_round == 0) and (self.need_calcualte_pi_from_start):
            # self.client_weights = load_probabilities_from_json(
            #     self.cfg.client_selector.weights_path
            # )
            # ---- estimate pi_i and pi_ij ----
            self.ht_pi_i_s = []
            self.ht_pi_ij_s = []

            for i in range(1, self.amount_of_clients + 1):
                print(f"Start Monte-Carlo estimation of pi_i and pi_ij for {i} clients")
                ht_pi, ht_pij = self.estimate_ht_pi_and_pij_by_monte_carlo(i)

                self.ht_pi_i_s.append(ht_pi)
                self.ht_pi_ij_s.append(ht_pij)

        current = self.num_clients_subset or len(self.list_clients)

        if self.need_calcualte_pi_from_start:
            self.ht_pi = self.ht_pi_i_s[current - 1]
            self.ht_pij = self.ht_pi_ij_s[current - 1]
            self.client_weights = self.ht_pi / sum(self.ht_pi) # temp stuff
            print(f"Clients weights:\n{self.client_weights}\n\n")
        else:
            self.ht_pi, self.ht_pij = self.estimate_ht_pi_and_pij_by_monte_carlo(
                current
            )
            self.client_weights = self.ht_pi / sum(self.ht_pi) # temp stuff
            print(f"Clients weights:\n{self.client_weights}\n\n")


        self.ht_pi_t = torch.tensor(self.ht_pi, dtype=torch.float)
        self.ht_pi_t[self.ht_pi_t == 0] = self.eps

        self.ht_pij_t = torch.tensor(self.ht_pij, dtype=torch.float)
        self.ht_pij_t[self.ht_pij_t == 0] = self.eps

        print(f"HT pi_i:\n{self.ht_pi_t}\n")
        print(f"HT pi_ij:\n{self.ht_pij_t}\n")

        results = {}  # {test_name: (ok, n_new)}

        for test_name in self.tests:
            method_name = f"{test_name}_test"
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                # try:
                ok, n_new = method()
                # except Exception as e:
                # print(f"[AdaBatch] ❌ Error while run {method_name}: {e}")
                # ok, n_new = False, None
                results[test_name] = (ok, n_new)
            else:
                print(f"[AdaBatch] ⚠️ Test {test_name} not found. Skip.")

        print("[AdaBatch] Results (test_res, new_bs):")
        for name, (ok, n_new) in results.items():
            print(f"{name}=({ok}, {n_new})")

        # parsing results
        if self.recalc_every_round:
            candidates = [n for _, n in results.values() if n is not None]
            if candidates:
                n_new = max(candidates)
                print(f"[AdaBatch] All test passed, new bs = {n_new}")
            else:
                n_new = current
                print(
                    f"[AdaBatch] None of the tests offered a new bs, keep bs ={n_new}"
                )
        else:
            if all(ok for ok, _ in results.values()):
                n_new = current
                print(f"[AdaBatch] All test passed, keep bs = {n_new}")
            else:
                candidates = [n for _, n in results.values() if n is not None]
                if candidates:
                    n_new = max(candidates)
                    print(f"[AdaBatch] Not all passed, new bs = {n_new}")
                else:
                    n_new = current
                    print(
                        f"[AdaBatch] None of the tests offered a new bs, keep bs ={n_new}"
                    )

        # if self.cur_round % 5 == 0:
        #     self.get_variance_clients_stat()
        # n_new = 99
        return n_new

    # ===== TESTS =====

    def ht_real_var_test(self):
        print("\nStart computing ht_real_var_test\n")

        # ---- initial data ----
        N = self.amount_of_clients
        S = list(self.list_clients)  # list[int]
        grads = torch.stack(self._collect_client_vectors(), dim=0)  # [m,d]

        def compute_test(S, grads):
            selected = torch.tensor(S, device=grads.device)
            mu_ht = self.get_ht_mean(grads, selected)
            rhs = self.get_ht_rhs(mu_ht)
            var = self.get_ht_var(grads, selected)
            return var, rhs

        # ---- initial test ----
        ht_var, rhs = compute_test(S, grads)
        print(f"[ht_real_var | |S|={len(S)}]: {ht_var:.6e} <= {rhs:.6e}")

        # ---------- CASE 1: test PASSES → try shrinking ----------
        if ht_var < rhs:
            S_cur = S.copy()
            grads_cur = grads.clone()

            while len(S_cur) > 2:
                # choose client to remove
                weights = self.client_weights[S_cur]
                idx_remove = int(np.argmax(weights))
                removed_client = S_cur[idx_remove]

                # remove
                S_next = S_cur[:idx_remove] + S_cur[idx_remove + 1 :]
                grads_next = torch.cat(
                    [grads_cur[:idx_remove], grads_cur[idx_remove + 1 :]],
                    dim=0,
                )

                new_var, new_rhs = compute_test(S_next, grads_next)
                print(f"[shrink | |S|={len(S_next)}]: {new_var:.6e} <= {new_rhs:.6e}")

                if new_var >= new_rhs:
                    # went too far
                    return True, len(S_cur)

                S_cur, grads_cur = S_next, grads_next

            return True, 2

        # ---------- CASE 2: test FAILS → try expanding ----------
        else:
            # Clients not yet in S
            remaining = [i for i in range(N) if i not in S]

            S_cur = S.copy()  # list[int]
            grads_cur = grads.clone()

            while remaining:
                # ---- choose client to add (heuristic) ----
                weights = self.client_weights[remaining]
                idx_add = int(np.argmin(weights))
                add_client = remaining.pop(idx_add)

                # ---- approximate gradient for added client ----
                selected_clients = torch.tensor(S_cur, device=grads_cur.device)

                add_grad = self.approx_grad_for_add_client(
                    client_grads=grads_cur,  # grads of S_cur
                    selected_clients=selected_clients,
                    add_client=add_client,
                ).unsqueeze(0)

                # ---- form new S and grads ----
                S_next = S_cur + [add_client]
                grads_next = torch.cat([grads_cur, add_grad], dim=0)

                # ---- compute test on expanded set ----
                new_var, new_rhs = compute_test(S_next, grads_next)

                print(f"[expand | |S|={len(S_next)}]: {new_var:.6e} <= {new_rhs:.6e}")

                # ---- check success ----
                if new_var < new_rhs:
                    return True, len(S_next)

                # ---- continue expanding ----
                S_cur, grads_cur = S_next, grads_next

            # If all clients added and still fails
            return True, N

    def get_ht_mean(self, client_grads, selected_clients):
        pi_sel = self.ht_pi_t[selected_clients]  # [m]
        grads_div = client_grads / pi_sel.unsqueeze(1)
        mu_ht = grads_div.sum(dim=0) / float(self.amount_of_clients)  # hat mu_HT
        return mu_ht

    def get_ht_var(self, client_grads, selected_clients):
        """
        Var(mu_HT) =
            1/N^2 * sum_{i,j in S}
            (g_i^T g_j) / (pi_i pi_j)
            * (pi_ij - pi_i pi_j) / pi_ij
        """
        N = self.amount_of_clients

        # ---- pi_i and pi_ij ----
        pi_sel = self.ht_pi_t[selected_clients]  # [m]
        pij_sel = self.ht_pij_t[
            selected_clients[:, None], selected_clients[None, :]
        ]  # [m, m]

        pi_i = pi_sel.unsqueeze(1)  # [m,1]
        pi_j = pi_sel.unsqueeze(0)  # [1,m]

        # ---- coefficient: (pi_ij - pi_i pi_j) / pi_ij ----
        coef = (pij_sel - pi_i * pi_j) / pij_sel
        coef = torch.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- y_i^T y_j / (pi_i pi_j) ----
        grads_div = client_grads / pi_sel.unsqueeze(1)  # [m,d]
        gram = grads_div @ grads_div.T  # [m,m]

        # ---- variance ----
        var_ht = (coef * gram).sum() / (N**2)

        return var_ht.item()

    def get_ht_rhs(self, mu_ht):
        rhs = (self.theta_norm**2) * mu_ht.pow(2).sum().item()
        return rhs

    def approx_grad_for_add_client(self, client_grads, selected_clients, add_client):
        """
        Approximate gradient for adding client j using
        sum_{i in S} (pi_{i,j} / (pi_i * pi_j)) * grad_i
        """

        ht_pi_t = torch.tensor(self.ht_pi, dtype=torch.float)
        ht_pij_t = torch.tensor(self.ht_pij, dtype=torch.float)

        device = client_grads.device
        dtype = client_grads.dtype

        # ---- pi_i, pi_j ----
        pi_i = ht_pi_t[selected_clients].to(device=device, dtype=dtype)  # [m]
        pi_j = ht_pi_t[add_client].to(device=device, dtype=dtype)  # scalar

        pi_i = torch.clamp(pi_i, min=self.eps)
        pi_j = max(pi_j, self.eps)

        # ---- pi_{i,j} ----
        pij = ht_pij_t[selected_clients, add_client]  # [m]
        pij = pij.to(device=device, dtype=dtype)
        pij = torch.clamp(pij, min=self.eps)

        # ---- weights ----
        # w_i = pi_{i,j} / (pi_i * pi_j)
        weights = pij / (pi_i * pi_j) # temp stuff

        # ---- weighted sum ----
        # grad = sum_i w_i * grad_i
        grad_j_hat = torch.sum(client_grads * weights.unsqueeze(1), dim=0)  # [d]

        return grad_j_hat

    # ===== HELPERS =====

    def estimate_ht_pi_by_monte_carlo(self, amount_sample_clients):
        N = int(self.amount_of_clients)
        counts = np.zeros(N, dtype=np.int64)

        num_workers = 100
        print(f"[HT-PI] Using {num_workers} threads for Monte-Carlo")
        start = time.time()

        def worker_fedcbs(_):
            return fedcbs_select(**args_fedcbs)

        def worker_fedavg(_):
            return fedavg_select(**args_fedavg)

        def worker_const_weights(_):
            return const_weight_select(**args_const_weights)

        # Choose worker
        if self.cfg.client_selector._target_ == "client_selectors.fedcbs.FedCBS":
            args_fedcbs = dict(
                num_clients_subset=amount_sample_clients,
                qcid_mtr=self.server.qcid_mtr,
                selection_counter=self.server.selection_counter.copy(),
                client_data_count=self.server.client_data_count.copy(),
                amount_classes=self.server.amount_classes,
                cur_round=self.server.cur_round,
                lambda_=self.server.lambda_,
            )
            worker = worker_fedcbs
        elif (
            self.cfg.client_selector._target_
            == "client_selectors.uniform.UniformSelector"
        ):
            args_fedavg = dict(
                num_clients_subset=amount_sample_clients,
                amount_of_clients=self.amount_of_clients,
            )
            worker = worker_fedavg
        elif (
            self.cfg.client_selector._target_
            == "client_selectors.const_weight.ConstWeightSelector"
        ):
            args_const_weights = dict(
                num_clients_subset=amount_sample_clients,
                amount_of_clients=self.amount_of_clients,
                weights=load_probabilities_from_json(
                    self.cfg.client_selector.weights_path
                ),
            )
            worker = worker_const_weights
        else:
            raise KeyError("No such client selection algorithm")

        # Thread pool
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(worker, None) for _ in range(self.ht_mk_k)]

            for idx, f in enumerate(as_completed(futures)):
                chosen = f.result()  # list[int]

                # always convert to list
                if isinstance(chosen, int):
                    chosen = [chosen]

                np.add.at(counts, chosen, 1)

                if idx % 200 == 0:
                    print(f"[HT-PI] Sampling {idx}/{self.ht_mk_k}", flush=True)

        pi_hat = counts.astype(float) / float(self.ht_mk_k)
        end = time.time() - start
        print(f"Counting took {end} time")
        return pi_hat

    def estimate_ht_pi_and_pij_by_monte_carlo(self, amount_sample_clients):
        """
        Monte-Carlo estimation of:
            pi_i  = P(i in S)
            pi_ij = P(i,j in S)

        Returns:
            pi_hat:  shape (N,)
            pij_hat: shape (N, N)
        """
        N = int(self.amount_of_clients)

        # Counters
        counts_i = np.zeros(N, dtype=np.int64)
        counts_ij = np.zeros((N, N), dtype=np.int64)

        num_workers = 100
        print(f"[HT-PI] Using {num_workers} threads for Monte-Carlo")

        # ---------- prepare selector args ----------

        def worker_fedcbs(_):
            return fedcbs_select(**args_fedcbs)

        def worker_fedavg(_):
            return fedavg_select(**args_fedavg)

        def worker_const_weights(_):
            return const_weight_select(**args_const_weights)

        if self.cfg.client_selector._target_ == "client_selectors.fedcbs.FedCBS":
            args_fedcbs = dict(
                num_clients_subset=amount_sample_clients,
                qcid_mtr=self.server.qcid_mtr,
                selection_counter=self.server.selection_counter.copy(),
                client_data_count=self.server.client_data_count.copy(),
                amount_classes=self.server.amount_classes,
                cur_round=self.server.cur_round,
                lambda_=self.server.lambda_,
            )
            worker = worker_fedcbs
        elif (
            self.cfg.client_selector._target_
            == "client_selectors.uniform.UniformSelector"
        ):
            args_fedavg = dict(
                num_clients_subset=amount_sample_clients,
                amount_of_clients=self.amount_of_clients,
            )
            worker = worker_fedavg
        elif (
            self.cfg.client_selector._target_
            == "client_selectors.const_weight.ConstWeightSelector"
        ):
            args_const_weights = dict(
                num_clients_subset=amount_sample_clients,
                amount_of_clients=self.amount_of_clients,
                weights=load_probabilities_from_json(
                    self.cfg.client_selector.weights_path
                ),
            )
            worker = worker_const_weights
        else:
            raise KeyError("No such client selection algorithm")

        # ---------- Monte-Carlo loop ----------
        start = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(worker, None) for _ in range(self.ht_mk_k)]

            for k, f in enumerate(as_completed(futures)):
                chosen = f.result()

                if isinstance(chosen, int):
                    chosen = [chosen]

                chosen = np.asarray(chosen, dtype=np.int64)

                # pi_i
                np.add.at(counts_i, chosen, 1)

                # pi_ij: all unordered pairs inside S
                # this includes (i,i), which is fine: pi_ii = pi_i
                counts_ij[np.ix_(chosen, chosen)] += 1

                if k % 200 == 0:
                    print(f"[HT-PI] Sampling {k}/{self.ht_mk_k}", flush=True)

        # ---------- normalize ----------
        pi_hat = counts_i.astype(float) / float(self.ht_mk_k)
        pij_hat = counts_ij.astype(float) / float(self.ht_mk_k)

        elapsed = time.time() - start
        print(f"[HT-PI] Estimation finished in {elapsed:.2f} sec")

        return pi_hat, pij_hat

    def compute_ht_mean(self, client_grads, pi):
        device = client_grads.device
        dtype = client_grads.dtype
        idx = np.asarray(self.list_clients, dtype=np.int64)
        pi_sel = pi[idx]  # shape (m,)

        pi_t = torch.from_numpy(pi_sel).to(device=device, dtype=dtype)  # shape (m,)
        weights = 1.0 / pi_t  # shape (m,)
        # expand to (m, D) for division
        weights = weights.view(-1, 1)  # (m, 1)

        # Weighted sum
        weighted_sum = (client_grads * weights).sum(dim=0)  # (D,)
        N = float(self.amount_of_clients)
        ht_mean = weighted_sum / N  # (D,)
        return ht_mean

    def _is_bn_key(self, key: str) -> bool:
        low = key.lower()
        return any(kw in low for kw in self.ignore_bn_keywords)

    def _collect_client_vectors(self):
        vectors = []
        for rank in self.list_clients:
            grad_dict: OrderedDict = self.server.client_gradients[rank]
            parts = []
            for k, v in grad_dict.items():
                if self._is_bn_key(k):
                    continue
                if not isinstance(v, torch.Tensor):
                    continue
                parts.append(v.detach().cpu().reshape(-1))
            if parts:
                vec = torch.cat(parts, dim=0)
                vectors.append(vec)
        return vectors

    def get_variance_clients_stat(self):
        saved_list_clients = self.list_clients

        args_fedcbs = dict(
            num_clients_subset=len(self.list_clients),
            qcid_mtr=self.server.qcid_mtr,
            selection_counter=self.server.selection_counter.copy(),
            client_data_count=self.server.client_data_count.copy(),
            amount_classes=self.server.amount_classes,
            cur_round=self.server.cur_round,
            lambda_=self.server.lambda_,
        )

        var_cl_dict = {}
        norm_cl_dict = {}

        step = 5
        attempts = 3

        print("Starting plotting graphs")
        for cl_amount in range(1, 100, step):
            lhs_vals = []
            rhs_vals = []

            for attempt in range(attempts):
                print(f"Amount clients: {cl_amount} | attempt {attempt}:")
                args_fedcbs["num_clients_subset"] = cl_amount

                self.list_clients = fedcbs_select(**args_fedcbs)
                self.horvitz_tompson_test()

                lhs_vals.append(self.ht_lhs)
                rhs_vals.append(self.ht_rhs)

            var_cl_dict[cl_amount] = sum(lhs_vals) / len(lhs_vals)
            norm_cl_dict[cl_amount] = sum(rhs_vals) / len(rhs_vals)

        print("Final dicts:\n")
        print(var_cl_dict)
        print()
        print(norm_cl_dict)
        print()

        # --- plotting ---
        xs = sorted(var_cl_dict.keys())
        lhs_ys = [var_cl_dict[x] for x in xs]
        rhs_ys = [norm_cl_dict[x] for x in xs]

        plt.figure(figsize=(8, 6))
        plt.plot(xs, lhs_ys, color="red", marker="o", label="LHS (variance)")
        plt.plot(xs, rhs_ys, color="green", marker="x", label="RHS (norm bound)")

        plt.xlabel("Number of clients |S|")
        plt.ylabel("Value")
        plt.title(f"HT norm test statistics vs |S| (round {self.cur_round})")
        plt.legend()
        plt.grid(True)

        out_path = (
            f"/home/dorofeev/federated_research/outputs/"
            f"adabatch_ht_estim/variance_plots/"
            f"{self.cur_round}_variance.png"
        )
        plt.savefig(out_path)
        plt.close()

        # restore
        self.list_clients = saved_list_clients
        print("End plotting graphs")


def fedavg_select(num_clients_subset, amount_of_clients):
    return rand.sample(list(range(amount_of_clients)), num_clients_subset)


# def fedcbs_select(
#     num_clients_subset,
#     qcid_mtr,
#     selection_counter,
#     client_data_count,
#     amount_classes,
#     cur_round,
#     lambda_,
# ):
#     N = qcid_mtr.shape[0]

#     # Local copy of selection_counter to avoid mutating global state
#     sel_counter = np.array(selection_counter, copy=True)

#     # Normalize client_data_count -> provide a helper to get counts for a list
#     is_dict_counts = isinstance(client_data_count, dict)

#     def total_count_for(clients):
#         if len(clients) == 0:
#             return 0
#         if is_dict_counts:
#             s = 0
#             for c in clients:
#                 s += client_data_count.get(c, 0)
#             return s
#         else:
#             # assume numpy-like
#             arr = np.asarray(client_data_count)
#             return int(arr[clients].sum())

#     # helper to compute qcid for a set of clients (safe for empty)
#     def qcid_fn(clients):
#         if len(clients) == 0:
#             return 1e-6
#         # sum of qcid_mtr over the clients x clients submatrix
#         sub = qcid_mtr[np.ix_(clients, clients)]
#         total = total_count_for(clients)
#         if total <= 0:
#             # avoid division by zero; return tiny positive value
#             return 1e-6
#         return float(np.sum(sub) / (total * total) - (1.0 / amount_classes))

#     # If selecting all clients, return trivial list
#     if num_clients_subset >= N:
#         return list(range(N))

#     selected = []
#     remaining = list(range(N))
#     betas = [m + 1 for m in range(num_clients_subset)]

#     for m in range(num_clients_subset):
#         # compute probs over remaining
#         if m == 0:
#             # first pick: function of singletons
#             probs = np.empty(len(remaining), dtype=float)
#             for idx, c in enumerate(remaining):
#                 # use sel_counter[c] but ensure nonzero
#                 sc = max(1.0, float(sel_counter[c]))
#                 val = 1.0 / (qcid_fn([c]) ** betas[0])
#                 val += lambda_ * np.sqrt(
#                     3.0 * np.log(max(1, cur_round) + 1.0) / (2.0 * sc)
#                 )
#                 probs[idx] = val

#         elif m == 1:
#             # second pick
#             # precompute qcid(selected) and term1
#             qcid_sel = qcid_fn(selected)
#             denom_term = (1.0 / (qcid_sel ** betas[0])) if qcid_sel != 0.0 else 1e-6

#             probs = np.empty(len(remaining), dtype=float)
#             for idx, c in enumerate(remaining):
#                 qcid_with_c = qcid_fn(selected + [c])
#                 numerator = 1.0 / (
#                     (qcid_with_c ** betas[1]) if qcid_with_c != 0.0 else 1e-6
#                 )
#                 denom = denom_term + lambda_ * np.sqrt(
#                     3.0
#                     * np.log(max(1, cur_round) + 1.0)
#                     / (2.0 * max(1.0, float(sel_counter[c])))
#                 )
#                 probs[idx] = numerator / denom

#         else:
#             # m >= 2
#             qcid_sel = qcid_fn(selected)
#             probs = np.empty(len(remaining), dtype=float)
#             for idx, c in enumerate(remaining):
#                 qcid_with_c = qcid_fn(selected + [c])
#                 # avoid division by zero
#                 qcid_with_c_safe = qcid_with_c if (qcid_with_c != 0.0) else 1e-6
#                 denom_safe = qcid_with_c_safe
#                 # original code had two similar formulas; using the stable final form:
#                 probs[idx] = ((qcid_sel / denom_safe) ** betas[m - 2]) / denom_safe

#         # sanitize probabilities
#         probs = np.where(np.isnan(probs) | np.isinf(probs), 1e-12, probs)
#         probs = np.maximum(probs, 1e-12)
#         probs_sum = probs.sum()
#         if probs_sum <= 0 or not np.isfinite(probs_sum):
#             probs = np.ones_like(probs) / float(len(probs))
#         else:
#             probs = probs / probs_sum

#         # sample one client from remaining using these probs
#         chosen = np.random.choice(remaining, p=probs).item()

#         selected.append(chosen)
#         remaining.remove(chosen)
#         sel_counter[chosen] += 1.0


#     return selected

def fedcbs_select(
    num_clients_subset,
    qcid_mtr,
    selection_counter,
    client_data_count,
    amount_classes,
    cur_round,
    lambda_,
):
    """
    Semantically identical to fedcbs_select, but optimized:
    - incremental QCID
    - vectorized candidate evaluation
    - no submatrix extraction
    """

    N = qcid_mtr.shape[0]

    # trivial case
    if num_clients_subset >= N:
        return list(range(N))

    # ---- prepare arrays ----
    qcid_mtr = np.asarray(qcid_mtr)
    qcid_diag = np.diag(qcid_mtr)

    client_counts = (
        np.asarray(client_data_count)
        if not isinstance(client_data_count, dict)
        else np.array([client_data_count.get(i, 0) for i in range(N)], dtype=float)
    )

    sel_counter = np.asarray(selection_counter, dtype=float)

    selected = []
    remaining = np.arange(N)

    betas = np.arange(1, num_clients_subset + 1)

    # ---- running statistics for QCID(S) ----
    sum_qcid_SS = 0.0  # sum_{i,j in S} q_ij
    sum_counts_S = 0.0  # sum_{i in S} n_i

    eps = 1e-12
    inv_classes = 1.0 / amount_classes

    for m in range(num_clients_subset):

        # ======================================================
        # m == 0 : singleton probabilities
        # ======================================================
        if m == 0:
            denom = np.maximum(client_counts[remaining] ** 2, eps)
            qcid_single = qcid_diag[remaining] / denom - inv_classes
            qcid_single = np.maximum(qcid_single, eps)

            explore = lambda_ * np.sqrt(
                3.0
                * np.log(max(1, cur_round) + 1.0)
                / (2.0 * np.maximum(sel_counter[remaining], 1.0))
            )

            probs = 1.0 / (qcid_single ** betas[0]) + explore

        # ======================================================
        # m >= 1 : incremental QCID(S ∪ {c})
        # ======================================================
        else:
            # sum_{i in S} q_{i,c}  for all c in remaining
            cross = qcid_mtr[np.ix_(remaining, selected)].sum(axis=1)

            sum_qcid_new = sum_qcid_SS + 2.0 * cross + qcid_diag[remaining]
            sum_counts_new = sum_counts_S + client_counts[remaining]

            denom = np.maximum(sum_counts_new**2, eps)
            qcid_with_c = sum_qcid_new / denom - inv_classes
            qcid_with_c = np.maximum(qcid_with_c, eps)

            if m == 1:
                # qcid(S) for denominator
                qcid_S = (
                    sum_qcid_SS / max(sum_counts_S**2, eps) - inv_classes
                    if sum_counts_S > 0
                    else eps
                )
                qcid_S = max(qcid_S, eps)

                denom_term = 1.0 / (qcid_S ** betas[0])

                explore = lambda_ * np.sqrt(
                    3.0
                    * np.log(max(1, cur_round) + 1.0)
                    / (2.0 * np.maximum(sel_counter[remaining], 1.0))
                )

                probs = (1.0 / (qcid_with_c ** betas[1])) / (denom_term + explore)

            else:
                qcid_S = (
                    sum_qcid_SS / max(sum_counts_S**2, eps) - inv_classes
                    if sum_counts_S > 0
                    else eps
                )
                qcid_S = max(qcid_S, eps)

                probs = ((qcid_S / qcid_with_c) ** betas[m - 2]) / qcid_with_c

        # ---- sanitize probabilities ----
        probs = np.nan_to_num(probs, nan=eps, posinf=eps, neginf=eps)
        probs = np.maximum(probs, eps)
        probs /= probs.sum()

        # ---- sample ----
        idx = np.random.choice(len(remaining), p=probs)
        chosen = remaining[idx]

        # ---- update running state ----
        if selected:
            cross_chosen = qcid_mtr[chosen, selected].sum()
        else:
            cross_chosen = 0.0

        sum_qcid_SS += 2.0 * cross_chosen + qcid_diag[chosen]
        sum_counts_S += client_counts[chosen]

        sel_counter[chosen] += 1.0

        selected.append(int(chosen))
        remaining = np.delete(remaining, idx)

    return selected


def const_weight_select(num_clients_subset, amount_of_clients, weights):
    selected_clients = np.random.choice(
        np.arange(amount_of_clients),
        size=num_clients_subset,
        p=weights,
        replace=False,
    )
    return selected_clients
