import time
import os
import torch
import numpy as np
import pandas as pd
import random as rand
from types import MethodType
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..fedavg.fedavg import FedAvg
from .baseline_methods import configure_aggregation_methods
from .cs_functions import *


class DynamicBS(FedAvg):
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
        ht_mk_k,
        dynamic_inclusion,
        grad_history_mix,
        round_to_grad_forgetting,
        prop_limit_expand,
        selection_depth,
        num_workers,
        tau,
        momentum_beta,
        times_in_a_row,
        sample_step,
        adaptive_window,
        load_precomputed_cache=False,
        eps=1e-5,
        debug=False,
        min_number_clients=2,
        aggregation_method="fedavg",
        **aggregation_kwargs,
    ):
        super().__init__()
        self.theta_norm = theta_norm
        self.ht_mk_k = ht_mk_k
        # dynamic_inclusion modes:
        #   0  -> static cache (no periodic refresh)
        #   >0 -> refresh cache every N rounds (starting from round 1)
        #   -1 -> fully dynamic (no cache usage)
        if isinstance(dynamic_inclusion, bool):
            dynamic_inclusion = -1 if dynamic_inclusion else 0
        self.dynamic_inclusion = int(dynamic_inclusion)
        self.eps = eps
        self.grad_history_mix = grad_history_mix
        self.round_to_grad_forgetting = round_to_grad_forgetting
        self.prop_limit_expand = prop_limit_expand
        self.selection_depth = selection_depth
        self.num_workers = num_workers
        self.tau = tau
        self.momentum_beta = momentum_beta
        self.times_in_a_row = times_in_a_row
        self.sample_step = sample_step
        self.adaptive_window = adaptive_window
        self.load_precomputed_cache = load_precomputed_cache
        self.debug = debug
        self.min_number_clients = min_number_clients
        self.aggregation_method = str(aggregation_method).lower()
        self.client_grad_history = {}
        # Inclusion caches keyed by subset size k.
        self.ht_pi_cache_by_k = {}
        self.ht_pij_cache_by_k = {}
        # Backward-compatible aliases (now dict instead of list).
        self.ht_pi_i_s = self.ht_pi_cache_by_k
        self.ht_pi_ij_s = self.ht_pij_cache_by_k
        self._cache_loaded = False
        self._cache_last_refresh_round = None
        self._base_init_federated = MethodType(FedAvg._init_federated, self)
        configure_aggregation_methods(self, **aggregation_kwargs)
        self._last_defined_clients = None

        if self.adaptive_window < 1:
            raise ValueError(
                f"adaptive_window must be >= 1, got {self.adaptive_window}"
            )
        if self.sample_step < 1:
            raise ValueError(f"sample_step must be >= 1, got {self.sample_step}")
        if self.times_in_a_row < 0:
            raise ValueError(f"times_in_a_row must be >= 0, got {self.times_in_a_row}")

    def _init_federated(self, cfg):
        if self.aggregation_method in {"fedprox", "fednova"}:
            selector = cfg.client_selector._target_
            if selector != "client_selectors.uniform.UniformSelector":
                raise ValueError(
                    "DynamicBS with aggregation_method='fedprox' or 'fednova' "
                    "requires client_selector='uniform'. "
                    f"Got: {selector}"
                )
        return self._base_init_federated(cfg)

    def aggregate(self):
        if getattr(self, "warmup", 0) <= self.cur_round:
            self.prev_num_clients_subset = int(self.num_clients_subset)
            self.num_clients_subset = self.define_amount_clients()
        aggr_weights = super().aggregate()

        # Clear memory after aggregate
        self.server.client_gradients = [
            OrderedDict() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
        self.server.server_metrics = [
            pd.DataFrame() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
        return aggr_weights

    def train_round(self):
        if self.cfg.client_selector._target_ == "client_selectors.pow.Pow":
            self.pow_clients_losses_snapshot = list(self.server.clients_losses)
        super().train_round()

    def define_amount_clients(self):
        print("\nStart computing our test\n")

        if (self.cur_round % self.adaptive_window) != 0:
            base = (
                self._last_defined_clients
                if self._last_defined_clients is not None
                else int(self.num_clients_subset)
            )
            return self._clamp_n_clients(base)

        # ---- initial data ----
        N = self.amount_of_clients
        S = list(self.list_clients)  # list[int]
        grads = torch.stack(self._collect_client_vectors(), dim=0)  # [m,d]

        # Update gradient history
        self._update_client_grad_history(S, grads)

        # ---- initial test ----
        lhs, rhs, ht_pi_t, ht_pij_t = self.compute_norm_test(S, grads)
        print(
            f"[our norm-test | |S|={len(S)}]: {lhs:.6e} {'<' if lhs < rhs else '>'} {rhs:.6e}"
        )

        # ---------- CASE 1: test PASSES → try shrinking ----------
        if lhs < rhs:
            n_new = self.shrink_clients(S, grads)
            n_new = self._clamp_n_clients(n_new)
            self._last_defined_clients = int(n_new)
            return n_new

        # ---------- CASE 2: test FAILS → try expanding ----------
        else:
            n_new = self.expand_clients(S, grads)
            n_new = self._clamp_n_clients(n_new)
            self._last_defined_clients = int(n_new)
            return n_new

    def _clamp_n_clients(self, n_new):
        step = max(0, int(self.sample_step) - 1)
        if step != 0:
            low = n_new - step
            high = n_new + step
            n_new = int(np.random.randint(low, high + 1))
        n_new = min(n_new, self.amount_of_clients)
        n_new = max(self.min_number_clients, n_new)
        if getattr(self, "momentum_beta", 0) > 0:
            prev_amount = getattr(self, "prev_num_clients_subset", n_new)
            print(f"Apply momentum: prev_amount={prev_amount}, n_new={n_new}")
            n_new = int(
                round(
                    self.momentum_beta * prev_amount + (1 - self.momentum_beta) * n_new
                )
            )
            print(f"After momentum: n_new={n_new}")
            n_new = min(n_new, self.amount_of_clients)
            n_new = max(self.min_number_clients, n_new)
        return n_new

    def compute_norm_test(self, S, grads):
        ht_pi, ht_pij = self.estimate_ht_pi_and_pij_by_monte_carlo(len(S))
        ht_pi_t = torch.tensor(ht_pi, dtype=torch.float)
        ht_pij_t = torch.tensor(ht_pij, dtype=torch.float)
        self.ht_pi_t = ht_pi_t
        self.ht_pij_t = ht_pij_t

        if self.debug:
            print(f"HT pi_i:\n{self.ht_pi_t}\n")
            # self.pretty_print_trial_scores(self.ht_pij_t, decimals=3, title="HT pi_ij")

        selected = torch.tensor(S, device=grads.device)
        if self.debug:
            self._debug_ht_rhs_norms(
                client_grads=grads,
                selected_clients=selected,
                label="[HT RHS] current subset",
                ht_pi_t=ht_pi_t,
            )
        rhs = self.get_ht_rhs(grads, selected, ht_pi_t)
        lhs = self.get_ht_var(grads, selected, ht_pi_t, ht_pij_t)
        return lhs, rhs, ht_pi_t, ht_pij_t

    def shrink_clients(self, S, grads):
        S_cur = S.copy()
        grads_cur = grads.clone()
        ht_pi_cur = self.ht_pi_t

        while len(S_cur) > 2:
            trials = []
            consecutive_success = 0
            early_continue = False
            for _ in range(self.selection_depth):
                step = max(1, int(self.sample_step))
                max_removals = max(1, len(S_cur) - 2)
                step = min(step, max_removals)
                remove_idx = np.random.choice(len(S_cur), size=step, replace=False)
                remove_idx = sorted(remove_idx)
                removed_clients = [S_cur[i] for i in remove_idx]
                keep_mask = np.ones(len(S_cur), dtype=bool)
                keep_mask[remove_idx] = False
                S_next = [s for i, s in enumerate(S_cur) if keep_mask[i]]
                grads_next = grads_cur[keep_mask]
                new_lhs, new_rhs, ht_pi_next, ht_pij_next = self.compute_norm_test(
                    S_next, grads_next
                )
                success = new_lhs < new_rhs
                trials.append(
                    (
                        success,
                        new_lhs,
                        new_rhs,
                        removed_clients,
                        S_next,
                        grads_next,
                        ht_pi_next,
                        ht_pij_next,
                    )
                )
                print(
                    f"[shrink | |S|={len(S_next)} | removed={removed_clients}]: "
                    f"{new_lhs:.6e} {'<' if new_lhs < new_rhs else '>'} {new_rhs:.6e}"
                )
                if success:
                    consecutive_success += 1
                else:
                    consecutive_success = 0
                if (
                    self.times_in_a_row > 0
                    and consecutive_success >= self.times_in_a_row
                ):
                    print(
                        f"[shrink] Early continue after {consecutive_success} successes in a row."
                    )
                    early_continue = True
                    break

            success_count = sum(1 for t in trials if t[0])
            if not early_continue and success_count <= self.selection_depth // 2:
                return len(S_cur)

            successful = [t for t in trials if t[0]]
            best = min(successful, key=lambda t: t[1])
            _, _, _, _, S_cur, grads_cur, ht_pi_cur, _ = best

        return len(S_cur)

    def expand_clients(self, S, grads):
        N = self.amount_of_clients
        valid_ids = set(getattr(self, "ht_pi_valid_ids", []))
        remaining = [
            i
            for i in range(N)
            if i not in S
            and i in self.client_grad_history
            and (not valid_ids or i in valid_ids)
        ]
        if not remaining:
            print(
                "[expand] No remaining clients with history and non-zero pi_i; skip expanding."
            )

        S_cur = S.copy()
        grads_cur = grads.clone()
        ht_pi_cur = self.ht_pi_t

        while remaining:
            success_count = 0
            trials_run = 0
            consecutive_failures = 0
            early_continue = False
            # We set ht_pi inside the loop
            best_ht_pi = ht_pi_cur
            best_S_next = S_cur
            best_grads_next = grads_cur
            best_delta = torch.inf
            for _ in range(min(self.selection_depth, len(remaining))):
                step = max(1, int(self.sample_step))
                add_count = min(step, len(remaining))
                add_idx = np.random.choice(len(remaining), size=add_count, replace=False)
                add_clients = [remaining[i] for i in add_idx]

                # Temporarily disable deterministic add-client selection:
                # if self.cfg.client_selector._target_ == "client_selectors.fedcor.FedCor":
                #     add_client = self._select_add_client_fedcor(S_cur)
                # elif self.cfg.client_selector._target_ == "client_selectors.pow.Pow":
                #     add_client = self._select_add_client_pow(S_cur)

                selected_clients = torch.tensor(S_cur, device=grads_cur.device)
                add_grads = []
                valid_add_clients = []
                for add_client in add_clients:
                    add_grad = self.approx_grad_for_add_client(
                        client_grads=grads_cur,
                        selected_clients=selected_clients,
                        add_client=add_client,
                    )
                    if add_grad is None:
                        if self.debug:
                            print(
                                f"[expand] Skip client {add_client}: no grad history."
                            )
                        remaining.remove(add_client)
                        continue
                    add_grads.append(add_grad)
                    valid_add_clients.append(add_client)

                if not valid_add_clients:
                    continue

                if self.debug:
                    for add_client, add_grad in zip(valid_add_clients, add_grads):
                        self._debug_ht_rhs_add_client_norms(
                            add_client=add_client,
                            add_grad=add_grad,
                            ht_pi_t=ht_pi_cur,
                        )

                add_grads_t = torch.stack(add_grads, dim=0)
                S_next = S_cur + valid_add_clients
                grads_next = torch.cat([grads_cur, add_grads_t], dim=0)
                new_lhs, new_rhs, ht_pi_next, ht_pij_next = self.compute_norm_test(
                    S_next, grads_next
                )
                print(
                    f"[expand | |S|={len(S_next)} | add={valid_add_clients}]: "
                    f"{new_lhs:.6e} {'<' if new_lhs < new_rhs else '>'} {new_rhs:.6e}"
                )
                trials_run += 1
                success = new_lhs < new_rhs
                success_count += int(success)
                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                if (
                    self.times_in_a_row > 0
                    and consecutive_failures >= self.times_in_a_row
                ):
                    print(
                        f"[expand] Early continue after {consecutive_failures} failures in a row."
                    )
                    early_continue = True
                    break
                if best_delta > new_lhs - new_rhs:
                    best_delta = new_lhs - new_rhs
                    best_ht_pi = ht_pi_next
                    best_S_next = S_next
                    best_grads_next = grads_next

            if trials_run == 0:
                print("[expand] No valid trials (all candidates lacked history).")
                break

            if not early_continue and success_count > trials_run // 2:
                print("Majority successful, Take the previous |S|")
                return len(S_cur)

            if len(best_S_next) >= self.prop_limit_expand * len(S):
                print("Reached expansion limit.")
                return len(best_S_next)

            valid_ids = np.where(best_ht_pi > self.eps)[0].tolist()
            if not valid_ids:
                print("[expand] No non-zero pi_i after HT estimation.")
            S_cur = best_S_next
            grads_cur = best_grads_next
            ht_pi_cur = best_ht_pi
            remaining = [
                i
                for i in range(N)
                if i not in S_cur
                and i in self.client_grad_history
                and (not valid_ids or i in valid_ids)
            ]
        return len(S_cur)

    def get_ht_var(self, client_grads, selected_clients, ht_pi_t, ht_pij_t):
        """
        Var(mu_HT) =
            1/N^2 * sum_{i,j in S}
            (g_i^T g_j) / (pi_i pi_j)
            * (pi_ij - pi_i pi_j) / pi_ij
        """
        N = self.amount_of_clients

        # ---- pi_i and pi_ij ----
        pi_sel = ht_pi_t[selected_clients]  # [m]
        pij_sel = ht_pij_t[
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

    def get_ht_rhs(self, client_grads, selected_clients, ht_pi_t):
        pi_sel = ht_pi_t[selected_clients]  # [m]
        grads_div = client_grads / pi_sel.unsqueeze(1)
        mu_ht = grads_div.sum(dim=0) / float(self.amount_of_clients)  # hat mu_HT
        rhs = (self.theta_norm**2) * mu_ht.pow(2).sum().item()
        return rhs

    def approx_grad_for_add_client(self, client_grads, selected_clients, add_client):
        """
        Approximate gradient for adding client j using:
            HT-weighted sum over current S
        and optionally mix with historical gradient of client j.

        grad_j_hat =
            alpha * grad_ht
            + (1 - alpha) * grad_hist_j   (if exists and not stale)
        """
        # Temporarily disabled: use historical gradient only.
        # device = client_grads.device
        # dtype = client_grads.dtype
        #
        # # ===== HT weighted approximation =====
        #
        # # pi_i, pi_j
        # pi_i = self.ht_pi_t[selected_clients].to(device=device, dtype=dtype)  # [m]
        # pi_j = self.ht_pi_t[add_client].to(device=device, dtype=dtype)  # scalar
        #
        # # pi_{i,j}
        # pij = self.ht_pij_t[selected_clients, add_client]
        # pij = pij.to(device=device, dtype=dtype)
        #
        # # weights: pi_ij / (pi_i * pi_j)
        # weights = pij / (pi_i * pi_j)
        # weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        #
        # # Approximation with weighted-sum
        # grad_weight_sum = torch.sum(client_grads * weights.unsqueeze(1), dim=0)  # [d]
        #
        # # ===== Historical gradient (if available) =====
        # grad_hist = None
        # use_history = False
        #
        # hist = self.client_grad_history.get(add_client, None)
        # if hist is not None:
        #     use_history = True
        #     round_hist, grad_hist = hist
        #     grad_hist = grad_hist.to(device=device, dtype=dtype)
        #
        # # ===== Mix =====
        #
        # if use_history:
        #     print(
        #         f"[Approx grad]: Use history for {add_client} client, rounds passed: {self.cur_round - round_hist}"
        #     )
        #     alpha = self.grad_history_mix
        #     grad_j_hat = alpha * grad_weight_sum + (1.0 - alpha) * grad_hist
        # else:
        #     grad_j_hat = grad_weight_sum
        #
        # return grad_j_hat

        hist = self.client_grad_history.get(add_client, None)
        if hist is None:
            return None
        round_hist, grad_hist = hist
        print(
            f"[Approx grad]: Use history for {add_client} client, rounds passed: {self.cur_round - round_hist}"
        )
        return grad_hist.to(device=client_grads.device, dtype=client_grads.dtype)

    # ===== HELPERS =====

    def _cache_path(self):
        run_dir = getattr(self.cfg, "single_run_dir", None) or "."
        return os.path.join(run_dir, "dynamic_bs_ht_cache.npz")

    def _load_cache_if_needed(self):
        if self._cache_loaded or not self.load_precomputed_cache:
            return

        path = self._cache_path()
        if not os.path.exists(path):
            self._cache_loaded = True
            return

        try:
            data = np.load(path, allow_pickle=False)
            keys = data["keys"].astype(int).tolist()
            pi = data["pi"]
            pij = data["pij"]

            for idx, k in enumerate(keys):
                self.ht_pi_cache_by_k[int(k)] = pi[idx]
                self.ht_pij_cache_by_k[int(k)] = pij[idx]

            print(f"[HT-PI] Loaded cached inclusion from {path}")
        except Exception as e:
            print(f"[HT-PI] Failed to load cache from {path}: {e}")
        finally:
            self._cache_loaded = True

    def _save_cache(self):
        if not self.ht_pi_cache_by_k:
            return

        path = self._cache_path()
        keys = sorted(self.ht_pi_cache_by_k.keys())
        pi = np.stack([self.ht_pi_cache_by_k[k] for k in keys], axis=0)
        pij = np.stack([self.ht_pij_cache_by_k[k] for k in keys], axis=0)
        np.savez_compressed(path, keys=np.asarray(keys, dtype=np.int64), pi=pi, pij=pij)

    def _refresh_cache_if_needed(self):
        if self.dynamic_inclusion <= 0:
            return
        if self.cur_round <= 0:
            return
        if self._cache_last_refresh_round == self.cur_round:
            return
        # Refresh cadence starts from round 1.
        if (self.cur_round - 1) % self.dynamic_inclusion == 0:
            self.ht_pi_cache_by_k.clear()
            self.ht_pij_cache_by_k.clear()
            self._cache_last_refresh_round = self.cur_round
            print(f"[HT-PI] Refresh cache at round {self.cur_round}")

    def estimate_ht_pi_and_pij_by_monte_carlo(self, amount_sample_clients):
        """
        Monte-Carlo estimation of:
            pi_i  = P(i in S)
            pi_ij = P(i,j in S)

        Returns:
            pi_hat:  shape (N,)
            pij_hat: shape (N, N)
        """
        # -1 means fully dynamic: always recompute current k without cache.
        if self.dynamic_inclusion == -1:
            pi_hat, pij_hat = self._estimate_ht_pi_and_pij_once(amount_sample_clients)
            self.ht_pi_valid_ids = np.where(pi_hat > self.eps)[0].tolist()
            return pi_hat, pij_hat

        self._load_cache_if_needed()
        self._refresh_cache_if_needed()

        if amount_sample_clients not in self.ht_pi_cache_by_k:
            print(
                f"[HT-PI] Cache miss for |S|={amount_sample_clients}, estimate and store."
            )
            pi_hat, pij_hat = self._estimate_ht_pi_and_pij_once(amount_sample_clients)
            self.ht_pi_cache_by_k[amount_sample_clients] = pi_hat
            self.ht_pij_cache_by_k[amount_sample_clients] = pij_hat
            self._save_cache()

        pi_hat = self.ht_pi_cache_by_k[amount_sample_clients]
        pij_hat = self.ht_pij_cache_by_k[amount_sample_clients]
        self.ht_pi_valid_ids = np.where(pi_hat > self.eps)[0].tolist()
        return pi_hat, pij_hat

    def _estimate_ht_pi_and_pij_once(self, amount_sample_clients):
        """Single Monte-Carlo estimate for a fixed subset size."""

        N = int(self.amount_of_clients)

        # Counters
        counts_i = np.zeros(N, dtype=np.int64)
        counts_ij = np.zeros((N, N), dtype=np.int64)

        # print(f"[HT-PI] Using {self.num_workers} threads for Monte-Carlo")

        # ---------- prepare selector args ----------

        def worker_fedcbs(_):
            return fedcbs_select(**args_fedcbs)

        def worker_fedavg(_):
            return fedavg_select(**args_fedavg)

        def worker_fedcor(_):
            return fedcor_select(**args_fedcor)

        def worker_delta(_):
            return delta_select(**args_delta)

        def worker_pow(_):
            return pow_select(**args_pow)

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
        elif self.cfg.client_selector._target_ == "client_selectors.fedcor.FedCor":
            warmup = getattr(self.server, "warmup", 0)
            gpr = getattr(self.server, "gpr", None)
            if gpr is None and self.server.cur_round > warmup:
                from client_selectors.fedcor import Kernel_GPR, Poly_Kernel

                gpr = Kernel_GPR(
                    num_users=self.amount_of_clients,
                    loss_type="MML",
                    reusable_history_length=500,
                    gamma=0.99,
                    device="cpu",
                    dimension=15,
                    kernel=Poly_Kernel,
                    order=1,
                    Normalize=0,
                )
            args_fedcor = dict(
                num_clients_subset=amount_sample_clients,
                amount_of_clients=self.amount_of_clients,
                cur_round=self.server.cur_round,
                warmup=warmup,
                gpr=gpr,
                ts=getattr(self.server, "ts", None),
                tau=self.tau,
            )
            worker = worker_fedcor
        elif self.cfg.client_selector._target_ == "client_selectors.delta.Delta":
            args_delta = dict(
                num_clients_subset=amount_sample_clients,
                amount_of_clients=self.amount_of_clients,
                client_probs=getattr(self.server, "client_probs", None),
            )
            worker = worker_delta
        elif self.cfg.client_selector._target_ == "client_selectors.pow.Pow":
            if hasattr(self, "pow_clients_losses_snapshot"):
                clients_losses = self.pow_clients_losses_snapshot
            else:
                clients_losses = getattr(self.server, "clients_losses", None)
            args_pow = dict(
                num_clients_subset=amount_sample_clients,
                amount_of_clients=self.amount_of_clients,
                candidate_set_size=getattr(self.server, "candidate_set_size", None),
                clients_probs=getattr(self.server, "clients_probs", None),
                clients_losses=clients_losses,
                topk_tau=self.tau,
            )
            worker = worker_pow
        elif (
            self.cfg.client_selector._target_
            == "client_selectors.uniform.UniformSelector"
        ):
            args_fedavg = dict(
                num_clients_subset=amount_sample_clients,
                amount_of_clients=self.amount_of_clients,
            )
            worker = worker_fedavg
        else:
            raise KeyError("No such client selection algorithm")

        # ---------- Monte-Carlo loop ----------
        start = time.time()
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
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

                # if k % 200 == 0:
                #     print(f"[HT-PI] Sampling {k}/{self.ht_mk_k}", flush=True)

        # ---------- normalize ----------
        pi_hat = counts_i.astype(float) / float(self.ht_mk_k)
        pij_hat = counts_ij.astype(float) / float(self.ht_mk_k)

        elapsed = time.time() - start
        print(f"[HT-PI] Estimation finished in {elapsed:.2f} sec")

        return pi_hat, pij_hat

    def _update_client_grad_history(self, S, grads):
        """
        Save current gradients of clients in S into history
        and remove stale entries.

        Stored format:
            self.client_grad_history[client_id] = (round_idx, grad_tensor)
        """

        cur_round = self.cur_round
        max_age = self.round_to_grad_forgetting

        # ---- save / update gradients for current S ----
        for idx, client_id in enumerate(S):
            self.client_grad_history[client_id] = (
                cur_round,
                grads[idx].detach().cpu(),  # keep on CPU to save GPU memory
            )

        # ---- remove stale gradients ----
        to_delete = []
        for client_id, (r, _) in self.client_grad_history.items():
            if (cur_round - r) > max_age:
                to_delete.append(client_id)

        for client_id in to_delete:
            del self.client_grad_history[client_id]

        if self.debug:
            print(f"\n[Method]: grad history:\n")
            for client_id, (r, _) in self.client_grad_history.items():
                print(f"Client {client_id}: {r}")
            print("\n")

    def _is_bn_key(self, key: str) -> bool:
        low = key.lower()
        return any(kw in low for kw in self.ignore_bn_keywords)

    def _update_pi_for_target_size(self, target_size):
        ht_pi, ht_pij = self.estimate_ht_pi_and_pij_by_monte_carlo(target_size)
        self.ht_pi_t = torch.tensor(ht_pi, dtype=torch.float)
        self.ht_pij_t = torch.tensor(ht_pij, dtype=torch.float)

    def _select_add_client_fedcor(self, S_cur):
        # Temporarily disabled.
        # target_size = len(S_cur) + 1
        # ht_pi_next, _ = self.estimate_ht_pi_and_pij_by_monte_carlo(target_size)
        # next_set = np.where(np.asarray(ht_pi_next) > 0.0)[0].tolist()
        # add_candidates = [c for c in next_set if c not in S_cur]
        # if add_candidates:
        #     return add_candidates[0]
        return None

    def _select_add_client_pow(self, S_cur):
        # Temporarily disabled.
        # next_set = np.where(self.ht_pi_t.detach().cpu().numpy() > 0.0)[0].tolist()
        # add_candidates = [c for c in next_set if c not in S_cur]
        # if add_candidates:
        #     return add_candidates[0]
        return None

    def _debug_ht_rhs_norms(self, client_grads, selected_clients, label, ht_pi_t):
        client_ids = (
            selected_clients.detach().cpu().tolist()
            if isinstance(selected_clients, torch.Tensor)
            else list(selected_clients)
        )
        pi_sel = ht_pi_t[selected_clients]  # [m]
        grads_div = client_grads / pi_sel.unsqueeze(1)
        mu_ht = grads_div.sum(dim=0) / float(self.amount_of_clients)

        client_norms = client_grads.pow(2).sum(dim=1).detach().cpu().tolist()
        grads_div_norms = grads_div.pow(2).sum(dim=1).detach().cpu().tolist()
        mu_ht_norm = mu_ht.pow(2).sum().item()

        print(f"{label} client_grads l2^2:")
        for cid, val in zip(client_ids, client_norms):
            print(f"  client {cid}: {val:.6e}")

        print(f"{label} grads_div l2^2:")
        for cid, val in zip(client_ids, grads_div_norms):
            print(f"  client {cid}: {val:.6e}")

        print(f"{label} mu_ht l2^2: {mu_ht_norm:.6e}")

    def _debug_ht_rhs_add_client_norms(self, add_client, add_grad, ht_pi_t):
        pi_j = ht_pi_t[add_client]
        grads_div = add_grad / pi_j
        add_norm = add_grad.pow(2).sum().item()
        div_norm = grads_div.pow(2).sum().item()

        print(f"[HT RHS] add client {add_client} client_grads l2^2: {add_norm:.6e}")
        print(f"[HT RHS] add client {add_client} grads_div l2^2: {div_norm:.6e}")

    def _ensure_square_tensor(self, matrix):
        data = torch.as_tensor(matrix).detach().cpu()
        if data.dim() != 2 or data.size(0) != data.size(1):
            raise ValueError("Expected a square 2D matrix for pretty print.")
        return data

    def pretty_print_trial_scores(self, matrix=None, decimals=3, title=None):
        if matrix is None:
            raise ValueError("pretty_print_trial_scores expects a matrix.")

        data = self._ensure_square_tensor(matrix)
        hide_diagonal = False
        title = title or "Client Matrix"

        num_clients = data.size(0)
        print(f"{title}:")
        if num_clients == 0:
            print("  <empty>")
            return

        row_labels = [f"Client {i}" for i in range(num_clients)]
        col_labels = row_labels
        row_label_width = max(len("Client"), max(len(label) for label in row_labels))
        col_width = max(len(label) for label in col_labels)

        def render_value(row_idx, col_idx, value):
            if hide_diagonal and row_idx == col_idx:
                return "-"
            if abs(value) < 1e-12:
                return "0"
            precision = (
                decimals(value, row_idx, col_idx)
                if callable(decimals)
                else int(decimals)
            )
            return f"{value:.{precision}f}"

        rendered_rows = []
        for row_idx in range(num_clients):
            rendered = []
            for col_idx in range(num_clients):
                value = data[row_idx, col_idx].item()
                display = render_value(row_idx, col_idx, value)
                col_width = max(col_width, len(display))
                rendered.append(display)
            rendered_rows.append(rendered)

        header = f"{'Client':>{row_label_width}} | " + " ".join(
            f"{label:>{col_width}}" for label in col_labels
        )
        print(header)
        for label, values in zip(row_labels, rendered_rows):
            row = " ".join(f"{value:>{col_width}}" for value in values)
            print(f"{label:>{row_label_width}} | {row}")

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

    def log_round(self):
        super().log_round()

        # Log amount of clients
        self.logger.log_scalar(
            int(self.num_clients_subset), "dynamic_bs/amount_of_clients", self.cur_round
        )
