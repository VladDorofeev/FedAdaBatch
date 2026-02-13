import time
import torch
import numpy as np
import pandas as pd
import random as rand


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    if (not np.isfinite(s)) or s <= 0:
        return np.ones_like(ex, dtype=np.float64) / float(len(ex))
    return ex / s


def fedavg_select(num_clients_subset, amount_of_clients):
    return rand.sample(list(range(amount_of_clients)), num_clients_subset)


def fedcbs_select(
    num_clients_subset,
    qcid_mtr,
    selection_counter,
    client_data_count,
    amount_classes,
    cur_round,
    lambda_,
):
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


def fedcor_select_legacy(
    num_clients_subset,
    amount_of_clients,
    cur_round,
    warmup,
    gpr,
    ts,
):
    if num_clients_subset >= amount_of_clients:
        return list(range(amount_of_clients))

    if cur_round <= warmup:
        return np.random.choice(
            amount_of_clients, num_clients_subset, replace=False
        ).tolist()

    if gpr is None:
        from client_selectors.fedcor import Kernel_GPR, Poly_Kernel

        gpr = Kernel_GPR(
            num_users=amount_of_clients,
            loss_type="MML",
            reusable_history_length=500,
            gamma=0.99,
            device="cpu",
            dimension=15,
            kernel=Poly_Kernel,
            order=1,
            Normalize=0,
        )

    return gpr.Select_Clients(
        number=num_clients_subset,
        epsilon=0,
        weights=ts,
        Dynamic=False,
        Dynamic_TH=0.0,
    )


def fedcor_select(
    num_clients_subset,
    amount_of_clients,
    cur_round,
    warmup,
    gpr,
    ts,
    *,
    tau=0.2,
    rng=None,
    cache_cov=True,
):
    """
    Stochastic FedCor selection: sequential softmax sampling over FedCor marginal gain.
    """
    if rng is None:
        rng = np.random.default_rng()

    if num_clients_subset >= amount_of_clients:
        return list(range(amount_of_clients))

    if cur_round <= warmup:
        return rng.choice(
            amount_of_clients, size=num_clients_subset, replace=False
        ).tolist()

    if gpr is None:
        from client_selectors.fedcor import Kernel_GPR, Poly_Kernel

        gpr = Kernel_GPR(
            num_users=amount_of_clients,
            loss_type="MML",
            reusable_history_length=500,
            gamma=0.99,
            device="cpu",
            dimension=15,
            kernel=Poly_Kernel,
            order=1,
            Normalize=0,
        )

    device = getattr(gpr, "device", torch.device("cpu"))

    Sigma0 = None
    if (
        cache_cov
        and hasattr(gpr, "_cov_cache_round")
        and getattr(gpr, "_cov_cache_round") == cur_round
    ):
        Sigma0 = getattr(gpr, "_cov_cache", None)
    if Sigma0 is None:
        Sigma0 = gpr.Covariance().detach()
        if cache_cov:
            gpr._cov_cache_round = cur_round
            gpr._cov_cache = Sigma0

    Sigma = Sigma0.clone()
    discount = getattr(gpr, "discount", torch.ones(amount_of_clients, device=device))
    discount = discount.detach()

    w_full = None
    if ts is not None:
        w_full = np.asarray(ts, dtype=np.float64)
        if w_full.shape[0] != amount_of_clients:
            raise ValueError(
                f"ts must have length {amount_of_clients}, got {w_full.shape[0]}"
            )

    remain = list(range(amount_of_clients))
    selected = []

    for _ in range(num_clients_subset):
        idx_t = torch.tensor(remain, device=Sigma.device, dtype=torch.long)
        Sigma_valid = Sigma.index_select(0, idx_t).index_select(1, idx_t)

        diag = torch.diagonal(Sigma_valid)
        diag = torch.clamp(diag, min=1e-12)

        Diag_valid = discount.index_select(0, idx_t) / torch.sqrt(diag)

        if w_full is None:
            total_loss_decrease = Sigma_valid.sum(dim=0) * Diag_valid
        else:
            w = torch.tensor(
                w_full[remain], device=Sigma.device, dtype=torch.float32
            ).view(-1, 1)
            total_loss_decrease = (w * Sigma_valid).sum(dim=0) * Diag_valid

        scores = total_loss_decrease.detach().cpu().numpy()

        if (tau is None) or (tau <= 0) or (not np.isfinite(tau)):
            j_local = int(np.argmax(scores))
        else:
            probs = _softmax_np(scores / float(tau))
            j_local = int(rng.choice(len(remain), p=probs))

        j = remain.pop(j_local)
        selected.append(j)

        denom = Sigma[j, j].item()
        if np.isfinite(denom) and denom > 1e-12:
            col = Sigma[:, j : j + 1]
            row = Sigma[j : j + 1, :]
            Sigma = Sigma - (col @ row) / Sigma[j, j]

    return selected


def delta_select(
    num_clients_subset,
    amount_of_clients,
    client_probs,
):
    if num_clients_subset >= amount_of_clients:
        return list(range(amount_of_clients))

    if client_probs is None:
        client_probs = np.ones(amount_of_clients, dtype=float) / float(
            amount_of_clients
        )
    else:
        client_probs = np.asarray(client_probs, dtype=float)
        client_probs = client_probs / client_probs.sum()

    return np.random.choice(
        amount_of_clients, size=num_clients_subset, replace=False, p=client_probs
    ).tolist()


def pow_select_legacy(
    num_clients_subset,
    amount_of_clients,
    candidate_set_size,
    clients_probs,
    clients_losses,
):
    if num_clients_subset >= amount_of_clients:
        return list(range(amount_of_clients))

    if candidate_set_size is None:
        candidate_set_size = num_clients_subset

    candidate_set_size = min(candidate_set_size, amount_of_clients)

    if clients_probs is None:
        clients_probs = np.ones(amount_of_clients, dtype=float) / float(
            amount_of_clients
        )
    else:
        clients_probs = np.asarray(clients_probs, dtype=float)
        clients_probs = clients_probs / clients_probs.sum()

    if clients_losses is None:
        clients_losses = np.zeros(amount_of_clients, dtype=float)
    else:
        clients_losses = np.asarray(clients_losses, dtype=float)

    candidate_clients_list = np.random.choice(
        amount_of_clients, size=candidate_set_size, replace=False, p=clients_probs
    ).tolist()

    candidate_clients_list.sort(
        key=lambda client_rank: clients_losses[client_rank], reverse=True
    )

    return candidate_clients_list[:num_clients_subset]


def pow_select(
    num_clients_subset,
    amount_of_clients,
    candidate_set_size,
    clients_probs,
    clients_losses,
    *,
    topk_tau=0.2,
    rng=None,
    soft_topk=True,
    use_prior_in_topk=False,
):
    """
    Power-of-Choice selection with stochastic top-k.
    """
    if rng is None:
        rng = np.random.default_rng()

    if num_clients_subset >= amount_of_clients:
        return list(range(amount_of_clients))

    if candidate_set_size is None:
        candidate_set_size = num_clients_subset
    candidate_set_size = int(min(candidate_set_size, amount_of_clients))

    if clients_probs is None:
        probs = np.ones(amount_of_clients, dtype=np.float64) / float(amount_of_clients)
    else:
        probs = np.asarray(clients_probs, dtype=np.float64)
        s = probs.sum()
        if s <= 0 or (not np.isfinite(s)):
            probs = np.ones(amount_of_clients, dtype=np.float64) / float(
                amount_of_clients
            )
        else:
            probs = probs / s

    if clients_losses is None:
        losses = np.zeros(amount_of_clients, dtype=np.float64)
    else:
        losses = np.asarray(clients_losses, dtype=np.float64)

    cand = rng.choice(
        amount_of_clients, size=candidate_set_size, replace=False, p=probs
    )

    if not soft_topk:
        cand_sorted = cand[np.argsort(losses[cand])[::-1]]
        k = min(num_clients_subset, len(cand_sorted))
        return cand_sorted[:k].tolist()

    if (topk_tau is None) or (topk_tau <= 0) or (not np.isfinite(topk_tau)):
        cand_sorted = cand[np.argsort(losses[cand])[::-1]]
        k = min(num_clients_subset, len(cand_sorted))
        return cand_sorted[:k].tolist()

    logits = losses[cand] / float(topk_tau)
    if use_prior_in_topk:
        logits = logits + np.log(probs[cand] + 1e-30)

    u = rng.random(size=len(cand))
    g = -np.log(-np.log(u + 1e-30) + 1e-30)
    keys = logits + g

    k = int(min(num_clients_subset, len(cand)))
    if k == 0:
        return []
    idx = np.argpartition(-keys, kth=k - 1)[:k]
    idx = idx[np.argsort(keys[idx])[::-1]]

    return cand[idx].tolist()
