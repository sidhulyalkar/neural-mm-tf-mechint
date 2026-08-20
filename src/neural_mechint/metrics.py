from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


def logit_difference(
    logits: torch.Tensor,
    positive_token_ids: int | torch.Tensor,
    negative_token_ids: int | torch.Tensor,
) -> torch.Tensor:
    """Return positive-minus-negative logits for the final sequence position.

    Args:
        logits: Tensor shaped ``[..., sequence, vocabulary]`` or ``[..., vocabulary]``.
        positive_token_ids: Preferred token id(s).
        negative_token_ids: Contrast token id(s).

    If token id tensors are supplied, their logits are averaged before subtraction.
    """

    if logits.ndim < 2:
        raise ValueError("logits must have at least batch/vocabulary dimensions")
    final_logits = logits[..., -1, :] if logits.ndim >= 3 else logits

    def gather(ids: int | torch.Tensor) -> torch.Tensor:
        ids_t = torch.as_tensor(ids, device=final_logits.device, dtype=torch.long)
        if ids_t.ndim == 0:
            return final_logits[..., ids_t]
        return final_logits.index_select(-1, ids_t.flatten()).mean(dim=-1)

    return gather(positive_token_ids) - gather(negative_token_ids)


def normalized_recovery(
    clean_score: float | torch.Tensor,
    corrupted_score: float | torch.Tensor,
    intervention_score: float | torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Fraction of the clean-vs-corrupted behavior recovered by an intervention.

    ``0`` means no recovery beyond the corrupted run and ``1`` means full recovery.
    Values outside [0, 1] are retained because overshoot and anti-recovery are informative.
    """

    clean = torch.as_tensor(clean_score, dtype=torch.float64)
    corrupted = torch.as_tensor(corrupted_score, dtype=torch.float64)
    intervention = torch.as_tensor(intervention_score, dtype=torch.float64)
    denom = clean - corrupted
    safe = torch.where(denom.abs() < eps, torch.full_like(denom, math.nan), denom)
    return (intervention - corrupted) / safe


def symmetric_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between categorical distributions parameterized by logits."""

    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    kl_pq = F.kl_div(log_q, p, reduction="none").sum(dim=-1)
    kl_qp = F.kl_div(log_p, q, reduction="none").sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    low: float
    high: float
    n: int


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile bootstrap CI for a sample mean."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_bootstrap, x.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return BootstrapInterval(float(x.mean()), float(low), float(high), int(x.size))
