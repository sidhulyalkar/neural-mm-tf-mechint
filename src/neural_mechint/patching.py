from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .metrics import normalized_recovery


@dataclass(frozen=True)
class PatchResult:
    layer: int
    clean_score: float
    corrupted_score: float
    patched_score: float
    recovery: float


def activation_patch_scan(
    adapter,
    *,
    clean_prompt: str,
    corrupted_prompt: str,
    score_fn: Callable[[torch.Tensor], torch.Tensor | float],
    position: int = -1,
) -> list[PatchResult]:
    """Patch the clean residual state into a corrupted run one layer at a time.

    This is a causal localization experiment. Large positive recovery indicates that the
    clean activation at that location contains information sufficient to restore behavior
    when transplanted into the corrupted computation.
    """

    clean = adapter.forward_with_cache(clean_prompt)
    corrupted = adapter.forward_with_cache(corrupted_prompt, layers=[])
    clean_score = float(torch.as_tensor(score_fn(clean.logits)).mean())
    corrupted_score = float(torch.as_tensor(score_fn(corrupted.logits)).mean())

    results: list[PatchResult] = []
    for layer in clean.cache.layers():
        patched_logits = adapter.forward_with_patch(
            corrupted_prompt,
            layer=layer,
            source_activation=clean.cache[layer],
            position=position,
        )
        patched_score = float(torch.as_tensor(score_fn(patched_logits)).mean())
        recovery = float(normalized_recovery(clean_score, corrupted_score, patched_score))
        results.append(PatchResult(layer, clean_score, corrupted_score, patched_score, recovery))
    return results


def attribution_patch_score(
    clean_activation: torch.Tensor,
    corrupted_activation: torch.Tensor,
    corrupted_gradient: torch.Tensor,
    reduce_dims: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """First-order attribution-patching approximation ``(clean-corrupt) * grad``.

    Attribution patching is fast candidate generation, not final causal evidence. Confirm
    high-scoring locations with actual activation patching or ablation.
    """

    if clean_activation.shape != corrupted_activation.shape or clean_activation.shape != corrupted_gradient.shape:
        raise ValueError("clean, corrupted, and gradient tensors must have identical shapes")
    score = (clean_activation - corrupted_activation) * corrupted_gradient
    if reduce_dims is None:
        reduce_dims = tuple(range(1, score.ndim))
    return score.sum(dim=reduce_dims)
