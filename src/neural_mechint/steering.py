from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SteeringResult:
    coefficient: float
    score: float


def steering_sweep(
    adapter,
    *,
    prompt: str,
    layer: int,
    direction: torch.Tensor,
    coefficients: Iterable[float],
    score_fn: Callable[[torch.Tensor], torch.Tensor | float],
    position: int | slice = -1,
) -> list[SteeringResult]:
    """Measure a causal dose-response curve for an activation direction."""

    results: list[SteeringResult] = []
    for coefficient in coefficients:
        logits = adapter.forward_with_steering(
            prompt,
            layer=layer,
            direction=direction,
            coefficient=float(coefficient),
            position=position,
        )
        score = float(torch.as_tensor(score_fn(logits)).mean())
        results.append(SteeringResult(float(coefficient), score))
    return results
