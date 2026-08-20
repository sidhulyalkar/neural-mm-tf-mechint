from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircuitPoint:
    name: str
    importance: float


def rank_components(names: Iterable[str], scores: Iterable[float], absolute: bool = True) -> list[CircuitPoint]:
    """Rank candidate circuit components from localization scores."""

    points = [CircuitPoint(str(n), float(s)) for n, s in zip(names, scores, strict=True)]
    key = (lambda p: abs(p.importance)) if absolute else (lambda p: p.importance)
    return sorted(points, key=key, reverse=True)


def area_under_faithfulness_curve(fractions: Iterable[float], recoveries: Iterable[float]) -> float:
    """Area under a circuit-size vs behavioral-recovery curve.

    A strong sparse circuit recovers substantial model behavior using a small retained fraction.
    """

    x = np.asarray(list(fractions), dtype=float)
    y = np.asarray(list(recoveries), dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("fractions and recoveries must be matching one-dimensional arrays")
    order = np.argsort(x)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y[order], x[order]))
