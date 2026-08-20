from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterventionAudit:
    necessity_auc: float
    sufficiency_auc: float
    collateral_kl_mean: float
    preferred_fraction: float


def _auc(fractions: list[float], values: list[float]) -> float:
    if len(fractions) != len(values) or len(fractions) < 2:
        raise ValueError("fractions and values must be matching arrays with >= 2 points")
    x = np.asarray(fractions, dtype=float)
    y = np.asarray(values, dtype=float)
    order = np.argsort(x)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y[order], x[order]))


def audit_intervention_curves(fractions: list[float], necessity_recovery: list[float], sufficiency_recovery: list[float], collateral_kl: list[float]) -> InterventionAudit:
    if len(collateral_kl) != len(fractions):
        raise ValueError("collateral_kl must match fractions")
    necessity_auc = _auc(fractions, necessity_recovery)
    sufficiency_auc = _auc(fractions, sufficiency_recovery)
    utility = np.asarray(sufficiency_recovery) - np.asarray(collateral_kl)
    preferred = float(fractions[int(np.argmax(utility))])
    return InterventionAudit(necessity_auc, sufficiency_auc, float(np.mean(collateral_kl)), preferred)
