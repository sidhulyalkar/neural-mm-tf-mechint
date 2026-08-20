from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_mechint.mechstate import MechanisticTrajectory


@dataclass(frozen=True)
class MechanisticAudit:
    factual_area: float
    pressure_area: float
    crossover_layer: int | None
    factual_final: float
    pressure_final: float


def _curve_area(points: list[tuple[int, float]]) -> float:
    if len(points) < 2:
        return 0.0
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x))


def audit_belief_compliance_trajectory(trajectory: MechanisticTrajectory, *, factual_signal: str = "factual_evidence", pressure_signal: str = "compliance_pressure") -> MechanisticAudit:
    factual_curve = trajectory.signal_curve(factual_signal)
    pressure_curve = trajectory.signal_curve(pressure_signal)
    final = trajectory.final()
    return MechanisticAudit(_curve_area(factual_curve), _curve_area(pressure_curve), trajectory.transition_layer(factual_signal, pressure_signal), final.value(factual_signal, 0.0), final.value(pressure_signal, 0.0))
