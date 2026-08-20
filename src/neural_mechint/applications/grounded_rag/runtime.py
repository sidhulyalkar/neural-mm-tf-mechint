from __future__ import annotations

import torch

from neural_mechint.mechguard import MechGuardRuntime, SteeringPlan, grounded_rag_profile
from neural_mechint.mechstate import LinearSignalProbe, StateEstimator


def build_grounded_rag_guard(adapter, *, retrieval_direction: torch.Tensor, pressure_direction: torch.Tensor, commitment_direction: torch.Tensor, correction_direction: torch.Tensor, intervention_layer: int, monitor_layers: list[int], coefficient: float = 1.0) -> MechGuardRuntime:
    """Assemble the reference grounded-RAG control profile from calibrated directions."""
    profile = grounded_rag_profile()
    estimator = StateEstimator([LinearSignalProbe("retrieval_grounding", retrieval_direction), LinearSignalProbe("compliance_pressure", pressure_direction), LinearSignalProbe("answer_commitment", commitment_direction)])
    plan = SteeringPlan(layer=intervention_layer, direction=correction_direction, coefficient=coefficient, label="grounded-rag-correction")
    return MechGuardRuntime(adapter, estimator, profile.monitors, profile.policy, layers=monitor_layers, steering_plan=plan)
