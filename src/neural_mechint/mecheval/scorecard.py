from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeploymentScorecard:
    """Report spanning behavior, internal state, causal control and operational cost."""
    task_accuracy: float
    false_compliance_rate: float
    mechanistic_failure_recall: float
    intervention_success_rate: float
    collateral_kl: float
    intervention_rate: float
    latency_overhead_fraction: float

    @property
    def control_efficiency(self) -> float:
        denominator = max(self.intervention_rate + self.latency_overhead_fraction + self.collateral_kl, 1e-8)
        return self.intervention_success_rate / denominator

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "control_efficiency": self.control_efficiency}


def compare_scorecards(before: DeploymentScorecard, after: DeploymentScorecard) -> dict[str, float]:
    before_dict = before.to_dict()
    after_dict = after.to_dict()
    return {key: after_dict[key] - before_dict[key] for key in before_dict}
