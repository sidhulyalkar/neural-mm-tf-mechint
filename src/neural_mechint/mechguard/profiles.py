from __future__ import annotations

from dataclasses import dataclass, field

from .monitors import ConflictMonitor, Severity, ThresholdMonitor
from .policies import RiskPolicy


@dataclass(frozen=True)
class DeploymentProfile:
    name: str
    monitors: tuple[object, ...]
    policy: RiskPolicy
    description: str
    metadata: dict[str, str] = field(default_factory=dict)


def grounded_rag_profile() -> DeploymentProfile:
    return DeploymentProfile(
        name="grounded-rag",
        monitors=(ConflictMonitor(evidence_signal="retrieval_grounding", pressure_signal="compliance_pressure", evidence_threshold=0.70, pressure_threshold=0.65), ThresholdMonitor(signal="answer_commitment", threshold=0.85, severity=Severity.WARN)),
        policy=RiskPolicy(intervene_on=Severity.WARN, escalate_critical_count=2),
        description="Protect trusted retrieval evidence from contradictory user pressure while keeping the intervention policy explicit and auditable.",
        metadata={"trusted_source": "retrieval-context"},
    )
