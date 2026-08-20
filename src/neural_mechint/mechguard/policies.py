from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .monitors import MonitorEvent, Severity


class GuardAction(StrEnum):
    PASS = "pass"
    INTERVENE = "intervene"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class GuardDecision:
    action: GuardAction
    reasons: list[str]
    max_score: float = 0.0


@dataclass(frozen=True)
class RiskPolicy:
    """Simple, auditable mapping from mechanistic events to control actions."""
    intervene_on: Severity = Severity.WARN
    escalate_critical_count: int = 2

    @staticmethod
    def _rank(severity: Severity) -> int:
        return {Severity.INFO: 0, Severity.WARN: 1, Severity.CRITICAL: 2}[severity]

    def decide(self, events: list[MonitorEvent]) -> GuardDecision:
        if not events:
            return GuardDecision(GuardAction.PASS, [])
        critical = [event for event in events if event.severity == Severity.CRITICAL]
        reasons = [event.message for event in events]
        max_score = max(event.score for event in events)
        if len(critical) >= self.escalate_critical_count:
            return GuardDecision(GuardAction.ESCALATE, reasons, max_score)
        if any(self._rank(event.severity) >= self._rank(self.intervene_on) for event in events):
            return GuardDecision(GuardAction.INTERVENE, reasons, max_score)
        return GuardDecision(GuardAction.PASS, reasons, max_score)
