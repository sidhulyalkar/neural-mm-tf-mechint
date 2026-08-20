from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from neural_mechint.mechstate import MechanisticTrajectory, StateEstimator, StateReceipt

from .interventions import SteeringPlan, apply_steering
from .monitors import MonitorEvent
from .policies import GuardAction, GuardDecision, RiskPolicy


@dataclass
class GuardRun:
    baseline_logits: torch.Tensor
    final_logits: torch.Tensor
    decision: GuardDecision
    events: list[MonitorEvent]
    receipt: StateReceipt


@dataclass
class GuardGeneration:
    response: str | None
    decision: GuardDecision
    events: list[MonitorEvent]
    receipt: StateReceipt


class MechGuardRuntime:
    """Low-overhead state monitor with optional conditional intervention."""
    def __init__(self, adapter, estimator: StateEstimator, monitors: Iterable[object], policy: RiskPolicy, *, layers: Iterable[int] | None = None, steering_plan: SteeringPlan | None = None):
        self.adapter = adapter
        self.estimator = estimator
        self.monitors = list(monitors)
        self.policy = policy
        self.layers = None if layers is None else list(layers)
        self.steering_plan = steering_plan

    def _inspect(self, prompt: str, prompt_id: str | None):
        baseline = self.adapter.forward_with_cache(prompt, layers=self.layers, detach_to_cpu=True)
        trajectory = self.estimator.estimate(baseline.cache, prompt_id=prompt_id)
        events = [event for monitor in self.monitors for event in monitor.evaluate(trajectory)]
        return baseline, trajectory, events, self.policy.decide(events)

    @staticmethod
    def _receipt(decision: GuardDecision, trajectory: MechanisticTrajectory, events: list[MonitorEvent], intervention: dict[str, object] | None) -> StateReceipt:
        return StateReceipt(decision=decision.action.value, reasons=decision.reasons, trajectory=trajectory, intervention=intervention, metrics={"risk_score": decision.max_score, "event_count": float(len(events))})

    def _resolve_intervention(self, decision: GuardDecision) -> GuardDecision:
        if decision.action == GuardAction.INTERVENE and self.steering_plan is None:
            return GuardDecision(GuardAction.ESCALATE, [*decision.reasons, "policy requested intervention but no steering plan is configured"], decision.max_score)
        return decision

    def run(self, prompt: str, *, prompt_id: str | None = None) -> GuardRun:
        baseline, trajectory, events, decision = self._inspect(prompt, prompt_id)
        decision = self._resolve_intervention(decision)
        final_logits = baseline.logits
        intervention = None
        if decision.action == GuardAction.INTERVENE:
            assert self.steering_plan is not None
            final_logits = apply_steering(self.adapter, prompt, self.steering_plan)
            intervention = self.steering_plan.public_dict()
        receipt = self._receipt(decision, trajectory, events, intervention)
        return GuardRun(baseline.logits, final_logits, decision, events, receipt)

    def generate(self, prompt: str, *, prompt_id: str | None = None, max_new_tokens: int = 96, **generation_kwargs) -> GuardGeneration:
        _baseline, trajectory, events, decision = self._inspect(prompt, prompt_id)
        decision = self._resolve_intervention(decision)
        intervention = None
        if decision.action == GuardAction.ESCALATE:
            response = None
        elif decision.action == GuardAction.INTERVENE:
            assert self.steering_plan is not None
            plan = self.steering_plan
            response = self.adapter.generate_with_steering(prompt, layer=plan.layer, direction=plan.direction, coefficient=plan.coefficient, position=plan.position, max_new_tokens=max_new_tokens, **generation_kwargs)
            intervention = plan.public_dict()
        else:
            response = self.adapter.generate(prompt, max_new_tokens=max_new_tokens, **generation_kwargs)
        receipt = self._receipt(decision, trajectory, events, intervention)
        return GuardGeneration(response, decision, events, receipt)
