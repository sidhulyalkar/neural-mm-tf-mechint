from dataclasses import dataclass

import torch

from neural_mechint.mechguard import ConflictMonitor, GuardAction, MechGuardRuntime, RiskPolicy, SteeringPlan
from neural_mechint.mechstate import LinearSignalProbe, StateEstimator
from neural_mechint.types import ActivationCache


@dataclass
class FakeForward:
    logits: torch.Tensor
    cache: ActivationCache


class FakeAdapter:
    def __init__(self):
        self.steering_calls = 0

    def forward_with_cache(self, prompt, layers=None, detach_to_cpu=True):
        del prompt, layers, detach_to_cpu
        return FakeForward(torch.tensor([[[0.0, 1.0]]]), ActivationCache(values={0: torch.tensor([[[2.0, 2.0]]])}))

    def forward_with_steering(self, prompt, *, layer, direction, coefficient, position):
        del prompt, layer, direction, coefficient, position
        self.steering_calls += 1
        return torch.tensor([[[2.0, 0.0]]])

    def generate(self, prompt, *, max_new_tokens=96, **kwargs):
        del prompt, max_new_tokens, kwargs
        return "baseline"

    def generate_with_steering(self, prompt, *, layer, direction, coefficient, position, max_new_tokens=96, **kwargs):
        del prompt, layer, direction, coefficient, position, max_new_tokens, kwargs
        self.steering_calls += 1
        return "corrected"


def make_runtime(adapter, plan=True):
    estimator = StateEstimator([LinearSignalProbe("factual_evidence", torch.tensor([1.0, 0.0])), LinearSignalProbe("compliance_pressure", torch.tensor([0.0, 1.0]))])
    steering_plan = SteeringPlan(layer=0, direction=torch.tensor([-1.0, 0.0]), coefficient=1.0) if plan else None
    return MechGuardRuntime(adapter, estimator, [ConflictMonitor()], RiskPolicy(), steering_plan=steering_plan)


def test_guard_intervenes_on_conflict():
    adapter = FakeAdapter()
    result = make_runtime(adapter).run("prompt")
    assert result.decision.action == GuardAction.INTERVENE
    assert adapter.steering_calls == 1
    assert result.receipt.intervention is not None


def test_guard_escalates_when_no_plan():
    result = make_runtime(FakeAdapter(), plan=False).run("prompt")
    assert result.decision.action == GuardAction.ESCALATE


def test_guarded_generation_uses_intervened_path():
    adapter = FakeAdapter()
    result = make_runtime(adapter).generate("prompt")
    assert result.response == "corrected"
    assert result.decision.action == GuardAction.INTERVENE
