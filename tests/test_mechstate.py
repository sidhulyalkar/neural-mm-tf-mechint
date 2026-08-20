import torch

from neural_mechint.mechstate import LinearSignalProbe, MechanisticState, MechanisticTrajectory, StateEstimator, StateSignal
from neural_mechint.types import ActivationCache


def test_state_estimator_builds_layer_trajectory():
    probes = [LinearSignalProbe("factual_evidence", torch.tensor([1.0, 0.0])), LinearSignalProbe("compliance_pressure", torch.tensor([0.0, 1.0]))]
    cache = ActivationCache(values={0: torch.tensor([[[2.0, -2.0]]]), 1: torch.tensor([[[1.0, 1.5]]])})
    trajectory = StateEstimator(probes).estimate(cache)
    assert [state.layer for state in trajectory.states] == [0, 1]
    assert trajectory.states[0].value("factual_evidence") > 0.8
    assert trajectory.states[0].value("compliance_pressure") < 0.2
    assert trajectory.transition_layer("factual_evidence", "compliance_pressure") == 1


def test_trajectory_peak_and_final():
    states = [MechanisticState(0, -1, {"x": StateSignal("x", 0.2)}), MechanisticState(1, -1, {"x": StateSignal("x", 0.9)})]
    trajectory = MechanisticTrajectory(states)
    assert trajectory.peak("x").layer == 1
    assert trajectory.final().layer == 1
