from neural_mechint.mecheval import MetricDelta, MechanisticModelDiff, audit_belief_compliance_trajectory, audit_intervention_curves
from neural_mechint.mechstate import MechanisticState, MechanisticTrajectory, StateSignal


def test_mechanistic_audit_finds_crossover():
    trajectory = MechanisticTrajectory([MechanisticState(0, -1, {"factual_evidence": StateSignal("factual_evidence", 0.9), "compliance_pressure": StateSignal("compliance_pressure", 0.1)}), MechanisticState(1, -1, {"factual_evidence": StateSignal("factual_evidence", 0.7), "compliance_pressure": StateSignal("compliance_pressure", 0.8)})])
    audit = audit_belief_compliance_trajectory(trajectory)
    assert audit.crossover_layer == 1
    assert audit.factual_final == 0.7


def test_intervention_audit_prefers_high_recovery_low_kl():
    audit = audit_intervention_curves([0.1, 0.5, 1.0], [0.2, 0.6, 0.9], [0.4, 0.9, 1.0], [0.01, 0.05, 0.4])
    assert audit.preferred_fraction == 0.5


def test_model_diff_ranks_absolute_changes():
    diff = MechanisticModelDiff(behavioral=[MetricDelta("accuracy", 0.7, 0.8)], state=[MetricDelta("pressure", 0.8, 0.2)], causal=[])
    assert diff.largest_changes(top_k=1)[0].name == "pressure"
