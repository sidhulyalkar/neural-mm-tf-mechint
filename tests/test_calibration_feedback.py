import torch

from neural_mechint.mechstate import fit_mean_difference_probe
from neural_mechint.mechtune import FeedbackRecord, select_training_candidates


def test_mean_difference_probe_separates_simple_classes():
    positive = torch.tensor([[2.0, 0.0], [3.0, 0.2], [2.5, -0.2]])
    negative = torch.tensor([[-2.0, 0.0], [-3.0, -0.1], [-2.5, 0.2]])
    probe, report = fit_mean_difference_probe("grounding", positive, negative)
    assert report.train_accuracy == 1.0
    assert probe.score(torch.tensor([2.0, 0.0])) > 0.5
    assert probe.score(torch.tensor([-2.0, 0.0])) < 0.5


def test_feedback_candidates_require_causal_gain():
    records = [FeedbackRecord("good", "intervene", 0.2, 0.9, 0.8, (10, 11), ("pressure",)), FeedbackRecord("flat", "intervene", 0.5, 0.51, 0.9, (8,), ("pressure",))]
    selected = select_training_candidates(records, min_gain=0.05)
    assert [row.prompt_id for row in selected] == ["good"]
    assert selected[0].target_layers == (10, 11)
