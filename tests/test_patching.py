import torch

from neural_mechint.patching import attribution_patch_score


def test_attribution_patch_score():
    clean = torch.tensor([[2.0, 4.0]])
    corrupt = torch.tensor([[1.0, 1.0]])
    grad = torch.tensor([[3.0, 2.0]])
    score = attribution_patch_score(clean, corrupt, grad)
    assert score.item() == 9.0
