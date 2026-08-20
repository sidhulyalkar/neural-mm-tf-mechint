import math

import torch

from neural_mechint.metrics import logit_difference, normalized_recovery, symmetric_kl


def test_logit_difference_final_position():
    logits = torch.zeros(2, 3, 5)
    logits[:, -1, 2] = 4
    logits[:, -1, 1] = 1
    assert torch.allclose(logit_difference(logits, 2, 1), torch.tensor([3.0, 3.0]))


def test_normalized_recovery():
    assert normalized_recovery(10, 2, 6).item() == 0.5
    assert math.isnan(normalized_recovery(2, 2, 2).item())


def test_symmetric_kl_identity_is_zero():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(symmetric_kl(logits, logits), torch.zeros(1), atol=1e-7)
