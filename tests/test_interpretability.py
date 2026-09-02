import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dataloaders import get_loader, prepare_batch
from interpretability.ablation_study import ablate_heads, masked_heads
from interpretability.attention_analysis import extract_attention_maps
from model import MultimodalTransformer


def test_attention_is_normalized_and_preserves_modes_and_predictions(cfg):
    model = MultimodalTransformer(cfg).eval()
    inputs, _ = prepare_batch(next(iter(get_loader(cfg))))
    with torch.no_grad():
        prediction = model(**inputs)
        inspected, maps = model(**inputs, return_attention=True)
    torch.testing.assert_close(prediction, inspected, atol=1e-6, rtol=1e-5)
    assert maps[0].shape == (4, 2, 6, 6)
    model.train()
    model.video_enc.eval()
    modes = [m.training for m in model.modules()]
    result = extract_attention_maps(model, inputs)
    assert result.shape == (2, 6, 6)
    np.testing.assert_allclose(result.sum(axis=-1), 1, atol=1e-6)
    assert modes == [m.training for m in model.modules()]


def test_head_removal_matches_zeroing_head_output_and_restores_on_error(cfg):
    model = MultimodalTransformer(cfg).eval()
    attn = model.layers[0].self_attn
    original = attn.out_proj.weight.detach().clone()
    x = torch.randn(2, 5, attn.embed_dim)
    # Independent attention calculation: remove one head before output projection.
    q, k, v = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q, k, v = [a.reshape(2, 5, attn.num_heads, attn.head_dim).transpose(1, 2) for a in (q, k, v)]
    probabilities = (q @ k.transpose(-2, -1) / attn.head_dim**0.5).softmax(dim=-1)
    heads = probabilities @ v
    heads[:, 1] = 0
    expected = F.linear(heads.transpose(1, 2).reshape(2, 5, attn.embed_dim), original, attn.out_proj.bias)
    with masked_heads(model, 0, [1, 1]):
        actual = attn(x, x, x, need_weights=False)[0]
        torch.testing.assert_close(actual, expected)
    assert torch.equal(attn.out_proj.weight, original)
    with pytest.raises(RuntimeError, match="evaluation failure"):
        with masked_heads(model, 0, [0]):
            raise RuntimeError("evaluation failure")
    assert torch.equal(attn.out_proj.weight, original)
    with pytest.raises(ValueError, match="head index"):
        with masked_heads(model, 0, [999]):
            pass
    assert torch.equal(attn.out_proj.weight, original)


def test_empty_head_intervention_and_encoded_modality_removal(cfg):
    model = MultimodalTransformer(cfg).eval()
    inputs, target = prepare_batch(next(iter(get_loader(cfg))))
    with torch.no_grad():
        baseline = (model(**inputs) - target).square().mean().item()
        removed = model(**inputs, ablate_modalities=["video"])
        changed_inputs = {**inputs, "video": inputs["video"] + 100}
        removed_changed = model(**changed_inputs, ablate_modalities=["video"])
    torch.testing.assert_close(removed, removed_changed)
    model.train()
    assert ablate_heads(model, {**inputs, "target": target}, 0, []) == pytest.approx(baseline)
    assert model.training
