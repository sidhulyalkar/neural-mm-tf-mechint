"""Reversible attention-head interventions and paired modality removal."""

from contextlib import contextmanager

import torch

from dataloaders import INPUT_NAMES
from interpretability.evaluation import evaluate, evaluation_mode
from model import MODALITIES


@contextmanager
def masked_heads(model, layer_idx, head_indices):
    """Zero head input columns of W_O; restore even if evaluation raises.

    Rows of W_O are output coordinates, not heads. This context temporarily
    mutates weights and must not share a model with concurrent inference.
    """
    if not 0 <= layer_idx < len(model.layers):
        raise ValueError("Invalid layer index")
    attn = model.layers[layer_idx].self_attn
    heads = sorted(set(head_indices))
    if any(type(h) is not int or not 0 <= h < attn.num_heads for h in heads):
        raise ValueError("Invalid head index")
    weight = attn.out_proj.weight
    backup = weight.detach().clone()
    try:
        with torch.no_grad():
            for head in heads:
                weight[:, head * attn.head_dim : (head + 1) * attn.head_dim].zero_()
        yield
    finally:
        with torch.no_grad():
            weight.copy_(backup)


def ablate_heads(model, inputs, layer_idx, head_indices):
    """Return post-ablation MSE on one batch (not the change from baseline)."""
    with masked_heads(model, layer_idx, head_indices), evaluation_mode(model), torch.no_grad():
        predictions = model(**{key: inputs[key] for key in INPUT_NAMES})
        return (predictions - inputs["target"]).square().mean().item()


def modality_ablation(model, loader, device="cpu"):
    """Evaluate every encoded-branch removal against identical held-out samples."""
    baseline = evaluate(model, loader, device)
    results = {}
    for modality in MODALITIES:
        mse = evaluate(model, loader, device, ablate_modalities=(modality,))
        results[modality] = {"mse": mse, "delta_mse": mse - baseline}
    return {"baseline_mse": baseline, "ablations": results}
