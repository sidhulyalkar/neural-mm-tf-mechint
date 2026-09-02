"""Inspect temporal attention; weights alone do not establish feature importance."""

import torch

from dataloaders import INPUT_NAMES
from interpretability.evaluation import evaluation_mode


def extract_attention_maps(model, inputs, layer_idx=0, head_idx=None):
    """Return [H,T,T] (or [T,T]) for the first sample, without installing hooks."""
    if not 0 <= layer_idx < len(model.layers):
        raise ValueError("Invalid layer index")
    if head_idx is not None and not 0 <= head_idx < model.layers[layer_idx].self_attn.num_heads:
        raise ValueError("Invalid head index")
    with evaluation_mode(model), torch.no_grad():
        _, maps = model(**{key: inputs[key] for key in INPUT_NAMES}, return_attention=True)
    result = maps[layer_idx][0].detach().cpu().numpy()
    return result if head_idx is None else result[head_idx]
