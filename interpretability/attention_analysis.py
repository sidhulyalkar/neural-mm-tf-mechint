"""Attention inspection utilities for the legacy multimodal teaching model.

Attention maps are observational evidence. Use them to nominate hypotheses, then validate
those hypotheses with ablation or activation patching.
"""
from __future__ import annotations

import numpy as np
import torch


def extract_attention_maps(model, inputs, layer_idx: int = 0, head_idx: int | None = None):
    """Return per-head attention weights from one legacy transformer layer.

    PyTorch's ``TransformerEncoderLayer`` normally calls ``MultiheadAttention`` with
    ``need_weights=False``. The old implementation attempted to read a non-existent
    ``attn_output_weights`` attribute. Here we temporarily wrap the attention forward
    pass so weights are explicitly returned and captured.
    """

    layer = model.transformer.layers[layer_idx].self_attn
    original_forward = layer.forward
    captured: dict[str, torch.Tensor] = {}

    def wrapped_forward(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        output, weights = original_forward(*args, **kwargs)
        captured["weights"] = weights.detach().cpu()
        return output, weights

    model_inputs = {k: v for k, v in inputs.items() if k != "target"}
    layer.forward = wrapped_forward
    try:
        model.eval()
        with torch.inference_mode():
            model(**model_inputs)
    finally:
        layer.forward = original_forward

    if "weights" not in captured:
        raise RuntimeError("attention weights were not captured")

    weights = captured["weights"]
    if weights.ndim == 3:
        batch = inputs["neural"].shape[0]
        weights = weights.reshape(batch, layer.num_heads, *weights.shape[-2:])
    first_example = weights[0].numpy()
    if head_idx is not None:
        return first_example[head_idx]
    return first_example


def attention_entropy(attention: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Entropy of each attention row; useful as a descriptive diagnostic only."""

    attn = np.asarray(attention, dtype=float)
    attn = np.clip(attn, eps, 1.0)
    return -(attn * np.log(attn)).sum(axis=-1)
