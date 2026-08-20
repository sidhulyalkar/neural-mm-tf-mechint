"""Causal ablations for the legacy multimodal teaching model."""
from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def _temporarily_ablate_heads(model, layer_idx: int, head_indices: list[int]):
    """Remove selected heads at the attention output projection.

    Multi-head attention concatenates head outputs before ``out_proj``. Therefore head
    ``h`` occupies *input columns* ``[h*d_head:(h+1)*d_head]`` of ``out_proj.weight``.
    Zeroing rows, as the original code did, removes output coordinates rather than heads.
    """

    attention = model.transformer.layers[layer_idx].self_attn
    columns: list[tuple[int, int, torch.Tensor]] = []
    try:
        for head in head_indices:
            if head < 0 or head >= attention.num_heads:
                raise IndexError(f"head {head} outside [0, {attention.num_heads})")
            start = head * attention.head_dim
            end = (head + 1) * attention.head_dim
            backup = attention.out_proj.weight.data[:, start:end].clone()
            columns.append((start, end, backup))
            attention.out_proj.weight.data[:, start:end].zero_()
        yield
    finally:
        for start, end, backup in columns:
            attention.out_proj.weight.data[:, start:end].copy_(backup)


def ablate_heads(model, inputs, layer_idx: int, head_indices: list[int]) -> float:
    """Return post-ablation MSE for backward compatibility with the original dashboard."""

    model_inputs = {k: v for k, v in inputs.items() if k != "target"}
    target = inputs["target"]
    model.eval()
    with _temporarily_ablate_heads(model, layer_idx, list(head_indices)):
        with torch.inference_mode():
            predictions = model(**model_inputs)
    return torch.nn.functional.mse_loss(predictions, target).item()


def head_ablation_effect(model, inputs, layer_idx: int, head_indices: list[int]) -> dict[str, float]:
    """Return baseline, ablated, and delta loss so the intervention has a causal reference."""

    model_inputs = {k: v for k, v in inputs.items() if k != "target"}
    target = inputs["target"]
    model.eval()
    with torch.inference_mode():
        baseline = torch.nn.functional.mse_loss(model(**model_inputs), target).item()
    ablated = ablate_heads(model, inputs, layer_idx, head_indices)
    return {"baseline_loss": baseline, "ablated_loss": ablated, "delta_loss": ablated - baseline}
