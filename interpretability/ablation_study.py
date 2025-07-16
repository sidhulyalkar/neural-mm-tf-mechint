# interpretability/ablation_study.py
"""
Ablation study for the multimodal transformer model.

Functions:
    ablate_heads(model, inputs, layer_idx, head_indices): Ablates specified heads by zeroing their projection matrices.
    Returns performance drop.
"""
import torch
import numpy as np

def ablate_heads(model, inputs, layer_idx, head_indices):
    """
    Ablates specified heads by zeroing their projection matrices.
    Returns performance drop.
    """
    # Backup
    backup = []
    layer = model.transformer.layers[layer_idx].self_attn
    for h in head_indices:
        # backup out_proj for head
        start = h * layer.head_dim
        end = (h+1) * layer.head_dim
        backup.append(layer.out_proj.weight.data[start:end].clone())
        # zero out
        layer.out_proj.weight.data[start:end].zero_()
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(**inputs)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(preds, inputs['target'])
    # Restore
    for h, orig in zip(head_indices, backup):
        start = h * layer.head_dim
        end = (h+1) * layer.head_dim
        layer.out_proj.weight.data[start:end] = orig
    return loss.item()