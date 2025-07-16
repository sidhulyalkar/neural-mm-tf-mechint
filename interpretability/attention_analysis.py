# interpretability/attention_analysis.py
"""
Attention analysis for the multimodal transformer model.

Functions:
    extract_attention_maps(model, inputs, layer_idx=0, head_idx=None): Extracts attention weights from specified transformer layer.
    plot_attention_map(attn, title=None): Plots the attention map.
    plot_attention_map_over_time(attn, title=None): Plots the attention map over time.
"""
import torch
import numpy as np

def extract_attention_maps(model, inputs, layer_idx=0, head_idx=None):
    """
    Extracts attention weights from specified transformer layer.
    Args:
        model: MultimodalTransformer
        inputs: dict with tensors {neural, video, behavior, meta}
        layer_idx: index of the transformer encoder layer
        head_idx: if specified, returns only that head
    Returns:
        attn: [heads, T, T] or [T, T] if head_idx is set
    """
    # Attach hook to capture attention
    attn_data = {}
    def hook(module, inp, outp):
        """
        Hook to capture attention weights from a transformer layer.
        Args:
            module: nn.Module, the transformer layer
            inp: tuple of input tensors
            outp: output tensor
        Returns:
            None
        """
        attn_data['weights'] = module.self_attn.attn_output_weights.detach().cpu().numpy()

    # register hook
    layer = model.transformer.layers[layer_idx].self_attn
    handle = layer.register_forward_hook(hook)
    # forward
    _ = model(**inputs)
    handle.remove()
    attn = attn_data['weights']  # [B*num_heads, T, T]
    H = layer.num_heads
    B = inputs['neural'].shape[0]
    attn = attn.reshape(B, H, *attn.shape[-2:])[0]
    return attn if head_idx is None else attn[head_idx]