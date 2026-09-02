"""Feature concatenation and a standalone cross-attention building block."""

import torch
from torch import nn


def early_fusion(features):
    """Concatenate aligned [B,T,d] representations along the feature axis."""
    return torch.cat(features, dim=-1)


class CrossModalAttention(nn.Module):
    """Experimental primitive, not used by the current MultimodalTransformer.

    Inputs q, k, v use [T,B,d] layout; output uses the query sequence length.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)

    def forward(self, q, k, v):
        return self.attn(q, k, v, need_weights=False)[0]
