# fusion.py
"""
This module contains functions for early fusion and cross-modal attention.
    
Functions:
    early_fusion(features): Concatenates a list of features along the last dimension.
    CrossModalAttention(d_model, n_heads): A class for cross-modal attention.
    forward(self, q, k, v): Computes cross-modal attention between query, key, and value.
    
Classes:
    EarlyFusion: A class for early fusion of input features.
    CrossModalAttention: A class for cross-modal attention.
    
Imports:
    torch: Module for defining neural network layers.
    torch.nn: Module for defining neural network layers.
"""

import torch
import torch.nn as nn

def early_fusion(features):
    # features: list of [B, T, d_model]
    """
    Concatenate a list of features along the last dimension.
    
    Args:
        features (list of torch.Tensor): The list of features to concatenate.
            Each element should be of shape [B, T, d_model].
    
    Returns:
        torch.Tensor: The concatenated features of shape [B, T, d_model * len(features)].
    """
    return torch.cat(features, dim=-1)

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        Args:
            d_model (int): The number of features in the model
            n_heads (int): The number of attention heads
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
    def forward(self, q, k, v):  # [T, B, d]
        """
        Compute cross-modal attention between query, key, and value.
        
        Args:
            q (torch.Tensor): The query tensor, of shape [T, B, d_model].
            k (torch.Tensor): The key tensor, of shape [T, B, d_model].
            v (torch.Tensor): The value tensor, of shape [T, B, d_model].
        
        Returns:
            torch.Tensor: The output of the attention layer, of shape [T, B, d_model].
        """
        return self.attn(q, k, v)[0]