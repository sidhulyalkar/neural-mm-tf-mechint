# encoders/behavior_encoder.py
"""
This module defines the BehaviorEncoder class for encoding behavioral events.

Classes:
    BehaviorEncoder: A PyTorch module that represents the BehaviorEncoder.

Functions: 
    forward: Performs a forward pass through the BehaviorEncoder.

Imports:            
    torch.nn: Module for defining neural network layers.            
    nn.Module: Base class for all neural network modules.            
"""
import torch.nn as nn
class BehaviorEncoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size (int): The size of the vocabulary for the embedding layer.
            d_model (int): The dimension of the model's output.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):  # [B, T]
        """
        Args:
            x (torch.Tensor): The input data, of shape [B, T]

        Returns:
            torch.Tensor: The output of the embedding layer, of shape [B, T, d_model]
        """
        return self.embed(x)