# encoders/metadata_encoder.py
"""
This module defines the MetadataEncoder class for encoding metadata.

Classes:
    MetadataEncoder: A PyTorch module that represents the MetadataEncoder.

Functions:
    forward: Performs a forward pass through the MetadataEncoder.

Imports:
    torch.nn: Module for defining neural network layers.
    nn.Module: Base class for all neural network modules.
"""
import torch.nn as nn
class MetadataEncoder(nn.Module):
    def __init__(self, in_dim, d_model):
        """
        Args:
            in_dim (int): The input dimension of the metadata.
            d_model (int): The output dimension of the model.
        """
        super().__init__()
        self.fc = nn.Linear(in_dim, d_model)
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input data, of shape [B, T, in_dim]

        Returns:
            torch.Tensor: The output of the fully connected layer, of shape [B, T, d_model]
        """

        return self.fc(x)