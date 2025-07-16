# encoders/neural_encoder.py

import torch.nn as nn
class NeuralEncoder(nn.Module):
    def __init__(self, in_channels, d_model):
        """
        Args:
            in_channels (int): The number of channels in the input
            d_model (int): The output dimension of the model
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
    def forward(self, x):  # x: [B, T, C]
        """
        Args:
            x (torch.Tensor): The input data, of shape [B, T, C]

        Returns:
            torch.Tensor: The output of the convolutional layer, of shape [B, T, d_model]
        """
        x = x.transpose(1,2)  # -> [B, C, T]
        out = self.conv(x)
        return out.transpose(1,2)  # -> [B, T, d_model]