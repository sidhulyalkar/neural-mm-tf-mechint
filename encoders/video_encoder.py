# encoders/video_encoder.py

import torch.nn as nn
class VideoEncoder(nn.Module):
    def __init__(self, in_dim, d_model):
        """
        Args:
            in_dim (int): The dimension of the input features.
            d_model (int): The dimension of the model's output.
        """

        super().__init__()
        self.fc = nn.Linear(in_dim, d_model)
    def forward(self, x):  # [B, T, feat]
        """
        Args:
            x (torch.Tensor): The input data, of shape [B, T, in_dim]

        Returns:
            torch.Tensor: The output of the fully connected layer, of shape [B, T, d_model]
        """
        return self.fc(x)