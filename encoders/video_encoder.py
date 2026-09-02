"""Projection of precomputed video features; no raw-frame extraction."""

from torch import nn


class VideoEncoder(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.fc = nn.Linear(in_dim, d_model)

    def forward(self, x):
        """Map [B,T,video_feature_dim] to [B,T,d]."""
        return self.fc(x)
