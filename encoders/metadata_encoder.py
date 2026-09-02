"""Projection of numeric task context available at prediction time."""

from torch import nn


class MetadataEncoder(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.fc = nn.Linear(in_dim, d_model)

    def forward(self, x):
        """Map [B,T,metadata_dim] to [B,T,d]."""
        return self.fc(x)
