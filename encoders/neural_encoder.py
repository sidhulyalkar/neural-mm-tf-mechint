"""Local temporal features from continuous neural channels."""

from torch import nn


class NeuralEncoder(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)

    def forward(self, x):
        """Map [B,T,C] to [B,T,d]; the centered kernel is noncausal."""
        return self.conv(x.transpose(1, 2)).transpose(1, 2)
