"""Learned representations for discrete behavioral event IDs."""

from torch import nn


class BehaviorEncoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """Map int64 IDs [B,T] to embeddings [B,T,d]."""
        return self.embed(x)
