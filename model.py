"""Four modality encoders, early fusion, and inspectable temporal attention."""

import math

import torch
from torch import nn

from encoders.behavior_encoder import BehaviorEncoder
from encoders.metadata_encoder import MetadataEncoder
from encoders.neural_encoder import NeuralEncoder
from encoders.video_encoder import VideoEncoder
from fusion import early_fusion

MODALITIES = ("neural", "video", "behavior", "meta")


class TemporalBlock(nn.Module):
    """Explicit post-norm transformer block with optional per-head attention output."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width)
        )

    def forward(self, x, return_attention=False):
        attended, weights = self.self_attn(x, x, x, need_weights=return_attention, average_attn_weights=False)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, weights


class MultimodalTransformer(nn.Module):
    """Offline sequence regression from already aligned multimodal features.

    d_model is the per-modality width; temporal attention operates at 4*d_model.
    Modality ablations zero an encoded branch, including its learned bias.
    """

    def __init__(self, cfg):
        super().__init__()
        data, model = cfg["data"], cfg["model"]
        d = model["d_model"]
        self.input_dims = (
            data["num_neural_channels"],
            data["video_embed_dim"],
            data["behavioral_vocab"],
            data["metadata_dim"],
        )
        self.neural_enc = NeuralEncoder(self.input_dims[0], d)
        self.video_enc = VideoEncoder(self.input_dims[1], d)
        self.behav_enc = BehaviorEncoder(self.input_dims[2], d)
        self.meta_enc = MetadataEncoder(self.input_dims[3], d)
        self.layers = nn.ModuleList(
            [TemporalBlock(d * 4, model["n_heads"], model["dropout"]) for _ in range(model["num_layers"])]
        )
        self.output = nn.Linear(d * 4, 1)

    def _validate_inputs(self, neural, video, behavior, meta):
        inputs = (neural, video, behavior, meta)
        for name, value, width in zip(MODALITIES, inputs, self.input_dims):
            rank = 2 if name == "behavior" else 3
            if value.ndim != rank or value.shape[:2] != neural.shape[:2]:
                raise ValueError(f"{name} must have aligned batch/time axes and rank {rank}")
            if name != "behavior" and value.shape[-1] != width:
                raise ValueError(f"{name} feature width must be {width}")
            if name != "behavior" and not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
        if not neural.shape[0] or not neural.shape[1]:
            raise ValueError("Batch and sequence dimensions must be nonempty")
        if behavior.dtype != torch.int64:
            raise ValueError("behavior must contain int64 event IDs")
        if torch.any((behavior < 0) | (behavior >= self.input_dims[2])):
            raise ValueError("behavior event IDs are outside the configured vocabulary")

    def forward(self, neural, video, behavior, meta, *, ablate_modalities=(), return_attention=False):
        self._validate_inputs(neural, video, behavior, meta)
        unknown = set(ablate_modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"Unknown modalities: {sorted(unknown)}")
        features = [
            self.neural_enc(neural),
            self.video_enc(video),
            self.behav_enc(behavior),
            self.meta_enc(meta),
        ]
        features = [
            torch.zeros_like(x) if name in ablate_modalities else x for name, x in zip(MODALITIES, features)
        ]
        x = early_fusion(features)
        position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(
            torch.arange(0, x.shape[2], 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / x.shape[2])
        )
        positional = torch.zeros_like(x[0])
        positional[:, 0::2] = torch.sin(position * frequency)
        positional[:, 1::2] = torch.cos(position * frequency)
        x = x + positional
        maps = []
        for layer in self.layers:
            x, weights = layer(x, return_attention)
            if return_attention:
                maps.append(weights)
        predictions = self.output(x)
        return (predictions, maps) if return_attention else predictions
