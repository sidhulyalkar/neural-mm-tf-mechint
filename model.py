# model.py
"""
This module defines the MultimodalTransformer model for multimodal data fusion.

Classes:
    MultimodalTransformer: A PyTorch module that represents the MultimodalTransformer model.
    EarlyFusion: A class for early fusion of input features.

Functions:
    early_fusion: Concatenates a list of features along the last dimension.
    forward: Performs a forward pass through the MultimodalTransformer model.

Imports:
    torch.nn: Module for defining neural network layers.
    encoders.neural_encoder: Module for encoding neural signals.
    encoders.video_encoder: Module for encoding video frames.
    encoders.behavior_encoder: Module for encoding behavioral events.
    encoders.metadata_encoder: Module for encoding metadata.
    fusion: Module for early fusion of input features.
"""

import torch.nn as nn
from encoders.neural_encoder import NeuralEncoder
from encoders.video_encoder import VideoEncoder
from encoders.behavior_encoder import BehaviorEncoder
from encoders.metadata_encoder import MetadataEncoder
from fusion import early_fusion

class MultimodalTransformer(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the MultimodalTransformer model with specified configurations.

        This constructor sets up encoders for different modalities (neural signals, video frames,
        behavioral events, and metadata) using the provided configuration. It also initializes a
        transformer encoder and an output layer to produce the final model output.

        Args:
            cfg (dict): Configuration dictionary containing model parameters such as `d_model`,
                        `n_heads`, `num_layers`, and `dropout`, as well as data parameters for each
                        modality, including `num_neural_channels`, `video_embed_dim`, `behavioral_vocab`,
                        and `metadata_dim`.

        """

        super().__init__()
        d = cfg['model']['d_model']
        # encoders
        self.neural_enc = NeuralEncoder(cfg['data']['num_neural_channels'], d)
        self.video_enc = VideoEncoder(cfg['data']['video_embed_dim'], d)
        self.behav_enc = BehaviorEncoder(cfg['data']['behavioral_vocab'], d)
        self.meta_enc = MetadataEncoder(cfg['data']['metadata_dim'], d)
        # fusion & transformer
        ff_dim = d*4
        encoder_layer = nn.TransformerEncoderLayer(d*4, cfg['model']['n_heads'], dim_feedforward=ff_dim, dropout=cfg['model']['dropout'])
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg['model']['num_layers'])
        self.output = nn.Linear(d*4, 1)

    def forward(self, neural, video, behavior, meta):
        """
        Perform a forward pass through the model.

        This function takes in four inputs for the different modalities and passes them
        through the respective encoders to obtain a set of features. These features are
        then fused together using early fusion and passed through a transformer encoder
        to obtain a set of contextualized features. Finally, the output layer is used
        to produce the final output of the model.

        Args:
            neural (torch.Tensor): The input neural signal data of shape [B, T, C].
            video (torch.Tensor): The input video data of shape [B, T, embed].
            behavior (torch.Tensor): The input behavioral event sequence of shape [B, T].
            meta (torch.Tensor): The input metadata of shape [B, T, metadata_dim].

        Returns:
            torch.Tensor: The output of the model of shape [B, T, 1].
        """
        f1 = self.neural_enc(neural)
        f2 = self.video_enc(video)
        f3 = self.behav_enc(behavior)
        f4 = self.meta_enc(meta)
        x = early_fusion([f1,f2,f3,f4])  # [B, T, d*4]
        x = x.permute(1,0,2)  # -> [T, B, d*4]
        out = self.transformer(x)
        out = out.permute(1,0,2)  # -> [B, T, d*4]
        return self.output(out)