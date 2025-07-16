# data_simulation.py
"""
This module provides functions to simulate data for training and evaluation.

Functions:
    simulate_batch(cfg, batch_size): Simulates a batch of data for training or evaluation.
"""


import numpy as np

def simulate_batch(cfg, batch_size):
    """
    Simulate a batch of data for training or evaluation.

    The simulated data will be structured as follows:

    - neural: [batch, seq, channels]
    - video: [batch, seq, embed]
    - behavior: [batch, seq]
    - metadata: [batch, seq, metadata_dim]
    - target: [batch, seq, 1]

    Args:
        cfg (dict): Configuration dictionary
        batch_size (int): The size of the batch to simulate

    Returns:
        tuple of 5 arrays: (neural, video, behavior, metadata, target)
    """
    seq = cfg['data']['seq_length']
    # neural: [batch, seq, channels]
    neural = np.random.randn(batch_size, seq, cfg['data']['num_neural_channels'])
    # video: [batch, seq, embed]
    video = np.random.randn(batch_size, seq, cfg['data']['video_embed_dim'])
    # behavior: [batch, seq]
    behavior = np.random.randint(0, cfg['data']['behavioral_vocab'], size=(batch_size, seq))
    # metadata: [batch, seq, metadata_dim]
    meta = np.random.randn(batch_size, seq, cfg['data']['metadata_dim'])
    # target regression as example
    target = np.random.randn(batch_size, seq, 1)
    return neural, video, behavior, meta, target