"""A learnable synthetic task; these arrays are not biological recordings."""

import numpy as np


def simulate_batch(cfg, batch_size, rng=None):
    """Return aligned float32 features/targets and int64 event IDs.

    Supply a NumPy Generator for reproducibility. The target combines all four
    modalities, a one-step neural lag, and independent observation noise.
    """
    rng = np.random.default_rng() if rng is None else rng
    data = cfg["data"]
    shape = (batch_size, data["seq_length"])
    neural = rng.standard_normal((*shape, data["num_neural_channels"])).astype("float32")
    video = rng.standard_normal((*shape, data["video_embed_dim"])).astype("float32")
    behavior = rng.integers(data["behavioral_vocab"], size=shape, dtype=np.int64)
    meta = rng.standard_normal((*shape, data["metadata_dim"])).astype("float32")
    event = 2.0 * behavior / (data["behavioral_vocab"] - 1) - 1.0
    lag = np.zeros(shape, dtype="float32")
    lag[:, 1:] = neural[:, :-1, 0]
    target = (
        0.9 * neural[..., 0]
        + 0.65 * video[..., 0]
        + 0.5 * event
        + 0.35 * meta[..., 0]
        + 0.25 * lag
        + data.get("noise_std", 0.05) * rng.standard_normal(shape)
    )
    return neural, video, behavior, meta, target[..., None].astype("float32")
