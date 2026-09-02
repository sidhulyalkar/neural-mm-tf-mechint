"""Load and validate the public experiment configuration."""

import math
from pathlib import Path

import yaml


def validate_config(cfg):
    def positive_int(section, key, minimum=1):
        value = cfg[section][key] if section else cfg[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{section or 'config'}.{key} must be an integer >= {minimum}")

    positive_int(None, "seed", 0)
    for key in ("d_model", "n_heads", "num_layers"):
        positive_int("model", key)
    for key in ("batch_size", "epochs"):
        positive_int("train", key)
    for key in (
        "seq_length",
        "num_neural_channels",
        "video_embed_dim",
        "metadata_dim",
        "train_size",
        "val_size",
        "test_size",
    ):
        positive_int("data", key)
    positive_int("data", "behavioral_vocab", 2)
    if 4 * cfg["model"]["d_model"] % cfg["model"]["n_heads"]:
        raise ValueError("4 * d_model must be divisible by n_heads")
    if not 0 <= cfg["model"]["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    lr = cfg["train"]["lr"]
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    noise = cfg["data"].get("noise_std", 0.05)
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    return cfg


def load_config(path):
    with Path(path).open() as handle:
        return validate_config(yaml.safe_load(handle))
