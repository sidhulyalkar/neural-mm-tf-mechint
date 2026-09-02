"""Stable per-example simulation and independently seeded train/validation/test sets."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_simulation import simulate_batch

SPLITS = {"train": 0, "val": 1, "test": 2}
INPUT_NAMES = ("neural", "video", "behavior", "meta")


class MultimodalDataset(Dataset):
    def __init__(self, cfg, size=None, split="train"):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.cfg = cfg
        self.split = split
        self.size = cfg["data"][f"{split}_size"] if size is None else size
        if self.size < 1:
            raise ValueError("Dataset size must be positive")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not 0 <= idx < self.size:
            raise IndexError(idx)
        seed = np.random.SeedSequence([self.cfg["seed"], SPLITS[self.split], idx])
        arrays = simulate_batch(self.cfg, 1, np.random.default_rng(seed))
        return tuple(torch.from_numpy(x[0]) for x in arrays)


def get_loader(cfg, split="train", shuffle=None):
    shuffle = split == "train" if shuffle is None else shuffle
    generator = torch.Generator().manual_seed(cfg["seed"] + SPLITS[split])
    return DataLoader(
        MultimodalDataset(cfg, split=split),
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def prepare_batch(batch, device="cpu"):
    """Separate model inputs from the regression target at the pipeline boundary."""
    tensors = [x.to(device) for x in batch]
    return dict(zip(INPUT_NAMES, tensors[:4])), tensors[4]
