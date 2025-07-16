# dataloaders.py
"""
This module provides functions to create DataLoader objects for the MultimodalDataset.

Classes:
    MultimodalDataset: A custom Dataset class for the MultimodalDataset.

Functions:
    get_loader(cfg): Creates a DataLoader object from a MultimodalDataset.
"""

from torch.utils.data import Dataset, DataLoader
import data_simulation as sim

class MultimodalDataset(Dataset):
    def __init__(self, cfg, size=1000):
        """
        Initialize a MultimodalDataset object.

        Args:
            cfg (dict): Configuration dictionary
            size (int): The size of the dataset. Defaults to 1000.
        """
        self.cfg = cfg
        self.size = size
    def __len__(self): return self.size
    def __getitem__(self, idx):
        return sim.simulate_batch(self.cfg, 1)

def get_loader(cfg):
    """
    Create a DataLoader from a MultimodalDataset.

    Args:
        cfg (dict): Configuration dictionary

    Returns:
        DataLoader: The DataLoader object.
    """
    ds = MultimodalDataset(cfg)
    return DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=True)