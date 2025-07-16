# train.py

"""
This script trains a multimodal transformer model on simulated data.

Usage:
    python train.py --config configs/config.yaml

Configuration:
    - config (str): Path to the configuration file.

Example:
    python train.py --config configs/config.yaml
"""

import yaml
import torch
import torch.nn as nn
from dataloaders import get_loader
from model import MultimodalTransformer

def main():
    """
    Train the MultimodalTransformer model on simulated data.

    This function loads configuration settings, initializes data loaders,
    model, optimizer, and loss function, and iteratively trains the model
    for a specified number of epochs. During each epoch, it processes each
    batch of data, computes the loss, and updates the model parameters.
    The progress and average loss for each epoch are printed to the console.

    The training loop performs the following steps:
    1. Loads the configuration from a YAML file.
    2. Initializes the data loader and model, moving the model to the
       appropriate device (CPU or GPU).
    3. Sets up the optimizer and loss function.
    4. Iterates over the specified number of epochs, processing each batch
       of data, computing predictions and loss, performing backpropagation,
       and updating model weights.
    5. Prints the average loss for each epoch.

    Assumes the configuration file, data loaders, and model are set up
    correctly and available.
    """

    cfg = yaml.safe_load(open('configs/config.yaml'))
    loader = get_loader(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalTransformer(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    loss_fn = nn.MSELoss()

    for epoch in range(cfg['train']['epochs']):
        total_loss = 0.0
        count = 0
        for batch in loader:
            neural, video, behavior, meta, target = batch
            # Move to device and correct dtypes
            neural = neural.squeeze(1).to(device).float()
            video = video.squeeze(1).to(device).float()
            behavior = behavior.squeeze(1).to(device).long()
            meta = meta.squeeze(1).to(device).float()
            target = target.squeeze(1).to(device).float()

            optimizer.zero_grad()
            preds = model(neural, video, behavior, meta)
            loss = loss_fn(preds, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} - Loss: {total_loss/count:.4f}")

if __name__=='__main__':
    main()