"""Evaluation helpers that restore the caller's per-module training modes."""

from contextlib import contextmanager

import torch

from dataloaders import prepare_batch


@contextmanager
def evaluation_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def evaluate(model, loader, device="cpu", **forward_kwargs):
    total, count = 0.0, 0
    with evaluation_mode(model), torch.no_grad():
        for batch in loader:
            inputs, target = prepare_batch(batch, device)
            predictions = model(**inputs, **forward_kwargs)
            total += (predictions - target).square().sum().item()
            count += target.numel()
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return total / count
