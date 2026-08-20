from __future__ import annotations

from dataclasses import dataclass

import torch

from neural_mechint.types import ActivationCache

from .state import MechanisticState, MechanisticTrajectory, StateSignal


@dataclass(frozen=True)
class LinearSignalProbe:
    """A cheap deployment probe over the residual stream."""
    name: str
    direction: torch.Tensor
    bias: float = 0.0
    temperature: float = 1.0
    source: str = "linear-probe"

    def __post_init__(self) -> None:
        if self.direction.ndim != 1:
            raise ValueError("direction must be one-dimensional")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def score(self, activation: torch.Tensor) -> float:
        vector = activation.detach().float().cpu()
        if vector.ndim != 1 or vector.shape[0] != self.direction.shape[0]:
            raise ValueError("activation must be a 1D vector matching the probe direction")
        logit = torch.dot(vector, self.direction.detach().float().cpu()) + self.bias
        return float(torch.sigmoid(logit / self.temperature))


class StateEstimator:
    """Convert cached residual activations into an interpretable state trajectory."""
    def __init__(self, probes: list[LinearSignalProbe], *, position: int = -1):
        names = [probe.name for probe in probes]
        if len(names) != len(set(names)):
            raise ValueError("probe names must be unique")
        self.probes = probes
        self.position = position

    @staticmethod
    def _position_vector(tensor: torch.Tensor, position: int) -> torch.Tensor:
        if tensor.ndim == 3:
            return tensor[0, position, :]
        if tensor.ndim == 2:
            return tensor[position, :]
        if tensor.ndim == 1:
            return tensor
        raise ValueError("activation must have shape [B,T,D], [T,D], or [D]")

    def estimate(self, cache: ActivationCache, *, prompt_id: str | None = None) -> MechanisticTrajectory:
        states = []
        for layer in cache.layers():
            vector = self._position_vector(cache[layer], self.position)
            signals = {probe.name: StateSignal(probe.name, probe.score(vector), source=probe.source) for probe in self.probes}
            states.append(MechanisticState(layer=layer, position=self.position, signals=signals))
        return MechanisticTrajectory(states=states, prompt_id=prompt_id, metadata=dict(cache.metadata))
