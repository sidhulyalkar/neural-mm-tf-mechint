from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StateSignal:
    """One interpretable coordinate of model state."""
    name: str
    value: float
    confidence: float = 1.0
    source: str = "probe"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class MechanisticState:
    layer: int
    position: int
    signals: dict[str, StateSignal]
    metadata: dict[str, Any] = field(default_factory=dict)

    def value(self, name: str, default: float | None = None) -> float:
        signal = self.signals.get(name)
        if signal is None:
            if default is None:
                raise KeyError(name)
            return default
        return signal.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "position": self.position,
            "signals": {name: asdict(signal) for name, signal in self.signals.items()},
            "metadata": self.metadata,
        }


@dataclass
class MechanisticTrajectory:
    """Layer-ordered state trajectory for one forward pass."""
    states: list[MechanisticState]
    prompt_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states.sort(key=lambda state: state.layer)

    def signal_curve(self, name: str) -> list[tuple[int, float]]:
        return [(state.layer, state.value(name)) for state in self.states if name in state.signals]

    def final(self) -> MechanisticState:
        if not self.states:
            raise ValueError("trajectory has no states")
        return self.states[-1]

    def peak(self, name: str) -> MechanisticState:
        candidates = [state for state in self.states if name in state.signals]
        if not candidates:
            raise KeyError(name)
        return max(candidates, key=lambda state: state.value(name))

    def transition_layer(self, lhs: str, rhs: str) -> int | None:
        """First layer at which lhs becomes smaller than rhs."""
        for state in self.states:
            if lhs in state.signals and rhs in state.signals and state.value(lhs) < state.value(rhs):
                return state.layer
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt_id, "metadata": self.metadata, "states": [s.to_dict() for s in self.states]}
