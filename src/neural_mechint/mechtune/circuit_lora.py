from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerEffect:
    layer: int
    causal_effect: float
    collateral_kl: float = 0.0


@dataclass(frozen=True)
class CircuitLoRAPlan:
    target_layers: tuple[int, ...]
    rank: int
    alpha: int
    rationale: str


def plan_circuit_lora(effects: list[LayerEffect], *, top_k: int = 4, max_collateral_kl: float | None = None, rank: int = 8, alpha: int = 16) -> CircuitLoRAPlan:
    """Choose LoRA target layers from causal effect rather than architectural convention."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidates = effects
    if max_collateral_kl is not None:
        candidates = [effect for effect in candidates if effect.collateral_kl <= max_collateral_kl]
    ranked = sorted(candidates, key=lambda effect: abs(effect.causal_effect), reverse=True)
    selected = tuple(sorted(effect.layer for effect in ranked[:top_k]))
    if not selected:
        raise ValueError("no layers satisfy the selection criteria")
    rationale = f"Selected {len(selected)} layers with largest absolute causal effect"
    if max_collateral_kl is not None:
        rationale += f" subject to collateral_kl <= {max_collateral_kl}"
    return CircuitLoRAPlan(selected, rank=rank, alpha=alpha, rationale=rationale)
