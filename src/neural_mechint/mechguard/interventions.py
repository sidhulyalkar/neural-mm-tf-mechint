from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SteeringPlan:
    layer: int
    direction: torch.Tensor
    coefficient: float
    position: int = -1
    label: str = "mechanistic-steering"

    def public_dict(self) -> dict[str, object]:
        return {"layer": self.layer, "coefficient": self.coefficient, "position": self.position, "label": self.label, "direction_norm": float(self.direction.detach().float().norm())}


def apply_steering(adapter, prompt: str, plan: SteeringPlan) -> torch.Tensor:
    return adapter.forward_with_steering(prompt, layer=plan.layer, direction=plan.direction, coefficient=plan.coefficient, position=plan.position)
