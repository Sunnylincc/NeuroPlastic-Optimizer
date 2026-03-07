from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParameterMemoryConfig:
    momentum_decay: float = 0.9
    variance_decay: float = 0.99


class ParameterStateMemory:
    """Tracks parameter history used by plasticity rules."""

    def __init__(self, config: ParameterMemoryConfig | None = None):
        self.config = config or ParameterMemoryConfig()

    def initialize(self, param: torch.Tensor) -> dict[str, torch.Tensor]:
        zeros = torch.zeros_like(param)
        return {
            "momentum": zeros.clone(),
            "variance": zeros.clone(),
            "activity_trace": zeros.clone(),
            "step": torch.zeros((), device=param.device),
        }

    def update_stats(self, state: dict[str, torch.Tensor], grad: torch.Tensor) -> None:
        beta1 = self.config.momentum_decay
        beta2 = self.config.variance_decay
        state["momentum"].mul_(beta1).add_(grad, alpha=1 - beta1)
        state["variance"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        state["step"] += 1
