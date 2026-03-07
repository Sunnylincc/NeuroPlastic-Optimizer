from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HomeostaticConfig:
    max_update_norm: float = 1.0
    target_rms: float = 0.02
    adaptation_rate: float = 0.01
    eps: float = 1e-8


class HomeostaticStabilizer:
    """Constrains updates to maintain stable optimization dynamics."""

    def __init__(self, config: HomeostaticConfig | None = None):
        self.config = config or HomeostaticConfig()

    def stabilize(self, update: torch.Tensor) -> torch.Tensor:
        norm = update.norm()
        if norm > self.config.max_update_norm:
            update = update * (self.config.max_update_norm / (norm + self.config.eps))

        rms = torch.sqrt(torch.mean(update.pow(2)) + self.config.eps)
        gain = 1 + self.config.adaptation_rate * (self.config.target_rms - rms)
        return update * gain.clamp(0.5, 1.5)
