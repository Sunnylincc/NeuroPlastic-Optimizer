from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ActivityTraceConfig:
    decay: float = 0.95
    eps: float = 1e-8


class ActivityTraceExtractor:
    """Maintains local activity proxy from gradient statistics."""

    def __init__(self, config: ActivityTraceConfig | None = None):
        self.config = config or ActivityTraceConfig()

    def update(self, trace: torch.Tensor | None, grad: torch.Tensor) -> torch.Tensor:
        activity = grad.detach().abs()
        if trace is None:
            return activity
        return self.config.decay * trace + (1 - self.config.decay) * activity

    def normalized(self, trace: torch.Tensor) -> torch.Tensor:
        return trace / (trace.mean() + self.config.eps)
