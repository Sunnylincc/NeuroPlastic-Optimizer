from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class PlasticityMode(str, Enum):
    RULE_BASED = "rule_based"
    ABLATION_GRAD_ONLY = "ablation_grad_only"


@dataclass(slots=True)
class PlasticityConfig:
    mode: PlasticityMode = PlasticityMode.RULE_BASED
    activity_weight: float = 0.4
    gradient_weight: float = 0.4
    memory_weight: float = 0.2
    min_alpha: float = 0.2
    max_alpha: float = 2.0
    layerwise: bool = True
    parameterwise: bool = True
    eps: float = 1e-8


def _standardize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / (x.mean() + eps)


def _expand_scalar_like(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return torch.full_like(ref, fill_value=float(value.item()))


def compute_plasticity(
    grad: torch.Tensor,
    activity_trace: torch.Tensor,
    momentum: torch.Tensor,
    variance: torch.Tensor,
    config: PlasticityConfig,
) -> torch.Tensor:
    grad_signal = _standardize(grad.abs(), config.eps)

    if config.mode is PlasticityMode.ABLATION_GRAD_ONLY:
        if config.layerwise or not config.parameterwise:
            alpha = _expand_scalar_like(grad_signal.mean(), grad_signal)
        else:
            alpha = grad_signal
        return alpha.clamp(config.min_alpha, config.max_alpha)

    activity_signal = _standardize(activity_trace, config.eps)
    memory_signal = _standardize(momentum.abs() / (variance.sqrt() + config.eps), config.eps)

    if config.layerwise:
        fused_scalar = (
            config.activity_weight * activity_signal.mean()
            + config.gradient_weight * grad_signal.mean()
            + config.memory_weight * memory_signal.mean()
        )
        alpha = _expand_scalar_like(fused_scalar, grad_signal)
    else:
        alpha = (
            config.activity_weight * activity_signal
            + config.gradient_weight * grad_signal
            + config.memory_weight * memory_signal
        )

    if not config.parameterwise:
        alpha = _expand_scalar_like(alpha.mean(), alpha)

    return alpha.clamp(config.min_alpha, config.max_alpha)
