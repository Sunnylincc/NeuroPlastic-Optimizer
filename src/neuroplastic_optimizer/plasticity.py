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
    hybrid_base: str = "classic"
    modulation_target: str = "gain"
    modulation_scope: str = "all"
    modulation_schedule: str = "constant"
    lr_controller_mode: str = "off"
    layer_scope: str = "all"
    phase_scope: str = "full"
    modulation_strength: float = 0.15
    modulation_max_ratio: float = 0.25
    controller_alpha: float = 0.1
    controller_low: float = 0.8
    controller_high: float = 1.2
    base_momentum: float = 0.9
    base_beta2: float = 0.999
    base_eps: float = 1e-8
    late_start_fraction: float = 0.6
    activity_weight: float = 0.4
    gradient_weight: float = 0.4
    memory_weight: float = 0.2
    plasticity_scale: float = 1.0
    warmup_epochs: int = 0
    bounded_residual: bool = False
    residual_max_ratio: float = 0.5
    orthogonal_residual: bool = False
    orthogonal_lambda: float = 0.1
    orthogonal_max_ratio: float = 0.5
    orthogonal_normalization: str = "raw"
    orthogonal_schedule: str = "constant"
    orthogonal_late_start_fraction: float = 0.3
    min_alpha: float = 0.2
    max_alpha: float = 2.0
    saturation_threshold_fraction: float = 0.05
    layerwise: bool = True
    parameterwise: bool = True
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.layer_scope == "all" and self.modulation_scope != "all":
            self.layer_scope = self.modulation_scope
        if self.phase_scope == "full" and self.modulation_schedule != "constant":
            self.phase_scope = {
                "warmup": "early",
                "late": "late",
            }.get(self.modulation_schedule, "full")


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
        if config.parameterwise:
            alpha = grad_signal
        else:
            alpha = _expand_scalar_like(grad_signal.mean(), grad_signal)
        return alpha.clamp(config.min_alpha, config.max_alpha)

    activity_signal = _standardize(activity_trace, config.eps)
    memory_signal = _standardize(momentum.abs() / (variance.sqrt() + config.eps), config.eps)

    alpha = (
        config.activity_weight * activity_signal
        + config.gradient_weight * grad_signal
        + config.memory_weight * memory_signal
    )

    # When parameterwise modulation is enabled, preserve the per-parameter signal.
    # Collapsing the mean of already standardized signals yields alpha ~= 1 and
    # makes the rule-based and grad-only paths nearly identical.
    if not config.parameterwise:
        alpha = _expand_scalar_like(alpha.mean(), alpha)

    return alpha.clamp(config.min_alpha, config.max_alpha)
