from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from neuroplastic_optimizer.plasticity import PlasticityConfig, compute_plasticity
from neuroplastic_optimizer.stabilization import HomeostaticConfig, HomeostaticStabilizer
from neuroplastic_optimizer.state import ParameterStateMemory
from neuroplastic_optimizer.traces import ActivityTraceExtractor


class NeuroPlasticOptimizer(torch.optim.Optimizer):
    """PyTorch optimizer with synaptic-plasticity-inspired adaptive modulation."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        plasticity_config: PlasticityConfig | None = None,
        homeostatic_config: HomeostaticConfig | None = None,
    ) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.plasticity_config = plasticity_config or PlasticityConfig()
        self.state_memory = ParameterStateMemory()
        self.trace_extractor = ActivityTraceExtractor()
        self.stabilizer = HomeostaticStabilizer(homeostatic_config)
        self._diagnostic_bins = 64
        self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        self._diagnostics: dict[str, Any] = {
            "alpha_sum": 0.0,
            "alpha_count": 0,
            "alpha_min": float("inf"),
            "alpha_max": float("-inf"),
            "alpha_at_min_count": 0,
            "alpha_at_max_count": 0,
            "raw_gradient_norm_sq": 0.0,
            "raw_update_norm_sq": 0.0,
            "effective_update_norm_sq": 0.0,
            "alpha_histogram": torch.zeros(self._diagnostic_bins, dtype=torch.float64),
        }

    def collect_diagnostics(self) -> dict[str, float]:
        alpha_count = int(self._diagnostics["alpha_count"])
        raw_gradient_norm = float(self._diagnostics["raw_gradient_norm_sq"]) ** 0.5
        raw_update_norm = float(self._diagnostics["raw_update_norm_sq"]) ** 0.5
        effective_update_norm = float(self._diagnostics["effective_update_norm_sq"]) ** 0.5

        diagnostics = {
            "alpha_mean": 0.0,
            "alpha_median": 0.0,
            "alpha_min": 0.0,
            "alpha_max": 0.0,
            "alpha_fraction_at_min": 0.0,
            "alpha_fraction_at_max": 0.0,
            "raw_gradient_norm": raw_gradient_norm,
            "raw_update_norm": raw_update_norm,
            "effective_update_norm": effective_update_norm,
            "effective_to_gradient_norm_ratio": 0.0,
            "stabilization_norm_ratio": 0.0,
        }
        if alpha_count == 0:
            return diagnostics

        histogram = self._diagnostics["alpha_histogram"]
        cumulative = torch.cumsum(histogram, dim=0)
        median_threshold = alpha_count / 2
        median_index = int(
            torch.searchsorted(
                cumulative, torch.tensor(median_threshold, dtype=torch.float64)
            ).item()
        )
        bin_width = (
            self.plasticity_config.max_alpha - self.plasticity_config.min_alpha
        ) / self._diagnostic_bins
        alpha_median = self.plasticity_config.min_alpha + (median_index + 0.5) * bin_width

        diagnostics.update(
            {
                "alpha_mean": float(self._diagnostics["alpha_sum"]) / alpha_count,
                "alpha_median": alpha_median,
                "alpha_min": float(self._diagnostics["alpha_min"]),
                "alpha_max": float(self._diagnostics["alpha_max"]),
                "alpha_fraction_at_min": float(self._diagnostics["alpha_at_min_count"])
                / alpha_count,
                "alpha_fraction_at_max": float(self._diagnostics["alpha_at_max_count"])
                / alpha_count,
                "effective_to_gradient_norm_ratio": effective_update_norm
                / max(raw_gradient_norm, 1e-12),
                "stabilization_norm_ratio": effective_update_norm / max(raw_update_norm, 1e-12),
            }
        )
        return diagnostics

    @torch.no_grad()
    def step(self, closure: Any = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state.update(self.state_memory.initialize(p))

                state["activity_trace"] = self.trace_extractor.update(state["activity_trace"], grad)
                self.state_memory.update_stats(state, grad)

                alpha = compute_plasticity(
                    grad=grad,
                    activity_trace=state["activity_trace"],
                    momentum=state["momentum"],
                    variance=state["variance"],
                    config=self.plasticity_config,
                )

                update = alpha * grad
                if wd > 0:
                    update = update + wd * p

                stabilized = self.stabilizer.stabilize(update)

                alpha_detached = alpha.detach()
                grad_detached = grad.detach()
                update_detached = update.detach()
                stabilized_detached = stabilized.detach()
                alpha_count = alpha_detached.numel()

                self._diagnostics["alpha_sum"] += float(alpha_detached.sum().item())
                self._diagnostics["alpha_count"] += alpha_count
                self._diagnostics["alpha_min"] = min(
                    float(self._diagnostics["alpha_min"]),
                    float(alpha_detached.min().item()),
                )
                self._diagnostics["alpha_max"] = max(
                    float(self._diagnostics["alpha_max"]),
                    float(alpha_detached.max().item()),
                )
                self._diagnostics["alpha_at_min_count"] += int(
                    torch.isclose(
                        alpha_detached,
                        torch.full_like(alpha_detached, self.plasticity_config.min_alpha),
                        atol=1e-6,
                    )
                    .sum()
                    .item()
                )
                self._diagnostics["alpha_at_max_count"] += int(
                    torch.isclose(
                        alpha_detached,
                        torch.full_like(alpha_detached, self.plasticity_config.max_alpha),
                        atol=1e-6,
                    )
                    .sum()
                    .item()
                )
                self._diagnostics["raw_gradient_norm_sq"] += float(
                    grad_detached.pow(2).sum().item()
                )
                self._diagnostics["raw_update_norm_sq"] += float(
                    update_detached.pow(2).sum().item()
                )
                self._diagnostics["effective_update_norm_sq"] += float(
                    stabilized_detached.pow(2).sum().item()
                )
                self._diagnostics["alpha_histogram"] += torch.histc(
                    alpha_detached.float().cpu(),
                    bins=self._diagnostic_bins,
                    min=self.plasticity_config.min_alpha,
                    max=self.plasticity_config.max_alpha,
                ).to(dtype=torch.float64)
                p.add_(stabilized, alpha=-lr)

        return loss
