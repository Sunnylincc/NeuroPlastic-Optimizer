from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import math
from typing import Any

import torch

from neuroplastic_optimizer.plasticity import (
    PlasticityConfig,
    PlasticityMode,
    compute_plasticity,
)
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
        *,
        decoupled_weight_decay: bool = False,
    ) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.plasticity_config = plasticity_config or PlasticityConfig()
        self.decoupled_weight_decay = decoupled_weight_decay
        self._grad_only_config = replace(
            self.plasticity_config,
            mode=PlasticityMode.ABLATION_GRAD_ONLY,
        )
        self.state_memory = ParameterStateMemory()
        self.trace_extractor = ActivityTraceExtractor()
        self.stabilizer = HomeostaticStabilizer(homeostatic_config)
        self._layer_target_rms_scales = self._build_layer_target_rms_scales()
        self._scope_lookups = self._build_scope_lookups()
        self._diagnostic_bins = 64
        self._current_epoch = 1
        self._total_epochs = 1
        self.reset_diagnostics()

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = max(1, int(epoch))

    def set_total_epochs(self, total_epochs: int) -> None:
        self._total_epochs = max(1, int(total_epochs))

    def _plasticity_warmup_gate(self) -> float:
        warmup_epochs = self.plasticity_config.warmup_epochs
        if self.plasticity_config.mode is PlasticityMode.ABLATION_GRAD_ONLY:
            return 0.0
        if warmup_epochs <= 0:
            return 1.0
        return min(max((self._current_epoch - 1) / warmup_epochs, 0.0), 1.0)

    def reset_diagnostics(self) -> None:
        alpha_range = self.plasticity_config.max_alpha - self.plasticity_config.min_alpha
        saturation_margin = alpha_range * self.plasticity_config.saturation_threshold_fraction
        self._diagnostics: dict[str, Any] = {
            "alpha_sum": 0.0,
            "alpha_sum_sq": 0.0,
            "alpha_count": 0,
            "alpha_min": float("inf"),
            "alpha_max": float("-inf"),
            "alpha_near_min_count": 0,
            "alpha_near_max_count": 0,
            "raw_gradient_norm_sq": 0.0,
            "parameter_norm_sq": 0.0,
            "raw_update_norm_sq": 0.0,
            "effective_update_norm_sq": 0.0,
            "plasticity_delta_norm_sq": 0.0,
            "orthogonal_signal_norm_sq": 0.0,
            "orthogonal_residual_norm_sq": 0.0,
            "gradient_plasticity_dot": 0.0,
            "gradient_orthogonal_dot": 0.0,
            "orthogonal_skip_count": 0,
            "orthogonal_active_count": 0,
            "weight_decay_term_norm_sq": 0.0,
            "alpha_histogram": torch.zeros(self._diagnostic_bins, dtype=torch.float64),
            "alpha_near_min_threshold": self.plasticity_config.min_alpha + saturation_margin,
            "alpha_near_max_threshold": self.plasticity_config.max_alpha - saturation_margin,
            "modulation_param_count": 0,
            "modulation_active_param_count": 0,
            "modulation_delta_norm_sq_conv": 0.0,
            "modulation_delta_norm_sq_classifier": 0.0,
            "modulation_delta_norm_sq_early_blocks": 0.0,
            "modulation_delta_norm_sq_late_blocks": 0.0,
            "modulation_param_count_conv": 0,
            "modulation_param_count_classifier": 0,
            "modulation_param_count_early_blocks": 0,
            "modulation_param_count_late_blocks": 0,
            "modulation_active_param_count_conv": 0,
            "modulation_active_param_count_classifier": 0,
            "modulation_active_param_count_early_blocks": 0,
            "modulation_active_param_count_late_blocks": 0,
            "controller_signal_sum": 0.0,
            "controller_signal_count": 0,
            "controller_signal_min": float("inf"),
            "controller_signal_max": float("-inf"),
            "controller_multiplier_sum": 0.0,
            "controller_multiplier_count": 0,
            "controller_multiplier_min": float("inf"),
            "controller_multiplier_max": float("-inf"),
            "controller_clamp_low_count": 0,
            "controller_clamp_high_count": 0,
            "controller_group_count_stem": 0,
            "controller_group_count_classifier": 0,
            "controller_group_count_early_blocks": 0,
            "controller_group_count_late_blocks": 0,
            "controller_signal_sum_stem": 0.0,
            "controller_signal_sum_classifier": 0.0,
            "controller_signal_sum_early_blocks": 0.0,
            "controller_signal_sum_late_blocks": 0.0,
            "controller_multiplier_sum_stem": 0.0,
            "controller_multiplier_sum_classifier": 0.0,
            "controller_multiplier_sum_early_blocks": 0.0,
            "controller_multiplier_sum_late_blocks": 0.0,
        }

    def _build_layer_target_rms_scales(self) -> dict[int, float]:
        params = [p for group in self.param_groups for p in group["params"]]
        if not params:
            return {}
        total = len(params)
        early_cutoff = math.ceil(total / 3)
        middle_cutoff = math.ceil((2 * total) / 3)
        scales: dict[int, float] = {}
        for index, param in enumerate(params):
            if index < early_cutoff:
                scale = self.stabilizer.config.early_target_rms_scale
            elif index < middle_cutoff:
                scale = self.stabilizer.config.middle_target_rms_scale
            else:
                scale = self.stabilizer.config.late_target_rms_scale
            scales[id(param)] = scale
        return scales

    def _build_scope_lookups(self) -> dict[str, dict[int, bool]]:
        params = [p for group in self.param_groups for p in group["params"]]
        if not params:
            return {}
        total = len(params)
        classifier_start = max(total - 2, 0)
        body_total = classifier_start
        early_cutoff = math.ceil(body_total / 2) if body_total > 0 else 0
        late_start = math.floor(body_total / 2) if body_total > 0 else 0
        lookups: dict[str, dict[int, bool]] = {
            "all": {},
            "conv_only": {},
            "classifier_only": {},
            "early_blocks": {},
            "late_blocks": {},
        }
        for index, param in enumerate(params):
            identifier = id(param)
            lookups["all"][identifier] = True
            lookups["conv_only"][identifier] = param.ndim == 4
            lookups["classifier_only"][identifier] = index >= classifier_start
            lookups["early_blocks"][identifier] = index < early_cutoff
            lookups["late_blocks"][identifier] = late_start <= index < classifier_start
        return lookups

    def _phase_gate(self) -> float:
        scope = self.plasticity_config.phase_scope
        if scope == "full":
            return 1.0
        progress = self._current_epoch / max(self._total_epochs, 1)
        if scope == "early":
            return 1.0 if progress <= 0.3 else 0.0
        if scope == "middle":
            return 1.0 if 0.3 < progress < 0.7 else 0.0
        return 1.0 if progress >= 0.7 else 0.0

    def _orthogonal_schedule_gate(self) -> float:
        schedule = self.plasticity_config.orthogonal_schedule
        if schedule == "constant":
            return 1.0
        progress = self._current_epoch / max(self._total_epochs, 1)
        if schedule == "late":
            late_start = self.plasticity_config.orthogonal_late_start_fraction
            if progress <= late_start:
                return 0.0
            return min((progress - late_start) / max(1.0 - late_start, 1e-12), 1.0)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 - math.cos(math.pi * progress))

    def _layer_scope_gate(self, param: torch.nn.Parameter) -> float:
        scope = self.plasticity_config.layer_scope
        if scope == "head_only":
            scope = "classifier_only"
        elif scope == "late":
            scope = "late_blocks"
        lookup = self._scope_lookups.get(scope, self._scope_lookups.get("all", {}))
        return 1.0 if lookup.get(id(param), False) else 0.0

    def _layer_group_flags(self, param: torch.nn.Parameter) -> dict[str, bool]:
        identifier = id(param)
        return {
            "conv": self._scope_lookups.get("conv_only", {}).get(identifier, False),
            "classifier": self._scope_lookups.get("classifier_only", {}).get(identifier, False),
            "early_blocks": self._scope_lookups.get("early_blocks", {}).get(identifier, False),
            "late_blocks": self._scope_lookups.get("late_blocks", {}).get(identifier, False),
        }

    def _controller_group_name(self, param: torch.nn.Parameter) -> str:
        flags = self._layer_group_flags(param)
        if flags["classifier"]:
            return "classifier"
        if flags["early_blocks"]:
            return "early_blocks"
        if flags["late_blocks"]:
            return "late_blocks"
        return "stem"

    def _extract_controller_signal(
        self,
        alpha: torch.Tensor,
        grad_only_alpha: torch.Tensor,
    ) -> float:
        if self.plasticity_config.mode is PlasticityMode.ABLATION_GRAD_ONLY:
            return 0.0
        signal = torch.nan_to_num(
            torch.log(
                alpha.detach().clamp_min(self.plasticity_config.eps)
                / grad_only_alpha.detach().clamp_min(self.plasticity_config.eps)
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if signal.numel() == 0:
            return 0.0
        value = float(signal.mean().item())
        if not math.isfinite(value):
            return 0.0
        return value

    def _resolve_controller_multipliers(
        self,
        contexts: list[dict[str, Any]],
    ) -> dict[int, tuple[float, float]]:
        mode = self.plasticity_config.lr_controller_mode
        if mode == "off" or not contexts:
            return {id(ctx["param"]): (0.0, 1.0) for ctx in contexts}

        controller_alpha = self.plasticity_config.controller_alpha
        low = self.plasticity_config.controller_low
        high = self.plasticity_config.controller_high

        def to_multiplier(signal: float) -> float:
            if not math.isfinite(signal):
                return 1.0
            return min(max(1.0 + controller_alpha * signal, low), high)

        if mode == "global":
            signal = sum(ctx["controller_signal"] for ctx in contexts) / max(len(contexts), 1)
            multiplier = to_multiplier(signal)
            return {id(ctx["param"]): (signal, multiplier) for ctx in contexts}

        groups: dict[str, list[dict[str, Any]]] = {}
        for ctx in contexts:
            groups.setdefault(str(ctx["controller_group"]), []).append(ctx)

        resolved: dict[int, tuple[float, float]] = {}
        for group_name, group_contexts in groups.items():
            if mode == "classifier_only" and group_name != "classifier":
                for ctx in group_contexts:
                    resolved[id(ctx["param"])] = (0.0, 1.0)
                continue
            signal = sum(ctx["controller_signal"] for ctx in group_contexts) / max(
                len(group_contexts), 1
            )
            multiplier = to_multiplier(signal)
            for ctx in group_contexts:
                resolved[id(ctx["param"])] = (signal, multiplier)
        return resolved

    def collect_diagnostics(self) -> dict[str, float]:
        alpha_count = int(self._diagnostics["alpha_count"])
        raw_gradient_norm = float(self._diagnostics["raw_gradient_norm_sq"]) ** 0.5
        raw_update_norm = float(self._diagnostics["raw_update_norm_sq"]) ** 0.5
        effective_update_norm = float(self._diagnostics["effective_update_norm_sq"]) ** 0.5

        diagnostics = {
            "alpha_mean": 0.0,
            "alpha_std": 0.0,
            "alpha_median": 0.0,
            "alpha_min": 0.0,
            "alpha_max": 0.0,
            "alpha_fraction_near_min": 0.0,
            "alpha_fraction_near_max": 0.0,
            "raw_gradient_norm": raw_gradient_norm,
            "parameter_norm": 0.0,
            "raw_update_norm": raw_update_norm,
            "effective_update_norm": effective_update_norm,
            "plasticity_delta_norm": 0.0,
            "orthogonal_signal_norm": 0.0,
            "orthogonal_residual_norm": 0.0,
            "cosine_similarity_grad_plasticity": 0.0,
            "cosine_similarity_grad_orthogonal": 0.0,
            "orthogonal_skip_fraction": 0.0,
            "weight_decay_term_norm": 0.0,
            "effective_to_gradient_norm_ratio": 0.0,
            "stabilization_norm_ratio": 0.0,
            "modulation_active_fraction": 0.0,
            "modulation_magnitude_conv": 0.0,
            "modulation_magnitude_classifier": 0.0,
            "modulation_magnitude_early_blocks": 0.0,
            "modulation_magnitude_late_blocks": 0.0,
            "modulation_active_fraction_conv": 0.0,
            "modulation_active_fraction_classifier": 0.0,
            "modulation_active_fraction_early_blocks": 0.0,
            "modulation_active_fraction_late_blocks": 0.0,
            "controller_signal_mean": 0.0,
            "controller_signal_min": 0.0,
            "controller_signal_max": 0.0,
            "controller_multiplier_mean": 1.0,
            "controller_multiplier_min": 1.0,
            "controller_multiplier_max": 1.0,
            "controller_clamp_fraction_low": 0.0,
            "controller_clamp_fraction_high": 0.0,
            "controller_signal_mean_stem": 0.0,
            "controller_signal_mean_classifier": 0.0,
            "controller_signal_mean_early_blocks": 0.0,
            "controller_signal_mean_late_blocks": 0.0,
            "controller_multiplier_mean_stem": 1.0,
            "controller_multiplier_mean_classifier": 1.0,
            "controller_multiplier_mean_early_blocks": 1.0,
            "controller_multiplier_mean_late_blocks": 1.0,
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
        alpha_mean = float(self._diagnostics["alpha_sum"]) / alpha_count
        alpha_mean_sq = float(self._diagnostics["alpha_sum_sq"]) / alpha_count
        alpha_variance = max(alpha_mean_sq - alpha_mean**2, 0.0)

        diagnostics.update(
            {
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_variance**0.5,
                "alpha_median": alpha_median,
                "alpha_min": float(self._diagnostics["alpha_min"]),
                "alpha_max": float(self._diagnostics["alpha_max"]),
                "alpha_fraction_near_min": float(self._diagnostics["alpha_near_min_count"])
                / alpha_count,
                "alpha_fraction_near_max": float(self._diagnostics["alpha_near_max_count"])
                / alpha_count,
                "parameter_norm": float(self._diagnostics["parameter_norm_sq"]) ** 0.5,
                "plasticity_delta_norm": float(self._diagnostics["plasticity_delta_norm_sq"])
                ** 0.5,
                "orthogonal_signal_norm": float(self._diagnostics["orthogonal_signal_norm_sq"])
                ** 0.5,
                "orthogonal_residual_norm": float(self._diagnostics["orthogonal_residual_norm_sq"])
                ** 0.5,
                "weight_decay_term_norm": float(self._diagnostics["weight_decay_term_norm_sq"])
                ** 0.5,
                "effective_to_gradient_norm_ratio": effective_update_norm
                / max(raw_gradient_norm, 1e-12),
                "stabilization_norm_ratio": effective_update_norm / max(raw_update_norm, 1e-12),
            }
        )
        orthogonal_signal_norm = float(diagnostics["orthogonal_signal_norm"])
        orthogonal_residual_norm = float(diagnostics["orthogonal_residual_norm"])
        diagnostics["cosine_similarity_grad_plasticity"] = float(
            self._diagnostics["gradient_plasticity_dot"]
        ) / max(raw_gradient_norm * orthogonal_signal_norm, 1e-12)
        diagnostics["cosine_similarity_grad_orthogonal"] = float(
            self._diagnostics["gradient_orthogonal_dot"]
        ) / max(raw_gradient_norm * orthogonal_residual_norm, 1e-12)
        active_count = int(self._diagnostics["orthogonal_active_count"])
        if active_count > 0:
            diagnostics["orthogonal_skip_fraction"] = float(
                self._diagnostics["orthogonal_skip_count"]
            ) / active_count
        modulation_param_count = int(self._diagnostics["modulation_param_count"])
        if modulation_param_count > 0:
            diagnostics["modulation_active_fraction"] = float(
                self._diagnostics["modulation_active_param_count"]
            ) / modulation_param_count
        controller_count = int(self._diagnostics["controller_signal_count"])
        if controller_count > 0:
            diagnostics["controller_signal_mean"] = float(
                self._diagnostics["controller_signal_sum"]
            ) / controller_count
            diagnostics["controller_signal_min"] = float(self._diagnostics["controller_signal_min"])
            diagnostics["controller_signal_max"] = float(self._diagnostics["controller_signal_max"])
            diagnostics["controller_multiplier_mean"] = float(
                self._diagnostics["controller_multiplier_sum"]
            ) / controller_count
            diagnostics["controller_multiplier_min"] = float(
                self._diagnostics["controller_multiplier_min"]
            )
            diagnostics["controller_multiplier_max"] = float(
                self._diagnostics["controller_multiplier_max"]
            )
            diagnostics["controller_clamp_fraction_low"] = float(
                self._diagnostics["controller_clamp_low_count"]
            ) / controller_count
            diagnostics["controller_clamp_fraction_high"] = float(
                self._diagnostics["controller_clamp_high_count"]
            ) / controller_count
        for group in ("conv", "classifier", "early_blocks", "late_blocks"):
            diagnostics[f"modulation_magnitude_{group}"] = float(
                self._diagnostics[f"modulation_delta_norm_sq_{group}"]
            ) ** 0.5
            group_param_count = int(self._diagnostics[f"modulation_param_count_{group}"])
            if group_param_count > 0:
                diagnostics[f"modulation_active_fraction_{group}"] = float(
                    self._diagnostics[f"modulation_active_param_count_{group}"]
                ) / group_param_count
        for group in ("stem", "classifier", "early_blocks", "late_blocks"):
            group_count = int(self._diagnostics[f"controller_group_count_{group}"])
            if group_count > 0:
                diagnostics[f"controller_signal_mean_{group}"] = float(
                    self._diagnostics[f"controller_signal_sum_{group}"]
                ) / group_count
                diagnostics[f"controller_multiplier_mean_{group}"] = float(
                    self._diagnostics[f"controller_multiplier_sum_{group}"]
                ) / group_count
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
            contexts: list[dict[str, Any]] = []
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state.update(self.state_memory.initialize(p))
                    state["base_momentum"] = torch.zeros_like(p)
                    state["base_variance"] = torch.zeros_like(p)

                state["activity_trace"] = self.trace_extractor.update(state["activity_trace"], grad)
                self.state_memory.update_stats(state, grad)

                grad_only_alpha = compute_plasticity(
                    grad=grad,
                    activity_trace=state["activity_trace"],
                    momentum=state["momentum"],
                    variance=state["variance"],
                    config=self._grad_only_config,
                )
                alpha = grad_only_alpha
                if self.plasticity_config.mode is not PlasticityMode.ABLATION_GRAD_ONLY:
                    full_alpha = compute_plasticity(
                        grad=grad,
                        activity_trace=state["activity_trace"],
                        momentum=state["momentum"],
                        variance=state["variance"],
                        config=self.plasticity_config,
                    )
                    plasticity_gate = self._plasticity_warmup_gate()
                    plasticity_delta = full_alpha - grad_only_alpha
                    alpha = grad_only_alpha + (
                        plasticity_gate
                        * self.plasticity_config.plasticity_scale
                        * plasticity_delta
                    )

                contexts.append(
                    {
                        "param": p,
                        "grad": grad,
                        "state": state,
                        "alpha": alpha,
                        "grad_only_alpha": grad_only_alpha,
                        "controller_signal": self._extract_controller_signal(alpha, grad_only_alpha),
                        "controller_group": self._controller_group_name(p),
                    }
                )

            controller_resolved = self._resolve_controller_multipliers(contexts)
            for ctx in contexts:
                p = ctx["param"]
                grad = ctx["grad"]
                state = ctx["state"]
                alpha = ctx["alpha"]
                grad_only_alpha = ctx["grad_only_alpha"]
                controller_signal, controller_multiplier = controller_resolved.get(id(p), (0.0, 1.0))

                if self.plasticity_config.hybrid_base == "adamw":
                    beta1 = self.plasticity_config.base_momentum
                    beta2 = self.plasticity_config.base_beta2
                    state["base_momentum"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state["base_variance"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    step_count = max(float(state["step"].item()), 1.0)
                    bias_correction1 = 1.0 - beta1**step_count
                    bias_correction2 = 1.0 - beta2**step_count
                    base_momentum = state["base_momentum"] / max(bias_correction1, 1e-12)
                    base_variance = state["base_variance"] / max(bias_correction2, 1e-12)
                    base_update = base_momentum / (
                        base_variance.sqrt() + self.plasticity_config.base_eps
                    )
                    momentum_reference = base_momentum
                elif self.plasticity_config.hybrid_base == "sgd_momentum":
                    grad_with_decay = grad
                    if wd > 0:
                        grad_with_decay = grad + wd * p
                    state["base_momentum"].mul_(self.plasticity_config.base_momentum).add_(
                        grad_with_decay
                    )
                    base_update = state["base_momentum"]
                    momentum_reference = state["base_momentum"]
                else:
                    plasticity_signal = torch.zeros_like(grad)
                    orthogonal_signal = torch.zeros_like(grad)
                    if self.plasticity_config.orthogonal_residual:
                        base_update = grad
                        plasticity_signal = torch.nan_to_num(
                            (alpha - grad_only_alpha) * grad,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        gradient_norm = grad.norm()
                        gradient_norm_sq = grad.pow(2).sum()
                        self._diagnostics["orthogonal_active_count"] += 1
                        if float(gradient_norm_sq.item()) <= self.plasticity_config.eps:
                            plasticity_delta_update = torch.zeros_like(plasticity_signal)
                            self._diagnostics["orthogonal_skip_count"] += 1
                        else:
                            projection_scale = plasticity_signal.mul(grad).sum() / (
                                gradient_norm_sq + self.plasticity_config.eps
                            )
                            orthogonal_signal = torch.nan_to_num(
                                plasticity_signal - projection_scale * grad,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                            orthogonal_norm = orthogonal_signal.norm()
                            if (
                                self.plasticity_config.orthogonal_normalization
                                == "match_grad_norm"
                                and float(orthogonal_norm.item()) > self.plasticity_config.eps
                            ):
                                orthogonal_signal = orthogonal_signal * (
                                    gradient_norm / (orthogonal_norm + self.plasticity_config.eps)
                                )
                            lambda_gate = self._orthogonal_schedule_gate()
                            plasticity_delta_update = (
                                self.plasticity_config.orthogonal_lambda
                                * lambda_gate
                                * orthogonal_signal
                            )
                            delta_norm = plasticity_delta_update.norm()
                            delta_limit = (
                                self.plasticity_config.orthogonal_max_ratio * gradient_norm
                            )
                            if float(delta_norm.item()) > float(delta_limit.item()):
                                plasticity_delta_update = plasticity_delta_update * (
                                    delta_limit / (delta_norm + self.plasticity_config.eps)
                                )
                        update = base_update + plasticity_delta_update
                    elif self.plasticity_config.bounded_residual:
                        base_update = grad
                        plasticity_delta_update = (alpha - 1.0) * grad
                        residual_norm = plasticity_delta_update.norm()
                        gradient_norm = grad.norm()
                        residual_limit = (
                            self.plasticity_config.residual_max_ratio * gradient_norm
                        )
                        if residual_norm > residual_limit:
                            plasticity_delta_update = plasticity_delta_update * (
                                residual_limit / (residual_norm + 1e-12)
                            )
                        update = base_update + plasticity_delta_update
                    else:
                        base_update = grad_only_alpha * grad
                        plasticity_delta_update = (alpha - grad_only_alpha) * grad
                        update = base_update + plasticity_delta_update
                    momentum_reference = state["momentum"]

                modulation_gate = 0.0
                if self.plasticity_config.hybrid_base in {"adamw", "sgd_momentum"}:
                    modulation_signal = alpha - 1.0
                    phase_gate = self._phase_gate()
                    layer_scope_gate = self._layer_scope_gate(p)
                    modulation_gate = (
                        self.plasticity_config.modulation_strength
                        * phase_gate
                        * layer_scope_gate
                    )
                    if self.plasticity_config.modulation_target == "gain":
                        plasticity_delta_update = modulation_signal * base_update
                    elif self.plasticity_config.modulation_target == "momentum":
                        plasticity_delta_update = modulation_signal * momentum_reference
                    else:
                        trust_ratio = p.norm() / (base_update.norm() + 1e-12)
                        trust_gate = trust_ratio.clamp(0.5, 1.5) - 1.0
                        plasticity_delta_update = modulation_signal * trust_gate * base_update
                    plasticity_delta_update = modulation_gate * plasticity_delta_update
                    delta_norm = plasticity_delta_update.norm()
                    base_norm = base_update.norm()
                    delta_limit = self.plasticity_config.modulation_max_ratio * base_norm
                    if delta_norm > delta_limit:
                        plasticity_delta_update = plasticity_delta_update * (
                            delta_limit / (delta_norm + 1e-12)
                        )
                    update = base_update + plasticity_delta_update

                lr_effective = lr * controller_multiplier
                weight_decay_term = torch.zeros_like(update)
                if wd > 0 and self.plasticity_config.hybrid_base != "sgd_momentum":
                    weight_decay_term = wd * p
                    if self.decoupled_weight_decay or self.plasticity_config.hybrid_base == "adamw":
                        p.add_(p, alpha=-lr_effective * wd)
                    else:
                        update = update + weight_decay_term
                stabilized = self.stabilizer.stabilize(
                    update,
                    target_rms_scale=self._layer_target_rms_scales.get(id(p), 1.0),
                )

                alpha_detached = alpha.detach()
                grad_detached = grad.detach()
                update_detached = update.detach()
                stabilized_detached = stabilized.detach()
                alpha_count = alpha_detached.numel()
                controller_group = str(ctx["controller_group"])

                self._diagnostics["alpha_sum"] += float(alpha_detached.sum().item())
                self._diagnostics["alpha_sum_sq"] += float(alpha_detached.pow(2).sum().item())
                self._diagnostics["alpha_count"] += alpha_count
                self._diagnostics["alpha_min"] = min(
                    float(self._diagnostics["alpha_min"]),
                    float(alpha_detached.min().item()),
                )
                self._diagnostics["alpha_max"] = max(
                    float(self._diagnostics["alpha_max"]),
                    float(alpha_detached.max().item()),
                )
                self._diagnostics["alpha_near_min_count"] += int(
                    (
                        alpha_detached
                        <= float(self._diagnostics["alpha_near_min_threshold"]) + 1e-6
                    )
                    .sum()
                    .item()
                )
                self._diagnostics["alpha_near_max_count"] += int(
                    (
                        alpha_detached
                        >= float(self._diagnostics["alpha_near_max_threshold"]) - 1e-6
                    )
                    .sum()
                    .item()
                )
                self._diagnostics["raw_gradient_norm_sq"] += float(
                    grad_detached.pow(2).sum().item()
                )
                self._diagnostics["parameter_norm_sq"] += float(p.detach().pow(2).sum().item())
                self._diagnostics["raw_update_norm_sq"] += float(
                    update_detached.pow(2).sum().item()
                )
                self._diagnostics["effective_update_norm_sq"] += float(
                    (controller_multiplier * stabilized_detached).pow(2).sum().item()
                )
                self._diagnostics["plasticity_delta_norm_sq"] += float(
                    plasticity_delta_update.detach().pow(2).sum().item()
                )
                self._diagnostics["controller_signal_sum"] += controller_signal
                self._diagnostics["controller_signal_count"] += 1
                self._diagnostics["controller_signal_min"] = min(
                    float(self._diagnostics["controller_signal_min"]), controller_signal
                )
                self._diagnostics["controller_signal_max"] = max(
                    float(self._diagnostics["controller_signal_max"]), controller_signal
                )
                self._diagnostics["controller_multiplier_sum"] += controller_multiplier
                self._diagnostics["controller_multiplier_count"] += 1
                self._diagnostics["controller_multiplier_min"] = min(
                    float(self._diagnostics["controller_multiplier_min"]), controller_multiplier
                )
                self._diagnostics["controller_multiplier_max"] = max(
                    float(self._diagnostics["controller_multiplier_max"]), controller_multiplier
                )
                self._diagnostics[f"controller_group_count_{controller_group}"] += 1
                self._diagnostics[f"controller_signal_sum_{controller_group}"] += controller_signal
                self._diagnostics[
                    f"controller_multiplier_sum_{controller_group}"
                ] += controller_multiplier
                if controller_multiplier <= self.plasticity_config.controller_low + 1e-9:
                    self._diagnostics["controller_clamp_low_count"] += 1
                if controller_multiplier >= self.plasticity_config.controller_high - 1e-9:
                    self._diagnostics["controller_clamp_high_count"] += 1
                if self.plasticity_config.hybrid_base in {"adamw", "sgd_momentum"}:
                    group_flags = self._layer_group_flags(p)
                    param_count = p.numel()
                    is_active = modulation_gate > 0.0
                    delta_sq = float(plasticity_delta_update.detach().pow(2).sum().item())
                    self._diagnostics["modulation_param_count"] += param_count
                    if is_active:
                        self._diagnostics["modulation_active_param_count"] += param_count
                    for group_name, enabled in group_flags.items():
                        if not enabled:
                            continue
                        self._diagnostics[f"modulation_param_count_{group_name}"] += param_count
                        self._diagnostics[f"modulation_delta_norm_sq_{group_name}"] += delta_sq
                        if is_active:
                            self._diagnostics[f"modulation_active_param_count_{group_name}"] += param_count
                if (
                    self.plasticity_config.orthogonal_residual
                    and self.plasticity_config.hybrid_base not in {"adamw", "sgd_momentum"}
                ):
                    plasticity_signal_detached = plasticity_signal.detach()
                    orthogonal_signal_detached = orthogonal_signal.detach()
                    self._diagnostics["orthogonal_signal_norm_sq"] += float(
                        plasticity_signal_detached.pow(2).sum().item()
                    )
                    self._diagnostics["orthogonal_residual_norm_sq"] += float(
                        orthogonal_signal_detached.pow(2).sum().item()
                    )
                    self._diagnostics["gradient_plasticity_dot"] += float(
                        grad_detached.mul(plasticity_signal_detached).sum().item()
                    )
                    self._diagnostics["gradient_orthogonal_dot"] += float(
                        grad_detached.mul(orthogonal_signal_detached).sum().item()
                    )
                self._diagnostics["weight_decay_term_norm_sq"] += float(
                    weight_decay_term.detach().pow(2).sum().item()
                )
                self._diagnostics["alpha_histogram"] += torch.histc(
                    alpha_detached.float().cpu(),
                    bins=self._diagnostic_bins,
                    min=self.plasticity_config.min_alpha,
                    max=self.plasticity_config.max_alpha,
                ).to(dtype=torch.float64)
                p.add_(stabilized, alpha=-lr_effective)

        return loss
