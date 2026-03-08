from __future__ import annotations

from dataclasses import dataclass, fields
from difflib import get_close_matches
from typing import Any, Mapping

from neuroplastic_optimizer.plasticity import PlasticityConfig, PlasticityMode
from neuroplastic_optimizer.stabilization import HomeostaticConfig

ALLOWED_SCHEDULERS = {"exponential"}


def _raise_unknown_fields(section: str, unknown_fields: set[str], allowed_fields: set[str]) -> None:
    suggestions: list[str] = []
    for field_name in sorted(unknown_fields):
        close = get_close_matches(field_name, sorted(allowed_fields), n=1)
        if close:
            suggestions.append(f"'{field_name}' (did you mean '{close[0]}'?)")
        else:
            suggestions.append(f"'{field_name}'")
    raise ValueError(
        f"unknown field(s) in '{section}': {', '.join(suggestions)}. "
        f"Allowed fields: {sorted(allowed_fields)}"
    )


def _ensure_dict(section: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"'{section}' must be a mapping/object")
    return dict(value)


def _validate_device(device: str) -> None:
    if device == "cpu" or device == "cuda":
        return
    if device.startswith("cuda:"):
        index = device.split(":", maxsplit=1)[1]
        if index.isdigit():
            return
    raise ValueError("device must be one of: 'cpu', 'cuda', or 'cuda:<index>'")


@dataclass(slots=True)
class ExperimentConfig:
    dataset: str = "mnist"
    batch_size: int = 128
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "neuroplastic"
    seed: int = 42
    num_workers: int = 2
    data_root: str = "data"
    download: bool = True
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"
    scheduler: str | None = None
    scheduler_gamma: float = 0.95
    mixed_precision: bool = False
    amp_dtype: str = "fp16"
    gradient_accumulation_steps: int = 1
    run_name: str | None = None
    resume_from: str | None = None
    save_every_n_epochs: int = 1
    save_best_only: bool = False
    log_level: str = "INFO"
    log_json: bool = False
    metrics_flush_every_epoch: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.save_every_n_epochs <= 0:
            raise ValueError("save_every_n_epochs must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.optimizer not in {"neuroplastic", "sgd", "adam", "adamw"}:
            raise ValueError(f"unsupported optimizer: {self.optimizer}")
        if self.scheduler is not None and self.scheduler not in ALLOWED_SCHEDULERS:
            raise ValueError(
                f"unsupported scheduler: {self.scheduler}. Supported schedulers: {sorted(ALLOWED_SCHEDULERS)}"
            )
        if not (0.0 < self.scheduler_gamma <= 1.0):
            raise ValueError("scheduler_gamma must be in (0, 1]")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be > 0 when set")
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"unsupported log_level: {self.log_level}")
        if self.amp_dtype not in {"fp16", "bf16"}:
            raise ValueError(f"unsupported amp_dtype: {self.amp_dtype}")
        _validate_device(self.device)


def validate_plasticity_config(config: PlasticityConfig) -> None:
    if config.activity_weight < 0 or config.gradient_weight < 0 or config.memory_weight < 0:
        raise ValueError("plasticity weights must be >= 0")
    weight_sum = config.activity_weight + config.gradient_weight + config.memory_weight
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError("plasticity weights must sum to 1.0")
    if config.min_alpha > config.max_alpha:
        raise ValueError("plasticity min_alpha must be <= max_alpha")
    if config.min_alpha <= 0 or config.max_alpha <= 0:
        raise ValueError("plasticity alpha bounds must be > 0")


def validate_homeostatic_config(config: HomeostaticConfig) -> None:
    if config.max_update_norm <= 0:
        raise ValueError("homeostatic max_update_norm must be > 0")
    if config.target_rms <= 0:
        raise ValueError("homeostatic target_rms must be > 0")
    if not (0 <= config.adaptation_rate <= 1):
        raise ValueError("homeostatic adaptation_rate must be in [0, 1]")
    if config.eps <= 0:
        raise ValueError("homeostatic eps must be > 0")


def plasticity_config_from_dict(data: dict) -> PlasticityConfig:
    mode = PlasticityMode(data.get("mode", "rule_based"))
    config = PlasticityConfig(
        mode=mode,
        activity_weight=float(data.get("activity_weight", 0.4)),
        gradient_weight=float(data.get("gradient_weight", 0.4)),
        memory_weight=float(data.get("memory_weight", 0.2)),
        min_alpha=float(data.get("min_alpha", 0.2)),
        max_alpha=float(data.get("max_alpha", 2.0)),
        layerwise=bool(data.get("layerwise", True)),
        parameterwise=bool(data.get("parameterwise", True)),
    )
    validate_plasticity_config(config)
    return config


def homeostatic_config_from_dict(data: dict) -> HomeostaticConfig:
    config = HomeostaticConfig(
        max_update_norm=float(data.get("max_update_norm", 1.0)),
        target_rms=float(data.get("target_rms", 0.02)),
        adaptation_rate=float(data.get("adaptation_rate", 0.01)),
    )
    validate_homeostatic_config(config)
    return config


@dataclass(slots=True)
class TrainingConfigSchema:
    experiment: ExperimentConfig
    plasticity: PlasticityConfig
    homeostatic: HomeostaticConfig


def parse_and_validate_training_config(raw: Mapping[str, Any]) -> TrainingConfigSchema:
    payload = _ensure_dict("root", raw)

    root_allowed = {"experiment", "plasticity", "homeostatic"}
    root_unknown = set(payload) - root_allowed
    if root_unknown:
        _raise_unknown_fields("root", root_unknown, root_allowed)

    if "experiment" not in payload:
        raise ValueError("missing required section: 'experiment'")

    experiment_data = _ensure_dict("experiment", payload.get("experiment"))
    plasticity_data = _ensure_dict("plasticity", payload.get("plasticity"))
    homeostatic_data = _ensure_dict("homeostatic", payload.get("homeostatic"))

    experiment_allowed = {field.name for field in fields(ExperimentConfig)}
    plasticity_allowed = {field.name for field in fields(PlasticityConfig)}
    homeostatic_allowed = {field.name for field in fields(HomeostaticConfig)}

    experiment_unknown = set(experiment_data) - experiment_allowed
    if experiment_unknown:
        _raise_unknown_fields("experiment", experiment_unknown, experiment_allowed)

    plasticity_unknown = set(plasticity_data) - plasticity_allowed
    if plasticity_unknown:
        _raise_unknown_fields("plasticity", plasticity_unknown, plasticity_allowed)

    homeostatic_unknown = set(homeostatic_data) - homeostatic_allowed
    if homeostatic_unknown:
        _raise_unknown_fields("homeostatic", homeostatic_unknown, homeostatic_allowed)

    experiment = ExperimentConfig(**experiment_data)
    experiment.validate()
    plasticity = plasticity_config_from_dict(plasticity_data)
    homeostatic = homeostatic_config_from_dict(homeostatic_data)
    return TrainingConfigSchema(experiment=experiment, plasticity=plasticity, homeostatic=homeostatic)
