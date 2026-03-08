from __future__ import annotations

from dataclasses import dataclass

from neuroplastic_optimizer.plasticity import PlasticityConfig, PlasticityMode
from neuroplastic_optimizer.stabilization import HomeostaticConfig


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
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be > 0 when set")
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"unsupported log_level: {self.log_level}")
        if self.amp_dtype not in {"fp16", "bf16"}:
            raise ValueError(f"unsupported amp_dtype: {self.amp_dtype}")


def plasticity_config_from_dict(data: dict) -> PlasticityConfig:
    mode = PlasticityMode(data.get("mode", "rule_based"))
    return PlasticityConfig(
        mode=mode,
        activity_weight=float(data.get("activity_weight", 0.4)),
        gradient_weight=float(data.get("gradient_weight", 0.4)),
        memory_weight=float(data.get("memory_weight", 0.2)),
        min_alpha=float(data.get("min_alpha", 0.2)),
        max_alpha=float(data.get("max_alpha", 2.0)),
        layerwise=bool(data.get("layerwise", True)),
        parameterwise=bool(data.get("parameterwise", True)),
    )


def homeostatic_config_from_dict(data: dict) -> HomeostaticConfig:
    return HomeostaticConfig(
        max_update_norm=float(data.get("max_update_norm", 1.0)),
        target_rms=float(data.get("target_rms", 0.02)),
        adaptation_rate=float(data.get("adaptation_rate", 0.01)),
    )
