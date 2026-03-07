from __future__ import annotations

from dataclasses import dataclass

from neuroplastic_optimizer.plasticity import PlasticityConfig, PlasticityMode
from neuroplastic_optimizer.stabilization import HomeostaticConfig


@dataclass
class ExperimentConfig:
    dataset: str = "mnist"
    batch_size: int = 128
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "neuroplastic"
    seed: int = 42
    num_workers: int = 2
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"
    scheduler: str | None = None
    scheduler_gamma: float = 0.95



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
