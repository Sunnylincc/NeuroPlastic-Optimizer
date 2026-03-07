"""NeuroPlastic Optimizer package."""

from importlib import import_module
from typing import Any

__all__ = [
    "NeuroPlasticOptimizer",
    "PlasticityConfig",
    "PlasticityMode",
    "compute_plasticity",
]


def __getattr__(name: str) -> Any:
    if name == "NeuroPlasticOptimizer":
        return import_module("neuroplastic_optimizer.optimizer").NeuroPlasticOptimizer
    if name in {"PlasticityConfig", "PlasticityMode", "compute_plasticity"}:
        module = import_module("neuroplastic_optimizer.plasticity")
        return getattr(module, name)
    raise AttributeError(f"module 'neuroplastic_optimizer' has no attribute {name!r}")
