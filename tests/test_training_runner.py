import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch missing")


def test_artifact_stem_uses_run_name_when_present():
    from neuroplastic_optimizer.training.config import ExperimentConfig
    from neuroplastic_optimizer.training.runner import _artifact_stem

    cfg = ExperimentConfig(dataset="mnist", optimizer="adamw", run_name="exp01")
    assert _artifact_stem("configs/mnist/adamw.yaml", cfg) == "exp01_mnist_adamw"


def test_artifact_stem_falls_back_to_config_name():
    from neuroplastic_optimizer.training.config import ExperimentConfig
    from neuroplastic_optimizer.training.runner import _artifact_stem

    cfg = ExperimentConfig(dataset="mnist", optimizer="sgd")
    assert _artifact_stem("configs/mnist/sgd.yaml", cfg) == "sgd_mnist_sgd"


def test_resolve_device_returns_requested_cpu():
    from neuroplastic_optimizer.training.runner import _resolve_device

    assert str(_resolve_device("cpu")) == "cpu"
