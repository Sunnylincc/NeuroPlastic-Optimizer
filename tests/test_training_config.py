import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch missing")


def test_experiment_config_validation_rejects_invalid_values():
    from neuroplastic_optimizer.training.config import ExperimentConfig

    cfg = ExperimentConfig(batch_size=0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_experiment_config_validation_rejects_unknown_optimizer():
    from neuroplastic_optimizer.training.config import ExperimentConfig

    cfg = ExperimentConfig(optimizer="unknown")
    with pytest.raises(ValueError):
        cfg.validate()


def test_experiment_config_validation_rejects_non_positive_checkpoint_interval():
    from neuroplastic_optimizer.training.config import ExperimentConfig

    cfg = ExperimentConfig(save_every_n_epochs=0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_experiment_config_validation_rejects_non_positive_max_grad_norm():
    from neuroplastic_optimizer.training.config import ExperimentConfig

    cfg = ExperimentConfig(max_grad_norm=0.0)
    with pytest.raises(ValueError):
        cfg.validate()
