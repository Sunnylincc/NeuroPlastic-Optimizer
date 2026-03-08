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



def test_experiment_config_validation_rejects_unknown_log_level():
    from neuroplastic_optimizer.training.config import ExperimentConfig

    cfg = ExperimentConfig(log_level="verbose")
    with pytest.raises(ValueError):
        cfg.validate()


def test_experiment_config_validation_rejects_non_positive_gradient_accumulation_steps():
    from neuroplastic_optimizer.training.config import ExperimentConfig

    cfg = ExperimentConfig(gradient_accumulation_steps=0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_experiment_config_validation_rejects_unsupported_amp_dtype():
    from neuroplastic_optimizer.training.config import ExperimentConfig

    cfg = ExperimentConfig(amp_dtype="fp8")
    with pytest.raises(ValueError):
        cfg.validate()
