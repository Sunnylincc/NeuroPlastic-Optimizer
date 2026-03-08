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


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("scheduler", "cosine"),
        ("scheduler_gamma", 0.0),
        ("scheduler_gamma", 1.5),
        ("device", "cuda:"),
        ("device", "tpu"),
    ],
)
def test_experiment_config_validation_rejects_scheduler_and_device_variants(field_name, field_value):
    from neuroplastic_optimizer.training.config import ExperimentConfig

    cfg = ExperimentConfig(**{field_name: field_value})
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize(
    "payload",
    [
        {
            "experiment": {"dataset": "mnist", "unknown_knob": True},
        },
        {
            "experiment": {"dataset": "mnist"},
            "plasticity": {"gradient_weigth": 0.2},
        },
        {
            "experiment": {"dataset": "mnist"},
            "homeostatic": {"adapt_rate": 0.1},
        },
    ],
)
def test_parse_and_validate_training_config_rejects_unknown_fields(payload):
    from neuroplastic_optimizer.training.config import parse_and_validate_training_config

    with pytest.raises(ValueError, match="unknown field"):
        parse_and_validate_training_config(payload)


@pytest.mark.parametrize(
    "plasticity",
    [
        {"activity_weight": -0.1, "gradient_weight": 0.9, "memory_weight": 0.2},
        {"activity_weight": 0.5, "gradient_weight": 0.5, "memory_weight": 0.2},
        {"min_alpha": 2.0, "max_alpha": 1.0},
    ],
)
def test_parse_and_validate_training_config_rejects_invalid_plasticity(plasticity):
    from neuroplastic_optimizer.training.config import parse_and_validate_training_config

    with pytest.raises(ValueError):
        parse_and_validate_training_config({"experiment": {}, "plasticity": plasticity})


@pytest.mark.parametrize(
    "homeostatic",
    [
        {"max_update_norm": 0},
        {"target_rms": 0},
        {"adaptation_rate": -0.1},
        {"adaptation_rate": 1.2},
    ],
)
def test_parse_and_validate_training_config_rejects_invalid_homeostatic(homeostatic):
    from neuroplastic_optimizer.training.config import parse_and_validate_training_config

    with pytest.raises(ValueError):
        parse_and_validate_training_config({"experiment": {}, "homeostatic": homeostatic})
