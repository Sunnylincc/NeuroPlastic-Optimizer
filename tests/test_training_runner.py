import importlib.util
import json
from pathlib import Path

import torch
import yaml

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


def test_run_experiment_resume_from_checkpoint_continues_epoch_and_metric(tmp_path, monkeypatch):
    from neuroplastic_optimizer.training.runner import run_experiment

    epoch_log: list[int] = []

    def fake_build_dataloaders(dataset: str, batch_size: int, num_workers: int, **kwargs):
        return [object()], [object()]

    def fake_run_epoch(model, loader, criterion, optimizer, device, train: bool, **kwargs):
        if train:
            optimizer.param_groups[0]["lr"] *= 0.99
            return {"loss": 1.0, "accuracy": 0.2}, 1
        epoch = len(epoch_log) + 1
        epoch_log.append(epoch)
        return {"loss": 1.0 / epoch, "accuracy": 0.5 + 0.1 * epoch}, 0

    monkeypatch.setattr(
        "neuroplastic_optimizer.training.runner.build_dataloaders",
        fake_build_dataloaders,
    )
    monkeypatch.setattr("neuroplastic_optimizer.training.runner._run_epoch", fake_run_epoch)

    base_experiment = {
        "dataset": "synthetic_mnist",
        "batch_size": 2,
        "epochs": 2,
        "lr": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "seed": 123,
        "num_workers": 0,
        "output_dir": str(tmp_path / "results"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "device": "cpu",
        "run_name": "resume_case",
        "scheduler": "exponential",
        "scheduler_gamma": 0.9,
        "save_every_n_epochs": 1,
        "save_best_only": False,
    }

    config_path = tmp_path / "first.yaml"
    config_path.write_text(yaml.safe_dump({"experiment": base_experiment}), encoding="utf-8")

    first_summary = run_experiment(str(config_path))
    checkpoint_path = Path(first_summary["checkpoint"])
    assert checkpoint_path.exists()
    first_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert first_checkpoint["epoch"] == 2
    assert first_checkpoint["best_metric"] == pytest.approx(0.7)
    assert "model_state_dict" in first_checkpoint
    assert "optimizer_state_dict" in first_checkpoint
    assert "scheduler_state_dict" in first_checkpoint
    assert "rng_state" in first_checkpoint

    resumed_experiment = {
        **base_experiment,
        "epochs": 4,
        "resume_from": str(checkpoint_path),
    }
    resume_config_path = tmp_path / "resume.yaml"
    resume_config_path.write_text(yaml.safe_dump({"experiment": resumed_experiment}), encoding="utf-8")

    second_summary = run_experiment(str(resume_config_path))
    resumed_checkpoint = torch.load(Path(second_summary["checkpoint"]), map_location="cpu", weights_only=False)

    assert resumed_checkpoint["epoch"] == 4
    assert resumed_checkpoint["best_metric"] == pytest.approx(0.9)
    assert second_summary["best_test_accuracy"] == pytest.approx(0.9)



def test_run_experiment_writes_jsonl_events_with_required_fields(tmp_path, monkeypatch):
    from neuroplastic_optimizer.training.runner import run_experiment

    def fake_build_dataloaders(dataset: str, batch_size: int, num_workers: int, **kwargs):
        return [object()], [object()]

    def fake_run_epoch(model, loader, criterion, optimizer, device, train: bool, **kwargs):
        if train:
            return {"loss": 0.8, "accuracy": 0.7}, 1
        return {"loss": 0.6, "accuracy": 0.75}, 0

    monkeypatch.setattr(
        "neuroplastic_optimizer.training.runner.build_dataloaders",
        fake_build_dataloaders,
    )
    monkeypatch.setattr("neuroplastic_optimizer.training.runner._run_epoch", fake_run_epoch)

    experiment = {
        "dataset": "synthetic_mnist",
        "batch_size": 2,
        "epochs": 2,
        "lr": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "seed": 123,
        "num_workers": 0,
        "output_dir": str(tmp_path / "results"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "device": "cpu",
        "run_name": "event_case",
        "save_every_n_epochs": 1,
        "save_best_only": False,
        "log_json": True,
    }

    config_path = tmp_path / "event.yaml"
    config_path.write_text(yaml.safe_dump({"experiment": experiment}), encoding="utf-8")

    run_experiment(str(config_path))

    events_path = tmp_path / "results" / "event_case_synthetic_mnist_adam_events.jsonl"
    assert events_path.exists()
    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    required = {
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "lr",
        "time_per_epoch",
        "device",
    }
    assert required.issubset(lines[0].keys())


def test_run_experiment_flushes_metrics_on_exception(tmp_path, monkeypatch):
    from neuroplastic_optimizer.training.runner import run_experiment

    calls = {"count": 0}

    def fake_build_dataloaders(dataset: str, batch_size: int, num_workers: int, **kwargs):
        return [object()], [object()]

    def fake_run_epoch(model, loader, criterion, optimizer, device, train: bool, **kwargs):
        if train:
            return {"loss": 0.9, "accuracy": 0.1}, 1
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("boom")
        return {"loss": 0.8, "accuracy": 0.2}, 0

    monkeypatch.setattr(
        "neuroplastic_optimizer.training.runner.build_dataloaders",
        fake_build_dataloaders,
    )
    monkeypatch.setattr("neuroplastic_optimizer.training.runner._run_epoch", fake_run_epoch)

    experiment = {
        "dataset": "synthetic_mnist",
        "batch_size": 2,
        "epochs": 3,
        "lr": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "seed": 123,
        "num_workers": 0,
        "output_dir": str(tmp_path / "results"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "device": "cpu",
        "run_name": "exception_case",
        "save_every_n_epochs": 1,
        "save_best_only": False,
    }

    config_path = tmp_path / "exception.yaml"
    config_path.write_text(yaml.safe_dump({"experiment": experiment}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        run_experiment(str(config_path))

    metrics_path = tmp_path / "results" / "exception_case_synthetic_mnist_adam_metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(metrics["test"]) == 1


def test_run_epoch_gradient_accumulation_counts_update_steps():
    from neuroplastic_optimizer.training.runner import _run_epoch

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    batches = [
        (torch.randn(2, 4), torch.tensor([0, 1])),
        (torch.randn(2, 4), torch.tensor([1, 0])),
        (torch.randn(2, 4), torch.tensor([0, 1])),
        (torch.randn(2, 4), torch.tensor([1, 0])),
        (torch.randn(2, 4), torch.tensor([0, 1])),
    ]

    metrics, update_steps = _run_epoch(
        model,
        batches,
        criterion,
        optimizer,
        torch.device("cpu"),
        train=True,
        mixed_precision=False,
        gradient_accumulation_steps=2,
        scaler=None,
    )

    assert metrics["loss"] > 0
    assert 0 <= metrics["accuracy"] <= 1
    assert update_steps == 3


def test_run_epoch_mixed_precision_disabled_does_not_use_scaler():
    from neuroplastic_optimizer.training.runner import _run_epoch

    class FailingScaler:
        def is_enabled(self):
            return True

        def scale(self, _):
            raise AssertionError("scale should not be called when mixed_precision=False")

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    loader = [(torch.randn(2, 4), torch.tensor([0, 1]))]

    metrics, update_steps = _run_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        train=True,
        mixed_precision=False,
        gradient_accumulation_steps=1,
        scaler=FailingScaler(),
    )

    assert metrics["loss"] > 0
    assert update_steps == 1


def test_run_experiment_validates_before_model_construction(tmp_path, monkeypatch):
    from neuroplastic_optimizer.training.runner import run_experiment

    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        yaml.safe_dump({"experiment": {"dataset": "synthetic_mnist", "device": "cuda:"}}),
        encoding="utf-8",
    )

    called = {"model": False}

    def fail_if_called(*args, **kwargs):
        called["model"] = True
        raise AssertionError("model construction should not happen for invalid config")

    monkeypatch.setattr("neuroplastic_optimizer.training.runner._make_model", fail_if_called)

    with pytest.raises(ValueError, match="device"):
        run_experiment(str(config_path))

    assert called["model"] is False
