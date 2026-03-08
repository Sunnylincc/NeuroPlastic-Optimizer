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

    def fake_build_dataloaders(dataset: str, batch_size: int, num_workers: int):
        return [object()], [object()]

    def fake_run_epoch(model, loader, criterion, optimizer, device, train: bool, **kwargs):
        if train:
            optimizer.param_groups[0]["lr"] *= 0.99
            return {"loss": 1.0, "accuracy": 0.2}
        epoch = len(epoch_log) + 1
        epoch_log.append(epoch)
        return {"loss": 1.0 / epoch, "accuracy": 0.5 + 0.1 * epoch}

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


def test_run_epoch_non_finite_loss_writes_failure_snapshot(tmp_path):
    import torch

    from neuroplastic_optimizer.training.config import ExperimentConfig
    from neuroplastic_optimizer.training.runner import NonFiniteLossError, _run_epoch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

    class NaNLoss(torch.nn.Module):
        def forward(self, logits, target):
            return logits.sum() * torch.tensor(float("nan"), device=logits.device)

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = [(torch.randn(3, 4), torch.zeros(3, dtype=torch.long))]
    cfg = ExperimentConfig(
        dataset="synthetic_mnist",
        optimizer="sgd",
        fail_on_non_finite=True,
        output_dir=str(tmp_path),
    )
    snapshot_path = tmp_path / "failure_snapshot.json"

    with pytest.raises(NonFiniteLossError):
        _run_epoch(
            model,
            loader,
            NaNLoss(),
            optimizer,
            torch.device("cpu"),
            train=True,
            cfg=cfg,
            failure_snapshot_path=snapshot_path,
        )

    assert snapshot_path.exists()
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["batch_index"] == 0
    assert payload["lr"] == pytest.approx(0.01)
    assert payload["grad_norm"] is None
    assert payload["config_summary"]["fail_on_non_finite"] is True
