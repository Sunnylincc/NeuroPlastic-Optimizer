from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import typer
from torch import nn
from torch.optim import Optimizer

from neuroplastic_optimizer.models.cnn import SmallCIFARNet
from neuroplastic_optimizer.models.mlp import MLPClassifier
from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
from neuroplastic_optimizer.training.config import (
    ExperimentConfig,
    homeostatic_config_from_dict,
    plasticity_config_from_dict,
)
from neuroplastic_optimizer.training.data import build_dataloaders
from neuroplastic_optimizer.utils.io import dump_json, load_yaml
from neuroplastic_optimizer.utils.seed import set_seed

app = typer.Typer(no_args_is_help=True)


class Metrics(dict[str, float]):
    pass


def _artifact_stem(config_path: str, cfg: ExperimentConfig) -> str:
    run_name = cfg.run_name or Path(config_path).stem
    return f"{run_name}_{cfg.dataset}_{cfg.optimizer}"


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested_device)


def _make_model(dataset: str) -> nn.Module:
    if dataset in {"mnist", "fashionmnist", "synthetic_mnist"}:
        return MLPClassifier(28 * 28, 256, 10)
    if dataset == "cifar10":
        return SmallCIFARNet(10)
    raise ValueError(f"unsupported dataset: {dataset}")


def _make_optimizer(model: nn.Module, cfg: ExperimentConfig, raw: dict[str, Any]) -> Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=0.9,
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    p_cfg = plasticity_config_from_dict(raw.get("plasticity", {}))
    h_cfg = homeostatic_config_from_dict(raw.get("homeostatic", {}))
    return NeuroPlasticOptimizer(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        plasticity_config=p_cfg,
        homeostatic_config=h_cfg,
    )


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    train: bool,
) -> Metrics:
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(train):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_x.size(0)
    return Metrics(loss=total_loss / total, accuracy=correct / total)


def run_experiment(config_path: str) -> dict[str, Any]:
    raw = load_yaml(config_path)
    cfg = ExperimentConfig(**raw["experiment"])
    cfg.validate()
    device = _resolve_device(cfg.device)
    set_seed(cfg.seed)

    train_loader, test_loader = build_dataloaders(cfg.dataset, cfg.batch_size, cfg.num_workers)
    model = _make_model(cfg.dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, cfg, raw)

    scheduler = None
    if cfg.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler_gamma)

    history: dict[str, Any] = {
        "train": [],
        "test": [],
        "config": asdict(cfg),
        "device": str(device),
    }

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_metrics = _run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        if scheduler is not None:
            scheduler.step()
        history["train"].append(train_metrics)
        history["test"].append(test_metrics)
        msg = (
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )
        print(msg)

    out_dir = Path(cfg.output_dir)
    ckpt_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stem = _artifact_stem(config_path, cfg)
    dump_json(out_dir / f"{stem}_metrics.json", history)

    checkpoint_path = ckpt_dir / f"{stem}_model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    summary = {
        "run_name": cfg.run_name,
        "best_test_accuracy": max(item["accuracy"] for item in history["test"]),
        "last_test_loss": history["test"][-1]["loss"],
        "optimizer": cfg.optimizer,
        "dataset": cfg.dataset,
        "checkpoint": str(checkpoint_path),
    }
    with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    return summary


@app.command()
def main(
    config_path: str = typer.Option(..., "--config", help="Path to YAML experiment config"),
) -> None:
    run_experiment(config_path)


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
