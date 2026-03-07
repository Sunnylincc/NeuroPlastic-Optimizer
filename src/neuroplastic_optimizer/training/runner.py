from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import typer
from torch import nn

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


def _make_model(dataset: str) -> nn.Module:
    if dataset in {"mnist", "fashionmnist"}:
        return MLPClassifier(28 * 28, 256, 10)
    if dataset == "cifar10":
        return SmallCIFARNet(10)
    raise ValueError(dataset)


def _make_optimizer(model: nn.Module, cfg: ExperimentConfig, raw: dict):
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
    if cfg.optimizer == "neuroplastic":
        p_cfg = plasticity_config_from_dict(raw.get("plasticity", {}))
        h_cfg = homeostatic_config_from_dict(raw.get("homeostatic", {}))
        return NeuroPlasticOptimizer(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            plasticity_config=p_cfg,
            homeostatic_config=h_cfg,
        )
    raise ValueError(cfg.optimizer)


def _epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
    return {"loss": total_loss / total, "acc": correct / total}


@app.command()
def main(
    config_path: str = typer.Option(..., "--config", help="Path to YAML experiment config")
) -> None:
    raw = load_yaml(config_path)
    cfg = ExperimentConfig(**raw["experiment"])
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    train_loader, test_loader = build_dataloaders(cfg.dataset, cfg.batch_size, cfg.num_workers)
    model = _make_model(cfg.dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, cfg, raw)

    scheduler = None
    if cfg.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler_gamma)

    history = {"train": [], "test": [], "config": asdict(cfg)}

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_metrics = _epoch(model, test_loader, criterion, optimizer, device, train=False)
        if scheduler is not None:
            scheduler.step()
        history["train"].append(train_metrics)
        history["test"].append(test_metrics)
        msg = (
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['acc']:.4f}"
        )
        print(msg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(config_path).stem
    dump_json(out_dir / f"{stem}_metrics.json", history)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{stem}_model.pt")

    summary = {
        "best_test_acc": max(x["acc"] for x in history["test"]),
        "last_test_loss": history["test"][-1]["loss"],
        "optimizer": cfg.optimizer,
        "dataset": cfg.dataset,
    }
    with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    app()
