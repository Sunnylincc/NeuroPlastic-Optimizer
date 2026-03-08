from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import typer
from torch import nn
from torch.optim import Optimizer

from neuroplastic_optimizer.models.cnn import SmallCIFARNet
from neuroplastic_optimizer.models.mlp import MLPClassifier
from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
from neuroplastic_optimizer.plasticity import PlasticityConfig
from neuroplastic_optimizer.stabilization import HomeostaticConfig
from neuroplastic_optimizer.training.config import (
    ExperimentConfig,
    parse_and_validate_training_config,
)
from neuroplastic_optimizer.training.data import build_dataloaders
from neuroplastic_optimizer.utils.io import dump_json, load_yaml
from neuroplastic_optimizer.utils.seed import set_seed

app = typer.Typer(no_args_is_help=True)


class Metrics(dict[str, float]):
    pass


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "event_payload", None)
        if isinstance(payload, dict):
            return " ".join(f"{key}={value}" for key, value in payload.items())
        return record.getMessage()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "event_payload", None)
        if isinstance(payload, dict):
            return json.dumps(payload, ensure_ascii=False)
        return json.dumps({"message": record.getMessage()}, ensure_ascii=False)


def _build_logger(cfg: ExperimentConfig) -> logging.Logger:
    logger = logging.getLogger("neuroplastic_optimizer.training.runner")
    logger.propagate = False
    logger.handlers.clear()
    level_name = cfg.log_level.upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if cfg.log_json else KeyValueFormatter())
    logger.addHandler(handler)
    return logger


def _write_event(file_obj, payload: dict[str, Any]) -> None:
    file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _flush_metrics_history(path: Path, history: dict[str, Any]) -> None:
    dump_json(path, history)


def _get_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(rng_state: dict[str, Any]) -> None:
    torch_rng = rng_state.get("torch")
    if torch_rng is not None:
        torch.set_rng_state(torch_rng)
    torch_cuda_rng = rng_state.get("torch_cuda")
    if torch_cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_rng)
    numpy_rng = rng_state.get("numpy")
    if numpy_rng is not None:
        np.random.set_state(numpy_rng)
    python_rng = rng_state.get("python")
    if python_rng is not None:
        random.setstate(python_rng)


def _build_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_metric: float,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "rng_state": _get_rng_state(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    return checkpoint


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


def _make_optimizer(
    model: nn.Module,
    cfg: ExperimentConfig,
    *,
    plasticity_cfg: PlasticityConfig,
    homeostatic_cfg: HomeostaticConfig,
) -> Optimizer:
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
    return NeuroPlasticOptimizer(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        plasticity_config=plasticity_cfg,
        homeostatic_config=homeostatic_cfg,
    )


def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}
    return mapping[amp_dtype]


def init_distributed_if_needed(cfg: ExperimentConfig) -> dict[str, Any]:
    # Placeholder for future DDP/FSDP setup.
    return {"enabled": False, "world_size": 1, "rank": 0}


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    train: bool,
    *,
    mixed_precision: bool = False,
    amp_dtype: str = "fp16",
    gradient_accumulation_steps: int = 1,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[Metrics, int]:
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    update_steps = 0
    accumulation = max(1, gradient_accumulation_steps)
    amp_enabled = mixed_precision and device.type in {"cuda", "cpu"}
    total_batches = len(loader) if hasattr(loader, "__len__") else None

    if train:
        optimizer.zero_grad(set_to_none=True)

    use_scaler = train and mixed_precision and scaler is not None and scaler.is_enabled()

    with torch.set_grad_enabled(train):
        for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=_resolve_amp_dtype(amp_dtype), enabled=amp_enabled)
                if device.type in {"cuda", "cpu"}
                else nullcontext()
            )
            with autocast_context:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            if train:
                step_loss = loss / accumulation
                if use_scaler:
                    scaler.scale(step_loss).backward()
                else:
                    step_loss.backward()
                should_step = batch_idx % accumulation == 0
                if total_batches is not None and batch_idx == total_batches:
                    should_step = True
                if should_step:
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_steps += 1
            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_x.size(0)
    return Metrics(loss=total_loss / total, accuracy=correct / total), update_steps


def run_experiment(config_path: str) -> dict[str, Any]:
    raw = load_yaml(config_path)
    parsed = parse_and_validate_training_config(raw)
    cfg = parsed.experiment
    device = _resolve_device(cfg.device)
    distributed_state = init_distributed_if_needed(cfg)
    set_seed(cfg.seed)

    train_loader, test_loader = build_dataloaders(
        cfg.dataset,
        cfg.batch_size,
        cfg.num_workers,
        data_root=cfg.data_root,
        download=cfg.download,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
    )
    model = _make_model(cfg.dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(
        model,
        cfg,
        plasticity_cfg=parsed.plasticity,
        homeostatic_cfg=parsed.homeostatic,
    )

    scheduler = None
    if cfg.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler_gamma)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")

    start_epoch = 1
    best_metric = float("-inf")
    global_update_step = 0

    if cfg.resume_from is not None:
        resume_path = Path(cfg.resume_from)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        _set_rng_state(checkpoint.get("rng_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        global_update_step = int(checkpoint.get("global_update_step", global_update_step))

    history: dict[str, Any] = {
        "train": [],
        "test": [],
        "config": asdict(cfg),
        "device": str(device),
        "distributed": distributed_state,
    }

    out_dir = Path(cfg.output_dir)
    ckpt_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stem = _artifact_stem(config_path, cfg)
    checkpoint_path = ckpt_dir / f"{stem}_model.pt"
    metrics_path = out_dir / f"{stem}_metrics.json"
    events_path = out_dir / f"{stem}_events.jsonl"
    logger = _build_logger(cfg)

    with open(events_path, "a", encoding="utf-8") as events_file:
        try:
            for epoch in range(start_epoch, cfg.epochs + 1):
                epoch_start = time.perf_counter()
                train_metrics, update_steps = _run_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    train=True,
                    mixed_precision=cfg.mixed_precision,
                    amp_dtype=cfg.amp_dtype,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    scaler=scaler,
                )
                global_update_step += update_steps
                test_metrics, _ = _run_epoch(
                    model,
                    test_loader,
                    criterion,
                    optimizer,
                    device,
                    train=False,
                    mixed_precision=cfg.mixed_precision,
                    amp_dtype=cfg.amp_dtype,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                )
                improved = test_metrics["accuracy"] > best_metric
                best_metric = max(best_metric, test_metrics["accuracy"])
                if scheduler is not None:
                    for _ in range(update_steps):
                        scheduler.step()
                lr = float(optimizer.param_groups[0]["lr"])
                elapsed = time.perf_counter() - epoch_start
                history["train"].append(train_metrics)
                history["test"].append(test_metrics)
                event = {
                    "epoch": epoch,
                    "train_loss": float(train_metrics["loss"]),
                    "train_acc": float(train_metrics["accuracy"]),
                    "test_loss": float(test_metrics["loss"]),
                    "test_acc": float(test_metrics["accuracy"]),
                    "lr": lr,
                    "time_per_epoch": elapsed,
                    "device": str(device),
                    "update_steps": update_steps,
                    "global_update_step": global_update_step,
                }
                logger.info("epoch_metrics", extra={"event_payload": event})
                _write_event(events_file, event)
                events_file.flush()

                checkpoint = _build_checkpoint(model, optimizer, scheduler, epoch, best_metric)
                checkpoint["scaler_state_dict"] = scaler.state_dict()
                checkpoint["global_update_step"] = global_update_step
                should_save = epoch % cfg.save_every_n_epochs == 0
                if cfg.save_best_only:
                    should_save = should_save and improved
                if should_save:
                    torch.save(checkpoint, checkpoint_path)

                if cfg.metrics_flush_every_epoch:
                    _flush_metrics_history(metrics_path, history)
        except Exception:
            _flush_metrics_history(metrics_path, history)
            raise

    _flush_metrics_history(metrics_path, history)

    if not checkpoint_path.exists():
        checkpoint_epoch = cfg.epochs if cfg.epochs >= start_epoch else start_epoch - 1
        checkpoint = _build_checkpoint(model, optimizer, scheduler, checkpoint_epoch, best_metric)
        checkpoint["scaler_state_dict"] = scaler.state_dict()
        checkpoint["global_update_step"] = global_update_step
        torch.save(checkpoint, checkpoint_path)

    last_test_loss = history["test"][-1]["loss"] if history["test"] else None
    summary = {
        "run_name": cfg.run_name,
        "best_test_accuracy": best_metric,
        "last_test_loss": last_test_loss,
        "optimizer": cfg.optimizer,
        "dataset": cfg.dataset,
        "checkpoint": str(checkpoint_path),
        "global_update_step": global_update_step,
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
