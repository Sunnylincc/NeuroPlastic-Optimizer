from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFERRED_ORDER = [
    "neuroplastic",
    "ablation_grad_only",
    "adamw",
    "adam",
    "sgd",
]
SEED_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")


@dataclass(slots=True)
class RunData:
    label: str
    dataset: str
    optimizer: str
    summary_path: Path | None
    metrics_path: Path | None
    events_path: Path | None
    epochs: list[int]
    test_accuracy: list[float]
    test_loss: list[float]
    best_test_accuracy: float | None
    final_test_accuracy: float | None
    final_test_loss: float | None
    device: str | None


@dataclass(slots=True)
class AggregateData:
    label: str
    seeds: list[int]
    epochs: list[int]
    mean_test_accuracy: list[float]
    std_test_accuracy: list[float]
    mean_test_loss: list[float]
    std_test_loss: list[float]
    mean_best_test_accuracy: float | None
    std_best_test_accuracy: float | None
    mean_final_test_accuracy: float | None
    std_final_test_accuracy: float | None
    mean_final_test_loss: float | None
    std_final_test_loss: float | None
    run_count: int


def _warn(message: str) -> None:
    print(f"[paper_figures] warning: {message}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_metric(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in payload:
            return _coerce_float(payload[key])
    return None


def _series_from_metrics(payload: dict[str, Any]) -> tuple[list[int], list[float], list[float]]:
    test_entries = payload.get("test")
    if not isinstance(test_entries, list):
        return [], [], []

    epochs: list[int] = []
    acc: list[float] = []
    loss: list[float] = []
    for index, entry in enumerate(test_entries, start=1):
        if not isinstance(entry, dict):
            continue
        acc_value = _pick_metric(entry, ["accuracy", "test_accuracy", "test_acc", "acc"])
        loss_value = _pick_metric(entry, ["loss", "test_loss"])
        if acc_value is None and loss_value is None:
            continue
        epochs.append(index)
        acc.append(acc_value if acc_value is not None else float("nan"))
        loss.append(loss_value if loss_value is not None else float("nan"))
    return epochs, acc, loss


def _series_from_events(rows: list[dict[str, Any]]) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    acc: list[float] = []
    loss: list[float] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        epoch = row.get("epoch", index)
        acc_value = _pick_metric(row, ["test_acc", "test_accuracy", "accuracy", "acc"])
        loss_value = _pick_metric(row, ["test_loss", "loss"])
        if acc_value is None and loss_value is None:
            continue
        epochs.append(int(epoch))
        acc.append(acc_value if acc_value is not None else float("nan"))
        loss.append(loss_value if loss_value is not None else float("nan"))
    return epochs, acc, loss


def _infer_dataset(
    label: str,
    summary_payload: dict[str, Any] | None,
    metrics_payload: dict[str, Any] | None,
) -> str | None:
    for payload in (summary_payload, metrics_payload):
        if not isinstance(payload, dict):
            continue
        dataset = payload.get("dataset")
        if isinstance(dataset, str):
            return dataset.lower()
        config = payload.get("config")
        if isinstance(config, dict):
            dataset = config.get("dataset")
            if isinstance(dataset, str):
                return dataset.lower()
    if "_mnist_" in label or label.endswith("_mnist"):
        return "mnist"
    return None


def _label_from_payloads(
    stem: str, summary_payload: dict[str, Any] | None, metrics_payload: dict[str, Any] | None
) -> str:
    for payload in (summary_payload, metrics_payload):
        if not isinstance(payload, dict):
            continue
        run_name = payload.get("run_name")
        if isinstance(run_name, str) and run_name:
            return run_name
        config = payload.get("config")
        if isinstance(config, dict):
            run_name = config.get("run_name")
            if isinstance(run_name, str) and run_name:
                return run_name
    if "_mnist_" in stem:
        return stem.split("_mnist_")[0]
    return stem


def _optimizer_from_payloads(
    stem: str,
    summary_payload: dict[str, Any] | None,
    metrics_payload: dict[str, Any] | None,
) -> str:
    for payload in (summary_payload, metrics_payload):
        if not isinstance(payload, dict):
            continue
        optimizer = payload.get("optimizer")
        if isinstance(optimizer, str) and optimizer:
            return optimizer
        config = payload.get("config")
        if isinstance(config, dict):
            optimizer = config.get("optimizer")
            if isinstance(optimizer, str) and optimizer:
                return optimizer
    return stem.rsplit("_", maxsplit=1)[-1]


def discover_mnist_runs(results_dir: Path) -> tuple[list[RunData], list[str]]:
    warnings: list[str] = []
    runs_by_stem: dict[str, dict[str, Path]] = {}
    for path in results_dir.glob("*_summary.json"):
        runs_by_stem.setdefault(path.name[: -len("_summary.json")], {})["summary"] = path
    for path in results_dir.glob("*_metrics.json"):
        runs_by_stem.setdefault(path.name[: -len("_metrics.json")], {})["metrics"] = path
    for path in results_dir.glob("*_events.jsonl"):
        runs_by_stem.setdefault(path.name[: -len("_events.jsonl")], {})["events"] = path

    runs: list[RunData] = []
    for stem, paths in sorted(runs_by_stem.items()):
        summary_payload = None
        metrics_payload = None
        events_payload: list[dict[str, Any]] = []

        summary_path = paths.get("summary")
        if summary_path is not None:
            try:
                summary_payload = _read_json(summary_path)
            except json.JSONDecodeError as exc:
                warnings.append(f"Skipping malformed summary file '{summary_path}': {exc}")
                continue

        metrics_path = paths.get("metrics")
        if metrics_path is not None:
            try:
                metrics_payload = _read_json(metrics_path)
            except json.JSONDecodeError as exc:
                warnings.append(f"Ignoring malformed metrics file '{metrics_path}': {exc}")
                metrics_path = None

        events_path = paths.get("events")
        if events_path is not None:
            try:
                events_payload = _read_jsonl(events_path)
            except json.JSONDecodeError as exc:
                warnings.append(f"Ignoring malformed events file '{events_path}': {exc}")
                events_path = None

        label = _label_from_payloads(stem, summary_payload, metrics_payload)
        dataset = _infer_dataset(stem, summary_payload, metrics_payload)
        if dataset != "mnist":
            continue

        epochs, test_accuracy, test_loss = _series_from_metrics(metrics_payload or {})
        if not epochs:
            epochs, test_accuracy, test_loss = _series_from_events(events_payload)

        if not epochs:
            warnings.append(
                f"Run '{label}' has no usable per-epoch test metrics; line plots may be skipped."
            )

        best_test_accuracy = None
        final_test_accuracy = None
        final_test_loss = None
        device = None
        if isinstance(summary_payload, dict):
            best_test_accuracy = _coerce_float(summary_payload.get("best_test_accuracy"))
            final_test_loss = _coerce_float(summary_payload.get("last_test_loss"))
            device_value = summary_payload.get("device")
            if isinstance(device_value, str):
                device = device_value
        if isinstance(metrics_payload, dict):
            device_value = metrics_payload.get("device")
            if isinstance(device_value, str):
                device = device_value

        if test_accuracy:
            clean_acc = [value for value in test_accuracy if value == value]
            if clean_acc:
                final_test_accuracy = clean_acc[-1]
                if best_test_accuracy is None:
                    best_test_accuracy = max(clean_acc)
        if test_loss and final_test_loss is None:
            clean_loss = [value for value in test_loss if value == value]
            if clean_loss:
                final_test_loss = clean_loss[-1]

        runs.append(
            RunData(
                label=label,
                dataset="mnist",
                optimizer=_optimizer_from_payloads(stem, summary_payload, metrics_payload),
                summary_path=summary_path,
                metrics_path=metrics_path,
                events_path=events_path,
                epochs=epochs,
                test_accuracy=test_accuracy,
                test_loss=test_loss,
                best_test_accuracy=best_test_accuracy,
                final_test_accuracy=final_test_accuracy,
                final_test_loss=final_test_loss,
                device=device,
            )
        )

    def sort_key(run: RunData) -> tuple[int, str]:
        try:
            return (PREFERRED_ORDER.index(run.label), run.label)
        except ValueError:
            return (len(PREFERRED_ORDER), run.label)

    return sorted(runs, key=sort_key), warnings


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.6),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.dpi": 300,
        }
    )


def _split_seed_suffix(label: str) -> tuple[str, int | None]:
    match = SEED_SUFFIX_PATTERN.match(label)
    if not match:
        return label, None
    return match.group("base"), int(match.group("seed"))


def _sort_name(label: str) -> tuple[int, str]:
    try:
        return (PREFERRED_ORDER.index(label), label)
    except ValueError:
        return (len(PREFERRED_ORDER), label)


def _is_smoke_run(label: str) -> bool:
    return "smoke" in label.lower()


def _display_label(label: str) -> str:
    base_label, seed = _split_seed_suffix(label)
    mapping = {
        "neuroplastic": "NeuroPlastic",
        "ablation_grad_only": "Ablation: Grad Only",
        "adamw": "AdamW",
        "adam": "Adam",
        "sgd": "SGD",
        "neuroplastic_cpu_smoke": "NeuroPlastic Smoke",
    }
    display = mapping.get(base_label, base_label.replace("_", " ").title())
    if seed is not None:
        return f"{display} (seed {seed})"
    return display


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _aggregate_series(
    series_list: list[list[float]], epochs: list[int]
) -> tuple[list[float], list[float]]:
    mean_values: list[float] = []
    std_values: list[float] = []
    for index in range(len(epochs)):
        values = [
            series[index]
            for series in series_list
            if index < len(series) and series[index] == series[index]
        ]
        if values:
            mean_values.append(_mean(values))
            std_values.append(_std(values))
        else:
            mean_values.append(float("nan"))
            std_values.append(float("nan"))
    return mean_values, std_values


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_accuracy_vs_epoch(runs: list[RunData], output_dir: Path) -> Path | None:
    plotted = [
        run
        for run in runs
        if not _is_smoke_run(run.label)
        and run.epochs
        and any(value == value for value in run.test_accuracy)
    ]
    if not plotted:
        _warn(
            "Skipping accuracy-vs-epoch plot because no runs contain usable test accuracy curves."
        )
        return None
    plt.figure()
    for run in plotted:
        plt.plot(
            run.epochs, run.test_accuracy, marker="o", linewidth=2, label=_display_label(run.label)
        )
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("MNIST Test Accuracy vs Epoch")
    plt.legend()
    output_path = output_dir / "mnist_test_accuracy_vs_epoch.png"
    _save_current_figure(output_path)
    return output_path


def _plot_loss_vs_epoch(runs: list[RunData], output_dir: Path) -> Path | None:
    plotted = [
        run
        for run in runs
        if not _is_smoke_run(run.label)
        and run.epochs
        and any(value == value for value in run.test_loss)
    ]
    if not plotted:
        _warn("Skipping loss-vs-epoch plot because no runs contain usable test loss curves.")
        return None
    plt.figure()
    for run in plotted:
        plt.plot(
            run.epochs, run.test_loss, marker="o", linewidth=2, label=_display_label(run.label)
        )
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("MNIST Test Loss vs Epoch")
    plt.legend()
    output_path = output_dir / "mnist_test_loss_vs_epoch.png"
    _save_current_figure(output_path)
    return output_path


def _plot_early_convergence(
    runs: list[RunData], output_dir: Path, max_epoch: int = 3
) -> Path | None:
    plotted: list[RunData] = []
    plt.figure()
    for run in runs:
        if _is_smoke_run(run.label):
            continue
        points = [
            (epoch, acc)
            for epoch, acc in zip(run.epochs, run.test_accuracy)
            if epoch <= max_epoch and acc == acc
        ]
        if not points:
            continue
        plotted.append(run)
        xs = [epoch for epoch, _ in points]
        ys = [acc for _, acc in points]
        plt.plot(xs, ys, marker="o", linewidth=2, label=_display_label(run.label))
    if not plotted:
        plt.close()
        _warn(
            "Skipping early-convergence plot because no runs contain usable early-epoch test accuracy."
        )
        return None
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title(f"MNIST Early Convergence (Epochs 1-{max_epoch})")
    plt.legend()
    output_path = output_dir / "mnist_early_convergence_accuracy.png"
    _save_current_figure(output_path)
    return output_path


def _plot_best_final_bar(runs: list[RunData], output_dir: Path) -> Path | None:
    plotted = [
        run
        for run in runs
        if not _is_smoke_run(run.label)
        if run.best_test_accuracy is not None or run.final_test_accuracy is not None
    ]
    if not plotted:
        _warn(
            "Skipping best/final accuracy bar chart because no runs expose summary accuracy values."
        )
        return None
    labels = [_display_label(run.label) for run in plotted]
    best = [
        run.best_test_accuracy if run.best_test_accuracy is not None else float("nan")
        for run in plotted
    ]
    final = [
        run.final_test_accuracy if run.final_test_accuracy is not None else float("nan")
        for run in plotted
    ]
    x_positions = list(range(len(plotted)))
    width = 0.38

    plt.figure(figsize=(8.0, 4.8))
    plt.bar([x - width / 2 for x in x_positions], best, width=width, label="Best Test Accuracy")
    plt.bar([x + width / 2 for x in x_positions], final, width=width, label="Final Test Accuracy")
    plt.xticks(x_positions, labels)
    plt.ylabel("Accuracy")
    plt.title("MNIST Best and Final Test Accuracy")
    plt.legend()
    output_path = output_dir / "mnist_best_final_test_accuracy.png"
    _save_current_figure(output_path)
    return output_path


def aggregate_seed_runs(runs: list[RunData]) -> list[AggregateData]:
    grouped: dict[str, list[tuple[int, RunData]]] = {}
    for run in runs:
        base_label, seed = _split_seed_suffix(run.label)
        if seed is None:
            continue
        grouped.setdefault(base_label, []).append((seed, run))

    aggregates: list[AggregateData] = []
    for label, seed_runs in grouped.items():
        if len(seed_runs) < 2:
            continue
        seed_runs = sorted(seed_runs, key=lambda item: item[0])
        runs_only = [run for _, run in seed_runs]
        min_len = min((len(run.epochs) for run in runs_only if run.epochs), default=0)
        epochs = runs_only[0].epochs[:min_len] if min_len else []
        accuracy_series = [run.test_accuracy[:min_len] for run in runs_only]
        loss_series = [run.test_loss[:min_len] for run in runs_only]
        mean_test_accuracy, std_test_accuracy = _aggregate_series(accuracy_series, epochs)
        mean_test_loss, std_test_loss = _aggregate_series(loss_series, epochs)

        best_values = [
            run.best_test_accuracy for run in runs_only if run.best_test_accuracy is not None
        ]
        final_acc_values = [
            run.final_test_accuracy for run in runs_only if run.final_test_accuracy is not None
        ]
        final_loss_values = [
            run.final_test_loss for run in runs_only if run.final_test_loss is not None
        ]

        aggregates.append(
            AggregateData(
                label=label,
                seeds=[seed for seed, _ in seed_runs],
                epochs=epochs,
                mean_test_accuracy=mean_test_accuracy,
                std_test_accuracy=std_test_accuracy,
                mean_test_loss=mean_test_loss,
                std_test_loss=std_test_loss,
                mean_best_test_accuracy=_mean(best_values) if best_values else None,
                std_best_test_accuracy=_std(best_values) if best_values else None,
                mean_final_test_accuracy=_mean(final_acc_values) if final_acc_values else None,
                std_final_test_accuracy=_std(final_acc_values) if final_acc_values else None,
                mean_final_test_loss=_mean(final_loss_values) if final_loss_values else None,
                std_final_test_loss=_std(final_loss_values) if final_loss_values else None,
                run_count=len(runs_only),
            )
        )

    return sorted(aggregates, key=lambda aggregate: _sort_name(aggregate.label))


def _plot_seed_aggregated_accuracy(
    aggregates: list[AggregateData], output_dir: Path
) -> Path | None:
    plotted = [aggregate for aggregate in aggregates if aggregate.epochs]
    if not plotted:
        return None
    plt.figure()
    for aggregate in plotted:
        plt.plot(
            aggregate.epochs,
            aggregate.mean_test_accuracy,
            marker="o",
            linewidth=2,
            label=_display_label(aggregate.label),
        )
        lower = [
            mean - std
            for mean, std in zip(aggregate.mean_test_accuracy, aggregate.std_test_accuracy)
        ]
        upper = [
            mean + std
            for mean, std in zip(aggregate.mean_test_accuracy, aggregate.std_test_accuracy)
        ]
        plt.fill_between(aggregate.epochs, lower, upper, alpha=0.18)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Test Accuracy")
    plt.title("MNIST Seed-Aggregated Test Accuracy")
    plt.legend()
    output_path = output_dir / "mnist_seed_aggregated_test_accuracy_vs_epoch.png"
    _save_current_figure(output_path)
    return output_path


def _plot_seed_aggregated_early_convergence(
    aggregates: list[AggregateData],
    output_dir: Path,
    max_epoch: int = 3,
) -> Path | None:
    plotted = [aggregate for aggregate in aggregates if aggregate.epochs]
    if not plotted:
        return None
    plt.figure()
    for aggregate in plotted:
        points = [
            (epoch, mean, std)
            for epoch, mean, std in zip(
                aggregate.epochs,
                aggregate.mean_test_accuracy,
                aggregate.std_test_accuracy,
            )
            if epoch <= max_epoch and mean == mean
        ]
        if not points:
            continue
        xs = [epoch for epoch, _, _ in points]
        ys = [mean for _, mean, _ in points]
        stds = [std for _, _, std in points]
        plt.plot(xs, ys, marker="o", linewidth=2, label=_display_label(aggregate.label))
        plt.fill_between(
            xs, [y - s for y, s in zip(ys, stds)], [y + s for y, s in zip(ys, stds)], alpha=0.18
        )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Test Accuracy")
    plt.title(f"MNIST Seed-Aggregated Early Convergence (Epochs 1-{max_epoch})")
    plt.legend()
    output_path = output_dir / "mnist_seed_aggregated_early_convergence_accuracy.png"
    _save_current_figure(output_path)
    return output_path


def _plot_seed_aggregated_best_final_bar(
    aggregates: list[AggregateData], output_dir: Path
) -> Path | None:
    plotted = [
        aggregate
        for aggregate in aggregates
        if aggregate.mean_best_test_accuracy is not None
        or aggregate.mean_final_test_accuracy is not None
    ]
    if not plotted:
        return None
    labels = [_display_label(aggregate.label) for aggregate in plotted]
    best = [
        aggregate.mean_best_test_accuracy
        if aggregate.mean_best_test_accuracy is not None
        else float("nan")
        for aggregate in plotted
    ]
    best_std = [
        aggregate.std_best_test_accuracy if aggregate.std_best_test_accuracy is not None else 0.0
        for aggregate in plotted
    ]
    final = [
        aggregate.mean_final_test_accuracy
        if aggregate.mean_final_test_accuracy is not None
        else float("nan")
        for aggregate in plotted
    ]
    final_std = [
        aggregate.std_final_test_accuracy if aggregate.std_final_test_accuracy is not None else 0.0
        for aggregate in plotted
    ]
    x_positions = list(range(len(plotted)))
    width = 0.38

    plt.figure(figsize=(8.0, 4.8))
    plt.bar(
        [x - width / 2 for x in x_positions],
        best,
        width=width,
        yerr=best_std,
        capsize=4,
        label="Mean Best Test Accuracy",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        final,
        width=width,
        yerr=final_std,
        capsize=4,
        label="Mean Final Test Accuracy",
    )
    plt.xticks(x_positions, labels)
    plt.ylabel("Accuracy")
    plt.title("MNIST Seed-Aggregated Best and Final Test Accuracy")
    plt.legend()
    output_path = output_dir / "mnist_seed_aggregated_best_final_test_accuracy.png"
    _save_current_figure(output_path)
    return output_path


def _write_seed_aggregate_table(aggregates: list[AggregateData], output_dir: Path) -> Path | None:
    if not aggregates:
        return None
    output_path = output_dir / "seed_aggregate_table.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_group",
                "run_count",
                "seeds",
                "epochs_aggregated",
                "mean_best_test_accuracy",
                "std_best_test_accuracy",
                "mean_final_test_accuracy",
                "std_final_test_accuracy",
                "mean_final_test_loss",
                "std_final_test_loss",
            ],
        )
        writer.writeheader()
        for aggregate in aggregates:
            writer.writerow(
                {
                    "run_group": aggregate.label,
                    "run_count": aggregate.run_count,
                    "seeds": " ".join(str(seed) for seed in aggregate.seeds),
                    "epochs_aggregated": len(aggregate.epochs),
                    "mean_best_test_accuracy": aggregate.mean_best_test_accuracy,
                    "std_best_test_accuracy": aggregate.std_best_test_accuracy,
                    "mean_final_test_accuracy": aggregate.mean_final_test_accuracy,
                    "std_final_test_accuracy": aggregate.std_final_test_accuracy,
                    "mean_final_test_loss": aggregate.mean_final_test_loss,
                    "std_final_test_loss": aggregate.std_final_test_loss,
                }
            )
    return output_path


def _write_benchmark_table(runs: list[RunData], output_dir: Path) -> Path:
    output_path = output_dir / "benchmark_table.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "optimizer",
                "dataset",
                "epochs_recorded",
                "best_test_accuracy",
                "final_test_accuracy",
                "final_test_loss",
                "device",
                "summary_path",
                "metrics_path",
                "events_path",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_name": run.label,
                    "optimizer": run.optimizer,
                    "dataset": run.dataset,
                    "epochs_recorded": len(run.epochs),
                    "best_test_accuracy": run.best_test_accuracy,
                    "final_test_accuracy": run.final_test_accuracy,
                    "final_test_loss": run.final_test_loss,
                    "device": run.device,
                    "summary_path": str(run.summary_path) if run.summary_path else "",
                    "metrics_path": str(run.metrics_path) if run.metrics_path else "",
                    "events_path": str(run.events_path) if run.events_path else "",
                }
            )
    return output_path


def _write_run_notes(
    runs: list[RunData],
    aggregates: list[AggregateData],
    warnings: list[str],
    generated_files: list[Path],
    output_dir: Path,
) -> Path:
    lines = [
        "# CPU MNIST Paper Figure Notes",
        "",
        "## Runs Found",
    ]
    if runs:
        for run in runs:
            lines.append(
                f"- `{run.label}`: optimizer=`{run.optimizer}`, epochs_recorded={len(run.epochs)}, "
                f"device=`{run.device or 'unknown'}`"
            )
    else:
        lines.append("- No MNIST runs were discovered in the results directory.")

    lines.extend(["", "## Seed Aggregates"])
    if aggregates:
        for aggregate in aggregates:
            lines.append(
                f"- `{aggregate.label}`: runs={aggregate.run_count}, seeds=`{' '.join(str(seed) for seed in aggregate.seeds)}`"
            )
    else:
        lines.append("- No multi-seed groups were available for aggregation.")

    lines.extend(["", "## Figures Generated"])
    if generated_files:
        for path in generated_files:
            lines.append(f"- `{path.name}`")
    else:
        lines.append("- No figures were generated.")

    lines.extend(["", "## Warnings"])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No warnings.")

    output_path = output_dir / "run_notes.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def generate_paper_figures(results_dir: Path, output_dir: Path) -> dict[str, Any]:
    _apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs, warnings = discover_mnist_runs(results_dir)
    aggregates = aggregate_seed_runs(runs)
    for warning in warnings:
        _warn(warning)

    generated_files: list[Path] = []
    for maybe_path in (
        _plot_accuracy_vs_epoch(runs, output_dir),
        _plot_best_final_bar(runs, output_dir),
        _plot_loss_vs_epoch(runs, output_dir),
        _plot_early_convergence(runs, output_dir),
    ):
        if maybe_path is not None:
            generated_files.append(maybe_path)

    table_path = _write_benchmark_table(runs, output_dir)
    generated_files.append(table_path)
    for maybe_path in (
        _plot_seed_aggregated_accuracy(aggregates, output_dir),
        _plot_seed_aggregated_best_final_bar(aggregates, output_dir),
        _plot_seed_aggregated_early_convergence(aggregates, output_dir),
        _write_seed_aggregate_table(aggregates, output_dir),
    ):
        if maybe_path is not None:
            generated_files.append(maybe_path)

    notes_path = _write_run_notes(runs, aggregates, warnings, generated_files, output_dir)
    generated_files.append(notes_path)

    return {
        "runs_found": [run.label for run in runs],
        "seed_aggregates_found": [aggregate.label for aggregate in aggregates],
        "warnings": warnings,
        "generated_files": [str(path) for path in generated_files],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compact CPU-friendly MNIST paper figures"
    )
    parser.add_argument(
        "--results-dir", default="results", help="Directory containing result artifacts"
    )
    parser.add_argument(
        "--output-dir",
        default="paper_artifacts/cpu_mnist",
        help="Directory where paper-style figures and tables will be written",
    )
    args = parser.parse_args()

    summary = generate_paper_figures(Path(args.results_dir), Path(args.output_dir))
    print(
        "[paper_figures] runs found:",
        ", ".join(summary["runs_found"]) if summary["runs_found"] else "none",
    )
    if summary["seed_aggregates_found"]:
        print("[paper_figures] seed aggregates:", ", ".join(summary["seed_aggregates_found"]))
    for path in summary["generated_files"]:
        print(f"[paper_figures] wrote: {path}")


if __name__ == "__main__":
    main()
