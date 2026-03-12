from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import yaml


def _slug(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _parse_alpha_range(value: str) -> tuple[float, float]:
    left, right = value.split(":", maxsplit=1)
    return float(left), float(right)


def _run_name(
    lr: float,
    alpha_range: tuple[float, float],
    max_update_norm: float,
    adaptation_rate: float,
    epochs: int,
) -> str:
    return (
        "debug_np"
        f"_lr{_slug(lr)}"
        f"_a{_slug(alpha_range[0])}_{_slug(alpha_range[1])}"
        f"_norm{_slug(max_update_norm)}"
        f"_adapt{_slug(adaptation_rate)}"
        f"_e{epochs}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _build_config(
    base_config_path: Path,
    output_config_path: Path,
    *,
    results_dir: Path,
    checkpoints_dir: Path,
    data_root: Path,
    lr: float,
    alpha_range: tuple[float, float],
    max_update_norm: float,
    adaptation_rate: float,
    epochs: int,
) -> str:
    payload = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    experiment = dict(payload.get("experiment", {}))
    plasticity = dict(payload.get("plasticity", {}))
    homeostatic = dict(payload.get("homeostatic", {}))
    run_name = _run_name(lr, alpha_range, max_update_norm, adaptation_rate, epochs)

    experiment.update(
        {
            "run_name": run_name,
            "lr": lr,
            "epochs": epochs,
            "num_workers": 0,
            "device": "cpu",
            "data_root": str(data_root),
            "download": True,
            "output_dir": str(results_dir),
            "checkpoint_dir": str(checkpoints_dir),
        }
    )
    plasticity.update({"min_alpha": alpha_range[0], "max_alpha": alpha_range[1]})
    homeostatic.update({"max_update_norm": max_update_norm, "adaptation_rate": adaptation_rate})
    payload["experiment"] = experiment
    payload["plasticity"] = plasticity
    payload["homeostatic"] = homeostatic
    output_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return run_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact NeuroPlastic MNIST debug sweep")
    parser.add_argument("--base-config", default="configs/mnist/neuroplastic.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="paper_artifacts/cpu_mnist_debug")
    parser.add_argument("--epochs-list", nargs="+", type=int, default=[10])
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-1, 1e-2, 1e-3, 1e-4])
    parser.add_argument("--alpha-ranges", nargs="+", default=["0.2:2.0"])
    parser.add_argument("--max-update-norms", nargs="+", type=float, default=[1.0])
    parser.add_argument("--adaptation-rates", nargs="+", type=float, default=[0.01])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = repo_root / args.base_config
    results_dir = (repo_root / args.results_dir).resolve()
    checkpoints_dir = (repo_root / args.checkpoints_dir).resolve()
    data_root = (repo_root / args.data_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    generated_configs_dir = output_dir / "_generated_configs"
    summary_path = output_dir / "debug_summary.csv"

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_configs_dir.mkdir(parents=True, exist_ok=True)

    alpha_ranges = [_parse_alpha_range(value) for value in args.alpha_ranges]
    rows: list[dict[str, Any]] = []

    for epochs, lr, alpha_range, max_update_norm, adaptation_rate in product(
        args.epochs_list,
        args.lrs,
        alpha_ranges,
        args.max_update_norms,
        args.adaptation_rates,
    ):
        run_name = _run_name(lr, alpha_range, max_update_norm, adaptation_rate, epochs)
        config_path = generated_configs_dir / f"{run_name}.yaml"
        run_name = _build_config(
            base_config_path,
            config_path,
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
            data_root=data_root,
            lr=lr,
            alpha_range=alpha_range,
            max_update_norm=max_update_norm,
            adaptation_rate=adaptation_rate,
            epochs=epochs,
        )
        command = [
            sys.executable,
            "-m",
            "neuroplastic_optimizer.training.runner",
            "--config",
            str(config_path),
        ]
        print("[debug_mnist] running:", " ".join(command))
        subprocess.run(command, cwd=repo_root, check=True)

        metrics_path = results_dir / f"{run_name}_mnist_neuroplastic_metrics.json"
        summary_json_path = results_dir / f"{run_name}_mnist_neuroplastic_summary.json"
        metrics = _load_json(metrics_path)
        summary = _load_json(summary_json_path)
        diagnostics = (metrics.get("optimizer_diagnostics") or [{}])[-1]

        rows.append(
            {
                "run_name": run_name,
                "epochs": epochs,
                "lr": lr,
                "min_alpha": alpha_range[0],
                "max_alpha": alpha_range[1],
                "max_update_norm": max_update_norm,
                "adaptation_rate": adaptation_rate,
                "best_test_accuracy": summary.get("best_test_accuracy"),
                "final_test_accuracy": metrics["test"][-1]["accuracy"]
                if metrics.get("test")
                else None,
                "final_test_loss": summary.get("last_test_loss"),
                "alpha_mean": diagnostics.get("alpha_mean"),
                "alpha_median": diagnostics.get("alpha_median"),
                "alpha_fraction_at_min": diagnostics.get("alpha_fraction_at_min"),
                "alpha_fraction_at_max": diagnostics.get("alpha_fraction_at_max"),
                "raw_gradient_norm": diagnostics.get("raw_gradient_norm"),
                "effective_update_norm": diagnostics.get("effective_update_norm"),
                "effective_to_gradient_norm_ratio": diagnostics.get(
                    "effective_to_gradient_norm_ratio"
                ),
                "stabilization_norm_ratio": diagnostics.get("stabilization_norm_ratio"),
            }
        )

    rows.sort(
        key=lambda row: (
            -(row["best_test_accuracy"] or 0.0),
            row["final_test_loss"] or float("inf"),
        )
    )
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[debug_mnist] wrote summary: {summary_path}")
    print("[debug_mnist] top configurations:")
    for row in rows[:5]:
        print(
            "[debug_mnist] "
            f"{row['run_name']} | best_acc={row['best_test_accuracy']:.4f} | "
            f"final_acc={row['final_test_accuracy']:.4f} | "
            f"alpha_mean={row['alpha_mean']:.4f} | "
            f"update/grad={row['effective_to_gradient_norm_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
