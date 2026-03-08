from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_CONFIGS = [
    "configs/mnist/neuroplastic.yaml",
    "configs/mnist/ablation_grad_only.yaml",
    "configs/mnist/adamw.yaml",
    "configs/mnist/adam.yaml",
    "configs/mnist/sgd.yaml",
]


def _build_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    entries = ["src"]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _artifact_stem(config_path: str, summary: dict[str, Any]) -> str:
    run_name = summary.get("run_name") or Path(config_path).stem
    dataset = summary.get("dataset", "unknown")
    optimizer = summary.get("optimizer", "unknown")
    return f"{run_name}_{dataset}_{optimizer}"


def _load_events(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _collect_metrics(configs: list[str], failures: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        summary_files = sorted(Path("results").glob(f"{Path(cfg).stem}*_summary.json"))
        if not summary_files:
            continue
        summary = json.loads(summary_files[-1].read_text(encoding="utf-8"))
        stem = _artifact_stem(cfg, summary)
        events = _load_events(Path("results") / f"{stem}_events.jsonl")
        latest_event = events[-1] if events else {}
        rows.append(
            {
                "config": cfg,
                "best_test_accuracy": summary.get("best_test_accuracy"),
                "last_test_loss": summary.get("last_test_loss"),
                "last_epoch": latest_event.get("epoch"),
                "last_lr": latest_event.get("lr"),
            }
        )

    if failures:
        print("[benchmark] failed configs:")
        for cfg in failures:
            print(f"  - {cfg}")

    if rows:
        print("[benchmark] key metrics:")
        header = "config | best_test_accuracy | last_test_loss | last_epoch | last_lr"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['config']} | {row['best_test_accuracy']} | {row['last_test_loss']} | "
                f"{row['last_epoch']} | {row['last_lr']}"
            )


def run_all(configs: list[str], stop_on_error: bool = True) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    failures: list[str] = []
    for cfg in configs:
        print(f"[benchmark] running: {cfg}")
        proc = subprocess.run(
            ["python", "-m", "neuroplastic_optimizer.training.runner", "--config", cfg],
            check=False,
            env=_build_env(),
        )
        if proc.returncode != 0:
            failures.append(cfg)
            if stop_on_error:
                break

    _collect_metrics(configs, failures)

    if failures:
        raise SystemExit(f"Benchmark run failed for configs: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run configured benchmark sweep")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--keep-going", action="store_true", help="Continue after failures")
    args = parser.parse_args()
    run_all(configs=args.configs, stop_on_error=not args.keep_going)
