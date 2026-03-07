from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

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

    if failures:
        raise SystemExit(f"Benchmark run failed for configs: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run configured benchmark sweep")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--keep-going", action="store_true", help="Continue after failures")
    args = parser.parse_args()
    run_all(configs=args.configs, stop_on_error=not args.keep_going)
