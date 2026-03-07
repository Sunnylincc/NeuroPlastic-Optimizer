from __future__ import annotations

import subprocess
from pathlib import Path

CONFIGS = [
    "configs/mnist/neuroplastic.yaml",
    "configs/mnist/adamw.yaml",
    "configs/mnist/sgd.yaml",
]


def run_all() -> None:
    for cfg in CONFIGS:
        print(f"Running benchmark: {cfg}")
        subprocess.run(
            ["python", "-m", "neuroplastic_optimizer.training.runner", "--config", cfg],
            check=True,
        )


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    run_all()
