from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-files", nargs="+", required=True)
    parser.add_argument("--out", default="results/benchmark_plot.png")
    args = parser.parse_args()

    plt.figure(figsize=(8, 5))
    for file in args.result_files:
        with open(file, encoding="utf-8") as f:
            payload = json.load(f)
        acc = [x["accuracy"] for x in payload["test"]]
        plt.plot(acc, marker="o", label=Path(file).stem)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("NeuroPlastic Optimizer benchmark comparison")
    plt.legend()
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
