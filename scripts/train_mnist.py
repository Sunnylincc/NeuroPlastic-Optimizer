from __future__ import annotations

import argparse

from scripts._bootstrap import bootstrap_src_path


def main() -> None:
    bootstrap_src_path()
    from neuroplastic_optimizer.training.runner import run_experiment

    parser = argparse.ArgumentParser(description="Run MNIST NeuroPlastic training")
    parser.add_argument("--config", default="configs/mnist/neuroplastic.yaml")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
