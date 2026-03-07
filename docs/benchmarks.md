# Benchmarks and experimental plan

## Included optimizers

- NeuroPlastic (`rule_based`)
- NeuroPlastic (`ablation_grad_only`)
- SGD
- Adam
- AdamW (optional scheduler)

## Implemented benchmark tasks

1. MNIST and FashionMNIST sanity checks.
2. CIFAR-10 compact CNN benchmark.
3. Synthetic MNIST smoke dataset for CI-like quick checks.
4. Text benchmark scaffolding reserved for the next release.

## Reproducibility protocol

- YAML config driven execution (`configs/`).
- Fixed random seed for Python, NumPy, and torch backends.
- Run-specific metrics and summary JSON artifacts.
- Model checkpoint persistence.

Artifact naming convention:

- `results/<run>_<dataset>_<optimizer>_metrics.json`
- `results/<run>_<dataset>_<optimizer>_summary.json`
- `checkpoints/<run>_<dataset>_<optimizer>_model.pt`

## Suggested ablation matrix

- remove activity traces,
- remove parameter memory,
- disable homeostatic stabilization,
- switch to `ablation_grad_only`,
- toggle layerwise versus parameterwise modulation.

## Experimental claims this layout supports

With sufficient runs and hardware, this framework is structured to measure:

- convergence speed and final accuracy,
- stability under bounded updates,
- adaptation sensitivity to plasticity signals,
- contribution of each component via ablations.
