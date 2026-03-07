# Benchmarks and experimental plan

## Included baselines

- SGD
- Adam
- AdamW (+ optional exponential scheduler)
- NeuroPlastic Optimizer (rule-based)
- NeuroPlastic ablation (grad-only)

## Current benchmark tasks

1. MNIST / FashionMNIST sanity checks
2. CIFAR-10 compact CNN benchmark
3. Text benchmark placeholder for subsequent release

## Reproducibility standards

- config-driven runs from YAML,
- fixed seeds,
- metric JSON artifacts,
- checkpoint outputs,
- post-hoc plotting script.

## Suggested ablation matrix

- remove activity traces,
- remove parameter memory,
- disable homeostatic stabilizer,
- switch to grad-only mode,
- layerwise vs parameterwise modulation toggles.

## Sample result schema

Results are saved as JSON with:

- per-epoch train/test loss and accuracy,
- serialized experiment config,
- summary JSON including best test accuracy and final test loss.
