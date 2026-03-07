# NeuroPlastic Optimizer

NeuroPlastic Optimizer is a PyTorch-first optimization framework that augments gradient descent with synaptic-plasticity-inspired adaptive modulation. It integrates local activity traces, gradient signals, and parameter-state memory to compute bounded per-parameter (or layer-wise) plasticity coefficients under homeostatic stabilization.

## Why this project exists

Modern optimizers are powerful but often rely on limited local signals. This repository explores a practical middle ground between biologically inspired learning principles and production-minded deep learning systems:

- **biological inspiration**: local activity, memory, and homeostatic pressure,
- **engineering implementation**: modular, ablation-ready components for controlled experimentation,
- **research utility**: reproducible baselines and benchmark scripts for fair comparisons.

## Method overview

The optimizer follows:

\[
\theta_{t+1} = \theta_t - \eta \cdot \alpha_t \odot g_t
\]

Where `alpha_t` is computed from:

1. activity traces (EMA of |gradient|),
2. gradient magnitude statistics,
3. parameter-state memory (momentum/variance history),
4. bounded homeostatic stabilization.

Supported plasticity modes:

- `rule_based` (default): weighted fusion of activity + gradient + memory signals,
- `ablation_grad_only`: reduced variant for controlled comparisons.

## Architecture figure

![NeuroPlastic Optimizer architecture](assets/neuroplastic_optimizer_architecture.svg)

*NeuroPlastic Optimizer augments gradient-based learning with synaptic-plasticity-inspired adaptive modulation. Local neural activity traces, gradient signals, and parameter-state memory are integrated by a plasticity encoder and controller to compute adaptive plasticity coefficients, while a homeostatic stabilization module constrains update magnitude for stable training.*

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart

### Minimal usage in code

```python
from neuroplastic_optimizer import NeuroPlasticOptimizer

optimizer = NeuroPlasticOptimizer(model.parameters(), lr=1e-3)
```

### Run MNIST experiment

```bash
python -m neuroplastic_optimizer.training.runner --config configs/mnist/neuroplastic.yaml
```

## Benchmark usage

Run baseline comparisons (MNIST):

```bash
python scripts/benchmark_all.py
```

Plot results:

```bash
python scripts/plot_results.py --result-files \
  results/neuroplastic_metrics.json \
  results/adamw_metrics.json \
  results/sgd_metrics.json
```

## Repository layout

```text
src/neuroplastic_optimizer/
  optimizer.py          # optimizer update loop
  plasticity.py         # plasticity coefficient computation
  traces.py             # local activity trace extraction
  state.py              # parameter memory state
  stabilization.py      # homeostatic constraints
  training/             # experiment runner, config mapping, data loading
  models/               # benchmark models
configs/                # YAML experiment definitions
scripts/                # benchmark and plotting scripts
docs/                   # architecture, method, benchmark plans
tests/                  # unit + smoke tests
assets/                 # figures
```

## Design principles

- modular boundaries around biological abstractions,
- reproducible and config-driven experiments,
- explicit ablation support,
- stable defaults with bounded update dynamics,
- minimal dependency footprint.

## Current limitations

- v0.1 focuses on small-to-mid scale supervised benchmarks,
- distributed training is not yet implemented,
- text benchmark configs are scaffolded but not fully shipped,
- no claim of biological realism beyond algorithmic inspiration.

## Results

`results/` stores JSON metrics and summary files.

Current release includes benchmark pipelines and plotting utilities; full benchmark tables can be populated by running the provided scripts on your hardware.

## Roadmap

- [ ] Add lightweight text classification benchmark suite.
- [ ] Add distributed/mixed-precision training support.
- [ ] Add richer diagnostics (update norms, alpha distributions, adaptation dynamics).
- [ ] Add optimizer-state checkpoint resume tests.
- [ ] Publish reproducible benchmark table and ablation report.

## Citation

```bibtex
@software{neuroplastic_optimizer_2026,
  title={NeuroPlastic Optimizer},
  author={NeuroPlastic Optimizer Contributors},
  year={2026},
  url={https://github.com/<org>/NeuroPlastic-Optimizer}
}
```
