# NeuroPlastic Optimizer

NeuroPlastic Optimizer is a research-engineering framework for synaptic-plasticity-inspired optimization in PyTorch. It augments gradient-based updates with adaptive plasticity coefficients computed from local activity traces, gradient statistics, and parameter-state memory, while enforcing homeostatic stabilization to keep updates bounded.

## Positioning

This repository targets an applied-research use case:

- **Biological inspiration** is encoded as algorithmic signals (activity, memory, homeostatic control).
- **Engineering discipline** is prioritized through modular components, reproducible configs, and baseline comparisons.
- **Scientific utility** is supported by ablation-ready switches and consistent experiment artifacts.

The project does **not** claim biological fidelity or neuroscience simulation.

## Method overview

NeuroPlastic updates follow:

\[
\theta_{t+1} = \theta_t - \eta \cdot \alpha_t \odot g_t
\]

where `alpha_t` combines:

1. local activity traces (EMA of `|g_t|`),
2. gradient magnitude signal,
3. parameter-state memory (momentum/variance history),
4. bounded homeostatic stabilization.

### Plasticity modes

- `rule_based` (default): weighted fusion of activity, gradient, and memory signals.
- `ablation_grad_only`: gradient-driven ablation mode for controlled comparison.

## Architecture

![NeuroPlastic Optimizer architecture](assets/neuroplastic_optimizer_architecture.svg)

*NeuroPlastic Optimizer augments gradient-based learning with synaptic-plasticity-inspired adaptive modulation. Local neural activity traces, gradient signals, and parameter-state memory are integrated by a plasticity encoder and controller to compute adaptive plasticity coefficients, while a homeostatic stabilization module constrains update magnitude for stable training.*

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart

### API usage

```python
from neuroplastic_optimizer import NeuroPlasticOptimizer

optimizer = NeuroPlasticOptimizer(model.parameters(), lr=1e-3)
```

### Run a single experiment

```bash
python -m neuroplastic_optimizer.training.runner --config configs/mnist/neuroplastic.yaml
```

Or use convenience scripts:

```bash
python scripts/train_mnist.py
python scripts/train_cifar10.py
```

## Benchmarks

Run the MNIST benchmark sweep (NeuroPlastic + ablation + SGD/Adam/AdamW):

```bash
python scripts/benchmark_all.py
```

Plot test accuracy curves:

```bash
python scripts/plot_results.py --result-files \
  results/neuroplastic_mnist_neuroplastic_metrics.json \
  results/ablation_grad_only_mnist_neuroplastic_metrics.json \
  results/adamw_mnist_adamw_metrics.json \
  results/adam_mnist_adam_metrics.json \
  results/sgd_mnist_sgd_metrics.json
```

## Reproducibility and artifacts

Every run writes:

- `results/<run>_<dataset>_<optimizer>_metrics.json`
- `results/<run>_<dataset>_<optimizer>_summary.json`
- `checkpoints/<run>_<dataset>_<optimizer>_model.pt`

## Repository layout

```text
src/neuroplastic_optimizer/
  optimizer.py          # NeuroPlastic optimizer update rule
  plasticity.py         # plasticity coefficient computation
  traces.py             # activity trace extraction
  state.py              # parameter-state memory
  stabilization.py      # homeostatic constraints
  training/             # experiment runner, config parsing, data
  models/               # benchmark models
configs/                # YAML experiment definitions
scripts/                # benchmark orchestration and plotting
docs/                   # method, architecture, benchmark plan
tests/                  # unit and integration tests
assets/                 # figures
```

## Current limitations

- Current benchmarks are lightweight by design (MNIST/FashionMNIST/CIFAR-10).
- Text benchmark integration is scaffolded but not finalized.
- Distributed training and AMP orchestration are future work.

## Roadmap

- [ ] Finalize compact text classification benchmark.
- [ ] Add distributed and mixed-precision training support.
- [ ] Add richer diagnostics (alpha distribution, update norm traces).
- [ ] Publish reproducible benchmark tables with ablation summaries.

## Citation

```bibtex
@software{neuroplastic_optimizer_2026,
  title={NeuroPlastic Optimizer},
  author={NeuroPlastic Optimizer Contributors},
  year={2026},
  url={https://github.com/<org>/NeuroPlastic-Optimizer}
}
```
