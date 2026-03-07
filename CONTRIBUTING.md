# Contributing

Thanks for your interest in NeuroPlastic Optimizer.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quality checks

```bash
ruff check .
pytest
```

## Contribution guidelines

- Keep changes modular and well-tested.
- For optimizer logic changes, include at least one unit test and one smoke-level integration check.
- Update docs/configs when changing training behaviors.
