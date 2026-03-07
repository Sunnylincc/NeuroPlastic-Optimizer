# Results artifacts

Each benchmark run writes structured artifacts:

- `*_metrics.json`: per-epoch train/test metrics
- `*_summary.json`: condensed run summary
- `*.png`: generated benchmark plots

Naming convention:

- `<run_name>_<dataset>_<optimizer>_metrics.json`
- `<run_name>_<dataset>_<optimizer>_summary.json`

Populate these artifacts via:

```bash
python scripts/benchmark_all.py
```
