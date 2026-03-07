import importlib.util
import json
import subprocess
from pathlib import Path


def test_smoke_train_config(tmp_path):
    if not all(importlib.util.find_spec(m) for m in ["torch", "torchvision", "typer", "yaml"]):
        return

    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(
        """
experiment:
  dataset: mnist
  batch_size: 64
  epochs: 1
  lr: 0.001
  weight_decay: 0.0
  optimizer: neuroplastic
  seed: 1
  num_workers: 0
  output_dir: results
  checkpoint_dir: checkpoints
  device: cpu

plasticity:
  mode: ablation_grad_only

homeostatic:
  max_update_norm: 1.0
""",
        encoding="utf-8",
    )

    subprocess.run(
        ["python", "-m", "neuroplastic_optimizer.training.runner", "--config", str(cfg_path)],
        check=True,
    )

    summary_path = Path("results") / f"{cfg_path.stem}_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "best_test_acc" in summary
