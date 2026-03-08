import importlib.util
import json
import sys
from pathlib import Path


def test_benchmark_script_creates_output_dirs(monkeypatch, tmp_path):
    script_dir = Path(__file__).resolve().parents[1]
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    benchmark_all = importlib.util.find_spec("scripts.benchmark_all")
    assert benchmark_all is not None
    module = __import__("scripts.benchmark_all", fromlist=["run_all"])

    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd, check, env):
        calls.append(cmd)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.run_all(configs=["configs/mnist/neuroplastic.yaml"], stop_on_error=True)

    assert Path("results").exists()
    assert Path("checkpoints").exists()
    assert calls



def test_benchmark_script_collects_jsonl_and_summary(monkeypatch, tmp_path, capsys):
    script_dir = Path(__file__).resolve().parents[1]
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    module = __import__("scripts.benchmark_all", fromlist=["run_all"])

    monkeypatch.chdir(tmp_path)
    Path("results").mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": "neuroplastic",
        "dataset": "mnist",
        "optimizer": "neuroplastic",
        "best_test_accuracy": 0.93,
        "last_test_loss": 0.12,
    }
    (Path("results") / "neuroplastic_mnist_neuroplastic_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (Path("results") / "neuroplastic_mnist_neuroplastic_events.jsonl").write_text(
        json.dumps(
            {
                "epoch": 3,
                "train_loss": 0.1,
                "train_acc": 0.95,
                "test_loss": 0.12,
                "test_acc": 0.93,
                "lr": 0.001,
                "time_per_epoch": 0.3,
                "device": "cpu",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_run(cmd, check, env):
        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.run_all(configs=["configs/mnist/neuroplastic.yaml"], stop_on_error=True)

    out = capsys.readouterr().out
    assert "[benchmark] key metrics:" in out
    assert "best_test_accuracy" in out
    assert "0.93" in out
