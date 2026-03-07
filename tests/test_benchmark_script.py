import importlib.util
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
