import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch missing")


def test_stabilizer_clips_large_norm():
    import torch

    from neuroplastic_optimizer.stabilization import HomeostaticConfig, HomeostaticStabilizer

    stabilizer = HomeostaticStabilizer(HomeostaticConfig(max_update_norm=1.0))
    update = torch.ones(100) * 10
    stabilized = stabilizer.stabilize(update)
    assert stabilized.norm() <= 1.1
