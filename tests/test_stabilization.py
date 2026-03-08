import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch missing")


def test_stabilizer_clips_large_norm_before_gain_scaling():
    import torch

    from neuroplastic_optimizer.stabilization import HomeostaticConfig, HomeostaticStabilizer

    cfg = HomeostaticConfig(max_update_norm=1.0, adaptation_rate=0.0)
    stabilizer = HomeostaticStabilizer(cfg)
    update = torch.ones(100) * 10

    stabilized = stabilizer.stabilize(update)

    assert torch.isclose(stabilized.norm(), torch.tensor(1.0), atol=1e-4)


@pytest.mark.parametrize(
    "target_rms,adaptation_rate,update,expected_gain",
    [
        (0.0, 10.0, [10.0, 10.0, 10.0, 10.0], 0.5),
        (1.0, 1.0, [0.0, 0.0, 0.0, 0.0], 1.5),
    ],
)
def test_stabilizer_gain_is_clamped(target_rms, adaptation_rate, update, expected_gain):
    import torch

    from neuroplastic_optimizer.stabilization import HomeostaticConfig, HomeostaticStabilizer

    tensor_update = torch.tensor(update)
    cfg = HomeostaticConfig(
        max_update_norm=1e9,
        target_rms=target_rms,
        adaptation_rate=adaptation_rate,
    )
    stabilizer = HomeostaticStabilizer(cfg)

    stabilized = stabilizer.stabilize(tensor_update)

    assert torch.allclose(stabilized, tensor_update * expected_gain, atol=1e-6)
