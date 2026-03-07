import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch missing")


def test_plasticity_bounds_respected():
    import torch

    from neuroplastic_optimizer.plasticity import PlasticityConfig, compute_plasticity

    grad = torch.ones(8)
    trace = torch.ones(8) * 2
    momentum = torch.ones(8) * 0.5
    variance = torch.ones(8) * 0.25
    cfg = PlasticityConfig(min_alpha=0.3, max_alpha=0.7)

    alpha = compute_plasticity(grad, trace, momentum, variance, cfg)

    assert torch.all(alpha >= 0.3)
    assert torch.all(alpha <= 0.7)


def test_ablation_mode_uses_gradient_only():
    import torch

    from neuroplastic_optimizer.plasticity import (
        PlasticityConfig,
        PlasticityMode,
        compute_plasticity,
    )

    grad = torch.tensor([1.0, 2.0, 3.0])
    zeros = torch.zeros_like(grad)
    cfg = PlasticityConfig(mode=PlasticityMode.ABLATION_GRAD_ONLY)

    alpha = compute_plasticity(grad, zeros, zeros, torch.ones_like(grad), cfg)

    assert alpha.shape == grad.shape
