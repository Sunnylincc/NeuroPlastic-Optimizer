import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch missing")


@pytest.mark.parametrize(
    "mode,layerwise,parameterwise,expect_uniform",
    [
        ("rule_based", True, True, False),
        ("rule_based", True, False, True),
        ("rule_based", False, False, True),
        ("rule_based", False, True, False),
        ("ablation_grad_only", True, True, False),
        ("ablation_grad_only", True, False, True),
        ("ablation_grad_only", False, False, True),
        ("ablation_grad_only", False, True, False),
    ],
)
def test_plasticity_mode_and_granularity_combinations(
    mode, layerwise, parameterwise, expect_uniform
):
    import torch

    from neuroplastic_optimizer.plasticity import (
        PlasticityConfig,
        PlasticityMode,
        compute_plasticity,
    )

    grad = torch.tensor([0.0, 25.0, 0.0, 0.0, 1e6], dtype=torch.float32)
    trace = torch.tensor([1.0, 0.0, 50.0, 0.0, 0.5], dtype=torch.float32)
    momentum = torch.tensor([0.0, 2.0, 0.0, 5.0, 0.0], dtype=torch.float32)
    variance = torch.tensor([1.0, 0.1, 2.0, 0.5, 4.0], dtype=torch.float32)

    cfg = PlasticityConfig(
        mode=PlasticityMode(mode),
        layerwise=layerwise,
        parameterwise=parameterwise,
        min_alpha=0.05,
        max_alpha=4.0,
    )

    alpha = compute_plasticity(grad, trace, momentum, variance, cfg)

    assert alpha.shape == grad.shape
    if expect_uniform:
        assert torch.unique(alpha).numel() == 1
    else:
        assert torch.unique(alpha).numel() > 1


@pytest.mark.parametrize(
    "grad,trace,momentum,variance,min_alpha,max_alpha,expected",
    [
        (
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            0.3,
            2.0,
            "min",
        ),
        (
            [1000.0, 0.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            0.1,
            1.2,
            "max",
        ),
    ],
)
def test_plasticity_alpha_clamp_boundaries(
    grad, trace, momentum, variance, min_alpha, max_alpha, expected
):
    import torch

    from neuroplastic_optimizer.plasticity import PlasticityConfig, compute_plasticity

    cfg = PlasticityConfig(
        layerwise=False,
        parameterwise=True,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
    )

    alpha = compute_plasticity(
        grad=torch.tensor(grad),
        activity_trace=torch.tensor(trace),
        momentum=torch.tensor(momentum),
        variance=torch.tensor(variance),
        config=cfg,
    )

    if expected == "min":
        assert torch.allclose(alpha, torch.full_like(alpha, min_alpha))
    else:
        assert torch.isclose(alpha.max(), torch.tensor(max_alpha))
        assert torch.all(alpha <= max_alpha)


def test_rule_based_parameterwise_path_does_not_collapse_to_constant_one():
    import torch

    from neuroplastic_optimizer.plasticity import PlasticityConfig, compute_plasticity

    grad = torch.tensor([1.0, 2.0, 8.0, 16.0], dtype=torch.float32)
    trace = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
    momentum = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32)
    variance = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    cfg = PlasticityConfig(
        layerwise=True,
        parameterwise=True,
        min_alpha=0.2,
        max_alpha=2.0,
    )
    alpha = compute_plasticity(grad, trace, momentum, variance, cfg)

    assert torch.unique(alpha).numel() > 1
    assert not torch.allclose(alpha, torch.ones_like(alpha))
