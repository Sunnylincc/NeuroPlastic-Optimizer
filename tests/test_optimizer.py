import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch missing")


def test_optimizer_step_updates_parameters():
    import torch
    from torch import nn

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer

    model = nn.Linear(4, 2)
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    criterion = nn.MSELoss()

    opt = NeuroPlasticOptimizer(model.parameters(), lr=1e-2)
    before = model.weight.detach().clone()

    loss = criterion(model(x), y)
    loss.backward()
    opt.step()

    after = model.weight.detach().clone()
    assert not torch.allclose(before, after)


def test_optimizer_combines_weight_decay_and_plastic_update_correctly():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
    from neuroplastic_optimizer.plasticity import PlasticityConfig, PlasticityMode
    from neuroplastic_optimizer.stabilization import HomeostaticConfig

    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    param.grad = torch.tensor([0.5, -0.25], dtype=torch.float32)

    lr = 0.1
    wd = 0.2
    opt = NeuroPlasticOptimizer(
        [param],
        lr=lr,
        weight_decay=wd,
        plasticity_config=PlasticityConfig(
            mode=PlasticityMode.ABLATION_GRAD_ONLY,
            layerwise=True,
            parameterwise=False,
            min_alpha=0.0,
            max_alpha=10.0,
        ),
        homeostatic_config=HomeostaticConfig(
            max_update_norm=1e9,
            adaptation_rate=0.0,
        ),
    )

    before = param.detach().clone()
    grad = param.grad.detach().clone()
    alpha = torch.tensor(1.0)

    opt.step()

    expected_update = alpha * grad + wd * before
    expected_after = before - lr * expected_update
    assert torch.allclose(param.detach(), expected_after, atol=1e-6)


def test_short_regression_run_does_not_diverge_on_synthetic_data():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer

    torch.manual_seed(7)

    x = torch.randn(128, 4)
    true_w = torch.tensor([[1.5], [-2.0], [0.5], [3.0]])
    y = x @ true_w + 0.05 * torch.randn(128, 1)

    model = torch.nn.Linear(4, 1)
    loss_fn = torch.nn.MSELoss()
    opt = NeuroPlasticOptimizer(model.parameters(), lr=5e-2, weight_decay=1e-3)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    final_loss = losses[-1]
    weight_norm = float(model.weight.detach().norm().item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert final_loss < losses[0]
    assert final_loss < 6.0
    assert 0.01 < weight_norm < 20.0
