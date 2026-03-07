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
