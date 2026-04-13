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


def test_optimizer_collects_lightweight_diagnostics():
    import torch
    from torch import nn

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer

    model = nn.Linear(4, 2)
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    criterion = nn.MSELoss()

    opt = NeuroPlasticOptimizer(model.parameters(), lr=1e-2)
    opt.reset_diagnostics()
    loss = criterion(model(x), y)
    loss.backward()
    opt.step()

    diagnostics = opt.collect_diagnostics()
    required = {
        "alpha_mean",
        "alpha_median",
        "alpha_min",
        "alpha_max",
        "alpha_fraction_near_min",
        "alpha_fraction_near_max",
        "raw_gradient_norm",
        "raw_update_norm",
        "effective_update_norm",
        "effective_to_gradient_norm_ratio",
        "stabilization_norm_ratio",
    }
    assert required.issubset(diagnostics.keys())
    assert diagnostics["raw_gradient_norm"] > 0


def test_full_neuroplastic_warmup_gate_matches_grad_only_on_first_warmup_epoch():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
    from neuroplastic_optimizer.plasticity import PlasticityConfig, PlasticityMode
    from neuroplastic_optimizer.stabilization import HomeostaticConfig

    def make_param() -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))

    grad = torch.tensor([0.5, -0.25], dtype=torch.float32)
    homeostatic = HomeostaticConfig(max_update_norm=1e9, adaptation_rate=0.0)

    full_param = make_param()
    grad_only_param = make_param()
    full_param.grad = grad.clone()
    grad_only_param.grad = grad.clone()

    full = NeuroPlasticOptimizer(
        [full_param],
        lr=0.1,
        plasticity_config=PlasticityConfig(
            mode=PlasticityMode.RULE_BASED,
            warmup_epochs=1,
            plasticity_scale=1.0,
            min_alpha=0.0,
            max_alpha=10.0,
        ),
        homeostatic_config=homeostatic,
    )
    grad_only = NeuroPlasticOptimizer(
        [grad_only_param],
        lr=0.1,
        plasticity_config=PlasticityConfig(
            mode=PlasticityMode.ABLATION_GRAD_ONLY,
            min_alpha=0.0,
            max_alpha=10.0,
        ),
        homeostatic_config=homeostatic,
    )

    full.set_epoch(1)
    full.step()
    grad_only.step()

    assert torch.allclose(full_param.detach(), grad_only_param.detach(), atol=1e-6)


def test_plasticity_scale_zero_reduces_full_mode_to_grad_only_path():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
    from neuroplastic_optimizer.plasticity import PlasticityConfig, PlasticityMode
    from neuroplastic_optimizer.stabilization import HomeostaticConfig

    def make_param() -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.tensor([0.25, -0.75], dtype=torch.float32))

    grad = torch.tensor([0.4, -0.1], dtype=torch.float32)
    homeostatic = HomeostaticConfig(max_update_norm=1e9, adaptation_rate=0.0)

    full_zero = make_param()
    grad_only_param = make_param()
    full_scaled = make_param()
    full_zero.grad = grad.clone()
    grad_only_param.grad = grad.clone()
    full_scaled.grad = grad.clone()

    zero_scale_opt = NeuroPlasticOptimizer(
        [full_zero],
        lr=0.05,
        plasticity_config=PlasticityConfig(
            mode=PlasticityMode.RULE_BASED,
            plasticity_scale=0.0,
            min_alpha=0.0,
            max_alpha=10.0,
        ),
        homeostatic_config=homeostatic,
    )
    grad_only_opt = NeuroPlasticOptimizer(
        [grad_only_param],
        lr=0.05,
        plasticity_config=PlasticityConfig(
            mode=PlasticityMode.ABLATION_GRAD_ONLY,
            min_alpha=0.0,
            max_alpha=10.0,
        ),
        homeostatic_config=homeostatic,
    )
    scaled_opt = NeuroPlasticOptimizer(
        [full_scaled],
        lr=0.05,
        plasticity_config=PlasticityConfig(
            mode=PlasticityMode.RULE_BASED,
            plasticity_scale=2.0,
            min_alpha=0.0,
            max_alpha=10.0,
        ),
        homeostatic_config=homeostatic,
    )

    zero_scale_opt.step()
    grad_only_opt.step()
    scaled_opt.step()

    assert torch.allclose(full_zero.detach(), grad_only_param.detach(), atol=1e-6)
    assert not torch.allclose(full_scaled.detach(), grad_only_param.detach(), atol=1e-6)


def test_bounded_residual_caps_extra_plasticity_contribution():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
    from neuroplastic_optimizer.plasticity import PlasticityConfig, PlasticityMode
    from neuroplastic_optimizer.stabilization import HomeostaticConfig

    base_param = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
    residual_param = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
    grad = torch.tensor([0.5, -0.25], dtype=torch.float32)
    base_param.grad = grad.clone()
    residual_param.grad = grad.clone()

    homeostatic = HomeostaticConfig(max_update_norm=1e9, adaptation_rate=0.0)
    common = dict(
        lr=0.1,
        plasticity_config=PlasticityConfig(
            plasticity_scale=10.0,
            bounded_residual=True,
            residual_max_ratio=0.5,
            min_alpha=0.0,
            max_alpha=10.0,
        ),
        homeostatic_config=homeostatic,
    )

    residual_opt = NeuroPlasticOptimizer([residual_param], **common)
    residual_opt.step()

    base_opt = NeuroPlasticOptimizer(
        [base_param],
        lr=0.1,
        plasticity_config=PlasticityConfig(
            mode=PlasticityMode.ABLATION_GRAD_ONLY,
            min_alpha=0.0,
            max_alpha=10.0,
        ),
        homeostatic_config=homeostatic,
    )
    base_opt.step()

    capped_delta = base_param.detach() - residual_param.detach()
    assert torch.linalg.vector_norm(capped_delta) <= 0.5 * torch.linalg.vector_norm(grad) + 1e-6


def test_layerwise_target_rms_scales_change_parameter_updates():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
    from neuroplastic_optimizer.plasticity import PlasticityConfig, PlasticityMode
    from neuroplastic_optimizer.stabilization import HomeostaticConfig

    params = [
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
    ]
    for param in params:
        param.grad = torch.tensor([0.5], dtype=torch.float32)

    opt = NeuroPlasticOptimizer(
        params,
        lr=0.1,
        plasticity_config=PlasticityConfig(
            mode=PlasticityMode.ABLATION_GRAD_ONLY,
            parameterwise=False,
            min_alpha=1.0,
            max_alpha=1.0,
        ),
        homeostatic_config=HomeostaticConfig(
            max_update_norm=1e9,
            target_rms=1.0,
            adaptation_rate=1.0,
            early_target_rms_scale=0.5,
            middle_target_rms_scale=1.0,
            late_target_rms_scale=1.5,
        ),
    )
    before = [param.detach().clone() for param in params]
    opt.step()

    deltas = [float((old - new.detach()).abs().item()) for old, new in zip(before, params)]
    assert deltas[0] < deltas[1]
    assert deltas[1] <= deltas[2]


def test_hybrid_gain_modulation_stays_secondary_to_base_update():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
    from neuroplastic_optimizer.plasticity import PlasticityConfig
    from neuroplastic_optimizer.stabilization import HomeostaticConfig

    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    param.grad = torch.tensor([0.5, -0.25], dtype=torch.float32)
    grad_norm = torch.linalg.vector_norm(param.grad.detach())

    opt = NeuroPlasticOptimizer(
        [param],
        lr=0.1,
        weight_decay=0.0,
        plasticity_config=PlasticityConfig(
            hybrid_base="adamw",
            modulation_target="gain",
            modulation_scope="all",
            modulation_schedule="constant",
            layer_scope="all",
            phase_scope="full",
            modulation_strength=1.0,
            modulation_max_ratio=0.25,
            min_alpha=0.5,
            max_alpha=1.5,
        ),
        homeostatic_config=HomeostaticConfig(max_update_norm=1e9, adaptation_rate=0.0),
        decoupled_weight_decay=True,
    )
    opt.set_total_epochs(10)
    opt.step()

    diagnostics = opt.collect_diagnostics()
    assert diagnostics["plasticity_delta_norm"] <= 0.25 * diagnostics["raw_update_norm"] + 1e-6
    assert diagnostics["raw_gradient_norm"] <= grad_norm + 1e-6


def test_hybrid_phase_scope_disables_modulation_outside_selected_window():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
    from neuroplastic_optimizer.plasticity import PlasticityConfig
    from neuroplastic_optimizer.stabilization import HomeostaticConfig

    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    param.grad = torch.tensor([0.5, -0.25], dtype=torch.float32)

    opt = NeuroPlasticOptimizer(
        [param],
        lr=0.1,
        weight_decay=0.0,
        plasticity_config=PlasticityConfig(
            hybrid_base="adamw",
            modulation_target="gain",
            layer_scope="all",
            phase_scope="late",
            modulation_strength=1.0,
            modulation_max_ratio=0.25,
            min_alpha=0.5,
            max_alpha=1.5,
        ),
        homeostatic_config=HomeostaticConfig(max_update_norm=1e9, adaptation_rate=0.0),
        decoupled_weight_decay=True,
    )
    opt.set_total_epochs(10)
    opt.set_epoch(2)
    opt.step()

    diagnostics = opt.collect_diagnostics()
    assert diagnostics["modulation_active_fraction"] == pytest.approx(0.0)
    assert diagnostics["plasticity_delta_norm"] == pytest.approx(0.0)


def test_hybrid_classifier_scope_targets_only_classifier_group():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
    from neuroplastic_optimizer.plasticity import PlasticityConfig
    from neuroplastic_optimizer.stabilization import HomeostaticConfig

    params = [
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32)),
    ]
    for param in params:
        param.grad = torch.full_like(param, 0.5)

    opt = NeuroPlasticOptimizer(
        params,
        lr=0.1,
        weight_decay=0.0,
        plasticity_config=PlasticityConfig(
            hybrid_base="adamw",
            modulation_target="gain",
            layer_scope="classifier_only",
            phase_scope="full",
            modulation_strength=1.0,
            modulation_max_ratio=0.25,
            min_alpha=0.5,
            max_alpha=1.5,
        ),
        homeostatic_config=HomeostaticConfig(max_update_norm=1e9, adaptation_rate=0.0),
        decoupled_weight_decay=True,
    )
    opt.set_total_epochs(10)
    opt.set_epoch(10)
    opt.step()

    diagnostics = opt.collect_diagnostics()
    assert diagnostics["modulation_active_fraction_classifier"] == pytest.approx(1.0)
    assert diagnostics["modulation_active_fraction_early_blocks"] == pytest.approx(0.0)


def test_lr_controller_global_scales_effective_learning_rate_without_changing_base_update():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
    from neuroplastic_optimizer.plasticity import PlasticityConfig
    from neuroplastic_optimizer.stabilization import HomeostaticConfig

    baseline_param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    controlled_param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    grad = torch.tensor([0.5, -0.25], dtype=torch.float32)
    baseline_param.grad = grad.clone()
    controlled_param.grad = grad.clone()

    homeostatic = HomeostaticConfig(max_update_norm=1e9, adaptation_rate=0.0)
    common = dict(
        lr=0.1,
        weight_decay=0.0,
        homeostatic_config=homeostatic,
        decoupled_weight_decay=True,
    )

    baseline = NeuroPlasticOptimizer(
        [baseline_param],
        plasticity_config=PlasticityConfig(
            hybrid_base="adamw",
            modulation_strength=0.0,
            min_alpha=0.5,
            max_alpha=1.5,
        ),
        **common,
    )
    controlled = NeuroPlasticOptimizer(
        [controlled_param],
        plasticity_config=PlasticityConfig(
            hybrid_base="adamw",
            modulation_strength=0.0,
            lr_controller_mode="global",
            controller_alpha=0.5,
            controller_low=0.8,
            controller_high=1.2,
            activity_weight=1.0,
            gradient_weight=0.0,
            memory_weight=0.0,
            min_alpha=0.5,
            max_alpha=1.5,
        ),
        **common,
    )

    controlled_state = controlled.state[controlled_param]
    controlled_state.update(controlled.state_memory.initialize(controlled_param))
    controlled_state["base_momentum"] = torch.zeros_like(controlled_param)
    controlled_state["base_variance"] = torch.zeros_like(controlled_param)
    controlled_state["activity_trace"] = torch.tensor([10.0, 1.0], dtype=torch.float32)

    baseline.step()
    controlled.step()

    baseline_diag = baseline.collect_diagnostics()
    controlled_diag = controlled.collect_diagnostics()
    baseline_delta = torch.linalg.vector_norm(torch.tensor([1.0, -2.0]) - baseline_param.detach())
    controlled_delta = torch.linalg.vector_norm(torch.tensor([1.0, -2.0]) - controlled_param.detach())

    assert controlled_diag["raw_update_norm"] == pytest.approx(
        baseline_diag["raw_update_norm"], rel=1e-6, abs=1e-6
    )
    assert controlled_diag["controller_multiplier_mean"] < 1.0
    assert controlled_delta < baseline_delta


def test_lr_controller_classifier_only_targets_only_classifier_group():
    import torch

    from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer
    from neuroplastic_optimizer.plasticity import PlasticityConfig
    from neuroplastic_optimizer.stabilization import HomeostaticConfig

    params = [
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
    ]
    for param in params:
        param.grad = torch.tensor([0.5], dtype=torch.float32)

    opt = NeuroPlasticOptimizer(
        params,
        lr=0.1,
        weight_decay=0.0,
        plasticity_config=PlasticityConfig(
            hybrid_base="adamw",
            modulation_strength=0.0,
            lr_controller_mode="classifier_only",
            controller_alpha=0.5,
            controller_low=0.8,
            controller_high=1.2,
            activity_weight=1.0,
            gradient_weight=0.0,
            memory_weight=0.0,
            min_alpha=0.5,
            max_alpha=1.5,
        ),
        homeostatic_config=HomeostaticConfig(max_update_norm=1e9, adaptation_rate=0.0),
        decoupled_weight_decay=True,
    )

    classifier_param = params[-1]
    classifier_state = opt.state[classifier_param]
    classifier_state.update(opt.state_memory.initialize(classifier_param))
    classifier_state["base_momentum"] = torch.zeros_like(classifier_param)
    classifier_state["base_variance"] = torch.zeros_like(classifier_param)
    classifier_state["activity_trace"] = torch.tensor([2.0, 0.5], dtype=torch.float32)
    opt.step()

    diagnostics = opt.collect_diagnostics()
    assert diagnostics["controller_multiplier_mean_classifier"] < 1.0
    assert diagnostics["controller_multiplier_mean_early_blocks"] == pytest.approx(1.0)
