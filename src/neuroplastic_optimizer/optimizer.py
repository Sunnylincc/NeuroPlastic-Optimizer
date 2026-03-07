from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from neuroplastic_optimizer.plasticity import PlasticityConfig, compute_plasticity
from neuroplastic_optimizer.stabilization import HomeostaticConfig, HomeostaticStabilizer
from neuroplastic_optimizer.state import ParameterStateMemory
from neuroplastic_optimizer.traces import ActivityTraceExtractor


class NeuroPlasticOptimizer(torch.optim.Optimizer):
    """PyTorch optimizer with synaptic-plasticity-inspired adaptive modulation."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        plasticity_config: PlasticityConfig | None = None,
        homeostatic_config: HomeostaticConfig | None = None,
    ) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.plasticity_config = plasticity_config or PlasticityConfig()
        self.state_memory = ParameterStateMemory()
        self.trace_extractor = ActivityTraceExtractor()
        self.stabilizer = HomeostaticStabilizer(homeostatic_config)

    @torch.no_grad()
    def step(self, closure: Any = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state.update(self.state_memory.initialize(p))

                state["activity_trace"] = self.trace_extractor.update(state["activity_trace"], grad)
                self.state_memory.update_stats(state, grad)

                alpha = compute_plasticity(
                    grad=grad,
                    activity_trace=state["activity_trace"],
                    momentum=state["momentum"],
                    variance=state["variance"],
                    config=self.plasticity_config,
                )

                update = alpha * grad
                if wd > 0:
                    update = update + wd * p

                stabilized = self.stabilizer.stabilize(update)
                p.add_(stabilized, alpha=-lr)

        return loss
