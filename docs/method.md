# Method

NeuroPlastic Optimizer applies updates of the form:

$$
\theta_{t+1} = \theta_t - \eta \, (\alpha_t \odot g_t)
$$

where \(g_t = \nabla_{\theta_t}\mathcal{L}(\theta_t)\) is the gradient and \(\alpha_t\) is a plasticity coefficient computed from:

$$
\alpha_t = \mathrm{clip}_{[\alpha_{\min},\alpha_{\max}]}\!\left(w_a \cdot \hat a_t + w_g \cdot \hat g_t + w_m \cdot \hat m_t\right)
$$

1. local activity traces (EMA of absolute gradients),
2. gradient magnitude signal,
3. parameter memory signal based on momentum-to-variance ratio,
4. bounded homeostatic stabilization.

## Plasticity modes

- **rule_based**: weighted fusion of activity, gradient, and memory signals.
- **ablation_grad_only**: simplified mode using only gradient signal; useful for isolating contributions.

## Stabilization

Homeostatic stabilization uses:

- norm clipping on update vectors,
- adaptive gain scaling toward a target RMS update.

This is biologically inspired by homeostatic regulation but remains an engineering abstraction designed for training stability.

## Scope and non-claims

- Biologically inspired components are **signal analogies** (activity traces and stabilization pressure), not mechanistic neural simulations.
- The framework does **not** claim biological fidelity, brain equivalence, or AGI-like properties.
