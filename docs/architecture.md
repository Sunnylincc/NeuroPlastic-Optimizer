# Architecture

![NeuroPlastic Optimizer architecture](../assets/neuroplastic_optimizer_architecture.svg)

**Figure caption.** NeuroPlastic Optimizer augments gradient-based learning with synaptic-plasticity-inspired adaptive modulation. Local neural activity traces, gradient signals, and parameter-state memory are integrated by a plasticity encoder and controller to compute adaptive plasticity coefficients, while a homeostatic stabilization module constrains update magnitude for stable training.

## Component boundaries

- **Task execution pathway**: input batch → backbone network → predictions → task loss.
- **Biological signal abstraction**:
  - activity trace extraction,
  - gradient/error signal collection,
  - parameter-state memory.
- **Optimization engine**:
  - plasticity encoder,
  - plasticity controller,
  - homeostatic stabilizer,
  - update application back to model parameters.

These boundaries are designed so future work can replace each module independently (e.g., adding learned controller heads or distributed-memory implementations).
