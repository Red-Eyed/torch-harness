# Uncertainty weighting development log

## 0.8.0

### Motivating observation

A fixed weighted sum makes multi-task training sensitive to arbitrary loss scales
and requires manual weight searches whose result can become stale as training
dynamics change.

### Decision

Add a standalone `UncertaintyWeighting` module implementing learned homoscedastic
task uncertainty. Accept only an already-computed rank-one tensor of scalar task
losses so the component remains independent of models, task criteria, and training
frameworks. Treat all task positions symmetrically and expose detached inverse
variances for monitoring.

Parameterize each variance through softplus plus a positive floor rather than
optimizing a positive value directly. Initialize the raw values through inverse
softplus so every effective weight starts at one. Disable autocast around the
numerically sensitive calculation and use at least float32 while preserving
explicit float64 operation.

### Verified result

Deterministic focused tests verify known-value aggregation, task-permutation
symmetry, unit initialization, detached monitoring weights, one-step weight
adaptation, gradients to both loss inputs and uncertainty parameters, shape and
dtype failures, configuration validation, public exports, and CPU bfloat16
autocast behavior.

The complete repository suite passes with 78 tests. Ruff linting and formatting,
Pyrefly type checking, and the `0.8.0` wheel build also pass.

### Unproven claims

The utility has not been benchmarked on a downstream multi-task dataset. No claim
is made that uncertainty weighting provides semantic task prioritization, resolves
gradient conflicts, or outperforms gradient-aware multi-objective methods.
