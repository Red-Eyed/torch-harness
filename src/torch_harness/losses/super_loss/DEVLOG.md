# SuperLoss development log

## 0.7.0

### Motivating observation

Ordinary task losses give difficult or mislabeled samples their full gradient
weight. Robust curriculum learning should downweight those samples without
requiring task-specific code, a second optimizer, or persistent per-sample state.

### Decision

Implement the closed-form SuperLoss from Castells, Weinzaepfel, and Revaud as a
standalone transformation over an unreduced floating-point loss tensor. Keep the
threshold explicit rather than embedding a running-average policy whose update and
distributed synchronization semantics belong to the training loop.

Compute the real principal Lambert W branch with a fixed-count Halley iteration in
PyTorch. This keeps execution on the loss device and avoids adding SciPy or moving
training data through the CPU. Compute half-precision inputs in float32, preserve
float64 evaluation, cap confidence at the paper's branch point, and detach the
optimal confidence before applying it to the task loss. Express floating tensor
contracts with `jaxtyping` while retaining explicit runtime rejection of integer
tensors.

### Verified result

Analytical tests verify three closed-form points and their expected confidence
gradients, including the capped branch point. Further tests verify the optimality
equation across the real principal branch, reductions, all four supported floating
dtypes, large finite float64 input, configuration failures, and public exports.

An automated 600-point comparison against `scipy.special.lambertw`, spanning
values immediately above `-1/e` through `1e300`, passes with relative and absolute
tolerances of `1e-10`. An independent high-precision `mpmath` comparison over the
same range found a maximum scaled error of `2.77e-11` and a maximum scaled
defining-identity residual of `5.45e-14`.

The full test suite, Ruff linting and formatting, Pyrefly, and the portable wheel
build pass.

### Unproven claims

The tensor implementation has not been benchmarked against a lookup-table Lambert
W approximation on accelerators. Convergence has not been characterized for every
finite value of every PyTorch floating-point dtype.
