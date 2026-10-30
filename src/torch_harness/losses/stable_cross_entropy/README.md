# Stable softmax and cross-entropy

Opt-in PyTorch implementations of the numerical fixes in
[A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642)
(Barron, Gnanapragasam, and Tschannen, 2026).

```python
import torch
import torch_harness as th

logits = torch.tensor([[9.0, -9.0, -9.0]], requires_grad=True)
loss = th.stable_cross_entropy(logits, torch.tensor([0]))
loss.backward()
# loss ≈ 3.046e-8; logits.grad ≈ [[-3.046e-8, 1.523e-8, 1.523e-8]]

student = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
teacher = student.detach().clone()
distillation = th.stable_distillation_cross_entropy(student, teacher)
distillation.backward()
# The student gradient is exactly zero. The loss is the teacher's entropy.
```

Functions are exported from `torch_harness`, `torch_harness.losses`, and
`torch_harness.losses.stable_cross_entropy`.

## Contracts

- `stable_log_softmax(logits, dim=-1)` excludes one selected maximum from the
  exponential sum and uses `log1p`. Replacing that maximum's shifted logit with
  a constant routes its gradient through the remaining classes, avoiding the
  high-confidence cancellation. Tied maxima are supported.
- `stable_softmax(logits, dim=-1)` exponentiates the stable log probabilities.
  Its output can still round to one; backward retains the small derivative.
- `stable_cross_entropy(logits, targets, dim=-1, reduction="mean")` gathers the
  negative stable log probability of each int64 target. The target shape must
  equal the logits shape with the class axis removed, on the same device.
  Out-of-range indices raise a PyTorch runtime error. Unlike native PyTorch CE,
  the default class axis is **last**, and class weights, ignored labels, label
  smoothing, and probability targets are not supported.
- `stable_distillation_cross_entropy(student_logits, teacher_logits, dim=-1,
  reduction="mean")` accepts matching shapes, devices, and dtypes. Teacher
  logits are detached. Its forward is teacher cross-entropy, not KL divergence;
  identical logits produce teacher entropy and exactly zero student gradient.

Loss reductions are `none`, `mean`, and `sum`. `none` removes only the class
axis; `mean` averages over all remaining items. Empty means are NaN; empty sums
are zero. Class axes must exist and have at least one class. Invalid dtype,
shape, axis, device, or reduction raises `TypeError` or `ValueError` as described
in the function docstrings.

Inputs must be dense, finite tensors of dtype FP16, BF16, FP32, or FP64.
FP16/BF16 inputs compute and return FP32; FP32 and FP64 keep their dtype.
Inputs are never mutated. Gradients returning to low-precision inputs still use
the input dtype, so sufficiently small gradients can underflow during that cast.
This fixes cancellation, not exponent underflow or overflow of differences
between logits at opposite extremes of the dtype range. Non-finite values and
fully masked rows are outside the contract; there is no synchronizing finite
value scan in the hot path.

## Distillation gradient

Near equality, the implementation centers the student-minus-teacher logits
under the teacher distribution and uses `expm1`/`log1p` to compute probability
differences. It applies the paper's correction `g = u - q * sum(u)` to remove
normalization residuals, up to floating-point reduction error.

To avoid overflow in eagerly evaluated branches, the log-space expression is
used only for rows whose centered differences all have magnitude below `0.5`.
Other rows use probability differences and the same residual correction. This
is a bounded row-wise alternative to the paper's per-entry fallback. The
substituted student gradient is detached: **higher-order distillation
derivatives are unsupported**. Hard-target CE and normalization use ordinary
autograd and support higher derivatives.

Temperature scaling and its loss multiplier belong to the caller:

```python
temperature = 2.0
loss = (
    th.stable_distillation_cross_entropy(
        student_logits / temperature,
        teacher_logits / temperature,
    )
    * temperature**2
)
```

## Performance and compilation

These are portable reference implementations using ordinary PyTorch operations,
with no Triton dependency or automatic compilation. They allocate intermediate
tensors proportional to logits size. They do not fuse the output projection
with cross-entropy or avoid materializing logits.

Run `uv run benchmarks/stable_cross_entropy.py` to compare native, stable eager,
and compiled stable forward and forward-plus-backward timings at explicit
small, medium, and vocabulary-sized FP32 shapes. CUDA is selected when available;
otherwise CPU is used. The script excludes compilation/warmup, uses one CPU
timer thread, reports timing dispersion, and reports CUDA peak temporary memory.
It checks small-loss and gradient regressions before accepting compiled variants;
numerically failing variants are logged with a traceback and omitted.

**Do not assume `torch.compile` preserves this fix.** On the development CPU with
PyTorch 2.13.0, Inductor passed the hard-target checks but failed near-equality
distillation gradients. Use eager distillation unless the compiled path passes
the numerical checks on your actual backend. GPU throughput, GPU compilation
accuracy, and custom Triton speedups remain unverified. A fused Triton backend
should be justified by GPU measurements and checked against these forward and
backward reference tests.

No claim is made that this component reproduces the paper's grokking or
generalization results.
