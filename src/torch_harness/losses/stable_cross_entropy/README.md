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

## PyTorch-compatible API

Use the compatible functional imports when replacing existing PyTorch calls:

```python
from torch_harness.functional import cross_entropy, log_softmax, softmax
from torch_harness import StableCrossEntropyLoss

loss = cross_entropy(
    logits, targets, weight=class_weights, ignore_index=-100, label_smoothing=0.1
)
criterion = StableCrossEntropyLoss(
    weight=class_weights, ignore_index=-100, reduction="mean", label_smoothing=0.1
)
loss = criterion(logits, targets)
```

These functions are also exported from the package root as `th.cross_entropy`,
`th.log_softmax`, and `th.softmax`. This namespace contains only the three
normalization/loss replacements; import other functional operations from PyTorch.

`softmax(input, dim=None, _stacklevel=3, dtype=None)` and `log_softmax` preserve
PyTorch's argument order, explicit input cast through `dtype`, output dtype,
scalar/empty behavior, and deprecated implicit-dimension selection/warning.
FP16/BF16 normalization computes internally in FP32 and casts back to the input
or requested dtype. Softmax output dtype is unchanged by autocast.

`cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
reduce=None, reduction="mean", label_smoothing=0.0)` accepts native shapes:
unbatched `(C)`, batched `(N, C)`, and spatial `(N, C, ...)`. Targets can be
indices with the class axis removed or floating probabilities of the input shape.
It preserves class weighting, ignored labels, smoothing, reductions, and the
deprecated flags' warning/precedence. Weighted index means divide by the sum of
non-ignored target weights; probability-target means divide by item count.
Probability targets are not silently normalized or range-validated, matching
PyTorch. Gradients to probability targets and their class weights are preserved.

Outside autocast, index losses keep the logits dtype. Under CPU/CUDA autocast,
low-precision CE computation and output use FP32. Probability targets and weights
follow native promotion. Returning gradients to FP16/BF16 input tensors still
permits underflow at the input dtype's limit. This compatible dtype policy
differs from the existing `stable_*` functions, which return FP32 for low-precision
inputs regardless of autocast.

`StableCrossEntropyLoss` directly subclasses `torch.nn.CrossEntropyLoss`. It
inherits the native constructor, registered `weight` buffer, state-dict keys,
device/dtype moves, and reduction settings. Native state dictionaries can be
loaded directly. Existing `CrossEntropyLoss` type checks continue to work.
Only `forward` is overridden. It supports ordinary higher derivatives.

These APIs support dense finite tensors and preserve `__torch_function__`
dispatch. Invalid targets and weights are rejected; exact error wording and
bitwise equivalence with native arithmetic are not promised. Sparse/non-finite
inputs, TorchScript/export, and numerical behavior under every compiler backend
are outside the verified compatibility contract. CUDA tests are conditional;
the development machine can verify only CPU behavior.

The numerical improvement intentionally changes high-confidence values and
gradients. Probability targets support the same call interface as PyTorch but
cannot recover teacher logits already discarded by a prior softmax. Continue
using `stable_distillation_cross_entropy(student_logits, teacher_logits)` for
the paper's near-agreement cancellation fix.

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
compatible stable eager, and compiled stable forward and forward-plus-backward timings at explicit
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
