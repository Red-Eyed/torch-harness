# Stable cross-entropy development log

## 2026-10-30 — numerical reference implementation

### Motivation

[Barron et al.](https://arxiv.org/abs/2610.31642) identify cancellation in
high-confidence normalization and near-equal student/teacher cross-entropy.
Add explicit opt-in utilities to reproduce the numerical fixes independently
of any trainer or model. Preserve forward values and backward gradients.

### Design

Use excluded-maximum normalization with `log1p`, and gather hard-target losses
from those log probabilities. This avoids a separate target-relative hard-loss
formula and uses ordinary autograd, including higher derivatives. FP16/BF16
arithmetic promotes to FP32; FP64 remains FP64. Defaults use the last class axis.

Distillation accepts detached teacher logits and substitutes the paper's
first-order student gradient. A bounded row-wise near-equality branch prevents
overflow even in unused `where` operands; other rows use corrected probability
differences. The component README records this difference from the paper's
per-entry fallback. No higher-order distillation support is claimed.

Use plain PyTorch as the portable numerical reference. Do not add a Triton
dependency without GPU measurements and forward/backward numerical validation.
Provide a reproducible benchmark that checks compiled small-gradient behavior
before accepting a compiled variant.

### Verified results

- `uv sync --locked` succeeds.
- `uv run ruff check .` and `uv run ruff format --check .` pass.
- `uv run pyrefly check` and an explicit check of the benchmark pass.
- `uv run pytest`: 265 passed, 3 skipped (CUDA unavailable). The new component
  contributes 42 passing tests and the 3 CUDA skips. Coverage includes analytical
  high-margin losses/gradients, the paper's near-equal distillation example,
  exactly zero equal-logit gradients, axes, noncontiguous inputs, tied maxima,
  nonuniform upstream gradients, reductions, accumulation dtypes, detached
  teachers, singleton classes, empty batches, invalid inputs, gradcheck, and
  hard-target gradgradcheck.
- `uv build --wheel` creates `torch_harness-0.17.0-py3-none-any.whl`.
- `uv run benchmarks/stable_cross_entropy.py` completes on arm64 macOS 27.2,
  PyTorch 2.13.0, CPU, FP32, one timer thread. Compilation/warmup are excluded.
  The table contains median milliseconds from the final run; these are
  synthetic CPU measurements, not training throughput or GPU predictions.

| Objective, shape | Native forward | Stable forward | Native forward + backward | Stable forward + backward |
| --- | ---: | ---: | ---: | ---: |
| Hard, `(128, 10)` | 0.003 | 0.016 | 0.013 | 0.043 |
| Hard, `(256, 1000)` | 0.187 | 0.378 | 0.382 | 0.565 |
| Hard, `(64, 32768)` | 1.536 | 2.843 | 3.120 | 4.508 |
| Distillation, `(256, 1000)` | 0.448 | 2.289 | 0.652 | 2.326 |
| Distillation, `(64, 32768)` | 3.588 | 18.874 | 5.279 | 19.158 |

Compiled hard-target CE passes the margin 18/40/80 checks but is slower here:
6.054 ms forward + backward at `(64, 32768)` versus 4.508 ms eager stable.
Compiled distillation fails the near-equality backward check (maximum absolute
gradient discrepancy about `1.33e-8`, relative discrepancy about 79%). The
benchmark logs the failure with its traceback and omits that compiled variant.
The compiler failure's root cause has not been established.

### Unverified and follow-up work

- Run the conditional CUDA tests and benchmark on the target GPU, including
  memory measurements and compiler numerical checks.
- Consider a separate fused Triton backend only after measuring the GPU gap.
  Validate its forward and backward against the reference, including low
  precision, high confidence, near agreement, and large class counts.
- The paper's grokking, slingshot, and generalization experiments have not been
  reproduced. No corresponding training-result claims are made.

## 2026-10-30 — PyTorch-compatible replacements

### Motivation and design

The original functions require callers to change class axes, supported options,
and low-precision output assumptions. Add a compatible functional interface
without changing those existing APIs. Public functions live in
`torch_harness.functional` and are also exported at the package root.

Separate compatibility normalization from CE target/reduction handling. Reuse
the stable normalization core, casting output to the native requested dtype.
Use native `nll_loss` for index targets so ignored labels, target validation,
class weighting, and weighted means retain native behavior. Add weighted uniform
loss only when smoothing is enabled; probability targets use their own native
per-item normalization and remain differentiable. Legacy flags and tensor
override dispatch are handled at the functional boundary. CPU/CUDA autocast
promotes low-precision CE inputs before normalization; the query retains a
fallback for older supported PyTorch versions.

`StableCrossEntropyLoss` directly inherits `nn.CrossEntropyLoss` and overrides
only `forward`, retaining constructor, buffer, state-dict, and device/dtype
lifecycle behavior. Logits-based distillation remains separate because a
probability-target API cannot recover teacher-logit information.

### Verified results

- `uv sync --locked`, Ruff lint/format checks, Pyrefly (including the benchmark),
  and `uv build --wheel` pass for version `0.18.0`.
- Full suite: 517 passed, 9 CUDA skips. The new compatibility file contributes
  251 passing cases and 6 CUDA skips. Tests compare forward values and backward
  gradients with native PyTorch for index/probability targets, class weights,
  smoothing, ignored indices, reductions/legacy flags, unbatched/spatial/strided
  layouts, dtype casts, CPU autocast, mixed-dtype promotion, scalar/empty inputs,
  all-ignored and zero-weight means, gradient checks including higher derivatives,
  and error paths. Tests preserve the high-confidence numerical fix, native
  parameter signatures, override dispatch, state serialization and dtype moves,
  and a real optimizer step.
- An explicit CPU Inductor check of compatible CE passes both forward and
  backward comparisons to eager at margins 18, 40, and 80, with zero absolute
  tolerance and `2e-5` relative tolerance. This covers the high-margin regression,
  not every compiled combination of CE options.
- `uv run benchmarks/stable_cross_entropy.py` completes on the same CPU setup
  as above, including the compatible eager hard-target loss. At `(64, 32768)`,
  median forward / forward-plus-backward times are 1.503 / 3.062 ms native,
  2.776 / 4.241 ms explicit stable, and 2.781 / 4.230 ms compatible stable.
  The compatible API adds no meaningful overhead in this particular workload;
  this is not a general performance guarantee. Compiled distillation is still
  rejected by the benchmark's existing numerical check.

### Remaining limits

CUDA behavior, all supported PyTorch versions, TorchScript/export, and arbitrary
compiler option combinations have not been verified locally. Conditional CUDA
parity tests are included. Dense finite inputs remain the supported numerical
contract. No GPU speedup or new training-result claim is made.
