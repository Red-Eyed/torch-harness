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
