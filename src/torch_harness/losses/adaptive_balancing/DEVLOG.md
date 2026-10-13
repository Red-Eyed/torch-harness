# Adaptive loss balancing development log

## 2026-10-13 — Gradient-ratio controller

Motivation: auxiliary objectives can dominate or disappear at a shared training
representation even when their scalar loss values look comparable. Preserve the
primary weight and let callers specify each auxiliary's relative influence.

Decision: measure with `autograd.grad`, smooth norms using EMA, and move bounded
weights in log space. Keep gradient measurement, pure arithmetic, controller
state, and aggregation separate. Use one packed reduction of mean local norms
and validity flags in DDP. Choose shared activations for DDP instead of probing
wrapped parameters. Retain optional-loss state rather than modifying configuration
during a distributed step. Configuration fingerprints prevent incompatible resume.

Alternatives: uncertainty weighting is already available but learns peer loss
scales rather than explicit primary-relative gradient ratios. Equal gradient
targets would discard caller priorities. Reducing full gradients would add
communication and memory cost. An extensible strategy hierarchy has no current
need; focused functions cover activation and local parameter measurements.

Validation: `uv sync --locked`, Ruff lint and formatting, and strict Pyrefly
checking pass. `uv run pytest` passes all 127 tests, including 49 new balancing
tests covering independent ratios, EMA and bounded updates, extreme finite
norms, invalid or absent objectives, local parameter groups, mixed precision,
ordinary backward preservation, checkpoint continuation and atomic rejection,
and real two-rank CPU Gloo DDP consistency. `uv build --wheel` produces the
portable `torch_harness-0.9.0-py3-none-any.whl`. The runnable synthetic example
finishes with auxiliary/reference ratios approximately 0.3 and 0.2 after 100
optimization steps; the primary weight stays exactly one.

Unproven: benefits for multilingual OCR accuracy, CUDA/NCCL lifecycle behavior,
long-running production stability, and acceptable overhead on large models.
Activation norms do not establish parameter-gradient ratios or task quality.
