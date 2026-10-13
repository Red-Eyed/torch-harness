# Adaptive loss balancing

`LossBalancer` maintains a primary loss weight of exactly one and adapts each
auxiliary weight toward a user-selected relative gradient norm. It owns no
trainable parameters and requires no optimizer changes.

```python
from torch_harness.losses import (
    BalancerConfig,
    LossBalancer,
    LossTarget,
    weighted_loss,
)

balancer = LossBalancer(
    reference="decoder",
    adaptive={
        "ctc": LossTarget(ratio=0.3, initial_weight=0.1),
        "script": LossTarget(ratio=0.2, initial_weight=0.1),
    },
    config=BalancerConfig(update_every=10),
)

# Inside training_step, before ordinary backward or GradScaler.scale:
losses = {"decoder": decoder_ce + z_loss, "ctc": ctc_loss, "script": script_loss}
weights = balancer.update(losses, shared=encoder_features, step=global_step)
total = weighted_loss(losses, weights)
# Return total to Lightning automatic optimization, or call total.backward().
```

The OCR names above are caller configuration. Supply any named scalar floating
losses and a shared differentiable activation. `shared` can also be a sequence
of tensors, with the norm taken over their concatenated gradients. Targets must
be unique, require gradients, and share a device with the losses. Outside DDP,
passing `tuple(model.shared_encoder.parameters())` measures parameter norms.
An objective disconnected from every selected target is explicitly skipped;
partial connectivity contributes the connected gradients.

## Algorithm and stability

On scheduled steps, `torch.autograd.grad` measures each loss gradient without
populating parameter `.grad` or creating higher-order graphs. The graph is
retained for the caller's subsequent ordinary backward. Gradient magnitudes,
not scalar loss magnitudes, determine the weights:

```text
g_i = ||grad_shared(L_i)||_2
ema_i = decay * ema_i + (1 - decay) * g_i
desired_i = clamp(ratio_i * ema_reference / ema_i, min_weight_i, max_weight_i)
change_i = clamp(rate * (log(desired_i) - log(weight_i)), -limit, limit)
weight_i = clamp(exp(log(weight_i) + change_i), min_weight_i, max_weight_i)
total = L_reference + sum(weight_i * L_i) + sum(fixed_i * L_fixed_i)
```

The first usable measurement initializes its EMA directly. `ema_decay` applies
per measurement, not per training step. `adaptation_rate` and `max_log_change`
control adaptation independently of cadence. For example, reference norm 10,
auxiliary norm 20, and ratio 0.3 imply a target weight of 0.15. Smoothing means
the current weight approaches that target over multiple measurements.

Nonfinite losses or gradients, disconnected losses, and norms at or below
`gradient_floor` freeze the affected auxiliary weight and EMA. An unusable
reference freezes all adaptive state. These skips count per auxiliary per
scheduled update. The absolute floor, positive bounds, and bounded log motion
prevent unbounded escalation as an objective is solved. Choose the floor for
the scale and reduction of your losses; a floor is not a universal convergence
detector. The balancer does not repair a nonfinite training objective: the
caller's existing fault policy must handle it before backward.

Use `fixed={"regularizer": 0.01}` for a term excluded from measurement and
adaptation. A fixed weight of zero omits that term entirely during aggregation,
including a disabled NaN-valued loss. All supplied names require configuration.
Configured auxiliaries and fixed losses may disappear from the loss dictionary
and later reappear; their weights and EMA survive the absence. New names require
a new controller configuration. The reference is always required.

Repeated calls at the same step do not remeasure. With gradient accumulation,
the first microbatch at a scheduled optimizer step supplies the measurement;
this is not a measurement of the complete accumulated gradient. Steps must be
nonnegative and cannot move backwards.

## DDP and mixed precision

Synchronization is enabled by default when a process group is initialized.
One packed all-reduce on scheduled steps combines mean rank-local norms and
validity flags. No balancing collective runs on other steps. Loss ordering is
canonical, and an absent or invalid loss on any rank skips that loss everywhere.
All ranks must construct the same configuration, start from the same checkpoint,
and call updates at identical steps. Rank-dependent configuration or validation
errors violate this contract and may hang collective operations. Restore the
checkpoint on every rank. `synchronize=False` is for local measurements; it
does not guarantee common weights in distributed training.

The reduced statistic is the mean of rank-local norms, **not** the norm of the
DDP-averaged gradient. Gradient directions can cancel across ranks. Losses and
their reductions must therefore be defined consistently across ranks.

For DDP, use shared activations downstream of the wrapped model's parameters.
Measuring at those activations stops the measurement traversal before parameter
accumulation and leaves ordinary DDP backward intact. Direct `nn.Parameter`
targets are rejected with synchronization enabled. Do not bypass this restriction
using parameter views or disabled synchronization. PyTorch documents that DDP
does not support using `autograd.grad` as its parameter-gradient mechanism.
[DDP reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

Pass unscaled losses before `GradScaler.scale(total)`. FP16/BF16 norm arithmetic
and loss aggregation use at least float32; float64 inputs preserve precision.
Scaling before norm evaluation cannot recover overflow or underflow that already
occurred upstream. CPU Gloo DDP is tested; CUDA/NCCL remains unverified locally.
Use non-reentrant activation checkpointing for graph regions traversed by
`autograd.grad`; reentrant checkpointing is incompatible with this measurement.
[Autograd reference](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)

## Checkpoints and diagnostics

```python
checkpoint["loss_balancer"] = balancer.state_dict()
balancer.load_state_dict(checkpoint["loss_balancer"])

for name, value in balancer.metrics().items():
    writer.add_scalar(name, value, global_step)
```

In Lightning, save and restore this entry from the module's existing
`on_save_checkpoint` and `on_load_checkpoint` hooks, and pass `balancer.metrics()`
to the existing logger. There is no callback, second optimizer, or logger setup.
State uses primitive Python records and includes the configuration fingerprint,
current weights, EMA, last step, and skip count. Loading rejects mismatched
configuration or malformed state before changing live state. Diagnostics are
transient and are recomputed after restoring.

`diagnostics` exposes raw loss values, weights, measured norms, weighted norms,
actual weighted auxiliary/reference ratios, EMA norms, saturation, and typed
update statuses. Missing metrics carry `Unavailable(reason=...)`; `metrics()`
omits them rather than inventing zero. Fixed losses have loss and weight metrics
but no gradient measurements. Off-cadence calls expose current loss/weight/EMA
metrics without presenting old gradient norms as current. Norms in DDP are
global statistics; raw loss values remain rank-local. Scalar diagnostic
extraction synchronizes device execution, so include it in overhead assessment.

Monitor gradient ratios against targets, sustained bound saturation, and skip
counts. Activation norms are proxies for parameter influence, and do not
measure objective quality or resolve conflicting gradient directions. Ratios
may be infeasible because of bounds, vanishing gradients, changing objectives,
or the selected representation. Cosine diagnostics are not implemented.

Run `uv run examples/adaptive_loss_balancing.py` for deterministic optimization
with two independent auxiliary objectives and checkpoint restoration.
