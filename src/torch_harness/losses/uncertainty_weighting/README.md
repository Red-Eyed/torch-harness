# Uncertainty weighting

`UncertaintyWeighting` learns one positive scale for each task and combines a
one-dimensional tensor of scalar losses into one training objective. All task
positions use the same computation; the component has no primary-task or
auxiliary-task policy.

```python
import torch

from torch_harness.losses import UncertaintyWeighting

weighting = UncertaintyWeighting(num_tasks=3)
optimizer = torch.optim.Adam(
    [*model.parameters(), *weighting.parameters()],
    lr=1e-3,
)

task_losses = torch.stack((classification_loss, depth_loss, normal_loss))
loss = weighting(task_losses)
loss.backward()

weights_for_logging = weighting.effective_weights
```

If `weighting` is registered as a child module of `model`, its parameters are
already included in `model.parameters()` and must not be added twice.

## Behavior

For task losses `L_i`, unconstrained learned parameters `r_i`, and the configured
positive floor `epsilon`, the module computes

```text
v_i = softplus(r_i) + epsilon
total = sum(L_i / v_i + log(v_i))
```

The effective task weights are `1 / v_i`. Every variance starts at exactly one
up to floating-point rounding, so the initial effective weights are approximately
one. `effective_weights` returns a detached tensor intended for metrics and logs;
the training result retains gradients to both the input losses and
`raw_uncertainties`.

The input must be a floating-point rank-one tensor whose length matches
`num_tasks`. The output is a scalar. Under autocast, the division, softplus, and
logarithm run outside autocast in float32. Explicit float64 inputs or module
parameters retain double-precision calculation.

## Scope and interpretation

This is homoscedastic uncertainty weighting: each task receives one learned scale
that is shared by every sample and batch. The formulation follows the uncertainty
weighting family introduced by Kendall, Gal, and Cipolla. With `v` interpreted as
variance, the regression likelihood objective matches the expression above up to
an overall constant factor, which does not alter its optimum.

The learned values balance numerical loss scale and estimated task uncertainty.
They do not express semantic importance, guarantee equal task quality, or account
for conflicting gradient directions. Gradient-aware methods require per-task
gradients or access to shared model parameters and therefore belong outside this
scalar-loss component.

Recent benchmarking treats uncertainty weighting as an established baseline and
also documents limitations such as slow parameter adaptation and overfitting in
some settings. It should not be interpreted as universally state of the art.

## References

- Alex Kendall, Yarin Gal, and Roberto Cipolla. “Multi-Task Learning Using
  Uncertainty to Weigh Losses for Scene Geometry and Semantics.” CVPR 2018.
  [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)
- Lukas Kirchdorfer et al. “Analytical Uncertainty-Based Loss Weighting in
  Multi-Task Learning.” 2024. [Paper](https://arxiv.org/abs/2408.07985)
