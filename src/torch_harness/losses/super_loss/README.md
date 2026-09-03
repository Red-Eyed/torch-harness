# SuperLoss

`SuperLoss` transforms any unreduced task loss into a robust curriculum-learning
loss. It assigns larger gradient weights to easy samples and smaller weights to
hard samples without storing per-sample parameters or depending on the task that
produced the loss.

```python
from math import log

import torch.nn.functional as F

from torch_harness.losses import SuperLoss

criterion = SuperLoss(threshold=log(10), regularization=1.0)
task_loss = F.cross_entropy(logits, targets, reduction="none")
loss = criterion(task_loss)
```

For classification, `log(number_of_classes)` is a natural fixed threshold: it is
the cross-entropy of a uniform prediction. Other tasks should choose the threshold
and regularization in the same units as their unreduced task loss.

## Behavior

For each task loss `l`, threshold `tau`, and positive regularization `lambda`, the
module computes the optimal confidence from the real principal branch of the
Lambert W function and evaluates

```text
confidence * (l - tau) + lambda * log(confidence)^2
```

The confidence is capped at Euler's number for very easy samples. During
backpropagation, both the confidence and threshold are constants, matching the
paper: the incoming gradient for each item is scaled by that item's confidence.
The default reduction is `mean`; `none` and `sum` follow standard PyTorch loss
semantics.

The implementation uses a fixed-count, tensor-only Halley iteration for Lambert W.
It therefore stays on the input device and requires neither SciPy nor a CPU
transfer. Inputs with `float16` or `bfloat16` are evaluated internally in
`float32`; `float64` inputs retain double precision. Public tensor contracts use
`jaxtyping` to state that inputs and outputs are floating-point tensors of
arbitrary shape. PyTorch does not currently expose a native Lambert W operator;
[pytorch/pytorch#108948](https://github.com/pytorch/pytorch/issues/108948) tracks
that feature request.

## Architecture

`SuperLoss` accepts an already-computed loss tensor instead of owning a task-loss
module. This boundary keeps it usable with classification, regression, detection,
metric-learning, or custom objectives while leaving task-specific argument and
reduction conventions with the caller.

The threshold is explicit and stateless. The paper also describes a running loss
average as a dynamic threshold, but that policy is intentionally not hidden inside
this component because its synchronization and update semantics depend on the
training loop.

## Reference

Thibault Castells, Philippe Weinzaepfel, and Jerome Revaud. “SuperLoss: A
Generic Loss for Robust Curriculum Learning.” NeurIPS 2020.
[Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf) ·
[Supplementary material](https://proceedings.neurips.cc/paper_files/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Supplemental.pdf)

```bibtex
@inproceedings{castells2020superloss,
  title = {SuperLoss: A Generic Loss for Robust Curriculum Learning},
  author = {Castells, Thibault and Weinzaepfel, Philippe and Revaud, Jerome},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {33},
  year = {2020},
}
```
