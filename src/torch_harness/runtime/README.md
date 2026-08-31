# Runtime model transformations

`replace_layers` recursively replaces every exact match of a layer class,
including registrations inside `Sequential`, `ModuleList`, `ModuleDict`, and
nested custom modules.

```python
from torch import nn

from torch_harness.runtime import replace_layers

model = replace_layers(
    model,
    old_layer_cls=nn.SiLU,
    replacement_factory=lambda silu: nn.ReLU(inplace=silu.inplace),
)
```

The replacement factory receives the old layer, making configuration and state
transfer explicit. For example, BatchNorm-to-GroupNorm conversion must choose a
group count and decide which affine parameters to retain rather than relying on
constructor guessing.

Exact-class matching leaves custom subclasses untouched.

Replacement should run before creating an optimizer, DDP/FSDP wrapper, or
compiled model because those objects may retain references to the original
modules or parameters.
