"""Recursively replace registered layers in an assembled PyTorch model."""

from collections.abc import Callable
from typing import TypeVar

from torch import nn

_OldLayerT = TypeVar("_OldLayerT", bound=nn.Module)
ReplacementFactory = Callable[[_OldLayerT], nn.Module]


def replace_layers(
    model: nn.Module,
    old_layer_cls: type[_OldLayerT],
    replacement_factory: ReplacementFactory[_OldLayerT],
) -> nn.Module:
    """Replace every exact instance of one layer class recursively.

    The factory receives each matched layer and owns all construction and state
    transfer decisions. The root itself is replaced when it exactly matches
    ``old_layer_cls``.

    Replacement creates new module objects in the general case. Run this before
    constructing optimizers, distributed wrappers, or compiled models whose
    state refers to the original modules or parameters.

    Args:
        model: Root module to transform.
        old_layer_cls: Exact class to match; subclasses are left unchanged.
        replacement_factory: Callable that builds one replacement from each
            matched layer.

    Returns:
        The original root or its replacement when the root matched.

    Raises:
        TypeError: If the replacement factory returns a non-module value.

    Example:
        Replace SiLU activations throughout a nested model::

            >>> from torch import nn
            >>> from torch_harness.runtime import replace_layers
            >>> model = nn.Sequential(nn.SiLU(), nn.Sequential(nn.SiLU()))
            >>> model = replace_layers(
            ...     model,
            ...     old_layer_cls=nn.SiLU,
            ...     replacement_factory=lambda _silu: nn.ReLU(),
            ... )
            >>> isinstance(model[0], nn.ReLU)
            True
            >>> isinstance(model[1][0], nn.ReLU)
            True
    """
    if type(model) is old_layer_cls:
        replacement = replacement_factory(model)
        if not isinstance(replacement, nn.Module):
            raise TypeError("replacement_factory must return a torch.nn.Module")
        return replacement

    for name, child in model.named_children():
        replacement = replace_layers(
            child,
            old_layer_cls,
            replacement_factory,
        )
        if replacement is not child:
            setattr(model, name, replacement)

    return model
