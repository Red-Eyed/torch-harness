"""Tests for recursive runtime layer replacement."""

from typing import cast

import pytest
from torch import nn

from torch_utils.runtime import replace_layers


def _relu_from_silu(silu: nn.SiLU) -> nn.Module:
    """Build ReLU with the source activation's in-place policy."""
    return nn.ReLU(inplace=silu.inplace)


def _group_norm_from_batch_norm(batch_norm: nn.BatchNorm2d) -> nn.Module:
    """Build four-group normalization for a BatchNorm channel count."""
    return nn.GroupNorm(
        num_groups=4,
        num_channels=batch_norm.num_features,
        eps=batch_norm.eps,
        affine=batch_norm.affine,
    )


def test_replaces_layers_inside_nested_sequential() -> None:
    """Verify recursive traversal reaches nested registered containers."""
    model = nn.Sequential(
        nn.SiLU(inplace=True),
        nn.Sequential(nn.Linear(3, 3), nn.SiLU()),
    )

    converted = replace_layers(model, nn.SiLU, _relu_from_silu)

    assert converted is model
    assert isinstance(model[0], nn.ReLU)
    assert model[0].inplace is True
    assert isinstance(model[1], nn.Sequential)
    assert isinstance(model[1][1], nn.ReLU)
    assert model[1][1].inplace is False


def test_replaces_root_layer() -> None:
    """Verify a matching root is returned as its replacement."""
    model = nn.SiLU(inplace=True)

    converted = replace_layers(model, nn.SiLU, _relu_from_silu)

    assert isinstance(converted, nn.ReLU)
    assert converted.inplace is True


def test_replaces_heterogeneous_batch_norm_layers() -> None:
    """Verify each factory call receives its matched layer configuration."""
    model = nn.Sequential(
        nn.BatchNorm2d(8),
        nn.Sequential(nn.BatchNorm2d(16)),
    )

    converted = replace_layers(
        model,
        nn.BatchNorm2d,
        _group_norm_from_batch_norm,
    )

    assert converted is model
    assert isinstance(model[0], nn.GroupNorm)
    assert model[0].num_channels == 8
    assert isinstance(model[1], nn.Sequential)
    assert isinstance(model[1][0], nn.GroupNorm)
    assert model[1][0].num_channels == 16


def test_leaves_subclasses_unchanged() -> None:
    """Verify exact matching does not replace custom subclasses."""

    class CustomSiLU(nn.SiLU):
        """Distinct SiLU policy used to verify exact-class matching."""

    custom = CustomSiLU()
    model = nn.Sequential(custom)

    replace_layers(model, nn.SiLU, _relu_from_silu)

    assert model[0] is custom


def test_rejects_non_module_replacement() -> None:
    """Verify an invalid factory result fails at the transformation boundary."""

    def invalid_factory(_silu: nn.SiLU) -> nn.Module:
        """Return an invalid value while satisfying the static test boundary."""
        return cast(nn.Module, "not a module")

    with pytest.raises(TypeError, match=r"must return a torch\.nn\.Module"):
        replace_layers(nn.Sequential(nn.SiLU()), nn.SiLU, invalid_factory)
