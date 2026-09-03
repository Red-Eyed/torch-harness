"""Reference tests for the tensor-only principal Lambert W implementation."""

from math import e

import torch
from scipy.special import lambertw

from torch_harness.losses.super_loss.super_loss import _principal_lambert_w


def test_principal_lambert_w_matches_scipy() -> None:
    """Compare the complete SuperLoss domain with SciPy's principal branch."""
    negative_inputs = torch.linspace(
        -1.0 / e + 1e-12,
        0.0,
        steps=256,
        dtype=torch.float64,
    )
    positive_inputs = torch.logspace(
        -12,
        300,
        steps=344,
        dtype=torch.float64,
    )
    inputs = torch.cat((negative_inputs, positive_inputs))
    expected = torch.from_numpy(lambertw(inputs.numpy(), k=0).real)

    actual = _principal_lambert_w(inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
