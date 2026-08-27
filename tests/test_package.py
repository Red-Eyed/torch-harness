"""Smoke tests for the top-level package."""

import torch_utils


def test_package_imports() -> None:
    """Verify that the installed top-level package can be imported."""
    assert torch_utils.__name__ == "torch_utils"
