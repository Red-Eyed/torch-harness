"""Smoke tests for the top-level package."""

from importlib.util import find_spec

import torch_harness


def test_package_imports() -> None:
    """Verify that the installed top-level package can be imported."""
    assert torch_harness.__name__ == "torch_harness"


def test_old_import_namespace_is_absent() -> None:
    """Do not retain the superseded pre-release import namespace."""
    assert find_spec("torch_utils") is None
