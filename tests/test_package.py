"""Smoke tests for the top-level package."""

import subprocess
import sys
from importlib.util import find_spec

import torch_harness


def test_package_imports() -> None:
    """Verify that the installed top-level package can be imported."""
    assert torch_harness.__name__ == "torch_harness"


def test_old_import_namespace_is_absent() -> None:
    """Do not retain the superseded pre-release import namespace."""
    assert find_spec("torch_utils") is None


def test_base_package_without_optional_integrations() -> None:
    """Core imports must work when ONNX and Lightning are unavailable."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['onnx'] = None; "
            "sys.modules['lightning'] = None; import torch_harness; "
            "assert 'torch_harness.onnx' not in sys.modules; "
            "assert 'torch_harness.lightning' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
