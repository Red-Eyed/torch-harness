"""Smoke tests for the top-level package."""

import subprocess
import sys
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

import pytest

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
            "assert callable(torch_harness.replace_layers); "
            "assert callable(torch_harness.summarize_flops); "
            "assert 'FlopCounter' in dir(torch_harness); "
            "assert 'write_text_graph' in dir(torch_harness); "
            "assert 'torch_harness.onnx' not in sys.modules; "
            "assert 'torch_harness.lightning' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "namespace", ["layers", "losses", "runtime", "flops", "lightning", "onnx"]
)
def test_root_exports_match_namespaces(namespace: str) -> None:
    """Re-export every namespace's public utilities without wrapping objects."""
    module = import_module(f"torch_harness.{namespace}")
    for name in module.__all__:
        assert name in torch_harness.__all__
        assert getattr(torch_harness, name) is getattr(module, name)
        assert vars(torch_harness)[name] is getattr(module, name)


@pytest.mark.parametrize(
    ("dependency", "export"),
    [("lightning", "FlopCounter"), ("onnx", "write_text_graph")],
)
def test_missing_optional_export(dependency: str, export: str) -> None:
    """Explain the missing extra when an optional utility is requested."""
    script = f"""
import sys
sys.modules[{dependency!r}] = None
import torch_harness as th
try:
    getattr(th, {export!r})
except ModuleNotFoundError as error:
    assert error.name == {dependency!r}
    assert 'torch-harness[{dependency}]' in str(error)
else:
    raise AssertionError('optional import unexpectedly succeeded')
assert {export!r} not in vars(th)
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


def test_unknown_export() -> None:
    """Preserve standard attribute lookup behavior for misspelled names."""
    with pytest.raises(AttributeError, match="has no attribute 'missing_utility'"):
        _ = torch_harness.missing_utility


def test_nested_import_failure_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not misreport an integration's broken dependency as a missing extra."""
    error = ModuleNotFoundError("nested dependency failed", name="nested_dependency")

    def fail_import(name: str, package: str | None = None) -> ModuleType:
        """Simulate an installed integration failing to import a dependency."""
        raise error

    monkeypatch.setattr(torch_harness, "import_module", fail_import)
    with pytest.raises(ModuleNotFoundError, match="nested dependency failed") as caught:
        torch_harness.__getattr__("FlopCounter")
    assert caught.value is error
