"""Tests for recursive tensor descriptions and finite-value statistics."""

import pytest
import torch

from torch_harness import describe_tensor


def test_exact_statistics(capsys: pytest.CaptureFixture[str]) -> None:
    """Print metadata, population deviation, and ordered linear quantiles."""
    describe_tensor(torch.tensor([0.0, 2.0, 4.0]), percentiles=(0.75, 0.0, 1.0))
    assert capsys.readouterr().out == (
        "\nroot\n"
        "  shape        (3,)\n"
        "  dtype        torch.float32\n"
        "  device       cpu\n"
        "  count        3\n"
        "  null_count   0\n"
        "  mean         2.0\n"
        "  std          1.632993221282959\n"
        "  min          0.0\n"
        "  p75          3.0\n"
        "  p0           0.0\n"
        "  p100         4.0\n"
        "  max          4.0\n"
    )


def test_nested_paths(capsys: pytest.CaptureFixture[str]) -> None:
    """Traverse mappings and generators while ignoring ordinary scalar leaves."""
    nested = {"weights": [torch.tensor(2), (torch.tensor(3) for _ in range(1))]}
    describe_tensor((nested, "text", b"bytes", None, 42), prefix="batch")
    output = capsys.readouterr().out
    assert "\nbatch[0].weights[0]\n" in output
    assert "\nbatch[0].weights[1][0]\n" in output
    assert output.count("null_count") == 2
    assert "p99" in output
    assert "std          0.0" in output


def test_nonfinite_values(capsys: pytest.CaptureFixture[str]) -> None:
    """Exclude both infinity signs and NaN from finite distribution statistics."""
    tensor = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), 3.0])
    describe_tensor(tensor, percentiles=(0.5,))
    output = capsys.readouterr().out
    assert "null_count   3" in output
    assert "mean         2.0" in output
    assert "std          1.0" in output
    assert "p50          2.0" in output


@pytest.mark.parametrize("values", [[], [float("nan"), float("inf")]])
def test_no_finite_values(
    values: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    """Print metadata without inventing statistics for an empty population."""
    describe_tensor(torch.tensor(values))
    output = capsys.readouterr().out
    assert f"null_count   {len(values)}" in output
    assert "mean" not in output
    assert "std" not in output
    assert "p50" not in output


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64]
)
def test_dtypes_and_input_preservation(
    dtype: torch.dtype, capsys: pytest.CaptureFixture[str]
) -> None:
    """Calculate in supported precision without changing input values or metadata."""
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=dtype).mT
    before = tensor.clone()
    strides = tensor.stride()
    describe_tensor(tensor)
    output = capsys.readouterr().out
    assert f"dtype        {dtype}" in output
    assert "mean         2.5" in output
    assert tensor.stride() == strides
    torch.testing.assert_close(tensor, before)


def test_float64_precision(capsys: pytest.CaptureFixture[str]) -> None:
    """Keep finite float64 values that would overflow or round away in float32."""
    describe_tensor(torch.tensor([1e100], dtype=torch.float64))
    output = capsys.readouterr().out
    assert "null_count   0" in output
    assert "mean         1e+100" in output
    describe_tensor(torch.tensor([1.0, 1.0 + 2e-12], dtype=torch.float64))
    assert "mean         1.000000000001" in capsys.readouterr().out


def test_preserves_gradients(capsys: pytest.CaptureFixture[str]) -> None:
    """Leave existing gradients and the live computation graph intact."""
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    tensor.square().sum().backward()
    gradient = tensor.grad
    assert gradient is not None
    before = gradient.clone()
    describe_tensor(tensor)
    capsys.readouterr()
    assert tensor.grad is gradient
    torch.testing.assert_close(gradient, before)
    tensor.sum().backward()
    torch.testing.assert_close(gradient, before + 1)


@pytest.mark.parametrize("percentile", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_percentiles(
    percentile: float, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validate quantile bounds before writing any output."""
    with pytest.raises(ValueError, match="finite fractions"):
        describe_tensor(torch.ones(2), percentiles=(percentile,))
    assert capsys.readouterr().out == ""


def test_empty_percentiles_and_boolean(capsys: pytest.CaptureFixture[str]) -> None:
    """Allow no quantiles and treat booleans as numeric indicator values."""
    describe_tensor(torch.tensor([False, True]), percentiles=())
    output = capsys.readouterr().out
    assert "mean         0.5" in output
    assert "p50" not in output


def test_rejects_complex(capsys: pytest.CaptureFixture[str]) -> None:
    """Reject complex statistics rather than silently dropping imaginary values."""
    with pytest.raises(TypeError, match="complex"):
        describe_tensor(torch.tensor([1 + 2j]))
    assert capsys.readouterr().out == ""


def test_rejects_meta() -> None:
    """Reject metadata-only tensors before attempting to read their values."""
    with pytest.raises(ValueError, match="materialized"):
        describe_tensor(torch.empty(2, device="meta"))


def test_rejects_sparse() -> None:
    """Reject sparse layouts without implicitly densifying their storage."""
    with pytest.raises(TypeError, match="dense"):
        describe_tensor(torch.eye(2).to_sparse())


def test_rejects_quantized() -> None:
    """Reject quantized storage without guessing a dequantization policy."""
    tensor = torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.qint8)
    with pytest.raises(TypeError, match="non-quantized"):
        describe_tensor(tensor)
