# Run all checks and build distribution artifacts.
all: check build

# Install the locked development dependencies.
sync:
    uv sync --locked

# Run the CI quality checks.
check: sync
    uv run --locked ruff check .
    uv run --locked ruff format --check .
    uv run --locked pyrefly check
    uv run --locked pytest

# Build the wheel and source distribution.
build:
    uv build
