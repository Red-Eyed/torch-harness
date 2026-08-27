# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Purpose and Sources of Truth

This repository is a typed Python library of reusable utilities for PyTorch and
related training frameworks.

Use the root `README.md` and the relevant colocated documentation as the source of
truth for available features, public behavior, limitations, and usage. Read those
documents and the corresponding tests before changing an implementation. Update them
when behavior changes.

Do not add component inventories, class descriptions, or API contracts to this file.
Those details change frequently and belong in user-facing or component documentation.
`AGENTS.md` should contain durable guidance about how to work on the project.

## Development Workflow

Use `uv` for every Python and package-management command. Never use bare `python`,
`pip`, or an unmanaged environment.

```bash
uv sync --locked                 # Install locked project and development dependencies.
uv run pytest                    # Run the complete test suite.
uv run pytest path/to/test.py::test_name  # Run one test by pytest node ID.
uv run ruff check .              # Lint the repository.
uv run ruff format --check .     # Verify formatting without modifying files.
uv run ruff format .             # Format Python files after editing them.
uv run pyrefly check             # Run strict static type checking.
uv build --wheel                 # Build the distributable wheel.
```

Python 3.11 is the development baseline. Treat all Python versions declared in
`pyproject.toml` as supported. Do not use a newer-language feature without checking
that it is valid on the minimum supported version.

Before making a non-trivial change:

1. Inspect `git status`, relevant docs, implementation, tests, and configuration.
2. State the files to change, the intended behavior, and concrete success criteria.
3. Run the SOLID checklist below. Redesign before proceeding if an item fails.
4. Wait for user approval when the work touches more than two files or changes
   architecture.

After implementation, run checks in proportion to the change. Before a commit, run
Ruff linting, Ruff formatting, Pyrefly, and the relevant tests. CI additionally builds
the wheel; reproduce the complete CI sequence for package or release changes.

Preserve unrelated working-tree changes. Existing modifications belong to the user
unless explicitly identified otherwise. Make surgical edits and never discard or
rewrite unrelated work to obtain a clean tree.

## Design Principles

### SOLID checklist

- **Single Responsibility:** each module, type, and function has one reason to change.
  Do not mix business logic, framework integration, I/O, and output concerns.
- **Open/Closed:** add a new implementation or wrapper for a new behavior instead of
  threading observer-specific flags, writers, or callbacks through stable logic.
- **Liskov Substitution:** implementations must honor the complete inherited or
  declared contract, including signatures, state behavior, errors, and teardown.
- **Interface Segregation:** depend on the smallest interface needed by the caller.
- **Dependency Inversion:** high-level logic depends on abstractions; inject
  side-effecting dependencies and choose concrete implementations at the boundary.

Apply SOLID proportionally. Do not create an interface, factory, or layer of
indirection without a real axis of change or second implementation. Readability is
more important than architectural ceremony.

### Modularity and dependency boundaries

Keep reusable utilities standalone and free of host-application knowledge. Do not
hardcode project-specific paths, data shapes, trainer policies, or downstream
workflows. Put application-specific wiring in the outermost caller.

Keep optional framework integrations isolated in their own namespaces and package
extras. The base package must remain importable without optional dependencies. A new
integration must not make an optional framework an unconditional runtime dependency.

Expose public APIs deliberately through the appropriate package `__init__.py` and an
explicit `__all__`. Keep implementation modules private unless users are expected to
import them directly. Treat exported names, annotations, errors, state layout, and
documented behavior as compatibility-sensitive.

Prefer a functional core and imperative shell:

- Put deterministic transformations in small free functions.
- Confine logging, framework hooks, environment access, disk I/O, and mutable
  lifecycle state to thin boundary code.
- Inject side effects so core behavior can be tested without frameworks, files, or
  external services.
- Never mutate a caller-owned input unless mutation is the documented purpose of the
  API.

When a transform will later be undone, return the transformed value together with an
inverse closure bound to the exact parameters used. Do not require callers to
reconstruct and pass matching shape or configuration values later.

### Clarity and decomposition

Prefer explicit, sparse code over compressed expressions. Use guard clauses and early
returns to keep control flow flat. Extract distinct steps into named helpers when a
function accumulates nesting, multiple branches, or unrelated operations.

If code needs a comment to explain what it does, simplify the code. Comments are for
why a design exists, which invariant it preserves, or which external constraint makes
an otherwise surprising choice necessary.

Automate repeatable setup, validation, build, and migration work. Do not add a manual
instruction when the project can establish the condition safely in code or tooling.

## Python Conventions

### Types

- Fully annotate function and method arguments and return values, including private
  helpers and tests.
- Use `Path` for filesystem paths, `date` for dates, and `datetime` for timestamps.
  Parse external strings once at the boundary.
- Use `Literal` or an enum for a fixed vocabulary instead of a free-form `str`.
- Use a dataclass, `NamedTuple`, `TypedDict`, or another precise typed structure for a
  known record shape. Do not pass structured data between helpers as a bare `dict`.
- Use descriptive `TypeVar` names. A short `T` is acceptable only when the type is
  genuinely generic and obvious in a small scope.
- Use `typing_extensions.override` for overridden methods so signature drift is caught
  statically.
- Avoid `Any`. Use it only at a genuinely untyped third-party boundary, keep it local,
  and convert to a precise internal type immediately.
- Do not broaden a type merely to silence Pyrefly. Fix the model or use the narrowest
  targeted suppression on the affected line when an external typing defect requires
  it.

Required identity fields must be required. For legitimate domain-level absence, use a
typed value that carries the reason rather than a bare `None`; test presence by type,
not truthiness. A local implementation sentinel may use `None` when it does not escape
its immediate scope.

### Documentation and logging

Every Python module, class, method, and function, including private and test helpers,
must have a concise Google-style docstring. Public docstrings explain observable
behavior, side effects, failure semantics, and non-obvious parameter meaning. Private
docstrings state the helper's responsibility or invariant.

Review related docstrings and comments whenever behavior changes. Do not copy tunable
defaults into prose where they will become stale; refer to the configuration or
signature instead.

Use Loguru for runtime diagnostics. Do not use `print` for diagnostics or configure a
second logging framework inside the library. Include tracebacks for tolerated
exceptions when the failure remains relevant to the caller or operator.

Do not add copied corporate copyright blocks or authorship headers. Package and release
authorship comes from Git configuration and project metadata.

## Testing Guidelines

Test observable behavior and invariants rather than mirroring implementation details.
Every bug fix needs a regression test that fails for the original defect. Every new
public behavior needs happy-path, boundary, and failure coverage appropriate to its
risk.

Use pytest fixtures for reusable setup and inject them through test arguments. Let
pytest build fixture dependency trees; do not call fixtures like ordinary functions.
Use plain helpers only for pure transformations or reusable assertions.

Use `pytest.mark.parametrize` for repeated inputs, modes, or error cases. Prefer small,
deterministic tensors and explicit assertions. Avoid randomness when a fixed example
expresses the invariant; otherwise seed it locally.

For framework integrations, combine focused unit tests with at least one minimal
end-to-end test through the real lifecycle. Check preservation properties as well as
new behavior, including state, gradients, exception identity, cleanup, aliasing, and
serialization when relevant.

Keep tests alongside the package's namespace structure. Do not require network access,
external datasets, accelerators, or user-specific paths for the default test suite.

## Documentation and Change Records

User-facing features and examples belong in the root README. Detailed behavior,
limitations, architecture decisions, and usage belong in colocated documentation for
the relevant area. Keep docs and tests synchronized with code in the same change.

For experimental or model-related work, maintain a development log that records:

- the motivating observation or hypothesis;
- the design decision and alternatives considered;
- the verified result and the exact validation performed;
- remaining unproven claims or follow-up experiments.

Do not claim a result is verified when tests or experiments are pending. Once
verification is complete, update any stale pending status in the same change.

## Packaging and Releases

This is a typed library and ships `py.typed`; annotations are part of the public
package. Prefer backward-compatible additions. If a breaking change is necessary,
document the compatibility impact and migration path explicitly.

Every package or user-visible project change must update the version before committing
on `main`. Version bumps are main-branch-only unless the user explicitly says
otherwise. Keep all version declarations and lockfiles synchronized.

Build portable wheels for supported platforms. Normal installation must not depend on
building from source on the user's machine. Do not introduce native build assumptions
tied to the builder's operating system, libc, or incidental shared libraries without
first discussing the portability tradeoff.

Only the user publishes package artifacts. Agents may update versions, verify, build,
commit, tag, and push when requested, but must never run an upload or publish command.
Never force-push `main`; force-pushing another branch requires explicit approval
immediately before the push.

Use the author identity from Git configuration for package metadata, commits, and
release configuration. Do not guess identity from the repository path or OS username.

Every commit message must have three sections:

```text
<short imperative title, at most 72 characters>

- <concrete change>
- <concrete change>

<motivation, problem, and relevant design decision>
```
