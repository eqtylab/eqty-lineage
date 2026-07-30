_:
  @just --list

# Tests needing neither eqty-sdk nor a Rust toolchain. Named once so `test-pure`, `test-nosdk` and CI
# cannot drift -- a file quietly dropped from one of them is a suite that stops being run.
PURE_TESTS := "tests/test_tool_results.py tests/test_transcript.py tests/test_semiring.py tests/test_semiring_laws.py tests/test_engine.py tests/test_codex.py tests/test_dialects.py tests/test_cli.py tests/test_redaction.py tests/test_determination.py tests/test_policy.py tests/test_invariants.py tests/test_serialize.py"

# UV sync to install dependencies
sync:
  uv sync --group dev --python "$(command -v python)" --no-managed-python --no-python-downloads

# Build the whl and sdist for every workspace package (outputs to ./dist)
build:
  uv build --all-packages --python "$(command -v python)" --no-managed-python --no-python-downloads

# Publish the whl and sdist to pypi.eqtylab.io
publish:
  uv publish --index eqty

# Build the optional Rust accelerator (needs a Rust toolchain + maturin)
build-accel:
  cd packages/eqty-lineage-query-rs && maturin build --release

# Run the test suite. Tests needing eqty-sdk skip themselves when it is absent.
test *ARGS:
  uv run --no-sync pytest {{ARGS}}

# Run only the tests that need no SDK and no Rust toolchain
test-pure:
  uv run --no-sync pytest {{PURE_TESTS}}

# The same tests in an interpreter with no eqty-sdk at all, which is what CI runs. `just test-pure`
# uses the project venv, where the SDK is present -- so it cannot prove the suite runs without it.
# This installs the packages with --no-deps into a throwaway venv and proves it.
test-nosdk:
  #!/usr/bin/env bash
  set -euo pipefail
  venv=$(mktemp -d)/venv
  python3 -m venv "$venv"
  "$venv/bin/pip" -q install pytest
  "$venv/bin/pip" -q install --no-deps ./packages/eqty-lineage-core ./packages/eqty-lineage-transcript \
      ./packages/eqty-lineage-agent-hooks ./packages/eqty-lineage-query
  ! "$venv/bin/python" -c 'import eqty_sdk' 2>/dev/null || { echo "eqty-sdk leaked in"; exit 1; }
  "$venv/bin/python" -m pytest {{PURE_TESTS}}

# Delete build artifacts
clean:
  rm -rf ./dist

# Format all Python code in the repo
fmt:
  ruff format .
