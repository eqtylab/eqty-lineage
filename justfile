_:
  @just --list

# UV sync to install dependencies
sync:
  uv sync --group dev --python "$(command -v python)" --no-managed-python --no-python-downloads

# Build the whl and sdist for every workspace package (outputs to ./dist)
build:
  uv build --all-packages --python "$(command -v python)" --no-managed-python --no-python-downloads

# Build the whl and sdist for one workspace package (outputs to ./dist/<package>)
build-package package:
  uv build --package "{{package}}" --out-dir "dist/{{package}}" --clear --python "$(command -v python)" --no-managed-python --no-python-downloads

# Publish the whl and sdist to pypi.eqtylab.io
publish:
  uv publish --index eqty

# Publish the whl and sdist for one workspace package
publish-package package:
  uv publish --index eqty "dist/{{package}}/*"

# Delete build artifacts
clean:
  rm -rf ./dist

# Run the test suite
test *ARGS:
  uv run --no-sync pytest {{ARGS}}

# Run only the tests that need no eqty-sdk, in an interpreter that does not have it. The recorder
# reaches the SDK lazily, and this is what proves that claim rather than restating it.
test-nosdk:
  #!/usr/bin/env bash
  set -euo pipefail
  venv="$(mktemp -d)/venv"
  python3 -m venv "$venv"
  "$venv/bin/pip" -q install pytest
  "$venv/bin/pip" -q install --no-deps ./packages/eqty-lineage-core
  ! "$venv/bin/python" -c 'import eqty_sdk' 2>/dev/null || { echo "eqty-sdk leaked in"; exit 1; }
  "$venv/bin/python" -m pytest tests/test_redaction.py tests/test_serialize.py tests/test_tool_results.py

# Format all Python code in the repo
fmt:
  ruff format .

# Lint all Python code in the repo
lint:
  ruff check .
