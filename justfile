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

# Compare the LangChain lineage the handler produces now against a released one (default: newest tag)
langchain-diff-demo REF="":
  #!/usr/bin/env bash
  set -euo pipefail
  # three processes: eqty_sdk.init() is process-global and a second call is ignored rather than
  # refused, so two runs in one process would silently share the first one's store; the comparison
  # reads both summaries from stdin rather than from disk
  before=$(uv run --no-sync python examples/langchain/diff_demo.py --baseline {{REF}})
  after=$(uv run --no-sync python examples/langchain/diff_demo.py)
  printf '%s\n%s\n' "$before" "$after" \
    | uv run --no-sync python examples/langchain/diff_demo.py --compare

# Run only the tests that need no eqty-sdk, in an interpreter that does not have it. The recorder
# reaches the SDK lazily, and this is what proves that claim rather than restating it.
test-nosdk:
  #!/usr/bin/env bash
  set -euo pipefail
  venv="$(mktemp -d)/venv"
  # The interpreter the workspace itself runs on, not whatever `python3` resolves to -- on macOS that
  # is the system 3.9, below this package's >=3.11 floor, so the recipe failed on the install step
  # rather than proving anything about the SDK.
  "$(command -v python)" -m venv "$venv"
  "$venv/bin/pip" -q install pytest
  "$venv/bin/pip" -q install --no-deps ./packages/eqty-lineage-recorder
  ! "$venv/bin/python" -c 'import eqty_sdk' 2>/dev/null || { echo "eqty-sdk leaked in"; exit 1; }
  "$venv/bin/python" -m pytest -p no:cacheprovider packages/eqty-lineage-recorder/tests/test_redaction.py packages/eqty-lineage-recorder/tests/test_serialize.py packages/eqty-lineage-recorder/tests/test_tool_results.py

# Format all Python code in the repo
fmt:
  ruff format .

# Lint all Python code in the repo
lint:
  ruff check .
