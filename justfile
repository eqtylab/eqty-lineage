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

# Run the before/after lineage demo (writes manifests/before.json and manifests/after.json)
demo:
  #!/usr/bin/env bash
  set -euo pipefail
  # separate processes: eqty_sdk.init() is process-global and raises on a second call
  uv run --no-sync python examples/lineage_diff_demo.py --baseline
  uv run --no-sync python examples/lineage_diff_demo.py
  uv run --no-sync python examples/lineage_diff_demo.py --compare manifests/before.json manifests/after.json

# Format all Python code in the repo
fmt:
  ruff format .

# Lint all Python code in the repo
lint:
  ruff check .
