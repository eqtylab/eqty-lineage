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

# Run the test suite
test *args:
  uv run pytest tests {{args}}

# Drive a real Codex session and assert the lineage it produces (needs codex on PATH)
validate-codex *args:
  ./scripts/validate-codex.sh {{args}}

# Publish the whl and sdist to pypi.eqtylab.io
publish:
  uv publish --index eqty

# Publish the whl and sdist for one workspace package
publish-package package:
  uv publish --index eqty "dist/{{package}}/*"

# Delete build artifacts
clean:
  rm -rf ./dist

# Format all Python code in the repo
fmt:
  ruff format .

# Lint all Python code in the repo
lint:
  ruff check .
