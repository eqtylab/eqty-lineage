_:
  @just --list

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
  uv run --no-sync pytest tests/test_tool_results.py tests/test_transcript.py \
      tests/test_semiring.py tests/test_engine.py tests/test_codex.py \
      tests/test_redaction.py tests/test_determination.py tests/test_policy.py

# Delete build artifacts
clean:
  rm -rf ./dist

# Format all Python code in the repo
fmt:
  ruff format .
