set positional-arguments

_:
  @just --list

# UV sync to install dependencies
sync:
  uv sync --group dev --python "$(command -v python)" --no-managed-python --no-python-downloads

# Build the whl and sdist for every workspace package (outputs to ./dist), and the Relay cdylib
build:
  uv build --all-packages --python "$(command -v python)" --no-managed-python --no-python-downloads
  # Not a wheel: `kind = "rust_dynamic"` loads this through a C ABI, so the artifact is a per-platform
  # shared library that ships with its own `relay-plugin.toml` and sha256 digest.
  cd packages/eqty-lineage-nemo-relay && cargo build --release

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
  uv run --no-sync pytest "$@"
  cd packages/eqty-lineage-nemo-relay && cargo test
  # Its own crate, with its own dependency graph: it needs Relay's core, which cannot coexist with
  # `integrity`. See packages/eqty-lineage-nemo-relay/abi-test/Cargo.toml.
  cd packages/eqty-lineage-nemo-relay/abi-test && cargo test

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

# Run the DeepAgents research agent and export its lineage to manifests/deep-agent.json
deepagents-demo *ARGS:
  uv run --no-sync python examples/deepagents/research_agent.py "$@"

# Regenerate the golden content-CID vector the Relay plugin checks itself against
#
# The plugin has no Python dependency and must not grow one; this writes a fixture from eqty_sdk so
# the Rust tests can assert cross-language agreement on asset identity without importing anything.
relay-cid-vector:
  #!/usr/bin/env bash
  set -euo pipefail
  out="$PWD/packages/eqty-lineage-nemo-relay/tests/fixtures/content-cids.json"
  # eqty_sdk writes its store relative to the working directory, so run somewhere disposable.
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  cd "$work"
  "$OLDPWD/.venv/bin/python" "$OLDPWD/packages/eqty-lineage-nemo-relay/tests/fixtures/content_cids.py" > "$out"
  echo "wrote $out"

# Format every package in the repo
fmt:
  ruff format .
  cd packages/eqty-lineage-nemo-relay && cargo fmt
  cd packages/eqty-lineage-nemo-relay/abi-test && cargo fmt

# Verify formatting without rewriting anything
fmt-check:
  ruff format --check .
  cd packages/eqty-lineage-nemo-relay && cargo fmt --check
  cd packages/eqty-lineage-nemo-relay/abi-test && cargo fmt --check

# Lint every package in the repo
lint:
  ruff check .
  cd packages/eqty-lineage-nemo-relay && cargo clippy --all-targets -- -D warnings
  cd packages/eqty-lineage-nemo-relay/abi-test && cargo clippy --all-targets -- -D warnings
