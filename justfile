set positional-arguments

_:
  @just --list

# UV sync to install dependencies
sync:
  uv sync --group dev --python "$(command -v python)" --no-managed-python --no-python-downloads

# Build the whl and sdist for every workspace package (outputs to ./dist), and the Relay cdylib
build: build-python build-relay

# Build the whl and sdist for every workspace package (outputs to ./dist)
build-python:
  uv build --all-packages --python "$(command -v python)" --no-managed-python --no-python-downloads

# Not a wheel: `kind = "rust_dynamic"` loads this through a C ABI, so the artifact is a per-platform
# shared library that ships with its own `relay-plugin.toml` and sha256 digest.
#
# Kept out of PR CI: it recompiles the whole dependency tree at a profile nothing else on the PR path
# uses. `release-nemo-relay-plugin.yml` builds it for every shipped target on a release instead, so a
# release is the first time a release build runs for a given commit. `release.yml` cannot do this: it
# resolves a `PACKAGE@X.Y.Z` tag to a `pyproject.toml` and this package has none, which is why the
# plugin has a workflow of its own rather than a job in that one.
#
# Build the Relay cdylib in release
build-relay:
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

# Run every test suite
test *ARGS: (test-python ARGS) test-rust

# Run the Python test suite
test-python *ARGS:
  uv run --no-sync pytest "$@"

# Run the Rust test suites
test-rust: test-rust-plugin test-rust-plugin-abi

# Run the plugin's own test suite
test-rust-plugin:
  cd packages/eqty-lineage-nemo-relay && cargo test

# Its own crate with its own dependency graph: it needs Relay's core, which cannot coexist with
# `integrity`. See packages/eqty-lineage-nemo-relay/abi-test/Cargo.toml.
#
# `build_cdylib` inside this suite builds the plugin into the plugin's *own* target directory, so
# running `test-rust-plugin` first leaves the dependencies it needs already compiled. Running this
# one on its own compiles that tree from nothing.
#
# Run the ABI load test
test-rust-plugin-abi:
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

# The plugin has no Python dependency and must not grow one; this writes a fixture from eqty_sdk so
# the Rust tests can assert cross-language agreement on asset identity without importing anything.
#
# Regenerate the golden content-CID vector the Relay plugin checks itself against
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
fmt: fmt-python fmt-rust

fmt-python:
  ruff format .

fmt-rust:
  cd packages/eqty-lineage-nemo-relay && cargo fmt
  cd packages/eqty-lineage-nemo-relay/abi-test && cargo fmt

# Verify formatting without rewriting anything
fmt-check: fmt-check-python fmt-check-rust

fmt-check-python:
  ruff format --check .

fmt-check-rust:
  cd packages/eqty-lineage-nemo-relay && cargo fmt --check
  cd packages/eqty-lineage-nemo-relay/abi-test && cargo fmt --check

# Lint every package in the repo
lint: lint-python lint-rust

lint-python:
  ruff check .

# Dependencies are compiled, never linted -- `Checking nemo-relay-plugin v0.8.3` in a CI log is cargo
# making a dependency's types available, not checking its code. So there is nothing here to narrow to
# "only our code"; the cost is compiling two deliberately incompatible dependency graphs from scratch.
#
# Clippy over both Rust crates
lint-rust: lint-rust-plugin lint-rust-plugin-abi

# Split per workspace for the same reason as the test recipes: CI runs the two in separate jobs, and
# a job that lints both needs both dependency trees, which is the duplication the split removes.
#
# Lint the plugin
lint-rust-plugin:
  cd packages/eqty-lineage-nemo-relay && cargo clippy --all-targets -- -D warnings

# Lint the ABI load test
lint-rust-plugin-abi:
  cd packages/eqty-lineage-nemo-relay/abi-test && cargo clippy --all-targets -- -D warnings

# Stage a signed, installable Relay plugin into dist/relay-plugin.
#
# The staging directory is the point, not a convenience. `nemo-relay` copies the WHOLE directory
# containing relay-plugin.toml into an activation snapshot, recursively, against a 512 MiB budget.
# Installed from the package root that closure is src/ + tests/ + target/ + abi-test/target/ -- 11 GB
# -- and the gateway refuses to start, naming some unrelated rlib. So the installed manifest lives
# beside just the dylib, its signature, and the config schema.
#
# Signing is not optional at runtime. `plugins add` evaluates trust with attestation defaulting to
# `integrity_only`, but ACTIVATION calls apply_secure_runtime_defaults(), which forces
# `signature_required` unless plugins.toml says otherwise. An unsigned plugin therefore installs
# cleanly and then refuses to start.
#
# Relay verifies a raw Ed25519 signature over the ARTIFACT BYTES (not over the digest), read from
# the file named by `integrity.signature`, base64 with an optional `ed25519:` prefix.
#
# Override the key with RELAY_SIGNING_KEY=/path/to/ed25519.pem. A dev key is generated on first use;
# release CI should pass EQTY's real key instead.
nemo-relay-package target="" out="dist/relay-plugin":
  #!/usr/bin/env bash
  set -euo pipefail
  key="${RELAY_SIGNING_KEY:-$HOME/.config/eqty-lineage/relay-dev-signing-key.pem}"
  if [ ! -f "$key" ]; then
    mkdir -p "$(dirname "$key")"
    (umask 077 && openssl genpkey -algorithm ed25519 -out "$key")
    echo "generated a NEW dev signing key at $key (private -- do not commit or share)"
  fi
  # Ask cargo where it put the cdylib rather than assuming: the name is per-platform and the
  # directory moves with CARGO_TARGET_DIR, which `just linux-check` sets. A hardcoded
  # `target/release` can find a stale artifact and package it with a digest over the same stale
  # bytes, so the check passes while verifying nothing.
  root="$PWD"
  out="{{out}}"
  export TARGET_TRIPLE="{{target}}"
  cd packages/eqty-lineage-nemo-relay
  if [ -z "$TARGET_TRIPLE" ]; then
    TARGET_TRIPLE="${CARGO_BUILD_TARGET:-$(rustc -vV | sed -n 's/^host: //p')}"
  fi
  test -n "$TARGET_TRIPLE"
  # `--target` moves the artifact under `target/<triple>/release`, which is exactly why the path is
  # read from cargo's own output rather than assembled here.
  artifact=$(cargo build --locked --release --lib --message-format=json-render-diagnostics \
    --target "$TARGET_TRIPLE" \
    | python3 -c '
  import json, sys
  for line in sys.stdin:
      try:
          message = json.loads(line)
      except ValueError:
          continue
      if message.get("reason") != "compiler-artifact":
          continue
      if message.get("target", {}).get("name") != "eqty_lineage_nemo_relay":
          continue
      for name in message.get("filenames", []):
          if name.endswith((".dylib", ".so", ".dll")):
              print(name)
  ' | tail -1)
  cd "$root"
  if [ -z "$artifact" ] || [ ! -f "$artifact" ]; then
    echo "cargo reported no cdylib for eqty_lineage_nemo_relay" >&2
    exit 1
  fi
  lib=$(basename "$artifact")
  rm -rf "$out" && mkdir -p "$out"
  cp "$artifact" "$out/"
  cp packages/eqty-lineage-nemo-relay/config.schema.json "$out/"
  cp packages/eqty-lineage-nemo-relay/relay-plugin.toml "$out/"
  cp LICENSE "$out/"
  # The cdylib links its whole tree in, so the bundle carries those crates and owes their notices.
  # Generated per bundle rather than committed: it has to describe the tree this artifact was built
  # from, and a stale copy would attribute code the binary does not contain.
  python3 packages/eqty-lineage-nemo-relay/tools/third_party_licenses.py \
    packages/eqty-lineage-nemo-relay/Cargo.toml --target "$TARGET_TRIPLE" > "$out/THIRD-PARTY-LICENSES.md"
  cd "$out"
  # `shasum` is not everywhere; `sha256sum` is the GNU coreutils spelling.
  if command -v shasum >/dev/null 2>&1; then
    digest=$(shasum -a 256 "$lib" | awk '{print $1}')
  else
    digest=$(sha256sum "$lib" | awk '{print $1}')
  fi
  openssl pkeyutl -sign -rawin -inkey "$key" -in "$lib" -out "$lib.sig.raw"
  base64 < "$lib.sig.raw" | tr -d '\n' > "$lib.sig"
  rm -f "$lib.sig.raw"
  pub="ed25519:$(openssl pkey -in "$key" -pubout -outform DER | tail -c 32 | base64 | tr -d '\n')"
  python3 - "$digest" "$lib.sig" "$lib" <<'EOF'
  import re, sys
  digest, signature, lib = sys.argv[1], sys.argv[2], sys.argv[3]
  p = "relay-plugin.toml"
  s = open(p).read()
  s = re.sub(r'sha256 = "sha256:[^"]*"', f'sha256 = "sha256:{digest}"', s)
  # The committed manifest is a template naming the macOS artifact. Both references have to follow
  # the platform, or activation looks for a library that was never built.
  s = re.sub(r'artifact = "[^"]*"', f'artifact = "{lib}"', s)
  s = re.sub(r'library = "[^"]*"', f'library = "{lib}"', s)
  if "signature =" not in s:
      s = s.replace(f'sha256 = "sha256:{digest}"',
                    f'sha256 = "sha256:{digest}"\nsignature = "{signature}"')
  open(p, "w").write(s)
  EOF
  echo "staged $out  sha256:$digest"
  echo
  echo "Add this to the [plugins.policy] block of your plugins.toml, or activation will refuse:"
  echo
  echo "  [plugins.policy.overrides.\"eqty.lineage\"]"
  echo "  attestation = \"signature_required\""
  echo "  trusted_public_keys = [\"$pub\"]"
  echo
  echo "then: nemo-relay plugins add --user ./$out/relay-plugin.toml"

# Development here is on macOS, and three things are only knowable on Linux -- the cdylib's name,
# the digest over it, and whether the staged `.so` actually loads. `just nemo-relay-package` was
# macOS-only for a while and failed *after* a successful release build; no test covers a recipe, so
# this is the check that would have caught it.
#
# Runs against `git archive HEAD`, not the working tree: this is what CI would build, and it keeps
# the container from writing into `target/` or `dist/`. Commit before running, or run
# `nemo-relay-package` directly to test uncommitted work on this platform.
#
# The registry and target caches are named volumes, so a re-run is minutes rather than a cold build.
#
# **One target volume per workspace, and the plugin's is mounted where the nested build looks for
# it.** `abi-test` is a second workspace with its own lockfile, and cargo does not isolate workspaces
# inside one target directory: sharing one made both resolve `darling` -- to different versions --
# into the same `.fingerprint`, and whichever built second failed with `E0460` naming a crate neither
# workspace mentions. Separate volumes make that unrepresentable rather than merely unlikely.
#
# The plugin's volume lands on `packages/eqty-lineage-nemo-relay/target` because `build_cdylib` in
# `abi-test/tests/lifecycle.rs` passes exactly that path to `--target-dir`, deliberately: the flag
# beats `CARGO_TARGET_DIR` so the nested build never waits on the lock its own parent holds. Mounting
# the cache anywhere else left that path empty inside the container, so the nested build compiled
# `integrity`, iroh and ssi from nothing on every run -- the cold rebuild `build_cdylib` documents
# itself as avoiding.
#
# Run the Linux checks in Docker: test suite, packaging recipe, and ABI load test
linux-check:
  #!/usr/bin/env bash
  set -euo pipefail
  work=$(mktemp -d)
  trap 'rm -rf "$work"' EXIT
  git archive HEAD | tar x -C "$work"
  docker volume create eqty-cargo-registry >/dev/null
  docker volume create eqty-plugin-target >/dev/null
  docker volume create eqty-abi-target >/dev/null
  docker run --rm \
    -v "$work:/work" \
    -v "$PWD/packages/eqty-lineage-nemo-relay/docs/linux-check.sh:/linux-check.sh:ro" \
    -v eqty-cargo-registry:/usr/local/cargo/registry \
    -v eqty-plugin-target:/work/packages/eqty-lineage-nemo-relay/target \
    -v eqty-abi-target:/target-abi \
    -e CARGO_TERM_COLOR=never \
    rust:1-bookworm bash /linux-check.sh
