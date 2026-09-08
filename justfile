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
nemo-relay-package:
  #!/usr/bin/env bash
  set -euo pipefail
  key="${RELAY_SIGNING_KEY:-$HOME/.config/eqty-lineage/relay-dev-signing-key.pem}"
  if [ ! -f "$key" ]; then
    mkdir -p "$(dirname "$key")"
    (umask 077 && openssl genpkey -algorithm ed25519 -out "$key")
    echo "generated a NEW dev signing key at $key (private -- do not commit or share)"
  fi
  # Cargo names the cdylib per platform, and the digest is over one artifact -- so the name has to be
  # selected here and written into the manifest, not hardcoded. `abi-test` has resolved it this way
  # from the start; this recipe hardcoded `.dylib` and so failed on Linux *after* a successful build.
  case "$(uname -s)" in
    Darwin) lib=libeqty_lineage_nemo_relay.dylib ;;
    Linux)  lib=libeqty_lineage_nemo_relay.so ;;
    *) echo "unsupported platform for packaging: $(uname -s)" >&2; exit 1 ;;
  esac
  cd packages/eqty-lineage-nemo-relay && cargo build --release --lib && cd ../..
  rm -rf dist/relay-plugin && mkdir -p dist/relay-plugin
  cp "packages/eqty-lineage-nemo-relay/target/release/$lib" dist/relay-plugin/
  cp packages/eqty-lineage-nemo-relay/config.schema.json dist/relay-plugin/
  cp packages/eqty-lineage-nemo-relay/relay-plugin.toml dist/relay-plugin/
  cd dist/relay-plugin
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
  echo "staged dist/relay-plugin  sha256:$digest"
  echo
  echo "Add this to the [plugins.policy] block of your plugins.toml, or activation will refuse:"
  echo
  echo "  [plugins.policy.overrides.\"eqty.lineage\"]"
  echo "  attestation = \"signature_required\""
  echo "  trusted_public_keys = [\"$pub\"]"
  echo
  echo "then: nemo-relay plugins add --user ./dist/relay-plugin/relay-plugin.toml"

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
# Run the Linux checks in Docker: test suite, packaging recipe, and ABI load test
linux-check:
  #!/usr/bin/env bash
  set -euo pipefail
  work=$(mktemp -d)
  trap 'rm -rf "$work"' EXIT
  git archive HEAD | tar x -C "$work"
  docker volume create eqty-cargo-registry >/dev/null
  docker volume create eqty-cargo-target >/dev/null
  docker run --rm \
    -v "$work:/work" \
    -v "$PWD/packages/eqty-lineage-nemo-relay/docs/linux-check.sh:/linux-check.sh:ro" \
    -v eqty-cargo-registry:/usr/local/cargo/registry \
    -v eqty-cargo-target:/target \
    -e CARGO_TARGET_DIR=/target \
    -e CARGO_TERM_COLOR=never \
    rust:1-bookworm bash /linux-check.sh
