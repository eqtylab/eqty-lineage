#!/usr/bin/env bash
# Everything a Linux CI would check, run inside the container by `just linux-check`.
#
# Three things are only knowable on Linux: the test suite catches platform assumptions in the port;
# `just nemo-relay-package` is a recipe, so no test covers its platform-specific paths; and
# `abi-test` dlopens the cdylib, because a `.so` that stages correctly and cannot be loaded is still
# a broken package.
#
# The digest and signature are re-derived rather than trusted: Relay verifies a raw Ed25519 signature
# over the ARTIFACT BYTES, and a manifest whose digest names a different build activates and then
# refuses to start.
set -uo pipefail

banner() { echo; echo "######## $* ########"; echo; }

banner "environment"
uname -srm
rustc --version
cargo --version

banner "toolchain deps"
apt-get update -qq >/dev/null 2>&1
# openssl signs, python3 rewrites the manifest -- both used by the packaging recipe.
apt-get install -y -qq openssl python3 curl file >/dev/null 2>&1
openssl version
python3 --version
# `just` is not in bookworm; the official installer picks the right target.
curl -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin >/dev/null 2>&1
just --version

cd /work

# One target directory per workspace -- sharing one is not merely slow but wrong; see the recipe.
# This path is also what `build_cdylib` hands to `--target-dir`, so the nested build inside
# `abi-test` reuses what this suite just compiled.
export CARGO_TARGET_DIR=/work/packages/eqty-lineage-nemo-relay/target

banner "plugin test suite"
# Never pipe this through `tail`: it swallows the per-suite counts and leaves the total looking like
# a fraction of itself.
cargo test --manifest-path packages/eqty-lineage-nemo-relay/Cargo.toml 2>&1 \
  | grep -E "^(test |test result|error|running)"
plugin_status=${PIPESTATUS[0]}

banner "packaging recipe"
just nemo-relay-package
package_status=$?
echo
echo "--- staged artifacts ---"
ls -l dist/relay-plugin/
echo
echo "--- manifest references ---"
grep -E '^(artifact|library|sha256|signature) = ' dist/relay-plugin/relay-plugin.toml

echo
echo "--- the artifact is an ELF shared object, and the manifest describes that build ---"
lib=$(ls dist/relay-plugin/*.so 2>/dev/null | head -1)
if [ -z "$lib" ]; then
  echo "FAIL: no .so was staged"
  package_status=1
else
  file "$lib"
  recomputed=$(sha256sum "$lib" | awk '{print $1}')
  declared=$(grep -oE 'sha256:[0-9a-f]{64}' dist/relay-plugin/relay-plugin.toml | cut -d: -f2)
  echo "recomputed: $recomputed"
  echo "declared:   $declared"
  if [ "$recomputed" = "$declared" ]; then
    echo "DIGEST OK"
  else
    echo "DIGEST MISMATCH"
    package_status=1
  fi

  key="${RELAY_SIGNING_KEY:-$HOME/.config/eqty-lineage/relay-dev-signing-key.pem}"
  base64 -d < "$lib.sig" > /tmp/sig.raw
  if openssl pkeyutl -verify -rawin -pubin \
      -inkey <(openssl pkey -in "$key" -pubout) \
      -in "$lib" -sigfile /tmp/sig.raw >/dev/null 2>&1; then
    echo "SIGNATURE OK (raw Ed25519 over artifact bytes)"
  else
    echo "SIGNATURE FAILED TO VERIFY"
    package_status=1
  fi
fi

banner "abi load test"
export CARGO_TARGET_DIR=/target-abi
cargo test --manifest-path packages/eqty-lineage-nemo-relay/abi-test/Cargo.toml 2>&1 \
  | grep -E "^(test |test result|error)"
abi_status=${PIPESTATUS[0]}

banner "summary"
echo "plugin suite : exit $plugin_status"
echo "packaging    : exit $package_status"
echo "abi test     : exit $abi_status"
if [ "$plugin_status" = 0 ] && [ "$package_status" = 0 ] && [ "$abi_status" = 0 ]; then
  echo "ALL LINUX CHECKS PASSED"
else
  echo "SOME LINUX CHECKS FAILED"
  exit 1
fi
