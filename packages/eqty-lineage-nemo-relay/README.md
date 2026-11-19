# eqty-lineage-nemo-relay

A [NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) plugin that records each Claude Code or
Codex session as a signed EQTY lineage manifest: the prompts, model calls, tool calls and files the
agent read and wrote.

Why it is built the way it is: [docs/design.md](docs/design.md).

## Install

### From a release

Download the bundle for your platform from the repository's releases, then:

```bash
tar xzf eqty-lineage-nemo-relay-<version>-<target>.tar.gz
nemo-relay plugins validate ./eqty-lineage-nemo-relay-<version>-<target>/relay-plugin.toml
nemo-relay plugins add --user ./eqty-lineage-nemo-relay-<version>-<target>/relay-plugin.toml
nemo-relay plugins enable eqty.lineage
```

Then pin EQTY's release key in `~/.config/nemo-relay/plugins.toml`, so only a bundle EQTY signed
will load:

```toml
[plugins.policy.overrides."eqty.lineage"]
attestation = "signature_required"
trusted_public_keys = ["ed25519:kHMtfiQL4mOOBP+apTlOGRFkSR/B/Flth5EOVsgCd9g="]
```

Bundles are shipped for `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu` and
`aarch64-unknown-linux-gnu`. Other platforms build from source.

### From source

```bash
just nemo-relay-package                                   # stages and signs dist/relay-plugin
nemo-relay plugins validate ./dist/relay-plugin/relay-plugin.toml
nemo-relay plugins add --user ./dist/relay-plugin/relay-plugin.toml
nemo-relay plugins enable eqty.lineage
```

Install from `dist/relay-plugin`, never from this package directory: Relay copies the whole
directory and refuses one this large. The recipe prints the `[plugins.policy.overrides]` block for
your local signing key; paste it into `plugins.toml`.

Changes take effect the next time Relay starts. `nemo-relay plugins remove` deletes the plugin's
configuration too, so add it again after re-adding the plugin.

## Configure

Settings go in `~/.config/nemo-relay/plugins.toml`, under the plugin's `[[plugins.dynamic]]` entry.
All are optional.

```toml
[[plugins.dynamic]]
manifest = "/path/to/relay-plugin.toml"

[plugins.dynamic.config]
manifest_dir      = ".eqty/manifests"
deny_globs        = [".env*", "*.pem", "id_rsa*", "*/.ssh/*", "*credentials*"]
max_content_bytes = 104857600
```

| Key | Default | Meaning |
|---|---|---|
| `manifest_dir` | `.eqty/manifests` | Where manifests are written, relative to Relay's working directory. One file per session, named by session id. |
| `deny_globs` | as above | Files whose content is left out of the manifest. The file still appears, with its path and content hash. This does not cover a file read through a shell command, or what the model was told about it. |
| `max_content_bytes` | 100 MiB | Content larger than this is recorded by hash only. |
| `provider_manifests` | none | Gateways to fetch per-call manifests from. See below. |

`nemo-relay plugins validate eqty.lineage` reports every problem in the configuration at once.

## Run

```bash
nemo-relay codex
nemo-relay claude
```

Everything after `--` goes to the agent, for example `nemo-relay claude -- -p "<prompt>"`. When the
session ends, its manifest is in `manifest_dir`. Codex has no session-end event, so its manifest is
written when Relay shuts down.

### Through a model gateway

To send the agent's model calls through a gateway such as LiteLLM, give Relay the gateway's address
and key:

```bash
# Codex
OPENAI_API_KEY=<gateway key> \
  nemo-relay --openai-base-url http://localhost:4000/v1 codex -- -m <model>

# Claude Code
ANTHROPIC_API_KEY=<gateway key> \
  nemo-relay --anthropic-base-url http://localhost:4000 claude -- --model <model>
```

The OpenAI address ends in `/v1`; the Anthropic one does not. `NEMO_RELAY_OPENAI_BASE_URL` and
`NEMO_RELAY_ANTHROPIC_BASE_URL` work in place of the flags. Codex always sends `prompt_cache_key` and
a `reasoning` setting; if the gateway's model rejects them, drop them in the gateway's configuration.

## Merge the gateway's manifest

A gateway running the EQTY LiteLLM plugin (`litellm-eqty-plugin`) signs its own manifest for each
call. Add a `provider_manifests` block and this plugin fetches that manifest for every matching call,
merges it into the session's manifest, and links each call to it.

```toml
[[plugins.dynamic.config.provider_manifests]]
base_url    = "http://localhost:4000"
api_key_env = "OPENAI_API_KEY"     # ANTHROPIC_API_KEY for Claude Code
models      = ["gpt-6-astra-*"]    # leave out to ask about every model
```

| Key | Required | Meaning |
|---|---|---|
| `base_url` | yes | The gateway's address. `https://`, or `http://` only to `localhost`, `127.0.0.1` or `::1`, because the key is sent to it. |
| `api_key_env` | no | The environment variable holding the gateway key: the same one you gave Relay. Leave it out only if the gateway needs no key. |
| `models` | no | Model names to ask this gateway about, as glob patterns, matched against the model the agent asked for. |

List only models that go through this gateway; the plugin sees a call's model name, not where it
went. For a second gateway add a second block. Each call is asked of the first block that matches,
so put the more specific block first.

The gateway content is merged whole: `deny_globs` and `max_content_bytes` do not apply to it.

### Check that it worked

Each manifest has a coverage node that counts what happened to the calls that matched a block:

| Counter | Meaning |
|---|---|
| `ProviderManifestMerged` | Fetched, merged and linked. |
| `ProviderManifestMissing` | The gateway had no manifest for the call, could not be reached, or did not answer in time. |
| `ProviderKeyUnset` | The variable in `api_key_env` was not set where Relay runs, so nothing was sent. |

None of the three means no call matched a block.

## Develop

```bash
cargo test --locked
cargo test --locked --manifest-path abi-test/Cargo.toml   # the C ABI suite
```

`just test-rust` from the repository root runs both. `docs/live-session-script.md` drives a real
session through every path the recorder covers.
