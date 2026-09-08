# eqty-lineage-nemo-relay

Records a Claude Code or Codex session as a signed EQTY lineage manifest, by registering an EQTY
subscriber inside [NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) and calling the
`integrity` Rust core directly.

## What this is

Relay already installs itself into Claude Code and Codex, receives their lifecycle hooks, and proxies
their LLM traffic through a local gateway. Internally it emits one stream of events and lets plugins
subscribe to it — which is exactly how its own ATOF and ATIF exporters are built:

```rust
// nemo-relay: crates/core/src/observability/plugin_component.rs
ctx.register_subscriber("atof", subscriber)?;   // :1259
ctx.register_subscriber("atif", dispatcher)?;   // :1344
```

This crate registers a third subscriber on the same stream, as a peer rather than as a consumer of
their output files:

```rust
ctx.register_subscriber("eqty_lineage", move |event: &Event| { … })?;
```

That matters for fidelity. We see raw events before ATIF normalization drops marks and before ATOF
serializes anything, and an in-process subscriber can reach `Event::annotated_request` — the typed
LLM request object, rather than the serialized `data` that file consumers get.

The manifest is written by the same code the Python SDK uses. `eqty_sdk.Context.export()` calls
`integrity::lineage::models::manifest::generate_manifest`; so do we, directly:

```rust
pub async fn generate_manifest(
    include_context: bool,
    statements: Vec<Statement>,
    blobs: HashMap<String, String>,
) -> Result<Manifest>
```

Same function, same `Statement` type, same signing and CID code underneath — so a manifest produced
here verifies exactly as one produced by the Python SDK today.

## Status

Phases 2 and 3 of [the plan](../../../eqty-lineage-nemo-relay-plugin-plan.md). The plugin loads,
validates its configuration, classifies a real event stream, and writes a signed manifest whose
statement and asset types match the shipped DeepAgents and deep-research manifests.

| | |
|---|---|
| loads through the C ABI and registers `eqty.lineage` | yes — `abi-test/tests/lifecycle.rs` |
| config validated, all problems reported at once | yes — `tests/config.rs` |
| classifies a real Codex capture | yes — `tests/classify.rs` |
| attributes gateway events to a session | yes — `tests/session.rs` |
| builds a signed manifest from Rust | yes — `tests/lineage.rs` |
| asset CIDs match `eqty_sdk` exactly | yes — `tests/lineage.rs`, golden vector |
| derives file versions from tool results | yes — `tests/files.rs` |
| file identity, replay chain, redaction gate | yes — `tests/recorder.rs` |
| classified events reach the recorder | yes — `src/mailbox.rs` |
| a real event stream writes a manifest | yes — `tests/end_to_end.rs` |
| model calls recorded from typed payloads | yes — `tests/end_to_end.rs` |
| prompts, subagents, compaction, `apply_patch` | yes |
| same document shape as the shipped integrations | yes — see below |
| a live session writes a manifest | yes — two Claude Code sessions |
| a live session writes **file** lineage | **not yet** — see `docs/live-session-script.md` |

## The completeness bar, set by evidence

"Complete" here means *the same kind of document the LangChain and DeepAgents integrations already
produce*, checked against the manifests in `manifests/` rather than against an opinion:

```
NeMo Relay plugin       {DataRegistration, MetadataRegistration, CredentialRegistration, ComputationRegistration}
                        {Agent, Dataset, Document, Model, Prompt, Reasoning, System_Prompt, Tool}

DeepAgents (shipped)    {same four}
                        {same eight}
```

Same statement types, same asset vocabulary, everything content-addressed. The shipped manifests
contain **no** `EntityRegistration` at all, and neither does this one: a node whose identity is a
fresh UUID cannot join across runs.

Activities carry `computation_type` and `performedBy`, and the shapes match too — a model call is
`[Model, System_Prompt, Prompt] → [Reasoning]`, a tool run is `[Tool, reads…] → [result, writes…]`.

**Attribution is metadata on the activity, never an input.** Putting an agent in `inputs` would
assert that the activity *consumed* the agent. PROV keeps association and usage apart, so a subagent
is named in the computation's metadata and the graph stays honest about what was used.

## Why native, in one test

`the_same_conversation_through_two_providers_is_one_prompt` is the reason this is an in-process
plugin rather than a consumer of exported ATOF files.

Model calls are recorded from `annotated_request` / `annotated_response` — Relay's **typed,
provider-normalized** objects, not the serialized `data` a file consumer sees. So the prompt CID is
computed over a normalized conversation, and the identical exchange through Anthropic Messages and
through OpenAI Responses hashes to the same prompt. Two sessions on different providers join on that
node instead of forking on wire format.

Hashing raw provider JSON would fail this *silently*: both manifests would look perfectly well-formed
while describing the same prompt as two different things. Normalization reaches the vocabulary too —
Anthropic's `end_turn` and OpenAI's `stop` both arrive as `FinishReason::Complete`.

## How an event reaches disk

```
Relay subscriber (sync, must return promptly)
  └─ classify + attribute + try_send        ← the only work on this thread
       └─ bounded queue (4096)
            └─ worker thread, current-thread runtime
                 └─ one Recorder per session
                      └─ SessionEnded *or* Drop → generate_manifest → {session_id}.json
```

Three constraints decide that shape. The subscriber is synchronous and must return, so it classifies
and hands off. The recorder is async and single-owner, so one worker thread owns every recorder — an
actor, not a shared lock. And **Codex never closes its agent scope**, so `Drop` is not cleanup: it is
the only export path a Codex session will ever take.

The queue is bounded on purpose. Unbounded would turn a slow recorder into unbounded memory inside
the agent's process; blocking would stall the agent, which a collector must never do. Overflow is
counted instead, so the manifest can say it happened rather than quietly under-reporting.

## What a file node means

Three rules, each of which looks like over-thinking until the case that motivates it appears.

**`(path, content)` is the dedup key; content alone is the identity.** The same bytes are one asset
wherever they live — the path travels as metadata, which is the property
`eqty-lineage-deepagents@0.2.0` shipped. A file copied or moved is one node with two things said
about it. But each path keeps its own version record, because keying versions on content alone would
lose the fact that two files were touched.

The exception is the identities below that have no content: `unknown:{path}` is path-derived because
with no bytes there is nothing else to be identical about, and two files nobody read cannot be shown
to be the same file.

**There are three ways not to know, and they must not collapse.** A real content CID means we hold
the bytes. `unknown:{path}` means we saw the path and never established its content. `deleted:{path}`
means the file is gone. Merging the last two would let a deletion deduplicate against a failed read
of the same path, and the graph would assert a removal nobody observed.

`deleted:{path}` is **not reachable yet**: `FileMode` has only `Read` and `Wrote`, so nothing
constructs it. A Codex `Delete File` currently records as a write whose content is unknown, which is
a weaker and different claim. Claude Code has no delete tool at all — deletions go through `Bash`,
where no file effect is observable in the first place.

**Identity is computed before redaction, never after.** Two different secrets at one path scrub to
the same placeholder; hashing what was stored rather than what was seen would merge them into one
version. Content withheld by policy still becomes a node carrying its path and its true content CID —
provenance does not require publication — and that node is content-addressed on the true CID alone,
so it is deterministic across runs *and* independent of where the file sat. Two recordings that read
the same secret join on it whatever path each saw it at.

## Running it against a live session

`docs/live-session-script.md` is a twelve-turn script that drives every path this recorder can record
today — file versions and the replay chain, partial reads, subagents, redaction, the size ceiling,
non-UTF-8 content, a failing tool, and compaction — plus a verification block that prints `YES` or
`MISSING` per capability.

It exists because the first two live sessions reached for `Bash` almost exclusively and therefore
exercised roughly a third of the recorder: no `Document` nodes, no subagents, and none of
`ContentRecovered`, `ContentUnknown`, `Compaction` or `PayloadTooLarge`. Fixtures cover those; a real
session had not.

## `src/lineage/` is written to be liftable

The statement composition lives in `src/lineage/`, and nothing Relay-shaped is allowed in there. That
is deliberate: `integrity` supplies primitives, but "an asset" is three statements in an arrangement
that differs depending on whether the asset has content, and that composition is what `integrity-py`
contributes. We are the second implementation of it.

Whether it should become a feature of `integrity` or a standalone `integrity-rs` is open. Writing it
already-extracted costs nothing now and makes the answer cheap later.

It does **not** copy `integrity-py`'s process-global config. This plugin is long-lived and records
many sessions concurrently, so a global active signer is a shared mutable several sessions would race
over — it is also why `eqty_sdk.init()` is silently ignored on a second call. Here the session owns
its signer and is passed explicitly.

## Two dependency facts that look like bugs

**The ABI test is a separate crate.** `integrity` reaches `sha2` through `iroh-blobs → iroh-base`,
which pins it at exactly `=0.11.0-rc.5`. Relay's core crate asks for `^0.11`, and semver excludes
pre-releases from that range. The two are unsatisfiable together, so `abi-test/` has its own
workspace and lockfile. It costs nothing real — that crate needs no `integrity`, and the plugin needs
no `nemo-relay` core, since `nemo-relay-plugin` re-exports the types it uses.

**Statement CIDs are never reproducible.** They cover a credential carrying `validFrom`, so two
recordings of identical work never match byte for byte. Content CIDs have no such problem, which is
why the cross-language check in `tests/lineage.rs` is over content and not over statements.

## Two things the fixture taught us

`tests/fixtures/codex-session.jsonl` is a real Codex session captured through Relay. Two findings
from it are baked into the design, and both are easy to get wrong by assumption.

**Gateway events carry no session id.** Hook-path events have `session_id` in metadata. LLM scopes
arrive through the gateway instead, and their metadata is `gateway_path`, `llm_correlation_status`
and `otel.status_code` — nothing else. Session attribution therefore comes from the `parent_uuid`
scope tree, not from the event, which is why `session.rs` exists at all.

**Codex never closes its agent scope.** Its plugin hook schema has no `SessionEnd`, so Relay emits no
`agent` scope end and `SessionEnded` never fires. Flushing on `Drop` is not tidy-shutdown hygiene
here; on Codex it is the only path that will ever write a manifest.

## Not knowing is recorded as not knowing

ATOF 0.1 defers a terminal `status` field on scope end. Relay fills the gap in metadata — deriving
`error` from `PostToolUseFailure`, `denied` from a permission denial — and strips nulls, so the key
is present only when it knows. `is_error` is therefore `Option<bool>` and stays that way.

On Codex that means every tool call is `None`: its hook schema has no failure event, so `ls
/nonexistent` and `echo hello` are indistinguishable. Recording `false` would turn *we did not
observe a failure* into *we observed a success*, and attest something nobody saw.

All three reach the graph. Each tool activity carries `outcome` — `"failed"`, `"succeeded"`, or
`null` — and coverage counts them as `ToolCallFailed`, `ToolCallSucceeded` and
`ToolCallOutcomeUnknown`. A manifest where the third is the only nonzero count is telling a reader
that this capture path cannot see tool failure, which is a different claim from a session that had
none.

The same rule governs correlation. Relay reports how confident its own join was, and
`agent_fallback` / `ambiguous_fallback` mean it guessed. Those become `Correlation::Inferred`, never
`Observed`.

## The lockfile is committed, and has to be

`integrity` does not resolve from a clean crates.io index at any revision: `core2 0.4.0` is yanked
and reachable through two different paths. Cargo permits a yanked crate already present in a lockfile
and refuses to select one that is not.

So `Cargo.lock` is committed here, and this is not the usual "libraries don't commit lockfiles"
question — without it, a clean checkout cannot build. A CI job that deletes it, or a `cargo update`
that drops the `core2` entry, fails with an error pointing five levels down the dependency tree
rather than at the real cause.

`integrity` is pinned by SHA to v0.0.13 (`0d85ae5b9f7204812771c4f8e4985c671b831b0a`).

## Building

```bash
cargo test                       # 19 tests, including a real load through the C ABI
cargo build --release            # target/release/libeqty_lineage_nemo_relay.dylib
shasum -a 256 target/release/libeqty_lineage_nemo_relay.dylib
```

The digest goes in `relay-plugin.toml` under `[integrity] sha256`, which Relay verifies against the
library before loading it regardless of attestation policy. Release CI generates it; a stale digest
is an install Relay refuses, and that should be caught before release rather than by a user.

Note that `[integrity] sha256` is **NVIDIA's** artifact digest and has nothing to do with the EQTY
`integrity` crate this plugin links. The collision is unfortunate; do not conflate them.

## Installing into Relay

```bash
just nemo-relay-package                                            # stage dist/relay-plugin
nemo-relay plugins validate ./dist/relay-plugin/relay-plugin.toml
nemo-relay plugins add --user ./dist/relay-plugin/relay-plugin.toml
nemo-relay plugins inspect eqty.lineage
nemo-relay plugins enable  eqty.lineage
```

**Install from the staged directory, never from this package.** Relay copies the whole directory
containing `relay-plugin.toml` into an activation snapshot against a 512 MiB budget; from the package
root that closure is `src/` + `tests/` + `target/` + `abi-test/target/`, and the gateway refuses to
start citing an unrelated file. `just nemo-relay-package` stages a directory holding only the dylib,
its signature and the config schema, and stamps the digest into the staged copy.

**It also signs, and that is not optional.** `plugins add` evaluates trust with attestation
defaulting to `integrity_only`, but every activation path hardens it to `signature_required` first —
so an unsigned plugin installs cleanly, validates cleanly, and then will not start. The recipe prints
the `[plugins.policy.overrides."eqty.lineage"]` block to paste into `plugins.toml`.

`enable` changes lifecycle state only — it does not load code. Relay validates and loads enabled
plugins when the gateway starts, so a change takes effect on the next sidecar start. A clean
`validate` is **not** an activation dry-run: it evaluates the un-hardened policy.
