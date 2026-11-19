# eqty-lineage-nemo-relay design notes

Why the plugin is built the way it is. Installation and configuration are in the [README](../README.md).

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
subscriber::register(ctx, "eqty_lineage", Box::new(move |delivery| { … }))?;
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

Phases 2 through 4 of the design plan, which is tracked outside this repository; workspace
snapshots are what remains. The plugin loads, validates its configuration, classifies a real event
stream, and writes a signed manifest whose statement and asset types match the shipped DeepAgents
and deep-research manifests -- one document per session, with the model's own conversation sealed
into an iroh collection per type, and a mark for it on Relay's own event stream.

| | |
|---|---|
| loads through the C ABI and registers `eqty.lineage` | yes — `abi-test/tests/lifecycle.rs` |
| records events the host emits across the C ABI | yes — `abi-test/tests/lifecycle.rs` |
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
| seals the conversation into iroh collections | yes — `tests/end_to_end.rs`, `src/recorder.rs` |
| announces the document as a mark | yes — `tests/end_to_end.rs` |
| a live session writes a manifest | yes — Claude Code and Codex |
| merges the manifest a provider gateway signed for a call | yes — `tests/end_to_end.rs`; live through LiteLLM |
| a live session writes **file** lineage | yes on Claude Code — `FileRead: 5`, `FileWritten: 3` |
| the same, on Codex | `apply_patch` only — every read is a shell command, so `FileRead: 0` |

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
                      └─ SessionEnded *or* Drop → seal + generate_manifest → {session_id}.json
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

A Codex `*** Delete File` constructs `deleted:{path}`. Its node says `contentState: "deleted"` and is
an *output* of the run that removed it — as an input it would say the tool consumed what it deleted.

That directive is the only route to it, and in practice neither agent takes it when deleting is the
task: both reach for `rm`, Claude Code because it has no delete tool and Codex by preference. A shell
command carries no observable file effect, so the path keeps its last version and the manifest
disagrees with the disk. Only a deletion riding inside a patch the model was already writing reaches
`deleted:{path}`. Closing the rest needs a filesystem delta rather than a better reading of tool
results — a shell command can only be guessed at, and a guessed removal is exactly the claim this
section refuses to make.

**Identity is computed before redaction, never after.** Two different secrets at one path scrub to
the same placeholder; hashing what was stored rather than what was seen would merge them into one
version. Content withheld by policy still becomes a node carrying its path and its true content CID —
provenance does not require publication — and that node is content-addressed on the true CID alone,
so it is deterministic across runs *and* independent of where the file sat. Two recordings that read
the same secret join on it whatever path each saw it at.

**Withholding a file withholds it from the call that touched it, too.** A `Read` result *is* the file;
a `Write`'s arguments *are* the file. So a policy applied only to the file node left the same bytes
in the graph beside it, and the manifest reported a redaction it had not performed — the worst of the
three outcomes, because it tells a reader the secret is not there. Both payloads now inherit the
policy of every path their call touched, and say `withheldBecause: quotes-a-denied-file`. Withholding
is all-or-nothing per payload: partially scrubbing a JSON blob is a guess about where the bytes are,
and a wrong guess is a leak wearing the shape of a redaction.

The floor this cannot reach has two parts. The **shell**: `cat /app/.env` through `Bash` attributes
its output to no path we can see, so nothing marks it — the same visibility gap as tool file effects,
and on Codex that is every read. And the **model payloads**: a `prompt` or `completion` carries
whatever a tool result put into the conversation, and neither is attributed to a path either.

So `deny_globs` withholds a file's node and the tool payloads named against it, and nothing further.
It is a floor rather than a guarantee. A session that reads secrets through a shell is outside it,
and so is the conversation that read carried.

## How a provider manifest is linked

Relay does not show a plugin any response headers, so the plugin never sees the gateway's own
request id. Both sides do see the `x-nemo-relay-request-id` that Relay adds to every upstream
request. The plugin uses it in two requests:

1. `GET /integrity/logs/by-client-request-id/{id}` returns the gateway's record of the call.
2. `GET /integrity/manifest/{request_id}` returns the manifest that record points to.

The gateway has to serve the first route. `litellm-eqty-plugin` does, and its
`EQTY_CLIENT_REQUEST_ID_HEADER` setting names the header it reads.

The two manifests have no content id in common. The gateway hashes its own copy of each request
and response. Relay gives a plugin a re-encoded copy, and after the first call of a turn that copy
no longer holds the whole input. So the plugin writes the link as two `provider_link` computations
per call, signed by the plugin and marked `matchedBy: x-nemo-relay-request-id`:

- one from the session's prompts to the gateway's `Client Request`,
- one from the gateway's `Delivered Response` (or `Response Body`) to the session's completions.

The gateway's statements are merged exactly as the gateway served them. They are not decoded and
re-encoded, so each one still hashes to the id its credential signed. Its anchors are kept too.

## Running it against a live session

`docs/live-session-script.md` is an eleven-turn script that drives every path this recorder can record
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

**The ABI test is a separate crate.** `integrity` turns on serde_json's `arbitrary_precision`
through `ssi`, and cargo unifies features across a dependency graph. In one graph the Relay host
under test would serialize and decode with a feature no real gateway has, so `abi-test/` has its own
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

## The lockfile is committed

`Cargo.lock` is committed here, and this is not the usual "libraries don't commit lockfiles"
question: the shipped artifact is a signed cdylib, and `just nemo-relay-package` builds `--locked`
so a release bundle is built from the dependency tree that was tested.

`integrity` is pinned by SHA to v0.0.21 (`13af564f0061d7b15819efc99bccef5b6140a1ad`).

