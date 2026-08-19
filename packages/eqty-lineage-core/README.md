# eqty-lineage-core

Framework-agnostic lineage recorder shared by the EQTY lineage integrations. Adapters translate their
source — a Claude Code hook POST, a Codex hook, a session transcript on disk — into a small event
vocabulary; this package turns that vocabulary into EQTY assets, computation statements, and a parallel
triple fact set.

```python
from eqty_lineage.core import LineageRecorder, SessionStarted, ToolCallStarted, FileObserved

recorder = LineageRecorder(framework="claude-code")
recorder.handle(SessionStarted(session_id="...", agent="claude-code", agent_version="2.1.220"))
recorder.handle(ToolCallStarted(tool_use_id="t1", tool_name="Edit", tool_input={...}))
recorder.handle(FileObserved(path="/repo/app.py", content=b"...", mode="wrote", tool_use_id="t1"))
```

Nothing but the recorder touches `eqty_sdk`. That is what makes the offline and live capture paths
comparable: if they emit the same events they must produce the same graph, so any divergence is a bug in
one adapter rather than a difference of opinion about the SDK.

## Why not just reuse the LangChain handler

Most of `eqty-lineage-langchain` is generic lineage bookkeeping, but it is fused to two assumptions that
coding agents break.

**One CID per path.** The LangChain handler keys registered paths on the path alone and treats a second
sighting as *carried, not created*. A coding agent's entire purpose is `read → edit → read again`, so
that rule either hides every edit or puts a cycle in the graph. Here the key is `(path, content CID)`:
same bytes are the same entity, different bytes are a new version carrying `prov:wasDerivedFrom` back to
its predecessor and `prov:wasInvalidatedBy` forward.

**One in-process run tree.** Hook events arrive as separate HTTP posts or separate processes. The
recorder holds no framework objects and can be rehydrated per session.

## Events

| Event | Notes |
| --- | --- |
| `SessionStarted` / `SessionEnded` | agent, version, model, `permission_mode`, `effort`, git branch |
| `InstructionsLoaded` | CLAUDE.md / AGENTS.md / rules — *inputs*, not decoration |
| `PromptSubmitted`, `ModelCall` | `ModelCall` is offline-only; hooks never carry the messages |
| `ToolCallStarted` / `ToolCallEnded` | correlated on `tool_use_id`; `is_error` keeps failures in the graph |
| `FileObserved` | `mode` read/wrote/changed, plus the `observed` honesty flag |
| `SubagentStarted` / `SubagentEnded` | `opaque=True` when the capture path cannot see inside |
| `Compacted` | a lossy edge, with the token numbers attached |
| `PermissionDecision` | what was *permitted*, not what happened |

Unknown event types are ignored rather than fatal — adapters run against agents that add hook events
between releases, and a recorder that crashed on one would take down the session it is observing.

## Asset mapping

The SDK already ships 22 asset types; coding agents map onto the existing set with no new primitives.

| Concept | Asset |
| --- | --- |
| the coding agent (CLI + version + model + permission mode) | `Agent` |
| source file version | `Code` (docs → `Document`, config → `Configuration`, else `Dataset`) |
| CLAUDE.md, rules | `SystemPrompt` |
| permission decision | `Guardrail` |
| tool definition or Bash command | `Tool` |
| prompt / model / response | `Prompt`, `Model`, `Reasoning` |

## PROV mapping, and three edges the standards lack

`add_computation_statement(inputs, outputs)` already *is* `prov:Activity` + `prov:used` +
`prov:wasGeneratedBy`; a `Signer`/DID is a `prov:Agent`. See `prov.py` for the full mapping. Three edges
have no standard equivalent:

- **`eqty:wasCompactedFrom`** — a *lossy* derivation. Nothing in PROV, OpenLineage, or in-toto models
  "derived from a summary of X where most of X was dropped". Real sessions routinely drop the
  overwhelming majority of their context; a graph that omits the boundary asserts a completeness it does
  not have.
- **`eqty:authorizedBy`** — what the agent was *permitted* to do. Recorded via the SDK's own
  `Association` (`CERTIFIES`) against the activity. This is the difference between a lineage graph and a
  governance record.
- **`observed` vs inferred** — a flag on every statement, not an edge. A filesystem change attributed to
  a Bash command by diffing snapshots is attribution, not observation.

## Triples

Every edge is also written as `(subject, predicate, object)` over CIDs, optionally appended to a JSONL
sidecar. One artifact, two purposes: an RDF/PROV-O serialization of the same graph, and directly the EDB
a Datalog evaluator consumes — so the query engine never has to be chosen at capture time. Only identity
lives there; content stays in the SDK blob store, which keeps a heavy session to low thousands of facts.

```python
from eqty_lineage.core import TripleSink

recorder = LineageRecorder(triples=TripleSink(".eqty_sdk/triples.jsonl"))
```

## Redaction is a gate, not an option

Instrumenting a coding agent means every file read, every command's stdout, and every diff is a candidate
for the blob store. `ContentPolicy` denies by path pattern (`.env*`, `*.pem`, `id_rsa*`, `*/.ssh/*`,
`*credentials*`, the SDK's own `signers/`…), scrubs a short list of content patterns, and caps stored
size at 1 MiB.

A denied file **still becomes a graph node**: its path and true content CID are recorded so lineage stays
complete, while its bytes are withheld — the denied bytes are never handed to an asset constructor at
all. Provenance does not require publication.

```python
from eqty_lineage.core import ContentPolicy

LineageRecorder(policy=ContentPolicy(deny_globs=(*ContentPolicy.deny_globs, "*.internal")))
```

`PERMISSIVE` exists for fixtures and is never appropriate against a real repository.

## Projection: the part a human can read

A coding session produces one to two orders of magnitude more graph than a LangGraph run — measured on
real sessions, 136 to 1,995 assets and up to 1,359 activities, of which `Dataset` and `Reasoning` are
around **88%**. Every model response and tool-argument blob is a node, so the file provenance is buried.

```python
from eqty_lineage.core import project_manifest, select_file_lineage

project_manifest("full.json", "lineage.json", select_file_lineage(recorder.triples))
```

Keeps file versions, the tool calls that touched them, the `Tool` asset carrying the command, the agent,
any authorizing `Guardrail`, and compaction edges. Drops model calls. On a real session: 44 computations
→ 6 (all 20 model calls and 18 of 21 non-file tool calls removed), 908 KB → 130 KB.

**A projection is a subset of the signed manifest, never a re-signing.** Every kept statement keeps its
original CID and its `CredentialRegistration`, so the projection verifies exactly as the full manifest
does — and it round-trips through `Context.import_manifest`. Re-recording the selection into a fresh
context would mint new statement CIDs (they carry `validFrom`) and sign a *different* document.

## Canonical comparison

`canonical_triples` / `graph_diff` compare two graphs. Entities are content-addressed so their CIDs
match across runs; **statements are not** — a statement CID covers a credential carrying `validFrom`,
so the same computation recorded twice gets different activity CIDs. Activities are therefore relabeled
by their `(inputs, outputs)` signature. Corollary: two manifests of the same computation are never
byte-identical, so byte-comparison regression gates cannot work.
