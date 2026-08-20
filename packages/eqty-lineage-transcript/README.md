# eqty-lineage-transcript

Offline EQTY lineage ingestion from Claude Code session transcripts
(`~/.claude/projects/<slug>/<session>.jsonl`).

```bash
python -m eqty_lineage.transcript <session.jsonl> -o manifest.json --triples triples.jsonl
python -m eqty_lineage.transcript --sweep 250      # parser coverage over local sessions
```

```python
from eqty_lineage.transcript import ingest, check_invariants, graph_stats

result = ingest("~/.claude/projects/<slug>/<session>.jsonl", manifest="out.json")
assert not check_invariants(result.triples)
```

Requires no live agent and no hook wiring, which makes it the conformance oracle for the live hook
adapter: both consume the same session and must produce the same graph, modulo an enumerated set of
divergences neither can avoid.

## What the transcript gives you that hooks do not

**Both sides of every edit, for free.** `Edit` and `Write` results carry `originalFile` (full pre-edit
content) alongside `oldString`/`newString`/`structuredPatch`. The post-state is reconstructed by replaying
the literal replacement, which is exact — so nothing has to be re-read from disk and a file that changed
again afterwards cannot corrupt the version chain. `userModified` is propagated: human co-authorship is
not recoverable any other way.

**Token accounting.** `message.usage` carries input/output/cache token counts per model call.

## What it does not give you, and how that is recorded

| Gap | Handling |
| --- | --- |
| Bash filesystem effects | `file-history-snapshot` deltas establish *that* a path changed, not what it became. The backup store holds *pre*-change content, so these are emitted identity-only with `observed=False` and attached to the conversation state rather than to a guessed command. |
| Subagent internals | `isSidechain` was false across all 2,072 local sessions; an `Agent` result carries aggregate `toolStats` only. Emitted with `opaque=True` and an explicit opacity reason, not as an empty subgraph. |
| Instructions and permission decisions | Absent from transcripts entirely. Hook-path only. |
| The model's request payload | A transcript stores the conversation, not the request. No `Prompt` entity is created rather than inventing one. |
| Partial reads | A `Read` with `startLine`/`totalLines` indicating a fragment cannot be content-addressed as a file version — a fragment's hash is not the file's hash. Recorded identity-only. |

## Transcript gotchas this handles

- **Compaction severs the record chain.** A `system` record with `subtype: "compact_boundary"` has
  `parentUuid: null` and reconnects via **`logicalParentUuid`**. Anything walking `parentUuid` naively
  splits a session into disconnected components.
- **`toolUseResult` is sometimes a bare string**, not a dict (626 of the Bash results in the surveyed
  corpus).
- **Resumed sessions carry orphan results** whose `tool_use` block lives in a different transcript file
  (~0.5% of tool results locally). File observations are dispatched on result *shape*, not tool name, so
  those reads and edits still land in the graph; the call itself is named `unknown` rather than guessed.
- **UI-state record types are allowlisted out** (`ai-title`, `mode`, `permission-mode`, `last-prompt`,
  `agent-name`, `attachment`, `pr-link`, `queue-operation`). Allowlisting means a record type added in a
  future release is ignored rather than misparsed.

## Known limitation: manifest export ceiling

`eqty-sdk` 2.2.0's `Context.export()` builds one query binding roughly three parameters per
`statement_graph_link` row and never chunks, so it hits SQLite's `SQLITE_MAX_VARIABLE_NUMBER` (32766):

```
RuntimeError: error returned from database: (code: 1) variable number must be between ?1 and ?32766
```

Measured against a real session: **10,730 links export, 11,285 fail** — and 32766 / 3 = 10,922 sits
between them. Roughly one local session in six is large enough to trip it. Statement recording and the
triple sidecar are unaffected, so `ingest()` records the failure as a warning and returns a usable
result rather than discarding work that succeeded. Chunking the export is an upstream fix.

## Verification

`--sweep N` parses without touching the SDK, so parser coverage is cheap:

```
swept 250 sessions, 0 failed
   45142  ModelCall      19760  ToolCallStarted     19760  ToolCallEnded
   11346  FileObserved    2675  PromptSubmitted        34  Compacted
```

`ToolCallStarted` and `ToolCallEnded` must balance — an imbalance means results are being dropped.

Recording 60 real sessions through the full recorder produced **zero invariant violations**
(`check_invariants`: no derivation cycles, no outputs without inputs).

`graph_stats()` reports `entities_multi_generator`, the alternative-derivation count. On that sample it
was **0.5% of generated entities, in 7 of 60 sessions** — the same tool result arriving from several
calls, e.g. four Bash calls all returning `"Error: This command requires approval"`. Content-addressed
entities are *values*, so one value legitimately having several generators is correct, not a defect. It
is also the measure that decides whether provenance polynomials are worth anything here: if it were
uniformly zero, agent graphs would be trees and how-provenance would buy nothing over plain reachability.
