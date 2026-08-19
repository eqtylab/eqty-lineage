# eqty-lineage-agent-hooks

Live EQTY lineage capture from Claude Code and Codex hooks.

```bash
export EQTY_LINEAGE_TOKEN=$(openssl rand -hex 16)
eqty-lineage-hooks serve --port 8787 --manifests ./manifests --watch /path/to/repo
eqty-lineage-hooks install --print          # settings.json / hooks.json wiring
```

```jsonc
// .claude/settings.json
{ "hooks": { "PreToolUse": [{ "hooks": [{ "type": "http", "url": "http://127.0.0.1:8787/hook" }] }] } }
```

## HTTP is the recommended transport

Claude Code can deliver hooks as **HTTP POSTs**, not only as command invocations. That matters more than
latency: with a persistent daemon the session's recorder and run tree stay in memory for the session's
lifetime — the same shape the LangChain handler has always had.

**Except `SessionStart`, which accepts no HTTP handler** — only `command` and `mcp_tool`. Wiring it as
HTTP is accepted by `settings.json` and then never delivered, and it is the worst event to lose: it
creates the `Agent` asset *and* returns `watchPaths`, so without it the manifest attests a transcript
rather than a computation and no `FileChanged` ever fires. `install` emits `SessionStart` as a `command`
hook that curls the daemon, and everything else over HTTP. Verified against Claude Code 2.1.220 by
running a session with each wiring and counting what arrived: 5 events with SessionStart over HTTP, 20
with it as a command hook.

The command fallback (`eqty-lineage-hooks hook`, one payload on stdin) is genuinely lossier. A fresh
process cannot hold an open tool call in memory, so Pre/Post correlation depends on the sidecar rather
than on the run tree, and it pays interpreter startup per event (~40 ms measured with `init()` and a
signer). Use it only where a daemon cannot run.

Bind to loopback. Every payload carries prompts, file contents, and tool output. The bearer token is read
from `$EQTY_LINEAGE_TOKEN`, not a flag, so it is not visible in `ps`; `install` emits it as an env-var
reference (`headers` + `allowedEnvVars`) so the settings file carries no secret. File contents are not
stored unless `--no-blobs` is passed.

## What the live path gets that transcripts cannot

**Bash side effects become `observed`.** `SessionStart` returns `watchPaths` in its `hookSpecificOutput`;
the resulting `FileChanged` events record what a shell command actually did. The offline path can only
infer this from snapshot deltas and marks it `observed=False`. This is the single strongest reason to
run the daemon.

Two things bound that claim. A watcher names a path and the adapter then reads it, so the content is
*as-of-read*, not as-of-change — a second write landing in the gap is recorded as the content of the
first event. And `change_type` distinguishes `change` from `unlink`: a removal is recorded as a tombstone
rather than a version whose bytes failed to load, because those are opposite claims. Observations of the
daemon's own storage (the manifest and sidecar directories, anything under `.eqty_sdk`) are dropped —
otherwise recording a file version writes blobs, the watcher reports them, and the recorder records
those. Left in, one real edit came back as 55 file versions, 54 of them the recorder watching itself.

**Instructions are inputs.** `InstructionsLoaded` names each CLAUDE.md and rules file as it enters
context; they become `SystemPrompt` assets that everything downstream depends on. Transcripts have no
record of them at all.

**Permission decisions exist.** `PermissionRequest` / `PermissionDenied` become `Guardrail` assets linked
to the activity by `eqty:authorizedBy` — what the agent was *permitted* to do, not just what it did.

**Subagents are transparent** (`opaque=False`), where a transcript sees only aggregate `toolStats`.

## Correctness details that are easy to get wrong

- **`PostToolUse` fires only on success.** Without also subscribing to `PostToolUseFailure`, every failed
  tool call vanishes — and a failed call is often the one an audit cares about. The failure payload
  carries its outcome under `error`, not `tool_response`; reading the success key records the call with
  its substance missing, which is worse than dropping it, and a test written against the same assumption
  passes.
- **`PostToolBatch`** is the correlation primitive for parallel tool calls: one payload closing several.
  Entries are `{tool_name, tool_input, tool_use_id, tool_response}` — the same result key as
  `PostToolUse`, with no per-call error flag. (The published docs describe the array as `batch`; the
  CLI sends `tool_calls`.)
- **Codex reports outcomes in a string, and mostly does not report failure at all.** Its `tool_response`
  is plain text, so a dict-only error check can never see a Codex failure. Where the text begins
  `Exit code: N` — the `apply_patch` shape — that is read. Shell results carry no such marker: captured
  from codex-cli 0.145.0, a failing `ls /nonexistent` returns `"ls: /nonexistent: No such file or
  directory\n"` and a succeeding `echo hello` returns `"hello\n"`. Bare output either way. **A failing
  Codex shell command is therefore recorded as a successful one**, and nothing in this adapter can fix
  that; the signal is not in the payload. Pinned by a test so a future codex-cli that starts reporting it
  shows up as a failure here.
- **Dialect detection is automatic** — `turn_id` is Codex-only, `prompt_id` is Claude-Code-only.
- **Unknown hook events yield nothing rather than raising.** Both agents add events between releases;
  an adapter that crashed on one would take down the session it is observing.
- **A recording failure never propagates.** The receiver logs and returns `{}` rather than failing the
  hook.

## Policy: recording must not grant authority

`HookPolicy` answers `PreToolUse` with `deny` or defers. It never answers `allow` — returning `allow`
from a hook *overrides* the user's own settings, so a lineage recorder that allowed by default would
silently widen the agent's permissions. Returning nothing defers to the normal permission flow.

```bash
eqty-lineage-hooks serve --allow-write '/repo/*' --deny-write '*.pem' --dry-run
```

`--dry-run` logs what would be denied and answers `defer` — the honest way to roll a policy out. It
answers `defer` rather than `allow` for the reason above: reporting mode must not quietly grant
permission on precisely the calls it is flagging.

`apply_patch` is checked by parsing the patch document. It carries no path key, so a policy that looked
for one deferred on every Codex write while listing `apply_patch` as a write tool — enforced-looking and
checking nothing. Every file in a patch is checked, not just the first.

## Contexts

The daemon holds one SDK `Context` per session, and recording runs inside `graph_context(ctx)`.

This matters because the SDK resolves context in two different ways: asset constructors fall back to
`get_active_context()`, but statement constructors take an explicit `context=` kwarg and do **not** fall
back. Omitting it sends every computation statement to the process default context while the assets it
references sit in the session context — the graph splits in half and the manifest exports empty.
`LineageRecorder` resolves the active context and passes it explicitly for exactly this reason.

`init()` is process-global and raises on a second call, so a multi-session daemon cannot use
`init(default_context=…)` the way the offline ingester does.

## Codex

Verified end-to-end against **codex-cli 0.145.0**: 18 hook events, 0 errors, a manifest and triple
sidecar written for a live session. `eqty-lineage-hooks install --dialect codex` prints the config.

Four things differ from Claude Code and all of them were found by capturing real payloads, not by
reading docs:

- **Config is TOML, not JSON**, and there is no HTTP hook type — the command hook shells out to `curl`.
  Writing it as a named profile (`~/.codex/eqty.config.toml`, used via `codex exec -p eqty`) leaves the
  user's own `config.toml` untouched.
- **Hooks must be trusted before they run.** A freshly configured hook silently does not fire; automation
  that vets its own hook sources passes `--dangerously-bypass-hook-trust`.
- **Codex writes with `apply_patch`**, whose `tool_input.command` is a patch document and whose
  `tool_response` is a plain string. Neither carries `filePath`, `content` or `structuredPatch`, so the
  shape-dispatched parser in core sees nothing — a Codex session yields *no file lineage at all* unless
  the patch itself is parsed. `parse_apply_patch` does that: an `Add File` section is fully recoverable
  (every line is a `+` line), while an `Update File` section carries only hunks, so the path and its
  transition are recorded and the content is left unknown rather than guessed.
- **Lifecycle events are not turn-scoped.** `SessionStart` and `SessionEnd` carry no `turn_id`, so
  dialect detection cannot rely on it alone and falls back to `transcript_path`
  (`~/.codex/sessions/…` versus `~/.claude/projects/…`). This is not cosmetic: `SessionStart` sets the
  agent name recorded in the `Agent` asset, so getting it wrong attests a Codex session as Claude Code.

The offline path does **not** cover Codex. Its rollout files under `~/.codex/sessions/` are a different
format from Claude Code transcripts, and `eqty-lineage-transcript` parses only the latter. "Supports
Codex" today means the live hook path.

## Counterfactual policy replay

```bash
eqty-lineage-hooks replay session-triples.jsonl \
    --deny 'no-web=*.html' --deny 'no-web=*.js' --allow 'rust-only=*.rs'
```

```
as-recorded: no tool call refused; the session is unchanged
no-web:      21 call(s) refused, 42 entities unreachable (10 file versions)
rust-only:   34 call(s) refused, 68 entities unreachable (19 file versions)

611 artifacts survive every policy; 53 are policy-contingent.
    …/physics/build.sh  survives only under: as-recorded, no-web
```

A permission hook is a **handler**: it interprets a tool operation and answers allow, deny, or defer. A
recorded trace is a term over those operations, so reinterpreting it under a different handler is what
"what if the policy had been stricter" means. Denying an activity removes what it produced *and
everything derived from that*, which is ordinary reachability over the graph the manifest already
contains.

Replay calls **the same `HookPolicy` object that runs live**, given a synthesised payload. A separate
implementation would let the counterfactual and the enforcement drift apart, which would make the answer
worse than not having it.

**What replay cannot tell you.** Denying a tool call changes what the model would have done next, and
that is unknowable without re-running. Replay reports *which recorded artifacts become unreachable* —
never *what the agent would have done instead*. For the latter, replicate under the policy with
`eqty-lineage-replicate` and compare; that is a different and more expensive question.

Replaying under K policies yields a determination space over **policies** rather than runs, so the
supports, `qdepth` and robustness queries from `eqty_lineage.query.determination` apply unchanged. One
algebra, two instantiations: an artifact with full support is invariant to the governance change, one
with partial support exists only because a particular policy allowed it.

## Conformance: both capture paths must agree

```bash
eqty-lineage-hooks verify --limit 25
```

Ingests a real session transcript, separately replays it as hook payloads through the same recorder the
daemon uses, and classifies every divergence. **25/25 local sessions conform**, with **zero** live-only
divergence.

The assertion is not that the graphs are identical — it is that every divergence falls into a declared
bucket, that each declared bucket is non-empty (one that silently empties means the capture path moved
underneath its declaration), and that nothing else divides them.

Two facts about identity make this possible:

- **Entities are content-addressed**, so their CIDs match across runs, machines, and capture paths. No
  RDF canonicalization is needed — RDFC-1.0 exists to label blank nodes, and there are none here.
- **Statements are not.** A statement CID covers a signed credential carrying `validFrom`, so the same
  computation recorded twice gets two different activity CIDs (measured: identical inputs, outputs,
  signer and context, 2.2 s apart, different CID). Activities are therefore compared by their
  `(inputs, outputs)` signature via `eqty_lineage.core.canonical`. A corollary worth knowing: **two
  manifests of the same computation are never byte-identical**, so byte-comparison regression gates
  cannot work.

Declared divergences: model-call artifacts and token usage are transcript-only (hooks never carry
messages); `InstructionsLoaded` and permission decisions are hook-only; snapshot-inferred file versions
are transcript-only (the live path observes them via `FileChanged` instead).

**Two limits, stated rather than papered over.** Both paths share `eqty-lineage-core` — and tool-result
parsing was moved there deliberately so they cannot drift — so this exercises the two *adapters*, not
the recorder; correlated faults in shared code are invisible to it, the classic N-version result.
Replay also cannot produce events a transcript has no record of, so `InstructionsLoaded`, `FileChanged`,
`PermissionRequest` and subagent hooks are declared rather than demonstrated.

## Verification

`test_daemon.py` runs the real server over HTTP and asserts: bearer-token rejection, `watchPaths`
returned on `SessionStart`, three versions of one file across an Edit and a watched out-of-band change,
a failed call surviving in the graph, a parallel batch closing two calls, a denied out-of-set write, an
in-set write deferring, zero inferred edges (everything observed), and a non-empty exported manifest.
