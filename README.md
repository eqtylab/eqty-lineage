# eqty-lineage

Workspace of EQTY lineage integrations, published to `pypi.eqtylab.io`. Each integration lives under
`packages/` as its own distribution sharing the `eqty_lineage` import namespace:

| Package | Import | Purpose |
| --- | --- | --- |
| `eqty-lineage-core` | `eqty_lineage.core` | Agent-agnostic event model, recorder, redaction policy, triple sidecar |
| `eqty-lineage-transcript` | `eqty_lineage.transcript` | Offline ingester for Claude Code session transcripts |
| `eqty-lineage-agent-hooks` | `eqty_lineage.agent_hooks` | Live capture daemon for Claude Code and Codex hooks |
| `eqty-lineage-query` | `eqty_lineage.query` | Semiring-annotated Datalog over the recorded graph |
| `eqty-lineage-query-rs` | `eqty_lineage_query_rs` | Optional native accelerator for the query engine |
| `eqty-lineage-replicate` | `eqty_lineage.replicate` | N-run driver and robustness report |
| `eqty-lineage-langchain` | `eqty_lineage.langchain` | Callback handler registering LangChain/LangGraph runs |

Because `eqty_lineage` is an implicit namespace package, no package may ship an
`eqty_lineage/__init__.py`.

## What this captures

A coding agent's session becomes a signed W3C PROV graph: which files it read, what it wrote, which
tool call produced which version, what the model was asked and what it returned. Every file version is
content-addressed, so identical artifacts collapse to one node across sessions, machines and capture
paths with no correlation ids and no shared database.

```bash
# Offline: ingest a finished Claude Code session
eqty-lineage-transcript ~/.claude/projects/<project>/<session>.jsonl -o manifest.json

# Live: run the hook daemon and point an agent at it
export EQTY_LINEAGE_TOKEN=$(openssl rand -hex 16)
eqty-lineage-hooks serve --port 8787 --manifests ./manifests --watch /path/to/repo
eqty-lineage-hooks install --print              # settings.json wiring for Claude Code
eqty-lineage-hooks install --dialect codex      # ~/.codex config for Codex
```

File *contents* are not stored unless you ask for them (`--blobs`). Identity, paths and derivation edges
are recorded either way; the flag only decides whether the bytes of everything the agent read become
durable on disk. The token is read from `$EQTY_LINEAGE_TOKEN` rather than a flag, because a secret on the
command line is visible in `ps`, and the emitted config passes it by env-var reference so a
`settings.json` carrying it can still be committed.

Then query it:

```python
from eqty_lineage.core import TripleSink
from eqty_lineage.query import blast_radius, reaches, taint

triples = TripleSink.load("session.jsonl")
blast_radius(triples, cid)          # everything downstream of one artifact
reaches(triples, src, dst)          # the minimal sets of edges that connect them
taint(triples, untrusted=[cid])     # did anything untrusted reach each artifact
```

## Two capture paths, one recorder

The offline and live paths share `eqty-lineage-core`, including tool-result parsing, so they cannot
drift on the details that matter. `eqty-lineage-hooks verify` ingests a real transcript, separately
replays it as hook payloads, and asserts every divergence falls into a declared bucket. 12/12 sessions
conform at the time of writing.

**What that does and does not prove.** Because both paths share `core`, this exercises the two
*adapters*, not the recorder: a fault in shared code is invisible to it, which is the classic N-version
result. Replay also cannot produce events a transcript has no record of — `InstructionsLoaded`,
`FileChanged`, `PermissionRequest` — so those asymmetries are *declared*, not demonstrated. Conformance
means no undeclared divergence, not that both paths are correct.

The live path gets three things a transcript cannot provide: Bash side effects as *observed* rather than
inferred, the instruction files that entered context, and the permission decisions themselves. See
[`packages/eqty-lineage-agent-hooks/README.md`](packages/eqty-lineage-agent-hooks/README.md).

**`SessionStart` is wired as a command hook, not HTTP.** It is the one event that accepts no HTTP
handler. Configured as HTTP it is accepted by `settings.json` and then silently never delivered — and
since `SessionStart` both creates the `Agent` asset and returns `watchPaths`, losing it means the
manifest attests a transcript rather than a computation *and* no `FileChanged` ever fires, so every Bash
side effect stays unobserved. `install` emits the correct mix; this was found by running a session and
watching the event never arrive.

## Codex

Verified end-to-end against **codex-cli 0.145.0** — a live session through the daemon: hooks
authenticated, `apply_patch` parsed, `slug.py` recorded as `Code` with its path, manifest exported, zero
errors. Five things differ from Claude Code, all found by capturing real payloads rather than reading
documentation. Most importantly Codex writes with `apply_patch`, whose payload carries a patch document
rather than a file path and content, so a Codex session yields *no file lineage at all* unless the patch
is parsed.

One is a gap rather than a difference: **a failing Codex shell command is indistinguishable from a
successful one.** Its `tool_response` is bare output text with no exit code and no error flag —
`ls /nonexistent` and `echo hello` differ only in what they printed. Codex failures are therefore
recorded as successes, and no care in the adapter changes that. Pinned by a test so a future codex-cli
that starts reporting it shows up as a failure.

The offline path does not cover Codex; its rollout files are a different format.

## Develop

```bash
just sync        # install the workspace + dev deps into .venv
just test        # run the test suite
just test-pure   # only the tests needing neither eqty-sdk nor a Rust toolchain
just build       # build wheel + sdist for all packages into ./dist
just build-accel # build the optional Rust accelerator (needs cargo + maturin)
just publish     # upload ./dist to pypi.eqtylab.io
```

`eqty-sdk` resolves from the eqty index configured in `pyproject.toml`, not public PyPI.

### Tests

`tests/` covers the recorder, both adapters, the semirings and the query engine — 286 tests, of which
the backend-agreement ones skip when the accelerator is not built.

`just test-pure` is the subset needing neither `eqty-sdk` nor a Rust toolchain: 234 of them, and they
genuinely run with no SDK installed at all. `just test-nosdk` proves it, by building a throwaway venv,
asserting `eqty_sdk` is absent, and running them there — which is what CI does, since `eqty-sdk` comes
from a private index CI has no credentials for. That claim used to be false: importing anything from
`eqty_lineage.core` pulled in the recorder, and the recorder imports the SDK at module scope. The
recorder is now loaded on first use.

Fixtures are synthetic or sanitised. Real Claude Code transcripts carry the full contents of whatever
repository the session touched, so `tests/fixtures/claude_session.jsonl` is hand-built to exercise the
parser's edge cases, and `tests/fixtures/codex_hooks.json` is a real capture with paths and home
directory rewritten.

Two facts worth knowing before writing a test here:

- **`eqty_sdk.init()` is process-global.** Tests share one session-scoped context rather than
  initialising per test.
- **Statement CIDs are not stable.** A statement CID covers a signed credential carrying `validFrom`,
  so recording the same computation twice yields different activity CIDs — two manifests of the same
  work are never byte-identical, and byte-comparison regression gates cannot work. Graphs are compared
  through `(inputs, outputs)` signatures in `eqty_lineage.core.canonical`.
