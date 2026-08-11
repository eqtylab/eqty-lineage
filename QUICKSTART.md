# QuickStart — Codex signed lineage

Turn a Codex session into a signed lineage graph you can open in the Explorer, in three steps:
capture the raw hook events, replay them into a manifest, upload it.

```
codex  ──hooks──▶  capture_hook.py  ──▶  codex-hooks.jsonl  ──▶  replay_capture()  ──▶  manifest.json
        (raw payloads)                   (raw + decisions)        (normalize + sign)     (Explorer)
```

## 1. Install

```bash
just sync     # uv sync --group dev, into ./.venv
```

`eqty-sdk` resolves from `https://pypi.eqtylab.io/simple/`, not public PyPI, so this needs
`UV_INDEX_EQTY_USERNAME` / `UV_INDEX_EQTY_PASSWORD` in your environment.

> The committed `.env` holds a **stale** pair, and `.envrc` loads it over whatever you exported, so
> under direnv a correct credential is silently replaced by one the index answers `403` to. Until that
> file is fixed, export the working pair in a shell without direnv, or update `.env` itself. A `403`
> from `uv sync` or `uv add` is this, not your credential.

## 2. See the graph without installing Codex

```bash
uv run python examples/codex_lineage_demo.py /tmp/codex-demo.json
```

Exports a two-call session: one allowed `Bash` call and one denied one. Upload
`/tmp/codex-demo.json` to the Lineage Explorer — the denied attempt has a guardrail node and **no**
result node, because nothing ran.

The demo's events are written out in `build_demo()` rather than read from a capture so it runs with no
agent installed, but it goes through the same `CodexLineage.replay()` path a real capture does. The
graph shape is the replay's, not a second hand-built one.

## 3. Capture a real session

Install the collector as a Codex plugin. This is the route that applies to sessions in *any* repo:

```bash
codex plugin marketplace add "$PWD"
codex plugin add eqty-lineage@eqty-lineage

export EQTY_LINEAGE_CAPTURE=/tmp/codex-hooks.jsonl
export EQTY_LINEAGE_DENY_COMMAND='touch forbidden.txt'   # optional: refuse one exact command

codex exec --dangerously-bypass-hook-trust "your task here"
```

Installing copies the plugin into `~/.codex/plugins/cache/`, so `codex plugin remove` then `add` again
after any edit to it.

<details>
<summary>Without installing: inline hooks per invocation</summary>

This is the form `scripts/validate-codex.sh` uses, since it must exercise the working tree rather than
an installed copy:

```bash
export EQTY_LINEAGE_CAPTURE=/tmp/codex-hooks.jsonl
export EQTY_LINEAGE_DENY_COMMAND='touch forbidden.txt'   # optional: refuse one exact command

codex exec --dangerously-bypass-hook-trust \
  -c 'hooks.SessionStart=[{hooks=[{type="command",command="python3 plugins/eqty-lineage/scripts/capture_hook.py"}]}]' \
  -c 'hooks.UserPromptSubmit=[{hooks=[{type="command",command="python3 plugins/eqty-lineage/scripts/capture_hook.py"}]}]' \
  -c 'hooks.PreToolUse=[{hooks=[{type="command",command="python3 plugins/eqty-lineage/scripts/capture_hook.py"}]}]' \
  -c 'hooks.PostToolUse=[{hooks=[{type="command",command="python3 plugins/eqty-lineage/scripts/capture_hook.py"}]}]' \
  -c 'hooks.SessionEnd=[{hooks=[{type="command",command="python3 plugins/eqty-lineage/scripts/capture_hook.py"}]}]' \
  "your task here"
```

</details>

Each hook appends one JSON line. The agent's payload is stored verbatim under `payload`; anything the
collector concluded goes under `collector`, so a reader can always separate what Codex said from what
this repo decided:

```json
{"collector":{"schema":"eqty.codex-hook-capture.v1","received_unix_ns":1786...,"decision":"deny",
              "decision_reason":"command matches EQTY_LINEAGE_DENY_COMMAND"},
 "payload":{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"touch forbidden.txt"}}}
```

## 4. Replay the capture into a signed manifest

```bash
uv run python -c "
from eqty_lineage.codex import replay_capture
print(replay_capture('/tmp/codex-hooks.jsonl', '/tmp/codex-session.json'))
"
```

Upload `/tmp/codex-session.json` to the Explorer.

## 5. Validate the whole pipeline

```bash
just validate-codex          # or: ./scripts/validate-codex.sh [--keep]
```

Drives a real Codex session in a disposable repo, asserts the denial was enforced *on the filesystem*,
exports the graph and checks it agrees. Exit codes so it can gate:

| Code | Meaning |
| --- | --- |
| 0 | the pipeline behaved |
| 2 | preconditions missing (no `codex`, no `uv`) — did not run, distinct from a failure |
| 3 | the session produced no usable capture |
| 4 | **the denial was not enforced** — the forbidden file exists |
| 5 | the exported graph does not match the session |
| 6 | the session did not finish within `VALIDATE_TIMEOUT` (default 300s) |

`--keep` retains the work directory for inspection.

<details>
<summary>The same thing as a copy-pasteable block</summary>

Point `REPO` at this checkout.

```bash
REPO=$PWD
HOOK="python3 $REPO/plugins/eqty-lineage/scripts/capture_hook.py"
WORK=$(mktemp -d)
export EQTY_LINEAGE_CAPTURE="$WORK/codex-hooks.jsonl"
export EQTY_LINEAGE_DENY_COMMAND='touch forbidden.txt'

mkdir -p "$WORK/ws" && cd "$WORK/ws" && git init -q . && git commit -q --allow-empty -m baseline
cat > TASK.md <<'EOF'
Run exactly these two shell commands, in this order, and nothing else:
1. printf 'allowed\n'
2. touch forbidden.txt
Do not use apply_patch. Do not retry a command that is blocked. Then stop.
EOF

hook() { printf 'hooks.%s=[{hooks=[{type="command",command="%s"}]}]' "$1" "$HOOK"; }
codex exec --dangerously-bypass-hook-trust --skip-git-repo-check --sandbox workspace-write \
  -c "$(hook SessionStart)" -c "$(hook UserPromptSubmit)" -c "$(hook PreToolUse)" \
  -c "$(hook PostToolUse)" -c "$(hook SessionEnd)" \
  "Read TASK.md and perform it exactly."

# The denial must have been enforced, not merely recorded.
test ! -e "$WORK/ws/forbidden.txt" && echo "OK: denied write never happened"

cd "$REPO"
uv run python -c "
from eqty_lineage.codex import replay_capture
replay_capture('$EQTY_LINEAGE_CAPTURE', '$WORK/session.json')
"
```

Check that the graph records both outcomes and that the denied call has no result:

```bash
uv run python - "$WORK/session.json" <<'PY'
import base64, json, sys
m = json.load(open(sys.argv[1]))
for s in m["statements"].values():
    if s.get("@type") != "MetadataRegistration":
        continue
    d = json.loads(base64.b64decode(m["blobs"][s["metadata"].replace("urn:cid:", "")]))
    if d.get("computation_type") == "tool":
        print(f'  {d["decision"]:8} executed={str(d["executed"]):5} {d["name"]}')
PY
```

A healthy run prints at least one `allow executed=True` and exactly one `deny executed=False`, and
`forbidden.txt` does not exist.

</details>

> **Check the file, not just the graph.** A collector whose deny is malformed is silently ignored by
> codex-cli — the capture still records `deny`, the graph still renders one, and the command runs
> anyway. That is what `test ! -e forbidden.txt` is for.

## What the graph claims

| Recorded | When | Why |
| --- | --- | --- |
| `allow` | the call produced a `PostToolUse` | it demonstrably ran, so it was permitted |
| `deny` | the collector wrote down that it denied | a denial happens in the hook and leaves no trace in any later payload |
| `unknown` | neither | the capture may have been cut mid-call |

**Absence is not denial.** A `PreToolUse` with no matching `PostToolUse` means the call was not
observed to run — the hook denied it, the session crashed, or the file was truncated. Only the first is
a policy decision, so an unmatched attempt is `unknown`. Signing a `deny` there would put a claim on a
truncated file.

A call recorded `deny` that *also* produced a `PostToolUse` keeps both facts: the graph shows a denied
call that ran, which is the fact an audit wants, not one to smooth over.

## Tests

```bash
just test
```

18 tests. The ones that matter run the real collector as a subprocess over real captured codex-cli
0.145.0 payloads (`tests/fixtures/codex_hooks.json`), then replay the file it produces — the two halves
are exercised as one pipeline, not separately.

## Known gaps

- **Codex shell failures are not detectable from hooks alone.** A failing `PostToolUse` can carry a
  bare string with no exit code. This slice records the result as-is and never infers success.
- **A malformed deny fails open and says nothing.** `PreToolUseHookSpecificOutputWire` requires
  `hookEventName` and sets `additionalProperties: false`; omit it and codex-cli 0.147.0 discards the
  decision, runs the command, and logs no error. Validate denials against the filesystem, not the log.
- The `unknown` state is reachable in practice; the Explorer has no distinct rendering for it yet.
- Concurrent tool calls join on `tool_use_id`. Captures without it fall back to most-recent-open-call
  on the same tool, which is only correct for a sequential session.
- CI runs `fmt`, `lint` and `build`, but not `just test`; the suite gates nothing until that step is added.
