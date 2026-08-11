# EQTY Lineage Codex plugin

Captures the raw JSON payload of every subscribed Codex lifecycle hook, and refuses one exact command
so a session has something to attest about policy. Keeping the payload intact lets the lineage adapter
be tested against what Codex actually emits, before any field is normalized or discarded.

Subscribed in `hooks/hooks.json`: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`,
`SessionEnd`. Codex accepts more (permission requests, compaction, subagent start/stop, turn stop);
they are not subscribed here because nothing downstream models them yet.

## Output

One JSON line per event, appended to `$EQTY_LINEAGE_CAPTURE`, or `codex-hooks.jsonl` in the working
directory when that variable is unset. Writes take an exclusive `flock` — Codex runs tool calls
concurrently, so hooks race for the file.

```json
{"collector": {"schema": "eqty.codex-hook-capture.v1", "received_unix_ns": 1786...,
               "decision": "deny", "decision_reason": "command matches EQTY_LINEAGE_DENY_COMMAND"},
 "payload":   {"hook_event_name": "PreToolUse", "tool_name": "Bash", ...}}
```

The agent's payload is verbatim under `payload`; everything this script concluded is under `collector`.
A reader can always separate what Codex said from what we decided. That split is not cosmetic: a
denial happens *here* and leaves no trace in any later payload, so without recording it, a denied call
is indistinguishable from a capture that was truncated mid-flight.

`eqty-codex-lineage <capture>` turns the file into a signed graph;
`eqty_lineage.codex.replay_capture()` is the library form.

## Configuration

| Variable | Effect |
| --- | --- |
| `EQTY_LINEAGE_CAPTURE` | where to append; defaults to `./codex-hooks.jsonl` |
| `EQTY_LINEAGE_DENY_COMMAND` | one exact command string to refuse at `PreToolUse` |

The match is exact string equality against `tool_input.command`, which makes this a demonstration of
the gate rather than a policy engine. A command that differs by a byte is allowed.

## Two things this cost us, so they are written down

**A malformed decision fails open, silently.** `PreToolUseHookSpecificOutputWire` requires
`hookEventName` and sets `additionalProperties: false`. Emit a denial without it and codex-cli 0.147.0
discards the decision, runs the command, and logs nothing — stderr still reads `PreToolUse Completed`.
A denied `touch forbidden.txt` created its file while the capture recorded a denial and the graph
rendered one. Validate denials against the filesystem, never against the log; that is what
`scripts/validate-codex.sh` does, and reintroducing the missing field makes it exit 4.

**The hook payload is not a complete execution log.** In a live 0.147.0 pilot, `codex exec --json`
exposed a shell failure's `exit_code: 1` while the matching `PostToolUse` hook carried only a plain
traceback string. The adapter must preserve that outcome as unknown, or join it with the
JSONL/app-server event stream. It must not infer success from the absence of a flag.

## Installing it

Install as a plugin, which is the only route that applies to sessions in *other* repos. The
marketplace manifest at the repo root makes this checkout installable directly:

```bash
codex plugin marketplace add /path/to/eqty-lineage
codex plugin add eqty-lineage@eqty-lineage
codex plugin list | grep eqty          # installed, enabled
```

Then trust the hooks once, interactively:

```
codex
/hooks          # review the eqty-lineage hooks, trust them
```

After that, any session captures with no per-invocation flags:

```bash
export EQTY_LINEAGE_CAPTURE=/tmp/codex-hooks.jsonl
codex exec "your task"
```

For headless runs on a machine nobody sits at — CI, a container — pass
`--dangerously-bypass-hook-trust` instead of trusting interactively. That is what
`scripts/validate-codex.sh` does.

## Untrusted hooks fail silently

This is the failure you will actually hit, so it goes above the fold: an untrusted hook is **skipped,
not reported**. Verified against codex-cli 0.147.0 — a session with untrusted hooks runs the task,
exits 0, prints nothing about hooks, and writes no capture at all. "I installed it and got no lineage"
almost always means untrusted, not broken.

Trust is recorded against the hook definition's *hash*, which has two consequences:

- **Installing copies the plugin into `~/.codex/plugins/cache/`**, so editing your checkout changes
  nothing until `codex plugin remove` then `add` again — and that re-flags the hooks, so you must
  re-trust through `/hooks`.
- Any edit to `hooks.json` or the command it runs drops trust the same way.

A wrong or invented `trusted_hash` in config is rejected just as silently, so it is not a way to
pre-trust a hook from an installer. There is no supported programmatic route today
(<https://github.com/openai/codex/issues/21615>).

Codex also loads hooks from `~/.codex/hooks.json`, `~/.codex/config.toml`, `<repo>/.codex/hooks.json`
and `<repo>/.codex/config.toml`, and layers `$CODEX_HOME/<name>.config.toml` when passed
`--profile <name>`. Those are fine for a single repo or a one-off; the plugin is what makes the
collector follow you across repos.

If you script `codex exec`, redirect `</dev/null` — with stdin left open it reads it as additional
input and blocks forever on EOF.
