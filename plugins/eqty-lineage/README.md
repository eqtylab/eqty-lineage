# eqty-lineage: signed lineage for Codex sessions

A Codex plugin that relays every hook event to the `eqty-lineage` daemon, which records it through the
same core recorder as Claude Code. One graph shape for both agents, and one conformance harness
checking them against each other.

## Why a plugin rather than a copied config

Codex requires hooks to be trusted before they run. A plugin is trusted once through `/hooks`, which
is the reviewable way to install a hook that sees every event — the alternative is pasting a command
into `~/.codex/config.toml` and passing `--dangerously-bypass-hook-trust` forever.

## Install

```bash
# 1. start the daemon (file contents are stored by default -- see below)
eqty-lineage-hooks --state-dir ~/.eqty/state serve --port 8787 --manifests ~/.eqty/manifests

# 2. point the plugin at it, and trust the plugin once via /hooks in Codex
export EQTY_LINEAGE_URL=http://127.0.0.1:8787/hook
export EQTY_LINEAGE_TOKEN=...      # optional; forwarded as a bearer token
```

| variable | effect |
| --- | --- |
| `EQTY_LINEAGE_URL` | daemon endpoint (default `http://127.0.0.1:8787/hook`) |
| `EQTY_LINEAGE_TOKEN` | sent as `Authorization: Bearer`; read from the environment, never committed |
| `EQTY_LINEAGE_CAPTURE` | path for the raw capture, or `off` (default `./codex-hooks.jsonl`) |
| `EQTY_LINEAGE_DENY_COMMAND` | a command to refuse **only when the daemon is unreachable** |

## What it subscribes to

All **eleven** events codex-cli emits, read out of the 0.148.0 binary's own contiguous enum rather
than from documentation:

```
PreToolUse PermissionRequest PostToolUse PreCompact PostCompact
SessionStart SessionEnd UserPromptSubmit SubagentStart SubagentStop Stop
```

`Stop` matters more than it looks: it carries `last_assistant_message`, the only hook payload on
either dialect containing model output, and therefore the only way the live path records a model call
at all. `PreCompact` is subscribed and deliberately silent — the boundary is recorded on `PostCompact`,
and emitting on both would double-count compactions and inflate the coverage counters. It is kept
because a `PreCompact` with no matching `PostCompact` is a compaction that began and never finished.

## What it records

Routed through the daemon, a Codex session yields the same lineage a Claude Code session does. On a
real captured session (8 events, `apply_patch` + `Bash`):

```
68 statements, 28 blobs
/repo/slug.py   type=Code   content-basis=stated   version=1
```

The file lineage is the part that needs the daemon. Codex writes through `apply_patch`, whose
`tool_input.command` is a patch document and whose `tool_response` is a plain string — neither carries
`filePath`, `content` or `structuredPatch`, so a collector that does not parse the patch records the
tool call and nothing about the file it wrote.

## Permission decisions come from the daemon

The daemon already answers in Codex's own `hookSpecificOutput` shape, so its verdict is passed through
verbatim and the harness's policy governs Codex sessions. A `--deny-write '*.py'` policy refuses a
Codex `apply_patch`, naming the path parsed out of the patch:

```
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
 "permissionDecisionReason": "eqty-lineage: write to '/repo/slug.py' matches denied pattern '*.py'"}}
```

`hookEventName` is required and unknown keys are rejected. Omitting it does not raise — the decision is
silently dropped and the command runs. Verified against codex-cli 0.147.0, where an earlier output
shape let a denied `touch` create its file.

## PermissionRequest is observed, not enforced

Codex emits `PermissionRequest` when a tool needs an escalation, and it is subscribed and recorded --
including the agent's own account of what it is asking for, which lives in `tool_input.description`
rather than in a `reason` field.

**It is not a gate.** Measured against codex-cli 0.148.0 with a session built to force an escalation
(`codex exec --approve-for-me`, writing outside the workspace), three plausible deny shapes were each
accepted and then silently ignored, and the write went through every time:

| response | honoured |
| --- | --- |
| `hookSpecificOutput: {permissionDecision: "deny", ...}` | no |
| `{"decision": "block", "reason": ...}` | no |
| `hookSpecificOutput: {behavior: "deny"}` | no |
| **`PreToolUse` with `permissionDecision: "deny"`** (control, same session) | **yes** |

The control is what makes that a finding rather than three failed guesses: the transport and the shape
work, on the same run, for `PreToolUse`. So the enforcement point is `PreToolUse`, and a policy that
also answered `PermissionRequest` would be writing a decision nothing reads.

Two further limits worth knowing. `PermissionRequest` carries no `tool_use_id`, so an escalation is
attributable to the turn but not to the call it authorizes. And it never fires under plain
`codex exec`, which disables escalation outright -- it took a session deliberately constructed to
need one before the event appeared at all.

## The manifest carries the data, not just its identity

On by default. The manifest holds the bytes rather than only their content addresses — measured on the
session above, **28 blobs against 16 with `--no-blobs`**: the patch document, `slug.py`'s actual
content, the user's prompt, the tool outputs and the model's final message. The session is
reconstructable from the manifest alone.

What bounds the risk is the redaction policy rather than the flag. Every file the agent reads is a
candidate for the store, which is how a `.env` or a signing key becomes durable on disk; the policy
denies those by pattern (`.env`, `*.pem`, `*.key`, `id_rsa*`, `*credentials*`, `*secrets*`, …) and
scrubs the rest. Verified with storage on: a `.env` the agent read has its bytes withheld from every
blob, and the omission is *counted* in `coverage.content_redacted` rather than passing silently.

One caveat measured rather than assumed: a denied file keeps its identity in the **triple sidecar**
but does **not** appear in the exported manifest at all, so a query asking "did anything touch X"
against the manifest alone will miss it.

Pass `--no-blobs` for a tree whose contents must not be made durable even in redacted form.

## It never breaks the session

Every failure path exits 0. An unreachable daemon, an unparseable payload, an unwritable capture file —
each is recorded and the session continues. When the relay fails the raw payload is still appended,
with `collector.relayed = false` and the error named, so a capture file that is the only surviving copy
says so rather than looking like a complete record.

## Replaying a capture

The capture keeps the verbatim payloads, so a session recorded while the daemon was down can be fed
back through the same relay once it is up:

```bash
jq -c '.payload' codex-hooks.jsonl |
  while read -r p; do printf '%s' "$p" | python3 plugins/eqty-lineage/scripts/eqty_hook.py; done
```

Order matters — the daemon correlates `PreToolUse` with `PostToolUse` by `tool_use_id` — and the file
is written in order, so a plain sequential read is correct.

Note that `eqty-lineage-hooks replay` is a *different* thing: it takes a triples sidecar and
reinterprets an already-recorded trace under alternative policies. It does not read hook captures.
