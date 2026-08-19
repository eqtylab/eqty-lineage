#!/usr/bin/env python3
"""Relay one Codex hook payload to the eqty-lineage daemon, and keep the raw capture.

This replaces a collector that wrote payloads to a file for later replay. Replay produced a manifest
of prompts and tool calls and no file lineage at all, because Codex writes through ``apply_patch``
whose ``tool_input.command`` is a patch document -- the daemon's dialect parses that, so the same
session now records the files it wrote, their content CIDs and their derivation edges.

Three jobs, in the order they must not be got wrong:

1. **Relay the daemon's answer.** The daemon already returns Codex's own ``hookSpecificOutput`` shape,
   so its permission decisions and ``watchPaths`` are passed through verbatim. This is what makes the
   harness's policy govern Codex sessions rather than only Claude Code ones.
2. **Keep the raw capture.** The payload is still appended verbatim, so a session remains replayable
   offline and a daemon that was down is not a hole in the record.
3. **Never break the session.** Every failure is caught and the hook exits 0. A lineage collector that
   takes down the agent it observes is worse than no collector.

Nothing here is specific to a transport the daemon does not already speak; ``install --dialect codex``
emits an equivalent curl form. The plugin exists because Codex trusts plugins through ``/hooks``,
which is the reviewable way to install a hook that sees every event.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "eqty.codex-hook-capture.v2"
DEFAULT_URL = "http://127.0.0.1:8787/hook"
#: Shorter than the hook timeout declared in hooks.json, so a slow daemon is a recorded miss rather
#: than a stalled agent.
TIMEOUT_SECONDS = 5.0


def _deny_command() -> str | None:
    """The exact command this collector refuses on its own, if any.

    Only consulted when the daemon is unreachable -- the daemon's policy is the real one. Returned as
    ``None`` when unset and compared only after that check: reading the variable straight into an
    equality test makes an unconfigured collector deny every tool call whose ``tool_input`` carries no
    ``command``, because both sides are then ``None``.
    """
    command = os.environ.get("EQTY_LINEAGE_DENY_COMMAND")
    return command or None


def _post(payload: dict) -> tuple[dict | None, str | None]:
    """Send the payload to the daemon. Returns (response, error)."""
    url = os.environ.get("EQTY_LINEAGE_URL", DEFAULT_URL)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    token = os.environ.get("EQTY_LINEAGE_TOKEN")
    if token:
        # Read from the environment rather than baked into hooks.json, which is committed.
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8").strip()
        return (json.loads(body) if body else {}), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - a collector must not take down the session
        return None, type(exc).__name__


def _deny_output(reason: str) -> dict:
    """Codex's deny shape.

    ``hookEventName`` is required and unknown keys are rejected
    (``PreToolUseHookSpecificOutputWire``: required ``["hookEventName"]``, ``additionalProperties``
    false). Omitting it does not raise -- the decision is silently dropped and the command runs.
    Verified against codex-cli 0.147.0, where an earlier output shape let a denied ``touch`` create
    its file.
    """
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _locked_append(path: Path, line: str) -> None:
    """Append one record. Codex runs tool calls concurrently, so hooks race for this file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        try:
            import fcntl
        except ImportError:  # non-POSIX; a single append is the best available
            stream.write(line)
            return
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(line)
            stream.flush()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:  # noqa: BLE001 - an unreadable payload is not worth a failed session
        return 0

    collector: dict[str, object] = {"schema": SCHEMA, "received_unix_ns": time.time_ns()}
    response, error = _post(payload)

    if error is None:
        collector["relayed"] = True
    else:
        # Say which, and say it in the record rather than only on stderr. A silent fallback is how a
        # session ends up with a capture file nobody knows is the only copy.
        collector["relayed"] = False
        collector["relay_error"] = error

    # The payload is stored verbatim under `payload`; everything this script concluded goes under
    # `collector`, so a reader can always tell what Codex said from what the collector decided.
    if os.environ.get("EQTY_LINEAGE_CAPTURE") != "off":
        destination = os.environ.get("EQTY_LINEAGE_CAPTURE")
        path = Path(destination).expanduser() if destination else Path("codex-hooks.jsonl")
        try:
            _locked_append(path, json.dumps({"collector": collector, "payload": payload}) + "\n")
        except OSError:
            pass

    if error is None:
        # The daemon speaks Codex's own output shape, so its decision passes straight through.
        if response:
            print(json.dumps(response))
        return 0

    # Daemon unreachable: fall back to the collector's own rule so a configured denial still holds.
    deny = _deny_command()
    tool_input = payload.get("tool_input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if deny is not None and payload.get("hook_event_name") == "PreToolUse" and command == deny:
        print(json.dumps(_deny_output("command matches EQTY_LINEAGE_DENY_COMMAND")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
