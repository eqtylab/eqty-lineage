#!/usr/bin/env python3
"""Persist the exact Codex hook payload, and record any decision this collector made about it.

The payload is stored verbatim under ``payload``; everything the collector adds goes under
``collector``, so a reader can always tell what Codex said from what this script concluded. That split
is what lets the replay attach a signed ``deny`` to a call: a denial happens *here*, leaves no trace in
any later payload, and would otherwise be indistinguishable from a truncated capture.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SCHEMA = "eqty.codex-hook-capture.v1"


def _deny_command() -> str | None:
    """The exact command this collector is configured to refuse, if any.

    Returned as ``None`` when unset, and compared only after that check. Reading the variable straight
    into an equality test makes an unconfigured collector deny every tool call whose ``tool_input``
    carries no ``command`` -- both sides are then ``None``.
    """
    command = os.environ.get("EQTY_LINEAGE_DENY_COMMAND")
    return command or None


def main() -> int:
    payload = json.load(sys.stdin)

    destination = os.environ.get("EQTY_LINEAGE_CAPTURE")
    path = Path(destination).expanduser() if destination else Path("codex-hooks.jsonl")

    collector: dict[str, object] = {"schema": SCHEMA, "received_unix_ns": time.time_ns()}

    deny = _deny_command()
    tool_input = payload.get("tool_input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    denied = deny is not None and payload.get("hook_event_name") == "PreToolUse" and command == deny
    if denied:
        collector["decision"] = "deny"
        collector["decision_reason"] = "command matches EQTY_LINEAGE_DENY_COMMAND"

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"collector": collector, "payload": payload}, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        _locked_write(stream, line)

    if denied:
        # codex-cli requires `hookEventName` here and rejects unknown keys
        # (PreToolUseHookSpecificOutputWire: required ["hookEventName"], additionalProperties false).
        # Omitting it does not raise -- the decision is silently dropped and the command runs. Verified
        # against codex-cli 0.147.0, where the earlier output shape let a denied `touch` create its file.
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": collector["decision_reason"],
                    }
                }
            )
        )
    return 0


def _locked_write(stream, line: str) -> None:
    """Append one record. Codex runs tool calls concurrently, so hooks race for this file."""
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


if __name__ == "__main__":
    raise SystemExit(main())
