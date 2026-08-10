#!/usr/bin/env python3
"""Persist the exact Codex hook payload and optionally deny one demo command."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    payload = json.load(sys.stdin)
    path = Path(os.environ.get("EQTY_LINEAGE_CAPTURE", "codex-hooks.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"payload": payload, "schema": "eqty.codex-hook-capture.v1", "received_unix_ns": time.time_ns()}) + "\n")
    command = payload.get("tool_input", {}).get("command")
    if payload.get("hook_event_name") == "PreToolUse" and command == os.environ.get("EQTY_LINEAGE_DENY_COMMAND"):
        print(json.dumps({"hookSpecificOutput": {"permissionDecision": "deny"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
