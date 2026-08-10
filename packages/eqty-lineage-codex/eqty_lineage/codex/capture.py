"""Turning a raw Codex hook capture into the session the graph builder replays.

The collector (``plugins/eqty-lineage/scripts/capture_hook.py``) writes one JSON record per hook event
and interprets nothing, so this is the layer that has to decide what the payload stream actually
attests. Three record shapes are accepted, because captures outlive collector versions:

===========================================  =========================================
``{"collector": {...}, "payload": {...}}``   current collector
``{"payload": {...}, "schema": ...}``        earlier flat collector
``{"hook_event_name": ...}``                 a bare payload, as the test fixtures store them
===========================================  =========================================

**Absence is not denial.** A ``PreToolUse`` with no matching ``PostToolUse`` means the call was not
observed to run -- the hook denied it, the session crashed, or the capture was cut mid-flight. Only the
first of those is a policy decision, so an unmatched attempt is recorded ``unknown`` and a ``deny`` is
recorded only where the collector wrote down that it denied. Guessing would put a signed ``deny`` claim
on a truncated file.

The converse direction *is* sound and is used: a call that produced a ``PostToolUse`` demonstrably ran,
so it was permitted, and it is recorded ``allow`` without needing the collector to say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CAPTURE_SCHEMA = "eqty.codex-hook-capture.v1"

ALLOW = "allow"
DENY = "deny"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class CaptureRecord:
    """One captured hook event: the agent's payload, plus whatever the collector added around it."""

    payload: dict[str, Any]
    collector: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolAttempt:
    """One ``PreToolUse``, and whatever became of it."""

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    decision: str = UNKNOWN
    reason: str | None = None
    result: Any = None
    executed: bool = False


@dataclass
class CodexSession:
    """The subset of a Codex session this slice models."""

    session_id: str = ""
    model: str | None = None
    agent_version: str | None = None
    prompts: list[str] = field(default_factory=list)
    attempts: list[ToolAttempt] = field(default_factory=list)


def _unwrap(record: Any) -> CaptureRecord:
    if not isinstance(record, dict):
        raise TypeError(f"capture record is {type(record).__name__}, expected an object")
    payload = record.get("payload")
    if isinstance(payload, dict):
        collector = record.get("collector")
        if not isinstance(collector, dict):
            # The flat collector kept its own fields as siblings of the payload rather than under a
            # `collector` key. Read them back into the same shape so callers see one schema.
            collector = {key: record[key] for key in ("schema", "received_unix_ns", "decision") if key in record}
        return CaptureRecord(payload=payload, collector=collector)
    return CaptureRecord(payload=record)


def load_capture(source: str | Path) -> list[CaptureRecord]:
    """Read a capture written as JSONL, or a fixture written as a JSON array."""
    text = Path(source).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return [_unwrap(record) for record in json.loads(text)]
    return [_unwrap(json.loads(line)) for line in text.splitlines() if line.strip()]


def _open_attempt(session: CodexSession, tool_use_id: str, tool_name: str) -> ToolAttempt | None:
    """Find the attempt a ``PostToolUse`` closes.

    ``tool_use_id`` is the join key and is present on every real codex-cli tool payload. The fallback
    matters only for hand-written fixtures: with no id, close the most recent still-open attempt on the
    same tool, which is correct for a sequential session and the reason this stays a fallback rather
    than the primary path -- Codex can run tool calls concurrently.
    """
    if tool_use_id:
        for attempt in reversed(session.attempts):
            if attempt.tool_use_id == tool_use_id:
                return attempt
        return None
    for attempt in reversed(session.attempts):
        if not attempt.executed and attempt.tool_name == tool_name:
            return attempt
    return None


def normalize(records: list[CaptureRecord]) -> CodexSession:
    """Fold a capture into one session. Unrecognized hook events are skipped, never fatal."""
    session = CodexSession()

    for record in records:
        payload = record.payload
        event = payload.get("hook_event_name") or ""

        if event == "SessionStart":
            session.session_id = payload.get("session_id") or session.session_id
            session.model = payload.get("model") or session.model
            session.agent_version = payload.get("version") or payload.get("agent_version") or session.agent_version

        elif event == "UserPromptSubmit":
            text = payload.get("prompt")
            if isinstance(text, str) and text:
                session.prompts.append(text)

        elif event == "PreToolUse":
            tool_input = payload.get("tool_input")
            attempt = ToolAttempt(
                tool_use_id=payload.get("tool_use_id") or "",
                tool_name=payload.get("tool_name") or "tool",
                tool_input=tool_input if isinstance(tool_input, dict) else {},
            )
            decision = record.collector.get("decision")
            if decision in (ALLOW, DENY):
                attempt.decision = decision
                attempt.reason = record.collector.get("decision_reason") or "collector policy decision"
            session.attempts.append(attempt)

        elif event == "PostToolUse":
            attempt = _open_attempt(session, payload.get("tool_use_id") or "", payload.get("tool_name") or "tool")
            if attempt is None:
                continue
            attempt.executed = True
            attempt.result = payload.get("tool_response")
            if attempt.decision == UNKNOWN:
                attempt.decision = ALLOW
                attempt.reason = "observed to execute"

        # SessionEnd, Stop, PermissionRequest and anything a later codex-cli adds carry nothing this
        # slice models. Skipping them keeps an unknown event from taking down the replay.

    if not session.session_id:
        session.session_id = "codex-capture"
    return session


def load_session(source: str | Path) -> CodexSession:
    """Read a capture file and fold it into a session."""
    return normalize(load_capture(source))


__all__ = [
    "ALLOW",
    "CAPTURE_SCHEMA",
    "DENY",
    "UNKNOWN",
    "CaptureRecord",
    "CodexSession",
    "ToolAttempt",
    "load_capture",
    "load_session",
    "normalize",
]
