"""Replaying a Claude Code transcript as the hook payloads the agent would have emitted.

This exists for the equivalence test. Comparing the offline and live capture paths needs both to consume
the *same* session, and a genuinely live capture would require driving a real agent — expensive,
nondeterministic, and impossible to pin as a fixture.

The mapping is deliberately small and auditable: an assistant record's ``tool_use`` block becomes a
``PreToolUse`` payload, the matching ``toolUseResult`` becomes ``PostToolUse`` (or
``PostToolUseFailure``). Nothing is invented that the agent would not have sent.

**What replay cannot show.** It reproduces the hook payloads derivable *from a transcript*, so it can
never exercise the events a transcript has no record of — ``InstructionsLoaded``, ``FileChanged``,
``PermissionRequest``. Those are exactly the live path's advantages, so the equivalence test measures
agreement on the shared substrate and takes the asymmetries as declared, not as demonstrated.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def replay_payloads(transcript: Path) -> Iterator[dict[str, Any]]:
    """Yield hook payloads equivalent to what the agent would have posted for this session."""
    records: list[dict[str, Any]] = []
    with Path(transcript).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue

    if not records:
        return

    first = next((r for r in records if r.get("type") in ("assistant", "user", "system")), {})
    session_id = first.get("sessionId") or Path(transcript).stem
    cwd = first.get("cwd")
    version = first.get("version")
    model = next(
        (
            r["message"]["model"]
            for r in records
            if r.get("type") == "assistant" and (r.get("message") or {}).get("model")
        ),
        None,
    )
    permission_mode = next((r["permissionMode"] for r in records if r.get("type") == "permission-mode"), None)

    def base(name: str, **kw: Any) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "hook_event_name": name,
            "cwd": cwd,
            "permission_mode": permission_mode,
            "version": version,
            "prompt_id": None,
            **kw,
        }

    yield base("SessionStart", model=model, source=first.get("entrypoint"), timestamp=first.get("timestamp"))

    pending: dict[str, str] = {}

    for record in records:
        rtype = record.get("type")
        at = record.get("timestamp")
        message = record.get("message") or {}
        content = message.get("content")

        if rtype == "system" and record.get("subtype") == "compact_boundary":
            meta = record.get("compactMetadata") or {}
            yield base("PostCompact", trigger=meta.get("trigger"), timestamp=at)
            continue

        if rtype == "assistant" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_use_id = block.get("id")
                if not tool_use_id:
                    continue
                pending[tool_use_id] = block.get("name") or "tool"
                yield base(
                    "PreToolUse",
                    tool_use_id=tool_use_id,
                    tool_name=pending[tool_use_id],
                    tool_input=block.get("input"),
                    timestamp=at,
                )

        elif rtype == "user":
            if "toolUseResult" not in record:
                text = _text_of(content)
                if text:
                    yield base("UserPromptSubmit", prompt=text, timestamp=at)
                continue

            tool_use_id = _tool_result_id(content)
            if tool_use_id is None:
                continue
            name = pending.pop(tool_use_id, "unknown")
            result = record["toolUseResult"]
            is_error = _is_error(content, result)
            # A failure carries its payload under `error`; only success uses `tool_response`. Emitting
            # the success key for both would make this harness disagree with the CLI it stands in for,
            # and the equivalence test would then be comparing the offline path against a fiction.
            outcome = {"error": result} if is_error else {"tool_response": result}
            yield base(
                "PostToolUseFailure" if is_error else "PostToolUse",
                tool_use_id=tool_use_id,
                tool_name=name,
                timestamp=at,
                **outcome,
            )

    yield base("SessionEnd", reason="other", timestamp=records[-1].get("timestamp"))


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _tool_result_id(content: Any) -> str | None:
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return block.get("tool_use_id")
    return None


def _is_error(content: Any, result: Any) -> bool:
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                return True
    if isinstance(result, dict):
        return bool(result.get("is_error") or result.get("interrupted"))
    return False


__all__ = ["replay_payloads"]
