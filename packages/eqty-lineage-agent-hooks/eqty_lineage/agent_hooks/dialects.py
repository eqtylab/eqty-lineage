"""Normalizing Claude Code and Codex hook payloads into core events.

Both agents ship hook systems whose stdin schemas are near-identical -- ``session_id``,
``transcript_path``, ``cwd``, ``hook_event_name``, ``tool_name``, ``tool_use_id``, ``tool_input``,
``tool_response`` -- so one adapter covers both and the dialect layer is thin. What it has to reconcile:

===================  ==========================  =============================
                     Claude Code                 Codex
===================  ==========================  =============================
turn identifier      ``prompt_id``               ``turn_id``
edit tool            ``Edit`` / ``Write``        ``apply_patch``
failure              ``PostToolUseFailure``      ``PostToolUse`` with an error
batch completion     ``PostToolBatch``           --
filesystem watch     ``FileChanged``             --
instructions         ``InstructionsLoaded``      --
subagent transcript  --                          ``agent_transcript_path``
===================  ==========================  =============================

The asymmetry that matters most is ``PostToolUse`` firing **only on success** in Claude Code. Without
also subscribing to ``PostToolUseFailure``, every failed tool call vanishes from the graph -- and a
failed call is often exactly the one an audit cares about.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from eqty_lineage.core import (
    Compacted,
    Event,
    FileObserved,
    InstructionsLoaded,
    PermissionDecision,
    PromptSubmitted,
    SessionEnded,
    SessionStarted,
    SubagentEnded,
    SubagentStarted,
    ToolCallEnded,
    ToolCallStarted,
    file_events_from_result,
)

logger = logging.getLogger("eqty.lineage.hooks")

CLAUDE_CODE = "claude-code"
CODEX = "codex"

# Events worth subscribing to. Everything else Claude Code emits is either UI-facing or describes state
# the recorder does not model; subscribing anyway would cost a round trip per occurrence.
CLAUDE_CODE_EVENTS = (
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PostToolBatch",
    "PermissionRequest",
    "PermissionDenied",
    "SubagentStart",
    "SubagentStop",
    "InstructionsLoaded",
    "FileChanged",
    "CwdChanged",
    "PreCompact",
    "PostCompact",
)

# Verified to fire against codex-cli 0.145.0: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse,
# Stop, SessionEnd. The rest are accepted by its config schema but were not exercised by the sessions
# tested, so they are subscribed optimistically -- an event that never fires costs nothing.
CODEX_EVENTS = (
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "PreCompact",
    "PostCompact",
    "Stop",
)


def detect_dialect(payload: Dict[str, Any]) -> str:
    """Identify the agent from the payload alone.

    ``turn_id`` is Codex-only and ``prompt_id`` is Claude-Code-only, so the pair discriminates on any
    turn-scoped payload. Lifecycle events are not turn-scoped and carry neither -- verified against real
    codex-cli payloads, where ``SessionStart`` and ``SessionEnd`` have no ``turn_id``. That matters:
    ``SessionStart`` sets the agent name recorded in the ``Agent`` asset, so misreading it attests a
    Codex session as Claude Code.

    ``transcript_path`` is the fallback, since both agents name their session store distinctively
    (``~/.codex/sessions/...`` versus ``~/.claude/projects/...``) and every payload carries it.
    """
    if "turn_id" in payload and "prompt_id" not in payload:
        return CODEX
    if "prompt_id" in payload:
        return CLAUDE_CODE

    transcript = payload.get("transcript_path") or ""
    if "/.codex/" in transcript:
        return CODEX
    if "/.claude/" in transcript:
        return CLAUDE_CODE
    return CLAUDE_CODE


def _agent_version(payload: Dict[str, Any]) -> Optional[str]:
    return payload.get("version") or payload.get("agent_version") or payload.get("cli_version")


def _turn_id(payload: Dict[str, Any]) -> Optional[str]:
    return payload.get("prompt_id") or payload.get("turn_id")


def _effort(payload: Dict[str, Any]) -> Optional[str]:
    effort = payload.get("effort")
    if isinstance(effort, dict):
        return effort.get("level")
    return effort if isinstance(effort, str) else None


def to_events(payload: Dict[str, Any], dialect: Optional[str] = None) -> List[Event]:
    """Translate one hook payload into zero or more core events.

    Unknown hook events yield nothing rather than raising. Both agents add events between releases, and
    an adapter that crashed on an unrecognized one would take down the session it is observing.
    """
    dialect = dialect or detect_dialect(payload)
    name = payload.get("hook_event_name") or ""
    return list(_dispatch(name, payload, dialect))


def _dispatch(name: str, p: Dict[str, Any], dialect: str) -> Iterator[Event]:
    at = p.get("timestamp")
    tool_use_id = p.get("tool_use_id")

    if name == "SessionStart":
        yield SessionStarted(
            at=at,
            session_id=p.get("session_id", ""),
            agent=dialect,
            agent_version=_agent_version(p),
            model=p.get("model"),
            cwd=p.get("cwd"),
            permission_mode=p.get("permission_mode"),
            effort=_effort(p),
            source=p.get("source"),
        )

    elif name == "SessionEnd":
        yield SessionEnded(at=at, session_id=p.get("session_id", ""), reason=p.get("reason"))

    elif name == "UserPromptSubmit":
        yield PromptSubmitted(at=at, prompt_id=_turn_id(p), text=p.get("prompt", ""))

    elif name == "InstructionsLoaded":
        # Claude Code only. CLAUDE.md and rules files are *inputs* to everything the agent then does;
        # omitting them makes the manifest attest a conversation rather than a computation.
        yield InstructionsLoaded(
            at=at,
            path=p.get("file_path", ""),
            content=_read_text(p.get("file_path")),
            load_reason=p.get("load_reason"),
        )

    elif name == "PreToolUse":
        yield ToolCallStarted(
            at=at,
            tool_use_id=tool_use_id or "",
            tool_name=p.get("tool_name", "tool"),
            tool_input=p.get("tool_input"),
            tool_source=_command_of(p.get("tool_name"), p.get("tool_input")),
        )

    elif name in ("PostToolUse", "PostToolUseFailure"):
        response = p.get("tool_response")
        # PostToolUse fires only on success in Claude Code; the failure case is a separate event.
        is_error = name == "PostToolUseFailure" or _looks_like_error(response)
        events, _ = file_events_from_result(response, tool_use_id, at)
        yield from events

        if p.get("tool_name") == "apply_patch" and not is_error:
            tool_input = p.get("tool_input") or {}
            for path, mode, content in parse_apply_patch(tool_input.get("command"), p.get("cwd")):
                yield FileObserved(
                    at=at,
                    path=path,
                    content=content.encode("utf-8") if isinstance(content, str) else None,
                    mode=mode,
                    tool_use_id=tool_use_id,
                )
        yield ToolCallEnded(at=at, tool_use_id=tool_use_id or "", result=response, is_error=is_error)

    elif name == "PostToolBatch":
        # The correlation primitive for parallel tool calls: one payload closing several at once.
        for call in p.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("tool_use_id")
            events, _ = file_events_from_result(call.get("result"), call_id, at)
            yield from events
            yield ToolCallEnded(
                at=at,
                tool_use_id=call_id or "",
                result=call.get("result"),
                is_error=bool(call.get("is_error")),
            )

    elif name in ("PermissionRequest", "PermissionDenied"):
        yield PermissionDecision(
            at=at,
            tool_use_id=tool_use_id,
            tool_name=p.get("tool_name"),
            mode=p.get("permission_mode"),
            decision="deny" if name == "PermissionDenied" else "ask",
            reason=p.get("reason"),
            source="classifier" if name == "PermissionDenied" else "user",
        )

    elif name == "FileChanged":
        # The one capability the offline path cannot match: a watcher event makes a Bash side effect
        # *observed* rather than reconstructed from a snapshot diff. Content is read from disk here,
        # which is sound only because the event fires on the change -- see the daemon's note on races.
        path = p.get("file_path", "")
        yield FileObserved(
            at=at,
            path=path,
            content=_read_bytes(path),
            mode="changed",
            observed=True,
        )

    elif name == "SubagentStart":
        yield SubagentStarted(
            at=at,
            agent_id=p.get("agent_id", ""),
            agent_type=p.get("agent_type"),
            prompt=p.get("initial_message"),
        )

    elif name == "SubagentStop":
        # opaque=False: unlike the transcript path, hooks see the subagent's own events.
        yield SubagentEnded(
            at=at,
            agent_id=p.get("agent_id", ""),
            result=p.get("last_assistant_message"),
            opaque=False,
        )

    elif name == "PostCompact":
        yield Compacted(at=at, trigger=p.get("trigger"))

    else:
        logger.debug("no mapping for hook event %r", name)


# Codex writes files with `apply_patch`, whose tool_input.command is a patch document and whose
# tool_response is a plain string. Neither carries `filePath`/`content`/`structuredPatch`, so the
# shape-dispatched parser in core sees nothing and a Codex session yields no file lineage at all
# unless the patch itself is read. Verified against real payloads from codex-cli 0.145.0.
_PATCH_ADD = "*** Add File: "
_PATCH_UPDATE = "*** Update File: "
_PATCH_DELETE = "*** Delete File: "


def parse_apply_patch(patch: str, cwd: Optional[str] = None):
    """Yield ``(path, mode, content)`` for each file an apply_patch document touches.

    ``content`` is the full new text for an added file -- every line is a ``+`` line, so it is exactly
    recoverable. For an *updated* file it is ``None``: the document carries only hunks, and the
    pre-image is not in the payload, so the post-image cannot be reconstructed without reading the
    disk. Returning ``None`` records the path and its transition while declining to invent content,
    which is the same choice the transcript adapter makes when a reconstruction is not confident.
    """
    if not isinstance(patch, str):
        return

    current: Optional[str] = None
    mode = "wrote"
    added: List[str] = []

    def emit():
        if current is None:
            return None
        path = current if cwd is None or current.startswith("/") else f"{cwd.rstrip('/')}/{current}"
        if mode == "added":
            return (path, "wrote", "".join(added))
        return (path, "wrote", None)

    for line in patch.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        for prefix, kind in ((_PATCH_ADD, "added"), (_PATCH_UPDATE, "updated"), (_PATCH_DELETE, "deleted")):
            if stripped.startswith(prefix):
                out = emit()
                if out is not None:
                    yield out
                current, mode, added = stripped[len(prefix):].strip(), kind, []
                break
        else:
            if mode == "added" and line.startswith("+"):
                added.append(line[1:])

    out = emit()
    if out is not None:
        yield out


def _command_of(tool_name: Optional[str], tool_input: Any) -> Optional[str]:
    """A shell command is its own implementation, so use it as the Tool asset's content.

    Content-addressing the asset to the command means a changed command shows up as a changed tool
    rather than silently reusing the previous node.
    """
    if tool_name in ("Bash", "shell", "local_shell") and isinstance(tool_input, dict):
        command = tool_input.get("command")
        if isinstance(command, list):
            return " ".join(str(part) for part in command)
        return command if isinstance(command, str) else None
    return None


def _looks_like_error(response: Any) -> bool:
    if isinstance(response, dict):
        return bool(response.get("is_error") or response.get("interrupted"))
    return False


def _read_text(path: Optional[str]) -> Optional[str]:
    data = _read_bytes(path)
    return data.decode("utf-8", errors="replace") if data is not None else None


def _read_bytes(path: Optional[str], limit: int = 4 << 20) -> Optional[bytes]:
    """Read a file the hook named. Failure is normal and never fatal -- the path may already be gone."""
    if not path:
        return None
    try:
        import os

        if os.path.getsize(path) > limit:
            return None
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


__all__ = [
    "CLAUDE_CODE",
    "CLAUDE_CODE_EVENTS",
    "CODEX",
    "CODEX_EVENTS",
    "detect_dialect",
    "parse_apply_patch",
    "to_events",
]
