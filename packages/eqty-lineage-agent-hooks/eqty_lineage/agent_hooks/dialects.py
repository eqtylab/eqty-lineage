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
import re
from collections.abc import Iterator
from typing import Any

from eqty_lineage.core import (
    Compacted,
    Event,
    FileObserved,
    InstructionsLoaded,
    ModelCall,
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
#
# `Stop` is deliberately absent: it fires once per turn and carries only `last_assistant_message`, which
# the recorder has no event for. Subscribing would buy a round trip per turn and produce nothing.
#: Every hook event codex-cli emits, in the order the binary's own enum lists them.
#:
#: Read out of the codex-cli 0.148.0 binary rather than from documentation, which is how every Codex
#: finding in this package was made. The enum is contiguous in the image:
#: ``PreToolUse PermissionRequest PostToolUse PreCompact PostCompact SessionStart SessionEnd
#: UserPromptSubmit SubagentStart SubagentStop Stop``.
#:
#: There is no ``PostToolUseFailure``, no ``PostToolBatch`` and no ``FileChanged`` -- those are Claude
#: Code events. ``_dispatch`` handles them anyway because both dialects share it, and a Codex release
#: that added one would then work without a code change.
CODEX_EVENTS = (
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "PreCompact",
    "PostCompact",
    "SubagentStart",
    "SubagentStop",
    "Stop",
)


def detect_dialect(payload: dict[str, Any]) -> str:
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


def _agent_version(payload: dict[str, Any]) -> str | None:
    return payload.get("version") or payload.get("agent_version") or payload.get("cli_version")


def _turn_id(payload: dict[str, Any]) -> str | None:
    return payload.get("prompt_id") or payload.get("turn_id")


def _effort(payload: dict[str, Any]) -> str | None:
    effort = payload.get("effort")
    if isinstance(effort, dict):
        return effort.get("level")
    return effort if isinstance(effort, str) else None


def to_events(payload: dict[str, Any], dialect: str | None = None) -> list[Event]:
    """Translate one hook payload into zero or more core events.

    Unknown hook events yield nothing rather than raising. Both agents add events between releases, and
    an adapter that crashed on an unrecognized one would take down the session it is observing.
    """
    dialect = dialect or detect_dialect(payload)
    name = payload.get("hook_event_name") or ""
    return list(_dispatch(name, payload, dialect))


def _dispatch(name: str, p: dict[str, Any], dialect: str) -> Iterator[Event]:
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
        # PostToolUse fires only on success in Claude Code; the failure case is a separate event --
        # and it carries the outcome under `error`, not `tool_response`. Reading `tool_response` for a
        # failure yields None, so the call that an audit most wants to see records "no result".
        response = p.get("error") if name == "PostToolUseFailure" else p.get("tool_response")
        is_error = name == "PostToolUseFailure" or _looks_like_error(response)
        if name == "PostToolUseFailure" and p.get("is_interrupt"):
            # An interrupt is not a tool failure. Both end the call, and only this flag separates
            # "the tool broke" from "the user stopped it".
            response = {"error": response, "interrupted": True}
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
        # Each entry is {tool_name, tool_input, tool_use_id, tool_response} -- the same result key as
        # PostToolUse, not `result`, and there is no per-call error flag to read.
        for call in p.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("tool_use_id")
            response = call.get("tool_response")
            events, _ = file_events_from_result(response, call_id, at)
            yield from events
            yield ToolCallEnded(
                at=at,
                tool_use_id=call_id or "",
                result=response,
                is_error=_looks_like_error(response),
            )

    elif name in ("PermissionRequest", "PermissionDenied"):
        # Codex's PermissionRequest carries no `reason`; the human-readable account of what is being
        # asked for -- and it is the agent's own words -- lives in `tool_input.description`. Captured
        # live from codex-cli 0.148.0: {"command": "printf ... > /outside.txt", "description": "Allow
        # writing the text 'hello' to /outside.txt outside the workspace?"}. Reading only `reason`
        # recorded every Codex escalation with no stated cause, which is the field an audit reads first.
        #
        # It also carries no `tool_use_id`, so the decision cannot be tied to the call it authorizes --
        # only to the turn. `PreToolUse` for the same call does carry one.
        tool_input = p.get("tool_input")
        described = tool_input.get("description") if isinstance(tool_input, dict) else None
        yield PermissionDecision(
            at=at,
            tool_use_id=tool_use_id,
            tool_name=p.get("tool_name"),
            mode=p.get("permission_mode"),
            decision="deny" if name == "PermissionDenied" else "ask",
            reason=p.get("reason") or described,
            source="classifier" if name == "PermissionDenied" else "user",
        )

    elif name == "FileChanged":
        # The one capability the offline path cannot match: a watcher event makes a Bash side effect
        # *observed* rather than reconstructed from a snapshot diff. Content is read from disk here and
        # is therefore as-of-read, not as-of-change -- see the daemon's note on watcher races.
        #
        # `change_type` is "change" or "unlink". Treating a removal as a write would record a version
        # whose content merely failed to load, which is the opposite claim.
        path = p.get("file_path", "")
        removed = p.get("change_type") == "unlink"
        yield FileObserved(
            at=at,
            path=path,
            content=None if removed else _read_bytes(path),
            mode="deleted" if removed else "changed",
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

    elif name == "Stop":
        # The turn's final assistant message. This is the only hook payload on either dialect that
        # carries model output, so it is the one place the live path can record a ModelCall at all --
        # everywhere else the messages are absent and inventing a node would attest an input nobody
        # observed. `messages_in` stays None for exactly that reason: the request is not in the
        # payload, so the response is recorded without pretending to know what produced it.
        #
        # `stop_hook_active` marks a Stop raised by a previous Stop hook rather than by the turn
        # ending. Recording those as model calls would count one turn many times.
        if not p.get("stop_hook_active"):
            yield ModelCall(
                at=at,
                request_id=_turn_id(p),
                model=p.get("model", ""),
                output=p.get("last_assistant_message"),
                stop_reason="stop",
            )

    elif name == "PreCompact":
        # Subscribed but deliberately silent. The boundary is recorded on PostCompact, where the
        # compaction actually happened; emitting on both would count every compaction twice and
        # inflate the coverage counters that exist to say how much context was lost.
        #
        # The subscription is kept because the *pair* is informative in a way neither half is: a
        # PreCompact with no matching PostCompact is a compaction that began and never finished,
        # which is a session whose later work has no recorded provenance for its context. Costs one
        # round trip per compaction, which is a handful per session.
        return

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


def parse_apply_patch(patch: str, cwd: str | None = None):
    """Yield ``(path, mode, content)`` for each file an apply_patch document touches.

    ``content`` is the full new text for an added file -- every line is a ``+`` line, so it is exactly
    recoverable. For an *updated* file it is ``None``: the document carries only hunks, and the
    pre-image is not in the payload, so the post-image cannot be reconstructed without reading the
    disk. Returning ``None`` records the path and its transition while declining to invent content,
    which is the same choice the transcript adapter makes when a reconstruction is not confident.
    """
    if not isinstance(patch, str):
        return

    current: str | None = None
    mode = "wrote"
    added: list[str] = []

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
                current, mode, added = stripped[len(prefix) :].strip(), kind, []
                break
        else:
            if mode == "added" and line.startswith("+"):
                added.append(line[1:])

    out = emit()
    if out is not None:
        yield out


def _command_of(tool_name: str | None, tool_input: Any) -> str | None:
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


_EXIT_CODE = re.compile(r"^\s*Exit code:\s*(-?\d+)")


def _looks_like_error(response: Any) -> bool:
    """Whether a tool result reports failure.

    Codex returns a plain string rather than a mapping, so inspecting only dicts can never see a Codex
    failure at all. Some of those strings begin ``Exit code: N``, which is a real signal and is read
    here.

    **Codex shell failures are not detectable.** Captured from codex-cli 0.145.0: a failing
    ``ls /nonexistent`` returns ``"ls: /nonexistent: No such file or directory\\n"`` and a succeeding
    ``echo hello`` returns ``"hello\\n"``. Bare output either way -- no exit code, no flag, nothing that
    separates them. The ``Exit code:`` prefix appears on the ``apply_patch`` shape, not on shell results.
    Guessing from content (looking for "No such file", say) would attach a signed ``is-error`` claim to a
    string that merely mentions an error, so this reports failure only where the payload states it.
    """
    if isinstance(response, dict):
        return bool(response.get("is_error") or response.get("interrupted"))
    if isinstance(response, str):
        match = _EXIT_CODE.match(response)
        return match is not None and match.group(1) != "0"
    return False


def _read_text(path: str | None) -> str | None:
    data = _read_bytes(path)
    return data.decode("utf-8", errors="replace") if data is not None else None


def _read_bytes(path: str | None, limit: int = 4 << 20) -> bytes | None:
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
