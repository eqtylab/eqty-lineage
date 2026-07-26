"""Claude Code session transcripts (``~/.claude/projects/<slug>/<session>.jsonl``) -> core events.

The transcript is richer than the live hook stream in one respect that matters enormously: ``Edit`` and
``Write`` results carry ``originalFile`` (the full pre-edit content) alongside ``structuredPatch``, so
both sides of every edit are recoverable without touching the filesystem. Nothing has to be re-read, and
a file that changed again afterwards cannot corrupt the version chain.

It is poorer in three respects, all of them recorded rather than hidden:

*Bash effects are not recoverable.* A command's filesystem impact appears only as a delta in the
``file-history-snapshot`` records. That establishes *that* a path changed, not what it became, so those
observations are emitted identity-only and flagged ``observed=False``.

*Subagents are opaque.* ``isSidechain`` was false across all 2,072 sessions surveyed; an ``Agent`` result
carries aggregate ``toolStats`` and nothing else. Emitted with ``opaque=True``.

*Instructions and permission decisions are absent.* There is no record of which CLAUDE.md files were
loaded or which hook allowed a call. Those are hook-path-only, by construction.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from eqty_lineage.core import (
    Compacted,
    Event,
    FileObserved,
    ModelCall,
    PromptSubmitted,
    SessionEnded,
    SessionStarted,
    SubagentEnded,
    SubagentStarted,
    ToolCallEnded,
    ToolCallStarted,
    file_events_from_result,
)

logger = logging.getLogger("eqty.lineage.transcript")

# Record types carrying conversation or lineage. Everything else in the file is UI state --
# ai-title, mode, permission-mode, last-prompt, agent-name, attachment, pr-link, queue-operation --
# and is skipped. Allowlisting rather than denylisting means a new UI record type added in a future
# release is ignored by default instead of parsed as something it is not.
CONVERSATIONAL_TYPES = frozenset({"assistant", "user", "system"})

DEFAULT_FILE_HISTORY_ROOT = Path.home() / ".claude" / "file-history"


def _text_of(content: Any) -> str:
    """Flatten an assistant/user content list to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


class ClaudeCodeTranscript:
    """Parses one session file into an ordered stream of core events."""

    def __init__(
        self,
        path: Path,
        file_history_root: Optional[Path] = None,
        include_partial_reads: bool = True,
    ) -> None:
        self.path = Path(path)
        self.file_history_root = file_history_root if file_history_root is not None else DEFAULT_FILE_HISTORY_ROOT
        self.include_partial_reads = include_partial_reads
        self.warnings: List[str] = []
        # tool_use id -> (name, input), populated from assistant records before the result arrives
        self._pending: Dict[str, Tuple[str, Any]] = {}
        # path -> backup version last seen in a file-history-snapshot
        self._backup_versions: Dict[str, int] = {}
        # paths already attributed to an Edit/Write this session, so a snapshot delta does not
        # double-report a change the tool result already described in full
        self._attributed: set = set()

    # ------------------------------------------------------------------ record loading
    def _records(self) -> Iterator[Dict[str, Any]]:
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    # A truncated tail is normal for a session that is still running.
                    self.warnings.append(f"{self.path.name}:{line_no}: malformed JSON, skipped")

    def _session_header(self, records: List[Dict[str, Any]]) -> SessionStarted:
        """Assemble SessionStarted from fields scattered across several record types."""
        first = next((r for r in records if r.get("type") in CONVERSATIONAL_TYPES), {})
        model = next(
            (
                r["message"]["model"]
                for r in records
                if r.get("type") == "assistant" and (r.get("message") or {}).get("model")
            ),
            None,
        )
        permission_mode = next((r["permissionMode"] for r in records if r.get("type") == "permission-mode"), None)

        return SessionStarted(
            at=first.get("timestamp"),
            session_id=first.get("sessionId") or self.path.stem,
            agent="claude-code",
            agent_version=first.get("version"),
            model=model,
            cwd=first.get("cwd"),
            permission_mode=permission_mode,
            git_branch=first.get("gitBranch"),
            source=first.get("entrypoint"),
        )

    # ------------------------------------------------------------------ file observations
    def _file_events_for(self, tool_use_id: str, result: Any, at: Optional[str]):
        """Delegate to the shared parser, tracking which paths a result fully explained."""
        events, attributed = file_events_from_result(
            result, tool_use_id, at, include_partial_reads=self.include_partial_reads
        )
        if attributed is not None:
            self._attributed.add(attributed)
        if events and any(e.content is None and e.mode == "wrote" for e in events):
            self.warnings.append(f"could not reconstruct post-edit content for {events[-1].path}")
        return events

    def _snapshot_events(self, record: Dict[str, Any], at: Optional[str]) -> Iterator[FileObserved]:
        """Emit inferred change observations from a file-history-snapshot delta.

        The backup store holds *pre-change* content, so a delta establishes that a path changed without
        establishing what it became. These are emitted identity-only and flagged ``observed=False``;
        claiming a post-state here would be a guess wearing a signature.
        """
        backups = ((record.get("snapshot") or {}).get("trackedFileBackups")) or {}
        for path, info in backups.items():
            if not isinstance(info, dict):
                continue
            version = info.get("version")
            if not isinstance(version, int):
                continue
            previous = self._backup_versions.get(path)
            self._backup_versions[path] = version
            if previous is None or version <= previous:
                continue
            if path in self._attributed:
                # An Edit/Write already described this transition in full; the snapshot is the same
                # event seen from a worse angle.
                self._attributed.discard(path)
                continue
            yield FileObserved(at=at, path=path, content=None, mode="changed", observed=False)

    # ------------------------------------------------------------------ main pass
    def events(self) -> Iterator[Event]:
        records = list(self._records())
        if not records:
            return

        header = self._session_header(records)
        yield header

        for record in records:
            rtype = record.get("type")
            at = record.get("timestamp")

            if rtype == "file-history-snapshot":
                yield from self._snapshot_events(record, at)
                continue

            if rtype == "system":
                if record.get("subtype") == "compact_boundary":
                    meta = record.get("compactMetadata") or {}
                    yield Compacted(
                        at=at,
                        pre_tokens=meta.get("preTokens"),
                        post_tokens=meta.get("postTokens"),
                        dropped_tokens=meta.get("cumulativeDroppedTokens"),
                        trigger=meta.get("trigger"),
                        # the chain is severed here: parentUuid is null and the real predecessor is
                        # logicalParentUuid. Anything walking parentUuid naively splits the session.
                        logical_parent=record.get("logicalParentUuid"),
                    )
                continue

            if rtype not in CONVERSATIONAL_TYPES:
                continue

            message = record.get("message") or {}
            content = message.get("content")

            if rtype == "assistant":
                yield from self._assistant_events(record, message, content, at)
            elif rtype == "user":
                yield from self._user_events(record, message, content, at)

        # Anything still pending never reported a result.
        for tool_use_id, (name, _) in list(self._pending.items()):
            self.warnings.append(f"tool call {name}/{tool_use_id} has no result in transcript")

        yield SessionEnded(at=records[-1].get("timestamp"), session_id=header.session_id)

    def _assistant_events(self, record, message, content, at) -> Iterator[Event]:
        blocks = content if isinstance(content, list) else []
        tool_uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        text = _text_of(content)

        if text or message.get("usage"):
            # The transcript stores the conversation, not the request payload, so the model's input is
            # not recoverable. Recording the output and the usage without inventing a prompt keeps this
            # honest; the live hook path cannot see either.
            yield ModelCall(
                at=at,
                request_id=record.get("requestId") or message.get("id"),
                model=message.get("model") or "unknown-model",
                messages_in=None,
                output=text or [b for b in (content or []) if isinstance(b, dict)],
                usage=message.get("usage"),
                stop_reason=message.get("stop_reason"),
            )

        for block in tool_uses:
            tool_use_id = block.get("id")
            name = block.get("name") or "tool"
            if not tool_use_id:
                continue
            self._pending[tool_use_id] = (name, block.get("input"))
            yield ToolCallStarted(
                at=at,
                tool_use_id=tool_use_id,
                tool_name=name,
                tool_input=block.get("input"),
                tool_source=_bash_command(name, block.get("input")),
            )
            if name in ("Agent", "Task"):
                agent_input = block.get("input") or {}
                yield SubagentStarted(
                    at=at,
                    agent_id=tool_use_id,
                    agent_type=agent_input.get("subagent_type") or agent_input.get("agentType"),
                    prompt=agent_input.get("prompt"),
                    parent_tool_use_id=tool_use_id,
                )

    def _user_events(self, record, message, content, at) -> Iterator[Event]:
        if "toolUseResult" not in record:
            text = _text_of(content)
            if text:
                yield PromptSubmitted(at=at, prompt_id=record.get("promptId"), text=text)
            return

        result = record["toolUseResult"]
        tool_use_id = None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    break
        if tool_use_id is None:
            return

        pending = self._pending.pop(tool_use_id, None)
        if pending is None:
            # A resumed session carries results whose tool_use block lives in the parent transcript.
            # Synthesizing the start keeps the result -- often a file read or edit -- in the graph
            # instead of dropping it, while naming the tool "unknown" so nothing is claimed that is not
            # known. Roughly 0.5% of tool results in the local corpus land here.
            self.warnings.append(f"result for {tool_use_id} has no tool_use in this transcript (resumed session?)")
            yield ToolCallStarted(at=at, tool_use_id=tool_use_id, tool_name="unknown", tool_input=None)
            name = "unknown"
        else:
            name = pending[0]
        is_error = _is_error(content, result)

        # File observations must precede the call's end so the recorder can attach them to the open run.
        yield from self._file_events_for(tool_use_id, result, at)

        if name in ("Agent", "Task") and isinstance(result, dict):
            yield SubagentEnded(
                at=at,
                agent_id=tool_use_id,
                result=result.get("content"),
                stats=result.get("toolStats"),
                opaque=True,
            )

        yield ToolCallEnded(at=at, tool_use_id=tool_use_id, result=result, is_error=is_error)


def _bash_command(tool_name: str, tool_input: Any) -> Optional[str]:
    """Use a Bash command string as the Tool asset's content, mirroring @eqty_tool's source capture.

    The command *is* the implementation for a Bash call, so content-addressing the asset to it means a
    changed command shows up as a changed tool rather than silently reusing the previous node.
    """
    if tool_name == "Bash" and isinstance(tool_input, dict):
        command = tool_input.get("command")
        return command if isinstance(command, str) else None
    return None


def _is_error(content: Any, result: Any) -> bool:
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                return True
    if isinstance(result, dict):
        if result.get("is_error") or result.get("interrupted"):
            return True
        if isinstance(result.get("stderr"), str) and result.get("stderr") and not result.get("stdout"):
            # a command that produced only stderr is not necessarily failed; treat as success unless
            # the transcript says otherwise, to avoid over-reporting errors
            return False
    return False


def parse(path, **kwargs) -> Iterator[Event]:
    """Convenience wrapper: yield core events for one transcript file."""
    return ClaudeCodeTranscript(Path(path), **kwargs).events()


def find_sessions(root: Optional[Path] = None) -> List[Path]:
    """All Claude Code session transcripts on this machine, newest first."""
    base = root if root is not None else Path.home() / ".claude" / "projects"
    files: Iterable[Path] = base.glob("*/*.jsonl")
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


__all__ = ["ClaudeCodeTranscript", "find_sessions", "parse"]
