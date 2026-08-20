"""Claude Code session transcripts (``~/.claude/projects/<slug>/<session>.jsonl``) -> core events.

The transcript is richer than the live hook stream in one respect that matters enormously: ``Edit`` and
``Write`` results carry ``originalFile`` (the full pre-edit content) alongside ``structuredPatch``, so
both sides of every edit are recoverable without touching the filesystem. Nothing has to be re-read, and
a file that changed again afterwards cannot corrupt the version chain.

It is poorer in three respects, all of them recorded rather than hidden:

*Bash effects are not recoverable.* A command's filesystem impact appears only as a delta in the
``file-history-snapshot`` records. That establishes *that* a path changed, not what it became, so those
observations are emitted identity-only and flagged ``observed=False``.

*Subagents are opaque only when their transcript is missing.* An ``Agent`` result in the main
transcript carries aggregate stats and nothing else, which is why this was long recorded with
``opaque=True``. But Claude Code writes each subagent's *own* transcript to a sibling directory --
``<session>/subagents/agent-<agentId>.jsonl``, with workflow agents nested one level deeper -- and the
``agentId`` on the Agent result names the file exactly. Following that link recovers the subagent's
reads, writes and tool calls in full.

The reason this went unnoticed is that :func:`find_sessions` globbed a single directory level, so
those files were never discovered: 990 of 2,236 transcripts on the machine measured, 262 MB, carrying
26% of all file operations. ``opaque=True`` is now emitted only when the transcript is genuinely
absent, which is the honest meaning of the flag.

*Instructions and permission decisions are absent.* There is no record of which CLAUDE.md files were
loaded or which hook allowed a call. Those are hook-path-only, by construction.
"""

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

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

# A backup larger than this is not worth holding in memory to seed a reconstruction; the redaction
# policy applies its own limit to what is ultimately stored.
_MAX_BACKUP_BYTES = 4 << 20


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
        file_history_root: Path | None = None,
        include_partial_reads: bool = True,
        include_subagents: bool = True,
        include_file_history: bool = True,
    ) -> None:
        self.path = Path(path)
        self.file_history_root = file_history_root if file_history_root is not None else DEFAULT_FILE_HISTORY_ROOT
        self.include_partial_reads = include_partial_reads
        self.include_subagents = include_subagents
        self.include_file_history = include_file_history
        self._subagent_transcripts = find_subagent_transcripts(self.path) if include_subagents else {}
        # agentIds whose transcript was found and inlined, so the summary node can say so honestly
        self.subagents_recovered: list[str] = []
        self.warnings: list[str] = []
        # tool_use id -> (name, input), populated from assistant records before the result arrives
        self._pending: dict[str, tuple[str, Any]] = {}
        # path -> backup version last seen in a file-history-snapshot
        self._backup_versions: dict[str, int] = {}
        # paths already attributed to an Edit/Write this session, so a snapshot delta does not
        # double-report a change the tool result already described in full
        self._attributed: set = set()

    # ------------------------------------------------------------------ record loading
    def _records(self) -> Iterator[dict[str, Any]]:
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

    def _session_header(self, records: list[dict[str, Any]]) -> SessionStarted:
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
    def _file_events_for(self, tool_use_id: str, result: Any, at: str | None):
        """Delegate to the shared parser, tracking which paths a result fully explained."""
        events, attributed = file_events_from_result(
            result, tool_use_id, at, include_partial_reads=self.include_partial_reads
        )
        if attributed is not None:
            self._attributed.add(attributed)
        if events and any(e.content is None and e.mode == "wrote" for e in events):
            self.warnings.append(f"could not reconstruct post-edit content for {events[-1].path}")
        return events

    def _backup_content(self, info: dict[str, Any]) -> bytes | None:
        """The pre-change bytes a snapshot entry points at, if they are still on disk.

        Claude Code keeps them under ``~/.claude/file-history/<session>/<backupFileName>``, and the
        snapshot names the file exactly. Measured on a real machine: 131,751 of 131,989 references
        still resolve, 134 MB in the store. The adapter has accepted a ``file_history_root`` since
        the beginning and never read from it.

        Failure is normal and never fatal -- the store is pruned, and an older session's backups may
        be long gone.
        """
        name = info.get("backupFileName")
        if not isinstance(name, str) or not name:
            return None
        candidate = self.file_history_root / self.path.stem / name
        try:
            if candidate.stat().st_size > _MAX_BACKUP_BYTES:
                return None
            return candidate.read_bytes()
        except OSError:
            return None

    def _snapshot_events(self, record: dict[str, Any], at: str | None) -> Iterator[FileObserved]:
        """Emit change observations from a file-history-snapshot delta.

        A delta establishes that a path changed without establishing what it became, so the change
        itself stays identity-only and ``observed=False``: claiming a post-state here would be a
        guess wearing a signature.

        The *pre*-change content, though, is not a guess. It is sitting in the backup store, and
        emitting it does two things. It gives the version chain real bytes on the side we actually
        know, and -- because the recorder keeps the last content it saw per path -- it seeds the
        chain, so a later edit carrying only ``oldString``/``newString`` can be replayed against it.
        The bytes are marked ``backup-store`` rather than folded in with the tool's own testimony,
        because they are evidence about this machine's disk rather than about the session.
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

            content = self._backup_content(info) if self.include_file_history else None
            if content is not None:
                # What the file was before this change. Read-mode because the agent did not produce
                # it: it is the state something else then modified.
                yield FileObserved(
                    at=at,
                    path=path,
                    content=content,
                    mode="read",
                    observed=True,
                    content_source="backup-store",
                )

            yield FileObserved(at=at, path=path, content=None, mode="changed", observed=False)

    # ------------------------------------------------------------------ main pass
    def events(self) -> Iterator[Event]:
        records = list(self._records())
        if not records:
            return

        header = self._session_header(records)
        yield header

        yield from self._body_events(records)

        # Anything still pending never reported a result.
        for tool_use_id, (name, _) in list(self._pending.items()):
            self.warnings.append(f"tool call {name}/{tool_use_id} has no result in transcript")

        yield SessionEnded(at=records[-1].get("timestamp"), session_id=header.session_id)

    def _body_events(self, records) -> Iterator[Event]:
        """Everything between the session brackets.

        Separated so a subagent's transcript can be inlined into its parent's stream. A subagent has
        its own SessionStarted/SessionEnded, and yielding those would reset the recorder's session
        state mid-session -- the agent asset, the context anchor and the permission mode would all be
        replaced by the child's.
        """
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
            # The result names its own transcript. Following that link turns an opaque summary node
            # into the subagent's actual reads, writes and tool calls.
            agent_id = result.get("agentId")
            transcript = self._subagent_transcripts.get(agent_id) if agent_id else None

            if transcript is not None:
                inner = ClaudeCodeTranscript(
                    transcript,
                    file_history_root=self.file_history_root,
                    include_partial_reads=self.include_partial_reads,
                )
                try:
                    yield from inner._body_events(list(inner._records()))
                except OSError as exc:  # a transcript that vanished mid-read is not fatal
                    self.warnings.append(f"subagent {agent_id}: {exc}")
                    transcript = None
                else:
                    self.warnings.extend(inner.warnings)
                    self.subagents_recovered.append(agent_id)

            yield SubagentEnded(
                at=at,
                agent_id=agent_id or tool_use_id,
                result=result.get("content"),
                stats=result.get("toolStats"),
                # Only true when the internals really were not visible. Saying it while the
                # transcript sits in a sibling directory asserts a limitation that does not exist.
                opaque=transcript is None,
            )

        yield ToolCallEnded(at=at, tool_use_id=tool_use_id, result=result, is_error=is_error)


def _bash_command(tool_name: str, tool_input: Any) -> str | None:
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


def find_sessions(root: Path | None = None) -> list[Path]:
    """Top-level Claude Code session transcripts on this machine, newest first.

    Deliberately *not* recursive. A subagent's transcript lives under
    ``<session>/subagents/`` and belongs to its parent session rather than standing alone, so
    returning it here would double-count every subagent's work as an independent session. Use
    :func:`find_subagent_transcripts` to reach them, or let :class:`ClaudeCodeTranscript` follow the
    link itself.
    """
    base = root if root is not None else Path.home() / ".claude" / "projects"
    files: Iterable[Path] = base.glob("*/*.jsonl")
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def find_subagent_transcripts(session: Path) -> dict[str, Path]:
    """``agentId`` -> transcript, for the subagents a session spawned.

    The layout, from real sessions::

        <project>/<session>.jsonl                                  the session
        <project>/<session>/subagents/agent-<id>.jsonl             a subagent
        <project>/<session>/subagents/workflows/wf_<id>/agent-<id>.jsonl

    Workflow agents nest one level deeper, so this recurses. Nesting is flattened into a single map:
    every agent found is attributed to the session, which is accurate about *what ran* while not
    reconstructing the parent/child chain among the agents themselves.
    """
    session = Path(session)
    root = session.parent / session.stem / "subagents"
    if not root.is_dir():
        return {}
    return {p.stem[len("agent-") :]: p for p in sorted(root.rglob("agent-*.jsonl"))}


__all__ = ["ClaudeCodeTranscript", "find_sessions", "parse"]
