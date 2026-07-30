"""The event vocabulary every coding-agent adapter targets.

An adapter's only job is to turn its source -- a Claude Code hook POST, a Codex hook, a session
transcript on disk -- into this sequence of events. :class:`~eqty_lineage.core.recorder.LineageRecorder`
turns the sequence into EQTY assets and statements. Keeping the vocabulary in the middle is what lets the
offline and live capture paths be checked against each other: same session, same events, same graph.

Events are frozen because the recorder keeps references to them while a tool call is open; an adapter that
mutated an event after emitting it would silently corrupt the run it belongs to.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# How a file was touched. "read" makes the file version an input to the enclosing computation;
# "wrote" and "changed" make it an output. The distinction between the latter two is provenance, not
# direction: "wrote" came from a tool whose declared purpose was to write, "changed" was noticed
# afterwards (a watcher event, a snapshot diff) and may not be attributable to any single tool call.
FileMode = Literal["read", "wrote", "changed", "deleted"]
"""``deleted`` is not a write with missing content. A watcher reports removal and a read-back returns
nothing, which is byte-identical to a file the capture path simply failed to read -- and those two mean
opposite things in a lineage graph. Kept as its own mode so the tombstone is explicit."""

PermissionOutcome = Literal["allow", "deny", "ask", "defer"]


@dataclass(frozen=True)
class Event:
    """Base for every event. ``at`` is the source's own timestamp, not wall-clock at ingest time.

    The offline path replays sessions long after they ran, so the recorder must never call ``now()``
    itself -- doing so would make the two capture paths disagree on every statement.
    """

    at: Optional[str] = None


@dataclass(frozen=True)
class SessionStarted(Event):
    session_id: str = ""
    # "claude-code" | "codex"; recorded on the Agent asset so a manifest names what produced it
    agent: str = ""
    agent_version: Optional[str] = None
    model: Optional[str] = None
    cwd: Optional[str] = None
    # default|plan|acceptEdits|auto|dontAsk|bypassPermissions -- provenance, not a footnote
    permission_mode: Optional[str] = None
    effort: Optional[str] = None
    git_branch: Optional[str] = None
    source: Optional[str] = None


@dataclass(frozen=True)
class SessionEnded(Event):
    session_id: str = ""
    reason: Optional[str] = None


@dataclass(frozen=True)
class InstructionsLoaded(Event):
    """A CLAUDE.md, AGENTS.md, or .claude/rules/* file entering the agent's context.

    These are *inputs* to everything the agent subsequently does. A manifest that omits them attests a
    conversation rather than a computation.
    """

    path: str = ""
    content: Optional[str] = None
    load_reason: Optional[str] = None


@dataclass(frozen=True)
class PromptSubmitted(Event):
    prompt_id: Optional[str] = None
    text: str = ""


@dataclass(frozen=True)
class ModelCall(Event):
    """One completed model request. Only the offline path sees these -- hooks never carry the messages."""

    request_id: Optional[str] = None
    model: str = ""
    messages_in: Any = None
    output: Any = None
    usage: Optional[Dict[str, Any]] = None
    stop_reason: Optional[str] = None


@dataclass(frozen=True)
class ToolCallStarted(Event):
    tool_use_id: str = ""
    tool_name: str = ""
    tool_input: Any = None
    # tool_use_id of the enclosing call, or the subagent id when running inside one
    parent_id: Optional[str] = None
    # source code or command text, when the adapter can recover it -- content-addresses the Tool asset
    # to its implementation the way @eqty_tool does for LangChain tools
    tool_source: Optional[str] = None
    tool_description: Optional[str] = None


@dataclass(frozen=True)
class ToolCallEnded(Event):
    tool_use_id: str = ""
    result: Any = None
    is_error: bool = False


@dataclass(frozen=True)
class FileObserved(Event):
    """A file version seen by the agent, attached to the tool call that touched it.

    ``observed`` is the honesty flag. True means the capture path saw the content directly (an Edit
    result carrying ``originalFile``, a watcher event). False means it was reconstructed after the fact
    -- a snapshot diff attributing a filesystem change to the Bash command that probably caused it.
    An attestation that cannot distinguish the two is not worth signing.
    """

    path: str = ""
    content: Optional[bytes] = None
    mode: FileMode = "read"
    observed: bool = True
    tool_use_id: Optional[str] = None
    # True when the human, not the agent, is known to have edited this file
    user_modified: bool = False


@dataclass(frozen=True)
class SubagentStarted(Event):
    agent_id: str = ""
    agent_type: Optional[str] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    parent_tool_use_id: Optional[str] = None


@dataclass(frozen=True)
class SubagentEnded(Event):
    agent_id: str = ""
    result: Any = None
    # aggregate counters when the internals are not visible (the offline path's only view of a subagent)
    stats: Optional[Dict[str, Any]] = None
    # True when the capture path could not see the subagent's own tool calls
    opaque: bool = False


@dataclass(frozen=True)
class Compacted(Event):
    """A context compaction -- a lossy derivation edge, recorded with how much it cost.

    Real sessions routinely drop the overwhelming majority of their context here. If the graph does not
    carry the boundary it silently implies a completeness it does not have.
    """

    pre_tokens: Optional[int] = None
    post_tokens: Optional[int] = None
    dropped_tokens: Optional[int] = None
    trigger: Optional[str] = None
    logical_parent: Optional[str] = None


@dataclass(frozen=True)
class PermissionDecision(Event):
    """What the agent was *permitted* to do, as opposed to what it did.

    Neither PROV, OpenLineage, nor in-toto model this. It is the difference between a lineage graph and
    a governance record.
    """

    tool_use_id: Optional[str] = None
    tool_name: Optional[str] = None
    mode: Optional[str] = None
    decision: PermissionOutcome = "allow"
    reason: Optional[str] = None
    # "hook" | "settings-rule" | "user" | "classifier"
    source: Optional[str] = None
    rules: List[str] = field(default_factory=list)


__all__ = [
    "Compacted",
    "Event",
    "FileMode",
    "FileObserved",
    "InstructionsLoaded",
    "ModelCall",
    "PermissionDecision",
    "PermissionOutcome",
    "PromptSubmitted",
    "SessionEnded",
    "SessionStarted",
    "SubagentEnded",
    "SubagentStarted",
    "ToolCallEnded",
    "ToolCallStarted",
]
