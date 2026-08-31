"""The event vocabulary every coding-agent adapter targets.

An adapter's only job is to turn its source -- a Claude Code hook POST, a Codex hook, a session
transcript on disk -- into this sequence of events. :class:`~eqty_lineage.recorder.recorder.LineageRecorder`
turns the sequence into EQTY assets and statements. Keeping the vocabulary in the middle is what lets the
offline and live capture paths be checked against each other: same session, same events, same graph.

Events are frozen because the recorder keeps references to them while a tool call is open; an adapter that
mutated an event after emitting it would silently corrupt the run it belongs to.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

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

    at: str | None = None


@dataclass(frozen=True)
class SessionStarted(Event):
    session_id: str = ""
    # "claude-code" | "codex"; recorded on the Agent asset so a manifest names what produced it
    agent: str = ""
    agent_version: str | None = None
    model: str | None = None
    cwd: str | None = None
    # default|plan|acceptEdits|auto|dontAsk|bypassPermissions -- provenance, not a footnote
    permission_mode: str | None = None
    effort: str | None = None
    git_branch: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class SessionEnded(Event):
    session_id: str = ""
    reason: str | None = None
    outcome: Literal["completed", "failed", "cancelled", "timed_out", "incomplete"] | None = None
    error_type: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class InstructionsLoaded(Event):
    """A CLAUDE.md, AGENTS.md, or .claude/rules/* file entering the agent's context.

    These are *inputs* to everything the agent subsequently does. A manifest that omits them attests a
    conversation rather than a computation.
    """

    path: str = ""
    content: str | None = None
    load_reason: str | None = None


@dataclass(frozen=True)
class PromptSubmitted(Event):
    prompt_id: str | None = None
    text: str = ""


@dataclass(frozen=True)
class ModelCall(Event):
    """One completed model request.

    The offline path fills these in from the transcript. The live path can record one *only* from
    Codex's ``Stop`` hook, which carries ``last_assistant_message`` -- the single hook payload on
    either dialect that contains model output. In that case ``messages_in`` stays None: the request is
    not in the payload, and inventing a prompt entity would attest an input nobody observed.

    Reasoning is absent either way, and attested as absent rather than missing: all three surfaces
    (transcript, Codex rollout, raw API body) carry a signature over withheld content.
    """

    request_id: str | None = None
    model: str = ""
    messages_in: Any = None
    output: Any = None
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class ToolCallStarted(Event):
    tool_use_id: str = ""
    tool_name: str = ""
    tool_input: Any = None
    # tool_use_id of the enclosing call, or the subagent id when running inside one
    parent_id: str | None = None
    # source code or command text, when the adapter can recover it -- content-addresses the Tool asset
    # to its implementation the way @eqty_tool does for LangChain tools
    tool_source: str | None = None
    tool_description: str | None = None


@dataclass(frozen=True)
class ToolCallEnded(Event):
    tool_use_id: str = ""
    result: Any = None
    is_error: bool = False


@dataclass(frozen=True)
class EditAttempt:
    """The replacement an edit performed, carried when its post-image could not be reconstructed.

    A tool result usually states either the new content outright or the ``originalFile`` to replay
    against. When it states neither -- which is most `Edit` results in practice -- the replacement
    itself is still known, and the recorder can often supply the missing pre-image from content the
    session already established for that path. Carrying the parameters is what makes that possible
    without the parser needing any session state of its own.
    """

    old: str
    new: str
    replace_all: bool = False


@dataclass(frozen=True)
class FileObserved(Event):
    """A file version seen by the agent, attached to the tool call that touched it.

    ``observed`` is the honesty flag. True means the capture path saw the content directly (an Edit
    result carrying ``originalFile``, a watcher event). False means it was reconstructed after the fact
    -- a snapshot diff attributing a filesystem change to the Bash command that probably caused it.
    An attestation that cannot distinguish the two is not worth signing.
    """

    path: str = ""
    content: bytes | None = None
    mode: FileMode = "read"
    observed: bool = True
    tool_use_id: str | None = None
    # True when the human, not the agent, is known to have edited this file
    user_modified: bool = False
    # Set only when ``content`` is None and the post-image might still be recoverable by replaying
    # this replacement against a pre-image the recorder holds.
    edit: Optional["EditAttempt"] = None
    # Where the bytes came from, when it is not the tool result itself. "backup-store" means they
    # were read from Claude Code's local file-history backups: real content, but evidence about this
    # machine's disk rather than about the session, and a reader should be able to tell the two
    # apart before signing anything.
    content_source: str | None = None


@dataclass(frozen=True)
class SubagentStarted(Event):
    agent_id: str = ""
    agent_type: str | None = None
    prompt: str | None = None
    model: str | None = None
    parent_tool_use_id: str | None = None


@dataclass(frozen=True)
class SubagentEnded(Event):
    agent_id: str = ""
    result: Any = None
    # aggregate counters when the internals are not visible (the offline path's only view of a subagent)
    stats: dict[str, Any] | None = None
    # True when the capture path could not see the subagent's own tool calls
    opaque: bool = False


@dataclass(frozen=True)
class Compacted(Event):
    """A context compaction -- a lossy derivation edge, recorded with how much it cost.

    Real sessions routinely drop the overwhelming majority of their context here. If the graph does not
    carry the boundary it silently implies a completeness it does not have.
    """

    pre_tokens: int | None = None
    post_tokens: int | None = None
    dropped_tokens: int | None = None
    trigger: str | None = None
    logical_parent: str | None = None


@dataclass(frozen=True)
class PermissionDecision(Event):
    """What the agent was *permitted* to do, as opposed to what it did.

    Neither PROV, OpenLineage, nor in-toto model this. It is the difference between a lineage graph and
    a governance record.
    """

    tool_use_id: str | None = None
    tool_name: str | None = None
    mode: str | None = None
    decision: PermissionOutcome = "allow"
    reason: str | None = None
    # "hook" | "settings-rule" | "user" | "classifier"
    source: str | None = None
    rules: list[str] = field(default_factory=list)


__all__ = [
    "Compacted",
    "EditAttempt",
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
