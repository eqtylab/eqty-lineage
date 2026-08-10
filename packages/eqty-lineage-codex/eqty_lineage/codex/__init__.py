"""Small, signed Codex hook-to-graph adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eqty_lineage.codex.capture import (
    ALLOW,
    DENY,
    UNKNOWN,
    CodexSession,
    load_session,
    normalize,
)
from eqty_sdk import Context, Dataset, Guardrail, Prompt, Signer, Tool, init, set_active_signer
from eqty_sdk.context import graph_context
from eqty_sdk.metadata import Metadata
from eqty_sdk.statements import add_computation_statement


class CodexLineage:
    """Capture the minimum useful Codex session graph without another framework dependency."""

    def __init__(self, output: str | Path, session_id: str = "codex-demo") -> None:
        self.output = Path(output)
        self.session_id = session_id
        init().set_store_all_blobs(False)
        set_active_signer(Signer.new(name="eqty-lineage-codex", _load_if_exists=True))
        self.context = Context.new(f"Codex lineage: {session_id}")
        self._prompt: Prompt | None = None

    def _metadata(self, cid: Any, **fields: Any) -> None:
        Metadata(session_id=self.session_id, **fields).create_statement(cid, None, self.context)

    def prompt(self, text: str) -> None:
        with graph_context(self.context):
            self._prompt = Prompt.from_object(text, name="Codex user prompt")
            self._metadata(self._prompt.cid, asset_type="Prompt")

    def tool(
        self,
        name: str,
        tool_input: dict[str, Any],
        *,
        decision: str,
        executed: bool | None = None,
        result: Any = None,
        reason: str = "policy decision",
    ) -> None:
        """Record one tool attempt: its input, the decision about it, and its result if it ran.

        ``decision`` is ``allow``, ``deny`` or ``unknown``; ``unknown`` is a real outcome and is
        recorded as one, because a capture that ends mid-call cannot distinguish a denial from a
        truncation. A result node is attached only when the call actually executed, so a denied or
        unresolved attempt has a guardrail node and no result -- which is the claim the graph should
        make.
        """
        if decision not in (ALLOW, DENY, UNKNOWN):
            raise ValueError(f"decision must be {ALLOW!r}, {DENY!r} or {UNKNOWN!r}, got {decision!r}")
        ran = decision == ALLOW if executed is None else executed

        with graph_context(self.context):
            tool_asset = Tool.from_object({"name": name}, name=name)
            request = Dataset.from_object(tool_input, name=f"{name} input")
            guardrail = Guardrail.from_object(
                {"decision": decision, "reason": reason},
                name=f"{decision}: {name}",
            )
            outputs = [guardrail.cid]
            if ran:
                outputs.append(Dataset.from_object(result if result is not None else {}, name=f"{name} result").cid)
            inputs = [asset.cid for asset in (self._prompt, tool_asset, request) if asset is not None]
            activity = add_computation_statement(inputs=inputs, outputs=outputs, context=self.context)[0]
            Metadata(
                name=f"Codex {name}",
                computation_type="tool",
                framework="codex",
                session_id=self.session_id,
                decision=decision,
                executed=ran,
            ).create_statement(activity, None, self.context)

    def replay(self, session: CodexSession) -> None:
        """Build the graph for an already-normalized capture."""
        for text in session.prompts:
            self.prompt(text)
        for attempt in session.attempts:
            self.tool(
                attempt.tool_name,
                attempt.tool_input,
                decision=attempt.decision,
                executed=attempt.executed,
                result=attempt.result,
                reason=attempt.reason or "no decision recorded",
            )

    def export(self) -> Path:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.context.export(self.output)
        return self.output


def replay_capture(capture: str | Path, output: str | Path) -> Path:
    """Read a raw hook capture and export the signed graph it attests.

    This is the whole point of the collector: the manifest is derived from bytes Codex emitted, not
    from a script's idea of what a session looks like.
    """
    session = load_session(capture)
    lineage = CodexLineage(output, session_id=session.session_id)
    lineage.replay(session)
    return lineage.export()


def build_demo(output: str | Path) -> Path:
    """Export the two-call allow/deny demo graph.

    The events are written out here rather than read from a capture so the demo runs with no Codex
    installed. It goes through the same :meth:`CodexLineage.replay` path a real capture does, so the
    graph shape is the replay's, not a second hand-built one.
    """
    from eqty_lineage.codex.capture import CaptureRecord

    records = [
        CaptureRecord({"hook_event_name": "SessionStart", "session_id": "codex-demo", "model": "demo"}),
        CaptureRecord(
            {"hook_event_name": "UserPromptSubmit", "prompt": "Run one safe command; block the forbidden write."}
        ),
        CaptureRecord(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_use_id": "exec-allowed",
                "tool_input": {"command": "printf 'allowed\\n'"},
            }
        ),
        CaptureRecord(
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_use_id": "exec-allowed",
                "tool_response": {"exit_code": 0},
            }
        ),
        CaptureRecord(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_use_id": "exec-denied",
                "tool_input": {"command": "touch forbidden.txt"},
            },
            {"decision": DENY, "decision_reason": "outside task scope"},
        ),
        CaptureRecord({"hook_event_name": "SessionEnd", "session_id": "codex-demo"}),
    ]
    session = normalize(records)
    lineage = CodexLineage(output, session_id=session.session_id)
    lineage.replay(session)
    return lineage.export()


__all__ = ["ALLOW", "DENY", "UNKNOWN", "CodexLineage", "build_demo", "replay_capture"]
