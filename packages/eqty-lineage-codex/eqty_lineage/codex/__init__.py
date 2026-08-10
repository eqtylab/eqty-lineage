"""Small, signed Codex hook-to-graph adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eqty_sdk import Configuration, Context, Dataset, Guardrail, Prompt, Signer, Tool, init, set_active_signer
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
        allowed: bool,
        result: dict[str, Any] | None = None,
        reason: str = "policy decision",
    ) -> None:
        with graph_context(self.context):
            tool_asset = Tool.from_object({"name": name}, name=name)
            request = Dataset.from_object(tool_input, name=f"{name} input")
            decision = Guardrail.from_object(
                {"decision": "allow" if allowed else "deny", "reason": reason},
                name=f"{'allow' if allowed else 'deny'}: {name}",
            )
            outputs = [decision.cid]
            if allowed:
                outputs.append(Dataset.from_object(result or {}, name=f"{name} result").cid)
            inputs = [asset.cid for asset in (self._prompt, tool_asset, request) if asset is not None]
            activity = add_computation_statement(inputs=inputs, outputs=outputs, context=self.context)[0]
            Metadata(
                name=f"Codex {name}",
                computation_type="tool",
                framework="codex",
                session_id=self.session_id,
                decision="allow" if allowed else "deny",
            ).create_statement(activity, None, self.context)

    def export(self) -> Path:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.context.export(self.output)
        return self.output


def build_demo(output: str | Path) -> Path:
    lineage = CodexLineage(output)
    lineage.prompt("Run one safe command; block the forbidden write.")
    lineage.tool("Bash", {"command": "printf 'allowed\\n'"}, allowed=True, result={"exit_code": 0})
    lineage.tool("Bash", {"command": "touch forbidden.txt"}, allowed=False, reason="outside task scope")
    return lineage.export()


__all__ = ["CodexLineage", "build_demo"]
