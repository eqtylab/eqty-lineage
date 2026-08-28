"""A DeepAgents research agent instrumented with the eqty_sdk through callbacks.

The agent plans with ``write_todos``, delegates to a ``librarian`` subagent through ``task``, looks a term
up in a local knowledge base, and writes, revises and reads files in the virtual filesystem. All EQTY
registration happens through one ``EqtyDeepAgentsHandler`` passed in the callbacks config -- no node or
subagent is decorated.

Tools are the exception, because a callback only ever carries a tool's *name*. ``search_knowledge_base`` is
decorated with ``@eqty_tool`` at definition time; the built-in belt -- ``write_file``, ``edit_file``,
``task`` and the rest, which DeepAgents' middleware builds rather than us -- is passed through the same
function once the agent is compiled (see ``register_tool_sources``). Every Tool asset in the manifest is
then content-addressed to the code that ran, not to a name stub.

    uv run python examples/deepagents/research_agent.py            # scripted model, no API key needed
    uv run python examples/deepagents/research_agent.py --live "your question"

The model is scripted by default. That is not a shortcut around an API key: a manifest is only worth
reading against a run you can reproduce, and a live model would make each run's lineage differ for
reasons that have nothing to do with the handler. ``--live`` uses OpenAI and needs ``OPENAI_API_KEY``.

``TodoListMiddleware`` is passed explicitly because it is *not* part of the default deep agent stack --
it comes from ``langchain``, and a stock ``create_deep_agent`` has neither ``write_todos`` nor the
``todos`` state key.

Writes ``manifests/deep-agent.json`` and prints what the run recorded.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from deepagents import create_deep_agent
from eqty_sdk import Context, Signer, init, set_active_signer
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from eqty_lineage.deepagents import EqtyDeepAgentsHandler, eqty_tool
from eqty_lineage.langchain._tools import _registered_tool_sources

MANIFEST = Path("./manifests/deep-agent.json")

SYSTEM_PROMPT = (
    "You are a research assistant. Plan with write_todos, delegate lookups to the librarian subagent, "
    "keep your notes and drafts in files, and finish with a short report in /report.md."
)

LIBRARIAN = {
    "name": "librarian",
    "description": "Looks up a topic and writes what it found to a file.",
    "system_prompt": "You look things up and write concise notes to /notes.md. Do not editorialise.",
}

KNOWLEDGE_BASE = {
    "cid": (
        "A CID (content identifier) is a self-describing hash that names content rather than a location, "
        "so the same bytes have the same name wherever they are stored."
    ),
    "lineage": (
        "Data lineage links inputs, computations and outputs. EQTY records it as signed statements: data "
        "statements register assets by CID, computation statements link input CIDs to output CIDs."
    ),
}


@tool
@eqty_tool
def search_knowledge_base(query: str) -> str:
    """Search the local knowledge base for entries matching the query."""
    words = {w.strip("?.,!").lower() for w in query.split()}
    matches = {key: text for key, text in KNOWLEDGE_BASE.items() if key in words}
    if not matches:
        return f"No entries for '{query}'. Available: {', '.join(KNOWLEDGE_BASE)}."
    return "\n\n".join(f"[{key}] {text}" for key, text in matches.items())


NOTES = (
    "A CID is a self-describing hash: it names content rather than a location, so the same bytes have\n"
    "the same name wherever they are stored.\n"
)
DRAFT = "# CIDs\n\nA CID names content.\n"
FINAL = "# CIDs\n\nA CID names content rather than a location, which is what makes lineage verifiable.\n"


def _call(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "id": call_id, "args": args}])


SCRIPT: List[AIMessage] = [
    _call(
        "write_todos",
        "p1",
        todos=[
            {"content": "look up what a CID is", "status": "in_progress"},
            {"content": "draft the report", "status": "pending"},
        ],
    ),
    _call("task", "p2", description="Look up what a CID is.", subagent_type="librarian"),
    # the librarian's own turns
    _call("search_knowledge_base", "s1", query="cid"),
    _call("write_file", "s2", file_path="/notes.md", content=NOTES),
    AIMessage(content="Notes are in /notes.md."),
    # back in the research agent
    _call(
        "write_todos",
        "p3",
        todos=[
            {"content": "look up what a CID is", "status": "completed"},
            {"content": "draft the report", "status": "in_progress"},
        ],
    ),
    _call("read_file", "p4", file_path="/notes.md"),
    _call("write_file", "p5", file_path="/report.md", content=DRAFT),
    _call(
        "edit_file",
        "p6",
        file_path="/report.md",
        old_string="A CID names content.",
        new_string="A CID names content rather than a location, which is what makes lineage verifiable.",
    ),
    _call(
        "write_todos",
        "p7",
        todos=[
            {"content": "look up what a CID is", "status": "completed"},
            {"content": "draft the report", "status": "completed"},
        ],
    ),
    AIMessage(content="The report is in /report.md."),
]


class ScriptedModel(GenericFakeChatModel):
    """Replays the script above, and accepts the agent's tool belt without binding it."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def build_agent(live: bool) -> "tuple[Any, List[str]]":
    if live:
        from langchain_openai import ChatOpenAI

        model: Any = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), max_tokens=2048)
    else:
        model = ScriptedModel(messages=iter(SCRIPT))

    agent = create_deep_agent(
        model=model,
        tools=[search_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
        subagents=[LIBRARIAN],
        middleware=[TodoListMiddleware()],
        name="research-agent",
    )
    return agent, register_tool_sources(agent)


def register_tool_sources(agent: Any) -> List[str]:
    """Content-address the built-in tools to their implementations, the way ``@eqty_tool`` does ours.

    ``search_knowledge_base`` is decorated at definition time, which is the ordinary way to do this. The
    tools that do the interesting work here are not ours to decorate: ``write_file``, ``edit_file`` and
    ``task`` are built by DeepAgents' own middleware, and without their source each registers from a
    name/description stub -- so the manifest would record *that* a file was written but not by what code.
    Their source is perfectly readable, so the same function the decorator calls is applied to the belt
    the compiled agent assembled. Upgrade DeepAgents and these assets change, which is the point.

    Reaching into the compiled graph for the belt is the demo's own liberty, not something the handler
    does: it is the only place the assembled tools exist before the first call, and a name whose source
    cannot be read is simply left to fall back to its stub.
    """
    tools_node = getattr(agent.nodes.get("tools"), "bound", None)
    belt = getattr(tools_node, "tools_by_name", None) or {}
    for tool_obj in belt.values():
        eqty_tool(tool_obj)
    return sorted(name for name in belt if name in _registered_tool_sources)


def summarize(handler: EqtyDeepAgentsHandler, result: Dict[str, Any], sources: List[str]) -> str:
    files = sorted(result.get("files") or {})
    versions: Dict[str, int] = {}
    for path, _digest in handler._file_versions:
        versions[path] = versions.get(path, 0) + 1

    lines = [
        f"files:            {len(files)} ({', '.join(files) or 'none'})",
        f"file versions:    {sum(versions.values())} ({', '.join(f'{p} x{n}' for p, n in sorted(versions.items()))})",
        f"plan revisions:   {len(handler._todo_versions)}",
        f"agents:           {', '.join(sorted(name for name, _ in handler._agent_cids)) or 'none'}",
        f"skills:           {', '.join(sorted(name for name, _ in handler._skill_cids)) or 'none'}",
        f"system prompts:   {len(handler._system_prompt_cids)}",
        f"tool sources:     {len(sources)} ({', '.join(sources) or 'none'})",
    ]
    return "\n".join(lines)


def init_logger() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="(%(asctime)s) %(levelname)s - %(name)s %(funcName)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("eqty_sdk").setLevel(logging.INFO)
    logging.getLogger("eqty").setLevel(logging.INFO)


def init_sdk(fresh: bool):
    if fresh:
        # init() writes a .eqty_sdk store under the working directory, so two runs sharing one would
        # make the second appear to depend on the first
        os.chdir(tempfile.mkdtemp(prefix="deep-agent-"))
    ctx = Context.new("DeepAgents Research Agent")
    cfg = init(default_context=ctx).set_store_all_blobs(True)
    set_active_signer(Signer.new(name="deepagents_research_agent", _load_if_exists=True))
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EQTY-instrumented deep research agent.")
    parser.add_argument("question", nargs="?", default="What is a CID? Write me a short report.")
    parser.add_argument("--live", action="store_true", help="use OpenAI instead of the scripted model")
    parser.add_argument("--in-place", action="store_true", help="keep the .eqty_sdk store in the current directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_logger()
    manifest = MANIFEST.resolve()
    cfg = init_sdk(fresh=not args.in_place)

    # the belt this agent assembled, not the process-global registry: that dict is never cleared, so
    # reading it would credit this run with sources some other handler registered
    agent, sources = build_agent(args.live)

    handler = EqtyDeepAgentsHandler(verbose=True)
    result = agent.invoke(
        {"messages": [HumanMessage(args.question)]},
        config={"callbacks": [handler], "recursion_limit": 80},
    )

    print(result["messages"][-1].content)
    print()
    print(summarize(handler, result, sources))

    manifest.parent.mkdir(parents=True, exist_ok=True)
    cfg.get_default_context().export(manifest)
    print(f"\nmanifest: {manifest}")


if __name__ == "__main__":
    main()
