"""A DeepAgents research agent instrumented with the eqty_sdk through callbacks.

The agent plans with ``write_todos``, delegates to a ``librarian`` subagent through ``task``, writes and
revises files in the virtual filesystem, and reads them back. All EQTY registration happens through one
``EqtyDeepAgentsHandler`` passed in the callbacks config -- no node, tool or subagent is decorated.

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

from eqty_lineage.deepagents import EqtyDeepAgentsHandler

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
    _call("write_file", "s1", file_path="/notes.md", content=NOTES),
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


def build_agent(live: bool) -> Any:
    if live:
        from langchain_openai import ChatOpenAI

        model: Any = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), max_tokens=2048)
    else:
        model = ScriptedModel(messages=iter(SCRIPT))

    return create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        subagents=[LIBRARIAN],
        middleware=[TodoListMiddleware()],
        name="research-agent",
    )


def summarize(handler: EqtyDeepAgentsHandler, result: Dict[str, Any]) -> str:
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

    handler = EqtyDeepAgentsHandler(verbose=True)
    result = build_agent(args.live).invoke(
        {"messages": [HumanMessage(args.question)]},
        config={"callbacks": [handler], "recursion_limit": 80},
    )

    print(result["messages"][-1].content)
    print()
    print(summarize(handler, result))

    manifest.parent.mkdir(parents=True, exist_ok=True)
    cfg.get_default_context().export(manifest)
    print(f"\nmanifest: {manifest}")


if __name__ == "__main__":
    main()
