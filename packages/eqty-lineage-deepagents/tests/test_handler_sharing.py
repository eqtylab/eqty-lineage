"""One handler must observe one run at a time, and say so when it is not.

The file and plan registries are keyed by path, or by nothing at all, because they describe a single run's
filesystem. Share a handler between two runs in flight and those keys collide: one run's write chains off
the other's version and the manifest asserts a revision between runs that share nothing. The graph still
looks well-formed, which is exactly why it needs to be noisy.
"""

import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage

from scripted import call, deep_agent

LOGGER = "eqty.deepagents"


def _script(tag: str):
    return [
        call("write_file", "w", file_path="/report.md", content=f"content from run {tag}\n"),
        AIMessage(content="done"),
    ]


def _run(agent, handler):
    return agent.invoke(
        {"messages": [HumanMessage("go")]},
        config={"callbacks": [handler], "recursion_limit": 40},
    )


def test_concurrent_runs_on_one_handler_are_flagged(recording_handler, caplog):
    """The failure this catches: run B's write recorded as a revision of run A's file."""

    async def both():
        await asyncio.gather(
            deep_agent(_script("A")).ainvoke(
                {"messages": [HumanMessage("go")]},
                config={"callbacks": [recording_handler], "recursion_limit": 40},
            ),
            deep_agent(_script("B")).ainvoke(
                {"messages": [HumanMessage("go")]},
                config={"callbacks": [recording_handler], "recursion_limit": 40},
            ),
        )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        asyncio.run(both())

    warnings = [r for r in caplog.records if "runs at once" in r.message]
    assert warnings, "sharing a handler between concurrent runs must not be silent"


def test_sequential_reuse_is_not_flagged(recording_handler, caplog):
    """Reusing a handler across the turns of one conversation is deliberate, not a mistake: it is what
    chains a file written in an early turn to an edit in a later one."""
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _run(deep_agent(_script("A")), recording_handler)
        _run(deep_agent(_script("B")), recording_handler)

    assert [r for r in caplog.records if "runs at once" in r.message] == []
    assert recording_handler._open_roots == [], "each root is released when it ends"


def test_a_failed_run_releases_its_root(recording_handler, caplog):
    """A root that raises must not leave the handler reporting a collision on every run after it."""

    def explode(state):
        raise RuntimeError("boom")

    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class S(TypedDict):
        messages: list

    graph = StateGraph(S)
    graph.add_node("boom", explode)
    graph.add_edge(START, "boom")
    graph.add_edge("boom", END)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        try:
            graph.compile().invoke({"messages": []}, config={"callbacks": [recording_handler]})
        except RuntimeError:
            pass
        assert recording_handler._open_roots == []
        _run(deep_agent(_script("A")), recording_handler)

    assert [r for r in caplog.records if "runs at once" in r.message] == []
