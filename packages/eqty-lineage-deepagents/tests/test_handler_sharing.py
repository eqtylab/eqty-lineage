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


def test_on_tool_start_never_releases_the_lock_mid_call(recording_handler, monkeypatch):
    """A sibling tool call must not be able to act between registering arguments and reading the registry.

    LangGraph turns the tool calls of one AI message into concurrent tasks on a thread pool, in plain
    `.invoke()` as well as under `ainvoke`. When this callback took the lock twice, a `write_file`
    completing in the gap replaced `_file_latest` for the path, and the `read_file` that opened before it
    linked the writer's version as what it had read -- attesting the model reasoned over bytes it never saw.

    The race itself cannot be scheduled deterministically, so what is asserted is the invariant that
    removes it: the lock is held for the whole body. `_normalize_path` runs in what used to be the gap, so
    a probe there sees the lock free exactly when the bug is present. The probe runs on another thread
    because the lock is reentrant and would always be acquirable from this one.
    """
    import threading
    from uuid import uuid4

    import eqty_lineage.deepagents as handler_module

    original = handler_module._normalize_path
    acquired_by_another_thread = []

    def probe(path):
        def attempt():
            got = recording_handler._lock.acquire(blocking=False)
            if got:
                recording_handler._lock.release()
            acquired_by_another_thread.append(got)

        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join(timeout=5)
        return original(path)

    monkeypatch.setattr(handler_module, "_normalize_path", probe)

    recording_handler.on_tool_start(
        {"name": "read_file"},
        "",
        run_id=uuid4(),
        inputs={"file_path": "/r.md"},
    )

    assert acquired_by_another_thread == [False], (
        "the lock was free while on_tool_start was between registering the call and reading the "
        "version registry, which is the window a concurrent sibling write lands in"
    )
