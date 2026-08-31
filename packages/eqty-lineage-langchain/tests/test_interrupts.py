"""A run that pauses for a human has not failed, and must not be recorded as though it had.

LangGraph signals control flow by raising: `interrupt()` unwinds the graph with a `GraphInterrupt`, and
`Command(goto=...)` across a subgraph boundary with a `ParentCommand`. Both reach `on_chain_error` looking
exactly like a crash, so the handler recorded a `*_error` computation for a node that had merely suspended
and would go on to finish successfully on resume. Every human-in-the-loop approval produced a manifest
asserting a failure that never happened.
"""

from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class S(TypedDict):
    trail: Annotated[list, lambda a, b: a + b]


def _approval_graph():
    def ask(state):
        decision = interrupt({"question": "may I?"})
        return {"trail": [f"approved:{decision}"]}

    graph = StateGraph(S)
    graph.add_node("ask", ask)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    return graph.compile(checkpointer=MemorySaver())


def _kinds(handler):
    return [kind for _, kind, _, _ in handler.computations]


def test_a_pause_for_approval_is_not_a_failure(recording_handler):
    """The pause itself: nothing may be recorded as an error."""
    app = _approval_graph()
    config = {"configurable": {"thread_id": "t1"}, "callbacks": [recording_handler]}

    result = app.invoke({"trail": []}, config=config)

    assert "__interrupt__" in result, "the run really did pause"
    assert not [k for k in _kinds(recording_handler) if k.endswith("error")], (
        f"a suspended run was recorded as failed: {recording_handler.computations}"
    )


def test_a_resumed_run_records_the_node_that_paused(recording_handler):
    """And the work is not lost: the turn that finishes it records it normally."""
    app = _approval_graph()
    config = {"configurable": {"thread_id": "t2"}, "callbacks": [recording_handler]}

    app.invoke({"trail": []}, config=config)
    result = app.invoke(Command(resume="yes"), config=config)

    assert result["trail"] == ["approved:yes"]
    names = [name for name, _, _, _ in recording_handler.computations]
    assert "ask" in names, "the resumed node is recorded"
    assert not [k for k in _kinds(recording_handler) if k.endswith("error")]


def test_a_real_failure_is_still_recorded(recording_handler):
    """The guard must not swallow ordinary exceptions -- only LangGraph's control-flow signals."""

    def boom(state):
        raise RuntimeError("actually broken")

    graph = StateGraph(S)
    graph.add_node("boom", boom)
    graph.add_edge(START, "boom")
    graph.add_edge("boom", END)

    try:
        graph.compile().invoke({"trail": []}, config={"callbacks": [recording_handler]})
    except RuntimeError:
        pass

    assert [k for k in _kinds(recording_handler) if k.endswith("error")], "a crash is still a failure"
