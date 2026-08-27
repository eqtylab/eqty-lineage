"""Nodes that LangGraph runs in the same superstep must all reach their successor.

LangGraph executes one superstep's nodes concurrently on a thread pool, in plain synchronous
``.invoke()`` as much as under ``ainvoke``. Keying a node's predecessor on "the last sibling to finish"
loses every branch but one, silently, because the callback that overwrote the entry raised nothing.
"""

import time
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph


class FanState(TypedDict):
    trail: Annotated[list, lambda a, b: a + b]


def _fan_out_graph():
    """START -> split -> {left, right} in one superstep -> join -> END."""

    def split(state: FanState) -> dict:
        return {"trail": ["split"]}

    def left(state: FanState) -> dict:
        time.sleep(0.05)
        return {"trail": ["left"]}

    def right(state: FanState) -> dict:
        time.sleep(0.05)
        return {"trail": ["right"]}

    def join(state: FanState) -> dict:
        return {"trail": ["join"]}

    graph = StateGraph(FanState)
    for name, fn in (("split", split), ("left", left), ("right", right), ("join", join)):
        graph.add_node(name, fn)
    graph.add_edge(START, "split")
    graph.add_edge("split", "left")
    graph.add_edge("split", "right")
    graph.add_edge("left", "join")
    graph.add_edge("right", "join")
    graph.add_edge("join", END)
    return graph.compile()


def test_join_consumes_every_parallel_branch(recording_handler):
    app = _fan_out_graph()
    result = app.invoke({"trail": []}, config={"callbacks": [recording_handler]})

    assert set(result["trail"]) == {"split", "left", "right", "join"}

    left_out = recording_handler.outputs_of("left")
    right_out = recording_handler.outputs_of("right")
    join_in = recording_handler.inputs_of("join")

    assert left_out and right_out, "both parallel branches should have produced an output state"
    # the regression: only whichever branch finished last used to survive into join's inputs
    assert set(left_out) <= set(join_in), "join lost the 'left' branch"
    assert set(right_out) <= set(join_in), "join lost the 'right' branch"


def test_no_node_output_is_orphaned(recording_handler):
    """Every node's output state must be consumed by something downstream."""
    app = _fan_out_graph()
    app.invoke({"trail": []}, config={"callbacks": [recording_handler]})

    all_inputs = {i for _, _, ins, _ in recording_handler.computations for i in ins}
    orphaned = [
        (name, out)
        for name, kind, _, outs in recording_handler.computations
        if kind == "graph_node"
        for out in outs
        if out not in all_inputs
    ]
    assert orphaned == [], f"outputs consumed by nothing: {orphaned}"


def test_sibling_bookkeeping_is_released(recording_handler):
    """A finished run must not leave its per-parent state behind."""
    app = _fan_out_graph()
    app.invoke({"trail": []}, config={"callbacks": [recording_handler]})

    assert recording_handler._runs == {}
    assert recording_handler._parents == {}
    # the root run's own entry is dropped when it ends; nothing should outlive the invocation
    assert recording_handler._sibling_outputs == {}
    assert recording_handler._fallback_steps == {}


@pytest.mark.parametrize("branches", [2, 4])
def test_wide_fan_out(recording_handler, branches):
    """A superstep of N parallel nodes feeds all N outputs forward, not one."""

    def make(i):
        def node(state: FanState) -> dict:
            time.sleep(0.02)
            return {"trail": [f"n{i}"]}

        return node

    graph = StateGraph(FanState)
    graph.add_node("start", lambda s: {"trail": ["start"]})
    graph.add_node("end", lambda s: {"trail": ["end"]})
    for i in range(branches):
        graph.add_node(f"n{i}", make(i))
        graph.add_edge("start", f"n{i}")
        graph.add_edge(f"n{i}", "end")
    graph.add_edge(START, "start")
    graph.add_edge("end", END)

    graph.compile().invoke({"trail": []}, config={"callbacks": [recording_handler]})

    end_in = set(recording_handler.inputs_of("end"))
    for i in range(branches):
        assert set(recording_handler.outputs_of(f"n{i}")) <= end_in, f"branch n{i} was dropped"


def test_untracked_chain_runs_do_not_accumulate(recording_handler):
    """Every chain run gets an _agent_names entry at start; most chain runs are never tracked.

    LangGraph emits far more internal runs (channel reads, task wrappers) than nodes, so releasing this
    only on the tracked path left the dict growing for the length of the session.
    """
    from langchain_core.runnables import RunnableLambda

    # a deliberately nested chain, so plenty of runs start and are never tracked
    chain = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2) | RunnableLambda(str)
    for _ in range(5):
        chain.invoke(1, config={"callbacks": [recording_handler]})

    assert recording_handler._agent_names == {}, recording_handler._agent_names
    assert recording_handler._runs == {}
    assert recording_handler._parents == {}
