"""A nested run that names a different agent is a subagent boundary.

DeepAgents spawns a subagent through the `task` tool, and the subagent's root run carries its own name
while inheriting `langgraph_node` from the tool that spawned it. It therefore matches neither the node
rule nor the root rule, and used to be skipped entirely -- so the delegated work was recorded but linked
to nothing that asked for it.

The rule here is LangChain's own, from `langchain.agents._subagent_transformer`: a boundary is a nested
run whose `lc_agent_name` differs from its parent's. These tests reproduce that shape with plain
LangGraph, so they need neither `langchain>=1` nor `deepagents` installed.
"""

from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph


class S(TypedDict):
    trail: Annotated[list, lambda a, b: a + b]


def _inner_graph(label: str):
    graph = StateGraph(S)
    graph.add_node("work", lambda state: {"trail": [f"{label}-worked"]})
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile()


def _outer_graph(delegate):
    graph = StateGraph(S)
    graph.add_node("delegate", delegate)
    graph.add_edge(START, "delegate")
    graph.add_edge("delegate", END)
    return graph.compile()


def test_nested_run_naming_a_different_agent_is_recorded(recording_handler):
    inner = _inner_graph("researcher")

    def delegate(state: S) -> dict:
        out = inner.invoke({"trail": []}, config={"metadata": {"lc_agent_name": "researcher"}})
        return {"trail": [f"outer saw: {out['trail']}"]}

    _outer_graph(delegate).invoke({"trail": []}, config={"callbacks": [recording_handler]})

    agents = [(name, kind) for name, kind, _, _ in recording_handler.computations if kind == "agent"]
    assert agents == [("researcher", "agent")], f"got {agents}"


def test_a_subgraph_inheriting_the_name_is_not_a_boundary(recording_handler):
    """Plain subgraphs inherit the parent's lc_agent_name; only a *different* one is a delegation."""
    inner = _inner_graph("same")

    def delegate(state: S) -> dict:
        out = inner.invoke({"trail": []}, config={"metadata": {"lc_agent_name": "outer-agent"}})
        return {"trail": out["trail"]}

    _outer_graph(delegate).invoke(
        {"trail": []},
        config={"callbacks": [recording_handler], "metadata": {"lc_agent_name": "outer-agent"}},
    )

    kinds = [kind for _, kind, _, _ in recording_handler.computations]
    assert "agent" not in kinds, "a subgraph under the same agent is not a subagent"


def test_subagent_result_feeds_the_tool_that_spawned_it(recording_handler):
    """The `task`-tool shape: a tool whose whole job is to run another agent."""
    inner = _inner_graph("researcher")

    @tool
    def task(description: str) -> str:
        """Delegate to a subagent."""
        out = inner.invoke({"trail": []}, config={"metadata": {"lc_agent_name": "researcher"}})
        # deliberately not the subagent's payload verbatim: identical payloads share a CID, and a
        # collision between the subagent's output and the caller's would mask an orphan rather than
        # prove there isn't one
        return f"delegated -> {','.join(out['trail'])}"

    def delegate(state: S) -> dict:
        return {"trail": [f"outer saw: {task.invoke({'description': 'research'})}"]}

    _outer_graph(delegate).invoke({"trail": []}, config={"callbacks": [recording_handler]})

    agent_out = {o for name, kind, _, outs in recording_handler.computations if kind == "agent" for o in outs}
    task_in = set(recording_handler.inputs_of("task"))
    assert agent_out, "the subagent produced nothing"
    assert agent_out <= task_in, "the tool's result must derive from the agent it delegated to"


def test_no_subagent_output_is_orphaned(recording_handler):
    inner = _inner_graph("researcher")

    @tool
    def task(description: str) -> str:
        """Delegate to a subagent."""
        out = inner.invoke({"trail": []}, config={"metadata": {"lc_agent_name": "researcher"}})
        # deliberately not the subagent's payload verbatim: identical payloads share a CID, and a
        # collision between the subagent's output and the caller's would mask an orphan rather than
        # prove there isn't one
        return f"delegated -> {','.join(out['trail'])}"

    def delegate(state: S) -> dict:
        return {"trail": [f"outer saw: {task.invoke({'description': 'research'})}"]}

    _outer_graph(delegate).invoke({"trail": []}, config={"callbacks": [recording_handler]})

    consumed = {i for _, _, ins, _ in recording_handler.computations for i in ins}
    orphans = [
        (name, out)
        for name, kind, _, outs in recording_handler.computations
        if kind != "graph"
        for out in outs
        if out not in consumed
    ]
    assert orphans == [], f"outputs consumed by nothing: {orphans}"
