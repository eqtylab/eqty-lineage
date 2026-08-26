"""Shapes the handler used to choke on: absent state updates, nesting, Commands, failures, metadata.

Each of these raised inside a callback, which LangChain logs and swallows, so the only visible symptom
was a statement quietly missing from the manifest.
"""

from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph


class S(TypedDict):
    trail: Annotated[list, lambda a, b: a + b]


def _single_node_graph(fn, name="worker"):
    graph = StateGraph(S)
    graph.add_node(name, fn)
    graph.add_edge(START, name)
    graph.add_edge(name, END)
    return graph.compile()


# --------------------------------------------------------------------- D1 ----
def test_node_returning_none_still_produces_a_statement(recording_handler):
    """LangChain middleware nodes return None for 'no state update'; the SDK cannot hash None."""
    app = _single_node_graph(lambda state: None, name="noop")
    app.invoke({"trail": []}, config={"callbacks": [recording_handler]})

    names = [name for name, _, _, _ in recording_handler.computations]
    assert "noop" in names, "a node that made no state update still ran and must be recorded"
    assert recording_handler.outputs_of("noop"), "the computation needs an output to chain from"


def test_none_updates_do_not_converge_on_one_entity(recording_handler):
    """Two different nodes returning None must not share an output asset."""
    graph = StateGraph(S)
    graph.add_node("first", lambda state: None)
    graph.add_node("second", lambda state: None)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    graph.compile().invoke({"trail": []}, config={"callbacks": [recording_handler]})

    first = set(recording_handler.outputs_of("first"))
    second = set(recording_handler.outputs_of("second"))
    assert first and second
    assert not (first & second), "distinct nodes must not collapse onto one 'empty' entity"


# --------------------------------------------------------------------- D2 ----
def test_model_called_inside_a_tool(recording_handler):
    """The enclosing run is a tool, which has no node state -- this used to raise KeyError('state_in')."""
    inner = GenericFakeChatModel(messages=iter([AIMessage("inner answer")]))

    @tool
    def summarize(text: str) -> str:
        """Summarize text by calling a model from inside the tool."""
        return inner.invoke(text).content

    def worker(state: S) -> dict:
        return {"trail": [summarize.invoke({"text": "hello"})]}

    result = _single_node_graph(worker).invoke({"trail": []}, config={"callbacks": [recording_handler]})

    assert result["trail"] == ["inner answer"]
    kinds = {kind for _, kind, _, _ in recording_handler.computations}
    assert "chat_model" in kinds, "the nested model call was dropped"
    assert "tool" in kinds


def test_nested_model_output_reaches_the_enclosing_tool(recording_handler):
    inner = GenericFakeChatModel(messages=iter([AIMessage("inner answer")]))

    @tool
    def summarize(text: str) -> str:
        """Summarize text by calling a model from inside the tool."""
        return inner.invoke(text).content

    def worker(state: S) -> dict:
        return {"trail": [summarize.invoke({"text": "hello"})]}

    _single_node_graph(worker).invoke({"trail": []}, config={"callbacks": [recording_handler]})

    model_outputs = {o for _, kind, _, outs in recording_handler.computations if kind == "chat_model" for o in outs}
    tool_inputs = {i for _, kind, ins, _ in recording_handler.computations if kind == "tool" for i in ins}
    assert model_outputs, "no model output recorded"
    assert model_outputs <= tool_inputs, "the tool's result must derive from the model call nested inside it"


# --------------------------------------------------------------------- D7 ----
def test_failing_tool_is_recorded(recording_handler):
    @tool
    def explode(x: str) -> str:
        """Always fails."""
        raise RuntimeError("boom")

    def worker(state: S) -> dict:
        try:
            explode.invoke({"x": "1"})
        except RuntimeError:
            pass
        return {"trail": ["survived"]}

    _single_node_graph(worker).invoke({"trail": []}, config={"callbacks": [recording_handler]})

    kinds = [kind for _, kind, _, _ in recording_handler.computations]
    assert "tool_error" in kinds, "a failed tool must leave a record, not vanish"


def test_failing_node_is_recorded(recording_handler):
    def boom(state: S) -> dict:
        raise ValueError("node failed")

    with pytest.raises(ValueError):
        _single_node_graph(boom, name="boom").invoke({"trail": []}, config={"callbacks": [recording_handler]})

    kinds = [kind for _, kind, _, _ in recording_handler.computations]
    assert "graph_node_error" in kinds


# --------------------------------------------------------------------- D8 ----
def test_framework_is_langgraph_for_a_state_graph(recording_handler):
    _single_node_graph(lambda state: {"trail": ["x"]}).invoke({"trail": []}, config={"callbacks": [recording_handler]})
    assert set(recording_handler.frameworks) == {"langgraph"}


def test_framework_defaults_to_langchain_without_a_graph(recording_handler):
    """A plain runnable is LangChain, not LangGraph."""
    from langchain_core.runnables import RunnableLambda

    RunnableLambda(lambda x: x + 1).invoke(1, config={"callbacks": [recording_handler]})
    assert recording_handler.computations, "the root runnable should still be recorded"
    assert set(recording_handler.frameworks) == {"langchain"}, "a plain runnable is not LangGraph"


# --------------------------------------------------------------------- D4 ----
def test_command_tool_output_keeps_its_state_update(recording_handler):
    """A Command carries the state update; str() would throw it away."""
    from eqty_lineage.langchain import _to_jsonable
    from langgraph.types import Command

    payload = _to_jsonable(Command(update={"files": {"/a.md": "hello"}}))
    assert isinstance(payload, dict), f"Command was flattened to {type(payload).__name__}"
    assert payload["command"]["update"]["files"]["/a.md"] == "hello"


def test_command_paths_are_still_collected(tmp_path):
    """Paths nested inside a Command's update must reach the extractor hook, key path intact."""
    from pathlib import Path

    from eqty_lineage.langchain import UNCLAIMED, _to_jsonable
    from langgraph.types import Command

    target = tmp_path / "written.md"
    target.write_text("content")

    seen: list[Any] = []

    def collect(key_path, value):
        if isinstance(value, Path):
            seen.append((key_path, value))
        return UNCLAIMED

    _to_jsonable(Command(update={"report": target}), on_value=collect)
    assert seen == [(("update", "report"), target)]


# ----------------------------------------------------- model identity ----
def test_model_named_from_its_class_when_params_are_bare(recording_handler):
    """GenericFakeChatModel declares no model name; the class name beats 'unknown-model'."""
    from langchain_core.runnables import RunnableLambda

    model = GenericFakeChatModel(messages=iter([AIMessage("hi")]))
    RunnableLambda(lambda _: model.invoke("x")).invoke(1, config={"callbacks": [recording_handler]})

    names = [name for name, kind, _, _ in recording_handler.computations if kind == "chat_model"]
    assert names == ["GenericFakeChatModel"], f"got {names}"


def test_model_identity_prefers_invocation_params(recording_handler):
    """A provider that names itself in invocation_params wins over every fallback."""
    resolve = recording_handler._model_identity

    assert resolve({"name": "ChatOpenAI"}, {"model": "gpt-4o-mini"}, {"ls_model_name": "other"}) == (
        "gpt-4o-mini",
        "unknown",
    )
    assert resolve({"name": "ChatOpenAI"}, {"model_name": "gpt-4o-mini"}, None)[0] == "gpt-4o-mini"


def test_model_identity_falls_back_through_metadata_then_class(recording_handler):
    resolve = recording_handler._model_identity

    # ls_model_name is LangSmith's standardised field, set from _get_ls_params
    assert resolve({"name": "ChatFoo"}, {}, {"ls_model_name": "foo-1", "ls_provider": "foo"}) == ("foo-1", "foo")
    # nothing names the model, but the runnable's class does
    assert resolve({"name": "ChatFoo"}, {}, {})[0] == "ChatFoo"
    # nothing at all
    assert resolve(None, {}, None) == ("unknown-model", "unknown")


def test_provider_prefers_ls_provider_over_type(recording_handler):
    resolve = recording_handler._model_identity

    assert resolve(None, {"_type": "openai-chat"}, {"ls_provider": "openai"})[1] == "openai"
    assert resolve(None, {"_type": "openai-chat"}, {})[1] == "openai-chat"


def test_a_model_call_does_not_rename_the_framework(recording_handler):
    """langchain-core tags every model run `langchain_chat_model`; that names a component, not a harness."""
    from langchain_core.runnables import RunnableLambda

    model = GenericFakeChatModel(messages=iter([AIMessage("hi")]))
    RunnableLambda(lambda _: model.invoke("x")).invoke(1, config={"callbacks": [recording_handler]})

    assert set(recording_handler.frameworks) == {"langchain"}
