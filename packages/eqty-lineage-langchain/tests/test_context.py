"""LangGraph threads are isolated beneath the SDK's configured root context."""

from types import SimpleNamespace
from typing import TypedDict
from uuid import uuid4


def test_thread_id_creates_and_reuses_a_child_context(sdk, monkeypatch):
    import eqty_lineage.langchain as mod
    from eqty_lineage.langchain import EqtyCallbackHandler

    handler = EqtyCallbackHandler()
    handler._activate_thread_context({"thread_id": "one"}, agent_name="Travel Assistant")
    session_context = handler._context
    assert session_context != sdk
    assert session_context.name.startswith("Travel Assistant: ")
    assert handler.context == session_context

    asset, _, _ = handler._register_state({"message": "hello"}, "input", "test input")
    assert asset._ctx == session_context
    assert asset._ctx != sdk

    computation_contexts = []
    metadata_contexts = []

    def record_computation(**kwargs):
        computation_contexts.append(kwargs["context"])
        return [asset.cid]

    def record_metadata(self, subject_cid, skip_proof, context):
        metadata_contexts.append(context)
        return []

    monkeypatch.setattr(mod, "add_computation_statement", record_computation)
    monkeypatch.setattr(mod.Metadata, "create_statement", record_metadata)

    handler._finalize("work", "graph_node", [asset.cid], [asset.cid])

    assert computation_contexts == [session_context]
    assert metadata_contexts == [session_context]

    another_handler = EqtyCallbackHandler()
    another_handler._activate_thread_context({"thread_id": "one"}, agent_name="Travel Assistant")
    assert another_handler._context == session_context

    another_handler._activate_thread_context({"thread_id": "two"}, agent_name="Travel Assistant")
    assert another_handler._context != session_context


def test_integrity_service_registers_active_context(sdk, monkeypatch):
    import eqty_lineage.langchain as mod
    from eqty_lineage.langchain import EqtyCallbackHandler

    created = []
    registered = []
    service = object()

    def new_service(url):
        created.append(url)
        return service

    def register(context, configured_service):
        registered.append((context, configured_service))

    monkeypatch.setattr(mod.Service, "new", new_service)
    monkeypatch.setattr(type(sdk), "register", register)

    handler = EqtyCallbackHandler(
        integrity_service_url="https://integrity.example.test",
    )
    handler._activate_thread_context({"thread_id": "one"}, agent_name="Travel Assistant")
    handler._register_context()

    assert created == ["https://integrity.example.test"]
    assert registered == [(handler.context, service)]


def test_registration_runs_at_the_end_of_each_root_graph_call(sdk, monkeypatch):
    from eqty_lineage.langchain import EqtyCallbackHandler

    handler = EqtyCallbackHandler()
    run_id = uuid4()
    handler._runs[run_id] = {
        "name": "Travel Assistant",
        "kind": "graph",
        "parent": None,
        "step": 0,
        "state_in": "in",
        "inputs": [],
        "child_outputs": [],
        "context": "thread-context",
    }
    monkeypatch.setattr(handler, "_register_state", lambda *args, **kwargs: (SimpleNamespace(cid="out"), [], []))
    monkeypatch.setattr(handler, "_finalize", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(handler, "_register_context", lambda context: calls.append(context))

    handler.on_chain_end({}, run_id=run_id)

    assert calls == ["thread-context"]


def test_nested_callback_without_thread_id_keeps_the_graph_context(sdk):
    from eqty_lineage.langchain import EqtyCallbackHandler

    handler = EqtyCallbackHandler()
    handler._activate_thread_context({"thread_id": "one"}, agent_name="Travel Assistant")
    thread_context = handler.context

    handler._activate_thread_context(None)

    assert handler.context == thread_context


def test_nested_thread_metadata_updates_the_context_registered_by_the_root_run(sdk):
    from eqty_lineage.langchain import EqtyCallbackHandler

    handler = EqtyCallbackHandler()
    root_run_id = uuid4()
    handler._runs[root_run_id] = {"parent": None, "context": sdk}

    handler._activate_thread_context({"thread_id": "one"}, agent_name="Travel Assistant")
    handler._bind_context_to_root_run(root_run_id)

    assert handler._runs[root_run_id]["context"] == handler.context
    assert handler.context != sdk


def test_langgraph_invocation_registers_its_thread_child_context(sdk, monkeypatch):
    from eqty_lineage.langchain import EqtyCallbackHandler
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        value: int

    def increment(state: State) -> dict[str, int]:
        return {"value": state["value"] + 1}

    graph = StateGraph(State)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    app = graph.compile(checkpointer=MemorySaver())

    handler = EqtyCallbackHandler()
    registered = []
    monkeypatch.setattr(handler, "_register_context", lambda context: registered.append(context))

    app.invoke(
        {"value": 1},
        config={
            "callbacks": [handler],
            "configurable": {"thread_id": "session-one"},
            "run_name": "Travel Assistant",
        },
    )

    assert registered == [handler.context]
    assert handler.context != sdk
    assert handler.context.parent == sdk.id
