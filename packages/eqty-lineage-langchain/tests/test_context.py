"""LangGraph threads are isolated beneath the SDK's configured root context."""


def test_thread_id_creates_and_reuses_a_child_context(sdk, monkeypatch):
    import eqty_lineage.langchain as mod
    from eqty_lineage.langchain import EqtyCallbackHandler
    from eqty_sdk import Context

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
