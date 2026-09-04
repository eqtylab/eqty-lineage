"""vNIM integrity-manifest behavior layered on the copied LangChain model."""

from uuid import uuid4

import httpx
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk


def test_model_keeps_upstream_fields_and_enables_response_headers():
    from eqty_lineage.openai import ChatEqtyOpenAI
    from langchain_openai import ChatOpenAI as UpstreamChatOpenAI

    model = ChatEqtyOpenAI(model="test", api_key="test", include_response_headers=False)

    assert UpstreamChatOpenAI.model_fields.keys() <= ChatEqtyOpenAI.model_fields.keys()
    assert model.include_response_headers is True
    assert model.streaming is True
    assert "manifest_base_url" in ChatEqtyOpenAI.model_fields


def test_stream_attaches_manifest_after_the_complete_upstream_stream(monkeypatch):
    import eqty_lineage.openai.chat_models.vnim as module
    from eqty_lineage.openai import ChatEqtyOpenAI

    request_id = uuid4()
    first = ChatGenerationChunk(
        message=AIMessageChunk(
            content="first",
            response_metadata={"headers": {"x-eqty-request-id": str(request_id)}},
        )
    )
    final = ChatGenerationChunk(message=AIMessageChunk(content="final"))

    def upstream_stream(self, *args, **kwargs):
        yield first
        yield final

    requests = []
    monkeypatch.setattr(module.ChatOpenAI, "_stream", upstream_stream)
    monkeypatch.setattr(
        module.httpx,
        "get",
        lambda url, timeout: requests.append((url, timeout))
        or httpx.Response(200, json={"statements": {"one": {}}, "blobs": {"two": "value"}}),
    )

    model = ChatEqtyOpenAI(
        model="test",
        api_key="test",
        manifest_base_url="http://127.0.0.1:8000/",
    )
    chunks = list(model.stream("hello"))

    assert [chunk.text for chunk in chunks] == ["first", "final"]
    assert requests == [
        (f"http://127.0.0.1:8000/integrity/manifest/{request_id}", 10.0)
    ]
    assert chunks[-1].response_metadata["eqty_request_id"] == str(request_id)
    assert chunks[-1].response_metadata["eqty_integrity_manifest"] == {
        "statements": {"one": {}},
        "blobs": {"two": "value"},
    }
    assert model.last_integrity_result is not None
    assert model.last_integrity_result.request_id == request_id


def test_manifest_fetch_retries_only_not_found(monkeypatch):
    import eqty_lineage.openai.chat_models.vnim as module
    from eqty_lineage.openai import ChatEqtyOpenAI

    responses = iter(
        [
            httpx.Response(404),
            httpx.Response(200, json={"statements": [], "blobs": {}}),
        ]
    )
    sleeps = []
    monkeypatch.setattr(module.httpx, "get", lambda url, timeout: next(responses))
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    result = ChatEqtyOpenAI(
        model="test",
        api_key="test",
        manifest_base_url="http://127.0.0.1:8000",
    )._fetch_integrity_manifest(uuid4())

    assert result.manifest == {"statements": [], "blobs": {}}
    assert result.attempts == 2
    assert sleeps == [3.0]


def test_request_id_can_be_read_from_generation_info():
    from eqty_lineage.openai import ChatEqtyOpenAI

    request_id = uuid4()
    chunk = ChatGenerationChunk(
        message=AIMessageChunk(content=""),
        generation_info={"headers": {"X-EQTY-Request-ID": str(request_id)}},
    )

    assert ChatEqtyOpenAI._eqty_request_id_from_chunk(chunk) == (request_id, None)


def test_final_request_payload_is_retained_without_exposing_internal_mutation(monkeypatch):
    import eqty_lineage.openai.chat_models.vnim as module
    from eqty_lineage.openai import ChatEqtyOpenAI

    payload = {"model": "test", "messages": [{"role": "user", "content": "hello"}], "stream": True}
    monkeypatch.setattr(module.ChatOpenAI, "_get_request_payload", lambda *args, **kwargs: payload)
    model = ChatEqtyOpenAI(model="test", api_key="test")

    assert model._get_request_payload("input") == payload
    exposed = model.last_request_payload
    assert exposed == payload
    exposed["model"] = "changed"
    assert model.last_request_payload == payload
