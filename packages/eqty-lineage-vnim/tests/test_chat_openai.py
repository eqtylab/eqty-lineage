"""vNIM integrity-manifest behavior layered on LangChain's ChatOpenAI."""

from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def test_model_keeps_upstream_fields_and_enables_response_headers(caplog):
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI
    from langchain_openai import ChatOpenAI as UpstreamChatOpenAI

    with caplog.at_level("WARNING"):
        model = ChatEqtyVnimOpenAI(model="test", api_key="test", include_response_headers=False)

    assert UpstreamChatOpenAI.model_fields.keys() <= ChatEqtyVnimOpenAI.model_fields.keys()
    assert model.include_response_headers is True
    assert model.streaming is True
    assert "manifest_base_url" in ChatEqtyVnimOpenAI.model_fields
    assert "include_response_headers=False is ignored" in caplog.text


def test_stream_attaches_manifest_after_the_complete_upstream_stream(monkeypatch):
    import eqty_lineage.vnim.chat_models.vnim as module
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI

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
        lambda url, timeout: (
            requests.append((url, timeout))
            or httpx.Response(200, json={"statements": {"one": {}}, "blobs": {"two": "value"}})
        ),
    )

    model = ChatEqtyVnimOpenAI(
        model="test",
        api_key="test",
        manifest_base_url="http://127.0.0.1:8000/",
    )
    chunks = list(model.stream("hello"))

    assert [chunk.text for chunk in chunks] == ["first", "final"]
    assert requests == [(f"http://127.0.0.1:8000/integrity/manifest/{request_id}", 10.0)]
    assert chunks[-1].response_metadata["eqty_request_id"] == str(request_id)
    assert chunks[-1].response_metadata["eqty_integrity_manifest"] == {
        "statements": {"one": {}},
        "blobs": {"two": "value"},
    }
    assert model.last_integrity_result is not None
    assert model.last_integrity_result.request_id == request_id


def test_manifest_fetch_retries_only_not_found(monkeypatch):
    import eqty_lineage.vnim.chat_models.vnim as module
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI

    responses = iter(
        [
            httpx.Response(404),
            httpx.Response(200, json={"statements": [], "blobs": {}}),
        ]
    )
    sleeps = []
    monkeypatch.setattr(module.httpx, "get", lambda url, timeout: next(responses))
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    result = ChatEqtyVnimOpenAI(
        model="test",
        api_key="test",
        manifest_base_url="http://127.0.0.1:8000",
    )._fetch_integrity_manifest(uuid4())

    assert result.manifest == {"statements": [], "blobs": {}}
    assert result.attempts == 2
    assert sleeps == [3.0]


def test_request_id_can_be_read_from_generation_info():
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI

    request_id = uuid4()
    chunk = ChatGenerationChunk(
        message=AIMessageChunk(content=""),
        generation_info={"headers": {"X-EQTY-Request-ID": str(request_id)}},
    )

    assert ChatEqtyVnimOpenAI._eqty_request_id_from_chunk(chunk) == (request_id, None)


def test_final_request_payload_is_retained_without_exposing_internal_mutation(sdk, monkeypatch):
    import eqty_lineage.vnim.chat_models.vnim as module
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI

    payload = {"model": "test", "messages": [{"role": "user", "content": "hello"}], "stream": True}
    monkeypatch.setattr(module.ChatOpenAI, "_get_request_payload", lambda *args, **kwargs: payload)
    model = ChatEqtyVnimOpenAI(model="test", api_key="test")

    assert model._get_request_payload("input") == payload
    exposed = model.last_request_payload
    assert exposed == payload
    exposed["model"] = "changed"
    assert model.last_request_payload == payload


def test_stream_yields_the_buffered_chunk_before_propagating_an_upstream_error(monkeypatch):
    import eqty_lineage.vnim.chat_models.vnim as module
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI

    first = ChatGenerationChunk(message=AIMessageChunk(content="first"))
    final = ChatGenerationChunk(message=AIMessageChunk(content="final"))

    def upstream_stream(self, *args, **kwargs):
        yield first
        yield final
        raise RuntimeError("upstream failed")

    monkeypatch.setattr(module.ChatOpenAI, "_stream", upstream_stream)
    model = ChatEqtyVnimOpenAI(model="test", api_key="test")
    stream = model.stream("hello")

    assert next(stream).text == "first"
    assert next(stream).text == "final"
    with pytest.raises(RuntimeError, match="upstream failed"):
        next(stream)
    assert model.last_integrity_result is not None
    assert model.last_integrity_result.error == "upstream stream did not complete"


def test_generate_attaches_integrity_metadata(monkeypatch):
    import eqty_lineage.vnim.chat_models.vnim as module
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI

    request_id = uuid4()
    generation = ChatGeneration(
        message=AIMessage(
            content="complete",
            response_metadata={"headers": {"X-EQTY-Request-ID": str(request_id)}},
        )
    )
    monkeypatch.setattr(module.ChatOpenAI, "_generate", lambda *args, **kwargs: ChatResult(generations=[generation]))
    monkeypatch.setattr(
        module.httpx,
        "get",
        lambda url, timeout: httpx.Response(200, json={"statements": {}, "blobs": {}}),
    )

    model = ChatEqtyVnimOpenAI(model="test", api_key="test", manifest_base_url="http://127.0.0.1:8000")
    result = model._generate([])

    assert result.generations[0].message.response_metadata["eqty_request_id"] == str(request_id)
    assert model.last_integrity_result is not None
    assert model.last_integrity_result.manifest == {"statements": {}, "blobs": {}}
