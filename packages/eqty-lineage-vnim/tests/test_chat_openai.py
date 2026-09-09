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


def _installed_transport(model):
    """The tee that ``_install_response_body_capture`` put on the shared HTTPX client."""
    return model.root_client._client._transport


def test_a_second_model_on_the_same_endpoint_records_its_own_response_body(sdk, monkeypatch):
    """LangChain caches one HTTPX client per endpoint, so the tee is shared by every model on it.

    A graph that builds its chat model more than once -- per node, or per request -- must still see
    its own response bytes.  Otherwise the response CID is missing and eqty-lineage-langchain falls
    back to hashing the assembled message, which no longer matches what vNIM registered.
    """
    import eqty_lineage.vnim.chat_models.vnim as module
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI

    body = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    def build():
        return ChatEqtyVnimOpenAI(model="test", api_key="test", manifest_base_url="http://127.0.0.1:8000")

    first, second = build(), build()
    assert first.root_client._client is second.root_client._client, "expected LangChain's cached client"

    def upstream_stream(self, *args, **kwargs):
        # Stand in for HTTPX handing the transport a complete response body mid-iteration.
        yield ChatGenerationChunk(message=AIMessageChunk(content="hi"))
        _installed_transport(second)._on_complete(body)

    monkeypatch.setattr(module.ChatOpenAI, "_stream", upstream_stream)
    monkeypatch.setattr(module.httpx, "get", lambda url, timeout: httpx.Response(200, json={}))

    chunks = list(second.stream("hello"))

    from eqty_sdk import get_cid_for_bytes

    assert chunks[-1].response_metadata.get("eqty_openai_response_cid") == str(get_cid_for_bytes(body))
    assert first._last_response_cid.get() is None, "bytes must not be attributed to another model"


def test_a_proxied_client_is_still_teed(sdk):
    """A client built for a proxy resolves every request through ``_mounts``.

    ``httpx.Client._transport_for_url`` consults ``_mounts`` before ``_transport``, so wrapping
    ``_transport`` alone tees nothing for such a client -- and langchain-openai builds exactly that
    shape for ``openai_proxy``: ``httpx.Client(mounts={"all://": transport})``.
    """
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI
    from eqty_lineage.vnim.chat_models.vnim import _ResponseBodyCapturingTransport

    proxied = httpx.Client(mounts={"all://": httpx.HTTPTransport()})
    model = ChatEqtyVnimOpenAI(model="test", api_key="test", http_client=proxied, base_url="http://vnim.example/v1")
    picked = model.root_client._client._transport_for_url(httpx.URL("http://vnim.example/v1/chat/completions"))

    assert isinstance(picked, _ResponseBodyCapturingTransport)


def test_an_uninitialized_sdk_does_not_escape_the_response_teardown(monkeypatch, caplog):
    """``_record_response_body`` runs inside HTTPX's stream teardown, so nothing may escape it.

    An uninitialized SDK raises pyo3's ``PanicException``, which derives from ``BaseException`` and
    is therefore not caught by ``except RuntimeError``.
    """
    import eqty_lineage.vnim.chat_models.vnim as module
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI

    class Panic(BaseException):
        """Stands in for pyo3_runtime.PanicException."""

    def panics(_body):
        raise Panic("Config not initialized")

    monkeypatch.setattr(module, "get_cid_for_bytes", panics)
    model = ChatEqtyVnimOpenAI(model="test", api_key="test")

    with caplog.at_level("WARNING"):
        model._record_response_body(b"data: [DONE]\n\n")

    assert "response_body_cid_unavailable" in caplog.text


def test_bytes_from_another_model_are_not_attributed_after_a_stream_ends(sdk, monkeypatch):
    """A finished request must release the shared tee.

    Every chat model on the endpoint shares the transport, so a sink left in place outlives the
    request that set it.  The next request through that client -- a plain ``ChatOpenAI``, another
    model, a manifest fetch -- would then have its bytes recorded as this model's response, which
    is a false attestation rather than a missing one.
    """
    import eqty_lineage.vnim.chat_models.vnim as module
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI

    mine = b'data: {"choices":[{"delta":{"content":"mine"}}]}\n\ndata: [DONE]\n\n'
    theirs = b'data: {"choices":[{"delta":{"content":"theirs"}}]}\n\ndata: [DONE]\n\n'

    def upstream_stream(self, *args, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="mine"))
        _installed_transport(model)._on_complete(mine)

    monkeypatch.setattr(module.ChatOpenAI, "_stream", upstream_stream)
    monkeypatch.setattr(module.httpx, "get", lambda url, timeout: httpx.Response(200, json={}))
    model = ChatEqtyVnimOpenAI(model="test", api_key="test", manifest_base_url="http://127.0.0.1:8000")

    list(model.stream("hello"))
    from eqty_sdk import get_cid_for_bytes

    recorded = model._last_response_cid.get()
    assert recorded == str(get_cid_for_bytes(mine))

    # Someone else's request now travels through the same shared transport.
    _installed_transport(model)._on_complete(theirs)

    assert model._last_response_cid.get() == recorded


def test_capture_is_installed_on_a_client_built_after_validation(sdk, monkeypatch):
    """The tee must be present for the client the request will actually use.

    ChatOpenAI builds its OpenAI client in its own validation hook, so installing only from this
    class's validator depends on hook ordering and on the client never being replaced afterwards.
    """
    import eqty_lineage.vnim.chat_models.vnim as module
    from eqty_lineage.vnim import ChatEqtyVnimOpenAI
    from eqty_lineage.vnim.chat_models.vnim import _ResponseBodyCapturingTransport

    body = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    model = ChatEqtyVnimOpenAI(model="test", api_key="test", manifest_base_url="http://127.0.0.1:8000")
    # Stand in for a client that did not exist when this class's validator ran.
    model.root_client._client = httpx.Client()
    assert not isinstance(_installed_transport(model), _ResponseBodyCapturingTransport)

    def upstream_stream(self, *args, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="hi"))
        _installed_transport(model)._on_complete(body)

    monkeypatch.setattr(module.ChatOpenAI, "_stream", upstream_stream)
    monkeypatch.setattr(module.httpx, "get", lambda url, timeout: httpx.Response(200, json={}))

    chunks = list(model.stream("hello"))

    from eqty_sdk import get_cid_for_bytes

    assert chunks[-1].response_metadata.get("eqty_openai_response_cid") == str(get_cid_for_bytes(body))
