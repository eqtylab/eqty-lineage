"""EQTY VNIM extensions for LangChain's OpenAI-compatible chat model."""

import json
import logging
import time
from copy import deepcopy
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

import httpx
from eqty_sdk import CID, get_cid_for_bytes
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils.utils import from_env
from langchain_openai import ChatOpenAI
from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntegrityManifestResult:
    """Result of retrieving the integrity manifest for one vNIM request.

    ``local_request_cid``/``local_response_cid`` are computed independently by this application from
    the literal bytes it sent/received -- they do not come from vNIM's manifest. Compare them against
    the manifest's own claimed CIDs before trusting that the two sides agree on what was transported.
    """

    request_id: UUID | None
    manifest: dict[str, Any] | None = None
    fetch_status: int | None = None
    error: str | None = None
    attempts: int = 0
    duration_ms: int = 0
    local_request_cid: CID | None = None
    local_response_cid: CID | None = None


class _ByteCapturingStream(httpx.SyncByteStream):
    """Tees the literal bytes read off an httpx response stream to a callback, unmodified."""

    def __init__(self, inner: httpx.SyncByteStream, on_complete: Any) -> None:
        self._inner = inner
        self._on_complete = on_complete

    def __iter__(self) -> Iterator[bytes]:
        buffer = bytearray()
        try:
            for chunk in self._inner:
                buffer.extend(chunk)
                yield chunk
        finally:
            self._on_complete(bytes(buffer))

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()


class _ByteCapturingTransport(httpx.BaseTransport):
    """Wraps a sync httpx transport to independently attest to the exact bytes read from the wire.

    This sits below langchain-openai and the openai SDK's own SSE parsing, so the captured bytes are
    what actually crossed the wire -- not a re-serialization that could quietly diverge from it. Both
    directions are hashed as raw bytes: vNIM's own manifest hashes the literal request and response
    bytes it observed (not a JCS-canonicalized reconstruction), so this must match that scheme exactly
    or CIDs on an untampered call would disagree.
    """

    def __init__(self, wrapped: httpx.BaseTransport, on_request_complete: Any, on_response_complete: Any) -> None:
        self._wrapped = wrapped
        self._on_request_complete = on_request_complete
        self._on_response_complete = on_response_complete

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._on_request_complete(request.content)
        response = self._wrapped.handle_request(request)
        response.stream = _ByteCapturingStream(response.stream, self._on_response_complete)
        return response

    def close(self) -> None:
        self._wrapped.close()


class _AsyncByteCapturingStream(httpx.AsyncByteStream):
    """Async counterpart of ``_ByteCapturingStream``."""

    def __init__(self, inner: httpx.AsyncByteStream, on_complete: Any) -> None:
        self._inner = inner
        self._on_complete = on_complete

    async def __aiter__(self) -> Any:
        buffer = bytearray()
        try:
            async for chunk in self._inner:
                buffer.extend(chunk)
                yield chunk
        finally:
            self._on_complete(bytes(buffer))

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()


class _AsyncByteCapturingTransport(httpx.AsyncBaseTransport):
    """Async counterpart of ``_ByteCapturingTransport``."""

    def __init__(self, wrapped: httpx.AsyncBaseTransport, on_request_complete: Any, on_response_complete: Any) -> None:
        self._wrapped = wrapped
        self._on_request_complete = on_request_complete
        self._on_response_complete = on_response_complete

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._on_request_complete(request.content)
        response = await self._wrapped.handle_async_request(request)
        response.stream = _AsyncByteCapturingStream(response.stream, self._on_response_complete)
        return response

    async def aclose(self) -> None:
        await self._wrapped.aclose()


class ChatEqtyVnimOpenAI(ChatOpenAI):
    """Streaming ChatOpenAI client for EQTY VNIM models and their integrity manifests.

    ``streaming`` is enabled by default because vNIM's request ID is available on a streamed response
    header. ``last_integrity_result`` and ``last_request_payload`` are scoped to the current thread or
    async task, so callers sharing a model instance must read them from the same invocation context.
    """

    streaming: bool = True
    include_response_headers: bool = True
    manifest_base_url: str | None = Field(default_factory=from_env("VNIM_INTEGRITY_BASE_URL", default=None))
    _last_integrity_result: ContextVar[IntegrityManifestResult | None] = PrivateAttr(
        default_factory=lambda: ContextVar("last_integrity_result", default=None)
    )
    _last_request_payload: ContextVar[dict[str, Any] | None] = PrivateAttr(
        default_factory=lambda: ContextVar("last_request_payload", default=None)
    )
    # Populated by the transport wrapper below with the literal bytes read from/written to the wire for
    # the most recent request/response, keyed to this thread/task the same way the other ContextVars are.
    _last_local_request_bytes: ContextVar[bytes | None] = PrivateAttr(
        default_factory=lambda: ContextVar("last_local_request_bytes", default=None)
    )
    _last_local_response_bytes: ContextVar[bytes | None] = PrivateAttr(
        default_factory=lambda: ContextVar("last_local_response_bytes", default=None)
    )

    @property
    def last_integrity_result(self) -> IntegrityManifestResult | None:
        """Manifest retrieval result for this thread or async task's most recent invocation."""
        return self._last_integrity_result.get()

    @property
    def last_request_payload(self) -> dict[str, Any] | None:
        """The exact OpenAI-compatible request payload handed to the upstream client."""
        payload = self._last_request_payload.get()
        return deepcopy(payload) if payload is not None else None

    @model_validator(mode="after")
    def _require_response_headers_for_integrity(self) -> Self:
        if "include_response_headers" in self.model_fields_set and not self.include_response_headers:
            logger.warning("include_response_headers=False is ignored because vNIM integrity retrieval requires it")
        self.include_response_headers = True
        return self

    @model_validator(mode="after")
    def _wrap_transports_for_integrity(self) -> Self:
        """Tee the literal request/response bytes read off the wire for independent local attestation.

        Runs after ``ChatOpenAI.validate_environment`` (base-class validators run before subclass
        ones), by which point ``root_client``/``root_async_client`` already own their httpx clients.
        Wrapping the transport in place -- rather than pre-building a custom ``http_client`` -- avoids
        re-implementing upstream's default-client/proxy/socket-options construction.
        """
        root_client = getattr(self, "root_client", None)
        if root_client is not None:
            transport = root_client._client._transport
            if not isinstance(transport, _ByteCapturingTransport):
                root_client._client._transport = _ByteCapturingTransport(
                    transport, self._last_local_request_bytes.set, self._last_local_response_bytes.set
                )
        root_async_client = getattr(self, "root_async_client", None)
        if root_async_client is not None:
            async_transport = root_async_client._client._transport
            if not isinstance(async_transport, _AsyncByteCapturingTransport):
                root_async_client._client._transport = _AsyncByteCapturingTransport(
                    async_transport, self._last_local_request_bytes.set, self._last_local_response_bytes.set
                )
        return self

    @staticmethod
    def _eqty_request_id_from_chunk(chunk: ChatGenerationChunk) -> tuple[UUID | None, str | None]:
        response_headers = (chunk.message.response_metadata or {}).get("headers")
        generation_headers = (chunk.generation_info or {}).get("headers")
        request_id_values: list[str | None] = []
        for headers in (response_headers, generation_headers):
            if not isinstance(headers, Mapping):
                request_id_values.append(None)
                continue
            value = next((value for name, value in headers.items() if str(name).lower() == "x-eqty-request-id"), None)
            request_id_values.append(str(value) if value is not None else None)
            if value is None:
                continue
            try:
                return UUID(str(value)), None
            except (TypeError, ValueError, AttributeError):
                return None, "X-EQTY-Request-ID is not a valid UUID"
        logger.info(
            "eqty_vnim.request_id_headers response_metadata=%s generation_info=%s",
            request_id_values[0],
            request_id_values[1],
        )
        return None, None

    def _fetch_integrity_manifest(self, request_id: UUID) -> IntegrityManifestResult:
        if not self.manifest_base_url:
            return IntegrityManifestResult(request_id=request_id, error="manifest_base_url is not configured")
        url = f"{self.manifest_base_url.rstrip('/')}/integrity/manifest/{request_id}"
        started, attempts = time.monotonic(), 0
        while True:
            attempts += 1
            try:
                response = httpx.get(url, timeout=10.0)
            except httpx.HTTPError as error:
                duration_ms = round((time.monotonic() - started) * 1000)
                logger.warning(
                    "eqty_integrity_manifest.fetch_failed request_id=%s attempts=%d duration_ms=%d error=%s",
                    request_id,
                    attempts,
                    duration_ms,
                    type(error).__name__,
                )
                return IntegrityManifestResult(
                    request_id=request_id,
                    error=f"manifest request failed: {type(error).__name__}",
                    attempts=attempts,
                    duration_ms=duration_ms,
                )
            status, duration_ms = response.status_code, round((time.monotonic() - started) * 1000)
            logger.info(
                "eqty_integrity_manifest.fetch request_id=%s attempt=%d status=%d duration_ms=%d",
                request_id,
                attempts,
                status,
                duration_ms,
            )
            if 200 <= status < 300:
                try:
                    manifest = response.json()
                except (ValueError, json.JSONDecodeError):
                    return IntegrityManifestResult(
                        request_id=request_id,
                        fetch_status=status,
                        error="manifest response is not valid JSON",
                        attempts=attempts,
                        duration_ms=duration_ms,
                    )
                if not isinstance(manifest, dict):
                    return IntegrityManifestResult(
                        request_id=request_id,
                        fetch_status=status,
                        error="manifest response must be a JSON object",
                        attempts=attempts,
                        duration_ms=duration_ms,
                    )
                statements, blobs = manifest.get("statements"), manifest.get("blobs")
                logger.info(
                    "eqty_integrity_manifest.received request_id=%s attempts=%d status=%d duration_ms=%d statements=%d blobs=%d",
                    request_id,
                    attempts,
                    status,
                    duration_ms,
                    len(statements) if isinstance(statements, (dict, list)) else 0,
                    len(blobs) if isinstance(blobs, (dict, list)) else 0,
                )
                return IntegrityManifestResult(
                    request_id=request_id,
                    manifest=manifest,
                    fetch_status=status,
                    attempts=attempts,
                    duration_ms=duration_ms,
                )
            if status != 404:
                return IntegrityManifestResult(
                    request_id=request_id,
                    fetch_status=status,
                    error=f"manifest request returned HTTP {status}",
                    attempts=attempts,
                    duration_ms=duration_ms,
                )
            remaining = 45.0 - (time.monotonic() - started)
            if remaining <= 0:
                return IntegrityManifestResult(
                    request_id=request_id,
                    fetch_status=status,
                    error="manifest was not available within 45 seconds",
                    attempts=attempts,
                    duration_ms=duration_ms,
                )
            time.sleep(min(3.0, remaining))

    def _finish_integrity_result(
        self, request_id: UUID | None, header_error: str | None = None
    ) -> IntegrityManifestResult:
        # Computed independently of anything vNIM claims: raw-byte CIDs of the exact request/response
        # bytes this client wrote to and read from the wire. vNIM hashes the literal bytes it observed
        # on both sides too (not a JCS-canonicalized reconstruction), so this must match that scheme --
        # callers compare these against the manifest's own CIDs rather than trusting them.
        request_bytes = self._last_local_request_bytes.get()
        local_request_cid = get_cid_for_bytes(request_bytes, _store=True) if request_bytes is not None else None
        response_bytes = self._last_local_response_bytes.get()
        local_response_cid = get_cid_for_bytes(response_bytes, _store=True) if response_bytes is not None else None

        if header_error:
            result = IntegrityManifestResult(request_id=None, error=header_error)
        elif request_id is None:
            result = IntegrityManifestResult(request_id=None, error="X-EQTY-Request-ID header was not received")
        else:
            result = self._fetch_integrity_manifest(request_id)
        result = replace(result, local_request_cid=local_request_cid, local_response_cid=local_response_cid)
        self._last_integrity_result.set(result)
        return result

    @staticmethod
    def _attach_integrity_metadata(chunk: ChatGenerationChunk, result: IntegrityManifestResult) -> None:
        metadata = dict(chunk.message.response_metadata or {})
        if result.request_id is not None:
            metadata["eqty_request_id"] = str(result.request_id)
        if result.manifest is not None:
            metadata["eqty_integrity_manifest"] = result.manifest
        if result.error:
            metadata["eqty_integrity_error"] = result.error
        chunk.message.response_metadata = metadata

    def _get_request_payload(self, *args: Any, **kwargs: Any) -> dict:
        """Retain the finalized payload so it can be inspected via ``last_request_payload``.

        The independently-attested request CID is computed separately, from the literal bytes the
        transport wrapper sees on the wire -- not from this dict -- since it must match vNIM's own
        raw-byte hash of what it received, not a re-serialization of this payload.
        """
        payload = super()._get_request_payload(*args, **kwargs)
        self._last_request_payload.set(deepcopy(payload))
        return payload

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Preserve the upstream SSE stream and enrich only its final parsed chunk."""
        self._last_integrity_result.set(None)
        self._last_local_request_bytes.set(None)
        self._last_local_response_bytes.set(None)
        request_id, header_error, pending = None, None, None
        try:
            for chunk in super()._stream(*args, **kwargs):
                if request_id is None and header_error is None:
                    request_id, header_error = self._eqty_request_id_from_chunk(chunk)
                if pending is not None:
                    yield pending
                pending = chunk
        except Exception:
            if pending is not None:
                yield pending
            self._last_integrity_result.set(
                IntegrityManifestResult(request_id=request_id, error="upstream stream did not complete")
            )
            raise
        if pending is None:
            self._last_integrity_result.set(
                IntegrityManifestResult(request_id=None, error="upstream stream produced no chunks")
            )
            return
        result = self._finish_integrity_result(request_id, header_error)
        self._attach_integrity_metadata(pending, result)
        if isinstance(pending.message, AIMessageChunk):
            pending.message.chunk_position = "last"
        yield pending

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._last_integrity_result.set(None)
        self._last_local_request_bytes.set(None)
        self._last_local_response_bytes.set(None)
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if result.generations:
            generation = result.generations[0]
            request_id, header_error = self._eqty_request_id_from_chunk(generation)
            self._attach_integrity_metadata(generation, self._finish_integrity_result(request_id, header_error))
        return result
