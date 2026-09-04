"""EQTY VNIM extensions for LangChain's OpenAI-compatible chat model."""

import json
import logging
import time
from copy import deepcopy
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
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
    """Result of retrieving the integrity manifest for one vNIM request."""

    request_id: UUID | None
    manifest: dict[str, Any] | None = None
    fetch_status: int | None = None
    error: str | None = None
    attempts: int = 0
    duration_ms: int = 0


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
        if header_error:
            result = IntegrityManifestResult(request_id=None, error=header_error)
        elif request_id is None:
            result = IntegrityManifestResult(request_id=None, error="X-EQTY-Request-ID header was not received")
        else:
            result = self._fetch_integrity_manifest(request_id)
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
        """Retain the finalized payload so it can be registered as the vNIM request input."""
        payload = super()._get_request_payload(*args, **kwargs)
        self._last_request_payload.set(deepcopy(payload))
        return payload

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Preserve the upstream SSE stream and enrich only its final parsed chunk."""
        self._last_integrity_result.set(None)
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
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if result.generations:
            generation = result.generations[0]
            request_id, header_error = self._eqty_request_id_from_chunk(generation)
            self._attach_integrity_metadata(generation, self._finish_integrity_result(request_id, header_error))
        return result
