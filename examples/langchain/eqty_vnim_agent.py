"""A small conversational agent for an EQTY VNIM deployment.

By default this connects to a local SSH-tunnel sidecar:

    uv run python examples/langchain/eqty_vnim_agent.py

Set ``OPENAI_API_KEY`` when the VNIM deployment requires one. Each complete response
retrieves the server-side integrity manifest through ``ChatEqtyVnimOpenAI``; the response
bytes themselves are never reconstructed.
"""

import argparse
import base64
import binascii
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from eqty_sdk import CID, Context, Signer, init, set_active_signer
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from eqty_lineage.langchain import EqtyCallbackHandler
from eqty_lineage.vnim import ChatEqtyVnimOpenAI, IntegrityManifestResult


DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MANIFEST_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_ROOT_CONTEXT_ID = "11111111-2222-3333-4444-555555555557"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with an EQTY VNIM model and retrieve its integrity manifest.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)),
        help="VNIM OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--manifest-base-url",
        default=os.environ.get("VNIM_INTEGRITY_BASE_URL", DEFAULT_MANIFEST_BASE_URL),
        help="VNIM server-side integrity-manifest base URL.",
    )
    parser.add_argument(
        "message",
        nargs="?",
        default="Suggest a memorable three-day trip to Lisbon.",
        help="The one-shot request to send to the VNIM model.",
    )
    parser.add_argument(
        "--root-context-id",
        default=os.environ.get("EQTY_ROOT_CONTEXT_ID", DEFAULT_ROOT_CONTEXT_ID),
        help="EQTY root context UUID; child context is created automatically for this invocation.",
    )
    parser.add_argument("--manifest-out", type=Path, help="Where to write the merged EQTY manifest.")
    return parser.parse_args()


def _manifest_metadata(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return manifest metadata keyed by its subject CID without modifying the remote evidence."""
    blobs = manifest.get("blobs")
    statements = manifest.get("statements")
    if not isinstance(blobs, dict) or not isinstance(statements, dict):
        raise ValueError("vNIM manifest must contain object-valued statements and blobs")

    metadata_by_subject: dict[str, dict[str, Any]] = {}
    for statement in statements.values():
        if not isinstance(statement, dict) or statement.get("@type") != "MetadataRegistration":
            continue
        subject = statement.get("subject")
        metadata_cid = statement.get("metadata")
        if not isinstance(subject, str) or not isinstance(metadata_cid, str):
            continue
        encoded = blobs.get(metadata_cid.removeprefix("urn:cid:"))
        if not isinstance(encoded, str):
            continue
        try:
            metadata = json.loads(base64.b64decode(encoded))
        except (binascii.Error, ValueError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict):
            metadata_by_subject[subject] = metadata
    return metadata_by_subject


def _vnim_transport_cids(manifest: dict[str, Any]) -> tuple[CID, CID]:
    """Locate vNIM's canonical request and verbatim streamed-response assets by their metadata."""
    request_cids: list[str] = []
    response_cids: list[str] = []
    for subject, metadata in _manifest_metadata(manifest).items():
        name = metadata.get("name")
        if name == "Request Body":
            request_cids.append(subject)
        elif name == "Response Body":
            response_cids.append(subject)

    if len(request_cids) != 1 or len(response_cids) != 1:
        raise ValueError(
            "vNIM manifest must contain exactly one Request Body and one Response Body asset "
            f"(found requests={len(request_cids)} responses={len(response_cids)})"
        )
    return CID(request_cids[0]), CID(response_cids[0])


def register_vnim_bridge(handler: EqtyCallbackHandler, result: IntegrityManifestResult) -> None:
    """Connect the deferred local model call to independently attested vNIM I/O.

    The vNIM manifest is deliberately not imported or rewritten here. ``result.local_request_cid`` and
    ``result.local_response_cid`` were computed by this application from the literal bytes it sent and
    received -- not copied from vNIM's manifest -- so the handler can independently confirm they match
    vNIM's own claimed CIDs before linking them as the same evidence. A mismatch (e.g. a MITM tampering
    with the request or response in transit) is recorded as a visible mismatch statement rather than
    silently trusted; see ``EqtyCallbackHandler.register_external_chat_transport``.
    """
    assert result.manifest is not None
    vnim_request_cid, vnim_response_cid = _vnim_transport_cids(result.manifest)
    handler.register_external_chat_transport(
        local_request_cid=result.local_request_cid,
        vnim_request_cid=vnim_request_cid,
        vnim_response_cid=vnim_response_cid,
        local_response_cid=result.local_response_cid,
        name="Travel Agent -> vNIM inference",
    )
    logging.info(
        "vNIM bridge registered context_id=%s local_request_cid=%s vnim_request_cid=%s "
        "vnim_response_cid=%s local_response_cid=%s",
        handler.context.id,
        result.local_request_cid,
        vnim_request_cid,
        vnim_response_cid,
        result.local_response_cid,
    )


def merge_manifests(local_manifest: dict[str, Any], vnim_manifest: dict[str, Any]) -> dict[str, Any]:
    """Create a lossless bundle; duplicate CIDs may only carry byte-identical evidence."""
    local_version = local_manifest.get("version")
    vnim_version = vnim_manifest.get("version")
    if local_version != vnim_version:
        raise ValueError(f"cannot merge manifest versions {local_version!r} and {vnim_version!r}")

    bundle: dict[str, Any] = {"version": local_version}
    for section in ("contexts", "statements", "blobs"):
        merged: dict[str, Any] = {}
        for source_name, manifest in (("local", local_manifest), ("vNIM", vnim_manifest)):
            entries = manifest.get(section, {})
            if not isinstance(entries, dict):
                raise ValueError(f"{source_name} manifest {section} must be an object")
            for cid, value in entries.items():
                if cid in merged and merged[cid] != value:
                    raise ValueError(f"{section} CID conflict while merging {source_name} manifest: {cid}")
                merged[cid] = value
        bundle[section] = merged
    return bundle


def export_bundle(context: Context, path: Path, vnim_manifest: dict[str, Any] | None) -> None:
    """Export local lineage, then union the unmodified vNIM evidence into one portable manifest."""
    if vnim_manifest is None:
        context.export(path)
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as local_file:
        local_path = Path(local_file.name)
        context.export(local_path)
        local_manifest = json.loads(local_path.read_text())
    path.write_text(json.dumps(merge_manifests(local_manifest, vnim_manifest)))


def run_agent(
    model: ChatEqtyVnimOpenAI, prompt: str, handler: EqtyCallbackHandler, thread_id: str
) -> tuple[AIMessage, dict[str, Any] | None]:
    """Run the one-shot travel-planning agent and report manifest retrieval status."""
    messages: list[BaseMessage] = [
        SystemMessage(
            "You are a concise travel-planning agent running on an EQTY VNIM deployment. "
            "Give practical, well-structured advice."
        ),
        HumanMessage(prompt),
    ]
    content: list[str] = []
    print("\nassistant> ", end="", flush=True)
    for chunk in model.stream(
        messages,
        config={
            "callbacks": [handler],
            "configurable": {"thread_id": thread_id},
            "run_name": "EQTY VNIM Agent",
        },
    ):
        text = chunk.text
        if text:
            content.append(text)
            print(text, end="", flush=True)
    print()

    result = model.last_integrity_result
    if result is None:
        logging.warning("integrity manifest result was unavailable")
    elif result.manifest is not None and result.local_request_cid is not None:
        register_vnim_bridge(handler, result)
        statements = result.manifest.get("statements", {})
        blobs = result.manifest.get("blobs", {})
        logging.info(
            "integrity manifest available request_id=%s statements=%d blobs=%d attempts=%d duration_ms=%d",
            result.request_id,
            len(statements) if isinstance(statements, (dict, list)) else 0,
            len(blobs) if isinstance(blobs, (dict, list)) else 0,
            result.attempts,
            result.duration_ms,
        )
        logging.info("vNIM manifest queued for lossless bundle export context_id=%s", handler.context.id)
    else:
        handler.finalize_deferred_chat_models()
        logging.warning(
            "integrity manifest unavailable request_id=%s status=%s attempts=%d error=%s",
            result.request_id,
            result.fetch_status,
            result.attempts,
            result.error,
        )

    return AIMessage(content="".join(content)), result.manifest if result is not None else None


def init_sdk(root_context_id: str):
    root_context = Context.from_uuid(UUID(root_context_id))
    config = init(default_context=root_context).set_store_all_blobs(True)
    set_active_signer(Signer.new(name="eqty_vnim_agent", _load_if_exists=True))
    return config


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    init_sdk(args.root_context_id)
    model = ChatEqtyVnimOpenAI(
        model=args.model,
        base_url=args.base_url,
        # A local tunnel may not require authentication, but the OpenAI client requires a value.
        api_key=os.environ.get("OPENAI_API_KEY", "vnim"),
        manifest_base_url=args.manifest_base_url,
    )
    logging.info(
        "eqty-vnim-agent started model=%s base_url=%s manifest_base_url=%s",
        args.model,
        args.base_url,
        args.manifest_base_url,
    )
    thread_id = str(uuid4())
    handler = EqtyCallbackHandler(defer_chat_model_computations=True)
    _, vnim_manifest = run_agent(model, args.message, handler, thread_id)
    manifest_path = args.manifest_out or Path("manifests") / f"eqty-vnim-agent-{thread_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    export_bundle(handler.context, manifest_path, vnim_manifest)
    logging.info("merged manifest exported path=%s context_id=%s", manifest_path, handler.context.id)


if __name__ == "__main__":
    main()
