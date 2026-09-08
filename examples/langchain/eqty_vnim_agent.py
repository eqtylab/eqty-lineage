"""A small conversational agent for an EQTY VNIM deployment.

By default this connects to a local SSH-tunnel sidecar:

    uv run python examples/langchain/eqty_vnim_agent.py

Set ``OPENAI_API_KEY`` when the VNIM deployment requires one. Each complete response
retrieves the server-side integrity manifest through ``ChatEqtyVnimOpenAI``; the response
bytes themselves are never reconstructed.
"""

import argparse
import logging
import os
from pathlib import Path
from uuid import UUID, uuid4

from eqty_sdk import Context, Signer, init, set_active_signer
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from eqty_lineage.langchain import EqtyCallbackHandler
from eqty_lineage.vnim import ChatEqtyVnimOpenAI


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


def run_agent(
    model: ChatEqtyVnimOpenAI, prompt: str, handler: EqtyCallbackHandler, thread_id: str
) -> AIMessage:
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

    return AIMessage(content="".join(content))


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
    handler = EqtyCallbackHandler()
    run_agent(model, args.message, handler, thread_id)
    manifest_path = args.manifest_out or Path("manifests") / f"eqty-vnim-agent-{thread_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    handler.context.export(manifest_path)
    logging.info("merged manifest exported path=%s context_id=%s", manifest_path, handler.context.id)


if __name__ == "__main__":
    main()
