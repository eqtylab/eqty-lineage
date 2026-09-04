"""Serve the instrumented travel agent behind a small local browser chat UI.

Requires ``OPENAI_API_KEY`` (and optionally ``OPENAI_MODEL``). Each browser gets a UUID-backed LangGraph
``thread_id``. ``EqtyCallbackHandler`` automatically records that thread beneath the EQTY root context
configured at startup.

    uv run python examples/langchain/travel_chat_ui.py
"""

import argparse
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any
from uuid import UUID
import webbrowser

from eqty_lineage.langchain import EqtyCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from travel_agent import build_graph, init_sdk


logger = logging.getLogger("eqty.travel_chat_ui")
# The SDK stores lineage in one local SQLite database. HTTP requests may arrive concurrently, but the
# instrumented graph invocation must be serialized so its asset and statement writes cannot contend.
lineage_lock = Lock()
thread_contexts: dict[str, Any] = {}


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EQTY Travel Agent</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; background: #101828; color: #e5e7eb; }
    body { margin: 0; } main { max-width: 760px; margin: auto; min-height: 100vh; display: flex; flex-direction: column; }
    header { padding: 24px 16px 12px; display: flex; align-items: start; gap: 12px; justify-content: space-between; } h1 { margin: 0; font-size: 1.5rem; } p { color: #98a2b3; }
    #messages { flex: 1; padding: 8px 16px 100px; } .message { white-space: pre-wrap; padding: 12px; margin: 10px 0; border-radius: 10px; max-width: 85%; }
    .user { background: #155eef; margin-left: auto; } .assistant { background: #27364e; }
    form { position: fixed; bottom: 0; left: 0; right: 0; background: #101828; border-top: 1px solid #344054; padding: 12px; display: flex; gap: 8px; justify-content: center; }
    textarea { width: min(620px, 75vw); resize: vertical; min-height: 42px; padding: 10px; border-radius: 8px; border: 1px solid #475467; background: #182230; color: inherit; }
    button { border: 0; border-radius: 8px; padding: 0 18px; background: #2e90fa; color: white; font-weight: 600; cursor: pointer; } #new-chat, #download-manifest { min-height: 36px; background: #344054; white-space: nowrap; } button:disabled { opacity: .55; }
  </style>
</head>
<body><main>
  <header><div><h1>EQTY Travel Agent</h1><p>Each browser conversation is a separate LangGraph thread and EQTY child context.</p></div><button id="new-chat" type="button">New chat</button></header>
  <section id="messages" aria-live="polite"></section>
</main>
<form id="chat"><button id="download-manifest" type="button" disabled>Download manifest</button><textarea id="message" placeholder="Plan a weekend in Lisbon" required></textarea><button>Send</button></form>
<script>
  const sessionKey = "eqty-travel-agent-session";
  let sessionId = sessionStorage.getItem(sessionKey) || crypto.randomUUID();
  sessionStorage.setItem(sessionKey, sessionId);
  const messages = document.querySelector("#messages"), form = document.querySelector("#chat"), input = document.querySelector("#message"), button = form.querySelector("button"), newChat = document.querySelector("#new-chat"), downloadManifest = document.querySelector("#download-manifest");
  function add(role, text) { const item = document.createElement("article"); item.className = `message ${role}`; item.textContent = text; messages.append(item); item.scrollIntoView({block: "end"}); }
  newChat.addEventListener("click", () => { sessionId = crypto.randomUUID(); sessionStorage.setItem(sessionKey, sessionId); messages.replaceChildren(); downloadManifest.disabled = true; input.focus(); });
  downloadManifest.addEventListener("click", async () => {
    const response = await fetch(`/manifest/${sessionId}`); if (!response.ok) { add("assistant", `Manifest download failed: ${await response.text()}`); return; }
    const link = document.createElement("a"); link.href = URL.createObjectURL(await response.blob()); link.download = `travel-agent-${sessionId}.json`; link.click(); URL.revokeObjectURL(link.href);
  });
  form.addEventListener("submit", async event => {
    event.preventDefault(); const message = input.value.trim(); if (!message) return;
    add("user", message); input.value = ""; button.disabled = true;
    try {
      const response = await fetch("/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({session_id: sessionId, message})});
      const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "Request failed");
      add("assistant", payload.answer); downloadManifest.disabled = false;
    } catch (error) { add("assistant", `Error: ${error.message}`); }
    finally { button.disabled = false; input.focus(); }
  });
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
  });
</script></body></html>"""


class TravelChatHandler(BaseHTTPRequestHandler):
    app: Any

    def send_json(self, status: HTTPStatus, payload: dict[str, str]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_manifest(self, session_id: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as manifest_file:
            manifest_path = Path(manifest_file.name)
        try:
            with lineage_lock:
                context = thread_contexts.get(session_id)
                if context is None:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "no lineage exists for this chat yet"})
                    return
                context.export(manifest_path)
            body = manifest_path.read_bytes()
        finally:
            manifest_path.unlink(missing_ok=True)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Disposition", f'attachment; filename="travel-agent-{session_id}.json"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        logger.info("chat.manifest_downloaded session_id=%s context_id=%s", session_id, context.id)

    def do_GET(self) -> None:  # noqa: N802
        manifest_prefix = "/manifest/"
        if self.path.startswith(manifest_prefix):
            try:
                session_id = str(UUID(self.path[len(manifest_prefix) :]))
            except ValueError:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid session id"})
                return
            self.send_manifest(session_id)
            return
        if self.path != "/":
            logger.info("http.not_found method=GET path=%s", self.path)
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        logger.info("ui.page_served client=%s", self.client_address[0])
        body = PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/chat":
            logger.info("http.not_found method=POST path=%s", self.path)
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        started = time.monotonic()
        session_id = "unknown"
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            session_id = str(UUID(str(payload["session_id"])))
            message = str(payload["message"]).strip()
            if not message:
                raise ValueError("message is required")

            logger.info("chat.started session_id=%s input_chars=%d", session_id, len(message))

            with lineage_lock:
                lineage_handler = EqtyCallbackHandler()
                result = self.app.invoke(
                    {"messages": [HumanMessage(message)], "question": message, "answer": ""},
                    config={
                        "callbacks": [lineage_handler],
                        "configurable": {"thread_id": session_id},
                        "run_name": "Travel Assistant",
                        "recursion_limit": 25,
                    },
                )
            answer = str(result["answer"])
            self.send_json(HTTPStatus.OK, {"answer": answer})
            logger.info(
                "chat.completed session_id=%s output_chars=%d duration_ms=%d",
                session_id,
                len(answer),
                round((time.monotonic() - started) * 1000),
            )
            logger.info(
                "chat.lineage_context session_id=%s context_id=%s context_name=%s",
                session_id,
                lineage_handler.context.id,
                lineage_handler.context.name,
            )
            with lineage_lock:
                thread_contexts[session_id] = lineage_handler.context
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logger.warning("chat.rejected session_id=%s error=%s", session_id, error)
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:  # noqa: BLE001
            logger.exception("chat.failed session_id=%s", session_id)
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the EQTY travel-agent chat UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    init_sdk()  # The root context; EqtyCallbackHandler creates a child per UI thread_id.
    TravelChatHandler.app = build_graph(checkpointer=MemorySaver())
    server = ThreadingHTTPServer((args.host, args.port), TravelChatHandler)
    url = f"http://{args.host}:{args.port}"
    logger.info("server.started url=%s model=%s", url, os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
