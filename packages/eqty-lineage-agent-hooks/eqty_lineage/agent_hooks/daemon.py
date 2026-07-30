"""HTTP hook receiver.

Claude Code can deliver hooks as HTTP POSTs rather than only as command invocations, which is the whole
reason this is the recommended transport: the session's recorder and run tree stay in memory for the
session's lifetime, exactly the shape the LangChain handler has always had. The command-hook fallback
pays process startup per event and cannot hold an open tool call in memory at all.

Stdlib ``http.server`` on purpose. A published package should not drag a web framework in to serve one
route on localhost, and ``ThreadingHTTPServer`` is more than adequate for one agent's hook traffic.

**Bind to loopback.** Every payload contains prompts, file contents and tool output. The default bind is
127.0.0.1 and there is no reason to change it; a token is supported for the case where something else on
the machine could reach the port.

**Watcher content is as-of-read, not as-of-change.** A ``FileChanged`` payload names a path; the adapter
then reads that path off disk. Those are two moments, and nothing holds the file still between them. A
second write landing in the gap is recorded as the content of the first event, and a path deleted in the
gap reads as absent. This is why a removal is carried by ``change_type`` rather than inferred from a
failed read -- an inference would be indistinguishable from losing the race. The window is small and
unavoidable without an inotify payload carrying bytes, so the honest handling is to bound what is claimed:
watcher versions are ``observed`` (the change really happened) but their *content* is only what the file
held when it was read.
"""

import json
import logging
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from eqty_lineage.core import FileObserved

from .dialects import detect_dialect, to_events
from .policy import HookPolicy
from .session import SessionRegistry

logger = logging.getLogger("eqty.lineage.hooks")

MAX_BODY_BYTES = 32 << 20


class HookReceiver:
    """Applies hook payloads to per-session recorders and produces the hook's JSON response."""

    def __init__(
        self,
        registry: SessionRegistry,
        policy: Optional[HookPolicy] = None,
        watch_paths: Optional[list] = None,
        manifest_dir: Optional[Path] = None,
        projection_dir: Optional[Path] = None,
        service_url: Optional[str] = None,
        service_key: Optional[str] = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.watch_paths = list(watch_paths or [])
        self.manifest_dir = Path(manifest_dir) if manifest_dir else None
        self.projection_dir = Path(projection_dir) if projection_dir else None
        self.service_url = service_url
        self.service_key = service_key
        self.handled = 0
        self.errors = 0
        self._ignored_roots = self._own_storage()

    def _own_storage(self) -> list:
        """Directories this daemon writes to, which a watcher must not report back to it.

        ``--watch <repo>`` asks the agent to report every change under a tree. If the manifest
        directory, the triple sidecars or the SDK's blob store live inside that tree, recording a file
        version writes blobs, which the watcher reports as changes, which the recorder records. A live
        session with the default layout produced 55 file versions of which 54 were the recorder's own
        content-addressed blobs -- the real edit was one node in a graph made almost entirely of
        self-observation.
        """
        roots = []
        for candidate in (
            self.manifest_dir,
            self.projection_dir,
            getattr(self.registry, "state_dir", None),
            getattr(self.registry, "triples_dir", None),
        ):
            if candidate is not None:
                roots.append(Path(candidate).resolve())
        return roots

    def _is_own_storage(self, path: str) -> bool:
        # `.eqty_sdk` is matched by name wherever it appears: the SDK resolves its store relative to the
        # process's working directory, which the daemon does not choose and cannot report.
        resolved = Path(path)
        if any(part == ".eqty_sdk" for part in resolved.parts):
            return True
        try:
            resolved = resolved.resolve()
        except OSError:  # a path that no longer exists still has to be judged
            return False
        return any(resolved == root or root in resolved.parents for root in self._ignored_roots)

    def _is_self_observation(self, event: Any) -> bool:
        """Drop a file observation that is this daemon writing its own records.

        Only watcher-derived observations are filtered. A tool call that genuinely edits a file under
        one of these directories is the agent's doing and belongs in the graph.
        """
        return (
            isinstance(event, FileObserved)
            and event.tool_use_id is None
            and bool(event.path)
            and self._is_own_storage(event.path)
        )

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = payload.get("session_id") or "unknown-session"
        event_name = payload.get("hook_event_name") or ""
        dialect = detect_dialect(payload)

        with self.registry.lock_for(session_id):
            state = self.registry.get(session_id)
            if state is None:
                return {}

            from eqty_sdk.context import graph_context

            context = self.registry.context_for(session_id)
            events = [e for e in to_events(payload, dialect) if not self._is_self_observation(e)]
            try:
                if context is not None:
                    with graph_context(context):
                        state.recorder.handle_all(events)
                else:
                    state.recorder.handle_all(events)
                state.events_seen += len(events)
                self.handled += 1
            except Exception:  # noqa: BLE001 - never take down the agent over a recording failure
                self.errors += 1
                logger.exception("failed to record %s for session %s", event_name, session_id)
                return {}

            return self._response(event_name, payload, state, session_id)

    def _response(self, event_name, payload, state, session_id) -> Dict[str, Any]:
        if event_name == "SessionStart" and self.watch_paths:
            # Returning watchPaths is what upgrades Bash side effects from *inferred* to *observed*:
            # the resulting FileChanged events are the one thing the offline path cannot match.
            state.watch_paths = self.watch_paths
            return {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "watchPaths": self.watch_paths,
                }
            }

        if event_name == "PreToolUse" and self.policy is not None:
            decision = self.policy.decide(payload, state.recorder)
            if decision is not None:
                verdict, reason = decision
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": verdict,
                        "permissionDecisionReason": reason,
                    }
                }

        if event_name == "SessionEnd" and self.manifest_dir is not None:
            error = self.registry.export(
                session_id,
                self.manifest_dir / f"{session_id}.json",
                projection=(self.projection_dir / f"{session_id}.json") if self.projection_dir else None,
                service_url=self.service_url,
                service_key=self.service_key,
            )
            self.registry.close(session_id)
            if error:
                logger.warning("manifest export failed for %s: %s", session_id, error)
                return {"systemMessage": f"eqty-lineage: manifest export failed ({error[:80]})"}

        return {}


class _Handler(BaseHTTPRequestHandler):
    receiver: HookReceiver
    token: Optional[str] = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - silence per-request stderr spam
        logger.debug(fmt, *args)

    def _send(self, code: int, body: Dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._send(200, {"ok": True, "handled": self.receiver.handled, "errors": self.receiver.errors})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.token is not None:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {self.token}"
            # constant-time compare: the token is the only thing guarding a port carrying prompts,
            # file contents, and tool output
            if not secrets.compare_digest(supplied, expected):
                self._send(401, {"error": "unauthorized"})
                return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, {"error": "bad content-length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(400, {"error": "bad body length"})
            return

        try:
            payload = json.loads(self.rfile.read(length))
        except ValueError:
            self._send(400, {"error": "malformed json"})
            return
        if not isinstance(payload, dict):
            self._send(400, {"error": "payload must be an object"})
            return

        try:
            self._send(200, self.receiver.handle(payload))
        except Exception:  # noqa: BLE001 - a 500 here would surface as a hook failure to the agent
            logger.exception("unhandled error serving hook")
            self._send(200, {})


def serve(
    receiver: HookReceiver,
    host: str = "127.0.0.1",
    port: int = 8787,
    token: Optional[str] = None,
) -> Tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the receiver in a background thread. Returns ``(server, thread)``."""
    handler = type("_BoundHandler", (_Handler,), {"receiver": receiver, "token": token})
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="eqty-lineage-hooks", daemon=True)
    thread.start()
    logger.info("listening on http://%s:%d", host, port)
    return server, thread


__all__ = ["HookReceiver", "serve"]
