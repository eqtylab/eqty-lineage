"""Per-session recorder registry, shared by the HTTP daemon and the command-hook CLI.

The two transports differ in exactly one way that matters. The daemon is a persistent process, so a
session's recorder and its run tree live in memory for the session's lifetime -- the same shape the
LangChain handler has always had. A command hook is a *fresh process per event*, so nothing survives
except what is written down; ``Context.from_uuid`` rehydrates the SDK context from a sidecar and the
recorder is rebuilt from scratch each time.

That second mode is genuinely lossy: a recorder rebuilt per event cannot remember an open tool call, so
correlation across the Pre/Post pair depends on the sidecar rather than on memory. HTTP is the
recommended transport for that reason, not merely for latency.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eqty_lineage.core import (
    ContentPolicy,
    TripleSink,
    project_manifest,
    select_file_lineage,
)

if TYPE_CHECKING:  # LineageRecorder pulls in eqty_sdk; see the note in eqty_lineage.core
    from eqty_lineage.core import LineageRecorder

logger = logging.getLogger("eqty.lineage.hooks")

DEFAULT_STATE_DIR = Path(".eqty_sdk") / "sessions"


@dataclass
class SessionState:
    session_id: str
    recorder: "LineageRecorder"
    context_uuid: str | None = None
    triples_path: Path | None = None
    events_seen: int = 0
    watch_paths: list[str] = field(default_factory=list)


class SessionRegistry:
    """Maps session ids to recorders, creating them on first sight.

    Serializes per session with a lock. The recorder is not thread-safe by design -- a tool call's start
    and end must be applied in order -- and the daemon is free to serve different sessions concurrently
    because they share nothing.

    ``store_blobs`` defaults to **on**, and maps to the SDK's ``set_store_all_blobs``. It is what
    makes a manifest carry the data rather than only its content addresses -- measured on a real Codex
    session, 16 blobs versus 28: the patch document, the file's bytes, the user's prompt and the
    model's reply. Without it the manifest attests that a file with a given CID existed and cannot show
    it, which is a weaker artifact than most readers assume they are holding.

    What bounds the risk is :mod:`eqty_lineage.core.redaction`, not this flag. Every file the agent
    reads is a candidate for the blob store, which is how a ``.env`` or a signing key becomes durable
    on disk; the policy denies those by pattern and scrubs the rest. Verified with blobs on: a ``.env``
    read by the agent has its bytes withheld from every blob, and the omission is *counted* in
    ``coverage.content_redacted`` rather than passing silently.

    One caveat that the redaction module's own docstring overstates: a denied file keeps its identity in
    the **triple sidecar** (its ``hasPath`` fact is emitted) but does **not** appear in the exported
    manifest at all. A query asking "did anything touch X" against the manifest alone will miss it.

    Pass ``--no-blobs`` for a tree whose contents must not be made durable even in redacted form.
    """

    def __init__(
        self,
        state_dir: Path | None = None,
        policy: ContentPolicy | None = None,
        triples_dir: Path | None = None,
        verbose: bool = False,
        store_blobs: bool = True,
    ) -> None:
        self.state_dir = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
        self.policy = policy if policy is not None else ContentPolicy()
        self.triples_dir = Path(triples_dir) if triples_dir else None
        self.verbose = verbose
        self.store_blobs = store_blobs
        self._sessions: dict[str, SessionState] = {}
        # SDK Context objects, held alongside the recorder: a context cannot be reconstructed from the
        # sidecar mid-session without losing the statements already attached to it.
        self._contexts: dict[str, Any] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._sdk_ready = False

    # ------------------------------------------------------------------ SDK lifecycle
    def _ensure_sdk(self) -> None:
        """Initialize the SDK once per process. ``init()`` is process-global and raises on a second call."""
        if self._sdk_ready:
            return
        from eqty_sdk import Signer, init, set_active_signer

        init().set_store_all_blobs(self.store_blobs)
        set_active_signer(Signer.new(name="eqty-lineage-agent-hooks", _load_if_exists=True))
        self._sdk_ready = True

    def lock_for(self, session_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(session_id, threading.Lock())

    # ------------------------------------------------------------------ sidecar
    def _sidecar(self, session_id: str) -> Path:
        return self.state_dir / f"{session_id}.json"

    def _load_sidecar(self, session_id: str) -> dict[str, Any]:
        path = self._sidecar(session_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("unreadable session sidecar %s", path)
            return {}

    def _save_sidecar(self, state: SessionState) -> None:
        path = self._sidecar(state.session_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "session_id": state.session_id,
                        "context_uuid": state.context_uuid,
                        "events_seen": state.events_seen,
                        "pid": os.getpid(),
                    }
                ),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("could not write session sidecar %s", path, exc_info=True)

    # ------------------------------------------------------------------ recorders
    def get(self, session_id: str, create: bool = True) -> SessionState | None:
        state = self._sessions.get(session_id)
        if state is not None or not create:
            return state

        self._ensure_sdk()
        from eqty_lineage.core import LineageRecorder
        from eqty_sdk import Context

        sidecar = self._load_sidecar(session_id)
        context_uuid = sidecar.get("context_uuid")
        context = None
        if context_uuid:
            try:
                # Rehydrate rather than fork a second context for the same session, which would split
                # the manifest in two and lose every edge that crossed the boundary.
                import uuid as _uuid

                context = Context.from_uuid(_uuid.UUID(context_uuid))
            except Exception:  # noqa: BLE001 - a stale or foreign uuid must not block the session
                logger.warning("could not reopen context %s; starting a new one", context_uuid)

        if context is None:
            context = Context.new(f"agent session {session_id[:8]}")

        triples_path = self.triples_dir / f"{session_id}.jsonl" if self.triples_dir else None
        recorder = LineageRecorder(
            policy=self.policy,
            triples=TripleSink(triples_path),
            framework="agent-hooks",
            verbose=self.verbose,
        )
        state = SessionState(
            session_id=session_id,
            recorder=recorder,
            context_uuid=str(getattr(context, "id", "") or "") or context_uuid,
            triples_path=triples_path,
        )
        self._sessions[session_id] = state
        self._contexts[session_id] = context
        self._save_sidecar(state)
        return state

    def context_for(self, session_id: str):
        return self._contexts.get(session_id)

    def close(self, session_id: str) -> SessionState | None:
        state = self._sessions.pop(session_id, None)
        self._contexts.pop(session_id, None)
        if state is not None:
            self._save_sidecar(state)
        return state

    def export(
        self,
        session_id: str,
        path: Path,
        projection: Path | None = None,
        service_url: str | None = None,
        service_key: str | None = None,
    ) -> str | None:
        """Export the session manifest. Returns an error string, or ``None`` on success."""
        context = self._contexts.get(session_id)
        state = self._sessions.get(session_id)
        if context is None:
            return "no context for session"
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)  # export() will not create it
            context.export(Path(path))

            if projection is not None and state is not None:
                # Cut from the exported file rather than re-recording: a projection has to be a subset
                # of the *signed* statements, and re-recording would mint new statement CIDs.
                stats = project_manifest(path, projection, select_file_lineage(state.recorder.triples))
                logger.info("projection for %s: %s", session_id, stats.summary())

            if service_url:
                from eqty_sdk import Service

                context.register(Service.new(service_url, service_key))
            return None
        except RuntimeError as exc:
            # eqty-sdk 2.2.0 binds ~3 parameters per statement_graph_link row in a single un-chunked
            # query, so a context over ~10,922 statements exceeds SQLITE_MAX_VARIABLE_NUMBER. The
            # statements and the triple sidecar are unaffected; only the manifest file is lost.
            #
            # Say so loudly and name the surviving record. A long session failing this way is exactly
            # when someone later asks where the manifest went, and "it is in the sidecar" is the answer
            # -- silently returning a string left that discoverable only by reading this code.
            if state is not None and state.triples_path is not None:
                logger.warning(
                    "manifest export failed for %s (%s statements recorded); the triple sidecar at %s "
                    "is unaffected and remains the complete record of this session",
                    session_id,
                    len(state.recorder.triples),
                    state.triples_path,
                )
            return str(exc)


__all__ = ["DEFAULT_STATE_DIR", "SessionRegistry", "SessionState"]
