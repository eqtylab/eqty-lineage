"""Drive a transcript through the recorder and export a manifest."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eqty_lineage.core import ContentPolicy, LineageRecorder, TripleSink, project_manifest, select_file_lineage

from .claude_code import ClaudeCodeTranscript

logger = logging.getLogger("eqty.lineage.transcript")

# eqty-sdk 2.2.0: Context.export() builds one query binding ~3 parameters per statement_graph_link row
# and never chunks, so it hits SQLITE_MAX_VARIABLE_NUMBER (32766) and raises
#   RuntimeError: variable number must be between ?1 and ?32766
# Measured on a real session: 10,730 links export, 11,285 fail -- 32766/3 = 10,922 sits between them.
# Roughly one in six local sessions is large enough to trip this. Statement recording itself is
# unaffected, and so is the triple sidecar, so a session over the limit still yields a usable fact set.
SDK_EXPORT_STATEMENT_LIMIT = 10_922


@dataclass
class IngestResult:
    session_id: str | None
    transcript: Path
    manifest: Path | None
    triples: TripleSink
    events: int
    file_versions: int
    projection: Path | None = None
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def ingest(
    transcript: str | Path,
    manifest: str | Path | None = None,
    context_name: str | None = None,
    signer_name: str = "eqty-lineage-transcript",
    policy: ContentPolicy | None = None,
    triples_path: str | Path | None = None,
    verbose: bool = False,
    store_blobs: bool = True,
    projection: str | Path | None = None,
    service_url: str | None = None,
    service_key: str | None = None,
) -> IngestResult:
    """Ingest one Claude Code session transcript into EQTY lineage.

    The SDK is imported lazily so that parsing, the invariant checks, and the test suite can run
    without an ``eqty_sdk`` install or a signing key -- which is what makes the corpus sweep cheap
    enough to run over hundreds of sessions.
    """
    from eqty_sdk import Context, Signer, init, set_active_signer

    path = Path(transcript).expanduser()
    parser = ClaudeCodeTranscript(path)
    events = list(parser.events())

    session_id = next((getattr(e, "session_id", None) for e in events if getattr(e, "session_id", None)), path.stem)

    ctx = Context.new(context_name or f"claude-code session {session_id}")
    cfg = init(default_context=ctx).set_store_all_blobs(store_blobs)
    set_active_signer(Signer.new(name=signer_name, _load_if_exists=True))

    sink = TripleSink(triples_path)
    recorder = LineageRecorder(
        policy=policy if policy is not None else ContentPolicy(),
        triples=sink,
        framework="claude-code",
        verbose=verbose,
    )
    recorder.handle_all(events)

    warnings = list(parser.warnings)
    out: Path | None = None
    if manifest is not None:
        out = Path(manifest).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)  # export() will not create it
        try:
            cfg.get_default_context().export(out)
        except RuntimeError as exc:
            # See SDK_EXPORT_STATEMENT_LIMIT. The statements are recorded and the triple sidecar is
            # complete; only the manifest file is lost. Failing the whole ingest over an SDK export
            # limit would discard work that succeeded.
            out = None
            warnings.append(
                f"manifest export failed (session exceeds the ~{SDK_EXPORT_STATEMENT_LIMIT}-statement "
                f"eqty-sdk export limit); statements and triples are intact: {exc}"
            )

    projected: Path | None = None
    if projection is not None and out is not None:
        # A projection is a subset of the signed manifest, so it has to be cut from an exported file
        # rather than re-recorded -- re-recording would mint new statement CIDs and sign a different
        # document.
        projected = Path(projection).expanduser()
        stats = project_manifest(out, projected, select_file_lineage(sink))
        warnings.append(f"projection: {stats.summary()}")

    if service_url:
        from eqty_sdk import Service

        try:
            cfg.get_default_context().register(Service.new(service_url, service_key))
        except Exception as exc:  # noqa: BLE001 - a failed upload must not lose the local manifest
            warnings.append(f"service registration failed: {exc}")

    return IngestResult(
        session_id=session_id,
        projection=projected,
        transcript=path,
        manifest=out,
        triples=sink,
        events=len(events),
        file_versions=sum(len(v) for v in recorder.file_versions.values()),
        warnings=warnings,
        stats=dict(recorder.stats),
    )


def parse_only(transcript: str | Path) -> dict[str, Any]:
    """Parse without touching the SDK. Used by the sweep to check parser coverage cheaply."""
    parser = ClaudeCodeTranscript(Path(transcript).expanduser())
    events = list(parser.events())
    counts: dict[str, int] = {}
    for event in events:
        counts[type(event).__name__] = counts.get(type(event).__name__, 0) + 1
    return {"events": events, "counts": counts, "warnings": parser.warnings}


__all__ = ["IngestResult", "ingest", "parse_only"]
