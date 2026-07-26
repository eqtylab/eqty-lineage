"""Framework-agnostic lineage recorder shared by the EQTY lineage integrations.

Adapters translate their source into :mod:`~eqty_lineage.core.events` and hand them to
:class:`~eqty_lineage.core.recorder.LineageRecorder`, which produces EQTY assets, computation statements,
and a parallel triple fact set. Nothing but the recorder touches ``eqty_sdk``.

    from eqty_lineage.core import LineageRecorder, SessionStarted, ToolCallStarted

    recorder = LineageRecorder()
    recorder.handle(SessionStarted(session_id="...", agent="claude-code"))
"""

from .events import (
    Compacted,
    Event,
    FileMode,
    FileObserved,
    InstructionsLoaded,
    ModelCall,
    PermissionDecision,
    PermissionOutcome,
    PromptSubmitted,
    SessionEnded,
    SessionStarted,
    SubagentEnded,
    SubagentStarted,
    ToolCallEnded,
    ToolCallStarted,
)
from .canonical import activity_signatures, canonical_triples, graph_diff, signature_label
from .projection import ProjectionStats, project_manifest, select_file_lineage
from .recorder import FileVersion, LineageRecorder
from .redaction import PERMISSIVE, ContentPolicy
from .tool_results import apply_edit, file_events_from_result
from .serialize import as_bytes, scalar_metadata, to_jsonable
from .triples import Triple, TripleSink

__all__ = [
    "PERMISSIVE",
    "ProjectionStats",
    "Compacted",
    "ContentPolicy",
    "Event",
    "FileMode",
    "FileObserved",
    "FileVersion",
    "InstructionsLoaded",
    "LineageRecorder",
    "ModelCall",
    "PermissionDecision",
    "PermissionOutcome",
    "PromptSubmitted",
    "SessionEnded",
    "SessionStarted",
    "SubagentEnded",
    "SubagentStarted",
    "ToolCallEnded",
    "ToolCallStarted",
    "Triple",
    "TripleSink",
    "activity_signatures",
    "apply_edit",
    "canonical_triples",
    "project_manifest",
    "select_file_lineage",
    "graph_diff",
    "signature_label",
    "as_bytes",
    "file_events_from_result",
    "scalar_metadata",
    "to_jsonable",
]
