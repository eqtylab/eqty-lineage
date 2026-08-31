"""Framework-agnostic lineage recorder shared by the EQTY lineage integrations.

Adapters translate their source into :mod:`~eqty_lineage.recorder.events` and hand them to
:class:`~eqty_lineage.recorder.recorder.LineageRecorder`, which produces EQTY assets, computation statements,
and a parallel triple fact set. Nothing but the recorder touches ``eqty_sdk``.

    from eqty_lineage.recorder import LineageRecorder, SessionStarted, ToolCallStarted

    recorder = LineageRecorder()
    recorder.handle(SessionStarted(session_id="...", agent="claude-code"))
"""

from .canonical import activity_signatures, canonical_triples, graph_diff, signature_label
from .coverage import Coverage
from .events import (
    Compacted,
    EditAttempt,
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
from .projection import (
    ProjectionStats,
    project_manifest,
    select_determination,
    select_file_lineage,
)
from .redaction import PERMISSIVE, ContentPolicy
from .serialize import as_bytes, scalar_metadata, to_jsonable
from .tool_results import apply_edit, file_events_from_result
from .triples import Triple, TripleSink

# `.recorder` is the one module here that imports eqty_sdk, and it is loaded on first use rather than at
# import time. Everything else in this package -- the event vocabulary, the tool-result parser, the
# redaction policy, the triple sidecar -- is pure Python, and importing any of it used to require the
# SDK from a private index purely because this line sat at module scope. That is what made the
# "runs without eqty-sdk" test suite impossible to run without eqty-sdk.
_LAZY = {"FileVersion": ".recorder", "LineageRecorder": ".recorder"}


def __getattr__(name: str):  # PEP 562
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)


__all__ = [
    "PERMISSIVE",
    "Compacted",
    "ContentPolicy",
    "Coverage",
    "EditAttempt",
    "Event",
    "FileMode",
    "FileObserved",
    "FileVersion",
    "InstructionsLoaded",
    "LineageRecorder",
    "ModelCall",
    "PermissionDecision",
    "PermissionOutcome",
    "ProjectionStats",
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
    "as_bytes",
    "canonical_triples",
    "file_events_from_result",
    "graph_diff",
    "project_manifest",
    "scalar_metadata",
    "select_determination",
    "select_file_lineage",
    "signature_label",
    "to_jsonable",
]
