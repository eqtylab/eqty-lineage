"""Offline EQTY lineage ingestion from coding-agent session transcripts.

    from eqty_lineage.transcript import ingest
    result = ingest("~/.claude/projects/<slug>/<session>.jsonl", manifest="out.json")

The offline path is the conformance oracle for the live hook adapter: both consume the same session and
must produce the same graph, modulo an enumerated set of divergences neither can avoid.
"""

from .claude_code import ClaudeCodeTranscript, find_sessions, parse
from .invariants import InvariantViolation, check_invariants, graph_stats

# `ingest` drives the recorder and so needs eqty_sdk; the transcript *parser* does not. Loading it here
# would make parsing a transcript require the SDK from a private index. See eqty_lineage.core.
_LAZY = {"IngestResult": ".ingest", "ingest": ".ingest"}


def __getattr__(name: str):  # PEP 562
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)


__all__ = [
    "ClaudeCodeTranscript",
    "IngestResult",
    "InvariantViolation",
    "check_invariants",
    "find_sessions",
    "graph_stats",
    "ingest",
    "parse",
]
