"""Offline EQTY lineage ingestion from coding-agent session transcripts.

    from eqty_lineage.transcript import ingest
    result = ingest("~/.claude/projects/<slug>/<session>.jsonl", manifest="out.json")

The offline path is the conformance oracle for the live hook adapter: both consume the same session and
must produce the same graph, modulo an enumerated set of divergences neither can avoid.
"""

from .claude_code import ClaudeCodeTranscript, find_sessions, parse
from .ingest import IngestResult, ingest
from .invariants import InvariantViolation, check_invariants, graph_stats

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
