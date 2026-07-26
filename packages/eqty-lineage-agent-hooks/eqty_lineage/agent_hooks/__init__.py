"""Live EQTY lineage capture from Claude Code and Codex hooks.

    eqty-lineage-hooks serve --port 8787 --manifests ./manifests
    eqty-lineage-hooks install --print      # settings.json wiring

HTTP is the recommended transport: Claude Code can deliver hooks as POSTs, so the session's recorder
and run tree stay in memory rather than being rebuilt per event.
"""

from .daemon import HookReceiver, serve
from .equivalence import HOOKS_ONLY_TYPES, TRANSCRIPT_ONLY_TYPES, Report, compare
from .replay import replay_payloads
from .dialects import CLAUDE_CODE, CLAUDE_CODE_EVENTS, CODEX, CODEX_EVENTS, detect_dialect, to_events
from .policy import Decision, HookPolicy
from .replay_policy import PolicyComparison, PolicyOutcome, compare_policies, replay
from .session import SessionRegistry, SessionState

__all__ = [
    "CLAUDE_CODE",
    "CLAUDE_CODE_EVENTS",
    "CODEX",
    "CODEX_EVENTS",
    "HOOKS_ONLY_TYPES",
    "TRANSCRIPT_ONLY_TYPES",
    "Decision",
    "Report",
    "compare",
    "replay_payloads",
    "HookPolicy",
    "HookReceiver",
    "PolicyComparison",
    "PolicyOutcome",
    "compare_policies",
    "replay",
    "SessionRegistry",
    "SessionState",
    "detect_dialect",
    "serve",
    "to_events",
]
