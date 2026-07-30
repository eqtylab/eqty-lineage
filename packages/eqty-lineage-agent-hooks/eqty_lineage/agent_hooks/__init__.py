"""Live EQTY lineage capture from Claude Code and Codex hooks.

    eqty-lineage-hooks serve --port 8787 --manifests ./manifests
    eqty-lineage-hooks install --print      # settings.json wiring

HTTP is the recommended transport: Claude Code can deliver hooks as POSTs, so the session's recorder
and run tree stay in memory rather than being rebuilt per event.
"""

from .dialects import CLAUDE_CODE, CLAUDE_CODE_EVENTS, CODEX, CODEX_EVENTS, detect_dialect, to_events
from .policy import Decision, HookPolicy
from .replay_policy import PolicyComparison, PolicyOutcome, compare_policies, replay

# The dialects and the policy are pure payload handling; the daemon, the session registry and the
# equivalence checker all reach eqty_sdk through the recorder. Importing them here would mean that
# `from eqty_lineage.agent_hooks.policy import HookPolicy` needed the SDK from a private index, which
# is not a dependency the policy has. Loaded on first use instead.
_LAZY = {
    "HookReceiver": ".daemon",
    "serve": ".daemon",
    "HOOKS_ONLY_TYPES": ".equivalence",
    "TRANSCRIPT_ONLY_TYPES": ".equivalence",
    "Report": ".equivalence",
    "compare": ".equivalence",
    "replay_payloads": ".replay",
    "SessionRegistry": ".session",
    "SessionState": ".session",
}


def __getattr__(name: str):  # PEP 562
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)


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
