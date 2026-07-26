"""PreToolUse enforcement, evaluated against the lineage graph built so far.

The point of putting policy here rather than in a separate linter is that the same rules serve both
directions: offline they are an audit over a finished manifest, live they are the reason a tool call is
denied. A rule that can only be checked after the fact is a report; one checked before the write happens
is a control.

``eqty-lineage-query`` is an optional import. A policy expressed as path globs needs no Datalog, and a
deployment that only wants "deny writes outside the repo" should not have to install a query engine.
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("eqty.lineage.hooks")

Decision = Tuple[str, str]
"""(permissionDecision, reason) -- "allow" | "deny" | "ask"."""

# Tool inputs that name a file, by the key each agent uses.
_PATH_KEYS = ("file_path", "path", "filePath", "notebook_path")


@dataclass
class HookPolicy:
    """Deny writes to paths outside a permitted set.

    ``deny_write_globs`` wins over ``allow_write_globs``, matching the SDK's own deny-over-allow
    ordering. An empty ``allow_write_globs`` means "no path restriction", not "deny everything" -- a
    policy object that silently bricked the agent on construction would be worse than no policy.
    """

    allow_write_globs: Sequence[str] = ()
    deny_write_globs: Sequence[str] = ()
    deny_tools: Sequence[str] = ()
    # Tools that write. Anything not listed is not path-checked, because its input paths are reads.
    write_tools: Sequence[str] = ("Edit", "Write", "NotebookEdit", "apply_patch")
    dry_run: bool = False
    """Report what would be denied without denying it. The honest way to roll a policy out."""

    violations: List[str] = field(default_factory=list)

    def decide(self, payload: Dict[str, Any], recorder: Any = None) -> Optional[Decision]:
        """Return a decision, or ``None`` to defer to the agent's normal permission flow.

        Deferring rather than allowing is deliberate: returning ``allow`` from a hook *overrides* the
        user's own settings, so a lineage recorder that answered ``allow`` by default would silently
        widen the agent's permissions. Recording provenance must never grant authority.
        """
        tool = payload.get("tool_name") or ""
        tool_input = payload.get("tool_input") or {}

        if tool in self.deny_tools:
            return self._verdict(f"tool '{tool}' is denied by lineage policy")

        if tool not in self.write_tools:
            return None

        path = None
        if isinstance(tool_input, dict):
            path = next((tool_input[k] for k in _PATH_KEYS if isinstance(tool_input.get(k), str)), None)
        if not path:
            return None

        for pattern in self.deny_write_globs:
            if fnmatch.fnmatch(path, pattern):
                return self._verdict(f"write to '{path}' matches denied pattern '{pattern}'")

        if self.allow_write_globs and not any(fnmatch.fnmatch(path, p) for p in self.allow_write_globs):
            return self._verdict(f"write to '{path}' is outside the permitted set")

        return None

    def _verdict(self, reason: str) -> Decision:
        self.violations.append(reason)
        if self.dry_run:
            logger.warning("would deny: %s", reason)
            return ("allow", f"eqty-lineage (dry run): {reason}")
        return ("deny", f"eqty-lineage: {reason}")


__all__ = ["Decision", "HookPolicy"]
