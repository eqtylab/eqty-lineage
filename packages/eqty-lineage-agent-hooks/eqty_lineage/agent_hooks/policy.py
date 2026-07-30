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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger("eqty.lineage.hooks")

Decision = Tuple[str, str]
"""(permissionDecision, reason) -- "allow" | "deny" | "ask" | "defer"."""

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
        """Return a decision, or ``None`` to say nothing at all.

        Deferring rather than allowing is deliberate: returning ``allow`` from a hook *overrides* the
        user's own settings, so a lineage recorder that answered ``allow`` by default would silently
        widen the agent's permissions. Recording provenance must never grant authority. This policy
        therefore only ever emits ``deny`` or ``defer`` -- never ``allow``.

        ``None`` and ``("defer", ...)`` reach the agent the same way; the difference is that ``None``
        means the policy had no opinion, while ``defer`` means it had one and is declining to enforce it.
        """
        tool = payload.get("tool_name") or ""
        tool_input = payload.get("tool_input") or {}

        if tool in self.deny_tools:
            return self._verdict(f"tool '{tool}' is denied by lineage policy")

        if tool not in self.write_tools:
            return None

        paths = list(self._written_paths(tool, tool_input, payload.get("cwd")))
        if not paths:
            return None

        for path in paths:
            for pattern in self.deny_write_globs:
                if fnmatch.fnmatch(path, pattern):
                    return self._verdict(f"write to '{path}' matches denied pattern '{pattern}'")

            if self.allow_write_globs and not any(fnmatch.fnmatch(path, p) for p in self.allow_write_globs):
                return self._verdict(f"write to '{path}' is outside the permitted set")

        return None

    def _written_paths(self, tool: str, tool_input: Any, cwd: Optional[str]) -> Iterator[str]:
        """Every path this call would write.

        ``apply_patch`` carries a patch document rather than a path, so the key lookup finds nothing and
        the call defers -- which left every Codex write unchecked while ``apply_patch`` sat in
        ``write_tools`` looking enforced. One patch can also touch several files, and a policy that
        stopped at the first would pass a patch whose second hunk escapes the permitted set.
        """
        if not isinstance(tool_input, dict):
            return

        for key in _PATH_KEYS:
            if isinstance(tool_input.get(key), str) and tool_input[key]:
                yield tool_input[key]
                return

        command = tool_input.get("command")
        if isinstance(command, str):
            from .dialects import parse_apply_patch

            for path, _mode, _content in parse_apply_patch(command, cwd):
                yield path

    def _verdict(self, reason: str) -> Decision:
        self.violations.append(reason)
        if self.dry_run:
            # "defer" hands the call back to the user's own permission flow. Returning "allow" here --
            # as this did -- *overrides* their settings, so rolling a policy out in report-only mode
            # silently widened the agent's permissions on exactly the calls it was flagging.
            logger.warning("would deny: %s", reason)
            return ("defer", f"eqty-lineage (dry run): {reason}")
        return ("deny", f"eqty-lineage: {reason}")


__all__ = ["Decision", "HookPolicy"]
