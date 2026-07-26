"""Deriving file observations from a tool result payload.

Shared by both capture paths on purpose. A transcript's ``toolUseResult`` and a ``PostToolUse`` hook's
``tool_response`` are the same structure -- the agent serializes a tool's outcome once and both surfaces
carry it. Parsing it in two places would guarantee the offline and live graphs drift apart on exactly
the details the equivalence test is meant to police.

Dispatch is on the result's *shape*, never the tool name. A resumed session carries results whose
``tool_use`` block lives in a different transcript, so the name is simply not recoverable -- and those
results are often reads and edits whose file lineage would otherwise be lost. It also means a new
edit-shaped tool works without this being updated.
"""

from typing import Any, List, Optional, Tuple

from .events import FileObserved


def apply_edit(
    original: Optional[str], old: Optional[str], new: Optional[str], replace_all: bool
) -> Optional[str]:
    """Reconstruct post-edit content by replaying the literal replacement.

    Exact, not approximate: an edit performs literal string replacement, so replaying it against the
    pre-edit content reproduces the file byte for byte. Returns ``None`` when the inputs do not permit a
    confident reconstruction -- the caller then records the transition identity-only rather than
    guessing, because a wrong post-state would content-address to a version that never existed.
    """
    if original is None or old is None or new is None:
        return None
    if old not in original:
        # The payload disagrees with itself. Refusing is the honest outcome.
        return None
    return original.replace(old, new) if replace_all else original.replace(old, new, 1)


def file_events_from_result(
    result: Any,
    tool_use_id: Optional[str],
    at: Optional[str] = None,
    include_partial_reads: bool = True,
) -> Tuple[List[FileObserved], Optional[str]]:
    """Derive file versions from one tool result.

    Returns ``(events, attributed_path)``. ``attributed_path`` is set when the result fully described a
    file transition, so a caller reconciling a filesystem-snapshot delta can tell that the change is
    already accounted for and avoid double-reporting it from a worse angle.

    A result is sometimes a bare string rather than a mapping (626 of the Bash results in the surveyed
    corpus). Those carry no structured file information and yield nothing.
    """
    if not isinstance(result, dict):
        return [], None

    file_info = result.get("file")
    is_read = isinstance(file_info, dict) and "filePath" in file_info
    is_edit = "filePath" in result and any(k in result for k in ("originalFile", "content", "structuredPatch"))

    if is_read:
        path = file_info.get("filePath")
        if not path:
            return [], None
        content = file_info.get("content")
        start = file_info.get("startLine")
        num, total = file_info.get("numLines"), file_info.get("totalLines")
        partial = bool((start not in (None, 1)) or (num is not None and total is not None and num < total))

        if partial:
            # A fragment's hash is not the file's hash. Recording it as a version would invent a state
            # the file never had, so the node is identity-only: "this path was read, content not
            # established".
            if include_partial_reads:
                return [FileObserved(at=at, path=path, content=None, mode="read", tool_use_id=tool_use_id)], None
            return [], None

        return (
            [
                FileObserved(
                    at=at,
                    path=path,
                    content=content.encode("utf-8") if isinstance(content, str) else None,
                    mode="read",
                    tool_use_id=tool_use_id,
                )
            ],
            None,
        )

    if is_edit:
        path = result.get("filePath")
        if not path:
            return [], None

        original = result.get("originalFile")
        user_modified = bool(result.get("userModified"))
        events: List[FileObserved] = []

        if isinstance(original, str):
            events.append(
                FileObserved(
                    at=at,
                    path=path,
                    content=original.encode("utf-8"),
                    mode="read",
                    tool_use_id=tool_use_id,
                    user_modified=user_modified,
                )
            )

        if isinstance(result.get("content"), str):
            # A write gives the post-state directly.
            updated: Optional[str] = result["content"]
        else:
            updated = apply_edit(
                original if isinstance(original, str) else None,
                result.get("oldString"),
                result.get("newString"),
                bool(result.get("replaceAll")),
            )

        events.append(
            FileObserved(
                at=at,
                path=path,
                content=updated.encode("utf-8") if isinstance(updated, str) else None,
                mode="wrote",
                tool_use_id=tool_use_id,
                user_modified=user_modified,
            )
        )
        return events, path

    return [], None


__all__ = ["apply_edit", "file_events_from_result"]
