"""Turning LangChain values into plain JSON, and the hook extractors claim values through."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from langchain_core.messages import BaseMessage

logger = logging.getLogger("eqty.langgraph")


#: Returned by a claim hook that does not want the value; a plain None cannot serve, because None is
#: itself a legitimate replacement.
UNCLAIMED = object()


def _to_jsonable(
    obj: Any,
    on_value: Optional[Callable[[Tuple[str, ...], Any], Any]] = None,
    _key_path: Tuple[str, ...] = (),
) -> Any:
    """Convert LangChain/LangGraph values into plain JSON-serializable data.

    ``on_value`` is offered every value encountered, with the sequence of dict keys that led to it. It
    returns ``UNCLAIMED`` to decline, or a replacement to substitute into the payload -- which is how a
    :class:`StateExtractor` lifts something out of the bulk state blob and into an asset of its own.
    """
    if on_value is not None:
        claimed = on_value(_key_path, obj)
        if claimed is not UNCLAIMED:
            return claimed
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BaseMessage):
        data: Dict[str, Any] = {"role": obj.type, "content": _to_jsonable(obj.content, on_value, _key_path)}
        tool_calls = getattr(obj, "tool_calls", None)
        if tool_calls:
            data["tool_calls"] = _to_jsonable(tool_calls, on_value, _key_path)
        usage = getattr(obj, "usage_metadata", None)
        if usage:
            data["usage"] = _to_jsonable(usage, on_value, _key_path)
        return data
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, on_value, (*_key_path, str(k))) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # the index is not part of the key path: an extractor claims a state key, not a position in a list
        return [_to_jsonable(item, on_value, _key_path) for item in obj]
    if type(obj).__name__ == "Command":
        # A LangGraph Command carries the state update a tool applied -- which is the whole payload of
        # DeepAgents' `task` tool, including any files the subagent wrote. Falling through to str() below
        # would record it as an opaque blob. Duck-typed on the class name because this package depends on
        # langchain-core alone and must not import langgraph.
        command = {
            field: _to_jsonable(getattr(obj, field), on_value, (*_key_path, field))
            for field in ("update", "goto", "graph", "resume")
            if getattr(obj, field, None) is not None
        }
        if command:
            return {"command": command}
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump(), on_value, _key_path)
        except Exception:  # noqa: BLE001 - best-effort serialization
            pass
    return str(obj)


# tool name -> Python source captured by @eqty_tool, read by every EqtyCallbackHandler instance
_registered_tool_sources: Dict[str, str] = {}
