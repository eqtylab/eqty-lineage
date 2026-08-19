"""Turning arbitrary adapter payloads into things the SDK will accept.

Generalized from the LangChain handler's ``_to_jsonable`` / ``_verbose_metadata`` pair. The behaviour
these encode is not LangChain-specific -- it is a set of hard-won rules about what the SDK constructors
and the graph explorer will and will not tolerate -- but the original was fused to ``BaseMessage``.
Here the framework-specific cases are supplied by the adapter via ``extra``.
"""

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger("eqty.lineage.core")

# kwargs the SDK asset constructors claim for themselves; caller metadata must never shadow them
RESERVED_SDK_KWARGS = frozenset({"obj", "path", "name", "description", "_store"})

# Prefix applied to metadata keys that would collide with the SDK's own kwargs. A dash, not an
# underscore: the graph explorer camel-cases keys and renders "_" as a space.
COLLISION_PREFIX = "x-"

JsonableHook = Callable[[Any], Any]
"""Adapter-supplied converter. Returns ``NotImplemented`` to defer to the default handling."""


def to_jsonable(obj: Any, extra: JsonableHook | None = None) -> Any:
    """Convert ``obj`` into plain JSON-serializable data, best effort.

    ``extra`` lets an adapter handle its own types (a LangChain ``BaseMessage``, a Claude Code content
    block) without core depending on them. It is tried first and may return ``NotImplemented`` to fall
    through.
    """
    if extra is not None:
        handled = extra(obj)
        if handled is not NotImplemented:
            return handled

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        # bytes are not JSON; decode leniently so binary payloads degrade to text rather than exploding
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, extra) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(item, extra) for item in obj]
    if hasattr(obj, "model_dump"):
        try:
            return to_jsonable(obj.model_dump(), extra)
        except Exception:  # noqa: BLE001, S110 - see below
            # Deliberately silent. This is the fallback chain that turns an arbitrary object into
            # something JSON can hold, and the next branch is the recovery. Logging here would emit a
            # line per unserializable attribute of every tool payload, which is noise in the one place
            # a recorder must stay quiet -- it is observing a session, not competing with it.
            pass
    if hasattr(obj, "__dict__"):
        try:
            return to_jsonable(vars(obj), extra)
        except Exception:  # noqa: BLE001, S110 - see below
            # Deliberately silent. This is the fallback chain that turns an arbitrary object into
            # something JSON can hold, and the next branch is the recovery. Logging here would emit a
            # line per unserializable attribute of every tool payload, which is noise in the one place
            # a recorder must stay quiet -- it is observing a session, not competing with it.
            pass
    return str(obj)


def scalar_metadata(
    fields: dict[str, Any],
    extra: JsonableHook | None = None,
    reserved: Iterable[str] = RESERVED_SDK_KWARGS,
) -> dict[str, Any]:
    """Sanitize ``fields`` into scalar values safe to unpack into an SDK asset constructor.

    Every value is run through :func:`to_jsonable` and anything that is still not a scalar is
    JSON-encoded -- the graph explorer renders each metadata value as a string, so an un-encoded dict
    shows up as ``[object Object]``. ``None`` is kept and encoded as ``"null"`` so a present-but-empty
    key stays distinguishable from an absent one.

    Keys colliding with the SDK's own constructor kwargs are prefixed until they don't, which is what
    stops ``Dataset.from_object(..., name=...)`` from raising "got multiple values for keyword
    argument". Keys are also de-duplicated against each other for the same reason.
    """
    reserved_set = frozenset(reserved)
    out: dict[str, Any] = {}

    for key, value in fields.items():
        safe_key = key
        while safe_key in reserved_set or safe_key in out:
            safe_key = f"{COLLISION_PREFIX}{safe_key}"
        jsonable = to_jsonable(value, extra)
        out[safe_key] = jsonable if isinstance(jsonable, (str, int, float, bool)) else json.dumps(jsonable)

    return out


def as_bytes(value: Any) -> bytes:
    """Coerce a payload to the bytes used for content-addressing a file version.

    File versions are keyed on the CID of their content, so this has to be stable across capture paths:
    the same file seen live and seen in a transcript must hash identically or the equivalence test fails
    and, worse, the graph grows a spurious version.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if value is None:
        return b""
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = [
    "COLLISION_PREFIX",
    "RESERVED_SDK_KWARGS",
    "JsonableHook",
    "as_bytes",
    "scalar_metadata",
    "to_jsonable",
]
