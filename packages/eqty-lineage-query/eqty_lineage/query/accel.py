"""Optional native accelerator.

``eqty-lineage-query-rs`` is a straight port of :mod:`~eqty_lineage.query.engine` -- same semi-naive
fixpoint, same semirings, same flow orientation -- measured at 33-105x on real sessions. It is optional
by design: this package works identically without it, and the Python implementation stays the reference
oracle the port is verified against.

Query functions take ``backend="auto" | "python" | "rust"``. ``"python"`` forces the reference path,
which is what the conformance test uses to compare the two.
"""

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("eqty.lineage.query")

try:  # pragma: no cover - presence depends on the install
    import eqty_lineage_query_rs as _rs
except ImportError:  # pragma: no cover
    _rs = None

AVAILABLE = _rs is not None


class BackendUnavailable(RuntimeError):
    """Raised when ``backend="rust"`` is demanded and the accelerator is not installed."""


def resolve(backend: str = "auto") -> Optional[Any]:
    """Return the native module to use, or ``None`` for the pure-Python path."""
    if backend == "python":
        return None
    if backend == "rust":
        if _rs is None:
            raise BackendUnavailable(
                "backend='rust' requested but eqty-lineage-query-rs is not installed"
            )
        return _rs
    if backend != "auto":
        raise ValueError(f"unknown backend {backend!r}; expected auto, python or rust")
    return _rs


def as_triples(triples) -> List[Tuple[str, str, str]]:
    """Marshal core Triples into the ``(s, p, o)`` tuples the accelerator accepts."""
    return [(t.subject, t.predicate, t.object) for t in triples]


__all__ = ["AVAILABLE", "BackendUnavailable", "as_triples", "resolve"]
