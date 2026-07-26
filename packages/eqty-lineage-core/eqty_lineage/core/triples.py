"""The fact set emitted alongside the SDK statements.

Every edge the recorder creates is also written as ``(subject, predicate, object)`` over CIDs. This is
one artifact serving two purposes: it is an RDF/PROV-O serialization of the same graph, and it is
directly the EDB a Datalog evaluator consumes -- so the choice of query engine never has to be made at
capture time.

Only identity lives here. Content stays in the SDK's blob store, which keeps the fact set small: a heavy
session is low thousands of triples, a size at which recursive queries are the hard part and volume is
not.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger("eqty.lineage.core")


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    # Provenance about the provenance: which session asserted this edge, and whether it was observed
    # directly or reconstructed after the fact.
    session_id: Optional[str] = None
    observed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "s": self.subject,
            "p": self.predicate,
            "o": self.object,
            "session": self.session_id,
            "observed": self.observed,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Triple":
        return Triple(
            subject=data["s"],
            predicate=data["p"],
            object=data["o"],
            session_id=data.get("session"),
            observed=bool(data.get("observed", True)),
        )


class TripleSink:
    """Collects triples in memory and, optionally, appends them to a JSONL file.

    Appending rather than rewriting is deliberate: the live hook path writes one triple at a time from a
    long-running daemon, and a crash mid-session should lose the tail, not the session.
    """

    def __init__(self, path: Optional[Union[str, os.PathLike]] = None, dedupe: bool = True) -> None:
        self._triples: List[Triple] = []
        self._seen: set = set()
        self._dedupe = dedupe
        self._path = Path(path) if path is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        subject: Any,
        predicate: str,
        object: Any,  # noqa: A002 - matches the RDF term
        session_id: Optional[str] = None,
        observed: bool = True,
    ) -> Optional[Triple]:
        """Record one edge. Returns the triple, or ``None`` if it was a duplicate."""
        if subject is None or object is None:
            return None

        triple = Triple(str(subject), predicate, str(object), session_id, observed)
        key = (triple.subject, triple.predicate, triple.object)
        if self._dedupe and key in self._seen:
            return None

        self._seen.add(key)
        self._triples.append(triple)

        if self._path is not None:
            try:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(triple.to_dict(), separators=(",", ":")) + "\n")
            except OSError:
                # Losing the sidecar must never take the session down -- the SDK statements are the
                # signed artifact; these triples are a derived view that can be rebuilt from them.
                logger.warning("could not append triple to %s", self._path, exc_info=True)

        return triple

    def __iter__(self) -> Iterator[Triple]:
        return iter(self._triples)

    def __len__(self) -> int:
        return len(self._triples)

    @property
    def triples(self) -> List[Triple]:
        return list(self._triples)

    @staticmethod
    def load(path: Union[str, os.PathLike]) -> List[Triple]:
        """Read a triples file back. Malformed lines are skipped rather than fatal."""
        out: List[Triple] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Triple.from_dict(json.loads(line)))
                except (ValueError, KeyError):
                    logger.warning("skipping malformed triple at %s:%d", path, line_no)
        return out


__all__ = ["Triple", "TripleSink"]
