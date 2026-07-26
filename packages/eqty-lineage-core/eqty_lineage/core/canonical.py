"""Canonical relabeling, for comparing two lineage graphs.

The graph has two identity regimes and they behave differently:

*Entities are content-addressed.* An asset's CID is a hash of its payload, so the same bytes yield the
same CID on any machine, in any run, from either capture path. Entity identity is canonical for free --
which is also why comparing these graphs needs no RDF canonicalization: RDFC-1.0 exists to label blank
nodes, and there are none here.

*Statements are time-dependent.* A statement's CID covers a signed credential carrying ``validFrom``, so
the same computation recorded twice gets two different activity CIDs. Measured directly: identical
inputs, outputs, signer and context, 2.2 seconds apart, different CID.

The practical consequences are worth stating plainly. Two manifests of the same computation are never
byte-identical, so a byte-comparison regression gate cannot work. And an activity can only be compared
by what it *is* -- the set of entities it consumed and produced -- not by what it is called.

:func:`canonical_triples` rewrites activity CIDs to a hash of that signature, leaving entity CIDs alone.
Two graphs describing the same computations then compare as plain sets.
"""

import hashlib
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

from . import prov
from .triples import Triple

Signature = Tuple[FrozenSet[str], FrozenSet[str]]


def activity_signatures(triples: Iterable[Triple]) -> Dict[str, Signature]:
    """Map each activity CID to ``(inputs, outputs)`` -- the only stable thing about it."""
    inputs: Dict[str, Set[str]] = {}
    outputs: Dict[str, Set[str]] = {}

    for t in triples:
        if t.predicate == prov.USED:
            inputs.setdefault(t.subject, set()).add(t.object)
        elif t.predicate == prov.WAS_GENERATED_BY:
            outputs.setdefault(t.object, set()).add(t.subject)

    return {
        activity: (frozenset(inputs.get(activity, ())), frozenset(outputs.get(activity, ())))
        for activity in set(inputs) | set(outputs)
    }


def signature_label(signature: Signature) -> str:
    """A stable name for an activity, derived from what it consumed and produced."""
    payload = "|".join(sorted(signature[0])) + ">" + "|".join(sorted(signature[1]))
    return "act:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def canonical_triples(triples: Iterable[Triple]) -> List[Triple]:
    """Rewrite activity CIDs to signature labels; leave content-addressed entities untouched.

    Activities sharing a signature collapse to one label. That is the honest outcome: two computations
    with the same inputs and the same outputs are indistinguishable in the graph, and pretending
    otherwise would only reintroduce the time dependence this removes.
    """
    triples = list(triples)
    labels = {activity: signature_label(sig) for activity, sig in activity_signatures(triples).items()}

    return [
        Triple(
            subject=labels.get(t.subject, t.subject),
            predicate=t.predicate,
            object=labels.get(t.object, t.object),
            session_id=t.session_id,
            observed=t.observed,
        )
        for t in triples
    ]


def graph_diff(left: Iterable[Triple], right: Iterable[Triple]) -> Tuple[Set[Tuple], Set[Tuple]]:
    """``(left_only, right_only)`` over canonicalized ``(subject, predicate, object)`` triples.

    Session id and the observed flag are excluded from the comparison key -- they are properties of the
    *capture*, not of the computation, and two paths recording the same fact with different confidence
    should show up as one shared edge, not two divergent ones.
    """

    def keys(triples):
        return {(t.subject, t.predicate, t.object) for t in canonical_triples(triples)}

    lk, rk = keys(left), keys(right)
    return lk - rk, rk - lk


__all__ = ["Signature", "activity_signatures", "canonical_triples", "graph_diff", "signature_label"]
