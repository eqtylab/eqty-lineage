"""Provenance circuits: the correct evaluator for non-idempotent semirings.

Semi-naive evaluation propagates a tuple's *merged* annotation when it changes, not the increment. For
an idempotent semiring that is harmless -- ``plus`` absorbs the repeat -- but for ``COUNTING`` it double
counts. On ``a->b, b->c, a->c, c->d`` the pair ``a->d`` has exactly two derivations
(``{a->c, c->d}`` and ``{a->b, b->c, c->d}``); semi-naive reports three, because ``(a,c)`` re-enters the
delta carrying its whole value and contributes to ``(a,d)`` a second time.

Fixing that inside semi-naive would need the *difference* between the old and new annotations, which a
semiring does not provide -- subtraction is exactly what a semiring lacks. So non-idempotent semirings
are evaluated a different way: record the derivations, then evaluate the recorded structure.

A circuit is built by *recording* an ordinary boolean fixpoint rather than by running a circuit-valued
one. Circuit nodes are syntactically distinct even when semantically equal, so a fixpoint over them
would never satisfy its termination test. Semi-naive enumerates each body combination exactly once
across the whole run, so recording as it goes captures every derivation with no extra pass, and an
instantiation generated in round *k* references only tuples known before round *k* -- the tuple graph is
stratified by round, hence acyclic, even when the edge graph is not.

The circuit is also a compact representation in its own right: the expanded polynomial for a pair can
run to millions of monomials while the circuit generating it stays polynomial, because each derivation
step contributes exactly one product term.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

from .engine import Database, NonTerminating
from .semiring import Semiring

logger = logging.getLogger("eqty.lineage.query")

Key = Tuple[str, str]
Fact = Tuple[str]  # a leaf reference, distinguished from a tuple reference by length


@dataclass
class Circuit:
    """Every derivation of every tuple, shared rather than expanded."""

    derivations: Dict[Key, List[List[Any]]] = field(default_factory=dict)

    def evaluate(self, semiring: Semiring) -> Dict[Key, Any]:
        memo: Dict[Key, Any] = {}
        for key in list(self.derivations):
            self._value(key, semiring, memo, set())
        return memo

    def _value(self, key: Key, semiring: Semiring, memo: Dict[Key, Any], active: Set[Key]) -> Any:
        if key in memo:
            return memo[key]
        if key in active:
            # Defensive only: round stratification should make this unreachable.
            return semiring.zero
        active.add(key)
        total = semiring.zero
        for body in self.derivations.get(key, ()):
            product = semiring.one
            for ref in body:
                value = semiring.lift(ref[0]) if len(ref) == 1 else self._value(ref, semiring, memo, active)
                product = semiring.times(product, value)
            total = semiring.plus(total, product)
        active.discard(key)
        memo[key] = total
        return total


def build_circuit(edb: Database, max_iterations: int = 1000) -> Circuit:
    """Record the derivations of the influence closure over ``edb['edge']``.

    ``edb`` values are fact identifiers, not semiring annotations -- the loop only needs to know whether
    a tuple is new, so it runs in the cheapest possible mode and leaves interpretation to
    :meth:`Circuit.evaluate`.
    """
    edges = edb.get("edge", {})
    by_source: Dict[str, List[Tuple[str, str]]] = {}
    for (a, b), fact in edges.items():
        by_source.setdefault(a, []).append((b, fact))

    derivations: Dict[Key, List[List[Any]]] = {}
    known: Set[Key] = set()
    delta: List[Key] = []

    for key, fact in edges.items():
        derivations.setdefault(key, []).append([(fact,)])
        if key not in known:
            known.add(key)
            delta.append(key)

    for _ in range(max_iterations):
        nxt: List[Key] = []
        for (x, y) in delta:
            for (z, fact) in by_source.get(y, ()):
                derivations.setdefault((x, z), []).append([(x, y), (fact,)])
                if (x, z) not in known:
                    known.add((x, z))
                    nxt.append((x, z))
        if not nxt:
            return Circuit(derivations=derivations)
        delta = nxt

    raise NonTerminating(f"circuit construction did not close within {max_iterations} iterations")


__all__ = ["Circuit", "build_circuit"]
