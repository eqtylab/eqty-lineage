"""The seed queries, and the measurement that decides whether how-provenance earns its keep here.

Everything is positive Datalog. Semiring provenance under negation is not settled theory, so rules that
would want ``not`` instead take the complement as an explicit input relation -- the caller materializes
"denied paths" rather than the engine deriving "not allowed".
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .accel import as_triples, resolve
from .circuit import build_circuit
from .engine import Database, Var, annotate, atom, edb_from_triples, evaluate, rule
from .semiring import ABSORPTIVE, BOOLEAN, COUNTING, Level, Semiring, integrity

logger = logging.getLogger("eqty.lineage.query")

X, Y, Z, A, E, P = Var("X"), Var("Y"), Var("Z"), Var("A"), Var("E"), Var("P")

INFLUENCE_RULES = (
    rule(atom("influenced", X, Y), atom("edge", X, Y)),
    rule(atom("influenced", X, Z), atom("influenced", X, Y), atom("edge", Y, Z)),
)
"""Transitive closure over all lineage edges. The recursive case is where a semiring earns its keep:
``times`` composes a path, ``plus`` merges alternative paths between the same endpoints."""

VIOLATION_RULES = (
    rule(
        atom("violation", A, P),
        atom("edge_wasGeneratedBy", E, A),
        atom("edge_hasPath", E, P),
        atom("denied", P, P),
    ),
)
"""An activity that generated an entity whose path is in the denied set. ``denied`` is supplied by the
caller, which is what keeps this negation-free."""


@dataclass
class QueryResult:
    semiring: Semiring
    facts: Dict[Tuple[str, ...], Any]

    def rendered(self, limit: int = 20) -> List[Tuple[Tuple[str, ...], str]]:
        return [(key, self.semiring.render(value)) for key, value in list(self.facts.items())[:limit]]

    def __len__(self) -> int:
        return len(self.facts)


def _run(
    triples,
    rules,
    semiring: Semiring,
    extra: Optional[Database] = None,
    **kw,
) -> Database:
    edb = annotate(edb_from_triples(triples), semiring)
    if extra:
        for relation, facts in extra.items():
            edb.setdefault(relation, {}).update({k: semiring.one for k in facts})
    return evaluate(rules, edb, semiring, **kw)


def blast_radius(
    triples, source: str, semiring: Semiring = BOOLEAN, backend: str = "auto", **kw
) -> QueryResult:
    """Everything downstream of ``source``. "This input turned out to be wrong -- what is affected?"

    The archetypal recursive query, and the one that is genuinely awkward as a SQL recursive CTE once
    edges are typed and the annotation has to compose along the path.
    """
    native = resolve(backend)
    if native is not None and semiring is BOOLEAN:
        pairs = native.closure_bool(as_triples(triples), kw.get("max_iterations", 1000))
        return QueryResult(semiring, {(a, b): True for a, b in pairs if a == source})
    if native is not None and semiring is COUNTING:
        rows = native.closure_count(as_triples(triples), kw.get("max_iterations", 1000))
        return QueryResult(semiring, {(a, b): c for a, b, c in rows if a == source})

    derived = _run(triples, INFLUENCE_RULES, semiring, **kw)
    return QueryResult(semiring, {k: v for k, v in derived.get("influenced", {}).items() if k[0] == source})


def reaches(
    triples, source: str, target: str, semiring: Semiring = ABSORPTIVE, backend: str = "auto", **kw
) -> Any:
    """The annotation on ``source -> target``: under ABSORPTIVE, the minimal witness sets."""
    native = resolve(backend)
    if native is not None and semiring is ABSORPTIVE:
        sets = native.witnesses(as_triples(triples), source, target, kw.get("max_iterations", 1000))
        return frozenset(frozenset(m) for m in sets)

    derived = _run(triples, INFLUENCE_RULES, semiring, **kw)
    return derived.get("influenced", {}).get((source, target), semiring.zero)


def taint(
    triples,
    untrusted: Iterable[str],
    trusted_level: Level = Level(0, "trusted"),
    untrusted_level: Level = Level(2, "untrusted"),
    backend: str = "auto",
    **kw,
) -> QueryResult:
    """Did anything untrusted reach each artifact?

    Uses the *integrity* semiring, where ``plus`` is ``max``: one tainted derivation taints the result
    and no alternative can launder it. Running this with the confidentiality semiring (``plus = min``)
    returns "clean" as soon as any single clean path exists, which is exactly the wrong answer.
    """
    marked = set(untrusted)
    semiring = integrity(lambda _fact: trusted_level, top=untrusted_level, bottom=trusted_level)

    native = resolve(backend)
    if native is not None:
        pairs = native.taint(as_triples(triples), sorted(marked), kw.get("max_iterations", 1000))
        return QueryResult(semiring, {(a, b): untrusted_level for a, b in pairs})

    # Annotate from the *oriented* edge key, not from a fact identifier. The identifier is built from
    # stored PROV endpoints, which are reversed relative to information flow -- classifying on it marks
    # the wrong end of every edge and silently reports everything clean.
    edb = {
        relation: {
            key: (untrusted_level if key[0] in marked else trusted_level) for key in facts
        }
        for relation, facts in edb_from_triples(triples).items()
    }
    derived = evaluate(INFLUENCE_RULES, edb, semiring, **kw)
    tainted = {k: v for k, v in derived.get("influenced", {}).items() if v.rank >= untrusted_level.rank}
    return QueryResult(semiring, tainted)


def policy_violations(triples, denied_paths: Iterable[str], semiring: Semiring = ABSORPTIVE, **kw) -> QueryResult:
    """Activities that wrote to a path outside the permitted set.

    The same rule serves offline audit and the live ``PreToolUse`` hook: evaluated against the graph so
    far, a non-empty result is the reason to deny.
    """
    denied = {(p, p): True for p in denied_paths}
    derived = _run(triples, VIOLATION_RULES, semiring, extra={"denied": denied}, **kw)
    return QueryResult(semiring, derived.get("violation", {}))


# ------------------------------------------------------------------ the measurement
@dataclass
class AlternativesReport:
    """Does ``+`` ever do any work on real agent graphs?

    ``entities_multi_generator`` counts entities produced by more than one activity. It is *not* the
    right proxy for polynomial non-triviality and was a mis-measurement on the first pass: under
    transitive closure, ``+`` appears whenever two nodes are connected by more than one *path*, which
    ordinary fan-in produces even when every entity has exactly one generator.
    ``pairs_multi_derivation`` is the number that actually decides it.
    """

    reachable_pairs: int
    pairs_multi_derivation: int
    max_derivations: int
    monomials_total: int
    max_monomials: int
    entities_multi_generator: int

    @property
    def fraction_multi(self) -> float:
        return self.pairs_multi_derivation / self.reachable_pairs if self.reachable_pairs else 0.0

    def summary(self) -> str:
        return (
            f"reachable pairs {self.reachable_pairs}, "
            f"multi-derivation {self.pairs_multi_derivation} ({100 * self.fraction_multi:.1f}%), "
            f"max derivations {self.max_derivations}, max minimal-witnesses {self.max_monomials}"
        )


def measure_alternatives(triples, max_iterations: int = 1000, backend: str = "auto") -> AlternativesReport:
    """Run COUNTING and ABSORPTIVE over the closure and report how much structure ``+`` recovers.

    COUNTING gives the number of distinct derivations per reachable pair -- the direct test of whether
    the graph is a tree. ABSORPTIVE gives the number of *minimal* witness sets, which is what a
    practical how-provenance answer would actually show a user. If the first is uniformly 1, the
    machinery buys nothing; if the second is uniformly 1 while the first is large, the derivations are
    all subsumed and absorption alone is the useful part.
    """
    triples = list(triples)

    generators: Dict[str, Set[str]] = {}
    for t in triples:
        if t.predicate.endswith("wasGeneratedBy"):
            generators.setdefault(t.subject, set()).add(t.object)
    multi_generator = sum(1 for g in generators.values() if len(g) > 1)

    native = resolve(backend)
    if native is not None:
        pairs, multi, max_der, mono_total, max_mono = native.measure(as_triples(triples), max_iterations)
        return AlternativesReport(
            reachable_pairs=pairs,
            pairs_multi_derivation=multi,
            max_derivations=max_der,
            monomials_total=mono_total,
            max_monomials=max_mono,
            entities_multi_generator=multi_generator,
        )

    # COUNTING is non-idempotent, so it must be evaluated over the circuit rather than by semi-naive.
    # ABSORPTIVE is idempotent and either route agrees; it goes through the same circuit so that both
    # come from one traversal.
    circuit = build_circuit(edb_from_triples(triples), max_iterations=max_iterations)
    counts = circuit.evaluate(COUNTING)
    witnesses = circuit.evaluate(ABSORPTIVE)

    multi = sum(1 for v in counts.values() if v > 1)
    monomials = [len(v) for v in witnesses.values()]

    return AlternativesReport(
        reachable_pairs=len(counts),
        pairs_multi_derivation=multi,
        max_derivations=max(counts.values(), default=0),
        monomials_total=sum(monomials),
        max_monomials=max(monomials, default=0),
        entities_multi_generator=multi_generator,
    )


__all__ = [
    "INFLUENCE_RULES",
    "VIOLATION_RULES",
    "AlternativesReport",
    "QueryResult",
    "blast_radius",
    "measure_alternatives",
    "policy_violations",
    "reaches",
    "taint",
]
