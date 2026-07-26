"""A small semi-naive Datalog evaluator, parameterized by a semiring.

Deliberately dependency-free. The obvious alternative was CozoDB, whose Python bindings have had no
release since December 2023 and are flagged inactive -- not something to put in a package published
under contract. Writing the evaluator is also what makes the semiring parameterization possible at all:
an off-the-shelf engine answers "is this derivable", and the whole point here is to ask "how".

Scale is comfortable. A heavy agent session yields low thousands of triples, so recursion is the hard
part and volume is not.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .semiring import Semiring

logger = logging.getLogger("eqty.lineage.query")


@dataclass(frozen=True)
class Var:
    """A logic variable. ``Var("X")`` in a rule; anything else in a term position is a constant."""

    name: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name


Term = Union[Var, str]
Tuple_ = Tuple[str, ...]


@dataclass(frozen=True)
class Atom:
    relation: str
    terms: Tuple[Term, ...]

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.relation}({', '.join(str(t) for t in self.terms)})"


@dataclass(frozen=True)
class Rule:
    head: Atom
    body: Tuple[Atom, ...]

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.head} :- {', '.join(str(a) for a in self.body)}."


def atom(relation: str, *terms: Term) -> Atom:
    return Atom(relation, tuple(terms))


def rule(head: Atom, *body: Atom) -> Rule:
    return Rule(head, tuple(body))


Relation = Dict[Tuple_, Any]
Database = Dict[str, Relation]


class NonTerminating(RuntimeError):
    """Raised when a fixpoint was not reached within the iteration bound.

    Almost always one of two things: a non-absorptive semiring (COUNTING, or true N[X]) over a graph with
    a cycle, where the provenance is a genuinely infinite formal power series; or a bound set too low.
    """


def _unify(terms: Sequence[Term], values: Tuple_, binding: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Match one atom's terms against one tuple, extending ``binding``. ``None`` on conflict."""
    out = binding
    for term, value in zip(terms, values):
        if isinstance(term, Var):
            bound = out.get(term.name)
            if bound is None:
                if out is binding:
                    out = dict(binding)
                out[term.name] = value
            elif bound != value:
                return None
        elif term != value:
            return None
    return out


def _ground(terms: Sequence[Term], binding: Mapping[str, str]) -> Optional[Tuple_]:
    values: List[str] = []
    for term in terms:
        if isinstance(term, Var):
            value = binding.get(term.name)
            if value is None:
                return None
            values.append(value)
        else:
            values.append(term)
    return tuple(values)


def _match(
    body: Sequence[Atom],
    index: int,
    sources: Sequence[Database],
    binding: Dict[str, str],
    annotation: Any,
    semiring: Semiring,
) -> Iterator[Tuple[Dict[str, str], Any]]:
    """Join the body from ``index`` onward, drawing atom *i* from ``sources[i]``.

    Splitting the source per position is what makes semi-naive evaluation expressible: one position is
    restricted to the delta while the rest range over everything known.
    """
    if index >= len(body):
        yield binding, annotation
        return

    current = body[index]
    facts = sources[index].get(current.relation)
    if not facts:
        return

    grounded = _ground(current.terms, binding)
    if grounded is not None:
        # fully bound by earlier atoms -- a dict lookup instead of a scan
        found = facts.get(grounded)
        if found is not None:
            yield from _match(body, index + 1, sources, binding, semiring.times(annotation, found), semiring)
        return

    for values, fact_annotation in list(facts.items()):
        extended = _unify(current.terms, values, binding)
        if extended is None:
            continue
        yield from _match(
            body, index + 1, sources, extended, semiring.times(annotation, fact_annotation), semiring
        )


def evaluate(
    rules: Sequence[Rule],
    edb: Database,
    semiring: Semiring,
    max_iterations: int = 1000,
) -> Database:
    """Evaluate ``rules`` against ``edb`` in ``semiring``; return the derived relations.

    Semi-naive: each round, every rule is evaluated once per body position with that position drawn
    from the previous round's changes. Termination is by *annotation* equality, not tuple presence --
    a tuple can be rederived with a strictly larger annotation, and stopping at first sighting would
    silently truncate its provenance.
    """
    if not semiring.idempotent:
        # Semi-naive propagates a tuple's *merged* annotation when it changes, not the increment, so a
        # non-idempotent `plus` counts the same derivation more than once. Correcting that would need
        # the difference between old and new -- subtraction, which a semiring does not have. Use
        # `circuit.build_circuit(...).evaluate(semiring)` for these.
        logger.warning(
            "semi-naive is unsound for non-idempotent semiring %r (double counts re-derivations); "
            "evaluate it over a provenance circuit instead",
            semiring.name,
        )

    known: Database = {name: dict(facts) for name, facts in edb.items()}
    delta: Database = {name: dict(facts) for name, facts in edb.items()}

    for iteration in range(max_iterations):
        pending: Database = {}

        for r in rules:
            for position in range(len(r.body)):
                sources: List[Database] = [known] * len(r.body)
                sources[position] = delta
                if not delta.get(r.body[position].relation):
                    continue

                for binding, annotation in _match(r.body, 0, sources, {}, semiring.one, semiring):
                    head = _ground(r.head.terms, binding)
                    if head is None:
                        continue
                    current = pending.setdefault(r.head.relation, {})
                    existing = current.get(head)
                    current[head] = annotation if existing is None else semiring.plus(existing, annotation)

        changed: Database = {}
        for relation, facts in pending.items():
            target = known.setdefault(relation, {})
            for key, annotation in facts.items():
                previous = target.get(key)
                merged = annotation if previous is None else semiring.plus(previous, annotation)
                if previous is None or merged != previous:
                    target[key] = merged
                    changed.setdefault(relation, {})[key] = merged

        if not changed:
            return known
        delta = changed

    raise NonTerminating(
        f"no fixpoint after {max_iterations} iterations in semiring '{semiring.name}'"
        + ("" if semiring.absorptive else " (semiring is not absorptive; a cycle gives infinite provenance)")
    )


ANNOTATION_PREDICATES = frozenset({"eqty:hasPath", "eqty:assetType", "eqty:label"})
"""Predicates that describe a node rather than connecting two of them.

These are excluded from the generic ``edge`` relation. Letting a path string into ``edge`` would make it
a node in the transitive closure, so a blast-radius query would start returning filenames as things
downstream of a file version.
"""

REVERSE = "reverse"
FORWARD = "forward"

FLOW_ORIENTATION: Dict[str, Optional[str]] = {
    # PROV predicates are written pointing *backwards* in time -- an activity `used` its inputs, an
    # entity `wasGeneratedBy` the activity before it. Information flows the other way, so these are
    # reversed when building the flow relation.
    "prov:used": REVERSE,
    "prov:wasGeneratedBy": REVERSE,
    "prov:wasDerivedFrom": REVERSE,
    "prov:wasAttributedTo": REVERSE,
    "prov:wasAssociatedWith": REVERSE,
    "eqty:wasCompactedFrom": REVERSE,
    "eqty:authorizedBy": REVERSE,
    "eqty:ranAs": REVERSE,
    "eqty:dependsOn": REVERSE,
    "eqty:supports": REVERSE,
    # Already written in flow direction: the model's request caused the tool input to exist.
    "eqty:triggered": FORWARD,
    # Dropped: the exact inverse of wasDerivedFrom, emitted for PROV completeness. Keeping both would
    # put a 2-cycle in the flow relation for every file that was edited twice -- which is what made
    # non-absorptive semirings diverge on real sessions, and made reachability meaningless besides.
    "prov:wasInvalidatedBy": None,
    "eqty:hasPath": None,
    "eqty:assetType": None,
    "eqty:label": None,
}
"""How each predicate maps onto information flow.

Getting this wrong is silent: a closure over mixed orientations still returns answers, they are just
not the answers to "what influenced what". Unknown predicates default to ``REVERSE`` on the PROV
convention; ``None`` excludes the predicate from the flow relation entirely.
"""


def edb_from_triples(
    triples,
    predicates: Optional[Sequence[str]] = None,
    fact_id=None,
    orientation: Optional[Mapping[str, Optional[str]]] = None,
) -> Database:
    """Build an EDB from core ``Triple``s.

    ``edge(X, Y)`` is the *information-flow* relation: X influenced Y. Predicates are oriented via
    :data:`FLOW_ORIENTATION` before being added, because PROV writes edges pointing backwards in time
    and a closure over the stored direction answers a question nobody asked.

    Each triple also lands in a typed ``edge_<predicate>`` relation in its **stored** direction, so
    rules that want the literal PROV shape (``edge_wasGeneratedBy(entity, activity)``) still get it.
    """
    keep = frozenset(predicates) if predicates else None
    flow = dict(FLOW_ORIENTATION)
    if orientation:
        flow.update(orientation)

    edges: Relation = {}
    typed: Database = {}

    for t in triples:
        if keep is not None and t.predicate not in keep:
            continue
        name = fact_id(t) if fact_id is not None else f"{t.predicate}:{t.subject[-8:]}->{t.object[-8:]}"
        typed.setdefault(_relation_name(t.predicate), {})[(t.subject, t.object)] = name

        if t.predicate in ANNOTATION_PREDICATES:
            continue
        direction = flow.get(t.predicate, REVERSE)
        if direction is None:
            continue
        edges[(t.object, t.subject) if direction == REVERSE else (t.subject, t.object)] = name

    database: Database = {"edge": edges}
    database.update(typed)
    return database


def _relation_name(predicate: str) -> str:
    return "edge_" + predicate.split(":", 1)[-1]


def annotate(database: Database, semiring: Semiring) -> Database:
    """Lift an EDB of fact identifiers into ``semiring`` annotations."""
    return {
        relation: {key: semiring.lift(name) for key, name in facts.items()}
        for relation, facts in database.items()
    }


__all__ = [
    "Atom",
    "Database",
    "NonTerminating",
    "Relation",
    "Rule",
    "Term",
    "Var",
    "annotate",
    "atom",
    "edb_from_triples",
    "evaluate",
    "rule",
]
