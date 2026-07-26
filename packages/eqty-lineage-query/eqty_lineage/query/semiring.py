"""Semirings the evaluator can be instantiated at.

The point of parameterizing by a semiring is that one evaluator answers structurally different
questions. Annotate each base fact with a variable, evaluate the query in a semiring of polynomials,
and the result's annotation records not just *whether* something is derivable but *how*: ``times`` is
joint dependency, ``plus`` is alternative derivation.

    Boolean            why-provenance -- plain reachability
    Counting           how many distinct derivations
    Absorptive         how-provenance, kept finite (the default for anything recursive)
    Why                every distinct witness set, no absorption
    Confidentiality    what clearance a result may be read at
    Integrity          what taint a result must be assumed to carry

**Recursion needs absorption.** For recursive Datalog the provenance semiring is the semiring of formal
power series, which is not finite in general: a cycle produces infinitely many derivations. Finiteness
is guaranteed for commutative, absorptive, omega-continuous semirings. ``ABSORPTIVE`` is one;
``WHY`` is idempotent and converges too, but keeps subsumed derivations; ``COUNTING`` is neither, and
the engine refuses to run past its iteration bound rather than spin.

**Confidentiality and integrity are duals, and confusing them silently gives wrong answers.** For
confidentiality, ``plus`` is ``min``: if a result is derivable from public facts alone it may be read as
public, regardless of what other derivations exist. For integrity the opposite holds -- if *any*
derivation passes through tainted input, the result must be assumed tainted, so ``plus`` is ``max``.
Asking "did anything untrusted reach this signed artifact" with the confidentiality semiring returns
"no" the moment one clean path exists, which is exactly backwards.
"""

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Generic, Optional, Set, Tuple, TypeVar

T = TypeVar("T")

Monomial = FrozenSet[str]
"""A conjunction of base facts. Sets, not multisets: for lineage "used twice" is not a distinction
worth an unbounded representation."""


@dataclass(frozen=True)
class Semiring(Generic[T]):
    """(K, plus, times, zero, one).

    ``absorptive`` marks semirings where ``a + a*b = a`` -- the property that keeps recursive evaluation
    finite. ``idempotent`` (``a + a = a``) is what makes the fixpoint converge at all.
    """

    name: str
    zero: T
    one: T
    plus: Callable[[T, T], T]
    times: Callable[[T, T], T]
    lift: Callable[[str], T]
    """Annotate a base fact, given its identifier."""
    idempotent: bool = True
    absorptive: bool = False
    describe: Optional[Callable[[T], str]] = None

    def render(self, value: T) -> str:
        return self.describe(value) if self.describe is not None else str(value)


# ------------------------------------------------------------------ boolean: why-provenance
BOOLEAN: Semiring[bool] = Semiring(
    name="boolean",
    zero=False,
    one=True,
    plus=lambda a, b: a or b,
    times=lambda a, b: a and b,
    lift=lambda _fact: True,
    idempotent=True,
    absorptive=True,
    describe=lambda v: "true" if v else "false",
)

# ------------------------------------------------------------------ counting: how many derivations
# Not idempotent and not absorptive: over a cycle the count diverges. Useful on a DAG for exactly the
# measurement this package exists to make -- how many distinct derivations a fact actually has.
COUNTING: Semiring[int] = Semiring(
    name="counting",
    zero=0,
    one=1,
    plus=lambda a, b: a + b,
    times=lambda a, b: a * b,
    lift=lambda _fact: 1,
    idempotent=False,
    absorptive=False,
    describe=str,
)


# ------------------------------------------------------------------ absorptive polynomials
WITNESS_CAP = 64
"""Maximum minimal witnesses retained per annotation; ``None`` for exact.

Not a performance tweak -- a bound on an output that is genuinely exponential. One real session has a
pair with 118,096 minimal witnesses in a 2,465-pair graph, taking 104s; at a cap of 8 the same query
takes 0.005s. Computing minimal elements of a set family provably needs exponential space even as a
ZDD, independent of variable ordering, so no representation or faster code avoids this.

Capping changes the semantics: you get *k genuine minimal witnesses*, not the complete basis, and
``plus`` stops being associative once truncation bites. Shortest monomials are kept, so what survives is
the most general explanations. The qualitative answer is unaffected -- the multi-derivation
classification is identical at every cap measured, because a pair with two witnesses still has two.
"""


def _minimalize(monomials: FrozenSet[Monomial], cap: Optional[int] = None) -> FrozenSet[Monomial]:
    """Keep only subset-minimal monomials.

    A derivation needing strictly more facts than another is subsumed by it: if ``{a}`` suffices, the
    derivation ``{a, b}`` adds nothing about *whether* the fact holds and only inflates the
    representation. Discarding it is what bounds the annotation to an antichain and makes recursion
    terminate. This is the minimal witness basis; the cost is losing "b was also involved in some
    longer route", which WHY keeps if you need it.
    """
    result: Set[Monomial] = set()
    limit = WITNESS_CAP if cap is None else cap
    for candidate in sorted(monomials, key=len):
        if not any(existing <= candidate for existing in result):
            result.add(candidate)
            if limit and len(result) >= limit:
                break
    return frozenset(result)


def _absorptive_plus(a: FrozenSet[Monomial], b: FrozenSet[Monomial]) -> FrozenSet[Monomial]:
    return _minimalize(a | b)


def _absorptive_times(a: FrozenSet[Monomial], b: FrozenSet[Monomial]) -> FrozenSet[Monomial]:
    return _minimalize(frozenset(x | y for x in a for y in b))


def _render_monomials(value: FrozenSet[Monomial]) -> str:
    if not value:
        return "0"
    if value == frozenset({frozenset()}):
        return "1"
    terms = sorted("·".join(sorted(m)) if m else "1" for m in value)
    return " + ".join(terms)


ABSORPTIVE: Semiring[FrozenSet[Monomial]] = Semiring(
    name="absorptive",
    zero=frozenset(),
    one=frozenset({frozenset()}),
    plus=_absorptive_plus,
    times=_absorptive_times,
    lift=lambda fact: frozenset({frozenset({fact})}),
    idempotent=True,
    absorptive=True,
    describe=_render_monomials,
)

# ------------------------------------------------------------------ why-provenance
# Same carrier as ABSORPTIVE, no absorption: every distinct witness set is kept.
#
# This is Why(X), *not* the free provenance polynomials N[X]. The distinction is easy to get wrong and
# matters: monomials here are sets, so `times` is idempotent -- traversing a cycle a second time
# contributes the same facts and yields a monomial that is already present. Why(X) therefore converges
# on any finite graph, cyclic or not.
#
# True N[X] tracks *multiplicity* with multisets, which is what diverges under recursion (its provenance
# is a genuinely infinite formal power series). It is not provided: the multiplicity information it adds
# over Why(X) is, for lineage, the answer to "how many times was this fact used", and COUNTING already
# gives the useful projection of it -- COUNTING is N[X] under the homomorphism mapping every variable
# to 1, and it is non-idempotent precisely because it keeps that multiplicity.
WHY: Semiring[FrozenSet[Monomial]] = Semiring(
    name="why",
    zero=frozenset(),
    one=frozenset({frozenset()}),
    plus=lambda a, b: a | b,
    times=lambda a, b: frozenset(x | y for x in a for y in b),
    lift=lambda fact: frozenset({frozenset({fact})}),
    idempotent=True,
    absorptive=False,
    describe=_render_monomials,
)


# ------------------------------------------------------------------ security lattices
@dataclass(frozen=True)
class Level:
    """A point on a total order of sensitivity. Higher rank is more restricted / more tainted."""

    rank: int
    label: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.label


PUBLIC = Level(0, "public")
INTERNAL = Level(1, "internal")
SECRET = Level(2, "secret")

UNTRUSTED = Level(2, "untrusted")
TRUSTED = Level(0, "trusted")


def confidentiality(classify: Callable[[str], Level], top: Level = SECRET, bottom: Level = PUBLIC):
    """plus = min, times = max. "What is the least restricted level this may be read at?"

    A derivation is as classified as its most classified input (times = max); if some derivation is
    cleaner, the result may be read at that cleaner level (plus = min).
    """
    return Semiring[Level](
        name="confidentiality",
        zero=top,
        one=bottom,
        plus=lambda a, b: a if a.rank <= b.rank else b,
        times=lambda a, b: a if a.rank >= b.rank else b,
        lift=classify,
        idempotent=True,
        absorptive=True,
        describe=lambda v: v.label,
    )


def integrity(classify: Callable[[str], Level], top: Level = UNTRUSTED, bottom: Level = TRUSTED):
    """plus = max, times = max. "What is the worst taint this must be assumed to carry?"

    The dual of :func:`confidentiality` and the one you want for "did anything untrusted reach this".
    A single tainted derivation taints the result, so alternatives cannot launder it.

    **This one is deliberately not a semiring**, and the deviation is load-bearing. ``zero`` and ``one``
    are both ``bottom``, so ``zero`` does not annihilate: ``times(untrusted, trusted) = untrusted``.
    That is the point. Here ``zero`` means "carries no taint", not "has no derivation" -- and
    :func:`~.queries.taint` annotates every *clean* edge with exactly that value. Were ``zero`` to
    annihilate, any path crossing a single clean edge would multiply down to ``trusted`` and taint
    would never propagate at all.

    Structurally this is a bounded distributive lattice. Both identities still hold (``max(a, bottom) =
    a`` for ``plus`` and ``times`` alike) and it is idempotent and absorptive, so the fixpoint
    terminates and distributes correctly; annihilation is the only law given up, and nothing in the
    evaluator depends on it.
    """
    return Semiring[Level](
        name="integrity",
        zero=bottom,
        one=bottom,
        plus=lambda a, b: a if a.rank >= b.rank else b,
        times=lambda a, b: a if a.rank >= b.rank else b,
        lift=classify,
        idempotent=True,
        absorptive=True,
        describe=lambda v: v.label,
    )


# ------------------------------------------------------------------ determination: which resolutions
def determination(runs_of: Callable[[str], int], all_runs: int):
    """plus = union, times = intersection over a set of *resolutions*.

    The determination semiring ``(2^D, ∪, ∩, ∅, D)``. A value is the **support** of a fact: the set of
    resolutions under which it holds. ``times`` intersects because a derivation needs all its inputs to
    hold in the same resolution; ``plus`` unions because alternative derivations each contribute their
    own.

    ``D`` here is the set of *observed* runs of a task, which is what makes this cheap. Deciding
    robustness over an implicitly-described determination space is coNP-complete; over an enumerated one
    it is a bitwise AND. Taking the resolution space as "the N runs we actually performed" sidesteps the
    hard case entirely.

    Supports are represented as bitmasks -- run *i* is bit *i* -- so ``plus`` and ``times`` are single
    machine instructions. ``one`` is the full set, not a singleton: a fact present in every run is the
    multiplicative identity, because intersecting with it changes nothing.

    Idempotent (``a | a = a``) and absorptive (``a | (a & b) = a``), so recursion terminates.
    """
    return Semiring[int](
        name="determination",
        zero=0,
        one=all_runs,
        plus=lambda a, b: a | b,
        times=lambda a, b: a & b,
        lift=runs_of,
        idempotent=True,
        absorptive=True,
        describe=lambda v: f"{bin(v).count('1')}/{bin(all_runs).count('1')} runs",
    )


# ------------------------------------------------------------------ tropical: cheapest derivation
TROPICAL: Semiring[float] = Semiring(
    name="tropical",
    zero=float("inf"),
    one=0.0,
    plus=min,
    times=lambda a, b: a + b,
    lift=lambda _fact: 1.0,
    idempotent=True,
    absorptive=True,
    describe=lambda v: "inf" if v == float("inf") else f"{v:g}",
)


ALL: Tuple[Any, ...] = (BOOLEAN, COUNTING, ABSORPTIVE, WHY, TROPICAL)

__all__ = [
    "ABSORPTIVE",
    "ALL",
    "BOOLEAN",
    "COUNTING",
    "INTERNAL",
    "WHY",
    "PUBLIC",
    "SECRET",
    "TROPICAL",
    "TRUSTED",
    "UNTRUSTED",
    "Level",
    "Monomial",
    "Semiring",
    "confidentiality",
    "determination",
    "integrity",
]
