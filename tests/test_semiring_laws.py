"""The algebraic laws the semirings claim, checked rather than asserted.

`semiring.py` makes precise claims in prose -- idempotent, absorptive, distributive, and one deliberate
deviation where `zero` does not annihilate. Those claims are load-bearing: absorption is what makes
recursive evaluation terminate, idempotence is what makes the fixpoint converge, and the deviation is
what lets taint propagate through clean edges. None of them was verified.

Every law is checked exhaustively over a small finite domain rather than sampled, so these tests are
deterministic and need no property-testing dependency. The domains are tiny on purpose: a law that holds
for every triple drawn from a handful of representative values and fails on larger ones would be a very
strange law.

Where a semiring deliberately breaks a law, the break is asserted too. A deviation that quietly repaired
itself would mean the reasoning built on top of it no longer applies.
"""

import itertools

import pytest

from eqty_lineage.query.semiring import (
    ABSORPTIVE,
    BOOLEAN,
    COUNTING,
    INTERNAL,
    PUBLIC,
    SECRET,
    TROPICAL,
    TRUSTED,
    UNTRUSTED,
    WHY,
    Level,
    Semiring,
    _minimalize,
    confidentiality,
    determination,
    integrity,
)

# ------------------------------------------------------------------ domains
A, B, C = "a", "b", "c"


def _families(facts, antichain_only):
    """Every set of monomials drawable from ``facts``, optionally restricted to antichains."""
    monomials = [frozenset(s) for r in range(len(facts) + 1) for s in itertools.combinations(facts, r)]
    out = []
    for r in range(len(monomials) + 1):
        for combo in itertools.combinations(monomials, r):
            family = frozenset(combo)
            if antichain_only and family != _minimalize(family, cap=0):
                continue
            out.append(family)
    return out


_CLASSIFY = {A: SECRET, B: INTERNAL, C: PUBLIC}
_TAINT = {A: UNTRUSTED, B: TRUSTED, C: TRUSTED}

# (semiring, values). Kept small: distributivity alone is cubic in the domain.
DOMAINS = [
    (BOOLEAN, [False, True]),
    (COUNTING, [0, 1, 2, 3]),
    (ABSORPTIVE, _families([A, B], antichain_only=True)),
    (WHY, _families([A, B], antichain_only=False)),
    (TROPICAL, [float("inf"), 0.0, 1.0, 2.5]),
    (confidentiality(_CLASSIFY.__getitem__), [PUBLIC, INTERNAL, SECRET]),
    (integrity(_TAINT.__getitem__), [TRUSTED, UNTRUSTED]),
    (determination(_TAINT.__contains__, 0b111), [0b000, 0b001, 0b011, 0b111]),
]

IDS = [s.name for s, _ in DOMAINS]


def _eq(semiring, x, y):
    """Level compares by rank; everything else by value."""
    if isinstance(x, Level) and isinstance(y, Level):
        return x.rank == y.rank
    return x == y


@pytest.fixture(params=DOMAINS, ids=IDS)
def algebra(request):
    return request.param


class TestIdentities:
    def test_zero_is_the_additive_identity(self, algebra):
        semiring, values = algebra
        for a in values:
            assert _eq(semiring, semiring.plus(a, semiring.zero), a), semiring.name
            assert _eq(semiring, semiring.plus(semiring.zero, a), a), semiring.name

    def test_one_is_the_multiplicative_identity(self, algebra):
        semiring, values = algebra
        for a in values:
            assert _eq(semiring, semiring.times(a, semiring.one), a), semiring.name
            assert _eq(semiring, semiring.times(semiring.one, a), a), semiring.name


class TestCommutativity:
    """Both operations must commute: the evaluator combines derivations in whatever order the
    fixpoint reaches them, and a non-commutative operation would make results order-dependent."""

    def test_plus_commutes(self, algebra):
        semiring, values = algebra
        for a, b in itertools.product(values, repeat=2):
            assert _eq(semiring, semiring.plus(a, b), semiring.plus(b, a)), semiring.name

    def test_times_commutes(self, algebra):
        semiring, values = algebra
        for a, b in itertools.product(values, repeat=2):
            assert _eq(semiring, semiring.times(a, b), semiring.times(b, a)), semiring.name


class TestAssociativity:
    def test_plus_associates(self, algebra):
        semiring, values = algebra
        for a, b, c in itertools.product(values, repeat=3):
            left = semiring.plus(semiring.plus(a, b), c)
            right = semiring.plus(a, semiring.plus(b, c))
            assert _eq(semiring, left, right), semiring.name

    def test_times_associates(self, algebra):
        semiring, values = algebra
        for a, b, c in itertools.product(values, repeat=3):
            left = semiring.times(semiring.times(a, b), c)
            right = semiring.times(a, semiring.times(b, c))
            assert _eq(semiring, left, right), semiring.name


class TestDistributivity:
    """`times` must distribute over `plus`.

    This is the law the whole construction rests on: it is what makes "annotate the base facts and
    evaluate" agree with "enumerate the derivations and combine them". Without it the annotation on a
    result would depend on the evaluation strategy rather than on the query.
    """

    def test_times_distributes_over_plus(self, algebra):
        semiring, values = algebra
        for a, b, c in itertools.product(values, repeat=3):
            left = semiring.times(a, semiring.plus(b, c))
            right = semiring.plus(semiring.times(a, b), semiring.times(a, c))
            assert _eq(semiring, left, right), f"{semiring.name}: {a} x ({b} + {c})"


class TestDeclaredProperties:
    """The `idempotent` and `absorptive` flags drive real decisions, so they must be true."""

    def test_idempotence_matches_the_flag(self, algebra):
        semiring, values = algebra
        if not semiring.idempotent:
            pytest.skip(f"{semiring.name} does not claim idempotence")
        for a in values:
            assert _eq(semiring, semiring.plus(a, a), a), semiring.name

    def test_absorption_matches_the_flag(self, algebra):
        semiring, values = algebra
        if not semiring.absorptive:
            pytest.skip(f"{semiring.name} does not claim absorption")
        for a, b in itertools.product(values, repeat=2):
            assert _eq(semiring, semiring.plus(a, semiring.times(a, b)), a), f"{semiring.name}: {a}, {b}"

    def test_counting_really_is_neither(self):
        # The engine refuses semi-naive evaluation for non-idempotent semirings because it double
        # counts re-derivations -- a bug that shipped identically in both backends. The flag is what
        # that guard reads, so it has to be honest.
        assert not COUNTING.idempotent and not COUNTING.absorptive
        assert COUNTING.plus(2, 2) != 2


class TestAnnihilation:
    """`zero` must annihilate -- except where the code says it deliberately does not."""

    def test_zero_annihilates(self, algebra):
        semiring, values = algebra
        if semiring.name == "integrity":
            pytest.skip("integrity gives up annihilation deliberately; see TestIntegrityIsALattice")
        for a in values:
            assert _eq(semiring, semiring.times(a, semiring.zero), semiring.zero), semiring.name


class TestIntegrityIsAJoinSemilattice:
    """The deliberate deviation, pinned so it cannot quietly repair itself -- and one that was not.

    `zero` and `one` are both bottom, so `zero` does not annihilate. That is the point: here `zero`
    means "carries no taint", not "has no derivation", and `taint()` annotates every clean edge with
    exactly that value. Were `zero` to annihilate, a path crossing a single clean edge would multiply
    down to trusted and taint would never propagate at all.

    Absorption fails too. The docstring used to say annihilation was the only law given up and that
    this was a bounded distributive lattice; both were wrong, and these tests are what found it.
    `plus` and `times` are the same operation, so there is no meet to pair with the join.
    """

    SEMIRING = integrity(_TAINT.__getitem__)

    def test_zero_and_one_are_the_same_element(self):
        assert self.SEMIRING.zero.rank == self.SEMIRING.one.rank == TRUSTED.rank

    def test_zero_does_not_annihilate(self):
        assert self.SEMIRING.times(UNTRUSTED, self.SEMIRING.zero).rank == UNTRUSTED.rank

    def test_taint_survives_a_clean_edge(self):
        # The concrete consequence: untrusted x clean must stay untrusted, or taint stops at the first
        # clean edge and taint() silently reports everything downstream as trusted.
        assert self.SEMIRING.times(UNTRUSTED, TRUSTED).rank == UNTRUSTED.rank

    def test_absorption_fails_and_the_flag_says_so(self):
        # plus(a, times(a, b)) = max(a, b), not a.
        assert self.SEMIRING.plus(TRUSTED, self.SEMIRING.times(TRUSTED, UNTRUSTED)).rank == UNTRUSTED.rank
        assert self.SEMIRING.absorptive is False

    def test_it_is_still_idempotent(self):
        for a in (TRUSTED, UNTRUSTED):
            assert self.SEMIRING.plus(a, a).rank == a.rank

    def test_termination_comes_from_finite_height(self):
        # Not from absorption. Both operations are monotone non-decreasing over a finite chain, so
        # iteration reaches a fixpoint within the chain's height regardless of starting point.
        for start in (TRUSTED, UNTRUSTED):
            value = start
            for _ in range(4):
                nxt = self.SEMIRING.plus(value, UNTRUSTED)
                if nxt.rank == value.rank:
                    break
                value = nxt
            assert value.rank == UNTRUSTED.rank


class TestConfidentialityIntegrityDuality:
    """Confusing the two silently gives the opposite answer, so the difference is pinned."""

    CONF = confidentiality(_CLASSIFY.__getitem__)
    INTEG = integrity(_TAINT.__getitem__)

    def test_confidentiality_takes_the_cleanest_alternative(self):
        # Derivable from public facts alone => readable as public, whatever else exists.
        assert self.CONF.plus(SECRET, PUBLIC).rank == PUBLIC.rank

    def test_integrity_takes_the_worst_alternative(self):
        # Any derivation through tainted input taints the result; alternatives cannot launder it.
        assert self.INTEG.plus(UNTRUSTED, TRUSTED).rank == UNTRUSTED.rank

    def test_they_disagree_on_the_same_inputs(self):
        # The failure mode this guards: asking "did anything untrusted reach this?" with the
        # confidentiality semiring returns "no" the moment one clean path exists.
        hi, lo = Level(2, "high"), Level(0, "low")
        assert self.CONF.plus(hi, lo).rank == 0
        assert self.INTEG.plus(hi, lo).rank == 2


class TestWitnessCapChangesSemantics:
    """`plus` stops being associative once truncation bites -- documented, now demonstrated."""

    def test_uncapped_absorptive_plus_associates(self):
        values = _families([A, B, C], antichain_only=True)[:14]
        for a, b, c in itertools.product(values, repeat=3):
            left = _minimalize(_minimalize(a | b, cap=0) | c, cap=0)
            right = _minimalize(a | _minimalize(b | c, cap=0), cap=0)
            assert left == right

    def test_capping_discards_genuine_minimal_witnesses(self):
        # The demonstrable cost: the result is a subset of the true minimal basis, not all of it. So
        # "these are the minimal witnesses" becomes "these are k of them".
        family = frozenset({frozenset({A}), frozenset({B}), frozenset({C})})
        assert _minimalize(family, cap=0) == family  # cap=0 means no truncation
        assert len(_minimalize(family, cap=2)) == 2
        assert _minimalize(family, cap=2) < family

    def test_associativity_loss_needs_ties_and_is_not_pinned_here(self):
        # The docstring says `plus` stops being associative once truncation bites. That is true in
        # general but is *not* deterministically demonstrable: with monomials of distinct lengths the
        # globally shortest always survives every grouping, so capped plus stays associative. Breaking
        # it requires ties at the cap boundary, and ties are resolved by frozenset iteration order,
        # which depends on PYTHONHASHSEED. Asserting a specific counterexample here would be a flaky
        # test. What is pinned instead: with distinct lengths, associativity does hold.
        a = frozenset({frozenset({A})})
        b = frozenset({frozenset({A, B})})
        c = frozenset({frozenset({A, B, C})})

        def capped_plus(x, y):
            return _minimalize(x | y, cap=1)

        assert capped_plus(capped_plus(a, b), c) == capped_plus(a, capped_plus(b, c))

    def test_capping_keeps_the_shortest_witnesses(self):
        # What survives truncation is the most general explanation, not an arbitrary one.
        family = frozenset({frozenset({A, B, C}), frozenset({A}), frozenset({A, B})})
        assert _minimalize(family, cap=1) == frozenset({frozenset({A})})


class TestWhyConvergesOnCycles:
    """Why(X) is not N[X], and the difference is why recursion terminates.

    Monomials are *sets*, so `times` is idempotent: traversing a cycle a second time contributes facts
    already present and yields a monomial that is already there. True N[X] tracks multiplicity with
    multisets, and that is what diverges into an infinite formal power series.
    """

    def test_times_is_idempotent_on_monomials(self):
        a = frozenset({frozenset({A, B})})
        assert WHY.times(a, a) == a

    def test_repeated_multiplication_reaches_a_fixpoint(self):
        value = frozenset({frozenset({A}), frozenset({B})})
        seen = value
        for _ in range(10):
            seen = WHY.times(seen, value)
        assert seen == WHY.times(value, value)

    def test_counting_is_why_under_the_map_sending_every_variable_to_one(self):
        # COUNTING is N[X] under that homomorphism, and non-idempotent precisely because it keeps the
        # multiplicity Why(X) discards.
        assert COUNTING.lift("anything") == 1
        assert COUNTING.times(3, 4) == 12
