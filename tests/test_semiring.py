"""Semiring laws, and the counting ground truth that caught a fault in both backends at once.

The laws are checked mechanically because a semiring that is not one produces answers that look
plausible: a broken ``times`` still returns a value, and every downstream query still returns rows.
"""

import itertools

import pytest

from eqty_lineage.query.circuit import build_circuit
from eqty_lineage.query.engine import Database, annotate, evaluate
from eqty_lineage.query.queries import INFLUENCE_RULES
from eqty_lineage.query.semiring import (
    ABSORPTIVE,
    BOOLEAN,
    COUNTING,
    TROPICAL,
    WHY,
    Level,
    confidentiality,
    determination,
    integrity,
)

FACTS = ["e1", "e2", "e3"]


def samples(semiring):
    """A handful of values in the carrier: the constants and some lifted facts and their products."""
    lifted = [semiring.lift(f) for f in FACTS]
    combined = [semiring.times(a, b) for a, b in itertools.combinations(lifted, 2)]
    return [semiring.zero, semiring.one, *lifted, *combined]


ALL_SEMIRINGS = [
    BOOLEAN,
    COUNTING,
    ABSORPTIVE,
    WHY,
    TROPICAL,
    confidentiality(lambda f: Level(1, "internal")),
    integrity(lambda f: Level(1, "suspect")),
    determination(lambda f: 0b101, all_runs=0b111),
]


@pytest.mark.parametrize("semiring", ALL_SEMIRINGS, ids=lambda s: s.name)
class TestLaws:
    def test_plus_is_commutative_and_associative(self, semiring):
        vs = samples(semiring)
        for a, b in itertools.product(vs, repeat=2):
            assert semiring.plus(a, b) == semiring.plus(b, a)
        for a, b, c in itertools.product(vs[:5], repeat=3):
            assert semiring.plus(semiring.plus(a, b), c) == semiring.plus(a, semiring.plus(b, c))

    def test_times_is_associative(self, semiring):
        for a, b, c in itertools.product(samples(semiring)[:5], repeat=3):
            assert semiring.times(semiring.times(a, b), c) == semiring.times(a, semiring.times(b, c))

    def test_zero_is_the_additive_identity(self, semiring):
        for a in samples(semiring):
            assert semiring.plus(a, semiring.zero) == a

    def test_one_is_the_multiplicative_identity(self, semiring):
        for a in samples(semiring):
            assert semiring.times(a, semiring.one) == a

    def test_zero_annihilates(self, semiring):
        if semiring.name == "integrity":
            pytest.skip("integrity gives up annihilation deliberately; see TestIntegrityIsALattice")
        for a in samples(semiring):
            assert semiring.times(a, semiring.zero) == semiring.zero

    def test_times_distributes_over_plus(self, semiring):
        for a, b, c in itertools.product(samples(semiring)[:5], repeat=3):
            assert semiring.times(a, semiring.plus(b, c)) == semiring.plus(
                semiring.times(a, b), semiring.times(a, c)
            )

    def test_the_idempotence_flag_tells_the_truth(self, semiring):
        # The engine branches on this flag to decide whether semi-naive evaluation is sound, so a
        # mislabelled semiring is not cosmetic -- it silently selects an unsound evaluator.
        actual = all(semiring.plus(a, a) == a for a in samples(semiring))
        assert semiring.idempotent == actual


class TestCountingGroundTruth:
    """``a->b, b->c, a->c, c->d``: ``a->d`` has exactly two derivations.

    Hand-counted: ``{a->c, c->d}`` and ``{a->b, b->c, c->d}``. Semi-naive reports **three**, because
    ``(a,c)`` re-enters the delta carrying its whole merged value and contributes to ``(a,d)`` a second
    time. Both the Python and the Rust backend had this bug, identically -- so the backend-agreement
    test could not see it. Only an independently derived expected value could.
    """

    EDGES = {("a", "b"): "f1", ("b", "c"): "f2", ("a", "c"): "f3", ("c", "d"): "f4"}

    def edb(self) -> Database:
        return {"edge": dict(self.EDGES)}

    def test_the_circuit_counts_derivations_correctly(self):
        values = build_circuit(self.edb()).evaluate(COUNTING)
        assert values[("a", "d")] == 2
        assert values[("a", "c")] == 2  # direct, and via b
        assert values[("a", "b")] == 1

    def test_semi_naive_over_counting_warns_that_it_is_unsound(self, caplog):
        with caplog.at_level("WARNING", logger="eqty.lineage.query"):
            evaluate(INFLUENCE_RULES, annotate(self.edb(), COUNTING), COUNTING)
        assert any("unsound" in r.getMessage() for r in caplog.records)

    def test_semi_naive_is_correct_for_idempotent_semirings(self, caplog):
        # The same fixpoint, in a semiring where re-derivation is absorbed rather than counted.
        with caplog.at_level("WARNING", logger="eqty.lineage.query"):
            derived = evaluate(INFLUENCE_RULES, annotate(self.edb(), BOOLEAN), BOOLEAN)
        assert derived["influenced"][("a", "d")] is True
        assert not caplog.records

    def test_circuit_and_semi_naive_agree_wherever_semi_naive_is_sound(self):
        circuit = build_circuit(self.edb()).evaluate(ABSORPTIVE)
        derived = evaluate(INFLUENCE_RULES, annotate(self.edb(), ABSORPTIVE), ABSORPTIVE)["influenced"]
        assert circuit == derived

    def test_the_witness_sets_for_a_to_d_are_the_two_paths(self):
        witnesses = build_circuit(self.edb()).evaluate(ABSORPTIVE)[("a", "d")]
        assert witnesses == frozenset(
            {frozenset({"f3", "f4"}), frozenset({"f1", "f2", "f4"})}
        )


class TestAbsorptive:
    def test_a_superset_witness_is_absorbed(self):
        # {e1} already suffices, so {e1, e2} carries no information and must not be retained --
        # this is what keeps the witness sets minimal rather than exponential in path count.
        a = ABSORPTIVE.lift("e1")
        both = ABSORPTIVE.times(a, ABSORPTIVE.lift("e2"))
        assert ABSORPTIVE.plus(a, both) == a

    def test_why_retains_alternatives_without_minimising(self):
        a = WHY.lift("e1")
        both = WHY.times(a, WHY.lift("e2"))
        # WHY(X) keeps both monomials -- it answers "which sets of facts sufficed", not "which
        # minimal sets". Mislabelling this as a polynomial semiring is what made an earlier version
        # claim provenance polynomials it was not computing.
        assert WHY.plus(a, both) == frozenset({frozenset({"e1"}), frozenset({"e1", "e2"})})


class TestSecurityLattices:
    def test_integrity_takes_the_worst_path(self):
        # plus = max: one tainted derivation taints the result, and no clean alternative launders it.
        s = integrity(lambda f: Level(2, "untrusted") if f == "bad" else Level(0, "trusted"))
        assert s.plus(s.lift("bad"), s.lift("good")).rank == 2

    def test_confidentiality_takes_the_best_path(self):
        # plus = min: a value is only as secret as its most public route out.
        s = confidentiality(lambda f: Level(2, "secret") if f == "s" else Level(0, "public"))
        assert s.plus(s.lift("s"), s.lift("p")).rank == 0

    def test_the_two_lattices_disagree_which_is_the_point(self):
        classify = lambda f: Level(2, "high") if f == "x" else Level(0, "low")  # noqa: E731
        hi, lo = "x", "y"
        assert integrity(classify).plus(integrity(classify).lift(hi), integrity(classify).lift(lo)).rank == 2
        assert confidentiality(classify).plus(
            confidentiality(classify).lift(hi), confidentiality(classify).lift(lo)
        ).rank == 0


class TestIntegrityIsALattice:
    """``integrity`` is a bounded distributive lattice, not a semiring, and that is deliberate.

    ``zero == one == bottom``, so ``zero`` does not annihilate. Here ``zero`` means "carries no taint",
    not "has no derivation" -- and :func:`taint` annotates every clean edge with exactly that value. If
    ``zero`` annihilated, a path crossing one clean edge would multiply down to ``trusted`` and taint
    would never propagate. Pinned as a test so nobody "fixes" it into uselessness.
    """

    def setup_method(self):
        self.trusted, self.untrusted = Level(0, "trusted"), Level(2, "untrusted")
        self.s = integrity(lambda f: self.untrusted if f == "bad" else self.trusted)

    def test_zero_does_not_annihilate(self):
        assert self.s.times(self.untrusted, self.s.zero) == self.untrusted

    def test_taint_survives_a_path_of_clean_edges(self):
        through_clean = self.s.times(self.s.times(self.untrusted, self.trusted), self.trusted)
        assert through_clean == self.untrusted

    def test_the_laws_the_evaluator_actually_needs_still_hold(self):
        # Identities, idempotence and distributivity are what make the fixpoint compose correctly.
        for a in (self.trusted, self.untrusted):
            assert self.s.plus(a, self.s.zero) == a
            assert self.s.times(a, self.s.one) == a
            assert self.s.plus(a, a) == a
        for a, b, c in itertools.product((self.trusted, self.untrusted), repeat=3):
            assert self.s.times(a, self.s.plus(b, c)) == self.s.plus(self.s.times(a, b), self.s.times(a, c))

    def test_absorption_does_not_hold_either(self):
        # This comment used to say "annihilation is the only law given up", and nothing tested it.
        # Absorption fails too, and cannot hold: `plus` and `times` are the same operation, so there
        # is no meet to pair with the join, and absorption is a statement about how the two interact.
        assert self.s.plus(self.trusted, self.s.times(self.trusted, self.untrusted)) == self.untrusted
        assert self.s.absorptive is False

    def test_termination_rests_on_finite_height_not_absorption(self):
        # Both operations are monotone non-decreasing over a finite chain, so iterating from any
        # starting value reaches a fixpoint within the height of the chain. That is what bounds the
        # evaluator here -- not the absorption property the other recursive semirings rely on.
        value = self.trusted
        for _ in range(len(("trusted", "untrusted")) + 1):
            nxt = self.s.plus(value, self.untrusted)
            if nxt == value:
                break
            value = nxt
        assert value == self.untrusted


class TestDetermination:
    """Supports as ``(2^D, union, intersection, empty, D)``, with D the observed runs."""

    def test_a_fact_carries_the_runs_it_appeared_in(self):
        s = determination({"f": 0b011}.get, all_runs=0b111)
        assert s.lift("f") == 0b011

    def test_composition_intersects_and_alternatives_union(self):
        runs = {"f1": 0b011, "f2": 0b110}
        s = determination(lambda f: runs.get(f, 0), all_runs=0b111)
        # A path needs every step, so its support is the intersection: only run 2 has both.
        assert s.times(s.lift("f1"), s.lift("f2")) == 0b010
        # Alternative paths union: either route makes the artifact exist.
        assert s.plus(s.lift("f1"), s.lift("f2")) == 0b111

    def test_full_support_means_robust(self):
        s = determination({"f": 0b111}.get, all_runs=0b111)
        assert s.lift("f") == s.one

    def test_an_unknown_fact_has_empty_support(self):
        s = determination(lambda f: 0, all_runs=0b111)
        assert s.lift("nope") == s.zero
