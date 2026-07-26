"""The Python and Rust evaluators must agree.

**What this test cannot do.** Two implementations of the same algorithm share its bugs. Semi-naive
double-counting was present in both backends, identically, and this suite passed throughout -- the
classic correlated-fault result for N-version comparison. Agreement here means "the port is faithful",
never "the answer is right". Independently derived expected values live in ``test_semiring.py``.
"""

import pytest

from eqty_lineage.core import Triple, prov
from eqty_lineage.query import RUST_AVAILABLE
from eqty_lineage.query.accel import BackendUnavailable, resolve
from eqty_lineage.query.determination import influence_support, union_runs
from eqty_lineage.query.queries import blast_radius, reaches, taint
from eqty_lineage.query.semiring import ABSORPTIVE, BOOLEAN

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="native accelerator not built")


def t(s, p, o):
    return Triple(subject=s, predicate=p, object=o)


def graph(n=12):
    """A chain with shortcuts, so there are genuinely several paths between distant pairs."""
    triples = []
    for i in range(n):
        triples.append(t(f"act{i}", prov.USED, f"v{i}"))
        triples.append(t(f"v{i+1}", prov.WAS_GENERATED_BY, f"act{i}"))
        triples.append(t(f"v{i+1}", prov.HAS_PATH, f"/repo/f{i}.py"))
        if i >= 2:
            triples.append(t(f"v{i+1}", prov.WAS_DERIVED_FROM, f"v{i-2}"))
    return triples


class TestAgreement:
    def test_reachability_agrees(self):
        py = blast_radius(graph(), "v0", semiring=BOOLEAN, backend="python")
        rs = blast_radius(graph(), "v0", semiring=BOOLEAN, backend="rust")
        assert set(py.facts) == set(rs.facts)
        assert len(py.facts) > 0

    def test_witness_sets_agree(self):
        py = reaches(graph(), "v0", "v6", semiring=ABSORPTIVE, backend="python")
        rs = reaches(graph(), "v0", "v6", semiring=ABSORPTIVE, backend="rust")
        assert py == rs
        assert py != ABSORPTIVE.zero

    def test_taint_agrees(self):
        py = taint(graph(), untrusted=["v0"], backend="python")
        rs = taint(graph(), untrusted=["v0"], backend="rust")
        assert set(py.facts) == set(rs.facts)

    def test_determination_supports_agree(self):
        multi = union_runs({"r1": graph(6), "r2": graph(6), "r3": graph(5)})
        assert influence_support(multi, backend="python") == influence_support(multi, backend="rust")

    def test_agreement_holds_on_an_empty_graph(self):
        assert (
            set(blast_radius([], "v0", semiring=BOOLEAN, backend="python").facts)
            == set(blast_radius([], "v0", semiring=BOOLEAN, backend="rust").facts)
            == set()
        )

    def test_agreement_holds_across_a_cycle(self):
        cyclic = [t("a", prov.TRIGGERED, "b"), t("b", prov.TRIGGERED, "c"), t("c", prov.TRIGGERED, "a")]
        py = blast_radius(cyclic, "a", semiring=BOOLEAN, backend="python")
        rs = blast_radius(cyclic, "a", semiring=BOOLEAN, backend="rust")
        assert set(py.facts) == set(rs.facts) == {("a", "a"), ("a", "b"), ("a", "c")}

    def test_orientation_is_shared_not_reimplemented(self):
        """A predicate the Rust side does not orient the same way is a silent divergence.

        This is how the backends drifted when ``eqty:label`` was introduced: no error, just different
        answers on graphs containing the new predicate.
        """
        for predicate in (prov.USED, prov.WAS_GENERATED_BY, prov.WAS_DERIVED_FROM, prov.TRIGGERED,
                          prov.WAS_INVALIDATED_BY, prov.HAS_PATH, prov.LABEL, prov.ASSET_TYPE,
                          prov.RAN_AS, prov.AUTHORIZED_BY, prov.WAS_COMPACTED_FROM):
            triples = [t("x", predicate, "y")]
            py = set(blast_radius(triples, "x", semiring=BOOLEAN, backend="python").facts)
            rs = set(blast_radius(triples, "x", semiring=BOOLEAN, backend="rust").facts)
            assert py == rs, f"backends disagree on orientation of {predicate}"


class TestWitnessCap:
    """Absorptive witness sets blow up structurally, not accidentally.

    A 2,465-pair graph produced 118,096 minimal witnesses. The blowup is a property of the answer, not
    of the representation -- ZDDs do not escape it -- so the cap is the honest fix: bound the witnesses
    retained, keep the multi-derivation classification exact.
    """

    def test_the_cap_is_settable_and_bounds_the_result(self):
        native = resolve("rust")
        native.set_witness_cap(4)
        try:
            witnesses = reaches(graph(14), "v0", "v10", semiring=ABSORPTIVE, backend="rust")
            assert len(witnesses) <= 4
        finally:
            native.set_witness_cap(64)

    def test_capping_preserves_whether_more_than_one_derivation_exists(self):
        native = resolve("rust")
        uncapped = reaches(graph(14), "v0", "v10", semiring=ABSORPTIVE, backend="rust")
        native.set_witness_cap(2)
        try:
            capped = reaches(graph(14), "v0", "v10", semiring=ABSORPTIVE, backend="rust")
            # The classification that queries actually use -- "was there more than one way?" -- is what
            # the cap must not disturb.
            assert (len(uncapped) > 1) == (len(capped) > 1)
        finally:
            native.set_witness_cap(64)


class TestBackendSelection:
    def test_an_unknown_backend_name_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            resolve("nonexistent-backend")

    def test_requesting_rust_when_it_is_absent_raises_rather_than_falling_back(self, monkeypatch):
        # A silent fallback would turn "the accelerator failed to load" into "the suite got slower",
        # which is exactly the kind of regression that survives a release.
        from eqty_lineage.query import accel

        monkeypatch.setattr(accel, "_rs", None)
        with pytest.raises(BackendUnavailable):
            accel.resolve("rust")

    def test_auto_falls_back_silently_which_is_the_point_of_auto(self, monkeypatch):
        from eqty_lineage.query import accel

        monkeypatch.setattr(accel, "_rs", None)
        assert accel.resolve("auto") is None

    def test_python_is_always_selectable(self):
        assert resolve("python") is None

    def test_auto_selects_the_accelerator_when_present(self):
        assert resolve("auto") is not None
