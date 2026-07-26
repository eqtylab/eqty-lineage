"""EDB construction and the fixpoint.

The orientation tests matter more than they look. A closure over mixed edge directions still returns
answers -- they are just not answers to "what influenced what". There is no exception and no empty
result to notice; the query is simply wrong.
"""

import pytest

from eqty_lineage.core import Triple, prov
from eqty_lineage.query.engine import (
    ANNOTATION_PREDICATES,
    FLOW_ORIENTATION,
    NonTerminating,
    annotate,
    atom,
    edb_from_triples,
    evaluate,
    rule,
    Var,
)
from eqty_lineage.query.queries import INFLUENCE_RULES, blast_radius, reaches, taint
from eqty_lineage.query.semiring import ABSORPTIVE, BOOLEAN


def t(s, p, o):
    return Triple(subject=s, predicate=p, object=o)


class TestOrientation:
    def test_prov_predicates_are_reversed_into_flow_direction(self):
        # "activity USED entity" points backwards in time; information flows entity -> activity.
        edb = edb_from_triples([t("act", prov.USED, "input")])
        assert list(edb["edge"]) == [("input", "act")]

    def test_generated_by_is_reversed(self):
        # "entity WAS_GENERATED_BY activity": flow is activity -> entity.
        edb = edb_from_triples([t("out", prov.WAS_GENERATED_BY, "act")])
        assert list(edb["edge"]) == [("act", "out")]

    def test_triggered_is_already_in_flow_direction(self):
        edb = edb_from_triples([t("call", prov.TRIGGERED, "input")])
        assert list(edb["edge"]) == [("call", "input")]

    def test_was_invalidated_by_is_excluded(self):
        # It is the exact inverse of wasDerivedFrom, emitted for PROV completeness. Keeping both puts a
        # 2-cycle in the flow relation for every twice-edited file, which is what made non-absorptive
        # semirings diverge on real sessions.
        edb = edb_from_triples([t("v1", prov.WAS_INVALIDATED_BY, "act")])
        assert edb["edge"] == {}

    def test_derived_from_and_invalidated_by_together_do_not_make_a_cycle(self):
        triples = [
            t("v2", prov.WAS_DERIVED_FROM, "v1"),
            t("v1", prov.WAS_INVALIDATED_BY, "act"),
        ]
        edges = edb_from_triples(triples)["edge"]
        assert edges == {("v1", "v2"): edges[("v1", "v2")]}

    def test_annotation_predicates_stay_out_of_the_edge_relation(self):
        # Letting a path string into `edge` makes it a node in the closure, so a blast-radius query
        # starts returning filenames as things downstream of a file.
        triples = [t("v1", p, "value") for p in ANNOTATION_PREDICATES]
        assert edb_from_triples(triples)["edge"] == {}

    def test_annotation_predicates_are_still_reachable_in_typed_relations(self):
        edb = edb_from_triples([t("v1", prov.HAS_PATH, "/repo/a.py")])
        assert edb["edge_hasPath"] == {("v1", "/repo/a.py"): "eqty:hasPath:v1->epo/a.py"}

    def test_every_predicate_the_recorder_emits_has_a_declared_orientation(self):
        emitted = {
            v for k, v in vars(prov).items()
            if k.isupper() and isinstance(v, str) and v.startswith(("prov:", "eqty:"))
            and not k.startswith(("K_", "KIND_"))
        }
        undeclared = emitted - set(FLOW_ORIENTATION)
        # Unknown predicates default to REVERSE, which is right for PROV but a guess for anything new.
        # This is how `eqty:label` slipped in with no declared orientation.
        assert undeclared == set(), f"predicates with no declared flow orientation: {undeclared}"

    def test_orientation_can_be_overridden_per_call(self):
        edb = edb_from_triples([t("a", prov.USED, "b")], orientation={prov.USED: "forward"})
        assert list(edb["edge"]) == [("a", "b")]

    def test_a_predicate_can_be_excluded_per_call(self):
        edb = edb_from_triples([t("a", prov.USED, "b")], orientation={prov.USED: None})
        assert edb["edge"] == {}

    def test_predicate_filtering(self):
        triples = [t("a", prov.USED, "b"), t("c", prov.WAS_GENERATED_BY, "d")]
        edb = edb_from_triples(triples, predicates=[prov.USED])
        assert len(edb["edge"]) == 1


class TestFixpoint:
    """A three-step chain: read -> activity -> write."""

    TRIPLES = [
        t("act", prov.USED, "v_in"),
        t("v_out", prov.WAS_GENERATED_BY, "act"),
        t("v_out", prov.HAS_PATH, "/repo/out.py"),
    ]

    def test_transitive_closure_follows_information_flow(self):
        result = blast_radius(self.TRIPLES, "v_in", semiring=BOOLEAN, backend="python")
        assert {k[1] for k in result.facts} == {"act", "v_out"}

    def test_nothing_is_downstream_of_the_output(self):
        assert len(blast_radius(self.TRIPLES, "v_out", semiring=BOOLEAN, backend="python")) == 0

    def test_witnesses_name_the_edges_that_were_needed(self):
        witnesses = reaches(self.TRIPLES, "v_in", "v_out", semiring=ABSORPTIVE, backend="python")
        assert len(witnesses) == 1
        # Both edges are required, and the annotation-only hasPath triple is not among them.
        assert len(next(iter(witnesses))) == 2

    def test_unreachable_pairs_carry_zero(self):
        assert reaches(self.TRIPLES, "v_out", "v_in", semiring=ABSORPTIVE, backend="python") == ABSORPTIVE.zero

    def test_a_cycle_terminates_under_an_absorptive_semiring(self):
        cyclic = [t("a", prov.TRIGGERED, "b"), t("b", prov.TRIGGERED, "a")]
        result = blast_radius(cyclic, "a", semiring=ABSORPTIVE, backend="python")
        assert {k[1] for k in result.facts} == {"a", "b"}

    def test_a_non_terminating_evaluation_raises_rather_than_hanging(self):
        cyclic = [t("a", prov.TRIGGERED, "b"), t("b", prov.TRIGGERED, "a")]
        from eqty_lineage.query.semiring import COUNTING

        with pytest.raises(NonTerminating) as excinfo:
            evaluate(INFLUENCE_RULES, annotate(edb_from_triples(cyclic), COUNTING), COUNTING,
                     max_iterations=25)
        assert "not absorptive" in str(excinfo.value)

    def test_max_iterations_is_enforced(self):
        chain = [t(f"n{i+1}", prov.WAS_DERIVED_FROM, f"n{i}") for i in range(30)]
        with pytest.raises(NonTerminating):
            evaluate(INFLUENCE_RULES, annotate(edb_from_triples(chain), BOOLEAN), BOOLEAN,
                     max_iterations=3)


class TestTaint:
    """``plus = max``: one tainted derivation taints the result and alternatives cannot launder it."""

    TRIPLES = [
        t("act", prov.USED, "untrusted_in"),
        t("act", prov.USED, "clean_in"),
        t("v_out", prov.WAS_GENERATED_BY, "act"),
    ]

    def test_taint_reaches_the_output_through_the_activity(self):
        result = taint(self.TRIPLES, untrusted=["untrusted_in"], backend="python")
        assert ("untrusted_in", "v_out") in result.facts

    def test_a_clean_input_does_not_make_the_output_clean(self):
        # The bug this pins: annotating from the *stored* fact identifier rather than the oriented
        # edge key marks the wrong end of every edge and reports everything clean.
        result = taint(self.TRIPLES, untrusted=["untrusted_in"], backend="python")
        assert len(result.facts) > 0

    def test_nothing_is_tainted_when_nothing_is_marked(self):
        assert len(taint(self.TRIPLES, untrusted=[], backend="python").facts) == 0


class TestCustomRules:
    def test_a_hand_written_rule_evaluates(self):
        X, Y = Var("X"), Var("Y")
        rules = (rule(atom("touched", X, Y), atom("edge", X, Y)),)
        edb = annotate(edb_from_triples([t("act", prov.USED, "in")]), BOOLEAN)
        assert evaluate(rules, edb, BOOLEAN)["touched"] == {("in", "act"): True}

    def test_a_constant_in_a_rule_body_filters(self):
        X = Var("X")
        rules = (rule(atom("from_in", X), atom("edge", "in", X)),)
        edb = annotate(edb_from_triples([t("act", prov.USED, "in"), t("other", prov.USED, "z")]), BOOLEAN)
        assert evaluate(rules, edb, BOOLEAN)["from_in"] == {("act",): True}
