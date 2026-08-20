"""Structural invariants over a produced graph, and the metrics that justify the query layer.

`check_invariants` is the sweep's assertion set: a graph passing it can still describe the wrong
session, but a graph failing it is wrong regardless of what it describes. It had no tests, which means
the sweep's headline "no violations across N sessions" was produced by unverified code.

`graph_stats` matters for a different reason. `entities_multi_generator` counts entities reachable by
more than one derivation, and if that were uniformly zero then agent execution graphs would be trees,
every provenance polynomial would degenerate to a monomial, and the whole semiring layer would buy
nothing over plain reachability. The number is the justification, so it needs to be right.
"""

from eqty_lineage.core import Triple, prov
from eqty_lineage.transcript.invariants import InvariantViolation, check_invariants, graph_stats


def t(s, p, o, observed=True):
    return Triple(subject=s, predicate=p, object=o, observed=observed)


def activity(name, inputs=(), outputs=()):
    """A well-formed activity: consumed something, produced something."""
    return [t(name, prov.USED, i) for i in inputs] + [t(o, prov.WAS_GENERATED_BY, name) for o in outputs]


class TestOutputWithoutInput:
    """An output with no input is a node nothing can be upstream of, which silently breaks
    blast-radius queries -- the query returns an answer, just not the right one."""

    def test_a_well_formed_activity_passes(self):
        assert check_invariants(activity("act1", ["in"], ["out"])) == []

    def test_an_activity_generating_from_nothing_is_a_violation(self):
        violations = check_invariants([t("out", prov.WAS_GENERATED_BY, "act1")])
        assert [v.invariant for v in violations] == ["output-without-input"]

    def test_the_violation_names_the_activity_and_the_count(self):
        graph = [t("o1", prov.WAS_GENERATED_BY, "act1"), t("o2", prov.WAS_GENERATED_BY, "act1")]
        [violation] = check_invariants(graph)
        assert "act1" in violation.detail
        assert "2 output(s)" in violation.detail

    def test_an_activity_that_only_consumes_is_not_a_violation(self):
        # A tool call whose result was never captured produces nothing. That is a gap in the capture,
        # not a malformed graph, and it is recorded elsewhere as an absence.
        assert check_invariants([t("act1", prov.USED, "in")]) == []

    def test_each_offending_activity_is_reported_once(self):
        graph = [
            t("o1", prov.WAS_GENERATED_BY, "bad1"),
            t("o2", prov.WAS_GENERATED_BY, "bad2"),
            *activity("good", ["in"], ["out"]),
        ]
        violations = check_invariants(graph)
        assert len(violations) == 2
        assert {v.invariant for v in violations} == {"output-without-input"}


class TestDerivationCycles:
    """A cycle is the signature of the path-keyed file identity bug this package exists to avoid:
    v2 derived from v1 derived from v2. Keying on (path, content CID) is what prevents it."""

    def test_a_version_chain_is_acyclic(self):
        graph = [t("v2", prov.WAS_DERIVED_FROM, "v1"), t("v3", prov.WAS_DERIVED_FROM, "v2")]
        assert check_invariants(graph) == []

    def test_a_two_node_cycle_is_caught(self):
        graph = [t("v2", prov.WAS_DERIVED_FROM, "v1"), t("v1", prov.WAS_DERIVED_FROM, "v2")]
        assert [v.invariant for v in check_invariants(graph)] == ["derivation-cycle"]

    def test_a_longer_cycle_is_caught(self):
        graph = [
            t("a", prov.WAS_DERIVED_FROM, "b"),
            t("b", prov.WAS_DERIVED_FROM, "c"),
            t("c", prov.WAS_DERIVED_FROM, "a"),
        ]
        [violation] = check_invariants(graph)
        assert violation.invariant == "derivation-cycle"
        assert "->" in violation.detail

    def test_a_self_loop_is_caught(self):
        assert [v.invariant for v in check_invariants([t("v", prov.WAS_DERIVED_FROM, "v")])] == ["derivation-cycle"]

    def test_a_diamond_is_not_a_cycle(self):
        # Two versions derived from a common ancestor and merged is legitimate shape, not a defect.
        graph = [
            t("b", prov.WAS_DERIVED_FROM, "a"),
            t("c", prov.WAS_DERIVED_FROM, "a"),
            t("d", prov.WAS_DERIVED_FROM, "b"),
            t("d", prov.WAS_DERIVED_FROM, "c"),
        ]
        assert check_invariants(graph) == []


class TestMultipleGeneratorsAreLegal:
    """Explicitly *not* a violation, though it would be under PROV's occurrence semantics.

    These entities are content-addressed, so they are values: four Bash calls that all return
    "Error: This command requires approval" share one CID and legitimately have four generators.
    """

    def test_one_entity_with_several_generators_passes(self):
        graph = [
            *activity("act1", ["in1"], ["shared"]),
            *activity("act2", ["in2"], ["shared"]),
        ]
        assert check_invariants(graph) == []

    def test_and_it_is_counted_rather_than_flagged(self):
        graph = [
            *activity("act1", ["in1"], ["shared"]),
            *activity("act2", ["in2"], ["shared"]),
        ]
        assert graph_stats(graph)["entities_multi_generator"] == 1


class TestGraphStats:
    def test_it_counts_entities_and_activities_from_both_edge_directions(self):
        stats = graph_stats(activity("act1", ["in"], ["out"]))
        assert stats["activities"] == 1
        assert stats["entities"] == 2
        assert stats["entities_generated"] == 1

    def test_multi_generator_is_zero_for_a_tree(self):
        # If this were uniformly zero across real sessions, how-provenance would buy nothing over
        # plain reachability and the semiring layer would not be worth its weight.
        graph = [*activity("act1", ["a"], ["b"]), *activity("act2", ["b"], ["c"])]
        assert graph_stats(graph)["entities_multi_generator"] == 0

    def test_inferred_edges_are_counted_separately(self):
        # The honesty flag: a change attributed by diffing snapshots is attribution, not observation.
        graph = [
            t("act1", prov.USED, "in"),
            t("out", prov.WAS_GENERATED_BY, "act1", observed=False),
        ]
        assert graph_stats(graph)["inferred_edges"] == 1

    def test_triples_are_counted_including_annotations(self):
        graph = [*activity("act1", ["in"], ["out"]), t("out", prov.HAS_PATH, "/repo/a.py")]
        assert graph_stats(graph)["triples"] == 3

    def test_an_empty_graph_reports_zeros_rather_than_failing(self):
        assert graph_stats([]) == {
            "triples": 0,
            "entities": 0,
            "activities": 0,
            "entities_generated": 0,
            "entities_multi_generator": 0,
            "inferred_edges": 0,
        }


class TestViolationType:
    def test_a_violation_carries_both_the_name_and_the_detail(self):
        violation = InvariantViolation("some-invariant", "why")
        assert violation.invariant == "some-invariant"
        assert violation.detail == "why"

    def test_an_empty_list_is_the_passing_result(self):
        # Not None, not True -- callers iterate it, so the passing case has to be iterable.
        assert check_invariants([]) == []
