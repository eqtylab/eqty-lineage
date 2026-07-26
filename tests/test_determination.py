"""Determination provenance: which artifacts survive across N runs of the same task.

The question a signature cannot answer. Six runs of a task specified down to the algorithm produced five
distinct implementations, all passing the dictated test and disagreeing on real input. A manifest proves
you got *one* of these; robustness says which parts were never in doubt.
"""

from eqty_lineage.core import Triple, prov
from eqty_lineage.query.determination import (
    divergence_report,
    influence_support,
    qdepth,
    robust,
    support,
    union_runs,
)


def t(s, p, o):
    return Triple(subject=s, predicate=p, object=o)


def run(path_cid, root="/work", extra=()):
    """A minimal one-file run: an activity generating one version of util.py."""
    return [
        t("act", prov.USED, "prompt"),
        t(path_cid, prov.WAS_GENERATED_BY, "act"),
        t(path_cid, prov.HAS_PATH, f"{root}/util.py"),
        *extra,
    ]


class TestUnion:
    def test_identical_content_across_runs_collapses_to_one_node(self):
        # No correlation ids, no shared database: the graphs join because identical bytes hash
        # identically.
        multi = union_runs({"r1": run("v_same"), "r2": run("v_same")})
        assert support(multi, "v_same") == multi.all_runs
        assert multi.count(support(multi, "v_same")) == 2

    def test_divergent_content_carries_partial_support(self):
        multi = union_runs({"r1": run("v_a"), "r2": run("v_b")})
        assert multi.names(support(multi, "v_a")) == ["r1"]
        assert multi.names(support(multi, "v_b")) == ["r2"]

    def test_paths_are_normalised_against_each_run_root(self):
        """Replicated runs execute in *different directories*.

        Without ``roots``, ``runs/a/util.py`` and ``runs/b/util.py`` never group and the report claims
        every path appeared in exactly one run. Entity CIDs are unaffected -- content-addressed -- which
        is precisely what makes the mistake quiet instead of loud.
        """
        multi = union_runs(
            {"r1": run("v_same", root="/runs/a"), "r2": run("v_same", root="/runs/b")},
            roots={"r1": "/runs/a", "r2": "/runs/b"},
        )
        assert set(multi.paths.values()) == {"util.py"}

    def test_without_roots_the_paths_do_not_group(self):
        multi = union_runs({"r1": run("v1", root="/runs/a"), "r2": run("v2", root="/runs/b")})
        assert len({v.path for v in divergence_report(multi)}) == 2


class TestRobustness:
    def test_an_artifact_in_every_run_is_robust(self):
        multi = union_runs({"r1": run("v_same"), "r2": run("v_same"), "r3": run("v_same")})
        assert "v_same" in robust(multi)
        assert qdepth(multi, "v_same") == 0

    def test_an_artifact_in_some_runs_is_fragile(self):
        multi = union_runs({"r1": run("v_a"), "r2": run("v_b")})
        assert "v_a" not in robust(multi)
        assert qdepth(multi, "v_a") == 1

    def test_an_unknown_artifact_has_empty_support_and_depth_zero(self):
        multi = union_runs({"r1": run("v_a")})
        assert support(multi, "never-seen") == 0
        assert qdepth(multi, "never-seen") == 0


class TestDivergenceReport:
    def test_a_path_with_one_content_is_reported_robust(self):
        multi = union_runs({"r1": run("v_same"), "r2": run("v_same")})
        verdict = next(v for v in divergence_report(multi) if v.path.endswith("util.py"))
        assert verdict.is_robust
        assert verdict.distinct == 1

    def test_a_path_with_three_contents_across_six_runs(self):
        # The shape of the recorded slugify experiment: 3 distinct behaviours over 6 runs.
        runs = {f"r{i}": run(cid) for i, cid in enumerate(["a", "a", "a", "b", "b", "c"])}
        multi = union_runs(runs)
        verdict = next(v for v in divergence_report(multi) if v.path.endswith("util.py"))
        assert verdict.distinct == 3
        assert sorted(multi.count(m) for m in verdict.contents.values()) == [1, 2, 3]

    def test_superseded_versions_are_excluded_by_default(self):
        # An agent that edits a file twice in one run and once in another would otherwise read as
        # divergence. What matters is whether the runs *ended* in the same place.
        with_history = run("v_final", extra=[t("v_final", prov.WAS_DERIVED_FROM, "v_draft"),
                                             t("v_draft", prov.HAS_PATH, "/work/util.py")])
        multi = union_runs({"r1": with_history, "r2": run("v_final")})
        verdict = next(v for v in divergence_report(multi) if v.path.endswith("util.py"))
        assert verdict.is_robust

    def test_intermediate_versions_are_visible_when_asked_for(self):
        with_history = run("v_final", extra=[t("v_final", prov.WAS_DERIVED_FROM, "v_draft"),
                                             t("v_draft", prov.HAS_PATH, "/work/util.py")])
        multi = union_runs({"r1": with_history, "r2": run("v_final")})
        verdict = next(v for v in divergence_report(multi, only_final=False) if v.path.endswith("util.py"))
        assert verdict.distinct == 2


class TestInfluenceSupport:
    def test_influence_present_in_every_run_has_full_support(self):
        multi = union_runs({"r1": run("v_same"), "r2": run("v_same")})
        pairs = influence_support(multi, backend="python")
        assert pairs[("prompt", "v_same")] == multi.all_runs

    def test_a_phantom_path_spliced_across_runs_is_not_reported(self):
        """Unioning runs can connect nodes by a route no single run ever took.

        Run 1 has ``a -> b``, run 2 has ``b -> c``; the union graph contains ``a -> c`` but neither run
        does. The semiring detects this exactly -- the supports intersect to zero, which is the
        semiring's absent value -- so the pair is not a result. Reporting it would be reporting
        influence that never happened.
        """
        r1 = [t("b", prov.WAS_DERIVED_FROM, "a")]
        r2 = [t("c", prov.WAS_DERIVED_FROM, "b")]
        multi = union_runs({"r1": r1, "r2": r2})
        pairs = influence_support(multi, backend="python")
        assert ("a", "b") in pairs and ("b", "c") in pairs
        assert ("a", "c") not in pairs

    def test_a_genuine_two_step_path_within_one_run_is_reported(self):
        both = [t("b", prov.WAS_DERIVED_FROM, "a"), t("c", prov.WAS_DERIVED_FROM, "b")]
        multi = union_runs({"r1": both, "r2": both})
        pairs = influence_support(multi, backend="python")
        assert pairs[("a", "c")] == multi.all_runs
