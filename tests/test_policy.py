"""Permission policy, and replaying a recorded trace under a different one.

The invariant that shapes both halves: **recording must not grant authority.** A hook that answers
``allow`` *overrides* the user's own settings, so a lineage recorder answering allow-by-default would
silently widen what the agent may do. Deferring is the only safe default.
"""

from eqty_lineage.agent_hooks.policy import HookPolicy
from eqty_lineage.agent_hooks.replay_policy import compare_policies, replay
from eqty_lineage.core import Triple, prov


def t(s, p, o):
    return Triple(subject=s, predicate=p, object=o)


def payload(tool="Edit", path="/repo/a.py"):
    return {"tool_name": tool, "tool_input": {"file_path": path}, "permission_mode": "default"}


class TestPolicyNeverGrantsAuthority:
    def test_a_permitted_write_defers_rather_than_allowing(self):
        policy = HookPolicy(allow_write_globs=("/repo/*",))
        # None means "defer to the agent's normal permission flow", not "allow".
        assert policy.decide(payload(path="/repo/a.py")) is None

    def test_an_empty_allow_set_means_no_restriction_not_deny_everything(self):
        # A policy object that bricked the agent on construction would be worse than no policy.
        assert HookPolicy().decide(payload()) is None

    def test_a_write_outside_the_permitted_set_is_denied(self):
        policy = HookPolicy(allow_write_globs=("/repo/*",))
        decision = policy.decide(payload(path="/etc/passwd"))
        assert decision[0] == "deny"

    def test_deny_wins_over_allow(self):
        policy = HookPolicy(allow_write_globs=("/repo/*",), deny_write_globs=("*.pem",))
        assert policy.decide(payload(path="/repo/key.pem"))[0] == "deny"

    def test_a_denied_tool_is_refused_outright(self):
        assert HookPolicy(deny_tools=("WebFetch",)).decide(payload(tool="WebFetch"))[0] == "deny"

    def test_non_write_tools_are_not_path_checked(self):
        # A Read's input path is a read, not a write; checking it against the write policy would deny
        # the agent the ability to look at what it is not allowed to change.
        policy = HookPolicy(allow_write_globs=("/repo/*",))
        assert policy.decide(payload(tool="Read", path="/etc/passwd")) is None

    def test_codex_apply_patch_counts_as_a_write(self):
        policy = HookPolicy(allow_write_globs=("/repo/*",))
        assert policy.decide(payload(tool="apply_patch", path="/etc/passwd"))[0] == "deny"

    def test_a_write_with_no_path_defers(self):
        assert HookPolicy(allow_write_globs=("/repo/*",)).decide(
            {"tool_name": "Edit", "tool_input": {}}
        ) is None

    def test_dry_run_reports_without_denying(self):
        # The honest way to roll a policy out.
        policy = HookPolicy(allow_write_globs=("/repo/*",), dry_run=True)
        decision = policy.decide(payload(path="/etc/passwd"))
        assert decision[0] == "allow"
        assert policy.violations, "a dry-run denial must still be recorded"


class TestReplay:
    """A recorded trace, reinterpreted under a different handler."""

    TRACE = [
        # act1 writes a.py; act2 reads a.py and writes b.py -- so b.py is downstream of a.py.
        t("act1", prov.USED, "prompt"),
        t("v_a", prov.WAS_GENERATED_BY, "act1"),
        t("v_a", prov.HAS_PATH, "/repo/a.py"),
        t("act1", prov.LABEL, "Edit"),
        t("act2", prov.USED, "v_a"),
        t("v_b", prov.WAS_GENERATED_BY, "act2"),
        t("v_b", prov.HAS_PATH, "/repo/b.py"),
        t("act2", prov.LABEL, "Edit"),
    ]

    def test_the_recorded_policy_refuses_nothing(self):
        outcome = replay(self.TRACE, HookPolicy(), name="as-recorded")
        assert outcome.denied == set()
        assert outcome.lost == set()

    def test_denying_a_write_removes_what_it_produced(self):
        outcome = replay(self.TRACE, HookPolicy(deny_write_globs=("/repo/b.py",)), name="no-b")
        assert outcome.denied == {"act2"}
        assert outcome.lost == {"v_b"}
        assert outcome.lost_paths == ["/repo/b.py"]

    def test_the_loss_cascades_to_everything_downstream(self):
        # Denying a.py must also remove b.py, which was derived from it. This is ordinary reachability
        # over the graph the manifest already contains.
        outcome = replay(self.TRACE, HookPolicy(deny_write_globs=("/repo/a.py",)), name="no-a")
        assert outcome.denied == {"act1"}
        assert outcome.lost == {"v_a", "v_b"}
        assert outcome.lost_paths == ["/repo/a.py", "/repo/b.py"]

    def test_surviving_and_lost_partition_the_entities(self):
        outcome = replay(self.TRACE, HookPolicy(deny_write_globs=("/repo/a.py",)), name="no-a")
        assert not (outcome.surviving & outcome.lost)

    def test_replay_uses_the_live_handler_object(self):
        """The counterfactual must not be a reimplementation of the enforcement.

        A separate implementation would let the two drift apart, which would make the answer worse
        than not having it. Pinned by observing that replay populates the policy's own violation log.
        """
        policy = HookPolicy(deny_write_globs=("/repo/a.py",))
        replay(self.TRACE, policy, name="no-a")
        assert policy.violations

    def test_a_dry_run_policy_denies_nothing_in_replay_either(self):
        outcome = replay(self.TRACE, HookPolicy(deny_write_globs=("/repo/*",), dry_run=True), name="dry")
        assert outcome.denied == set()


class TestPolicyComparison:
    """Determination over *policies* rather than runs: same algebra, different resolution space."""

    def test_artifacts_surviving_every_policy_are_robust(self):
        comparison = compare_policies(
            TestReplay.TRACE,
            {
                "as-recorded": HookPolicy(),
                "no-b": HookPolicy(deny_write_globs=("/repo/b.py",)),
            },
        )
        assert "v_a" in comparison.robust()
        assert "v_b" not in comparison.robust()

    def test_a_contingent_artifact_names_the_policies_that_permit_it(self):
        comparison = compare_policies(
            TestReplay.TRACE,
            {
                "as-recorded": HookPolicy(),
                "no-b": HookPolicy(deny_write_globs=("/repo/b.py",)),
            },
        )
        assert comparison.contingent()["/repo/b.py"] == ["as-recorded"]

    def test_everything_is_robust_when_every_policy_permits_everything(self):
        comparison = compare_policies(
            TestReplay.TRACE, {"a": HookPolicy(), "b": HookPolicy(deny_tools=("WebFetch",))}
        )
        assert comparison.contingent() == {}

    def test_the_rendered_report_states_the_soundness_limit(self):
        comparison = compare_policies(TestReplay.TRACE, {"as-recorded": HookPolicy()})
        rendered = comparison.render()
        # Replay reports which recorded artifacts become unreachable -- never what the agent would
        # have done instead. That caveat belongs in the output, not only in the docs.
        assert "would have done instead" in rendered
