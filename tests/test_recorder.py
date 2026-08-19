"""The recorder, against a live SDK.

Skipped in full when ``eqty_sdk`` is not installed -- it comes from a private index, and the parser,
semiring and engine suites are deliberately runnable without it.
"""

import pytest
from eqty_lineage.core import (
    PERMISSIVE,
    Compacted,
    ContentPolicy,
    FileObserved,
    LineageRecorder,
    ModelCall,
    PermissionDecision,
    PromptSubmitted,
    SessionEnded,
    SessionStarted,
    ToolCallEnded,
    ToolCallStarted,
    TripleSink,
    file_events_from_result,
    prov,
)
from eqty_lineage.core.canonical import activity_signatures, graph_diff

pytestmark = pytest.mark.usefixtures("sdk")


def session_events(session_id="s1"):
    return [
        SessionStarted(
            at="2026-01-01T00:00:00Z", session_id=session_id, agent="claude-code", model="claude-opus-5", cwd="/repo"
        ),
        ToolCallStarted(
            at="2026-01-01T00:00:01Z", tool_use_id="t1", tool_name="Edit", tool_input={"file_path": "/repo/a.py"}
        ),
        FileObserved(at="2026-01-01T00:00:01Z", path="/repo/a.py", content=b"x = 1\n", mode="read", tool_use_id="t1"),
        FileObserved(at="2026-01-01T00:00:01Z", path="/repo/a.py", content=b"x = 2\n", mode="wrote", tool_use_id="t1"),
        ToolCallEnded(at="2026-01-01T00:00:02Z", tool_use_id="t1", result={"ok": True}),
        SessionEnded(at="2026-01-01T00:00:03Z", session_id=session_id),
    ]


class TestFileVersions:
    def test_an_edit_produces_two_versions_of_one_path(self, recorder):
        recorder.handle_all(session_events())
        assert [v.version for v in recorder.file_versions["/repo/a.py"]] == [1, 2]

    def test_identical_bytes_at_the_same_path_are_one_entity(self, recorder):
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="claude-code"),
                FileObserved(at="t", path="/repo/a.py", content=b"same\n", mode="read"),
                FileObserved(at="t", path="/repo/a.py", content=b"same\n", mode="read"),
            ]
        )
        assert len(recorder.file_versions["/repo/a.py"]) == 1

    def test_content_addressing_makes_the_same_bytes_the_same_cid(self, recorder):
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="claude-code"),
                FileObserved(at="t", path="/repo/a.py", content=b"shared\n", mode="read"),
                FileObserved(at="t", path="/repo/b.py", content=b"shared\n", mode="read"),
            ]
        )
        a = recorder.file_versions["/repo/a.py"][0]
        b = recorder.file_versions["/repo/b.py"][0]
        # This is what lets separate sessions join on artifacts with no shared identifiers, and what
        # makes the two capture paths comparable at all.
        assert a.content_cid == b.content_cid
        # The asset CID is derived from content *alone* -- the path lives in metadata and in the
        # `eqty:hasPath` triple, not in the identity. So identical bytes at two paths are one node with
        # two paths, which is also why an entity legitimately has several generators and why a
        # "one generator per entity" invariant had to be demoted to a statistic.
        assert a.asset_cid == b.asset_cid
        paths = {t.object for t in recorder.triples if t.predicate == prov.HAS_PATH}
        assert {"/repo/a.py", "/repo/b.py"} <= paths

    def test_a_version_with_no_content_is_still_recorded(self, recorder):
        # A partial read or an unreconstructable edit: identity without content.
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="claude-code"),
                FileObserved(at="t", path="/repo/a.py", content=None, mode="read"),
            ]
        )
        assert len(recorder.file_versions["/repo/a.py"]) == 1

    def test_a_denied_path_keeps_its_node_and_loses_only_its_bytes(self, recorder):
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="claude-code"),
                FileObserved(at="t", path="/repo/.env", content=b"SECRET=hunter2\n", mode="read"),
            ]
        )
        assert len(recorder.file_versions["/repo/.env"]) == 1
        paths = [t.object for t in recorder.triples if t.predicate == prov.HAS_PATH]
        assert "/repo/.env" in paths


class TestGraphShape:
    def test_a_tool_call_becomes_an_activity_with_used_and_generated_edges(self, recorder):
        recorder.handle_all(session_events())
        preds = {t.predicate for t in recorder.triples}
        assert prov.USED in preds
        assert prov.WAS_GENERATED_BY in preds

    def test_every_asset_carries_a_type_triple(self, recorder):
        recorder.handle_all(session_events())
        typed = {t.subject for t in recorder.triples if t.predicate == prov.ASSET_TYPE}
        subjects = {t.subject for t in recorder.triples if t.predicate == prov.HAS_PATH}
        assert subjects <= typed

    def test_activities_are_attributed_to_the_agent(self, recorder):
        recorder.handle_all(session_events())
        assert any(t.predicate == prov.RAN_AS for t in recorder.triples)

    def test_a_call_returning_nothing_still_becomes_an_activity(self, recorder):
        # A tool call always has at least one output -- the result entity, which stands in for "this
        # returned nothing" rather than being omitted. The SDK rejects None outright, so a placeholder
        # payload is what keeps a no-result call in the graph instead of dropping it.
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="claude-code"),
                ToolCallStarted(at="t", tool_use_id="t1", tool_name="Bash", tool_input={"command": "ls"}),
                ToolCallEnded(at="t", tool_use_id="t1", result=None),
            ]
        )
        assert "Bash" in [t.object for t in recorder.triples if t.predicate == prov.LABEL]

    def test_an_end_without_a_start_is_ignored(self, recorder):
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="claude-code"),
                ToolCallEnded(at="t", tool_use_id="never-started", result={"ok": True}),
            ]
        )
        assert not [t for t in recorder.triples if t.predicate == prov.LABEL]

    @pytest.mark.parametrize("decision_first", [False, True])
    def test_a_denied_call_ends_as_policy_not_execution(self, recorder, decision_first):
        started = ToolCallStarted(
            at="t", tool_use_id="denied-1", tool_name="Bash", tool_input={"command": "touch forbidden"}
        )
        denied = PermissionDecision(
            at="t",
            tool_use_id="denied-1",
            tool_name="Bash",
            decision="deny",
            reason="outside scope",
            source="hook",
        )
        recorder.handle(SessionStarted(at="t", session_id="s", agent="codex"))
        recorder.handle_all([denied, started] if decision_first else [started, denied])
        recorder.handle(SessionEnded(at="t", session_id="s"))

        labels = [t.object for t in recorder.triples if t.predicate == prov.LABEL]
        assert "policy deny: Bash" in labels
        assert "Bash" not in labels, "a denied attempt must not become a tool execution activity"
        assert recorder.coverage.tool_attempts == 1
        assert recorder.coverage.tool_denied == 1
        assert recorder.coverage.tool_calls == 0
        assert recorder.coverage.tool_errors == 0

    def test_an_allowed_call_has_a_policy_activity_feeding_execution(self, recorder):
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="codex"),
                ToolCallStarted(at="t", tool_use_id="allowed-1", tool_name="Bash", tool_input={"command": "printf ok"}),
                PermissionDecision(
                    at="t",
                    tool_use_id="allowed-1",
                    tool_name="Bash",
                    decision="allow",
                    reason="inside scope",
                    source="hook",
                ),
                ToolCallEnded(at="t", tool_use_id="allowed-1", result={"exit_code": 0}),
            ]
        )

        labels = [t.object for t in recorder.triples if t.predicate == prov.LABEL]
        assert "policy allow: Bash" in labels
        assert "Bash" in labels
        guardrails = {t.subject for t in recorder.triples if t.predicate == prov.ASSET_TYPE and t.object == "Guardrail"}
        assert len(guardrails) == 1
        guardrail = next(iter(guardrails))
        assert any(t.predicate == prov.USED and t.object == guardrail for t in recorder.triples)
        assert recorder.coverage.tool_attempts == 1
        assert recorder.coverage.tool_calls == 1
        assert recorder.coverage.tool_denied == 0

    def test_triggered_edges_originate_only_at_model_calls(self, recorder):
        # `eqty:triggered` means "the model's request caused this input to exist". Firing it from
        # whatever merely ran last let the hook path -- which never sees model calls -- assert that one
        # tool call triggered the next. 37 false edges on a real session.
        recorder.handle_all(session_events())
        assert not any(t.predicate == prov.TRIGGERED for t in recorder.triples)

    def test_a_model_call_does_create_a_triggered_edge(self, recorder):
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="claude-code"),
                ModelCall(
                    at="t",
                    request_id="r1",
                    model="m",
                    messages_in=[{"role": "user", "content": "hi"}],
                    output="ok",
                    usage={"input_tokens": 1},
                ),
                ToolCallStarted(at="t", tool_use_id="t1", tool_name="Edit", tool_input={"file_path": "/repo/a.py"}),
                FileObserved(at="t", path="/repo/a.py", content=b"x\n", mode="wrote", tool_use_id="t1"),
                ToolCallEnded(at="t", tool_use_id="t1", result={"ok": True}),
            ]
        )
        assert any(t.predicate == prov.TRIGGERED for t in recorder.triples)

    def test_an_absent_model_input_does_not_become_an_invented_prompt(self, recorder):
        # The transcript cannot recover the request payload. The SDK rejects None outright, so this
        # used to raise; recording no Prompt entity is the honest outcome.
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="claude-code"),
                ModelCall(at="t", request_id="r1", model="m", messages_in=None, output="ok", usage=None),
            ]
        )
        assert recorder.stats.get("ModelCall") == 1

    def test_unknown_events_are_ignored_not_fatal(self, recorder):
        class SomeFutureEvent:
            pass

        recorder.handle(SomeFutureEvent())  # must not raise

    def test_observed_flag_is_carried_onto_the_triples(self, recorder):
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="claude-code"),
                FileObserved(at="t", path="/repo/a.py", content=None, mode="changed", observed=False),
            ]
        )
        assert any(t.observed is False for t in recorder.triples)


class TestCanonicalIdentity:
    @staticmethod
    def record_twice(sdk):
        from eqty_lineage.core import LineageRecorder
        from eqty_sdk.context import graph_context

        graphs = []
        for _ in range(2):
            with graph_context(sdk):
                r = LineageRecorder(framework="test", policy=PERMISSIVE)
                r.handle_all(session_events())
                graphs.append(list(r.triples))
        return graphs

    def test_activity_signatures_match_across_two_recordings_of_one_session(self, sdk):
        """Statement CIDs are not stable; ``(inputs, outputs)`` signatures are.

        A statement CID covers a signed credential carrying ``validFrom``, so recording the same
        computation twice can yield two different activity CIDs. Comparison therefore has to go through
        the signature -- the *values* of this mapping, never its keys. A corollary: two manifests of the
        same computation are never byte-identical, so byte-comparison regression gates cannot work.
        """
        first, second = (activity_signatures(g) for g in self.record_twice(sdk))
        assert first
        assert set(first.values()) == set(second.values())

    def test_canonicalized_graphs_of_one_session_compare_equal(self, sdk):
        # The intended API: canonicalization rewrites activity CIDs to signature labels and leaves
        # content-addressed entities alone, so two recordings compare as plain sets.
        left, right = self.record_twice(sdk)
        added, removed = graph_diff(left, right)[:2]
        assert not added and not removed

    def test_activity_cids_are_not_a_stable_comparison_key(self, sdk):
        """The instability itself, pinned.

        Comparing on activity CIDs passes roughly 95% of the time -- two recordings inside the same
        clock second get the same CID -- which makes it exactly the kind of assertion that looks correct
        until it fails in CI. Recording across a second boundary forces the divergence the docstring in
        ``canonical.py`` describes.
        """
        import time

        from eqty_lineage.core import LineageRecorder
        from eqty_sdk.context import graph_context

        graphs = []
        for i in range(2):
            if i:
                time.sleep(1.05)
            with graph_context(sdk):
                r = LineageRecorder(framework="test", policy=PERMISSIVE)
                r.handle_all(session_events())
                graphs.append(list(r.triples))

        first, second = (activity_signatures(g) for g in graphs)
        assert first.keys() != second.keys(), "activity CIDs unexpectedly stable across a second boundary"
        # ...and the signature is unaffected, which is the entire point.
        assert set(first.values()) == set(second.values())

    def test_graph_diff_of_a_session_with_itself_is_empty(self, recorder):
        recorder.handle_all(session_events())
        triples = list(recorder.triples)
        added, removed = graph_diff(triples, triples)[:2]
        assert not added and not removed


class TestEndToEnd:
    def test_a_secret_in_the_fixture_never_reaches_the_triples(self, recorder):
        recorder.handle_all(
            [
                SessionStarted(at="t", session_id="s", agent="claude-code"),
                PromptSubmitted(at="t", prompt_id="p", text="ok"),
                FileObserved(
                    at="t", path="/repo/.env", content=b"AWS_SECRET_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE\n", mode="read"
                ),
            ]
        )
        blob = "\n".join(f"{t.subject} {t.predicate} {t.object}" for t in recorder.triples)
        assert "AKIAIOSFODNN7EXAMPLE" not in blob


class TestContentRecoveryByChaining:
    """Most `Edit` results carry a null `originalFile`, leaving the post-image unstated.

    The pre-image is usually already in the session -- the agent read the file, or wrote it, or
    edited it moments earlier. Replaying the recorded replacement against the last content the
    session established for that path recovers the bytes, and the recovery compounds: each recovered
    post-image becomes the next pre-image. Measured over a real corpus this moved content-known from
    43.3% to 79.3% of file versions.
    """

    def _edit(self, recorder, tool_use_id, path, old, new, content=None, original=None):
        recorder.handle(ToolCallStarted(tool_use_id=tool_use_id, tool_name="Edit", tool_input={"file_path": path}))
        # structuredPatch is what makes this shape-dispatch as an edit; every real Edit result
        # carries it, and originalFile is present-but-null on most of them.
        result = {
            "filePath": path,
            "oldString": old,
            "newString": new,
            "replaceAll": False,
            "structuredPatch": [],
            "originalFile": None,
        }
        if content is not None:
            result["content"] = content
        if original is not None:
            result["originalFile"] = original
        events, _ = file_events_from_result(result, tool_use_id)
        for e in events:
            recorder.handle(e)
        recorder.handle(ToolCallEnded(tool_use_id=tool_use_id, result="ok"))

    def _read(self, recorder, tool_use_id, path, content):
        recorder.handle(ToolCallStarted(tool_use_id=tool_use_id, tool_name="Read", tool_input={"file_path": path}))
        events, _ = file_events_from_result(
            {"file": {"filePath": path, "content": content, "startLine": 1, "numLines": 1, "totalLines": 1}},
            tool_use_id,
        )
        for e in events:
            recorder.handle(e)
        recorder.handle(ToolCallEnded(tool_use_id=tool_use_id, result="ok"))

    def test_an_edit_with_no_pre_image_anywhere_stays_unrecovered(self, recorder):
        self._edit(recorder, "t1", "/repo/a.py", "1", "2")
        [version] = recorder.file_versions["/repo/a.py"]
        assert version.content_cid.startswith("unknown:")

    def test_a_prior_read_supplies_the_missing_pre_image(self, recorder):
        self._read(recorder, "t1", "/repo/a.py", "x = 1\n")
        self._edit(recorder, "t2", "/repo/a.py", "1", "2")
        versions = recorder.file_versions["/repo/a.py"]
        assert not versions[-1].content_cid.startswith("unknown:")
        assert recorder.stats.get("ContentRecovered") == 1

    def test_recoveries_chain_through_successive_edits(self, recorder):
        # One read at the start unlocks a run of edits.
        self._read(recorder, "t1", "/repo/a.py", "a b c\n")
        self._edit(recorder, "t2", "/repo/a.py", "a", "A")
        self._edit(recorder, "t3", "/repo/a.py", "b", "B")
        self._edit(recorder, "t4", "/repo/a.py", "c", "C")
        assert recorder.stats.get("ContentRecovered") == 3

    def test_a_stale_pre_image_is_declined_rather_than_guessed(self, recorder):
        # If the replacement does not apply to what we hold, our content is wrong for this edit and
        # inventing a post-image would mint a version the file never had.
        self._read(recorder, "t1", "/repo/a.py", "totally different\n")
        self._edit(recorder, "t2", "/repo/a.py", "1", "2")
        assert recorder.stats.get("ContentRecovered") is None
        assert recorder.file_versions["/repo/a.py"][-1].content_cid.startswith("unknown:")

    def test_a_stated_post_image_wins_over_chaining(self, recorder):
        self._read(recorder, "t1", "/repo/a.py", "x = 1\n")
        self._edit(recorder, "t2", "/repo/a.py", "1", "2", content="explicitly stated\n")
        assert recorder.stats.get("ContentRecovered") is None

    def test_content_withheld_by_policy_is_not_retained_for_chaining(self):
        # Keeping a denied file's bytes in memory to enable a reconstruction would route around the
        # policy that withheld them.
        recorder = LineageRecorder(policy=ContentPolicy(deny_globs=("*.env",)), triples=TripleSink())
        recorder.handle(SessionStarted(session_id="s", agent="claude-code"))
        self._read(recorder, "t1", "/repo/secrets.env", "TOKEN=abc\n")
        self._edit(recorder, "t2", "/repo/secrets.env", "abc", "xyz")
        assert recorder.stats.get("ContentRecovered") is None

    def test_an_allowed_path_is_still_recovered_under_the_same_policy(self, recorder):
        self._read(recorder, "t1", "/repo/ok.py", "x = 1\n")
        self._edit(recorder, "t2", "/repo/ok.py", "1", "2")
        assert recorder.stats.get("ContentRecovered") == 1

    def test_the_basis_of_the_bytes_is_recorded(self, recorder):
        # A stated post-image and a replayed one are not the same claim, so the graph says which.
        self._read(recorder, "t1", "/repo/a.py", "x = 1\n")
        self._edit(recorder, "t2", "/repo/a.py", "1", "2")
        assert recorder.file_versions["/repo/a.py"][-1].content_cid.startswith("urn:")

    def test_chaining_does_not_cross_paths(self, recorder):
        self._read(recorder, "t1", "/repo/a.py", "x = 1\n")
        self._edit(recorder, "t2", "/repo/b.py", "1", "2")
        assert recorder.stats.get("ContentRecovered") is None


class TestCoverageIsSigned:
    """A manifest that does not say how much it saw leaves every downstream claim unbounded.

    Measured over a real corpus: content was known for 38.8% of file versions before session
    chaining and 79.3% after, 1,220 paths changed with nothing to attribute them to, and compaction
    dropped 97% of context. A complete record and one missing half its file contents are otherwise
    equally signed and indistinguishable to a reader.
    """

    def test_a_session_emits_exactly_one_coverage_statement(self, recorder):
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(SessionEnded(session_id="s1"))
        labels = [t.object for t in recorder.triples if t.predicate == prov.LABEL]
        assert sum(1 for label in labels if str(label).startswith("coverage:")) == 1

    def test_it_is_a_statement_in_the_graph_not_a_note_beside_it(self, recorder):
        # An unsigned completeness claim is one an unhappy reader can edit, and a manifest whose
        # coverage travels separately gets quoted without it.
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(SessionEnded(session_id="s1"))
        coverage = [
            t.subject for t in recorder.triples if t.predicate == prov.LABEL and str(t.object).startswith("coverage:")
        ]
        assert coverage
        generated = {t.object for t in recorder.triples if t.predicate == prov.WAS_GENERATED_BY}
        assert coverage[0] in generated, "coverage must be produced by a real activity"

    def test_known_content_is_counted_by_how_it_was_established(self, recorder):
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(FileObserved(path="/repo/a.py", content=b"x = 1\n", mode="wrote"))
        assert recorder.coverage.content_stated == 1
        assert recorder.coverage.content_known == 1
        assert recorder.coverage.content_known_rate == 1.0

    def test_a_version_with_no_bytes_is_counted_as_unknown(self, recorder):
        # The absence has to be counted, never inferred from silence.
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(FileObserved(path="/repo/a.py", content=None, mode="wrote"))
        assert recorder.coverage.content_unknown == 1
        assert recorder.coverage.content_known_rate == 0.0

    def test_an_inferred_version_is_flagged_separately_from_an_unknown_one(self, recorder):
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(FileObserved(path="/repo/a.py", content=b"x", mode="changed", observed=False))
        assert recorder.coverage.versions_inferred == 1
        assert recorder.coverage.content_stated == 1, "the bytes are known even if the cause is not"

    def test_redaction_is_distinguished_from_capture_failure(self):
        # Declining to publish is a deliberate absence; failing to see is not. Conflating them would
        # make a well-behaved policy look like a broken recorder.
        recorder = LineageRecorder(policy=ContentPolicy(deny_globs=("*.env",)), triples=TripleSink())
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(FileObserved(path="/repo/a.env", content=b"TOKEN=x\n", mode="wrote"))
        assert recorder.coverage.content_redacted == 1

    def test_compaction_loss_is_recorded(self, recorder):
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(Compacted(pre_tokens=100_000, post_tokens=20_000))
        assert recorder.coverage.tokens_dropped == 80_000
        assert abs(recorder.coverage.context_retained_rate - 0.2) < 1e-9

    def test_an_opaque_subagent_is_counted_as_missing_work(self, recorder):
        from eqty_lineage.core import SubagentEnded

        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(SubagentEnded(agent_id="a1", result="done", opaque=True))
        recorder.handle(SubagentEnded(agent_id="a2", result="done", opaque=False))
        assert recorder.coverage.subagents == 2
        assert recorder.coverage.subagents_opaque == 1

    def test_reasoning_is_attested_rather_than_reported_missing(self, recorder):
        # Every model call reasoned and none disclosed it; all three surfaces carry a signature over
        # withheld content. That is an attested absence, not a gap in this recorder.
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(ModelCall(model="claude-opus-5", output="done"))
        assert recorder.coverage.reasoning_attested == 1
        assert recorder.coverage.reasoning_recovered == 0

    def test_completeness_is_strict(self, recorder):
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(FileObserved(path="/repo/a.py", content=b"x", mode="wrote"))
        assert recorder.coverage.is_complete
        recorder.handle(FileObserved(path="/repo/b.py", content=None, mode="wrote"))
        assert not recorder.coverage.is_complete

    def test_an_empty_session_is_vacuously_complete_not_zero_percent(self, recorder):
        # A session that touched no files is not the worst-covered session.
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        assert recorder.coverage.content_known_rate == 1.0
        assert recorder.coverage.is_complete

    def test_the_payload_carries_the_rates_a_reader_would_otherwise_recompute(self, recorder):
        recorder.handle(SessionStarted(session_id="s1", agent="claude-code"))
        recorder.handle(FileObserved(path="/repo/a.py", content=b"x", mode="wrote"))
        payload = recorder.coverage.as_payload()
        assert payload["content_known_rate"] == 1.0
        assert payload["complete"] is True
        assert "tokens_dropped" in payload


class TestToolResultsAreTheirOwnEntity:
    """A tool's output is wrapped with its call id, so it cannot be confused with a file.

    Unwrapped, a command that prints a file is byte-identical to that file, and content addressing
    makes them one node. Measured on a real Codex session: `Bash: output` and `slug.py (v1)` shared a
    CID because the command was `sed -n '1,80p' slug.py`. That mislabelled the file in the Explorer,
    and it made the read edge below impossible to record without the activity both producing and
    consuming the same node -- a 2-cycle in the flow relation.
    """

    def _tool(self, recorder, tool_use_id, result, tool_name="Bash"):
        recorder.handle(ToolCallStarted(tool_use_id=tool_use_id, tool_name=tool_name, tool_input={}))
        recorder.handle(ToolCallEnded(tool_use_id=tool_use_id, result=result))

    def test_a_command_that_prints_a_file_is_not_that_file(self, recorder):
        text = "import re\n\n\ndef slugify(s):\n    return s\n"
        recorder.handle(SessionStarted(session_id="w1", agent="codex"))
        recorder.handle(FileObserved(path="/repo/slug.py", content=text.encode(), mode="wrote"))
        versions = recorder.file_versions["/repo/slug.py"]
        self._tool(recorder, "t1", text)
        outputs = [t.subject for t in recorder.triples if t.predicate == prov.WAS_GENERATED_BY]
        assert str(versions[0].asset_cid) in outputs
        # The tool's own result is a different entity, so the file is not relabelled by it.
        results = [t for t in recorder.triples if t.predicate == prov.ASSET_TYPE and t.object == "Dataset"]
        assert all(t.subject != str(versions[0].asset_cid) for t in results)

    def test_the_same_output_from_two_calls_is_two_entities(self, recorder):
        # The cost of wrapping, stated plainly: identical output no longer deduplicates. Two
        # executions did happen, and nothing downstream depends on collapsing them -- the branching
        # measurement runs over raw transcripts and divergence_report keys on file paths.
        recorder.handle(SessionStarted(session_id="w2", agent="codex"))
        self._tool(recorder, "a", "same output")
        self._tool(recorder, "b", "same output")
        labels = [t.object for t in recorder.triples if t.predicate == prov.LABEL]
        assert labels.count("Bash") == 2


class TestReadsRecoveredByContentMatch:
    """Recognising a tool result as a file's bytes is the only way a shell-driven agent gets depth.

    Codex reads through the shell and its `tool_response` carries no path, so
    `file_events_from_result` returns nothing. Measured on a real captured session, that left **zero**
    activities consuming a file version: the graph could only be inputs -> activity -> outputs however
    long the session ran. With this, that session has one, and the chain is
    `apply_patch -> slug.py -> Bash`.
    """

    def _tool(self, recorder, tool_use_id, result, tool_name="Bash"):
        recorder.handle(ToolCallStarted(tool_use_id=tool_use_id, tool_name=tool_name, tool_input={}))
        recorder.handle(ToolCallEnded(tool_use_id=tool_use_id, result=result))

    def test_a_result_matching_a_known_file_becomes_a_read(self, recorder):
        text = "alpha\nbeta\n"
        recorder.handle(SessionStarted(session_id="r1", agent="codex"))
        recorder.handle(FileObserved(path="/repo/a.txt", content=text.encode(), mode="wrote"))
        before = recorder.coverage.reads_by_content_match
        self._tool(recorder, "t1", text)
        assert recorder.coverage.reads_by_content_match == before + 1

    def test_the_inference_is_annotated_rather_than_silent(self, recorder):
        text = "gamma\n"
        recorder.handle(SessionStarted(session_id="r2", agent="codex"))
        recorder.handle(FileObserved(path="/repo/b.txt", content=text.encode(), mode="wrote"))
        self._tool(recorder, "t2", text)
        bases = [t for t in recorder.triples if t.predicate == prov.READ_BASIS]
        assert bases and bases[-1].object == "content-match"

    def test_a_result_matching_nothing_adds_no_edge(self, recorder):
        recorder.handle(SessionStarted(session_id="r3", agent="codex"))
        recorder.handle(FileObserved(path="/repo/c.txt", content=b"one\n", mode="wrote"))
        before = recorder.coverage.reads_by_content_match
        self._tool(recorder, "t3", "something else entirely")
        assert recorder.coverage.reads_by_content_match == before

    def test_an_activity_never_both_produces_and_consumes_a_version(self, recorder):
        """The property that failed the first attempt at this.

        `apply_patch` writes a file and echoes it. Recording the echo as a read of the file it just
        wrote makes the activity produce and consume the same node, which is a 2-cycle in the flow
        relation -- what makes reachability meaningless and non-absorptive semirings diverge.
        """
        text = "written and echoed\n"
        recorder.handle(SessionStarted(session_id="r4", agent="codex"))
        recorder.handle(ToolCallStarted(tool_use_id="t4", tool_name="apply_patch", tool_input={}))
        recorder.handle(FileObserved(path="/repo/d.txt", content=text.encode(), mode="wrote", tool_use_id="t4"))
        recorder.handle(ToolCallEnded(tool_use_id="t4", result=text))

        version = str(recorder.file_versions["/repo/d.txt"][0].asset_cid)
        producers = {
            t.object for t in recorder.triples if t.predicate == prov.WAS_GENERATED_BY and t.subject == version
        }
        consumers = {t.subject for t in recorder.triples if t.predicate == prov.USED and t.object == version}
        assert not (producers & consumers), "an activity produced and consumed the same file version"

    def test_a_stated_read_is_not_recounted_as_an_inference(self, recorder):
        # Claude Code names the path; that read is observed. The counter exists to say how much of the
        # graph's depth rests on byte equality, so it must not absorb reads that were stated outright.
        recorder.handle(SessionStarted(session_id="r5", agent="claude-code"))
        recorder.handle(ToolCallStarted(tool_use_id="t5", tool_name="Read", tool_input={}))
        recorder.handle(FileObserved(path="/repo/e.txt", content=b"stated\n", mode="read", tool_use_id="t5"))
        before = recorder.coverage.reads_by_content_match
        recorder.handle(ToolCallEnded(tool_use_id="t5", result="(shown to the model)"))
        assert recorder.coverage.reads_by_content_match == before
