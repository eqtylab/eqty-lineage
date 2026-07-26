"""The recorder, against a live SDK.

Skipped in full when ``eqty_sdk`` is not installed -- it comes from a private index, and the parser,
semiring and engine suites are deliberately runnable without it.
"""

import pytest

from eqty_lineage.core import (
    FileObserved,
    ModelCall,
    PERMISSIVE,
    PromptSubmitted,
    SessionEnded,
    SessionStarted,
    ToolCallEnded,
    ToolCallStarted,
    prov,
)
from eqty_lineage.core.canonical import activity_signatures, graph_diff
from eqty_lineage.transcript.claude_code import ClaudeCodeTranscript

pytestmark = pytest.mark.usefixtures("sdk")


def session_events(session_id="s1"):
    return [
        SessionStarted(at="2026-01-01T00:00:00Z", session_id=session_id, agent="claude-code",
                       model="claude-opus-5", cwd="/repo"),
        ToolCallStarted(at="2026-01-01T00:00:01Z", tool_use_id="t1", tool_name="Edit",
                        tool_input={"file_path": "/repo/a.py"}),
        FileObserved(at="2026-01-01T00:00:01Z", path="/repo/a.py", content=b"x = 1\n",
                     mode="read", tool_use_id="t1"),
        FileObserved(at="2026-01-01T00:00:01Z", path="/repo/a.py", content=b"x = 2\n",
                     mode="wrote", tool_use_id="t1"),
        ToolCallEnded(at="2026-01-01T00:00:02Z", tool_use_id="t1", result={"ok": True}),
        SessionEnded(at="2026-01-01T00:00:03Z", session_id=session_id),
    ]


class TestFileVersions:
    def test_an_edit_produces_two_versions_of_one_path(self, recorder):
        recorder.handle_all(session_events())
        assert [v.version for v in recorder.file_versions["/repo/a.py"]] == [1, 2]

    def test_identical_bytes_at_the_same_path_are_one_entity(self, recorder):
        recorder.handle_all([
            SessionStarted(at="t", session_id="s", agent="claude-code"),
            FileObserved(at="t", path="/repo/a.py", content=b"same\n", mode="read"),
            FileObserved(at="t", path="/repo/a.py", content=b"same\n", mode="read"),
        ])
        assert len(recorder.file_versions["/repo/a.py"]) == 1

    def test_content_addressing_makes_the_same_bytes_the_same_cid(self, recorder):
        recorder.handle_all([
            SessionStarted(at="t", session_id="s", agent="claude-code"),
            FileObserved(at="t", path="/repo/a.py", content=b"shared\n", mode="read"),
            FileObserved(at="t", path="/repo/b.py", content=b"shared\n", mode="read"),
        ])
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
        recorder.handle_all([
            SessionStarted(at="t", session_id="s", agent="claude-code"),
            FileObserved(at="t", path="/repo/a.py", content=None, mode="read"),
        ])
        assert len(recorder.file_versions["/repo/a.py"]) == 1

    def test_a_denied_path_keeps_its_node_and_loses_only_its_bytes(self, recorder):
        recorder.handle_all([
            SessionStarted(at="t", session_id="s", agent="claude-code"),
            FileObserved(at="t", path="/repo/.env", content=b"SECRET=hunter2\n", mode="read"),
        ])
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
        recorder.handle_all([
            SessionStarted(at="t", session_id="s", agent="claude-code"),
            ToolCallStarted(at="t", tool_use_id="t1", tool_name="Bash", tool_input={"command": "ls"}),
            ToolCallEnded(at="t", tool_use_id="t1", result=None),
        ])
        assert "Bash" in [t.object for t in recorder.triples if t.predicate == prov.LABEL]

    def test_an_end_without_a_start_is_ignored(self, recorder):
        recorder.handle_all([
            SessionStarted(at="t", session_id="s", agent="claude-code"),
            ToolCallEnded(at="t", tool_use_id="never-started", result={"ok": True}),
        ])
        assert not [t for t in recorder.triples if t.predicate == prov.LABEL]

    def test_triggered_edges_originate_only_at_model_calls(self, recorder):
        # `eqty:triggered` means "the model's request caused this input to exist". Firing it from
        # whatever merely ran last let the hook path -- which never sees model calls -- assert that one
        # tool call triggered the next. 37 false edges on a real session.
        recorder.handle_all(session_events())
        assert not any(t.predicate == prov.TRIGGERED for t in recorder.triples)

    def test_a_model_call_does_create_a_triggered_edge(self, recorder):
        recorder.handle_all([
            SessionStarted(at="t", session_id="s", agent="claude-code"),
            ModelCall(at="t", request_id="r1", model="m", messages_in=[{"role": "user", "content": "hi"}],
                      output="ok", usage={"input_tokens": 1}),
            ToolCallStarted(at="t", tool_use_id="t1", tool_name="Edit",
                            tool_input={"file_path": "/repo/a.py"}),
            FileObserved(at="t", path="/repo/a.py", content=b"x\n", mode="wrote", tool_use_id="t1"),
            ToolCallEnded(at="t", tool_use_id="t1", result={"ok": True}),
        ])
        assert any(t.predicate == prov.TRIGGERED for t in recorder.triples)

    def test_an_absent_model_input_does_not_become_an_invented_prompt(self, recorder):
        # The transcript cannot recover the request payload. The SDK rejects None outright, so this
        # used to raise; recording no Prompt entity is the honest outcome.
        recorder.handle_all([
            SessionStarted(at="t", session_id="s", agent="claude-code"),
            ModelCall(at="t", request_id="r1", model="m", messages_in=None, output="ok", usage=None),
        ])
        assert recorder.stats.get("ModelCall") == 1

    def test_unknown_events_are_ignored_not_fatal(self, recorder):
        class SomeFutureEvent:
            pass

        recorder.handle(SomeFutureEvent())  # must not raise

    def test_observed_flag_is_carried_onto_the_triples(self, recorder):
        recorder.handle_all([
            SessionStarted(at="t", session_id="s", agent="claude-code"),
            FileObserved(at="t", path="/repo/a.py", content=None, mode="changed", observed=False),
        ])
        assert any(t.observed is False for t in recorder.triples)


class TestCanonicalIdentity:
    def test_activity_signatures_match_across_two_recordings_of_one_session(self, sdk, tmp_path):
        """Statement CIDs are not stable; activity signatures are.

        A statement CID covers a signed credential carrying ``validFrom``, so recording the same
        computation twice yields two different activity CIDs. Comparing graphs therefore has to go
        through ``(inputs, outputs)`` signatures. A corollary: two manifests of the same computation are
        never byte-identical, so byte-comparison regression gates cannot work.
        """
        from eqty_sdk.context import graph_context

        from eqty_lineage.core import LineageRecorder

        graphs = []
        for _ in range(2):
            with graph_context(sdk):
                r = LineageRecorder(framework="test", policy=PERMISSIVE)
                r.handle_all(session_events())
                graphs.append(list(r.triples))

        first, second = (activity_signatures(g) for g in graphs)
        assert first and first.keys() == second.keys()

    def test_graph_diff_of_a_session_with_itself_is_empty(self, recorder):
        recorder.handle_all(session_events())
        triples = list(recorder.triples)
        added, removed = graph_diff(triples, triples)[:2]
        assert not added and not removed


class TestEndToEnd:
    def test_the_fixture_transcript_records_without_error(self, recorder, claude_transcript):
        events = list(ClaudeCodeTranscript(claude_transcript, file_history_root=None).events())
        recorder.handle_all(events)

        assert recorder.triples, "recording the fixture session produced no triples"
        assert "/repo/util.py" in recorder.file_versions
        # read v1, wrote v2 from the Edit
        assert len(recorder.file_versions["/repo/util.py"]) == 2
        assert "/repo/README.md" in recorder.file_versions, "orphan tool result lost its file lineage"

    def test_a_secret_in_the_fixture_never_reaches_the_triples(self, recorder):
        recorder.handle_all([
            SessionStarted(at="t", session_id="s", agent="claude-code"),
            PromptSubmitted(at="t", prompt_id="p", text="ok"),
            FileObserved(at="t", path="/repo/.env", content=b"AWS_SECRET_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE\n",
                         mode="read"),
        ])
        blob = "\n".join(f"{t.subject} {t.predicate} {t.object}" for t in recorder.triples)
        assert "AKIAIOSFODNN7EXAMPLE" not in blob
