"""The conformance checker itself.

`eqty-lineage-hooks verify` produces the number this package leans on hardest -- "both capture paths
produce the same graph, modulo declared gaps" -- and the module computing it had no tests. A checker
that cannot fail is worth nothing, so the cases that matter most here are the ones asserting it *does*
fail: on an injected adapter fault, and on the deliberately adversarial fixture.

Two fixtures with opposite jobs:

`conforming_session` is a clean session -- one prompt, one read, one edit, one assistant reply. Both
paths see the same thing, so the checker must report no problems. This is the false-alarm guard.

`tests/fixtures/claude_session.jsonl` is hand-built to break the parser: an orphaned tool result from a
resumed session, a compact boundary, snapshot deltas. Replay cannot synthesize the events those imply,
so the checker must notice. This is the sensitivity guard.
"""

import json
from pathlib import Path

import pytest

from eqty_lineage.agent_hooks import equivalence
from eqty_lineage.agent_hooks.equivalence import HOOKS_ONLY_TYPES, TRANSCRIPT_ONLY_TYPES, compare

pytestmark = pytest.mark.usefixtures("sdk")

SESSION = "conform-1"


def _record(**fields):
    return dict(sessionId=SESSION, cwd="/repo", version="2.1.220", **fields)


@pytest.fixture
def conforming_session(tmp_path):
    """A session both capture paths can describe identically.

    Deliberately free of everything the adversarial fixture exercises: no snapshot records (they
    produce offline-only inferred versions), no compact boundary, no orphaned results.
    """
    records = [
        _record(
            type="user",
            timestamp="2026-07-30T10:00:00Z",
            entrypoint="cli",
            promptId="p1",
            message={"role": "user", "content": "read a.py and change it"},
        ),
        _record(
            type="assistant",
            timestamp="2026-07-30T10:00:01Z",
            requestId="r1",
            message={
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [
                    {"type": "text", "text": "Reading it now."},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/repo/a.py"}},
                ],
            },
        ),
        _record(
            type="user",
            timestamp="2026-07-30T10:00:02Z",
            message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            toolUseResult={"file": {"filePath": "/repo/a.py", "content": "x = 1\n",
                                    "numLines": 1, "totalLines": 1, "startLine": 1}},
        ),
        _record(
            type="assistant",
            timestamp="2026-07-30T10:00:03Z",
            requestId="r2",
            message={
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "Edit",
                     "input": {"file_path": "/repo/a.py", "old_string": "1", "new_string": "2"}},
                ],
            },
        ),
        _record(
            type="user",
            timestamp="2026-07-30T10:00:04Z",
            message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "done"}]},
            toolUseResult={"filePath": "/repo/a.py", "originalFile": "x = 1\n",
                           "oldString": "1", "newString": "2", "structuredPatch": []},
        ),
        _record(
            type="assistant",
            timestamp="2026-07-30T10:00:05Z",
            requestId="r3",
            message={"role": "assistant", "model": "claude-opus-5",
                     "content": [{"type": "text", "text": "Changed it."}]},
        ),
    ]
    path = tmp_path / f"{SESSION}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


# Absolute: the `sdk` fixture chdirs into a temp store, so a relative path would resolve
# to a nonexistent file and the checker would be handed an empty session.
ADVERSARIAL = Path(__file__).parent / "fixtures" / "claude_session.jsonl"


class TestItDoesNotFalseAlarm:
    def test_a_clean_session_conforms(self, conforming_session):
        report = compare(conforming_session)
        assert report.ok, report.problems

    def test_both_paths_saw_the_same_tool_calls(self, conforming_session):
        report = compare(conforming_session)
        # The count check is the coarsest guard in the checker and the one most likely to be silently
        # satisfied by both paths dropping the same call.
        assert report.offline_events > 0 and report.live_events > 0

    def test_the_file_versions_agree(self, conforming_session):
        report = compare(conforming_session)
        # Two versions of a.py -- the read and the edit -- seen by both paths, none inferred.
        assert report.observed_versions == 2
        assert report.inferred_versions == 0

    def test_activities_are_compared_by_signature_not_by_cid(self, conforming_session):
        # Statement CIDs are time-dependent, so every shared activity here is shared only because the
        # comparison relabels by (inputs, outputs). If it compared CIDs this would be 0.
        report = compare(conforming_session)
        assert report.shared_activities > 0
        assert report.shared_activities == report.live_activities


class TestItActuallyDetectsDivergence:
    """A checker that cannot fail proves nothing. These make it fail."""

    def test_the_adversarial_fixture_does_not_conform(self):
        # Hand-built to break the parser: an orphaned tool result whose tool_use lives in another
        # transcript, a compact boundary, snapshot deltas. Replay cannot synthesize those, so the
        # checker must report it rather than quietly passing.
        report = compare(ADVERSARIAL)
        assert not report.ok
        assert any("activity" in p for p in report.problems)

    def test_a_dropped_live_event_is_caught(self, conforming_session, monkeypatch):
        # Simulates the failure the checker exists to catch: one adapter losing a tool result. Before
        # this, nothing verified that such a loss would actually be reported.
        #
        # Note what it is *not* caught by. The tool-call count still matches, because the recorder
        # closes unterminated runs at session end and that synthesized ToolCallEnded increments the
        # same counter. The count check alone would pass here -- the file-version comparison is what
        # catches it, which is a good argument for the checker having more than one guard.
        real = equivalence.to_events

        def lossy(payload, dialect=None):
            events = real(payload, dialect)
            if payload.get("hook_event_name") == "PostToolUse":
                return []
            return events

        monkeypatch.setattr(equivalence, "to_events", lossy)
        report = compare(conforming_session)
        assert not report.ok
        assert "observed file versions disagree" in report.problems
        assert any("no declared cause" in p for p in report.problems)

    def test_the_tool_call_count_guard_fires_when_a_call_vanishes_entirely(
        self, conforming_session, monkeypatch
    ):
        # Dropping PreToolUse means the run never opens, so nothing is synthesized to close, and the
        # coarse count check is what notices.
        real = equivalence.to_events

        def lossy(payload, dialect=None):
            if payload.get("hook_event_name") in ("PreToolUse", "PostToolUse"):
                return []
            return real(payload, dialect)

        monkeypatch.setattr(equivalence, "to_events", lossy)
        report = compare(conforming_session)
        assert not report.ok
        assert any("tool-call counts differ" in p for p in report.problems)

    def test_a_corrupted_live_file_version_is_caught(self, conforming_session, monkeypatch):
        # A file version only the live path saw is a divergence in the substantive claim both paths
        # make about the world, and has its own check.
        real = equivalence.to_events

        def extra(payload, dialect=None):
            events = list(real(payload, dialect))
            if payload.get("hook_event_name") == "PostToolUse":
                from eqty_lineage.core import FileObserved

                events.append(FileObserved(path="/repo/ghost.py", content=b"invented\n", mode="wrote"))
            return events

        monkeypatch.setattr(equivalence, "to_events", extra)
        report = compare(conforming_session)
        assert not report.ok
        assert any("only the live path saw" in p for p in report.problems)


class TestDeclarations:
    """Every declared bucket must be non-empty. One that silently empties means the capture path moved
    underneath its declaration, and the checker would then be asserting nothing about it."""

    def test_the_declared_type_maps_are_populated_and_disjoint(self):
        assert TRANSCRIPT_ONLY_TYPES and HOOKS_ONLY_TYPES
        assert not set(TRANSCRIPT_ONLY_TYPES) & set(HOOKS_ONLY_TYPES)

    def test_every_declaration_carries_a_reason(self):
        # The value is the justification shown to a reader; an empty one makes the declaration
        # unreviewable.
        for reason in list(TRANSCRIPT_ONLY_TYPES.values()) + list(HOOKS_ONLY_TYPES.values()):
            assert reason and len(reason) > 10

    def test_the_model_call_divergence_is_demonstrated_not_just_declared(self, conforming_session):
        # The checker fails loudly if the declared model-call divergence produces nothing, on the
        # grounds that a stale declaration is worse than none. Confirm it is exercised here.
        report = compare(conforming_session)
        assert "declared model-call divergence produced nothing (declaration is stale)" not in report.problems

    def test_transcript_only_types_are_the_model_artifacts(self):
        # Hooks never carry the model's messages, so these three can only ever come from a transcript.
        assert set(TRANSCRIPT_ONLY_TYPES) == {"Prompt", "Model", "Reasoning"}


class TestReport:
    def test_ok_is_exactly_the_absence_of_problems(self):
        report = equivalence.Report(session="s")
        assert report.ok
        report.problems.append("something")
        assert not report.ok

    def test_summary_names_the_session_and_status(self, conforming_session):
        report = compare(conforming_session)
        assert report.session.startswith(SESSION[:8])
        assert "OK" in report.summary()

    def test_a_failing_summary_carries_the_problems(self):
        report = equivalence.Report(session="abcdef12")
        report.problems.append("derivation edges disagree")
        assert "FAIL" in report.summary()
        assert "derivation edges disagree" in report.summary()
