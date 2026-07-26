"""Offline transcript parsing, against a hand-built session covering the edge cases that bit us."""

from eqty_lineage.core import (
    Compacted,
    FileObserved,
    ModelCall,
    PromptSubmitted,
    SessionEnded,
    SessionStarted,
    SubagentEnded,
    SubagentStarted,
    ToolCallEnded,
    ToolCallStarted,
)
from eqty_lineage.transcript.claude_code import ClaudeCodeTranscript


def parse(path):
    t = ClaudeCodeTranscript(path, file_history_root=None)
    return list(t.events()), t


class TestSessionHeader:
    def test_header_is_assembled_from_fields_scattered_across_record_types(self, claude_transcript):
        events, _ = parse(claude_transcript)
        header = events[0]
        assert isinstance(header, SessionStarted)
        assert header.session_id == "11111111-2222-3333-4444-555555555555"
        assert header.agent == "claude-code"
        assert header.cwd == "/repo"
        assert header.git_branch == "main"
        # model comes from the first assistant record, permission_mode from a permission-mode record
        assert header.model == "claude-opus-5"
        assert header.permission_mode == "acceptEdits"

    def test_the_stream_is_bracketed_by_start_and_end(self, claude_transcript):
        events, _ = parse(claude_transcript)
        assert isinstance(events[0], SessionStarted)
        assert isinstance(events[-1], SessionEnded)
        assert events[-1].session_id == events[0].session_id


class TestToolCalls:
    def test_every_started_call_is_ended(self, claude_transcript):
        events, _ = parse(claude_transcript)
        started = {e.tool_use_id for e in events if isinstance(e, ToolCallStarted)}
        ended = {e.tool_use_id for e in events if isinstance(e, ToolCallEnded)}
        # Ends must never outnumber starts: an unmatched end is a dropped file version. This is the
        # invariant that caught orphaned results in resumed sessions (19,754 ends vs 19,659 starts).
        assert ended <= started
        assert started == ended

    def test_bash_command_becomes_the_tool_source(self, claude_transcript):
        events, _ = parse(claude_transcript)
        bash = next(e for e in events if isinstance(e, ToolCallStarted) and e.tool_name == "Bash")
        assert bash.tool_source == "pytest -q"

    def test_a_failed_call_survives_in_the_stream(self, claude_transcript):
        events, _ = parse(claude_transcript)
        bash_end = next(
            e for e in events if isinstance(e, ToolCallEnded) and e.tool_use_id == "toolu_bash"
        )
        # A failed call is often the one an audit cares about.
        assert bash_end.is_error is True

    def test_orphan_result_is_kept_as_an_unknown_call_rather_than_dropped(self, claude_transcript):
        events, transcript = parse(claude_transcript)
        orphan = next(e for e in events if isinstance(e, ToolCallStarted) and e.tool_use_id == "toolu_orphan")
        # Named "unknown" so nothing is claimed that is not known -- but kept, because the result is a
        # file read whose lineage would otherwise vanish.
        assert orphan.tool_name == "unknown"
        assert orphan.tool_input is None
        assert any("no tool_use in this transcript" in w for w in transcript.warnings)

        observed = [e for e in events if isinstance(e, FileObserved) and e.path == "/repo/README.md"]
        assert [e.mode for e in observed] == ["read"]


class TestFileObservations:
    def test_file_events_precede_the_call_they_belong_to(self, claude_transcript):
        events, _ = parse(claude_transcript)
        index = {id(e): i for i, e in enumerate(events)}
        for e in events:
            if isinstance(e, FileObserved) and e.tool_use_id:
                end = next(
                    x for x in events
                    if isinstance(x, ToolCallEnded) and x.tool_use_id == e.tool_use_id
                )
                # The recorder attaches observations to the open run, so they must arrive first.
                assert index[id(e)] < index[id(end)]

    def test_an_edit_yields_the_pre_and_post_states(self, claude_transcript):
        events, _ = parse(claude_transcript)
        edit = [e for e in events if isinstance(e, FileObserved) and e.tool_use_id == "toolu_edit"]
        assert [(e.mode, e.content) for e in edit] == [
            ("read", b"def helper():\n    return 1\n"),
            ("wrote", b"def helper():\n    return 2\n"),
        ]

    def test_a_snapshot_delta_already_explained_by_an_edit_is_not_double_reported(self, claude_transcript):
        events, _ = parse(claude_transcript)
        # util.py's backup version goes 1 -> 2 across the Edit. The Edit described that transition in
        # full, so the snapshot must not add a second, weaker "changed" observation.
        inferred = [
            e for e in events
            if isinstance(e, FileObserved) and e.path == "/repo/util.py" and e.observed is False
        ]
        assert inferred == []

    def test_inferred_observations_are_flagged_and_carry_no_content(self, claude_transcript, tmp_path):
        # A snapshot delta with no Edit to explain it: the backup store holds *pre*-change content, so
        # the delta establishes that a path changed, never what it became.
        session = tmp_path / "s.jsonl"
        session.write_text(
            '{"type":"user","uuid":"u1","timestamp":"2026-01-01T00:00:00Z","sessionId":"s",'
            '"message":{"role":"user","content":[{"type":"text","text":"hi"}]}}\n'
            '{"type":"file-history-snapshot","timestamp":"2026-01-01T00:00:01Z",'
            '"snapshot":{"trackedFileBackups":{"/repo/x.py":{"version":1}}}}\n'
            '{"type":"file-history-snapshot","timestamp":"2026-01-01T00:00:02Z",'
            '"snapshot":{"trackedFileBackups":{"/repo/x.py":{"version":2}}}}\n'
        )
        events, _ = parse(session)
        inferred = [e for e in events if isinstance(e, FileObserved)]
        assert len(inferred) == 1
        assert inferred[0].observed is False
        assert inferred[0].content is None
        assert inferred[0].mode == "changed"


class TestSubagentsAndCompaction:
    def test_a_task_call_brackets_a_subagent(self, claude_transcript):
        events, _ = parse(claude_transcript)
        start = next(e for e in events if isinstance(e, SubagentStarted))
        end = next(e for e in events if isinstance(e, SubagentEnded))
        assert start.agent_id == end.agent_id == "toolu_task"
        assert start.agent_type == "general-purpose"
        # Offline, a subagent is opaque: the transcript sees aggregate stats, not the child's calls.
        assert end.opaque is True
        assert end.stats == {"Read": 2, "Grep": 1}

    def test_compaction_records_the_logical_parent(self, claude_transcript):
        events, _ = parse(claude_transcript)
        compact = next(e for e in events if isinstance(e, Compacted))
        assert (compact.pre_tokens, compact.post_tokens) == (50000, 8000)
        assert compact.dropped_tokens == 42000
        assert compact.trigger == "auto"
        # parentUuid is null across a compact boundary; the real predecessor is logicalParentUuid, and
        # anything walking parentUuid naively splits the session in two.
        assert compact.logical_parent == "u8"


class TestModelCalls:
    def test_usage_is_captured_and_no_prompt_is_invented(self, claude_transcript):
        events, _ = parse(claude_transcript)
        calls = [e for e in events if isinstance(e, ModelCall)]
        assert len(calls) == 5
        assert calls[0].usage == {"input_tokens": 100, "output_tokens": 20}
        # The transcript stores the conversation, not the request payload, so the input is not
        # recoverable. Recording None is honest; synthesising a prompt would not be.
        assert all(c.messages_in is None for c in calls)

    def test_the_user_prompt_is_captured(self, claude_transcript):
        events, _ = parse(claude_transcript)
        prompts = [e for e in events if isinstance(e, PromptSubmitted)]
        assert [p.text for p in prompts] == ["Add a slugify helper to util.py"]
        assert prompts[0].prompt_id == "p1"


class TestRobustness:
    def test_a_truncated_tail_is_skipped_not_raised(self, tmp_path):
        # Normal for a session that is still being written.
        session = tmp_path / "s.jsonl"
        session.write_text(
            '{"type":"user","uuid":"u1","timestamp":"2026-01-01T00:00:00Z","sessionId":"s",'
            '"message":{"role":"user","content":[{"type":"text","text":"hi"}]}}\n'
            '{"type":"assistant","uuid":"a1","messa\n'
        )
        events, transcript = parse(session)
        assert any(isinstance(e, PromptSubmitted) for e in events)
        assert any("malformed JSON" in w for w in transcript.warnings)

    def test_an_empty_transcript_yields_nothing(self, tmp_path):
        session = tmp_path / "empty.jsonl"
        session.write_text("")
        assert parse(session)[0] == []

    def test_a_call_with_no_result_is_warned_about(self, tmp_path):
        session = tmp_path / "s.jsonl"
        session.write_text(
            '{"type":"assistant","uuid":"a1","timestamp":"2026-01-01T00:00:00Z","sessionId":"s",'
            '"message":{"role":"assistant","model":"m","content":[{"type":"tool_use","id":"t1",'
            '"name":"Read","input":{}}]}}\n'
        )
        _, transcript = parse(session)
        assert any("has no result in transcript" in w for w in transcript.warnings)
