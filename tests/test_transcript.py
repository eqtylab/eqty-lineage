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
        bash_end = next(e for e in events if isinstance(e, ToolCallEnded) and e.tool_use_id == "toolu_bash")
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
                end = next(x for x in events if isinstance(x, ToolCallEnded) and x.tool_use_id == e.tool_use_id)
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
            e for e in events if isinstance(e, FileObserved) and e.path == "/repo/util.py" and e.observed is False
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


class TestSubagentTranscripts:
    """A subagent's work lives in its own file, and following the link is what makes it visible.

    Claude Code writes each subagent's transcript to `<session>/subagents/agent-<agentId>.jsonl`,
    with workflow agents nested one level deeper, and the `agentId` on the Agent tool result names
    the file exactly. `find_sessions` globs a single level, so these were never discovered -- 990 of
    2,236 transcripts on the machine measured, carrying 26% of all file operations -- and every
    subagent was recorded `opaque=True` on the grounds that its internals were not visible.
    """

    def _session(self, tmp_path, agent_id="a1", with_transcript=True, nested=False):
        import json

        project = tmp_path / "project"
        project.mkdir(parents=True)
        session = project / "s1.jsonl"
        records = [
            {
                "type": "user",
                "sessionId": "s1",
                "cwd": "/repo",
                "version": "2.1.220",
                "timestamp": "2026-07-30T10:00:00Z",
                "message": {"role": "user", "content": "delegate"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-07-30T10:00:01Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-opus-5",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Agent",
                            "input": {"description": "do work", "subagent_type": "general-purpose"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-07-30T10:00:09Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "done"}],
                },
                "toolUseResult": {"agentId": agent_id, "status": "completed", "content": "done"},
            },
        ]
        session.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        if with_transcript:
            base = project / "s1" / "subagents"
            if nested:
                base = base / "workflows" / "wf_1"
            base.mkdir(parents=True)
            inner = [
                {
                    "type": "assistant",
                    "isSidechain": True,
                    "agentId": agent_id,
                    "timestamp": "2026-07-30T10:00:03Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-5",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "s-t1",
                                "name": "Write",
                                "input": {"file_path": "/repo/made_by_subagent.py"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "isSidechain": True,
                    "timestamp": "2026-07-30T10:00:04Z",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "s-t1", "content": "ok"}],
                    },
                    "toolUseResult": {"filePath": "/repo/made_by_subagent.py", "content": "x = 1\n"},
                },
            ]
            (base / f"agent-{agent_id}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in inner) + "\n", encoding="utf-8"
            )
        return session

    def test_the_transcript_is_found_by_agent_id(self, tmp_path):
        from eqty_lineage.transcript.claude_code import find_subagent_transcripts

        session = self._session(tmp_path)
        found = find_subagent_transcripts(session)
        assert set(found) == {"a1"}

    def test_workflow_agents_nested_a_level_deeper_are_found_too(self, tmp_path):
        from eqty_lineage.transcript.claude_code import find_subagent_transcripts

        session = self._session(tmp_path, nested=True)
        assert set(find_subagent_transcripts(session)) == {"a1"}

    def test_a_session_with_no_subagents_finds_none(self, tmp_path):
        from eqty_lineage.transcript.claude_code import find_subagent_transcripts

        session = self._session(tmp_path, with_transcript=False)
        assert find_subagent_transcripts(session) == {}

    def test_the_subagents_file_lineage_reaches_the_event_stream(self, tmp_path):
        session = self._session(tmp_path)
        events = list(ClaudeCodeTranscript(session).events())
        paths = {e.path for e in events if isinstance(e, FileObserved)}
        assert "/repo/made_by_subagent.py" in paths

    def test_it_is_absent_when_the_link_is_not_followed(self, tmp_path):
        session = self._session(tmp_path)
        events = list(ClaudeCodeTranscript(session, include_subagents=False).events())
        paths = {e.path for e in events if isinstance(e, FileObserved)}
        assert "/repo/made_by_subagent.py" not in paths

    def test_opaque_is_false_only_when_the_transcript_was_actually_read(self, tmp_path):
        found = list(ClaudeCodeTranscript(self._session(tmp_path)).events())
        other = tmp_path / "b"
        other.mkdir()
        missing = list(ClaudeCodeTranscript(self._session(other, with_transcript=False)).events())

        def opacity(events):
            return [e.opaque for e in events if isinstance(e, SubagentEnded)]

        assert opacity(found) == [False], "the internals were visible"
        assert opacity(missing) == [True], "they genuinely were not"

    def test_the_subagents_own_session_brackets_are_not_emitted(self, tmp_path):
        # A subagent has its own SessionStarted/SessionEnded. Yielding those mid-session would reset
        # the recorder's agent asset, context anchor and permission mode to the child's.
        events = list(ClaudeCodeTranscript(self._session(tmp_path)).events())
        assert sum(1 for e in events if isinstance(e, SessionStarted)) == 1
        assert sum(1 for e in events if isinstance(e, SessionEnded)) == 1

    def test_the_subagent_is_identified_by_its_agent_id(self, tmp_path):
        events = list(ClaudeCodeTranscript(self._session(tmp_path)).events())
        [ended] = [e for e in events if isinstance(e, SubagentEnded)]
        assert ended.agent_id == "a1"

    def test_recovery_is_reported_for_the_caller(self, tmp_path):
        transcript = ClaudeCodeTranscript(self._session(tmp_path))
        list(transcript.events())
        assert transcript.subagents_recovered == ["a1"]


class TestFileHistoryBackups:
    """The pre-change bytes are on disk, and the adapter had never read them.

    A snapshot delta names its backup exactly -- `~/.claude/file-history/<session>/<backupFileName>`
    -- and 131,751 of 131,989 references still resolved on the machine measured. The change itself
    stays identity-only, because the store holds the state *before* it. What the store does give is
    the pre-image, which is real content and which also seeds the recorder's chaining so a later
    edit carrying only oldString/newString can be replayed.
    """

    def _session(self, tmp_path, backup="was = 1\n", write_backup=True):
        import json

        project = tmp_path / "projects"
        project.mkdir(parents=True)
        session = project / "s1.jsonl"
        entry = {"backupFileName": "abc123@v2", "version": 2}
        records = [
            {
                "type": "user",
                "sessionId": "s1",
                "cwd": "/repo",
                "version": "2.1.220",
                "timestamp": "2026-07-30T10:00:00Z",
                "message": {"role": "user", "content": "go"},
            },
            {
                "type": "file-history-snapshot",
                "timestamp": "2026-07-30T10:00:01Z",
                "snapshot": {"trackedFileBackups": {"/repo/a.py": {**entry, "version": 1}}},
            },
            {
                "type": "file-history-snapshot",
                "timestamp": "2026-07-30T10:00:02Z",
                "snapshot": {"trackedFileBackups": {"/repo/a.py": entry}},
            },
        ]
        session.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        history = tmp_path / "file-history"
        if write_backup:
            (history / "s1").mkdir(parents=True)
            (history / "s1" / "abc123@v2").write_text(backup, encoding="utf-8")
        else:
            history.mkdir()
        return session, history

    def _observed(self, session, history, **kw):
        return [
            e
            for e in ClaudeCodeTranscript(session, file_history_root=history, **kw).events()
            if isinstance(e, FileObserved)
        ]

    def test_the_pre_change_bytes_are_recovered(self, tmp_path):
        session, history = self._session(tmp_path)
        with_content = [e for e in self._observed(session, history) if e.content is not None]
        assert [e.content for e in with_content] == [b"was = 1\n"]

    def test_they_are_marked_as_coming_from_the_backup_store(self, tmp_path):
        # Real content, but evidence about this machine's disk rather than about the session.
        session, history = self._session(tmp_path)
        [recovered] = [e for e in self._observed(session, history) if e.content is not None]
        assert recovered.content_source == "backup-store"

    def test_the_change_itself_stays_identity_only(self, tmp_path):
        # The store holds the state *before* the change; claiming a post-state would be a guess.
        session, history = self._session(tmp_path)
        changed = [e for e in self._observed(session, history) if e.mode == "changed"]
        assert changed and all(e.content is None and not e.observed for e in changed)

    def test_a_missing_backup_is_not_fatal(self, tmp_path):
        # The store is pruned; an older session's backups are routinely gone.
        session, history = self._session(tmp_path, write_backup=False)
        events = self._observed(session, history)
        assert events and all(e.content is None for e in events)

    def test_reading_the_store_can_be_switched_off(self, tmp_path):
        session, history = self._session(tmp_path)
        assert not [e for e in self._observed(session, history, include_file_history=False) if e.content is not None]

    def test_the_recovered_bytes_seed_the_recorders_chain(self, tmp_path):
        # The point of emitting it: a later edit with no pre-image of its own can replay against it.
        session, history = self._session(tmp_path, backup="x = 1\n")
        [recovered] = [e for e in self._observed(session, history) if e.content is not None]
        assert recovered.mode == "read", "seeding requires it to be an input, not an output"
        assert recovered.content == b"x = 1\n"
