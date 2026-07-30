"""Claude Code hook payload shapes, pinned to the schema the CLI actually sends.

Every case here was a silent failure: the adapter read a key the payload does not carry, so the event
was recorded with its substance missing rather than dropped. Nothing raised, nothing was logged, and
the graph looked complete.

The field names are taken from the hook schema embedded in the Claude Code 2.1.220 binary, not from the
documentation -- which describes ``PostToolBatch`` as carrying a ``batch`` array. It carries
``tool_calls``.
"""

from eqty_lineage.agent_hooks.dialects import to_events
from eqty_lineage.core import FileObserved, ToolCallEnded


def events_of(payload):
    return to_events(payload, "claude-code")


def hook(event, **fields):
    return {"hook_event_name": event, "session_id": "s1", "prompt_id": "p1", **fields}


class TestFailures:
    """``PostToolUseFailure`` carries ``error``; only success carries ``tool_response``."""

    def test_the_error_text_reaches_the_event(self):
        [ended] = events_of(hook("PostToolUseFailure", tool_name="Bash", tool_use_id="t1",
                                 error="ENOENT: no such file"))
        assert isinstance(ended, ToolCallEnded)
        assert ended.result == "ENOENT: no such file"
        assert ended.is_error

    def test_a_failure_is_an_error_even_when_the_payload_says_nothing_else(self):
        [ended] = events_of(hook("PostToolUseFailure", tool_name="Bash", tool_use_id="t1"))
        assert ended.is_error

    def test_an_interrupt_is_distinguishable_from_a_tool_failure(self):
        # Both end the call. Only this flag separates "the tool broke" from "the user stopped it".
        [ended] = events_of(hook("PostToolUseFailure", tool_name="Bash", tool_use_id="t1",
                                 error="stopped", is_interrupt=True))
        assert ended.result == {"error": "stopped", "interrupted": True}

    def test_success_still_reads_tool_response(self):
        [ended] = events_of(hook("PostToolUse", tool_name="Bash", tool_use_id="t1",
                                 tool_response="ok"))
        assert ended.result == "ok"
        assert not ended.is_error


class TestBatch:
    """``PostToolBatch`` closes several calls at once; entries use the PostToolUse result key."""

    BATCH = hook("PostToolBatch", tool_calls=[
        {"tool_name": "Read", "tool_use_id": "t1", "tool_input": {"file_path": "/repo/a.py"},
         "tool_response": {"file": {"filePath": "/repo/a.py", "content": "a = 1\n"}}},
        {"tool_name": "Bash", "tool_use_id": "t2", "tool_input": {"command": "false"},
         "tool_response": "Exit code: 1\nOutput:\nboom\n"},
    ])

    def test_every_call_in_the_batch_is_closed(self):
        ended = [e for e in events_of(self.BATCH) if isinstance(e, ToolCallEnded)]
        assert [e.tool_use_id for e in ended] == ["t1", "t2"]

    def test_the_file_lineage_a_batch_carries_is_not_lost(self):
        # Reading `result` instead of `tool_response` produced no file events at all here.
        observed = [e for e in events_of(self.BATCH) if isinstance(e, FileObserved)]
        assert [e.path for e in observed] == ["/repo/a.py"]

    def test_results_survive_the_batch(self):
        ended = [e for e in events_of(self.BATCH) if isinstance(e, ToolCallEnded)]
        assert ended[0].result == {"file": {"filePath": "/repo/a.py", "content": "a = 1\n"}}

    def test_an_entry_that_is_not_a_mapping_is_skipped_rather_than_fatal(self):
        payload = hook("PostToolBatch", tool_calls=["nonsense", {"tool_use_id": "t1"}])
        assert [e.tool_use_id for e in events_of(payload) if isinstance(e, ToolCallEnded)] == ["t1"]


class TestWatcherRemovals:
    """``change_type`` is ``change`` or ``unlink``."""

    def test_a_removal_is_recorded_as_a_deletion(self, tmp_path):
        gone = tmp_path / "gone.py"
        [event] = events_of(hook("FileChanged", file_path=str(gone), change_type="unlink"))
        assert event.mode == "deleted"
        assert event.content is None

    def test_a_modification_still_reads_content_from_disk(self, tmp_path):
        live = tmp_path / "live.py"
        live.write_text("x = 1\n")
        [event] = events_of(hook("FileChanged", file_path=str(live), change_type="change"))
        assert event.mode == "changed"
        assert event.content == b"x = 1\n"

    def test_a_removal_does_not_borrow_content_from_a_path_recreated_since(self, tmp_path):
        # The read happens at hook time, not at change time. For an unlink the file may well exist
        # again by now, and reading it would attribute the new bytes to the deletion.
        path = tmp_path / "churn.py"
        path.write_text("recreated\n")
        [event] = events_of(hook("FileChanged", file_path=str(path), change_type="unlink"))
        assert event.content is None
