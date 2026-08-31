"""Deriving file versions from a tool result.

This module is shared by both capture paths, so a bug here is a bug in the offline graph *and* the live
one, and the conformance check cannot see it -- the classic correlated-fault blind spot in N-version
comparison. It gets tested directly for that reason.
"""

from eqty_lineage.recorder import apply_edit
from eqty_lineage.recorder.tool_results import file_events_from_result


class TestApplyEdit:
    def test_replays_the_replacement_exactly(self):
        assert apply_edit("a b a", "b", "c", replace_all=False) == "a c a"

    def test_replace_all_is_honoured(self):
        assert apply_edit("a b a", "a", "z", replace_all=True) == "z b z"
        assert apply_edit("a b a", "a", "z", replace_all=False) == "z b a"

    def test_refuses_when_the_payload_disagrees_with_itself(self):
        # The old string is not in the original: something is wrong with the payload. Guessing here
        # would content-address to a version the file never had.
        assert apply_edit("hello", "absent", "x", replace_all=False) is None

    def test_refuses_on_missing_inputs(self):
        assert apply_edit(None, "a", "b", replace_all=False) is None
        assert apply_edit("orig", None, "b", replace_all=False) is None
        assert apply_edit("orig", "a", None, replace_all=False) is None


class TestShapeDispatch:
    """Dispatch is on the result's shape, never the tool name.

    A resumed session carries results whose ``tool_use`` block lives in a different transcript, so the
    name is not recoverable -- and those results are often the reads and edits whose lineage matters
    most. Dispatching on shape is what keeps them.
    """

    def test_read_shape_yields_one_version_with_content(self):
        events, attributed = file_events_from_result(
            {"file": {"filePath": "/repo/a.py", "content": "x = 1\n", "numLines": 1, "totalLines": 1}},
            tool_use_id="t1",
        )
        assert [(e.path, e.mode) for e in events] == [("/repo/a.py", "read")]
        assert events[0].content == b"x = 1\n"
        # A read did not change the file, so there is no transition for a snapshot delta to duplicate.
        assert attributed is None

    def test_partial_read_records_identity_without_content(self):
        # A fragment's hash is not the file's hash.
        events, _ = file_events_from_result(
            {
                "file": {
                    "filePath": "/repo/a.py",
                    "content": "line 40\n",
                    "startLine": 40,
                    "numLines": 1,
                    "totalLines": 200,
                }
            },
            tool_use_id="t1",
        )
        assert len(events) == 1
        assert events[0].content is None
        assert events[0].mode == "read"

    def test_partial_read_can_be_excluded(self):
        events, _ = file_events_from_result(
            {"file": {"filePath": "/repo/a.py", "content": "x", "startLine": 40, "numLines": 1, "totalLines": 200}},
            tool_use_id="t1",
            include_partial_reads=False,
        )
        assert events == []

    def test_edit_shape_yields_pre_and_post_versions(self):
        events, attributed = file_events_from_result(
            {
                "filePath": "/repo/a.py",
                "originalFile": "x = 1\n",
                "oldString": "1",
                "newString": "2",
                "replaceAll": False,
                "structuredPatch": [],
            },
            tool_use_id="t1",
        )
        assert [(e.mode, e.content) for e in events] == [("read", b"x = 1\n"), ("wrote", b"x = 2\n")]
        # The result described the transition in full, so a snapshot delta for this path is the same
        # event seen from a worse angle and must not be re-reported.
        assert attributed == "/repo/a.py"

    def test_write_shape_takes_the_post_state_directly(self):
        events, _ = file_events_from_result({"filePath": "/repo/new.py", "content": "fresh\n"}, tool_use_id="t1")
        assert [(e.mode, e.content) for e in events] == [("wrote", b"fresh\n")]

    def test_unreconstructable_edit_records_the_transition_without_content(self):
        events, _ = file_events_from_result(
            {
                "filePath": "/repo/a.py",
                "originalFile": "x = 1\n",
                "oldString": "nope",
                "newString": "2",
                "structuredPatch": [],
            },
            tool_use_id="t1",
        )
        assert events[-1].mode == "wrote"
        assert events[-1].content is None

    def test_user_modified_is_carried_through(self):
        events, _ = file_events_from_result(
            {"filePath": "/repo/a.py", "content": "x\n", "userModified": True}, tool_use_id="t1"
        )
        assert events[-1].user_modified is True

    def test_bare_string_result_yields_nothing(self):
        # 626 of the Bash results in the surveyed corpus are bare strings.
        assert file_events_from_result("some stdout", tool_use_id="t1") == ([], None)

    def test_unrecognised_shape_yields_nothing(self):
        assert file_events_from_result({"stdout": "hi", "exitCode": 0}, tool_use_id="t1") == ([], None)

    def test_missing_path_yields_nothing(self):
        assert file_events_from_result({"file": {"content": "x"}}, tool_use_id="t1") == ([], None)
        assert file_events_from_result({"filePath": "", "content": "x"}, tool_use_id="t1") == ([], None)
