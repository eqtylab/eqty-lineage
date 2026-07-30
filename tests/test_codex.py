"""Codex support, against a real captured session.

Every finding pinned here came from reading the eight payloads in ``fixtures/codex_hooks.json``, not
from reading documentation. Both were silent failures: one produced a Codex session with no file
lineage whatsoever, the other attested a Codex session as Claude Code.
"""

from eqty_lineage.agent_hooks.dialects import (
    CLAUDE_CODE,
    CODEX,
    detect_dialect,
    parse_apply_patch,
    to_events,
)
from eqty_lineage.core import FileObserved, SessionStarted, ToolCallEnded, ToolCallStarted


class TestDialectDetection:
    def test_every_captured_payload_is_recognised_as_codex(self, codex_payloads):
        # This was 6/8 before the transcript_path fallback: SessionStart and SessionEnd carry no
        # turn_id, and SessionStart is what sets the agent name in the Agent asset -- so getting it
        # wrong does not merely mislabel an event, it attests the whole session as the wrong agent.
        assert [detect_dialect(p) for p in codex_payloads] == [CODEX] * len(codex_payloads)

    def test_lifecycle_payloads_are_the_ones_lacking_turn_id(self, codex_payloads):
        without = {p["hook_event_name"] for p in codex_payloads if "turn_id" not in p}
        assert without == {"SessionStart", "SessionEnd"}

    def test_turn_id_alone_identifies_codex(self):
        assert detect_dialect({"turn_id": "t1", "hook_event_name": "PreToolUse"}) == CODEX

    def test_prompt_id_alone_identifies_claude_code(self):
        assert detect_dialect({"prompt_id": "p1", "hook_event_name": "UserPromptSubmit"}) == CLAUDE_CODE

    def test_transcript_path_disambiguates_when_neither_id_is_present(self):
        assert detect_dialect({"transcript_path": "/home/u/.codex/sessions/x.jsonl"}) == CODEX
        assert detect_dialect({"transcript_path": "/home/u/.claude/projects/x.jsonl"}) == CLAUDE_CODE

    def test_an_unidentifiable_payload_falls_back_rather_than_raising(self):
        # An adapter that raised here would take down the session it is observing.
        assert detect_dialect({"hook_event_name": "SessionStart"}) == CLAUDE_CODE


class TestApplyPatch:
    """Codex writes with ``apply_patch``, whose input is a patch document.

    There is no ``filePath``, no ``content`` and no ``structuredPatch``, so the shape-dispatched parser
    in core sees nothing at all. Before this, a Codex session produced *no file lineage*.
    """

    ADD = (
        "*** Begin Patch\n"
        "*** Add File: slug.py\n"
        "+def slugify(s):\n"
        "+    return s.lower()\n"
        "*** End Patch\n"
    )
    UPDATE = (
        "*** Begin Patch\n"
        "*** Update File: util.py\n"
        "@@\n"
        "-    return 1\n"
        "+    return 2\n"
        "*** End Patch\n"
    )

    def test_an_added_file_is_fully_recoverable(self):
        # Every line of a new file is a `+` line, so the post-image is exact, not inferred.
        assert list(parse_apply_patch(self.ADD)) == [
            ("slug.py", "wrote", "def slugify(s):\n    return s.lower()\n")
        ]

    def test_an_updated_file_records_the_transition_without_inventing_content(self):
        # The document carries only hunks and the pre-image is not in the payload, so the post-image
        # cannot be reconstructed. Recording the path with content None is the honest outcome.
        assert list(parse_apply_patch(self.UPDATE)) == [("util.py", "wrote", None)]

    def test_relative_paths_are_resolved_against_cwd(self):
        assert list(parse_apply_patch(self.ADD, cwd="/repo"))[0][0] == "/repo/slug.py"

    def test_absolute_paths_are_left_alone(self):
        patch = self.ADD.replace("Add File: slug.py", "Add File: /abs/slug.py")
        assert list(parse_apply_patch(patch, cwd="/repo"))[0][0] == "/abs/slug.py"

    def test_a_multi_file_patch_yields_every_file(self):
        combined = (
            "*** Begin Patch\n"
            "*** Add File: a.py\n+x = 1\n"
            "*** Update File: b.py\n@@\n-y\n+z\n"
            "*** Delete File: c.py\n"
            "*** End Patch\n"
        )
        assert [p for p, _, _ in parse_apply_patch(combined)] == ["a.py", "b.py", "c.py"]

    def test_a_non_string_patch_yields_nothing(self):
        assert list(parse_apply_patch(None)) == []
        assert list(parse_apply_patch({"not": "a patch"})) == []

    def test_an_empty_patch_yields_nothing(self):
        assert list(parse_apply_patch("*** Begin Patch\n*** End Patch\n")) == []


class TestCapturedSessionReplay:
    """The eight captured payloads, run through the adapter end to end."""

    def events(self, payloads):
        out = []
        for p in payloads:
            out.extend(to_events(p))
        return out

    def test_the_session_is_bracketed_and_names_codex_as_the_agent(self, codex_payloads):
        events = self.events(codex_payloads)
        started = next(e for e in events if isinstance(e, SessionStarted))
        assert started.agent == "codex"

    def test_no_payload_raises(self, codex_payloads):
        # Both agents add hook events between releases; an unknown one must yield nothing, not raise.
        for p in codex_payloads:
            to_events(p)
        to_events({"hook_event_name": "SomeFutureEvent", "turn_id": "t1"})

    def test_tool_calls_are_paired(self, codex_payloads):
        events = self.events(codex_payloads)
        started = [e for e in events if isinstance(e, ToolCallStarted)]
        ended = [e for e in events if isinstance(e, ToolCallEnded)]
        assert len(started) == len(ended) == 2
        assert {e.tool_use_id for e in started} == {e.tool_use_id for e in ended}

    def test_the_captured_session_produces_file_lineage(self, codex_payloads):
        # The regression that matters: this list was empty before apply_patch was parsed.
        observed = [e for e in self.events(codex_payloads) if isinstance(e, FileObserved)]
        assert observed, "Codex session produced no file lineage"
        assert all(e.path.startswith("/") for e in observed)

    def test_the_captured_session_records_no_spurious_failures(self, codex_payloads):
        ended = [e for e in self.events(codex_payloads) if isinstance(e, ToolCallEnded)]
        assert not any(e.is_error for e in ended)


class TestCodexFailures:
    """Codex returns a plain string, so a dict-only error check calls every failure a success.

    The captured ``tool_response`` values begin ``Exit code: N`` -- the only failure signal a Codex
    payload carries. ``is_error`` drives the result asset's name and the ``is-error`` metadata on the
    activity, so getting it wrong means a graph in which nothing ever failed.
    """

    def ended(self, response):
        events = to_events({"hook_event_name": "PostToolUse", "turn_id": "t1", "session_id": "s1",
                            "tool_name": "Bash", "tool_use_id": "c1", "tool_response": response})
        return next(e for e in events if isinstance(e, ToolCallEnded))

    def test_a_nonzero_exit_is_an_error(self):
        assert self.ended("Exit code: 1\nWall time: 0.1 seconds\nOutput:\nboom\n").is_error

    def test_a_zero_exit_is_not(self):
        assert not self.ended("Exit code: 0\nWall time: 0.1 seconds\nOutput:\nSuccess.\n").is_error

    def test_a_negative_exit_is_an_error(self):
        # A signal death reports a negative code; treating it as success would hide a kill.
        assert self.ended("Exit code: -9\nOutput:\n").is_error

    def test_a_string_that_is_not_an_exit_report_is_not_guessed_at(self):
        # 626 of the Bash results in the surveyed corpus are bare strings. Only the ones that actually
        # report a code say anything about success.
        assert not self.ended("some tool output mentioning Exit code: 1 in passing").is_error

    def test_a_failed_shell_command_is_indistinguishable_from_a_successful_one(self):
        """A declared gap, captured from codex-cli 0.145.0 rather than assumed.

        These two payloads are what the CLI actually sent for `ls /nonexistent-path-xyz` and
        `echo hello-from-codex`. Bare output either way: no exit code, no flag, nothing separating
        them. The `Exit code:` prefix rides on the apply_patch shape, not on shell results.

        So a failing Codex shell command is recorded as a successful one, and no amount of care in
        this adapter changes that -- the signal is not in the payload. Pinned as a test so that if a
        later codex-cli starts reporting it, this fails and says so.
        """
        failed = self.ended("ls: /nonexistent-path-xyz: No such file or directory\n")
        succeeded = self.ended("hello-from-codex\n")
        assert failed.is_error == succeeded.is_error is False

    def test_the_apply_patch_shape_does_carry_a_code(self):
        # The other half of the same finding: where the payload states the outcome, it is read.
        assert not self.ended("Exit code: 0\nWall time: 0.1 seconds\nOutput:\nSuccess.\n").is_error
        assert self.ended("Exit code: 2\nWall time: 0.1 seconds\nOutput:\nfailed\n").is_error
