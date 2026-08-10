"""The collector and the graph builder, tested as one pipeline.

Each half is easy to pass alone: a collector that writes a file, a builder that reads a list. What the
PR claims is that the manifest is derived from what Codex emitted, and only running the real collector
over real captured payloads and replaying the file it produces tests that claim.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
COLLECTOR = ROOT / "plugins" / "eqty-lineage" / "scripts" / "capture_hook.py"
FIXTURE = ROOT / "tests" / "fixtures" / "codex_hooks.json"

# The Bash call in the fixture, used as the command the collector is configured to refuse.
DENIED_COMMAND = "sed -n '1,80p' slug.py"


def _payloads():
    return json.loads(FIXTURE.read_text())


def _collect(tmp_path, payloads, deny=None):
    """Run the real collector once per payload, as Codex would, and return the capture file."""
    capture = tmp_path / "codex-hooks.jsonl"
    env = os.environ.copy()
    env["EQTY_LINEAGE_CAPTURE"] = str(capture)
    if deny is not None:
        env["EQTY_LINEAGE_DENY_COMMAND"] = deny
    else:
        env.pop("EQTY_LINEAGE_DENY_COMMAND", None)

    denied_ids = set()
    for payload in payloads:
        # A denied call never reaches the tool, so it never emits PostToolUse. Feeding one anyway would
        # test a session that cannot happen.
        if payload.get("hook_event_name") == "PostToolUse" and payload.get("tool_use_id") in denied_ids:
            continue
        result = subprocess.run(
            [sys.executable, str(COLLECTOR)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
        if result.stdout.strip():
            decision = json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"]
            assert decision == "deny"
            denied_ids.add(payload.get("tool_use_id"))
    return capture, denied_ids


def test_the_fixture_is_a_real_session_with_two_completed_tool_calls():
    """Non-vacuity: every assertion below is worthless if the fixture is empty or already denied."""
    payloads = _payloads()
    events = [payload.get("hook_event_name") for payload in payloads]
    assert events.count("PreToolUse") == 2
    assert events.count("PostToolUse") == 2
    commands = [(payload.get("tool_input") or {}).get("command") for payload in payloads]
    assert DENIED_COMMAND in commands


def test_collector_stores_the_payload_verbatim_and_its_own_decision_separately(tmp_path):
    payloads = _payloads()
    capture, denied_ids = _collect(tmp_path, payloads, deny=DENIED_COMMAND)

    records = [json.loads(line) for line in capture.read_text().splitlines() if line.strip()]
    assert len(denied_ids) == 1
    assert len(records) == len(payloads) - 1  # the denied call's PostToolUse never happens

    written = [record["payload"] for record in records]
    assert written == [
        p for p in payloads if p.get("tool_use_id") not in denied_ids or p.get("hook_event_name") != "PostToolUse"
    ]
    assert all(record["collector"]["schema"] == "eqty.codex-hook-capture.v1" for record in records)
    assert all(isinstance(record["collector"]["received_unix_ns"], int) for record in records)

    decisions = [record["collector"].get("decision") for record in records]
    assert decisions.count("deny") == 1


def test_an_unconfigured_collector_denies_nothing(tmp_path):
    """Regression: comparing an unset variable to a missing command made both ``None`` and matched."""
    capture = tmp_path / "capture.jsonl"
    env = os.environ.copy()
    env["EQTY_LINEAGE_CAPTURE"] = str(capture)
    env.pop("EQTY_LINEAGE_DENY_COMMAND", None)

    result = subprocess.run(
        [sys.executable, str(COLLECTOR)],
        input=json.dumps({"hook_event_name": "PreToolUse", "tool_name": "apply_patch", "tool_input": {}}),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    assert result.stdout.strip() == ""
    assert "decision" not in json.loads(capture.read_text())["collector"]


def test_the_deny_output_matches_the_schema_codex_actually_enforces(tmp_path):
    """codex-cli drops a decision whose shape it does not recognize, and runs the command anyway.

    ``PreToolUseHookSpecificOutputWire`` requires ``hookEventName`` and sets
    ``additionalProperties: false``. Nothing raises when it is missing -- against codex-cli 0.147.0 a
    denied ``touch forbidden.txt`` silently created its file. Only the shape is asserted here; that it
    is the *enforced* shape was confirmed by a live session.
    """
    capture = tmp_path / "capture.jsonl"
    env = os.environ.copy()
    env["EQTY_LINEAGE_CAPTURE"] = str(capture)
    env["EQTY_LINEAGE_DENY_COMMAND"] = "touch forbidden.txt"

    result = subprocess.run(
        [sys.executable, str(COLLECTOR)],
        input=json.dumps(
            {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "touch forbidden.txt"}}
        ),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    output = json.loads(result.stdout)["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert output["permissionDecisionReason"]
    assert set(output) <= {
        "hookEventName",
        "permissionDecision",
        "permissionDecisionReason",
        "additionalContext",
        "updatedInput",
    }


def test_a_null_tool_input_does_not_crash_the_collector(tmp_path):
    env = os.environ.copy()
    env["EQTY_LINEAGE_CAPTURE"] = str(tmp_path / "capture.jsonl")
    env["EQTY_LINEAGE_DENY_COMMAND"] = "rm -rf /"

    subprocess.run(
        [sys.executable, str(COLLECTOR)],
        input=json.dumps({"hook_event_name": "PreToolUse", "tool_input": None}),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )


# --- normalization -------------------------------------------------------------------------------


def test_normalize_reads_a_real_capture(tmp_path):
    from eqty_lineage.codex.capture import ALLOW, DENY, load_session

    capture, _ = _collect(tmp_path, _payloads(), deny=DENIED_COMMAND)
    session = load_session(capture)

    assert session.session_id == "019f9ff2-2dd9-77f0-b687-22723134122c"
    assert session.model == "gpt-5.4-mini"
    assert len(session.prompts) == 1
    assert len(session.attempts) == 2

    by_decision = {attempt.decision: attempt for attempt in session.attempts}
    assert set(by_decision) == {ALLOW, DENY}
    assert by_decision[ALLOW].executed is True
    assert by_decision[ALLOW].tool_name == "apply_patch"
    assert by_decision[DENY].executed is False
    assert by_decision[DENY].result is None


def test_an_unfinished_call_is_unknown_rather_than_denied():
    """A capture cut mid-call must not become a signed denial."""
    from eqty_lineage.codex.capture import UNKNOWN, CaptureRecord, normalize

    session = normalize(
        [
            CaptureRecord({"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_use_id": "x", "tool_input": {}}),
        ]
    )

    assert [attempt.decision for attempt in session.attempts] == [UNKNOWN]
    assert session.attempts[0].executed is False


def test_a_denial_contradicted_by_execution_keeps_both_facts():
    """If a call ran despite a deny, the graph should show that, not smooth it over."""
    from eqty_lineage.codex.capture import DENY, CaptureRecord, normalize

    session = normalize(
        [
            CaptureRecord(
                {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_use_id": "x", "tool_input": {}},
                {"decision": "deny"},
            ),
            CaptureRecord(
                {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_use_id": "x", "tool_response": "ran"}
            ),
        ]
    )

    assert session.attempts[0].decision == DENY
    assert session.attempts[0].executed is True


@pytest.mark.parametrize(
    "record",
    [
        {"collector": {"schema": "eqty.codex-hook-capture.v1"}, "payload": {"hook_event_name": "SessionStart"}},
        {"payload": {"hook_event_name": "SessionStart"}, "schema": "eqty.codex-hook-capture.v1"},
        {"hook_event_name": "SessionStart"},
    ],
    ids=["collector-envelope", "flat-envelope", "bare-payload"],
)
def test_load_capture_accepts_every_record_shape(tmp_path, record):
    from eqty_lineage.codex.capture import load_capture

    path = tmp_path / "capture.jsonl"
    path.write_text(json.dumps(record) + "\n")

    assert load_capture(path)[0].payload["hook_event_name"] == "SessionStart"


def test_load_capture_reads_a_fixture_stored_as_a_json_array():
    from eqty_lineage.codex.capture import load_capture

    records = load_capture(FIXTURE)

    assert len(records) == 8
    assert records[0].payload["hook_event_name"] == "SessionStart"
    assert records[0].collector == {}
