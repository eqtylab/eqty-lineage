import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from eqty_lineage.codex import build_demo, replay_capture

ROOT = Path(__file__).parents[1]
COLLECTOR = ROOT / "plugins" / "eqty-lineage" / "scripts" / "capture_hook.py"
FIXTURE = ROOT / "tests" / "fixtures" / "codex_hooks.json"
DENIED_COMMAND = "sed -n '1,80p' slug.py"


def _outputs(computation):
    output = computation.get("output")
    if output is None:
        return []
    return output if isinstance(output, list) else [output]


def _metadata(manifest):
    items = []
    for statement in manifest["statements"].values():
        if statement.get("@type") != "MetadataRegistration":
            continue
        blob = manifest["blobs"][statement["metadata"].replace("urn:cid:", "")]
        items.append(json.loads(base64.b64decode(blob)) | statement)
    return items


def test_codex_demo_exports_signed_allow_and_deny_graph(tmp_path):
    manifest = json.loads(Path(build_demo(tmp_path / "codex.json")).read_text())
    metadata = _metadata(manifest)

    assert metadata
    assert all(item["registeredBy"].startswith("did:key:") for item in metadata)
    assert sum(item.get("name") == "Codex user prompt" for item in metadata) == 1
    assert sum(item.get("decision") == "allow" for item in metadata) == 1
    assert sum(item.get("decision") == "deny" for item in metadata) == 1
    assert sum(item.get("computation_type") == "tool" for item in metadata) == 2


def test_denied_call_has_a_guardrail_and_no_result(tmp_path):
    manifest = json.loads(Path(build_demo(tmp_path / "codex.json")).read_text())
    tools = [item for item in _metadata(manifest) if item.get("computation_type") == "tool"]

    executed = {item["decision"]: item["executed"] for item in tools}
    assert executed == {"allow": True, "deny": False}

    names = {
        statement["@id"]: statement
        for statement in manifest["statements"].values()
        if statement.get("@type") == "ComputationRegistration"
    }
    denied = next(item for item in tools if item["decision"] == "deny")
    allowed = next(item for item in tools if item["decision"] == "allow")
    # The SDK writes a lone output as a bare CID and several as a list.
    assert len(_outputs(names[denied["subject"]])) == 1  # guardrail only
    assert len(_outputs(names[allowed["subject"]])) == 2  # guardrail + result


def test_a_real_capture_replays_into_the_same_shaped_graph(tmp_path):
    """End to end: real codex-cli payloads -> real collector -> capture file -> signed manifest."""
    capture = tmp_path / "codex-hooks.jsonl"
    env = os.environ.copy()
    env["EQTY_LINEAGE_CAPTURE"] = str(capture)
    env["EQTY_LINEAGE_DENY_COMMAND"] = DENIED_COMMAND

    denied_ids = set()
    for payload in json.loads(FIXTURE.read_text()):
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
            denied_ids.add(payload.get("tool_use_id"))

    assert denied_ids, "the collector never denied anything; the rest of this test proves nothing"

    manifest = json.loads(Path(replay_capture(capture, tmp_path / "replayed.json")).read_text())
    tools = [item for item in _metadata(manifest) if item.get("computation_type") == "tool"]

    assert {item["decision"] for item in tools} == {"allow", "deny"}
    assert all(item["session_id"] == "019f9ff2-2dd9-77f0-b687-22723134122c" for item in tools)
    assert {item["name"] for item in tools} == {"Codex apply_patch", "Codex Bash"}


def test_replay_of_an_empty_capture_is_an_empty_graph(tmp_path):
    capture = tmp_path / "empty.jsonl"
    capture.write_text("")

    manifest = json.loads(Path(replay_capture(capture, tmp_path / "empty.json")).read_text())

    assert [item for item in _metadata(manifest) if item.get("computation_type") == "tool"] == []


@pytest.mark.parametrize("decision", ["Allow", "denied", ""])
def test_an_unrecognized_decision_is_rejected_rather_than_recorded(tmp_path, decision):
    from eqty_lineage.codex import CodexLineage

    lineage = CodexLineage(tmp_path / "x.json")
    with pytest.raises(ValueError):
        lineage.tool("Bash", {}, decision=decision)
