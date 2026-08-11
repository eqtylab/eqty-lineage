"""The command line, tested through `main()` and through the installed console script."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from eqty_lineage.codex.__main__ import default_output, main

FIXTURE = Path(__file__).parents[1] / "tests" / "fixtures" / "codex_hooks.json"


def _tools(manifest_path):
    import base64

    manifest = json.loads(Path(manifest_path).read_text())
    out = []
    for statement in manifest["statements"].values():
        if statement.get("@type") != "MetadataRegistration":
            continue
        data = json.loads(base64.b64decode(manifest["blobs"][statement["metadata"].replace("urn:cid:", "")]))
        if data.get("computation_type") == "tool":
            out.append(data)
    return out


def test_replays_a_capture_and_prints_the_manifest_path(tmp_path, capsys):
    output = tmp_path / "session.json"

    assert main([str(FIXTURE), "-o", str(output)]) == 0

    printed = capsys.readouterr().out.strip().splitlines()
    assert printed[-1] == str(output)
    assert output.exists()
    assert len(_tools(output)) == 2


def test_summary_lists_every_attempt(tmp_path, capsys):
    main([str(FIXTURE), "-o", str(tmp_path / "s.json")])

    out = capsys.readouterr().out
    assert "session 019f9ff2-2dd9-77f0-b687-22723134122c" in out
    assert "gpt-5.4-mini" in out
    assert out.count("allow") == 2
    assert "apply_patch" in out and "Bash" in out


def test_quiet_prints_only_the_path(tmp_path, capsys):
    output = tmp_path / "s.json"

    main([str(FIXTURE), "-o", str(output), "--quiet"])

    assert capsys.readouterr().out.strip() == str(output)


def test_json_summary_is_machine_readable(tmp_path, capsys):
    main([str(FIXTURE), "-o", str(tmp_path / "s.json"), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["session_id"] == "019f9ff2-2dd9-77f0-b687-22723134122c"
    assert payload["prompts"] == 1
    assert [a["decision"] for a in payload["attempts"]] == ["allow", "allow"]


def test_output_defaults_to_a_sibling_of_the_capture(tmp_path, capsys):
    capture = tmp_path / "codex-hooks.jsonl"
    capture.write_text(FIXTURE.read_text())

    assert main([str(capture)]) == 0

    expected = tmp_path / "codex-hooks.lineage.json"
    assert expected.exists()
    assert capsys.readouterr().out.strip().endswith(str(expected))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("codex-hooks.jsonl", "codex-hooks.lineage.json"),
        ("run.json", "run.lineage.json"),
        ("capture", "capture.lineage.json"),
    ],
)
def test_default_output_naming(name, expected):
    assert default_output(Path("/tmp") / name).name == expected


def test_a_missing_capture_exits_2(tmp_path, capsys):
    assert main([str(tmp_path / "nope.jsonl")]) == 2
    assert "no such capture" in capsys.readouterr().err


def test_a_malformed_capture_exits_2_rather_than_traceback(tmp_path, capsys):
    capture = tmp_path / "bad.jsonl"
    capture.write_text("{not json\n")

    assert main([str(capture)]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_an_empty_capture_exits_3_rather_than_writing_a_silent_empty_graph(tmp_path, capsys):
    capture = tmp_path / "empty.jsonl"
    capture.write_text("")

    assert main([str(capture)]) == 3
    assert "no prompts or tool calls" in capsys.readouterr().err


def test_unresolved_calls_are_called_out_on_stderr(tmp_path, capsys):
    capture = tmp_path / "cut.jsonl"
    capture.write_text(
        json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_use_id": "x", "tool_input": {}}) + "\n"
    )

    main([str(capture), "-o", str(tmp_path / "s.json")])

    assert "unresolved" in capsys.readouterr().err


def test_a_deny_that_executed_is_called_out_on_stderr(tmp_path, capsys):
    capture = tmp_path / "failopen.jsonl"
    capture.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "collector": {"decision": "deny"},
                        "payload": {
                            "hook_event_name": "PreToolUse",
                            "tool_name": "Bash",
                            "tool_use_id": "x",
                            "tool_input": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "payload": {
                            "hook_event_name": "PostToolUse",
                            "tool_name": "Bash",
                            "tool_use_id": "x",
                            "tool_response": "ran",
                        }
                    }
                ),
            ]
        )
    )

    main([str(capture), "-o", str(tmp_path / "s.json")])

    assert "failed open" in capsys.readouterr().err


def test_the_console_script_is_installed_and_runnable(tmp_path):
    """`uv run eqty-codex-lineage` is what the docs tell people to type."""
    result = subprocess.run(
        ["eqty-codex-lineage", str(FIXTURE), "-o", str(tmp_path / "s.json"), "--quiet"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "s.json")


def test_module_entry_point_works(tmp_path):
    """`python -m eqty_lineage.codex`, for anyone who did not install the script."""
    result = subprocess.run(
        [sys.executable, "-m", "eqty_lineage.codex", str(FIXTURE), "-o", str(tmp_path / "s.json"), "--quiet"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "s.json")
