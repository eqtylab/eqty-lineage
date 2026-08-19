"""The Codex plugin hook: a relay to the daemon that keeps the raw capture.

Its predecessor wrote payloads to a file for offline replay, which produced a manifest of prompts and
tool calls and **no file lineage at all** -- Codex writes through ``apply_patch``, whose
``tool_input.command`` is a patch document that only the daemon's dialect parses. Routed through the
daemon the same session records the files it wrote, with content CIDs and derivation edges.

Driven by subprocess rather than by import: the script is what Codex actually executes, and a test
that imported it would not exercise the stdin/stdout contract that carries permission decisions.
"""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

HOOK = Path(__file__).parent.parent / "plugins" / "eqty-lineage" / "scripts" / "eqty_hook.py"
HOOKS_JSON = Path(__file__).parent.parent / "plugins" / "eqty-lineage" / "hooks" / "hooks.json"

#: Every event codex-cli 0.148.0 emits, from the binary's own contiguous enum.
BINARY_ENUM = {
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "SubagentStart",
    "SubagentStop",
    "Stop",
}


@pytest.fixture
def daemon():
    """A stub daemon recording what it received and replying with what the test sets."""
    received = []
    reply = {"body": {}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            received.append(
                {
                    "payload": json.loads(self.rfile.read(length) or b"{}"),
                    "authorization": self.headers.get("Authorization"),
                }
            )
            body = json.dumps(reply["body"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{server.server_port}/hook", "received": received, "reply": reply}
    server.shutdown()


def run_hook(payload, env=None, url=None):
    environment = {"PATH": "/usr/bin:/bin", "EQTY_LINEAGE_CAPTURE": "off"}
    if url:
        environment["EQTY_LINEAGE_URL"] = url
    environment.update(env or {})
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
        check=False,  # a non-zero exit is a result to assert on, not an error
    )
    return result


class TestSubscription:
    def test_the_plugin_subscribes_to_every_event_codex_emits(self):
        subscribed = set(json.loads(HOOKS_JSON.read_text())["hooks"])
        assert subscribed == BINARY_ENUM

    def test_every_subscription_runs_the_relay(self):
        hooks = json.loads(HOOKS_JSON.read_text())["hooks"]
        for name, entries in hooks.items():
            command = entries[0]["hooks"][0]["command"]
            assert "eqty_hook.py" in command, name
            assert "$PLUGIN_ROOT" in command, f"{name} must resolve relative to the plugin"


class TestRelay:
    def test_the_payload_reaches_the_daemon_verbatim(self, daemon):
        payload = {"hook_event_name": "PreToolUse", "session_id": "s", "tool_name": "Bash"}
        run_hook(payload, url=daemon["url"])
        assert daemon["received"][0]["payload"] == payload

    def test_the_daemons_answer_is_passed_through(self, daemon):
        # The daemon already speaks Codex's output shape, so its decisions govern Codex sessions.
        daemon["reply"]["body"] = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "denied by policy",
            }
        }
        result = run_hook({"hook_event_name": "PreToolUse"}, url=daemon["url"])
        assert json.loads(result.stdout) == daemon["reply"]["body"]

    def test_an_empty_answer_prints_nothing(self, daemon):
        # Codex rejects unknown keys; emitting `{}` for every event is noise at best.
        result = run_hook({"hook_event_name": "PostToolUse"}, url=daemon["url"])
        assert result.stdout.strip() == ""

    def test_the_token_is_sent_when_set_and_absent_otherwise(self, daemon):
        run_hook({"hook_event_name": "Stop"}, env={"EQTY_LINEAGE_TOKEN": "abc"}, url=daemon["url"])
        assert daemon["received"][0]["authorization"] == "Bearer abc"
        run_hook({"hook_event_name": "Stop"}, url=daemon["url"])
        assert daemon["received"][1]["authorization"] is None


class TestItNeverBreaksTheSession:
    """A lineage collector that takes down the agent it observes is worse than no collector."""

    def test_an_unreachable_daemon_is_not_an_error(self):
        result = run_hook({"hook_event_name": "PreToolUse"}, url="http://127.0.0.1:9/hook")
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_unreadable_stdin_is_not_an_error(self):
        result = subprocess.run(
            [sys.executable, str(HOOK)],
            input="not json",
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "EQTY_LINEAGE_CAPTURE": "off"},
            timeout=30,
            check=False,
        )
        assert result.returncode == 0


class TestFallbackDeny:
    def test_a_configured_denial_still_holds_when_the_daemon_is_down(self):
        result = run_hook(
            {"hook_event_name": "PreToolUse", "tool_input": {"command": "rm -rf /"}},
            env={"EQTY_LINEAGE_DENY_COMMAND": "rm -rf /"},
            url="http://127.0.0.1:9/hook",
        )
        assert json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_the_deny_output_carries_the_event_name(self):
        # `hookEventName` is required and unknown keys are rejected. Omitting it does not raise --
        # the decision is silently dropped and the command runs. Verified against codex-cli 0.147.0,
        # where an earlier shape let a denied `touch` create its file.
        result = run_hook(
            {"hook_event_name": "PreToolUse", "tool_input": {"command": "x"}},
            env={"EQTY_LINEAGE_DENY_COMMAND": "x"},
            url="http://127.0.0.1:9/hook",
        )
        assert json.loads(result.stdout)["hookSpecificOutput"]["hookEventName"] == "PreToolUse"

    def test_an_unconfigured_collector_denies_nothing(self):
        # The trap: comparing an unset variable straight against a missing `command` makes both None
        # and denies every tool call that carries no command.
        result = run_hook(
            {"hook_event_name": "PreToolUse", "tool_input": {}},
            url="http://127.0.0.1:9/hook",
        )
        assert result.stdout.strip() == ""

    def test_the_daemon_outranks_the_local_rule(self, daemon):
        # The daemon's policy is the real one; the local rule exists only for when it cannot be asked.
        result = run_hook(
            {"hook_event_name": "PreToolUse", "tool_input": {"command": "x"}},
            env={"EQTY_LINEAGE_DENY_COMMAND": "x"},
            url=daemon["url"],
        )
        assert result.stdout.strip() == "", "a reachable daemon that allowed it must not be overridden"


class TestCapture:
    def test_the_raw_payload_is_kept_alongside_the_relay(self, daemon, tmp_path):
        capture = tmp_path / "codex-hooks.jsonl"
        payload = {"hook_event_name": "SessionStart", "session_id": "s"}
        run_hook(payload, env={"EQTY_LINEAGE_CAPTURE": str(capture)}, url=daemon["url"])
        record = json.loads(capture.read_text().strip())
        assert record["payload"] == payload
        assert record["collector"]["relayed"] is True

    def test_a_failed_relay_is_recorded_as_such(self, tmp_path):
        # A silent fallback is how a session ends up with a capture file nobody knows is the only copy.
        capture = tmp_path / "codex-hooks.jsonl"
        run_hook(
            {"hook_event_name": "SessionStart"},
            env={"EQTY_LINEAGE_CAPTURE": str(capture)},
            url="http://127.0.0.1:9/hook",
        )
        collector = json.loads(capture.read_text().strip())["collector"]
        assert collector["relayed"] is False
        assert collector["relay_error"]

    def test_capture_can_be_turned_off(self, daemon, tmp_path):
        run_hook({"hook_event_name": "Stop"}, env={"EQTY_LINEAGE_CAPTURE": "off"}, url=daemon["url"])
        assert not list(tmp_path.glob("*.jsonl"))
