"""The live hook path, over real HTTP.

Exercised through the socket rather than by calling ``HookReceiver.handle`` directly, because the parts
most likely to break in deployment -- the bearer check, the JSON envelope, the response shape the agent
actually reads -- only exist at that boundary.
"""

import json
import urllib.error
import urllib.request

import pytest

from eqty_lineage.agent_hooks.daemon import HookReceiver, serve
from eqty_lineage.agent_hooks.policy import HookPolicy
from eqty_lineage.agent_hooks.session import SessionRegistry
from eqty_lineage.core import PERMISSIVE, prov

pytestmark = pytest.mark.usefixtures("sdk")

SESSION = "test-session-1"


@pytest.fixture
def daemon(tmp_path):
    """A receiver on an ephemeral port, with a policy and a manifest directory."""
    registry = SessionRegistry(
        state_dir=tmp_path / "state",
        policy=PERMISSIVE,
        triples_dir=tmp_path / "triples",
    )
    receiver = HookReceiver(
        registry=registry,
        policy=HookPolicy(allow_write_globs=("/repo/*",)),
        watch_paths=["/repo"],
        manifest_dir=tmp_path / "manifests",
    )
    (tmp_path / "manifests").mkdir()
    server, thread = serve(receiver, host="127.0.0.1", port=0, token="secret-token")
    try:
        yield server.server_address[1], receiver, tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def post(port, payload, token="secret-token"):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/hook",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read() or b"{}")


def hook(event, **fields):
    return {"hook_event_name": event, "session_id": SESSION, "prompt_id": "p1",
            "cwd": "/repo", "transcript_path": "/home/u/.claude/projects/x.jsonl", **fields}


class TestTransport:
    def test_a_missing_token_is_rejected(self, daemon):
        port, _, _ = daemon
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            post(port, hook("SessionStart"), token=None)
        assert excinfo.value.code == 401

    def test_a_wrong_token_is_rejected(self, daemon):
        port, _, _ = daemon
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            post(port, hook("SessionStart"), token="wrong")
        assert excinfo.value.code == 401

    def test_malformed_json_does_not_crash_the_daemon(self, daemon):
        port, _, _ = daemon
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/hook", data=b"{not json",
            headers={"Authorization": "Bearer secret-token"},
        )
        try:
            urllib.request.urlopen(request, timeout=10)
        except urllib.error.HTTPError:
            pass
        # The session it is observing must survive a bad request.
        assert post(port, hook("SessionStart")) is not None

    def test_an_unknown_hook_event_returns_empty_rather_than_failing(self, daemon):
        port, _, _ = daemon
        post(port, hook("SessionStart"))
        assert post(port, hook("SomeFutureEvent")) == {}


class TestRecording:
    def test_session_start_returns_watch_paths(self, daemon):
        port, _, _ = daemon
        response = post(port, hook("SessionStart"))
        # Returning watchPaths is what upgrades Bash side effects from inferred to observed -- the one
        # thing the offline path cannot match.
        assert response["hookSpecificOutput"]["watchPaths"] == ["/repo"]

    def test_a_tool_call_is_recorded_through_the_socket(self, daemon):
        port, receiver, _ = daemon
        post(port, hook("SessionStart"))
        post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1",
                        tool_input={"file_path": "/repo/a.py"}))
        post(port, hook("PostToolUse", tool_name="Edit", tool_use_id="t1",
                        tool_input={"file_path": "/repo/a.py"},
                        tool_response={"filePath": "/repo/a.py", "originalFile": "x = 1\n",
                                       "oldString": "1", "newString": "2", "structuredPatch": []}))
        state = receiver.registry.get(SESSION)
        assert [v.version for v in state.recorder.file_versions["/repo/a.py"]] == [1, 2]

    def test_a_failed_call_survives_in_the_graph(self, daemon):
        # PostToolUse fires only on success; without PostToolUseFailure every failed call vanishes --
        # and a failed call is often the one an audit cares about.
        port, receiver, _ = daemon
        post(port, hook("SessionStart"))
        post(port, hook("PreToolUse", tool_name="Bash", tool_use_id="t1",
                        tool_input={"command": "false"}))
        post(port, hook("PostToolUseFailure", tool_name="Bash", tool_use_id="t1",
                        tool_input={"command": "false"}, tool_response="exit 1"))
        state = receiver.registry.get(SESSION)
        assert "Bash" in [t.object for t in state.recorder.triples if t.predicate == prov.LABEL]

    def test_everything_recorded_live_is_observed(self, daemon):
        # The live path sees events happen; nothing here is inferred from a snapshot delta.
        port, receiver, _ = daemon
        post(port, hook("SessionStart"))
        post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1",
                        tool_input={"file_path": "/repo/a.py"}))
        post(port, hook("PostToolUse", tool_name="Edit", tool_use_id="t1",
                        tool_input={"file_path": "/repo/a.py"},
                        tool_response={"filePath": "/repo/a.py", "content": "new\n"}))
        state = receiver.registry.get(SESSION)
        assert all(t.observed for t in state.recorder.triples)

    def test_a_recording_failure_never_propagates_to_the_agent(self, daemon):
        port, receiver, _ = daemon
        post(port, hook("SessionStart"))
        # A payload the adapter cannot make sense of must return {} and count an error, not 500.
        assert post(port, hook("PostToolUse", tool_name="Edit", tool_use_id="nonexistent",
                               tool_response={"filePath": "/repo/a.py", "content": "x"})) == {}


class TestPolicyOverTheWire:
    def test_a_write_outside_the_permitted_set_is_denied(self, daemon):
        port, _, _ = daemon
        post(port, hook("SessionStart"))
        response = post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1",
                                   tool_input={"file_path": "/etc/passwd"}))
        assert response["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_a_permitted_write_returns_no_decision_at_all(self, daemon):
        # Deferring, not allowing: answering "allow" would override the user's own settings.
        port, _, _ = daemon
        post(port, hook("SessionStart"))
        response = post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1",
                                   tool_input={"file_path": "/repo/a.py"}))
        assert "hookSpecificOutput" not in response


class TestExport:
    def test_session_end_writes_a_manifest(self, daemon):
        port, _, tmp_path = daemon
        post(port, hook("SessionStart"))
        post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1",
                        tool_input={"file_path": "/repo/a.py"}))
        post(port, hook("PostToolUse", tool_name="Edit", tool_use_id="t1",
                        tool_input={"file_path": "/repo/a.py"},
                        tool_response={"filePath": "/repo/a.py", "content": "new\n"}))
        post(port, hook("SessionEnd"))

        manifest = tmp_path / "manifests" / f"{SESSION}.json"
        assert manifest.exists()
        # An empty export is the failure mode when statements land in a different context from the
        # assets they reference -- it does not raise, it just exports nothing.
        assert manifest.stat().st_size > 100

    def test_a_triple_sidecar_is_written(self, daemon):
        port, _, tmp_path = daemon
        post(port, hook("SessionStart"))
        post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1",
                        tool_input={"file_path": "/repo/a.py"}))
        post(port, hook("PostToolUse", tool_name="Edit", tool_use_id="t1",
                        tool_input={"file_path": "/repo/a.py"},
                        tool_response={"filePath": "/repo/a.py", "content": "new\n"}))
        assert list((tmp_path / "triples").glob("*.jsonl"))
