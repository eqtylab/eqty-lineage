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
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read() or b"{}")


def hook(event, **fields):
    return {
        "hook_event_name": event,
        "session_id": SESSION,
        "prompt_id": "p1",
        "cwd": "/repo",
        "transcript_path": "/home/u/.claude/projects/x.jsonl",
        **fields,
    }


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
            f"http://127.0.0.1:{port}/hook",
            data=b"{not json",
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
        post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1", tool_input={"file_path": "/repo/a.py"}))
        post(
            port,
            hook(
                "PostToolUse",
                tool_name="Edit",
                tool_use_id="t1",
                tool_input={"file_path": "/repo/a.py"},
                tool_response={
                    "filePath": "/repo/a.py",
                    "originalFile": "x = 1\n",
                    "oldString": "1",
                    "newString": "2",
                    "structuredPatch": [],
                },
            ),
        )
        state = receiver.registry.get(SESSION)
        assert [v.version for v in state.recorder.file_versions["/repo/a.py"]] == [1, 2]

    def test_a_failed_call_survives_in_the_graph(self, daemon):
        # PostToolUse fires only on success; without PostToolUseFailure every failed call vanishes --
        # and a failed call is often the one an audit cares about.
        #
        # The payload shape is the real one: a failure carries `error`, not `tool_response`. Reading
        # the success key here recorded every failure as "tool returned no result", and a test written
        # against the same wrong shape passed throughout.
        port, receiver, _ = daemon
        post(port, hook("SessionStart"))
        post(port, hook("PreToolUse", tool_name="Bash", tool_use_id="t1", tool_input={"command": "false"}))
        post(
            port,
            hook(
                "PostToolUseFailure",
                tool_name="Bash",
                tool_use_id="t1",
                tool_input={"command": "false"},
                error="command failed: exit 1",
                duration_ms=12,
            ),
        )
        state = receiver.registry.get(SESSION)
        assert "Bash" in [t.object for t in state.recorder.triples if t.predicate == prov.LABEL]
        assert state.recorder.stats.get("ToolCallEnded") == 1

    def test_a_batch_closes_every_call_with_its_real_result(self, daemon):
        # PostToolBatch entries are {tool_name, tool_input, tool_use_id, tool_response} -- the same
        # result key as PostToolUse. Reading `result`/`is_error` closed each call with nothing and
        # dropped the file lineage the batch carried.
        port, receiver, _ = daemon
        post(port, hook("SessionStart"))
        for call_id in ("t1", "t2"):
            post(
                port,
                hook(
                    "PreToolUse", tool_name="Edit", tool_use_id=call_id, tool_input={"file_path": f"/repo/{call_id}.py"}
                ),
            )
        post(
            port,
            hook(
                "PostToolBatch",
                tool_calls=[
                    {
                        "tool_name": "Edit",
                        "tool_use_id": "t1",
                        "tool_input": {"file_path": "/repo/t1.py"},
                        "tool_response": {"filePath": "/repo/t1.py", "content": "a = 1\n"},
                    },
                    {
                        "tool_name": "Edit",
                        "tool_use_id": "t2",
                        "tool_input": {"file_path": "/repo/t2.py"},
                        "tool_response": {"filePath": "/repo/t2.py", "content": "b = 2\n"},
                    },
                ],
            ),
        )
        state = receiver.registry.get(SESSION)
        assert sorted(state.recorder.file_versions) == ["/repo/t1.py", "/repo/t2.py"]
        assert state.recorder.stats.get("ToolCallEnded") == 2

    def test_the_recorder_does_not_observe_its_own_writes(self, daemon):
        # --watch <repo> reports every change under a tree. If the sidecars and the SDK blob store live
        # inside it, recording a file version writes blobs, the watcher reports them, and the recorder
        # records those. A live session produced 55 versions of which 54 were its own blobs.
        port, receiver, tmp_path = daemon
        post(port, hook("SessionStart"))
        for noise in (
            tmp_path / "triples" / "s.jsonl",
            tmp_path / "manifests" / "s.json",
            tmp_path / "state" / "s.json",
            tmp_path / ".eqty_sdk" / "blobs" / "baga6yaq6e",
        ):
            noise.parent.mkdir(parents=True, exist_ok=True)
            noise.write_text("internal")
            post(port, hook("FileChanged", file_path=str(noise), change_type="change"))

        real = tmp_path / "real.py"
        real.write_text("x = 1\n")
        post(port, hook("FileChanged", file_path=str(real), change_type="change"))

        state = receiver.registry.get(SESSION)
        assert list(state.recorder.file_versions) == [str(real)]

    def test_an_edit_the_agent_really_made_is_kept_even_under_a_watched_dir(self, daemon):
        # Only watcher events are filtered. A tool call that edits a file there is the agent's doing.
        port, receiver, tmp_path = daemon
        target = tmp_path / "manifests" / "checked_in.json"
        post(port, hook("SessionStart"))
        post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1", tool_input={"file_path": str(target)}))
        post(
            port,
            hook(
                "PostToolUse",
                tool_name="Edit",
                tool_use_id="t1",
                tool_response={"filePath": str(target), "content": "{}\n"},
            ),
        )
        state = receiver.registry.get(SESSION)
        assert str(target) in state.recorder.file_versions

    def test_a_watcher_removal_is_a_tombstone_not_a_write(self, daemon):
        # change_type is "change" or "unlink". A removal read back off disk yields no content, which is
        # byte-identical to a file we failed to read -- and means the opposite thing.
        port, receiver, tmp_path = daemon
        target = tmp_path / "gone.py"
        target.write_text("x = 1\n")
        post(port, hook("SessionStart"))
        post(port, hook("FileChanged", file_path=str(target), change_type="change"))
        target.unlink()
        post(port, hook("FileChanged", file_path=str(target), change_type="unlink"))

        state = receiver.registry.get(SESSION)
        versions = state.recorder.file_versions[str(target)]
        assert [v.content_cid.startswith("deleted:") for v in versions] == [False, True]

    def test_everything_recorded_live_is_observed(self, daemon):
        # The live path sees events happen; nothing here is inferred from a snapshot delta.
        port, receiver, _ = daemon
        post(port, hook("SessionStart"))
        post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1", tool_input={"file_path": "/repo/a.py"}))
        post(
            port,
            hook(
                "PostToolUse",
                tool_name="Edit",
                tool_use_id="t1",
                tool_input={"file_path": "/repo/a.py"},
                tool_response={"filePath": "/repo/a.py", "content": "new\n"},
            ),
        )
        state = receiver.registry.get(SESSION)
        assert all(t.observed for t in state.recorder.triples)

    def test_a_recording_failure_never_propagates_to_the_agent(self, daemon):
        port, _receiver, _ = daemon
        post(port, hook("SessionStart"))
        # A payload the adapter cannot make sense of must return {} and count an error, not 500.
        assert (
            post(
                port,
                hook(
                    "PostToolUse",
                    tool_name="Edit",
                    tool_use_id="nonexistent",
                    tool_response={"filePath": "/repo/a.py", "content": "x"},
                ),
            )
            == {}
        )


class TestPolicyOverTheWire:
    def test_a_write_outside_the_permitted_set_is_denied(self, daemon):
        port, _, _ = daemon
        post(port, hook("SessionStart"))
        response = post(
            port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1", tool_input={"file_path": "/etc/passwd"})
        )
        assert response["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_a_permitted_write_returns_no_decision_at_all(self, daemon):
        # Deferring, not allowing: answering "allow" would override the user's own settings.
        port, _, _ = daemon
        post(port, hook("SessionStart"))
        response = post(
            port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1", tool_input={"file_path": "/repo/a.py"})
        )
        assert "hookSpecificOutput" not in response


class TestExport:
    def test_session_end_writes_a_manifest(self, daemon):
        port, _, tmp_path = daemon
        post(port, hook("SessionStart"))
        post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1", tool_input={"file_path": "/repo/a.py"}))
        post(
            port,
            hook(
                "PostToolUse",
                tool_name="Edit",
                tool_use_id="t1",
                tool_input={"file_path": "/repo/a.py"},
                tool_response={"filePath": "/repo/a.py", "content": "new\n"},
            ),
        )
        post(port, hook("SessionEnd"))

        manifest = tmp_path / "manifests" / f"{SESSION}.json"
        assert manifest.exists()
        # An empty export is the failure mode when statements land in a different context from the
        # assets they reference -- it does not raise, it just exports nothing.
        assert manifest.stat().st_size > 100

    def test_a_triple_sidecar_is_written(self, daemon):
        port, _, tmp_path = daemon
        post(port, hook("SessionStart"))
        post(port, hook("PreToolUse", tool_name="Edit", tool_use_id="t1", tool_input={"file_path": "/repo/a.py"}))
        post(
            port,
            hook(
                "PostToolUse",
                tool_name="Edit",
                tool_use_id="t1",
                tool_input={"file_path": "/repo/a.py"},
                tool_response={"filePath": "/repo/a.py", "content": "new\n"},
            ),
        )
        assert list((tmp_path / "triples").glob("*.jsonl"))


class TestResponsesAreDialectCorrect:
    """A hook answer that names a key the agent does not know can lose the whole response.

    Codex's output wires are `additionalProperties: false`. `watchPaths` appears **zero** times in the
    codex-cli 0.148.0 binary -- it is a Claude Code affordance -- so returning it to Codex risks the
    response being discarded, taking any permission decision in it along with it.
    """

    def _receiver(self, tmp_path, watch=("/repo",)):
        from eqty_lineage.agent_hooks.daemon import HookReceiver
        from eqty_lineage.agent_hooks.session import SessionRegistry

        return HookReceiver(
            registry=SessionRegistry(state_dir=tmp_path / "state", triples_dir=tmp_path / "triples"),
            watch_paths=list(watch),
        )

    def _session_start(self, dialect):
        payload = {
            "hook_event_name": "SessionStart",
            "session_id": f"s-{dialect}",
            "cwd": "/repo",
            "model": "m",
            "source": "startup",
        }
        # Lifecycle events carry no turn id, so the dialect comes from the transcript path.
        payload["transcript_path"] = (
            "/home/u/.codex/sessions/x.jsonl" if dialect == "codex" else "/home/u/.claude/projects/x.jsonl"
        )
        return payload

    def test_claude_code_still_receives_watch_paths(self, sdk, tmp_path):
        # The feature this guards must keep working; it is what makes Bash side effects observed.
        response = self._receiver(tmp_path).handle(self._session_start("claude-code"))
        assert response["hookSpecificOutput"]["watchPaths"] == ["/repo"]

    def test_codex_does_not(self, sdk, tmp_path):
        response = self._receiver(tmp_path).handle(self._session_start("codex"))
        assert "watchPaths" not in json.dumps(response)

    def test_a_codex_session_start_is_still_recorded(self, sdk, tmp_path):
        # Withholding the response must not withhold the recording.
        receiver = self._receiver(tmp_path)
        receiver.handle(self._session_start("codex"))
        assert receiver.handled == 1
        assert receiver.errors == 0
