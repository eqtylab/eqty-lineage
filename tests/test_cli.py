"""The command line, including the wiring it tells people to paste into their settings.

`install` output is the first thing anyone runs, and a config that names a flag the parser does not
accept -- or forwards no credential to a daemon that requires one -- fails at the point of first use.
"""

import json
import re

import pytest

from eqty_lineage.agent_hooks.__main__ import API_KEY_ENV, TOKEN_ENV, _secret, main


def run(capsys, *argv):
    assert main(list(argv)) == 0
    return capsys.readouterr().out


class TestInstall:
    def test_the_documented_print_flag_is_accepted(self, capsys):
        # Both READMEs document `install --print`. It used to exit 2.
        assert "hooks" in run(capsys, "install", "--print")

    def test_claude_code_wiring_is_valid_json(self, capsys):
        out = run(capsys, "install")
        body = out[out.index("{"):]
        config = json.loads(body)
        assert set(config["hooks"]) >= {"SessionStart", "PreToolUse", "PostToolUse", "SessionEnd"}

    def test_the_token_travels_as_an_env_reference_not_a_literal(self, capsys):
        # settings.json is commonly committed; a literal secret in this output would end up in git.
        out = run(capsys, "install")
        config = json.loads(out[out.index("{"):])
        handler = config["hooks"]["PreToolUse"][0]["hooks"][0]
        assert handler["headers"]["Authorization"] == f"Bearer ${TOKEN_ENV}"
        assert handler["allowedEnvVars"] == [TOKEN_ENV]

    def test_the_token_header_can_be_omitted(self, capsys):
        out = run(capsys, "install", "--no-token")
        config = json.loads(out[out.index("{"):])
        assert "headers" not in config["hooks"]["PreToolUse"][0]["hooks"][0]

    def test_the_port_reaches_the_emitted_url(self, capsys):
        out = run(capsys, "install", "--port", "9999")
        config = json.loads(out[out.index("{"):])
        assert config["hooks"]["PreToolUse"][0]["hooks"][0]["url"].endswith(":9999/hook")

    def test_session_start_is_a_command_hook(self, capsys):
        # SessionStart accepts no HTTP handler. Wiring it as HTTP is accepted by the settings file and
        # then never delivered -- so no Agent asset is created and, because SessionStart is where
        # watchPaths is returned, no FileChanged ever fires. Confirmed by running a live session.
        out = run(capsys, "install")
        config = json.loads(out[out.index("{"):])
        handler = config["hooks"]["SessionStart"][0]["hooks"][0]
        assert handler["type"] == "command"
        assert "curl" in handler["command"]

    def test_only_session_start_falls_back_to_a_command(self, capsys):
        out = run(capsys, "install")
        config = json.loads(out[out.index("{"):])
        by_type = {e: h[0]["hooks"][0]["type"] for e, h in config["hooks"].items()}
        assert {e for e, t in by_type.items() if t == "command"} == {"SessionStart"}

    def test_the_command_hook_quotes_survive_json_encoding(self, capsys):
        # json.dumps escapes quotes itself. Escaping them again first put a literal \" in the file, and
        # the shell then split the Authorization header into three arguments -- the hook ran, curl
        # failed, and the daemon simply never heard from SessionStart.
        out = run(capsys, "install")
        config = json.loads(out[out.index("{"):])
        command = config["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        assert "\\" not in command
        assert f'-H "Authorization: Bearer ${TOKEN_ENV}"' in command

    def test_the_command_hook_is_shell_parseable(self, capsys):
        import shlex

        out = run(capsys, "install")
        config = json.loads(out[out.index("{"):])
        argv = shlex.split(config["hooks"]["SessionStart"][0]["hooks"][0]["command"])
        assert argv[0] == "curl"
        # The header must survive as one argument, not three.
        assert f"Authorization: Bearer ${TOKEN_ENV}" in argv

    def test_codex_wiring_forwards_the_same_credential(self, capsys):
        # Codex has no HTTP hook type, so the header has to ride on the curl -- otherwise every hook
        # 401s against a daemon started with a token.
        out = run(capsys, "install", "--dialect", "codex")
        assert f"Authorization: Bearer ${TOKEN_ENV}" in out
        assert out.count("[[hooks.") >= 8

    def test_codex_wiring_is_toml_shaped(self, capsys):
        out = run(capsys, "install", "--dialect", "codex")
        for line in out.splitlines():
            if line.startswith("[[hooks."):
                assert re.fullmatch(r"\[\[hooks\.\w+(\.hooks)?\]\]", line), line

    def test_codex_wiring_parses_as_toml(self, capsys):
        # TOML needs the inner quotes escaped by hand -- the opposite of the JSON path, which is
        # exactly why one helper serves both and the difference is a parameter.
        tomllib = pytest.importorskip("tomllib")

        out = run(capsys, "install", "--dialect", "codex")
        body = "\n".join(line for line in out.splitlines() if not line.startswith("#"))
        config = tomllib.loads(body)
        command = config["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        assert f'-H "Authorization: Bearer ${TOKEN_ENV}"' in command


class TestSecrets:
    def test_the_environment_supplies_the_token(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV, "from-env")
        assert _secret(None, TOKEN_ENV) == "from-env"

    def test_an_explicit_flag_still_wins(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV, "from-env")
        assert _secret("from-flag", TOKEN_ENV) == "from-flag"

    def test_an_empty_environment_variable_is_not_a_token(self, monkeypatch):
        # "" would otherwise configure a daemon whose bearer check compares against the empty string.
        monkeypatch.setenv(API_KEY_ENV, "")
        assert _secret(None, API_KEY_ENV) is None

    def test_nothing_configured_is_none(self, monkeypatch):
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        assert _secret(None, TOKEN_ENV) is None


class TestDefaults:
    def test_blob_storage_is_off_unless_asked_for(self):
        # --blobs maps to set_store_all_blobs, which is how a .env the agent read becomes durable.
        from eqty_lineage.agent_hooks.__main__ import main as _  # noqa: F401
        from eqty_lineage.agent_hooks.session import SessionRegistry

        assert SessionRegistry().store_blobs is False

    def test_an_unknown_subcommand_is_rejected(self):
        with pytest.raises(SystemExit):
            main(["nonsense"])
