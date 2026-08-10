#!/usr/bin/env bash
#
# Drive a real Codex session through the collector and assert the lineage it produces.
#
# The unit tests replay recorded payloads; only this exercises the part that cannot be faked -- that
# codex-cli honours the collector's denial. A malformed decision is discarded silently: the capture
# still records `deny`, the graph still renders one, and the command runs anyway. So the check that
# matters is the filesystem, not the log.
#
# Exit codes
#   0  the pipeline behaved
#   2  preconditions missing (no codex, no uv, no workspace) -- did not run, distinct from a failure
#   3  the session produced no usable capture
#   4  the denial was NOT enforced: the forbidden file exists
#   5  the graph does not match the session (missing allow/deny, or a denied call marked executed)
#   6  the session did not finish within VALIDATE_TIMEOUT seconds (default 300)
#
set -uo pipefail

KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTOR="$ROOT/plugins/eqty-lineage/scripts/capture_hook.py"
DENY_COMMAND='touch forbidden.txt'

fail() { printf '\033[31mFAIL\033[0m %s\n' "$1" >&2; exit "$2"; }
ok() { printf '\033[32m  ok\033[0m %s\n' "$1"; }

command -v codex >/dev/null || fail "codex is not on PATH" 2
command -v uv >/dev/null || fail "uv is not on PATH" 2
[ -f "$COLLECTOR" ] || fail "collector not found at $COLLECTOR" 2

WORK="$(mktemp -d)"
cleanup() { [ "$KEEP" = 1 ] && echo "kept: $WORK" || rm -rf "$WORK"; }
trap cleanup EXIT

export EQTY_LINEAGE_CAPTURE="$WORK/codex-hooks.jsonl"
export EQTY_LINEAGE_DENY_COMMAND="$DENY_COMMAND"

echo "codex:  $(codex --version 2>&1 | head -1)"
echo "work:   $WORK"

mkdir -p "$WORK/ws"
(
  cd "$WORK/ws"
  git init -q .
  git -c user.name=validate -c user.email=validate@example.invalid commit -q --allow-empty -m baseline
)
cat > "$WORK/ws/TASK.md" <<'EOF'
Run exactly these two shell commands, in this order, and nothing else:
1. printf 'allowed\n'
2. touch forbidden.txt
Do not use apply_patch. Do not retry a command that is blocked. Then stop.
EOF

hook() { printf 'hooks.%s=[{hooks=[{type="command",command="python3 %s"}]}]' "$1" "$COLLECTOR"; }

# The agent is free to fail the task; what is under test is the runtime around it, so its exit status
# is deliberately not the verdict.
#
# `</dev/null` is load-bearing. With stdin left open codex reads it as additional input and blocks on
# EOF forever -- "Reading additional input from stdin..." and no session ever starts.
(
  cd "$WORK/ws"
  codex exec \
    --dangerously-bypass-hook-trust \
    --skip-git-repo-check \
    --sandbox workspace-write \
    -c "$(hook SessionStart)" \
    -c "$(hook UserPromptSubmit)" \
    -c "$(hook PreToolUse)" \
    -c "$(hook PostToolUse)" \
    -c "$(hook SessionEnd)" \
    "Read TASK.md and perform it exactly."
) >"$WORK/codex.out" 2>"$WORK/codex.err" </dev/null &
CODEX_PID=$!

# A gate that can hang is not a gate. `timeout` is not on a stock macOS, so poll for the deadline.
DEADLINE=$(( SECONDS + ${VALIDATE_TIMEOUT:-300} ))
while kill -0 "$CODEX_PID" 2>/dev/null; do
  if [ "$SECONDS" -ge "$DEADLINE" ]; then
    kill -TERM "$CODEX_PID" 2>/dev/null
    sleep 2
    kill -KILL "$CODEX_PID" 2>/dev/null
    fail "codex did not finish within ${VALIDATE_TIMEOUT:-300}s (see $WORK/codex.err)" 6
  fi
  sleep 2
done
wait "$CODEX_PID" 2>/dev/null || true

# --- 1. the session produced a capture the collector actually wrote to -----------------------------

[ -s "$EQTY_LINEAGE_CAPTURE" ] || fail "no capture at $EQTY_LINEAGE_CAPTURE (see $WORK/codex.err)" 3

python3 - <<'PY' || exit 3
import json, os, sys
rows = [json.loads(l) for l in open(os.environ["EQTY_LINEAGE_CAPTURE"]) if l.strip()]
pre = [r for r in rows if r["payload"].get("hook_event_name") == "PreToolUse"]
denied = [r for r in rows if r["collector"].get("decision") == "deny"]
if not pre:
    print("no PreToolUse events captured; the agent ran no tools", file=sys.stderr)
    sys.exit(3)
if len(denied) != 1:
    print(f"expected exactly one recorded denial, got {len(denied)}", file=sys.stderr)
    print("the agent likely never attempted the exact command, or varied it", file=sys.stderr)
    sys.exit(3)
PY
ok "capture recorded a denial"

# --- 2. the denial was enforced, not merely recorded -----------------------------------------------

[ -e "$WORK/ws/forbidden.txt" ] && fail "denied command still created forbidden.txt -- the hook failed open" 4
ok "denied write never happened"

# --- 3. the exported graph matches the session -----------------------------------------------------

( cd "$ROOT" && uv run python -c "
from eqty_lineage.codex import replay_capture
replay_capture('$EQTY_LINEAGE_CAPTURE', '$WORK/session.json')
" ) >"$WORK/replay.log" 2>&1 || fail "replay_capture failed (see $WORK/replay.log)" 5

( cd "$ROOT" && uv run python - "$WORK/session.json" <<'PY'
import base64, json, sys

manifest = json.load(open(sys.argv[1]))
tools = []
for statement in manifest["statements"].values():
    if statement.get("@type") != "MetadataRegistration":
        continue
    blob = manifest["blobs"][statement["metadata"].replace("urn:cid:", "")]
    data = json.loads(base64.b64decode(blob))
    if data.get("computation_type") == "tool":
        tools.append(data)

for tool in tools:
    print(f'       {tool["decision"]:8} executed={str(tool["executed"]):5} {tool["name"]}')

denied = [t for t in tools if t["decision"] == "deny"]
allowed = [t for t in tools if t["decision"] == "allow" and t["executed"]]
problems = []
if len(denied) != 1:
    problems.append(f"expected one denied call in the graph, got {len(denied)}")
if any(t["executed"] for t in denied):
    problems.append("a denied call is marked executed")
if not allowed:
    problems.append("no allowed call was recorded as executed")
for problem in problems:
    print(f"  {problem}", file=sys.stderr)
sys.exit(5 if problems else 0)
PY
) || fail "exported graph does not match the session" 5
ok "graph records one enforced denial and at least one executed call"

printf '\033[32mPASS\033[0m  codex integration validated end to end\n'
