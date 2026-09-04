# Live session script

Drives every path the recorder can record today, and reports `YES` / `MISSING` per capability.

It exists because the first two live sessions reached for `Bash` almost exclusively and so exercised
about a third of the recorder: no `Document` nodes, no subagents, and none of the file counters.
Fixtures cover those; a real session had not.

Everything below has been run. The results quoted are from a real session, not expectations.

---

## What a human has to do

Four things. Everything else is automated by the driver in Path B.

1. **Turn Claude Code's auto mode OFF** (interactive runs only — Path A).
2. **Lower the content ceiling** in `plugins.toml`, and put it back afterwards.
3. **Build, stage and register the plugin** — `just nemo-relay-package`, then `plugins add`.
4. **Read the verification output**, starting with the captured-prompt count.

### 1. Auto mode off

Auto mode routes work through `Bash` in preference to `Read`, `Write` and `Edit` — exactly the path
with no observable file effects. With it on, this script cannot exercise file lineage however the
turns are phrased: measured, a full run under auto mode produced two files on disk and not one write
in the graph.

Path B does not need this: `--disallowedTools Bash` forbids the escape route outright, which is a
stronger lever than prompt phrasing.

Auto mode is worth recording *later*, once the declarative path is proven — it is a faithful picture
of the executive-tool gap. It is useless for proving the recorder works.

### 2. Lower the ceiling

The default is 100 MiB and generating a file that large to prove the ceiling works is a waste. Add to
the `[plugins.dynamic.config]` block of `~/.config/nemo-relay/plugins.toml`:

```toml
max_content_bytes = 8192       # 8 KiB, for this test only
```

Worth doing for its own sake: it is the only step that exercises the config override end to end, and
a value the host sets but the plugin ignores looks identical to one that worked.

**Put it back afterwards**, or every later session records file bytes by CID only.

### 3. Stage and register

```bash
just nemo-relay-package
nemo-relay plugins add --user ./dist/relay-plugin/relay-plugin.toml
nemo-relay plugins enable eqty.lineage
```

The recipe prints the `[plugins.policy.overrides."eqty.lineage"]` block to paste into `plugins.toml`.
Activation hardens attestation to `signature_required`, so an unsigned plugin installs cleanly,
validates cleanly, and then refuses to start. Re-run `just nemo-relay-package` after any code change
— the digest changes and a stale one is an install Relay refuses.

## Setup

```bash
rm -rf /tmp/relay-live && mkdir -p /tmp/relay-live && cd /tmp/relay-live
printf 'alpha\nbeta\ngamma\n' > notes.md
printf 'SECRET_KEY=do-not-record-me\n' > .env
python3 -c "open('big.txt','w').write('x'*200000)"     # Read caps output near 21 KB, so the
                                                       # ceiling has to sit below that
printf '\x89PNG\r\n\x1a\n\xff\xfe\x00\x01' > tiny.bin  # deliberately not valid UTF-8
```

---

## Path A — by hand

```bash
nemo-relay run -- claude
```

Send the turns below one at a time, letting each finish. Auto mode must be off. Claude picks tools on
its own, so each turn names one explicitly — and may still route around it, which is what Path B
prevents.

## Path B — headless, reproducible

One process, many turns, `Bash` forbidden for turns 1-9. This is how the quoted results were
produced.

```bash
cat > /tmp/drive.py <<'PYEOF'
import json, subprocess, threading, queue, time

TURNS = [
 "Use the Write tool to create report.md containing exactly three lines: one, two, three",
 "Use the Read tool to read report.md",
 'Use the Edit tool to change "two" to "TWO" in report.md',
 "Use the Read tool to read only lines 1 to 2 of report.md",
 "Use the Read tool on notes.md, then the Write tool to create summary.md holding its first line",
 "Use the Task tool to launch a subagent that reads summary.md and reports how many characters it has",
 "Use the Read tool on .env",
 "Use the Read tool on big.txt",
 "Use the Read tool on tiny.bin",
]

cmd = ["nemo-relay","run","--","claude","-p",
       "--input-format","stream-json","--output-format","stream-json","--verbose",
       "--permission-mode","acceptEdits","--disallowedTools","Bash"]
p = subprocess.Popen(cmd, cwd="/tmp/relay-live", stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
q = queue.Queue()
def reader():
    for line in p.stdout:
        q.put(line.rstrip("\n"))
    q.put(None)
threading.Thread(target=reader, daemon=True).start()

def send(text):
    p.stdin.write(json.dumps({"type":"user","message":{"role":"user",
        "content":[{"type":"text","text":text}]}}) + "\n")
    p.stdin.flush()

def await_result(n, budget=300):
    end = time.time() + budget
    while time.time() < end:
        try: line = q.get(timeout=5)
        except queue.Empty: continue
        if line is None: return "EOF"
        try: ev = json.loads(line)
        except Exception: continue
        if ev.get("type") == "assistant":
            for c in ev.get("message",{}).get("content",[]):
                if c.get("type") == "tool_use":
                    print("    tool: " + str(c.get("name")), flush=True)
        if ev.get("type") == "result":
            print("  turn %d: %s" % (n, ev.get("subtype")), flush=True)
            return "ok"
    return "timeout"

for i, t in enumerate(TURNS, 1):
    print("turn %d: %s" % (i, t[:70]), flush=True)
    send(t)
    if await_result(i) != "ok":
        print("  turn %d did not complete" % i, flush=True); break
p.stdin.close()
try: p.wait(timeout=90)
except Exception: p.kill()
print("finished, exit %s" % p.returncode, flush=True)
PYEOF
python3 /tmp/drive.py 2>&1 | tee /tmp/drive-out.txt
```

**Do not pipe the driver through `tail`.** It buffers everything until the process exits, so nothing
is visible while it runs — and the checkpointing means the manifest *is* readable mid-session, which
is how you tell early whether a fix held.

Two turns Path B cannot drive, because they are interactive slash commands with no `-p` equivalent:
`/compact` (exercises `Compaction`) and `/exit`. Run those by hand in Path A if you need them.

## Turns

| # | Prompt | Exercises |
|---|---|---|
| 1 | Write `report.md`, three lines | `Document` node, exact content CID, `FileWritten` |
| 2 | Read `report.md` | read-by-content-match; must **not** create a second version or a 2-cycle |
| 3 | Edit `two` → `TWO` | version chain read→write |
| 4 | Read only lines 1-2 | fragment → no content → `ContentUnknown` |
| 5 | Read `notes.md`, Write `summary.md` | two files, one turn; read of A → write of B |
| 6 | Task a subagent to read `summary.md` | subagent node, named by `agent_type`; `performedByInstance` |
| 7 | Read `.env` | `ContentDenied`: path and true CID kept, bytes withheld |
| 8 | Read `big.txt` | truncated read → **no content**; a fragment's hash is not the file's |
| 9 | Read `tiny.bin` | Read refuses binaries outright — the node records the refusal |

## Verify afterwards

```bash
python3 - <<'PY'
import json, base64, collections, glob
p = sorted(glob.glob('/tmp/relay-live/.eqty/manifests/*.json'))
print('manifests:', len(p), '-- more than one means a session was split')
m = json.load(open(p[-1])); B = m['blobs']
def blob(c):
    raw = B.get(c.replace('urn:cid:','')) or B.get(c)
    try: return json.loads(base64.b64decode(raw)) if raw else None
    except Exception: return None
assets, cov, redacted, agents = collections.Counter(), None, [], []
for c in B:
    d = blob(c)
    if not isinstance(d, dict): continue
    if 'assetType' in d:
        assets[d['assetType']] += 1
        if d['assetType'] == 'Agent': agents.append(d.get('name'))
    if d.get('name') == 'coverage': cov = d['coverage']
    if d.get('redacted'): redacted.append(str(d.get('name')).split('/')[-1])
prompts = [d for d in map(blob, B) if isinstance(d, dict)
           and d.get('assetType') == 'Prompt' and d.get('name') == 'user prompt']
print('statements', len(m['statements']))
print(f'user prompts captured: {len(prompts)} of the 9 sent')
print('agents    ', sorted(agents))
print('assets    ', dict(assets))
print('coverage  ', cov)
print('withheld  ', redacted)
print('secret leaked:', b"do-not-record-me" in open(p[-1],'rb').read())
required = [
    ('FileRead',        'a file was read into the graph at all'),
    ('FileWritten',     'a file version was written'),
    ('ContentUnknown',  'a fragment became an identity-only node'),
    ('ContentDenied',   'deny_globs withheld .env'),
]
expected_absent = [
    ('ContentRecovered','Edit states its new content, so the replay chain is never needed'),
    ('ContentTooLarge', 'Read truncates near 21 KB and truncation is detected first'),
]
for key, why in required:
    print(f'  {key:18} {"YES" if cov and key in cov else "MISSING":8} {why}')
for key, why in expected_absent:
    seen = cov and key in cov
    print(f'  {key:18} {"YES" if seen else "absent":8} {"unexpected -- investigate" if seen else why}')
print(f'  {"Document nodes":18} {"YES" if assets.get("Document") else "MISSING":8} file lineage produced nodes')
print(f'  {"subagent":18} {"YES" if assets.get("Agent",0) >= 2 else "MISSING":8} a subagent was recorded')
PY
```

### Reading the output

**Check the prompt count and the manifest count first.** If fewer turns were captured than you sent,
or there is more than one manifest, the session was split and every `MISSING` below is unexplained
rather than informative.

That happened, and it took two runs to understand: a subagent's scope end was classified as the
session ending, so the manifest exported at turn 6 and turns 7-9 overwrote it in a fresh recorder.
The survivor read as a complete short session rather than the tail of a truncated one.

A run reporting no `Agent` has a related cause — `session.start` was never seen, so nothing is
attributed to an actor.

**Two counters are expected absent on Claude Code**, and their absence is correct. `ContentTooLarge`
is unreachable through `Read`: the tool caps its output near 21 KB and truncation is detected before
the size decision, so the node records no content and never reaches the ceiling. `ContentRecovered`
is unreachable because `Edit` states its new content, so the `_last_content` replay chain is never
needed — it earns its place on Codex and older hosts, not this one. If either reports `YES`,
something changed upstream.

### A known-good result

```
manifests: 1
statements 546   user prompts captured: 9 of 9
agents     ['Explore', 'claude-code']
coverage   {'Activity': 13, 'Agent': 2, 'ContentDenied': 1, 'ContentUnknown': 3,
            'FileRead': 5, 'FileWritten': 3, 'Model': 1, 'ModelCall': 23,
            'PayloadTooLarge': 31, 'Tool': 5}
secret leaked: False
```

with documents

```
report.md  v1 stated(14)  v2 stated(14)  v3 withheld(0)
summary.md v1 stated(6)   v2 withheld(0)
notes.md   v1 stated(17)
.env       v1 withheld(0)      big.txt v1 withheld(0)
```

---

## Running it against Codex

**Not yet done.** Everything in this section is derived from Relay's source and from a committed
Codex capture, not from a live Codex run. Treat it as a plan to be corrected by the first attempt.

Codex differs from Claude Code in ways that change what this script tests:

**The export trigger is different, and it is the whole risk.** Codex's plugin hook schema has no
`SessionEnd`, so Relay never emits a root agent scope end and `SessionEnded` never fires. The manifest
is written from `Drop` — when the Relay process exits. A Codex session that is killed rather than
exited cleanly may write nothing at all. Since checkpointing landed, a partial manifest should exist
regardless; **confirming that is the single most valuable thing a first Codex run can establish.**

**File edits arrive as `apply_patch`, not `Write`/`Edit`.** `Add File` carries the full content and is
recorded exactly. `Update File` carries hunks and no pre-image, so it is recorded identity-only —
reconstructing from hunks would content-address a state the file may never have had. `Delete File`
currently records as a write with unknown content, because `FileMode` has no `Deleted` variant.

**`ContentRecovered` may become reachable here**, unlike on Claude Code, if Codex omits post-images.
That is worth watching: it is the one host where the replay chain earns its keep.

### What a human has to do

```bash
# 1. Install Codex, and check whether it has a non-interactive mode.
codex --help | grep -iE "exec|--print|-p\b"

# 2. Same staging as above, then:
rm -rf /tmp/relay-codex && mkdir -p /tmp/relay-codex && cd /tmp/relay-codex
printf 'alpha\nbeta\n' > notes.md
printf 'SECRET_KEY=do-not-record-me\n' > .env
nemo-relay run -- codex
```

Then, by hand:

| # | Ask Codex to | Exercises |
|---|---|---|
| 1 | create `report.md` with three lines | `apply_patch` `Add File` — exact content |
| 2 | read `report.md` back | `FileRead` |
| 3 | change one line in `report.md` | `apply_patch` `Update File` — identity-only, and possibly `ContentRecovered` |
| 4 | read `.env` | `ContentDenied` |
| 5 | delete `report.md` | currently recorded as a write with unknown content — see above |

**Exit Codex cleanly**, then run the same verification block against
`/tmp/relay-codex/.eqty/manifests/`. Before exiting, check that a manifest already exists — that is
the checkpointing claim, and on Codex it is the difference between a recording and nothing.

Expect `Compaction` and the `Agent`/subagent rows to differ: Codex's subagent model is not Claude
Code's, and Relay aligns it through a different path.

---

## What this deliberately cannot reach

- **File effects from `Bash`** — unobservable. See the upstream proposal.
- **Deletion tombstone** (`deleted:{path}`) — documented in `recorder.rs` but no `FileMode::Deleted`
  exists, so nothing constructs it.
- **`PermissionDecision` as a graph node** — classified, never recorded.
- **`/compact`** under Path B — no `-p` equivalent.
