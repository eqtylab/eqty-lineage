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

**Wait for the export, not for the process.** The manifest is written from `Drop`, during shutdown,
so the process stops matching `pgrep` *before* the file is final. Reading in that window returns a
checkpoint — no coverage node, and whatever the last turn produced still missing — which looks
exactly like a session that recorded almost nothing. Files on disk race the same way: a patch the
agent has already applied may not be visible yet.

That window produced two wrong readings and a plausible-sounding conclusion about Codex asserting
writes it never made. Both readings were premature; the run was fine. Wait for the process to be
*reaped*, then check that the manifest carries a coverage node before believing anything it says:

```bash
while pgrep -f "nemo-relay run" >/dev/null; do sleep 1; done
sleep 2   # Drop still has the export to finish
```

Two turns Path B cannot drive, because they are interactive slash commands with no `-p` equivalent:
`/compact` (exercises `Compaction`) and `/exit`. Run those by hand in Path A if you need them.

## Turns

**For Claude Code.** Codex has its own list further down, and they are not interchangeable — each
names tools the other host does not have.

Verbatim, in order. These are exactly the strings the driver in Path B sends, so both paths exercise
the same thing — send them one at a time and let each finish.

**1.**
```
Use the Write tool to create report.md containing exactly three lines: one, two, three
```
`Document` node, exact content CID, `FileWritten`.

**2.**
```
Use the Read tool to read report.md
```
Read-by-content-match. Must **not** create a second version, and must not produce a 2-cycle.

**3.**
```
Use the Edit tool to change "two" to "TWO" in report.md
```
Version chain, read → write.

**4.**
```
Use the Read tool to read only lines 1 to 2 of report.md
```
Fragment → no content → `ContentUnknown`.

**5.**
```
Use the Read tool on notes.md, then the Write tool to create summary.md holding its first line
```
Two files in one turn: read of A → write of B.

**6.**
```
Use the Task tool to launch a subagent that reads summary.md and reports how many characters it has
```
Subagent node named by `agent_type`, and `performedByInstance` on its activities.

**7.**
```
Use the Read tool on .env
```
`ContentDenied` — the node keeps its path and true content CID, the bytes are withheld.

**8.**
```
Use the Read tool on big.txt
```
Read returns roughly 21 KB of the 200 KB file with `truncatedByTokenCap: true`, so the node must
record **no content**. A fragment's hash is not the file's hash.

**9.**
```
Use the Read tool on tiny.bin
```
Read refuses binaries outright; the node records the refusal.

Two more, interactive only — Path B cannot drive them because they are slash commands with no `-p`
equivalent:

**10.** `/compact` — exercises `Compaction`.
**11.** `/exit` — ends the session.

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

**Check for a coverage node before anything else.** A manifest without one is a checkpoint, not a
finished recording — see the export race above. Every other number in it is a lower bound.

**Then check the prompt count and the manifest count.** If fewer turns were captured than you sent,
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

**Run, and it works** — with one finding that matters more than the rest.

Codex has two paths, and they exercise different things. **`codex exec` gives each invocation its own
session**, so a file written by one run and changed by another crosses a session boundary — which is
why `ContentRecovered` can never fire there. The interactive path keeps one session, and is the only
way to reach it.

### Codex, interactive

```bash
rm -rf /tmp/relay-codex && mkdir -p /tmp/relay-codex && cd /tmp/relay-codex
printf 'alpha\nbeta\ngamma\n' > notes.md
printf 'SECRET_KEY=do-not-record-me\n' > .env
nemo-relay run -- codex
```

**Use these turns, not the ones above.** The Claude Code turns name `Write`, `Read`, `Edit` and
`Task`; Codex has none of them. Sending "Use the Write tool to create report.md" makes Codex spend a
turn discovering the tool does not exist before falling back to `apply_patch` — the file still gets
written, but the run carries two extra shell probes that are an artefact of the wrong prompt rather
than anything about the recorder.

Then send these verbatim, one at a time. Codex has **no read tool**, so it will reach for `sed`,
`awk` or `rg` on anything that reads — that is the point of turns 3 and 4, not a mistake in them.

**1.**
```
Create a file report.md containing exactly three lines: one, two, three
```
`apply_patch` `*** Add File` — content recorded exactly, `FileWritten`.

**2.**
```
Change line 2 of report.md from 'two' to 'TWO'
```
`apply_patch` `*** Update File` — hunks with no post-image. Recorded identity-only, `ContentUnknown`.
**This is the turn that can reach `ContentRecovered`**, because turn 1 established the content for
that path in the same session. If it fires, the replay chain has finally earned its keep.

**3.**
```
Read notes.md and tell me its first line
```
Expect no file node at all. Codex reads through the shell, so this counts
`ToolCallWithoutFileObservation` and nothing else.

**4.**
```
Read .env and tell me only how many lines it has, not its contents
```
Expect `ContentDenied` **not** to fire, for the same reason. The secret stays out of the manifest,
but because the read was invisible rather than because the gate withheld it.

**5.**
```
Delete report.md
```
Currently recorded as a write with unknown content — `FileMode` has no `Deleted` variant, so the
tombstone identity documented in `recorder.rs` is unreachable.

**6.** `/compact` if Codex offers it, then exit cleanly.

**Before exiting, check a manifest already exists.** Codex has no `SessionEnd`, so the final export
runs from `Drop`. A checkpoint appearing mid-session is the evidence that a killed Codex session
would still leave a recording.

### Codex, one-shot

What was actually run. Each command is its own session and its own manifest.

```bash
nemo-relay run -- codex exec "Create a file report.md containing exactly three lines: one, two, three" --sandbox workspace-write --skip-git-repo-check

nemo-relay run -- codex exec "Change line 2 of report.md from 'two' to 'TWO'. Then read .env and tell me only how many lines it has." --sandbox workspace-write --skip-git-repo-check
```

`--sandbox workspace-write` is what lets it write at all; `--skip-git-repo-check` is only needed
because `/tmp` is not a repository. Relay requires codex-cli >= 0.143.0.

### What the two runs established

**Export from `Drop` alone works.** Relay's Codex descriptor has ten hook events and no `SessionEnd`,
so `SessionEnded` never fires and the manifest is written when the Relay process exits. This was the
largest unknown about Codex and it is settled: both runs produced a manifest.

**Both `apply_patch` branches are proven.**

| patch | recorded |
| --- | --- |
| `*** Add File` | exactly — `report.md` v1, 14 bytes, `reconstructed: "stated"` |
| `*** Update File` | identity-only — v1, no content, `ContentUnknown` |

`Update File` carries hunks and no pre-image, so the node keeps the path and records no content.
Reconstructing from the hunks would content-address a state the file may never have had.

**Two sessions, two manifests, neither clobbered.**

### The finding: Codex has no read tool

Every read went through the shell:

```
Bash input: sed -n '1p' notes.md && sed -n '1,4p' report.md
Bash input: awk 'END { print NR }' .env
Bash input: pwd && rg --files -g 'notes.md' -g 'report.md' -g '.env'
```

So `FileRead` is **0**, `notes.md` never became a node despite being read, and — the part worth
sitting with — **`ContentDenied` never fired**. `.env` *was* read, by `awk`, and the redaction gate
never saw it, because we never saw the file.

The secret did not reach the manifest, but not because the gate withheld it: because the read was
invisible. On Claude Code `.env` produces a node carrying its path and true content CID with the
bytes withheld, which is a claim a reader can act on. On Codex it produces nothing at all, which is
indistinguishable from the file never having been touched.

Claude Code's version of this gap is "the agent sometimes chooses `Bash`". Codex's is "there is no
other option" — it has no read tool. That is the strongest case for the upstream proposal.

### What still has not fired on Codex

`ContentRecovered` did not fire in either one-shot run, and cannot: the replay chain needs prior
known content for that path in the *same* session, and `codex exec` gives each run its own. Turn 2 of
the interactive path above is the remaining way to reach it, and has not been tried.

`Compaction`, `PermissionDecision` and subagents are untested on Codex.

## What this deliberately cannot reach

- **File effects from `Bash`** — unobservable. See the upstream proposal.
- **Deletion tombstone** (`deleted:{path}`) — documented in `recorder.rs` but no `FileMode::Deleted`
  exists, so nothing constructs it.
- **`PermissionDecision` as a graph node** — classified, never recorded.
- **`/compact`** under Path B — no `-p` equivalent.
