# Live session script — exercise everything recordable today

## Setup (before starting Claude)

```bash
mkdir -p /tmp/relay-live3 && cd /tmp/relay-live3
printf 'alpha\nbeta\ngamma\n' > notes.md
printf 'SECRET_KEY=do-not-record-me\n' > .env
python3 -c "open('big.txt','w').write('x'*1500000)"      # > max_content_bytes (1 MiB)
printf '\x89PNG\r\n\x1a\n\xff\xfe\x00\x01' > tiny.bin     # deliberately not valid UTF-8
nemo-relay run -- claude
```

Claude picks tools on its own, so each turn names the tool explicitly. Send them one at a time and
let each finish.

## Turns

| # | Prompt | Exercises |
|---|---|---|
| 1 | `Use the Write tool to create report.md containing exactly three lines: one, two, three` | `Document` node, exact content CID, Write activity |
| 2 | `Use the Read tool to read report.md` | read-by-content-match; must **not** create a second version or a 2-cycle |
| 3 | `Use the Edit tool to change "two" to "TWO" in report.md` | version chain read→write; **ContentRecovered** (`_last_content` replay) |
| 4 | `Use the Read tool to read only lines 1 to 2 of report.md` | partial read → no content → **ContentUnknown**, identity-only node |
| 5 | `Use the Read tool on notes.md, then the Write tool to create summary.md holding its first line` | two files, one turn; read of A → write of B |
| 6 | `Use the Task tool to launch a subagent that reads summary.md and reports how many characters it has` | **SubagentStarted/Ended**, second `Agent` node, `performedBy` attribution, likely a second model |
| 7 | `Use the Read tool on .env` | **redaction**: node keeps path + true content CID, bytes withheld, `redacted: true` |
| 8 | `Use the Read tool on big.txt` | **PayloadTooLarge**: recorded by CID, descriptor only |
| 9 | `Use the Read tool on tiny.bin` | non-UTF-8 content identity — the case the Python path gets wrong (§6.1a) |
| 10 | `Run: ls /nonexistent/path` | failing tool → terminal status / `is_error` |
| 11 | `/compact` | **Compacted** → Dataset node, `Compaction` counter |
| 12 | `/exit` | `SessionEnded`, manifest written |

Turns 1, 3, 5 and 7 will likely raise permission prompts — that is wanted, it drives the
`PermissionDecision` classification path (still not a graph node).

## Verify afterwards

```bash
python3 - <<'PY'
import json, base64, collections, glob
p = sorted(glob.glob('/tmp/relay-live3/.eqty/manifests/*.json'))[-1]
m = json.load(open(p)); B = m['blobs']
def blob(c):
    raw = B.get(c.replace('urn:cid:','')) or B.get(c)
    try: return json.loads(base64.b64decode(raw)) if raw else None
    except Exception: return None
assets, cov, redacted = collections.Counter(), None, []
for c in B:
    d = blob(c)
    if not isinstance(d, dict): continue
    if 'assetType' in d: assets[d['assetType']] += 1
    if d.get('name') == 'coverage': cov = d['coverage']
    if d.get('redacted'): redacted.append(d.get('name'))
print('statements', len(m['statements']))
print('assets    ', dict(assets))
print('coverage  ', cov)
print('withheld  ', redacted)
for want in ['ContentRecovered','ContentUnknown','Compaction','PayloadTooLarge']:
    print(f'  {want:18}', 'YES' if cov and want in cov else 'MISSING')
print('  Document nodes    ', 'YES' if assets.get('Document') else 'MISSING')
print('  two agents        ', 'YES' if assets.get('Agent',0) >= 2 else 'MISSING')
PY
```

Every line should read `YES`. Anything `MISSING` is a path that has still never run live.

## What this deliberately cannot reach

- **File effects from Bash** — unobservable, see the upstream proposal.
- **Deletion tombstone** (`deleted:{path}`) — documented in `recorder.rs` but **no `FileMode::Deleted`
  exists**, so nothing constructs it. Claude Code has no delete tool either; deletions go through
  Bash and are invisible twice over.
- **`PermissionDecision` as a graph node** — classified, never recorded.
- **Codex paths** — `apply_patch`, and the `Drop`-only export. Needs a Codex session.
