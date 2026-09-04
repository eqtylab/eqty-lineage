# Live session script — exercise everything recordable today

## Setup (before starting Claude)

**Lower the content ceiling for this session first.** The default is 100 MiB, and generating a file
that large to prove the ceiling works is a waste of disk and time. Set it small in the
`[plugins.dynamic.config]` block of your `plugins.toml` — usually
`~/.config/nemo-relay/plugins.toml` — so a few hundred kilobytes trips it:

```toml
[plugins.dynamic.config]
max_content_bytes = 65536      # 64 KiB, for this test only
```

That is worth doing for its own sake: it is the only step here that exercises the config override
end to end, and a value the host sets but the plugin ignores would otherwise look identical to a
value that worked.

```bash
mkdir -p /tmp/relay-live3 && cd /tmp/relay-live3
printf 'alpha\nbeta\ngamma\n' > notes.md
printf 'SECRET_KEY=do-not-record-me\n' > .env
python3 -c "open('big.txt','w').write('x'*200000)"        # 200 KB: over the 64 KiB test ceiling
printf '\x89PNG\r\n\x1a\n\xff\xfe\x00\x01' > tiny.bin     # deliberately not valid UTF-8
nemo-relay run -- claude
```

Put the ceiling back afterwards, or every later session records file bytes by CID only.

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
| 8 | `Use the Read tool on big.txt` | **ContentTooLarge**: over the ceiling set above, so recorded by CID and descriptor only — and proof the config override took effect |
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
required = [
    ('FileRead',        'a file was read into the graph at all'),
    ('FileWritten',     'a file version was written'),
    ('ContentRecovered','the _last_content replay chain ran'),
    ('ContentUnknown',  'a partial read became an identity-only node'),
    ('ContentDenied',   'deny_globs withheld .env'),
    ('ContentTooLarge', 'the configured ceiling was applied'),
    ('Compaction',      'compaction was recorded'),
]
for key, why in required:
    print(f'  {key:18} {"YES" if cov and key in cov else "MISSING":8} {why}')
print(f'  {"Document nodes":18} {"YES" if assets.get("Document") else "MISSING":8} file lineage produced nodes')
print(f'  {"two agents":18} {"YES" if assets.get("Agent",0) >= 2 else "MISSING":8} a subagent was recorded')
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
