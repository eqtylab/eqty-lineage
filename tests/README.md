# Tests

```bash
just test        # everything
just test-pure   # only what needs neither eqty-sdk nor a Rust toolchain
```

| Module | Needs | Covers |
| --- | --- | --- |
| `test_tool_results.py` | — | Deriving file versions from a tool result; shape dispatch, edit replay |
| `test_transcript.py` | — | Claude Code transcript parsing against a synthetic session |
| `test_codex.py` | — | Dialect detection and `apply_patch`, against a real captured session |
| `test_semiring.py` | — | Semiring laws, and independently derived counting ground truth |
| `test_engine.py` | — | Edge orientation, the fixpoint, taint |
| `test_determination.py` | — | Multi-run supports, robustness, divergence |
| `test_redaction.py` | — | What may be stored versus only identified |
| `test_policy.py` | — | Permission policy and counterfactual replay |
| `test_recorder.py` | `eqty-sdk` | The recorder against a live SDK |
| `test_daemon.py` | `eqty-sdk` | The live hook path over real HTTP |
| `test_backends.py` | accelerator | Python and Rust evaluators must agree |

## Fixtures are synthetic or sanitised

Real Claude Code transcripts under `~/.claude/projects/` carry the full contents of whatever repository
the session touched, along with absolute paths and prompts. None of that can be committed.

- `fixtures/claude_session.jsonl` is hand-built to exercise the parser's edge cases: an orphaned tool
  result from a resumed session, a compact boundary whose real predecessor is `logicalParentUuid`, a
  snapshot delta already explained by an Edit, a bare-string Bash result, a failed call, a subagent.
- `fixtures/codex_hooks.json` is a real capture from codex-cli 0.145.0 with paths and home directory
  rewritten. Kept as a recording rather than a hand-written approximation, because every Codex-specific
  finding here came from reading these payloads and not from reading documentation.

## What these tests can and cannot establish

**Backend agreement is not correctness.** Two implementations of one algorithm share its bugs.
Semi-naive double-counting was present in both the Python and the Rust evaluator, identically, and
`test_backends.py` passed throughout — the classic correlated-fault result for N-version comparison.
That is why `TestCountingGroundTruth` exists with a hand-counted expected value: on
`a→b, b→c, a→c, c→d` the pair `a→d` has exactly two derivations, and only an independently derived
number could catch both backends reporting three.

**Conformance exercises the adapters, not the recorder.** `eqty-lineage-hooks verify` compares the
offline and live paths, but they deliberately share `eqty-lineage-core` so they cannot drift; a fault in
shared code is invisible to it. `test_tool_results.py` and `test_recorder.py` test that shared code
directly for exactly this reason.

## Writing new tests

- **`eqty_sdk.init()` is process-global** and warns on a second call. Use the session-scoped `sdk`
  fixture; do not initialise per test.
- **Statement CIDs are not stable** — they cover a signed credential carrying `validFrom`. Compare
  graphs through `activity_signatures` in `eqty_lineage.core.canonical`, never by bytes.
- **Asset CIDs are derived from content alone.** The path lives in metadata and in the `eqty:hasPath`
  triple, so identical bytes at two paths are one node with two paths.
- Prefer pinning a *reason*: several tests here exist because a plausible-looking graph was silently
  wrong, and the comment explaining what went wrong is the point of the test.
