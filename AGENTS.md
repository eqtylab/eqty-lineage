# Comments

Write comments for the next reader, not as a record of how the code got here.

- **Terse and focused.** One or two lines. If the explanation is longer than the code, cut the
  explanation.
- **Say why, never what.** Skip anything inferable from the code it sits on. `fail-fast: false` does
  not need a sentence saying both results are wanted.
- **No history.** Never "used to", "an earlier version", "originally", "was wrong". State the
  invariant instead: not "the dedup path skipped the base refresh, so A -> B -> A left B cached" but
  "a file that goes A -> B -> A must not leave B cached as the replay base".
- **No measurements or run ids.** Benchmark numbers, timings and CI run links are true the day they
  are written and unverifiable after. Keep the conclusion, drop the evidence.
- **No commented-out code.** Git has it.
- **Keep what stops someone breaking it** — a non-obvious constraint, an external system's behavior,
  a footgun. That is what earns the space.

A comment that needs the reader to know what happened during development is offloaded context, not
documentation. Delete it.
