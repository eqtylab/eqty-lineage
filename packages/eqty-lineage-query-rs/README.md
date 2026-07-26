# eqty-lineage-query-rs

Native accelerator for `eqty-lineage-query`. Install it and the Python API gets faster; nothing else
changes. Uninstall it and everything still works.

```bash
pip install eqty-lineage-query-rs     # optional
```

```python
from eqty_lineage.query import blast_radius, measure_alternatives
blast_radius(triples, source)          # uses the accelerator when present
blast_radius(triples, source, backend="python")   # force the reference implementation
```

## Why this and nothing else

Profiling the whole pipeline showed where the time actually is:

| stage | time/session | self-time in Python | in the SDK |
| --- | --- | --- | --- |
| transcript parse | 0.005 s | 82% | 18% |
| **record** | **6.3 s** | **~0%** | **100%** |
| query — blast radius | 4.0 s | 81% | 19% |
| query — measure | 20.1 s | 81% | 19% |
| projection | 0.006 s | 73% | 27% |

Recording is 100% inside `eqty_sdk` — `add_metadata_statement`, `add_data_statement`,
`add_computation_statement`, about **4 ms per statement of cryptographic signing already in Rust**.
Porting the recorder would optimize the ~0% that is ours.

The query engine is the opposite: 81% of its self-time is `_unify` and `_match`, and it is the slow
stage. It is also the **only** component that never touches `eqty_sdk`, so it is the only one portable
without EQTY's private crates. Portability and payoff coincide exactly.

## Measured

Same algorithm, same data, best of three:

| session | triples | semiring | Python | Rust | speedup |
| --- | --- | --- | --- | --- | --- |
| 4ca013c3 | 1,176 | boolean | 0.18 s | 0.003 s | 60× |
| 6163498a | 1,398 | counting | 0.48 s | 0.007 s | 69× |
| d5d95111 | 6,102 | absorptive | 13.26 s | 0.159 s | 83× |
| 28daf0cc | 12,849 | boolean | 55.09 s | 0.771 s | 71× |
| 28daf0cc | 12,849 | counting | 106.73 s | 1.021 s | 105× |
| 28daf0cc | 12,849 | absorptive | 158.98 s | 4.798 s | 33× |
| **total** | | | **356 s** | **7.0 s** | **51×** |

Absorptive on the largest graph is the weakest at 33× — monomial minimalization is allocation-heavy in
both languages, and it is where the remaining headroom is (bitsets rather than `Vec<u32>`).

## Correctness

The Python implementation is the reference oracle, and this is a straight port of the same algorithm
rather than a different engine — precisely so it can be checked against it. Equivalence is asserted on
**values, not row counts**: two engines can agree on which tuples are derivable and still disagree on
their annotations, which is the failure a port is prone to.

Verified identical across three semirings and three real sessions — derived counts, annotation sums and
maxima all match to the digit.

Two deliberate divergences, both documented in `lib.rs`:

- `counting` saturates at `u64::MAX` where Python uses arbitrary-precision integers. No real session has
  come close (largest observed derivation count: 359,055,106), but a pathological graph would differ.
- The accelerator implements the **influence closure** specifically, not the general rule engine. The
  Python engine keeps its general `Rule`/`Atom` machinery for custom rules; only the closure's cost
  justifies crossing the boundary.

## API

The boundary is whole queries, not individual rules — the fixpoint is essentially all of the cost, so
one crossing per query makes marshalling irrelevant. Exposing `evaluate(rules, edb)` would instead
marshal set-of-set annotations on every iteration.

| function | returns |
| --- | --- |
| `closure_bool(triples)` | reachable `(subject, object)` pairs |
| `closure_count(triples)` | pairs with derivation counts |
| `closure_absorptive_sizes(triples)` | pairs with the *number* of minimal witnesses |
| `witnesses(triples, source, target)` | the witness sets for one pair |
| `taint(triples, untrusted)` | pairs reachable from an untrusted source (integrity semiring) |
| `measure(triples)` | the alternatives report, computed without marshalling the closure |

Sizes rather than witness sets by default: a large session has tens of thousands of pairs, and callers
who want the witnesses want them for one pair.

## Build

```bash
maturin build --release          # abi3 wheel, one per platform for Python 3.11+
maturin develop --release        # into the active venv
```

`abi3-py311` means a single wheel covers every supported Python rather than one per minor version.
