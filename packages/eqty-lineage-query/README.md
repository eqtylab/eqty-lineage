# eqty-lineage-query

Semiring-parameterized Datalog over EQTY lineage graphs. One evaluator, instantiated at different
semirings, answers structurally different questions over the same fact set.

```python
from eqty_lineage.query import blast_radius, taint, policy_violations, ABSORPTIVE

downstream = blast_radius(triples, source_cid)                     # what does this affect?
tainted    = taint(triples, untrusted=[env_file_cid])              # did anything untrusted reach it?
bad        = policy_violations(triples, denied_paths=["/etc/*"])   # what wrote outside the set?
witnesses  = reaches(triples, source, target, ABSORPTIVE)          # by which minimal sets of facts?
```

Dependency-free. CozoDB was the obvious alternative and was rejected: its Python bindings have had no
release since December 2023 and are flagged inactive, which is not something to put in a package
published under contract. Writing the evaluator is also what makes semiring parameterization possible
at all — an off-the-shelf engine answers *whether* something is derivable, and the point here is *how*.

## Optional native accelerator

`eqty-lineage-query-rs` is a Rust port of this evaluator. Install it and every query gets faster;
nothing else changes. It is optional by design — this package is the reference implementation and the
oracle the port is verified against.

```python
blast_radius(triples, source)                      # uses it when installed
blast_radius(triples, source, backend="python")    # force the reference path
measure_alternatives(triples, backend="rust")      # require it, or raise
```

Measured on a 6,102-triple session, decomposed honestly:

| | time | gain |
| --- | --- | --- |
| Python, generic rule engine | 8.109 s | — |
| Rust, **same algorithm** | 0.113 s | **72× — the language** |
| Rust, closure indexed by source | 0.0053 s | 21× — **algorithmic, also available in Python** |
| combined | | 1530× |

Only the 72× is attributable to Rust. The remaining 21× comes from indexing the join by source instead
of scanning, which the pure-Python engine could adopt too.

Both backends are verified to agree on real sessions — reachable pairs, derivation counts, minimal
witness sets, taint results, and the full alternatives report.

## Semirings

| Semiring | `plus` | `times` | Answers |
| --- | --- | --- | --- |
| `BOOLEAN` | ∨ | ∧ | is it reachable |
| `COUNTING` | + | × | how many distinct derivations |
| `ABSORPTIVE` | ∪, minimalized | ∪, minimalized | minimal witness sets — **the default for how-provenance** |
| `WHY` | ∪ | ∪ | every witness set, subsumed ones included |
| `integrity(…)` | max | max | worst taint the result must be assumed to carry |
| `confidentiality(…)` | min | max | least restricted level it may be read at |
| `TROPICAL` | min | + | cheapest derivation |

**Recursion needs absorption.** For recursive Datalog the provenance semiring is the semiring of formal
power series, finite only for commutative absorptive ω-continuous semirings. `COUNTING` is neither
idempotent nor absorptive and genuinely diverges on a cyclic graph; the engine raises `NonTerminating`
rather than spinning.

**`WHY` is Why(X), not ℕ[X].** Monomials are sets, so `times` is idempotent and a second lap round a
cycle contributes nothing new — Why(X) converges on any finite graph. True ℕ[X] tracks multiplicity
with multisets and is what diverges. It is not provided: `COUNTING` is ℕ[X] under the homomorphism
sending every variable to 1, which is the projection that is actually useful here.

**Confidentiality and integrity are duals, and confusing them silently gives the wrong answer.** For
confidentiality `plus` is `min` — if a result is derivable from public facts alone it may be read as
public. For integrity `plus` is `max` — one tainted derivation taints the result and no alternative can
launder it. Ask "did anything untrusted reach this artifact" with the confidentiality semiring and it
answers "no" as soon as a single clean path exists.

## Edge orientation

`edge(X, Y)` is the **information-flow** relation: X influenced Y. PROV predicates are written pointing
*backwards* in time (an activity `used` its inputs), so `FLOW_ORIENTATION` reverses them when building
the flow relation. Two consequences worth knowing:

- Getting this wrong is silent. A closure over mixed orientations still returns answers; they are just
  not answers to "what influenced what".
- `prov:wasInvalidatedBy` is dropped: it is the exact inverse of `prov:wasDerivedFrom`, emitted for PROV
  completeness. Keeping both puts a 2-cycle in the flow relation for every twice-edited file — which is
  what made non-absorptive semirings diverge on real sessions before the orientation map existed.

## Positive Datalog only

Semiring provenance under negation is not settled theory, so rules that would want `not` take the
complement as an explicit input relation instead — the caller materializes "denied paths" rather than
the engine deriving "not allowed".

## Findings: does `+` do real work on agent lineage?

Measured over local sessions, **after correcting a counting bug described below**:

| session | reachable pairs | >1 derivation | max derivations | max minimal witnesses |
| --- | --- | --- | --- | --- |
| 4ca013c3 | 3,081 | 1,619 (52.5%) | 40 | 40 |
| 6163498a | 4,485 | 2,543 (56.7%) | 156 | 156 |
| d5d95111 | 21,182 | 12,378 (58.4%) | 1,452 | 1,452 |
| **overall** | **28,748** | **16,540 (57.5%)** | | |

**`+` is not vacuous.** Agent execution graphs are emphatically not trees — more than half of all
reachable pairs are connected by more than one derivation.

### Correction: semi-naive double-counts for non-idempotent semirings

Earlier revisions of this file reported figures up to 359,055,106 derivations. **Those were inflated by
a real bug**, since fixed.

Semi-naive propagates a tuple's *merged* annotation when it changes, not the increment. For an
idempotent `plus` that is harmless — the repeat is absorbed — but `COUNTING` is precisely the
non-idempotent semiring, so re-derivations were counted more than once. Ground truth on
`a->b, b->c, a->c, c->d`: the pair `a->d` has exactly two derivations, `{a->c, c->d}` and
`{a->b, b->c, c->d}`. Both the Python engine and the Rust accelerator reported **three**.

They agreed on the wrong answer because they share the algorithm — a **correlated fault**, the classic
N-version blind spot, and one the backend-agreement test could not have caught by construction.

Fixing it inside semi-naive would require the *difference* between the old and new annotations —
subtraction, which is exactly what a semiring lacks. So non-idempotent semirings are now evaluated over
a **provenance circuit** (`circuit.py`): record the derivations during a boolean fixpoint, then evaluate
the recorded structure. `evaluate()` warns if asked to run a non-idempotent semiring directly.

The `>1 derivation` classification was unaffected — a pair with two derivations still has two — so the
qualitative finding survives; only the magnitudes were wrong.

### Absorptive does not scale, and the reason is structural

Circuits gave a ~2.2× speedup on mid-sized graphs and fixed the correctness bug, but they **do not**
contain the witness blowup.

On one session's *projection* — only 1,358 triples and 2,465 reachable pairs, 5% of the full graph — a
single pair has **118,096 minimal witnesses**, and the computation takes 93 s. A comparable projection
from another session (1,476 triples, 1,897 pairs) finishes in 0.009 s with a maximum of 484.

So the blowup is not driven by graph size but by graph *shape*: it is combinatorial in the number of
parallel routes between two nodes. Cutting the graph down does not help, because the projection can
retain exactly the structure that causes it.

Two consequences worth stating plainly:

- The claim that minimal-witness answers "stay readable" holds for some sessions and fails badly for
  others. It is a property of the session, not of the method.
- Whole-closure absorptive evaluation is not a safe default at any size. The workable shape is
  per-query — expand witnesses for *one* pair, with a cap — leaving the circuit itself, which stays
  compact, as the whole-graph representation.

### Can this be optimised away? No — but it can be bounded

The blowup is in the *output*, so no faster code addresses it. That is not a guess:
["Single Family Algebra Operation on BDDs and ZDDs Leads To Exponential Blow-Up"](https://arxiv.org/pdf/2403.05074)
shows that implementing **minimal/maximal** — precisely this operation — can require exponential size
even as a ZDD, the canonical compact representation for families of sets, and independently of variable
ordering or dynamic reordering. Switching representation does not help. Neither does Rust.

What works is a bound. `WITNESS_CAP` (default **64**, matched in both backends) retains at most *k*
minimal witnesses, shortest first, so the most general explanations survive. On the pathological case:

| cap | time | max witnesses | multi-derivation pairs |
| --- | --- | --- | --- |
| 8 | 0.005 s | 8 | 1,829 / 2,465 |
| 32 | 0.014 s | 32 | 1,829 / 2,465 |
| 128 | 0.066 s | 128 | 1,829 / 2,465 |
| exact | 104.6 s | 118,096 | 1,829 / 2,465 |

**The multi-derivation classification is identical at every cap.** Truncation loses the enumeration of
witnesses, never the answer to whether alternatives exist — which is the measurement this package makes.

### Shipping the circuit: measured

A circuit is compact enough to travel *inside* a manifest, which means the producer need not anticipate
the question — the verifier picks a semiring and evaluates the circuit themselves.

| session | triples | circuit (gz) | manifest | overhead |
| --- | --- | --- | --- | --- |
| 66f2315c | 669 | 5.5 KB | 1,104 → 1,121 KB | **1.016×** |
| d5d95111 | 6,734 | 82.9 KB | 11,271 → 11,429 KB | **1.014×** |
| c8179cce | 23,272 | 406.4 KB | 20,041 → 20,775 KB | **1.037×** |

Circuit size is linear in the input — roughly three instantiations per triple — so the ratio stays flat
as sessions grow; compressed, it is 4–7% of the triple sidecar it annotates, because interning removes
the CID repetition that dominates the JSONL form. For context, Soufflé reports ~1.45× memory overhead
for proof-annotated provenance and up to 100× for naively storing each tuple's full subproof.

It also costs **8 statements, not thousands** — the circuit rides as a single content-addressed asset —
so the ~10,922-statement export ceiling is untouched.

**One SDK behaviour to know.** A manifest carries only assets that participate in a *computation*
statement. An asset that is merely constructed, or registered with `add_data_statement`, or given a
`Metadata` statement, does **not** appear in the export — measured: 0 statements in all three cases,
versus a populated manifest once the asset is a computation output. So the circuit is attached as the
output of a computation over the session's own file versions, which is also what it honestly is.

Capping is an approximation, not an optimisation: `plus` stops being associative once truncation bites,
and the result is *k genuine minimal witnesses* rather than the complete basis. Set `WITNESS_CAP = None`
(Python) or `set_witness_cap(0)` (Rust) for exact evaluation, and expect it to hang on some real
sessions.

## Findings: does agent nondeterminism diverge the lineage?

The question behind any "under which resolutions does this hold" work: if repeated runs of one task
converge on identical artifacts, there is nothing to explain. Tested directly — 12 real agent sessions,
two tasks at different specification tightness, 6 runs each, same starting repo.

**Tight task.** Specified down to the algorithm ("lowercase, replace each space with a single hyphen,
then remove every character that is not a lowercase letter, a digit, or a hyphen") with the test's
literal assertion dictated.

| | result |
| --- | --- |
| distinct `test_util.py` contents | **1 of 6** — converged, because its text was dictated |
| distinct `util.py` contents | **5 of 6** |
| distinct tool-call sequences | 3 of 6 |
| file versions per run | identical (4) |

The five implementations are not cosmetic variants. They use `c.islower()`, `c.isalpha()`,
`c.isalnum()`, and `re.sub(r"[^a-z0-9\-]", "", s)` — which disagree on non-ASCII input:

```
slugify("Café Ω 2 ½")  ->  'café-ω-2-'    (3 runs)
                       ->  'café-ω-2-½'   (2 runs — isalnum keeps ½)
                       ->  'caf--2-'      (1 run  — ASCII regex strips é and Ω)
```

**All six pass the dictated test.** Three behaviours, one test, no discrimination.

**Loose task** ("add a useful string helper of your choosing"): 6 of 6 distinct artifact sets, 6 of 6
distinct tool-call sequences.

Three conclusions.

**You cannot prompt your way to reproducible artifacts.** Tightening the specification reduced
*tool-sequence* variance (3 of 6 distinct, versus 6 of 6) but not *artifact* variance (5 of 6 either
way). The only thing that converged was the file whose literal content was dictated.

**Testing does not discriminate.** A signature proves you got one of these; a passing test proves
nothing more. Neither tells you which of three semantics you shipped.

**The resolution space is small and discrete.** The runs differ by a single choice — which
character-class predicate — not by an unbounded token distribution. That is what makes "under which
resolutions does this hold" a tractable question rather than a philosophical one, and it is the reason
to restrict resolutions to decision points rather than model outputs.

Content addressing already records the divergence: five distinct implementations are five distinct
entities with five distinct CIDs. What is missing is the algebra to reason over them.
