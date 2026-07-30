# Determination: how much of what an agent produced was determined?

The empirical case for the determination semiring existing at all. Re-runnable, so the numbers below
are a measurement rather than a claim.

## Run it

```bash
eqty-lineage-replicate \
  --task "Implement the slugify function in textutils.py so the tests in test_textutils.py pass. Do not modify the tests." \
  --template experiments/determination/template \
  -n 6 --workspace ./runs

# behavioural classes, on inputs the dictated tests do not cover
for d in runs/run-*; do python experiments/determination/probe.py "$d/textutils.py"; done
```

Costs six agent sessions. Runs are sequential on purpose: concurrent runs share CPU, disk and rate
limits in ways that could correlate the outcomes, and correlation is exactly what is being measured.

## The template

Three files. `textutils.py` has a working `titlecase` and a `slugify` stub whose docstring specifies the
behaviour in prose; `test_textutils.py` dictates four cases; `README.md` is untouched context. The task
points at the tests rather than spelling out the algorithm, so the specification is deliberately
underdetermined in exactly one region — what counts as "alphanumeric".

## Result, 2026-07-30 (n=6, 216s)

```
6 runs   2/3 paths robust (67%)

DIVERGENT  textutils.py   4 distinct contents
    2 run(s): run-2, run-5
    2 run(s): run-4, run-6
    1 run(s): run-1
    1 run(s): run-3
robust     README.md
robust     test_textutils.py
```

**6 runs → 4 distinct implementations → 2 behavioural classes → 6/6 pass the dictated tests.**

### Verified outside the lineage graph

A 100%-robust result is indistinguishable from the path-normalisation bug documented in `union_runs`, so
the grouping was checked with `sha256` over the run directories — a method that touches none of this
code. Four distinct hashes, identical grouping. `README.md` and `test_textutils.py` returning
1-distinct is the control: had normalisation failed, every path would have looked unique to its run and
everything would have reported robust.

### Byte divergence overstates behavioural divergence

All six pass all four dictated tests. Probed on 12 inputs the tests do *not* dictate, the four
implementations collapse to two behavioural classes differing on exactly one input:

| input | runs 1, 3, 4, 6 | runs 2, 5 |
| --- | --- | --- |
| `"Café Ω x"` | `"cafe-ω-x"` | `"cafe-x"` |

Both NFKD-normalise `é → e`. They disagree on whether non-Latin alphanumerics survive, because the
docstring says "removes characters that are not alphanumeric or hyphens" and is silent on Unicode.
Every other probe — leading hyphens, empty string, pure punctuation, runs of hyphens, tabs, digits,
pre-existing hyphens — agrees across all six.

The divergence is not noise. It sits precisely where the specification was underdetermined, and the
dictated test suite cannot see it.

## What this supports

A signature over `textutils.py` from run-2 proves you got *that* artifact. It does not prove it was the
only possible outcome, and nothing in a classical provenance graph distinguishes the two. Determination
support — the set of resolutions under which a fact holds — is what makes "this was robust across every
run" checkable rather than assumed. Here 2 of 3 paths were robust and the one that mattered was not.

## Caveats

n=6, one task, one model, one template: an existence demonstration, not a rate. `only_final=True` means
intermediate edits are excluded, so what is compared is where the runs *ended*, not how they got there.
