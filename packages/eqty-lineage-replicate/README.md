# eqty-lineage-replicate

Run an agent task N times and report which artifacts survived its nondeterminism.

```bash
eqty-lineage-replicate --task "…" --template ./work -n 6 --model haiku --only .py
eqty-lineage-replicate --report-only ./runs --only .py      # reuse runs already performed
```

```
6 runs   1/2 paths robust (50%)

DIVERGENT  util.py   5 distinct contents
    2 run(s): run-3, run-5
    1 run(s): run-1
    …
robust     test_util.py
```

**This costs N times the tokens of a single run.** That is the price of knowing whether an artifact is
stable, and it belongs in the decision to run it.

## Why

A signature proves you got *one* outcome. It does not tell you which, or whether another run would have
produced something else. Replication is the only way to find out.

Measured on a task specified down to the algorithm, with the test's literal assertion dictated: six runs
produced **five distinct implementations** using `islower()`, `isalpha()`, `isalnum()` and an ASCII
regex. They disagree on real input —

```
slugify("Café Ω 2 ½")  ->  'café-ω-2-'   (3 runs)   'café-ω-2-½'  (2 runs)   'caf--2-'  (1 run)
```

— and **all six pass the dictated test**. You cannot prompt your way to reproducible artifacts, and a
green test does not discriminate.

## How the comparison works

Entities are content-addressed, so identical bytes get identical CIDs and runs join automatically — no
correlation ids, no shared database. Each run's lineage is unioned into one graph and every fact is
annotated with the set of runs it appears in. That annotation is the **determination semiring**
(`eqty_lineage.query.semiring.determination`): `plus` unions, `times` intersects. A path is *robust*
when all runs ended at one content — full support, `qdepth` 0.

Two details that are easy to get wrong and silent when you do:

- **Paths must be normalised per run.** Runs execute in different directories, so without stripping each
  run's root nothing ever groups across runs and every path reports as robust.
- **The union contains phantom paths.** Splicing edges from different runs can connect nodes by a route
  no single run took. Their support intersects to zero, and zero is absence — `influence_support` drops
  them rather than reporting influence that never happened.

## Runs are sequential

On purpose. Running them concurrently would share CPU, disk and rate limits in ways that could correlate
outcomes — and the whole measurement is whether outcomes are independent.
