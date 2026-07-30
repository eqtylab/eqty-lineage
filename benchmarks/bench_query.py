"""Reproducible measurements for the two performance claims this package makes.

Both numbers were previously asserted in prose with nothing behind them: the accelerator's speedup over
the Python reference, and the cost of the minimal-witness computation that ``WITNESS_CAP`` exists to
bound. A reviewer could not check either.

    python benchmarks/bench_query.py                 # both benchmarks, synthetic graphs
    python benchmarks/bench_query.py --json out.json # machine-readable
    python benchmarks/bench_query.py --transcript ~/.claude/projects/<p>/<s>.jsonl

The graphs are synthetic and seeded, so the numbers are reproducible on any machine without needing a
private transcript. Shape matters more than provenance here: what makes the closure expensive is fan-in
and fan-out over a DAG, and that is generated directly. ``--transcript`` runs the same benchmarks over a
real session for anyone who has one.

Absolute timings are machine-dependent. The *ratio* is the claim worth checking.
"""

import argparse
import json
import random
import statistics
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from eqty_lineage.core import Triple, prov  # noqa: E402
from eqty_lineage.query import RUST_AVAILABLE, blast_radius, measure_alternatives  # noqa: E402
from eqty_lineage.query.semiring import ABSORPTIVE, BOOLEAN  # noqa: E402


def synthetic_graph(activities: int, fan_in: int = 2, fan_out: int = 2, seed: int = 0) -> List[Triple]:
    """A lineage-shaped DAG: entities consumed and produced by activities.

    Deliberately not a random graph. Real lineage is layered -- an activity consumes a few entities
    that already exist and produces a few new ones -- and the closure's cost comes from the resulting
    fan-in/fan-out, not from arbitrary connectivity. Entities are numbered rather than hashed so the
    generator stays dependency-free and the graph is identical on every machine.
    """
    rng = random.Random(seed)
    triples: List[Triple] = []
    entities = [f"e{i}" for i in range(max(fan_in, 4))]

    for a in range(activities):
        activity = f"a{a}"
        for src in rng.sample(entities, min(fan_in, len(entities))):
            triples.append(Triple(subject=activity, predicate=prov.USED, object=src))
        for o in range(fan_out):
            entity = f"e{len(entities) + o}"
            triples.append(Triple(subject=entity, predicate=prov.WAS_GENERATED_BY, object=activity))
        entities.extend(f"e{len(entities) + o}" for o in range(fan_out))

    return triples


def _time(fn, repeat: int = 3) -> float:
    """Best-of-N wall clock, in seconds. Best rather than mean: we are measuring the work, and the
    slower samples are measuring whatever else the machine was doing."""
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return min(samples)


def bench_backends(sizes: List[int], seed: int = 0) -> List[Dict]:
    """Python reference vs native accelerator on the influence closure."""
    rows = []
    for size in sizes:
        triples = synthetic_graph(size, seed=seed)
        source = "e0"

        py = _time(lambda: blast_radius(triples, source, semiring=BOOLEAN, backend="python"))
        row = {"activities": size, "triples": len(triples), "python_s": py}

        if RUST_AVAILABLE:
            rs = _time(lambda: blast_radius(triples, source, semiring=BOOLEAN, backend="rust"))
            row["rust_s"] = rs
            row["speedup"] = py / rs if rs else None

            # Agreement is asserted in tests/test_backends.py; re-checked here because a benchmark
            # that measures two backends computing different answers is measuring nothing.
            same = set(blast_radius(triples, source, backend="python").facts) == set(
                blast_radius(triples, source, backend="rust").facts
            )
            row["agree"] = same
        rows.append(row)
    return rows


def bench_witness_cap(size: int, caps: List[Optional[int]], seed: int = 0) -> List[Dict]:
    """Cost and yield of the minimal-witness computation at each cap.

    ``WITNESS_CAP`` bounds an output that is genuinely exponential -- computing minimal elements of a
    set family provably needs exponential space even as a ZDD -- so this measures what the bound buys
    and what it costs in fidelity.
    """
    from eqty_lineage.query import semiring as semiring_module

    triples = synthetic_graph(size, fan_in=3, fan_out=2, seed=seed)
    original = semiring_module.WITNESS_CAP
    rows = []
    try:
        for cap in caps:
            semiring_module.WITNESS_CAP = cap if cap is not None else 0
            elapsed = _time(lambda: blast_radius(triples, "e0", semiring=ABSORPTIVE, backend="python"), repeat=3)
            result = blast_radius(triples, "e0", semiring=ABSORPTIVE, backend="python")
            witnesses = [len(v) for v in result.facts.values()]
            rows.append({
                "cap": cap,
                "seconds": elapsed,
                "reached": len(result),
                "max_witnesses": max(witnesses) if witnesses else 0,
                "mean_witnesses": statistics.mean(witnesses) if witnesses else 0,
            })
    finally:
        semiring_module.WITNESS_CAP = original
    return rows


def bench_alternatives(triples) -> Dict:
    """The multi-derivation classification the cap is claimed not to affect."""
    report = measure_alternatives(triples, backend="python")
    return {"summary": report.summary()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[50, 150, 300])
    p.add_argument("--cap-size", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transcript", help="benchmark a real session instead of synthetic graphs")
    p.add_argument("--json", help="write results here")
    args = p.parse_args(argv)

    out: Dict = {"rust_available": RUST_AVAILABLE, "seed": args.seed}

    if args.transcript:
        from eqty_lineage.core import TripleSink

        triples = list(TripleSink.load(args.transcript))
        out["transcript_triples"] = len(triples)
        print(f"loaded {len(triples)} triples from {args.transcript}")

    print(f"accelerator {'available' if RUST_AVAILABLE else 'NOT BUILT (python-only numbers)'}\n")

    print("influence closure, boolean semiring")
    print(f"  {'activities':>10} {'triples':>8} {'python':>10} {'rust':>10} {'speedup':>8}  agree")
    backends = bench_backends(args.sizes, args.seed)
    for row in backends:
        rust = f"{row['rust_s'] * 1000:.2f}ms" if "rust_s" in row else "-"
        speed = f"{row['speedup']:.1f}x" if row.get("speedup") else "-"
        agree = row.get("agree", "-")
        print(f"  {row['activities']:>10} {row['triples']:>8} {row['python_s'] * 1000:>8.2f}ms "
              f"{rust:>10} {speed:>8}  {agree}")
    out["backends"] = backends

    print("\nminimal witnesses, absorptive semiring "
          f"({args.cap_size} activities, fan-in 3)")
    print(f"  {'cap':>6} {'seconds':>10} {'reached':>8} {'max witnesses':>14}")
    caps = bench_witness_cap(args.cap_size, [4, 8, 16, 64, None], args.seed)
    for row in caps:
        cap = "none" if row["cap"] is None else row["cap"]
        print(f"  {cap:>6} {row['seconds']:>10.4f} {row['reached']:>8} {row['max_witnesses']:>14}")
    out["witness_cap"] = caps

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
