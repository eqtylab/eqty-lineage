"""CLI: run a task N times and report which artifacts survived.

    eqty-lineage-replicate --task "..." --template ./work -n 6 --model haiku
    eqty-lineage-replicate --report-only ./runs        # reuse runs already performed

Costs N times the tokens of a single run. That is the price of knowing whether an artifact is stable.
"""

import argparse
import logging
import sys
from pathlib import Path

from .driver import RunSpec, replicate, transcript_for
from .report import build_report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eqty-lineage-replicate", description=__doc__)
    p.add_argument("--task", help="the prompt to run N times")
    p.add_argument("--template", help="working tree copied fresh for each run")
    p.add_argument("-n", type=int, default=6, help="number of runs (default 6)")
    p.add_argument("--workspace", default="./replicate-runs")
    p.add_argument("--model", help="model to pass to the agent")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--only", help="restrict the report to paths with this suffix, e.g. .py")
    p.add_argument("--report-only", metavar="DIR", help="skip running; report on runs already in DIR")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.report_only:
        base = Path(args.report_only)
        run_dirs = {d.name: d for d in sorted(base.iterdir()) if d.is_dir()}
        transcripts = {n: t for n, d in run_dirs.items() if (t := transcript_for(d)) is not None}
        if len(transcripts) < 2:
            print(f"need at least 2 runs with transcripts, found {len(transcripts)}", file=sys.stderr)
            return 1
    else:
        if not (args.task and args.template):
            p.error("--task and --template are required unless --report-only is given")
        spec = RunSpec(
            task=args.task,
            template=Path(args.template),
            n=args.n,
            model=args.model,
            max_turns=args.max_turns,
        )
        print(f"running {spec.n} times (this costs {spec.n}x the tokens of one run)...")
        result = replicate(spec, Path(args.workspace), on_run=lambda n, *_: print(f"  {n} done"))
        print(result.summary())
        for f in result.failures:
            print(f"  ! {f}", file=sys.stderr)
        if not result.ok:
            return 1
        run_dirs, transcripts = result.run_dirs, result.transcripts

    report = build_report(transcripts, run_dirs, only=args.only)
    print()
    print(report.render(color=not args.no_color))
    return 0 if not report.divergent else 2


if __name__ == "__main__":
    raise SystemExit(main())
