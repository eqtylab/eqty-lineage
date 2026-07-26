"""CLI: ingest a Claude Code session transcript into an EQTY manifest.

    python -m eqty_lineage.transcript <session.jsonl> -o manifest.json
    python -m eqty_lineage.transcript --sweep 200        # parser coverage over local sessions
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from .claude_code import find_sessions
from .ingest import ingest, parse_only
from .invariants import check_invariants


def _sweep(limit: int, verbose: bool) -> int:
    """Parse many local sessions and report coverage, warnings, and failures.

    Deliberately parser-only: no SDK, no signer, no blobs. The point is to find transcripts whose shape
    the parser mishandles, and that answer should not cost a manifest per session.
    """
    sessions = find_sessions()[:limit]
    if not sessions:
        print("no sessions found under ~/.claude/projects", file=sys.stderr)
        return 1

    totals: Counter = Counter()
    warnings: Counter = Counter()
    failures = []

    for path in sessions:
        try:
            result = parse_only(path)
        except Exception as exc:  # noqa: BLE001 - the sweep exists to find these
            failures.append((path, repr(exc)))
            continue
        totals.update(result["counts"])
        for warning in result["warnings"]:
            warnings[warning.split(":")[-1].strip()[:70]] += 1

    print(f"swept {len(sessions)} sessions, {len(failures)} failed\n")
    print("=== events ===")
    for name, count in totals.most_common():
        print(f"{count:8d}  {name}")

    if warnings:
        print("\n=== warnings ===")
        for warning, count in warnings.most_common(10):
            print(f"{count:8d}  {warning}")

    if failures:
        print("\n=== failures ===")
        for path, exc in failures[:20]:
            print(f"  {path.name}: {exc}")
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="eqty-lineage-transcript", description=__doc__)
    parser.add_argument("transcript", nargs="?", help="path to a session .jsonl")
    parser.add_argument("-o", "--output", help="manifest path to write")
    parser.add_argument("--triples", help="also append the fact set to this JSONL path")
    parser.add_argument("--projection", metavar="PATH",
                        help="also write a file-lineage + tools projection of the manifest")
    parser.add_argument("--service", help="register the context with an Integrity Service URL")
    parser.add_argument("--api-key", help="API key for --service")
    parser.add_argument("--sweep", type=int, metavar="N", help="parse N local sessions and report coverage")
    parser.add_argument("--no-blobs", action="store_true", help="register assets without storing content")
    parser.add_argument("--verbose", action="store_true", help="attach raw transcript context to assets")
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.sweep:
        return _sweep(args.sweep, args.verbose)

    if not args.transcript:
        parser.error("a transcript path or --sweep is required")

    result = ingest(
        args.transcript,
        manifest=args.output,
        triples_path=args.triples,
        verbose=args.verbose,
        store_blobs=not args.no_blobs,
        projection=args.projection,
        service_url=args.service,
        service_key=args.api_key,
    )
    violations = check_invariants(result.triples)

    summary = {
        "session": result.session_id,
        "events": result.events,
        "file_versions": result.file_versions,
        "triples": len(result.triples),
        "manifest": str(result.manifest) if result.manifest else None,
        "projection": str(result.projection) if result.projection else None,
        "warnings": result.warnings,
        "violations": [f"{v.invariant}: {v.detail}" for v in violations],
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"session       {summary['session']}")
        print(f"events        {summary['events']}")
        print(f"file versions {summary['file_versions']}")
        print(f"triples       {summary['triples']}")
        if result.manifest:
            print(f"manifest      {result.manifest} ({Path(result.manifest).stat().st_size} bytes)")
        if result.projection:
            print(f"projection    {result.projection} ({Path(result.projection).stat().st_size} bytes)")
        for warning in result.warnings[:10]:
            print(f"  warning: {warning}")
        for violation in violations:
            print(f"  VIOLATION {violation.invariant}: {violation.detail}")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
