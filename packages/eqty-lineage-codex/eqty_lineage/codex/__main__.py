"""``eqty-codex-lineage`` — turn a raw Codex hook capture into a signed manifest.

Also reachable as ``python -m eqty_lineage.codex``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eqty_lineage.codex import replay_session
from eqty_lineage.codex.capture import DENY, UNKNOWN, CodexSession, load_session


def default_output(capture: Path) -> Path:
    """``codex-hooks.jsonl`` -> ``codex-hooks.lineage.json``, next to the capture."""
    return (
        capture.with_suffix("").with_suffix(".lineage.json")
        if capture.suffix
        else capture.with_name(capture.name + ".lineage.json")
    )


def summarize(session: CodexSession) -> list[str]:
    lines = [f"session {session.session_id}" + (f" · {session.model}" if session.model else "")]
    lines.append(f"  {len(session.prompts)} prompt(s), {len(session.attempts)} tool attempt(s)")
    for attempt in session.attempts:
        lines.append(f"    {attempt.decision:<8} executed={str(attempt.executed):<5} {attempt.tool_name}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eqty-codex-lineage",
        description="Replay a Codex hook capture into a signed EQTY lineage manifest.",
        epilog="A denied or unresolved call is reported but carries no result node in the graph.",
    )
    parser.add_argument("capture", type=Path, help="hook capture written by the collector (JSONL, or a JSON array)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="manifest to write (default: the capture with a .lineage.json suffix)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the manifest path")
    parser.add_argument("--json", action="store_true", help="print a machine-readable summary instead of a table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.capture.exists():
        print(f"eqty-codex-lineage: no such capture: {args.capture}", file=sys.stderr)
        return 2

    try:
        session = load_session(args.capture)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"eqty-codex-lineage: cannot read {args.capture}: {exc}", file=sys.stderr)
        return 2

    if not session.attempts and not session.prompts:
        # An empty graph is a valid export, but it is almost always a wiring mistake rather than a
        # session in which the agent genuinely did nothing, so say so rather than exit 0 in silence.
        print(f"eqty-codex-lineage: {args.capture} contains no prompts or tool calls", file=sys.stderr)
        return 3

    output = args.output or default_output(args.capture)
    manifest = replay_session(session, output)

    if args.json:
        print(
            json.dumps(
                {
                    "manifest": str(manifest),
                    "session_id": session.session_id,
                    "model": session.model,
                    "prompts": len(session.prompts),
                    "attempts": [
                        {"tool": a.tool_name, "decision": a.decision, "executed": a.executed} for a in session.attempts
                    ],
                }
            )
        )
        return 0

    if not args.quiet:
        for line in summarize(session):
            print(line)
        unresolved = sum(a.decision == UNKNOWN for a in session.attempts)
        if unresolved:
            print(f"  {unresolved} call(s) unresolved — the capture may have been cut mid-call", file=sys.stderr)
        if any(a.decision == DENY and a.executed for a in session.attempts):
            print("  a denied call is recorded as having executed — the hook may have failed open", file=sys.stderr)

    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
