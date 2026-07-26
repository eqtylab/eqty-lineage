"""CLI: run the hook daemon, handle a single command hook, or print settings wiring.

    eqty-lineage-hooks serve --port 8787 --manifests ./manifests
    eqty-lineage-hooks hook                        # reads one payload on stdin (fallback transport)
    eqty-lineage-hooks install --print             # settings.json / hooks.json snippet
"""

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

from .dialects import CLAUDE_CODE, CLAUDE_CODE_EVENTS, CODEX, CODEX_EVENTS
from .daemon import HookReceiver, serve
from .policy import HookPolicy
from .session import SessionRegistry


def _replay(args) -> int:
    """Reinterpret a recorded trace under alternative policies."""
    from eqty_lineage.core import TripleSink

    from .policy import HookPolicy
    from .replay_policy import compare_policies

    groups: dict = {}
    for spec, key in ((args.deny or [], "deny"), (args.allow or [], "allow")):
        for item in spec:
            if "=" not in item:
                print(f"expected NAME=GLOB, got {item!r}", file=sys.stderr)
                return 2
            name, glob = item.split("=", 1)
            groups.setdefault(name, {"deny": [], "allow": []})[key].append(glob)

    policies = {"as-recorded": HookPolicy()}
    for name, rules in groups.items():
        policies[name] = HookPolicy(
            deny_write_globs=tuple(rules["deny"]),
            allow_write_globs=tuple(rules["allow"]),
        )
    if len(policies) == 1:
        print("give at least one --deny or --allow to compare against", file=sys.stderr)
        return 2

    print(compare_policies(TripleSink.load(args.triples), policies).render())
    return 0


def _verify(args) -> int:
    """Conformance-check both capture paths against real sessions."""
    import os
    import tempfile

    from eqty_lineage.transcript.claude_code import find_sessions

    from .equivalence import compare

    sessions = [
        p for p in find_sessions()
        if args.min_records <= sum(1 for _ in p.open(errors="replace")) <= args.max_records
    ][: args.limit]
    if not sessions:
        print("no sessions in the requested size range", file=sys.stderr)
        return 1

    os.chdir(tempfile.mkdtemp(prefix="eqty-verify-"))
    from eqty_sdk import Signer, init, set_active_signer

    init().set_store_all_blobs(False)
    set_active_signer(Signer.new(name="eqty-lineage-verify", _load_if_exists=True))

    failed = 0
    buckets: dict = {}
    for path in sessions:
        report = compare(path)
        print(" ", report.summary())
        failed += 0 if report.ok else 1
        for k, v in report.buckets.items():
            buckets[k] = buckets.get(k, 0) + v

    print(f"\n{len(sessions) - failed}/{len(sessions)} sessions conform")
    print("divergence buckets across all sessions:")
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {v:7d}  {k}")
    return 1 if failed else 0


def _registry(args) -> SessionRegistry:
    return SessionRegistry(
        state_dir=Path(args.state_dir) if args.state_dir else None,
        triples_dir=Path(args.triples) if args.triples else None,
        verbose=args.verbose,
        store_blobs=not args.no_blobs,
    )


def _policy(args) -> HookPolicy:
    return HookPolicy(
        allow_write_globs=tuple(args.allow_write or ()),
        deny_write_globs=tuple(args.deny_write or ()),
        dry_run=args.dry_run,
    )


def _serve(args) -> int:
    receiver = HookReceiver(
        registry=_registry(args),
        policy=_policy(args) if (args.allow_write or args.deny_write) else None,
        watch_paths=args.watch or [],
        manifest_dir=Path(args.manifests) if args.manifests else None,
        projection_dir=Path(args.projections) if args.projections else None,
        service_url=args.service,
        service_key=args.api_key,
    )
    server, _thread = serve(receiver, host=args.host, port=args.port, token=args.token)
    print(f"eqty-lineage hook daemon on http://{args.host}:{args.port}  (ctrl-c to stop)")
    if args.watch:
        print(f"  watching {len(args.watch)} path(s) -> Bash effects recorded as observed")

    stop = signal.SIGINT

    def _shutdown(*_a):
        server.shutdown()

    signal.signal(stop, _shutdown)
    try:
        _thread.join()
    except KeyboardInterrupt:
        server.shutdown()
    print(f"handled {receiver.handled} hook events, {receiver.errors} errors")
    return 0


def _hook(args) -> int:
    """Single-shot command-hook mode: one payload on stdin, one JSON response on stdout.

    Lossier than HTTP -- a fresh process cannot remember an open tool call -- and it pays interpreter
    startup per event. Provided for environments where the daemon cannot run.
    """
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        print(json.dumps({}))
        return 0
    if not isinstance(payload, dict):
        print(json.dumps({}))
        return 0

    receiver = HookReceiver(registry=_registry(args), policy=_policy(args) if args.allow_write else None)
    try:
        print(json.dumps(receiver.handle(payload)))
    except Exception:  # noqa: BLE001 - a crashing hook must not break the agent's turn
        logging.getLogger("eqty.lineage.hooks").exception("hook failed")
        print(json.dumps({}))
    return 0


def _install(args) -> int:
    url = f"http://{args.host}:{args.port}/hook"
    if args.dialect == CODEX:
        # Verified against codex-cli 0.145.0. Codex reads TOML, not JSON, and has no HTTP hook type --
        # the command shells out to curl. Writing this as a *named profile*
        # ($CODEX_HOME/<name>.config.toml, selected with `codex exec -p <name>`) leaves the user's
        # config.toml untouched.
        forwarder = "curl -s -m 10 -X POST -H 'Content-Type: application/json' --data-binary @- " + url
        print("# Codex: write to ~/.codex/eqty.config.toml, then run:  codex exec -p eqty ...")
        print(f"# Start the daemon first: eqty-lineage-hooks serve --port {args.port}")
        print("#")
        print("# Codex requires hooks to be trusted. For automation that vets its own hook sources,")
        print("# pass --dangerously-bypass-hook-trust; otherwise trust them once interactively.")
        for event in CODEX_EVENTS:
            print(f"\n[[hooks.{event}]]")
            print(f"[[hooks.{event}.hooks]]")
            print('type = "command"')
            print(f'command = "{forwarder}"')
        return 0

    config = {
        "hooks": {
            event: [{"hooks": [{"type": "http", "url": url, "timeout": 10}]}]
            for event in CLAUDE_CODE_EVENTS
        }
    }
    print("# Claude Code: merge into .claude/settings.json (or settings.local.json)")
    print(f"# Start the daemon first: eqty-lineage-hooks serve --port {args.port}")
    print(json.dumps(config, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="eqty-lineage-hooks", description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="attach raw hook context to assets")
    parser.add_argument("--no-blobs", action="store_true", help="register assets without storing content")
    parser.add_argument("--state-dir", help="where per-session sidecars live")
    parser.add_argument("--triples", help="directory for per-session triple sidecars")
    parser.add_argument("--allow-write", action="append", help="glob a write must match (repeatable)")
    parser.add_argument("--deny-write", action="append", help="glob a write must not match (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="log policy violations without denying")

    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the HTTP hook daemon")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--token", help="require this bearer token on every request")
    s.add_argument("--watch", action="append", help="path to watch for out-of-band changes (repeatable)")
    s.add_argument("--manifests", help="export a manifest per session into this directory")
    s.add_argument("--projections", help="also write a file-lineage + tools projection per session")
    s.add_argument("--service", help="register each session with an Integrity Service URL")
    s.add_argument("--api-key", help="API key for --service")
    s.set_defaults(func=_serve)

    h = sub.add_parser("hook", help="handle one hook payload from stdin")
    h.set_defaults(func=_hook)

    v = sub.add_parser("verify", help="conformance-check the offline and live paths against real sessions")
    v.add_argument("--limit", type=int, default=10)
    v.add_argument("--min-records", type=int, default=150)
    v.add_argument("--max-records", type=int, default=900)
    v.set_defaults(func=_verify)

    r = sub.add_parser("replay", help="what would this session have produced under another policy?")
    r.add_argument("triples", help="a triples JSONL sidecar")
    r.add_argument("--deny", action="append", metavar="NAME=GLOB",
                   help="deny writes matching GLOB under policy NAME (repeatable)")
    r.add_argument("--allow", action="append", metavar="NAME=GLOB",
                   help="permit only writes matching GLOB under policy NAME (repeatable)")
    r.set_defaults(func=_replay)

    i = sub.add_parser("install", help="print the settings wiring")
    i.add_argument("--dialect", choices=[CLAUDE_CODE, CODEX], default=CLAUDE_CODE)
    i.add_argument("--host", default="127.0.0.1")
    i.add_argument("--port", type=int, default=8787)
    i.set_defaults(func=_install)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
