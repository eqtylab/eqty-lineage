"""CLI: run the hook daemon, handle a single command hook, or print settings wiring.

    eqty-lineage-hooks serve --port 8787 --manifests ./manifests
    eqty-lineage-hooks hook                        # reads one payload on stdin (fallback transport)
    eqty-lineage-hooks install --print             # settings.json / hooks.json snippet
"""

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


TOKEN_ENV = "EQTY_LINEAGE_TOKEN"
API_KEY_ENV = "EQTY_LINEAGE_API_KEY"


def _secret(explicit: Optional[str], env_var: str) -> Optional[str]:
    """Prefer the environment; accept a flag as an override.

    A secret passed on the command line is visible in ``ps`` to every user on the box, and the daemon's
    token is the only thing guarding a port carrying prompts and file contents. Claude Code's HTTP hooks
    can forward an env var into the Authorization header (``headers`` + ``allowedEnvVars``), so the
    environment is also where the *other* end of this already reads it from.
    """
    if explicit:
        return explicit
    return os.environ.get(env_var) or None


def _registry(args) -> SessionRegistry:
    return SessionRegistry(
        state_dir=Path(args.state_dir) if args.state_dir else None,
        triples_dir=Path(args.triples) if args.triples else None,
        verbose=args.verbose,
        store_blobs=args.blobs,
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
        service_key=_secret(args.api_key, API_KEY_ENV),
    )
    token = _secret(args.token, TOKEN_ENV)
    server, _thread = serve(receiver, host=args.host, port=args.port, token=token)
    print(f"eqty-lineage hook daemon on http://{args.host}:{args.port}  (ctrl-c to stop)")
    if not token and args.host != "127.0.0.1":
        # Loopback is the only bind for which "no token" is a defensible default; every payload here
        # carries prompts, file contents and tool output.
        print(f"  WARNING: bound to {args.host} with no token; set {TOKEN_ENV}")
    if args.blobs:
        print("  storing file contents as blobs (--blobs)")
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

    receiver = HookReceiver(
        registry=_registry(args),
        # Both directions, matching `serve`. Checking only --allow-write meant a deny-only policy was
        # accepted on the command line and silently enforced nothing over this transport.
        policy=_policy(args) if (args.allow_write or args.deny_write) else None,
    )
    try:
        print(json.dumps(receiver.handle(payload)))
    except Exception:  # noqa: BLE001 - a crashing hook must not break the agent's turn
        logging.getLogger("eqty.lineage.hooks").exception("hook failed")
        print(json.dumps({}))
    return 0


def _curl_forwarder(url: str, no_token: bool, escape_quotes: bool = False) -> str:
    """A shell command that POSTs the hook payload on stdin to the daemon.

    The transport of last resort, used wherever an HTTP hook type is unavailable. The credential is a
    shell variable reference rather than a literal, so the emitted config can be committed.

    ``escape_quotes`` is for TOML, where this string is emitted inside a double-quoted value and its own
    quotes must be escaped by hand. JSON output goes through ``json.dumps``, which escapes them itself --
    doing it twice yields ``\\"`` in the file, and the shell then splits the header into three arguments.
    """
    quote = '\\"' if escape_quotes else '"'
    auth = "" if no_token else f"-H {quote}Authorization: Bearer ${TOKEN_ENV}{quote} "
    return f"curl -s -m 10 -X POST -H 'Content-Type: application/json' {auth}--data-binary @- {url}"


def _install(args) -> int:
    url = f"http://{args.host}:{args.port}/hook"
    if args.dialect == CODEX:
        # Verified against codex-cli 0.145.0. Codex reads TOML, not JSON, and has no HTTP hook type --
        # the command shells out to curl. Writing this as a *named profile*
        # ($CODEX_HOME/<name>.config.toml, selected with `codex exec -p <name>`) leaves the user's
        # config.toml untouched.
        forwarder = _curl_forwarder(url, args.no_token, escape_quotes=True)
        print("# Codex: write to ~/.codex/eqty.config.toml, then run:  codex exec -p eqty ...")
        print(f"# Start the daemon first: eqty-lineage-hooks serve --port {args.port}")
        if not args.no_token:
            print(f"# Export {TOKEN_ENV} in the shell you launch Codex from; the hook forwards it.")
        print("#")
        print("# Codex requires hooks to be trusted. For automation that vets its own hook sources,")
        print("# pass --dangerously-bypass-hook-trust; otherwise trust them once interactively.")
        for event in CODEX_EVENTS:
            print(f"\n[[hooks.{event}]]")
            print(f"[[hooks.{event}.hooks]]")
            print('type = "command"')
            print(f'command = "{forwarder}"')
        return 0

    # The token travels as an env var reference, not a literal: settings.json is commonly committed,
    # and `allowedEnvVars` is what permits the substitution at all.
    handler: Dict[str, Any] = {"type": "http", "url": url, "timeout": 10}
    if not args.no_token:
        handler["headers"] = {"Authorization": f"Bearer ${TOKEN_ENV}"}
        handler["allowedEnvVars"] = [TOKEN_ENV]

    # SessionStart does not accept an HTTP hook -- only "command" and "mcp_tool". Wiring it as HTTP is
    # accepted by the settings file and then silently never delivered, which costs more than it looks:
    # SessionStart is what creates the Agent asset (so the manifest attests a computation rather than a
    # transcript) and it is where `watchPaths` is returned, so without it no FileChanged ever fires and
    # every Bash side effect stays unobserved. Confirmed against Claude Code 2.1.220 by running a
    # session and watching the event never arrive.
    session_start = {"type": "command", "command": _curl_forwarder(url, args.no_token), "timeout": 10}

    config = {
        "hooks": {
            event: [{"hooks": [session_start if event == "SessionStart" else handler]}]
            for event in CLAUDE_CODE_EVENTS
        }
    }
    print("# Claude Code: merge into .claude/settings.json (or settings.local.json)")
    print(f"# Start the daemon first: {TOKEN_ENV}=$(openssl rand -hex 16) \\")
    print(f"#     eqty-lineage-hooks serve --port {args.port}")
    if not args.no_token:
        print(f"# Export the same {TOKEN_ENV} in the shell you launch the agent from.")
    print("#")
    print("# SessionStart is a command hook on purpose: it is the one event that accepts no HTTP")
    print("# handler, and it carries watchPaths. Everything else is delivered over HTTP.")
    print(json.dumps(config, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="eqty-lineage-hooks", description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="attach raw hook context to assets")
    parser.add_argument("--blobs", action="store_true",
                        help="store file contents in the blob store (off by default; lineage does not "
                             "need it and it makes every file the agent read durable on disk)")
    parser.add_argument("--state-dir", help="where per-session sidecars live")
    parser.add_argument("--triples", help="directory for per-session triple sidecars")
    parser.add_argument("--allow-write", action="append", help="glob a write must match (repeatable)")
    parser.add_argument("--deny-write", action="append", help="glob a write must not match (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="log policy violations without denying")

    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the HTTP hook daemon")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--token", help=f"require this bearer token on every request (prefer ${TOKEN_ENV}, "
                                   "which is not visible in the process list)")
    s.add_argument("--watch", action="append", help="path to watch for out-of-band changes (repeatable)")
    s.add_argument("--manifests", help="export a manifest per session into this directory")
    s.add_argument("--projections", help="also write a file-lineage + tools projection per session")
    s.add_argument("--service", help="register each session with an Integrity Service URL")
    s.add_argument("--api-key", help=f"API key for --service (prefer ${API_KEY_ENV})")
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
    # Accepted because both READMEs document `install --print`, and printing is all this does. Nothing
    # is written to the user's settings by this command.
    i.add_argument("--print", action="store_true", dest="print_only",
                   help="print the wiring to stdout (the only behaviour; accepted for symmetry)")
    i.add_argument("--no-token", action="store_true",
                   help=f"omit the ${TOKEN_ENV} Authorization header from the emitted config")
    i.set_defaults(func=_install)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
