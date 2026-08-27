"""Run one identical agent through the old handler and the new one, and compare the lineage.

The point of the demo is that *only the handler changes*. The graph, the tool calls and the model's
replies are byte-identical between the two runs -- the model is scripted rather than live -- so every
difference in the exported manifest is attributable to the handler and nothing else. A live model would
make the two runs incomparable, which is exactly what a before/after demo must not do.

    just langchain-diff-demo        # working tree vs. the most recent release tag
    just langchain-diff-demo eqty-lineage-langchain@0.0.1   # ...or any ref you name

writes ``manifests/before.json`` and ``manifests/after.json``, then prints the comparison. Load the two
manifests side by side in the graph explorer to see the topology differ.

The baseline defaults to the newest ``eqty-lineage-langchain@*`` tag rather than a pinned one, so this
keeps answering "what changed since the last release" as releases are cut, instead of freezing into a
comparison against whichever version happened to be current the day it was written.

The graph is a document-review agent that exercises five of the ten defects at once::

    START -> research -> draft -> checkpoint -> {verify_a, verify_b} -> consult -> revise -> publish -> END

- ``draft`` writes report.md, ``revise`` rewrites it, ``publish`` reads it   (D10)
- ``checkpoint`` returns None, the way LangChain middleware signals no update (D1)
- ``verify_a`` and ``verify_b`` run in one superstep, in parallel            (D9)
- ``verify_a`` calls a model from inside a tool                              (D2)
- ``verify_b`` calls a tool that raises                                      (D7)
- ``research`` retrieves from a fixed corpus                                 (D6)
- ``consult`` delegates to a subagent through a tool, as `task` does         (D3)
"""

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, TypedDict

from eqty_sdk import Context, Signer, init, set_active_signer
from langchain_core.documents import Document as LCDocument
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

HANDLER_PATH = "packages/eqty-lineage-langchain/eqty_lineage/langchain/__init__.py"


class DemoError(RuntimeError):
    """A failure in the demo's own scaffolding, reported without a traceback."""


def latest_release_ref() -> str:
    """The newest eqty-lineage-langchain release tag."""
    tags = subprocess.run(
        ["git", "tag", "-l", "eqty-lineage-langchain@*", "--sort=-v:refname"],
        capture_output=True,
        text=True,
        check=False,  # no tags is a case this handles itself, with a better message than a traceback
    ).stdout.split()
    if not tags:
        raise DemoError(
            "no eqty-lineage-langchain@* tag found to use as a baseline. Fetch tags with "
            "`git fetch --tags`, or name a ref explicitly: just langchain-diff-demo <ref>"
        )
    return tags[0]


def load_handler(ref: str | None):
    """Import the handler under test: the working tree's, or the one at ``ref``.

    Read out of git and loaded from a file rather than by swapping the checkout, so one command can run
    both and the baseline is whatever was released rather than whatever is lying around locally.
    """
    if ref is None:
        from eqty_lineage.langchain import EqtyCallbackHandler

        return EqtyCallbackHandler

    result = subprocess.run(
        ["git", "show", f"{ref}:{HANDLER_PATH}"],
        capture_output=True,
        text=True,
        check=False,  # a bad ref gets an explanation below, not a CalledProcessError
    )
    if result.returncode != 0:
        raise DemoError(
            f"could not read {HANDLER_PATH} at '{ref}'.\n"
            f"  git said: {result.stderr.strip()}\n"
            "  A shallow clone does not carry old trees -- try `git fetch --unshallow --tags`."
        )

    tmp = Path(tempfile.mkdtemp()) / "baseline_handler.py"
    tmp.write_text(result.stdout)
    spec = importlib.util.spec_from_file_location("eqty_baseline_handler", tmp)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.EqtyCallbackHandler
    except Exception as exc:
        raise DemoError(
            f"the handler at '{ref}' no longer imports against the installed dependencies "
            f"({type(exc).__name__}: {exc}). Pick a more recent baseline ref."
        ) from exc


# ------------------------------------------------------------------ the agent ----

REVIEWER = GenericFakeChatModel(messages=iter([AIMessage("The draft cites no sources.")]))


@tool
def critique(text: str) -> str:
    """Critique a passage by asking a model -- a model call nested inside a tool call."""
    return REVIEWER.invoke(text).content


@tool
def check_links(text: str) -> str:
    """Validate the links in a passage. Always fails, to show a failure in the graph."""
    raise RuntimeError("link checker unavailable")


class SourceLibrary(BaseRetriever):
    """A fixed corpus, so the retrieved documents are the same on every run."""

    def _get_relevant_documents(self, query, *, run_manager=None):
        return [
            LCDocument(page_content="CIDs are self-describing hashes.", metadata={"source": "cid.md"}),
            LCDocument(page_content="Lineage links inputs to outputs.", metadata={"source": "lineage.md"}),
        ]


def _specialist_graph():
    """A second agent, invoked through a tool the way DeepAgents' `task` invokes a subagent."""

    class SpecialistState(TypedDict):
        findings: Annotated[list, lambda a, b: a + b]

    graph = StateGraph(SpecialistState)
    graph.add_node("assess", lambda state: {"findings": ["specialist: the sources check out"]})
    graph.add_edge(START, "assess")
    graph.add_edge("assess", END)
    return graph.compile()


SPECIALIST = _specialist_graph()


@tool
def consult_specialist(question: str) -> str:
    """Delegate to a specialist subagent, the way DeepAgents' `task` tool does."""
    out = SPECIALIST.invoke({"findings": []}, config={"metadata": {"lc_agent_name": "specialist"}})
    return f"consulted -> {'; '.join(out['findings'])}"


class ReviewState(TypedDict):
    notes: Annotated[list, lambda a, b: a + b]
    report: Path


def build_graph(report: Path):
    def research(state: ReviewState) -> dict:
        docs = SourceLibrary().invoke("what is lineage")
        return {"notes": [f"researched {len(docs)} sources"], "report": report}

    def draft(state: ReviewState) -> dict:
        report.write_text("# Report\n\nInitial draft.\n")
        return {"notes": ["drafted"], "report": report}

    def checkpoint(state: ReviewState) -> None:
        """Returns None: no state update, the way LangChain middleware nodes do."""
        return

    # the two verify nodes run in the same superstep, so neither writes `report` -- a LastValue channel
    # takes one write per step, and neither of them changes the file anyway
    def verify_a(state: ReviewState) -> dict:
        return {"notes": [critique.invoke({"text": state["report"].read_text()})]}

    def verify_b(state: ReviewState) -> dict:
        try:
            check_links.invoke({"text": state["report"].read_text()})
        except RuntimeError as exc:
            return {"notes": [f"link check failed: {exc}"]}
        return {"notes": ["links ok"]}

    def consult(state: ReviewState) -> dict:
        return {"notes": [consult_specialist.invoke({"question": "are the sources sound?"})]}

    def revise(state: ReviewState) -> dict:
        report.write_text("# Report\n\nRevised draft, now with sources.\n")
        return {"notes": ["revised"], "report": report}

    def publish(state: ReviewState) -> dict:
        return {"notes": [f"published {len(state['report'].read_text())} bytes"], "report": report}

    graph = StateGraph(ReviewState)
    for name, fn in (
        ("research", research),
        ("draft", draft),
        ("checkpoint", checkpoint),
        ("verify_a", verify_a),
        ("verify_b", verify_b),
        ("consult", consult),
        ("revise", revise),
        ("publish", publish),
    ):
        graph.add_node(name, fn)

    graph.add_edge(START, "research")
    graph.add_edge("research", "draft")
    graph.add_edge("draft", "checkpoint")
    graph.add_edge("checkpoint", "verify_a")
    graph.add_edge("checkpoint", "verify_b")
    graph.add_edge("verify_a", "consult")
    graph.add_edge("verify_b", "consult")
    graph.add_edge("consult", "revise")
    graph.add_edge("revise", "publish")
    graph.add_edge("publish", END)
    return graph.compile()


# ------------------------------------------------------------------- the run ----


def run(ref: str | None, out: Path) -> dict:
    handler_cls = load_handler(ref)
    # a release tag reads as its version; anything else (a sha, a branch) is shown abbreviated
    shown = ref.split("@")[-1] if ref else ""
    label = f"baseline ({shown[:12]})" if ref else "working tree"

    # The SDK's Rust side logs unresolvable JSON-LD contexts at ERROR while exporting. Harmless, but it
    # reads as a failure to anyone seeing this for the first time, which is the whole audience for a
    # demo. Silenced with a handler rather than a level: the level is cached across the Rust/Python
    # logging bridge, so setLevel here arrives too late, whereas giving the logger a handler stops
    # logging's last-resort fallback from printing to stderr.
    quiet = logging.getLogger("integrity_lineage_models")
    quiet.addHandler(logging.NullHandler())
    quiet.propagate = False
    # the handler warns about the tool this graph deliberately fails; the table already reports it
    logging.getLogger("eqty.langgraph").setLevel(logging.ERROR)

    recorded: list[dict[str, Any]] = []

    class Recording(handler_cls):  # type: ignore[misc, valid-type]
        def _finalize(self, name, kind, input_cids, output_cids):
            recorded.append(
                {
                    "name": name,
                    "kind": kind,
                    "inputs": [str(c) for c in input_cids],
                    "outputs": [str(c) for c in output_cids],
                }
            )
            return super()._finalize(name, kind, input_cids, output_cids)

    # init() puts its .eqty_sdk store under the working directory, so each run gets a fresh one in a
    # temp dir. Two runs sharing a store make the second one's manifest depend on what the first left
    # behind -- the statement counts drifted between invocations before this -- and a demo whose numbers
    # move when you run it twice is not a demo. It also keeps 6MB of store out of the repo.
    out = out.resolve()
    store = Path(tempfile.mkdtemp(prefix="eqty-demo-"))
    os.chdir(store)

    ctx = Context.new(f"Document review -- {label}")
    cfg = init(default_context=ctx).set_store_all_blobs(True)
    set_active_signer(Signer.new(name="lineage-demo", _load_if_exists=True))

    # LangChain catches whatever a callback raises and logs it at WARNING, so a handler that blows up
    # costs a statement without failing anything. Counting those log records is the most direct way to
    # show what the old handler was losing.
    swallowed: list[str] = []

    class CountSwallowed(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if "Error in" in message and "callback" in message:
                swallowed.append(message.split(": ", 1)[-1])

    logging.getLogger("langchain_core.callbacks.manager").addHandler(CountSwallowed())

    report = Path(tempfile.mkdtemp()) / "report.md"
    handler = Recording()
    build_graph(report).invoke({"notes": [], "report": report}, config={"callbacks": [handler]})

    out.parent.mkdir(parents=True, exist_ok=True)
    cfg.get_default_context().export(out)

    # Deliberately not reported: the manifest's raw statement and asset counts. Assets are
    # content-addressed, so two state payloads that happen to coincide collapse into one registration and
    # the totals move by one between otherwise identical runs. Everything below is a property of the
    # lineage rather than of the store's accounting, and is stable run to run.
    consumed = {i for c in recorded for i in c["inputs"]}
    orphans = [(c["name"], o) for c in recorded if c["kind"] == "graph_node" for o in c["outputs"] if o not in consumed]
    publish_inputs = {i for c in recorded if c["name"] == "publish" for i in c["inputs"]}

    # `Dataset.from_path` content-addresses the file, so the CID of the bytes on disk right now is the
    # asset a correct manifest must have linked into `publish`. Asking the question this way rather than
    # asking the handler what it thinks the latest version is: the old handler's answer is the defect.
    from eqty_sdk import get_cid_for_path

    current_content = str(get_cid_for_path(report))
    versions = getattr(handler, "_path_versions", None)
    tracked = len(versions) if versions is not None else len(handler._path_cids)

    summary = {
        "label": label,
        "manifest": str(out),
        "computations": len(recorded),
        "node_names": sorted({c["name"] for c in recorded if c["kind"].startswith("graph_node")}),
        "kinds": sorted({c["kind"] for c in recorded}),
        "orphaned_outputs": [name for name, _ in orphans],
        "file_versions_registered": tracked,
        "swallowed_exceptions": sorted(set(swallowed)),
        "retrievals": sum(1 for c in recorded if c["kind"] == "retriever"),
        "documents_registered": sum(len(c["outputs"]) for c in recorded if c["kind"] == "retriever"),
        "subagents": sorted(c["name"] for c in recorded if c["kind"] == "agent"),
        "publish_linked_to_current_bytes": current_content in publish_inputs,
    }
    return summary


def compare(stream) -> None:
    """Read the two runs' summaries, one compact JSON object per line."""
    lines = [line for line in stream.read().splitlines() if line.strip()]
    if len(lines) != 2:
        raise DemoError(f"expected two summary lines on stdin, got {len(lines)}")
    b, a = (json.loads(line) for line in lines)

    rows = [
        ("handler", b["label"], a["label"]),
        (
            "exceptions swallowed by LangChain",
            "; ".join(b["swallowed_exceptions"]) or "none",
            "; ".join(a["swallowed_exceptions"]) or "none",
        ),
        ("computations recorded", b["computations"], a["computations"]),
        ("retrievals recorded", b["retrievals"], a["retrievals"]),
        ("documents registered", b["documents_registered"], a["documents_registered"]),
        ("subagents recorded", ", ".join(b["subagents"]) or "none", ", ".join(a["subagents"]) or "none"),
        ("graph nodes present", ", ".join(b["node_names"]) or "-", ", ".join(a["node_names"]) or "-"),
        ("computation kinds", ", ".join(b["kinds"]), ", ".join(a["kinds"])),
        (
            "orphaned node outputs",
            ", ".join(b["orphaned_outputs"]) or "none",
            ", ".join(a["orphaned_outputs"]) or "none",
        ),
        ("report.md versions tracked", b["file_versions_registered"], a["file_versions_registered"]),
        (
            "publish linked to the bytes it read",
            b["publish_linked_to_current_bytes"],
            a["publish_linked_to_current_bytes"],
        ),
    ]

    width = max(len(r[0]) for r in rows)
    print()
    print(f"{'':<{width}}   {'BEFORE':<62}  AFTER")
    print("-" * (width + 3 + 62 + 2 + 22))
    for name, before_value, after_value in rows:
        mark = " " if str(before_value) == str(after_value) else "*"
        print(f"{name:<{width}} {mark} {before_value!s:<62}  {after_value}")
    print()
    root = Path.cwd()
    for entry in (b, a):
        path = Path(entry["manifest"])
        try:
            entry["manifest"] = str(path.relative_to(root))
        except ValueError:
            pass
    print(f"  before: {b['manifest']}")
    print(f"  after:  {a['manifest']}")
    print("\nLoad both in the graph explorer to see the topology differ.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        nargs="?",
        const="",
        metavar="REF",
        help="run the handler at this git ref instead of the working tree "
        "(defaults to the newest eqty-lineage-langchain@* tag)",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="read two summary lines from stdin and print the comparison",
    )
    args = parser.parse_args()

    try:
        if args.compare:
            compare(sys.stdin)
            return

        ref = None
        if args.baseline is not None:
            ref = args.baseline or latest_release_ref()

        out = args.out or Path("manifests") / ("before.json" if ref else "after.json")
        summary = run(ref, out)
        # the summary goes to stdout for the compare step; progress goes to stderr for the human
        print(json.dumps(summary), flush=True)
        print(f"{summary['label']}: {summary['computations']} computations -> {out}", file=sys.stderr)
    except DemoError as exc:
        print(f"demo: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
