"""Run one identical agent through the old handler and the new one, and compare the lineage.

The point of the demo is that *only the handler changes*. The graph, the tool calls and the model's
replies are byte-identical between the two runs -- the model is scripted rather than live -- so every
difference in the exported manifest is attributable to the handler and nothing else. A live model would
make the two runs incomparable, which is exactly what a before/after demo must not do.

    just demo

writes ``manifests/before.json`` and ``manifests/after.json``, then prints the comparison. Load the two
manifests side by side in the graph explorer to see the topology differ.

The graph is a document-review agent that exercises five of the ten defects at once::

    START -> draft -> checkpoint -> {verify_a, verify_b} -> revise -> publish -> END

- ``draft`` writes report.md, ``revise`` rewrites it, ``publish`` reads it   (D10)
- ``checkpoint`` returns None, the way LangChain middleware signals no update (D1)
- ``verify_a`` and ``verify_b`` run in one superstep, in parallel            (D9)
- ``verify_a`` calls a model from inside a tool                              (D2)
- ``verify_b`` calls a tool that raises                                      (D7)
"""

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, TypedDict

from eqty_sdk import Context, Signer, init, set_active_signer
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

HANDLER_PATH = "packages/eqty-lineage-langchain/eqty_lineage/langchain/__init__.py"


def load_handler(baseline: bool):
    """Import the handler under test: the committed one, or the one released as 0.0.1.

    Loaded from a file rather than by swapping the checkout so a single command can run both, and so the
    baseline is read from git rather than from whatever happens to be in the working tree.
    """
    if not baseline:
        from eqty_lineage.langchain import EqtyCallbackHandler

        return EqtyCallbackHandler

    source = subprocess.run(
        ["git", "show", f"eqty-lineage-langchain@0.0.1:{HANDLER_PATH}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / "baseline_handler.py"
    tmp.write_text(source)
    spec = importlib.util.spec_from_file_location("eqty_baseline_handler", tmp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EqtyCallbackHandler


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


class ReviewState(TypedDict):
    notes: Annotated[list, lambda a, b: a + b]
    report: Path


def build_graph(report: Path):
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

    def revise(state: ReviewState) -> dict:
        report.write_text("# Report\n\nRevised draft, now with sources.\n")
        return {"notes": ["revised"], "report": report}

    def publish(state: ReviewState) -> dict:
        return {"notes": [f"published {len(state['report'].read_text())} bytes"], "report": report}

    graph = StateGraph(ReviewState)
    for name, fn in (
        ("draft", draft),
        ("checkpoint", checkpoint),
        ("verify_a", verify_a),
        ("verify_b", verify_b),
        ("revise", revise),
        ("publish", publish),
    ):
        graph.add_node(name, fn)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "checkpoint")
    graph.add_edge("checkpoint", "verify_a")
    graph.add_edge("checkpoint", "verify_b")
    graph.add_edge("verify_a", "revise")
    graph.add_edge("verify_b", "revise")
    graph.add_edge("revise", "publish")
    graph.add_edge("publish", END)
    return graph.compile()


# ------------------------------------------------------------------- the run ----


def run(baseline: bool, out: Path) -> dict:
    handler_cls = load_handler(baseline)
    label = "before (0.0.1)" if baseline else "after (this PR)"

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
        "publish_linked_to_current_bytes": current_content in publish_inputs,
        "manifest_statements": len(json.loads(out.read_text()).get("statements", {})),
    }
    (out.parent / f"{out.stem}-summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def compare(before: Path, after: Path) -> None:
    b = json.loads((before.parent / f"{before.stem}-summary.json").read_text())
    a = json.loads((after.parent / f"{after.stem}-summary.json").read_text())

    rows = [
        (
            "exceptions swallowed by LangChain",
            "; ".join(b["swallowed_exceptions"]) or "none",
            "; ".join(a["swallowed_exceptions"]) or "none",
        ),
        ("computations recorded", b["computations"], a["computations"]),
        ("statements in manifest", b["manifest_statements"], a["manifest_statements"]),
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
    print(f"{'':<{width}}   {'BEFORE (0.0.1)':<62}  AFTER (this PR)")
    print("-" * (width + 3 + 62 + 2 + 22))
    for name, before_value, after_value in rows:
        mark = " " if str(before_value) == str(after_value) else "*"
        print(f"{name:<{width}} {mark} {before_value!s:<62}  {after_value}")
    print()
    print(f"  before: {b['manifest']}")
    print(f"  after:  {a['manifest']}")
    print("\nLoad both in the graph explorer to see the topology differ.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="use the handler released as 0.0.1")
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    out = args.out or Path("manifests") / ("before.json" if args.baseline else "after.json")
    summary = run(args.baseline, out)
    print(f"{summary['label']}: {summary['computations']} computations -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
