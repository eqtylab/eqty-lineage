"""A `jj log`-style rendering of a lineage graph.

Lineage is a DAG whose interesting structure is exactly what a revision-graph renderer is good at:
chains of file versions, fan-in where one tool call consumed several inputs, fan-out where one version
fed several later ones. So it gets drawn the same way -- lanes, glyphs, and connector lines.

Nodes are file versions and the activities that produced them. Everything else -- prompts, model
responses, tool-argument blobs -- is omitted for the same reason the projection omits it: on a real
session those are ~88% of the graph and none of them are what you are reading the log to find.

    ○  app.py#v3                    Bash  sed -i s/2/3/ app.py        inferred
    │
    ◆  Edit                         tool
    ├─╮
    │ ○  app.py#v2
    ○ │  CLAUDE.md#v1
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from eqty_lineage.core import prov

# jj uses @ for the working copy, ○ for ordinary commits, ◆ for immutable ones. Same idea here:
# ● marks the newest version of a path, ○ an earlier one, ◆ an activity, ⊗ something withheld.
GLYPH_HEAD = "●"
GLYPH_VERSION = "○"
GLYPH_ACTIVITY = "◆"
GLYPH_ELIDED = "~"

_C = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "cyan": "\033[36m", "yellow": "\033[33m", "green": "\033[32m",
    "red": "\033[31m", "magenta": "\033[35m", "grey": "\033[90m",
}


def _c(text: str, *styles: str, color: bool = True) -> str:
    if not color or not styles:
        return text
    return "".join(_C[s] for s in styles) + text + _C["reset"]


@dataclass
class Node:
    cid: str
    kind: str  # "version" | "activity"
    label: str
    detail: str = ""
    observed: bool = True
    order: int = 0
    parents: List[str] = field(default_factory=list)


def build_nodes(triples: Sequence) -> Dict[str, Node]:
    """File versions and the activities that produced them, with parent links.

    An activity's parents are the versions it consumed; a version's parent is the activity that
    produced it, or its predecessor version when no activity is recorded. Preferring the activity
    avoids drawing the `wasDerivedFrom` edge alongside the path that explains it -- both are true, but
    the log should show *how* a version came about, not merely that it did.
    """
    triples = list(triples)
    order = {}
    for i, t in enumerate(triples):
        order.setdefault(t.subject, i)
        order.setdefault(t.object, i)

    paths: Dict[str, str] = {}
    kinds: Dict[str, str] = {}
    derived: Dict[str, str] = {}
    used: Dict[str, List[str]] = {}
    generated: Dict[str, str] = {}
    observed: Dict[str, bool] = {}
    labels: Dict[str, str] = {}

    for t in triples:
        if t.predicate == prov.HAS_PATH:
            paths[t.subject] = t.object
            observed[t.subject] = observed.get(t.subject, True) and t.observed
        elif t.predicate == prov.ASSET_TYPE:
            kinds[t.subject] = t.object
        elif t.predicate == prov.WAS_DERIVED_FROM:
            derived[t.subject] = t.object
        elif t.predicate == prov.USED:
            used.setdefault(t.subject, []).append(t.object)
        elif t.predicate == prov.WAS_GENERATED_BY:
            generated[t.subject] = t.object
        elif t.predicate == prov.LABEL:
            labels[t.subject] = t.object

    # version numbering along each path's derivation chain
    version: Dict[str, int] = {}
    for cid in paths:
        n, cur, seen = 1, cid, set()
        while cur in derived and cur not in seen:
            seen.add(cur)
            cur = derived[cur]
            n += 1
        version[cid] = n

    keep_activities = {generated[v] for v in paths if v in generated}
    nodes: Dict[str, Node] = {}

    common = ""
    if paths:
        dirs = [p.rsplit("/", 1)[0] for p in paths.values()]
        common = dirs[0]
        for d in dirs[1:]:
            while common and not d.startswith(common):
                common = common.rsplit("/", 1)[0] if "/" in common else ""
        common = common + "/" if common else ""

    basenames: Dict[str, int] = {}
    for path in paths.values():
        basenames[path.split("/")[-1]] = basenames.get(path.split("/")[-1], 0) + 1

    for cid, path in paths.items():
        parts = path.split("/")
        # a bare basename is ambiguous when several paths share it, which is the common case in a repo
        # (lib.rs, Cargo.toml, mod.rs); qualify with the parent directory only when it is.
        stem = parts[-1] if basenames[parts[-1]] == 1 else "/".join(parts[-2:])
        nodes[cid] = Node(
            cid=cid,
            kind="version",
            label=f"{stem}#v{version[cid]}",
            detail=path[len(common):] if common and path.startswith(common) else path,
            observed=observed.get(cid, True),
            order=order.get(cid, 0),
        )
    for cid in keep_activities:
        nodes[cid] = Node(cid=cid, kind="activity", label="", order=order.get(cid, 0))

    for cid, node in nodes.items():
        if node.kind == "version":
            if cid in generated and generated[cid] in nodes:
                node.parents = [generated[cid]]
            elif cid in derived and derived[cid] in nodes:
                node.parents = [derived[cid]]
        else:
            node.parents = [c for c in used.get(cid, []) if c in nodes and nodes[c].kind == "version"]

    # an activity is named by the file versions it produced
    produced_by: Dict[str, List[str]] = {}
    for v, a in generated.items():
        if a in nodes and v in nodes:
            produced_by.setdefault(a, []).append(nodes[v].label)
    for a in list(nodes):
        if nodes[a].kind != "activity":
            continue
        made = produced_by.get(a, [])
        nodes[a].label = labels.get(a) or (" + ".join(sorted(made)[:2]) if made else "activity")
        n_in = len(nodes[a].parents)
        nodes[a].detail = f"{n_in} input{'' if n_in == 1 else 's'}"

    return nodes


def _topo(nodes: Dict[str, Node]) -> List[str]:
    """Children before parents, newest first -- the order `jj log` uses.

    Ties break on first appearance in the triple stream, which is chronological because the sink
    appends as the session runs.
    """
    children: Dict[str, int] = {c: 0 for c in nodes}
    for node in nodes.values():
        for p in node.parents:
            if p in children:
                children[p] += 1

    ready = sorted([c for c, n in children.items() if n == 0], key=lambda c: -nodes[c].order)
    out: List[str] = []
    seen: Set[str] = set()

    while ready:
        cid = ready.pop(0)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
        for p in nodes[cid].parents:
            if p in children:
                children[p] -= 1
                if children[p] == 0:
                    ready.append(p)
                    ready.sort(key=lambda c: -nodes[c].order)

    for cid in sorted(nodes, key=lambda c: -nodes[c].order):  # cycles / leftovers
        if cid not in seen:
            out.append(cid)
    return out


def render(triples: Sequence, limit: int = 40, color: bool = True, width: int = 34) -> str:
    """Render the lineage DAG as a jj-style graph log."""
    nodes = build_nodes(triples)
    if not nodes:
        return "(no file lineage in this graph)"

    order = _topo(nodes)[:limit]
    shown = set(order)
    heads = {n.detail: n.cid for n in sorted(nodes.values(), key=lambda n: n.order) if n.kind == "version"}

    lines: List[str] = []
    lanes: List[Optional[str]] = []

    for cid in order:
        node = nodes[cid]
        if cid in lanes:
            col = lanes.index(cid)
        else:
            col = len(lanes)
            lanes.append(cid)

        if node.kind == "activity":
            glyph = _c(GLYPH_ACTIVITY, "magenta", color=color)
        elif heads.get(node.detail) == cid:
            glyph = _c(GLYPH_HEAD, "bold", "green", color=color)
        else:
            glyph = _c(GLYPH_VERSION, "cyan", color=color)

        prefix = ["│ "] * len(lanes)
        prefix[col] = glyph + " "
        for i, lane in enumerate(lanes):
            if lane is None:
                prefix[i] = "  "

        label = node.label if node.kind == "version" else f"{node.label}"
        style = ("bold",) if heads.get(node.detail) == cid else ()
        row = "".join(prefix) + " " + _c(f"{label:<{width}}", *style, color=color)

        notes = []
        if node.kind == "activity":
            notes.append(_c("activity", "magenta", color=color))
            if node.detail:
                notes.append(_c(node.detail, "grey", color=color))
        else:
            if not node.observed:
                notes.append(_c("inferred", "yellow", color=color))
            if node.detail and node.detail != node.label.split("#")[0]:
                d = node.detail if len(node.detail) <= 46 else "…" + node.detail[-45:]
                notes.append(_c(d, "grey", color=color))
        lines.append(row + " " + "  ".join(notes))

        # a spacer row carrying the open lanes -- without it consecutive nodes look unrelated
        live = ["│ " if lane is not None else "  " for lane in lanes]
        if any(lane is not None for lane in lanes):
            lines.append(_c("".join(live).rstrip(), "grey", color=color))

        # expand this lane into the node's parents
        parents = [p for p in node.parents if p in shown]
        if not parents:
            lanes[col] = None
            if all(lane is None for lane in lanes):
                lanes = []
            continue

        lanes[col] = parents[0]
        extra = [p for p in parents[1:] if p not in lanes]
        if extra:
            connector = list("│ " * len(lanes))
            # lanes are two columns wide, so the joint must be too, or the branch lands off-lane
            joint = "├─" + "┬─" * (len(extra) - 1) + "╮"
            merged = "".join(connector)[: col * 2] + joint
            lines.append(_c(merged, "grey", color=color))
            lanes.extend(extra)

    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    from eqty_lineage.core import TripleSink

    parser = argparse.ArgumentParser(prog="eqty-lineage-graph", description=render.__doc__)
    parser.add_argument("triples", help="a triples JSONL sidecar")
    parser.add_argument("-n", "--limit", type=int, default=40)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    print(render(TripleSink.load(args.triples), limit=args.limit, color=not args.no_color))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Node", "build_nodes", "main", "render"]
