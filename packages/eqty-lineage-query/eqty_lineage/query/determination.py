"""Determination provenance over a set of agent runs.

An agent is a system with multiple possible outcomes for identical inputs, which is precisely the class
classical provenance cannot describe: it explains a result only *after* an outcome has been chosen.
Determination provenance annotates each fact with its **support** -- the set of resolutions under which
it holds -- and asks which conclusions survive every resolution.

Here a resolution is an observed run. That choice is what makes this cheap: robustness over an
implicitly-described determination space is coNP-complete, but over an enumerated one it is a bitwise
AND. We never describe the space, we run it.

Two supports are worth distinguishing and both are computed here:

*Entity support* -- which runs contain this artifact. A direct lookup, and the basis of the robustness
report: did every run produce the same bytes at this path?

*Derivation support* -- in which runs is Y reachable from X. Needs the closure, evaluated in the
determination semiring, and answers "in which runs did this input actually influence that output".

The empirical case for caring: six runs of a task specified down to the algorithm produced five
distinct implementations, three behaviourally distinct, all passing the same dictated test.
"""

import collections
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from eqty_lineage.core import prov

from .accel import as_triples, resolve
from .engine import edb_from_triples, evaluate
from .queries import INFLUENCE_RULES
from .semiring import determination


@dataclass
class MultiRun:
    """The union of several runs' lineage, with per-fact run membership.

    Entities are content-addressed, so identical artifacts across runs collapse to one node with no
    correlation ids and no shared database -- the graphs join because identical bytes hash identically.
    """

    runs: List[str]
    triples: List
    entity_runs: Dict[str, int] = field(default_factory=dict)
    edge_runs: Dict[Tuple[str, str], int] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)

    @property
    def all_runs(self) -> int:
        return (1 << len(self.runs)) - 1

    def names(self, mask: int) -> List[str]:
        return [r for i, r in enumerate(self.runs) if mask >> i & 1]

    def count(self, mask: int) -> int:
        return bin(mask).count("1")


def union_runs(runs: Mapping[str, Iterable], roots: Optional[Mapping[str, str]] = None) -> MultiRun:
    """Join several runs into one graph, recording which runs each fact came from.

    ``roots`` maps each run to the working-tree prefix to strip from its paths. Replicated runs execute
    in *different directories*, so without normalization ``runs/a/util.py`` and ``runs/b/util.py`` are
    distinct paths and nothing ever groups across runs -- the comparison silently reports every path as
    appearing in exactly one run. Entity CIDs are unaffected either way, being content-addressed, which
    is what makes the mistake quiet rather than loud.
    """
    names = list(runs)
    index = {name: i for i, name in enumerate(names)}
    roots = dict(roots or {})

    triples: List = []
    seen: Set[Tuple[str, str, str]] = set()
    entity_runs: Dict[str, int] = {}
    edge_runs: Dict[Tuple[str, str], int] = {}
    paths: Dict[str, str] = {}

    for name, sink in runs.items():
        bit = 1 << index[name]
        for t in sink:
            key = (t.subject, t.predicate, t.object)
            if key not in seen:
                seen.add(key)
                triples.append(t)
            entity_runs[t.subject] = entity_runs.get(t.subject, 0) | bit
            if t.predicate == prov.HAS_PATH:
                paths[t.subject] = _relative(t.object, roots.get(name))
            else:
                entity_runs[t.object] = entity_runs.get(t.object, 0) | bit
            edge_runs[(t.subject, t.object)] = edge_runs.get((t.subject, t.object), 0) | bit

    return MultiRun(runs=names, triples=triples, entity_runs=entity_runs, edge_runs=edge_runs, paths=paths)


def _relative(path: str, root: Optional[str]) -> str:
    if not root:
        return path
    root = root.rstrip("/") + "/"
    return path[len(root):] if path.startswith(root) else path


# ------------------------------------------------------------------ entity-level
def support(multi: MultiRun, cid: str) -> int:
    """The set of runs in which this artifact exists, as a bitmask."""
    return multi.entity_runs.get(cid, 0)


def qdepth(multi: MultiRun, cid: str) -> int:
    """Query-relative depth: 0 when the support is full or empty, 1 when it depends on the run.

    With a flat resolution space -- one layer, "which run" -- the filtration has only two levels, so
    ``qdepth`` is the robust/fragile classification and nothing finer. Deeper filtrations become
    meaningful once resolutions are layered (policy *and* run), which is what policy replay adds.
    """
    s = support(multi, cid)
    return 0 if s in (0, multi.all_runs) else 1


def robust(multi: MultiRun) -> Set[str]:
    """Artifacts present in every run: full support, ``qdepth`` 0."""
    return {cid for cid, mask in multi.entity_runs.items() if mask == multi.all_runs}


# ------------------------------------------------------------------ path-level (the product answer)
@dataclass
class PathVerdict:
    path: str
    contents: Dict[str, int]  # file-version CID -> run mask
    """One entry per distinct content this path took across the runs."""

    @property
    def distinct(self) -> int:
        return len(self.contents)

    @property
    def is_robust(self) -> bool:
        return self.distinct == 1

    def summary(self, multi: MultiRun) -> str:
        if self.is_robust:
            return f"robust    {self.path}  (identical in all {len(multi.runs)} runs)"
        parts = " | ".join(
            f"{multi.count(m)}x" for _, m in sorted(self.contents.items(), key=lambda kv: -multi.count(kv[1]))
        )
        return f"DIVERGENT {self.path}  ({self.distinct} distinct contents: {parts})"


def divergence_report(multi: MultiRun, only_final: bool = True) -> List[PathVerdict]:
    """Per path, how many distinct contents the runs produced and which runs agreed.

    ``only_final`` keeps just the last version of each path per run -- intermediate states differ
    constantly (an agent edits a file twice in one run and once in another) and reporting those as
    divergence would drown the signal. What matters is whether the runs *ended* in the same place.
    """
    by_path: Dict[str, Dict[str, int]] = collections.defaultdict(dict)

    derived_from = {t.subject: t.object for t in multi.triples if t.predicate == prov.WAS_DERIVED_FROM}
    superseded = set(derived_from.values())

    for cid, path in multi.paths.items():
        if only_final and cid in superseded:
            continue
        by_path[path][cid] = support(multi, cid)

    return sorted(
        (PathVerdict(path=p, contents=c) for p, c in by_path.items()),
        key=lambda v: (v.is_robust, v.path),
    )


# ------------------------------------------------------------------ derivation-level
def influence_support(multi: MultiRun, backend: str = "auto", max_iterations: int = 1000) -> Dict:
    """For each reachable pair, the set of runs in which that influence holds.

    This is the closure evaluated in the determination semiring: ``times`` intersects because a path
    needs all its edges present in the *same* run, ``plus`` unions because alternative paths each
    contribute their own runs.
    """
    native = resolve(backend)
    if native is not None and hasattr(native, "closure_determination"):
        rows = native.closure_determination(
            as_triples(multi.triples),
            [multi.edge_runs.get((t.subject, t.object), 0) for t in multi.triples],
            len(multi.runs),
            max_iterations,
        )
        return {(a, b): m for a, b, m in rows}

    semiring = determination(lambda _f: 0, multi.all_runs)
    edb = {
        relation: {key: multi.edge_runs.get(key, 0) or multi.edge_runs.get((key[1], key[0]), 0) for key in facts}
        for relation, facts in edb_from_triples(multi.triples).items()
    }
    derived = evaluate(INFLUENCE_RULES, edb, semiring, max_iterations=max_iterations)
    # Drop empty supports. Unioning runs creates *phantom paths*: the union graph can connect two nodes
    # by splicing edges that came from different runs, a route no single run ever took. The semiring
    # detects them exactly -- their support intersects to zero -- and zero is the semiring's absent
    # value, so such a pair is not a result. Keeping them would report influence that never happened.
    return {k: v for k, v in derived.get("influenced", {}).items() if v}


__all__ = [
    "MultiRun",
    "PathVerdict",
    "divergence_report",
    "influence_support",
    "qdepth",
    "robust",
    "support",
    "union_runs",
]
