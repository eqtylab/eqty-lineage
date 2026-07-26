"""Conformance: the offline and live capture paths must produce the same graph, modulo declared gaps.

Not a unit test -- a checker that can be run against any real session, which is what makes "both paths
produce the same graph" a claim you can audit rather than assert.

Method. Ingest a session transcript. Separately replay it as hook payloads through the same recorder the
daemon uses. Compare canonicalized triples.

Comparison is plain set difference, not RDF canonicalization: RDFC-1.0 exists to label blank nodes and
this graph has none, since every entity is content-addressed. Activities, however, are *not* stable --
a statement CID covers a signed credential carrying ``validFrom``, so the same computation recorded
twice gets two different activity CIDs. They are therefore relabeled by their ``(inputs, outputs)``
signature before comparison.

The assertion is not that the graphs are identical. It is that every divergence falls into a declared
bucket, that each declared bucket is non-empty, and that nothing else divides them. A bucket that
silently empties means the capture path moved underneath its declaration.

**Known limit.** Both paths share ``eqty-lineage-core``, and tool-result parsing lives there
deliberately so they cannot drift. This therefore exercises the two *adapters*, not the recorder;
correlated faults in shared code are invisible to it, which is the classic N-version result. Replay also
cannot exercise events a transcript has no record of -- ``InstructionsLoaded``, ``FileChanged``,
``PermissionRequest`` -- so those asymmetries are declared, not demonstrated.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from eqty_lineage.core import LineageRecorder, TripleSink, canonical_triples, prov
from eqty_lineage.core.canonical import activity_signatures, signature_label

from .dialects import to_events
from .replay import replay_payloads

logger = logging.getLogger("eqty.lineage.hooks")

# Asset types only the transcript can produce, and why.
TRANSCRIPT_ONLY_TYPES: Dict[str, str] = {
    "Prompt": "hooks never carry the model's messages",
    "Model": "hooks never carry the model's messages",
    "Reasoning": "hooks never carry model output or token usage",
}
# Asset types only the live path can produce. Replay cannot demonstrate these, so they are declared.
HOOKS_ONLY_TYPES: Dict[str, str] = {
    "System_Prompt": "InstructionsLoaded is hook-only; transcripts have no record of loaded instructions",
    "Guardrail": "permission decisions are hook-only",
}

_MODEL_ARTIFACTS = frozenset({"Reasoning", "Prompt", "Model"})


@dataclass
class Report:
    session: str
    offline_events: int = 0
    live_events: int = 0
    offline_triples: int = 0
    live_triples: int = 0
    observed_versions: int = 0
    inferred_versions: int = 0
    shared_activities: int = 0
    live_activities: int = 0
    offline_only: int = 0
    live_only: int = 0
    buckets: Dict[str, int] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        status = "OK  " if self.ok else "FAIL"
        return (
            f"{status} {self.session[:8]}  "
            f"files {self.observed_versions}+{self.inferred_versions}i  "
            f"activities {self.shared_activities}/{self.live_activities}  "
            f"diverge {self.offline_only}/{self.live_only}"
            + ("" if self.ok else "  :: " + "; ".join(self.problems[:3]))
        )


def _record(events, framework: str, label: str) -> Tuple[TripleSink, LineageRecorder]:
    from eqty_sdk import Context
    from eqty_sdk.context import graph_context

    sink = TripleSink()
    recorder = LineageRecorder(triples=sink, framework=framework)
    with graph_context(Context.new(f"{framework} {label}")):
        recorder.handle_all(events)
    return sink, recorder


def compare(transcript: Path) -> Report:
    """Run both capture paths over one session and classify every divergence."""
    from eqty_lineage.transcript.claude_code import ClaudeCodeTranscript

    transcript = Path(transcript)
    report = Report(session=transcript.stem)

    offline_events = list(ClaudeCodeTranscript(transcript).events())
    live_events = [e for payload in replay_payloads(transcript) for e in to_events(payload)]
    report.offline_events, report.live_events = len(offline_events), len(live_events)

    offline, offline_rec = _record(offline_events, "claude-code", transcript.stem[:8])
    live, live_rec = _record(live_events, "agent-hooks", transcript.stem[:8])
    report.offline_triples, report.live_triples = len(offline), len(live)

    kinds_o = {t.subject: t.object for t in offline if t.predicate == prov.ASSET_TYPE}
    kinds_l = {t.subject: t.object for t in live if t.predicate == prov.ASSET_TYPE}
    paths_o = {t.subject: t.object for t in offline if t.predicate == prov.HAS_PATH}
    paths_l = {t.subject: t.object for t in live if t.predicate == prov.HAS_PATH}

    inferred: Set[str] = {
        str(v.asset_cid)
        for versions in offline_rec.file_versions.values()
        for v in versions
        if not v.observed
    }
    cids_o, cids_l = set(paths_o), set(paths_l)
    report.observed_versions, report.inferred_versions = len(cids_l), len(inferred)

    # --- file lineage: the substantive claim both paths make about the world
    if cids_o - cids_l != (inferred & cids_o):
        report.problems.append("file-version difference is not exactly the snapshot-inferred set")
    if cids_l - cids_o:
        report.problems.append(f"{len(cids_l - cids_o)} file version(s) only the live path saw")
    observed_o = {(paths_o[c], kinds_o.get(c)) for c in cids_o - inferred}
    if observed_o != {(paths_l[c], kinds_l.get(c)) for c in cids_l}:
        report.problems.append("observed file versions disagree")

    derived_o = {(t.subject, t.object) for t in offline if t.predicate == prov.WAS_DERIVED_FROM}
    derived_l = {(t.subject, t.object) for t in live if t.predicate == prov.WAS_DERIVED_FROM}
    if derived_o != derived_l:
        report.problems.append("derivation edges disagree")

    # --- activities, compared by signature because their CIDs are time-dependent
    sigs_o = activity_signatures(offline)
    sigs_l = activity_signatures(live)
    labels_o = {signature_label(v) for v in sigs_o.values()}
    labels_l = {signature_label(v) for v in sigs_l.values()}
    report.shared_activities, report.live_activities = len(labels_o & labels_l), len(labels_l)
    if not labels_l <= labels_o:
        report.problems.append(f"{len(labels_l - labels_o)} live activity signature(s) absent offline")

    offline_only_acts = labels_o - labels_l
    model_acts, inferred_acts, unexplained = [], [], []
    for sig in sigs_o.values():
        if signature_label(sig) not in offline_only_acts:
            continue
        outputs = sig[1]
        if any(kinds_o.get(c) in _MODEL_ARTIFACTS for c in outputs):
            model_acts.append(sig)
        elif outputs and outputs <= inferred:
            inferred_acts.append(sig)
        else:
            unexplained.append(sig)
    if unexplained:
        report.problems.append(f"{len(unexplained)} offline-only activity(ies) with no declared cause")
    if not model_acts:
        report.problems.append("declared model-call divergence produced nothing (declaration is stale)")

    # --- every remaining divergent triple must be attributable
    keys_o = {(t.subject, t.predicate, t.object) for t in canonical_triples(offline)}
    keys_l = {(t.subject, t.predicate, t.object) for t in canonical_triples(live)}
    only_o, only_l = keys_o - keys_l, keys_l - keys_o
    report.offline_only, report.live_only = len(only_o), len(only_l)

    buckets: Dict[str, int] = {}

    def entities_of(triples, path_only_acts) -> Set[str]:
        """Entities that only ever participate in path-only activities.

        Their annotation triples (assetType, hasPath) name neither the activity nor another entity, so
        without this they escape the edge-level bucketing and look like undeclared divergence. The
        subagent spec Dataset is the case that surfaced it: replay cannot synthesize SubagentStart /
        SubagentStop from a transcript, so the whole subagent subgraph is replay-only -- a limitation of
        this harness, not an asymmetry between the real capture paths.
        """
        touched: Dict[str, Set[str]] = {}
        for t in canonical_triples(triples):
            if t.predicate == prov.USED:
                touched.setdefault(t.object, set()).add(t.subject)
            elif t.predicate == prov.WAS_GENERATED_BY:
                touched.setdefault(t.subject, set()).add(t.object)
        return {e for e, acts in touched.items() if acts and acts <= path_only_acts}

    offline_only_entities = entities_of(offline, offline_only_acts)
    live_only_entities = entities_of(live, labels_l - labels_o)

    def bucket(triples, kinds, path_only_acts, path_only_entities, declared_types):
        undeclared = set()
        for subject, _predicate, obj in triples:
            if subject in path_only_acts or obj in path_only_acts:
                key = "(edge of a path-only activity)"
            elif subject in path_only_entities or obj in path_only_entities:
                key = "(entity of a path-only activity)"
            elif subject in inferred or obj in inferred:
                key = "(snapshot-inferred file version)"
            else:
                key = kinds.get(subject) or kinds.get(obj) or "unnamed"
                if key not in declared_types:
                    undeclared.add(key)
            buckets[key] = buckets.get(key, 0) + 1
        return undeclared

    undeclared_o = bucket(only_o, kinds_o, offline_only_acts, offline_only_entities, TRANSCRIPT_ONLY_TYPES)
    undeclared_l = bucket(only_l, kinds_l, labels_l - labels_o, live_only_entities, HOOKS_ONLY_TYPES)
    report.buckets = buckets
    if undeclared_o:
        report.problems.append(f"undeclared offline-only entity types: {sorted(undeclared_o)}")
    if undeclared_l:
        report.problems.append(f"undeclared live-only entity types: {sorted(undeclared_l)}")

    if offline_rec.stats.get("ToolCallEnded") != live_rec.stats.get("ToolCallEnded"):
        report.problems.append(
            f"tool-call counts differ: offline {offline_rec.stats.get('ToolCallEnded')} "
            f"vs live {live_rec.stats.get('ToolCallEnded')}"
        )

    return report


__all__ = ["HOOKS_ONLY_TYPES", "TRANSCRIPT_ONLY_TYPES", "Report", "compare"]
