"""Projecting a manifest down to the part a human can actually read.

A full coding session produces one to two orders of magnitude more graph than a LangGraph run: measured
on real sessions, 136 to 1,995 assets and up to 1,359 activities, of which ``Dataset`` and ``Reasoning``
are around 88%. Every model response and every tool argument blob is a node, so the file provenance --
the substantive claim -- is buried under conversation.

The fix is a projection, not a different viewer: keep the file lineage and the tool calls that produced
it, drop the model calls.

**A projection is a subset of the signed manifest, never a re-signing.** Every kept statement keeps its
original CID and its original ``CredentialRegistration``, so the projection verifies exactly as the full
manifest does. Re-recording the selected statements into a fresh context would mint new statement CIDs
(they carry ``validFrom``) and produce a *different* attestation that merely resembled the original.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from . import prov

logger = logging.getLogger("eqty.lineage.core")

# Asset types that constitute file lineage. These are the nodes worth looking at.
FILE_TYPES = frozenset({"Code", "Document", "Configuration", "Dataset"})

# Activities producing these are conversation, not provenance over artifacts.
MODEL_ARTIFACTS = frozenset({"Reasoning", "Prompt", "Model"})


@dataclass
class ProjectionStats:
    statements_in: int = 0
    statements_out: int = 0
    blobs_in: int = 0
    blobs_out: int = 0
    entities_kept: int = 0
    activities_kept: int = 0

    def summary(self) -> str:
        pct = 100 * self.statements_out / self.statements_in if self.statements_in else 0
        return (
            f"{self.statements_out}/{self.statements_in} statements ({pct:.0f}%), "
            f"{self.blobs_out}/{self.blobs_in} blobs, "
            f"{self.entities_kept} entities across {self.activities_kept} activities"
        )


def select_file_lineage(triples: Iterable, include_tools: bool = True) -> Set[str]:
    """CIDs to keep: file versions, the activities that touched them, and what those reference.

    With ``include_tools``, an activity's other inputs and outputs come along -- the ``Tool`` asset
    carrying the command, its arguments, and its result. That is deliberate: "which command changed this
    file" is the question the projection exists to answer, and dropping them would leave an edit
    attributed to an anonymous activity.

    Model-call and subagent activities are excluded entirely unless they also touched a file.
    """
    triples = list(triples)

    kinds: Dict[str, str] = {}
    paths: Set[str] = set()
    for t in triples:
        if t.predicate == prov.ASSET_TYPE:
            kinds[t.subject] = t.object
        elif t.predicate == prov.HAS_PATH:
            paths.add(t.subject)

    # A file version is an entity that has a path. Type alone is not enough: tool arguments are also
    # Datasets, and a path is what distinguishes a state of a file from an arbitrary blob.
    file_versions = {c for c in paths if kinds.get(c) in FILE_TYPES or c in paths}

    inputs: Dict[str, Set[str]] = {}
    outputs: Dict[str, Set[str]] = {}
    for t in triples:
        if t.predicate == prov.USED:
            inputs.setdefault(t.subject, set()).add(t.object)
        elif t.predicate == prov.WAS_GENERATED_BY:
            outputs.setdefault(t.object, set()).add(t.subject)

    keep: Set[str] = set(file_versions)
    activities: Set[str] = set()

    for activity in set(inputs) | set(outputs):
        touched = inputs.get(activity, set()) | outputs.get(activity, set())
        if not (touched & file_versions):
            continue
        # An activity whose only outputs are model artifacts is conversation that happened to read a
        # file; keep the file edge, not the monologue.
        produced = outputs.get(activity, set())
        if produced and all(kinds.get(c) in MODEL_ARTIFACTS for c in produced):
            continue
        activities.add(activity)
        keep.add(activity)
        if include_tools:
            keep |= touched

    # The agent and any guardrail that authorized a kept activity: small, and they carry the
    # "what ran, under what permission" context that makes the projection attestable on its own.
    for t in triples:
        if t.predicate in (prov.RAN_AS, prov.AUTHORIZED_BY) and t.subject in activities:
            keep.add(t.object)
        # Compaction is a lossy edge; a projection that dropped it would imply a completeness it lacks.
        elif t.predicate == prov.WAS_COMPACTED_FROM:
            keep.add(t.subject)
            keep.add(t.object)

    return keep


_CID_PREFIX = "urn:cid:"


def _bare(cid: str) -> str:
    """Blob keys are bare CIDs; statement fields carry the ``urn:cid:`` prefix. Compare on the bare form.

    Getting this wrong is silent and total: the prefixed and bare sets simply never intersect, so blob
    pruning keeps nothing and the projection renders with no content whatsoever.
    """
    return cid[len(_CID_PREFIX) :] if cid.startswith(_CID_PREFIX) else cid


def _cids_in(value: Any, found: Set[str]) -> None:
    if isinstance(value, str):
        if value.startswith(_CID_PREFIX) or value.startswith("baf") or value.startswith("bag"):
            found.add(_bare(value))
    elif isinstance(value, dict):
        for item in value.values():
            _cids_in(item, found)
    elif isinstance(value, list):
        for item in value:
            _cids_in(item, found)


def project_manifest(source: Path, dest: Path, keep: Set[str]) -> ProjectionStats:
    """Write a manifest containing only statements about ``keep``, with signatures intact."""
    manifest = json.loads(Path(source).read_text(encoding="utf-8"))
    statements: Dict[str, Any] = manifest.get("statements", {})
    blobs: Dict[str, Any] = manifest.get("blobs", {})

    stats = ProjectionStats(statements_in=len(statements), blobs_in=len(blobs))

    kept: Dict[str, Any] = {}

    # Pass 1: registrations that are *about* something we are keeping.
    for cid, statement in statements.items():
        if not isinstance(statement, dict):
            continue
        kind = statement.get("@type")
        if kind == "DataRegistration":
            if statement.get("data") in keep:
                kept[cid] = statement
        elif kind == "ComputationRegistration":
            if cid in keep:
                kept[cid] = statement
        elif kind == "MetadataRegistration":
            if statement.get("subject") in keep:
                kept[cid] = statement

    # Pass 2: metadata about a kept statement (not just about a kept entity).
    for cid, statement in statements.items():
        if isinstance(statement, dict) and statement.get("@type") == "MetadataRegistration":
            if statement.get("subject") in kept:
                kept[cid] = statement

    # Pass 3: the credentials that sign everything kept. Without these the projection is a plausible
    # document rather than an attestation, so this pass is the point of doing it as a subset at all.
    for cid, statement in statements.items():
        if not isinstance(statement, dict) or statement.get("@type") != "CredentialRegistration":
            continue
        subject = ((statement.get("credential") or {}).get("credentialSubject") or {}).get("id")
        if subject in kept:
            kept[cid] = statement

    referenced: Set[str] = set()
    for statement in kept.values():
        _cids_in(statement, referenced)

    keep_bare = {_bare(c) for c in keep}
    kept_blobs = {cid: data for cid, data in blobs.items() if _bare(cid) in referenced or _bare(cid) in keep_bare}

    stats.statements_out = len(kept)
    stats.blobs_out = len(kept_blobs)
    stats.entities_kept = sum(1 for s in kept.values() if s.get("@type") == "DataRegistration")
    stats.activities_kept = sum(1 for s in kept.values() if s.get("@type") == "ComputationRegistration")

    out = {
        "version": manifest.get("version"),
        "contexts": manifest.get("contexts", {}),
        "statements": kept,
        "blobs": kept_blobs,
    }
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out), encoding="utf-8")
    return stats


__all__ = ["FILE_TYPES", "MODEL_ARTIFACTS", "ProjectionStats", "project_manifest", "select_file_lineage"]
