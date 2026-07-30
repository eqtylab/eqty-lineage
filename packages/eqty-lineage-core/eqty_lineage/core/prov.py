"""PROV-DM mapping and the edge vocabulary emitted alongside EQTY statements.

The SDK's own model is already a PROV subset, so this is a declaration rather than a translation:

    Dataset/Code/Prompt/Model/... asset          ->  prov:Entity (identified by CID)
    add_computation_statement(inputs, outputs)   ->  prov:Activity + prov:used + prov:wasGeneratedBy
    Signer / DID                                 ->  prov:Agent + prov:wasAttributedTo
    Context / Context.with_parent()              ->  prov:Bundle (nested)

Declaring it buys interop with the existing provenance tooling and an RDF serialization for free.
Three edges have no standard equivalent and are the reason this vocabulary exists at all:

``wasCompactedFrom``
    A *lossy* derivation. No standard models "derived from a summary of X where most of X was dropped".

``authorizedBy``
    What was permitted, not what happened. PROV, OpenLineage and in-toto all model the latter only.

``observed`` / ``inferred``
    Carried as a flag on every statement rather than as an edge. A filesystem change attributed to a
    Bash command by diffing snapshots is attribution, not observation.
"""

from typing import Final

# ------------------------------------------------------------------ PROV-DM core
USED: Final = "prov:used"
WAS_GENERATED_BY: Final = "prov:wasGeneratedBy"
WAS_DERIVED_FROM: Final = "prov:wasDerivedFrom"
WAS_ATTRIBUTED_TO: Final = "prov:wasAttributedTo"
WAS_ASSOCIATED_WITH: Final = "prov:wasAssociatedWith"
# a file version superseded by a later edit; PROV has this natively, which is why versions are entities
WAS_INVALIDATED_BY: Final = "prov:wasInvalidatedBy"

# ------------------------------------------------------------------ agent-execution relations
# From the agent-provenance literature: base Use/Generate/Derive plus relations that separate semantic
# grounding from procedural dependency. Only the ones a *coding* agent actually exercises are defined
# here -- an unused predicate in a signed vocabulary is a liability, not an option.
TRIGGERED: Final = "eqty:triggered"
"""The model's tool_use block caused this tool execution. Distinguishes "the model asked" from "it ran"."""

DEPENDS_ON: Final = "eqty:dependsOn"
"""Procedural dependency: the computation read this to produce its output."""

SUPPORTS: Final = "eqty:supports"
"""Semantic grounding: this evidence backs a claim the agent made (e.g. test output backing "tests pass")."""

# ------------------------------------------------------------------ edges the standards lack
WAS_COMPACTED_FROM: Final = "eqty:wasCompactedFrom"
AUTHORIZED_BY: Final = "eqty:authorizedBy"
RAN_AS: Final = "eqty:ranAs"
"""Links a computation to the Agent asset (CLI + version + model) that performed it."""

HAS_PATH: Final = "eqty:hasPath"
"""Names the filesystem path a file-version entity is a state of.

An annotation, not a lineage edge -- it describes a node rather than connecting two. Emitted because
policy questions are asked about paths ("did anything write outside the permitted set?") while the
graph is addressed by CID, and a query layer cannot bridge the two from metadata alone.
"""

ASSET_TYPE: Final = "eqty:assetType"
"""Names the SDK asset type of an entity (Code, Reasoning, Guardrail, ...).

An annotation, not a lineage edge. Emitted because the graph is addressed by CID and CIDs are opaque:
without it, neither a query ("what Code is downstream of this?") nor a differential comparison between
two capture paths can say *what kind of thing* a node is.
"""

LABEL: Final = "eqty:label"
"""Human-readable name of an activity (the tool or computation that ran).

An annotation, not a lineage edge. The name is already on the Metadata statement, but the triple
sidecar is consumed on its own -- by the query engine and by any renderer -- and an activity with no
name is an opaque hash in every view built from it.
"""

# ------------------------------------------------------------------ metadata keys
# Attached via Metadata(...).create_statement so they survive into the manifest and the graph explorer.
# Dashes, not underscores: the explorer camel-cases keys and renders "_" as a space.
K_PROV_TYPE: Final = "prov-type"
K_EDGE_KIND: Final = "edge-kind"
K_OBSERVED: Final = "observed"
K_FRAMEWORK: Final = "framework"
K_AGENT: Final = "agent"
K_SESSION: Final = "session-id"
K_PERMISSION_MODE: Final = "permission-mode"
K_EFFORT: Final = "effort"
K_TOOL_USE_ID: Final = "tool-use-id"
K_USER_MODIFIED: Final = "user-modified"
K_FILE_PATH: Final = "file-path"
K_FILE_VERSION: Final = "file-version"
K_OPAQUE: Final = "opaque"
K_REDACTED: Final = "redacted"
K_DELETED: Final = "deleted"
"""Marks a tombstone: the path was removed, as distinct from a version whose bytes were not recovered."""

# ------------------------------------------------------------------ computation kinds
KIND_TOOL: Final = "tool"
KIND_MODEL: Final = "model_call"
KIND_SESSION: Final = "session"
KIND_SUBAGENT: Final = "subagent"
KIND_COMPACTION: Final = "compaction"
KIND_FILE_VERSION: Final = "file_version"


__all__ = [
    "ASSET_TYPE",
    "AUTHORIZED_BY",
    "DEPENDS_ON",
    "HAS_PATH",
    "LABEL",
    "K_AGENT",
    "K_DELETED",
    "K_EDGE_KIND",
    "K_EFFORT",
    "K_FILE_PATH",
    "K_FILE_VERSION",
    "K_FRAMEWORK",
    "K_OBSERVED",
    "K_OPAQUE",
    "K_PERMISSION_MODE",
    "K_PROV_TYPE",
    "K_REDACTED",
    "K_SESSION",
    "K_TOOL_USE_ID",
    "K_USER_MODIFIED",
    "KIND_COMPACTION",
    "KIND_FILE_VERSION",
    "KIND_MODEL",
    "KIND_SESSION",
    "KIND_SUBAGENT",
    "KIND_TOOL",
    "RAN_AS",
    "SUPPORTS",
    "TRIGGERED",
    "USED",
    "WAS_ASSOCIATED_WITH",
    "WAS_ATTRIBUTED_TO",
    "WAS_COMPACTED_FROM",
    "WAS_DERIVED_FROM",
    "WAS_GENERATED_BY",
    "WAS_INVALIDATED_BY",
]
