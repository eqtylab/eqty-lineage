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

CONTENT_BASIS: Final = "eqty:contentBasis"
"""How a file version's bytes were established -- stated, replayed, from a backup store, or unknown.

An annotation, and in the triple set rather than only in metadata for the same reason `hasPath` and
`assetType` are: the sidecar is consumed on its own, and a verifier asked to confirm the record's
coverage claim cannot do it from statements it has to parse the SDK to read.
"""

COVERAGE_CLAIM: Final = "eqty:coverageClaim"
"""The signed completeness counters, carried as a fact so they can be recounted against the graph.

A coverage claim nothing checks is a number a reader has to take on faith. Putting it in the fact set
lets an independent verifier recount the graph and compare, which is the difference between a signed
assertion and a checkable one.
"""

READ_BASIS: Final = "eqty:readBasis"
"""How a read edge was established, when the tool never named the file.

Only ``content-match``: a tool result whose bytes are exactly a file version this session recorded.
Codex reads through the shell and its ``tool_response`` is a bare string with no path, so
``file_events_from_result`` returns nothing for it -- without this a Codex session has *zero*
activities consuming a file version and the graph can only ever be three columns deep.

Annotated rather than folded in silently, because it infers causation from identity: byte equality
proves the tool emitted those bytes, not that it read that file. A command printing a string that
happens to equal a file's contents earns a spurious edge. The alternative was parsing shell commands
for read patterns, which fabricates a path from a string and breaks on `sed -n '1,80p' "$f"`.

This is only safe because a tool result is now its own entity (wrapped with its call id). While a
result and a file version shared a CID, adding this edge made the activity both produce and consume
the same node -- a 2-cycle in the flow relation, which is what makes reachability meaningless and
non-absorptive semirings diverge.
"""

IN_SERVICE_OF: Final = "eqty:inServiceOf"
"""The user instruction an activity was working toward.

A **stated scope, not observed causation.** The rule is "the most recent prompt at the time the
activity ran", and it is wrong whenever a later turn is genuinely still serving an earlier
instruction. It is recorded anyway because the alternative was worse: every activity used to take the
whole accumulated list of prompts as input, so on a real 367-prompt session the median file write was
attributed to 142 separate intents at once. Naming one intent that is sometimes the wrong one is a
weaker claim than naming 142 that are mostly wrong, and unlike the latter it is falsifiable.

Read it as "this is the instruction that was on the table", and check ``eqty:intentAge`` before
trusting it -- an activity serving an instruction from three compactions ago is exactly the case where
the scoping rule is least likely to hold.
"""

INTENT_AGE: Final = "eqty:intentAge"
"""How many compaction boundaries stand between an activity and the instruction it serves.

Zero means the instruction was still in the agent's context window when the activity ran. Above zero
means it was not: compaction drops ~97% of context, so the model was acting on a summary of the
instruction rather than the instruction. The graph still holds the link the window no longer does,
which is the whole reason to record the distance rather than just the edge.
"""

DETERMINATION_SUPPORT: Final = "eqty:determinationSupport"
"""Which runs of a replication produced this exact content.

The determination semiring's support set, carried as a fact. A signature over a file version proves
you received that artifact; it says nothing about whether it was the only outcome the task could have
produced. This is the annotation that makes "every run agreed" checkable rather than assumed, and it
is meaningless on a single run -- support is a statement about a set of resolutions.
"""

CONTESTED_REGION: Final = "eqty:contestedRegion"
"""A line range that some runs produced and others did not.

Where the specification ran out and the agent chose. Recorded as a range rather than a count because
the count is confounded -- a four-line fix in a small file scores high without anything being
underdetermined -- while the range is what a reviewer acts on. See the pilot in
`experiments/variance/pilot/RESULTS.md` for the measurement behind that distinction.
"""

BEHAVIOURAL_CLASS: Final = "eqty:behaviouralClass"
"""How many distinct observable behaviours the runs exhibited, and by what means that was decided.

Stated, never derived: establishing it requires executing the artifact against inputs the graph does
not contain, so it enters as an assertion with its method named. Recorded because it is the signal
that discriminated specification completeness in the pilot (1, 1, 2, 4 across four tasks) while
byte-level divergence stayed flat (3, 3, 4, 4) and separated nothing.
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
K_RECONSTRUCTED: Final = "content-basis"
"""How a file version's bytes were established: stated by the tool, or replayed against a pre-image.

Recorded because the two are not the same claim. A stated post-image is what the tool reported; a
replayed one is exact inference from a literal replacement, and inference the graph does not label
is inference a reader cannot discount.
"""

K_DELETED: Final = "deleted"
"""Marks a tombstone: the path was removed, as distinct from a version whose bytes were not recovered."""

# ------------------------------------------------------------------ computation kinds
KIND_TOOL: Final = "tool"
KIND_POLICY: Final = "policy"
KIND_MODEL: Final = "model_call"
KIND_SESSION: Final = "session"
KIND_TERMINAL: Final = "session_terminal"
KIND_SUBAGENT: Final = "subagent"
KIND_COMPACTION: Final = "compaction"
KIND_FILE_VERSION: Final = "file_version"
KIND_DETERMINATION: Final = "determination"
"""One path's outcome across N runs: which contents occurred, and under how many resolutions.

The first activity in this vocabulary whose inputs come from more than one session. Its inputs are the
distinct contents the runs produced, so content-addressing does the work -- a path every run agreed on
has exactly one input and the node is degenerate, while a divergent path fans in from as many nodes as
there were outcomes. The shape carries the finding without anyone having to read the payload.
"""

KIND_REPLICATION: Final = "replication"
"""The whole replication: N runs of one task, and what survived across them.

Signed for the same reason the coverage claim is. An unsigned determination verdict is a number
anyone who dislikes it can edit, and this one is load-bearing -- it is the claim that an artifact was
not merely produced but was the outcome the task actually forced.
"""

KIND_COVERAGE: Final = "coverage"
"""A statement about the record itself: how much of the session it managed to observe.

In the graph rather than beside it, because a completeness claim that travels separately from the
manifest is a completeness claim that gets dropped when the manifest is quoted."""


__all__ = [
    "ASSET_TYPE",
    "AUTHORIZED_BY",
    "CONTENT_BASIS",
    "COVERAGE_CLAIM",
    "DEPENDS_ON",
    "HAS_PATH",
    "KIND_COMPACTION",
    "KIND_COVERAGE",
    "KIND_FILE_VERSION",
    "KIND_MODEL",
    "KIND_POLICY",
    "KIND_SESSION",
    "KIND_SUBAGENT",
    "KIND_TERMINAL",
    "KIND_TOOL",
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
    "K_RECONSTRUCTED",
    "K_REDACTED",
    "K_SESSION",
    "K_TOOL_USE_ID",
    "K_USER_MODIFIED",
    "LABEL",
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
