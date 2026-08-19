"""What the record saw, and what it did not.

A manifest attests artifacts and derivations. It does not, on its own, say how much of the session it
managed to observe -- so a complete record and one that lost half its file contents are equally
signed and look identical to a reader. Everything downstream inherits that ambiguity: an audit over
a graph that knows 40% of file content is not an audit, and a "no write to X occurred" claim is only
as strong as the fraction of writes the recorder actually saw.

Measured over a real corpus, the gap is not small:

    content known for 38.8% of file versions before session chaining, 79.3% after
    1,220 paths changed with nothing to attribute them to
    97% of context dropped at compaction boundaries
    990 of 2,236 transcripts invisible because subagents write their own files

So the coverage statement is not a diagnostic bolted on the side. It is the part of the manifest that
tells a reader how much the rest of it is worth, and it is signed for the same reason the graph is:
an unsigned completeness claim can be edited by anyone who dislikes the number.

The design rule throughout is that an absence must be *counted*, never inferred from silence. A file
version with no content still increments ``content_unknown``; a subagent whose transcript was missing
increments ``subagents_opaque``. Reading a zero should mean "none occurred", never "we did not look".
"""

from dataclasses import asdict, dataclass
from typing import Any

# How a file version's bytes came to be known. Ordered from strongest evidence to weakest, which is
# also the order a reader should discount them in.
BASIS_STATED = "stated"
"""The tool result stated the post-image outright."""

BASIS_REPLAYED_RESULT = "replayed-from-result"
"""Replayed against a pre-image the same result carried. Exact, and recheckable by anyone."""

BASIS_REPLAYED_SESSION = "replayed-from-session"
"""Replayed against content the session established earlier. Exact given that content was current,
and self-validating: the replacement had to apply. Recheckable from the manifest alone."""

BASIS_BACKUP_STORE = "backup-store"
"""Read from the agent's local file-history backups. Real bytes, but evidence about one machine's
disk rather than about the session, and *not* recheckable by a reader who lacks that store."""

BASIS_UNKNOWN = "unknown"
"""Identity recorded, bytes never established by any route."""


@dataclass
class Coverage:
    """Counters describing the completeness of one session's record."""

    # ---------------------------------------------------------------- files
    file_versions: int = 0
    content_stated: int = 0
    content_replayed_result: int = 0
    content_replayed_session: int = 0
    content_from_backup: int = 0
    content_unknown: int = 0
    #: Bytes withheld by the redaction policy. Identity is recorded; publication was declined. This
    #: is a deliberate absence and should not be read as a capture failure.
    content_redacted: int = 0
    #: Versions the capture path did not witness directly -- a snapshot delta attributing a change to
    #: no particular command. The live path's watcher exists to shrink this number.
    versions_inferred: int = 0
    tombstones: int = 0

    # ---------------------------------------------------------------- context
    compactions: int = 0
    tokens_before_compaction: int = 0
    tokens_after_compaction: int = 0

    # ---------------------------------------------------------------- delegation
    subagents: int = 0
    #: Subagents whose own tool calls were not visible. A non-zero value here means the graph is
    #: missing work that definitely happened.
    subagents_opaque: int = 0

    # ---------------------------------------------------------------- reasoning
    #: Model calls recorded. Every one of them reasoned; none of them disclosed it. All three
    #: surfaces -- transcript, Codex rollout, raw API body -- carry a signature over withheld
    #: content, so reasoning is an attested absence rather than a gap in this recorder.
    reasoning_attested: int = 0
    reasoning_recovered: int = 0

    # ---------------------------------------------------------------- activity
    tool_attempts: int = 0
    #: Reads recovered by recognising a tool result as a file version's exact bytes rather than from a
    #: stated path. Counted apart from observed reads because it is an inference, and because it is
    #: what gives a shell-driven agent any lineage depth at all -- a reader weighing that depth should
    #: see how much of it rests on byte equality.
    reads_by_content_match: int = 0

    tool_calls: int = 0
    tool_errors: int = 0
    tool_denied: int = 0
    permission_decisions: int = 0

    # ------------------------------------------------------------------ derived
    @property
    def content_known(self) -> int:
        return (
            self.content_stated
            + self.content_replayed_result
            + self.content_replayed_session
            + self.content_from_backup
        )

    @property
    def content_known_rate(self) -> float:
        """Fraction of file versions whose bytes are known.

        A record with no file versions is vacuously complete rather than 0% complete -- the
        distinction matters, because a reader comparing rates across sessions should not see a
        session that touched no files as the worst one.
        """
        if not self.file_versions:
            return 1.0
        return self.content_known / self.file_versions

    @property
    def tokens_dropped(self) -> int:
        return max(0, self.tokens_before_compaction - self.tokens_after_compaction)

    @property
    def context_retained_rate(self) -> float:
        """Fraction of context surviving compaction. Low values mean any claim about the agent's
        later behaviour cannot appeal to what it was told at the start -- the instruction is no
        longer in the window, and only the graph still holds it."""
        if not self.tokens_before_compaction:
            return 1.0
        return self.tokens_after_compaction / self.tokens_before_compaction

    @property
    def is_complete(self) -> bool:
        """True only when nothing is missing: every byte known, nothing inferred, no opaque
        subagent, no compaction loss. Deliberately strict -- this is the flag that says a reader
        needs no caveats, and it should be rare."""
        return (
            self.content_unknown == 0
            and self.versions_inferred == 0
            and self.subagents_opaque == 0
            and self.tokens_dropped == 0
        )

    def record_basis(self, basis: str) -> None:
        """Count one file version by how its bytes were established."""
        self.file_versions += 1
        if basis == BASIS_STATED:
            self.content_stated += 1
        elif basis == BASIS_REPLAYED_RESULT:
            self.content_replayed_result += 1
        elif basis == BASIS_REPLAYED_SESSION:
            self.content_replayed_session += 1
        elif basis == BASIS_BACKUP_STORE:
            self.content_from_backup += 1
        else:
            self.content_unknown += 1

    def as_payload(self) -> dict[str, Any]:
        """The signed form.

        Derived rates are included rather than left for the reader to compute: the point is that a
        verifier and a human see the same summary, and a rate recomputed from counters by hand is a
        rate someone can get wrong.
        """
        payload = asdict(self)
        payload.update(
            {
                "content_known": self.content_known,
                "content_known_rate": round(self.content_known_rate, 4),
                "tokens_dropped": self.tokens_dropped,
                "context_retained_rate": round(self.context_retained_rate, 4),
                "complete": self.is_complete,
            }
        )
        return payload

    def summary(self) -> str:
        known = f"{self.content_known}/{self.file_versions} file versions with known content"
        parts = [f"{known} ({100 * self.content_known_rate:.0f}%)"]
        if self.versions_inferred:
            parts.append(f"{self.versions_inferred} inferred")
        if self.subagents_opaque:
            parts.append(f"{self.subagents_opaque} opaque subagent(s)")
        if self.tokens_dropped:
            parts.append(f"{100 * (1 - self.context_retained_rate):.0f}% of context dropped")
        if self.content_redacted:
            parts.append(f"{self.content_redacted} redacted")
        return "; ".join(parts)


__all__ = [
    "BASIS_BACKUP_STORE",
    "BASIS_REPLAYED_RESULT",
    "BASIS_REPLAYED_SESSION",
    "BASIS_STATED",
    "BASIS_UNKNOWN",
    "Coverage",
]
