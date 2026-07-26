"""What content is allowed into the blob store.

This is a gate, not a nicety. Instrumenting a coding agent means every file it reads, every command's
stdout, and every diff is a candidate for ``set_store_all_blobs(True)`` -- which is exactly how a signing
key, a ``.env``, or a customer's source ends up content-addressed and durable on disk.

The policy is deny-by-pattern on paths plus a scrub pass over content. A denied file still becomes a
graph node: its *identity* (path, content CID) is recorded so lineage stays complete, while its bytes are
withheld. Provenance does not require publication.
"""

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import Optional, Pattern, Sequence, Tuple

# Paths whose contents are never stored. Matched against the full resolved path and against the
# basename, so "*.pem" catches a key anywhere without needing a "**/" prefix on every entry.
DEFAULT_DENY_GLOBS: Tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".htpasswd",
    "*credentials*",
    "*secrets*",
    "*.secret",
    "*/.ssh/*",
    "*/.gnupg/*",
    "*/.aws/*",
    "*/.config/gcloud/*",
    "*/.kube/config",
    # the SDK's own generated signing keys -- storing these would be self-defeating
    "*/.eqty_sdk/signers/*",
)

# Content patterns scrubbed even from allowed files. Deliberately short: this is a backstop for the
# obvious cases, not a secret scanner. Anything relying on this list for real safety is misconfigured.
DEFAULT_CONTENT_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"]?([^\s'\"]{8,})"),
    re.compile(r"(?i)\bauthorization\s*:\s*(?:bearer|basic)\s+\S+"),
)

REDACTION_MARKER = b"[eqty-lineage: redacted]"


@dataclass(frozen=True)
class ContentPolicy:
    """Decides whether a file's bytes may be stored, and scrubs what is stored.

    ``allow_globs`` re-permits paths inside an otherwise denied region and takes precedence, mirroring
    the SDK's own ``allowRead``-over-``denyRead`` ordering.
    """

    deny_globs: Sequence[str] = DEFAULT_DENY_GLOBS
    allow_globs: Sequence[str] = ()
    content_patterns: Sequence[Pattern[str]] = DEFAULT_CONTENT_PATTERNS
    scrub_content: bool = True
    # Beyond this, content is identified but not stored. Whole-repo reads and multi-megabyte logs are
    # noise in a lineage graph and dominate the blob store.
    max_content_bytes: int = 1 << 20

    def path_allowed(self, path: str) -> bool:
        """True when this path's *content* may be stored. Identity is always recorded regardless."""
        normalized = str(PurePosixPath(path.replace("\\", "/")))
        name = normalized.rsplit("/", 1)[-1]

        for pattern in self.allow_globs:
            if fnmatch(normalized, pattern) or fnmatch(name, pattern):
                return True
        for pattern in self.deny_globs:
            if fnmatch(normalized, pattern) or fnmatch(name, pattern):
                return False
        return True

    def scrub(self, data: bytes) -> Tuple[bytes, bool]:
        """Replace matched secrets in ``data``. Returns ``(scrubbed, was_modified)``."""
        if not self.scrub_content or not data:
            return data, False

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            # binary content is not pattern-scannable; store as-is and let path rules carry the weight
            return data, False

        modified = False
        for pattern in self.content_patterns:
            text, count = pattern.subn(REDACTION_MARKER.decode(), text)
            modified = modified or bool(count)

        return (text.encode("utf-8"), True) if modified else (data, False)

    def prepare(self, path: Optional[str], content: Optional[bytes]) -> Tuple[Optional[bytes], bool]:
        """Apply the policy to one file version.

        Returns ``(storable_bytes_or_None, redacted)``. ``None`` means "record identity, withhold bytes";
        the caller still registers the asset so the lineage edge survives.
        """
        if content is None:
            return None, False
        if path is not None and not self.path_allowed(path):
            return None, True
        if len(content) > self.max_content_bytes:
            return None, True
        return self.scrub(content)


PERMISSIVE = ContentPolicy(deny_globs=(), scrub_content=False, max_content_bytes=1 << 30)
"""Escape hatch for fixtures and tests. Never appropriate against a real repository."""


__all__ = [
    "DEFAULT_CONTENT_PATTERNS",
    "DEFAULT_DENY_GLOBS",
    "PERMISSIVE",
    "REDACTION_MARKER",
    "ContentPolicy",
]
