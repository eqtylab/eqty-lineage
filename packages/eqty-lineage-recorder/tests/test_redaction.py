"""What may be stored, and what must only be identified.

The distinction that runs through this module: **identity is always recorded, content is not.** A denied
path still gets a node and still gets its lineage edges -- withholding the bytes must never withhold the
fact that the file participated, or the graph would quietly lie about what the agent touched.
"""

from eqty_lineage.recorder import PERMISSIVE, ContentPolicy
from eqty_lineage.recorder.redaction import REDACTION_MARKER


class TestPathRules:
    def test_the_default_policy_denies_obvious_secret_paths(self):
        p = ContentPolicy()
        assert not p.path_allowed("/repo/.env")
        assert not p.path_allowed("/home/u/.ssh/id_rsa")
        assert not p.path_allowed("/repo/key.pem")

    def test_ordinary_source_is_allowed(self):
        assert ContentPolicy().path_allowed("/repo/util.py")

    def test_allow_globs_re_permit_inside_a_denied_region(self):
        # Mirrors the SDK's allowRead-over-denyRead ordering.
        p = ContentPolicy(deny_globs=("*.pem",), allow_globs=("/repo/fixtures/*.pem",))
        assert not p.path_allowed("/repo/secret.pem")
        assert p.path_allowed("/repo/fixtures/test.pem")

    def test_matching_is_on_both_the_full_path_and_the_basename(self):
        p = ContentPolicy(deny_globs=(".env",))
        assert not p.path_allowed("/deep/nested/.env")

    def test_windows_separators_are_normalised(self):
        assert not ContentPolicy().path_allowed(r"C:\repo\.env")


class TestPathCanonicalisation:
    """A path that reaches a denied file must be denied however it spells itself.

    The gate matches globs against text. Any spelling that reaches the same file without matching the
    pattern is a way to get the file's bytes into the blob store, so these are the cases where getting
    normalisation wrong costs a secret rather than a node.
    """

    def test_a_parent_segment_cannot_walk_back_into_a_denied_region(self):
        # `/tmp/../secrets/key.txt` is `/secrets/key.txt`. Matched verbatim it is neither, and the
        # bytes were stored.
        p = ContentPolicy(deny_globs=("/secrets/*",))
        assert not p.path_allowed("/secrets/key.txt")
        assert not p.path_allowed("/tmp/../secrets/key.txt")
        assert not p.path_allowed("/a/b/../../secrets/key.txt")

    def test_a_drive_letter_does_not_hide_an_absolute_path(self):
        p = ContentPolicy(deny_globs=("/secrets/*",))
        assert not p.path_allowed("C:/secrets/key.txt")
        assert not p.path_allowed(r"C:\secrets\key.txt")

    def test_a_current_directory_segment_is_collapsed(self):
        p = ContentPolicy(deny_globs=("/secrets/*",))
        assert not p.path_allowed("/secrets/./key.txt")

    def test_a_sibling_directory_is_not_swept_up_by_the_collapse(self):
        # `/var/secrets` is a different directory from `/secrets`, and normalising must not conflate
        # them -- over-denying is safe but wrong, and this is the case that catches a lazy substring fix.
        p = ContentPolicy(deny_globs=("/secrets/*",))
        assert p.path_allowed("/var/./secrets/key.txt")
        assert p.path_allowed("/secretsandsuch/key.txt")

    def test_a_path_that_escapes_its_own_root_is_denied(self):
        # Where `../elsewhere/x` really points depends on a working directory this module does not
        # have, so no pattern can be shown not to apply. The gate fails closed.
        p = ContentPolicy(deny_globs=("/secrets/*",))
        assert not p.path_allowed("../elsewhere/x.txt")
        assert not p.path_allowed("..")

    def test_a_relative_path_that_stays_inside_itself_is_still_matched_normally(self):
        p = ContentPolicy(deny_globs=("*.pem",))
        assert p.path_allowed("repo/util.py")
        assert p.path_allowed("./repo/util.py")
        assert not p.path_allowed("./repo/key.pem")

    def test_an_empty_path_is_denied_rather_than_matched(self):
        assert not ContentPolicy().path_allowed("")

    def test_an_anchored_glob_is_as_strong_as_a_wildcard_one(self):
        """Why the shipped defaults never leaked, and why that was luck rather than design.

        Every entry in ``DEFAULT_DENY_GLOBS`` is matched on the basename or begins with ``*``, and
        ``fnmatch``'s ``*`` crosses ``/`` -- so ``*/.ssh/*`` already caught ``/home/u/x/../.ssh/id_rsa``
        before this fix. Only a glob anchored at ``/`` could be walked around, which is exactly the
        shape a deployment writes by hand for its own directories.

        Pinned because the two spellings must now be equally strong. Tightening a default to
        ``/home/*/.ssh/*`` some day must not quietly reopen this.
        """
        anchored = ContentPolicy(deny_globs=("/home/u/.ssh/*",))
        wildcard = ContentPolicy(deny_globs=("*/.ssh/*",))
        walked = "/home/u/project/../.ssh/id_rsa"

        assert not wildcard.path_allowed(walked)
        assert not anchored.path_allowed(walked)
        assert not ContentPolicy().path_allowed(walked)


class TestContentScrubbing:
    def test_a_recognised_secret_is_replaced(self):
        scrubbed, modified = ContentPolicy().scrub(b'AWS_SECRET_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"')
        assert modified
        assert REDACTION_MARKER in scrubbed
        assert b"AKIAIOSFODNN7EXAMPLE" not in scrubbed

    def test_ordinary_content_is_untouched(self):
        data = b"def helper():\n    return 1\n"
        assert ContentPolicy().scrub(data) == (data, False)

    def test_binary_content_is_passed_through_rather_than_mangled(self):
        data = b"\x00\x01\x02\xff"
        assert ContentPolicy().scrub(data) == (data, False)

    def test_scrubbing_can_be_disabled(self):
        data = b'password = "hunter2hunter2hunter2"'
        assert ContentPolicy(scrub_content=False).scrub(data) == (data, False)


class TestPrepare:
    def test_a_denied_path_withholds_bytes_but_reports_redaction(self):
        content, redacted = ContentPolicy().prepare("/repo/.env", b"SECRET=1")
        # None means "record identity, withhold bytes" -- the asset is still created.
        assert content is None
        assert redacted is True

    def test_oversized_content_is_identified_but_not_stored(self):
        p = ContentPolicy(max_content_bytes=16)
        content, redacted = p.prepare("/repo/big.log", b"x" * 100)
        assert content is None
        assert redacted is True

    def test_content_at_the_limit_is_stored(self):
        p = ContentPolicy(max_content_bytes=16, scrub_content=False)
        assert p.prepare("/repo/a.py", b"x" * 16) == (b"x" * 16, False)

    def test_absent_content_is_not_a_redaction(self):
        # A partial read establishes no content; that is not the same as withholding it.
        assert ContentPolicy().prepare("/repo/a.py", None) == (None, False)

    def test_allowed_content_passes_through(self):
        assert ContentPolicy().prepare("/repo/a.py", b"x = 1\n") == (b"x = 1\n", False)

    def test_the_permissive_policy_stores_everything(self):
        content, redacted = PERMISSIVE.prepare("/repo/.env", b"SECRET=1")
        assert (content, redacted) == (b"SECRET=1", False)
