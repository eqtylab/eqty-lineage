//! File extraction, and the things it must refuse to guess.
//!
//! Ported from the existing recorder's `tool_results.py`. The payload shapes below are Claude Code's
//! `tool_response`, which is what Relay hands us verbatim as the tool-end scope's `data`.

use eqty_lineage_nemo_relay::{
    FileMode, apply_edit, file_events_from_patch, file_events_from_result,
};
use serde_json::json;

#[test]
fn a_whole_read_carries_its_content() {
    let (events, attributed) = file_events_from_result(
        &json!({"type": "text", "file": {
            "filePath": "/report.md", "content": "# Report\n", "numLines": 1, "totalLines": 1, "startLine": 1
        }}),
        Some("t1"),
        true,
    );

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].path, "/report.md");
    assert_eq!(events[0].mode, FileMode::Read);
    assert_eq!(events[0].content.as_deref(), Some(b"# Report\n".as_slice()));
    assert_eq!(
        attributed, None,
        "a read changes nothing, so it attributes nothing"
    );
}

#[test]
fn a_sliced_read_establishes_the_path_but_not_the_content() {
    // The hazard this whole module exists to avoid. Hashing a fragment as the file would content-
    // address a version the file never had, and the manifest would assert it without hedging.
    for slice in [
        json!({"filePath": "/big.md", "content": "line 40\n", "startLine": 40, "numLines": 1, "totalLines": 900}),
        json!({"filePath": "/big.md", "content": "head\n", "startLine": 1, "numLines": 10, "totalLines": 900}),
    ] {
        let (events, _) = file_events_from_result(&json!({ "file": slice }), Some("t1"), true);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].path, "/big.md");
        assert_eq!(
            events[0].content, None,
            "a partial read must not be recorded as the file's content"
        );
    }
}

#[test]
fn an_edit_records_both_the_version_read_and_the_version_written() {
    let (events, attributed) = file_events_from_result(
        &json!({
            "filePath": "/a.py", "originalFile": "x = 1\n",
            "oldString": "x = 1", "newString": "x = 2", "replaceAll": false
        }),
        Some("t2"),
        true,
    );

    assert_eq!(events.len(), 2, "an edit is a read and a write: {events:?}");
    assert_eq!(events[0].mode, FileMode::Read);
    assert_eq!(events[0].content.as_deref(), Some(b"x = 1\n".as_slice()));
    assert_eq!(events[1].mode, FileMode::Wrote);
    assert_eq!(
        events[1].content.as_deref(),
        Some(b"x = 2\n".as_slice()),
        "replaying the replacement reproduces the file exactly"
    );
    assert_eq!(attributed.as_deref(), Some("/a.py"));
}

#[test]
fn an_edit_without_its_pre_image_hands_on_the_replacement() {
    // `originalFile` is null on most Edit results, so this is the common case rather than the edge.
    // Dropping it would lose the transition; guessing would invent one. It is carried instead, for
    // the recorder to replay against content the session already knows.
    let (events, _) = file_events_from_result(
        &json!({"filePath": "/a.py", "oldString": "x = 1", "newString": "x = 2", "structuredPatch": []}),
        Some("t3"),
        true,
    );

    assert_eq!(events.len(), 1, "no pre-image means no read event");
    assert_eq!(events[0].content, None);
    let attempt = events[0]
        .edit
        .as_ref()
        .expect("the replacement is carried forward");
    assert_eq!(attempt.hunks.len(), 1, "a literal Edit is one hunk");
    assert_eq!(attempt.hunks[0].old, "x = 1");
    assert_eq!(attempt.hunks[0].new, "x = 2");
}

#[test]
fn a_write_gives_the_post_state_directly() {
    let (events, _) = file_events_from_result(
        &json!({"filePath": "/new.md", "content": "created\n", "type": "create"}),
        Some("t4"),
        true,
    );
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].mode, FileMode::Wrote);
    assert_eq!(events[0].content.as_deref(), Some(b"created\n".as_slice()));
}

#[test]
fn a_payload_that_disagrees_with_itself_is_refused() {
    // `old` does not occur in `original`. Replaying anyway would produce a version that never
    // existed, and content-address it as though it had.
    assert_eq!(
        apply_edit(Some("x = 1\n"), Some("y = 9"), Some("z"), false),
        None
    );
    assert_eq!(apply_edit(None, Some("a"), Some("b"), false), None);
    assert_eq!(apply_edit(Some("aaa"), None, Some("b"), false), None);
}

#[test]
fn replace_all_is_honoured() {
    assert_eq!(
        apply_edit(Some("a a a"), Some("a"), Some("b"), true).as_deref(),
        Some("b b b")
    );
    assert_eq!(
        apply_edit(Some("a a a"), Some("a"), Some("b"), false).as_deref(),
        Some("b a a")
    );
}

#[test]
fn a_bash_result_yields_nothing() {
    // Every Bash result in the reference capture is a bare string. A shell command that writes a
    // file is not attributable from its result alone, and inventing an attribution would be worse
    // than the gap.
    let (events, attributed) = file_events_from_result(&json!("/Users/b/Dev\n"), Some("t5"), true);
    assert!(events.is_empty());
    assert_eq!(attributed, None);
}

#[test]
fn an_added_file_is_recovered_exactly_from_a_patch() {
    let events = file_events_from_patch(
        "*** Begin Patch\n*** Add File: src/new.rs\n+fn main() {}\n+// done\n*** End Patch\n",
        Some("t6"),
    );
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].path, "src/new.rs");
    assert_eq!(events[0].mode, FileMode::Wrote);
    assert_eq!(
        events[0].content.as_deref(),
        Some(b"fn main() {}\n// done\n".as_slice()),
        "an Add File body is the whole file, so stripping the `+` reproduces it byte for byte"
    );
}

#[test]
fn an_updated_file_is_named_but_not_reconstructed() {
    // `Update File` carries hunks and no pre-image, so the post-image cannot be computed from the
    // document. Guessing would content-address a state the file may never have had.
    let events = file_events_from_patch(
        "*** Begin Patch\n*** Update File: src/old.rs\n@@\n-was\n+is\n*** End Patch\n",
        Some("t7"),
    );
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].path, "src/old.rs");
    assert_eq!(events[0].content, None, "hunks are not a file");
}

#[test]
fn a_patch_touching_several_files_yields_one_event_each() {
    let events = file_events_from_patch(
        "*** Begin Patch\n\
         *** Add File: a.txt\n+alpha\n\
         *** Update File: b.txt\n@@\n-x\n+y\n\
         *** Delete File: c.txt\n\
         *** End Patch\n",
        Some("t8"),
    );
    let paths: Vec<&str> = events.iter().map(|e| e.path.as_str()).collect();
    assert_eq!(paths, vec!["a.txt", "b.txt", "c.txt"]);
    assert_eq!(events[0].content.as_deref(), Some(b"alpha\n".as_slice()));
    assert_eq!(events[1].content, None);
    assert_eq!(
        events[2].content, None,
        "a deleted file has no content to hash"
    );
}

#[test]
fn text_that_is_not_a_patch_yields_nothing() {
    assert!(file_events_from_patch("just some output\n", Some("t9")).is_empty());
}

#[test]
fn a_read_cut_off_mid_line_is_a_fragment() {
    // Observed live, and the reason this exists: a 200 KB single-line file came back as 21 KB with
    // `numLines == totalLines == 1`, because line counts cannot express a cut *within* a line. The
    // line-count check therefore reads it as complete, and the manifest asserts a content CID for
    // bytes that are not what is on disk -- a confident wrong claim in a signed document.
    let result = serde_json::json!({
        "file": {
            "filePath": "/tmp/big.txt",
            "content": "xxxxxxxxxx",
            "numLines": 1,
            "startLine": 1,
            "totalLines": 1,
            "truncatedByTokenCap": true
        },
        "type": "text"
    });
    let (events, _) = file_events_from_result(&result, Some("toolu_1"), true);
    assert_eq!(events.len(), 1, "the path must still be recorded");
    assert_eq!(
        events[0].content, None,
        "a truncated read must not claim the bytes it returned are the file"
    );
    assert_eq!(events[0].path, "/tmp/big.txt");
}

#[test]
fn an_untruncated_read_still_carries_its_content() {
    // The guard on the test above: the flag must gate the fragment path, not the whole extraction.
    let result = serde_json::json!({
        "file": {
            "filePath": "/tmp/small.txt",
            "content": "hello\n",
            "numLines": 1,
            "startLine": 1,
            "totalLines": 1,
            "truncatedByTokenCap": false
        },
        "type": "text"
    });
    let (events, _) = file_events_from_result(&result, Some("toolu_2"), true);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].content.as_deref(), Some(&b"hello\n"[..]));
}

#[test]
fn an_update_hunk_becomes_a_replayable_edit() {
    // The post-image is never in an `apply_patch` update, so before this the node was always
    // identity-only. But the hunk is exactly the pair `apply_edit` takes, and on Codex `Update File`
    // is how every edit arrives -- so every Codex edit recorded a file whose content we declined to
    // work out despite holding everything needed.
    let patch = "*** Begin Patch\n\
                 *** Update File: /work/report.md\n\
                 @@\n\
                 one\n\
                 -two\n\
                 +TWO\n\
                 three\n\
                 *** End Patch";
    let events = file_events_from_patch(patch, Some("call-1"));
    assert_eq!(events.len(), 1);
    let edit = events[0].edit.as_ref().expect("the hunk is an edit");
    assert_eq!(events[0].path, "/work/report.md");
    assert_eq!(
        events[0].content, None,
        "the post-image is still not stated"
    );
    assert!(
        edit.hunks
            .iter()
            .any(|hunk| hunk.old.contains("two") && hunk.new.contains("TWO")),
        "the halves carry the change: {edit:?}"
    );

    assert!(
        edit.line_oriented,
        "a patch hunk carries no promise of uniqueness, so it replays over whole lines, which \
         enforces one itself"
    );

    // Against a pre-image where the removed text occurs once, the replay is exact.
    let replayed = apply_edit(
        Some("one\ntwo\nthree\n"),
        Some(&edit.hunks[0].old),
        Some(&edit.hunks[0].new),
        edit.replace_all,
    );
    assert_eq!(replayed.as_deref(), Some("one\nTWO\nthree\n"));
}

#[test]
fn a_context_free_hunk_stays_identity_only() {
    // Without context or removals there is nothing to anchor a literal replacement against, and
    // guessing would content-address a state the file may never have had.
    let patch = "*** Begin Patch\n\
                 *** Update File: /work/report.md\n\
                 @@\n\
                 +appended\n\
                 *** End Patch";
    let events = file_events_from_patch(patch, Some("call-2"));
    assert_eq!(events.len(), 1);
    assert!(events[0].edit.is_none(), "no anchor, no replay");
    assert_eq!(events[0].content, None);
}
