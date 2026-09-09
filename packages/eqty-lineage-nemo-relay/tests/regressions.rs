//! Defects a review found in the recorder, each pinned so it cannot come back.
//!
//! Two themes run through them, and both are worse than a missing feature. One is a **manifest that
//! states something false**: a content CID for a version the file never held, a rejected write
//! recorded as a write. The other is a **redaction that did not redact**: the file node said
//! `withheld` while the tool call beside it carried the bytes in full.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use eqty_lineage_nemo_relay::{
    CompactionPhase, EditAttempt, FileMode, FileObserved, LineageSession, Mailbox, ManifestMark,
    Policy, Recorder, ReplayRefusal, SessionFinished, SessionRouter, SignerFactory, apply_edit,
    apply_line_edit, classify, file_events_from_patch,
};
use integrity::cid::blake3::blake3_cid_raw_binary;
use integrity::lineage::models::manifest::Manifest;
use integrity::signer::{SignerType, ed25519_signer::Ed25519Signer};
use nemo_relay_plugin::Event;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

/// Parse a metadata field that `encode_nested_values` renders as a JSON string.
///
/// Nested metadata reaches the manifest JSON-encoded, because the graph explorer displays each
/// value as a string and shows an object as `[object Object]`. A reader wanting the structure
/// parses one string; these tests do the same rather than asserting on the escaping.
fn nested(node: &serde_json::Value, key: &str) -> serde_json::Value {
    let text = node[key]
        .as_str()
        .unwrap_or_else(|| panic!("`{key}` should be a JSON string, got {}", node[key]));
    serde_json::from_str(text).unwrap_or_else(|_| panic!("`{key}` should parse back: {text}"))
}

fn signer_factory() -> SignerFactory {
    Box::new(|| {
        Ed25519Signer::create()
            .ok()
            .map(|signer| LineageSession::new(SignerType::ED25519(signer)))
    })
}

fn policy() -> Policy {
    Policy::new(vec![".env*".into(), "*.pem".into()], 1_048_576)
}

fn recorder() -> Recorder {
    let signer = Ed25519Signer::create().expect("a signer");
    Recorder::new(LineageSession::new(SignerType::ED25519(signer)), policy())
}

fn seen(path: &str, content: Option<&[u8]>, mode: FileMode) -> FileObserved {
    FileObserved {
        path: path.to_string(),
        content: content.map(<[u8]>::to_vec),
        mode,
        tool_use_id: Some("t1".into()),
        user_modified: false,
        edit: None,
    }
}

/// A one-file `Update File` patch whose hunk is the given lines.
fn hunk(path: &str, lines: &[&str]) -> String {
    format!(
        "*** Begin Patch\n*** Update File: {path}\n@@\n{}\n*** End Patch",
        lines.join("\n")
    )
}

fn anchored(old: &str, new: &str) -> Option<EditAttempt> {
    Some(EditAttempt {
        old: old.into(),
        new: new.into(),
        replace_all: false,
        unique_only: false,
        line_oriented: false,
        replay_from: None,
    })
}

fn forget_with(router: &Arc<Mutex<SessionRouter>>) -> SessionFinished {
    let router = Arc::clone(router);
    Box::new(move |session_id: &str| {
        router.lock().expect("lock").forget(session_id);
    })
}

fn replay(events: &[Event], into: &TempDir) -> Vec<ManifestMark> {
    let router = Arc::new(Mutex::new(SessionRouter::new()));
    let announced: Arc<Mutex<Vec<ManifestMark>>> = Arc::new(Mutex::new(Vec::new()));
    let marks = Arc::clone(&announced);
    let mailbox = Mailbox::start(
        into.path().to_path_buf(),
        policy(),
        signer_factory(),
        forget_with(&router),
        Box::new(move |mark: &ManifestMark| {
            marks.lock().expect("the marks lock").push(mark.clone())
        }),
    );
    for event in events {
        let Some(session_id) = router.lock().expect("lock").attribute(event) else {
            continue;
        };
        if let Some(lineage) = classify(event) {
            mailbox.send(&session_id, event.timestamp().to_rfc3339(), lineage);
        }
    }
    // Dropped before the marks are read: the final export runs when the last handle goes, so a
    // read before this returns whatever the checkpoints managed rather than what the session says.
    drop(mailbox);
    Arc::into_inner(announced)
        .expect("the mailbox thread has ended, so this is the only handle")
        .into_inner()
        .expect("the marks lock")
}

fn manifests(dir: &TempDir) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = fs::read_dir(dir.path())
        .expect("dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    found.sort();
    found
}

fn blobs_of(manifest: &serde_json::Value) -> Vec<(String, Vec<u8>)> {
    manifest["blobs"]
        .as_object()
        .expect("blobs")
        .iter()
        .filter_map(|(cid, b)| {
            BASE64
                .decode(b.as_str()?)
                .ok()
                .map(|bytes| (cid.clone(), bytes))
        })
        .collect()
}

fn decoded(manifest: &serde_json::Value) -> String {
    blobs_of(manifest)
        .into_iter()
        .map(|(_, b)| String::from_utf8_lossy(&b).into_owned())
        .collect()
}

fn on_disk(path: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

async fn exported(rec: Recorder) -> serde_json::Value {
    let manifest: Manifest = rec.finish(None).await.expect("manifest");
    serde_json::to_value(&manifest).expect("serializes")
}

/// The stored bytes of the *latest* `Document` node for `path`.
///
/// Selected by `fileVersion`, because a path has one node per version it was seen at and the blob
/// map has no order. Taking whichever came first made two of these tests fail against correct code.
fn document_content(manifest: &serde_json::Value, path: &str) -> Option<String> {
    let all = blobs_of(manifest);
    let node = all
        .iter()
        .filter_map(|(_, bytes)| {
            let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
            (value["assetType"] == "Document" && value["filePath"] == path).then_some(value)
        })
        .max_by_key(|value| value["fileVersion"].as_u64().unwrap_or(0))?;
    let cid = node["content-cid"].as_str()?.to_string();
    all.iter()
        .find(|(blob_cid, _)| *blob_cid == cid)
        .map(|(_, bytes)| String::from_utf8_lossy(bytes).into_owned())
}

fn mark(session: &str, uuid: &str, parent: &str, name: &str, metadata: serde_json::Value) -> Event {
    let mut meta = serde_json::json!({ "session_id": session, "agent_kind": "claude-code" });
    if let (Some(t), Some(x)) = (meta.as_object_mut(), metadata.as_object()) {
        for (k, v) in x {
            t.insert(k.clone(), v.clone());
        }
    }
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1", "kind": "mark", "name": name,
        "uuid": uuid, "parent_uuid": parent,
        "timestamp": "2026-09-08T12:00:00.000000+00:00", "metadata": meta
    }))
    .expect("mark")
}

#[allow(clippy::too_many_arguments)]
fn tool_scope(
    session: &str,
    uuid: &str,
    parent: &str,
    phase: &str,
    name: &str,
    call_id: &str,
    data: serde_json::Value,
    status: Option<&str>,
) -> Event {
    let mut meta = serde_json::json!({
        "session_id": session, "agent_kind": "claude-code",
        "hook_event_name": if phase == "start" { "PreToolUse" } else { "PostToolUse" },
        "source": "hook", "tool_correlation_status": "explicit"
    });
    if let Some(s) = status {
        meta["status"] = serde_json::json!(s);
    }
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1", "kind": "scope", "category": "tool",
        "scope_category": phase, "name": name, "uuid": uuid, "parent_uuid": parent,
        "timestamp": "2026-09-08T12:00:00.000000+00:00", "attributes": [],
        "data": data, "data_schema": null,
        "category_profile": { "tool_call_id": call_id }, "metadata": meta
    }))
    .expect("tool scope")
}

// ------------------------------------------------------- a redaction that did not redact

#[test]
fn withholding_a_file_withholds_it_from_the_call_that_read_it() {
    // The file node correctly said `withheld` -- and the tool result beside it stored the same bytes
    // in full, because the deny list is written for paths and this payload is named `Read result`.
    // A manifest that reports a redaction it did not perform is worse than one that redacts nothing:
    // it tells the reader the secret is not there.
    let into = TempDir::new().unwrap();
    let session = "01a040aa-0000-0000-0000-0000000000e1";
    let root = "01a040aa-0000-0000-0000-0000000000e2";
    let call = "01a040aa-0000-0000-0000-0000000000e3";
    let secret = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI\n";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({"model":"opus"}),
        ),
        tool_scope(
            session,
            call,
            root,
            "start",
            "Read",
            "toolu_e1",
            serde_json::json!({ "file_path": "/app/.env" }),
            None,
        ),
        tool_scope(
            session,
            call,
            root,
            "end",
            "Read",
            "toolu_e1",
            serde_json::json!({ "file": { "filePath": "/app/.env", "content": secret,
                "numLines": 1, "startLine": 1, "totalLines": 1 }, "type": "text" }),
            None,
        ),
    ];
    replay(&events, &into);

    let manifest = on_disk(&manifests(&into)[0]);
    let text = decoded(&manifest);
    assert!(
        !text.contains("wJalrXUtnFEMI"),
        "the secret must not survive anywhere in the manifest"
    );
    assert!(
        text.contains("\"contentState\":\"withheld\""),
        "the file node still records that it was seen and withheld"
    );
    assert!(
        text.contains("\"withheldBecause\":\"quotes-a-denied-file\""),
        "and the enclosing payload says why it was withheld too:\n{text}"
    );
    assert!(
        text.contains("/app/.env"),
        "the path is still recorded -- provenance does not require publication"
    );
}

#[test]
fn writing_to_a_denied_path_withholds_the_arguments_that_carried_the_content() {
    // The other direction: a `Write` puts the file's content in its *arguments*, so the input
    // payload leaks what the output node withholds.
    let into = TempDir::new().unwrap();
    let session = "01a040aa-0000-0000-0000-0000000000ec";
    let root = "01a040aa-0000-0000-0000-0000000000ed";
    let call = "01a040aa-0000-0000-0000-0000000000ee";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({"model":"opus"}),
        ),
        tool_scope(
            session,
            call,
            root,
            "start",
            "Write",
            "toolu_ec",
            serde_json::json!({ "file_path": "/app/.env", "content": "DB_PASSWORD=hunter2\n" }),
            None,
        ),
        tool_scope(
            session,
            call,
            root,
            "end",
            "Write",
            "toolu_ec",
            serde_json::json!({ "filePath": "/app/.env", "content": "DB_PASSWORD=hunter2\n",
                "type": "create" }),
            None,
        ),
    ];
    replay(&events, &into);

    let text = decoded(&on_disk(&manifests(&into)[0]));
    assert!(
        !text.contains("hunter2"),
        "neither the file node nor the arguments may carry it:\n{text}"
    );
}

#[tokio::test]
async fn a_payload_quoting_no_denied_file_is_still_stored() {
    // The guard: inheriting a path's policy must not withhold every payload. A tool result that
    // quotes nothing denied is ordinary content and belongs in the graph.
    let mut rec = recorder();
    rec.register_quoting_payload(
        "Dataset",
        "Read result",
        "What the 'Read' tool returned.",
        b"{\"content\":\"ordinary\"}",
        serde_json::json!({}),
        None,
        &["/work/notes.md".to_string()],
    )
    .await
    .unwrap();
    assert_eq!(rec.stats().get("PayloadDenied"), None);

    // And a denied path in the same position does withhold it.
    let mut rec = recorder();
    rec.register_quoting_payload(
        "Dataset",
        "Read result",
        "What the 'Read' tool returned.",
        b"{\"content\":\"SECRET=1\"}",
        serde_json::json!({}),
        None,
        &["/app/.env".to_string()],
    )
    .await
    .unwrap();
    assert_eq!(rec.stats().get("PayloadDenied"), Some(&1));
}

// ------------------------------------------------------- a CID for a version that never existed

#[tokio::test]
async fn returning_to_an_earlier_version_moves_the_replay_base_back() {
    // The dedup return skipped the base refresh, so a file that went A -> B -> A left B cached. A
    // later edit anchored on A was then refused, or -- worse, if its `old` text also occurred in B --
    // replayed against content the file no longer held.
    let mut rec = recorder();
    for bytes in [&b"x = 1\n"[..], &b"x = 2\n"[..], &b"x = 1\n"[..]] {
        rec.observe_file(&seen("/a.py", Some(bytes), FileMode::Read), true, None)
            .await
            .unwrap();
    }
    let mut edit = seen("/a.py", None, FileMode::Wrote);
    edit.edit = anchored("x = 1", "x = 3");
    rec.observe_file(&edit, true, None).await.unwrap();

    assert_eq!(
        rec.stats().get("ContentRecovered"),
        Some(&1),
        "the base is the version the file actually holds: {:?}",
        rec.stats()
    );
    assert_eq!(
        document_content(&exported(rec).await, "/a.py").as_deref(),
        Some("x = 3\n")
    );
}

#[tokio::test]
async fn a_write_we_could_not_reconstruct_invalidates_the_replay_base() {
    // The stale base was retained, so a later edit whose `old` text still occurred in the *pre-write*
    // content was reported as recovered -- signing a file version that never existed. Every Codex
    // `Update File` whose hunk does not replay lands here, so this is the common path, not the edge.
    let mut rec = recorder();
    rec.observe_file(
        &seen("/a.md", Some(b"hello world\n"), FileMode::Read),
        true,
        None,
    )
    .await
    .unwrap();
    rec.observe_file(&seen("/a.md", None, FileMode::Wrote), true, None)
        .await
        .unwrap();

    let mut edit = seen("/a.md", None, FileMode::Wrote);
    edit.edit = anchored("hello", "goodbye");
    rec.observe_file(&edit, true, None).await.unwrap();

    assert_eq!(
        rec.stats().get("ContentRecovered"),
        None,
        "what is on disk is unknown, so nothing may be reconstructed from what it replaced"
    );
}

#[tokio::test]
async fn a_read_we_could_not_establish_keeps_the_replay_base() {
    // The complement, and the reason the fix is mode-sensitive rather than a blanket invalidation: a
    // truncated `Read` tells us nothing new but changes nothing on disk, so what we hold is still
    // the file's content and still a valid base.
    let mut rec = recorder();
    rec.observe_file(
        &seen("/a.md", Some(b"hello world\n"), FileMode::Read),
        true,
        None,
    )
    .await
    .unwrap();
    rec.observe_file(&seen("/a.md", None, FileMode::Read), true, None)
        .await
        .unwrap();

    let mut edit = seen("/a.md", None, FileMode::Wrote);
    edit.edit = anchored("hello", "goodbye");
    rec.observe_file(&edit, true, None).await.unwrap();

    assert_eq!(rec.stats().get("ContentRecovered"), Some(&1));
    assert_eq!(
        document_content(&exported(rec).await, "/a.md").as_deref(),
        Some("goodbye world\n")
    );
}

// ------------------------------------------------------- patch parsing

#[test]
fn deleting_a_line_takes_its_newline_with_it() {
    // The halves were joined with `join("\n")`, which drops the last line's terminator. Deleting `b`
    // from `a\nb\nc\n` became the replacement `b` -> `` and produced `a\n\nc\n` -- a blank line the
    // patch never created.
    let events = file_events_from_patch(&hunk("/work/x.txt", &["-b"]), Some("c1"));
    let edit = events[0].edit.as_ref().expect("a replayable edit");
    assert_eq!(edit.old, "b\n", "the removed line keeps its terminator");
    assert_eq!(edit.new, "");
    assert!(
        edit.line_oriented,
        "and a hunk replays in lines, not substrings"
    );

    assert_eq!(
        apply_line_edit("a\nb\nc\n", &edit.old, &edit.new).as_deref(),
        Ok("a\nc\n"),
        "which is what the patch does to the file"
    );
}

#[test]
fn an_eof_anchored_hunk_cannot_be_answered_by_an_earlier_line() {
    // Restoring the terminator alone made one case *worse* than before it. It narrows what the anchor
    // matches, so against the unterminated file `b\nb` the hunk `-b` / `+c` searched for `b\n`,
    // matched the FIRST line exactly once, passed the uniqueness guard and recorded `c\nb` -- while
    // the two bare `b`s had previously refused the replay outright. A fix that turns a refusal into a
    // wrong answer is worse than the bug it fixed.
    //
    // The uniqueness of an anchor is only meaningful in the unit the patch speaks in. Matched as runs
    // of whole lines, `b` occurs twice in `b\nb` and the hunk has not said which one it meant.
    let events = file_events_from_patch(&hunk("/work/x.txt", &["-b", "+c"]), Some("c2"));
    let edit = events[0].edit.as_ref().unwrap();

    assert_eq!(
        apply_line_edit("b\nb", &edit.old, &edit.new),
        Err(ReplayRefusal::Ambiguous),
        "two candidate lines, so no reconstruction"
    );
    // The substring replay it replaced would have answered `c\nb`, at the wrong end of the file.
    assert_eq!(
        apply_edit(Some("b\nb"), Some(&edit.old), Some(&edit.new), false).as_deref(),
        Some("c\nb")
    );
}

#[tokio::test]
async fn an_ambiguous_hunk_is_counted_rather_than_reconstructed() {
    // End to end: the refusal has to reach coverage, or a reader cannot tell a file whose edit we
    // declined to replay from one that was never edited.
    let mut rec = recorder();
    rec.observe_file(&seen("/w/x.txt", Some(b"b\nb"), FileMode::Read), true, None)
        .await
        .unwrap();
    let events = file_events_from_patch(&hunk("/w/x.txt", &["-b", "+c"]), Some("c3"));
    rec.observe_file(&events[0], true, None).await.unwrap();

    assert_eq!(rec.stats().get("ContentRecovered"), None);
    assert_eq!(rec.stats().get("EditTooAmbiguousToReplay"), Some(&1));
}

#[test]
fn a_substitution_hunk_still_replays_exactly() {
    // The guard on the fix above: restoring terminators and matching in lines must not break the case
    // that already worked.
    let events = file_events_from_patch(
        &hunk("/work/report.md", &["one", "-two", "+TWO", "three"]),
        Some("c4"),
    );
    let edit = events[0].edit.as_ref().expect("a replayable edit");
    assert_eq!(
        apply_line_edit("one\ntwo\nthree\n", &edit.old, &edit.new).as_deref(),
        Ok("one\nTWO\nthree\n")
    );
}

#[test]
fn a_hunk_touching_an_unterminated_final_line_is_refused() {
    // Deleting `b` from `a\nb` leaves `a\n` or `a` depending on what the tool did with the preceding
    // line's terminator, and the hunk does not say. Refused with its own reason rather than guessed,
    // and rather than lumped in with a stale-belief mismatch.
    let events = file_events_from_patch(&hunk("/work/x.txt", &["-b"]), Some("c5"));
    let edit = events[0].edit.as_ref().unwrap();
    assert_eq!(
        apply_line_edit("a\nb", &edit.old, &edit.new),
        Err(ReplayRefusal::UnterminatedAtEof)
    );
    // The same hunk against the terminated file is fine.
    assert_eq!(
        apply_line_edit("a\nb\n", &edit.old, &edit.new).as_deref(),
        Ok("a\n")
    );
}

#[test]
fn an_unterminated_file_edited_above_its_last_line_stays_unterminated() {
    // Reassembly has to restore exactly what the split removed, not append a terminator by habit.
    // Here the window does not touch the final line, so the replay is allowed -- and `c` must keep
    // its missing newline, or the reconstruction is a different file from the one on disk.
    let events = file_events_from_patch(&hunk("/work/x.txt", &["-b"]), Some("c7"));
    let edit = events[0].edit.as_ref().unwrap();
    assert_eq!(
        apply_line_edit("a\nb\nc", &edit.old, &edit.new).as_deref(),
        Ok("a\nc")
    );
}

#[test]
fn deleting_every_line_leaves_an_empty_file_not_a_blank_one() {
    let events = file_events_from_patch(&hunk("/work/x.txt", &["-only"]), Some("c8"));
    let edit = events[0].edit.as_ref().unwrap();
    assert_eq!(
        apply_line_edit("only\n", &edit.old, &edit.new).as_deref(),
        Ok(""),
        "an emptied file is empty, not a single newline"
    );
}

#[test]
fn a_crlf_file_is_not_silently_rewritten_to_lf() {
    // `str::lines` strips `\r` as well as `\n` and says nothing about which it removed, so rebuilding
    // with `\n` rewrote every line of a CRLF file. `a\r\nb\r\nc\r\nc` with `-b` / `+B` came back as
    // `a\nB\nc\n`: three lines changed where the patch named one, attested as recovered content.
    //
    // The substring replay this replaced refused the same input outright, so the line-oriented fix
    // turned a refusal into a confident wrong answer -- the same shape of regression twice running,
    // and the reason a fix here has to be checked against what it *stops* refusing.
    let events = file_events_from_patch(&hunk("/work/x.txt", &["-b", "+B"]), Some("c9"));
    let edit = events[0].edit.as_ref().unwrap();

    assert_eq!(
        apply_line_edit("a\r\nb\r\nc\r\n", &edit.old, &edit.new),
        Err(ReplayRefusal::TerminatorsNotEstablished),
        "a line the patch introduces has no terminator in the patch or in the pre-image"
    );
}

#[test]
fn deleting_a_line_from_a_crlf_file_is_exact() {
    // The case that *is* establishable, and worth keeping rather than refusing wholesale: a pure
    // deletion introduces no line, so there is no terminator to invent and every surviving byte is
    // copied. Untouched lines keep their `\r\n`.
    let events = file_events_from_patch(&hunk("/work/x.txt", &["-b"]), Some("c10"));
    let edit = events[0].edit.as_ref().unwrap();
    assert_eq!(
        apply_line_edit("a\r\nb\r\nc\r\n", &edit.old, &edit.new).as_deref(),
        Ok("a\r\nc\r\n")
    );
}

#[test]
fn an_untouched_line_keeps_its_own_bytes() {
    // Mixed terminators are the sharpest version: whatever the file did per line, the lines the patch
    // did not name come back exactly as they were. Matching is on content, reassembly is on bytes.
    let events = file_events_from_patch(&hunk("/work/x.txt", &["-b"]), Some("c11"));
    let edit = events[0].edit.as_ref().unwrap();
    assert_eq!(
        apply_line_edit("a\r\nb\nc\r\nd\n", &edit.old, &edit.new).as_deref(),
        Ok("a\r\nc\r\nd\n")
    );
}

#[tokio::test]
async fn a_crlf_substitution_is_counted_rather_than_reconstructed() {
    let mut rec = recorder();
    rec.observe_file(
        &seen("/w/x.txt", Some(b"a\r\nb\r\nc\r\n"), FileMode::Read),
        true,
        None,
    )
    .await
    .unwrap();
    let events = file_events_from_patch(&hunk("/w/x.txt", &["-b", "+B"]), Some("c12"));
    rec.observe_file(&events[0], true, None).await.unwrap();

    assert_eq!(rec.stats().get("ContentRecovered"), None);
    assert_eq!(rec.stats().get("EditTerminatorsNotEstablished"), Some(&1));
}

#[test]
fn a_hunk_written_against_content_we_do_not_hold_is_refused() {
    let events = file_events_from_patch(&hunk("/work/x.txt", &["-zzz", "+q"]), Some("c6"));
    let edit = events[0].edit.as_ref().unwrap();
    assert_eq!(
        apply_line_edit("a\nb\n", &edit.old, &edit.new),
        Err(ReplayRefusal::NotFound)
    );
}

#[tokio::test]
async fn a_moved_file_is_recorded_at_its_destination() {
    // `*** Move to:` was ignored outright, so the observation claimed the source had been rewritten
    // in place and the destination -- the file that actually ends up holding the post-image -- never
    // entered the graph. The hunk still applies to the source's content, which is why the edit has to
    // name where to replay from.
    let patch = "*** Begin Patch\n\
                 *** Update File: /work/a.txt\n\
                 *** Move to: /work/b.txt\n\
                 @@\n\
                 -one\n\
                 +two\n\
                 *** End Patch";
    let events = file_events_from_patch(patch, Some("c4"));
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].path, "/work/b.txt", "the post-image lands here");
    assert_eq!(
        events[0].edit.as_ref().unwrap().replay_from.as_deref(),
        Some("/work/a.txt"),
        "but it is computed from the source's content"
    );

    // End to end: the source's content is known, so the destination is fully established.
    let mut rec = recorder();
    rec.observe_file(
        &seen("/work/a.txt", Some(b"one\n"), FileMode::Read),
        true,
        None,
    )
    .await
    .unwrap();
    rec.observe_file(&events[0], true, None).await.unwrap();

    assert_eq!(rec.stats().get("ContentRecovered"), Some(&1));
    let manifest = exported(rec).await;
    assert_eq!(
        document_content(&manifest, "/work/b.txt").as_deref(),
        Some("two\n"),
        "the destination holds the patched content"
    );
}

// ------------------------------------------------------- inference after an explicit failure

#[test]
fn a_rejected_patch_registers_no_file_at_all() {
    // The patch branch derives file versions from what the agent *asked for*. A rejected `Add File`
    // therefore registered the requested bytes as a written file, counted a `FileWritten`, and seeded
    // the replay chain with content that never reached disk -- so a later edit could be "recovered"
    // from it. The activity and its arguments are still recorded; only the claim is dropped.
    let into = TempDir::new().unwrap();
    let session = "01a040aa-0000-0000-0000-0000000000e4";
    let root = "01a040aa-0000-0000-0000-0000000000e5";
    let call = "01a040aa-0000-0000-0000-0000000000e6";
    let patch =
        "*** Begin Patch\n*** Add File: /work/rejected.rs\n+fn never_written() {}\n*** End Patch\n";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({"model":"gpt"}),
        ),
        tool_scope(
            session,
            call,
            root,
            "start",
            "apply_patch",
            "toolu_e6",
            serde_json::json!({ "command": patch }),
            None,
        ),
        tool_scope(
            session,
            call,
            root,
            "end",
            "apply_patch",
            "toolu_e6",
            serde_json::json!("error: patch rejected\n"),
            Some("error"),
        ),
    ];
    replay(&events, &into);

    let text = decoded(&on_disk(&manifests(&into)[0]));
    assert!(
        !text.contains("\"FileWritten\""),
        "nothing was written:\n{text}"
    );
    assert!(
        text.contains("\"FileInferenceSkippedAfterFailure\":1"),
        "and the gap is counted rather than silent"
    );
    assert!(
        text.contains("\"outcome\":\"failed\""),
        "the failed call is still an activity in the graph"
    );
    assert!(
        text.contains("never_written"),
        "and its arguments are still recorded -- what was attempted is evidence too"
    );
}

#[test]
fn a_failed_write_is_not_counted_as_a_write() {
    // Wider than the patch branch: the arguments fallback infers the mode from the tool's *name*, so
    // a `Write` that returned `EACCES` was recorded as a file written.
    let into = TempDir::new().unwrap();
    let session = "01a040aa-0000-0000-0000-0000000000e9";
    let root = "01a040aa-0000-0000-0000-0000000000ea";
    let call = "01a040aa-0000-0000-0000-0000000000eb";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({"model":"opus"}),
        ),
        tool_scope(
            session,
            call,
            root,
            "start",
            "Write",
            "toolu_e9",
            serde_json::json!({ "file_path": "/work/denied.md", "content": "x\n" }),
            None,
        ),
        tool_scope(
            session,
            call,
            root,
            "end",
            "Write",
            "toolu_e9",
            serde_json::json!("EACCES: permission denied\n"),
            Some("error"),
        ),
    ];
    replay(&events, &into);

    let text = decoded(&on_disk(&manifests(&into)[0]));
    assert!(
        !text.contains("\"FileWritten\""),
        "no write happened:\n{text}"
    );
    assert!(text.contains("\"outcome\":\"failed\""));
}

#[test]
fn a_successful_call_still_infers_from_its_arguments() {
    // The guard: gating on failure must not disable the arguments fallback, which is the only thing
    // that gives a Codex shell command or a bare-string result any file lineage at all.
    let into = TempDir::new().unwrap();
    let session = "01a040aa-0000-0000-0000-0000000000ef";
    let root = "01a040aa-0000-0000-0000-0000000000f0";
    let call = "01a040aa-0000-0000-0000-0000000000f1";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({"model":"opus"}),
        ),
        tool_scope(
            session,
            call,
            root,
            "start",
            "Write",
            "toolu_ef",
            serde_json::json!({ "file_path": "/work/ok.md", "content": "x\n" }),
            None,
        ),
        tool_scope(
            session,
            call,
            root,
            "end",
            "Write",
            "toolu_ef",
            serde_json::json!("wrote 2 bytes\n"),
            Some("ok"),
        ),
    ];
    replay(&events, &into);

    let text = decoded(&on_disk(&manifests(&into)[0]));
    assert!(text.contains("\"FileWritten\":1"), "{text}");
    assert!(text.contains("/work/ok.md"));
}

// ------------------------------------------------------- manifest naming

#[test]
fn a_second_agentless_recording_gets_its_own_manifest() {
    // `manifest_path` checked `{id}.json` for collisions, but an agentless session is exported as
    // `{id}.unattributed.json` and its checkpoint deleted -- so the next recording found `{id}.json`
    // free, took it, and overwrote the earlier export. The guard was testing a filename the code does
    // not use. Codex produces exactly this: an ancillary title-generation call under its own id.
    let into = TempDir::new().unwrap();
    let session = "01a040aa-0000-0000-0000-0000000000e7";
    let root = "01a040aa-0000-0000-0000-0000000000e8";

    // No `session.start`, so no agent is registered.
    let one_call = |tag: &str, body: &str| {
        vec![
            tool_scope(
                session,
                root,
                root,
                "start",
                "Bash",
                tag,
                serde_json::json!({ "command": body }),
                None,
            ),
            tool_scope(
                session,
                root,
                root,
                "end",
                "Bash",
                tag,
                serde_json::json!(body),
                None,
            ),
        ]
    };

    replay(&one_call("toolu_a", "first recording"), &into);
    replay(&one_call("toolu_b", "second recording"), &into);

    let written = manifests(&into);
    assert_eq!(
        written.len(),
        2,
        "two recordings are two manifests: {written:?}"
    );
    let all: String = written.iter().map(|p| decoded(&on_disk(p))).collect();
    assert!(
        all.contains("first recording") && all.contains("second recording"),
        "and neither replaced the other"
    );
    assert!(
        written
            .iter()
            .all(|p| p.to_string_lossy().contains(".unattributed")),
        "both still say what they are: {written:?}"
    );
}

/// Every node states the size of the content behind it, stored or not.
///
/// A `larger-than-ceiling` node used to say only that it was too big. A reader could not tell 8 KiB
/// from 8 GiB, could not tell whether raising the ceiling would recover the content or bury the
/// manifest, and could not audit the decision at all -- the one number that justified withholding
/// was the one number missing. It is `decide`'s own argument, so it was known where it was dropped.
///
/// Null only for content never established, which is the single case with no length to state rather
/// than a length deliberately not stored.
#[tokio::test]
async fn a_node_states_how_large_its_content_was() {
    let signer = Ed25519Signer::create().expect("a signer");
    let mut rec = Recorder::new(
        LineageSession::new(SignerType::ED25519(signer)),
        // Eight bytes, so a short string trips the ceiling and the test needs no megabyte.
        Policy::new(vec![".env*".into()], 8),
    );

    rec.observe_file(
        &seen("/at.txt", Some(b"12345678"), FileMode::Wrote),
        true,
        None,
    )
    .await
    .unwrap();
    rec.observe_file(
        &seen("/over.txt", Some(b"123456789"), FileMode::Wrote),
        true,
        None,
    )
    .await
    .unwrap();
    rec.observe_file(&seen("/unknown.txt", None, FileMode::Read), true, None)
        .await
        .unwrap();

    let manifest = exported(rec).await;
    let nodes: Vec<serde_json::Value> = blobs_of(&manifest)
        .into_iter()
        .filter_map(|(_, bytes)| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .filter(|value| value["assetType"] == "Document")
        .collect();

    let node = |path: &str| -> serde_json::Value {
        nodes
            .iter()
            .find(|value| value["filePath"] == path)
            .unwrap_or_else(|| panic!("a node for {path}"))
            .clone()
    };

    let at = node("/at.txt");
    assert_eq!(at["contentBytes"], 8, "a stored node states its length");
    assert_eq!(
        at["contentState"], "stored",
        "eight bytes is at the ceiling, not over it"
    );

    let over = node("/over.txt");
    assert_eq!(over["contentState"], "withheld");
    assert_eq!(
        over["contentBytes"], 9,
        "the size that tripped the ceiling is the point of recording it"
    );

    let unknown = node("/unknown.txt");
    assert_eq!(unknown["contentState"], "unknown");
    assert!(
        unknown["contentBytes"].is_null(),
        "content never established has no length to state: {unknown}"
    );
}

/// A compaction node is an `Entity`, because nothing backs the other claim.
///
/// It used to be the only node in the manifest carrying `provType: "Activity"`, and it carried it in
/// the metadata of a `DataRegistration`. An activity here *is* a `ComputationRegistration` -- that
/// statement holds the inputs, the outputs and the `performedBy` attribution -- so a consumer
/// enumerating activities never reached this node, and one trusting `provType` found an activity
/// with no statement behind it. `record_tool_run` refuses an empty computation for the same reason.
///
/// What is recorded is a marker: an ordinal, a phase and a timestamp. Phase 4's context snapshots
/// would earn the other type; asserting it without them did not.
#[tokio::test]
async fn a_compaction_is_an_entity_not_an_activity() {
    let mut rec = recorder();
    rec.record_compaction(CompactionPhase::Before, None)
        .await
        .unwrap();
    rec.record_compaction(CompactionPhase::After, None)
        .await
        .unwrap();

    let manifest = exported(rec).await;
    let nodes: Vec<serde_json::Value> = blobs_of(&manifest)
        .into_iter()
        .filter_map(|(_, bytes)| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .filter(|value| {
            value["name"]
                .as_str()
                .is_some_and(|name| name.contains("compaction"))
        })
        .collect();

    assert_eq!(nodes.len(), 2, "both halves of one compaction: {nodes:?}");
    for node in &nodes {
        assert_eq!(
            node["provType"], "Entity",
            "a marker is a thing, not a process: {node}"
        );
    }
    let names: Vec<&str> = nodes.iter().filter_map(|n| n["name"].as_str()).collect();
    assert!(
        names.contains(&"pre-compaction 1") && names.contains(&"post-compaction 1"),
        "one compaction, its two halves named for what they are: {names:?}"
    );
}

/// A call that never answered leaves a reason in the graph, not just a counter.
///
/// Its inputs are registered before the response is known and nothing links them afterwards, because
/// an activity with no output is not one. They used to dangle with the explanation living only in
/// `ModelCallWithoutResponse` at session level, so a reader inspecting the node saw no reason at all.
///
/// The reason is on a marker for the call rather than a field on those nodes, because the nodes are
/// content addressed and shared: one live session had a single `Model` node feeding 31 successful
/// calls and one failure, and writing "unlinked" onto it would have been false.
#[tokio::test]
async fn an_unanswered_model_call_says_why_its_inputs_dangle() {
    let mut rec = recorder();
    let recorded = rec
        .record_model_call(
            Some("some-model"),
            None,
            b"[{\"role\":\"user\",\"content\":\"quota\"}]",
            None, // no completion ever arrived
            None,
            serde_json::json!({}),
            serde_json::json!({}),
            true,
            None,
        )
        .await
        .expect("registering the inputs still succeeds");
    assert!(!recorded, "no computation is claimed without an output");

    let manifest = exported(rec).await;
    let nodes: Vec<serde_json::Value> = blobs_of(&manifest)
        .into_iter()
        .filter_map(|(_, bytes)| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .collect();

    let marker = nodes
        .iter()
        .find(|n| n["unlinkedBecause"] == "model-call-had-no-response")
        .unwrap_or_else(|| panic!("a marker naming the reason: {nodes:#?}"));
    assert_eq!(marker["provType"], "Entity", "a marker is a thing");

    // The prompt is reachable through the marker, which is the point of recording it.
    let prompt = nodes
        .iter()
        .find(|n| n["assetType"] == "Prompt")
        .expect("the prompt was registered before the response was known");
    let inputs = nested(marker, "unlinkedInputs");
    let listed = inputs
        .as_array()
        .expect("the inputs it could not link")
        .iter()
        .filter_map(|v| v.as_str())
        .any(|cid| cid == prompt["content-cid"].as_str().unwrap_or_default());
    assert!(
        listed,
        "the marker names the prompt it could not link: {marker}"
    );
}

/// Coverage states how many bytes were behind the nodes, split by what was done with them.
///
/// The counters answer "how many times", never "how much", and putting a size into that flat map
/// would have made `PayloadTooLarge: 70` even easier to read as a size than it already was. So the
/// totals are their own field, and they are in the metadata rather than the content because the
/// content is what this node is addressed by: two sessions that saw the same things should produce
/// the same coverage node, and the same session against a different ceiling sees exactly as much
/// while storing a different amount.
#[tokio::test]
async fn coverage_states_how_many_bytes_were_behind_the_nodes() {
    let signer = Ed25519Signer::create().expect("a signer");
    let mut rec = Recorder::new(
        LineageSession::new(SignerType::ED25519(signer)),
        Policy::new(vec![".env*".into()], 8),
    );

    // One of each disposition, with lengths chosen so the three totals cannot be confused.
    rec.observe_file(
        &seen("/at.txt", Some(b"12345678"), FileMode::Wrote),
        true,
        None,
    )
    .await
    .unwrap();
    rec.observe_file(
        &seen("/over.txt", Some(b"123456789"), FileMode::Wrote),
        true,
        None,
    )
    .await
    .unwrap();
    rec.observe_file(
        &seen("/.env", Some(b"secret\n"), FileMode::Read),
        true,
        None,
    )
    .await
    .unwrap();
    // No content established: nothing to attribute, and zero stored bytes would claim an empty file.
    rec.observe_file(&seen("/never.txt", None, FileMode::Read), true, None)
        .await
        .unwrap();

    let manifest = exported(rec).await;
    let coverage = blobs_of(&manifest)
        .into_iter()
        .filter_map(|(_, bytes)| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .find(|value| value["name"] == "coverage")
        .expect("a coverage node");

    // Scalars, not a nested object: the graph explorer renders each metadata value as a string, so
    // an object reaches a reader as `[object Object]`. It did, on a real run.
    assert_eq!(
        coverage["bytesStored"], 8,
        "the file inside the ceiling: {coverage}"
    );
    assert_eq!(coverage["bytesTooLarge"], 9, "the file over it: {coverage}");
    assert_eq!(
        coverage["bytesDenied"], 7,
        "the file the policy withheld: {coverage}"
    );
    for key in ["bytesStored", "bytesTooLarge", "bytesDenied"] {
        assert!(
            coverage[key].is_number(),
            "{key} renders as a number, not `[object Object]`: {coverage}"
        );
    }

    // The counters stay counts, and stay where they were.
    let counts = nested(&coverage, "coverage");
    assert_eq!(counts["FileWritten"], 2);
    assert_eq!(counts["ContentUnknown"], 1);
    assert!(
        counts["stored"].is_null(),
        "sizes do not leak into the counter map: {coverage}"
    );
}

/// Content never established contributes no bucket at all, not a zero one.
///
/// Separated from the test above because a zero is invisible beside a real total: attributing
/// `0` bytes there passed that test untouched. On its own it is the whole difference between "we
/// withheld nothing" and "there was nothing to withhold", which is the distinction `contentState`
/// exists to keep and the totals must not undo.
#[tokio::test]
async fn content_never_established_contributes_no_byte_total() {
    let signer = Ed25519Signer::create().expect("a signer");
    let mut rec = Recorder::new(
        LineageSession::new(SignerType::ED25519(signer)),
        Policy::new(vec![".env*".into()], 8),
    );
    rec.observe_file(&seen("/never.txt", None, FileMode::Read), true, None)
        .await
        .unwrap();

    let manifest = exported(rec).await;
    let coverage = blobs_of(&manifest)
        .into_iter()
        .filter_map(|(_, bytes)| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .find(|value| value["name"] == "coverage")
        .expect("a coverage node");

    for key in ["bytesStored", "bytesTooLarge", "bytesDenied"] {
        assert!(
            coverage[key].is_null(),
            "no content, so no total -- not `{key}: 0`: {coverage}"
        );
    }
    assert_eq!(
        nested(&coverage, "coverage")["ContentUnknown"],
        1,
        "it was still seen"
    );
}

// ------------------------------------------------------- announcing the manifest

/// A session with an agent, so the export is a real session rather than a fragment.
fn one_attributed_write(session: &str, root: &str) -> Vec<Event> {
    let mut events = vec![mark(
        session,
        root,
        root,
        "session.start",
        serde_json::json!({"model":"opus"}),
    )];
    events.extend([
        tool_scope(
            session,
            root,
            root,
            "start",
            "Write",
            "toolu_mark",
            serde_json::json!({ "file_path": "/work/announced.md", "content": "hi\n" }),
            None,
        ),
        tool_scope(
            session,
            root,
            root,
            "end",
            "Write",
            "toolu_mark",
            serde_json::json!("wrote 3 bytes\n"),
            Some("ok"),
        ),
    ]);
    events
}

#[test]
fn a_finished_recording_says_where_it_was_written() {
    // Without this the manifest is discoverable only by knowing our path convention, and a
    // convention is not an attestation. The mark is the one thing that puts the recording's
    // location inside the event stream it was recorded from.
    let into = TempDir::new().unwrap();
    let marks = replay(
        &one_attributed_write(
            "01a040aa-0000-0000-0000-000000000fa1",
            "01a040aa-0000-0000-0000-000000000fa2",
        ),
        &into,
    );

    let written = manifests(&into);
    assert_eq!(written.len(), 1, "one session, one manifest: {written:?}");
    assert_eq!(marks.len(), 1, "and one mark for it: {marks:?}");
    let mark = &marks[0];
    assert_eq!(mark.path, written[0], "the mark names the file on disk");
    assert!(!mark.unattributed, "this session registered an agent");

    let manifest: Manifest =
        serde_json::from_slice(&fs::read(&written[0]).expect("read")).expect("parse");
    assert_eq!(
        mark.statements,
        manifest.statements.len(),
        "and counts what is in it"
    );
    assert!(mark.statements > 0, "a recording with nothing in it");
}

#[test]
fn the_announced_cid_is_the_cid_of_the_bytes_on_disk() {
    // A CID over anything else -- the statements, the graph, a re-serialization -- cannot answer the
    // only question the mark exists for: is the file I just fetched the file that was announced?
    let into = TempDir::new().unwrap();
    let marks = replay(
        &one_attributed_write(
            "01a040aa-0000-0000-0000-000000000fa3",
            "01a040aa-0000-0000-0000-000000000fa4",
        ),
        &into,
    );

    let bytes = fs::read(&marks[0].path).expect("the announced path is readable");
    assert_eq!(
        marks[0].cid,
        blake3_cid_raw_binary(&bytes).expect("a cid over the file"),
        "the announced CID must verify against the file, byte for byte"
    );
}

#[test]
fn a_checkpoint_is_not_announced() {
    // Checkpoints are superseded by the next one, so a mark per checkpoint would put one on the
    // stream per unit of work and leave a consumer to guess which named the finished recording --
    // and the CID of a manifest that is still growing identifies nothing worth holding onto.
    let into = TempDir::new().unwrap();
    let session = "01a040aa-0000-0000-0000-000000000fa5";
    let root = "01a040aa-0000-0000-0000-000000000fa6";
    let mut events = vec![mark(
        session,
        root,
        root,
        "session.start",
        serde_json::json!({"model":"opus"}),
    )];
    for (index, tag) in ["toolu_c1", "toolu_c2", "toolu_c3"].iter().enumerate() {
        events.extend([
            tool_scope(
                session,
                root,
                root,
                "start",
                "Write",
                tag,
                serde_json::json!({ "file_path": format!("/work/{index}.md"), "content": "x\n" }),
                None,
            ),
            tool_scope(
                session,
                root,
                root,
                "end",
                "Write",
                tag,
                serde_json::json!("wrote 2 bytes\n"),
                Some("ok"),
            ),
        ]);
    }

    let marks = replay(&events, &into);

    assert_eq!(
        marks.len(),
        1,
        "three completed tool calls checkpoint three times and finish once: {marks:?}"
    );
}

#[test]
fn an_agentless_fragment_announces_itself_as_one() {
    // Codex's title-generation call exports as `{id}.unattributed.json`, and a consumer counting
    // manifests to count sessions gets the wrong answer. The flag is in the mark's data so that
    // does not require parsing a filename -- which is the mistake the live-session script made.
    let into = TempDir::new().unwrap();
    let session = "01a040aa-0000-0000-0000-000000000fa7";
    let root = "01a040aa-0000-0000-0000-000000000fa8";
    // No `session.start`, so no agent is ever registered.
    let marks = replay(
        &[
            tool_scope(
                session,
                root,
                root,
                "start",
                "Bash",
                "toolu_frag",
                serde_json::json!({ "command": "title this" }),
                None,
            ),
            tool_scope(
                session,
                root,
                root,
                "end",
                "Bash",
                "toolu_frag",
                serde_json::json!("a title"),
                None,
            ),
        ],
        &into,
    );

    assert_eq!(marks.len(), 1, "a fragment is still announced: {marks:?}");
    assert!(
        marks[0].unattributed,
        "and says it is one, rather than being counted as a session"
    );
    assert!(
        marks[0]
            .path
            .to_string_lossy()
            .ends_with(".unattributed.json"),
        "naming the fragment it wrote: {:?}",
        marks[0].path
    );
    assert_eq!(marks[0].path, manifests(&into)[0]);
}

// ------------------------------------------------------- rendering in the explorer

#[test]
fn no_metadata_value_is_a_bare_object_or_array() {
    // The graph explorer displays each metadata value as a string, so an object or an array renders
    // as the literal text `[object Object]`. This shipped three times before it was fixed as a
    // class: as `contentBytes` (a `{stored, denied}` map), then as the coverage counts, then as a
    // model call's `usage` -- thirteen nodes a session, on a manifest a reader was meant to read.
    //
    // Asserted over every node of a full session rather than per field, because the bug is not any
    // one field: it is that nothing stopped the next one.
    let into = TempDir::new().unwrap();
    let session = "01a040aa-0000-0000-0000-000000000fb1";
    let root = "01a040aa-0000-0000-0000-000000000fb2";
    let mut events = vec![mark(
        session,
        root,
        root,
        "session.start",
        serde_json::json!({"model":"opus"}),
    )];
    events.extend([
        tool_scope(
            session,
            root,
            root,
            "start",
            "Read",
            "toolu_fb",
            serde_json::json!({ "file_path": "/app/.env" }),
            None,
        ),
        tool_scope(
            session,
            root,
            root,
            "end",
            "Read",
            "toolu_fb",
            serde_json::json!("SECRET=x\n"),
            Some("ok"),
        ),
    ]);
    replay(&events, &into);

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifests(&into)[0]).unwrap()).unwrap();
    let mut offenders: Vec<String> = Vec::new();
    for (_, bytes) in blobs_of(&manifest) {
        let Ok(node) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        let Some(fields) = node.as_object() else {
            continue;
        };
        // The coverage node's *content* is a bare object and must stay one: it is what the node is
        // addressed by, so two sessions that saw the same things produce the same node. Only
        // metadata is displayed field by field, and metadata always names what it describes.
        if !fields.contains_key("assetType") {
            continue;
        }
        for (key, value) in fields {
            if value.is_object() || value.is_array() {
                offenders.push(format!("{key} = {value}"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "these would render as `[object Object]`: {offenders:#?}"
    );
}

#[test]
fn a_nested_metadata_value_survives_being_encoded() {
    // Encoding for display must not cost a reader the values. The string is JSON, so the structure
    // is still there for anyone who parses it -- which is the difference between this and
    // flattening or dropping the field.
    let into = TempDir::new().unwrap();
    let marks = replay(
        &one_attributed_write(
            "01a040aa-0000-0000-0000-000000000fb3",
            "01a040aa-0000-0000-0000-000000000fb4",
        ),
        &into,
    );

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&marks[0].path).unwrap()).unwrap();
    let coverage = blobs_of(&manifest)
        .into_iter()
        .filter_map(|(_, bytes)| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .find(|value| value["name"] == "coverage")
        .expect("a coverage node");

    let counts = nested(&coverage, "coverage");
    assert_eq!(
        counts["FileWritten"], 1,
        "the counts are still readable: {counts}"
    );
    // And the content is untouched, because that is what the node's identity is over.
    let body = blobs_of(&manifest)
        .into_iter()
        .find(|(cid, _)| Some(cid.as_str()) == coverage["content-cid"].as_str())
        .map(|(_, bytes)| serde_json::from_slice::<serde_json::Value>(&bytes).expect("json"))
        .expect("the coverage content");
    assert!(
        body.is_object() && body["FileWritten"] == 1,
        "the content stays a real object: {body}"
    );
}

#[test]
fn the_plugins_own_manifest_mark_is_not_recorded_as_lineage() {
    // Export emits an `eqty.manifest` mark on the host's event stream, and this plugin subscribes
    // to that stream -- so the announcement comes back as an event. It must classify to nothing.
    //
    // Today it does, because `classify_mark` matches `session.start` by name and everything else by
    // `hook_event_name`, which our mark does not carry. That is a property of the current match
    // arms rather than a decision anyone recorded, and the next name-based arm could quietly turn a
    // recording into one that records its own announcements -- growing a session that has already
    // been exported, and on Codex re-opening one at teardown.
    let announced = mark(
        "01a040aa-0000-0000-0000-000000000fc1",
        "01a040aa-0000-0000-0000-000000000fc2",
        "01a040aa-0000-0000-0000-000000000fc3",
        "eqty.manifest",
        serde_json::json!({
            "cid": "bafkr4iaq4gqw6cafxvysrux7cete77jtkzxqvuozqdv5b4vtqsebluxhfm",
            "path": "/work/.eqty/manifests/session.json",
            "statements": 300,
            "unattributed": false,
        }),
    );

    assert!(
        classify(&announced).is_none(),
        "the plugin's own announcement is not lineage"
    );
}
