//! Defects a review found in the recorder, each pinned so it cannot come back.
//!
//! Two themes run through them, and both are worse than a missing feature. One is a **manifest that
//! states something false**: a content CID for a version the file never held, a rejected write
//! recorded as a write. The other is a **redaction that did not redact**: the file node said
//! `withheld` while the tool call beside it carried the bytes in full.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use eqty_lineage_nemo_relay::{
    EditAttempt, FileMode, FileObserved, LineageSession, Mailbox, Policy, Recorder, ReplayRefusal,
    SessionFinished, SessionRouter, SignerFactory, apply_edit, apply_line_edit, classify,
    file_events_from_patch,
};
use integrity::lineage::models::manifest::Manifest;
use integrity::signer::{SignerType, ed25519_signer::Ed25519Signer};
use nemo_relay_plugin::Event;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

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

fn replay(events: &[Event], into: &TempDir) {
    let router = Arc::new(Mutex::new(SessionRouter::new()));
    let mailbox = Mailbox::start(
        into.path().to_path_buf(),
        policy(),
        signer_factory(),
        forget_with(&router),
    );
    for event in events {
        let Some(session_id) = router.lock().expect("lock").attribute(event) else {
            continue;
        };
        if let Some(lineage) = classify(event) {
            mailbox.send(&session_id, event.timestamp().to_rfc3339(), lineage);
        }
    }
    drop(mailbox);
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
