//! The identity rules a lineage graph rests on.
//!
//! Each test here corresponds to a distinction that is invisible until the case that motivates it
//! shows up, at which point the graph asserts something false. They are the reason the recorder is a
//! port rather than a rewrite.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use eqty_lineage_nemo_relay::{
    EditAttempt, FileMode, FileObserved, LineageSession, Policy, Recorder,
};
use integrity::signer::{SignerType, ed25519_signer::Ed25519Signer};

fn recorder() -> Recorder {
    let signer = Ed25519Signer::create().expect("a signer");
    Recorder::new(
        LineageSession::new(SignerType::ED25519(signer)),
        Policy::new(
            vec![".env*".into(), "*.pem".into(), "*/.ssh/*".into()],
            1024,
        ),
    )
}

/// Every blob in a manifest, base64-decoded and concatenated.
///
/// Blobs are base64, so a substring search over the raw manifest JSON finds nothing and any
/// assertion built on one would pass or fail for the wrong reason.
fn decoded_blobs(manifest: integrity::lineage::models::manifest::Manifest) -> String {
    let json = serde_json::to_value(&manifest).expect("the manifest serializes");
    json["blobs"]
        .as_object()
        .expect("a blobs map")
        .values()
        .filter_map(|blob| blob.as_str())
        .filter_map(|blob| BASE64.decode(blob).ok())
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .collect()
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

#[tokio::test]
async fn the_same_bytes_at_the_same_path_are_one_node() {
    let mut rec = recorder();
    let first = rec
        .observe_file(&seen("/a.md", Some(b"x\n"), FileMode::Read), true, None)
        .await
        .unwrap();
    let again = rec
        .observe_file(&seen("/a.md", Some(b"x\n"), FileMode::Read), true, None)
        .await
        .unwrap();
    assert_eq!(
        first, again,
        "re-reading an unchanged file must not mint a version"
    );
}

#[tokio::test]
async fn different_bytes_at_the_same_path_are_different_nodes() {
    // Keying on path alone would make `read -> edit -> read` either hide the edit or produce a node
    // that is both input and output of the same activity.
    let mut rec = recorder();
    let before = rec
        .observe_file(&seen("/a.md", Some(b"x\n"), FileMode::Read), true, None)
        .await
        .unwrap();
    let after = rec
        .observe_file(&seen("/a.md", Some(b"y\n"), FileMode::Wrote), true, None)
        .await
        .unwrap();
    assert_ne!(before, after);
}

#[tokio::test]
async fn the_same_bytes_at_different_paths_are_one_asset() {
    // Content addressing means the bytes decide identity and the path is metadata. This is the same
    // property `eqty-lineage-deepagents@0.2.0` shipped, and the whole point of a content-addressed
    // graph: the identical file appearing in two places is one node that both places point at.
    //
    // `(path, content)` is the recorder's *dedup* key, not the asset's identity -- it decides whether
    // a new version record is warranted, which is why both paths below still get their own version
    // and their own metadata statement over the shared asset.
    let mut rec = recorder();
    let one = rec
        .observe_file(&seen("/a.md", Some(b"same\n"), FileMode::Read), true, None)
        .await
        .unwrap();
    let two = rec
        .observe_file(&seen("/b.md", Some(b"same\n"), FileMode::Read), true, None)
        .await
        .unwrap();
    assert_eq!(
        one, two,
        "identical bytes are one asset, wherever they live"
    );
    assert_eq!(
        rec.stats().get("FileRead"),
        Some(&2),
        "but each path is still its own observation"
    );
}

#[tokio::test]
async fn a_file_never_established_does_not_dedupe_against_another() {
    // Two different paths whose content was never established must stay distinct: `unknown:` is
    // scoped to the path precisely so a failed read of one file cannot merge with another.
    let mut rec = recorder();
    let one = rec
        .observe_file(&seen("/a.md", None, FileMode::Read), false, None)
        .await
        .unwrap();
    let two = rec
        .observe_file(&seen("/b.md", None, FileMode::Read), false, None)
        .await
        .unwrap();
    assert_ne!(one, two);

    // Distinct assets are not enough. `content-cid` is what a *reader* uses to decide whether two
    // nodes are the same thing, and an unscoped `unknown` would tell them a failed read of one file
    // and a failed read of another are the same unestablished content.
    let decoded = decoded_blobs(rec.finish(None).await.unwrap());
    assert!(
        decoded.contains("unknown:/a.md"),
        "content-cid must name the path it could not establish"
    );
    assert!(
        decoded.contains("unknown:/b.md"),
        "content-cid must name the path it could not establish"
    );
}

#[tokio::test]
async fn an_edit_is_replayed_against_what_the_session_already_knows() {
    // `originalFile` is null on most Edit results, so without this chain the post-state of most
    // edits is simply unknown. The Python recorder recovered 5,863 of 11,658 versions this way.
    let mut rec = recorder();
    rec.observe_file(&seen("/a.py", Some(b"x = 1\n"), FileMode::Read), true, None)
        .await
        .unwrap();

    let mut edit = seen("/a.py", None, FileMode::Wrote);
    edit.edit = Some(EditAttempt {
        old: "x = 1".into(),
        new: "x = 2".into(),
        replace_all: false,
    });
    rec.observe_file(&edit, true, None).await.unwrap();

    assert_eq!(
        rec.stats().get("ContentRecovered"),
        Some(&1),
        "the replacement should have been replayed: {:?}",
        rec.stats()
    );
}

#[tokio::test]
async fn a_replay_against_the_wrong_pre_image_is_refused() {
    // The guard that keeps a stale pre-image from minting a version the file never had.
    let mut rec = recorder();
    rec.observe_file(
        &seen("/a.py", Some(b"totally different\n"), FileMode::Read),
        true,
        None,
    )
    .await
    .unwrap();

    let mut edit = seen("/a.py", None, FileMode::Wrote);
    edit.edit = Some(EditAttempt {
        old: "x = 1".into(),
        new: "x = 2".into(),
        replace_all: false,
    });
    rec.observe_file(&edit, true, None).await.unwrap();

    assert_eq!(
        rec.stats().get("ContentRecovered"),
        None,
        "a replacement that does not occur in the held content must not be replayed"
    );
}

#[tokio::test]
async fn a_denied_file_is_still_a_node() {
    // Provenance does not require publication. Dropping the node would make the manifest silently
    // incomplete -- a reader could not tell "never touched .env" from "touched it and we hid that".
    let mut rec = recorder();
    let asset = rec
        .observe_file(
            &seen("/srv/.env.production", Some(b"SECRET=1\n"), FileMode::Read),
            true,
            None,
        )
        .await
        .unwrap();

    assert!(
        asset.is_some(),
        "a denied file must still appear in the graph"
    );
    assert_eq!(
        rec.stats().get("ContentDenied"),
        Some(&1),
        "{:?}",
        rec.stats()
    );
}

#[tokio::test]
async fn two_different_secrets_at_one_path_stay_two_versions() {
    // The reason identity is computed before redaction rather than after: both of these scrub to the
    // same withheld node if you hash what you stored instead of what you saw.
    let mut rec = recorder();
    let one = rec
        .observe_file(
            &seen("/app/.env", Some(b"TOKEN=aaa\n"), FileMode::Read),
            true,
            None,
        )
        .await
        .unwrap();
    let two = rec
        .observe_file(
            &seen("/app/.env", Some(b"TOKEN=bbb\n"), FileMode::Read),
            true,
            None,
        )
        .await
        .unwrap();
    assert_ne!(
        one, two,
        "identity must come from the original bytes, not the withheld stand-in"
    );
}

#[tokio::test]
async fn a_withheld_file_has_the_same_identity_in_two_recordings() {
    // Determinism matters even for content we do not store: two runs that both read the same secret
    // should join on that node rather than producing an unjoinable pair.
    let first = {
        let mut rec = recorder();
        rec.observe_file(
            &seen("/app/.env", Some(b"TOKEN=aaa\n"), FileMode::Read),
            true,
            None,
        )
        .await
        .unwrap()
    };
    let second = {
        let mut rec = recorder();
        rec.observe_file(
            &seen("/app/.env", Some(b"TOKEN=aaa\n"), FileMode::Read),
            true,
            None,
        )
        .await
        .unwrap()
    };
    assert_eq!(first, second);
}

#[tokio::test]
async fn content_over_the_ceiling_is_withheld_but_recorded() {
    let mut rec = recorder();
    let big = vec![b'x'; 2048];
    let asset = rec
        .observe_file(&seen("/big.bin", Some(&big), FileMode::Read), true, None)
        .await
        .unwrap();
    assert!(asset.is_some());
    assert_eq!(
        rec.stats().get("ContentTooLarge"),
        Some(&1),
        "{:?}",
        rec.stats()
    );
}

#[tokio::test]
async fn a_run_with_no_outputs_is_not_recorded() {
    let mut rec = recorder();
    let input = rec
        .observe_file(&seen("/in.md", Some(b"in\n"), FileMode::Read), true, None)
        .await
        .unwrap()
        .unwrap();
    let recorded = rec
        .record_tool_run(&[input], &[], serde_json::json!({}), None)
        .await
        .unwrap();
    assert!(!recorded);
    assert_eq!(rec.stats().get("ActivityWithoutOutputs"), Some(&1));
}

#[tokio::test]
async fn a_session_exports_a_manifest_stating_its_own_coverage() {
    let mut rec = recorder();
    rec.record_actor(
        "Agent",
        "claude-code",
        "The coding agent.",
        serde_json::json!({}),
        None,
    )
    .await
    .unwrap();
    let input = rec
        .observe_file(&seen("/in.md", Some(b"in\n"), FileMode::Read), true, None)
        .await
        .unwrap()
        .unwrap();
    let output = rec
        .observe_file(
            &seen("/out.md", Some(b"out\n"), FileMode::Wrote),
            true,
            None,
        )
        .await
        .unwrap()
        .unwrap();
    rec.record_tool_run(&[input], &[output], serde_json::json!({}), None)
        .await
        .unwrap();

    let decoded = decoded_blobs(rec.finish(None).await.expect("the manifest exports"));

    assert!(
        decoded.contains("coverage"),
        "the manifest must state what it did not see, inside the graph"
    );
    assert!(
        decoded.contains("FileWritten"),
        "coverage should carry the counts a reader needs to weigh the graph"
    );
}
