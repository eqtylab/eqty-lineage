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
        vacated: None,
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
        unique_only: false,
        line_oriented: false,
        replay_from: None,
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
        unique_only: false,
        line_oriented: false,
        replay_from: None,
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

#[tokio::test]
async fn one_file_at_two_paths_is_one_node() {
    // Identity is content; the path is metadata. A file copied or moved is the same bytes, so it is
    // one node with two things said about it -- not two nodes that no reader can join.
    let mut rec = recorder();
    let here = rec
        .observe_file(
            &seen("/src/config.toml", Some(b"port = 8080\n"), FileMode::Read),
            true,
            None,
        )
        .await
        .unwrap();
    let there = rec
        .observe_file(
            &seen(
                "/backup/config.toml",
                Some(b"port = 8080\n"),
                FileMode::Read,
            ),
            true,
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        here, there,
        "the same bytes are the same node, wherever they sit"
    );
}

#[tokio::test]
async fn one_withheld_file_at_two_paths_is_one_node() {
    // The case that actually broke: content we refuse to store is stood in for by a canonical
    // descriptor, and hashing the path into that descriptor split exactly the nodes -- secrets and
    // large artifacts -- where knowing two recordings saw the same bytes is worth the most.
    let mut rec = recorder();
    let here = rec
        .observe_file(
            &seen("/app/.env", Some(b"TOKEN=aaa\n"), FileMode::Read),
            true,
            None,
        )
        .await
        .unwrap();
    let there = rec
        .observe_file(
            &seen("/deploy/.env", Some(b"TOKEN=aaa\n"), FileMode::Read),
            true,
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        here, there,
        "a withheld file is identified by the bytes it withheld, not by where it sat"
    );
}

#[tokio::test]
async fn a_file_never_read_stays_distinct_per_path() {
    // The deliberate exception. With no content established there is nothing else to be identical
    // about, so identity falls back to `unknown:{path}`. Two files nobody read cannot be shown to be
    // the same file, and claiming they are would be an assertion the recording never observed.
    let mut rec = recorder();
    let here = rec
        .observe_file(&seen("/src/a.txt", None, FileMode::Read), true, None)
        .await
        .unwrap();
    let there = rec
        .observe_file(&seen("/src/b.txt", None, FileMode::Read), true, None)
        .await
        .unwrap();
    assert_ne!(
        here, there,
        "unread files must not be merged on absence alone"
    );
}

#[tokio::test]
async fn withheld_and_unknown_are_different_claims() {
    // A node reporting both as `redacted` tells a reader the wrong one. A withheld file was seen and
    // its bytes deliberately kept out; an unknown one was never established, so there was nothing to
    // withhold. Codex makes this constant rather than occasional: every `apply_patch` `Update File`
    // carries hunks and no post-image, so every one lands as unknown.
    let mut rec = recorder();
    rec.observe_file(
        &seen("/app/notes.md", Some(b"visible\n"), FileMode::Read),
        true,
        None,
    )
    .await
    .unwrap();
    rec.observe_file(
        &seen("/app/.env", Some(b"TOKEN=aaa\n"), FileMode::Read),
        true,
        None,
    )
    .await
    .unwrap();
    rec.observe_file(&seen("/app/patched.md", None, FileMode::Wrote), true, None)
        .await
        .unwrap();

    let json = serde_json::to_value(rec.finish(None).await.expect("manifest")).unwrap();
    let mut states = std::collections::HashMap::new();
    for blob in json["blobs"].as_object().expect("blobs").values() {
        use base64::Engine as _;
        let Ok(raw) =
            base64::engine::general_purpose::STANDARD.decode(blob.as_str().unwrap_or_default())
        else {
            continue;
        };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&raw) else {
            continue;
        };
        if value["assetType"] == "Document" {
            states.insert(
                value["name"].as_str().unwrap_or_default().to_string(),
                (
                    value["contentState"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    value["redacted"].as_bool().unwrap_or(false),
                ),
            );
        }
    }
    assert_eq!(
        states.get("/app/notes.md"),
        Some(&("stored".to_string(), false))
    );
    assert_eq!(
        states.get("/app/.env"),
        Some(&("withheld".to_string(), true)),
        "policy withholding is redaction"
    );
    assert_eq!(
        states.get("/app/patched.md"),
        Some(&("unknown".to_string(), false)),
        "content never established is not redaction -- nothing was withheld"
    );
}

#[tokio::test]
async fn an_ambiguous_patch_hunk_is_refused_rather_than_guessed() {
    // Codex emits `apply_patch` hunks with no context lines -- the patch that prompted this was
    // `-two` / `+TWO` and nothing else. Replaying that against a file containing "two" twice would
    // rewrite the first occurrence and content-address a version the file never had.
    let mut rec = recorder();
    rec.observe_file(
        &seen("/work/report.md", Some(b"two\none\ntwo\n"), FileMode::Wrote),
        true,
        None,
    )
    .await
    .unwrap();

    let mut update = seen("/work/report.md", None, FileMode::Wrote);
    update.edit = Some(EditAttempt {
        old: "two".into(),
        new: "TWO".into(),
        replace_all: false,
        unique_only: true,
        line_oriented: false,
        replay_from: None,
    });
    rec.observe_file(&update, true, None).await.unwrap();

    let stats = rec.stats();
    assert_eq!(
        stats.get("EditTooAmbiguousToReplay"),
        Some(&1),
        "an ambiguous hunk must be refused and counted: {stats:?}"
    );
    assert!(
        !stats.contains_key("ContentRecovered"),
        "and must not be replayed: {stats:?}"
    );
}

#[tokio::test]
async fn an_unambiguous_patch_hunk_is_replayed() {
    // The guard is uniqueness, not patches. When the removed text occurs once the replay is exact,
    // which is the case that makes Codex's edits recordable at all.
    let mut rec = recorder();
    rec.observe_file(
        &seen(
            "/work/report.md",
            Some(b"one\ntwo\nthree\n"),
            FileMode::Wrote,
        ),
        true,
        None,
    )
    .await
    .unwrap();

    let mut update = seen("/work/report.md", None, FileMode::Wrote);
    update.edit = Some(EditAttempt {
        old: "two".into(),
        new: "TWO".into(),
        replace_all: false,
        unique_only: true,
        line_oriented: false,
        replay_from: None,
    });
    rec.observe_file(&update, true, None).await.unwrap();

    assert_eq!(
        rec.stats().get("ContentRecovered"),
        Some(&1),
        "a unique hunk replays: {:?}",
        rec.stats()
    );
}

#[tokio::test]
async fn an_actor_named_like_json_cannot_forge_its_own_description() {
    // The descriptor is the node's content, and it was built by interpolating a host-supplied name
    // into a string: `format!(r#"{{"kind":"{kind}","name":"{name}"}}"#)`. A tool named
    // `a","kind":"Model` therefore produced bytes that parse -- and parse to `kind: Model`, because a
    // duplicate JSON key takes the last value. The node described itself as a model.
    //
    // Identity was never the exposure: `kind` is our own literal and comes first, so no two
    // `(kind, name)` pairs can produce the same bytes. What was exposed was the claim inside.
    let mut rec = recorder();
    let hostile = r#"a","kind":"Model"#;
    rec.record_actor(
        "Tool",
        hostile,
        "A tool with an awkward name.",
        serde_json::json!({}),
        None,
    )
    .await
    .expect("the actor registers");

    let decoded = decoded_blobs(rec.finish(None).await.expect("the manifest exports"));
    // Find the descriptor blob and parse it as a reader would.
    let descriptor: serde_json::Value = decoded
        .split_inclusive('}')
        .filter_map(|chunk| chunk.rfind('{').map(|at| &chunk[at..]))
        .filter_map(|chunk| serde_json::from_str::<serde_json::Value>(chunk).ok())
        .find(|value| value.get("kind").is_some() && value.get("name").is_some())
        .expect("the actor descriptor is in the manifest and parses");

    assert_eq!(
        descriptor["kind"], "Tool",
        "the kind is what we said it was, not what the name claimed: {descriptor}"
    );
    assert_eq!(
        descriptor["name"], hostile,
        "and the name survives verbatim"
    );
}

#[tokio::test]
async fn an_ordinary_actor_keeps_the_identity_it_always_had() {
    // The guard on the fix above. Going through the serializer must not renumber every actor node
    // ever recorded: `serde_json`'s map is sorted and `kind` sorts before `name`, so for a name that
    // needs no escaping the bytes are exactly what the old `format!` produced. Every manifest already
    // on disk still joins with every manifest written from here on.
    let mut rec = recorder();
    rec.record_actor(
        "Tool",
        "Read",
        "The Read tool.",
        serde_json::json!({}),
        None,
    )
    .await
    .expect("the actor registers");

    let decoded = decoded_blobs(rec.finish(None).await.expect("the manifest exports"));
    assert!(
        decoded.contains(r#"{"kind":"Tool","name":"Read"}"#),
        "the descriptor bytes must be unchanged:\n{decoded}"
    );
}

#[tokio::test]
async fn a_denied_payload_does_not_report_itself_as_too_large() {
    // Both dispositions withhold the bytes, and both were counted as `PayloadTooLarge`. The deny
    // list is matched against the payload's *name* -- there is no path here -- so a tool called
    // `get_credentials` has its arguments withheld by the default `*credentials*` glob, and coverage
    // then said the payload was oversized. A reader raising the size ceiling to recover it would find
    // nothing changed, because size was never the reason.
    // `*credentials*` is one of the shipped defaults, which is what makes this the ordinary case
    // rather than a contrived one.
    let signer = Ed25519Signer::create().expect("a signer");
    let mut rec = Recorder::new(
        LineageSession::new(SignerType::ED25519(signer)),
        Policy::new(vec!["*credentials*".into()], 1_048_576),
    );
    rec.register_payload(
        "Dataset",
        "get_credentials input",
        "What the tool was invoked with.",
        b"{\"account\":\"acme\"}",
        serde_json::json!({}),
        None,
    )
    .await
    .expect("the payload registers as a node either way");

    assert_eq!(
        rec.stats().get("PayloadDenied"),
        Some(&1),
        "withheld by policy, and said so: {:?}",
        rec.stats()
    );
    assert_eq!(
        rec.stats().get("PayloadTooLarge"),
        None,
        "and not blamed on a ceiling it never hit"
    );

    let decoded = decoded_blobs(rec.finish(None).await.expect("the manifest exports"));
    assert!(
        decoded.contains("\"withheldBecause\":\"denied-by-policy\""),
        "the reason travels with the node, not only in coverage:\n{decoded}"
    );
}

#[tokio::test]
async fn a_payload_over_the_ceiling_still_reports_its_size() {
    // The complement: the counter must still distinguish the size case, or the fix above has just
    // moved the confusion.
    let mut rec = recorder();
    let oversized = vec![b'x'; 2048]; // the test policy's ceiling is 1024
    rec.register_payload(
        "Dataset",
        "Bash result",
        "What the tool returned.",
        &oversized,
        serde_json::json!({}),
        None,
    )
    .await
    .expect("the payload registers");

    assert_eq!(rec.stats().get("PayloadTooLarge"), Some(&1));
    assert_eq!(rec.stats().get("PayloadDenied"), None);
}

#[tokio::test]
async fn a_contentless_write_moves_the_replay_base_even_when_its_node_already_exists() {
    // A truncated read and a contentless write both key on `unknown:{path}`, so the write
    // deduplicates against the read's node. The read correctly leaves the replay base alone -- it
    // changed nothing on disk -- but the write must still invalidate it, or the next edit replays
    // against pre-write bytes.
    let mut rec = recorder();
    rec.observe_file(&seen("/a.py", None, FileMode::Read), true, None)
        .await
        .unwrap();
    rec.observe_file(&seen("/a.py", Some(b"x = 1\n"), FileMode::Read), true, None)
        .await
        .unwrap();
    rec.observe_file(&seen("/a.py", None, FileMode::Wrote), true, None)
        .await
        .unwrap();

    let mut edit = seen("/a.py", None, FileMode::Wrote);
    edit.edit = Some(EditAttempt {
        old: "x = 1".into(),
        new: "x = 2".into(),
        replace_all: false,
        unique_only: false,
        line_oriented: false,
        replay_from: None,
    });
    rec.observe_file(&edit, true, None).await.unwrap();

    assert_eq!(
        rec.stats().get("ContentRecovered"),
        None,
        "an edit must not replay against bytes an earlier write already replaced: {:?}",
        rec.stats()
    );
}

#[tokio::test]
async fn a_truncated_read_still_leaves_the_replay_base_standing() {
    // The guard on the test above: the fix must invalidate on writes, not on everything contentless.
    // A read that established nothing changed nothing, so what the session already holds is still
    // the file -- and replaying against it is the recovery this recorder exists to do.
    let mut rec = recorder();
    rec.observe_file(&seen("/a.py", Some(b"x = 1\n"), FileMode::Read), true, None)
        .await
        .unwrap();
    rec.observe_file(&seen("/a.py", None, FileMode::Read), true, None)
        .await
        .unwrap();

    let mut edit = seen("/a.py", None, FileMode::Wrote);
    edit.edit = Some(EditAttempt {
        old: "x = 1".into(),
        new: "x = 2".into(),
        replace_all: false,
        unique_only: false,
        line_oriented: false,
        replay_from: None,
    });
    rec.observe_file(&edit, true, None).await.unwrap();

    assert_eq!(
        rec.stats().get("ContentRecovered"),
        Some(&1),
        "a truncated read must not discard a base that is still valid: {:?}",
        rec.stats()
    );
}
