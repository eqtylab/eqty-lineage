//! Building a lineage graph, checked without a baseline to diff against.
//!
//! There is no second implementation to compare with -- phase 1 was skipped, and the Python recorder
//! is deliberately not a dependency of this crate. So these assert properties that hold on their own
//! terms: that a manifest is internally consistent, that asset identity is content identity, and
//! that the things which must not be silently defaulted are not.

use eqty_lineage_nemo_relay::LineageSession;
use integrity::signer::{SignerType, ed25519_signer::Ed25519Signer};
use serde_json::json;

fn session() -> LineageSession {
    let signer = Ed25519Signer::create().expect("a signer");
    LineageSession::new(SignerType::ED25519(signer))
}

fn metadata(name: &str) -> serde_json::Value {
    json!({ "name": name, "assetType": "Document" })
}

#[tokio::test]
async fn a_session_produces_a_manifest() {
    let mut lineage = session();
    let input = lineage
        .register_content(b"# Report\n", metadata("report.md"), None)
        .await
        .expect("content registers");
    let output = lineage
        .register_content(b"# Summary\n", metadata("summary.md"), None)
        .await
        .expect("content registers");
    lineage
        .record_computation(&[input], &[output], None)
        .await
        .expect("computation records");

    let manifest = lineage.into_manifest().await.expect("manifest generates");
    let json = serde_json::to_value(&manifest).expect("manifest serializes");

    assert_eq!(json["version"], "3");
    assert!(
        json["statements"]
            .as_object()
            .is_some_and(|s| !s.is_empty()),
        "a manifest with no statements attests nothing"
    );
}

#[tokio::test]
async fn an_assets_identity_is_its_contents_hash() {
    // The property the whole graph rests on: the same bytes are the same node, in any session, on
    // any machine. Two independent sessions, no shared state, same content.
    let content = b"identical bytes\n";

    let mut first = session();
    let one = first
        .register_content(content, metadata("a.md"), None)
        .await
        .unwrap();
    let mut second = session();
    let two = second
        .register_content(content, metadata("b.md"), None)
        .await
        .unwrap();

    assert_eq!(one, two, "identical content must be one asset, not two");
    assert!(
        one.as_str().starts_with("bafkr4i"),
        "expected a raw-binary blake3 CID, got {one}"
    );
}

#[tokio::test]
async fn binary_content_is_hashed_as_given() {
    // The Python path decodes with `errors="replace"` before hashing, so for non-UTF-8 content the
    // manifest asserts a content-cid that does not match the asset's own CID. Hashing the bytes
    // directly is the fix, and this is the test that would catch a regression back to it: the two
    // byte strings below differ only in bytes that a lossy decode collapses to the same replacement
    // character.
    let mut lineage = session();
    let one = lineage
        .register_content(&[0xff, 0xfe, 0x00, 0x01], metadata("a.bin"), None)
        .await
        .unwrap();
    let two = lineage
        .register_content(&[0xff, 0xfd, 0x00, 0x01], metadata("b.bin"), None)
        .await
        .unwrap();

    assert_ne!(one, two, "two different binaries collapsed to one asset");
}

#[tokio::test]
async fn metadata_bytes_travel_with_the_manifest() {
    // A metadata statement stores only a CID of the canonicalized metadata. If those bytes are not
    // put in the blob map, the manifest carries a reference that resolves to nothing -- and it still
    // parses, still verifies, and still looks complete. A reader simply cannot see what was claimed.
    //
    // Asserting "some blob was inlined" is not enough: the *content* blob is inlined either way, so
    // that assertion passes with the metadata dropped. This computes the exact metadata CID the way
    // the statement does, and demands that key.
    let describes = metadata("named.md");
    let (metadata_cid, _) =
        integrity::cid::jcs::compute_jcs_cid(&describes).expect("metadata canonicalizes");

    let mut lineage = session();
    lineage
        .register_content(b"x", describes, None)
        .await
        .unwrap();

    let manifest = lineage.into_manifest().await.unwrap();
    let json = serde_json::to_value(&manifest).unwrap();
    let blobs = json["blobs"].as_object().expect("a blobs map");

    let key = metadata_cid.trim_start_matches("urn:cid:");
    assert!(
        blobs.contains_key(key) || blobs.contains_key(&metadata_cid),
        "the metadata blob {key} is not in the manifest; its statement references nothing.\n\
         blobs present: {:?}",
        blobs.keys().collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn a_computation_with_no_outputs_is_refused() {
    // Not a style rule. An activity with no output cannot be reached from any asset, so it is a node
    // no reader can use -- it inflates the statement count and attests nothing.
    //
    // `integrity` rejects this too, which means asserting only `is_err()` tests upstream rather than
    // this crate: deleting our own guard leaves the test green. Asserting the message is what makes
    // it ours, and keeping the guard means the caller gets a domain error rather than whatever
    // wording the statement constructor happens to use.
    let mut lineage = session();
    let input = lineage
        .register_content(b"in", metadata("in.md"), None)
        .await
        .unwrap();

    let refused = lineage
        .record_computation(&[input], &[], None)
        .await
        .expect_err("an output-less computation was recorded");
    assert!(
        refused.to_string().contains("no outputs is not lineage"),
        "expected this crate's guard to reject it, got: {refused}"
    );
}

#[tokio::test]
async fn an_entity_is_a_new_node_every_time() {
    // Entities have no content to hash, so identity is a fresh UUID. Two registrations of "the same"
    // agent are two nodes -- which is exactly why anything that can be content-addressed should be.
    let mut lineage = session();
    let one = lineage
        .register_entity(metadata("claude-code"), None)
        .await
        .unwrap();
    let two = lineage
        .register_entity(metadata("claude-code"), None)
        .await
        .unwrap();

    assert_ne!(one, two, "entity identity is registration, not content");
}

/// Decode one fixture case's bytes, whether spelled as hex or as a repeat.
fn case_bytes(case: &serde_json::Value) -> Vec<u8> {
    if let Some(repeat) = case.get("repeat") {
        let byte = u8::from_str_radix(repeat["byte_hex"].as_str().unwrap(), 16).unwrap();
        let count = repeat["count"].as_u64().unwrap() as usize;
        return vec![byte; count];
    }
    let hex = case["bytes_hex"].as_str().unwrap();
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

#[tokio::test]
async fn asset_cids_match_the_python_sdk_exactly() {
    // The one cross-language check that can be exact.
    //
    // Statement CIDs cannot be compared: they cover a credential carrying `validFrom`, so two
    // recordings of the same work never match byte for byte. Content CIDs have no such problem --
    // they are a pure function of the bytes. So this pins the property that actually matters for
    // interoperability: a manifest this plugin writes names its assets by the same identifiers
    // `eqty_sdk` would, and a graph merged across both paths joins on them.
    //
    // The vector is committed, generated by `eqty_sdk.get_cid_for_bytes`. Regenerate it with
    // `just relay-cid-vector`; that it needs Python is exactly why it is a fixture and not a
    // dependency.
    let raw = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/content-cids.json"),
    )
    .expect("the CID vector is readable");
    let vector: serde_json::Value = serde_json::from_str(&raw).expect("the CID vector parses");
    let cases = vector["cases"].as_array().expect("cases is an array");
    assert!(!cases.is_empty(), "an empty vector proves nothing");

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let bytes = case_bytes(case);
        // `eqty_sdk` returns the `urn:cid:` form; `integrity` returns the bare CID.
        let expected = case["cid"].as_str().unwrap().trim_start_matches("urn:cid:");

        let mut lineage = session();
        let asset = lineage
            .register_content(&bytes, metadata(name), None)
            .await
            .unwrap_or_else(|error| panic!("case {name} failed to register: {error}"));

        assert_eq!(
            asset.as_str(),
            expected,
            "case {name}: this plugin and eqty_sdk disagree on what {} bytes are called",
            bytes.len()
        );
    }
}

/// Count statements of one `@type` in a serialized manifest.
fn count_of(json: &serde_json::Value, kind: &str) -> usize {
    json["statements"]
        .as_object()
        .expect("statements")
        .values()
        .filter(|statement| statement["@type"] == kind)
        .count()
}

#[tokio::test]
async fn identical_content_is_registered_once() {
    // Under content addressing a second registration of the same bytes asserts nothing the first
    // did not -- same CID, same node, same edges. A real session repeats its system prompt on every
    // call, which made 22 of 77 data statements redundant before this held.
    let mut lineage = session();
    let first = lineage
        .register_content(b"you are careful", metadata("system prompt"), None)
        .await
        .expect("content registers");
    for _ in 0..4 {
        let again = lineage
            .register_content(b"you are careful", metadata("system prompt"), None)
            .await
            .expect("content registers");
        assert_eq!(again, first, "the same bytes must resolve to the same node");
    }

    let json = serde_json::to_value(lineage.into_manifest().await.expect("manifest")).unwrap();
    assert_eq!(count_of(&json, "DataRegistration"), 1);
    assert_eq!(count_of(&json, "MetadataRegistration"), 1);
    // The credential rides along, so collapsing the registration must collapse it too.
    assert_eq!(count_of(&json, "CredentialRegistration"), 2);
}

#[tokio::test]
async fn the_same_bytes_seen_two_ways_keep_both_descriptions() {
    // The guard on the test above. Deduping on content alone would be wrong: one file's contents
    // seen at two paths is a single node with two things said about it, and dropping the second
    // metadata would silently lose a path the recording did observe.
    let mut lineage = session();
    let here = lineage
        .register_content(b"shared bytes", metadata("/a/config.toml"), None)
        .await
        .expect("content registers");
    let there = lineage
        .register_content(b"shared bytes", metadata("/b/config.toml"), None)
        .await
        .expect("content registers");
    assert_eq!(here, there, "content decides identity; path is metadata");

    let json = serde_json::to_value(lineage.into_manifest().await.expect("manifest")).unwrap();
    assert_eq!(
        count_of(&json, "DataRegistration"),
        1,
        "one set of bytes is one node"
    );
    assert_eq!(
        count_of(&json, "MetadataRegistration"),
        2,
        "but both paths must survive"
    );
}

#[tokio::test]
async fn the_blob_total_tracks_what_is_actually_held() {
    // Read to decide whether a mid-session snapshot is worth its cost, so it has to be right in the
    // one case that happens constantly: the same content registered twice. A total that counted
    // every insert would climb on a session that never grew, and the checkpoint interval derived
    // from it would widen until the recording stopped being written at all.
    let mut session = session();
    let before = session.blob_bytes();
    assert_eq!(before, 0, "nothing held yet");

    let body = b"the same forty-two bytes, give or take a few";
    session
        .register_content(body, serde_json::json!({ "name": "first" }), None)
        .await
        .expect("registers");
    let once = session.blob_bytes();
    assert!(
        once >= body.len(),
        "the content and its metadata are both held: {once}"
    );

    // Identical bytes, described the same way. Nothing new is stored.
    session
        .register_content(body, serde_json::json!({ "name": "first" }), None)
        .await
        .expect("registers again");
    assert_eq!(
        session.blob_bytes(),
        once,
        "re-registering identical content must not inflate the total"
    );

    // Different content does grow it.
    session
        .register_content(b"other", serde_json::json!({ "name": "second" }), None)
        .await
        .expect("registers");
    assert!(
        session.blob_bytes() > once,
        "but genuinely new content does"
    );
}
