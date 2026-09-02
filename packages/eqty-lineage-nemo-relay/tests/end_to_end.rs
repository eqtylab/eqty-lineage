//! From Relay's event stream to a manifest on disk.
//!
//! Every other test here checks one piece. This drives the whole path the plugin actually takes:
//! deserialize the events Relay would hand a subscriber, attribute them to a session, classify them,
//! push them through the mailbox, and read back what landed on disk.

use std::fs;
use std::path::PathBuf;

use eqty_lineage_nemo_relay::{
    LineageSession, Mailbox, Policy, SessionRouter, SignerFactory, classify,
};
use integrity::signer::{SignerType, ed25519_signer::Ed25519Signer};
use nemo_relay_plugin::Event;
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

/// Replay events through the real path and return the directory they were written to.
fn replay(events: &[Event], into: &TempDir) {
    let mailbox = Mailbox::start(into.path().to_path_buf(), policy(), signer_factory());
    let mut router = SessionRouter::new();

    for event in events {
        let Some(session_id) = router.attribute(event) else {
            continue;
        };
        if let Some(lineage) = classify(event) {
            assert!(
                mailbox.send(&session_id, event.timestamp().to_rfc3339(), lineage),
                "the queue should not be full for a fixture this size"
            );
        }
    }

    // Dropping is the flush. On Codex it is the only export trigger there will ever be.
    drop(mailbox);
}

fn manifests(dir: &TempDir) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = fs::read_dir(dir.path())
        .expect("the manifest directory exists")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    found.sort();
    found
}

fn fixture() -> Vec<Event> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/codex-session.jsonl");
    fs::read_to_string(path)
        .expect("fixture is readable")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("fixture line is an Event"))
        .collect()
}

#[test]
fn a_real_codex_session_writes_a_signed_manifest() {
    let into = TempDir::new().expect("a temp dir");
    replay(&fixture(), &into);

    let written = manifests(&into);
    assert_eq!(written.len(), 1, "one session, one manifest: {written:?}");

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&written[0]).unwrap()).expect("the manifest is JSON");
    assert_eq!(manifest["version"], "3");
    assert!(
        manifest["statements"]
            .as_object()
            .is_some_and(|s| !s.is_empty()),
        "a manifest with no statements attests nothing"
    );
}

#[test]
fn the_manifest_is_named_for_its_session() {
    let into = TempDir::new().expect("a temp dir");
    replay(&fixture(), &into);

    let written = manifests(&into);
    let name = written[0]
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        name, "01a040aa-45b9-7562-a16b-37314dd082d7",
        "a manifest must be findable by the session it describes"
    );
}

#[test]
fn a_codex_session_of_shell_commands_yields_no_file_lineage() {
    // The plan predicted exactly this and it is the expected result, not a failure: that session ran
    // everything through `exec`, and a shell command that writes a file is not attributable from its
    // result. A manifest with no file provenance is still a real, signable manifest -- and the
    // coverage statement inside it is what tells a reader to read it that way.
    let into = TempDir::new().expect("a temp dir");
    replay(&fixture(), &into);

    let raw = fs::read_to_string(&manifests(&into)[0]).unwrap();
    assert!(
        !raw.contains("filePath"),
        "the Bash-only capture should produce no file nodes"
    );
}

/// A Claude Code tool scope, spelled the way Relay emits one.
///
/// Built rather than captured because the committed fixture is Codex, which never touches a file
/// through a structured tool. `Event` is `Deserialize`, and Relay deserializes these same bytes
/// before calling a subscriber, so this exercises the real decoding path.
fn claude_read(
    session: &str,
    uuid: &str,
    parent: &str,
    phase: &str,
    data: serde_json::Value,
) -> Event {
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1",
        "kind": "scope",
        "category": "tool",
        "scope_category": phase,
        "name": "Read",
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": "2026-09-02T12:00:00.000000+00:00",
        "attributes": [],
        "data": data,
        "data_schema": null,
        "category_profile": { "tool_call_id": "toolu_01" },
        "metadata": {
            "session_id": session,
            "agent_kind": "claude-code",
            "harness": "claude-code",
            "hook_event_name": if phase == "start" { "PreToolUse" } else { "PostToolUse" },
            "source": "hook",
            "tool_correlation_status": "explicit"
        }
    }))
    .expect("a well-formed Relay event")
}

#[test]
fn a_claude_read_becomes_a_file_node_in_the_manifest() {
    let session = "01a040aa-0000-0000-0000-000000000001";
    let root = "01a040aa-0000-0000-0000-0000000000ff";
    let events = vec![
        claude_read(
            session,
            root,
            root,
            "start",
            serde_json::json!({ "file_path": "/report.md" }),
        ),
        claude_read(
            session,
            root,
            root,
            "end",
            serde_json::json!({
                "type": "text",
                "file": {
                    "filePath": "/report.md",
                    "content": "# Report\n\nFindings.\n",
                    "numLines": 3, "totalLines": 3, "startLine": 1
                }
            }),
        ),
    ];

    let into = TempDir::new().expect("a temp dir");
    replay(&events, &into);

    let raw = fs::read_to_string(&manifests(&into)[0]).unwrap();
    // Metadata is base64 inside the manifest, so assert on the decoded blobs.
    let manifest: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let decoded: String = manifest["blobs"]
        .as_object()
        .expect("a blobs map")
        .values()
        .filter_map(|blob| blob.as_str())
        .filter_map(|blob| {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.decode(blob).ok()
        })
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .collect();

    assert!(
        decoded.contains("/report.md"),
        "the file the agent read should be a node"
    );
    assert!(
        decoded.contains("\"observed\":true"),
        "an explicit correlation must be recorded as observed, not inferred"
    );
}

#[test]
fn a_guessed_correlation_reaches_the_manifest_as_inferred() {
    // The correlation ladder only earns its keep if the bottom rung survives all the way into the
    // graph. `agent_fallback` means Relay had no hints pending and parented the call to the root
    // turn scope by default -- a guess. A reader who cannot tell that from direct evidence cannot
    // weigh the graph, so it must land as `observed: false`.
    let session = "01a040aa-0000-0000-0000-000000000003";
    let root = "01a040aa-0000-0000-0000-0000000000fd";
    let mut events = vec![
        claude_read(
            session,
            root,
            root,
            "start",
            serde_json::json!({ "file_path": "/guessed.md" }),
        ),
        claude_read(
            session,
            root,
            root,
            "end",
            serde_json::json!({
                "type": "text",
                "file": { "filePath": "/guessed.md", "content": "hi\n", "numLines": 1, "totalLines": 1, "startLine": 1 }
            }),
        ),
    ];
    for event in &mut events {
        let mut json = event.to_json_value();
        json["metadata"]["tool_correlation_status"] = serde_json::json!("agent_fallback");
        *event = serde_json::from_value(json).expect("still a well-formed event");
    }

    let into = TempDir::new().expect("a temp dir");
    replay(&events, &into);

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifests(&into)[0]).unwrap()).unwrap();
    let decoded: String = manifest["blobs"]
        .as_object()
        .expect("a blobs map")
        .values()
        .filter_map(|blob| blob.as_str())
        .filter_map(|blob| {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.decode(blob).ok()
        })
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .collect();

    assert!(
        decoded.contains("/guessed.md"),
        "the file should still be a node"
    );
    assert!(
        decoded.contains("\"observed\":false"),
        "a guessed correlation must not be recorded as an observation"
    );
    assert!(
        !decoded.contains("\"observed\":true"),
        "nothing in this capture was directly evidenced"
    );
}

#[test]
fn a_tool_whose_result_says_nothing_still_names_its_path() {
    // Relay passes tool arguments through verbatim, so a path is recoverable from the input even
    // when the result is a bare string. The node is identity-only -- this path was written, content
    // not established -- which is honest about what was seen and better than silence.
    let session = "01a040aa-0000-0000-0000-000000000002";
    let root = "01a040aa-0000-0000-0000-0000000000fe";
    let mut start = claude_read(
        session,
        root,
        root,
        "start",
        serde_json::json!({ "file_path": "/notes.md" }),
    );
    let mut end = claude_read(session, root, root, "end", serde_json::json!("ok"));
    // `claude_read` names the tool `Read`; rename both halves to a writing tool.
    for event in [&mut start, &mut end] {
        let mut json = event.to_json_value();
        json["name"] = serde_json::json!("Write");
        *event = serde_json::from_value(json).expect("still a well-formed event");
    }

    let into = TempDir::new().expect("a temp dir");
    replay(&[start, end], &into);

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifests(&into)[0]).unwrap()).unwrap();
    let decoded: String = manifest["blobs"]
        .as_object()
        .expect("a blobs map")
        .values()
        .filter_map(|blob| blob.as_str())
        .filter_map(|blob| {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.decode(blob).ok()
        })
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .collect();

    assert!(
        decoded.contains("/notes.md"),
        "the path from the arguments should be a node"
    );
    assert!(
        decoded.contains("unknown:/notes.md"),
        "and it must be marked as content never established, not as empty content"
    );
}

/// An LLM scope, spelled the way Relay emits one through its gateway.
///
/// Note what is *absent*: no `session_id` anywhere. Gateway events never carry one, so these are
/// attributed purely by walking `parent_uuid` up to a scope that does — which is the mechanism
/// `session.rs` exists for, exercised here end to end rather than in isolation.
fn llm_scope(
    name: &str,
    uuid: &str,
    parent: &str,
    phase: &str,
    profile: serde_json::Value,
) -> Event {
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1",
        "kind": "scope",
        "category": "llm",
        "scope_category": phase,
        "name": name,
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": "2026-09-02T12:00:00.000000+00:00",
        "attributes": [],
        "data": null,
        "data_schema": null,
        "category_profile": profile,
        "metadata": { "gateway_path": "/v1/messages", "llm_correlation_status": "single_hint" }
    }))
    .expect("a well-formed Relay LLM event")
}

/// A turn scope, which is the hook-path event that names the session the LLM scopes hang from.
fn turn_scope(session: &str, uuid: &str, parent: &str, phase: &str) -> Event {
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1",
        "kind": "scope",
        "category": "custom",
        "scope_category": phase,
        "name": "claude-code-turn",
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": "2026-09-02T12:00:00.000000+00:00",
        "attributes": [],
        "data": null,
        "data_schema": null,
        "category_profile": null,
        "metadata": {
            "session_id": session,
            "agent_kind": "claude-code",
            "nemo_relay_scope_role": "turn"
        }
    }))
    .expect("a well-formed turn event")
}

fn decoded_blobs(path: &std::path::Path) -> String {
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    manifest["blobs"]
        .as_object()
        .expect("a blobs map")
        .values()
        .filter_map(|blob| blob.as_str())
        .filter_map(|blob| {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.decode(blob).ok()
        })
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .collect()
}

/// One turn containing one model call, with the given normalized conversation and reply.
///
/// Scope UUIDs are derived from `session` rather than fixed. Two sessions never share them in
/// reality, and a portability test whose two runs agreed on every identifier would pass for anything
/// derived from those identifiers -- which is a test that proves nothing.
fn one_model_call(
    session: &str,
    api: &str,
    messages: serde_json::Value,
    reply: &str,
) -> Vec<Event> {
    let tag = &session[session.len() - 2..];
    let root = format!("01a040aa-0000-0000-0000-00000000a{tag}0");
    let turn = format!("01a040aa-0000-0000-0000-00000000a{tag}1");
    let call = format!("01a040aa-0000-0000-0000-00000000a{tag}2");
    let (root, turn, call) = (root.as_str(), turn.as_str(), call.as_str());
    vec![
        turn_scope(session, turn, root, "start"),
        llm_scope(
            api,
            call,
            turn,
            "start",
            serde_json::json!({
                "model_name": "test-model",
                "annotated_request": { "messages": messages, "model": "test-model" }
            }),
        ),
        llm_scope(
            api,
            call,
            turn,
            "end",
            serde_json::json!({
                "model_name": "test-model",
                "annotated_response": {
                    "model": "test-model",
                    "message": reply,
                    // Relay's normalized vocabulary, not the provider's: Anthropic's `end_turn` and
                    // OpenAI's `stop` both arrive here as `complete`.
                    "finish_reason": "complete",
                    "usage": { "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14 }
                }
            }),
        ),
    ]
}

#[test]
fn a_model_call_becomes_a_prompt_and_a_completion() {
    let into = TempDir::new().expect("a temp dir");
    replay(
        &one_model_call(
            "01a040aa-0000-0000-0000-00000000000a",
            "anthropic.messages",
            serde_json::json!([{ "role": "user", "content": "what is lineage?" }]),
            "provenance you can verify",
        ),
        &into,
    );

    let decoded = decoded_blobs(&manifests(&into)[0]);
    assert!(
        decoded.contains("what is lineage?"),
        "the prompt should be a node"
    );
    assert!(
        decoded.contains("provenance you can verify"),
        "and so should the completion"
    );
    assert!(decoded.contains("\"assetType\":\"Prompt\""), "{decoded}");
    assert!(decoded.contains("\"assetType\":\"Reasoning\""), "{decoded}");
    assert!(
        decoded.contains("\"finishReason\":\"complete\"")
            && decoded.contains("\"total_tokens\":14"),
        "usage and finish reason travel with the completion"
    );
}

#[test]
fn the_same_conversation_through_two_providers_is_one_prompt() {
    // The claim that justifies reaching for `annotated_request` rather than the serialized `data`:
    // Relay normalizes the conversation, so the identical exchange through Anthropic Messages and
    // through OpenAI Responses hashes to the same prompt. Two sessions on different providers then
    // *join* on that node instead of forking on wire format.
    //
    // Hashing the raw provider JSON would fail this, and would fail it silently -- both manifests
    // would look perfectly well-formed while describing the same prompt as two different things.
    let messages = serde_json::json!([{ "role": "user", "content": "identical question" }]);

    let anthropic = TempDir::new().unwrap();
    replay(
        &one_model_call(
            "01a040aa-0000-0000-0000-00000000000b",
            "anthropic.messages",
            messages.clone(),
            "same",
        ),
        &anthropic,
    );
    let openai = TempDir::new().unwrap();
    replay(
        &one_model_call(
            "01a040aa-0000-0000-0000-00000000000c",
            "openai.responses",
            messages,
            "same",
        ),
        &openai,
    );

    let content_cid = |dir: &TempDir| -> String {
        let decoded = decoded_blobs(&manifests(dir)[0]);
        let marker = "\"assetType\":\"Prompt\"";
        let at = decoded
            .find(marker)
            .unwrap_or_else(|| panic!("no prompt node in {decoded}"));
        let tail = &decoded[at..];
        let key = "\"content-cid\":\"";
        let start = tail.find(key).expect("a prompt carries its content CID") + key.len();
        tail[start..].split('"').next().unwrap().to_string()
    };

    assert_eq!(
        content_cid(&anthropic),
        content_cid(&openai),
        "the same normalized conversation must be one prompt, whichever provider carried it"
    );
}

#[test]
fn a_response_without_its_request_is_not_half_recorded() {
    // A prompt is what makes a completion attributable. Recording output with no record of what was
    // asked would attest that the model said something, not that it was asked anything.
    let session = "01a040aa-0000-0000-0000-00000000000d";
    let root = "01a040aa-0000-0000-0000-0000000000ba";
    let turn = "01a040aa-0000-0000-0000-0000000000bb";
    let events = vec![
        turn_scope(session, turn, root, "start"),
        llm_scope(
            "anthropic.messages",
            "01a040aa-0000-0000-0000-0000000000bc",
            turn,
            "end",
            serde_json::json!({ "annotated_response": { "message": "orphan reply" } }),
        ),
    ];

    let into = TempDir::new().expect("a temp dir");
    replay(&events, &into);

    let decoded = decoded_blobs(&manifests(&into)[0]);
    assert!(
        !decoded.contains("orphan reply"),
        "a completion with no prompt must not become a node on its own"
    );
}

#[test]
fn a_repeated_response_records_the_call_once() {
    // Relay can deliver an end more than once -- a retried hook, a reconnect. The request is retired
    // when it is paired, so the second end finds nothing and is ignored. Without that, one exchange
    // would appear as two, and a reader counting model calls would be counting deliveries.
    let session = "01a040aa-0000-0000-0000-00000000000e";
    let mut events = one_model_call(
        session,
        "anthropic.messages",
        serde_json::json!([{ "role": "user", "content": "once" }]),
        "only once",
    );
    let repeated = events.last().expect("an end event").clone();
    events.push(repeated);

    let into = TempDir::new().expect("a temp dir");
    replay(&events, &into);

    let decoded = decoded_blobs(&manifests(&into)[0]);
    assert!(
        decoded.contains("\"ModelCall\":1"),
        "a repeated end must not double-count the exchange: {decoded}"
    );
}

#[test]
fn a_prompt_produces_a_completion_and_not_the_reverse() {
    // Direction is the whole content of a lineage edge. Reversed, the graph says the model's answer
    // produced the question -- which still parses, still verifies, and is backwards.
    let into = TempDir::new().expect("a temp dir");
    replay(
        &one_model_call(
            "01a040aa-0000-0000-0000-00000000000f",
            "anthropic.messages",
            serde_json::json!([{ "role": "user", "content": "the question" }]),
            "the answer",
        ),
        &into,
    );

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifests(&into)[0]).unwrap()).unwrap();
    let decoded = decoded_blobs(&manifests(&into)[0]);

    // Pull each payload's content CID out of the metadata blob that names it.
    let cid_of = |asset_type: &str| -> String {
        let at = decoded
            .find(&format!("\"assetType\":\"{asset_type}\""))
            .unwrap_or_else(|| panic!("no {asset_type} node in {decoded}"));
        let key = "\"content-cid\":\"";
        let start = decoded[at..].find(key).expect("a content CID") + key.len();
        decoded[at + start..].split('"').next().unwrap().to_string()
    };
    let (prompt, completion) = (cid_of("Prompt"), cid_of("Reasoning"));

    let computation = manifest["statements"]
        .as_object()
        .expect("statements")
        .values()
        .find(|statement| statement["@type"] == "ComputationRegistration")
        .expect("the model call should be a computation");
    let rendered = serde_json::to_string(computation).unwrap();

    assert!(
        rendered.contains(&format!("\"input\":\"urn:cid:{prompt}\""))
            || rendered.contains(&format!("\"input\":[\"urn:cid:{prompt}\"]")),
        "the prompt must be the input: {rendered}"
    );
    assert!(
        rendered.contains(&format!("\"output\":\"urn:cid:{completion}\""))
            || rendered.contains(&format!("\"output\":[\"urn:cid:{completion}\"]")),
        "the completion must be the output: {rendered}"
    );
}

#[test]
fn the_real_capture_records_its_model_calls() {
    // The reference Codex session made four model calls and touched no files through a structured
    // tool. Before this, its manifest had no computations at all.
    let into = TempDir::new().expect("a temp dir");
    replay(&fixture(), &into);

    let decoded = decoded_blobs(&manifests(&into)[0]);
    assert!(
        decoded.contains("\"assetType\":\"Prompt\""),
        "the capture's prompts should be nodes"
    );
    assert!(
        decoded.contains("\"ModelCall\":4"),
        "coverage should report four recorded model calls: {decoded}"
    );
}
