//! From Relay's event stream to a manifest on disk.
//!
//! Every other test here checks one piece. This drives the whole path the plugin actually takes:
//! deserialize the events Relay would hand a subscriber, attribute them to a session, classify them,
//! push them through the mailbox, and read back what landed on disk.

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use eqty_lineage_nemo_relay::{
    LineageSession, Mailbox, ManifestAnnounced, ManifestMark, Policy, SessionFinished,
    SessionRouter, SignerFactory, classify,
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
///
/// The router is shared with the mailbox exactly as `register` shares it, so the scope-map pruning
/// at export is on the path these tests drive rather than a production-only branch.
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
        let Some(session_id) = router.lock().expect("the router lock").attribute(event) else {
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
    // Dropped before the marks are read: the final export runs when the last handle goes.
    drop(mailbox);
    Arc::into_inner(announced)
        .expect("the mailbox thread has ended, so this is the only handle")
        .into_inner()
        .expect("the marks lock")
}

/// For tests about something other than the manifest mark. `replay` collects them instead.
fn ignore_marks() -> ManifestAnnounced {
    Box::new(|_mark: &ManifestMark| {})
}

/// The session-finished callback the plugin installs: prune the finished session's scopes.
fn forget_with(router: &Arc<Mutex<SessionRouter>>) -> SessionFinished {
    let router = Arc::clone(router);
    Box::new(move |session_id: &str| {
        router.lock().expect("the router lock").forget(session_id);
    })
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

/// Wait until the worker has written `how_many` manifests, rather than for a fixed duration.
///
/// A checkpoint happens on another thread. A sleep long enough on an idle laptop is a coin toss on a
/// CI runner building two other jobs beside it: `a_manifest_exists_before_the_session_ends` passed
/// here every time and failed there at 600ms. Polling asserts the same property without encoding an
/// assumption about how fast the machine is.
fn await_manifests(dir: &TempDir, how_many: usize) -> Vec<PathBuf> {
    for _ in 0..300 {
        let found = manifests(dir);
        if found.len() >= how_many {
            return found;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!(
        "expected {how_many} manifest(s) within 30s, found {:?}",
        manifests(dir)
    );
}

/// Every decoded blob that parses as a JSON object, for assertions about metadata *shape*.
///
/// Substring matching on the concatenated blobs cannot see shape: a nested value and the JSON
/// string encoding of that value differ only by escaping, which is exactly the difference the
/// explorer renders as `[object Object]`.
fn metadata_nodes(path: &std::path::Path) -> Vec<serde_json::Value> {
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
        .filter_map(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .filter(|value| value.is_object())
        .collect()
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
    let completion = metadata_nodes(&manifests(&into)[0])
        .into_iter()
        .find(|node| node["assetType"] == "Reasoning")
        .expect("the completion's metadata");
    assert_eq!(
        completion["finishReason"], "complete",
        "the finish reason travels with the completion: {completion}"
    );
    // Carried as a JSON string, not as a nested object -- see `encode_nested_values`. Parsed here
    // rather than substring-matched, so the assertion is about the value and not about escaping.
    let usage: serde_json::Value = serde_json::from_str(
        completion["usage"]
            .as_str()
            .expect("usage is a JSON string, so the explorer can render it"),
    )
    .expect("and it parses back");
    assert_eq!(usage["total_tokens"], 14, "and so does the usage: {usage}");
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

    // `input` and `output` are each a single value or an array, so normalize before asserting.
    let cids = |field: &str| -> Vec<String> {
        match &computation[field] {
            serde_json::Value::String(one) => vec![one.clone()],
            serde_json::Value::Array(many) => many
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        }
    };
    let inputs = cids("input");
    let outputs = cids("output");

    assert!(
        inputs.contains(&format!("urn:cid:{prompt}")),
        "the prompt must be an input: inputs={inputs:?} prompt={prompt}"
    );
    assert_eq!(
        outputs,
        vec![format!("urn:cid:{completion}")],
        "the completion must be the sole output"
    );
    assert!(
        !outputs.contains(&format!("urn:cid:{prompt}")),
        "the prompt must never be an output -- that edge reads as the answer producing the question"
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

/// A mark, which is how Relay carries session start, subagent lifecycle and compaction.
fn mark(session: &str, uuid: &str, parent: &str, name: &str, metadata: serde_json::Value) -> Event {
    let mut meta = serde_json::json!({ "session_id": session, "agent_kind": "claude-code" });
    if let (Some(target), Some(extra)) = (meta.as_object_mut(), metadata.as_object()) {
        for (key, value) in extra {
            target.insert(key.clone(), value.clone());
        }
    }
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1",
        "kind": "mark",
        "name": name,
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": "2026-09-02T12:00:00.000000+00:00",
        "metadata": meta
    }))
    .expect("a well-formed mark")
}

/// A whole session: agent, prompt, a model call, a tool call that reads and writes, a subagent, and
/// a compaction. Everything the plugin can currently record, in one run.
fn full_session(session: &str) -> Vec<Event> {
    let root = "01a040aa-0000-0000-0000-0000000000c0";
    let turn = "01a040aa-0000-0000-0000-0000000000c1";
    let call = "01a040aa-0000-0000-0000-0000000000c2";
    let tool = "01a040aa-0000-0000-0000-0000000000c3";

    let mut events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "opus" }),
        ),
        turn_scope(session, turn, root, "start"),
    ];

    // The prompt that opened the turn.
    let mut prompt_event = turn_scope(session, turn, root, "start");
    let mut json = prompt_event.to_json_value();
    json["data"] = serde_json::json!({ "prompt": "summarise the report" });
    json["metadata"]["nemo_relay_scope_role"] = serde_json::json!("turn");
    prompt_event = serde_json::from_value(json).expect("a turn carrying a prompt");
    events.push(prompt_event);

    events.push(mark(session, "01a040aa-0000-0000-0000-0000000000c4", turn, "subagent",
        serde_json::json!({ "hook_event_name": "SubagentStart", "subagent_id": "sub-1", "agent_type": "researcher" })));

    events.push(llm_scope(
        "anthropic.messages",
        call,
        turn,
        "start",
        serde_json::json!({
            "model_name": "opus",
            "annotated_request": {
                "model": "opus",
                "instructions": "You are careful.",
                "messages": [{ "role": "user", "content": "summarise the report" }]
            }
        }),
    ));
    events.push(llm_scope(
        "anthropic.messages",
        call,
        turn,
        "end",
        serde_json::json!({
            "model_name": "opus",
            "annotated_response": {
                "model": "opus",
                "message": "Here is the summary.",
                "finish_reason": "complete",
                "usage": { "prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17 }
            }
        }),
    ));

    events.push(claude_read(
        session,
        tool,
        turn,
        "start",
        serde_json::json!({ "file_path": "/report.md" }),
    ));
    events.push(claude_read(
        session,
        tool,
        turn,
        "end",
        serde_json::json!({
            "filePath": "/report.md",
            "originalFile": "old body\n",
            "oldString": "old body",
            "newString": "new body",
            "replaceAll": false
        }),
    ));

    events.push(mark(
        session,
        "01a040aa-0000-0000-0000-0000000000c5",
        turn,
        "subagent",
        serde_json::json!({ "hook_event_name": "SubagentStop", "subagent_id": "sub-1" }),
    ));
    // Both halves, because Claude Code sends both. Sending only `PreCompact` is what let a single
    // compaction record as `compaction 1` and `compaction 2` without a test noticing.
    events.push(mark(
        session,
        "01a040aa-0000-0000-0000-0000000000c6",
        turn,
        "compact",
        serde_json::json!({ "hook_event_name": "PreCompact" }),
    ));
    events.push(mark(
        session,
        "01a040aa-0000-0000-0000-0000000000c7",
        turn,
        "compact",
        serde_json::json!({ "hook_event_name": "PostCompact" }),
    ));

    events
}

#[test]
fn a_full_session_produces_the_same_kind_of_manifest_as_the_shipped_integrations() {
    // The completeness bar, set by evidence rather than by opinion: the LangChain and DeepAgents
    // manifests in `manifests/` carry these asset types and these statement types, and a manifest
    // from this plugin should be recognisable as the same kind of document.
    let into = TempDir::new().expect("a temp dir");
    replay(&full_session("01a040aa-0000-0000-0000-000000000010"), &into);

    let path = &manifests(&into)[0];
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let decoded = decoded_blobs(path);

    for kind in [
        "Agent",
        "Model",
        "Prompt",
        "Reasoning",
        "System_Prompt",
        "Tool",
        "Document",
    ] {
        assert!(
            decoded.contains(&format!("\"assetType\":\"{kind}\"")),
            "a complete manifest should carry a {kind} node:\n{decoded}"
        );
    }

    let types: std::collections::BTreeSet<String> = manifest["statements"]
        .as_object()
        .expect("statements")
        .values()
        .filter_map(|statement| statement["@type"].as_str().map(str::to_string))
        .collect();
    for kind in [
        "DataRegistration",
        "MetadataRegistration",
        "ComputationRegistration",
        "CredentialRegistration",
    ] {
        assert!(types.contains(kind), "missing {kind}; present: {types:?}");
    }

    // Every asset is content-addressed. The shipped manifests contain no EntityRegistration at all,
    // and a node whose identity is a fresh UUID cannot join across runs.
    assert!(
        !types.contains("EntityRegistration"),
        "assets should be content-addressed, not minted: {types:?}"
    );

    assert!(
        decoded.contains("\"name\":") && decoded.contains("\"description\":"),
        "every asset should be named and described, as in the shipped manifests"
    );
}

#[test]
fn a_full_session_records_every_kind_of_activity_it_saw() {
    let into = TempDir::new().expect("a temp dir");
    replay(&full_session("01a040aa-0000-0000-0000-000000000011"), &into);
    let decoded = decoded_blobs(&manifests(&into)[0]);

    for (key, what) in [
        ("\"ModelCall\":1", "the model call"),
        // One, not two, though two hooks fired and two nodes exist.
        ("\"Compaction\":1", "the compaction"),
        ("pre-compaction 1", "the node for the moment before it"),
        ("post-compaction 1", "the node for the moment after it"),
        ("\"Activity\":1", "the tool run"),
        ("\"Agent\":2", "the agent and its subagent"),
        ("\"Tool\":1", "the tool it used"),
        ("\"Model\":1", "the model it called"),
    ] {
        assert!(
            decoded.contains(key),
            "coverage should report {what} ({key}):\n{decoded}"
        );
    }
}

#[test]
fn an_edit_links_the_version_read_to_the_version_written() {
    // The shape a file edit must have: the pre-image is an input and the post-image an output of the
    // same run, so the graph shows which version was changed into which.
    let into = TempDir::new().expect("a temp dir");
    replay(&full_session("01a040aa-0000-0000-0000-000000000012"), &into);
    let decoded = decoded_blobs(&manifests(&into)[0]);

    assert!(decoded.contains("\"fileVersion\":1"), "the version read");
    assert!(
        decoded.contains("\"fileVersion\":2"),
        "and the version written:\n{decoded}"
    );
}

#[test]
#[ignore = "diagnostic: writes a manifest to /tmp for inspection"]
fn dump_a_full_manifest() {
    let into = TempDir::new().expect("a temp dir");
    replay(&full_session("01a040aa-0000-0000-0000-000000000099"), &into);
    fs::copy(&manifests(&into)[0], "/tmp/relay-manifest.json").expect("copied");
}

/// Every computation's `computation_type` paired with its input CIDs.
///
/// The prompt rule differs by kind -- one model call, every tool call -- so a test that only counted
/// inputs could not tell the two apart, and the count alone would pass for the wrong reason.
fn computations_by_type(path: &std::path::Path) -> Vec<(String, Vec<String>)> {
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let statements = manifest["statements"].as_object().expect("statements");

    // `computation_type` lives on a metadata blob whose `subject` is the computation statement, so
    // the type is reached through that indirection rather than read off the statement.
    let decode = |metadata: &str| -> Option<String> {
        use base64::Engine as _;
        let bare = metadata.trim_start_matches("urn:cid:");
        let blob = manifest["blobs"][bare].as_str()?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(blob)
            .ok()?;
        let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        value["computation_type"].as_str().map(str::to_string)
    };

    statements
        .iter()
        .filter(|(_, statement)| statement["@type"] == "ComputationRegistration")
        .map(|(cid, statement)| {
            let kind = statements
                .values()
                .filter(|s| s["@type"] == "MetadataRegistration" && s["subject"] == cid.as_str())
                .filter_map(|s| s["metadata"].as_str())
                .find_map(decode)
                .unwrap_or_default();
            let inputs = match &statement["input"] {
                serde_json::Value::String(one) => vec![one.clone()],
                serde_json::Value::Array(many) => many
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_string))
                    .collect(),
                _ => Vec::new(),
            };
            (kind, inputs)
        })
        .collect()
}

/// Pull the input CIDs of every computation in a manifest.
fn computation_inputs(path: &std::path::Path) -> Vec<Vec<String>> {
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    manifest["statements"]
        .as_object()
        .expect("statements")
        .values()
        .filter(|statement| statement["@type"] == "ComputationRegistration")
        .map(|statement| match &statement["input"] {
            serde_json::Value::String(one) => vec![one.clone()],
            serde_json::Value::Array(many) => many
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}

/// Every blob in a manifest that parses as a JSON object.
///
/// Blobs are decoded one at a time rather than concatenated. A concatenated string cannot tell one
/// node's fields from the next one's, and both an actor's *descriptor* and its *metadata* are blobs
/// -- so a substring search for a name finds the descriptor, which carries no content CID.
fn blob_objects(path: &std::path::Path) -> Vec<serde_json::Value> {
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
        .filter_map(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .filter(|value| value.is_object())
        .collect()
}

/// The content CID of the asset node with the given `name`.
///
/// Keyed on name rather than on `assetType`, because several nodes share a type -- the user's
/// instruction and the conversation sent to the model are both `Prompt`.
fn cid_for_named(path: &std::path::Path, name: &str) -> String {
    blob_objects(path)
        .into_iter()
        .find(|object| object["name"] == name && object["content-cid"].is_string())
        .map(|object| object["content-cid"].as_str().unwrap().to_string())
        .unwrap_or_else(|| panic!("no asset node named {name}"))
}

#[test]
fn the_tool_is_an_input_to_the_run_it_performed() {
    // Registering a Tool node is not enough. If it is not wired into the computation, "which runs
    // used this tool" has no edge to follow and the node is decoration.
    let into = TempDir::new().expect("a temp dir");
    replay(&full_session("01a040aa-0000-0000-0000-000000000013"), &into);

    let path = &manifests(&into)[0];
    let tool = format!("urn:cid:{}", cid_for_named(path, "Read"));
    assert!(
        computation_inputs(path)
            .iter()
            .any(|inputs| inputs.contains(&tool)),
        "the Tool node should be an input to the run that used it"
    );
}

/// A session that calls the same tool twice and makes two model calls.
/// Two turns, each opening with its own instruction and making one model call.
///
/// `repeated_work` has one instruction and several calls, which pins "the first call only" but says
/// nothing about the *next* turn -- so a change that never reset the per-turn flag left every later
/// turn's call with no cause and no test noticed.
fn two_turns(session: &str) -> Vec<Event> {
    let root = "01a040aa-0000-0000-0000-0000000000f0";
    let mut events = Vec::new();
    for (index, ask) in [(1u8, "first instruction"), (2u8, "second instruction")] {
        let turn = format!("01a040aa-0000-0000-0000-0000000000f{index}");
        events.push(turn_scope(session, &turn, root, "start"));

        let mut prompt = turn_scope(session, &turn, root, "start");
        let mut json = prompt.to_json_value();
        json["data"] = serde_json::json!({ "prompt": ask });
        json["metadata"]["nemo_relay_scope_role"] = serde_json::json!("turn");
        prompt = serde_json::from_value(json).unwrap();
        events.push(prompt);

        let call = format!("01a040aa-0000-0000-0000-0000000000a{index}");
        events.push(llm_scope("anthropic.messages", &call, &turn, "start", serde_json::json!({
            "model_name": "opus",
            "annotated_request": { "model": "opus", "messages": [{ "role": "user", "content": ask }] }
        })));
        events.push(llm_scope("anthropic.messages", &call, &turn, "end", serde_json::json!({
            "model_name": "opus",
            "annotated_response": { "model": "opus", "message": ask, "finish_reason": "complete" }
        })));
    }
    events
}

fn repeated_work(session: &str) -> Vec<Event> {
    let root = "01a040aa-0000-0000-0000-0000000000d0";
    let turn = "01a040aa-0000-0000-0000-0000000000d1";
    let mut events = vec![turn_scope(session, turn, root, "start")];

    let mut prompt = turn_scope(session, turn, root, "start");
    let mut json = prompt.to_json_value();
    json["data"] = serde_json::json!({ "prompt": "do the thing" });
    json["metadata"]["nemo_relay_scope_role"] = serde_json::json!("turn");
    prompt = serde_json::from_value(json).unwrap();
    events.push(prompt);

    for (index, body) in [(2u8, "first"), (3u8, "second")] {
        let call = format!("01a040aa-0000-0000-0000-0000000000d{index}");
        events.push(llm_scope("anthropic.messages", &call, turn, "start", serde_json::json!({
            "model_name": "opus",
            "annotated_request": { "model": "opus", "messages": [{ "role": "user", "content": body }] }
        })));
        events.push(llm_scope("anthropic.messages", &call, turn, "end", serde_json::json!({
            "model_name": "opus",
            "annotated_response": { "model": "opus", "message": body, "finish_reason": "complete" }
        })));

        let tool = format!("01a040aa-0000-0000-0000-0000000000e{index}");
        events.push(claude_read(
            session,
            &tool,
            turn,
            "start",
            serde_json::json!({ "file_path": format!("/{body}.md") }),
        ));
        events.push(claude_read(
            session,
            &tool,
            turn,
            "end",
            serde_json::json!({
                "type": "text",
                "file": { "filePath": format!("/{body}.md"), "content": format!("{body}\n"),
                          "numLines": 1, "totalLines": 1, "startLine": 1 }
            }),
        ));
    }
    events
}

#[test]
fn a_tool_used_twice_is_one_node_with_two_edges() {
    // A tool called forty times must not be forty nodes. Deduplicating actors by identity is what
    // makes the graph answer "which runs used this" instead of listing forty lookalikes.
    let into = TempDir::new().expect("a temp dir");
    replay(
        &repeated_work("01a040aa-0000-0000-0000-000000000014"),
        &into,
    );
    let decoded = decoded_blobs(&manifests(&into)[0]);

    assert!(
        decoded.contains("\"Tool\":1"),
        "one Tool node for two calls:\n{decoded}"
    );
    assert!(
        decoded.contains("\"Model\":1"),
        "one Model node for two calls"
    );
    assert!(decoded.contains("\"Activity\":2"), "but two runs");
    assert!(decoded.contains("\"ModelCall\":2"), "and two model calls");
}

#[test]
fn the_user_prompt_feeds_the_call_it_caused_and_not_the_later_ones() {
    // The prompt caused the first call. The later ones were caused by what came back in between,
    // and repeating the prompt across all of them would assert the user asked several times.
    let into = TempDir::new().expect("a temp dir");
    replay(
        &repeated_work("01a040aa-0000-0000-0000-000000000015"),
        &into,
    );

    let path = &manifests(&into)[0];
    let prompt = format!("urn:cid:{}", cid_for_named(path, "user prompt"));
    let computations = computations_by_type(path);

    let model_calls: Vec<_> = computations
        .iter()
        .filter(|(kind, _)| kind == "model_call")
        .collect();
    let fed_model_calls = model_calls
        .iter()
        .filter(|(_, inputs)| inputs.contains(&prompt))
        .count();
    assert!(
        model_calls.len() > 1,
        "this fixture needs several model calls to be worth asserting on"
    );
    assert_eq!(
        fed_model_calls, 1,
        "the instruction caused the first call and not the later ones"
    );

    // Every tool call in the turn takes it, which is what keeps a prompt joined to the files it led
    // to once a view drops the conversation. A weaker claim than the model-call one on purpose: the
    // prompt is in the conversation behind every tool request in the turn.
    let tool_calls: Vec<_> = computations
        .iter()
        .filter(|(kind, _)| kind == "tool_call")
        .collect();
    assert!(!tool_calls.is_empty(), "the fixture runs tools");
    assert!(
        tool_calls
            .iter()
            .all(|(_, inputs)| inputs.contains(&prompt)),
        "every tool call in the turn serves the instruction: {tool_calls:#?}"
    );
}

#[test]
fn each_turn_is_caused_by_its_own_instruction() {
    // Every turn's first model call must name *that* turn's instruction. A per-turn flag that is set
    // and never reset gives the first turn a cause and leaves every later one with none -- which is
    // invisible in a one-turn fixture and survived a mutation until this existed.
    let into = TempDir::new().expect("a temp dir");
    replay(&two_turns("01a040aa-0000-0000-0000-0000000000f9"), &into);
    let path = &manifests(&into)[0];

    let prompts: Vec<String> = blob_objects(path)
        .into_iter()
        .filter(|object| object["name"] == "user prompt")
        .filter_map(|object| {
            object["content-cid"]
                .as_str()
                .map(|c| format!("urn:cid:{c}"))
        })
        .collect();
    assert_eq!(prompts.len(), 2, "two turns, two instructions: {prompts:?}");

    let model_calls: Vec<Vec<String>> = computations_by_type(path)
        .into_iter()
        .filter(|(kind, _)| kind == "model_call")
        .map(|(_, inputs)| inputs)
        .collect();
    assert_eq!(model_calls.len(), 2, "one call per turn");

    for prompt in &prompts {
        assert_eq!(
            model_calls
                .iter()
                .filter(|inputs| inputs.contains(prompt))
                .count(),
            1,
            "each instruction causes exactly one call: {prompt}"
        );
    }
}

#[test]
fn work_done_inside_a_subagent_is_credited_to_it() {
    // The point of tracking subagents. A manifest that credited everything to the root agent would
    // say one actor did work that several actors did, which is the thing a provenance record exists
    // to prevent.
    //
    // Attribution rides as metadata *on the activity*, not as one of its inputs -- an agent in
    // `inputs` would assert the activity consumed the agent, which is false. PROV keeps association
    // and usage apart.
    let into = TempDir::new().expect("a temp dir");
    replay(&full_session("01a040aa-0000-0000-0000-000000000016"), &into);
    let path = &manifests(&into)[0];

    let subagent = cid_for_named(path, "researcher");
    let root = cid_for_named(path, "claude-code");
    assert_ne!(subagent, root, "the subagent is its own node");

    let performers: Vec<String> = blob_objects(path)
        .into_iter()
        .filter_map(|object| object["performedBy"].as_str().map(str::to_string))
        .collect();
    assert!(
        !performers.is_empty(),
        "activities should say who performed them"
    );
    assert!(
        performers.iter().all(|who| who == &subagent),
        "the session's work happened inside the subagent, so it should be credited: {performers:?}"
    );

    // And the attribution must not be an input edge.
    let subagent_ref = format!("urn:cid:{subagent}");
    assert!(
        !computation_inputs(path)
            .iter()
            .any(|inputs| inputs.contains(&subagent_ref)),
        "an agent must never be an input -- that reads as the activity consuming the agent"
    );
}

#[test]
fn an_activity_says_what_kind_of_work_it_was() {
    let into = TempDir::new().expect("a temp dir");
    replay(&full_session("01a040aa-0000-0000-0000-000000000017"), &into);

    let kinds: std::collections::BTreeSet<String> = blob_objects(&manifests(&into)[0])
        .into_iter()
        .filter_map(|object| object["computation_type"].as_str().map(str::to_string))
        .collect();

    assert!(kinds.contains("model_call"), "got {kinds:?}");
    assert!(kinds.contains("tool_call"), "got {kinds:?}");
}

#[test]
fn every_statement_carries_a_credential() {
    // A metadata statement holds the claims a reader acts on -- a file's path, an activity's
    // `performedBy`. Without a credential over it, those claims are unattributed: the manifest
    // still verifies with them altered or added, so the attribution in the graph asserts nothing
    // about who made it. The shipped manifests credential every statement; this one did not,
    // and no existing check noticed, because "the manifest is internally sound" was only ever
    // asserted over the statements that happened to be signed.
    let into = TempDir::new().expect("a temp dir");
    replay(&full_session("01a040aa-0000-0000-0000-000000000098"), &into);

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifests(&into)[0]).unwrap()).unwrap();
    let statements = manifest["statements"].as_object().expect("statements");

    let credentialed: std::collections::HashSet<&str> = statements
        .values()
        .filter(|s| s["@type"] == "CredentialRegistration")
        .filter_map(|s| s["credential"]["credentialSubject"]["id"].as_str())
        .collect();

    let uncredentialed: Vec<(&str, &str)> = statements
        .iter()
        .filter(|(_, s)| s["@type"] != "CredentialRegistration")
        .filter(|(id, _)| !credentialed.contains(id.as_str()))
        .map(|(id, s)| (s["@type"].as_str().unwrap_or("?"), id.as_str()))
        .collect();

    assert!(
        uncredentialed.is_empty(),
        "every statement should be attested, but these are not: {uncredentialed:?}"
    );
    assert!(
        statements
            .values()
            .any(|s| s["@type"] == "MetadataRegistration"),
        "the fixture must actually produce metadata statements, or this test proves nothing"
    );
}

#[test]
fn a_tool_run_records_what_it_was_invoked_with() {
    // Two live sessions recorded five tool runs each as `[Tool] -> Dataset`: something ran, here is
    // its output, and no record of what was asked of it. Most of the work went through Bash, so
    // most of the information was the command -- which the event carries and we were discarding.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000097";
    let tool = "01a040aa-0000-0000-0000-0000000000d1";
    let root = "01a040aa-0000-0000-0000-0000000000d0";

    let mut events = vec![mark(
        session,
        root,
        root,
        "session.start",
        serde_json::json!({ "model": "opus" }),
    )];
    events.push(claude_read(
        session,
        tool,
        root,
        "start",
        serde_json::json!({ "command": "curl -s https://example.com/weather" }),
    ));
    events.push(claude_read(
        session,
        tool,
        root,
        "end",
        serde_json::json!("18 degrees and raining"),
    ));
    replay(&events, &into);

    let path = &manifests(&into)[0];
    let decoded = decoded_blobs(path);
    assert!(
        decoded.contains("curl -s https://example.com/weather"),
        "the command the tool ran must be in the manifest:\n{decoded}"
    );

    // And it must be an *input* to the run -- the activity consumed it. Asserting only that the
    // bytes appear somewhere would pass with the node dangling off the graph.
    let arguments = format!("urn:cid:{}", cid_for_named(path, "Read input"));
    assert!(
        computation_inputs(path)
            .iter()
            .any(|inputs| inputs.contains(&arguments)),
        "the arguments must be an input to the run that used them"
    );
}

#[test]
fn a_second_export_never_overwrites_the_first() {
    // A session should end once. When it does not -- and a subagent scope end made that happen on a
    // real nine-turn session -- overwriting means the surviving file looks like a complete short
    // session instead of the tail of a truncated one. The turns that vanished were the only
    // evidence anything was wrong.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000096";
    replay(&full_session(session), &into);
    replay(&full_session(session), &into);

    let written = manifests(&into);
    assert_eq!(
        written.len(),
        2,
        "the second export must sit beside the first, not on top of it"
    );
    for path in &written {
        let bytes = fs::read(path).expect("each manifest is readable");
        assert!(!bytes.is_empty(), "neither manifest may be truncated");
    }
}

#[test]
fn a_manifest_exists_before_the_session_ends() {
    // The recording used to be all-or-nothing: one write, at the very end. A crash, a kill, or a
    // machine losing power took the whole session with it, and nothing was visible while the agent
    // was still working. The first completed unit of work now checkpoints.
    let into = TempDir::new().expect("a temp dir");
    let events = full_session("01a040aa-0000-0000-0000-000000000095");
    // Everything except the events that close the session, so nothing triggers a final export.
    let mut mid = events;
    mid.truncate(6);

    let router = Arc::new(Mutex::new(SessionRouter::new()));
    let mailbox = Mailbox::start(
        into.path().to_path_buf(),
        policy(),
        signer_factory(),
        forget_with(&router),
        ignore_marks(),
    );
    for event in &mid {
        let Some(session_id) = router.lock().expect("the router lock").attribute(event) else {
            continue;
        };
        if let Some(classified) = classify(event) {
            mailbox.send(&session_id, event.timestamp().to_rfc3339(), classified);
        }
    }

    // Waited for, not slept through, and the mailbox is deliberately not dropped -- dropping it
    // would flush and prove nothing about mid-session writes.
    let written = await_manifests(&into, 1);
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&written[0]).unwrap()).expect("checkpoint is valid JSON");
    assert!(
        !manifest["statements"].as_object().unwrap().is_empty(),
        "the checkpoint must carry the statements recorded so far"
    );
    assert!(
        !decoded_blobs(&written[0]).contains("\"coverage\""),
        "a mid-session checkpoint carries no coverage node -- that is how a reader tells it is not final"
    );
}

#[test]
fn two_subagents_of_one_kind_share_a_node_but_not_an_identity() {
    // Naming a subagent by its kind is what makes "what did a general-purpose agent do" answerable
    // across sessions -- but `record_actor` deduplicates by name, so parallel workers of the same
    // kind share the node. A live session fanned out to four identical subagents; without the
    // instance on each activity they become four indistinguishable performers.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000094";
    let root = "01a040aa-0000-0000-0000-000000000b00";
    let turn = "01a040aa-0000-0000-0000-000000000b01";

    let mut events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "opus" }),
        ),
        turn_scope(session, turn, root, "start"),
    ];
    // Two subagents of the same kind, one after the other, each doing one tool call.
    for (index, uuid) in [
        "01a040aa-0000-0000-0000-000000000b10",
        "01a040aa-0000-0000-0000-000000000b20",
    ]
    .iter()
    .enumerate()
    {
        events.push(subagent_scope(
            session,
            uuid,
            turn,
            "start",
            "general-purpose",
        ));
        let tool = format!("01a040aa-0000-0000-0000-000000000c{index}0");
        events.push(claude_read(
            session,
            &tool,
            uuid,
            "start",
            serde_json::json!({ "file_path": format!("/work/{index}.md") }),
        ));
        events.push(claude_read(
            session,
            &tool,
            uuid,
            "end",
            serde_json::json!({ "file": { "filePath": format!("/work/{index}.md"),
                "content": format!("body {index}\n"), "numLines": 1, "startLine": 1,
                "totalLines": 1 }, "type": "text" }),
        ));
        events.push(subagent_scope(
            session,
            uuid,
            turn,
            "end",
            "general-purpose",
        ));
    }
    replay(&events, &into);

    let path = &manifests(&into)[0];
    let blobs: Vec<serde_json::Value> = blob_objects(path);
    let kinds = blobs
        .iter()
        .filter(|blob| blob["assetType"] == "Agent" && blob["name"] == "general-purpose")
        .count();
    assert_eq!(
        kinds, 1,
        "one kind of subagent is one node, however many ran"
    );
    let instances: std::collections::HashSet<String> = blobs
        .iter()
        .filter_map(|blob| blob.get("performedByInstance")?.as_str())
        .map(str::to_string)
        .collect();
    assert_eq!(
        instances.len(),
        2,
        "but the two instances stay distinguishable on the activities they performed: {instances:?}"
    );
}

/// A synthesized subagent scope, as Relay emits one: `agent` category, parented to the turn.
fn subagent_scope(session: &str, uuid: &str, parent: &str, phase: &str, kind: &str) -> Event {
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1",
        "kind": "scope",
        "uuid": uuid,
        "parent_uuid": parent,
        "name": format!("subagent:{uuid}"),
        "category": "agent",
        "scope_category": phase,
        "attributes": [],
        "timestamp": "2026-09-04T12:00:00Z",
        "metadata": {
            "session_id": session,
            "nemo_relay_scope_role": "subagent",
            "agent_type": kind
        }
    }))
    .expect("a subagent scope")
}

#[test]
fn a_tool_call_that_saw_no_files_says_so() {
    // Absence and silence look identical in a graph. A Codex session reads through `sed` and `awk`
    // because it has no read tool, so its manifest has no file nodes at all -- which reads exactly
    // like a session that touched nothing. The count is what separates the two.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000093";
    let root = "01a040aa-0000-0000-0000-000000000d00";
    let tool = "01a040aa-0000-0000-0000-000000000d01";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "gpt" }),
        ),
        claude_read(
            session,
            tool,
            root,
            "start",
            serde_json::json!({ "command": "awk 'END { print NR }' .env" }),
        ),
        claude_read(session, tool, root, "end", serde_json::json!("1")),
    ];
    replay(&events, &into);

    let decoded = decoded_blobs(&manifests(&into)[0]);
    assert!(
        decoded.contains("ToolCallWithoutFileObservation"),
        "a run that observed no files must record that it observed none:\n{decoded}"
    );
}

#[test]
fn a_session_with_no_agent_is_not_counted_as_one() {
    // Codex issues an ancillary model call to title the conversation, under a session id of its own.
    // One interactive session therefore produced two manifests: 432 statements of work, and 20
    // holding `{"title":"Create report.md"}`. Counting manifests to count sessions gets the wrong
    // answer, and the fragment attests a model call performed by nobody.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000092";

    // A model call and nothing else -- no `session.start`, so no agent is ever registered.
    let events = one_model_call(
        session,
        "openai.responses",
        serde_json::json!([{ "role": "user", "content": "title this conversation" }]),
        "{\"title\":\"Create report.md\"}",
    );
    replay(&events, &into);

    let written = manifests(&into);
    assert_eq!(written.len(), 1, "the call is still recorded, not dropped");
    let name = written[0]
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    assert!(
        name.ends_with(".unattributed.json"),
        "a manifest with no agent must say so in its name, got {name}"
    );
}

/// A tool scope with a chosen name, payload and terminal status.
///
/// `claude_read` fixes all three, which is why the gaps below went untested: every tool call in this
/// file used to arrive with both halves, a payload, and no stated outcome.
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
        "session_id": session,
        "agent_kind": "claude-code",
        "hook_event_name": if phase == "start" { "PreToolUse" } else { "PostToolUse" },
        "source": "hook",
        "tool_correlation_status": "explicit"
    });
    if let Some(status) = status {
        meta["status"] = serde_json::json!(status);
    }
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1",
        "kind": "scope",
        "category": "tool",
        "scope_category": phase,
        "name": name,
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": "2026-09-02T12:00:00.000000+00:00",
        "attributes": [],
        "data": data,
        "data_schema": null,
        "category_profile": { "tool_call_id": call_id },
        "metadata": meta
    }))
    .expect("a well-formed tool scope")
}

/// Every tool-call activity in a manifest, as its metadata object.
fn tool_activities(path: &std::path::Path) -> Vec<serde_json::Value> {
    blob_objects(path)
        .into_iter()
        .filter(|blob| blob["computation_type"] == "tool_call")
        .collect()
}

#[test]
fn a_failed_tool_call_is_not_attested_as_ordinary_work() {
    // `is_error` was classified and then discarded: `ToolCallEnded` was destructured with `..` and
    // no field of the activity metadata carried it. So a call Relay had explicitly told us failed
    // was recorded identically to one that succeeded -- while the README claimed the opposite.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000f01";
    let root = "01a040aa-0000-0000-0000-000000000f02";
    let call = "01a040aa-0000-0000-0000-000000000f03";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "opus" }),
        ),
        tool_scope(
            session,
            call,
            root,
            "start",
            "Bash",
            "toolu_f1",
            serde_json::json!({ "command": "false" }),
            None,
        ),
        tool_scope(
            session,
            call,
            root,
            "end",
            "Bash",
            "toolu_f1",
            serde_json::json!("exit status 1\n"),
            Some("error"),
        ),
    ];
    replay(&events, &into);

    let path = &manifests(&into)[0];
    let activities = tool_activities(path);
    let outcomes: Vec<&str> = activities
        .iter()
        .filter_map(|activity| activity["outcome"].as_str())
        .collect();
    assert_eq!(
        outcomes,
        vec!["failed"],
        "the activity must say the call failed"
    );
    assert!(
        decoded_blobs(path).contains("\"ToolCallFailed\":1"),
        "and coverage must count it:\n{}",
        decoded_blobs(path)
    );
}

#[test]
fn a_tool_call_of_unknown_outcome_is_not_rounded_to_success() {
    // The guard on the test above. Codex states no terminal status at all, so `None` is the common
    // case; recording it as `succeeded` would attest a clean run over one whose failures were simply
    // invisible. `null` and the separate counter are the honest record.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000f11";
    let root = "01a040aa-0000-0000-0000-000000000f12";
    let call = "01a040aa-0000-0000-0000-000000000f13";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "gpt" }),
        ),
        tool_scope(
            session,
            call,
            root,
            "start",
            "Bash",
            "toolu_f2",
            serde_json::json!({ "command": "ls" }),
            None,
        ),
        tool_scope(
            session,
            call,
            root,
            "end",
            "Bash",
            "toolu_f2",
            serde_json::json!("a.txt\n"),
            None,
        ),
    ];
    replay(&events, &into);

    let path = &manifests(&into)[0];
    for activity in tool_activities(path) {
        assert!(
            activity["outcome"].is_null(),
            "an unstated outcome stays unstated: {activity}"
        );
    }
    assert!(
        decoded_blobs(path).contains("\"ToolCallOutcomeUnknown\":1"),
        "coverage must separate 'we did not see' from 'it succeeded'"
    );
}

#[test]
fn a_tool_end_without_its_start_is_still_recorded() {
    // Relay synthesizes ids for post-only hooks, and an end whose id matches no stored start left
    // `open` empty -- which dropped the Tool actor, the arguments and the result together. For a
    // read that was worse than losing the run: with no outputs the activity was refused outright,
    // leaving the file node in the graph with nothing to say which run produced it.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000f21";
    let root = "01a040aa-0000-0000-0000-000000000f22";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "opus" }),
        ),
        // No start half at all.
        tool_scope(
            session,
            "01a040aa-0000-0000-0000-000000000f23",
            root,
            "end",
            "Read",
            "toolu_orphan",
            serde_json::json!({ "file": { "filePath": "/work/only.md", "content": "hi\n",
                "numLines": 1, "startLine": 1, "totalLines": 1 }, "type": "text" }),
            None,
        ),
    ];
    replay(&events, &into);

    let path = &manifests(&into)[0];
    let activities = tool_activities(path);
    assert_eq!(
        activities.len(),
        1,
        "the end event names the tool, so the run is recordable: {activities:?}"
    );
    assert_eq!(activities[0]["tool"], "Read");
    let blobs = blob_objects(path);
    assert!(
        blobs
            .iter()
            .any(|blob| blob["assetType"] == "Tool" && blob["name"] == "Read"),
        "and the Tool actor is registered from that name"
    );
    assert!(
        blobs
            .iter()
            .any(|blob| blob["assetType"] == "Document" && blob["filePath"] == "/work/only.md"),
        "with the file it read reachable from the run"
    );
}

#[test]
fn a_tool_end_carrying_no_payload_is_counted_rather_than_dropped() {
    // The early return on a missing result also skipped the coverage note, so such a call was
    // invisible in the graph *and* absent from the counts -- the one combination a reader cannot
    // detect. The arguments still name the tool and, often, a path.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000f31";
    let root = "01a040aa-0000-0000-0000-000000000f32";
    let call = "01a040aa-0000-0000-0000-000000000f33";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "opus" }),
        ),
        tool_scope(
            session,
            call,
            root,
            "start",
            "Write",
            "toolu_f4",
            serde_json::json!({ "file_path": "/work/silent.md", "content": "x\n" }),
            None,
        ),
        tool_scope(
            session,
            call,
            root,
            "end",
            "Write",
            "toolu_f4",
            serde_json::Value::Null,
            None,
        ),
    ];
    replay(&events, &into);

    let path = &manifests(&into)[0];
    let decoded = decoded_blobs(path);
    assert!(
        decoded.contains("\"ToolCallWithoutResult\":1"),
        "the missing payload is stated:\n{decoded}"
    );
    assert!(
        blob_objects(path)
            .iter()
            .any(|blob| blob["filePath"] == "/work/silent.md"),
        "and the path from the arguments still reaches the graph"
    );
}

#[test]
fn a_sibling_subagent_keeps_its_attribution_when_another_finishes() {
    // The mis-attribution this whole mechanism exists to prevent, arriving through the mechanism
    // itself. `SubagentEnded` ignored its own id and cleared the single active slot, so the *first*
    // stop in a fan-out un-attributed every sibling still running: their later calls were credited
    // to the root agent with no instance. A live session fanned out to four workers at once, and the
    // existing test passed only because it ran its two subagents strictly one after the other.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000f41";
    let root = "01a040aa-0000-0000-0000-000000000f42";
    let turn = "01a040aa-0000-0000-0000-000000000f43";
    let first = "01a040aa-0000-0000-0000-000000000f44";
    let second = "01a040aa-0000-0000-0000-000000000f45";

    let read = |uuid: &str, parent: &str, phase: &str, path: &str| {
        tool_scope(
            session,
            uuid,
            parent,
            phase,
            "Read",
            uuid,
            if phase == "start" {
                serde_json::json!({ "file_path": path })
            } else {
                serde_json::json!({ "file": { "filePath": path, "content": "body\n",
                    "numLines": 1, "startLine": 1, "totalLines": 1 }, "type": "text" })
            },
            None,
        )
    };

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "opus" }),
        ),
        turn_scope(session, turn, root, "start"),
        // Both start before either finishes: a genuine fan-out.
        subagent_scope(session, first, turn, "start", "Explore"),
        subagent_scope(session, second, turn, "start", "Plan"),
        // The first one finishes. The second is still working.
        subagent_scope(session, first, turn, "end", "Explore"),
        read(
            "01a040aa-0000-0000-0000-000000000f46",
            second,
            "start",
            "/work/after.md",
        ),
        read(
            "01a040aa-0000-0000-0000-000000000f46",
            second,
            "end",
            "/work/after.md",
        ),
        subagent_scope(session, second, turn, "end", "Plan"),
    ];
    replay(&events, &into);

    let path = &manifests(&into)[0];
    let activities = tool_activities(path);
    assert_eq!(activities.len(), 1, "one tool call: {activities:?}");
    assert_eq!(
        activities[0]["performedByInstance"], second,
        "work after a sibling stopped still belongs to the subagent that did it: {}",
        activities[0]
    );
    assert_eq!(activities[0]["attribution"], "sole-live-subagent");

    // And the performer is the surviving subagent's node, not the root agent's.
    let plan = blob_objects(path)
        .into_iter()
        .find(|blob| blob["assetType"] == "Agent" && blob["name"] == "Plan")
        .expect("the second subagent is a node");
    assert_eq!(
        activities[0]["performedBy"], plan["content-cid"],
        "credited to the subagent, not the session root"
    );
}

#[test]
fn work_done_while_several_subagents_run_says_it_cannot_tell_which() {
    // Relay reports that *a* subagent is running, never which one performed a given call. With
    // siblings in flight the performer is genuinely unknown, so the run is filed under the root
    // agent and marked -- a specific wrong attribution would be worse, because a reader would act on
    // it. `performedBy` alone cannot express this, which is what `attribution` is for.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000f51";
    let root = "01a040aa-0000-0000-0000-000000000f52";
    let turn = "01a040aa-0000-0000-0000-000000000f53";
    let call = "01a040aa-0000-0000-0000-000000000f56";

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "opus" }),
        ),
        turn_scope(session, turn, root, "start"),
        subagent_scope(
            session,
            "01a040aa-0000-0000-0000-000000000f54",
            turn,
            "start",
            "Explore",
        ),
        subagent_scope(
            session,
            "01a040aa-0000-0000-0000-000000000f55",
            turn,
            "start",
            "Explore",
        ),
        tool_scope(
            session,
            call,
            turn,
            "start",
            "Read",
            "toolu_f5",
            serde_json::json!({ "file_path": "/work/both.md" }),
            None,
        ),
        tool_scope(
            session,
            call,
            turn,
            "end",
            "Read",
            "toolu_f5",
            serde_json::json!({ "file": { "filePath": "/work/both.md", "content": "b\n",
                "numLines": 1, "startLine": 1, "totalLines": 1 }, "type": "text" }),
            None,
        ),
    ];
    replay(&events, &into);

    let path = &manifests(&into)[0];
    let activities = tool_activities(path);
    assert_eq!(activities.len(), 1);
    assert_eq!(
        activities[0]["attribution"], "ambiguous-parallel-subagents",
        "the uncertainty is stated rather than resolved by guessing: {}",
        activities[0]
    );
    assert!(
        activities[0]["performedByInstance"].is_null(),
        "and no instance is claimed"
    );
    assert!(
        decoded_blobs(path).contains("\"AmbiguousSubagentAttribution\":1"),
        "coverage counts it, so it is visible without walking every activity"
    );
}

#[test]
fn a_manifest_states_whether_events_were_dropped() {
    // `dropped_events` was initialized, never incremented, and explicitly discarded (`let _ =
    // dropped`), while the queue's own doc comment promised the manifest would state the loss. A
    // recording with holes that does not say so reads exactly like a complete one.
    let into = TempDir::new().expect("a temp dir");
    replay(&full_session("01a040aa-0000-0000-0000-000000000f61"), &into);

    let decoded = decoded_blobs(&manifests(&into)[0]);
    assert!(
        decoded.contains("\"EventsDropped\":0"),
        "stated even at zero: a reader asking whether the graph is complete should not have to \
         know that absence means none:\n{decoded}"
    );
}

#[test]
fn a_finished_session_stops_being_tracked() {
    // `SessionRouter::forget` existed, was documented as "called at export", and was called from
    // nowhere but its own test. `attribute` inserts an entry for *every* event -- including the
    // `llm.chunk` marks that are 428 of 445 events in the reference capture -- so a long-lived
    // gateway accumulated scopes for every session it ever saw.
    let into = TempDir::new().expect("a temp dir");
    let router = Arc::new(Mutex::new(SessionRouter::new()));
    let mailbox = Mailbox::start(
        into.path().to_path_buf(),
        policy(),
        signer_factory(),
        forget_with(&router),
        ignore_marks(),
    );

    let events = full_session("01a040aa-0000-0000-0000-000000000f71");
    for event in &events {
        let Some(session_id) = router.lock().expect("the router lock").attribute(event) else {
            continue;
        };
        if let Some(classified) = classify(event) {
            mailbox.send(&session_id, event.timestamp().to_rfc3339(), classified);
        }
    }
    assert!(
        router.lock().expect("the router lock").tracked_scopes() > 0,
        "the session's scopes are tracked while it runs"
    );

    // Dropping the mailbox exports every open session, which is what reports them finished.
    drop(mailbox);
    assert_eq!(
        router.lock().expect("the router lock").tracked_scopes(),
        0,
        "and released once its manifest is written"
    );
}

/// A turn scope carrying the user's instruction, which is where `PromptSubmitted` comes from.
fn turn_with_prompt(session: &str, uuid: &str, parent: &str, text: &str) -> Event {
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1",
        "kind": "scope",
        "category": "custom",
        "scope_category": "start",
        "name": "claude-code-turn",
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": "2026-09-02T12:00:00.000000+00:00",
        "attributes": [],
        "data": { "prompt": text },
        "data_schema": null,
        "category_profile": null,
        "metadata": {
            "session_id": session,
            "agent_kind": "claude-code",
            "nemo_relay_scope_role": "turn"
        }
    }))
    .expect("a turn scope carrying a prompt")
}

#[test]
fn a_call_that_was_never_recorded_does_not_consume_the_turn_prompt() {
    // The prompt was taken with `Option::take` before the call was known to be recordable, and
    // `record_model_call` bails after registering its inputs when no response arrived. So a first
    // call that ended without a reply spent the prompt on an activity that was never written: the
    // prompt node was left orphaned, and the next call in the same turn -- the one that did produce
    // an answer -- had nothing to say what it had been asked.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000f81";
    let root = "01a040aa-0000-0000-0000-000000000f82";
    let turn = "01a040aa-0000-0000-0000-000000000f83";
    let dropped = "01a040aa-0000-0000-0000-000000000f84";
    let answered = "01a040aa-0000-0000-0000-000000000f85";

    let request = serde_json::json!({
        "model_name": "test-model",
        "annotated_request": {
            "messages": [{ "role": "user", "content": "go" }],
            "model": "test-model"
        }
    });

    let events = vec![
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "test-model" }),
        ),
        turn_with_prompt(session, turn, root, "summarize the report"),
        // A call that ends with no response at all. Nothing is recorded for it.
        llm_scope(
            "anthropic.messages",
            dropped,
            turn,
            "start",
            request.clone(),
        ),
        llm_scope(
            "anthropic.messages",
            dropped,
            turn,
            "end",
            serde_json::json!({ "model_name": "test-model" }),
        ),
        // The next call in the same turn does answer, and it is the one the prompt caused.
        llm_scope("anthropic.messages", answered, turn, "start", request),
        llm_scope(
            "anthropic.messages",
            answered,
            turn,
            "end",
            serde_json::json!({
                "model_name": "test-model",
                "annotated_response": {
                    "model": "test-model",
                    "message": "here it is",
                    "finish_reason": "complete"
                }
            }),
        ),
    ];
    replay(&events, &into);

    let path = &manifests(&into)[0];
    let decoded = decoded_blobs(path);
    assert!(
        decoded.contains("\"ModelCallWithoutResponse\":1"),
        "the first call is still counted as unrecorded:\n{decoded}"
    );
    assert!(
        decoded.contains("\"ModelCall\":1"),
        "and the second is recorded"
    );

    let prompt = format!("urn:cid:{}", cid_for_named(path, "user prompt"));
    let feeding = computation_inputs(path)
        .iter()
        .filter(|inputs| inputs.contains(&prompt))
        .count();
    assert_eq!(
        feeding, 1,
        "the instruction must reach the call that answered it, not be spent on the one that did not"
    );
}

#[test]
fn an_event_that_completes_nothing_writes_no_manifest() {
    // The checkpoint used to run after *every* event, and a snapshot copies the entire recording
    // rather than the part that changed -- so a long session rewrote its whole manifest hundreds of
    // times, and a session holding a large file rewrote those bytes with it. Only a completed unit of
    // work is worth the cost: a manifest written between a tool's start and its end holds the
    // arguments of a call whose result is still coming, which is a document with no reader.
    let into = TempDir::new().expect("a temp dir");
    let session = "01a040aa-0000-0000-0000-000000000f91";
    let root = "01a040aa-0000-0000-0000-000000000f92";
    let call = "01a040aa-0000-0000-0000-000000000f93";

    let router = Arc::new(Mutex::new(SessionRouter::new()));
    let mailbox = Mailbox::start(
        into.path().to_path_buf(),
        policy(),
        signer_factory(),
        forget_with(&router),
        ignore_marks(),
    );

    // A session start and a tool call that has not come back yet. Neither completes anything.
    for event in [
        mark(
            session,
            root,
            root,
            "session.start",
            serde_json::json!({ "model": "opus" }),
        ),
        tool_scope(
            session,
            call,
            root,
            "start",
            "Read",
            "toolu_f9",
            serde_json::json!({ "file_path": "/work/pending.md" }),
            None,
        ),
    ] {
        let Some(session_id) = router.lock().expect("the router lock").attribute(&event) else {
            continue;
        };
        if let Some(classified) = classify(&event) {
            mailbox.send(&session_id, event.timestamp().to_rfc3339(), classified);
        }
    }
    // Absence is the one thing polling cannot establish: "no manifest yet" and "the worker has not
    // started" look identical, and on a loaded machine a bare sleep really does mean the second. So
    // the absence is paired with a positive control below -- if the worker were merely slow, that
    // control would fail too, and it is what makes this half mean anything.
    std::thread::sleep(std::time::Duration::from_millis(600));
    assert!(
        manifests(&into).is_empty(),
        "nothing has completed, so nothing is worth rewriting the manifest for: {:?}",
        manifests(&into)
    );

    // The control: complete the call. Now a checkpoint is warranted, and the worker writes one.
    let end = tool_scope(
        session,
        call,
        root,
        "end",
        "Read",
        "toolu_f9",
        serde_json::json!({ "file": { "filePath": "/work/pending.md", "content": "done\n",
            "numLines": 1, "startLine": 1, "totalLines": 1 }, "type": "text" }),
        None,
    );
    let session_id = router
        .lock()
        .expect("the router lock")
        .attribute(&end)
        .expect("the tool end attributes to the session");
    mailbox.send(
        &session_id,
        end.timestamp().to_rfc3339(),
        classify(&end).expect("a tool end classifies"),
    );
    let written = await_manifests(&into, 1);
    assert!(
        !decoded_blobs(&written[0]).contains("\"coverage\""),
        "still a checkpoint, not an export"
    );

    // And the export at shutdown writes over the same path, now with coverage.
    drop(mailbox);
    let final_manifests = manifests(&into);
    assert_eq!(
        final_manifests.len(),
        1,
        "one session is one manifest: {final_manifests:?}"
    );
    assert!(
        decoded_blobs(&final_manifests[0]).contains("\"coverage\""),
        "the finished recording states its coverage"
    );
}
