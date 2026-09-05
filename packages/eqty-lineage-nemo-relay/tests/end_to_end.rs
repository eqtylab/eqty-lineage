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
    events.push(mark(
        session,
        "01a040aa-0000-0000-0000-0000000000c6",
        turn,
        "compact",
        serde_json::json!({ "hook_event_name": "PreCompact" }),
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
        ("\"Compaction\":1", "the compaction"),
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
    let feeding = computation_inputs(path)
        .iter()
        .filter(|inputs| inputs.contains(&prompt))
        .count();

    assert_eq!(
        feeding, 1,
        "the turn's prompt should be an input exactly once"
    );
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
    // was still working. Each event now checkpoints.
    let into = TempDir::new().expect("a temp dir");
    let events = full_session("01a040aa-0000-0000-0000-000000000095");
    // Everything except the events that close the session, so nothing triggers a final export.
    let mut mid = events;
    mid.truncate(6);

    let mailbox = Mailbox::start(into.path().to_path_buf(), policy(), signer_factory());
    let mut router = SessionRouter::new();
    for event in &mid {
        let Some(session_id) = router.attribute(event) else {
            continue;
        };
        if let Some(classified) = classify(event) {
            mailbox.send(&session_id, event.timestamp().to_rfc3339(), classified);
        }
    }

    // Give the worker a moment to drain, without dropping the mailbox -- dropping it would flush.
    std::thread::sleep(std::time::Duration::from_millis(600));

    let written = manifests(&into);
    assert_eq!(written.len(), 1, "a checkpoint should already be on disk");
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
