//! Classification, checked against a real session.
//!
//! `tests/fixtures/codex-session.jsonl` is a capture from an actual Codex run through NeMo Relay,
//! trimmed and redacted: every key, type and nesting level is exactly as Relay emitted it, and the
//! `llm.chunk` marks are cut from 428 to 2 because the point of keeping any is to prove they get
//! dropped. Long strings and long arrays inside `data` and `category_profile` are truncated -- that
//! payload is another person's session content and does not belong in this repository, and
//! classification never reads it. `metadata` is verbatim, because that is what these tests are about.
//!
//! The fixture earns its place by failing loudly: Relay deserializes these same bytes into `Event`
//! before it calls a subscriber, so if an upgrade moves a field, [`deserializes_as_relay_would`]
//! breaks here rather than silently thinning every manifest we produce afterwards.

use std::fs;
use std::path::PathBuf;

use eqty_lineage_nemo_relay::{Correlation, LineageEvent, classify};
use nemo_relay_plugin::Event;

fn fixture() -> Vec<Event> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/codex-session.jsonl");
    let text = fs::read_to_string(&path).expect("fixture is readable");
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("fixture line is a Relay Event"))
        .collect()
}

#[test]
fn deserializes_as_relay_would() {
    let events = fixture();
    assert_eq!(
        events.len(),
        19,
        "fixture changed size; update the expectations below"
    );
}

#[test]
fn streaming_chunks_carry_no_lineage() {
    let dropped = fixture()
        .iter()
        .filter(|event| event.name() == "llm.chunk")
        .filter(|event| classify(event).is_some())
        .count();
    assert_eq!(dropped, 0, "llm.chunk marks must not reach the recorder");
}

#[test]
fn a_real_session_classifies_end_to_end() {
    let events = fixture();
    let classified: Vec<_> = events.iter().filter_map(classify).collect();

    let starts = classified
        .iter()
        .filter(|e| matches!(e, LineageEvent::SessionStarted { .. }))
        .count();
    // Counted at the end only: a call is not lineage until its response exists. The starts are
    // classified too -- they carry the request -- and paired by scope UUID downstream.
    let model_calls = classified
        .iter()
        .filter(|e| matches!(e, LineageEvent::ModelCallEnded { .. }))
        .count();
    let model_starts = classified
        .iter()
        .filter(|e| matches!(e, LineageEvent::ModelCallStarted { .. }))
        .count();
    let tool_starts = classified
        .iter()
        .filter(|e| matches!(e, LineageEvent::ToolCallStarted { .. }))
        .count();
    let tool_ends = classified
        .iter()
        .filter(|e| matches!(e, LineageEvent::ToolCallEnded { .. }))
        .count();

    let prompts = classified
        .iter()
        .filter(|e| matches!(e, LineageEvent::PromptSubmitted { .. }))
        .count();

    assert_eq!(starts, 1, "one session.start mark");
    assert_eq!(
        prompts, 1,
        "the turn scope carries the prompt that opened it"
    );
    assert_eq!(
        model_calls, 4,
        "four LLM scopes, counted at their ends only"
    );
    assert_eq!(tool_starts, 3, "three Bash calls");
    assert_eq!(tool_ends, 3, "each Bash call closed");
    assert_eq!(
        classified.len(),
        starts + prompts + model_starts + model_calls + tool_starts + tool_ends,
        "nothing else in this capture is lineage: {classified:#?}"
    );
}

#[test]
fn codex_never_closes_the_agent_scope() {
    // Codex's plugin hook schema has no `SessionEnd`, so Relay never emits an `agent` scope end and
    // `SessionEnded` -- the export trigger in the plan -- never fires. The last thing this capture
    // produces is a turn end.
    //
    // This is why flushing on `Drop` is not a nicety for tidy shutdown: on Codex it is the *only*
    // path that writes a manifest. A plugin that exported solely on `SessionEnded` would record
    // every Codex session and write none of them.
    let classified: Vec<_> = fixture().iter().filter_map(classify).collect();
    assert!(
        !classified
            .iter()
            .any(|e| matches!(e, LineageEvent::SessionEnded)),
        "if Codex ever gains a session end, the export trigger can be simplified -- check Drop first"
    );
}

#[test]
fn a_tool_start_carries_its_arguments_verbatim() {
    let events = fixture();
    let first = events
        .iter()
        .filter_map(classify)
        .find_map(|event| match event {
            LineageEvent::ToolCallStarted {
                tool_name,
                tool_input,
                ..
            } => Some((tool_name, tool_input)),
            _ => None,
        })
        .expect("the fixture contains a tool call");

    let (tool_name, tool_input) = first;
    assert_eq!(tool_name, "Bash");
    let input = tool_input.expect("Relay passes the arguments object through as `data`");
    assert!(
        input
            .get("command")
            .and_then(|c| c.as_str())
            .is_some_and(|c| c.starts_with("pwd")),
        "the arguments object should be the hook payload's `tool_input`, unwrapped: {input:?}"
    );
}

#[test]
fn codex_cannot_tell_us_whether_a_tool_failed() {
    // Not a limitation of this crate. Codex's plugin hook schema has no `PostToolUseFailure`, so
    // Relay has nothing to derive a status from and strips the null. Recording `false` here would
    // turn "we did not observe a failure" into "we observed a success" -- an attestation nobody
    // made. If this test ever fails because `is_error` became `Some(false)`, that is the bug.
    let unknown = fixture()
        .iter()
        .filter_map(classify)
        .filter_map(|event| match event {
            LineageEvent::ToolCallEnded { is_error, .. } => Some(is_error),
            _ => None,
        })
        .all(|is_error| is_error.is_none());

    assert!(
        unknown,
        "a Codex tool end must not claim a terminal status Relay never reported"
    );
}

#[test]
fn a_guessed_correlation_is_not_an_observation() {
    // Every tool call in this capture is `agent_fallback`: Codex ran them at the top level with no
    // subagent hints pending, so Relay parented them to the root turn scope by default rather than
    // by evidence. That must reach the recorder as inferred.
    let correlations: Vec<_> = fixture()
        .iter()
        .filter_map(classify)
        .filter_map(|event| match event {
            LineageEvent::ToolCallStarted { correlation, .. } => Some(correlation),
            _ => None,
        })
        .collect();

    assert!(!correlations.is_empty(), "the fixture contains tool calls");
    assert!(
        correlations.iter().all(|c| *c == Correlation::Inferred),
        "agent_fallback is Relay guessing, and must not be recorded as observed"
    );
}

/// An `agent`-category scope, in a chosen phase, with a chosen parent.
fn agent_scope(uuid: &str, parent: &str, phase: &str, name: &str) -> Event {
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1",
        "kind": "scope",
        "uuid": uuid,
        "parent_uuid": parent,
        "name": name,
        "category": "agent",
        "scope_category": phase,
        "attributes": [],
        "timestamp": "2026-09-04T11:30:00Z",
        "metadata": { "session_id": "s-1" }
    }))
    .expect("an agent scope")
}

#[test]
fn a_subagent_scope_is_a_subagent() {
    // Relay does not forward `SubagentStart` as a mark. It consumes the hook and synthesizes an
    // `agent`-category scope named for the subagent, parented to the scope that spawned it -- so the
    // scope *is* the subagent, and the hook-name path never fires on Claude Code.
    let root = "01a040aa-0000-0000-0000-0000000000f0";
    let child = "01a040aa-0000-0000-0000-0000000000f1";

    match classify(&agent_scope(child, root, "start", "researcher")) {
        Some(LineageEvent::SubagentStarted { subagent_id, name }) => {
            assert_eq!(subagent_id, child);
            assert_eq!(name.as_deref(), Some("researcher"));
        }
        other => panic!("a nested agent scope start is a subagent, got {other:?}"),
    }
    match classify(&agent_scope(child, root, "end", "researcher")) {
        Some(LineageEvent::SubagentEnded { subagent_id }) => assert_eq!(subagent_id, child),
        other => panic!("a nested agent scope end ends that subagent, got {other:?}"),
    }
    // And the root is still the session, in both directions.
    assert!(matches!(
        classify(&agent_scope(root, root, "end", "agent")),
        Some(LineageEvent::SessionEnded)
    ));
}

/// An `agent`-category scope end, with a chosen parent.
fn agent_scope_end(uuid: &str, parent: &str) -> Event {
    serde_json::from_value(serde_json::json!({
        "atof_version": "0.1",
        "kind": "scope",
        "uuid": uuid,
        "parent_uuid": parent,
        "name": "agent",
        "category": "agent",
        "scope_category": "end",
        "attributes": [],
        "timestamp": "2026-09-04T11:30:00Z",
        "metadata": { "session_id": "s-1" }
    }))
    .expect("an agent scope end")
}

#[test]
fn only_the_root_agent_scope_ends_the_session() {
    // A subagent -- Claude Code's `Task` tool -- opens a scope in this same category. Treating its
    // end as the session's ends the recording mid-session: the manifest is exported, the router
    // forgets the session, and the turns after it accumulate in a fresh recorder that overwrites the
    // file. Measured live on a nine-turn session, which lost turns 1-6 exactly this way.
    let root = "01a040aa-0000-0000-0000-0000000000e0";
    let child = "01a040aa-0000-0000-0000-0000000000e1";

    assert!(
        matches!(
            classify(&agent_scope_end(root, root)),
            Some(LineageEvent::SessionEnded)
        ),
        "the self-parented root scope is the session"
    );
    assert!(
        !matches!(
            classify(&agent_scope_end(child, root)),
            Some(LineageEvent::SessionEnded)
        ),
        "a subagent finishing must not end the session"
    );
}

#[test]
fn a_subagent_is_named_for_what_it_is() {
    // Relay names the scope `subagent:{id}` -- unique and stable, and it tells a reader nothing.
    // The hook's own metadata is merged into that scope, so the agent type travels with it. One
    // live session produced four anonymous subagents; the type is the difference between counting
    // them and asking what they were.
    let scope = |extra: serde_json::Value| -> Event {
        let mut metadata = serde_json::json!({ "session_id": "s-1" });
        if let (Some(target), Some(extra)) = (metadata.as_object_mut(), extra.as_object()) {
            for (key, value) in extra {
                target.insert(key.clone(), value.clone());
            }
        }
        serde_json::from_value(serde_json::json!({
            "atof_version": "0.1",
            "kind": "scope",
            "uuid": "01a040aa-0000-0000-0000-0000000000f5",
            "parent_uuid": "01a040aa-0000-0000-0000-0000000000f4",
            "name": "subagent:abc123",
            "category": "agent",
            "scope_category": "start",
            "attributes": [],
            "timestamp": "2026-09-04T12:00:00Z",
            "metadata": metadata
        }))
        .expect("an agent scope")
    };

    match classify(&scope(
        serde_json::json!({ "agent_type": "general-purpose" }),
    )) {
        Some(LineageEvent::SubagentStarted { name, .. }) => {
            assert_eq!(name.as_deref(), Some("general-purpose"))
        }
        other => panic!("expected a named subagent, got {other:?}"),
    }
    // Without a type, Relay's scope name is still better than nothing.
    match classify(&scope(serde_json::json!({}))) {
        Some(LineageEvent::SubagentStarted { name, .. }) => {
            assert_eq!(name.as_deref(), Some("subagent:abc123"))
        }
        other => panic!("expected a fallback name, got {other:?}"),
    }
}

#[test]
fn the_stated_role_beats_the_tree_shape() {
    // Relay marks a synthesized subagent scope with `nemo_relay_scope_role: "subagent"`, exactly as
    // it spells the turn role. Prefer what it states over what the parent chain implies -- a
    // self-parented scope that says it is a subagent is one.
    let event: Event = serde_json::from_value(serde_json::json!({
        "atof_version": "0.1",
        "kind": "scope",
        "uuid": "01a040aa-0000-0000-0000-0000000000f6",
        "parent_uuid": "01a040aa-0000-0000-0000-0000000000f6",
        "name": "subagent:xyz",
        "category": "agent",
        "scope_category": "start",
        "attributes": [],
        "timestamp": "2026-09-04T12:00:00Z",
        "metadata": { "session_id": "s-1", "nemo_relay_scope_role": "subagent" }
    }))
    .expect("a self-parented scope that states its role");
    assert!(
        matches!(classify(&event), Some(LineageEvent::SubagentStarted { .. })),
        "the stated role must win over the parent check"
    );
}
