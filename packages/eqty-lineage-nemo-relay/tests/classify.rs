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
