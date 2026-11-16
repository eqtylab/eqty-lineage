//! What the subscriber can decode, under this crate's real serde_json feature set.

use eqty_lineage_nemo_relay::{LineageEvent, classify, decode_event};
use nemo_relay_plugin::Event;
use std::path::PathBuf;

/// The committed capture's first LLM scope end, with the field Relay 0.11 adds to a streamed call.
fn streamed_llm_end() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/codex-session.jsonl");
    let text = std::fs::read_to_string(path).expect("fixture is readable");
    let mut event: serde_json::Value = text
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("fixture line is JSON"))
        .find(|event| event["category"] == "llm" && event["scope_category"] == "end")
        .expect("the fixture has an LLM scope end");
    event["category_profile"]["time_to_first_chunk"] = serde_json::json!(2.698236917);
    event.to_string()
}

#[test]
fn a_streamed_llm_end_decodes_and_carries_its_response() {
    let event = decode_event(&streamed_llm_end()).expect("the event should decode");
    match classify(&event) {
        Some(LineageEvent::ModelCallEnded { response, .. }) => {
            assert!(response.is_some(), "the response must survive decoding")
        }
        other => panic!("expected a model call end, got {other:?}"),
    }
}

/// Pins why `decode_event` exists. If this starts passing, `arbitrary_precision` has left the
/// dependency graph and the SDK's typed subscriber would do.
#[test]
fn from_str_alone_cannot_decode_it() {
    assert!(serde_json::from_str::<Event>(&streamed_llm_end()).is_err());
}
