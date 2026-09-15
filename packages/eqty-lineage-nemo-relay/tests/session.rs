//! Session attribution, checked against the same real capture.
//!
//! The interesting case is the one that motivated this module existing at all: an LLM scope carries
//! no `session_id`, because it reached Relay through the gateway rather than through a hook. If
//! these tests pass while [`SessionRouter`] is stubbed out to read metadata only, they are not
//! testing anything -- so [`gateway_events_are_attributed_through_the_scope_tree`] asserts both
//! halves: that the events genuinely lack the field, and that they are attributed anyway.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use eqty_lineage_nemo_relay::SessionRouter;
use nemo_relay_plugin::Event;

fn fixture() -> Vec<Event> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/codex-session.jsonl");
    let text = fs::read_to_string(&path).expect("fixture is readable");
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("fixture line is a Relay Event"))
        .collect()
}

fn states_own_session(event: &Event) -> bool {
    event
        .metadata()
        .and_then(|metadata| metadata.get("session_id"))
        .is_some()
}

#[test]
fn every_event_in_a_real_session_is_attributed() {
    let mut router = SessionRouter::new();
    let unattributed: Vec<_> = fixture()
        .iter()
        .filter(|event| router.attribute(event).is_none())
        .map(|event| event.name().to_string())
        .collect();

    assert!(
        unattributed.is_empty(),
        "left unattributed: {unattributed:?}"
    );
}

#[test]
fn one_session_and_not_several() {
    let mut router = SessionRouter::new();
    let sessions: BTreeSet<_> = fixture()
        .iter()
        .filter_map(|event| router.attribute(event))
        .collect();

    assert_eq!(sessions.len(), 1, "one capture, one session: {sessions:?}");
}

#[test]
fn gateway_events_are_attributed_through_the_scope_tree() {
    let events = fixture();

    // Half one: these events really do lack the field. If a future Relay starts stamping
    // `session_id` on gateway events, this fails and the router becomes redundant -- which is worth
    // learning deliberately rather than keeping dead code forever.
    let silent: Vec<_> = events
        .iter()
        .filter(|event| !states_own_session(event))
        .collect();
    assert!(
        !silent.is_empty(),
        "no event lacks session_id; the scope-tree walk is no longer load-bearing"
    );

    // Half two: they are attributed regardless.
    let mut router = SessionRouter::new();
    let inherited = events
        .iter()
        .filter(|event| {
            let attributed = router.attribute(event).is_some();
            attributed && !states_own_session(event)
        })
        .count();

    assert_eq!(
        inherited,
        silent.len(),
        "every event without its own session_id should inherit one from its parent scope"
    );
}

#[test]
fn an_orphan_is_dropped_rather_than_guessed_at() {
    // An event whose parent was never seen cannot be placed. Attaching it to whatever session
    // happens to be open would attest work that session never did.
    let mut router = SessionRouter::new();
    let orphan = fixture()
        .into_iter()
        .find(|event| !states_own_session(event))
        .expect("the fixture has a gateway event");

    assert_eq!(
        router.attribute(&orphan),
        None,
        "no parent known, so no session"
    );
}

#[test]
fn forgetting_a_session_releases_its_scopes() {
    let mut router = SessionRouter::new();
    let events = fixture();
    let session = events
        .iter()
        .find_map(|event| router.attribute(event))
        .expect("the first event names its session");

    for event in &events {
        router.attribute(event);
    }
    assert!(
        router.tracked_scopes() > 0,
        "scopes accumulate while a session runs"
    );

    router.forget(&session);
    assert_eq!(
        router.tracked_scopes(),
        0,
        "a finished session must not keep one map entry per scope forever"
    );
}
