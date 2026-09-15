//! Deciding which session an event belongs to.
//!
//! Relay puts `session_id` in event metadata, but only on events that came in through an agent's
//! *hooks*. Events produced by the **gateway** -- every LLM call, which is the half of the stream a
//! hook could never give us -- carry no session identifier at all. Their metadata is `gateway_path`,
//! `llm_correlation_status` and `otel.status_code`, and nothing more.
//!
//! What they do carry is `parent_uuid`. Relay maintains a scope tree, and the LLM scopes hang off
//! the turn scope, which is a hook event and does know its session:
//!
//! ```text
//!   agent root  469f              (session.start marks it; self-parented)
//!     └─ codex-turn  4736         session_id present
//!          ├─ openai.responses  473e     session_id ABSENT  ─┐ resolved by
//!          │    └─ llm.chunk  4f7c       session_id ABSENT  ─┘ walking up
//!          └─ Bash  567b                 session_id present
//! ```
//!
//! So attribution is a lookup up the tree, and this type owns it. It relies on Relay emitting a
//! scope's start before any of its children, which is what "scope" means.

use std::collections::HashMap;

use nemo_relay_plugin::Event;
use uuid::Uuid;

use crate::classify::string_at;

/// Maps Relay's scope tree onto session identifiers.
///
/// Not internally synchronised. The plugin shares one router between the subscriber and the
/// mailbox's `on_finished` closure, which runs on the worker thread, so it is held behind a mutex
/// and locked once per event -- see `register` in `lib.rs`. Keep that lock uncontended: it sits on
/// the hot path of an event stream, which is what the enqueue-and-return contract exists to protect.
#[derive(Debug, Default)]
pub struct SessionRouter {
    /// Scope UUID to the session it belongs to.
    scopes: HashMap<Uuid, String>,
}

impl SessionRouter {
    /// Start with no known scopes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve the session for one event, remembering the answer for its children.
    ///
    /// An event states its own session, or inherits its parent's. Returning `None` means neither
    /// held -- an event arriving before the scope that would explain it. That is dropped rather than
    /// guessed at: attaching a model call to the wrong session is worse than omitting it, because
    /// the manifest would then attest work that session never did.
    pub fn attribute(&mut self, event: &Event) -> Option<String> {
        let parent = event.parent_uuid();

        let session = match string_at(event.metadata(), "session_id") {
            Some(session_id) => session_id.to_string(),
            // A single level is enough because we insert as we descend, and Relay opens a scope
            // before it emits anything inside it.
            None => self.scopes.get(&parent?)?.clone(),
        };

        // Only a scope can be a parent, so only a scope is worth remembering. Marks are the bulk of
        // a stream -- 325 of the 361 events in a live capture, almost all `llm.chunk` -- and an entry
        // per mark is an entry nothing ever looks up, held until the session exports. On Codex no
        // `SessionEnded` arrives, so `forget` may never run at all.
        if event.kind() == "scope" {
            self.scopes.insert(event.uuid(), session.clone());
        }
        // The parent is remembered whatever this event is: `session.start` is a mark, and the scope
        // it names is the agent root every turn hangs from.
        if let Some(parent) = parent {
            self.scopes.entry(parent).or_insert_with(|| session.clone());
        }

        Some(session)
    }

    /// Drop everything known about a finished session.
    ///
    /// Called at export. Without it a long-lived gateway accumulates one entry per scope for every
    /// session it ever saw, which is a slow leak in a process meant to outlive many agents.
    pub fn forget(&mut self, session_id: &str) {
        self.scopes.retain(|_, known| known != session_id);
    }

    /// How many scopes are currently mapped. For tests and diagnostics.
    pub fn tracked_scopes(&self) -> usize {
        self.scopes.len()
    }
}
