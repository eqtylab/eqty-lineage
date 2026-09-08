//! Turning a Relay `Event` into the vocabulary the EQTY recorder already speaks.
//!
//! Relay hands a subscriber every event it emits, in one flat stream: scope starts and ends for
//! agents, turns, LLM calls and tools, plus point-in-time marks. Most of that stream is not lineage.
//! This module is the filter and the translation, and it is deliberately the only place that knows
//! ATOF's spelling of anything -- everything downstream sees [`LineageEvent`].
//!
//! Two rules from the ATOF specification shape the code below and are easy to violate by accident:
//!
//! * **`data` is opaque.** Its shape is producer-defined and consumers must not dispatch on its
//!   contents to decide what an event *is*. So classification reads `kind`, `category` and
//!   `scope_category` -- never `data`. Reading fields *out of* `data` once an event is already
//!   classified is fine, and is how file lineage will work; deciding an event is a tool call
//!   *because* `data` has a `command` key is not.
//! * **Unknown values are preserved, not rejected.** A newer Relay may emit categories this build
//!   has never heard of. Those classify as `None` and are dropped, which loses a node; treating them
//!   as an error would lose the whole session.
//!
//! Classification says *what an event is*, and nothing about which session it belongs to. That
//! separation is forced by the data: events arriving through Relay's gateway carry no session
//! identifier at all. See [`crate::session`].

use std::sync::Arc;

use nemo_relay_plugin::{
    AnnotatedLlmRequest, AnnotatedLlmResponse, Event, EventCategory, Json, ScopeCategory,
};

/// How much Relay trusts its own join between an event and the scope it parented that event to.
///
/// Relay records this on correlated events, and it is the first capture path where the recorder's
/// `observed` flag comes from a field rather than from a heuristic of ours. The two fallback
/// statuses mean Relay had to guess: `agent_fallback` when no hint was pending at all, and
/// `ambiguous_fallback` when hints were pending and none matched -- which Relay's own source
/// documents as possibly wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Correlation {
    /// Relay had direct evidence for the association.
    Observed,
    /// Relay inferred the association and may be wrong.
    Inferred,
}

impl Correlation {
    /// Read the correlation status Relay recorded on this event.
    ///
    /// Tool and LLM events use different metadata keys for the same idea, so both are checked.
    /// Absent metadata is [`Correlation::Inferred`] rather than [`Correlation::Observed`]: an event
    /// carrying no statement about its own correlation has not earned the stronger claim.
    fn from_metadata(metadata: Option<&Json>) -> Self {
        let status = string_at(metadata, "tool_correlation_status")
            .or_else(|| string_at(metadata, "llm_correlation_status"));
        match status {
            Some("explicit" | "single_hint" | "matched_hint" | "active_subagent") => Self::Observed,
            _ => Self::Inferred,
        }
    }

    /// Whether the recorder should mark statements from this event as observed.
    pub fn is_observed(self) -> bool {
        matches!(self, Self::Observed)
    }
}

/// The subset of Relay's event stream that means something to a lineage graph.
///
/// This mirrors the frozen event dataclasses in `eqty_lineage.recorder.events`. Keeping the two in
/// step is what lets a manifest recorded through Relay be compared against one recorded through the
/// Python capture paths: same vocabulary in the middle, so a difference in the graph is a real
/// difference and not a translation artifact.
///
/// No variant carries a session identifier. Session attribution is [`crate::session`]'s job.
#[derive(Debug, Clone, PartialEq)]
pub enum LineageEvent {
    /// A session began. Carries the agent identity the manifest will attest.
    SessionStarted {
        agent: Option<String>,
        model: Option<String>,
    },
    /// A user turn opened with a prompt.
    PromptSubmitted { text: String },
    /// A model call began, carrying the request that opened it.
    ///
    /// Split from the end for the same reason a tool call is: the request is known at the start and
    /// the response only at the end, and they arrive as two events sharing one scope UUID. Pairing
    /// them is the mailbox's job.
    ///
    /// The payload is the **typed** request object, not the serialized `data` a file consumer sees.
    /// This is the in-process advantage that motivated choosing a native plugin: `messages` and
    /// `instructions` here are normalized across providers, so the same conversation through
    /// Anthropic and through OpenAI Responses produces the same bytes and therefore the same CID.
    /// Held behind an `Arc`, so carrying it off the subscriber thread is a refcount bump rather than
    /// a deep copy of a conversation that can run to tens of kilobytes.
    ModelCallStarted {
        call_id: String,
        request: Arc<AnnotatedLlmRequest>,
    },
    /// A model call completed.
    ModelCallEnded {
        call_id: String,
        model: Option<String>,
        response: Option<Arc<AnnotatedLlmResponse>>,
        correlation: Correlation,
    },
    /// A tool call began. `tool_input` is the arguments object verbatim.
    ToolCallStarted {
        tool_use_id: Option<String>,
        tool_name: String,
        tool_input: Option<Json>,
        correlation: Correlation,
    },
    /// A tool call finished.
    ToolCallEnded {
        tool_use_id: Option<String>,
        tool_name: String,
        result: Option<Json>,
        /// `None` means Relay did not know, which is not the same as "succeeded". See
        /// [`terminal_status`].
        is_error: Option<bool>,
        correlation: Correlation,
    },
    /// A subagent began, as its own actor within the session.
    SubagentStarted {
        subagent_id: String,
        name: Option<String>,
    },
    /// A subagent finished.
    SubagentEnded { subagent_id: String },
    /// The agent compacted its context. Recorded because everything before it left the transcript.
    ///
    /// Worth a node of its own: everything before a compaction has left the model's context, so a
    /// reader who cannot see where it happened cannot tell which later steps could still have been
    /// informed by earlier ones.
    Compacted,
    /// The session ended. This is the export trigger.
    SessionEnded,
}

/// Classify one Relay event, or `None` when it carries no lineage.
///
/// The overwhelming majority of a real session is `llm.chunk` marks -- 428 of the 445 events in the
/// reference Codex capture -- and they are streaming detail already summarized by the enclosing LLM
/// scope. Dropping them here keeps the rest of the plugin from ever seeing them.
pub fn classify(event: &Event) -> Option<LineageEvent> {
    let metadata = event.metadata();
    match event.kind() {
        "mark" => classify_mark(event, metadata),
        "scope" => classify_scope(event, metadata),
        // ATOF may grow event kinds. An unknown one is not an error.
        _ => None,
    }
}

fn classify_mark(event: &Event, metadata: Option<&Json>) -> Option<LineageEvent> {
    match event.name() {
        "session.start" => Some(LineageEvent::SessionStarted {
            agent: string_at(metadata, "agent_kind").map(str::to_string),
            model: string_at(metadata, "model").map(str::to_string),
        }),
        _ => match string_at(metadata, "hook_event_name") {
            Some("PreCompact" | "PostCompact") => Some(LineageEvent::Compacted),
            Some("SubagentStart") => Some(LineageEvent::SubagentStarted {
                subagent_id: subagent_id(event, metadata)?,
                name: string_at(metadata, "agent_type")
                    .or_else(|| string_at(metadata, "subagent_type"))
                    .map(str::to_string),
            }),
            Some("SubagentStop") => Some(LineageEvent::SubagentEnded {
                subagent_id: subagent_id(event, metadata)?,
            }),
            _ => None,
        },
    }
}

fn classify_scope(event: &Event, metadata: Option<&Json>) -> Option<LineageEvent> {
    let category = event.category().map(EventCategory::as_str)?;
    let is_end = matches!(event.scope_category(), Some(ScopeCategory::End));

    match (category, is_end) {
        // Scope start and end share one UUID, which is what lets the two halves be paired.
        ("llm", false) => event
            .annotated_request()
            .map(|request| LineageEvent::ModelCallStarted {
                call_id: event.uuid().to_string(),
                request: Arc::clone(request),
            }),
        ("llm", true) => Some(LineageEvent::ModelCallEnded {
            call_id: event.uuid().to_string(),
            model: event.model_name().map(str::to_string),
            response: event.annotated_response().map(Arc::clone),
            correlation: Correlation::from_metadata(metadata),
        }),

        ("tool", false) => Some(LineageEvent::ToolCallStarted {
            tool_use_id: event.tool_call_id().map(str::to_string),
            tool_name: event.name().to_string(),
            tool_input: event.data().cloned(),
            correlation: Correlation::from_metadata(metadata),
        }),
        ("tool", true) => Some(LineageEvent::ToolCallEnded {
            tool_use_id: event.tool_call_id().map(str::to_string),
            tool_name: event.name().to_string(),
            result: event.data().cloned(),
            is_error: terminal_status(metadata),
            correlation: Correlation::from_metadata(metadata),
        }),

        // The turn scope is where a prompt arrives. Relay spells the role in metadata rather than in
        // the scope name, which differs per agent (`codex-turn` against `claude-code-turn`).
        ("custom", false) if string_at(metadata, "nemo_relay_scope_role") == Some("turn") => {
            let text = string_at(event.data(), "prompt")?.to_string();
            Some(LineageEvent::PromptSubmitted { text })
        }

        // An `agent` scope is either the session itself or a subagent inside it, and `parent_uuid`
        // separates them: the root is self-parented, a subagent is not.
        //
        // Relay does not forward `SubagentStart` as a mark. It consumes the hook and synthesizes a
        // scope -- `push_scope(name: subagent_name, scope_type: ScopeType::Agent, parent:
        // parent_scope)` in its session manager -- so the subagent *is* the scope, and the
        // hook-name arm in `classify_mark` never fires on Claude Code. Verified two ways: a hook
        // probe against Claude Code 2.1.236 shows SubagentStart and SubagentStop firing, and no
        // mark carrying either name ever reaches a subscriber.
        //
        // Reading the end of one of these as the session ending truncated a nine-turn live session
        // to its last three turns.
        ("agent", false) if is_subagent_scope(event, metadata) => {
            Some(LineageEvent::SubagentStarted {
                subagent_id: event.uuid().to_string(),
                name: subagent_name(event, metadata),
            })
        }
        ("agent", true) if is_subagent_scope(event, metadata) => {
            Some(LineageEvent::SubagentEnded {
                subagent_id: event.uuid().to_string(),
            })
        }
        ("agent", true) => Some(LineageEvent::SessionEnded),

        _ => None,
    }
}

/// Whether a tool call failed, as far as Relay could tell.
///
/// ATOF 0.1 defers a terminal `status` field on scope end, so there is no specified place to read
/// this from. Relay fills the gap in metadata: it takes an explicit `status`/`decision`/`permission`
/// from the hook payload, and otherwise derives one from the hook event name -- `error` for
/// `PostToolUseFailure`, `denied` for a permission denial. Crucially Relay strips null metadata
/// before emitting, so the key is present *only when it knows*.
///
/// That three-valued result is preserved here rather than flattened. `None` means Relay had no
/// evidence either way, which on Codex is every tool call -- its hook schema has no failure event,
/// so a failing command and a succeeding one are indistinguishable. Defaulting that to `false` would
/// turn "we did not observe a failure" into "we observed a success", and attest something nobody saw.
fn terminal_status(metadata: Option<&Json>) -> Option<bool> {
    match string_at(metadata, "status")? {
        "error" | "failed" | "failure" | "denied" => Some(true),
        "ok" | "success" | "allow" | "allowed" => Some(false),
        // A spelling this build does not know. Preserve the uncertainty.
        _ => None,
    }
}

/// The subagent an event belongs to.
///
/// Relay spells this differently depending on where it recovered the identity from, and falls back
/// to the scope UUID when the payload named no subagent at all -- which still gives the session a
/// stable handle for the actor, even though it says nothing about what kind of actor it was.
/// Whether this `agent` scope is a subagent rather than the session's own root scope.
///
/// Relay says so outright. When it synthesizes a subagent scope it merges
/// `{"nemo_relay_scope_role": "subagent"}` into the metadata, exactly as it spells the turn role a
/// few arms above. Prefer the stated role over any inference from it.
///
/// The parent check is the fallback for a host that omits the role: the root scope is
/// self-parented and a subagent is parented to the scope that spawned it. Keeping both means a
/// Relay that stops setting the role degrades to a working heuristic rather than to silence.
fn is_subagent_scope(event: &Event, metadata: Option<&Json>) -> bool {
    if string_at(metadata, "nemo_relay_scope_role") == Some("subagent") {
        return true;
    }
    // Read directly rather than as `!is_none_or(==)`: a scope with a parent that is not itself.
    event
        .parent_uuid()
        .is_some_and(|parent| parent != event.uuid())
}

/// What to call a subagent in the graph.
///
/// Relay's scope name is `subagent:{id}` -- unique and stable, but it tells a reader nothing about
/// what kind of subagent ran. The hook's own metadata is merged into that scope, so the agent type
/// travels with it when the host reports one, and that is the name worth showing.
fn subagent_name(event: &Event, metadata: Option<&Json>) -> Option<String> {
    string_at(metadata, "agent_type")
        .or_else(|| string_at(metadata, "subagent_type"))
        .map(str::to_string)
        .or_else(|| Some(event.name().to_string()).filter(|name| !name.is_empty()))
}

fn subagent_id(event: &Event, metadata: Option<&Json>) -> Option<String> {
    string_at(metadata, "subagent_id")
        .or_else(|| string_at(metadata, "agent_id"))
        .map(str::to_string)
        .or_else(|| Some(event.uuid().to_string()))
}

/// Read a string field from a JSON object that may be absent or may not be an object.
pub(crate) fn string_at<'a>(value: Option<&'a Json>, key: &str) -> Option<&'a str> {
    value?.get(key)?.as_str()
}
