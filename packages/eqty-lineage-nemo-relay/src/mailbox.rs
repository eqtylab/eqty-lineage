//! Getting events off the subscriber thread and into a recorder.
//!
//! Three constraints meet here and between them they decide the whole design.
//!
//! **The subscriber is synchronous and must return promptly.** Relay enqueues subscriber work rather
//! than running it on the agent's critical path, but that guarantee only holds if the callback
//! itself returns. So the callback does the cheapest possible thing: classify, attribute, hand off.
//!
//! **The recorder is async, and single-owner.** Statement construction and signing are `async`, and a
//! recorder is not concurrency-safe by design. So exactly one worker thread owns every recorder and
//! drives them on its own runtime -- an actor, not a shared lock. A lock here would put contention
//! on an event stream, which is the thing the enqueue-and-return contract exists to avoid.
//!
//! **Codex never closes its agent scope.** There is no `SessionEnd` in its plugin hook schema, so
//! `SessionEnded` never arrives and a design that exported only on that event would record every
//! Codex session and write none of them. Shutdown is therefore a real export path, not cleanup:
//! dropping the mailbox flushes every session still open.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use nemo_relay_plugin::{AnnotatedLlmRequest, AnnotatedLlmResponse};
use serde_json::Value as Json;

use crate::classify::LineageEvent;
use crate::files::{FileMode, FileObserved, file_events_from_patch, file_events_from_result};
use crate::lineage::{AssetRef, LineageSession};
use crate::recorder::Recorder;
use crate::redaction::Policy;

/// How many events may be in flight before the subscriber starts dropping them.
///
/// Bounded on purpose. An unbounded queue turns a slow recorder into unbounded memory growth inside
/// the agent's process, and blocking instead would stall the agent -- which is the one thing a
/// collector must never do. Dropping is the least-bad third option, and it is counted per session
/// and reported in that session's coverage as `EventsDropped`, so an incomplete graph says so rather
/// than reading as a complete one.
const QUEUE_DEPTH: usize = 4096;

/// Statements that must accumulate before a mid-session manifest is rewritten.
///
/// A checkpoint costs the whole manifest, not the delta -- see [`LineageSession::snapshot`] -- so
/// writing after every event made a long session quadratic in its own length. The first checkpoint
/// still happens at the first opportunity, because until one exists a crash loses everything.
const CHECKPOINT_EVERY: usize = 24;

/// Blob bytes that widen the checkpoint interval by one statement.
///
/// The interval scales with what a rewrite would cost, so a session holding a large file is
/// checkpointed rarely and a small one often. Without this a 20 MB read makes every subsequent
/// checkpoint copy 20 MB, the worker falls behind, and the queue starts dropping events -- turning a
/// resilience feature into the cause of an incomplete recording.
const CHECKPOINT_BYTES_PER_STATEMENT: usize = 65_536;

/// What the subscriber hands to the worker.
enum Message {
    Observed {
        session_id: String,
        at: String,
        event: Box<LineageEvent>,
    },
}

/// A tool call that has started and not yet ended.
struct OpenTool {
    name: String,
    input: Option<Json>,
}

/// Everything one session accumulates.
struct SessionState {
    recorder: Recorder,
    open_tools: HashMap<String, OpenTool>,
    /// Model calls whose request has arrived and whose response has not, keyed by scope UUID.
    open_calls: HashMap<String, Arc<AnnotatedLlmRequest>>,
    /// The agent that ran the session.
    agent: Option<AssetRef>,
    /// The prompt that opened the current turn, and an input to everything it caused.
    turn_prompt: Option<AssetRef>,
    /// Subagents that have started and not yet finished, in the order they started.
    ///
    /// A list rather than one "active" slot, because a session can fan out: a live run delegated to
    /// four workers at once. With one slot the first `SubagentStop` cleared it, and every later call
    /// by a still-running sibling was credited to the root agent -- the exact mis-attribution
    /// subagent tracking exists to prevent, arriving silently.
    ///
    /// Each entry pairs the instance id with the *kind* node. The node is deduplicated by name, so
    /// `general-purpose` is one node across every session that used one -- which is what makes "what
    /// did this kind of agent do" a graph question. Two parallel workers of the same kind therefore
    /// share it, and the instance travels on each activity instead.
    live_subagents: Vec<(String, AssetRef)>,
    /// Where this session's manifest is written, resolved once when the session opens.
    ///
    /// Held rather than recomputed so that every write for one session lands on one file. Resolving
    /// it per write would either clobber an unrelated session's manifest or, with the
    /// never-overwrite rule, spray a numbered file per turn.
    path: PathBuf,
    /// Statement count at the last write, so an event that changed nothing writes nothing.
    written_at: usize,
}

impl SessionState {
    /// Who is doing the work right now: the sole live subagent, or the root agent.
    ///
    /// Recorded as metadata *on the activity*, never as one of its inputs. Putting an agent in
    /// `inputs` would say the activity consumed the agent; PROV keeps association and usage apart,
    /// and so does this.
    ///
    /// With several subagents live this returns the root agent, because Relay reports that *a*
    /// subagent is running and not which one performed a given call. Guessing the most recent would
    /// produce a specific, checkable, wrong claim -- worse than a general true one, since a reader
    /// would act on it.
    fn actor_name(&self) -> Option<String> {
        match self.live_subagents.as_slice() {
            [(_, only)] => Some(only.as_str().to_string()),
            _ => self.agent.as_ref().map(|asset| asset.as_str().to_string()),
        }
    }

    /// Which instance of the acting subagent, when exactly one is acting.
    ///
    /// `None` for the root agent -- there is only ever one of it, so an instance would say nothing --
    /// and `None` when siblings are live, where the instance is genuinely not known.
    fn actor_instance(&self) -> Option<String> {
        match self.live_subagents.as_slice() {
            [(id, _)] => Some(id.clone()),
            _ => None,
        }
    }

    /// How firmly the performer is known, stated on every activity.
    ///
    /// A reader comparing two manifests needs to tell "the root agent did this" from "we could not
    /// tell which of four workers did this, so it is filed under the root". Both carry the same
    /// `performedBy`; only this distinguishes them.
    fn attribution_basis(&self) -> &'static str {
        match self.live_subagents.len() {
            0 => "root-agent",
            1 => "sole-live-subagent",
            _ => "ambiguous-parallel-subagents",
        }
    }

    /// Note ambiguity in coverage, so the count is visible without walking every activity.
    fn note_attribution(&mut self) {
        if self.live_subagents.len() > 1 {
            self.recorder.note_ambiguous_attribution();
        }
    }
}

/// The handle the plugin holds. Dropping it flushes every open session.
///
/// Deliberately not `Debug`: it owns a channel and a thread handle, neither of which says anything
/// useful in a log, and a derived impl on the plugin struct would drag them into one.
pub struct Mailbox {
    sender: Option<SyncSender<Message>>,
    worker: Option<JoinHandle<()>>,
    /// Events dropped per session, shared with the worker so they reach that session's coverage.
    ///
    /// Counted per session rather than process-wide because the question a reader asks is whether
    /// *this* graph is complete. Behind a lock because the drop path is the one place the subscriber
    /// cannot hand work to the worker -- the queue being full is precisely the condition -- and drops
    /// are rare enough that the lock is never contended in practice.
    dropped: Arc<Mutex<HashMap<String, u64>>>,
}

impl Mailbox {
    /// Start the worker that owns every recorder.
    ///
    /// `on_finished` is called with a session's id once its manifest is written, so state keyed by
    /// session elsewhere in the process can be released. Passed in rather than reached for, because
    /// this module must not know what a [`crate::SessionRouter`] is.
    pub fn start(
        manifest_dir: PathBuf,
        policy: Policy,
        signer: SignerFactory,
        on_finished: SessionFinished,
    ) -> Self {
        let (sender, receiver) = sync_channel(QUEUE_DEPTH);
        let dropped: Arc<Mutex<HashMap<String, u64>>> = Arc::new(Mutex::new(HashMap::new()));
        let counts = Arc::clone(&dropped);
        let worker = std::thread::Builder::new()
            .name("eqty-lineage".into())
            .spawn(move || run(receiver, manifest_dir, policy, signer, counts, on_finished))
            .ok();

        Self {
            sender: Some(sender),
            worker,
            dropped,
        }
    }

    /// Hand one attributed event to the worker, or drop it if the queue is full.
    ///
    /// Returns whether it was accepted, so the caller can count what was lost. Never blocks and
    /// never panics: a full queue and a dead worker are both "not recorded", and neither is worth
    /// taking the agent down for. A drop is also recorded against the session, so the manifest can
    /// say it is incomplete instead of reading as a complete short recording.
    pub fn send(&self, session_id: &str, at: String, event: LineageEvent) -> bool {
        let Some(sender) = &self.sender else {
            self.note_dropped(session_id);
            return false;
        };
        let message = Message::Observed {
            session_id: session_id.to_string(),
            at,
            event: Box::new(event),
        };
        if matches!(
            sender.try_send(message),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
        ) {
            self.note_dropped(session_id);
            return false;
        }
        true
    }

    /// Count one event this session will never see.
    ///
    /// A poisoned lock is recovered from rather than propagated: the map behind it is counters, and
    /// giving up on counting losses is the one response strictly worse than the loss itself.
    fn note_dropped(&self, session_id: &str) {
        let mut counts = self
            .dropped
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *counts.entry(session_id.to_string()).or_insert(0) += 1;
    }
}

impl Drop for Mailbox {
    /// Flush every session still open, then wait for the worker to finish writing.
    ///
    /// Joining rather than detaching is what makes this correct: the library can be unloaded
    /// immediately after the component is deactivated, and a detached thread would be writing
    /// manifests into an address space that is being torn down. On Codex this is also the *only*
    /// path that writes anything.
    fn drop(&mut self) {
        // Closing the channel is the shutdown signal; the worker's loop ends when it drains.
        self.sender = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// How the worker mints a signer for a new session.
///
/// A function rather than a signer, because each session gets its own recorder and the plugin must
/// not hold process-global signing state -- the mistake `integrity-py`'s `active_signer` makes.
pub type SignerFactory = Box<dyn Fn() -> Option<LineageSession> + Send>;

/// Told that a session is finished and its manifest written.
///
/// Exists so the scope-to-session map that feeds this mailbox can be pruned. Without it a
/// long-running gateway holds one entry per scope for every session it ever saw, and the great
/// majority of a session's scopes are streaming chunks -- 428 of 445 events in the reference
/// capture -- so the map grows fastest in the process least able to afford it.
pub type SessionFinished = Box<dyn Fn(&str) + Send>;

fn run(
    receiver: Receiver<Message>,
    manifest_dir: PathBuf,
    policy: Policy,
    signer: SignerFactory,
    dropped: Arc<Mutex<HashMap<String, u64>>>,
    on_finished: SessionFinished,
) {
    // A current-thread runtime: this thread is the only one driving these futures, and a
    // multi-threaded pool would add threads to a process we are only supposed to observe.
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return;
    };

    let mut sessions: HashMap<String, SessionState> = HashMap::new();

    while let Ok(message) = receiver.recv() {
        let Message::Observed {
            session_id,
            at,
            event,
        } = message;

        if !sessions.contains_key(&session_id) {
            let Some(lineage) = signer() else {
                continue;
            };
            sessions.insert(
                session_id.clone(),
                SessionState {
                    path: manifest_path(&manifest_dir, &session_id),
                    written_at: 0,
                    recorder: Recorder::new(lineage, policy.clone()),
                    open_tools: HashMap::new(),
                    open_calls: HashMap::new(),
                    agent: None,
                    turn_prompt: None,
                    live_subagents: Vec::new(),
                },
            );
        }

        // Checked before `apply` consumes the event. Without a checkpoint the only manifest a
        // session ever produces is written at the very end, so a crash, a kill, or a machine losing
        // power takes the whole recording with it -- and nothing is visible while the agent works.
        let boundary = completes_work(&event);

        let finished = {
            let Some(state) = sessions.get_mut(&session_id) else {
                continue;
            };
            let finished = runtime.block_on(apply(state, *event, &at));
            if !finished && boundary {
                runtime.block_on(checkpoint(state));
            }
            finished
        };

        if finished && let Some(state) = sessions.remove(&session_id) {
            export(&runtime, &session_id, state, &dropped);
            on_finished(&session_id);
        }
    }

    // The channel closed. Everything still open is a session whose agent never told us it ended --
    // which on Codex is every session.
    for (session_id, state) in sessions {
        export(&runtime, &session_id, state, &dropped);
        on_finished(&session_id);
    }
}

/// Whether this event completed a unit of work, and so is worth checkpointing after.
///
/// Ends, not starts: a manifest written between a tool's start and its end holds the arguments of a
/// call whose result is still coming, and rewriting the whole document to capture that is cost
/// without a reader.
fn completes_work(event: &LineageEvent) -> bool {
    matches!(
        event,
        LineageEvent::PromptSubmitted { .. }
            | LineageEvent::ModelCallEnded { .. }
            | LineageEvent::ToolCallEnded { .. }
            | LineageEvent::SubagentEnded { .. }
            | LineageEvent::Compacted
    )
}

/// Apply one event. Returns whether the session ended.
async fn apply(state: &mut SessionState, event: LineageEvent, at: &str) -> bool {
    let at = Some(at.to_string());
    match event {
        LineageEvent::SessionStarted { agent, model } => {
            let name = agent.clone().unwrap_or_else(|| "unknown-agent".into());
            if let Ok(asset) = state
                .recorder
                .record_actor(
                    "Agent",
                    &name,
                    &format!("The coding agent '{name}' that ran this session."),
                    serde_json::json!({ "model": model }),
                    at,
                )
                .await
            {
                state.agent = Some(asset);
            }
        }
        LineageEvent::PromptSubmitted { text } => {
            // The turn's instruction. Everything the agent does afterwards is downstream of it, so
            // it becomes the input the turn's activities hang from -- without it a manifest attests
            // what an agent did and not what it was asked to do.
            if let Ok(asset) = state
                .recorder
                .register_payload(
                    "Prompt",
                    "user prompt",
                    "The instruction that opened this turn.",
                    text.as_bytes(),
                    serde_json::json!({ "role": "user" }),
                    at,
                )
                .await
            {
                state.turn_prompt = Some(asset);
            }
        }
        LineageEvent::SubagentStarted { subagent_id, name } => {
            let label = name.unwrap_or_else(|| "subagent".into());
            if let Ok(asset) = state
                .recorder
                .record_actor(
                    "Agent",
                    &label,
                    &format!("The subagent '{label}', running inside this session."),
                    // No instance id here: this node is the subagent's *kind*, and naming one
                    // instance on a node that stands for several would be false.
                    serde_json::json!({ "role": "subagent" }),
                    at,
                )
                .await
            {
                // Delegated work is attributed to the subagent that did it, not to the root agent.
                // A manifest that credited everything to the root would say one actor did work that
                // several actors did, which is the thing a provenance record exists to prevent.
                state.live_subagents.push((subagent_id, asset));
            }
        }
        LineageEvent::SubagentEnded { subagent_id } => {
            // Retire the one that ended, by id. Clearing unconditionally would end the attribution
            // of every sibling still running, and with parallel workers the first stop arrives long
            // before the last one finishes.
            state
                .live_subagents
                .retain(|(live, _)| *live != subagent_id);
        }
        LineageEvent::Compacted => {
            // A compaction is a real transformation of the agent's context: everything before it has
            // left the model's window. Recorded as an activity so a reader can see which later steps
            // could no longer have been informed by earlier ones.
            let _ = state.recorder.record_compaction(at).await;
        }
        // Keyed by tool-call id when there is one. Relay synthesizes ids for post-only hooks, so a
        // missing one means the pre hook never arrived and the end will have to stand alone.
        LineageEvent::ToolCallStarted {
            tool_use_id: Some(id),
            tool_name,
            tool_input,
            ..
        } => {
            state.open_tools.insert(
                id,
                OpenTool {
                    name: tool_name,
                    input: tool_input,
                },
            );
        }
        LineageEvent::ToolCallEnded {
            tool_use_id,
            tool_name,
            result,
            is_error,
            correlation,
        } => {
            let open = tool_use_id
                .as_ref()
                .and_then(|id| state.open_tools.remove(id));
            record_tool(
                state,
                tool_use_id.as_deref(),
                &tool_name,
                open,
                result,
                is_error,
                correlation.is_observed(),
                at,
            )
            .await;
        }
        LineageEvent::ModelCallStarted { call_id, request } => {
            state.open_calls.insert(call_id, request);
        }
        LineageEvent::ModelCallEnded {
            call_id,
            model,
            response,
            correlation,
        } => {
            record_model_call(
                state,
                &call_id,
                model.as_deref(),
                response,
                correlation.is_observed(),
                at,
            )
            .await;
        }
        LineageEvent::SessionEnded => return true,
        // Prompts, subagents and compaction are classified and counted but are not yet nodes.
        // Recording them badly would be worse than the gap.
        _ => {}
    }
    false
}

/// Recover a path from a tool's arguments when its result carried none.
///
/// Claude Code spells the input in snake case (`file_path`) and the result in camel case
/// (`filePath`); both are checked because a tool that only reports one of them still moved a file.
fn observation_from_arguments(
    open: Option<&OpenTool>,
    tool_use_id: Option<&str>,
) -> Option<FileObserved> {
    let open = open?;
    let input = open.input.as_ref()?;
    let path = ["file_path", "filePath", "path", "notebook_path"]
        .iter()
        .find_map(|key| input.get(*key).and_then(Json::as_str))
        .filter(|path| !path.is_empty())?;

    // Named by a tool whose purpose is to write, so the file is an output; anything else is a read.
    let writes = matches!(
        open.name.as_str(),
        "Write" | "Edit" | "MultiEdit" | "NotebookEdit" | "apply_patch"
    );

    Some(FileObserved {
        path: path.to_string(),
        content: None,
        mode: if writes {
            FileMode::Wrote
        } else {
            FileMode::Read
        },
        tool_use_id: tool_use_id.map(str::to_string),
        user_modified: false,
        edit: None,
    })
}

/// Pair a completed model call with the request that opened it, and record it.
///
/// A response whose request never arrived is dropped rather than recorded half-way. The prompt is
/// what makes a completion attributable -- a manifest carrying model output with no record of what
/// was asked attests that the model said something, not that it was asked anything.
async fn record_model_call(
    state: &mut SessionState,
    call_id: &str,
    model: Option<&str>,
    response: Option<Arc<AnnotatedLlmResponse>>,
    observed: bool,
    at: Option<String>,
) {
    let Some(request) = state.open_calls.remove(call_id) else {
        return;
    };

    // Serialized from Relay's *normalized* types rather than from provider JSON. That is what makes
    // the CID portable: the same conversation through Anthropic and through OpenAI Responses
    // normalizes to the same messages and therefore hashes the same, so two sessions on different
    // providers join on the prompt rather than forking on its wire format.
    let Ok(prompt) = serde_json::to_vec(&request.messages) else {
        return;
    };
    let instructions = request
        .instructions
        .as_ref()
        .and_then(|instructions| serde_json::to_vec(instructions).ok());

    let completion = response.as_ref().and_then(|response| {
        serde_json::to_vec(&serde_json::json!({
            "message": response.message,
            "tool_calls": response.tool_calls,
        }))
        .ok()
    });

    let details = match &response {
        Some(response) => serde_json::json!({
            "model": response.model.as_deref().or(model),
            "finishReason": response.finish_reason,
            "usage": response.usage,
            "observed": observed,
        }),
        None => serde_json::json!({ "model": model, "observed": observed }),
    };

    // The instruction that opened the turn is an input to the call it caused, and only to the first
    // one: repeating it on every subsequent call would assert that the user asked the same thing
    // several times, when the later calls were caused by the tool results in between.
    //
    // Cloned rather than taken, because a call whose response never arrived is not recorded at all.
    // Taking it there would spend the prompt on an activity that was never written, orphaning the
    // prompt node and leaving the next call in the turn with nothing to say what it was asked.
    let caused_by = state.turn_prompt.clone();

    let describes = serde_json::json!({
        "computation_type": "model_call",
        "model": model,
        "performedBy": state.actor_name(),
        "performedByInstance": state.actor_instance(),
        "attribution": state.attribution_basis(),
        "observed": observed,
    });
    state.note_attribution();

    let recorded = state
        .recorder
        .record_model_call(
            model,
            instructions.as_deref(),
            &prompt,
            completion.as_deref(),
            caused_by,
            details,
            describes,
            observed,
            at,
        )
        .await;

    if matches!(recorded, Ok(true)) {
        state.turn_prompt = None;
    }
}

#[allow(clippy::too_many_arguments)]
async fn record_tool(
    state: &mut SessionState,
    tool_use_id: Option<&str>,
    tool_name: &str,
    open: Option<OpenTool>,
    result: Option<Json>,
    is_error: Option<bool>,
    observed: bool,
    at: Option<String>,
) {
    // The end event names the tool even when its start never reached us. Relay synthesizes ids for
    // post-only hooks, and an end whose id matches no stored start leaves `open` empty -- so relying
    // on the start alone dropped the whole call. For a read-shaped result that is worse than it
    // sounds: with no Tool actor and no result output the activity has no outputs at all and is
    // refused, leaving the file node in the graph with nothing to say which run produced it.
    let name = open
        .as_ref()
        .map(|open| open.name.clone())
        .or_else(|| Some(tool_name.to_string()))
        .filter(|name| !name.is_empty());

    // Whether it failed, including "the host did not say". Recorded before anything can return
    // early, because an unrecorded outcome is exactly the case coverage exists to expose.
    state.recorder.note_tool_outcome(is_error);

    let (mut observations, _attributed) = match &result {
        Some(result) => file_events_from_result(result, tool_use_id, true),
        None => {
            // A tool end carrying no payload at all. The arguments may still name a path, so the
            // call is recorded from what is known rather than abandoned -- and returning here also
            // skipped the coverage note, so the call was invisible *and* uncounted, which is the one
            // outcome a reader cannot detect.
            state.recorder.note_tool_without_result();
            (Vec::new(), None)
        }
    };

    // Codex edits files by handing a patch document to the shell, so nothing above sees it. The
    // patch is in the tool's *arguments*, not its result -- another case where having both halves
    // of the scope is what makes the lineage recoverable at all.
    if observations.is_empty()
        && let Some(open) = &open
        && let Some(patch) = open
            .input
            .as_ref()
            .and_then(|input| {
                ["command", "patch", "input"]
                    .iter()
                    .find_map(|key| input.get(*key))
            })
            .and_then(Json::as_str)
        && patch.contains("*** Begin Patch")
    {
        observations = file_events_from_patch(patch, tool_use_id);
    }

    if observations.is_empty() {
        // The result said nothing about a file. The *arguments* still might: Relay passes the tool
        // input through verbatim, so a `Write` or `Read` names its path there even when the result
        // is a bare string. That yields an identity-only node -- this path was touched, content not
        // established -- which is worth more than silence and is honest about what was seen.
        if let Some(observation) = observation_from_arguments(open.as_ref(), tool_use_id) {
            observations.push(observation);
        }
    }

    // Say so when a tool call told us nothing about files. Absence and silence look identical in a
    // graph otherwise, and on Codex -- which has no read tool, so every read is a shell command --
    // that is the difference between "read no files" and "its reads were invisible to us".

    if observations.is_empty() {
        state.recorder.note_no_file_observation();
    }

    let mut inputs: Vec<AssetRef> = Vec::new();
    let mut outputs: Vec<AssetRef> = Vec::new();

    // The tool that did the work is an input to it. Naming it as a node rather than as a label makes
    // "which runs used this tool" a question the graph answers, and it is the shape the LangChain
    // and DeepAgents manifests already have.
    if let Some(name) = &name
        && let Ok(tool) = state
            .recorder
            .record_actor(
                "Tool",
                name,
                &format!("The '{name}' tool, as invoked by the agent."),
                serde_json::json!({}),
                at.clone(),
            )
            .await
    {
        inputs.push(tool);
    }
    // What the tool was invoked with. Without it the graph says "Bash ran and produced this output"
    // and never what command ran -- and on a real session most of the work went through Bash, so
    // that is most of the information missing. The arguments are an input for the same reason the
    // result is an output: the run consumed them.
    //
    // `register_payload` applies the size policy, so a large `Write` body is withheld and recorded
    // by CID rather than inlined.
    if let Some(open) = &open
        && let Some(input) = open.input.as_ref()
        && let Ok(body) = serde_json::to_vec(input)
        && let Ok(asset) = state
            .recorder
            .register_payload(
                "Dataset",
                &format!("{} input", open.name),
                &format!("What the '{}' tool was invoked with.", open.name),
                &body,
                serde_json::json!({ "toolUseId": tool_use_id }),
                at.clone(),
            )
            .await
    {
        inputs.push(asset);
    }

    for observation in &observations {
        // A failed registration must not take the session down with it: the observation is skipped
        // and the count survives in coverage, so the manifest still says something was seen and not
        // recorded.
        if let Ok(Some(asset)) = state
            .recorder
            .observe_file(observation, observed, at.clone())
            .await
        {
            match observation.mode {
                FileMode::Read => inputs.push(asset),
                FileMode::Wrote => outputs.push(asset),
            }
        }
    }

    // The tool's own result is an output of the run, which is the shape the shipped manifests have:
    // `[Tool, reads...] -> [result, writes...]`. Without it a read-only call has no output at all
    // and is refused as an activity -- so a `Read` would leave a file node dangling with no record
    // of the run that produced it.
    if let Some(name) = &name
        && let Some(result) = &result
        && let Ok(body) = serde_json::to_vec(result)
        && let Ok(asset) = state
            .recorder
            .register_payload(
                "Dataset",
                &format!("{name} result"),
                &format!("What the '{name}' tool returned."),
                &body,
                serde_json::json!({ "toolUseId": tool_use_id, "observed": observed }),
                at.clone(),
            )
            .await
    {
        outputs.push(asset);
    }

    // Who performed it. The sole live subagent when there is one, otherwise the root agent -- so
    // delegated work is credited to the actor that did it rather than to the session as a whole.
    let performed_by = state.actor_name();
    let performed_by_instance = state.actor_instance();
    let attribution = state.attribution_basis();
    // Three outcomes, not two. `null` is the honest record for a host that never said -- which on
    // Codex is every call -- and collapsing it into "succeeded" would attest a clean run over one
    // whose failures were simply invisible to us.
    let outcome = match is_error {
        Some(true) => Some("failed"),
        Some(false) => Some("succeeded"),
        None => None,
    };
    let describes = serde_json::json!({
        "computation_type": "tool_call",
        "tool": name,
        "outcome": outcome,
        "performedBy": performed_by,
        "performedByInstance": performed_by_instance,
        "attribution": attribution,
        "observed": observed,
    });
    state.note_attribution();

    let _ = state
        .recorder
        .record_tool_run(&inputs, &outputs, describes, at)
        .await;
}

/// Where one session's manifest lives, resolved once when the session opens.
///
/// Never clobbers an existing file. One session should produce one manifest, but "should" is doing
/// work there: a stray `SessionEnded` makes the router forget the session, and everything after it
/// lands in a fresh recorder writing under the same name. Overwriting turns that into silent data
/// loss -- the survivor reads as a complete short session rather than the tail of a truncated one.
/// A subagent scope end caused exactly that before it was fixed, and the next cause will not
/// announce itself either.
fn manifest_path(manifest_dir: &Path, session_id: &str) -> PathBuf {
    // Session ids come from the agent and end up in a path, so anything separator-shaped is
    // replaced rather than trusted.
    let safe: String = session_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let mut path = manifest_dir.join(format!("{safe}.json"));
    for sequence in 1..1000 {
        if !path.exists() {
            break;
        }
        path = manifest_dir.join(format!("{safe}.{sequence}.json"));
    }
    path
}

/// Replace `path` with `bytes`, or leave what is already there untouched.
///
/// Written to a sibling and renamed, because a checkpoint runs while an agent is working and a
/// reader may open the file at any moment. A partial write would hand them a truncated JSON
/// document, which is worse than the slightly older complete one it replaced.
fn write_atomically(path: &Path, bytes: &[u8]) {
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let temporary = path.with_extension("json.writing");
    if std::fs::write(&temporary, bytes).is_err() {
        let _ = std::fs::remove_file(&temporary);
        return;
    }
    if std::fs::rename(&temporary, path).is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
}

/// Write what has been recorded so far, so the session is not all-or-nothing.
///
/// The checkpoint carries no coverage node. That is the signal a reader needs: coverage states what
/// a recording could not see, which is only knowable once it has stopped, so a manifest without one
/// was written mid-session and may still grow.
async fn checkpoint(state: &mut SessionState) {
    let count = state.recorder.statement_count();
    if count == state.written_at {
        return;
    }
    // The first one happens as soon as there is anything to write: until a manifest exists on disk,
    // a crash loses the session outright. After that the interval widens with the cost of a rewrite,
    // because a snapshot copies the entire recording rather than the part that changed.
    if state.written_at > 0 {
        let interval =
            CHECKPOINT_EVERY.max(state.recorder.blob_bytes() / CHECKPOINT_BYTES_PER_STATEMENT);
        if count - state.written_at < interval {
            return;
        }
    }
    let Ok(manifest) = state.recorder.snapshot().await else {
        return;
    };
    if let Ok(json) = serde_json::to_vec_pretty(&manifest) {
        write_atomically(&state.path, &json);
        state.written_at = count;
    }
}

fn export(
    runtime: &tokio::runtime::Runtime,
    session_id: &str,
    mut state: SessionState,
    dropped: &Arc<Mutex<HashMap<String, u64>>>,
) {
    // What the queue lost for this session, taken out of the shared map so it does not outlive the
    // session it describes -- the same unbounded-growth mistake this export path exists to close.
    let lost = dropped
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(session_id)
        .unwrap_or(0);
    state.recorder.note_events_dropped(lost);

    // A session that never registered an agent is not a session anyone ran.
    //
    // Codex issues an ancillary model call to title the conversation, through a different provider
    // and under a session id of its own, so one interactive session produced two manifests: 432
    // statements of work, and 20 statements holding `{"title":"Create report.md"}`. Counting
    // manifests to count sessions gets the wrong answer, and the fragment attests a model call
    // performed by nobody -- it has no agent, so every computation in it lacks `performedBy`.
    //
    // It is still written, under a name that says what it is. Dropping it would lose a real model
    // call, and would silently record nothing at all if a host ever stopped emitting
    // `session.start` -- the failure this recorder is least willing to have.
    let path = if state.agent.is_none() {
        state.path.with_extension("unattributed.json")
    } else {
        state.path.clone()
    };
    let checkpoint = state.path.clone();
    let Ok(manifest) = runtime.block_on(state.recorder.finish(None)) else {
        return;
    };
    if let Ok(json) = serde_json::to_vec_pretty(&manifest) {
        write_atomically(&path, &json);
        // The checkpoints went to the session-shaped name before we knew this was a fragment.
        if path != checkpoint {
            let _ = std::fs::remove_file(&checkpoint);
        }
    }
}
