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
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
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
/// collector must never do. Dropping is the least-bad third option, and it is *counted*, so the
/// manifest can state that it happened rather than quietly under-reporting.
const QUEUE_DEPTH: usize = 4096;

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
    /// Subagents seen this session.
    subagents: HashMap<String, AssetRef>,
    /// The subagent currently doing the work, when one is.
    active_subagent: Option<AssetRef>,
    dropped_events: u64,
}

impl SessionState {
    /// Who is doing the work right now: the active subagent, or the root agent.
    ///
    /// Recorded as metadata *on the activity*, never as one of its inputs. Putting an agent in
    /// `inputs` would say the activity consumed the agent; PROV keeps association and usage apart,
    /// and so does this.
    fn actor_name(&self) -> Option<String> {
        self.active_subagent
            .as_ref()
            .or(self.agent.as_ref())
            .map(|asset| asset.as_str().to_string())
    }
}

/// The handle the plugin holds. Dropping it flushes every open session.
///
/// Deliberately not `Debug`: it owns a channel and a thread handle, neither of which says anything
/// useful in a log, and a derived impl on the plugin struct would drag them into one.
pub struct Mailbox {
    sender: Option<SyncSender<Message>>,
    worker: Option<JoinHandle<()>>,
}

impl Mailbox {
    /// Start the worker that owns every recorder.
    pub fn start(manifest_dir: PathBuf, policy: Policy, signer: SignerFactory) -> Self {
        let (sender, receiver) = sync_channel(QUEUE_DEPTH);
        let worker = std::thread::Builder::new()
            .name("eqty-lineage".into())
            .spawn(move || run(receiver, manifest_dir, policy, signer))
            .ok();

        Self {
            sender: Some(sender),
            worker,
        }
    }

    /// Hand one attributed event to the worker, or drop it if the queue is full.
    ///
    /// Returns whether it was accepted, so the caller can count what was lost. Never blocks and
    /// never panics: a full queue and a dead worker are both "not recorded", and neither is worth
    /// taking the agent down for.
    pub fn send(&self, session_id: &str, at: String, event: LineageEvent) -> bool {
        let Some(sender) = &self.sender else {
            return false;
        };
        let message = Message::Observed {
            session_id: session_id.to_string(),
            at,
            event: Box::new(event),
        };
        !matches!(
            sender.try_send(message),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
        )
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

fn run(receiver: Receiver<Message>, manifest_dir: PathBuf, policy: Policy, signer: SignerFactory) {
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
                    recorder: Recorder::new(lineage, policy.clone()),
                    open_tools: HashMap::new(),
                    open_calls: HashMap::new(),
                    agent: None,
                    turn_prompt: None,
                    subagents: HashMap::new(),
                    active_subagent: None,
                    dropped_events: 0,
                },
            );
        }

        let finished = {
            let Some(state) = sessions.get_mut(&session_id) else {
                continue;
            };
            runtime.block_on(apply(state, *event, &at))
        };

        if finished && let Some(state) = sessions.remove(&session_id) {
            export(&runtime, state, &manifest_dir, &session_id);
        }
    }

    // The channel closed. Everything still open is a session whose agent never told us it ended --
    // which on Codex is every session.
    for (session_id, state) in sessions {
        export(&runtime, state, &manifest_dir, &session_id);
    }
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
                    serde_json::json!({ "subagentId": subagent_id, "role": "subagent" }),
                    at,
                )
                .await
            {
                // Delegated work is attributed to the subagent that did it, not to the root agent.
                // A manifest that credited everything to the root would say one actor did work that
                // several actors did, which is the thing a provenance record exists to prevent.
                state.subagents.insert(subagent_id, asset.clone());
                state.active_subagent = Some(asset);
            }
        }
        LineageEvent::SubagentEnded { .. } => {
            state.active_subagent = None;
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
            result,
            correlation,
            ..
        } => {
            let open = tool_use_id
                .as_ref()
                .and_then(|id| state.open_tools.remove(id));
            record_tool(
                state,
                tool_use_id.as_deref(),
                open,
                result,
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
    let caused_by = state.turn_prompt.take();

    let _ = state
        .recorder
        .record_model_call(
            model,
            instructions.as_deref(),
            &prompt,
            completion.as_deref(),
            caused_by,
            details,
            serde_json::json!({
                "computation_type": "model_call",
                "model": model,
                "performedBy": state.actor_name(),
                "observed": observed,
            }),
            observed,
            at,
        )
        .await;
}

async fn record_tool(
    state: &mut SessionState,
    tool_use_id: Option<&str>,
    open: Option<OpenTool>,
    result: Option<Json>,
    observed: bool,
    at: Option<String>,
) {
    let Some(result) = result else {
        return;
    };

    let (mut observations, _attributed) = file_events_from_result(&result, tool_use_id, true);

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

    let mut inputs: Vec<AssetRef> = Vec::new();
    let mut outputs: Vec<AssetRef> = Vec::new();

    // The tool that did the work is an input to it. Naming it as a node rather than as a label makes
    // "which runs used this tool" a question the graph answers, and it is the shape the LangChain
    // and DeepAgents manifests already have.
    if let Some(open) = &open
        && let Ok(tool) = state
            .recorder
            .record_actor(
                "Tool",
                &open.name,
                &format!("The '{}' tool, as invoked by the agent.", open.name),
                serde_json::json!({}),
                at.clone(),
            )
            .await
    {
        inputs.push(tool);
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
    if let Some(name) = open.as_ref().map(|open| open.name.as_str())
        && let Ok(body) = serde_json::to_vec(&result)
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

    // Who performed it. The active subagent when one is running, otherwise the root agent -- so
    // delegated work is credited to the actor that did it rather than to the session as a whole.
    let performed_by = state.actor_name();
    let _ = state
        .recorder
        .record_tool_run(
            &inputs,
            &outputs,
            serde_json::json!({
                "computation_type": "tool_call",
                "tool": open.as_ref().map(|open| open.name.clone()),
                "performedBy": performed_by,
                "observed": observed,
            }),
            at,
        )
        .await;
}

fn export(
    runtime: &tokio::runtime::Runtime,
    state: SessionState,
    manifest_dir: &PathBuf,
    session_id: &str,
) {
    let dropped = state.dropped_events;
    let Ok(manifest) = runtime.block_on(state.recorder.finish(None)) else {
        return;
    };
    let _ = dropped;

    if std::fs::create_dir_all(manifest_dir).is_err() {
        return;
    }
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
    let path = manifest_dir.join(format!("{safe}.json"));
    if let Ok(json) = serde_json::to_vec_pretty(&manifest) {
        let _ = std::fs::write(path, json);
    }
}
