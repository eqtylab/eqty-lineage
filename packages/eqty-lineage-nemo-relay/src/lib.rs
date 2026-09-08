//! EQTY lineage as a NeMo Relay native plugin.
//!
//! Relay installs itself into Claude Code and Codex, receives their lifecycle hooks, and proxies
//! their LLM traffic. Internally it emits one stream of events and lets plugins subscribe to it --
//! which is how its own ATOF and ATIF exporters are built. This crate registers a third subscriber
//! and turns that stream into a signed, content-addressed EQTY manifest, written beside Relay's own
//! `atof/` and `atif/` output.
//!
//! # Why a native plugin
//!
//! Relay offers three execution models. Language bindings run inside the *application* process,
//! which for a coding agent is NVIDIA's prebuilt `nemo-relay` binary -- nothing of ours can be
//! injected there. A gRPC worker runs out of process, which is the safe choice for a component with
//! heavy dependencies. A native `rust_dynamic` component runs in the Relay process itself, with no
//! process hop and no JSON envelope, and it is the only model that can see
//! [`Event::annotated_request`] -- the typed LLM request object, rather than the serialized `data`
//! that file consumers get. That fidelity is the reason for the cost.
//!
//! # The cost, and the two rules it imposes
//!
//! In-process and not sandboxed means this crate can take the agent down with it.
//!
//! 1. **Never panic across the FFI boundary.** A Rust panic unwinding through `extern "C"` is
//!    undefined behavior. Every callback body is wrapped in [`catch_unwind`]. A collector that
//!    crashes the agent it observes is worse than no collector.
//! 2. **Never block in the subscriber.** Relay enqueues subscriber work and returns without waiting,
//!    so a slow recorder cannot stall the agent -- but only as long as the callback itself returns
//!    promptly. Classify and hand off; do the recording elsewhere.

mod classify;
mod config;
mod files;
mod lineage;
mod mailbox;
mod recorder;
mod redaction;
mod session;

pub use classify::{Correlation, LineageEvent, classify};
pub use config::Config;
pub use files::{
    EditAttempt, FileMode, FileObserved, ReplayRefusal, apply_edit, apply_line_edit,
    file_events_from_patch, file_events_from_result,
};
pub use lineage::{AssetRef, LineageSession};
pub use mailbox::{Mailbox, SessionFinished, SignerFactory};
pub use recorder::Recorder;
pub use redaction::{Disposition, Policy, glob_match};
pub use session::SessionRouter;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use integrity::signer::{SignerType, ed25519_signer::Ed25519Signer};
use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Event, Json, NativePlugin, PluginContext, Result,
    nemo_relay_plugin,
};
use serde_json::Map;

/// The stable plugin kind. This string must equal `[plugin] id` in `relay-plugin.toml`, and it is
/// what `components[].kind` in a host's `plugins.toml` refers to.
const PLUGIN_KIND: &str = "eqty.lineage";

/// The subscriber name Relay registers us under, alongside its own `atof` and `atif`.
const SUBSCRIBER_NAME: &str = "eqty_lineage";

/// Counts of what the subscriber has seen, shared with every registered component.
///
/// This is the Phase 2 stand-in for the recorder: it proves the event stream arrives, that
/// classification runs against real events, and that neither panics -- without yet building a graph.
#[derive(Debug, Default)]
pub struct Tally {
    /// Events delivered to the subscriber.
    pub seen: AtomicU64,
    /// Events that classified into the lineage vocabulary.
    pub classified: AtomicU64,
    /// Classified events that could not be attributed to any session, and were dropped.
    pub unattributed: AtomicU64,
    /// Events dropped because the recorder's queue was full. Blocking instead would stall the agent.
    ///
    /// Process-wide, and for the diagnostics surface only. The count a reader needs is per session
    /// and lives in that session's coverage node as `EventsDropped`; this one cannot say which
    /// manifest is incomplete.
    pub queue_overflow: AtomicU64,
    /// Callback invocations that panicked and were contained.
    pub panicked: AtomicU64,
}

/// The plugin object Relay owns for the lifetime of the component.
#[derive(Default)]
pub struct EqtyLineagePlugin {
    tally: Arc<Tally>,
    /// One of the handles keeping the mailbox alive; the subscriber closure Relay owns holds another.
    ///
    /// So dropping the plugin does *not* by itself flush anything -- the flush happens when the last
    /// `Arc` goes, which is whenever Relay releases the subscriber. Relay has no explicit teardown
    /// hook, and on Codex that release is the only thing that ever writes a manifest. Held here so
    /// the mailbox survives at least as long as the plugin, not because this handle is the one that
    /// ends it.
    mailbox: Option<Arc<Mailbox>>,
}

impl EqtyLineagePlugin {
    /// Construct a plugin with a fresh tally.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the tally, for tests and for the diagnostics surface.
    pub fn tally(&self) -> &Arc<Tally> {
        &self.tally
    }
}

impl NativePlugin for EqtyLineagePlugin {
    fn plugin_kind(&self) -> &str {
        PLUGIN_KIND
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let (_, diagnostics) = Config::parse(plugin_config);
        diagnostics
    }

    fn register(
        &mut self,
        plugin_config: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> Result<()> {
        // Registration is the last place a bad config can be refused. Relay calls `validate` first,
        // but a component can also be configured by hand-writing the TOML block, which skips it.
        let (config, diagnostics) = Config::parse(plugin_config);
        // Only errors refuse the registration. A warning says a setting will not do what its name
        // suggests, which is worth surfacing and not worth declining to record over.
        if let Some(problem) = diagnostics
            .iter()
            .find(|problem| problem.level == DiagnosticLevel::Error)
        {
            return Err(problem.message.clone());
        }

        let policy = Policy::new(config.deny_globs.clone(), config.max_content_bytes);
        // A factory rather than one signer: each session gets its own, so nothing signing-related is
        // shared process-wide. Signer creation can fail, and a session that cannot be signed is not
        // recorded rather than recorded unsigned.
        let signer: SignerFactory = Box::new(|| {
            Ed25519Signer::create()
                .ok()
                .map(|signer| LineageSession::new(SignerType::ED25519(signer)))
        });
        // `register_subscriber` takes `Fn`, not `FnMut`, so the router's state lives behind a lock.
        // Relay drives subscribers from its own queue rather than from the agent's critical path, so
        // this is uncontended in practice -- but it must never be held across anything slow.
        //
        // Shared with the mailbox as well as the subscriber, because the router accumulates one
        // entry per scope and only the mailbox knows when a session is over. The alternative -- a
        // map that grows for the life of the process -- is a leak in exactly the long-lived gateway
        // this plugin is meant to be safe inside.
        let router = Arc::new(Mutex::new(SessionRouter::new()));

        let finished = Arc::clone(&router);
        let mailbox = Arc::new(Mailbox::start(
            config.manifest_dir.clone(),
            policy,
            signer,
            Box::new(move |session_id: &str| {
                finished
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .forget(session_id);
            }),
        ));
        self.mailbox = Some(Arc::clone(&mailbox));

        let tally = Arc::clone(&self.tally);
        ctx.register_subscriber(SUBSCRIBER_NAME, move |event: &Event| {
            observe(&tally, &router, &mailbox, event);
        })?;

        Ok(())
    }
}

/// Handle one event, containing any panic rather than letting it cross the ABI boundary.
///
/// The SDK wraps the plugin *entry* symbol in `catch_unwind`, but not each subscriber invocation --
/// that is ours to do, and it is the difference between a dropped event and a dead agent.
fn observe(
    tally: &Arc<Tally>,
    router: &Mutex<SessionRouter>,
    mailbox: &Arc<Mailbox>,
    event: &Event,
) {
    tally.seen.fetch_add(1, Ordering::Relaxed);

    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // Attribute every event, including the ones that carry no lineage themselves: an LLM scope
        // start classifies to nothing, but it is the parent that lets its children be attributed.
        //
        // A poisoned lock is recovered from rather than propagated. Poisoning means an earlier
        // callback panicked while holding it -- already counted below -- and the router behind it is
        // a map that may have missed one insert, not corrupt state. Treating poisoning as fatal
        // would turn one dropped event into a session that silently records nothing from then on.
        let session_id = router
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .attribute(event);
        // Relay's own timestamp, never wall-clock at ingest: the two capture paths must not disagree
        // on when something happened just because one of them replayed it later.
        let at = event.timestamp().to_rfc3339();
        classify(event).map(|lineage| (session_id, at, lineage))
    }));

    match outcome {
        Ok(Some((Some(session_id), at, lineage))) => {
            tally.classified.fetch_add(1, Ordering::Relaxed);
            // Hand off and return. Everything expensive -- hashing, signing, writing -- happens on
            // the mailbox's own thread, so a slow recorder cannot become a slow agent.
            if !mailbox.send(&session_id, at, lineage) {
                tally.queue_overflow.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(Some((None, _at, _lineage))) => {
            tally.unattributed.fetch_add(1, Ordering::Relaxed);
        }
        Ok(None) => {}
        Err(_) => {
            tally.panicked.fetch_add(1, Ordering::Relaxed);
        }
    }
}

nemo_relay_plugin!(nemo_relay_register_plugin, EqtyLineagePlugin::new);
