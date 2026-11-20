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
//! Of Relay's three execution models, language bindings run inside the prebuilt `nemo-relay` binary
//! where nothing of ours can be injected, and a gRPC worker runs out of process. A native
//! `rust_dynamic` component runs in the Relay process itself and is the only model that can see
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
mod notary;
mod provider;
mod recorder;
mod redaction;
mod session;
mod subscriber;
use classify::event_may_carry_lineage;
pub use classify::{CompactionPhase, Correlation, LineageEvent, classify, may_carry_lineage};
pub use config::Config;
pub use files::{
    EditAttempt, FileMode, FileObserved, Hunk, ReplayRefusal, apply_edit, apply_line_edit,
    file_events_from_patch, file_events_from_result, paths_in_patch,
};
pub use lineage::{AssetRef, LineageSession};
pub use mailbox::{
    MANIFEST_MARK, Mailbox, ManifestAnnounced, ManifestMark, SessionFinished, SignerFactory,
};
pub use notary::{NOTARY_URL, session_factory};
pub use provider::Provider;
pub use recorder::Recorder;
pub use redaction::{Disposition, Policy, glob_match};
pub use session::SessionRouter;
use subscriber::Delivery;
pub use subscriber::decode_event;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Event, Json, NativePlugin, PluginContext, PluginRuntime,
    Result, nemo_relay_plugin,
};
use serde_json::{Map, json};

/// The stable plugin kind. This string must equal `[plugin] id` in `relay-plugin.toml`, and it is
/// what `components[].kind` in a host's `plugins.toml` refers to.
const PLUGIN_KIND: &str = "eqty.lineage";

/// The subscriber name Relay registers us under, alongside its own `atof` and `atif`.
const SUBSCRIBER_NAME: &str = "eqty_lineage";

/// Counts of what the subscriber has seen, shared with every registered component.
#[derive(Debug, Default)]
pub struct Tally {
    /// Events delivered to the subscriber.
    pub seen: AtomicU64,
    /// Events whose JSON could not be decoded into an `Event`, and so were never seen.
    pub undecodable: AtomicU64,
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
    /// Manifests written whose location could not be announced on the host's event stream.
    ///
    /// The manifest is on disk regardless -- this counts recordings a consumer can only find by
    /// path convention. Silent would be the wrong default for the one step whose entire purpose is
    /// making a recording discoverable.
    pub unannounced: AtomicU64,
}

/// The plugin object Relay owns for the lifetime of the component.
#[derive(Default)]
pub struct EqtyLineagePlugin {
    tally: Arc<Tally>,
    /// The subscriber also owns a handle, so final export waits for the last `Arc` to drop.
    /// Open sessions can already have checkpoints on disk.
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
        // A factory rather than one signer: each session gets its own signing state. A notary's
        // identity is shared by all of them; a software key is made per session. Signer creation
        // can fail, and a session that cannot be signed is not recorded rather than recorded
        // unsigned.
        let signer = notary::session_factory(notary::url_from_env());
        // The subscriber is `Fn`, not `FnMut`, so the router's state lives behind a lock.
        // Relay drives subscribers from its own queue rather than from the agent's critical path, so
        // this is uncontended in practice -- but it must never be held across anything slow.
        //
        // Shared with the mailbox as well as the subscriber, because the router accumulates one
        // entry per scope and only the mailbox knows when a session is over. The alternative -- a
        // map that grows for the life of the process -- is a leak in exactly the long-lived gateway
        // this plugin is meant to be safe inside.
        let router = Arc::new(Mutex::new(SessionRouter::new()));

        let finished = Arc::clone(&router);
        // `PluginRuntime` owns the host table by value and retains a reference-counted host
        // capability, and the SDK declares it `Send + Sync` -- so it can be moved onto the worker
        // thread, which is the only place a final manifest is written. `ctx` itself cannot: it
        // borrows the host table for the duration of registration.
        let runtime = ctx.runtime();
        let announced = Arc::clone(&self.tally);
        let mailbox = Arc::new(Mailbox::start_with_providers(
            config.manifest_dir.clone(),
            policy,
            signer,
            Box::new(move |session_id: &str| {
                finished
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .forget(session_id);
            }),
            Box::new(move |mark: &ManifestMark| announce(&announced, &runtime, mark)),
            config.provider_manifests.clone(),
        ));
        self.mailbox = Some(Arc::clone(&mailbox));

        let tally = Arc::clone(&self.tally);
        subscriber::register(
            ctx,
            SUBSCRIBER_NAME,
            Box::new(move |delivery| match delivery {
                Delivery::Event(event) => observe(&tally, &router, &mailbox, event),
                Delivery::Undecodable(raw) => note_undecodable(&tally, &router, &mailbox, raw),
                Delivery::Panicked(event) => note_panicked(&tally, &router, &mailbox, event),
            }),
        )?;

        Ok(())
    }
}

/// Tell the host where the manifest is, so it is discoverable without knowing our path convention.
///
/// Without it a consumer finds `.eqty/manifests/{session}.json` by convention, and a convention is
/// not an attestation.
///
/// The scope stack is **thread-local** and the final manifest is written from the worker thread,
/// which Relay never bound a stack to. So a stack is created for the call, putting the mark under a
/// root of its own: the mark's *data* names the session's manifest, its position in the scope tree
/// does not.
///
/// Failures are counted rather than raised -- this runs during teardown, a panic crossing the ABI
/// takes the gateway with it, and the manifest is on disk either way. `unannounced` is where a
/// reader sees it.
///
/// **Not asserted across the ABI.** `tests/regressions.rs` pins what is announced and when;
/// `abi-test` drives events through the subscriber but does not observe the mark.
fn announce(tally: &Tally, runtime: &PluginRuntime, mark: &ManifestMark) {
    let data = json!({
        "cid": mark.cid,
        "path": mark.path.to_string_lossy(),
        "statements": mark.statements,
        // Named in the data rather than left to the filename, because counting manifests to count
        // sessions is a mistake a reader makes silently.
        "unattributed": mark.unattributed,
    });
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if runtime.scope_stack_active() {
            return runtime.emit_mark(MANIFEST_MARK, Some(&data), None);
        }
        let stack = runtime.create_scope_stack()?;
        let _bound = runtime.bind_scope_stack_thread(&stack)?;
        runtime.emit_mark(MANIFEST_MARK, Some(&data), None)
    }));
    // A panic and an `Err` are the same outcome for a reader -- the manifest is on disk and nothing
    // on the stream points at it -- so they share a counter rather than pretending to differ.
    if !matches!(outcome, Ok(Ok(()))) {
        tally.unannounced.fetch_add(1, Ordering::Relaxed);
    }
}

/// Account for an event that could not be decoded, against its session when it names one.
///
/// The process-wide tally cannot tell a reader which manifest is incomplete; `EventsDropped` can.
fn note_undecodable(
    tally: &Tally,
    router: &Mutex<SessionRouter>,
    mailbox: &Mailbox,
    raw: Option<&Json>,
) {
    tally.undecodable.fetch_add(1, Ordering::Relaxed);
    let Some(raw) = raw else { return };
    // Attributed even when it is not charged: a scope must be remembered for its children.
    let session_id = router
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .attribute_raw(raw);
    if let Some(session_id) = session_id.filter(|_| may_carry_lineage(raw)) {
        mailbox.note_dropped(&session_id);
    }
}

/// Account for an event whose handling panicked, against its session when it decoded.
///
/// Must not call `classify`, which may be what panicked.
fn note_panicked(
    tally: &Tally,
    router: &Mutex<SessionRouter>,
    mailbox: &Mailbox,
    event: Option<&Event>,
) {
    tally.panicked.fetch_add(1, Ordering::Relaxed);
    let Some(event) = event.filter(|event| event_may_carry_lineage(event)) else {
        return;
    };
    let session_id = router
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .attribute(event);
    if let Some(session_id) = session_id {
        mailbox.note_dropped(&session_id);
    }
}

/// Handle one event. A panic here is contained and counted by the subscriber's trampoline.
fn observe(tally: &Tally, router: &Mutex<SessionRouter>, mailbox: &Mailbox, event: &Event) {
    tally.seen.fetch_add(1, Ordering::Relaxed);

    // Attributed before classifying: a scope that carries no lineage is still its children's
    // parent. A poisoned lock is recovered, since the map behind it has at worst missed one insert.
    let session_id = router
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .attribute(event);
    let Some(lineage) = classify(event) else {
        return;
    };
    let Some(session_id) = session_id else {
        tally.unattributed.fetch_add(1, Ordering::Relaxed);
        return;
    };
    tally.classified.fetch_add(1, Ordering::Relaxed);
    // Relay's own timestamp, never wall-clock at ingest: the two capture paths must not disagree
    // on when something happened just because one of them replayed it later.
    let at = event.timestamp().to_rfc3339();
    // Hand off and return. Everything expensive -- hashing, signing, writing -- happens on the
    // mailbox's own thread, so a slow recorder cannot become a slow agent.
    if !mailbox.send(&session_id, at, lineage) {
        tally.queue_overflow.fetch_add(1, Ordering::Relaxed);
    }
}

nemo_relay_plugin!(nemo_relay_register_plugin, EqtyLineagePlugin::new);
