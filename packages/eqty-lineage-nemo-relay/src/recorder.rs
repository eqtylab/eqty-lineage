//! Turning observed events into a lineage graph.
//!
//! Owns what the session has seen so far, and is the only place that decides what a file node
//! *means*. Ported from the Python recorder.
//!
//! # File identity is content. The path is metadata
//!
//! Keying on path would make `read → edit → read` either hide every edit or produce a cycle, since
//! the same node would be both an input and an output of one activity. So a node is its content,
//! addressed by the CID of the bytes, and the path travels in the metadata beside it.
//!
//! One file in two locations is therefore one node with two things said about it, and two files
//! holding identical bytes are also one node. `(path, content CID)` is still the in-session dedup
//! key, because a path is how a version chain is followed, but it never reaches the graph.
//!
//! # There are three ways not to know, and they must not collapse
//!
//! * a real content CID -- we hold the bytes,
//! * `unknown:{path}` -- we saw the path but never established its content,
//! * `deleted:{path}` -- the file is gone.
//!
//! Collapsing the last two would let a deletion deduplicate against a failed read of the same path,
//! and the graph would assert the file was removed when nobody saw it removed. Both are
//! path-derived: with no content there is nothing else to be identical about. Only nodes whose
//! content was established obey the content-identity rule above.
//!
//! # Identity is computed before redaction, never after
//!
//! The content CID comes from the original bytes. Hashing post-redaction content would collapse two
//! different secrets into one version, because both scrub to the same placeholder.

use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use integrity::cid::blake3::blake3_cid_raw_binary;
use integrity::lineage::models::manifest::Manifest;
use serde_json::{Value, json};

use crate::classify::CompactionPhase;
use crate::files::{FileMode, FileObserved, ReplayRefusal, apply_edit, apply_line_edit};
use crate::lineage::{AssetRef, LineageSession};
use crate::redaction::{Disposition, Policy};

/// How a version's content was established. Recorded so a reader can weigh it.
const BASIS_STATED: &str = "stated";
const BASIS_REPLAYED: &str = "replayed-from-session";

/// Records one session's lineage.
///
/// Single-owner by design and not concurrency-safe: one recorder belongs to one session, reached
/// through that session's mailbox. A shared lock on this would put contention on the path of an
/// event stream, which is what the enqueue-and-return contract exists to avoid.
pub struct Recorder {
    lineage: LineageSession,
    policy: Policy,

    /// The most recent content established for a path, for replaying an edit whose pre-image the
    /// agent did not report. This is the recovery chain that recovered 5,863 of 11,658 file versions
    /// on the corpus the Python recorder was measured against.
    last_content: HashMap<String, Vec<u8>>,
    /// `(path, content identity)` to the asset already registered for it.
    by_content: HashMap<(String, String), AssetRef>,
    /// How many versions of each path have been seen, for the `fileVersion` metadata.
    versions: HashMap<String, usize>,
    /// Counts a reader needs in order to discount the graph correctly.
    stats: BTreeMap<String, u64>,
    /// Actors already registered this session, keyed by `(kind, name)`.
    actors: HashMap<(String, String), AssetRef>,
    /// Events this session never saw because the queue was full when they arrived.
    ///
    /// Not a `stats` entry until export, because unlike every other counter this one is set from
    /// outside: the subscriber counts what it could not hand over, and the recorder is by definition
    /// unaware of it.
    events_dropped: u64,
    /// Bytes of content behind the nodes, split by what was done with them. Kept apart from
    /// `stats` because those are counts of occurrences and these are sizes, and a reader who meets
    /// `PayloadTooLarge: 70` in the same flat map has every reason to read it as a size.
    bytes: BTreeMap<String, u64>,
    /// How many compactions have happened, which is not how many compaction hooks have fired.
    compactions: u64,
    /// Whether the compaction currently counted is still waiting for its second half.
    awaiting_post_compaction: bool,
}

impl Recorder {
    /// Start recording into `lineage`, applying `policy` to file content.
    pub fn new(lineage: LineageSession, policy: Policy) -> Self {
        Self {
            lineage,
            policy,
            last_content: HashMap::new(),
            by_content: HashMap::new(),
            versions: HashMap::new(),
            stats: BTreeMap::new(),
            actors: HashMap::new(),
            events_dropped: 0,
            bytes: BTreeMap::new(),
            compactions: 0,
            awaiting_post_compaction: false,
        }
    }

    /// Counts of what was seen and what could not be established.
    pub fn stats(&self) -> &BTreeMap<String, u64> {
        &self.stats
    }

    /// Attribute `len` bytes to what the policy decided about them.
    ///
    /// Only called where content actually existed. A file whose content was never established has
    /// no length to attribute, and counting it as zero stored bytes would say we saw an empty file.
    fn note_bytes(&mut self, disposition: Disposition, len: usize) {
        // The field name, not a bucket name, because these are merged into the coverage node's
        // metadata as scalars rather than as a nested object. The graph explorer renders each
        // metadata value as a string, so an object arrives as `[object Object]` -- documented in
        // `eqty-lineage-langchain`'s README, which JSON-encodes for the same reason. It encodes
        // because it forwards nested context it does not control; this shape is three known keys,
        // so flat numbers beat an encoded string.
        //
        // Still prefixed rather than folded into the counter map: those are counts of occurrences
        // and these are sizes, and `PayloadTooLarge: 70` has been read as a size more than once.
        let key = match disposition {
            Disposition::Store => "bytesStored",
            Disposition::Denied => "bytesDenied",
            Disposition::TooLarge => "bytesTooLarge",
        };
        *self.bytes.entry(key.to_string()).or_insert(0) += len as u64;
    }

    fn count(&mut self, key: &str) {
        *self.stats.entry(key.to_string()).or_insert(0) += 1;
    }

    /// Register one file version, deduplicating on content.
    ///
    /// The same bytes at the same path are the same node however often they are seen; different
    /// bytes are a new version. Returns the asset, or `None` for an event carrying no path.
    pub async fn observe_file(
        &mut self,
        event: &FileObserved,
        observed: bool,
        at: Option<String>,
    ) -> Result<Option<AssetRef>> {
        if event.path.is_empty() {
            return Ok(None);
        }
        let path = event.path.clone();

        let mut data = event.content.clone();
        let mut basis = data.as_ref().map(|_| BASIS_STATED);

        // The post-image was not stated. If the replacement is known and this session already
        // established content for the path, replay it. The check that `old` occurs in what we hold
        // is what stops a stale pre-image minting a version the file never had.
        if data.is_none()
            && let Some(edit) = &event.edit
            // A patch that moves a file applies its hunk to the *source*, so the base is the source's
            // content and the result belongs at the destination this event names.
            && let Some(previous) = self
                .last_content
                .get(edit.replay_from.as_deref().unwrap_or(path.as_str()))
        {
            let previous = String::from_utf8_lossy(previous).into_owned();
            if edit.line_oriented {
                // A patch hunk. Matched as runs of whole lines, which is the unit the patch speaks
                // in -- and which is what makes its uniqueness check mean anything. Matched as a
                // substring, a line terminator decides uniqueness instead, and `b\n` occurring once
                // in `b\nb` where `b` occurs twice let the replacement land at the wrong end.
                match apply_line_edit(&previous, &edit.old, &edit.new) {
                    Ok(replayed) => {
                        data = Some(replayed.into_bytes());
                        basis = Some(BASIS_REPLAYED);
                        self.count("ContentRecovered");
                    }
                    Err(ReplayRefusal::Ambiguous) => self.count("EditTooAmbiguousToReplay"),
                    Err(ReplayRefusal::NotFound) => self.count("EditDidNotMatchHeldContent"),
                    Err(ReplayRefusal::UnterminatedAtEof) => self.count("EditAtUnterminatedEof"),
                    Err(ReplayRefusal::TerminatorsNotEstablished) => {
                        self.count("EditTerminatorsNotEstablished")
                    }
                }
            } else if edit.unique_only && previous.matches(&edit.old).count() != 1 {
                // A literal edit that cannot promise its own uniqueness must find exactly one match,
                // or the replay is a guess about which occurrence the tool meant.
                self.count("EditTooAmbiguousToReplay");
            } else if let Some(replayed) = apply_edit(
                Some(&previous),
                Some(&edit.old),
                Some(&edit.new),
                edit.replace_all,
            ) {
                data = Some(replayed.into_bytes());
                basis = Some(BASIS_REPLAYED);
                self.count("ContentRecovered");
            }
        }

        // Identity, from the original bytes, before any redaction decision.
        let content_cid = match &data {
            Some(bytes) => blake3_cid_raw_binary(bytes)?,
            None => format!("unknown:{path}"),
        };

        let key = (path.clone(), content_cid.clone());
        if let Some(existing) = self.by_content.get(&key).cloned() {
            // The node already exists, but the replay base must still move -- by exactly the rule
            // the new-node path uses below, because the base is a fact about the file and not about
            // whether we happened to have seen this version before.
            //
            // A file that went from A to B and back to A leaves B cached otherwise, and the next
            // edit anchored on A is either refused or -- if its `old` text happens to occur in B as
            // well -- replayed against content the file no longer holds.
            //
            // The contentless-write arm is the one that was missing, and it is reachable: an
            // unreconstructable write keys on `unknown:{path}`, so the *second* such observation for
            // a path deduplicates against the first and returned here without clearing the base. A
            // truncated read seeds that key without touching the base, so `read (truncated)`,
            // `read (full)`, `write (unreconstructable)`, `edit` replayed the edit against the
            // pre-write bytes and signed a version of the file that never existed.
            Self::move_replay_base(&mut self.last_content, path, data, event.mode);
            return Ok(Some(existing));
        }

        let version = self.versions.entry(path.clone()).or_insert(0);
        *version += 1;
        let version = *version;

        let disposition = match &data {
            Some(bytes) => self.policy.decide(&path, bytes.len()),
            // Nothing to withhold, and nothing to store.
            None => Disposition::Denied,
        };
        let withheld = disposition != Disposition::Store;

        // "We refused to store this" and "we never had it" are different claims, and a node that
        // reports both as `redacted` tells a reader the wrong one. A withheld file was seen and its
        // bytes deliberately kept out; an unknown one was never established at all, so there was
        // nothing to withhold. Codex makes the difference constant rather than occasional: every
        // `apply_patch` `Update File` carries hunks and no post-image, so every one of them lands
        // here as unknown.
        let content_state = match (&data, withheld) {
            (Some(_), false) => "stored",
            (Some(_), true) => "withheld",
            (None, _) => "unknown",
        };
        if let Some(bytes) = &data {
            self.note_bytes(disposition, bytes.len());
        }

        let metadata = json!({
            "name": path,
            "assetType": "Document",
            "description": format!("File '{path}' observed by the agent."),
            "provType": "Entity",
            "filePath": path,
            "fileVersion": version,
            "observed": observed,
            "userModified": event.user_modified,
            "contentState": content_state,
            // Policy withholding only. Absent content is `contentState`, not redaction.
            "redacted": data.is_some() && withheld,
            "reconstructed": basis,
            "content-cid": content_cid,
            // Null when content was never established, which is the one case where there is no
            // length to state rather than a length we chose not to store.
            "contentBytes": data.as_ref().map(Vec::len),
        });

        let asset = match (&data, withheld) {
            (Some(bytes), false) => self.lineage.register_content(bytes, metadata, at).await?,
            // Identity-only. The descriptor is canonical and derived from the true content CID
            // alone, so the node is deterministic across runs and independent of where the file sat
            // -- an entity would mint a fresh UUID each time and two recordings of the same withheld
            // file would not join.
            _ => {
                if withheld && data.is_some() {
                    self.count(match disposition {
                        Disposition::Denied => "ContentDenied",
                        _ => "ContentTooLarge",
                    });
                } else {
                    self.count("ContentUnknown");
                }
                let descriptor = canonical_descriptor(&content_cid, withheld)?;
                self.lineage
                    .register_content(&descriptor, metadata, at)
                    .await?
            }
        };

        Self::move_replay_base(&mut self.last_content, path.clone(), data, event.mode);
        self.by_content.insert(key, asset.clone());
        self.count(match event.mode {
            FileMode::Read => "FileRead",
            FileMode::Wrote => "FileWritten",
        });
        Ok(Some(asset))
    }

    /// What a later edit replays against, after observing `path`.
    ///
    /// One function because both call sites must agree: the deduplication path skipped the
    /// contentless-write arm for as long as it was written out twice.
    fn move_replay_base(
        last_content: &mut HashMap<String, Vec<u8>>,
        path: String,
        data: Option<Vec<u8>>,
        mode: FileMode,
    ) {
        match (data, mode) {
            (Some(bytes), _) => {
                last_content.insert(path, bytes);
            }
            // A write we could not reconstruct means what is on disk is no longer what we hold.
            // Keeping the old bytes lets a later edit "recover" a version built from content that
            // write replaced -- a fabricated file version, content-addressed and signed. Every Codex
            // `Update File` whose hunk does not replay lands here, so this is the common path.
            (None, FileMode::Wrote) => {
                last_content.remove(&path);
            }
            // A read we could not establish -- a truncated `Read` -- changed nothing on disk, so
            // what we already hold is still the file's content and still a valid base.
            (None, FileMode::Read) => {}
        }
    }

    /// Register a payload that is not a file: a prompt, a completion, a set of instructions.
    ///
    /// Same identity rule as a file -- the CID is over the bytes as given -- and the same content
    /// ceiling, because a long conversation is exactly the thing that blows past it. Content over
    /// the ceiling becomes a deterministic descriptor rather than being dropped, so the graph still
    /// names what was sent even when it does not carry it.
    pub async fn register_payload(
        &mut self,
        kind: &str,
        name: &str,
        description: &str,
        bytes: &[u8],
        extra: Value,
        at: Option<String>,
    ) -> Result<AssetRef> {
        self.register(kind, name, description, bytes, extra, at, &[])
            .await
    }

    /// Register a payload that may carry the contents of files, and inherit their policy.
    ///
    /// A tool's result *is* the file, for a `Read`; its arguments *are* the file, for a `Write`. So
    /// withholding `/app/.env` on its own node while storing the call that produced it left the
    /// secret in the manifest in full -- the file node said `withheld` and the graph beside it held
    /// the bytes. The deny list is written for paths, and a payload named `Read result` is not one,
    /// so nothing matched.
    ///
    /// `quotes` names the paths this payload may reproduce. If policy denies any of them, the whole
    /// payload is withheld: a partial redaction of a JSON blob is a guess about where the bytes are,
    /// and the wrong guess is a leak that looks like a redaction.
    ///
    /// **This does not reach content a tool never attributed to a file.** `cat /app/.env` through
    /// `Bash` names no path we can see, so its output is stored -- the same shell-visibility gap as
    /// §7.1's file effects, and the reason `deny_globs` is a floor rather than a guarantee.
    #[allow(clippy::too_many_arguments)]
    pub async fn register_quoting_payload(
        &mut self,
        kind: &str,
        name: &str,
        description: &str,
        bytes: &[u8],
        extra: Value,
        at: Option<String>,
        quotes: &[String],
    ) -> Result<AssetRef> {
        self.register(kind, name, description, bytes, extra, at, quotes)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn register(
        &mut self,
        kind: &str,
        name: &str,
        description: &str,
        bytes: &[u8],
        extra: Value,
        at: Option<String>,
        quotes: &[String],
    ) -> Result<AssetRef> {
        let content_cid = blake3_cid_raw_binary(bytes)?;
        // The deny list is matched against the payload's *name* as well as any path it quotes. A
        // payload is named after what produced it (`prompt`, `completion`, `Bash input`), so a glob
        // like `*credentials*` withholds the arguments of a tool called `get_credentials`. That is
        // the intended reach, and it is why the reason travels with the node: "denied", "quotes a
        // denied file" and "too large" are different claims and a reader acts on them differently.
        let mut disposition = self.policy.decide(name, bytes.len());
        let mut quoted: Option<&str> = None;
        if disposition == Disposition::Store
            && let Some(path) = quotes.iter().find(|path| self.policy.denies(path))
        {
            disposition = Disposition::Denied;
            quoted = Some(path.as_str());
        }
        let withheld = disposition != Disposition::Store;
        self.note_bytes(disposition, bytes.len());
        let reason = match (disposition, quoted) {
            (Disposition::Store, _) => None,
            (Disposition::Denied, Some(_)) => Some("quotes-a-denied-file"),
            (Disposition::Denied, None) => Some("denied-by-policy"),
            (Disposition::TooLarge, _) => Some("larger-than-ceiling"),
        };

        let mut metadata = json!({
            "name": name,
            "assetType": kind,
            "description": description,
            "provType": "Entity",
            "redacted": withheld,
            "contentState": if withheld { "withheld" } else { "stored" },
            "withheldBecause": reason,
            "withheldFor": quoted,
            "content-cid": content_cid,
            // Stated whether or not the bytes were kept: without it a `larger-than-ceiling` node
            // cannot tell a reader whether raising the ceiling recovers the content or buries the
            // manifest.
            "contentBytes": bytes.len(),
        });
        if let (Some(target), Some(extra)) = (metadata.as_object_mut(), extra.as_object()) {
            for (key, value) in extra {
                target.insert(key.clone(), value.clone());
            }
        }

        if withheld {
            self.count(match disposition {
                Disposition::Denied => "PayloadDenied",
                _ => "PayloadTooLarge",
            });
            let descriptor = canonical_descriptor(&content_cid, true)?;
            return self
                .lineage
                .register_content(&descriptor, metadata, at)
                .await;
        }
        self.lineage.register_content(bytes, metadata, at).await
    }

    /// Record one completed model call.
    ///
    /// The computation's inputs are everything that determined the answer -- the model itself, the
    /// system prompt, and the conversation -- and its output is the response. Naming the model as an
    /// input rather than as a label on the edge is what makes "which runs used this model" a graph
    /// question instead of a string search.
    ///
    /// A call whose response never arrived is counted and not recorded: an activity with no output
    /// is a node no reader can reach, and a prompt with no completion attests nothing about what the
    /// model did.
    #[allow(clippy::too_many_arguments)]
    pub async fn record_model_call(
        &mut self,
        model: Option<&str>,
        instructions: Option<&[u8]>,
        prompt: &[u8],
        completion: Option<&[u8]>,
        caused_by: Option<AssetRef>,
        details: Value,
        describes: Value,
        observed: bool,
        at: Option<String>,
    ) -> Result<bool> {
        let mut inputs = Vec::new();

        // The user instruction this call answers, when this is the first call of the turn.
        if let Some(caused_by) = caused_by {
            inputs.push(caused_by);
        }

        if let Some(model) = model {
            inputs.push(
                self.record_actor(
                    "Model",
                    model,
                    &format!("The model '{model}' that served this call."),
                    json!({}),
                    at.clone(),
                )
                .await?,
            );
        }

        // The system prompt is a separate node from the conversation because it changes on a
        // different cadence: one system prompt governs many turns, and keeping them apart lets a
        // reader see that a run's instructions were unchanged while its messages were not.
        if let Some(instructions) = instructions {
            inputs.push(
                self.register_payload(
                    "System_Prompt",
                    "system prompt",
                    "Provider-level instructions sent alongside the conversation.",
                    instructions,
                    json!({ "model": model, "observed": observed }),
                    at.clone(),
                )
                .await?,
            );
        }

        inputs.push(
            self.register_payload(
                "Prompt",
                "prompt",
                "The normalized conversation sent to the model.",
                prompt,
                json!({ "model": model, "observed": observed }),
                at.clone(),
            )
            .await?,
        );

        let Some(completion) = completion else {
            self.count("ModelCallWithoutResponse");
            // The inputs above are already registered and nothing will link them, because an
            // activity needs an output and there is none. Left alone they are nodes a reader finds
            // dangling with no reason on them, and the only explanation is a session-level counter.
            //
            // The reason goes on a marker for the call rather than as a field on those nodes,
            // because the nodes are content addressed and shared. One live session had a single
            // `Model` node feeding 31 successful calls and this one failure: writing "unlinked" onto
            // it would have been false, and at registration time there is no way to know whether a
            // node will later be reused by a call that did answer. What failed is the call, so the
            // claim belongs on the call.
            let index = self
                .stats
                .get("ModelCallWithoutResponse")
                .copied()
                .unwrap_or(1);
            self.register_payload(
                "Dataset",
                &format!("unanswered model call {index}"),
                "A model call whose response never arrived. Its inputs were registered and no \
                 computation links them, because an activity with no output is not one.",
                // The ordinal is in the content for the reason it is in a compaction's: two failures
                // with identical inputs would otherwise collapse into one node.
                &serde_json::to_vec(&json!({ "unansweredCall": index }))?,
                json!({
                    "unlinkedBecause": "model-call-had-no-response",
                    "unlinkedInputs": inputs.iter().map(AssetRef::as_str).collect::<Vec<_>>(),
                }),
                at,
            )
            .await?;
            return Ok(false);
        };

        let output = self
            .register_payload(
                "Reasoning",
                "completion",
                "The model's response, including any tool calls it requested.",
                completion,
                details,
                at.clone(),
            )
            .await?;

        self.lineage
            .record_computation_described(&inputs, &[output], describes, at)
            .await?;
        self.count("ModelCall");
        Ok(true)
    }

    /// Record that the agent compacted its context.
    ///
    /// A node rather than a log line, because compaction changes what the agent could possibly have
    /// known: everything before it has left the model's window. A reader tracing why a later step
    /// ignored an earlier one needs to see where the boundary was.
    /// The ordinal counts compactions, not hook events, and the name says which half of one this is:
    /// Claude Code fires `PreCompact` and `PostCompact` around a *single* compaction. Two nodes is
    /// right -- the boundary has two edges, each a real observation -- but they are two halves of
    /// one, not two compactions.
    ///
    /// The ordinal stays in the content as well as the name because these nodes are content
    /// addressed: two compactions whose descriptors matched would collapse into one node, and the
    /// graph would under-report.
    ///
    /// An `After` with no `Before` ahead of it still opens a new ordinal. A host that emits only the
    /// second hook, or an auto-compaction that skips the first, is better recorded as the half we saw
    /// than not recorded at all.
    pub async fn record_compaction(
        &mut self,
        phase: CompactionPhase,
        at: Option<String>,
    ) -> Result<AssetRef> {
        let fresh = match phase {
            CompactionPhase::Before => true,
            CompactionPhase::After => !self.awaiting_post_compaction,
        };
        if fresh {
            self.compactions += 1;
            self.count("Compaction");
        }
        self.awaiting_post_compaction = matches!(phase, CompactionPhase::Before);

        let index = self.compactions;
        let (label, description) = match phase {
            CompactionPhase::Before => (
                "pre-compaction",
                "The last moment before a context compaction: everything so far is still in the \
                 model's window.",
            ),
            CompactionPhase::After => (
                "post-compaction",
                "The first moment after a context compaction: everything before it has left the \
                 model's window.",
            ),
        };
        let name = format!("{label} {index}");
        // An `Entity`, not an `Activity`. An activity in this manifest *is* a
        // `ComputationRegistration`, carrying inputs, outputs and `performedBy`; this is a
        // `DataRegistration`, so claiming `provType: Activity` would describe graph structure no
        // statement backs. What is captured is a marker -- an ordinal, a phase, a timestamp -- and
        // nothing about the compaction was observed beyond its having happened, which is why
        // `record_tool_run` refuses an empty computation for the same reason
        // (`ActivityWithoutOutputs`).
        //
        // Phase 4's context snapshots (§8) would earn the other type: pre- and post-compaction
        // contexts as entities, and one real computation between them.
        self.register_payload(
            "Dataset",
            &name,
            description,
            &serde_json::to_vec(&json!({ "compaction": index, "phase": label }))?,
            json!({}),
            at,
        )
        .await
    }

    /// Record a tool run that consumed and produced the given files.
    ///
    /// A run with no outputs is not recorded. It cannot be reached from any asset, so it is a node
    /// no reader can use, and the inputs it names are already attested by their own registrations.
    /// Note a tool call that yielded no file observation at all.
    ///
    /// Not the same as "touched no files": `ls /nowhere` legitimately touches none, and a shell
    /// command that rewrites a tree touches many. The counter says only what it can -- that this
    /// activity told us nothing about files -- which is what lets a reader tell a session that read
    /// nothing from one whose reads were invisible. On Codex that is every read, since it has no
    /// read tool.
    pub fn note_no_file_observation(&mut self) {
        self.count("ToolCallWithoutFileObservation");
    }

    /// Note whether a tool call reported failure, including when it reported nothing.
    ///
    /// Three outcomes, and the third is not a rounding of the second. A host that says nothing about
    /// success has not said the call succeeded, and recording it as ordinary work would attest a
    /// clean run over a failed one. Codex reports no terminal status at all, so there every call
    /// lands in the unknown bucket -- which is the honest reading of what that capture path can see.
    pub fn note_tool_outcome(&mut self, is_error: Option<bool>) {
        self.count(match is_error {
            Some(true) => "ToolCallFailed",
            Some(false) => "ToolCallSucceeded",
            None => "ToolCallOutcomeUnknown",
        });
    }

    /// Note that a failed call's file effects were not inferred from what it asked for.
    ///
    /// The alternative was worse than a gap: a rejected patch registering its requested bytes as a
    /// written file, and seeding the replay chain so a later edit could be "recovered" from content
    /// that never reached disk. Counted so the gap is visible, since a failed call that touched
    /// nothing and a failed call whose effects we declined to guess at look identical otherwise.
    pub fn note_inference_skipped_after_failure(&mut self) {
        self.count("FileInferenceSkippedAfterFailure");
    }

    /// Note a tool call whose end carried no result payload at all.
    ///
    /// Distinct from a call that returned nothing about files: this one returned nothing, period, so
    /// the run is recorded from its arguments alone. Counted so a reader can tell a quiet tool from a
    /// capture path that lost the reply.
    pub fn note_tool_without_result(&mut self) {
        self.count("ToolCallWithoutResult");
    }

    /// Note work done while more than one subagent was live.
    ///
    /// Relay reports that *a* subagent is running, not which one performed a given call, so with
    /// siblings in flight the performer cannot be established. The activity is credited to the root
    /// agent and marked ambiguous rather than assigned to whichever sibling started last -- a wrong
    /// specific attribution is worse than an honest general one, because a reader can act on it.
    /// Count a turn nobody typed, by the kind of thing that opened it.
    ///
    /// In coverage rather than left to the nodes, because the nodes are what went unread: every
    /// harness counted prompts and none opened one, so two live sessions attested instructions
    /// their user never sent and nothing flagged it.
    pub fn note_turn_author(&mut self, counter: &'static str) {
        self.count(counter);
    }

    pub fn note_ambiguous_attribution(&mut self) {
        self.count("AmbiguousSubagentAttribution");
    }

    /// Record how many events never reached this recorder because the queue was full.
    ///
    /// Set from the mailbox at export. The recorder cannot observe its own gaps, and a graph with
    /// holes that does not say so reads as a complete one.
    pub fn note_events_dropped(&mut self, count: u64) {
        self.events_dropped = count;
    }

    /// Total bytes of blob content held, for deciding how expensive a snapshot would be.
    pub fn blob_bytes(&self) -> usize {
        self.lineage.blob_bytes()
    }

    pub async fn record_tool_run(
        &mut self,
        inputs: &[AssetRef],
        outputs: &[AssetRef],
        describes: Value,
        at: Option<String>,
    ) -> Result<bool> {
        if outputs.is_empty() {
            self.count("ActivityWithoutOutputs");
            return Ok(false);
        }
        self.lineage
            .record_computation_described(inputs, outputs, describes, at)
            .await?;
        self.count("Activity");
        Ok(true)
    }

    /// Register an actor -- an agent, a subagent, a model, a tool -- deduplicated by identity.
    ///
    /// Content-addressed on a canonical descriptor of *what it is* rather than minted fresh, so the
    /// same model or the same tool is one node across every session that used it. That is what lets
    /// "which runs used this tool" be a graph question rather than a string search across manifests.
    ///
    /// Registered once per session: a tool called forty times is one node with forty edges, not
    /// forty nodes.
    ///
    /// The descriptor goes through the serializer, never `format!`. `name` is host-supplied, and
    /// interpolating it raw makes the node's own content forgeable: a tool named `a","kind":"Model`
    /// yields `{"kind":"Tool","name":"a","kind":"Model"}`, which parses as *`kind: Model`* because a
    /// duplicate key takes the last value. Identity is not at risk -- `kind` is our own literal and
    /// precedes `name` -- but the claim inside the node is.
    pub async fn record_actor(
        &mut self,
        kind: &str,
        name: &str,
        description: &str,
        extra: Value,
        at: Option<String>,
    ) -> Result<AssetRef> {
        let key = (kind.to_string(), name.to_string());
        if let Some(existing) = self.actors.get(&key) {
            return Ok(existing.clone());
        }

        let descriptor = serde_json::to_vec(&json!({ "kind": kind, "name": name }))?;
        let asset = self
            .register_payload(kind, name, description, &descriptor, extra, at)
            .await?;
        self.actors.insert(key, asset.clone());
        self.count(kind);
        Ok(asset)
    }

    /// Emit what this recording did not see, then build the manifest.
    ///
    /// Coverage goes inside the graph, signed, rather than into a log beside it. A reader who cannot
    /// see the gaps cannot weigh the evidence: "no failed tool calls" and "this capture path cannot
    /// observe tool failure" look identical from outside, and on Codex it is always the second.
    /// A manifest of everything recorded so far, without the coverage node and without consuming
    /// the recorder.
    ///
    /// The missing coverage node is the point, not an omission: coverage states what the recording
    /// could not see, and that is only knowable once it has stopped. A manifest carrying one is
    /// complete; a manifest without one was written while the session was still running.
    pub async fn snapshot(&self) -> Result<Manifest> {
        self.lineage.snapshot().await
    }

    /// How many statements exist, so a caller can skip a snapshot that would say nothing new.
    pub fn statement_count(&self) -> usize {
        self.lineage.statement_count()
    }

    pub async fn finish(mut self, at: Option<String>) -> Result<Manifest> {
        // Stated even when it is zero, unlike every other counter. For the rest, absent means none
        // happened and a reader loses nothing by inferring it. This one answers "is this graph
        // complete?", and a reader who has to know that absence means zero cannot distinguish an
        // intact recording from a manifest written before the counter existed.
        self.stats
            .insert("EventsDropped".to_string(), self.events_dropped);

        let coverage: Value = self
            .stats
            .iter()
            .map(|(key, count)| (key.clone(), json!(count)))
            .collect::<serde_json::Map<_, _>>()
            .into();

        // Content-addressed like everything else. Two runs that saw the same things produce the
        // same coverage node, which is what lets a reader compare what two sessions could observe
        // rather than only what they did.
        let body = serde_json::to_vec(&coverage).unwrap_or_default();

        // The totals ride in the metadata rather than the body on purpose. The body is what this
        // node is addressed by, and the comment above is the reason: two sessions that saw the same
        // things should produce the same coverage node. Byte volume is not an observability fact --
        // the same session recorded twice against a different ceiling sees exactly as much and
        // stores a different amount -- so folding it into the content would make comparable runs
        // stop matching. The metadata statement is signed either way.
        //
        // A bucket with no bytes is absent rather than zero, which is the same distinction
        // `contentState` keeps: "we withheld nothing" is not "there was nothing to withhold".
        let mut extra = json!({ "provType": "Entity", "coverage": coverage });
        let totals: Vec<(String, u64)> = self.bytes.iter().map(|(k, v)| (k.clone(), *v)).collect();
        if let Some(target) = extra.as_object_mut() {
            for (key, total) in totals {
                target.insert(key, json!(total));
            }
        }

        self.register_payload(
            "Dataset",
            "coverage",
            "What this recording saw, and what it could not: counts a reader needs to weigh the graph.",
            &body,
            extra,
            at,
        )
        .await?;

        self.lineage.into_manifest().await
    }
}

/// A deterministic stand-in for content we may not store.
///
/// Canonical by construction: `serde_json`'s map is sorted and its output carries no whitespace, so
/// the same withheld file in two recordings hashes to the same identity and the graphs join. This is
/// a JSON envelope rather than a content hash, which is exactly why the node's metadata marks it
/// `redacted` -- a reader must be able to tell which nodes are content-addressed and which are not.
///
/// Built through the serializer rather than `format!`: `content_cid` is not always a hash -- for an
/// unestablished file it is `unknown:{path}` -- and a path is arbitrary bytes from the host. A path
/// shaped like `x","withheld":true}` would otherwise make the node's own `withheld` claim say the
/// opposite of the decision taken.
///
/// **The path is deliberately not in here.** Identity is content, so hashing the path in would split
/// the withheld nodes -- exactly the secrets and large artifacts where knowing two recordings saw
/// the same bytes is worth the most. When content was never established the caller passes
/// `unknown:{path}`, so the path does determine identity there; that is unavoidable rather than
/// intended.
fn canonical_descriptor(content_cid: &str, withheld: bool) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&json!({
        "content-cid": content_cid,
        "withheld": withheld,
    }))?)
}
