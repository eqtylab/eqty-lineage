//! Turning observed events into a lineage graph.
//!
//! This is the state machine the plan calls the port: it owns what the session has seen so far, and
//! it is the only place that decides what a file node *means*. Ported faithfully from the existing
//! Python recorder rather than redesigned -- the semantics below each exist for a measured reason,
//! and several of them look like over-thinking until the case that motivated them shows up.
//!
//! # File identity is content. The path is metadata
//!
//! Keying on path would make `read → edit → read` either hide every edit or produce a cycle, since
//! the same node would be both an input and an output of the same activity. So a node is its
//! content, addressed by the CID of the bytes, and the path travels in the metadata beside it.
//!
//! The consequence is deliberate: one file copied or moved to a second location is **one node with
//! two things said about it**, not two nodes. Two files that happen to hold identical bytes are also
//! one node -- which is the same statement, since under content addressing they are the same thing.
//! `(path, content CID)` is still the in-session dedup key, because a path is how a *version chain*
//! is followed, but it never reaches the graph as identity.
//!
//! # There are three ways not to know, and they must not collapse
//!
//! * a real content CID -- we hold the bytes,
//! * `unknown:{path}` -- we saw the path but never established its content,
//! * `deleted:{path}` -- the file is gone.
//!
//! Collapsing the last two would let a deletion deduplicate against a failed read of the same path,
//! and the graph would then assert the file was removed when nobody ever saw it removed.
//!
//! The last two are path-derived, and unavoidably so: with no content there is nothing else to be
//! identical about, and two unread files cannot be shown to be the same file. Only nodes whose
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

use crate::files::{FileMode, FileObserved, apply_edit};
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
        }
    }

    /// Counts of what was seen and what could not be established.
    pub fn stats(&self) -> &BTreeMap<String, u64> {
        &self.stats
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
            && let Some(previous) = self.last_content.get(&path)
        {
            let previous = String::from_utf8_lossy(previous).into_owned();
            // An edit that cannot promise its own uniqueness must find exactly one match, or the
            // replay is a guess about which occurrence the tool meant.
            if edit.unique_only && previous.matches(&edit.old).count() != 1 {
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
        if let Some(existing) = self.by_content.get(&key) {
            return Ok(Some(existing.clone()));
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
                let descriptor = canonical_descriptor(&content_cid, withheld);
                self.lineage
                    .register_content(&descriptor, metadata, at)
                    .await?
            }
        };

        if let Some(bytes) = data {
            self.last_content.insert(path.clone(), bytes);
        }
        self.by_content.insert(key, asset.clone());
        self.count(match event.mode {
            FileMode::Read => "FileRead",
            FileMode::Wrote => "FileWritten",
        });
        Ok(Some(asset))
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
        let content_cid = blake3_cid_raw_binary(bytes)?;
        let withheld = self.policy.decide(name, bytes.len()) != Disposition::Store;

        let mut metadata = json!({
            "name": name,
            "assetType": kind,
            "description": description,
            "provType": "Entity",
            "redacted": withheld,
            "content-cid": content_cid,
        });
        if let (Some(target), Some(extra)) = (metadata.as_object_mut(), extra.as_object()) {
            for (key, value) in extra {
                target.insert(key.clone(), value.clone());
            }
        }

        if withheld {
            self.count("PayloadTooLarge");
            let descriptor = canonical_descriptor(&content_cid, true);
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
    pub async fn record_compaction(&mut self, at: Option<String>) -> Result<AssetRef> {
        let index = self.stats.get("Compaction").copied().unwrap_or(0) + 1;
        let name = format!("compaction {index}");
        let asset = self
            .register_payload(
                "Dataset",
                &name,
                "A context compaction: everything before this point left the model's window.",
                format!(r#"{{"compaction":{index}}}"#).as_bytes(),
                json!({ "provType": "Activity" }),
                at,
            )
            .await?;
        self.count("Compaction");
        Ok(asset)
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

        let descriptor = format!(r#"{{"kind":"{kind}","name":"{name}"}}"#).into_bytes();
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
        self.register_payload(
            "Dataset",
            "coverage",
            "What this recording saw, and what it could not: counts a reader needs to weigh the graph.",
            &body,
            json!({ "provType": "Entity", "coverage": coverage }),
            at,
        )
        .await?;

        self.lineage.into_manifest().await
    }
}

/// A deterministic stand-in for content we may not store.
///
/// Canonical by construction: the fields are emitted in a fixed order with no whitespace, so the
/// same withheld file in two recordings hashes to the same identity and the graphs join. This is a
/// JSON envelope rather than a content hash, which is exactly why the node's metadata marks it
/// `redacted` -- a reader must be able to tell which nodes are content-addressed and which are not.
///
/// **The path is deliberately not in here.** Identity is content, and the path is metadata, so one
/// file copied or moved to a second location is one node with two things said about it. Hashing the
/// path in would split the withheld nodes -- exactly the files, secrets and large artifacts, where
/// knowing two recordings saw the same bytes is worth the most.
///
/// When content was never established the caller passes `unknown:{path}` as `content_cid`, so the
/// path does still determine identity there. That is unavoidable rather than intended: with no
/// content there is nothing else to be identical about, and two unread files cannot be shown to be
/// the same file.
fn canonical_descriptor(content_cid: &str, withheld: bool) -> Vec<u8> {
    format!(r#"{{"content-cid":"{content_cid}","withheld":{withheld}}}"#).into_bytes()
}
