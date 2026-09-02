//! Turning observed events into a lineage graph.
//!
//! This is the state machine the plan calls the port: it owns what the session has seen so far, and
//! it is the only place that decides what a file node *means*. Ported faithfully from the existing
//! Python recorder rather than redesigned -- the semantics below each exist for a measured reason,
//! and several of them look like over-thinking until the case that motivated them shows up.
//!
//! # File identity is `(path, content CID)`, never path alone
//!
//! Keying on path would make `read → edit → read` either hide every edit or produce a cycle, since
//! the same node would be both an input and an output of the same activity. Keying on content alone
//! would merge two different files that happen to hold the same bytes. The pair is what makes a
//! version.
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
            if let Some(replayed) = apply_edit(
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

        let metadata = json!({
            "name": path,
            "assetType": "Document",
            "provType": "Entity",
            "filePath": path,
            "fileVersion": version,
            "observed": observed,
            "userModified": event.user_modified,
            "redacted": withheld,
            "reconstructed": basis,
            "content-cid": content_cid,
        });

        let asset = match (&data, withheld) {
            (Some(bytes), false) => self.lineage.register_content(bytes, metadata, at).await?,
            // Identity-only. The descriptor is canonical and derived from the path and the true
            // content CID, so the node is deterministic across runs -- an entity would mint a fresh
            // UUID each time and two recordings of the same withheld file would not join.
            _ => {
                if withheld && data.is_some() {
                    self.count(match disposition {
                        Disposition::Denied => "ContentDenied",
                        _ => "ContentTooLarge",
                    });
                } else {
                    self.count("ContentUnknown");
                }
                let descriptor = canonical_descriptor(&path, &content_cid, withheld);
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

    /// Record a tool run that consumed and produced the given files.
    ///
    /// A run with no outputs is not recorded. It cannot be reached from any asset, so it is a node
    /// no reader can use, and the inputs it names are already attested by their own registrations.
    pub async fn record_tool_run(
        &mut self,
        inputs: &[AssetRef],
        outputs: &[AssetRef],
        at: Option<String>,
    ) -> Result<bool> {
        if outputs.is_empty() {
            self.count("ActivityWithoutOutputs");
            return Ok(false);
        }
        self.lineage.record_computation(inputs, outputs, at).await?;
        self.count("Activity");
        Ok(true)
    }

    /// Register the agent that ran the session.
    pub async fn record_agent(
        &mut self,
        agent: Option<&str>,
        model: Option<&str>,
        at: Option<String>,
    ) -> Result<AssetRef> {
        let asset = self
            .lineage
            .register_entity(
                json!({
                    "name": agent.unwrap_or("unknown-agent"),
                    "assetType": "Agent",
                    "provType": "Agent",
                    "model": model,
                }),
                at,
            )
            .await?;
        self.count("Agent");
        Ok(asset)
    }

    /// Emit what this recording did not see, then build the manifest.
    ///
    /// Coverage goes inside the graph, signed, rather than into a log beside it. A reader who cannot
    /// see the gaps cannot weigh the evidence: "no failed tool calls" and "this capture path cannot
    /// observe tool failure" look identical from outside, and on Codex it is always the second.
    pub async fn finish(mut self, at: Option<String>) -> Result<Manifest> {
        let coverage: Value = self
            .stats
            .iter()
            .map(|(key, count)| (key.clone(), json!(count)))
            .collect::<serde_json::Map<_, _>>()
            .into();

        self.lineage
            .register_entity(
                json!({
                    "name": "coverage",
                    "assetType": "Configuration",
                    "provType": "Entity",
                    "coverage": coverage,
                }),
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
fn canonical_descriptor(path: &str, content_cid: &str, withheld: bool) -> Vec<u8> {
    format!(r#"{{"content-cid":"{content_cid}","path":"{path}","withheld":{withheld}}}"#)
        .into_bytes()
}
