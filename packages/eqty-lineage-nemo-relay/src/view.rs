//! The session view: what happened, minus the model's own conversation.
//!
//! A full recording is mostly monologue. Measured on real sessions, 68% of the asset nodes are
//! `prompt`, `completion` and `system prompt`, and 5% are the files -- so the substantive claim is
//! buried under the reasoning that produced it. This writes a second manifest holding everything
//! *except* that conversation.
//!
//! **It is a subset of the signed manifest, never a re-signing.** Every kept statement keeps its own
//! CID and its own `CredentialRegistration`, so the view verifies exactly as the full manifest does.
//! Re-recording the selection into a fresh context would mint new statement CIDs -- they carry
//! `validFrom` -- and produce a *different* attestation that merely resembled the original.
//!
//! Defined by what it excludes rather than by what it requires, which is the difference between this
//! and a file-lineage view. A selector keyed on "has a path" collapses to zero statements on a
//! session that only talked, and a zero-statement manifest still parses and still verifies: a
//! document that looks like an attestation and asserts nothing. Excluding the monologue instead
//! leaves every session with a true record -- small when the session did little, never empty.
//!
//! The reduction is also a disclosure boundary. A live session's full manifest carried a secret in
//! its `prompt` and `completion` blobs while the file node and the tool payloads were correctly
//! withheld; the view does not carry it, because it keeps neither. That is a *scope* boundary and
//! not a redaction mechanism -- the view keeps a kept activity's tool result, so content the gate
//! failed to withhold there would still be in it.

use std::collections::{HashMap, HashSet};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Map, Value};

/// The `(assetType, name)` pairs that are the model's conversation with itself.
///
/// Keyed on both, because `assetType` alone cannot separate them: a human's instruction and the
/// conversation sent to the model are both `Prompt`, and dropping the type would take the
/// instruction with it -- the one thing a record of "what happened" must keep.
const MONOLOGUE: [(&str, &str); 3] = [
    ("Prompt", "prompt"),
    ("Reasoning", "completion"),
    ("System_Prompt", "system prompt"),
];

/// A written view, and the statement count its mark reports.
pub struct SessionView {
    pub bytes: Vec<u8>,
    pub statements: usize,
}

/// CIDs appear prefixed in statement fields and bare as blob keys and in `performedBy`.
///
/// Everything here is normalised to the bare form on the way in. Comparing the two forms is silent
/// and total rather than wrong in part: the sets simply never intersect, so nothing resolves, and
/// the result is an empty view rather than an error.
fn bare(cid: &str) -> &str {
    cid.strip_prefix("urn:cid:").unwrap_or(cid)
}

fn decode(blobs: &Map<String, Value>, cid: &str) -> Option<Value> {
    let encoded = blobs.get(bare(cid))?.as_str()?;
    let raw = BASE64.decode(encoded).ok()?;
    serde_json::from_slice(&raw).ok()
}

fn strings(value: &Value) -> Vec<&str> {
    match value {
        Value::String(one) => vec![one.as_str()],
        Value::Array(many) => many.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

/// Build the view for a manifest, or `None` when there is no *reduced* document to write.
///
/// `None` rather than an empty document on purpose: "nothing to project" is a fact about the
/// session, and a manifest with no statements is indistinguishable from a recording that failed.
/// `None` also when nothing was dropped -- see the guard below.
pub fn session_view(manifest: &Value) -> Option<SessionView> {
    let statements = manifest.get("statements")?.as_object()?;
    let blobs = manifest.get("blobs")?.as_object()?;

    // Every node's merged metadata, keyed bare. Merged rather than taken from one statement because
    // a node is described by several: the recorder states its type and name separately from what a
    // later statement adds about it.
    let mut described: HashMap<&str, Map<String, Value>> = HashMap::new();
    for statement in statements.values() {
        if statement.get("@type").and_then(Value::as_str) != Some("MetadataRegistration") {
            continue;
        }
        let (Some(subject), Some(metadata)) = (
            statement.get("subject").and_then(Value::as_str),
            statement.get("metadata").and_then(Value::as_str),
        ) else {
            continue;
        };
        if let Some(Value::Object(fields)) = decode(blobs, metadata) {
            described.entry(bare(subject)).or_default().extend(fields);
        }
    }

    let is_monologue = |cid: &str| -> bool {
        let Some(fields) = described.get(bare(cid)) else {
            return false;
        };
        let kind = fields
            .get("assetType")
            .and_then(Value::as_str)
            .unwrap_or("");
        let name = fields.get("name").and_then(Value::as_str).unwrap_or("");
        MONOLOGUE.iter().any(|(monologue_kind, monologue_name)| {
            *monologue_kind == kind && *monologue_name == name
        })
    };

    let mut keep: HashSet<String> = HashSet::new();
    for statement in statements.values() {
        if statement.get("@type").and_then(Value::as_str) != Some("DataRegistration") {
            continue;
        }
        if let Some(data) = statement.get("data") {
            for cid in strings(data) {
                if !is_monologue(cid) {
                    keep.insert(bare(cid).to_string());
                }
            }
        }
    }

    // An activity whose every output is monologue *is* the monologue. One whose outputs include
    // anything else stays, and brings its own inputs and outputs with it -- the tool that ran, what
    // it was invoked with, what it returned. Dropping those would leave a file edit attributed to an
    // anonymous activity, which is the question a reader came with.
    let mut activities: Vec<&str> = Vec::new();
    for (id, statement) in statements {
        if statement.get("@type").and_then(Value::as_str) != Some("ComputationRegistration") {
            continue;
        }
        let outputs: Vec<&str> = statement.get("output").map(strings).unwrap_or_default();
        if !outputs.is_empty() && outputs.iter().all(|cid| is_monologue(cid)) {
            continue;
        }
        activities.push(id);
        keep.insert(bare(id).to_string());
        let inputs: Vec<&str> = statement.get("input").map(strings).unwrap_or_default();
        for cid in inputs.into_iter().chain(outputs) {
            if !is_monologue(cid) {
                keep.insert(bare(cid).to_string());
            }
        }
    }

    // The actor a kept activity was associated with. Reached through `performedBy` in the activity's
    // metadata rather than through an edge, because an agent in `input` would assert the activity
    // consumed the agent -- PROV keeps association and usage apart (§7.2).
    for id in &activities {
        if let Some(who) = described
            .get(bare(id))
            .and_then(|fields| fields.get("performedBy"))
            .and_then(Value::as_str)
        {
            keep.insert(bare(who).to_string());
        }
    }

    let kept = select(statements, &keep);
    // Nothing to project, and nothing *reduced* to project. The first is a session that recorded
    // no non-monologue node; the second is one that held no conversation at all, so the view would
    // repeat the manifest under a second name and a second mark -- a document a reader has to open
    // to discover says the same thing.
    if kept.is_empty() || kept.len() == statements.len() {
        return None;
    }

    let mut referenced: HashSet<String> = HashSet::new();
    for statement in kept.values() {
        collect_cids(statement, &mut referenced);
    }
    let kept_blobs: Map<String, Value> = blobs
        .iter()
        .filter(|(cid, _)| referenced.contains(bare(cid)) || keep.contains(bare(cid)))
        .map(|(cid, value)| (cid.clone(), value.clone()))
        .collect();

    let mut view = Map::new();
    for field in ["version", "contexts"] {
        if let Some(value) = manifest.get(field) {
            view.insert(field.to_string(), value.clone());
        }
    }
    let count = kept.len();
    view.insert("statements".into(), Value::Object(kept));
    view.insert("blobs".into(), Value::Object(kept_blobs));
    // Anchors are proof that these statements were published. They are not monologue, and the
    // reference implementation dropped them by building its output from four named fields.
    if let Some(anchors) = manifest.get("anchors") {
        view.insert("anchors".into(), anchors.clone());
    }

    Some(SessionView {
        bytes: serde_json::to_vec_pretty(&Value::Object(view)).ok()?,
        statements: count,
    })
}

/// The statements that are *about* something kept, and the credentials that sign them.
fn select(statements: &Map<String, Value>, keep: &HashSet<String>) -> Map<String, Value> {
    let kept_by = |field: &str, statement: &Value| -> bool {
        statement
            .get(field)
            .map(|value| strings(value).iter().any(|cid| keep.contains(bare(cid))))
            .unwrap_or(false)
    };

    let mut kept: Map<String, Value> = Map::new();
    for (id, statement) in statements {
        let matched = match statement.get("@type").and_then(Value::as_str) {
            Some("DataRegistration") => kept_by("data", statement),
            Some("ComputationRegistration") => keep.contains(bare(id)),
            Some("MetadataRegistration") => kept_by("subject", statement),
            _ => false,
        };
        if matched {
            kept.insert(id.clone(), statement.clone());
        }
    }

    // Metadata about a kept *statement*, not only about a kept node -- an activity's own
    // `computation_type` and `performedBy` are described this way.
    let ids: HashSet<String> = kept.keys().map(|id| bare(id).to_string()).collect();
    for (id, statement) in statements {
        if statement.get("@type").and_then(Value::as_str) != Some("MetadataRegistration") {
            continue;
        }
        if statement
            .get("subject")
            .and_then(Value::as_str)
            .is_some_and(|subject| ids.contains(bare(subject)))
        {
            kept.insert(id.clone(), statement.clone());
        }
    }

    // The credentials over everything kept. Without this pass the view is a plausible document
    // rather than an attestation, which is the whole reason it is a subset and not a summary.
    let signed: HashSet<String> = kept.keys().map(|id| bare(id).to_string()).collect();
    for (id, statement) in statements {
        if statement.get("@type").and_then(Value::as_str) != Some("CredentialRegistration") {
            continue;
        }
        let subject = statement
            .get("credential")
            .and_then(|credential| credential.get("credentialSubject"))
            .and_then(|subject| subject.get("id"))
            .and_then(Value::as_str);
        if subject.is_some_and(|subject| signed.contains(bare(subject))) {
            kept.insert(id.clone(), statement.clone());
        }
    }
    kept
}

fn collect_cids(value: &Value, found: &mut HashSet<String>) {
    match value {
        Value::String(text) => {
            if text.starts_with("urn:cid:") || text.starts_with("baf") || text.starts_with("bag") {
                found.insert(bare(text).to_string());
            }
        }
        Value::Object(fields) => {
            for nested in fields.values() {
                collect_cids(nested, found);
            }
        }
        Value::Array(items) => {
            for nested in items {
                collect_cids(nested, found);
            }
        }
        _ => {}
    }
}
