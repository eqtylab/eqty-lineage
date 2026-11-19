//! The manifest a provider gateway signed for a model call, fetched and merged into the session's.
//!
//! Relay shows a plugin no response headers, so the gateway's own request id is out of reach. What
//! both sides do see is the `x-nemo-relay-request-id` Relay puts on every upstream request: the
//! gateway indexes its record by it, and the record names the manifest.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::Value as Json;

use crate::redaction::glob_match;

/// The request header Relay stamps on each upstream LLM request, unique per request.
pub const RELAY_REQUEST_ID: &str = "x-nemo-relay-request-id";

/// The gateway writes its record in the background after the response, so a miss is retried.
const RETRY_EVERY: Duration = Duration::from_millis(500);
/// Export runs at gateway shutdown on Codex; this bounds how long one session can hold it up.
const SESSION_BUDGET: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
/// A manifest inlines its blobs, so one can be large; this only stops a response without end.
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// One gateway that serves manifests, and the model calls it is asked about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provider {
    /// Where the gateway's `/integrity` routes live. From configuration only, never from traffic.
    pub base_url: String,
    /// The environment variable holding the gateway key. Relay strips `Authorization` from what a
    /// plugin sees, so the key the agent used has to be read from where Relay read it.
    pub api_key_env: Option<String>,
    /// Model name globs this gateway is asked about. Empty means every model.
    pub models: Vec<String>,
}

impl Provider {
    pub fn serves(&self, model: Option<&str>) -> bool {
        self.models.is_empty()
            || model.is_some_and(|model| self.models.iter().any(|glob| glob_match(glob, model)))
    }
}

/// A provider manifest, and the two nodes in it a model call joins.
pub struct Fetched {
    /// The gateway's own id for the call, which is what its manifest route is keyed by.
    pub request_id: String,
    pub client_request: Option<Vec<u8>>,
    pub response: Option<Vec<u8>>,
    pub manifest: Json,
}

/// An id is interpolated into a URL path, so it is held to characters that cannot leave it.
fn is_safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
}

/// Fetch the provider manifest for each `(relay request id, model)`, keyed by relay request id.
/// A call with none is absent, whatever the reason.
pub async fn fetch_all(
    providers: &[Provider],
    calls: Vec<(String, Option<String>)>,
) -> HashMap<String, Fetched> {
    let mut pending: Vec<(String, &Provider)> = calls
        .into_iter()
        .filter(|(id, _)| is_safe_id(id))
        .filter_map(|(id, model)| {
            let provider = providers.iter().find(|p| p.serves(model.as_deref()))?;
            Some((id, provider))
        })
        .collect();
    let mut fetched = HashMap::new();
    if pending.is_empty() {
        return fetched;
    }
    let Ok(client) = reqwest::Client::builder().timeout(REQUEST_TIMEOUT).build() else {
        return fetched;
    };

    let deadline = Instant::now() + SESSION_BUDGET;
    loop {
        let mut retry = Vec::new();
        for (id, provider) in pending {
            match attempt(&client, provider, &id).await {
                Attempt::Ready(found) => {
                    fetched.insert(id, *found);
                }
                Attempt::NotYet => retry.push((id, provider)),
                Attempt::Never => {}
            }
        }
        pending = retry;
        if pending.is_empty() || Instant::now() + RETRY_EVERY > deadline {
            return fetched;
        }
        tokio::time::sleep(RETRY_EVERY).await;
    }
}

enum Attempt {
    Ready(Box<Fetched>),
    NotYet,
    Never,
}

async fn attempt(client: &reqwest::Client, provider: &Provider, relay_request_id: &str) -> Attempt {
    let base = provider.base_url.trim_end_matches('/');
    let key = provider
        .api_key_env
        .as_deref()
        .and_then(|name| std::env::var(name).ok());

    let record = match get(
        client,
        &format!("{base}/integrity/logs/by-client-request-id/{relay_request_id}"),
        key.as_deref(),
    )
    .await
    {
        Ok(Some(record)) => record,
        // Not written yet and not ours look the same from here.
        Ok(None) => return Attempt::NotYet,
        Err(()) => return Attempt::Never,
    };
    match record.get("status").and_then(Json::as_str) {
        Some("ready") => {}
        Some("failed") => return Attempt::Never,
        _ => return Attempt::NotYet,
    }
    let Some(request_id) = record
        .get("request_id")
        .and_then(Json::as_str)
        .filter(|id| is_safe_id(id))
    else {
        return Attempt::Never;
    };

    let manifest = match get(
        client,
        &format!("{base}/integrity/manifest/{request_id}"),
        key.as_deref(),
    )
    .await
    {
        Ok(Some(manifest)) => manifest,
        Ok(None) => return Attempt::NotYet,
        Err(()) => return Attempt::Never,
    };
    Attempt::Ready(Box::new(Fetched {
        request_id: request_id.to_string(),
        client_request: named_content(&manifest, &["Client Request"]),
        // The gateway registers a delivered response only when it differs from the provider's.
        response: named_content(&manifest, &["Delivered Response", "Response Body"]),
        manifest,
    }))
}

/// `Ok(None)` is a 404, which the caller may retry; `Err` is anything it must not.
async fn get(client: &reqwest::Client, url: &str, key: Option<&str>) -> Result<Option<Json>, ()> {
    let mut request = client.get(url);
    if let Some(key) = key {
        request = request.bearer_auth(key);
    }
    let response = request.send().await.map_err(|_| ())?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !response.status().is_success() {
        return Err(());
    }
    // ponytail: the body is read whole before the size check; stream it if a gateway is untrusted.
    let body = response.bytes().await.map_err(|_| ())?;
    if body.len() > MAX_RESPONSE_BYTES {
        return Err(());
    }
    serde_json::from_slice(&body).map(Some).map_err(|_| ())
}

/// The bytes of the first asset the manifest names with one of `names`, in that order of preference.
fn named_content(manifest: &Json, names: &[&str]) -> Option<Vec<u8>> {
    let statements = manifest.get("statements")?.as_object()?;
    let blobs = manifest.get("blobs")?.as_object()?;
    let blob = |reference: &Json| {
        let cid = reference.as_str()?.trim_start_matches("urn:cid:");
        STANDARD.decode(blobs.get(cid)?.as_str()?).ok()
    };
    names.iter().find_map(|wanted| {
        statements.values().find_map(|statement| {
            let metadata: Json = serde_json::from_slice(&blob(statement.get("metadata")?)?).ok()?;
            (metadata.get("name")?.as_str()? == *wanted)
                .then(|| blob(statement.get("subject")?))
                .flatten()
        })
    })
}

/// Fold a provider manifest into ours; `false` when its version differs. Ours wins a key clash.
///
/// JSON, not the typed `Manifest`: a statement decoded and re-encoded can lose a field this build
/// does not know, and then no longer hashes to the id its credential signed.
pub fn merge(ours: &mut Json, theirs: Json) -> bool {
    if ours.get("version") != theirs.get("version") {
        return false;
    }
    let Json::Object(mut theirs) = theirs else {
        return false;
    };
    for section in ["contexts", "statements", "blobs"] {
        let (Some(Json::Object(into)), Some(Json::Object(from))) =
            (ours.get_mut(section), theirs.remove(section))
        else {
            continue;
        };
        for (key, value) in from {
            into.entry(key).or_insert(value);
        }
    }
    true
}
