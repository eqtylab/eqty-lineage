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
/// How long one session's export may wait on its gateways, in total. The worker that waits is the
/// one every session's events queue behind, and on Codex export runs at gateway shutdown.
const SESSION_BUDGET: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
/// The manifest version this build writes, and so the only one it can merge.
const MANIFEST_VERSION: &str = "3";
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

    /// The gateway key, `Ok(None)` when none is configured, `Err` when the named variable is unset.
    /// Unset is not "no key": asking without one would only be refused.
    pub fn key(&self) -> Result<Option<String>, std::env::VarError> {
        self.api_key_env.as_deref().map(std::env::var).transpose()
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

/// An id is interpolated into a URL path, so it is held to characters that cannot leave it. All
/// dots is refused too: `..` is a path segment a URL parser removes.
fn is_safe_id(id: &str) -> bool {
    id.len() <= 128
        && id.bytes().any(|byte| byte != b'.')
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
}

/// Fetch the provider manifest for each `(relay request id, model)`, keyed by relay request id.
/// A call with none is absent, whatever the reason.
///
/// The first provider serving a call's model is the only one asked about it.
pub async fn fetch_all(
    providers: &[Provider],
    calls: Vec<(String, Option<String>)>,
) -> HashMap<String, Fetched> {
    let keys: Vec<_> = providers.iter().map(Provider::key).collect();
    let mut pending: Vec<(String, usize)> = calls
        .into_iter()
        .filter(|(id, _)| is_safe_id(id))
        .filter_map(|(id, model)| {
            let provider = providers.iter().position(|p| p.serves(model.as_deref()))?;
            keys[provider].is_ok().then_some((id, provider))
        })
        .collect();
    let mut fetched = HashMap::new();
    let Ok(client) = reqwest::Client::builder().build() else {
        return fetched;
    };

    let deadline = Instant::now() + SESSION_BUDGET;
    // A gateway that cannot be reached is not asked again: every further call would wait as long.
    let mut unreachable = vec![false; providers.len()];
    while !pending.is_empty() {
        let mut retry = Vec::new();
        for (id, provider) in pending {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return fetched;
            };
            if unreachable[provider] {
                continue;
            }
            let ask = Ask {
                client: &client,
                base: providers[provider].base_url.trim_end_matches('/'),
                key: keys[provider].as_ref().ok().and_then(Option::as_deref),
                timeout: REQUEST_TIMEOUT.min(left),
            };
            match ask.manifest_for(&id).await {
                Attempt::Ready(found) => {
                    fetched.insert(id, *found);
                }
                Attempt::NotYet => retry.push((id, provider)),
                Attempt::Never => {}
                Attempt::Unreachable => unreachable[provider] = true,
            }
        }
        pending = retry;
        if !pending.is_empty() {
            tokio::time::sleep(RETRY_EVERY).await;
        }
    }
    fetched
}

enum Attempt {
    Ready(Box<Fetched>),
    NotYet,
    Never,
    Unreachable,
}

/// `NotFound` is a 404, which may be retried; `Refused` is an answer that must not be.
enum Miss {
    NotFound,
    Refused,
    Unreachable,
}

impl From<Miss> for Attempt {
    fn from(miss: Miss) -> Self {
        match miss {
            // Not written yet and not ours look the same from here.
            Miss::NotFound => Attempt::NotYet,
            Miss::Refused => Attempt::Never,
            Miss::Unreachable => Attempt::Unreachable,
        }
    }
}

struct Ask<'a> {
    client: &'a reqwest::Client,
    base: &'a str,
    key: Option<&'a str>,
    timeout: Duration,
}

impl Ask<'_> {
    async fn manifest_for(&self, relay_request_id: &str) -> Attempt {
        let record = match self
            .get(&format!(
                "integrity/logs/by-client-request-id/{relay_request_id}"
            ))
            .await
        {
            Ok(record) => record,
            Err(miss) => return miss.into(),
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

        let manifest = match self.get(&format!("integrity/manifest/{request_id}")).await {
            Ok(manifest) => manifest,
            Err(miss) => return miss.into(),
        };
        // Checked before anything is linked to it: a manifest that cannot be merged must not be
        // counted as merged, nor joined to by statements that would then lead nowhere.
        if manifest.get("version").and_then(Json::as_str) != Some(MANIFEST_VERSION)
            || !manifest.get("statements").is_some_and(Json::is_object)
            || !manifest.get("blobs").is_some_and(Json::is_object)
        {
            return Attempt::Never;
        }
        Attempt::Ready(Box::new(Fetched {
            request_id: request_id.to_string(),
            client_request: named_content(&manifest, &["Client Request"]),
            // The gateway registers a delivered response only when it differs from the provider's.
            response: named_content(&manifest, &["Delivered Response", "Response Body"]),
            manifest,
        }))
    }

    async fn get(&self, path: &str) -> Result<Json, Miss> {
        let mut request = self
            .client
            .get(format!("{}/{path}", self.base))
            .timeout(self.timeout);
        if let Some(key) = self.key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(|_| Miss::Unreachable)?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(Miss::NotFound);
        }
        if !response.status().is_success() {
            return Err(Miss::Refused);
        }
        // ponytail: the body is read whole before the size check; stream it if a gateway is untrusted.
        let body = response.bytes().await.map_err(|_| Miss::Unreachable)?;
        if body.len() > MAX_RESPONSE_BYTES {
            return Err(Miss::Refused);
        }
        serde_json::from_slice(&body).map_err(|_| Miss::Refused)
    }
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
    let (Json::Object(ours), Json::Object(mut theirs)) = (ours, theirs) else {
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
    // Absent from a manifest that has none, so the section is made only when there is one to keep.
    if let Some(Json::Array(anchors)) = theirs.remove("anchors") {
        match ours.get_mut("anchors") {
            Some(Json::Array(into)) => into.extend(anchors),
            _ => {
                ours.insert("anchors".into(), Json::Array(anchors));
            }
        }
    }
    true
}
