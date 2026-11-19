//! Fetches the manifest a provider gateway signed for a model call, and merges it into ours.
//!
//! Relay never shows a plugin response headers, so the gateway's own request id cannot be read.
//! Both sides do see the `x-nemo-relay-request-id` Relay adds to each upstream request. The
//! gateway's record is looked up by that id, and the record gives the id of the manifest.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use futures::stream::{self, StreamExt};
use serde_json::Value as Json;

use crate::redaction::glob_match;

/// The header Relay adds to each upstream LLM request. Its value is unique per request.
pub const RELAY_REQUEST_ID: &str = "x-nemo-relay-request-id";

/// The gateway writes its record after it has sent the response, so a miss is retried.
const RETRY_EVERY: Duration = Duration::from_millis(500);
/// The most time one session's export may spend waiting on gateways. Every session's events queue
/// behind this wait, and on Codex it happens at gateway shutdown.
const SESSION_BUDGET: Duration = Duration::from_secs(10);
/// How long a 404 is retried, at least. A call the gateway never saw answers 404 for good.
const NOT_FOUND_BUDGET: Duration = Duration::from_secs(3);
/// How long after a call ends the gateway may still be writing its record. A 404 for a call that
/// ended more recently than this is retried until this has passed.
const RECORD_WRITE: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
/// Calls asked about at once. Enough to fit a long session in the budget without flooding the
/// gateway.
const CONCURRENT_ASKS: usize = 8;
/// The manifest version this build writes. A manifest of any other version cannot be merged.
const MANIFEST_VERSION: &str = "3";
/// Manifests carry their blobs inline and can be large. This only stops an endless response.
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// One gateway that serves manifests, and which model calls to ask it about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provider {
    /// Base URL of the gateway's `/integrity` routes. From configuration, never from traffic.
    pub base_url: String,
    /// Name of the environment variable holding the gateway key. Relay hides `Authorization` from
    /// plugins, so the key is read from the same variable Relay read it from.
    pub api_key_env: Option<String>,
    /// Model name globs to ask this gateway about. Empty means every model.
    pub models: Vec<String>,
}

impl Provider {
    pub fn serves(&self, model: Option<&str>) -> bool {
        self.models.is_empty()
            || model.is_some_and(|model| self.models.iter().any(|glob| glob_match(glob, model)))
    }

    /// The gateway key. `Ok(None)` means no key is configured; `Err` means the variable is unset.
    /// An unset variable must not count as "no key", or requests would go out unauthenticated.
    pub fn key(&self) -> Result<Option<String>, std::env::VarError> {
        self.api_key_env.as_deref().map(std::env::var).transpose()
    }
}

/// A fetched provider manifest, and the two nodes in it that a model call links to.
pub struct Fetched {
    /// The gateway's own id for the call. Its manifest route is keyed by this.
    pub request_id: String,
    pub client_request: Option<Vec<u8>>,
    pub response: Option<Vec<u8>>,
    pub manifest: Json,
}

/// Ids go into a URL path, so only characters that cannot change the path are allowed. An id of
/// only dots is refused as well, because a URL parser reads `..` as "go up one level".
fn is_safe_id(id: &str) -> bool {
    id.len() <= 128
        && id.bytes().any(|byte| byte != b'.')
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
}

/// Fetch the provider manifest for each `(relay request id, model, ended)` call, keyed by relay
/// request id. A call with no manifest is left out of the result.
///
/// Only the first provider whose `models` match a call is asked about it.
pub async fn fetch_all(
    providers: &[Provider],
    calls: Vec<(String, Option<String>, Instant)>,
) -> HashMap<String, Fetched> {
    let keys: Vec<_> = providers.iter().map(Provider::key).collect();
    // The key must not follow a redirect off the configured origin or down to plain http.
    let builder = || reqwest::Client::builder().redirect(reqwest::redirect::Policy::none());
    // Plain http is only allowed to loopback, which never needs a proxy, and a proxy would see the
    // key in clear. An https gateway keeps the environment's proxy: the key stays inside TLS.
    let (Ok(proxied), Ok(direct)) = (builder().build(), builder().no_proxy().build()) else {
        return HashMap::new();
    };
    let deadline = Instant::now() + SESSION_BUDGET;
    let asks = calls
        .into_iter()
        .filter(|(id, _, _)| is_safe_id(id))
        .filter_map(|(id, model, ended)| {
            let provider = providers.iter().position(|p| p.serves(model.as_deref()))?;
            let key = keys[provider].as_ref().ok()?.as_deref();
            let base = providers[provider].base_url.trim_end_matches('/');
            Some((id, base, key, ended))
        })
        .map(|(id, base, key, ended)| {
            let client = if base.starts_with("http://") {
                &direct
            } else {
                &proxied
            };
            async move {
                let found = fetch_one(client, base, key, &id, ended, deadline).await;
                found.map(|found| (id, found))
            }
        });
    stream::iter(asks)
        .buffer_unordered(CONCURRENT_ASKS)
        .filter_map(std::future::ready)
        .collect()
        .await
}

/// Ask about one call until it is answered, refused, or out of time.
async fn fetch_one(
    client: &reqwest::Client,
    base: &str,
    key: Option<&str>,
    id: &str,
    ended: Instant,
    deadline: Instant,
) -> Option<Fetched> {
    // A 404 also means "a call this gateway never saw", so it gets less time than "fetching",
    // unless the call ended so recently that its record may not be written yet.
    let not_found_deadline = (Instant::now() + NOT_FOUND_BUDGET)
        .max(ended + RECORD_WRITE)
        .min(deadline);
    loop {
        let left = deadline.checked_duration_since(Instant::now())?;
        let ask = Ask {
            client,
            base,
            key,
            timeout: REQUEST_TIMEOUT.min(left),
        };
        match ask.manifest_for(id).await {
            Attempt::Ready(found) => return Some(*found),
            Attempt::NotYet(Miss::NotFound) if Instant::now() >= not_found_deadline => return None,
            Attempt::NotYet(_) => tokio::time::sleep(RETRY_EVERY).await,
            Attempt::Never => return None,
        }
    }
}

enum Attempt {
    Ready(Box<Fetched>),
    /// Worth asking again. Carries why, so a 404 can be given up on sooner.
    NotYet(Miss),
    Never,
}

/// `NotFound` is a 404 and may be retried. `Refused` is any other answer, or no connection at all,
/// and must not be. `Unreachable` is a timeout or a broken body, which may pass, so it is retried.
#[derive(Debug)]
enum Miss {
    NotFound,
    Refused,
    Unreachable,
    /// The record exists but the gateway is still fetching the response.
    Pending,
}

impl From<Miss> for Attempt {
    fn from(miss: Miss) -> Self {
        match miss {
            Miss::Refused => Attempt::Never,
            // A 404 can mean "not written yet" or "not ours". They look the same, so retry.
            miss => Attempt::NotYet(miss),
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
            _ => return Attempt::NotYet(Miss::Pending),
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
        // Check this before linking. A manifest that cannot be merged must not be counted as
        // merged, and our statements must not point into it.
        if manifest.get("version").and_then(Json::as_str) != Some(MANIFEST_VERSION)
            || !manifest.get("statements").is_some_and(Json::is_object)
            || !manifest.get("blobs").is_some_and(Json::is_object)
        {
            return Attempt::Never;
        }
        Attempt::Ready(Box::new(Fetched {
            request_id: request_id.to_string(),
            client_request: named_content(&manifest, &["Client Request"]),
            // The gateway only registers a delivered response when it differs from the provider's.
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
        // A refused connection means the gateway is down, and asking again in 500 ms will not help.
        let response = request.send().await.map_err(|error| {
            if error.is_connect() {
                Miss::Refused
            } else {
                Miss::Unreachable
            }
        })?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(Miss::NotFound);
        }
        if !response.status().is_success() {
            return Err(Miss::Refused);
        }
        let body = read_capped(response, MAX_RESPONSE_BYTES).await?;
        serde_json::from_slice(&body).map_err(|_| Miss::Refused)
    }
}

/// Read a body of at most `limit` bytes. The limit is checked as the body arrives, so a larger
/// body is never held in memory.
async fn read_capped(mut response: reqwest::Response, limit: usize) -> Result<Vec<u8>, Miss> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(Miss::Refused);
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| Miss::Unreachable)? {
        if body.len() + chunk.len() > limit {
            return Err(Miss::Refused);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// The bytes of the first asset the manifest names with one of `names`, tried in the order given.
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

/// Merge a provider manifest into ours. Returns `false` if the versions differ. On a key clash,
/// ours is kept.
///
/// Merged as JSON, not through the typed `Manifest`. Decoding and re-encoding a statement can drop
/// fields this build does not know, and then it no longer hashes to the id its credential signed.
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
    // Our manifest has no `anchors` key when it has none, so add the key only if there are some.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A gateway that answers every request with `answer(hits so far)`, and counts the hits.
    fn gateway(
        answer: impl Fn(usize) -> (u16, &'static str) + Send + 'static,
    ) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use std::io::{Read, Write};
        use std::sync::atomic::Ordering;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a free port");
        let url = format!("http://{}", listener.local_addr().expect("an address"));
        let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&hits);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let _ = stream.read(&mut [0u8; 4096]);
                let (status, body) = answer(counted.fetch_add(1, Ordering::SeqCst));
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        (url, hits)
    }

    fn fetch_ended(url: String, ended: Instant) -> HashMap<String, Fetched> {
        let provider = Provider {
            base_url: url,
            api_key_env: None,
            models: Vec::new(),
        };
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime")
            .block_on(fetch_all(&[provider], vec![("call-1".into(), None, ended)]))
    }

    #[test]
    fn a_loopback_gateway_is_asked_directly_even_when_a_proxy_is_set() {
        // Safe: every other test in this binary uses a client that ignores the proxy variables.
        unsafe { std::env::set_var("HTTP_PROXY", "http://127.0.0.1:9") };
        let (url, hits) = gateway(|_| (200, r#"{"status":"failed"}"#));
        assert!(fetch_ended(url, Instant::now()).is_empty());
        unsafe { std::env::remove_var("HTTP_PROXY") };
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn a_404_for_a_call_that_ended_long_ago_is_given_up_early() {
        let (url, hits) = gateway(|_| (404, "{}"));
        let long_ago = Instant::now() - Duration::from_secs(60);
        let started = Instant::now();
        assert!(fetch_ended(url, long_ago).is_empty());
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(
            hits.load(std::sync::atomic::Ordering::SeqCst) > 1,
            "a 404 is retried"
        );
    }

    #[test]
    fn a_404_for_a_call_that_just_ended_is_retried_until_the_gateway_has_had_time_to_record_it() {
        // 404 for about 4 s, past the short window, then an answer that ends the wait.
        let (url, hits) = gateway(|hit| {
            if hit < 8 {
                (404, "{}")
            } else {
                (200, r#"{"status":"failed"}"#)
            }
        });
        assert!(fetch_ended(url, Instant::now()).is_empty());
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 9);
    }

    #[test]
    fn a_body_over_the_limit_is_refused_before_it_has_all_arrived() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a free port");
        let url = format!("http://{}", listener.local_addr().expect("an address"));
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("a connection");
            let _ = stream.read(&mut [0u8; 4096]);
            // No content-length, and the body never ends.
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nconnection: close\r\n\r\n");
            while stream.write_all(&[b' '; 4096]).is_ok() {}
        });

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime");
        let read = runtime.block_on(async {
            let client = reqwest::Client::builder().no_proxy().build();
            let response = client
                .expect("a client")
                .get(&url)
                .send()
                .await
                .expect("a response");
            read_capped(response, 64 * 1024).await
        });
        assert!(matches!(read, Err(Miss::Refused)));
    }
}
