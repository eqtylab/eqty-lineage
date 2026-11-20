//! Signs with a notary when `EQTY_NOTARY_URL` is set, and otherwise with a key per session.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use integrity::signer::{
    SignerType, ed25519_signer::Ed25519Signer, vcomp_notary::VCompNotarySigner,
};
use serde_json::Value as Json;

use crate::lineage::LineageSession;
use crate::mailbox::SignerFactory;
use crate::provider::{MANIFEST_VERSION, MAX_RESPONSE_BYTES, is_mergeable, read_capped};

pub const NOTARY_URL: &str = "EQTY_NOTARY_URL";
/// For all of connecting. The first session waits on it.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// The notary URL from the environment. Unset, empty and blank all mean "no notary".
pub fn url_from_env() -> Option<String> {
    std::env::var(NOTARY_URL)
        .ok()
        .map(|url| url.trim().to_string())
        .filter(|url| !url.is_empty())
}

/// Every session signs with the notary at `url`, or with a key of its own when there is none or it
/// fails. Connected on first use, on the worker, so a dead notary never holds up Relay's startup.
pub fn session_factory(url: Option<String>) -> SignerFactory {
    let Some(url) = url else {
        return own_key(false);
    };
    let notary = OnceLock::new();
    let failed = Arc::new(AtomicBool::new(false));
    let fallback = own_key(true);
    Box::new(move || {
        let notary = notary.get_or_init(|| {
            connect(&url)
                .map_err(|reason| {
                    eprintln!(
                        "eqty.lineage: warning: cannot use the notary at {url} ({reason}); \
                         signing with a software key instead"
                    )
                })
                .ok()
        });
        match notary {
            Some(notary) if !failed.load(Ordering::Relaxed) => Some(
                LineageSession::new(SignerType::VCompNotarySigner(notary.signer.clone()))
                    .with_signer_identity(
                        notary.signer.executed_on.clone(),
                        notary.signer.operated_by.clone(),
                        Some(Arc::clone(&notary.registration)),
                    )
                    .reporting_failure_to(Arc::clone(&failed)),
            ),
            _ => fallback(),
        }
    })
}

fn own_key(fell_back: bool) -> SignerFactory {
    Box::new(move || {
        let session = LineageSession::new(SignerType::ED25519(Ed25519Signer::create().ok()?));
        Some(if fell_back {
            session.fell_back()
        } else {
            session
        })
    })
}

struct Notary {
    signer: VCompNotarySigner,
    registration: Arc<Json>,
}

fn connect(url: &str) -> Result<Notary, String> {
    if !is_signable_url(url) {
        return Err(format!(
            "{NOTARY_URL} must be http://<host>:<port>, with an explicit port and no path"
        ));
    }
    let url = url.trim_end_matches('/');
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    let connecting = async {
        let mut signer = VCompNotarySigner::create(url, None)
            .await
            .map_err(|error| error.to_string())?;
        // The signer is cloned for every signature; the registration travels separately.
        signer.credentials = None;
        signer.did_blobs = None;
        // Fetched whole, because the signer keeps only the statements and blobs and drops the
        // contexts a verifier needs to read them.
        let response = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|error| error.to_string())?
            .get(format!("{url}/v1/did_registration"))
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| error.to_string())?;
        let body = read_capped(response, MAX_RESPONSE_BYTES)
            .await
            .map_err(|_| "its DID registration could not be read".to_string())?;
        let registration: Json =
            serde_json::from_slice(&body).map_err(|error| error.to_string())?;
        Ok(Notary {
            signer,
            registration: Arc::new(as_manifest(registration)?),
        })
    };
    let connected =
        runtime.block_on(async { tokio::time::timeout(CONNECT_TIMEOUT, connecting).await });
    // Dropping a runtime waits for its blocking tasks, and a DNS lookup that has not given up is
    // one of them. Shut down without waiting, so the timeout is the whole wait.
    runtime.shutdown_background();
    connected.map_err(|_| "it did not answer in time".to_string())?
}

/// Whether the notary signer would sign at `url` itself: it signs over plain TCP to the host and
/// port, at a fixed path, and on port 8000 when none is given.
fn is_signable_url(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    // Read from the text: the parser drops a port that is the scheme's default, so `:80` would
    // look like no port at all.
    let authority = url
        .strip_prefix("http://")
        .and_then(|rest| rest.split('/').next())
        .unwrap_or_default();
    let has_port = authority
        .rsplit_once(':')
        .is_some_and(|(_, port)| !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()));
    // The signer parses with `http::Uri` and panics on a URL it cannot read, or one with no host.
    let signable = url
        .parse::<http::Uri>()
        .is_ok_and(|uri| uri.host().is_some());
    signable
        && parsed.scheme() == "http"
        && has_port
        && parsed.path() == "/"
        && parsed.query().is_none()
}

/// Check the registration can be merged into a manifest of ours.
fn as_manifest(mut registration: Json) -> Result<Json, String> {
    if let Some(manifest) = registration.as_object_mut() {
        manifest
            .entry("version")
            .or_insert_with(|| MANIFEST_VERSION.into());
    }
    if !is_mergeable(&registration) {
        return Err(format!(
            "its DID registration is not a version {MANIFEST_VERSION} manifest"
        ));
    }
    Ok(registration)
}

#[cfg(test)]
mod tests {
    use super::is_signable_url;

    #[test]
    fn a_notary_url_needs_http_an_explicit_port_and_no_path() {
        for good in [
            "http://127.0.0.1:8066",
            "http://notary:80",
            "http://notary:8000/",
            "http://[::1]:8000",
        ] {
            assert!(is_signable_url(good), "{good}");
        }
        for bad in [
            "http://notary",
            "http://[::1]",
            "http://notary:8000/notary",
            "http://notary:8000/?a=b",
            "http://nötary:8000",
            "https://notary:8000",
            "notary:8000",
        ] {
            assert!(!is_signable_url(bad), "{bad}");
        }
    }
}
