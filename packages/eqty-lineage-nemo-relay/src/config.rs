//! Component configuration, and the diagnostics Relay shows when it is wrong.
//!
//! Relay validates a component's config twice: once when `nemo-relay plugins validate` runs, before
//! the gateway is started, and again when the component is registered. Both go through
//! [`Config::parse`], so a mistake surfaces at install time rather than as a session that quietly
//! recorded nothing.

use std::path::PathBuf;

use nemo_relay_plugin::{ConfigDiagnostic, DiagnosticLevel, Json};
use serde_json::Map;

use crate::provider::Provider;

/// Where manifests are written, relative to the directory Relay runs in.
const DEFAULT_MANIFEST_DIR: &str = ".eqty/manifests";

/// Content larger than this is recorded by CID and metadata but its bytes are not stored.
///
/// Override per host with `max_content_bytes` in the component's config block.
///
/// A withheld node still carries the path and the true content CID computed from the bytes before
/// the decision, so raising this trades manifest size for readable content, never for lineage:
/// nothing is dropped from the graph either way. Size is the thing to weigh -- blobs are inlined
/// into the manifest as base64, so a stored payload costs about 4/3 of its own length on disk.
const DEFAULT_MAX_CONTENT_BYTES: u64 = 104_857_600;

/// Paths whose bytes are withheld from the file node and from the tool payloads named against them.
///
/// That is the whole reach, and it is a floor rather than a guarantee. A payload is matched on its
/// own name and on the paths it quotes, so a shell command that names no path is outside it -- and
/// so is the conversation, whose prompts are named `prompt` and quote nothing. Bytes a denied read
/// carried into the next model call are stored.
///
/// Provenance does not require publication: a denied file still becomes a graph node carrying its
/// path and its true content CID, computed from the bytes before redaction. Only the bytes are
/// withheld. Dropping the node instead would make the manifest silently incomplete.
const DEFAULT_DENY_GLOBS: &[&str] = &[".env*", "*.pem", "id_rsa*", "*/.ssh/*", "*credentials*"];

/// Validated configuration for one `eqty.lineage` component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Directory manifests are written to.
    pub manifest_dir: PathBuf,
    /// Glob patterns whose file contents are withheld.
    pub deny_globs: Vec<String>,
    /// Ceiling on stored content size, in bytes.
    pub max_content_bytes: u64,
    /// Gateways to ask for the manifest they signed for each model call. Empty by default.
    pub provider_manifests: Vec<Provider>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            manifest_dir: PathBuf::from(DEFAULT_MANIFEST_DIR),
            deny_globs: DEFAULT_DENY_GLOBS
                .iter()
                .map(|glob| (*glob).to_string())
                .collect(),
            max_content_bytes: DEFAULT_MAX_CONTENT_BYTES,
            provider_manifests: Vec::new(),
        }
    }
}

impl Config {
    /// Parse a component config, collecting every problem rather than stopping at the first.
    ///
    /// Returning all diagnostics at once matters for the install path: `nemo-relay plugins validate`
    /// prints what it is given, and a user who has three fields wrong should learn that once.
    pub fn parse(raw: &Map<String, Json>) -> (Self, Vec<ConfigDiagnostic>) {
        let mut config = Self::default();
        let mut diagnostics = Vec::new();

        if let Some(value) = raw.get("manifest_dir") {
            match value.as_str() {
                Some(dir) if !dir.trim().is_empty() => config.manifest_dir = PathBuf::from(dir),
                _ => diagnostics.push(error(
                    "manifest_dir.invalid",
                    "manifest_dir",
                    "manifest_dir must be a non-empty string",
                )),
            }
        }

        if let Some(value) = raw.get("deny_globs") {
            match value.as_array() {
                Some(globs) if globs.iter().all(Json::is_string) => {
                    // An empty list is legal and turns every default off at once. It is also what a
                    // templating bug produces, and the failure mode is a leak rather than a crash,
                    // so it says so. A warning and not an error: recording a tree with nothing to
                    // withhold is a real choice, and a diagnostic must not cost a session its
                    // manifest.
                    if globs.is_empty() {
                        diagnostics.push(warning(
                            "deny_globs.empty",
                            "deny_globs",
                            "deny_globs is empty, so nothing is withheld -- the defaults \
                             (.env*, *.pem, id_rsa*, */.ssh/*, *credentials*) do not apply",
                        ));
                    }
                    config.deny_globs = globs
                        .iter()
                        .filter_map(|glob| glob.as_str().map(str::to_string))
                        .collect();
                }
                _ => diagnostics.push(error(
                    "deny_globs.invalid",
                    "deny_globs",
                    "deny_globs must be an array of strings",
                )),
            }
        }

        if let Some(value) = raw.get("max_content_bytes") {
            match value.as_u64() {
                Some(max) if max > 0 => config.max_content_bytes = max,
                _ => diagnostics.push(error(
                    "max_content_bytes.invalid",
                    "max_content_bytes",
                    "max_content_bytes must be a positive integer",
                )),
            }
        }

        if let Some(value) = raw.get("provider_manifests") {
            match value
                .as_array()
                .map(|entries| entries.iter().map(provider).collect::<Option<Vec<_>>>())
            {
                Some(Some(providers)) => config.provider_manifests = providers,
                _ => diagnostics.push(error(
                    "provider_manifests.invalid",
                    "provider_manifests",
                    "provider_manifests must be an array of tables, each with an https base_url \
                     (http only to localhost), and optionally api_key_env (a non-empty string) \
                     and models (an array of strings)",
                )),
            }
        }

        (config, diagnostics)
    }
}

/// Parse one `provider_manifests` entry. Returns `None` if any part of it is malformed.
///
/// Unknown keys are refused, not ignored. A misspelt `models` would otherwise match every model
/// and send the gateway key with all of their request ids.
fn provider(entry: &Json) -> Option<Provider> {
    let entry = entry.as_object()?;
    if !entry
        .keys()
        .all(|key| matches!(key.as_str(), "base_url" | "api_key_env" | "models"))
    {
        return None;
    }
    let base_url = entry.get("base_url")?.as_str()?;
    if !is_safe_base_url(base_url) {
        return None;
    }
    let api_key_env = match entry.get("api_key_env") {
        Some(name) => Some(name.as_str().filter(|name| !name.trim().is_empty())?),
        None => None,
    };
    let models = match entry.get("models") {
        Some(models) => models
            .as_array()?
            .iter()
            .map(|glob| glob.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()?,
        None => Vec::new(),
    };
    Some(Provider {
        base_url: base_url.to_string(),
        api_key_env: api_key_env.map(str::to_string),
        models,
    })
}

/// The gateway key is sent as a bearer token, so plain `http` is allowed only to this machine.
fn is_safe_base_url(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    url.scheme() == "https"
        || url.scheme() == "http"
            && (host == "localhost"
                || host
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|ip| ip.is_loopback()))
}

/// `register` refuses on errors alone, never on any diagnostic. That is what lets `deny_globs.empty`
/// say a dangerous-but-legal thing is dangerous without costing the session its manifest.
fn error(code: &str, field: &str, message: &str) -> ConfigDiagnostic {
    diagnostic(DiagnosticLevel::Error, code, field, message)
}

fn warning(code: &str, field: &str, message: &str) -> ConfigDiagnostic {
    diagnostic(DiagnosticLevel::Warning, code, field, message)
}

fn diagnostic(level: DiagnosticLevel, code: &str, field: &str, message: &str) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level,
        code: code.to_string(),
        component: None,
        field: Some(field.to_string()),
        message: message.to_string(),
    }
}
