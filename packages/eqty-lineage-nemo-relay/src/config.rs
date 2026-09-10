//! Component configuration, and the diagnostics Relay shows when it is wrong.
//!
//! Relay validates a component's config twice: once when `nemo-relay plugins validate` runs, before
//! the gateway is started, and again when the component is registered. Both go through
//! [`Config::parse`], so a mistake surfaces at install time rather than as a session that quietly
//! recorded nothing.

use std::path::PathBuf;

use nemo_relay_plugin::{ConfigDiagnostic, DiagnosticLevel, Json};
use serde_json::Map;

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

/// Paths whose *contents* never enter a manifest.
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

        (config, diagnostics)
    }
}

/// Every diagnostic this config can produce is an error, since `triples` -- the one setting that was
/// understood but did not do what its name suggested -- is gone. `register` still refuses only on
/// errors rather than on any diagnostic, which is deliberate and currently unexercised: a warning
/// must not cost a session its manifest, and the next reserved field should not have to rediscover
/// that. The level stays an explicit argument for the same reason.
fn error(code: &str, field: &str, message: &str) -> ConfigDiagnostic {
    diagnostic(DiagnosticLevel::Error, code, field, message)
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
