//! Does Relay actually load this thing?
//!
//! Everything else in `tests/` exercises our logic by calling it directly. That proves the recorder
//! is right and proves nothing about the boundary it lives behind -- a plugin can be perfectly
//! correct and still fail to load, because its `[plugin] id` disagrees with `plugin_kind()`, or its
//! ABI level is unsupported, or its entry symbol was renamed.
//!
//! This test closes that gap without needing the `nemo-relay` binary installed. It builds the
//! cdylib, materializes a manifest with the artifact's real digest, and activates it through
//! `PluginHostActivation` -- the same dynamic-plugin host path the gateway runs. If the C ABI
//! contract is broken, it breaks here.
//!
//! It lives in its own crate because it must: Relay's core crate needs `sha2 ^0.11`, and `integrity`
//! pins `=0.11.0-rc.5` transitively through `iroh-base`. Semver excludes pre-releases from `^0.11`,
//! so the plugin's dependency graph and this test's cannot be the same graph. See `Cargo.toml`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationSpec, DynamicPluginKind, PluginHostActivation,
};
use nemo_relay::plugin::{PluginConfig, list_plugin_kinds};
use serde_json::{Map, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::sync::Mutex as AsyncMutex;

/// Activation is process-global, so the tests in this file must not overlap.
static HOST_LOCK: AsyncMutex<()> = AsyncMutex::const_new(());

const PLUGIN_ID: &str = "eqty.lineage";

#[tokio::test]
async fn the_built_cdylib_loads_and_registers_its_kind() {
    let _guard = HOST_LOCK.lock().await;
    let library = build_cdylib();
    let manifest_dir = TempDir::new().expect("manifest directory");
    let manifest = write_manifest(manifest_dir.path(), library);
    let manifests = TempDir::new().expect("output directory");

    let config = config(json!({
        "manifest_dir": manifests.path().to_string_lossy(),
    }));

    let (activation, report) = PluginHostActivation::activate(
        PluginConfig::default(),
        [DynamicPluginActivationSpec {
            plugin_id: PLUGIN_ID.into(),
            kind: DynamicPluginKind::RustDynamic,
            manifest_ref: manifest.to_string_lossy().into_owned(),
            environment_ref: None,
            config,
        }],
    )
    .await
    .expect("the native manifest should activate");

    assert!(
        report.diagnostics.is_empty(),
        "activation complained: {report:?}"
    );
    assert!(
        list_plugin_kinds().iter().any(|kind| kind == PLUGIN_ID),
        "Relay should know the kind `{PLUGIN_ID}` after activation: {:?}",
        list_plugin_kinds()
    );

    activation
        .clear()
        .expect("callbacks should clear before the library unloads");
}

#[tokio::test]
async fn a_bad_config_is_refused_rather_than_recorded_badly() {
    // `validate` runs inside the loaded library, across the ABI boundary. This asserts the wiring,
    // not the rule -- `tests/config.rs` owns the rule. A component that activated with
    // `max_content_bytes: 0` would record a session whose every file was too large to store.
    let _guard = HOST_LOCK.lock().await;
    let library = build_cdylib();
    let manifest_dir = TempDir::new().expect("manifest directory");
    let manifest = write_manifest(manifest_dir.path(), library);

    let outcome = PluginHostActivation::activate(
        PluginConfig::default(),
        [DynamicPluginActivationSpec {
            plugin_id: PLUGIN_ID.into(),
            kind: DynamicPluginKind::RustDynamic,
            manifest_ref: manifest.to_string_lossy().into_owned(),
            environment_ref: None,
            config: config(json!({ "max_content_bytes": 0 })),
        }],
    )
    .await;

    match outcome {
        Err(_) => {}
        Ok((activation, report)) => {
            assert!(
                !report.diagnostics.is_empty(),
                "a zero content ceiling should not activate silently: {report:?}"
            );
            activation
                .clear()
                .expect("clear after a diagnosed activation");
        }
    }
}

fn config(value: serde_json::Value) -> Map<String, serde_json::Value> {
    value.as_object().cloned().expect("config is an object")
}

/// Build the plugin's cdylib **once**, into the plugin's **own** target directory.
///
/// `just test-rust` runs the plugin's `cargo test` immediately before this crate's, so
/// `../target/debug` already holds every dependency this build needs and only the cdylib has to be
/// linked. A scratch `TempDir` instead recompiles `integrity`, iroh and ssi from nothing per test,
/// which is slow enough to matter and large enough to exhaust the runner's disk. A new path every
/// run also means no cache can ever restore into it.
///
/// **`--target-dir` is explicit, and must stay that way.** Cargo locks per target directory, and
/// this runs *inside* `cargo test`. With `CARGO_TARGET_DIR` set -- which `just linux-check` does --
/// both would point at one directory and the nested build would wait on a lock its own parent
/// holds. The flag beats the environment variable.
fn build_cdylib() -> &'static Path {
    static BUILT: OnceLock<PathBuf> = OnceLock::new();
    BUILT.get_or_init(|| {
        let plugin = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let output = Command::new(env!("CARGO"))
            .args(["build", "--lib", "--message-format=json-render-diagnostics"])
            .arg("--manifest-path")
            .arg(plugin.join("Cargo.toml"))
            .arg("--target-dir")
            .arg(plugin.join("target"))
            .output()
            .expect("cargo build should start");
        assert!(
            output.status.success(),
            "cargo build should produce the native library: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Cargo names the artifact per platform and puts it wherever the profile says, so it is
        // asked rather than guessed -- the same reason `just nemo-relay-package` reads this stream.
        let library = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|message| message["reason"] == "compiler-artifact")
            .filter(|message| message["target"]["name"] == "eqty_lineage_nemo_relay")
            .flat_map(|message| {
                message["filenames"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
            })
            .filter_map(|name| name.as_str().map(PathBuf::from))
            .find(|name| {
                name.extension()
                    .is_some_and(|kind| kind == "dylib" || kind == "so" || kind == "dll")
            })
            .expect("cargo should report a cdylib for eqty_lineage_nemo_relay");

        assert!(library.exists(), "expected {}", library.display());
        library
    })
}

/// Write a manifest carrying the artifact's real digest.
///
/// Relay always verifies `[integrity] sha256` against the library it is about to load, regardless of
/// attestation policy. Computing it here rather than hardcoding it is the same thing release CI must
/// do -- a stale digest is an install Relay refuses.
fn write_manifest(directory: &Path, library: &Path) -> PathBuf {
    let digest = digest(library);
    let quoted = format!("{:?}", library.to_string_lossy());
    let manifest = directory.join("relay-plugin.toml");
    std::fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1

[plugin]
id = "{PLUGIN_ID}"
kind = "rust_dynamic"

[compat]
relay = ">=0.8.0,<1.0"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[integrity]
sha256 = "{digest}"

[load]
library = {quoted}
symbol = "nemo_relay_register_plugin"
"#,
        ),
    )
    .expect("manifest should write");
    manifest
}

fn digest(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("library should read");
    // sha2 0.11 returns a `hybrid-array` type, which does not implement `LowerHex` the way 0.10's
    // `GenericArray` did. Relay compares this string against its own digest, so the encoding has to
    // be plain lowercase hex with no separators.
    let mut hex = String::from("sha256:");
    for byte in Sha256::digest(&bytes) {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}
