//! Component configuration.
//!
//! These run at `nemo-relay plugins validate` time, before a gateway exists. A config mistake that
//! reaches a live session shows up as a manifest that is quietly missing things, so every field is
//! refused loudly rather than coerced.

use eqty_lineage_nemo_relay::Config;
use serde_json::{Map, json};

fn parse(value: serde_json::Value) -> (Config, Vec<String>) {
    let map: Map<String, serde_json::Value> = value.as_object().cloned().unwrap_or_default();
    let (config, diagnostics) = Config::parse(&map);
    (config, diagnostics.into_iter().map(|d| d.code).collect())
}

#[test]
fn an_empty_config_is_valid_and_has_defaults() {
    let (config, codes) = parse(json!({}));
    assert!(
        codes.is_empty(),
        "an omitted config is not a broken one: {codes:?}"
    );
    assert_eq!(config.manifest_dir.to_str(), Some(".eqty/manifests"));
    assert_eq!(config.max_content_bytes, 104_857_600);
    assert!(config.deny_globs.iter().any(|glob| glob == ".env*"));
}

#[test]
fn a_key_this_config_does_not_know_is_ignored_rather_than_refused() {
    // `triples` used to live here, reserved and warned about, and is now gone -- so a config still
    // carrying it must not take a session down with it. This side is deliberately tolerant while
    // `config.schema.json` sets `additionalProperties: false`, which is the asymmetry that matters:
    // `plugins validate` reports the stale key against the schema, and the recorder still records.
    let (config, codes) = parse(json!({ "triples": true, "manifest_dir": "out" }));
    assert!(
        codes.is_empty(),
        "an unknown key is not a broken config: {codes:?}"
    );
    assert_eq!(config.manifest_dir.to_str(), Some("out"));
}

#[test]
fn every_field_can_be_set() {
    let (config, codes) = parse(json!({
        "manifest_dir": "out/manifests",
        "deny_globs": ["*.key"],
        "max_content_bytes": 2048,
    }));
    assert!(codes.is_empty(), "{codes:?}");
    assert_eq!(config.manifest_dir.to_str(), Some("out/manifests"));
    assert_eq!(config.deny_globs, vec!["*.key".to_string()]);
    assert_eq!(config.max_content_bytes, 2048);
}

#[test]
fn every_problem_is_reported_at_once() {
    // `nemo-relay plugins validate` prints what it is given. A user with three fields wrong should
    // learn that once, not across three reinstalls.
    let (_, codes) = parse(json!({
        "manifest_dir": "",
        "deny_globs": "*.key",
        "max_content_bytes": 0,
    }));
    assert_eq!(codes.len(), 3, "expected all three refused: {codes:?}");
    assert!(codes.contains(&"manifest_dir.invalid".to_string()));
    assert!(codes.contains(&"deny_globs.invalid".to_string()));
    assert!(codes.contains(&"max_content_bytes.invalid".to_string()));
}

#[test]
fn a_rejected_field_keeps_its_default_rather_than_a_broken_value() {
    // Registration refuses on the first diagnostic, so this only matters if that check is ever
    // relaxed -- at which point a half-applied config must not silently write manifests to "".
    let (config, codes) = parse(json!({"manifest_dir": ""}));
    assert!(!codes.is_empty());
    assert_eq!(config.manifest_dir.to_str(), Some(".eqty/manifests"));
}

#[test]
fn an_empty_deny_list_is_accepted_and_says_so() {
    // Legal, and the one setting whose failure mode is a leak rather than a crash: an empty array
    // turns every default off at once, and a templating bug produces it as readily as an operator
    // does. It must still parse -- a diagnostic that cost the session its manifest would be worse
    // than the thing it warns about -- so the warning is the whole protection.
    let (config, codes) = parse(json!({ "deny_globs": [] }));
    assert!(
        config.deny_globs.is_empty(),
        "the operator's list is honoured, not silently replaced: {:?}",
        config.deny_globs
    );
    assert!(
        codes.contains(&"deny_globs.empty".to_string()),
        "an empty deny list is not a silent one: {codes:?}"
    );
}

#[test]
fn deny_globs_must_be_strings() {
    let (config, codes) = parse(json!({"deny_globs": [".env*", 7]}));
    assert_eq!(codes, vec!["deny_globs.invalid".to_string()]);
    assert!(
        config.deny_globs.iter().any(|glob| glob == "*.pem"),
        "a refused deny list must fall back to the protective default, never to empty"
    );
}
