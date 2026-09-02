//! Deriving file versions from a tool result.
//!
//! On the Relay path the tool-end scope's `data` is the agent's tool result verbatim (§7.1 of the
//! plan), which is the same structure a `PostToolUse` hook payload carries and the same one a
//! transcript stores. So this is a port of the existing recorder's `tool_results.py`, kept faithful
//! rather than redesigned: the two paths must not drift apart on exactly the details that decide
//! what a file node means.
//!
//! # Dispatch is on shape, never on tool name
//!
//! This looks wrong the first time and is deliberate. A resumed session carries results whose
//! `tool_use` block lives in a different transcript, so the tool name is simply not recoverable --
//! and those results are disproportionately reads and edits, whose lineage would otherwise be lost.
//! Matching on shape also means a new edit-shaped tool works without touching this file.
//!
//! # A fragment's hash is not the file's hash
//!
//! A `Read` with `offset`/`limit` returns a slice. Content-addressing that slice as the file would
//! invent a version the file never had, and the manifest would assert it with a straight face. Those
//! reads become identity-only nodes: *this path was read, content not established*. Relay models the
//! same idea for its own purposes in `has_partial_read_controls`.

use serde_json::Value as Json;

/// How a file was touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileMode {
    /// An input to the enclosing computation.
    Read,
    /// An output of a tool whose declared purpose was to write.
    Wrote,
}

/// A replacement whose post-image could not be established from this payload alone.
///
/// Handed on rather than dropped: `originalFile` is null on most `Edit` results, so this is the
/// common case, and the recorder can usually replay it against content the session already knows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditAttempt {
    pub old: String,
    pub new: String,
    pub replace_all: bool,
}

/// One observed file version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileObserved {
    pub path: String,
    /// `None` means the content was not established -- not that the file was empty.
    pub content: Option<Vec<u8>>,
    pub mode: FileMode,
    pub tool_use_id: Option<String>,
    /// The user edited the file by hand between the agent reading it and writing it.
    pub user_modified: bool,
    pub edit: Option<EditAttempt>,
}

/// Reconstruct post-edit content by replaying the literal replacement.
///
/// Exact, not approximate: an edit performs literal string replacement, so replaying it against the
/// pre-edit content reproduces the file byte for byte.
///
/// Returns `None` when the inputs do not permit a confident reconstruction -- including the case
/// where `old` does not occur in `original`, which means the payload disagrees with itself. The
/// caller then records the transition identity-only rather than guessing, because a wrong post-state
/// would content-address to a version that never existed.
pub fn apply_edit(
    original: Option<&str>,
    old: Option<&str>,
    new: Option<&str>,
    replace_all: bool,
) -> Option<String> {
    let (original, old, new) = (original?, old?, new?);
    if !original.contains(old) {
        return None;
    }
    Some(if replace_all {
        original.replace(old, new)
    } else {
        original.replacen(old, new, 1)
    })
}

/// Derive file versions from one tool result.
///
/// Returns `(events, attributed_path)`. `attributed_path` is set when the result fully described a
/// file transition, so a caller reconciling a filesystem delta can tell the change is already
/// accounted for and avoid reporting it again from a worse angle.
///
/// A result is often a bare string rather than an object -- every `Bash` result in the reference
/// capture is one. Those carry no structured file information and yield nothing.
pub fn file_events_from_result(
    result: &Json,
    tool_use_id: Option<&str>,
    include_partial_reads: bool,
) -> (Vec<FileObserved>, Option<String>) {
    let Some(object) = result.as_object() else {
        return (Vec::new(), None);
    };

    let file_info = object.get("file").and_then(Json::as_object);
    if let Some(file_info) = file_info.filter(|info| info.contains_key("filePath")) {
        return read_events(file_info, tool_use_id, include_partial_reads);
    }

    let is_edit = object.contains_key("filePath")
        && ["originalFile", "content", "structuredPatch"]
            .iter()
            .any(|key| object.contains_key(*key));
    if is_edit {
        return edit_events(result, tool_use_id);
    }

    (Vec::new(), None)
}

fn read_events(
    file_info: &serde_json::Map<String, Json>,
    tool_use_id: Option<&str>,
    include_partial_reads: bool,
) -> (Vec<FileObserved>, Option<String>) {
    let Some(path) = file_info
        .get("filePath")
        .and_then(Json::as_str)
        .filter(|p| !p.is_empty())
    else {
        return (Vec::new(), None);
    };

    // Two independent signals that this is a slice: an offset read starts past line 1, and a bounded
    // read returns fewer lines than the file has.
    let starts_past_the_top = match file_info.get("startLine").and_then(Json::as_i64) {
        Some(line) => line != 1,
        None => false,
    };
    let truncated = match (
        file_info.get("numLines").and_then(Json::as_i64),
        file_info.get("totalLines").and_then(Json::as_i64),
    ) {
        (Some(num), Some(total)) => num < total,
        _ => false,
    };

    if starts_past_the_top || truncated {
        if !include_partial_reads {
            return (Vec::new(), None);
        }
        return (
            vec![FileObserved {
                path: path.to_string(),
                content: None,
                mode: FileMode::Read,
                tool_use_id: tool_use_id.map(str::to_string),
                user_modified: false,
                edit: None,
            }],
            None,
        );
    }

    let content = file_info
        .get("content")
        .and_then(Json::as_str)
        .map(|text| text.as_bytes().to_vec());

    (
        vec![FileObserved {
            path: path.to_string(),
            content,
            mode: FileMode::Read,
            tool_use_id: tool_use_id.map(str::to_string),
            user_modified: false,
            edit: None,
        }],
        None,
    )
}

fn edit_events(result: &Json, tool_use_id: Option<&str>) -> (Vec<FileObserved>, Option<String>) {
    let Some(path) = result
        .get("filePath")
        .and_then(Json::as_str)
        .filter(|p| !p.is_empty())
    else {
        return (Vec::new(), None);
    };

    let original = result.get("originalFile").and_then(Json::as_str);
    let user_modified = result
        .get("userModified")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let mut events = Vec::new();

    // The pre-image, when the agent reported it. This is what makes an edit a read as well as a
    // write, and it is what lets the graph show which version was changed.
    if let Some(original) = original {
        events.push(FileObserved {
            path: path.to_string(),
            content: Some(original.as_bytes().to_vec()),
            mode: FileMode::Read,
            tool_use_id: tool_use_id.map(str::to_string),
            user_modified,
            edit: None,
        });
    }

    let replace_all = result
        .get("replaceAll")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let old = result.get("oldString").and_then(Json::as_str);
    let new = result.get("newString").and_then(Json::as_str);

    let updated = match result.get("content").and_then(Json::as_str) {
        // A write gives the post-state directly.
        Some(content) => Some(content.to_string()),
        None => apply_edit(original, old, new, replace_all),
    };

    let edit = match (&updated, old, new) {
        (None, Some(old), Some(new)) => Some(EditAttempt {
            old: old.to_string(),
            new: new.to_string(),
            replace_all,
        }),
        _ => None,
    };

    events.push(FileObserved {
        path: path.to_string(),
        content: updated.map(|text| text.into_bytes()),
        mode: FileMode::Wrote,
        tool_use_id: tool_use_id.map(str::to_string),
        user_modified,
        edit,
    });

    (events, Some(path.to_string()))
}
