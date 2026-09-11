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
    /// Refuse the replay unless `old` occurs exactly once.
    ///
    /// Claude Code's `Edit` guarantees its own uniqueness -- the tool errors when `old_string`
    /// matches more than once and `replace_all` is unset -- so a first-occurrence replacement there
    /// is the edit that happened. A Codex `apply_patch` hunk carries no such promise: the real patch
    /// that prompted this was `-two` / `+TWO` with no context lines at all, and replacing the first
    /// `two` in a file containing several would content-address a version the file never had.
    ///
    /// Only consulted for a literal replay. A line-oriented one enforces uniqueness itself, over
    /// runs of whole lines, which is the check this flag was reaching for and could not express.
    pub unique_only: bool,
    /// Replay by replacing whole lines rather than a substring.
    ///
    /// True for a patch hunk, because a patch describes lines. A substring replay lets a line
    /// terminator decide uniqueness, and that is not a distinction the patch drew: `b\n` occurs once
    /// in `b\nb` where `b` occurs twice, so the anchor looks unique and the replacement lands at the
    /// wrong end of the file. False for a Claude Code `Edit`, whose `old_string` is a literal span
    /// that need not align to lines at all.
    pub line_oriented: bool,
    /// The path whose established content this replays against, when it is not the observed path.
    ///
    /// Set only by a patch that moves a file: `*** Update File: a.txt` with `*** Move to: b.txt`
    /// applies the hunk to `a.txt`'s content and writes the result at `b.txt`. Replaying against the
    /// destination would find nothing cached and give up; recording the result at the source would
    /// say the wrong file now holds those bytes.
    pub replay_from: Option<String>,
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
    /// A path this observation emptied: the source of a move, whose content now lives at `path`.
    ///
    /// Carried so the recorder stops replaying later edits against bytes the move took away. It
    /// attests nothing on its own -- a vacated path has no content to be a version of.
    pub vacated: Option<String>,
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

/// Derive file versions from a Codex `apply_patch` document.
///
/// Codex applies edits by handing the shell a patch in its own format rather than by calling a
/// structured tool, so none of the shape dispatch above sees it. The document is delimited by
/// `*** Begin Patch` / `*** End Patch` and names each file with an action:
///
/// ```text
/// *** Begin Patch
/// *** Add File: src/new.rs
/// +fn main() {}
/// *** Update File: src/old.rs
/// @@
/// -was
/// +is
/// *** Delete File: src/gone.rs
/// *** End Patch
/// ```
///
/// **`Add File` is exactly recoverable and nothing else is.** Its body is the whole file, every line
/// prefixed with `+`, so stripping the prefixes reproduces it byte for byte. `Update File` carries
/// only hunks -- there is no pre-image in the document, so the post-image cannot be computed from it
/// alone, and those become identity-only versions. Guessing from hunks would content-address a file
/// state that may never have existed.
pub fn file_events_from_patch(patch: &str, tool_use_id: Option<&str>) -> Vec<FileObserved> {
    let mut events = Vec::new();
    let mut adding: Option<(String, Vec<String>)> = None;
    // An update in progress: its path, and the before/after halves of its hunks.
    //
    // A patch hunk is exactly the pair `apply_edit` wants. Context lines belong in both halves, so
    // the replacement is anchored rather than matching the first bare occurrence of a changed line
    // -- `-two` alone would match the word anywhere in the file, and `apply_edit` replaces
    // literally.
    let mut updating: Option<Update> = None;

    let flush = |adding: &mut Option<(String, Vec<String>)>, events: &mut Vec<FileObserved>| {
        if let Some((path, lines)) = adding.take() {
            // A trailing newline: the patch body is line-oriented, and a file written from it ends
            // with one. Joining without it would hash to a different file than the one on disk.
            let mut content = lines.join("\n");
            content.push('\n');
            events.push(FileObserved {
                path,
                content: Some(content.into_bytes()),
                mode: FileMode::Wrote,
                tool_use_id: tool_use_id.map(str::to_string),
                user_modified: false,
                vacated: None,
                edit: None,
            });
        }
    };

    for line in patch.lines() {
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            flush(&mut adding, &mut events);
            flush_update(&mut updating, &mut events, tool_use_id);
            adding = Some((path.trim().to_string(), Vec::new()));
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            flush(&mut adding, &mut events);
            flush_update(&mut updating, &mut events, tool_use_id);
            updating = Some(Update {
                path: path.trim().to_string(),
                moved_to: None,
                before: Vec::new(),
                after: Vec::new(),
            });
        } else if let Some(path) = line.strip_prefix("*** Move to: ") {
            // A rename carried inside the update it accompanies. Ignoring it made the observation
            // claim the source was rewritten in place and left the destination -- the file that
            // actually ends up holding the post-image -- out of the graph entirely.
            if let Some(update) = updating.as_mut() {
                update.moved_to = Some(path.trim().to_string());
            }
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            flush(&mut adding, &mut events);
            flush_update(&mut updating, &mut events, tool_use_id);
            events.push(identity_only(path.trim(), FileMode::Wrote, tool_use_id));
        } else if line.starts_with("*** End Patch") {
            flush(&mut adding, &mut events);
            flush_update(&mut updating, &mut events, tool_use_id);
        } else if let Some((_, lines)) = adding.as_mut()
            && let Some(added) = line.strip_prefix('+')
        {
            lines.push(added.to_string());
        } else if let Some(update) = updating.as_mut() {
            // `@@` headers carry no content. Everything else is a context, removed or added line.
            if let Some(removed) = line.strip_prefix('-') {
                update.before.push(removed.to_string());
            } else if let Some(added) = line.strip_prefix('+') {
                update.after.push(added.to_string());
            } else if let Some(context) = line.strip_prefix(' ') {
                update.before.push(context.to_string());
                update.after.push(context.to_string());
            }
        }
    }
    flush(&mut adding, &mut events);
    flush_update(&mut updating, &mut events, tool_use_id);
    events
}

/// Emit an update, as a replayable edit when its hunks permit one.
///
/// The post-image is never in the patch, so the node is identity-only unless the session already
/// established this path's content -- which `observe_file` checks, and refuses when the `old` half
/// does not occur in what it holds. That refusal is the safety: a hunk that does not match what we
/// think the file contained means our belief is stale, and inventing a version from it would
/// content-address a state the file never had.
fn flush_update(
    updating: &mut Option<Update>,
    events: &mut Vec<FileObserved>,
    tool_use_id: Option<&str>,
) {
    let Some(update) = updating.take() else {
        return;
    };
    // The post-image lands at the destination when the patch moves the file, and at the source
    // otherwise.
    let source = update.path;
    let written = update.moved_to.clone().unwrap_or_else(|| source.clone());
    // A move empties the source. Both arms carry it: a rename with no hunks at all takes the
    // identity-only one below, and that is the commonest move there is.
    let vacated = update.moved_to.as_ref().map(|_| source.clone());
    // Nothing to anchor against, so nothing to replay from.
    if update.before.is_empty() || update.before == update.after {
        events.push(FileObserved {
            vacated,
            ..identity_only(&written, FileMode::Wrote, tool_use_id)
        });
        return;
    }
    events.push(FileObserved {
        path: written,
        content: None,
        mode: FileMode::Wrote,
        tool_use_id: tool_use_id.map(str::to_string),
        user_modified: false,
        vacated,
        edit: Some(EditAttempt {
            old: as_lines(&update.before),
            new: as_lines(&update.after),
            replace_all: false,
            unique_only: true,
            line_oriented: true,
            // Only meaningful for a move; `None` when the file stayed put.
            replay_from: update.moved_to.map(|_| source),
        }),
    });
}

/// Every path a patch document names, without reconstructing anything it describes.
///
/// Kept separate from [`file_events_from_patch`] because the questions differ in both directions.
/// That parser answers *what happened to which file*, so it emits one observation per block, at the
/// **destination** when a block moves a file -- the source survives only in `replay_from`, and in
/// the identity-only arm not at all. For deciding whose policy a payload inherits, the source
/// matters exactly as much: a patch moving `secrets.pem` to `secrets.txt` carries the pem's lines in
/// its own body, and a list of destinations alone never matches `*.pem`.
///
/// It is also the cheap half. `file_events_from_patch` materializes the full content of every
/// `*** Add File`, and this runs on the thread draining the event queue, so asking it for paths
/// alone would line-split, join and allocate a whole patch body to read four strings.
pub fn paths_in_patch(patch: &str) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    for line in patch.lines() {
        let named = [
            "*** Add File: ",
            "*** Update File: ",
            "*** Move to: ",
            "*** Delete File: ",
        ]
        .iter()
        .find_map(|directive| line.strip_prefix(directive));
        if let Some(path) = named {
            let path = path.trim();
            if !path.is_empty() && !paths.iter().any(|seen| seen == path) {
                paths.push(path.to_string());
            }
        }
    }
    paths
}

/// One `*** Update File` block, accumulated as its lines arrive.
struct Update {
    path: String,
    /// The destination of an accompanying `*** Move to:`, when there is one.
    moved_to: Option<String>,
    before: Vec<String>,
    after: Vec<String>,
}

/// Join hunk lines back into text, restoring the terminator each one had in the file.
///
/// `join("\n")` alone drops the last line's newline, and the loss is invisible until a deletion:
/// removing `b` from `a\nb\nc\n` becomes the replacement `b` -> `` and yields `a\n\nc\n` -- a file
/// carrying a blank line the patch never created, content-addressed and signed as though it were
/// what the tool produced. The uniqueness guard does not catch it, because `b` really does occur
/// once.
///
/// Restoring the terminator is necessary but **not sufficient**: it narrows what `old` matches,
/// which can turn an ambiguous anchor into an apparently unique one. Against the unterminated file
/// `b\nb`, the hunk `-b` / `+c` searches for `b\n`, matches the first line exactly once and passes
/// the uniqueness guard. So hunk replay is line-oriented rather than substring replacement; see
/// [`apply_line_edit`].
fn as_lines(lines: &[String]) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let mut text = lines.join("\n");
    text.push('\n');
    text
}

/// Why a line-oriented replay was refused.
///
/// Distinguished rather than collapsed into `None`, because coverage should say which kind of
/// not-knowing this was: a stale belief about the file, an anchor that could mean two things, or a
/// transformation whose exact bytes the hunk does not determine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayRefusal {
    /// The removed lines occur more than once, so which run the hunk meant is not established.
    Ambiguous,
    /// They do not occur at all: what we hold is not what the patch was written against.
    NotFound,
    /// They reach the end of a file with no final newline, where the result's exact bytes are not
    /// recoverable -- whether the preceding line keeps its terminator depends on what the tool did,
    /// not on what the hunk says.
    UnterminatedAtEof,
    /// The hunk introduces lines into a file that does not end its lines with a bare `\n`.
    ///
    /// A new line's terminator is in neither the patch nor the pre-image: it is new, so there are no
    /// bytes to copy and nothing establishes what the tool wrote. In an all-`\n` file `\n` is the
    /// only candidate. In a CRLF file it is a coin toss that decides the content CID, and the
    /// installed `apply_patch` was observed writing `\n` for the changed line while leaving every
    /// untouched line `\r\n` -- a shape no reasonable guess would have produced.
    TerminatorsNotEstablished,
}

/// Split text into `(content, terminator)`, keeping the exact bytes that ended each line.
///
/// `str::lines` cannot be used for reassembly: it strips `\r` as well as `\n` and reports nothing
/// about which it removed, so rebuilding with `\n` silently rewrites every line of a CRLF file. The
/// terminator has to travel with its line for untouched lines to survive byte-exact.
fn split_lines(text: &str) -> Vec<(&str, &str)> {
    let mut lines = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let Some(at) = rest.find('\n') else {
            // A final line with no terminator at all.
            lines.push((rest, ""));
            break;
        };
        let (line, tail) = rest.split_at(at + 1);
        let body = &line[..line.len() - 1];
        lines.push(match body.strip_suffix('\r') {
            Some(content) => (content, &line[line.len() - 2..]),
            None => (body, &line[line.len() - 1..]),
        });
        rest = tail;
    }
    lines
}

/// Replay a patch hunk as a replacement of whole lines.
///
/// A patch speaks in lines, so the anchor has to be matched in lines. Matching it as a substring
/// lets a terminator decide uniqueness -- `b\n` occurs once in `b\nb` where `b` occurs twice -- and
/// the uniqueness guard then approves a replacement at the wrong end of the file. Comparing runs of
/// whole lines makes the guard operate on the unit the patch actually describes.
///
/// Requires exactly one match. `replace_all` has no meaning here: a hunk describes one edit at one
/// place, and a hunk whose anchor appears twice has not said which.
///
/// **Untouched lines are copied, never rebuilt.** Matching happens on line *content*, with
/// terminators stripped, because that is what the patch names -- but every line outside the matched
/// window is written back byte for byte, terminator included. Rebuilding them with `\n` rewrote every
/// line of a CRLF file: `a\r\nb\r\nc\r\n` with `-b` / `+B` came back as `a\nB\nc\n`, three lines
/// changed where the patch named one, and the content CID attested it as recovered.
pub fn apply_line_edit(previous: &str, old: &str, new: &str) -> Result<String, ReplayRefusal> {
    let held = split_lines(previous);
    let contents: Vec<&str> = held.iter().map(|(content, _)| *content).collect();
    let removed: Vec<&str> = old.lines().collect();
    let added: Vec<&str> = new.lines().collect();

    if removed.is_empty() || removed.len() > held.len() {
        return Err(ReplayRefusal::NotFound);
    }

    let mut found: Option<usize> = None;
    for start in 0..=(held.len() - removed.len()) {
        if contents[start..start + removed.len()] == removed[..] {
            if found.is_some() {
                return Err(ReplayRefusal::Ambiguous);
            }
            found = Some(start);
        }
    }
    let start = found.ok_or(ReplayRefusal::NotFound)?;
    let end = start + removed.len();

    // The final line carries no terminator, so a hunk touching it cannot say whether the line before
    // it keeps one. Deleting `b` from `a\nb` yields `a\n` or `a` depending on the tool, and the patch
    // is silent on which.
    let terminated = held.last().is_some_and(|(_, end)| !end.is_empty());
    if !terminated && end == held.len() {
        return Err(ReplayRefusal::UnterminatedAtEof);
    }

    // A pure deletion introduces no line, so there is no terminator to invent and every surviving
    // byte is copied: exact for any line-ending style. Anything that *adds* a line has to write a
    // terminator the patch never stated, and only an all-`\n` file makes `\n` the sole candidate.
    if !added.is_empty() && held.iter().any(|(_, end)| *end == "\r\n") {
        return Err(ReplayRefusal::TerminatorsNotEstablished);
    }

    let mut text = String::with_capacity(previous.len() + new.len());
    for (content, end) in &held[..start] {
        text.push_str(content);
        text.push_str(end);
    }
    for line in &added {
        text.push_str(line);
        text.push('\n');
    }
    for (content, end) in &held[end..] {
        text.push_str(content);
        text.push_str(end);
    }
    Ok(text)
}

/// A file we know was touched and whose content we could not establish.
fn identity_only(path: &str, mode: FileMode, tool_use_id: Option<&str>) -> FileObserved {
    FileObserved {
        path: path.to_string(),
        content: None,
        mode,
        tool_use_id: tool_use_id.map(str::to_string),
        user_modified: false,
        vacated: None,
        edit: None,
    }
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

    // Three independent signals that this is a slice: an offset read starts past line 1, a bounded
    // read returns fewer lines than the file has, and the reader says outright that it cut the
    // content short.
    let starts_past_the_top = match file_info.get("startLine").and_then(Json::as_i64) {
        Some(line) => line != 1,
        None => false,
    };
    let fewer_lines_than_the_file = match (
        file_info.get("numLines").and_then(Json::as_i64),
        file_info.get("totalLines").and_then(Json::as_i64),
    ) {
        (Some(num), Some(total)) => num < total,
        _ => false,
    };
    // A one-line file cut mid-line reports `numLines == totalLines == 1` and is a fragment anyway.
    // Line counts cannot express a cut *within* a line, so without this a 200 KB single-line file
    // read back as 21 KB is recorded as that file's complete content, under a CID matching nothing
    // on disk. Observed live: `truncatedByTokenCap: true` with `numLines == totalLines == 1`.
    let cut_short = file_info
        .get("truncatedByTokenCap")
        .and_then(Json::as_bool)
        .unwrap_or(false);

    if starts_past_the_top || fewer_lines_than_the_file || cut_short {
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
                vacated: None,
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
            vacated: None,
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
            vacated: None,
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
            // The tool guarantees its own uniqueness; see `EditAttempt::unique_only`.
            unique_only: false,
            // `old_string` is a literal span that need not align to line boundaries.
            line_oriented: false,
            // An `Edit` acts in place.
            replay_from: None,
        }),
        _ => None,
    };

    events.push(FileObserved {
        path: path.to_string(),
        content: updated.map(|text| text.into_bytes()),
        mode: FileMode::Wrote,
        tool_use_id: tool_use_id.map(str::to_string),
        user_modified,
        vacated: None,
        edit,
    });

    (events, Some(path.to_string()))
}
