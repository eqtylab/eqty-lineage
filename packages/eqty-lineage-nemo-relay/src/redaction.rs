//! Deciding whose bytes may be stored.
//!
//! Redaction is a gate, not a filter. A denied file still becomes a node in the graph carrying its
//! path and its true content CID -- computed from the bytes *before* this module sees them -- and
//! only the bytes are withheld. Provenance does not require publication, and dropping the node
//! instead would make the manifest silently incomplete: a reader could not tell "the agent never
//! touched `.env`" from "the agent read `.env` and we hid the whole fact".

/// What may be done with a file's content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Store the bytes.
    Store,
    /// Record the node, withhold the bytes: the path matched a deny pattern.
    Denied,
    /// Record the node, withhold the bytes: the content is larger than the ceiling.
    TooLarge,
}

/// Match a path against a glob supporting `*` (any run, including separators) and `?`.
///
/// Separators are deliberately not special. A pattern like `*/.ssh/*` is meant to catch `.ssh`
/// anywhere in a tree, and a matcher that stopped `*` at `/` would quietly fail to protect exactly
/// the nested paths the pattern exists for.
pub fn glob_match(pattern: &str, value: &str) -> bool {
    let (pattern, value): (Vec<char>, Vec<char>) =
        (pattern.chars().collect(), value.chars().collect());
    // Iterative backtracking rather than recursion: a pathological pattern against a long path
    // should not be able to blow the stack of the process we are observing.
    let (mut p, mut v) = (0usize, 0usize);
    let (mut star, mut resume) = (None, 0usize);

    while v < value.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == value[v]) {
            p += 1;
            v += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = Some(p);
            resume = v;
            p += 1;
        } else if let Some(star_at) = star {
            p = star_at + 1;
            resume += 1;
            v = resume;
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|character| *character == '*')
}

/// The configured content policy, as applied to one file at a time.
#[derive(Debug, Clone)]
pub struct Policy {
    deny_globs: Vec<String>,
    max_content_bytes: u64,
}

impl Policy {
    /// Build a policy from validated configuration.
    pub fn new(deny_globs: Vec<String>, max_content_bytes: u64) -> Self {
        Self {
            deny_globs,
            max_content_bytes,
        }
    }

    /// Decide what may be stored for `path`, given content of `len` bytes.
    ///
    /// The deny list is checked against the full path and against its final component, so a pattern
    /// written as `.env*` catches `/srv/app/.env.production` as well as a bare `.env`.
    pub fn decide(&self, path: &str, len: usize) -> Disposition {
        let basename = path.rsplit('/').next().unwrap_or(path);
        if self
            .deny_globs
            .iter()
            .any(|glob| glob_match(glob, path) || glob_match(glob, basename))
        {
            return Disposition::Denied;
        }
        if len as u64 > self.max_content_bytes {
            return Disposition::TooLarge;
        }
        Disposition::Store
    }
}
