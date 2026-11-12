//! The shared folder: `/share` on the agent's machine (`share::share_dir`), where
//! the agent and the operator work on the same files in place.
//!
//! The operator has it in the share tab (Alt+S, `tabs/share_view.rs`): he views,
//! edits (in his own editor), creates, adds files from his machine and saves
//! copies to it. The agent has `shared_list`, `shared_read`, `shared_write` and
//! `shared_edit`. Neither copies a file to the other to work on it: a new
//! version is written over the old, where the other already looks.
//!
//! # Who saw what
//!
//! A file's stamp is its size and modification time (`find -printf
//! '%s:%T@'`). The agent's ledger keeps the stamp of each file as it last read or
//! wrote it, and the text when it is small, so:
//!
//! - `shared_list` says which files changed since the agent last read them;
//! - a write over a file the agent has not read, or that changed since it read it,
//!   is refused and says what changed: nothing of the operator's is written over
//!   unseen;
//! - `shared_edit` replaces one piece of the file as it is now, so the operator's
//!   other changes stay, and says what he changed since the agent's last read.
//!
//! What the operator does in the tab reaches the agent with his next message
//! (`take_note`, beside `share::arrival_note`), an edit as a diff: the agent knows
//! the file again without reading it.
//!
//! Every write goes through a temp file beside the target, checked by
//! SHA-256 and renamed over it only if the file is still what the writer saw
//! (`Expect`), so two writers never interleave.

use crate::share::{CHUNK_BYTES, RECEIVE_CHUNK_BYTES, Ran, STEP_TIMEOUT_SECS, human_size, step};
use crate::util::shell_quote;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use std::collections::HashMap;
use std::sync::Mutex;

/// A guest command and its timeout (`share::run_on_machine`, or a test's).
pub type Run<'a> = &'a mut dyn FnMut(&str, u64) -> Result<Ran, String>;

/// Folders this deep are listed; hidden ones, `node_modules` and
/// `__pycache__` are not.
const DEPTH: usize = 8;
const MAX_ENTRIES: usize = 3000;
/// The most a file read whole may be (an edit, a tool read).
pub const MAX_WHOLE_BYTES: u64 = 8 * 1024 * 1024;
/// The ledger keeps a file's text when it is no bigger than this, for diffs.
const KEEP_TEXT: usize = 256 * 1024;
/// A diff in a note or a tool result stops after this many lines.
const DIFF_LINES: usize = 80;
/// A tool read returns at most this many characters at once.
const READ_CHARS: usize = 60_000;

pub const CONTEXT: &str = "\n\nTHE SHARED FOLDER\n\
/share on your machine is the folder you and the operator work on together. It is mirrored both ways with ~/s0ma-share \
on his machine, within seconds, and he has it open in his share tab (Alt+S): he sees every file there as it is \
now, reads it, edits it in his own editor and adds files of his own. So a file meant \
for him to read, review or edit goes in /share, and a new version is written over the old one in place: do not \
send_to_operator a file to show it to him, and do not make report-v2 copies. Work on shared files with \
shared_list, shared_read, shared_edit and shared_write: they keep track of what you have read, refuse to write \
over a change of his you have not seen, and say what he changed. His edits from the tab come with his next \
message as a diff. A file in /share is already on his machine, so send_to_operator only what is elsewhere.";

// ------------------------------------------------------------------ paths

/// `path` as a path inside `dir`: `/share/notes/a.md`, `notes/a.md` and
/// `./notes/a.md` are all `notes/a.md`. Nothing outside it, no `..`.
pub fn relative(dir: &str, path: &str) -> Result<String, String> {
    let path = path.trim();
    let inside_of = |root: &str| path.strip_prefix(root).filter(|rest| rest.is_empty() || rest.starts_with('/')).map(|rest| rest.trim_start_matches('/'));
    // `/share/…` is the folder by the tools' own name for it, wherever it is
    // (host mode keeps it in the operator's home).
    let inside = inside_of(dir).or_else(|| (dir != crate::share::DEFAULT_SHARE_DIR).then(|| inside_of(crate::share::DEFAULT_SHARE_DIR)).flatten()).unwrap_or(path);
    if inside.starts_with('/') || inside.starts_with('~') {
        return Err(format!("{path} is not in {dir}; give a path inside it"));
    }
    if inside.chars().any(char::is_control) {
        return Err("the path has a control character in it".into());
    }
    let mut parts = Vec::new();
    for part in inside.split('/') {
        match part {
            "" | "." => {}
            ".." => return Err(format!("{path}: no `..`; give a path inside {dir}")),
            part => parts.push(part),
        }
    }
    if parts.is_empty() {
        return Err(format!("give a file in {dir}, not the folder itself"));
    }
    Ok(parts.join("/"))
}

/// `notes/a.md` as the agent and the operator see it: `/share/notes/a.md`.
pub fn shown(dir: &str, relative: &str) -> String {
    format!("{dir}/{relative}")
}

/// The path, decoded into `$p` on the far side: it never reaches the shell
/// as text.
fn path_var(relative: &str) -> String {
    // The `x` keeps a name's trailing newlines, which `$(…)` would drop.
    format!("p=$(printf %s '{}' | base64 -d; printf x); p=${{p%x}}", STANDARD.encode(relative.as_bytes()))
}

/// A file's version: size, modification time, inode and change time. The
/// inode changes with every write here (a rename over it); the change time
/// catches a same-size change on a filesystem with coarse modification
/// times.
const STAMP: &str = "-maxdepth 0 -printf '%s:%T@:%i:%C@'";

/// Writes and deletes check a stamp and act under this lock, so two of them
/// never both pass the check. Without `flock` (not in util-linux), they go
/// unlocked.
const LOCK: &str = "if command -v flock >/dev/null 2>&1; then exec 9>\"$PWD/.s0ma-lock\" && flock -w 20 9; fi";

// ------------------------------------------------------------------ listing

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// Inside the folder: `notes/a.md`.
    pub path: String,
    pub dir: bool,
    pub bytes: u64,
    /// Seconds since the epoch.
    pub modified: f64,
    pub stamp: String,
}

/// Everything in `dir`, folders and files, in path order.
pub fn list(dir: &str, run: Run) -> Result<Vec<Entry>, String> {
    let command = format!(
        "d={dir}\n\
         [ -d \"$d\" ] || {{ echo \"$d does not exist on s0ma's machine\" >&2; exit 3; }}\n\
         cd \"$d\" && find . -mindepth 1 -maxdepth {DEPTH} \\( -name '.*' -o -name node_modules -o -name __pycache__ \\) -prune \
         -o \\( -type f -o -type d \\) -printf '%y\\t%s\\t%T@\\t%i\\t%C@\\t%P\\0' | head -z -n {MAX_ENTRIES}",
        dir = shell_quote(dir)
    );
    Ok(parse_list(&step(run, &command)?.stdout))
}

fn parse_list(out: &str) -> Vec<Entry> {
    let mut entries: Vec<Entry> = out
        .split('\0')
        .filter_map(|record| {
            let mut fields = record.splitn(6, '\t');
            let (kind, bytes, modified, inode, changed, path) =
                (fields.next()?, fields.next()?, fields.next()?, fields.next()?, fields.next()?, fields.next()?);
            if path.is_empty() || !matches!(kind, "f" | "d") {
                return None;
            }
            Some(Entry {
                path: path.to_string(),
                dir: kind == "d",
                bytes: bytes.parse().ok()?,
                modified: modified.parse().ok()?,
                stamp: format!("{bytes}:{modified}:{inode}:{changed}"),
            })
        })
        .collect();
    entries.sort_by(|a, b| a.path.split('/').cmp(b.path.split('/')));
    entries
}

// ------------------------------------------------------------------ reading

pub struct Read {
    pub bytes: Vec<u8>,
    pub stamp: String,
    /// The whole file's size (`bytes` may be its beginning only).
    pub size: u64,
}

/// The file at `relative`: the first `limit` bytes when `whole` is false, or
/// all of it, up to `limit`, when it is true. A file changing as it is read
/// is read again once.
pub fn read(dir: &str, relative: &str, limit: u64, whole: bool, run: Run) -> Result<Read, String> {
    match read_once(dir, relative, limit, whole, run) {
        Err(error) if error == CHANGING => read_once(dir, relative, limit, whole, run),
        other => other,
    }
}

const CHANGING: &str = "it changed as it was read; try again";

fn read_once(dir: &str, relative: &str, limit: u64, whole: bool, run: Run) -> Result<Read, String> {
    let head = limit.min(RECEIVE_CHUNK_BYTES as u64);
    // Small (or only its beginning wanted): in one command, its stamp before
    // and after.
    let command = format!(
        "cd {dir} || exit 3\n{path}\n\
         [ -e \"./$p\" ] || {{ echo \"no such file: $PWD/$p\" >&2; exit 4; }}\n\
         [ -f \"./$p\" ] || {{ echo \"$PWD/$p is not a file\" >&2; exit 5; }}\n\
         s=$(find \"./$p\" {STAMP}); size=${{s%%:*}}\n\
         if {whole} && [ \"$size\" -gt {limit} ]; then echo \"$PWD/$p is $(numfmt --to=iec \"$size\" 2>/dev/null || echo \"$size bytes\"), over the {max} read whole\" >&2; exit 7; fi\n\
         printf '%s\\n' \"$s\"\n\
         if {whole} && [ \"$size\" -gt {head} ]; then t=$(mktemp) && cp -- \"./$p\" \"$t\" || exit 8; printf 'COPY %s\\n' \"$t\"; \
         else head -c {head} -- \"./$p\" | base64 -w0; printf '\\n'; fi\n\
         find \"./$p\" {STAMP} 2>/dev/null || printf gone",
        dir = shell_quote(dir),
        path = path_var(relative),
        whole = if whole { "true" } else { "false" },
        max = human_size(limit),
    );
    let ran = step(run, &command)?;
    let mut lines = ran.stdout.splitn(3, '\n');
    let (Some(before), Some(body), Some(after)) = (lines.next(), lines.next(), lines.next()) else {
        return Err(format!("s0ma's machine described the file unreadably: {:?}", clip_debug(&ran.stdout)));
    };
    let size: u64 = before.split(':').next().and_then(|size| size.parse().ok()).ok_or("the file's size was unreadable")?;
    if let Some(temp) = body.strip_prefix("COPY ") {
        let temp = temp.to_string();
        let result = pull(run, &temp, size);
        let _ = run(&format!("rm -f {}", shell_quote(&temp)), STEP_TIMEOUT_SECS);
        if before != after.trim() {
            return Err(CHANGING.into());
        }
        return result.map(|bytes| Read { bytes, stamp: before.to_string(), size });
    }
    if before != after.trim() {
        return Err(CHANGING.into());
    }
    let bytes = STANDARD.decode(body.trim()).map_err(|error| format!("the file arrived unreadable: {error}"))?;
    Ok(Read { bytes, stamp: before.to_string(), size })
}

fn clip_debug(text: &str) -> String {
    text.chars().take(200).collect()
}

/// A snapshot on the agent's machine, piece by piece.
fn pull(run: Run, temp: &str, size: u64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(size as usize);
    while (bytes.len() as u64) < size {
        let offset = bytes.len() as u64;
        let length = (size - offset).min(RECEIVE_CHUNK_BYTES as u64);
        let ran = step(
            run,
            &format!("dd if={} iflag=skip_bytes,count_bytes skip={offset} count={length} status=none | base64 -w0", shell_quote(temp)),
        )?;
        let piece = STANDARD.decode(ran.stdout.trim()).map_err(|error| format!("a piece arrived unreadable: {error}"))?;
        if piece.len() as u64 != length {
            return Err(format!("a piece arrived short: {} of {length} bytes at {offset}", piece.len()));
        }
        bytes.extend_from_slice(&piece);
    }
    Ok(bytes)
}

/// The file's stamp now, or `None` when it is gone (the editor watches an
/// open file with this).
pub fn stamp_of(dir: &str, relative: &str, run: Run) -> Result<Option<String>, String> {
    let command = format!(
        "cd {dir} || exit 3\n{path}\n[ -f \"./$p\" ] || {{ echo S0MA-GONE; exit 0; }}\nfind \"./$p\" {STAMP}",
        dir = shell_quote(dir),
        path = path_var(relative),
    );
    let ran = step(run, &command)?;
    let out = ran.stdout.trim();
    Ok((out != "S0MA-GONE" && !out.is_empty()).then(|| out.to_string()))
}

// ------------------------------------------------------------------ writing

/// What the file must still be for a write to go over it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expect {
    /// Not there: a new file.
    New,
    /// As it was when read: this stamp.
    Stamp(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum WriteError {
    /// It is not what the writer saw any more (or, for a new file, it is
    /// there now).
    Changed,
    Failed(String),
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::Changed => f.write_str("it changed since it was read"),
            WriteError::Failed(error) => f.write_str(error),
        }
    }
}

const CONFLICT: &str = "S0MA-SHARED-CONFLICT";

/// Write `bytes` to `relative`, making its folders, if the file is still as
/// `expect` says. An existing file keeps its permissions. Its new stamp.
pub fn write(dir: &str, relative: &str, bytes: &[u8], expect: &Expect, run: Run) -> Result<String, WriteError> {
    let failed = WriteError::Failed;
    let prepare = format!(
        "cd {dir} || exit 3\n{path}\n\
         [ -d \"./$p\" ] && {{ echo \"$PWD/$p is a folder\" >&2; exit 4; }}\n\
         case $p in */*) parent=\"./${{p%/*}}\";; *) parent=.;; esac\n\
         mkdir -p -- \"$parent\" || exit 5\n\
         mktemp \"$parent/.s0ma-write.XXXXXXXX\"",
        dir = shell_quote(dir),
        path = path_var(relative),
    );
    let temp = step(run, &prepare).map_err(failed)?.stdout.trim().to_string();
    if temp.is_empty() {
        return Err(WriteError::Failed("s0ma's machine gave no temp file to write into".into()));
    }
    let digest = format!("{:x}", Sha256::digest(bytes));
    let check = match expect {
        Expect::New => "[ -z \"$cur\" ]".to_string(),
        Expect::Stamp(stamp) => format!("[ \"$cur\" = {} ]", shell_quote(stamp)),
    };
    let result = (|| {
        for piece in bytes.chunks(CHUNK_BYTES) {
            step(run, &format!("cd {} && printf %s '{}' | base64 -d >> {}", shell_quote(dir), STANDARD.encode(piece), shell_quote(&temp)))
                .map_err(failed)?;
        }
        let finish = format!(
            "cd {dir} || exit 3\n{path}\nt={temp}\n\
             got=$(sha256sum < \"$t\" | cut -d' ' -f1)\n\
             [ \"$got\" = {digest} ] || {{ echo \"arrived damaged (sha256 $got)\" >&2; exit 6; }}\n\
             {LOCK}\n\
             cur=; [ -e \"./$p\" ] && cur=$(find \"./$p\" {STAMP})\n\
             {check} || {{ echo {CONFLICT} >&2; exit 9; }}\n\
             if [ -e \"./$p\" ]; then chmod --reference=\"./$p\" \"$t\" 2>/dev/null || chmod 644 \"$t\"; else chmod 644 \"$t\"; fi\n\
             mv -f -- \"$t\" \"./$p\" || exit 8\n\
             find \"./$p\" {STAMP}",
            dir = shell_quote(dir),
            path = path_var(relative),
            temp = shell_quote(&temp),
            digest = shell_quote(&digest),
        );
        match step(run, &finish) {
            Ok(ran) => Ok(ran.stdout.trim().to_string()),
            Err(error) if error.contains(CONFLICT) => Err(WriteError::Changed),
            Err(error) => Err(WriteError::Failed(error)),
        }
    })();
    if result.is_err() {
        let _ = run(&format!("cd {} && rm -f {}", shell_quote(dir), shell_quote(&temp)), STEP_TIMEOUT_SECS);
    }
    result
}

/// A new folder, and any folders on the way to it. `Changed` when something
/// (a file, a folder, a link) is already there by that name.
pub fn make_dir(dir: &str, relative: &str, run: Run) -> Result<(), WriteError> {
    let command = format!(
        "cd {dir} || exit 3\n{path}\n{LOCK}\n\
         if [ -e \"./$p\" ] || [ -L \"./$p\" ]; then echo {CONFLICT} >&2; exit 9; fi\n\
         mkdir -p -- \"./$p\"",
        dir = shell_quote(dir),
        path = path_var(relative),
    );
    match step(run, &command) {
        Ok(_) => Ok(()),
        Err(error) if error.contains(CONFLICT) => Err(WriteError::Changed),
        Err(error) => Err(WriteError::Failed(error)),
    }
}

/// Delete a folder and everything in it (a link to one is not followed: it
/// is not a folder here). `relative` can't be the shared folder itself.
pub fn remove_tree(dir: &str, relative: &str, run: Run) -> Result<(), WriteError> {
    if relative.trim_matches('/').is_empty() {
        return Err(WriteError::Failed("the shared folder itself is not deleted".into()));
    }
    let command = format!(
        "cd {dir} || exit 3\n{path}\n{LOCK}\n\
         if [ -d \"./$p\" ] && [ ! -L \"./$p\" ]; then rm -rf -- \"./$p\"; else echo \"not a folder\" >&2; exit 4; fi",
        dir = shell_quote(dir),
        path = path_var(relative),
    );
    match step(run, &command) {
        Ok(_) => Ok(()),
        Err(error) => Err(WriteError::Failed(error)),
    }
}

/// Delete a file, if it is still at `stamp` (a file's) — or a folder with
/// nothing in it (`stamp` none).
pub fn remove(dir: &str, relative: &str, stamp: Option<&str>, run: Run) -> Result<(), WriteError> {
    let check = match stamp {
        Some(stamp) => format!("cur=$(find \"./$p\" {STAMP} 2>/dev/null); [ \"$cur\" = {} ] || {{ echo {CONFLICT} >&2; exit 9; }}", shell_quote(stamp)),
        None => String::new(),
    };
    let command = format!(
        "cd {dir} || exit 3\n{path}\n{LOCK}\n{check}\n\
         if [ -d \"./$p\" ] && [ ! -L \"./$p\" ]; then rmdir -- \"./$p\" 2>/dev/null || {{ echo \"the folder is not empty\" >&2; exit 4; }}; \
         else rm -f -- \"./$p\"; fi",
        dir = shell_quote(dir),
        path = path_var(relative),
    );
    match step(run, &command) {
        Ok(_) => Ok(()),
        Err(error) if error.contains(CONFLICT) => Err(WriteError::Changed),
        Err(error) => Err(WriteError::Failed(error)),
    }
}

// ------------------------------------------------------------------ text

/// Text to show and edit: UTF-8 with no NUL.
pub fn text_of(bytes: &[u8]) -> Option<&str> {
    if bytes.contains(&0) {
        return None;
    }
    std::str::from_utf8(bytes).ok()
}

/// What changed from `before` to `after`, as a unified diff, and whether it
/// is all there (`DIFF_LINES` at most).
pub fn diff(before: &str, after: &str) -> (String, bool) {
    let text = similar::TextDiff::from_lines(before, after).unified_diff().context_radius(2).to_string();
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= DIFF_LINES {
        return (text.trim_end().to_string(), true);
    }
    let mut cut = lines[..DIFF_LINES].join("\n");
    cut.push_str(&format!("\n… {} more diff lines", lines.len() - DIFF_LINES));
    (cut, false)
}

// ------------------------------------------------------------------ the ledger

#[derive(Clone, Debug)]
struct Seen {
    stamp: String,
    /// When small enough to keep, for diffs.
    text: Option<String>,
}

impl Seen {
    fn of(stamp: &str, bytes: &[u8]) -> Seen {
        let text = text_of(bytes).filter(|text| text.len() <= KEEP_TEXT).map(str::to_string);
        Seen { stamp: stamp.to_string(), text }
    }
}

#[derive(Default)]
struct Ledger {
    /// The agent's: each file as it last read or wrote it.
    seen: HashMap<String, Seen>,
    /// What the operator did in the tab, for his next message.
    changes: Vec<Change>,
}

static LEDGER: Mutex<Option<Ledger>> = Mutex::new(None);

fn ledger<T>(f: impl FnOnce(&mut Ledger) -> T) -> T {
    let mut guard = LEDGER.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    f(guard.get_or_insert_with(Ledger::default))
}

/// Something the operator did in the share tab, for the agent.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Change {
    /// Edited `path` (inside the folder) from `before`, stamped
    /// `before_stamp`, to `after`, stamped `stamp`. Texts too big to carry are
    /// left out (`None`).
    Edited { path: String, before_stamp: String, before: Option<String>, after: Option<String>, stamp: String },
    Created { path: String, after: Option<String>, stamp: String },
    /// From his machine: `/share/x (2.1 KB)`.
    Added { line: String },
    Removed { path: String },
    /// His edit could not go over a change of the agent's, so it was kept beside
    /// it as `kept_as`.
    KeptBeside { path: String, kept_as: String },
}

/// The texts a `Change` carries: small ones only.
pub fn carried(text: &str) -> Option<String> {
    (text.len() <= KEEP_TEXT).then(|| text.to_string())
}

/// What the operator did, kept for his next message (the agent that runs the turns).
pub fn record(change: Change) {
    ledger(|ledger| ledger.changes.push(change));
}

/// The note for the operator's next message: what he did in the share tab since the agent
/// was last told. An edit the agent is shown in full counts as read.
pub fn take_note() -> Option<String> {
    let dir = crate::share::share_dir();
    let changes = ledger(|ledger| std::mem::take(&mut ledger.changes));
    if changes.is_empty() {
        return None;
    }
    let mut note = String::from("\n\n[SHARED FILES]\nThe operator, in his share tab or editor, or in ~/s0ma-share on his machine (it mirrors /share):\n");
    for change in changes {
        match change {
            Change::Edited { path, before_stamp, before, after, stamp } => {
                let name = shown(&dir, &path);
                match (before, after) {
                    (Some(before), Some(after)) => {
                        let (diff, complete) = diff(&before, &after);
                        note.push_str(&format!("- edited {name}:\n```diff\n{diff}\n```\n"));
                        // Shown all of it over what the agent had: the agent has it now.
                        ledger(|ledger| {
                            if complete && ledger.seen.get(&path).is_some_and(|seen| seen.stamp == before_stamp) {
                                ledger.seen.insert(path.clone(), Seen::of(&stamp, after.as_bytes()));
                            }
                        });
                    }
                    _ => note.push_str(&format!("- edited {name} (too big to show the change; shared_read it)\n")),
                }
            }
            Change::Created { path, after, stamp } => {
                let name = shown(&dir, &path);
                match after.filter(|after| after.lines().count() <= DIFF_LINES) {
                    Some(after) => {
                        note.push_str(&format!("- created {name}:\n```\n{}\n```\n", after.trim_end()));
                        ledger(|ledger| ledger.seen.insert(path.clone(), Seen::of(&stamp, after.as_bytes())));
                    }
                    None => note.push_str(&format!("- created {name}\n")),
                }
            }
            Change::Added { line } => note.push_str(&format!("- added from his machine: {line}\n")),
            Change::Removed { path } => {
                note.push_str(&format!("- deleted {}\n", shown(&dir, &path)));
                ledger(|ledger| ledger.seen.remove(&path));
            }
            Change::KeptBeside { path, kept_as } => note.push_str(&format!(
                "- edited {} while you changed it, so his version is beside it as {}\n",
                shown(&dir, &path),
                shown(&dir, &kept_as)
            )),
        }
    }
    note.truncate(note.trim_end().len());
    Some(note)
}

fn seen(path: &str) -> Option<Seen> {
    ledger(|ledger| ledger.seen.get(path).cloned())
}

fn saw(path: &str, stamp: &str, bytes: &[u8]) {
    ledger(|ledger| ledger.seen.insert(path.to_string(), Seen::of(stamp, bytes)));
}

/// What changed in a file since the agent last saw it, for a tool result: a diff
/// when both texts are known, or that it changed.
fn since_seen(seen: Option<&Seen>, stamp: &str, now: &[u8]) -> Option<Value> {
    since(seen, stamp, now).map(|(shown, _)| shown)
}

/// What changed since the agent last saw the file, and whether what it is shown
/// is all of it.
fn since(seen: Option<&Seen>, stamp: &str, now: &[u8]) -> Option<(Value, bool)> {
    let seen = seen?;
    if seen.stamp == stamp {
        return None;
    }
    match (seen.text.as_deref(), text_of(now)) {
        (Some(before), Some(after)) if before == after => None,
        (Some(before), Some(after)) => {
            let (diff, complete) = diff(before, after);
            Some((json!(diff), complete))
        }
        _ => Some((json!(true), false)),
    }
}

/// Whether the agent knows the file as it is now: it saw this version, or the
/// whole change to it since is in what it is shown.
fn knows(seen: Option<&Seen>, stamp: &str, now: &[u8]) -> bool {
    seen.is_some() && since(seen, stamp, now).is_none_or(|(_, complete)| complete)
}

fn forget(path: &str) {
    ledger(|ledger| ledger.seen.remove(path));
}

fn gone(error: &str) -> bool {
    error.starts_with("no such file")
}

// ------------------------------------------------------------------ the agent's tools

pub fn handles(name: &str) -> bool {
    matches!(name, "shared_list" | "shared_read" | "shared_write" | "shared_edit")
}

pub fn tool_definitions() -> Vec<crate::llm::ToolDefinition> {
    let path = json!({"type": "string", "description": "A file in /share: /share/notes.md or notes.md."});
    vec![
        crate::llm::ToolDefinition::function(
            "shared_list",
            "List /share, the folder you and the operator work on together (he has it open in his share tab, Alt+S): each \
             file's size and when it changed, and which changed since you last read them (by the operator, most likely).",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
        ),
        crate::llm::ToolDefinition::function(
            "shared_read",
            "Read a text file in /share. Says what changed since you last read it, as a diff. Long files come in \
             parts: from_line and lines read on.",
            json!({
                "type": "object",
                "properties": {
                    "path": path,
                    "from_line": {"type": "integer", "minimum": 1, "description": "The first line to return (1 is the first)."},
                    "lines": {"type": "integer", "minimum": 1, "description": "How many lines to return."}
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        ),
        crate::llm::ToolDefinition::function(
            "shared_write",
            "Write a whole text file in /share, new or over the old version in place (the operator sees it at once in \
             his share tab). Folders are made as needed. An existing file is only written over when you have read \
             it and it has not changed since: otherwise you are told what changed. To change part of a file, \
             shared_edit.",
            json!({
                "type": "object",
                "properties": {"path": path, "content": {"type": "string", "description": "The whole file."}},
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        ),
        crate::llm::ToolDefinition::function(
            "shared_edit",
            "Change part of a text file in /share in place: old_text, exactly as it is in the file now, becomes \
             new_text. The operator's other changes stay. old_text must be found once (or say replace_all). Says what \
             changed since you last read the file.",
            json!({
                "type": "object",
                "properties": {
                    "path": path,
                    "old_text": {"type": "string", "description": "Exactly as in the file, with enough around it to be found once."},
                    "new_text": {"type": "string"},
                    "replace_all": {"type": "boolean", "description": "Replace every place old_text is found."}
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": false
            }),
        ),
    ]
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadArgs {
    path: String,
    from_line: Option<usize>,
    lines: Option<usize>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteArgs {
    path: String,
    content: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EditArgs {
    path: String,
    old_text: String,
    new_text: String,
    #[serde(default)]
    replace_all: bool,
}

/// The tools, on the production guest command path.
pub fn execute(call: &crate::types::ToolCall) -> Result<String, Box<dyn std::error::Error>> {
    let dir = crate::share::share_dir();
    let mut result = run_tool(&call.function.name, &call.function.arguments, &dir, &mut crate::share::run_on_machine);
    // A file the operator has open: the change shows there, marked as the agent's.
    if let Ok(value) = &mut result
        && matches!(call.function.name.as_str(), "shared_write" | "shared_edit")
        && value.get("ok").and_then(Value::as_bool) != Some(false)
        && let Some(path) = serde_json::from_str::<Value>(&call.function.arguments).ok().and_then(|args| args.get("path").and_then(Value::as_str).map(str::to_string))
        && let Ok(relative) = relative(&dir, &path)
        && let Some(unsaved) = crate::tabs::editor_has(&dir, &relative)
    {
        value["open_in_editor"] = json!(if unsaved {
            "The operator has this file open with unsaved changes: yours is merged in when they save, marked as yours."
        } else {
            "The operator has this file open: your change shows there now, marked as yours."
        });
    }
    Ok(match result {
        Ok(value) => value.to_string(),
        Err(error) => json!({"ok": false, "error": error}).to_string(),
    })
}

fn run_tool(name: &str, arguments: &str, dir: &str, run: Run) -> Result<Value, String> {
    let bad = |error: serde_json::Error| format!("bad arguments: {error}");
    match name {
        "shared_list" => list_tool(dir, run),
        "shared_read" => read_tool(dir, serde_json::from_str(arguments).map_err(bad)?, run),
        "shared_write" => write_tool(dir, serde_json::from_str(arguments).map_err(bad)?, run),
        "shared_edit" => edit_tool(dir, serde_json::from_str(arguments).map_err(bad)?, run),
        other => Err(format!("unknown tool {other}")),
    }
}

fn when(modified: f64) -> String {
    chrono::DateTime::from_timestamp(modified as i64, 0)
        .map(|at| at.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_default()
}

fn list_tool(dir: &str, run: Run) -> Result<Value, String> {
    let entries = list(dir, run)?;
    let files: Vec<Value> = entries
        .iter()
        .filter(|entry| !entry.dir)
        .map(|entry| {
            let mut file = json!({"path": shown(dir, &entry.path), "size": human_size(entry.bytes), "modified": when(entry.modified)});
            match seen(&entry.path) {
                None => file["state"] = json!("not read by you"),
                Some(seen) if seen.stamp != entry.stamp => file["state"] = json!("changed since you last read it"),
                Some(_) => {}
            }
            file
        })
        .collect();
    Ok(json!({"folder": dir, "files": files}))
}

/// A file read whole, as text, for the tools.
fn read_text(dir: &str, relative: &str, run: Run) -> Result<(String, String), String> {
    let read = read(dir, relative, MAX_WHOLE_BYTES, true, run)?;
    let text = text_of(&read.bytes)
        .ok_or_else(|| format!("{} is not text ({}); use vm_exec for it", shown(dir, relative), human_size(read.size)))?
        .to_string();
    Ok((text, read.stamp))
}

fn read_tool(dir: &str, args: ReadArgs, run: Run) -> Result<Value, String> {
    let relative = relative(dir, &args.path)?;
    if args.lines == Some(0) {
        return Err("lines must be 1 or more".into());
    }
    let before = seen(&relative);
    let (text, stamp) = read_text(dir, &relative, run).inspect_err(|error| {
        if gone(error) {
            forget(&relative);
        }
    })?;
    let changed = since_seen(before.as_ref(), &stamp, text.as_bytes());
    saw(&relative, &stamp, text.as_bytes());
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let total = lines.len();
    let from = args.from_line.unwrap_or(1).max(1);
    let mut content = String::new();
    let mut to = from.saturating_sub(1);
    let mut cut = None;
    for line in lines.iter().skip(from - 1).take(args.lines.unwrap_or(usize::MAX)) {
        if content.len() + line.len() > READ_CHARS {
            if content.is_empty() {
                // One line longer than a read: its beginning.
                let mut end = READ_CHARS;
                while !line.is_char_boundary(end) {
                    end -= 1;
                }
                content.push_str(&line[..end]);
                to += 1;
                cut = Some(format!("line {to} is {} long; this is its first {}", human_size(line.len() as u64), human_size(end as u64)));
            }
            break;
        }
        content.push_str(line);
        to += 1;
    }
    let mut result = json!({"path": shown(dir, &relative), "size": human_size(text.len() as u64), "lines": total, "content": content});
    if from > 1 || to < total {
        result["showing"] = json!(format!("lines {from}–{to} of {total}"));
    }
    if to < total && args.lines.is_none_or(|lines| (to + 1).saturating_sub(from) < lines) {
        result["more"] = json!(format!("read on with from_line {}", to + 1));
    }
    if let Some(cut) = cut {
        result["cut"] = json!(cut);
    }
    if let Some(changed) = changed {
        result["changed_since_you_last_read"] = changed;
    }
    Ok(result)
}

fn write_tool(dir: &str, args: WriteArgs, run: Run) -> Result<Value, String> {
    let relative = relative(dir, &args.path)?;
    let name = shown(dir, &relative);
    let mut before = seen(&relative);
    let expect = before.as_ref().map_or(Expect::New, |seen| Expect::Stamp(seen.stamp.clone()));
    let mut wrote = write(dir, &relative, args.content.as_bytes(), &expect, run);
    if wrote == Err(WriteError::Changed) && before.is_some() {
        // Gone since the agent read it: nothing is written over, so it is new.
        if let Err(error) = read(dir, &relative, 0, false, run)
            && gone(&error)
        {
            forget(&relative);
            before = None;
            wrote = write(dir, &relative, args.content.as_bytes(), &Expect::New, run);
        }
    }
    match wrote {
        Ok(stamp) => {
            saw(&relative, &stamp, args.content.as_bytes());
            let verb = if before.is_some() { "updated" } else { "wrote" };
            crate::llm::background_status(format_args!("[share] s0ma {verb} {name} · alt+s"));
            Ok(json!({"ok": true, "path": name, "size": human_size(args.content.len() as u64)}))
        }
        Err(WriteError::Changed) => {
            // Say what is there, so the next step is informed.
            let now = read(dir, &relative, MAX_WHOLE_BYTES, true, run).ok();
            let mut refused = json!({"ok": false, "path": name});
            match (&before, now) {
                (None, _) => {
                    refused["error"] = json!(format!(
                        "{name} already exists and you have not read it: shared_read it first (or shared_edit part of it), \
                         so nothing of the operator's is written over unseen"
                    ));
                }
                (Some(_), Some(now)) => {
                    refused["error"] = json!(format!("{name} changed since you last read it; nothing was written"));
                    if let Some(changed) = since_seen(before.as_ref(), &now.stamp, &now.bytes) {
                        refused["what_changed"] = changed;
                    }
                }
                (Some(_), None) => {
                    refused["error"] = json!(format!("{name} changed since you last read it; nothing was written"));
                }
            }
            Ok(refused)
        }
        Err(WriteError::Failed(error)) => Err(error),
    }
}

fn edit_tool(dir: &str, args: EditArgs, run: Run) -> Result<Value, String> {
    let relative = relative(dir, &args.path)?;
    let name = shown(dir, &relative);
    if args.old_text.is_empty() {
        return Err("old_text is empty; to write a whole file, shared_write".into());
    }
    let before = seen(&relative);
    // As the file is now; written back only if it still is.
    for _ in 0..3 {
        let (text, stamp) = read_text(dir, &relative, run).inspect_err(|error| {
            if gone(error) {
                forget(&relative);
            }
        })?;
        let changed = since_seen(before.as_ref(), &stamp, text.as_bytes());
        // Shown no more than a diff: the file counts as read only when the
        // diff was all of it.
        let known = knows(before.as_ref(), &stamp, text.as_bytes());
        let found = text.matches(&args.old_text).count();
        if found == 0 || (found > 1 && !args.replace_all) {
            if known {
                saw(&relative, &stamp, text.as_bytes());
            }
            let error = if found == 0 {
                format!("old_text is not in {name} as it is now; nothing was changed")
            } else {
                format!("old_text is in {name} {found} times; give more of the text around it, or replace_all")
            };
            let mut refused = json!({"ok": false, "error": error});
            if let Some(changed) = changed {
                refused["changed_since_you_last_read"] = changed;
            }
            return Ok(refused);
        }
        let edited = if args.replace_all {
            text.replace(&args.old_text, &args.new_text)
        } else {
            text.replacen(&args.old_text, &args.new_text, 1)
        };
        match write(dir, &relative, edited.as_bytes(), &Expect::Stamp(stamp), run) {
            Ok(stamp) => {
                if known {
                    saw(&relative, &stamp, edited.as_bytes());
                }
                crate::llm::background_status(format_args!("[share] s0ma edited {name} · alt+s"));
                let mut done = json!({"ok": true, "path": name, "replaced": if args.replace_all { found } else { 1 }});
                if let Some(changed) = changed {
                    done["changed_since_you_last_read"] = changed;
                }
                return Ok(done);
            }
            // Changed between the read and the write: again, from the new text.
            Err(WriteError::Changed) => continue,
            Err(WriteError::Failed(error)) => return Err(error),
        }
    }
    Err(format!("{name} kept changing as it was edited; nothing was changed"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn local(command: &str, _timeout: u64) -> Result<Ran, String> {
        let output = std::process::Command::new("bash")
            .args(["-c", command])
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|error| error.to_string())?;
        Ok(Ran {
            exit_code: output.status.code().map(i64::from),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    /// Held by a test that records or takes notes: they are one queue.
    pub(crate) static NOTES: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(crate) struct Folder(pub String);

    impl Folder {
        pub(crate) fn new(tag: &str) -> Folder {
            let path = std::env::temp_dir().join(format!("s0ma-shared-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Folder(path.display().to_string())
        }

        pub(crate) fn put(&self, relative: &str, text: &str) {
            let path = std::path::Path::new(&self.0).join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }

        pub(crate) fn get(&self, relative: &str) -> String {
            std::fs::read_to_string(std::path::Path::new(&self.0).join(relative)).unwrap()
        }
    }

    impl Drop for Folder {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tool(name: &str, arguments: Value, dir: &str) -> Value {
        match run_tool(name, &arguments.to_string(), dir, &mut local) {
            Ok(value) => value,
            Err(error) => json!({"error": error}),
        }
    }

    /// The operator changing a file under the agent, with a new modification time.
    fn operator_writes(folder: &Folder, relative: &str, text: &str) {
        std::thread::sleep(std::time::Duration::from_millis(15));
        folder.put(relative, text);
    }

    #[test]
    fn s0ma_never_writes_over_what_it_has_not_seen_and_is_told_what_changed() {
        let folder = Folder::new("tools");
        let dir = folder.0.as_str();
        folder.put("plan.md", "# Plan\n- one\n- two\n");
        // Not read: refused.
        let refused = tool("shared_write", json!({"path": "plan.md", "content": "mine\n"}), dir);
        assert_eq!(refused["ok"], false);
        assert!(refused["error"].as_str().unwrap().contains("have not read it"), "{refused}");
        assert_eq!(folder.get("plan.md"), "# Plan\n- one\n- two\n");
        // Listed as not read, then read.
        let listed = tool("shared_list", json!({}), dir);
        assert_eq!(listed["files"][0]["state"], "not read by you", "{listed}");
        let read = tool("shared_read", json!({"path": format!("{dir}/plan.md")}), dir);
        assert_eq!((read["content"].as_str(), read["lines"].as_u64()), (Some("# Plan\n- one\n- two\n"), Some(3)));
        assert!(tool("shared_list", json!({}), dir)["files"][0].get("state").is_none());
        // The operator changes it: listed, refused with the diff, and an edit keeps his change.
        operator_writes(&folder, "plan.md", "# Plan\n- one\n- two\n- the operator's three\n");
        assert_eq!(tool("shared_list", json!({}), dir)["files"][0]["state"], "changed since you last read it");
        let refused = tool("shared_write", json!({"path": "plan.md", "content": "# Plan\n- ONE\n- two\n"}), dir);
        assert_eq!(refused["ok"], false);
        assert!(refused["what_changed"].as_str().unwrap().contains("+- the operator's three"), "{refused}");
        let edited = tool("shared_edit", json!({"path": "plan.md", "old_text": "- one", "new_text": "- ONE"}), dir);
        assert_eq!(edited["ok"], true, "{edited}");
        assert!(edited["changed_since_you_last_read"].as_str().unwrap().contains("the operator's three"));
        assert_eq!(folder.get("plan.md"), "# Plan\n- ONE\n- two\n- the operator's three\n");
        // Seen now: a whole write goes.
        let wrote = tool("shared_write", json!({"path": "plan.md", "content": "done\n"}), dir);
        assert_eq!(wrote["ok"], true, "{wrote}");
        assert_eq!(folder.get("plan.md"), "done\n");
        // Ambiguous and missing old_text.
        folder.put("dup.txt", "a a a");
        assert!(tool("shared_edit", json!({"path": "dup.txt", "old_text": "a", "new_text": "b"}), dir)["error"].as_str().unwrap().contains("3 times"));
        let all = tool("shared_edit", json!({"path": "dup.txt", "old_text": "a", "new_text": "b", "replace_all": true}), dir);
        assert_eq!((all["replaced"].as_u64(), folder.get("dup.txt").as_str()), (Some(3), "b b b"));
        assert!(tool("shared_edit", json!({"path": "dup.txt", "old_text": "zzz", "new_text": "b"}), dir)["error"].as_str().unwrap().contains("not in"));
        // A new file, with its folders.
        assert_eq!(tool("shared_write", json!({"path": "new/dir/x.md", "content": "x"}), dir)["ok"], true);
        assert_eq!(folder.get("new/dir/x.md"), "x");
        // Not text.
        std::fs::write(std::path::Path::new(dir).join("bin.dat"), [0u8, 1, 2]).unwrap();
        assert!(tool("shared_read", json!({"path": "bin.dat"}), dir)["error"].as_str().unwrap().contains("not text"));
        assert!(tool("shared_read", json!({"path": "/etc/passwd"}), dir)["error"].as_str().unwrap().contains("not in"));
    }

    #[test]
    fn of_writers_racing_over_one_version_only_one_wins() {
        let folder = Folder::new("race-many");
        folder.put("r.md", "base\n");
        let stamp = read(&folder.0, "r.md", 100, true, &mut local).unwrap().stamp;
        let wins: usize = std::thread::scope(|scope| {
            let writers: Vec<_> = (0..8)
                .map(|n| {
                    let (dir, stamp) = (folder.0.clone(), stamp.clone());
                    scope.spawn(move || write(&dir, "r.md", format!("writer {n}\n").as_bytes(), &Expect::Stamp(stamp), &mut local).is_ok())
                })
                .collect();
            writers.into_iter().map(|writer| usize::from(writer.join().unwrap())).sum()
        });
        assert_eq!(wins, 1, "{}", folder.get("r.md"));
    }

    #[test]
    fn reads_keep_to_their_size_and_refuse_no_lines() {
        let folder = Folder::new("bounds");
        folder.put("one-line.txt", &"é".repeat(READ_CHARS));
        let read = tool("shared_read", json!({"path": "one-line.txt"}), &folder.0);
        let content = read["content"].as_str().unwrap();
        assert!(content.len() <= READ_CHARS && !content.is_empty(), "{}", content.len());
        assert!(read["cut"].as_str().unwrap().contains("line 1"), "{read}");
        assert!(tool("shared_read", json!({"path": "one-line.txt", "lines": 0}), &folder.0)["error"].as_str().unwrap().contains("1 or more"));
        // Past the end: nothing, and no panic.
        let past = tool("shared_read", json!({"path": "one-line.txt", "from_line": 5, "lines": 1}), &folder.0);
        assert_eq!(past["content"], "");
    }
}
