//! `~/s0ma-share` on the operator's machine (`share::receive_dir`) kept in step with
//! `/share` on the agent's (`share::share_dir`), both ways, while the agent runs.
//!
//! Every few seconds the two sides are listed: the operator's folder from the disk,
//! the agent's through one guest command (`crate::share::files::list`). Each file is set
//! against how it was when last in step (`Synced`: its local size and
//! modification time, and its stamp on the agent's side), kept in
//! `.s0ma-sync.json` in the operator's folder, and the one side that changed is
//! copied to the other:
//!
//! | here | there | was in step | does |
//! |---|---|---|---|
//! | new | — | no | upload |
//! | — | new | no | download |
//! | both | both | no | the same: recorded; different: a conflict |
//! | changed | as it was | yes | upload |
//! | as it was | changed | yes | download |
//! | changed | changed | yes | the same: recorded; different: a conflict |
//! | deleted | as it was | yes | delete there |
//! | as it was | deleted | yes | put aside here, in `.s0ma-deleted/`, and said |
//! | deleted | changed | yes | download (the change wins over the delete) |
//! | changed | deleted | yes | upload |
//!
//! **A conflict loses nothing:** the operator's version is renamed beside it here,
//! `plan (operator copy).md`, and both go to both sides on the next rounds.
//!
//! **Uploads** go through `crate::share::files::write`, so they are written over a
//! file only when it is still the version listed, and the agent hears of them as
//! of the share tab's edits: with the operator's next message, an edit as a diff.
//! **Downloads** are written beside the file and renamed over it, only if
//! the file here is still as listed.
//!
//! Hidden files and folders, `node_modules` and `__pycache__` are left out
//! on both sides, and files over 32 MB are not copied. `S0MA_SHARE_SYNC=off`
//! turns it off.

use crate::share::files::{Change, Expect, Run, WriteError};

use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, UNIX_EPOCH};

/// How often the two sides are compared, and how long after a failure.
const EVERY: Duration = Duration::from_secs(3);
const AFTER_FAILURE: Duration = Duration::from_secs(30);
/// Bigger files are not copied.
const MAX_BYTES: u64 = crate::share::MAX_SEND_BYTES;
const DEPTH: usize = 8;
const MAX_FILES: usize = 3000;
const STATE_FILE: &str = ".s0ma-sync.json";

/// A file as it was when last in step on both sides.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Synced {
    /// Here: `size:modified-nanoseconds`.
    local: String,
    /// There: `shared_files` stamp.
    remote: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct SyncState {
    /// The folder on the agent's side these records are for.
    remote_dir: String,
    files: BTreeMap<String, Synced>,
    /// A round has run: what was here before it is not news.
    #[serde(default)]
    primed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LocalFile {
    sig: String,
    bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RemoteFile {
    stamp: String,
    bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    Nothing,
    Upload,
    Download,
    /// Both there, not known to be in step: the same is recorded, different
    /// is a conflict.
    Compare,
    DeleteRemote,
    DeleteLocal,
    Forget,
}

/// What one file needs.
pub(crate) fn decide(local: Option<&LocalFile>, remote: Option<&RemoteFile>, base: Option<&Synced>) -> Step {
    match (local, remote, base) {
        (None, None, _) => Step::Forget,
        (Some(_), None, None) => Step::Upload,
        (None, Some(_), None) => Step::Download,
        (Some(_), Some(_), None) => Step::Compare,
        (Some(local), Some(remote), Some(base)) => match (local.sig != base.local, remote.stamp != base.remote) {
            (false, false) => Step::Nothing,
            (true, false) => Step::Upload,
            (false, true) => Step::Download,
            (true, true) => Step::Compare,
        },
        (Some(local), None, Some(base)) => {
            if local.sig != base.local {
                Step::Upload
            } else {
                Step::DeleteLocal
            }
        }
        (None, Some(remote), Some(base)) => {
            if remote.stamp != base.remote {
                Step::Download
            } else {
                Step::DeleteRemote
            }
        }
    }
}

// ------------------------------------------------------------------ this side

fn signature(metadata: &std::fs::Metadata) -> String {
    let nanos = metadata.modified().ok().and_then(|at| at.duration_since(UNIX_EPOCH).ok()).map_or(0, |at| at.as_nanos());
    format!("{}:{nanos}", metadata.len())
}

fn skipped(name: &str) -> bool {
    name.starts_with('.') || name == "node_modules" || name == "__pycache__"
}

/// The operator's folder: its files by path inside it.
pub(crate) fn scan(root: &Path) -> BTreeMap<String, LocalFile> {
    let mut found = BTreeMap::new();
    let mut stack = vec![(root.to_path_buf(), String::new(), 0usize)];
    while let Some((dir, prefix, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.filter_map(Result::ok) {
            let Ok(name) = entry.file_name().into_string() else { continue };
            if skipped(&name) || found.len() >= MAX_FILES {
                continue;
            }
            let Ok(metadata) = std::fs::symlink_metadata(entry.path()) else { continue };
            let path = format!("{prefix}{name}");
            if metadata.is_dir() {
                if depth + 1 < DEPTH {
                    stack.push((entry.path(), format!("{path}/"), depth + 1));
                }
            } else if metadata.is_file() {
                found.insert(path, LocalFile { sig: signature(&metadata), bytes: metadata.len() });
            }
        }
    }
    found
}

/// The files of the mirror at `root` the sync keeps, with each one's
/// signature: what a backup holds (`share/backup.rs`), and what tells it
/// whether anything changed since the last.
pub(crate) fn listing(root: &Path) -> Vec<(String, String)> {
    scan(root).into_iter().map(|(path, file)| (path, file.sig)).collect()
}

fn local_sig(path: &Path) -> Option<String> {
    std::fs::symlink_metadata(path).ok().filter(|metadata| metadata.is_file()).map(|metadata| signature(&metadata))
}

/// `bytes` in place of `path`, written beside it and renamed over it, only
/// if the file there is still `expected` (none: not there). Its signature.
fn put_local(path: &Path, bytes: &[u8], expected: Option<&str>) -> Result<Option<String>, String> {
    let parent = path.parent().ok_or("no folder")?;
    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    let temp = parent.join(format!(".{name}.s0ma-sync-{}", std::process::id()));
    std::fs::write(&temp, bytes).map_err(|error| error.to_string())?;
    if local_sig(path).as_deref() != expected {
        let _ = std::fs::remove_file(&temp);
        return Ok(None);
    }
    std::fs::rename(&temp, path).map_err(|error| {
        let _ = std::fs::remove_file(&temp);
        error.to_string()
    })?;
    Ok(local_sig(path))
}

/// A free name beside `path` on both sides: `plan (operator copy).md`.
fn beside(path: &str, taken: impl Fn(&str) -> bool) -> String {
    let (parent, name) = match path.rsplit_once('/') {
        Some((parent, name)) => (format!("{parent}/"), name),
        None => (String::new(), path),
    };
    let (stem, extension) = crate::share::split_name(name);
    (1..)
        .map(|n| if n == 1 { format!("{parent}{stem} (operator copy){extension}") } else { format!("{parent}{stem} (operator copy {n}){extension}") })
        .find(|candidate| !taken(candidate))
        .expect("an unbounded range")
}

// ------------------------------------------------------------------ a round

/// The folder here that a file deleted on the agent's side is moved into, under
/// the path it had: hidden, so the sync leaves it out.
pub const ASIDE: &str = ".s0ma-deleted";

/// How a round's line says a file was put aside (`DeleteLocal`).
const DELETED: &str = "deleted on s0ma's machine: ";

/// `path` (inside `root`) moved into `root/.s0ma-deleted/`, under its path;
/// a name taken there gets the time after it. Where it went.
fn put_aside(root: &Path, path: &str) -> Result<std::path::PathBuf, String> {
    let mut kept = root.join(ASIDE).join(path);
    if kept.exists() {
        let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let name = kept.file_name().map(|name| name.to_string_lossy().to_string()).unwrap_or_default();
        kept.set_file_name(format!("{name}.{stamp}"));
    }
    if let Some(parent) = kept.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    std::fs::rename(root.join(path), &kept).map_err(|error| error.to_string())?;
    Ok(kept)
}

/// What a round did, a line each, for the status line.
pub(crate) fn round(root: &Path, dir: &str, state: &mut SyncState, run: Run) -> Result<Vec<String>, String> {
    if state.remote_dir != dir {
        *state = SyncState { remote_dir: dir.to_string(), ..SyncState::default() };
    }
    std::fs::create_dir_all(root).map_err(|error| format!("cannot make {}: {error}", root.display()))?;
    let remote: BTreeMap<String, RemoteFile> = crate::share::files::list(dir, run)?
        .into_iter()
        .filter(|entry| !entry.dir)
        .map(|entry| (entry.path, RemoteFile { stamp: entry.stamp, bytes: entry.bytes }))
        .collect();
    let local = scan(root);
    // The first round: what was already here is announced by name only.
    let first = !state.primed;
    let paths: BTreeSet<String> = local.keys().chain(remote.keys()).chain(state.files.keys()).cloned().collect();
    let mut did = Vec::new();
    for path in paths {
        let (here, there) = (local.get(&path), remote.get(&path));
        if here.is_some_and(|file| file.bytes > MAX_BYTES) || there.is_some_and(|file| file.bytes > MAX_BYTES) {
            continue;
        }
        let base = state.files.get(&path).cloned();
        let step = decide(here, there, base.as_ref());
        let result = match step {
            Step::Nothing => Ok(None),
            Step::Forget => {
                state.files.remove(&path);
                Ok(None)
            }
            Step::Upload => upload(root, dir, &path, here.expect("here"), there, base.as_ref(), first, state, run),
            Step::Download => download(root, dir, &path, here, there.expect("there"), state, run),
            Step::Compare => compare(root, dir, &path, here.expect("here"), there.expect("there"), &local, &remote, state, run),
            Step::DeleteRemote => {
                let there = there.expect("there");
                match crate::share::files::remove(dir, &path, Some(&there.stamp), run) {
                    Ok(()) => {
                        state.files.remove(&path);
                        crate::share::files::record(Change::Removed { path: path.clone() });
                        Ok(Some(format!("deleted {path} on s0ma's side")))
                    }
                    // Changed there meanwhile: the next round downloads it.
                    Err(WriteError::Changed) => Ok(None),
                    Err(WriteError::Failed(error)) => Err(error),
                }
            }
            // Gone on the agent's side: put aside here, not deleted, and said.
            Step::DeleteLocal => {
                let file = root.join(&path);
                if local_sig(&file).as_deref() == here.map(|file| file.sig.as_str()) {
                    let kept = put_aside(root, &path)?;
                    state.files.remove(&path);
                    Ok(Some(format!("{DELETED}{path} · kept as {}", home_short(&kept))))
                } else {
                    Ok(None)
                }
            }
        };
        match result {
            Ok(Some(line)) => did.push(line),
            Ok(None) => {}
            Err(error) => did.push(format!("{path}: {error}")),
        }
    }
    state.primed = true;
    Ok(did)
}

#[allow(clippy::too_many_arguments)]
fn upload(
    root: &Path,
    dir: &str,
    path: &str,
    here: &LocalFile,
    there: Option<&RemoteFile>,
    base: Option<&Synced>,
    first: bool,
    state: &mut SyncState,
    run: Run,
) -> Result<Option<String>, String> {
    let file = root.join(path);
    let bytes = std::fs::read(&file).map_err(|error| error.to_string())?;
    // Changing as it is read: the next round.
    if local_sig(&file).as_deref() != Some(here.sig.as_str()) {
        return Ok(None);
    }
    // For the agent: what it was, when it is text small enough to show a diff.
    let before = match (there, base) {
        (Some(there), Some(_)) if there.bytes <= 256 * 1024 => crate::share::files::read(dir, path, there.bytes.max(1), true, run)
            .ok()
            .filter(|read| read.stamp == there.stamp)
            .and_then(|read| crate::share::files::text_of(&read.bytes).map(str::to_string)),
        _ => None,
    };
    let expect = there.map_or(Expect::New, |there| Expect::Stamp(there.stamp.clone()));
    let stamp = match crate::share::files::write(dir, path, &bytes, &expect, run) {
        Ok(stamp) => stamp,
        Err(WriteError::Changed) => return Ok(None),
        Err(WriteError::Failed(error)) => return Err(error),
    };
    state.files.insert(path.to_string(), Synced { local: here.sig.clone(), remote: stamp.clone() });
    crate::tabs::share_seen(dir, path, &stamp);
    wrote(dir, path, &stamp);
    let after = crate::share::files::text_of(&bytes).and_then(crate::share::files::carried);
    crate::share::files::record(match there {
        Some(there) => Change::Edited { path: path.to_string(), before_stamp: there.stamp.clone(), before, after, stamp },
        None if first => Change::Added {
            line: format!("{} ({}, already in ~/s0ma-share)", crate::share::files::shown(dir, path), crate::share::human_size(bytes.len() as u64)),
        },
        None => Change::Created { path: path.to_string(), after, stamp },
    });
    Ok(Some(format!("{path} → s0ma")))
}

fn download(
    root: &Path,
    dir: &str,
    path: &str,
    here: Option<&LocalFile>,
    there: &RemoteFile,
    state: &mut SyncState,
    run: Run,
) -> Result<Option<String>, String> {
    let read = crate::share::files::read(dir, path, MAX_BYTES, true, run)?;
    let Some(sig) = put_local(&root.join(path), &read.bytes, here.map(|file| file.sig.as_str()))? else {
        // The operator changed it meanwhile: the next round sees both changed.
        return Ok(None);
    };
    let _ = there;
    state.files.insert(path.to_string(), Synced { local: sig, remote: read.stamp });
    Ok(Some(format!("{path} ← s0ma")))
}

#[allow(clippy::too_many_arguments)]
fn compare(
    root: &Path,
    dir: &str,
    path: &str,
    here: &LocalFile,
    there: &RemoteFile,
    local: &BTreeMap<String, LocalFile>,
    remote: &BTreeMap<String, RemoteFile>,
    state: &mut SyncState,
    run: Run,
) -> Result<Option<String>, String> {
    let file = root.join(path);
    let mine = std::fs::read(&file).map_err(|error| error.to_string())?;
    let theirs = if here.bytes == there.bytes { Some(crate::share::files::read(dir, path, MAX_BYTES, true, run)?) } else { None };
    if let Some(theirs) = theirs.filter(|theirs| theirs.bytes == mine) {
        state.files.insert(path.to_string(), Synced { local: here.sig.clone(), remote: theirs.stamp });
        return Ok(None);
    }
    // Both changed: the operator's is kept beside, and both reach both sides.
    let kept = beside(path, |candidate| local.contains_key(candidate) || remote.contains_key(candidate) || root.join(candidate).exists());
    if local_sig(&file).as_deref() != Some(here.sig.as_str()) {
        return Ok(None);
    }
    std::fs::rename(&file, root.join(&kept)).map_err(|error| error.to_string())?;
    state.files.remove(path);
    crate::share::files::record(Change::KeptBeside { path: path.to_string(), kept_as: kept.clone() });
    Ok(Some(format!("{path} changed on both sides · yours is {kept}")))
}

// ------------------------------------------------------------------ running

fn load(root: &Path) -> SyncState {
    std::fs::read(root.join(STATE_FILE)).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default()
}

fn save(root: &Path, state: &SyncState) {
    let Ok(json) = serde_json::to_vec_pretty(state) else { return };
    // Missed, the next sync compares by content and loses nothing.
    let _ = crate::util::write_atomic(&root.join(STATE_FILE), json);
}

struct Status {
    root: PathBuf,
    dir: String,
    problem: Option<String>,
}

static STATUS: Mutex<Option<Status>> = Mutex::new(None);
/// On an attached terminal: the status as the agent running the sync last
/// reported it (`mirror`).
static MIRRORED: Mutex<Option<Status>> = Mutex::new(None);

/// The sync's status here, or as reported to this attached terminal.
fn status<T>(read: impl FnOnce(&Status) -> Option<T>) -> Option<T> {
    for status in [&STATUS, &MIRRORED] {
        let status = status.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(status) = status.as_ref() {
            return read(status);
        }
    }
    None
}

/// For the share tab: `⇄ ~/s0ma-share`, and why it is paused, if it is.
pub fn label() -> Option<(String, Option<String>)> {
    status(|status| Some((format!("⇄ {}", home_short(&status.root)), status.problem.clone())))
}

/// Where a file of `/share` is on the operator's machine, while they are in step.
pub fn local_path(dir: &str, path: &str) -> Option<PathBuf> {
    status(|status| (status.dir == dir).then(|| status.root.join(path)))
}

/// What the sync is doing, for an attached terminal's share tab: its
/// folders and problem, and the files it wrote after mark `since`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// Not running (turned off, or not started yet).
    pub root: Option<PathBuf>,
    pub dir: String,
    pub problem: Option<String>,
    /// (folder, path, stamp) of each file written, oldest first.
    pub wrote: Vec<(String, String, String)>,
    /// To ask from next time.
    pub mark: u64,
}

/// The files the sync wrote, kept for attached terminals: (mark, folder,
/// path, stamp), the latest `WROTE_KEPT`.
type Wrote = (u64, String, String, String);
static WROTE: Mutex<(u64, std::collections::VecDeque<Wrote>)> = Mutex::new((0, std::collections::VecDeque::new()));
const WROTE_KEPT: usize = 500;

fn wrote(dir: &str, path: &str, stamp: &str) {
    let mut wrote = WROTE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    wrote.0 += 1;
    let mark = wrote.0;
    wrote.1.push_back((mark, dir.to_string(), path.to_string(), stamp.to_string()));
    if wrote.1.len() > WROTE_KEPT {
        wrote.1.pop_front();
    }
}

pub fn report(since: u64) -> Report {
    let (root, dir, problem) = STATUS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .as_ref()
        .map(|status| (Some(status.root.clone()), status.dir.clone(), status.problem.clone()))
        .unwrap_or_default();
    let wrote = WROTE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // A mark from before this agent started (the terminal attached to the
    // one before): everything kept.
    let since = if since > wrote.0 { 0 } else { since };
    Report {
        root,
        dir,
        problem,
        wrote: wrote.1.iter().filter(|(mark, ..)| *mark > since).map(|(_, dir, path, stamp)| (dir.clone(), path.clone(), stamp.clone())).collect(),
        mark: wrote.0,
    }
}

/// On an attached terminal: the status `report` gave.
pub fn mirror(report: Report) {
    let status = report.root.map(|root| Status { root, dir: report.dir, problem: report.problem });
    *MIRRORED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = status;
}

pub fn home_short(path: &Path) -> String {
    let shown = path.display().to_string();
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() && shown.starts_with(&format!("{home}/")) => format!("~{}", &shown[home.len()..]),
        _ => shown,
    }
}

/// Once, in the agent that runs the turns.
pub fn start() {
    if cfg!(test) {
        return;
    }
    if std::env::var("S0MA_SHARE_SYNC").is_ok_and(|value| matches!(value.trim(), "off" | "0" | "false" | "no")) {
        return;
    }
    // Host mode: the shared folder is this computer's, with nothing to sync.
    if crate::vm::machine::on_host() && crate::share::share_dir() == crate::share::receive_dir().display().to_string().trim_end_matches('/') {
        return;
    }
    let root = crate::share::receive_dir();
    let dir = crate::share::share_dir();
    if let Ok(mut status) = STATUS.lock() {
        *status = Some(Status { root: root.clone(), dir: dir.clone(), problem: None });
    }
    let _ = std::thread::Builder::new().name("share-sync".into()).spawn(move || {
        let mut state = load(&root);
        let mut failing = false;
        loop {
            let started = Instant::now();
            let before = serde_json::to_string(&state).unwrap_or_default();
            let result = round(&root, &dir, &mut state, &mut crate::share::run_on_machine);
            if serde_json::to_string(&state).unwrap_or_default() != before {
                save(&root, &state);
            }
            let problem = match result {
                Ok(did) => {
                    if failing {
                        crate::terminal::say_done(&format!("[share] in step again: {} ⇄ {dir}", home_short(&root)));
                    }
                    failing = false;
                    // Files the agent's side deleted are said, one line each, as
                    // are problems; what was copied is the share tab's to show.
                    for line in &did {
                        if line.starts_with(DELETED) {
                            crate::terminal::warn_status(&format!("[share] {line}"));
                        } else if line.contains(": ") {
                            crate::terminal::warn_status(&format!("[share] sync: {line}"));
                        }
                    }
                    None
                }
                Err(error) => {
                    if !failing {
                        crate::terminal::warn_status(&format!(
                            "[share] {} and {dir} are not in step: {error}; trying again every {}s",
                            home_short(&root),
                            AFTER_FAILURE.as_secs()
                        ));
                    }
                    failing = true;
                    Some(error)
                }
            };
            if let Ok(mut status) = STATUS.lock()
                && let Some(status) = status.as_mut()
            {
                status.problem = problem;
            }
            let wait = if failing { AFTER_FAILURE } else { EVERY };
            std::thread::sleep(wait.saturating_sub(started.elapsed()));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::share::files::tests::{Folder, NOTES, local as run_local};

    struct Pair {
        here: Folder,
        there: Folder,
        state: SyncState,
    }

    impl Pair {
        fn new(tag: &str) -> Pair {
            Pair { here: Folder::new(&format!("{tag}-here")), there: Folder::new(&format!("{tag}-there")), state: SyncState::default() }
        }

        fn round(&mut self) -> Vec<String> {
            // A change within the same clock tick would look like none.
            std::thread::sleep(Duration::from_millis(15));
            round(Path::new(&self.here.0), &self.there.0, &mut self.state, &mut run_local).unwrap()
        }
    }

    #[test]
    fn a_change_on_both_sides_keeps_both() {
        let _notes = NOTES.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut pair = Pair::new("sync-both");
        pair.here.put("both.md", "one\n");
        pair.round();
        pair.here.put("both.md", "the operator's\n");
        pair.there.put("both.md", "s0ma's two\n");
        let did = pair.round();
        assert!(did.iter().any(|line| line.contains("both (operator copy).md")), "{did:?}");
        pair.round();
        assert_eq!(pair.here.get("both.md"), "s0ma's two\n");
        assert_eq!(pair.there.get("both.md"), "s0ma's two\n");
        assert_eq!(pair.here.get("both (operator copy).md"), "the operator's\n");
        assert_eq!(pair.there.get("both (operator copy).md"), "the operator's\n");
        assert!(pair.round().is_empty());
        // Deleted here while changed there: the change wins.
        std::fs::remove_file(Path::new(&pair.here.0).join("both.md")).unwrap();
        pair.there.put("both.md", "s0ma's three\n");
        pair.round();
        assert_eq!(pair.here.get("both.md"), "s0ma's three\n");
        // Two different files of one name, both new: both kept.
        pair.here.put("fresh.md", "mine");
        pair.there.put("fresh.md", "theirs");
        pair.round();
        pair.round();
        assert_eq!((pair.here.get("fresh.md").as_str(), pair.here.get("fresh (operator copy).md").as_str()), ("theirs", "mine"));
        assert_eq!(pair.there.get("fresh (operator copy).md"), "mine");
        let _ = crate::share::files::take_note();
    }

    #[test]
    fn a_download_never_goes_over_a_file_changed_meanwhile() {
        let here = Folder::new("sync-put");
        here.put("a.md", "the operator's new words");
        let path = Path::new(&here.0).join("a.md");
        let listed = "0:0".to_string();
        assert_eq!(put_local(&path, b"s0ma's", Some(&listed)).unwrap(), None);
        assert_eq!(here.get("a.md"), "the operator's new words");
        assert_eq!(put_local(&Path::new(&here.0).join("b.md"), b"x", Some(&listed)).unwrap(), None, "gone meanwhile");
        let sig = local_sig(&path);
        assert!(put_local(&path, b"s0ma's", sig.as_deref()).unwrap().is_some());
        assert_eq!(here.get("a.md"), "s0ma's");
        assert_eq!(std::fs::read_dir(&here.0).unwrap().count(), 1, "no temp file left");
    }
}
