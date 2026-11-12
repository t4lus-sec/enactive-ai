//! Backups of the shared folder: every few hours, `~/s0ma-share` (the
//! operator's copy of `/share`, kept in step by `share/sync.rs`) is written
//! as a tar, gzipped, to `~/s0ma/backups/share-<time>.tar.gz`.
//!
//! - **What is in one:** the files the sync keeps, as they stand here (not
//!   the hidden ones: `.s0ma-deleted/`, the sync's own state).
//! - **When:** five minutes after the agent starts, then every six hours, and
//!   only when something changed since the last one. `/backup` makes one
//!   now; `/backup status` says when the last was.
//! - **How many are kept:** the newest 30; older ones are deleted.
//! - **Settings:** `S0MA_SHARE_BACKUP_HOURS` (0 turns them off),
//!   `S0MA_SHARE_BACKUP_KEEP`, `S0MA_SHARE_BACKUP_DIR`.
//!
//! A backup is written beside its final name and renamed into place, so a
//! file named `share-….tar.gz` is always whole.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const FIRST_AFTER: Duration = Duration::from_secs(5 * 60);
const DEFAULT_HOURS: u64 = 6;
const DEFAULT_KEEP: usize = 30;

/// The last backup made (or why the last try did not), and what the
/// folder held then.
#[derive(Default)]
struct Last {
    made: Option<(PathBuf, chrono::DateTime<chrono::Local>, usize)>,
    problem: Option<String>,
    listing: Vec<(String, String)>,
}

static LAST: Mutex<Option<Last>> = Mutex::new(None);
/// One backup at a time: the timer's and `/backup`'s.
static MAKING: Mutex<()> = Mutex::new(());

fn last<T>(read: impl FnOnce(&mut Last) -> T) -> T {
    let mut last = LAST.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    read(last.get_or_insert_with(Last::default))
}

/// Where backups go: `~/s0ma/backups`, or `S0MA_SHARE_BACKUP_DIR`.
pub fn backup_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("S0MA_SHARE_BACKUP_DIR").filter(|dir| !dir.is_empty()) {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join("s0ma").join("backups")
}

fn every() -> Option<Duration> {
    let hours = std::env::var("S0MA_SHARE_BACKUP_HOURS").ok().and_then(|value| value.trim().parse::<f64>().ok()).unwrap_or(DEFAULT_HOURS as f64);
    (hours > 0.0).then(|| Duration::from_secs_f64(hours * 3600.0))
}

fn keep() -> usize {
    std::env::var("S0MA_SHARE_BACKUP_KEEP").ok().and_then(|value| value.trim().parse().ok()).filter(|keep| *keep > 0).unwrap_or(DEFAULT_KEEP)
}

/// Once, in the agent that runs the turns: the timer.
pub fn start() {
    if cfg!(test) {
        return;
    }
    let Some(every) = every() else { return };
    let _ = std::thread::Builder::new().name("share-backup".into()).spawn(move || {
        std::thread::sleep(FIRST_AFTER);
        loop {
            let started = Instant::now();
            match make(&crate::share::receive_dir(), &backup_dir(), keep(), false) {
                Ok(Some(made)) => crate::terminal::say_note(&format!("[backup] {made}")),
                Ok(None) => {}
                Err(error) => crate::terminal::warn_status(&format!("[backup] the shared folder was not backed up: {error}")),
            }
            std::thread::sleep(every.saturating_sub(started.elapsed()));
        }
    });
}

/// A backup of `root` into `into`, unless nothing changed since the last
/// (`force` makes one anyway); the newest `keep` kept. What was made.
pub fn make(root: &Path, into: &Path, keep: usize, force: bool) -> Result<Option<String>, String> {
    let _one = MAKING.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if !root.is_dir() {
        let problem = format!("{} is not there (is the share sync on?)", crate::share::sync::home_short(root));
        last(|last| last.problem = Some(problem.clone()));
        return Err(problem);
    }
    let listing = crate::share::sync::listing(root);
    if !force && last(|last| last.made.is_some() && last.listing == listing) {
        return Ok(None);
    }
    let result = write_tar(root, into, &listing).inspect(|_| prune(into, keep));
    match result {
        Ok(path) => {
            let size = std::fs::metadata(&path).map(|metadata| metadata.len()).unwrap_or(0);
            let made = format!(
                "{} files backed up to {} ({})",
                listing.len(),
                crate::share::sync::home_short(&path),
                crate::share::human_size(size)
            );
            let count = listing.len();
            last(|last| {
                last.made = Some((path, chrono::Local::now(), count));
                last.problem = None;
                last.listing = listing;
            });
            Ok(Some(made))
        }
        Err(error) => {
            last(|last| last.problem = Some(error.clone()));
            Err(error)
        }
    }
}

/// The tar of `listing` (paths inside `root`), gzipped: written beside its
/// name, then renamed to it.
fn write_tar(root: &Path, into: &Path, listing: &[(String, String)]) -> Result<PathBuf, String> {
    std::fs::create_dir_all(into).map_err(|error| format!("{} could not be made: {error}", into.display()))?;
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let mut path = into.join(format!("share-{stamp}.tar.gz"));
    let mut n = 2;
    while path.exists() {
        path = into.join(format!("share-{stamp}-{n}.tar.gz"));
        n += 1;
    }
    let partial = into.join(format!(".{}.partial", path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()));
    // The names, one per NUL, read by tar: any name works, spaces and all.
    let list = into.join(".share-backup.list");
    let mut names = std::fs::File::create(&list).map_err(|error| error.to_string())?;
    for (file, _) in listing {
        names.write_all(file.as_bytes()).and_then(|()| names.write_all(b"\0")).map_err(|error| error.to_string())?;
    }
    drop(names);
    let ran = std::process::Command::new("tar")
        .arg("-czf")
        .arg(&partial)
        .arg("-C")
        .arg(root)
        .arg("--null")
        .arg("--no-recursion")
        .arg("-T")
        .arg(&list)
        .output()
        .map_err(|error| format!("tar could not run: {error}"));
    let _ = std::fs::remove_file(&list);
    let ran = ran?;
    // A file gone between the listing and tar (the agent deleted it) is not a
    // failure: tar exits 1 for it, and what is there is in the backup.
    if !ran.status.success() && ran.status.code() != Some(1) {
        let _ = std::fs::remove_file(&partial);
        return Err(format!("tar failed: {}", String::from_utf8_lossy(&ran.stderr).trim()));
    }
    std::fs::rename(&partial, &path).map_err(|error| {
        let _ = std::fs::remove_file(&partial);
        error.to_string()
    })?;
    Ok(path)
}

/// The newest `keep` backups kept; the rest deleted.
fn prune(into: &Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(into) else { return };
    let mut backups: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("share-") && name.ends_with(".tar.gz")))
        .collect();
    // Oldest first: by when each was written (two in one second share the
    // time in their names).
    backups.sort_by_key(|path| (std::fs::metadata(path).and_then(|metadata| metadata.modified()).ok(), path.clone()));
    let drop = backups.len().saturating_sub(keep);
    for old in &backups[..drop] {
        let _ = std::fs::remove_file(old);
    }
}

/// `/backup` (one now), `/backup status`; `None` for anything else.
pub fn handle_command(command: &str) -> Option<Result<String, String>> {
    let words: Vec<&str> = command.split_whitespace().collect();
    match words.as_slice() {
        // On a thread of its own: a large folder takes a while to tar.
        ["/backup"] | ["/backup", "now"] => Some(
            std::thread::Builder::new()
                .name("share-backup-now".into())
                .spawn(|| match make(&crate::share::receive_dir(), &backup_dir(), keep(), true) {
                    Ok(Some(made)) => crate::terminal::say_done(&format!("[backup] {made}")),
                    Ok(None) => crate::terminal::say_status("[backup] nothing to back up"),
                    Err(error) => crate::terminal::warn_status(&format!("[backup] not made: {error}")),
                })
                .map(|_| format!("[backup] backing up {}…", crate::share::sync::home_short(&crate::share::receive_dir())))
                .map_err(|error| format!("[backup] could not start: {error}")),
        ),
        ["/backup", "status"] => Some(Ok(last(|last| {
            let when = match every() {
                Some(every) => format!("every {}h, when something changed", every.as_secs_f64() / 3600.0),
                None => "off (S0MA_SHARE_BACKUP_HOURS=0)".into(),
            };
            let made = match &last.made {
                Some((path, at, count)) => format!("last: {} files at {}, {}", count, at.format("%b %-d %H:%M"), crate::share::sync::home_short(path)),
                None => "none made yet this run".into(),
            };
            let problem = last.problem.as_ref().map(|problem| format!("; last try failed: {problem}")).unwrap_or_default();
            format!("[backup] {when}, the newest {} kept in {}; {made}{problem}", keep(), crate::share::sync::home_short(&backup_dir()))
        }))),
        ["/backup", ..] => Some(Err("usage: /backup [now|status]".into())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backup_holds_the_shared_files_and_only_the_newest_are_kept() {
        let base = std::env::temp_dir().join(format!("s0ma-backup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("s0ma-share");
        let into = base.join("backups");
        std::fs::create_dir_all(root.join("notes/deep")).unwrap();
        std::fs::create_dir_all(root.join(".s0ma-deleted")).unwrap();
        std::fs::write(root.join("plan.md"), "plan").unwrap();
        std::fs::write(root.join("notes/deep/with space.md"), "deep").unwrap();
        std::fs::write(root.join(".s0ma-deleted/old.md"), "gone").unwrap();
        *LAST.lock().unwrap() = None;

        let made = make(&root, &into, 2, false).unwrap().expect("made");
        assert!(made.starts_with("2 files backed up to"), "{made}");
        let first = last(|last| last.made.as_ref().unwrap().0.clone());
        let listed = std::process::Command::new("tar").arg("-tzf").arg(&first).output().unwrap();
        let listed = String::from_utf8_lossy(&listed.stdout).to_string();
        assert!(listed.contains("plan.md") && listed.contains("notes/deep/with space.md"), "{listed}");
        assert!(!listed.contains("old.md"), "hidden ones left out: {listed}");

        // Nothing changed: none made; changed, or asked: one more.
        assert_eq!(make(&root, &into, 2, false).unwrap(), None);
        std::fs::write(root.join("plan.md"), "plan, changed").unwrap();
        assert!(make(&root, &into, 2, false).unwrap().is_some());
        assert!(make(&root, &into, 2, true).unwrap().is_some());
        let kept: Vec<String> = std::fs::read_dir(&into).unwrap().filter_map(Result::ok).map(|entry| entry.file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(kept.iter().filter(|name| name.ends_with(".tar.gz")).count(), 2, "the newest 2: {kept:?}");
        assert!(!kept.iter().any(|name| name.ends_with(".partial") || name.ends_with(".list")), "{kept:?}");
        assert!(!into.join(first.file_name().unwrap()).exists(), "the oldest deleted");

        // No folder to back up: said.
        assert!(make(&base.join("nowhere"), &into, 2, true).unwrap_err().contains("is not there"));
        let _ = std::fs::remove_dir_all(&base);
    }
}
