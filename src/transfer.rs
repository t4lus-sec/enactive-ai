//! Moving files between the share and the machine a terminal runs on, over
//! the attach protocol (`session.rs`): `/download` and `/upload`, typed on a
//! terminal attached with `s0ma connect` (or locally).
//!
//! - `/download <file in the share> …` fetches each file a part at a time and
//!   saves it in this machine's Downloads (`S0MA_DOWNLOAD_DIR`), under a free
//!   name: nothing there is replaced.
//! - `/upload <path on this machine> … [--to <folder in the share>]` sends
//!   each file a part at a time into the share (`~/s0ma-share` on the server,
//!   which the sync keeps in step with `/share`). It is written beside its
//!   name and put in place only when whole, under a free name.
//!
//! The primary's side (`read_part`, `write_part`) only ever touches the
//! share: a path is taken apart, and one with `..`, a hidden name, or a
//! symlink on the way is refused, so neither a typed path nor a crafted
//! request reaches past it.

use base64::Engine;
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Bytes a part carries: one request and answer each.
pub const PART_BYTES: usize = 512 * 1024;

/// Part of a file, as the primary answers it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Part {
    /// Base64.
    pub data: String,
    /// The file's whole size.
    pub size: u64,
    pub eof: bool,
}

/// What a command asks of the primary, through the terminal's connection.
pub enum Request {
    Fetch { path: String, offset: u64 },
    Put { upload: String, name: String, offset: u64, data: String, done: bool },
    /// Not for the primary: a line for this terminal's progress.
    Progress(String),
}

pub enum Reply {
    Part(Part),
    /// Where an upload landed, in the share.
    Saved(String),
    Noted,
}

pub fn is_command(text: &str) -> bool {
    let first = text.split_whitespace().next().unwrap_or_default();
    first == "/download" || first == "/upload"
}

// -------------------------------------------------------------- the primary

/// `path` inside `root`, taken apart and checked: no `..`, no hidden name,
/// no symlink on the way. A leading `/share/` or `~/s0ma-share/` is the
/// share itself.
fn inside(root: &Path, path: &str) -> Result<PathBuf, String> {
    let trimmed = path.trim();
    let relative = ["/share/", "~/s0ma-share/", "s0ma-share/", "/share", "~/s0ma-share"]
        .iter()
        .find_map(|prefix| trimmed.strip_prefix(prefix))
        .unwrap_or(trimmed)
        .trim_start_matches('/');
    let mut at = root.to_path_buf();
    for part in relative.split('/').filter(|p| !p.is_empty()) {
        if part == "." || part == ".." || part.starts_with('.') {
            return Err(format!("{path}: only files in the share, by their own names (no `..` or hidden names)"));
        }
        at.push(part);
        if std::fs::symlink_metadata(&at).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(format!("{path}: a link; only files in the share itself"));
        }
    }
    Ok(at)
}

/// From the share: `PART_BYTES` of `path` from `offset`.
pub fn read_part(root: &Path, path: &str, offset: u64) -> Result<Part, String> {
    let file = inside(root, path)?;
    let metadata = std::fs::metadata(&file).map_err(|_| format!("{path}: not in the share"))?;
    if metadata.is_dir() {
        return Err(format!("{path}: a folder; download its files"));
    }
    let size = metadata.len();
    let mut opened = std::fs::File::open(&file).map_err(|error| format!("{path}: {error}"))?;
    opened.seek(SeekFrom::Start(offset.min(size))).map_err(|error| error.to_string())?;
    let mut bytes = Vec::with_capacity(PART_BYTES.min(size as usize));
    opened.take(PART_BYTES as u64).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
    let eof = offset + bytes.len() as u64 >= size;
    Ok(Part { data: base64::engine::general_purpose::STANDARD.encode(&bytes), size, eof })
}

/// `name`, or `name (2)`, `name (3)` … beside it: the first not taken.
fn free(path: &Path) -> PathBuf {
    if std::fs::symlink_metadata(path).is_err() {
        return path.to_path_buf();
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let (stem, extension) = crate::share::split_name(&name);
    (2..)
        .map(|n| path.with_file_name(format!("{stem} ({n}){extension}")))
        .find(|candidate| std::fs::symlink_metadata(candidate).is_err())
        .expect("a free name")
}

/// `partial` put in place as `target`, or `target (2)` … when taken: linked
/// to the name first (which fails when the name is taken, where a rename
/// would replace what took it meanwhile), then the partial removed. Where
/// links can't be made, renamed to the free name. Where it landed.
fn land(partial: &Path, target: &Path) -> std::io::Result<PathBuf> {
    for _ in 0..1000 {
        let landed = free(target);
        match std::fs::hard_link(partial, &landed) {
            Ok(()) => {
                let _ = std::fs::remove_file(partial);
                return Ok(landed);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => {
                std::fs::rename(partial, &landed)?;
                return Ok(landed);
            }
        }
    }
    Err(std::io::Error::other("no free name"))
}

/// Uploads that never finished (`.upload-*.partial`) in `folder`, a day old
/// or more: removed when another starts there.
fn clear_stale_partials(folder: &Path) {
    let Ok(entries) = std::fs::read_dir(folder) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let old = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .is_ok_and(|at| at.elapsed().unwrap_or_default() >= std::time::Duration::from_secs(24 * 3600));
        if name.starts_with(".upload-") && name.ends_with(".partial") && old {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Into the share: part of an upload. `upload` names it (letters and digits),
/// so its parts go to one file beside `name`, which takes its place, under a
/// free name, at `done`. What it landed as, in the share, or `""` before.
pub fn write_part(root: &Path, upload: &str, name: &str, offset: u64, data: &str, done: bool) -> Result<String, String> {
    if upload.len() < 8 || upload.len() > 64 || !upload.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err("a bad upload id".into());
    }
    let target = inside(root, name)?;
    if name.trim().trim_end_matches('/').is_empty() || target == root {
        return Err("an upload needs a file name".into());
    }
    let folder = target.parent().ok_or("no folder")?.to_path_buf();
    if !folder.is_dir() {
        return Err(format!("{}: no such folder in the share", folder.strip_prefix(root).unwrap_or(&folder).display()));
    }
    let partial = folder.join(format!(".upload-{upload}.partial"));
    let bytes = base64::engine::general_purpose::STANDARD.decode(data).map_err(|_| "a part that isn't base64".to_string())?;
    let mut file = if offset == 0 {
        clear_stale_partials(&folder);
        std::fs::File::create(&partial)
    } else {
        std::fs::OpenOptions::new().append(true).open(&partial)
    }
    .map_err(|error| format!("{name}: {error}"))?;
    let have = file.metadata().map(|m| m.len()).unwrap_or(0);
    if have != offset {
        let _ = std::fs::remove_file(&partial);
        return Err(format!("{name}: a part out of order ({have} bytes written, this one at {offset}); send it again"));
    }
    file.write_all(&bytes).map_err(|error| {
        let _ = std::fs::remove_file(&partial);
        format!("{name}: {error}")
    })?;
    if !done {
        return Ok(String::new());
    }
    file.sync_all().map_err(|error| error.to_string())?;
    drop(file);
    let landed = land(&partial, &target).map_err(|error| {
        let _ = std::fs::remove_file(&partial);
        format!("{name}: {error}")
    })?;
    crate::terminal::say_status(&format!(
        "[share] uploaded {} ({})",
        landed.strip_prefix(root).unwrap_or(&landed).display(),
        crate::share::human_size(offset + bytes.len() as u64)
    ));
    Ok(landed.strip_prefix(root).unwrap_or(&landed).to_string_lossy().into_owned())
}

// --------------------------------------------------------------- a terminal

/// Words of a command line, `"…"` and `'…'` keeping spaces in a path.
fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut quote: Option<char> = None;
    let mut any = false;
    for c in text.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => word.push(c),
            (None, '"' | '\'') => {
                quote = Some(c);
                any = true;
            }
            (None, c) if c.is_whitespace() => {
                if any || !word.is_empty() {
                    out.push(std::mem::take(&mut word));
                }
                any = false;
            }
            (None, c) => {
                word.push(c);
                any = true;
            }
        }
    }
    if any || !word.is_empty() {
        out.push(word);
    }
    out
}

/// Where downloads go on this machine.
fn downloads() -> PathBuf {
    if let Some(dir) = std::env::var_os("S0MA_DOWNLOAD_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(".")).join("Downloads")
}

fn home(path: &str) -> PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join(rest),
        None => PathBuf::from(path),
    }
}

/// A random id for one upload.
fn upload_id() -> String {
    let mut bytes = [0u8; 12];
    if std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut bytes)).is_err() {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        bytes[..8].copy_from_slice(&(now as u64 ^ std::process::id() as u64).to_le_bytes());
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `/download …` or `/upload …`, run through `ask`. What to tell the
/// operator, or why it failed (a failure of one file is said, and the rest
/// still go).
pub fn run_command(text: &str, ask: &mut dyn FnMut(Request) -> Result<Reply, String>) -> Result<String, String> {
    run_command_into(text, &downloads(), ask)
}

/// `run_command`, saving downloads into `into`.
fn run_command_into(text: &str, into: &Path, ask: &mut dyn FnMut(Request) -> Result<Reply, String>) -> Result<String, String> {
    let words = words(text);
    let Some((command, rest)) = words.split_first() else { return Err("nothing to do".into()) };
    match command.as_str() {
        "/download" if rest.is_empty() => Err("usage: /download <file in the share> … (saved in ~/Downloads)".into()),
        "/download" => {
            let into = into.to_path_buf();
            std::fs::create_dir_all(&into).map_err(|error| format!("{}: {error}", into.display()))?;
            let results: Vec<Result<String, String>> = rest.iter().map(|path| download(path, &into, ask)).collect();
            report(results)
        }
        "/upload" => {
            let (paths, to) = match rest.iter().position(|w| w == "--to") {
                Some(at) => (&rest[..at], rest.get(at + 1).cloned().unwrap_or_default()),
                None => (rest, String::new()),
            };
            if paths.is_empty() {
                return Err("usage: /upload <file on this machine> … [--to <folder in the share>]".into());
            }
            let results: Vec<Result<String, String>> = paths.iter().map(|path| upload(path, &to, ask)).collect();
            report(results)
        }
        _ => Err("not a transfer command".into()),
    }
}

fn report(results: Vec<Result<String, String>>) -> Result<String, String> {
    let failed = results.iter().filter(|r| r.is_err()).count();
    let lines: Vec<String> = results.into_iter().map(|r| r.unwrap_or_else(|e| format!("not moved: {e}"))).collect();
    if failed == lines.len() { Err(lines.join("\n")) } else { Ok(lines.join("\n")) }
}

fn download(path: &str, into: &Path, ask: &mut dyn FnMut(Request) -> Result<Reply, String>) -> Result<String, String> {
    let name = path.trim_end_matches('/').rsplit('/').next().unwrap_or(path).to_string();
    if name.is_empty() {
        return Err(format!("{path}: no file name"));
    }
    // Its own partial: another download of the same name at once (another
    // terminal) writes to another.
    static DOWNLOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = DOWNLOADS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let partial = into.join(format!(".{name}.{}-{n}.s0ma-download", std::process::id()));
    let mut file = std::fs::File::create(&partial).map_err(|error| format!("{}: {error}", partial.display()))?;
    let mut offset = 0u64;
    let outcome = loop {
        let part = match ask(Request::Fetch { path: path.to_string(), offset }) {
            Ok(Reply::Part(part)) => part,
            Ok(_) => break Err("s0ma answered something else".to_string()),
            Err(error) => break Err(error),
        };
        let bytes = match base64::engine::general_purpose::STANDARD.decode(&part.data) {
            Ok(bytes) => bytes,
            Err(_) => break Err("a part that isn't base64".into()),
        };
        if let Err(error) = file.write_all(&bytes) {
            break Err(error.to_string());
        }
        offset += bytes.len() as u64;
        if part.eof || bytes.is_empty() {
            break Ok(part.size);
        }
        let _ = ask(Request::Progress(format!(
            "downloading {name}: {} of {}",
            crate::share::human_size(offset),
            crate::share::human_size(part.size)
        )));
    };
    match outcome {
        Ok(size) => {
            file.sync_all().map_err(|error| error.to_string())?;
            drop(file);
            let landed = land(&partial, &into.join(&name)).map_err(|error| {
                let _ = std::fs::remove_file(&partial);
                error.to_string()
            })?;
            Ok(format!("downloaded {name} ({}) to {}", crate::share::human_size(size), crate::share_sync::home_short(&landed)))
        }
        Err(error) => {
            let _ = std::fs::remove_file(&partial);
            // The primary's errors name the file already.
            if error.starts_with(&format!("{path}:")) { Err(error) } else { Err(format!("{path}: {error}")) }
        }
    }
}

fn upload(path: &str, to: &str, ask: &mut dyn FnMut(Request) -> Result<Reply, String>) -> Result<String, String> {
    let local = home(path);
    let metadata = std::fs::metadata(&local).map_err(|error| format!("{path}: {error}"))?;
    if metadata.is_dir() {
        return Err(format!("{path}: a folder; upload its files (or a .tar.gz of it)"));
    }
    let name = local.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or_else(|| format!("{path}: no file name"))?;
    let destination = if to.trim_matches('/').is_empty() { name.clone() } else { format!("{}/{name}", to.trim_matches('/')) };
    let mut file = std::fs::File::open(&local).map_err(|error| format!("{path}: {error}"))?;
    let size = metadata.len();
    let id = upload_id();
    let mut offset = 0u64;
    let mut buffer = vec![0u8; PART_BYTES];
    loop {
        let mut read = 0;
        while read < buffer.len() {
            match file.read(&mut buffer[read..]) {
                Ok(0) => break,
                Ok(n) => read += n,
                Err(error) => return Err(format!("{path}: {error}")),
            }
        }
        let done = offset + read as u64 >= size || read < buffer.len();
        let data = base64::engine::general_purpose::STANDARD.encode(&buffer[..read]);
        match ask(Request::Put { upload: id.clone(), name: destination.clone(), offset, data, done })? {
            Reply::Saved(landed) if done => {
                return Ok(format!("uploaded {name} ({}) to the share as {landed}", crate::share::human_size(offset + read as u64)));
            }
            _ => {}
        }
        offset += read as u64;
        if done {
            return Err(format!("{path}: s0ma never said where it landed"));
        }
        let _ = ask(Request::Progress(format!(
            "uploading {name}: {} of {}",
            crate::share::human_size(offset),
            crate::share::human_size(size)
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("s0ma-transfer-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The primary's side, called straight from a terminal's: what the
    /// protocol would carry between them.
    fn through(root: &Path) -> impl FnMut(Request) -> Result<Reply, String> + '_ {
        move |request| match request {
            Request::Fetch { path, offset } => read_part(root, &path, offset).map(Reply::Part),
            Request::Put { upload, name, offset, data, done } => {
                write_part(root, &upload, &name, offset, &data, done).map(Reply::Saved)
            }
            Request::Progress(_) => Ok(Reply::Noted),
        }
    }

    #[test]
    fn a_file_goes_down_and_up_whole_in_parts_and_never_replaces_one() {
        let base = fixture("roundtrip");
        let share = base.join("share");
        let local = base.join("laptop");
        std::fs::create_dir_all(share.join("notes")).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        // Three parts and a bit.
        let big: Vec<u8> = (0..PART_BYTES * 3 + 1234).map(|i| (i % 251) as u8).collect();
        std::fs::write(share.join("notes/big file.bin"), &big).unwrap();
        let downloads = local.join("Downloads");
        let said = run_command_into("/download \"/share/notes/big file.bin\"", &downloads, &mut through(&share)).unwrap();
        assert!(said.starts_with("downloaded big file.bin"), "{said}");
        assert_eq!(std::fs::read(local.join("Downloads/big file.bin")).unwrap(), big);
        // Again: a free name beside it, the first kept.
        run_command_into("/download 'notes/big file.bin'", &downloads, &mut through(&share)).unwrap();
        assert!(local.join("Downloads/big file (2).bin").exists());
        // Up: whole, into a folder, under a free name; no partial left.
        std::fs::write(local.join("plan.md"), "the plan").unwrap();
        let said = run_command_into(&format!("/upload {} --to notes", local.join("plan.md").display()), &downloads, &mut through(&share)).unwrap();
        assert!(said.contains("as notes/plan.md"), "{said}");
        let said = run_command_into(&format!("/upload {}", local.join("plan.md").display()), &downloads, &mut through(&share)).unwrap();
        assert!(said.contains("as plan.md"), "{said}");
        run_command_into(&format!("/upload {}", local.join("plan.md").display()), &downloads, &mut through(&share)).unwrap();
        assert_eq!(std::fs::read_to_string(share.join("plan (2).md")).unwrap(), "the plan");
        std::fs::write(local.join("big.bin"), &big).unwrap();
        run_command_into(&format!("/upload {}", local.join("big.bin").display()), &downloads, &mut through(&share)).unwrap();
        assert_eq!(std::fs::read(share.join("big.bin")).unwrap(), big);
        let leftovers: Vec<_> = std::fs::read_dir(&share).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with('.')).collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        // One file missing: said, and the rest still moved.
        let said = run_command_into("/download notes/plan.md nothing.txt", &downloads, &mut through(&share)).unwrap();
        assert!(said.contains("downloaded plan.md") && said.contains("not moved: nothing.txt"), "{said}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_file_lands_beside_one_that_took_its_name_and_stale_uploads_go() {
        let dir = std::env::temp_dir().join(format!("s0ma-land-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("report.md");
        std::fs::write(&target, "theirs").unwrap();
        let partial = dir.join(".p");
        std::fs::write(&partial, "mine").unwrap();
        let landed = land(&partial, &target).unwrap();
        assert_eq!(landed, dir.join("report (2).md"));
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "theirs", "never replaced");
        assert_eq!(std::fs::read_to_string(&landed).unwrap(), "mine");
        assert!(!partial.exists());
        // An upload begun a day ago and never finished goes when one starts.
        let stale = dir.join(".upload-oldoldold.partial");
        std::fs::write(&stale, "half").unwrap();
        let day_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(25 * 3600);
        std::fs::File::options().write(true).open(&stale).unwrap().set_modified(day_ago).unwrap();
        let fresh = dir.join(".upload-newnewnew.partial");
        std::fs::write(&fresh, "half").unwrap();
        clear_stale_partials(&dir);
        assert!(!stale.exists() && fresh.exists(), "only the stale one");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nothing_outside_the_share_is_read_or_written() {
        let base = fixture("outside");
        let share = base.join("share");
        std::fs::create_dir_all(&share).unwrap();
        std::fs::write(base.join("secret"), "no").unwrap();
        std::fs::write(share.join(".sync-state"), "no").unwrap();
        std::os::unix::fs::symlink(base.join("secret"), share.join("link")).unwrap();
        for path in ["../secret", "/share/../secret", ".sync-state", "link", "a/../../secret"] {
            assert!(read_part(&share, path, 0).is_err(), "{path} was read");
        }
        let data = base64::engine::general_purpose::STANDARD.encode("x");
        assert!(write_part(&share, "abcdefgh12", "../escaped", 0, &data, true).is_err());
        assert!(write_part(&share, "abcdefgh12", "link", 0, &data, true).is_err());
        assert!(write_part(&share, "../../x", "fine.txt", 0, &data, true).is_err(), "a bad upload id");
        assert!(write_part(&share, "abcdefgh12", "nofolder/x.txt", 0, &data, true).is_err());
        // A part out of order is refused, and nothing half-written stays.
        assert!(write_part(&share, "abcdefgh12", "late.txt", 5, &data, false).is_err());
        assert!(!base.join("escaped").exists());
        assert!(!share.join("late.txt").exists());
        assert!(is_command("/download x") && is_command("/upload") && !is_command("/downloads"));
        assert_eq!(words(r#"/upload "a b.txt" 'c d' e"#), ["/upload", "a b.txt", "c d", "e"]);
        let _ = std::fs::remove_dir_all(&base);
    }
}
