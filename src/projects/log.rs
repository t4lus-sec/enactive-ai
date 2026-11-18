//! The files a project is made of (`.board`, `.audit`, `.folder`): JSON
//! Lines, a header first, then one change per line. Every line after the
//! header carries its own id, `n` (`<writer>.<number>`), so two copies of a
//! file can always be merged line by line. Lines are only ever added (under
//! the file's lock, synced to disk), except when a long file is folded into
//! a snapshot or text is struck from it, both written whole beside it and
//! renamed into place.
//!
//! A line cut short (a crash mid-write, or a copy taken while it was
//! written) is skipped on reading, and so is any line that is not JSON:
//! the rest of the file still reads.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Map, Value};

pub type Result<T> = std::result::Result<T, String>;

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// A file read whole: its header, its lines in file order, and how many
/// lines could not be read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Log {
    pub header: Map<String, Value>,
    pub lines: Vec<Map<String, Value>>,
    pub skipped: usize,
}

impl Log {
    pub fn id(&self) -> &str {
        self.header.get("id").and_then(Value::as_str).unwrap_or("")
    }

    pub fn kind(&self) -> &str {
        self.header.get("s0ma").and_then(Value::as_str).unwrap_or("")
    }

    /// The lines that make up the state now: the last snapshot's (as one
    /// line) and those after it.
    pub fn current(&self) -> &[Map<String, Value>] {
        let from = self.lines.iter().rposition(|line| op(line) == "snapshot").unwrap_or(0);
        &self.lines[from..]
    }
}

pub fn op(line: &Map<String, Value>) -> &str {
    line.get("op").and_then(Value::as_str).unwrap_or("")
}

pub fn n(line: &Map<String, Value>) -> &str {
    line.get("n").and_then(Value::as_str).unwrap_or("")
}

/// This process's name on the lines it writes.
pub fn writer() -> String {
    format!("h{}", std::process::id())
}

/// Line numbers count up from the time in microseconds, so a restart never
/// reuses one, and never go back.
fn next_number() -> u64 {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_micros() as u64).unwrap_or(0);
    let mut last = LAST.load(Ordering::SeqCst);
    loop {
        let next = now.max(last + 1);
        match LAST.compare_exchange(last, next, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return next,
            Err(seen) => last = seen,
        }
    }
}

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// The file's lock, held while it lives: exclusive to write, shared to read.
pub struct Locked {
    pub file: File,
}

impl Locked {
    pub fn exclusive(path: &Path) -> Result<Self> {
        // A rewrite renames a new file into place while holding the old
        // one's lock: whoever waited on that lock opens the path again.
        for _ in 0..100 {
            let file = OpenOptions::new().read(true).append(true).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            lock(&file, libc::LOCK_EX)?;
            if same_file(&file, path) {
                return Ok(Self { file });
            }
        }
        Err(format!("{}: kept being replaced", path.display()))
    }

    pub fn shared(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        lock(&file, libc::LOCK_SH)?;
        Ok(Self { file })
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

fn same_file(file: &File, path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (file.metadata(), std::fs::metadata(path)) {
        (Ok(a), Ok(b)) => a.ino() == b.ino() && a.dev() == b.dev(),
        _ => false,
    }
}

fn lock(file: &File, how: i32) -> Result<()> {
    loop {
        let done = unsafe { libc::flock(file.as_raw_fd(), how) };
        if done == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(format!("lock: {error}"));
        }
    }
}

/// A new file: its header alone. Refused when the path is taken.
pub fn create(path: &Path, header: &Map<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(err)?;
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut text = serde_json::to_string(&Value::Object(header.clone())).map_err(err)?;
    text.push('\n');
    file.write_all(text.as_bytes()).map_err(err)?;
    file.sync_all().map_err(err)?;
    Ok(())
}

/// Lines added at the end under the file's lock, each given its `n` and
/// `at` when it has none, and synced to disk. Returns the lines as written.
pub fn append(path: &Path, lines: Vec<Map<String, Value>>) -> Result<Vec<Map<String, Value>>> {
    if lines.is_empty() {
        return Ok(lines);
    }
    let locked = Locked::exclusive(path)?;
    append_locked(&locked, lines)
}

pub fn append_locked(locked: &Locked, lines: Vec<Map<String, Value>>) -> Result<Vec<Map<String, Value>>> {
    let mut file = &locked.file;
    // A line cut short by a crash is ended first, so the new one starts on
    // its own line (the cut one stays unreadable, and skipped).
    let length = file.metadata().map_err(err)?.len();
    let mut text = String::new();
    if length > 0 {
        let mut last = [0u8; 1];
        let mut reader = &locked.file;
        reader.seek(SeekFrom::Start(length - 1)).map_err(err)?;
        reader.read_exact(&mut last).map_err(err)?;
        if last[0] != b'\n' {
            text.push('\n');
        }
    }
    let writer = writer();
    let at = now();
    let mut written = Vec::new();
    for mut line in lines {
        if !line.contains_key("n") {
            line.insert("n".into(), Value::String(format!("{writer}.{}", next_number())));
        }
        if !line.contains_key("at") {
            line.insert("at".into(), Value::String(at.clone()));
        }
        // `n` and `at` first, for the eye.
        let mut ordered = Map::new();
        for key in ["n", "at", "op"] {
            if let Some(value) = line.remove(key) {
                ordered.insert(key.into(), value);
            }
        }
        ordered.extend(line);
        text.push_str(&serde_json::to_string(&Value::Object(ordered.clone())).map_err(err)?);
        text.push('\n');
        written.push(ordered);
    }
    file.write_all(text.as_bytes()).map_err(err)?;
    file.sync_data().map_err(err)?;
    Ok(written)
}

/// The file read whole, under a shared lock.
pub fn read(path: &Path) -> Result<Log> {
    let locked = Locked::shared(path)?;
    read_from(&locked.file)
}

pub fn read_from(file: &File) -> Result<Log> {
    let mut reader = BufReader::new(file);
    reader.seek(SeekFrom::Start(0)).map_err(err)?;
    parse(reader)
}

pub fn parse(reader: impl BufRead) -> Result<Log> {
    let mut log = Log::default();
    let mut first = true;
    for line in reader.split(b'\n') {
        let line = line.map_err(err)?;
        let text = String::from_utf8_lossy(&line);
        if text.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) if first => {
                if !map.contains_key("s0ma") {
                    return Err("not an s0ma file: its first line is no header".into());
                }
                log.header = map;
                first = false;
            }
            Ok(Value::Object(map)) if map.contains_key("n") => log.lines.push(map),
            _ if first => return Err("not an s0ma file: its first line is no header".into()),
            _ => log.skipped += 1,
        }
    }
    if first {
        return Err("empty".into());
    }
    Ok(log)
}

/// Only the header: the first line.
pub fn header(path: &Path) -> Result<Map<String, Value>> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut first = String::new();
    BufReader::new(file).read_line(&mut first).map_err(err)?;
    match serde_json::from_str::<Value>(first.trim()) {
        Ok(Value::Object(map)) if map.contains_key("s0ma") => Ok(map),
        _ => Err("not an s0ma file: its first line is no header".into()),
    }
}

/// The file written whole, beside it and renamed into place, under its
/// lock (so no append lands in between).
pub fn rewrite_locked(path: &Path, _locked: &Locked, log: &Log) -> Result<()> {
    let text = to_text(log)?;
    let aside = temp_beside(path);
    {
        let mut file = File::create(&aside).map_err(err)?;
        file.write_all(text.as_bytes()).map_err(err)?;
        file.sync_all().map_err(err)?;
    }
    std::fs::rename(&aside, path).map_err(|e| {
        let _ = std::fs::remove_file(&aside);
        err(e)
    })
}

/// The file's text: the header, then a line each.
pub fn to_text(log: &Log) -> Result<String> {
    let mut text = serde_json::to_string(&Value::Object(log.header.clone())).map_err(err)?;
    text.push('\n');
    for line in &log.lines {
        text.push_str(&serde_json::to_string(&Value::Object(line.clone())).map_err(err)?);
        text.push('\n');
    }
    Ok(text)
}

fn temp_beside(path: &Path) -> PathBuf {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    path.with_file_name(format!(".{name}.{}.tmp", next_number()))
}

/// Two copies of one file as one: every line of both, each once (by its
/// `n`), in time order. Lines a later snapshot folded in are left out.
/// The header is `a`'s.
pub fn merge(a: &Log, b: &Log) -> Log {
    let mut seen = std::collections::HashSet::new();
    let mut lines: Vec<Map<String, Value>> = Vec::new();
    for line in a.lines.iter().chain(b.lines.iter()) {
        if seen.insert(n(line).to_string()) {
            lines.push(line.clone());
        }
    }
    let key = |line: &Map<String, Value>| {
        let at = line.get("at").and_then(Value::as_str).unwrap_or("").to_string();
        let (writer, number) = n(line).rsplit_once('.').unwrap_or(("", "0"));
        (at, writer.to_string(), number.parse::<u64>().unwrap_or(0))
    };
    lines.sort_by_key(|line| key(line));
    // The newest snapshot, and anything it folded in, before it or not.
    if let Some(at) = lines.iter().rposition(|line| op(line) == "snapshot") {
        let folded: Map<String, Value> = lines[at].get("folded").and_then(Value::as_object).cloned().unwrap_or_default();
        let snapshot = lines[at].clone();
        let mut kept = vec![snapshot.clone()];
        for (index, line) in lines.iter().enumerate() {
            if index == at || op(line) == "snapshot" {
                continue;
            }
            let (writer, number) = n(line).rsplit_once('.').unwrap_or(("", "0"));
            let number: u64 = number.parse().unwrap_or(0);
            let upto = folded.get(writer).and_then(Value::as_u64).unwrap_or(0);
            if number > upto {
                kept.push(line.clone());
            }
        }
        lines = kept;
    }
    Log { header: a.header.clone(), lines, skipped: 0 }
}

/// What each writer's lines in `lines` go up to: what a snapshot folds in
/// (for compacting a long file, not done yet).
#[allow(dead_code)]
pub fn folded(lines: &[Map<String, Value>]) -> Map<String, Value> {
    let mut out = Map::new();
    for line in lines {
        if op(line) == "snapshot" {
            if let Some(map) = line.get("folded").and_then(Value::as_object) {
                for (writer, upto) in map {
                    let upto = upto.as_u64().unwrap_or(0);
                    if out.get(writer).and_then(Value::as_u64).unwrap_or(0) < upto {
                        out.insert(writer.clone(), Value::from(upto));
                    }
                }
            }
            continue;
        }
        let (writer, number) = n(line).rsplit_once('.').unwrap_or(("", "0"));
        let number: u64 = number.parse().unwrap_or(0);
        if out.get(writer).and_then(Value::as_u64).unwrap_or(0) < number {
            out.insert(writer.into(), Value::from(number));
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("s0ma-test-{}-{}", std::process::id(), next_number()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn header(id: &str) -> Map<String, Value> {
        serde_json::json!({"s0ma":"board","v":1,"id":id}).as_object().unwrap().clone()
    }

    fn line(op: &str, text: &str) -> Map<String, Value> {
        serde_json::json!({"op":op,"text":text}).as_object().unwrap().clone()
    }

    #[test]
    fn lines_are_added_with_ids_and_a_cut_line_is_skipped() {
        let dir = scratch();
        let path = dir.join("a.board");
        create(&path, &header("b-1")).unwrap();
        assert!(create(&path, &header("b-1")).is_err(), "never over a file");
        append(&path, vec![line("lead", "one"), line("finding", "two")]).unwrap();
        // A crash mid-line, then another write.
        std::fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(b"{\"n\":\"x.1\",\"op\":\"le").unwrap();
        append(&path, vec![line("lead", "three")]).unwrap();
        let log = read(&path).unwrap();
        assert_eq!(log.id(), "b-1");
        assert_eq!(log.lines.iter().map(|l| l["text"].as_str().unwrap()).collect::<Vec<_>>(), ["one", "two", "three"]);
        assert_eq!(log.skipped, 1);
        let ids: std::collections::HashSet<&str> = log.lines.iter().map(n).collect();
        assert_eq!(ids.len(), 3, "each line its own id");
    }

    #[test]
    fn two_writers_at_once_lose_nothing() {
        let dir = scratch();
        let path = dir.join("a.board");
        create(&path, &header("b-1")).unwrap();
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for i in 0..50 {
                        append(&path, vec![line("lead", &format!("{t}-{i}"))]).unwrap();
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        let log = read(&path).unwrap();
        assert_eq!((log.lines.len(), log.skipped), (200, 0));
    }

    #[test]
    fn two_copies_merge_line_by_line() {
        let dir = scratch();
        let path = dir.join("a.board");
        create(&path, &header("b-1")).unwrap();
        append(&path, vec![line("lead", "both")]).unwrap();
        let mut a = read(&path).unwrap();
        let mut b = a.clone();
        a.lines.extend(append(&path, vec![line("lead", "only a")]).unwrap());
        b.lines.push(serde_json::json!({"n":"h9.1","at":"2099-01-01T00:00:00.000Z","op":"lead","text":"only b"}).as_object().unwrap().clone());
        let merged = merge(&a, &b);
        assert_eq!(merged.lines.iter().map(|l| l["text"].as_str().unwrap()).collect::<Vec<_>>(), ["both", "only a", "only b"]);
        assert_eq!(merge(&merged, &a), merged, "merging again changes nothing");
        // A snapshot folds the lines before it in; a line it hadn't seen stays.
        let mut snap = merged.clone();
        let folded = folded(&snap.lines[..2]);
        snap.lines = vec![serde_json::json!({"n":"h1.99999999999999","at":"2098-01-01T00:00:00.000Z","op":"snapshot","folded":folded}).as_object().unwrap().clone()];
        let after = merge(&snap, &b);
        assert_eq!(after.lines.len(), 2, "{:?}", after.lines);
        assert_eq!(op(&after.lines[0]), "snapshot");
        assert_eq!(after.lines[1]["text"], "only b");
        assert_eq!(after.current().len(), 2);
    }

    #[test]
    fn a_rewrite_is_whole_or_not_at_all() {
        let dir = scratch();
        let path = dir.join("a.board");
        create(&path, &header("b-1")).unwrap();
        append(&path, vec![line("lead", "one")]).unwrap();
        let mut log = read(&path).unwrap();
        log.lines[0].insert("text".into(), "struck".into());
        rewrite_locked(&path, &Locked::exclusive(&path).unwrap(), &log).unwrap();
        assert_eq!(read(&path).unwrap().lines[0]["text"], "struck");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "nothing left beside it");
    }
}
