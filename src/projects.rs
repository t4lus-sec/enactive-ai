//! Projects as folders. The projects root is `projects/` in the shared
//! folder (on the operator's computer, `~/s0ma-share/projects/`; the share
//! sync keeps `/share/projects/` in step). A project is a folder directly
//! under it, folders nest, each folder's page is its `index.md`, a board is
//! a `.board` file and an audit an `.audit` file (`log.rs`).
//!
//! The board and audit stores' databases are an index of those files
//! (`journal.rs`): every change a store makes is written to its file, and a
//! file changed, moved, archived, trashed or gone outside the stores is
//! picked up by `scan`, every few seconds, and brought into the index.
//! Archived and trashed boards and audits are taken out of the index, so
//! nothing works on them, and come back from their files when restored.

pub mod journal;
pub mod log;
pub mod pages;
pub mod summary;
pub mod tools;
pub mod tree;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    Board,
    Audit,
}

impl Kind {
    pub fn word(self) -> &'static str {
        match self {
            Kind::Board => "board",
            Kind::Audit => "audit",
        }
    }

    pub fn ext(self) -> &'static str {
        match self {
            Kind::Board => "board",
            Kind::Audit => "audit",
        }
    }

    pub fn of_path(path: &Path) -> Option<Kind> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("board") => Some(Kind::Board),
            Some("audit") => Some(Kind::Audit),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------- where

static ROOT: Mutex<Option<PathBuf>> = Mutex::new(None);
static STORES: Mutex<Option<HashMap<journal::Store, PathBuf>>> = Mutex::new(None);

#[cfg(test)]
thread_local! {
    static TEST_ROOT: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
    static TEST_STORES: std::cell::RefCell<Option<HashMap<journal::Store, PathBuf>>> = const { std::cell::RefCell::new(None) };
}

/// The projects root, when there is one (`S0MA_PROJECTS=off`: none).
pub fn root() -> Option<PathBuf> {
    #[cfg(test)]
    {
        TEST_ROOT.with(|root| root.borrow().clone())
    }
    #[cfg(not(test))]
    {
        crate::util::locked(&ROOT).clone()
    }
}

#[cfg(test)]
pub fn set_for_test(root: Option<PathBuf>, data: &Path) {
    TEST_ROOT.with(|r| *r.borrow_mut() = root);
    let stores = HashMap::from([
        (journal::Store::Board, data.join("board.sqlite3")),
        (journal::Store::Flow, data.join("board_flow.sqlite3")),
        (journal::Store::Audit, data.join("audit.sqlite3")),
    ]);
    TEST_STORES.with(|s| *s.borrow_mut() = Some(stores));
    INDEX.with_test(|index| *index = Index::default());
}

pub fn store_path(store: journal::Store) -> Option<PathBuf> {
    #[cfg(test)]
    {
        TEST_STORES.with(|s| s.borrow().as_ref().and_then(|m| m.get(&store).cloned()))
    }
    #[cfg(not(test))]
    {
        crate::util::locked(&STORES).as_ref().and_then(|m| m.get(&store).cloned())
    }
}

fn data_dir() -> Option<PathBuf> {
    store_path(journal::Store::Board).and_then(|p| p.parent().map(Path::to_path_buf))
}

pub fn warn_once(text: &str) {
    static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut said = crate::util::locked(&SAID);
    if !said.iter().any(|s| s == text) {
        said.push(text.to_string());
        if !cfg!(test) {
            crate::terminal::warn_status(text);
        }
    }
}

// ---------------------------------------------------------------- the index of files

/// What the index knows of a board's or audit's file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    pub kind: Option<Kind>,
    pub size: u64,
    pub mtime: i128,
    /// In the stores' databases now (not archived, trashed or a copy).
    pub live: bool,
    pub archived: bool,
    /// Another file had this id first: not used until renamed or removed.
    pub copy_of: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Index {
    /// By path under the root.
    pub files: HashMap<String, Entry>,
}

/// The index, shared by the process (or by the test's thread).
struct Shared(Mutex<Option<Index>>);

impl Shared {
    fn with<T>(&self, f: impl FnOnce(&mut Index) -> T) -> T {
        #[cfg(test)]
        {
            thread_local! { static T: std::cell::RefCell<Index> = std::cell::RefCell::new(Index::default()); }
            let _ = &self.0;
            T.with(|index| f(&mut index.borrow_mut()))
        }
        #[cfg(not(test))]
        {
            f(crate::util::locked(&self.0).get_or_insert_with(Index::default))
        }
    }

    #[cfg(test)]
    fn with_test(&self, f: impl FnOnce(&mut Index)) {
        self.with(f)
    }
}

static INDEX: Shared = Shared(Mutex::new(None));

/// Writes and scans one at a time: a scan never takes the stores' own
/// write, half recorded, for a change made outside.
static WRITING: Mutex<()> = Mutex::new(());

fn index_file() -> Option<PathBuf> {
    data_dir().map(|d| d.join("projects-index.json"))
}

/// An attached terminal: the primary keeps the index (and scans); this
/// process reads it, and never writes it.
static ATTACHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn attached() -> bool {
    ATTACHED.load(std::sync::atomic::Ordering::Relaxed)
}

fn save_index() {
    if attached() {
        return;
    }
    let Some(path) = index_file() else { return };
    let text = INDEX.with(|index| serde_json::to_string(index).unwrap_or_default());
    let aside = path.with_extension("json.tmp");
    if std::fs::write(&aside, text).is_ok() {
        let _ = std::fs::rename(&aside, &path);
    }
}

fn load_index() {
    let Some(path) = index_file() else { return };
    if let Ok(text) = std::fs::read_to_string(&path)
        && let Ok(loaded) = serde_json::from_str::<Index>(&text)
    {
        INDEX.with(|index| *index = loaded);
    }
}

pub fn rel(path: &Path) -> Option<String> {
    let root = root()?;
    path.strip_prefix(&root).ok().map(|p| p.to_string_lossy().into_owned())
}

pub fn abs(rel: &str) -> Option<PathBuf> {
    root().map(|root| root.join(rel))
}

pub fn stamp(path: &Path) -> Option<(u64, i128)> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos() as i128;
    Some((meta.len(), mtime))
}

/// Where a board's or audit's file is (live or not, never a copy).
pub fn path_of(kind: Kind, id: &str) -> Option<PathBuf> {
    let found = INDEX.with(|index| index.files.iter().find(|(_, e)| e.kind == Some(kind) && e.id == id && e.copy_of.is_none()).map(|(p, _)| p.clone()));
    found.and_then(|rel| abs(&rel))
}

/// Whether a board or audit of this id is archived (itself, or with a
/// folder above it): out of the stores, its file still its own.
pub fn archived(kind: Kind, id: &str) -> bool {
    INDEX.with(|index| index.files.values().any(|e| e.kind == Some(kind) && e.id == id && e.copy_of.is_none() && !e.live))
}

// ---------------------------------------------------------------- writing

thread_local! {
    /// The folder the next new board or audit file goes in (the projects tab
    /// sets it before making one).
    static PLACE_NEXT: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

pub fn place_next_in(folder: Option<PathBuf>) {
    PLACE_NEXT.with(|p| *p.borrow_mut() = folder);
}

/// Lines a store's change made, into the file of the board or audit they
/// belong to (made, if it is new).
pub fn record(kind: Kind, owner: &str, lines: Vec<Map<String, Value>>) -> Result<(), String> {
    let Some(root) = root() else { return Ok(()) };
    let _writing = crate::util::locked(&WRITING);
    // Moved by hand since the last scan: found by its id, never made again
    // (two files would split the board).
    let path = match path_of(kind, owner).filter(|path| path.exists()).or_else(|| find_by_id(&root, kind, owner)) {
        Some(path) => path,
        None => {
            let path = place_new(&root, kind, owner, &lines);
            let mut header = Map::new();
            header.insert("s0ma".into(), Value::String(kind.word().into()));
            header.insert("v".into(), Value::from(1));
            header.insert("id".into(), Value::String(owner.into()));
            header.insert("created".into(), Value::String(log::now()));
            log::create(&path, &header)?;
            path
        }
    };
    // Moved between the lookup and the write: there, by its id.
    let path = match log::append(&path, lines.clone()) {
        Ok(_) => path,
        Err(_) if !path.exists() => {
            let found = find_by_id(&root, kind, owner).ok_or_else(|| format!("{} {owner}: its file is gone", kind.word()))?;
            log::append(&found, lines)?;
            found
        }
        Err(error) => return Err(error),
    };
    if let (Some(rel), Some((size, mtime))) = (rel(&path), stamp(&path)) {
        INDEX.with(|index| {
            let entry = index.files.entry(rel).or_default();
            entry.id = owner.to_string();
            entry.kind = Some(kind);
            entry.size = size;
            entry.mtime = mtime;
            entry.live = true;
        });
        save_index();
    }
    Ok(())
}

/// A board's or audit's file wherever it is now (not archived away, not in
/// the trash), by the id in its header; the index follows.
fn find_by_id(root: &Path, kind: Kind, id: &str) -> Option<PathBuf> {
    // Only a board or audit the index has had a file for: a new one has
    // none to find, and must never take another's that happens to bear its
    // id (one archived or trashed, from before ids were kept unique).
    let known = INDEX.with(|index| index.files.values().any(|e| e.kind == Some(kind) && e.id == id && e.copy_of.is_none()));
    if !known {
        return None;
    }
    let found = walk(root).into_iter().filter(|(_, hidden)| !hidden).map(|(path, _)| path).find(|path| {
        Kind::of_path(path) == Some(kind) && log::header(path).is_ok_and(|header| header.get("id").and_then(Value::as_str) == Some(id))
    })?;
    if let Some(rel_path) = rel(&found) {
        INDEX.with(|index| {
            let old = index.files.iter().find(|(_, e)| e.kind == Some(kind) && e.id == id && e.copy_of.is_none()).map(|(p, e)| (p.clone(), e.clone()));
            if let Some((was, entry)) = old {
                index.files.remove(&was);
                index.files.insert(rel_path, entry);
            }
        });
    }
    Some(found)
}

/// Every row's highest `seq` in the files, by table: archived and trashed
/// boards' too, whose rows are out of the index. What a store's numbering
/// starts after, so no id is ever given twice.
pub fn highest_seqs() -> HashMap<String, i64> {
    // Only read when a table has no mark yet; the mark takes over after that.
    let mut highest: HashMap<String, i64> = HashMap::new();
    let Some(root) = root() else { return highest };
    let mut note = |table: &str, row: &Map<String, Value>| {
        if let Some(seq) = row.get("seq").and_then(Value::as_i64) {
            let at = highest.entry(table.to_string()).or_insert(0);
            *at = (*at).max(seq);
        }
    };
    for path in every_file(&root) {
        let Ok(file) = log::read(&path) else { continue };
        for line in &file.lines {
            if log::op(line) == "snapshot" {
                for item in line.get("rows").and_then(Value::as_array).into_iter().flatten() {
                    if let (Some(table), Some(row)) = (item.get("table").and_then(Value::as_str), item.get("row").and_then(Value::as_object)) {
                        note(table, row);
                    }
                }
            } else if let (Some(table), Some(row)) = (line.get("table").and_then(Value::as_str), line.get("row").and_then(Value::as_object)) {
                note(table, row);
            }
        }
    }
    highest
}

/// Every board's and audit's file under `root`: hidden places too (the
/// trash, what is set aside).
fn every_file(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(path),
                Ok(t) if t.is_file() && Kind::of_path(&path).is_some() => out.push(path),
                _ => {}
            }
        }
    }
    out
}

/// A board made while an archived or trashed board's id could be given
/// again took that id, and its lines went into the old board's file. Such a
/// file (a second `boards` row of its id, made at another time) is split:
/// the new board's own file is made in `inbox/` (from its lines there and
/// in any file written for it since, as one), and the old board's part is
/// set aside in `.s0ma/reused-ids/`. What was done, said.
fn repair_reused_ids(root: &Path) -> Vec<String> {
    let _writing = crate::util::locked(&WRITING);
    let mut said = Vec::new();
    let files: Vec<(PathBuf, log::Log)> = walk(root).into_iter().filter_map(|(path, _)| log::read(&path).ok().map(|log| (path, log))).collect();
    for (path, file) in &files {
        if file.kind() != Kind::Board.word() {
            continue;
        }
        let id = file.id().to_string();
        let boards_rows = |line: &Map<String, Value>| -> Vec<Map<String, Value>> {
            if log::op(line) == "snapshot" {
                line.get("rows").and_then(Value::as_array).into_iter().flatten()
                    .filter(|item| item.get("table").and_then(Value::as_str) == Some("boards"))
                    .filter_map(|item| item.get("row").and_then(Value::as_object).cloned())
                    .collect()
            } else if line.get("table").and_then(Value::as_str) == Some("boards") {
                line.get("row").and_then(Value::as_object).cloned().into_iter().collect()
            } else {
                Vec::new()
            }
        };
        let made_at = |row: &Map<String, Value>| row.get("created_at").and_then(Value::as_str).unwrap_or("").to_string();
        let mut first: Option<String> = None;
        let mut split = None;
        'lines: for (index, line) in file.lines.iter().enumerate() {
            for row in boards_rows(line) {
                if row.get("id").and_then(Value::as_str) != Some(id.as_str()) {
                    continue;
                }
                let made = made_at(&row);
                match &first {
                    None => first = Some(made),
                    Some(earlier) if *earlier != made => {
                        split = Some(index);
                        break 'lines;
                    }
                    Some(_) => {}
                }
            }
        }
        let Some(split) = split else { continue };
        let new_made = file.lines[split..].iter().flat_map(boards_rows).find(|r| r.get("id").and_then(Value::as_str) == Some(id.as_str())).map(|r| made_at(&r));
        let new_name = file.lines[split..].iter().flat_map(boards_rows).filter_map(|r| r.get("name").and_then(Value::as_str).map(str::to_string)).next_back().unwrap_or_else(|| id.clone());
        // Other files of the same id made for the new board (written out
        // when its own could not be found): their lines are its too.
        let mut theirs: Vec<Map<String, Value>> = file.lines[split..].to_vec();
        let mut others: Vec<PathBuf> = Vec::new();
        for (other, other_file) in &files {
            if other == path || other_file.id() != id || other_file.kind() != Kind::Board.word() {
                continue;
            }
            let made = other_file.lines.iter().flat_map(boards_rows).find(|r| r.get("id").and_then(Value::as_str) == Some(id.as_str())).map(|r| made_at(&r));
            if made == new_made {
                theirs.extend(other_file.lines.iter().cloned());
                others.push(other.clone());
            }
        }
        // Taken apart before, and the file is back as it was (a copy from
        // another machine, a restore): the board keeps the file it has, and
        // takes any lines of its own it lacks; nothing new is made or said.
        if let Some(own) = others.first() {
            let had: std::collections::HashSet<String> =
                log::read(own).map(|log| log.lines.iter().map(|line| log::n(line).to_string()).collect()).unwrap_or_default();
            let missing: Vec<Map<String, Value>> = file.lines[split..].iter().filter(|line| !had.contains(log::n(line))).cloned().collect();
            if log::append(own, missing).is_ok() && old_part_kept(root, &file.lines[..split]) {
                let _ = std::fs::remove_file(path);
                INDEX.with(|index| {
                    if let Some(rel_path) = rel(path) {
                        index.files.remove(&rel_path);
                    }
                });
                continue;
            }
        }
        // As one: the newest snapshot, then what came after it; or all.
        let mut seen = std::collections::HashSet::new();
        theirs.retain(|line| seen.insert(log::n(line).to_string()));
        theirs.sort_by(|a, b| a.get("at").and_then(Value::as_str).unwrap_or("").cmp(b.get("at").and_then(Value::as_str).unwrap_or("")));
        if let Some(at) = theirs.iter().rposition(|line| log::op(line) == "snapshot") {
            theirs.drain(..at);
        }
        let mut header = file.header.clone();
        header.insert("created".into(), Value::String(new_made.clone().unwrap_or_else(log::now)));
        let target = unique(&root.join("inbox"), &safe_name(&new_name), Kind::Board.ext());
        let _ = std::fs::create_dir_all(root.join("inbox"));
        if let Err(error) = log::create(&target, &header).and_then(|()| log::append(&target, theirs)) {
            said.push(format!("{}: {error}", rel(path).unwrap_or_default()));
            continue;
        }
        // The old board's part, aside.
        let aside_dir = root.join(".s0ma").join("reused-ids");
        let _ = std::fs::create_dir_all(&aside_dir);
        let old_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let aside = unique(&aside_dir, old_name.trim_end_matches(".board"), Kind::Board.ext());
        let old = log::Log { header: file.header.clone(), lines: file.lines[..split].to_vec(), skipped: 0 };
        let set_aside = log::create(&aside, &old.header).and_then(|()| log::append(&aside, old.lines.clone()));
        if set_aside.is_ok() {
            let _ = std::fs::remove_file(path);
            for other in &others {
                let _ = std::fs::remove_file(other);
            }
        }
        // The index learns it again from the files.
        INDEX.with(|index| index.files.retain(|_, e| !(e.kind == Some(Kind::Board) && e.id == id)));
        said.push(format!(
            "{new_name} had taken the id of the board in {} and was written into its file: it has its own now, {}; that board's part is kept in {}",
            rel(path).unwrap_or_default(),
            rel(&target).unwrap_or_default(),
            aside.strip_prefix(root).map(|p| p.display().to_string()).unwrap_or_default()
        ));
    }
    said
}

/// Whether the lines set aside in `.s0ma/reused-ids/` already hold `old`
/// (every line of it, by its id), so a file split again loses nothing.
fn old_part_kept(root: &Path, old: &[Map<String, Value>]) -> bool {
    let wanted: std::collections::HashSet<&str> = old.iter().map(log::n).collect();
    std::fs::read_dir(root.join(".s0ma").join("reused-ids")).into_iter().flatten().flatten().any(|entry| {
        log::read(&entry.path()).is_ok_and(|kept| {
            let have: std::collections::HashSet<&str> = kept.lines.iter().map(log::n).collect();
            wanted.is_subset(&have)
        })
    })
}

/// `work` while no store writes a project file and no scan runs: what the
/// share sync renames, it renames under this.
pub fn while_writing<T>(work: impl FnOnce() -> T) -> T {
    let _writing = crate::util::locked(&WRITING);
    work()
}

/// Where a new board's or audit's file goes: the folder the projects tab
/// chose; for an audit, beside its board; else `inbox/`.
fn place_new(root: &Path, kind: Kind, owner: &str, lines: &[Map<String, Value>]) -> PathBuf {
    let name_of = |key: &str| {
        lines.iter().filter_map(|l| l.get("row").and_then(|r| r.get(key)).and_then(Value::as_str)).next().map(str::to_string)
    };
    let name = match kind {
        Kind::Board => name_of("name").unwrap_or_else(|| owner.to_string()),
        Kind::Audit => owner.to_string(),
    };
    let chosen = PLACE_NEXT.with(|p| p.borrow_mut().take());
    let folder = chosen.or_else(|| {
        if kind != Kind::Audit {
            return None;
        }
        let board = name_of("board")?;
        let id = board_id_by_name(&board)?;
        path_of(Kind::Board, &id).and_then(|p| p.parent().map(Path::to_path_buf))
    });
    let folder = folder.unwrap_or_else(|| root.join("inbox"));
    unique(&folder, &safe_name(&name), kind.ext())
}

pub fn safe_name(name: &str) -> String {
    let cleaned: String = name.chars().map(|c| if c == '/' || c == '\0' { '-' } else { c }).collect();
    let cleaned = cleaned.trim().trim_start_matches('.').to_string();
    if cleaned.is_empty() { "untitled".into() } else { cleaned }
}

/// `folder/name.ext`, or `name-2.ext` and on when taken.
pub fn unique(folder: &Path, name: &str, ext: &str) -> PathBuf {
    let dotted = if ext.is_empty() { String::new() } else { format!(".{ext}") };
    let first = folder.join(format!("{name}{dotted}"));
    if !first.exists() {
        return first;
    }
    (2..).map(|n| folder.join(format!("{name}-{n}{dotted}"))).find(|p| !p.exists()).unwrap_or(first)
}

pub fn board_id_by_name(name: &str) -> Option<String> {
    let path = store_path(journal::Store::Board)?;
    let db = crate::db::open(&path).ok()?;
    db.query_row("SELECT id FROM boards WHERE name=?1", [name], |row| row.get(0)).ok()
}

/// A board by its path in the projects folder (`calc/parser/fix-overflow`,
/// `.board` or not; a leading `projects/` or `/share/projects/` is fine).
pub fn board_id_by_path(path: &str) -> Option<String> {
    let mut rel_path = path.trim().trim_start_matches('/');
    for prefix in ["share/projects/", "projects/"] {
        rel_path = rel_path.strip_prefix(prefix).unwrap_or(rel_path);
    }
    if rel_path.is_empty() {
        return None;
    }
    let with_ext = if rel_path.ends_with(".board") { rel_path.to_string() } else { format!("{rel_path}.board") };
    INDEX.with(|index| index.files.get(&with_ext).filter(|e| e.kind == Some(Kind::Board) && e.copy_of.is_none()).map(|e| e.id.clone()))
}

// ---------------------------------------------------------------- reading the files back

/// What a scan did, for the agent's next message and the tab.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scanned {
    pub moved: Vec<(String, String)>,
    pub found: Vec<String>,
    pub changed: Vec<String>,
    pub gone: Vec<String>,
    pub copies: Vec<String>,
    pub problems: Vec<String>,
}

impl Scanned {
    pub fn is_empty(&self) -> bool {
        self.moved.is_empty() && self.found.is_empty() && self.changed.is_empty() && self.gone.is_empty() && self.copies.is_empty() && self.problems.is_empty()
    }
}

/// Every board and audit file under the root (not in the trash), with
/// whether it is hidden by an archived folder above it.
fn walk(root: &Path) -> Vec<(PathBuf, bool)> {
    let mut out = Vec::new();
    fn go(dir: &Path, hidden: bool, out: &mut Vec<(PathBuf, bool)>) {
        let Ok(read) = std::fs::read_dir(dir) else { return };
        let archived = tree::archived_names(dir);
        let mut entries: Vec<_> = read.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let path = entry.path();
            let hide = hidden || archived.contains(&name);
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                go(&path, hide, out);
            } else if Kind::of_path(&path).is_some() {
                out.push((path, hide));
            }
        }
    }
    go(root, false, &mut out);
    out
}

/// The files brought into the index: moves, new files, changes made
/// outside, archived and trashed ones taken out, restored ones put back,
/// and boards and audits the stores have with no file yet written out.
pub fn scan() -> Scanned {
    let mut said = Scanned::default();
    // The primary's to do; an attached terminal reads the index it keeps.
    if attached() {
        return said;
    }
    let Some(root) = root() else { return said };
    if !root.is_dir() {
        return said;
    }
    let _writing = crate::util::locked(&WRITING);
    let on_disk = walk(&root);
    let known: HashMap<String, Entry> = INDEX.with(|index| index.files.clone());
    let mut next: HashMap<String, Entry> = HashMap::new();
    let mut seen_ids: HashMap<(Kind, String), String> = HashMap::new();
    // Files already known at their path (and not as copies) come first, so
    // a copy elsewhere is the one taken for the copy, at every scan.
    let mut ordered = on_disk.clone();
    ordered.sort_by_key(|(p, _)| !known.get(&rel(p).unwrap_or_default()).is_some_and(|e| e.copy_of.is_none()));
    for (path, hidden) in ordered {
        let Some(rel_path) = rel(&path) else { continue };
        let Some(kind) = Kind::of_path(&path) else { continue };
        let Some((size, mtime)) = stamp(&path) else { continue };
        let old = known.get(&rel_path).cloned();
        let same = old.as_ref().is_some_and(|e| e.size == size && e.mtime == mtime);
        let (log, id) = if same {
            (None, old.as_ref().map(|e| e.id.clone()).unwrap_or_default())
        } else {
            match log::read(&path) {
                Ok(log) => {
                    let id = log.id().to_string();
                    (Some(log), id)
                }
                Err(error) => {
                    said.problems.push(format!("{rel_path}: {error}"));
                    continue;
                }
            }
        };
        if id.is_empty() {
            said.problems.push(format!("{rel_path}: no id in its header"));
            continue;
        }
        if let Some(first) = seen_ids.get(&(kind, id.clone())) {
            said.copies.push(rel_path.clone());
            next.insert(rel_path, Entry { id, kind: Some(kind), size, mtime, live: false, archived: false, copy_of: Some(first.clone()) });
            continue;
        }
        seen_ids.insert((kind, id.clone()), rel_path.clone());
        let archived_self = match (&log, &old) {
            (Some(log), _) => tree::archived_in(log),
            (None, Some(e)) => e.archived,
            _ => false,
        };
        let want_live = !hidden && !archived_self;
        // Where it was before, if it moved.
        let was = old.clone().filter(|e| e.id == id).or_else(|| {
            known.iter().find(|(p, e)| e.id == id && e.kind == Some(kind) && e.copy_of.is_none() && !on_disk.iter().any(|(d, _)| rel(d).as_deref() == Some(p.as_str()))).map(|(p, e)| {
                said.moved.push((p.clone(), rel_path.clone()));
                e.clone()
            })
        });
        let was_live = was.as_ref().is_some_and(|e| e.live);
        let changed = !same && was.as_ref().is_some_and(|e| !(e.size == size && e.mtime == mtime));
        if was.is_none() {
            said.found.push(rel_path.clone());
        } else if changed && old.is_some() {
            said.changed.push(rel_path.clone());
        }
        let mut live = was_live;
        if want_live && (!was_live || (changed && old.is_some()) || was.is_none()) {
            // Into the index from the file: new, restored, unarchived, or
            // changed outside the stores.
            let log = match log {
                Some(log) => Some(log),
                None => log::read(&path).ok(),
            };
            if let Some(log) = log {
                match journal::replay(kind, &id, &log) {
                    Ok(0) => live = true,
                    Ok(left_out) => {
                        live = true;
                        said.problems.push(format!("{rel_path}: {left_out} of its rows have ids another {} uses, and were left out", kind.word()));
                    }
                    Err(error) => said.problems.push(format!("{rel_path}: {error}")),
                }
            }
        } else if !want_live && was_live {
            stop_work(kind, &id);
            match journal::remove(kind, &id) {
                Ok(()) => live = false,
                Err(error) => said.problems.push(format!("{rel_path}: {error}")),
            }
        }
        next.insert(rel_path, Entry { id, kind: Some(kind), size, mtime, live, archived: archived_self, copy_of: None });
    }
    // Gone: known, live, and nowhere on disk now.
    for (path, entry) in &known {
        let still = next.values().any(|e| e.id == entry.id && e.kind == entry.kind && e.copy_of.is_none());
        if !still && entry.copy_of.is_none() {
            said.gone.push(path.clone());
            if entry.live
                && let Some(kind) = entry.kind
            {
                stop_work(kind, &entry.id);
                if let Err(error) = journal::remove(kind, &entry.id) {
                    said.problems.push(format!("{path}: {error}"));
                }
            }
        }
    }
    INDEX.with(|index| index.files = next);
    drop(_writing);
    // What the stores have that no file holds yet: written out.
    export_missing(&root, &mut said);
    save_index();
    said
}

/// Boards and audits in the stores with no file: written as one snapshot
/// each, into `imported/` (the first start), or beside their board.
fn export_missing(root: &Path, said: &mut Scanned) {
    let mut owners: Vec<(Kind, String, String)> = Vec::new();
    if let Some(path) = store_path(journal::Store::Board).filter(|p| p.exists())
        && let Ok(db) = crate::db::open(&path)
        && let Ok(mut statement) = db.prepare("SELECT id, name FROM boards ORDER BY seq")
    {
        owners.extend(statement.query_map([], |row| Ok((Kind::Board, row.get::<_, String>(0)?, row.get::<_, String>(1)?))).into_iter().flatten().flatten());
    }
    if let Some(path) = store_path(journal::Store::Audit).filter(|p| p.exists())
        && let Ok(db) = crate::db::open(&path)
        && let Ok(mut statement) = db.prepare("SELECT name, name FROM audits ORDER BY seq")
    {
        owners.extend(statement.query_map([], |row| Ok((Kind::Audit, row.get::<_, String>(0)?, row.get::<_, String>(1)?))).into_iter().flatten().flatten());
    }
    for (kind, id, name) in owners {
        if path_of(kind, &id).is_some() {
            continue;
        }
        let _writing = crate::util::locked(&WRITING);
        let rows = match journal::rows_of(kind, &id) {
            Ok(rows) => rows,
            Err(error) => {
                said.problems.push(format!("{} {name}: {error}", kind.word()));
                continue;
            }
        };
        let board_folder = (kind == Kind::Audit)
            .then(|| rows.iter().find_map(|r| r.get("row").and_then(|row| row.get("board")).and_then(Value::as_str).map(str::to_string)))
            .flatten()
            .and_then(|board| board_id_by_name(&board))
            .and_then(|board| path_of(Kind::Board, &board))
            .and_then(|p| p.parent().map(Path::to_path_buf));
        // Before projects: into imported/. Since: one whose file could not be
        // written as it was made, into inbox/ with the new ones.
        let first = !root.join(".s0ma").join("migrated").exists();
        let folder = board_folder.unwrap_or_else(|| root.join(if first { "imported" } else { "inbox" }));
        let path = unique(&folder, &safe_name(&name), kind.ext());
        let mut header = Map::new();
        header.insert("s0ma".into(), Value::String(kind.word().into()));
        header.insert("v".into(), Value::from(1));
        header.insert("id".into(), Value::String(id.clone()));
        header.insert("created".into(), Value::String(log::now()));
        let snapshot: Map<String, Value> = serde_json::json!({"op": "snapshot", "rows": rows, "folded": {}}).as_object().cloned().unwrap_or_default();
        let written = log::create(&path, &header).and_then(|()| log::append(&path, vec![snapshot]));
        match (written, rel(&path), stamp(&path)) {
            (Ok(_), Some(rel_path), Some((size, mtime))) => {
                INDEX.with(|index| {
                    index.files.insert(rel_path.clone(), Entry { id, kind: Some(kind), size, mtime, live: true, archived: false, copy_of: None });
                });
                said.found.push(rel_path);
            }
            (Err(error), ..) => said.problems.push(format!("{} {name}: {error}", kind.word())),
            _ => {}
        }
    }
    if root.join("imported").is_dir() && !root.join("imported").join("index.md").exists() {
        let _ = std::fs::write(
            root.join("imported").join("index.md"),
            "# Imported\n\nThe boards and audits that were here before projects. Make projects (folders here) and move them in.\n",
        );
    }
}

/// Work on a board or audit stopped before it leaves the index.
fn stop_work(kind: Kind, id: &str) {
    if cfg!(test) {
        return;
    }
    match kind {
        Kind::Audit => {
            crate::audit::run::stop_workers(id);
            // Its shell goes with it.
            crate::audit::run::close_shell_named(crate::audit::shell_name(id));
        }
        Kind::Board => {
            // Workers holding its leads.
            let holders: Vec<String> = store_path(journal::Store::Board)
                .and_then(|path| crate::db::open(&path).ok())
                .and_then(|db| {
                    let mut statement = db.prepare("SELECT DISTINCT claimed_by FROM intents WHERE board_id=?1 AND claimed_by IS NOT NULL").ok()?;
                    let rows: Vec<String> = statement.query_map([id], |row| row.get(0)).ok()?.flatten().collect();
                    Some(rows)
                })
                .unwrap_or_default();
            for holder in holders {
                let _ = crate::subagent::kill(&holder);
            }
        }
    }
}

// ---------------------------------------------------------------- start

/// An attached terminal's: the projects folder and the stores, the index as
/// the primary keeps it (read again as it changes). No scan, no import.
pub fn init_attached(archive: &Path) {
    if std::env::var("S0MA_PROJECTS").is_ok_and(|v| v == "off") {
        return;
    }
    let root = crate::share::receive_dir().join("projects");
    if !root.is_dir() {
        return;
    }
    ATTACHED.store(true, std::sync::atomic::Ordering::Relaxed);
    *crate::util::locked(&ROOT) = Some(root);
    *crate::util::locked(&STORES) = Some(HashMap::from([
        (journal::Store::Board, archive.with_file_name("board.sqlite3")),
        (journal::Store::Flow, archive.with_file_name("board_flow.sqlite3")),
        (journal::Store::Audit, archive.with_file_name("audit.sqlite3")),
    ]));
    load_index();
    let _ = std::thread::Builder::new().name("projects-index".into()).spawn(|| {
        let mut seen = None;
        loop {
            let now = index_file().and_then(|p| stamp(&p));
            if now != seen {
                seen = now;
                load_index();
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    });
}

/// The projects root set up, the files read into the index, and a scan
/// every two seconds from then on.
pub fn init(archive: &Path) {
    if std::env::var("S0MA_PROJECTS").is_ok_and(|v| v == "off") {
        return;
    }
    let root = crate::share::receive_dir().join("projects");
    if std::fs::create_dir_all(&root).is_err() {
        return;
    }
    *crate::util::locked(&ROOT) = Some(root.clone());
    *crate::util::locked(&STORES) = Some(HashMap::from([
        (journal::Store::Board, archive.with_file_name("board.sqlite3")),
        (journal::Store::Flow, archive.with_file_name("board_flow.sqlite3")),
        (journal::Store::Audit, archive.with_file_name("audit.sqlite3")),
    ]));
    load_index();
    // The first start: the databases as they were, kept beside them.
    let marker = root.join(".s0ma").join("migrated");
    if !marker.exists() {
        for name in ["board.sqlite3", "board_flow.sqlite3", "audit.sqlite3"] {
            let path = archive.with_file_name(name);
            if path.exists() {
                let _ = std::fs::copy(&path, archive.with_file_name(format!("{name}.before-projects")));
            }
        }
    }
    // Boards that took an archived or trashed board's id (before ids were
    // kept unique) and so were written into its file: taken out of it.
    let repaired = repair_reused_ids(&root);
    if !repaired.is_empty() {
        warn_once(&format!("[projects] {}", repaired.join("; ")));
    }
    let first = scan();
    if !first.problems.is_empty() {
        warn_once(&format!("[projects] {}", first.problems.join("; ")));
    }
    if !marker.exists() {
        let _ = std::fs::create_dir_all(marker.parent().unwrap_or(&root));
        let _ = std::fs::write(&marker, log::now());
    }
    let _ = std::thread::Builder::new().name("projects-scan".into()).spawn(|| {
        let mut standing = Standing::default();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(2));
            let said = standing.news(scan());
            if !said.is_empty() {
                note(said);
            }
        }
    });
}

/// Copies and unreadable files are found again by every scan for as long
/// as they are there: said once, and again only after they went away.
#[derive(Default)]
struct Standing(std::collections::HashSet<String>);

impl Standing {
    fn news(&mut self, mut said: Scanned) -> Scanned {
        let now: std::collections::HashSet<String> = said.copies.iter().chain(&said.problems).cloned().collect();
        said.copies.retain(|copy| !self.0.contains(copy));
        said.problems.retain(|problem| !self.0.contains(problem));
        self.0 = now;
        said
    }
}

static NEWS: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn note(said: Scanned) {
    let mut news = crate::util::locked(&NEWS);
    for (from, to) in said.moved {
        news.push(format!("moved {from} → {to}"));
    }
    for path in said.found {
        news.push(format!("new {path}"));
    }
    for path in said.changed {
        news.push(format!("changed outside s0ma {path}"));
    }
    for path in said.gone {
        news.push(format!("gone {path}"));
    }
    for path in said.copies {
        news.push(format!("a copy (not used until renamed apart) {path}"));
    }
    for problem in said.problems {
        news.push(format!("can't read {problem}"));
    }
    crate::tabs::poke();
}

/// What changed in the projects since last asked, for the operator's next
/// message to the agent.
pub fn take_note() -> Option<String> {
    let news = std::mem::take(&mut *crate::util::locked(&NEWS));
    if news.is_empty() {
        return None;
    }
    let root = crate::share::share_dir();
    let mut note = format!("\n\n[PROJECTS]\nIn {root}/projects (the operator's, through the projects tab or a file manager):\n");
    for line in news {
        note.push_str(&format!("- {line}\n"));
    }
    Some(note)
}

#[cfg(test)]
mod tests;
