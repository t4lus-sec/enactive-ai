//! The project tree as the projects tab shows and changes it: folders,
//! pages (`index.md` and other `.md` files), boards, audits (shown under
//! their board), other files; archiving, the trash, and making, renaming
//! and moving things. Everything here works on the files; `scan` then
//! brings the boards and audits it touched into (or out of) the index.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::{Kind, abs, log, rel, root};

pub const PAGE: &str = "index.md";
pub const FOLDER_FILE: &str = ".folder";
pub const TRASH: &str = ".trash";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Project,
    Folder,
    Page,
    Board,
    Audit,
    File,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// Its path under the root.
    pub rel: String,
    pub name: String,
    pub kind: NodeKind,
    pub depth: usize,
    /// Archived itself.
    pub archived: bool,
    /// Under an archived folder.
    pub hidden: bool,
    /// A folder with something in it (other than its page).
    pub has_children: bool,
    /// An audit shown under its board, not where its file sits.
    pub attached: bool,
}

// ---------------------------------------------------------------- .folder

fn folder_file(dir: &Path) -> PathBuf {
    dir.join(FOLDER_FILE)
}

/// A folder's `.folder`, made when missing.
pub fn folder_log(dir: &Path) -> Result<PathBuf, String> {
    let path = folder_file(dir);
    if !path.exists() {
        let mut header = Map::new();
        header.insert("s0ma".into(), Value::String("folder".into()));
        header.insert("v".into(), Value::from(1));
        header.insert("id".into(), Value::String(format!("d-{}", log::now().replace([':', '-', '.'], ""))));
        match log::create(&path, &header) {
            Ok(()) => {}
            Err(_) if path.exists() => {}
            Err(error) => return Err(error),
        }
    }
    Ok(path)
}

/// The names in a folder that are archived (from its `.folder`).
pub fn archived_names(dir: &Path) -> HashSet<String> {
    let mut out = HashSet::new();
    let Ok(log) = log::read(&folder_file(dir)) else { return out };
    for line in &log.lines {
        if log::op(line) == "archived"
            && let Some(name) = line.get("name").and_then(Value::as_str)
        {
            if line.get("on").and_then(Value::as_bool).unwrap_or(false) {
                out.insert(name.to_string());
            } else {
                out.remove(name);
            }
        }
    }
    out
}

/// Whether a board's or audit's own file says it is archived.
pub fn archived_in(log: &log::Log) -> bool {
    log.lines.iter().rev().find(|l| log::op(l) == "archived").and_then(|l| l.get("on")).and_then(Value::as_bool).unwrap_or(false)
}

// ---------------------------------------------------------------- the tree

/// Each audit's board, by the board's file: where the tree shows it.
fn audit_boards() -> HashMap<String, String> {
    let mut out = HashMap::new();
    let files = super::INDEX.with(|index| index.files.clone());
    // The boards' names, from the index or their files.
    let mut board_names: HashMap<String, String> = HashMap::new();
    for (path, entry) in &files {
        if entry.kind == Some(Kind::Board)
            && let Some(name) = board_name(path)
        {
            board_names.insert(name, path.clone());
        }
    }
    for (path, entry) in &files {
        if entry.kind != Some(Kind::Audit) {
            continue;
        }
        if let Some(board) = audit_board(path)
            && let Some(board_path) = board_names.get(&board)
        {
            out.insert(path.clone(), board_path.clone());
        }
    }
    out
}

/// A board's name and an audit's board, as their files say now (read
/// once per version of the file: the tree asks at every draw).
pub fn board_name(rel_path: &str) -> Option<String> {
    super::summary::board(&abs(rel_path)?).map(|info| info.name).filter(|name| !name.is_empty())
}

pub fn audit_board(rel_path: &str) -> Option<String> {
    super::summary::audit(&abs(rel_path)?).and_then(|info| info.board)
}

/// The tree, flattened in display order: folders first, then pages,
/// boards (each with its audits under it), then other files. `open` is the
/// folders shown open; archived things only with `archived` on.
pub fn list(open: &HashSet<String>, show_archived: bool) -> Vec<Node> {
    let Some(root) = root() else { return Vec::new() };
    let attached = audit_boards();
    let by_board: HashMap<String, Vec<String>> = attached.iter().fold(HashMap::new(), |mut map, (audit, board)| {
        map.entry(board.clone()).or_default().push(audit.clone());
        map
    });
    let mut out = Vec::new();
    walk(&root, &root, 0, false, open, show_archived, &attached, &by_board, &mut out);
    out
}

#[allow(clippy::too_many_arguments)]
fn walk(
    root: &Path,
    dir: &Path,
    depth: usize,
    hidden: bool,
    open: &HashSet<String>,
    show_archived: bool,
    attached: &HashMap<String, String>,
    by_board: &HashMap<String, Vec<String>>,
    out: &mut Vec<Node>,
) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    let archived = archived_names(dir);
    let mut folders = Vec::new();
    let mut pages = Vec::new();
    let mut boards = Vec::new();
    let mut audits = Vec::new();
    let mut files = Vec::new();
    for entry in read.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || (depth > 0 || dir != root) && name == PAGE {
            continue;
        }
        let path = entry.path();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_dir {
            folders.push((name, path));
        } else if name.ends_with(".md") {
            pages.push((name, path));
        } else if name.ends_with(".board") {
            boards.push((name, path));
        } else if name.ends_with(".audit") {
            audits.push((name, path));
        } else {
            files.push((name, path));
        }
    }
    for list in [&mut folders, &mut pages, &mut boards, &mut audits, &mut files] {
        list.sort_by_key(|(name, _)| name.to_lowercase());
    }
    let node = |name: &str, path: &Path, kind: NodeKind, depth: usize, attached_here: bool| -> Option<Node> {
        let rel_path = rel(path)?;
        let self_archived = archived.contains(name) || matches!(kind, NodeKind::Board | NodeKind::Audit) && entry_archived(&rel_path);
        if (self_archived || hidden) && !show_archived {
            return None;
        }
        let has_children = kind == NodeKind::Project || kind == NodeKind::Folder;
        Some(Node { rel: rel_path, name: name.to_string(), kind, depth, archived: self_archived, hidden, has_children, attached: attached_here })
    };
    for (name, path) in folders {
        let kind = if dir == root { NodeKind::Project } else { NodeKind::Folder };
        let Some(mut folder) = node(&name, &path, kind, depth, false) else { continue };
        let inside = std::fs::read_dir(&path).map(|r| r.flatten().any(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            !n.starts_with('.') && n != PAGE
        })).unwrap_or(false);
        folder.has_children = inside;
        let is_open = open.contains(&folder.rel);
        let child_hidden = hidden || folder.archived;
        out.push(folder);
        if is_open {
            walk(root, &path, depth + 1, child_hidden, open, show_archived, attached, by_board, out);
        }
    }
    for (name, path) in pages {
        if let Some(page) = node(&name, &path, NodeKind::Page, depth, false) {
            out.push(page);
        }
    }
    for (name, path) in boards {
        let Some(board) = node(&name, &path, NodeKind::Board, depth, false) else { continue };
        let rel_board = board.rel.clone();
        out.push(board);
        // Its audits, wherever their files are.
        for audit_rel in by_board.get(&rel_board).into_iter().flatten() {
            let Some(audit_path) = abs(audit_rel) else { continue };
            let audit_name = audit_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let audit_archived = entry_archived(audit_rel) || archived_names(audit_path.parent().unwrap_or(root)).contains(&audit_name);
            if (audit_archived || hidden) && !show_archived {
                continue;
            }
            out.push(Node {
                rel: audit_rel.clone(),
                name: audit_name,
                kind: NodeKind::Audit,
                depth: depth + 1,
                archived: audit_archived,
                hidden,
                has_children: false,
                attached: true,
            });
        }
    }
    for (name, path) in audits {
        let rel_audit = rel(&path).unwrap_or_default();
        if attached.contains_key(&rel_audit) {
            continue;
        }
        if let Some(audit) = node(&name, &path, NodeKind::Audit, depth, false) {
            out.push(audit);
        }
    }
    for (name, path) in files {
        if let Some(file) = node(&name, &path, NodeKind::File, depth, false) {
            out.push(file);
        }
    }
}

fn entry_archived(rel_path: &str) -> bool {
    super::INDEX.with(|index| index.files.get(rel_path).is_some_and(|e| e.archived))
}

// ---------------------------------------------------------------- making and changing

fn check_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() || name.starts_with('.') || name.contains('/') || name.contains('\0') || name == ".." {
        return Err(format!("'{name}' can't be a name"));
    }
    Ok(name.to_string())
}

fn folder_of(rel_folder: &str) -> Result<PathBuf, String> {
    let root = root().ok_or("no projects folder")?;
    let dir = if rel_folder.is_empty() { root } else { root.join(rel_folder) };
    if !dir.is_dir() {
        return Err(format!("{rel_folder} is not a folder"));
    }
    Ok(dir)
}

/// A new folder, with its page.
pub fn create_folder(parent: &str, name: &str) -> Result<String, String> {
    let name = check_name(name)?;
    let dir = folder_of(parent)?.join(&name);
    if dir.exists() {
        return Err(format!("{name} is there already"));
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(PAGE), format!("# {name}\n")).map_err(|e| e.to_string())?;
    let _ = folder_log(&dir);
    rel(&dir).ok_or_else(|| "outside the projects folder".to_string())
}

/// A new markdown file.
pub fn create_page(parent: &str, name: &str) -> Result<String, String> {
    let name = check_name(name)?;
    let name = if name.ends_with(".md") { name } else { format!("{name}.md") };
    let path = folder_of(parent)?.join(&name);
    if path.exists() {
        return Err(format!("{name} is there already"));
    }
    let title = name.trim_end_matches(".md");
    std::fs::write(&path, format!("# {title}\n")).map_err(|e| e.to_string())?;
    rel(&path).ok_or_else(|| "outside the projects folder".to_string())
}

/// Renamed in place (a board or audit keeps its extension).
pub fn rename(rel_path: &str, new_name: &str) -> Result<String, String> {
    let from = abs(rel_path).ok_or("no projects folder")?;
    let mut name = check_name(new_name)?;
    for ext in ["board", "audit"] {
        if from.extension().and_then(|e| e.to_str()) == Some(ext) && !name.ends_with(&format!(".{ext}")) {
            name = format!("{name}.{ext}");
        }
    }
    let to = from.with_file_name(&name);
    if to.exists() {
        return Err(format!("{name} is there already"));
    }
    std::fs::rename(&from, &to).map_err(|e| e.to_string())?;
    carry_archived(&from, &to);
    super::scan();
    rel(&to).ok_or_else(|| "outside the projects folder".to_string())
}

/// An archived mark in the old folder's `.folder` follows a plain file or
/// folder to its new name or place.
fn carry_archived(from: &Path, to: &Path) {
    let (Some(old_dir), Some(old_name), Some(new_dir), Some(new_name)) = (from.parent(), from.file_name(), to.parent(), to.file_name()) else { return };
    let old_name = old_name.to_string_lossy().into_owned();
    if archived_names(old_dir).contains(&old_name) {
        let _ = mark(old_dir, &old_name, false);
        let _ = mark(new_dir, &new_name.to_string_lossy(), true);
    }
    carry_comments(old_dir, &old_name, new_dir, &new_name.to_string_lossy());
}

/// A page's comments follow it: their lines move from the old folder's
/// `.folder` to the new one's, under its new name.
fn carry_comments(old_dir: &Path, old_name: &str, new_dir: &Path, new_name: &str) {
    let old_file = folder_file(old_dir);
    let Ok(locked) = log::Locked::exclusive(&old_file) else { return };
    let Ok(mut old) = log::read_from(&locked.file) else { return };
    let ids: HashSet<String> = old
        .lines
        .iter()
        .filter(|l| log::op(l) == "comment" && l.get("on").and_then(Value::as_str) == Some(old_name))
        .filter_map(|l| l.get("id").and_then(Value::as_str).map(str::to_string))
        .collect();
    if ids.is_empty() {
        return;
    }
    let ours = |l: &Map<String, Value>| matches!(log::op(l), "comment" | "resolve") && l.get("id").and_then(Value::as_str).is_some_and(|id| ids.contains(id));
    let moving: Vec<Map<String, Value>> = old
        .lines
        .iter()
        .filter(|l| ours(l))
        .map(|l| {
            let mut l = l.clone();
            if log::op(&l) == "comment" {
                l.insert("on".into(), new_name.into());
            }
            l
        })
        .collect();
    if old_dir == new_dir {
        old.lines = old.lines.into_iter().map(|l| if ours(&l) { moving.iter().find(|m| m.get("n") == l.get("n")).cloned().unwrap_or(l) } else { l }).collect();
        let _ = log::rewrite_locked(&old_file, &locked, &old);
        return;
    }
    let Ok(new_file) = folder_log(new_dir) else { return };
    if log::append(&new_file, moving).is_ok() {
        old.lines.retain(|l| !ours(l));
        let _ = log::rewrite_locked(&old_file, &locked, &old);
    }
}

fn mark(dir: &Path, name: &str, on: bool) -> Result<(), String> {
    let path = folder_log(dir)?;
    let line = serde_json::json!({"op": "archived", "name": name, "on": on}).as_object().cloned().unwrap_or_default();
    log::append(&path, vec![line]).map(|_| ())
}

/// Moved into a folder. A board takes the audits beside it that are its own.
pub fn move_to(rel_path: &str, folder: &str) -> Result<String, String> {
    let from = abs(rel_path).ok_or("no projects folder")?;
    let dir = folder_of(folder)?;
    if dir.starts_with(&from) {
        return Err("a folder can't go inside itself".into());
    }
    let name = from.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or("no name")?;
    if from.parent() == Some(dir.as_path()) {
        return Ok(rel_path.to_string());
    }
    let to = unique_like(&dir, &name);
    let audits = own_audits_beside(rel_path);
    std::fs::rename(&from, &to).map_err(|e| e.to_string())?;
    carry_archived(&from, &to);
    for audit in audits {
        if let Some(path) = abs(&audit) {
            let audit_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let _ = std::fs::rename(&path, unique_like(&dir, &audit_name));
        }
    }
    super::scan();
    rel(&to).ok_or_else(|| "outside the projects folder".to_string())
}

/// `dir/name`, or `name (2)` and on when taken (before the extension).
fn unique_like(dir: &Path, name: &str) -> PathBuf {
    let first = dir.join(name);
    if !first.exists() {
        return first;
    }
    let (stem, ext) = match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => (stem.to_string(), format!(".{ext}")),
        _ => (name.to_string(), String::new()),
    };
    (2..).map(|n| dir.join(format!("{stem} ({n}){ext}"))).find(|p| !p.exists()).unwrap_or(first)
}

/// A board's audits whose files sit in its folder.
fn own_audits_beside(rel_path: &str) -> Vec<String> {
    if !rel_path.ends_with(".board") {
        return Vec::new();
    }
    let folder = Path::new(rel_path).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    audit_boards()
        .into_iter()
        .filter(|(audit, board)| board == rel_path && Path::new(audit).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default() == folder)
        .map(|(audit, _)| audit)
        .collect()
}

/// Archived (`on`) or brought back: a board or audit in its own file,
/// anything else in its folder's `.folder`. Archiving a board archives its
/// audits too.
pub fn archive(rel_path: &str, on: bool) -> Result<(), String> {
    let path = abs(rel_path).ok_or("no projects folder")?;
    if !path.exists() {
        return Err(format!("{rel_path} is gone"));
    }
    match Kind::of_path(&path) {
        Some(kind) => {
            let line = serde_json::json!({"op": "archived", "on": on}).as_object().cloned().unwrap_or_default();
            log::append(&path, vec![line])?;
            if kind == Kind::Board {
                for (audit, board) in audit_boards() {
                    if board == rel_path
                        && let Some(audit_path) = abs(&audit)
                    {
                        let line = serde_json::json!({"op": "archived", "on": on, "with": "its board"}).as_object().cloned().unwrap_or_default();
                        let _ = log::append(&audit_path, vec![line]);
                    }
                }
            }
        }
        None => {
            let dir = path.parent().ok_or("no folder")?;
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or("no name")?;
            mark(dir, &name, on)?;
        }
    }
    super::scan();
    Ok(())
}

// ---------------------------------------------------------------- the trash

#[derive(Clone, Debug, PartialEq)]
pub struct Trashed {
    /// Its folder in the trash, under the root.
    pub slot: String,
    pub name: String,
    pub from: String,
    pub at: String,
}

fn trash_dir() -> Result<PathBuf, String> {
    let dir = root().ok_or("no projects folder")?.join(TRASH);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// To the trash, with where it came from. A board takes its audits beside
/// it. Restorable until the trash is emptied.
pub fn trash(rel_path: &str) -> Result<(), String> {
    let path = abs(rel_path).ok_or("no projects folder")?;
    if !path.exists() {
        return Err(format!("{rel_path} is gone"));
    }
    let mut going = vec![rel_path.to_string()];
    going.extend(own_audits_beside(rel_path));
    for item in going {
        let Some(from) = abs(&item) else { continue };
        let name = from.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let slot = trash_dir()?.join(format!("{}-{}", chrono::Utc::now().format("%Y%m%dT%H%M%S%3f"), std::process::id() % 1000));
        let slot = unique_like(slot.parent().unwrap_or(Path::new(".")), &slot.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        std::fs::create_dir_all(&slot).map_err(|e| e.to_string())?;
        let origin = serde_json::json!({"from": item, "at": log::now()});
        std::fs::write(slot.join(".origin.json"), origin.to_string()).map_err(|e| e.to_string())?;
        std::fs::rename(&from, slot.join(&name)).map_err(|e| e.to_string())?;
    }
    super::scan();
    Ok(())
}

pub fn trash_list() -> Vec<Trashed> {
    let Ok(dir) = trash_dir() else { return Vec::new() };
    let mut out = Vec::new();
    for slot in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let origin: Value = std::fs::read_to_string(slot.path().join(".origin.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
        let name = std::fs::read_dir(slot.path())
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .find(|n| !n.starts_with('.'))
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        out.push(Trashed {
            slot: rel(&slot.path()).unwrap_or_default(),
            name,
            from: origin.get("from").and_then(Value::as_str).unwrap_or("").to_string(),
            at: origin.get("at").and_then(Value::as_str).unwrap_or("").to_string(),
        });
    }
    out.sort_by(|a, b| b.at.cmp(&a.at));
    out
}

/// Back where it came from (its folder made again if gone).
pub fn restore(slot: &str) -> Result<String, String> {
    let item = trash_list().into_iter().find(|t| t.slot == slot).ok_or("not in the trash")?;
    let root = root().ok_or("no projects folder")?;
    let target = root.join(&item.from);
    let parent = target.parent().unwrap_or(&root).to_path_buf();
    std::fs::create_dir_all(&parent).map_err(|e| e.to_string())?;
    let to = unique_like(&parent, &item.name);
    std::fs::rename(root.join(&item.slot).join(&item.name), &to).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(root.join(&item.slot));
    super::scan();
    rel(&to).ok_or_else(|| "outside the projects folder".to_string())
}

/// Gone for good. An audit's work tree and branch go with it.
pub fn delete_forever(slot: &str) -> Result<(), String> {
    let item = trash_list().into_iter().find(|t| t.slot == slot).ok_or("not in the trash")?;
    let root = root().ok_or("no projects folder")?;
    let inside = root.join(&item.slot);
    for audit in audit_files(&inside) {
        remove_audit_files(&audit);
    }
    std::fs::remove_dir_all(&inside).map_err(|e| e.to_string())
}

pub fn empty_trash() -> Result<usize, String> {
    let all = trash_list();
    let mut done = 0;
    for item in &all {
        delete_forever(&item.slot)?;
        done += 1;
    }
    Ok(done)
}

fn audit_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            out.extend(audit_files(&path));
        } else if path.extension().and_then(|e| e.to_str()) == Some("audit") {
            out.push(path);
        }
    }
    out
}

/// An audit's work tree and branch, from what its file says of it.
fn remove_audit_files(path: &Path) {
    if cfg!(test) {
        return;
    }
    let Ok(log) = log::read(path) else { return };
    let mut row = None;
    for line in log.current() {
        if log::op(line) == "snapshot" {
            for item in line.get("rows").and_then(Value::as_array).into_iter().flatten() {
                if item.get("table").and_then(Value::as_str) == Some("audits") {
                    row = item.get("row").cloned();
                }
            }
        } else if line.get("table").and_then(Value::as_str) == Some("audits") && line.get("row").is_some() {
            row = line.get("row").cloned();
        }
    }
    let audit: Option<crate::audit::Audit> = row.and_then(|r| r.get("data").and_then(Value::as_str).and_then(|d| serde_json::from_str(d).ok()));
    if let Some(audit) = audit {
        let _ = crate::audit::run::remove_files(&audit);
    }
}

