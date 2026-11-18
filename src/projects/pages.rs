//! Comments on pages: kept in the page's folder's `.folder`, so they move
//! with the folder. A line per comment (`op: comment`, what it is on by the
//! page's name in the folder, its thread's first comment as `parent`) and
//! per resolve or reopen (`op: resolve`, `on`).

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::log;
use super::tree;
use crate::board::comments::{Comment, Thread, OPERATOR, group};

fn folder_and_name(page_rel: &str) -> Result<(PathBuf, String), String> {
    let path = super::abs(page_rel).ok_or("no projects folder")?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or("not a page")?;
    let dir = path.parent().ok_or("not a page")?.to_path_buf();
    Ok((dir, name))
}

fn new_id() -> String {
    format!("c-{}", chrono::Local::now().format("%y%m%d%H%M%S%6f"))
}

/// A new thread on a page.
pub fn comment(page_rel: &str, author: &str, text: &str) -> Result<Comment, String> {
    let (dir, name) = folder_and_name(page_rel)?;
    write(&dir, &name, author, None, text)
}

fn write(dir: &Path, name: &str, author: &str, parent: Option<&str>, text: &str) -> Result<Comment, String> {
    let file = tree::folder_log(dir)?;
    let id = new_id();
    let mut line = Map::new();
    line.insert("op".into(), "comment".into());
    line.insert("id".into(), id.clone().into());
    line.insert("on".into(), name.into());
    line.insert("author".into(), author.into());
    line.insert("parent".into(), parent.map_or(Value::Null, |p| p.into()));
    line.insert("text".into(), text.into());
    log::append(&file, vec![line])?;
    Ok(Comment {
        id,
        board_id: String::new(),
        on: name.to_string(),
        round: None,
        author: author.to_string(),
        parent: parent.map(str::to_string),
        text: text.to_string(),
        created_at: log::now(),
        resolved_at: None,
        resolved_by: None,
        seen_at: None,
    })
}

/// Every thread in a folder's `.folder`, by the page it is on.
fn folder_threads(dir: &Path) -> Vec<Thread> {
    let Ok(file) = log::read(&dir.join(tree::FOLDER_FILE)) else { return Vec::new() };
    let mut comments: Vec<Comment> = Vec::new();
    for line in &file.lines {
        let text = |key: &str| line.get(key).and_then(Value::as_str).map(str::to_string);
        match log::op(line) {
            "comment" => comments.push(Comment {
                id: text("id").unwrap_or_default(),
                board_id: String::new(),
                on: text("on").unwrap_or_default(),
                round: None,
                author: text("author").unwrap_or_else(|| OPERATOR.into()),
                parent: text("parent"),
                text: text("text").unwrap_or_default(),
                created_at: text("at").unwrap_or_default(),
                resolved_at: None,
                resolved_by: None,
                seen_at: None,
            }),
            "resolve" => {
                let id = text("id").unwrap_or_default();
                let on = line.get("on").and_then(Value::as_bool).unwrap_or(true);
                if let Some(comment) = comments.iter_mut().find(|c| c.id == id) {
                    comment.resolved_at = on.then(|| text("at").unwrap_or_default());
                    comment.resolved_by = on.then(|| text("by").unwrap_or_default());
                }
            }
            _ => {}
        }
    }
    group(comments)
}

/// A page's threads, oldest first.
pub fn threads(page_rel: &str) -> Vec<Thread> {
    let Ok((dir, name)) = folder_and_name(page_rel) else { return Vec::new() };
    folder_threads(&dir).into_iter().filter(|t| t.root.on == name).collect()
}

/// A page comment by id: its folder and thread.
fn find(id: &str) -> Option<(PathBuf, Thread)> {
    let root = super::root()?;
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        if let Some(thread) = folder_threads(&dir).into_iter().find(|t| t.root.id == id || t.replies.iter().any(|r| r.id == id)) {
            return Some((dir, thread));
        }
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with('.') && entry.file_type().is_ok_and(|t| t.is_dir()) {
                stack.push(entry.path());
            }
        }
    }
    None
}

/// The page a comment is on, by its id, if it is a page's.
pub fn page_of(id: &str) -> Option<String> {
    let (dir, thread) = find(id)?;
    super::rel(&dir.join(&thread.root.on))
}

/// A reply in a page comment's thread; the operator's opens a resolved one
/// again. Err: no such page comment.
pub fn reply(id: &str, author: &str, text: &str) -> Result<Comment, String> {
    let (dir, thread) = find(id).ok_or_else(|| format!("no comment {id}"))?;
    let reply = write(&dir, &thread.root.on, author, Some(&thread.root.id), text)?;
    if author == OPERATOR && !thread.root.open() {
        resolve_line(&dir, &thread.root.id, author, false)?;
    }
    Ok(reply)
}

/// A page comment's thread resolved (`on`) or opened again. Whether it
/// changed.
pub fn resolve(id: &str, by: &str, on: bool) -> Result<bool, String> {
    let (dir, thread) = find(id).ok_or_else(|| format!("no comment {id}"))?;
    if thread.root.open() != on {
        return Ok(false);
    }
    resolve_line(&dir, &thread.root.id, by, on)?;
    Ok(true)
}

fn resolve_line(dir: &Path, id: &str, by: &str, on: bool) -> Result<(), String> {
    let file = tree::folder_log(dir)?;
    let mut line = Map::new();
    line.insert("op".into(), "resolve".into());
    line.insert("id".into(), id.into());
    line.insert("by".into(), by.into());
    line.insert("on".into(), on.into());
    log::append(&file, vec![line]).map(|_| ())
}
