//! The agent's project tools: read a project, comment and reply (on a
//! board, a row of one, a report, a claim, or a page), and write a page.
//! Moving, renaming and deleting stay the operator's: no tool does them.

use serde_json::{Value, json};

use super::tree::{self, NodeKind};
use super::{pages, summary};
use crate::board::comments::{self, AGENT, OPERATOR, Thread};
use crate::llm::ToolDefinition;
use crate::types::ToolCall;

pub const TOOLS: [&str; 4] = ["project_read", "comment", "comment_reply", "page_write"];

/// Most a read returns.
const READ_CHARS: usize = 24_000;
/// Versions of a page kept from before the agent's writes.
const HISTORY_KEPT: usize = 20;
const MAX_PAGE_CHARS: usize = 200_000;

pub fn handles(name: &str) -> bool {
    TOOLS.contains(&name)
}

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::function(
            "project_read",
            "Read the projects in /share/projects. With no path: every project and the boards in it, with \
             their stage and the comments waiting on you. With a folder's path (calc/parser): its page \
             (index.md), what is in it, and its open comments. With a page, board or audit path: it, with its \
             open comments and their replies.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "path": {"type": "string", "description": "Inside projects/, e.g. calc/parser or calc/parser/fix-overflow."}
            }}),
        ),
        ToolDefinition::function(
            "comment",
            "Start a comment thread, to ask the operator something or show him something: on a board (its path \
             or name), a row of it (board#i8, board#fact-0003), a round's report (board#report:2), an audit's \
             claim (board#claim:slice/c4), or a page (calc/notes.md, or a folder for its index.md). It shows \
             as waiting on him until he replies or resolves it.",
            json!({"type": "object", "additionalProperties": false, "required": ["on", "text"], "properties": {
                "on": {"type": "string"},
                "text": {"type": "string"}
            }}),
        ),
        ToolDefinition::function(
            "comment_reply",
            "Reply in a comment's thread (by any comment's id in it, e.g. c-0012): the answer to the operator's \
             comment or review note. `resolve` closes the thread too: only a thread you started.",
            json!({"type": "object", "additionalProperties": false, "required": ["comment", "text"], "properties": {
                "comment": {"type": "string"},
                "text": {"type": "string"},
                "resolve": {"type": "boolean"}
            }}),
        ),
        ToolDefinition::function(
            "page_write",
            "Write a markdown page in a project: a folder's path writes its index.md, a path ending in .md that \
             page (made, with its folders, if new). The version it replaces is kept in \
             projects/.s0ma/history/ for the operator to see and undo. Refused while he has the page open \
             with unsaved changes.",
            json!({"type": "object", "additionalProperties": false, "required": ["path", "markdown"], "properties": {
                "path": {"type": "string"},
                "markdown": {"type": "string"}
            }}),
        ),
    ]
}

fn rejected(error: impl Into<String>) -> String {
    json!({"ok": false, "error": error.into()}).to_string()
}

/// A path inside projects/: cleaned, never out of it.
fn clean(path: &str) -> Result<String, String> {
    let mut path = path.trim().trim_start_matches('/');
    for prefix in ["share/projects/", "projects/"] {
        path = path.strip_prefix(prefix).unwrap_or(path);
    }
    let path = path.trim_end_matches('/');
    if path.split('/').any(|part| part == ".." || part.starts_with('.')) {
        return Err(format!("'{path}' is not a path inside projects/"));
    }
    Ok(path.to_string())
}

pub fn execute(call: &ToolCall) -> Result<String, Box<dyn std::error::Error>> {
    if super::root().is_none() {
        return Ok(rejected("projects are off here"));
    }
    let arguments: Value = serde_json::from_str(if call.function.arguments.trim().is_empty() { "{}" } else { &call.function.arguments })?;
    let text = |key: &str| arguments.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    let Some(boards) = crate::audit::boards() else { return Ok(rejected("the board store is not ready")) };
    Ok(match call.function.name.as_str() {
        "project_read" => read(&boards, &text("path")),
        "comment" => comment(&boards, &text("on"), &text("text")),
        "comment_reply" => reply(&boards, &text("comment"), &text("text"), arguments.get("resolve").and_then(Value::as_bool).unwrap_or(false)),
        "page_write" => page_write(&text("path"), &text("markdown")),
        other => rejected(format!("unknown project tool '{other}'")),
    })
}

fn threads_json(threads: &[Thread]) -> Vec<Value> {
    threads
        .iter()
        .filter(|t| t.root.open())
        .map(|t| {
            json!({
                "id": t.root.id, "on": t.on_said(), "by": t.root.author, "text": t.root.text,
                "replies": t.replies.iter().map(|r| json!({"id": r.id, "by": r.author, "text": r.text})).collect::<Vec<_>>(),
                "waits_on": if t.waits_on_operator() { "the operator" } else { "you" },
            })
        })
        .collect()
}

fn clipped(text: String) -> String {
    if text.len() <= READ_CHARS {
        return text;
    }
    let mut end = READ_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n\n[… {} more characters]", &text[..end], text.len() - end)
}

fn read(boards: &crate::board::BoardStore, path: &str) -> String {
    let path = match clean(path) {
        Ok(path) => path,
        Err(error) => return rejected(error),
    };
    let all_open: std::collections::HashSet<String> = folders();
    if path.is_empty() {
        let nodes = tree::list(&all_open, false);
        let mut projects: Vec<Value> = Vec::new();
        for node in nodes.iter().filter(|n| n.depth == 0 && matches!(n.kind, NodeKind::Project | NodeKind::Folder)) {
            let inside: Vec<Value> = nodes
                .iter()
                .filter(|n| n.kind == NodeKind::Board && n.rel.starts_with(&format!("{}/", node.rel)))
                .filter_map(|n| board_line(boards, &n.rel))
                .collect();
            projects.push(json!({"project": node.rel, "boards": inside}));
        }
        let loose: Vec<Value> = nodes.iter().filter(|n| n.depth == 0 && n.kind == NodeKind::Board).filter_map(|n| board_line(boards, &n.rel)).collect();
        let mut out = json!({"ok": true, "projects": projects});
        if !loose.is_empty() {
            out["boards_at_the_top"] = json!(loose);
        }
        return out.to_string();
    }
    let Some(abs) = super::abs(&path) else { return rejected("no projects folder") };
    let board_path = if path.ends_with(".board") { path.clone() } else { format!("{path}.board") };
    if let Some(board_abs) = super::abs(&board_path).filter(|p| p.is_file()) {
        let Some(info) = summary::board(&board_abs) else { return rejected(format!("{board_path} can't be read as a board")) };
        let page = summary::board_page(&board_path, &info, &[]);
        let threads = boards.threads(&info.id).unwrap_or_default();
        return json!({"ok": true, "board": board_path.trim_end_matches(".board"), "page": clipped(page), "comments": threads_json(&threads)}).to_string();
    }
    if abs.is_file() && path.ends_with(".audit") {
        let Some(info) = summary::audit(&abs) else { return rejected(format!("{path} can't be read as an audit")) };
        return json!({"ok": true, "audit": path, "page": clipped(summary::audit_page(&path, &info))}).to_string();
    }
    if abs.is_file() {
        let Ok(text) = std::fs::read_to_string(&abs) else { return rejected(format!("{path} is not text")) };
        return json!({"ok": true, "page": path, "text": clipped(text), "comments": threads_json(&pages::threads(&path))}).to_string();
    }
    if abs.is_dir() {
        let index = format!("{path}/{}", tree::PAGE);
        let text = std::fs::read_to_string(abs.join(tree::PAGE)).unwrap_or_default();
        let open: std::collections::HashSet<String> = [path.clone()].into_iter().collect();
        let depth = path.matches('/').count() + 1;
        let children: Vec<Value> = tree::list(&open.union(&parents(&path)).cloned().collect(), false)
            .into_iter()
            .filter(|n| n.depth == depth && n.rel.starts_with(&format!("{path}/")) && !n.attached)
            .map(|n| match n.kind {
                NodeKind::Board => board_line(boards, &n.rel).unwrap_or_else(|| json!({"board": n.rel})),
                NodeKind::Audit => json!({"audit": n.rel}),
                NodeKind::Folder | NodeKind::Project => json!({"folder": n.rel}),
                _ => json!({"page": n.rel}),
            })
            .collect();
        return json!({"ok": true, "folder": path, "page": index, "text": clipped(text), "inside": children,
            "comments": threads_json(&pages::threads(&index))})
        .to_string();
    }
    rejected(format!("nothing at projects/{path}"))
}

fn parents(path: &str) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let mut at = std::path::Path::new(path).parent();
    while let Some(p) = at.filter(|p| !p.as_os_str().is_empty()) {
        out.insert(p.to_string_lossy().into_owned());
        at = p.parent();
    }
    out
}

fn folders() -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let Some(root) = super::root() else { return out };
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with('.') && entry.file_type().is_ok_and(|t| t.is_dir()) {
                if let Some(rel) = super::rel(&entry.path()) {
                    out.insert(rel);
                }
                stack.push(entry.path());
            }
        }
    }
    out
}

fn board_line(boards: &crate::board::BoardStore, rel_path: &str) -> Option<Value> {
    let info = summary::board(&super::abs(rel_path)?)?;
    let threads = boards.threads(&info.id).unwrap_or_default();
    let waiting = threads.iter().filter(|t| t.waits_on_agent()).count();
    let mut line = json!({"board": rel_path.trim_end_matches(".board"), "stage": info.stage, "status": info.status,
        "steps": format!("{}/{}", info.steps_used, info.budget)});
    if waiting > 0 {
        line["comments_waiting_on_you"] = json!(waiting);
    }
    Some(line)
}

/// What a comment is on: a board (and a row of it), or a page.
enum Target {
    Board { id: String, row: String },
    Page(String),
}

fn target(boards: &crate::board::BoardStore, on: &str) -> Result<Target, String> {
    let (what, row) = match on.split_once('#') {
        Some((what, row)) => (what.trim(), row.trim()),
        None => (on.trim(), ""),
    };
    if let Ok(Some(board)) = boards.board(what) {
        return Ok(Target::Board { id: board.id, row: if row.is_empty() { "board".into() } else { row.to_string() } });
    }
    if let Ok(Some(board)) = boards.board_by_id(what) {
        return Ok(Target::Board { id: board.id, row: if row.is_empty() { "board".into() } else { row.to_string() } });
    }
    let path = clean(what)?;
    let abs = super::abs(&path).ok_or("no projects folder")?;
    if abs.is_dir() {
        return Ok(Target::Page(format!("{path}/{}", tree::PAGE)));
    }
    if abs.is_file() && path.ends_with(".md") {
        return Ok(Target::Page(path));
    }
    Err(format!("no board or page '{what}'"))
}

fn comment(boards: &crate::board::BoardStore, on: &str, text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return rejected("the comment is empty");
    }
    match target(boards, on) {
        Ok(Target::Board { id, row }) => match boards.comment(&id, &row, None, AGENT, text) {
            Ok(comment) => json!({"ok": true, "id": comment.id, "on": comments::said_on(&row),
                "note": "It waits on the operator until he replies or resolves it."}).to_string(),
            Err(error) => rejected(error.to_string()),
        },
        Ok(Target::Page(page)) => match pages::comment(&page, AGENT, text) {
            Ok(comment) => json!({"ok": true, "id": comment.id, "on": page}).to_string(),
            Err(error) => rejected(error),
        },
        Err(error) => rejected(error),
    }
}

/// `/comment <on> <text>`: the operator's comment, from the prompt.
pub fn operator_comment(on: &str, text: &str) -> Result<String, String> {
    let boards = crate::audit::boards().ok_or("the board store is not ready")?;
    let text = text.trim();
    if text.is_empty() {
        return Err("/comment <board or page>[#row] <text>".into());
    }
    match target(&boards, on)? {
        Target::Board { id, row } => boards.comment(&id, &row, None, OPERATOR, text).map(|c| format!("{} on {}", c.id, comments::said_on(&row))).map_err(|e| e.to_string()),
        Target::Page(page) => {
            let comment = pages::comment(&page, OPERATOR, text)?;
            crate::audit::ops::needs_s0ma(format!("· a comment from the operator on projects/{page}: {text} (project_read it; comment_reply to answer)"));
            Ok(format!("{} on {page}", comment.id))
        }
    }
}

/// `/resolve <comment>`: a thread resolved by the operator.
pub fn operator_resolve(id: &str) -> Result<String, String> {
    let boards = crate::audit::boards().ok_or("the board store is not ready")?;
    let id = id.trim();
    if let Ok(Some(found)) = boards.comment_by_id(id) {
        return match boards.resolve(&found.board_id, id, OPERATOR) {
            Ok(true) => Ok(format!("{id} resolved")),
            Ok(false) => Err(format!("{id} is resolved already")),
            Err(error) => Err(error.to_string()),
        };
    }
    match pages::resolve(id, OPERATOR, true)? {
        true => Ok(format!("{id} resolved")),
        false => Err(format!("{id} is resolved already")),
    }
}

fn reply(boards: &crate::board::BoardStore, id: &str, text: &str, resolve: bool) -> String {
    let (id, text) = (id.trim(), text.trim());
    if text.is_empty() {
        return rejected("the reply is empty");
    }
    match boards.comment_by_id(id) {
        Ok(Some(found)) => {
            let root = found.parent.clone().unwrap_or(found.id.clone());
            let root_comment = boards.comment_by_id(&root).ok().flatten();
            if resolve && root_comment.as_ref().is_some_and(|c| c.author == OPERATOR) {
                return rejected("only the operator resolves a thread he started: reply, and he resolves it");
            }
            match boards.reply(&found.board_id, id, AGENT, text) {
                Ok(Ok(reply)) => {
                    if resolve {
                        let _ = boards.resolve(&found.board_id, &root, AGENT);
                    }
                    json!({"ok": true, "id": reply.id, "thread": root, "resolved": resolve}).to_string()
                }
                Ok(Err(error)) => rejected(error),
                Err(error) => rejected(error.to_string()),
            }
        }
        _ => {
            if pages::page_of(id).is_none() {
                return rejected(format!("no comment {id}"));
            }
            let thread = pages::page_of(id).map(|page| pages::threads(&page)).unwrap_or_default();
            let root = thread.iter().find(|t| t.root.id == id || t.replies.iter().any(|r| r.id == id)).map(|t| t.root.clone());
            if resolve && root.as_ref().is_some_and(|c| c.author == OPERATOR) {
                return rejected("only the operator resolves a thread he started: reply, and he resolves it");
            }
            match pages::reply(id, AGENT, text) {
                Ok(reply) => {
                    if resolve {
                        let _ = pages::resolve(id, AGENT, true);
                    }
                    json!({"ok": true, "id": reply.id, "resolved": resolve}).to_string()
                }
                Err(error) => rejected(error),
            }
        }
    }
}

fn page_write(path: &str, markdown: &str) -> String {
    let path = match clean(path) {
        Ok(path) if !path.is_empty() => path,
        Ok(_) => return rejected("a path inside projects/ is needed"),
        Err(error) => return rejected(error),
    };
    if markdown.len() > MAX_PAGE_CHARS {
        return rejected(format!("at most {MAX_PAGE_CHARS} characters"));
    }
    let Some(abs) = super::abs(&path) else { return rejected("no projects folder") };
    let page = if abs.is_dir() || !path.ends_with(".md") {
        if abs.is_file() {
            return rejected(format!("{path} is not a page"));
        }
        format!("{path}/{}", tree::PAGE)
    } else {
        path
    };
    let Some(file) = super::abs(&page) else { return rejected("no projects folder") };
    // One editor at a time: the operator's unsaved edit is his.
    let shared = format!("projects/{page}");
    if crate::tabs::editor_has(&crate::share::share_dir(), &shared) == Some(true) {
        return rejected(format!("the operator has {page} open with unsaved changes: comment on it instead, or write it later"));
    }
    let before = std::fs::read_to_string(&file).ok();
    if before.as_deref() == Some(markdown) {
        return json!({"ok": true, "path": page, "unchanged": true}).to_string();
    }
    let mut kept = None;
    if let (Some(before), Some(root)) = (&before, super::root()) {
        let dir = root.join(".s0ma").join("history").join(&page);
        let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        if std::fs::create_dir_all(&dir).is_ok() && std::fs::write(dir.join(format!("{stamp}.md")), before).is_ok() {
            kept = super::rel(&dir.join(format!("{stamp}.md")));
            let mut versions: Vec<_> = std::fs::read_dir(&dir).into_iter().flatten().flatten().map(|e| e.path()).collect();
            versions.sort();
            for old in versions.iter().take(versions.len().saturating_sub(HISTORY_KEPT)) {
                let _ = std::fs::remove_file(old);
            }
        }
    }
    if let Some(parent) = file.parent()
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        return rejected(error.to_string());
    }
    if let Err(error) = crate::util::write_atomic(&file, markdown) {
        return rejected(error.to_string());
    }
    let mut out = json!({"ok": true, "path": page, "created": before.is_none()});
    if let Some(kept) = kept {
        out["previous_kept_at"] = json!(kept);
    }
    out.to_string()
}
