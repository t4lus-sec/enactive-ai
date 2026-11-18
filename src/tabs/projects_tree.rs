//! The projects tab: the project tree beside the selected item's page, as
//! the share tab shows its folder. One click (or ↑↓) shows an item's page:
//! a folder's `index.md` (with what is in it), a markdown file, a board's
//! or an audit's page made from its file, another file's text. As in the
//! share tab, ⏎ gives the page the keys (↑↓ scroll it, Esc gives them back)
//! and `e` edits a page in the editor, which closes back here. A double
//! click on a board or an audit (or `o`) opens it: a board in its stage's
//! subtab, an audit in the audit tab, Esc coming back here. The wheel
//! scrolls what it is over, the page or the tree. Folders, pages, boards
//! and audits are made, renamed, moved, archived and trashed here
//! (`projects/tree.rs`).
//!
//! It is the projects tab's first subtab (Tab steps to the boards by
//! stage). The tab keeps its place (the open folders, the selection, the
//! scroll) across tab switches and restarts, in `projects-tab.json` beside
//! the archive; Alt+P comes back to a board opened from here.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Mutex;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde::{Deserialize, Serialize};

use super::gui_view::{Click, Pick, pick_row};
use super::{Footer, Open, Outcome, Target, View, clip, go_to, text_width};
use crate::projects::summary;
use crate::projects::tree::{self, Node, NodeKind};
use crate::style::{Line, Style};

/// What is kept across switches and restarts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct Saved {
    open: BTreeSet<String>,
    selected: Option<String>,
    top: usize,
    page_scroll: usize,
    show_archived: bool,
    trash_open: bool,
    /// Where the selection was: what is selected when it goes (archived,
    /// trashed, moved by hand), until it is back.
    at: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum Purpose {
    Folder(String),
    Page(String),
    Project,
    Board(String),
    Goal { folder: String, name: String },
    Audit { folder: String, board: String },
    Rename(String),
    /// A new thread on a board (`on` its row) or a page.
    Comment(On),
    /// A reply in a thread.
    Reply(On, String),
}

/// What a comment is on, from the tab.
#[derive(Clone, Debug, PartialEq)]
enum On {
    Board { id: String, on: String },
    Page(String),
}

/// `x`: the open threads of the selected item, to resolve or reply to.
#[derive(Clone, Debug)]
struct Threads {
    on: On,
    /// (id, line), each thread's.
    list: Vec<(String, String)>,
    pick: usize,
}

#[derive(Clone, Debug)]
struct Input {
    purpose: Purpose,
    text: Vec<char>,
    cursor: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PageAt {
    Column(usize),
    Row(usize),
}

#[derive(Default)]
struct Live {
    saved: Saved,
    loaded: bool,
    input: Option<Input>,
    /// `n`: what to make, picked next.
    making: bool,
    /// `m`: the folder to move the selected item to.
    moving: Option<(String, Vec<String>, usize)>,
    threads: Option<Threads>,
    armed: Option<(char, String)>,
    /// The page as last rendered: for what, how wide, its lines.
    page: Option<(String, usize, Vec<String>)>,
    /// Opened from here: Esc in the board's or audit's view comes back.
    returning: bool,
    /// A click just opened this folder: the double click's ⏎ leaves it open.
    clicked_open: Option<(String, std::time::Instant)>,
    /// The row last clicked, and when: its ⏎ a double click's.
    clicked: Option<(String, std::time::Instant)>,
    /// ⏎ on an item: its page has the keys (↑↓ scroll it), as the share
    /// tab's file strip does.
    viewing: bool,
    /// Where the page was drawn: beside the tree from this column, or under
    /// it from this body row. For the wheel.
    page_at: Option<PageAt>,
    /// `e`: the page being read for the editor (its path, the copy here).
    opening: Option<(String, String, Opening)>,
    tree_rows: usize,
    page_rows: usize,
}

static LIVE: Mutex<Option<Live>> = Mutex::new(None);

fn with<T>(f: impl FnOnce(&mut Live) -> T) -> T {
    let mut guard = crate::util::locked(&LIVE);
    let live = guard.get_or_insert_with(Live::default);
    if !live.loaded {
        live.loaded = true;
        if let Some(path) = saved_path()
            && let Ok(text) = std::fs::read_to_string(path)
            && let Ok(saved) = serde_json::from_str(&text)
        {
            live.saved = saved;
        }
    }
    f(live)
}

fn saved_path() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    crate::projects::store_path(crate::projects::journal::Store::Board).and_then(|p| p.parent().map(|d| d.join("projects-tab.json")))
}

fn save(live: &Live) {
    if let Some(path) = saved_path()
        && let Ok(text) = serde_json::to_string(&live.saved)
    {
        let aside = path.with_extension("json.tmp");
        if std::fs::write(&aside, text).is_ok() {
            let _ = std::fs::rename(&aside, &path);
        }
    }
}

// ------------------------------------------------------------------ rows

#[derive(Clone, Debug, PartialEq)]
enum Row {
    Node(Node),
    Trash(usize),
    Trashed(tree::Trashed),
}

impl Row {
    fn key(&self) -> String {
        match self {
            Row::Node(node) => node.rel.clone(),
            Row::Trash(_) => "\u{0}trash".into(),
            Row::Trashed(item) => format!("\u{0}trash:{}", item.slot),
        }
    }
}

fn all_folders() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Some(root) = crate::projects::root() else { return out };
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            if let Some(rel) = crate::projects::rel(&entry.path()) {
                out.insert(rel);
            }
            stack.push(entry.path());
        }
    }
    out
}

fn board_info(rel: &str) -> Option<summary::BoardInfo> {
    summary::board(&crate::projects::abs(rel)?)
}

fn rows(saved: &Saved) -> Vec<Row> {
    let open: std::collections::HashSet<String> = saved.open.iter().cloned().collect();
    let mut out: Vec<Row> = tree::list(&open, saved.show_archived).into_iter().map(Row::Node).collect();
    let trashed = tree::trash_list();
    if !trashed.is_empty() {
        out.push(Row::Trash(trashed.len()));
        if saved.trash_open {
            out.extend(trashed.into_iter().map(Row::Trashed));
        }
    }
    out
}

fn selected_index(saved: &Saved, rows: &[Row]) -> Option<usize> {
    if rows.is_empty() {
        return None;
    }
    Some(saved.selected.as_ref().and_then(|key| rows.iter().position(|r| &r.key() == key)).unwrap_or(saved.at.min(rows.len() - 1)))
}

// ------------------------------------------------------------------ drawing

fn mark_of(row: &Row) -> (Style, String) {
    match row {
        Row::Node(node) if node.archived || node.hidden => (Style::Dim, "[~] ".into()),
        Row::Node(node) => match node.kind {
            NodeKind::Board => match board_info(&node.rel) {
                Some(info) if info.stage == "review" || info.stage == "verification" => (Style::Warn, "[!] ".into()),
                Some(info) if info.threads.iter().any(|t| t.waits_on_operator()) => (Style::Warn, "[!] ".into()),
                Some(info) if info.status == "open" => (Style::Warn, "[*] ".into()),
                Some(info) if info.stage == "deliverable" || info.status == "solved" => (Style::Success, "[+] ".into()),
                _ => (Style::Dim, "[ ] ".into()),
            },
            NodeKind::Audit => (Style::Dim, String::new()),
            NodeKind::Page if crate::projects::pages::threads(&node.rel).iter().any(|t| t.waits_on_operator()) => (Style::Warn, "[!] ".into()),
            NodeKind::Project | NodeKind::Folder
                if crate::projects::pages::threads(&format!("{}/{}", node.rel, tree::PAGE)).iter().any(|t| t.waits_on_operator()) =>
            {
                (Style::Warn, "[!] ".into())
            }
            _ => (Style::Plain, String::new()),
        },
        _ => (Style::Plain, String::new()),
    }
}

fn row_line(row: &Row, saved: &Saved, here: bool, width: usize) -> Line {
    let lead = if here { (Style::Cursor, "❯ ".to_string()) } else { (Style::Plain, "  ".to_string()) };
    let mut line: Line = vec![lead];
    let (name, depth, info): (String, usize, String) = match row {
        Row::Node(node) => {
            let fold = match node.kind {
                NodeKind::Project | NodeKind::Folder if node.has_children => if saved.open.contains(&node.rel) { "▾ " } else { "▸ " },
                NodeKind::Project | NodeKind::Folder => "  ",
                _ => "",
            };
            let name = match node.kind {
                NodeKind::Board => node.name.trim_end_matches(".board").to_string(),
                NodeKind::Audit if node.attached => format!("└─ {}", node.name.trim_end_matches(".audit")),
                NodeKind::Audit => node.name.trim_end_matches(".audit").to_string(),
                _ => node.name.clone(),
            };
            let info = match node.kind {
                NodeKind::Board => board_info(&node.rel).map(|i| format!("{} · r{} · {}/{}", if i.stage == "working" { i.status.clone() } else { i.stage.clone() }, i.round, i.steps_used, i.budget)).unwrap_or_default(),
                NodeKind::Audit => crate::projects::abs(&node.rel).and_then(|p| summary::audit(&p)).map(|a| format!("audit · {}", a.status)).unwrap_or_else(|| "audit".into()),
                _ => String::new(),
            };
            let info = if node.archived { format!("archived{}", if info.is_empty() { String::new() } else { format!(" · {info}") }) } else { info };
            (format!("{fold}{name}"), node.depth, info)
        }
        Row::Trash(count) => (format!("{} trash", if saved.trash_open { "▾" } else { "▸" }), 0, count.to_string()),
        Row::Trashed(item) => (format!("  {}", item.name), 1, format!("from {}", item.from)),
    };
    let (mark_style, mark) = mark_of(row);
    // An audit's └─ under its board's mark.
    let indent = match row {
        Row::Node(node) if node.kind == NodeKind::Audit && node.attached => format!("{} ", "  ".repeat(depth.saturating_sub(1))),
        _ => "  ".repeat(depth),
    };
    let style = match row {
        Row::Node(node) if node.archived || node.hidden => Style::Dim,
        Row::Node(node) if matches!(node.kind, NodeKind::Project) => Style::Bold,
        Row::Trash(_) | Row::Trashed(_) => Style::Dim,
        _ => Style::Plain,
    };
    let left = format!("{indent}{name}");
    let used = 2 + text_width(&indent) + text_width(&mark) + text_width(&name);
    let room = width.saturating_sub(used + 2);
    let info = clip(&info, room.min(34));
    let gap = width.saturating_sub(used + text_width(&info) + 1);
    line.push((Style::Plain, indent));
    line.push((mark_style, mark));
    line.push((style, left.trim_start().to_string()));
    line.push((Style::Plain, " ".repeat(gap.max(1))));
    line.push((Style::Dim, info));
    line
}

/// The selected item's page, as markdown.
fn page_text(row: &Row) -> String {
    match row {
        Row::Node(node) => {
            let Some(path) = crate::projects::abs(&node.rel) else { return String::new() };
            match node.kind {
                NodeKind::Project | NodeKind::Folder => {
                    let mut text = std::fs::read_to_string(path.join(tree::PAGE)).unwrap_or_else(|_| format!("# {}\n", node.name));
                    let inside = tree::list(&[node.rel.clone()].into_iter().collect(), false);
                    let children: Vec<&Node> = inside.iter().filter(|n| n.depth == node.depth + 1 && !n.attached).collect();
                    if !children.is_empty() {
                        text.push_str("\n\n---\n\n");
                        for child in children {
                            let line = match child.kind {
                                NodeKind::Folder | NodeKind::Project => format!("- **{}/**\n", child.name),
                                NodeKind::Board => match board_info(&child.rel) {
                                    Some(info) => format!("- **{}** · {} · {}\n", child.name.trim_end_matches(".board"), if info.stage == "working" { info.status } else { info.stage }, if info.title.is_empty() { info.goal } else { info.title }),
                                    None => format!("- **{}**\n", child.name),
                                },
                                _ => format!("- {}\n", child.name),
                            };
                            text.push_str(&line);
                        }
                    }
                    let comments = summary::comments_section(&crate::projects::pages::threads(&format!("{}/{}", node.rel, tree::PAGE)));
                    if !comments.is_empty() {
                        text.push_str(&format!("\n\n{comments}"));
                    }
                    text
                }
                NodeKind::Page => {
                    let mut text = std::fs::read_to_string(&path).unwrap_or_default();
                    let comments = summary::comments_section(&crate::projects::pages::threads(&node.rel));
                    if !comments.is_empty() {
                        text.push_str(&format!("\n\n---\n\n{comments}"));
                    }
                    text
                }
                NodeKind::Board => match summary::board(&path) {
                    Some(info) => {
                        let audits: Vec<(String, summary::AuditInfo)> = tree::list(&parents_open(&node.rel), true)
                            .into_iter()
                            .filter(|n| n.kind == NodeKind::Audit && n.attached)
                            .filter_map(|n| crate::projects::abs(&n.rel).and_then(|p| summary::audit(&p)).filter(|a| a.board.as_deref() == Some(info.name.as_str())).map(|a| (n.name.trim_end_matches(".audit").to_string(), a)))
                            .collect();
                        summary::board_page(&node.rel, &info, &audits)
                    }
                    None => format!("# {}\n\nThis file can't be read as a board.\n", node.name),
                },
                NodeKind::Audit => match summary::audit(&path) {
                    Some(info) => summary::audit_page(&node.rel, &info),
                    None => format!("# {}\n\nThis file can't be read as an audit.\n", node.name),
                },
                NodeKind::File => match std::fs::read(&path) {
                    Ok(bytes) if bytes.len() <= 256 * 1024 && std::str::from_utf8(&bytes).is_ok() => format!("```\n{}\n```\n", String::from_utf8_lossy(&bytes)),
                    Ok(bytes) => format!("*Not text · {} bytes*\n", bytes.len()),
                    Err(error) => format!("*{error}*\n"),
                },
            }
        }
        Row::Trash(count) => {
            let mut text = format!("# Trash · {count}\n\n");
            for item in tree::trash_list() {
                text.push_str(&format!("- **{}** from `{}` · {}\n", item.name, item.from, item.at.get(..16).unwrap_or(&item.at)));
            }
            text
        }
        Row::Trashed(item) => format!("# {}\n\nFrom `{}`, trashed {}.\n", item.name, item.from, item.at.get(..16).unwrap_or(&item.at)),
    }
}

/// The folders above a path, open: to list what sits beside it.
fn parents_open(rel: &str) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let mut parent = std::path::Path::new(rel).parent();
    while let Some(p) = parent.filter(|p| !p.as_os_str().is_empty()) {
        out.insert(p.to_string_lossy().into_owned());
        parent = p.parent();
    }
    out
}

fn stamp_of(row: &Row) -> String {
    match row {
        Row::Node(node) => {
            let path = crate::projects::abs(&node.rel);
            let own = path.as_ref().and_then(|p| crate::projects::stamp(p));
            let page = path.as_ref().and_then(|p| crate::projects::stamp(&p.join(tree::PAGE)));
            // A page's comments are in its folder's file.
            let folder = path.as_ref().and_then(|p| p.parent()).and_then(|d| crate::projects::stamp(&d.join(tree::FOLDER_FILE)));
            let inner = path.as_ref().and_then(|p| crate::projects::stamp(&p.join(tree::FOLDER_FILE)));
            format!("{}:{own:?}:{page:?}:{folder:?}:{inner:?}:{}", node.rel, node.archived)
        }
        Row::Trash(count) => format!("trash:{count}"),
        Row::Trashed(item) => format!("trashed:{}", item.slot),
    }
}

fn input_line(input: &Input) -> Line {
    let label = match &input.purpose {
        Purpose::Folder(_) => "new folder › ",
        Purpose::Page(_) => "new markdown file › ",
        Purpose::Project => "new project › ",
        Purpose::Board(_) => "new board, its name › ",
        Purpose::Goal { .. } => "its goal › ",
        Purpose::Audit { .. } => "audit what (name path… --scope …) › ",
        Purpose::Rename(_) => "rename › ",
        Purpose::Comment(_) => "comment › ",
        Purpose::Reply(..) => "reply › ",
    };
    let before: String = input.text[..input.cursor].iter().collect();
    let at: String = input.text.get(input.cursor).map_or(" ".into(), |c| c.to_string());
    let after: String = input.text.get(input.cursor + 1..).map(|r| r.iter().collect()).unwrap_or_default();
    vec![(Style::Status, format!("  {label}")), (Style::Plain, before), (Style::Caret, at), (Style::Plain, after)]
}

pub(super) fn body(tab: &mut Open, width: usize, rows_left: usize) -> (Vec<Line>, Footer) {
    with(|live| live.returning = false);
    opened(tab);
    if crate::projects::root().is_none() {
        let lines = vec![Vec::new(), vec![(Style::Dim, "  no projects folder (S0MA_PROJECTS=off)".to_string())]];
        return (lines, Footer::new(vec![(Style::Status, "projects".into())], vec![("esc", "back")]));
    }
    with(|live| {
        let all = rows(&live.saved);
        let at = selected_index(&live.saved, &all);
        if let Some(at) = at {
            live.saved.at = at;
            if live.saved.selected.is_none() {
                live.saved.selected = Some(all[at].key());
            }
        }
        super::projects_view::set_last(View::Projects);
        let mut out: Vec<Line> = vec![super::projects_view::subtab_line(&View::Projects, width)];
        if live.saved.show_archived {
            out[0].push((Style::Dim, "  archived shown".into()));
        }
        out.push(Vec::new());
        let body_rows = rows_left.saturating_sub(out.len() + 1);
        let split = super::in_split();
        let beside = width >= 100 && !split;
        let tree_width = if beside { (width * 45 / 100).clamp(40, 70) } else { width };
        let page_width = if beside { width.saturating_sub(tree_width + 3) } else { width.saturating_sub(2) };
        let (tree_rows, page_rows) = if beside { (body_rows, body_rows) } else { (body_rows / 2, body_rows.saturating_sub(body_rows / 2 + 1)) };
        live.tree_rows = tree_rows;
        live.page_rows = page_rows;
        // The tree, scrolled to keep the selection in view.
        if let Some(at) = at {
            if at < live.saved.top {
                live.saved.top = at;
            }
            if at >= live.saved.top + tree_rows {
                live.saved.top = at + 1 - tree_rows;
            }
        }
        live.saved.top = live.saved.top.min(all.len().saturating_sub(tree_rows.max(1)));
        let mut tree_lines: Vec<Line> = Vec::new();
        if all.is_empty() {
            tree_lines.push(vec![(Style::Dim, "  no projects yet · N makes one".into())]);
        }
        for (i, row) in all.iter().enumerate().skip(live.saved.top).take(tree_rows) {
            let here = Some(i) == at;
            let mut line = row_line(row, &live.saved, here, tree_width);
            // The page has the keys: its row stays marked, dimmed.
            if here && live.viewing
                && let Some(lead) = line.first_mut()
            {
                lead.0 = Style::Dim;
            }
            tree_lines.push(pick_row(line, Click::Pick(Pick::ProjectsRow(i))));
        }
        // The page.
        let page_lines: Vec<String> = match at.map(|i| &all[i]) {
            Some(row) => {
                let key = stamp_of(row);
                let fresh = live.page.as_ref().is_some_and(|(k, w, _)| *k == key && *w == page_width);
                if !fresh {
                    let text = page_text(row);
                    live.page = Some((key, page_width, crate::markdown::render_lines(&text, page_width.max(10))));
                    live.saved.page_scroll = 0;
                }
                live.page.as_ref().map(|(_, _, l)| l.clone()).unwrap_or_default()
            }
            None => Vec::new(),
        };
        live.saved.page_scroll = live.saved.page_scroll.min(page_lines.len().saturating_sub(page_rows.max(1)));
        let shown: Vec<Line> = page_lines
            .iter()
            .skip(live.saved.page_scroll)
            .take(page_rows)
            .map(|row| vec![(Style::Raw, row.clone())])
            .collect();
        let rule = if live.viewing { Style::Cursor } else { Style::Dim };
        live.page_at = Some(if beside { PageAt::Column(tree_width + 3) } else { PageAt::Row(out.len() + tree_rows + 1) });
        if beside {
            for r in 0..body_rows {
                let mut line = super::fit_line(&tree_lines.get(r).cloned().unwrap_or_default(), tree_width);
                line.push((rule, " │ ".into()));
                line.extend(shown.get(r).cloned().unwrap_or_default());
                out.push(line);
            }
        } else {
            out.extend(tree_lines.into_iter().chain(std::iter::repeat_with(Vec::new)).take(tree_rows));
            out.push(vec![(rule, format!("  {}", "─".repeat(width.saturating_sub(4))))]);
            out.extend(shown.into_iter().map(|mut l| {
                l.insert(0, (Style::Plain, "  ".into()));
                l
            }));
        }
        // The prompt, or the folders to move to.
        if let Some((what, folders, pick)) = &live.moving {
            out.truncate(rows_left.saturating_sub(folders.len().min(8) + 2));
            out.push(vec![(Style::Status, format!("  move {what} to:"))]);
            let start = pick.saturating_sub(7);
            for (i, folder) in folders.iter().enumerate().skip(start).take(8) {
                let shown = if folder.is_empty() { "(the projects folder)".to_string() } else { format!("{folder}/") };
                let style = if i == *pick { Style::Cursor } else { Style::Plain };
                out.push(pick_row(vec![(style, format!("    {shown}"))], Click::Pick(Pick::ProjectsMove(i))));
            }
        } else if let Some(threads) = &live.threads
            && live.input.is_none()
        {
            out.truncate(rows_left.saturating_sub(threads.list.len().min(8) + 2));
            out.push(vec![(Style::Status, "  open comments:".to_string())]);
            let start = threads.pick.saturating_sub(7);
            for (i, (_, line)) in threads.list.iter().enumerate().skip(start).take(8) {
                let style = if i == threads.pick { Style::Cursor } else { Style::Plain };
                out.push(pick_row(vec![(style, format!("    {}", clip(line, width.saturating_sub(6))))], Click::Pick(Pick::ProjectsMove(i))));
            }
        } else if let Some(input) = &live.input {
            out.truncate(rows_left.saturating_sub(1));
            out.push(input_line(input));
        }
        out.resize(rows_left, Vec::new());
        // The footer.
        let selected = at.map(|i| all[i].clone());
        let left = vec![(Style::Status, "projects".to_string()), (Style::Dim, format!("  {}", selected.as_ref().map(shown_path).unwrap_or_default()))];
        let hints = hints(live, selected.as_ref());
        save(live);
        (out, Footer::new(left, hints))
    })
}

fn shown_path(row: &Row) -> String {
    match row {
        Row::Node(node) => match node.kind {
            NodeKind::Project | NodeKind::Folder => format!("{}/{}", node.rel, tree::PAGE),
            _ => node.rel.clone(),
        },
        Row::Trash(_) => ".trash".into(),
        Row::Trashed(item) => item.from.clone(),
    }
}

fn hints(live: &Live, row: Option<&Row>) -> Vec<(&'static str, &'static str)> {
    if live.moving.is_some() {
        return vec![("↑↓", "pick"), ("⏎", "move here"), ("esc", "cancel")];
    }
    if live.threads.is_some() && live.input.is_none() {
        return vec![("↑↓", "pick"), ("⏎", "resolve"), ("r", "reply"), ("esc", "close")];
    }
    if live.input.is_some() {
        return vec![("⏎", "ok"), ("esc", "cancel")];
    }
    if live.making {
        let mut out = vec![("f", "folder"), ("m", "markdown file"), ("b", "board")];
        if matches!(row, Some(Row::Node(n)) if n.kind == NodeKind::Board) {
            out.push(("a", "audit"));
        }
        out.push(("esc", "cancel"));
        return out;
    }
    if live.viewing {
        let mut out = vec![("↑↓", "scroll")];
        if matches!(row, Some(Row::Node(n)) if matches!(n.kind, NodeKind::Project | NodeKind::Folder | NodeKind::Page | NodeKind::File)) {
            out.push(("e", "edit"));
        }
        if matches!(row, Some(Row::Node(n)) if matches!(n.kind, NodeKind::Board | NodeKind::Audit)) {
            out.push(("o", "open"));
        }
        out.push(("esc", "back"));
        return out;
    }
    let mut out: Vec<(&'static str, &'static str)> = vec![("tab", "subtab")];
    match row {
        Some(Row::Node(node)) => {
            out.push(("n", "new"));
            out.push(("N", "new project"));
            match node.kind {
                NodeKind::Project | NodeKind::Folder => out.push(("e", "edit")),
                NodeKind::Page | NodeKind::File => {
                    out.push(("⏎", "view"));
                    out.push(("e", "edit"));
                }
                NodeKind::Board | NodeKind::Audit => {
                    out.push(("⏎", "view"));
                    out.push(("o", "open"));
                }
            }
            out.push(("r", "rename"));
            out.push(("m", "move"));
            out.push(("A", if node.archived { "unarchive" } else { "archive" }));
            if !matches!(node.kind, NodeKind::File) {
                out.push(("c", "comment"));
                out.push(("x", "comments"));
            }
            out.push(("del", "delete"));
        }
        Some(Row::Trash(_)) => out.push(("del", "empty trash")),
        Some(Row::Trashed(_)) => {
            out.push(("u", "put back"));
            out.push(("del", "delete for good"));
        }
        None => out.push(("N", "new project")),
    }
    out.push(("h", if live.saved.show_archived { "hide archived" } else { "show archived" }));
    out
}

// ------------------------------------------------------------------ keys

fn note(tab: &mut Open, result: Result<String, String>) {
    match result {
        // Done: the question it answered goes.
        Ok(_) => tab.note("", false),
        Err(error) => tab.note(error, true),
    }
}

fn folder_for(row: Option<&Row>) -> String {
    match row {
        Some(Row::Node(node)) if matches!(node.kind, NodeKind::Project | NodeKind::Folder) => node.rel.clone(),
        Some(Row::Node(node)) => std::path::Path::new(&node.rel).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
        _ => String::new(),
    }
}

pub(super) fn key(tab: &mut Open, key: &KeyEvent) -> Outcome {
    let (all, at) = with(|live| {
        let all = rows(&live.saved);
        let at = selected_index(&live.saved, &all);
        (all, at)
    });
    let row = at.map(|i| all[i].clone());
    // Typing.
    if with(|live| live.input.is_some()) {
        typing(tab, key, row.as_ref());
        return Outcome::Stay;
    }
    // The open comments.
    if let Some(threads) = with(|live| live.threads.clone()) {
        let picked = threads.list.get(threads.pick).map(|(id, _)| id.clone());
        match key.code {
            KeyCode::Up => with(|live| live.threads.as_mut().map(|t| t.pick = t.pick.saturating_sub(1))).unwrap_or(()),
            KeyCode::Down => with(|live| live.threads.as_mut().map(|t| t.pick = (t.pick + 1).min(t.list.len().saturating_sub(1)))).unwrap_or(()),
            KeyCode::Enter => {
                if let Some(id) = picked {
                    note(tab, resolve(&threads.on, &id).map(|()| String::new()));
                    refresh_threads(threads.on.clone(), threads.pick);
                }
            }
            KeyCode::Char('r') => {
                if let Some(id) = picked {
                    start(Purpose::Reply(threads.on.clone(), id), "");
                }
            }
            _ => with(|live| live.threads = None),
        }
        return Outcome::Stay;
    }
    // Picking where to move.
    if let Some((what, folders, pick)) = with(|live| live.moving.clone()) {
        match key.code {
            KeyCode::Up => with(|live| live.moving = Some((what, folders, pick.saturating_sub(1)))),
            KeyCode::Down => with(|live| live.moving = Some((what, folders.clone(), (pick + 1).min(folders.len().saturating_sub(1))))),
            KeyCode::Enter => {
                with(|live| live.moving = None);
                let to = folders.get(pick).cloned().unwrap_or_default();
                let moved = tree::move_to(&what, &to);
                if let Ok(rel) = &moved {
                    select(rel);
                }
                note(tab, moved.map(|rel| format!("moved to {rel}")));
            }
            _ => with(|live| live.moving = None),
        }
        return Outcome::Stay;
    }
    // Picking what to make.
    if with(|live| std::mem::take(&mut live.making)) {
        let folder = folder_for(row.as_ref());
        let purpose = match key.code {
            KeyCode::Char('f') => Some(Purpose::Folder(folder)),
            KeyCode::Char('m') => Some(Purpose::Page(folder)),
            KeyCode::Char('b') => Some(Purpose::Board(folder)),
            KeyCode::Char('a') => match row.as_ref() {
                Some(Row::Node(node)) if node.kind == NodeKind::Board => board_info(&node.rel).map(|info| Purpose::Audit { folder, board: info.name }),
                _ => None,
            },
            _ => None,
        };
        if let Some(purpose) = purpose {
            start(purpose, "");
        }
        return Outcome::Stay;
    }
    // The page has the keys.
    if with(|live| live.viewing) {
        let page = with(|live| live.page_rows.max(1)) as isize;
        match key.code {
            KeyCode::Up => scroll_page(-1),
            KeyCode::Down => scroll_page(1),
            KeyCode::PageUp => scroll_page(-page),
            KeyCode::PageDown => scroll_page(page),
            KeyCode::Home => with(|live| live.saved.page_scroll = 0),
            KeyCode::End => scroll_page(1_000_000),
            KeyCode::Char('e') => {
                with(|live| live.viewing = false);
                if let Some(Row::Node(node)) = row.as_ref() {
                    edit(tab, node);
                }
            }
            KeyCode::Char('o') => {
                with(|live| live.viewing = false);
                if let Some(Row::Node(node)) = row.as_ref() {
                    open_full(tab, node);
                }
            }
            KeyCode::Esc | KeyCode::Enter | KeyCode::Left | KeyCode::Char('q') => with(|live| live.viewing = false),
            _ => {}
        }
        return Outcome::Stay;
    }
    // Shift+↑↓ scroll the page a line, as PgUp/PgDn a page.
    if key.modifiers.contains(KeyModifiers::SHIFT) {
        match key.code {
            KeyCode::Up => return scrolled(-1),
            KeyCode::Down => return scrolled(1),
            _ => {}
        }
    }
    let armed = with(|live| live.armed.take());
    match key.code {
        KeyCode::Esc => return Outcome::Closed,
        KeyCode::Up => move_by(&all, at, -1),
        KeyCode::Down => move_by(&all, at, 1),
        // PgUp/PgDn page the page, as the share tab's strip.
        KeyCode::PageUp => scroll_page(-(with(|l| l.page_rows.max(1)) as isize)),
        KeyCode::PageDown => scroll_page(with(|l| l.page_rows.max(1)) as isize),
        KeyCode::Home => move_by(&all, at, -1_000_000),
        KeyCode::End => move_by(&all, at, 1_000_000),
        KeyCode::Left => match row.as_ref() {
            Some(Row::Node(node)) if matches!(node.kind, NodeKind::Project | NodeKind::Folder) && with(|l| l.saved.open.contains(&node.rel)) => {
                with(|live| live.saved.open.remove(&node.rel));
            }
            Some(Row::Node(node)) => {
                if let Some(parent) = std::path::Path::new(&node.rel).parent().filter(|p| !p.as_os_str().is_empty()) {
                    select(&parent.to_string_lossy());
                }
            }
            Some(Row::Trashed(_)) => select("\u{0}trash"),
            Some(Row::Trash(_)) => with(|live| live.saved.trash_open = false),
            None => {}
        },
        KeyCode::Right => match row.as_ref() {
            Some(Row::Node(node)) if matches!(node.kind, NodeKind::Project | NodeKind::Folder) => {
                with(|live| live.saved.open.insert(node.rel.clone()));
            }
            Some(Row::Trash(_)) => with(|live| live.saved.trash_open = true),
            _ => {}
        },
        KeyCode::Enter => {
            if let Some(row) = row {
                return enter(tab, &row);
            }
        }
        KeyCode::Char('n') => with(|live| live.making = true),
        KeyCode::Char('N') => start(Purpose::Project, ""),
        KeyCode::Char('h') => with(|live| live.saved.show_archived = !live.saved.show_archived),
        KeyCode::Char('e') => {
            if let Some(Row::Node(node)) = row.as_ref() {
                edit(tab, node);
            }
        }
        KeyCode::Char('o') => {
            if let Some(Row::Node(node)) = row.as_ref() {
                open_full(tab, node);
            }
        }
        KeyCode::Char('r') => {
            if let Some(Row::Node(node)) = row.as_ref() {
                start(Purpose::Rename(node.rel.clone()), &node.name);
            }
        }
        KeyCode::Char('m') => {
            if let Some(Row::Node(node)) = row.as_ref() {
                let mut folders: Vec<String> = std::iter::once(String::new()).chain(all_folders()).filter(|f| !f.starts_with(&format!("{}/", node.rel)) && *f != node.rel).collect();
                folders.sort();
                let here = std::path::Path::new(&node.rel).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
                let pick = folders.iter().position(|f| *f == here).unwrap_or(0);
                with(|live| live.moving = Some((node.rel.clone(), folders, pick)));
            }
        }
        KeyCode::Char('c') => {
            if let Some(on) = row.as_ref().and_then(on_of) {
                start(Purpose::Comment(on), "");
            }
        }
        KeyCode::Char('x') => {
            if let Some(on) = row.as_ref().and_then(on_of)
                && !refresh_threads(on, 0)
            {
                tab.note("no open comments", true);
            }
        }
        KeyCode::Char('A') => {
            if let Some(Row::Node(node)) = row.as_ref() {
                let on = !node.archived;
                let done = tree::archive(&node.rel, on).and_then(|()| {
                    // A closed board brought back is open again.
                    match board_info(&node.rel).filter(|_| !on && node.kind == NodeKind::Board) {
                        Some(info) => super::act(super::Action::Project { board: info.id, done: super::projects_view::Move::BringBack }),
                        None => Ok(String::new()),
                    }
                });
                note(tab, done);
            }
        }
        KeyCode::Char('u') => {
            if let Some(Row::Trashed(item)) = row.as_ref() {
                let back = tree::restore(&item.slot);
                if let Ok(rel) = &back {
                    open_parents(rel);
                    select(rel);
                }
                note(tab, back.map(|rel| format!("put back: {rel}")));
            }
        }
        KeyCode::Delete | KeyCode::Char('D') => match row.as_ref() {
            Some(Row::Node(node)) => {
                let what = describe_going(node);
                if twice(tab, armed, 'D', &node.rel, format!("del again moves {what} to the trash")) {
                    note(tab, tree::trash(&node.rel).map(|()| format!("{} is in the trash", node.name)));
                }
            }
            Some(Row::Trashed(item)) => {
                if twice(tab, armed, 'D', &item.slot, format!("del again deletes {} for good", item.name)) {
                    note(tab, tree::delete_forever(&item.slot).map(|()| format!("{} deleted", item.name)));
                }
            }
            Some(Row::Trash(count)) if twice(tab, armed, 'D', "trash", format!("del again deletes all {count} in the trash for good")) => {
                note(tab, tree::empty_trash().map(|n| format!("{n} deleted")));
            }
            Some(Row::Trash(_)) | None => {}
        },
        _ => {}
    }
    Outcome::Stay
}

fn describe_going(node: &Node) -> String {
    match node.kind {
        NodeKind::Project | NodeKind::Folder => {
            let open: std::collections::HashSet<String> = all_folders().into_iter().filter(|f| *f == node.rel || f.starts_with(&format!("{}/", node.rel))).collect();
            let inside = tree::list(&open, true);
            let start = inside.iter().position(|n| n.rel == node.rel).map(|i| i + 1).unwrap_or(inside.len());
            let below: Vec<&Node> = inside[start..].iter().take_while(|n| n.depth > node.depth).collect();
            let boards = below.iter().filter(|n| n.kind == NodeKind::Board).count();
            let audits = below.iter().filter(|n| n.kind == NodeKind::Audit).count();
            let mut parts = Vec::new();
            if boards > 0 {
                parts.push(format!("{boards} board{}", if boards == 1 { "" } else { "s" }));
            }
            if audits > 0 {
                parts.push(format!("{audits} audit{}", if audits == 1 { "" } else { "s" }));
            }
            if parts.is_empty() { format!("{}/", node.name) } else { format!("{}/ with {}", node.name, parts.join(" and ")) }
        }
        NodeKind::Board => format!("{} and its audits", node.name.trim_end_matches(".board")),
        _ => node.name.clone(),
    }
}

fn twice(tab: &mut Open, armed: Option<(char, String)>, key: char, what: &str, say: String) -> bool {
    if armed.as_ref().is_some_and(|(k, w)| *k == key && w == what) {
        return true;
    }
    with(|live| live.armed = Some((key, what.to_string())));
    tab.ask(say);
    false
}

fn move_by(all: &[Row], at: Option<usize>, by: isize) {
    let Some(at) = at else { return };
    let to = (at as isize + by).clamp(0, all.len().saturating_sub(1) as isize) as usize;
    if let Some(row) = all.get(to) {
        let key = row.key();
        with(|live| live.saved.selected = Some(key));
    }
}

fn select(key: &str) {
    with(|live| live.saved.selected = Some(key.to_string()));
}

fn open_parents(rel: &str) {
    let mut parent = std::path::Path::new(rel).parent();
    with(|live| {
        while let Some(p) = parent.filter(|p| !p.as_os_str().is_empty()) {
            live.saved.open.insert(p.to_string_lossy().into_owned());
            parent = p.parent();
        }
    });
}

fn scrolled(by: isize) -> Outcome {
    scroll_page(by);
    Outcome::Stay
}

fn scroll_page(by: isize) {
    with(|live| live.saved.page_scroll = (live.saved.page_scroll as isize + by).max(0) as usize);
}

fn start(purpose: Purpose, text: &str) {
    with(|live| {
        let text: Vec<char> = text.chars().collect();
        let cursor = text.len();
        live.input = Some(Input { purpose, text, cursor });
    });
}

fn typing(tab: &mut Open, key: &KeyEvent, row: Option<&Row>) {
    let _ = row;
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let done = with(|live| {
        let input = live.input.as_mut()?;
        match key.code {
            KeyCode::Esc => {
                live.input = None;
            }
            KeyCode::Enter => return live.input.take(),
            KeyCode::Left => input.cursor = input.cursor.saturating_sub(1),
            KeyCode::Right => input.cursor = (input.cursor + 1).min(input.text.len()),
            KeyCode::Home => input.cursor = 0,
            KeyCode::End => input.cursor = input.text.len(),
            KeyCode::Backspace if input.cursor > 0 => {
                input.cursor -= 1;
                input.text.remove(input.cursor);
            }
            KeyCode::Delete if input.cursor < input.text.len() => {
                input.text.remove(input.cursor);
            }
            KeyCode::Char('u') if ctrl => {
                input.text.drain(..input.cursor);
                input.cursor = 0;
            }
            KeyCode::Char(c) if !ctrl => {
                input.text.insert(input.cursor, c);
                input.cursor += 1;
            }
            _ => {}
        }
        None
    });
    if let Some(input) = done {
        submit(tab, input);
    }
}

pub(super) fn paste(text: &str) {
    with(|live| {
        if let Some(input) = live.input.as_mut() {
            for c in text.chars().filter(|c| !c.is_control()) {
                input.text.insert(input.cursor, c);
                input.cursor += 1;
            }
        }
    });
}

fn submit(tab: &mut Open, input: Input) {
    let text: String = input.text.iter().collect::<String>().trim().to_string();
    if text.is_empty() {
        return;
    }
    let made = match input.purpose {
        Purpose::Project => tree::create_folder("", &text),
        Purpose::Folder(parent) => tree::create_folder(&parent, &text),
        Purpose::Page(parent) => tree::create_page(&parent, &text),
        Purpose::Rename(rel) => tree::rename(&rel, &text),
        Purpose::Board(folder) => {
            start(Purpose::Goal { folder, name: text }, "");
            return;
        }
        Purpose::Goal { folder, name } => make_board(&folder, &name, &text),
        Purpose::Audit { folder, board } => make_audit(&folder, &board, &text),
        Purpose::Comment(on) => {
            let said = match &on {
                On::Board { id, on: what } => crate::audit::boards()
                    .ok_or_else(|| "no board store".to_string())
                    .and_then(|store| store.comment(id, what, None, crate::board::comments::OPERATOR, &text).map_err(|e| e.to_string()))
                    .map(|_| ()),
                On::Page(page) => crate::projects::pages::comment(page, crate::board::comments::OPERATOR, &text).map(|_| ()),
            };
            note(tab, said.map(|()| String::new()));
            told_agent(&on, &text);
            return;
        }
        Purpose::Reply(on, id) => {
            let said = match &on {
                On::Board { id: board, .. } => crate::audit::boards()
                    .ok_or_else(|| "no board store".to_string())
                    .and_then(|store| match store.reply(board, &id, crate::board::comments::OPERATOR, &text) {
                        Ok(Ok(_)) => Ok(()),
                        Ok(Err(error)) => Err(error),
                        Err(error) => Err(error.to_string()),
                    }),
                On::Page(_) => crate::projects::pages::reply(&id, crate::board::comments::OPERATOR, &text).map(|_| ()),
            };
            note(tab, said.map(|()| String::new()));
            told_agent(&on, &text);
            refresh_threads(on, with(|live| live.threads.as_ref().map_or(0, |t| t.pick)));
            return;
        }
    };
    match made {
        Ok(rel) => {
            open_parents(&rel);
            select(&rel);
        }
        Err(error) => tab.note(error, true),
    }
}

/// What `c` and `x` act on for a row: a board, an audit (its board, on
/// the audit), a page, a folder's page.
fn on_of(row: &Row) -> Option<On> {
    let Row::Node(node) = row else { return None };
    match node.kind {
        NodeKind::Board => board_info(&node.rel).map(|info| On::Board { id: info.id, on: "board".into() }),
        NodeKind::Audit => {
            let info = crate::projects::abs(&node.rel).and_then(|p| summary::audit(&p))?;
            let board = crate::audit::boards()?.board(info.board.as_deref()?).ok().flatten()?;
            Some(On::Board { id: board.id, on: format!("audit:{}", info.name) })
        }
        NodeKind::Page => Some(On::Page(node.rel.clone())),
        NodeKind::Project | NodeKind::Folder => Some(On::Page(format!("{}/{}", node.rel, tree::PAGE))),
        NodeKind::File => None,
    }
}

/// The open threads on `on`, listed for `x`; whether there are any.
fn refresh_threads(on: On, pick: usize) -> bool {
    let threads = match &on {
        On::Board { id, on: what } => crate::audit::boards()
            .and_then(|store| store.threads(id).ok())
            .unwrap_or_default()
            .into_iter()
            .filter(|t| what == "board" || t.root.on == *what)
            .collect::<Vec<_>>(),
        On::Page(page) => crate::projects::pages::threads(page),
    };
    let who = |author: &str| if author == crate::board::comments::OPERATOR { "you" } else { "s0ma" };
    let list: Vec<(String, String)> = threads
        .iter()
        .filter(|t| t.root.open())
        .map(|t| {
            let mark = if t.waits_on_operator() { "[!] " } else { "" };
            let last = t.last();
            let replies = if t.replies.is_empty() { String::new() } else { format!(" · {} {}: {}", t.replies.len(), who(&last.author), last.text) };
            (t.root.id.clone(), format!("{mark}{} {}{replies}", who(&t.root.author), t.root.text))
        })
        .collect();
    let any = !list.is_empty();
    with(|live| {
        live.threads = any.then(|| Threads { on, pick: pick.min(list.len().saturating_sub(1)), list });
        live.page = None;
    });
    any
}

fn resolve(on: &On, id: &str) -> Result<(), String> {
    match on {
        On::Board { id: board, .. } => {
            let store = crate::audit::boards().ok_or("no board store")?;
            store.resolve(board, id, crate::board::comments::OPERATOR).map(|_| ()).map_err(|e| e.to_string())
        }
        On::Page(_) => crate::projects::pages::resolve(id, crate::board::comments::OPERATOR, true).map(|_| ()),
    }
}

/// A page comment or reply of the operator's reaches the agent with his
/// next message (a board's shows in its board_read).
fn told_agent(on: &On, text: &str) {
    if let On::Page(page) = on {
        crate::audit::ops::needs_s0ma(format!("· a comment from the operator on projects/{page}: {text} (project_read it; comment_reply to answer)"));
    }
}

/// A board made in a folder: opened in the store, its file placed there.
fn make_board(folder: &str, name: &str, goal: &str) -> Result<String, String> {
    let store = crate::audit::boards().ok_or("no board store")?;
    let folder_path = crate::projects::abs(folder).ok_or("no projects folder")?;
    crate::projects::place_next_in(Some(folder_path));
    let made = store.open_board(name, name, goal, goal, "", crate::board::GoalTestKind::Judgment, 60);
    crate::projects::place_next_in(None);
    match made {
        Ok(Ok(board)) => crate::projects::path_of(crate::projects::Kind::Board, &board.id).and_then(|p| crate::projects::rel(&p)).ok_or_else(|| "made, but its file is not there".to_string()),
        Ok(Err(refused)) => Err(refused.to_string()),
        Err(error) => Err(error.to_string()),
    }
}

/// An audit of a board: `/audit start`, attached to it, its file beside it.
fn make_audit(folder: &str, board: &str, args: &str) -> Result<String, String> {
    let folder_path = crate::projects::abs(folder).ok_or("no projects folder")?;
    // An attached terminal: the primary starts it (it runs the audits).
    if super::remote() {
        super::act_on(super::Action::Command { text: format!("/audit start {args} --for {board}") })?;
        let name = args.split_whitespace().next().unwrap_or_default();
        return Ok(format!("{folder}/{name}.audit").trim_start_matches('/').to_string());
    }
    crate::projects::place_next_in(Some(folder_path));
    let said = crate::audit::ops::handle_command(&format!("/audit start {args} --for {board}")).unwrap_or_else(|| Err("not an audit".into()));
    crate::projects::place_next_in(None);
    said?;
    let name = args.split_whitespace().next().unwrap_or_default();
    crate::projects::path_of(crate::projects::Kind::Audit, name).and_then(|p| crate::projects::rel(&p)).ok_or_else(|| "started, but its file is not there".to_string())
}

fn edit(tab: &mut Open, node: &Node) {
    let rel = match node.kind {
        NodeKind::Project | NodeKind::Folder => {
            let page = format!("{}/{}", node.rel, tree::PAGE);
            if let Some(path) = crate::projects::abs(&page)
                && !path.exists()
            {
                let _ = std::fs::write(&path, format!("# {}\n", node.name));
            }
            page
        }
        NodeKind::Page => node.rel.clone(),
        NodeKind::File => node.rel.clone(),
        _ => return,
    };
    let Some(path) = crate::projects::abs(&rel) else { return };
    let Ok(local) = std::fs::read_to_string(&path) else {
        return tab.note("not text", true);
    };
    // Read where the editor writes (the agent's /share), for its stamp; the
    // copy here while it has not got there yet.
    let slot: Opening = std::sync::Arc::new(Mutex::new(None));
    let into = slot.clone();
    let dir = crate::share::share_dir();
    let shared = format!("projects/{rel}");
    let started = std::thread::Builder::new().name("projects-open".into()).spawn(move || {
        let read = crate::share::files::read(&dir, &shared, crate::share::files::MAX_WHOLE_BYTES, true, &mut crate::share::run_on_machine);
        *crate::util::locked(&into) = Some(read);
    });
    match started {
        Ok(_) => with(|live| live.opening = Some((rel, local, slot))),
        Err(error) => tab.note(format!("could not start: {error}"), true),
    }
}

type Opening = std::sync::Arc<Mutex<Option<Result<crate::share::files::Read, String>>>>;

/// A page `e` asked for, read: into the editor.
fn opened(tab: &mut Open) {
    let Some((rel, local, read)) = with(|live| {
        let ready = live.opening.as_ref().and_then(|(_, _, slot)| crate::util::locked(slot).take())?;
        let (rel, local, _) = live.opening.take()?;
        Some((rel, local, ready))
    }) else {
        return;
    };
    let (text, stamp) = match read {
        Ok(read) => match crate::share::files::text_of(&read.bytes) {
            Some(text) => (text.to_string(), Some(read.stamp)),
            None => return tab.note("not text", true),
        },
        Err(_) => (local, None),
    };
    super::editor_view::open(&crate::share::share_dir(), &format!("projects/{rel}"), &text, stamp);
    with(|live| live.returning = true);
    go_to(tab, Target::View(View::Edit));
}

/// ⏎ or a double click.
fn enter(tab: &mut Open, row: &Row) -> Outcome {
    match row {
        Row::Node(node) => match node.kind {
            NodeKind::Project | NodeKind::Folder => {
                with(|live| {
                    let just_opened = live.clicked_open.take().is_some_and(|(rel, at)| rel == node.rel && at.elapsed().as_millis() < 800);
                    if just_opened {
                        return;
                    }
                    if !live.saved.open.remove(&node.rel) {
                        live.saved.open.insert(node.rel.clone());
                    }
                });
            }
            // A double click opens a board or an audit in full; ⏎ (and a
            // double click on a page) gives the page the keys, as the share
            // tab's strip.
            NodeKind::Board | NodeKind::Audit if double_clicked(&node.rel) => open_full(tab, node),
            NodeKind::Page | NodeKind::File | NodeKind::Board | NodeKind::Audit => with(|live| live.viewing = true),
        },
        Row::Trash(_) => with(|live| live.saved.trash_open = !live.saved.trash_open),
        Row::Trashed(_) => {}
    }
    Outcome::Stay
}

/// ⏎ right after a click on the same row: the double click's.
fn double_clicked(key: &str) -> bool {
    with(|live| live.clicked.take().is_some_and(|(clicked, at)| clicked == key && at.elapsed().as_millis() < 800))
}

/// A board in its stage's subtab, an audit in the audit tab.
fn open_full(tab: &mut Open, node: &Node) {
    match node.kind {
        NodeKind::Board => open_board(tab, node),
        NodeKind::Audit => open_audit(tab, node),
        _ => {}
    }
}

/// The stage's subtab a board is in.
fn stage_view(id: &str) -> Option<View> {
    let path = crate::projects::path_of(crate::projects::Kind::Board, id)?;
    let info = summary::board(&path)?;
    Some(match info.stage.as_str() {
        "review" => View::Review,
        "verification" => View::Verification,
        "working" => View::Board,
        _ => View::Deliverables,
    })
}

fn open_board(tab: &mut Open, node: &Node) {
    if node.archived || node.hidden {
        return tab.note("archived: A brings it back first", true);
    }
    let Some(info) = board_info(&node.rel) else {
        return tab.note("this file can't be read as a board", true);
    };
    let Some(view) = stage_view(&info.id) else { return };
    with(|live| {
        live.returning = true;
        save(live);
    });
    go_to(tab, Target::View(view.clone()));
    show_board(tab, &view, &info.id);
}

/// The board shown in its subtab.
pub(super) fn show_board(tab: &mut Open, view: &View, id: &str) {
    if *view == View::Board {
        if let Some(store) = crate::audit::boards()
            && let Ok(boards) = super::board_tab::shown_boards(&store)
            && let Some(index) = boards.iter().position(|b| b.id == id)
        {
            tab.board = index;
            tab.board_read = None;
        }
    } else {
        tab.projects.selected = Some(id.to_string());
    }
}

fn open_audit(tab: &mut Open, node: &Node) {
    if node.archived || node.hidden {
        return tab.note("archived: A brings it back first", true);
    }
    let Some(info) = crate::projects::abs(&node.rel).and_then(|p| summary::audit(&p)) else {
        return tab.note("this file can't be read as an audit", true);
    };
    with(|live| {
        live.returning = true;
        save(live);
    });
    go_to(tab, Target::View(View::Audit));
    tab.audit.audit = Some(info.name);
}

/// An audit opened from here: Esc at its top comes back here (rather than
/// to every audit).
pub(super) fn intercept(tab: &mut Open, key: &KeyEvent) -> Option<Outcome> {
    let back = tab.view == View::Audit
        && key.code == KeyCode::Esc
        && key.modifiers.is_empty()
        && tab.audit.audit.is_some()
        && super::audit_view::at_top(tab)
        && with(|live| std::mem::take(&mut live.returning));
    back.then(|| go_to(tab, Target::View(View::Projects)))
}

/// A board's subtab or the editor opened from here, closed with Esc: here
/// again, not the agent's page.
pub(super) fn closed(tab: &mut Open, key: &KeyEvent, outcome: Outcome) -> Outcome {
    if outcome != Outcome::Closed {
        return outcome;
    }
    let from = matches!(tab.view, View::Board | View::Review | View::Verification | View::Deliverables);
    if from && key.code == KeyCode::Esc && with(|live| std::mem::take(&mut live.returning)) {
        return go_to(tab, Target::View(View::Projects));
    }
    // The editor closed (Esc, Ctrl+Q on its last file): back to the tab it
    // was opened from, as it was, not the agent's page.
    if tab.view == View::Edit && tab.previous.as_ref().is_some_and(|previous| previous.view != View::Page) {
        with(|live| live.returning = false);
        return go_to(tab, Target::Back);
    }
    outcome
}

// ------------------------------------------------------------------ mouse

pub(super) fn pick_tree_row(index: usize) {
    with(|live| {
        live.making = false;
        live.armed = None;
        live.viewing = false;
        let all = rows(&live.saved);
        if let Some(row) = all.get(index) {
            live.saved.selected = Some(row.key());
            live.clicked = Some((row.key(), std::time::Instant::now()));
            // A folder opens on a click.
            if let Row::Node(node) = row
                && matches!(node.kind, NodeKind::Project | NodeKind::Folder)
                && live.saved.open.insert(node.rel.clone())
            {
                live.clicked_open = Some((node.rel.clone(), std::time::Instant::now()));
            }
        }
    });
}

pub(super) fn pick_move(index: usize) {
    with(|live| {
        if let Some((_, _, pick)) = live.moving.as_mut() {
            *pick = index;
        }
        if let Some(threads) = live.threads.as_mut() {
            threads.pick = index.min(threads.list.len().saturating_sub(1));
        }
    });
}

/// The wheel scrolls what it is over: the page, or the tree's selection.
/// `row`: in the body.
pub(super) fn wheel(up: bool, body_row: usize, column: usize) {
    let over_page = match with(|live| live.page_at) {
        Some(PageAt::Column(from)) => column >= from,
        Some(PageAt::Row(from)) => body_row >= from,
        None => false,
    } || with(|live| live.viewing);
    if over_page {
        scroll_page(if up { -(super::WHEEL_ROWS as isize) } else { super::WHEEL_ROWS as isize });
        return;
    }
    let (all, at) = with(|live| {
        let all = rows(&live.saved);
        let at = selected_index(&live.saved, &all);
        (all, at)
    });
    move_by(&all, at, if up { -1 } else { 1 });
}

/// A page being read for the editor: drawn again until it is.
pub(super) fn busy() -> bool {
    with(|live| live.opening.is_some())
}

/// Typing here: the tab's letters are not switches.
pub(super) fn typing_now() -> bool {
    with(|live| live.input.is_some() || live.moving.is_some() || live.threads.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(tab: &mut Open, code: KeyCode) {
        key(tab, &KeyEvent::new(code, KeyModifiers::NONE));
    }

    fn typed(tab: &mut Open, text: &str) {
        for c in text.chars() {
            press(tab, KeyCode::Char(c));
        }
        press(tab, KeyCode::Enter);
    }

    fn shown(tab: &mut Open) -> String {
        let (lines, _) = body(tab, 120, 30);
        lines.iter().map(|l| l.iter().map(|(_, t)| super::super::shell_tab::plain_text(t)).collect::<String>()).collect::<Vec<_>>().join("\n")
    }

    fn selected() -> Option<String> {
        with(|live| live.saved.selected.clone())
    }

    /// The tab's state is the process's: one test at a time.
    static ONE: Mutex<()> = Mutex::new(());

    /// Made, trashed, put back, archived from the keys; the place kept for
    /// a new tab.
    #[test]
    fn the_tree_makes_trashes_and_keeps_its_place() {
        let _one = crate::util::locked(&ONE);
        let base = crate::projects::log::tests::scratch();
        let root = base.join("projects");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(base.join("data")).unwrap();
        crate::projects::set_for_test(Some(root.clone()), &base.join("data"));
        with(|live| *live = Live { loaded: true, ..Live::default() });
        let mut tab = Open::new(View::Projects);
        press(&mut tab, KeyCode::Char('N'));
        typed(&mut tab, "calc");
        assert!(root.join("calc/index.md").exists());
        assert_eq!(selected().as_deref(), Some("calc"));
        // A page in it, with the folder's page beside the tree.
        press(&mut tab, KeyCode::Char('n'));
        press(&mut tab, KeyCode::Char('m'));
        typed(&mut tab, "notes");
        assert!(root.join("calc/notes.md").exists());
        assert_eq!(selected().as_deref(), Some("calc/notes.md"));
        let page = shown(&mut tab);
        assert!(page.contains("notes.md"), "{page}");
        // The trash, twice, and back.
        press(&mut tab, KeyCode::Delete);
        assert!(root.join("calc/notes.md").exists(), "once only asks");
        press(&mut tab, KeyCode::Delete);
        assert!(!root.join("calc/notes.md").exists());
        let page = shown(&mut tab);
        assert!(page.contains("trash"), "{page}");
        let all = with(|live| rows(&live.saved));
        let trash = all.iter().position(|r| matches!(r, Row::Trash(_))).unwrap();
        pick_tree_row(trash);
        press(&mut tab, KeyCode::Right);
        press(&mut tab, KeyCode::Down);
        press(&mut tab, KeyCode::Char('u'));
        assert!(root.join("calc/notes.md").exists(), "put back");
        assert_eq!(selected().as_deref(), Some("calc/notes.md"));
        // Archived: hidden, then shown with h.
        press(&mut tab, KeyCode::Char('A'));
        assert!(!with(|live| rows(&live.saved)).iter().any(|r| r.key() == "calc/notes.md"));
        press(&mut tab, KeyCode::Char('h'));
        assert!(with(|live| rows(&live.saved)).iter().any(|r| r.key() == "calc/notes.md"));
        // A new tab (a switch away and back) is where this one was.
        drop(tab);
        let mut tab = Open::new(View::Projects);
        shown(&mut tab);
        assert_eq!(selected().as_deref(), Some("calc/notes.md"));
        assert!(with(|live| live.saved.open.contains("calc")));
    }

    /// `c` comments on the selected item; `x` lists its open threads, ⏎
    /// resolves one, `r` replies.
    #[test]
    fn comments_from_the_tree() {
        let _one = crate::util::locked(&ONE);
        let base = crate::projects::log::tests::scratch();
        let root = base.join("projects");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(base.join("data")).unwrap();
        crate::projects::set_for_test(Some(root.clone()), &base.join("data"));
        with(|live| *live = Live { loaded: true, ..Live::default() });
        let mut tab = Open::new(View::Projects);
        press(&mut tab, KeyCode::Char('N'));
        typed(&mut tab, "calc");
        press(&mut tab, KeyCode::Char('c'));
        typed(&mut tab, "a section on the lexer?");
        let threads = crate::projects::pages::threads("calc/index.md");
        assert_eq!(threads.len(), 1);
        assert!(shown(&mut tab).contains("a section on the lexer?"), "the page shows it");
        press(&mut tab, KeyCode::Char('x'));
        assert!(with(|live| live.threads.as_ref().is_some_and(|t| t.list.len() == 1)));
        press(&mut tab, KeyCode::Char('r'));
        typed(&mut tab, "and the parser");
        assert_eq!(crate::projects::pages::threads("calc/index.md")[0].replies.len(), 1);
        press(&mut tab, KeyCode::Enter);
        assert!(!crate::projects::pages::threads("calc/index.md")[0].root.open(), "resolved");
        assert!(with(|live| live.threads.is_none()), "none open: the list closes");
    }

    /// ⏎ on a page gives it the keys, as the share tab's strip: ↑↓ scroll
    /// it, not the tree; Esc gives them back.
    #[test]
    fn enter_views_the_page() {
        let _one = crate::util::locked(&ONE);
        let base = crate::projects::log::tests::scratch();
        let root = base.join("projects");
        std::fs::create_dir_all(root.join("calc")).unwrap();
        std::fs::create_dir_all(base.join("data")).unwrap();
        crate::projects::set_for_test(Some(root.clone()), &base.join("data"));
        let long: String = (1..200).map(|i| format!("line {i}\n\n")).collect();
        std::fs::write(root.join("calc/long.md"), long).unwrap();
        std::fs::write(root.join("calc/short.md"), "short\n").unwrap();
        with(|live| *live = Live { loaded: true, ..Live::default() });
        let mut tab = Open::new(View::Projects);
        press(&mut tab, KeyCode::Right);
        press(&mut tab, KeyCode::Down);
        assert_eq!(selected().as_deref(), Some("calc/long.md"));
        shown(&mut tab);
        press(&mut tab, KeyCode::Enter);
        assert!(with(|live| live.viewing));
        press(&mut tab, KeyCode::Down);
        press(&mut tab, KeyCode::Down);
        assert_eq!(selected().as_deref(), Some("calc/long.md"), "the tree stays");
        assert_eq!(with(|live| live.saved.page_scroll), 2, "the page scrolls");
        press(&mut tab, KeyCode::Esc);
        assert!(!with(|live| live.viewing));
        press(&mut tab, KeyCode::Down);
        assert_eq!(selected().as_deref(), Some("calc/short.md"), "the tree has the keys again");
        // The wheel: over the page it scrolls the page, over the tree it moves.
        shown(&mut tab);
        press(&mut tab, KeyCode::Up);
        shown(&mut tab);
        let page_column = match with(|live| live.page_at) {
            Some(PageAt::Column(column)) => column,
            other => panic!("{other:?}"),
        };
        let before = with(|live| live.saved.page_scroll);
        wheel(false, 5, page_column + 5);
        assert_eq!(with(|live| live.saved.page_scroll), before + super::super::WHEEL_ROWS);
        assert_eq!(selected().as_deref(), Some("calc/long.md"));
        wheel(false, 5, 3);
        assert_eq!(selected().as_deref(), Some("calc/short.md"));
    }
}
