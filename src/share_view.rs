//! The share tab (Alt+S), a submodule of `tabs.rs`: `/share` on s0ma's
//! machine, the folder the operator and s0ma work on together (`shared_files.rs`).
//!
//! The list is the folder as a tree, the folder itself its first row, a
//! teal `[!]` on each file that changed since the operator last looked at it
//! (s0ma's work, mostly). Folders fold: ← folds one (or goes up to its
//! folder), → unfolds it, ⏎ does either, and Ctrl+O unfolds everything, or
//! folds every folder when all are open. The strip under it shows the
//! selected file; ⏎ views it, the file taking the whole tab (Esc for the
//! list again). The wheel over the list moves the selection, over the file
//! scrolls it. Markdown is shown
//! rendered, as s0ma's page renders it; `m` switches to the raw text and
//! back.
//!
//! - `e` opens the file in the operator's editor and writes it back in place. s0ma is
//!   told with his next message, as a diff. If the file changed while he
//!   edited, his version is kept beside it, `name (operator copy).md`, and nothing is
//!   written over.
//! - `n` makes a new file: a path, then the editor.
//! - `f` makes a new folder (and any on the way to it) at the path typed,
//!   from the selected folder.
//! - `a` adds files from his machine (`share::send`) to the selected folder.
//! - `s` saves a copy to his machine (`~/s0ma-share`). Saved again, it
//!   replaces the copy it saved before, if that copy is untouched.
//! - `x`, twice, deletes the file, or the folder and everything in it (the
//!   question says how many files that is).

use super::{Action, Footer, Open, Outcome, act, clip, text_width};
use crate::mistake_view::{Line, Style};
use crate::share::human_size;
use crate::shared_files::{self, Change, Entry, Expect, WriteError};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use sha2::{Digest, Sha256};

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The folder is listed again this often while the tab is up.
const RELOAD: Duration = Duration::from_secs(2);
/// The strip shows this much of a file.
const PREVIEW_BYTES: u64 = 256 * 1024;
/// A preview is read once the selection has rested this long.
const PREVIEW_AFTER: Duration = Duration::from_millis(150);
const PREVIEWS_KEPT: usize = 8;

type Slot<T> = Arc<Mutex<Option<T>>>;

fn spawn<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Result<Slot<T>, String> {
    let slot: Slot<T> = Arc::new(Mutex::new(None));
    let answer = slot.clone();
    std::thread::Builder::new()
        .name("share-tab".into())
        .spawn(move || {
            let result = work();
            if let Ok(mut answer) = answer.lock() {
                *answer = Some(result);
            }
            // Shown at once, not at the next second's redraw.
            super::poke();
        })
        .map_err(|error| error.to_string())?;
    Ok(slot)
}

fn take<T>(slot: &Slot<T>) -> Option<T> {
    slot.lock().ok().and_then(|mut answer| answer.take())
}

// ------------------------------------------------------------------ what the operator has seen

/// What the operator has seen, across openings of the tab: each file's stamp when he
/// last looked at it (keyed by folder and path), which folders were taken as
/// seen when first listed, and the copies saved to his machine.
#[derive(Default)]
struct Looked {
    seen: HashMap<String, String>,
    baselined: HashSet<String>,
    /// Copies saved to the operator's machine: from which file in the folder, to
    /// where, and their SHA-256 then.
    saved: HashMap<String, (PathBuf, String)>,
    /// Added by the operator from his machine: seen as soon as they are listed.
    own: HashSet<String>,
}

static LOOKED: Mutex<Option<Looked>> = Mutex::new(None);

fn looked<T>(f: impl FnOnce(&mut Looked) -> T) -> T {
    let mut guard = LOOKED.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    f(guard.get_or_insert_with(Looked::default))
}

fn seen_key(dir: &str, path: &str) -> String {
    format!("{dir}/{path}")
}

pub(super) fn mark_seen(dir: &str, path: &str, stamp: &str) {
    looked(|looked| looked.seen.insert(seen_key(dir, path), stamp.to_string()));
}

/// Changed since the operator last looked: a file whose stamp is not the one he saw.
fn changed(dir: &str, entry: &Entry) -> bool {
    !entry.dir && looked(|looked| looked.seen.get(&seen_key(dir, &entry.path)) != Some(&entry.stamp))
}

// ------------------------------------------------------------------ state

#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum Purpose {
    #[default]
    New,
    Add,
    /// A new folder.
    Folder,
}

#[derive(Default)]
struct Input {
    purpose: Purpose,
    text: Vec<char>,
    cursor: usize,
    /// What Tab found when it could not settle on one: shown under the line.
    choices: Vec<String>,
}

enum Shown {
    Text { lines: Vec<String>, cut: bool },
    Binary,
}

struct Preview {
    path: String,
    stamp: String,
    size: u64,
    shown: Result<Shown, String>,
}

/// Where a read of a file on s0ma's machine lands.
type ReadSlot = Slot<Result<shared_files::Read, String>>;

struct Work {
    what: &'static str,
    started: Instant,
    answer: Slot<Result<String, String>>,
}

pub(super) struct State {
    /// The folder: `share::share_dir()`.
    dir: String,
    entries: Option<Result<Vec<Entry>, String>>,
    listing: Option<Slot<Result<Vec<Entry>, String>>>,
    listed: Option<Instant>,
    /// By path, so a listing that moves rows keeps it.
    selected: Option<String>,
    top: usize,
    in_strip: bool,
    scroll: usize,
    /// The row of the tab's body where the file's strip starts (its rule),
    /// as last drawn: the wheel below it scrolls the file.
    strip_at: usize,
    /// How many rows the list had, as last drawn (the wheel scrolls it).
    list_rows: usize,
    preview: Option<Preview>,
    /// The last few files' previews, by path and stamp: stepping back to
    /// one is instant.
    previews: std::collections::VecDeque<Preview>,
    reading: Option<(String, String, ReadSlot)>,
    input: Option<Input>,
    work: Option<Work>,
    /// `x` once: this path is deleted by `x` again.
    /// `x` once: what `x` again deletes; a file's stamp; a folder's files.
    armed: Option<(String, Option<String>, Vec<String>)>,
    /// `e`: the file being read for the editor.
    opening: Option<(String, Slot<Result<shared_files::Read, String>>)>,
    /// Folders folded by hand, by path (`""` is the shared folder itself).
    folded: HashSet<String>,
    /// Markdown files shown rendered (`m` switches to the raw text).
    markdown: bool,
    /// The selected file rendered: its path, stamp and width, and the rows.
    rendered: Option<(String, String, usize, Vec<String>)>,
    /// Shown inside another tab (a board's files, in the projects tab):
    /// browsed and read there, not added to or deleted from. The folder's
    /// own row is named the first, and the line over the tree says the
    /// second where the folder's path would be.
    embedded: Option<(String, String)>,
}

impl Default for State {
    fn default() -> Self {
        State::in_dir(crate::share::share_dir())
    }
}

impl State {
    #[cfg(test)]
    pub(super) fn selected_path(&self) -> Option<&str> {
        self.selected.as_deref()
    }

    fn in_dir(dir: String) -> State {
        State {
            dir,
            entries: None,
            listing: None,
            listed: None,
            selected: None,
            top: 0,
            in_strip: false,
            strip_at: usize::MAX,
            list_rows: 0,
            scroll: 0,
            preview: None,
            previews: std::collections::VecDeque::new(),
            reading: None,
            input: None,
            work: None,
            armed: None,
            opening: None,
            folded: HashSet::new(),
            markdown: true,
            rendered: None,
            embedded: None,
        }
    }

    /// A folder inside the shared one shown in another tab (`body` there,
    /// with `embedded_key`): its own row named `label`, and `header` said
    /// over the tree for its path.
    pub(super) fn embedded(dir: String, label: &str, header: &str) -> State {
        State { embedded: Some((label.to_string(), header.to_string())), ..State::in_dir(dir) }
    }

    /// Viewing a file (⏎), not the list.
    pub(super) fn viewing(&self) -> bool {
        self.in_strip
    }

    /// The rows the list shows: the folder itself, then what is in it, less
    /// what is inside a folded folder.
    fn visible(&self) -> Vec<Entry> {
        let root = Entry { path: String::new(), dir: true, bytes: 0, modified: 0.0, stamp: String::new() };
        if self.folded.contains("") {
            return vec![root];
        }
        let mut shown = vec![root];
        shown.extend(self.entries().iter().filter(|entry| !self.hidden(&entry.path)).cloned());
        shown
    }

    /// Inside a folded folder (the shared folder itself included).
    fn hidden(&self, path: &str) -> bool {
        if self.folded.contains("") && !path.is_empty() {
            return true;
        }
        let mut at = path;
        while let Some((parent, _)) = at.rsplit_once('/') {
            if self.folded.contains(parent) {
                return true;
            }
            at = parent;
        }
        false
    }

    fn entries(&self) -> &[Entry] {
        match &self.entries {
            Some(Ok(entries)) => entries,
            _ => &[],
        }
    }

    /// Among the rows shown: the selected one, or the nearest folder holding
    /// it when it is folded away, or the first.
    fn selected_index(&self) -> Option<usize> {
        let shown = self.visible();
        // At first, the first thing in the folder (the folder itself when
        // it is empty or folded).
        let Some(mut path) = self.selected.clone() else { return Some(usize::from(shown.len() > 1)) };
        loop {
            if let Some(index) = shown.iter().position(|entry| entry.path == path) {
                return Some(index);
            }
            match path.rsplit_once('/') {
                Some((parent, _)) => path = parent.to_string(),
                None if !path.is_empty() => path = String::new(),
                None => return Some(0),
            }
        }
    }

    fn selected_entry(&self) -> Option<Entry> {
        let index = self.selected_index().unwrap_or(0);
        self.visible().into_iter().nth(index)
    }

    /// The folder new and added files go in: the selected folder, or the
    /// selected file's.
    fn folder(&self) -> String {
        match self.selected_entry() {
            Some(entry) if entry.dir => entry.path.clone(),
            Some(entry) => entry.path.rsplit_once('/').map(|(parent, _)| parent.to_string()).unwrap_or_default(),
            None => String::new(),
        }
    }

    fn reload(&mut self) {
        self.listed = None;
    }
}

// ------------------------------------------------------------------ background work

/// Whatever finished: the listing, a preview, the work.
fn settle(tab: &mut Open) {
    let state = &mut tab.share;
    if let Some(done) = state.listing.as_ref().and_then(take) {
        state.listing = None;
        state.listed = Some(Instant::now());
        if let Ok(entries) = &done {
            // Opened the first time: what is there now counts as seen.
            let dir = state.dir.clone();
            looked(|looked| {
                let first = looked.baselined.insert(dir.clone());
                for entry in entries.iter().filter(|entry| !entry.dir) {
                    let key = seen_key(&dir, &entry.path);
                    if first || looked.own.remove(&key) {
                        looked.seen.insert(key, entry.stamp.clone());
                    }
                }
            });
        }
        state.entries = Some(done);
    }
    if let Some((path, stamp, slot)) = &state.reading
        && let Some(done) = take(slot)
    {
        let (path, stamp) = (path.clone(), stamp.clone());
        state.reading = None;
        let size = done.as_ref().map_or(0, |read| read.size);
        let shown = done.map(|read| match shared_files::text_of(&read.bytes) {
            Some(text) => Shown::Text {
                lines: text.lines().map(|line| line.replace('\t', "    ")).collect(),
                cut: read.size > read.bytes.len() as u64,
            },
            // Cut mid-character at the preview's end: still text.
            None if read.size > read.bytes.len() as u64 && !read.bytes.contains(&0) => {
                let text = String::from_utf8_lossy(&read.bytes);
                Shown::Text { lines: text.lines().map(|line| line.replace('\t', "    ")).collect(), cut: true }
            }
            None => Shown::Binary,
        });
        if shown.is_ok() {
            mark_seen(&state.dir, &path, &stamp);
        }
        if let Some(old) = state.preview.replace(Preview { path, stamp, size, shown }) {
            remember(state, old);
        }
    }
    if let Some((path, slot)) = &state.opening
        && let Some(read) = take(slot)
    {
        let path = path.clone();
        tab.share.opening = None;
        opened(tab, path, read);
    }
    let state = &mut tab.share;
    if let Some(done) = state.work.as_ref().and_then(|work| take(&work.answer)) {
        state.work = None;
        state.reload();
        match done {
            Ok(message) => tab.note(message, false),
            Err(error) => tab.note(error, true),
        }
    }
}

fn load(state: &mut State) {
    let due = state.listed.is_none_or(|at| at.elapsed() >= RELOAD);
    if state.listing.is_some() || state.work.is_some() || !due {
        return;
    }
    let dir = state.dir.clone();
    match spawn(move || {
        // An attached terminal: what the sync is doing, before the listing
        // (the files it wrote are seen before they could show as news).
        crate::tabs::follow_share_sync();
        shared_files::list(&dir, &mut crate::share::run_on_machine)
    }) {
        Ok(slot) => state.listing = Some(slot),
        Err(error) => state.entries = Some(Err(error)),
    }
}

/// The selected file, read for the strip when it is not already (or has
/// changed since).
fn preview(state: &mut State) {
    let Some(entry) = state.selected_entry().filter(|entry| !entry.dir) else {
        return;
    };
    let fresh = |path: &str, stamp: &str| path == entry.path && stamp == entry.stamp;
    if state.preview.as_ref().is_some_and(|preview| fresh(&preview.path, &preview.stamp))
        || state.reading.as_ref().is_some_and(|(path, stamp, _)| fresh(path, stamp))
    {
        return;
    }
    // Seen lately, as it is now: from the cache.
    if let Some(at) = state.previews.iter().position(|preview| fresh(&preview.path, &preview.stamp)) {
        let cached = state.previews.remove(at).expect("found");
        if let Some(old) = state.preview.replace(cached) {
            remember(state, old);
        }
        return;
    }
    let dir = state.dir.clone();
    let path = entry.path.clone();
    let slot: Slot<Result<shared_files::Read, String>> = Arc::new(Mutex::new(None));
    let answer = slot.clone();
    // Read once the selection rests: holding ↓ through a folder reads only
    // where it stops. A read the tab no longer waits for (it moved on, and
    // dropped its end) is not made.
    let spawned = std::thread::Builder::new().name("share-preview".into()).spawn(move || {
        std::thread::sleep(PREVIEW_AFTER);
        if Arc::strong_count(&answer) < 2 {
            return;
        }
        let read = shared_files::read(&dir, &path, PREVIEW_BYTES, false, &mut crate::share::run_on_machine);
        if let Ok(mut answer) = answer.lock() {
            *answer = Some(read);
        }
        super::poke();
    });
    if spawned.is_ok() {
        state.reading = Some((entry.path, entry.stamp, slot));
    }
}

/// A preview no longer up, kept for a while.
fn remember(state: &mut State, preview: Preview) {
    state.previews.retain(|kept| kept.path != preview.path);
    state.previews.push_front(preview);
    state.previews.truncate(PREVIEWS_KEPT);
}

fn start(tab: &mut Open, what: &'static str, work: impl FnOnce() -> Result<String, String> + Send + 'static) {
    if tab.share.work.is_some() {
        tab.note("busy", true);
        return;
    }
    match spawn(work) {
        Ok(answer) => tab.share.work = Some(Work { what, started: Instant::now(), answer }),
        Err(error) => tab.note(format!("could not start: {error}"), true),
    }
}

/// s0ma hears of it with the operator's next message (from the s0ma that runs the
/// turns, on an attached terminal).
pub(super) fn tell(change: Change) {
    let _ = act(Action::Shared { change });
}

// ------------------------------------------------------------------ drawing

/// Shown rendered unless `m` says raw.
fn is_markdown(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.ends_with(".md") || lower.ends_with(".markdown") || lower.ends_with(".mdx")
}

fn when(modified: f64) -> String {
    let Some(at) = chrono::DateTime::from_timestamp(modified as i64, 0) else { return String::new() };
    let at = at.with_timezone(&chrono::Local);
    if at.date_naive() == chrono::Local::now().date_naive() {
        at.format("%H:%M").to_string()
    } else {
        at.format("%b %d %H:%M").to_string()
    }
}

/// A row of the tree: under its folder, a folder marked folded (▸) or open
/// (▾). The shared folder itself (`dir`) is the first row.
fn row(entry: &Entry, dir: &str, folded: bool, selected: bool, active: bool, changed: bool, width: usize) -> Line {
    let depth = if entry.path.is_empty() { 0 } else { entry.path.matches('/').count() + 1 };
    let name = if entry.path.is_empty() { dir } else { entry.path.rsplit('/').next().unwrap_or(&entry.path) };
    let right = if entry.dir { String::new() } else { format!("{:>9}  {:>12}", human_size(entry.bytes), when(entry.modified)) };
    let fold = match (entry.dir, folded) {
        (true, true) => "▸ ",
        (true, false) => "▾ ",
        _ => "  ",
    };
    let label = format!("{}{fold}{name}{}", "  ".repeat(depth), if entry.dir { "/" } else { "" });
    // The cursor (2) and the change mark (4) before it, the size and time after.
    let room = width.saturating_sub(6 + text_width(&right) + 2).max(1);
    let label = clip(&label, room);
    let pad = " ".repeat(room.saturating_sub(text_width(&label)));
    let (lead, lead_style) = match (selected, active) {
        (true, true) => ("❯ ", Style::Cursor),
        (true, false) => ("❯ ", Style::Dim),
        _ => ("  ", Style::Plain),
    };
    vec![
        (lead_style, lead.into()),
        (Style::Warn, if changed { "[!] ".into() } else { "    ".into() }),
        (if entry.dir { Style::Status } else if selected { Style::Bold } else { Style::Plain }, format!("{label}{pad}")),
        (Style::Dim, format!("  {right}")),
    ]
}

pub(super) fn body(tab: &mut Open, width: usize, rows: usize) -> (Vec<Line>, Footer) {
    settle(tab);
    let state = &mut tab.share;
    load(state);
    // Settled on what is listed, once the listing is current: a file just
    // made keeps the selection until it is listed.
    if state.listing.is_none()
        && state.work.is_none()
        && let Some(index) = state.selected_index()
    {
        state.selected = state.visible().get(index).map(|entry| entry.path.clone());
    }
    preview(state);
    let dir = state.dir.clone();
    let mut out: Vec<Line> = Vec::new();

    // What the folder is, or the line being typed.
    let all = state.entries().to_vec();
    let files = all.iter().filter(|entry| !entry.dir).count();
    let fresh = all.iter().filter(|entry| changed(&dir, entry)).count();
    let entries = state.visible();
    out.push(match (&state.input, &state.entries) {
        (Some(input), _) => {
            let label = match input.purpose {
                Purpose::New => format!("  new file › {dir}/"),
                Purpose::Folder => format!("  new folder › {dir}/"),
                Purpose::Add => "  add from your machine › ".to_string(),
            };
            // Scrolled to keep the cursor in sight: a long path loses its
            // beginning to an ellipsis.
            let room = width.saturating_sub(text_width(&label) + 2).max(8);
            let shown_from = |start: usize| text_width(&input.text[start..input.cursor].iter().collect::<String>()) + usize::from(start > 0);
            let mut start = 0;
            while start < input.cursor && shown_from(start) > room {
                start += 1;
            }
            let mut before: String = input.text[start..input.cursor].iter().collect();
            if start > 0 {
                before.insert(0, '…');
            }
            let at: String = input.text.get(input.cursor).map_or(" ".to_string(), |c| c.to_string());
            let after: String = input.text.get(input.cursor + 1..).map(|rest| rest.iter().collect()).unwrap_or_default();
            vec![(Style::Cursor, label), (Style::Plain, before), (Style::Caret, at), (Style::Plain, after)]
        }
        (None, Some(Err(error))) => vec![(Style::Problem, format!("  {dir}: {error}"))],
        (None, None) => vec![(Style::Work, format!("  reading {dir}…"))],
        (None, Some(Ok(_))) => {
            let named = state.embedded.as_ref().map_or(dir.as_str(), |(_, header)| header.as_str());
            let mut line = vec![(Style::Dim, format!("  {named} · {files} file{}", if files == 1 { "" } else { "s" }))];
            match crate::share_sync::label().filter(|_| state.embedded.is_none()) {
                Some((mirror, None)) => line.push((Style::Status, format!("  {mirror}"))),
                Some((mirror, Some(problem))) => line.push((Style::Problem, format!("  {mirror} paused: {problem}"))),
                None => {}
            }
            if fresh > 0 {
                line.push((Style::Warn, format!("  [!] {fresh} changed")));
            }
            line
        }
    });

    // What Tab found, when it is more than one.
    if let Some(input) = state.input.as_ref().filter(|input| !input.choices.is_empty()) {
        out.push(choices_line(&input.choices, width));
    }
    let selected = state.selected_index();
    let left = rows.saturating_sub(out.len());
    // Viewing a file (⏎): it has the whole tab.
    let list_rows = if state.in_strip { 0 } else { (left * 9 / 20).max(3).min(left.saturating_sub(2)) };
    let strip_rows = left.saturating_sub(list_rows + 1);

    // The list.
    if let Some(index) = selected {
        if index < state.top {
            state.top = index;
        } else if index >= state.top + list_rows {
            state.top = index + 1 - list_rows;
        }
    }
    state.top = state.top.min(entries.len().saturating_sub(list_rows));
    if !matches!(state.entries, Some(Ok(_))) {
        // Nothing listed yet: no tree.
    } else {
        let root = state.embedded.as_ref().map_or_else(|| dir.clone(), |(label, _)| label.clone());
        for (index, entry) in entries.iter().enumerate().skip(state.top).take(list_rows) {
            let folded = state.folded.contains(&entry.path);
            let line = row(entry, &root, folded, Some(index) == selected, !state.in_strip, changed(&dir, entry), width);
            out.push(super::gui_view::pick_row(line, super::gui_view::Click::Pick(super::gui_view::Pick::Share(entry.path.clone()))));
        }
        if all.is_empty() {
            let empty = if state.embedded.is_some() { "      nothing here" } else { "      nothing here yet" };
            out.push(vec![(Style::Dim, empty.into())]);
        }
    }
    out.resize(1 + list_rows, Vec::new());
    state.strip_at = out.len();
    state.list_rows = list_rows;

    // The strip: the selected file.
    let entry = selected.map(|index| entries[index].clone());
    let mut lines: Vec<Line> = Vec::new();
    let mut about = String::new();
    match &entry {
        None => {}
        Some(entry) if entry.dir => {
            let inside = if entry.path.is_empty() { all.len() } else { all.iter().filter(|other| other.path.starts_with(&format!("{}/", entry.path))).count() };
            let fold = if state.folded.contains(&entry.path) { "⏎ unfolds" } else { "⏎ folds" };
            let more = if state.embedded.is_some() { "" } else { " · n new here · a add here" };
            lines.push(vec![(Style::Dim, format!("folder · {inside} inside · {fold}{more}"))]);
        }
        Some(entry) => {
            // The file's own preview; while it is read, the last one shown,
            // dimmed, rather than a flash of "reading…".
            let own = state.preview.as_ref().is_some_and(|preview| preview.path == entry.path);
            if !own {
                about = " reading… ".into();
            }
            let text_style = if own { Style::Plain } else { Style::Dim };
            match state.preview.as_ref() {
                None => lines.push(vec![(Style::Work, "reading…".into())]),
                Some(Preview { shown: Err(error), .. }) => lines.push(vec![(Style::Problem, error.clone())]),
                Some(Preview { shown: Ok(Shown::Binary), size, .. }) => {
                    lines.push(vec![(Style::Dim, format!("not text · {} · s saves a copy", human_size(*size)))]);
                }
                Some(Preview { shown: Ok(Shown::Text { lines: text, cut }), size, path, stamp })
                    if own && state.markdown && is_markdown(path) =>
                {
                    // Rendered as s0ma's page renders markdown, once per
                    // file, stamp and width.
                    let fits = width.saturating_sub(3).max(10);
                    let fresh = state.rendered.as_ref().is_some_and(|(p, s, w, _)| p == path && s == stamp && *w == fits);
                    if !fresh {
                        let mut source = text.join("\n");
                        if *cut {
                            source.push_str(&format!("\n\n*… the first {} of {} · e opens all of it*", human_size(PREVIEW_BYTES), human_size(*size)));
                        }
                        state.rendered = Some((path.clone(), stamp.clone(), fits, crate::markdown::render_lines(&source, fits)));
                    }
                    let rows = state.rendered.as_ref().map(|(_, _, _, rows)| rows.clone()).unwrap_or_default();
                    state.scroll = state.scroll.min(rows.len().saturating_sub(strip_rows));
                    for (index, row) in rows.into_iter().enumerate() {
                        let visible = index >= state.scroll && index < state.scroll + strip_rows;
                        lines.push(if visible { vec![(Style::Raw, row)] } else { Vec::new() });
                    }
                }
                Some(Preview { shown: Ok(Shown::Text { lines: text, cut }), size, .. }) => {
                    // Only the rows on screen are cut to the width; the rest
                    // are counted, for the scroll.
                    let total = text.len() + usize::from(*cut) + usize::from(text.is_empty());
                    state.scroll = state.scroll.min(total.saturating_sub(strip_rows));
                    for index in 0..total {
                        if index < state.scroll || index >= state.scroll + strip_rows {
                            lines.push(Vec::new());
                            continue;
                        }
                        lines.push(match text.get(index) {
                            Some(line) => vec![(text_style, clip(line, width.saturating_sub(2).max(1)))],
                            None if *cut => vec![(Style::Dim, format!("… the first {} of {} · e opens all of it", human_size(PREVIEW_BYTES), human_size(*size)))],
                            None => vec![(Style::Dim, "empty".into())],
                        });
                    }
                }
            }
        }
    }
    state.scroll = state.scroll.min(lines.len().saturating_sub(strip_rows));
    let below = lines.len().saturating_sub(state.scroll + strip_rows);
    if about.is_empty() && (state.scroll > 0 || below > 0) {
        about = format!(" {}–{} of {} ", state.scroll + 1, (state.scroll + strip_rows).min(lines.len()), lines.len());
    }
    let title = entry.as_ref().filter(|entry| !entry.dir).map(|entry| format!(" {} ", entry.path)).unwrap_or_default();
    let rule = width.saturating_sub(2 + text_width(&title) + text_width(&about) + 2);
    out.push(vec![
        (Style::Dim, "  ".into()),
        (if state.in_strip { Style::Cursor } else { Style::Bold }, title),
        (Style::Dim, "─".repeat(rule)),
        (Style::Cursor, about),
    ]);
    for line in lines.into_iter().skip(state.scroll).take(strip_rows) {
        let mut shown = vec![(Style::Plain, "  ".to_string())];
        shown.extend(line);
        out.push(shown);
    }
    out.resize(rows, Vec::new());

    // The footer.
    let mut status: Line = vec![(Style::Status, "share".into())];
    let hints: Vec<(&'static str, &'static str)> = if let Some((path, _)) = &state.opening {
        status.push((Style::Work, format!("  opening {path}…")));
        vec![("esc", "back")]
    } else if let Some(work) = &state.work {
        status.push((Style::Work, format!("  {}… {}s", work.what, work.started.elapsed().as_secs())));
        vec![("esc", "back")]
    } else if state.input.is_some() {
        vec![("tab", "complete"), ("⏎", "ok"), ("esc", "cancel")]
    } else if let Some((path, _, inside)) = &state.armed {
        let what = match inside.len() {
            0 => format!("  delete {path}?"),
            1 => format!("  delete {path}/ and the file in it?"),
            count => format!("  delete {path}/ and the {count} files in it?"),
        };
        status.push((Style::Danger, what));
        vec![("x", "delete"), ("esc", "keep")]
    } else {
        let fold_all = if state.folded.is_empty() { ("^o", "fold all") } else { ("^o", "unfold all") };
        let markdown = entry.as_ref().is_some_and(|entry| !entry.dir && is_markdown(&entry.path));
        let mode = if state.markdown { ("m", "raw") } else { ("m", "rendered") };
        if state.in_strip {
            let mut hints = vec![("↑↓", "scroll"), ("e", "edit")];
            if markdown {
                hints.push(mode);
            }
            hints.push(("esc", "list"));
            hints
        } else if entry.as_ref().is_some_and(|entry| !entry.dir) {
            let save = if crate::share_sync::label().is_some() { ("s", "where") } else { ("s", "save") };
            let mut hints = vec![("⏎", "view"), ("e", "edit")];
            if markdown {
                hints.push(mode);
            }
            if state.embedded.is_some() {
                hints.extend([save, fold_all]);
            } else {
                hints.extend([("n", "new"), ("f", "folder"), ("a", "add"), save, ("x", "delete"), fold_all, ("esc", "back")]);
            }
            hints
        } else if state.embedded.is_some() {
            let fold = if entry.as_ref().is_some_and(|entry| state.folded.contains(&entry.path)) { "unfold" } else { "fold" };
            vec![("⏎", fold), fold_all]
        } else {
            let fold = if entry.as_ref().is_some_and(|entry| state.folded.contains(&entry.path)) { "unfold" } else { "fold" };
            let mut hints = vec![("⏎", fold), ("n", "new"), ("f", "folder"), ("a", "add")];
            if entry.as_ref().is_some_and(|entry| !entry.path.is_empty()) {
                hints.push(("x", "delete"));
            }
            hints.extend([fold_all, ("esc", "back")]);
            hints
        }
    };
    (out, Footer::new(status, hints))
}

// ------------------------------------------------------------------ keys

/// A turn of the mouse wheel at `row` of the tab's body: over the file (or
/// while viewing it) it scrolls the file, over the list it moves the
/// selection a row.
/// A click on an entry: it is selected (and shown under the list), as the
/// arrows would select it.
pub(super) fn pick(tab: &mut Open, path: String) {
    let state = &mut tab.share;
    if state.input.is_some() || state.work.is_some() || state.opening.is_some() || state.in_strip {
        return;
    }
    state.armed = None;
    if state.selected.as_deref() != Some(path.as_str()) && state.visible().iter().any(|entry| entry.path == path) {
        state.selected = Some(path);
        state.scroll = 0;
    }
}

pub(super) fn wheel(tab: &mut Open, up: bool, row: usize, page: usize) {
    let state = &mut tab.share;
    if state.input.is_some() || state.work.is_some() || state.opening.is_some() {
        return;
    }
    // A delete asked about is for the row it was asked on: the wheel moves
    // away from it, as any other key does.
    state.armed = None;
    if state.in_strip || row >= state.strip_at {
        state.scroll = if up { state.scroll.saturating_sub(super::WHEEL_ROWS) } else { state.scroll.saturating_add(super::WHEEL_ROWS) };
        return;
    }
    // The list scrolls first; the selection moves with it only when it would
    // leave the rows shown, and on its own once the list is at its end.
    let shown = state.visible();
    let rows = state.list_rows.max(1);
    let last_top = shown.len().saturating_sub(rows);
    let scrolls = if up { state.top > 0 } else { state.top < last_top };
    if scrolls {
        state.top = if up { state.top.saturating_sub(super::WHEEL_ROWS) } else { (state.top + super::WHEEL_ROWS).min(last_top) };
        if let Some(at) = state.selected_index() {
            let kept = at.clamp(state.top, (state.top + rows).saturating_sub(1).min(shown.len().saturating_sub(1)));
            if kept != at {
                state.selected = shown.get(kept).map(|entry| entry.path.clone());
                state.scroll = 0;
            }
        }
        return;
    }
    let code = if up { KeyCode::Up } else { KeyCode::Down };
    key(tab, &KeyEvent::new(code, KeyModifiers::NONE), page);
}

/// A key for a folder shown inside another tab (`State::embedded`, swapped
/// into `tab.share` for the call): what the share tab does to browse and
/// read, nothing that adds or deletes. `None`: not one of these keys, the
/// other tab's own. Esc is taken only while a file is viewed (back to the
/// list).
pub(super) fn embedded_key(tab: &mut Open, key: &KeyEvent, page: usize) -> Option<Outcome> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let state = &tab.share;
    let busy = state.work.is_some() || state.opening.is_some();
    let ours = match key.code {
        KeyCode::Up | KeyCode::Down | KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home | KeyCode::End | KeyCode::Enter => true,
        KeyCode::Esc => state.in_strip || busy,
        KeyCode::Char('o' | 'O') => ctrl,
        KeyCode::Char('m' | 'e' | 's') => !ctrl,
        _ => false,
    };
    if !ours {
        return None;
    }
    if key.code == KeyCode::Esc && busy {
        tab.share.opening = None;
        return Some(Outcome::Stay);
    }
    Some(self::key(tab, key, page))
}

pub(super) fn key(tab: &mut Open, key: &KeyEvent, page: usize) -> Outcome {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('c' | 'C') if ctrl => return Outcome::Interrupt,
        KeyCode::Char('d' | 'D') if ctrl => return Outcome::Eof,
        _ => {}
    }
    if tab.share.work.is_some() || tab.share.opening.is_some() {
        if key.code == KeyCode::Esc {
            // Leaving while a file is read for the editor: it is not opened.
            tab.share.opening = None;
            return Outcome::Closed;
        }
        tab.note("working", false);
        return Outcome::Stay;
    }
    if tab.share.input.is_some() {
        typing(tab, key);
        return Outcome::Stay;
    }
    if let Some((path, stamp, inside)) = tab.share.armed.take() {
        match key.code {
            KeyCode::Char('x') if inside.is_empty() => remove(tab, path, stamp),
            KeyCode::Char('x') => remove_folder(tab, path, inside),
            KeyCode::Esc => tab.note("kept", false),
            _ => return self::key(tab, key, page),
        }
        return Outcome::Stay;
    }
    let state = &mut tab.share;
    if state.in_strip {
        match key.code {
            KeyCode::Up => state.scroll = state.scroll.saturating_sub(1),
            KeyCode::Down => state.scroll = state.scroll.saturating_add(1),
            KeyCode::PageUp => state.scroll = state.scroll.saturating_sub(page),
            KeyCode::PageDown | KeyCode::Char(' ') => state.scroll = state.scroll.saturating_add(page),
            KeyCode::Home => state.scroll = 0,
            KeyCode::End => state.scroll = usize::MAX,
            KeyCode::Esc | KeyCode::Enter | KeyCode::Left | KeyCode::Char('q') => state.in_strip = false,
            KeyCode::Char('m') => {
                state.markdown = !state.markdown;
                state.scroll = 0;
            }
            KeyCode::Char('e') => edit(tab),
            _ => {}
        }
        return Outcome::Stay;
    }
    let shown = state.visible();
    let count = shown.len();
    let at = state.selected_index().unwrap_or(0);
    let go = |state: &mut State, to: usize| {
        if let Some(entry) = shown.get(to.min(count.saturating_sub(1))) {
            state.selected = Some(entry.path.clone());
            state.scroll = 0;
        }
    };
    let here = shown.get(at).cloned();
    // Ctrl+O: everything unfolded, or, when nothing is folded, every
    // folder folded (the shared folder itself stays open).
    if ctrl && matches!(key.code, KeyCode::Char('o' | 'O')) {
        if state.folded.is_empty() {
            state.folded = state.entries().iter().filter(|entry| entry.dir).map(|entry| entry.path.clone()).collect();
        } else {
            state.folded.clear();
        }
        return Outcome::Stay;
    }
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => return Outcome::Closed,
        KeyCode::Up => go(state, at.saturating_sub(1)),
        KeyCode::Down => go(state, at + 1),
        KeyCode::PageUp => go(state, at.saturating_sub(page / 2)),
        KeyCode::PageDown => go(state, at + page / 2),
        KeyCode::Home => go(state, 0),
        KeyCode::End => go(state, usize::MAX),
        KeyCode::Enter | KeyCode::Right if here.as_ref().is_some_and(|entry| !entry.dir) => {
            state.in_strip = true;
        }
        // A folder: ⏎ folds or unfolds it; → unfolds it, or steps in; ←
        // folds it, or goes up to the folder holding it.
        KeyCode::Enter => {
            if let Some(entry) = here
                && !state.folded.remove(&entry.path)
            {
                state.folded.insert(entry.path);
            }
        }
        KeyCode::Right => {
            if let Some(entry) = here
                && !state.folded.remove(&entry.path)
            {
                go(state, at + 1);
            }
        }
        KeyCode::Left => match here {
            Some(entry) if entry.dir && !state.folded.contains(&entry.path) => {
                state.folded.insert(entry.path);
            }
            Some(entry) if !entry.path.is_empty() => {
                state.selected = Some(entry.path.rsplit_once('/').map(|(parent, _)| parent.to_string()).unwrap_or_default());
                state.scroll = 0;
            }
            _ => {}
        },
        KeyCode::Char('m') => {
            state.markdown = !state.markdown;
            state.scroll = 0;
        }
        KeyCode::Char('e') => edit(tab),
        KeyCode::Char('n') => {
            let folder = state.folder();
            let text: Vec<char> = if folder.is_empty() { Vec::new() } else { format!("{folder}/").chars().collect() };
            state.input = Some(Input { purpose: Purpose::New, cursor: text.len(), text, ..Input::default() });
        }
        KeyCode::Char('f') => {
            let folder = state.folder();
            let text: Vec<char> = if folder.is_empty() { Vec::new() } else { format!("{folder}/").chars().collect() };
            state.input = Some(Input { purpose: Purpose::Folder, cursor: text.len(), text, ..Input::default() });
        }
        KeyCode::Char('a') => state.input = Some(Input { purpose: Purpose::Add, ..Input::default() }),
        KeyCode::Char('s') => save(tab),
        KeyCode::Char('x') => {
            if let Some(entry) = state.selected_entry().filter(|entry| !entry.path.is_empty()) {
                // A folder: the files in it, at any depth, said and deleted with it.
                let inside: Vec<String> = if entry.dir {
                    let under = format!("{}/", entry.path);
                    state.entries().iter().filter(|each| !each.dir && each.path.starts_with(&under)).map(|each| each.path.clone()).collect()
                } else {
                    Vec::new()
                };
                state.armed = Some((entry.path.clone(), (!entry.dir).then(|| entry.stamp.clone()), inside));
            }
        }
        KeyCode::Char('r') => state.reload(),
        _ => {}
    }
    Outcome::Stay
}

fn typing(tab: &mut Open, key: &KeyEvent) {
    if key.code == KeyCode::Tab {
        return complete(tab);
    }
    let Some(input) = tab.share.input.as_mut() else { return };
    input.choices.clear();
    match key.code {
        KeyCode::Esc => tab.share.input = None,
        KeyCode::Enter => {
            let input = tab.share.input.take().expect("typed");
            let text: String = input.text.iter().collect();
            match input.purpose {
                Purpose::New => new_file(tab, text.trim()),
                Purpose::Add => add(tab, text.trim()),
                Purpose::Folder => new_folder(tab, text.trim()),
            }
        }
        KeyCode::Backspace if input.cursor > 0 => {
            input.cursor -= 1;
            input.text.remove(input.cursor);
        }
        KeyCode::Delete if input.cursor < input.text.len() => {
            input.text.remove(input.cursor);
        }
        KeyCode::Left => input.cursor = input.cursor.saturating_sub(1),
        KeyCode::Right => input.cursor = (input.cursor + 1).min(input.text.len()),
        KeyCode::Home => input.cursor = 0,
        KeyCode::End => input.cursor = input.text.len(),
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            input.text.insert(input.cursor, c);
            input.cursor += 1;
        }
        _ => {}
    }
}

/// What Tab can put in place of the path being typed: each candidate's
/// whole text, the name to list it by, and whether a space follows it.
struct Found {
    /// Where the path being completed starts, in characters.
    start: usize,
    candidates: Vec<(String, String, bool)>,
}

/// The new file's path, from the folder as listed: the names one level
/// under what is typed so far.
fn in_folder(entries: &[Entry], typed: &str) -> Found {
    let (folder, prefix) = match typed.rfind('/') {
        Some(slash) => (&typed[..=slash], &typed[slash + 1..]),
        None => ("", typed),
    };
    let candidates = entries
        .iter()
        .filter_map(|entry| {
            let rest = entry.path.strip_prefix(folder)?;
            if rest.contains('/') || !rest.starts_with(prefix) || (rest.starts_with('.') && !prefix.starts_with('.')) {
                return None;
            }
            let slash = if entry.dir { "/" } else { "" };
            Some((format!("{}{slash}", entry.path), format!("{rest}{slash}"), false))
        })
        .collect();
    Found { start: 0, candidates }
}

/// Paths on the operator's machine, as `/share` at the prompt completes them: the
/// path under the cursor, `~` and `\ ` and all.
fn on_machine(before: &str) -> Found {
    let start = crate::completion::token_start(before);
    let candidates = crate::completion::paths(&before[start..])
        .into_iter()
        .map(|candidate| {
            let name = crate::completion::unescape_path(&candidate.value);
            let name = name.trim_end_matches('/');
            let mut label = name.rsplit('/').next().unwrap_or(name).to_string();
            if candidate.value.ends_with('/') {
                label.push('/');
            }
            (candidate.value, label, candidate.append_space)
        })
        .collect();
    Found { start: before[..start].chars().count(), candidates }
}

/// Tab in the line being typed: the path completed as far as it is one
/// thing; when it is several, they are listed under the line.
fn complete(tab: &mut Open) {
    let entries = tab.share.entries().to_vec();
    let Some(input) = tab.share.input.as_mut() else { return };
    let before: String = input.text[..input.cursor].iter().collect();
    let found = match input.purpose {
        Purpose::New | Purpose::Folder => in_folder(&entries, &before),
        Purpose::Add => on_machine(&before),
    };
    let typed: String = input.text[found.start..input.cursor].iter().collect();
    let (with, choices) = match found.candidates.as_slice() {
        [] => {
            input.choices.clear();
            return tab.note("no match", true);
        }
        [(value, _, space)] => (format!("{value}{}", if *space { " " } else { "" }), Vec::new()),
        many => {
            let values: Vec<&str> = many.iter().map(|(value, _, _)| value.as_str()).collect();
            let common = crate::completion::longest_common_prefix(&values);
            let common = if common.chars().count() > typed.chars().count() { common } else { typed.clone() };
            (common, many.iter().map(|(_, label, _)| label.clone()).collect())
        }
    };
    let with: Vec<char> = with.chars().collect();
    input.cursor = found.start + with.len();
    input.text.splice(found.start..found.start + typed.chars().count(), with);
    input.choices = choices;
}

/// The choices, one row: as many as fit, then how many more.
fn choices_line(choices: &[String], width: usize) -> Line {
    let mut line: Line = vec![(Style::Dim, "    ".into())];
    let mut used = 4;
    for (index, choice) in choices.iter().enumerate() {
        let left = choices.len() - index;
        let more = format!("+{left} more");
        let need = text_width(choice) + 2;
        if used + need + (if left > 1 { more.len() } else { 0 }) > width {
            line.push((Style::Dim, more));
            break;
        }
        let style = if choice.ends_with('/') { Style::Status } else { Style::Plain };
        line.push((style, choice.clone()));
        line.push((Style::Plain, "  ".into()));
        used += need;
    }
    line
}

/// A paste: into the line being typed, if one is.
pub(super) fn paste(tab: &mut Open, text: &str) {
    if let Some(input) = tab.share.input.as_mut() {
        for c in text.chars().filter(|c| !c.is_control()) {
            input.text.insert(input.cursor, c);
            input.cursor += 1;
        }
    }
}

// ------------------------------------------------------------------ doing

/// `text` in the operator's editor, under the file's own name (so the editor knows
/// its kind): what he saved, or `None` when the editor failed.
pub(super) fn in_editor(tab: &mut Open, relative: &str, text: &str) -> Result<Option<String>, String> {
    let folder = std::env::temp_dir().join(format!("s0ma-share-edit-{}", std::process::id()));
    std::fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    let name = relative.rsplit('/').next().unwrap_or(relative);
    let path = folder.join(name);
    std::fs::write(&path, text).map_err(|error| error.to_string())?;
    crate::terminal::set_mouse(false);
    let saved = crate::mistake_view::edit_path(&path);
    crate::terminal::set_mouse(true);
    crate::page::mark_stale();
    // The editor had the screen: all of it is drawn again.
    tab.frame.clear();
    let _ = crossterm::queue!(std::io::stdout(), crossterm::terminal::Clear(crossterm::terminal::ClearType::All));
    let edited = match saved {
        Ok(true) => std::fs::read_to_string(&path).map(Some).map_err(|error| error.to_string()),
        Ok(false) => Ok(None),
        Err(error) => Err(format!("the editor could not be run: {error}")),
    };
    let _ = std::fs::remove_file(&path);
    edited
}

/// `S0MA_EDITOR=external`: files are edited in `$VISUAL`/`$EDITOR`, as before
/// s0ma had its own editor.
pub(super) fn external_editor() -> bool {
    std::env::var("S0MA_EDITOR").is_ok_and(|value| value.trim().eq_ignore_ascii_case("external"))
}

/// An editor ends the file with a newline it may not have had: that alone is
/// no change.
pub(super) fn same(before: &str, after: &str) -> bool {
    after == before || (!before.ends_with('\n') && after.strip_suffix('\n') == Some(before))
}

/// `e`: the file is read in the background; the editor opens when it is in
/// (`opened`).
fn edit(tab: &mut Open) {
    let Some(entry) = tab.share.selected_entry().filter(|entry| !entry.dir) else {
        tab.note("pick a file", true);
        return;
    };
    let dir = tab.share.dir.clone();
    let path = entry.path.clone();
    match spawn(move || shared_files::read(&dir, &path, shared_files::MAX_WHOLE_BYTES, true, &mut crate::share::run_on_machine)) {
        Ok(slot) => tab.share.opening = Some((entry.path, slot)),
        Err(error) => tab.note(format!("could not start: {error}"), true),
    }
}

/// The file `e` asked for, read: into the editor, and written back.
fn opened(tab: &mut Open, path: String, read: Result<shared_files::Read, String>) {
    let dir = tab.share.dir.clone();
    let read = match read {
        Ok(read) => read,
        Err(error) => return tab.note(error, true),
    };
    let Some(before) = shared_files::text_of(&read.bytes).map(str::to_string) else {
        return tab.note("not text · s saves a copy", true);
    };
    // s0ma's own editor (Alt+E), unless the operator asked for theirs.
    if !external_editor() {
        mark_seen(&dir, &path, &read.stamp);
        super::editor_view::open(&dir, &path, &before, Some(read.stamp));
        super::go_to(tab, super::Target::View(super::View::Edit));
        return;
    }
    let entry = Entry { path, dir: false, bytes: read.size, modified: 0.0, stamp: read.stamp.clone() };
    let after = match in_editor(tab, &entry.path, &before) {
        Ok(Some(after)) => after,
        Ok(None) => return tab.note("editor failed · nothing saved", true),
        Err(error) => return tab.note(error, true),
    };
    mark_seen(&dir, &entry.path, &read.stamp);
    if same(&before, &after) {
        return tab.note("unchanged", false);
    }
    let (path, stamp) = (entry.path, read.stamp);
    start(tab, "saving", move || save_edit(&dir, &path, &stamp, &before, &after, &mut crate::share::run_on_machine));
}

fn save_edit(dir: &str, path: &str, stamp: &str, before: &str, after: &str, run: shared_files::Run) -> Result<String, String> {
    let shown = shared_files::shown(dir, path);
    match shared_files::write(dir, path, after.as_bytes(), &Expect::Stamp(stamp.to_string()), run) {
        Ok(new) => {
            mark_seen(dir, path, &new);
            tell(Change::Edited {
                path: path.to_string(),
                before_stamp: stamp.to_string(),
                before: shared_files::carried(before),
                after: shared_files::carried(after),
                stamp: new,
            });
            Ok(format!("saved {shown} · s0ma sees the change"))
        }
        Err(WriteError::Changed) => {
            let (kept, _) = beside(dir, path, after, run)?;
            tell(Change::KeptBeside { path: path.to_string(), kept_as: kept.clone() });
            Err(format!("{shown} changed while you edited · yours is {}", shared_files::shown(dir, &kept)))
        }
        Err(WriteError::Failed(error)) => Err(error),
    }
}

/// The operator's version beside a file that changed under him: `plan (operator copy).md`.
pub(super) fn beside(dir: &str, path: &str, text: &str, run: shared_files::Run) -> Result<(String, String), String> {
    let (parent, name) = match path.rsplit_once('/') {
        Some((parent, name)) => (format!("{parent}/"), name),
        None => (String::new(), path),
    };
    let (stem, extension) = crate::share::split_name(name);
    for n in 1..100 {
        let mark = if n == 1 { "operator copy".to_string() } else { format!("operator copy {n}") };
        let candidate = format!("{parent}{stem} ({mark}){extension}");
        match shared_files::write(dir, &candidate, text.as_bytes(), &Expect::New, run) {
            Ok(stamp) => {
                mark_seen(dir, &candidate, &stamp);
                return Ok((candidate, stamp));
            }
            Err(WriteError::Changed) => continue,
            Err(WriteError::Failed(error)) => return Err(format!("your version could not be kept: {error}")),
        }
    }
    Err("your version could not be kept: too many copies beside it".into())
}

fn new_file(tab: &mut Open, typed: &str) {
    let dir = tab.share.dir.clone();
    let path = match shared_files::relative(&dir, typed) {
        Ok(path) => path,
        Err(error) => return tab.note(error, true),
    };
    if tab.share.entries().iter().any(|entry| entry.path == path) {
        return tab.note("already there · e edits it", true);
    }
    if !external_editor() {
        tab.share.selected = Some(path.clone());
        super::editor_view::open(&dir, &path, "", None);
        super::go_to(tab, super::Target::View(super::View::Edit));
        return;
    }
    let text = match in_editor(tab, &path, "") {
        Ok(Some(text)) if !text.trim().is_empty() => text,
        Ok(_) => return tab.note("empty · nothing made", false),
        Err(error) => return tab.note(error, true),
    };
    tab.share.selected = Some(path.clone());
    start(tab, "making", move || make(&dir, &path, &text, &mut crate::share::run_on_machine));
}

/// `f`: a new folder at the path typed (folders on the way made too).
fn new_folder(tab: &mut Open, typed: &str) {
    let dir = tab.share.dir.clone();
    let path = match shared_files::relative(&dir, typed.trim_end_matches('/')) {
        Ok(path) => path,
        Err(error) => return tab.note(error, true),
    };
    // A name starting with a dot is left out of the list, and of the sync.
    if path.split('/').any(|part| part.starts_with('.')) {
        return tab.note("a name starting with . would be hidden · give another", true);
    }
    if tab.share.entries().iter().any(|entry| entry.path == path) {
        return tab.note("already there", true);
    }
    tab.share.selected = Some(path.clone());
    start(tab, "making", move || make_folder(&dir, &path, &mut crate::share::run_on_machine));
}

fn make_folder(dir: &str, path: &str, run: shared_files::Run) -> Result<String, String> {
    match shared_files::make_dir(dir, path, run) {
        Ok(()) => Ok(format!("made {}/", shared_files::shown(dir, path))),
        Err(WriteError::Changed) => Err(format!("{path} is already there")),
        Err(WriteError::Failed(error)) => Err(error),
    }
}

fn make(dir: &str, path: &str, text: &str, run: shared_files::Run) -> Result<String, String> {
    match shared_files::write(dir, path, text.as_bytes(), &Expect::New, run) {
        Ok(stamp) => {
            mark_seen(dir, path, &stamp);
            tell(Change::Created { path: path.to_string(), after: shared_files::carried(text), stamp });
            Ok(format!("made {} · s0ma sees it", shared_files::shown(dir, path)))
        }
        Err(WriteError::Changed) => {
            let (kept, stamp) = beside(dir, path, text, run)?;
            tell(Change::Created { path: kept.clone(), after: shared_files::carried(text), stamp });
            Err(format!("{path} appeared meanwhile · yours is {kept}"))
        }
        Err(WriteError::Failed(error)) => Err(error),
    }
}

fn add(tab: &mut Open, typed: &str) {
    let paths = match crate::share::parse_paths(typed) {
        Ok(paths) => paths,
        Err(error) => return tab.note(error.replace("usage: /share", "give"), true),
    };
    let folder = tab.share.folder();
    let into = if folder.is_empty() { tab.share.dir.clone() } else { format!("{}/{folder}", tab.share.dir) };
    start(tab, "adding", move || {
        let mut added = Vec::new();
        let mut failed = Vec::new();
        for path in paths {
            let name = home_short(&path.display().to_string());
            match crate::share::send(&path, &into, &mut crate::share::run_on_machine) {
                Ok(sent) => {
                    looked(|looked| looked.own.insert(sent.destination.clone()));
                    tell(Change::Added { line: sent.describe() });
                    added.push(sent.destination);
                }
                Err(error) => failed.push(format!("{name}: {error}")),
            }
        }
        match (added.len(), failed.is_empty()) {
            (_, false) => Err(failed.join(" · ")),
            (1, true) => Ok(format!("added {}", added[0])),
            (n, true) => Ok(format!("added {n} files")),
        }
    });
}

fn sha256_of(path: &std::path::Path) -> Option<String> {
    std::fs::read(path).ok().map(|bytes| format!("{:x}", Sha256::digest(bytes)))
}

fn home_short(path: &str) -> String {
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() && path.starts_with(&format!("{home}/")) => format!("~{}", &path[home.len()..]),
        _ => path.to_string(),
    }
}

fn save(tab: &mut Open) {
    let Some(entry) = tab.share.selected_entry().filter(|entry| !entry.dir) else {
        return tab.note("pick a file", true);
    };
    // Mirrored already: it is there.
    if let Some(there) = crate::share_sync::local_path(&tab.share.dir, &entry.path) {
        return tab.tell(format!("on your machine: {}", crate::share_sync::home_short(&there)));
    }
    let source = shared_files::shown(&tab.share.dir, &entry.path);
    start(tab, "saving a copy", move || save_copy(&source, &crate::share::receive_dir(), &mut crate::share::run_on_machine));
}

/// A copy on the operator's machine. The one this tab saved there before, untouched
/// since, is replaced; anything else there is kept (`name (2).md`).
fn save_copy(source: &str, into: &std::path::Path, run: shared_files::Run) -> Result<String, String> {
    let ours = looked(|looked| looked.saved.get(source).cloned());
    // The copy of this same file saved before, untouched since: replaced,
    // once the new one is safely here.
    if let Some((target, sha)) = ours.filter(|(target, sha)| sha256_of(target).as_ref() == Some(sha))
        && let (Some(folder), Some(name)) = (target.parent(), target.file_name().and_then(|name| name.to_str()))
    {
        let _ = sha;
        let staging = folder.join(format!(".s0ma-saving-{}", std::process::id()));
        let received = crate::share::receive(source, Some(name), &staging, run).and_then(|received| {
            std::fs::rename(&received.destination, &target)
                .map(|()| received)
                .map_err(|error| format!("cannot put it in place: {error}"))
        });
        let _ = std::fs::remove_dir_all(&staging);
        let received = received?;
        looked(|looked| looked.saved.insert(source.to_string(), (target.clone(), received.sha256)));
        return Ok(format!("saved → {} (over your last copy)", home_short(&target.display().to_string())));
    }
    let received = crate::share::receive(source, None, into, run)?;
    let destination = PathBuf::from(&received.destination);
    looked(|looked| looked.saved.insert(source.to_string(), (destination, received.sha256.clone())));
    Ok(format!("saved → {}", home_short(&received.destination)))
}

/// Deleted only as it was when `x` was first pressed: a file that changed
/// since (s0ma's work, say) is kept.
fn remove(tab: &mut Open, path: String, stamp: Option<String>) {
    let dir = tab.share.dir.clone();
    start(tab, "deleting", move || match shared_files::remove(&dir, &path, stamp.as_deref(), &mut crate::share::run_on_machine) {
        Ok(()) => {
            tell(Change::Removed { path: path.clone() });
            Ok(format!("deleted {}", shared_files::shown(&dir, &path)))
        }
        Err(WriteError::Changed) => Err(format!("{path} changed since x · kept")),
        Err(WriteError::Failed(error)) => Err(error),
    });
}

/// `x` twice on a folder with files in it: the folder and all of them; s0ma
/// is told of each.
fn remove_folder(tab: &mut Open, path: String, inside: Vec<String>) {
    let dir = tab.share.dir.clone();
    start(tab, "deleting", move || match shared_files::remove_tree(&dir, &path, &mut crate::share::run_on_machine) {
        Ok(()) => {
            for file in &inside {
                tell(Change::Removed { path: file.clone() });
            }
            Ok(format!("deleted {}/ and {} file(s)", shared_files::shown(&dir, &path), inside.len()))
        }
        Err(WriteError::Changed) => Err(format!("{path} changed since x · kept")),
        Err(WriteError::Failed(error)) => Err(error),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_files::tests::{Folder, local};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn text(lines: &[Line]) -> String {
        lines.iter().map(|line| line.iter().map(|(_, text)| text.as_str()).collect::<String>()).collect::<Vec<_>>().join("\n")
    }

    /// The tab on `folder`, drawn until what it reads in the background is in.
    fn settled(tab: &mut Open, width: usize, rows: usize) -> (String, Footer) {
        for _ in 0..200 {
            let (body, footer) = body(tab, width, rows);
            let busy = tab.share.listing.is_some() || tab.share.reading.is_some() || tab.share.work.is_some();
            if !busy && tab.share.entries.is_some() {
                let (body, footer) = super::body(tab, width, rows);
                return (text(&body), footer);
            }
            let _ = (body, footer);
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("the tab never settled");
    }

    fn notes() -> std::sync::MutexGuard<'static, ()> {
        crate::shared_files::tests::NOTES.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn open(folder: &Folder) -> Open {
        let mut tab = Open::new(super::super::View::Share);
        tab.share = State::in_dir(folder.0.clone());
        tab
    }

    #[test]
    fn a_click_selects_a_double_click_opens_and_buttons_and_menus_do_what_keys_do() {
        use crossterm::event::{Event, MouseButton, MouseEvent, MouseEventKind};
        let folder = Folder::new("clicks");
        folder.put("a.txt", "alpha\n");
        folder.put("b.txt", "bee\n");
        let mut tab = open(&folder);
        settled(&mut tab, 80, 30);
        let display = crate::terminal::ActiveDisplay::new();
        let mut open = Some(tab);
        let draw = |open: &mut Option<Open>| {
            let tab = open.as_mut().unwrap();
            let (frame, _) = super::super::compose(tab, &display, (80, 30), std::time::Instant::now());
            tab.frame = frame;
        };
        let at = |open: &Option<Open>, label: &str| {
            let hit = open.as_ref().unwrap().hits.iter().find(|hit| hit.label.contains(label)).unwrap_or_else(|| panic!("no {label:?}"));
            (hit.row as u16, hit.from as u16 + 1)
        };
        let mouse = |open: &mut Option<Open>, kind: MouseEventKind, (row, column): (u16, u16)| {
            let event = Event::Mouse(MouseEvent { kind, column, row, modifiers: KeyModifiers::NONE });
            super::super::pane_event(open, &event, &display).unwrap()
        };
        let press = |open: &mut Option<Open>, place| {
            let outcome = mouse(open, MouseEventKind::Down(MouseButton::Left), place);
            mouse(open, MouseEventKind::Up(MouseButton::Left), place);
            outcome
        };
        draw(&mut open);
        // No key is named anywhere in the footer: its keys are buttons.
        let footer = text(&open.as_ref().unwrap().frame[29..]);
        assert!(footer.contains(" Back ") && !footer.contains("esc"), "{footer}");
        // A click on a row selects it.
        let b = at(&open, "b.txt");
        press(&mut open, b);
        assert_eq!(open.as_ref().unwrap().share.selected.as_deref(), Some("b.txt"));
        assert!(open.as_ref().unwrap().selection.is_none(), "a click selects no text");
        // Clicked again soon after: opened, as Enter opens it.
        press(&mut open, b);
        assert!(open.as_ref().unwrap().share.in_strip, "the file has the tab");
        draw(&mut open);
        // Its List button: back to the list.
        let list = at(&open, " List ");
        press(&mut open, list);
        assert!(!open.as_ref().unwrap().share.in_strip);
        draw(&mut open);
        // A press alone does nothing: the click is done on release.
        let a = at(&open, "a.txt");
        mouse(&mut open, MouseEventKind::Down(MouseButton::Left), a);
        assert_eq!(open.as_ref().unwrap().share.selected.as_deref(), Some("b.txt"));
        mouse(&mut open, MouseEventKind::Up(MouseButton::Left), a);
        assert_eq!(open.as_ref().unwrap().share.selected.as_deref(), Some("a.txt"));
        // A right-click on a row: selected, and a menu of what the footer offers.
        let b = at(&open, "b.txt");
        mouse(&mut open, MouseEventKind::Down(MouseButton::Right), b);
        mouse(&mut open, MouseEventKind::Up(MouseButton::Right), b);
        assert_eq!(open.as_ref().unwrap().share.selected.as_deref(), Some("b.txt"));
        mouse(&mut open, MouseEventKind::Down(MouseButton::Left), (0, 0));
        assert!(open.as_ref().unwrap().menu.is_none(), "a click elsewhere closes it");
        mouse(&mut open, MouseEventKind::Up(MouseButton::Left), (0, 0));
        mouse(&mut open, MouseEventKind::Down(MouseButton::Right), a);
        let menu = open.as_ref().unwrap().menu.clone().expect("a menu");
        assert_eq!(open.as_ref().unwrap().share.selected.as_deref(), Some("a.txt"));
        let names: Vec<&str> = menu.items.iter().map(|(name, _)| name.as_str()).collect();
        assert!(names.contains(&"View") && names.contains(&"Back"), "{names:?}");
        // Esc closes it, and nothing else happens.
        assert_eq!(super::super::key_event(open.as_mut().unwrap(), &key(KeyCode::Esc), &display).unwrap(), Outcome::Stay);
        assert!(open.as_ref().unwrap().menu.is_none());
        // Chosen with a click, on release: View opens the file.
        mouse(&mut open, MouseEventKind::Down(MouseButton::Right), a);
        draw(&mut open);
        let (row, column) = at(&open, "View");
        let _ = (row, column);
        let view = open.as_ref().unwrap().frame.iter().position(|line| text(std::slice::from_ref(line)).contains("│ View")).unwrap() as u16;
        let left = text(&open.as_ref().unwrap().frame[view as usize..=view as usize]).find("│ View").unwrap() as u16 + 3;
        mouse(&mut open, MouseEventKind::Down(MouseButton::Left), (view, left));
        assert!(!open.as_ref().unwrap().share.in_strip, "not on the press");
        mouse(&mut open, MouseEventKind::Up(MouseButton::Left), (view, left));
        assert!(open.as_ref().unwrap().share.in_strip && open.as_ref().unwrap().menu.is_none(), "on the release");
        super::super::key_event(open.as_mut().unwrap(), &key(KeyCode::Esc), &display).unwrap();
        // Chosen with the keys: Back leaves the tab.
        mouse(&mut open, MouseEventKind::Down(MouseButton::Right), a);
        let back = names.iter().position(|name| *name == "Back").unwrap();
        for _ in 0..back {
            super::super::key_event(open.as_mut().unwrap(), &key(KeyCode::Down), &display).unwrap();
        }
        assert_eq!(super::super::key_event(open.as_mut().unwrap(), &key(KeyCode::Enter), &display).unwrap(), Outcome::Closed);
        // A drag still selects text, and copies it.
        draw(&mut open);
        let (row, column) = at(&open, "a.txt");
        mouse(&mut open, MouseEventKind::Down(MouseButton::Left), (row, column));
        mouse(&mut open, MouseEventKind::Drag(MouseButton::Left), (row, column + 4));
        draw(&mut open);
        assert!(open.as_ref().unwrap().selection.as_ref().is_some_and(|selection| selection.dragging), "selecting text");
    }

    #[test]
    fn viewing_a_file_takes_the_tab_and_the_wheel_scrolls_it() {
        let folder = Folder::new("wheel");
        let long: String = (1..=80).map(|n| format!("line {n}\n")).collect();
        folder.put("a.txt", &long);
        folder.put("b.txt", "bee\n");
        let mut tab = open(&folder);
        let (shown, _) = settled(&mut tab, 80, 30);
        assert!(shown.contains("b.txt"), "the list first: {shown}");
        // Over the list the wheel moves the selection; over the file it scrolls it.
        let strip = tab.share.strip_at;
        assert!(strip > 2 && strip < 30, "{strip}");
        super::wheel(&mut tab, false, strip + 3, 10);
        assert_eq!(tab.share.scroll, super::super::WHEEL_ROWS, "scrolled the file");
        assert_eq!(tab.share.selected.as_deref(), Some("a.txt"));
        super::wheel(&mut tab, false, 1, 10);
        assert_eq!(tab.share.selected.as_deref(), Some("b.txt"), "moved down the list");
        super::wheel(&mut tab, true, 1, 10);
        assert_eq!(tab.share.selected.as_deref(), Some("a.txt"));
        // ⏎: the file has the whole tab, no list; the wheel anywhere scrolls it.
        let _ = super::key(&mut tab, &key(KeyCode::Enter), 10);
        let (shown, _) = settled(&mut tab, 80, 30);
        assert!(!shown.contains("b.txt") && shown.contains("line 1") && shown.contains("line 27"), "{shown}");
        assert_eq!(tab.share.strip_at, 1, "under the folder's line");
        super::wheel(&mut tab, false, 0, 10);
        let (shown, _) = settled(&mut tab, 80, 30);
        assert!(!shown.lines().any(|line| line.trim() == "line 1") && shown.contains("line 4"), "{shown}");
        // Esc: the list again.
        let _ = super::key(&mut tab, &key(KeyCode::Esc), 10);
        let (shown, _) = settled(&mut tab, 80, 30);
        assert!(shown.contains("b.txt"), "{shown}");
    }

    #[test]
    fn the_folder_is_listed_with_the_selected_file_under_it_and_changes_marked() {
        let folder = Folder::new("view");
        folder.put("notes.md", "# Notes\nfirst\tline\n");
        folder.put("src/main.rs", "fn main() {}\n");
        let mut tab = open(&folder);
        let (shown, footer) = settled(&mut tab, 80, 20);
        assert!(shown.contains(&format!("{} · 2 files", folder.0)), "{shown}");
        assert!(shown.contains("notes.md") && shown.contains("src/") && shown.contains("  main.rs"), "{shown}");
        // The first row is selected and shown below, tabs as spaces.
        assert!(shown.contains("❯") && shown.contains("first    line"), "{shown}");
        assert!(footer.hints.iter().any(|(key, label)| *key == "e" && *label == "edit"));
        assert!(!shown.contains("[!]"), "seen on first opening: {shown}");
        // s0ma changes a file: marked until the operator looks at it.
        std::thread::sleep(Duration::from_millis(15));
        folder.put("src/main.rs", "fn main() { println!(\"hi\"); }\n");
        tab.share.reload();
        let (shown, _) = settled(&mut tab, 80, 20);
        assert!(shown.contains("[!] 1 changed") && shown.lines().any(|line| line.contains("[!] ") && line.contains("main.rs")), "{shown}");
        // Down to src/, then main.rs: looked at, so no longer marked.
        let _ = super::key(&mut tab, &key(KeyCode::Down), 10);
        let _ = super::key(&mut tab, &key(KeyCode::Down), 10);
        let (shown, _) = settled(&mut tab, 80, 20);
        assert!(shown.contains("println!") && !shown.contains("[!]"), "{shown}");
        // Enter puts the cursor in the strip; Esc back; Esc again leaves.
        assert_eq!(super::key(&mut tab, &key(KeyCode::Enter), 10), Outcome::Stay);
        assert!(tab.share.in_strip);
        let _ = super::key(&mut tab, &key(KeyCode::Esc), 10);
        assert!(!tab.share.in_strip);
        assert_eq!(super::key(&mut tab, &key(KeyCode::Esc), 10), Outcome::Closed);
    }

    #[test]
    fn folders_fold_the_shared_folder_too_and_ctrl_o_does_them_all() {
        let folder = Folder::new("fold");
        folder.put("notes.md", "notes");
        folder.put("src/main.rs", "fn main() {}\n");
        folder.put("src/deep/lib.rs", "pub fn x() {}\n");
        let mut tab = open(&folder);
        let press = |tab: &mut Open, code| super::key(tab, &key(code), 10);
        let (shown, _) = settled(&mut tab, 90, 20);
        assert!(shown.contains(&format!("▾ {}/", folder.0)) && shown.contains("▾ src/") && shown.contains("lib.rs"), "{shown}");
        assert_eq!(tab.share.selected_path(), Some("notes.md"), "the first file at first");
        // src/: ← folds it, its files go; → opens it again.
        press(&mut tab, KeyCode::Down);
        press(&mut tab, KeyCode::Left);
        let (shown, footer) = settled(&mut tab, 90, 20);
        assert!(shown.contains("▸ src/") && !shown.contains("main.rs"), "{shown}");
        assert!(footer.hints.contains(&("⏎", "unfold")));
        press(&mut tab, KeyCode::Right);
        let (shown, _) = settled(&mut tab, 90, 20);
        assert!(shown.contains("main.rs"), "{shown}");
        // ← on a file goes up to its folder; ⏎ on a folder folds it.
        press(&mut tab, KeyCode::Down);
        press(&mut tab, KeyCode::Down);
        press(&mut tab, KeyCode::Left);
        assert_eq!(tab.share.selected_path(), Some("src/deep"));
        press(&mut tab, KeyCode::Enter);
        let (shown, _) = settled(&mut tab, 90, 20);
        assert!(shown.contains("▸ deep/") && !shown.contains("lib.rs"), "{shown}");
        // Ctrl+O: everything open again; again, every folder folded.
        let ctrl_o = KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL);
        let _ = super::key(&mut tab, &ctrl_o, 10);
        let (shown, _) = settled(&mut tab, 90, 20);
        assert!(shown.contains("lib.rs") && shown.contains("main.rs"), "{shown}");
        let _ = super::key(&mut tab, &ctrl_o, 10);
        let (shown, _) = settled(&mut tab, 90, 20);
        assert!(shown.contains("▸ src/") && shown.contains("notes.md") && !shown.contains("main.rs"), "{shown}");
        assert_eq!(tab.share.selected_path(), Some("src"), "folded away: the folder holding it");
        // The shared folder itself folds: only it is left.
        press(&mut tab, KeyCode::Home);
        press(&mut tab, KeyCode::Enter);
        let (shown, _) = settled(&mut tab, 90, 20);
        assert!(shown.contains(&format!("▸ {}/", folder.0)) && !shown.contains("notes.md"), "{shown}");
        // And it cannot be deleted.
        press(&mut tab, KeyCode::Char('x'));
        assert!(tab.share.armed.is_none());
    }

    #[test]
    fn markdown_is_shown_rendered_and_m_shows_it_raw() {
        let folder = Folder::new("markdown");
        folder.put("notes.md", "# Plan\n\nSome **bold** words.\n");
        folder.put("raw.txt", "# not markdown **here**\n");
        let mut tab = open(&folder);
        let (shown, footer) = settled(&mut tab, 90, 20);
        let plain = strip_codes(&shown);
        assert!(plain.contains("Plan") && plain.contains("Some bold words.") && !plain.contains("**"), "{plain}");
        assert!(footer.hints.contains(&("m", "raw")));
        let _ = super::key(&mut tab, &key(KeyCode::Char('m')), 10);
        let (shown, footer) = settled(&mut tab, 90, 20);
        assert!(shown.contains("# Plan") && shown.contains("Some **bold** words."), "{shown}");
        assert!(footer.hints.contains(&("m", "rendered")));
        // A file that is not markdown is shown as it is, either way.
        let _ = super::key(&mut tab, &key(KeyCode::Char('m')), 10);
        let _ = super::key(&mut tab, &key(KeyCode::Down), 10);
        let (shown, footer) = settled(&mut tab, 90, 20);
        assert!(shown.contains("# not markdown **here**"), "{shown}");
        assert!(!footer.hints.iter().any(|(key, _)| *key == "m"));
    }

    fn strip_codes(text: &str) -> String {
        let mut plain = String::new();
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                plain.push(c);
            }
        }
        plain
    }

    #[test]
    fn moving_on_keeps_the_last_preview_up_and_coming_back_is_instant() {
        let folder = Folder::new("preview-move");
        folder.put("a.md", "alpha text");
        folder.put("b.md", "beta text");
        let mut tab = open(&folder);
        let (shown, _) = settled(&mut tab, 80, 20);
        assert!(shown.contains("alpha text"), "{shown}");
        // Down: b's preview is on its way; a's stays up meanwhile.
        let _ = super::key(&mut tab, &key(KeyCode::Down), 10);
        let (drawn, _) = super::body(&mut tab, 80, 20);
        let shown = text(&drawn);
        assert!(shown.contains("alpha text") && shown.contains("reading…"), "{shown}");
        assert!(!shown.lines().any(|line| line.trim() == "reading…"), "no bare flash: {shown}");
        let (shown, _) = settled(&mut tab, 80, 20);
        assert!(shown.contains("beta text") && !shown.contains("reading…"), "{shown}");
        // Back up: from the cache, at once.
        let _ = super::key(&mut tab, &key(KeyCode::Up), 10);
        let (drawn, _) = super::body(&mut tab, 80, 20);
        assert!(text(&drawn).contains("alpha text") && !text(&drawn).contains("reading…"), "{}", text(&drawn));
        assert!(tab.share.reading.is_none(), "nothing read again");
    }

    #[test]
    fn delete_takes_two_presses_and_tells_s0ma() {
        let _notes = notes();
        let folder = Folder::new("delete");
        folder.put("old.txt", "x");
        folder.put("keep.txt", "y");
        let mut tab = open(&folder);
        let _ = settled(&mut tab, 80, 20);
        assert_eq!(tab.share.selected.as_deref(), Some("keep.txt"));
        let _ = super::key(&mut tab, &key(KeyCode::Down), 10);
        let _ = super::key(&mut tab, &key(KeyCode::Char('x')), 10);
        let (shown, footer) = settled(&mut tab, 80, 20);
        assert!(text(&[footer.left.clone()]).contains("delete old.txt?"), "{shown}");
        // Anything else keeps it.
        let _ = super::key(&mut tab, &key(KeyCode::Esc), 10);
        assert!(std::path::Path::new(&folder.0).join("old.txt").exists());
        let _ = super::key(&mut tab, &key(KeyCode::Char('x')), 10);
        let _ = super::key(&mut tab, &key(KeyCode::Char('x')), 10);
        let (shown, _) = settled(&mut tab, 80, 20);
        assert!(!std::path::Path::new(&folder.0).join("old.txt").exists());
        assert!(!shown.contains("old.txt") && shown.contains("keep.txt"), "{shown}");
        let note = crate::shared_files::take_note().unwrap_or_default();
        assert!(note.contains("deleted /share/old.txt"), "{note}");
    }

    #[test]
    fn an_edit_over_a_change_made_meanwhile_is_kept_beside_it() {
        let _notes = notes();
        let folder = Folder::new("conflict");
        folder.put("draft.md", "one\n");
        let stamp = shared_files::read(&folder.0, "draft.md", 100, true, &mut local).unwrap().stamp;
        // Saved over the version the operator opened.
        assert!(save_edit(&folder.0, "draft.md", &stamp, "one\n", "one\ntwo\n", &mut local).unwrap().starts_with("saved"));
        assert_eq!(folder.get("draft.md"), "one\ntwo\n");
        // s0ma wrote meanwhile: the operator's goes beside it, s0ma's stays.
        std::thread::sleep(Duration::from_millis(15));
        folder.put("draft.md", "s0ma's\n");
        let error = save_edit(&folder.0, "draft.md", &stamp, "one\n", "the operator's\n", &mut local).unwrap_err();
        assert!(error.contains("changed while you edited") && error.contains("draft (operator copy).md"), "{error}");
        assert_eq!((folder.get("draft.md").as_str(), folder.get("draft (operator copy).md").as_str()), ("s0ma's\n", "the operator's\n"));
        let again = save_edit(&folder.0, "draft.md", &stamp, "one\n", "again\n", &mut local).unwrap_err();
        assert!(again.contains("draft (operator copy 2).md"), "{again}");
        let note = crate::shared_files::take_note().unwrap_or_default();
        assert!(note.contains("+two") && note.contains("beside it as /share/draft (operator copy).md"), "{note}");
    }

    #[test]
    fn a_copy_saved_again_replaces_the_last_one_unless_it_was_changed() {
        let folder = Folder::new("copy");
        let into = Folder::new("copy-into");
        let into = std::path::Path::new(&into.0).to_path_buf();
        folder.put("report.md", "v1\n");
        let source = format!("{}/report.md", folder.0);
        assert!(save_copy(&source, &into, &mut local).unwrap().contains("report.md"));
        folder.put("report.md", "v2\n");
        save_copy(&source, &into, &mut local).unwrap();
        assert_eq!(std::fs::read_to_string(into.join("report.md")).unwrap(), "v2\n");
        assert!(!into.join("report (2).md").exists(), "the old copy was replaced");
        // The operator changed his copy: it is kept.
        std::fs::write(into.join("report.md"), "mine\n").unwrap();
        folder.put("report.md", "v3\n");
        save_copy(&source, &into, &mut local).unwrap();
        assert_eq!(std::fs::read_to_string(into.join("report.md")).unwrap(), "mine\n");
        assert_eq!(std::fs::read_to_string(into.join("report (2).md")).unwrap(), "v3\n");
    }

    #[test]
    fn what_operator_adds_is_not_news_to_him_and_a_new_file_is_selected_once_listed() {
        let folder = Folder::new("own");
        folder.put("a.md", "a");
        let mut tab = open(&folder);
        let _ = settled(&mut tab, 80, 20);
        // Added by him: listed unmarked.
        folder.put("b.md", "b");
        looked(|looked| looked.own.insert(seen_key(&folder.0, "b.md")));
        tab.share.reload();
        let (shown, _) = settled(&mut tab, 80, 20);
        assert!(shown.contains("b.md") && !shown.contains("[!]"), "{shown}");
        // A file being made keeps the selection until the listing has it.
        tab.share.selected = Some("c.md".into());
        let path = std::path::Path::new(&folder.0).join("c.md");
        start(&mut tab, "making", move || {
            std::thread::sleep(Duration::from_millis(50));
            std::fs::write(path, "c").map(|_| "made".to_string()).map_err(|error| error.to_string())
        });
        let _ = settled(&mut tab, 80, 20);
        assert_eq!(tab.share.selected.as_deref(), Some("c.md"));
    }

    #[test]
    fn x_twice_deletes_a_folder_with_its_files_after_saying_how_many() {
        let _notes = notes();
        let folder = Folder::new("rm-folder");
        folder.put("old/a.md", "a");
        folder.put("old/deep/b.md", "b");
        folder.put("keep.md", "k");
        let mut tab = open(&folder);
        let _ = settled(&mut tab, 80, 20);
        tab.share.selected = Some("old".into());
        let _ = super::key(&mut tab, &key(KeyCode::Char('x')), 10);
        let (_, footer) = settled(&mut tab, 80, 20);
        let asked: String = footer.left.iter().map(|(_, text)| text.as_str()).collect();
        assert!(asked.contains("delete old/ and the 2 files in it?"), "{asked}");
        // Esc keeps it.
        let _ = super::key(&mut tab, &key(KeyCode::Esc), 10);
        assert!(std::path::Path::new(&folder.0).join("old/deep/b.md").exists());
        // x twice: gone, all of it; the rest stays.
        let _ = super::key(&mut tab, &key(KeyCode::Char('x')), 10);
        let _ = super::key(&mut tab, &key(KeyCode::Char('x')), 10);
        let (shown, _) = settled(&mut tab, 80, 20);
        assert!(!std::path::Path::new(&folder.0).join("old").exists(), "{shown}");
        assert!(std::path::Path::new(&folder.0).join("keep.md").exists());
        let told = crate::shared_files::take_note().unwrap_or_default();
        assert!(told.contains("old/a.md") && told.contains("old/deep/b.md"), "s0ma hears of each: {told}");
    }

    #[test]
    fn f_makes_a_folder_where_the_selection_is() {
        let folder = Folder::new("folders");
        folder.put("ideas/first.md", "1");
        let mut tab = open(&folder);
        let _ = settled(&mut tab, 80, 20);
        tab.share.selected = Some("ideas".into());
        let (_, footer) = settled(&mut tab, 80, 20);
        assert!(footer.hints.contains(&("f", "folder")), "{:?}", footer.hints);
        let _ = super::key(&mut tab, &key(KeyCode::Char('f')), 10);
        assert_eq!(typed(&tab), "ideas/", "from the selected folder");
        paste(&mut tab, "drafts/late");
        let _ = super::key(&mut tab, &key(KeyCode::Enter), 10);
        let (shown, _) = settled(&mut tab, 80, 20);
        assert!(std::path::Path::new(&folder.0).join("ideas/drafts/late").is_dir(), "{shown}");
        assert!(shown.contains("late/"), "listed: {shown}");
        assert_eq!(tab.share.selected.as_deref(), Some("ideas/drafts/late"));
        // Already there, or hidden: nothing made.
        for (path, said) in [("ideas", "already there"), (".secret", "would be hidden")] {
            let _ = super::key(&mut tab, &key(KeyCode::Char('f')), 10);
            tab.share.input.as_mut().unwrap().text.clear();
            tab.share.input.as_mut().unwrap().cursor = 0;
            paste(&mut tab, path);
            let _ = super::key(&mut tab, &key(KeyCode::Enter), 10);
            assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| text.contains(said) && *problem), "{path}: {:?}", tab.note);
            tab.note = None;
        }
        assert!(!std::path::Path::new(&folder.0).join(".secret").exists());
    }

    fn typed(tab: &Open) -> String {
        tab.share.input.as_ref().map(|input| input.text.iter().collect()).unwrap_or_default()
    }

    #[test]
    fn tab_completes_a_new_files_path_from_the_folder() {
        let folder = Folder::new("complete-new");
        folder.put("ideas/first.md", "1");
        folder.put("ideas/fun.md", "2");
        folder.put("notes.md", "n");
        folder.put("plan.md", "p");
        let mut tab = open(&folder);
        let _ = settled(&mut tab, 80, 20);
        // On notes.md: the new file starts at the top.
        tab.share.selected = Some("notes.md".into());
        let _ = super::key(&mut tab, &key(KeyCode::Char('n')), 10);
        assert_eq!(typed(&tab), "");
        paste(&mut tab, "id");
        let _ = super::key(&mut tab, &key(KeyCode::Tab), 10);
        assert_eq!(typed(&tab), "ideas/", "one match: the folder, no space");
        let _ = super::key(&mut tab, &key(KeyCode::Tab), 10);
        assert_eq!(typed(&tab), "ideas/f", "as far as both go");
        let (shown, footer) = settled(&mut tab, 80, 20);
        assert!(shown.contains("first.md  fun.md"), "both listed under the line: {shown}");
        assert!(footer.hints.iter().any(|(key, _)| *key == "tab"));
        paste(&mut tab, "u");
        let _ = super::key(&mut tab, &key(KeyCode::Tab), 10);
        assert_eq!(typed(&tab), "ideas/fun.md");
        assert!(tab.share.input.as_ref().unwrap().choices.is_empty());
        // Typing on clears the list; nothing matching says so.
        paste(&mut tab, "x");
        let _ = super::key(&mut tab, &key(KeyCode::Tab), 10);
        assert_eq!(typed(&tab), "ideas/fun.mdx");
        assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| text == "no match" && *problem));
        // From the top: the folder and the files beside it.
        let found = in_folder(tab.share.entries(), "");
        let labels: Vec<&str> = found.candidates.iter().map(|(_, label, _)| label.as_str()).collect();
        assert_eq!(labels, ["ideas/", "notes.md", "plan.md"]);
    }

    #[test]
    fn tab_completes_paths_on_operators_machine_to_add() {
        let folder = Folder::new("complete-add");
        let mine = Folder::new("complete-mine");
        folder.put("x", "x");
        mine.put("report one.md", "1");
        mine.put("report two.md", "2");
        mine.put("solo.txt", "s");
        mine.put("sub/inner.txt", "i");
        let mut tab = open(&folder);
        let _ = settled(&mut tab, 80, 20);
        let _ = super::key(&mut tab, &key(KeyCode::Char('a')), 10);
        paste(&mut tab, &format!("{}/so", mine.0));
        let _ = super::key(&mut tab, &key(KeyCode::Tab), 10);
        assert_eq!(typed(&tab), format!("{}/solo.txt ", mine.0), "a file: a space after it for the next");
        // A second path, with a space in its name.
        paste(&mut tab, &format!("{}/rep", mine.0));
        let _ = super::key(&mut tab, &key(KeyCode::Tab), 10);
        assert_eq!(typed(&tab), format!("{0}/solo.txt {0}/report\\ ", mine.0));
        assert_eq!(tab.share.input.as_ref().unwrap().choices, ["report one.md", "report two.md"]);
        paste(&mut tab, "t");
        let _ = super::key(&mut tab, &key(KeyCode::Tab), 10);
        assert_eq!(typed(&tab), format!("{0}/solo.txt {0}/report\\ two.md ", mine.0));
        // Both arrive as they are named.
        let paths = crate::share::parse_paths(&typed(&tab)).unwrap();
        assert_eq!(paths[1].file_name().unwrap(), "report two.md");
        // A folder: no space, ready for what is in it.
        let found = on_machine(&format!("{}/su", mine.0));
        assert_eq!(found.candidates, [(format!("{}/sub/", mine.0), "sub/".to_string(), false)]);
    }

    #[test]
    fn copies_of_two_files_with_one_name_are_both_kept_and_each_replaced_in_its_place() {
        let folder = Folder::new("copy-two");
        let into = Folder::new("copy-two-into");
        let into = std::path::Path::new(&into.0).to_path_buf();
        folder.put("a/report.md", "A1\n");
        folder.put("b/report.md", "B1\n");
        let (a, b) = (format!("{}/a/report.md", folder.0), format!("{}/b/report.md", folder.0));
        save_copy(&a, &into, &mut local).unwrap();
        save_copy(&b, &into, &mut local).unwrap();
        assert_eq!(std::fs::read_to_string(into.join("report.md")).unwrap(), "A1\n");
        assert_eq!(std::fs::read_to_string(into.join("report (2).md")).unwrap(), "B1\n");
        folder.put("b/report.md", "B2\n");
        assert!(save_copy(&b, &into, &mut local).unwrap().contains("over your last copy"));
        assert_eq!(std::fs::read_to_string(into.join("report (2).md")).unwrap(), "B2\n");
        assert_eq!(std::fs::read_to_string(into.join("report.md")).unwrap(), "A1\n");
        // Gone on s0ma's side: the copy there stays.
        std::fs::remove_file(std::path::Path::new(&folder.0).join("b/report.md")).unwrap();
        assert!(save_copy(&b, &into, &mut local).is_err());
        assert_eq!(std::fs::read_to_string(into.join("report (2).md")).unwrap(), "B2\n");
        let left: Vec<_> = std::fs::read_dir(&into).unwrap().filter_map(|entry| entry.ok()).map(|entry| entry.file_name()).collect();
        assert_eq!(left.len(), 2, "nothing left over: {left:?}");
    }

    #[test]
    fn a_file_kept_beside_is_recorded_as_it_is() {
        let _notes = notes();
        let folder = Folder::new("kept-new");
        folder.put("idea.md", "s0ma's\n");
        let error = make(&folder.0, "idea.md", "the operator's\n", &mut local).unwrap_err();
        assert!(error.contains("idea (operator copy).md"), "{error}");
        let note = crate::shared_files::take_note().unwrap_or_default();
        assert!(note.contains("created /share/idea (operator copy).md"), "{note}");
    }

    #[test]
    fn a_trailing_newline_from_the_editor_is_no_change() {
        assert!(same("a", "a\n") && same("a\n", "a\n"));
        assert!(!same("a\n", "a\n\n") && !same("a", "b\n"));
    }

    #[test]
    fn new_files_start_in_the_selected_folder() {
        let _notes = notes();
        let folder = Folder::new("new");
        folder.put("docs/a.md", "a");
        let mut tab = open(&folder);
        let _ = settled(&mut tab, 80, 20);
        let _ = super::key(&mut tab, &key(KeyCode::Down), 10);
        assert_eq!(tab.share.selected.as_deref(), Some("docs/a.md"));
        let _ = super::key(&mut tab, &key(KeyCode::Char('n')), 10);
        let typed: String = tab.share.input.as_ref().unwrap().text.iter().collect();
        assert_eq!(typed, "docs/");
        paste(&mut tab, "b.md");
        let (shown, _) = settled(&mut tab, 80, 20);
        assert!(shown.contains("new file ›") && shown.contains("docs/b.md"), "{shown}");
        // A path longer than the line: its end, with the cursor, in sight.
        paste(&mut tab, &format!("/{}/end.md", "x".repeat(150)));
        let (shown, _) = settled(&mut tab, 80, 20);
        let line = shown.lines().find(|line| line.contains("new file ›")).unwrap();
        assert!(line.contains('…') && line.contains("/end.md") && text_width(line) <= 80, "{line}");
        let _ = super::key(&mut tab, &key(KeyCode::Esc), 10);
        assert!(tab.share.input.is_none());
        // Made (as the editor would leave it).
        assert!(make(&folder.0, "docs/b.md", "hello\n", &mut local).unwrap().starts_with("made"));
        assert_eq!(folder.get("docs/b.md"), "hello\n");
        let note = crate::shared_files::take_note().unwrap_or_default();
        assert!(note.contains("created /share/docs/b.md") && note.contains("hello"), "{note}");
    }
}
