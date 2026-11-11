//! The editor tab (Alt+E): files of the shared folder edited in s0ma's own
//! look, with s0ma's changes shown in it (`claude/editor-spec.md`, E1).
//!
//! The text and its rules are `editor.rs`; the colours `editor_highlight.rs`.
//! This is the tab: the files open, keys, drawing, and the work that goes to
//! s0ma's machine (reading, saving, watching), which runs in the background
//! as the share tab's does.
//!
//! The open files outlive the tab: Esc goes back to s0ma's page and Alt+E
//! comes back to them, unsaved changes and all. Unsaved changes are also
//! copied to `data/editor/` every few seconds, so a crash loses little;
//! Alt+E offers them back after a restart.

use super::share_view;
use super::{Footer, Open, Outcome};
use crate::editor::{Buffer, Indent, Merged, Pos};
use crate::editor_highlight::{Class, Highlighter, Language};
use crate::line_edit::Draft;
use crate::mistake_view::{Line, Style};
use crate::shared_files::{self, Change, Expect, WriteError};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// How often an open file is checked for changes made elsewhere.
const WATCH_EVERY: Duration = Duration::from_secs(2);
/// How often unsaved changes are copied for recovery.
const RECOVER_EVERY: Duration = Duration::from_secs(5);
/// The gutter's marks are worked out again after typing pauses this long.
const MARKS_AFTER: Duration = Duration::from_millis(250);
/// Side by side with the preview from this many columns.
const SIDE_BY_SIDE: usize = 120;
/// Matches drawn and counted, at most.
const MATCHES_SHOWN: usize = 10_000;

// Colours: s0ma's palette (claude/editor-spec.md, "Colours").
const TEXT: &str = "\x1b[0;37m";
const BOLD: &str = "\x1b[0;1;37m";
const GREEN: &str = "\x1b[0;32m";
const CYAN: &str = "\x1b[0;36m";
const GREY: &str = "\x1b[0;3;90m";
const RED: &str = "\x1b[0;31m";
const NUMBER: &str = "\x1b[0;90m";
const OPERATOR_MARK: &str = "\x1b[0;38;2;95;175;255m";
const S0MA_MARK: &str = "\x1b[0;38;2;51;255;51m";
const RESET: &str = "\x1b[0m";
/// The selection's background (a dark teal), the text's colour kept.
const SELECTED: &str = "48;5;23";

type Slot<T> = Arc<Mutex<Option<T>>>;

fn spawn<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Option<Slot<T>> {
    let slot: Slot<T> = Arc::new(Mutex::new(None));
    let answer = slot.clone();
    std::thread::Builder::new()
        .name("editor".into())
        .spawn(move || {
            let result = work();
            if let Ok(mut answer) = answer.lock() {
                *answer = Some(result);
            }
            super::poke();
        })
        .ok()?;
    Some(slot)
}

fn take<T>(slot: &Slot<T>) -> Option<T> {
    slot.lock().ok().and_then(|mut answer| answer.take())
}

// ------------------------------------------------------------------ the open files

pub(super) struct Editor {
    docs: Vec<Doc>,
    current: usize,
    /// What Ctrl+C and Ctrl+X took last (the system clipboard can't be read
    /// back from a terminal).
    clipboard: String,
    /// Unsaved changes from before a restart, offered back (`None`: not
    /// looked for yet).
    recovered: Option<Vec<Recovery>>,
    selected_recovery: usize,
    /// The open files as s0ma was last told of them.
    reported: Option<String>,
}

static EDITOR: Mutex<Editor> =
    Mutex::new(Editor { docs: Vec::new(), current: 0, clipboard: String::new(), recovered: None, selected_recovery: 0, reported: None });

fn editor() -> MutexGuard<'static, Editor> {
    EDITOR.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What is open in the editor: the current file's name and whether any file
/// has unsaved changes, for the tab bar; `None` with nothing open.
pub(super) fn label() -> Option<(String, usize, bool)> {
    let editor = editor();
    let doc = editor.docs.get(editor.current)?;
    let dirty = editor.docs.iter().any(Doc::dirty_now);
    Some((doc.name(), editor.docs.len(), dirty))
}

/// The find bar, go to line, or nothing.
enum Bar {
    Find { query: Draft, replace: Draft, in_replace: bool, origin: Pos },
    Goto(Draft),
}

/// Work on s0ma's machine for an open file.
enum Job {
    Saving { slot: Slot<SaveOutcome>, revision: u64, text: String, since: Instant },
    Watching(Slot<Result<Option<String>, String>>),
    Reloading(Slot<Result<shared_files::Read, String>>),
}

/// What a save came to.
enum SaveOutcome {
    /// Written as it was: the new stamp.
    Saved { stamp: String },
    /// The file had changed: both sides kept, written, with the lines that
    /// came from the other side marked.
    Merged { stamp: String, text: String, theirs: Vec<bool>, base: String },
    /// The file had changed in the same places: nothing written.
    Conflict { stamp: String, theirs_text: String, merged: Vec<Merged> },
    Failed(String),
}

/// A stretch where your change and s0ma's touch the same lines: yours is in
/// the text from `line` (`mine` lines), s0ma's drawn above it.
#[derive(Clone, Debug)]
struct Conflict {
    line: usize,
    mine: usize,
    theirs: Vec<String>,
}

struct Doc {
    dir: String,
    path: String,
    buffer: Buffer,
    language: Language,
    highlight: Highlighter,
    /// The file as it is on s0ma's machine, as far as this buffer knows:
    /// what a save writes over, and the merge's base.
    base: String,
    /// Its stamp; `None` for a new file not made yet.
    stamp: Option<String>,
    /// The lines as opened, for the marks of your changes.
    opened: Vec<String>,
    /// Whether the text differs from `base`, as of a revision.
    dirty: (u64, bool),
    marks: Marks,
    wrap: bool,
    /// The split view (Ctrl+P twice): the text beside it rendered, or the
    /// rendered text alone when the tab is narrow.
    preview: bool,
    /// The live preview (Ctrl+P): the markdown rendered in place, the line
    /// being edited as written (`editor_live.rs`).
    live: bool,
    /// Each line's place with fenced code, as of a revision, for `live`.
    fences: Option<(u64, Vec<crate::editor_live::Fence>)>,
    top: (usize, usize),
    left: usize,
    bar: Option<Bar>,
    job: Option<Job>,
    watched: Instant,
    /// The file changed on s0ma's machine while this had unsaved changes.
    changed_there: bool,
    conflicts: Vec<Conflict>,
    /// Ctrl+Q once with unsaved changes; again within a few seconds drops them.
    /// Ctrl+D or Ctrl+Q pressed once with unsaved changes, which, and when:
    /// only the same key again does it (the other asks for itself).
    close_armed: Option<(char, Instant)>,
    recovered_at: (u64, Instant),
    /// The rows as last drawn: each screen row's line and first column, for
    /// a click; and the text's first column on screen.
    rows: Vec<Option<(usize, usize)>>,
    gutter: usize,
    rendered: Option<(u64, usize, Vec<String>)>,
    preview_scroll: usize,
    /// The wheel moved the split view's rendered side: it stays where it
    /// was put, not following the cursor, until a key.
    preview_free: bool,
    /// The wheel moved the view: it stays where it was put until a key.
    free_scroll: bool,
    /// The left button is down on the text: a selection is being dragged,
    /// so the wheel, or the mouse past the top or bottom, extends it.
    dragging: bool,
}

#[derive(Default)]
struct Marks {
    revision: u64,
    edited: Option<Instant>,
    mine: Vec<bool>,
    removed: Vec<bool>,
}

impl Doc {
    fn new(dir: &str, path: &str, text: &str, stamp: Option<String>) -> Self {
        let language = crate::editor_highlight::language(path);
        let default_indent = if language.tabs { Indent::Tabs } else { Indent::Spaces(language.width) };
        let buffer = Buffer::from_text(text, if language.tabs { 8 } else { 4 }, default_indent);
        let highlight = Highlighter::new(&language, text.len());
        let opened = buffer.lines().to_vec();
        Self {
            dir: dir.to_string(),
            path: path.to_string(),
            wrap: language.prose,
            language,
            highlight,
            buffer,
            base: text.to_string(),
            stamp,
            opened,
            dirty: (0, false),
            marks: Marks::default(),
            preview: false,
            live: false,
            fences: None,
            top: (0, 0),
            left: 0,
            bar: None,
            job: None,
            watched: Instant::now(),
            changed_there: false,
            conflicts: Vec::new(),
            close_armed: None,
            recovered_at: (0, Instant::now()),
            rows: Vec::new(),
            gutter: 0,
            rendered: None,
            preview_scroll: 0,
            preview_free: false,
            free_scroll: false,
            dragging: false,
        }
    }

    fn name(&self) -> String {
        self.path.rsplit('/').next().unwrap_or(&self.path).to_string()
    }

    fn shown(&self) -> String {
        shared_files::shown(&self.dir, &self.path)
    }

    /// Unsaved changes: the text is not what is on s0ma's machine (or, for a
    /// new file, there is any).
    fn dirty_now(&self) -> bool {
        if self.dirty.0 == self.buffer.revision {
            return self.dirty.1;
        }
        self.buffer.text() != self.base || (self.stamp.is_none() && !self.buffer.text().is_empty())
    }

    fn refresh_dirty(&mut self) -> bool {
        if self.dirty.0 != self.buffer.revision {
            let dirty = self.buffer.text() != self.base || (self.stamp.is_none() && !self.buffer.text().is_empty());
            self.dirty = (self.buffer.revision, dirty);
        }
        self.dirty.1
    }

    fn markdown(&self) -> bool {
        self.language.name == "markdown"
    }

    /// A change of the text: the colours from its line on, and the marks
    /// after a pause.
    fn edited(&mut self) {
        if let Some(line) = self.buffer.take_changed_from() {
            self.highlight.invalidate(line);
        }
        self.marks.edited = Some(Instant::now());
        self.close_armed = None;
    }
}

// ------------------------------------------------------------------ opening

/// Open `path` of the shared folder, read as `text` at `stamp` (or a new
/// file, `None`), or go to it if it is open.
pub(super) fn open(dir: &str, path: &str, text: &str, stamp: Option<String>) {
    let mut editor = editor();
    if let Some(index) = editor.docs.iter().position(|doc| doc.dir == dir && doc.path == path) {
        editor.current = index;
        return;
    }
    editor.docs.push(Doc::new(dir, path, text, stamp));
    editor.current = editor.docs.len() - 1;
}

/// Whether `path` of the shared folder is open in the editor, and with
/// unsaved changes (for s0ma's tools' results).
pub(crate) fn is_open(dir: &str, path: &str) -> Option<bool> {
    let editor = editor();
    editor.docs.iter().find(|doc| doc.dir == dir && doc.path == path).map(Doc::dirty_now)
}

/// The files open in the operator's editor, for s0ma: said when it changes.
pub(crate) fn take_note() -> Option<String> {
    let mut editor = editor();
    let now: Vec<String> = editor
        .docs
        .iter()
        .map(|doc| format!("{}{}", doc.shown(), if doc.dirty_now() { " (unsaved changes)" } else { "" }))
        .collect();
    let line = if now.is_empty() { String::new() } else { now.join(", ") };
    if editor.reported.as_deref().unwrap_or("") == line {
        return None;
    }
    editor.reported = Some(line.clone());
    Some(if line.is_empty() {
        "\n\n[SHARED FILES] no shared file is open in the operator's editor now".into()
    } else {
        format!("\n\n[SHARED FILES] open in the operator's editor: {line}")
    })
}

// ------------------------------------------------------------------ recovery

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Recovery {
    dir: String,
    path: String,
    stamp: Option<String>,
    base: String,
    text: String,
    saved_at: chrono::DateTime<chrono::Local>,
}

fn recovery_dir() -> Option<PathBuf> {
    super::ARCHIVE.get().and_then(|archive| archive.parent()).map(|data| data.join("editor"))
}

fn recovery_file(dir: &str, path: &str) -> Option<PathBuf> {
    let digest = format!("{:x}", Sha256::digest(format!("{dir}\n{path}").as_bytes()));
    recovery_dir().map(|folder| folder.join(format!("{}.json", &digest[..16])))
}

fn write_recovery(doc: &Doc) {
    // A job's content is kept off the disk while the job is open.
    if crate::jobs::open_job().is_some() {
        return;
    }
    let (Some(folder), Some(file)) = (recovery_dir(), recovery_file(&doc.dir, &doc.path)) else { return };
    let recovery = Recovery {
        dir: doc.dir.clone(),
        path: doc.path.clone(),
        stamp: doc.stamp.clone(),
        base: doc.base.clone(),
        text: doc.buffer.text(),
        saved_at: chrono::Local::now(),
    };
    if std::fs::create_dir_all(&folder).is_err() {
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&folder, std::fs::Permissions::from_mode(0o700));
    }
    if let Ok(json) = serde_json::to_vec(&recovery) {
        // A recovery copy: one missed is made again at the next change.
        let _ = crate::util::write_atomic(&file, json);
    }
}

fn drop_recovery(dir: &str, path: &str) {
    if let Some(file) = recovery_file(dir, path) {
        let _ = std::fs::remove_file(file);
    }
}

fn read_recoveries() -> Vec<Recovery> {
    let Some(folder) = recovery_dir() else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(folder) else { return Vec::new() };
    let mut found: Vec<Recovery> = entries
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| std::fs::read(entry.path()).ok())
        .filter_map(|bytes| serde_json::from_slice(&bytes).ok())
        .collect();
    found.sort_by_key(|copy| std::cmp::Reverse(copy.saved_at));
    found
}

/// At exit: every file with unsaved changes copied now.
pub(crate) fn flush() {
    let editor = editor();
    for doc in editor.docs.iter().filter(|doc| doc.dirty_now()) {
        write_recovery(doc);
    }
}

/// A job closed or a redaction struck: the copies on disk go (their text
/// may hold what was taken out).
pub(crate) fn forget_recoveries() {
    if let Some(folder) = recovery_dir() {
        let _ = std::fs::remove_dir_all(folder);
    }
}

// ------------------------------------------------------------------ background work

/// Saving: written over the version this was based on; if the file changed
/// meanwhile, the two merged, and written if they don't clash.
fn save_work(dir: &str, path: &str, stamp: Option<&str>, base: &str, mine: &str, run: shared_files::Run) -> SaveOutcome {
    let expect = match stamp {
        Some(stamp) => Expect::Stamp(stamp.to_string()),
        None => Expect::New,
    };
    let mut against = (expect, base.to_string());
    for _ in 0..3 {
        let (expect, base_now) = &against;
        match shared_files::write(dir, path, mine_or_merged(mine, base_now, base).as_bytes(), expect, run) {
            Ok(new) if base_now == base => return SaveOutcome::Saved { stamp: new },
            Ok(new) => {
                // Written merged (below): what was written, and its marks.
                let merged = merge(base, mine, base_now);
                if let MergeResult::Clean { text, theirs } = merged {
                    return SaveOutcome::Merged { stamp: new, text, theirs, base: base_now.clone() };
                }
                return SaveOutcome::Saved { stamp: new };
            }
            Err(WriteError::Changed) => {
                let read = match shared_files::read(dir, path, shared_files::MAX_WHOLE_BYTES, true, run) {
                    Ok(read) => read,
                    Err(error) => return SaveOutcome::Failed(error),
                };
                let Some(theirs) = shared_files::text_of(&read.bytes).map(str::to_string) else {
                    return SaveOutcome::Failed("the file on s0ma's machine is no longer text".into());
                };
                if theirs == mine {
                    return SaveOutcome::Saved { stamp: read.stamp };
                }
                match merge(base, mine, &theirs) {
                    MergeResult::Clean { .. } => against = (Expect::Stamp(read.stamp), theirs),
                    MergeResult::Clashes(merged) => return SaveOutcome::Conflict { stamp: read.stamp, theirs_text: theirs, merged },
                }
            }
            Err(WriteError::Failed(error)) => return SaveOutcome::Failed(error),
        }
    }
    SaveOutcome::Failed("the file kept changing while it was saved; try again".into())
}

/// What to write: `mine` when the file is still `base`, else the merge.
fn mine_or_merged(mine: &str, theirs: &str, base: &str) -> String {
    if theirs == base {
        return mine.to_string();
    }
    match merge(base, mine, theirs) {
        MergeResult::Clean { text, .. } => text,
        MergeResult::Clashes(_) => mine.to_string(),
    }
}

enum MergeResult {
    Clean { text: String, theirs: Vec<bool> },
    Clashes(Vec<Merged>),
}

fn merge(base: &str, mine: &str, theirs: &str) -> MergeResult {
    let split = crate::editor::split_lines;
    let merged = crate::editor::merge3(&split(base), &split(mine), &split(theirs));
    if merged.iter().any(|line| matches!(line, Merged::Conflict { .. })) {
        return MergeResult::Clashes(merged);
    }
    let (lines, theirs_marks): (Vec<String>, Vec<bool>) = merged
        .into_iter()
        .filter_map(|line| match line {
            Merged::Line { text, theirs } => Some((text, theirs)),
            Merged::Conflict { .. } => None,
        })
        .unzip();
    let ending = if mine.contains("\r\n") { "\r\n" } else { "\n" };
    let mut text = lines.join(ending);
    if mine.ends_with('\n') || (mine.is_empty() && theirs.ends_with('\n')) {
        text.push_str(ending);
    }
    MergeResult::Clean { text, theirs: theirs_marks }
}

/// Answers from the background, taken in.
fn settle(doc: &mut Doc, tab: &mut Open) {
    let Some(job) = doc.job.take() else { return };
    match job {
        Job::Saving { slot, revision, text, since } => match take(&slot) {
            None => doc.job = Some(Job::Saving { slot, revision, text, since }),
            Some(outcome) => saved(doc, tab, outcome, revision, text),
        },
        Job::Watching(slot) => match take(&slot) {
            None => doc.job = Some(Job::Watching(slot)),
            Some(Ok(stamp)) if stamp != doc.stamp && doc.stamp.is_some() => {
                if stamp.is_none() {
                    tab.note(format!("{} was deleted on s0ma's machine · Save makes it again", doc.name()), true);
                    doc.stamp = None;
                } else if doc.refresh_dirty() {
                    if !doc.changed_there {
                        doc.changed_there = true;
                        tab.tell("changed on s0ma's machine · Save merges");
                    }
                } else {
                    let (dir, path) = (doc.dir.clone(), doc.path.clone());
                    doc.job = spawn(move || shared_files::read(&dir, &path, shared_files::MAX_WHOLE_BYTES, true, &mut crate::share::run_on_machine))
                        .map(Job::Reloading);
                }
            }
            Some(_) => {}
        },
        Job::Reloading(slot) => match take(&slot) {
            None => doc.job = Some(Job::Reloading(slot)),
            Some(Ok(read)) => match shared_files::text_of(&read.bytes) {
                Some(text) if !doc.refresh_dirty() => reload(doc, text, read.stamp),
                Some(_) => doc.changed_there = true,
                None => tab.note("the file is no longer text on s0ma's machine", true),
            },
            Some(Err(error)) => tab.note(error, true),
        },
    }
}

/// The file changed on s0ma's machine and nothing here was unsaved: taken
/// in, with the changed lines marked as s0ma's.
fn reload(doc: &mut Doc, text: &str, stamp: String) {
    let (changed, _) = crate::editor::changed_lines(&crate::editor::split_lines(&doc.base), &crate::editor::split_lines(text));
    doc.buffer.replace_all(text, changed);
    doc.base = text.to_string();
    doc.stamp = Some(stamp.clone());
    doc.changed_there = false;
    doc.edited();
    share_view::mark_seen(&doc.dir, &doc.path, &stamp);
}

fn saved(doc: &mut Doc, tab: &mut Open, outcome: SaveOutcome, revision: u64, text: String) {
    let before = doc.base.clone();
    let before_stamp = doc.stamp.clone();
    let tell_change = |after: &str, stamp: &str| {
        share_view::tell(match &before_stamp {
            Some(before_stamp) => Change::Edited {
                path: doc.path.clone(),
                before_stamp: before_stamp.clone(),
                before: shared_files::carried(&before),
                after: shared_files::carried(after),
                stamp: stamp.to_string(),
            },
            None => Change::Created { path: doc.path.clone(), after: shared_files::carried(after), stamp: stamp.to_string() },
        });
    };
    match outcome {
        SaveOutcome::Saved { stamp } => {
            tell_change(&text, &stamp);
            share_view::mark_seen(&doc.dir, &doc.path, &stamp);
            doc.base = text;
            doc.stamp = Some(stamp);
            doc.changed_there = false;
            doc.dirty = (0, false);
            if doc.buffer.revision == revision {
                drop_recovery(&doc.dir, &doc.path);
            }
        }
        SaveOutcome::Merged { stamp, text: merged, theirs, base } => {
            tell_change(&merged, &stamp);
            share_view::mark_seen(&doc.dir, &doc.path, &stamp);
            // Typed while it saved: kept, on top of the merge.
            let now = doc.buffer.text();
            let result = if doc.buffer.revision == revision { merged.clone() } else { mine_or_merged(&now, &merged, &text) };
            let marks = if doc.buffer.revision == revision { theirs } else { crate::editor::changed_lines(&crate::editor::split_lines(&now), &crate::editor::split_lines(&result)).0 };
            doc.buffer.replace_all(&result, marks);
            let _ = base;
            doc.base = merged;
            doc.stamp = Some(stamp);
            doc.changed_there = false;
            doc.edited();
            tab.tell("saved, with s0ma's changes kept (marked green)");
            if doc.buffer.revision == revision + 1 {
                drop_recovery(&doc.dir, &doc.path);
            }
        }
        SaveOutcome::Conflict { stamp, theirs_text, merged } => {
            // Yours in the text, s0ma's above each clash, until each is settled.
            let mut lines = Vec::new();
            let mut theirs_marks = Vec::new();
            let mut conflicts = Vec::new();
            for part in merged {
                match part {
                    Merged::Line { text, theirs } => {
                        lines.push(text);
                        theirs_marks.push(theirs);
                    }
                    Merged::Conflict { theirs, mine } => {
                        conflicts.push(Conflict { line: lines.len(), mine: mine.len(), theirs });
                        theirs_marks.extend(std::iter::repeat_n(false, mine.len()));
                        lines.extend(mine);
                    }
                }
            }
            let ending = if text.contains("\r\n") { "\r\n" } else { "\n" };
            let mut joined = lines.join(ending);
            if text.ends_with('\n') {
                joined.push_str(ending);
            }
            doc.buffer.replace_all(&joined, theirs_marks);
            doc.base = theirs_text;
            doc.stamp = Some(stamp);
            doc.conflicts = conflicts;
            doc.changed_there = false;
            doc.edited();
            if let Some(first) = doc.conflicts.first() {
                doc.buffer.go(Pos::new(first.line, 0), false);
            }
            tab.note(format!("not saved: s0ma changed the same lines · {} to settle, each with S0ma's or Yours", doc.conflicts.len()), true);
        }
        SaveOutcome::Failed(error) => tab.note(format!("not saved: {error}"), true),
    }
}

fn start_save(doc: &mut Doc, tab: &mut Open) {
    if !doc.conflicts.is_empty() {
        return tab.note("settle the marked stretches first, each with S0ma's or Yours", true);
    }
    if matches!(doc.job, Some(Job::Saving { .. })) {
        return;
    }
    let text = doc.buffer.text();
    if doc.stamp.is_some() && text == doc.base && !doc.changed_there {
        return tab.tell("no changes");
    }
    doc.buffer.seal();
    let (dir, path, stamp, base, mine) = (doc.dir.clone(), doc.path.clone(), doc.stamp.clone(), doc.base.clone(), text.clone());
    match spawn(move || save_work(&dir, &path, stamp.as_deref(), &base, &mine, &mut crate::share::run_on_machine)) {
        Some(slot) => doc.job = Some(Job::Saving { slot, revision: doc.buffer.revision, text, since: Instant::now() }),
        None => tab.note("could not start saving", true),
    }
}

/// Every couple of seconds, with nothing else under way: has the file
/// changed on s0ma's machine?
fn watch(doc: &mut Doc) {
    if doc.job.is_some() || doc.stamp.is_none() || doc.watched.elapsed() < WATCH_EVERY {
        return;
    }
    doc.watched = Instant::now();
    let (dir, path) = (doc.dir.clone(), doc.path.clone());
    doc.job = spawn(move || shared_files::stamp_of(&dir, &path, &mut crate::share::run_on_machine)).map(Job::Watching);
}

/// Background work for every open file: called as the tab draws, and from
/// the tick while it is up.
fn upkeep(editor: &mut Editor, tab: &mut Open) {
    let current = editor.current;
    for (index, doc) in editor.docs.iter_mut().enumerate() {
        // Notes about other files than the one shown would mislead: theirs wait.
        if index == current {
            settle(doc, tab);
            watch(doc);
        }
        if doc.recovered_at.0 != doc.buffer.revision && doc.recovered_at.1.elapsed() >= RECOVER_EVERY {
            if doc.refresh_dirty() {
                write_recovery(doc);
            }
            doc.recovered_at = (doc.buffer.revision, Instant::now());
        }
    }
}

// ------------------------------------------------------------------ keys

fn plain(key: &KeyEvent) -> bool {
    !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
}

pub(super) fn key(tab: &mut Open, key: &KeyEvent, page: usize) -> Outcome {
    let mut editor = editor();
    if editor.docs.is_empty() {
        return recovery_key(&mut editor, tab, key);
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let count = editor.docs.len();

    // Between the open files, and closing one.
    match key.code {
        KeyCode::PageUp | KeyCode::PageDown if ctrl => {
            editor.current = if key.code == KeyCode::PageDown { (editor.current + 1) % count } else { (editor.current + count - 1) % count };
            return Outcome::Stay;
        }
        KeyCode::Char('q' | 'Q') if ctrl => return close_current(&mut editor, tab),
        _ => {}
    }
    let clipboard = editor.clipboard.clone();
    let current = editor.current;
    let doc = &mut editor.docs[current];
    settle(doc, tab);

    // A key brings the split view's rendered side back to the cursor.
    doc.preview_free = false;
    if doc.bar.is_some() {
        bar_key(doc, tab, key);
        return Outcome::Stay;
    }
    // The preview alone, full screen: it scrolls; Ctrl+P or Esc back.
    if doc.preview && tab_width_is_narrow() {
        match key.code {
            KeyCode::Up => doc.preview_scroll = doc.preview_scroll.saturating_sub(1),
            KeyCode::Down => doc.preview_scroll += 1,
            KeyCode::PageUp => doc.preview_scroll = doc.preview_scroll.saturating_sub(page),
            KeyCode::PageDown | KeyCode::Char(' ') => doc.preview_scroll += page,
            KeyCode::Home => doc.preview_scroll = 0,
            KeyCode::Esc => doc.preview = false,
            KeyCode::Char('p' | 'P') if ctrl => doc.preview = false,
            _ => {}
        }
        return Outcome::Stay;
    }

    // Clashes to settle first: moving and Alt+Y / Alt+N only.
    if !doc.conflicts.is_empty() {
        if alt && matches!(key.code, KeyCode::Char('y' | 'Y' | 'n' | 'N')) {
            settle_conflict(doc, tab, matches!(key.code, KeyCode::Char('y' | 'Y')));
            return Outcome::Stay;
        }
        let moving = matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right | KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home | KeyCode::End | KeyCode::Esc)
            || (ctrl && matches!(key.code, KeyCode::Char('f' | 'F' | 'g' | 'G')));
        if !moving {
            tab.note("settle the marked stretches first, each with S0ma's or Yours", true);
            return Outcome::Stay;
        }
    }

    doc.free_scroll = false;
    let before = doc.buffer.revision;
    let mut outcome = Outcome::Stay;
    let mut copied: Option<String> = None;
    match key.code {
        KeyCode::Esc => {
            if doc.buffer.selection().is_some() {
                doc.buffer.clear_selection();
            } else {
                outcome = Outcome::Closed;
            }
        }
        // Moving, Shift selecting.
        KeyCode::Left if ctrl || alt => doc.buffer.word_left(shift),
        KeyCode::Right if ctrl || alt => doc.buffer.word_right(shift),
        KeyCode::Up if alt => doc.buffer.move_lines(false),
        KeyCode::Down if alt => doc.buffer.move_lines(true),
        KeyCode::Left => doc.buffer.left(shift),
        KeyCode::Right => doc.buffer.right(shift),
        KeyCode::Up => doc.buffer.vertical(-1, shift),
        KeyCode::Down => doc.buffer.vertical(1, shift),
        KeyCode::PageUp => doc.buffer.vertical(-(page as isize), shift),
        KeyCode::PageDown => doc.buffer.vertical(page as isize, shift),
        KeyCode::Home if ctrl => doc.buffer.start_of_file(shift),
        KeyCode::End if ctrl => doc.buffer.end_of_file(shift),
        KeyCode::Home => doc.buffer.home(shift),
        KeyCode::End => doc.buffer.end(shift),
        // Editing.
        KeyCode::Enter if plain(key) || shift => doc.buffer.newline(),
        KeyCode::Backspace if ctrl || alt => doc.buffer.delete_word_before(),
        KeyCode::Backspace => doc.buffer.backspace(),
        KeyCode::Delete => doc.buffer.delete(),
        KeyCode::Tab if plain(key) => match doc.buffer.selection() {
            Some((start, end)) if start.line != end.line => doc.buffer.indent_lines(false),
            _ => {
                let unit = doc.buffer.indent.unit();
                let insert = if unit == "\t" {
                    unit
                } else {
                    let at = doc.buffer.display_col(doc.buffer.cursor.line, doc.buffer.cursor.col);
                    " ".repeat(unit.len() - at % unit.len())
                };
                doc.buffer.insert(&insert);
            }
        },
        KeyCode::BackTab => doc.buffer.indent_lines(true),
        KeyCode::Char(c) if ctrl && !alt => match c.to_ascii_lowercase() {
            'a' => doc.buffer.select_all(),
            'c' => copied = Some(doc.buffer.copy_text()),
            'x' => copied = Some(doc.buffer.cut()),
            // The system clipboard where it can be read, else what was
            // copied here.
            'v' => match crate::clipboard::read().map(|(text, _)| text).or_else(|| (!clipboard.is_empty()).then(|| clipboard.clone())) {
                Some(text) => doc.buffer.paste(&text),
                None => tab.tell("nothing to paste · the terminal's own paste (ctrl+shift+v) brings anything in"),
            },
            'z' if shift => {
                doc.buffer.redo();
            }
            'z' => {
                doc.buffer.undo();
            }
            'y' => {
                doc.buffer.redo();
            }
            'k' => doc.buffer.delete_lines(),
            'l' => {
                tab.frame.clear();
                let _ = crossterm::queue!(std::io::stdout(), crossterm::terminal::Clear(crossterm::terminal::ClearType::All));
            }
            'w' | 'h' => doc.buffer.delete_word_before(),
            '/' | '_' | '7' => {
                if let Some(prefix) = doc.language.comment {
                    doc.buffer.toggle_comment(prefix);
                }
            }
            's' => start_save(doc, tab),
            'f' => {
                let selected = doc.buffer.selected_text().filter(|text| !text.contains('\n'));
                let mut query = Draft::default();
                if let Some(text) = selected {
                    query.insert_str(&text);
                }
                let origin = doc.buffer.selection().map_or(doc.buffer.cursor, |(start, _)| start);
                doc.bar = Some(Bar::Find { query, replace: Draft::default(), in_replace: false, origin });
            }
            'g' => doc.bar = Some(Bar::Goto(Draft::default())),
            // Text, then the live preview, then the split view, then text.
            'p' => {
                if doc.markdown() {
                    (doc.live, doc.preview) = match (doc.live, doc.preview) {
                        (false, false) => (true, false),
                        (true, _) => (false, true),
                        (false, true) => (false, false),
                    };
                    doc.preview_scroll = 0;
                    doc.preview_free = false;
                } else {
                    tab.tell("preview is for markdown files");
                }
            }
            'e' => {
                let text = doc.buffer.text();
                let path = doc.path.clone();
                match share_view::in_editor(tab, &path, &text) {
                    Ok(Some(edited)) if !share_view::same(&text, &edited) => {
                        let theirs = vec![false; crate::editor::split_lines(&edited).len()];
                        doc.buffer.replace_all(&edited, theirs);
                    }
                    Ok(_) => {}
                    Err(error) => tab.note(error, true),
                }
            }
            'd' => {
                // Leaving s0ma with unsaved changes asks first.
                if doc.refresh_dirty() && doc.close_armed.is_none_or(|(key, at)| key != 'd' || at.elapsed() > super::CLOSE_AGAIN) {
                    doc.close_armed = Some(('d', Instant::now()));
                    tab.ask("unsaved changes · again leaves (they are kept for next time)");
                } else {
                    outcome = Outcome::Eof;
                }
            }
            _ => {}
        },
        KeyCode::Char('z' | 'Z') if alt && !ctrl => doc.wrap = !doc.wrap,
        KeyCode::Char(c) if plain(key) || (shift && !ctrl && !alt) => doc.buffer.insert(&c.to_string()),
        _ => {}
    }
    if doc.buffer.revision != before {
        doc.edited();
    }
    if let Some(text) = copied.filter(|text| !text.is_empty()) {
        editor.clipboard = text.clone();
        drop(editor);
        let said = crate::clipboard::copy(&text);
        tab.tell(said.trim_start_matches("[copy] ").replacen(" via ", " copied via ", 1));
    }
    outcome
}

fn tab_width_is_narrow() -> bool {
    crossterm::terminal::size().map_or(true, |(columns, _)| (columns as usize) < SIDE_BY_SIDE)
}

/// A paste from the terminal: typed in one step.
pub(super) fn paste(tab: &mut Open, text: &str) {
    let mut editor = editor();
    let current = editor.current;
    let Some(doc) = editor.docs.get_mut(current) else { return };
    if let Some(bar) = &mut doc.bar {
        let line = text.lines().next().unwrap_or("");
        match bar {
            Bar::Find { query, replace, in_replace, .. } => {
                if *in_replace { replace.insert_str(line) } else { query.insert_str(line) }
            }
            Bar::Goto(draft) => draft.insert_str(line),
        }
        return;
    }
    if !doc.conflicts.is_empty() {
        return tab.note("settle the marked stretches first, each with S0ma's or Yours", true);
    }
    doc.buffer.paste(text);
    doc.edited();
}

/// Ctrl+Q: the file closed; with unsaved changes, asked first.
fn close_current(editor: &mut Editor, tab: &mut Open) -> Outcome {
    let current = editor.current;
    let doc = &mut editor.docs[current];
    if doc.refresh_dirty() && doc.close_armed.is_none_or(|(key, at)| key != 'q' || at.elapsed() > super::CLOSE_AGAIN) {
        doc.close_armed = Some(('q', Instant::now()));
        tab.ask("unsaved · again drops the changes · Save keeps them");
        return Outcome::Stay;
    }
    let doc = editor.docs.remove(current);
    drop_recovery(&doc.dir, &doc.path);
    editor.current = editor.current.min(editor.docs.len().saturating_sub(1));
    if editor.docs.is_empty() { Outcome::Closed } else { Outcome::Stay }
}

fn bar_key(doc: &mut Doc, tab: &mut Open, key: &KeyEvent) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let Some(bar) = doc.bar.as_mut() else { return };
    match bar {
        Bar::Goto(draft) => match key.code {
            KeyCode::Esc => doc.bar = None,
            KeyCode::Enter => {
                let typed = draft.text().trim().to_string();
                doc.bar = None;
                match goto_target(&typed, doc.buffer.cursor.line, doc.buffer.line_count()) {
                    Some(pos) => doc.buffer.go(pos, false),
                    None => tab.note("a line: 12, 12:4 or +5", true),
                }
            }
            _ => edit_draft(draft, key),
        },
        Bar::Find { query, replace, in_replace, origin } => match key.code {
            KeyCode::Esc => doc.bar = None,
            KeyCode::Tab | KeyCode::BackTab => *in_replace = !*in_replace,
            KeyCode::Enter if ctrl => {
                let (query, with) = (query.text().to_string(), replace.text().to_string());
                let replaced = doc.buffer.replace_matches(&query, &with);
                doc.edited();
                tab.tell(format!("{replaced} replaced"));
            }
            KeyCode::Enter if *in_replace => {
                // The match at the cursor replaced, and on to the next.
                let (query_text, with) = (query.text().to_string(), replace.text().to_string());
                if let Some((start, end)) = doc.buffer.selection()
                    && doc.buffer.slice(start, end).to_lowercase() == query_text.to_lowercase()
                {
                    doc.buffer.insert(&with);
                    doc.edited();
                }
                let from = doc.buffer.cursor;
                if let Some((start, end)) = doc.buffer.find(&query_text, from, true, true) {
                    doc.buffer.go(start, false);
                    doc.buffer.go(end, true);
                }
            }
            KeyCode::Enter => {
                let from = doc.buffer.selection().map_or(doc.buffer.cursor, |(start, _)| start);
                let query_text = query.text().to_string();
                match doc.buffer.find(&query_text, from, !shift, false) {
                    Some((start, end)) => {
                        doc.buffer.go(start, false);
                        doc.buffer.go(end, true);
                    }
                    None => tab.note("not found", true),
                }
            }
            _ if *in_replace => edit_draft(replace, key),
            _ => {
                let before = query.text().to_string();
                edit_draft(query, key);
                // Typing the query: the first match from where the search began.
                if query.text() != before {
                    let query_text = query.text().to_string();
                    let origin = *origin;
                    match doc.buffer.find(&query_text, origin, true, true) {
                        Some((start, end)) => {
                            doc.buffer.go(start, false);
                            doc.buffer.go(end, true);
                        }
                        None => doc.buffer.go(origin, false),
                    }
                }
            }
        },
    }
}

fn edit_draft(draft: &mut Draft, key: &KeyEvent) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Backspace if ctrl => draft.delete_word_before(),
        KeyCode::Backspace => draft.backspace(),
        KeyCode::Delete => draft.delete(),
        KeyCode::Left => draft.left(),
        KeyCode::Right => draft.right(),
        KeyCode::Home => draft.home(),
        KeyCode::End => draft.end(),
        KeyCode::Char('u' | 'U') if ctrl => draft.kill_line_before(),
        KeyCode::Char('w' | 'W') if ctrl => draft.delete_word_before(),
        KeyCode::Char(c) if !ctrl => draft.insert(c),
        _ => {}
    }
}

/// Go to line: `12`, `12:4`, `+5`, `-3` (lines and columns from 1).
fn goto_target(typed: &str, current: usize, count: usize) -> Option<Pos> {
    let (line, col) = typed.split_once(':').map_or((typed, None), |(line, col)| (line, Some(col)));
    let line = line.trim();
    let target = if let Some(down) = line.strip_prefix('+') {
        current + down.parse::<usize>().ok()?
    } else if let Some(up) = line.strip_prefix('-') {
        current.saturating_sub(up.parse::<usize>().ok()?)
    } else {
        line.parse::<usize>().ok()?.saturating_sub(1)
    };
    let col = col.map_or(Some(0), |col| col.trim().parse::<usize>().ok().map(|col| col.saturating_sub(1)))?;
    Some(Pos::new(target.min(count.saturating_sub(1)), col))
}

/// Alt+Y (s0ma's) or Alt+N (yours) on the clash at or after the cursor.
fn settle_conflict(doc: &mut Doc, tab: &mut Open, take_theirs: bool) {
    let cursor = doc.buffer.cursor.line;
    let index = doc
        .conflicts
        .iter()
        .position(|conflict| cursor < conflict.line + conflict.mine.max(1))
        .unwrap_or(doc.conflicts.len().saturating_sub(1));
    let conflict = doc.conflicts.remove(index);
    if take_theirs {
        let start = Pos::new(conflict.line, 0);
        let replacement = conflict.theirs.join("\n");
        let delta = conflict.theirs.len() as isize - conflict.mine as isize;
        // The lines from `start` to the end of the stretch, replaced.
        if conflict.mine == 0 {
            doc.buffer.go(start, false);
            doc.buffer.paste(&format!("{replacement}\n"));
        } else {
            let end_line = conflict.line + conflict.mine - 1;
            doc.buffer.go(start, false);
            doc.buffer.go(Pos::new(end_line, doc.buffer.line(end_line).len()), true);
            doc.buffer.paste(&replacement);
        }
        for line in conflict.line..(conflict.line + conflict.theirs.len()).min(doc.buffer.theirs.len()) {
            doc.buffer.theirs[line] = true;
        }
        for later in doc.conflicts.iter_mut().filter(|later| later.line > conflict.line) {
            later.line = (later.line as isize + delta).max(0) as usize;
        }
        doc.edited();
    }
    match doc.conflicts.first() {
        Some(next) => {
            let line = next.line;
            doc.buffer.go(Pos::new(line, 0), false);
        }
        None => tab.tell("all settled"),
    }
}

// ------------------------------------------------------------------ the mouse

pub(super) fn mouse(mouse: &MouseEvent) {
    let mut editor = editor();
    let current = editor.current;
    let Some(doc) = editor.docs.get_mut(current) else { return };
    let row = (mouse.row as usize).checked_sub(super::body_top());
    let shift = mouse.modifiers.contains(KeyModifiers::SHIFT);
    let wheel = super::WHEEL_ROWS as isize;
    match mouse.kind {
        // Selecting (dragging, or with Shift held): the wheel carries the
        // selection with it, so it can run past one screen.
        MouseEventKind::ScrollUp | MouseEventKind::ScrollDown if doc.dragging || shift => {
            let rows = if mouse.kind == MouseEventKind::ScrollUp { -wheel } else { wheel };
            doc.buffer.vertical(rows, true);
            doc.free_scroll = false;
        }
        MouseEventKind::ScrollUp => scroll(doc, -wheel, mouse.column as usize),
        MouseEventKind::ScrollDown => scroll(doc, wheel, mouse.column as usize),
        MouseEventKind::Up(MouseButton::Left) => doc.dragging = false,
        MouseEventKind::Down(MouseButton::Left) | MouseEventKind::Drag(MouseButton::Left) => {
            let drag = matches!(mouse.kind, MouseEventKind::Drag(_));
            let visible = doc.rows.iter().take_while(|row| row.is_some()).count();
            // Dragged past the top or the bottom of the text: the view
            // scrolls that way, a few lines each move, the selection with it.
            if drag && doc.dragging {
                let past = match row {
                    None | Some(0) => Some(-3),
                    Some(row) if row + 1 >= visible.max(1) => Some(3),
                    _ => None,
                };
                if let Some(rows) = past {
                    doc.buffer.vertical(rows, true);
                    doc.free_scroll = false;
                    return;
                }
            }
            let Some(Some((line, first))) = row.and_then(|row| doc.rows.get(row)).copied() else { return };
            let column = (mouse.column as usize).saturating_sub(doc.gutter) + first;
            let col = doc.buffer.col_at(line, column);
            let extend = shift || drag;
            doc.buffer.go(Pos::new(line, col), extend);
            doc.free_scroll = false;
            doc.dragging = true;
        }
        _ => {}
    }
}

/// The view moved by `rows`, the cursor left where it is (the wheel at
/// `column`). In the split view, over the rendered side that side scrolls
/// on its own; over the text, the text; the rendered text alone, it.
fn scroll(doc: &mut Doc, rows: isize, column: usize) {
    let width = crossterm::terminal::size().map_or(80, |(columns, _)| columns as usize);
    let rendered_side = width < SIDE_BY_SIDE || column >= width / 2;
    if doc.preview && rendered_side {
        doc.preview_scroll = (doc.preview_scroll as isize + rows).max(0) as usize;
        doc.preview_free = true;
        return;
    }
    let line = (doc.top.0 as isize + rows).clamp(0, doc.buffer.line_count() as isize - 1) as usize;
    doc.top = (line, 0);
    doc.free_scroll = true;
}

// ------------------------------------------------------------------ recovery, with nothing open

fn recovery_key(editor: &mut Editor, tab: &mut Open, key: &KeyEvent) -> Outcome {
    let mut found = editor.recovered.take().unwrap_or_else(read_recoveries);
    let outcome = match key.code {
        KeyCode::Esc => Outcome::Closed,
        KeyCode::Up => {
            editor.selected_recovery = editor.selected_recovery.saturating_sub(1);
            Outcome::Stay
        }
        KeyCode::Down => {
            editor.selected_recovery = (editor.selected_recovery + 1).min(found.len().saturating_sub(1));
            Outcome::Stay
        }
        KeyCode::Enter => {
            if let Some(recovery) = found.get(editor.selected_recovery).cloned() {
                found.remove(editor.selected_recovery);
                let mut doc = Doc::new(&recovery.dir, &recovery.path, &recovery.base, recovery.stamp.clone());
                let theirs = vec![false; crate::editor::split_lines(&recovery.text).len()];
                doc.buffer.replace_all(&recovery.text, theirs);
                doc.edited();
                editor.docs.push(doc);
                editor.current = editor.docs.len() - 1;
                tab.tell("restored · Save merges it with any change made since");
            }
            Outcome::Stay
        }
        KeyCode::Char('x') if plain(key) => {
            if let Some(recovery) = found.get(editor.selected_recovery).cloned() {
                drop_recovery(&recovery.dir, &recovery.path);
                found.remove(editor.selected_recovery);
                editor.selected_recovery = editor.selected_recovery.min(found.len().saturating_sub(1));
            }
            Outcome::Stay
        }
        _ => Outcome::Stay,
    };
    editor.recovered = Some(found);
    outcome
}

// ------------------------------------------------------------------ drawing

/// The tab's rows, its footer, and where the cursor is among the rows.
pub(super) fn body(tab: &mut Open, width: usize, rows: usize) -> (Vec<Line>, Footer, Option<(usize, usize)>) {
    let mut editor = editor();
    upkeep(&mut editor, tab);
    if editor.docs.is_empty() {
        let has = editor.recovered.as_ref().is_none_or(|found| !found.is_empty());
        let body = recovery_body(&mut editor, width);
        let mut hints = vec![("esc", "back")];
        if has && editor.recovered.as_ref().is_some_and(|found| !found.is_empty()) {
            hints = vec![("⏎", "restore"), ("x", "drop"), ("esc", "back")];
        }
        return (body, Footer::new(Vec::new(), hints), None);
    }
    let count = editor.docs.len();
    let current = editor.current;
    let doc = &mut editor.docs[current];
    let dirty = doc.refresh_dirty();
    // The marks, once typing pauses.
    if doc.marks.revision != doc.buffer.revision && doc.marks.edited.is_none_or(|at| at.elapsed() >= MARKS_AFTER) {
        let (mine, removed) = crate::editor::changed_lines(&doc.opened, doc.buffer.lines());
        doc.marks = Marks { revision: doc.buffer.revision, edited: None, mine, removed };
    }
    let bar_rows = usize::from(doc.bar.is_some());
    let text_rows = rows.saturating_sub(bar_rows).max(1);
    let (mut lines, cursor) = if doc.preview && width < SIDE_BY_SIDE {
        (preview_rows(doc, width, text_rows, None), None)
    } else if doc.preview {
        let left = width / 2;
        let (mut text, cursor) = text_view(doc, left - 1, text_rows);
        // Each row as wide as its half, so the preview lines up beside it.
        for row in &mut text {
            let used: usize = row.iter().map(|(style, text)| if *style == Style::Raw { crate::page::width(text) } else { super::text_width(text) }).sum();
            row.push((Style::Plain, " ".repeat((left - 1).saturating_sub(used))));
        }
        let fraction = doc.buffer.cursor.line as f64 / doc.buffer.line_count().max(1) as f64;
        let preview = preview_rows(doc, width - left - 2, text_rows, Some(fraction));
        let joined = text
            .into_iter()
            .zip(preview)
            .map(|(left_row, right_row)| {
                let mut row = left_row;
                row.push((Style::Dim, " │".into()));
                row.extend(right_row);
                row
            })
            .collect();
        (joined, cursor)
    } else {
        text_view(doc, width, text_rows)
    };
    let mut cursor = cursor;
    if let Some(bar) = &doc.bar {
        let (row, column) = bar_row(doc, bar, width);
        lines.push(row);
        cursor = Some((column, text_rows));
    }
    // The footer.
    let mut left: Line = vec![(Style::Status, doc.name())];
    let folder = doc.shown();
    let folder = folder.rsplit_once('/').map_or(String::new(), |(folder, _)| folder.to_string());
    left.push((Style::Dim, format!("  {folder}")));
    let pos = doc.buffer.cursor;
    left.push((Style::Dim, format!("  ln {}, col {}", pos.line + 1, doc.buffer.display_col(pos.line, pos.col) + 1)));
    left.push((Style::Dim, format!("  {}", doc.language.name)));
    if doc.buffer.ending == crate::editor::Ending::Crlf {
        left.push((Style::Dim, "  CRLF".into()));
    }
    let default_indent = if doc.language.tabs { Indent::Tabs } else { Indent::Spaces(doc.language.width) };
    if doc.buffer.indent != default_indent {
        left.push((Style::Dim, format!("  {}", doc.buffer.indent.label())));
    }
    if count > 1 {
        left.push((Style::Dim, format!("  {} of {count}", current + 1)));
    }
    if doc.highlight.off && doc.buffer.text().len() > crate::editor_highlight::MAX_BYTES {
        left.push((Style::Dim, "  no colours (over 1 MB)".into()));
    }
    match &doc.job {
        Some(Job::Saving { since, .. }) => left.push((Style::Work, format!("  saving… {}s", since.elapsed().as_secs()))),
        _ if !doc.conflicts.is_empty() => left.push((Style::Warn, format!("  {} to settle", doc.conflicts.len()))),
        _ if doc.stamp.is_none() && dirty => left.push((Style::Warn, "  [+] new".into())),
        _ if dirty => left.push((Style::Warn, "  [+]".into())),
        _ => {}
    }
    if doc.changed_there {
        left.push((Style::Problem, "  changed on s0ma's machine".into()));
    }
    let hints: Vec<(&'static str, &'static str)> = if !doc.conflicts.is_empty() {
        vec![("alt+y", "s0ma's"), ("alt+n", "yours"), ("esc", "back")]
    } else if doc.bar.is_some() {
        match doc.bar {
            Some(Bar::Find { .. }) => vec![("⏎", "next"), ("⇧⏎", "previous"), ("tab", "replace"), ("^⏎", "all"), ("esc", "close")],
            _ => vec![("⏎", "go"), ("esc", "close")],
        }
    } else {
        let mut hints = vec![("^s", "save"), ("^f", "find")];
        if doc.markdown() {
            hints.push(("^p", match (doc.live, doc.preview) {
                (true, _) => "split",
                (false, true) => "text",
                (false, false) => "preview",
            }));
        }
        hints.extend([("^q", "close"), ("esc", "back")]);
        hints
    };
    (std::mem::take(&mut lines), Footer::new(left, hints), cursor)
}

fn recovery_body(editor: &mut Editor, width: usize) -> Vec<Line> {
    let found = editor.recovered.take().unwrap_or_else(read_recoveries);
    let mut out = Vec::new();
    if found.is_empty() {
        out = super::empty("No file open.", "In the share tab, Edit opens one; New makes one.", width);
    } else {
        out.extend([Vec::new(), vec![(Style::Plain, "  Unsaved changes from before:".into())], Vec::new()]);
        for (index, recovery) in found.iter().enumerate() {
            let selected = index == editor.selected_recovery;
            let name = shared_files::shown(&recovery.dir, &recovery.path);
            out.push(vec![
                (if selected { Style::Accent } else { Style::Plain }, format!("{} {name}", if selected { "❯" } else { " " })),
                (Style::Dim, format!("  {}", recovery.saved_at.format("%b %-d, %H:%M"))),
            ]);
        }

    }
    editor.recovered = Some(found);
    out
}

/// The find or go-to bar, and the cursor's column in it.
fn bar_row(doc: &Doc, bar: &Bar, width: usize) -> (Line, usize) {
    match bar {
        Bar::Goto(draft) => {
            let label = " go to line: ";
            (vec![(Style::Accent, label.into()), (Style::Plain, draft.text().to_string())], label.len() + super::text_width(&draft.text()[..draft.cursor()]))
        }
        Bar::Find { query, replace, in_replace, .. } => {
            let found = doc.buffer.matches(query.text(), MATCHES_SHOWN).len();
            let count = if query.is_empty() {
                String::new()
            } else if found == 0 {
                "  no matches".into()
            } else {
                format!("  {found}{} matches", if found >= MATCHES_SHOWN { "+" } else { "" })
            };
            let mut row: Line = vec![(Style::Accent, " find: ".into()), (Style::Plain, query.text().to_string())];
            let mut column = 7 + super::text_width(&query.text()[..query.cursor()]);
            row.push((Style::Dim, count.clone()));
            if *in_replace || !replace.is_empty() {
                let before: usize = row.iter().map(|(_, text)| super::text_width(text)).sum();
                row.push((Style::Accent, "   replace: ".into()));
                row.push((Style::Plain, replace.text().to_string()));
                if *in_replace {
                    column = before + 12 + super::text_width(&replace.text()[..replace.cursor()]);
                }
            }
            let _ = width;
            (row, column)
        }
    }
}

/// The markdown rendered, `rows` of it: scrolled by hand (alone) or to the
/// same place as the text (`fraction`, side by side).
fn preview_rows(doc: &mut Doc, width: usize, rows: usize, fraction: Option<f64>) -> Vec<Line> {
    let width = width.max(10);
    if doc.rendered.as_ref().is_none_or(|(revision, at, _)| *revision != doc.buffer.revision || *at != width) {
        let rendered = crate::markdown::render_lines(&doc.buffer.text(), width);
        doc.rendered = Some((doc.buffer.revision, width, rendered));
    }
    let rendered = doc.rendered.as_ref().map(|(_, _, rows)| rows.as_slice()).unwrap_or_default();
    let most = rendered.len().saturating_sub(rows);
    let top = match fraction {
        // Beside the text: where the cursor is, unless the wheel moved it.
        Some(_) if doc.preview_free => doc.preview_scroll.min(most),
        Some(fraction) => ((rendered.len() as f64 * fraction) as usize).saturating_sub(rows / 3).min(most),
        None => doc.preview_scroll.min(most),
    };
    doc.preview_scroll = top;
    let mut out: Vec<Line> = rendered.iter().skip(top).take(rows).map(|row| vec![(Style::Raw, format!("{row}{RESET}"))]).collect();
    out.resize(rows, Vec::new());
    out
}

/// What a character is drawn as.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Look {
    class: Class,
    selected: bool,
    matched: bool,
    current: bool,
    control: bool,
}

impl Look {
    fn sgr(&self) -> String {
        let base = match (self.control, self.class) {
            (true, _) | (_, Class::Error) => RED,
            (_, Class::Plain) => TEXT,
            (_, Class::Keyword | Class::Heading) => BOLD,
            (_, Class::String | Class::Raw) => GREEN,
            (_, Class::Constant) => CYAN,
            (_, Class::Comment) => GREY,
        };
        let mut extra = Vec::new();
        if self.selected {
            extra.push(SELECTED);
        }
        if self.current {
            extra.push("7");
        } else if self.matched {
            extra.push("4");
        }
        if extra.is_empty() {
            base.to_string()
        } else {
            format!("{}{}", base, extra.iter().map(|code| format!("\x1b[{code}m")).collect::<String>())
        }
    }
}

/// The gutter of a line's row `sub`: on its first, the number and its mark
/// (s0ma's change, yours, or lines removed); blank on the rest.
fn gutter_text(doc: &Doc, line: usize, sub: usize, digits: usize, gutter: usize, cursor_line: usize) -> String {
    if sub > 0 {
        return " ".repeat(gutter);
    }
    let number_style = if line == cursor_line { TEXT } else { NUMBER };
    let mut row = format!("{number_style}{:>digits$}{RESET} ", line + 1);
    let theirs = doc.buffer.theirs.get(line).copied().unwrap_or(false);
    let mine = doc.marks.mine.get(line).copied().unwrap_or(false) && doc.marks.revision == doc.buffer.revision;
    let removed = doc.marks.removed.get(line).copied().unwrap_or(false) && doc.marks.revision == doc.buffer.revision;
    row.push_str(&if theirs {
        format!("{S0MA_MARK}▎{RESET}")
    } else if mine {
        format!("{OPERATOR_MARK}▎{RESET}")
    } else if removed {
        format!("{RED}▁{RESET}")
    } else {
        " ".into()
    });
    row.push(' ');
    row
}

/// The text with its gutter, `rows` rows, scrolled to keep the cursor in
/// sight; and where the cursor is.
fn text_view(doc: &mut Doc, width: usize, rows: usize) -> (Vec<Line>, Option<(usize, usize)>) {
    let count = doc.buffer.line_count();
    let digits = count.to_string().len().max(3);
    let gutter = digits + 3;
    doc.gutter = gutter;
    let room = width.saturating_sub(gutter).max(1);
    let tab = doc.buffer.tab_width;
    let cursor = doc.buffer.cursor;
    let cursor_col = doc.buffer.display_col(cursor.line, cursor.col);

    // The live preview: each line's place with fenced code, worked out again
    // when the text changes.
    let live = doc.live && doc.bar.is_none() && doc.markdown();
    if live && doc.fences.as_ref().is_none_or(|(revision, _)| *revision != doc.buffer.revision) {
        doc.fences = Some((doc.buffer.revision, crate::editor_live::fences(doc.buffer.lines().iter().map(String::as_str))));
    }
    let fences: Vec<crate::editor_live::Fence> = if live { doc.fences.as_ref().map(|(_, fences)| fences.clone()).unwrap_or_default() } else { Vec::new() };
    let held = doc.buffer.selection();
    // A line shown rendered: in the live preview, every one but the
    // cursor's, those the selection touches and those with a clash above.
    let rendered = |doc: &Doc, line: usize| -> Option<Vec<(crate::editor_live::Look, String)>> {
        let raw = !live
            || line == cursor.line
            || held.is_some_and(|(start, end)| start.line <= line && line <= end.line)
            || doc.conflicts.iter().any(|conflict| conflict.line == line);
        if raw {
            return None;
        }
        let text = doc.buffer.line(line).replace('\t', &" ".repeat(tab));
        Some(crate::editor_live::render(&text, fences.get(line).copied().unwrap_or(crate::editor_live::Fence::Outside), room))
    };

    // How many rows a line takes (wrapped), and the rows of conflict above it.
    let line_rows = |doc: &Doc, line: usize| -> usize {
        let extra = doc.conflicts.iter().filter(|conflict| conflict.line == line).map(|conflict| conflict.theirs.len() + 2).sum::<usize>();
        let width = match rendered(doc, line) {
            Some(pieces) => crate::editor_live::width(&pieces),
            None => crate::editor::display_width(doc.buffer.line(line), tab),
        };
        let text = if doc.wrap { width.div_ceil(room).max(1) } else { 1 };
        text + extra
    };
    // Scrolled: the cursor's line in sight, wrapped rows counted.
    let cursor_sub = if doc.wrap { cursor_col / room } else { 0 };
    if doc.free_scroll {
        // Left where the wheel put it.
    } else if doc.top.0 > cursor.line || (doc.top.0 == cursor.line && doc.top.1 > cursor_sub) {
        doc.top = (cursor.line, 0);
    }
    while !doc.free_scroll {
        let mut used = 0;
        for line in doc.top.0..cursor.line {
            used += line_rows(doc, line);
        }
        used = used.saturating_sub(doc.top.1) + cursor_sub + 1;
        let extra_above = doc.conflicts.iter().filter(|conflict| conflict.line == cursor.line).map(|conflict| conflict.theirs.len() + 2).sum::<usize>();
        if used + extra_above <= rows || doc.top.0 >= cursor.line {
            break;
        }
        doc.top = (doc.top.0 + 1, 0);
    }
    if !doc.wrap {
        if cursor_col < doc.left {
            doc.left = cursor_col.saturating_sub(room / 4);
        } else if cursor_col >= doc.left + room {
            doc.left = cursor_col + 1 - room + room / 4;
        }
    } else {
        doc.left = 0;
    }

    let selection = doc.buffer.selection();
    let query = match &doc.bar {
        Some(Bar::Find { query, .. }) if !query.is_empty() => Some(query.text().to_string()),
        _ => None,
    };
    let mut out: Vec<Line> = Vec::with_capacity(rows);
    let mut map: Vec<Option<(usize, usize)>> = Vec::with_capacity(rows);
    let mut cursor_at = None;
    let mut line = doc.top.0;
    let mut skip = doc.top.1;
    while out.len() < rows && line < count {
        // s0ma's side of a clash, above yours.
        for conflict in doc.conflicts.clone().iter().filter(|conflict| conflict.line == line) {
            let header = format!("{}{} ▎ s0ma's{RESET}", " ".repeat(digits + 1), S0MA_MARK);
            for row in std::iter::once(header)
                .chain(conflict.theirs.iter().map(|text| format!("{}{S0MA_MARK}▎ {GREEN}{}{RESET}", " ".repeat(digits + 1), clip_display(text, room, tab))))
                .chain(std::iter::once(format!("{}{} ▎ yours{RESET}", " ".repeat(digits + 1), OPERATOR_MARK)))
            {
                if skip > 0 {
                    skip -= 1;
                    continue;
                }
                if out.len() < rows {
                    out.push(vec![(Style::Raw, row)]);
                    map.push(None);
                }
            }
        }
        // Rendered in place (the live preview): its pieces, wrapped.
        if let Some(pieces) = rendered(doc, line) {
            let mut rows_of: Vec<String> = vec![String::new()];
            let mut used = 0;
            for (look, text) in &pieces {
                let sgr = look.sgr();
                rows_of.last_mut().expect("a row").push_str(&sgr);
                for c in text.chars() {
                    let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
                    if used + w > room {
                        if !doc.wrap {
                            break;
                        }
                        rows_of.push(sgr.clone());
                        used = 0;
                    }
                    rows_of.last_mut().expect("a row").push(c);
                    used += w;
                }
            }
            for (sub, content) in rows_of.into_iter().enumerate() {
                if skip > 0 {
                    skip -= 1;
                    continue;
                }
                if out.len() >= rows {
                    break;
                }
                out.push(vec![(Style::Raw, format!("{}{content}{RESET}", gutter_text(doc, line, sub, digits, gutter, cursor.line)))]);
                map.push(Some((line, sub * room)));
            }
            line += 1;
            skip = 0;
            continue;
        }
        let text = doc.buffer.line(line).to_string();
        let spans = doc.highlight.line(doc.buffer.lines(), line).to_vec();
        let matches: Vec<(usize, usize)> = query
            .as_ref()
            .map(|query| {
                let fold = !query.chars().any(char::is_uppercase);
                let (hay, needle) = if fold { (text.to_lowercase(), query.to_lowercase()) } else { (text.clone(), query.clone()) };
                if hay.len() != text.len() {
                    return Vec::new();
                }
                hay.match_indices(&needle).map(|(at, found)| (at, at + found.len())).collect()
            })
            .unwrap_or_default();
        // The characters, each with its look and display column.
        let mut cells: Vec<(String, usize, Look)> = Vec::new();
        let mut col = 0;
        for (index, c) in text.char_indices() {
            let class = spans.iter().find(|(start, end, _)| index >= *start && index < *end).map_or(Class::Plain, |(_, _, class)| *class);
            let pos = Pos::new(line, index);
            let selected = selection.is_some_and(|(start, end)| pos >= start && pos < end);
            let matched = matches.iter().any(|(start, end)| index >= *start && index < *end);
            let current = matched && selection.is_some_and(|(start, end)| start.line == line && end.line == line && matches.iter().any(|(s, e)| *s == start.col && *e == end.col && index >= *s && index < *e));
            let w = crate::editor::char_width(c, col, tab);
            let (shown, control) = match c {
                '\t' => (" ".repeat(w), false),
                c if c.is_control() => (format!("^{}", ((c as u8) ^ 0x40) as char), true),
                c => (c.to_string(), false),
            };
            cells.push((shown, w, Look { class, selected, matched, current, control }));
            col += w;
        }
        // A selection running past the line's end shows as one cell.
        let past_end = selection.is_some_and(|(start, end)| start <= Pos::new(line, text.len()) && end > Pos::new(line, text.len()));
        let total = col;
        let segments: Vec<(usize, usize)> = if doc.wrap {
            let mut segments = Vec::new();
            let mut start = 0;
            while start < total.max(1) {
                segments.push((start, start + room));
                start += room;
            }
            segments
        } else {
            vec![(doc.left, doc.left + room)]
        };
        let last_segment = segments.len() - 1;
        for (sub, (from, to)) in segments.into_iter().enumerate() {
            if skip > 0 {
                skip -= 1;
                continue;
            }
            if out.len() >= rows {
                break;
            }
            let mut row = gutter_text(doc, line, sub, digits, gutter, cursor.line);
            let mut at = 0;
            let mut look: Option<Look> = None;
            let mut used = 0;
            for (shown, w, cell) in &cells {
                let start = at;
                at += w;
                if at <= from || start >= to {
                    continue;
                }
                // A wide character cut by the edge: blank.
                if start < from || at > to {
                    row.push_str(TEXT);
                    let visible = at.min(to) - start.max(from);
                    row.push_str(&" ".repeat(visible));
                    used += visible;
                    look = None;
                    continue;
                }
                if look != Some(*cell) {
                    row.push_str(&cell.sgr());
                    look = Some(*cell);
                }
                row.push_str(shown);
                used += w;
            }
            if past_end && sub == last_segment && used < room {
                row.push_str(&format!("{TEXT}\x1b[{SELECTED}m \x1b[0m"));
            }
            row.push_str(RESET);
            if line == cursor.line && cursor_col >= from && (cursor_col < to || (sub == last_segment)) {
                let column = gutter + cursor_col.saturating_sub(from).min(room.saturating_sub(1));
                cursor_at = Some((column, out.len()));
            }
            out.push(vec![(Style::Raw, row)]);
            map.push(Some((line, from)));
        }
        line += 1;
        skip = 0;
    }
    // Past the end: a dim `~`.
    while out.len() < rows {
        out.push(vec![(Style::Raw, format!("{NUMBER}{:>digits$}{RESET}", "~"))]);
        map.push(None);
    }
    doc.rows = map;
    (out, cursor_at)
}

/// `text` cut to `room` display columns (a clash's lines, unwrapped).
fn clip_display(text: &str, room: usize, tab: usize) -> String {
    let mut out = String::new();
    let mut width = 0;
    for c in text.chars() {
        let w = crate::editor::char_width(c, width, tab);
        if width + w > room.saturating_sub(2) {
            out.push('…');
            break;
        }
        if c == '\t' {
            out.push_str(&" ".repeat(w));
        } else if !c.is_control() {
            out.push(c);
        }
        width += w;
    }
    out
}

#[cfg(test)]
pub(super) fn reset_for_test() {
    let mut editor = editor();
    editor.docs.clear();
    editor.current = 0;
    editor.clipboard.clear();
    editor.recovered = Some(Vec::new());
    editor.reported = None;
}

#[cfg(test)]
pub(super) fn text_for_test() -> Option<String> {
    let editor = editor();
    editor.docs.get(editor.current).map(|doc| doc.buffer.text())
}

#[cfg(test)]
pub(super) fn wait_for_test() {
    for _ in 0..200 {
        let busy = editor().docs.iter().any(|doc| doc.job.as_ref().is_some_and(|job| match job {
            Job::Saving { slot, .. } => slot.lock().map(|answer| answer.is_none()).unwrap_or(false),
            Job::Watching(slot) => slot.lock().map(|answer| answer.is_none()).unwrap_or(false),
            Job::Reloading(slot) => slot.lock().map(|answer| answer.is_none()).unwrap_or(false),
        }));
        if !busy {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_files::tests::Folder;

    #[test]
    fn go_to_line_reads_its_forms() {
        assert_eq!(goto_target("12", 0, 100), Some(Pos::new(11, 0)));
        assert_eq!(goto_target("12:4", 0, 100), Some(Pos::new(11, 3)));
        assert_eq!(goto_target("+5", 10, 100), Some(Pos::new(15, 0)));
        assert_eq!(goto_target("-5", 3, 100), Some(Pos::new(0, 0)));
        assert_eq!(goto_target("500", 0, 100), Some(Pos::new(99, 0)), "past the end: the last line");
        assert_eq!(goto_target("x", 0, 100), None);
    }

    #[test]
    fn a_save_writes_or_merges_or_finds_the_clash() {
        let _notes = crate::shared_files::tests::NOTES.lock().unwrap_or_else(|p| p.into_inner());
        let folder = Folder::new("editor-save");
        let dir = folder.0.clone();
        let run = &mut crate::shared_files::tests::local;
        let stamp = shared_files::write(&dir, "plan.md", b"a\nb\nc\nd\ne\n", &Expect::New, run).unwrap();
        // Unchanged meanwhile: written as it is.
        let SaveOutcome::Saved { stamp: second } = save_work(&dir, "plan.md", Some(&stamp), "a\nb\nc\nd\ne\n", "a\nB\nc\nd\ne\n", run) else { panic!() };
        assert_eq!(std::fs::read_to_string(format!("{dir}/plan.md")).unwrap(), "a\nB\nc\nd\ne\n");
        // s0ma changed another line meanwhile: both kept.
        let theirs = shared_files::write(&dir, "plan.md", b"a\nB\nc\nd\nE\n", &Expect::Stamp(second), run).unwrap();
        let _ = theirs;
        let outcome = save_work(&dir, "plan.md", Some(&stamp_of(&dir)), "a\nB\nc\nd\ne\n", "x\na\nB\nc\nd\ne\n", run);
        let SaveOutcome::Saved { .. } = outcome else { panic!("its own stamp: written") };
        // Against an old stamp: merged.
        std::fs::write(format!("{dir}/plan.md"), "a\nb\nc\nd\nE\n").unwrap();
        let outcome = save_work(&dir, "plan.md", Some("0:0:0:0"), "a\nb\nc\nd\ne\n", "a\nB\nc\nd\ne\n", run);
        let SaveOutcome::Merged { text, theirs, .. } = outcome else { panic!("merged") };
        assert_eq!(text, "a\nB\nc\nd\nE\n");
        assert_eq!(std::fs::read_to_string(format!("{dir}/plan.md")).unwrap(), text, "written merged");
        assert_eq!(theirs, [false, false, false, false, true]);
        // The same line on both sides: nothing written.
        std::fs::write(format!("{dir}/plan.md"), "a\nS0MA\nc\nd\ne\n").unwrap();
        let outcome = save_work(&dir, "plan.md", Some("0:0:0:0"), "a\nb\nc\nd\ne\n", "a\nMINE\nc\nd\ne\n", run);
        let SaveOutcome::Conflict { theirs_text, merged, .. } = outcome else { panic!("a clash") };
        assert_eq!(theirs_text, "a\nS0MA\nc\nd\ne\n");
        assert!(merged.iter().any(|m| matches!(m, Merged::Conflict { .. })));
        assert_eq!(std::fs::read_to_string(format!("{dir}/plan.md")).unwrap(), "a\nS0MA\nc\nd\ne\n", "left alone");
        // A new file is made; one already there is not written over.
        let SaveOutcome::Saved { .. } = save_work(&dir, "new.md", None, "", "hello\n", run) else { panic!() };
        assert_eq!(std::fs::read_to_string(format!("{dir}/new.md")).unwrap(), "hello\n");
    }

    static LOCK: Mutex<()> = Mutex::new(());

    fn press(tab: &mut Open, code: KeyCode, modifiers: KeyModifiers) -> Outcome {
        key(tab, &KeyEvent::new(code, modifiers), 20)
    }

    fn typed(tab: &mut Open, text: &str) {
        for c in text.chars() {
            press(tab, KeyCode::Char(c), KeyModifiers::NONE);
        }
    }

    /// The background work done and taken in (as a draw does).
    fn settled(tab: &mut Open) -> Vec<String> {
        wait_for_test();
        let (rows, _, _) = body(tab, 80, 20);
        wait_for_test();
        let (rows, _, _) = if rows.is_empty() { body(tab, 80, 20) } else { (rows, Footer::new(Vec::new(), Vec::new()), None) };
        rows.iter().map(|row| row.iter().map(|(_, text)| strip(text)).collect()).collect()
    }

    fn strip(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn a_drag_past_the_edge_or_with_the_wheel_selects_more_than_a_screen() {
        let _lock = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_test();
        let text: String = (1..=200).map(|n| format!("line {n}\n")).collect();
        open("/share", "long.txt", &text, None);
        let mut tab = Open::new(super::super::View::Edit);
        let top = super::super::body_top() as u16;
        let event = |kind, row: u16| MouseEvent { kind, column: 10, row, modifiers: KeyModifiers::NONE };
        // Start at the end, drawn.
        press(&mut tab, KeyCode::End, KeyModifiers::CONTROL);
        let _ = body(&mut tab, 80, 12);
        mouse(&event(MouseEventKind::Down(MouseButton::Left), top + 5));
        let start = editor().docs[0].buffer.cursor;
        // The wheel while the button is held carries the selection up.
        mouse(&event(MouseEventKind::ScrollUp, top + 5));
        let _ = body(&mut tab, 80, 12);
        let after_wheel = editor().docs[0].buffer.cursor;
        assert_eq!(after_wheel.line + super::super::WHEEL_ROWS, start.line, "{start:?} → {after_wheel:?}");
        // Dragged onto the top row, again and again: up three lines a move.
        for _ in 0..30 {
            mouse(&event(MouseEventKind::Drag(MouseButton::Left), top));
            let _ = body(&mut tab, 80, 12);
        }
        let (from, to) = editor().docs[0].buffer.selection().expect("selected");
        assert_eq!(to, start, "the selection keeps where it began");
        assert!(start.line - from.line > 60, "far more than a screen: {from:?}..{to:?}");
        // Let go: the wheel scrolls the view again, the selection stays.
        mouse(&event(MouseEventKind::Up(MouseButton::Left), top));
        let before = editor().docs[0].buffer.cursor;
        mouse(&event(MouseEventKind::ScrollDown, top + 3));
        assert_eq!(editor().docs[0].buffer.cursor, before);
        assert!(editor().docs[0].free_scroll);
        // Ctrl+C copies it all and says so.
        press(&mut tab, KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(editor().clipboard.lines().count() > 60);
        let note = tab.note.as_ref().map(|(text, _, _)| text.clone()).unwrap_or_default();
        assert!(note.contains("lines copied via"), "{note}");
    }

    #[test]
    fn ctrl_p_goes_text_live_preview_split_and_back() {
        let _lock = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_test();
        let text = "# Plan\n\nShip **today**, then [ask](https://x.example).\n- [ ] call the bank\n```\nlet a = **b**;\n```\n";
        open("/share", "plan.md", text, None);
        let mut tab = Open::new(super::super::View::Edit);
        let shown = |tab: &mut Open| -> (Vec<String>, Footer) {
            let (rows, footer, _) = body(tab, 80, 12);
            (rows.iter().map(|row| row.iter().map(|(_, text)| strip(text)).collect()).collect(), footer)
        };
        let (rows, footer) = shown(&mut tab);
        assert!(rows[0].ends_with("# Plan"), "text: {rows:?}");
        assert!(footer.hints.contains(&("^p", "preview")));

        // Live: rendered in place, but the cursor's line as written.
        press(&mut tab, KeyCode::Char('p'), KeyModifiers::CONTROL);
        let (rows, footer) = shown(&mut tab);
        assert!(rows[0].ends_with("# Plan"), "the cursor's line stays as written: {rows:?}");
        assert!(rows[2].ends_with("Ship today, then ask."), "{rows:?}");
        assert!(rows[3].ends_with("☐ call the bank"), "{rows:?}");
        assert!(rows[5].ends_with("let a = **b**;"), "code as written: {rows:?}");
        assert!(footer.hints.contains(&("^p", "split")));
        // Still the editor: the cursor moves down and that line is the raw one now.
        press(&mut tab, KeyCode::Down, KeyModifiers::NONE);
        press(&mut tab, KeyCode::Down, KeyModifiers::NONE);
        let (rows, _) = shown(&mut tab);
        assert!(rows[0].ends_with(" Plan") && !rows[0].contains('#'), "{rows:?}");
        assert!(rows[2].ends_with("Ship **today**, then [ask](https://x.example)."), "{rows:?}");
        typed(&mut tab, "x");
        assert!(editor().docs[0].buffer.line(2).starts_with('x'), "typed in place");

        // Split: the text, and beside or instead of it the rendered text.
        press(&mut tab, KeyCode::Char('p'), KeyModifiers::CONTROL);
        assert!(editor().docs[0].preview && !editor().docs[0].live);
        // The wheel over the rendered side scrolls it on its own.
        scroll(&mut editor().docs[0], 3, usize::MAX);
        assert!(editor().docs[0].preview_free && editor().docs[0].preview_scroll == 3);
        press(&mut tab, KeyCode::Right, KeyModifiers::NONE);
        press(&mut tab, KeyCode::Char('p'), KeyModifiers::CONTROL);
        let doc = &editor().docs[0];
        assert!(!doc.preview && !doc.live && !doc.preview_free, "back to the text");
    }

    #[test]
    fn edit_save_merge_and_settle_a_clash_in_the_tab() {
        let _lock = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _notes = crate::shared_files::tests::NOTES.lock().unwrap_or_else(|p| p.into_inner());
        let _ = crate::shared_files::take_note();
        reset_for_test();
        let folder = Folder::new("editor-tab");
        let dir = folder.0.clone();
        let file = format!("{dir}/plan.md");
        let stamp = shared_files::write(&dir, "plan.md", b"a\nb\nc\n", &Expect::New, &mut crate::shared_files::tests::local).unwrap();
        open(&dir, "plan.md", "a\nb\nc\n", Some(stamp));
        let mut tab = Open::new(super::super::View::Edit);
        assert_eq!(label(), Some(("plan.md".to_string(), 1, false)));

        // Typed and saved.
        press(&mut tab, KeyCode::Down, KeyModifiers::NONE);
        press(&mut tab, KeyCode::End, KeyModifiers::NONE);
        typed(&mut tab, "!");
        assert_eq!(label().unwrap().2, true, "[+] in the bar");
        press(&mut tab, KeyCode::Char('s'), KeyModifiers::CONTROL);
        settled(&mut tab);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "a\nb!\nc\n");
        assert_eq!(label().unwrap().2, false);
        let note = crate::shared_files::take_note().unwrap_or_default();
        assert!(note.contains("edited") && note.contains("+b!"), "s0ma hears of it: {note}");

        // s0ma changed another line meanwhile: both kept, s0ma's marked.
        typed(&mut tab, "?");
        std::fs::write(&file, "a\nb!\nC\n").unwrap();
        press(&mut tab, KeyCode::Char('s'), KeyModifiers::CONTROL);
        let rows = settled(&mut tab);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "a\nb!?\nC\n", "{:?} {:?}", tab.note, text_for_test());
        assert_eq!(text_for_test().as_deref(), Some("a\nb!?\nC\n"));
        assert!(rows.iter().any(|row| row.contains("3 ▎ C")), "s0ma's line marked: {rows:?}");

        // The same line on both sides: shown, settled with Alt+Y, saved.
        press(&mut tab, KeyCode::Home, KeyModifiers::CONTROL);
        typed(&mut tab, "mine ");
        std::fs::write(&file, "theirs a\nb!?\nC\n").unwrap();
        press(&mut tab, KeyCode::Char('s'), KeyModifiers::CONTROL);
        let rows = settled(&mut tab).join("\n");
        assert!(rows.contains("s0ma's") && rows.contains("theirs a") && rows.contains("yours") && rows.contains("mine a"), "{rows}");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "theirs a\nb!?\nC\n", "nothing written over it");
        typed(&mut tab, "x");
        assert_eq!(text_for_test().as_deref(), Some("mine a\nb!?\nC\n"), "no typing until it is settled");
        press(&mut tab, KeyCode::Char('y'), KeyModifiers::ALT);
        assert_eq!(text_for_test().as_deref(), Some("theirs a\nb!?\nC\n"));
        press(&mut tab, KeyCode::Char('s'), KeyModifiers::CONTROL);
        settled(&mut tab);
        assert_eq!(text_for_test().as_deref(), Some("theirs a\nb!?\nC\n"));

        // Find, go to line, undo.
        press(&mut tab, KeyCode::Char('f'), KeyModifiers::CONTROL);
        typed(&mut tab, "c");
        press(&mut tab, KeyCode::Esc, KeyModifiers::NONE);
        typed(&mut tab, "Z");
        assert_eq!(text_for_test().as_deref(), Some("theirs a\nb!?\nZ\n"), "the match was selected and typed over");
        press(&mut tab, KeyCode::Char('z'), KeyModifiers::CONTROL);
        assert_eq!(text_for_test().as_deref(), Some("theirs a\nb!?\nC\n"));
        press(&mut tab, KeyCode::Char('g'), KeyModifiers::CONTROL);
        typed(&mut tab, "2");
        press(&mut tab, KeyCode::Enter, KeyModifiers::NONE);
        typed(&mut tab, ">");
        assert_eq!(text_for_test().as_deref(), Some("theirs a\n>b!?\nC\n"));

        // Ctrl+D asked about leaving: Ctrl+Q after it asks for itself, and
        // does not drop the changes on Ctrl+D's question.
        assert_eq!(press(&mut tab, KeyCode::Char('d'), KeyModifiers::CONTROL), Outcome::Stay);
        assert!(tab.note.as_ref().is_some_and(|(text, _, _)| text.contains("again leaves")));
        assert_eq!(press(&mut tab, KeyCode::Char('q'), KeyModifiers::CONTROL), Outcome::Stay, "asked, not dropped");
        assert!(label().is_some(), "still open");
        // Closing with unsaved changes asks; again drops them.
        assert!(tab.note.as_ref().is_some_and(|(text, _, _)| text.contains("again drops")));
        assert!(tab.note.as_ref().is_some_and(|(text, _, _)| text.contains("again drops")));
        assert_eq!(press(&mut tab, KeyCode::Char('q'), KeyModifiers::CONTROL), Outcome::Closed);
        assert_eq!(label(), None);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "theirs a\nb!?\nC\n", "dropped, not saved");
    }

    #[test]
    fn a_clean_file_takes_in_a_change_made_on_the_machine() {
        let _lock = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_test();
        let folder = Folder::new("editor-reload");
        let dir = folder.0.clone();
        let file = format!("{dir}/notes.txt");
        let stamp = shared_files::write(&dir, "notes.txt", b"one\ntwo\n", &Expect::New, &mut crate::shared_files::tests::local).unwrap();
        open(&dir, "notes.txt", "one\ntwo\n", Some(stamp));
        let mut tab = Open::new(super::super::View::Edit);
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&file, "one\nTWO\nthree\n").unwrap();
        editor().docs[0].watched = Instant::now() - WATCH_EVERY;
        // Watched, then read, then taken in.
        for _ in 0..3 {
            settled(&mut tab);
        }
        assert_eq!(text_for_test().as_deref(), Some("one\nTWO\nthree\n"));
        let rows = settled(&mut tab);
        assert!(rows.iter().any(|row| row.contains("2 ▎ TWO")) && rows.iter().any(|row| row.contains("3 ▎ three")), "{rows:?}");
        assert_eq!(label().unwrap().2, false, "still nothing unsaved");
        reset_for_test();
    }

    #[test]
    fn the_view_wraps_scrolls_and_places_the_cursor() {
        let _lock = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_test();
        let long = "word ".repeat(40);
        let text = format!("short\n{long}\n{}", (0..60).map(|n| format!("line {n}\n")).collect::<String>());
        open("/tmp/none", "wrap.md", &text, Some("1:1:1:1".into()));
        let mut tab = Open::new(super::super::View::Edit);
        let (rows, _, cursor) = body(&mut tab, 80, 20);
        let rows: Vec<String> = rows.iter().map(|row| row.iter().map(|(_, text)| strip(text)).collect()).collect();
        assert!(rows[0].starts_with("  1   short"), "{:?}", rows[0]);
        assert!(rows[2].starts_with("      ") && rows[2].contains("word"), "prose wraps under the gutter: {:?}", rows[2]);
        assert_eq!(cursor, Some((6, 0)));
        press(&mut tab, KeyCode::End, KeyModifiers::CONTROL);
        let (rows, _, cursor) = body(&mut tab, 80, 20);
        let shown: Vec<String> = rows.iter().map(|row| row.iter().map(|(_, text)| strip(text)).collect()).collect();
        assert!(shown.iter().any(|row| row.contains("line 59")), "scrolled to the end: {shown:?}");
        assert!(cursor.is_some_and(|(_, row)| row < 20));
        // Alt+Z: no wrap; a long line scrolls sideways to the cursor.
        press(&mut tab, KeyCode::Home, KeyModifiers::CONTROL);
        press(&mut tab, KeyCode::Char('z'), KeyModifiers::ALT);
        press(&mut tab, KeyCode::Down, KeyModifiers::NONE);
        press(&mut tab, KeyCode::End, KeyModifiers::NONE);
        let (rows, _, cursor) = body(&mut tab, 80, 20);
        let shown: Vec<String> = rows.iter().map(|row| row.iter().map(|(_, text)| strip(text)).collect()).collect();
        assert!(shown.iter().all(|row| super::super::text_width(row) <= 80), "nothing wider than the tab");
        // Scrolled up from the end just far enough: line 2 on top.
        assert!(shown[0].starts_with("  2 ") && shown[0].contains("word"), "{:?}", shown[0]);
        assert!(cursor.is_some_and(|(column, row)| row == 0 && column > 6 && column < 80), "{cursor:?}");
        // Ctrl+P twice: the split view, the markdown rendered beside it, or
        // instead (once is the live preview, in place).
        press(&mut tab, KeyCode::Char('p'), KeyModifiers::CONTROL);
        press(&mut tab, KeyCode::Char('p'), KeyModifiers::CONTROL);
        let (rows, footer, _) = body(&mut tab, 140, 20);
        assert!(rows.iter().any(|row| row.iter().any(|(_, text)| text.contains('│'))), "side by side at 140 columns");
        assert!(footer.hints.contains(&("^p", "text")));
        reset_for_test();
    }

    fn stamp_of(dir: &str) -> String {
        shared_files::stamp_of(dir, "plan.md", &mut crate::shared_files::tests::local).unwrap().unwrap()
    }
}
