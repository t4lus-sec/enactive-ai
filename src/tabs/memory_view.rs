//! The memory tab (Alt+M), a submodule of `tabs.rs`: what the agent remembers,
//! read where it is kept, and struck out of all of it.
//!
//! Four sections, Tab between them:
//!
//! - **archive** -- every event, newest first; `/` searches it (every word
//!   must appear; `since:` `until:` `speaker:` `kind:` and `#id` narrow it).
//!   The strip shows the event whole, what cites it, and `[` `]` step to the
//!   events beside it.
//! - **threads** -- grouped by state; resolved and abandoned start folded.
//! - **world** -- each model and its records; a record that needs review is
//!   teal. The strip has the record's statement, basis, conditions,
//!   alternatives and evidence (Enter on an evidence line goes to the event),
//!   and `h` its history. `p` pins a fact into the agent's context (from its next
//!   start or sleep), or unpins it; one sleep pinned from use is marked too.
//! - **redactions** -- what the agent asked to have removed, with the change as a
//!   diff (`a` approve, `e` edit, `d` decline), and what was struck before.
//!
//! # Marking text
//!
//! Enter on a row puts the cursor in the strip. The arrows move it; with
//! Shift they mark text (red). Enter keeps what is marked -- from as many
//! places as wanted -- and `a` shows every place in every store where it is
//! kept, and `a` again strikes it from all of them (`strike.rs`). The
//! operator approves his own redactions by making them: nothing waits.

use super::{Action, Footer, Open, Outcome, act, clip, text_width};
use crate::hippocampus::{Hippocampus, RecalledEvent, Speaker};
use crate::style::{Line, Style};
use crate::strike::{EXCERPT_CLOSE, EXCERPT_OPEN, Impact};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;
use unicode_width::UnicodeWidthChar;

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The running agent's archive handle, so a redaction from this tab is
/// written under the same lock as its appends (`tabs::init_memory`).
static MEMORY: OnceLock<Hippocampus> = OnceLock::new();

pub fn init(memory: &Hippocampus) {
    let _ = MEMORY.set(memory.clone());
}

/// The archive: the running agent's handle, or (on an attached terminal) one
/// that only reads.
fn memory() -> Option<Hippocampus> {
    MEMORY.get().cloned().or_else(|| Hippocampus::open_read_only(super::ARCHIVE.get()?).ok())
}

/// The running agent's own handle, for a write: never a second one, whose
/// lock would not keep its appends out of a rewrite.
pub(super) fn writer() -> Result<Hippocampus, String> {
    MEMORY.get().cloned().ok_or_else(|| "redactions are written by the running s0ma, which is not this process".into())
}

fn archive_path() -> Option<PathBuf> {
    MEMORY.get().map(|memory| memory.archive_path().to_path_buf()).or_else(|| super::ARCHIVE.get().cloned())
}

/// Where the next opening of the tab starts (`/redact`).
static START: Mutex<Option<(Section, Option<u64>)>> = Mutex::new(None);

pub fn start_at(section: Section, request: Option<u64>) {
    if let Ok(mut start) = START.lock() {
        *start = Some((section, request));
    }
}

pub(super) fn take_start(state: &mut State) {
    if let Some((section, request)) = START.lock().ok().and_then(|mut start| start.take()) {
        state.section = section;
        state.focus = Focus::List;
        state.review = None;
        if let Some(request) = request {
            state.want_request = Some(request);
        }
    }
}

/// How often the lists are read again while the tab is up.
const RELOAD: Duration = Duration::from_secs(5);
/// The archive list, newest first, at most this long.
const EVENTS: usize = 300;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Section {
    #[default]
    Archive,
    Threads,
    World,
    /// The agent's open loops (and those closed), which the operator can close.
    Loops,
    Redactions,
}

const SECTIONS: [Section; 5] = [Section::Archive, Section::Threads, Section::World, Section::Loops, Section::Redactions];

impl Section {
    fn name(self) -> &'static str {
        match self {
            Section::Archive => "archive",
            Section::Threads => "threads",
            Section::World => "world",
            Section::Loops => "loops",
            Section::Redactions => "redactions",
        }
    }

    fn index(self) -> usize {
        SECTIONS.iter().position(|section| *section == self).unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Focus {
    #[default]
    List,
    Strip,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Purpose {
    Search,
    Decline(u64),
    /// Closing a loop: done (true) or dropped, and what the operator says of it.
    CloseLoop(String, bool),
}

struct Input {
    purpose: Purpose,
    text: Vec<char>,
    cursor: usize,
}

/// What a row of a list is.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    Event(u64),
    Group(String),
    Thread(u64),
    Model(String),
    Record(String),
    Request(u64),
    Loop(String),
}

impl Kind {
    fn key(&self) -> String {
        format!("{self:?}")
    }
}

struct Row {
    kind: Kind,
    line: Line,
}

/// One line of the strip: its text in pieces with their styles, whether it is
/// the store's own words (and so can be marked), and the event it leads to.
#[derive(Clone, Debug, Default)]
struct DocLine {
    spans: Vec<(Style, String)>,
    words: bool,
    link: Option<u64>,
}

impl DocLine {
    fn text(&self) -> String {
        self.spans.iter().map(|(_, text)| text.as_str()).collect()
    }
    fn label(style: Style, text: impl Into<String>) -> Self {
        Self { spans: vec![(style, text.into())], words: false, link: None }
    }
    fn words(style: Style, text: impl Into<String>) -> Self {
        Self { spans: vec![(style, text.into())], words: true, link: None }
    }
    fn blank() -> Self {
        Self::default()
    }
}

/// The store's text, a line each, marked as the store's words.
fn text_lines(out: &mut Vec<DocLine>, style: Style, text: &str) {
    for line in text.split('\n') {
        out.push(DocLine::words(style, line));
    }
}

#[derive(Clone, Debug)]
enum Reviewing {
    Strike(Vec<String>),
    Approve(u64),
}

struct Review {
    what: Reviewing,
    impact: Result<Impact, String>,
    /// Pieces of an approval too short to look for elsewhere.
    short: usize,
    scroll: usize,
}

/// What the lists are read from, every `RELOAD`.
#[derive(Default)]
struct Data {
    threads: Vec<crate::threads::Thread>,
    models: Vec<(String, String, String)>,
    /// id → (model, body, needs review, revision).
    records: BTreeMap<String, (String, Value, bool, u64)>,
    requests: Vec<crate::redact::Request>,
    loops: Vec<crate::open_loops::OpenLoop>,
    /// The loops' odometer now, for how long since each was touched.
    loop_distance: u64,
    /// Pinned facts, and who pinned each (`operator` or `sleep`).
    pins: BTreeMap<String, String>,
}

#[derive(Default)]
pub struct State {
    pub section: Section,
    focus: Focus,
    input: Option<Input>,
    /// What the archive list was searched for (empty: the latest events).
    query: String,
    events: Option<Result<Vec<RecalledEvent>, String>>,
    /// An event shown in the strip in place of the selected row: one stepped
    /// to with `[` `]`, or gone to from a record's evidence.
    shown_event: Option<u64>,
    selected: [Option<String>; 5],
    top: [usize; 5],
    /// Groups and models folded or unfolded by hand (against their default).
    toggled: HashSet<String>,
    data: Option<(Instant, Data)>,
    /// The strip: what it shows (by key), its text, where the cursor is and
    /// where marking began, and its first row.
    doc_for: Option<String>,
    doc: Vec<DocLine>,
    /// Bumped each time `doc` is built, so its layout is laid out again.
    doc_generation: u64,
    /// The strip's text laid out for a width (`Layout`), kept between frames.
    layout: Option<Layout>,
    cursor: (usize, usize),
    anchor: Option<(usize, usize)>,
    strip_top: usize,
    /// As last drawn: how many rows the list had, and the body row the strip
    /// starts on (the wheel scrolls one or the other by where it is turned).
    list_rows: usize,
    strip_at: Option<usize>,
    /// Text kept to be struck, from anywhere.
    marks: Vec<String>,
    review: Option<Review>,
    world_history: bool,
    want_request: Option<u64>,
    /// The events the weakness log cites, read once (it is hash-chained and
    /// read whole) until something is struck.
    weakness_cites: Option<HashSet<u64>>,
    /// The strip's text is out of date (something was struck): built again
    /// where it is, the cursor and the scroll kept.
    doc_stale: bool,
    /// What else cites the event shown, looked up on a thread of its own:
    /// for the text of this generation, from this line on, after these.
    citing: Option<Citing>,
    /// A review or a strike running on a thread of its own, so the tab keeps
    /// drawing and taking keys meanwhile.
    working: Option<Working>,
}

/// What the stores say about an event, and the weakness log's citations.
type Answered = (Vec<String>, HashSet<u64>);

/// The stores' citations of an event, on their way.
struct Citing {
    generation: u64,
    from_line: usize,
    inline: Vec<String>,
    /// What the stores say, and the weakness log's citations (read whole,
    /// so kept for the next event).
    answer: std::sync::Arc<Mutex<Option<Answered>>>,
}

/// What a background review or strike is, and where its answer lands.
struct Working {
    what: Reviewing,
    /// True for the strike (or approval) itself, false for its review.
    writing: bool,
    started: Instant,
    answer: std::sync::Arc<Mutex<Option<Result<String, String>>>>,
}

// ------------------------------------------------------------------ reading

fn open_db(path: &Path) -> Option<Connection> {
    path.exists()
        .then(|| Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX).ok())
        .flatten()
}

fn load(archive: &Path) -> Data {
    let mut data = Data::default();
    if let Ok(text) = std::fs::read_to_string(archive.with_file_name("threads.json")) {
        data.threads = serde_json::from_str(&text).unwrap_or_default();
    }
    if let Some(db) = open_db(&archive.with_file_name("world_model.sqlite3")) {
        if let Ok(mut statement) = db.prepare("SELECT id, title, summary FROM wm_models ORDER BY title") {
            data.models = statement
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                .map(|rows| rows.flatten().collect())
                .unwrap_or_default();
        }
        if let Ok(mut statement) = db.prepare("SELECT id, model_id, body, needs_review, revision FROM wm_records ORDER BY id") {
            data.records = statement
                .query_map([], |row| {
                    let body: String = row.get(2)?;
                    Ok((
                        row.get::<_, String>(0)?,
                        (
                            row.get::<_, String>(1)?,
                            serde_json::from_str(&body).unwrap_or(Value::Null),
                            row.get::<_, i64>(3)? != 0,
                            row.get::<_, i64>(4)?.max(0) as u64,
                        ),
                    ))
                })
                .map(|rows| rows.flatten().collect())
                .unwrap_or_default();
        }
        // Absent until the first pin or use: nothing pinned.
        if let Ok(mut statement) = db.prepare("SELECT record_id, pinned_by FROM fact_pins") {
            data.pins = statement
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .map(|rows| rows.flatten().collect())
                .unwrap_or_default();
        }
    }
    data.requests = crate::redact::Book::beside_archive(archive).all().unwrap_or_default();
    let loops = crate::open_loops::LoopStore::beside_archive(archive);
    data.loops = loops.all().unwrap_or_default();
    data.loop_distance = loops.distance().unwrap_or_default();
    data
}

/// The archive's latest events, or those `query` finds, newest first.
fn search(memory: &Hippocampus, query: &str) -> Result<Vec<RecalledEvent>, String> {
    let mut terms: Vec<String> = Vec::new();
    let (mut since, mut until, mut speaker, mut kind, mut id) = (None, None, None, None, None);
    let mut rest = query.trim();
    while !rest.is_empty() {
        let (word, after) = if let Some(quoted) = rest.strip_prefix('"') {
            match quoted.find('"') {
                Some(end) => (&quoted[..end], &quoted[end + 1..]),
                None => (quoted, ""),
            }
        } else {
            match rest.find(char::is_whitespace) {
                Some(end) => (&rest[..end], &rest[end..]),
                None => (rest, ""),
            }
        };
        rest = after.trim_start();
        if let Some(value) = word.strip_prefix("since:") {
            since = Some(value.to_string());
        } else if let Some(value) = word.strip_prefix("until:") {
            until = Some(value.to_string());
        } else if let Some(value) = word.strip_prefix("speaker:") {
            speaker = Some(match value.to_lowercase().as_str() {
                "operator" | "you" | "me" => "operator".to_string(),
                _ => "s0ma".to_string(),
            });
        } else if let Some(value) = word.strip_prefix("kind:") {
            kind = Some(value.to_lowercase());
        } else if let Some(n) = word.strip_prefix('#').and_then(|n| n.parse::<u64>().ok()) {
            id = Some(n);
        } else if !word.is_empty() {
            terms.push(word.to_lowercase());
        }
    }
    if let Some(id) = id {
        return memory.get_event(id).map(|event| event.into_iter().collect()).map_err(|error| error.to_string());
    }
    let window = crate::archive::rank::TimeWindow::parse(since.as_deref(), until.as_deref(), crate::clock::now())?;
    // Nothing to narrow by: the latest, read by id through the line index
    // rather than the whole archive parsed to keep its end.
    if terms.is_empty() && speaker.is_none() && kind.is_none() && window.is_none() {
        let last = memory.last_event_id();
        let ids: Vec<u64> = (last.saturating_sub(EVENTS as u64 - 1).max(1)..=last).collect();
        let mut events = memory.get_events(&ids).map_err(|error| error.to_string())?;
        events.sort_by_key(|event| std::cmp::Reverse(event.id));
        return Ok(events);
    }
    // A line that does not hold every word, as JSON writes it, is not parsed
    // at all. (A word JSON would escape is left to the check after parsing.)
    let plain: Vec<&String> = terms.iter().filter(|term| !term.contains(['"', '\\']) && !term.chars().any(char::is_control)).collect();
    let prefilter = |line: &str| {
        if plain.is_empty() {
            return true;
        }
        let lower = line.to_lowercase();
        plain.iter().all(|term| lower.contains(term.as_str()))
    };
    let mut found: std::collections::VecDeque<RecalledEvent> = std::collections::VecDeque::new();
    memory
        .each_event_where(prefilter, |event| {
            let who = match event.speaker {
                Speaker::Operator => "operator",
                Speaker::S0ma => "s0ma",
            };
            if speaker.as_deref().is_some_and(|speaker| speaker != who) {
                return;
            }
            if let Some(kind) = &kind
                && !crate::archive::source_kind(who, &event.content).contains(kind.as_str())
            {
                return;
            }
            if let Some(window) = window
                && !crate::clock::parse(&event.timestamp).is_some_and(|at| window.contains(at.timestamp()))
            {
                return;
            }
            if !terms.is_empty() {
                let lower = event.content.to_lowercase();
                if !terms.iter().all(|term| lower.contains(term.as_str())) {
                    return;
                }
            }
            if found.len() == EVENTS {
                found.pop_front();
            }
            found.push_back(event);
        })
        .map_err(|error| error.to_string())?;
    Ok(found.into_iter().rev().collect())
}

/// The "cited by" block: `waiting` while the stores are still looked in.
fn cited_lines(out: &mut Vec<DocLine>, cited: &[String], waiting: bool) {
    if cited.is_empty() && !waiting {
        out.push(DocLine::label(Style::Dim, "cited by nothing"));
        return;
    }
    out.push(DocLine::label(Style::Bold, "cited by"));
    for what in cited {
        out.push(DocLine::label(Style::Dim, format!("  {what}")));
    }
    if waiting {
        out.push(DocLine::label(Style::Work, "  looking in the board, open loops and episodes…"));
    }
}

/// The stores' citations, once in: the block finished, if the strip still
/// shows the event they are for.
fn settle_citing(state: &mut State) {
    let done = state.citing.as_ref().and_then(|citing| citing.answer.lock().ok().and_then(|mut answer| answer.take()));
    let Some((found, weakness)) = done else { return };
    let Some(citing) = state.citing.take() else { return };
    state.weakness_cites.get_or_insert(weakness);
    if citing.generation != state.doc_generation || citing.from_line > state.doc.len() {
        return;
    }
    state.doc.truncate(citing.from_line);
    let mut all = citing.inline;
    all.extend(found);
    cited_lines(&mut state.doc, &all, false);
    // Laid out again; the cursor and the scroll stay where they are.
    state.doc_generation += 1;
}

/// What cites `id` among what the tab has in hand: the world's records and
/// the threads.
fn cited_inline(data: &Data, id: u64) -> Vec<String> {
    let mut out = Vec::new();
    for (record, (_, body, _, _)) in &data.records {
        let cites = |key: &str| body[key].as_array().is_some_and(|ids| ids.iter().any(|v| v.as_u64() == Some(id)));
        if cites("supporting_evidence_ids") {
            out.push(format!("world record {record} (supports)"));
        } else if cites("counterevidence_ids") {
            out.push(format!("world record {record} (against)"));
        }
    }
    for thread in &data.threads {
        if thread.event_ids.contains(&id) {
            out.push(format!("thread {} — {}", thread.id, clip(&thread.title, 50)));
        }
    }
    out
}

/// What cites `id` in the stores read from disk (the board, the open loops,
/// the episodes, the weakness log): a thread's work.
fn cited_stored(archive: &Path, weakness_cites: &HashSet<u64>, id: u64) -> Vec<String> {
    let mut out = Vec::new();
    let ids = |path: &Path, sql: &str| -> Vec<String> {
        open_db(path)
            .and_then(|db| {
                let mut statement = db.prepare(sql).ok()?;
                let rows = statement.query_map([id as i64], |row| row.get::<_, String>(0)).ok()?;
                Some(rows.flatten().collect())
            })
            .unwrap_or_default()
    };
    for fact in ids(
        &archive.with_file_name("board.sqlite3"),
        "SELECT f.id FROM facts f, json_each(f.evidence_event_ids) j WHERE j.value = ?1",
    ) {
        out.push(format!("board finding {}", super::board_view::short(&fact)));
    }
    for looped in ids(
        &archive.with_file_name("open_loops.sqlite3"),
        "SELECT l.id FROM loops l, json_each(l.source_event_ids) j WHERE j.value = ?1",
    ) {
        out.push(format!("open loop {looped}"));
    }
    for episode in ids(
        &archive.with_extension("semantic.sqlite3"),
        "SELECT e.id FROM consolidated_episodes e, json_each(e.source_digests_json) j WHERE j.key = CAST(?1 AS TEXT)",
    ) {
        out.push(format!("episode {}", clip(&episode, 14)));
    }
    if weakness_cites.contains(&id) {
        out.push("the weakness log (hash-chained: a redaction cannot reach it)".into());
    }
    out
}

// ------------------------------------------------------------------ the lists

fn speaker_style(event: &RecalledEvent) -> (Style, &'static str) {
    match event.speaker {
        Speaker::Operator => (Style::Operator, "you"),
        Speaker::S0ma if crate::hippocampus::is_tool_result(event) => (Style::Dim, "tool"),
        Speaker::S0ma => (Style::Accent, "s0ma"),
    }
}

fn when(timestamp: &str) -> String {
    crate::clock::parse(timestamp).map_or_else(|| clip(timestamp, 16), |at| crate::clock::short(&at))
}

/// The start of a text, enough for one row.
fn preview(text: &str) -> &str {
    match text.char_indices().nth(400) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

fn flat(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn state_name(state: crate::threads::ThreadState) -> &'static str {
    use crate::threads::ThreadState as S;
    match state {
        S::Active => "active",
        S::Candidate => "candidate",
        S::Dormant => "dormant",
        S::Transformed => "transformed",
        S::Resolved => "resolved",
        S::Abandoned => "abandoned",
    }
}

const THREAD_ORDER: [&str; 6] = ["active", "candidate", "dormant", "transformed", "resolved", "abandoned"];

impl State {
    fn data(&mut self) -> &Data {
        let stale = self.data.as_ref().is_none_or(|(at, _)| at.elapsed() >= RELOAD);
        if stale {
            let data = archive_path().map(|archive| load(&archive)).unwrap_or_default();
            self.data = Some((Instant::now(), data));
        }
        &self.data.as_ref().expect("loaded above").1
    }

    fn reload(&mut self) {
        self.data = None;
        // Built again where it is: the cursor and scroll stay.
        self.doc_stale = true;
        self.weakness_cites = None;
        // The archive list too, whichever section struck: its lines quote
        // the events.
        self.events = None;
    }

    /// Whether a group or model is unfolded: by default, hand-toggled otherwise.
    fn is_open(&self, key: &str, default: bool) -> bool {
        default != self.toggled.contains(key)
    }

    fn toggle(&mut self, key: &str) {
        if !self.toggled.remove(key) {
            self.toggled.insert(key.to_string());
        }
    }

    fn rows(&mut self, width: usize) -> Vec<Row> {
        match self.section {
            Section::Archive => {
                if self.events.is_none() {
                    self.events = Some(match memory() {
                        Some(memory) => search(&memory, &self.query),
                        None => Err("the archive is not available in this process".into()),
                    });
                }
                let Some(Ok(events)) = &self.events else { return Vec::new() };
                events
                    .iter()
                    .map(|event| {
                        let (style, who) = speaker_style(event);
                        let head = format!("  #{:<6} {:<17} ", event.id, when(&event.timestamp));
                        let used = text_width(&head) + 6;
                        Row {
                            kind: Kind::Event(event.id),
                            line: vec![
                                (Style::Dim, head),
                                (style, format!("{who:<5} ")),
                                // Only as much as a row shows is flattened, not a
                                // tool result's megabyte.
                                (Style::Plain, clip(&flat(preview(&event.content)), width.saturating_sub(used + 1))),
                            ],
                        }
                    })
                    .collect()
            }
            Section::Threads => {
                let threads = self.data().threads.clone();
                let mut out = Vec::new();
                for group in THREAD_ORDER {
                    let members: Vec<&crate::threads::Thread> =
                        threads.iter().filter(|thread| state_name(thread.state) == group).collect();
                    if members.is_empty() {
                        continue;
                    }
                    let key = format!("threads:{group}");
                    let open = self.is_open(&key, !matches!(group, "resolved" | "abandoned"));
                    out.push(Row {
                        kind: Kind::Group(key),
                        line: vec![
                            (Style::Dim, format!("  {} ", if open { "▾" } else { "▸" })),
                            (Style::Bold, group.to_string()),
                            (Style::Dim, format!("  {}", members.len())),
                        ],
                    });
                    if open {
                        for thread in members {
                            let head = format!("     {:<4} ", thread.id);
                            out.push(Row {
                                kind: Kind::Thread(thread.id),
                                line: vec![
                                    (Style::Dim, head.clone()),
                                    (Style::Plain, clip(&flat(&thread.title), width.saturating_sub(text_width(&head) + 1))),
                                ],
                            });
                        }
                    }
                }
                out
            }
            Section::World => {
                let data = self.data();
                let models = data.models.clone();
                let pins = data.pins.clone();
                let records: Vec<(String, String, String, bool)> = data
                    .records
                    .iter()
                    .map(|(id, (model, body, review, _))| {
                        (id.clone(), model.clone(), body["title"].as_str().unwrap_or_default().to_string(), *review)
                    })
                    .collect();
                let mut out = Vec::new();
                for (model, title, _) in &models {
                    let mine: Vec<&(String, String, String, bool)> = records.iter().filter(|r| &r.1 == model).collect();
                    let review = mine.iter().filter(|r| r.3).count();
                    let key = format!("world:{model}");
                    let open = self.is_open(&key, models.len() <= 3);
                    let mut line = vec![
                        (Style::Dim, format!("  {} ", if open { "▾" } else { "▸" })),
                        (Style::Bold, clip(title, width.saturating_sub(30).max(10))),
                        (Style::Dim, format!("  {} records", mine.len())),
                    ];
                    if review > 0 {
                        line.push((Style::Warn, format!(" · {review} to review")));
                    }
                    out.push(Row { kind: Kind::Model(model.clone()), line });
                    if open {
                        for (id, _, title, review) in mine {
                            let head = format!("     {} ", clip(id, 24));
                            let style = if *review { Style::Warn } else { Style::Plain };
                            let pin = match pins.get(id).map(String::as_str) {
                                Some("operator") => "  pinned",
                                Some(_) => "  pinned by sleep",
                                None => "",
                            };
                            out.push(Row {
                                kind: Kind::Record(id.clone()),
                                line: vec![
                                    (Style::Dim, head.clone()),
                                    (style, clip(title, width.saturating_sub(text_width(&head) + pin.len() + 1))),
                                    (Style::Success, pin.to_string()),
                                ],
                            });
                        }
                    }
                }
                out
            }
            Section::Loops => {
                let distance = self.data().loop_distance;
                let mut loops = self.data().loops.clone();
                // Open first, most recently touched first; then the closed.
                loops.sort_by_key(|entry| {
                    (entry.status != crate::open_loops::LoopStatus::Open, std::cmp::Reverse(entry.last_touched_distance))
                });
                loops
                    .iter()
                    .map(|entry| {
                        let open = entry.status == crate::open_loops::LoopStatus::Open;
                        let (style, state) = match entry.status {
                            crate::open_loops::LoopStatus::Open if entry.is_stale(distance) => (Style::Warn, "stale"),
                            crate::open_loops::LoopStatus::Open => (Style::Success, "open"),
                            crate::open_loops::LoopStatus::Done => (Style::Dim, "done"),
                            crate::open_loops::LoopStatus::Dropped => (Style::Dim, "dropped"),
                        };
                        let head = format!("  {:<11}", clip(&entry.id, 10));
                        let state = format!("{state:<8}{:<9}", entry.kind.label());
                        let room = width.saturating_sub(text_width(&head) + text_width(&state) + 1);
                        Row {
                            kind: Kind::Loop(entry.id.clone()),
                            line: vec![
                                (Style::Dim, head),
                                (style, state),
                                (if open { Style::Plain } else { Style::Dim }, clip(entry.first_line(), room)),
                            ],
                        }
                    })
                    .collect()
            }
            Section::Redactions => {
                let mut requests = self.data().requests.clone();
                // Waiting first, then the rest newest first.
                requests.sort_by_key(|request| (request.state != crate::redact::State::Pending, std::cmp::Reverse(request.request)));
                requests
                    .iter()
                    .map(|request| {
                        let (style, state) = match request.state {
                            crate::redact::State::Pending => (Style::Warn, "waiting"),
                            crate::redact::State::Approved if request.author == "operator" => (Style::Danger, "struck"),
                            crate::redact::State::Approved => (Style::Success, "approved"),
                            crate::redact::State::Rejected => (Style::Dim, "declined"),
                            crate::redact::State::Stale => (Style::Dim, "stale"),
                        };
                        let what = if request.author == "operator" {
                            format!("you · {} places", request.places.len())
                        } else {
                            format!("s0ma · event #{}", request.event_id)
                        };
                        let head = format!("  #{:<4} ", request.request);
                        let used = text_width(&head) + 10 + text_width(&what) + 4;
                        Row {
                            kind: Kind::Request(request.request),
                            line: vec![
                                (Style::Dim, head),
                                (style, format!("{state:<9} ")),
                                (Style::Plain, what),
                                (Style::Dim, format!("  {}", clip(&flat(&request.reason), width.saturating_sub(used)))),
                            ],
                        }
                    })
                    .collect()
            }
        }
    }

    // -------------------------------------------------------------- the strip

    /// Which text the strip shows for `kind`: the event stepped to, or the row.
    fn doc_key(&self, kind: Option<&Kind>) -> String {
        match (self.section, self.shown_event, kind) {
            (Section::Archive, Some(id), _) => format!("event:{id}"),
            (_, _, Some(kind)) => format!("{}:{}", kind.key(), self.world_history),
            _ => String::new(),
        }
    }

    /// The strip's text for `kind` (or the event stepped to).
    fn build_doc(&mut self, kind: Option<&Kind>) -> Vec<DocLine> {
        if self.section == Section::Archive
            && let Some(id) = self.shown_event
        {
            return self.event_doc(id);
        }
        match kind {
            None => Vec::new(),
            Some(Kind::Event(id)) => self.event_doc(*id),
            Some(Kind::Group(key)) => {
                vec![DocLine::label(Style::Dim, format!("{} · ⏎ folds", key.rsplit(':').next().unwrap_or(key)))]
            }
            Some(Kind::Thread(id)) => self.thread_doc(*id),
            Some(Kind::Model(id)) => self.model_doc(id),
            Some(Kind::Record(id)) => self.record_doc(id),
            Some(Kind::Request(n)) => self.request_doc(*n),
            Some(Kind::Loop(id)) => self.loop_doc(id),
        }
    }

    fn event_doc(&mut self, id: u64) -> Vec<DocLine> {
        let Some(memory) = memory() else { return vec![DocLine::label(Style::Problem, "the archive is not available")] };
        let event = match memory.get_event(id) {
            Ok(Some(event)) => event,
            Ok(None) => return vec![DocLine::label(Style::Problem, format!("there is no event #{id}"))],
            Err(error) => return vec![DocLine::label(Style::Problem, error.to_string())],
        };
        let (style, who) = speaker_style(&event);
        let mut out = vec![DocLine {
            spans: vec![
                (Style::Heading, format!("event #{id}")),
                (Style::Dim, format!(" · {} · ", when(&event.timestamp))),
                (style, who.to_string()),
            ],
            words: false,
            link: None,
        }];
        out.push(DocLine::blank());
        text_lines(&mut out, Style::Plain, &event.content);
        out.push(DocLine::blank());
        let archive = memory.archive_path().to_path_buf();
        // What the tab has in hand now; the stores on disk on a thread, so
        // walking the archive with the arrows never waits on them
        // (`settle_citing`).
        let inline = cited_inline(self.data(), id);
        let from_line = out.len();
        cited_lines(&mut out, &inline, true);
        let answer = std::sync::Arc::new(Mutex::new(None));
        let slot = answer.clone();
        let known = self.weakness_cites.clone();
        let spawned = std::thread::Builder::new().name("memory-cites".into()).spawn(move || {
            let weakness = known.unwrap_or_else(|| crate::weaknesses::cited_events(&memory));
            let found = cited_stored(&archive, &weakness, id);
            if let Ok(mut slot) = slot.lock() {
                *slot = Some((found, weakness));
            }
            super::poke();
        });
        self.citing = spawned.ok().map(|_| Citing { generation: self.doc_generation + 1, from_line, inline, answer });
        out
    }

    fn thread_doc(&mut self, id: u64) -> Vec<DocLine> {
        let Some(thread) = self.data().threads.iter().find(|thread| thread.id == id).cloned() else {
            return vec![DocLine::label(Style::Dim, "that thread is gone")];
        };
        let mut out = Vec::new();
        text_lines(&mut out, Style::Bold, &thread.title);
        out.push(DocLine::label(
            Style::Dim,
            format!("thread {} · {} · {:?} · updated {}", thread.id, state_name(thread.state), thread.category, when(&thread.updated_at)),
        ));
        for (label, text) in [("description", &thread.description), ("why it continues", &thread.significance)] {
            if !text.trim().is_empty() {
                out.push(DocLine::blank());
                out.push(DocLine::label(Style::Heading, label));
                text_lines(&mut out, Style::Plain, text);
            }
        }
        for (label, items) in [("questions", &thread.questions), ("next", &thread.next_actions)] {
            if !items.is_empty() {
                out.push(DocLine::blank());
                out.push(DocLine::label(Style::Heading, label));
                for item in items {
                    text_lines(&mut out, Style::Plain, item);
                }
            }
        }
        if !thread.event_ids.is_empty() {
            out.push(DocLine::blank());
            out.push(DocLine::label(Style::Heading, "events"));
            for event in &thread.event_ids {
                out.push(DocLine { link: Some(*event), ..DocLine::label(Style::Cursor, format!("  → event #{event}")) });
            }
        }
        out
    }

    fn model_doc(&mut self, id: &str) -> Vec<DocLine> {
        let Some((_, title, summary)) = self.data().models.iter().find(|model| model.0 == id).cloned() else {
            return vec![DocLine::label(Style::Dim, "that model is gone")];
        };
        let mut out = Vec::new();
        text_lines(&mut out, Style::Bold, &title);
        out.push(DocLine::label(Style::Dim, format!("model {id}")));
        out.push(DocLine::blank());
        text_lines(&mut out, Style::Plain, &summary);
        out
    }

    fn record_doc(&mut self, id: &str) -> Vec<DocLine> {
        let Some((model, body, review, revision)) = self.data().records.get(id).cloned() else {
            return vec![DocLine::label(Style::Dim, "that record is gone")];
        };
        let mut out = Vec::new();
        text_lines(&mut out, Style::Bold, body["title"].as_str().unwrap_or(id));
        let mut head = vec![(
            Style::Dim,
            format!(
                "record {id} · {model} · {} · {} · revision {revision}",
                body["kind"].as_str().unwrap_or("?"),
                body["status"].as_str().unwrap_or("?")
            ),
        )];
        if review {
            head.push((Style::Warn, " · needs review".into()));
        }
        match self.data().pins.get(id).map(String::as_str) {
            Some("operator") => head.push((Style::Success, " · pinned".into())),
            Some(_) => head.push((Style::Success, " · pinned by sleep (from use)".into())),
            None => {}
        }
        if let Some(level) = body["certainty"].as_str() {
            head.push((Style::Dim, format!(" · {level}")));
        }
        out.push(DocLine { spans: head, words: false, link: None });
        if let Some(reason) = body["certainty_reason"].as_str().filter(|r| !r.is_empty()) {
            out.push(DocLine::label(Style::Dim, format!("how sure: {reason}")));
        }
        let field = |out: &mut Vec<DocLine>, label: &str, value: &Value| {
            let items: Vec<String> = match value {
                Value::String(text) if !text.trim().is_empty() => vec![text.clone()],
                Value::Array(items) => items.iter().filter_map(|item| item.as_str().map(str::to_string)).collect(),
                _ => Vec::new(),
            };
            if items.is_empty() {
                return;
            }
            out.push(DocLine::blank());
            out.push(DocLine::label(Style::Heading, label));
            for item in items {
                text_lines(out, Style::Plain, &item);
            }
        };
        field(&mut out, "statement", &body["statement"]);
        field(&mut out, "basis", &body["basis"]);
        field(&mut out, "applies under", &body["applies_under"]);
        field(&mut out, "alternatives", &body["alternatives"]);
        field(&mut out, "predictions", &body["predictions"]);
        let ids = |key: &str| -> Vec<u64> { body[key].as_array().map(|ids| ids.iter().filter_map(Value::as_u64).collect()).unwrap_or_default() };
        let (supporting, against) = (ids("supporting_evidence_ids"), ids("counterevidence_ids"));
        if !supporting.is_empty() || !against.is_empty() {
            out.push(DocLine::blank());
            out.push(DocLine::label(Style::Heading, "evidence"));
            for (ids, how) in [(supporting, "supports"), (against, "against")] {
                for event in ids {
                    out.push(DocLine { link: Some(event), ..DocLine::label(Style::Cursor, format!("  → event #{event}  ({how})")) });
                }
            }
        }
        if self.world_history
            && let Some(db) = archive_path().and_then(|archive| open_db(&archive.with_file_name("world_model.sqlite3")))
        {
            out.push(DocLine::blank());
            out.push(DocLine::label(Style::Heading, "history"));
            let rows: Vec<(i64, String, String, String)> = db
                .prepare("SELECT revision, body, reason, created_at FROM wm_history WHERE id=?1 ORDER BY revision DESC")
                .and_then(|mut statement| {
                    statement.query_map([id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)))?.collect()
                })
                .unwrap_or_default();
            for (revision, body, reason, at) in rows {
                let body: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
                out.push(DocLine::label(Style::Bold, format!("  revision {revision} · {}", when(&at))));
                text_lines(&mut out, Style::Dim, &reason);
                text_lines(&mut out, Style::Plain, body["statement"].as_str().unwrap_or_default());
            }
        }
        out
    }

    fn loop_doc(&mut self, id: &str) -> Vec<DocLine> {
        let distance = self.data().loop_distance;
        let Some(entry) = self.data().loops.iter().find(|entry| entry.id == id).cloned() else {
            return vec![DocLine::label(Style::Dim, "that loop is gone")];
        };
        let open = entry.status == crate::open_loops::LoopStatus::Open;
        let mut out = vec![DocLine {
            spans: vec![
                (Style::Heading, format!("loop {}", entry.id)),
                (
                    Style::Dim,
                    format!(
                        " · {} · {} · opened by {} {}",
                        entry.status.label(),
                        entry.kind.label(),
                        entry.opened_by.label(),
                        when(&entry.opened_at)
                    ),
                ),
            ],
            words: false,
            link: None,
        }];
        if open {
            let since = entry.distance_since(distance);
            out.push(DocLine::label(
                if entry.is_stale(distance) { Style::Warn } else { Style::Dim },
                format!(
                    "touched {} · {since} tokens of context ago{}",
                    when(&entry.last_touched_at),
                    if entry.is_stale(distance) { " · stale" } else { "" }
                ),
            ));
        }
        out.push(DocLine::blank());
        text_lines(&mut out, Style::Plain, &entry.text);
        if let Some(resolution) = entry.resolution.as_ref().filter(|resolution| !resolution.is_empty()) {
            out.push(DocLine::blank());
            out.push(DocLine::label(Style::Heading, if entry.status == crate::open_loops::LoopStatus::Done { "how it closed" } else { "why it was dropped" }));
            text_lines(&mut out, Style::Plain, resolution);
        }
        if !entry.source_event_ids.is_empty() {
            out.push(DocLine::blank());
            out.push(DocLine::label(Style::Heading, "from"));
            for event in &entry.source_event_ids {
                out.push(DocLine { link: Some(*event), ..DocLine::label(Style::Cursor, format!("  → event #{event}")) });
            }
        }
        if open {
            out.push(DocLine::blank());
            out.push(DocLine::label(Style::Dim, "d closes it as done · x drops it"));
        }
        out
    }

    fn request_doc(&mut self, n: u64) -> Vec<DocLine> {
        let Some(request) = self.data().requests.iter().find(|request| request.request == n).cloned() else {
            return vec![DocLine::label(Style::Dim, "that request is gone")];
        };
        let mut out = vec![DocLine {
            spans: vec![
                (Style::Heading, format!("redaction #{n}")),
                (
                    Style::Dim,
                    format!(
                        " · {} · asked by {} {}",
                        crate::redact::state_word(request.state),
                        if request.author == "operator" { "you" } else { "s0ma" },
                        when(&request.at)
                    ),
                ),
            ],
            words: false,
            link: None,
        }];
        out.push(DocLine::blank());
        out.push(DocLine::label(Style::Heading, if request.author == "operator" { "note" } else { "s0ma's reason" }));
        text_lines(&mut out, Style::Plain, &request.reason);
        if !request.note.is_empty() {
            out.push(DocLine::label(Style::Dim, format!("your note: {}", request.note)));
        }
        if request.state == crate::redact::State::Pending {
            let now = memory().and_then(|memory| memory.get_event(request.event_id).ok().flatten()).map(|event| event.content);
            out.push(DocLine::blank());
            match now {
                None => out.push(DocLine::label(Style::Problem, format!("event #{} is no longer in the archive", request.event_id))),
                Some(now) => {
                    if crate::hippocampus::content_digest(&now) != request.was {
                        out.push(DocLine::label(
                            Style::Problem,
                            "stale: decline it",
                        ));
                    }
                    if request.edited() {
                        out.push(DocLine::label(Style::Dim, "your edit"));
                    }
                    out.push(DocLine {
                        link: Some(request.event_id),
                        ..DocLine::label(Style::Heading, format!("event #{} now", request.event_id))
                    });
                    out.extend(diff_lines(&now, request.current(), false));
                    out.push(DocLine::blank());
                    out.push(DocLine::label(Style::Heading, "after"));
                    out.extend(diff_lines(&now, request.current(), true));
                }
            }
        } else if !request.places.is_empty() {
            out.push(DocLine::blank());
            out.push(DocLine::label(Style::Heading, format!("{} occurrence(s), struck from", request.struck)));
            for place in &request.places {
                out.push(DocLine {
                    link: place.event,
                    ..DocLine::label(Style::Plain, format!("  {} · {}  ×{}", place.store, place.item, place.count))
                });
            }
        }
        out
    }

    // -------------------------------------------------------------- layout

    fn selected_index(&mut self, rows: &[Row]) -> Option<usize> {
        let at = self.section.index();
        if let Some(n) = self.want_request.take() {
            self.selected[at] = Some(Kind::Request(n).key());
        }
        let index = self.selected[at]
            .as_ref()
            .and_then(|key| rows.iter().position(|row| &row.kind.key() == key))
            .or_else(|| (!rows.is_empty()).then_some(0));
        self.selected[at] = index.map(|index| rows[index].kind.key());
        index
    }
}

/// Word diff lines of `was` → `now`: the old text with what goes in red, or
/// (`after`) the new with what comes in green.
fn diff_lines(was: &str, now: &str, after: bool) -> Vec<DocLine> {
    use similar::{ChangeTag, TextDiff};
    let mut out = vec![DocLine { words: !after, ..DocLine::default() }];
    for change in TextDiff::from_words(was, now).iter_all_changes() {
        let style = match change.tag() {
            ChangeTag::Equal => Style::Plain,
            ChangeTag::Delete if !after => Style::RemovedEmph,
            ChangeTag::Insert if after => Style::AddedEmph,
            _ => continue,
        };
        let mut pieces = change.value().split('\n').peekable();
        while let Some(piece) = pieces.next() {
            if !piece.is_empty() {
                out.last_mut().expect("a line").spans.push((style, piece.to_string()));
            }
            if pieces.peek().is_some() {
                out.push(DocLine { words: !after, ..DocLine::default() });
            }
        }
    }
    out
}

/// Where `chars` breaks into rows `width` wide: at spaces, or inside a word
/// longer than a row. Character ranges, at least one.
fn wrap_ranges(chars: &[char], width: usize) -> Vec<(usize, usize)> {
    let width = width.max(1);
    if chars.is_empty() {
        return vec![(0, 0)];
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mut used = 0;
        let mut end = start;
        let mut space = None;
        while end < chars.len() {
            let w = cell(chars[end]);
            if used + w > width {
                break;
            }
            used += w;
            if chars[end] == ' ' {
                space = Some(end);
            }
            end += 1;
        }
        if end < chars.len()
            && let Some(space) = space.filter(|space| *space > start)
        {
            end = space + 1;
        }
        if end == start {
            end = start + 1;
        }
        out.push((start, end));
        start = end;
    }
    out
}

/// The columns a character of the store's text takes in the strip: a control
/// character (a tab) is drawn as one space.
fn cell(c: char) -> usize {
    if c.is_control() { 1 } else { c.width().unwrap_or(0) }
}

/// The strip's rows: each doc line wrapped, with the marked text, the text
/// being marked and the cursor drawn over it. Returns the rows and, for each,
/// its (line, start) in the doc.
/// The strip's text laid out for a width: each line's characters and their
/// styles, and each row's line and range of characters. Laid out once per
/// text and width; a frame styles only the rows it shows (`render_rows`).
pub struct Layout {
    key: (u64, usize, usize),
    chars: Vec<Vec<char>>,
    styles: Vec<Vec<Style>>,
    words: Vec<bool>,
    at: Vec<(usize, usize, usize)>,
}

fn build_layout(doc: &[DocLine], width: usize, key: (u64, usize, usize)) -> Layout {
    let mut layout = Layout { key, chars: Vec::new(), styles: Vec::new(), words: Vec::new(), at: Vec::new() };
    for (index, line) in doc.iter().enumerate() {
        let mut styles: Vec<Style> = Vec::new();
        let mut chars: Vec<char> = Vec::new();
        for (style, text) in &line.spans {
            for c in text.chars() {
                chars.push(c);
                styles.push(*style);
            }
        }
        for (start, end) in wrap_ranges(&chars, width) {
            layout.at.push((index, start, end));
        }
        layout.chars.push(chars);
        layout.styles.push(styles);
        layout.words.push(line.words);
    }
    layout
}

/// The row the cursor is on: the last row of its line starting at or
/// before it (a cursor at a row's end is on the next row, unless the line
/// ends there).
fn cursor_row(layout: &Layout, cursor: (usize, usize)) -> Option<usize> {
    let after = layout.at.partition_point(|&(line, start, _)| (line, start) <= cursor);
    after.checked_sub(1).filter(|row| layout.at[*row].0 == cursor.0)
}

/// `rows` of the layout, styled: kept marks red wherever they occur, the
/// selection, and the caret. Only the rows asked for are worked on, so a
/// long text costs what is on screen.
fn render_rows(state: &State, layout: &Layout, rows: std::ops::Range<usize>, caret: bool) -> Vec<Line> {
    let selection = state.anchor.filter(|_| caret).map(|anchor| order(anchor, state.cursor));
    let marks: Vec<Vec<char>> =
        state.marks.iter().filter(|mark| !mark.contains('\n') && !mark.is_empty()).map(|mark| mark.chars().collect()).collect();
    let mut out = Vec::new();
    for &(index, start, end) in &layout.at[rows.start.min(layout.at.len())..rows.end.min(layout.at.len())] {
        let chars = &layout.chars[index];
        let mut styles: Vec<Style> = layout.styles[index][start..end].to_vec();
        if layout.words[index] {
            for mark in &marks {
                // Matches that touch this row: looked for from a mark's
                // length before it to a mark's length after.
                let from = start.saturating_sub(mark.len() - 1);
                let to = (end + mark.len() - 1).min(chars.len());
                let mut at = from;
                while at + mark.len() <= to {
                    if chars[at..at + mark.len()] == mark[..] {
                        for i in at.max(start)..(at + mark.len()).min(end) {
                            styles[i - start] = Style::Danger;
                        }
                        at += mark.len();
                    } else {
                        at += 1;
                    }
                }
            }
        }
        if let Some(((from_line, from), (to_line, to))) = selection
            && (from_line..=to_line).contains(&index)
        {
            let first = if index == from_line { from } else { 0 };
            let last = if index == to_line { to } else { chars.len() };
            for i in first.max(start)..last.min(end) {
                styles[i - start] = Style::Marked;
            }
        }
        let mut row: Line = Vec::new();
        for i in start..end {
            let mut style = styles[i - start];
            if caret && state.cursor == (index, i) {
                style = Style::Caret;
            }
            let c = if chars[i].is_control() { ' ' } else { chars[i] };
            match row.last_mut() {
                Some((last, text)) if *last == style => text.push(c),
                _ => row.push((style, c.to_string())),
            }
        }
        // The cursor past the end of a line: a cell of its own.
        if caret && state.cursor == (index, chars.len()) && end == chars.len() {
            row.push((Style::Caret, " ".into()));
        }
        out.push(row);
    }
    out
}

fn order(a: (usize, usize), b: (usize, usize)) -> ((usize, usize), (usize, usize)) {
    if a <= b { (a, b) } else { (b, a) }
}

/// The text between two positions of the doc, lines joined as the store has
/// them.
fn text_between(doc: &[DocLine], from: (usize, usize), to: (usize, usize)) -> String {
    let (from, to) = order(from, to);
    let mut out = String::new();
    for (index, line) in doc.iter().enumerate().take(to.0 + 1).skip(from.0) {
        let chars: Vec<char> = line.text().chars().collect();
        let start = if index == from.0 { from.1.min(chars.len()) } else { 0 };
        let end = if index == to.0 { to.1.min(chars.len()) } else { chars.len() };
        if index > from.0 {
            out.push('\n');
        }
        out.extend(&chars[start..end.max(start)]);
    }
    out
}

/// The review: every place the text is kept, store by store, with a little
/// of each around it.
fn review_lines(review: &Review, width: usize) -> Vec<Line> {
    let mut out: Vec<Line> = Vec::new();
    let intro = match &review.what {
        Reviewing::Strike(marks) => format!("Strike {} from:", if marks.len() == 1 { "1 mark".to_string() } else { format!("{} marks", marks.len()) }),
        Reviewing::Approve(n) => format!("Approve #{n}, striking:"),
    };
    out.push(vec![(Style::Bold, clip(&format!("  {intro}"), width))]);
    out.push(Vec::new());
    let impact = match &review.impact {
        Ok(impact) => impact,
        Err(error) => {
            out.push(vec![(Style::Problem, clip(&format!("  {error}"), width))]);
            return out;
        }
    };
    if impact.places.is_empty() && impact.unreachable.is_empty() {
        out.push(vec![(Style::Dim, "  nowhere else".into())]);
    }
    for (store, places, count) in impact.stores() {
        out.push(vec![
            (Style::Heading, format!("  {store}")),
            (Style::Dim, if places == 1 { format!("  ×{count}") } else { format!("  ×{count} in {places}") }),
        ]);
        for place in impact.places.iter().filter(|place| place.store == store) {
            out.push(place_line(place, Style::Plain, width));
        }
    }
    if !impact.unreachable.is_empty() {
        out.push(vec![(Style::Warn, "  not reached".into())]);
        for place in &impact.unreachable {
            out.push(place_line(place, Style::Problem, width));
        }
    }
    if review.short > 0 {
        out.push(vec![(Style::Dim, format!("  {} short: event only", review.short))]);
    }
    for note in &impact.notes {
        out.push(vec![(Style::Problem, clip(&format!("  {note}"), width))]);
    }
    out.push(Vec::new());
    out.push(vec![(Style::Danger, "  a strike · esc cancel".into())]);
    out
}

fn place_line(place: &crate::strike::Place, style: Style, width: usize) -> Line {
    let head = format!("    {}  ×{}  ", place.item, place.count);
    let mut line: Line = vec![(style, head.clone())];
    if let Some(excerpt) = &place.excerpt {
        let room = width.saturating_sub(text_width(&head) + 1);
        let mut used = 0;
        let mut marked = false;
        for c in excerpt.chars() {
            match c {
                EXCERPT_OPEN => marked = true,
                EXCERPT_CLOSE => marked = false,
                c => {
                    let w = c.width().unwrap_or(0);
                    if used + w > room {
                        break;
                    }
                    used += w;
                    let style = if marked { Style::Marked } else { Style::Dim };
                    match line.last_mut() {
                        Some((last, text)) if *last == style => text.push(c),
                        _ => line.push((style, c.to_string())),
                    }
                }
            }
        }
    }
    line
}

/// The tab's body, `rows` rows: the sections, the search line, the list,
/// and the strip (or the review).
pub(super) fn body(tab: &mut Open, width: usize, rows: usize) -> (Vec<Line>, Footer) {
    settle_work(tab);
    let state = &mut tab.memory;
    let mut out: Vec<Line> = Vec::new();
    // The sections.
    let mut bar: Line = vec![(Style::Plain, " ".into())];
    let waiting = state.data().requests.iter().filter(|request| request.state == crate::redact::State::Pending).count();
    for (index, section) in SECTIONS.into_iter().enumerate() {
        let label = match section {
            Section::Redactions if waiting > 0 => format!(" redactions · {waiting} "),
            other => format!(" {} ", other.name()),
        };
        // A click on a section shows it.
        let look = if section == state.section { Style::Tab } else { Style::Plain };
        bar.push(super::gui_view::hot(look, label, super::gui_view::Click::Pick(super::gui_view::Pick::MemorySection(index))));
        bar.push((Style::Plain, " ".into()));
    }
    if !state.marks.is_empty() {
        bar.push((Style::Danger, format!("   {} marked", state.marks.len())));
    }
    out.push(bar);
    // The search line, or what the section is.
    let line: Line = match (&state.input, state.section) {
        (Some(input), _) => {
            let label = match &input.purpose {
                Purpose::Search => "  search › ".to_string(),
                Purpose::Decline(n) => format!("  #{n} why? › "),
                Purpose::CloseLoop(id, true) => format!("  {id} done, how? › "),
                Purpose::CloseLoop(id, false) => format!("  drop {id}, why? › "),
            };
            let mut line = vec![(Style::Cursor, label)];
            let before: String = input.text[..input.cursor].iter().collect();
            let at: String = input.text.get(input.cursor).map_or(" ".to_string(), |c| c.to_string());
            let after: String = input.text.get(input.cursor + 1..).map(|rest| rest.iter().collect()).unwrap_or_default();
            line.extend([(Style::Plain, before), (Style::Caret, at), (Style::Plain, after)]);
            line
        }
        (None, Section::Archive) => {
            let found = match &state.events {
                Some(Ok(events)) => format!("{} event(s)", events.len()),
                Some(Err(error)) => error.clone(),
                None => String::new(),
            };
            if state.query.is_empty() {
                vec![(Style::Dim, format!("  latest · {found}"))]
            } else {
                vec![(Style::Dim, "  search › ".into()), (Style::Plain, state.query.clone()), (Style::Dim, format!("  · {found}"))]
            }
        }
        (None, Section::Threads) => vec![(Style::Dim, "  by state".into())],
        (None, Section::World) => vec![(Style::Dim, "  teal: review".into())],
        (None, Section::Loops) => {
            let open = state.data().loops.iter().filter(|entry| entry.status == crate::open_loops::LoopStatus::Open).count();
            vec![(Style::Dim, format!("  {open} open · teal: stale"))]
        }
        (None, Section::Redactions) => vec![(Style::Dim, "  asked · struck".into())],
    };
    out.push(line);

    let list = state.rows(width);
    let selected = state.selected_index(&list);
    let left = rows.saturating_sub(out.len());
    // The strip takes more of the screen while it has the cursor or the review.
    let strip_share = if state.review.is_some() {
        left.saturating_sub(4)
    } else if state.focus == Focus::Strip {
        left * 3 / 5
    } else {
        left * 2 / 5
    };
    let list_rows = left.saturating_sub(strip_share + 1).max(1);
    let strip_height = left.saturating_sub(list_rows + 1);

    // The list.
    let at = state.section.index();
    if let Some(index) = selected {
        if index < state.top[at] {
            state.top[at] = index;
        } else if index >= state.top[at] + list_rows {
            state.top[at] = index + 1 - list_rows;
        }
    }
    state.top[at] = state.top[at].min(list.len().saturating_sub(list_rows));
    if list.is_empty() {
        out.push(vec![(Style::Dim, format!("  {}", match state.section {
            Section::Archive => "nothing found",
            Section::Threads => "no threads yet",
            Section::World => "no models yet",
            Section::Loops => "no loops yet",
            Section::Redactions => "no redactions yet",
        }))]);
    }
    for (index, row) in list.iter().enumerate().skip(state.top[at]).take(list_rows) {
        let mut line = row.line.clone();
        if Some(index) == selected && state.shown_event.is_none() {
            if let Some((_, first)) = line.first_mut() {
                // The chevron where the row's indent was.
                *first = format!("❯{}", first.chars().skip(1).collect::<String>());
            }
            line[0].0 = if state.focus == Focus::List { Style::Cursor } else { Style::Dim };
        }
        out.push(super::gui_view::pick_row(line, super::gui_view::Click::Pick(super::gui_view::Pick::Memory(index))));
    }
    out.resize(2 + list_rows, Vec::new());
    state.list_rows = list_rows;
    state.strip_at = Some(out.len());

    // The strip: the selected thing's text in full, or the review.
    let kind = selected.map(|index| list[index].kind.clone());
    let key = state.doc_key(kind.as_ref());
    if state.doc_for.as_deref() != Some(key.as_str()) {
        state.doc = state.build_doc(kind.as_ref());
        state.doc_generation += 1;
        state.doc_for = Some(key);
        state.doc_stale = false;
        state.cursor = (0, 0);
        state.anchor = None;
        state.strip_top = 0;
    } else if state.doc_stale {
        // The same thing, its text out of date: built again in place.
        state.doc = state.build_doc(kind.as_ref());
        state.doc_generation += 1;
        state.doc_stale = false;
        let last = state.doc.len().saturating_sub(1);
        state.cursor.0 = state.cursor.0.min(last);
        let width = state.doc.get(state.cursor.0).map_or(0, |line| line.text().chars().count());
        state.cursor.1 = state.cursor.1.min(width);
        state.anchor = None;
    } else {
        settle_citing(state);
    }
    let mut more = String::new();
    let working = state.working.as_ref().map(|working| {
        let what = match (working.writing, &working.what) {
            (false, _) => "looking everywhere it is kept…",
            (true, Reviewing::Strike(_)) => "striking…",
            (true, Reviewing::Approve(_)) => "approving…",
        };
        vec![(Style::Cursor, format!("  {what} {}s", working.started.elapsed().as_secs()))]
    });
    let strip: Vec<Line> = match (&mut state.review, working) {
        (_, Some(line)) => vec![line],
        (Some(review), None) => {
            let lines = review_lines(review, width);
            review.scroll = review.scroll.min(lines.len().saturating_sub(strip_height));
            lines.into_iter().skip(review.scroll).take(strip_height).collect()
        }
        (None, None) => {
            let caret = state.focus == Focus::Strip;
            let key = (state.doc_generation, width.saturating_sub(2), state.doc.len());
            let layout = match state.layout.take() {
                Some(layout) if layout.key == key => layout,
                _ => build_layout(&state.doc, width.saturating_sub(2), key),
            };
            let total = layout.at.len();
            // The cursor's row on screen.
            if caret && let Some(row) = cursor_row(&layout, state.cursor) {
                if row < state.strip_top {
                    state.strip_top = row;
                } else if row >= state.strip_top + strip_height {
                    state.strip_top = row + 1 - strip_height;
                }
            }
            state.strip_top = state.strip_top.min(total.saturating_sub(strip_height));
            let below = total.saturating_sub(state.strip_top + strip_height);
            more = match (state.strip_top, below) {
                (0, 0) => String::new(),
                (0, below) => format!(" {below} more "),
                (above, 0) => format!(" {above} above "),
                (above, below) => format!(" {above} above · {below} more "),
            };
            let shown = render_rows(state, &layout, state.strip_top..state.strip_top + strip_height, caret);
            state.layout = Some(layout);
            shown
                .into_iter()
                .map(|row| {
                    let mut line = vec![(Style::Plain, "  ".to_string())];
                    line.extend(row);
                    line
                })
                .collect()
        }
    };
    let rule = width.saturating_sub(4 + text_width(&more));
    out.push(vec![(Style::Dim, format!("  {}", "─".repeat(rule))), (Style::Cursor, more)]);
    out.extend(strip);
    out.resize(rows, Vec::new());

    // The footer.
    let hints: Vec<(&'static str, &'static str)> = if state.working.is_some() {
        vec![("esc", "back")]
    } else if state.input.is_some() {
        vec![("⏎", "go"), ("esc", "cancel")]
    } else if state.review.is_some() {
        vec![("a", "strike"), ("↑↓", "scroll"), ("esc", "cancel")]
    } else if state.focus == Focus::Strip {
        let mut hints = vec![("←→↑↓", "move"), ("⇧+arrows", "mark")];
        if state.anchor.is_some() {
            hints.push(("⏎", "keep"));
        } else if state.doc.get(state.cursor.0).is_some_and(|line| line.link.is_some()) {
            hints.push(("⏎", "open"));
        }
        if !state.marks.is_empty() {
            hints.push(("a", "review"));
            hints.push(("u", "unmark"));
        }
        hints.push(("esc", "list"));
        hints
    } else {
        let mut hints = vec![("tab", "section"), ("↑↓", "move"), ("⏎", "mark text")];
        match state.section {
            Section::Archive => {
                hints.push(("/", "search"));
                hints.push(("[ ]", "beside"));
            }
            Section::World => {
                hints.push(("h", "history"));
                hints.push(("p", "pin"));
            }
            Section::Loops => {
                hints.push(("d", "done"));
                hints.push(("x", "drop"));
            }
            Section::Redactions => {
                hints.push(("a", "approve"));
                hints.push(("e", "edit"));
                hints.push(("d", "decline"));
            }
            Section::Threads => {}
        }
        if !state.marks.is_empty() && state.section != Section::Redactions {
            hints.push(("a", "review"));
        }
        hints.push(("esc", "back"));
        hints
    };
    let left = match state.shown_event {
        Some(id) if state.section == Section::Archive => vec![(Style::Status, format!("event #{id}"))],
        _ => vec![(Style::Status, state.section.name().to_string())],
    };
    (out, Footer::new(left, hints))
}

// ------------------------------------------------------------------ keys

/// What the selected row is now (laid out as `body` lays it out).
fn selected_kind(state: &mut State) -> Option<Kind> {
    let rows = state.rows(200);
    let index = state.selected_index(&rows)?;
    Some(rows[index].kind.clone())
}

/// A turn of the mouse wheel at `row` of the tab's body: the review, or the
/// strip under the list (its cursor, while it has one), scrolls; over the
/// list, the list scrolls first, the selection moving with it only when it
/// would leave the rows shown, and on its own once the list is at its end.
pub(super) fn wheel(tab: &mut Open, up: bool, row: usize) {
    let state = &mut tab.memory;
    let step = super::WHEEL_ROWS;
    let by = |at: usize| if up { at.saturating_sub(step) } else { at.saturating_add(step) };
    if let Some(review) = state.review.as_mut() {
        review.scroll = by(review.scroll);
        return;
    }
    if state.strip_at.is_some_and(|strip| row >= strip) {
        if state.focus == Focus::Strip {
            let last = state.doc.len().saturating_sub(1);
            state.cursor.0 = by(state.cursor.0).min(last);
            state.cursor.1 = state.cursor.1.min(line_len(state, state.cursor.0));
        } else {
            state.strip_top = by(state.strip_top);
        }
        return;
    }
    let list = state.rows(200);
    let at = state.section.index();
    let rows = state.list_rows.max(1);
    let last_top = list.len().saturating_sub(rows);
    let scrolls = if up { state.top[at] > 0 } else { state.top[at] < last_top };
    if !scrolls {
        move_selection(state, if up { -1 } else { 1 });
        return;
    }
    state.top[at] = by(state.top[at]).min(last_top);
    if let Some(index) = state.selected_index(&list) {
        let kept = index.clamp(state.top[at], (state.top[at] + rows).saturating_sub(1).min(list.len().saturating_sub(1)));
        if kept != index {
            state.selected[at] = Some(list[kept].kind.key());
            state.shown_event = None;
        }
    }
}

/// A click on a row of the list: it is selected, and the list has the keys.
pub(super) fn pick(tab: &mut Open, index: usize) {
    let state = &mut tab.memory;
    if state.input.is_some() || state.review.is_some() || state.working.is_some() {
        return;
    }
    let rows = state.rows(200);
    if let Some(row) = rows.get(index) {
        state.selected[state.section.index()] = Some(row.kind.key());
        state.shown_event = None;
        state.focus = Focus::List;
        state.anchor = None;
    }
}

/// A click on a section's name: it is shown, as Tab steps to it.
pub(super) fn pick_section(tab: &mut Open, index: usize) {
    let state = &mut tab.memory;
    if state.input.is_some() || state.review.is_some() || state.working.is_some() {
        return;
    }
    if let Some(section) = SECTIONS.get(index) {
        state.section = *section;
        state.focus = Focus::List;
        state.anchor = None;
        state.doc_for = None;
    }
}

fn move_selection(state: &mut State, by: isize) {
    let rows = state.rows(200);
    let Some(index) = state.selected_index(&rows) else { return };
    let index = (index as isize + by).clamp(0, rows.len() as isize - 1) as usize;
    state.selected[state.section.index()] = Some(rows[index].kind.key());
    state.shown_event = None;
}

fn line_len(state: &State, line: usize) -> usize {
    state.doc.get(line).map_or(0, |line| line.text().chars().count())
}

/// Move the strip's cursor, by character or line, staying on the doc.
fn move_cursor(state: &mut State, code: KeyCode, page: usize) {
    let (mut line, mut at) = state.cursor;
    let last = state.doc.len().saturating_sub(1);
    match code {
        KeyCode::Left => {
            if at > 0 {
                at -= 1;
            } else if line > 0 {
                line -= 1;
                at = line_len(state, line);
            }
        }
        KeyCode::Right => {
            if at < line_len(state, line) {
                at += 1;
            } else if line < last {
                line += 1;
                at = 0;
            }
        }
        KeyCode::Up => {
            line = line.saturating_sub(1);
            at = at.min(line_len(state, line));
        }
        KeyCode::Down => {
            line = (line + 1).min(last);
            at = at.min(line_len(state, line));
        }
        KeyCode::PageUp => {
            line = line.saturating_sub(page);
            at = at.min(line_len(state, line));
        }
        KeyCode::PageDown => {
            line = (line + page).min(last);
            at = at.min(line_len(state, line));
        }
        KeyCode::Home => at = 0,
        KeyCode::End => at = line_len(state, line),
        _ => {}
    }
    state.cursor = (line, at);
}

/// Run `action` on a thread of its own: a review reads every store and a
/// strike rewrites them, which on a large archive is minutes, and the tab
/// (and the agent behind it) must not stop for it. The answer is picked up by
/// `settle_work` when the tab is next drawn.
fn start(tab: &mut Open, what: Reviewing, writing: bool, action: Action) {
    let answer = std::sync::Arc::new(Mutex::new(None));
    let slot = answer.clone();
    let spawned = std::thread::Builder::new().name("memory-tab".into()).spawn(move || {
        let result = act(action);
        if let Ok(mut slot) = slot.lock() {
            *slot = Some(result);
        }
        // Shown at once, not at the next second's redraw.
        super::poke();
    });
    if let Err(error) = spawned {
        tab.note(format!("could not start: {error}"), true);
        return;
    }
    tab.memory.review = None;
    tab.memory.working = Some(Working { what, writing, started: Instant::now(), answer });
}

fn begin_review(tab: &mut Open) {
    let marks = tab.memory.marks.clone();
    // Asked of the agent that keeps the stores (this one, or the one this
    // terminal is attached to), so the review is the strike's own list.
    start(tab, Reviewing::Strike(marks.clone()), false, Action::Survey { fragments: marks });
}

fn begin_approval(tab: &mut Open, n: u64) {
    start(tab, Reviewing::Approve(n), false, Action::Review { request: n });
}

/// `a` again: the review's strike or approval, done by the agent that keeps
/// the stores (this one, or the one this terminal is attached to).
fn confirm(tab: &mut Open) {
    let Some(review) = tab.memory.review.take() else { return };
    let action = match &review.what {
        Reviewing::Strike(marks) => Action::Strike { fragments: marks.clone() },
        Reviewing::Approve(n) => Action::Approve { request: *n },
    };
    start(tab, review.what, true, action);
}

/// A review or strike that finished: its review shown, or what it did said.
pub(super) fn settle_work(tab: &mut Open) {
    let done = tab
        .memory
        .working
        .as_ref()
        .and_then(|working| working.answer.lock().ok().and_then(|mut answer| answer.take()));
    let Some(result) = done else { return };
    let Some(working) = tab.memory.working.take() else { return };
    if working.writing {
        match result {
            Ok(message) => {
                if matches!(working.what, Reviewing::Strike(_)) {
                    tab.memory.marks.clear();
                }
                tab.memory.anchor = None;
                tab.memory.focus = Focus::List;
                tab.note(message, false);
            }
            Err(error) => tab.note(error, true),
        }
        tab.memory.reload();
        return;
    }
    let (impact, short) = match &working.what {
        Reviewing::Strike(_) => (
            result.and_then(|json| serde_json::from_str::<Impact>(&json).map_err(|error| error.to_string())),
            0,
        ),
        Reviewing::Approve(_) => match result
            .and_then(|json| serde_json::from_str::<(Impact, usize)>(&json).map_err(|error| error.to_string()))
        {
            Ok((impact, short)) => (Ok(impact), short),
            Err(error) => (Err(error), 0),
        },
    };
    tab.memory.review = Some(Review { what: working.what, impact, short, scroll: 0 });
}

fn selected_request(state: &mut State) -> Option<crate::redact::Request> {
    let Some(Kind::Request(n)) = selected_kind(state) else { return None };
    state.data().requests.iter().find(|request| request.request == n).cloned()
}

/// A key on the memory tab (Alt switching is taken before this).
pub(super) fn key(tab: &mut Open, key: &KeyEvent, page: usize) -> Outcome {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    match key.code {
        KeyCode::Char('c' | 'C') if ctrl => return Outcome::Interrupt,
        KeyCode::Char('d' | 'D') if ctrl => return Outcome::Eof,
        _ => {}
    }

    // A review or strike running: Esc leaves the tab (it carries on), the
    // rest waits for it.
    if tab.memory.working.is_some() {
        if key.code == KeyCode::Esc {
            return Outcome::Closed;
        }
        tab.note("working", false);
        return Outcome::Stay;
    }

    // A line being typed: the search, or why a request is declined.
    if let Some(input) = tab.memory.input.as_mut() {
        match key.code {
            KeyCode::Esc => tab.memory.input = None,
            KeyCode::Enter => {
                let input = tab.memory.input.take().expect("typed");
                let text: String = input.text.iter().collect();
                match input.purpose {
                    Purpose::Search => {
                        tab.memory.query = text.trim().to_string();
                        tab.memory.events = None;
                        tab.memory.shown_event = None;
                        tab.memory.selected[0] = None;
                    }
                    Purpose::Decline(n) => {
                        match act(Action::Decline { request: n, note: text }) {
                            Ok(message) => tab.note(message, false),
                            Err(error) => tab.note(error, true),
                        }
                        tab.memory.reload();
                    }
                    Purpose::CloseLoop(id, done) => {
                        match act(Action::CloseLoop { id, done, said: text }) {
                            Ok(message) => tab.note(message, false),
                            Err(error) => tab.note(error, true),
                        }
                        tab.memory.reload();
                    }
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
            KeyCode::Char('u' | 'U') if ctrl => {
                input.text.drain(..input.cursor);
                input.cursor = 0;
            }
            KeyCode::Char(c) if !ctrl => {
                input.text.insert(input.cursor, c);
                input.cursor += 1;
            }
            _ => {}
        }
        return Outcome::Stay;
    }

    // The review: `a` again does it.
    if let Some(review) = tab.memory.review.as_mut() {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => tab.memory.review = None,
            KeyCode::Up => review.scroll = review.scroll.saturating_sub(1),
            KeyCode::Down => review.scroll += 1,
            KeyCode::PageUp => review.scroll = review.scroll.saturating_sub(page),
            KeyCode::PageDown => review.scroll += page,
            KeyCode::Char('a') if review.impact.is_ok() => confirm(tab),
            _ => {}
        }
        return Outcome::Stay;
    }

    match key.code {
        KeyCode::Tab | KeyCode::BackTab => {
            let step = if key.code == KeyCode::BackTab { SECTIONS.len() - 1 } else { 1 };
            let state = &mut tab.memory;
            state.section = SECTIONS[(state.section.index() + step) % SECTIONS.len()];
            state.focus = Focus::List;
            state.anchor = None;
            state.doc_for = None;
            return Outcome::Stay;
        }
        _ => {}
    }

    if tab.memory.focus == Focus::Strip {
        let state = &mut tab.memory;
        match key.code {
            KeyCode::Esc if state.anchor.is_some() => state.anchor = None,
            KeyCode::Esc => state.focus = Focus::List,
            KeyCode::Left | KeyCode::Right | KeyCode::Up | KeyCode::Down | KeyCode::Home | KeyCode::End | KeyCode::PageUp
            | KeyCode::PageDown => {
                if shift {
                    if state.anchor.is_none() {
                        state.anchor = Some(state.cursor);
                    }
                } else {
                    state.anchor = None;
                }
                move_cursor(state, key.code, page);
            }
            KeyCode::Enter | KeyCode::Char('m') if state.anchor.is_some() => {
                let anchor = state.anchor.take().expect("marking");
                let text = text_between(&state.doc, anchor, state.cursor);
                let words = {
                    let (from, to) = order(anchor, state.cursor);
                    (from.0..=to.0).all(|line| state.doc.get(line).is_some_and(|line| line.words))
                };
                let trimmed = text.trim();
                if !words {
                    tab.note("labels can't be marked", true);
                } else if trimmed.chars().count() < crate::strike::MIN_FRAGMENT {
                    tab.note(format!("{}+ characters", crate::strike::MIN_FRAGMENT), true);
                } else if !state.marks.iter().any(|mark| mark == trimmed) {
                    state.marks.push(trimmed.to_string());
                    let n = state.marks.len();
                    tab.note(format!("{n} marked"), false);
                }
            }
            KeyCode::Enter => {
                if let Some(event) = state.doc.get(state.cursor.0).and_then(|line| line.link) {
                    go_to_event(state, event);
                }
            }
            KeyCode::Char('u') => {
                if state.marks.pop().is_some() {
                    tab.note(format!("{} marked", tab.memory.marks.len()), false);
                }
            }
            KeyCode::Char('a') if !state.marks.is_empty() => begin_review(tab),
            _ => {}
        }
        return Outcome::Stay;
    }

    // The list.
    let section = tab.memory.section;
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => {
            if tab.memory.shown_event.take().is_none() {
                return Outcome::Closed;
            }
        }
        KeyCode::Up => move_selection(&mut tab.memory, -1),
        KeyCode::Down => move_selection(&mut tab.memory, 1),
        KeyCode::PageUp => move_selection(&mut tab.memory, -(page as isize)),
        KeyCode::PageDown => move_selection(&mut tab.memory, page as isize),
        KeyCode::Home => move_selection(&mut tab.memory, isize::MIN / 2),
        KeyCode::End => move_selection(&mut tab.memory, isize::MAX / 2),
        KeyCode::Enter | KeyCode::Right | KeyCode::Left => {
            let state = &mut tab.memory;
            let kind = if state.shown_event.is_some() { None } else { selected_kind(state) };
            let fold = match &kind {
                Some(Kind::Group(name)) => Some((name.clone(), !(name.ends_with("resolved") || name.ends_with("abandoned")))),
                Some(Kind::Model(id)) => Some((format!("world:{id}"), state.data().models.len() <= 3)),
                _ => None,
            };
            match fold {
                Some((name, default)) => {
                    let open = state.is_open(&name, default);
                    match key.code {
                        // Unfolded already: → reads the model's own text.
                        KeyCode::Right if open && matches!(kind, Some(Kind::Model(_))) => to_strip(state),
                        KeyCode::Right if !open => state.toggle(&name),
                        KeyCode::Left if open => state.toggle(&name),
                        KeyCode::Enter => state.toggle(&name),
                        _ => {}
                    }
                }
                None if key.code != KeyCode::Left => to_strip(state),
                None => {}
            }
        }
        KeyCode::Char('/') if section == Section::Archive => {
            let text: Vec<char> = tab.memory.query.chars().collect();
            let cursor = text.len();
            tab.memory.input = Some(Input { purpose: Purpose::Search, text, cursor });
        }
        KeyCode::Char('[' | ']') if section == Section::Archive => {
            let state = &mut tab.memory;
            let base = state.shown_event.or_else(|| match selected_kind(state) {
                Some(Kind::Event(id)) => Some(id),
                _ => None,
            });
            if let Some(base) = base {
                let next = if key.code == KeyCode::Char('[') { base.saturating_sub(1).max(1) } else { base + 1 };
                let last = memory().map_or(base, |memory| memory.last_event_id());
                if next <= last {
                    state.shown_event = Some(next);
                }
            }
        }
        KeyCode::Char('h') if section == Section::World => {
            tab.memory.world_history = !tab.memory.world_history;
        }
        KeyCode::Char('p') if section == Section::World => match selected_kind(&mut tab.memory) {
            Some(Kind::Record(id)) => {
                let pinned = tab.memory.data().pins.contains_key(&id);
                let result = archive_path()
                    .ok_or_else(|| "no archive here".to_string())
                    .and_then(|archive| {
                        crate::facts::set_pin(&archive.with_file_name("world_model.sqlite3"), &id, !pinned).map_err(|e| e.to_string())
                    });
                match result {
                    Ok(()) if pinned => tab.note(format!("{id} unpinned; sleep won't pin it again"), false),
                    Ok(()) => tab.note(format!("{id} pinned: in s0ma's context from its next start or sleep"), false),
                    Err(error) => tab.note(format!("not pinned: {error}"), true),
                }
                tab.memory.reload();
            }
            _ => tab.note("p pins a record: select one", true),
        },
        KeyCode::Char('a') if section == Section::Redactions && selected_request(&mut tab.memory).is_some() => {
            match selected_request(&mut tab.memory) {
                Some(request) if request.state == crate::redact::State::Pending => begin_approval(tab, request.request),
                Some(_) if !tab.memory.marks.is_empty() => begin_review(tab),
                Some(request) => tab.note(format!("#{} is {} already", request.request, crate::redact::state_word(request.state)), true),
                None => {}
            }
        }
        KeyCode::Char('a') if !tab.memory.marks.is_empty() => begin_review(tab),
        KeyCode::Char('e') if section == Section::Redactions => match selected_request(&mut tab.memory) {
            Some(request) if request.state == crate::redact::State::Pending => edit(tab, &request),
            Some(request) => tab.note(format!("#{} is decided", request.request), true),
            None => {}
        },
        KeyCode::Char(c @ ('d' | 'x')) if section == Section::Loops => match selected_loop(&mut tab.memory) {
            Some(entry) if entry.status == crate::open_loops::LoopStatus::Open => {
                tab.memory.input = Some(Input { purpose: Purpose::CloseLoop(entry.id.clone(), c == 'd'), text: Vec::new(), cursor: 0 });
            }
            Some(entry) => tab.note(format!("{} is {} already", entry.id, entry.status.label()), true),
            None => {}
        },
        KeyCode::Char('d') if section == Section::Redactions => match selected_request(&mut tab.memory) {
            Some(request) if request.state == crate::redact::State::Pending => {
                tab.memory.input = Some(Input { purpose: Purpose::Decline(request.request), text: Vec::new(), cursor: 0 });
            }
            Some(request) => tab.note(format!("#{} is decided", request.request), true),
            None => {}
        },
        _ => {}
    }
    Outcome::Stay
}

fn selected_loop(state: &mut State) -> Option<crate::open_loops::OpenLoop> {
    let Some(Kind::Loop(id)) = selected_kind(state) else { return None };
    state.data().loops.iter().find(|entry| entry.id == id).cloned()
}

fn to_strip(state: &mut State) {
    state.focus = Focus::Strip;
    state.cursor = (0, 0);
    state.anchor = None;
}

/// The archive, at event `id`.
fn go_to_event(state: &mut State, id: u64) {
    state.section = Section::Archive;
    state.shown_event = Some(id);
    state.focus = Focus::Strip;
    state.cursor = (0, 0);
    state.anchor = None;
    state.doc_for = None;
}

/// `e`: the proposal in the operator's editor, on the shell's screen, and
/// what comes back saved as his version.
fn edit(tab: &mut Open, request: &crate::redact::Request) {
    // A proposal to take the whole content out is only the marker: there is
    // nothing in it to edit. The event's own text is opened instead, to cut
    // what should go; saved as it was, the agent's proposal stands.
    let whole = request.current().trim() == crate::redact::MARKER;
    let original = whole
        .then(|| memory().and_then(|memory| memory.get_event(request.event_id).ok().flatten()).map(|event| event.content))
        .flatten();
    let start = original.as_deref().unwrap_or(request.current());
    crate::terminal::set_mouse(false);
    let edited = crate::mistakes::view::edit_text(start, &format!("redact-{}", request.request));
    crate::terminal::set_mouse(true);
    crate::page::mark_stale();
    // The editor had the screen: all of it is drawn again.
    tab.frame.clear();
    let _ = crossterm::queue!(std::io::stdout(), crossterm::terminal::Clear(crossterm::terminal::ClearType::All));
    // An editor ends the file with a newline the event did not have.
    let edited = edited.map(|text| {
        text.map(|text| match text.strip_suffix('\n') {
            Some(trimmed) if !start.ends_with('\n') => trimmed.to_owned(),
            _ => text,
        })
    });
    match edited {
        Ok(Some(text)) if text == start || text == request.current() => tab.note("unchanged", false),
        Ok(Some(text)) => {
            match act(Action::Edit { request: request.request, text }) {
                Ok(message) => tab.note(message, false),
                Err(error) => tab.note(error, true),
            }
            tab.memory.reload();
        }
        Ok(None) => tab.note("editor failed", true),
        Err(error) => tab.note(format!("the editor could not be run: {error}"), true),
    }
}

/// A paste: into the line being typed, if one is.
pub(super) fn paste(tab: &mut Open, text: &str) {
    if let Some(input) = tab.memory.input.as_mut() {
        for c in text.chars().filter(|c| !c.is_control()) {
            input.text.insert(input.cursor, c);
            input.cursor += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(tab: &mut Open, code: KeyCode, modifiers: KeyModifiers) -> Outcome {
        key(tab, &KeyEvent::new(code, modifiers), 10)
    }

    /// Shift and the arrows mark; Enter keeps it; the labels around the
    /// store's words are not the store's and are refused.
    #[test]
    fn marking_keeps_the_stores_words_and_refuses_the_labels() {
        let mut tab = Open::new(super::super::View::Memory);
        tab.memory.focus = Focus::Strip;
        tab.memory.doc = vec![DocLine::label(Style::Heading, "statement"), DocLine::words(Style::Plain, "key sk-123456 here")];
        tab.memory.doc_for = Some("x".into());
        // Across the label: refused.
        tab.memory.cursor = (0, 3);
        for _ in 0..8 {
            press(&mut tab, KeyCode::Right, KeyModifiers::SHIFT);
        }
        press(&mut tab, KeyCode::Enter, KeyModifiers::NONE);
        assert!(tab.memory.marks.is_empty(), "{:?}", tab.memory.marks);
        // In the words: kept.
        tab.memory.cursor = (1, 4);
        for _ in 0..9 {
            press(&mut tab, KeyCode::Right, KeyModifiers::SHIFT);
        }
        press(&mut tab, KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(tab.memory.marks, ["sk-123456"]);
        // Too short to keep.
        tab.memory.cursor = (1, 0);
        for _ in 0..3 {
            press(&mut tab, KeyCode::Right, KeyModifiers::SHIFT);
        }
        press(&mut tab, KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(tab.memory.marks.len(), 1);
        // u takes the last back; Esc leaves the strip, and Esc again the tab.
        press(&mut tab, KeyCode::Char('u'), KeyModifiers::NONE);
        assert!(tab.memory.marks.is_empty());
        assert_eq!(press(&mut tab, KeyCode::Esc, KeyModifiers::NONE), Outcome::Stay);
        assert_eq!(tab.memory.focus, Focus::List);
        assert_eq!(press(&mut tab, KeyCode::Esc, KeyModifiers::NONE), Outcome::Closed);
    }
}
