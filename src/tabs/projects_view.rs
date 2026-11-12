//! The projects tab's subtabs (a submodule of `tabs.rs`): open, review,
//! verification and complete. Each board in them has sections: the open
//! subtab's are its board (the board's own tab, `tabs/board_view.rs`), its files,
//! notes and reports; the later subtabs' are the reports, files and notes,
//! the checks and the verification where they have them, and the board
//! itself, walked as on its own tab. The workflow is in `board/flow.rs`.
//!
//! ```text
//!   open   review 1   verification   complete 2                     tab next
//! ❯ [!] self-model-drift   The self-model drifts after sleep    round 2 · 3 files · 2 notes
//!     upload-retries     Uploads retry forever                round 1 · 1 file
//!   ── reports 2 │ files 3 │ notes 2 │ board ─────────────────────────────
//!   (the section, scrolled)
//! ```
//!
//! Tab and Shift+Tab step through the subtabs (as through the memory tab's
//! sections; Alt+P too); Ctrl+N and Ctrl+P step through the boards, in every
//! subtab. ←→ pick a section (Shift+←→ too; on the open board, where ←→
//! fold, only those), and ↑↓ scroll it (or pick a file or a note); in a
//! board section ↑↓ walk the tree and ⏎ folds. Every
//! round's report is shown, the latest first, as kept in
//! `reports/round-<n>.md`.
//! In review, `n` leaves a note, `b` twice sends the board back to the agent with them,
//! `v` twice sends it to verification. In verification, `a` twice accepts the
//! verified board. A board goes back to any stage before, as often as he
//! likes: `r` takes it back to review and `b` to the agent (its board, open
//! again), each with why, from verification or complete; `v` twice
//! verifies a complete board again. Every move is done by the agent that
//! runs the turns (`Action::Project`), which tells the agent.
//!
//! The files section is the share tab's own browser on the board's files
//! folder (`share_view::State::embedded`): a tree that folds (⏎ on a
//! folder, Ctrl+O for all), the selected file under it, markdown rendered
//! (`m` for raw), ⏎ on a file to give it the section (Esc back), `e` to
//! edit it, `s` to save a copy. Nothing is added or deleted there.

use super::gui_view::{Click, Pick, pick_row};
use super::{Action, Footer, Open, Outcome, View, act, clip, flow, text_width};
use crate::board::{Board, BoardStatus};
use crate::board::flow::{Flow, FlowStore, Note, Stage};
use crate::style::{Line, Style};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde::{Deserialize, Serialize};

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Read again this often while up.
const POLL: Duration = Duration::from_millis(1500);

/// The subtabs, in order: Alt+P steps through them.
pub(super) const SUBTABS: [View; 4] = [View::Board, View::Review, View::Verification, View::Deliverables];

pub(super) fn is_projects(view: &View) -> bool {
    SUBTABS.contains(view)
}

/// The subtab after `view` (Tab, Alt+P).
pub(super) fn next(view: &View) -> View {
    let at = SUBTABS.iter().position(|subtab| subtab == view).unwrap_or(0);
    SUBTABS[(at + 1) % SUBTABS.len()].clone()
}

/// The subtab before `view` (Shift+Tab).
pub(super) fn previous(view: &View) -> View {
    let at = SUBTABS.iter().position(|subtab| subtab == view).unwrap_or(0);
    SUBTABS[(at + SUBTABS.len() - 1) % SUBTABS.len()].clone()
}

/// A note or a reason is being typed here: Tab is not a subtab switch then.
pub(super) fn typing_now(tab: &Open) -> bool {
    tab.projects.input.is_some() || (section(tab) == Section::Audit && super::audit_view::holds_keys(tab))
}

static LAST: Mutex<Option<View>> = Mutex::new(None);

/// The subtab last up (Alt+P from elsewhere opens it).
pub(super) fn last() -> View {
    LAST.lock().ok().and_then(|last| last.clone()).unwrap_or(View::Board)
}

fn stage_of(view: &View) -> Stage {
    match view {
        View::Review => Stage::Review,
        View::Verification => Stage::Verification,
        View::Deliverables => Stage::Deliverable,
        _ => Stage::Working,
    }
}

/// How many boards each subtab has, as last counted (the tab's reads, and
/// `tabs::boards_waiting`'s).
static COUNTS: Mutex<[usize; 4]> = Mutex::new([0; 4]);

pub(super) fn set_counts(counts: [usize; 4]) {
    if let Ok(mut kept) = COUNTS.lock() {
        *kept = counts;
    }
}

pub(super) fn counts() -> [usize; 4] {
    COUNTS.lock().map(|counts| *counts).unwrap_or_default()
}

// ------------------------------------------------------------------ what is read

/// A board in a stage, with its workflow and the operator's notes.
#[derive(Clone)]
pub(super) struct Item {
    pub board: Board,
    pub flow: Flow,
    pub notes: Vec<Note>,
    /// Its leads, findings and hints, for its board section.
    pub snapshot: Option<super::board_view::Snapshot>,
}

pub(super) struct Read {
    at: Instant,
    items: Result<Vec<Item>, String>,
}

/// The boards in `stage` (newest move first), and the counts for the
/// subtab line; the leads, findings and hints of `tree_of` only (the board
/// selected, for its board section), not of every board each read.
pub(super) fn read(boards: &crate::board::BoardStore, stage: Stage, tree_of: Option<&str>) -> Result<(Vec<Item>, [usize; 4]), Box<dyn std::error::Error>> {
    let flows = FlowStore::beside(boards);
    let mut all = flows.all()?;
    all.sort_by(|a, b| b.entered_at.cmp(&a.entered_at));
    let mut counts = [super::shown_boards(boards)?.len(), 0, 0, 0];
    let mut items = Vec::new();
    for flow in all {
        match flow.stage() {
            Stage::Review => counts[1] += 1,
            Stage::Verification => counts[2] += 1,
            Stage::Deliverable => counts[3] += 1,
            Stage::Working | Stage::Discarded => {}
        }
        if flow.stage() != stage {
            continue;
        }
        let Some(board) = boards.board_by_id(&flow.board_id)? else { continue };
        let notes = flows.notes(&board.id)?;
        let snapshot = if tree_of == Some(board.id.as_str()) {
            Some(super::board_view::Snapshot {
                intents: boards.intents(&board.id)?,
                facts: boards.facts(&board.id)?,
                hints: boards.hints(&board.id, false)?,
                board: board.clone(),
                flow: None,
                notes: Vec::new(),
            })
        } else {
            None
        };
        items.push(Item { board, flow, notes, snapshot });
    }
    Ok((items, counts))
}

type Slot<T> = Arc<Mutex<Option<T>>>;

// ------------------------------------------------------------------ the tab's state

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Typing {
    Note,
    /// Why it goes back to the agent (its board).
    Why,
    /// Why it goes back to review.
    WhyReview,
}

struct Input {
    typing: Typing,
    text: Vec<char>,
    cursor: usize,
}

#[derive(Default)]
pub(super) struct State {
    read: Option<Read>,
    reading: Option<Slot<Read>>,
    /// By board id, so boards arriving or leaving don't move it.
    selected: Option<String>,
    top: usize,
    section: usize,
    scroll: usize,
    /// The file or note picked in its section.
    pick: usize,
    input: Option<Input>,
    /// A key pressed once on a board: again does it.
    pub(super) armed: Option<(char, String)>,
    /// The files section: the board's files folder in the share tab's own
    /// browser (a tree that folds, the selected file under it, markdown
    /// rendered), by board id.
    files: Option<(String, super::share_view::State)>,
    /// The row of the tab's body where that browser starts, and where the
    /// list of boards ends, as last drawn (for the wheel).
    files_at: usize,
    list_end: usize,
    /// The board its board section was last drawn for: another starts with
    /// its tree as new.
    pub(super) board_for: Option<String>,
}

/// `run` with the board's files browser as the share tab's state (its
/// functions work on `tab.share`); put back after.
fn with_files<T>(tab: &mut Open, board: &Board, run: impl FnOnce(&mut Open) -> T) -> T {
    let fresh = !tab.projects.files.as_ref().is_some_and(|(id, _)| *id == board.id);
    if fresh {
        let folder = crate::board::flow::folder(&board.name);
        let dir = format!("{}/{folder}/files", crate::share::share_dir());
        let header = format!("~/s0ma-share/{folder}/files · as they were on s0ma's machine");
        tab.projects.files = Some((board.id.clone(), super::share_view::State::embedded(dir, "files", &header)));
    }
    let (id, mut state) = tab.projects.files.take().expect("made above");
    std::mem::swap(&mut tab.share, &mut state);
    let result = run(tab);
    std::mem::swap(&mut tab.share, &mut state);
    tab.projects.files = Some((id, state));
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Section {
    /// The board itself: its leads, findings and hints, as a tree.
    Board,
    /// The agent's reports, every round's, the latest first.
    Reports,
    Files,
    Notes,
    Checks,
    Verification,
    /// The project's audits, the audit tab embedded (only when it has one).
    Audit,
}

impl Section {
    fn label(self) -> &'static str {
        match self {
            Section::Board => "board",
            Section::Reports => "reports",
            Section::Files => "files",
            Section::Notes => "notes",
            Section::Checks => "checks",
            Section::Verification => "verification",
            Section::Audit => "audit",
        }
    }
}

fn all_sections(view: &View) -> &'static [Section] {
    match view {
        View::Board => &[Section::Board, Section::Audit, Section::Files, Section::Notes, Section::Reports],
        View::Review => &[Section::Reports, Section::Audit, Section::Files, Section::Notes, Section::Board],
        View::Verification => &[Section::Checks, Section::Verification, Section::Audit, Section::Reports, Section::Files, Section::Notes, Section::Board],
        _ => &[Section::Reports, Section::Verification, Section::Audit, Section::Notes, Section::Files, Section::Board],
    }
}

/// The sections of the subtab up, for the board shown: the audit section
/// only for a project with an audit (one without looks as it always did).
fn sections(tab: &Open) -> Vec<Section> {
    let audits = current_board(tab).is_some_and(|board| has_audits(&board));
    all_sections(&tab.view).iter().copied().filter(|s| *s != Section::Audit || audits).collect()
}

/// The board the subtab shows: the open board's, or the one selected.
pub(super) fn current_board(tab: &Open) -> Option<String> {
    if tab.view == View::Board {
        return tab.board_read.as_ref()?.snapshot.as_ref().ok()?.as_ref().map(|snapshot| snapshot.board.name.clone());
    }
    let items = tab.projects.read.as_ref()?.items.as_ref().ok()?;
    let id = tab.projects.selected.as_deref();
    items.iter().find(|item| Some(item.board.id.as_str()) == id).or(items.first()).map(|item| item.board.name.clone())
}

/// Whether a project has audits, asked at most every two seconds per board.
fn has_audits(board: &str) -> bool {
    thread_local! {
        static SEEN: std::cell::RefCell<std::collections::HashMap<String, (bool, Instant)>> = std::cell::RefCell::new(std::collections::HashMap::new());
    }
    SEEN.with(|seen| {
        let mut seen = seen.borrow_mut();
        if let Some((has, at)) = seen.get(board)
            && at.elapsed() < Duration::from_secs(2)
        {
            return *has;
        }
        let has = !crate::audit::ops::for_board(board).is_empty();
        seen.insert(board.to_string(), (has, Instant::now()));
        has
    })
}

/// The section up in `tab`.
pub(super) fn section(tab: &Open) -> Section {
    let sections = sections(tab);
    sections[tab.projects.section.min(sections.len() - 1)]
}

/// ←→ or Shift+←→: the section before or after, if `key` is one. In every
/// section, the boards' too, ←→ step the sections and ⏎ is the section's
/// own (it folds the tree, views a file).
fn section_step(tab: &mut Open, key: &KeyEvent) -> bool {
    let shift = key.modifiers.contains(KeyModifiers::SHIFT) && !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
    let plain = key.modifiers.is_empty();
    let forward = match key.code {
        KeyCode::Left if shift || plain => false,
        KeyCode::Right if shift || plain => true,
        _ => return false,
    };
    let count = sections(tab).len();
    let at = tab.projects.section.min(count - 1);
    tab.projects.section = if forward { (at + 1) % count } else { (at + count - 1) % count };
    tab.projects.scroll = 0;
    tab.projects.pick = 0;
    true
}

/// What the operator does to a board, done by the agent that runs the turns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Move {
    Note { text: String },
    Unnote { id: String },
    Back { why: String },
    /// From verification or the deliverables, to review again.
    Review { why: String },
    Verify,
    Accept,
    /// Thrown away, wherever it is.
    Discard,
    /// The agent's proposed exhaustion: closed exhausted, or open again with why.
    Exhaust,
    Reopen { why: String },
}

/// `Action::Project`, in that process: the move, and the agent woken for it.
pub(super) fn perform(archive: &std::path::Path, board: &str, done: Move) -> Result<String, String> {
    let boards = crate::board::BoardStore::beside_archive(archive);
    let outcome = match done {
        Move::Note { text } => crate::board::flow::annotate(&boards, board, &text),
        Move::Unnote { id } => crate::board::flow::unannotate(&boards, board, &id),
        Move::Back { why } => crate::board::flow::send_back(&boards, board, &why),
        Move::Review { why } => crate::board::flow::to_review(&boards, board, &why),
        Move::Verify => crate::board::flow::to_verification(&boards, board),
        Move::Accept => crate::board::flow::accept(&boards, board),
        Move::Discard => crate::board::flow::discard(&boards, board),
        Move::Exhaust => crate::board::flow::exhaust(&boards, board),
        Move::Reopen { why } => crate::board::flow::reopen_exhausted(&boards, board, &why),
    };
    let outcome = outcome.map_err(|error| format!("the board could not be written: {error}"))?;
    crate::subagent::wake_prompt();
    outcome
}

// ------------------------------------------------------------------ reading

fn refresh(tab: &mut Open) {
    if let Some(slot) = &tab.projects.reading
        && let Some(read) = slot.lock().ok().and_then(|mut slot| slot.take())
    {
        tab.projects.reading = None;
        // A read that failed keeps what was read before up.
        if read.items.is_ok() || tab.projects.read.is_none() {
            tab.projects.read = Some(read);
        } else if let Some(old) = tab.projects.read.as_mut() {
            old.at = Instant::now();
        }
    }
    let stale = tab.projects.read.as_ref().is_none_or(|read| read.at.elapsed() >= POLL);
    if !stale || tab.projects.reading.is_some() {
        return;
    }
    let stage = stage_of(&tab.view);
    // The selected board's tree, or the first's when none is selected yet.
    let tree_of = tab.projects.selected.clone().or_else(|| items(tab).first().map(|item| item.board.id.clone()));
    let archive = super::ARCHIVE.get().cloned();
    let slot: Slot<Read> = Arc::new(Mutex::new(None));
    let answer = slot.clone();
    let spawned = std::thread::Builder::new().name("projects-tab".into()).spawn(move || {
        let items = match archive {
            Some(archive) => match read(&crate::board::BoardStore::beside_archive(&archive), stage, tree_of.as_deref()) {
                Ok((items, counts)) => {
                    set_counts(counts);
                    Ok(items)
                }
                Err(error) => Err(error.to_string()),
            },
            None => Err("the boards are not available in this process".into()),
        };
        if let Ok(mut answer) = answer.lock() {
            *answer = Some(Read { at: Instant::now(), items });
        }
        super::poke();
    });
    if spawned.is_ok() {
        tab.projects.reading = Some(slot);
    }
}

/// Read again at once (after a move).
fn reread(tab: &mut Open) {
    if let Some(read) = tab.projects.read.as_mut() {
        read.at = Instant::now().checked_sub(POLL).unwrap_or(read.at);
    }
}

fn items(tab: &Open) -> Vec<Item> {
    tab.projects.read.as_ref().and_then(|read| read.items.as_ref().ok()).cloned().unwrap_or_default()
}

/// The selected board's index, kept by id; the first when it is gone.
fn selected(tab: &mut Open, items: &[Item]) -> Option<usize> {
    let index = tab
        .projects
        .selected
        .as_deref()
        .and_then(|id| items.iter().position(|item| item.board.id == id))
        .or_else(|| (!items.is_empty()).then_some(0));
    let id = index.map(|index| items[index].board.id.clone());
    if id != tab.projects.selected {
        tab.projects.selected = id;
        tab.projects.scroll = 0;
        tab.projects.pick = 0;
    }
    // Its tree is read with the next read: at once, when it has none yet.
    if let Some(index) = index
        && items[index].snapshot.is_none()
        && section(tab) == Section::Board
    {
        reread(tab);
    }
    index
}

// ------------------------------------------------------------------ drawing

/// The subtab line: which is up, and how many boards each has.
pub(super) fn subtab_line(view: &View, width: usize) -> Line {
    let counts = counts();
    let mut line: Line = vec![(Style::Plain, " ".into())];
    for (index, subtab) in SUBTABS.iter().enumerate() {
        let name = match subtab {
            View::Board => "open",
            View::Review => "review",
            View::Verification => "verification",
            _ => "complete",
        };
        let count = if index == 0 || counts[index] == 0 { String::new() } else { format!(" {}", counts[index]) };
        // A click on a subtab opens it.
        let click = || Click::Pick(Pick::Subtab(subtab.clone()));
        if subtab == view {
            line.push(super::gui_view::hot(Style::Tab, format!(" {name}{count} "), click()));
        } else {
            line.push(super::gui_view::hot(Style::Dim, format!(" {name}"), click()));
            line.push(super::gui_view::hot(if index == 1 && counts[1] > 0 { Style::Warn } else { Style::Dim }, format!("{count} "), click()));
        }
        line.push((Style::Plain, " ".into()));
    }
    let _ = width;
    line
}

fn ago(stamp: &str) -> String {
    let Ok(at) = chrono::DateTime::parse_from_rfc3339(stamp) else { return String::new() };
    let seconds = (chrono::Local::now() - at.with_timezone(&chrono::Local)).num_seconds().max(0);
    match seconds {
        0..60 => format!("{seconds}s ago"),
        60..3600 => format!("{}m ago", seconds / 60),
        3600..86400 => format!("{}h ago", seconds / 3600),
        _ => format!("{}d ago", seconds / 86400),
    }
}

fn plural(count: usize, one: &str) -> String {
    format!("{count} {one}{}", if count == 1 { "" } else { "s" })
}

/// A board's marker in the lists: `[!]` waits for the operator, `[*]` the agent or the
/// verifier is at it, `[+]` delivered (`[-]` when verification failed).
fn marker(item: &Item) -> (Style, &'static str) {
    let failed = item.flow.verdict.as_deref() == Some("failed");
    match item.flow.stage() {
        _ if item.flow.waits_for_operator() => (Style::Warn, "[!] "),
        Stage::Deliverable if failed => (Style::Problem, "[-] "),
        Stage::Deliverable if item.board.status == BoardStatus::Exhausted => (Style::Dim, "[-] "),
        Stage::Deliverable => (Style::Success, "[+] "),
        _ => (Style::Work, "[*] "),
    }
}

fn list_row(item: &Item, view: &View, selected: bool, width: usize) -> Line {
    let mark = marker(item);
    let lead = if selected { (Style::Cursor, "❯ ".to_string()) } else { (Style::Plain, "  ".to_string()) };
    let mut meta = vec![format!("round {}", item.flow.round), plural(item.flow.files.len(), "file")];
    if !item.notes.is_empty() {
        meta.push(plural(item.notes.len(), "note"));
    }
    // A proposed exhaustion, and one closed so: said, as there is no verdict.
    match item.board.status {
        BoardStatus::ProposedExhausted => meta.insert(0, "nothing left to try, says s0ma".into()),
        BoardStatus::Exhausted => meta.insert(0, "closed exhausted".into()),
        _ => {}
    }
    if *view == View::Verification {
        meta.push(match (&item.flow.verdict, item.flow.verifier_state.as_deref(), &item.flow.own_verdict) {
            (Some(verdict), _, _) => format!("verified: {verdict}"),
            (None, state, own) => format!("verifier {} · s0ma {}", state.unwrap_or("?"), if own.is_some() { "done" } else { "checking" }),
        });
    }
    if *view == View::Deliverables
        && let Some(verdict) = &item.flow.verdict
    {
        meta.push(format!("verified {verdict}"));
    }
    meta.push(ago(&item.flow.entered_at));
    let meta = meta.join(" · ");
    let name = format!("{:<22}", clip(&item.board.name, 22));
    let room = width.saturating_sub(2 + 4 + 22 + 2 + text_width(&meta) + 2);
    vec![
        lead,
        (mark.0, mark.1.to_string()),
        (if selected { Style::Bold } else { Style::Accent }, name),
        (Style::Plain, "  ".into()),
        (Style::Plain, format!("{:<room$}", clip(&item.board.title, room))),
        (Style::Plain, "  ".into()),
        (Style::Dim, meta),
    ]
}

/// Markdown as the agent's page renders it (`markdown::render_lines`), set in
/// by two columns.
fn markdown(text: &str, width: usize) -> Vec<Line> {
    crate::markdown::render_lines(text, width.saturating_sub(3).max(10))
        .into_iter()
        .map(|line| vec![(Style::Raw, format!("  {line}"))])
        .collect()
}

fn notes_lines(item: &Item, pick: Option<usize>, width: usize) -> Vec<Line> {
    if item.notes.is_empty() {
        return vec![vec![(Style::Dim, "  No notes. In review, Note leaves one; s0ma answers each when the board comes back to it.".into())]];
    }
    let mut out = Vec::new();
    let mut round = 0;
    for (index, note) in item.notes.iter().enumerate() {
        if note.round != round {
            round = note.round;
            out.push(vec![(Style::Heading, format!("  round {round}"))]);
        }
        let lead = if pick == Some(index) { (Style::Cursor, "❯ ".to_string()) } else { (Style::Plain, "  ".to_string()) };
        // [?] waiting for the agent's answer; answered, [+].
        let (mark, style) = if note.answer.is_some() { ("[+]", Style::Success) } else { ("[?]", Style::Operator) };
        let mut lines = flow(
            &[lead, (style, format!("{mark} ")), (Style::Bold, format!("{}  ", note.id)), (Style::Plain, note.text.clone())],
            width,
            10,
        );
        match &note.answer {
            Some(answer) => lines.extend(flow(&[(Style::Dim, "          s0ma  ".into()), (Style::Plain, answer.clone())], width, 16)),
            None => lines.push(vec![(Style::Dim, "          s0ma has not answered yet".into())]),
        }
        // A click anywhere on the note picks it.
        out.extend(lines.into_iter().map(|line| pick_row(line, Click::Pick(Pick::Item(index)))));
    }
    out
}

fn checks_lines(item: &Item, width: usize) -> Vec<Line> {
    let flow = &item.flow;
    let row = |label: &str, value: Vec<(Style, String)>| -> Vec<Line> {
        let mut segments = vec![(Style::Dim, format!("  {label:<22}"))];
        segments.extend(value);
        super::flow(&segments, width, 24)
    };
    let mut out = Vec::new();
    // [+] passed, [-] failed, [*] at it.
    let verdict = |verdict: &str| -> (Style, String) {
        match verdict {
            "passed" => (Style::Success, "[+] passed".into()),
            "failed" => (Style::Problem, "[-] failed".into()),
            other => (Style::Warn, format!("[?] {other}")),
        }
    };
    let verifier = match (flow.verifier_state.as_deref(), &flow.verifier_job) {
        (Some("wanted"), _) => vec![(Style::Work, format!("[*] starting · {}", flow.verifier_note.as_deref().unwrap_or("when a worker is free")))],
        (Some("running"), Some(job)) => vec![(Style::Work, format!("[*] running · {} (its tab shows it working)", super::short_id(job)))],
        (Some("done"), _) => vec![verdict(crate::board::flow::verdict_of(flow.verifier_report.as_deref().unwrap_or("")))],
        (Some("failed"), _) => vec![(Style::Problem, format!("[-] {}", flow.verifier_note.clone().unwrap_or_else(|| "failed".into())))],
        _ => vec![(Style::Dim, "[*] not started".into())],
    };
    out.extend(row("independent verifier", verifier));
    out.extend(row(
        "s0ma's own check",
        match &flow.own_verdict {
            Some(own) => vec![verdict(own)],
            None => vec![(Style::Work, "[*] checking".into())],
        },
    ));
    out.extend(row(
        "reconciled report",
        match &flow.verdict {
            Some(reconciled) => vec![verdict(reconciled), (Style::Warn, "   [!] yours: a twice accepts it as complete, b sends it back".into())],
            None => vec![(Style::Dim, "[*] when both are in, s0ma reconciles them".into())],
        },
    ));
    out.push(Vec::new());
    out.push(vec![(Style::Dim, "  The verifier's report is held from s0ma until s0ma files its own check, so neither leans on the other.".into())]);
    out
}

fn section_lines(item: &Item, section: Section, pick: Option<usize>, width: usize) -> Vec<Line> {
    match section {
        Section::Reports => reports_lines(item, width),
        // Drawn by the share tab's browser (`with_files`) and the board's
        // tab (`embedded_board`), not here.
        Section::Files | Section::Board | Section::Audit => Vec::new(),
        Section::Notes => notes_lines(item, pick, width),
        Section::Checks => checks_lines(item, width),
        Section::Verification => match &item.flow.verification {
            Some(_) => markdown(&crate::board::flow::verification_text(&item.board.name, &item.flow), width),
            None => vec![vec![(Style::Dim, "  Not reconciled yet: see the checks.".into())]],
        },
    }
}

/// Every round's report, the latest first, each under where it is kept.
fn reports_lines(item: &Item, width: usize) -> Vec<Line> {
    let reports = item.flow.all_reports();
    if reports.is_empty() {
        return vec![vec![(Style::Dim, "  No reports yet. s0ma writes one when it finishes a round (board_report).".into())]];
    }
    let folder = crate::board::flow::folder(&item.board.name);
    let mut out = Vec::new();
    for (at, report) in reports.iter().rev().enumerate() {
        if at > 0 {
            out.push(Vec::new());
        }
        let latest = if at == 0 { " · latest" } else { "" };
        out.push(vec![
            (Style::Heading, format!("  round {}", report.round)),
            (Style::Dim, format!("{latest} · written {} · ~/s0ma-share/{folder}/reports/round-{}.md", ago(&report.at), report.round)),
        ]);
        out.extend(markdown(&report.text, width));
    }
    out
}

/// How many things the section has to pick from (files, notes).
fn pickable(item: &Item, section: Section) -> usize {
    match section {
        Section::Notes => item.notes.len(),
        _ => 0,
    }
}

fn input_line(input: &Input, width: usize) -> Line {
    let label = match input.typing {
        Typing::Note => " note ›",
        Typing::Why => " back to s0ma · why (optional) ›",
        Typing::WhyReview => " back to review · why (optional) ›",
    };
    let before: String = input.text[..input.cursor].iter().collect();
    let at: String = input.text.get(input.cursor).map(|c| c.to_string()).unwrap_or_else(|| " ".into());
    let after: String = input.text.iter().skip(input.cursor + 1).collect();
    let room = width.saturating_sub(text_width(label) + 2);
    let before = if text_width(&before) > room.saturating_sub(1) { before.chars().rev().take(room.saturating_sub(1)).collect::<Vec<_>>().into_iter().rev().collect() } else { before };
    vec![(Style::Operator, label.to_string()), (Style::Plain, format!(" {before}")), (Style::Cursor, at), (Style::Plain, after)]
}

pub(super) fn body(tab: &mut Open, width: usize, rows: usize) -> (Vec<Line>, Footer) {
    if let Ok(mut last) = LAST.lock() {
        *last = Some(tab.view.clone());
    }
    refresh(tab);
    let view = tab.view.clone();
    let mut out = vec![subtab_line(&view, width), Vec::new()];
    let back = vec![("tab", "subtab"), ("esc", "back")];
    let Some(read) = &tab.projects.read else {
        out.push(vec![(Style::Dim, "  Reading the boards…".into())]);
        return (out, Footer::new(Vec::new(), back));
    };
    if let Err(error) = &read.items {
        out.push(vec![(Style::Problem, format!("  the boards could not be read: {error}"))]);
        return (out, Footer::new(Vec::new(), back));
    }
    let all = items(tab);
    if all.is_empty() {
        let empty = match view {
            View::Review => "Nothing to review. A board s0ma finishes and reports on comes here.",
            View::Verification => "Nothing in verification. From review, Verify (twice) sends a board here.",
            _ => "Nothing complete yet. A verified board you accept comes here.",
        };
        out.push(vec![(Style::Dim, format!("  {empty}"))]);
        return (out, Footer::new(Vec::new(), back));
    }
    let index = selected(tab, &all).unwrap_or(0);
    let item = &all[index];
    let section = section(tab);

    // The list: up to a third of the rows.
    let list_rows = all.len().min((rows / 3).max(2));
    if index < tab.projects.top {
        tab.projects.top = index;
    } else if index >= tab.projects.top + list_rows {
        tab.projects.top = index + 1 - list_rows;
    }
    for (at, item) in all.iter().enumerate().skip(tab.projects.top).take(list_rows) {
        out.push(pick_row(list_row(item, &view, at == index, width), Click::Pick(Pick::Project(at))));
    }
    tab.projects.list_end = out.len();
    out.push(section_bar(tab, item, width));

    let input_rows = usize::from(tab.projects.input.is_some());
    let room = rows.saturating_sub(out.len() + input_rows).max(1);
    let shown = section_body(tab, item, width, room, out.len());
    out.extend(shown.lines);
    if let Some(input) = &tab.projects.input {
        out.resize(rows.saturating_sub(1), Vec::new());
        out.push(input_line(input, width));
    }
    out.truncate(rows);
    if tab.projects.input.is_some() {
        return (out, Footer::new(Vec::new(), vec![("⏎", "add"), ("esc", "cancel")]));
    }

    let mut hints = shown.hints;
    if !shown.viewing {
        hints.push(("←→", "section"));
    }
    if all.len() > 1 {
        hints.push(("^n ^p", "board"));
    }
    match view {
        View::Review if exhausted(&item.board) => hints.extend([("n", "note"), ("b", "to s0ma"), ("a", "close")]),
        View::Review => hints.extend([("n", "note"), ("b", "to s0ma"), ("v", "verify")]),
        View::Verification if item.flow.verification.is_some() => hints.extend([("a", "accept"), ("r", "to review"), ("b", "to s0ma")]),
        View::Verification => hints.extend([("r", "to review"), ("b", "to s0ma")]),
        View::Deliverables if item.board.status == BoardStatus::Exhausted => hints.extend([("r", "to review"), ("b", "to s0ma")]),
        View::Deliverables => hints.extend([("v", "verify"), ("r", "to review"), ("b", "to s0ma")]),
        _ => {}
    }
    if section == Section::Notes && pickable(item, section) > 0 && view == View::Review {
        hints.push(("x", "take back"));
    }
    if !shown.viewing {
        hints.push(("d", "discard"));
        hints.extend(back);
    }
    let mut left = vec![(Style::Status, format!("{} of {}", index + 1, all.len()))];
    left.extend(shown.left);
    (out, Footer::new(left, hints))
}

/// The open subtab, in a section other than its board: the open board's
/// files, notes and reports (`item`, from the board tab's read) under its
/// subtab line and section bar.
pub(super) fn open_body(tab: &mut Open, item: Option<&Item>, width: usize, rows: usize) -> (Vec<Line>, Footer) {
    let mut out = vec![subtab_line(&View::Board, width)];
    let Some(item) = item else {
        out.push(Vec::new());
        // Read again (after Ctrl+N), or none open.
        let none = matches!(tab.board_read.as_ref().map(|read| &read.snapshot), Some(Ok(None)));
        out.push(vec![(Style::Dim, if none { "  No open board." } else { "  Reading the board…" }.into())]);
        return (out, Footer::new(Vec::new(), vec![("tab", "subtab"), ("esc", "back")]));
    };
    out.push(section_bar(tab, item, width));
    let room = rows.saturating_sub(out.len()).max(1);
    let shown = section_body(tab, item, width, room, out.len());
    out.extend(shown.lines);
    out.truncate(rows);
    let mut hints = shown.hints;
    if !shown.viewing {
        hints.push(("←→", "section"));
        hints.extend([("tab", "subtab"), ("esc", "back")]);
    }
    let mut left = vec![(Style::Status, item.board.name.clone())];
    left.extend(shown.left);
    (out, Footer::new(left, hints))
}

/// The section bar: each section, how many it has, the one up marked.
pub(super) fn section_bar(tab: &Open, item: &Item, width: usize) -> Line {
    let sections = sections(tab);
    let up = tab.projects.section.min(sections.len() - 1);
    let mut bar: Line = vec![(Style::Dim, "  ── ".into())];
    for (at, each) in sections.iter().enumerate() {
        let count = match each {
            Section::Files if !item.flow.files.is_empty() => format!(" {}", item.flow.files.len()),
            Section::Notes if !item.notes.is_empty() => format!(" {}", item.notes.len()),
            Section::Reports if item.flow.all_reports().len() > 1 => format!(" {}", item.flow.all_reports().len()),
            Section::Audit => {
                let audits = crate::audit::ops::for_board(&item.board.name);
                let waiting = audits.iter().any(|a| a.status == crate::audit::Status::Proposed);
                format!("{}{}", if audits.len() > 1 { format!(" {}", audits.len()) } else { String::new() }, if waiting { " [!]" } else { "" })
            }
            _ => String::new(),
        };
        let text = format!("{}{count}", each.label());
        // A click on a section shows it.
        let look = if at == up { Style::Tab } else { Style::Dim };
        bar.push(super::gui_view::hot(look, format!(" {text} "), Click::Pick(Pick::Section(at))));
        bar.push((Style::Dim, "│".into()));
    }
    bar.pop();
    let used: usize = bar.iter().map(|(_, text)| text_width(text)).sum();
    bar.push((Style::Dim, format!(" {}", "─".repeat(width.saturating_sub(used + 2)))));
    bar
}

/// A section as drawn: its rows, its keys, what the footer says on the
/// left, and whether a file has the whole of it (its own keys then).
struct Shown {
    lines: Vec<Line>,
    hints: Vec<(&'static str, &'static str)>,
    left: Line,
    viewing: bool,
}

/// The section up, for `item`, in `room` rows starting at row `at` of the
/// tab's body.
fn section_body(tab: &mut Open, item: &Item, width: usize, room: usize, at: usize) -> Shown {
    let section = section(tab);
    tab.projects.files_at = at;
    match section {
        Section::Files if item.flow.files.is_empty() => Shown {
            lines: vec![vec![(Style::Dim, "  No files yet: they come with s0ma's report.".into())]],
            hints: Vec::new(),
            left: Vec::new(),
            viewing: false,
        },
        Section::Files => {
            let (lines, footer) = with_files(tab, &item.board, |tab| super::share_view::body(tab, width, room));
            let viewing = tab.projects.files.as_ref().is_some_and(|(_, state)| state.viewing());
            Shown { lines, hints: footer.hints, left: footer.left.into_iter().skip(1).collect(), viewing }
        }
        Section::Audit => {
            let (lines, footer) = super::audit_view::embedded_body(tab, &item.board.name, width, room);
            Shown { lines, hints: footer.hints, left: footer.left.into_iter().skip(1).collect(), viewing: false }
        }
        Section::Board => match &item.snapshot {
            Some(snapshot) => {
                let (lines, hints) = super::embedded_board(tab, snapshot, width, room, at);
                Shown { lines, hints, left: Vec::new(), viewing: false }
            }
            None => Shown { lines: vec![vec![(Style::Dim, "  Reading the board…".into())]], hints: Vec::new(), left: Vec::new(), viewing: false },
        },
        _ => {
            let count = pickable(item, section);
            tab.projects.pick = tab.projects.pick.min(count.saturating_sub(1));
            let pick = (count > 0).then_some(tab.projects.pick);
            let lines = section_lines(item, section, pick, width);
            // A picked note kept in sight.
            if pick.is_some()
                && let Some(marker) = lines.iter().position(|line| line.first().is_some_and(|(style, text)| *style == Style::Cursor && text == "❯ "))
            {
                if marker < tab.projects.scroll {
                    tab.projects.scroll = marker;
                } else if marker >= tab.projects.scroll + room {
                    tab.projects.scroll = marker + 1 - room;
                }
            }
            tab.projects.scroll = tab.projects.scroll.min(lines.len().saturating_sub(room));
            let more = lines.len().saturating_sub(tab.projects.scroll + room);
            let lines = lines.into_iter().skip(tab.projects.scroll).take(room).collect();
            let left = if more > 0 { vec![(Style::Dim, format!(" · {more} more below"))] } else { Vec::new() };
            Shown { lines, hints: vec![("↑↓", if count > 0 { "pick" } else { "scroll" })], left, viewing: false }
        }
    }
}

// ------------------------------------------------------------------ keys

/// A turn of the mouse wheel at `row` of the tab's body: in the files, as
/// in the share tab (the list moves, the file scrolls); elsewhere the
/// section scrolls.
pub(super) fn wheel(tab: &mut Open, up: bool, row: usize, page: usize) {
    let all = items(tab);
    let Some(index) = selected(tab, &all) else { return };
    // Over the boards: the one above or below.
    if row < tab.projects.list_end {
        let to = if up { index.saturating_sub(1) } else { (index + 1).min(all.len() - 1) };
        if to != index {
            tab.projects.selected = Some(all[to].board.id.clone());
        }
        return;
    }
    let item = all[index].clone();
    let section = section(tab);
    if section == Section::Audit {
        if row >= tab.projects.files_at {
            super::audit_view::wheel(tab, up);
        }
        return;
    }
    if section == Section::Board {
        if row >= tab.projects.files_at {
            super::board_wheel(tab, up, row, page);
        }
        return;
    }
    if section == Section::Files && !item.flow.files.is_empty() {
        if row >= tab.projects.files_at {
            let row = row - tab.projects.files_at;
            with_files(tab, &item.board, |tab| super::share_view::wheel(tab, up, row, page));
        }
        return;
    }
    tab.projects.scroll = if up { tab.projects.scroll.saturating_sub(super::WHEEL_ROWS) } else { tab.projects.scroll.saturating_add(super::WHEEL_ROWS) };
}

fn send(tab: &mut Open, board: &str, done: Move) {
    match act(Action::Project { board: board.to_string(), done }) {
        Ok(reply) => tab.note(reply, false),
        Err(reply) => tab.note(reply, true),
    }
    reread(tab);
}

/// The agent's word that nothing is left to try, waiting on the operator's.
fn exhausted(board: &Board) -> bool {
    board.status == BoardStatus::ProposedExhausted
}

/// `key` pressed on `board` once arms it (`say` tells what again does); the
/// second time it is done.
fn twice(tab: &mut Open, armed: Option<(char, String)>, key: char, board: &str, say: String) -> bool {
    if armed.as_ref().is_some_and(|(was, id)| *was == key && id == board) {
        return true;
    }
    tab.ask(say);
    tab.projects.armed = Some((key, board.to_string()));
    false
}

pub(super) fn key(tab: &mut Open, key: &KeyEvent, page: usize) -> Outcome {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let on_board = section(tab) == Section::Board && tab.projects.input.is_none();
    // The board section acts on the tree last drawn: a key that comes before
    // another board's is drawn (Ctrl+N, then at once Ctrl+C) does nothing.
    let drawn = tab.board_read.as_ref().and_then(|read| read.snapshot.as_ref().ok()?.as_ref()).map(|snapshot| snapshot.board.id.clone());
    let fresh = drawn.is_some() && drawn == tab.projects.selected;
    match key.code {
        // In the board section, Ctrl+C greys the selected row out (or back).
        KeyCode::Char('c' | 'C') if ctrl && on_board && (!fresh || super::board_grey(tab)) => return Outcome::Stay,
        KeyCode::Char('c' | 'C') if ctrl => return Outcome::Interrupt,
        KeyCode::Char('d' | 'D') if ctrl => return Outcome::Eof,
        _ => {}
    }
    if tab.projects.input.is_some() {
        typing(tab, key);
        return Outcome::Stay;
    }
    // Ctrl+N, Ctrl+P: the next board, the one before (in a section too).
    if ctrl && matches!(key.code, KeyCode::Char('n' | 'N' | 'p' | 'P')) {
        let all = items(tab);
        if let Some(index) = selected(tab, &all) {
            let to = if matches!(key.code, KeyCode::Char('n' | 'N')) { (index + 1) % all.len() } else { (index + all.len() - 1) % all.len() };
            tab.projects.selected = Some(all[to].board.id.clone());
        }
        return Outcome::Stay;
    }
    let viewing = section(tab) == Section::Files && tab.projects.files.as_ref().is_some_and(|(_, state)| state.viewing());
    // The audit section: the audit tab's own keys, plain ←→ aside.
    if section(tab) == Section::Audit
        && let Some(board) = current_board(tab)
        && (super::audit_view::holds_keys(tab) || !(key.modifiers.is_empty() && matches!(key.code, KeyCode::Left | KeyCode::Right)))
    {
        return super::audit_view::embedded_key(tab, &board, key);
    }
    if !viewing && section_step(tab, key) {
        return Outcome::Stay;
    }
    // The board section: walked as the board's own tab is.
    let walks = matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right | KeyCode::Home | KeyCode::End | KeyCode::PageUp | KeyCode::PageDown | KeyCode::Enter);
    if on_board && if fresh { super::board_section_key(tab, key, page) } else { walks } {
        return Outcome::Stay;
    }
    let armed = tab.projects.armed.take();
    let all = items(tab);
    let view = tab.view.clone();
    let index = selected(tab, &all);
    let item = index.map(|index| all[index].clone());
    let section = section(tab);
    let count = item.as_ref().map_or(0, |item| pickable(item, section));
    let picks = count > 0;
    // The files: browsed as in the share tab, keys it has taken (moving,
    // folding, viewing, m, e, s) before the projects tab's own.
    if section == Section::Files
        && let Some(item) = item.as_ref().filter(|item| !item.flow.files.is_empty())
        && let Some(outcome) = with_files(tab, &item.board, |tab| super::share_view::embedded_key(tab, key, page))
    {
        return outcome;
    }
    match key.code {
        KeyCode::Esc => return Outcome::Closed,
        KeyCode::Up if picks => tab.projects.pick = tab.projects.pick.saturating_sub(1),
        KeyCode::Down if picks => tab.projects.pick = (tab.projects.pick + 1).min(count - 1),
        KeyCode::Up => tab.projects.scroll = tab.projects.scroll.saturating_sub(1),
        KeyCode::Down => tab.projects.scroll += 1,
        KeyCode::Home if picks => tab.projects.pick = 0,
        KeyCode::End if picks => tab.projects.pick = count - 1,
        KeyCode::Home => tab.projects.scroll = 0,
        KeyCode::End => tab.projects.scroll = usize::MAX,
        KeyCode::PageUp => tab.projects.scroll = tab.projects.scroll.saturating_sub(page / 2),
        KeyCode::PageDown => tab.projects.scroll += page / 2,
        _ => {
            let Some(item) = item else { return Outcome::Stay };
            let id = item.board.id.clone();
            match (view, key.code) {
                (View::Review, KeyCode::Char('n')) => tab.projects.input = Some(Input { typing: Typing::Note, text: Vec::new(), cursor: 0 }),
                (_, KeyCode::Char('d') | KeyCode::Delete) => {
                    if twice(tab, armed, 'd', &id, format!("d again discards {}: it is closed, and leaves the projects tab", item.board.name)) {
                        send(tab, &id, Move::Discard);
                    }
                }
                (View::Review, KeyCode::Char('x')) if section == Section::Notes && picks => {
                    if let Some(note) = item.notes.get(tab.projects.pick) {
                        let id_note = note.id.clone();
                        if twice(tab, armed, 'x', &id, format!("x again takes back {id_note}")) {
                            send(tab, &id, Move::Unnote { id: id_note });
                        }
                    }
                }
                (View::Review, KeyCode::Char('b')) => {
                    let open = item.notes.iter().filter(|note| note.answer.is_none()).count();
                    let with = if open == 0 { "with no new notes".to_string() } else { format!("with your {}", plural(open, "note")) };
                    if twice(tab, armed, 'b', &id, format!("b again sends {} back to s0ma {with}", item.board.name)) {
                        send(tab, &id, Move::Back { why: String::new() });
                    }
                }
                (View::Review, KeyCode::Char('v')) if exhausted(&item.board) => {
                    tab.note("nothing to verify for an exhausted board: a closes it exhausted, b sends it back", true);
                }
                (View::Review, KeyCode::Char('a')) if exhausted(&item.board) => {
                    if twice(tab, armed, 'a', &id, format!("a again closes {} exhausted", item.board.name)) {
                        send(tab, &id, Move::Exhaust);
                    }
                }
                (View::Review, KeyCode::Char('v')) => {
                    if twice(tab, armed, 'v', &id, format!("v again sends {} to verification", item.board.name)) {
                        send(tab, &id, Move::Verify);
                    }
                }
                (View::Verification, KeyCode::Char('a')) => {
                    if item.flow.verification.is_none() {
                        tab.note("not reconciled yet: s0ma files the verification report when both checks are in", true);
                    } else if twice(tab, armed, 'a', &id, format!("a again accepts {} as complete ({})", item.board.name, item.flow.verdict.as_deref().unwrap_or("?"))) {
                        send(tab, &id, Move::Accept);
                    }
                }
                // Back to any stage before: its board (the agent works on it
                // again), review, and from the deliverables verification.
                (View::Verification | View::Deliverables, KeyCode::Char('b')) => {
                    tab.projects.input = Some(Input { typing: Typing::Why, text: Vec::new(), cursor: 0 });
                }
                (View::Verification | View::Deliverables, KeyCode::Char('r')) => {
                    tab.projects.input = Some(Input { typing: Typing::WhyReview, text: Vec::new(), cursor: 0 });
                }
                (View::Deliverables, KeyCode::Char('v')) if matches!(item.board.status, BoardStatus::Exhausted) => {
                    tab.note("nothing to verify for an exhausted board: r takes it back to review, b to its board", true);
                }
                (View::Deliverables, KeyCode::Char('v'))
                    if twice(tab, armed, 'v', &id, format!("v again sends {} to verification again", item.board.name)) => {
                        send(tab, &id, Move::Verify);
                    }
                _ => {}
            }
        }
    }
    Outcome::Stay
}

/// A key in the open subtab, unless its board section is up (the board's
/// own keys then, `None`; ←→ still step the sections while no hint is being
/// typed): the files, the notes and the reports of `item`, the open board.
pub(super) fn open_key(tab: &mut Open, item: Option<&Item>, key: &KeyEvent, page: usize) -> Option<Outcome> {
    let viewing = section(tab) == Section::Files && tab.projects.files.as_ref().is_some_and(|(_, state)| state.viewing());
    if section(tab) == Section::Board {
        // With a hint being typed, ←→ move through it.
        return (tab.draft.is_empty() && section_step(tab, key)).then_some(Outcome::Stay);
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('c' | 'C') if ctrl => return Some(Outcome::Interrupt),
        KeyCode::Char('d' | 'D') if ctrl => return Some(Outcome::Eof),
        // The boards: Ctrl+N and Ctrl+P, as in its board section.
        KeyCode::Char('n' | 'N' | 'p' | 'P') if ctrl => return None,
        _ => {}
    }
    if section(tab) == Section::Audit
        && let Some(board) = current_board(tab)
        && (super::audit_view::holds_keys(tab) || !(key.modifiers.is_empty() && matches!(key.code, KeyCode::Left | KeyCode::Right)))
    {
        return Some(super::audit_view::embedded_key(tab, &board, key));
    }
    if !viewing && section_step(tab, key) {
        return Some(Outcome::Stay);
    }
    let item = item?;
    let section = section(tab);
    if section == Section::Files
        && !item.flow.files.is_empty()
        && let Some(outcome) = with_files(tab, &item.board, |tab| super::share_view::embedded_key(tab, key, page))
    {
        return Some(outcome);
    }
    let count = pickable(item, section);
    let picks = count > 0;
    match key.code {
        KeyCode::Esc => return Some(Outcome::Closed),
        KeyCode::Up if picks => tab.projects.pick = tab.projects.pick.saturating_sub(1),
        KeyCode::Down if picks => tab.projects.pick = (tab.projects.pick + 1).min(count - 1),
        KeyCode::Up => tab.projects.scroll = tab.projects.scroll.saturating_sub(1),
        KeyCode::Down => tab.projects.scroll += 1,
        KeyCode::Home if picks => tab.projects.pick = 0,
        KeyCode::End if picks => tab.projects.pick = count - 1,
        KeyCode::Home => tab.projects.scroll = 0,
        KeyCode::End => tab.projects.scroll = usize::MAX,
        KeyCode::PageUp => tab.projects.scroll = tab.projects.scroll.saturating_sub(page / 2),
        KeyCode::PageDown => tab.projects.scroll += page / 2,
        _ => {}
    }
    Some(Outcome::Stay)
}

/// The wheel in the open subtab, outside its board section.
pub(super) fn open_wheel(tab: &mut Open, item: Option<&Item>, up: bool, row: usize, page: usize) {
    let Some(item) = item else { return };
    if section(tab) == Section::Files {
        if !item.flow.files.is_empty() && row >= tab.projects.files_at {
            let row = row - tab.projects.files_at;
            with_files(tab, &item.board, |tab| super::share_view::wheel(tab, up, row, page));
        }
        return;
    }
    tab.projects.scroll = if up { tab.projects.scroll.saturating_sub(super::WHEEL_ROWS) } else { tab.projects.scroll.saturating_add(super::WHEEL_ROWS) };
}

/// A click on a board in the list: it is selected.
pub(super) fn pick(tab: &mut Open, index: usize) {
    if tab.projects.input.is_some() {
        return;
    }
    let all = items(tab);
    if let Some(item) = all.get(index)
        && tab.projects.selected.as_deref() != Some(item.board.id.as_str())
    {
        tab.projects.selected = Some(item.board.id.clone());
        tab.projects.scroll = 0;
        tab.projects.pick = 0;
    }
}

/// A click on a section's name: it is shown.
pub(super) fn pick_section(tab: &mut Open, index: usize) {
    if index < sections(tab).len() && tab.projects.section != index {
        tab.projects.section = index;
        tab.projects.scroll = 0;
        tab.projects.pick = 0;
    }
}

/// A click on a note: it is picked.
pub(super) fn pick_item(tab: &mut Open, index: usize) {
    tab.projects.pick = index;
}

/// A click on a file in a board's files: picked in its browser.
pub(super) fn pick_file(tab: &mut Open, path: String) {
    let item = if tab.view == View::Board {
        super::open_item(tab)
    } else {
        let all = items(tab);
        selected(tab, &all).and_then(|index| all.get(index).cloned())
    };
    if let Some(item) = item {
        with_files(tab, &item.board, |tab| super::share_view::pick(tab, path));
    }
}

fn typing(tab: &mut Open, key: &KeyEvent) {
    let Some(input) = tab.projects.input.as_mut() else { return };
    match key.code {
        KeyCode::Esc => tab.projects.input = None,
        KeyCode::Enter => {
            let input = tab.projects.input.take().expect("typed");
            let text: String = input.text.iter().collect::<String>().trim().to_string();
            let Some(id) = tab.projects.selected.clone() else { return };
            match input.typing {
                Typing::Why => send(tab, &id, Move::Back { why: text }),
                Typing::WhyReview => send(tab, &id, Move::Review { why: text }),
                Typing::Note if text.is_empty() => {}
                Typing::Note => send(tab, &id, Move::Note { text }),
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
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            input.text.drain(..input.cursor);
            input.cursor = 0;
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            input.text.insert(input.cursor, c);
            input.cursor += 1;
        }
        _ => {}
    }
}

pub(super) fn paste(tab: &mut Open, text: &str) {
    if let Some(input) = tab.projects.input.as_mut() {
        for c in text.chars().map(|c| if c == '\n' { ' ' } else { c }).filter(|c| !c.is_control()) {
            input.text.insert(input.cursor, c);
            input.cursor += 1;
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::board::{BoardStatus, GoalTestKind};
    use crate::board::flow::Delivered;

    pub(crate) fn item(stage: Stage) -> Item {
        let board = Board {
            id: "board-0002".into(),
            name: "self-model-drift".into(),
            title: "The self-model drifts after sleep".into(),
            origin: "s0ma noticed".into(),
            goal: "Know why".into(),
            goal_test: "The operator agrees".into(),
            goal_test_kind: GoalTestKind::Judgment,
            status: BoardStatus::ProposedSolved,
            step_budget: 20,
            steps_used: 9,
            created_at: chrono::Local::now().to_rfc3339(),
            closed_at: None,
            outcome: None,
        };
        let flow = Flow {
            board_id: board.id.clone(),
            name: board.name.clone(),
            stage: Some(stage),
            round: 1,
            report: Some("# What was done\nPinned the want-phrases.\n\n## How it was checked\nTwo sleeps.".into()),
            report_round: 1,
            report_at: Some(chrono::Local::now().to_rfc3339()),
            files: vec![Delivered { path: "home/s0ma/identity.md".into(), bytes: 2048 }, Delivered { path: "home/s0ma/src/fix.rs".into(), bytes: 120 }],
            entered_at: chrono::Local::now().to_rfc3339(),
            ..Flow::default()
        };
        let notes = vec![Note {
            id: "n1".into(),
            board_id: board.id.clone(),
            round: 1,
            text: "Does it hold after two sleeps?".into(),
            created_at: chrono::Local::now().to_rfc3339(),
            answer: None,
            answered_at: None,
        }];
        Item { board, flow, notes, snapshot: None }
    }

    pub(crate) fn tab_with(view: View, items: Vec<Item>) -> Open {
        let mut tab = Open::new(view);
        tab.projects.read = Some(Read { at: Instant::now(), items: Ok(items) });
        tab
    }

    /// What the lines say, without the colour codes of rendered markdown.
    fn text(lines: &[Line]) -> String {
        let all = lines.iter().map(|line| line.iter().map(|(_, text)| text.as_str()).collect::<String>()).collect::<Vec<_>>().join("\n");
        let mut plain = String::new();
        let mut chars = all.chars();
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

    fn press(tab: &mut Open, code: KeyCode) -> Outcome {
        key(tab, &KeyEvent::new(code, KeyModifiers::NONE), 10)
    }

    #[test]
    fn an_exhaustion_in_review_is_closed_with_a_twice_not_verified() {
        let mut exhausted = item(Stage::Review);
        exhausted.board.status = BoardStatus::ProposedExhausted;
        let mut tab = tab_with(View::Review, vec![exhausted]);
        let (lines, footer) = body(&mut tab, 120, 30);
        assert!(text(&lines).contains("nothing left to try, says s0ma"), "{}", text(&lines));
        assert!(footer.hints.contains(&("a", "close")) && !footer.hints.contains(&("v", "verify")));
        press(&mut tab, KeyCode::Char('v'));
        assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| *problem && text.contains("nothing to verify")), "{:?}", tab.note);
        press(&mut tab, KeyCode::Char('a'));
        assert!(tab.note.as_ref().is_some_and(|(text, _, _)| text.starts_with("[?] closes self-model-drift exhausted · again to confirm")), "{:?}", tab.note);
        press(&mut tab, KeyCode::Char('a'));
        assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| *problem && text.contains("not available")), "sent: {:?}", tab.note);
    }
}
