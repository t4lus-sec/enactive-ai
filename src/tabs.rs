//! Tabs. The chats are on Alt: Alt+1 is the agent's conversation, Alt+2…Alt+9
//! (and Alt+0) its subagents; Alt+B goes back to the tab before, as it was
//! left. The terminals are on
//! Alt+Shift: Alt+Shift+1… the shells on the agent's machine (`vm_terms`), the
//! netboxes first. Alt and Alt+Shift arrive as
//! themselves in every terminal (Esc before the key), GNOME Terminal and tmux
//! included, where Ctrl and Ctrl+Shift with a digit do not.
//!
//! The agent's own page is on the alternate screen with the tab bar on top
//! (`page.rs`). A tab other than the agent's takes the whole screen in its place
//! and draws itself live: a subagent's reasoning, text, tool calls and results
//! as they happen (`worker_feed`), the open board as it changes, a shell as it
//! is. The agent's display holds what it writes
//! meanwhile, and draws its page again from its record when Alt+1 or Esc
//! brings it back (`ActiveDisplay::cover`). Either way the agent keeps working: a
//! turn can start, run and end behind a tab. PgUp or the wheel on the agent's page
//! opens its own conversation further back (`View::History`), drawn the same
//! way.
//!
//! A subagent's tab has a prompt: what is typed there goes to that subagent
//! as a direction from the operator, taken before its next step
//! (`subagent::steer_from_operator`); `/wrap` asks it to wrap up and report,
//! `/kill` stops it. On the board's tab the prompt leaves a hint. In a shell's
//! tab every key goes to the shell, Esc and Ctrl+C included; Shift+PgUp/PgDn
//! and the mouse wheel scroll it. Every tab scrolls with the wheel; on the
//! board and in the share tab it walks the list, or scrolls the strip under
//! it when turned over the strip.
//!
//! The keys are read by whichever thread has the keyboard: the idle prompt's
//! (`InputReader::run_idle_tab`), or a turn's keyboard worker. Only one of
//! them reads at a time, and everything here is behind one lock.

use crate::line_edit::Draft;
use crate::style::{Line, Style, apply_style};
use crate::terminal::ActiveDisplay;
use crate::subagent::feed::{Feed, Item, Sender};

mod board_view;
mod memory_view;
mod share_view;
mod overview_view;
mod projects_view;
mod editor_view;
mod select_view;
mod gui_view;
mod audit_view;
mod claim_view;
mod draw;
mod subagent_tab;
mod board_tab;
mod history_tab;
mod shell_tab;
pub(crate) use draw::*;
use subagent_tab::*;
use board_tab::*;
use history_tab::*;
use shell_tab::*;
pub use memory_view::{Section as MemorySection, init as init_memory, start_at as memory_start_at};

use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind},
    queue,
    style::Print,
    terminal::{Clear, ClearType},
};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use unicode_width::UnicodeWidthChar;

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// A tab is drawn again at most this often while what it shows keeps
/// changing (a subagent streaming), and at least once a second for its clocks.
const FRAME: Duration = Duration::from_millis(80);
const CLOCK: Duration = Duration::from_secs(1);
/// The board is read again at most this often.
const BOARD_POLL: Duration = Duration::from_secs(1);
/// A board that cannot be read for this long says so; before, what was read
/// last stays up.
const BOARD_GIVE_UP: Duration = Duration::from_secs(5);
/// How long "sent" and the like stay on the status line (the feed says the
/// rest: a direction is marked while it waits).
const NOTE_FOR: Duration = Duration::from_secs(4);
/// How long a key or button pressed once on something that asks first (a
/// discard, a close, an accept) waits for the second press; its `[?]` note
/// shows as long. After it, the next press asks again.
const ARM_FOR: Duration = Duration::from_secs(10);
/// Collapsed (Ctrl+O shows all of it): this many lines of the task.
const TASK_LINES: usize = 6;
/// Expanded, a tool result still stops here.
const RESULT_LINES_EXPANDED: usize = 400;
/// How soon a second Ctrl+Q has to come to close a busy shell.
const CLOSE_AGAIN: Duration = Duration::from_secs(3);
/// What `/wrap` alone sends.
const WRAP_UP: &str = "Wrap up now: finish or safely stop what is in flight and report what you have.";
/// Rows a turn of the mouse wheel scrolls.
pub const WHEEL_ROWS: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum View {
    /// The tab with this number: 2 is the first subagent, in slot order.
    Subagent(usize),
    Board,
    /// Alt+Shift+1…: a shell on the agent's machine, the netboxes first. `number`
    /// is its key; `session` the shell, once there is one on that key. Kept by name, so keys typed here never
    /// go to another shell that has taken its place.
    Shell { number: usize, session: Option<String> },
    /// The agent's own conversation, further back than the page shows: PgUp or
    /// the wheel on the page. Scrolled back down to the end, it is the page
    /// again.
    History,
    /// Alt+M: the archive, threads, world model and redactions
    /// (`tabs/memory_view.rs`).
    Memory,
    /// Alt+S: /share, the folder the operator and the agent work on together
    /// (`tabs/share_view.rs`).
    Share,
    /// Alt+O: the agent's settings and state, MCP's servers among them
    /// (`tabs/overview_view.rs`).
    Overview,
    /// The projects tab's later subtabs (Alt+P steps through them, the board
    /// being the first): boards in the operator's review, in verification, and
    /// delivered (`tabs/projects_view.rs`, `board/flow.rs`).
    Review,
    Verification,
    Deliverables,
    /// Alt+E: files of the shared folder, in the agent's own editor
    /// (`tabs/editor_view.rs`).
    Edit,
    /// Alt+U: the audits, attached to projects and standing
    /// (`tabs/audit_view.rs`, `tabs/claim_view.rs`).
    Audit,
    /// The agent's own page in a pane of its own, while two are up (Alt+\):
    /// the conversation as the page draws it, following the end, and a prompt
    /// to the agent (a new turn, or steering one under way).
    Page,
}

impl View {
    /// A chat's tab: it has a prompt.
    fn has_prompt(&self) -> bool {
        matches!(self, View::Subagent(_) | View::Board | View::Page)
    }

    fn same(&self, other: &View) -> bool {
        match (self, other) {
            (View::Shell { session: Some(a), .. }, View::Shell { session: Some(b), .. }) => a == b,
            (View::Shell { number: a, .. }, View::Shell { number: b, .. }) => a == b,
            (a, b) => a == b,
        }
    }
}

/// Where a key leads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Main,
    View(View),
    /// Alt+B: back to the tab up before, as it was.
    Back,
}

/// Which keys: Alt's (the chats) or Alt+Shift's (the terminals).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Chat,
    Terminal,
}

/// Alt+1 → the agent, Alt+2…9 → the subagents, Alt+0 → the tenth tab;
/// Alt+Shift+1…9 and 0 → the shells, the netboxes first; B → back to the tab
/// up before, as it was.
pub fn target(group: Group, key: char) -> Option<Target> {
    let number = match key {
        '1'..='9' => key as usize - '0' as usize,
        '0' => 10,
        'b' | 'B' => return Some(Target::Back),
        'p' | 'P' => return Some(Target::View(projects_view::last())),
        'm' | 'M' => return Some(Target::View(View::Memory)),
        's' | 'S' => return Some(Target::View(View::Share)),
        'o' | 'O' => return Some(Target::View(View::Overview)),
        'e' | 'E' => return Some(Target::View(View::Edit)),
        'u' | 'U' => return Some(Target::View(View::Audit)),
        _ => return None,
    };
    Some(match (group, number) {
        (Group::Chat, 1) => Target::Main,
        (Group::Chat, number) => Target::View(View::Subagent(number)),
        (Group::Terminal, number) => {
            Target::View(View::Shell { number, session: crate::vm::terms::sessions().get(number - 1).cloned() })
        }
    })
}

/// The key that switches tabs, if `key` is one: Alt+digit (a chat), Alt+B
/// (back), or Alt+Shift+digit (a terminal). Most terminals send
/// Alt+Shift+1 as Alt and the shifted character, `!`; terminals with the
/// kitty keyboard protocol as Alt+Shift and the digit. Both are taken, the
/// shifted characters as a US keyboard has them (`SHIFTED_DIGITS`).
pub fn switch_key(key: &KeyEvent) -> Option<(Group, char)> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    if !alt || ctrl {
        return None;
    }
    match key.code {
        KeyCode::Char(c @ '0'..='9') if shift => Some((Group::Terminal, c)),
        KeyCode::Char(c @ '0'..='9') => Some((Group::Chat, c)),
        KeyCode::Char('b' | 'B') => Some((Group::Chat, 'b')),
        KeyCode::Char('m' | 'M') => Some((Group::Chat, 'm')),
        KeyCode::Char('s' | 'S') => Some((Group::Chat, 's')),
        KeyCode::Char('o' | 'O') => Some((Group::Chat, 'o')),
        KeyCode::Char('e' | 'E') => Some((Group::Chat, 'e')),
        KeyCode::Char('p' | 'P') => Some((Group::Chat, 'p')),
        KeyCode::Char('u' | 'U') => Some((Group::Chat, 'u')),
        KeyCode::Char(c) => shifted_digit(c).map(|digit| (Group::Terminal, digit)),
        _ => None,
    }
}

/// Shift+1…Shift+0 on a US keyboard.
pub const SHIFTED_DIGITS: [(char, char); 10] =
    [('!', '1'), ('@', '2'), ('#', '3'), ('$', '4'), ('%', '5'), ('^', '6'), ('&', '7'), ('*', '8'), ('(', '9'), (')', '0')];

/// The digit whose key `c` is shifted, on a US keyboard.
fn shifted_digit(c: char) -> Option<char> {
    SHIFTED_DIGITS.iter().find(|(shifted, _)| *shifted == c).map(|(_, digit)| *digit)
}

/// What a key in a tab leaves the caller to do.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Stay,
    /// Typed to the agent in a pane that shows its page: a new turn, or steering
    /// the one under way (`now`: taken at the next steering point).
    Submit { text: String, now: bool },
    /// Back to the agent: the caller uncovers its display.
    Closed,
    /// Ctrl+C with nothing typed: what it does in the agent's own pane.
    Interrupt,
    /// Ctrl+D.
    Eof,
}

/// The board as last read (`None`: no board is open), which board, and how
/// many are open.
struct BoardRead {
    at: Instant,
    snapshot: Result<Option<board_view::Snapshot>, String>,
    name: Option<String>,
    /// Boards shown: those waiting for the operator (proposed solved) first, then
    /// the open ones.
    open: usize,
    waiting: usize,
}

struct Open {
    view: View,
    /// The tab up before this one, as it was (its subtab, section,
    /// selection, scroll): Alt+B goes back to it.
    previous: Option<Box<Open>>,
    draft: Draft,
    /// Rows scrolled back from the end; 0 follows the end.
    back: usize,
    /// Ctrl+O: reasoning, results and the task in full.
    expanded: bool,
    /// A line for the status bar, and whether it reports a problem.
    note: Option<(String, bool, Instant)>,
    /// Which open board (Tab steps through them).
    board: usize,
    /// How many boards the subtab had when last read (for stepping while
    /// the next read is on its way).
    boards_known: usize,
    board_read: Option<BoardRead>,
    /// The next read of the board, on a thread of its own: for which board.
    board_reading: Option<(usize, std::sync::Arc<Mutex<Option<BoardRead>>>)>,
    /// Since when reads of the board have failed, while what was read before
    /// stays up.
    board_failing: Option<Instant>,
    /// Delete pressed once on this note: again takes it off the board.
    board_unhint: Option<String>,
    /// What the board's prompt adds (Ctrl+T steps through them).
    board_adding: Adding,
    /// A busy shell's tab: Ctrl+Q pressed once, for that shell; the next
    /// press within `CLOSE_AGAIN` closes it anyway.
    close_armed: Option<(String, Instant)>,
    /// What is on the screen, row by row, so only what changed is written.
    frame: Vec<Line>,
    painted: Option<(u64, (u16, u16), Instant)>,
    /// A key or a resize: draw now, whatever the throttle says.
    dirty: bool,
    /// The subagent a draft is for: the one this tab showed while the draft
    /// was empty. Tabs follow the slots, so another can take this tab's
    /// place; what was typed for one is never sent to another.
    pinned: Option<String>,
    /// The conversation's rows as last drawn for `View::History`.
    history: Option<HistoryRows>,
    /// The board's tree: what is folded, which node is selected (by id), and
    /// the first row shown.
    folds: board_view::Folds,
    board_selected: Option<String>,
    board_top: usize,
    /// The board's tree as last drawn: its rows' node ids, in order, and how
    /// many rows it had (the wheel scrolls it, then moves the selection).
    board_shown: Vec<String>,
    board_tree_rows: usize,
    /// How far the strip under the board's tree is scrolled, and for which
    /// selection: another selection starts at its top.
    strip_scroll: usize,
    strip_for: Option<String>,
    /// The row of the tab's body where that strip starts (its rule), as
    /// last drawn: the wheel below it scrolls the strip.
    strip_at: usize,
    /// The memory tab's sections, lists, strip and marks.
    memory: memory_view::State,
    /// The share tab's listing, selection and work.
    share: share_view::State,
    /// The overview's latest snapshot and what is being typed.
    overview: overview_view::State,
    /// The projects tab's review, verification and deliverables.
    projects: projects_view::State,
    /// The audit tab: which audit, its section, selection, prompt and the
    /// claim view (`tabs/audit_view.rs`).
    audit: audit_view::State,
    /// Text being selected with the mouse, or just copied (`tabs/select_view.rs`).
    selection: Option<select_view::Selection>,
    /// Where the clickable parts of the frame drawn are (`tabs/gui_view.rs`), and
    /// what its footer's buttons do (a right-click's menu offers them).
    hits: Vec<gui_view::Hit>,
    actions: Vec<(String, gui_view::Click)>,
    /// A right-click's menu, while it is open.
    menu: Option<gui_view::Menu>,
    /// Clicks on rows, to tell a double-click.
    clicks: gui_view::Clicks,
    /// Where the left button went down, on what: a click is done when it
    /// comes up there (an item of a menu: `Click::Pick` of none, by row).
    pressed: Option<(usize, usize, Option<gui_view::Click>)>,
    /// The size this tab was last composed for (a pane's, split).
    size: (u16, u16),
    /// When a press last asked to be pressed again (`ask`): past `ARM_FOR`,
    /// nothing armed then is.
    asked_at: Option<Instant>,
}

/// The conversation as `View::History` shows it, and what it was drawn
/// from: the record's revision, the width, the other detail mode or not.
struct HistoryRows {
    from: (u64, usize, bool),
    rows: Vec<Line>,
    built: Instant,
}

/// The conversation as History shows it is laid out again at most this often.
const HISTORY_EVERY: Duration = Duration::from_secs(1);

impl Open {
    fn new(view: View) -> Self {
        Open {
            view,
            previous: None,
            draft: Draft::default(),
            back: 0,
            expanded: false,
            note: None,
            board: 0,
            boards_known: 1,
            board_read: None,
            board_reading: None,
            board_failing: None,
            board_unhint: None,
            board_adding: Adding::Hint,
            close_armed: None,
            frame: Vec::new(),
            painted: None,
            dirty: true,
            pinned: None,
            history: None,
            folds: board_view::Folds::default(),
            board_selected: None,
            board_top: 0,
            board_shown: Vec::new(),
            board_tree_rows: 0,
            strip_scroll: 0,
            strip_for: None,
            strip_at: usize::MAX,
            memory: memory_view::State::default(),
            share: share_view::State::default(),
            overview: overview_view::State::default(),
            projects: projects_view::State::default(),
            audit: audit_view::State::default(),
            selection: None,
            hits: Vec::new(),
            actions: Vec::new(),
            menu: None,
            clicks: gui_view::Clicks::default(),
            pressed: None,
            size: (80, 24),
            asked_at: None,
        }
    }

    /// A problem, in red where the footer is. What a key just did is not
    /// said: the tab shows it (`problem` false is nothing).
    /// What it did clears a question it answered ("… again does it").
    fn note(&mut self, text: impl Into<String>, problem: bool) {
        self.note = problem.then(|| (text.into(), true, Instant::now()));
    }

    /// A key that wants pressing again: what it will do, `[?]` in teal.
    /// Said without the key (`gui_view::again`): the key or the button,
    /// pressed again, does it.
    fn ask(&mut self, text: impl Into<String>) {
        self.note = Some((format!("[?] {}", gui_view::again(&text.into())), false, Instant::now()));
        self.asked_at = Some(Instant::now());
    }

    /// What was pressed once to be pressed again, forgotten when it has
    /// waited past `ARM_FOR`: a second press long after, its question gone
    /// from the footer, asks again rather than doing it.
    fn disarm_if_stale(&mut self, now: Instant) {
        if self.asked_at.is_some_and(|at| now.duration_since(at) > ARM_FOR) {
            self.asked_at = None;
            self.board_unhint = None;
            self.projects.armed = None;
            self.audit.armed = None;
            self.overview.clearing = false;
            self.close_armed = None;
        }
    }

    /// An answer the operator asked for with a key (where a shared file is on his
    /// machine), plain.
    fn tell(&mut self, text: impl Into<String>) {
        self.note = Some((text.into(), false, Instant::now()));
    }
}

static OPEN: Mutex<Option<Open>> = Mutex::new(None);
static ARCHIVE: OnceLock<PathBuf> = OnceLock::new();

fn lock() -> MutexGuard<'static, Option<Open>> {
    OPEN.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Where the board's database is (beside the archive); and the panic hook
/// (`crash.rs`), installed once with it.
pub fn init(archive: &Path) {
    if ARCHIVE.set(archive.to_path_buf()).is_err() {
        return;
    }
    crate::crash::install(archive);
}

pub fn is_open() -> bool {
    lock().is_some()
}

// ------------------------------------------------------------------ two panes

/// Two panes side by side (Alt+\, or Ctrl+Shift+P): `OPEN` is the one with the
/// keyboard, `other` the one beside it, each as any tab is (the agent's page among
/// them, `View::Page`). Alt+← and Alt+→ (Ctrl+1, Ctrl+2) give the keyboard to the left one or
/// the right; Alt+\ again keeps only the one that has it.
struct Split {
    other: Open,
    /// 0: the pane with the keyboard is the left one; 1: the right.
    focus: usize,
    /// The whole screen as last painted, both panes.
    frame: Vec<Line>,
    painted: Option<((u16, u16), Instant)>,
}

/// Taken after `OPEN`, never before.
static SPLIT: Mutex<Option<Split>> = Mutex::new(None);

fn split_lock() -> MutexGuard<'static, Option<Split>> {
    SPLIT.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Whether two panes are up: kept beside `SPLIT`, so a view drawn while it
/// is held can ask.
static SPLIT_UP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn is_split() -> bool {
    SPLIT_UP.load(std::sync::atomic::Ordering::Relaxed)
}

fn set_split_up(up: bool) {
    SPLIT_UP.store(up, std::sync::atomic::Ordering::Relaxed);
}

/// One pane or two: Alt+\ (Alt+| too), which every terminal sends; and
/// Ctrl+Shift+P, where the kitty keyboard protocol tells it from Ctrl+P.
pub fn is_split_key(key: &KeyEvent) -> bool {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    (alt && !ctrl && matches!(key.code, KeyCode::Char('\\' | '|')))
        || (ctrl && shift && !alt && matches!(key.code, KeyCode::Char('p' | 'P')))
}

/// The pane to give the keyboard to (0 left, 1 right), while two are up:
/// Alt+← or Alt+→, which every terminal sends (in a prompt they move by a
/// word only while one pane is up; Ctrl+← and Ctrl+→ always do); Ctrl+1 or
/// Ctrl+2 where the kitty keyboard protocol sends them.
fn focus_key(key: &KeyEvent) -> Option<usize> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    match key.code {
        KeyCode::Left if alt && !ctrl => Some(0),
        KeyCode::Right if alt && !ctrl => Some(1),
        KeyCode::Char('1') if ctrl && !alt => Some(0),
        KeyCode::Char('2') if ctrl && !alt => Some(1),
        _ => None,
    }
}

/// The columns of the left pane and of the right, the `│` between them.
fn pane_widths(columns: usize) -> (usize, usize) {
    let left = columns.saturating_sub(1) / 2;
    (left, columns.saturating_sub(left + 1))
}

/// Alt+\ (or Ctrl+Shift+P) on the agent's page (no tab up): two panes, the page in each, the
/// right one with the keyboard. `idle_prompt`: the idle prompt's draft, kept
/// in the left one's.
pub fn split_from_page(display: &ActiveDisplay, idle_prompt: Option<(&str, usize)>) -> io::Result<Outcome> {
    let mut open = lock();
    if open.is_some() {
        let mut split = split_lock();
        return toggle_split(&mut open, &mut split, display);
    }
    display.cover(idle_prompt)?;
    let mut page = Open::new(View::Page);
    if let Some((draft, _)) = idle_prompt {
        page.draft.insert_str(draft);
    }
    let (focused, state) = two_panes(page);
    *open = Some(focused);
    let mut split = split_lock();
    *split = Some(state);
    set_split_up(true);
    draw_any(&mut open, &mut split, display)
}

/// The pane put aside when two became one (Alt+\), as it was, and the side
/// it was on: two again bring it back there.
static PARKED: Mutex<Option<(Open, usize)>> = Mutex::new(None);

fn parked() -> MutexGuard<'static, Option<(Open, usize)>> {
    PARKED.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Two panes around `current` (what is up now): the pane put aside last
/// time back on its side, as it was, `current` on the other with the
/// keyboard; or, the first time, `current` on the left and the agent's page on
/// the right, with the keyboard. The pane with the keyboard, and the split.
fn two_panes(current: Open) -> (Open, Split) {
    match parked().take() {
        Some((mut aside, side)) => {
            aside.frame.clear();
            aside.painted = None;
            aside.dirty = true;
            (current, Split { other: aside, focus: 1 - side, frame: Vec::new(), painted: None })
        }
        None => (Open::new(View::Page), Split { other: current, focus: 1, frame: Vec::new(), painted: None }),
    }
}

/// One pane, or two. Back to one: the pane with the keyboard (the agent's page
/// alone is the page itself: `Closed`, the caller uncovers it), the other
/// put aside as it is. Two: that pane back on its side, as it was, beside
/// the tab up now, which keeps the keyboard (`two_panes`).
fn toggle_split(open: &mut MutexGuard<'static, Option<Open>>, split: &mut MutexGuard<'static, Option<Split>>, display: &ActiveDisplay) -> io::Result<Outcome> {
    if let Some(state) = split.take() {
        set_split_up(false);
        // The pane beside, kept as it is for two panes again.
        *parked() = Some((state.other, 1 - state.focus));
        if open.as_ref().is_some_and(|tab| tab.view == View::Page) {
            open.take();
            return Ok(Outcome::Closed);
        }
        if let Some(tab) = open.as_mut() {
            tab.frame.clear();
            tab.painted = None;
            tab.dirty = true;
        }
        return draw(open, display);
    }
    let Some(mut current) = open.take() else { return Ok(Outcome::Stay) };
    if current.view == View::History {
        current = Open { back: current.back, expanded: current.expanded, ..Open::new(View::Page) };
    }
    current.frame.clear();
    current.painted = None;
    current.dirty = true;
    let (focused, state) = two_panes(current);
    **open = Some(focused);
    **split = Some(state);
    set_split_up(true);
    draw_any(open, split, display)
}

/// The keyboard to the pane on `side` (0 left, 1 right).
fn focus(open: &mut MutexGuard<'static, Option<Open>>, split: &mut MutexGuard<'static, Option<Split>>, side: usize) {
    let Some(state) = split.as_mut() else { return };
    if state.focus == side {
        return;
    }
    if let Some(tab) = open.as_mut() {
        std::mem::swap(tab, &mut state.other);
        state.focus = side;
        tab.dirty = true;
    }
}

/// Draw what is up: one tab, or both panes.
fn draw_any(open: &mut Option<Open>, split: &mut Option<Split>, display: &ActiveDisplay) -> io::Result<Outcome> {
    match split.as_mut() {
        Some(state) => draw_split(open, state, display).map(|_| Outcome::Stay),
        None => draw(open, display),
    }
}

/// Both panes, each composed as its tab would be at its half of the width,
/// side by side with a `│` between; the one without the keyboard has its tab
/// in the bar dimmed. Drawn at most every `FRAME`, and every second for the
/// clocks.
fn draw_split(open: &mut Option<Open>, state: &mut Split, display: &ActiveDisplay) -> io::Result<()> {
    let Some(tab) = open.as_mut() else { return Ok(()) };
    let size = crossterm::terminal::size()?;
    let (columns, rows) = (size.0 as usize, size.1 as usize);
    let now = Instant::now();
    if let Some((drawn, at)) = state.painted
        && drawn == size
        && !tab.dirty
        && !state.other.dirty
        && now.duration_since(at) < FRAME
    {
        return Ok(());
    }
    if columns < 40 || rows < 8 {
        let mut stdout = io::stdout().lock();
        queue!(stdout, Clear(ClearType::All), MoveTo(0, 0), Print("too small for two panes · alt+\\ one"))?;
        stdout.flush()?;
        state.frame.clear();
        state.painted = Some((size, now));
        return Ok(());
    }
    crate::vm::terms::touch();
    let (left_width, right_width) = pane_widths(columns);
    let (mine, other) = if state.focus == 0 { (left_width, right_width) } else { (right_width, left_width) };
    let (focused, cursor) = compose(tab, display, (mine as u16, rows as u16), now);
    let (beside, _) = compose(&mut state.other, display, (other as u16, rows as u16), now);
    // The pane without the keyboard: its tab in the bar not lit.
    let beside: Vec<Line> = beside
        .into_iter()
        .map(|line| line.into_iter().map(|(style, text)| (if style == Style::Tab { Style::Status } else { style }, text)).collect())
        .collect();
    let (left, right) = if state.focus == 0 { (focused, beside) } else { (beside, focused) };
    let empty = Vec::new();
    let frame: Vec<Line> = (0..rows)
        .map(|at| {
            let mut line = fit_line(left.get(at).unwrap_or(&empty), left_width);
            line.push((Style::Dim, "│".into()));
            line.extend(fit_line(right.get(at).unwrap_or(&empty), right_width));
            line
        })
        .collect();
    let cursor = cursor.map(|(column, row)| if state.focus == 1 { (column + left_width as u16 + 1, row) } else { (column, row) });
    if state.painted.is_none_or(|(drawn, _)| drawn != size) {
        state.frame.clear();
    }
    paint(&state.frame, &frame, cursor, columns)?;
    state.frame = frame;
    state.painted = Some((size, now));
    tab.dirty = false;
    state.other.dirty = false;
    Ok(())
}

/// `line` exactly `width` columns: cut (a raw row at its codes' edges) and
/// padded.
fn fit_line(line: &Line, width: usize) -> Line {
    let mut out: Line = Vec::with_capacity(line.len() + 1);
    let mut used = 0;
    for (style, text) in line {
        if used >= width {
            break;
        }
        let room = width - used;
        if *style == Style::Raw {
            let shown = text_width(&plain_text(text));
            if shown <= room {
                out.push((Style::Raw, text.clone()));
                used += shown;
            } else {
                out.push((Style::Raw, format!("{}\x1b[0m", cut_at_width(text, room))));
                used = width;
            }
            continue;
        }
        let shown = text_width(text);
        if shown <= room {
            out.push((*style, text.clone()));
            used += shown;
        } else {
            let mut cut = String::new();
            let mut taken = 0;
            for c in text.chars() {
                let w = c.width().unwrap_or(0);
                if taken + w > room {
                    break;
                }
                taken += w;
                cut.push(c);
            }
            out.push((*style, cut));
            used += taken;
        }
    }
    if used < width {
        out.push((Style::Plain, " ".repeat(width - used)));
    }
    out
}

/// Go to `target`: a tab opens (covering the agent's display) or the open one
/// changes; `Target::Main` closes it, and the caller uncovers the display.
/// `idle_prompt`: the idle prompt's draft and cursor, when it is opened from
/// there.
pub fn go(target: Target, display: &ActiveDisplay, idle_prompt: Option<(&str, usize)>) -> io::Result<Outcome> {
    let mut open = lock();
    let view = match target {
        Target::Main => {
            return Ok(match open.take() {
                Some(tab) => {
                    stash(tab);
                    Outcome::Closed
                }
                None => Outcome::Stay,
            });
        }
        // From the agent's page: the tab last up, as it was; from a tab, the one
        // before it.
        Target::Back => {
            if let Some(tab) = open.as_mut() {
                let outcome = go_to(tab, Target::Back);
                if outcome == Outcome::Closed {
                    if let Some(tab) = open.take() {
                        stash(tab);
                    }
                    return Ok(Outcome::Closed);
                }
                let mut split = split_lock();
                return draw_any(&mut open, &mut split, display);
            }
            let Some(mut tab) = closed().take() else {
                return Ok(Outcome::Stay);
            };
            display.cover(idle_prompt)?;
            tab.frame.clear();
            tab.painted = None;
            tab.dirty = true;
            *open = Some(tab);
            let mut split = split_lock();
            return draw_any(&mut open, &mut split, display);
        }
        Target::View(view) => view,
    };
    match open.as_mut() {
        Some(tab) if tab.view.same(&view) => {}
        Some(tab) => {
            let _ = go_to(tab, Target::View(view));
        }
        None => {
            display.cover(idle_prompt)?;
            // The tab last closed is the one Alt+B goes back to.
            let previous = closed().take().filter(|left| !left.view.same(&view)).map(Box::new);
            *open = Some(Open { previous, ..Open::new(view) });
        }
    }
    if let Some(tab) = open.as_mut()
        && tab.view == View::Memory
    {
        memory_view::take_start(&mut tab.memory);
    }
    let mut split = split_lock();
    draw_any(&mut open, &mut split, display)
}

/// Bumped when a view's background work finishes, so the tab draws it now.
static POKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn poke() {
    POKED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// The files open in the operator's editor, for the agent's next message: said
/// when it changes.
pub fn editor_note() -> Option<String> {
    editor_view::take_note()
}

/// Whether a shared file is open in the editor: `Some(unsaved)`.
pub fn editor_has(dir: &str, path: &str) -> Option<bool> {
    editor_view::is_open(dir, path)
}

/// At exit: unsaved changes copied for next time.
pub fn editor_flush() {
    editor_view::flush();
}

/// A job closed or a redaction made: the editor's copies of unsaved changes
/// go from the disk.
pub fn editor_forget_copies() {
    editor_view::forget_recoveries();
}

/// A file of the shared folder the operator has (`share_sync`): not news to him in
/// the share tab.
pub fn share_seen(dir: &str, path: &str, stamp: &str) {
    share_view::mark_seen(dir, path, stamp);
}

/// A tab to open when the prompt is next up (`/redact` asks for the memory
/// tab from a command, which only the prompt can open a tab over).
static REQUESTED: Mutex<Option<Target>> = Mutex::new(None);

pub fn request(target: Target) {
    *REQUESTED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(target);
}

pub fn request_pending() -> bool {
    REQUESTED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_some()
}

pub fn take_request() -> Option<Target> {
    REQUESTED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take()
}

/// PgUp or the wheel on the agent's page: its conversation, `back` rows further
/// back than the end, in `View::History` (covering the page). Nothing opens
/// when all of it fits on the page.
pub fn history(display: &ActiveDisplay, idle_prompt: Option<(&str, usize)>, back: usize) -> io::Result<Outcome> {
    let mut open = lock();
    if open.is_some() {
        return Ok(Outcome::Stay);
    }
    let (columns, rows) = crossterm::terminal::size()?;
    let mut tab = Open::new(View::History);
    if history_rows(&mut tab, display, columns as usize)? <= display.page_rows(rows) {
        return Ok(Outcome::Stay);
    }
    tab.back = back.max(1);
    display.cover(idle_prompt)?;
    *open = Some(tab);
    draw(&mut open, display)
}

/// Close the tab (both panes, when two are up) without drawing anything
/// (the caller uncovers).
pub fn close() {
    let mut open = lock();
    open.take();
    split_lock().take();
    set_split_up(false);
}

/// Draw the open tab again if what it shows has changed. Cheap to call often.
pub fn tick(display: &ActiveDisplay) -> io::Result<()> {
    if crate::terminal::torn_down() {
        return Ok(());
    }
    // Keys waiting: they are read first, and drawn once after them (the
    // mouse reports presses and the wheel only, so this never waits long).
    if crossterm::event::poll(Duration::ZERO).unwrap_or(false) {
        return Ok(());
    }
    let mut open = lock();
    if open.is_none() {
        return Ok(());
    }
    let mut split = split_lock();
    draw_any(&mut open, &mut split, display).map(|_| ())
}

/// A key, a paste or a resize, while a tab is open (or two panes are).
pub fn event(event: &Event, display: &ActiveDisplay) -> io::Result<Outcome> {
    let mut open = lock();
    if open.is_none() {
        return Ok(Outcome::Stay);
    }
    let mut split = split_lock();
    // The panes' own keys: Ctrl+Shift+P one pane or two, Ctrl+1 and Ctrl+2
    // the pane that has the keyboard.
    if let Event::Key(key) = event
        && key.kind != KeyEventKind::Release
    {
        if is_split_key(key) {
            return toggle_split(&mut open, &mut split, display);
        }
        if let Some(side) = focus_key(key)
            && split.is_some()
        {
            focus(&mut open, &mut split, side);
            return draw_any(&mut open, &mut split, display).map(|_| Outcome::Stay);
        }
    }
    // The mouse over the other pane: that pane takes it (the keyboard stays
    // where it is), at its own columns. A click there gives it the keyboard
    // too, and is its own click as well.
    if let Event::Mouse(mouse) = event
        && split.is_some()
    {
        let (columns, _) = crossterm::terminal::size()?;
        let (left, _) = pane_widths(columns as usize);
        let side = if (mouse.column as usize) < left { 0 } else { 1 };
        if matches!(mouse.kind, MouseEventKind::Down(_)) {
            focus(&mut open, &mut split, side);
        }
        let mut local = *mouse;
        if side == 1 {
            local.column = mouse.column.saturating_sub(left as u16 + 1);
        }
        let local = Event::Mouse(local);
        let Some(state) = split.as_mut() else { return Ok(Outcome::Stay) };
        if side != state.focus {
            let mut other = Some(std::mem::replace(&mut state.other, Open::new(View::Page)));
            let outcome = pane_event(&mut other, &local, display)?;
            state.other = other.unwrap_or_else(|| Open::new(View::Page));
            let _ = outcome;
            return draw_any(&mut open, &mut split, display).map(|_| Outcome::Stay);
        }
        let outcome = pane_event(&mut open, &local, display)?;
        return finish(&mut open, &mut split, display, outcome);
    }
    let outcome = pane_event(&mut open, event, display)?;
    finish(&mut open, &mut split, display, outcome)
}

/// What an event did to the pane that has the keyboard, made good: drawn,
/// or closed. With two panes up, a pane closed shows the agent's page instead.
fn finish(open: &mut MutexGuard<'static, Option<Open>>, split: &mut MutexGuard<'static, Option<Split>>, display: &ActiveDisplay, outcome: Outcome) -> io::Result<Outcome> {
    // Scrolled back down to the end: the page again.
    let outcome = match open.as_ref() {
        Some(tab) if outcome == Outcome::Stay && tab.view == View::History && tab.back == 0 => Outcome::Closed,
        _ => outcome,
    };
    match outcome {
        Outcome::Stay | Outcome::Submit { .. } => {
            if let Some(tab) = open.as_mut() {
                tab.dirty = true;
            }
            // More keys (or the rest of a resize drag) already waiting: they
            // are taken first, and drawn once.
            if !crossterm::event::poll(Duration::ZERO).unwrap_or(false) {
                draw_any(open, split, display)?;
            }
            Ok(outcome)
        }
        Outcome::Closed if split.is_some() => {
            // The pane shows the agent's page; Alt+B there comes back.
            if let Some(tab) = open.as_mut()
                && tab.view != View::Page
            {
                let frame = std::mem::take(&mut tab.frame);
                let page = Open { expanded: tab.expanded, ..Open::new(View::Page) };
                let mut left = std::mem::replace(tab, page);
                left.previous = None;
                tab.frame = frame;
                tab.previous = Some(Box::new(left));
            }
            draw_any(open, split, display)?;
            Ok(Outcome::Stay)
        }
        Outcome::Closed | Outcome::Eof => {
            if let Some(tab) = open.take() {
                stash(tab);
            }
            split.take();
            set_split_up(false);
            Ok(outcome)
        }
        Outcome::Interrupt => Ok(outcome),
    }
}

/// An event for one pane (`open`, taken in or out of the split as it may be).
fn pane_event(open: &mut Option<Open>, event: &Event, display: &ActiveDisplay) -> io::Result<Outcome> {
    let Some(tab) = open.as_mut() else {
        return Ok(Outcome::Stay);
    };
    let outcome = match event {
        Event::Key(key) if key.kind != KeyEventKind::Release => key_event(tab, key, display)?,
        Event::Paste(text) => {
            let text = text.replace("\r\n", "\n").replace('\r', "\n");
            match &tab.view {
                View::Shell { session: Some(session), .. } => {
                    crate::vm::terms::send(session, crate::vm::terms::Input::Paste(text));
                    tab.back = 0;
                }
                View::Memory => memory_view::paste(tab, &text),
                View::Share => share_view::paste(tab, &text),
                View::Overview => overview_view::paste(tab, &text),
                View::Audit => audit_view::paste(tab, &text),
                View::Edit => editor_view::paste(tab, &text),
                View::Review | View::Verification | View::Deliverables => projects_view::paste(tab, &text),
                view if view.has_prompt() => tab.draft.insert_str(&text),
                _ => {}
            }
            Outcome::Stay
        }
        // What can be clicked: the tab bar, the buttons, the rows, a menu.
        Event::Mouse(mouse) if let Some(outcome) = gui_mouse(tab, mouse, display)? => outcome,
        // Selecting text, in every tab but the editor (it has its own).
        Event::Mouse(mouse) if tab.view != View::Edit && select_event(tab, mouse) => Outcome::Stay,
        Event::Mouse(mouse) if tab.view == View::Edit => {
            if matches!(mouse.kind, MouseEventKind::Moved | MouseEventKind::Up(_)) {
                return Ok(Outcome::Stay);
            }
            editor_view::mouse(mouse);
            Outcome::Stay
        }
        Event::Mouse(mouse) => {
            let up = match mouse.kind {
                MouseEventKind::ScrollUp => true,
                MouseEventKind::ScrollDown => false,
                _ => return Ok(Outcome::Stay),
            };
            wheel_view(tab, up, (mouse.row as usize).saturating_sub(body_top()));
            Outcome::Stay
        }
        Event::Resize(..) => {
            // Everything moves: every row is painted again (each clears to
            // its end), with no blank screen in between.
            tab.frame.clear();
            Outcome::Stay
        }
        _ => return Ok(Outcome::Stay),
    };
    Ok(outcome)
}

/// The mouse on what can be clicked (`tabs/gui_view.rs`): a menu while one is
/// open, then the frame's clickable parts. A click is done when the button
/// comes up where it went down, as a GUI's is (and the release is then the
/// last of the click's reports, none left to arrive while a tab is opened):
/// on a button or a tab, what its key does; on a row, it is selected, and a
/// second click on it soon after is Enter. A press that drags selects text
/// instead (`select_event`). A right-click selects the row and opens a menu
/// of the row's and the footer's buttons. `None`: not one of these, for the
/// other handling (text selection, the editor's own mouse, the wheel).
fn gui_mouse(tab: &mut Open, mouse: &crossterm::event::MouseEvent, display: &ActiveDisplay) -> io::Result<Option<Outcome>> {
    use crossterm::event::MouseButton;
    let (row, column) = (mouse.row as usize, mouse.column as usize);
    let rows = tab.frame.len();
    if let Some(menu) = tab.menu.as_mut() {
        let item = menu.item_at(rows, row, column);
        match mouse.kind {
            MouseEventKind::Moved | MouseEventKind::Drag(_) => {
                if let Some(item) = item {
                    menu.at = item;
                    tab.dirty = true;
                }
            }
            MouseEventKind::Down(_) => match item {
                Some(item) => {
                    menu.at = item;
                    tab.pressed = Some((row, column, None));
                    tab.dirty = true;
                }
                None => {
                    tab.menu = None;
                    tab.pressed = None;
                    tab.dirty = true;
                }
            },
            MouseEventKind::Up(_) => {
                let pressed = tab.pressed.take().is_some_and(|(at, _, click)| at == row && click.is_none());
                let chosen = item.filter(|_| pressed).and_then(|item| menu.items.get(item).map(|(_, click)| click.clone()));
                if let Some(click) = chosen {
                    tab.menu = None;
                    tab.dirty = true;
                    return do_click(tab, click, display).map(Some);
                }
            }
            _ => {}
        }
        return Ok(Some(Outcome::Stay));
    }
    let hit = gui_view::at(&tab.hits, row, column).map(|hit| hit.click.clone());
    match (mouse.kind, hit) {
        // On a row: pressed there, and a drag from it still selects text.
        (MouseEventKind::Down(MouseButton::Left), Some(click @ gui_view::Click::Pick(_))) => {
            tab.pressed = Some((row, column, Some(click)));
            Ok(None)
        }
        (MouseEventKind::Down(MouseButton::Left), Some(click)) => {
            tab.pressed = Some((row, column, Some(click)));
            tab.selection = None;
            Ok(Some(Outcome::Stay))
        }
        (MouseEventKind::Down(MouseButton::Left), None) => {
            tab.pressed = None;
            Ok(None)
        }
        (MouseEventKind::Up(MouseButton::Left), hit) => {
            let pressed = tab.pressed.take();
            // Dragged: text was selected, for `select_event` to copy.
            if tab.selection.as_ref().is_some_and(|selection| selection.text().is_some()) {
                return Ok(None);
            }
            let Some((_, _, Some(click))) = pressed.filter(|(at, _, click)| *at == row && click.is_some() && *click == hit) else {
                return Ok(None);
            };
            tab.selection = None;
            tab.dirty = true;
            match click {
                gui_view::Click::Pick(pick) => {
                    let double = tab.clicks.double(&pick, Instant::now());
                    pick_one(tab, pick.clone());
                    // A second click on a row soon after: Enter, as on the
                    // row (a section's or subtab's name has nothing to open).
                    if double && !matches!(pick, gui_view::Pick::Section(_) | gui_view::Pick::MemorySection(_) | gui_view::Pick::Subtab(_)) {
                        return key_event(tab, &KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), display).map(Some);
                    }
                    Ok(Some(Outcome::Stay))
                }
                click => do_click(tab, click, display).map(Some),
            }
        }
        (MouseEventKind::Down(MouseButton::Right), hit) => {
            if let Some(gui_view::Click::Pick(pick)) = hit {
                pick_one(tab, pick);
            }
            tab.selection = None;
            tab.pressed = None;
            // The footer as it is for what is selected now, after the row's
            // own buttons (composed as last drawn: a pane's size, split).
            let size = tab.size;
            let _ = compose(tab, display, size, Instant::now());
            let mut items: Vec<(String, gui_view::Click)> = tab
                .hits
                .iter()
                .filter(|hit| hit.row == row && matches!(hit.click, gui_view::Click::Keys(_)) && !hit.label.trim().is_empty())
                .map(|hit| (hit.label.trim().to_string(), hit.click.clone()))
                .collect();
            for (name, click) in &tab.actions {
                if !items.iter().any(|(had, _)| had == name) {
                    items.push((name.clone(), click.clone()));
                }
            }
            if !items.is_empty() {
                tab.menu = Some(gui_view::Menu { row, column, items, at: 0, columns: tab.size.0 as usize });
            }
            tab.dirty = true;
            Ok(Some(Outcome::Stay))
        }
        (MouseEventKind::Up(MouseButton::Right), _) => Ok(Some(Outcome::Stay)),
        _ => Ok(None),
    }
}

/// What a click does (`gui_view::Click`).
fn do_click(tab: &mut Open, click: gui_view::Click, display: &ActiveDisplay) -> io::Result<Outcome> {
    match click {
        gui_view::Click::Keys(keys) => {
            for key in keys {
                let outcome = key_event(tab, &key, display)?;
                if !matches!(outcome, Outcome::Stay) {
                    return Ok(outcome);
                }
            }
            Ok(Outcome::Stay)
        }
        gui_view::Click::Command(command) => {
            if !tab.view.has_prompt() {
                return Ok(Outcome::Stay);
            }
            // Sent from the prompt as typed; what was being typed is kept.
            let typed = std::mem::take(&mut tab.draft);
            tab.draft.insert_str(command);
            let outcome = key_event(tab, &KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), display);
            // What was being typed comes back, sent or not (a refused send
            // leaves the command in the prompt, never in its place).
            if tab.draft.is_empty() || tab.draft.text() == command {
                tab.draft = typed;
            }
            outcome
        }
        gui_view::Click::Approve => key_event(tab, &KeyEvent::new(KeyCode::Char('a'), KeyModifiers::ALT), display),
        gui_view::Click::Pick(pick) => {
            pick_one(tab, pick);
            tab.dirty = true;
            Ok(Outcome::Stay)
        }
    }
}

/// A row picked by a click, put into effect by its view.
fn pick_one(tab: &mut Open, pick: gui_view::Pick) {
    // A pick moves away from a row asked about, as the wheel does.
    tab.board_unhint = None;
    tab.projects.armed = None;
    match pick {
        gui_view::Pick::Share(path) if tab.view == View::Share => share_view::pick(tab, path),
        // A board's files, in the projects tab: its own browser.
        gui_view::Pick::Share(path) => projects_view::pick_file(tab, path),
        gui_view::Pick::Memory(index) => memory_view::pick(tab, index),
        gui_view::Pick::MemorySection(index) => memory_view::pick_section(tab, index),
        gui_view::Pick::Overview(index) => overview_view::pick(tab, index),
        gui_view::Pick::Project(index) => projects_view::pick(tab, index),
        gui_view::Pick::Section(index) => projects_view::pick_section(tab, index),
        gui_view::Pick::Item(index) => projects_view::pick_item(tab, index),
        gui_view::Pick::Board(index) => board_pick(tab, index),
        gui_view::Pick::Subtab(view) => {
            let _ = go_to(tab, Target::View(view));
        }
        gui_view::Pick::AuditSection(index) => audit_view::pick_section(tab, index),
        gui_view::Pick::AuditRow(index) => audit_view::pick(tab, index),
        gui_view::Pick::AuditFile(file) => audit_view::pick_file(tab, &file),
        gui_view::Pick::ClaimPane(index) => claim_view::pick_pane(tab, index),
        gui_view::Pick::Clause(k) => claim_view::pick_clause(tab, k),
        gui_view::Pick::CodeLine(line) => claim_view::pick_line(tab, line),
        gui_view::Pick::ClaimBoard => audit_view::to_board(tab),
        gui_view::Pick::OpenAudit(audit, claim) => {
            let _ = go_to(tab, Target::View(View::Audit));
            audit_view::open_at(tab, &audit, claim.as_deref());
        }
    }
}

/// A turn of the wheel over body row `row` of a tab. The board and the
/// share tab, and the other lists: the wheel walks the list, or scrolls the
/// strip under it, by where it is turned; the rest scroll back.
fn wheel_view(tab: &mut Open, up: bool, row: usize) {
    if tab.view == View::Audit {
        tab.audit.armed = None;
        if tab.audit.view.is_some() {
            claim_view::wheel(tab, up);
        } else {
            audit_view::wheel(tab, up);
        }
        return;
    }
    if !matches!(tab.view, View::Board | View::Share | View::Review | View::Verification | View::Deliverables | View::Memory | View::Overview) {
        wheel(tab, if up { MouseEventKind::ScrollUp } else { MouseEventKind::ScrollDown });
        return;
    }
    // A turn of the wheel moves away from a row asked about: "again"
    // must be pressed on the row it asked about, right after.
    tab.board_unhint = None;
    tab.projects.armed = None;
    let (_, rows) = crossterm::terminal::size().unwrap_or((80, 24));
    let page = (rows as usize).saturating_sub(5).max(1);
    if tab.view == View::Share {
        share_view::wheel(tab, up, row, page);
    } else if tab.view == View::Memory {
        memory_view::wheel(tab, up, row);
    } else if tab.view == View::Overview {
        // A short list of settings: the wheel picks the one above or below.
        let code = if up { KeyCode::Up } else { KeyCode::Down };
        overview_view::key(tab, &KeyEvent::new(code, KeyModifiers::NONE));
    } else if tab.view != View::Board {
        projects_view::wheel(tab, up, row, page);
    } else if projects_view::section(tab) != projects_view::Section::Board {
        let item = open_item(tab);
        projects_view::open_wheel(tab, item.as_ref(), up, row, page);
    } else {
        board_wheel(tab, up, row, page);
    }
}

/// The mouse selecting text in a tab (`tabs/select_view.rs`): the button down
/// starts a selection, a drag moves its end (onto the top or bottom row, or
/// with the wheel, the tab scrolls and the selection goes on), and letting
/// go copies it. Whether the event was the selection's.
fn select_event(tab: &mut Open, mouse: &crossterm::event::MouseEvent) -> bool {
    use crossterm::event::MouseButton;
    let rows = BODY_ROWS.load(std::sync::atomic::Ordering::Relaxed).max(1);
    let row = (mouse.row as usize).checked_sub(body_top());
    let column = mouse.column as usize;
    let dragging = tab.selection.as_ref().is_some_and(|selection| selection.dragging);
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            // Down on the body starts one; anywhere else clears the last.
            tab.selection = row.filter(|row| *row < rows).map(|row| select_view::Selection::new((row, column)));
            tab.dirty = true;
            true
        }
        MouseEventKind::Drag(MouseButton::Left) if dragging => {
            let on = row.map_or(0, |row| row.min(rows - 1));
            if let Some(selection) = tab.selection.as_mut() {
                selection.mouse = (on, column);
            }
            // On the top row (or above it) or the bottom one (or below):
            // the tab scrolls that way, and the selection with it.
            let edge = match row {
                None | Some(0) => Some(true),
                Some(row) if row + 1 >= rows => Some(false),
                _ => None,
            };
            if let Some(up) = edge {
                wheel_view(tab, up, on);
                if let Some(selection) = tab.selection.as_mut() {
                    selection.scrolled(if up { -(WHEEL_ROWS as isize) } else { WHEEL_ROWS as isize });
                }
            }
            true
        }
        MouseEventKind::ScrollUp | MouseEventKind::ScrollDown if dragging => {
            let up = mouse.kind == MouseEventKind::ScrollUp;
            wheel_view(tab, up, row.unwrap_or(0));
            if let Some(selection) = tab.selection.as_mut() {
                selection.scrolled(if up { -(WHEEL_ROWS as isize) } else { WHEEL_ROWS as isize });
            }
            true
        }
        MouseEventKind::Up(MouseButton::Left) if dragging => {
            let text = tab.selection.as_mut().and_then(|selection| {
                selection.dragging = false;
                selection.text()
            });
            match text {
                Some(text) => {
                    let said = crate::clipboard::copy(&text);
                    tab.tell(said.trim_start_matches("[copy] ").replacen(" via ", " copied via ", 1));
                }
                // A click, not a drag: nothing selected, nothing kept.
                None => tab.selection = None,
            }
            true
        }
        _ => false,
    }
}

/// The mouse wheel scrolls a tab; whether `kind` was a turn of it.
fn wheel(tab: &mut Open, kind: MouseEventKind) -> bool {
    match kind {
        MouseEventKind::ScrollUp => tab.back = tab.back.saturating_add(WHEEL_ROWS),
        MouseEventKind::ScrollDown => tab.back = tab.back.saturating_sub(WHEEL_ROWS),
        _ => return false,
    }
    true
}

fn key_event(tab: &mut Open, key: &KeyEvent, display: &ActiveDisplay) -> io::Result<Outcome> {
    // A key puts away a selection made with the mouse.
    tab.selection = None;
    tab.disarm_if_stale(Instant::now());
    // A right-click's menu takes the keys while it is open: ↑↓ choose, Enter
    // does it, anything else closes it.
    if let Some(menu) = tab.menu.as_mut() {
        tab.dirty = true;
        match key.code {
            KeyCode::Up => menu.at = menu.at.checked_sub(1).unwrap_or(menu.items.len().saturating_sub(1)),
            KeyCode::Down => menu.at = (menu.at + 1) % menu.items.len().max(1),
            KeyCode::Enter => {
                let click = menu.items.get(menu.at).map(|(_, click)| click.clone());
                tab.menu = None;
                if let Some(click) = click {
                    return do_click(tab, click, display);
                }
            }
            _ => tab.menu = None,
        }
        return Ok(Outcome::Stay);
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let (_, rows) = crossterm::terminal::size().unwrap_or((80, 24));
    let page = (rows as usize).saturating_sub(5).max(1);

    // Alt+P in the projects tab: it stays where it is (Tab steps through
    // its subtabs).
    if switch_key(key) == Some((Group::Chat, 'p')) && projects_view::is_projects(&tab.view) {
        return Ok(Outcome::Stay);
    }
    if let Some((group, c)) = switch_key(key)
        && let Some(target) = target(group, c)
    {
        return Ok(go_to(tab, target));
    }
    // Alt+A (twice) and Alt+X in any tab, a shell's and the editor's too: the oldest request
    // waiting for approval (`approval.rs`).
    if alt
        && !ctrl
        && let KeyCode::Char(c @ ('a' | 'A' | 'x' | 'X')) = key.code
    {
        match crate::approval::key(matches!(c, 'a' | 'A')) {
            Ok(said) if said.is_empty() => tab.tell("[approval] answered"),
            Ok(said) => tab.tell(said),
            Err(said) => tab.note(said, true),
        }
        return Ok(Outcome::Stay);
    }
    // Tab and Shift+Tab in the projects tab: its next and previous subtab,
    // as Tab steps through the memory tab's sections.
    if projects_view::is_projects(&tab.view)
        && matches!(key.code, KeyCode::Tab | KeyCode::BackTab)
        && !ctrl
        && !alt
        && !projects_view::typing_now(tab)
    {
        let to = if key.code == KeyCode::BackTab { projects_view::previous(&tab.view) } else { projects_view::next(&tab.view) };
        return Ok(go_to(tab, Target::View(to)));
    }
    // Over `s0ma connect`, a tab that reads the server's files has nothing
    // to act on yet: Esc closes it, Ctrl+C and Ctrl+D do as anywhere, and
    // the other keys do nothing.
    if server_only(&tab.view) {
        return Ok(match key.code {
            KeyCode::Esc | KeyCode::Char('q') if !ctrl && !alt => Outcome::Closed,
            KeyCode::Char('c' | 'C') if ctrl => Outcome::Interrupt,
            KeyCode::Char('d' | 'D') if ctrl => Outcome::Eof,
            _ => Outcome::Stay,
        });
    }
    if matches!(tab.view, View::Review | View::Verification | View::Deliverables) {
        return Ok(projects_view::key(tab, key, page));
    }
    // The editor's keys are its own, Shift's and Ctrl's among them.
    if tab.view == View::Edit {
        return Ok(editor_view::key(tab, key, page));
    }

    // The memory tab has its own keys, Shift's among them (it marks text).
    if tab.view == View::Memory {
        return Ok(memory_view::key(tab, key, page));
    }
    if tab.view == View::Share {
        return Ok(share_view::key(tab, key, page));
    }
    if tab.view == View::Overview {
        return Ok(overview_view::key(tab, key));
    }
    if tab.view == View::Audit {
        return Ok(audit_view::key(tab, key));
    }

    // The open subtab's sections: ←→ step them; the files, notes
    // and reports have their own keys.
    if tab.view == View::Board {
        let item = open_item(tab);
        if let Some(outcome) = projects_view::open_key(tab, item.as_ref(), key, page) {
            return Ok(outcome);
        }
        // On the board, Shift scrolls the strip with the selected node in full.
        if strip_key(tab, key, page) {
            return Ok(Outcome::Stay);
        }
    }

    // Scrolling with Shift works in every tab: in a shell's, the plain keys
    // are the shell's.
    if shift && !ctrl && !alt {
        let scrolled = match key.code {
            KeyCode::PageUp => Some(tab.back.saturating_add(page)),
            KeyCode::PageDown => Some(tab.back.saturating_sub(page)),
            KeyCode::Up => Some(tab.back.saturating_add(1)),
            KeyCode::Down => Some(tab.back.saturating_sub(1)),
            KeyCode::Home => Some(usize::MAX),
            KeyCode::End => Some(0),
            _ => None,
        };
        if let Some(back) = scrolled {
            tab.back = back;
            return Ok(Outcome::Stay);
        }
    }

    match tab.view {
        View::Shell { .. } => Ok(shell_key(tab, key)),
        View::Subagent(_) | View::Page => chat_key(tab, key, display, page),
        View::Board => board_key(tab, key, display, page),
        View::History => Ok(history_key(tab, key, page)),
        View::Memory => Ok(memory_view::key(tab, key, page)),
        View::Share => Ok(share_view::key(tab, key, page)),
        View::Overview => Ok(overview_view::key(tab, key)),
        View::Audit => Ok(audit_view::key(tab, key)),
        View::Review | View::Verification | View::Deliverables => Ok(projects_view::key(tab, key, page)),
        View::Edit => Ok(editor_view::key(tab, key, page)),
    }
}

/// A key in the agent's conversation scrolled back: scrolling, and back to the
/// page (Esc, q, Enter, End, or scrolling down to the end).
fn history_key(tab: &mut Open, key: &KeyEvent, page: usize) -> Outcome {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Esc | KeyCode::Enter | KeyCode::End | KeyCode::Char('q') => return Outcome::Closed,
        KeyCode::Char('c' | 'C') if ctrl => return Outcome::Interrupt,
        KeyCode::Char('d' | 'D') if ctrl => return Outcome::Eof,
        KeyCode::Char('o' | 'O') if ctrl => tab.expanded = !tab.expanded,
        KeyCode::Up => tab.back = tab.back.saturating_add(1),
        KeyCode::Down => tab.back = tab.back.saturating_sub(1),
        KeyCode::PageUp => tab.back = tab.back.saturating_add(page),
        KeyCode::PageDown => tab.back = tab.back.saturating_sub(page),
        KeyCode::Home => tab.back = usize::MAX,
        _ => {}
    }
    Outcome::Stay
}

/// Go to `target` from the open tab.
fn go_to(tab: &mut Open, target: Target) -> Outcome {
    match target {
        Target::Main => Outcome::Closed,
        Target::View(view) if view.same(&tab.view) => Outcome::Stay,
        Target::View(view) => {
            // A draft meant for one subagent never goes to another. The tab
            // left is kept as it is, for Alt+B; the projects tab's subtabs
            // are one tab, so stepping through them keeps what was before it.
            let frame = std::mem::take(&mut tab.frame);
            let next = Open { expanded: tab.expanded, ..Open::new(view) };
            let mut left = std::mem::replace(tab, next);
            tab.frame = frame;
            tab.previous = if projects_view::is_projects(&left.view) && projects_view::is_projects(&tab.view) {
                left.previous.take()
            } else {
                left.previous = None;
                Some(Box::new(left))
            };
            Outcome::Stay
        }
        // Back to the tab before, as it was; this one becomes the one
        // before it (Alt+B again comes back). None before: the agent's page.
        Target::Back => {
            let Some(previous) = tab.previous.take() else {
                return Outcome::Closed;
            };
            let frame = std::mem::take(&mut tab.frame);
            let mut left = std::mem::replace(tab, *previous);
            left.previous = None;
            tab.frame = frame;
            tab.painted = None;
            tab.dirty = true;
            tab.previous = Some(Box::new(left));
            Outcome::Stay
        }
    }
}

/// The tab closed last (back to the agent's page), as it was: Alt+B on the page
/// opens it again.
static CLOSED: Mutex<Option<Open>> = Mutex::new(None);

fn closed() -> MutexGuard<'static, Option<Open>> {
    CLOSED.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Kept as the tab closed last. The agent's page in a pane, or its history, is
/// the page itself: nothing to keep.
fn stash(mut tab: Open) {
    if matches!(tab.view, View::Page | View::History) {
        match tab.previous.take() {
            Some(previous) => tab = *previous,
            None => return,
        }
    }
    tab.previous = None;
    *closed() = Some(tab);
}

/// A key in a shell's tab: to the shell, all of them (switching and Shift
/// scrolling are taken before this).
fn shell_key(tab: &mut Open, key: &KeyEvent) -> Outcome {
    // Alt+N: a new netbox of the operator's, and its tab.
    if key.modifiers.contains(KeyModifiers::ALT) && !key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('n' | 'N')) {
        match act(Action::NetboxOpen) {
            Ok(opened) => {
                let number = crate::vm::terms::sessions().len() + 1;
                tab.view = View::Shell { number, session: Some(opened) };
                tab.back = 0;
            }
            Err(error) => tab.note(format!("no new shell: {error}"), true),
        }
        return Outcome::Stay;
    }
    let View::Shell { session: Some(session), .. } = &tab.view else {
        tab.note("no shell here yet", true);
        return Outcome::Stay;
    };
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    // Ctrl+Q twice: the shell closed, and back to the page. The first says
    // what closing it would cut short, when something runs in it.
    // An audit's shell stays until the audit ends: Ctrl+Q is refused while a
    // command runs in it, and otherwise closes the tab only.
    if ctrl
        && !alt
        && matches!(key.code, KeyCode::Char('q' | 'Q'))
        && let Some(audit) = crate::vm::netboxes::name_of(session).and_then(|name| name.strip_prefix("audit-").map(str::to_string))
    {
        if crate::audit::run::running(&audit) || crate::vm::terms::netboxes().into_iter().any(|n| n.name == format!("audit-{audit}") && n.busy().is_some()) {
            tab.note(format!("refused: a command runs in audit·{audit} (Ctrl+C interrupts it, z in Alt+U pauses the audit)"), true);
            return Outcome::Stay;
        }
        return Outcome::Closed;
    }
    if ctrl && !alt && matches!(key.code, KeyCode::Char('q' | 'Q')) && crate::vm::netboxes::name_of(session).is_some() {
        let session = session.clone();
        let name = crate::vm::netboxes::name_of(&session).unwrap_or_default().to_string();
        let busy = crate::vm::terms::netboxes().into_iter().find(|netbox| netbox.name == name).and_then(|netbox| netbox.busy());
        let armed = tab.close_armed.take().is_some_and(|(armed, at)| armed == session && at.elapsed() < CLOSE_AGAIN);
        if !armed {
            tab.close_armed = Some((session, Instant::now()));
            tab.ask(match busy {
                Some(busy) => format!("{busy} · closes it · again to confirm"),
                None => format!("closes {} · again to confirm", crate::vm::netboxes::display(&name)),
            });
            return Outcome::Stay;
        }
        return match act(Action::ShellClose { session }) {
            Ok(_) => Outcome::Closed,
            Err(error) => {
                tab.note(format!("not closed: {error}"), true);
                Outcome::Stay
            }
        };
    }
    // Ctrl+O in a netbox at its prompt: each command folded back to one line,
    // or opened out again. A program running in it gets the key as usual.
    if ctrl
        && !alt
        && matches!(key.code, KeyCode::Char('o' | 'O'))
        && crate::vm::netboxes::name_of(session).is_some()
        && crate::vm::terms::live(session).is_some_and(|screen| at_prompt(&screen.command))
    {
        SHELL_COMPACT.fetch_xor(true, std::sync::atomic::Ordering::Relaxed);
        return Outcome::Stay;
    }
    // Ctrl+\ in a netbox: the operator takes the wheel, or gives it back. (Some
    // terminals send it as Ctrl+4.)
    if ctrl && !alt && matches!(key.code, KeyCode::Char('\\' | '4')) && crate::vm::netboxes::name_of(session).is_some() {
        crate::vm::terms::toggle_wheel(session);
        return Outcome::Stay;
    }
    let input = match key.code {
        // AltGr arrives as Ctrl+Alt on some platforms; its characters are text.
        KeyCode::Char(c) if ctrl && alt && !c.is_ascii_alphabetic() => Some(crate::vm::terms::Input::Text(c.to_string())),
        _ => crate::vm::terms::key_input(key),
    };
    if let Some(input) = input {
        crate::vm::terms::send(session, input);
        tab.back = 0;
    }
    Outcome::Stay
}

/// A key in a chat's tab (a subagent's, the board's): its prompt, scrolling.
fn chat_key(tab: &mut Open, key: &KeyEvent, display: &ActiveDisplay, page: usize) -> io::Result<Outcome> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    // The agent's page in a pane: Enter sends to the agent (a new turn, or steering
    // one), Ctrl+S sends it now; Esc leaves the pane as it is.
    if tab.view == View::Page {
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let now = ctrl && !alt && matches!(key.code, KeyCode::Char('s' | 'S'));
        match key.code {
            KeyCode::Enter if !alt && !shift && !ctrl => {}
            _ if now => {}
            KeyCode::Esc => return Ok(Outcome::Stay),
            _ => return chat_key_text(tab, key, display, page),
        }
        let text = tab.draft.text().to_string();
        if text.trim().is_empty() && !now {
            return Ok(Outcome::Stay);
        }
        tab.draft = Draft::default();
        tab.back = 0;
        return Ok(Outcome::Submit { text, now });
    }
    chat_key_text(tab, key, display, page)
}

/// A chat's prompt and scrolling (a subagent's, the board's, the agent's page's).
fn chat_key_text(tab: &mut Open, key: &KeyEvent, display: &ActiveDisplay, page: usize) -> io::Result<Outcome> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    // AltGr arrives as Ctrl+Alt on some platforms; its characters are text.
    if alt && ctrl {
        if let KeyCode::Char(c) = key.code {
            tab.draft.insert(c);
        }
        return Ok(Outcome::Stay);
    }

    if alt {
        match key.code {
            KeyCode::Enter => tab.draft.insert('\n'),
            KeyCode::Left => tab.draft.word_left(),
            KeyCode::Right => tab.draft.word_right(),
            KeyCode::Backspace => tab.draft.delete_word_before(),
            KeyCode::Char('f' | 'F') => tab.draft.word_right(),
            KeyCode::Char('d' | 'D') => tab.draft.delete_word_after(),
            _ => {}
        }
        return Ok(Outcome::Stay);
    }

    if ctrl {
        match key.code {
            // The projects tab's boards: the next one, the one before.
            KeyCode::Char('n' | 'N') if tab.view == View::Board => step_board(tab, 1),
            KeyCode::Char('p' | 'P') if tab.view == View::Board => step_board(tab, usize::MAX),
            KeyCode::Char('c' | 'C') => {
                if tab.draft.is_empty() {
                    return Ok(Outcome::Interrupt);
                }
                tab.draft = Draft::default();
            }
            // A finished subagent's tab: closed.
            KeyCode::Char('q' | 'Q') => {
                if let View::Subagent(number) = tab.view
                    && let Some(worker) = subagent_at(number)
                {
                    if worker.running {
                        tab.note("still running; /kill stops it", true);
                        return Ok(Outcome::Stay);
                    }
                    // Twice: the first asks.
                    let armed = tab.close_armed.take().is_some_and(|(armed, at)| armed == worker.run_id && at.elapsed() < CLOSE_AGAIN);
                    if !armed {
                        tab.ask(format!("ctrl+q again closes {}'s tab", short_id(&worker.run_id)));
                        tab.close_armed = Some((worker.run_id, Instant::now()));
                        return Ok(Outcome::Stay);
                    }
                    return Ok(match act(Action::Forget { run_id: worker.run_id }) {
                        Ok(_) => Outcome::Closed,
                        Err(why) => {
                            tab.note(why, true);
                            Outcome::Stay
                        }
                    });
                }
            }
            KeyCode::Char('d' | 'D') => return Ok(Outcome::Eof),
            KeyCode::Char('o' | 'O') => tab.expanded = !tab.expanded,
            KeyCode::Char('l' | 'L') => {
                tab.frame.clear();
                queue!(io::stdout(), Clear(ClearType::All))?;
            }
            KeyCode::Char('u' | 'U') => tab.draft.kill_line_before(),
            KeyCode::Char('k' | 'K') => tab.draft.kill_line_after(),
            KeyCode::Char('w' | 'W') | KeyCode::Backspace => tab.draft.delete_word_before(),
            KeyCode::Char('a' | 'A') => tab.draft.home(),
            KeyCode::Char('e' | 'E') => tab.draft.end(),
            KeyCode::Char('b' | 'B') => tab.draft.left(),
            KeyCode::Char('f' | 'F') => tab.draft.right(),
            KeyCode::Left => tab.draft.word_left(),
            KeyCode::Right => tab.draft.word_right(),
            KeyCode::Home => tab.back = usize::MAX,
            KeyCode::End => tab.back = 0,
            _ => {}
        }
        return Ok(Outcome::Stay);
    }

    match key.code {
        KeyCode::Esc => return Ok(Outcome::Closed),
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => tab.draft.insert('\n'),
        KeyCode::Enter => submit(tab, display),
        KeyCode::Up => tab.back = tab.back.saturating_add(1),
        KeyCode::Down => tab.back = tab.back.saturating_sub(1),
        KeyCode::PageUp => tab.back = tab.back.saturating_add(page),
        KeyCode::PageDown => tab.back = tab.back.saturating_sub(page),
        KeyCode::Backspace => tab.draft.backspace(),
        KeyCode::Delete => tab.draft.delete(),
        KeyCode::Left => tab.draft.left(),
        KeyCode::Right => tab.draft.right(),
        KeyCode::Home => tab.draft.home(),
        KeyCode::End if tab.draft.is_empty() => tab.back = 0,
        KeyCode::End => tab.draft.end(),
        KeyCode::Char(c) => tab.draft.insert(c),
        _ => {}
    }
    Ok(Outcome::Stay)
}

/// Another board on the board subtab: `step` on (`usize::MAX`: back one),
/// wrapping (`read_board` takes it modulo how many there are).
fn step_board(tab: &mut Open, step: usize) {
    if let Some(read) = &tab.board_read {
        tab.boards_known = read.open.max(1);
    }
    let open = tab.boards_known;
    tab.board = (tab.board % open).wrapping_add(if step == usize::MAX { open - 1 } else { step });
    tab.board_read = None;
    tab.back = 0;
    tab.board_selected = None;
    tab.board_top = 0;
}

/// The board tab at board `name`, with `node` (a finding or a lead)
/// selected: a board reference clicked in the audit tab.
fn show_board(tab: &mut Open, name: &str, node: Option<&str>) {
    let Some(store) = crate::audit::boards() else { return };
    let Ok(boards) = shown_boards(&store) else { return };
    let Some(index) = boards.iter().position(|board| board.name == name) else {
        tab.note(format!("{name} is not on the board subtab (it is further along: Alt+P)"), true);
        return;
    };
    let _ = go_to(tab, Target::View(View::Board));
    tab.board = index;
    tab.boards_known = boards.len();
    tab.board_read = None;
    tab.board_selected = node.map(str::to_string);
    tab.board_top = 0;
}

/// The subagent on tab `number` (2 is the first), if there is one.
fn subagent_at(number: usize) -> Option<crate::subagent::feed::Tab> {
    number.checked_sub(2).and_then(|index| crate::subagent::feed::tabs().into_iter().nth(index))
}

/// The tab number a subagent's run id has, for the board's claims.
fn tab_of(run_id: &str) -> Option<usize> {
    crate::subagent::feed::tabs().iter().position(|tab| tab.run_id == run_id).map(|index| index + 2)
}

/// Enter: the draft to the subagent on this tab, or a hint to the board.
fn submit(tab: &mut Open, _display: &ActiveDisplay) {
    let text = tab.draft.text().trim().to_string();
    if text.is_empty() {
        return;
    }
    match tab.view {
        View::Shell { .. } | View::History | View::Memory | View::Share | View::Overview | View::Audit | View::Edit | View::Page => {}
        View::Review | View::Verification | View::Deliverables => {}
        View::Subagent(number) => {
            let Some(worker) = subagent_at(number) else {
                tab.note("no subagent here to send it to", true);
                return;
            };
            if !worker.running {
                tab.note(format!("{} has finished", short_id(&worker.run_id)), true);
                return;
            }
            if tab.pinned.as_deref() != Some(worker.run_id.as_str()) {
                let was = tab.pinned.as_deref().map_or_else(|| "no subagent".to_string(), short_id);
                tab.note(
                    format!("not sent: you began this for {was}; sending again sends it to {}", short_id(&worker.run_id)),
                    true,
                );
                tab.pinned = Some(worker.run_id);
                return;
            }
            let run_id = worker.run_id.clone();
            let action = if text == "/kill" {
                Action::Kill { run_id }
            } else if text == "/wrap" || text.starts_with("/wrap ") {
                let rest = text["/wrap".len()..].trim();
                Action::Steer { run_id, text: if rest.is_empty() { WRAP_UP } else { rest }.to_string(), wrap_up: true }
            } else {
                Action::Steer { run_id, text, wrap_up: false }
            };
            match act(action) {
                Ok(note) => {
                    tab.draft = Draft::default();
                    tab.back = 0;
                    tab.note(note, false);
                }
                Err(error) => tab.note(error, true),
            }
        }
        View::Board => {
            let Some(board) = tab.board_read.as_ref().and_then(|read| read.name.clone()) else {
                tab.note("no open board", true);
                return;
            };
            // About the selected lead or finding, when one is.
            let mut about = None;
            let adding = tab.board_adding;
            let hint = match (board_tree(tab), tab.board_read.as_ref().and_then(|read| read.snapshot.as_ref().ok()?.as_ref())) {
                (Some((tree, shown, selected)), Some(snapshot)) => {
                    let node = selected.map(|index| shown[index].node);
                    let hint = board_view::hint_about(&tree, snapshot, node, &text);
                    // Something added to a greyed-out lead or finding brings
                    // its branch back, from the root down to it.
                    if board_view::prompt_target(&tree, node).is_some() {
                        about = node.map(|node| (tree, snapshot.board.id.clone(), node));
                    }
                    hint
                }
                _ => text.clone(),
            };
            let under = about.as_ref().map(|(tree, _, node)| tree.nodes[*node].id.clone());
            let action = match adding {
                Adding::Hint => Action::Hint { board, text: hint },
                _ => Action::Add { board, adding, about: under, text },
            };
            match act(action) {
                Ok(reply) => {
                    if let Some((tree, board, node)) = about {
                        with_greyed(&board, |greyed| board_view::revive(&tree, greyed, node));
                    }
                    tab.draft = Draft::default();
                    tab.board_read = None;
                    tab.note(reply, false);
                }
                Err(error) => tab.note(error, true),
            }
        }
    }
}

/// What the board's prompt adds to the board: a hint for the agent, a lead of
/// the operator's own, or a finding of his (for the lead, or against it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Adding {
    Hint,
    Lead,
    Supports,
    RulesOut,
    /// A claim of the operator's, from the selected finding, to the board's
    /// owned or linked audit with this index (`crate::audit::ops::for_board`).
    Claim(usize),
}

impl Adding {
    /// The next, for Ctrl+T: the claim mode, once per audit, only on a
    /// board with audits.
    fn next(self, audits: usize) -> Adding {
        match self {
            Adding::Hint => Adding::Lead,
            Adding::Lead => Adding::Supports,
            Adding::Supports => Adding::RulesOut,
            Adding::RulesOut if audits > 0 => Adding::Claim(0),
            Adding::Claim(i) if i + 1 < audits => Adding::Claim(i + 1),
            Adding::RulesOut | Adding::Claim(_) => Adding::Hint,
        }
    }
}

/// The operator's lead or finding, on board `board` (its name), under the
/// lead or finding `about` (its id) when one was selected.
fn add_to_board(board: &str, adding: Adding, about: Option<&str>, text: &str) -> Result<String, String> {
    let archive = ARCHIVE.get().ok_or("the board is not available in this process")?;
    let store = crate::board::BoardStore::beside_archive(archive);
    let written = |error: Box<dyn std::error::Error>| format!("the board could not be written: {error}");
    let board = store.board(board).map_err(written)?.ok_or_else(|| format!("no board named {board}"))?;
    let text = crate::board::bounded(text, 2, 2_000, "text")?;
    let fact = about.filter(|id| id.starts_with("fact-"));
    match adding {
        Adding::Hint => Err("a hint goes as a hint".into()),
        Adding::Claim(index) => {
            let audits = crate::audit::ops::for_board(&board.name);
            let audit = audits.get(index).ok_or("no audit is attached or linked to this board")?;
            let store = crate::audit::store().ok_or("the audit store is not ready")?;
            let source = match fact {
                Some(id) => format!("board:{}/{}", board.name, board_view::short(id)),
                None => format!("board:{}", board.name),
            };
            crate::audit::ops::operator_claim_new(&store, &audit.name, &text, &source).map(|said| said.trim_start_matches("[audit] ").to_string())
        }
        Adding::Lead => {
            // `p8 …` first: its priority (5 when not said).
            let (priority, direction) = match text.split_once(' ') {
                Some((head, rest)) if head.len() > 1 && head.starts_with('p') && head[1..].parse::<i64>().is_ok_and(|p| (1..=10).contains(&p)) => {
                    (head[1..].parse().unwrap_or(5), rest.trim().to_string())
                }
                _ => (5, text),
            };
            let lead = store.operator_lead(&board.id, &direction, fact, priority).map_err(written)?;
            Ok(format!("lead {} added to {}", board_view::short(&lead.id), board.name))
        }
        Adding::Supports | Adding::RulesOut => {
            let polarity = if adding == Adding::Supports { crate::board::Polarity::Supports } else { crate::board::Polarity::RulesOut };
            let lead = about.filter(|id| id.starts_with("intent-"));
            let found = store.operator_fact(&board.id, &text, polarity, lead, fact).map_err(written)?;
            Ok(format!("finding {} added to {}", board_view::short(&found.id), board.name))
        }
    }
}

/// What Enter in a tab does, done by the agent that runs the subagents: this
/// process, or, on an attached terminal, the one it is attached to.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Action {
    Steer { run_id: String, text: String, wrap_up: bool },
    Kill { run_id: String },
    Hint { board: String, text: String },
    /// A lead or a finding of the operator's put on a board, under `about`.
    Add { board: String, adding: Adding, about: Option<String>, text: String },
    /// A note of the operator's taken off a board.
    Unhint { board: String, id: String },
    /// The operator's move on a board in the workflow (`board/flow.rs`), by id.
    Project { board: String, done: projects_view::Move },
    /// The overview: the app's latest snapshot (and ask it for a fresh one).
    Overview,
    /// A loop of the agent's closed by the operator (done, or dropped), with what he said.
    CloseLoop { id: String, done: bool, said: String },
    /// The overview: a command for the app to run as typed (only its own).
    Command { text: String },
    /// The overview: the model server's API key, kept by the app
    /// (`model_config.rs`). Never a command, so it is never echoed.
    ModelKey { key: crate::model_config::Secret },
    /// Where these strings are kept, as `strike::Impact` in JSON: asked of
    /// the agent that keeps the stores, whose skills and worker journal an
    /// attached terminal does not know.
    Survey { fragments: Vec<String> },
    /// What approving request `request` would reach, as JSON: the impact and
    /// how many pieces are too short to look for elsewhere.
    Review { request: u64 },
    /// Strike these strings from every store (the memory tab's marks).
    Strike { fragments: Vec<String> },
    /// The agent's redaction request: approve it, decline it, or save the
    /// operator's own version of it.
    Approve { request: u64 },
    Decline { request: u64, note: String },
    /// Approval mode: the operator's answer to request `id` (`approval.rs`).
    Answer { id: u64, approve: bool },
    Edit { request: u64, text: String },
    /// What the operator did in the share tab, for his next message to the agent.
    Shared { change: crate::share::files::Change },
    /// Typed into a netbox on an attached terminal, for the agent's next read
    /// (the first netbox's when an older terminal does not say).
    NetboxTyped {
        #[serde(default = "first_netbox")]
        session: String,
        typed: Vec<crate::vm::terms::Input>,
    },
    /// Alt+N on an attached terminal: a netbox of the operator's, opened by the agent
    /// that runs the turns; the shell session it will be.
    NetboxOpen,
    /// A finished subagent's tab closed: its feed forgotten everywhere.
    Forget { run_id: String },
    /// Ctrl+Q in a shell's tab: that shell closed, whatever runs in it.
    ShellClose { session: String },
    /// What the sync of `/share` with the operator's machine is doing, and the files
    /// it wrote after mark `since` (the operator's own: not news to him), as
    /// `crate::share::sync::Report` in JSON. It runs in the agent that runs the turns.
    ShareSync { since: u64 },
}

type Remote = Box<dyn Fn(Action) -> Result<String, String> + Send + Sync>;
static REMOTE: OnceLock<Remote> = OnceLock::new();

/// On an attached terminal: actions go to `remote` (`session.rs`).
pub fn act_remotely(remote: Remote) {
    let _ = REMOTE.set(remote);
}

/// This is an attached terminal: what it does goes to the agent that runs
/// the turns.
pub(crate) fn remote() -> bool {
    REMOTE.get().is_some()
}

fn first_netbox() -> String {
    crate::vm::netboxes::DEFAULT_SESSION.to_string()
}

/// Typed into netbox `session` here, on an attached terminal (`vm_terms`).
pub(crate) fn netbox_typed(session: String, typed: Vec<crate::vm::terms::Input>) {
    let _ = act(Action::NetboxTyped { session, typed });
}

/// On an attached terminal: the sync of `/share` as the agent that runs it
/// says, for the share tab (`crate::share::sync::label`, `local_path`), and the
/// files it wrote since last asked, marked seen here too.
pub(crate) fn follow_share_sync() {
    static SINCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if !remote() {
        return;
    }
    let since = SINCE.load(std::sync::atomic::Ordering::SeqCst);
    let Some(report) = act(Action::ShareSync { since })
        .ok()
        .and_then(|json| serde_json::from_str::<crate::share::sync::Report>(&json).ok())
    else {
        return;
    };
    for (dir, path, stamp) in &report.wrote {
        share_view::mark_seen(dir, path, stamp);
    }
    SINCE.store(report.mark, std::sync::atomic::Ordering::SeqCst);
    crate::share::sync::mirror(report);
}

/// `action`, done by the agent this terminal shows: here, or sent to it.
pub fn act_on(action: Action) -> Result<String, String> {
    act(action)
}

fn act(action: Action) -> Result<String, String> {
    match REMOTE.get() {
        Some(remote) => remote(action),
        None => perform(action),
    }
}

/// `action`, here: what to say, or why not.
pub fn perform(action: Action) -> Result<String, String> {
    match action {
        Action::Steer { run_id, text, wrap_up } => crate::subagent::steer_from_operator(&run_id, &text, wrap_up),
        Action::Kill { run_id } => crate::subagent::kill(&run_id).map(|id| format!("stopping {id}")),
        Action::Answer { id, approve } => crate::approval::answer(id, approve),
        Action::Hint { board, text } => {
            let archive = ARCHIVE.get().ok_or("the board is not available in this process")?;
            let store = crate::board::BoardStore::beside_archive(archive);
            let reply = crate::board::handle_operator_command(&store, &format!("/board hint {board} {text}"))
                .map_err(|error| format!("the board could not be written: {error}"))?;
            let reply = reply.trim_start_matches("[board] ").to_string();
            if reply.contains(" left on ") { Ok(reply) } else { Err(reply) }
        }
        Action::Add { board, adding, about, text } => add_to_board(&board, adding, about.as_deref(), &text),
        Action::Project { board, done } => {
            let archive = ARCHIVE.get().ok_or("the boards are not available in this process")?;
            projects_view::perform(archive, &board, done)
        }
        Action::Unhint { board, id } => {
            let archive = ARCHIVE.get().ok_or("the board is not available in this process")?;
            let store = crate::board::BoardStore::beside_archive(archive);
            let reply = crate::board::handle_operator_command(&store, &format!("/board unhint {board} {id}"))
                .map_err(|error| format!("the board could not be written: {error}"))?;
            let reply = reply.trim_start_matches("[board] ").to_string();
            if reply.contains(" taken off ") { Ok(reply) } else { Err(reply) }
        }
        Action::Survey { fragments } => {
            let memory = memory_view::writer()?;
            let impact = crate::strike::survey(&memory, &crate::strike::Fragments::new(&fragments)?)?;
            serde_json::to_string(&impact).map_err(|error| error.to_string())
        }
        Action::Review { request } => {
            let memory = memory_view::writer()?;
            let book = crate::redact::Book::beside_archive(memory.archive_path());
            let (impact, short) = crate::redact::review(&memory, &book, request)?;
            serde_json::to_string(&(impact, short)).map_err(|error| error.to_string())
        }
        Action::Strike { fragments } => {
            let memory = memory_view::writer()?;
            let fragments = crate::strike::Fragments::new(&fragments)?;
            let book = crate::redact::Book::beside_archive(memory.archive_path());
            let (request, impact) = crate::redact::strike_by_operator(&memory, &book, &fragments, "")?;
            Ok(format!("struck, redaction #{}: {}", request.request, impact.summary()))
        }
        Action::Approve { request } => {
            let memory = memory_view::writer()?;
            let book = crate::redact::Book::beside_archive(memory.archive_path());
            let (_, impact) = crate::redact::approve(&memory, &book, request)?;
            Ok(format!("#{request} approved: {}", impact.summary()))
        }
        Action::Decline { request, note } => {
            let memory = memory_view::writer()?;
            crate::redact::reject(&crate::redact::Book::beside_archive(memory.archive_path()), request, &note)?;
            Ok(format!("#{request} declined; the event is untouched"))
        }
        Action::Edit { request, text } => {
            let memory = memory_view::writer()?;
            crate::redact::save_edit(&crate::redact::Book::beside_archive(memory.archive_path()), request, &text)?;
            Ok(format!("#{request}: your version is what approving would write"))
        }
        Action::Overview => {
            let snapshot = crate::overview::want().map(|(snapshot, age)| (snapshot, age.as_millis() as u64));
            serde_json::to_string(&snapshot).map_err(|error| error.to_string())
        }
        Action::Command { text } => crate::overview::queue(&text).map(|()| format!("{text} queued")),
        Action::ModelKey { key } => {
            let said = crate::model_config::set_key(&key.0);
            // The tab shows the settings as they are now.
            crate::overview::want();
            said
        }
        Action::CloseLoop { id, done, said } => {
            let archive = ARCHIVE.get().ok_or("the loops are not available in this process")?;
            let store = crate::open_loops::LoopStore::beside_archive(archive);
            match crate::open_loops::close_by_operator(&store, &id, done, &said) {
                Ok(Ok(entry)) => Ok(format!("{} {} · s0ma hears with your next message", entry.id, if done { "closed as done" } else { "dropped" })),
                Ok(Err(why)) => Err(why),
                Err(error) => Err(format!("the loops could not be written: {error}")),
            }
        }
        Action::Shared { change } => {
            crate::share::files::record(change);
            Ok(String::new())
        }
        Action::NetboxTyped { session, typed } => {
            crate::vm::terms::note_netbox(&session, &typed);
            Ok(String::new())
        }
        Action::NetboxOpen => Ok(crate::vm::netboxes::open_for_operator()),
        Action::ShellClose { session } => crate::vm::netboxes::close_for_operator(&session),
        Action::Forget { run_id } => {
            if crate::subagent::feed::forget(&run_id) { Ok("closed".into()) } else { Err("it is still running".into()) }
        }
        Action::ShareSync { since } => serde_json::to_string(&crate::share::sync::report(since)).map_err(|error| error.to_string()),
    }
}

// ------------------------------------------------------------------ small things

fn text_width(text: &str) -> usize {
    text.chars().map(|c| c.width().unwrap_or(0)).sum()
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
    out.truncate(out.trim_end().len());
    out.push('…');
    out
}

/// `subagent-1730…-3` → `…-3` would say nothing; the batch's own number and
/// the slot are what tell runs apart: `b17-3`.
fn short_id(run_id: &str) -> String {
    let parts: Vec<&str> = run_id.split('-').collect();
    match parts.as_slice() {
        ["subagent", _, _, batch, slot] => format!("b{batch}-{slot}"),
        _ => run_id.to_string(),
    }
}

fn span(duration: Duration) -> String {
    let seconds = duration.as_secs();
    match seconds {
        0..60 => format!("{seconds}s"),
        60..3_600 => format!("{}m {:02}s", seconds / 60, seconds % 60),
        _ => format!("{}h {:02}m", seconds / 3_600, seconds % 3_600 / 60),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctrl_q_closes_a_finished_subagents_tab() {
        let _guard = crate::subagent::feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        crate::subagent::feed::clear_for_test();
        crate::subagent::feed::begin(0, "r-run", "running task");
        crate::subagent::feed::begin(1, "r-done", "finished task");
        crate::subagent::feed::finish("r-done", "completed");
        let display = ActiveDisplay::new();
        let press = |tab: &mut Open, c| key_event(tab, &KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL), &display).unwrap();

        let mut running = Open::new(View::Subagent(2));
        assert_eq!(press(&mut running, 'q'), Outcome::Stay);
        assert!(running.note.as_ref().is_some_and(|(text, problem, _)| *problem && text.contains("/kill")), "a running one stays");
        assert_eq!(press(&mut running, 'c'), Outcome::Interrupt, "ctrl+c: as on s0ma's page");

        let mut done = Open::new(View::Subagent(3));
        assert_eq!(press(&mut done, 'c'), Outcome::Interrupt, "ctrl+c closes nothing");
        assert_eq!(crate::subagent::feed::tabs().len(), 2);
        assert_eq!(press(&mut done, 'q'), Outcome::Stay, "the first asks");
        assert!(done.note.as_ref().is_some_and(|(text, _, _)| text.starts_with("[?] closes")), "{:?}", done.note);
        assert_eq!(crate::subagent::feed::tabs().len(), 2);
        assert_eq!(press(&mut done, 'q'), Outcome::Closed);
        let left: Vec<String> = crate::subagent::feed::tabs().into_iter().map(|tab| tab.run_id).collect();
        assert_eq!(left, vec!["r-run".to_string()], "the finished one is gone, the running one kept");
        assert!(!crate::subagent::feed::forget("r-run"), "one still running is never forgotten");
    }
}
