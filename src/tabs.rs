//! Tabs. The chats are on Alt: Alt+1 is s0ma's conversation, Alt+2…Alt+9
//! (and Alt+0) its subagents; Alt+B goes back to the tab before, as it was
//! left. The terminals are on
//! Alt+Shift: Alt+Shift+1… the shells on s0ma's machine (`vm_terms`), the
//! netboxes first. Alt and Alt+Shift arrive as
//! themselves in every terminal (Esc before the key), GNOME Terminal and tmux
//! included, where Ctrl and Ctrl+Shift with a digit do not.
//!
//! s0ma's own page is on the alternate screen with the tab bar on top
//! (`page.rs`). A tab other than s0ma's takes the whole screen in its place
//! and draws itself live: a subagent's reasoning, text, tool calls and results
//! as they happen (`worker_feed`), the open board as it changes, a shell as it
//! is. s0ma's display holds what it writes
//! meanwhile, and draws its page again from its record when Alt+1 or Esc
//! brings it back (`ActiveDisplay::cover`). Either way s0ma keeps working: a
//! turn can start, run and end behind a tab. PgUp or the wheel on s0ma's page
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
use crate::mistake_view::{Line, Style, apply_style};
use crate::terminal::ActiveDisplay;
use crate::worker_feed::{self, Feed, Item, Sender};

#[path = "board_view.rs"]
mod board_view;
#[path = "memory_view.rs"]
mod memory_view;
#[path = "share_view.rs"]
mod share_view;
#[path = "overview_view.rs"]
mod overview_view;
#[path = "projects_view.rs"]
mod projects_view;
#[path = "editor_view.rs"]
mod editor_view;
#[path = "select_view.rs"]
mod select_view;
#[path = "gui_view.rs"]
mod gui_view;
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
    /// Alt+Shift+1…: a shell on s0ma's machine, the netboxes first. `number`
    /// is its key; `session` the shell, once there is one on that key. Kept by name, so keys typed here never
    /// go to another shell that has taken its place.
    Shell { number: usize, session: Option<String> },
    /// s0ma's own conversation, further back than the page shows: PgUp or
    /// the wheel on the page. Scrolled back down to the end, it is the page
    /// again.
    History,
    /// Alt+M: the archive, threads, world model and redactions
    /// (`memory_view.rs`).
    Memory,
    /// Alt+S: /share, the folder the operator and s0ma work on together
    /// (`share_view.rs`).
    Share,
    /// Alt+O: s0ma's settings and state, MCP's servers among them
    /// (`overview_view.rs`).
    Overview,
    /// The projects tab's later subtabs (Alt+P steps through them, the board
    /// being the first): boards in the operator's review, in verification, and
    /// delivered (`projects_view.rs`, `board_flow.rs`).
    Review,
    Verification,
    Deliverables,
    /// Alt+E: files of the shared folder, in s0ma's own editor
    /// (`editor_view.rs`).
    Edit,
    /// s0ma's own page in a pane of its own, while two are up (Alt+\):
    /// the conversation as the page draws it, following the end, and a prompt
    /// to s0ma (a new turn, or steering one under way).
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

/// Alt+1 → s0ma, Alt+2…9 → the subagents, Alt+0 → the tenth tab;
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
        _ => return None,
    };
    Some(match (group, number) {
        (Group::Chat, 1) => Target::Main,
        (Group::Chat, number) => Target::View(View::Subagent(number)),
        (Group::Terminal, number) => {
            Target::View(View::Shell { number, session: crate::vm_terms::sessions().get(number - 1).cloned() })
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
    /// Typed to s0ma in a pane that shows its page: a new turn, or steering
    /// the one under way (`now`: taken at the next steering point).
    Submit { text: String, now: bool },
    /// Back to s0ma: the caller uncovers its display.
    Closed,
    /// Ctrl+C with nothing typed: what it does in s0ma's own pane.
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
    /// Text being selected with the mouse, or just copied (`select_view.rs`).
    selection: Option<select_view::Selection>,
    /// Where the clickable parts of the frame drawn are (`gui_view.rs`), and
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
/// keyboard, `other` the one beside it, each as any tab is (s0ma's page among
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

/// Alt+\ (or Ctrl+Shift+P) on s0ma's page (no tab up): two panes, the page in each, the
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
/// keyboard; or, the first time, `current` on the left and s0ma's page on
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

/// One pane, or two. Back to one: the pane with the keyboard (s0ma's page
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
    crate::vm_terms::touch();
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

/// Go to `target`: a tab opens (covering s0ma's display) or the open one
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
        // From s0ma's page: the tab last up, as it was; from a tab, the one
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

/// The files open in the operator's editor, for s0ma's next message: said
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

/// PgUp or the wheel on s0ma's page: its conversation, `back` rows further
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
/// or closed. With two panes up, a pane closed shows s0ma's page instead.
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
            // The pane shows s0ma's page; Alt+B there comes back.
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
                    crate::vm_terms::send(session, crate::vm_terms::Input::Paste(text));
                    tab.back = 0;
                }
                View::Memory => memory_view::paste(tab, &text),
                View::Share => share_view::paste(tab, &text),
                View::Overview => overview_view::paste(tab, &text),
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

/// The mouse on what can be clicked (`gui_view.rs`): a menu while one is
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
    }
}

/// A turn of the wheel over body row `row` of a tab. The board and the
/// share tab, and the other lists: the wheel walks the list, or scrolls the
/// strip under it, by where it is turned; the rest scroll back.
fn wheel_view(tab: &mut Open, up: bool, row: usize) {
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

/// The mouse selecting text in a tab (`select_view.rs`): the button down
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
        View::Review | View::Verification | View::Deliverables => Ok(projects_view::key(tab, key, page)),
        View::Edit => Ok(editor_view::key(tab, key, page)),
    }
}

/// A key in s0ma's conversation scrolled back: scrolling, and back to the
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
        // before it (Alt+B again comes back). None before: s0ma's page.
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

/// The tab closed last (back to s0ma's page), as it was: Alt+B on the page
/// opens it again.
static CLOSED: Mutex<Option<Open>> = Mutex::new(None);

fn closed() -> MutexGuard<'static, Option<Open>> {
    CLOSED.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Kept as the tab closed last. s0ma's page in a pane, or its history, is
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
                let number = crate::vm_terms::sessions().len() + 1;
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
    if ctrl && !alt && matches!(key.code, KeyCode::Char('q' | 'Q')) && crate::netboxes::name_of(session).is_some() {
        let session = session.clone();
        let name = crate::netboxes::name_of(&session).unwrap_or_default().to_string();
        let busy = crate::vm_terms::netboxes().into_iter().find(|netbox| netbox.name == name).and_then(|netbox| netbox.busy());
        let armed = tab.close_armed.take().is_some_and(|(armed, at)| armed == session && at.elapsed() < CLOSE_AGAIN);
        if !armed {
            tab.close_armed = Some((session, Instant::now()));
            tab.ask(match busy {
                Some(busy) => format!("{busy} · closes it · again to confirm"),
                None => format!("closes {} · again to confirm", crate::netboxes::display(&name)),
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
        && crate::netboxes::name_of(session).is_some()
        && crate::vm_terms::live(session).is_some_and(|screen| at_prompt(&screen.command))
    {
        SHELL_COMPACT.fetch_xor(true, std::sync::atomic::Ordering::Relaxed);
        return Outcome::Stay;
    }
    // Ctrl+\ in a netbox: the operator takes the wheel, or gives it back. (Some
    // terminals send it as Ctrl+4.)
    if ctrl && !alt && matches!(key.code, KeyCode::Char('\\' | '4')) && crate::netboxes::name_of(session).is_some() {
        crate::vm_terms::toggle_wheel(session);
        return Outcome::Stay;
    }
    let input = match key.code {
        // AltGr arrives as Ctrl+Alt on some platforms; its characters are text.
        KeyCode::Char(c) if ctrl && alt && !c.is_ascii_alphabetic() => Some(crate::vm_terms::Input::Text(c.to_string())),
        _ => crate::vm_terms::key_input(key),
    };
    if let Some(input) = input {
        crate::vm_terms::send(session, input);
        tab.back = 0;
    }
    Outcome::Stay
}

/// A key in a chat's tab (a subagent's, the board's): its prompt, scrolling.
fn chat_key(tab: &mut Open, key: &KeyEvent, display: &ActiveDisplay, page: usize) -> io::Result<Outcome> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    // s0ma's page in a pane: Enter sends to s0ma (a new turn, or steering
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

/// A chat's prompt and scrolling (a subagent's, the board's, s0ma's page's).
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

/// The subagent on tab `number` (2 is the first), if there is one.
fn subagent_at(number: usize) -> Option<worker_feed::Tab> {
    number.checked_sub(2).and_then(|index| worker_feed::tabs().into_iter().nth(index))
}

/// The tab number a subagent's run id has, for the board's claims.
fn tab_of(run_id: &str) -> Option<usize> {
    worker_feed::tabs().iter().position(|tab| tab.run_id == run_id).map(|index| index + 2)
}

/// Enter: the draft to the subagent on this tab, or a hint to the board.
fn submit(tab: &mut Open, _display: &ActiveDisplay) {
    let text = tab.draft.text().trim().to_string();
    if text.is_empty() {
        return;
    }
    match tab.view {
        View::Shell { .. } | View::History | View::Memory | View::Share | View::Overview | View::Edit | View::Page => {}
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

/// What the board's prompt adds to the board: a hint for s0ma, a lead of
/// the operator's own, or a finding of his (for the lead, or against it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Adding {
    Hint,
    Lead,
    Supports,
    RulesOut,
}

impl Adding {
    /// The next, for Ctrl+T.
    fn next(self) -> Adding {
        match self {
            Adding::Hint => Adding::Lead,
            Adding::Lead => Adding::Supports,
            Adding::Supports => Adding::RulesOut,
            Adding::RulesOut => Adding::Hint,
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

/// What Enter in a tab does, done by the s0ma that runs the subagents: this
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
    /// The operator's move on a board in the workflow (`board_flow.rs`), by id.
    Project { board: String, done: projects_view::Move },
    /// The overview: the app's latest snapshot (and ask it for a fresh one).
    Overview,
    /// A loop of s0ma's closed by the operator (done, or dropped), with what he said.
    CloseLoop { id: String, done: bool, said: String },
    /// The overview: a command for the app to run as typed (only its own).
    Command { text: String },
    /// Where these strings are kept, as `strike::Impact` in JSON: asked of
    /// the s0ma that keeps the stores, whose skills and worker journal an
    /// attached terminal does not know.
    Survey { fragments: Vec<String> },
    /// What approving request `request` would reach, as JSON: the impact and
    /// how many pieces are too short to look for elsewhere.
    Review { request: u64 },
    /// Strike these strings from every store (the memory tab's marks).
    Strike { fragments: Vec<String> },
    /// s0ma's redaction request: approve it, decline it, or save the
    /// operator's own version of it.
    Approve { request: u64 },
    Decline { request: u64, note: String },
    /// Approval mode: the operator's answer to request `id` (`approval.rs`).
    Answer { id: u64, approve: bool },
    Edit { request: u64, text: String },
    /// What the operator did in the share tab, for his next message to s0ma.
    Shared { change: crate::shared_files::Change },
    /// Typed into a netbox on an attached terminal, for s0ma's next read
    /// (the first netbox's when an older terminal does not say).
    NetboxTyped {
        #[serde(default = "first_netbox")]
        session: String,
        typed: Vec<crate::vm_terms::Input>,
    },
    /// Alt+N on an attached terminal: a netbox of the operator's, opened by the s0ma
    /// that runs the turns; the shell session it will be.
    NetboxOpen,
    /// A finished subagent's tab closed: its feed forgotten everywhere.
    Forget { run_id: String },
    /// Ctrl+Q in a shell's tab: that shell closed, whatever runs in it.
    ShellClose { session: String },
    /// What the sync of `/share` with the operator's machine is doing, and the files
    /// it wrote after mark `since` (the operator's own: not news to him), as
    /// `share_sync::Report` in JSON. It runs in the s0ma that runs the turns.
    ShareSync { since: u64 },
}

type Remote = Box<dyn Fn(Action) -> Result<String, String> + Send + Sync>;
static REMOTE: OnceLock<Remote> = OnceLock::new();

/// On an attached terminal: actions go to `remote` (`session.rs`).
pub fn act_remotely(remote: Remote) {
    let _ = REMOTE.set(remote);
}

/// This is an attached terminal: what it does goes to the s0ma that runs
/// the turns.
pub(crate) fn remote() -> bool {
    REMOTE.get().is_some()
}

fn first_netbox() -> String {
    crate::netboxes::DEFAULT_SESSION.to_string()
}

/// Typed into netbox `session` here, on an attached terminal (`vm_terms`).
pub(crate) fn netbox_typed(session: String, typed: Vec<crate::vm_terms::Input>) {
    let _ = act(Action::NetboxTyped { session, typed });
}

/// On an attached terminal: the sync of `/share` as the s0ma that runs it
/// says, for the share tab (`share_sync::label`, `local_path`), and the
/// files it wrote since last asked, marked seen here too.
pub(crate) fn follow_share_sync() {
    static SINCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if !remote() {
        return;
    }
    let since = SINCE.load(std::sync::atomic::Ordering::SeqCst);
    let Some(report) = act(Action::ShareSync { since })
        .ok()
        .and_then(|json| serde_json::from_str::<crate::share_sync::Report>(&json).ok())
    else {
        return;
    };
    for (dir, path, stamp) in &report.wrote {
        share_view::mark_seen(dir, path, stamp);
    }
    SINCE.store(report.mark, std::sync::atomic::Ordering::SeqCst);
    crate::share_sync::mirror(report);
}

/// `action`, done by the s0ma this terminal shows: here, or sent to it.
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
            crate::shared_files::record(change);
            Ok(String::new())
        }
        Action::NetboxTyped { session, typed } => {
            crate::vm_terms::note_netbox(&session, &typed);
            Ok(String::new())
        }
        Action::NetboxOpen => Ok(crate::netboxes::open_for_operator()),
        Action::ShellClose { session } => crate::netboxes::close_for_operator(&session),
        Action::Forget { run_id } => {
            if worker_feed::forget(&run_id) { Ok("closed".into()) } else { Err("it is still running".into()) }
        }
        Action::ShareSync { since } => serde_json::to_string(&crate::share_sync::report(since)).map_err(|error| error.to_string()),
    }
}

// ------------------------------------------------------------------ drawing

fn draw(open: &mut Option<Open>, display: &ActiveDisplay) -> io::Result<Outcome> {
    let Some(tab) = open.as_mut() else {
        return Ok(Outcome::Stay);
    };
    let size = crossterm::terminal::size()?;
    // The tab bar lists the shells, whichever tab is up.
    crate::vm_terms::touch();
    // What this view shows changed: its own feed only, so a subagent's
    // tokens do not lay out the commands, nor a shell's output a subagent.
    // The bar (shells, workers, running commands) follows at the second's
    // redraw.
    let feed = match &tab.view {
        View::Subagent(_) => worker_feed::version(),
        View::Shell { .. } => crate::vm_terms::version(),
        View::History | View::Page => display.record_revision(),
        View::Board | View::Memory | View::Share | View::Overview | View::Edit => 0,
        View::Review | View::Verification | View::Deliverables => 0,
    };
    let version = feed.wrapping_add(POKED.load(std::sync::atomic::Ordering::SeqCst));
    let now = Instant::now();
    if let Some((drawn_version, drawn_size, at)) = tab.painted
        && !tab.dirty
    {
        let since = now.duration_since(at);
        let changed = drawn_version != version || drawn_size != size;
        let board_due = (tab.view == View::Board && tab.board_read.as_ref().is_none_or(|read| now.duration_since(read.at) >= BOARD_POLL))
            || matches!(tab.view, View::Review | View::Verification | View::Deliverables);
        if since < FRAME || (!changed && !board_due && since < CLOCK) {
            return Ok(Outcome::Stay);
        }
    }
    if size.0 < 20 || size.1 < 6 {
        // Written once per size, not again every second.
        if tab.painted.is_none_or(|(_, drawn, _)| drawn != size) || !tab.frame.is_empty() {
            let mut stdout = io::stdout().lock();
            queue!(stdout, Clear(ClearType::All), MoveTo(0, 0), Print("too small · alt 1 back"))?;
            stdout.flush()?;
        }
        tab.frame.clear();
        tab.painted = Some((version, size, now));
        tab.dirty = false;
        return Ok(Outcome::Stay);
    }
    let (frame, cursor) = compose(tab, display, size, now);
    paint(&tab.frame, &frame, cursor, size.0 as usize)?;
    tab.frame = frame;
    tab.painted = Some((version, size, now));
    tab.dirty = false;
    Ok(Outcome::Stay)
}

/// The screen row a tab's body starts on, as last drawn: under the tab bar's
/// rows and its rule.
static BODY_TOP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2);
/// How many rows the body had, as last drawn.
static BODY_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(20);

pub(super) fn body_top() -> usize {
    BODY_TOP.load(std::sync::atomic::Ordering::Relaxed)
}

/// Every row of the screen, and where the cursor goes: the tab bar, a rule
/// under it, the view, its footer, and on a chat its prompt.
fn compose(tab: &mut Open, display: &ActiveDisplay, (columns, rows): (u16, u16), now: Instant) -> (Vec<Line>, Option<(u16, u16)>) {
    // What can be clicked in it is registered as it is drawn.
    gui_view::begin();
    let (frame, cursor) = compose_frame(tab, display, (columns, rows), now);
    let mut frame = frame;
    if let Some(menu) = tab.menu.as_mut() {
        menu.draw(&mut frame, columns as usize);
    }
    tab.hits = gui_view::hits(&frame);
    tab.size = (columns, rows);
    (frame, cursor)
}

fn compose_frame(tab: &mut Open, display: &ActiveDisplay, (columns, rows): (u16, u16), now: Instant) -> (Vec<Line>, Option<(u16, u16)>) {
    let width = columns as usize;
    let rows = rows as usize;
    // The open subtab's prompt is its board section's.
    let prompt = tab.view.has_prompt() && !(tab.view == View::Board && projects_view::section(tab) != projects_view::Section::Board);
    // What s0ma is doing, beside its key while another tab covers it.
    let working = if matches!(tab.view, View::History | View::Page) { None } else { display.working() };
    let bar = tab_bar(Some(&tab.view), working, width);
    // The body starts under the bar's rows and the rule.
    let top = (bar.len() + 1).min(rows.saturating_sub(4));
    BODY_TOP.store(top, std::sync::atomic::Ordering::Relaxed);
    let body_rows = rows.saturating_sub(top + 1 + usize::from(prompt)).max(1);
    BODY_ROWS.store(body_rows, std::sync::atomic::Ordering::Relaxed);
    if let View::Subagent(number) = tab.view
        && tab.draft.is_empty()
    {
        tab.pinned = subagent_at(number).map(|worker| worker.run_id);
    }
    // A shell's key with no shell on it yet takes the first that comes.
    if !server_only(&tab.view)
        && let View::Shell { number, session: None } = tab.view
    {
        let session = crate::vm_terms::sessions().get(number - 1).cloned();
        tab.view = View::Shell { number, session };
    }
    let mut cursor_in_body = None;
    let mut history_loading = false;
    let (body, footer) = match &tab.view {
        view if server_only(view) => remote_body(view, width),
        View::Subagent(number) => subagent_body(*number, tab.expanded, width, now),
        // The projects tab's first subtab: its line on top, then its
        // sections (the board, its files, notes and reports).
        View::Board if projects_view::section(tab) == projects_view::Section::Board => {
            let (mut body, footer) = board_body(tab, width, body_rows.saturating_sub(2), now);
            let bar = open_item(tab).map(|item| projects_view::section_bar(tab, &item, width)).unwrap_or_default();
            body.insert(0, projects_view::subtab_line(&View::Board, width));
            body.insert(1, bar);
            (body, footer)
        }
        View::Board => {
            board_refresh(tab, now);
            let item = open_item(tab);
            projects_view::open_body(tab, item.as_ref(), width, body_rows)
        }
        View::Review | View::Verification | View::Deliverables => projects_view::body(tab, width, body_rows),
        View::History => history_body(tab, display, width),
        View::Page => page_body(tab, display, width),
        View::Memory => memory_view::body(tab, width, body_rows),
        View::Share => share_view::body(tab, width, body_rows),
        View::Overview => overview_view::body(tab, width, body_rows),
        View::Edit => {
            tab.back = 0;
            let (body, footer, cursor) = editor_view::body(tab, width, body_rows);
            cursor_in_body = cursor;
            (body, footer)
        }
        View::Shell { number, session } => {
            let (body, footer, cursor, loading) = shell_body(*number, session.as_deref(), tab.back, (width, body_rows));
            cursor_in_body = cursor;
            history_loading = loading;
            (body, footer)
        }
    };
    let mut frame = Vec::with_capacity(rows);
    frame.extend(bar.into_iter().take(top.saturating_sub(1)));
    frame.push(vec![(Style::Dim, "─".repeat(width))]);
    let longest = body.len().saturating_sub(body_rows);
    // A shell's history is on its way: keep the place asked for until it is.
    if !history_loading {
        tab.back = tab.back.min(longest);
    }
    let start = longest.saturating_sub(tab.back);
    frame.extend(body.into_iter().skip(start).take(body_rows));
    frame.resize(top + body_rows, Vec::new());
    // Text selected with the mouse: lined up with what this frame shows,
    // and shown.
    if let Some(selection) = tab.selection.as_mut() {
        let shown: Vec<String> =
            frame[top..top + body_rows].iter().map(|row| row.iter().map(|(_, text)| plain_text(text)).collect()).collect();
        selection.take_frame(&shown);
        selection.mark(&mut frame[top..top + body_rows]);
    }
    let Footer { mut left, mut hints } = footer;
    if tab.back > 0 {
        // First, so a narrow footer keeps it.
        left.insert(0, (Style::Dim, format!("↑ {} rows  ", tab.back)));
        if tab.view != View::History {
            hints.insert(0, (if prompt { "end" } else { "⇧end" }, "follow"));
        }
    }
    // What a key or a click just did takes the left of the footer for a
    // while; the buttons stay, to be clicked again.
    let left = match &tab.note {
        Some((text, problem, at)) if now.duration_since(*at) < if text.starts_with("[?] ") { ARM_FOR } else { NOTE_FOR } => {
            let style = match (problem, text.starts_with("[?] ")) {
                (true, _) => Style::Problem,
                (false, true) => Style::Warn,
                (false, false) => Style::Plain,
            };
            vec![(style, text.clone())]
        }
        _ => left,
    };
    let (line, actions) = gui_view::footer(left, &hints, width);
    tab.actions = actions;
    frame.push(line);
    if prompt {
        let (prompt, cursor) = prompt_line(tab, width);
        frame.push(prompt);
        return (frame, Some((cursor as u16, (rows - 1) as u16)));
    }
    // A shell's cursor where the shell has it, when that row is on screen;
    // otherwise none: a cursor parked in a corner reads as one stuck there.
    let cursor = cursor_in_body
        .filter(|_| tab.back == 0)
        .and_then(|(column, row): (usize, usize)| row.checked_sub(start).filter(|row| *row < body_rows).map(|row| (column, row)))
        .map(|(column, row)| (column.min(width.saturating_sub(1)) as u16, (row + top) as u16));
    (frame, cursor)
}

/// A view's footer: what it shows on the left, its keys on the right.
struct Footer {
    left: Line,
    hints: Vec<(&'static str, &'static str)>,
}

impl Footer {
    fn new(left: Line, hints: Vec<(&'static str, &'static str)>) -> Self {
        Self { left, hints }
    }
}


/// A view with nothing to show yet: what it is, and (when it helps) how it fills.
fn empty(title: &str, detail: &str, width: usize) -> Vec<Line> {
    let mut out = vec![Vec::new(), vec![(Style::Plain, format!("  {title}"))]];
    if !detail.is_empty() {
        out.extend(flow(&[(Style::Dim, format!("  {detail}"))], width, 2));
    }
    out
}

/// `segments` wrapped at spaces to `width`, every row after the first
/// indented `indent` columns (the first is too, when it starts a paragraph:
/// pass the indent in its text). A word longer than a row is broken.
fn flow(segments: &[(Style, String)], width: usize, indent: usize) -> Vec<Line> {
    let width = width.max(indent + 8);
    let mut lines: Vec<Line> = vec![Vec::new()];
    let mut used = 0;
    let push = |lines: &mut Vec<Line>, style: Style, text: &str| {
        let line = lines.last_mut().expect("one line");
        match line.last_mut() {
            Some((last, piece)) if *last == style => piece.push_str(text),
            _ => line.push((style, text.to_string())),
        }
    };
    for (style, text) in segments {
        // Words with the spaces after them, so a row never starts with one.
        let mut words: Vec<String> = Vec::new();
        for c in text.chars() {
            let c = if c == '\t' || c == '\n' { ' ' } else { c };
            if c.is_control() {
                continue;
            }
            match words.last_mut() {
                Some(word) if !(c != ' ' && word.ends_with(' ')) => word.push(c),
                _ => words.push(c.to_string()),
            }
        }
        for word in words {
            let bare = text_width(word.trim_end());
            if used > indent && used + bare > width {
                trim_end(lines.last_mut().expect("one line"));
                lines.push(vec![(Style::Plain, " ".repeat(indent))]);
                used = indent;
                if word.trim().is_empty() {
                    continue;
                }
            }
            let mut piece = String::new();
            for c in word.chars() {
                let w = c.width().unwrap_or(0);
                if used + w > width {
                    if c == ' ' {
                        break;
                    }
                    push(&mut lines, *style, &piece);
                    piece.clear();
                    trim_end(lines.last_mut().expect("one line"));
                    lines.push(vec![(Style::Plain, " ".repeat(indent))]);
                    used = indent;
                }
                piece.push(c);
                used += w;
            }
            push(&mut lines, *style, &piece);
        }
    }
    if let Some(last) = lines.last_mut() {
        trim_end(last);
    }
    lines
}

/// A row without the spaces at its end (a wrapped word's).
fn trim_end(line: &mut Line) {
    while let Some((_, text)) = line.last_mut() {
        let kept = text.trim_end_matches(' ').len();
        if kept > 0 {
            text.truncate(kept);
            return;
        }
        line.pop();
    }
}

/// Write the rows that changed since `previous`, then put the cursor on the
/// prompt.
/// One write per frame, as one synchronized update where the terminal
/// supports it (`?2026`): it shows whole frames, never one half drawn.
fn paint(previous: &[Line], frame: &[Line], cursor: Option<(u16, u16)>, width: usize) -> io::Result<()> {
    let mut bytes: Vec<u8> = Vec::with_capacity(16 * 1024);
    queue!(bytes, crossterm::terminal::BeginSynchronizedUpdate)?;
    paint_to(&mut bytes, previous, frame, cursor, width)?;
    queue!(bytes, crossterm::terminal::EndSynchronizedUpdate)?;
    let mut stdout = io::stdout().lock();
    stdout.write_all(&bytes)?;
    stdout.flush()
}

/// `cursor`: where the cursor shows, or `None` to leave it hidden (a view
/// with no text cursor, or a shell whose program hid its own).
fn paint_to(stdout: &mut impl Write, previous: &[Line], frame: &[Line], cursor: Option<(u16, u16)>, width: usize) -> io::Result<()> {
    queue!(stdout, Hide)?;
    for (index, line) in frame.iter().enumerate() {
        if previous.get(index) == Some(line) {
            continue;
        }
        queue!(stdout, MoveTo(0, index as u16))?;
        let mut used = 0;
        'segments: for (style, text) in line {
            apply_style(stdout, *style)?;
            if *style == Style::Raw {
                // Cut to the width already (`page::rows`), codes and all.
                queue!(stdout, Print(text))?;
                used += crate::page::width(text);
                continue;
            }
            let mut piece = String::new();
            for c in text.chars() {
                let c = if c.is_control() { ' ' } else { c };
                let w = c.width().unwrap_or(0);
                if used + w > width {
                    queue!(stdout, Print(piece))?;
                    break 'segments;
                }
                used += w;
                piece.push(c);
            }
            queue!(stdout, Print(piece))?;
        }
        apply_style(stdout, Style::Plain)?;
        // A row that reaches the last column leaves the cursor waiting to
        // wrap, on that column: erasing to the end of the line from there
        // would take its last character with it. Full, it has nothing to
        // erase anyway.
        if used < width {
            queue!(stdout, Clear(ClearType::UntilNewLine))?;
        }
    }
    if let Some((column, row)) = cursor {
        queue!(stdout, MoveTo(column, row), Show)?;
    }
    stdout.flush()
}

/// The tab bar: s0ma's own tabs (alt) on the left half, the shells (alt⇧)
/// on the right, the `│` between them pinned at the middle; a half that
/// runs out of room carries on, on the next row. The tab that is up is in
/// the chevron's cyan.
///
/// ```text
///  alt  1 s0ma  2 check the logs [+]  p projects   │  alt⇧  1 main  2 build
///       m memory  s share  o overview              │
/// ```
///
/// `view`: the tab that is up, or none for s0ma's own page. `working`: what
/// s0ma is doing, shown beside its tab while another covers it.
fn tab_bar(view: Option<&View>, working: Option<String>, width: usize) -> Vec<Line> {
    // Labels as long as they can be on two rows, then shorter, then closer;
    // never more than three rows (a narrow terminal keeps room for the tab).
    let mut best: Vec<Line> = Vec::new();
    for (limit, gap) in [(18, 2), (12, 2), (8, 2), (8, 1), (5, 1)] {
        let bar = tab_bar_with(view, working.as_deref(), limit, gap, width);
        let fewer = best.is_empty() || bar.len() < best.len();
        if fewer {
            best = bar;
        }
        if best.len() <= 2 {
            break;
        }
    }
    best.truncate(MAX_BAR_ROWS);
    best
}

const MAX_BAR_ROWS: usize = 3;

/// The tab bar for s0ma's own page, where `1 s0ma` is the one that is up:
/// written out with its colours, a row to a line, and how many columns its
/// widest row takes. Above the idle prompt and atop the steering pane; the
/// page keeps as many rows for it (`page::set_bar_rows`).
pub fn main_bar(width: usize) -> (String, usize) {
    // The VM's shells are read once at start for it (shells from before a
    // restart); after that, when s0ma opens or closes one and while a tab is up.
    static FIRST_LIST: std::sync::Once = std::sync::Once::new();
    FIRST_LIST.call_once(|| {
        if !cfg!(test) {
            crate::vm_terms::touch();
        }
    });
    gui_view::begin();
    let rows = tab_bar(None, None, width);
    crate::page::set_bar_rows(rows.len());
    *page_hits() = gui_view::hits(&rows);
    let cells = rows.iter().map(|line| line.iter().map(|(_, text)| text_width(text)).sum::<usize>()).max().unwrap_or(0);
    let mut written: Vec<String> = Vec::new();
    for line in &rows {
        let mut out: Vec<u8> = Vec::new();
        for (style, text) in line {
            let _ = apply_style(&mut out, *style);
            out.extend_from_slice(text.as_bytes());
        }
        let _ = apply_style(&mut out, Style::Plain);
        written.push(String::from_utf8_lossy(&out).into_owned());
    }
    (written.join("\n"), cells)
}

/// Where the page's tab bar (`main_bar`) can be clicked, as last made.
static PAGE_HITS: Mutex<Vec<gui_view::Hit>> = Mutex::new(Vec::new());

fn page_hits() -> MutexGuard<'static, Vec<gui_view::Hit>> {
    PAGE_HITS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A click on s0ma's own page at `(column, row)`: on its tab bar, the key
/// that opens the tab clicked (Alt+A for the approval mark), as the page
/// takes keys.
pub fn page_click(column: u16, row: u16) -> Option<KeyEvent> {
    let hits = page_hits();
    match &gui_view::at(&hits, row as usize, column as usize)?.click {
        gui_view::Click::Keys(keys) => keys.first().copied(),
        gui_view::Click::Approve => Some(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::ALT)),
        _ => None,
    }
}

/// One tab in the bar: its label, and a mark after it; a click on it opens
/// the tab, as its key does (`key`, Alt's; `terminal`, Alt+Shift's).
fn bar_item(up: bool, key: char, label: String, style: Style, mark: Option<(Style, &str)>, gap: usize) -> Line {
    bar_item_in(Group::Chat, up, key, label, style, mark, gap)
}

fn bar_item_in(group: Group, up: bool, key: char, label: String, style: Style, mark: Option<(Style, &str)>, gap: usize) -> Line {
    let modifiers = if group == Group::Terminal { KeyModifiers::ALT | KeyModifiers::SHIFT } else { KeyModifiers::ALT };
    let click = gui_view::Click::Keys(vec![KeyEvent::new(KeyCode::Char(key), modifiers)]);
    // The approval mark is its own button: a click is one press of Alt+A.
    let approve = |mark: &str| gui_view::hot(Style::Warn, format!(" {mark}"), gui_view::Click::Approve);
    let mut line = Line::new();
    if up {
        line.push((Style::Plain, " ".repeat(gap - 1)));
        match mark {
            Some((_, text)) if text.contains("approve") => {
                line.push(gui_view::hot(Style::Tab, format!(" {label}"), click));
                line.push(approve(text));
                line.push((Style::Tab, " ".into()));
            }
            _ => {
                let mark = mark.map(|(_, mark)| format!(" {mark}")).unwrap_or_default();
                line.push(gui_view::hot(Style::Tab, format!(" {label}{mark} "), click));
            }
        }
    } else {
        line.push((Style::Plain, " ".repeat(gap)));
        line.push(gui_view::hot(style, label, click.clone()));
        match mark {
            Some((_, text)) if text.contains("approve") => line.push(approve(text)),
            Some((style, text)) => line.push(gui_view::hot(style, format!(" {text}"), click)),
            None => {}
        }
        line.push((Style::Plain, " ".into()));
    }
    line
}

/// `items` after `lead`, flowed into rows `room` wide; a row after the
/// first starts under the first item.
fn flow_items(lead: &str, items: Vec<Line>, room: usize) -> Vec<Line> {
    let width = |line: &Line| line.iter().map(|(_, text)| text_width(text)).sum::<usize>();
    let indent = text_width(lead);
    let mut rows: Vec<Line> = vec![vec![(Style::Dim, lead.to_string())]];
    for item in items {
        let used = rows.last().map(width).unwrap_or(0);
        if used + width(&item) > room && used > indent {
            rows.push(vec![(Style::Plain, " ".repeat(indent))]);
        }
        rows.last_mut().expect("one row at least").extend(item);
    }
    rows
}

fn tab_bar_with(view: Option<&View>, working: Option<&str>, limit: usize, gap: usize, width: usize) -> Vec<Line> {
    let mut left: Vec<Line> = Vec::new();
    // [!]: something waits for the operator's approval (alt+a twice).
    let approve = (!crate::approval::pending().is_empty()).then_some((Style::Warn, "[!] approve"));
    let mut first = bar_item(view.is_none_or(|view| matches!(view, View::History | View::Page)), '1', "s0ma".into(), Style::Plain, approve, gap);
    if let Some(working) = working {
        first.pop();
        first.push((Style::Status, format!(" · {} ", clip(working, limit.max(12)))));
    }
    left.push(first);
    let key_of = |number: usize| if number == 10 { '0' } else { char::from(b'0' + number as u8) };
    let workers = worker_feed::tabs();
    for (index, worker) in workers.iter().enumerate().take(9) {
        let number = index + 2;
        let mark = match worker.status.as_deref() {
            None => None,
            Some("completed") => Some((Style::Added, "[+]")),
            Some(_) => Some((Style::Removed, "[-]")),
        };
        let label = clip(&worker.label, limit);
        let style = if worker.running { Style::Plain } else { Style::Dim };
        left.push(bar_item(view == Some(&View::Subagent(number)), key_of(number), label, style, mark, gap));
    }
    // A key with nothing on it is shown only while it is the one up, so
    // the bar says where you are.
    if let Some(View::Subagent(number)) = view
        && *number >= workers.len().min(9) + 2
    {
        left.push(bar_item(true, key_of(*number), "empty".into(), Style::Plain, None, gap));
    }
    // [!]: a board waits for the operator: in review, or verified.
    let waiting = (boards_waiting() > 0).then_some((Style::Warn, "[!]"));
    let projects = view.is_some_and(projects_view::is_projects);
    left.push(bar_item(projects, 'p', "projects".into(), Style::Plain, waiting, gap));
    // [!]: s0ma asks to redact something (the memory tab's redactions).
    let asked = (redactions_waiting() > 0).then_some((Style::Warn, "[!]"));
    left.push(bar_item(view == Some(&View::Memory), 'm', "memory".into(), Style::Plain, asked, gap));
    left.push(bar_item(view == Some(&View::Share), 's', "share".into(), Style::Plain, None, gap));
    // The editor, while a file is open in it (or it is up): the file, how
    // many are open, and [+] for unsaved changes.
    match editor_view::label() {
        Some((name, count, dirty)) => {
            let label = if count > 1 { format!("{} ·{count}", clip(&name, limit)) } else { clip(&name, limit) };
            left.push(bar_item(view == Some(&View::Edit), 'e', label, Style::Plain, dirty.then_some((Style::Warn, "[+]")), gap));
        }
        None if view == Some(&View::Edit) => left.push(bar_item(true, 'e', "edit".into(), Style::Plain, None, gap)),
        None => {}
    }
    left.push(bar_item(view == Some(&View::Overview), 'o', "overview".into(), Style::Plain, None, gap));
    let mut right: Vec<Line> = Vec::new();
    let sessions = crate::vm_terms::sessions();
    let netboxes = crate::vm_terms::netboxes();
    for (index, session) in sessions.iter().enumerate().take(10) {
        let number = index + 1;
        let up = matches!(view, Some(View::Shell { session: Some(shown), .. }) if shown == session);
        right.push(bar_item_in(Group::Terminal, up, key_of(number), clip(&shell_label(session), limit.max(8)), Style::Plain, netbox_mark(&netboxes, session), gap));
    }
    if let Some(View::Shell { number, session: None }) = view {
        right.push(bar_item_in(Group::Terminal, true, key_of(*number), "empty".into(), Style::Plain, None, gap));
    }
    // The halves, and the pipe between them on every row.
    let half = width.saturating_sub(1) / 2;
    let left = flow_items("", left, half);
    let right = flow_items("", right, width.saturating_sub(half + 1));
    let rows = left.len().max(right.len());
    (0..rows)
        .map(|at| {
            let mut line: Line = left.get(at).cloned().unwrap_or_default();
            let used: usize = line.iter().map(|(_, text)| text_width(text)).sum();
            line.push((Style::Plain, " ".repeat(half.saturating_sub(used))));
            line.push((Style::Dim, "│".into()));
            line.extend(right.get(at).cloned().unwrap_or_default());
            line
        })
        .collect()
}

/// A netbox's mark in the tab bar: `[!]` the operator has the wheel, `[*]` a
/// command of s0ma's is running in it.
fn netbox_mark(netboxes: &[crate::netboxes::Netbox], session: &str) -> Option<(Style, &'static str)> {
    let name = crate::netboxes::name_of(session)?;
    let netbox = netboxes.iter().find(|netbox| netbox.name == name)?;
    if netbox.wheel {
        Some((Style::Warn, "[!]"))
    } else if !netbox.running.is_empty() {
        Some((Style::Work, "[*]"))
    } else {
        None
    }
}

/// A shell as the tab bar names it: by its name (`w·b3-1` a subagent's).
fn shell_label(session: &str) -> String {
    match crate::netboxes::name_of(session) {
        Some(name) => crate::netboxes::display(name),
        None => clip(session, 16),
    }
}

/// What the operator greyed out on each board, by board id: for his
/// reading only (s0ma never sees it). Kept beside the board's database, so it
/// stays as he left it.
static GREYED: Mutex<Option<Greyed>> = Mutex::new(None);

/// When the greyed file was read, and the rows greyed on each board.
type Greyed = (Option<std::time::SystemTime>, HashMap<String, HashSet<String>>);

fn grey_path() -> Option<PathBuf> {
    ARCHIVE.get().map(|archive| archive.with_file_name("board-grey.json"))
}

fn grey_stamp(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|metadata| metadata.modified()).ok()
}

/// `change` made to what is greyed on `board`, and kept. The file is read
/// again whenever another s0ma (an attached terminal) wrote it since, so
/// neither writes over the other's; a file that does not read is left as it
/// is and what this one had is kept; it is written whole, beside, and then
/// renamed over the old.
fn with_greyed<T>(board: &str, change: impl FnOnce(&mut HashSet<String>) -> T) -> T {
    let mut kept = GREYED.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let path = grey_path();
    let stamp = path.as_deref().and_then(grey_stamp);
    let stale = kept.as_ref().is_none_or(|(read_at, _)| *read_at != stamp);
    if stale {
        let read = path.as_deref().and_then(|path| std::fs::read_to_string(path).ok()).and_then(|json| serde_json::from_str(&json).ok());
        match (read, kept.as_mut()) {
            (Some(all), _) => *kept = Some((stamp, all)),
            (None, Some((read_at, _))) => *read_at = stamp,
            (None, None) => *kept = Some((stamp, HashMap::new())),
        }
    }
    let (read_at, all) = kept.as_mut().expect("read above");
    let before = all.get(board).cloned().unwrap_or_default();
    let set = all.entry(board.to_string()).or_default();
    let out = change(set);
    let changed = *set != before;
    if set.is_empty() {
        all.remove(board);
    }
    if changed && let (Some(path), Ok(json)) = (path, serde_json::to_string(&*all))
        && crate::util::write_atomic(&path, json).is_ok() {
            *read_at = grey_stamp(&path);
        }
    out
}

/// The board's tree, greyed out as the operator left it.
fn grey_tree(snapshot: &board_view::Snapshot, now: chrono::DateTime<chrono::Local>) -> board_view::Tree {
    let mut tree = board_view::Tree::build(snapshot, now);
    with_greyed(&snapshot.board.id, |greyed| tree.sink(greyed));
    tree
}

/// The selected lead or finding's short id, for the prompt.
fn board_target(tab: &Open) -> Option<String> {
    let snapshot = tab.board_read.as_ref()?.snapshot.as_ref().ok()?.as_ref()?;
    let tree = grey_tree(snapshot, chrono::Local::now());
    let node = tree.find(tab.board_selected.as_deref()?)?;
    board_view::prompt_target(&tree, Some(node))
}

fn prompt_line(tab: &Open, width: usize) -> (Line, usize) {
    let (label, live) = match &tab.view {
        View::Page => (" you ›".to_string(), true),
        View::Shell { .. } | View::History | View::Memory | View::Share | View::Overview | View::Edit => return (Vec::new(), 0),
        View::Review | View::Verification | View::Deliverables => return (Vec::new(), 0),
        View::Subagent(number) => match subagent_at(*number) {
            // Named for the one the draft is for, which `submit` checks.
            Some(worker) if worker.running => {
                let to = tab.pinned.as_deref().filter(|_| !tab.draft.is_empty()).unwrap_or(&worker.run_id);
                (format!(" {} ›", short_id(to)), true)
            }
            Some(_) => (" finished".to_string(), false),
            None => (String::new(), false),
        },
        View::Board => match tab.board_read.as_ref().and_then(|read| read.name.as_deref()) {
            // Named for what it adds, and the lead or finding it will be
            // about (a lead hangs from a finding only).
            Some(_) => {
                let target = board_target(tab);
                let label = match (tab.board_adding, target) {
                    (Adding::Hint, Some(target)) => format!(" hint on {target} ›"),
                    (Adding::Hint, None) => " hint ›".to_string(),
                    (Adding::Lead, Some(target)) if target.starts_with('f') => format!(" lead from {target} ›"),
                    (Adding::Lead, _) => " lead ›".to_string(),
                    (Adding::Supports, Some(target)) => format!(" finding on {target}, supports ›"),
                    (Adding::Supports, None) => " finding, supports ›".to_string(),
                    (Adding::RulesOut, Some(target)) => format!(" finding on {target}, rules out ›"),
                    (Adding::RulesOut, None) => " finding, rules out ›".to_string(),
                };
                (label, true)
            }
            None => (String::new(), false),
        },
    };
    if !live {
        return (vec![(Style::Dim, label)], 0);
    }
    let label_width = text_width(&label) + 1;
    let mut line: Line = vec![(Style::Operator, label), (Style::Plain, " ".into())];
    // One row: a newline shows as ⏎, and the text scrolls to keep the cursor
    // in sight.
    let shown: String = tab.draft.text().chars().map(|c| if c == '\n' { '⏎' } else { c }).collect();
    let before: String = tab.draft.text()[..tab.draft.cursor()].chars().map(|c| if c == '\n' { '⏎' } else { c }).collect();
    let room = width.saturating_sub(label_width + 1).max(1);
    let cursor_at = text_width(&before);
    let skip = cursor_at.saturating_sub(room.saturating_sub(1));
    let mut visible = String::new();
    let mut at = 0;
    for c in shown.chars() {
        let w = c.width().unwrap_or(0);
        if at >= skip && at + w <= skip + room {
            visible.push(c);
        }
        at += w;
    }
    if visible.is_empty() {
        let hint = match (&tab.view, tab.board_adding) {
            (View::Board, Adding::Hint) => "hint for s0ma · ^t a lead or finding",
            (View::Board, Adding::Lead) => "a lead of yours to explore (p1–p10 first: its priority)",
            (View::Board, _) => "what you found",
            (View::Page, _) => "to s0ma",
            _ => "a direction, taken before its next step",
        };
        line.push((Style::Work, hint.to_string()));
        return (line, label_width.min(width.saturating_sub(1)));
    }
    line.push((Style::Plain, visible));
    (line, (label_width + cursor_at - skip).min(width.saturating_sub(1)))
}

// ------------------------------------------------------------------ a subagent's tab

/// A tab that reads s0ma's own files and state from the disk of the machine
/// it runs on, shown in a terminal attached from another computer (`s0ma
/// connect`, `session::is_remote`): those reads don't go over the
/// connection yet. The page, its history and the subagents' tabs do, as
/// the primary sends them.
fn server_only(view: &View) -> bool {
    crate::session::is_remote() && !matches!(view, View::Subagent(_) | View::History | View::Page)
}

/// What a tab `server_only` shows instead.
fn remote_body(view: &View, width: usize) -> (Vec<Line>, Footer) {
    let what = match view {
        View::Shell { .. } => "Shells",
        View::Memory => "The memory tab",
        View::Share => "The shared folder's tab",
        View::Overview => "The overview",
        View::Edit => "The editor",
        _ => "The projects tab",
    };
    let text = [
        format!("{what} reads the files of the machine s0ma runs on, and a terminal attached with s0ma connect can't show it yet."),
        String::new(),
        "For it now, ssh to that machine and run s0ma there. /download and /upload move files from and to the shared folder from here.".to_string(),
    ];
    let mut body = vec![Vec::new()];
    for paragraph in text {
        if paragraph.is_empty() {
            body.push(Vec::new());
            continue;
        }
        body.extend(crate::mistake_view::wrap(&[(Style::Dim, format!("  {paragraph}"))], width));
    }
    (body, Footer::new(vec![(Style::Dim, "not available here".into())], vec![]))
}

fn subagent_body(number: usize, expanded: bool, width: usize, now: Instant) -> (Vec<Line>, Footer) {
    let Some(worker) = subagent_at(number) else {
        let key = if number == 10 { 0 } else { number };
        let body = empty(&format!("No subagent on alt {key}."), "", width);
        return (body, Footer::new(Vec::new(), vec![("esc", "back")]));
    };
    worker_feed::with_feed(worker.slot, |feed| feed_lines(feed, expanded, width, now, feed.paused.as_deref()))
        .unwrap_or_else(|| (Vec::new(), Footer::new(Vec::new(), vec![("esc", "back")])))
}

/// s0ma's label in a subagent's tab, where the subagent's own name has the
/// green s0ma has on its page.
const DIRECTOR_STYLE: &str = "\x1b[1;97m";

/// A subagent's feed as rows, and its footer: drawn by the renderer s0ma's own
/// page uses, so it reads the same, with Ctrl+O its expanded form (reasoning
/// and every tool line).
fn feed_lines(feed: &Feed, expanded: bool, width: usize, now: Instant, paused: Option<&str>) -> (Vec<Line>, Footer) {
    use crate::terminal::{NAME_STYLE, OPERATOR_STYLE, OPERATOR_TEXT_STYLE, RESET, STATUS_STYLE, TEXT_STYLE, safe_text};
    use crate::transcript::{Detail, Entry, Mark, Out, Renderer, WORK_STYLE};
    let id = short_id(&feed.run_id);
    let detail = if expanded { Detail::Expanded } else { Detail::Collapsed };
    let mut renderer = Renderer::new(detail, crate::transcript::DEFAULT_CHARS_PER_TOKEN).named(&id, NAME_STYLE);
    let mut text = String::new();
    let mut out = Out { text: &mut text, columns: width, wrapping: true };
    let start = feed.started;
    renderer.apply(&mut out, start, start, &Entry::Turn);
    // The task is s0ma's first direction to it.
    let task_lines: Vec<&str> = feed.task.lines().filter(|line| !line.trim().is_empty()).collect();
    let shown = if expanded { task_lines.len() } else { TASK_LINES.min(task_lines.len()) };
    let mut task = safe_text(&task_lines[..shown].join("\n"));
    if shown < task_lines.len() {
        task.push_str(&format!("\n{WORK_STYLE}{} more lines{RESET}", task_lines.len() - shown));
    }
    renderer.speaker_line(&mut out, "s0ma", DIRECTOR_STYLE, TEXT_STYLE, &task);
    if feed.dropped {
        renderer.apply(&mut out, start, start, &Entry::Line(format!("{WORK_STYLE}earlier activity is in the worker journal{RESET}")));
    }
    let direction = |renderer: &mut Renderer, out: &mut Out, item: &Item| {
        let Item::Direction { from, text, wrap_up, delivered, .. } = item else {
            return;
        };
        let mut body = safe_text(text);
        if *wrap_up {
            body.push_str(&format!("{WORK_STYLE}  wrap up{RESET}"));
        }
        match (delivered, feed.running()) {
            (true, _) => {}
            (false, true) => body.push_str(&format!("{WORK_STYLE}  queued{RESET}")),
            (false, false) => body.push_str(&format!("{RESET}\x1b[31m  not taken: it had finished{RESET}")),
        }
        match from {
            Sender::Operator => renderer.speaker_line(out, "you", OPERATOR_STYLE, OPERATOR_TEXT_STYLE, &body),
            Sender::S0ma => renderer.speaker_line(out, "s0ma", DIRECTOR_STYLE, TEXT_STYLE, &body),
        }
        // Its next words are a new message, under its name again.
        renderer.apply(out, start, start, &Entry::Turn);
    };
    // The call in progress: its name, and when it began.
    let mut calling: Option<(String, Instant)> = None;
    for (index, item) in feed.items.iter().enumerate() {
        let (first, last) = feed.time(index);
        match item {
            Item::Reasoning(text) => renderer.apply(&mut out, first, last, &Entry::Reasoning(safe_text(text))),
            Item::Text(text) => {
                let mut markdown = crate::markdown::MarkdownStream::new();
                let mut styled = markdown.push(&safe_text(text));
                styled.push_str(&markdown.finish());
                renderer.apply(&mut out, first, last, &Entry::Answer(styled));
            }
            Item::Call { name, arguments } => {
                calling = Some((name.clone(), first));
                let status = safe_text(&format!("[tool] running: {name}, arguments: {arguments}"));
                renderer.apply(&mut out, first, last, &Entry::Status(status));
            }
            Item::Result(raw) => {
                let (name, began) = calling.take().unwrap_or_else(|| ("tool".to_string(), first));
                let (lines, problem) = result_gist(raw);
                let status = match &problem {
                    Some(problem) => format!("[tool] failed: {name}: {problem}"),
                    None => format!("[tool] finished: {name} ({:.1}s)", last.duration_since(began).as_secs_f64()),
                };
                renderer.apply(&mut out, first, last, &Entry::Status(safe_text(&status)));
                // Expanded, what it gave back, in short.
                for line in lines.iter().take(RESULT_LINES_EXPANDED) {
                    renderer.apply(&mut out, first, last, &Entry::Note(format!("{STATUS_STYLE}  {}{RESET}", safe_text(line))));
                }
                if lines.len() > RESULT_LINES_EXPANDED {
                    let more = format!("{WORK_STYLE}  {} more lines{RESET}", lines.len() - RESULT_LINES_EXPANDED);
                    renderer.apply(&mut out, first, last, &Entry::Note(more));
                }
            }
            Item::Direction { .. } => direction(&mut renderer, &mut out, item),
            Item::Note(text) => {
                let text = safe_text(text);
                renderer.apply(&mut out, first, last, &Entry::Notice { brief: text.clone(), full: text, mark: Mark::Work });
            }
        }
    }
    let mut left: Line = Vec::new();
    match &feed.finished {
        None => {
            left.push((Style::Status, format!("{id} · running {}", span(now.duration_since(feed.started)))));
            if let Some(paused) = paused {
                left.push((Style::Problem, format!(" · {paused}, waiting for s0ma")));
            }
        }
        Some((at, how)) => {
            let after = format!("{how} after {}", span(at.duration_since(feed.started)));
            let mark = if how == "completed" { Mark::Done } else { Mark::Problem };
            renderer.apply(&mut out, *at, *at, &Entry::Notice { brief: after.clone(), full: after.clone(), mark });
            left.push((Style::Status, format!("{id} · {after}")));
        }
    }
    // Sent and not taken in yet: after everything it is in the middle of.
    for item in &feed.waiting {
        direction(&mut renderer, &mut out, item);
    }
    if feed.running() {
        renderer.refresh(&mut out, now, true);
    } else {
        renderer.finish(&mut out);
    }
    let mut rows: Vec<Line> = vec![Vec::new()];
    rows.extend(crate::page::rows(&text, width).rows.into_iter().map(|row| vec![(Style::Raw, format!("{row}{RESET}"))]));
    while rows.last().is_some_and(|row| row.iter().all(|(_, text)| crate::page::width(text) == 0)) && rows.len() > 1 {
        rows.pop();
    }
    let waiting = feed.waiting_directions();
    if waiting > 0 {
        left.push((Style::Dim, format!(" · {waiting} queued")));
    }
    let detail = if expanded { "less" } else { "more" };
    let hints = if feed.running() {
        vec![("⏎", "steer"), ("/wrap", ""), ("/kill", ""), ("^o", detail), ("esc", "back")]
    } else {
        vec![("^o", detail), ("^q", "close"), ("esc", "back")]
    };
    (rows, Footer::new(left, hints))
}

/// A result in short: its output's lines, and what went wrong, if anything.
/// The envelope's bookkeeping (sizes, flags, ids) is for Ctrl+O.
fn result_gist(raw: &str) -> (Vec<String>, Option<String>) {
    let Ok(Value::Object(map)) = serde_json::from_str::<Value>(raw) else {
        return (raw.trim_end().lines().map(str::to_string).collect(), None);
    };
    let text = |key: &str| map.get(key).and_then(Value::as_str).map(str::trim_end).filter(|text| !text.trim().is_empty());
    let mut problem = None;
    if let Some(code) = map.get("exit_code").and_then(Value::as_i64).filter(|code| *code != 0) {
        problem = Some(format!("exit {code}"));
    }
    if map.get("ok").and_then(Value::as_bool) == Some(false) || map.get("error").is_some_and(|error| !error.is_null()) {
        let why = text("error").or_else(|| text("message")).unwrap_or("failed");
        problem = Some(match problem {
            Some(exit) => format!("{exit} · {}", crate::mistakes::one_line(why, 200)),
            None => crate::mistakes::one_line(why, 200),
        });
    }
    let mut lines: Vec<String> = Vec::new();
    if let Some(body) = ["stdout", "output", "result", "content", "text", "screen"].iter().find_map(|key| text(key)) {
        lines.extend(body.lines().map(str::to_string));
    }
    if problem.is_some()
        && let Some(stderr) = text("stderr")
    {
        lines.extend(stderr.lines().map(str::to_string));
    }
    (lines, problem)
}

// ------------------------------------------------------------------ the board's tab

/// The board's tab: its head, the search as a tree to walk (`board_view`),
/// and the selected node in full in a strip under it. Laid out to fill
/// `rows` exactly, the tree scrolled to keep the selection in sight.
fn board_body(tab: &mut Open, width: usize, rows: usize, now: Instant) -> (Vec<Line>, Footer) {
    board_refresh(tab, now);
    let Some(read) = tab.board_read.take() else {
        return (empty("Reading the board…", "", width), Footer::new(Vec::new(), vec![("esc", "back")]));
    };
    // Out of the tab while it is laid out (the tree's folds and selection are
    // the tab's to change), and back after. Under the subtab line and the
    // section bar.
    let laid_out = board_layout(tab, &read, width, rows, 2, false);
    tab.board_read = Some(read);
    laid_out
}

/// The open board, read again on a thread when the last read is stale.
fn board_refresh(tab: &mut Open, now: Instant) {
    // A read that came in: kept, unless it is for another board than the
    // one up now. One that failed keeps what was read before on screen (a
    // worker mid-write), until the board has failed for a while.
    if let Some((index, slot)) = &tab.board_reading
        && let Some(read) = slot.lock().ok().and_then(|mut slot| slot.take())
    {
        let index = *index;
        tab.board_reading = None;
        if index == tab.board {
            if read.snapshot.is_ok() {
                tab.board_failing = None;
                tab.board_read = Some(read);
            } else {
                let since = *tab.board_failing.get_or_insert(now);
                let keep_old = now.duration_since(since) < BOARD_GIVE_UP
                    && tab.board_read.as_ref().is_some_and(|old| old.snapshot.is_ok());
                if keep_old {
                    // What was read before stays up; asked again at the next poll.
                    if let Some(old) = tab.board_read.as_mut() {
                        old.at = now;
                    }
                } else {
                    tab.board_read = Some(read);
                }
            }
        }
    }
    // Read again, on a thread: a write-locked store (a worker writing the
    // board) never holds the tab up.
    let stale = tab.board_read.as_ref().is_none_or(|read| now.duration_since(read.at) >= BOARD_POLL);
    if stale && tab.board_reading.is_none() {
        let index = tab.board;
        let archive = ARCHIVE.get().cloned();
        let slot = std::sync::Arc::new(Mutex::new(None));
        let answer = slot.clone();
        let spawned = std::thread::Builder::new().name("board-tab".into()).spawn(move || {
            let read = match archive {
                Some(archive) => read_board(&crate::board::BoardStore::beside_archive(&archive), index)
                    .unwrap_or_else(|error| BoardRead { at: Instant::now(), snapshot: Err(error.to_string()), name: None, open: 0, waiting: 0 }),
                None => BoardRead {
                    at: Instant::now(),
                    snapshot: Err("the board is not available in this process".into()),
                    name: None,
                    open: 0,
                    waiting: 0,
                },
            };
            if let Ok(mut answer) = answer.lock() {
                *answer = Some(read);
            }
            poke();
        });
        if spawned.is_ok() {
            tab.board_reading = Some((index, slot));
        }
    }
}

/// The open board as the open subtab's other sections show it: its files,
/// notes and reports.
fn open_item(tab: &Open) -> Option<projects_view::Item> {
    let snapshot = tab.board_read.as_ref()?.snapshot.as_ref().ok()?.as_ref()?;
    let mut flow = snapshot.flow.clone().unwrap_or_default();
    if flow.board_id.is_empty() {
        flow.board_id = snapshot.board.id.clone();
        flow.name = snapshot.board.name.clone();
    }
    Some(projects_view::Item { board: snapshot.board.clone(), flow, notes: snapshot.notes.clone(), snapshot: None })
}

/// A board as a section of a later subtab (review, verification,
/// complete): its tree and the strip under it, in `rows` rows from row
/// `above` of the tab's body, walked as on the board's own tab. Its keys.
fn embedded_board(tab: &mut Open, snapshot: &board_view::Snapshot, width: usize, rows: usize, above: usize) -> (Vec<Line>, Vec<(&'static str, &'static str)>) {
    // Another board: its tree as new.
    if tab.projects.board_for.as_deref() != Some(snapshot.board.id.as_str()) {
        tab.projects.board_for = Some(snapshot.board.id.clone());
        tab.folds = board_view::Folds::default();
        tab.board_selected = None;
        tab.board_top = 0;
        tab.strip_scroll = 0;
        tab.strip_for = None;
    }
    let read = BoardRead { at: Instant::now(), snapshot: Ok(Some(snapshot.clone())), name: Some(snapshot.board.name.clone()), open: 1, waiting: 0 };
    let (lines, footer) = board_layout(tab, &read, width, rows, above, true);
    // Kept for its keys (`board_section_key`).
    tab.board_read = Some(read);
    (lines, footer.hints)
}

/// A key in a board section of a later subtab: Shift+↑↓ the strip, the
/// tree's own walking and folding keys. Whether it was one.
fn board_section_key(tab: &mut Open, key: &KeyEvent, page: usize) -> bool {
    strip_key(tab, key, page) || board_walk(tab, key, page)
}

/// On the board, Shift scrolls the strip with the selected node in full;
/// whether `key` did.
fn strip_key(tab: &mut Open, key: &KeyEvent, page: usize) -> bool {
    if key.modifiers != KeyModifiers::SHIFT {
        return false;
    }
    let scroll = match key.code {
        KeyCode::Up => tab.strip_scroll.saturating_sub(1),
        KeyCode::Down => tab.strip_scroll.saturating_add(1),
        KeyCode::PageUp => tab.strip_scroll.saturating_sub(page / 2),
        KeyCode::PageDown => tab.strip_scroll.saturating_add(page / 2),
        KeyCode::Home => 0,
        KeyCode::End => usize::MAX,
        _ => return false,
    };
    // Kept within the strip when it is laid out.
    tab.strip_scroll = scroll;
    true
}

/// A turn of the wheel at `row` of the tab's body over a board: below the
/// strip's rule it scrolls the strip; over the tree, the tree scrolls
/// first, and the selection moves with it only when it would leave the rows
/// shown, and on its own once the tree is at its end.
fn board_wheel(tab: &mut Open, up: bool, row: usize, page: usize) {
    if row >= tab.strip_at {
        tab.strip_scroll = if up { tab.strip_scroll.saturating_sub(WHEEL_ROWS) } else { tab.strip_scroll.saturating_add(WHEEL_ROWS) };
        return;
    }
    let len = tab.board_shown.len();
    let rows = tab.board_tree_rows.max(1);
    let last_top = len.saturating_sub(rows);
    let at = tab.board_selected.as_ref().and_then(|id| tab.board_shown.iter().position(|shown| shown == id));
    let scrolls = if up { tab.board_top > 0 } else { tab.board_top < last_top };
    if scrolls {
        tab.board_top = if up { tab.board_top.saturating_sub(WHEEL_ROWS) } else { (tab.board_top + WHEEL_ROWS).min(last_top) };
        if let Some(at) = at {
            let kept = at.clamp(tab.board_top, (tab.board_top + rows).saturating_sub(1).min(len.saturating_sub(1)));
            if kept != at {
                tab.board_selected = tab.board_shown.get(kept).cloned();
                tab.strip_scroll = 0;
            }
        }
    } else {
        let code = if up { KeyCode::Up } else { KeyCode::Down };
        board_walk(tab, &KeyEvent::new(code, KeyModifiers::NONE), page);
    }
}

/// The board, laid out in `rows` rows from row `at_row` of the tab's body:
/// its head (on its own tab), the tree, the strip with the selected node in
/// full. `embedded`: a section of a later subtab, under its list of boards
/// (no head, and only the tree's keys).
fn board_layout(tab: &mut Open, read: &BoardRead, width: usize, rows: usize, at_row: usize, embedded: bool) -> (Vec<Line>, Footer) {
    let footer = |left: Line, hints: Vec<(&'static str, &'static str)>| Footer::new(left, hints);
    let snapshot = match &read.snapshot {
        Err(error) => {
            return (
                vec![Vec::new(), vec![(Style::Problem, format!("  the board could not be read: {error}"))]],
                footer(Vec::new(), vec![("esc", "back")]),
            );
        }
        Ok(None) => {
            return (
                empty("No open board.", "", width),
                footer(Vec::new(), vec![("esc", "back")]),
            );
        }
        Ok(Some(snapshot)) => snapshot,
    };
    let clock = chrono::Local::now();
    let tree = grey_tree(snapshot, clock);
    tab.folds.update(&tree, snapshot);
    let shown = board_view::rows(&tree, &tab.folds);
    let selected = selected_row(tab, &tree, &shown);

    let mut out = if embedded { Vec::new() } else { board_view::header(snapshot, width, clock) };
    // Finished: s0ma's report is due, then it goes to the operator's review.
    let proposed_exhausted = snapshot.board.status == crate::board::BoardStatus::ProposedExhausted;
    if embedded {
    } else if matches!(snapshot.board.status, crate::board::BoardStatus::ProposedSolved | crate::board::BoardStatus::Solved) {
        out.extend(board_view::write_up_due(snapshot, width));
    } else if proposed_exhausted {
        out.extend(board_view::exhaustion_proposed(snapshot, width));
    }
    let left_rows = rows.saturating_sub(out.len());
    // The strip under the tree: the selected node in full, as many rows as it
    // needs up to two fifths of what is left (at least six when there is
    // room, three when short, none when very short) or all the tree leaves
    // over; past that it scrolls.
    let strip_cap = match left_rows {
        12.. => (left_rows * 2 / 5).max(6),
        7..12 => 3,
        _ => 0,
    };
    // And whatever the tree leaves over (a short tree, a long text).
    let strip_cap = if strip_cap == 0 { 0 } else { strip_cap.max(left_rows.saturating_sub(shown.len().max(1) + 1)) };
    let full = match selected {
        Some(index) if strip_cap > 0 => board_view::detail(&tree, snapshot, shown[index].node, width, clock),
        _ => Vec::new(),
    };
    if tab.strip_for != tab.board_selected {
        tab.strip_for = tab.board_selected.clone();
        tab.strip_scroll = 0;
    }
    let shown_rows = full.len().min(strip_cap);
    tab.strip_scroll = tab.strip_scroll.min(full.len() - shown_rows);
    let (above, below) = (tab.strip_scroll, full.len() - shown_rows - tab.strip_scroll);
    let strip: Vec<Line> = full.into_iter().skip(above).take(shown_rows).collect();
    let strip_rows = if strip.is_empty() { 0 } else { strip.len() + 1 };
    let tree_rows = left_rows.saturating_sub(strip_rows).max(1);
    if let Some(index) = selected {
        if index < tab.board_top {
            tab.board_top = index;
        } else if index >= tab.board_top + tree_rows {
            tab.board_top = index + 1 - tree_rows;
        }
    }
    tab.board_top = tab.board_top.min(shown.len().saturating_sub(tree_rows));
    tab.board_shown = shown.iter().map(|row| tree.nodes[row.node].id.clone()).collect();
    tab.board_tree_rows = tree_rows;
    if shown.is_empty() {
        out.push(vec![(Style::Dim, "  No leads yet.".into())]);
    }
    for (index, row) in shown.iter().enumerate().skip(tab.board_top).take(tree_rows) {
        let line = board_view::row_line(&tree, snapshot, &tab.folds, row, Some(index) == selected, width, clock);
        out.push(gui_view::pick_row(line, gui_view::Click::Pick(gui_view::Pick::Board(index))));
    }
    // Where the strip starts in the tab's body (under the subtab line).
    tab.strip_at = if strip.is_empty() { usize::MAX } else { at_row + rows.saturating_sub(strip_rows) };
    if !strip.is_empty() {
        out.resize(rows.saturating_sub(strip_rows), Vec::new());
        // The rule says when there is more of the strip than it shows.
        let more = match (above, below) {
            (0, 0) => String::new(),
            (0, below) => format!(" {below} more "),
            (above, 0) => format!(" {above} above "),
            (above, below) => format!(" {above} above · {below} more "),
        };
        let rule = width.saturating_sub(4 + text_width(&more));
        out.push(vec![(Style::Dim, format!("  {}", "─".repeat(rule))), (Style::Cursor, more)]);
        out.extend(strip);
    }
    out.truncate(rows);

    // ←→ step the sections, as in every other section, and ⏎ folds (on the
    // open board, while no hint is being typed).
    let mut hints: Vec<(&'static str, &'static str)> = vec![("↑↓", "move")];
    if embedded || tab.draft.is_empty() {
        hints.push(("⏎", "fold"));
    }
    if above + below > 0 {
        hints.push(("⇧↑↓", "detail"));
    }
    if !tab.draft.is_empty() {
        hints.push(("⏎", "send"));
    }
    if embedded {
        if let Some(index) = selected.filter(|&index| board_view::can_grey(&tree, shown[index].node)) {
            hints.push(("^c", if tree.grey[shown[index].node] { "ungrey" } else { "grey" }));
        }
        return (out, footer(Vec::new(), hints));
    }
    // One of the operator's notes: Delete takes it off (twice).
    if tab.draft.is_empty() && selected.is_some_and(|index| matches!(tree.nodes[shown[index].node].kind, board_view::Kind::Hint(_))) {
        hints.push(("del", "remove"));
    }
    if tab.draft.is_empty() && selected.is_some_and(|index| matches!(tree.nodes[shown[index].node].kind, board_view::Kind::Board)) {
        hints.push(("del", "discard"));
    }
    if tab.draft.is_empty() && selected.is_some_and(|index| board_view::can_grey(&tree, shown[index].node)) {
        hints.push(("^c", if tree.grey[shown[selected.unwrap()].node] { "ungrey" } else { "grey" }));
    }
    if read.open > 1 {
        hints.push(("^n ^p", "board"));
    }
    hints.push(("^q", "close"));
    if tab.draft.is_empty() {
        hints.push(("←→", "section"));
    }
    hints.push(("tab", "subtab"));
    hints.push(("esc", "back"));
    let mut left = match (&read.name, read.open) {
        (Some(_), open) if open > 1 => vec![(Style::Status, format!("board {} of {open}", tab.board % open + 1))],
        (Some(name), _) => vec![(Style::Status, name.clone())],
        (None, _) => Vec::new(),
    };
    if read.waiting > 0 {
        left.push((Style::Warn, format!("  [!] {} in review", read.waiting)));
    }
    (out, footer(left, hints))
}

/// The selected row's index among `shown`: the node selected before, by id,
/// or the first row. Records it, so it stays put as rows come and go.
fn selected_row(tab: &mut Open, tree: &board_view::Tree, shown: &[board_view::Row]) -> Option<usize> {
    let wanted = tab.board_selected.as_deref().and_then(|id| tree.find(id));
    // A node folded away under its parent: the nearest ancestor shown.
    let mut node = wanted;
    let index = loop {
        match node {
            Some(at) => match shown.iter().position(|row| row.node == at) {
                Some(index) => break Some(index),
                None => node = tree.nodes[at].parent,
            },
            // At first, the first lead or finding: the row after the board's
            // own and the hints under it.
            None => {
                let first = shown.iter().skip(1).position(|row| row.guides.is_empty()).map(|index| index + 1);
                break (!shown.is_empty()).then(|| first.unwrap_or(if shown.len() > 1 { 1 } else { 0 }));
            }
        }
    };
    tab.board_selected = index.map(|index| tree.nodes[shown[index].node].id.clone());
    index
}

/// The board as the tab has it now: its tree and the rows shown.
fn board_tree(tab: &mut Open) -> Option<(board_view::Tree, Vec<board_view::Row>, Option<usize>)> {
    let read = tab.board_read.take()?;
    let found = read.snapshot.as_ref().ok().and_then(Option::as_ref).map(|snapshot| {
        let tree = grey_tree(snapshot, chrono::Local::now());
        tab.folds.update(&tree, snapshot);
        let shown = board_view::rows(&tree, &tab.folds);
        let selected = selected_row(tab, &tree, &shown);
        (tree, shown, selected)
    });
    tab.board_read = Some(read);
    found
}

/// Ctrl+C, with nothing typed: the selected row greyed out, or back. For
/// the operator's reading only; s0ma's board is as it was. Whether there
/// was a row to grey.
fn board_grey(tab: &mut Open) -> bool {
    if !tab.draft.is_empty() {
        return false;
    }
    let board = tab.board_read.as_ref().and_then(|read| read.snapshot.as_ref().ok()?.as_ref()).map(|snapshot| snapshot.board.id.clone());
    let (Some((tree, shown, Some(index))), Some(board)) = (board_tree(tab), board) else {
        return false;
    };
    let node = shown[index].node;
    if !board_view::can_grey(&tree, node) {
        return false;
    }
    tab.board_unhint = None;
    with_greyed(&board, |greyed| board_view::toggle_grey(&tree, greyed, node));
    // The chevron stays where it was, on the row that comes up into the
    // place, so the next Ctrl+C takes the next one.
    if let Some((tree, shown, _)) = board_tree(tab) {
        tab.board_selected = shown.get(index.min(shown.len().saturating_sub(1))).map(|row| tree.nodes[row.node].id.clone());
    }
    true
}

/// A key on the board's tab. The arrows walk the tree, unless a hint is being
/// typed and they are the draft's: ↑↓ move, → unfolds (or steps into an
/// unfolded node), ← folds (or steps out to the parent), ⏎ with nothing typed
/// folds or unfolds, Home and End go to the first and last rows, Ctrl+O
/// unfolds or folds everything. The rest is the prompt's, as on any chat.
fn board_key(tab: &mut Open, key: &KeyEvent, display: &ActiveDisplay, page: usize) -> io::Result<Outcome> {
    let plain = !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
    // Ctrl+Q: once to ask, again to close the board (discarded: closed,
    // abandoned, out of the projects tab). Any other key keeps it.
    let ctrl_only = key.modifiers.contains(KeyModifiers::CONTROL) && !key.modifiers.contains(KeyModifiers::ALT);
    if ctrl_only && matches!(key.code, KeyCode::Char('q' | 'Q')) {
        let board = tab.board_read.as_ref().and_then(|read| read.snapshot.as_ref().ok()?.as_ref()).map(|snapshot| (snapshot.board.id.clone(), snapshot.board.name.clone()));
        let armed = tab.board_unhint.take();
        if let Some((id, name)) = board {
            let mark = format!("close:{id}");
            if armed.as_deref() == Some(mark.as_str()) {
                match act(Action::Project { board: id, done: projects_view::Move::Discard }) {
                    Ok(reply) => tab.note(reply, false),
                    Err(reply) => tab.note(reply, true),
                }
                tab.board_read = None;
                tab.board_selected = None;
            } else {
                tab.ask(format!("ctrl+q again closes {name}: abandoned, and out of the projects tab"));
                tab.board_unhint = Some(mark);
            }
        }
        return Ok(Outcome::Stay);
    }
    if ctrl_only && matches!(key.code, KeyCode::Char('c' | 'C')) && board_grey(tab) {
        return Ok(Outcome::Stay);
    }
    // Ctrl+T: what the prompt adds, the next of hint, lead, finding.
    if ctrl_only && matches!(key.code, KeyCode::Char('t' | 'T')) {
        tab.board_unhint = None;
        tab.board_adding = tab.board_adding.next();
        return Ok(Outcome::Stay);
    }
    // Delete, with nothing typed, on one of the operator's notes: once to ask,
    // again to take it off the board. Any other key keeps it.
    let armed = tab.board_unhint.take();
    if plain && key.code == KeyCode::Delete && tab.draft.is_empty() {
        let picked = board_tree(tab).and_then(|(tree, shown, selected)| {
            let node = &tree.nodes[shown[selected?].node];
            match node.kind {
                board_view::Kind::Hint(_) => Some((false, node.id.clone())),
                board_view::Kind::Board => Some((true, node.id.clone())),
                _ => None,
            }
        });
        // On the board's own row: once to ask, again to discard the board.
        if let Some((true, row)) = &picked {
            let id = tab.board_read.as_ref().and_then(|read| read.snapshot.as_ref().ok()?.as_ref()).map(|snapshot| snapshot.board.id.clone());
            let name = tab.board_read.as_ref().and_then(|read| read.name.clone()).unwrap_or_default();
            if let Some(id) = id {
                let mark = format!("{row}:{id}");
                if armed.as_deref() == Some(mark.as_str()) {
                    match act(Action::Project { board: id, done: projects_view::Move::Discard }) {
                        Ok(reply) => tab.note(reply, false),
                        Err(reply) => tab.note(reply, true),
                    }
                    tab.board_read = None;
                    tab.board_selected = None;
                } else {
                    tab.ask(format!("delete again discards {name}: it is closed, and leaves the projects tab"));
                    tab.board_unhint = Some(mark);
                }
            }
            return Ok(Outcome::Stay);
        }
        let note = picked.map(|(_, id)| id);
        let board = tab.board_read.as_ref().and_then(|read| read.name.clone());
        match (note, board) {
            (Some(id), Some(board)) if armed.as_deref() == Some(id.as_str()) => {
                match act(Action::Unhint { board, id }) {
                    Ok(reply) => tab.note(reply, false),
                    Err(reply) => tab.note(reply, true),
                }
                // Read again at once, to show it gone.
                if let Some(read) = tab.board_read.as_mut() {
                    read.at = Instant::now().checked_sub(BOARD_POLL).unwrap_or(read.at);
                }
            }
            (Some(id), Some(_)) => {
                tab.ask(format!("delete again takes {} off the board", board_view::short(&id)));
                tab.board_unhint = Some(id);
            }
            _ => {}
        }
        return Ok(Outcome::Stay);
    }
    if board_walk(tab, key, page) {
        return Ok(Outcome::Stay);
    }
    chat_key(tab, key, display, page)
}

/// The keys that walk the board's tree (↑↓ PgUp PgDn, and with nothing
/// typed Home End, ⏎ to fold) and Ctrl+O; whether `key` was one of them.
/// ←→ step the sections before a key comes here.
/// A click on a node of the board's tree: it is selected (its row as last
/// drawn, `board_shown`).
fn board_pick(tab: &mut Open, index: usize) {
    if let Some(id) = tab.board_shown.get(index).cloned()
        && tab.board_selected.as_ref() != Some(&id)
    {
        tab.board_selected = Some(id);
        tab.strip_scroll = 0;
    }
}

fn board_walk(tab: &mut Open, key: &KeyEvent, page: usize) -> bool {
    let plain = !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SHIFT);
    let ctrl_o = key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('o' | 'O'));
    let walks = plain
        && match key.code {
            KeyCode::Up | KeyCode::Down | KeyCode::PageUp | KeyCode::PageDown => true,
            KeyCode::Home | KeyCode::End | KeyCode::Enter => tab.draft.is_empty(),
            _ => false,
        };
    if !(walks || ctrl_o) {
        return false;
    }
    let Some((tree, shown, selected)) = board_tree(tab) else {
        return ctrl_o;
    };
    if ctrl_o {
        tab.folds.toggle_all(&tree);
        return true;
    }
    let Some(index) = selected else {
        return true;
    };
    let node = shown[index].node;
    let has_children = !tree.nodes[node].children.is_empty();
    let open = tab.folds.is_open(&tree, node);
    let go = |tab: &mut Open, index: usize| {
        let index = index.min(shown.len() - 1);
        tab.board_selected = Some(tree.nodes[shown[index].node].id.clone());
    };
    match key.code {
        KeyCode::Up => go(tab, index.saturating_sub(1)),
        KeyCode::Down => go(tab, index + 1),
        KeyCode::PageUp => go(tab, index.saturating_sub(page)),
        KeyCode::PageDown => go(tab, index + page),
        KeyCode::Home => go(tab, 0),
        KeyCode::End => go(tab, shown.len() - 1),
        KeyCode::Enter if has_children => tab.folds.set(&tree, node, !open),
        _ => {}
    }
    true
}

/// The open board at `index` (wrapping) and what is on it, or `None` in the
/// snapshot when no board is open.
fn read_board(store: &crate::board::BoardStore, index: usize) -> Result<BoardRead, Box<dyn std::error::Error>> {
    let boards = shown_boards(store)?;
    let waiting = waiting_for_operator(store);
    WAITING.store(waiting, std::sync::atomic::Ordering::SeqCst);
    if boards.is_empty() {
        return Ok(BoardRead { at: Instant::now(), snapshot: Ok(None), name: None, open: 0, waiting });
    }
    let board = boards[index % boards.len()].clone();
    let flows = crate::board_flow::FlowStore::beside(store);
    let snapshot = board_view::Snapshot {
        intents: store.intents(&board.id)?,
        facts: store.facts(&board.id)?,
        hints: store.hints(&board.id, false)?,
        flow: flows.get(&board.id)?,
        notes: flows.notes(&board.id)?,
        board,
    };
    Ok(BoardRead {
        at: Instant::now(),
        name: Some(snapshot.board.name.clone()),
        snapshot: Ok(Some(snapshot)),
        open: boards.len(),
        waiting,
    })
}

/// The boards on the board subtab, newest first: the open ones, and those
/// s0ma finished whose write-up is due (still working, in `board_flow`; a
/// proposal from before the workflow too). Those written up are in review
/// and on (`projects_view.rs`).
fn shown_boards(store: &crate::board::BoardStore) -> Result<Vec<crate::board::Board>, Box<dyn std::error::Error>> {
    use crate::board::BoardStatus;
    use crate::board_flow::Stage;
    let flows: std::collections::HashMap<String, crate::board_flow::Flow> =
        crate::board_flow::FlowStore::beside(store).all()?.into_iter().map(|flow| (flow.board_id.clone(), flow)).collect();
    Ok(store
        .boards(false)?
        .into_iter()
        .filter(|board| match (board.closed_at.is_none(), board.status, flows.get(&board.id)) {
            (true, _, _) => true,
            // Finished, or s0ma's word that nothing is left: here until it
            // is written up, then in the operator's review.
            (false, BoardStatus::Solved | BoardStatus::ProposedSolved | BoardStatus::ProposedExhausted, Some(flow)) => flow.stage() == Stage::Working,
            (false, BoardStatus::ProposedSolved | BoardStatus::ProposedExhausted, None) => true,
            _ => false,
        })
        .collect())
}

/// Boards waiting for the operator: in review, or verified.
fn waiting_for_operator(store: &crate::board::BoardStore) -> usize {
    crate::board_flow::waiting(&crate::board_flow::FlowStore::beside(store))
}

/// How many boards wait for the operator (in review, verified, proposed exhausted), as last counted.
static WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// When they were last counted for the tab bar (at most every `WAITING_EVERY`).
static WAITING_COUNTED: Mutex<Option<Instant>> = Mutex::new(None);
const WAITING_EVERY: Duration = Duration::from_secs(10);

/// Boards waiting for the operator, for the tab bar's mark: the last count, and a
/// new one taken on a thread now and then (the store may be busy), with the
/// projects subtabs' counts.
/// s0ma's redaction requests waiting for the operator, as last counted (with
/// the boards, `boards_waiting`).
static REDACTIONS_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn redactions_waiting() -> usize {
    REDACTIONS_WAITING.load(std::sync::atomic::Ordering::SeqCst)
}

fn boards_waiting() -> usize {
    let due = {
        let mut counted = WAITING_COUNTED.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let due = counted.is_none_or(|at| at.elapsed() >= WAITING_EVERY);
        if due {
            *counted = Some(Instant::now());
        }
        due
    };
    if due && let Some(archive) = ARCHIVE.get().cloned() {
        let _ = std::thread::Builder::new().name("boards-waiting".into()).spawn(move || {
            let store = crate::board::BoardStore::beside_archive(&archive);
            let waiting = waiting_for_operator(&store);
            // And s0ma's redaction requests, for the memory tab's mark.
            let asked = crate::redact::Book::beside_archive(&archive).pending().map(|pending| pending.len()).unwrap_or(0);
            if REDACTIONS_WAITING.swap(asked, std::sync::atomic::Ordering::SeqCst) != asked {
                poke();
            }
            if let Ok((_, counts)) = projects_view::read(&store, crate::board_flow::Stage::Deliverable, None) {
                projects_view::set_counts(counts);
            }
            if WAITING.swap(waiting, std::sync::atomic::Ordering::SeqCst) != waiting {
                poke();
            }
        });
    }
    WAITING.load(std::sync::atomic::Ordering::SeqCst)
}

// ------------------------------------------------------------------ s0ma's, further back

/// s0ma's conversation, as the page draws it, for looking back through.
fn history_body(tab: &mut Open, display: &ActiveDisplay, width: usize) -> (Vec<Line>, Footer) {
    if let Err(error) = history_rows(tab, display, width) {
        return (empty("The conversation can't be shown.", &error.to_string(), width), Footer::new(Vec::new(), vec![("esc", "back")]));
    }
    let rows = tab.history.as_ref().map(|history| history.rows.clone()).unwrap_or_default();
    let detail = if tab.expanded { "less" } else { "more" };
    (rows, Footer::new(vec![(Style::Dim, "s0ma, further back".into())], vec![("esc", "back"), ("^o", detail)]))
}

/// s0ma's page in a pane: its conversation as the page draws it, following
/// the end (the wheel and PgUp scroll it back), and what s0ma is doing.
fn page_body(tab: &mut Open, display: &ActiveDisplay, width: usize) -> (Vec<Line>, Footer) {
    if let Err(error) = history_rows(tab, display, width) {
        return (empty("The conversation can't be shown.", &error.to_string(), width), Footer::new(Vec::new(), Vec::new()));
    }
    let mut rows = tab.history.as_ref().map(|history| history.rows.clone()).unwrap_or_default();
    let working = display.working();
    if let Some(working) = &working {
        rows.push(vec![(Style::Work, format!("  [*] {working}"))]);
    }
    let left = match working {
        Some(_) => vec![(Style::Status, "s0ma · what is sent steers it".to_string())],
        None => vec![(Style::Status, "s0ma".to_string())],
    };
    let detail = if tab.expanded { "less" } else { "more" };
    let mut hints = vec![("⏎", "send")];
    if working.is_some() {
        hints.push(("^s", "send now"));
    }
    hints.extend([("^o", detail), ("⇧pgup", "scroll")]);
    if is_split() {
        hints.extend([("alt ←→", "pane"), ("alt \\", "one pane")]);
    }
    (rows, Footer::new(left, hints))
}

/// Bring `View::History`'s rows up to date; how many there are. While
/// scrolled back, what is being read stays put as rows are added at the end
/// (and, in a long session, dropped from the front: the redraw starts from a
/// later point), found again by its rows (`reanchor`).
fn history_rows(tab: &mut Open, display: &ActiveDisplay, width: usize) -> io::Result<usize> {
    let from = (display.record_revision(), width, tab.expanded);
    // Laid out again at most once a second while the record grows (a turn
    // streaming): each is the whole conversation.
    if let Some(history) = &tab.history
        && (history.from == from
            || ((history.from.1, history.from.2) == (width, tab.expanded) && history.built.elapsed() < HISTORY_EVERY))
    {
        return Ok(history.rows.len());
    }
    let rows: Vec<Line> =
        display.conversation_rows(width, tab.expanded)?.into_iter().map(|row| vec![(Style::Raw, row)]).collect();
    if let Some(history) = &tab.history
        && (history.from.1, history.from.2) == (width, tab.expanded)
        && tab.back > 0
    {
        tab.back = reanchor(&history.rows, &rows, tab.back);
    }
    let count = rows.len();
    tab.history = Some(HistoryRows { from, rows, built: Instant::now() });
    Ok(count)
}

/// Where the view `back` rows from the end of `old` is in `new`, as rows back
/// from its end: found by the rows ending there (eight, or fewer), the nearest to
/// where they would be with nothing dropped from the front; failing that,
/// moved back by however many rows were added.
fn reanchor(old: &[Line], new: &[Line], back: usize) -> usize {
    let grown = new.len().saturating_sub(old.len());
    let Some(at) = old.len().checked_sub(1 + back) else {
        return back + grown;
    };
    let expected = (at + grown).min(new.len().saturating_sub(1));
    // Fewer rows when those before were dropped from the front too.
    for size in [8, 4, 2, 1] {
        let window = &old[(at + 1).saturating_sub(size)..=at];
        for end in (window.len().saturating_sub(1)..=expected).rev() {
            if new[end + 1 - window.len()..=end] == *window {
                return new.len() - 1 - end;
            }
        }
    }
    back + grown
}

// ------------------------------------------------------------------ a shell's tab

/// A shell in the VM as it is now (or its history, scrolled back), its footer,
/// where its cursor is among the rows, and whether the history asked for is
/// still being read.
fn shell_body(number: usize, session: Option<&str>, back: usize, (width, rows): (usize, usize)) -> (Vec<Line>, Footer, Option<(usize, usize)>, bool) {
    let back_hints = vec![("alt 1", "back")];
    let Some(session) = session else {
        let key = if number == 10 { 0 } else { number };
        let _ = key;
        let mut body = empty("No shell here.", "", width);
        if let Some(error) = crate::vm_terms::list_error() {
            body.push(Vec::new());
            body.extend(flow(&[(Style::Problem, format!("  The VM's shells could not be listed: {error}"))], width, 2));
        } else if !crate::vm_terms::listed() {
            body.push(Vec::new());
            body.push(vec![(Style::Work, "  reading the VM's shells…".into())]);
        }
        return (body, Footer::new(Vec::new(), back_hints), None, false);
    };
    crate::vm_terms::watch(session);
    crate::vm_terms::fit(session, width, rows);
    crate::vm_terms::want_history(session, back > 0);
    let history = (back > 0).then(|| crate::vm_terms::history(session)).flatten();
    let loading = back > 0 && history.is_none() && crate::vm_terms::error(session).is_none();
    let screen = history.or_else(|| crate::vm_terms::live(session));
    let mut left: Line = vec![(Style::Status, shell_label(session))];
    let Some(screen) = screen else {
        let text = match crate::vm_terms::error(session) {
            Some(error) => format!("  The shell could not be read: {error}"),
            None => "  reading the shell…".into(),
        };
        return (vec![Vec::new(), vec![(Style::Work, text)]], Footer::new(left, back_hints), None, loading);
    };
    // A netbox says who has it, and what it is for.
    let netbox = crate::netboxes::name_of(session)
        .and_then(|name| crate::vm_terms::netboxes().into_iter().find(|netbox| netbox.name == name));
    // Folded (Ctrl+O): a line a command.
    let compacted = netbox.as_ref().filter(|_| SHELL_COMPACT.load(std::sync::atomic::Ordering::Relaxed)).map(|netbox| {
        compact(&screen.lines, screen.cursor, &crate::netboxes::display(&netbox.name), screen.size.0.max(width))
    });
    let (lines, cursor) = match &compacted {
        Some((lines, cursor)) => (lines.as_slice(), *cursor),
        None => (screen.lines.as_slice(), screen.cursor),
    };
    // The pane's rows, less the blank ones under the cursor and the text,
    // in the pane's own colours.
    let last_text = lines.iter().rposition(|line| !plain_text(line).trim().is_empty()).unwrap_or(0);
    let end = (last_text.max(cursor.1) + 1).min(lines.len());
    let body: Vec<Line> = lines[..end].iter().map(|line| shell_row(line, width)).collect();
    if !screen.cwd.is_empty() {
        left.push((Style::Dim, format!("  {}", screen.cwd)));
    }
    if screen.size.0 > width {
        left.push((Style::Problem, format!("  {} columns cut off", screen.size.0 - width)));
    }
    if screen.dead {
        left.push((Style::Problem, "  exited".into()));
    }
    match crate::vm_terms::error(session) {
        Some(error) => left.push((Style::Problem, format!("  {error}"))),
        None if loading => left.push((Style::Work, "  reading its history…".into())),
        None => {}
    }
    if let Some(netbox) = &netbox {
        left.insert(1, (Style::Dim, format!("  as {}", netbox.account())));
        if netbox.wheel {
            left.insert(0, (Style::Warn, "[!] you have control  ".into()));
        } else if !netbox.running.is_empty() {
            left.insert(0, (Style::Work, "[*] s0ma is running  ".into()));
        }
        if netbox.asks_first() {
            left.push((Style::Dim, "  confirm".into()));
        }
        if !netbox.purpose.is_empty() {
            left.push((Style::Dim, format!("  · {}", netbox.purpose)));
        }
    }
    let hints = if netbox.is_some() {
        let fold = if compacted.is_some() { "expand" } else { "collapse" };
        vec![("⇧pgup", "scroll"), ("ctrl+o", fold), ("ctrl+\\", "control"), ("ctrl+q", "close"), ("alt n", "new"), ("alt 1", "back")]
    } else {
        vec![("⇧pgup", "scroll"), ("alt n", "new"), ("alt 1", "back")]
    };
    // A program that hid its cursor (a pager, an editor's menu) has none.
    (body, Footer::new(left, hints), screen.cursor_shown.then_some(cursor), loading)
}

/// The shells' tabs show each prompt folded to one line, until Ctrl+O at a
/// prompt opens them out (for every shell, until pressed again).
static SHELL_COMPACT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// A shell, not a program in it, has the pane.
fn at_prompt(command: &str) -> bool {
    matches!(command, "bash" | "sh" | "dash" | "zsh" | "fish" | "")
}

/// A netbox's rows with each prompt folded to one line, as the shell's
/// settings draw it (`netboxes::RC`): `account@name  folder` (and how the
/// last command went, on the right) over `$ command` becomes `account@name
/// folder $ command`; a failure is said in it (`[-] 101`), a success not at
/// all; s0ma's reason, on its own line between the two, goes after the
/// command when the row has room. `name` is the shell's, as its prompt says
/// it; `width` the pane's. The cursor moves with its row.
fn compact(lines: &[String], cursor: (usize, usize), name: &str, width: usize) -> (Vec<String>, (usize, usize)) {
    const RED: &str = "\x1b[38;2;255;95;95m";
    // `account@name  folder…`: the prompt's first line.
    let is_meta = |plain: &str| {
        plain.split_once(' ').is_some_and(|(first, rest)| {
            first.len() > name.len() + 1 && first.ends_with(name) && first[..first.len() - name.len()].ends_with('@') && rest.starts_with(' ')
        })
    };
    let is_reason = |plain: &str| plain.starts_with("# ") && plain.contains(": ");
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let (mut column, mut row) = cursor;
    let mut i = 0;
    while i < lines.len() {
        let raw = &lines[i];
        let plain = plain_text(raw);
        if !is_meta(&plain) {
            out.push(raw.clone());
            i += 1;
            continue;
        }
        // The reason, if any, then `$ `.
        let reason = (i + 1 < lines.len() && is_reason(&plain_text(&lines[i + 1]))).then(|| i + 1);
        let at = reason.map_or(i + 1, |reason| reason + 1);
        if at >= lines.len() || !plain_text(&lines[at]).starts_with('$') {
            out.push(raw.clone());
            i += 1;
            continue;
        }
        // The first line without what it says on the right.
        let trimmed = plain.trim_end();
        let (left, said) = match trimmed.rfind("  [") {
            Some(cut) if trimmed[cut + 2..].starts_with("[+] ") || trimmed[cut + 2..].starts_with("[-] ") => {
                (trimmed[..cut].trim_end(), Some(&trimmed[cut + 2..]))
            }
            _ => (trimmed, None),
        };
        let failed = said.and_then(|said| said.strip_prefix("[-] ")).map(|rest| {
            let words: Vec<&str> = rest.split([' ', '·']).filter(|word| !word.is_empty()).collect();
            match words.as_slice() {
                ["exit", code, ..] => code.to_string(),
                [word, ..] => word.to_string(),
                [] => "failed".into(),
            }
        });
        let head = cut_at_width(raw, text_width(left));
        // `folder $`: one space between, as in a one-line prompt.
        let head = match head.rfind("  ") {
            Some(gap) if plain_text(&head[gap..]).starts_with("  ") => format!("{}{}", &head[..gap], &head[gap + 1..]),
            _ => head,
        };
        let mut merged = format!("{head}\x1b[0m ");
        let mut added = text_width(&plain_text(&head)) + 1;
        if let Some(code) = &failed {
            merged.push_str(&format!("{RED}[-] {code}\x1b[0m "));
            added += text_width(&format!("[-] {code} "));
        }
        // A command that fills its row (it wraps) is left on its own: put
        // after the first line, it would run off the edge.
        if added + text_width(plain_text(&lines[at]).trim_end()) > width {
            out.push(raw.clone());
            i += 1;
            continue;
        }
        merged.push_str(&lines[at]);
        // The reason after the command, when it fits and isn't where the cursor is.
        let mut keep_reason = None;
        if let Some(reason) = reason {
            let why = plain_text(&lines[reason]);
            let used = text_width(plain_text(&merged).trim_end());
            if at != cursor.1 && used + 2 + text_width(why.trim_end()) <= width {
                merged = format!("{}\x1b[0m  \x1b[3;90m{}\x1b[0m", cut_at_width(&merged, used), why.trim_end());
            } else {
                keep_reason = Some(lines[reason].clone());
            }
        }
        // Rows before the cursor that went: the meta line, and the reason if merged.
        let gone = 1 + usize::from(reason.is_some() && keep_reason.is_none());
        if cursor.1 >= at {
            row -= gone.min(row);
            if cursor.1 == at {
                column += added;
            }
        } else if cursor.1 > i {
            row = out.len();
        }
        if let Some(reason) = keep_reason {
            out.push(reason);
        }
        out.push(merged);
        i = at + 1;
    }
    (out, (column, row))
}

/// `raw` up to its first `width` columns of text, escape codes kept: a row
/// without the blanks tmux padded it with.
fn cut_at_width(raw: &str, width: usize) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut used = 0;
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            out.push(c);
            out.extend(chars.next());
            for next in chars.by_ref() {
                out.push(next);
                if ('\x40'..='\x7e').contains(&next) {
                    break;
                }
            }
            continue;
        }
        let w = c.width().unwrap_or(0);
        if used + w > width {
            break;
        }
        used += w;
        out.push(c);
    }
    out
}

/// A row of a shell as tmux keeps it, its colours and all, cut to `width`.
/// The terminal's default colour is taken as the tabs' white: a reset, or a
/// `39`, would otherwise give the terminal's own (green, on a green one).
fn shell_row(line: &str, width: usize) -> Line {
    let row = crate::page::rows(&format!("\x1b[37m{}", whiten(line)), width).rows.into_iter().next().unwrap_or_default();
    vec![(Style::Raw, format!("{row}\x1b[0m"))]
}

/// `text`'s colour codes with the default foreground made white: `0` (or an
/// empty code) is followed by `37`, and `39` becomes `37`. The numbers inside
/// a `38;5;n` or `38;2;r;g;b` are colours, not resets, and are left alone.
fn whiten(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    let mut rest = text;
    while let Some(at) = rest.find("\x1b[") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 2..];
        let Some(end) = after.find(|c: char| ('\x40'..='\x7e').contains(&c)) else {
            out.push_str(&rest[at..]);
            return out;
        };
        let (parameters, last) = (&after[..end], &after[end..end + 1]);
        if last != "m" {
            out.push_str(&rest[at..at + 2 + end + 1]);
        } else {
            let parts: Vec<&str> = if parameters.is_empty() { vec![""] } else { parameters.split(';').collect() };
            let mut kept: Vec<String> = Vec::new();
            let mut index = 0;
            while index < parts.len() {
                let part = parts[index];
                match part {
                    "" | "0" => kept.extend(["0".to_string(), "37".to_string()]),
                    "39" => kept.push("37".into()),
                    "38" | "48" => {
                        let take = match parts.get(index + 1) {
                            Some(&"5") => 2,
                            Some(&"2") => 4,
                            _ => 0,
                        };
                        kept.extend(parts[index..(index + 1 + take).min(parts.len())].iter().map(|p| p.to_string()));
                        index += take;
                    }
                    other => kept.push(other.to_string()),
                }
                index += 1;
            }
            out.push_str(&format!("\x1b[{}m", kept.join(";")));
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    out
}

/// `text` without its escape codes.
fn plain_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if ('\x40'..='\x7e').contains(&next) {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
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

    /// The rows as they read, their colour codes left out.
    fn plain(lines: &[Line]) -> String {
        let text = text_of(lines);
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

    fn text_of(lines: &[Line]) -> String {
        lines.iter().map(|line| line.iter().map(|(_, text)| text.as_str()).collect::<String>()).collect::<Vec<_>>().join("\n")
    }

    /// A display whose record holds `lines` harness lines, `line 1`….
    fn display_with(lines: std::ops::RangeInclusive<usize>, display: &ActiveDisplay) {
        let entries = lines.map(|i| (0, 0, crate::transcript::Entry::Line(format!("line {i}")))).collect();
        display
            .load(crate::terminal::DisplaySnapshot {
                entries,
                dropped: false,
                active: false,
                turn_ms: 0,
                activity: crate::terminal::Activity::Waiting,
                queued: 0,
            })
            .unwrap();
    }

    #[test]
    fn an_attached_terminal_asks_the_sync_what_it_wrote_since_its_mark() {
        let _notes = crate::shared_files::tests::NOTES.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let ask = |since| serde_json::from_str::<crate::share_sync::Report>(&perform(Action::ShareSync { since }).unwrap()).unwrap();
        let mark = ask(0).mark;
        assert!(ask(mark).wrote.is_empty());
        // As the attached terminal sends it, over the wire.
        let action: Action = serde_json::from_str(&serde_json::to_string(&Action::ShareSync { since: mark }).unwrap()).unwrap();
        assert!(matches!(action, Action::ShareSync { since } if since == mark));
    }

    #[test]
    fn the_conversation_further_back() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        let display = ActiveDisplay::new();
        display_with(1..=30, &display);
        let mut tab = Open::new(View::History);
        let rows = history_rows(&mut tab, &display, 40).unwrap();
        assert!(rows >= 30, "{rows}");
        tab.back = 5;
        let (frame, _) = compose(&mut tab, &display, (40, 12), Instant::now());
        assert_eq!(frame.len(), 12);
        let text = text_of(&frame);
        assert!(frame[0].iter().any(|(style, text)| gui_view::look(*style) == Style::Tab && text.contains("s0ma")), "s0ma's key is the one up: {text}");
        // Under the bar (three rows at most) and its rule.
        let top = frame.iter().position(|line| text_of(std::slice::from_ref(line)).chars().all(|c| c == '─')).unwrap() + 1;
        assert!(top <= 4, "{text}");
        assert!(frame[top..11].iter().all(|line| line.iter().all(|(style, _)| *style == Style::Raw)), "the rows as the page draws them");
        assert!(text.contains("line 25") && !text.contains("line 30"), "five rows back from the end: {text}");
        assert!(text_of(&frame[11..]).contains("↑ 5 rows"));

        // More arrives while scrolled back: the place is kept.
        display_with(31..=33, &display);
        let grown = history_rows(&mut tab, &display, 40).unwrap();
        assert_eq!(tab.back, 5 + grown - rows);
        let (frame, _) = compose(&mut tab, &display, (40, 12), Instant::now());
        assert!(text_of(&frame).contains("line 25"), "{}", text_of(&frame));

        let press = |tab: &mut Open, code| history_key(tab, &KeyEvent::new(code, KeyModifiers::NONE), 4);
        let back = tab.back;
        assert_eq!(press(&mut tab, KeyCode::Down), Outcome::Stay);
        assert_eq!(tab.back, back - 1);
        assert_eq!(press(&mut tab, KeyCode::PageUp), Outcome::Stay);
        assert_eq!(tab.back, back + 3);
        assert_eq!(press(&mut tab, KeyCode::End), Outcome::Closed);
        assert_eq!(press(&mut tab, KeyCode::Esc), Outcome::Closed);
        assert_eq!(press(&mut tab, KeyCode::Char('q')), Outcome::Closed);
        let ctrl = |tab: &mut Open, c| history_key(tab, &KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL), 4);
        assert_eq!(ctrl(&mut tab, 'c'), Outcome::Interrupt);
        assert!(!tab.expanded);
        assert_eq!(ctrl(&mut tab, 'o'), Outcome::Stay);
        assert!(tab.expanded, "ctrl+o shows it the other way");
    }

    #[test]
    fn the_place_read_is_kept_as_rows_come_and_go() {
        let rows = |range: std::ops::Range<usize>| -> Vec<Line> { range.map(|i| vec![(Style::Raw, format!("row {i}"))]).collect() };
        // Three added at the end: three further back.
        assert_eq!(reanchor(&rows(0..20), &rows(0..23), 5), 8);
        // Ten dropped from the front as well: the same rows, still 8 back.
        assert_eq!(reanchor(&rows(0..20), &rows(10..23), 5), 8);
        // Nothing to find (it all changed): by how many were added.
        assert_eq!(reanchor(&rows(0..20), &rows(100..130), 5), 15);
    }

    #[test]
    fn rows_with_their_own_colours_are_painted_as_they_are() {
        let mut out = Vec::new();
        let frame = vec![vec![(Style::Raw, "\x1b[1mbold\x1b[0m plain".to_string())]];
        paint_to(&mut out, &[], &frame, Some((0, 0)), 20).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("\x1b[1mbold\x1b[0m plain"), "{text:?}");
        // Ten columns of twenty: the rest of the row is cleared.
        assert!(text.contains("\x1b[K"), "{text:?}");
        assert_eq!(crate::page::width("\x1b[1mbold\x1b[0m plain"), 10);
    }

    #[test]
    fn alt_keys_lead_to_the_chats_and_alt_shift_keys_to_the_terminals() {
        assert_eq!(target(Group::Chat, '1'), Some(Target::Main));
        assert_eq!(target(Group::Chat, '2'), Some(Target::View(View::Subagent(2))));
        assert_eq!(target(Group::Chat, '9'), Some(Target::View(View::Subagent(9))));
        assert_eq!(target(Group::Chat, '0'), Some(Target::View(View::Subagent(10))));
        assert_eq!(target(Group::Terminal, '1'), Some(Target::View(View::Shell { number: 1, session: None })), "the first shell: no commands tab");
        assert_eq!(target(Group::Terminal, '2'), Some(Target::View(View::Shell { number: 2, session: None })), "no shells listed here");
        assert_eq!(target(Group::Terminal, '0'), Some(Target::View(View::Shell { number: 10, session: None })));
        assert_eq!(target(Group::Terminal, 'b'), Some(Target::Back));
        assert_eq!(target(Group::Chat, 'B'), Some(Target::Back));
        assert_eq!(target(Group::Chat, 'x'), None);

        let key = |code, modifiers| KeyEvent::new(code, modifiers);
        let alt_shift = KeyModifiers::ALT | KeyModifiers::SHIFT;
        assert_eq!(switch_key(&key(KeyCode::Char('3'), KeyModifiers::ALT)), Some((Group::Chat, '3')));
        assert_eq!(switch_key(&key(KeyCode::Char('b'), KeyModifiers::ALT)), Some((Group::Chat, 'b')));
        assert_eq!(switch_key(&key(KeyCode::Char('e'), KeyModifiers::ALT)), Some((Group::Chat, 'e')));
        assert_eq!(target(Group::Chat, 'e'), Some(Target::View(View::Edit)));
        assert_eq!(switch_key(&key(KeyCode::Char('B'), alt_shift)), Some((Group::Chat, 'b')));
        // Alt+Shift+3: Alt+# from most terminals, Alt+Shift+3 with the kitty protocol.
        assert_eq!(switch_key(&key(KeyCode::Char('#'), KeyModifiers::ALT)), Some((Group::Terminal, '3')));
        assert_eq!(switch_key(&key(KeyCode::Char('3'), alt_shift)), Some((Group::Terminal, '3')));
        assert_eq!(switch_key(&key(KeyCode::Char(')'), alt_shift)), Some((Group::Terminal, '0')));
        assert_eq!(switch_key(&key(KeyCode::Char('!'), KeyModifiers::ALT)), Some((Group::Terminal, '1')));
        assert_eq!(switch_key(&key(KeyCode::Char('3'), KeyModifiers::CONTROL)), None, "Ctrl+digits are keys of their own again");
        assert_eq!(switch_key(&key(KeyCode::Char('4'), KeyModifiers::CONTROL)), None, "Ctrl+\\ as legacy terminals send it");
        assert_eq!(switch_key(&key(KeyCode::Char('3'), KeyModifiers::CONTROL | KeyModifiers::ALT)), None, "AltGr is text");
        assert_eq!(switch_key(&key(KeyCode::Char('#'), KeyModifiers::NONE)), None, "a plain # is typed");
        assert_eq!(switch_key(&key(KeyCode::Char('x'), KeyModifiers::ALT)), None);
        assert_eq!(switch_key(&key(KeyCode::Char('3'), KeyModifiers::NONE)), None);
    }

    #[test]
    fn a_feed_reads_as_what_the_subagent_did() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        worker_feed::begin(2, "subagent-1-2-17-2", "find the slow query\nin the access logs");
        worker_feed::stream("subagent-1-2-17-2", true, "I should look at the logs first.\nThen grep.");
        worker_feed::stream("subagent-1-2-17-2", false, "Looking at the logs.");
        worker_feed::push("subagent-1-2-17-2", Item::Call { name: "vm_exec".into(), arguments: "{\"command\": \"tail -n 3 access.log\"}".into() });
        worker_feed::push("subagent-1-2-17-2", Item::Result("{\"ok\":true,\"exit_code\":0,\"stdout\":\"GET /a 12ms\\nGET /b 9000ms\\nGET /c 8ms\\nline4\\nline5\\nline6\\nline7\\nline8\"}".into()));
        worker_feed::push("subagent-1-2-17-2", Item::Direction { seq: 4, from: Sender::Operator, text: "only /b matters".into(), wrap_up: false, delivered: false });
        let now = Instant::now();
        let (lines, footer) = worker_feed::with_feed(2, |feed| feed_lines(feed, false, 80, now, None)).unwrap();
        let text = plain(&lines);
        // Drawn as s0ma's own page is: the task is s0ma's first word to it.
        assert!(text.contains("s0ma> find the slow query\n      in the access logs"), "{text}");
        assert!(text.contains("[*] thought for") && text.contains("tokens"), "{text}");
        assert!(!text.contains("Then grep."), "collapsed, reasoning is counted, not shown");
        assert!(text.contains("b17-2> Looking at the logs."), "its own words under its name: {text}");
        assert!(text.contains("[+]") && text.contains("ran vm_exec"), "its work as one line: {text}");
        assert!(!text.contains("exit_code") && !text.contains("GET /b"), "the rest is for Ctrl+O: {text}");
        assert!(text.contains("you>  only /b matters  queued"), "{text}");
        let status = text_of(std::slice::from_ref(&footer.left));
        assert!(status.contains("b17-2 · running") && status.contains("1 queued"), "{status}");
        assert!(footer.hints.contains(&("⏎", "steer")) && footer.hints.contains(&("^o", "more")));

        let (expanded, _) = worker_feed::with_feed(2, |feed| feed_lines(feed, true, 80, now, Some("paused: context"))).unwrap();
        let expanded = plain(&expanded);
        assert!(expanded.contains("[Thinking]") && expanded.contains("Then grep."), "Ctrl+O shows its reasoning: {expanded}");
        assert!(expanded.contains("[tool] running: vm_exec") && expanded.contains("[tool] finished: vm_exec"), "{expanded}");
        assert!(expanded.contains("GET /b 9000ms") && expanded.contains("line8"), "{expanded}");

        worker_feed::delivered("subagent-1-2-17-2", &[4]);
        worker_feed::finish("subagent-1-2-17-2", "completed");
        let (done, footer) = worker_feed::with_feed(2, |feed| feed_lines(feed, false, 80, now, None)).unwrap();
        assert!(!plain(&done).contains("queued"));
        assert!(plain(&done).contains("completed after"));
        assert!(footer.hints.contains(&("^q", "close")), "a finished one can be closed");
        assert!(text_of(&[footer.left]).contains("completed after"));
        assert!(!footer.hints.contains(&("⏎", "steer")), "nothing to steer once it has finished");
    }

    #[test]
    fn ctrl_q_closes_a_finished_subagents_tab() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        worker_feed::begin(0, "r-run", "running task");
        worker_feed::begin(1, "r-done", "finished task");
        worker_feed::finish("r-done", "completed");
        let display = ActiveDisplay::new();
        let press = |tab: &mut Open, c| key_event(tab, &KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL), &display).unwrap();

        let mut running = Open::new(View::Subagent(2));
        assert_eq!(press(&mut running, 'q'), Outcome::Stay);
        assert!(running.note.as_ref().is_some_and(|(text, problem, _)| *problem && text.contains("/kill")), "a running one stays");
        assert_eq!(press(&mut running, 'c'), Outcome::Interrupt, "ctrl+c: as on s0ma's page");

        let mut done = Open::new(View::Subagent(3));
        assert_eq!(press(&mut done, 'c'), Outcome::Interrupt, "ctrl+c closes nothing");
        assert_eq!(worker_feed::tabs().len(), 2);
        assert_eq!(press(&mut done, 'q'), Outcome::Stay, "the first asks");
        assert!(done.note.as_ref().is_some_and(|(text, _, _)| text.starts_with("[?] closes")), "{:?}", done.note);
        assert_eq!(worker_feed::tabs().len(), 2);
        assert_eq!(press(&mut done, 'q'), Outcome::Closed);
        let left: Vec<String> = worker_feed::tabs().into_iter().map(|tab| tab.run_id).collect();
        assert_eq!(left, vec!["r-run".to_string()], "the finished one is gone, the running one kept");
        assert!(!worker_feed::forget("r-run"), "one still running is never forgotten");
    }

    #[test]
    fn a_result_in_short_says_what_went_wrong() {
        let (lines, problem) = result_gist("{\"ok\":true,\"exit_code\":2,\"stdout\":\"\",\"stderr\":\"no such file\\n\",\"stdout_truncated\":false}");
        assert_eq!(problem.as_deref(), Some("exit 2"));
        assert_eq!(lines, vec!["no such file".to_string()], "stderr when it failed");
        let (lines, problem) = result_gist("{\"ok\":false,\"error\":\"session gone\"}");
        assert_eq!(problem.as_deref(), Some("session gone"));
        assert!(lines.is_empty());
        let (lines, problem) = result_gist("{\"ok\":true,\"session\":\"w1\",\"written_bytes\":21}");
        assert!(lines.is_empty() && problem.is_none(), "bookkeeping alone says nothing");
    }

    #[test]
    fn prose_wraps_at_words_under_its_indent() {
        let lines = flow(&[(Style::Plain, "s0ma ›  the quick brown fox jumps over the lazy dog".to_string())], 24, 8);
        let text = text_of(&lines);
        assert_eq!(text, "s0ma ›  the quick brown\n        fox jumps over\n        the lazy dog", "{text}");
        let long = text_of(&flow(&[(Style::Plain, "x".repeat(30))], 20, 2));
        assert_eq!(long, format!("{}\n  {}", "x".repeat(20), "x".repeat(10)), "a word longer than a row is broken");
        let footer = text_of(&[gui_view::footer(vec![(Style::Status, "2 commands".into())], &[("^o", "unfold"), ("esc", "back")], 40).0]);
        assert_eq!(footer, " 2 commands              Unfold   Back ", "buttons at the right edge, a column in, no keys named");
        assert_eq!(text_width(&footer), 39, "a column spare at the right");
        let narrow = text_of(&[gui_view::footer(vec![(Style::Status, "2 commands".into())], &[("^o", "unfold"), ("esc", "back")], 26).0]);
        assert!(narrow.starts_with(" 2 commands") && narrow.contains(" Unfold ") && text_width(&narrow) <= 26, "the last buttons go first when they don't fit (the menu has them): {narrow:?}");
    }

    #[test]
    fn alt_b_goes_back_to_the_tab_as_it_was_left() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut tab = Open::new(View::Subagent(2));
        tab.draft.insert_str("half");
        assert_eq!(go_to(&mut tab, Target::Back), Outcome::Closed, "nothing before it: back to s0ma's page");

        assert_eq!(go_to(&mut tab, Target::View(View::Board)), Outcome::Stay);
        assert_eq!(go_to(&mut tab, Target::View(View::Review)), Outcome::Stay);
        assert_eq!(go_to(&mut tab, Target::View(View::Deliverables)), Outcome::Stay);
        assert_eq!(go_to(&mut tab, Target::Back), Outcome::Stay);
        assert_eq!(tab.view, View::Subagent(2), "the projects tab's subtabs are one step");
        assert_eq!(tab.draft.text(), "half");
        assert_eq!(go_to(&mut tab, Target::Back), Outcome::Stay);
        assert_eq!(tab.view, View::Deliverables, "the subtab it was on");
        assert_eq!(go_to(&mut tab, Target::Back), Outcome::Stay);
        assert_eq!(tab.view, View::Subagent(2));

        // Closed to the page, the tab is kept for Alt+B there; the page
        // itself is not.
        stash(tab);
        let kept = closed().take().expect("kept");
        assert_eq!(kept.view, View::Subagent(2));
        assert!(kept.previous.is_none());
        let page = Open { previous: Some(Box::new(kept)), ..Open::new(View::Page) };
        stash(page);
        assert_eq!(closed().take().map(|tab| tab.view), Some(View::Subagent(2)), "a page pane gives back what it covered");
        stash(Open::new(View::Page));
        assert!(closed().is_none());
    }

    #[test]
    fn the_last_thought_shows_while_it_is_being_thought() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        worker_feed::begin(1, "r-1", "task");
        worker_feed::stream("r-1", true, "first idea\nsecond idea");
        let (lines, _) = worker_feed::with_feed(1, |feed| feed_lines(feed, false, 80, Instant::now(), None)).unwrap();
        assert!(plain(&lines).contains("thinking"), "{}", plain(&lines));
    }

    #[test]
    fn keys_in_a_tab() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        let display = ActiveDisplay::new();
        let press = |tab: &mut Open, code, modifiers| key_event(tab, &KeyEvent::new(code, modifiers), &display).unwrap();
        let mut tab = Open::new(View::Subagent(2));
        tab.draft.insert_str("half");
        assert_eq!(press(&mut tab, KeyCode::Char('3'), KeyModifiers::ALT), Outcome::Stay);
        assert_eq!(tab.view, View::Subagent(3));
        assert!(tab.draft.is_empty(), "a draft meant for one subagent does not follow to another");
        assert_eq!(press(&mut tab, KeyCode::Char('b'), KeyModifiers::ALT), Outcome::Stay);
        assert_eq!(tab.view, View::Subagent(2), "Alt+B goes back");
        assert_eq!(tab.draft.text(), "half", "as it was left");
        assert_eq!(press(&mut tab, KeyCode::Char('b'), KeyModifiers::ALT), Outcome::Stay);
        assert_eq!(tab.view, View::Subagent(3), "and again, forward");
        assert!(tab.draft.is_empty());
        assert_eq!(press(&mut tab, KeyCode::Char('!'), KeyModifiers::ALT), Outcome::Stay);
        assert_eq!(tab.view, View::Shell { number: 1, session: None }, "Alt+Shift+1 is the first shell");
        assert_eq!(press(&mut tab, KeyCode::Char('2'), KeyModifiers::ALT), Outcome::Stay);
        assert_eq!(tab.view, View::Subagent(2));
        assert_eq!(press(&mut tab, KeyCode::Char('t'), KeyModifiers::CONTROL), Outcome::Stay);
        assert_eq!(press(&mut tab, KeyCode::Char('4'), KeyModifiers::NONE), Outcome::Stay);
        assert_eq!(tab.draft.text(), "4", "Ctrl+T is nothing special now");
        tab.draft = Draft::default();

        tab.draft.insert_str("x");
        assert_eq!(press(&mut tab, KeyCode::Char('c'), KeyModifiers::CONTROL), Outcome::Stay);
        assert!(tab.draft.is_empty(), "ctrl+c clears what is typed first");
        assert_eq!(press(&mut tab, KeyCode::Char('c'), KeyModifiers::CONTROL), Outcome::Interrupt);
        assert_eq!(press(&mut tab, KeyCode::Char('1'), KeyModifiers::ALT), Outcome::Closed);
        assert_eq!(press(&mut tab, KeyCode::Esc, KeyModifiers::NONE), Outcome::Closed);
        assert_eq!(press(&mut tab, KeyCode::Char('d'), KeyModifiers::CONTROL), Outcome::Eof);

        // Enter with no one to send it to keeps the draft and says why.
        let mut tab = Open::new(View::Subagent(2));
        tab.draft.insert_str("only /b");
        assert_eq!(press(&mut tab, KeyCode::Enter, KeyModifiers::NONE), Outcome::Stay);
        assert_eq!(tab.draft.text(), "only /b");
        assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| *problem && text.contains("no subagent")));
        worker_feed::begin(1, "subagent-1-2-3-1", "task");
        worker_feed::finish("subagent-1-2-3-1", "completed");
        assert_eq!(press(&mut tab, KeyCode::Enter, KeyModifiers::NONE), Outcome::Stay);
        assert_eq!(tab.draft.text(), "only /b");
        assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| *problem && text.contains("has finished")));
        // Typed for one worker; another took its place on the tab (its slot
        // was used again): not sent, and the tab says so.
        tab.pinned = Some("subagent-1-2-3-1".into());
        worker_feed::begin(1, "subagent-1-2-4-1", "another task");
        assert_eq!(press(&mut tab, KeyCode::Enter, KeyModifiers::NONE), Outcome::Stay);
        assert_eq!(tab.draft.text(), "only /b");
        assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| *problem && text.contains("not sent")
            && text.contains("b3-1") && text.contains("b4-1")), "{:?}", tab.note);
        assert_eq!(tab.pinned.as_deref(), Some("subagent-1-2-4-1"), "Enter again means the one it shows now");
        // Alt+Enter is a new line, not a send.
        assert_eq!(press(&mut tab, KeyCode::Enter, KeyModifiers::ALT), Outcome::Stay);
        assert_eq!(tab.draft.text(), "only /b\n");
        // Scrolling back stops at the top, End follows the end again.
        assert_eq!(press(&mut tab, KeyCode::PageUp, KeyModifiers::NONE), Outcome::Stay);
        assert!(tab.back > 0);
        tab.draft = Draft::default();
        assert_eq!(press(&mut tab, KeyCode::End, KeyModifiers::NONE), Outcome::Stay);
        assert_eq!(tab.back, 0);
    }

    #[test]
    fn tabs_follow_the_slots_and_a_finished_worker_stays_readable() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        worker_feed::begin(3, "subagent-1-2-5-3", "third slot");
        worker_feed::begin(1, "subagent-1-2-5-1", "first slot");
        worker_feed::finish("subagent-1-2-5-1", "cancelled");
        assert_eq!(subagent_at(2).unwrap().run_id, "subagent-1-2-5-1", "tab 2 is the lowest slot");
        assert_eq!(subagent_at(3).unwrap().run_id, "subagent-1-2-5-3");
        assert!(subagent_at(4).is_none() && subagent_at(1).is_none());
        assert_eq!(tab_of("subagent-1-2-5-3"), Some(3));
        let rows = tab_bar(Some(&View::Subagent(3)), None, 120);
        let bar = text_of(&rows);
        assert!(bar.starts_with("  s0ma "), "the keys are not named: {bar}");
        assert!(bar.contains("  first slot [-] ") && bar.contains("  third slot ") && bar.contains("  projects "), "{bar}");
        // The halves: the pipe at the middle on every row, the shells' after it.
        for row in bar.lines() {
            assert_eq!(row.chars().position(|c| c == '│'), Some(59), "pinned at the middle: {row:?}");
        }
        assert!(bar.lines().next().unwrap().trim_end().ends_with('│'), "no shells listed here: {bar}");
        let flat = |rows: Vec<Line>| rows.into_iter().flatten().collect::<Vec<_>>();
        let up = flat(tab_bar(Some(&View::Subagent(3)), None, 120));
        assert!(up.iter().any(|(style, text)| gui_view::look(*style) == Style::Tab && text == " third slot "), "the one that is up stands out: {up:?}");
        // A key with nothing on it: in the bar while it is the one up, not otherwise.
        let empty = flat(tab_bar(Some(&View::Subagent(7)), None, 120));
        assert!(empty.iter().any(|(style, text)| gui_view::look(*style) == Style::Tab && text == " empty "), "{empty:?}");
        assert!(!text_of(&tab_bar(Some(&View::Board), None, 120)).contains("empty"));
        let shell = flat(tab_bar(Some(&View::Shell { number: 4, session: None }), None, 120));
        assert!(shell.iter().any(|(style, text)| gui_view::look(*style) == Style::Tab && text == " empty "), "{shell:?}");
        // s0ma's own page: its tab is the one up.
        let main = tab_bar(None, None, 120);
        assert!(flat(main.clone()).iter().any(|(style, text)| gui_view::look(*style) == Style::Tab && text == " s0ma "));
        let (written, cells) = main_bar(120);
        assert_eq!(cells, main.iter().map(|row| text_width(&text_of(std::slice::from_ref(row)))).max().unwrap());
        assert!(written.contains(" s0ma ") && written.contains('\x1b'), "with its colours");
        // A click on the page's bar is the tab's key: Alt+P on projects.
        let row = text_of(&main);
        let column = row.lines().next().unwrap().find("projects").unwrap();
        assert_eq!(page_click(column as u16, 0), Some(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::ALT)));
        assert_eq!(page_click(0, 0), None, "the space before the first tab");
        // Narrow: a half runs onto more rows, each in its half.
        let narrow = tab_bar(Some(&View::Subagent(3)), None, 80);
        assert!(narrow.len() >= 2 && narrow.len() <= 3, "{}", text_of(&narrow));
        for row in text_of(&narrow).lines() {
            assert!(text_width(row) <= 80, "{row:?}");
            assert_eq!(row.chars().position(|c| c == '│'), Some(39), "{row:?}");
        }
        assert!(text_of(&narrow).contains("overview"), "everything still seen: {}", text_of(&narrow));
    }

    #[test]
    fn two_panes_keys_widths_and_rows_cut_to_fit() {
        let key = |code, modifiers| KeyEvent::new(code, modifiers);
        let ctrl_shift = KeyModifiers::CONTROL | KeyModifiers::SHIFT;
        assert!(is_split_key(&key(KeyCode::Char('P'), ctrl_shift)) && is_split_key(&key(KeyCode::Char('p'), ctrl_shift)));
        assert!(is_split_key(&key(KeyCode::Char('\\'), KeyModifiers::ALT)) && is_split_key(&key(KeyCode::Char('|'), KeyModifiers::ALT | KeyModifiers::SHIFT)));
        assert!(!is_split_key(&key(KeyCode::Char('\\'), KeyModifiers::CONTROL)), "Ctrl+\\ is a shell's control key");
        assert_eq!(focus_key(&key(KeyCode::Left, KeyModifiers::ALT)), Some(0));
        assert_eq!(focus_key(&key(KeyCode::Right, KeyModifiers::ALT)), Some(1));
        assert_eq!(focus_key(&key(KeyCode::Right, KeyModifiers::CONTROL)), None, "Ctrl+→ still moves by a word");
        assert!(!is_split_key(&key(KeyCode::Char('p'), KeyModifiers::CONTROL)), "Ctrl+P stays what it was");
        assert_eq!(focus_key(&key(KeyCode::Char('1'), KeyModifiers::CONTROL)), Some(0));
        assert_eq!(focus_key(&key(KeyCode::Char('2'), KeyModifiers::CONTROL)), Some(1));
        assert_eq!(focus_key(&key(KeyCode::Char('2'), KeyModifiers::ALT)), None);
        assert_eq!(pane_widths(150), (74, 75));
        assert_eq!(pane_widths(81), (40, 40));
        // A row cut to its pane and padded to it; a raw row cut at its codes.
        let plain: Line = vec![(Style::Plain, "abcdef".into()), (Style::Dim, "ghij".into())];
        assert_eq!(text_of(&[fit_line(&plain, 8)]), "abcdefgh");
        assert_eq!(text_of(&[fit_line(&plain, 12)]), "abcdefghij  ");
        let raw: Line = vec![(Style::Raw, "\x1b[32mhello\x1b[0m world".into())];
        let cut = fit_line(&raw, 7);
        assert_eq!(plain_text(&text_of(std::slice::from_ref(&cut))), "hello w");
        assert!(text_of(std::slice::from_ref(&cut)).ends_with("\x1b[0m"), "{cut:?}");
    }

    #[test]
    fn a_pane_put_aside_comes_back_on_its_side_as_it_was() {
        parked().take();
        // The first time: what is up on the left, s0ma's page on the right
        // with the keyboard.
        let (focused, state) = two_panes(Open::new(View::Memory));
        assert_eq!((focused.view.clone(), state.other.view.clone(), state.focus), (View::Page, View::Memory, 1));
        // The keyboard on the right (pane 2, the share tab), the left (pane 1)
        // scrolled; one pane: the left one is put aside as it is.
        let mut left = state.other;
        left.back = 7;
        *parked() = Some((left, 0));
        // Two again, from the share tab: pane 1 back on the left as it was,
        // the share tab on the right with the keyboard.
        let (focused, state) = two_panes(Open::new(View::Share));
        assert_eq!((focused.view.clone(), state.other.view.clone(), state.focus), (View::Share, View::Memory, 1));
        assert_eq!(state.other.back, 7, "as it was");
        assert!(parked().is_none(), "taken back");
        // Put aside from the right: it comes back on the right.
        *parked() = Some((Open::new(View::Overview), 1));
        let (focused, state) = two_panes(Open::new(View::Page));
        assert_eq!((focused.view.clone(), state.other.view.clone(), state.focus), (View::Page, View::Overview, 0));
    }

    #[test]
    fn the_page_in_a_pane_sends_to_s0ma() {
        let display = ActiveDisplay::new();
        let mut tab = Open::new(View::Page);
        let press = |tab: &mut Open, code, modifiers| key_event(tab, &KeyEvent::new(code, modifiers), &display).unwrap();
        assert_eq!(press(&mut tab, KeyCode::Enter, KeyModifiers::NONE), Outcome::Stay, "nothing typed, nothing sent");
        for c in "look at the logs".chars() {
            press(&mut tab, KeyCode::Char(c), KeyModifiers::NONE);
        }
        assert_eq!(press(&mut tab, KeyCode::Enter, KeyModifiers::NONE), Outcome::Submit { text: "look at the logs".into(), now: false });
        assert!(tab.draft.is_empty());
        press(&mut tab, KeyCode::Char('x'), KeyModifiers::NONE);
        assert_eq!(press(&mut tab, KeyCode::Char('s'), KeyModifiers::CONTROL), Outcome::Submit { text: "x".into(), now: true }, "Ctrl+S: now");
        assert_eq!(press(&mut tab, KeyCode::Esc, KeyModifiers::NONE), Outcome::Stay, "Esc leaves the pane as it is");
        let (line, _) = prompt_line(&tab, 60);
        assert!(text_of(&[line]).starts_with(" you ›"));
    }

    #[test]
    fn run_ids_shorten_to_their_batch_and_slot() {
        assert_eq!(short_id("subagent-1762100000000-4242-17-3"), "b17-3");
        assert_eq!(short_id("custom"), "custom");
    }

    fn board_tab() -> Open {
        let mut tab = Open::new(View::Board);
        tab.board_read = Some(BoardRead {
            at: Instant::now(),
            snapshot: Ok(Some(board_view::tests::snapshot())),
            name: Some("flaky-upload".into()),
            open: 1,
            waiting: 0,
        });
        tab
    }

    #[test]
    fn the_board_subtab_has_open_boards_and_write_ups_due_and_later_stages_go_on() {
        use crate::board::{BoardStatus, GoalTestKind};
        use crate::board_flow::{Flow, FlowStore, Stage};
        let root = std::env::temp_dir().join(format!("s0ma-tab-boards-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let store = crate::board::BoardStore::beside_archive(&root.join("hippocampus.jsonl"));
        let flows = FlowStore::beside(&store);
        let make = |name: &str| store.open_board(name, "t", "o", "g", "The operator agrees", GoalTestKind::Judgment, 20).unwrap().unwrap();
        let in_stage = |board: &crate::board::Board, stage: Stage| {
            flows.put(&Flow { board_id: board.id.clone(), name: board.name.clone(), stage: Some(stage), round: 1, ..Flow::default() }).unwrap();
        };
        let _open = make("open-one");
        let due = make("write-up-due");
        let legacy = make("old-proposal");
        let reviewed = make("in-review");
        let delivered = make("delivered");
        let gone = make("abandoned-one");
        let old_solved = make("solved-long-ago");
        for (board, status) in [(&due, BoardStatus::ProposedSolved), (&legacy, BoardStatus::ProposedSolved), (&reviewed, BoardStatus::ProposedSolved),
            (&delivered, BoardStatus::Solved), (&gone, BoardStatus::Abandoned), (&old_solved, BoardStatus::Solved)] {
            store.set_status(&board.id, status, "done").unwrap().unwrap();
        }
        in_stage(&due, Stage::Working);
        in_stage(&reviewed, Stage::Review);
        in_stage(&delivered, Stage::Deliverable);
        let names = |boards: Vec<crate::board::Board>| boards.into_iter().map(|board| board.name).collect::<Vec<_>>();
        assert_eq!(names(shown_boards(&store).unwrap()), ["old-proposal", "write-up-due", "open-one"]);
        let read = read_board(&store, 1).unwrap();
        assert_eq!((read.name.as_deref(), read.open, read.waiting), (Some("write-up-due"), 3, 1), "one in review waits for the operator");
        let (in_review, counts) = projects_view::read(&store, Stage::Review, None).unwrap();
        assert_eq!((in_review.len(), counts), (1, [3, 1, 0, 1]));

        // A finished board says its write-up is due, and where it goes.
        let mut tab = Open::new(View::Board);
        tab.board_read = Some(read);
        let (lines, footer) = board_body(&mut tab, 100, 30, Instant::now());
        let shown = text_of(&lines);
        assert!(shown.contains("[*] proposed solved") && shown.contains("[*] finished: done"), "{shown}");
        assert!(shown.contains("s0ma writes its report (board_report), then it comes to your review"), "{shown}");
        assert!(text_of(&[footer.left]).contains("[!] 1 in review"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn tab_steps_through_the_subtabs_and_ctrl_n_p_through_the_boards() {
        let display = ActiveDisplay::new();
        let press = |tab: &mut Open, code, modifiers| key_event(tab, &KeyEvent::new(code, modifiers), &display).unwrap();
        let mut tab = board_tab();
        for want in [View::Review, View::Verification, View::Deliverables, View::Board] {
            press(&mut tab, KeyCode::Tab, KeyModifiers::NONE);
            assert_eq!(tab.view, want);
        }
        press(&mut tab, KeyCode::BackTab, KeyModifiers::SHIFT);
        assert_eq!(tab.view, View::Deliverables, "Shift+Tab goes back");
        // On the board, Ctrl+N and Ctrl+P step through the boards, wrapping.
        let mut tab = board_tab();
        tab.board_read.as_mut().unwrap().open = 3;
        press(&mut tab, KeyCode::Char('n'), KeyModifiers::CONTROL);
        assert_eq!((tab.board, tab.board_read.is_none()), (1, true), "the next board is read");
        tab.board_read = board_tab().board_read.map(|mut read| { read.open = 3; read });
        press(&mut tab, KeyCode::Char('p'), KeyModifiers::CONTROL);
        press(&mut tab, KeyCode::Char('p'), KeyModifiers::CONTROL);
        assert_eq!(tab.board % 3, 2, "back past the first wraps to the last");
        // Typing a note in review, Tab is not a switch.
        let mut tab = Open::new(View::Review);
        tab.projects = Default::default();
        press(&mut tab, KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(tab.view, View::Verification);
    }

    #[test]
    fn alt_p_opens_the_projects_tab_and_leaves_its_subtab_alone() {
        let display = ActiveDisplay::new();
        let alt_p = KeyEvent::new(KeyCode::Char('p'), KeyModifiers::ALT);
        assert_eq!(switch_key(&alt_p), Some((Group::Chat, 'p')));
        let mut tab = Open::new(View::Share);
        key_event(&mut tab, &alt_p, &display).unwrap();
        assert!(projects_view::is_projects(&tab.view));
        for view in [View::Board, View::Review, View::Verification, View::Deliverables] {
            let mut tab = Open::new(view.clone());
            key_event(&mut tab, &alt_p, &display).unwrap();
            assert_eq!(tab.view, view, "alt+p again stays put");
        }
        // The tab bar names it once, whichever subtab is up.
        let bar = text_of(&tab_bar(Some(&View::Verification), None, 160));
        assert!(bar.contains(" projects ") && !bar.contains("board") && !bar.contains(" p "), "{bar}");
    }

    #[test]
    fn ctrl_c_greys_a_row_out_of_the_way_and_back() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        *GREYED.lock().unwrap_or_else(|p| p.into_inner()) = Some((None, HashMap::new()));
        let mut tab = board_tab();
        let display = ActiveDisplay::new();
        let press = |tab: &mut Open, code, modifiers| key_event(tab, &KeyEvent::new(code, modifiers), &display).unwrap();
        let at = |tab: &Open| tab.board_selected.clone().unwrap_or_default();
        let order = |tab: &mut Open| {
            let (tree, shown, _) = board_tree(tab).unwrap();
            shown.iter().map(|row| board_view::short(&tree.nodes[row.node].id)).collect::<Vec<_>>()
        };
        tab.board_selected = Some("intent-0008".into());
        let (_, footer) = board_layout(&mut tab, &BoardRead { at: Instant::now(), snapshot: Ok(Some(board_view::tests::snapshot())), name: None, open: 1, waiting: 0 }, 100, 30, 2, false);
        assert!(footer.hints.iter().any(|hint| *hint == ("^c", "grey")), "{:?}", footer.hints);

        assert_eq!(press(&mut tab, KeyCode::Char('c'), KeyModifiers::CONTROL), Outcome::Stay, "not an interrupt here");
        assert_eq!(order(&mut tab), ["board", "h1", "i7", "i4", "i8", "f3", "i9", "f4"]);
        assert_eq!(at(&tab), "intent-0007", "the chevron stays put, on the row that came up");
        press(&mut tab, KeyCode::End, KeyModifiers::NONE);
        press(&mut tab, KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(at(&tab), "intent-0009", "greyed rows walk as any");
        let (_, footer) = board_layout(&mut tab, &BoardRead { at: Instant::now(), snapshot: Ok(Some(board_view::tests::snapshot())), name: None, open: 1, waiting: 0 }, 100, 30, 2, false);
        assert!(footer.hints.iter().any(|hint| *hint == ("^c", "ungrey")));
        press(&mut tab, KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(order(&mut tab), ["board", "h1", "i8", "f3", "i9", "f4", "i7", "i4"], "back, and the branch with it");

        tab.draft.insert_str("x");
        press(&mut tab, KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(tab.draft.is_empty(), "with a hint typed, ctrl+c clears it first");
        assert_eq!(order(&mut tab), ["board", "h1", "i8", "f3", "i9", "f4", "i7", "i4"]);
        tab.board_selected = Some("board".into());
        assert_eq!(press(&mut tab, KeyCode::Char('c'), KeyModifiers::CONTROL), Outcome::Interrupt, "the board's row is never grey");
        *GREYED.lock().unwrap_or_else(|p| p.into_inner()) = Some((None, HashMap::new()));
    }

    #[test]
    fn the_open_subtab_has_the_boards_files_notes_and_reports() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        let mut tab = board_tab();
        let mut snapshot = board_view::tests::snapshot();
        let reports = vec![
            crate::board_flow::RoundReport { round: 1, text: "Round one: the fixture closes early.".into(), at: chrono::Local::now().to_rfc3339() },
        ];
        snapshot.flow = Some(crate::board_flow::Flow { board_id: snapshot.board.id.clone(), round: 2, reports, ..Default::default() });
        snapshot.notes = vec![crate::board_flow::Note {
            id: "n1".into(),
            board_id: snapshot.board.id.clone(),
            round: 1,
            text: "Is it the fixture or the client?".into(),
            created_at: chrono::Local::now().to_rfc3339(),
            answer: Some("The fixture.".into()),
            answered_at: None,
        }];
        tab.board_read.as_mut().unwrap().snapshot = Ok(Some(snapshot));
        let display = ActiveDisplay::new();
        let press = |tab: &mut Open, code, modifiers| key_event(tab, &KeyEvent::new(code, modifiers), &display).unwrap();
        let shown = |tab: &mut Open| {
            let item = open_item(tab);
            let (lines, footer) = projects_view::open_body(tab, item.as_ref(), 100, 30);
            (text_of(&lines), footer)
        };
        assert_eq!(projects_view::section(&tab), projects_view::Section::Board);
        let bar = projects_view::section_bar(&tab, &open_item(&tab).unwrap(), 100);
        assert!(text_of(&[bar]).contains(" board │ files │ notes 1 │ reports "), "the sections");

        // ⇧→: the files (none yet), then → the notes, the reports.
        press(&mut tab, KeyCode::Right, KeyModifiers::SHIFT);
        assert_eq!(projects_view::section(&tab), projects_view::Section::Files);
        let (text, footer) = shown(&mut tab);
        assert!(text.contains(" open ") && text.contains("No files yet"), "{text}");
        assert!(footer.hints.contains(&("←→", "section")));
        press(&mut tab, KeyCode::Right, KeyModifiers::NONE);
        let (text, _) = shown(&mut tab);
        assert!(text.contains("Is it the fixture or the client?") && text.contains("The fixture."), "{text}");
        press(&mut tab, KeyCode::Char('x'), KeyModifiers::NONE);
        assert!(tab.draft.is_empty(), "nothing is typed outside the board section");
        press(&mut tab, KeyCode::Right, KeyModifiers::NONE);
        let (text, _) = shown(&mut tab);
        assert!(text.contains("round 1 · latest") && text.contains("Round one: the fixture closes early."), "{text}");
        // Round to the board again: its keys are its own.
        press(&mut tab, KeyCode::Right, KeyModifiers::SHIFT);
        assert_eq!(projects_view::section(&tab), projects_view::Section::Board);
        press(&mut tab, KeyCode::Char('x'), KeyModifiers::NONE);
        assert_eq!(tab.draft.text(), "x", "a hint is typed here");
        press(&mut tab, KeyCode::Right, KeyModifiers::SHIFT);
        press(&mut tab, KeyCode::Right, KeyModifiers::NONE);
        assert_eq!(projects_view::section(&tab), projects_view::Section::Board, "not while a hint is being typed");
        tab.draft = Draft::default();
        press(&mut tab, KeyCode::Right, KeyModifiers::NONE);
        assert_eq!(projects_view::section(&tab), projects_view::Section::Files, "→ from the board, as in every section");
    }

    #[test]
    fn ctrl_t_steps_what_the_boards_prompt_adds() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        let mut tab = board_tab();
        let display = ActiveDisplay::new();
        let ctrl_t = |tab: &mut Open| key_event(tab, &KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL), &display).unwrap();
        let label = |tab: &Open| text_of(&[prompt_line(tab, 100).0]);
        tab.board_selected = Some("fact-0003".into());
        assert!(label(&tab).starts_with(" hint on f3 › hint for s0ma"), "{}", label(&tab));
        for want in [" lead from f3 › a lead of yours", " finding on f3, supports › what you found", " finding on f3, rules out ›", " hint on f3 ›"] {
            assert_eq!(ctrl_t(&mut tab), Outcome::Stay);
            assert!(label(&tab).starts_with(want), "{want}: {}", label(&tab));
        }
        ctrl_t(&mut tab);
        tab.board_selected = Some("intent-0008".into());
        assert!(label(&tab).starts_with(" lead ›"), "a lead hangs from a finding only: {}", label(&tab));
        ctrl_t(&mut tab);
        assert!(label(&tab).starts_with(" finding on i8, supports ›"));
    }

    #[test]
    fn walking_the_board_with_the_arrows() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        let mut tab = board_tab();
        let press = |tab: &mut Open, code| key_event(tab, &KeyEvent::new(code, KeyModifiers::NONE), &ActiveDisplay::new()).unwrap();
        let at = |tab: &Open| tab.board_selected.clone().unwrap_or_default();
        // Rows: board, h1, i8, f3, i9, f4, i7, i4 (folded).
        assert_eq!(press(&mut tab, KeyCode::Down), Outcome::Stay);
        assert_eq!(at(&tab), "fact-0003", "the first row was selected, then one down");
        press(&mut tab, KeyCode::Enter);
        assert_eq!(at(&tab), "fact-0003", "⏎ folds in place");
        press(&mut tab, KeyCode::Down);
        assert_eq!(at(&tab), "fact-0004", "its lead i9 is folded away");
        press(&mut tab, KeyCode::Up);
        press(&mut tab, KeyCode::Up);
        press(&mut tab, KeyCode::Enter);
        press(&mut tab, KeyCode::Down);
        assert_eq!(at(&tab), "intent-0007", "i8 folded: the next row is the next lead");
        press(&mut tab, KeyCode::Up);
        press(&mut tab, KeyCode::Enter);
        press(&mut tab, KeyCode::Down);
        press(&mut tab, KeyCode::Enter);
        assert_eq!(at(&tab), "fact-0003", "⏎ again unfolds");
        // ←→ step the sections, as in every subtab; the tree is kept.
        press(&mut tab, KeyCode::Right);
        assert_eq!(projects_view::section(&tab), projects_view::Section::Files);
        press(&mut tab, KeyCode::Left);
        assert_eq!(projects_view::section(&tab), projects_view::Section::Board);
        assert_eq!(at(&tab), "fact-0003");
        press(&mut tab, KeyCode::End);
        assert_eq!(at(&tab), "intent-0004");
        press(&mut tab, KeyCode::Home);
        assert_eq!(at(&tab), "board", "the board's own row is first");
        press(&mut tab, KeyCode::Down);
        assert_eq!(at(&tab), "hint-0001", "then the hints under it");
        press(&mut tab, KeyCode::Down);
        assert_eq!(at(&tab), "intent-0008");

        // A hint being typed: the arrows that edit are the draft's.
        press(&mut tab, KeyCode::Char('x'));
        press(&mut tab, KeyCode::Left);
        press(&mut tab, KeyCode::Right);
        assert_eq!((tab.draft.text(), at(&tab).as_str()), ("x", "intent-0008"));
        press(&mut tab, KeyCode::Down);
        assert_eq!(at(&tab), "fact-0003", "↑↓ still walk");
        let (line, _) = prompt_line(&tab, 80);
        assert!(text_of(&[line]).starts_with(" hint on f3 › x"));

        // Ctrl+O: everything open, then everything folded.
        let ctrl_o = |tab: &mut Open| key_event(tab, &KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL), &ActiveDisplay::new()).unwrap();
        tab.draft = Draft::default();
        ctrl_o(&mut tab);
        let (tree, shown, _) = board_tree(&mut tab).unwrap();
        assert_eq!(shown.len(), tree.nodes.len());
        tab.board_selected = Some("hint-0001".into());
        ctrl_o(&mut tab);
        let (_, shown, selected) = board_tree(&mut tab).unwrap();
        assert_eq!(shown.len(), 4, "the roots");
        assert_eq!(tab.board_selected.as_deref(), Some("board"), "a selection folded away moves to what holds it");
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn a_proposed_exhaustion_waits_for_its_write_up_and_ctrl_q_twice_closes_a_board() {
        let display = ActiveDisplay::new();
        let mut tab = board_tab();
        if let Some(BoardRead { snapshot: Ok(Some(snapshot)), .. }) = tab.board_read.as_mut() {
            snapshot.board.status = crate::board::BoardStatus::ProposedExhausted;
            snapshot.board.outcome = Some("Every direction was ruled out by a fact.".into());
        }
        let (lines, footer) = board_body(&mut tab, 110, 30, Instant::now());
        let shown = text_of(&lines);
        assert!(shown.contains("[*] proposed exhausted") && shown.contains("[*] s0ma says nothing is left to try: Every direction"), "{shown}");
        assert!(shown.contains("then it comes to your review"), "{shown}");
        assert!(footer.hints.contains(&("^q", "close")) && !footer.hints.iter().any(|(key, _)| *key == "^y"));
        // Ctrl+Q: asks; any other key keeps the board; twice closes it.
        let ctrl = |c| KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL);
        let _ = board_key(&mut tab, &ctrl('q'), &display, 10).unwrap();
        assert!(tab.note.as_ref().is_some_and(|(text, _, _)| text.starts_with("[?] closes flaky-upload")), "{:?}", tab.note);
        let _ = board_key(&mut tab, &KeyEvent::new(KeyCode::Up, KeyModifiers::NONE), &display, 10).unwrap();
        let _ = board_key(&mut tab, &ctrl('q'), &display, 10).unwrap();
        assert!(tab.note.as_ref().is_some_and(|(text, _, _)| text.starts_with("[?] closes")), "asked again: {:?}", tab.note);
        let _ = board_key(&mut tab, &ctrl('q'), &display, 10).unwrap();
        assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| text.contains("not available") && *problem), "asked of s0ma's process: {:?}", tab.note);
    }

    #[test]
    fn delete_twice_on_the_board_row_discards_the_board() {
        let display = ActiveDisplay::new();
        let mut tab = board_tab();
        tab.board_selected = Some("board".into());
        let delete = KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE);
        let _ = board_key(&mut tab, &delete, &display, 10).unwrap();
        assert!(tab.note.as_ref().is_some_and(|(text, _, _)| text.starts_with("[?] discards flaky-upload")), "{:?}", tab.note);
        let (_, footer) = board_body(&mut tab, 100, 30, Instant::now());
        assert!(footer.hints.contains(&("del", "discard")));
        let _ = board_key(&mut tab, &delete, &display, 10).unwrap();
        assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| text.contains("not available") && *problem), "asked of s0ma's process: {:?}", tab.note);
    }

    #[test]
    fn delete_twice_on_a_note_takes_it_off_the_board() {
        let display = ActiveDisplay::new();
        let mut tab = board_tab();
        tab.board_selected = Some("hint-0001".into());
        let delete = KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE);
        let _ = board_key(&mut tab, &delete, &display, 10).unwrap();
        assert_eq!(tab.board_unhint.as_deref(), Some("hint-0001"));
        assert!(tab.note.as_ref().is_some_and(|(text, _, _)| text.contains("takes h1 off the board · again to confirm")));
        // Any other key keeps it.
        let _ = board_key(&mut tab, &KeyEvent::new(KeyCode::Up, KeyModifiers::NONE), &display, 10).unwrap();
        assert!(tab.board_unhint.is_none());
        tab.board_selected = Some("hint-0001".into());
        let _ = board_key(&mut tab, &delete, &display, 10).unwrap();
        let _ = board_key(&mut tab, &delete, &display, 10).unwrap();
        // Asked of the board (which this test has none of).
        assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| text.contains("not available") && *problem), "{:?}", tab.note);
        // Not on a lead, and not while something is typed.
        let mut tab = board_tab();
        let _ = board_key(&mut tab, &delete, &display, 10).unwrap();
        assert!(tab.board_unhint.is_none());
        tab.board_selected = Some("hint-0001".into());
        tab.draft.insert('x');
        let _ = board_key(&mut tab, &delete, &display, 10).unwrap();
        assert!(tab.board_unhint.is_none());
    }

    #[test]
    fn a_failed_board_read_keeps_the_board_up_for_a_while() {
        let failed = |at| BoardRead { at, snapshot: Err("database is locked".into()), name: None, open: 0, waiting: 0 };
        let arrive = |tab: &mut Open, index: usize, read: BoardRead| {
            tab.board_reading = Some((index, std::sync::Arc::new(Mutex::new(Some(read)))));
        };
        let mut tab = board_tab();
        let now = Instant::now();
        arrive(&mut tab, 0, failed(now));
        let (body, _) = board_body(&mut tab, 100, 30, now);
        assert!(text_of(&body).contains("Race between the upload"), "{}", text_of(&body));
        // Still failing after a while: said.
        arrive(&mut tab, 0, failed(now));
        let later = now + BOARD_GIVE_UP + Duration::from_millis(1);
        let (body, _) = board_body(&mut tab, 100, 30, later);
        assert!(text_of(&body).contains("database is locked"), "{}", text_of(&body));
        // A read for another board than the one up is not kept.
        let mut tab = board_tab();
        arrive(&mut tab, 3, failed(now));
        let _ = board_body(&mut tab, 100, 30, now);
        assert!(tab.board_read.as_ref().is_some_and(|read| read.snapshot.is_ok()));
        assert!(tab.board_failing.is_none());
    }

    #[test]
    fn the_board_fills_its_rows_with_the_tree_and_the_strip() {
        let _guard = worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        worker_feed::clear_for_test();
        let mut tab = board_tab();
        let (frame, cursor) = compose(&mut tab, &ActiveDisplay::new(), (100, 30), Instant::now());
        assert_eq!(frame.len(), 30);
        assert_eq!(cursor.map(|cursor| cursor.1), Some(29), "the prompt on the last row");
        let text = text_of(&frame);
        assert!(text.contains(" ❯ ▾ [*] Race between the upload"), "{text}");
        assert!(text.contains("[*] lead i8  exploring"), "the strip shows the selected lead");
        assert!(text.contains("steps  ███████░░░░░"));
        assert!(text_of(&frame[28..29]).contains(" Fold ") && !text_of(&frame[28..29]).contains("↑↓"), "{}", text_of(&frame[28..29]));
        // A long lead: its row is cut, the strip has it whole, and scrolls.
        let long = "word ".repeat(120);
        if let Some(BoardRead { snapshot: Ok(Some(snapshot)), .. }) = tab.board_read.as_mut() {
            snapshot.intents[1].direction = format!("Does the CI runner's /tmp fill up mid-run? {long}");
            snapshot.intents[1].rationale = long.clone();
            snapshot.intents[2].rationale = long.clone();
        }
        tab.board_selected = Some("intent-0007".into());
        let (frame, _) = compose(&mut tab, &ActiveDisplay::new(), (100, 30), Instant::now());
        let text = text_of(&frame);
        assert!(frame.iter().all(|line| text_width(&text_of(std::slice::from_ref(line))) <= 100));
        assert!(text.contains(" more ") && !text.contains("⇧↓"), "{text}");
        let strip_top = |frame: &[Line]| {
            let rule = frame
                .iter()
                .position(|line| {
                    let row = text_of(std::slice::from_ref(line));
                    row.starts_with("  ─") && (row.contains("more ") || row.contains("above "))
                })
                .unwrap();
            text_of(&frame[rule + 1..rule + 2])
        };
        assert!(strip_top(&frame).contains("lead i7"), "the strip starts at its head");
        let shift = |tab: &mut Open, code| key_event(tab, &KeyEvent::new(code, KeyModifiers::SHIFT), &ActiveDisplay::new()).unwrap();
        shift(&mut tab, KeyCode::Down);
        shift(&mut tab, KeyCode::Down);
        let (frame, _) = compose(&mut tab, &ActiveDisplay::new(), (100, 30), Instant::now());
        assert!(!strip_top(&frame).contains("lead i7"), "scrolled two rows");
        assert!(text_of(&frame).contains(" 2 above"), "{}", text_of(&frame));
        shift(&mut tab, KeyCode::End);
        let (frame, _) = compose(&mut tab, &ActiveDisplay::new(), (100, 30), Instant::now());
        assert!(!text_of(&frame).contains("more "), "at the end: nothing below");
        // Another selection starts at the top of its own.
        tab.board_selected = Some("intent-0008".into());
        let (frame, _) = compose(&mut tab, &ActiveDisplay::new(), (100, 30), Instant::now());
        assert!(strip_top(&frame).contains("[*] lead i8") && !text_of(&frame).contains("above "), "{}", text_of(&frame));
        tab.board_read = Some(BoardRead { at: Instant::now(), snapshot: Ok(Some(board_view::tests::snapshot())), name: Some("flaky-upload".into()), open: 1, waiting: 0 });

        // Short: no strip, and the selection stays in sight.
        tab.board_selected = Some("hint-0001".into());
        let (frame, _) = compose(&mut tab, &ActiveDisplay::new(), (100, 14), Instant::now());
        let text = text_of(&frame);
        assert!(!text.contains("lead i8") && text.contains(" ❯ "), "{text}");
        assert!(text.contains("look at the fixture"), "{text}");
    }

    #[test]
    fn the_prompt_keeps_its_cursor_in_sight() {
        let mut tab = Open::new(View::Board);
        tab.board_read = Some(BoardRead { at: Instant::now(), snapshot: Ok(None), name: Some("perf".into()), open: 1, waiting: 0 });
        tab.draft.insert_str(&"x".repeat(200));
        let (line, cursor) = prompt_line(&tab, 60);
        assert!(cursor < 60);
        assert!(text_of(std::slice::from_ref(&line)).starts_with(" hint › "));
        assert!(text_width(&text_of(&[line])) <= 60);
        tab.draft.home();
        let (_, cursor) = prompt_line(&tab, 60);
        assert_eq!(cursor, text_width(" hint › "));
        // Nothing typed: what it is for, in the place the text will go.
        tab.draft = Draft::default();
        let (line, cursor) = prompt_line(&tab, 60);
        assert_eq!(text_of(&[line]), " hint › hint for s0ma · ^t a lead or finding");
        assert_eq!(cursor, text_width(" hint › "));
    }

    fn press_in(tab: &mut Open, code: KeyCode, modifiers: KeyModifiers) -> Outcome {
        key_event(tab, &KeyEvent::new(code, modifiers), &ActiveDisplay::new()).unwrap()
    }

    #[test]
    fn in_a_shell_every_key_is_the_shells_but_the_switches_and_shift_scrolling() {
        let _guard = crate::vm_terms::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        crate::vm_terms::take_outbox_for_test();
        let mut tab = Open::new(View::Shell { number: 2, session: Some("main".into()) });
        for (code, modifiers) in [
            (KeyCode::Esc, KeyModifiers::NONE),
            (KeyCode::Char('c'), KeyModifiers::CONTROL),
            (KeyCode::Char('d'), KeyModifiers::CONTROL),
            (KeyCode::Up, KeyModifiers::NONE),
            (KeyCode::PageUp, KeyModifiers::NONE),
            (KeyCode::Char('q'), KeyModifiers::NONE),
            (KeyCode::Enter, KeyModifiers::NONE),
        ] {
            assert_eq!(press_in(&mut tab, code, modifiers), Outcome::Stay, "{code:?} {modifiers:?} goes to the shell");
            assert!(matches!(tab.view, View::Shell { .. }));
        }
        assert_eq!(tab.back, 0, "plain PgUp is the shell's");
        use crate::vm_terms::Input;
        assert_eq!(
            crate::vm_terms::take_outbox_for_test(),
            [
                ("main".to_string(), Input::Key("Escape".into())),
                ("main".to_string(), Input::Bytes(vec![3])),
                ("main".to_string(), Input::Bytes(vec![4])),
                ("main".to_string(), Input::Key("Up".into())),
                ("main".to_string(), Input::Key("PPage".into())),
                ("main".to_string(), Input::Text("q".into())),
                ("main".to_string(), Input::Key("Enter".into())),
            ],
            "each to the shell it was typed in, in order"
        );
        assert_eq!(press_in(&mut tab, KeyCode::PageUp, KeyModifiers::SHIFT), Outcome::Stay);
        assert!(tab.back > 0, "Shift+PgUp scrolls the view");
        assert_eq!(press_in(&mut tab, KeyCode::Char('x'), KeyModifiers::NONE), Outcome::Stay);
        assert_eq!(tab.back, 0, "typing follows the shell again");
        assert!(tab.draft.is_empty(), "nothing is drafted: it went to the shell");
        // Ctrl+T, Ctrl+digits (Ctrl+\\ and its kin as most terminals send
        // them) are the shell's.
        crate::vm_terms::take_outbox_for_test();
        assert_eq!(press_in(&mut tab, KeyCode::Char('t'), KeyModifiers::CONTROL), Outcome::Stay);
        assert_eq!(press_in(&mut tab, KeyCode::Char('4'), KeyModifiers::CONTROL), Outcome::Stay);
        assert_eq!(press_in(&mut tab, KeyCode::Char('1'), KeyModifiers::CONTROL), Outcome::Stay);
        assert!(matches!(tab.view, View::Shell { .. }));
        let sent = crate::vm_terms::take_outbox_for_test();
        assert_eq!(sent[..2], [("main".to_string(), Input::Bytes(vec![0x14])), ("main".to_string(), Input::Bytes(vec![0x1c]))], "Ctrl+\\ quits the program");
        assert_eq!(press_in(&mut tab, KeyCode::Char('#'), KeyModifiers::ALT), Outcome::Stay);
        assert_eq!(tab.view, View::Shell { number: 3, session: None }, "Alt+Shift+3: another shell");
        let mut tab = Open::new(View::Shell { number: 2, session: Some("main".into()) });
        assert_eq!(press_in(&mut tab, KeyCode::Char('1'), KeyModifiers::ALT), Outcome::Closed, "Alt+1 leaves");
        // A key with no shell on it yet says so rather than typing anywhere.
        let mut empty = Open::new(View::Shell { number: 5, session: None });
        assert_eq!(press_in(&mut empty, KeyCode::Char('l'), KeyModifiers::NONE), Outcome::Stay);
        assert!(empty.note.as_ref().is_some_and(|(text, problem, _)| *problem && text.contains("no shell")));
    }

    #[test]
    fn the_wheel_scrolls_every_tab() {
        for view in [View::Board, View::Subagent(2), View::Shell { number: 2, session: Some("main".into()) }] {
            let mut tab = Open::new(view);
            assert!(wheel(&mut tab, MouseEventKind::ScrollUp));
            assert!(wheel(&mut tab, MouseEventKind::ScrollUp));
            assert_eq!(tab.back, 2 * WHEEL_ROWS);
            assert!(wheel(&mut tab, MouseEventKind::ScrollDown));
            assert_eq!(tab.back, WHEEL_ROWS);
            assert!(!wheel(&mut tab, MouseEventKind::Moved), "only the wheel");
        }
    }

    #[test]
    fn a_shell_shows_down_to_its_cursor_and_the_cursor_is_where_it_has_it() {
        let _guard = crate::vm_terms::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // No shell on the key: says so, and how shells get there.
        let (body, _, cursor, _) = shell_body(3, None, 0, (80, 20));
        assert!(text_of(&body).contains("No shell here.") && cursor.is_none());
        assert!(!text_of(&body).contains("take alt"), "what it is, not how keys fill: {}", text_of(&body));

        // A 10-row pane with 8 rows of text; the prompt on row 7.
        let mut lines: Vec<String> = (1..=7).map(|n| format!("line {n}")).collect();
        lines.push("$ ".into());
        lines.extend([String::new(), String::new()]);
        crate::vm_terms::set_live_for_test(crate::vm_terms::Screen {
            session: "main".into(), lines, cursor: (2, 7), size: (80, 10), dead: false, cursor_shown: true,
            command: "bash".into(), cwd: "/workspace".into(), at: Instant::now(),
        });
        let mut tab = Open::new(View::Shell { number: 2, session: Some("main".into()) });
        // The tab bar (rows as it takes at this width), its rule, 4 of the
        // shell's, the footer.
        // Where the body starts: under the rule (other tests draw at once,
        // so not `body_top`).
        let rule_at = |frame: &[Line]| frame.iter().position(|line| text_of(std::slice::from_ref(line)).chars().all(|c| c == '─')).unwrap();
        let (frame, _) = compose(&mut tab, &ActiveDisplay::new(), (60, 9), Instant::now());
        let top = rule_at(&frame) + 1;
        let (frame, cursor) = compose(&mut tab, &ActiveDisplay::new(), (60, (top + 5) as u16), Instant::now());
        let shown = plain_text(&text_of(&frame[top..top + 4]));
        assert_eq!(shown, "line 5\nline 6\nline 7\n$ ", "down to the cursor's row, not the pane's blank rows");
        assert_eq!(cursor, Some((2, (top + 3) as u16)), "the cursor where the shell has it");
        let footer = text_of(&frame[top + 4..]);
        assert!(footer.starts_with(" main  /workspace  20 columns cut off"), "{footer}");
        assert!(!footer.contains("bash") && !footer.contains("80×10"), "what it is, not everything about it: {footer}");
        // Scrolled back: the rows above, and the cursor out of the way.
        tab.back = 2;
        let (frame, cursor) = compose(&mut tab, &ActiveDisplay::new(), (60, (top + 5) as u16), Instant::now());
        assert_eq!(plain_text(&text_of(&frame[top..top + 4])), "line 3\nline 4\nline 5\nline 6");
        assert_eq!(cursor, None, "hidden while scrolled back");
        assert!(text_of(&frame[top + 4..]).contains("↑ 2 rows"));
        crate::vm_terms::want_history("main", false);
    }

    #[test]
    fn a_full_row_keeps_its_last_character() {
        let erase = "\x1b[K";
        let mut out = Vec::new();
        let full: Line = vec![(Style::Plain, "abcdefghij".into())];
        let short: Line = vec![(Style::Plain, "abc".into())];
        let long: Line = vec![(Style::Plain, "abcdefghijklmnop".into())];
        paint_to(&mut out, &[], &[full], Some((0, 0)), 10).unwrap();
        let written = String::from_utf8_lossy(&out);
        assert!(written.contains("abcdefghij") && !written.contains(erase), "no erase after a full row: {written:?}");
        out.clear();
        paint_to(&mut out, &[], &[long], Some((0, 0)), 10).unwrap();
        let written = String::from_utf8_lossy(&out);
        assert!(written.contains("abcdefghij") && !written.contains("k") && !written.contains(erase), "cut at the width: {written:?}");
        out.clear();
        paint_to(&mut out, &[], &[short], Some((0, 0)), 10).unwrap();
        let written = String::from_utf8_lossy(&out);
        let after = written.find("abc").map(|at| &written[at + 3..]).unwrap_or_default();
        assert!(after.contains(erase), "a short row erases what was after it: {written:?}");
    }

    /// A shell's colours kept, its default made the tabs' white, and its row
    /// cut to the width with the codes intact.
    #[test]
    fn a_shell_row_keeps_its_colours_on_white() {
        assert_eq!(whiten("\x1b[1;32mnetbox\x1b[0m ~ \x1b[39mx\x1b[m"), "\x1b[1;32mnetbox\x1b[0;37m ~ \x1b[37mx\x1b[0;37m");
        // A 256-colour or true-colour 0 is a colour, not a reset.
        assert_eq!(whiten("\x1b[38;5;0mk\x1b[38;2;0;0;0mk"), "\x1b[38;5;0mk\x1b[38;2;0;0;0mk");
        // Other codes pass as they are.
        assert_eq!(whiten("\x1b[2Ja"), "\x1b[2Ja");
        let row = shell_row("\x1b[34mabcdefghij\x1b[0m", 4);
        assert_eq!(row.len(), 1);
        assert_eq!(plain_text(&row[0].1), "abcd");
        assert!(row[0].1.starts_with("\x1b[37m\x1b[34m") && row[0].1.ends_with("\x1b[0m"), "{:?}", row[0].1);
    }

    #[test]
    fn ctrl_o_folds_a_shells_commands_to_a_line_each() {
        // Rows as tmux gives them back for the shell's settings.
        let meta = |said: &str| format!("\x1b[1mroot@build\x1b[0;2m\x1b[36m  /workspace\x1b[0m{said}");
        let ok = format!("{}\x1b[38;2;51;255;51m[+]\x1b[90m 41.2s\x1b[39m", " ".repeat(40));
        let failed = format!("{}\x1b[38;2;255;95;95m[-] exit 101 · 3.1s\x1b[39m", " ".repeat(30));
        let why = "\x1b[3m\x1b[90m# \x1b[0m\x1b[38;2;51;255;51ms0ma\x1b[0;3m\x1b[90m: build to check the fix\x1b[0m".to_string();
        let dollar = |command: &str| format!("\x1b[90m$\x1b[39m {command}");
        let lines: Vec<String> = vec![
            meta(""),
            why,
            dollar("\x1b[38;2;51;255;51mcargo build --release\x1b[39m"),
            "    Finished `release` profile".into(),
            String::new(),
            meta(&ok),
            dollar("\x1b[38;2;95;175;255mcargo test vm_terms\x1b[39m"),
            "  --lib".into(),
            "test result: FAILED".into(),
            String::new(),
            meta(&failed),
            dollar(""),
        ];
        let (rows, cursor) = compact(&lines, (2, 11), "build", 100);
        let plain: Vec<String> = rows.iter().map(|row| plain_text(row)).collect();
        assert_eq!(
            plain,
            [
                "root@build /workspace $ cargo build --release  # s0ma: build to check the fix",
                "    Finished `release` profile",
                "",
                "root@build /workspace $ cargo test vm_terms",
                "  --lib",
                "test result: FAILED",
                "",
                "root@build /workspace [-] 101 $ ",
            ]
        );
        assert!(rows[0].ends_with("\x1b[3;90m# s0ma: build to check the fix\x1b[0m"), "{:?}", rows[0]);
        assert!(rows[7].contains("\x1b[38;2;255;95;95m[-] 101\x1b[0m "), "{:?}", rows[7]);
        assert_eq!(cursor, (2 + "root@build /workspace [-] 101 ".len(), 7), "the cursor moves with its row, and along it");
        // No room beside the command: the reason stays above it.
        let (rows, cursor) = compact(&lines[..4], (0, 3), "build", 50);
        let plain: Vec<String> = rows.iter().map(|row| plain_text(row)).collect();
        assert_eq!(plain, ["# s0ma: build to check the fix", "root@build /workspace $ cargo build --release", "    Finished `release` profile"]);
        assert_eq!(cursor, (0, 2));
        // A command that fills its row stays under its first line.
        let long = vec![meta(""), dollar(&"x".repeat(90))];
        assert_eq!(compact(&long, (0, 1), "build", 92).0, long);
        // Another shell's prompt is left as it is.
        let other = vec![meta(""), dollar("ls")];
        assert_eq!(compact(&other, (0, 1), "main", 80).0, other);
    }

}
