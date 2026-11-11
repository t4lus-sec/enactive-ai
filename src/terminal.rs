use crate::llm::StreamEvent;
use crate::markdown::MarkdownStream;
use crate::page;
use crate::token_meter::TokenMeter;
use crate::transcript::{
    DEFAULT_CHARS_PER_TOKEN, Detail, EARLIER_NOT_SHOWN, Entry, Mark, Out, Recorded, Renderer, TranscriptRecord, WORK_STYLE,
    gutter_width,
};

use crossterm::{
    cursor::{Hide, MoveTo, RestorePosition, SavePosition, Show, position},
    queue,
    terminal::{self, BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use std::{
    io::{self, Write},
    sync::{
        Arc, Mutex, MutexGuard, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

/// s0ma's `s0ma>`, name and `>` alike: CRT phosphor green (#33FF33), the
/// markdown headings' colour, 24-bit so a terminal theme can't remap it.
pub const NAME_STYLE: &str = "\x1b[1;38;2;51;255;51m";
pub const CHEVRON_STYLE: &str = "\x1b[1;96m";
pub const TEXT_STYLE: &str = "\x1b[37m";
/// The operator's `you>`: blue beside s0ma's green, so a glance down the
/// left edge shows who is speaking. 24-bit, like the markdown headings, so a
/// terminal theme can't remap it.
pub const OPERATOR_STYLE: &str = "\x1b[1;38;2;95;175;255m";
/// The operator's words: a step brighter than s0ma's reply text (`37`).
pub const OPERATOR_TEXT_STYLE: &str = "\x1b[97m";
pub const THINKING_STYLE: &str = "\x1b[3;37m";
pub const STATUS_STYLE: &str = "\x1b[2;36m";
pub const RESET: &str = "\x1b[0m";

const BANNER: &str = r#"
 ░▒▓███████▓▒░▒▓████████▓▒░▒▓██████████████▓▒░ ░▒▓██████▓▒░
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
 ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░
       ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
       ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
"#;

const ACTIVE_INPUT_ROWS: u16 = 5;
/// Columns before the operator's text: `you>  `, as wide as `s0ma> `, and
/// the indent of every line after the first.
pub const OPERATOR_PREFIX: usize = 6;
/// One row above the input for the progress line, when there is room.
const STATUS_ROWS: u16 = 1;
/// The fewest rows a turn's pane is drawn in: the page's top rows, two rows
/// of conversation (the least a scroll region takes) and an input row.
/// The rows a pane needs: the page's top rows and three.
fn min_rows() -> u16 {
    page::top_rows() + 3
}
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const SPINNER_FRAME: Duration = Duration::from_millis(100);

pub fn banner() {
    say(BANNER);

    say(
        r#"
  ________________
< Hack the planet! >
  ----------------
         \   ^__^
          \  (oo)\_______
             (__)\       )\/\
                 ||----w |
                 ||     ||

"#,
    );
}

// ============================================================
// HARNESS OUTPUT
// ============================================================

/// The end of the conversation for the shell's screen on the way out (see
/// `ActiveDisplay::last_rows`), and the page left with it under the shell's
/// own lines.
pub fn leave_page() {
    if !page::on() {
        return;
    }
    let last = SHARED.get().map(ActiveDisplay::last_rows).unwrap_or_default();
    page::leave_with(&last);
}

/// The operator's display, once the interactive prompt has made one. Harness
/// lines printed from anywhere go through it, so they land where the display
/// puts output, the mode decides whether they show, and the page drawn again
/// from the record (Ctrl+O, a resize, a tab closing) keeps them.
static SHARED: OnceLock<ActiveDisplay> = OnceLock::new();

/// What this s0ma's display does that a terminal attached to it does too
/// (`session.rs`): the attached terminal draws each the way this one does,
/// at its own width and in its own mode, so it looks and works the same.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Mirrored {
    /// The steering pane opened, or closed: a turn's input mode.
    Begin,
    End,
    BeginTurn,
    Entry(Entry),
    Activity(Activity),
    RequestSent,
    GenerationStarted,
    Generated(usize),
    Usage { completion: u64, prompt: Option<u64>, cached: Option<u64>, last: bool },
    EndRequest,
    Transient(Option<String>),
    Queued(usize),
    /// A job's close purged the prompt history: attached terminals forget
    /// theirs too (`jobs.rs`).
    HistoryPurged,
}

/// Where mirrored events go, and whether anything is listening.
struct Mirror {
    sink: Box<dyn Fn(Mirrored) + Send + Sync>,
    listening: Arc<AtomicUsize>,
}

static MIRROR: OnceLock<Mirror> = OnceLock::new();

/// Send what the display does to `sink` while `listening` (how many
/// terminals are attached, or attaching) is above zero. Once.
pub fn mirror_to(sink: Box<dyn Fn(Mirrored) + Send + Sync>, listening: Arc<AtomicUsize>) {
    let _ = MIRROR.set(Mirror { sink, listening });
}

/// `event` to the attached terminals, if any. Built only when one listens.
pub fn mirror(event: impl FnOnce() -> Mirrored) {
    if let Some(mirror) = MIRROR.get()
        && mirror.listening.load(Ordering::SeqCst) > 0
    {
        (mirror.sink)(event());
    }
}

/// The display as an attached terminal starts from: the conversation so far
/// with each entry's age, and where the turn is.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DisplaySnapshot {
    /// (ms since it began, ms since it was last extended, entry).
    pub entries: Vec<(u64, u64, Entry)>,
    pub dropped: bool,
    pub active: bool,
    pub turn_ms: u64,
    pub activity: Activity,
    pub queued: usize,
}

/// Lines printed before the display existed (the banner, start-up notices),
/// with their total size, handed over when it is shared. Beyond the limit
/// they are only on screen.
static EARLY: Mutex<(Vec<Entry>, usize)> = Mutex::new((Vec::new(), 0));
const EARLY_LIMIT: usize = 64 << 10;

/// Set while this process is writing a machine-readable artefact to stdout
/// (`--print-system-prompt`). Harness lines then go to stderr instead.
///
/// Suppressing them was the other option and is the wrong one: a caller
/// comparing two assemblies needs stdout to hold the prompt and nothing else,
/// but it also needs to be told that a skill was skipped or a snapshot missed,
/// because either changes what the prompt contains. So the lines keep coming,
/// on the stream the caller is not parsing.
static STDOUT_RESERVED: AtomicBool = AtomicBool::new(false);

/// Called once, before anything that might print, by a mode whose stdout is an
/// artefact rather than a conversation.
pub fn reserve_stdout() {
    STDOUT_RESERVED.store(true, Ordering::SeqCst);
}

/// The plain-text form of an entry, for the stderr fallback. `Turn` and `Pass`
/// are display structure with no text of their own and have nothing to say
/// here.
fn entry_text(entry: &Entry) -> Option<&str> {
    match entry {
        Entry::Turn | Entry::Pass => None,
        Entry::Reasoning(text)
        | Entry::Answer(text)
        | Entry::AnswerTail(text)
        | Entry::Status(text)
        | Entry::Note(text)
        | Entry::Line(text)
        | Entry::Raw { text, .. }
        | Entry::Operator(text) => Some(text),
        Entry::Notice { full, .. } => Some(full),
    }
}

/// Make `display` the one harness output goes through, with what was printed
/// before it.
pub fn share(display: &ActiveDisplay) {
    let mut early = EARLY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if SHARED.set(display.clone()).is_err() {
        return;
    }
    let (entries, _) = std::mem::take(&mut *early);
    // Lock order: EARLY, then the display (never the other way round).
    if let Ok(mut state) = display.lock() {
        let now = Instant::now();
        for entry in entries {
            state.record.push(now, entry);
        }
    }
}

/// `entry` through the shared display; before there is one (or if it fails),
/// `print` writes it as plain output, and it is kept for the display.
fn harness_output(entry: Entry, print: impl FnOnce()) {
    // Before the display check: in artefact mode there is never a display, and
    // every line has to be kept off stdout, including the ones printed during
    // start-up before one could exist.
    if STDOUT_RESERVED.load(Ordering::SeqCst) {
        if let Some(text) = entry_text(&entry) {
            eprintln!("{text}");
        }
        return;
    }
    if let Some(display) = SHARED.get() {
        if display.emit(entry).is_err() {
            print();
        }
        return;
    }
    let mut early = EARLY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(display) = SHARED.get() {
        drop(early);
        if display.emit(entry).is_err() {
            print();
        }
        return;
    }
    print();
    let size = entry_bytes(&entry);
    if early.1 + size <= EARLY_LIMIT {
        early.1 += size;
        early.0.push(entry);
    }
}

/// About how much text an entry holds.
fn entry_bytes(entry: &Entry) -> usize {
    match entry {
        Entry::Turn | Entry::Pass => 1,
        Entry::Reasoning(text)
        | Entry::Answer(text)
        | Entry::AnswerTail(text)
        | Entry::Status(text)
        | Entry::Note(text)
        | Entry::Line(text)
        | Entry::Raw { text, .. } => text.len(),
        Entry::Notice { brief, full, .. } => brief.len() + full.len(),
        Entry::Operator(text) => text.len(),
    }
}

/// A line for the operator (a command's result, the sleep manifest), shown in
/// both modes. `text` is printed as it is.
pub fn say(text: &str) {
    harness_output(Entry::Line(text.to_string()), || println!("{text}"));
}

/// Harness status (per-turn mode and knowledge lines, fold and compaction
/// progress): shown expanded, left out of the conversation collapsed.
pub fn say_note(text: &str) {
    harness_output(Entry::Note(safe_text(text)), || println!("{text}"));
}

/// A harness status line, `[tag] what happened`: collapsed, a dim `[*]` line in
/// the gutter (`tag · what happened`); expanded, as it is, like `say`.
pub fn say_status(text: &str) {
    let text = safe_text(text);
    harness_output(Entry::Notice { brief: tagged_brief(&text), full: text.clone(), mark: Mark::Work }, || println!("{text}"));
}

/// `say_status` with its own collapsed wording, for a message too long to
/// read at a glance.
pub fn say_status_as(brief: &str, text: &str) {
    let text = safe_text(text);
    harness_output(Entry::Notice { brief: safe_text(brief), full: text.clone(), mark: Mark::Work }, || println!("{text}"));
}

/// `say_status` for a step that finished: a green `[+]` line collapsed
/// (`tag · what finished`); expanded, as it is.
pub fn say_done(text: &str) {
    let text = safe_text(text);
    harness_output(Entry::Notice { brief: tagged_brief(&text), full: text.clone(), mark: Mark::Done }, || println!("{text}"));
}

/// `say_done` with its own collapsed wording, for a message too long to read
/// at a glance (the sleep manifest).
pub fn say_done_as(brief: &str, text: &str) {
    let text = safe_text(text);
    harness_output(Entry::Notice { brief: safe_text(brief), full: text.clone(), mark: Mark::Done }, || println!("{text}"));
}

/// `say_status` for a problem: a `[-]` line collapsed; expanded, and before
/// the display exists, like `warn`.
pub fn warn_status(text: &str) {
    let text = safe_text(text);
    harness_output(Entry::Notice { brief: tagged_brief(&text), full: text.clone(), mark: Mark::Problem }, || eprintln!("{text}"));
}

/// `warn_status` with its own collapsed wording, for a message too long to
/// read at a glance.
pub fn warn_status_as(brief: &str, text: &str) {
    let text = safe_text(text);
    harness_output(Entry::Notice { brief: safe_text(brief), full: text.clone(), mark: Mark::Problem }, || eprintln!("{text}"));
}

/// `[compaction] window #1..#9: …` → `compaction · window #1..#9: …`. In a
/// block of several lines (`/mcp status`), later lines that repeat the first
/// one's tag drop it, and the rest keep their own indentation.
pub fn tagged_brief(text: &str) -> String {
    let text = text.trim_start_matches(['\n', '\r']).trim_end();
    let mut lines = text.lines();
    let Some(first) = lines.next() else { return String::new() };
    let split = |line: &str| -> Option<(String, String)> {
        let (tag, rest) = line.strip_prefix('[')?.split_once("] ")?;
        (!tag.is_empty() && !tag.contains(char::is_whitespace)).then(|| (tag.to_string(), rest.trim().to_string()))
    };
    let (mut brief, tag) = match split(first.trim()) {
        Some((tag, rest)) => (format!("{tag} · {rest}"), Some(tag)),
        None => (first.trim().to_string(), None),
    };
    for line in lines {
        brief.push('\n');
        match (split(line), &tag) {
            (Some((same, rest)), Some(tag)) if same == *tag => brief.push_str(&rest),
            _ => brief.push_str(line.trim_end()),
        }
    }
    brief
}

/// A warning or error, shown in both modes. Before the display exists it goes
/// to stderr, as it always has.
pub fn warn(text: &str) {
    let text = safe_text(text);
    harness_output(Entry::Line(text.clone()), || eprintln!("{text}"));
}

/// The sleep reflection's reasoning begins: shown like a turn's thinking
/// (one line that counts it, collapsed; the text itself, expanded).
pub fn reflection_begin() {
    harness_output(Entry::Pass, || print!("\n{THINKING_STYLE}[Reflecting]\n"));
}

/// More of the sleep reflection's reasoning.
pub fn reflection(text: &str) {
    let text = safe_text(text);
    harness_output(Entry::Reasoning(text.clone()), || {
        print!("{text}");
        let _ = io::stdout().flush();
    });
}

/// The sleep reflection's reasoning is over.
pub fn reflection_end() {
    harness_output(Entry::Pass, || println!("{RESET}"));
}

// ============================================================
// ACTIVE DISPLAY
// ============================================================

/// Terminal renderer used while s0ma is actively running.
///
/// The terminal is split into two logical areas:
///
/// - upper rows: scrolling s0ma/runtime output
/// - final rows: a progress line, then persistent multiline operator input
///
/// There is only one real hardware cursor. We use the terminal's saved
/// cursor position as s0ma's logical output cursor, while leaving the
/// physical cursor on the operator input row between streamed chunks.
#[derive(Clone)]
pub struct ActiveDisplay {
    state: Arc<Mutex<DisplayState>>,
    /// Shared with `DisplayState::queued`, outside the lock, so the threads
    /// that queue messages (attached terminals, voice) never wait on the
    /// terminal.
    queued: Arc<AtomicUsize>,
}

/// What s0ma is doing, for the progress line.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Activity {
    Waiting,
    Thinking,
    Writing,
    Tool(String),
    /// A deep sleep: the keys are read, and its own lines say how it goes.
    Sleeping,
}

impl Activity {
    fn label(&self) -> String {
        match self {
            Activity::Waiting => "waiting for the model".into(),
            Activity::Thinking => "thinking".into(),
            Activity::Writing => "writing".into(),
            Activity::Tool(name) => format!("running {name}"),
            Activity::Sleeping => "sleeping".into(),
        }
    }
}

struct DisplayState {
    active: bool,
    columns: u16,
    rows: u16,
    input: String,
    /// Byte offset of the operator's cursor within `input`.
    input_cursor: usize,
    operator_revision: u64,
    /// Draws the transcript collapsed or expanded, and word-wraps it.
    renderer: Renderer,
    /// What the display wrote this session, for Ctrl+O's redraw.
    record: TranscriptRecord,
    /// A transient line on screen (collapsed mode's "folding" notice), drawn
    /// again below anything printed while it is up.
    transient: Option<String>,
    /// Tokens per second, for the bottom-right corner.
    meter: TokenMeter,

    /// An external editor owns the screen; output is held until it exits,
    /// then laid out where the transcript turns out to be.
    suspended: bool,
    /// (first, last, entry), consecutive chunks merged.
    held: Vec<(Instant, Instant, Entry)>,

    activity: Activity,
    started: Instant,
    /// Spinner frame last drawn, so ticks only repaint when it changes.
    frame_drawn: u128,
    /// Shown on the progress line until the next key (completion lists etc.).
    hint: Option<String>,
    /// Operator messages sent during the turn and not yet handed to s0ma
    /// (they wait for a safe point; see `agent::SteerPoint`).
    queued: Arc<AtomicUsize>,
    /// A tab has the screen (`tabs.rs`, Alt+2…): output is held and nothing
    /// is drawn until it is closed.
    covered: Option<Cover>,
    /// The next `begin` opens the pane without starting a new pass: an
    /// attached terminal joining a turn in the middle (`load`).
    quiet_pass: bool,
    /// The idle prompt is being read: output is held in `held` and written
    /// above it when the prompt is woken for it (`flush_idle`).
    prompt_up: bool,
    /// The terminal is too small for the pane: output is held (`suspended`)
    /// until a resize makes room again, and the pane comes back then.
    shrunk: bool,
    /// The terminal changed size and the pane has not been laid out for it
    /// yet (`resize` runs on the keyboard thread, after a burst of resizes
    /// settles): since when. Output is held meanwhile, since drawing at the
    /// old size would land on rows the terminal has moved.
    waiting_resize: Option<Instant>,
    /// When `resize` last laid the pane out.
    laid_out_at: Option<Instant>,
    /// The tab bar atop the page, with its colours (`tabs::main_bar`).
    bar: String,
}


/// Set while a `/mistake` view has the alternate screen (`note_view_screen`).
static VIEW_SCREEN: AtomicBool = AtomicBool::new(false);
/// Set once `emergency_restore` has put the terminal back: s0ma is ending,
/// and nothing draws on the shell's terminal any more.
static TORN_DOWN: AtomicBool = AtomicBool::new(false);

/// A `/mistake` view took the alternate screen (`on`) or gave it back, so
/// `emergency_restore` knows to leave it.
pub fn note_view_screen(on: bool) {
    VIEW_SCREEN.store(on, Ordering::SeqCst);
}

/// The terminal was put back after a panic (`emergency_restore`): whatever
/// is still running draws nothing more, and screens it had are not left
/// again (that would move the cursor back over the panic's message).
pub fn torn_down() -> bool {
    TORN_DOWN.load(Ordering::SeqCst)
}

/// Put the terminal back the way a shell expects it, without taking any lock
/// and without the display's state: for a panic that ends s0ma, which may
/// have happened while the display or stdout was held. Writes straight to the
/// terminal (fd 1): leaves s0ma's page, resets the scroll region the
/// steering pane set, pops every keyboard-protocol level s0ma pushed (on
/// both screens), turns off bracketed paste, focus and mouse reports, shows
/// the cursor and resets colours, and leaves raw mode.
pub fn emergency_restore() {
    use std::os::fd::FromRawFd;
    // First, so the keyboard thread and the rest draw nothing after this.
    TORN_DOWN.store(true, Ordering::SeqCst);
    let page = page::take_on();
    let view = VIEW_SCREEN.swap(false, Ordering::SeqCst);
    let codes = restore_codes(page || view);
    MOUSE.store(false, Ordering::SeqCst);
    // SAFETY: fd 1 is the process's stdout, open for its whole life; the
    // `File` is never dropped, so it is not closed here.
    let mut terminal = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(1) });
    let _ = terminal.write_all(codes.as_bytes());
    let _ = terminal.flush();
    let _ = crossterm::terminal::disable_raw_mode();
}

/// What `emergency_restore` writes; `alternate`: the page or a view has the
/// alternate screen.
fn restore_codes(alternate: bool) -> String {
    let mut codes = String::new();
    if alternate {
        codes.push_str("\x1b[<99u\x1b[?1049l");
    }
    // Resetting the scroll region moves the cursor to the top left on most
    // terminals: saved and put back around it, so what follows goes below
    // the input row the cursor is on, over the pane's rows, not over the
    // conversation at the top of the screen.
    codes.push_str("\x1b7\x1b[r\x1b8");
    codes.push_str("\x1b[<99u\x1b[?2004l\x1b[?1004l\x1b[?1006l\x1b[?1002l\x1b[?1000l\x1b[?25h\x1b[0m\r\n\x1b[J");
    codes
}

/// While keys are being read (raw mode), the terminal reports the mouse's
/// buttons and wheel (xterm's 1000, in SGR form): the wheel scrolls a tab, or
/// back through s0ma's conversation on its page (`tabs::history`). Not
/// movement (crossterm's EnableMouseCapture asks for every motion too). Off
/// whenever raw mode is, or the wheel's reports would be echoed as text.
/// Selecting text with the mouse takes Shift in most terminals meanwhile.
/// 1002 adds movement while a button is held (dragging a selection in the
/// editor); plain movement is still not reported.
const MOUSE_ON: &str = "\x1b[?1000h\x1b[?1002h\x1b[?1006h";
const MOUSE_OFF: &str = "\x1b[?1006l\x1b[?1002l\x1b[?1000l";
static MOUSE: AtomicBool = AtomicBool::new(false);

/// Mouse reporting on or off; only a change is written.
pub fn set_mouse(on: bool) {
    if torn_down() {
        return;
    }
    if MOUSE.swap(on, Ordering::SeqCst) != on {
        let mut stdout = io::stdout();
        let _ = stdout.write_all(if on { MOUSE_ON } else { MOUSE_OFF }.as_bytes());
        let _ = stdout.flush();
    }
}

/// What a tab covered.
#[derive(Clone, Copy, Debug)]
struct Cover {
    /// The idle prompt's first row and its height, when the tab opened from
    /// it: reedline draws the prompt there again if the cursor comes back
    /// inside those rows, so the page drawn on closing steps past them.
    prompt: Option<(u16, u16)>,
}

/// How a message the operator sent mid-turn is marked in the transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorNote {
    /// Acted on at once (`/kill`), or nothing to say.
    Plain,
    /// Sent with "send now": cuts the model's output short. (A message that
    /// waits for s0ma's current step to end is shown when s0ma takes it;
    /// see `ActiveDisplay::queued`.)
    SendNow,
}

impl ActiveDisplay {
    pub fn new() -> Self {
        let queued = Arc::new(AtomicUsize::new(0));
        Self {
            queued: Arc::clone(&queued),
            state: Arc::new(Mutex::new(DisplayState {
                active: false,
                columns: 0,
                rows: 0,
                input: String::new(),
                input_cursor: 0,
                operator_revision: 0,
                renderer: Renderer::new(Detail::from_env(), DEFAULT_CHARS_PER_TOKEN),
                record: TranscriptRecord::default(),
                transient: None,
                meter: TokenMeter::default(),
                suspended: false,
                held: Vec::new(),
                activity: Activity::Waiting,
                started: Instant::now(),
                frame_drawn: 0,
                hint: None,
                queued,
                covered: None,
                quiet_pass: false,
                shrunk: false,
                waiting_resize: None,
                laid_out_at: None,
                bar: String::new(),
                prompt_up: false,
            })),
        }
    }

    /// The display's state. A panic while it was held (in a key's handling,
    /// say) is not a reason for every later write to fail: the state is taken
    /// as it was left, and the next draw puts the screen right.
    /// After `emergency_restore`, nothing: the terminal is the shell's again.
    fn lock(&self) -> io::Result<MutexGuard<'_, DisplayState>> {
        if torn_down() {
            return Err(io::Error::other("the terminal has been put back for the shell"));
        }
        Ok(self.state.lock().unwrap_or_else(|poisoned| {
            self.state.clear_poison();
            poisoned.into_inner()
        }))
    }

    /// Enter active display mode.
    ///
    /// The final terminal rows are reserved for progress and operator input.
    /// s0ma's current output location is saved as the logical model cursor.
    /// The physical cursor is then moved to the operator row and left there.
    pub fn begin(&self) -> io::Result<()> {
        // The tab bar the page has, made before the lock is taken.
        if let Ok((columns, _)) = terminal::size() {
            let (bar, _) = crate::tabs::main_bar(columns.saturating_sub(1) as usize);
            self.lock()?.bar = bar;
        }
        {
            // A tab has the screen: the turn starts behind it, and is drawn
            // when the tab closes (`uncover`).
            let mut state = self.lock()?;
            if state.covered.is_some() {
                mirror(|| Mirrored::Begin);
                state.active = true;
                state.suspended = true;
                // What was held before it stays held, to be laid out in order
                // when the tab closes.
                state.input.clear();
                state.input_cursor = 0;
                state.renderer.wrap = WordWrap::default();
                state.renderer.forget_open_line();
                state.activity = Activity::Waiting;
                state.hint = None;
                if std::mem::take(&mut state.quiet_pass) {
                    return Ok(());
                }
                let mut stdout = io::stdout().lock();
                return emit_locked(&mut stdout, &mut state, Entry::Pass);
            }
        }
        let (columns, rows) = terminal::size()?;

        if rows < min_rows() || columns < 8 {
            return Err(io::Error::other(format!(
                "terminal needs at least 8 columns and {} rows for active steering", min_rows()
            )));
        }

        let (column, row) = position()?;

        let mut state = self.lock()?;
        mirror(|| Mirrored::Begin);
        state.active = true;
        state.suspended = false;
        state.shrunk = false;
        state.waiting_resize = None;
        state.held.clear();
        state.input.clear();
        state.input_cursor = 0;
        state.renderer.wrap = WordWrap::default();
        state.renderer.forget_open_line();
        state.activity = Activity::Waiting;
        state.hint = None;

        let mut stdout = io::stdout().lock();
        if page::take_stale() {
            // Something drew over the page since the prompt was (a review
            // window): drawn again, with the pane.
            redraw_active_locked(&mut stdout, &mut state, columns, rows)?;
        } else {
            // Start a NEW transcript line. If it would fall inside the input
            // pane, scroll existing text up first instead of clamping onto it.
            establish_layout_locked(&mut stdout, &mut state, columns, rows, column, row, false)?;
            // The top rows again: a draft taller than the page, as reedline
            // drew it, can have run over them.
            draw_top_locked(&mut stdout, &state)?;
            place_input_cursor_locked(&mut stdout, &state)?;
        }
        // A new pass: reasoning and reply get their headers again.
        if !std::mem::take(&mut state.quiet_pass) {
            emit_locked(&mut stdout, &mut state, Entry::Pass)?;
        }
        stdout.flush()
    }

    /// A new turn starts: the progress line's clock counts from here, across
    /// the steers and folds that end and begin the display within the turn.
    /// The progress line's clock from now: a turn's admission (routing, the
    /// knowledge pass) waits on the model before `begin_turn`.
    pub fn start_clock(&self) -> io::Result<()> {
        self.lock()?.started = Instant::now();
        Ok(())
    }

    pub fn begin_turn(&self) -> io::Result<()> {
        let mut state = self.lock()?;
        mirror(|| Mirrored::BeginTurn);
        state.started = Instant::now();
        state.frame_drawn = 0;
        // s0ma's first words this turn start a message of their own, under
        // its name, even with no `you>` above them (an autonomous turn).
        let mut stdout = io::stdout().lock();
        emit_locked(&mut stdout, &mut state, Entry::Turn)?;
        stdout.flush()
    }

    /// Leave active display mode and restore the terminal to its
    /// ordinary full-screen scrolling behavior.
    pub fn end(&self) -> io::Result<()> {
        let mut state = self.lock()?;

        if !state.active {
            return Ok(());
        }
        mirror(|| Mirrored::End);
        state.shrunk = false;
        state.waiting_resize = None;

        if state.covered.is_some() {
            state.meter.end_request();
            state.active = false;
            state.suspended = false;
            state.input.clear();
            state.input_cursor = 0;
            return Ok(());
        }

        let mut stdout = io::stdout().lock();
        // Whatever the model was streaming is over.
        state.meter.end_request();

        // A very small resize invalidates the old pane coordinates (and the
        // line a collapsed summary was being drawn on).
        let size = terminal::size();
        let moved = size
            .as_ref()
            .is_ok_and(|(columns, rows)| *columns != state.columns || *rows != state.rows || state.suspended);
        if moved {
            state.renderer.forget_open_line();
        }

        // Output held for an editor, a reasoning summary still counting, and
        // the last word the word wrapper may still be holding back.
        let columns = state.columns as usize;
        let mut pending = render_held_locked(&mut state);
        {
            let mut out = Out { text: &mut pending, columns, wrapping: true };
            state.renderer.settle(&mut out);
            let rest = state.renderer.wrap.take_pending(columns);
            out.text.push_str(&rest);
        }

        // Reset margins and append a line rather than clearing clamped old rows.
        if let Ok((_, rows)) = size
            && moved {
                // Ended at a size the pane was not laid out for: the next
                // prompt draws the page again.
                page::mark_stale();
                write_normalized(&mut stdout, &pending)?;
                write!(stdout, "{}", page::region(rows))?;
                queue!(stdout, MoveTo(0, rows.saturating_sub(1)))?;
                write!(stdout, "\r\n")?;
                stdout.flush()?;
                state.active = false;
                state.suspended = false;
                state.input.clear();
                state.input_cursor = 0;
                return Ok(());
            }

        // Return from the visible operator cursor to s0ma's saved
        // logical output location.
        queue!(stdout, RestorePosition)?;
        write_normalized(&mut stdout, &pending)?;
        queue!(stdout, SavePosition)?;

        // The conversation's rows down to the bottom as the scrolling region,
        // under the page's top rows.
        //
        // Changing scrolling margins can affect the physical cursor on
        // some terminals, so s0ma's location was saved again immediately
        // before this.
        write!(stdout, "{}", page::region(state.rows))?;

        // Remove the old reserved pane, then return to the output position
        // where ordinary Reedline interaction can resume.
        clear_input_pane(&mut stdout, &state)?;
        queue!(stdout, RestorePosition,)?;

        stdout.flush()?;

        state.active = false;
        state.input.clear();
        state.input_cursor = 0;

        Ok(())
    }

    /// Reconfigure the active display after a terminal resize. On the thread
    /// that reads keyboard input.
    ///
    /// The page is drawn again for the new size, from the record: terminals
    /// don't reflow the alternate screen (some cut rows off, some leave them
    /// where they were) and may reset the scroll region. Too small for a
    /// pane, output is held until the terminal is big enough again, and the
    /// pane comes back.
    pub fn resize(&self, columns: u16, rows: u16) -> io::Result<()> {
        // The terminal's size once it has stopped changing, not the event's:
        // events arrive late, and dragging a window's edge sends many; each
        // would draw the page again. Output is held meanwhile
        // (`notice_resize`).
        let (columns, rows) = settled_size().unwrap_or((columns, rows));
        let (bar, _) = crate::tabs::main_bar(columns.saturating_sub(1) as usize);
        let mut state = self.lock()?;
        state.bar = bar;
        if state.covered.is_some() {
            // A tab has the screen; closing it lays the pane out for the
            // size then.
            state.waiting_resize = None;
            return Ok(());
        }
        if state.active
            && !state.shrunk
            && state.waiting_resize.is_none()
            && (columns, rows) == (state.columns, state.rows)
            && state.laid_out_at.is_some_and(|at| at.elapsed() < STALE_RESIZE)
        {
            // Just laid out for this size: an event that came after the fact
            // (sent while the last one was being laid out). Not one that went
            // away and came back (maximized and restored): the terminal reset
            // the scroll region then, and the pane is laid out again.
            return Ok(());
        }
        let waited = state.waiting_resize.take().is_some();
        if waited {
            state.suspended = false;
        }
        if !state.active || (state.suspended && !state.shrunk) {
            return Ok(());
        }
        // Everything goes to the terminal in one write: a resize landing
        // between two parts of it (stdout flushes at each line) would find the
        // cursor wherever the first part left it.
        let mut out: Vec<u8> = Vec::new();
        if rows < min_rows() || columns < 8 {
            if !state.shrunk {
                state.shrunk = true;
                state.suspended = true;
                write!(out, "\x1b[r\x1b[H\x1b[2J{STATUS_STYLE}too small{RESET}")?;
            }
        } else {
            state.shrunk = false;
            redraw_active_locked(&mut out, &mut state, columns, rows)?;
            state.laid_out_at = Some(Instant::now());
        }
        let mut stdout = io::stdout().lock();
        stdout.write_all(&out)?;
        stdout.flush()?;
        drop(stdout);
        // It changed again while this was laid out: held, and laid out again.
        if !state.shrunk && terminal::size().is_ok_and(|now| now != (columns, rows)) {
            state.suspended = true;
            state.waiting_resize = Some(Instant::now());
        }
        Ok(())
    }

    /// The turn is ending with a resize not laid out yet (the keyboard
    /// thread had not read its event): laid out now, so `end` finds the
    /// pane where it is. On the thread that read the keys, once it no longer
    /// does (it asks the terminal where the cursor is).
    pub fn catch_up_resize(&self) -> io::Result<()> {
        let (columns, rows) = {
            let mut state = self.lock()?;
            notice_resize(&mut state);
            if !state.active || state.covered.is_some() || (state.waiting_resize.is_none() && !state.shrunk) {
                return Ok(());
            }
            terminal::size()?
        };
        self.resize(columns, rows)
    }

    /// A key's handling panicked (it was caught, and the turn goes on): put
    /// back what it may have left half done. A page covered with no tab
    /// open is drawn again, and a pane put away for an editor or a view comes
    /// back. On the keyboard thread.
    pub fn recover_after_panic(&self, tab_open: bool) -> io::Result<()> {
        let state = self.lock()?;
        let covered = state.covered.is_some();
        let put_away = state.active && state.suspended && state.waiting_resize.is_none() && !state.shrunk;
        drop(state);
        if covered && !tab_open {
            return self.uncover();
        }
        if put_away && !covered {
            return self.resume();
        }
        Ok(())
    }

    /// Hand the whole screen to an external program (the operator's
    /// editor, a review window). Output that arrives meanwhile is held, and
    /// `resume` draws the page again with it.
    pub fn suspend(&self) -> io::Result<()> {
        let mut state = self.lock()?;
        if !state.active || state.suspended || state.covered.is_some() {
            return Ok(());
        }
        let mut stdout = io::stdout().lock();

        // The word in progress goes out; a space after it stays held, since
        // the line continues after the editor.
        let columns = state.columns as usize;
        let pending = state.renderer.wrap.take_word(columns);

        write!(stdout, "{}", page::region(state.rows))?;
        clear_input_pane(&mut stdout, &state)?;
        queue!(stdout, RestorePosition)?;
        write_normalized(&mut stdout, &pending)?;
        queue!(stdout, SavePosition)?;
        stdout.flush()?;

        state.suspended = true;
        Ok(())
    }

    /// Take the screen back after `suspend`: the page drawn again, since
    /// whatever had the screen drew over it (an editor may have left the
    /// alternate screen too). On the thread that reads keyboard input.
    pub fn resume(&self) -> io::Result<()> {
        if !self.lock().is_ok_and(|state| state.active && state.suspended && state.covered.is_none()) {
            return Ok(());
        }
        let (columns, rows) = terminal::size()?;
        let (bar, _) = crate::tabs::main_bar(columns.saturating_sub(1) as usize);
        let mut state = self.lock()?;
        if !state.active || !state.suspended || state.covered.is_some() {
            return Ok(());
        }
        state.bar = bar;
        state.suspended = false;
        state.waiting_resize = None;
        state.shrunk = false;
        if rows < min_rows() || columns < 8 {
            drop(state);
            return self.end();
        }
        page::take_stale();
        let mut stdout = io::stdout().lock();
        redraw_active_locked(&mut stdout, &mut state, columns, rows)
    }

    /// Update the live operator edit shown on the reserved bottom row.
    /// `cursor` is a byte offset into `input`.
    ///
    /// This does not touch s0ma's saved logical output cursor.
    pub fn set_input(&self, input: &str, cursor: usize) -> io::Result<()> {
        let mut state = self.lock()?;

        state.input = input.to_string();
        state.input_cursor = cursor.min(input.len());
        notice_resize(&mut state);

        if !state.active || state.suspended {
            return Ok(());
        }

        let mut stdout = io::stdout().lock();

        render_input_locked(&mut stdout, &state)?;

        stdout.flush()
    }

    /// Show `hint` on the progress line until it is replaced or cleared.
    pub fn set_hint(&self, hint: Option<String>) -> io::Result<()> {
        let mut state = self.lock()?;
        if state.hint == hint {
            return Ok(());
        }
        state.hint = hint;
        redraw_status(&mut state)
    }

    pub fn set_activity(&self, activity: Activity) -> io::Result<()> {
        let mut state = self.lock()?;
        if state.activity == activity {
            return Ok(());
        }
        mirror(|| Mirrored::Activity(activity.clone()));
        state.activity = activity;
        redraw_status(&mut state)
    }

    /// Advance the spinner, and redraw a collapsed line whose count moved on.
    /// Cheap to call often; repaints only when something changed.
    pub fn tick(&self) -> io::Result<()> {
        self.refresh_bar()?;
        let mut state = self.lock()?;
        let now = Instant::now();
        notice_resize(&mut state);
        // A resize whose event never came (or came while the keys were not
        // being read): laid out from here, on the keyboard thread, all the same.
        if state.waiting_resize.is_some_and(|since| now.duration_since(since) > RESIZE_FALLBACK)
            && let Ok((columns, rows)) = terminal::size()
        {
            drop(state);
            return self.resize(columns, rows);
        }
        if state.active && !state.suspended && state.renderer.wants_refresh(now) {
            let mut text = String::new();
            let columns = state.columns as usize;
            state.renderer.refresh(&mut Out { text: &mut text, columns, wrapping: true }, now, false);
            let mut stdout = io::stdout().lock();
            write_rendered_locked(&mut stdout, &mut state, &text)?;
            stdout.flush()?;
        }
        let frame = state.started.elapsed().as_millis() / SPINNER_FRAME.as_millis();
        if frame == state.frame_drawn {
            return Ok(());
        }
        redraw_status(&mut state)
    }

    /// The tab bar atop the page during a turn, made again (a subagent
    /// started or finished, a shell opened) and drawn if it changed. Made
    /// before the display's lock is taken: it reads the subagents', commands'
    /// and shells' own.
    pub fn refresh_bar(&self) -> io::Result<()> {
        let columns = {
            let state = self.lock()?;
            if !state.active {
                return Ok(());
            }
            state.columns
        };
        let (bar, _) = crate::tabs::main_bar(columns.saturating_sub(1) as usize);
        let mut state = self.lock()?;
        if state.bar == bar {
            return Ok(());
        }
        state.bar = bar;
        notice_resize(&mut state);
        if !state.active || state.suspended || state.covered.is_some() {
            return Ok(());
        }
        let mut stdout = io::stdout().lock();
        draw_top_locked(&mut stdout, &state)?;
        place_input_cursor_locked(&mut stdout, &state)?;
        stdout.flush()
    }

    /// The tab bar atop the page between turns, drawn for the terminal's
    /// width now; the cursor is left where it was (at the idle prompt). What
    /// was drawn, so the prompt's wait can tell when it is out of date.
    pub fn draw_idle_bar(&self) -> io::Result<String> {
        let (columns, rows) = terminal::size()?;
        let (bar, _) = crate::tabs::main_bar(columns.saturating_sub(1) as usize);
        let mut state = self.lock()?;
        // A bar of more rows, or fewer: the page is laid out again under it.
        let reflow = state.bar.lines().count() != bar.lines().count();
        state.bar = bar.clone();
        if state.active || state.covered.is_some() || !page::on() {
            return Ok(bar);
        }
        let mut stdout = io::stdout().lock();
        if reflow {
            redraw_idle_locked(&mut stdout, &mut state, columns, rows)?;
            return Ok(bar);
        }
        write!(stdout, "\x1b7{}\x1b8", page::top(&state.bar, columns))?;
        stdout.flush()?;
        Ok(bar)
    }

    /// Commit a submitted steer into the visible transcript.
    ///
    /// The live input row is cleared, then the submitted operator
    /// message is written into the normal scrolling transcript, with a status
    /// line under it saying when s0ma will read it.
    pub fn commit_operator(&self, input: &str, note: OperatorNote) -> io::Result<()> {
        let mut state = self.lock()?;
        let mut stdout = io::stdout().lock();
        state.input.clear();
        state.input_cursor = 0;
        let note = operator_note_text(note, &state.activity);
        mirror(|| Mirrored::Entry(Entry::Operator(safe_text(input))));
        emit_operator_locked(&mut stdout, &mut state, input)?;
        if let Some(note) = note {
            mirror(|| Mirrored::Entry(status_notice(&note, &note)));
            emit_locked(&mut stdout, &mut state, status_notice(&note, &note))?;
        }
        stdout.flush()
    }

    /// `commit_operator` for a message sent from an attached terminal: this
    /// terminal's own draft stays as it is.
    pub fn commit_remote_operator(&self, input: &str, note: OperatorNote) -> io::Result<()> {
        let mut state = self.lock()?;
        let mut stdout = io::stdout().lock();
        let note = operator_note_text(note, &state.activity);
        mirror(|| Mirrored::Entry(Entry::Operator(safe_text(input))));
        emit_operator_locked(&mut stdout, &mut state, input)?;
        if let Some(note) = note {
            mirror(|| Mirrored::Entry(status_notice(&note, &note)));
            emit_locked(&mut stdout, &mut state, status_notice(&note, &note))?;
        }
        stdout.flush()
    }

    /// An operator message written into the transcript where s0ma takes it
    /// (a queued message, at the safe point it waited for), leaving the
    /// draft being typed alone.
    pub fn show_operator(&self, input: &str) -> io::Result<()> {
        let mut state = self.lock()?;
        let mut stdout = io::stdout().lock();
        mirror(|| Mirrored::Entry(Entry::Operator(safe_text(input))));
        emit_operator_locked(&mut stdout, &mut state, input)?;
        stdout.flush()
    }

    /// A message was queued for s0ma's next safe point: a status line now;
    /// the message itself goes into the transcript when s0ma takes it.
    pub fn queued(&self, input: &str) -> io::Result<()> {
        let sleeping = self.lock()?.activity == Activity::Sleeping;
        let mut note = queued_note(input);
        if sleeping {
            note = note.replacen("after the current step", "after the sleep", 1);
        }
        self.emit(status_notice(&note, &note))
    }

    /// `count` messages reached s0ma.
    pub fn delivered(&self, count: usize) -> io::Result<()> {
        let note = delivered_note(count);
        self.emit(status_notice(&note, &format!("↳ {note}")))
    }

    /// A status line of the operator's own doing in the transcript (not
    /// s0ma's output), such as "sending your queued messages now".
    pub fn operator_line(&self, line: &str) -> io::Result<()> {
        let mut state = self.lock()?;
        let mut stdout = io::stdout().lock();
        state.operator_revision = state.operator_revision.wrapping_add(1);
        let line = safe_text(line);
        mirror(|| Mirrored::Entry(status_notice(&line, &line)));
        emit_locked(&mut stdout, &mut state, status_notice(&line, &line))?;
        stdout.flush()
    }

    /// A harness note between display passes (fold progress, context
    /// notices): printed as a plain line expanded, not shown collapsed.
    pub fn note(&self, line: &str) -> io::Result<()> {
        self.emit(Entry::Note(safe_text(line)))
    }

    /// A line for the operator outside a turn (command output, notices),
    /// printed as it is and kept for Ctrl+O's redraw. `text` is already safe.
    pub fn transcript_line(&self, text: &str) -> io::Result<()> {
        self.emit(Entry::Line(text.to_string()))
    }

    /// A notice for the operator with a quiet collapsed form (`Entry::Notice`):
    /// `brief` as a dim `[*]` line collapsed (`[-]` for a `problem`), `full`
    /// as it is expanded.
    pub fn notice(&self, brief: &str, full: &str, problem: bool) -> io::Result<()> {
        self.emit(Entry::Notice { brief: safe_text(brief), full: safe_text(full), mark: if problem { Mark::Problem } else { Mark::Work } })
    }

    /// What the operator submitted at the idle prompt. Reedline drew it
    /// wrapping at the left edge; it is drawn again in its place the way the
    /// transcript shows every message of theirs (`Entry::Operator`: lines
    /// after the first hang under its text), and kept for Ctrl+O's redraw.
    pub fn show_submitted(&self, input: &str) -> io::Result<()> {
        let (columns, _) = terminal::size()?;
        // Reedline leaves the cursor at the start of the line below the
        // prompt and draft, which took this many rows (one more when the
        // draft ends exactly at the edge, as it does).
        let (_, rows) = prompt_geometry(input, input.len(), columns);
        let mut state = self.lock()?;
        let mut stdout = io::stdout().lock();
        mirror(|| Mirrored::Entry(Entry::Operator(safe_text(input))));
        if state.covered.is_some() {
            return emit_locked(&mut stdout, &mut state, Entry::Operator(safe_text(input)));
        }
        write!(stdout, "\x1b[{rows}A\r\x1b[J")?;
        state.columns = columns;
        state.renderer.wrap.col = 0;
        state.renderer.wrap.indent = 0;
        // Lines held while the prompt was up came first, as the record (and
        // every attached terminal) has them.
        if !state.held.is_empty() {
            let held = std::mem::take(&mut state.held);
            let mut text = String::new();
            {
                let mut out = Out { text: &mut text, columns: columns as usize, wrapping: false };
                for (first, last, entry) in &held {
                    state.renderer.apply(&mut out, *first, *last, entry);
                }
                state.renderer.refresh(&mut out, Instant::now(), false);
            }
            write_rendered_locked(&mut stdout, &mut state, &text)?;
            state.renderer.wrap.col = 0;
            state.renderer.wrap.indent = 0;
        }
        emit_locked(&mut stdout, &mut state, Entry::Operator(safe_text(input)))?;
        stdout.flush()
    }

    /// Collapsed mode, outside the display: show `text` on a line of its own
    /// until it is replaced (`None` erases it), so a long harness step such as
    /// a fold doesn't look like a hang. Expanded mode prints the step's own
    /// notes instead.
    pub fn transient(&self, text: Option<&str>) -> io::Result<()> {
        let mut state = self.lock()?;
        mirror(|| Mirrored::Transient(text.map(str::to_string)));
        if state.active || state.covered.is_some() {
            return Ok(());
        }
        let mut stdout = io::stdout().lock();
        if state.transient.take().is_some() {
            write!(stdout, "\r\x1b[2K")?;
        }
        if let Some(text) = text.filter(|_| state.renderer.detail == Detail::Collapsed) {
            let text = safe_text(text);
            write!(stdout, "{}", transient_line(&state, &text))?;
            state.transient = Some(text);
        }
        stdout.flush()
    }

    /// A message was queued for s0ma; the progress line counts it until
    /// `release_steers` says it was handed over. The line catches up at the
    /// next spinner frame.
    pub fn queue_steer(&self) -> io::Result<()> {
        let _state = self.lock()?;
        let queued = self.queued.fetch_add(1, Ordering::SeqCst) + 1;
        mirror(|| Mirrored::Queued(queued));
        Ok(())
    }

    /// `count` queued messages were handed to s0ma (or started a turn).
    pub fn release_steers(&self, count: usize) -> io::Result<()> {
        let _state = self.lock()?;
        let before = self
            .queued
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |queued| Some(queued.saturating_sub(count)))
            .unwrap_or(0);
        mirror(|| Mirrored::Queued(before.saturating_sub(count)));
        Ok(())
    }

    /// Messages queued and not yet handed to s0ma.
    pub fn queued_steers(&self) -> usize {
        self.queued.load(Ordering::SeqCst)
    }

    fn operator_revision(&self) -> io::Result<u64> {
        self.lock().map(|state| state.operator_revision)
    }

    /// The transcript's width in columns, once the display has measured it.
    fn columns(&self) -> Option<usize> {
        self.lock().ok().map(|state| state.columns as usize).filter(|columns| *columns > 0)
    }

    /// What a line of reply text has once the collapsed gutter is taken out:
    /// the width tables are laid out for. The same in both modes, so a
    /// reply drawn again the other way (Ctrl+O) still fits.
    fn content_columns(&self) -> Option<usize> {
        self.columns().map(|columns| columns - gutter_width(columns))
    }

    /// Stream s0ma/runtime text into the scrolling output region.
    ///
    /// Between calls, the real cursor stays on the live `you>` input.
    ///
    /// For each streamed chunk:
    ///
    /// 1. restore s0ma's saved logical cursor;
    /// 2. write the new chunk;
    /// 3. save the new s0ma cursor;
    /// 4. redraw `you>`;
    /// 5. leave the physical cursor on `you>`.
    pub fn write_stream(&self, text: &str) -> io::Result<()> {
        self.emit(Entry::Raw { text: text.to_string(), operator: false })
    }

    /// Reasoning text (already safe), shown as the mode says.
    pub fn reasoning(&self, text: &str) -> io::Result<()> {
        self.emit(Entry::Reasoning(text.to_string()))
    }

    /// Reply text, styled by the markdown renderer; the first of a pass gets
    /// the `s0ma>` prefix.
    pub fn answer(&self, text: &str) -> io::Result<()> {
        self.emit(Entry::Answer(text.to_string()))
    }

    /// Reply text the markdown renderer held back and released late.
    pub fn answer_tail(&self, text: &str) -> io::Result<()> {
        self.emit(Entry::AnswerTail(text.to_string()))
    }

    /// A harness status line (already safe). Collapsed, tool lines fold into
    /// one line per call.
    pub fn status(&self, message: &str) -> io::Result<()> {
        self.emit(Entry::Status(message.to_string()))
    }

    fn emit(&self, entry: Entry) -> io::Result<()> {
        let mut state = self.lock()?;
        let mut stdout = io::stdout().lock();
        mirror(|| Mirrored::Entry(entry.clone()));
        emit_locked(&mut stdout, &mut state, entry)?;
        stdout.flush()
    }

    /// A model request was sent. Nothing is measured until its output starts.
    pub fn request_sent(&self) -> io::Result<()> {
        let mut state = self.lock()?;
        mirror(|| Mirrored::RequestSent);
        state.meter.request();
        Ok(())
    }

    /// A generation started streaming (see `StreamEvent::Started`); its
    /// prefill is timed from now.
    pub fn generation_started(&self) -> io::Result<()> {
        let mut state = self.lock()?;
        mirror(|| Mirrored::GenerationStarted);
        state.meter.started(Instant::now());
        Ok(())
    }

    /// `chars` characters of reasoning, reply or tool-call arguments streamed.
    pub fn generated(&self, chars: usize) -> io::Result<()> {
        let mut state = self.lock()?;
        mirror(|| Mirrored::Generated(chars));
        state.meter.text(chars, Instant::now());
        Ok(())
    }

    /// The server's count of tokens generated in this request so far; `last`
    /// for its closing report. Also calibrates the thinking line's estimate.
    pub fn usage(
        &self,
        completion_tokens: u64,
        prompt_tokens: Option<u64>,
        cached_tokens: Option<u64>,
        last: bool,
    ) -> io::Result<()> {
        let mut state = self.lock()?;
        mirror(|| Mirrored::Usage { completion: completion_tokens, prompt: prompt_tokens, cached: cached_tokens, last });
        // The prompt first: a closing report ends the request, and its prefill
        // is worked out as it ends.
        state.meter.prompt(prompt_tokens, cached_tokens);
        state.meter.usage(completion_tokens, last, Instant::now());
        state.renderer.chars_per_token = state.meter.chars_per_token();
        Ok(())
    }

    /// The model's request is over (its tool calls are about to run).
    pub fn end_request(&self) -> io::Result<()> {
        let mut state = self.lock()?;
        mirror(|| Mirrored::EndRequest);
        state.meter.end_request();
        Ok(())
    }

    /// The display as a terminal attaching now starts from, handed to
    /// `register` while nothing else can change it: whatever happens after
    /// is mirrored to it, and nothing before is mirrored twice.
    pub fn snapshot(&self, register: impl FnOnce(DisplaySnapshot)) -> io::Result<()> {
        let state = self.lock()?;
        let now = Instant::now();
        let ms = |at: Instant| now.saturating_duration_since(at).as_millis() as u64;
        let (entries, dropped) = state.record.all();
        let entries = entries.map(|recorded| (ms(recorded.first), ms(recorded.last), recorded.entry.clone())).collect();
        register(DisplaySnapshot {
            entries,
            dropped,
            active: state.active,
            turn_ms: ms(state.started),
            activity: state.activity.clone(),
            queued: self.queued.load(Ordering::SeqCst),
        });
        Ok(())
    }

    /// An attached terminal starts from `snapshot` (see `snapshot`): the
    /// record, the turn's clock and activity. Drawn by `redraw`; a turn in
    /// progress is joined without starting a new pass.
    pub fn load(&self, snapshot: DisplaySnapshot) -> io::Result<()> {
        let mut state = self.lock()?;
        let now = Instant::now();
        let at = |ms: u64| now.checked_sub(Duration::from_millis(ms)).unwrap_or(now);
        for (first, last, entry) in snapshot.entries {
            state.record.push_recorded(Recorded { first: at(first), last: at(last), entry });
        }
        if snapshot.dropped {
            state.record.set_dropped();
        }
        state.started = at(snapshot.turn_ms);
        state.activity = snapshot.activity;
        state.quiet_pass = snapshot.active;
        self.queued.store(snapshot.queued, Ordering::SeqCst);
        Ok(())
    }

    /// The page drawn from the record, as Ctrl+O draws it, with the turn's
    /// pane under it if one is running.
    pub fn redraw(&self) -> io::Result<()> {
        let (columns, rows) = terminal::size()?;
        let (bar, _) = crate::tabs::main_bar(columns.saturating_sub(1) as usize);
        let mut state = self.lock()?;
        state.bar = bar;
        let mut stdout = io::stdout().lock();
        if state.active && !state.suspended && state.covered.is_none() {
            return redraw_active_locked(&mut stdout, &mut state, columns, rows);
        }
        if state.active || state.covered.is_some() {
            return Ok(());
        }
        redraw_idle_locked(&mut stdout, &mut state, columns, rows)?;
        stdout.flush()
    }

    /// Between turns: the page drawn again for the terminal's size now (a
    /// resize, Ctrl+L, a window that drew over it), if nothing else has the
    /// screen. Reedline draws its prompt under it. `prompt`: the draft and
    /// cursor of the prompt on the screen now, if one is (see
    /// `toggle_detail_idle`).
    pub fn redraw_idle(&self, prompt: Option<(&str, usize)>) -> io::Result<()> {
        let (columns, rows) = terminal::size()?;
        let (bar, _) = crate::tabs::main_bar(columns.saturating_sub(1) as usize);
        let mut state = self.lock()?;
        state.bar = bar;
        if state.active || state.covered.is_some() {
            return Ok(());
        }
        let prompt = prompt.and_then(|(draft, cursor)| {
            let (above, prompt_rows) = prompt_geometry(draft, cursor, columns);
            position().ok().map(|(_, row)| (row.saturating_sub(above), prompt_rows))
        });
        let mut stdout = io::stdout().lock();
        redraw_idle_locked(&mut stdout, &mut state, columns, rows)?;
        stdout.flush()?;
        if let (Some((start, prompt_rows)), Ok((_, mut row))) = (prompt, position()) {
            while row > start && row <= start.saturating_add(prompt_rows) && row < rows.saturating_sub(1) {
                write!(stdout, "\r\n")?;
                row += 1;
            }
            stdout.flush()?;
        }
        Ok(())
    }

    /// The conversation as rows `columns` wide (`page::rows`), drawn the
    /// `detail` way (the display's own, or the other with `other_detail`),
    /// for looking back through it (`tabs::View::History`).
    pub fn conversation_rows(&self, columns: usize, other_detail: bool) -> io::Result<Vec<String>> {
        let state = self.lock()?;
        let mut detail = state.renderer.detail;
        if other_detail {
            detail = detail.toggled();
        }
        let (_, text) = redraw_conversation(&state.record, detail, state.renderer.chars_per_token, columns, !state.active);
        drop(state);
        Ok(page::rows(&text, columns).rows)
    }

    /// How many rows of the conversation the page shows at `rows` high: under
    /// the top two, above the turn's pane or the prompt's row.
    pub fn page_rows(&self, rows: u16) -> usize {
        let active = self.lock().is_ok_and(|state| state.active);
        let below = if active { pane_rows(rows) } else { 1 };
        rows.saturating_sub(page::top_rows() + below) as usize
    }

    /// The end of the conversation as the page shows it now, for the shell's
    /// screen when s0ma leaves the page (`page::leave_with`).
    pub fn last_rows(&self) -> Vec<String> {
        let Ok((columns, rows)) = terminal::size() else {
            return Vec::new();
        };
        let Ok(state) = self.lock() else {
            return Vec::new();
        };
        let (detail, chars_per_token) = (state.renderer.detail, state.renderer.chars_per_token);
        let (_, text) = redraw_conversation(&state.record, detail, chars_per_token, columns as usize, true);
        drop(state);
        let mut shown = page::rows(&text, columns as usize).rows;
        while shown.last().is_some_and(|row| page::width(row) == 0) {
            shown.pop();
        }
        let keep = rows.saturating_sub(page::top_rows() + 1) as usize;
        shown.split_off(shown.len().saturating_sub(keep))
    }

    /// Changes whenever the record does: what `conversation_rows` would give
    /// is the same while this is.
    pub fn record_revision(&self) -> u64 {
        self.lock().map_or(0, |state| state.record.revision())
    }

    /// What another terminal's display did, done here (`Mirrored`). `Begin`
    /// and `End` change the input mode, which is the caller's to do.
    pub fn apply(&self, event: Mirrored) -> io::Result<()> {
        match event {
            Mirrored::Begin | Mirrored::End | Mirrored::HistoryPurged => Ok(()),
            Mirrored::BeginTurn => self.begin_turn(),
            Mirrored::Entry(Entry::Operator(text)) => {
                let mut state = self.lock()?;
                let mut stdout = io::stdout().lock();
                emit_operator_locked(&mut stdout, &mut state, &text)?;
                stdout.flush()
            }
            Mirrored::Entry(entry) => self.emit(entry),
            Mirrored::Activity(activity) => self.set_activity(activity),
            Mirrored::RequestSent => self.request_sent(),
            Mirrored::GenerationStarted => self.generation_started(),
            Mirrored::Generated(chars) => self.generated(chars),
            Mirrored::Usage { completion, prompt, cached, last } => self.usage(completion, prompt, cached, last),
            Mirrored::EndRequest => self.end_request(),
            Mirrored::Transient(text) => self.transient(text.as_deref()),
            Mirrored::Queued(count) => {
                self.queued.store(count, Ordering::SeqCst);
                Ok(())
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn input_for_test(&self) -> String {
        self.lock().map(|state| state.input.clone()).unwrap_or_default()
    }

    /// The idle prompt is being read (`prompt_up`), or not any more.
    pub fn prompt_up(&self, up: bool) {
        if let Ok(mut state) = self.lock() {
            state.prompt_up = up;
        }
    }

    /// Output held while the idle prompt was up, waiting to be written.
    pub fn idle_output_waiting(&self) -> bool {
        self.lock().is_ok_and(|state| !state.active && state.covered.is_none() && !state.held.is_empty())
    }

    /// Write what was held while the idle prompt was up, at the cursor (where
    /// the prompt was erased).
    pub fn flush_idle(&self) -> io::Result<()> {
        let mut state = self.lock()?;
        if state.active || state.covered.is_some() {
            return Ok(());
        }
        let held = std::mem::take(&mut state.held);
        if let Ok((columns, rows)) = terminal::size() {
            state.columns = columns;
            state.rows = rows;
        }
        let columns = state.columns as usize;
        let mut text = String::new();
        {
            let mut out = Out { text: &mut text, columns, wrapping: false };
            for (first, last, entry) in &held {
                state.renderer.apply(&mut out, *first, *last, entry);
            }
            state.renderer.refresh(&mut out, Instant::now(), false);
        }
        let mut stdout = io::stdout().lock();
        write_rendered_locked(&mut stdout, &mut state, &text)?;
        stdout.flush()
    }

    /// The submitted line reedline left on the screen, taken off it: on an
    /// attached terminal it is shown when this s0ma takes it, as everywhere.
    pub fn erase_submitted(&self, input: &str) -> io::Result<()> {
        let (columns, _) = terminal::size()?;
        let (_, rows) = prompt_geometry(input, input.len(), columns);
        let state = self.lock()?;
        if state.covered.is_some() {
            return Ok(());
        }
        let mut stdout = io::stdout().lock();
        write!(stdout, "\x1b[{rows}A\r\x1b[J")?;
        stdout.flush()
    }

    /// The idle prompt's right side: the last request's rates,
    /// `prefill 3.1k tok/s · decode 57 tok/s`, or where the keys are listed
    /// before there is anything to measure.
    pub fn idle_hint(&self) -> String {
        let Ok(state) = self.lock() else {
            return String::new();
        };
        let rates = rate_segments(state.meter.prefill_label(), state.meter.last_label());
        if rates.is_empty() {
            return "/help".to_string();
        }
        rates.into_iter().map(|(text, _)| text).collect::<Vec<_>>().join(" · ")
    }

    /// Ctrl+O during a turn: switch between collapsed and expanded, and draw
    /// the page again the new way, from the session's record, then the turn
    /// carries on in the new mode.
    pub fn toggle_detail(&self) -> io::Result<Detail> {
        let mut state = self.lock()?;
        let detail = state.renderer.detail.toggled();
        let chars_per_token = state.renderer.chars_per_token;
        if !state.active || state.suspended {
            state.renderer = Renderer::new(detail, chars_per_token);
            return Ok(detail);
        }

        state.renderer.detail = detail;
        let mut stdout = io::stdout().lock();
        let (columns, rows) = (state.columns, state.rows);
        redraw_active_locked(&mut stdout, &mut state, columns, rows)?;
        Ok(detail)
    }

    /// A tab takes the whole screen (`tabs.rs`). Everything s0ma and the
    /// harness write meanwhile is held, a turn may begin or end behind it,
    /// and `uncover` draws the page again. `idle_prompt`: the idle prompt's
    /// draft and cursor, when the tab opens from it.
    pub fn cover(&self, idle_prompt: Option<(&str, usize)>) -> io::Result<()> {
        let mut state = self.lock()?;
        if state.covered.is_some() {
            return Ok(());
        }
        let was_active = state.active && !state.suspended;
        let mut prompt = None;
        let mut stdout = io::stdout().lock();
        if was_active {
            // Held from here, as for an editor; the page is drawn again
            // from the record when the tab closes.
            state.suspended = true;
        } else if let Some((draft, cursor)) = idle_prompt
            && let (Ok((columns, _)), Ok((_, row))) = (terminal::size(), position())
        {
            let (above, rows) = prompt_geometry(draft, cursor, columns);
            prompt = Some((row.saturating_sub(above), rows));
        }
        // The tab paints every row, each to its end, so nothing of the page
        // shows through; no clear first, which would leave the screen blank
        // while the tab is laid out.
        write!(stdout, "\x1b[r")?;
        stdout.flush()?;
        state.covered = Some(Cover { prompt });
        Ok(())
    }

    /// The tab is closed: the page is drawn again from the record, with what
    /// arrived meanwhile, and the turn's pane if one is running (the one the
    /// tab opened on, or one that began behind it). Between turns reedline
    /// draws its prompt under it. Called on the thread that reads the
    /// keyboard, since it may ask the terminal where the cursor is.
    pub fn uncover(&self) -> io::Result<()> {
        let (columns, rows) = terminal::size()?;
        let (bar, _) = crate::tabs::main_bar(columns.saturating_sub(1) as usize);
        let mut state = self.lock()?;
        let Some(cover) = state.covered.take() else {
            return Ok(());
        };
        state.bar = bar;
        // Laid out below for the size the terminal has now.
        state.waiting_resize = None;
        state.shrunk = false;
        let mut stdout = io::stdout().lock();

        if state.active {
            if rows < min_rows() || columns < 8 {
                state.suspended = false;
                drop(stdout);
                drop(state);
                return self.end();
            }
            return redraw_active_locked(&mut stdout, &mut state, columns, rows);
        }

        redraw_idle_locked(&mut stdout, &mut state, columns, rows)?;
        stdout.flush()?;
        // Reedline puts its prompt back at the old first row when the cursor
        // comes back inside the rows it had (see `toggle_detail_idle`).
        if let (Some((start, prompt_rows)), Ok((_, mut row))) = (cover.prompt, position()) {
            while row > start && row <= start.saturating_add(prompt_rows) && row < rows.saturating_sub(1) {
                write!(stdout, "\r\n")?;
                row += 1;
            }
            stdout.flush()?;
        }
        Ok(())
    }

    /// What s0ma is doing, for a tab's bar: None between turns.
    pub fn working(&self) -> Option<String> {
        let state = self.lock().ok()?;
        state.active.then(|| format!("{} · {}", state.activity.label(), format_elapsed(state.started.elapsed())))
    }

    /// Ctrl+O at the idle prompt: switch modes, and draw the conversation
    /// again the new way, in place. `draft` and `cursor` are what the prompt
    /// holds; reedline draws it again below the conversation.
    pub fn toggle_detail_idle(&self, draft: &str, cursor: usize) -> io::Result<Detail> {
        let mut state = self.lock()?;
        let detail = state.renderer.detail.toggled();
        let chars_per_token = state.renderer.chars_per_token;
        state.renderer = Renderer::new(detail, chars_per_token);
        if state.active || state.record.is_empty() || state.covered.is_some() {
            return Ok(detail);
        }
        // Lines held while the prompt was up are in the record, so drawn below.
        state.held.clear();

        let (columns, rows) = terminal::size()?;
        // Where the prompt was, so reedline can't reuse those rows over the
        // redraw (it re-anchors a prompt left by a host command).
        let (above, prompt_rows) = prompt_geometry(draft, cursor, columns);
        let prompt_start = position().ok().map(|(_, row)| row.saturating_sub(above));

        let mut stdout = io::stdout().lock();
        // The record ends as the screen did (a finished turn's blank line
        // included), so reedline draws the prompt where it was drawn before.
        redraw_idle_locked(&mut stdout, &mut state, columns, rows)?;
        stdout.flush()?;
        // Reedline puts the prompt back at its old first row when the cursor
        // comes back inside the rows it had: fine at that same row, but below
        // it the prompt would cover the redraw's last lines. Only then, step
        // past those rows. A shorter redraw ends above them, and the prompt
        // follows it there, right under the conversation.
        if let (Some(start), Ok((_, mut row))) = (prompt_start, position()) {
            while row > start && row <= start.saturating_add(prompt_rows) && row < rows.saturating_sub(1) {
                write!(stdout, "\r\n")?;
                row += 1;
            }
            stdout.flush()?;
        }
        Ok(detail)
    }
}

/// The page drawn again from the record, with the turn's pane under it:
/// Ctrl+O during a turn, a resize, a tab or an editor giving the screen
/// back. Leaves the display drawing at `columns` × `rows`.
fn redraw_active_locked(stdout: &mut impl Write, state: &mut DisplayState, columns: u16, rows: u16) -> io::Result<()> {
    state.columns = columns;
    state.rows = rows;
    state.suspended = false;
    state.shrunk = false;
    state.waiting_resize = None;
    state.held.clear();
    let output_rows = rows - pane_rows(rows);
    state.renderer = draw_page_locked(stdout, state, columns, output_rows, false)?;
    state.transient = None;
    // s0ma's output goes on from where the conversation ends.
    queue!(stdout, SavePosition)?;
    render_input_locked(stdout, state)?;
    stdout.flush()
}

/// Between turns: the page drawn again from the record, the cursor left
/// where the conversation ends, for the prompt.
fn redraw_idle_locked(stdout: &mut impl Write, state: &mut DisplayState, columns: u16, rows: u16) -> io::Result<()> {
    let (detail, chars_per_token) = (state.renderer.detail, state.renderer.chars_per_token);
    // Lines held while the prompt was up are in the record, so drawn too.
    state.held.clear();
    draw_page_locked(stdout, state, columns, rows, true)?;
    state.renderer = Renderer::new(detail, chars_per_token);
    state.transient = None;
    state.columns = columns;
    state.rows = rows;
    Ok(())
}

/// The page from the top: cleared, the tab bar and its rule, and as much of
/// the end of the conversation as fits above row `bottom` (the rows from it
/// down are the pane's), which with the rows above it becomes the scroll
/// region. The cursor is left where the conversation ends. `finish`: settle
/// a line still open (between turns). Gives the renderer that drew it.
fn draw_page_locked(stdout: &mut impl Write, state: &DisplayState, columns: u16, bottom: u16, finish: bool) -> io::Result<Renderer> {
    let (detail, chars_per_token) = (state.renderer.detail, state.renderer.chars_per_token);
    let (renderer, text) = redraw_conversation(&state.record, detail, chars_per_token, columns as usize, finish);
    let shown = page::rows(&text, columns as usize);
    let room = bottom.saturating_sub(page::top_rows()).max(1) as usize;
    let tail = &shown.rows[shown.rows.len().saturating_sub(room)..];
    write!(stdout, "\x1b[r\x1b[H\x1b[2J")?;
    if page::on() {
        write!(stdout, "{}", page::top(&state.bar, columns))?;
    }
    // The region first (setting it moves the cursor), then the rows, which
    // fit in it: the cursor ends after the last as the terminal left it,
    // waiting to wrap if the row is full.
    write!(stdout, "{}", page::region(bottom))?;
    queue!(stdout, MoveTo(0, page::top_rows()))?;
    write!(stdout, "{}", tail.join("\r\n"))?;
    Ok(renderer)
}

/// The session's record drawn the `detail` way, from the top. `finish`:
/// settle any line still open (at the idle prompt).
fn redraw_conversation(
    record: &TranscriptRecord,
    detail: Detail,
    chars_per_token: f64,
    columns: usize,
    finish: bool,
) -> (Renderer, String) {
    let mut renderer = Renderer::new(detail, chars_per_token);
    let mut text = String::new();
    let (entries, partial) = record.replay(detail);
    let mut out = Out { text: &mut text, columns, wrapping: true };
    if partial {
        let note = truncate_to_width(EARLIER_NOT_SHOWN, columns.saturating_sub(1));
        renderer.push_text(&mut out, &format!("{STATUS_STYLE}{note}{RESET}\r\n\r\n"));
    }
    for recorded in entries {
        renderer.apply(&mut out, recorded.first, recorded.last, &recorded.entry);
    }
    if finish {
        renderer.finish(&mut out);
    } else {
        renderer.refresh(&mut out, Instant::now(), true);
    }
    (renderer, text)
}

/// Where the cursor is now, if the terminal says.
pub fn cursor_row() -> Option<u16> {
    position().ok().map(|(_, row)| row)
}

/// Output was written where the idle prompt was erased (its first row
/// `start`, `prompt_rows` rows): reedline would draw the prompt back at
/// `start` over it, since the cursor is still within those rows, so step
/// past them first. Nothing written, and it is drawn where it was.
pub fn clear_of_prompt(start: u16, prompt_rows: u16) -> io::Result<()> {
    let (Ok((_, rows)), Ok((_, mut row))) = (terminal::size(), position()) else {
        return Ok(());
    };
    if row == start {
        return Ok(());
    }
    let mut stdout = io::stdout().lock();
    while row > start && row <= start.saturating_add(prompt_rows) && row < rows.saturating_sub(1) {
        write!(stdout, "\r\n")?;
        row += 1;
    }
    stdout.flush()
}

/// The draft's shape at the idle prompt: how many rows above the cursor it starts, and how
/// many rows it takes, for `you>  ` and its six-column indent on later lines,
/// wrapped at `columns`.
pub fn prompt_geometry(draft: &str, cursor: usize, columns: u16) -> (u16, u16) {
    let columns = columns.max(1) as usize;
    let mut cursor = cursor.min(draft.len());
    while !draft.is_char_boundary(cursor) {
        cursor -= 1;
    }
    let prefix = |_: usize| OPERATOR_PREFIX;
    let cursor_line = draft[..cursor].matches('\n').count();
    let line_start = draft[..cursor].rfind('\n').map_or(0, |at| at + 1);
    // The cursor's row within its line: a line filled exactly to the edge
    // puts it at the start of the next row.
    let (row, column) = wrap_end(prefix(cursor_line), &draft[line_start..cursor], columns);
    let cursor_row = if column >= columns { row + 1 } else { row };
    let (mut above, mut total) = (0, 0);
    for (index, line) in draft.split('\n').enumerate() {
        let mut rows = wrap_end(prefix(index), line, columns).0 + 1;
        if index == cursor_line {
            rows = rows.max(cursor_row + 1);
            above = total + cursor_row;
        }
        total += rows;
    }
    (above.min(u16::MAX as usize) as u16, total.min(u16::MAX as usize) as u16)
}

/// Where text that starts at column `start` ends on a terminal `columns`
/// wide, as (row, column): a wide character that doesn't fit moves whole to
/// the next row, as terminals do. A column of `columns` is a pending wrap.
fn wrap_end(start: usize, text: &str, columns: usize) -> (usize, usize) {
    let (mut row, mut column) = (0, start);
    for cluster in clusters(text) {
        let width = UnicodeWidthStr::width(cluster);
        if width == 0 {
            continue;
        }
        if column + width > columns {
            row += 1;
            column = 0;
        }
        column += width;
    }
    (row, column)
}

/// `text` cut into what a terminal draws as one character, near enough:
/// combining marks, variation selectors, skin tones and whatever a zero-width
/// joiner joins stay with the character before them, and regional indicators
/// pair into flags. (Reedline measures graphemes; this matches it for
/// everything a prompt is likely to hold.)
fn clusters(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        let mut characters = rest.char_indices();
        let (_, first) = characters.next()?;
        let regional = |c: char| ('\u{1F1E6}'..='\u{1F1FF}').contains(&c);
        let (mut previous, mut regional_pair) = (first, regional(first));
        let mut end = rest.len();
        for (at, character) in characters {
            // A flag is two indicators: only the second joins the first.
            let flag = std::mem::take(&mut regional_pair) && regional(character);
            let joined = flag
                || UnicodeWidthChar::width(character) == Some(0)
                || previous == '\u{200D}'
                || ('\u{1F3FB}'..='\u{1F3FF}').contains(&character);
            if !joined {
                end = at;
                break;
            }
            previous = character;
        }
        let (cluster, after) = rest.split_at(end);
        rest = after;
        Some(cluster)
    })
}

/// Erase the idle prompt reedline left on screen when something other than
/// the operator ended it (a background event): the draft moves into the
/// steering pane, or the prompt is drawn again on the same row, rather than
/// staying behind as a frozen copy in the transcript.
pub fn erase_idle_prompt(draft: &str, cursor: usize) -> io::Result<()> {
    let (columns, _) = terminal::size()?;
    let (_, row) = position()?;
    let (above, _) = prompt_geometry(draft, cursor, columns);
    let mut stdout = io::stdout().lock();
    queue!(stdout, MoveTo(0, row.saturating_sub(above).max(page::top_rows())), Clear(ClearType::FromCursorDown))?;
    stdout.flush()
}

/// Reserve the bottom pane and place s0ma's output cursor.
///
/// `keep_column`: continue at (`column`, `row`) if that is inside the output
/// area (used after an editor returns the screen); otherwise start a fresh
/// line, scrolling the transcript up if the pane would cover it.
/// Returns whether output continues where it was (`keep_column` honoured).
fn establish_layout_locked(
    stdout: &mut impl Write,
    state: &mut DisplayState,
    columns: u16,
    rows: u16,
    column: u16,
    row: u16,
    keep_column: bool,
) -> io::Result<bool> {
    let output_rows = rows - pane_rows(rows);
    state.columns = columns;
    state.rows = rows;

    let kept = keep_column && row < output_rows;
    let (model_row, model_column, scroll_rows) = if kept {
        (row, column.min(columns.saturating_sub(1)), 0)
    } else {
        let (model_row, scroll_rows) = reserve_layout(row, column, rows);
        (model_row, 0, scroll_rows)
    };

    // The conversation's rows down to the bottom scroll up to make room,
    // under the page's top rows.
    write!(stdout, "{}", page::region(rows))?;
    if scroll_rows > 0 {
        queue!(stdout, MoveTo(0, rows - 1))?;
        for _ in 0..scroll_rows {
            write!(stdout, "\r\n")?;
        }
    }

    // DECSTBM uses 1-based row numbers. Reserve a small editor pane
    // rather than forcing steering through one physical line.
    write!(stdout, "{}", page::region(output_rows))?;

    // Establish and save s0ma's logical output position.
    queue!(stdout, MoveTo(model_column, model_row), SavePosition)?;
    state.renderer.wrap.col = model_column as usize;

    // Draw the pane and deliberately leave the real cursor there.
    render_input_locked(stdout, state)?;
    Ok(kept)
}

/// The pane was laid out for one size and the terminal has another: hold
/// output (`suspended`) until `resize` lays it out again. Drawing meanwhile
/// would use rows the terminal has moved or taken away: the pane drawn over
/// the transcript, lines written onto other lines.
fn notice_resize(state: &mut DisplayState) {
    if !state.active || state.suspended || state.covered.is_some() || state.waiting_resize.is_some() {
        return;
    }
    if let Some(size) = current_size()
        && size != (state.columns, state.rows)
    {
        state.suspended = true;
        state.waiting_resize = Some(Instant::now());
    }
}

/// The terminal's size, for `notice_resize`. A test sets its own (there is
/// no terminal there), or none, and nothing is noticed.
#[cfg(not(test))]
fn current_size() -> Option<(u16, u16)> {
    terminal::size().ok()
}
#[cfg(test)]
thread_local! {
    static TEST_SIZE: std::cell::Cell<Option<(u16, u16)>> = const { std::cell::Cell::new(None) };
}
#[cfg(test)]
fn current_size() -> Option<(u16, u16)> {
    TEST_SIZE.with(std::cell::Cell::get)
}

/// The terminal's size once it has held still for `RESIZE_STILL` (at most
/// `RESIZE_WAIT`): a window edge being dragged changes it many times a second.
pub fn settled_size() -> io::Result<(u16, u16)> {
    let started = Instant::now();
    let mut size = terminal::size()?;
    let mut since = Instant::now();
    while started.elapsed() < RESIZE_WAIT {
        std::thread::sleep(Duration::from_millis(20));
        let now = terminal::size()?;
        if now != size {
            (size, since) = (now, Instant::now());
        } else if since.elapsed() >= RESIZE_STILL {
            break;
        }
    }
    Ok(size)
}

/// Short: the keys wait while it settles. A drag's steps come every few
/// tens of milliseconds, so 120 ms still tells its end.
const RESIZE_STILL: Duration = Duration::from_millis(120);
const RESIZE_WAIT: Duration = Duration::from_millis(700);

/// A resize event for the size the pane was laid out for this recently is
/// one that arrived while that was being done.
const STALE_RESIZE: Duration = Duration::from_secs(2);

/// How long held output waits for a resize's event before `tick` lays the
/// pane out itself.
const RESIZE_FALLBACK: Duration = Duration::from_millis(400);

/// Repaint just the progress line, then put the cursor back on the input.
fn redraw_status(state: &mut DisplayState) -> io::Result<()> {
    notice_resize(state);
    if !state.active || state.suspended || state.covered.is_some() {
        return Ok(());
    }
    // One write, the cursor hidden while the status line is drawn.
    let mut frame: Vec<u8> = Vec::with_capacity(512);
    queue!(frame, BeginSynchronizedUpdate, Hide)?;
    draw_status_locked(&mut frame, state)?;
    place_input_cursor_locked(&mut frame, state)?;
    queue!(frame, Show, EndSynchronizedUpdate)?;
    state.frame_drawn = state.started.elapsed().as_millis() / SPINNER_FRAME.as_millis();
    let mut stdout = io::stdout().lock();
    stdout.write_all(&frame)?;
    stdout.flush()
}

impl Default for ActiveDisplay {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// STREAM PRINTER
// ============================================================

pub struct StreamPrinter {
    display: ActiveDisplay,
    markdown: MarkdownStream,
    operator_revision: u64,
}

/// Feeds a model pass's stream to the display: styles reply text as markdown
/// and keeps the progress line's activity current. How reasoning, replies and
/// tool lines are drawn (collapsed or expanded) is the display's business.
impl StreamPrinter {
    pub fn new(display: ActiveDisplay) -> Self {
        let operator_revision = display.operator_revision().unwrap_or(0);
        Self {
            display,
            markdown: MarkdownStream::new(),
            operator_revision,
        }
    }

    pub fn event(&mut self, event: StreamEvent) -> Result<(), Box<dyn std::error::Error>> {
        // Nothing has arrived (llm.rs sends it while a call waits); nothing to show.
        if matches!(event, StreamEvent::Waiting) {
            return Ok(());
        }
        // The operator wrote into the transcript: reply text held back goes
        // out first, ahead of what follows.
        let revision = self.display.operator_revision()?;
        if revision != self.operator_revision {
            self.flush_markdown()?;
            self.operator_revision = revision;
        }
        match event {
            // Some servers send an empty reasoning field with every reply chunk.
            StreamEvent::Reasoning(text) if text.is_empty() => {}
            StreamEvent::Reasoning(text) => {
                self.display.generated(text.chars().count())?;
                self.display.set_activity(Activity::Thinking)?;
                self.display.reasoning(&safe_text(&text))?;
            }

            // A retry or resample of the same call streams again: measure this
            // generation on its own rather than continuing the last one.
            StreamEvent::Started => self.display.generation_started()?,

            StreamEvent::Waiting => {}

            StreamEvent::ToolArguments(text) => {
                self.display.generated(text.chars().count())?;
            }

            StreamEvent::Usage { completion_tokens, prompt_tokens, cached_tokens, last, .. } => {
                self.display.usage(completion_tokens, prompt_tokens, cached_tokens, last)?;
            }

            StreamEvent::Content(text) => {
                self.display.generated(text.chars().count())?;
                self.display.set_activity(Activity::Writing)?;
                // Tables are laid out for the width they will be shown at,
                // read each time so a resize applies from the next table.
                self.markdown.set_width(self.display.content_columns());
                // Sanitize first; the markdown renderer only ever adds its
                // own SGR styling on top of already-safe text.
                let text = self.markdown.push(&safe_text(&text));
                self.display.answer(&text)?;
            }
        }

        Ok(())
    }

    /// Write a harness status message through the active display so it
    /// appears in the scrolling transcript instead of on the reserved
    /// operator-input row.
    pub fn status(&mut self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.flush_markdown()?;
        if let Some(call) = message.strip_prefix("[tool] running: ") {
            let name = call.split(',').next().unwrap_or(call).trim();
            self.display.end_request()?;
            self.display.set_activity(Activity::Tool(safe_text(name)))?;
        }
        // A tool boundary starts a new model inference pass; the display
        // gives its reasoning and answer fresh headers.
        self.display.status(&safe_text(message))?;
        Ok(())
    }

    /// A model request is about to be sent.
    pub fn requesting(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.display.request_sent()?;
        self.display.set_activity(Activity::Waiting)?;
        Ok(())
    }

    pub fn finish(&mut self) {
        let _ = self.flush_markdown();
        let _ = self.display.write_stream("\r\n\r\n");
    }

    pub fn finish_interrupted(&mut self) {
        let _ = self.flush_markdown();
        let _ = self.display.write_stream("\r\n");
    }

    /// Emit any reply text the markdown renderer is still holding back (an
    /// unfinished line marker, `**`, link or table) and reset it.
    fn flush_markdown(&mut self) -> io::Result<()> {
        // A table still held back is laid out now.
        self.markdown.set_width(self.display.content_columns());
        let pending = self.markdown.finish();
        if pending.is_empty() {
            Ok(())
        } else {
            self.display.answer_tail(&pending)
        }
    }
}

// ============================================================
// INTERNAL TERMINAL HELPERS
// ============================================================

/// Draw the progress line and live operator input in the reserved pane.
///
/// Crucially, this function DOES NOT use SavePosition / RestorePosition.
///
/// The terminal's saved position belongs exclusively to s0ma's logical
/// output cursor. The physical cursor is intentionally left at the
/// operator's cursor so typing feels normal.
fn render_input_locked(stdout: &mut impl Write, state: &DisplayState) -> io::Result<()> {
    if !state.active || state.suspended {
        return Ok(());
    }

    let (first_row, status_rows, input_rows) = pane_geometry(state.rows);
    if status_rows > 0 {
        draw_status_locked(stdout, state)?;
    }
    let first_input_row = first_row + status_rows;
    let layout = input_layout(&state.input, state.input_cursor, state.columns, input_rows);

    for offset in 0..input_rows {
        queue!(
            stdout,
            MoveTo(0, first_input_row + offset),
            Clear(ClearType::CurrentLine)
        )?;

        let index = layout.first + offset as usize;
        let Some(line) = layout.lines.get(index) else {
            continue;
        };

        if index == 0 {
            write!(
                stdout,
                "{OPERATOR_STYLE}you>{RESET}  {OPERATOR_TEXT_STYLE}{line}{RESET}"
            )?;
        } else {
            // Later lines under the first one's text, as the transcript shows them.
            write!(stdout, "{}{OPERATOR_TEXT_STYLE}{line}{RESET}", " ".repeat(OPERATOR_PREFIX))?;
        }
    }

    place_input_cursor_locked(stdout, state)
}

fn place_input_cursor_locked(stdout: &mut impl Write, state: &DisplayState) -> io::Result<()> {
    let (first_row, status_rows, input_rows) = pane_geometry(state.rows);
    let layout = input_layout(&state.input, state.input_cursor, state.columns, input_rows);
    let (cursor_line, cursor_cells) = layout.cursor;
    let prefix = OPERATOR_PREFIX;
    let cursor_row = first_row + status_rows + (cursor_line - layout.first) as u16;
    let cursor_column = (prefix + cursor_cells) as u16;

    queue!(
        stdout,
        MoveTo(
            cursor_column.min(state.columns.saturating_sub(1)),
            cursor_row.min(state.rows.saturating_sub(1)),
        ),
    )
}

/// The tab bar atop the page (`tabs::main_bar`), as last refreshed, and its
/// rule. Moves the cursor.
fn draw_top_locked(stdout: &mut impl Write, state: &DisplayState) -> io::Result<()> {
    if page::on() {
        write!(stdout, "{}", page::top(&state.bar, state.columns))?;
    }
    Ok(())
}

/// `⠹ running web_search · 1m 04s · 1 queued     prefill 3.1k tok/s · decode 57 tok/s`
fn draw_status_locked(stdout: &mut impl Write, state: &DisplayState) -> io::Result<()> {
    let (first_row, status_rows, _) = pane_geometry(state.rows);
    if status_rows == 0 {
        return Ok(());
    }
    let elapsed = state.started.elapsed();
    let frame = (elapsed.as_millis() / SPINNER_FRAME.as_millis()) as usize;
    let spinner = SPINNER[frame % SPINNER.len()];
    let width = state.columns.saturating_sub(1) as usize;

    let mut left = format!("{spinner} {} · {}", state.activity.label(), format_elapsed(elapsed));
    let queued = state.queued.load(Ordering::SeqCst);
    if queued > 0 {
        left.push_str(&format!(" · {queued} queued"));
    }
    let left = truncate_to_width(&left, width);
    let used = UnicodeWidthStr::width(left.as_str());
    let room = width.saturating_sub(used + 3);
    let right = match &state.hint {
        // A partial completion list is still useful.
        Some(hint) => truncate_to_width(hint, room),
        // Each rate is shown whole or not at all.
        None => fit_segments(&status_segments(state), room),
    };

    queue!(stdout, MoveTo(0, first_row), Clear(ClearType::CurrentLine))?;
    let mut chars = left.chars();
    let spinner_char: String = chars.next().into_iter().collect();
    let rest: String = chars.collect();
    write!(stdout, "{CHEVRON_STYLE}{spinner_char}{RESET}{STATUS_STYLE}{rest}{RESET}")?;
    if !right.is_empty() {
        let gap = width - used - UnicodeWidthStr::width(right.as_str());
        let style = if state.hint.is_some() { TEXT_STYLE } else { STATUS_STYLE };
        write!(stdout, "{}{style}{right}{RESET}", " ".repeat(gap))?;
    }
    Ok(())
}

/// The bottom-right corner's pieces, in display order, each with a rank:
/// when they don't all fit, the highest ranks (least needed) go first. The
/// keys used to be listed here too; `/help` has them now.
fn status_segments(state: &DisplayState) -> Vec<(String, u8)> {
    rate_segments(state.meter.prefill_label(), state.meter.label(Instant::now()))
}

/// Prefill then decode, the order they happen in. Decode is the one that moves
/// while a reply streams, so on a narrow line prefill gives way first.
fn rate_segments(prefill: Option<String>, decode: Option<String>) -> Vec<(String, u8)> {
    let mut segments = Vec::new();
    if let Some(prefill) = prefill {
        segments.push((format!("prefill {prefill}"), 2));
    }
    if let Some(decode) = decode {
        segments.push((format!("decode {decode}"), 1));
    }
    segments
}

/// Join as many `segments` as fit in `room` columns with ` · `: the most
/// needed (lowest rank, then leftmost) are taken first, each only if it still
/// fits, and the ones taken are shown in their original order.
fn fit_segments(segments: &[(String, u8)], room: usize) -> String {
    let mut order: Vec<usize> = (0..segments.len()).collect();
    order.sort_by_key(|index| (segments[*index].1, *index));
    let mut keep = vec![false; segments.len()];
    let mut used = 0;
    for index in order {
        let width = UnicodeWidthStr::width(segments[index].0.as_str());
        let separator = if used == 0 { 0 } else { 3 };
        if used + separator + width <= room {
            keep[index] = true;
            used += separator + width;
        }
    }
    segments
        .iter()
        .zip(&keep)
        .filter(|(_, kept)| **kept)
        .map(|((text, _), _)| text.as_str())
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Printed where operator messages sent mid-turn reach s0ma.
pub fn delivered_note(count: usize) -> String {
    let what = if count == 1 { "your message".to_string() } else { format!("your {count} messages") };
    format!("s0ma has {what}")
}

/// The status line for a message queued for s0ma's next safe point, with
/// the start of it (the line is cut to the terminal's width when drawn).
fn queued_note(input: &str) -> String {
    let preview = safe_text(input).split_whitespace().collect::<Vec<_>>().join(" ");
    let preview = truncate_to_width(&preview, 40);
    format!("queued for after the current step: \u{201c}{preview}\u{201d}")
}

/// A status line: `[*] brief` in the gutter collapsed; `full` dim, expanded.
/// Both are already safe.
fn status_notice(brief: &str, full: &str) -> Entry {
    Entry::Notice { brief: brief.to_string(), full: format!("{STATUS_STYLE}{full}{RESET}"), mark: Mark::Work }
}

/// `you>  …` into the transcript, as the operator's own line.
fn emit_operator_locked(writer: &mut impl Write, state: &mut DisplayState, input: &str) -> io::Result<()> {
    state.operator_revision = state.operator_revision.wrapping_add(1);
    emit_locked(writer, state, Entry::Operator(safe_text(input)))
}

/// The dim line under an operator message sent mid-turn.
fn operator_note_text(note: OperatorNote, activity: &Activity) -> Option<String> {
    match note {
        OperatorNote::Plain => None,
        // A running tool is never cut short; /kill is how to stop one.
        OperatorNote::SendNow => Some(match activity {
            Activity::Tool(name) => format!("sending now · s0ma reads it when {name} returns (/kill stops it)"),
            _ => "sending now".into(),
        }),
    }
}

pub(crate) fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    }
}

pub(crate) fn truncate_to_width(text: &str, width: usize) -> String {
    if UnicodeWidthStr::width(text) <= width {
        return text.to_string();
    }
    if width == 0 {
        return String::new();
    }
    // Cut, leaving room for the ellipsis that marks the cut.
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let w = UnicodeWidthChar::width(c).unwrap_or(0);
        if used + w + 1 > width {
            break;
        }
        used += w;
        out.push(c);
    }
    out.push('…');
    out
}

/// Rows reserved at the bottom: the progress line (if room) plus input.
fn pane_rows(rows: u16) -> u16 {
    // DECSTBM requires at least two output rows, under the page's top rows.
    (ACTIVE_INPUT_ROWS + STATUS_ROWS)
        .min(rows.saturating_sub(page::top_rows() + 2))
        .max(1)
}

/// (first pane row, status rows, input rows).
fn pane_geometry(rows: u16) -> (u16, u16, u16) {
    let pane = pane_rows(rows);
    let status = if pane >= 3 { STATUS_ROWS } else { 0 };
    (rows - pane, status, pane - status)
}

/// The operator draft wrapped into pane lines, the window of them that
/// fits the pane, and where the cursor lands.
struct InputLayout {
    lines: Vec<String>,
    /// Index of the first line shown.
    first: usize,
    /// (line index, display column within the line)
    cursor: (usize, usize),
}

fn input_layout(input: &str, cursor: usize, columns: u16, pane_rows: u16) -> InputLayout {
    // After the prefix, leaving the final column unused so repainting
    // cannot trigger autowrap.
    let width = (columns as usize).saturating_sub(OPERATOR_PREFIX + 1).max(1);
    let mut lines = vec![String::new()];
    let mut cells = 0;
    let mut cursor_at = None;

    for (index, character) in input.char_indices() {
        if character == '\n' {
            if index == cursor {
                cursor_at = Some((lines.len() - 1, cells));
            }
            lines.push(String::new());
            cells = 0;
            continue;
        }
        if character.is_control() {
            // The editor sanitizes input; never draw control characters.
            if index == cursor {
                cursor_at = Some((lines.len() - 1, cells));
            }
            continue;
        }
        let n = UnicodeWidthChar::width(character).unwrap_or(0);
        if cells + n > width && cells > 0 {
            lines.push(String::new());
            cells = 0;
        }
        if index == cursor {
            cursor_at = Some((lines.len() - 1, cells));
        }
        lines.last_mut().unwrap().push(character);
        cells += n;
    }

    let cursor = cursor_at.unwrap_or((lines.len() - 1, cells));

    // Scroll the pane just enough to keep the cursor's line visible.
    let rows = pane_rows.max(1) as usize;
    let first = cursor.0.saturating_sub(rows - 1).min(lines.len().saturating_sub(rows));

    InputLayout {
        lines,
        first,
        cursor,
    }
}

// ============================================================
// WORD WRAP
// ============================================================

/// Wraps transcript output at word boundaries.
///
/// The terminal would otherwise break lines at the last column, splitting
/// words. Output is streamed, so the word currently being written is held
/// back until a space or newline shows it is complete; if it would not fit
/// on the current line, a line break goes in front of it. Spaces are held
/// back too, so a wrapped line never ends or starts with the space that was
/// replaced by the break. Words wider than the terminal are let through as
/// soon as they fill a line.
#[derive(Default)]
pub(crate) struct WordWrap {
    /// Display column of the output cursor.
    pub(crate) col: usize,
    /// Spaces seen since the last word, not yet written.
    gap: usize,
    word: String,
    word_width: usize,
    /// A hanging indent: blank columns in front of every line, put there
    /// when the line's first visible text is (so a reply's last line break
    /// leaves nothing behind it).
    pub(crate) indent: usize,
}

impl WordWrap {
    /// A word (or styling) is held back, not yet written.
    pub(crate) fn holding(&self) -> bool {
        !self.word.is_empty()
    }

    /// Where a line's text starts: after the indent, if the line has room
    /// for anything past it.
    fn line_start(&self, columns: usize) -> usize {
        self.indent.min(columns.saturating_sub(1))
    }

    pub(crate) fn push(&mut self, text: &str, columns: usize, out: &mut String) {
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                // Our own SGR styling: zero width, carried with the word.
                '\x1b' => {
                    self.word.push(c);
                    if chars.peek() == Some(&'[') {
                        self.word.push(chars.next().unwrap());
                        for c in chars.by_ref() {
                            self.word.push(c);
                            if ('\x40'..='\x7e').contains(&c) {
                                break;
                            }
                        }
                    }
                }
                '\n' | '\r' => {
                    self.place(columns, out);
                    // Trailing spaces before a line break are invisible.
                    self.gap = 0;
                    out.push(c);
                    self.col = 0;
                }
                ' ' => {
                    self.place_held(columns, out, true);
                    self.gap += 1;
                }
                _ => {
                    self.word.push(c);
                    self.word_width += UnicodeWidthChar::width(c).unwrap_or(0);
                    if self.word_width >= columns.saturating_sub(self.line_start(columns)) {
                        self.place(columns, out);
                    }
                }
            }
        }
    }

    /// Write the held-back word (and the spaces before it).
    fn place(&mut self, columns: usize, out: &mut String) {
        self.place_held(columns, out, false);
    }

    /// `place`, except that with `hold`, styling alone at the start of an
    /// indented line waits for the text it styles, so the indent in front of
    /// that text is never styled (an underlined heading would underline it).
    fn place_held(&mut self, columns: usize, out: &mut String, hold: bool) {
        let start = self.line_start(columns);
        if self.word_width == 0 {
            if hold && self.col == 0 && start > 0 {
                return;
            }
            // Only styling so far; the spaces wait for real text.
            out.push_str(&self.word);
            self.word.clear();
            return;
        }
        if self.col > start && self.col + self.gap + self.word_width > columns {
            out.push('\n');
            self.col = 0;
            self.gap = 0;
        }
        if self.col == 0 && start > 0 {
            out.extend(std::iter::repeat_n(' ', start));
            self.col = start;
        }
        out.extend(std::iter::repeat_n(' ', self.gap));
        self.col += self.gap;
        self.gap = 0;
        out.push_str(&self.word);
        self.word.clear();
        self.col += self.word_width;
        self.word_width = 0;
        if self.col > columns {
            // Only reachable with wide characters at the very edge.
            self.col %= columns;
        }
    }

    /// The word held back, laid out as if it were complete; spaces after it
    /// stay held for whatever continues the line.
    pub(crate) fn take_word(&mut self, columns: usize) -> String {
        let mut out = String::new();
        self.place(columns.max(1), &mut out);
        out
    }

    /// Everything still held back, laid out as if the word were complete.
    pub(crate) fn take_pending(&mut self, columns: usize) -> String {
        let mut out = String::new();
        self.place(columns.max(1), &mut out);
        self.gap = 0;
        out
    }
}

/// Never allow model/tool/pasted text to issue cursor or erase commands.
/// Styles generated by this module are added only AFTER this sanitization.
pub fn safe_text(text: &str) -> String {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\n' => out.push('\n'),
            '\t' => out.push_str("    "),
            '\x1b' => out.push('␛'),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

fn reserve_layout(row: u16, column: u16, rows: u16) -> (u16, u16) {
    let next_row = row.saturating_add(u16::from(column > 0));
    let last_output_row = rows.saturating_sub(pane_rows(rows) + 1);
    let scroll_rows = next_row.saturating_sub(last_output_row);
    (next_row.saturating_sub(scroll_rows), scroll_rows)
}

/// Draw `entry` the way the current mode shows it, and keep it in the turn's
/// record for Ctrl+O.
fn emit_locked(writer: &mut impl Write, state: &mut DisplayState, entry: Entry) -> io::Result<()> {
    let now = Instant::now();
    notice_resize(state);
    if (state.active && state.suspended) || state.covered.is_some() || (state.prompt_up && !state.active) {
        // An editor or a tab has the screen, or the idle prompt does: kept
        // as it is, laid out on resume (or drawn again from the record when
        // a tab closes), or above the prompt (`flush_idle`).
        state.record.push(now, entry.clone());
        match (state.held.last_mut(), &entry) {
            (Some((_, last, Entry::Reasoning(held))), Entry::Reasoning(more))
            | (Some((_, last, Entry::Answer(held))), Entry::Answer(more)) => {
                held.push_str(more);
                *last = now;
            }
            _ => state.held.push((now, now, entry)),
        }
        return Ok(());
    }
    let columns = state.columns as usize;
    let mut text = String::new();
    {
        // Outside the active display, text passes through unwrapped, as it
        // always has: the output column is unknown there.
        let mut out = Out { text: &mut text, columns, wrapping: state.active };
        state.renderer.apply(&mut out, now, now, &entry);
        state.renderer.refresh(&mut out, now, false);
    }
    state.record.push(now, entry);
    write_rendered_locked(writer, state, &text)
}

/// The transient line as drawn: dim, in the gutter like a work line.
fn transient_line(state: &DisplayState, text: &str) -> String {
    format!("\r{}{WORK_STYLE}{text}{RESET}", " ".repeat(gutter_width(state.columns as usize)))
}

/// Lay out what arrived while an editor had the screen, and bring a line that
/// kept counting meanwhile up to date.
fn render_held_locked(state: &mut DisplayState) -> String {
    let columns = state.columns as usize;
    let mut text = String::new();
    let held = std::mem::take(&mut state.held);
    let mut out = Out { text: &mut text, columns, wrapping: true };
    for (first, last, entry) in &held {
        state.renderer.apply(&mut out, *first, *last, entry);
    }
    state.renderer.refresh(&mut out, Instant::now(), true);
    text
}

/// Write rendered transcript text at s0ma's output cursor, then redraw the
/// pane and leave the cursor on the operator's input. Outside the active
/// display it is written as is. Never called while an editor has the screen:
/// output is held as entries then (`emit_locked`).
fn write_rendered_locked(writer: &mut impl Write, state: &mut DisplayState, text: &str) -> io::Result<()> {
    if text.is_empty() || (state.active && state.suspended) {
        // Everything is held back mid-word: nothing to draw yet.
        return Ok(());
    }
    if !state.active {
        // This path can also be used while raw mode is enabled after a tiny
        // resize, so CRLF normalization is required in both modes.
        let Some(transient) = state.transient.take() else {
            return write_normalized(writer, text);
        };
        // Over the transient line, which comes back below when the text
        // ends a line (otherwise it would be in the way, and goes).
        write!(writer, "\r\x1b[2K")?;
        write_normalized(writer, text)?;
        if text.ends_with('\n') {
            write!(writer, "{}", transient_line(state, &transient))?;
            state.transient = Some(transient);
        }
        return Ok(());
    }
    // The text at the conversation's end, then the pane drawn again: one
    // write, the cursor hidden meanwhile and the whole of it one
    // synchronized update, so the cursor never shows passing through the
    // conversation and the pane is never seen half drawn.
    let mut frame: Vec<u8> = Vec::with_capacity(text.len() + 1024);
    queue!(frame, BeginSynchronizedUpdate, Hide, RestorePosition)?;
    write_normalized(&mut frame, text)?;
    queue!(frame, SavePosition)?;
    render_input_locked(&mut frame, state)?;
    queue!(frame, Show, EndSynchronizedUpdate)?;
    writer.write_all(&frame)
}

#[cfg(test)]
mod scheduled_io_tests {
    use super::*;

    fn wrap(chunks: &[&str], columns: usize) -> String {
        let mut wrap = WordWrap::default();
        let mut out = String::new();
        for chunk in chunks {
            wrap.push(chunk, columns, &mut out);
        }
        out.push_str(&wrap.take_pending(columns));
        out
    }

    #[test]
    fn words_move_to_the_next_line_instead_of_splitting() {
        assert_eq!(wrap(&["the quick brown fox"], 10), "the quick\nbrown fox");
        // Streaming in odd pieces gives the same breaks.
        assert_eq!(wrap(&["the qu", "ick br", "own", " fox"], 10), "the quick\nbrown fox");
        // A word that exactly fills the line, then a space: one break, no blank line.
        assert_eq!(wrap(&["abcde fghij klm"], 5), "abcde\nfghij\nklm");
        // Explicit newlines reset the column.
        assert_eq!(wrap(&["aaaa\nbbbb cc"], 8), "aaaa\nbbbb cc");
        // Indentation (code blocks, table padding) is kept.
        assert_eq!(wrap(&["│     x  │\n    y"], 40), "│     x  │\n    y");
    }

    fn wrap_indented(chunks: &[&str], columns: usize, indent: usize) -> String {
        let mut wrap = WordWrap { indent, ..WordWrap::default() };
        let mut out = String::new();
        for chunk in chunks {
            wrap.push(chunk, columns, &mut out);
        }
        out.push_str(&wrap.take_pending(columns));
        out
    }

    #[test]
    fn a_hanging_indent_starts_every_line_that_has_text() {
        assert_eq!(wrap_indented(&["one two three four five\nsix"], 14, 4), "    one two\n    three four\n    five\n    six");
        // Streaming in odd pieces gives the same lines.
        assert_eq!(wrap_indented(&["one tw", "o thr", "ee four five\n", "six"], 14, 4), "    one two\n    three four\n    five\n    six");
        // A blank line stays blank, and a final line break leaves nothing after it.
        assert_eq!(wrap_indented(&["a\n\nb\n"], 10, 2), "  a\n\n  b\n");
        // Indentation of the text itself (code) comes after the gutter.
        assert_eq!(wrap_indented(&["fn x\n    y"], 20, 2), "  fn x\n      y");
        // Styling at the start of a line waits for its text: the gutter is never underlined.
        assert_eq!(wrap_indented(&["\x1b[4m  indented"], 20, 2), "    \x1b[4mindented");
        // A word too long for what is left of the line is broken to fit it.
        assert_eq!(wrap_indented(&[&"x".repeat(17)], 10, 2), "  xxxxxxxx\n  xxxxxxxx\n  x");
        // No room for a gutter: none.
        assert_eq!(wrap_indented(&["ab"], 1, 6), "a\nb");
    }

    #[test]
    fn wrapping_ignores_styling_and_counts_wide_chars() {
        let styled = "\x1b[0;1mbold\x1b[0m word";
        assert_eq!(wrap(&[styled], 9), styled);
        assert_eq!(wrap(&["你好 世界"], 6), "你好\n世界");
    }

    #[test]
    fn overlong_words_are_not_held_back_forever() {
        let mut w = WordWrap::default();
        let mut out = String::new();
        w.push("ab https://example.com/very/long", 10, &mut out);
        // The URL is released as soon as it fills a line, on a fresh line.
        assert!(out.starts_with("ab\nhttps://ex"));
    }

    #[test]
    fn entering_at_bottom_scrolls_instead_of_overwriting_the_prompt() {
        // A pane of six rows: the progress line, five of input.
        assert_eq!(reserve_layout(23, 0, 24), (17, 6));
        assert_eq!(reserve_layout(23, 7, 24), (17, 7));
        assert_eq!(reserve_layout(8, 0, 24), (8, 0));
        assert_eq!(reserve_layout(8, 7, 24), (9, 0));
    }

    #[test]
    fn content_cannot_move_the_cursor_or_erase_other_messages() {
        let text = safe_text("hello\x1b[2J\x1b8\rbye\x08\u{009b}2J");
        assert!(!text.contains('\x1b'));
        assert!(!text.contains('\r'));
        assert!(!text.contains('\x08'));
        assert!(!text.contains('\u{009b}'));
        assert!(text.contains("\nbye"));
    }

    fn active_state() -> DisplayState {
        DisplayState {
            active: true,
            columns: 40,
            rows: 24,
            input: String::new(),
            input_cursor: 0,
            operator_revision: 0,
            renderer: Renderer::new(Detail::Collapsed, DEFAULT_CHARS_PER_TOKEN),
            record: TranscriptRecord::default(),
            transient: None,
            meter: TokenMeter::default(),
            suspended: false,
            held: Vec::new(),
            activity: Activity::Waiting,
            started: Instant::now(),
            frame_drawn: 0,
            hint: None,
            queued: Arc::default(),
            covered: None,
            quiet_pass: false,
            shrunk: false,
            waiting_resize: None,
            laid_out_at: None,
            bar: String::new(),
            prompt_up: false,
        }
    }

    #[test]
    fn the_sleep_reflection_is_one_line_collapsed_and_output_keeps_a_transient_line() {
        let mut state = active_state();
        state.active = false;
        let mut bytes = Vec::new();
        emit_locked(&mut bytes, &mut state, Entry::Pass).unwrap();
        for _ in 0..50 {
            emit_locked(&mut bytes, &mut state, Entry::Reasoning("weighing what to keep ".into())).unwrap();
        }
        emit_locked(&mut bytes, &mut state, Entry::Note("[sleep] integration/reflection request: 3.20s".into())).unwrap();
        emit_locked(&mut bytes, &mut state, Entry::Pass).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(!text.contains("weighing") && !text.contains("[sleep]"), "{text:?}");
        let summary = text.rfind("[*] thought for").expect(&text);
        assert!(text[summary..].ends_with("\r\n"), "{text:?}");
        // Fifty chunks, one record entry.
        let (entries, _) = state.record.replay(Detail::Expanded);
        assert_eq!(entries.iter().filter(|recorded| matches!(recorded.entry, Entry::Reasoning(_))).count(), 1);

        // Expanded, the note after it still starts a line of its own.
        let mut state = active_state();
        state.active = false;
        state.renderer = Renderer::new(Detail::Expanded, DEFAULT_CHARS_PER_TOKEN);
        let mut bytes = Vec::new();
        emit_locked(&mut bytes, &mut state, Entry::Pass).unwrap();
        emit_locked(&mut bytes, &mut state, Entry::Reasoning("weighing what to keep".into())).unwrap();
        emit_locked(&mut bytes, &mut state, Entry::Note("[sleep] integration/reflection request: 3.20s".into())).unwrap();
        emit_locked(&mut bytes, &mut state, Entry::Pass).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("[Thinking]") && text.contains(&format!("keep{RESET}\r\n[sleep]")), "{text:?}");

        // A transient line comes back below a warning printed while it is up.
        state.transient = Some("[*] folding the context to make room…".into());
        let mut bytes = Vec::new();
        emit_locked(&mut bytes, &mut state, Entry::Line("[llm] repaired 1 argument".into())).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let warning = text.find("[llm] repaired").unwrap();
        assert!(text.starts_with("\r\x1b[2K") && text[warning..].contains("folding the context"), "{text:?}");
        assert!(text[warning..].contains("\r      \x1b[3;90m[*] folding"), "in the gutter: {text:?}");
        assert!(state.transient.is_some());
    }

    #[test]
    fn output_is_held_while_an_editor_has_the_screen() {
        let mut state = active_state();
        state.suspended = true;
        // The editor opened mid-line, after "two ".
        let mut bytes = Vec::new();
        let mut written = String::new();
        state.renderer.wrap.push("one two ", 40, &mut written);
        written.push_str(&state.renderer.wrap.take_word(40));
        assert_eq!(written, "one two", "the space after it stays held");
        let raw = |text: &str| Entry::Raw { text: text.into(), operator: false };
        emit_locked(&mut bytes, &mut state, raw("three four five six seven eight nine ")).unwrap();
        assert!(bytes.is_empty() && state.held.len() == 1);
        // A long editor session holds one entry per run of chunks, not per chunk.
        let mut thinking = active_state();
        thinking.suspended = true;
        for _ in 0..100 {
            emit_locked(&mut bytes, &mut thinking, Entry::Reasoning("more ".into())).unwrap();
        }
        assert_eq!(thinking.held.len(), 1);
        assert!(!state.record.is_empty(), "kept for Ctrl+O as well");

        // On resume it is laid out from where the transcript really is: here
        // column 7, so the line breaks before a word that would cross 40.
        state.suspended = false;
        let text = render_held_locked(&mut state);
        assert_eq!(text, " three four five six seven eight\nnine");
        assert!(state.held.is_empty());
    }

    #[test]
    fn escapes_inside_tool_arguments_never_reach_the_terminal() {
        let mut state = active_state();
        // What StreamPrinter::status passes on: the agent's line, made safe.
        // Collapsed, the arguments are decoded from JSON, escapes and all.
        let message = safe_text(r#"[tool] running: vm_shell_input, arguments: {"text": "\u001b[2J\u001b[r\u001b7\u001b]52;c;aGk=\u0007:wq"}"#);
        let mut bytes = Vec::new();
        emit_locked(&mut bytes, &mut state, Entry::Status(message)).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let line = text.split("[*] ").nth(1).unwrap();
        // Up to the end of the call's arguments, before the display's own
        // cursor bookkeeping.
        let line = &line[..line.find(")\x1b[0m").unwrap()];
        for sequence in ["\x1b[2J", "\x1b[r", "\x1b7", "\x1b]52", "\u{7}"] {
            assert!(!line.contains(sequence), "{sequence:?} in {line:?}");
        }
        assert!(line.contains("␛[2J"), "{line:?}");
    }

    #[test]
    fn the_idle_prompts_shape_is_known_for_erasing_it() {
        // One row: the cursor's row is where it starts.
        assert_eq!(prompt_geometry("half typed", 10, 80), (0, 1));
        assert_eq!(prompt_geometry("", 0, 80), (0, 1));
        // Wrapped: 5 + 100 columns at 80 is two rows, cursor on the second.
        let long = "x".repeat(100);
        assert_eq!(prompt_geometry(&long, 100, 80), (1, 2));
        assert_eq!(prompt_geometry(&long, 10, 80), (0, 2));
        // Several lines, cursor on the third.
        assert_eq!(prompt_geometry("one\ntwo\nthree", 11, 80), (2, 3));
        // A cursor inside a multibyte character is taken back to its start.
        assert_eq!(prompt_geometry("héllo", 2, 80), (0, 1));
        // A line filled exactly to the edge: the cursor waits at the start of
        // the next row, which the prompt then takes too.
        let full = "x".repeat(74);
        assert_eq!(prompt_geometry(&full, 74, 80), (1, 2));
        assert_eq!(prompt_geometry(&format!("{full}\nmore"), 74, 80), (1, 3));
        // A wide character that would straddle the edge moves to the next row.
        let wide = format!("{}界", "x".repeat(73));
        assert_eq!(prompt_geometry(&wide, wide.len(), 80), (1, 2));
        // Emoji sequences are as wide as one emoji (20 families fit on a row).
        let family = "👨\u{200D}👩\u{200D}👧".repeat(20);
        assert_eq!(prompt_geometry(&family, family.len(), 80), (0, 1));
        assert_eq!(clusters("👍🏽❤\u{FE0F}🇫🇷e\u{301}x").collect::<Vec<_>>(), ["👍🏽", "❤\u{FE0F}", "🇫🇷", "e\u{301}", "x"]);
        assert_eq!(clusters("🇫\u{FE0F}🇷🇫").collect::<Vec<_>>(), ["🇫\u{FE0F}", "🇷🇫"]);
        assert_eq!(UnicodeWidthStr::width("❤\u{FE0F}"), 2);
        // Later lines start six columns in too.
        assert_eq!(prompt_geometry(&format!("a\n{}", "y".repeat(74)), 2, 80), (1, 2));
        assert_eq!(prompt_geometry(&format!("a\n{}", "y".repeat(75)), 2, 80), (1, 3));
    }

    #[test]
    fn a_turn_is_marked_in_the_record_without_writing_anything() {
        let display = ActiveDisplay::new();
        display.begin_turn().unwrap();
        let state = display.lock().unwrap();
        let (entries, _) = state.record.replay(Detail::Collapsed);
        assert!(matches!(entries.as_slice(), [recorded] if recorded.entry == Entry::Turn));
    }

    #[test]
    fn during_a_sleep_the_line_says_sleeping_and_a_message_waits_for_it() {
        let display = ActiveDisplay::new();
        display.set_activity(Activity::Sleeping).unwrap();
        assert_eq!(display.lock().unwrap().activity.label(), "sleeping");
        display.queued("after you wake").unwrap();
        let state = display.lock().unwrap();
        let (entries, _) = state.record.replay(Detail::Collapsed);
        assert!(matches!(&entries[0].entry, Entry::Notice { brief, .. } if brief.starts_with("queued for after the sleep")), "{:?}", entries[0].entry);
    }

    #[test]
    fn a_queued_message_waits_to_be_shown_until_s0ma_takes_it() {
        let display = ActiveDisplay::new();
        display.queued("also check /var").unwrap();
        display.show_operator("also check /var").unwrap();
        display.delivered(1).unwrap();
        let state = display.lock().unwrap();
        let (entries, _) = state.record.replay(Detail::Collapsed);
        assert!(matches!(&entries[0].entry, Entry::Notice { brief, .. } if brief.starts_with("queued for after the current step")));
        assert!(matches!(&entries[1].entry, Entry::Operator(text) if text == "also check /var"));
        assert!(matches!(&entries[2].entry, Entry::Notice { brief, full, .. } if brief == "s0ma has your message" && full.contains("↳ s0ma")));
        // Sent now, it shows at once with its status line under it.
        drop(state);
        display.commit_operator("stop", OperatorNote::SendNow).unwrap();
        let state = display.lock().unwrap();
        let (entries, _) = state.record.replay(Detail::Collapsed);
        assert!(matches!(&entries[4].entry, Entry::Notice { brief, .. } if brief.starts_with("sending now")));
    }

    #[test]
    fn notes_lines_and_operator_input_are_kept_for_the_redraw() {
        let display = ActiveDisplay::new();
        display.show_operator("check the disk\nand /home").unwrap();
        display.note("[mode: work]").unwrap();
        let state = display.lock().unwrap();
        let (entries, _) = state.record.replay(Detail::Expanded);
        assert_eq!(entries.len(), 2);
        assert!(matches!(&entries[0].entry, Entry::Operator(text) if text == "check the disk\nand /home"));
        assert!(matches!(&entries[1].entry, Entry::Note(text) if text == "[mode: work]"));
    }

    #[test]
    fn a_reply_held_for_the_editor_is_not_erased_by_the_next_tool_line() {
        let mut state = active_state();
        state.suspended = true;
        let mut bytes = Vec::new();
        emit_locked(&mut bytes, &mut state, Entry::Answer("Let me check the logs. ".into())).unwrap();
        emit_locked(&mut bytes, &mut state, Entry::Status("[tool] running: vm_exec, arguments: {\"command\": \"ls\"}".into())).unwrap();
        assert!(bytes.is_empty());
        state.suspended = false;
        let text = render_held_locked(&mut state);
        // The tool line starts on a line of its own.
        let reply_end = text.find("logs.").unwrap();
        let tool = text.find("[*] ").unwrap();
        assert!(text[reply_end..tool].contains("\r\n"), "{text:?}");
    }

    #[test]
    fn progress_line_fits_and_shows_activity() {
        let mut state = active_state();
        state.activity = Activity::Tool("web_search".into());
        let mut bytes = Vec::new();
        draw_status_locked(&mut bytes, &state).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("running web_search · 0s") && !text.contains('…'));
        assert!(!text.contains("ctrl+"), "the keys live in /help now: {text}");
        measured(&mut state.meter);
        state.columns = 80;
        let mut bytes = Vec::new();
        draw_status_locked(&mut bytes, &state).unwrap();
        assert!(String::from_utf8(bytes).unwrap().contains("prefill 8.0k tok/s · decode 50 tok/s"));

        // A narrow terminal truncates instead of wrapping into the input.
        state.columns = 12;
        state.hint = Some("/loop  /loops".into());
        let mut bytes = Vec::new();
        draw_status_locked(&mut bytes, &state).unwrap();
        let visible: String = String::from_utf8(bytes)
            .unwrap()
            .split('\x1b')
            .map(|part| part.split_once(['m', 'H', 'K']).map_or(part, |(_, rest)| rest))
            .collect();
        assert!(UnicodeWidthStr::width(visible.as_str()) <= 11, "{visible:?}");
    }

    fn visible(bytes: Vec<u8>) -> String {
        String::from_utf8(bytes)
            .unwrap()
            .split('\x1b')
            .map(|part| part.split_once(['m', 'H', 'K']).map_or(part, |(_, rest)| rest))
            .collect()
    }

    #[test]
    fn queued_messages_show_on_the_progress_line() {
        let mut state = active_state();
        state.columns = 120;
        let mut bytes = Vec::new();
        draw_status_locked(&mut bytes, &state).unwrap();
        let idle = visible(bytes);
        assert!(!idle.contains("queued"), "{idle}");

        state.queued.store(2, Ordering::SeqCst);
        state.input = "half typed".into();
        let mut bytes = Vec::new();
        draw_status_locked(&mut bytes, &state).unwrap();
        let line = visible(bytes);
        assert!(line.contains("waiting for the model · 0s · 2 queued"), "{line}");
        assert!(!line.contains("ctrl+"), "no key help on the line any more: {line}");
    }

    /// A request measured in full: 12,000 prompt tokens of which the server
    /// had 10,000 cached, first output 250 ms after the generation started,
    /// then 100 tokens over two seconds. Prefill 8,000 tok/s, decode 50.
    fn measured(meter: &mut TokenMeter) {
        let base = Instant::now() - Duration::from_secs(10);
        meter.started(base);
        meter.prompt(Some(12_000), Some(10_000));
        for step in 0..=20u64 {
            let now = base + Duration::from_millis(250 + step * 100);
            meter.text(17, now);
            meter.usage(5 * step, false, now);
        }
        meter.usage(100, true, base + Duration::from_millis(2_250));
    }

    #[test]
    fn the_corner_shows_prefill_then_decode_and_gives_up_prefill_first() {
        let mut state = active_state();
        assert!(status_segments(&state).is_empty(), "nothing measured yet");
        measured(&mut state.meter);
        let segments = status_segments(&state);
        assert_eq!(fit_segments(&segments, 80), "prefill 8.0k tok/s · decode 50 tok/s");
        assert_eq!(fit_segments(&segments, 20), "decode 50 tok/s", "decode is the one worth keeping");
    }

    #[test]
    fn the_idle_prompt_shows_the_rates_or_where_the_keys_went() {
        assert!(rate_segments(None, None).is_empty());
        let both = rate_segments(Some("≈3.1k tok/s".into()), Some("57 tok/s".into()));
        assert_eq!(both, vec![("prefill ≈3.1k tok/s".to_string(), 2), ("decode 57 tok/s".to_string(), 1)]);
        // Decode alone, before any prefill was worth reporting.
        assert_eq!(rate_segments(None, Some("57 tok/s".into())), vec![("decode 57 tok/s".to_string(), 1)]);
    }

    #[test]
    fn segments_drop_the_least_needed_first_and_keep_order() {
        let segments = vec![
            ("first piece".to_string(), 1),
            ("third rank".to_string(), 3),
            ("second rank".to_string(), 2),
        ];
        assert_eq!(fit_segments(&segments, 80), "first piece · third rank · second rank");
        assert_eq!(fit_segments(&segments, 27), "first piece · second rank");
        assert_eq!(fit_segments(&segments, 12), "first piece");
        assert_eq!(fit_segments(&segments, 5), "");
        // A piece too wide to fit doesn't crowd out a narrower, less needed one.
        let segments = vec![("a rather long piece".to_string(), 1), ("short".to_string(), 2)];
        assert_eq!(fit_segments(&segments, 12), "short");
    }

    #[test]
    fn the_corner_carries_no_key_help_in_either_detail_mode() {
        // The keys are listed by /help; the corner is for the rates.
        let mut state = active_state();
        state.columns = 80;
        for detail in [Detail::Collapsed, Detail::Expanded] {
            state.renderer = Renderer::new(detail, DEFAULT_CHARS_PER_TOKEN);
            let mut bytes = Vec::new();
            draw_status_locked(&mut bytes, &state).unwrap();
            let line = visible(bytes);
            assert!(!line.contains("ctrl+") && line.trim_end().ends_with("0s"), "{line}");
        }
    }

    #[test]
    fn the_corner_shows_the_decode_rate_once_there_is_one() {
        let mut state = active_state();
        state.columns = 100;
        let base = Instant::now();
        // Three chunks, two measured intervals: 20 estimated tokens of decode
        // in half a second. (The first chunk's arrival is prefill, not decode,
        // so it sets the baseline rather than counting towards the rate.)
        state.meter.request();
        state.meter.text(38, base - Duration::from_millis(900));
        state.meter.text(38, base - Duration::from_millis(650));
        state.meter.text(38, base - Duration::from_millis(400));
        state.meter.end_request();
        let mut bytes = Vec::new();
        draw_status_locked(&mut bytes, &state).unwrap();
        let line = visible(bytes);
        // No prompt counts, so no prefill: decode alone, as estimated.
        assert!(line.ends_with("decode ≈40 tok/s"), "{line}");
        // Narrow: the rate is the last thing to go.
        state.columns = 48;
        let mut bytes = Vec::new();
        draw_status_locked(&mut bytes, &state).unwrap();
        assert!(visible(bytes).ends_with("decode ≈40 tok/s"));
    }

    #[test]
    fn output_waits_while_the_terminal_has_a_size_the_pane_was_not_laid_out_for() {
        let mut state = active_state();
        let laid_out = (state.columns, state.rows);
        TEST_SIZE.with(|size| size.set(Some((laid_out.0 - 20, laid_out.1 - 4))));
        let mut bytes = Vec::new();
        emit_locked(&mut bytes, &mut state, Entry::Raw { text: "after the resize\n".into(), operator: false }).unwrap();
        assert!(bytes.is_empty(), "nothing drawn at the old size");
        assert!(state.suspended && state.waiting_resize.is_some());
        assert_eq!(state.held.len(), 1, "held, to be laid out after the resize");
        // The spinner's redraw notices it too, with nothing written before it.
        let mut quiet = active_state();
        redraw_status(&mut quiet).unwrap();
        assert!(quiet.suspended && quiet.waiting_resize.is_some(), "the status line is not drawn at the old size");
        // The same size again (laid out, or the terminal came back to it).
        TEST_SIZE.with(|size| size.set(Some(laid_out)));
        state.suspended = false;
        state.waiting_resize = None;
        emit_locked(&mut bytes, &mut state, Entry::Raw { text: "drawn\n".into(), operator: false }).unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("drawn"));
        TEST_SIZE.with(|size| size.set(None));
    }

    #[test]
    fn the_emergency_restore_keeps_the_cursor_off_the_conversation() {
        let codes = restore_codes(false);
        // The scroll region's reset (which homes the cursor) is wrapped in a
        // save and restore, before the newline the message follows.
        let reset = codes.find("\x1b7\x1b[r\x1b8").expect("wrapped reset");
        assert!(reset < codes.find("\r\n").unwrap(), "{codes:?}");
        assert!(!codes.contains("\x1b[?1049l"));
        let codes = restore_codes(true);
        assert!(codes.starts_with("\x1b[<99u\x1b[?1049l"), "the page's screen left first: {codes:?}");
    }

    #[test]
    fn collapsed_output_reaches_the_screen_through_the_display() {
        let mut state = active_state();
        let mut bytes = Vec::new();
        emit_locked(&mut bytes, &mut state, Entry::Reasoning("weighing it".into())).unwrap();
        emit_locked(&mut bytes, &mut state, Entry::Status("[tool] running: vm_exec, arguments: {\"command\": \"ls\"}".into())).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("[*] thinking · ~3 tokens"), "{text:?}");
        // The call takes over the same line: one line per stretch of work.
        let call = text.rfind("[*] ").expect("the call is shown");
        assert!(text[..call].ends_with("\r\x1b[2K      \x1b[2;36m"), "rewritten in place, in the gutter: {text:?}");
        assert!(text.contains("vm_exec") && text.contains("(ls)"), "{text:?}");
        // Each write goes to s0ma's cursor and puts the operator's back.
        assert_eq!(text.matches("\x1b8").count(), 2);
        assert_eq!(state.record.replay(Detail::Expanded).0.len(), 2);
    }

    #[test]
    fn a_tagged_harness_line_reads_as_a_status() {
        assert_eq!(tagged_brief("[compaction] no newer completed history to activate"), "compaction · no newer completed history to activate");
        assert_eq!(tagged_brief("\n[compaction] window #1..#4: kept"), "compaction · window #1..#4: kept");
        assert_eq!(tagged_brief("plain text"), "plain text");
        assert_eq!(tagged_brief("[not a tag]"), "[not a tag]");
        // A block keeps its shape: one tag, later lines as they were.
        assert_eq!(
            tagged_brief("[mcp] config: /x/mcp.json\n  fs      connected · local\n  github  failed (logs: /x)\n  ! fs says its tool list changed"),
            "mcp · config: /x/mcp.json\n  fs      connected · local\n  github  failed (logs: /x)\n  ! fs says its tool list changed"
        );
        assert_eq!(tagged_brief("[mcp] fs: reconnected\n[mcp] github: failed"), "mcp · fs: reconnected\ngithub: failed");
    }

    #[test]
    fn notes_say_when_s0ma_reads_a_message() {
        assert_eq!(operator_note_text(OperatorNote::Plain, &Activity::Writing), None);
        assert_eq!(
            queued_note("hey s0ma\n  can you   check"),
            "queued for after the current step: \u{201c}hey s0ma can you check\u{201d}"
        );
        assert!(queued_note(&"word ".repeat(40)).contains('…'));
        assert!(UnicodeWidthStr::width(queued_note(&"word ".repeat(40)).as_str()) <= 80);
        assert_eq!(operator_note_text(OperatorNote::SendNow, &Activity::Thinking).unwrap(), "sending now");
        let during_tool = operator_note_text(OperatorNote::SendNow, &Activity::Tool("vm_exec".into())).unwrap();
        assert!(during_tool.contains("when vm_exec returns") && during_tool.contains("/kill"), "{during_tool}");
        assert_eq!(delivered_note(1), "s0ma has your message");
        assert_eq!(delivered_note(3), "s0ma has your 3 messages");
    }

    #[test]
    fn cut_text_always_says_so() {
        assert_eq!(truncate_to_width("short", 10), "short");
        assert_eq!(truncate_to_width("exactly ten", 11), "exactly ten");
        assert_eq!(truncate_to_width("abcdefghij", 5), "abcd…");
        assert_eq!(truncate_to_width("你好世界", 5), "你好…");
        assert_eq!(truncate_to_width("abc", 1), "…");
        assert_eq!(truncate_to_width("abc", 0), "");
    }

    #[test]
    fn pane_geometry_reserves_a_status_row_when_there_is_room() {
        // The progress line over five input rows.
        assert_eq!(pane_geometry(24), (18, 1, 5));
        // Short of room (the page's two top rows and two of conversation
        // kept): the progress line goes first.
        assert_eq!(pane_geometry(7), (4, 1, 2));
        assert_eq!(pane_geometry(6), (4, 0, 2));
        assert_eq!(pane_geometry(5), (4, 0, 1));
        assert_eq!(format_elapsed(Duration::from_secs(64)), "1m 04s");
    }

    #[test]
    fn the_pane_follows_the_cursor() {
        let input = "1\n2\n3\n4\n5\n6\n7";
        // Cursor at the end: the last five lines are shown.
        let end = input_layout(input, input.len(), 40, 5);
        assert_eq!((end.first, end.cursor), (2, (6, 1)));
        // Cursor on the first line: the pane scrolls back to the top.
        let top = input_layout(input, 0, 40, 5);
        assert_eq!((top.first, top.cursor), (0, (0, 0)));
        // Cursor before a character that wraps: it is drawn on the new line.
        let wrapped = input_layout("abcdefgh", 6, 13, 5);
        assert_eq!(wrapped.cursor, (1, 0));
    }

    #[test]
    fn unicode_drafts_fit_without_wrapping_into_the_transcript() {
        for input in [
            "你好世界你好世界",
            "e\u{0301}e\u{0301}e\u{0301}",
            "🌱🌱🌱🌱",
            "a\nb\nc\nd\ne\nf",
            "\t\x1b[H",
        ] {
            let layout = input_layout(input, input.len(), 12, 5);
            let shown = &layout.lines[layout.first..];
            assert!(shown.len() <= 5);
            assert!(
                shown
                    .iter()
                    .all(|line| UnicodeWidthStr::width(line.as_str()) <= 6)
            );
        }
    }

    #[test]
    fn while_the_idle_prompt_is_read_harness_lines_wait_to_go_above_it() {
        let mut state = active_state();
        state.active = false;
        state.prompt_up = true;
        let mut bytes = Vec::new();
        emit_locked(&mut bytes, &mut state, Entry::Line("[archive] indexing is stalled".into())).unwrap();
        assert!(bytes.is_empty(), "not written after what is being typed");
        assert_eq!(state.held.len(), 1);
        assert!(!state.record.is_empty());
        // During a turn the pane is there for it: written at once.
        state.held.clear();
        state.active = true;
        emit_locked(&mut bytes, &mut state, Entry::Line("[tool] ran".into())).unwrap();
        assert!(state.held.is_empty() && !bytes.is_empty());
    }

    #[test]
    fn a_snapshot_loads_into_another_display_as_it_was() {
        let primary = ActiveDisplay::new();
        primary.transcript_line("one").unwrap();
        primary.reasoning("weighing it").unwrap();
        primary.set_activity(Activity::Thinking).unwrap();
        let mut taken = None;
        primary.snapshot(|snapshot| taken = Some(snapshot)).unwrap();
        let taken: DisplaySnapshot = serde_json::from_str(&serde_json::to_string(&taken.unwrap()).unwrap()).unwrap();
        let attached = ActiveDisplay::new();
        attached.load(taken).unwrap();
        let state = attached.lock().unwrap();
        let (entries, _) = state.record.all();
        let entries: Vec<Entry> = entries.map(|recorded| recorded.entry.clone()).collect();
        assert_eq!(entries, [Entry::Line("one".into()), Entry::Reasoning("weighing it".into())]);
        assert_eq!(state.activity, Activity::Thinking);
        assert!(!state.quiet_pass, "between turns: the next turn starts a pass of its own");
    }

    #[test]
    fn the_page_is_drawn_from_the_record_with_as_much_as_fits() {
        let mut state = active_state();
        for i in 1..=60 {
            state.record.push(Instant::now(), Entry::Line(format!("line {i}")));
        }
        state.suspended = true;
        let mut bytes = Vec::new();
        redraw_active_locked(&mut bytes, &mut state, 40, 24).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        // Cleared, then the region (the rows under the top two, above the
        // pane's six), then the rows from under the top two.
        assert!(text.starts_with("\x1b[r\x1b[H\x1b[2J"), "{text:?}");
        let region = text.find("\x1b[3;18r").expect("the conversation's rows scroll");
        let first = text.find("\x1b[3;1H").expect("drawn from under the rule");
        assert!(region < first);
        // Sixteen rows fit: the last fifteen lines and the row after them.
        assert!(text.contains("\x1b[3;1Hline 46\r\n") && text.contains("line 60\r\n\x1b7"), "{text:?}");
        assert!(!text.contains("line 45"), "{text:?}");
        assert!(!state.suspended, "drawn: output goes on");
        assert!(text.contains("you>"), "the pane under it");
    }

    #[test]
    fn under_a_tab_output_is_held_and_kept_and_nothing_is_written() {
        for active in [false, true] {
            let mut state = active_state();
            state.active = active;
            state.covered = Some(Cover { prompt: None });
            let mut bytes = Vec::new();
            emit_locked(&mut bytes, &mut state, Entry::Line("[subagent] done".into())).unwrap();
            emit_locked(&mut bytes, &mut state, Entry::Answer("The worker ".into())).unwrap();
            emit_locked(&mut bytes, &mut state, Entry::Answer("reported.".into())).unwrap();
            assert!(bytes.is_empty(), "a tab has the screen: nothing is drawn under it");
            assert_eq!(state.held.len(), 2, "held for when it closes, streamed text joined");
            assert!(matches!(&state.held[1].2, Entry::Answer(text) if text == "The worker reported."));
            assert!(!state.record.is_empty(), "and kept for the redraw");
            let (kept, _) = state.record.replay(Detail::Expanded);
            assert!(kept.iter().any(|recorded| matches!(&recorded.entry, Entry::Line(text) if text == "[subagent] done")));
        }
    }

    #[test]
    fn stream_transaction_restores_output_before_writing_and_redraws_input_last() {
        let mut state = DisplayState {
            active: true,
            columns: 80,
            rows: 24,
            input: "my unfinished draft".into(),
            input_cursor: 19,
            operator_revision: 0,
            renderer: Renderer::new(Detail::Expanded, DEFAULT_CHARS_PER_TOKEN),
            record: TranscriptRecord::default(),
            transient: None,
            meter: TokenMeter::default(),
            suspended: false,
            held: Vec::new(),
            activity: Activity::Writing,
            started: Instant::now(),
            frame_drawn: 0,
            hint: None,
            queued: Arc::default(),
            covered: None,
            quiet_pass: false,
            shrunk: false,
            waiting_resize: None,
            laid_out_at: None,
            bar: String::new(),
            prompt_up: false,
        };
        let mut bytes = Vec::new();
        emit_locked(&mut bytes, &mut state, Entry::Raw { text: "model output\n".into(), operator: false }).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.find("model output").unwrap() < text.find("my unfinished draft").unwrap());
        assert_eq!(state.input, "my unfinished draft");
        assert!(text.contains("model output\r\n"));
    }
}

fn clear_input_pane(stdout: &mut impl Write, state: &DisplayState) -> io::Result<()> {
    let pane_rows = pane_rows(state.rows);
    let first_row = state.rows - pane_rows;

    for offset in 0..pane_rows {
        queue!(
            stdout,
            MoveTo(0, first_row + offset),
            Clear(ClearType::CurrentLine)
        )?;
    }

    Ok(())
}

/// Raw terminal mode does not perform the normal LF -> CRLF mapping.
///
/// Convert bare LF to CRLF so streamed multiline output always returns
/// to column zero. Existing CRLF sequences are preserved.
fn write_normalized(writer: &mut impl Write, text: &str) -> io::Result<()> {
    let bytes = text.as_bytes();

    let mut start = 0;

    for (index, byte) in bytes.iter().enumerate() {
        if *byte != b'\n' {
            continue;
        }

        if start < index {
            writer.write_all(&bytes[start..index])?;
        }

        if index > 0 && bytes[index - 1] == b'\r' {
            writer.write_all(b"\n")?;
        } else {
            writer.write_all(b"\r\n")?;
        }

        start = index + 1;
    }

    if start < bytes.len() {
        writer.write_all(&bytes[start..])?;
    }

    Ok(())
}

#[cfg(test)]
mod table_layout_tests {
    use super::*;

    const WIDE: &str = "| Port | Service | Version | Notes |\n|---|---|---|---|\n| 22 | ssh | OpenSSH 9.6p1 Ubuntu 3ubuntu13.5 | Password auth enabled; root login permitted, worth checking against the hardening baseline |\n| 80 | http | nginx 1.24.0 | Redirects to 443 |\n| 8080 | http-proxy | Apache Tomcat 9.0.65 | Manager app exposed at /manager/html with default credentials suspected |\n";

    /// What the transcript gets: rendered for `width` (if any), then word-wrapped at it.
    fn through_the_display(width: usize, set_width: bool) -> (String, String) {
        let mut md = crate::markdown::MarkdownStream::new();
        if set_width {
            md.set_width(Some(width));
        }
        let styled = md.push(&safe_text(WIDE)) + &md.finish();
        let mut wrap = WordWrap::default();
        let mut out = String::new();
        wrap.push(&styled, width, &mut out);
        out.push_str(&wrap.take_pending(width));
        (styled, out)
    }

    #[test]
    fn tables_fitted_beside_the_gutter_pass_through_word_wrap_untouched() {
        for width in [100, 70, 44] {
            let gutter = gutter_width(width);
            let mut md = crate::markdown::MarkdownStream::new();
            md.set_width(Some(width - gutter));
            let styled = md.push(&safe_text(WIDE)) + &md.finish();
            let mut wrap = WordWrap { indent: gutter, ..WordWrap::default() };
            let mut out = String::new();
            wrap.push(&styled, width, &mut out);
            out.push_str(&wrap.take_pending(width));
            assert_eq!(out.matches('\n').count(), styled.matches('\n').count(), "a line was broken at {width} columns");
        }
    }

    #[test]
    fn fitted_tables_pass_through_word_wrap_untouched() {
        // The bug: laid out at natural width, the wrapper breaks rows apart.
        let (styled, wrapped) = through_the_display(70, false);
        assert!(wrapped.matches('\n').count() > styled.matches('\n').count());
        // Fitted to the width, as a grid or as records, it never has to.
        for width in [100, 70, 44, 30] {
            let (styled, wrapped) = through_the_display(width, true);
            assert_eq!(wrapped.matches('\n').count(), styled.matches('\n').count(), "a line was broken at {width} columns");
        }
    }
}
