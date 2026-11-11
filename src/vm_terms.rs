//! s0ma's shells in the VM, for the operator's terminal views (Alt+2…,
//! `tabs.rs`): which are open, what each shows, and the keys typed into one.
//!
//! The shells are the tmux sessions the vm_shell_* tools open (`s0ma-<name>`:
//! s0ma's `main` and any others, and one per subagent that used a shell).
//! Everything goes through the guest agent, as those tools do, and from a
//! thread of its own, so a slow VM never holds up the keyboard or the screen:
//! the views read what it last found. It works only while a view is up
//! (`touch`): the list of shells every two seconds, each shell on screen a
//! few times a second, and its history once when scrolled back. Two panes
//! can show two shells (or one twice): each is read, sized and kept on its
//! own.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{
    Condvar, Mutex, MutexGuard,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

/// The shell on screen is read again this often; while it is changing, or
/// just typed into, this often, for `BUSY_FOR` after.
const CAPTURE_EVERY: Duration = Duration::from_millis(400);
const CAPTURE_BUSY: Duration = Duration::from_millis(120);
const BUSY_FOR: Duration = Duration::from_millis(1500);
/// The list of shells, this often; after an error, less often.
const LIST_EVERY: Duration = Duration::from_secs(2);
const LIST_AFTER_ERROR: Duration = Duration::from_secs(10);
/// A view that has not drawn for this long is gone: stop reading.
const IDLE_AFTER: Duration = Duration::from_secs(3);
/// History lines read when a view scrolls back.
const HISTORY_LINES: usize = 2_000;
/// Typed text sent in one go, at most (bytes); a paste, in pieces of this.
const SEND_CHARS: usize = 4_000;
const PASTE_BYTES: usize = 24_000;
/// After a shell could not be read, the next try waits this long.
const RETRY_AFTER: Duration = Duration::from_secs(2);

const MARK: &str = "__S0MA_VIEW_7E21__";
/// After the pane's rows: blank rows at the bottom survive whatever trims the
/// guest's output.
const END: &str = "__S0MA_VIEW_END_7E21__";

/// What a shell showed when it was last read.
#[derive(Clone, Debug, PartialEq)]
pub struct Screen {
    pub session: String,
    /// Rows, top to bottom: the pane, with its history before it when read
    /// for scrolling back. Each keeps the pane's own colours, as SGR codes
    /// (`capture-pane -e`).
    pub lines: Vec<String>,
    /// The cursor's column and row in `lines`.
    pub cursor: (usize, usize),
    /// The pane's columns and rows.
    pub size: (usize, usize),
    pub dead: bool,
    /// The program in it shows its cursor (a pager or a menu may hide it).
    pub cursor_shown: bool,
    pub command: String,
    pub cwd: String,
    pub at: Instant,
}

/// Something typed, as tmux sends it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Input {
    /// Literal text, typed.
    Text(String),
    /// Pasted text: given to the shell as a paste (bracketed, when the
    /// program in it asked for that), so a pasted line is not run before the
    /// rest of it arrives.
    Paste(String),
    /// A key by tmux's name for it: `Enter`, `BSpace`, `C-Up`.
    Key(String),
    /// Raw bytes: control characters, Alt as Esc before a character.
    Bytes(Vec<u8>),
}

/// A size asked of a shell (columns, rows), and when.
type Asked = ((usize, usize), Instant);

#[derive(Default)]
struct State {
    sessions: Vec<String>,
    listed: Option<Instant>,
    list_error: Option<String>,
    /// The shells views show (two panes, two), and when each last asked.
    watched: BTreeMap<String, Instant>,
    live: BTreeMap<String, Screen>,
    /// Scrolled back in (the sessions), and what was read of each.
    history_wanted: BTreeSet<String>,
    history: BTreeMap<String, Screen>,
    /// Why each shell could not be read or typed into, if it could not; at
    /// `""`, a problem not about one shell (opening a netbox).
    error: BTreeMap<String, String>,
    outbox: VecDeque<(String, Input)>,
    /// Read these shells again at once (just shown, or just typed into).
    capture_now: BTreeSet<String>,
    /// When reading the shell on screen, or its history, last failed: the
    /// next try waits `RETRY_AFTER`, so an unreachable VM is not asked over
    /// and over.
    capture_failed: Option<Instant>,
    history_failed: Option<Instant>,
    touched: Option<Instant>,
    running: bool,
    /// The netboxes among the shells, with their settings, as last listed.
    netboxes: Vec<crate::netboxes::Netbox>,
    /// What the operator typed into each netbox (by session) since s0ma last read it.
    typing: std::collections::BTreeMap<String, Typing>,
    /// On an attached terminal: typed into a netbox here, for the s0ma that
    /// reads it (sent on by the reader's thread).
    netbox_forward: Vec<(String, Input)>,
    /// The sizes the views showing each shell have room for, with when each
    /// said it, and the size last asked of each shell: a pane as big as its
    /// view, so the shell uses all of it (the smaller, when two show one).
    fit: BTreeMap<String, Vec<Asked>>,
    fitted: BTreeMap<String, ((usize, usize), Instant)>,
    /// Typed into, or seen changing, lately: read more often until then.
    busy_until: BTreeMap<String, Instant>,
}

/// The operator's typing in one netbox: lines he finished with Enter, the one he is
/// in the middle of, and when he last typed.
#[derive(Default)]
struct Typing {
    lines: Vec<String>,
    line: String,
    at: Option<Instant>,
}

/// The operator typed this lately, or left a line half-typed: s0ma waits, then
/// goes to its lane, rather than type over him.
const OPERATOR_TYPING: Duration = Duration::from_secs(3);

static STATE: Mutex<Option<State>> = Mutex::new(None);
static WAKE: Condvar = Condvar::new();
static VERSION: AtomicU64 = AtomicU64::new(0);

fn state() -> MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn with<T>(read: impl FnOnce(&mut State) -> T) -> T {
    let mut guard = state();
    read(guard.get_or_insert_with(State::default))
}

fn changed() {
    VERSION.fetch_add(1, Ordering::SeqCst);
}

/// Moves on whenever anything here does.
pub fn version() -> u64 {
    VERSION.load(Ordering::SeqCst)
}

/// A view is up: keep the list of shells fresh, starting the reader if need be.
pub fn touch() {
    let start = with(|state| {
        state.touched = Some(Instant::now());
        !std::mem::replace(&mut state.running, true)
    });
    if start && !cfg!(test) {
        let spawned = std::thread::Builder::new().name("vm-terminals".into()).spawn(run);
        if spawned.is_err() {
            with(|state| state.running = false);
        }
    }
}

/// The shells, in tab order: s0ma's `main`, its others, then subagents'.
pub fn sessions() -> Vec<String> {
    with(|state| state.sessions.clone())
}

/// Why the list of shells could not be read, if it could not.
pub fn list_error() -> Option<String> {
    with(|state| state.list_error.clone())
}

/// Whether the list has been read at least once.
pub fn listed() -> bool {
    with(|state| state.listed.is_some())
}

/// A view shows `session`: read it often. Shown again after a while (or
/// for the first time), it is read afresh.
pub fn watch(session: &str) {
    let now = Instant::now();
    let fresh = with(|state| {
        let fresh = state.watched.get(session).is_none_or(|at| now.duration_since(*at) > IDLE_AFTER);
        state.watched.insert(session.to_string(), now);
        if fresh {
            state.live.remove(session);
            state.error.remove(session);
            state.capture_now.insert(session.to_string());
        }
        fresh
    });
    if fresh {
        WAKE.notify_all();
    }
}

/// The columns and rows a view has for `session`: the shell is resized to
/// them, as a terminal resizes the shell in it; to the smaller, when two
/// views show it.
pub fn fit(session: &str, columns: usize, rows: usize) {
    let size = (columns.max(MIN_FIT.0), rows.max(MIN_FIT.1));
    let now = Instant::now();
    let asked = with(|state| {
        let before = fit_of(state, session);
        let sizes = state.fit.entry(session.to_string()).or_default();
        sizes.retain(|(said, at)| *said != size && now.duration_since(*at) <= FIT_KEEP);
        sizes.push((size, now));
        fit_of(state, session) != before
    });
    if asked {
        WAKE.notify_all();
    }
}

/// The size to give `session`: the smallest any view said within
/// `FIT_KEEP` of the last word (the last, when only one view shows it).
fn fit_of(state: &State, session: &str) -> Option<(usize, usize)> {
    let sizes = state.fit.get(session)?;
    let last = sizes.iter().map(|(_, at)| *at).max()?;
    sizes
        .iter()
        .filter(|(_, at)| last.duration_since(*at) <= FIT_KEEP)
        .map(|(size, _)| *size)
        .reduce(|a, b| (a.0.min(b.0), a.1.min(b.1)))
}

const MIN_FIT: (usize, usize) = (20, 5);
const REFIT_AFTER: Duration = Duration::from_secs(5);
/// Two views showing one shell each say their size every frame: a word
/// this old is from a view that has gone (or changed size).
const FIT_KEEP: Duration = Duration::from_secs(1);

/// What `session` showed when last read.
pub fn live(session: &str) -> Option<Screen> {
    with(|state| state.live.get(session).cloned())
}

/// Scrolled back in `session` (`back`), or following it again: its history
/// is read once, and kept until the view follows again.
pub fn want_history(session: &str, back: bool) {
    let ask = with(|state| {
        if state.history_wanted.contains(session) == back {
            return false;
        }
        if back {
            state.history_wanted.insert(session.to_string());
        } else {
            state.history_wanted.remove(session);
        }
        state.history.remove(session);
        back
    });
    if ask {
        WAKE.notify_all();
    }
}

pub fn history(session: &str) -> Option<Screen> {
    with(|state| state.history.get(session).cloned())
}

/// Why `session` could not be read or typed into, if it could not (or
/// what went wrong opening a shell).
pub fn error(session: &str) -> Option<String> {
    with(|state| state.error.get(session).or_else(|| state.error.get("")).cloned())
}

/// Type `input` into `session`.
pub fn send(session: &str, input: Input) {
    with(|state| {
        if crate::netboxes::name_of(session).is_some() {
            if crate::tabs::remote() {
                state.netbox_forward.push((session.to_string(), input.clone()));
            } else {
                note_typing(state, session, &input);
            }
        }
        let pieces = match input {
            Input::Paste(text) => pieces(&text, PASTE_BYTES).into_iter().map(Input::Paste).collect(),
            Input::Text(text) => pieces(&text, SEND_CHARS).into_iter().map(Input::Text).collect(),
            input => vec![input],
        };
        for input in pieces {
            // Text typed in a burst goes in one piece.
            if let (Some((last_session, Input::Text(text))), Input::Text(more)) = (state.outbox.back_mut(), &input)
                && last_session == session
                && text.len() + more.len() <= SEND_CHARS
            {
                text.push_str(more);
                continue;
            }
            state.outbox.push_back((session.to_string(), input));
        }
    });
    WAKE.notify_all();
}

/// The operator's typing in a netbox, kept as lines for s0ma's next read: what he
/// ran, not each key. Keys that are not text are named in brackets.
fn note_typing(state: &mut State, session: &str, input: &Input) {
    const KEEP: usize = 40;
    let typing = state.typing.entry(session.to_string()).or_default();
    typing.at = Some(Instant::now());
    let Typing { lines, line, .. } = typing;
    match input {
        Input::Text(text) | Input::Paste(text) => {
            for c in text.chars() {
                if c == '\n' || c == '\r' {
                    lines.push(std::mem::take(line));
                } else {
                    line.push(c);
                }
            }
        }
        Input::Key(name) if name == "Enter" || name == "KPEnter" => lines.push(std::mem::take(line)),
        Input::Key(name) if name == "BSpace" => {
            line.pop();
        }
        Input::Key(name) => line.push_str(&format!("[{name}]")),
        Input::Bytes(bytes) => match bytes.as_slice() {
            [b'\r'] | [b'\n'] => lines.push(std::mem::take(line)),
            [0x7f] | [0x08] => {
                line.pop();
            }
            [byte] if *byte < 0x20 => {
                let name = format!("[ctrl-{}]", (byte + b'a' - 1) as char);
                line.push_str(&name);
                if *byte == 0x03 || *byte == 0x04 {
                    lines.push(std::mem::take(line));
                }
            }
            other => line.push_str(&String::from_utf8_lossy(other)),
        },
    }
    if lines.len() > KEEP {
        let extra = lines.len() - KEEP;
        lines.drain(..extra);
    }
}

/// Typed into netbox `session` on an attached terminal: kept here, with
/// what was typed on this one, for s0ma's next read.
pub fn note_netbox(session: &str, typed: &[Input]) {
    with(|state| {
        for input in typed {
            note_typing(state, session, input);
        }
    });
}

/// What the operator typed into netbox `session` since this was last asked, oldest
/// first: each line he entered, then the one he is still typing (marked so).
/// From every terminal: an attached one sends its typing here (`note_netbox`).
pub fn take_netbox_typing(session: &str) -> Vec<String> {
    with(|state| {
        let Some(typing) = state.typing.get_mut(session) else { return Vec::new() };
        let mut out: Vec<String> = std::mem::take(&mut typing.lines);
        if !typing.line.is_empty() {
            out.push(format!("{} (not entered yet)", typing.line));
        }
        out
    })
}

/// Why s0ma should not type into netbox `session` now: the operator typed in it in
/// the last few seconds. (A line he left half-typed is seen on the screen
/// itself, by the netbox's own check: what is kept here of his line misses
/// his shell's editing keys.)
pub fn operator_busy(session: &str) -> Option<&'static str> {
    with(|state| {
        let typing = state.typing.get(session)?;
        typing.at.is_some_and(|at| at.elapsed() < OPERATOR_TYPING).then_some("The operator is typing in it")
    })
}

/// The netboxes, with their settings, as last listed (the first netbox
/// even before the list is read).
pub fn netboxes() -> Vec<crate::netboxes::Netbox> {
    with(|state| {
        if state.netboxes.is_empty() {
            crate::netboxes::parse("").1
        } else {
            state.netboxes.clone()
        }
    })
}

/// Something went wrong away from the view (opening a netbox): shown in it.
pub fn set_error(error: String) {
    with(|state| state.error.insert(String::new(), error));
    changed();
}

/// Ctrl+\ in a netbox: the operator takes the wheel (s0ma types nowhere in it; its
/// commands go to its lane), or gives it back. Shown at once; the machine
/// is told away from the keys.
pub fn toggle_wheel(session: &str) -> bool {
    let Some(name) = crate::netboxes::name_of(session).map(str::to_string) else { return false };
    let taken = with(|state| {
        let taken = !state.netboxes.iter().any(|netbox| netbox.name == name && netbox.wheel);
        if let Some(netbox) = state.netboxes.iter_mut().find(|netbox| netbox.name == name) {
            netbox.wheel = taken;
        }
        taken
    });
    changed();
    let script = crate::netboxes::option_script(&name, "@s0ma_wheel", taken.then_some("1"));
    if !cfg!(test) {
        let _ = std::thread::Builder::new().name("netbox-wheel".into()).spawn(move || {
            if let Err(error) = crate::netboxes::machine(&script, 30) {
                set_error(format!("control: {error}"));
            }
            with(|state| state.listed = None);
            WAKE.notify_all();
        });
    }
    taken
}

/// `text` in pieces of at most `bytes`, cut between characters.
fn pieces(text: &str, bytes: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while rest.len() > bytes {
        let mut cut = bytes;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        out.push(rest[..cut].to_string());
        rest = &rest[cut..];
    }
    if !rest.is_empty() || out.is_empty() {
        out.push(rest.to_string());
    }
    out
}

/// A key typed in a shell's view, as what tmux should send; `None` for one it
/// has no way to send.
pub fn key_input(key: &KeyEvent) -> Option<Input> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let named = |name: &str| {
        let mut prefix = String::new();
        if ctrl {
            prefix.push_str("C-");
        }
        if alt {
            prefix.push_str("M-");
        }
        if shift {
            prefix.push_str("S-");
        }
        Some(Input::Key(format!("{prefix}{name}")))
    };
    match key.code {
        KeyCode::Char(c) if ctrl => {
            let byte = match c.to_ascii_lowercase() {
                c @ 'a'..='z' => c as u8 - b'a' + 1,
                ' ' | '@' | '2' => 0,
                '[' | '3' => 0x1b,
                '\\' | '4' => 0x1c,
                ']' | '5' => 0x1d,
                '^' | '6' => 0x1e,
                '_' | '-' | '/' | '7' => 0x1f,
                '?' | '8' => 0x7f,
                _ => return None,
            };
            Some(Input::Bytes(if alt { vec![0x1b, byte] } else { vec![byte] }))
        }
        KeyCode::Char(c) if alt => {
            let mut bytes = vec![0x1b];
            bytes.extend(c.to_string().as_bytes());
            Some(Input::Bytes(bytes))
        }
        KeyCode::Char(c) => Some(Input::Text(c.to_string())),
        KeyCode::Enter if alt => Some(Input::Bytes(vec![0x1b, b'\r'])),
        KeyCode::Enter => Some(Input::Key("Enter".into())),
        KeyCode::Backspace if alt || ctrl => Some(Input::Bytes(vec![0x1b, 0x7f])),
        KeyCode::Backspace => Some(Input::Key("BSpace".into())),
        KeyCode::Tab => Some(Input::Key("Tab".into())),
        KeyCode::BackTab => Some(Input::Key("BTab".into())),
        KeyCode::Esc => Some(Input::Key("Escape".into())),
        KeyCode::Up => named("Up"),
        KeyCode::Down => named("Down"),
        KeyCode::Left => named("Left"),
        KeyCode::Right => named("Right"),
        KeyCode::Home => named("Home"),
        KeyCode::End => named("End"),
        KeyCode::PageUp => named("PPage"),
        KeyCode::PageDown => named("NPage"),
        KeyCode::Insert => named("IC"),
        KeyCode::Delete => named("DC"),
        KeyCode::F(n) if (1..=12).contains(&n) => named(&format!("F{n}")),
        _ => None,
    }
}

// ------------------------------------------------------------------ the reader

/// What the reader does next.
struct Work {
    sends: Vec<(String, Input)>,
    capture: Option<String>,
    /// Resize the shell read to this first.
    resize: Option<(usize, usize)>,
    history: Option<String>,
    list: bool,
}

fn next_work(state: &mut State, now: Instant) -> Option<Work> {
    // What was typed goes, a session's at a time, even after its view has
    // closed: a command typed and left at once still gets its Enter.
    let sends: Vec<_> = match state.outbox.front().map(|(session, _)| session.clone()) {
        Some(session) => {
            let mut taken = Vec::new();
            let mut bytes = 0;
            while let Some((next, input)) = state.outbox.front() {
                let size = match input {
                    Input::Text(text) | Input::Paste(text) => text.len(),
                    Input::Key(name) => name.len(),
                    Input::Bytes(bytes) => bytes.len(),
                };
                if *next != session || (!taken.is_empty() && bytes + size > PASTE_BYTES) {
                    break;
                }
                bytes += size;
                taken.extend(state.outbox.pop_front());
            }
            taken
        }
        None => Vec::new(),
    };
    if state.touched.is_none_or(|at| now.duration_since(at) > IDLE_AFTER) {
        return (!sends.is_empty()).then_some(Work { sends, capture: None, resize: None, history: None, list: false });
    }
    let watched: Vec<String> =
        state.watched.iter().filter(|(_, at)| now.duration_since(**at) <= IDLE_AFTER).map(|(session, _)| session.clone()).collect();
    let resting = |failed: Option<Instant>| failed.is_some_and(|at| now.duration_since(at) < RETRY_AFTER);
    // When a shell is not the size its view has room for (a new one, or the
    // tab resized), read at once, at most every `REFIT_AFTER` for a size and
    // shell: a tmux that cannot resize, or a shell another terminal keeps
    // smaller, is not asked over and over.
    let refit = |session: &str| {
        if resting(state.capture_failed) {
            return None;
        }
        let size = fit_of(state, session)?;
        let shown = state.live.get(session).map(|screen| screen.size);
        let asked = state.fitted.get(session).is_some_and(|(at_size, at)| *at_size == size && now.duration_since(*at) < REFIT_AFTER);
        (shown != Some(size) && !asked).then_some(size)
    };
    let typed_into = sends.first().map(|(session, _)| session.as_str());
    // Of the shells on screen, the one to read now: the one just typed
    // into, then one just shown or resized, then the one read longest ago
    // once it is due; one a trip, the others on the next.
    let capture = watched
        .iter()
        .filter_map(|session| {
            let soon = state.capture_now.contains(session) || refit(session).is_some();
            let due = !resting(state.capture_failed)
                && state.live.get(session).is_none_or(|screen| {
                    let busy = state.busy_until.get(session).is_some_and(|until| now < *until);
                    now.duration_since(screen.at) >= if busy { CAPTURE_BUSY } else { CAPTURE_EVERY }
                });
            let rank = if typed_into == Some(session.as_str()) { 0 } else if soon { 1 } else if due { 2 } else { return None };
            Some((rank, state.live.get(session).map(|screen| screen.at), session.clone()))
        })
        .min()
        .map(|(_, _, session)| session);
    let history = state
        .history_wanted
        .iter()
        .find(|session| !state.history.contains_key(*session) && watched.contains(session) && !resting(state.history_failed))
        .cloned();
    let every = if state.list_error.is_some() { LIST_AFTER_ERROR } else { LIST_EVERY };
    let list = state.listed.is_none_or(|at| now.duration_since(at) >= every);
    if sends.is_empty() && capture.is_none() && history.is_none() && !list {
        return None;
    }
    let refit = capture.as_deref().and_then(refit);
    if let Some(session) = &capture {
        state.capture_now.remove(session);
        if let Some(size) = refit {
            state.fitted.insert(session.clone(), (size, now));
        }
    }
    // Every read says the size its views have room for (`fit_script`).
    let resize = capture.as_deref().and_then(|session| fit_of(state, session));
    Some(Work { sends, capture, resize, history, list })
}

/// A read of the shell on screen, kept: whether it shows anything new (a
/// read that finds the same screen draws nothing). One that changed keeps
/// the reads quick for a while: output is streaming.
fn settle_live(state: &mut State, session: &str, read: Result<(Screen, bool), String>, typed: bool) -> bool {
    if !state.watched.contains_key(session) {
        return false;
    }
    match read {
        Ok((screen, typing_failed)) => {
            let same = state.live.get(session).is_some_and(|live| {
                live.session == screen.session
                    && live.lines == screen.lines
                    && live.cursor == screen.cursor
                    && live.size == screen.size
                    && live.dead == screen.dead
                    && live.cursor_shown == screen.cursor_shown
                    && live.cwd == screen.cwd
            });
            if !same {
                state.busy_until.insert(session.to_string(), screen.at + BUSY_FOR);
            }
            state.live.insert(session.to_string(), screen);
            state.capture_failed = None;
            let errors = |state: &State| (state.error.contains_key(session), state.error.contains_key(""));
            let had = errors(state);
            if typed && typing_failed {
                state.error.insert(session.to_string(), "typing: the keys could not be sent".into());
            } else if typed || state.error.get(session).is_some_and(|error| !error.starts_with("typing:")) {
                state.error.remove(session);
            }
            // A read that worked: the machine answers again.
            state.error.remove("");
            !same || had != errors(state)
        }
        Err(error) => {
            state.capture_failed = Some(Instant::now());
            state.error.insert(session.to_string(), if typed { format!("typing: {error}") } else { error });
            true
        }
    }
}

fn run() {
    loop {
        let work = {
            let mut guard = state();
            let state = guard.get_or_insert_with(State::default);
            match next_work(state, Instant::now()) {
                Some(work) => work,
                None => {
                    let _ = WAKE.wait_timeout(guard, Duration::from_millis(100));
                    continue;
                }
            }
        };
        // Typed into the netbox on an attached terminal: s0ma hears of it.
        let forward = with(|state| std::mem::take(&mut state.netbox_forward));
        let mut sessions: Vec<String> = forward.iter().map(|(session, _)| session.clone()).collect();
        sessions.dedup();
        for session in sessions {
            let typed: Vec<Input> = forward.iter().filter(|(to, _)| *to == session).map(|(_, input)| input.clone()).collect();
            crate::tabs::netbox_typed(session, typed);
        }
        // Typed into the shell on screen: the keys and the screen they
        // made, in one trip to the VM.
        let together = !work.sends.is_empty()
            && work.capture.as_ref().is_some_and(|session| work.sends.iter().all(|(to, _)| to == session));
        let typed_into = work.sends.first().map(|(session, _)| session.clone());
        if let Some(session) = &typed_into {
            with(|state| state.busy_until.insert(session.clone(), Instant::now() + BUSY_FOR));
        }
        if let Some(session) = typed_into.filter(|_| !together) {
            let result = send_now(&work.sends);
            with(|state| match result {
                Ok(()) => state.error.remove(&session),
                Err(error) => state.error.insert(session, error),
            });
            changed();
        }
        if let Some(session) = work.capture {
            let read = if together {
                match send_script(&work.sends) {
                    Ok(typed) => read_screen_after(&session, false, work.resize, Some(&typed)),
                    Err(error) => Err(error),
                }
            } else {
                read_screen_after(&session, false, work.resize, None)
            };
            let fresh = with(|state| settle_live(state, &session, read, together));
            if fresh {
                changed();
            }
        }
        if let Some(session) = work.history {
            let read = read_screen(&session, true, None);
            with(|state| {
                if state.history_wanted.contains(&session) {
                    match read {
                        Ok(screen) => {
                            state.history.insert(session, screen);
                            state.history_failed = None;
                        }
                        Err(error) => {
                            state.history_failed = Some(Instant::now());
                            state.error.insert(session, error);
                        }
                    }
                }
            });
            changed();
        }
        if work.list {
            let read = list_sessions();
            // Drawn again only when the list (or its error) changed.
            let fresh = with(|state| {
                state.listed = Some(Instant::now());
                match read {
                    Ok((sessions, netboxes)) => {
                        let fresh = state.sessions != sessions || state.netboxes != netboxes || state.list_error.is_some();
                        state.sessions = sessions;
                        state.netboxes = netboxes;
                        state.list_error = None;
                        fresh
                    }
                    Err(error) => {
                        let fresh = state.list_error.as_ref() != Some(&error);
                        state.list_error = Some(error);
                        fresh
                    }
                }
            });
            if fresh {
                changed();
            }
        }
    }
}

fn guest(command: &str) -> Result<String, String> {
    let result = crate::vm::run_shell_helper(command).map_err(|error| error.to_string())?;
    match result.exit_code {
        Some(0) => Ok(result.stdout),
        Some(44) => Err("this shell is not open (closed, or not opened yet)".into()),
        code => Err(format!(
            "the VM answered {}: {}",
            code.map_or_else(|| "with no exit code".to_string(), |code| format!("exit {code}")),
            result.stderr.trim()
        )),
    }
}

/// The shells open in the VM, in tab order, and the netboxes among them.
fn list_sessions() -> Result<(Vec<String>, Vec<crate::netboxes::Netbox>), String> {
    let out = guest(crate::netboxes::LIST)?;
    let (sessions, netboxes) = crate::netboxes::parse(&out);
    Ok((order_sessions(sessions, &netboxes), netboxes))
}

/// The shells, which are the netboxes: the first (open or not: looking at it
/// opens it), the operator's and s0ma's in their own order, then subagents'.
/// Any other tmux session (a private shell from before every shell was
/// shared) is left out.
fn order_sessions(_sessions: Vec<String>, netboxes: &[crate::netboxes::Netbox]) -> Vec<String> {
    let (theirs, ours): (Vec<_>, Vec<_>) = netboxes.iter().partition(|netbox| netbox.is_workers());
    ours.iter().chain(theirs.iter()).map(|netbox| crate::netboxes::session_of(&netbox.name)).collect()
}

fn read_screen(session: &str, history: bool, resize: Option<(usize, usize)>) -> Result<Screen, String> {
    read_screen_after(session, history, resize, None).map(|(screen, _)| screen)
}

/// Netboxes checked against these settings this run (`upgrade_script`):
/// once each, and again only while the check found them busy.
static UPGRADED: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn upgraded(session: &str) -> bool {
    UPGRADED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).iter().any(|done| done == session)
}

fn mark_upgraded(session: &str) {
    let mut done = UPGRADED.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if !done.iter().any(|seen| seen == session) {
        done.push(session.to_string());
    }
}

/// The netbox has been opened (and set up) by a read this run: later reads
/// only check that it is there. s0ma's netbox tools still set it up.
static NETBOX_READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn netbox_present() -> String {
    if NETBOX_READY.load(Ordering::SeqCst) {
        format!("tmux has-session -t s0ma-{0} 2>/dev/null || {{ {1}; }}", crate::netboxes::DEFAULT_SESSION, crate::vm::netbox_open_script())
    } else {
        crate::vm::netbox_open_script()
    }
}

const TYPING_FAILED: &str = "__S0MA_TYPING_FAILED__";

/// A view's word for the size it has, kept on the shell's tmux session for
/// this long (each read of the shell says it again).
const FIT_FRESH_SECS: u64 = 6;

/// The shell made the size of the smallest view showing it: each view (this
/// terminal, an attached one) leaves its size on the tmux session
/// (`@s0ma_fit_<pid>`, with when), and the window is resized to the
/// smallest of those said lately, as tmux sizes a window shared by clients.
/// Two terminals on one shell then agree on it instead of resizing it to
/// each one's own in turn. Words gone stale are taken off. `resize-window`
/// came with tmux 2.9; an older one keeps its size.
fn fit_script(target: &str, (columns, rows): (usize, usize)) -> String {
    let me = std::process::id();
    format!(
        "now=$(date +%s); \
         tmux set-option -t {target} @s0ma_fit_{me} {columns},{rows},$now >/dev/null 2>&1; \
         fit=$(tmux show-options -t {target} 2>/dev/null | awk -v now=\"$now\" -v t={target} '\
             $1 ~ /^@s0ma_fit_/ {{ v = $2; gsub(/\"/, \"\", v); split(v, f, \",\"); \
                 if (now - f[3] > {FIT_FRESH_SECS}) {{ system(\"tmux set-option -u -t \" t \" \" $1 \" >/dev/null 2>&1\"); next }} \
                 if (w == \"\" || f[1] + 0 < w + 0) w = f[1]; if (h == \"\" || f[2] + 0 < h + 0) h = f[2] }} \
             END {{ if (w != \"\") print w \"x\" h }}'); \
         fit=${{fit:-{columns}x{rows}}}; \
         [ \"$(tmux display-message -p -t {target}:0.0 '#{{pane_width}}x#{{pane_height}}')\" = \"$fit\" ] \
             || tmux resize-window -t {target} -x ${{fit%x*}} -y ${{fit#*x}} >/dev/null 2>&1; "
    )
}

/// The screen, after `typed` (a `send_script`) in the same guest command:
/// a key and what it did on screen in one trip. Whether the typing failed.
fn read_screen_after(
    session: &str,
    history: bool,
    resize: Option<(usize, usize)>,
    typed: Option<&str>,
) -> Result<(Screen, bool), String> {
    let target = crate::vm::tmux_session_name(session).map_err(|error| error.to_string())?;
    let range = if history { format!(" -S -{HISTORY_LINES}") } else { String::new() };
    // The netbox is opened by looking at it; any other shell has to be there.
    let upgrade = crate::netboxes::name_of(session).filter(|name| *name != crate::netboxes::DEFAULT && !upgraded(session)).map(str::to_string);
    let present = if session == crate::netboxes::DEFAULT_SESSION {
        netbox_present()
    } else if let Some(name) = &upgrade {
        // A netbox from an older s0ma: brought up to date the first time
        // it is looked at (once idle).
        format!("tmux has-session -t {target} 2>/dev/null || exit 44; {}", crate::netboxes::upgrade_script(name))
    } else {
        format!("tmux has-session -t {target} 2>/dev/null || exit 44")
    };
    let resize = resize.map_or(String::new(), |size| fit_script(&target, size));
    let typed = typed.map_or(String::new(), |typed| format!("{{ {typed}; }} >/dev/null || echo {TYPING_FAILED}; "));
    let command = format!(
        "{present}; {typed}{resize}\
         tmux display-message -p -t {target}:0.0 \
             '#{{cursor_x}}|#{{cursor_y}}|#{{pane_width}}|#{{pane_height}}|#{{pane_dead}}|#{{pane_current_command}}|#{{cursor_flag}}|#{{pane_current_path}}'; \
         printf '\\n{MARK}\\n'; \
         tmux capture-pane -p -e -t {target}:0.0{range}; \
         printf '{END}\\n'"
    );
    let out = guest(&command)?;
    // The typing's failure, a line before the screen's own.
    let (failed, out) = match out.strip_prefix(&format!("{TYPING_FAILED}\n")) {
        Some(rest) => (true, rest),
        None => (false, out.as_str()),
    };
    let screen = parse_screen(session, out)?;
    if session == crate::netboxes::DEFAULT_SESSION {
        NETBOX_READY.store(true, Ordering::SeqCst);
    }
    if upgrade.is_some() && matches!(screen.command.as_str(), "bash" | "sh" | "dash" | "zsh" | "fish") {
        mark_upgraded(session);
    }
    Ok((screen, failed))
}

fn parse_screen(session: &str, out: &str) -> Result<Screen, String> {
    let (meta, body) = out.split_once(&format!("\n{MARK}\n")).ok_or("the shell's screen could not be read")?;
    // Up to the end mark, when there is one (a test's screen may have none).
    let body = body.split_once(END).map_or(body, |(body, _)| body);
    // Split by `|`, not a tab: tmux 3.5 writes a tab in its output as `_`,
    // which once left every field unreadable (the cursor drawn in the
    // bottom-left corner, no folder in the footer). The folder is last, so a
    // `|` in it stays in it.
    let fields: Vec<&str> = meta.trim_end_matches('\n').splitn(8, '|').collect();
    let number = |index: usize| fields.get(index).and_then(|field| field.trim().parse::<usize>().ok()).unwrap_or(0);
    let (column, row, width, height) = (number(0), number(1), number(2), number(3));
    let mut lines = carry_styles(body.lines());
    // The pane's rows are all there, blank ones too: the cursor's row counts
    // from the top of the pane, which is the last `height` of them.
    while lines.len() < height {
        lines.push(String::new());
    }
    let top = lines.len().saturating_sub(height.max(1));
    Ok(Screen {
        session: session.to_string(),
        cursor: (column, top + row),
        size: (width, height),
        dead: fields.get(4).is_some_and(|field| field.trim() == "1"),
        command: fields.get(5).unwrap_or(&"").trim().to_string(),
        // Shown unless the pane says hidden (an older capture says nothing).
        cursor_shown: fields.get(6).is_none_or(|field| field.trim() != "0"),
        cwd: fields.get(7).unwrap_or(&"").trim().to_string(),
        lines,
        at: Instant::now(),
    })
}

/// The rows, each starting with the colours in force where the row before
/// ended. `capture-pane -e` writes a colour only where it changes, counting
/// from the row before: a command or output that wraps onto the next row has
/// its colour on the first row only, and drawn row by row the rest came out
/// in the default colour (white).
fn carry_styles<'a>(rows: impl Iterator<Item = &'a str>) -> Vec<String> {
    // The colour codes since the last reset, in order.
    let mut held: Vec<String> = Vec::new();
    rows.map(|row| {
        let carried = format!("{}{row}", held.concat());
        let mut rest = row;
        while let Some(at) = rest.find("\x1b[") {
            let after = &rest[at + 2..];
            let Some(end) = after.find(|c: char| ('\x40'..='\x7e').contains(&c)) else { break };
            if after[end..].starts_with('m') {
                let params = &after[..end];
                let sequence = format!("\x1b[{params}m");
                if params.is_empty() || params == "0" {
                    held.clear();
                } else if let Some(more) = params.strip_prefix("0;") {
                    held.clear();
                    held.push(format!("\x1b[{more}m"));
                } else {
                    held.push(sequence);
                }
                // Bounded: a run of colour changes with no reset keeps the latest.
                if held.len() > 32 {
                    held.drain(..held.len() - 32);
                }
            }
            rest = &after[end + 1..];
        }
        carried
    })
    .collect()
}


/// The guest command that types `inputs`, in order; per session.
fn send_script(inputs: &[(String, Input)]) -> Result<String, String> {
    let mut steps = Vec::new();
    let mut checked = Vec::new();
    for (session, input) in inputs {
        let target = crate::vm::tmux_session_name(session).map_err(|error| error.to_string())?;
        if !checked.contains(&target) {
            steps.push(if session == crate::netboxes::DEFAULT_SESSION {
                // Typing into the netbox opens it.
                format!("{{ {}; }}", netbox_present())
            } else {
                format!("tmux has-session -t {target} 2>/dev/null || exit 44")
            });
            checked.push(target.clone());
        }
        let hex = |bytes: &[u8]| bytes.iter().map(|byte| format!("{byte:02x}")).collect::<Vec<_>>().join(" ");
        steps.push(match input {
            // As bytes: tmux reads a `;` that ends an argument as the end of
            // a command, so literal text would lose it.
            Input::Text(text) => format!("tmux send-keys -t {target}:0.0 -H {}", hex(text.as_bytes())),
            Input::Paste(text) => format!(
                "printf %s {} | tmux load-buffer -b s0ma-view-paste - && tmux paste-buffer -p -d -b s0ma-view-paste -t {target}:0.0",
                crate::util::shell_quote(text)
            ),
            Input::Key(name) => format!("tmux send-keys -t {target}:0.0 {}", crate::util::shell_quote(name)),
            Input::Bytes(bytes) => format!("tmux send-keys -t {target}:0.0 -H {}", hex(bytes)),
        });
    }
    Ok(steps.join(" && "))
}

fn send_now(inputs: &[(String, Input)]) -> Result<(), String> {
    guest(&send_script(inputs)?).map(|_| ()).map_err(|error| format!("typing: {error}"))
}

/// Nothing typed into any netbox, for tests.
#[cfg(test)]
pub(crate) fn forget_typing_for_test() {
    with(|state| state.typing.clear());
}

/// What is waiting to be typed, for tests.
#[cfg(test)]
pub(crate) fn take_outbox_for_test() -> Vec<(String, Input)> {
    with(|state| state.outbox.drain(..).collect())
}

/// A shell as if just read, for tests.
#[cfg(test)]
pub(crate) fn set_live_for_test(screen: Screen) {
    with(|state| {
        state.watched.insert(screen.session.clone(), Instant::now());
        state.live.insert(screen.session.clone(), screen);
    });
}

#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> Option<Input> {
        key_input(&KeyEvent::new(code, modifiers))
    }

    #[test]
    fn keys_go_to_the_shell_as_tmux_sends_them() {
        assert_eq!(key(KeyCode::Char('l'), KeyModifiers::NONE), Some(Input::Text("l".into())));
        assert_eq!(key(KeyCode::Char('L'), KeyModifiers::SHIFT), Some(Input::Text("L".into())));
        assert_eq!(key(KeyCode::Char('c'), KeyModifiers::CONTROL), Some(Input::Bytes(vec![3])), "Ctrl+C interrupts the shell's program");
        assert_eq!(key(KeyCode::Char('d'), KeyModifiers::CONTROL), Some(Input::Bytes(vec![4])));
        assert_eq!(key(KeyCode::Char(' '), KeyModifiers::CONTROL), Some(Input::Bytes(vec![0])));
        assert_eq!(key(KeyCode::Char('b'), KeyModifiers::ALT), Some(Input::Bytes(vec![0x1b, b'b'])));
        assert_eq!(key(KeyCode::Enter, KeyModifiers::NONE), Some(Input::Key("Enter".into())));
        assert_eq!(key(KeyCode::Backspace, KeyModifiers::NONE), Some(Input::Key("BSpace".into())));
        assert_eq!(key(KeyCode::Up, KeyModifiers::NONE), Some(Input::Key("Up".into())));
        assert_eq!(key(KeyCode::Left, KeyModifiers::CONTROL), Some(Input::Key("C-Left".into())));
        assert_eq!(key(KeyCode::PageUp, KeyModifiers::NONE), Some(Input::Key("PPage".into())));
        assert_eq!(key(KeyCode::F(5), KeyModifiers::NONE), Some(Input::Key("F5".into())));
        assert_eq!(key(KeyCode::Esc, KeyModifiers::NONE), Some(Input::Key("Escape".into())));
        assert_eq!(key(KeyCode::Char('é'), KeyModifiers::CONTROL), None);
    }

    #[test]
    fn what_is_typed_is_quoted_for_the_guest_and_sent_in_order() {
        let script = send_script(&[
            ("main".into(), Input::Text("a;".into())),
            ("main".into(), Input::Key("Enter".into())),
            ("main".into(), Input::Bytes(vec![0x1b, b'b'])),
            ("main".into(), Input::Paste("echo 'hi' $HOME;\nls".into())),
        ])
        .unwrap();
        assert_eq!(
            script,
            "tmux has-session -t s0ma-main 2>/dev/null || exit 44 && \
             tmux send-keys -t s0ma-main:0.0 -H 61 3b && \
             tmux send-keys -t s0ma-main:0.0 'Enter' && \
             tmux send-keys -t s0ma-main:0.0 -H 1b 62 && \
             printf %s 'echo '\\''hi'\\'' $HOME;\nls' | tmux load-buffer -b s0ma-view-paste - && \
             tmux paste-buffer -p -d -b s0ma-view-paste -t s0ma-main:0.0"
        );
        assert!(send_script(&[("../x".into(), Input::Text("a".into()))]).is_err(), "names are checked as the tools check them");
    }

    #[test]
    fn a_read_that_finds_the_same_screen_draws_nothing_and_a_change_reads_quickly() {
        let now = Instant::now();
        let screen = |text: &str, at| Screen {
            session: "main".into(), lines: vec![text.into()], cursor: (0, 0), size: (80, 1), dead: false,
            cursor_shown: true, command: String::new(), cwd: String::new(), at,
        };
        let mut state = State { watched: BTreeMap::from([("main".into(), now)]), touched: Some(now), listed: Some(now), ..State::default() };
        assert!(settle_live(&mut state, "main", Ok((screen("a", now), false)), false), "the first is new");
        let later = now + BUSY_FOR + Duration::from_millis(1);
        assert!(!settle_live(&mut state, "main", Ok((screen("a", later), false)), false), "the same: nothing to draw");
        // Quiet: read at the slow pace.
        assert!(next_work(&mut state, later + CAPTURE_BUSY).is_none());
        assert!(next_work(&mut state, later + CAPTURE_EVERY).is_some());
        // Changing: read again soon.
        assert!(settle_live(&mut state, "main", Ok((screen("b", later), false)), false));
        assert!(next_work(&mut state, later + CAPTURE_BUSY).is_some_and(|work| work.capture.is_some()));
        // Typing that failed says so, on a screen read in the same trip.
        assert!(settle_live(&mut state, "main", Ok((screen("b", later), true)), true));
        assert!(state.error.get("main").is_some_and(|error| error.starts_with("typing:")));
        // Another shell's read is not kept.
        assert!(!settle_live(&mut state, "other", Ok((screen("c", later), false)), false));
    }

    #[test]
    fn a_screen_reads_with_its_cursor_where_the_pane_has_it() {
        let out = format!("4|1|120|3|0|bash|1|/workspace\n{MARK}\n$ ls\na  b\n$ \n");
        let screen = parse_screen("main", &out).unwrap();
        assert_eq!(screen.lines, ["$ ls", "a  b", "$ "]);
        assert_eq!((screen.cursor, screen.size, screen.dead), ((4, 1), (120, 3), false));
        assert_eq!((screen.command.as_str(), screen.cwd.as_str(), screen.cursor_shown), ("bash", "/workspace", true));
        // A program that hid its cursor.
        let hidden = parse_screen("main", &format!("0|0|80|2|0|less|0|/\n{MARK}\nx\n")).unwrap();
        assert!(!hidden.cursor_shown);
        // With history before the pane: the cursor's row counts from the pane's top.
        let out = format!("2|0|120|2|0|bash|1|/\n{MARK}\nold 1\nold 2\n$ \n\n");
        let screen = parse_screen("main", &out).unwrap();
        assert_eq!(screen.cursor, (2, 2));
        // Blank rows at the bottom that capture-pane left off still count.
        let screen = parse_screen("main", &format!("0|3|80|5|1|bash|1|/\n{MARK}\n$ exit\n")).unwrap();
        assert_eq!(screen.lines.len(), 5);
        assert!(screen.dead);
        assert!(parse_screen("main", "garbage").is_err());
    }

    #[test]
    fn a_coloured_line_that_wraps_keeps_its_colour_on_the_next_row() {
        // As tmux captures it: the colour once, where it starts.
        let out = format!("0|2|30|3|0|bash|1|/\n{MARK}\n\x1b[32maaaaaaaaaa\nbbbbbbbbbb\x1b[39m\n$ \x1b[34mecho long\nest\x1b[0m\nplain\n");
        let screen = parse_screen("main", &out).unwrap();
        assert_eq!(screen.lines[0], "\x1b[32maaaaaaaaaa");
        assert_eq!(screen.lines[1], "\x1b[32mbbbbbbbbbb\x1b[39m", "the green goes on");
        assert_eq!(screen.lines[2], "\x1b[32m\x1b[39m$ \x1b[34mecho long", "39 ends it, as the terminal reads it");
        assert_eq!(screen.lines[3], "\x1b[32m\x1b[39m\x1b[34mest\x1b[0m", "the operator's blue goes on");
        assert_eq!(screen.lines[4], "plain", "nothing after a reset");
        let reset_then = carry_styles(["\x1b[0;36mx", "y"].into_iter());
        assert_eq!(reset_then[1], "\x1b[36my");
    }

    #[test]
    fn the_shells_are_the_netboxes_main_first_and_subagents_last() {
        let listed = "s0ma-w1a2_19b0c_1|\ns0ma-build|\ns0ma-main|\ns0ma-nb-w-b3-1|3||worker:subagent-1-2-3-1\n\
                      s0ma-nb-tests|3||operator\ns0ma-w1a2_19b0c_0|\ns0ma-api|\n";
        let (sessions, netboxes) = crate::netboxes::parse(listed);
        assert_eq!(
            order_sessions(sessions, &netboxes),
            ["netbox", "nb-tests", "nb-w-b3-1"],
            "main is always there, first; other tmux sessions are not shells any more"
        );
        // Once, when it is open too.
        let (sessions, netboxes) = crate::netboxes::parse("s0ma-netbox|3\ns0ma-main|\n");
        assert_eq!(order_sessions(sessions, &netboxes), ["netbox"]);
    }

    #[test]
    fn every_read_of_the_shell_says_the_tabs_size_and_a_new_size_is_read_at_once() {
        let mut state = State::default();
        let now = Instant::now();
        let screen = |session: &str, size, at| Screen { session: session.into(), lines: vec![], cursor: (0, 0), size, dead: false, cursor_shown: true, command: String::new(), cwd: String::new(), at };
        let fit = |state: &mut State, session: &str, size, at| state.fit.insert(session.to_string(), vec![(size, at)]);
        state.touched = Some(now);
        state.listed = Some(now);
        state.watched = BTreeMap::from([("netbox".into(), now)]);
        state.live.insert("netbox".into(), screen("netbox", (120, 40), now));
        fit(&mut state, "netbox", (140, 70), now);
        let work = next_work(&mut state, now).unwrap();
        assert_eq!((work.capture.as_deref(), work.resize), (Some("netbox"), Some((140, 70))), "read at once, resized");
        // A tmux that kept its size (or another terminal keeps smaller) is
        // not read again straight away for it...
        assert!(next_work(&mut state, now).is_none());
        // ...but each read still says the size, which keeps the word fresh.
        assert_eq!(next_work(&mut state, now + CAPTURE_EVERY).unwrap().resize, Some((140, 70)));
        // After a while it is read at once again, and so is a new size.
        let later = now + REFIT_AFTER;
        state.touched = Some(later);
        state.watched = BTreeMap::from([("netbox".into(), later)]);
        state.listed = Some(later);
        state.live.insert("netbox".into(), screen("netbox", (120, 40), later));
        assert_eq!(next_work(&mut state, later).unwrap().resize, Some((140, 70)));
        fit(&mut state, "netbox", (100, 30), later);
        assert_eq!(next_work(&mut state, later).unwrap().resize, Some((100, 30)));
        // The right size already, and just read: nothing to do.
        state.live.insert("netbox".into(), screen("netbox", (100, 30), later));
        state.fitted.clear();
        assert!(next_work(&mut state, later).is_none());
        // Another shell on screen is fitted too, to its own view's size.
        state.watched.insert("main".into(), later);
        fit(&mut state, "main", (60, 30), later);
        assert_eq!(next_work(&mut state, later).unwrap().resize, Some((60, 30)));
        // An unreachable VM is not asked to resize while reads rest.
        state.watched = BTreeMap::from([("api".into(), later)]);
        fit(&mut state, "api", (60, 30), later);
        state.capture_failed = Some(later);
        assert!(next_work(&mut state, later).is_none());
    }

    /// Two panes, a shell in each: both are read, each keeps its screen,
    /// its size and its history; one shell in both takes the smaller size.
    #[test]
    fn two_shells_on_screen_are_each_read_and_kept() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let screen = |session: &str, at| Screen { session: session.into(), lines: vec![session.into()], cursor: (0, 0), size: (59, 30), dead: false, cursor_shown: true, command: String::new(), cwd: String::new(), at };
        // Only these two on screen (other tests draw shells too).
        let ours = |state: &mut State| state.watched.retain(|session, _| session == "netbox" || session == "nb-tests");
        with(|state| *state = State::default());
        // As two panes draw, frame after frame.
        for _ in 0..3 {
            watch("netbox");
            fit("netbox", 59, 30);
            watch("nb-tests");
            fit("nb-tests", 60, 30);
        }
        let now = Instant::now();
        let read = with(|state| {
            state.touched = Some(now);
            state.listed = Some(now);
            ours(state);
            assert!(state.capture_now.contains("netbox") && state.capture_now.contains("nb-tests"), "both read at once, the first time");
            let first = next_work(state, now).and_then(|work| work.capture).unwrap();
            assert!(settle_live(state, &first, Ok((screen(&first, now), false)), false));
            let second = next_work(state, now).and_then(|work| work.capture).unwrap();
            assert!(settle_live(state, &second, Ok((screen(&second, now), false)), false));
            (first, second)
        });
        assert_ne!(read.0, read.1, "the other on the next trip");
        // Each shown as read, however the panes take turns watching.
        watch("netbox");
        watch("nb-tests");
        assert_eq!(live("netbox").unwrap().lines, ["netbox"]);
        assert_eq!(live("nb-tests").unwrap().lines, ["nb-tests"]);
        // Read again in turn, the one read longest ago first.
        with(|state| {
            state.live.get_mut("nb-tests").unwrap().at = now - CAPTURE_EVERY * 3;
            state.live.get_mut("netbox").unwrap().at = now - CAPTURE_EVERY * 2;
            state.fitted.clear();
            ours(state);
            state.capture_now.clear();
            state.live.get_mut("nb-tests").unwrap().size = (60, 30);
            assert_eq!(next_work(state, now).and_then(|work| work.capture).as_deref(), Some("nb-tests"));
        });
        // Scrolled back in one: the other keeps following.
        want_history("netbox", true);
        want_history("nb-tests", false);
        assert!(with(|state| state.history_wanted.contains("netbox")));
        assert_eq!(with(|state| { ours(state); next_work(state, now).and_then(|work| work.history) }).as_deref(), Some("netbox"));
        want_history("netbox", false);
        // One shell in both panes: the smaller pane's size, without
        // going back and forth.
        for _ in 0..3 {
            fit("netbox", 59, 30);
            fit("netbox", 60, 31);
        }
        assert_eq!(with(|state| fit_of(state, "netbox")), Some((59, 30)));
        // An error reading one is that one's.
        with(|state| {
            settle_live(state, "nb-tests", Err("gone".into()), false);
            state.capture_failed = None;
        });
        assert_eq!(error("nb-tests").as_deref(), Some("gone"));
        assert_eq!(error("netbox"), None);
        with(|state| *state = State::default());
    }

    #[test]
    fn a_shell_two_terminals_show_takes_the_smaller_ones_size() {
        // The script each read runs, against a real tmux when there is one.
        let Ok(found) = std::process::Command::new("tmux").arg("-V").output() else { return };
        if !found.status.success() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("s0ma-fit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sh = |script: &str| {
            let out = std::process::Command::new("sh").arg("-c").arg(script).env("TMUX_TMPDIR", &dir).env_remove("TMUX").output().unwrap();
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        sh("tmux -f /dev/null new-session -d -s s0ma-fit -x 120 -y 40 'sleep 30'; tmux set-option -t s0ma-fit status off");
        let pane = || sh("tmux display-message -p -t s0ma-fit:0.0 '#{pane_width}x#{pane_height}'");
        // Another terminal, smaller, spoke lately; one gone long ago did too.
        sh("now=$(date +%s); tmux set-option -t s0ma-fit @s0ma_fit_1 90,23,$now; tmux set-option -t s0ma-fit @s0ma_fit_2 40,10,$((now - 60))");
        sh(&fit_script("s0ma-fit", (110, 31)));
        assert_eq!(pane(), "90x23", "the smaller of the two fresh words");
        assert!(!sh("tmux show-options -t s0ma-fit").lines().any(|line| line.starts_with("@s0ma_fit_2 ")), "the stale word is taken off");
        // The smaller one stops speaking: this one's size, once it is stale.
        sh("tmux set-option -t s0ma-fit @s0ma_fit_1 90,23,$(($(date +%s) - 60))");
        sh(&fit_script("s0ma-fit", (110, 31)));
        assert_eq!(pane(), "110x31");
        sh("tmux kill-server");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_reader_rests_without_a_view_and_batches_what_was_typed() {
        let mut state = State::default();
        let now = Instant::now();
        assert!(next_work(&mut state, now).is_none(), "no view: nothing to do");
        state.touched = Some(now);
        state.watched = BTreeMap::from([("main".into(), now)]);
        state.outbox.push_back(("main".into(), Input::Text("ls".into())));
        let work = next_work(&mut state, now).unwrap();
        assert_eq!(work.sends.len(), 1);
        assert_eq!(work.capture.as_deref(), Some("main"), "read again at once after typing");
        assert!(work.list);
        state.listed = Some(now);
        state.live.insert("main".into(), Screen { session: "main".into(), lines: vec![], cursor: (0, 0), size: (80, 24), dead: false, cursor_shown: true, command: String::new(), cwd: String::new(), at: now });
        assert!(next_work(&mut state, now).is_none(), "just read: nothing due");
        assert!(next_work(&mut state, now + CAPTURE_EVERY).unwrap().capture.is_some());
        state.history_wanted.insert("main".into());
        assert_eq!(next_work(&mut state, now).unwrap().history.as_deref(), Some("main"));
        // The view went away: what was typed for it still goes, nothing is read.
        state.outbox.push_back(("main".into(), Input::Text("x".into())));
        let later = now + IDLE_AFTER + Duration::from_secs(1);
        let work = next_work(&mut state, later).unwrap();
        assert_eq!(work.sends.len(), 1);
        assert!(work.capture.is_none() && work.history.is_none() && !work.list);
        assert!(next_work(&mut state, later).is_none());
        // A session's keys at a time: one failing never loses another's.
        state.touched = Some(later);
        state.watched = BTreeMap::from([("main".into(), later)]);
        state.listed = Some(later);
        state.history_wanted.clear();
        state.live.insert("main".into(), Screen { session: "main".into(), lines: vec![], cursor: (0, 0), size: (80, 24), dead: false, cursor_shown: true, command: String::new(), cwd: String::new(), at: later });
        state.outbox.push_back(("main".into(), Input::Text("a".into())));
        state.outbox.push_back(("build".into(), Input::Text("b".into())));
        assert_eq!(next_work(&mut state, later).unwrap().sends, [("main".to_string(), Input::Text("a".into()))]);
        assert_eq!(next_work(&mut state, later).unwrap().sends, [("build".to_string(), Input::Text("b".into()))]);
        // A shell that could not be read is not asked again at once.
        state.live.clear();
        state.capture_failed = Some(later);
        assert!(next_work(&mut state, later).is_none());
        assert!(next_work(&mut state, later + RETRY_AFTER).unwrap().capture.is_some());
        state.history_wanted.insert("main".into());
        state.history_failed = Some(later);
        state.capture_failed = Some(later);
        assert!(next_work(&mut state, later).is_none(), "nor its history");
    }

    #[test]
    fn long_text_and_pastes_go_in_pieces_cut_between_characters() {
        let text = "é".repeat(5_000);
        let parts = pieces(&text, SEND_CHARS);
        assert!(parts.iter().all(|part| part.len() <= SEND_CHARS));
        assert_eq!(parts.concat(), text);
        assert_eq!(pieces("", 10), [""]);
    }

    /// The operator's typing in a netbox becomes the lines he ran, for s0ma's next
    /// read, each netbox's its own; anyone else's shell is not noted. While
    /// he types, or has a line half-typed, s0ma keeps out.
    #[test]
    fn what_operator_types_in_a_netbox_is_kept_as_lines() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        forget_typing_for_test();
        assert_eq!(operator_busy("netbox"), None);
        send("netbox", Input::Text("source .venv/bin/activat".into()));
        assert_eq!(operator_busy("netbox"), Some("The operator is typing in it"));
        send("netbox", Input::Key("BSpace".into()));
        send("netbox", Input::Text("te".into()));
        send("netbox", Input::Key("Enter".into()));
        send("netbox", Input::Text("pip install -r req".into()));
        send("netbox", Input::Bytes(vec![0x03]));
        send("main", Input::Text("not the netbox".into()));
        send("nb-build", Input::Text("cargo b".into()));
        send("netbox", Input::Text("ls".into()));
        assert_eq!(
            take_netbox_typing("netbox"),
            ["source .venv/bin/activate", "pip install -r req[ctrl-c]", "ls (not entered yet)"]
        );
        assert!(take_netbox_typing("netbox").iter().all(|line| line.contains("not entered yet")), "taken once");
        assert_eq!(take_netbox_typing("nb-build"), ["cargo b (not entered yet)"], "its own");
        assert!(take_netbox_typing("main").is_empty());
        // A few seconds after his last key, s0ma may type there again.
        with(|state| state.typing.get_mut("netbox").unwrap().at = Some(Instant::now() - OPERATOR_TYPING));
        assert_eq!(operator_busy("netbox"), None);
        assert_eq!(operator_busy("nb-build"), Some("The operator is typing in it"), "each netbox its own");
        let _ = take_netbox_typing("netbox");
        // Typed on an attached terminal: sent here as typed, heard the same.
        let there = serde_json::to_string(&[Input::Text("make tes".into()), Input::Bytes(vec![0x7f]), Input::Text("st".into()), Input::Bytes(vec![b'\r'])]).unwrap();
        note_netbox("nb-build", &serde_json::from_str::<Vec<Input>>(&there).unwrap());
        assert_eq!(take_netbox_typing("nb-build"), ["cargo bmake test"]);
        let _ = take_outbox_for_test();
        forget_typing_for_test();
    }

}
