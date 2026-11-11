//! Approval mode (`/approval on`): nothing runs on s0ma's machine, and no
//! subagent starts, until the operator approves it.
//!
//! What waits for him:
//! - every command, s0ma's and its subagents', in every shell: `vm_exec`
//!   (in a shell or in the background), and what `shell_type` and
//!   `shell_key` would type into one;
//! - the goal tests a board runs (`vm::GoalTest`);
//! - each `subagent_spawn`, with the tasks it would start.
//!
//! The call that asks waits here (`ask`): the page says what is asked, and
//! the tab bar shows `[!] approve` while anything waits. Alt+A pressed twice
//! approves the oldest request (the first press says what it would approve;
//! the second, within `TWICE`, does it); Alt+X declines it. The keys work on
//! the page, during a turn, in every tab, and in a terminal attached from
//! elsewhere, which is sent the requests (`mirror_to`, `apply`) and answers
//! through `tabs::act`. A request not answered in `WAIT` is declined, as is
//! one whose call is stopped with /kill. A command approved here runs in its
//! shell without that shell's own confirm prompt: it has been confirmed.
//!
//! What the operator starts himself (a board sent to verification starting
//! its verifier) is his already (`by_operator`).
//!
//! Kept in `data/approval.json`, so it holds across a restart. A file that
//! can't be read keeps approval on: when in doubt, s0ma asks.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// A request not answered in this long is declined.
pub const WAIT: Duration = Duration::from_secs(600);
/// The second press of Alt+A counts within this of the first.
pub const TWICE: Duration = Duration::from_secs(3);

/// One request waiting for the operator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pending {
    pub id: u64,
    /// `s0ma`, or the subagent's name.
    pub who: String,
    /// What it would do, on one line: `run in main: ls -la`.
    pub what: String,
    /// All of it: the whole command, every task.
    pub detail: String,
}

#[derive(Default)]
struct State {
    pending: Vec<Pending>,
    /// Answers given, by id, for the call waiting on them.
    answers: Vec<(u64, bool)>,
    next: u64,
    /// Alt+A pressed once: on which request, and when.
    armed: Option<(u64, Instant)>,
}

static ON: AtomicBool = AtomicBool::new(false);
static PATH: OnceLock<PathBuf> = OnceLock::new();
static STATE: Mutex<State> = Mutex::new(State { pending: Vec::new(), answers: Vec::new(), next: 1, armed: None });
static ANSWERED: Condvar = Condvar::new();
/// Where the waiting list goes when it changes (the attached terminals).
static MIRROR: OnceLock<Box<dyn Fn(Vec<Pending>) + Send + Sync>> = OnceLock::new();

thread_local! {
    static OPERATOR: Cell<bool> = const { Cell::new(false) };
}

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Read the mode as it was left, from beside the archive.
pub fn init(archive: &Path) {
    let path = archive.with_file_name("approval.json");
    ON.store(kept_on(&path), Ordering::SeqCst);
    let _ = PATH.set(path);
}

/// No file: off. One that says `"on": false`: off. Anything else: on.
fn kept_on(path: &Path) -> bool {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str::<serde_json::Value>(&text).map_or(true, |value| value["on"].as_bool() != Some(false)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => true,
    }
}

/// Whether approval mode is on.
pub fn enabled() -> bool {
    ON.load(Ordering::SeqCst)
}

/// Turn it on or off, and keep it. What to say.
pub fn set(on: bool) -> Result<String, String> {
    let was = enabled();
    ON.store(on, Ordering::SeqCst);
    if let Some(path) = PATH.get() {
        let body = serde_json::json!({"on": on, "at": chrono::Local::now().to_rfc3339()}).to_string();
        crate::util::write_atomic(path, body).map_err(|error| {
            format!("[approval] {} for now, but could not be kept for a restart: {error}", if on { "on" } else { "off" })
        })?;
    }
    let waiting = state().pending.len();
    let still = if !on && waiting > 0 { format!("; {waiting} already asked still wait{} for an answer", if waiting == 1 { "s" } else { "" }) } else { String::new() };
    Ok(match (was, on) {
        (true, true) => "[approval] already on".into(),
        (false, false) => format!("[approval] already off{still}"),
        (_, true) => "[approval] on: every command in every shell, s0ma's and its subagents', and every subagent it starts \
                      waits for you. alt+a twice approves, alt+x declines"
            .into(),
        (_, false) => format!("[approval] off: commands and subagents start without asking{still}"),
    })
}

/// `/approval`, `/approval status|on|off`, and the waiting list with
/// `/approval list`; `None` for anything else.
pub fn handle_command(command: &str) -> Option<Result<String, String>> {
    let words: Vec<&str> = command.split_whitespace().collect();
    match words.as_slice() {
        ["/approval"] | ["/approval", "status"] => Some(Ok(format!(
            "[approval] {}{}",
            if enabled() { "on (/approval off turns it off)" } else { "off (/approval on turns it on)" },
            match pending().len() {
                0 => String::new(),
                n => format!("; {n} waiting (/approval list)"),
            }
        ))),
        ["/approval", "on"] => Some(set(true)),
        ["/approval", "off"] => Some(set(false)),
        ["/approval", "list"] => Some(Ok(list_text())),
        ["/approval", ..] => Some(Err("usage: /approval [on|off|status|list]".into())),
        _ => None,
    }
}

fn list_text() -> String {
    let pending = pending();
    if pending.is_empty() {
        return "[approval] nothing waits".into();
    }
    let mut out = format!("[approval] {} waiting, oldest first (alt+a twice approves the first, alt+x declines it):", pending.len());
    for request in pending {
        out.push_str(&format!("\n  #{} {}: {}", request.id, request.who, request.what));
    }
    out
}

/// `work`, as the operator's own: nothing it starts asks him.
pub fn by_operator<T>(work: impl FnOnce() -> T) -> T {
    let before = OPERATOR.with(|operator| operator.replace(true));
    let out = work();
    OPERATOR.with(|operator| operator.set(before));
    out
}

/// Send the waiting list to `sink` whenever it changes. Once.
pub fn mirror_to(sink: Box<dyn Fn(Vec<Pending>) + Send + Sync>) {
    let _ = MIRROR.set(sink);
}

fn changed(pending: Vec<Pending>) {
    if let Some(sink) = MIRROR.get() {
        sink(pending);
    }
}

/// An attached terminal: the waiting list as the primary has it now.
pub fn apply(pending: Vec<Pending>) {
    let mut state = state();
    if state.armed.is_some_and(|(id, _)| !pending.iter().any(|request| request.id == id)) {
        state.armed = None;
    }
    state.pending = pending;
}

/// The requests waiting, oldest first.
pub fn pending() -> Vec<Pending> {
    state().pending.clone()
}

/// What fits on one line of a request. A short text is shown whole, its
/// lines joined by ` ⏎ `, so nothing in it hides on a line not shown; a
/// longer one shows its first line and how much more there is (all of it is
/// in the request's detail: Ctrl+O on the page).
pub const WHOLE_CHARS: usize = 400;

pub fn one_line(text: &str) -> String {
    let text = visible(text.trim());
    let lines: Vec<&str> = text.lines().collect();
    if text.chars().count() <= WHOLE_CHARS {
        return lines.join(" ⏎ ");
    }
    let first = lines.first().copied().unwrap_or_default().trim();
    let first: String = if first.chars().count() > 120 { format!("{}…", first.chars().take(119).collect::<String>()) } else { first.to_string() };
    let more = lines.len().saturating_sub(1);
    let rest = if more > 0 { format!("+{more} line{}, ", if more == 1 { "" } else { "s" }) } else { String::new() };
    format!("{first} ({rest}{} chars in all: ctrl+o shows it)", text.chars().count())
}

/// `text` with every control character but the line break and the tab
/// written out (`^[`, `^U`, `\u{7f}`…), so what is approved reads as what runs:
/// a carriage return or an escape sequence can't draw one command over
/// another, and a Ctrl+U typed into a shell can't hide what came before it.
pub fn visible(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\n' | '\t' => out.push(c),
            c if (c as u32) < 0x20 => {
                out.push('^');
                out.push(char::from_u32(c as u32 + 0x40).unwrap_or('?'));
            }
            c if c.is_control() || matches!(c, '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}' | '\u{feff}') => {
                out.push_str(&format!("\\u{{{:x}}}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// The tools that run code, or press keys where code runs, other than the
/// shell tools (`vm.rs` asks for those) and `subagent_spawn` (`subagent.rs`):
/// the desktop's input (it can drive a terminal; on the Xephyr backend, one on
/// the operator's own machine), JavaScript in the browser, a wake watch's
/// command, and every MCP server's tools. `None` for any other tool, or when
/// the operator approves it; otherwise the result to give the call.
pub fn gate_tool(name: &str, arguments: &str) -> Option<String> {
    if !enabled() {
        return None;
    }
    let what = match name {
        "desktop_type" | "desktop_key" | "desktop_click" | "desktop_button" | "desktop_drag" | "desktop_launch" => {
            format!("use the desktop ({name}): {}", spelled(arguments))
        }
        "browser_eval" => format!("run JavaScript in the browser: {}", spelled(arguments)),
        "vm_wake_watch" => format!("start a wake watch (it runs out of view, in no shell): {}", spelled(arguments)),
        name if crate::mcp::handles(name) => format!("call the MCP tool {name}: {}", spelled(arguments)),
        _ => return None,
    };
    let detail = format!("{name} {arguments}");
    ask(&one_line(&what), &visible(&detail)).err().map(|refused| serde_json::json!({"ok": false, "error": refused}).to_string())
}

/// A call's arguments as the operator reads them: `key: value` (a string as
/// it is), joined by ` · `.
fn spelled(arguments: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(serde_json::Value::Object(map)) if !map.is_empty() => map
            .iter()
            .map(|(key, value)| match value {
                serde_json::Value::String(text) => format!("{key}: {text}"),
                other => format!("{key}: {other}"),
            })
            .collect::<Vec<_>>()
            .join(" · "),
        Ok(serde_json::Value::Object(_)) => "(no arguments)".into(),
        _ => arguments.to_string(),
    }
}

/// Wait for the operator to approve `what` (one line) with `detail` (all
/// of it), when approval mode is on: `Ok` to go ahead, `Err` with what to
/// tell the caller when it may not.
pub fn ask(what: &str, detail: &str) -> Result<(), String> {
    if !enabled() || OPERATOR.with(Cell::get) {
        return Ok(());
    }
    let who = crate::exec_feed::owner().map_or_else(|| "s0ma".to_string(), |run_id| crate::netboxes::agent_label(&run_id));
    wait(Pending { id: 0, who, what: visible(what), detail: visible(detail) }, WAIT, crate::kill::requested)
}

fn wait(mut request: Pending, limit: Duration, stopped: impl Fn() -> bool) -> Result<(), String> {
    let snapshot = {
        let mut state = state();
        request.id = state.next;
        state.next += 1;
        state.pending.push(request.clone());
        state.pending.clone()
    };
    changed(snapshot);
    let full = if request.detail.trim().is_empty() || request.detail.trim() == request.what.trim() {
        String::new()
    } else {
        format!("\n{}", request.detail.trim_end())
    };
    crate::terminal::say_status_as(
        &format!("approval · #{} {} asks to {} · alt+a twice approves, alt+x declines", request.id, request.who, request.what),
        &format!("[approval] #{} {} asks to {} (alt+a twice approves, alt+x declines){full}", request.id, request.who, request.what),
    );
    let started = Instant::now();
    let mut state = state();
    let outcome = loop {
        if let Some(at) = state.answers.iter().position(|(id, _)| *id == request.id) {
            let (_, approved) = state.answers.remove(at);
            break if approved { Ok(()) } else { Err(format!("not run: the operator declined it (approval #{})", request.id)) };
        }
        if stopped() {
            break Err(format!("not run: stopped while it waited for the operator's approval (#{})", request.id));
        }
        if started.elapsed() >= limit {
            break Err(format!(
                "not run: the operator did not answer in {} minutes (approval #{}); ask again when he is there",
                limit.as_secs() / 60,
                request.id
            ));
        }
        state = ANSWERED.wait_timeout(state, Duration::from_millis(200)).unwrap_or_else(|poisoned| poisoned.into_inner()).0;
    };
    state.pending.retain(|waiting| waiting.id != request.id);
    if state.armed.is_some_and(|(id, _)| id == request.id) {
        state.armed = None;
    }
    let snapshot = state.pending.clone();
    drop(state);
    changed(snapshot);
    match &outcome {
        Ok(()) => crate::terminal::say_done(&format!("[approval] #{} approved: {}", request.id, request.what)),
        Err(why) => crate::terminal::warn_status(&format!("[approval] #{} {}", request.id, why.trim_start_matches("not run: "))),
    }
    outcome
}

/// The operator's answer to request `id`, given here (the primary).
pub fn answer(id: u64, approve: bool) -> Result<String, String> {
    let mut state = state();
    if !state.pending.iter().any(|request| request.id == id) {
        return Err(format!("[approval] #{id} no longer waits"));
    }
    state.answers.retain(|(answered, _)| *answered != id);
    state.answers.push((id, approve));
    // Off the list at once: the next press is for the next request, not
    // this one while its call has yet to wake and take the answer.
    state.pending.retain(|request| request.id != id);
    if state.armed.is_some_and(|(armed, _)| armed == id) {
        state.armed = None;
    }
    let snapshot = state.pending.clone();
    drop(state);
    ANSWERED.notify_all();
    changed(snapshot);
    Ok(format!("[approval] #{id} {}", if approve { "approved" } else { "declined" }))
}

/// Alt+A (`approve`) or Alt+X: the oldest request, answered, or (Alt+A's
/// first press) armed. What to tell the operator: nothing once it is
/// answered, as the call that waited says how it went.
pub fn key(approve: bool) -> Result<String, String> {
    key_with(approve, Instant::now(), &mut |id, approve| crate::tabs::act_on(crate::tabs::Action::Answer { id, approve }))
}

fn key_with(approve: bool, now: Instant, send: &mut dyn FnMut(u64, bool) -> Result<String, String>) -> Result<String, String> {
    let first = {
        let mut state = state();
        let Some(first) = state.pending.first().cloned() else {
            state.armed = None;
            return Err("[approval] nothing waits for approval".into());
        };
        if approve {
            let armed = state.armed.is_some_and(|(id, at)| id == first.id && now.duration_since(at) <= TWICE);
            if !armed {
                state.armed = Some((first.id, now));
                return Ok(format!("[approval] #{} {}: {} · alt+a again approves it", first.id, first.who, first.what));
            }
        }
        state.armed = None;
        first
    };
    send(first.id, approve).map(|_| String::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// The state is the process's: these run one at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn reset() {
        let mut state = state();
        state.pending.clear();
        state.answers.clear();
        state.armed = None;
    }

    fn waiting_for(n: usize) {
        let until = Instant::now() + Duration::from_secs(5);
        while pending().len() != n && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(pending().len(), n);
    }

    #[test]
    fn a_request_waits_until_alt_a_is_pressed_twice_and_alt_x_declines() {
        let _serial = SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        reset();
        let asked = |what: &'static str| {
            std::thread::spawn(move || wait(Pending { id: 0, who: "s0ma".into(), what: what.into(), detail: String::new() }, Duration::from_secs(20), || false))
        };
        let first = asked("run in main: make");
        waiting_for(1);
        let second = asked("run in main: make install");
        waiting_for(2);
        let sent = std::cell::RefCell::new(Vec::new());
        let mut send = |id, approve| {
            sent.borrow_mut().push((id, approve));
            answer(id, approve)
        };
        let now = Instant::now();
        // One press arms the oldest and says what it would approve.
        let said = key_with(true, now, &mut send).unwrap();
        assert!(said.contains("make") && said.contains("again"), "{said}");
        assert!(sent.borrow().is_empty(), "one press approves nothing");
        // A second press too late arms it again; one in time approves it.
        key_with(true, now + TWICE + Duration::from_millis(1), &mut send).unwrap();
        assert!(sent.borrow().is_empty());
        key_with(true, now + TWICE + Duration::from_secs(1), &mut send).unwrap();
        assert_eq!(pending().len(), 1, "answered: off the list at once, before its call wakes");
        assert!(pending()[0].what.ends_with("install"), "the next press is for the next one");
        assert_eq!(first.join().unwrap(), Ok(()));
        waiting_for(1);
        // Alt+X declines the next at once.
        key_with(false, now, &mut send).unwrap();
        let declined = second.join().unwrap().unwrap_err();
        assert!(declined.contains("declined"), "{declined}");
        assert_eq!(sent.borrow().len(), 2);
        assert!(key_with(true, now, &mut send).is_err(), "nothing left to approve");
    }

    #[test]
    fn an_unanswered_or_stopped_request_is_declined_and_leaves_the_list() {
        let _serial = SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        reset();
        let late = wait(Pending { id: 0, who: "w·b3-1".into(), what: "run in w·b3-1: ls".into(), detail: String::new() }, Duration::from_millis(300), || false);
        assert!(late.unwrap_err().contains("did not answer"));
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let stopped = std::thread::spawn(move || {
            wait(Pending { id: 0, who: "s0ma".into(), what: "start 1 subagent".into(), detail: String::new() }, Duration::from_secs(20), move || flag.load(Ordering::SeqCst))
        });
        waiting_for(1);
        stop.store(true, Ordering::SeqCst);
        assert!(stopped.join().unwrap().unwrap_err().contains("stopped"));
        assert!(pending().is_empty());
        assert!(answer(999, true).is_err(), "an id that no longer waits");
    }

    #[test]
    fn the_mode_is_kept_and_an_unreadable_file_keeps_it_on() {
        let dir = std::env::temp_dir().join(format!("s0ma-approval-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("approval.json");
        assert!(!kept_on(&path), "no file: off");
        std::fs::write(&path, r#"{"on": false}"#).unwrap();
        assert!(!kept_on(&path));
        std::fs::write(&path, r#"{"on": true}"#).unwrap();
        assert!(kept_on(&path));
        std::fs::write(&path, "not json").unwrap();
        assert!(kept_on(&path), "when in doubt, ask");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(one_line("make\n  && make install\n"), "make ⏎   && make install", "a short one whole");
        let long = format!("make\n{}", "x".repeat(WHOLE_CHARS));
        assert!(one_line(&long).starts_with("make (+1 line, ") && one_line(&long).contains("ctrl+o"));
        // What runs is what is read: no carriage return, escape or Ctrl+U hides a part of it.
        assert_eq!(visible("ls\rrm -rf x\x1b[2K\x15"), "ls^Mrm -rf x^[[2K^U");
        assert_eq!(visible("a\u{202e}b"), "a\\u{202e}b");
        assert_eq!(spelled(r#"{"text": "ls -la", "enter": true}"#), "enter: true · text: ls -la");
        assert_eq!(spelled("{}"), "(no arguments)");
        // Off, nothing is asked; on or off, a tool that runs nothing never is.
        if !enabled() {
            assert!(gate_tool("desktop_type", r#"{"text": "x"}"#).is_none());
        }
        assert!(gate_tool("desktop_view", "{}").is_none() && gate_tool("archive_search", "{}").is_none());
        assert!(handle_command("/approval sometimes").unwrap().is_err());
        assert!(handle_command("/shell").is_none());
    }

    #[test]
    fn an_attached_terminal_follows_the_primarys_list() {
        let _serial = SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        reset();
        let request = Pending { id: 7, who: "s0ma".into(), what: "run in main: ls".into(), detail: String::new() };
        apply(vec![request.clone()]);
        let mut sent = Vec::new();
        let now = Instant::now();
        key_with(true, now, &mut |id, approve| {
            sent.push((id, approve));
            Ok(String::new())
        })
        .unwrap();
        // The primary's list moves on before the second press: nothing armed.
        apply(Vec::new());
        assert!(state().armed.is_none());
        apply(vec![request]);
        key_with(true, now, &mut |id, approve| {
            sent.push((id, approve));
            Ok(String::new())
        })
        .unwrap();
        assert!(sent.is_empty(), "a press before the list changed doesn't count toward the second");
        reset();
    }
}
