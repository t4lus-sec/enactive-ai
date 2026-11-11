//! Operator kill: stop the tool call that is running now, or a subagent,
//! without ending s0ma's turn.
//!
//! Ctrl+C ends the whole turn (once the current batch of tool calls is done)
//! and cancels every subagent. `/kill` is narrower:
//!
//! - `/kill` stops the tool call running right now. s0ma gets a result saying
//!   the operator stopped it, and carries on with the rest of its turn.
//! - `/kill <job>` stops one subagent (by job id, slot number or a unique part
//!   of the id); `/kill subagents` stops them all; `/kill all` stops both.
//! - `/kill list` shows what is running.
//!
//! The command acts the moment it is typed, from whichever thread the input
//! arrived on (the keyboard worker during a turn, the attach socket, or the
//! idle prompt). It never waits in the steering queue, which the agent only
//! drains between steps, behind the very call it is meant to stop.
//!
//! # How a running call notices
//!
//! Each tool call on the agent thread, and each subagent worker thread, runs
//! with a cancel token installed for that thread. Code that blocks checks it
//! with [`requested`]: the guest-agent poll in vm.rs (where vm_exec, the VM
//! shell, desktop, browser, Minecraft, vm_image and board tests all wait),
//! MCP requests, `subagent_wait`, `desktop_wait` and web search. A call that
//! notices stops and returns [`Killed`]. Calls that never block for long are
//! left to finish, and the operator is told the kill came too late.
use std::{
    cell::{Cell, RefCell},
    error::Error,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

const POLL: Duration = Duration::from_millis(50);

pub const USAGE: &str = "usage: /kill [list | all | subagents | <subagent job id, slot number or unique part of the id>]";

thread_local! {
    static TOKEN: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
    /// Whether code on this thread has seen the current token set: the
    /// difference between "killed" and "finished before the kill landed".
    static NOTICED: Cell<bool> = const { Cell::new(false) };
}

/// This thread's cancel token, until dropped; then the previous one returns.
pub struct Token {
    previous: Option<Arc<AtomicBool>>,
    noticed: bool,
}

/// Install `flag` as this thread's cancel token.
pub fn install(flag: Arc<AtomicBool>) -> Token {
    swap(Some(flag))
}

/// Run without any cancel token, for cleanup that must finish even though the
/// work around it was stopped (closing a killed worker's VM shell).
pub fn suspend() -> Token {
    swap(None)
}

fn swap(flag: Option<Arc<AtomicBool>>) -> Token {
    let previous = TOKEN.with(|token| token.replace(flag));
    let noticed = NOTICED.with(|noticed| noticed.replace(false));
    Token { previous, noticed }
}

impl Drop for Token {
    fn drop(&mut self) {
        TOKEN.with(|token| *token.borrow_mut() = self.previous.take());
        NOTICED.with(|noticed| noticed.set(self.noticed));
    }
}

/// Whether the work on this thread has been asked to stop. Blocking code
/// calls this in its wait loop and stops when it says so.
pub fn requested() -> bool {
    let set = TOKEN.with(|token| token.borrow().as_ref().is_some_and(|flag| flag.load(Ordering::SeqCst)));
    if set {
        NOTICED.with(|noticed| noticed.set(true));
    }
    set
}

/// What a stopped call returns: what was stopped, and anything it had
/// produced by then.
#[derive(Debug)]
pub struct Killed {
    pub detail: String,
    pub partial: Option<String>,
}

impl Killed {
    pub fn new(detail: impl Into<String>) -> Self {
        Self { detail: detail.into(), partial: None }
    }

    pub fn with_partial(mut self, partial: impl Into<String>) -> Self {
        self.partial = Some(partial.into());
        self
    }
}

// Neutral on purpose: the same error stops a subagent's VM command when s0ma
// (not the operator) cancels the worker. The operator's own kills are labelled
// as such in the tool result s0ma sees (`tool_result`).
impl fmt::Display for Killed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stopped: {}", self.detail)
    }
}

impl Error for Killed {}

/// Sleep, waking early if killed.
pub fn sleep(duration: Duration) -> Result<(), Killed> {
    let until = Instant::now() + duration;
    loop {
        if requested() {
            return Err(Killed::new("stopped while waiting"));
        }
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Ok(());
        }
        thread::sleep(left.min(POLL));
    }
}

/// Run `work` on a helper thread and wait for it, unless killed first. Only
/// for calls whose side effects are harmless to let finish unobserved (a web
/// search): a killed call is abandoned and its late result dropped. The inner
/// `Err` is an ordinary failure (the thread could not start), never a kill; a
/// panic in `work` resumes here, as it would have had `work` run inline.
pub fn abandonable<T: Send + 'static>(
    what: &str,
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<Result<T, String>, Killed> {
    let (sender, receiver) = mpsc::channel();
    let handle = match thread::Builder::new().name(format!("abandonable-{what}")).spawn(move || {
        let _ = sender.send(work());
    }) {
        Ok(handle) => handle,
        Err(error) => return Ok(Err(format!("could not start {what}: {error}"))),
    };
    loop {
        match receiver.recv_timeout(POLL) {
            Ok(value) => return Ok(Ok(value)),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return match handle.join() {
                    Err(panic) => std::panic::resume_unwind(panic),
                    Ok(()) => Ok(Err(format!("{what} ended without a result"))),
                };
            }
            Err(mpsc::RecvTimeoutError::Timeout) if requested() => {
                return Err(Killed::new(format!(
                    "{what} abandoned; it may still finish in the background, and its result will be dropped"
                )));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
}

/// `Command::output`, except that a kill stops the process.
pub fn output(mut command: std::process::Command) -> Result<std::process::Output, Box<dyn Error>> {
    use std::{io::Read, process::Stdio};
    let program = command.get_program().to_string_lossy().into_owned();
    let mut child = command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    // Drained on their own threads so a chatty process cannot fill a pipe and stall.
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut bytes);
            }
            bytes
        })
    };
    let stdout = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let stderr = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(std::process::Output {
                status,
                stdout: stdout.join().unwrap_or_default(),
                stderr: stderr.join().unwrap_or_default(),
            });
        }
        if requested() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Box::new(Killed::new(format!("the {program} process was stopped"))));
        }
        thread::sleep(POLL);
    }
}

// ------------------------------------------------------------------
// The tool call running on the agent thread
// ------------------------------------------------------------------

struct Running {
    id: u64,
    name: String,
    started: Instant,
    flag: Arc<AtomicBool>,
}

static RUNNING: Mutex<Option<Running>> = Mutex::new(None);
static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn running() -> std::sync::MutexGuard<'static, Option<Running>> {
    RUNNING.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Marks one tool call as the one `/kill` stops, with its own token installed
/// on this thread, for as long as it lives. A kill applies only to the call
/// running when it arrives, never to the next one.
pub struct ToolCall {
    id: u64,
    flag: Arc<AtomicBool>,
    token: Option<Token>,
}

pub fn tool_call(name: &str) -> ToolCall {
    let flag = Arc::new(AtomicBool::new(false));
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    *running() = Some(Running { id, name: name.to_owned(), started: Instant::now(), flag: Arc::clone(&flag) });
    let token = install(Arc::clone(&flag));
    ToolCall { id, flag, token: Some(token) }
}

/// How a tool call ended, as far as `/kill` is concerned.
#[derive(Debug, PartialEq, Eq)]
pub enum Ending {
    /// No kill arrived.
    Normal,
    /// A kill arrived and the call stopped for it.
    Killed,
    /// A kill arrived, but the call did not check for one before finishing,
    /// so its result is complete.
    TooLate,
}

impl ToolCall {
    /// Finish the call: uninstall its token and say how it ended.
    pub fn finish(mut self) -> Ending {
        let noticed = NOTICED.with(Cell::get);
        drop(self.token.take());
        let mut current = running();
        if current.as_ref().is_some_and(|r| r.id == self.id) {
            *current = None;
        }
        match (self.flag.load(Ordering::SeqCst), noticed) {
            (false, _) => Ending::Normal,
            (true, true) => Ending::Killed,
            (true, false) => Ending::TooLate,
        }
    }
}

impl Drop for ToolCall {
    fn drop(&mut self) {
        // A call abandoned by an early return still clears its claim.
        if self.token.is_some() {
            drop(self.token.take());
            let mut current = running();
            if current.as_ref().is_some_and(|r| r.id == self.id) {
                *current = None;
            }
        }
    }
}

/// The tool result s0ma sees for a call the operator stopped. `error` is what
/// the tool itself returned, if it returned an error.
pub fn tool_result(tool: &str, elapsed: Duration, killed: Option<&Killed>, own_result: Option<&str>) -> String {
    let detail = killed.map(|k| k.detail.as_str()).unwrap_or("the call was stopped where it was");
    let mut result = serde_json::json!({
        "ok": false,
        "tool": tool,
        "killed": true,
        "error": format!("stopped by the operator (/kill) after {:.1}s: {detail}", elapsed.as_secs_f64()),
        "instruction": "The operator stopped this call on purpose. Do not simply run it again; \
                        change course, and ask the operator if the reason is unclear.",
    });
    if let Some(partial) = killed.and_then(|k| k.partial.as_deref()).or(own_result) {
        result["partial"] = serde_json::json!(partial);
    }
    result.to_string()
}

fn kill_running_tool() -> Option<String> {
    let current = running();
    let running = current.as_ref()?;
    running.flag.store(true, Ordering::SeqCst);
    Some(format!("{} (running {})", running.name, elapsed(running.started.elapsed())))
}

fn describe_running_tool() -> Option<String> {
    running().as_ref().map(|r| format!("{} (running {})", r.name, elapsed(r.started.elapsed())))
}

pub(crate) fn elapsed(duration: Duration) -> String {
    let secs = duration.as_secs();
    match secs {
        0..60 => format!("{:.1}s", duration.as_secs_f64()),
        60..3600 => format!("{}m{:02}s", secs / 60, secs % 60),
        _ => format!("{}h{:02}m", secs / 3600, secs % 3600 / 60),
    }
}

// ------------------------------------------------------------------
// The operator command
// ------------------------------------------------------------------

/// Handle `/kill ...`, acting at once; `None` when `input` is not a kill
/// command. The returned text is the report for the operator.
pub fn handle_command(input: &str) -> Option<String> {
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.first() != Some(&"/kill") {
        return None;
    }
    Some(match &parts[1..] {
        [] => match kill_running_tool() {
            Some(tool) => format!("[kill] stopping {tool}"),
            None => {
                let mut report = "[kill] no tool call is running".to_owned();
                let jobs = crate::subagent::running();
                if !jobs.is_empty() {
                    report.push_str(&format!(
                        "; {} subagent{} running (/kill list shows them, /kill <slot> stops one)",
                        jobs.len(),
                        if jobs.len() == 1 { " is" } else { "s are" }
                    ));
                }
                report
            }
        },
        ["list"] => list(),
        ["all"] => {
            let mut stopped = Vec::new();
            if let Some(tool) = kill_running_tool() {
                stopped.push(format!("tool call {tool}"));
            }
            let jobs = crate::subagent::kill_all();
            if !jobs.is_empty() {
                stopped.push(format!("subagent{} {}", if jobs.len() == 1 { "" } else { "s" }, jobs.join(", ")));
            }
            if stopped.is_empty() {
                "[kill] nothing is running".to_owned()
            } else {
                format!("[kill] stopping {}", stopped.join("; "))
            }
        }
        ["subagents"] => {
            let jobs = crate::subagent::kill_all();
            if jobs.is_empty() {
                "[kill] no subagent is running".to_owned()
            } else {
                format!("[kill] stopping subagent{} {}", if jobs.len() == 1 { "" } else { "s" }, jobs.join(", "))
            }
        }
        [selector] => match crate::subagent::kill(selector) {
            Ok(job) => format!("[kill] stopping subagent {job}"),
            Err(error) => format!("[kill] {error}"),
        },
        _ => format!("[kill] {USAGE}"),
    })
}

fn list() -> String {
    let mut lines = vec![match describe_running_tool() {
        Some(tool) => format!("[kill] tool call: {tool}  (/kill stops it)"),
        None => "[kill] tool call: none running".to_owned(),
    }];
    let jobs = crate::subagent::running();
    if jobs.is_empty() {
        lines.push("[kill] subagents: none running".to_owned());
    } else {
        lines.push(format!("[kill] subagents ({}; /kill <slot> stops one, /kill subagents stops all):", jobs.len()));
        for job in jobs {
            lines.push(format!(
                "  slot {:<2} {}  {}{}  {}",
                job.slot,
                job.id,
                elapsed(job.elapsed),
                job.state.map(|s| format!(" ({s})")).unwrap_or_default(),
                job.task
            ));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_follows_the_installed_token_and_restores_the_previous_one() {
        assert!(!requested(), "no token, never killed");
        let outer = Arc::new(AtomicBool::new(false));
        let _outer = install(Arc::clone(&outer));
        {
            let inner = Arc::new(AtomicBool::new(true));
            let _inner = install(inner);
            assert!(requested());
            {
                let _quiet = suspend();
                assert!(!requested(), "cleanup runs without a token");
            }
            assert!(requested());
        }
        assert!(!requested(), "the outer token is back");
        outer.store(true, Ordering::SeqCst);
        assert!(requested());
    }

    #[test]
    fn sleep_wakes_early_when_killed() {
        let flag = Arc::new(AtomicBool::new(false));
        let _token = install(Arc::clone(&flag));
        let started = Instant::now();
        assert!(sleep(Duration::from_millis(60)).is_ok());
        let killer = Arc::clone(&flag);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            killer.store(true, Ordering::SeqCst);
        });
        let error = sleep(Duration::from_secs(30)).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
        assert_eq!(error.to_string(), "stopped: stopped while waiting");
    }

    #[test]
    fn abandonable_work_returns_its_value_or_is_left_behind() {
        let flag = Arc::new(AtomicBool::new(false));
        let _token = install(Arc::clone(&flag));
        assert_eq!(abandonable("quick", || 7).unwrap(), Ok(7));
        let killer = Arc::clone(&flag);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            killer.store(true, Ordering::SeqCst);
        });
        let started = Instant::now();
        let error = abandonable("slow search", || thread::sleep(Duration::from_secs(5))).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(error.detail.contains("slow search abandoned"), "{error}");
    }

    // One test for the shared RUNNING slot, so parallel tests cannot race on it.
    #[test]
    fn a_kill_stops_only_the_call_running_when_it_arrives() {
        let call = tool_call("vm_exec");
        assert!(list().contains("tool call: vm_exec (running"), "{}", list());
        let report = handle_command("/kill").unwrap();
        assert!(report.starts_with("[kill] stopping vm_exec (running "), "{report}");
        // The blocking code sees it.
        assert!(requested());
        assert_eq!(call.finish(), Ending::Killed);
        assert!(!requested(), "the token went with the call");

        // The next call starts clean.
        let next = tool_call("archive_search");
        assert!(!requested());
        assert_eq!(next.finish(), Ending::Normal);

        // A call that never checks finishes regardless; the kill was too late.
        let stubborn = tool_call("skill_read");
        handle_command("/kill").unwrap();
        assert_eq!(stubborn.finish(), Ending::TooLate);

        // Nothing running.
        assert!(handle_command("/kill").unwrap().starts_with("[kill] no tool call is running"));
        assert_eq!(handle_command("/kill all").unwrap(), "[kill] nothing is running");

        // A call dropped without finishing still releases its claim.
        drop(tool_call("web_search"));
        assert!(list().contains("tool call: none running"));
    }

    #[test]
    fn an_abandonable_panic_is_a_panic_not_a_kill() {
        let outcome = std::panic::catch_unwind(|| abandonable("search", || -> u8 { panic!("search exploded") }));
        assert!(outcome.is_err(), "the panic propagates as if run inline");
    }

    #[test]
    fn output_matches_command_output_and_stops_for_a_kill() {
        let mut echo = std::process::Command::new("sh");
        echo.args(["-c", "printf out; printf err >&2; exit 3"]);
        let output = output(echo).unwrap();
        assert_eq!((output.stdout.as_slice(), output.stderr.as_slice(), output.status.code()), (&b"out"[..], &b"err"[..], Some(3)));

        let flag = Arc::new(AtomicBool::new(false));
        let _token = install(Arc::clone(&flag));
        let killer = Arc::clone(&flag);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(150));
            killer.store(true, Ordering::SeqCst);
        });
        let mut slow = std::process::Command::new("sleep");
        slow.arg("30");
        let started = Instant::now();
        let error = super::output(slow).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(3));
        assert_eq!(error.to_string(), "stopped: the sleep process was stopped");
    }

    #[test]
    fn not_a_kill_command_is_left_alone() {
        assert!(handle_command("/killall").is_none());
        assert!(handle_command("please /kill it").is_none());
        assert!(handle_command("/kill a b").unwrap().contains("usage"));
    }

    #[test]
    fn killed_calls_read_as_deliberate_and_keep_what_they_produced() {
        let killed = Killed::new("the command was stopped on the machine").with_partial("{\"stdout\":\"half\"}");
        let result: serde_json::Value =
            serde_json::from_str(&tool_result("vm_exec", Duration::from_millis(4200), Some(&killed), None)).unwrap();
        assert_eq!(result["killed"], true);
        assert_eq!(result["ok"], false);
        assert_eq!(result["error"], "stopped by the operator (/kill) after 4.2s: the command was stopped on the machine");
        assert_eq!(result["partial"], "{\"stdout\":\"half\"}");
        assert!(result["instruction"].as_str().unwrap().contains("Do not simply run it again"));
        let generic: serde_json::Value =
            serde_json::from_str(&tool_result("mcp_call", Duration::from_secs(1), None, Some("{\"ok\":false}"))).unwrap();
        assert_eq!(generic["partial"], "{\"ok\":false}");
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(elapsed(Duration::from_millis(4200)), "4.2s");
        assert_eq!(elapsed(Duration::from_secs(134)), "2m14s");
        assert_eq!(elapsed(Duration::from_secs(3 * 3600 + 120)), "3h02m");
    }
}
