//! Browser watchdog: notices a wedged or dead `s0ma-browser-reader` and restarts it.
//!
//! # Why reactive, not a polling thread
//!
//! The browser is only worth healing when the agent is about to use it. A background
//! poller would touch the VM on a timer whether or not anyone is browsing, could
//! restart the helper out from under a page the agent left open on purpose, and would
//! need its own reporting channel. Instead the watchdog wraps every browser call:
//!
//! 1. The call fails (non-zero exit, guest timeout, QGA error, empty reply).
//! 2. A cheap `tabs` probe decides whether the helper is actually unhealthy. If
//!    the probe answers, the failure was about the page or the arguments (stale
//!    ref, slow site) and is returned unchanged: no restart for ordinary errors.
//! 3. Unhealthy: restart the helper (rate-limited), wait for it to answer again.
//! 4. Observation calls (`tabs`, `read`, `view`, `open`) are retried once. Actions
//!    that change the page are not: one may have half-applied before the hang,
//!    and repeating a click or a form submit is worse than reporting it.
//!
//! # Limits
//!
//! At most one restart per `RESTART_COOLDOWN`, and `MAX_RESTARTS` per
//! `RESTART_WINDOW`. Past that the breaker opens for a window: each call first
//! runs the short probe and fails fast if it is still dead, instead of every
//! call burning a full 45–90 s timeout. A probe that answers closes the breaker.
//!
//! # Environment
//!
//! - `S0MA_BROWSER_WATCHDOG=0` — disable (calls behave exactly as before).
//! - `S0MA_BROWSER_RESTART_COMMAND` — guest shell command that restarts the
//!   helper. Operator configuration only; nothing model-supplied reaches it.
//!   Default tries the user unit, then a passwordless-sudo system unit.
use serde_json::{json, Value};
use std::time::{Duration, Instant};

pub(super) const PROBE_TIMEOUT_SECS: u64 = 10;
pub(super) const RESTART_TIMEOUT_SECS: u64 = 30;
const RESTART_COOLDOWN: Duration = Duration::from_secs(60);
const RESTART_WINDOW: Duration = Duration::from_secs(10 * 60);
const MAX_RESTARTS: usize = 3;
/// Probes after a restart before giving up on it coming back.
const SETTLE_PROBES: u32 = 5;

pub(super) const DEFAULT_RESTART_COMMAND: &str = "export XDG_RUNTIME_DIR=/run/user/$(id -u); \
systemctl --user restart s0ma-browser-reader.service 2>/dev/null \
|| sudo -n systemctl restart s0ma-browser-reader.service";

/// Tools that only observe, or only ensure the browser is up: safe to repeat.
const RETRYABLE: &[&str] = &["browser_tabs", "browser_read", "browser_view", "browser_open"];

/// One round trip to the guest helper, reduced to what the watchdog needs.
pub(super) enum Attempt {
    /// Exit 0 with the helper's stdout.
    Reply(String),
    /// Anything else: non-zero exit, guest or host timeout, QGA transport error.
    Failed(String),
}

pub(super) trait Guest {
    fn call(&self, request: &Value, timeout_secs: u64) -> Attempt;
    fn restart(&self) -> Result<(), String>;
}

pub(super) struct Watchdog {
    pub(super) enabled: bool,
    restarts: Vec<Instant>,
    breaker_until: Option<Instant>,
    /// Pause between post-restart probes. Zero in tests.
    pub(super) settle_delay: Duration,
}

/// What the caller gets back: the helper's stdout plus, when the watchdog had
/// to intervene, a note for the model-visible result.
#[derive(Debug)]
pub(super) struct Outcome {
    pub(super) stdout: String,
    pub(super) note: Option<Value>,
}

fn probe_request() -> Value {
    json!({"action": "tabs"})
}

impl Watchdog {
    pub(super) const fn new() -> Self {
        Self { enabled: true, restarts: Vec::new(), breaker_until: None, settle_delay: Duration::from_secs(2) }
    }

    fn healthy(&self, guest: &dyn Guest) -> bool {
        matches!(guest.call(&probe_request(), PROBE_TIMEOUT_SECS), Attempt::Reply(_))
    }

    /// Whether a restart is allowed now; opens the breaker when the budget is spent.
    fn restart_refusal(&mut self, now: Instant) -> Option<String> {
        self.restarts.retain(|at| now.duration_since(*at) < RESTART_WINDOW);
        if let Some(last) = self.restarts.last() {
            let since = now.duration_since(*last);
            if since < RESTART_COOLDOWN {
                return Some(format!("helper was already restarted {}s ago; not restarting again yet", since.as_secs()));
            }
        }
        if self.restarts.len() >= MAX_RESTARTS {
            self.breaker_until = Some(now + RESTART_WINDOW);
            return Some(format!(
                "{MAX_RESTARTS} restarts in {} min did not keep it up; watchdog stopped restarting for {} min",
                RESTART_WINDOW.as_secs() / 60, RESTART_WINDOW.as_secs() / 60
            ));
        }
        None
    }

    pub(super) fn run(&mut self, guest: &dyn Guest, tool: &str, request: &Value, timeout_secs: u64, now: Instant) -> Result<Outcome, String> {
        if !self.enabled {
            return match guest.call(request, timeout_secs) {
                Attempt::Reply(stdout) => Ok(Outcome { stdout, note: None }),
                Attempt::Failed(error) => Err(error),
            };
        }

        // Breaker open: spend a short probe, not a full timeout, to learn it is still down.
        if let Some(until) = self.breaker_until {
            if now < until
                && !self.healthy(guest) {
                    return Err(format!(
                        "browser helper is down and the watchdog has stopped restarting it (retrying in {}s). \
                         Operator: inspect s0ma-browser-reader.service on the machine. Use vm_exec or desktop tools meanwhile.",
                        until.duration_since(now).as_secs()
                    ));
                }
            self.breaker_until = None;
            self.restarts.clear();
        }

        let error = match guest.call(request, timeout_secs) {
            Attempt::Reply(stdout) => return Ok(Outcome { stdout, note: None }),
            Attempt::Failed(error) => error,
        };

        // Stopped by the operator's /kill: the helper is not at fault, so it is
        // neither probed nor restarted.
        if crate::kill::requested() {
            return Err(error);
        }

        // A helper that answers a probe is healthy: the failure was this call's own.
        // A probe that failed because of a kill says nothing about the helper.
        if self.healthy(guest) || crate::kill::requested() {
            return Err(error);
        }

        if let Some(refusal) = self.restart_refusal(now) {
            return Err(format!("{error}\nBrowser watchdog: helper failed a health check; {refusal}."));
        }
        self.restarts.push(now);
        crate::llm::status(format_args!("[browser watchdog] helper unresponsive after {tool}; restarting s0ma-browser-reader"));

        if let Err(restart_error) = guest.restart() {
            crate::llm::status(format_args!("[browser watchdog] restart failed: {restart_error}"));
            return Err(format!(
                "{error}\nBrowser watchdog: helper failed a health check and the restart failed: {restart_error}. \
                 Operator: check S0MA_BROWSER_RESTART_COMMAND or the unit on the machine."
            ));
        }

        let mut back = false;
        for attempt in 0..SETTLE_PROBES {
            if attempt > 0 && !self.settle_delay.is_zero() {
                std::thread::sleep(self.settle_delay);
            }
            if crate::kill::requested() {
                return Err(format!("{error}\nBrowser watchdog: restarted the helper; stopped waiting for it to answer."));
            }
            if self.healthy(guest) {
                back = true;
                break;
            }
        }
        if !back {
            crate::llm::status(format_args!("[browser watchdog] helper restarted but still not answering"));
            return Err(format!("{error}\nBrowser watchdog: restarted the helper, but it is still not answering."));
        }
        crate::llm::status(format_args!("[browser watchdog] helper restarted and answering"));

        let note = json!({
            "restarted_helper": true,
            "retried": true,
            "reason": error.chars().take(300).collect::<String>(),
            "note": "Tabs and page state from before the restart may be gone, and earlier element refs are invalid."
        });

        if !RETRYABLE.contains(&tool) {
            return Err(format!(
                "{error}\nBrowser watchdog: the helper was hung and has been restarted. This {tool} was NOT retried: \
                 it may have partially applied before the hang. Earlier tabs and refs are invalid; run browser_tabs \
                 and browser_read, then redo it if it is still needed."
            ));
        }

        if crate::kill::requested() {
            return Err(format!("{error}\nBrowser watchdog: restarted the helper; this {tool} was not retried."));
        }
        match guest.call(request, timeout_secs) {
            Attempt::Reply(stdout) => Ok(Outcome { stdout, note: Some(note) }),
            Attempt::Failed(retry_error) => Err(format!(
                "{error}\nBrowser watchdog: restarted the helper, but the retried {tool} failed too: {retry_error}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;

    /// Scripted guest: each call pops the next scripted reply; restarts are counted.
    struct Script {
        replies: RefCell<VecDeque<bool>>,
        calls: RefCell<Vec<String>>,
        restarts: Cell<u32>,
        restart_ok: bool,
    }

    impl Script {
        fn new(replies: &[bool]) -> Self {
            Self { replies: RefCell::new(replies.iter().copied().collect()), calls: RefCell::new(Vec::new()), restarts: Cell::new(0), restart_ok: true }
        }
        fn actions(&self) -> Vec<String> { self.calls.borrow().clone() }
    }

    impl Guest for Script {
        fn call(&self, request: &Value, _: u64) -> Attempt {
            self.calls.borrow_mut().push(request["action"].as_str().unwrap_or("?").to_owned());
            if self.replies.borrow_mut().pop_front().unwrap_or(false) {
                Attempt::Reply(json!({"ok": true}).to_string())
            } else {
                Attempt::Failed("helper exited 1".into())
            }
        }
        fn restart(&self) -> Result<(), String> {
            self.restarts.set(self.restarts.get() + 1);
            if self.restart_ok { Ok(()) } else { Err("permission denied".into()) }
        }
    }

    fn dog() -> Watchdog {
        let mut dog = Watchdog::new();
        dog.settle_delay = Duration::ZERO;
        dog
    }
    fn read() -> Value { json!({"action":"snapshot"}) }
    fn click() -> Value { json!({"action":"click"}) }

    /// A guest whose calls all fail, and which sets the kill token on its
    /// `nth` call (counting from 1), as a /kill landing mid-call would.
    struct KilledDuring {
        flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
        nth: usize,
        calls: Cell<usize>,
        restarts: Cell<u32>,
    }

    impl Guest for KilledDuring {
        fn call(&self, _: &Value, _: u64) -> Attempt {
            self.calls.set(self.calls.get() + 1);
            if self.calls.get() == self.nth {
                self.flag.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Attempt::Failed("stopped".into())
        }
        fn restart(&self) -> Result<(), String> {
            self.restarts.set(self.restarts.get() + 1);
            Ok(())
        }
    }

    #[test]
    fn a_kill_never_leads_to_a_restart_or_retry() {
        // Killed during the call itself, or during the health probe after it.
        for nth in [1, 2] {
            let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let _token = crate::kill::install(std::sync::Arc::clone(&flag));
            let guest = KilledDuring { flag, nth, calls: Cell::new(0), restarts: Cell::new(0) };
            let mut dog = dog();
            assert!(dog.run(&guest, "browser_read", &read(), 45, Instant::now()).is_err());
            assert_eq!(guest.restarts.get(), 0, "killed on call {nth}");
            assert!(dog.restarts.is_empty(), "no restart counted against the budget");
            assert_eq!(guest.calls.get(), nth, "nothing after the kill");
        }
    }

    #[test]
    fn actions_are_restarted_but_never_repeated() {
        let guest = Script::new(&[false, false, true]);
        let error = dog().run(&guest, "browser_click", &click(), 45, Instant::now()).unwrap_err();
        assert!(error.contains("NOT retried"));
        assert_eq!(guest.restarts.get(), 1);
        assert_eq!(guest.actions().iter().filter(|a| *a == "click").count(), 1);
    }
}
