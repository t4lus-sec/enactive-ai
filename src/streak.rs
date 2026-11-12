//! Stuck detection by streaks of identical tool calls (A03; A80 would use the
//! same predicate with the agent as the actor).
//!
//! A streak is a run of consecutive calls with the same key: (actor, tool,
//! sha256 of the canonical arguments). Canonical means JSON with sorted keys
//! and no whitespace, so key order and spacing don't make two calls differ;
//! arguments that aren't JSON are compared as they are. Any different call
//! starts a new streak.
//!
//! Each streak counts `reps` (every call) and `fails` (calls whose result was
//! an error). At `reps == REMIND_AT` a reminder goes back with the result,
//! whatever the outcome; at `fails == STOP_AT_FAILS` the actor is stopped. A
//! streak of successful calls (polling until a check passes) is never stopped,
//! only reminded once. The reminder is advisory: the count keeps running. A
//! direction from the agent resets every streak, since repeating a call after one
//! is legitimate. There is no opt-out.
//!
//! Semantic loops (the same failure with paraphrased arguments) are out of
//! scope by design: canonical identity is the cheap predicate.
//!
//! The arguments keyed are the call as it ran, after the harness filled in
//! defaults and clamps (a worker's `vm_exec` without `timeout_secs` runs as
//! `"timeout_secs":60`, and so does one asking for 300). Keyed on what the
//! model wrote instead, two spellings of one command never form a streak,
//! while every record the caller can see shows the same call repeating.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Identical calls in a row before a reminder, whatever their outcome.
pub(crate) const REMIND_AT: u32 = 5;
/// Identical failing calls in a streak before the actor is stopped.
pub(crate) const STOP_AT_FAILS: u32 = 8;
/// How much of the last error a reminder or a stop repeats.
const ERROR_ECHO_CHARS: usize = 600;

/// How a call came out, as far as streaks care.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Succeeded,
    /// With the error text to echo back.
    Failed(String),
}

impl Outcome {
    /// Read a tool's result: an error is `"ok": false`, a command that exited
    /// nonzero or was killed by a signal, or a timeout. Anything else,
    /// including output that isn't JSON, is a success.
    pub(crate) fn of(output: &str) -> Self {
        let Ok(value) = serde_json::from_str::<Value>(output.trim()) else {
            return Self::Succeeded;
        };
        let text = |key: &str| value[key].as_str().map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
        if value["ok"] == Value::Bool(false) || value["status"] == "timeout" {
            let error = text("error").or_else(|| text("detail")).or_else(|| text("reason")).unwrap_or_else(|| "failed".into());
            return Self::Failed(clip(&error));
        }
        let exit = value["exit_code"].as_i64();
        let signal = value["signal"].as_i64();
        if exit.is_some_and(|code| code != 0) || signal.is_some() {
            let mut error = match (exit, signal) {
                (_, Some(signal)) => format!("killed by signal {signal}"),
                (Some(code), _) => format!("exit code {code}"),
                _ => unreachable!(),
            };
            if let Some(detail) = text("stderr").or_else(|| text("stdout")) {
                error.push_str(": ");
                let room = ERROR_ECHO_CHARS.saturating_sub(error.chars().count());
                error.push_str(&tail(&detail, room));
            }
            return Self::Failed(error);
        }
        Self::Succeeded
    }
}

/// JSON with sorted keys and no whitespace; anything else as it is.
pub(crate) fn canonical(arguments: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        // serde_json's map is ordered by key (no preserve_order), and
        // `to_string` writes no whitespace.
        Ok(value) => value.to_string(),
        Err(_) => arguments.to_string(),
    }
}

fn sha256(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

/// The streak in progress.
#[derive(Clone, Debug)]
struct Streak {
    tool: String,
    args_hash: String,
    digest: String,
    reps: u32,
    fails: u32,
    first_seen: String,
    /// The latest error in the streak, if any of its calls failed.
    last_error: Option<String>,
    /// Whether the latest call failed.
    last_failed: bool,
}

/// What a streak stands at, for a reminder or a stop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Report {
    pub tool: String,
    /// 16 hex digits identifying (actor, tool, canonical arguments).
    pub digest: String,
    pub reps: u32,
    pub fails: u32,
    pub first_seen: String,
    pub last_error: Option<String>,
    pub last_failed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Continue,
    /// Advisory: append `Report::reminder` to the result and carry on.
    Remind(Report),
    /// Stop the actor: stop_reason `stuck_terminated`.
    Stop(Report),
}

/// One actor's streak (a worker's, or the agent's for A80).
#[derive(Clone, Debug)]
pub(crate) struct Streaks {
    actor: String,
    current: Option<Streak>,
    /// The longest streak so far (the most failures among equals), kept across
    /// resets: what the caller reads to see what the detector saw.
    longest: Option<Report>,
    reminders: u32,
}

/// What the detector saw over a whole run, for the caller's envelope. Present
/// on every worker that made a call, so a run with no reminder still shows
/// how close it came, and which build was running.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct Summary {
    /// Reminders sent back with results.
    pub reminders: u32,
    pub longest_tool: String,
    pub longest_key_digest: String,
    pub longest_reps: u32,
    pub longest_fails: u32,
}

impl Streaks {
    pub(crate) fn new(actor: &str) -> Self {
        Self { actor: actor.to_string(), current: None, longest: None, reminders: 0 }
    }

    pub(crate) fn summary(&self) -> Option<Summary> {
        self.longest.as_ref().map(|longest| Summary {
            reminders: self.reminders,
            longest_tool: longest.tool.clone(),
            longest_key_digest: longest.digest.clone(),
            longest_reps: longest.reps,
            longest_fails: longest.fails,
        })
    }

    /// A call finished: count it (a different call starts a new streak) and
    /// say what to do. `now` is when it was made, for first-seen.
    pub(crate) fn record(&mut self, tool: &str, arguments: &str, outcome: &Outcome, now: &str) -> Verdict {
        let args_hash = sha256(&canonical(arguments));
        let same = self.current.as_ref().is_some_and(|s| s.tool == tool && s.args_hash == args_hash);
        if !same {
            let digest = sha256(&format!("{}\0{tool}\0{args_hash}", self.actor))[..16].to_string();
            self.current = Some(Streak {
                tool: tool.to_string(), args_hash, digest, reps: 0, fails: 0,
                first_seen: now.to_string(), last_error: None, last_failed: false,
            });
        }
        let streak = self.current.as_mut().expect("just set");
        streak.reps += 1;
        streak.last_failed = matches!(outcome, Outcome::Failed(_));
        if let Outcome::Failed(error) = outcome {
            streak.fails += 1;
            streak.last_error = Some(error.clone());
        }
        let report = Report {
            tool: streak.tool.clone(), digest: streak.digest.clone(), reps: streak.reps, fails: streak.fails,
            first_seen: streak.first_seen.clone(), last_error: streak.last_error.clone(), last_failed: streak.last_failed,
        };
        let longer = self.longest.as_ref().is_none_or(|longest| {
            (report.reps, report.fails) > (longest.reps, longest.fails)
        });
        if longer {
            self.longest = Some(report.clone());
        }
        if streak.fails >= STOP_AT_FAILS {
            Verdict::Stop(report)
        } else if streak.reps == REMIND_AT {
            self.reminders += 1;
            Verdict::Remind(report)
        } else {
            Verdict::Continue
        }
    }

    /// A direction from the agent arrived: every streak starts over.
    pub(crate) fn reset(&mut self) -> bool {
        self.current.take().is_some()
    }
}

impl Report {
    /// The reminder that goes back with the result: what is repeating, the
    /// last error (what a stuck loop has stopped reading), and a way out.
    pub(crate) fn reminder(&self) -> String {
        let error = match (&self.last_error, self.last_failed) {
            (Some(error), true) => format!("Its last error: \"{error}\"."),
            (Some(error), false) => format!("The last call succeeded; the latest error in the streak was: \"{error}\"."),
            (None, _) => "Every one of them succeeded.".to_string(),
        };
        format!(
            "\n[HARNESS — STUCK CHECK, NOT A NEW TASK] {reps} identical {tool} calls in a row (key {digest}, first at \
{first}; {fails} failed). {error}\n\
- If you are polling: wrap the loop in ONE {tool} call (a loop with a sleep inside one command) instead of repeating \
it; it saves your context.\n\
- If you are stuck: change what you are checking (a different command, path or condition); or ask for direction \
(stop and report what blocks you and what you need from s0ma); or write your findings so far to your notes file \
and stop.\n\
This is a reminder, not a stop: you may go on. {stop} identical failing calls in a row end this worker \
(stop_reason stuck_terminated).",
            reps = self.reps, tool = self.tool, digest = self.digest, first = self.first_seen, fails = self.fails,
            stop = STOP_AT_FAILS,
        )
    }

    /// Why the actor was stopped, for its error and journal.
    pub(crate) fn stop_message(&self) -> String {
        format!(
            "stuck: {} identical failing {} calls in a row (key {}, {} calls since {}); last error: {}",
            self.fails, self.tool, self.digest, self.reps, self.first_seen,
            self.last_error.as_deref().unwrap_or("none")
        )
    }
}

fn clip(text: &str) -> String {
    crate::util::clip(text, ERROR_ECHO_CHARS)
}

/// The end of a long output, where the error usually is: at most `room`
/// characters, an ellipsis included.
fn tail(text: &str, room: usize) -> String {
    let count = text.chars().count();
    if count <= room {
        return text.to_string();
    }
    let keep = room.saturating_sub(1);
    format!("…{}", text.chars().skip(count - keep).collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAILING: &str = r#"{"exit_code":1,"signal":null,"stdout":"","stderr":"pkill: killing pid 41 failed: Operation not permitted","stdout_truncated":false,"stderr_truncated":false}"#;
    const POLL_OK: &str = r#"{"exit_code":0,"signal":null,"stdout":"pending","stderr":""}"#;

    fn run(streaks: &mut Streaks, tool: &str, args: &str, output: &str, n: usize) -> Vec<Verdict> {
        (0..n).map(|i| streaks.record(tool, args, &Outcome::of(output), &format!("t{i}"))).collect()
    }

    fn reminders(verdicts: &[Verdict]) -> Vec<usize> {
        verdicts.iter().enumerate().filter(|(_, v)| matches!(v, Verdict::Remind(_))).map(|(i, _)| i + 1).collect()
    }

    fn stops(verdicts: &[Verdict]) -> Vec<usize> {
        verdicts.iter().enumerate().filter(|(_, v)| matches!(v, Verdict::Stop(_))).map(|(i, _)| i + 1).collect()
    }

    #[test]
    fn l2_a_successful_poll_is_reminded_once_and_never_stopped() {
        let mut streaks = Streaks::new("subagent-1-1");
        let verdicts = run(&mut streaks, "vm_exec", r#"{"command":"gh pr checks 41"}"#, POLL_OK, 6);
        assert_eq!((reminders(&verdicts), stops(&verdicts)), (vec![5], vec![]));
        let more = run(&mut streaks, "vm_exec", r#"{"command":"gh pr checks 41"}"#, POLL_OK, 40);
        assert!(reminders(&more).is_empty() && stops(&more).is_empty(), "one reminder per streak, no stop for successes");
        let Verdict::Remind(report) = &verdicts[4] else { unreachable!() };
        assert!(report.reminder().contains("Every one of them succeeded."));
    }

    #[test]
    fn l5_key_order_and_spacing_are_the_same_call() {
        let mut streaks = Streaks::new("subagent-1-1");
        let spellings = [
            r#"{"command":"ls /tmp","timeout_secs":30}"#,
            r#"{"timeout_secs":30,"command":"ls /tmp"}"#,
            "{ \"timeout_secs\" : 30 ,\n  \"command\" : \"ls /tmp\" }",
            r#"{"command":"ls /tmp","timeout_secs":30}"#,
            r#"{"timeout_secs":30, "command":"ls /tmp"}"#,
        ];
        let verdicts: Vec<Verdict> = spellings.iter().map(|args| streaks.record("vm_exec", args, &Outcome::of(FAILING), "t")).collect();
        assert_eq!(reminders(&verdicts), [5]);
        assert_eq!(canonical(spellings[2]), r#"{"command":"ls /tmp","timeout_secs":30}"#);
        // Values still matter, and nested keys are sorted too.
        assert_ne!(canonical(r#"{"command":"ls /tmp "}"#), canonical(r#"{"command":"ls /tmp"}"#));
        assert_eq!(canonical(r#"{"b":{"y":1,"x":2},"a":[3,1]}"#), r#"{"a":[3,1],"b":{"x":2,"y":1}}"#);
        assert_eq!(canonical("not json"), "not json");
    }
}
