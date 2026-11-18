//! Audits: claims about a codebase, checked by the harness, never by the agent
//! (`claude/audit-spec.md`).
//!
//! A project (a board) can pick an audit up to find where code stops doing
//! what it says: the agent proposes the claims its code and docs make, workers
//! write a check for each, and the harness vets and runs the checks and
//! writes every verdict. A failure becomes a finding on the project's board
//! only once a repro the harness ran fails on the real code.
//!
//! Four rules hold everywhere here:
//! 1. **the agent proposes, the harness decides.** No tool argument sets a verdict
//!    or a level; only `audit_run` writes them, and only it posts results to
//!    a board (`post_result`).
//! 2. **A failure is evidence only once reproduced** (`Repro::Fails`).
//! 3. **Only the operator decides what "correct" means** where it counts as
//!    assurance: unreviewed claims never enter the levels (`counts`).
//! 4. **Checks get stronger, never quietly weaker** once relied on
//!    (`crate::audit::run::strengthen_only`).
//!
//! The store is `audit.sqlite3` beside the archive. Claims, audits and checks
//! keep their many small fields as JSON beside the columns that are queried,
//! so a field added later needs no migration.

pub(crate) mod adapters;
pub(crate) mod ops;
pub(crate) mod report;
pub(crate) mod run;
pub(crate) mod scope;
pub(crate) mod tools;

use chrono::{DateTime, Local};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

const SCHEMA: &str = "audit.v1";
/// How long `audit_take` holds a claim for a worker, as a board lead is held.
pub const HOLD_SECS: i64 = 30 * 60;
const MAX_NAME: usize = 24;
const MAX_TEXT: usize = 600;

// ------------------------------------------------------------------ kinds

/// How far a claim has been checked, weakest first. `Assumed` is apart: an
/// assumption about code outside the slice, used and never checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    #[default]
    Documented,
    Tested,
    Property,
    Bounded,
    Model,
    Proven,
    Assumed,
}

impl Level {

    pub fn label(self) -> &'static str {
        match self {
            Level::Documented => "documented",
            Level::Tested => "tested",
            Level::Property => "property",
            Level::Bounded => "bounded",
            Level::Model => "model",
            Level::Proven => "proven",
            Level::Assumed => "assumed",
        }
    }

    pub fn parse(text: &str) -> Option<Level> {
        Some(match text.trim() {
            "documented" | "doc" => Level::Documented,
            "tested" | "test" => Level::Tested,
            "property" | "prop" => Level::Property,
            "bounded" => Level::Bounded,
            "model" | "tla" => Level::Model,
            "proven" | "proof" => Level::Proven,
            "assumed" => Level::Assumed,
            _ => return None,
        })
    }

    /// The level a check of `kind` reaches when it holds and catches mutants.
    pub fn of_kind(kind: &str) -> Level {
        match kind {
            "test" | "command" | "differential" => Level::Tested,
            "property" => Level::Property,
            "bounded" | "kani" | "cbmc" | "crosshair" | "jbmc" | "esbmc" => Level::Bounded,
            "tla" => Level::Model,
            "proof" => Level::Proven,
            _ => Level::Documented,
        }
    }

    /// The next deeper level a plan row cycles to (`d`), up to `ceiling`.
    pub fn deeper(self, ceiling: Level, tla: bool) -> Level {
        let order = [Level::Tested, Level::Property, Level::Bounded, Level::Model];
        let allowed: Vec<Level> = order.into_iter().filter(|level| *level <= ceiling || (*level == Level::Model && tla)).collect();
        let at = allowed.iter().position(|level| *level == self);
        match at {
            Some(i) if i + 1 < allowed.len() => allowed[i + 1],
            _ => allowed.first().copied().unwrap_or(Level::Tested),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Inside a project: finding the bug. Results are evidence, not assurance.
    #[default]
    Diagnose,
    /// A standing audit, or a project whose goal is the audit.
    Assure,
}

/// Where an audit is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// The agent's, waiting for the operator to start it (policy `ask`).
    #[default]
    Proposed,
    Declined,
    /// Started: claims are being collected.
    Harvesting,
    /// Checking the plan (after Go, when the gate is `plan`).
    Checking,
    /// The queue is empty: nothing left to check.
    Done,
    Stopped,
    /// Its project ended, and it was let go.
    Closed,
    /// Kept after its project: a standing audit, run when the operator starts it.
    Standing,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Proposed => "proposed",
            Status::Declined => "declined",
            Status::Harvesting => "harvesting",
            Status::Checking => "checking",
            Status::Done => "done",
            Status::Stopped => "stopped",
            Status::Closed => "closed",
            Status::Standing => "standing",
        }
    }

    /// Still part of the work: shown, and its queue served.
    pub fn live(self) -> bool {
        matches!(self, Status::Proposed | Status::Harvesting | Status::Checking | Status::Done | Status::Standing)
    }
}

/// Where results go (`post`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Post {
    /// Reproduced failures and holds at `bounded` or above.
    #[default]
    Auto,
    /// Reproduced failures only.
    Failures,
    /// Everything waits as `ready` for the operator to post or discard.
    Review,
}

/// What gates checking (`gate`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Gate {
    /// Nothing is checked until Go on the plan.
    #[default]
    Plan,
    /// Only claims the operator accepted.
    Reviewed,
    /// In plan order as soon as there is a plan.
    None,
}

/// A project's audit policy (`/board audits <board> off|ask|auto`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    Off,
    #[default]
    Ask,
    Auto,
}

impl Policy {
    pub fn parse(text: &str) -> Option<Policy> {
        Some(match text {
            "off" => Policy::Off,
            "ask" => Policy::Ask,
            "auto" => Policy::Auto,
            _ => return None,
        })
    }
    pub fn label(self) -> &'static str {
        match self {
            Policy::Off => "off",
            Policy::Ask => "ask",
            Policy::Auto => "auto",
        }
    }
}

/// An audit's limits, changed in its settings row or with `/audit set`.
/// A change takes effect at the next check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Its own cap on steps, within the project's budget (None: a third of
    /// the steps left when it started).
    pub steps: Option<i64>,
    pub check_secs: u64,
    pub total_secs: u64,
    pub workers: usize,
    /// The deepest level overall (None: the language's ceiling).
    pub depth: Option<Level>,
    /// Tools by name: `-kani` bars, `+tla` allows one not on by default.
    pub tools: Vec<String>,
    pub mutants: bool,
    pub back_translation: bool,
    pub loophole: bool,
    /// A run estimated past this asks first, even outside approval mode.
    pub ask_over_secs: u64,
    pub post: Post,
    pub gate: Gate,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            steps: None,
            check_secs: 600,
            total_secs: 7200,
            workers: 2,
            depth: None,
            tools: Vec::new(),
            mutants: true,
            back_translation: true,
            loophole: false,
            ask_over_secs: 900,
            post: Post::Auto,
            gate: Gate::Plan,
        }
    }
}

impl Settings {
    /// `key=value` pairs, as `/audit set` and the settings row give them.
    pub fn apply(&mut self, pairs: &str) -> std::result::Result<Vec<String>, String> {
        let mut changed = Vec::new();
        for pair in pairs.split_whitespace() {
            let (key, value) = pair.split_once('=').ok_or_else(|| format!("{pair}: key=value"))?;
            let secs = |value: &str| parse_duration(value).ok_or_else(|| format!("{key}: a time like 10m, 2h or 90s"));
            let on = |value: &str| match value {
                "on" | "yes" | "true" => Ok(true),
                "off" | "no" | "false" => Ok(false),
                _ => Err(format!("{key}: on or off")),
            };
            match key {
                "steps" if value == "auto" => self.steps = None,
                "steps" => self.steps = Some(value.parse().map_err(|_| "steps: a number".to_string())?),
                "time" => {
                    let (check, total) = value.split_once('/').unwrap_or((value, ""));
                    if !check.is_empty() {
                        self.check_secs = secs(check)?;
                    }
                    if !total.is_empty() {
                        self.total_secs = secs(total)?;
                    }
                }
                "workers" => self.workers = value.parse::<usize>().ok().filter(|n| (1..=8).contains(n)).ok_or("workers: 1 to 8")?,
                "depth" if matches!(value, "auto" | "test-first") => self.depth = None,
                "depth" => self.depth = Some(Level::parse(value).ok_or("depth: tested, property, bounded, model or proven")?),
                "tools" if value == "all" => self.tools.clear(),
                "tools" => {
                    for tool in value.split(',').filter(|tool| !tool.is_empty()) {
                        if !(tool.starts_with('+') || tool.starts_with('-')) {
                            return Err(format!("tools: +name or -name ({tool})"));
                        }
                        let name = &tool[1..];
                        self.tools.retain(|existing| &existing[1..] != name);
                        self.tools.push(tool.to_string());
                    }
                }
                "mutants" => self.mutants = on(value)?,
                "back-translation" | "backtranslation" => self.back_translation = on(value)?,
                "loophole" => self.loophole = on(value)?,
                "ask-over" => self.ask_over_secs = secs(value)?,
                "post" => {
                    self.post = match value {
                        "auto" => Post::Auto,
                        "failures" => Post::Failures,
                        "review" => Post::Review,
                        _ => return Err("post: auto, failures or review".into()),
                    }
                }
                "gate" => {
                    self.gate = match value {
                        "plan" => Gate::Plan,
                        "reviewed" => Gate::Reviewed,
                        "none" => Gate::None,
                        _ => return Err("gate: plan, reviewed or none".into()),
                    }
                }
                _ => return Err(format!("{key}: not a setting (steps, time, workers, depth, tools, mutants, back-translation, loophole, ask-over, post, gate)")),
            }
            changed.push(pair.to_string());
        }
        Ok(changed)
    }

    /// Whether a tool is allowed: barred by `-name`, or off by default (tla)
    /// unless `+name`.
    pub fn allows(&self, tool: &str) -> bool {
        if self.tools.iter().any(|t| t == &format!("-{tool}")) {
            return false;
        }
        tool != "tla" || self.tools.iter().any(|t| t == "+tla")
    }

    /// The row as the tab shows it.
    pub fn row(&self) -> String {
        format!(
            "steps {} · time {}/{} · workers {} · depth {} · tools {} · {} · ask-over {} · post {} · gate {}",
            self.steps.map_or("auto".to_string(), |s| s.to_string()),
            show_secs(self.check_secs),
            show_secs(self.total_secs),
            self.workers,
            self.depth.map_or("test first", Level::label),
            if self.tools.is_empty() { "all".to_string() } else { self.tools.join(",") },
            [("mutants", self.mutants), ("back-translation", self.back_translation), ("loophole", self.loophole)]
                .iter()
                .map(|(name, on)| format!("{name} {}", if *on { "on" } else { "off" }))
                .collect::<Vec<_>>()
                .join(" · "),
            show_secs(self.ask_over_secs),
            match self.post {
                Post::Auto => "auto",
                Post::Failures => "failures",
                Post::Review => "review",
            },
            match self.gate {
                Gate::Plan => "plan",
                Gate::Reviewed => "reviewed",
                Gate::None => "none",
            },
        )
    }
}

pub fn parse_duration(text: &str) -> Option<u64> {
    let text = text.trim();
    let (number, unit) = text.split_at(text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len()));
    let n: u64 = number.parse().ok()?;
    Some(match unit {
        "" | "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        _ => return None,
    })
}

pub fn show_secs(secs: u64) -> String {
    if secs >= 3600 && secs.is_multiple_of(3600) {
        format!("{}h", secs / 3600)
    } else if secs >= 60 && secs.is_multiple_of(60) {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

// ------------------------------------------------------------------ records

/// An audit: its target, where it works, its settings and where it stands.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Audit {
    pub name: String,
    /// The project it runs for (its owner), by board name; None standing.
    pub board: Option<String>,
    pub mode: Mode,
    pub status: Status,
    pub paused: bool,
    /// Folders on the agent's machine.
    pub targets: Vec<String>,
    /// `git` (a worktree on `s0ma-audit/<name>`) or `folder` (a copy).
    pub kind: String,
    /// Where it works on the agent's machine, and its branch and commit (git).
    pub work: String,
    pub branch: String,
    pub commit: String,
    /// Scope specs (`audit/scope.rs`); empty: the whole target.
    pub scope: Vec<String>,
    pub settings: Settings,
    /// The agent's reason for proposing it, and its estimate.
    pub why: String,
    pub estimate: String,
    /// The board lead it holds while it runs.
    pub lead: Option<String>,
    /// The step budget it may spend, fixed when it started.
    pub step_cap: i64,
    pub steps_used: i64,
    /// When Go was last pressed on the plan: claims added after are `new`.
    pub gone_at: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    /// The project a kept audit came from.
    pub kept_from: Option<String>,
    /// Languages found at start (the ceiling preview).
    pub languages: Vec<String>,
    /// The slice map (`audit/scope.rs`), once worked out.
    pub slice: Option<crate::audit::scope::Slice>,
    /// A re-run the agent proposed, waiting for the operator.
    pub rerun_proposed: Option<String>,
    /// An attach the agent proposed (owner), waiting for the operator.
    pub attach_proposed: Option<(String, String)>,
    /// What the harness's harvest scan found for the agent to turn into claims:
    /// comments with claim words, asserts, test names, fix commits.
    pub sources: Vec<String>,
    /// Paused because the operator took control of its shell (Ctrl+\):
    /// resumes when he hands it back.
    pub wheel_paused: bool,
    /// Wall-clock spent checking, against its `time` total.
    pub spent_secs: u64,
    /// What went wrong setting it up, if anything.
    pub problem: Option<String>,
    /// Set up or harvested and not yet finished: started again if the
    /// harness stopped part way.
    pub harvest_open: bool,
    /// Put away by the operator: out of the portfolio, the agent's context
    /// and the board, everything kept; brought back as it was.
    pub archived: bool,
    pub archived_at: Option<String>,
}

impl Audit {
    /// Shown in the portfolio, the projects tab and the agent's context:
    /// live and not archived.
    pub fn shown(&self) -> bool {
        self.status.live() && !self.archived
    }

    /// The audit's shell's netbox name.
    pub fn shell(&self) -> String {
        shell_name(&self.name)
    }

    pub fn standing(&self) -> bool {
        self.board.is_none()
    }

    pub fn status_mark(&self) -> (&'static str, &'static str) {
        if self.wheel_paused {
            return ("[!]", "paused · you have its shell");
        }
        if self.paused {
            return ("[!]", "paused");
        }
        match self.status {
            Status::Proposed => ("[!]", "proposed · waiting for you"),
            Status::Harvesting => ("[*]", "harvesting"),
            Status::Checking if self.mode == Mode::Diagnose => ("[*]", "diagnosing"),
            Status::Checking => ("[*]", "checking"),
            Status::Done => ("[+]", "done"),
            Status::Standing => ("[+]", "standing"),
            Status::Stopped => ("[+]", "stopped"),
            Status::Declined => ("[+]", "declined"),
            Status::Closed => ("[+]", "closed"),
        }
    }
}

pub fn shell_name(audit: &str) -> String {
    let mut name = format!("audit-{audit}");
    name.truncate(24);
    name
}

/// Where a claim stands with the operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Review {
    /// The agent's, not reviewed (`[?]`).
    #[default]
    Proposed,
    Accepted,
    Rejected,
    /// Accepted, then dropped by the operator.
    Dropped,
}

/// A check's outcome, as the harness wrote it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    #[default]
    Unchecked,
    Holds,
    Fails,
    /// Exit 2, a timeout, or an interruption: excluded.
    Error,
    /// It held, but caught no mutant: counts as documented.
    Vacuous,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Unchecked => "unchecked",
            Verdict::Holds => "holds",
            Verdict::Fails => "fails",
            Verdict::Error => "error",
            Verdict::Vacuous => "vacuous",
        }
    }
}

/// Whether a failure was reproduced on the real code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Repro {
    #[default]
    None,
    /// Submitted, waiting to run.
    Pending,
    /// It failed on the audited code: the failure is a finding.
    Fails,
    /// It passed on the audited code: not a repro.
    Passes,
    /// None can be written; the agent's reason is on the claim.
    ModelOnly,
}

/// What became of a result for the board (`post`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[allow(clippy::enum_variant_names)]
pub enum Posted {
    #[default]
    No,
    /// Waiting for the operator (`post review`).
    Ready,
    Posted,
    Discarded,
}

/// A claim: what the code or its docs say, and everything known about it.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Claim {
    /// `c<N>`, or `a<N>` for an assumption.
    pub id: String,
    pub audit: String,
    pub text: String,
    /// `s0ma` or `operator`: whose wording it is. The operator's is final.
    pub author: String,
    pub source: String,
    /// pure, invariant, protocol, equivalence, resource, assumes, conflict.
    pub kind: String,
    pub risk: i64,
    pub review: Review,
    /// The agent's reason it is in the plan, its tool and depth (the log has more).
    pub why: String,
    /// An open question: numbered readings, and the one picked.
    pub readings: Vec<String>,
    pub picked: Option<usize>,
    // ---- the plan
    pub priority: i64,
    pub pinned: bool,
    pub skipped: bool,
    /// Its check was refused for a reason only the operator or the harness
    /// can lift (an adapter that has not passed its doctor run, say): why.
    /// Not the same as skipped: lifted on its own once the reason goes.
    pub blocked: Option<String>,
    /// The adapter it waits on, when that is the reason.
    pub blocked_on: Option<String>,
    pub depth: Option<Level>,
    pub tool: String,
    pub estimate_secs: u64,
    /// Added after the last Go: waits for another.
    pub new: bool,
    // ---- the work
    /// the operator took it: he writes its check, the agent never does.
    pub taken: bool,
    pub held_by: Option<String>,
    pub held_until: Option<String>,
    /// A direction for its worker, waiting to be taken.
    pub steer: Option<String>,
    /// The check the direction is for (`k2`): written again if it exists,
    /// added if not. None: its first.
    pub steer_check: Option<String>,
    // ---- the result (the harness's only)
    pub verdict: Verdict,
    pub level: Level,
    pub kills: Option<(usize, usize)>,
    pub repro: Repro,
    pub repro_test: Option<String>,
    pub repro_output: Option<String>,
    pub model_only_why: Option<String>,
    pub model_only_accepted: bool,
    pub trace: Option<String>,
    pub checked_at: Option<String>,
    pub running: bool,
    /// A weaker check waiting for the operator (`[!] weaker`), and why.
    pub weaker: Option<String>,
    pub waived: bool,
    pub stale: Option<String>,
    pub unsound: Option<String>,
    pub fixed: Option<String>,
    // ---- where it went
    pub posted: Posted,
    pub fact: Option<String>,
    /// A board lead its results land under (`claim-to-lead`).
    pub lead: Option<(String, String)>,
    /// Assumptions its check used (`rests_on`), and the audit and claim
    /// that discharged this one if it is an assumption.
    pub rests_on: Vec<String>,
    pub discharged_by: Option<(String, String)>,
    /// Report rounds that cite it (frozen at the version cited).
    pub cited: Vec<String>,
    pub created_at: String,
}

impl Claim {
    pub fn is_assumption(&self) -> bool {
        self.kind == "assumes"
    }

    /// Whether its result counts as assurance: reviewed, checked, not
    /// waived, not resting on an unaccepted assumption. Diagnose results on
    /// unreviewed claims never do (rule 3).
    pub fn counts(&self) -> bool {
        self.review == Review::Accepted && matches!(self.verdict, Verdict::Holds) && !self.waived && self.unsound.is_none()
    }

    /// The mark and its style for the ledger, `[!]` first.
    pub fn mark(&self) -> (&'static str, Tone) {
        if self.weaker.is_some() || self.blocked_on_operator() || (self.repro == Repro::ModelOnly && !self.model_only_accepted) {
            return ("[!]", Tone::Warn);
        }
        if self.blocked.is_some() {
            return ("[*]", Tone::Warn);
        }
        if self.unsound.is_some() || self.stale.is_some() {
            return ("[?]", Tone::Warn);
        }
        if self.running || self.held_by.is_some() {
            return ("[*]", Tone::Warn);
        }
        if self.waived {
            return ("[-]", Tone::Dim);
        }
        match self.verdict {
            Verdict::Fails => ("[-]", Tone::Danger),
            Verdict::Holds if self.review == Review::Proposed => ("[+]", Tone::Dim),
            Verdict::Holds => ("[+]", Tone::Success),
            Verdict::Vacuous => ("[~]", Tone::Dim),
            Verdict::Error => ("[-]", Tone::Dim),
            Verdict::Unchecked if self.review == Review::Proposed => ("[?]", Tone::Plain),
            Verdict::Unchecked => ("[ ]", Tone::Dim),
        }
    }

    /// Sort key: what wants the operator first, then failing, checking,
    /// holding by risk.
    pub fn rank(&self) -> (u8, i64) {
        let group = match self.mark().0 {
            "[!]" => 0,
            "[?]" => 1,
            "[-]" if self.verdict == Verdict::Fails && !self.waived => 2,
            "[*]" => 3,
            "[ ]" => 4,
            "[+]" => 5,
            _ => 6,
        };
        (group, -self.risk)
    }

    /// Whether something about it is the operator's to answer before it
    /// can go on: a review the audit requires (assure mode, or the
    /// `reviewed` gate; elsewhere an unreviewed claim is checked anyway),
    /// a weaker check, a model-only reason, a result to post, an open
    /// question, or a refusal only he can lift.
    pub fn waits_on(&self, audit: &Audit) -> bool {
        if matches!(self.review, Review::Rejected | Review::Dropped) || self.waived {
            return false;
        }
        let review_needed = audit.mode == Mode::Assure || audit.settings.gate == Gate::Reviewed;
        (self.review == Review::Proposed && review_needed && !self.is_assumption())
            || self.weaker.is_some()
            || (self.repro == Repro::ModelOnly && !self.model_only_accepted)
            || self.posted == Posted::Ready
            || (!self.readings.is_empty() && self.picked.is_none())
            || self.blocked_on_operator()
    }

    /// Blocked by something only the operator can lift: any refusal but an
    /// adapter whose doctor run is still to pass (it runs on its own).
    fn blocked_on_operator(&self) -> bool {
        self.blocked.is_some()
            && self.blocked_on.as_deref().is_none_or(|lang| {
                store().and_then(|s| s.adapter_state(lang).ok().flatten()).is_some_and(|state| state.doctor_at.is_some() && !state.doctor_passed)
            })
    }

    /// Ready to be checked under the gate, in this mode.
    pub fn checkable(&self, audit: &Audit) -> bool {
        if matches!(self.review, Review::Rejected | Review::Dropped) || self.skipped || self.blocked.is_some() || self.waived {
            return false;
        }
        if self.is_assumption() {
            return false;
        }
        // Assure mode, or the `reviewed` gate: only what the operator accepted.
        let reviewed = !(audit.mode == Mode::Assure || audit.settings.gate == Gate::Reviewed) || self.review == Review::Accepted;
        // The `plan` gate: nothing before go, and rows added since wait for another.
        let gone = audit.settings.gate != Gate::Plan || (!self.new && audit.gone_at.is_some());
        reviewed && gone
    }
}

/// A tone, as the tab colours it (`audit_view` maps it to a style).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Plain,
    Dim,
    Warn,
    Danger,
    Success,
}

/// A check submitted for a claim.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Check {
    pub claim: String,
    /// Which of the claim's checks it is (`k1`, `k2`…): a claim has several,
    /// a test and a bounded check, say; a new version replaces its own.
    pub slot: String,
    pub version: i64,
    /// `checks/c<N>/` in the work tree.
    pub dir: String,
    pub lang: String,
    pub kind: String,
    pub run: String,
    pub timeout: u64,
    pub targets: Vec<String>,
    pub breaks: Vec<String>,
    /// A command that writes the check's coverage as lcov to `$COV`
    /// (`cargo llvm-cov`, `coverage lcov`, `c8`…): the claim view's gutter.
    pub coverage: String,
    pub assumes: Vec<String>,
    /// Clause anchors found in its files: `[1]`, `[2]`…
    pub anchors: Vec<usize>,
    pub by: String,
    /// The worker job that wrote it, by its full id (`by` is its label,
    /// which repeats across restarts).
    pub by_job: String,
    pub commit: String,
    pub back_translation: Option<String>,
    /// The job that read it back, never the one that wrote it.
    pub back_by: String,
    pub kills: Option<(usize, usize)>,
    /// Mutants it killed, by name (strengthen-only compares these).
    pub killed: Vec<String>,
    pub survived: Vec<String>,
    /// Mutants that never ran as mutants (didn't apply, a tool missing, the
    /// check couldn't start), with why: counted neither way.
    pub broken: Vec<String>,
    pub superseded_by: Option<i64>,
    /// Waiting for the operator, weaker than the one it would replace.
    pub held_weaker: bool,
    pub submitted_at: String,
    // ---- its last run (the harness's only)
    pub verdict: Verdict,
    pub level: Level,
    pub checked_at: Option<String>,
    /// Where its counterexample is kept, when it failed.
    pub trace: Option<String>,
}

impl Check {
    /// Its name within the claim; checks from before claims had several
    /// are `k1`.
    pub fn slot(&self) -> &str {
        if self.slot.is_empty() { "k1" } else { &self.slot }
    }

    fn slot_number(&self) -> u32 {
        slot_number(self.slot())
    }
}

/// Checks from before a claim had several: each becomes `k1`, and the one
/// in force takes its claim's last result (kept on the claim until now).
fn migrate_checks(db: &Connection) -> Result<()> {
    let rows: Vec<(String, String)> = {
        let mut statement = db.prepare("SELECT audit, data FROM checks WHERE data NOT LIKE '%\"slot\"%' ORDER BY version")?;
        let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    if rows.is_empty() {
        return Ok(());
    }
    let mut checks: Vec<(String, Check)> = Vec::new();
    for (audit, data) in rows {
        checks.push((audit, serde_json::from_str(&data)?));
    }
    for i in 0..checks.len() {
        let (audit, check) = &checks[i];
        let in_force = !check.held_weaker
            && check.superseded_by.is_none()
            && !checks[i + 1..].iter().any(|(a, later)| a == audit && later.claim == check.claim && !later.held_weaker && later.superseded_by.is_none());
        let mut check = check.clone();
        check.slot = "k1".into();
        if in_force {
            let data: Option<String> =
                db.query_row("SELECT data FROM claims WHERE audit=?1 AND id=?2", params![audit, check.claim], |row| row.get(0)).optional()?;
            if let Some(claim) = data.and_then(|d| serde_json::from_str::<Claim>(&d).ok()) {
                check.verdict = claim.verdict;
                check.level = claim.level;
                check.checked_at = claim.checked_at.clone();
                check.trace = claim.trace.clone();
            }
        }
        db.execute(
            "UPDATE checks SET data=?4 WHERE audit=?1 AND claim=?2 AND version=?3",
            params![audit, check.claim, check.version, serde_json::to_string(&check)?],
        )?;
    }
    Ok(())
}

/// `k3` → 3.
pub fn slot_number(slot: &str) -> u32 {
    slot.strip_prefix('k').and_then(|n| n.parse().ok()).unwrap_or(1)
}

/// A claim's verdict and level from its checks in force: failing if any
/// fails; else holding at the deepest level any holds at; else vacuous, or
/// an error; unchecked while none has run.
pub fn roll_up(checks: &[Check]) -> (Verdict, Level) {
    if checks.iter().any(|c| c.verdict == Verdict::Fails) {
        return (Verdict::Fails, Level::Documented);
    }
    if let Some(level) = checks.iter().filter(|c| c.verdict == Verdict::Holds).map(|c| c.level).max() {
        return (Verdict::Holds, level);
    }
    if checks.iter().any(|c| c.verdict == Verdict::Vacuous) {
        return (Verdict::Vacuous, Level::Documented);
    }
    if checks.iter().any(|c| c.verdict == Verdict::Error) {
        return (Verdict::Error, Level::Documented);
    }
    (Verdict::Unchecked, Level::Documented)
}

/// The check one line speaks for: a failing one, else the deepest that
/// holds, else the newest.
pub fn lead(checks: &[Check]) -> Option<&Check> {
    checks
        .iter()
        .find(|c| c.verdict == Verdict::Fails)
        .or_else(|| checks.iter().filter(|c| c.verdict == Verdict::Holds).max_by_key(|c| c.level))
        .or_else(|| checks.iter().max_by_key(|c| c.version))
}

/// Where a claim's check lives in the work tree: `checks/c3/k2`. A claim
/// whose first check predates several (`checks/c3` itself) puts the others
/// beside it (`checks/c3-k2`), never inside.
pub fn check_dir(claim: &str, slot: &str, existing: &[Check]) -> String {
    if let Some(same) = existing.iter().rev().find(|c| c.slot() == slot) {
        return same.dir.clone();
    }
    if existing.iter().any(|c| c.dir == format!("checks/{claim}")) {
        format!("checks/{claim}-{slot}")
    } else {
        format!("checks/{claim}/{slot}")
    }
}

/// The next check a claim can have: `k1` with none, else one past the last.
pub fn next_slot(existing: &[Check]) -> String {
    format!("k{}", existing.iter().map(Check::slot_number).max().unwrap_or(0) + 1)
}

/// One entry of the log: a choice and its reason, or a change of the
/// operator's.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LogEntry {
    pub at: String,
    pub who: String,
    pub what: String,
    pub why: String,
}

/// One `audit_run` batch.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Run {
    pub n: i64,
    pub claims: Vec<String>,
    pub commands: Vec<RunCommand>,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub approval: String,
    pub transcript: String,
    pub note: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RunCommand {
    pub claim: String,
    pub what: String,
    pub command: String,
    pub exit: Option<i32>,
    pub verdict: String,
    pub ms: u64,
    pub by_operator: bool,
    /// What it printed, stdout and stderr together as the shell showed them
    /// (clipped; the run's transcript has it whole).
    pub output: String,
}

/// A link between an audit and a board.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Link {
    pub audit: String,
    pub board: String,
    /// owner, linked, lead (a claim to a lead), gate, origin.
    pub kind: String,
    pub claim: Option<String>,
    pub lead: Option<String>,
    pub by: String,
    pub at: String,
    pub reason: String,
}

/// One piece of evidence: a mutant, a repro run, a trace, a loophole.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Evidence {
    pub claim: String,
    /// The check it is about (`k2`); empty for the claim's own (a repro).
    pub check: String,
    pub kind: String,
    pub name: String,
    pub path: String,
    pub result: String,
    pub at: String,
}

// ------------------------------------------------------------------ store

pub struct AuditStore {
    path: PathBuf,
}

/// Whose change it is: never set by a tool's argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Who {
    Operator,
    S0ma,
    Harness,
}

impl Who {
    pub fn label(self) -> &'static str {
        match self {
            Who::Operator => "operator",
            Who::S0ma => "s0ma",
            Who::Harness => "harness",
        }
    }
}

impl AuditStore {
    pub fn at(path: PathBuf) -> Self {
        Self { path }
    }

    /// A connection to the store. The schema is checked and its tables made
    /// once per file in a process: a read is only a read, with no write (and
    /// sync) behind it, so the tab drawing every frame never queues behind
    /// the scheduler and workers for the write lock.
    fn connect(&self) -> Result<crate::projects::journal::Db> {
        let db = self.connect_raw()?;
        Ok(crate::projects::journal::wrap(db, crate::projects::journal::Store::Audit, &self.path))
    }

    fn connect_raw(&self) -> Result<Connection> {
        static READY: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
        if self.path.exists() && crate::util::locked(&READY).contains(&self.path) {
            return Ok(crate::db::open(&self.path)?);
        }
        if self.path.exists() {
            let schema: Option<String> = Connection::open(&self.path)
                .ok()
                .and_then(|db| db.query_row("SELECT value FROM metadata WHERE key='schema'", [], |row| row.get(0)).optional().ok().flatten());
            if schema.as_deref().is_some_and(|schema| schema != SCHEMA) {
                let aside = self.path.with_extension(format!("sqlite3.superseded-{}", Local::now().format("%Y%m%dT%H%M%S")));
                std::fs::rename(&self.path, &aside)?;
            }
        }
        let db = crate::db::open(&self.path)?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS audits(name TEXT PRIMARY KEY, seq INTEGER NOT NULL, board TEXT, status TEXT NOT NULL, data TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS claims(audit TEXT NOT NULL, id TEXT NOT NULL, seq INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(audit, id));
             CREATE TABLE IF NOT EXISTS checks(audit TEXT NOT NULL, claim TEXT NOT NULL, version INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(audit, claim, version));
             CREATE TABLE IF NOT EXISTS evidence(seq INTEGER PRIMARY KEY AUTOINCREMENT, audit TEXT NOT NULL, data TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS links(seq INTEGER PRIMARY KEY AUTOINCREMENT, audit TEXT NOT NULL, board TEXT NOT NULL, data TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS log(seq INTEGER PRIMARY KEY AUTOINCREMENT, audit TEXT NOT NULL, data TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS runs(audit TEXT NOT NULL, n INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY(audit, n));
             CREATE TABLE IF NOT EXISTS policies(board TEXT PRIMARY KEY, data TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS adapters(lang TEXT PRIMARY KEY, data TEXT NOT NULL);",
        )?;
        db.execute("INSERT OR IGNORE INTO metadata VALUES('schema', ?1)", params![SCHEMA])?;
        migrate_checks(&db)?;
        let mut ready = crate::util::locked(&READY);
        if !ready.contains(&self.path) {
            ready.push(self.path.clone());
        }
        Ok(db)
    }

    // ---- audits

    pub fn audits(&self) -> Result<Vec<Audit>> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT data FROM audits ORDER BY seq")?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    }

    pub fn audit(&self, name: &str) -> Result<Option<Audit>> {
        let db = self.connect()?;
        let data: Option<String> = db.query_row("SELECT data FROM audits WHERE name=?1", params![name], |row| row.get(0)).optional()?;
        Ok(match data {
            Some(data) => Some(serde_json::from_str(&data)?),
            None => None,
        })
    }

    /// An audit and everything the store has of it, gone: its claims,
    /// checks, evidence, runs, log and links. Findings it posted are the
    /// boards' and stay.
    pub fn delete_audit(&self, name: &str) -> Result<()> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for table in ["claims", "checks", "evidence", "links", "log", "runs"] {
            tx.execute(&format!("DELETE FROM {table} WHERE audit=?1"), params![name])?;
        }
        tx.execute("DELETE FROM audits WHERE name=?1", params![name])?;
        tx.commit()?;
        Ok(())
    }

    pub fn create(&self, audit: &Audit) -> std::result::Result<(), String> {
        // An audit's name is its id in the files: one archived keeps it.
        if crate::projects::archived(crate::projects::Kind::Audit, &audit.name) {
            return Err(format!("an archived audit is named {} (bring it back from the archive, or choose another name)", audit.name));
        }
        let db = self.connect().map_err(|e| e.to_string())?;
        let seq: i64 = db.query_row("SELECT COALESCE(MAX(seq),0)+1 FROM audits", [], |row| row.get(0)).map_err(|e| e.to_string())?;
        db.execute(
            "INSERT INTO audits VALUES(?1, ?2, ?3, ?4, ?5)",
            params![audit.name, seq, audit.board, audit.status.label(), serde_json::to_string(audit).map_err(|e| e.to_string())?],
        )
        .map_err(|error| match error {
            rusqlite::Error::SqliteFailure(_, _) => format!("an audit named {} exists already", audit.name),
            other => other.to_string(),
        })?;
        Ok(())
    }

    /// Change an audit in one transaction. The change says what it did.
    pub fn change<T>(&self, name: &str, change: impl FnOnce(&mut Audit) -> std::result::Result<T, String>) -> Result<std::result::Result<T, String>> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let data: Option<String> = tx.query_row("SELECT data FROM audits WHERE name=?1", params![name], |row| row.get(0)).optional()?;
        let Some(data) = data else { return Ok(Err(format!("no audit named {name}"))) };
        let mut audit: Audit = serde_json::from_str(&data)?;
        let out = match change(&mut audit) {
            Ok(out) => out,
            Err(error) => return Ok(Err(error)),
        };
        tx.execute(
            "UPDATE audits SET board=?2, status=?3, data=?4 WHERE name=?1",
            params![name, audit.board, audit.status.label(), serde_json::to_string(&audit)?],
        )?;
        tx.commit()?;
        Ok(Ok(out))
    }

    // ---- claims

    pub fn claims(&self, audit: &str) -> Result<Vec<Claim>> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT data FROM claims WHERE audit=?1 ORDER BY seq")?;
        let rows = statement.query_map(params![audit], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    }

    pub fn claim(&self, audit: &str, id: &str) -> Result<Option<Claim>> {
        let db = self.connect()?;
        let data: Option<String> =
            db.query_row("SELECT data FROM claims WHERE audit=?1 AND id=?2", params![audit, id], |row| row.get(0)).optional()?;
        Ok(match data {
            Some(data) => Some(serde_json::from_str(&data)?),
            None => None,
        })
    }

    /// A new claim, numbered `c<N>` (or `a<N>` for an assumption).
    pub fn add_claim(&self, audit: &str, mut claim: Claim) -> Result<Claim> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let prefix = if claim.is_assumption() { "a" } else { "c" };
        let seq: i64 = tx.query_row("SELECT COALESCE(MAX(seq),0)+1 FROM claims WHERE audit=?1", params![audit], |row| row.get(0))?;
        let n: i64 = tx.query_row(
            "SELECT COUNT(*)+1 FROM claims WHERE audit=?1 AND id LIKE ?2",
            params![audit, format!("{prefix}%")],
            |row| row.get(0),
        )?;
        claim.id = format!("{prefix}{n}");
        claim.audit = audit.to_string();
        if claim.created_at.is_empty() {
            claim.created_at = Local::now().to_rfc3339();
        }
        tx.execute("INSERT INTO claims VALUES(?1, ?2, ?3, ?4)", params![audit, claim.id, seq, serde_json::to_string(&claim)?])?;
        tx.commit()?;
        Ok(claim)
    }

    pub fn change_claim<T>(&self, audit: &str, id: &str, change: impl FnOnce(&mut Claim) -> std::result::Result<T, String>) -> Result<std::result::Result<T, String>> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let data: Option<String> =
            tx.query_row("SELECT data FROM claims WHERE audit=?1 AND id=?2", params![audit, id], |row| row.get(0)).optional()?;
        let Some(data) = data else { return Ok(Err(format!("no claim {id} in {audit}"))) };
        let mut claim: Claim = serde_json::from_str(&data)?;
        let out = match change(&mut claim) {
            Ok(out) => out,
            Err(error) => return Ok(Err(error)),
        };
        tx.execute("UPDATE claims SET data=?3 WHERE audit=?1 AND id=?2", params![audit, id, serde_json::to_string(&claim)?])?;
        tx.commit()?;
        Ok(Ok(out))
    }

    // ---- checks

    pub fn checks(&self, audit: &str, claim: &str) -> Result<Vec<Check>> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT data FROM checks WHERE audit=?1 AND claim=?2 ORDER BY version")?;
        let rows = statement.query_map(params![audit, claim], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    }

    /// The checks in force, one per slot (the latest version not held as
    /// weaker nor superseded), in slot order.
    pub fn current_checks(&self, audit: &str, claim: &str) -> Result<Vec<Check>> {
        let mut by_slot: std::collections::BTreeMap<u32, Check> = std::collections::BTreeMap::new();
        for check in self.checks(audit, claim)? {
            if !check.held_weaker && check.superseded_by.is_none() {
                by_slot.insert(check.slot_number(), check);
            }
        }
        Ok(by_slot.into_values().collect())
    }

    /// The check one line speaks for (`lead`).
    pub fn current_check(&self, audit: &str, claim: &str) -> Result<Option<Check>> {
        Ok(lead(&self.current_checks(audit, claim)?).cloned())
    }

    pub fn add_check(&self, audit: &str, mut check: Check) -> Result<Check> {
        let db = self.connect()?;
        let version: i64 =
            db.query_row("SELECT COALESCE(MAX(version),0)+1 FROM checks WHERE audit=?1 AND claim=?2", params![audit, check.claim], |row| row.get(0))?;
        check.version = version;
        check.submitted_at = Local::now().to_rfc3339();
        db.execute("INSERT INTO checks VALUES(?1, ?2, ?3, ?4)", params![audit, check.claim, version, serde_json::to_string(&check)?])?;
        Ok(check)
    }

    pub fn put_check(&self, audit: &str, check: &Check) -> Result<()> {
        let db = self.connect()?;
        db.execute(
            "UPDATE checks SET data=?4 WHERE audit=?1 AND claim=?2 AND version=?3",
            params![audit, check.claim, check.version, serde_json::to_string(check)?],
        )?;
        Ok(())
    }

    // ---- evidence, log, runs, links

    pub fn add_evidence(&self, audit: &str, evidence: &Evidence) -> Result<()> {
        let db = self.connect()?;
        db.execute("INSERT INTO evidence(audit, data) VALUES(?1, ?2)", params![audit, serde_json::to_string(evidence)?])?;
        Ok(())
    }

    pub fn evidence(&self, audit: &str, claim: Option<&str>) -> Result<Vec<Evidence>> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT data FROM evidence WHERE audit=?1 ORDER BY seq")?;
        let rows = statement.query_map(params![audit], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            let evidence: Evidence = serde_json::from_str(&row?)?;
            if claim.is_none_or(|claim| evidence.claim == claim) {
                out.push(evidence);
            }
        }
        Ok(out)
    }

    pub fn log(&self, audit: &str, who: Who, what: &str, why: &str) -> Result<()> {
        let db = self.connect()?;
        let entry = LogEntry { at: Local::now().to_rfc3339(), who: who.label().into(), what: what.into(), why: why.into() };
        db.execute("INSERT INTO log(audit, data) VALUES(?1, ?2)", params![audit, serde_json::to_string(&entry)?])?;
        Ok(())
    }

    pub fn log_entries(&self, audit: &str) -> Result<Vec<LogEntry>> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT data FROM log WHERE audit=?1 ORDER BY seq")?;
        let rows = statement.query_map(params![audit], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    }

    pub fn new_run(&self, audit: &str, mut run: Run) -> Result<Run> {
        let db = self.connect()?;
        run.n = db.query_row("SELECT COALESCE(MAX(n),0)+1 FROM runs WHERE audit=?1", params![audit], |row| row.get(0))?;
        db.execute("INSERT INTO runs VALUES(?1, ?2, ?3)", params![audit, run.n, serde_json::to_string(&run)?])?;
        Ok(run)
    }

    pub fn put_run(&self, audit: &str, run: &Run) -> Result<()> {
        let db = self.connect()?;
        db.execute("UPDATE runs SET data=?3 WHERE audit=?1 AND n=?2", params![audit, run.n, serde_json::to_string(run)?])?;
        Ok(())
    }

    pub fn runs(&self, audit: &str) -> Result<Vec<Run>> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT data FROM runs WHERE audit=?1 ORDER BY n")?;
        let rows = statement.query_map(params![audit], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    }

    pub fn add_link(&self, link: &Link) -> Result<()> {
        let db = self.connect()?;
        db.execute("INSERT INTO links(audit, board, data) VALUES(?1, ?2, ?3)", params![link.audit, link.board, serde_json::to_string(link)?])?;
        Ok(())
    }

    pub fn links(&self, audit: Option<&str>, board: Option<&str>) -> Result<Vec<Link>> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT data FROM links ORDER BY seq")?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            let link: Link = serde_json::from_str(&row?)?;
            if audit.is_none_or(|audit| link.audit == audit) && board.is_none_or(|board| link.board == board) {
                out.push(link);
            }
        }
        Ok(out)
    }

    /// Links matching, removed. How many.
    pub fn remove_links(&self, audit: &str, board: &str, kind: Option<&str>, claim: Option<&str>) -> Result<usize> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT seq, data FROM links WHERE audit=?1 AND board=?2")?;
        let rows: Vec<(i64, String)> = statement.query_map(params![audit, board], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
        let mut removed = 0;
        for (seq, data) in rows {
            let link: Link = serde_json::from_str(&data)?;
            if kind.is_none_or(|kind| link.kind == kind) && claim.is_none_or(|claim| link.claim.as_deref() == Some(claim)) {
                db.execute("DELETE FROM links WHERE seq=?1", params![seq])?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    // ---- policies

    pub fn policy(&self, board: &str) -> Result<Policy> {
        let db = self.connect()?;
        let data: Option<String> = db.query_row("SELECT data FROM policies WHERE board=?1", params![board], |row| row.get(0)).optional()?;
        Ok(match data {
            Some(data) => serde_json::from_str::<PolicyRow>(&data)?.policy,
            None => self.default_policy()?,
        })
    }

    /// Whether a project has a policy of its own (else the default).
    pub fn has_policy(&self, board: &str) -> Result<bool> {
        let db = self.connect()?;
        Ok(db.query_row("SELECT 1 FROM policies WHERE board=?1", params![board], |_| Ok(())).optional()?.is_some())
    }

    pub fn ceilings(&self, board: &str) -> Result<Ceilings> {
        let db = self.connect()?;
        let data: Option<String> = db.query_row("SELECT data FROM policies WHERE board=?1", params![board], |row| row.get(0)).optional()?;
        Ok(match data {
            Some(data) => serde_json::from_str::<PolicyRow>(&data)?.ceilings,
            None => Ceilings::default(),
        })
    }

    pub fn set_policy(&self, board: &str, policy: Policy, ceilings: Option<Ceilings>) -> Result<()> {
        let db = self.connect()?;
        let ceilings = match ceilings {
            Some(ceilings) => ceilings,
            None => self.ceilings(board)?,
        };
        db.execute(
            "INSERT OR REPLACE INTO policies VALUES(?1, ?2)",
            params![board, serde_json::to_string(&PolicyRow { policy, ceilings })?],
        )?;
        Ok(())
    }

    /// The policy new projects start with (the overview's row).
    pub fn default_policy(&self) -> Result<Policy> {
        let db = self.connect()?;
        let value: Option<String> = db.query_row("SELECT value FROM metadata WHERE key='default_policy'", [], |row| row.get(0)).optional()?;
        Ok(value.as_deref().and_then(Policy::parse).unwrap_or_default())
    }

    pub fn set_default_policy(&self, policy: Policy) -> Result<()> {
        let db = self.connect()?;
        db.execute("INSERT OR REPLACE INTO metadata VALUES('default_policy', ?1)", params![policy.label()])?;
        Ok(())
    }

    pub fn adapter_state(&self, lang: &str) -> Result<Option<AdapterState>> {
        let db = self.connect()?;
        let data: Option<String> = db.query_row("SELECT data FROM adapters WHERE lang=?1", params![lang], |row| row.get(0)).optional()?;
        Ok(match data {
            Some(data) => Some(serde_json::from_str(&data)?),
            None => None,
        })
    }

    pub fn set_adapter_state(&self, lang: &str, state: &AdapterState) -> Result<()> {
        let db = self.connect()?;
        db.execute("INSERT OR REPLACE INTO adapters VALUES(?1, ?2)", params![lang, serde_json::to_string(state)?])?;
        Ok(())
    }
}

/// What `auto` may start without asking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Ceilings {
    pub steps: i64,
    pub depth: Level,
    pub workers: usize,
}

impl Default for Ceilings {
    fn default() -> Self {
        Ceilings { steps: 20, depth: Level::Property, workers: 2 }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PolicyRow {
    policy: Policy,
    #[serde(default)]
    ceilings: Ceilings,
}

/// An adapter's state on this machine: its doctor run, its file's hash, and
/// whether the operator approved a proposed one.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AdapterState {
    pub doctor_passed: bool,
    pub doctor_at: Option<String>,
    pub doctor_note: String,
    pub file_hash: String,
    /// A proposed adapter, waiting for the operator (`[!] adapter`).
    pub proposed: Option<String>,
    pub proposed_why: String,
}

// ------------------------------------------------------------------ the store, shared

static STORE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// The store beside the archive, set once at start.
pub fn init(archive: &Path) {
    *crate::util::locked(&STORE) = Some(archive.with_file_name("audit.sqlite3"));
    crate::audit::run::start_scheduler();
}

/// An attached terminal's: the board store (the projects tab makes boards
/// and comments there). Not the audit store: what the audit tab does runs
/// in the primary's scheduler, which this process hasn't.
pub fn init_attached(archive: &Path) {
    *crate::util::locked(&BOARDS_ONLY) = Some(archive.with_file_name("audit.sqlite3"));
}

static BOARDS_ONLY: Mutex<Option<PathBuf>> = Mutex::new(None);

/// The data folder (where adapters live, under `audit/langs`).
pub fn data_dir() -> Option<PathBuf> {
    store_path().and_then(|path| path.parent().map(Path::to_path_buf))
}

pub fn store() -> Option<AuditStore> {
    store_path().map(AuditStore::at)
}

/// The board store beside this one.
pub fn boards() -> Option<crate::board::BoardStore> {
    store_path()
        .or_else(|| crate::util::locked(&BOARDS_ONLY).clone())
        .map(|path| crate::board::BoardStore::beside_archive(&path))
}

#[cfg(not(test))]
fn store_path() -> Option<PathBuf> {
    crate::util::locked(&STORE).clone()
}

// Tests run side by side: each thread its own store.
#[cfg(test)]
thread_local! {
    static TEST_STORE: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn store_path() -> Option<PathBuf> {
    TEST_STORE.with(|path| path.borrow().clone())
}

#[cfg(test)]
pub fn set_store_for_test(path: PathBuf) {
    TEST_STORE.with(|store| *store.borrow_mut() = Some(path));
}

pub fn valid_name(name: &str) -> std::result::Result<(), String> {
    if name.is_empty() || name.len() > MAX_NAME - 6 {
        return Err(format!("an audit's name has 1 to {} characters", MAX_NAME - 6));
    }
    if name.starts_with('-') || !name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_') {
        return Err("an audit's name has lowercase letters, digits, - and _".into());
    }
    Ok(())
}

pub fn clip_text(text: &str) -> String {
    let text = text.trim();
    if text.chars().count() > MAX_TEXT { crate::util::clip(text, MAX_TEXT) } else { text.to_string() }
}

pub fn now() -> String {
    Local::now().to_rfc3339()
}

pub fn expired(until: Option<&str>, now: DateTime<Local>) -> bool {
    until.and_then(|at| DateTime::parse_from_rfc3339(at).ok()).is_none_or(|at| at.with_timezone(&Local) < now)
}

// ------------------------------------------------------------------ small parts

/// A claim's clauses: split at "and", ";" and ", and", numbered from 1. One
/// clause when it has no joints. Checks anchor each (`// [1]`).
pub fn clauses(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for part in text.split(';') {
        for piece in part.split(" and ") {
            let piece = piece.trim().trim_end_matches('.').trim_start_matches(", ").trim();
            if piece.split_whitespace().count() >= 2 {
                out.push(piece.to_string());
            } else if let Some(last) = out.last_mut()
                && !piece.is_empty()
            {
                last.push_str(" and ");
                last.push_str(piece);
            }
        }
    }
    if out.is_empty() {
        out.push(text.trim().to_string());
    }
    out
}

/// A value of the flat TOML the harness reads (check.toml, adapters):
/// strings, numbers, booleans, arrays of strings, and inline tables.
#[derive(Clone, Debug, PartialEq)]
pub enum TomlValue {
    Text(String),
    List(Vec<String>),
    Table(std::collections::BTreeMap<String, String>),
}

impl TomlValue {
    pub fn text(&self) -> String {
        match self {
            TomlValue::Text(text) => text.clone(),
            TomlValue::List(list) => list.join(" "),
            TomlValue::Table(table) => table.get("run").cloned().unwrap_or_default(),
        }
    }
    pub fn list(&self) -> Vec<String> {
        match self {
            TomlValue::List(list) => list.clone(),
            TomlValue::Text(text) if !text.is_empty() => vec![text.clone()],
            _ => Vec::new(),
        }
    }
}

/// Flat TOML: `key = value` per line, `#` comments, no sections.
pub fn toml_flat(text: &str) -> std::result::Result<std::collections::BTreeMap<String, TomlValue>, String> {
    fn scalar(raw: &str) -> String {
        let raw = raw.trim();
        if (raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2) || (raw.starts_with('\'') && raw.ends_with('\'') && raw.len() >= 2) {
            raw[1..raw.len() - 1].replace("\\\"", "\"").replace("\\\\", "\\")
        } else {
            raw.to_string()
        }
    }
    /// Split at commas outside quotes.
    fn split(inner: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut quote: Option<char> = None;
        for c in inner.chars() {
            match (quote, c) {
                (None, '"' | '\'') => {
                    quote = Some(c);
                    current.push(c);
                }
                (Some(q), c) if c == q => {
                    quote = None;
                    current.push(c);
                }
                (None, ',') => parts.push(std::mem::take(&mut current)),
                _ => current.push(c),
            }
        }
        if !current.trim().is_empty() {
            parts.push(current);
        }
        parts
    }
    /// The value, without a trailing comment outside quotes.
    fn uncomment(value: &str) -> &str {
        let mut quote: Option<char> = None;
        for (i, c) in value.char_indices() {
            match (quote, c) {
                (None, '"' | '\'') => quote = Some(c),
                (Some(q), c) if c == q => quote = None,
                (None, '#') => return &value[..i],
                _ => {}
            }
        }
        value
    }
    let mut out = std::collections::BTreeMap::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
            continue;
        }
        let (key, value) = line.split_once('=').ok_or_else(|| format!("line {}: key = value", n + 1))?;
        let key = key.trim().to_string();
        let value = uncomment(value).trim();
        let parsed = if let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
            TomlValue::List(split(inner).iter().map(|p| scalar(p)).filter(|p| !p.is_empty()).collect())
        } else if let Some(inner) = value.strip_prefix('{').and_then(|v| v.strip_suffix('}')) {
            let mut table = std::collections::BTreeMap::new();
            for pair in split(inner) {
                if let Some((k, v)) = pair.split_once('=') {
                    table.insert(k.trim().to_string(), scalar(v));
                }
            }
            TomlValue::Table(table)
        } else {
            TomlValue::Text(scalar(value))
        };
        out.insert(key, parsed);
    }
    Ok(out)
}
