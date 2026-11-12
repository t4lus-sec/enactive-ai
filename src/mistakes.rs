//! `/mistake`: flagging the agent's mistakes during use, for nightly correction
//! training (`claude/mistake-capture-spec.md` §2–§4).
//!
//! The operator flags a model call the call store saved (calls.rs). A rewriter
//! (qref's `mistake-rewrite`) proposes a minimally corrected response in the
//! background; the operator reviews it (`/mistake view`, mistakes/view.rs) and
//! approves it, edits it by hand, or discards it. Nothing here ever executes a
//! corrected tool call (R2): a proposal is text to train on. The operator's
//! note is the rewriter's hint and lives only in the mistake's own files (R3);
//! nothing from this module reaches the archive, the runtime context or the
//! conversation. Only the operator approves (R4).
//!
//! Layout under the data folder:
//!
//! ```text
//! data/mistakes/<n>.json                    the current state, owned by the harness
//! data/mistakes/history.jsonl               every change, appended
//! data/mistakes/notes/<n>-<attempt>.txt     the note an attempt is given (every attempt)
//! data/mistakes/proposals/<n>-<attempt>.json written by the rewriter
//! data/mistakes/edits/<n>-<k>.json          the operator's manual versions, k = 1, 2, …
//! data/mistakes/logs/<n>-<attempt>.log      the rewriter's stdout and stderr
//! data/mistakes/logs/<n>-<attempt>.exit     its exit status, written as it exits
//! ```
//!
//! Every change to a record and its history line happen under one lock file
//! (`.lock`), so the review window may run as a process of its own beside the agent.

pub(crate) mod view;

use crate::calls::{self, CallRef, FlagEvent, Latest, PendingFlag, Store};
use crate::llm::ToolDefinition;
use crate::types::ToolCall;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};

pub type Result<T> = std::result::Result<T, String>;

/// The rewriter, as the spec gives it. `{data}`, `{mistake}`, `{attempt}` and
/// `{note_file}` are replaced by shell-quoted values; every path is absolute.
pub const DEFAULT_REWRITER: &str = "cd ~/qref-work/qref && ~/qref-venv/bin/python -m qref mistake-rewrite \
     --data {data} --mistake {mistake} --attempt {attempt} --note-file {note_file}";

/// Fraction held out: never trained, used to measure whether the mistake recurs.
const HELDOUT_FRACTION: f64 = 0.2;
/// A rewrite with no result after this long is given up on. Not the spec's
/// hour from logging: the rewriter queues behind qref's lock, and a training
/// night stops the engine for longer than that, so an hour would mark sound
/// rewrites lost (and a regenerated week-old mistake lost at once). A process
/// that is gone without a result is marked lost at the next look instead.
const REWRITE_GIVE_UP: Duration = Duration::from_secs(12 * 3600);
/// How often the agent looks for rewrites it did not launch itself (one launched
/// before a restart, or by a review window running on its own).
const WATCH_INTERVAL: Duration = Duration::from_secs(15);
/// A lock file that names no process is broken after this long.
const LOCK_STALE: Duration = Duration::from_secs(30);

// ------------------------------------------------------------------ the record

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Pending,
    Ready,
    NeedsEdit,
    Failed,
    Approved,
    Edited,
    Discarded,
}

impl State {
    pub fn label(self) -> &'static str {
        match self {
            State::Pending => "pending",
            State::Ready => "ready",
            State::NeedsEdit => "needs_edit",
            State::Failed => "failed",
            State::Approved => "approved",
            State::Edited => "edited",
            State::Discarded => "discarded",
        }
    }

    /// Waiting for the operator's review.
    pub fn waiting(self) -> bool {
        matches!(self, State::Ready | State::NeedsEdit | State::Failed)
    }

    fn reviewed(self) -> bool {
        matches!(self, State::Approved | State::Edited | State::Discarded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Heldout,
}

/// The split, decided when the mistake is logged, from the turn alone (R5).
///
/// Keyed on the turn rather than the call: two flags in one turn (the reply,
/// then the reasoning call before it) are one incident with near-identical
/// contexts, and must land on the same side, or the held-out one is measured
/// after training on a near-copy of it.
pub fn split_for(turn_id: &str) -> Split {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(format!("mistake-split:{turn_id}").as_bytes());
    let head = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    if f64::from(head) / 4_294_967_296.0 < HELDOUT_FRACTION {
        Split::Heldout
    } else {
        Split::Train
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteEntry {
    pub at: String,
    pub attempt: u32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Final {
    /// `proposal` or `manual`.
    pub source: String,
    /// The call the approved version corrects: a step of the record's
    /// stretch. What trains is this call's own request with the corrected
    /// response; the steps after it are not trained (their context holds the
    /// mistake). None only on approvals made before stretches existed, which
    /// corrected `call_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_call_id: Option<String>,
    /// Relative to `data/mistakes`: `proposals/12-1.json` or `edits/12-2.json`.
    pub file: String,
    /// sha256 of the corrected response in that file, as canonical JSON.
    pub corrected_sha256: String,
    pub approved_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rewrite {
    pub attempt: u32,
    pub launched_at: String,
    pub pid: Option<u32>,
    pub log: String,
    pub exit_file: String,
}

/// One step of a mistake's stretch: a call the correction may target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StretchCall {
    pub call_id: String,
    pub call_in_turn: u32,
    pub at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub mistake: u64,
    pub logged_at: String,
    pub call_id: String,
    pub turn_id: String,
    pub call_in_turn: u32,
    /// The latest note: the rewriter's hint, never trained (R3).
    pub note: String,
    pub notes: Vec<NoteEntry>,
    pub split: Split,
    /// What the split was hashed from: `mistake-split:` + this.
    pub split_key: String,
    pub weakness: Option<String>,
    pub state: State,
    pub attempts: u32,
    /// The latest proposal file, relative to `data/mistakes`.
    pub proposal: Option<String>,
    /// The version shown, and the only one `a` approves: a proposal or an
    /// edit file. `None` shows the original.
    pub current: Option<String>,
    #[serde(rename = "final")]
    pub final_: Option<Final>,
    /// The state before approval or discard: where `reopen` returns.
    pub before: Option<State>,
    pub rewrite: Option<Rewrite>,
    pub error: Option<String>,
    pub discard_reason: Option<String>,
    /// The calls since the operator last spoke (or the turn began), oldest
    /// first and ending with the flagged one (`call_id`): the steps a
    /// correction may target. By the flagged call the mistake is often
    /// already in what the agent saw; correcting the step where it started is
    /// what can teach anything. Empty on records from before: then the only
    /// step is `call_id`.
    #[serde(default)]
    pub stretch: Vec<StretchCall>,
    /// The step the version shown corrects. None: `call_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Who chose `target`: `rewriter` (its proposal) or `operator` (`t`, or
    /// an edit). A rewrite run again corrects the operator's choice; the
    /// rewriter's own it may revise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_by: Option<String>,
    /// Who flagged it: `s0ma` for its own `mistake_flag`; absent for the
    /// operator's `/mistake`. Either way the note is the rewriter's hint, a
    /// correction may not quote it, and only the operator approves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flagged_by: Option<String>,
    /// For the agent's flags: the turn it flagged from, which its per-turn limit
    /// counts (the mistake's own turn may be an earlier one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flagged_in: Option<String>,
    /// For the agent's flags: the archive event it pointed at, if it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pointed_at: Option<u64>,
}

/// How a flag is logged, beyond the call and the note.
#[derive(Debug, Default)]
pub struct LogOptions {
    /// Who flagged it (`s0ma`); None for the operator.
    pub by: Option<String>,
    pub flagged_in: Option<String>,
    pub pointed_at: Option<u64>,
    /// The steps, when the caller worked them out; else the call's stretch.
    pub steps: Option<Vec<StretchCall>>,
}

impl Record {
    /// The call the version shown corrects.
    pub fn target_id(&self) -> &str {
        self.target.as_deref().unwrap_or(&self.call_id)
    }

    /// The steps, oldest first: the stretch, or the flagged call alone.
    pub fn steps(&self) -> Vec<StretchCall> {
        if self.stretch.is_empty() {
            vec![StretchCall { call_id: self.call_id.clone(), call_in_turn: self.call_in_turn, at: self.logged_at.clone() }]
        } else {
            self.stretch.clone()
        }
    }

    /// Which step `call_id` is, from 1, and how many there are.
    pub fn step_of(&self, call_id: &str) -> Option<(usize, usize)> {
        let steps = self.steps();
        steps.iter().position(|step| step.call_id == call_id).map(|index| (index + 1, steps.len()))
    }
}

/// A response: the original as the call store saved it, a proposal's
/// `corrected`, or an edit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Response {
    pub content: String,
    pub reasoning_content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

impl Response {
    pub fn from_value(value: &Value) -> Result<Self> {
        let response: Self = serde_json::from_value(value.clone()).map_err(|e| format!("not a response: {e}"))?;
        Ok(response)
    }

    /// Every tool call's arguments parse as JSON.
    pub fn check_tool_calls(&self) -> Result<()> {
        for (index, call) in self.tool_calls.iter().enumerate() {
            if call.function.name.trim().is_empty() {
                return Err(format!("tool call {} has no name", index + 1));
            }
            if call.id.trim().is_empty() {
                return Err(format!("tool call {} ({}) has no id", index + 1, call.function.name));
            }
            if let Err(error) = serde_json::from_str::<Value>(&call.function.arguments) {
                return Err(format!("tool call {} ({}): arguments are not JSON: {error}", index + 1, call.function.name));
            }
        }
        Ok(())
    }
}

/// Canonical JSON: keys sorted by code point, no whitespace, UTF-8 as is
/// (Python: `json.dumps(v, sort_keys=True, separators=(",", ":"),
/// ensure_ascii=False)`). Sorted here explicitly rather than trusting the map
/// type, which a dependency's feature could change.
pub fn canonical_json(value: &Value) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (index, key) in keys.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(key).expect("strings serialise"));
                    out.push(':');
                    write(&map[*key], out);
                }
                out.push('}');
            }
            Value::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            other => out.push_str(&other.to_string()),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

pub fn response_sha256(response: &Response) -> String {
    let value = serde_json::to_value(response).expect("responses serialise");
    calls::sha256_hex(canonical_json(&value).as_bytes())
}

fn same_response(a: &Response, b: &Response) -> bool {
    response_sha256(a) == response_sha256(b)
}

#[derive(Debug, Clone, Serialize)]
struct HistoryLine<'a> {
    mistake: u64,
    at: String,
    from: Option<State>,
    to: Option<State>,
    by: &'a str,
    detail: String,
}

fn now() -> String {
    chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
}

/// The version shown in the review window.
#[derive(Debug, Clone)]
pub struct Version {
    /// `original`, `proposals/12-1.json` or `edits/12-2.json`.
    pub file: String,
    pub response: Response,
    pub manual: bool,
    /// The call it corrects (for `original`, the call it is).
    pub target: String,
}

/// A proposal file, as read.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub status: String,
    pub error: Option<String>,
    /// The step it corrects (`target_call_id`); None: the flagged call.
    pub target: Option<String>,
    pub corrected: Option<Response>,
    pub summary: Option<String>,
    pub edits: Option<usize>,
    pub changed_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

// ------------------------------------------------------------------ the book

/// The mistakes under one data folder.
#[derive(Debug, Clone)]
pub struct Book {
    data_dir: PathBuf,
    dir: PathBuf,
    rewriter: String,
    base_url: String,
    model: String,
}

/// `data/mistakes/.lock`, held for each change to a record and its history
/// line, and by the call store's retention while it decides what to keep.
/// It names its holder (`pid nonce`); a waiter breaks it only when that
/// process is gone (or, where liveness can't be checked, after a long
/// silence), and a holder removes it only while it still names itself.
pub(crate) struct Lock {
    path: PathBuf,
    owner: String,
}

impl Drop for Lock {
    fn drop(&mut self) {
        if fs::read_to_string(&self.path).is_ok_and(|text| text.trim() == self.owner) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Drop every mistake whose correction is built on one of `calls`, with all of
/// its files and its lines of the history. Returns (mistakes, history lines).
///
/// **The caller holds the mistakes lock.** The one caller is the call store's
/// job purge, which takes it before deciding anything and keeps it until the
/// index is rewritten; taking it again here would be a second claim from the
/// same process on a lock that names its holder.
///
/// A mistake is a correction of one saved call, trained against that call's own
/// request. Once the call is gone there is nothing to train and nothing to show
/// in the review window, so the record goes rather than being left pointing at
/// a call nobody can read. That includes an approved one: the approval names a
/// corrected response whose context no longer exists.
///
/// Every file of the mistake goes, not just the record. `next_number` takes the
/// highest record file and adds one, so a number comes back into use once its
/// record is deleted -- leaving its edits or proposals behind would hand them
/// to the next mistake to take that number.
pub(crate) fn drop_on_calls(
    data_dir: &Path,
    calls: &std::collections::HashSet<String>,
) -> Result<(usize, usize)> {
    let dir = data_dir.join("mistakes");
    let Ok(entries) = fs::read_dir(&dir) else { return Ok((0, 0)) };
    let mut doomed: Vec<u64> = Vec::new();
    for entry in entries {
        let path = entry.map_err(|e| format!("cannot read {}: {e}", dir.display()))?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(number) = path.file_stem().and_then(|n| n.to_str()).and_then(|n| n.parse::<u64>().ok()) else {
            continue;
        };
        let Ok(bytes) = fs::read(&path) else { continue };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
        // Every field that names a call: the flagged one, each step of its
        // stretch, the step the shown version corrects, and the step an
        // approval was made against. Any of them being gone makes the record
        // unusable, so any of them matching is enough.
        let mut named: Vec<String> = Vec::new();
        for pointer in ["/call_id", "/target", "/final/target_call_id"] {
            if let Some(id) = value.pointer(pointer).and_then(serde_json::Value::as_str) {
                named.push(id.to_owned());
            }
        }
        for step in value.get("stretch").and_then(serde_json::Value::as_array).into_iter().flatten() {
            if let Some(id) = step.get("call_id").and_then(serde_json::Value::as_str) {
                named.push(id.to_owned());
            }
        }
        if named.iter().any(|id| calls.contains(id)) {
            doomed.push(number);
        }
    }
    if doomed.is_empty() {
        return Ok((0, 0));
    }

    for number in &doomed {
        let _ = fs::remove_file(dir.join(format!("{number}.json")));
        for (folder, extensions) in
            [("notes", ["txt"].as_slice()), ("proposals", &["json"]), ("edits", &["json"]), ("logs", &["log", "exit"])]
        {
            let Ok(files) = fs::read_dir(dir.join(folder)) else { continue };
            for file in files.flatten() {
                let name = file.file_name().to_string_lossy().into_owned();
                let matches_number = name
                    .split_once('-')
                    .is_some_and(|(head, _)| head.parse::<u64>().ok() == Some(*number));
                let matches_kind = extensions.iter().any(|extension| name.ends_with(&format!(".{extension}")));
                if matches_number && matches_kind {
                    let _ = fs::remove_file(file.path());
                }
            }
        }
    }

    // The history names each mistake and what was done to it, so its lines go
    // too. It is append-only in normal running, which is why this is a rewrite
    // rather than an edit.
    let mut removed_lines = 0;
    let history = dir.join("history.jsonl");
    if history.exists() {
        let text = fs::read_to_string(&history).map_err(|e| format!("cannot read the history: {e}"))?;
        let mut kept = String::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let number = serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|value| value.get("mistake").and_then(serde_json::Value::as_u64));
            if number.is_some_and(|number| doomed.contains(&number)) {
                removed_lines += 1;
                continue;
            }
            kept.push_str(line);
            kept.push('\n');
        }
        write_atomic(&history, kept.as_bytes())?;
    }
    Ok((doomed.len(), removed_lines))
}

/// Take the mistakes lock under `data_dir`, waiting up to `wait`.
pub(crate) fn lock_mistakes(data_dir: &Path, wait: Duration) -> Result<Lock> {
    let dir = data_dir.join("mistakes");
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = dir.join(".lock");
    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let owner = format!("{} {}", std::process::id(), NONCE.fetch_add(1, Ordering::Relaxed));
    let deadline = std::time::Instant::now() + wait;
    loop {
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                let _ = file.write_all(owner.as_bytes()).and_then(|_| file.sync_all());
                return Ok(Lock { path, owner });
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let holder = fs::read_to_string(&path).unwrap_or_default();
                let pid = holder.split_whitespace().next().and_then(|p| p.parse::<u32>().ok());
                let age = fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| SystemTime::now().duration_since(t).ok())
                    .unwrap_or_default();
                let abandoned = match pid {
                    // A holder in this process is alive by definition.
                    Some(pid) if pid == std::process::id() => false,
                    Some(pid) => !process_alive(pid),
                    // Written but not yet named (a holder between create and
                    // write), or unreadable: only a long silence breaks it.
                    None => age > LOCK_STALE,
                };
                if abandoned {
                    let _ = fs::remove_file(&path);
                    continue;
                }
                if std::time::Instant::now() > deadline {
                    return Err(format!("{} is held by {}; is something stuck?", path.display(), holder.trim()));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(error) => return Err(format!("cannot lock {}: {error}", path.display())),
        }
    }
}

impl Book {
    pub fn open(data_dir: &Path) -> Self {
        let rewriter = std::env::var("S0MA_MISTAKE_REWRITER")
            .ok()
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_REWRITER.to_owned());
        Self::with_rewriter(data_dir, rewriter)
    }

    pub fn with_rewriter(data_dir: &Path, rewriter: String) -> Self {
        let data_dir = absolute(data_dir);
        Self {
            dir: data_dir.join("mistakes"),
            data_dir,
            rewriter,
            base_url: crate::llm::endpoint(""),
            model: crate::llm::model_name(),
        }
    }

    pub fn store(&self) -> Store {
        Store::at(&self.data_dir)
    }

    fn record_path(&self, n: u64) -> PathBuf {
        self.dir.join(format!("{n}.json"))
    }

    fn lock(&self) -> Result<Lock> {
        // Long enough for retention, which holds it while deciding what to keep.
        lock_mistakes(&self.data_dir, Duration::from_secs(120))
    }

    pub fn load(&self, n: u64) -> Result<Record> {
        let path = self.record_path(n);
        let bytes = fs::read(&path).map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => format!("no mistake #{n}"),
            _ => format!("cannot read {}: {e}", path.display()),
        })?;
        serde_json::from_slice(&bytes).map_err(|e| format!("{} is not a mistake record: {e}", path.display()))
    }

    fn save(&self, record: &Record) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(record).map_err(|e| e.to_string())?;
        write_atomic(&self.record_path(record.mistake), &bytes)
    }

    fn history(&self, line: HistoryLine<'_>) -> Result<()> {
        let mut bytes = serde_json::to_vec(&line).map_err(|e| e.to_string())?;
        bytes.push(b'\n');
        let path = self.dir.join("history.jsonl");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        file.write_all(&bytes).and_then(|_| file.sync_data()).map_err(|e| format!("cannot append to {}: {e}", path.display()))
    }

    fn change(&self, record: &Record, from: Option<State>, by: &str, detail: String) -> Result<()> {
        self.save(record)?;
        self.history(HistoryLine { mistake: record.mistake, at: now(), from, to: Some(record.state), by, detail })
    }

    /// Every record, by number.
    pub fn all(&self) -> Result<Vec<Record>> {
        let mut records = Vec::new();
        let Ok(entries) = fs::read_dir(&self.dir) else { return Ok(records) };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(n) = name.strip_suffix(".json").and_then(|n| n.parse::<u64>().ok()) else { continue };
            // A number taken by a `log` that is still writing its record.
            if entry.metadata().is_ok_and(|m| m.len() == 0) {
                continue;
            }
            records.push(self.load(n)?);
        }
        records.sort_by_key(|r| r.mistake);
        Ok(records)
    }

    pub fn waiting(&self) -> Result<Vec<Record>> {
        Ok(self.all()?.into_iter().filter(|r| r.state.waiting()).collect())
    }

    fn next_number(&self) -> u64 {
        let Ok(entries) = fs::read_dir(&self.dir) else { return 1 };
        entries
            .flatten()
            .filter_map(|e| e.file_name().to_string_lossy().strip_suffix(".json").and_then(|n| n.parse::<u64>().ok()))
            .max()
            .unwrap_or(0)
            + 1
    }

    fn note_file(&self, n: u64, attempt: u32) -> PathBuf {
        self.dir.join("notes").join(format!("{n}-{attempt}.txt"))
    }

    fn proposal_rel(n: u64, attempt: u32) -> String {
        format!("proposals/{n}-{attempt}.json")
    }

    fn log_paths(&self, n: u64, attempt: u32) -> (PathBuf, PathBuf) {
        let logs = self.dir.join("logs");
        (logs.join(format!("{n}-{attempt}.log")), logs.join(format!("{n}-{attempt}.exit")))
    }

    // -------------------------------------------------------------- logging

    /// Log a mistake on a saved call and start its rewrite. The call must be in
    /// the store: the context is never rebuilt from anything else.
    pub fn log(&self, call: &CallRef, note: &str, weakness: Option<String>) -> Result<(Record, Launch)> {
        self.log_with(call, note, weakness, LogOptions::default())
    }

    /// `log`, with who flagged it and the rest of `LogOptions`.
    pub fn log_with(&self, call: &CallRef, note: &str, weakness: Option<String>, options: LogOptions) -> Result<(Record, Launch)> {
        let note = note.trim();
        if note.is_empty() {
            return Err("a note is required: one sentence saying what was wrong".into());
        }
        // Under the lock retention also takes, so the call cannot be pruned
        // between this check and the record that names it.
        let _lock = self.lock()?;
        if self.store().find(&call.call_id).map_err(|e| e.to_string())?.is_none() {
            return Err(format!("call {} is not in the store", call.call_id));
        }
        let mut n = self.next_number();
        // O_EXCL: two writers can never take the same number.
        loop {
            match OpenOptions::new().write(true).create_new(true).open(self.record_path(n)) {
                Ok(_) => break,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => n += 1,
                Err(error) => return Err(format!("cannot create mistake #{n}: {error}")),
            }
        }
        let created = self.record_path(n);
        let result = self.log_numbered(n, call, note, weakness, options);
        if result.is_err() && fs::metadata(&created).is_ok_and(|m| m.len() == 0) {
            let _ = fs::remove_file(&created);
        }
        result
    }

    fn log_numbered(
        &self,
        n: u64,
        call: &CallRef,
        note: &str,
        weakness: Option<String>,
        options: LogOptions,
    ) -> Result<(Record, Launch)> {
        let LogOptions { by, flagged_in, pointed_at, steps } = options;
        let at = now();
        let mut record = Record {
            mistake: n,
            logged_at: at.clone(),
            call_id: call.call_id.clone(),
            turn_id: call.turn_id.clone(),
            call_in_turn: call.call_in_turn,
            note: note.to_owned(),
            notes: vec![NoteEntry { at, attempt: 1, text: note.to_owned() }],
            split: split_for(&call.turn_id),
            split_key: call.turn_id.clone(),
            weakness,
            state: State::Pending,
            attempts: 1,
            proposal: None,
            current: None,
            final_: None,
            before: None,
            rewrite: None,
            error: None,
            discard_reason: None,
            stretch: steps.unwrap_or_else(|| self.stretch_for(call)),
            target: None,
            target_by: None,
            flagged_by: by.clone(),
            flagged_in,
            pointed_at,
        };
        self.write_note(n, 1, note)?;
        // On disk before the rewriter starts: it reads this record.
        self.change(
            &record,
            None,
            by.as_deref().unwrap_or("operator"),
            format!("logged on {} (call {} of turn {})", call.call_id, call.call_in_turn, call.turn_id),
        )?;
        let launch = self.start_rewrite(&mut record)?;
        Ok((record, launch))
    }

    /// The steps a mistake on `call` may correct: its stretch from the store,
    /// always ending with `call` itself.
    pub fn stretch_for(&self, call: &CallRef) -> Vec<StretchCall> {
        let step = |c: &CallRef| StretchCall { call_id: c.call_id.clone(), call_in_turn: c.call_in_turn, at: c.at.clone() };
        let mut steps: Vec<StretchCall> = self.store().stretch_of(call).unwrap_or_default().iter().map(step).collect();
        if steps.last().map(|last| last.call_id.as_str()) != Some(call.call_id.as_str()) {
            steps.retain(|s| s.call_id != call.call_id);
            steps.push(step(call));
        }
        steps
    }

    /// Launch the current attempt for a record already saved as `pending`, and
    /// record how it went: its process, or `failed` if it could not start.
    fn start_rewrite(&self, record: &mut Record) -> Result<Launch> {
        let launch = self.launch(record);
        match &launch {
            Launch::Started => self.save(record)?,
            Launch::Failed(error) => {
                record.state = State::Failed;
                record.error = Some(error.clone());
                self.change(record, Some(State::Pending), "harness", format!("rewrite {} could not start: {error}", record.attempts))?;
            }
        }
        Ok(launch)
    }

    fn write_note(&self, n: u64, attempt: u32, note: &str) -> Result<()> {
        write_atomic(&self.note_file(n, attempt), note.as_bytes())
    }

    // -------------------------------------------------------------- the rewriter

    /// The command line for one attempt, every value shell-quoted.
    pub fn rewriter_command(&self, n: u64, attempt: u32) -> String {
        self.rewriter
            .replace("{data}", &crate::util::shell_quote(&self.data_dir.to_string_lossy()))
            .replace("{mistake}", &n.to_string())
            .replace("{attempt}", &attempt.to_string())
            .replace("{note_file}", &crate::util::shell_quote(&self.note_file(n, attempt).to_string_lossy()))
    }

    /// Start the rewriter for `record.attempts`, detached: its own session, its
    /// output to a log file, its exit status to a file as it exits. Sets
    /// `record.rewrite` when it starts.
    fn launch(&self, record: &mut Record) -> Launch {
        let n = record.mistake;
        let attempt = record.attempts;
        let (log, exit_file) = self.log_paths(n, attempt);
        let result = (|| -> Result<std::process::Child> {
            fs::create_dir_all(self.dir.join("logs")).map_err(|e| e.to_string())?;
            fs::create_dir_all(self.dir.join("proposals")).map_err(|e| e.to_string())?;
            let _ = fs::remove_file(&exit_file);
            let out = OpenOptions::new().create(true).append(true).open(&log).map_err(|e| e.to_string())?;
            let err = out.try_clone().map_err(|e| e.to_string())?;
            let exit_temp = exit_file.with_extension("exit.tmp");
            let script = format!(
                "{}\nstatus=$?\nprintf '%s\\n' \"$status\" > {} && mv {} {}\nexit $status\n",
                self.rewriter_command(n, attempt),
                crate::util::shell_quote(&exit_temp.to_string_lossy()),
                crate::util::shell_quote(&exit_temp.to_string_lossy()),
                crate::util::shell_quote(&exit_file.to_string_lossy()),
            );
            let mut command = if have_setsid() {
                let mut command = Command::new("setsid");
                command.arg("sh");
                command
            } else {
                let mut command = Command::new("sh");
                std::os::unix::process::CommandExt::process_group(&mut command, 0);
                command
            };
            command
                .arg("-c")
                .arg(script)
                .stdin(Stdio::null())
                .stdout(Stdio::from(out))
                .stderr(Stdio::from(err))
                .env("S0MA_LLM_BASE_URL", &self.base_url)
                .env("S0MA_LLM_MODEL", &self.model);
            // The same server's key, when it takes one.
            if let Some((name, key)) = crate::model_config::key_env() {
                command.env(name, key);
            }
            command.spawn().map_err(|e| format!("cannot start the rewriter: {e}"))
        })();
        match result {
            Ok(mut child) => {
                record.rewrite = Some(Rewrite {
                    attempt,
                    launched_at: now(),
                    pid: Some(child.id()),
                    log: log.to_string_lossy().into_owned(),
                    exit_file: exit_file.to_string_lossy().into_owned(),
                });
                launched().lock().unwrap_or_else(|e| e.into_inner()).push((self.dir.clone(), n, attempt));
                let book = self.clone();
                // Reap it, and apply its result the moment it exits.
                std::thread::spawn(move || {
                    let code = child.wait().ok().and_then(|status| status.code());
                    let applied = book.finish_rewrite(n, attempt, code);
                    launched().lock().unwrap_or_else(|e| e.into_inner()).retain(|(d, m, a)| !(d == &book.dir && *m == n && *a == attempt));
                    if let Some(line) = applied {
                        push_status(line);
                    }
                });
                Launch::Started
            }
            Err(error) => Launch::Failed(error),
        }
    }

    /// Apply a finished attempt's result. Only the current attempt of a
    /// pending mistake is applied: a result that arrives after a regeneration
    /// or an operator decision is recorded as superseded and changes nothing.
    /// Returns the status line to show, if anything changed.
    pub fn finish_rewrite(&self, n: u64, attempt: u32, exit_code: Option<i32>) -> Option<String> {
        let lock = self.lock().ok()?;
        let mut record = self.load(n).ok()?;
        if record.attempts == attempt && record.state != State::Pending {
            // Applied already, by the other process watching it (the agent and a
            // review window of its own both may), or decided on since.
            return None;
        }
        if record.state != State::Pending || record.attempts != attempt {
            let detail = format!(
                "rewrite {attempt} finished after it was superseded (now {} at attempt {}); its result is not applied",
                record.state.label(),
                record.attempts
            );
            let _ = self.history(HistoryLine { mistake: n, at: now(), from: Some(record.state), to: Some(record.state), by: "rewriter", detail });
            return None;
        }
        let rel = Self::proposal_rel(n, attempt);
        let path = self.dir.join(&rel);
        let outcome = match fs::read(&path) {
            Ok(bytes) => read_proposal(&bytes, n, attempt),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let (log, _) = self.log_paths(n, attempt);
                let how = match exit_code {
                    Some(code) => format!("exited with status {code}"),
                    None => "ended without an exit status (killed?)".to_owned(),
                };
                Err(format!("the rewriter {how} and wrote no proposal{}", log_tail(&log)))
            }
            Err(error) => Err(format!("cannot read {rel}: {error}")),
        };
        let mut override_note = None;
        // A proposal may correct any step of the stretch, and no other call.
        let outcome = outcome.and_then(|proposal| match proposal.target.as_deref() {
            Some(target) if record.step_of(target).is_none() => Err(format!(
                "the proposal corrects call {target}, which is not a step of this mistake's stretch"
            )),
            _ => Ok(proposal),
        });
        let from = record.state;
        let detail;
        match outcome {
            Ok(proposal) => {
                record.proposal = Some(rel.clone());
                record.error = proposal.error.clone();
                if proposal.status != "failed" {
                    let proposed = proposal.target.clone().unwrap_or_else(|| record.call_id.clone());
                    let chosen = (record.target_by.as_deref() == Some("operator")).then(|| record.target_id().to_owned());
                    match chosen {
                        // It did as the operator asked: the choice stays his, so a
                        // rewrite run again keeps to it.
                        Some(chosen) if chosen == proposed => {}
                        // A qref that doesn't know steps rewrites the call
                        // flagged whatever was chosen. Shown as it came, and
                        // said so; `t` goes back to the step chosen.
                        Some(chosen) => {
                            let (k, _) = record.step_of(&chosen).unwrap_or((0, 0));
                            override_note = Some(format!("; it corrected another step than the one you chose (step {k}; t goes back to it)"));
                            record.target_by = Some("rewriter".into());
                        }
                        None => record.target_by = Some("rewriter".into()),
                    }
                    record.target = Some(proposed);
                }
                match proposal.status.as_str() {
                    "ready" => {
                        record.state = State::Ready;
                        record.current = Some(rel.clone());
                    }
                    "needs_edit" => {
                        record.state = State::NeedsEdit;
                        // Shown for reference; approving it needs an edit.
                        record.current = proposal.corrected.is_some().then(|| rel.clone());
                    }
                    _ => {
                        record.state = State::Failed;
                        record.current = None;
                    }
                }
                detail = format!("{rel}: {}", proposal.summary.clone().or(proposal.error.clone()).unwrap_or_default());
            }
            Err(error) => {
                record.state = State::Failed;
                record.error = Some(error.clone());
                detail = error;
            }
        }
        let detail = format!("{detail}{}", override_note.as_deref().unwrap_or(""));
        if self.change(&record, Some(from), "rewriter", detail).is_err() {
            return None;
        }
        drop(lock);
        Some(format!("{}{}", self.status_line(&record), override_note.unwrap_or_default()))
    }

    /// The one line shown when a rewrite finishes.
    pub fn status_line(&self, record: &Record) -> String {
        let n = record.mistake;
        let proposal = record.proposal.as_deref().and_then(|rel| self.read_proposal_file(rel).ok());
        let counts = proposal.as_ref().map(proposal_counts).unwrap_or_default();
        match record.state {
            State::Ready => format!("[mistake] #{n}: rewrite ready{counts} — /mistake view {n}"),
            State::NeedsEdit => format!(
                "[mistake] #{n}: the rewrite's edits did not apply cleanly{counts}; edit by hand or regenerate — /mistake view {n}"
            ),
            State::Failed => format!(
                "[mistake] #{n}: rewrite failed: {} — /mistake view {n}",
                one_line(record.error.as_deref().unwrap_or("no reason given"), 160)
            ),
            state => format!("[mistake] #{n}: {}", state.label()),
        }
    }

    pub fn read_proposal_file(&self, rel: &str) -> Result<Proposal> {
        let bytes = fs::read(self.dir.join(rel)).map_err(|e| format!("cannot read {rel}: {e}"))?;
        let (n, attempt) = parse_rel(rel).ok_or_else(|| format!("{rel} is not a proposal name"))?;
        read_proposal(&bytes, n, attempt)
    }

    /// Records left pending by a rewriter this process does not wait on: it
    /// ran before a restart, or was started by a review window on its own.
    pub fn reconcile(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let Ok(records) = self.all() else { return lines };
        for record in records.into_iter().filter(|r| r.state == State::Pending) {
            let attempt = record.attempts;
            let rewrite = record.rewrite.as_ref().filter(|r| r.attempt == attempt);
            let launched_at = rewrite
                .and_then(|r| chrono::DateTime::parse_from_rfc3339(&r.launched_at).ok())
                .map(|t| t.with_timezone(&chrono::Utc));
            let too_old = launched_at.is_none_or(|t| chrono::Utc::now() - t > chrono::Duration::from_std(REWRITE_GIVE_UP).expect("fits"));
            let ours = launched().lock().unwrap_or_else(|e| e.into_inner()).iter().any(|(d, m, a)| d == &self.dir && *m == record.mistake && *a == attempt);
            if ours {
                // Waited on by this process, which applies its result; only a
                // rewrite that hangs is given up on here.
                if too_old
                    && let Some(line) = self.fail_lost(record.mistake, attempt, "rewrite lost: no result 12 hours after it started") {
                        lines.push(line);
                    }
                continue;
            }
            let (_, exit_file) = self.log_paths(record.mistake, attempt);
            if let Ok(text) = fs::read_to_string(&exit_file) {
                if let Some(line) = self.finish_rewrite(record.mistake, attempt, text.trim().parse().ok()) {
                    lines.push(line);
                }
                continue;
            }
            let proposal = self.dir.join(Self::proposal_rel(record.mistake, attempt));
            let gone = rewrite.and_then(|r| r.pid).is_some_and(|pid| !process_alive(pid));
            if gone && proposal.exists() {
                // It wrote its proposal and died before reporting a status.
                if let Some(line) = self.finish_rewrite(record.mistake, attempt, None) {
                    lines.push(line);
                }
            } else if (gone || too_old)
                && let Some(line) = self.fail_lost(record.mistake, attempt, if gone {
                    "rewrite lost: the rewriter's process ended without reporting a result"
                } else {
                    "rewrite lost: no result 12 hours after it started"
                }) {
                    lines.push(line);
                }
        }
        lines
    }

    fn fail_lost(&self, n: u64, attempt: u32, why: &str) -> Option<String> {
        let _lock = self.lock().ok()?;
        let mut record = self.load(n).ok()?;
        if record.state != State::Pending || record.attempts != attempt {
            return None;
        }
        let (log, _) = self.log_paths(n, attempt);
        record.state = State::Failed;
        record.error = Some(format!("{why}{}", log_tail(&log)));
        self.change(&record, Some(State::Pending), "harness", why.to_owned()).ok()?;
        Some(self.status_line(&record))
    }

    // -------------------------------------------------------------- versions

    /// The response of the step the version shown corrects, as saved: the
    /// "original" side.
    pub fn original(&self, record: &Record) -> Result<Response> {
        self.original_of(record.target_id())
    }

    /// A call's response as the store saved it.
    pub fn original_of(&self, call_id: &str) -> Result<Response> {
        let store = self.store();
        let line = store
            .find(call_id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("call {call_id} is not in the store"))?;
        let value = store.response(&line).map_err(|e| e.to_string())?;
        Response::from_value(&value)
    }

    /// The version shown: `current` (a proposal or an edit), else the
    /// original of the step being corrected.
    pub fn shown(&self, record: &Record) -> Result<Version> {
        match record.current.as_deref() {
            Some(rel) if rel.starts_with("edits/") => {
                let (response, target) = read_edit_with_target(&self.dir.join(rel))?;
                let target = target.unwrap_or_else(|| record.call_id.clone());
                Ok(Version { file: rel.to_owned(), response, manual: true, target })
            }
            Some(rel) => {
                let proposal = self.read_proposal_file(rel)?;
                let response = proposal.corrected.ok_or_else(|| format!("{rel} has no corrected response"))?;
                let target = proposal.target.unwrap_or_else(|| record.call_id.clone());
                Ok(Version { file: rel.to_owned(), response, manual: false, target })
            }
            None => Ok(Version {
                file: "original".into(),
                response: self.original(record)?,
                manual: false,
                target: record.target_id().to_owned(),
            }),
        }
    }

    /// A proposal's or edit's `corrected` object, as the file holds it.
    fn stored_corrected(&self, rel: &str) -> Result<Value> {
        let path = self.dir.join(rel);
        let value: Value = serde_json::from_slice(&fs::read(&path).map_err(|e| format!("cannot read {rel}: {e}"))?)
            .map_err(|e| format!("{rel} is not JSON: {e}"))?;
        value.get("corrected").filter(|c| c.is_object()).cloned().ok_or_else(|| format!("{rel} has no corrected response"))
    }

    /// Why the version shown cannot be approved, if it cannot.
    pub fn approvable(&self, record: &Record, shown: &Version, original: &Response) -> Result<()> {
        if !record.state.waiting() {
            return Err(match record.state {
                State::Pending => "the rewrite is still running".into(),
                state => format!("it is {} already; reopen it first", state.label()),
            });
        }
        if shown.file == "original" {
            return Err("nothing to approve yet: edit it by hand (e) or regenerate (r)".into());
        }
        if !shown.manual && record.state != State::Ready {
            return Err("the rewriter's version did not apply cleanly; edit it by hand (e) first".into());
        }
        if same_response(&shown.response, original) {
            return Err("nothing changed: this version is identical to the original".into());
        }
        shown.response.check_tool_calls()
    }

    // -------------------------------------------------------------- the operator's actions

    /// Save a manual version. It becomes the version shown; `approve` then
    /// approves it as `edited`.
    ///
    /// `based_on` and `target` are the version the edit was made from and the
    /// step it corrects, as the window showed them. If either changed since
    /// (a rewrite landed, another window moved on, the editor stayed open a
    /// long time), nothing is saved: the edit would be filed against a version
    /// or step it wasn't written for.
    pub fn save_edit(&self, n: u64, response: &Response, based_on: &str, target: &str) -> Result<String> {
        response.check_tool_calls()?;
        let _lock = self.lock()?;
        let record = self.load(n)?;
        if !record.state.waiting() {
            return Err(match record.state {
                State::Pending => "a rewrite started since; nothing saved".into(),
                state => format!("it is {} now; nothing saved (reopen it to edit)", state.label()),
            });
        }
        let current = record.current.as_deref().unwrap_or("original");
        if based_on != current || target != record.target_id() {
            return Err(format!("the version changed since it was shown ({based_on} → {current}); nothing saved, look again"));
        }
        let edits = self.dir.join("edits");
        fs::create_dir_all(&edits).map_err(|e| e.to_string())?;
        let mut k = 1;
        while edits.join(format!("{n}-{k}.json")).exists() {
            k += 1;
        }
        let rel = format!("edits/{n}-{k}.json");
        let file = serde_json::json!({
            "mistake": n, "k": k, "based_on": based_on, "at": now(), "target_call_id": target, "corrected": response,
        });
        write_atomic(&self.dir.join(&rel), &serde_json::to_vec_pretty(&file).map_err(|e| e.to_string())?)?;
        let mut record = record;
        record.current = Some(rel.clone());
        record.target = Some(target.to_owned());
        self.change(&record, Some(record.state), "operator", format!("manual version {rel}, from {based_on}"))?;
        Ok(rel)
    }

    /// Approve the version shown. `displayed` is the file the operator was
    /// looking at: if another version became current since (a rewrite landed,
    /// another window saved an edit), nothing is approved.
    pub fn approve(&self, n: u64, displayed: Option<&str>) -> Result<Record> {
        let _lock = self.lock()?;
        let mut record = self.load(n)?;
        let shown = self.shown(&record)?;
        if let Some(displayed) = displayed.filter(|d| *d != shown.file) {
            return Err(format!("the version changed since it was shown ({displayed} → {}); look again", shown.file));
        }
        if record.step_of(&shown.target).is_none() {
            return Err(format!("{} corrects call {}, which is not a step of #{n}'s stretch", shown.file, shown.target));
        }
        let original = self.original_of(&shown.target)?;
        self.approvable(&record, &shown, &original)?;
        // The hash of `corrected` exactly as the file holds it, which is what
        // qref checks, not of this program's reading of it.
        let stored = self.stored_corrected(&shown.file)?;
        let from = record.state;
        record.before = Some(from);
        record.state = if shown.manual { State::Edited } else { State::Approved };
        record.final_ = Some(Final {
            source: if shown.manual { "manual" } else { "proposal" }.into(),
            target_call_id: Some(shown.target.clone()),
            file: shown.file.clone(),
            corrected_sha256: calls::sha256_hex(canonical_json(&stored).as_bytes()),
            approved_at: now(),
        });
        let step = record.step_of(&shown.target).map(|(k, of)| format!(" (step {k} of {of})")).unwrap_or_default();
        self.change(&record, Some(from), "operator", format!("approved {}{step}", shown.file))?;
        Ok(record)
    }

    /// Correct another step of the stretch. The version shown becomes the
    /// latest one made for that step (an edit, else the proposal), or its
    /// original when there is none: nothing to approve until it is edited.
    pub fn retarget(&self, n: u64, call_id: &str) -> Result<Record> {
        let _lock = self.lock()?;
        let mut record = self.load(n)?;
        let Some((k, of)) = record.step_of(call_id) else {
            return Err(format!("call {call_id} is not a step of #{n}'s stretch"));
        };
        if !record.state.waiting() {
            return Err(match record.state {
                State::Pending => "the rewrite is still running; wait for it".into(),
                state => format!("it is {} already; reopen it first", state.label()),
            });
        }
        if record.target_id() == call_id {
            return Ok(record);
        }
        record.current = self.latest_version_for(&record, call_id);
        record.target = Some(call_id.to_owned());
        record.target_by = Some("operator".into());
        self.change(&record, Some(record.state), "operator", format!("now correcting step {k} of {of} (call {call_id})"))?;
        Ok(record)
    }

    /// The newest version made for one step, by when its file was written:
    /// one of its edits, or the latest proposal if that corrects this step and
    /// has a response.
    fn latest_version_for(&self, record: &Record, call_id: &str) -> Option<String> {
        let n = record.mistake;
        let written = |rel: &str| fs::metadata(self.dir.join(rel)).and_then(|m| m.modified()).ok();
        let mut candidates: Vec<(Option<SystemTime>, usize, String)> = Vec::new();
        let mut k = 1;
        while self.dir.join(format!("edits/{n}-{k}.json")).exists() {
            let rel = format!("edits/{n}-{k}.json");
            if let Ok((_, target)) = read_edit_with_target(&self.dir.join(&rel))
                && target.as_deref().unwrap_or(&record.call_id) == call_id
            {
                candidates.push((written(&rel), k, rel));
            }
            k += 1;
        }
        if let Some(rel) = record.proposal.clone()
            && let Ok(proposal) = self.read_proposal_file(&rel)
            && proposal.corrected.is_some()
            && proposal.target.as_deref().unwrap_or(&record.call_id) == call_id
        {
            candidates.push((written(&rel), k, rel));
        }
        // Newest file; on equal times, the later one made.
        candidates.into_iter().max_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1))).map(|(_, _, rel)| rel)
    }

    pub fn discard(&self, n: u64, reason: Option<&str>) -> Result<Record> {
        let _lock = self.lock()?;
        let mut record = self.load(n)?;
        if !record.state.waiting() {
            return Err(match record.state {
                // Its result would land on a discarded record and be lost.
                State::Pending => format!("#{n}'s rewrite is still running; wait for it, then discard"),
                state => format!("#{n} is {} already; reopen it first", state.label()),
            });
        }
        let from = record.state;
        record.before = Some(from);
        record.state = State::Discarded;
        record.discard_reason = reason.map(str::trim).filter(|r| !r.is_empty()).map(str::to_owned);
        let detail = match &record.discard_reason {
            Some(reason) => format!("discarded: {reason}"),
            None => "discarded".to_owned(),
        };
        self.change(&record, Some(from), "operator", detail)?;
        Ok(record)
    }

    /// Back to review: out of training until approved again. The approval it
    /// had is kept in history, not in the record.
    pub fn reopen(&self, n: u64) -> Result<Record> {
        let _lock = self.lock()?;
        let mut record = self.load(n)?;
        if !record.state.reviewed() {
            return Err(format!("#{n} is {}, not approved, edited or discarded", record.state.label()));
        }
        let from = record.state;
        let cleared = record.final_.take();
        record.state = record.before.take().filter(|s| s.waiting()).unwrap_or(State::Ready);
        record.discard_reason = None;
        let detail = match cleared {
            Some(f) => format!("reopened; approval of {} (corrected_sha256 {}) withdrawn", f.file, f.corrected_sha256),
            None => "reopened".to_owned(),
        };
        self.change(&record, Some(from), "operator", detail)?;
        Ok(record)
    }

    /// Run the rewriter again, with an amended note if one is given.
    pub fn regenerate(&self, n: u64, note: Option<&str>) -> Result<(Record, Launch)> {
        self.regenerate_as(n, note, "operator")
    }

    /// `regenerate`, saying who asked (`s0ma` when its added reason does).
    fn regenerate_as(&self, n: u64, note: Option<&str>, by: &str) -> Result<(Record, Launch)> {
        let _lock = self.lock()?;
        let mut record = self.load(n)?;
        let from = record.state;
        let cleared = record.final_.take();
        record.before = None;
        record.discard_reason = None;
        record.error = None;
        record.attempts += 1;
        record.state = State::Pending;
        let attempt = record.attempts;
        if let Some(note) = note.map(str::trim).filter(|n| !n.is_empty() && *n != record.note) {
            record.note = note.to_owned();
            record.notes.push(NoteEntry { at: now(), attempt, text: note.to_owned() });
        }
        self.write_note(n, attempt, &record.note.clone())?;
        let mut detail = format!("regenerate: attempt {attempt}");
        if let Some(f) = cleared {
            detail.push_str(&format!("; approval of {} withdrawn", f.file));
        }
        self.change(&record, Some(from), by, detail)?;
        let launch = self.start_rewrite(&mut record)?;
        Ok((record, launch))
    }

    /// Replace the note with `note` (the old one and an addition), without
    /// running the rewriter: the next attempt reads it.
    fn amend_note(&self, n: u64, note: &str, by: &str) -> Result<Record> {
        let _lock = self.lock()?;
        let mut record = self.load(n)?;
        record.note = note.to_owned();
        record.notes.push(NoteEntry { at: now(), attempt: record.attempts, text: note.to_owned() });
        self.change(&record, Some(record.state), by, "note amended".into())?;
        Ok(record)
    }
}

pub enum Launch {
    Started,
    Failed(String),
}

fn parse_rel(rel: &str) -> Option<(u64, u32)> {
    let name = rel.rsplit('/').next()?.strip_suffix(".json")?;
    let (n, k) = name.split_once('-')?;
    Some((n.parse().ok()?, k.parse().ok()?))
}

fn read_proposal(bytes: &[u8], n: u64, attempt: u32) -> Result<Proposal> {
    let value: Value = serde_json::from_slice(bytes).map_err(|e| format!("the proposal is not JSON: {e}"))?;
    if value.get("mistake").and_then(Value::as_u64).is_some_and(|m| m != n)
        || value.get("attempt").and_then(Value::as_u64).is_some_and(|a| a != u64::from(attempt))
    {
        return Err(format!("the proposal names another mistake or attempt than #{n} attempt {attempt}"));
    }
    let status = value.get("status").and_then(Value::as_str).unwrap_or("").to_owned();
    if !matches!(status.as_str(), "ready" | "needs_edit" | "failed") {
        return Err(format!("the proposal's status {status:?} is not one of ready, needs_edit, failed"));
    }
    let corrected = match value.get("corrected").filter(|c| !c.is_null()) {
        Some(c) => Some(Response::from_value(c)?),
        None => None,
    };
    let mut proposal = Proposal {
        status,
        error: value.get("error").and_then(Value::as_str).map(str::to_owned),
        target: value.get("target_call_id").and_then(Value::as_str).map(str::to_owned),
        corrected,
        summary: value.get("summary").and_then(Value::as_str).map(str::to_owned),
        edits: value.get("edits").and_then(Value::as_array).map(Vec::len),
        changed_tokens: value.get("changed_tokens").and_then(Value::as_u64),
        total_tokens: value.get("total_tokens").and_then(Value::as_u64),
    };
    if proposal.status == "ready" {
        match &proposal.corrected {
            None => {
                proposal.status = "failed".into();
                proposal.error = Some("the proposal says ready but has no corrected response".into());
            }
            Some(corrected) => {
                if let Err(error) = corrected.check_tool_calls() {
                    proposal.status = "needs_edit".into();
                    proposal.error = Some(error);
                }
            }
        }
    }
    Ok(proposal)
}

fn proposal_counts(proposal: &Proposal) -> String {
    match (proposal.edits, proposal.changed_tokens, proposal.total_tokens) {
        (Some(edits), Some(changed), Some(total)) => {
            format!(" ({edits} span{}, {changed} of {total} tokens)", if edits == 1 { "" } else { "s" })
        }
        (Some(edits), _, _) => format!(" ({edits} span{})", if edits == 1 { "" } else { "s" }),
        _ => String::new(),
    }
}

/// An edit file's response, and the step it corrects (None on edits made
/// before stretches: the flagged call).
fn read_edit_with_target(path: &Path) -> Result<(Response, Option<String>)> {
    let value: Value = serde_json::from_slice(&fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?)
        .map_err(|e| format!("{} is not JSON: {e}", path.display()))?;
    let response =
        Response::from_value(value.get("corrected").ok_or_else(|| format!("{} has no corrected response", path.display()))?)?;
    Ok((response, value.get("target_call_id").and_then(Value::as_str).map(str::to_owned)))
}

fn log_tail(log: &Path) -> String {
    let Ok(text) = fs::read_to_string(log) else { return String::new() };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return String::new();
    }
    let tail = &lines[lines.len().saturating_sub(15)..];
    format!("; its output ended:\n{}", tail.join("\n"))
}

pub fn one_line(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        format!("{}…", flat.chars().take(max.saturating_sub(1)).collect::<String>())
    }
}

fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map(|dir| dir.join(path)).unwrap_or_else(|_| path.to_path_buf())
    }
}

fn have_setsid() -> bool {
    static HAVE: OnceLock<bool> = OnceLock::new();
    *HAVE.get_or_init(|| {
        std::env::var_os("PATH")
            .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join("setsid").is_file()))
    })
}

fn process_alive(pid: u32) -> bool {
    if Path::new("/proc/self").exists() {
        return Path::new(&format!("/proc/{pid}")).exists();
    }
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().ok_or_else(|| format!("{} has no folder", path.display()))?;
    fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    let name = path.file_name().and_then(|n| n.to_str()).ok_or("not a file path")?;
    let temp = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    (|| -> io::Result<()> {
        let mut file = File::create(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        if let Ok(dir) = File::open(parent) {
            dir.sync_all()?;
        }
        Ok(())
    })()
    .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn launched() -> &'static Mutex<Vec<(PathBuf, u64, u32)>> {
    static LAUNCHED: OnceLock<Mutex<Vec<(PathBuf, u64, u32)>>> = OnceLock::new();
    LAUNCHED.get_or_init(|| Mutex::new(Vec::new()))
}

// ------------------------------------------------------------------ status lines

fn statuses() -> &'static Mutex<VecDeque<String>> {
    static STATUSES: OnceLock<Mutex<VecDeque<String>>> = OnceLock::new();
    STATUSES.get_or_init(|| Mutex::new(VecDeque::new()))
}

static WAKE: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Queue a line for the operator, and wake the idle prompt to show it.
pub fn push_status(line: String) {
    {
        let mut queue = statuses().lock().unwrap_or_else(|e| e.into_inner());
        if queue.len() == 32 {
            queue.pop_front();
        }
        queue.push_back(line);
    }
    if let Some(wake) = WAKE.get() {
        wake.store(true, Ordering::Relaxed);
    }
}

/// Lines waiting to be shown (rewrites finished, deferred flags logged).
pub fn take_statuses() -> Vec<String> {
    statuses().lock().unwrap_or_else(|e| e.into_inner()).drain(..).collect()
}

// ------------------------------------------------------------------ the running agent's book

static BOOK: OnceLock<Book> = OnceLock::new();

/// Where `/mistake @k` points: the calls the last `/mistake calls` printed,
/// newest first. Resolved against that list, never against the live store,
/// which may have moved on since.
fn listing() -> &'static Mutex<Vec<CallRef>> {
    static LISTING: OnceLock<Mutex<Vec<CallRef>>> = OnceLock::new();
    LISTING.get_or_init(|| Mutex::new(Vec::new()))
}

/// Start `/mistake` for this agent: the call store, the watcher for rewrites
/// it is not waiting on, and the start-up notice.
pub fn init(data_dir: &Path, wake: Arc<AtomicBool>) -> Option<String> {
    let _ = WAKE.set(wake);
    let book = BOOK.get_or_init(|| Book::open(data_dir));
    if let Err(error) = calls::init(data_dir, Box::new(deferred_flag)) {
        crate::terminal::warn_status(&format!("[mistake] the call store could not start, so no calls are saved: {error}"));
    }
    for line in book.reconcile() {
        push_status(line);
    }
    let watcher = book.clone();
    let _ = std::thread::Builder::new().name("s0ma-mistakes".into()).spawn(move || loop {
        std::thread::sleep(WATCH_INTERVAL);
        for line in watcher.reconcile() {
            push_status(line);
        }
    });
    let waiting = book.waiting().map(|w| w.len()).unwrap_or(0);
    (waiting > 0).then(|| {
        format!(
            "[mistake] {waiting} mistake{} await{} review — /mistake list",
            if waiting == 1 { "" } else { "s" },
            if waiting == 1 { "s" } else { "" }
        )
    })
}

/// An attached terminal's `/mistake view`: the book only, for the review
/// window (the calls and the watcher are the running agent's).
pub fn open_for_view(data_dir: &Path) {
    let _ = BOOK.get_or_init(|| Book::open(data_dir));
}

pub fn book() -> Option<&'static Book> {
    BOOK.get()
}

/// A flag that waited for the call in flight, now that the call ended.
fn deferred_flag(event: FlagEvent) {
    let Some(book) = BOOK.get() else { return };
    match event {
        FlagEvent::Logged(call, flag) => match book.log(&call, &flag.note, flag.weakness) {
            Ok((record, launch)) => push_status(logged_line(book, &record, &launch)),
            Err(error) => push_status(format!("[mistake] not logged: {error} (note: {})", one_line(&flag.note, 80))),
        },
        FlagEvent::Dropped(flag, why) => {
            push_status(format!("[mistake] not logged: {why}. Note: {}", one_line(&flag.note, 120)))
        }
    }
}

fn local_time(at: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(at)
        .map(|t| t.with_timezone(&chrono::Local).format("%H:%M:%S").to_string())
        .unwrap_or_else(|_| at.to_owned())
}

/// What a saved response amounts to, in a line.
pub fn gist(response: &Value) -> String {
    let content = response.get("content").and_then(Value::as_str).unwrap_or("");
    if let Some(first) = content.lines().find(|l| !l.trim().is_empty()) {
        return format!("“{}”", one_line(first, 70));
    }
    let tools: Vec<&str> = response
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().filter_map(|c| c.pointer("/function/name").and_then(Value::as_str)).collect())
        .unwrap_or_default();
    if !tools.is_empty() {
        return format!("tools: {}", tools.join(", "));
    }
    if response.get("reasoning_content").and_then(Value::as_str).is_some_and(|r| !r.trim().is_empty()) {
        return "(reasoning only)".into();
    }
    "(empty)".into()
}

fn logged_line(book: &Book, record: &Record, launch: &Launch) -> String {
    let store = book.store();
    let what = store
        .find(&record.call_id)
        .ok()
        .flatten()
        .and_then(|line| store.response(&line).ok())
        .map(|response| format!(" · {}", gist(&response)))
        .unwrap_or_default();
    let held_out = if record.split == Split::Heldout { "yes" } else { "no" };
    let tail = match launch {
        Launch::Started => "rewriting".to_owned(),
        Launch::Failed(error) => format!("the rewriter could not start: {error}"),
    };
    let steps = match record.steps().len() {
        1 => String::new(),
        n => format!("; covers the turn's last {n} calls, so you can correct the one where it went wrong (t in the view)"),
    };
    format!(
        "[mistake] #{} logged (call {} of turn {}{what}{steps}; held out: {held_out}) — {tail}",
        record.mistake, record.call_in_turn, record.turn_id
    )
}

// ------------------------------------------------------------------ the command

/// Where a `/mistake` was typed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// At the idle prompt: the review window can open here.
    Idle,
    /// On the keyboard thread while a turn runs: never delays the turn.
    Turn,
    /// The local keyboard's steering pane while a turn runs: the review
    /// window opens over the paused transcript, as Ctrl+G's editor does, and
    /// the agent keeps working meanwhile (input.rs `open_mistake_view`).
    Steering,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Reply {
    Text(String),
    Error(String),
    /// Open the review window here, on this mistake or the first waiting one.
    View(Option<u64>),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Parsed {
    Calls,
    List(Filter),
    View(Option<u64>),
    Reopen(u64),
    Flag { call: Option<usize>, weakness: Option<String>, note: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    /// Not decided yet: being rewritten, or waiting for review. What a bare
    /// `/mistake list` shows, so a mistake doesn't vanish from it while its
    /// rewrite runs.
    Open,
    Waiting,
    Pending,
    Ready,
    NeedsEdit,
    Failed,
    Approved,
    Discarded,
    All,
}

impl Filter {
    fn parse(word: &str) -> Option<Self> {
        Some(match word {
            "open" => Filter::Open,
            "waiting" => Filter::Waiting,
            "pending" => Filter::Pending,
            "ready" => Filter::Ready,
            "needs_edit" => Filter::NeedsEdit,
            "failed" => Filter::Failed,
            "approved" => Filter::Approved,
            "discarded" => Filter::Discarded,
            "all" => Filter::All,
            _ => return None,
        })
    }

    fn keeps(self, state: State) -> bool {
        match self {
            Filter::Open => state == State::Pending || state.waiting(),
            Filter::Waiting => state.waiting(),
            Filter::Pending => state == State::Pending,
            Filter::Ready => state == State::Ready,
            Filter::NeedsEdit => state == State::NeedsEdit,
            Filter::Failed => state == State::Failed,
            Filter::Approved => matches!(state, State::Approved | State::Edited),
            Filter::Discarded => state == State::Discarded,
            Filter::All => true,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Filter::Open => "open (rewriting or waiting for review)",
            Filter::Waiting => "waiting for review",
            Filter::Pending => "pending",
            Filter::Ready => "ready",
            Filter::NeedsEdit => "needs_edit",
            Filter::Failed => "failed",
            Filter::Approved => "approved or edited",
            Filter::Discarded => "discarded",
            Filter::All => "all",
        }
    }
}

pub const USAGE: &str = "usage: /mistake [@k] [--weakness wk-NNNN] <what was wrong> · /mistake calls · \
    /mistake list [open|waiting|pending|ready|needs_edit|failed|approved|discarded|all] · /mistake view [n] · \
    /mistake reopen <n>";

/// Parse a `/mistake` line. A subcommand is recognised only as exactly that
/// grammar; anything else is a note (`/mistake list the files before deleting`
/// flags a mistake).
pub fn parse(input: &str) -> Option<std::result::Result<Parsed, String>> {
    let trimmed = input.trim();
    let rest = trimmed.strip_prefix("/mistake")?;
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let words: Vec<&str> = rest.split_whitespace().collect();
    let number = |w: &str| w.trim_start_matches('#').parse::<u64>().ok();
    match words.as_slice() {
        [] => return Some(Err(USAGE.into())),
        ["calls"] => return Some(Ok(Parsed::Calls)),
        ["list"] => return Some(Ok(Parsed::List(Filter::Open))),
        ["list", word] if Filter::parse(word).is_some() => {
            return Some(Ok(Parsed::List(Filter::parse(word).expect("checked"))))
        }
        ["view"] => return Some(Ok(Parsed::View(None))),
        ["view", n] if number(n).is_some() => return Some(Ok(Parsed::View(number(n)))),
        ["reopen", n] if number(n).is_some() => return Some(Ok(Parsed::Reopen(number(n).expect("checked")))),
        _ => {}
    }
    // A flag: [@k] [--weakness wk-NNNN] note. The note keeps its own spacing.
    let mut remaining = rest.trim_start();
    let mut call = None;
    let mut weakness = None;
    loop {
        let (word, after) = match remaining.find(char::is_whitespace) {
            Some(i) => (&remaining[..i], remaining[i..].trim_start()),
            None => (remaining, ""),
        };
        if call.is_none() && weakness.is_none() && word.starts_with('@') {
            match word[1..].parse::<usize>() {
                Ok(k) if k >= 1 => call = Some(k),
                _ => return Some(Err(format!("{word}: @k takes a number from /mistake calls"))),
            }
            remaining = after;
        } else if word == "--weakness" || word.starts_with("--weakness=") {
            let (id, after) = match word.strip_prefix("--weakness=") {
                Some(id) => (id, after),
                None => match after.find(char::is_whitespace) {
                    Some(i) => (&after[..i], after[i..].trim_start()),
                    None => (after, ""),
                },
            };
            let valid = id.strip_prefix("wk-").is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()));
            if !valid {
                return Some(Err(format!("--weakness takes a weakness id like wk-0027, not {id:?}")));
            }
            weakness = Some(id.to_owned());
            remaining = after;
        } else {
            break;
        }
    }
    let note = remaining.trim();
    if note.is_empty() {
        return Some(Err(format!("a note is required: one sentence saying what was wrong. {USAGE}")));
    }
    // A logged mistake is permanent (it can be discarded, not deleted), so a
    // mistyped subcommand must not become one: a note is a sentence, and one
    // that opens with a subcommand's word has to read like a sentence.
    let words: Vec<&str> = note.split_whitespace().collect();
    let first = words[0].to_ascii_lowercase();
    let keyword = SUBCOMMANDS.iter().find(|k| first.starts_with(*k) && first[k.len()..].bytes().all(|b| b.is_ascii_digit()));
    if let Some(keyword) = keyword.filter(|_| words.len() < 4) {
        return Some(Err(format!(
            "{note:?} looks like a mistyped /mistake {keyword}, so nothing was logged. {USAGE}"
        )));
    }
    if words.len() < 3 {
        return Some(Err(format!(
            "{note:?} was not logged: the note says what was wrong, in a sentence of at least three words"
        )));
    }
    Some(Ok(Parsed::Flag { call, weakness, note: note.to_owned() }))
}

const SUBCOMMANDS: [&str; 4] = ["calls", "list", "view", "reopen"];

/// Handle a `/mistake` line, or `None` if it is not one. Safe on any thread:
/// it never touches the conversation, and the slow parts (the rewrite) run
/// detached.
pub fn handle_command(input: &str, place: Place) -> Option<Reply> {
    let command = match parse(input)? {
        Ok(command) => command,
        Err(error) => return Some(Reply::Error(format!("[mistake] {error}"))),
    };
    let Some(book) = BOOK.get() else {
        return Some(Reply::Error("[mistake] not available: the mistake book is not open in this process".into()));
    };
    Some(dispatch(book, calls::recorder(), command, place))
}

/// What a command does now, and what it leaves for later.
enum Step {
    Done(Reply),
    /// Log a mistake on this call: waits for the writer and the lock.
    Flag { target: CallRef, note: String, weakness: Option<String> },
    Reopen(u64),
}

/// Answer at once; anything that can wait on the call store's writer or the
/// mistakes lock runs on a thread of its own and reports through the status
/// line. Called on the keyboard thread mid-turn, where a wait would hold up
/// every key, Ctrl+C included, and the end of the agent's pass.
fn dispatch(book: &'static Book, recorder: Option<&'static calls::Recorder>, command: Parsed, place: Place) -> Reply {
    match plan(book, recorder, command, place) {
        Step::Done(reply) => reply,
        Step::Flag { target, note, weakness } => {
            let reply = format!(
                "[mistake] flagging call {} of turn {} ({}) — logging",
                target.call_in_turn,
                target.turn_id,
                local_time(&target.at)
            );
            std::thread::spawn(move || {
                let line = match complete_flag(book, recorder, &target, &note, weakness) {
                    Reply::Text(line) | Reply::Error(line) => line,
                    Reply::View(_) => return,
                };
                push_status(line);
            });
            Reply::Text(reply)
        }
        Step::Reopen(n) if place == Place::Idle => reopen_reply(book, n),
        Step::Reopen(n) => {
            std::thread::spawn(move || {
                if let Reply::Text(line) | Reply::Error(line) = reopen_reply(book, n) {
                    push_status(line);
                }
            });
            Reply::Text(format!("[mistake] reopening #{n}"))
        }
    }
}

fn plan(book: &Book, recorder: Option<&calls::Recorder>, command: Parsed, place: Place) -> Step {
    match command {
        Parsed::Calls => Step::Done(list_calls(book)),
        Parsed::List(filter) => Step::Done(list_mistakes(book, filter)),
        Parsed::View(n) => Step::Done(match place {
            Place::Idle | Place::Steering => Reply::View(n),
            Place::Turn => {
                let which = n.map(|n| format!(" {n}")).unwrap_or_default();
                Reply::Text(format!(
                    "[mistake] the review window opens at the idle prompt (/mistake view{which} once this turn ends), \
                     or now in a terminal of its own: s0ma --mistake-view{which}"
                ))
            }
        }),
        Parsed::Reopen(n) => Step::Reopen(n),
        Parsed::Flag { call, weakness, note } => match resolve_flag(book, recorder, call, &weakness, &note) {
            Ok(target) => Step::Flag { target, note, weakness },
            Err(reply) => Step::Done(reply),
        },
    }
}

fn reopen_reply(book: &Book, n: u64) -> Reply {
    match book.reopen(n) {
        Ok(record) => Reply::Text(format!(
            "[mistake] #{n} reopened: {}; out of training until it is approved again",
            record.state.label()
        )),
        Err(error) => Reply::Error(format!("[mistake] {error}")),
    }
}

/// Which call a flag means, decided now, from memory: the listed call for
/// `@k`, else the newest saved call. A flag typed while a call streams is
/// handed to the call store to log when that call is saved.
fn resolve_flag(
    book: &Book,
    recorder: Option<&calls::Recorder>,
    call: Option<usize>,
    weakness: &Option<String>,
    note: &str,
) -> std::result::Result<CallRef, Reply> {
    if let Some(id) = weakness
        && let Err(error) = check_weakness(&book.data_dir, id) {
            return Err(Reply::Error(format!("[mistake] {error}")));
        }
    match call {
        Some(k) => {
            let listed = listing().lock().unwrap_or_else(|e| e.into_inner()).clone();
            if listed.is_empty() {
                return Err(Reply::Error("[mistake] @k refers to the list /mistake calls prints; run it first".into()));
            }
            listed.get(k - 1).cloned().ok_or_else(|| {
                Reply::Error(format!(
                    "[mistake] /mistake calls listed {} call{}; there is no @{k}",
                    listed.len(),
                    if listed.len() == 1 { "" } else { "s" }
                ))
            })
        }
        None => {
            let Some(recorder) = recorder else {
                return Err(Reply::Error("[mistake] the call store is not running, so there is no call to flag".into()));
            };
            match recorder.latest_or_defer(PendingFlag { note: note.to_owned(), weakness: weakness.clone() }) {
                (Latest::Deferred { turn_id, call_in_turn }, _) => Err(Reply::Text(format!(
                    "[mistake] flagging the call in progress (call {call_in_turn} of turn {turn_id}); \
                     it is logged when the call ends"
                ))),
                (Latest::Call(call), _) => Ok(call),
                (Latest::None, _) => Err(Reply::Error("[mistake] no model call has been saved yet".into())),
            }
        }
    }
}

/// Log the flag: once the call is on disk, under the lock.
fn complete_flag(
    book: &Book,
    recorder: Option<&calls::Recorder>,
    target: &CallRef,
    note: &str,
    weakness: Option<String>,
) -> Reply {
    if let Some(recorder) = recorder {
        // The call may still be on its way to disk.
        recorder.sync(Duration::from_secs(30));
        if let Some(error) = recorder.save_error(&target.call_id) {
            return Reply::Error(format!("[mistake] call {} was not saved ({error}), so it cannot be flagged", target.call_id));
        }
    }
    match book.log(target, note, weakness) {
        Ok((record, launch)) => Reply::Text(logged_line(book, &record, &launch)),
        Err(error) => Reply::Error(format!("[mistake] not logged: {error}. Note: {}", one_line(note, 120))),
    }
}

fn check_weakness(data_dir: &Path, id: &str) -> Result<()> {
    let export = data_dir.join("weaknesses-export.jsonl");
    let Ok(text) = fs::read_to_string(&export) else { return Ok(()) };
    let found = text.lines().any(|line| {
        serde_json::from_str::<Value>(line).ok().and_then(|v| v.get("id").and_then(Value::as_str).map(|s| s == id)).unwrap_or(false)
    });
    if found {
        Ok(())
    } else {
        Err(format!("no weakness {id} in the log"))
    }
}

fn list_calls(book: &Book) -> Reply {
    let store = book.store();
    let recent = match store.recent(20) {
        Ok(recent) => recent,
        Err(error) => return Reply::Error(format!("[mistake] cannot read the call store: {error}")),
    };
    if recent.is_empty() {
        return Reply::Text("[mistake] no calls saved yet".into());
    }
    // Numbered by position in the list `@k` resolves against.
    let entries: Vec<(CallRef, &Value)> = recent.iter().filter_map(|line| Some((calls::call_ref(line)?, line))).collect();
    let refs: Vec<CallRef> = entries.iter().map(|(call, _)| call.clone()).collect();
    let mut lines = vec![format!(
        "[mistake] the last {} call{} (@1 is the newest; /mistake @k <note> flags one):",
        refs.len(),
        if refs.len() == 1 { "" } else { "s" }
    )];
    for (index, (call, line)) in entries.iter().enumerate().rev() {
        let what = store.response(line).map(|r| gist(&r)).unwrap_or_else(|e| format!("(unreadable: {e})"));
        let finish = line.get("finish").and_then(Value::as_str).filter(|f| *f == "interrupted" || *f == "length");
        let finish = finish.map(|f| format!(" [{f}]")).unwrap_or_default();
        lines.push(format!(
            "  @{:<2} {}  call {} of {}  {what}{finish}",
            index + 1,
            local_time(&call.at),
            call.call_in_turn,
            call.turn_id
        ));
    }
    *listing().lock().unwrap_or_else(|e| e.into_inner()) = refs;
    Reply::Text(lines.join("\n"))
}

fn list_mistakes(book: &Book, filter: Filter) -> Reply {
    let records = match book.all() {
        Ok(records) => records,
        Err(error) => return Reply::Error(format!("[mistake] {error}")),
    };
    let kept: Vec<&Record> = records.iter().filter(|r| filter.keeps(r.state)).collect();
    if kept.is_empty() {
        return Reply::Text(format!("[mistake] none {}", filter.label()));
    }
    let mut lines = vec![format!("[mistake] {} ({}):", filter.label(), kept.len())];
    let now = crate::clock::now();
    for record in kept {
        let mut spans = match record.current.as_deref() {
            // A rewrite running: which attempt, and for how long.
            _ if record.state == State::Pending => rewriting(record, &now),
            Some(rel) if rel.starts_with("edits/") => "edited by hand".to_owned(),
            Some(rel) => book.read_proposal_file(rel).map(|p| proposal_counts(&p).trim().to_owned()).unwrap_or_default(),
            None => String::new(),
        };
        if let Some((k, of)) = record.step_of(record.target_id()).filter(|(_, of)| *of > 1) {
            spans = format!("step {k} of {of}{}{spans}", if spans.is_empty() { "" } else { " · " });
        }
        lines.push(format!(
            "  #{:<4} {}  {:<10} {:<7} {}  {}",
            record.mistake,
            chrono::DateTime::parse_from_rfc3339(&record.logged_at)
                .map(|t| t.format("%m-%d %H:%M").to_string())
                .unwrap_or_default(),
            record.state.label(),
            if record.split == Split::Heldout { "heldout" } else { "train" },
            match record.flagged_by.as_deref() {
                Some(by) => format!("{by}: {}", one_line(&record.note, 54)),
                None => one_line(&record.note, 60),
            },
            spans
        ));
    }
    Reply::Text(lines.join("\n"))
}

/// "rewriting (attempt 2, started 3 min ago)", for a pending mistake's row.
fn rewriting(record: &Record, now: &chrono::DateTime<chrono::Local>) -> String {
    let started = record
        .rewrite
        .as_ref()
        .and_then(|rewrite| crate::clock::parse(&rewrite.launched_at))
        .map(|at| format!(", started {} ago", crate::clock::span(crate::clock::between_shown(&at, now))))
        .unwrap_or_default();
    format!("rewriting (attempt {}{started})", record.attempts.max(1))
}

// ------------------------------------------------------------------ the agent's own flags

/// The agent's tool: flag a mistake of its own, for the same rewrite and review as
/// the operator's `/mistake`.
pub const TOOL: &str = "mistake_flag";

/// At most this many of the agent's own flags from one turn, and in any 24 hours:
/// a flag costs a rewrite on the engine and a review of the operator's time.
const SELF_FLAGS_PER_TURN: usize = 3;
const SELF_FLAGS_PER_DAY: usize = 20;

/// How far back an event the agent points at may be: its newest this many calls.
/// It bounds the scan of the call store, and covers days of work.
const EVENT_WITHIN_CALLS: usize = 2_000;

/// How far back, without an event, the step to flag is looked for.
const STEP_WITHIN_CALLS: usize = 50;

/// How long the tool waits for the flag to be logged. Past it, logging
/// finishes in the background and the operator is told how it went: the turn never
/// waits on the call store's writer or the mistakes lock.
const TOOL_WAIT: Duration = Duration::from_secs(3);

/// Calls that only flag or report: bookkeeping about a mistake, never the
/// mistake itself (a refused flag retried, a weakness reported first).
const BOOKKEEPING: [&str; 4] = [TOOL, "weakness_report", "weakness_add_instance", "weakness_list"];

const TOOL_DESCRIPTION: &str = "Flag a mistake you made, for the operator to review and for training. Use it when you see \
that something you did was wrong: a claim that turned out false, a tool result you misread, a command or edit that \
was wrong, a check you skipped, a conclusion you jumped to. Use it whether you noticed it yourself or the operator pointed \
it out. It records your steps from the operator's last message up to the mistake; a corrected version of the step where it \
went wrong is drafted in the background, and the operator approves, edits or discards it. Approved corrections train you. \
It changes nothing now: still fix what the mistake broke, or tell the operator. Each flag costs a rewrite and the operator's time, \
so flag mistakes that matter, once each; not uncertainty, an ordinary retry after a command failed, or failures that \
weren't your doing. For a pattern that keeps recurring, use weakness_report as well.";

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![ToolDefinition::function(
        TOOL,
        TOOL_DESCRIPTION,
        serde_json::json!({
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "What was wrong, and what would have been right if you \
know, in a sentence or two. The rewriter reads it as a hint, and the correction may not quote it."},
                "event_id": {"type": "integer", "description": "Optional, to point at an earlier step: an archive event \
from where it went wrong, such as the tool result you misread or that came from a wrong command (the \
archive_event_id a result ends with), or your reply (archive_search finds its id). The flag then covers the steps \
from that event's to your latest. Without it, your latest step before this call is flagged, with the steps since \
The operator last wrote."},
                "weakness": {"type": "string", "description": "Optional: a logged weakness this mistake is an instance \
of (wk-NNNN, from weakness_list), to link them."}
            },
            "required": ["reason"],
            "additionalProperties": false
        }),
    )]
}

/// What the agent asked for, with the call that asked.
#[derive(Debug)]
struct SelfFlag {
    reason: String,
    event_id: Option<u64>,
    weakness: Option<String>,
    /// The call whose tool call this is: its reply is the flag, not the mistake.
    current: CallRef,
    /// Its archive events so far, which the store doesn't have yet.
    current_events: Vec<u64>,
}

/// Run `mistake_flag`. Always a JSON result for the agent, never an error: a
/// refusal says why.
pub fn execute_tool(call: &ToolCall) -> String {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        reason: String,
        #[serde(default)]
        event_id: Option<u64>,
        #[serde(default)]
        weakness: Option<String>,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return refusal(&format!("bad arguments: {error}")),
    };
    let (Some(book), Some(recorder)) = (book(), calls::recorder()) else {
        return refusal("mistakes can't be flagged in this session: the call store is not running");
    };
    let flag = match prepare_own(book, recorder, args.reason, args.event_id, args.weakness) {
        Ok(flag) => flag,
        Err(error) => return refusal(&error),
    };
    let finished = bounded(
        TOOL_WAIT,
        move || log_own(book, recorder, &flag),
        // Too late for the agent's answer: the operator hears of a failure instead.
        |result| {
            if let Err(error) = result {
                push_status(format!("[mistake] s0ma's flag was not logged: {error}"));
            }
        },
    );
    match finished {
        Some(Ok(result)) => result.to_string(),
        Some(Err(error)) => refusal(&error),
        None => serde_json::json!({
            "ok": true, "flagged": "pending",
            "note": "The call store is busy, so the flag is being logged in the background; the operator will see it, or why \
it couldn't be logged. This changes nothing now: if the mistake still matters, fix what it broke or tell the operator.",
        })
        .to_string(),
    }
}

fn refusal(error: &str) -> String {
    serde_json::json!({"ok": false, "flagged": false, "error": error}).to_string()
}

/// Run `work` on a thread of its own and wait up to `wait` for it. Its result,
/// or None when it outlasts the wait; then `late` gets the result when it
/// comes. One lock decides which, so a result is never lost between them.
fn bounded<W, L>(wait: Duration, work: W, late: L) -> Option<Result<Value>>
where
    W: FnOnce() -> Result<Value> + Send + 'static,
    L: FnOnce(Result<Value>) + Send + 'static,
{
    use std::sync::Condvar;
    // (the result, if it came in time; whether the caller stopped waiting)
    let shared = Arc::new((Mutex::new((None::<Result<Value>>, false)), Condvar::new()));
    let worker = Arc::clone(&shared);
    let spawned = std::thread::Builder::new().name("s0ma-mistake-flag".into()).spawn(move || {
        let result = work();
        let (slot, ready) = &*worker;
        let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
        if slot.1 {
            drop(slot);
            late(result);
        } else {
            slot.0 = Some(result);
            ready.notify_all();
        }
    });
    if let Err(error) = spawned {
        return Some(Err(format!("cannot start logging the flag: {error}")));
    }
    let (slot, ready) = &*shared;
    let slot = slot.lock().unwrap_or_else(|e| e.into_inner());
    let (mut slot, _) = ready.wait_timeout_while(slot, wait, |slot| slot.0.is_none()).unwrap_or_else(|e| e.into_inner());
    match slot.0.take() {
        Some(result) => Some(result),
        None => {
            slot.1 = true;
            None
        }
    }
}

/// What can be checked on the agent's thread, at once: the arguments, and which
/// call is asking.
fn prepare_own(
    book: &Book,
    recorder: &calls::Recorder,
    reason: String,
    event_id: Option<u64>,
    weakness: Option<String>,
) -> Result<SelfFlag> {
    let reason = reason.trim().to_owned();
    if reason.split_whitespace().count() < 3 {
        return Err("give the reason in a sentence: what was wrong, and what would have been right".into());
    }
    if reason.chars().count() > 2_000 {
        return Err("keep the reason to a sentence or two (under 2,000 characters)".into());
    }
    if let Some(id) = &weakness {
        check_weakness(&book.data_dir, id)?;
    }
    let current = recorder.latest().ok_or("no call of yours is saved yet")?;
    let current_events = calls::open_call_events()
        .filter(|(call_id, _)| *call_id == current.call_id)
        .map(|(_, events)| events)
        .unwrap_or_default();
    Ok(SelfFlag { reason, event_id, weakness, current, current_events })
}

/// Whether a saved call only flagged or reported (see `BOOKKEEPING`).
fn is_bookkeeping(store: &Store, line: &Value) -> bool {
    let Ok(response) = store.response(line) else { return false };
    let content = response.get("content").and_then(Value::as_str).unwrap_or("");
    let names: Vec<&str> = response
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().filter_map(|c| c.pointer("/function/name").and_then(Value::as_str)).collect())
        .unwrap_or_default();
    content.trim().is_empty() && !names.is_empty() && names.iter().all(|name| BOOKKEEPING.contains(name))
}

/// One self-flag at a time, so two in one batch see each other.
fn self_flag_lock() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Find the step, check the limits, then log it as `/mistake` would, or add
/// the reason to an open mistake that already covers the step.
fn log_own(book: &Book, recorder: &calls::Recorder, flag: &SelfFlag) -> Result<Value> {
    let _serial = self_flag_lock();
    let records = book.all()?;
    let own: Vec<&Record> = records.iter().filter(|r| r.flagged_by.as_deref() == Some("s0ma")).collect();
    if own.iter().filter(|r| r.flagged_in.as_deref() == Some(flag.current.turn_id.as_str())).count() >= SELF_FLAGS_PER_TURN {
        return Err(format!("you have flagged {SELF_FLAGS_PER_TURN} mistakes this turn already; tell the operator about any others"));
    }
    let day_ago = chrono::Local::now() - chrono::Duration::hours(24);
    let recent = own
        .iter()
        .filter(|r| chrono::DateTime::parse_from_rfc3339(&r.logged_at).is_ok_and(|at| at >= day_ago))
        .count();
    if recent >= SELF_FLAGS_PER_DAY {
        return Err(format!(
            "you have flagged {SELF_FLAGS_PER_DAY} mistakes in the last 24 hours already; tell the operator about this one"
        ));
    }
    if !recorder.sync(Duration::from_secs(30)) {
        return Err("the call store is behind; try again in a minute".into());
    }
    let store = book.store();
    let pointed = flag.event_id.filter(|id| !flag.current_events.contains(id));
    let owner = match pointed {
        Some(id) => Some(store.call_with_event(id, EVENT_WITHIN_CALLS).map_err(|e| e.to_string())?.ok_or_else(|| {
            format!(
                "archive event {id} isn't one of your steps among the last {EVENT_WITHIN_CALLS} saved (it may be \
                 the operator's message, a harness note, or older); give the event of a tool result of yours or of your reply, \
                 or leave event_id out to flag your latest step"
            )
        })?),
        None => None,
    }
    .filter(|owner| owner.call_id != flag.current.call_id);

    // The newest step that is work, not bookkeeping: before this call, and
    // from the step pointed at on, in its turn.
    let candidates: Vec<Value> = match &owner {
        Some(owner) if owner.turn_id != flag.current.turn_id => {
            let mut lines = store.turn_lines(&owner.turn_id).map_err(|e| e.to_string())?;
            lines.reverse();
            lines
        }
        _ => store.calls_before(&flag.current.call_id, STEP_WITHIN_CALLS).map_err(|e| e.to_string())?,
    };
    let in_reach = |call: &CallRef| match &owner {
        Some(owner) => call.turn_id == owner.turn_id && call.call_in_turn >= owner.call_in_turn,
        None => true,
    };
    let flagged = candidates
        .iter()
        .filter_map(|line| calls::call_ref(line).filter(|call| in_reach(call)).map(|call| (call, line)))
        .find(|(_, line)| !is_bookkeeping(&store, line))
        .map(|(call, _)| call)
        .or_else(|| owner.clone())
        .ok_or("no earlier step of yours is saved to flag")?;
    if let Some(error) = recorder.save_error(&flagged.call_id) {
        return Err(format!("that step was not saved ({error}), so it can't be flagged"));
    }

    // Its steps: its stretch, reaching back to the step pointed at if that is
    // earlier (a steer may have come between).
    let mut steps = book.stretch_for(&flagged);
    if let Some(owner) = owner.as_ref().filter(|o| o.turn_id == flagged.turn_id)
        && steps.first().is_some_and(|first| first.call_in_turn > owner.call_in_turn)
    {
        steps = store
            .turn_calls(&flagged.turn_id)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|(call, _)| call)
            .filter(|call| (owner.call_in_turn..=flagged.call_in_turn).contains(&call.call_in_turn))
            .map(|call| StretchCall { call_id: call.call_id, call_in_turn: call.call_in_turn, at: call.at })
            .collect();
    }
    let mut note = flag.reason.clone();
    if let (Some(id), Some(owner)) = (pointed, &owner)
        && let Some(k) = steps.iter().position(|step| step.call_id == owner.call_id)
    {
        note.push_str(&format!("\n(s0ma pointed at archive event {id}, from step {} of {}.)", k + 1, steps.len()));
    }
    let what = store
        .find(&flagged.call_id)
        .ok()
        .flatten()
        .and_then(|line| store.response(&line).ok())
        .map(|response| gist(&response))
        .unwrap_or_default();
    let which = format!(
        "call {} of {} ({what})",
        flagged.call_in_turn,
        if flagged.turn_id == flag.current.turn_id { "this turn".to_owned() } else { format!("turn {}", flagged.turn_id) }
    );

    // An open mistake on that step already: one record, with every reason.
    let open = records.iter().find(|r| {
        (r.state == State::Pending || r.state.waiting())
            && (r.call_id == flagged.call_id || r.steps().iter().any(|step| step.call_id == flagged.call_id))
    });
    if let Some(open) = open {
        let normal = |text: &str| text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        if normal(&open.note).contains(&normal(&flag.reason)) {
            return Ok(serde_json::json!({
                "ok": true, "flagged": false, "already_flagged": open.mistake, "step": which,
                "note": format!("Mistake #{} already has this reason for that step; nothing new was logged.", open.mistake),
            }));
        }
        let combined = format!("{}\nAlso, from s0ma: {note}", open.note);
        let manual = open.current.as_deref().is_some_and(|rel| rel.starts_with("edits/"));
        // The agent's own flag, nothing of the operator's in it yet: drafted again with
        // both reasons. Otherwise the reason waits for the operator's `r`.
        let redraft = open.flagged_by.as_deref() == Some("s0ma") && open.state.waiting() && !manual;
        if redraft {
            book.regenerate_as(open.mistake, Some(&combined), "s0ma")?;
        } else {
            book.amend_note(open.mistake, &combined, "s0ma")?;
        }
        push_status(format!(
            "[mistake] s0ma added to #{}: “{}”{}",
            open.mistake,
            one_line(&flag.reason, 120),
            if redraft {
                " — rewriting with both reasons"
            } else if open.state == State::Pending {
                " — the rewrite running has the earlier note; r in /mistake view runs it with this too"
            } else {
                " — r in /mistake view rewrites with it"
            }
        ));
        return Ok(serde_json::json!({
            "ok": true, "flagged": true, "added_to": open.mistake, "step": which,
            "note": format!("Mistake #{} already covers that step; your reason was added to it for the operator. This \
changes nothing now: if the mistake still matters, fix what it broke or tell the operator.", open.mistake),
        }));
    }

    let options = LogOptions {
        by: Some("s0ma".into()),
        flagged_in: Some(flag.current.turn_id.clone()),
        pointed_at: pointed.filter(|_| owner.is_some()),
        steps: Some(steps),
    };
    let (record, launch) = book.log_with(&flagged, &note, flag.weakness.clone(), options)?;
    let steps = record.steps().len();
    let rewrite = match &launch {
        Launch::Started => "rewriting".to_owned(),
        Launch::Failed(error) => format!("the rewriter could not start: {error}"),
    };
    // The operator hears of it at once, as of his own flags.
    push_status(format!(
        "[mistake] s0ma flagged #{} (call {} of turn {}; {steps} step{}): “{}” — {rewrite}",
        record.mistake,
        record.call_in_turn,
        record.turn_id,
        if steps == 1 { "" } else { "s" },
        one_line(&flag.reason, 120),
    ));
    Ok(serde_json::json!({
        "ok": true,
        "flagged": true,
        "mistake": record.mistake,
        "step": which,
        "steps_covered": steps,
        "rewrite": rewrite,
        "note": "Logged for the operator to review; a correction is drafted in the background. This changes nothing now: if \
the mistake still matters, fix what it broke or tell the operator.",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calls::{Outcome, Recorder, Usage};
    use crate::llm::{S0maResponse, ToolDefinition};
    use crate::types::{FunctionCall, Message};
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;
    use std::time::Instant;

    fn temp_dir(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "s0ma-mistakes-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn original() -> S0maResponse {
        S0maResponse {
            content: "The build started ten minutes ago.".into(),
            reasoning_content: "The log says it started ten minutes ago.\nSo it is still early.".into(),
            tool_calls: vec![ToolCall {
                id: "call_8f2".into(),
                kind: "function".into(),
                function: FunctionCall { name: "run_command".into(), arguments: "{\"command\":\"cargo test -p parser\"}".into() },
            }],
            finish_reason: Some("tool_calls".into()),
        }
    }

    /// A data folder with one saved call, and the recorder that saved it.
    fn with_call(tag: &str) -> (PathBuf, Recorder, CallRef) {
        let dir = temp_dir(tag);
        let recorder = Recorder::start(&dir, "http://127.0.0.1:9".into(), "m".into(), Box::new(|_| {})).unwrap();
        let messages = vec![
            Message::system("SYSTEM".into()),
            Message::user("When did the build start? The log shows 09:02.".into()),
        ];
        let tools = vec![ToolDefinition::function("run_command", "Run.", json!({"type": "object"}))];
        let body = crate::llm::main_request_body_for_test(&messages, &tools);
        let _turn = calls::begin_turn();
        let ticket = recorder.begin();
        recorder.finish(ticket, Some(body), Outcome::Response(&original()), Usage::default());
        calls::wait_written(&recorder);
        recorder.end_turn();
        let line = Store::at(&dir).recent(1).unwrap().remove(0);
        let call = calls::call_ref(&line).unwrap();
        (dir, recorder, call)
    }

    fn corrected(touch: &Path) -> Value {
        json!({
            "content": "The build started at 09:02, about three hours ago.",
            "reasoning_content": "The log says it started at 09:02.\nSo it has been running for three hours.",
            "tool_calls": [
                {"id": "call_8f2", "type": "function",
                 "function": {"name": "run_command", "arguments": "{\"command\":\"cargo test -p parser\"}"}},
                {"id": "call_new1", "type": "function",
                 "function": {"name": "run_command",
                              "arguments": json!({"command": format!("touch {}", touch.display())}).to_string()}}
            ]
        })
    }

    /// A rewriter that writes `proposal` (with @M@ and @A@ filled in) as the
    /// attempt's proposal, after `delay` seconds.
    fn stub(dir: &Path, proposal: &Value, delay: &str) -> String {
        let canned = dir.join(format!("canned-{}.json", calls::sha256_hex(proposal.to_string().as_bytes())));
        fs::write(&canned, proposal.to_string()).unwrap();
        format!(
            "sleep {delay}; sed -e 's/\"@M@\"/{{mistake}}/; s/\"@A@\"/{{attempt}}/' {} > {{data}}/mistakes/proposals/{{mistake}}-{{attempt}}.json",
            crate::util::shell_quote(&canned.to_string_lossy())
        )
    }

    fn proposal(status: &str, corrected: Option<Value>) -> Value {
        json!({"mistake": "@M@", "attempt": "@A@", "status": status, "error": null, "corrected": corrected,
               "edits": [{"field": "reasoning_content", "find": "ten minutes ago", "replace": "at 09:02"},
                         {"field": "content", "find": "ten minutes ago", "replace": "at 09:02, about three hours ago"}],
               "summary": "elapsed time read from the wrong timestamp", "changed_tokens": 41, "total_tokens": 812})
    }

    fn wait_until(book: &Book, n: u64, done: impl Fn(&Record) -> bool) -> Record {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let record = book.load(n).unwrap();
            if done(&record) {
                return record;
            }
            assert!(Instant::now() < deadline, "timed out; record: {record:?}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn settled(book: &Book, n: u64) -> Record {
        wait_until(book, n, |r| r.state != State::Pending)
    }

    fn history(dir: &Path) -> Vec<Value> {
        fs::read_to_string(dir.join("mistakes/history.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn every_transition_is_recorded_and_approval_needs_a_real_change() {
        let (dir, _recorder, call) = with_call("states");
        let book = Book::with_rewriter(&dir, stub(&dir, &proposal("ready", Some(corrected(&dir.join("x")))), "0"));
        let n = book.log(&call, "wrong elapsed time", Some("wk-0027".into())).unwrap().0.mistake;
        settled(&book, n);
        let approved = book.approve(n, None).unwrap();
        assert_eq!((approved.state, approved.before), (State::Approved, Some(State::Ready)));
        assert!(book.approve(n, None).is_err(), "approving twice");
        // Reopen: back to review, and out of training until approved again.
        let reopened = book.reopen(n).unwrap();
        assert_eq!((reopened.state, reopened.final_.clone()), (State::Ready, None));
        let discarded = book.discard(n, Some("not a real mistake")).unwrap();
        assert_eq!(discarded.state, State::Discarded);
        assert_eq!(book.reopen(n).unwrap().state, State::Ready);
        // Regenerate from approved: the approval is withdrawn.
        book.approve(n, None).unwrap();
        let (again, _) = book.regenerate(n, Some("the log line at 09:02 is the start")).unwrap();
        assert_eq!((again.state, again.attempts, again.final_.clone()), (State::Pending, 2, None));
        assert_eq!(again.notes.len(), 2);
        assert_eq!(fs::read_to_string(dir.join(format!("mistakes/notes/{n}-2.txt"))).unwrap(), "the log line at 09:02 is the start");
        let ready = settled(&book, n);
        assert_eq!((ready.state, ready.proposal.as_deref()), (State::Ready, Some(format!("proposals/{n}-2.json").as_str())));
        assert_eq!(ready.weakness.as_deref(), Some("wk-0027"));

        let lines = history(&dir);
        let moves: Vec<(Option<&str>, Option<&str>, &str)> = lines
            .iter()
            .map(|l| (l["from"].as_str(), l["to"].as_str(), l["by"].as_str().unwrap()))
            .collect();
        assert_eq!(
            moves,
            vec![
                (None, Some("pending"), "operator"),
                (Some("pending"), Some("ready"), "rewriter"),
                (Some("ready"), Some("approved"), "operator"),
                (Some("approved"), Some("ready"), "operator"),
                (Some("ready"), Some("discarded"), "operator"),
                (Some("discarded"), Some("ready"), "operator"),
                (Some("ready"), Some("approved"), "operator"),
                (Some("approved"), Some("pending"), "operator"),
                (Some("pending"), Some("ready"), "rewriter"),
            ]
        );
        assert!(lines[3]["detail"].as_str().unwrap().contains("withdrawn"));
        assert!(lines[7]["detail"].as_str().unwrap().contains("withdrawn"));
        // Nothing was deleted along the way.
        for attempt in 1..=2 {
            assert!(dir.join(format!("mistakes/proposals/{n}-{attempt}.json")).exists());
        }
    }

    #[test]
    fn an_unchanged_or_unapplied_version_cannot_be_approved() {
        let (dir, _recorder, call) = with_call("refuse");
        let same = serde_json::to_value(Response {
            content: original().content,
            reasoning_content: original().reasoning_content,
            tool_calls: original().tool_calls,
        })
        .unwrap();
        let book = Book::with_rewriter(&dir, stub(&dir, &proposal("ready", Some(same)), "0"));
        let n = book.log(&call, "x", None).unwrap().0.mistake;
        settled(&book, n);
        assert!(book.approve(n, None).unwrap_err().contains("nothing changed"));

        let book = Book::with_rewriter(&dir, stub(&dir, &proposal("needs_edit", Some(corrected(&dir.join("x")))), "0"));
        let n = book.log(&call, "y", None).unwrap().0.mistake;
        let record = settled(&book, n);
        assert_eq!(record.state, State::NeedsEdit);
        assert!(book.approve(n, None).unwrap_err().contains("edit it by hand"));
        let shown = book.shown(&record).unwrap();
        book.save_edit(n, &shown.response, &shown.file, &shown.target).unwrap();
        assert_eq!(book.approve(n, None).unwrap().state, State::Edited);

        // A status the harness doesn't know is not taken.
        let book = Book::with_rewriter(&dir, stub(&dir, &proposal("approved", Some(corrected(&dir.join("x")))), "0"));
        let n = book.log(&call, "z", None).unwrap().0.mistake;
        let record = settled(&book, n);
        assert_eq!(record.state, State::Failed);
        assert!(record.error.unwrap().contains("not one of ready, needs_edit, failed"));
    }

    #[test]
    fn the_approval_hash_is_of_the_object_as_the_file_holds_it() {
        let (dir, _recorder, call) = with_call("hash");
        // A proposal whose `corrected` leaves out an empty tool_calls list.
        let corrected = json!({"content": "Started at 09:02 — three hours ago.", "reasoning_content": "From the log line."});
        let book = Book::with_rewriter(&dir, stub(&dir, &proposal("ready", Some(corrected)), "0"));
        let n = book.log(&call, "wrong elapsed time again", None).unwrap().0.mistake;
        settled(&book, n);
        let approved = book.approve(n, None).unwrap();
        // Written out by hand: sorted keys, no whitespace, UTF-8 as is.
        let canonical = "{\"content\":\"Started at 09:02 — three hours ago.\",\"reasoning_content\":\"From the log line.\"}";
        assert_eq!(approved.final_.as_ref().unwrap().corrected_sha256, calls::sha256_hex(canonical.as_bytes()));
        let parsed = book.shown(&approved).unwrap().response;
        assert_ne!(response_sha256(&parsed), calls::sha256_hex(canonical.as_bytes()), "not this program's re-reading");
    }

    #[test]
    fn approval_is_of_the_version_on_screen() {
        let (dir, _recorder, call) = with_call("displayed");
        let book = Book::with_rewriter(&dir, stub(&dir, &proposal("ready", Some(corrected(&dir.join("x")))), "0"));
        let n = book.log(&call, "wrong elapsed time", None).unwrap().0.mistake;
        settled(&book, n);
        let error = book.approve(n, Some("original")).unwrap_err();
        assert!(error.contains("changed since it was shown"), "{error}");
        assert_eq!(book.load(n).unwrap().state, State::Ready);
        assert_eq!(book.approve(n, Some(&format!("proposals/{n}-1.json"))).unwrap().state, State::Approved);
    }

    /// A data folder with one turn of four calls, the operator steering after the
    /// second: the steps of a mistake on the fourth are the third and fourth.
    fn with_stretch(tag: &str) -> (PathBuf, Recorder, Vec<CallRef>) {
        let dir = temp_dir(tag);
        let recorder = Recorder::start(&dir, "http://127.0.0.1:9".into(), "m".into(), Box::new(|_| {})).unwrap();
        let events = std::cell::Cell::new(100);
        let save = |text: &str| {
            let body = crate::llm::main_request_body_for_test(&[Message::user(text.into())], &[]);
            let ticket = recorder.begin();
            let response = S0maResponse { content: format!("reply to {text}"), ..original() };
            recorder.finish(ticket, Some(body), Outcome::Response(&response), Usage::default());
            calls::wait_written(&recorder);
            // What the call became, archived: the next call's start writes it
            // as an update line between this turn's calls, as in real use.
            events.set(events.get() + 1);
            calls::archived(events.get(), true);
        };
        {
            let _earlier = calls::begin_turn();
            save("an earlier turn");
        }
        recorder.end_turn();
        let _turn = calls::begin_turn();
        save("step one");
        save("step two");
        calls::operator_joined();
        save("step three");
        save("step four");
        recorder.end_turn();
        let mut refs: Vec<CallRef> = Store::at(&dir).recent(4).unwrap().iter().filter_map(calls::call_ref).collect();
        refs.reverse();
        (dir, recorder, refs)
    }

    #[test]
    fn a_step_the_operator_chose_survives_a_rewrite_that_honours_it() {
        let (dir, _recorder, refs) = with_stretch("chosen");
        let first = &refs[2];
        let mut honouring = proposal("ready", Some(json!({"content": "Checked first.", "reasoning_content": "", "tool_calls": []})));
        honouring["target_call_id"] = json!(first.call_id);
        // The first attempt fails (an old qref, say); then the operator picks step one and runs it again.
        let template = format!(
            "if [ {{attempt}} = 1 ]; then exit 2; fi; {}",
            stub(&dir, &honouring, "0")
        );
        let book = Book::with_rewriter(&dir, template);
        let (record, _) = book.log(&refs[3], "misread the log a step back", None).unwrap();
        assert_eq!(settled(&book, record.mistake).state, State::Failed);
        book.retarget(record.mistake, &first.call_id).unwrap();
        book.regenerate(record.mistake, None).unwrap();
        let record = settled(&book, record.mistake);
        assert_eq!((record.target_id(), record.target_by.as_deref()), (first.call_id.as_str(), Some("operator")));
    }

    #[test]
    fn a_correction_can_target_an_earlier_step_and_trains_on_it() {
        let (dir, _recorder, refs) = with_stretch("target");
        let first = &refs[2];
        let mut targeted = proposal("ready", Some(json!({"content": "Checking the log first.", "reasoning_content": "Read it again.",
            "tool_calls": []})));
        targeted["target_call_id"] = json!(first.call_id);
        let book = Book::with_rewriter(&dir, stub(&dir, &targeted, "0"));
        let (record, _) = book.log(&refs[3], "misread the log a step back", None).unwrap();
        let record = settled(&book, record.mistake);
        assert_eq!((record.state, record.target_id()), (State::Ready, first.call_id.as_str()));
        assert_eq!(record.target_by.as_deref(), Some("rewriter"));
        let shown = book.shown(&record).unwrap();
        assert_eq!(shown.target, first.call_id);
        // The original side is that step's own response.
        assert_eq!(book.original(&record).unwrap().content, "reply to step three");
        let approved = book.approve(record.mistake, Some(&shown.file)).unwrap();
        assert_eq!(approved.final_.unwrap().target_call_id.as_deref(), Some(first.call_id.as_str()));

        // The operator picks the other step, edits it, approves that.
        book.reopen(record.mistake).unwrap();
        let moved = book.retarget(record.mistake, &refs[3].call_id).unwrap();
        assert_eq!(moved.current, None, "nothing made for that step yet: its original");
        assert_eq!(moved.target_by.as_deref(), Some("operator"), "a rewrite run again corrects this step");
        let shown = book.shown(&moved).unwrap();
        assert_eq!((shown.file.as_str(), shown.response.content.as_str()), ("original", "reply to step four"));
        let edited = Response { content: "Fixed.".into(), ..shown.response.clone() };
        book.save_edit(record.mistake, &edited, &shown.file, &shown.target).unwrap();
        let final_ = book.approve(record.mistake, None).unwrap().final_.unwrap();
        assert_eq!((final_.source.as_str(), final_.target_call_id.as_deref()), ("manual", Some(refs[3].call_id.as_str())));

        // Back to the first step: the proposal made for it is shown again.
        book.reopen(record.mistake).unwrap();
        assert_eq!(book.retarget(record.mistake, &first.call_id).unwrap().current.as_deref(), Some("proposals/1-1.json"));
        // Nothing outside the stretch.
        assert!(book.retarget(record.mistake, &refs[0].call_id).unwrap_err().contains("not a step"));
    }

    #[test]
    fn a_proposal_for_a_call_outside_the_stretch_is_refused() {
        let (dir, _recorder, refs) = with_stretch("outside");
        let mut stray = proposal("ready", Some(corrected(&dir.join("x"))));
        stray["target_call_id"] = json!(refs[0].call_id);
        let book = Book::with_rewriter(&dir, stub(&dir, &stray, "0"));
        let (record, _) = book.log(&refs[3], "misread the log a step back", None).unwrap();
        let record = settled(&book, record.mistake);
        assert_eq!(record.state, State::Failed);
        assert!(record.error.unwrap().contains("not a step of this mistake's stretch"));
    }

    // ---- the agent's own flags ----

    /// A turn left open, as while the agent's tools run. An earlier turn has one
    /// call (event 101). Then each step is saved in order with one archive
    /// event each (102, 103, …): "steer" marks the operator's message joining the
    /// turn, and a step's tools are the tool calls it made. The last step is
    /// the call asking for `mistake_flag`; its event isn't in the store yet.
    fn live_turn(tag: &str, steps: &[(&str, &[&str])]) -> (PathBuf, Recorder, Vec<CallRef>, calls::TurnGuard) {
        let dir = temp_dir(tag);
        let recorder = Recorder::start(&dir, "http://127.0.0.1:9".into(), "m".into(), Box::new(|_| {})).unwrap();
        let event = std::cell::Cell::new(100);
        let save = |text: &str, tools: &[&str]| {
            let body = crate::llm::main_request_body_for_test(&[Message::user(text.into())], &[]);
            let ticket = recorder.begin();
            let tool_calls = tools
                .iter()
                .enumerate()
                .map(|(i, name)| ToolCall {
                    id: format!("call_{text}_{i}"),
                    kind: "function".into(),
                    function: FunctionCall { name: (*name).into(), arguments: "{}".into() },
                })
                .collect();
            let content = if tools.is_empty() { format!("reply: {text}") } else { String::new() };
            let response = S0maResponse { content, reasoning_content: format!("thinking: {text}"), tool_calls, finish_reason: None };
            recorder.finish(ticket, Some(body), Outcome::Response(&response), Usage::default());
            calls::wait_written(&recorder);
            event.set(event.get() + 1);
            calls::archived(event.get(), true);
        };
        {
            let _earlier = calls::begin_turn();
            save("earlier turn", &[]);
            recorder.end_turn();
        }
        let guard = calls::begin_turn();
        for (text, tools) in steps {
            if *text == "steer" {
                calls::operator_joined();
            } else {
                save(text, tools);
            }
        }
        calls::wait_written(&recorder);
        let turn = calls::current_turn().unwrap();
        let refs = Store::at(&dir).turn_calls(&turn).unwrap().into_iter().map(|(call, _)| call).collect();
        (dir, recorder, refs, guard)
    }

    fn flag_as_s0ma(book: &Book, recorder: &Recorder, reason: &str, event: Option<u64>) -> Result<Value> {
        let flag = prepare_own(book, recorder, reason.into(), event, None)?;
        log_own(book, recorder, &flag)
    }

    #[test]
    fn a_flag_or_report_is_never_taken_for_the_mistake() {
        // A refused flag and a weakness report came first: the step before
        // them is the one meant.
        let (dir, recorder, refs, _turn) = live_turn(
            "own-bookkeeping",
            &[("one", &["vm_exec"]), ("two", &[TOOL]), ("three", &["weakness_report"]), ("flag", &[TOOL])],
        );
        let book = Book::with_rewriter(&dir, "exit 2".into());
        let result = flag_as_s0ma(&book, &recorder, "I ran the wrong command", None).unwrap();
        assert_eq!(book.load(result["mistake"].as_u64().unwrap()).unwrap().call_id, refs[0].call_id);
        // A turn of bookkeeping only: the previous turn's work.
        let (dir, recorder, _refs, _turn) = live_turn("own-bookkeeping-turn", &[("two", &[TOOL]), ("flag", &[TOOL])]);
        let book = Book::with_rewriter(&dir, "exit 2".into());
        let result = flag_as_s0ma(&book, &recorder, "my last answer was wrong", None).unwrap();
        assert!(result["step"].as_str().unwrap().contains("(“reply: earlier turn”)"), "{result}");
    }
}

