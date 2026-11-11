//! The board workflow: from a finished board to a delivered one, through
//! The operator's review and an independent verification.
//!
//! ```text
//! working ──s0ma: board_report──▶ review ──the operator: back──▶ working (a new round)
//!                                   │
//!                                   └──the operator: verify──▶ verification ──the operator: accept──▶ deliverable
//!                                                          │
//!                                                          └──the operator: back──▶ working
//! ```
//!
//! Any stage goes back to any stage before it, as often as the operator likes:
//! from verification to review (`to_review`) or to working (`send_back`);
//! from the deliverables to verification (`to_verification`, verified again),
//! to review, or to working. A board can so be reviewed and verified any
//! number of times, delivered or not.
//!
//! A board s0ma proposes exhausted goes the same way to review, with a
//! write-up of what was tried and why nothing is left; from review the operator
//! closes it exhausted (it joins the deliverables: closed, nothing to
//! verify) or sends it back.
//!
//! - **Working.** An open board, or one s0ma finished (`proposed_solved`, or
//!   `solved` by its test) whose write-up is due. s0ma writes it with
//!   `board_report`: what was done, how it was checked, what is left, and the
//!   files that make it up. They are copied from s0ma's machine into
//!   `/share/boards/<name>/` (snapshots: `report.md`, `files/…`), which the
//!   share sync mirrors to the operator's `~/s0ma-share/boards/<name>/`.
//! - **Review.** the operator reads the write-up and the files in his projects tab and
//!   leaves questions and ideas. He sends it back (the board opens again, in a
//!   new round: s0ma answers each note with `board_answer`, works on them and
//!   writes it up again, which brings it back to review) or on to verification.
//! - **Verification.** Two checks, independent of each other: a fresh
//!   subagent the harness starts (it sees only the board, the write-up and the
//!   files), and s0ma's own (`board_verify`). The subagent's report is held
//!   from s0ma until s0ma has filed its own; then s0ma reconciles the two into
//!   the verification report (`board_verification`). The operator accepts it (the
//!   board is delivered: solved, verified) or sends it back to work.
//! - **Deliverable.** Closed and complete: the final report, the verification
//!   report, the review and the files, to read in the projects tab.
//!
//! Each move the operator makes reaches s0ma as an event (`take_events`): those that
//! need work start a turn; acceptance comes with his next message. All of it
//! is kept in `board_flow.sqlite3` beside the boards, so the board's own
//! store and its schema are untouched.

use crate::board::{Board, BoardStatus, BoardStore};
use crate::llm::ToolDefinition;
use crate::share::Ran;
use crate::util::shell_quote;
use crate::types::ToolCall;

use chrono::Local;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use std::error::Error;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Run<'a> = &'a mut dyn FnMut(&str, u64) -> std::result::Result<Ran, String>;

pub const MAX_REPORT_CHARS: usize = 40_000;
pub const MAX_NOTE_CHARS: usize = 2_000;
pub const MAX_ANSWER_CHARS: usize = 4_000;
pub const MAX_VERIFY_CHARS: usize = 20_000;
pub const MAX_FILES: usize = 60;
pub const MAX_FILES_BYTES: u64 = 50 * 1024 * 1024;

/// A verifier that could not start is tried again after this long.
pub const VERIFIER_RETRY: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Working,
    Review,
    Verification,
    Deliverable,
    /// The operator threw it away: closed, and in no subtab.
    Discarded,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Working => "working",
            Stage::Review => "review",
            Stage::Verification => "verification",
            Stage::Deliverable => "deliverable",
            Stage::Discarded => "discarded",
        }
    }
}

/// A file delivered with a write-up: its path inside `files/` (as it was on
/// s0ma's machine, without the leading `/`) and its size.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delivered {
    pub path: String,
    pub bytes: u64,
}

/// Where a board is in the workflow, and what it has gathered.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Flow {
    pub board_id: String,
    pub name: String,
    pub stage: Option<Stage>,
    /// 1, and one more each time the operator sends it back to work.
    pub round: i64,
    pub report: Option<String>,
    /// The round the write-up was written in (it is due again in a later one).
    pub report_round: i64,
    pub report_at: Option<String>,
    pub files: Vec<Delivered>,
    /// When it came into its stage.
    pub entered_at: String,
    /// The independent verifier: its job, `wanted`, `running`, `done` or
    /// `failed`, its report, and why it has not started (or failed).
    pub verifier_job: Option<String>,
    pub verifier_state: Option<String>,
    pub verifier_report: Option<String>,
    pub verifier_note: Option<String>,
    pub verifier_tried_at: Option<String>,
    /// s0ma's own check.
    pub own_verdict: Option<String>,
    pub own_report: Option<String>,
    /// The reconciled verification report.
    pub verdict: Option<String>,
    pub verification: Option<String>,
    pub verified_at: Option<String>,
    pub accepted_at: Option<String>,
    /// Every round's report, oldest first (`report` is the latest), each
    /// also kept as `reports/round-<n>.md` in the board's folder.
    #[serde(default)]
    pub reports: Vec<RoundReport>,
    /// One more at every write (`FlowStore::put`): a write made from a read
    /// older than the last write is refused, not laid over it.
    #[serde(default)]
    pub rev: i64,
}

/// A flow written by someone else since it was read: what was meant is not
/// written; read it again and decide again.
#[derive(Debug)]
pub struct Changed {
    pub board: String,
}

impl std::fmt::Display for Changed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "'{}' changed while this was being done; nothing was written, try again", self.board)
    }
}

impl Error for Changed {}

/// The report s0ma wrote in one round.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundReport {
    pub round: i64,
    pub text: String,
    pub at: String,
}

impl Flow {
    pub fn stage(&self) -> Stage {
        self.stage.unwrap_or(Stage::Working)
    }

    /// The write-up for this round is in.
    pub fn reported(&self) -> bool {
        self.report.is_some() && self.report_round == self.round
    }

    /// Every round's report, oldest first: those kept, or (a flow from
    /// before they were) the one report there is.
    pub fn all_reports(&self) -> Vec<RoundReport> {
        if !self.reports.is_empty() {
            return self.reports.clone();
        }
        match &self.report {
            Some(text) => vec![RoundReport { round: self.report_round, text: text.clone(), at: self.report_at.clone().unwrap_or_default() }],
            None => Vec::new(),
        }
    }

    /// Waiting for the operator: in review, or a verification report to accept.
    pub fn waits_for_operator(&self) -> bool {
        match self.stage() {
            Stage::Review => true,
            Stage::Verification => self.verification.is_some(),
            _ => false,
        }
    }
}

/// One of the operator's review notes (a question, an idea, whatever he has to say
/// about the work), and s0ma's answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Note {
    pub id: String,
    pub board_id: String,
    pub round: i64,
    pub text: String,
    pub created_at: String,
    pub answer: Option<String>,
    pub answered_at: Option<String>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS flow(
    board_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL,
    board_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    answer TEXT,
    answered_at TEXT);
CREATE TABLE IF NOT EXISTS events(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    wake INTEGER NOT NULL,
    created_at TEXT NOT NULL);
";

pub struct FlowStore {
    path: PathBuf,
}

impl FlowStore {
    pub fn beside(boards: &BoardStore) -> Self {
        Self { path: boards.flow_path() }
    }

    fn connect(&self) -> Result<Connection> {
        let mut db = Connection::open(&self.path)?;
        db.busy_timeout(Duration::from_secs(10))?;
        db.execute_batch(SCHEMA)?;
        repair_note_ids(&mut db)?;
        Ok(db)
    }

    pub fn get(&self, board_id: &str) -> Result<Option<Flow>> {
        let db = self.connect()?;
        let body: Option<String> =
            db.query_row("SELECT body FROM flow WHERE board_id=?1", params![board_id], |row| row.get(0)).optional()?;
        Ok(body.map(|body| serde_json::from_str(&body)).transpose()?)
    }

    pub fn all(&self) -> Result<Vec<Flow>> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT body FROM flow")?;
        let bodies = statement.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for body in bodies {
            out.push(serde_json::from_str(&body?)?);
        }
        Ok(out)
    }

    /// `flow` written, if the stored one is still the one it was read from
    /// (its `rev`); otherwise `Changed`, and nothing written. A flow made new
    /// (`rev` 0) is written over whatever is there.
    pub fn put(&self, flow: &Flow) -> Result<()> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let stored: Option<String> =
            tx.query_row("SELECT body FROM flow WHERE board_id=?1", params![flow.board_id], |row| row.get(0)).optional()?;
        let stored_rev = stored.as_deref().and_then(|body| serde_json::from_str::<Flow>(body).ok()).map_or(0, |stored| stored.rev);
        if flow.rev != 0 && stored_rev != flow.rev {
            return Err(Box::new(Changed { board: flow.name.clone() }));
        }
        let written = Flow { rev: stored_rev + 1, ..flow.clone() };
        tx.execute(
            "INSERT OR REPLACE INTO flow VALUES(?1,?2,?3)",
            params![written.board_id, written.name, serde_json::to_string(&written)?],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// `change` made to the board's flow as stored now, and written; read
    /// again and made again if another write came between (a few times).
    /// `None` when there is no flow, or `change` says to leave it.
    pub fn update(&self, board_id: &str, mut change: impl FnMut(&mut Flow) -> bool) -> Result<Option<Flow>> {
        for _ in 0..5 {
            let Some(mut flow) = self.get(board_id)? else { return Ok(None) };
            if !change(&mut flow) {
                return Ok(None);
            }
            match self.put(&flow) {
                Ok(()) => return Ok(Some(flow)),
                Err(error) if error.downcast_ref::<Changed>().is_some() => continue,
                Err(error) => return Err(error),
            }
        }
        Err(format!("the flow of {board_id} kept changing; not written").into())
    }

    /// The board's flow, made (working, round 1) if it has none.
    fn get_or_new(&self, board: &Board) -> Result<Flow> {
        Ok(self.get(&board.id)?.unwrap_or_else(|| Flow {
            board_id: board.id.clone(),
            name: board.name.clone(),
            stage: Some(Stage::Working),
            round: 1,
            entered_at: now(),
            ..Flow::default()
        }))
    }

    pub fn notes(&self, board_id: &str) -> Result<Vec<Note>> {
        let db = self.connect()?;
        let mut statement = db.prepare(
            "SELECT id, board_id, round, kind, text, created_at, answer, answered_at FROM notes \
             WHERE board_id=?1 ORDER BY seq",
        )?;
        let rows = statement.query_map(params![board_id], |row| {
            Ok(Note {
                id: row.get(0)?,
                board_id: row.get(1)?,
                round: row.get(2)?,
                text: row.get(4)?,
                created_at: row.get(5)?,
                answer: row.get(6)?,
                answered_at: row.get(7)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// The operator's note on a board in review, for its round: numbered one
    /// past the highest on the board (not its count, which a note taken back
    /// lowers, giving the next one an id already in use).
    pub fn add_note(&self, board_id: &str, round: i64, text: &str) -> Result<Note> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let id = format!("n{}", highest_note(&tx, board_id)? + 1);
        let created_at = now();
        tx.execute(
            "INSERT INTO notes(id, board_id, round, kind, text, created_at) VALUES(?1,?2,?3,?4,?5,?6)",
            params![id, board_id, round, "note", text, created_at],
        )?;
        tx.commit()?;
        Ok(Note { id, board_id: board_id.into(), round, text: text.into(), created_at, answer: None, answered_at: None })
    }

    /// The operator takes back a note s0ma has not answered.
    pub fn remove_note(&self, board_id: &str, id: &str) -> Result<bool> {
        let db = self.connect()?;
        Ok(db.execute("DELETE FROM notes WHERE board_id=?1 AND id=?2 AND answer IS NULL", params![board_id, id])? > 0)
    }

    fn answer(&self, board_id: &str, id: &str, answer: &str) -> Result<bool> {
        let db = self.connect()?;
        Ok(db.execute(
            "UPDATE notes SET answer=?1, answered_at=?2 WHERE board_id=?3 AND id=?4",
            params![answer, now(), board_id, id],
        )? > 0)
    }

    /// Something for s0ma: `wake` starts a turn, otherwise it comes with
    /// The operator's next message.
    pub fn push_event(&self, text: &str, wake: bool) -> Result<()> {
        let db = self.connect()?;
        db.execute("INSERT INTO events(text, wake, created_at) VALUES(?1,?2,?3)", params![text, wake, now()])?;
        Ok(())
    }

    /// The events waiting, oldest first, taken once.
    pub fn take_events(&self, wake: bool) -> Result<Vec<String>> {
        let mut db = self.connect()?;
        let tx = db.transaction()?;
        let texts = {
            let mut statement = tx.prepare("SELECT text FROM events WHERE wake=?1 ORDER BY seq")?;
            let rows = statement.query_map(params![wake], |row| row.get::<_, String>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        tx.execute("DELETE FROM events WHERE wake=?1", params![wake])?;
        tx.commit()?;
        Ok(texts)
    }

    pub fn has_events(&self, wake: bool) -> Result<bool> {
        let db = self.connect()?;
        Ok(db.query_row("SELECT EXISTS(SELECT 1 FROM events WHERE wake=?1)", params![wake], |row| row.get(0))?)
    }
}

fn now() -> String {
    Local::now().to_rfc3339()
}

/// The highest `n<number>` on a board's notes (0 with none; older `q1`, `i2`
/// ids are not counted).
fn highest_note(db: &Connection, board_id: &str) -> Result<i64> {
    let mut statement = db.prepare("SELECT id FROM notes WHERE board_id=?1")?;
    let ids = statement.query_map(params![board_id], |row| row.get::<_, String>(0))?;
    let mut highest = 0;
    for id in ids {
        if let Some(number) = id?.strip_prefix('n').and_then(|number| number.parse::<i64>().ok()) {
            highest = highest.max(number);
        }
    }
    Ok(highest)
}

/// Notes that came to share an id (numbered by count before): every one after
/// the first is given the board's next free number, oldest first, so an
/// answer or a take-back reaches one note again.
fn repair_note_ids(db: &mut Connection) -> Result<()> {
    let shared: Vec<(String, String)> = {
        let mut statement = db.prepare("SELECT board_id, id FROM notes GROUP BY board_id, id HAVING COUNT(*) > 1")?;
        let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    if shared.is_empty() {
        return Ok(());
    }
    let tx = db.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    for (board_id, id) in shared {
        let later: Vec<i64> = {
            let mut statement = tx.prepare("SELECT seq FROM notes WHERE board_id=?1 AND id=?2 ORDER BY seq")?;
            let rows = statement.query_map(params![board_id, id], |row| row.get(0))?;
            rows.collect::<rusqlite::Result<Vec<i64>>>()?.into_iter().skip(1).collect()
        };
        for seq in later {
            let next = format!("n{}", highest_note(&tx, &board_id)? + 1);
            tx.execute("UPDATE notes SET id=?1 WHERE seq=?2", params![next, seq])?;
        }
    }
    tx.commit()?;
    Ok(())
}

fn bounded(value: &str, min: usize, max: usize, label: &str) -> std::result::Result<String, String> {
    crate::board::bounded(value, min, max, label)
}

/// The board's folder in /share, inside the shared folder.
pub fn folder(name: &str) -> String {
    format!("boards/{name}")
}


// ------------------------------------------------------------------ the store, for the process

static STORE: OnceLock<PathBuf> = OnceLock::new();

/// Where the boards are (the archive's place), for what runs away from a
/// board call: the verifier's result, s0ma's notes.
pub fn init(archive: &std::path::Path) {
    let _ = STORE.set(archive.to_path_buf());
}

fn stores() -> Option<(BoardStore, FlowStore)> {
    let boards = BoardStore::beside_archive(STORE.get()?);
    let flows = FlowStore::beside(&boards);
    Some((boards, flows))
}

// ------------------------------------------------------------------ writing to /share

/// `text` as `relative` in the board's folder, over what was there.
fn put_text(share: &str, name: &str, file: &str, text: &str, run: Run) -> std::result::Result<(), String> {
    let relative = format!("{}/{file}", folder(name));
    crate::shared_files::remove(share, &relative, None, run).map_err(|error| error.to_string())?;
    crate::shared_files::write(share, &relative, text.as_bytes(), &crate::shared_files::Expect::New, run)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

/// The files, copied from s0ma's machine into the board's `files/` (each
/// under its own path), in place of the ones before. What was copied.
fn snapshot(share: &str, name: &str, paths: &[String], run: Run) -> std::result::Result<Vec<Delivered>, String> {
    let dest = format!("{}/{}", share.trim_end_matches('/'), folder(name));
    let quoted: Vec<String> = paths.iter().map(|path| shell_quote(path)).collect();
    let list = quoted.join(" ");
    let copy = if paths.is_empty() {
        String::new()
    } else {
        format!(
            "for p in {list}; do [ -e \"$p\" ] || {{ echo \"not on the machine: $p\" >&2; exit 8; }}; done\n\
             total=$(du -scb {list} | tail -n 1 | cut -f1)\n\
             [ \"$total\" -le {MAX_FILES_BYTES} ] || {{ echo \"the files come to $total bytes; at most {MAX_FILES_BYTES}\" >&2; exit 9; }}\n\
             for p in {list}; do cp -a --parents -- \"$p\" \"$new\"/ || exit 10; done\n\
             count=$(find \"$new\" -type f | wc -l)\n\
             [ \"$count\" -le {MAX_FILES} ] || {{ echo \"$count files; at most {MAX_FILES}\" >&2; exit 11; }}\n"
        )
    };
    let script = format!(
        "dest={dest}; new=\"$dest/.files.new.$$\"\n\
         mkdir -p \"$dest\" || exit 5\n\
         rm -rf -- \"$new\"; mkdir -p \"$new\" || exit 6\n\
         trap 'rm -rf -- \"$new\"' EXIT\n\
         {copy}\
         rm -rf -- \"$dest/files\" && mv -- \"$new\" \"$dest/files\" || exit 12\n\
         trap - EXIT\n\
         find \"$dest/files\" -type f -printf '%P\\t%s\\n' | sort",
        dest = shell_quote(&dest),
    );
    let ran = crate::share::step(run, &script)?;
    Ok(ran
        .stdout
        .lines()
        .filter_map(|line| {
            let (path, bytes) = line.rsplit_once('\t')?;
            Some(Delivered { path: path.to_string(), bytes: bytes.trim().parse().ok()? })
        })
        .collect())
}

/// The operator's notes and s0ma's answers, round by round, as `review.md`.
pub fn review_text(name: &str, notes: &[Note]) -> String {
    let mut text = format!("# Review of {name}\n");
    let mut round = 0;
    for note in notes {
        if note.round != round {
            round = note.round;
            text.push_str(&format!("\n## Round {round}\n"));
        }
        text.push_str(&format!("\n- **{}** (operator): {}\n", note.id, note.text));
        match &note.answer {
            Some(answer) => text.push_str(&format!("  - s0ma: {answer}\n")),
            None => text.push_str("  - (not answered yet)\n"),
        }
    }
    if notes.is_empty() {
        text.push_str("\nNo notes.\n");
    }
    text
}

/// The verification report as kept: the reconciled one, then both checks.
pub fn verification_text(name: &str, flow: &Flow) -> String {
    format!(
        "# Verification of {name}\n\nVerdict: **{}**\n\n{}\n\n## s0ma's own check ({})\n\n{}\n\n## The independent verifier ({})\n\n{}\n",
        flow.verdict.as_deref().unwrap_or("?"),
        flow.verification.as_deref().unwrap_or(""),
        flow.own_verdict.as_deref().unwrap_or("?"),
        flow.own_report.as_deref().unwrap_or("(none)"),
        verdict_of(flow.verifier_report.as_deref().unwrap_or("")),
        flow.verifier_report.as_deref().or(flow.verifier_note.as_deref()).unwrap_or("(none)"),
    )
}

/// `VERDICT: passed` (or failed) in a verifier's report.
pub fn verdict_of(report: &str) -> &'static str {
    let lower = report.to_ascii_lowercase();
    match lower.find("verdict:") {
        Some(at) => {
            let rest = lower[at + "verdict:".len()..].trim_start().trim_start_matches('*');
            if rest.starts_with("passed") || rest.starts_with("pass") {
                "passed"
            } else if rest.starts_with("failed") || rest.starts_with("fail") {
                "failed"
            } else {
                "unclear"
            }
        }
        None => "unclear",
    }
}

// ------------------------------------------------------------------ a board finished

/// A board closed `proposed_solved`, or `solved` by its test: its write-up is
/// due. What the closing call's answer says about it. Closed by a subagent's
/// test, s0ma is told too (it may be waiting on something else).
pub fn finished(boards: &BoardStore, board: &Board, by_subagent: bool) -> Result<Value> {
    let flows = FlowStore::beside(boards);
    let mut flow = flows.get_or_new(board)?;
    if flow.stage() != Stage::Working {
        return Ok(Value::Null);
    }
    flow.name = board.name.clone();
    flows.put(&flow)?;
    let say = if board.status == BoardStatus::ProposedExhausted {
        format!(
            "Write it up for the operator with board_report: what was tried, what each attempt ruled out, why \
             nothing is left to try, and the files that show it (absolute paths on your machine). They are \
             copied to /share/{}/ and it goes to his review: he closes it exhausted, or sends it back.",
            folder(&board.name)
        )
    } else {
        format!(
            "Write it up for the operator with board_report: what was done, how it was checked, what is left or \
             uncertain, and the files that make up the work (absolute paths on your machine). They are copied \
             to /share/{}/ and it goes to his review.",
            folder(&board.name)
        )
    };
    if by_subagent {
        flows.push_event(&format!("'{}' was closed solved by its goal test, run by a subagent. {say}", board.name), true)?;
    }
    Ok(json!(say))
}

// ------------------------------------------------------------------ s0ma's tools

pub const TOOLS: [&str; 4] = ["board_report", "board_answer", "board_verify", "board_verification"];

pub fn handles(name: &str) -> bool {
    TOOLS.contains(&name)
}

pub fn tool_definitions() -> Vec<ToolDefinition> {
    let name = json!({"type": "string", "description": "The board's name."});
    vec![
        ToolDefinition::function(
            "board_report",
            "Write up a finished board for the operator's review: after board_close proposed_solved, or a passing \
             board_test. The write-up is markdown: what was done, how it was checked, what is left or uncertain. \
             `files` are the absolute paths on your machine of the files that make up the work (code, configs, \
             outputs, logs; folders too): they are copied, as they are now, into /share/boards/<name>/files/, \
             beside report.md, and the board goes to the operator's review in his projects tab. Each round's report is \
             kept too, as reports/round-<n>.md. When he sends a board back with notes, answer every one \
             (board_answer) before writing it up again: the new report and files replace the old in report.md and \
             files/.",
            json!({"type": "object", "additionalProperties": false, "required": ["name", "report"], "properties": {
                "name": name,
                "report": {"type": "string", "description": "The write-up, in markdown."},
                "files": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_FILES,
                    "description": "Absolute paths on your machine."}
            }}),
        ),
        ToolDefinition::function(
            "board_answer",
            "Answer one of the operator's review notes on a board he sent back to you (board_read shows them under \
             `review`): what he asked, or what you did with what he suggested and why. Every note needs an answer \
             before the board can be written up again.",
            json!({"type": "object", "additionalProperties": false, "required": ["name", "note_id", "answer"], "properties": {
                "name": name,
                "note_id": {"type": "string", "description": "e.g. n1."},
                "answer": {"type": "string"}
            }}),
        ),
        ToolDefinition::function(
            "board_verify",
            "Your own check of a board the operator sent to verification, made independently: from the goal, the test \
             and the delivered files in /share/boards/<name>/, not from memory of having done it. Run the test, \
             read the files, try the claims. A fresh subagent checks it too; its report is held from you until \
             you file this, and comes back in this call's answer if it is done.",
            json!({"type": "object", "additionalProperties": false, "required": ["name", "verdict", "findings"], "properties": {
                "name": name,
                "verdict": {"type": "string", "enum": ["passed", "failed"]},
                "findings": {"type": "string", "description": "What you checked, how, and what you found, claim by claim."}
            }}),
        ),
        ToolDefinition::function(
            "board_verification",
            "The verification report for the operator, reconciling your own check (board_verify) with the independent \
             verifier's: where they agree, where they differ and which is right, and the verdict. It goes to \
             the operator, who accepts the board as delivered or sends it back to work.",
            json!({"type": "object", "additionalProperties": false, "required": ["name", "verdict", "report"], "properties": {
                "name": name,
                "verdict": {"type": "string", "enum": ["passed", "failed"]},
                "report": {"type": "string", "description": "The reconciled report, in markdown."}
            }}),
        ),
    ]
}

fn rejected(error: impl Into<String>) -> String {
    crate::board::rejected(error)
}

/// One of s0ma's workflow tools, on s0ma's machine through `run`, with the
/// shared folder at `share` (`/share`).
pub fn execute(boards: &BoardStore, call: &ToolCall, share: &str, run: Run) -> Result<String> {
    let flows = FlowStore::beside(boards);
    let arguments: Value = match serde_json::from_str(&call.function.arguments) {
        Ok(value) => value,
        Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
    };
    let text = |key: &str| arguments.get(key).and_then(Value::as_str).unwrap_or_default().to_string();
    let Some(board) = boards.board(&text("name"))? else {
        return Ok(rejected(format!("no board named '{}' (see board_list)", text("name"))));
    };
    let mut flow = flows.get_or_new(&board)?;
    match call.function.name.as_str() {
        "board_report" => {
            if !matches!(board.status, BoardStatus::Solved | BoardStatus::ProposedSolved | BoardStatus::ProposedExhausted) {
                return Ok(rejected(format!(
                    "'{}' is {}: close it proposed_solved or proposed_exhausted (or pass its board_test) before writing it up",
                    board.name,
                    board.status.label()
                )));
            }
            if flow.stage() != Stage::Working {
                return Ok(rejected(format!("'{}' is in the operator's {} already", board.name, flow.stage().label())));
            }
            let notes = flows.notes(&board.id)?;
            let open: Vec<String> = notes.iter().filter(|note| note.answer.is_none()).map(|note| note.id.clone()).collect();
            if !open.is_empty() {
                return Ok(rejected(format!("answer the operator's notes first (board_answer): {}", open.join(", "))));
            }
            let report = match bounded(&text("report"), 40, MAX_REPORT_CHARS, "report") {
                Ok(report) => report,
                Err(error) => return Ok(rejected(error)),
            };
            let paths: Vec<String> = arguments
                .get("files")
                .and_then(Value::as_array)
                .map(|files| files.iter().filter_map(Value::as_str).map(str::trim).map(str::to_string).collect())
                .unwrap_or_default();
            if paths.len() > MAX_FILES {
                return Ok(rejected(format!("at most {MAX_FILES} files")));
            }
            if let Some(path) = paths.iter().find(|path| !path.starts_with('/') || path.split('/').any(|part| part == "..")) {
                return Ok(rejected(format!("'{path}' is not an absolute path without '..'")));
            }
            let files = match snapshot(share, &board.name, &paths, run) {
                Ok(files) => files,
                Err(error) => return Ok(rejected(format!("the files could not be copied: {error}"))),
            };
            let round_file = format!("reports/round-{}.md", flow.round);
            let written = put_text(share, &board.name, "report.md", &report, run)
                .and_then(|()| put_text(share, &board.name, &round_file, &report, run))
                .and_then(|()| put_text(share, &board.name, "review.md", &review_text(&board.name, &notes), run));
            if let Err(error) = written {
                return Ok(rejected(format!("the report could not be written to /share: {error}")));
            }
            flow.name = board.name.clone();
            // The rounds before, from a flow older than the list of them.
            if flow.reports.is_empty() {
                flow.reports = flow.all_reports();
            }
            let at = now();
            flow.reports.retain(|kept| kept.round != flow.round);
            flow.reports.push(RoundReport { round: flow.round, text: report.clone(), at: at.clone() });
            flow.report = Some(report);
            flow.report_round = flow.round;
            flow.report_at = Some(at);
            flow.files = files;
            flow.stage = Some(Stage::Review);
            flow.entered_at = now();
            flows.put(&flow)?;
            Ok(json!({
                "ok": true,
                "stage": "review",
                "folder": format!("/share/{}", folder(&board.name)),
                "files_copied": flow.files.len(),
                "bytes": flow.files.iter().map(|file| file.bytes).sum::<u64>(),
                "note": (if board.status == BoardStatus::ProposedExhausted {
                    "It is in the operator's review now, in his projects tab. He closes it exhausted, or sends it \
                     back to you with questions and ideas."
                } else {
                    "It is in the operator's review now, in his projects tab. He may send it back to you with \
                     questions and ideas, or on to verification."
                }),
            })
            .to_string())
        }

        "board_answer" => {
            if flow.stage() != Stage::Working {
                return Ok(rejected(format!("'{}' is in the operator's {}: nothing to answer now", board.name, flow.stage().label())));
            }
            let answer = match bounded(&text("answer"), 2, MAX_ANSWER_CHARS, "answer") {
                Ok(answer) => answer,
                Err(error) => return Ok(rejected(error)),
            };
            let id = text("note_id");
            if !flows.answer(&board.id, id.trim(), &answer)? {
                return Ok(rejected(format!("no note '{}' on this board", id.trim())));
            }
            let open: Vec<String> =
                flows.notes(&board.id)?.into_iter().filter(|note| note.answer.is_none()).map(|note| note.id).collect();
            Ok(json!({"ok": true, "unanswered": open,
                "note": if open.is_empty() { "All of the operator's notes are answered. When the work is done, close the board and write it up again (board_report)." } else { "Answer the rest too." }})
            .to_string())
        }

        "board_verify" => {
            if flow.stage() != Stage::Verification {
                return Ok(rejected(format!("'{}' is not in verification (it is {})", board.name, flow.stage().label())));
            }
            if flow.own_report.is_some() {
                return Ok(rejected("you filed your own check already; reconcile with board_verification"));
            }
            let findings = match bounded(&text("findings"), 40, MAX_VERIFY_CHARS, "findings") {
                Ok(findings) => findings,
                Err(error) => return Ok(rejected(error)),
            };
            let verdict = text("verdict");
            if !matches!(verdict.as_str(), "passed" | "failed") {
                return Ok(rejected("verdict is passed or failed"));
            }
            flow.own_verdict = Some(verdict);
            flow.own_report = Some(findings);
            flows.put(&flow)?;
            let mut answer = json!({"ok": true});
            match (flow.verifier_state.as_deref(), &flow.verifier_report) {
                (Some("done" | "failed"), report) => {
                    answer["verifier"] = json!({
                        "state": flow.verifier_state,
                        "verdict": verdict_of(report.as_deref().unwrap_or("")),
                        "report": report.as_deref().or(flow.verifier_note.as_deref()).unwrap_or(""),
                    });
                    answer["note"] = json!("Reconcile the two with board_verification.");
                }
                _ => {
                    answer["verifier"] = json!({"state": flow.verifier_state, "note": flow.verifier_note});
                    answer["note"] = json!("The verifier is still at it; its report comes to you when it finishes. Then reconcile with board_verification.");
                }
            }
            Ok(answer.to_string())
        }

        "board_verification" => {
            if flow.stage() != Stage::Verification {
                return Ok(rejected(format!("'{}' is not in verification (it is {})", board.name, flow.stage().label())));
            }
            if flow.own_report.is_none() {
                return Ok(rejected("file your own check first (board_verify)"));
            }
            if !matches!(flow.verifier_state.as_deref(), Some("done" | "failed")) {
                return Ok(rejected("the independent verifier has not finished; reconcile when its report comes"));
            }
            let report = match bounded(&text("report"), 40, MAX_VERIFY_CHARS, "report") {
                Ok(report) => report,
                Err(error) => return Ok(rejected(error)),
            };
            let verdict = text("verdict");
            if !matches!(verdict.as_str(), "passed" | "failed") {
                return Ok(rejected("verdict is passed or failed"));
            }
            flow.verdict = Some(verdict);
            flow.verification = Some(report);
            flow.verified_at = Some(now());
            if let Err(error) = put_text(share, &board.name, "verification.md", &verification_text(&board.name, &flow), run) {
                return Ok(rejected(format!("the report could not be written to /share: {error}")));
            }
            flows.put(&flow)?;
            Ok(json!({"ok": true, "note": "It is with the operator: he accepts the board as delivered, or sends it back to work."}).to_string())
        }

        other => Err(format!("unknown board workflow tool '{other}'").into()),
    }
}

/// What `board_read` adds for a board in the workflow: its stage, and the operator's
/// notes with their answers.
pub fn read_extra(boards: &BoardStore, board: &Board) -> Result<Option<Value>> {
    let flows = FlowStore::beside(boards);
    let Some(flow) = flows.get(&board.id)? else { return Ok(None) };
    let notes = flows.notes(&board.id)?;
    Ok(Some(json!({
        "stage": flow.stage().label(),
        "round": flow.round,
        "write_up_due": flow.stage() == Stage::Working && !flow.reported()
            && matches!(board.status, BoardStatus::Solved | BoardStatus::ProposedSolved | BoardStatus::ProposedExhausted),
        "folder": format!("/share/{}", folder(&board.name)),
        "review": notes.iter().map(|note| json!({"id": note.id, "round": note.round,
            "from_operator": note.text, "your_answer": note.answer})).collect::<Vec<_>>(),
    })))
}

// ------------------------------------------------------------------ The operator's moves

/// A move's two writes, in the order that keeps them together: the flow
/// first (refused whole if another write came between: `Changed`), then the
/// board's side (`board.sqlite3`); if that fails, the flow is put back as it
/// was, so neither store is left a move ahead of the other.
fn commit(flows: &FlowStore, before: &Flow, after: &Flow, board_side: impl FnOnce() -> Result<()>) -> Result<()> {
    flows.put(after)?;
    if let Err(error) = board_side() {
        let back = Flow { rev: 0, ..before.clone() };
        if let Err(restore) = flows.put(&back) {
            return Err(format!("{error}; and its workflow could not be put back: {restore}").into());
        }
        return Err(error);
    }
    Ok(())
}

/// Which board, by id (the tab's) or name (a command's).
fn find(boards: &BoardStore, which: &str) -> Result<Option<Board>> {
    Ok(match boards.board_by_id(which)? {
        Some(board) => Some(board),
        None => boards.board(which)?,
    })
}

/// The operator sends a board in review, or in verification, back to s0ma: open
/// again, in a new round, and s0ma starts on his notes.
pub fn send_back(boards: &BoardStore, which: &str, why: &str) -> Result<std::result::Result<String, String>> {
    let flows = FlowStore::beside(boards);
    let Some(board) = find(boards, which)? else { return Ok(Err(format!("no board '{which}'"))) };
    let Some(mut flow) = flows.get(&board.id)? else { return Ok(Err(format!("'{}' is not in review, verification or complete", board.name))) };
    let from = flow.stage();
    if !matches!(from, Stage::Review | Stage::Verification | Stage::Deliverable) {
        return Ok(Err(format!("'{}' is {}, not in review, verification or complete", board.name, from.label())));
    }
    if let Some(job) = flow.verifier_job.as_deref().filter(|_| flow.verifier_state.as_deref() == Some("running")) {
        let _ = crate::subagent::kill(job);
    }
    let notes: Vec<Note> = flows.notes(&board.id)?.into_iter().filter(|note| note.answer.is_none()).collect();
    let why = why.trim();
    let event = match from {
        Stage::Review => {
            let listed: Vec<String> = notes.iter().map(|note| format!("  {}: {}", note.id, note.text)).collect();
            format!(
                "The operator reviewed '{}' (round {}) and sent it back to you{}. It is open again.{}\n\
                 Answer each of his notes with board_answer (board_read shows them), work on what they ask, \
                 then close it and write it up again with board_report: it goes back to his review.",
                board.name,
                flow.round,
                if notes.is_empty() { String::new() } else { format!(" with {} note{}", notes.len(), if notes.len() == 1 { "" } else { "s" }) },
                if listed.is_empty() { String::new() } else { format!("\n{}", listed.join("\n")) },
            )
        }
        Stage::Deliverable => format!(
            "The operator reopened '{}' from his completed projects{}. It is open again on its board; its last report \
             is /share/{}/report.md. Work on it, then close it and write it up again (board_report): it goes \
             back to his review.",
            board.name,
            if why.is_empty() { String::new() } else { format!(": {why}") },
            folder(&board.name),
        ),
        _ => format!(
            "The operator sent '{}' back from verification{}. It is open again; the verification report is at \
             /share/{}/verification.md{}. Work on what it found, then close it and write it up again \
             (board_report): it goes back to his review.",
            board.name,
            if why.is_empty() { String::new() } else { format!(": {why}") },
            folder(&board.name),
            if flow.verification.is_none() { " (not finished; the verifier was stopped)" } else { "" },
        ),
    };
    let before = flow.clone();
    flow.round += 1;
    flow.stage = Some(Stage::Working);
    flow.entered_at = now();
    flow.verifier_job = None;
    flow.verifier_state = None;
    flow.accepted_at = None;
    commit(&flows, &before, &flow, || {
        if from != Stage::Review && !why.is_empty() {
            let whence = if from == Stage::Deliverable { "complete" } else { "verification" };
            boards.add_hint(&board.id, &format!("Sent back from {whence}: {why}"))?;
        }
        boards.reopen(&board.id)
    })?;
    flows.push_event(&event, true)?;
    Ok(Ok(format!("{} is back with s0ma, open again (round {})", board.name, flow.round)))
}

/// The operator sends a board in review on to verification: a fresh subagent is
/// started for it (`verifiers_wanted`), and s0ma checks it too.
pub fn to_verification(boards: &BoardStore, which: &str) -> Result<std::result::Result<String, String>> {
    let flows = FlowStore::beside(boards);
    let Some(board) = find(boards, which)? else { return Ok(Err(format!("no board '{which}'"))) };
    // From review; or from the deliverables, to be verified again.
    let Some(mut flow) = flows.get(&board.id)?.filter(|flow| matches!(flow.stage(), Stage::Review | Stage::Deliverable)) else {
        return Ok(Err(format!("'{}' is not in review or complete", board.name)));
    };
    if matches!(board.status, BoardStatus::ProposedExhausted | BoardStatus::Exhausted) {
        return Ok(Err(format!("'{}' is exhausted: nothing to verify; take it back to review or to its board", board.name)));
    }
    let again = flow.stage() == Stage::Deliverable;
    let unanswered = flows.notes(&board.id)?.iter().filter(|note| note.answer.is_none()).count();
    flow.accepted_at = None;
    flow.stage = Some(Stage::Verification);
    flow.entered_at = now();
    flow.verifier_job = None;
    flow.verifier_state = Some("wanted".into());
    flow.verifier_report = None;
    flow.verifier_note = None;
    flow.verifier_tried_at = None;
    flow.own_verdict = None;
    flow.own_report = None;
    flow.verdict = None;
    flow.verification = None;
    flow.verified_at = None;
    flows.put(&flow)?;
    flows.push_event(
        &format!(
            "The operator sent '{}' to verification{}{}. Check it yourself now, independently: from its goal, its test and \
             the delivered files in /share/{}/ (report.md, files/), not from memory of having done it. Run the \
             test, read the files, try what the write-up claims. Then file your check with board_verify. A fresh \
             subagent verifies it too; its report is held from you until you file yours. Then reconcile the two \
             with board_verification.",
            board.name,
            if again { " again, from his completed projects" } else { "" },
            if unanswered > 0 { format!(" with {unanswered} note{} of his still open: answer them in the report", if unanswered == 1 { "" } else { "s" }) } else { String::new() },
            folder(&board.name),
        ),
        true,
    )?;
    Ok(Ok(format!("{} is in verification: a fresh subagent and s0ma check it, each on its own", board.name)))
}

/// The operator takes a board back to review: from verification (a running
/// verifier stopped) or from the deliverables. Its write-up and files are as
/// they were; a verification is done again if he sends it on. An exhausted
/// board is s0ma's proposal again. s0ma hears with his next message.
pub fn to_review(boards: &BoardStore, which: &str, why: &str) -> Result<std::result::Result<String, String>> {
    let flows = FlowStore::beside(boards);
    let Some(board) = find(boards, which)? else { return Ok(Err(format!("no board '{which}'"))) };
    let Some(mut flow) = flows.get(&board.id)?.filter(|flow| matches!(flow.stage(), Stage::Verification | Stage::Deliverable)) else {
        return Ok(Err(format!("'{}' is not in verification or complete", board.name)));
    };
    let from = flow.stage();
    if let Some(job) = flow.verifier_job.as_deref().filter(|_| flow.verifier_state.as_deref() == Some("running")) {
        let _ = crate::subagent::kill(job);
    }
    let before = flow.clone();
    flow.stage = Some(Stage::Review);
    flow.entered_at = now();
    flow.accepted_at = None;
    flow.verifier_job = None;
    flow.verifier_state = None;
    commit(&flows, &before, &flow, || {
        if board.status == BoardStatus::Exhausted {
            boards.restate(&board.id, BoardStatus::ProposedExhausted)?;
        }
        Ok(())
    })?;
    let why = why.trim();
    let whence = if from == Stage::Deliverable { "his completed projects" } else { "verification" };
    flows.push_event(
        &format!(
            "The operator took '{}' back to his review, from {whence}{}. Nothing is asked of you yet: he may leave \
             notes, send it back to you, or on to verification.",
            board.name,
            if why.is_empty() { String::new() } else { format!(": {why}") },
        ),
        true,
    )?;
    Ok(Ok(format!("{} is in review again", board.name)))
}

/// The operator accepts a verified board: delivered.
pub fn accept(boards: &BoardStore, which: &str) -> Result<std::result::Result<String, String>> {
    let flows = FlowStore::beside(boards);
    let Some(board) = find(boards, which)? else { return Ok(Err(format!("no board '{which}'"))) };
    let Some(mut flow) = flows.get(&board.id)?.filter(|flow| flow.stage() == Stage::Verification) else {
        return Ok(Err(format!("'{}' is not in verification", board.name)));
    };
    if flow.verification.is_none() {
        return Ok(Err(format!("'{}' has no verification report yet", board.name)));
    }
    let verdict = flow.verdict.clone().unwrap_or_default();
    let before = flow.clone();
    flow.stage = Some(Stage::Deliverable);
    flow.entered_at = now();
    flow.accepted_at = Some(now());
    commit(&flows, &before, &flow, || boards.deliver(&board.id, &format!(" [verified ({verdict}); accepted by the operator]")))?;
    flows.push_event(&format!("The operator accepted '{}' as delivered (verification: {verdict}). It is closed and complete.", board.name), false)?;
    Ok(Ok(format!("{} is complete", board.name)))
}

/// The operator throws a board away, wherever it is: closed (abandoned, "discarded
/// by the operator"), out of every subtab, a running verifier stopped. s0ma hears
/// with his next message. Its folder in /share stays, for him to delete.
pub fn discard(boards: &BoardStore, which: &str) -> Result<std::result::Result<String, String>> {
    let flows = FlowStore::beside(boards);
    let Some(board) = find(boards, which)? else { return Ok(Err(format!("no board '{which}'"))) };
    let mut flow = flows.get_or_new(&board)?;
    if flow.stage() == Stage::Discarded {
        return Ok(Err(format!("'{}' is discarded already", board.name)));
    }
    if let Some(job) = flow.verifier_job.as_deref().filter(|_| flow.verifier_state.as_deref() == Some("running")) {
        let _ = crate::subagent::kill(job);
    }
    let before = flow.clone();
    flow.stage = Some(Stage::Discarded);
    flow.entered_at = now();
    flow.verifier_state = None;
    commit(&flows, &before, &flow, || boards.discard(&board.id))?;
    flows.push_event(&format!("The operator discarded '{}': it is closed (abandoned); leave it be.", board.name), false)?;
    Ok(Ok(format!("{} discarded", board.name)))
}

/// The operator agrees with s0ma that nothing is left to try: closed exhausted,
/// and its write-up kept with the deliverables (closed and complete).
/// s0ma hears with his next message.
pub fn exhaust(boards: &BoardStore, which: &str) -> Result<std::result::Result<String, String>> {
    let Some(board) = find(boards, which)? else { return Ok(Err(format!("no board '{which}'"))) };
    Ok(match boards.confirm_exhausted(&board.name)? {
        Ok(board) => {
            let flows = FlowStore::beside(boards);
            let mut flow = flows.get_or_new(&board)?;
            flow.stage = Some(Stage::Deliverable);
            flow.entered_at = now();
            flow.accepted_at = Some(now());
            flows.put(&flow)?;
            FlowStore::beside(boards).push_event(&format!("The operator closed '{}' exhausted, as you proposed.", board.name), false)?;
            Ok(format!("{} closed exhausted", board.name))
        }
        Err(error) => Err(error),
    })
}

/// The operator does not agree that nothing is left: the board is open again, his
/// reason a note on it, and s0ma goes back to it.
pub fn reopen_exhausted(boards: &BoardStore, which: &str, why: &str) -> Result<std::result::Result<String, String>> {
    let Some(board) = find(boards, which)? else { return Ok(Err(format!("no board '{which}'"))) };
    if board.status != BoardStatus::ProposedExhausted {
        return Ok(Err(format!("'{}' is {}, not proposed exhausted", board.name, board.status.label())));
    }
    Ok(match boards.reject_solved(&board.name, why)? {
        Ok(board) => {
            let why = why.trim();
            FlowStore::beside(boards).push_event(
                &format!(
                    "The operator does not think '{}' is exhausted{}: it is open again, with his note on it. Look again at what \
                     is left to try (board_read), add the directions he sees, and go on.",
                    board.name,
                    if why.is_empty() { String::new() } else { format!(" ({why})") },
                ),
                true,
            )?;
            Ok(format!("{} is open again for s0ma", board.name))
        }
        Err(error) => Err(error),
    })
}

/// `/board confirm`: the operator's word without the workflow; delivered all the
/// same, so it is not left as a write-up due.
pub fn confirmed(boards: &BoardStore, board: &Board) -> Result<()> {
    let flows = FlowStore::beside(boards);
    let mut flow = flows.get_or_new(board)?;
    flow.stage = Some(Stage::Deliverable);
    flow.entered_at = now();
    flow.accepted_at = Some(now());
    flows.put(&flow)
}

/// The operator's note on a board in review.
pub fn annotate(boards: &BoardStore, which: &str, text: &str) -> Result<std::result::Result<String, String>> {
    let flows = FlowStore::beside(boards);
    let Some(board) = find(boards, which)? else { return Ok(Err(format!("no board '{which}'"))) };
    let Some(flow) = flows.get(&board.id)?.filter(|flow| flow.stage() == Stage::Review) else {
        return Ok(Err(format!("'{}' is not in review", board.name)));
    };
    let text = match bounded(text, 1, MAX_NOTE_CHARS, "note") {
        Ok(text) => text,
        Err(error) => return Ok(Err(error)),
    };
    let note = flows.add_note(&board.id, flow.round, &text)?;
    Ok(Ok(format!("note {} added; s0ma sees it if you send the board back", note.id)))
}

/// The operator takes back one of his notes s0ma has not answered.
pub fn unannotate(boards: &BoardStore, which: &str, id: &str) -> Result<std::result::Result<String, String>> {
    let flows = FlowStore::beside(boards);
    let Some(board) = find(boards, which)? else { return Ok(Err(format!("no board '{which}'"))) };
    Ok(if flows.remove_note(&board.id, id)? { Ok(format!("{id} taken back")) } else { Err(format!("no unanswered note {id}")) })
}

/// What s0ma has not heard of yet that needs no turn (an acceptance), for
/// The operator's next message.
pub fn take_note() -> Option<String> {
    let (_, flows) = stores()?;
    let events = flows.take_events(false).ok()?;
    (!events.is_empty()).then(|| format!("\n\n[BOARDS]\n{}", events.iter().map(|event| format!("- {event}")).collect::<Vec<_>>().join("\n")))
}

/// The events that start a turn, taken once: the turn's text.
pub fn take_turn() -> Option<String> {
    let (_, flows) = stores()?;
    if !flows.has_events(true).ok()? {
        return None;
    }
    let events = flows.take_events(true).ok()?;
    (!events.is_empty()).then(|| {
        format!(
            "[BOARDS — FROM THE OPERATOR'S PROJECTS TAB, RENDERED BY THE HARNESS]\n{}",
            events.iter().map(|event| format!("- {event}")).collect::<Vec<_>>().join("\n")
        )
    })
}

// ------------------------------------------------------------------ the verifier

/// The verifier's task: the board, where the work is, what to check, and
/// the shape of its report. It knows nothing else of the work.
pub fn verifier_task(board: &Board) -> String {
    let dir = format!("/share/{}", folder(&board.name));
    format!(
        "Verify board '{name}' for the operator, independently.\n\n\
         s0ma finished this piece of work and wrote it up. You check it, on your own: do not trust the write-up, \
         check each of its claims yourself, on the machine.\n\n\
         Board: {name} — {title}\n\
         Goal: {goal}\n\
         Goal test ({kind}): {test}\n\
         The write-up: {dir}/report.md\n\
         The files it delivered, as snapshots under their paths on the machine: {dir}/files/\n\
         the operator's review notes and s0ma's answers: {dir}/review.md\n\n\
         Read the write-up. Run the goal test if it is a command. Read the delivered files and check they do what \
         the write-up says: build them, run their tests, try them. Look for what is missing, wrong, untested or \
         overstated. Change nothing outside /tmp.\n\n\
         End your reply with your verification report, in this form:\n\
         VERDICT: passed (or: VERDICT: failed)\n\
         then what you checked, how, and what you found, claim by claim, and what you could not check.",
        name = board.name,
        title = board.title,
        goal = board.goal,
        kind = board.goal_test_kind.label(),
        test = board.goal_test,
    )
}

/// Boards whose verifier should be started now (wanted, and not tried in
/// the last `VERIFIER_RETRY`), with their tasks.
pub fn verifiers_wanted() -> Vec<(String, String)> {
    let Some((boards, flows)) = stores() else { return Vec::new() };
    let Ok(all) = flows.all() else { return Vec::new() };
    all.into_iter()
        .filter(|flow| flow.stage() == Stage::Verification && flow.verifier_state.as_deref() == Some("wanted"))
        .filter(|flow| {
            flow.verifier_tried_at
                .as_deref()
                .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
                .is_none_or(|at| (Local::now() - at.with_timezone(&Local)).to_std().unwrap_or_default() >= VERIFIER_RETRY)
        })
        .filter_map(|flow| {
            let board = boards.board_by_id(&flow.board_id).ok()??;
            Some((flow.board_id.clone(), verifier_task(&board)))
        })
        .collect()
}

/// The verifier for `board_id` started as `job`, or could not (`error`).
///
/// The board may have moved on while it started (sent back, taken to
/// review, discarded): then no board wants it, and it is stopped rather than
/// left running where nothing tracks it.
pub fn verifier_started(board_id: &str, outcome: std::result::Result<String, String>) {
    let Some((_, flows)) = stores() else {
        if let Ok(job) = &outcome {
            let _ = crate::subagent::kill(job);
        }
        return;
    };
    verifier_started_in(&flows, board_id, outcome, &mut |job| {
        let _ = crate::subagent::kill(job);
    });
}

fn verifier_started_in(flows: &FlowStore, board_id: &str, outcome: std::result::Result<String, String>, stop: &mut dyn FnMut(&str)) {
    let mut wanted = false;
    let written = flows.update(board_id, |flow| {
        flow.verifier_tried_at = Some(now());
        wanted = flow.stage() == Stage::Verification && flow.verifier_state.as_deref() == Some("wanted");
        match &outcome {
            Ok(_) if !wanted => return false,
            Ok(job) => {
                flow.verifier_job = Some(job.clone());
                flow.verifier_state = Some("running".into());
                flow.verifier_note = None;
            }
            Err(error) => flow.verifier_note = Some(format!("not started yet: {error}")),
        }
        true
    });
    if let Ok(job) = &outcome
        && (!wanted || !matches!(written, Ok(Some(_))))
    {
        stop(job);
    }
    if let Err(error) = written {
        crate::terminal::warn_status(&format!("[board] the verifier of {board_id} could not be recorded: {error}"));
    }
}

/// A subagent finished. A board's verifier: its report is kept on the board
/// (`Some`), not handed to s0ma, and s0ma hears of it only once it has filed
/// its own check (`Some(Some(note))`). Any other subagent: `None`.
pub fn verifier_finished(job: &str, status: &str, result: &str, error: Option<&str>) -> Option<Option<String>> {
    let (_, flows) = stores()?;
    verifier_finished_in(&flows, job, status, result, error)
}

fn verifier_finished_in(flows: &FlowStore, job: &str, status: &str, result: &str, error: Option<&str>) -> Option<Option<String>> {
    let running = |flow: &Flow| flow.verifier_job.as_deref() == Some(job) && flow.verifier_state.as_deref() == Some("running");
    let found = flows.all().ok()?.into_iter().find(|flow| running(flow))?;
    let succeeded = status == "completed" && !result.trim().is_empty();
    // Made to the flow as it is now: a move in between (sent back, to review)
    // is not undone by this.
    let written = flows.update(&found.board_id, |flow| {
        if !running(flow) {
            return false;
        }
        flow.verifier_state = Some(if succeeded { "done" } else { "failed" }.into());
        flow.verifier_report = Some(result.chars().take(MAX_VERIFY_CHARS).collect());
        if !succeeded {
            flow.verifier_note = Some(format!("the verifier ended {status}{}", error.map(|error| format!(": {error}")).unwrap_or_default()));
        }
        true
    });
    let flow = match written {
        Ok(Some(flow)) => flow,
        // The board moved on: its report is no one's now.
        Ok(None) => return Some(None),
        Err(error) => {
            crate::terminal::warn_status(&format!("[board] the verifier {job}'s report could not be kept: {error}"));
            return Some(None);
        }
    };
    if flow.own_report.is_none() {
        return Some(None);
    }
    let report = if succeeded { result.chars().take(MAX_VERIFY_CHARS).collect::<String>() } else { flow.verifier_note.clone().unwrap_or_default() };
    Some(Some(format!(
        "[BOARD VERIFIER — ITS REPORT, NOT OPERATOR INPUT]\nThe independent verifier of '{}' finished ({status}, verdict {}). \
         Reconcile it with your own check in board_verification.\n\n{report}",
        flow.name,
        verdict_of(result),
    )))
}

/// How many boards wait for the operator (in review, or verified), for the tab bar.
pub fn waiting(flows: &FlowStore) -> usize {
    flows.all().map(|all| all.iter().filter(|flow| flow.waits_for_operator()).count()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::GoalTestKind;
    use crate::types::FunctionCall;

    pub(crate) fn temp_boards(tag: &str) -> (PathBuf, BoardStore) {
        let root = std::env::temp_dir().join(format!("s0ma-flow-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let boards = BoardStore::beside_archive(&root.join("hippocampus.jsonl"));
        (root, boards)
    }

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall { id: "c".into(), kind: "function".into(), function: FunctionCall { name: name.into(), arguments: arguments.to_string() } }
    }

    /// s0ma's machine, standing in: commands run here.
    fn local() -> impl FnMut(&str, u64) -> std::result::Result<Ran, String> {
        move |command: &str, _| {
            let out = std::process::Command::new("sh").arg("-c").arg(command).output().map_err(|error| error.to_string())?;
            Ok(Ran { exit_code: out.status.code().map(i64::from), stdout: String::from_utf8_lossy(&out.stdout).into(), stderr: String::from_utf8_lossy(&out.stderr).into() })
        }
    }

    #[test]
    fn verdicts_are_read_from_a_report() {
        assert_eq!(verdict_of("...\nVERDICT: passed\nAll good."), "passed");
        assert_eq!(verdict_of("Verdict: **failed** — the test hangs"), "failed");
        assert_eq!(verdict_of("I think it works"), "unclear");
    }

    #[test]
    fn notes_events_and_review_text() {
        let (root, boards) = temp_boards("notes");
        let board = boards.open_board("drift", "t", "o", "g", "The operator agrees", GoalTestKind::Judgment, 20).unwrap().unwrap();
        let flows = FlowStore::beside(&boards);
        let first = flows.add_note(&board.id, 1, "Why not pin it in the template?").unwrap();
        let second = flows.add_note(&board.id, 1, "Try a smaller window").unwrap();
        assert_eq!((first.id.as_str(), second.id.as_str()), ("n1", "n2"));
        assert!(flows.answer(&board.id, "n1", "The template is shared with jobs.").unwrap());
        assert!(!flows.remove_note(&board.id, "n1").unwrap(), "an answered note stays");
        let text = review_text("drift", &flows.notes(&board.id).unwrap());
        assert!(text.contains("## Round 1") && text.contains("**n1** (operator): Why not pin") && text.contains("  - s0ma: The template is shared"));
        assert!(text.contains("**n2**") && text.contains("(not answered yet)"));
        flows.push_event("one", true).unwrap();
        flows.push_event("two", false).unwrap();
        assert!(flows.has_events(true).unwrap());
        assert_eq!(flows.take_events(true).unwrap(), ["one"]);
        assert!(flows.take_events(true).unwrap().is_empty(), "taken once");
        assert_eq!(flows.take_events(false).unwrap(), ["two"]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_stale_write_is_refused_and_a_late_verifier_is_stopped() {
        let (root, boards) = temp_boards("races");
        let board = boards.open_board("race", "t", "o", "g", "The operator agrees", GoalTestKind::Judgment, 20).unwrap().unwrap();
        let flows = FlowStore::beside(&boards);
        flows.put(&Flow { board_id: board.id.clone(), name: "race".into(), stage: Some(Stage::Verification), round: 1, verifier_state: Some("wanted".into()), ..Flow::default() }).unwrap();
        // Two read it; one writes; the other's write is refused.
        let mut first = flows.get(&board.id).unwrap().unwrap();
        let mut second = flows.get(&board.id).unwrap().unwrap();
        first.stage = Some(Stage::Review);
        flows.put(&first).unwrap();
        second.verifier_state = Some("running".into());
        let refused = flows.put(&second).unwrap_err();
        assert!(refused.downcast_ref::<Changed>().is_some(), "{refused}");
        assert_eq!(flows.get(&board.id).unwrap().unwrap().stage(), Stage::Review, "the first write stands");

        // A verifier that started after the board left verification: stopped,
        // and not recorded as running.
        let mut stopped = Vec::new();
        verifier_started_in(&flows, &board.id, Ok("job-9".into()), &mut |job| stopped.push(job.to_string()));
        assert_eq!(stopped, ["job-9"]);
        let flow = flows.get(&board.id).unwrap().unwrap();
        assert_eq!((flow.verifier_job, flow.verifier_state.as_deref()), (None, Some("wanted")));

        // In verification and wanted: recorded, not stopped.
        flows.update(&board.id, |flow| { flow.stage = Some(Stage::Verification); true }).unwrap();
        let mut stopped = Vec::new();
        verifier_started_in(&flows, &board.id, Ok("job-10".into()), &mut |job| stopped.push(job.to_string()));
        assert!(stopped.is_empty());
        assert_eq!(flows.get(&board.id).unwrap().unwrap().verifier_state.as_deref(), Some("running"));
        // Sent back before it finished: its report is no one's, and the stage stays.
        flows.update(&board.id, |flow| { flow.stage = Some(Stage::Working); flow.verifier_state = None; flow.verifier_job = None; true }).unwrap();
        assert_eq!(verifier_finished_in(&flows, "job-10", "completed", "VERDICT: passed", None), None, "not running any more: not found");
        assert_eq!(flows.get(&board.id).unwrap().unwrap().stage(), Stage::Working);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_note_taken_back_never_gives_its_id_to_two_notes() {
        let (root, boards) = temp_boards("note-ids");
        let board = boards.open_board("ids", "t", "o", "g", "The operator agrees", GoalTestKind::Judgment, 20).unwrap().unwrap();
        let flows = FlowStore::beside(&boards);
        flows.add_note(&board.id, 1, "first").unwrap();
        flows.add_note(&board.id, 1, "second").unwrap();
        assert!(flows.remove_note(&board.id, "n1").unwrap());
        let third = flows.add_note(&board.id, 1, "third").unwrap();
        assert_eq!(third.id, "n3", "past the highest, not the count");
        let ids: Vec<String> = flows.notes(&board.id).unwrap().into_iter().map(|note| note.id).collect();
        assert_eq!(ids, ["n2", "n3"]);

        // Two notes that came to share an id before: the later one renumbered.
        {
            let db = Connection::open(&flows.path).unwrap();
            db.execute("INSERT INTO notes(id, board_id, round, kind, text, created_at) VALUES('n3', ?1, 1, 'note', 'duplicate', 'x')", params![board.id]).unwrap();
        }
        let notes = flows.notes(&board.id).unwrap();
        let ids: Vec<&str> = notes.iter().map(|note| note.id.as_str()).collect();
        assert_eq!(ids, ["n2", "n3", "n4"], "repaired on the next open");
        assert_eq!(notes[2].text, "duplicate", "the older keeps its id");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_board_goes_through_review_and_verification_to_delivered() {
        let (root, boards) = temp_boards("walk");
        let share = root.join("share");
        let work = root.join("work");
        std::fs::create_dir_all(work.join("src")).unwrap();
        std::fs::write(work.join("src/fix.rs"), "fn fix() {}\n").unwrap();
        std::fs::write(work.join("NOTES.md"), "notes\n").unwrap();
        std::fs::create_dir_all(&share).unwrap();
        let share_dir = share.display().to_string();
        let mut machine = local();
        let execute = |boards: &BoardStore, call: &ToolCall, machine: Run| execute(boards, call, &share_dir, machine);
        let board = boards.open_board("drift", "The self-model drifts", "o", "Know why", "The operator agrees", GoalTestKind::Judgment, 20).unwrap().unwrap();
        let flows = FlowStore::beside(&boards);
        let report = "The want-phrases re-expand because the summary template quotes them back; pinned now.";
        let files = json!([work.join("src").display().to_string(), work.join("NOTES.md").display().to_string()]);

        // Not finished: no write-up yet.
        let early = execute(&boards, &call("board_report", json!({"name": "drift", "report": report, "files": files})), &mut machine).unwrap();
        assert!(early.contains("close it proposed_solved"), "{early}");
        boards.set_status(&board.id, BoardStatus::ProposedSolved, "pinned").unwrap().unwrap();
        let said = finished(&boards, &boards.board_by_id(&board.id).unwrap().unwrap(), false).unwrap();
        assert!(said.as_str().unwrap().contains("board_report"));
        // Relative paths are refused.
        let bad = execute(&boards, &call("board_report", json!({"name": "drift", "report": report, "files": ["src/fix.rs"]})), &mut machine).unwrap();
        assert!(bad.contains("not an absolute path"), "{bad}");

        // Written up: the files copied under their paths, the report beside them.
        let done: Value = serde_json::from_str(&execute(&boards, &call("board_report", json!({"name": "drift", "report": report, "files": files})), &mut machine).unwrap()).unwrap();
        assert_eq!(done["ok"], json!(true), "{done}");
        let folder = share.join("boards/drift");
        assert_eq!(std::fs::read_to_string(folder.join("report.md")).unwrap(), report);
        assert_eq!(std::fs::read_to_string(folder.join("reports/round-1.md")).unwrap(), report, "each round's report kept");
        let copied = folder.join("files").join(work.strip_prefix("/").unwrap());
        assert_eq!(std::fs::read_to_string(copied.join("src/fix.rs")).unwrap(), "fn fix() {}\n");
        let flow = flows.get(&board.id).unwrap().unwrap();
        assert_eq!((flow.stage(), flow.files.len(), flow.reported()), (Stage::Review, 2, true));
        assert!(flow.waits_for_operator());

        // The operator's notes, and back to s0ma: open again, a new round, told why.
        assert!(annotate(&boards, &board.id, "Does it hold after two sleeps?").unwrap().is_ok());
        assert!(annotate(&boards, &board.id, "Pin it in identity.md").unwrap().is_ok());
        assert!(to_verification(&boards, "nope").unwrap().is_err());
        send_back(&boards, &board.id, "").unwrap().unwrap();
        let reopened = boards.board_by_id(&board.id).unwrap().unwrap();
        assert_eq!((reopened.status, reopened.closed_at.clone()), (BoardStatus::Open, None));
        let flow = flows.get(&board.id).unwrap().unwrap();
        assert_eq!((flow.stage(), flow.round, flow.reported()), (Stage::Working, 2, false));
        let turn = take_turn_from(&flows);
        assert!(turn.contains("sent it back to you with 2 notes") && turn.contains("n1: Does it hold"), "{turn}");
        let extra = read_extra(&boards, &reopened).unwrap().unwrap();
        assert_eq!(extra["review"][0]["from_operator"], json!("Does it hold after two sleeps?"));

        // Answered, finished again, written up again: back in review.
        boards.set_status(&board.id, BoardStatus::ProposedSolved, "pinned, held two sleeps").unwrap().unwrap();
        let blocked = execute(&boards, &call("board_report", json!({"name": "drift", "report": report})), &mut machine).unwrap();
        assert!(blocked.contains("answer the operator's notes first") && blocked.contains("n1, n2"), "{blocked}");
        execute(&boards, &call("board_answer", json!({"name": "drift", "note_id": "n1", "answer": "Yes: two sleeps, no drift."})), &mut machine).unwrap();
        let last = execute(&boards, &call("board_answer", json!({"name": "drift", "note_id": "n2", "answer": "Done; it is in identity.md now."})), &mut machine).unwrap();
        assert!(last.contains("\"unanswered\":[]"), "{last}");
        execute(&boards, &call("board_report", json!({"name": "drift", "report": format!("{report} Held for two sleeps."), "files": files})), &mut machine).unwrap();
        assert!(std::fs::read_to_string(folder.join("review.md")).unwrap().contains("s0ma: Yes: two sleeps"));
        assert_eq!(std::fs::read_to_string(folder.join("reports/round-2.md")).unwrap(), format!("{report} Held for two sleeps."));
        assert_eq!(std::fs::read_to_string(folder.join("reports/round-1.md")).unwrap(), report, "round 1's stays");
        let rounds: Vec<i64> = flows.get(&board.id).unwrap().unwrap().all_reports().iter().map(|kept| kept.round).collect();
        assert_eq!(rounds, [1, 2]);

        // To verification: a verifier wanted, s0ma asked to check on its own.
        to_verification(&boards, &board.id).unwrap().unwrap();
        assert!(take_turn_from(&flows).contains("Check it yourself now, independently"));
        let flow = flows.get(&board.id).unwrap().unwrap();
        assert_eq!(flow.verifier_state.as_deref(), Some("wanted"));
        assert!(verifier_task(&boards.board_by_id(&board.id).unwrap().unwrap()).contains("VERDICT: passed"));
        let mut flow = flow;
        flow.verifier_job = Some("job-7".into());
        flow.verifier_state = Some("running".into());
        flows.put(&flow).unwrap();
        // Reconciling before both are in is refused.
        let early = execute(&boards, &call("board_verification", json!({"name": "drift", "verdict": "passed", "report": "Both checks agree that the drift is gone for good."})), &mut machine).unwrap();
        assert!(early.contains("board_verify"), "{early}");
        let own: Value = serde_json::from_str(&execute(&boards, &call("board_verify", json!({"name": "drift", "verdict": "passed", "findings": "Ran two sleeps on a copy; the phrases stayed pinned."})), &mut machine).unwrap()).unwrap();
        assert_eq!(own["verifier"]["state"], json!("running"));
        let too_soon = execute(&boards, &call("board_verification", json!({"name": "drift", "verdict": "passed", "report": "Both checks agree that the drift is gone for good."})), &mut machine).unwrap();
        assert!(too_soon.contains("has not finished"), "the verifier's report is needed: {too_soon}");
        assert!(!flows.get(&board.id).unwrap().unwrap().waits_for_operator(), "nothing for the operator until it is reconciled");
        assert!(!accept(&boards, &board.id).unwrap().is_ok(), "nothing to accept yet");
        flows.put(&{ let mut flow = flows.get(&board.id).unwrap().unwrap(); flow.verifier_state = Some("done".into()); flow.verifier_report = Some("VERDICT: passed\nChecked identity.md and the template.".into()); flow }).unwrap();
        execute(&boards, &call("board_verification", json!({"name": "drift", "verdict": "passed", "report": "Both checks pass: the drift is gone, pinned in identity.md."})), &mut machine).unwrap();
        let kept = std::fs::read_to_string(folder.join("verification.md")).unwrap();
        assert!(kept.contains("Verdict: **passed**") && kept.contains("## The independent verifier (passed)") && kept.contains("Ran two sleeps"), "{kept}");
        assert!(flows.get(&board.id).unwrap().unwrap().waits_for_operator());

        // Accepted: delivered, solved, s0ma told with the next message.
        accept(&boards, &board.id).unwrap().unwrap();
        let delivered = boards.board_by_id(&board.id).unwrap().unwrap();
        assert_eq!(delivered.status, BoardStatus::Solved);
        assert!(delivered.outcome.unwrap().ends_with("[verified (passed); accepted by the operator]"));
        assert_eq!(flows.get(&board.id).unwrap().unwrap().stage(), Stage::Deliverable);
        assert!(flows.take_events(false).unwrap()[0].contains("accepted 'drift' as delivered"));

        // Back from the deliverables, to any stage before: verified again,
        // reviewed again, worked on again.
        to_verification(&boards, &board.id).unwrap().unwrap();
        let again = flows.get(&board.id).unwrap().unwrap();
        assert_eq!(again.stage(), Stage::Verification);
        assert!(again.verification.is_none() && again.accepted_at.is_none() && again.verifier_state.as_deref() == Some("wanted"), "a fresh verification");
        assert!(take_turn_from(&flows).contains("again, from his completed projects"));
        to_review(&boards, &board.id, "one more look at the template").unwrap().unwrap();
        assert_eq!(flows.get(&board.id).unwrap().unwrap().stage(), Stage::Review);
        assert!(take_turn_from(&flows).contains("back to his review, from verification: one more look"));
        to_verification(&boards, &board.id).unwrap().unwrap();
        flows.put(&{ let mut flow = flows.get(&board.id).unwrap().unwrap(); flow.verification = Some("ok".into()); flow }).unwrap();
        accept(&boards, &board.id).unwrap().unwrap();
        to_review(&boards, &board.id, "").unwrap().unwrap();
        assert!(take_turn_from(&flows).contains("from his completed projects"));
        assert!(to_review(&boards, &board.id, "").unwrap().is_err(), "already in review");
        // To the board: open again, a new round, why kept as a hint.
        to_verification(&boards, &board.id).unwrap().unwrap();
        flows.put(&{ let mut flow = flows.get(&board.id).unwrap().unwrap(); flow.verification = Some("ok".into()); flow }).unwrap();
        accept(&boards, &board.id).unwrap().unwrap();
        let round = flows.get(&board.id).unwrap().unwrap().round;
        send_back(&boards, &board.id, "the drift is back").unwrap().unwrap();
        let reopened = flows.get(&board.id).unwrap().unwrap();
        assert_eq!((reopened.stage(), reopened.round), (Stage::Working, round + 1));
        assert_eq!(boards.board_by_id(&board.id).unwrap().unwrap().status, BoardStatus::Open);
        assert!(take_turn_from(&flows).contains("reopened 'drift' from his completed projects: the drift is back"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_exhausted_board_goes_back_to_review_as_a_proposal() {
        let (root, boards) = temp_boards("exhausted-back");
        let flows = FlowStore::beside(&boards);
        let board = boards.open_board("leak", "t", "o", "g", "make leakcheck", GoalTestKind::Command, 20).unwrap().unwrap();
        boards.set_status(&board.id, BoardStatus::ProposedExhausted, "nothing left").unwrap().unwrap();
        exhaust(&boards, &board.id).unwrap().unwrap();
        assert_eq!(flows.get(&board.id).unwrap().unwrap().stage(), Stage::Deliverable);
        assert!(to_verification(&boards, &board.id).unwrap().unwrap_err().contains("nothing to verify"));
        to_review(&boards, &board.id, "").unwrap().unwrap();
        assert_eq!(boards.board_by_id(&board.id).unwrap().unwrap().status, BoardStatus::ProposedExhausted, "s0ma's proposal again");
        assert_eq!(flows.get(&board.id).unwrap().unwrap().stage(), Stage::Review);
        exhaust(&boards, &board.id).unwrap().unwrap();
        assert_eq!(boards.board_by_id(&board.id).unwrap().unwrap().status, BoardStatus::Exhausted);
        let _ = std::fs::remove_dir_all(&root);
    }

    fn take_turn_from(flows: &FlowStore) -> String {
        flows.take_events(true).unwrap().join("\n")
    }

    #[test]
    fn operator_discards_a_board_from_any_stage() {
        let (root, boards) = temp_boards("discard");
        let flows = FlowStore::beside(&boards);
        let open = boards.open_board("open-one", "t", "o", "g", "The operator agrees", GoalTestKind::Judgment, 20).unwrap().unwrap();
        let reviewed = boards.open_board("reviewed", "t", "o", "g", "The operator agrees", GoalTestKind::Judgment, 20).unwrap().unwrap();
        boards.set_status(&reviewed.id, BoardStatus::ProposedSolved, "done").unwrap().unwrap();
        flows.put(&Flow { board_id: reviewed.id.clone(), name: "reviewed".into(), stage: Some(Stage::Review), round: 1, ..Flow::default() }).unwrap();
        for board in [&open, &reviewed] {
            assert!(discard(&boards, &board.id).unwrap().is_ok());
            let now = boards.board_by_id(&board.id).unwrap().unwrap();
            assert_eq!(now.status, BoardStatus::Abandoned);
            assert!(now.closed_at.is_some() && now.outcome.unwrap().ends_with("[discarded by the operator]"));
            assert_eq!(flows.get(&board.id).unwrap().unwrap().stage(), Stage::Discarded);
        }
        assert!(discard(&boards, &open.id).unwrap().is_err(), "once");
        let told = flows.take_events(false).unwrap();
        assert!(told.len() == 2 && told[0].contains("The operator discarded 'open-one'"), "{told:?}");
        assert!(flows.take_events(true).unwrap().is_empty(), "no turn for it");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn operator_closes_a_proposed_exhaustion_or_sends_s0ma_back_to_it() {
        let (root, boards) = temp_boards("exhaust");
        let flows = FlowStore::beside(&boards);
        let board = boards.open_board("leak", "t", "o", "g", "make leakcheck", GoalTestKind::Command, 20).unwrap().unwrap();
        assert!(exhaust(&boards, &board.id).unwrap().is_err(), "only a proposal");
        boards.set_status(&board.id, BoardStatus::ProposedExhausted, "nothing left").unwrap().unwrap();
        reopen_exhausted(&boards, &board.id, "the allocator was never looked at").unwrap().unwrap();
        assert_eq!(boards.board_by_id(&board.id).unwrap().unwrap().status, BoardStatus::Open);
        let turn = flows.take_events(true).unwrap().join("\n");
        assert!(turn.contains("does not think 'leak' is exhausted (the allocator was never looked at)"), "s0ma goes back to it: {turn}");
        boards.set_status(&board.id, BoardStatus::ProposedExhausted, "nothing left, really").unwrap().unwrap();
        exhaust(&boards, &board.id).unwrap().unwrap();
        assert_eq!(boards.board_by_id(&board.id).unwrap().unwrap().status, BoardStatus::Exhausted);
        assert!(flows.take_events(false).unwrap()[0].contains("closed 'leak' exhausted"));
        assert!(flows.take_events(true).unwrap().is_empty(), "no turn for it");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_proposed_exhaustion_is_written_up_and_closed_from_review() {
        let (root, boards) = temp_boards("exhaust-review");
        let share = root.join("share");
        std::fs::create_dir_all(&share).unwrap();
        let share_dir = share.display().to_string();
        let mut machine = local();
        let flows = FlowStore::beside(&boards);
        let board = boards.open_board("leak", "t", "o", "g", "make leakcheck", GoalTestKind::Command, 20).unwrap().unwrap();
        boards.set_status(&board.id, BoardStatus::ProposedExhausted, "nothing left").unwrap().unwrap();
        let closed = boards.board_by_id(&board.id).unwrap().unwrap();
        let said = finished(&boards, &closed, false).unwrap();
        assert!(said.as_str().unwrap().contains("what was tried"), "{said}");
        assert_eq!(read_extra(&boards, &closed).unwrap().unwrap()["write_up_due"], json!(true));
        let report = "Tried the allocator, the pool and the cache: each ruled out by a fact; nothing is left.";
        let done = execute(&boards, &call("board_report", json!({"name": "leak", "report": report, "files": []})), &share_dir, &mut machine).unwrap();
        assert!(done.contains("closes it exhausted"), "{done}");
        assert_eq!(flows.get(&board.id).unwrap().unwrap().stage(), Stage::Review);
        // Nothing to verify; closed exhausted from review, it joins the deliverables.
        assert!(to_verification(&boards, &board.id).unwrap().unwrap_err().contains("nothing to verify"));
        exhaust(&boards, &board.id).unwrap().unwrap();
        assert_eq!(boards.board_by_id(&board.id).unwrap().unwrap().status, BoardStatus::Exhausted);
        assert_eq!(flows.get(&board.id).unwrap().unwrap().stage(), Stage::Deliverable);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_verifiers_report_is_held_until_s0ma_files_its_own() {
        let (root, boards) = temp_boards("held");
        let board = boards.open_board("held", "t", "o", "g", "The operator agrees", GoalTestKind::Judgment, 20).unwrap().unwrap();
        let flows = FlowStore::beside(&boards);
        let running = |own: Option<&str>| Flow { board_id: board.id.clone(), name: "held".into(), stage: Some(Stage::Verification), round: 1,
            verifier_job: Some("job-held".into()), verifier_state: Some("running".into()), own_report: own.map(String::from), ..Flow::default() };
        flows.put(&running(None)).unwrap();
        assert_eq!(verifier_finished_in(&flows, "another-job", "completed", "x", None), None, "not a verifier");
        // Held: s0ma has not filed its own. Kept on the board.
        assert_eq!(verifier_finished_in(&flows, "job-held", "completed", "VERDICT: failed\nThe test hangs.", None), Some(None));
        let flow = flows.get(&board.id).unwrap().unwrap();
        assert_eq!((flow.verifier_state.as_deref(), flow.verifier_report.as_deref()), (Some("done"), Some("VERDICT: failed\nThe test hangs.")));
        assert_eq!(verifier_finished_in(&flows, "job-held", "completed", "again", None), None, "once");
        // s0ma's is in: it hears the verifier's report.
        flows.put(&running(Some("mine"))).unwrap();
        let note = verifier_finished_in(&flows, "job-held", "completed", "VERDICT: failed\nThe test hangs.", None).unwrap().unwrap();
        assert!(note.contains("verdict failed") && note.contains("The test hangs."), "{note}");
        // One that failed says so.
        flows.put(&running(Some("mine"))).unwrap();
        let note = verifier_finished_in(&flows, "job-held", "cancelled", "", Some("stopped")).unwrap().unwrap();
        assert!(note.contains("the verifier ended cancelled: stopped"), "{note}");
        assert_eq!(flows.get(&board.id).unwrap().unwrap().verifier_state.as_deref(), Some("failed"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
