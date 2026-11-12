//! Boards: a search structure for problems with an unknown path.
//!
//! Some problems have a known origin, a testable goal, and no plannable route
//! between them — an intermittent failure in a large system, a regression hiding
//! among a thousand candidates, a number that has to cross a threshold. The plan
//! is discovered by searching, and what makes that hard for an agent is not
//! reasoning but bookkeeping: remembering what has been tried, what it ruled
//! out, what it opened up, and which open direction is worth taking next.
//! Without that, a long run re-treads explored ground and loses the frontier.
//!
//! None of the existing stores is this one. `threads` holds purpose and open
//! questions, but a question is not a claimable unit of work and a thread has no
//! goal test. `world_model` holds what is believed, not what is unexplored.
//! `open_loops` tracks commitments, not branches, and has no parent/child
//! structure. Episodes are written at sleep and are historical. The gap is a
//! representation of *a direction worth trying that nobody has tried yet*, with
//! enough structure to claim it, rule it out, and connect it to what produced it.
//!
//! Four primitives. A **board** is the unit of search: named, with an
//! origin, a goal, and — load-bearing — an explicit goal *test*. A **Fact** is
//! something an exploration actually showed, carrying archive evidence validated
//! on write rather than asserted. An **Intent** is a declared but unexplored
//! direction; the frontier is the set of Intents still open. A **Hint** is
//! operator judgment: a note the operator leaves on the board at any time, never
//! written by the model. It stays on the board until he takes it off, and
//! every `board_read` shows it, marked `new` the first time the agent reads it.
//!
//! Two invariants do most of the work and neither is negotiable.
//!
//! The first is that a Fact cites the archive. Evidence IDs go through
//! `EvidenceSource` before a Fact is accepted, exactly as `open_loops` does, so a
//! Fact can never be a thing the model simply said.
//!
//! The second is that the model does not score its own search. An agent allowed
//! to decide whether it has succeeded will eventually decide yes, and every Fact
//! it gathered was then scored against a judgment it authored. So `board_test`
//! runs the test in the harness and writes the verdict itself; the model may ask
//! for the test and never reports its result, and nothing in the tool surface
//! can set a board `solved`. Where no mechanical test exists the board reaches
//! `proposed_solved` and waits for the operator.
//!
//! Storage is one sqlite database beside the archive, in the derived-store style
//! `world_model` and the keyword index already use: WAL, busy timeout, and a
//! schema row so a layout change rebuilds rather than corrupting. Board writes
//! also travel the normal tool-result path into the archive, so a search is
//! replayable from the authoritative record even if this database is deleted.

pub(crate) mod flow;

use crate::evidence::{EventSnapshot, EvidenceSource};
use crate::llm::ToolDefinition;
use crate::types::{Message, ToolCall};

use chrono::{DateTime, Local};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::Deserialize;
use serde_json::{Value, json};

use std::collections::BTreeSet;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Duration;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// ------------------------------------------------------------------ constants

/// Bumped whenever the table layout changes. A mismatch moves the old database
/// aside and starts fresh rather than half-migrating it.
const SCHEMA: &str = "board.v1";

/// A claim this old is treated as open again. A worker that dies holding an
/// Intent would otherwise freeze that branch for the life of the board.
pub const CLAIM_TTL_SECS: i64 = 30 * 60;

/// Steps a board gets unless one is chosen. Every Explore — a Fact written, a
/// goal test run — spends one.
pub const DEFAULT_STEP_BUDGET: i64 = 60;
const MAX_STEP_BUDGET: i64 = 1_000;

/// The frontier has to stay readable in full, because reading it in full is what
/// stands in for a search policy. Over the cap, declaring is refused until
/// something is explored or killed.
pub const MAX_FRONTIER: usize = 40;

/// Facts are the only thing that grows without limit, so a read returns the
/// whole frontier and a slice of Facts.
const DEFAULT_FACT_PAGE: usize = 20;
const MAX_FACT_PAGE: usize = 100;

/// Two Intents whose wording overlaps this much get a warning, never a refusal.
/// Judging sameness is the model's job; noticing the resemblance is cheap.
const SIMILARITY_WARN: f64 = 0.6;

const MAX_NAME_CHARS: usize = 64;
const MAX_TITLE_CHARS: usize = 200;
const MAX_ORIGIN_CHARS: usize = 4_000;
const MAX_GOAL_CHARS: usize = 2_000;
const MAX_GOAL_TEST_CHARS: usize = 4_000;
const MAX_DIRECTION_CHARS: usize = 1_000;
const MAX_RATIONALE_CHARS: usize = 2_000;
const MAX_STATEMENT_CHARS: usize = 2_000;
const MAX_HINT_CHARS: usize = 2_000;
const MAX_NOTE_CHARS: usize = 1_000;
const MIN_TEXT_CHARS: usize = 8;
pub(crate) const MAX_EVIDENCE_IDS: usize = 20;
const MAX_INTENTS_PER_CALL: usize = 8;

/// How long a `command` goal test may run before the harness gives up on it.
const TEST_TIMEOUT_SECS: u64 = 600;

/// Render budget for the standing one-line-per-board block.
pub const RENDER_BUDGET_TOKENS: usize = 300;
const BYTES_PER_TOKEN: usize = 4;
const MAX_RENDERED_BOARDS: usize = 5;

// ------------------------------------------------------------------ enums

macro_rules! labelled {
    ($name:ident, $what:expr, { $($variant:ident => $label:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $($variant),+ }

        impl $name {
            pub fn label(self) -> &'static str {
                match self { $(Self::$variant => $label),+ }
            }

            pub fn parse(value: &str) -> std::result::Result<Self, String> {
                match value {
                    $($label => Ok(Self::$variant),)+
                    other => Err(format!("unknown {} '{}'", $what, other)),
                }
            }
        }
    };
}

labelled!(BoardStatus, "board status", {
    Open => "open",
    Solved => "solved",
    ProposedSolved => "proposed_solved",
    // The agent's word that nothing is left to try; the operator's makes it exhausted.
    ProposedExhausted => "proposed_exhausted",
    Exhausted => "exhausted",
    Suspended => "suspended",
    Abandoned => "abandoned",
});

impl BoardStatus {
    /// Whether the board still accepts search writes. `suspended` deliberately
    /// does not: the graph is intact and the operator decides whether to extend it.
    fn accepts_work(self) -> bool {
        self == Self::Open
    }

    fn is_closed(self) -> bool {
        !matches!(self, Self::Open | Self::Suspended)
    }
}

labelled!(GoalTestKind, "goal test kind", {
    Command => "command",
    Observation => "observation",
    Judgment => "judgment",
});

labelled!(Polarity, "polarity", {
    Supports => "supports",
    RulesOut => "rules_out",
});

labelled!(Basis, "basis", {
    Observed => "observed",
    Inferred => "inferred",
});

labelled!(IntentState, "intent state", {
    Open => "open",
    Claimed => "claimed",
    Explored => "explored",
    Dead => "dead",
});

labelled!(LinkRelation, "link relation", {
    Background => "background",
    Updates => "updates",
});

labelled!(FactAuthor, "fact author", {
    Model => "model",
    Harness => "harness",
    Operator => "operator",
});

// ------------------------------------------------------------------ records

#[derive(Debug, Clone)]
pub struct Board {
    pub id: String,
    pub name: String,
    pub title: String,
    pub origin: String,
    pub goal: String,
    pub goal_test: String,
    pub goal_test_kind: GoalTestKind,
    pub status: BoardStatus,
    pub step_budget: i64,
    pub steps_used: i64,
    pub created_at: String,
    pub closed_at: Option<String>,
    pub outcome: Option<String>,
}

impl Board {
    pub fn steps_remaining(&self) -> i64 {
        (self.step_budget - self.steps_used).max(0)
    }
}

#[derive(Debug, Clone)]
pub struct Fact {
    pub id: String,
    pub statement: String,
    pub polarity: Polarity,
    pub basis: Basis,
    pub author: FactAuthor,
    pub from_intent: Option<String>,
    pub evidence_event_ids: Vec<u64>,
    pub from_fact_ids: Vec<String>,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct Intent {
    pub id: String,
    pub direction: String,
    pub rationale: String,
    pub from_fact_ids: Vec<String>,
    pub state: IntentState,
    pub priority: i64,
    pub claimed_by: Option<String>,
    pub claimed_at: Option<String>,
    pub prior_claims: Vec<String>,
    pub closed_by_fact: Option<String>,
    pub created_at: String,
}

impl Intent {
    /// A claim older than the TTL is treated as open by the frontier, so a dead
    /// worker cannot freeze a branch.
    pub fn claim_expired(&self, now: DateTime<Local>) -> bool {
        match (&self.claimed_at, self.state) {
            (Some(at), IntentState::Claimed) => match DateTime::parse_from_rfc3339(at) {
                Ok(claimed) => (now - claimed.with_timezone(&Local)).num_seconds() > CLAIM_TTL_SECS,
                Err(_) => true,
            },
            _ => false,
        }
    }

    /// On the frontier means: available to claim right now.
    pub fn on_frontier(&self, now: DateTime<Local>) -> bool {
        self.state == IntentState::Open || self.claim_expired(now)
    }
}

#[derive(Debug, Clone)]
pub struct Hint {
    pub id: String,
    pub text: String,
    pub created_at: String,
    pub consumed_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelLink {
    pub model_id: String,
    pub relation: LinkRelation,
    pub note: String,
    pub revision_seen: u64,
    pub linked_at: String,
}

// ------------------------------------------------------------------ harness seams

/// The archive, as this module needs it: citations are checked against it, and a
/// harness-authored Fact is written into it first so that it cites something
/// real rather than being exempted from the rule every other Fact obeys.
pub trait BoardArchive {
    fn events(&self, ids: &[u64]) -> Result<Vec<EventSnapshot>>;
    fn record(&mut self, content: String) -> Result<u64>;
}

impl BoardArchive for crate::hippocampus::Hippocampus {
    fn events(&self, ids: &[u64]) -> Result<Vec<EventSnapshot>> {
        EvidenceSource::events(self, ids)
    }

    fn record(&mut self, content: String) -> Result<u64> {
        self.remember_tool_result(content)
    }
}

/// The current head of a world model, for `board_link` and the stale marker.
pub struct ModelHead {
    pub title: String,
    pub revision: u64,
}

pub trait ModelRevisions {
    fn head(&self, model_id: &str) -> Result<Option<ModelHead>>;
}

#[derive(Debug, Clone)]
pub struct TestRun {
    pub exit_code: Option<i64>,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u128,
    pub timed_out: bool,
}

impl TestRun {
    pub fn passed(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }

    fn verdict(&self) -> String {
        if self.timed_out {
            "timed out".to_string()
        } else {
            match self.exit_code {
                Some(0) => "passed (exit 0)".to_string(),
                Some(code) => format!("failed (exit {code})"),
                None => "failed (no exit status)".to_string(),
            }
        }
    }
}

/// A goal test that ran, on the page: collapsed, its board, verdict and
/// time; expanded, the command and all it printed.
fn show_test_run(board: &str, command: &str, run: &TestRun, verdict: &str) {
    let seconds = run.duration_ms as f64 / 1000.0;
    let brief = format!("board · {board}: goal test {verdict} in {seconds:.1}s");
    let mut full = format!("[board] {board}: goal test {verdict} in {seconds:.1}s\n$ {command}");
    for (label, text) in [("stdout", &run.stdout), ("stderr", &run.stderr)] {
        let text = text.trim_end();
        if text.trim().is_empty() {
            continue;
        }
        // The end of it: the whole of it is in the archive.
        let chars = text.chars().count();
        let shown: String = text.chars().skip(chars.saturating_sub(4_000)).collect();
        let cut = if chars > 4_000 { format!(" (the last 4,000 of {chars} characters; all of it is in the archive)") } else { String::new() };
        full.push_str(&format!("\n{label}{cut}:\n{shown}"));
    }
    if run.passed() {
        crate::terminal::say_done_as(&brief, &full);
    } else {
        crate::terminal::warn_status_as(&brief, &full);
    }
}

/// Where a `command` goal test runs. The live implementation is the VM; tests
/// use a fake. Note which way this points: the harness runs the test and tells
/// the module the verdict, and the model is nowhere in that path.
pub trait TestRunner {
    fn run(&self, command: &str, timeout_secs: u64) -> Result<TestRun>;
}

// ------------------------------------------------------------------ store

pub struct BoardStore {
    path: PathBuf,
}

impl BoardStore {
    pub fn beside_archive(archive: &Path) -> Self {
        Self { path: archive.with_file_name("board.sqlite3") }
    }

    fn connect(&self) -> Result<Connection> {
        // A layout change rebuilds rather than corrupting, and rebuilding keeps
        // the old file: a live search is not derived data the way the keyword
        // index is, so it is moved aside rather than dropped.
        if self.path.exists() && schema_of(&self.path)?.as_deref() != Some(SCHEMA) {
            let aside = self.path.with_extension(format!(
                "sqlite3.superseded-{}",
                Local::now().format("%Y%m%dT%H%M%S")
            ));
            std::fs::rename(&self.path, &aside)?;
            crate::terminal::warn_status(&format!("[board] schema changed; previous database kept at {}", aside.display()));
        }

        let db = crate::db::open(&self.path)?;
        db.execute_batch(
            "PRAGMA foreign_keys=ON; CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);

             CREATE TABLE IF NOT EXISTS boards(
                id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL,
                name TEXT NOT NULL,
                title TEXT NOT NULL,
                origin TEXT NOT NULL,
                goal TEXT NOT NULL,
                goal_test TEXT NOT NULL,
                goal_test_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                step_budget INTEGER NOT NULL,
                steps_used INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                closed_at TEXT,
                outcome TEXT);
             -- One open board per name; closed boards may share a retired name.
             CREATE UNIQUE INDEX IF NOT EXISTS boards_open_name
                ON boards(name) WHERE closed_at IS NULL;

             CREATE TABLE IF NOT EXISTS facts(
                id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL,
                board_id TEXT NOT NULL REFERENCES boards(id),
                statement TEXT NOT NULL,
                polarity TEXT NOT NULL,
                basis TEXT NOT NULL,
                author TEXT NOT NULL,
                from_intent TEXT,
                evidence_event_ids TEXT NOT NULL,
                from_fact_ids TEXT NOT NULL,
                created_at TEXT NOT NULL);
             CREATE INDEX IF NOT EXISTS facts_board ON facts(board_id, seq DESC);

             CREATE TABLE IF NOT EXISTS intents(
                id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL,
                board_id TEXT NOT NULL REFERENCES boards(id),
                direction TEXT NOT NULL,
                rationale TEXT NOT NULL,
                from_fact_ids TEXT NOT NULL,
                state TEXT NOT NULL,
                priority INTEGER NOT NULL,
                claimed_by TEXT,
                claimed_at TEXT,
                prior_claims TEXT NOT NULL,
                closed_by_fact TEXT,
                created_at TEXT NOT NULL);
             CREATE INDEX IF NOT EXISTS intents_frontier
                ON intents(board_id, state, priority DESC);

             CREATE TABLE IF NOT EXISTS hints(
                id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL,
                board_id TEXT NOT NULL REFERENCES boards(id),
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consumed_at TEXT);
             CREATE INDEX IF NOT EXISTS hints_board ON hints(board_id, seq);

             CREATE TABLE IF NOT EXISTS board_models(
                board_id TEXT NOT NULL REFERENCES boards(id),
                model_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                note TEXT NOT NULL,
                revision_seen INTEGER NOT NULL,
                linked_at TEXT NOT NULL,
                PRIMARY KEY (board_id, model_id, relation));

             -- One-shot operator permission to reopen a retired name.
             CREATE TABLE IF NOT EXISTS name_reuse(
                name TEXT PRIMARY KEY, granted_at TEXT NOT NULL);",
        )?;
        db.execute("INSERT OR REPLACE INTO metadata VALUES('schema',?1)", params![SCHEMA])?;
        Ok(db)
    }

    // ---- reads ----

    fn board_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Board> {
        Ok(Board {
            id: row.get(0)?,
            name: row.get(1)?,
            title: row.get(2)?,
            origin: row.get(3)?,
            goal: row.get(4)?,
            goal_test: row.get(5)?,
            goal_test_kind: GoalTestKind::parse(&row.get::<_, String>(6)?)
                .unwrap_or(GoalTestKind::Judgment),
            status: BoardStatus::parse(&row.get::<_, String>(7)?).unwrap_or(BoardStatus::Open),
            step_budget: row.get(8)?,
            steps_used: row.get(9)?,
            created_at: row.get(10)?,
            closed_at: row.get(11)?,
            outcome: row.get(12)?,
        })
    }

    const BOARD_SELECT: &'static str = "SELECT id,name,title,origin,goal,goal_test,\
        goal_test_kind,status,step_budget,steps_used,created_at,closed_at,outcome FROM boards";

    /// Resolve a name. The open board wins; otherwise the most recently closed
    /// one with that name, so references in the archive keep resolving.
    pub fn board(&self, name: &str) -> Result<Option<Board>> {
        let db = self.connect()?;
        let sql = format!(
            "{} WHERE name=?1 ORDER BY (closed_at IS NULL) DESC, seq DESC LIMIT 1",
            Self::BOARD_SELECT
        );
        Ok(db.query_row(&sql, params![name], Self::board_row).optional()?)
    }

    /// The board workflow's database (`board/flow.rs`), beside this one.
    pub fn flow_path(&self) -> PathBuf {
        self.path.with_file_name("board_flow.sqlite3")
    }

    pub fn board_by_id(&self, id: &str) -> Result<Option<Board>> {
        let db = self.connect()?;
        let sql = format!("{} WHERE id=?1", Self::BOARD_SELECT);
        Ok(db.query_row(&sql, params![id], Self::board_row).optional()?)
    }

    pub fn boards(&self, open_only: bool) -> Result<Vec<Board>> {
        let db = self.connect()?;
        let sql = format!(
            "{}{} ORDER BY seq DESC",
            Self::BOARD_SELECT,
            if open_only { " WHERE closed_at IS NULL" } else { "" }
        );
        let mut statement = db.prepare(&sql)?;
        let rows = statement.query_map([], Self::board_row)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn intent_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Intent> {
        let from: String = row.get(2)?;
        let prior: String = row.get(8)?;
        Ok(Intent {
            id: row.get(0)?,
            direction: row.get(1)?,
            from_fact_ids: serde_json::from_str(&from).unwrap_or_default(),
            rationale: row.get(3)?,
            state: IntentState::parse(&row.get::<_, String>(4)?).unwrap_or(IntentState::Open),
            priority: row.get(5)?,
            claimed_by: row.get(6)?,
            claimed_at: row.get(7)?,
            prior_claims: serde_json::from_str(&prior).unwrap_or_default(),
            closed_by_fact: row.get(9)?,
            created_at: row.get(10)?,
        })
    }

    const INTENT_SELECT: &'static str = "SELECT id,direction,from_fact_ids,rationale,state,\
        priority,claimed_by,claimed_at,prior_claims,closed_by_fact,created_at FROM intents";

    pub fn intents(&self, board_id: &str) -> Result<Vec<Intent>> {
        let db = self.connect()?;
        let sql = format!(
            "{} WHERE board_id=?1 ORDER BY priority DESC, seq DESC",
            Self::INTENT_SELECT
        );
        let mut statement = db.prepare(&sql)?;
        let rows = statement.query_map(params![board_id], Self::intent_row)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn intent(&self, board_id: &str, id: &str) -> Result<Option<Intent>> {
        let db = self.connect()?;
        let sql = format!("{} WHERE board_id=?1 AND id=?2", Self::INTENT_SELECT);
        Ok(db.query_row(&sql, params![board_id, id], Self::intent_row).optional()?)
    }

    /// Intents available to claim: `open`, plus claims past the TTL.
    pub fn frontier(&self, board_id: &str, now: DateTime<Local>) -> Result<Vec<Intent>> {
        Ok(self
            .intents(board_id)?
            .into_iter()
            .filter(|intent| intent.on_frontier(now))
            .collect())
    }

    fn fact_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Fact> {
        let evidence: String = row.get(5)?;
        let from: String = row.get(6)?;
        Ok(Fact {
            id: row.get(0)?,
            statement: row.get(1)?,
            polarity: Polarity::parse(&row.get::<_, String>(2)?).unwrap_or(Polarity::Supports),
            basis: Basis::parse(&row.get::<_, String>(3)?).unwrap_or(Basis::Observed),
            author: FactAuthor::parse(&row.get::<_, String>(4)?).unwrap_or(FactAuthor::Model),
            evidence_event_ids: serde_json::from_str(&evidence).unwrap_or_default(),
            from_fact_ids: serde_json::from_str(&from).unwrap_or_default(),
            from_intent: row.get(7)?,
            created_at: row.get(8)?,
        })
    }

    const FACT_SELECT: &'static str = "SELECT id,statement,polarity,basis,author,\
        evidence_event_ids,from_fact_ids,from_intent,created_at FROM facts";

    /// Newest first, which is the order a read wants them in.
    pub fn facts(&self, board_id: &str) -> Result<Vec<Fact>> {
        let db = self.connect()?;
        let sql = format!("{} WHERE board_id=?1 ORDER BY seq DESC", Self::FACT_SELECT);
        let mut statement = db.prepare(&sql)?;
        let rows = statement.query_map(params![board_id], Self::fact_row)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn fact(&self, board_id: &str, id: &str) -> Result<Option<Fact>> {
        let db = self.connect()?;
        let sql = format!("{} WHERE board_id=?1 AND id=?2", Self::FACT_SELECT);
        Ok(db.query_row(&sql, params![board_id, id], Self::fact_row).optional()?)
    }

    pub fn hints(&self, board_id: &str, unconsumed_only: bool) -> Result<Vec<Hint>> {
        let db = self.connect()?;
        let sql = format!(
            "SELECT id,text,created_at,consumed_at FROM hints WHERE board_id=?1{} ORDER BY seq",
            if unconsumed_only { " AND consumed_at IS NULL" } else { "" }
        );
        let mut statement = db.prepare(&sql)?;
        let rows = statement.query_map(params![board_id], |row| {
            Ok(Hint {
                id: row.get(0)?,
                text: row.get(1)?,
                created_at: row.get(2)?,
                consumed_at: row.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn links(&self, board_id: &str) -> Result<Vec<ModelLink>> {
        let db = self.connect()?;
        let mut statement = db.prepare(
            "SELECT model_id,relation,note,revision_seen,linked_at FROM board_models \
             WHERE board_id=?1 ORDER BY model_id,relation",
        )?;
        let rows = statement.query_map(params![board_id], |row| {
            let revision: i64 = row.get(3)?;
            Ok(ModelLink {
                model_id: row.get(0)?,
                relation: LinkRelation::parse(&row.get::<_, String>(1)?)
                    .unwrap_or(LinkRelation::Background),
                note: row.get(2)?,
                revision_seen: revision.max(0) as u64,
                linked_at: row.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    // ---- writes ----

    fn next_id(db: &Connection, table: &str, prefix: &str) -> Result<(String, i64)> {
        let seq: i64 =
            db.query_row(&format!("SELECT COALESCE(MAX(seq),0)+1 FROM {table}"), [], |row| {
                row.get(0)
            })?;
        Ok((format!("{prefix}-{seq:04}"), seq))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_board(
        &self,
        name: &str,
        title: &str,
        origin: &str,
        goal: &str,
        goal_test: &str,
        goal_test_kind: GoalTestKind,
        step_budget: i64,
    ) -> Result<std::result::Result<Board, String>> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let open_exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM boards WHERE name=?1 AND closed_at IS NULL)",
            params![name],
            |row| row.get(0),
        )?;
        if open_exists {
            return Ok(Err(format!(
                "a board named '{name}' is already open — read it with board_read, or choose \
                 another name"
            )));
        }

        // A retired name still has references pointing at it in the archive, so
        // reusing it is the operator's call, not the model's.
        let retired: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM boards WHERE name=?1)",
            params![name],
            |row| row.get(0),
        )?;
        if retired {
            let granted: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM name_reuse WHERE name=?1)",
                params![name],
                |row| row.get(0),
            )?;
            if !granted {
                return Ok(Err(format!(
                    "'{name}' belonged to a closed board and references to it survive in the \
                     archive. Choose a different name, or ask the operator to run `/board reuse {name}`."
                )));
            }
            tx.execute("DELETE FROM name_reuse WHERE name=?1", params![name])?;
        }

        let (id, seq) = Self::next_id(&tx, "boards", "board")?;
        tx.execute(
            "INSERT INTO boards VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,0,?11,NULL,NULL)",
            params![
                id,
                seq,
                name,
                title,
                origin,
                goal,
                goal_test,
                goal_test_kind.label(),
                BoardStatus::Open.label(),
                step_budget,
                Local::now().to_rfc3339(),
            ],
        )?;
        tx.commit()?;

        let board = self.board(name)?.ok_or("board vanished immediately after creation")?;
        Ok(Ok(board))
    }

    /// Declare a batch of intents (direction, rationale, from facts, priority)
    /// in one transaction: their IDs can't collide with a batch another worker
    /// declares at the same moment, the batch is written whole or not at all,
    /// and it is refused (with the frontier's size) if it would take the
    /// frontier past `frontier_limit`.
    pub fn declare_intents(
        &self,
        board_id: &str,
        batch: &[(String, String, Vec<String>, i64)],
        frontier_limit: usize,
        now: DateTime<Local>,
    ) -> Result<std::result::Result<Vec<Intent>, usize>> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let open_now = {
            let sql = format!("{} WHERE board_id=?1", Self::INTENT_SELECT);
            let mut statement = tx.prepare(&sql)?;
            let intents = statement.query_map(params![board_id], Self::intent_row)?;
            let mut open_now = 0;
            for intent in intents {
                if intent?.on_frontier(now) {
                    open_now += 1;
                }
            }
            open_now
        };
        if open_now + batch.len() > frontier_limit {
            return Ok(Err(open_now));
        }
        let mut ids = Vec::new();
        for (direction, rationale, from_fact_ids, priority) in batch {
            let (id, seq) = Self::next_id(&tx, "intents", "intent")?;
            tx.execute(
                "INSERT INTO intents VALUES(?1,?2,?3,?4,?5,?6,?7,?8,NULL,NULL,'[]',NULL,?9)",
                params![
                    id,
                    seq,
                    board_id,
                    direction,
                    rationale,
                    serde_json::to_string(from_fact_ids)?,
                    IntentState::Open.label(),
                    priority,
                    Local::now().to_rfc3339(),
                ],
            )?;
            ids.push(id);
        }
        tx.commit()?;
        let intents: Result<Vec<Intent>> = ids
            .iter()
            .map(|id| self.intent(board_id, id)?.ok_or_else(|| "intent vanished after insert".into()))
            .collect();
        intents.map(Ok)
    }

    /// Give back every claim `worker` still holds, when it ends: its branches
    /// return to the frontier at once rather than after the claim goes stale.
    /// Each is recorded as a prior claim, as an expired one is.
    pub fn release_claims(&self, worker: &str) -> Result<usize> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let held: Vec<(String, Option<String>, String)> = {
            let mut statement = tx.prepare(
                "SELECT id,claimed_at,prior_claims FROM intents WHERE state='claimed' AND claimed_by=?1",
            )?;
            let rows = statement.query_map(params![worker], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
            rows.collect::<std::result::Result<_, _>>()?
        };
        for (id, claimed_at, prior) in &held {
            let mut prior: Vec<String> = serde_json::from_str(prior).unwrap_or_default();
            prior.push(format!("{worker} @ {} (released when it ended)", claimed_at.as_deref().unwrap_or("unknown")));
            tx.execute(
                "UPDATE intents SET state='open', claimed_by=NULL, claimed_at=NULL, prior_claims=?1 \
                 WHERE id=?2 AND state='claimed' AND claimed_by=?3",
                params![serde_json::to_string(&prior)?, id, worker],
            )?;
        }
        tx.commit()?;
        Ok(held.len())
    }

    /// Claim one Intent for one worker.
    ///
    /// The ordinary case is a single conditional update: zero rows changed means
    /// somebody else took it and the caller picks another. No read-then-write,
    /// and no lock held across a model call. The second path exists only for the
    /// expired-claim takeover, which has to record the claim it displaced.
    pub fn claim(
        &self,
        board_id: &str,
        intent_id: &str,
        worker: &str,
        now: DateTime<Local>,
    ) -> Result<std::result::Result<Intent, String>> {
        let mut db = self.connect()?;
        let stamp = now.to_rfc3339();

        let changed = db.execute(
            "UPDATE intents SET state='claimed', claimed_by=?1, claimed_at=?2 \
             WHERE id=?3 AND board_id=?4 AND state='open'",
            params![worker, stamp, intent_id, board_id],
        )?;
        if changed == 1 {
            return Ok(Ok(self
                .intent(board_id, intent_id)?
                .ok_or("claimed intent vanished")?));
        }

        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let sql = format!("{} WHERE board_id=?1 AND id=?2", Self::INTENT_SELECT);
        let existing =
            tx.query_row(&sql, params![board_id, intent_id], Self::intent_row).optional()?;

        let Some(intent) = existing else {
            return Ok(Err(format!("no intent '{intent_id}' on this board")));
        };

        match intent.state {
            IntentState::Explored => {
                return Ok(Err(format!("{intent_id} has already been explored")));
            }
            IntentState::Dead => {
                return Ok(Err(format!("{intent_id} is dead — it was ruled out")));
            }
            IntentState::Open => return Ok(Err("already_claimed".to_string())),
            IntentState::Claimed => {}
        }

        if !intent.claim_expired(now) {
            return Ok(Err(format!(
                "already_claimed: {intent_id} is held by {} since {}",
                intent.claimed_by.as_deref().unwrap_or("someone"),
                intent.claimed_at.as_deref().unwrap_or("unknown"),
            )));
        }

        // The displaced claim is recorded rather than erased, so a worker that
        // comes back from the dead gets a clear error instead of writing a Fact
        // against an Intent someone else now owns.
        let mut prior = intent.prior_claims.clone();
        prior.push(format!(
            "{} @ {} (expired)",
            intent.claimed_by.as_deref().unwrap_or("unknown"),
            intent.claimed_at.as_deref().unwrap_or("unknown"),
        ));
        tx.execute(
            "UPDATE intents SET claimed_by=?1, claimed_at=?2, prior_claims=?3 \
             WHERE id=?4 AND state='claimed'",
            params![worker, stamp, serde_json::to_string(&prior)?, intent_id],
        )?;
        tx.commit()?;

        Ok(Ok(self.intent(board_id, intent_id)?.ok_or("claimed intent vanished")?))
    }

    /// Write a Fact and, if it answers a claimed Intent, mark that Intent explored.
    #[allow(clippy::too_many_arguments)]
    pub fn record_fact(
        &self,
        board_id: &str,
        statement: &str,
        polarity: Polarity,
        basis: Basis,
        author: FactAuthor,
        from_intent: Option<&str>,
        evidence_event_ids: &[u64],
        from_fact_ids: &[String],
        spend_step: bool,
    ) -> Result<Fact> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (id, seq) = Self::next_id(&tx, "facts", "fact")?;

        tx.execute(
            "INSERT INTO facts VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![
                id,
                seq,
                board_id,
                statement,
                polarity.label(),
                basis.label(),
                author.label(),
                from_intent,
                serde_json::to_string(evidence_event_ids)?,
                serde_json::to_string(from_fact_ids)?,
                Local::now().to_rfc3339(),
            ],
        )?;

        if let Some(intent_id) = from_intent {
            tx.execute(
                "UPDATE intents SET state='explored' WHERE id=?1 AND board_id=?2",
                params![intent_id, board_id],
            )?;
        }

        if spend_step {
            tx.execute(
                "UPDATE boards SET steps_used=steps_used+1 WHERE id=?1",
                params![board_id],
            )?;
            // Budget spent suspends rather than closes: the graph is intact and
            // the operator decides whether to extend it.
            tx.execute(
                "UPDATE boards SET status='suspended' \
                 WHERE id=?1 AND status='open' AND steps_used>=step_budget",
                params![board_id],
            )?;
        }

        tx.commit()?;
        self.fact(board_id, &id)?.ok_or_else(|| "fact vanished after insert".into())
    }

    pub fn kill_intent(
        &self,
        board_id: &str,
        intent_id: &str,
        fact_id: &str,
    ) -> Result<std::result::Result<Intent, String>> {
        let db = self.connect()?;
        let changed = db.execute(
            "UPDATE intents SET state='dead', closed_by_fact=?1 \
             WHERE id=?2 AND board_id=?3 AND state IN ('open','claimed')",
            params![fact_id, intent_id, board_id],
        )?;
        if changed == 0 {
            return Ok(Err(format!("{intent_id} is not an open or claimed intent on this board")));
        }
        Ok(Ok(self.intent(board_id, intent_id)?.ok_or("killed intent vanished")?))
    }

    /// Operator only. One direction by construction: nothing in the tool surface
    /// reaches this.
    pub fn add_hint(&self, board_id: &str, text: &str) -> Result<Hint> {
        // Numbered and written in one transaction, as a worker's writes are,
        // so two never take the same id.
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (id, seq) = Self::next_id(&tx, "hints", "hint")?;
        let now = Local::now().to_rfc3339();
        tx.execute(
            "INSERT INTO hints VALUES(?1,?2,?3,?4,?5,NULL)",
            params![id, seq, board_id, text, now],
        )?;
        tx.commit()?;
        Ok(Hint { id, text: text.to_string(), created_at: now, consumed_at: None })
    }

    /// Operator only: a lead of his own, open at `priority` (1 to 10), under
    /// the finding it came from when there is one. Never refused for the
    /// frontier's size: it is his board.
    pub fn operator_lead(&self, board_id: &str, direction: &str, from_fact: Option<&str>, priority: i64) -> Result<Intent> {
        let from: Vec<String> = from_fact.map(|fact| vec![fact.to_string()]).unwrap_or_default();
        let batch = [(direction.to_string(), "the operator's own lead".to_string(), from, priority.clamp(1, 10))];
        match self.declare_intents(board_id, &batch, usize::MAX, Local::now())? {
            Ok(mut intents) if !intents.is_empty() => Ok(intents.remove(0)),
            _ => Err("the lead was not written".into()),
        }
    }

    /// Operator only: a finding of his own, observed, about the lead or under
    /// the finding it is about. The lead stays as it is: what to make of it
    /// is the agent's to decide.
    pub fn operator_fact(&self, board_id: &str, statement: &str, polarity: Polarity, about_intent: Option<&str>, from_fact: Option<&str>) -> Result<Fact> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (id, seq) = Self::next_id(&tx, "facts", "fact")?;
        let from: Vec<String> = from_fact.map(|fact| vec![fact.to_string()]).unwrap_or_default();
        tx.execute(
            "INSERT INTO facts VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![
                id,
                seq,
                board_id,
                statement,
                polarity.label(),
                Basis::Observed.label(),
                FactAuthor::Operator.label(),
                about_intent,
                "[]",
                serde_json::to_string(&from)?,
                Local::now().to_rfc3339(),
            ],
        )?;
        tx.commit()?;
        self.fact(board_id, &id)?.ok_or_else(|| "fact vanished after insert".into())
    }

    /// The harness's own finding from an audit's result (`audit.rs`): under
    /// the audit's lead (or a lead a claim was linked to), which stays as it
    /// is, citing the audit's evidence by its `audit:<name>/c<N>` reference
    /// in the statement. Never reached by a tool.
    pub fn harness_finding(&self, board_id: &str, statement: &str, polarity: Polarity, about_intent: Option<&str>) -> Result<Fact> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (id, seq) = Self::next_id(&tx, "facts", "fact")?;
        tx.execute(
            "INSERT INTO facts VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![
                id,
                seq,
                board_id,
                statement,
                polarity.label(),
                Basis::Observed.label(),
                FactAuthor::Harness.label(),
                about_intent,
                "[]",
                "[]",
                Local::now().to_rfc3339(),
            ],
        )?;
        tx.commit()?;
        self.fact(board_id, &id)?.ok_or_else(|| "fact vanished after insert".into())
    }

    /// Steps spent by work that is not a fact (an audit's checks), the board
    /// suspended as a fact's would at its budget. Whether it is still open.
    pub fn spend_steps(&self, board_id: &str, steps: i64) -> Result<bool> {
        let db = self.connect()?;
        db.execute("UPDATE boards SET steps_used=steps_used+?2 WHERE id=?1", params![board_id, steps])?;
        db.execute("UPDATE boards SET status='suspended' WHERE id=?1 AND status='open' AND steps_used>=step_budget", params![board_id])?;
        let status: String = db.query_row("SELECT status FROM boards WHERE id=?1", params![board_id], |row| row.get(0))?;
        Ok(status == "open")
    }

    /// A claim `worker` holds, renewed (an audit holds its lead while it
    /// runs, past the claim's usual half hour). Whether it held it.
    pub fn renew_claim(&self, board_id: &str, intent_id: &str, worker: &str) -> Result<bool> {
        let db = self.connect()?;
        Ok(db.execute(
            "UPDATE intents SET claimed_at=?1 WHERE id=?2 AND board_id=?3 AND state='claimed' AND claimed_by=?4",
            params![Local::now().to_rfc3339(), intent_id, board_id, worker],
        )? == 1)
    }

    /// A lead closed as explored without a finding of its own (an audit's,
    /// when its queue is empty or it is detached). Whether it was open.
    pub fn close_explored(&self, board_id: &str, intent_id: &str) -> Result<bool> {
        let db = self.connect()?;
        Ok(db.execute(
            "UPDATE intents SET state='explored' WHERE id=?1 AND board_id=?2 AND state IN ('open','claimed')",
            params![intent_id, board_id],
        )? == 1)
    }

    /// Operator only: a note taken off the board. Whether it was there.
    pub fn remove_hint(&self, board_id: &str, id: &str) -> Result<bool> {
        let db = self.connect()?;
        Ok(db.execute("DELETE FROM hints WHERE board_id=?1 AND id=?2", params![board_id, id])? > 0)
    }

    fn consume_hints(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let db = self.connect()?;
        let now = Local::now().to_rfc3339();
        for id in ids {
            db.execute(
                "UPDATE hints SET consumed_at=?1 WHERE id=?2 AND consumed_at IS NULL",
                params![now, id],
            )?;
        }
        Ok(())
    }

    pub fn link_model(
        &self,
        board_id: &str,
        model_id: &str,
        relation: LinkRelation,
        note: &str,
        revision_seen: u64,
    ) -> Result<()> {
        let db = self.connect()?;
        db.execute(
            "INSERT INTO board_models VALUES(?1,?2,?3,?4,?5,?6) \
             ON CONFLICT(board_id,model_id,relation) DO UPDATE SET \
             note=excluded.note, revision_seen=excluded.revision_seen, \
             linked_at=excluded.linked_at",
            params![
                board_id,
                model_id,
                relation.label(),
                note,
                revision_seen as i64,
                Local::now().to_rfc3339()
            ],
        )?;
        Ok(())
    }

    pub fn unlink_model(
        &self,
        board_id: &str,
        model_id: &str,
        relation: LinkRelation,
    ) -> Result<bool> {
        let db = self.connect()?;
        let changed = db.execute(
            "DELETE FROM board_models WHERE board_id=?1 AND model_id=?2 AND relation=?3",
            params![board_id, model_id, relation.label()],
        )?;
        Ok(changed > 0)
    }

    pub fn set_status(
        &self,
        board_id: &str,
        status: BoardStatus,
        outcome: &str,
    ) -> Result<std::result::Result<Board, String>> {
        let closed_at =
            if status.is_closed() { Some(Local::now().to_rfc3339()) } else { None };
        let db = self.connect()?;
        let changed = db.execute(
            "UPDATE boards SET status=?1, outcome=?2, closed_at=?3 \
             WHERE id=?4 AND closed_at IS NULL",
            params![status.label(), outcome, closed_at, board_id],
        )?;
        if changed == 0 {
            return Ok(Err("that board is already closed".to_string()));
        }
        Ok(Ok(self.board_by_id(board_id)?.ok_or("board vanished")?))
    }

    /// The operator's confirmation on a `judgment` board. The one transition the model
    /// has no path to, by construction rather than by instruction.
    pub fn confirm_solved(&self, name: &str) -> Result<std::result::Result<Board, String>> {
        let Some(board) = self.board(name)? else {
            return Ok(Err(format!("no board named '{name}'")));
        };
        if board.status != BoardStatus::ProposedSolved {
            return Ok(Err(format!(
                "'{name}' is {}, not proposed_solved — only a proposed solution needs confirming",
                board.status.label()
            )));
        }
        let db = self.connect()?;
        db.execute(
            "UPDATE boards SET status='solved', \
             outcome=COALESCE(outcome,'') || ' [confirmed by the operator]' WHERE id=?1",
            params![board.id],
        )?;
        Ok(Ok(self.board_by_id(&board.id)?.ok_or("board vanished")?))
    }

    /// The operator's no to a proposed solution: the board is open again, as the agent
    /// left it, with his reason (if he gave one) as a note on it, which the agent
    /// sees on its next read. Its outcome goes, as it is not the outcome.
    pub fn reject_solved(&self, name: &str, why: &str) -> Result<std::result::Result<Board, String>> {
        let Some(board) = self.board(name)? else {
            return Ok(Err(format!("no board named '{name}'")));
        };
        let said = match board.status {
            BoardStatus::ProposedSolved => "Not solved: the operator rejected the proposed solution.",
            BoardStatus::ProposedExhausted => "Not exhausted: the operator thinks something is left to try.",
            other => {
                return Ok(Err(format!(
                    "'{name}' is {}, not proposed_solved or proposed_exhausted — only a proposal can be rejected",
                    other.label()
                )));
            }
        };
        self.reopen(&board.id)?;
        let why = why.trim();
        let note = if why.is_empty() { said.to_string() } else { format!("{said} {why}") };
        self.add_hint(&board.id, &note.chars().take(MAX_HINT_CHARS).collect::<String>())?;
        Ok(Ok(self.board_by_id(&board.id)?.ok_or("board vanished")?))
    }

    /// The operator agrees nothing is left to try: the agent's proposal made exhausted.
    pub fn confirm_exhausted(&self, name: &str) -> Result<std::result::Result<Board, String>> {
        let Some(board) = self.board(name)? else {
            return Ok(Err(format!("no board named '{name}'")));
        };
        if board.status != BoardStatus::ProposedExhausted {
            return Ok(Err(format!(
                "'{name}' is {}, not proposed_exhausted — only s0ma's proposal can be closed exhausted",
                board.status.label()
            )));
        }
        let db = self.connect()?;
        db.execute(
            "UPDATE boards SET status='exhausted', \
             outcome=COALESCE(outcome,'') || ' [closed exhausted by the operator]' WHERE id=?1",
            params![board.id],
        )?;
        Ok(Ok(self.board_by_id(&board.id)?.ok_or("board vanished")?))
    }

    /// Open again, as it was left, without the outcome it was closed with:
    /// The operator sent it back (a rejected proposal, or from review or
    /// verification, `board/flow.rs`).
    pub fn reopen(&self, board_id: &str) -> Result<()> {
        let db = self.connect()?;
        db.execute(
            "UPDATE boards SET status='open', closed_at=NULL, outcome=NULL WHERE id=?1",
            params![board_id],
        )?;
        Ok(())
    }

    /// Its status put back to `status`, closed as it was: an exhausted board the
    /// operator takes back to review is the agent's proposal again
    /// (`crate::board::flow::to_review`).
    pub fn restate(&self, board_id: &str, status: BoardStatus) -> Result<()> {
        let db = self.connect()?;
        db.execute("UPDATE boards SET status=?1 WHERE id=?2", params![status.label(), board_id])?;
        Ok(())
    }

    /// Delivered: solved, with `note` (how) after its outcome. The operator's
    /// acceptance of a verified board (`crate::board::flow::accept`) — like
    /// `confirm_solved`, a transition no tool has.
    pub fn deliver(&self, board_id: &str, note: &str) -> Result<()> {
        let db = self.connect()?;
        db.execute(
            "UPDATE boards SET status='solved', outcome=COALESCE(outcome,'') || ?1, \
             closed_at=COALESCE(closed_at, ?2) WHERE id=?3",
            params![note, Local::now().to_rfc3339(), board_id],
        )?;
        Ok(())
    }

    /// The operator threw it away (`crate::board::flow::discard`): abandoned, closed, said so
    /// in its outcome, whatever it was.
    pub fn discard(&self, board_id: &str) -> Result<()> {
        let db = self.connect()?;
        db.execute(
            "UPDATE boards SET status='abandoned', outcome=COALESCE(outcome,'') || ' [discarded by the operator]', \
             closed_at=COALESCE(closed_at, ?1) WHERE id=?2",
            params![Local::now().to_rfc3339(), board_id],
        )?;
        Ok(())
    }

    /// Reopen a suspended board with more steps. Operator only — the budget is a
    /// stopping condition, and a model that could raise its own would not have one.
    pub fn extend_budget(
        &self,
        name: &str,
        extra: i64,
    ) -> Result<std::result::Result<Board, String>> {
        let Some(board) = self.board(name)? else {
            return Ok(Err(format!("no board named '{name}'")));
        };
        if board.closed_at.is_some() {
            return Ok(Err(format!("'{name}' is closed ({})", board.status.label())));
        }
        let db = self.connect()?;
        db.execute(
            "UPDATE boards SET step_budget=step_budget+?1, \
             status=CASE WHEN status='suspended' THEN 'open' ELSE status END WHERE id=?2",
            params![extra, board.id],
        )?;
        Ok(Ok(self.board_by_id(&board.id)?.ok_or("board vanished")?))
    }

    pub fn allow_name_reuse(&self, name: &str) -> Result<()> {
        let db = self.connect()?;
        db.execute(
            "INSERT OR REPLACE INTO name_reuse VALUES(?1,?2)",
            params![name, Local::now().to_rfc3339()],
        )?;
        Ok(())
    }
}

fn schema_of(path: &Path) -> Result<Option<String>> {
    let db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(10))?;
    let has_metadata: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata')",
        [],
        |row| row.get(0),
    )?;
    if !has_metadata {
        // No metadata table: either an empty file, or something from before this
        // module existed. Only the empty case is safe to treat as current.
        let any: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table')",
            [],
            |row| row.get(0),
        )?;
        return Ok(if any { Some(String::new()) } else { Some(SCHEMA.to_string()) });
    }
    Ok(db
        .query_row("SELECT value FROM metadata WHERE key='schema'", [], |row| row.get(0))
        .optional()?)
}

// ------------------------------------------------------------------ helpers

pub(crate) fn rejected(error: impl Into<String>) -> String {
    json!({"ok": false, "error": error.into(), "recorded": false}).to_string()
}

pub(crate) fn bounded(
    value: &str,
    min: usize,
    max: usize,
    label: &str,
) -> std::result::Result<String, String> {
    let trimmed = value.trim();
    let chars = trimmed.chars().count();
    if chars < min {
        return Err(format!("{label} is too short (at least {min} characters)"));
    }
    if chars > max {
        return Err(format!("{label} is too long (at most {max} characters)"));
    }
    Ok(trimmed.to_string())
}

/// A board name is a handle that gets typed in conversation and in operator
/// commands, so it is lowercase kebab and nothing else.
fn valid_name(name: &str) -> std::result::Result<String, String> {
    let name = name.trim().to_string();
    if name.is_empty() || name.chars().count() > MAX_NAME_CHARS {
        return Err(format!("name must be 1..{MAX_NAME_CHARS} characters"));
    }
    let shaped = name
        .bytes()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
        && !name.starts_with('-')
        && !name.ends_with('-');
    if !shaped {
        return Err(
            "name must be lowercase letters, digits and hyphens, e.g. intermittent-flush-failure"
                .to_string(),
        );
    }
    Ok(name)
}

fn relative_age(timestamp: &str, now: DateTime<Local>) -> String {
    let Ok(then) = DateTime::parse_from_rfc3339(timestamp) else {
        return "unknown".to_string();
    };
    let seconds = (now - then.with_timezone(&Local)).num_seconds().max(0);
    match seconds {
        0..=89 => format!("{seconds}s ago"),
        90..=5_399 => format!("{}m ago", seconds / 60),
        5_400..=172_799 => format!("{}h ago", seconds / 3_600),
        _ => format!("{}d ago", seconds / 86_400),
    }
}

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "whether", "check", "try", "see", "that", "this", "from", "its",
    "has", "was", "are", "not", "but", "any", "all",
];

fn signature(text: &str) -> BTreeSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.len() > 2 && !STOPWORDS.contains(word))
        .map(|word| word.to_string())
        .collect()
}

/// Jaccard overlap of content words. Deliberately crude: it warns, it never
/// blocks, and deciding whether two directions are really the same is the one
/// thing the model does better than any threshold.
fn similarity(left: &str, right: &str) -> f64 {
    let a = signature(left);
    let b = signature(right);
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.intersection(&b).count() as f64 / a.union(&b).count() as f64
}

fn describe_board(board: &Board, now: DateTime<Local>) -> Value {
    json!({
        "name": board.name,
        "title": board.title,
        "origin": board.origin,
        "goal": board.goal,
        "goal_test": board.goal_test,
        "goal_test_kind": board.goal_test_kind.label(),
        "status": board.status.label(),
        "steps_used": board.steps_used,
        "step_budget": board.step_budget,
        "steps_remaining": board.steps_remaining(),
        "opened": relative_age(&board.created_at, now),
        "outcome": board.outcome,
    })
}

fn describe_intent(intent: &Intent, now: DateTime<Local>) -> Value {
    let mut value = json!({
        "id": intent.id,
        "direction": intent.direction,
        "rationale": intent.rationale,
        "from_fact_ids": intent.from_fact_ids,
        "state": intent.state.label(),
        "priority": intent.priority,
        "declared": relative_age(&intent.created_at, now),
    });
    if intent.state == IntentState::Claimed {
        value["claimed_by"] = json!(intent.claimed_by);
        value["claimed"] = json!(
            intent
                .claimed_at
                .as_deref()
                .map(|at| relative_age(at, now))
                .unwrap_or_else(|| "unknown".to_string())
        );
        if intent.claim_expired(now) {
            value["claim_expired"] = json!(true);
            value["note"] = json!("the claim went stale; this is back on the frontier");
        }
    }
    if let Some(fact) = &intent.closed_by_fact {
        value["ruled_out_by"] = json!(fact);
    }
    value
}

fn describe_fact(fact: &Fact, now: DateTime<Local>) -> Value {
    json!({
        "id": fact.id,
        "statement": fact.statement,
        "polarity": fact.polarity.label(),
        "basis": fact.basis.label(),
        "author": fact.author.label(),
        "from_intent": fact.from_intent,
        "evidence_event_ids": fact.evidence_event_ids,
        "from_fact_ids": fact.from_fact_ids,
        "established": relative_age(&fact.created_at, now),
    })
}

/// Facts are evidence-backed by construction, so this names exactly which
/// citations failed rather than refusing vaguely.
fn check_evidence(
    archive: &dyn BoardArchive,
    ids: &[u64],
) -> Result<std::result::Result<(), String>> {
    if ids.len() > MAX_EVIDENCE_IDS {
        return Ok(Err(format!("at most {MAX_EVIDENCE_IDS} evidence event IDs")));
    }
    let found = archive.events(ids)?;
    let missing: Vec<u64> = ids
        .iter()
        .copied()
        .filter(|id| !found.iter().any(|event| event.id == *id))
        .collect();
    if !missing.is_empty() {
        return Ok(Err(format!(
            "these event IDs don't exist: {missing:?} (find real ones with archive_search)"
        )));
    }
    Ok(Ok(()))
}

fn tail(text: &str, chars: usize) -> String {
    let count = text.chars().count();
    if count <= chars {
        return text.to_string();
    }
    let skipped = count - chars;
    format!("…[{skipped} earlier characters]…{}", text.chars().skip(skipped).collect::<String>())
}

// ------------------------------------------------------------------ tools

pub fn tool_definitions() -> Vec<ToolDefinition> {
    let board_name = json!({
        "type": "string",
        "description": "The board's name, e.g. intermittent-flush-failure."
    });

    vec![
        ToolDefinition::function(
            "board_list",
            "List your open boards: name, title, how big the frontier is, how much of the \
             step budget is spent. Cheap orientation — use it to see what searches are live \
             before committing to one.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "all": {"type": "boolean", "description": "Include closed boards. Default false."}
            }}),
        ),
        ToolDefinition::function(
            "board_open",
            "Open a board for a problem whose origin you know, whose goal you can test, and \
             whose path you don't know — an intermittent failure, a regression hiding among a lot \
             of candidates, a number that has to cross a threshold. Not for questions (threads) \
             and not for things you simply have to remember to do (loops). Before building any \
             structure, try the problem directly, once: most problems are not actually hard and \
             the board is overhead for them. If the direct attempt works, run the test and close \
             having spent one step; if it doesn't, what you learned is your first facts and \
             intents. goal_test_kind decides who gets to say you succeeded — prefer 'command', \
             because a shell command's exit status is a verdict you don't author. 'judgment' is \
             the fallback for goals with no mechanical test, and such a board can only ever reach \
             proposed_solved until the operator confirms it.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "name": {"type": "string",
                         "description": "Short stable handle, lowercase-with-hyphens, unique among open boards. You refer to the board by this everywhere."},
                "title": {"type": "string", "description": "One line of prose."},
                "origin": {"type": "string", "description": "The starting state: what is known at the outset."},
                "goal": {"type": "string", "description": "What success is."},
                "goal_test": {"type": "string",
                              "description": "How success gets checked. For 'command', the exact shell command — exit 0 means solved."},
                "goal_test_kind": {"type": "string", "enum": ["command", "observation", "judgment"]},
                "step_budget": {"type": "integer", "minimum": 1, "maximum": MAX_STEP_BUDGET,
                                "description": "Explorations allowed before the board suspends itself. Default 60."}
            }, "required": ["name", "title", "origin", "goal", "goal_test", "goal_test_kind"]}),
        ),
        ToolDefinition::function(
            "board_read",
            "Read the whole graph for one board: anything the operator has left you first, then linked \
             models, the open frontier in priority order, then the most recent facts. This is \
             your observe step — read before deciding anything. Facts are paged and the response \
             says exactly how many were left out, so never read absence into a short list.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "name": board_name.clone(),
                "offset": {"type": "integer", "minimum": 0,
                           "description": "Skip this many facts, newest first. Default 0."},
                "fact_limit": {"type": "integer", "minimum": 1, "maximum": MAX_FACT_PAGE,
                               "description": "Facts to return. Default 20."}
            }, "required": ["name"]}),
        ),
        ToolDefinition::function(
            "board_link",
            "Link a world model to this board, or unlink one. 'background' means the model informs \
             the search — read it before declaring intents. 'updates' means the search is expected \
             to change it, and on close the board reports which facts haven't made it in yet. \
             Linking copies nothing and promotes nothing: a fact from a half-finished search \
             quietly becoming a belief is exactly what this avoids. Promotion stays a deliberate \
             world_record_put citing the fact as evidence.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "name": board_name.clone(),
                "model_id": {"type": "string", "description": "A world_model model id."},
                "relation": {"type": "string", "enum": ["background", "updates"]},
                "note": {"type": "string", "description": "One line on why this model bears on the search."},
                "remove": {"type": "boolean", "description": "Unlink instead. Default false."}
            }, "required": ["name", "model_id", "relation"]}),
        ),
        ToolDefinition::function(
            "board_intent",
            "Declare one or more directions worth trying that nobody has tried yet. Each needs a \
             rationale naming the facts that suggest it — a direction you can't justify from what \
             you know is a guess with structure around it. Declaring costs no steps; exploring \
             does. Kill what the newest facts made pointless before declaring more.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "name": board_name.clone(),
                "intents": {"type": "array", "minItems": 1, "maxItems": MAX_INTENTS_PER_CALL,
                    "items": {"type": "object", "additionalProperties": false, "properties": {
                        "direction": {"type": "string", "description": "What to try, concrete enough to act on."},
                        "rationale": {"type": "string", "description": "Why this looks worth trying, naming the facts that suggest it."},
                        "from_fact_ids": {"type": "array", "items": {"type": "string"},
                                          "description": "Facts this descends from. Empty only for intents seeded at the start."},
                        "priority": {"type": "integer", "minimum": 0, "maximum": 9,
                                     "description": "Higher explores first. Default 5."}
                    }, "required": ["direction", "rationale"]}}
            }, "required": ["name", "intents"]}),
        ),
        ToolDefinition::function(
            "board_claim",
            "Claim one open intent before exploring it, so two workers never explore the same \
             branch. This can fail — if somebody else took it you get ok:false and pick another \
             rather than retrying. A claim goes stale after half an hour and returns to the \
             frontier.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "name": board_name.clone(),
                "intent_id": {"type": "string", "description": "e.g. intent-0012."}
            }, "required": ["name", "intent_id"]}),
        ),
        ToolDefinition::function(
            "board_fact",
            "Record exactly one thing an exploration actually showed, with the archive events that \
             back it. One fact per exploration, deliberately: say what you established rather than \
             dumping a narrative. A negative result is a fact, and matters more than it sounds — \
             most of a search is elimination, and a search that records only what worked re-tries \
             dead branches. Use polarity 'rules_out' for those. A fact written against your \
             claimed intent marks it explored and spends a step.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "name": board_name.clone(),
                "intent_id": {"type": "string",
                              "description": "The intent you claimed. Omit only for something learned outside any claimed branch."},
                "statement": {"type": "string", "description": "One claim, in prose."},
                "polarity": {"type": "string", "enum": ["supports", "rules_out"]},
                "basis": {"type": "string", "enum": ["observed", "inferred"],
                          "description": "'inferred' requires the facts it came from in from_fact_ids."},
                "evidence_event_ids": {"type": "array", "items": {"type": "integer", "minimum": 1},
                                       "maxItems": MAX_EVIDENCE_IDS,
                                       "description": "Archive events showing this. Checked: the call is rejected if any don't exist."},
                "from_fact_ids": {"type": "array", "items": {"type": "string"},
                                  "description": "For basis='inferred', the facts this was reasoned from."}
            }, "required": ["name", "statement", "polarity", "basis", "evidence_event_ids"]}),
        ),
        ToolDefinition::function(
            "board_kill",
            "Mark an intent dead because a fact ruled it out. Requires that fact — a branch you \
             merely lost interest in stays on the frontier, and a branch ruled out by evidence is \
             never re-explored.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "name": board_name.clone(),
                "intent_id": {"type": "string"},
                "fact_id": {"type": "string",
                            "description": "A 'rules_out' fact on this board that eliminates it."}
            }, "required": ["name", "intent_id", "fact_id"]}),
        ),
        ToolDefinition::function(
            "board_test",
            "Ask the harness to run this board's goal test. The harness runs it and writes the \
             resulting fact; you never report the verdict, and you cannot mark a board solved. \
             This is why the structure stays honest: the facts you gathered mean something only \
             because success wasn't scored by the thing that gathered them. Costs a step.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "name": board_name.clone()
            }, "required": ["name"]}),
        ),
        ToolDefinition::function(
            "board_close",
            "Close a board. 'solved' is not available to you — a passing goal test does that, and \
             on a judgment board the most you can reach is proposed_solved. Nor is 'exhausted': the \
             most you can reach is proposed_exhausted, meaning the frontier is empty and the rationale \
             argues it is genuinely empty rather than that you stopped thinking (a result, not a \
             failure); you write it up (board_report) for the operator's review, and he closes it exhausted \
             or sends it back. 'abandoned' means the \
             search isn't worth continuing.",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "name": board_name,
                "status": {"type": "string", "enum": ["proposed_solved", "proposed_exhausted", "abandoned"]},
                "rationale": {"type": "string",
                              "description": "Why it closes this way. For 'proposed_exhausted', the argument that nothing is left to try."}
            }, "required": ["name", "status", "rationale"]}),
        ),
    ]
}

// ------------------------------------------------------------------ execution

/// Everything a board tool call needs from the rest of the harness, so the
/// dispatch site stays one line.
pub struct Context<'a> {
    pub store: &'a BoardStore,
    pub archive: &'a mut dyn BoardArchive,
    pub models: &'a dyn ModelRevisions,
    pub tests: &'a dyn TestRunner,
    /// Who is calling. Supplied by the harness, never by the model — a worker
    /// that could name itself could steal another worker's claim.
    pub worker: &'a str,
    /// Set when a subagent is calling: its own tool results, which its facts
    /// cite by ref. A subagent works inside the agent's search: its reads leave
    /// the operator's hints unread for the agent, its facts answer an intent it claimed, it
    /// holds one claim at a time and can't kill a branch someone else holds.
    pub subagent: Option<&'a mut dyn EvidenceRefs>,
}

/// A subagent's own tool results, cited by ref in its facts. Only a fact that
/// is otherwise accepted gets its evidence copied into the archive.
pub trait EvidenceRefs {
    /// Which of `refs` it has no tool result for.
    fn unknown(&self, refs: &[String]) -> Vec<String>;
    /// The archive event holding each of `refs`, copying in any not yet copied.
    fn archive(&mut self, refs: &[String], archive: &mut dyn BoardArchive) -> Result<Vec<u64>>;
}

/// The board tools a subagent gets: enough to work a search the agent opened
/// (read it, add directions, claim one, record what it showed, rule branches
/// out, ask for the test). Opening, linking and closing stay with the agent.
pub const WORKER_TOOLS: [&str; 7] =
    ["board_list", "board_read", "board_intent", "board_claim", "board_fact", "board_kill", "board_test"];

/// Boards whose goal test is running now, in this process, by store and board.
static TESTS_RUNNING: std::sync::Mutex<BTreeSet<String>> = std::sync::Mutex::new(BTreeSet::new());

/// A board's goal test, marked running until dropped.
struct TestGuard(String);

impl TestGuard {
    fn take(board_id: &str) -> Option<Self> {
        let mut running = TESTS_RUNNING.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        running.insert(board_id.to_string()).then(|| Self(board_id.to_string()))
    }
}

impl Drop for TestGuard {
    fn drop(&mut self) {
        TESTS_RUNNING.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).remove(&self.0);
    }
}

/// The board named, or the tool's result saying there is none.
macro_rules! board_or_reject {
    ($context:expr, $name:expr) => {
        match $context.store.board(&$name)? {
            Some(board) => board,
            None => {
                return Ok(rejected(format!("no board named '{}' (see board_list)", $name)));
            }
        }
    };
}

/// Writes stop at a suspended or closed board; reads never do.
macro_rules! must_accept_work {
    ($context:expr, $board:expr) => {
        if $board.status == BoardStatus::Suspended {
            let who = if $context.subagent.is_some() { "say so in your report; s0ma can ask the operator" } else { "ask the operator" };
            return Ok(rejected(format!(
                "'{}' spent its {}-step budget and suspended. The graph is intact — {who} \
                 to extend it with `/board extend {} <steps>`.",
                $board.name, $board.step_budget, $board.name
            )));
        } else if !$board.status.accepts_work() {
            return Ok(rejected(format!(
                "'{}' is closed ({})",
                $board.name,
                $board.status.label()
            )));
        }
    };
}

/// One board tool, by name: each has a function of its own below. An
/// argument the model got wrong comes back as a rejection it can act on.
pub fn execute(context: &mut Context<'_>, call: &ToolCall) -> Result<String> {
    let now = Local::now();
    match call.function.name.as_str() {
        "board_list" => execute_list(context, call, now),
        "board_open" => execute_open(context, call, now),
        "board_read" => execute_read(context, call, now),
        "board_link" => execute_link(context, call),
        "board_intent" => execute_intent(context, call, now),
        "board_claim" => execute_claim(context, call, now),
        "board_fact" => execute_fact(context, call, now),
        "board_kill" => execute_kill(context, call, now),
        "board_test" => execute_test(context, call, now),
        "board_close" => execute_close(context, call, now),
        other => Err(format!("unknown board tool '{other}'").into()),
    }
}

/// `board_list`.
fn execute_list(context: &mut Context<'_>, call: &ToolCall, now: DateTime<Local>) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        #[serde(default)]
        all: bool,
    }
    let args: Args =
        serde_json::from_str(&call.function.arguments).unwrap_or(Args { all: false });

    let mut listed = Vec::new();
    for board in context.store.boards(!args.all)? {
        let hints = context.store.hints(&board.id, true)?.len();
        let mut entry = json!({
            "name": board.name,
            "title": board.title,
            "status": board.status.label(),
            "frontier": context.store.frontier(&board.id, now)?.len(),
            "facts": context.store.facts(&board.id)?.len(),
            "steps": format!("{}/{}", board.steps_used, board.step_budget),
            "opened": relative_age(&board.created_at, now),
        });
        if hints > 0 {
            entry["unread_from_operator"] = json!(hints);
        }
        listed.push(entry);
    }
    Ok(json!({"ok": true, "boards": listed}).to_string())
}

/// `board_open`.
fn execute_open(context: &mut Context<'_>, call: &ToolCall, now: DateTime<Local>) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        name: String,
        title: String,
        origin: String,
        goal: String,
        goal_test: String,
        goal_test_kind: String,
        #[serde(default)]
        step_budget: Option<i64>,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
    };

    let name = match valid_name(&args.name) {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };
    let title = match bounded(&args.title, MIN_TEXT_CHARS, MAX_TITLE_CHARS, "title") {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };
    let origin = match bounded(&args.origin, MIN_TEXT_CHARS, MAX_ORIGIN_CHARS, "origin") {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };
    let goal = match bounded(&args.goal, MIN_TEXT_CHARS, MAX_GOAL_CHARS, "goal") {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };
    let goal_test = match bounded(&args.goal_test, 1, MAX_GOAL_TEST_CHARS, "goal_test") {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };
    let kind = match GoalTestKind::parse(&args.goal_test_kind) {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };
    let budget = args.step_budget.unwrap_or(DEFAULT_STEP_BUDGET);
    if !(1..=MAX_STEP_BUDGET).contains(&budget) {
        return Ok(rejected(format!(
            "step_budget must be between 1 and {MAX_STEP_BUDGET}"
        )));
    }

    match context
        .store
        .open_board(&name, &title, &origin, &goal, &goal_test, kind, budget)?
    {
        Ok(board) => {
            let mut response = json!({
                "ok": true,
                "board": describe_board(&board, now),
                "next": "Try the problem directly, once, before building any structure. \
                         If that works, board_test and close. If it doesn't, what you \
                         learned is your first facts and intents.",
            });
            if kind == GoalTestKind::Judgment {
                response["note"] = json!(
                    "No mechanical test, so you can never mark this solved — the most it \
                     reaches is proposed_solved, and the operator confirms. If a command could \
                     actually test this goal, close this board and open one that uses it."
                );
            }
            Ok(response.to_string())
        }
        Err(error) => Ok(rejected(error)),
    }
}

/// `board_read`.
fn execute_read(context: &mut Context<'_>, call: &ToolCall, now: DateTime<Local>) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        name: String,
        #[serde(default)]
        offset: usize,
        #[serde(default)]
        fact_limit: Option<usize>,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
    };

    let board = board_or_reject!(context, args.name);
    let limit = args.fact_limit.unwrap_or(DEFAULT_FACT_PAGE).clamp(1, MAX_FACT_PAGE);

    let frontier: Vec<Value> = context
        .store
        .intents(&board.id)?
        .iter()
        .filter(|intent| intent.on_frontier(now))
        .map(|intent| describe_intent(intent, now))
        .collect();

    let all_facts = context.store.facts(&board.id)?;
    let total = all_facts.len();
    let shown: Vec<Value> = all_facts
        .iter()
        .skip(args.offset)
        .take(limit)
        .map(|fact| describe_fact(fact, now))
        .collect();
    let end = args.offset + shown.len();

    let mut links = Vec::new();
    for link in context.store.links(&board.id)? {
        let mut value = json!({
            "model_id": link.model_id,
            "relation": link.relation.label(),
            "note": link.note,
            "revision_seen": link.revision_seen,
            "linked": relative_age(&link.linked_at, now),
        });
        match context.models.head(&link.model_id)? {
            Some(head) => {
                value["title"] = json!(head.title);
                value["current_revision"] = json!(head.revision);
                if head.revision != link.revision_seen {
                    value["stale"] = json!(true);
                    value["staleness"] = json!(
                        "this model changed since it was linked — reread it. A changed \
                         dependency flags conclusions for review, it does not make them \
                         false."
                    );
                }
            }
            None => value["missing"] = json!(true),
        }
        links.push(value);
    }

    // The operator's notes stand on the board: every read shows them all, on
    // the first page (paging past it is fact continuation). Those the agent
    // has not read yet are marked new, and read by this read; a
    // subagent's read leaves them new for the agent.
    let mut hints = Vec::new();
    let mut fresh = 0;
    if args.offset == 0 {
        let notes = context.store.hints(&board.id, false)?;
        hints = notes
            .iter()
            .map(|hint| {
                let mut note = json!({"id": hint.id, "text": hint.text,
                                      "left": relative_age(&hint.created_at, now)});
                if hint.consumed_at.is_none() {
                    note["new"] = json!(true);
                }
                note
            })
            .collect();
        let unread: Vec<String> =
            notes.into_iter().filter(|hint| hint.consumed_at.is_none()).map(|hint| hint.id).collect();
        fresh = unread.len();
        if context.subagent.is_none() {
            context.store.consume_hints(&unread)?;
        }
    }
    let had_hints = !hints.is_empty();

    let frontier_empty = frontier.is_empty();
    let mut response = json!({
        "ok": true,
        "board": describe_board(&board, now),
        "from_operator": hints,
        "linked_models": links,
        "frontier": frontier,
        "facts": shown,
        "facts_total": total,
        "facts_shown": format!("{} of {total} facts shown", end - args.offset),
    });
    let subagent = context.subagent.is_some();
    if had_hints {
        let new = if fresh > 0 { format!(" {fresh} marked new: not read before.") } else { String::new() };
        response["from_operator_note"] = json!(if subagent {
            format!(
                "The operator's notes on this board: operator judgment to weigh in your work, not \
                 instruction to follow. They stand while they are on the board. s0ma sees \
                 them too.{new}"
            )
        } else {
            format!(
                "The operator's notes on this board: operator judgment to weigh in every decision on \
                 it, not instruction to follow — you may kill an intent a note suggested. \
                 They stay on the board, and every read shows them, until he takes one off.{new}"
            )
        });
    }
    if end < total {
        response["facts_next_offset"] = json!(end);
        response["elided"] = json!(format!(
            "{} older facts not shown — read again with offset={end}",
            total - end
        ));
    }
    if frontier_empty && board.status == BoardStatus::Open {
        response["frontier_empty"] = json!(true);
        response["termination"] = json!(if subagent {
            "Nothing is left to claim. Declare what the newest facts suggest, or say in \
             your report why the frontier looks genuinely empty; closing is s0ma's."
        } else {
            "Nothing is left to claim. Either declare what the newest facts suggest, or \
             close the board 'proposed_exhausted' with the argument that the frontier is \
             genuinely empty rather than that you stopped thinking; the operator closes it."
        });
    }
    if board.steps_remaining() <= 3 && board.status == BoardStatus::Open {
        response["budget_warning"] =
            json!(format!("{} steps left of {}", board.steps_remaining(), board.step_budget));
    }
    // Where it is in the operator's workflow, and his review notes.
    if context.subagent.is_none()
        && let Some(workflow) = crate::board::flow::read_extra(context.store, &board)?
    {
        response["workflow"] = workflow;
    }
    Ok(response.to_string())
}

/// `board_link`.
fn execute_link(context: &mut Context<'_>, call: &ToolCall) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        name: String,
        model_id: String,
        relation: String,
        #[serde(default)]
        note: String,
        #[serde(default)]
        remove: bool,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
    };

    let board = board_or_reject!(context, args.name);
    let relation = match LinkRelation::parse(&args.relation) {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };

    if args.remove {
        let removed = context.store.unlink_model(&board.id, &args.model_id, relation)?;
        return Ok(json!({
            "ok": removed,
            "removed": removed,
            "note": if removed {
                "Unlinked. Facts already promoted stay promoted — links are navigational, \
                 not ownership."
            } else {
                "No such link on this board."
            }
        })
        .to_string());
    }

    must_accept_work!(context, board);
    if args.note.chars().count() > MAX_NOTE_CHARS {
        return Ok(rejected(format!("note is too long (at most {MAX_NOTE_CHARS})")));
    }

    let Some(head) = context.models.head(&args.model_id)? else {
        return Ok(rejected(format!(
            "no world model '{}' (see world_model_list)",
            args.model_id
        )));
    };

    context.store.link_model(
        &board.id,
        &args.model_id,
        relation,
        args.note.trim(),
        head.revision,
    )?;

    Ok(json!({
        "ok": true,
        "model_id": args.model_id,
        "title": head.title,
        "relation": relation.label(),
        "revision_seen": head.revision,
        "note": "Linked, not merged. Read it with world_model_get when you need it; this \
                 board flags the link stale if the model is revised. Nothing is promoted \
                 automatically.",
    })
    .to_string())
}

/// `board_intent`.
fn execute_intent(context: &mut Context<'_>, call: &ToolCall, now: DateTime<Local>) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct NewIntent {
        direction: String,
        rationale: String,
        #[serde(default)]
        from_fact_ids: Vec<String>,
        #[serde(default)]
        priority: Option<i64>,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        name: String,
        intents: Vec<NewIntent>,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
    };

    let board = board_or_reject!(context, args.name);
    must_accept_work!(context, board);

    if args.intents.is_empty() || args.intents.len() > MAX_INTENTS_PER_CALL {
        return Ok(rejected(format!(
            "declare between 1 and {MAX_INTENTS_PER_CALL} intents per call"
        )));
    }

    let existing = context.store.intents(&board.id)?;
    let frontier_full = |open_now: usize| {
        rejected(format!(
            "the frontier is at {open_now} of {MAX_FRONTIER}. Explore or kill something \
             before declaring more — a frontier that grows faster than it is explored is \
             a list nobody reads."
        ))
    };
    let open_now = existing.iter().filter(|intent| intent.on_frontier(now)).count();
    if open_now + args.intents.len() > MAX_FRONTIER {
        return Ok(frontier_full(open_now));
    }

    let known: Vec<String> =
        context.store.facts(&board.id)?.into_iter().map(|fact| fact.id).collect();

    // Validate the whole batch before writing any of it: a half-written
    // batch is worse than a rejected one.
    let mut prepared = Vec::new();
    for candidate in &args.intents {
        let direction = match bounded(
            &candidate.direction,
            MIN_TEXT_CHARS,
            MAX_DIRECTION_CHARS,
            "direction",
        ) {
            Ok(value) => value,
            Err(error) => return Ok(rejected(error)),
        };
        let rationale = match bounded(
            &candidate.rationale,
            MIN_TEXT_CHARS,
            MAX_RATIONALE_CHARS,
            "rationale",
        ) {
            Ok(value) => value,
            Err(error) => return Ok(rejected(error)),
        };
        let missing: Vec<&String> =
            candidate.from_fact_ids.iter().filter(|id| !known.contains(id)).collect();
        if !missing.is_empty() {
            return Ok(rejected(format!(
                "from_fact_ids {missing:?} are not facts on this board"
            )));
        }
        prepared.push((
            direction,
            rationale,
            candidate.from_fact_ids.clone(),
            candidate.priority.unwrap_or(5).clamp(0, 9),
        ));
    }

    let mut declared = Vec::new();
    let mut warnings = Vec::new();
    let mut batch = Vec::new();
    for (direction, rationale, from_fact_ids, priority) in prepared {
        for other in &existing {
            if other.state != IntentState::Dead
                && similarity(&direction, &other.direction) >= SIMILARITY_WARN
            {
                warnings.push(format!(
                    "'{direction}' reads a lot like {} ('{}'). If they are the same \
                     direction, kill one.",
                    other.id, other.direction
                ));
            }
        }
        batch.push((direction, rationale, from_fact_ids, priority));
    }
    // Counted again as the batch is written: workers declaring at the
    // same moment can't push the frontier past its cap between them.
    match context.store.declare_intents(&board.id, &batch, MAX_FRONTIER, now)? {
        Ok(intents) => declared.extend(intents.iter().map(|intent| describe_intent(intent, now))),
        Err(open_now) => return Ok(frontier_full(open_now)),
    }

    let mut response = json!({
        "ok": true,
        "declared": declared,
        "frontier": context.store.frontier(&board.id, now)?.len(),
    });
    if !warnings.is_empty() {
        response["warnings"] = json!(warnings);
    }
    Ok(response.to_string())
}

/// `board_claim`.
fn execute_claim(context: &mut Context<'_>, call: &ToolCall, now: DateTime<Local>) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        name: String,
        intent_id: String,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
    };

    let board = board_or_reject!(context, args.name);
    must_accept_work!(context, board);

    // A subagent explores one branch at a time, so a few workers can't
    // take the whole frontier between them.
    if context.subagent.is_some() {
        let held = context.store.intents(&board.id)?.into_iter().find(|intent| {
            intent.state == IntentState::Claimed
                && intent.claimed_by.as_deref() == Some(context.worker)
                && !intent.claim_expired(now)
                && intent.id != args.intent_id.trim()
        });
        if let Some(held) = held {
            return Ok(rejected(format!(
                "you already hold {}; record its fact before claiming another",
                held.id
            )));
        }
    }

    match context.store.claim(&board.id, args.intent_id.trim(), context.worker, now)? {
        Ok(intent) => Ok(json!({
            "ok": true,
            "intent": describe_intent(&intent, now),
            "next": "Explore it with whatever tools the problem needs, then write exactly \
                     one fact saying what you established.",
        })
        .to_string()),
        Err(reason) if reason.starts_with("already_claimed") => Ok(json!({
            "ok": false,
            "reason": "already_claimed",
            "detail": reason,
            "next": "Pick another intent from the frontier rather than retrying this one.",
        })
        .to_string()),
        Err(reason) => Ok(rejected(reason)),
    }
}

/// `board_fact`.
fn execute_fact(context: &mut Context<'_>, call: &ToolCall, now: DateTime<Local>) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        name: String,
        #[serde(default)]
        intent_id: Option<String>,
        statement: String,
        polarity: String,
        basis: String,
        #[serde(default)]
        evidence_event_ids: Vec<u64>,
        /// A subagent's own tool results, in place of event IDs.
        #[serde(default)]
        evidence_refs: Vec<String>,
        #[serde(default)]
        from_fact_ids: Vec<String>,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
    };

    let board = board_or_reject!(context, args.name);
    must_accept_work!(context, board);
    let subagent = context.subagent.is_some();

    let statement =
        match bounded(&args.statement, MIN_TEXT_CHARS, MAX_STATEMENT_CHARS, "statement") {
            Ok(value) => value,
            Err(error) => return Ok(rejected(error)),
        };
    let polarity = match Polarity::parse(&args.polarity) {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };
    let basis = match Basis::parse(&args.basis) {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };

    // The agent cites archive events; a subagent cites its own tool results,
    // which are copied into the archive only once the fact is accepted.
    if subagent {
        if !args.evidence_event_ids.is_empty() {
            return Ok(rejected(
                "cite your own tool results in evidence_refs; archive event IDs are s0ma's to cite",
            ));
        }
        if args.evidence_refs.is_empty() {
            return Ok(rejected(
                "a fact needs evidence — the evidence_ref of each of your tool results that \
                 shows it. Without it this is a thing you said, not a thing the search \
                 established.",
            ));
        }
        if args.evidence_refs.len() > MAX_EVIDENCE_IDS {
            return Ok(rejected(format!("at most {MAX_EVIDENCE_IDS} evidence refs")));
        }
        let unknown = context.subagent.as_deref().map(|refs| refs.unknown(&args.evidence_refs)).unwrap_or_default();
        if !unknown.is_empty() {
            return Ok(rejected(format!(
                "no tool result of yours has evidence_ref {unknown:?}: use the ref a result ends with"
            )));
        }
        if args.intent_id.is_none() {
            return Ok(rejected(
                "name the intent you claimed: a subagent's fact is what its exploration of \
                 that intent showed",
            ));
        }
    } else {
        if !args.evidence_refs.is_empty() {
            return Ok(rejected("evidence_refs are a subagent's; cite archive event IDs"));
        }
        if args.evidence_event_ids.is_empty() {
            return Ok(rejected(
                "a fact needs archive evidence — the event IDs that show it. Without them \
                 this is a thing you said, not a thing the search established.",
            ));
        }
        if let Err(error) = check_evidence(context.archive, &args.evidence_event_ids)? {
            return Ok(rejected(error));
        }
    }

    let known: Vec<String> =
        context.store.facts(&board.id)?.into_iter().map(|fact| fact.id).collect();
    let missing: Vec<&String> =
        args.from_fact_ids.iter().filter(|id| !known.contains(id)).collect();
    if !missing.is_empty() {
        return Ok(rejected(format!(
            "from_fact_ids {missing:?} are not facts on this board"
        )));
    }
    if basis == Basis::Inferred && args.from_fact_ids.is_empty() {
        return Ok(rejected(
            "an inferred fact has to cite the facts it came from in from_fact_ids",
        ));
    }

    // The one-fact-per-exploration discipline enforces itself here: the
    // claim flips to `explored` on the first fact, so a second against
    // the same intent finds nothing to write against.
    let mut attached = None;
    if let Some(intent_id) = args.intent_id.as_deref().map(str::trim) {
        let Some(intent) = context.store.intent(&board.id, intent_id)? else {
            return Ok(rejected(format!("no intent '{intent_id}' on this board")));
        };
        match intent.state {
            IntentState::Claimed
                if intent.claimed_by.as_deref() == Some(context.worker) => {}
            IntentState::Claimed => {
                return Ok(rejected(format!(
                    "{intent_id} is held by {}, not you — claim an intent before writing \
                     a fact against it",
                    intent.claimed_by.as_deref().unwrap_or("another worker")
                )));
            }
            IntentState::Explored => {
                return Ok(rejected(format!(
                    "{intent_id} has already been explored. One fact per exploration — if \
                     you learned something further, it belongs to a new intent."
                )));
            }
            IntentState::Dead => {
                return Ok(rejected(format!("{intent_id} was ruled out")));
            }
            IntentState::Open => {
                return Ok(rejected(format!(
                    "claim {intent_id} with board_claim before exploring it"
                )));
            }
        }
        attached = Some(intent_id.to_string());
    }

    // Everything else checked: only now does a subagent's evidence go
    // into the archive.
    let evidence = match context.subagent.as_deref_mut() {
        Some(refs) => refs.archive(&args.evidence_refs, &mut *context.archive)?,
        None => args.evidence_event_ids.clone(),
    };
    let fact = context.store.record_fact(
        &board.id,
        &statement,
        polarity,
        basis,
        FactAuthor::Model,
        attached.as_deref(),
        &evidence,
        &args.from_fact_ids,
        true,
    )?;

    let after = context.store.board_by_id(&board.id)?.unwrap_or(board);
    let mut response = json!({
        "ok": true,
        "fact": describe_fact(&fact, now),
        "steps": format!("{}/{}", after.steps_used, after.step_budget),
        "frontier": context.store.frontier(&after.id, now)?.len(),
    });
    if attached.is_none() {
        response["note"] = json!(
            "Recorded without an intent, as something learned outside any claimed branch."
        );
    }
    if after.status == BoardStatus::Suspended {
        response["suspended"] =
            json!("That spent the step budget. The graph is intact; ask the operator to extend it.");
    }
    Ok(response.to_string())
}

/// `board_kill`.
fn execute_kill(context: &mut Context<'_>, call: &ToolCall, now: DateTime<Local>) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        name: String,
        intent_id: String,
        fact_id: String,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
    };

    let board = board_or_reject!(context, args.name);
    must_accept_work!(context, board);

    let Some(fact) = context.store.fact(&board.id, args.fact_id.trim())? else {
        return Ok(rejected(format!("no fact '{}' on this board", args.fact_id.trim())));
    };
    if fact.polarity != Polarity::RulesOut {
        return Ok(rejected(format!(
            "{} is a 'supports' fact. Killing a branch needs a fact that rules it out — \
             otherwise this is losing interest, not eliminating.",
            fact.id
        )));
    }

    // A subagent leaves a branch someone else is exploring alone.
    let held_elsewhere = match context.subagent {
        Some(_) => context.store.intent(&board.id, args.intent_id.trim())?.filter(|intent| {
            intent.state == IntentState::Claimed
                && !intent.claim_expired(now)
                && intent.claimed_by.as_deref() != Some(context.worker)
        }),
        None => None,
    };
    if let Some(intent) = held_elsewhere {
        return Ok(rejected(format!(
            "{} is being explored by {}; leave it to them, or say in your report which fact \
             you think rules it out",
            intent.id,
            intent.claimed_by.as_deref().unwrap_or("someone")
        )));
    }
    match context.store.kill_intent(&board.id, args.intent_id.trim(), &fact.id)? {
        Ok(intent) => Ok(json!({
            "ok": true,
            "intent": describe_intent(&intent, now),
            "frontier": context.store.frontier(&board.id, now)?.len(),
        })
        .to_string()),
        Err(error) => Ok(rejected(error)),
    }
}

/// `board_test`.
fn execute_test(context: &mut Context<'_>, call: &ToolCall, now: DateTime<Local>) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        name: String,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
    };

    let board = board_or_reject!(context, args.name);
    must_accept_work!(context, board);

    match board.goal_test_kind {
        GoalTestKind::Judgment => Ok(json!({
            "ok": false,
            "error": if context.subagent.is_some() {
                "this board's goal test is 'judgment' — there is nothing mechanical to \
                 run. Put your argument that it is met in your report; s0ma decides \
                 whether to propose it solved."
            } else {
                "this board's goal test is 'judgment' — there is nothing mechanical \
                 to run. Close it proposed_solved with your argument, and the operator \
                 confirms or doesn't."
            },
        })
        .to_string()),

        GoalTestKind::Observation => Ok(json!({
            "ok": false,
            "error": "this board's goal test is an observation to be checked in tool \
                      output, not a command the harness can run. Produce the observation \
                      and record it as a fact; closing stays proposed_solved until the operator \
                      confirms.",
            "goal_test": board.goal_test,
        })
        .to_string()),

        GoalTestKind::Command => {
            // One run per board at a time: with subagents, two could
            // otherwise start the same test in the same VM and record
            // verdicts that interfered.
            let Some(_running) = TestGuard::take(&format!("{}#{}", context.store.path.display(), board.id)) else {
                return Ok(rejected(format!(
                    "the goal test for '{}' is already running; its verdict will be a fact \
                     on the board (board_read shows it)",
                    board.name
                )));
            };
            // It runs outside a shell: said on the page as it starts,
            // and with its output when it ends, so the operator sees
            // every command it runs.
            let command_line = board.goal_test.split_whitespace().collect::<Vec<_>>().join(" ");
            if crate::vm::access::allowed() {
                crate::terminal::say_status(&format!("[board] {}: goal test running: {command_line}", board.name));
            }
            let run = context.tests.run(&board.goal_test, TEST_TIMEOUT_SECS)?;
            let verdict = run.verdict();
            let passed = run.passed();
            show_test_run(&board.name, &board.goal_test, &run, &verdict);

            // The transcript goes into the archive first, so the fact the
            // harness writes cites something real exactly like every
            // other fact on the board.
            let event_id = context.archive.record(format!(
                "[board:goal_test] board={} kind=command\ncommand:\n{}\nverdict: {}\n\
                 stdout:\n{}\nstderr:\n{}\nduration_ms: {}",
                board.name, board.goal_test, verdict, run.stdout, run.stderr,
                run.duration_ms,
            ))?;

            let statement = if passed {
                format!("Goal test passed: `{}` exited 0.", board.goal_test)
            } else {
                format!("Goal test {verdict}: `{}`.", board.goal_test)
            };
            let fact = context.store.record_fact(
                &board.id,
                &statement,
                if passed { Polarity::Supports } else { Polarity::RulesOut },
                Basis::Observed,
                FactAuthor::Harness,
                None,
                &[event_id],
                &[],
                true,
            )?;

            if passed {
                // Already-closed is not an error worth failing the call
                // over: the verdict and its fact are recorded either way.
                let _closed = context.store.set_status(
                    &board.id,
                    BoardStatus::Solved,
                    &format!("goal test passed; see {}", fact.id),
                )?;
            }

            let after = context.store.board_by_id(&board.id)?.unwrap_or(board);
            let write_up = if passed && after.status == BoardStatus::Solved {
                crate::board::flow::finished(context.store, &after, context.subagent.is_some())?
            } else {
                Value::Null
            };
            Ok(json!({
                "write_up": write_up,
                "ok": true,
                "verdict": verdict,
                "passed": passed,
                "fact": describe_fact(&fact, now),
                "stdout_tail": tail(&run.stdout, 2_000),
                "stderr_tail": tail(&run.stderr, 2_000),
                "board_status": after.status.label(),
                "steps": format!("{}/{}", after.steps_used, after.step_budget),
                "note": if passed {
                    "The harness ran the test and closed the board solved. You did not \
                     score this."
                } else {
                    "Not solved. The failure is now a fact on the board; let it suggest \
                     the next intents."
                },
            })
            .to_string())
        }
    }
}

/// `board_close`.
fn execute_close(context: &mut Context<'_>, call: &ToolCall, now: DateTime<Local>) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Args {
        name: String,
        status: String,
        rationale: String,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
    };

    let board = board_or_reject!(context, args.name);
    if board.status.is_closed() {
        return Ok(rejected(format!(
            "'{}' is already closed ({})",
            board.name,
            board.status.label()
        )));
    }

    let status = match BoardStatus::parse(&args.status) {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };
    if status == BoardStatus::Solved {
        return Ok(rejected(
            "you cannot close a board solved. A passing goal test does that, and on a \
             judgment board the most you can reach is proposed_solved.",
        ));
    }
    if status == BoardStatus::Exhausted {
        return Ok(rejected(
            "you cannot close a board exhausted: the operator does. Close it proposed_exhausted \
             with the argument that nothing is left to try; he closes it, or opens it again.",
        ));
    }
    if !matches!(
        status,
        BoardStatus::ProposedSolved | BoardStatus::ProposedExhausted | BoardStatus::Abandoned
    ) {
        return Ok(rejected("close as proposed_solved, proposed_exhausted or abandoned"));
    }

    let rationale = match bounded(
        &args.rationale,
        MIN_TEXT_CHARS,
        MAX_RATIONALE_CHARS,
        "rationale",
    ) {
        Ok(value) => value,
        Err(error) => return Ok(rejected(error)),
    };

    // Exhausted means the frontier is empty. Mechanically that is not
    // distinguishable from not having thought hard enough, so the least
    // it has to be is true.
    let frontier = context.store.frontier(&board.id, now)?;
    if status == BoardStatus::ProposedExhausted && !frontier.is_empty() {
        return Ok(rejected(format!(
            "{} intents are still on the frontier. Exhausted means nothing is left to \
             try: kill what the facts ruled out, or close 'abandoned' if you are stopping \
             for another reason.",
            frontier.len()
        )));
    }

    match context.store.set_status(&board.id, status, &rationale)? {
        Ok(closed) => {
            let mut response =
                json!({"ok": true, "board": describe_board(&closed, now)});

            // An `updates` link reports what has not reached the model.
            // It reports; it never writes.
            let mut candidates = Vec::new();
            for link in context.store.links(&board.id)? {
                if link.relation == LinkRelation::Updates {
                    let supports: Vec<String> = context
                        .store
                        .facts(&board.id)?
                        .into_iter()
                        .filter(|fact| fact.polarity == Polarity::Supports)
                        .map(|fact| format!("{}: {}", fact.id, fact.statement))
                        .collect();
                    candidates
                        .push(json!({"model_id": link.model_id, "candidate_facts": supports}));
                }
            }
            if !candidates.is_empty() {
                response["for_the_world_model"] = json!(candidates);
                response["promotion_note"] = json!(
                    "Nothing was promoted. These are candidates for world_record_put \
                     citing the fact as evidence — deliberately, because a rules_out fact \
                     from a half-finished search should not quietly become a belief."
                );
            }
            if status == BoardStatus::ProposedExhausted {
                response["awaiting"] = json!(
                    "It is exhausted once the operator agrees: your write-up goes to his review, and he \
                     closes it exhausted from there (or `/board exhaust`), or sends it back to you."
                );
                response["write_up"] = crate::board::flow::finished(context.store, &closed, false)?;
            }
            if status == BoardStatus::ProposedSolved {
                response["awaiting"] = json!(
                    "It counts as solved once the operator has reviewed it and accepted it after verification \
                     (or confirms it with `/board confirm`)."
                );
                response["write_up"] = crate::board::flow::finished(context.store, &closed, false)?;
            }
            Ok(response.to_string())
        }
        Err(error) => Ok(rejected(error)),
    }
}

// ------------------------------------------------------------------ context block

/// First line of the appended board block. Identifies it for stripping, and
/// marks it as harness-rendered state rather than something anyone said.
pub const BLOCK_MARKER: &str = "[boards — current state, rendered by the harness]";

/// First line of an update appended after the block: the boards whose line
/// changed since the block and the updates before it.
pub const UPDATE_MARKER: &str = "[boards — update, rendered by the harness]";

/// One line per open board, and nothing at all when none is open.
///
/// `board_list` is the orientation tool, but a search nobody looks at after a
/// fold is a search that quietly stops. This is the smallest thing that keeps it
/// visible. It is appended at the end of the context once and never moved:
/// what changes after that (a step spent, a lead claimed, a hint) arrives as a
/// short update appended after it (`refresh_context`), so nothing already in
/// the context is rewritten and the KV cache before the end is never lost.
pub fn render(boards: &[(Board, usize, usize)]) -> String {
    let header = "BOARDS — searches I have open. board_read for the graph; nothing here \
                  advances on its own.";
    if boards.is_empty() {
        return format!("{header}\nNone open.");
    }

    // Whole lines only: past the budget, the last boards are counted in the
    // "more" line rather than cut, so no line is ever half a board's.
    let budget = RENDER_BUDGET_TOKENS * BYTES_PER_TOKEN;
    let mut shown = boards.len().min(MAX_RENDERED_BOARDS);
    loop {
        let mut lines: Vec<String> =
            boards.iter().take(shown).map(|(board, frontier, hints)| board_line(board, *frontier, *hints)).collect();
        if boards.len() > shown {
            lines.push(format!("· …and {} more (board_list)", boards.len() - shown));
        }
        let block = format!("{header}\n{}", lines.join("\n"));
        if block.chars().count() <= budget || shown <= 1 {
            return block;
        }
        shown -= 1;
    }
}

/// One board's line in the block, and in an update.
fn board_line(board: &Board, frontier: usize, hints: usize) -> String {
    let mut line =
        format!("· {} — {} [frontier {frontier}, steps {}/{}]", board.name, board.title, board.steps_used, board.step_budget);
    if board.status == BoardStatus::Suspended {
        line.push_str(" — suspended, budget spent");
    }
    if hints > 0 {
        line.push_str(&format!(" — {hints} unread from the operator"));
    }
    line
}

/// The block's text, and every open board's line by name (the block shows at
/// most `MAX_RENDERED_BOARDS`); None when none is open.
/// A board's name, and its line in the block.
type BoardLine = (String, String);

fn current(store: &BoardStore) -> Result<Option<(String, Vec<BoardLine>)>> {
    let open = store.boards(true)?;
    if open.is_empty() {
        return Ok(None);
    }
    let now = Local::now();
    let mut rows = Vec::new();
    for board in open {
        let frontier = store.frontier(&board.id, now)?.len();
        let hints = store.hints(&board.id, true)?.len();
        rows.push((board, frontier, hints));
    }
    let lines = rows.iter().map(|(board, frontier, hints)| (board.name.clone(), board_line(board, *frontier, *hints))).collect();
    Ok(Some((render(&rows), lines)))
}

/// Whether this message is harness-rendered board state (the block or an
/// update after it), not a conversation turn.
pub fn is_block(message: &Message) -> bool {
    message.content.starts_with(BLOCK_MARKER) || message.content.starts_with(UPDATE_MARKER) || crate::audit::ops::is_block(&message.content)
}

/// Take the board block, or an update after it, out before a fold, so a gist
/// never swallows the harness's own words. One at a time: call until false.
pub fn strip_block(messages: &mut Vec<Message>) -> bool {
    match messages.iter().position(is_block) {
        Some(index) => {
            messages.remove(index);
            true
        }
        None => false,
    }
}

/// A board's line in the block, keyed by the board's name (`· name — …`); the
/// line saying more are open than shown is keyed by itself.
fn board_lines(rendered: &str) -> Vec<(String, String)> {
    rendered
        .lines()
        // The "…and N more" line is not a board.
        .filter(|line| line.starts_with("· ") && !line.starts_with("· …") && line.contains(" — "))
        .map(|line| {
            let key = line.trim_start_matches("· ").split(" — ").next().unwrap_or(line).to_string();
            (key, line.to_string())
        })
        .collect()
}

/// What an update says of a board that is no longer open.
const NO_LONGER_OPEN: &str = "no longer open";
/// What an update says when none is.
const NONE_OPEN: &str = "No boards are open now.";

/// The boards as the context says they are: the block's lines with every
/// update after it applied. None when there is no block.
fn known(messages: &[Message]) -> Option<Vec<(String, String)>> {
    let start = messages.iter().position(|message| message.content.starts_with(BLOCK_MARKER))?;
    let mut lines = board_lines(&messages[start].content);
    for message in &messages[start + 1..] {
        if !message.content.starts_with(UPDATE_MARKER) {
            continue;
        }
        if message.content.lines().any(|line| line == NONE_OPEN) {
            lines.clear();
        }
        for (key, line) in board_lines(&message.content) {
            lines.retain(|(known, _)| *known != key);
            if !line.ends_with(NO_LONGER_OPEN) {
                lines.push((key, line));
            }
        }
    }
    Some(lines)
}

/// Keep the boards current at the end of the context without rewriting what
/// is already in it. Returns whether the context changed.
///
/// The first time a board is open, the block goes at the end. After that the
/// block stays where it is, however far back the conversation leaves it, and
/// a change is appended as an update naming only the boards whose line
/// changed (or that closed). Moving the block instead -- which this did --
/// took it out from wherever it sat, and everything after it had to be
/// prefilled again: with workers spending steps and claiming leads, nearly
/// every call. A fold strips the block and its updates together and puts back
/// one fresh block.
pub fn refresh_context(messages: &mut Vec<Message>, store: &BoardStore) -> Result<bool> {
    // The audits' block is kept the same way, beside this one.
    let audits = crate::audit::ops::refresh_context(messages).unwrap_or(false);
    Ok(refresh_boards(messages, store)? || audits)
}

fn refresh_boards(messages: &mut Vec<Message>, store: &BoardStore) -> Result<bool> {
    let fresh = current(store)?;
    let Some(known) = known(messages) else {
        return Ok(match fresh {
            Some((rendered, _)) => {
                messages.push(Message::user(format!("{BLOCK_MARKER}\n{rendered}")));
                true
            }
            None => false,
        });
    };
    // Every open board, not only those the block had room for: a board the
    // block left out is still open, and is said once, the first time.
    let fresh_lines = fresh.map(|(_, lines)| lines).unwrap_or_default();
    if fresh_lines.is_empty() {
        if known.is_empty() {
            return Ok(false);
        }
        messages.push(Message::user(format!("{UPDATE_MARKER}\n{NONE_OPEN}")));
        return Ok(true);
    }
    let mut changed: Vec<String> = fresh_lines
        .iter()
        .filter(|(key, line)| !known.iter().any(|(k, l)| k == key && l == line))
        .map(|(_, line)| line.clone())
        .collect();
    for (key, _) in &known {
        if !fresh_lines.iter().any(|(k, _)| k == key) {
            changed.push(format!("· {key} — {NO_LONGER_OPEN}"));
        }
    }
    if changed.is_empty() {
        return Ok(false);
    }
    messages.push(Message::user(format!("{UPDATE_MARKER}\n{}", changed.join("\n"))));
    Ok(true)
}

// ------------------------------------------------------------------ operator

pub const OPERATOR_HELP: &str = "usage: /board [list|all]\n       \
     /board show <name> [facts]\n       \
     /board hint <name> <text>    (also: /hint <name> <text>)\n       \
     /board unhint <name> <hint>  (also: /unhint; h2 or hint-0002)\n       \
     /board extend <name> <steps>\n       \
     /board confirm <name>\n       \
     /board reject <name> [why]   (a proposal: open again)\n       \
     /board exhaust <name>        (s0ma's proposed_exhausted: closed exhausted)\n       \
     /board abandon <name> <why>\n       \
     /board reuse <name>";

/// The operator's decisions on proposed solutions, for the agent with his next message.
static DECIDED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

fn decided(line: String) {
    DECIDED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(line);
}

/// What the operator decided on proposed solutions since last asked, once.
pub fn take_note() -> Option<String> {
    let lines = std::mem::take(&mut *DECIDED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
    (!lines.is_empty()).then(|| format!("\n\n[BOARDS]\nThe operator, on proposed solutions:\n{}", lines.join("\n")))
}

/// A hint named as the operator would: `hint-0002`, `h2` or `2`.
fn hint_id(which: &str) -> Option<String> {
    let number = which.strip_prefix("hint-").or_else(|| which.strip_prefix('h')).unwrap_or(which);
    number.parse::<u64>().ok().map(|number| format!("hint-{number:04}"))
}

/// `/board …` and `/hint …`. Returns the lines to print.
///
/// Hints land here and only here: nothing in the tool surface writes one, which
/// is what makes the direction one-way by construction rather than by rule.
pub fn handle_operator_command(store: &BoardStore, command: &str) -> Result<String> {
    let parts: Vec<&str> = command.split_whitespace().collect();
    let now = Local::now();

    match parts.as_slice() {
        ["/board"] | ["/board", "list"] | ["/board", "all"] => {
            let all = parts.last() == Some(&"all");
            let boards = store.boards(!all)?;
            if boards.is_empty() {
                return Ok(format!("[board] no {}boards", if all { "" } else { "open " }));
            }
            let mut out = format!("[board] {} board(s)", boards.len());
            for board in boards {
                let hints = store.hints(&board.id, true)?.len();
                out.push_str(&format!(
                    "\n  {} [{}] {} — frontier {}, {} facts, steps {}/{}{}",
                    board.name,
                    board.status.label(),
                    board.title,
                    store.frontier(&board.id, now)?.len(),
                    store.facts(&board.id)?.len(),
                    board.steps_used,
                    board.step_budget,
                    if hints > 0 { format!(", {hints} unread hint(s)") } else { String::new() },
                ));
            }
            Ok(out)
        }

        ["/board", "show", name] | ["/board", "show", name, "facts"] => {
            let with_facts = parts.len() == 4;
            let Some(board) = store.board(name)? else {
                return Ok(format!("[board] no board named '{name}'"));
            };
            let mut out = format!(
                "[board] {} [{}]\n  {}\n  goal: {}\n  test ({}): {}\n  steps {}/{}",
                board.name,
                board.status.label(),
                board.title,
                board.goal,
                board.goal_test_kind.label(),
                board.goal_test,
                board.steps_used,
                board.step_budget,
            );
            if let Some(outcome) = &board.outcome {
                out.push_str(&format!("\n  outcome: {outcome}"));
            }

            let hints = store.hints(&board.id, false)?;
            if !hints.is_empty() {
                out.push_str("\n  hints:");
                for hint in hints {
                    out.push_str(&format!(
                        "\n    {} {} [{}]",
                        hint.id,
                        hint.text,
                        if hint.consumed_at.is_some() { "read" } else { "unread" },
                    ));
                }
            }

            let links = store.links(&board.id)?;
            if !links.is_empty() {
                out.push_str("\n  models:");
                for link in links {
                    out.push_str(&format!(
                        "\n    {} [{}] rev {} seen, linked {}",
                        link.model_id,
                        link.relation.label(),
                        link.revision_seen,
                        relative_age(&link.linked_at, now),
                    ));
                }
            }

            let frontier = store.frontier(&board.id, now)?;
            out.push_str(&format!("\n  frontier ({}):", frontier.len()));
            for intent in &frontier {
                out.push_str(&format!(
                    "\n    {} (p{}) {}",
                    intent.id, intent.priority, intent.direction
                ));
            }

            let facts = store.facts(&board.id)?;
            out.push_str(&format!("\n  facts ({}):", facts.len()));
            let shown = if with_facts { facts.len() } else { 10.min(facts.len()) };
            for fact in facts.iter().take(shown) {
                out.push_str(&format!(
                    "\n    {} [{}/{}] {}",
                    fact.id,
                    fact.polarity.label(),
                    fact.author.label(),
                    fact.statement
                ));
            }
            if shown < facts.len() {
                out.push_str(&format!(
                    "\n    … {} more (/board show {name} facts)",
                    facts.len() - shown
                ));
            }
            Ok(out)
        }

        ["/board", "hint", name, rest @ ..] | ["/hint", name, rest @ ..] => {
            let text = match bounded(&rest.join(" "), 1, MAX_HINT_CHARS, "hint") {
                Ok(text) => text,
                Err(error) => return Ok(format!("[board] {error}")),
            };
            let Some(board) = store.board(name)? else {
                return Ok(format!("[board] no board named '{name}'"));
            };
            if board.status.is_closed() {
                return Ok(format!("[board] '{name}' is closed ({})", board.status.label()));
            }
            let hint = store.add_hint(&board.id, &text)?;
            Ok(format!(
                "[board] {} left on {name}; it stays on the board, and s0ma sees it on every \
                 board_read, new on its next. The current exploration is not interrupted.",
                hint.id
            ))
        }

        ["/board", "unhint", name, which] | ["/unhint", name, which] => {
            let Some(board) = store.board(name)? else {
                return Ok(format!("[board] no board named '{name}'"));
            };
            let Some(id) = hint_id(which) else {
                return Ok(format!("[board] '{which}' is not a hint: give h2 or hint-0002"));
            };
            if store.remove_hint(&board.id, &id)? {
                Ok(format!("[board] {id} taken off {name}; s0ma's next board_read no longer shows it."))
            } else {
                Ok(format!("[board] {name} has no {id}"))
            }
        }

        ["/board", "extend", name, steps] => {
            let Ok(extra) = steps.parse::<i64>() else {
                return Ok("[board] steps must be a number".to_string());
            };
            if !(1..=MAX_STEP_BUDGET).contains(&extra) {
                return Ok(format!("[board] extend by 1..{MAX_STEP_BUDGET} steps"));
            }
            match store.extend_budget(name, extra)? {
                Ok(board) => Ok(format!(
                    "[board] {} now at steps {}/{} [{}]",
                    board.name,
                    board.steps_used,
                    board.step_budget,
                    board.status.label()
                )),
                Err(error) => Ok(format!("[board] {error}")),
            }
        }

        ["/board", "confirm", name] => match store.confirm_solved(name)? {
            Ok(board) => {
                crate::board::flow::confirmed(store, &board)?;
                decided(format!("- confirmed '{}' solved", board.name));
                Ok(format!("[board] {} confirmed solved", board.name))
            }
            Err(error) => Ok(format!("[board] {error}")),
        },

        ["/board", "exhaust", name] => match store.confirm_exhausted(name)? {
            Ok(board) => {
                // Out of review too, if it was there: kept with the deliverables.
                crate::board::flow::confirmed(store, &board)?;
                decided(format!("- closed '{}' exhausted, as you proposed", board.name));
                Ok(format!("[board] {} closed exhausted", board.name))
            }
            Err(error) => Ok(format!("[board] {error}")),
        },

        ["/board", "reject", name, rest @ ..] => {
            let why = rest.join(" ");
            match store.reject_solved(name, &why)? {
                Ok(board) => {
                    let said = if why.trim().is_empty() { String::new() } else { format!(" He said: {}", why.trim()) };
                    let budget = if board.steps_used >= board.step_budget {
                        " Its steps are spent: the operator can /board extend it."
                    } else {
                        ""
                    };
                    decided(format!(
                        "- rejected your proposal on '{}': it is open again, with his note on it.{said}{budget}",
                        board.name
                    ));
                    Ok(format!(
                        "[board] {} is open again; s0ma sees why on its next board_read{}",
                        board.name,
                        if budget.is_empty() { "" } else { " (its steps are spent: /board extend it)" }
                    ))
                }
                Err(error) => Ok(format!("[board] {error}")),
            }
        }

        ["/board", "abandon", name, rest @ ..] => {
            let Some(board) = store.board(name)? else {
                return Ok(format!("[board] no board named '{name}'"));
            };
            match store.set_status(&board.id, BoardStatus::Abandoned, &rest.join(" "))? {
                Ok(board) => Ok(format!("[board] {} abandoned", board.name)),
                Err(error) => Ok(format!("[board] {error}")),
            }
        }

        ["/board", "reuse", name] => {
            store.allow_name_reuse(name)?;
            Ok(format!(
                "[board] '{name}' may be reused once. References to the old board in the archive \
                 become ambiguous when it is."
            ))
        }

        _ => Ok(OPERATOR_HELP.to_string()),
    }
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionCall;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct FakeArchive {
        ids: Vec<u64>,
        next: u64,
    }

    impl BoardArchive for FakeArchive {
        fn events(&self, ids: &[u64]) -> Result<Vec<EventSnapshot>> {
            Ok(ids
                .iter()
                .filter(|id| self.ids.contains(id))
                .map(|id| EventSnapshot { id: *id })
                .collect())
        }

        fn record(&mut self, _content: String) -> Result<u64> {
            self.next += 1;
            self.ids.push(self.next);
            Ok(self.next)
        }
    }

    struct FakeModels {
        models: RefCell<Vec<(String, String, u64)>>,
    }

    impl ModelRevisions for FakeModels {
        fn head(&self, model_id: &str) -> Result<Option<ModelHead>> {
            Ok(self
                .models
                .borrow()
                .iter()
                .find(|(id, _, _)| id == model_id)
                .map(|(_, title, revision)| ModelHead {
                    title: title.clone(),
                    revision: *revision,
                }))
        }
    }

    struct FakeTests {
        exit: RefCell<i64>,
    }

    impl TestRunner for FakeTests {
        fn run(&self, _command: &str, _timeout_secs: u64) -> Result<TestRun> {
            Ok(TestRun {
                exit_code: Some(*self.exit.borrow()),
                stdout: "ok".to_string(),
                stderr: String::new(),
                duration_ms: 4,
                timed_out: false,
            })
        }
    }

    struct Fixture {
        store: BoardStore,
        archive: FakeArchive,
        models: FakeModels,
        tests: FakeTests,
    }

    impl Fixture {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "s0ma-board-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            std::fs::create_dir_all(&root).expect("test directory");
            Self {
                store: BoardStore::beside_archive(&root.join("hippocampus.jsonl")),
                archive: FakeArchive { ids: vec![10, 11, 12], next: 1_000 },
                models: FakeModels {
                    models: RefCell::new(vec![(
                        "storage-subsystem".to_string(),
                        "The storage layer".to_string(),
                        12,
                    )]),
                },
                tests: FakeTests { exit: RefCell::new(1) },
            }
        }

        fn call_as(&mut self, worker: &str, name: &str, arguments: Value) -> Value {
            let call = ToolCall {
                id: "call-1".to_string(),
                kind: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
                },
            };
            let mut context = Context {
                store: &self.store,
                archive: &mut self.archive,
                models: &self.models,
                tests: &self.tests,
                worker,
                subagent: None,
            };
            let raw = execute(&mut context, &call).expect("tool executes");
            serde_json::from_str(&raw).expect("tool returns JSON")
        }

        fn call(&mut self, name: &str, arguments: Value) -> Value {
            self.call_as("s0ma", name, arguments)
        }

        /// A board with a `command` goal test, one intent, claimed.
        fn started(&mut self) -> String {
            self.call(
                "board_open",
                json!({
                    "name": "intermittent-flush-failure",
                    "title": "Flush fails sometimes under load",
                    "origin": "The 2.2 build, failing roughly one run in forty.",
                    "goal": "The flush test suite is green across fifty consecutive runs.",
                    "goal_test": "cargo test --test flush -- --repeat 50",
                    "goal_test_kind": "command",
                    "step_budget": 10
                }),
            );
            let declared = self.call(
                "board_intent",
                json!({"name": "intermittent-flush-failure", "intents": [{
                    "direction": "check whether the failure correlates with input size",
                    "rationale": "The failures cluster in the large-batch CI job."
                }]}),
            );
            declared["declared"][0]["id"].as_str().unwrap().to_string()
        }
    }

    fn ok(value: &Value) -> bool {
        value["ok"] == json!(true)
    }

    fn error(value: &Value) -> String {
        value["error"].as_str().unwrap_or_default().to_string()
    }

    #[test]
    fn a_fact_without_resolvable_evidence_is_refused_and_nothing_is_stored() {
        let mut fx = Fixture::new();
        let intent = fx.started();
        fx.call(
            "board_claim",
            json!({"name": "intermittent-flush-failure", "intent_id": intent}),
        );

        let ghost = fx.call(
            "board_fact",
            json!({"name": "intermittent-flush-failure", "intent_id": intent,
                   "statement": "It only fails above 4096 rows.",
                   "polarity": "supports", "basis": "observed",
                   "evidence_event_ids": [10, 99]}),
        );
        assert!(error(&ghost).contains("don't exist"), "{ghost}");

        let board = fx.store.board("intermittent-flush-failure").unwrap().unwrap();
        assert!(fx.store.facts(&board.id).unwrap().is_empty(), "nothing stored on rejection");
        assert_eq!(board.steps_used, 0, "a rejected fact spends no step");

        let empty = fx.call(
            "board_fact",
            json!({"name": "intermittent-flush-failure", "intent_id": intent,
                   "statement": "It only fails above 4096 rows.",
                   "polarity": "supports", "basis": "observed",
                   "evidence_event_ids": []}),
        );
        assert!(error(&empty).contains("a thing you said"), "{empty}");
    }

    #[test]
    fn a_claim_is_atomic_and_a_second_worker_is_told_to_pick_another() {
        let mut fx = Fixture::new();
        let intent = fx.started();

        let mine = fx.call_as("s0ma", "board_claim",
            json!({"name": "intermittent-flush-failure", "intent_id": intent}));
        assert!(ok(&mine), "{mine}");

        let theirs = fx.call_as("worker-3", "board_claim",
            json!({"name": "intermittent-flush-failure", "intent_id": intent}));
        assert!(!ok(&theirs));
        assert_eq!(theirs["reason"], json!("already_claimed"));
        assert!(theirs["next"].as_str().unwrap().contains("another"));

        // And a worker that did not claim it cannot write against it.
        let stolen = fx.call_as(
            "worker-3",
            "board_fact",
            json!({"name": "intermittent-flush-failure", "intent_id": intent,
                   "statement": "Something I did not actually explore.",
                   "polarity": "supports", "basis": "observed", "evidence_event_ids": [10]}),
        );
        assert!(error(&stolen).contains("not you"), "{stolen}");
    }

    #[test]
    fn the_model_cannot_score_its_own_search() {
        let mut fx = Fixture::new();
        fx.started();

        let claimed = fx.call(
            "board_close",
            json!({"name": "intermittent-flush-failure", "status": "solved",
                   "rationale": "I am confident this is fixed now."}),
        );
        assert!(error(&claimed).contains("cannot close a board solved"), "{claimed}");

        // Nothing in the schema offers it either.
        let schema = tool_definitions()
            .into_iter()
            .find(|tool| tool.function.name == "board_close")
            .unwrap();
        let allowed = schema.function.parameters["properties"]["status"]["enum"].clone();
        assert_eq!(allowed, json!(["proposed_solved", "proposed_exhausted", "abandoned"]));
    }

    #[test]
    fn a_passing_goal_test_is_run_by_the_harness_and_closes_the_board() {
        let mut fx = Fixture::new();
        fx.started();
        *fx.tests.exit.borrow_mut() = 0;

        let tested = fx.call("board_test", json!({"name": "intermittent-flush-failure"}));
        assert!(ok(&tested), "{tested}");
        assert_eq!(tested["passed"], json!(true));
        assert_eq!(tested["board_status"], json!("solved"));

        // The verdict is a fact the harness wrote, citing an archive event it
        // recorded first — not something the model asserted.
        assert_eq!(tested["fact"]["author"], json!("harness"));
        assert_eq!(tested["fact"]["polarity"], json!("supports"));
        assert!(!tested["fact"]["evidence_event_ids"].as_array().unwrap().is_empty());
    }

    #[test]
    fn near_duplicate_intents_are_warned_about_but_still_written() {
        let mut fx = Fixture::new();
        fx.started();
        let similar = fx.call(
            "board_intent",
            json!({"name": "intermittent-flush-failure", "intents": [{
                "direction": "check whether the failure correlates with input size somehow",
                "rationale": "Worth confirming the size relationship independently."}]}),
        );
        assert!(ok(&similar), "a warning is not a refusal: {similar}");
        assert!(similar["warnings"][0].as_str().unwrap().contains("reads a lot like"));
    }

    #[test]
    fn a_linked_model_goes_stale_when_it_is_revised() {
        let mut fx = Fixture::new();
        fx.started();

        let linked = fx.call(
            "board_link",
            json!({"name": "intermittent-flush-failure", "model_id": "storage-subsystem",
                   "relation": "background", "note": "What s0ma already believes about flushing."}),
        );
        assert!(ok(&linked), "{linked}");
        assert_eq!(linked["revision_seen"], json!(12));

        let fresh = fx.call("board_read", json!({"name": "intermittent-flush-failure"}));
        assert!(fresh["linked_models"][0]["stale"].is_null());
        assert!(fresh["linked_models"][0]["linked"].as_str().unwrap().ends_with("ago"));

        fx.models.models.borrow_mut()[0].2 = 13;
        let stale = fx.call("board_read", json!({"name": "intermittent-flush-failure"}));
        assert_eq!(stale["linked_models"][0]["stale"], json!(true));
        assert!(stale["linked_models"][0]["staleness"].as_str().unwrap().contains("not make them false"));

        let missing = fx.call(
            "board_link",
            json!({"name": "intermittent-flush-failure", "model_id": "no-such-model",
                   "relation": "background", "note": "Wishful."}),
        );
        assert!(error(&missing).contains("no world model"), "{missing}");
    }

    #[test]
    fn a_closed_board_refuses_work_but_keeps_its_name_resolvable() {
        let mut fx = Fixture::new();
        fx.started();
        fx.call(
            "board_close",
            json!({"name": "intermittent-flush-failure", "status": "abandoned",
                   "rationale": "Superseded by the batching rewrite."}),
        );

        let write = fx.call(
            "board_fact",
            json!({"name": "intermittent-flush-failure", "statement": "One more thought.",
                   "polarity": "supports", "basis": "observed", "evidence_event_ids": [10]}),
        );
        assert!(error(&write).contains("closed"), "{write}");

        let read = fx.call("board_read", json!({"name": "intermittent-flush-failure"}));
        assert!(ok(&read), "the archive still points here, so it must still read");
        assert_eq!(read["board"]["status"], json!("abandoned"));
    }

    #[test]
    fn a_board_the_block_has_no_room_for_is_never_said_closed() {
        let mut fx = Fixture::new();
        fx.started();
        let open = |fx: &mut Fixture, n: usize| {
            fx.call(
                "board_open",
                json!({"name": format!("search-{n}"), "title": format!("Search {n}"), "origin": "Another thing to find.",
                       "goal": "It is found.", "goal_test": "cargo test --test it",
                       "goal_test_kind": "command", "step_budget": 10}),
            );
        };
        for n in 1..=4 {
            open(&mut fx, n);
        }
        let mut messages = vec![Message::system("prefix".into())];
        refresh_context(&mut messages, &fx.store).unwrap();
        // A sixth: the oldest drops out of what the block would show.
        open(&mut fx, 5);
        assert!(refresh_context(&mut messages, &fx.store).unwrap());
        let last = &messages.last().unwrap().content;
        assert!(last.contains("search-5"), "{last}");
        assert!(!last.contains(NO_LONGER_OPEN), "every board is still open: {last}");
        assert!(!refresh_context(&mut messages, &fx.store).unwrap());
    }
}
