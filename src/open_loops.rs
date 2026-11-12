//! Open loops: the agent's unfinished business.
//!
//! Everything else in working memory ages by context distance — the further back
//! something is, the coarser it gets. Loops are the exception. An unanswered
//! question, a half-done task, a promise to check something: these fade only when
//! they are resolved, because an intention that quietly got coarser is an
//! intention that got dropped.
//!
//! There is no mode and no gate. An earlier version had a "work mode" that
//! refused to run VM, browser, desktop or Minecraft tools until a loop was in
//! focus. Three things were wrong with it. A refusal mid-task is the shape that
//! produces hedging — the giving-up failure the persistence rows exist to prevent —
//! spent on bookkeeping rather than safety. It forced a loop change at the start
//! of every work episode, which is the most expensive possible moment to edit
//! context. And it wasn't how anyone actually works: you don't declare that you
//! are working, you just have unfinished things and keep them where you can see
//! them.
//!
//! So this is a list, kept in front of the agent, that only it can cross anything off.
//! It sits at the *end* of the context and moves only when its text changes (see
//! `refresh_context`), it is bounded by a cap that degrades detail rather than
//! hiding entries, and a loop untouched across a lot of context starts asking
//! whether it is still open.
//!
//! There is no explicit focus either. The most recently touched loop renders
//! first and in full, which is the same information without a flag to maintain:
//! what you last worked on is what is on your mind.
//!
//! Storage is a single sqlite table beside the archive, in the same style as
//! `world_model.rs` and continuous compaction.

use crate::llm::ToolDefinition;
use crate::types::{Message, ToolCall};
use crate::evidence::EvidenceSource;

use chrono::Local;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use std::error::Error;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Render budget for the whole loops block, in tokens.
pub const RENDER_BUDGET_TOKENS: usize = crate::context::guard::LOOPS_BUDGET_TOKENS;

/// A loop untouched across this much context gets a "still open?" marker. It is a
/// prompt to pick it up, close it or drop it — never a reason to hide it.
pub const STALE_DISTANCE_TOKENS: u64 = 300_000;

/// Bytes per token for the render estimate. It only decides how much detail fits,
/// never what is kept.
const BYTES_PER_TOKEN: usize = 4;

const MAX_TEXT_CHARS: usize = 400;
const MIN_TEXT_CHARS: usize = 8;
const MAX_SOURCE_IDS: usize = 20;

// ------------------------------------------------------------------ records

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoopKind {
    /// Something the operator asked for.
    Request,
    /// Work started and not finished.
    Task,
    /// Something the agent said it would do or check.
    Promise,
    /// A question the agent left open for itself.
    Question,
}

impl LoopKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Task => "task",
            Self::Promise => "promise",
            Self::Question => "question",
        }
    }

    fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "request" => Ok(Self::Request),
            "task" => Ok(Self::Task),
            "promise" => Ok(Self::Promise),
            "question" => Ok(Self::Question),
            other => Err(format!("unknown loop kind '{other}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoopStatus {
    Open,
    Done,
    Dropped,
}

impl LoopStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Done => "done",
            Self::Dropped => "dropped",
        }
    }

    fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "open" => Ok(Self::Open),
            "done" => Ok(Self::Done),
            "dropped" => Ok(Self::Dropped),
            other => Err(format!("unknown loop status '{other}'")),
        }
    }
}

/// Who opened the loop. A fold-opened loop is one agent's own gist noticed was
/// still hanging, which is the safety net for work it forgot to write down.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenedBy {
    S0ma,
    Fold,
}

impl OpenedBy {
    pub fn label(self) -> &'static str {
        match self {
            Self::S0ma => "s0ma",
            Self::Fold => "fold",
        }
    }

    fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "s0ma" => Ok(Self::S0ma),
            "fold" => Ok(Self::Fold),
            other => Err(format!("unknown opener '{other}'")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenLoop {
    pub id: String,
    pub text: String,
    pub kind: LoopKind,
    #[serde(default)]
    pub source_event_ids: Vec<u64>,
    pub opened_by: OpenedBy,
    pub status: LoopStatus,
    pub opened_at: String,
    pub last_touched_at: String,
    /// The distance odometer's reading when this loop was last referred to or
    /// updated. Distance *since* then is `now - last_touched_distance`, and it is
    /// also the render order: most recently touched first.
    pub last_touched_distance: u64,
    #[serde(default)]
    pub resolution: Option<String>,
}

impl OpenLoop {
    pub fn first_line(&self) -> &str {
        self.text.lines().next().unwrap_or("").trim()
    }

    /// Context distance since this loop was last touched.
    pub fn distance_since(&self, now: u64) -> u64 {
        now.saturating_sub(self.last_touched_distance)
    }

    pub fn is_stale(&self, now: u64) -> bool {
        self.distance_since(now) >= STALE_DISTANCE_TOKENS
    }
}

// ------------------------------------------------------------------ store

pub struct LoopStore {
    path: PathBuf,
}

impl LoopStore {
    pub fn beside_archive(archive: &Path) -> Self {
        Self { path: archive.with_file_name("open_loops.sqlite3") }
    }

    fn connect(&self) -> Result<Connection> {
        let db = crate::db::open(&self.path)?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS loops(
                id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL,
                text TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_event_ids TEXT NOT NULL,
                opened_by TEXT NOT NULL,
                status TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                last_touched_at TEXT NOT NULL,
                last_touched_distance INTEGER NOT NULL,
                resolution TEXT);
             CREATE TABLE IF NOT EXISTS loop_state(key TEXT PRIMARY KEY, value TEXT NOT NULL);",
        )?;
        Ok(db)
    }

    fn row(row: &rusqlite::Row<'_>) -> rusqlite::Result<OpenLoop> {
        let ids: String = row.get(3)?;
        let distance: i64 = row.get(8)?;
        Ok(OpenLoop {
            id: row.get(0)?,
            text: row.get(1)?,
            kind: LoopKind::parse(&row.get::<_, String>(2)?).unwrap_or(LoopKind::Task),
            source_event_ids: serde_json::from_str(&ids).unwrap_or_default(),
            opened_by: OpenedBy::parse(&row.get::<_, String>(4)?).unwrap_or(OpenedBy::S0ma),
            status: LoopStatus::parse(&row.get::<_, String>(5)?).unwrap_or(LoopStatus::Open),
            opened_at: row.get(6)?,
            last_touched_at: row.get(7)?,
            last_touched_distance: distance.max(0) as u64,
            resolution: row.get(9)?,
        })
    }

    const SELECT: &'static str = "SELECT id,text,kind,source_event_ids,opened_by,status,\
        opened_at,last_touched_at,last_touched_distance,resolution FROM loops";

    /// Every loop, most recently touched first — which is render order.
    pub fn all(&self) -> Result<Vec<OpenLoop>> {
        let db = self.connect()?;
        let sql = format!("{} ORDER BY last_touched_distance DESC, seq DESC", Self::SELECT);
        let mut statement = db.prepare(&sql)?;
        let rows = statement.query_map([], Self::row)?;
        let mut loops = Vec::new();
        for entry in rows {
            loops.push(entry?);
        }
        Ok(loops)
    }

    /// The loops that are actually open, most recently touched first.
    pub fn open_loops(&self) -> Result<Vec<OpenLoop>> {
        Ok(self.all()?.into_iter().filter(|entry| entry.status == LoopStatus::Open).collect())
    }

    pub fn get(&self, id: &str) -> Result<Option<OpenLoop>> {
        let db = self.connect()?;
        let sql = format!("{} WHERE id=?1", Self::SELECT);
        Ok(db.query_row(&sql, params![id], Self::row).optional()?)
    }

    fn next_id(&self, db: &Connection) -> Result<(String, i64)> {
        let seq: i64 =
            db.query_row("SELECT COALESCE(MAX(seq),0)+1 FROM loops", [], |row| row.get(0))?;
        Ok((format!("loop-{seq:04}"), seq))
    }

    // ---- the distance odometer ----

    fn state_value(&self, db: &Connection, key: &str) -> Result<Option<String>> {
        Ok(db
            .query_row("SELECT value FROM loop_state WHERE key=?1", params![key], |row| row.get(0))
            .optional()?)
    }

    fn set_state(&self, db: &Connection, key: &str, value: &str) -> Result<()> {
        db.execute(
            "INSERT INTO loop_state VALUES(?1,?2) \
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Cumulative context volume seen so far.
    pub fn distance(&self) -> Result<u64> {
        let db = self.connect()?;
        Ok(self.state_value(&db, "distance")?.and_then(|value| value.parse().ok()).unwrap_or(0))
    }

    /// Add to the odometer.
    ///
    /// Distance is *cumulative context volume*, not the current context size, so
    /// it has to be accumulated rather than sampled: the live context tops out
    /// around 220k and then drops at every fold, while the odometer must keep
    /// climbing for the whole session. Two things feed it — the growth between
    /// turns, and, once folding lands, the amount each fold removes. Without the
    /// second, every fold silently loses that much distance and loops look
    /// younger than they are.
    pub fn advance_distance(&self, by: u64) -> Result<u64> {
        if by == 0 {
            return self.distance();
        }
        let db = self.connect()?;
        let current: u64 =
            self.state_value(&db, "distance")?.and_then(|value| value.parse().ok()).unwrap_or(0);
        let next = current.saturating_add(by);
        self.set_state(&db, "distance", &next.to_string())?;
        Ok(next)
    }

    // ---- mutation ----

    pub fn open(
        &self,
        text: &str,
        kind: LoopKind,
        source_event_ids: &[u64],
        opened_by: OpenedBy,
        distance: u64,
    ) -> Result<OpenLoop> {
        let db = self.connect()?;
        let (id, seq) = self.next_id(&db)?;
        let now = Local::now().to_rfc3339();

        let entry = OpenLoop {
            id,
            text: text.trim().to_string(),
            kind,
            source_event_ids: source_event_ids.to_vec(),
            opened_by,
            status: LoopStatus::Open,
            opened_at: now.clone(),
            last_touched_at: now,
            last_touched_distance: distance,
            resolution: None,
        };

        db.execute(
            "INSERT INTO loops VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![
                entry.id,
                seq,
                entry.text,
                entry.kind.label(),
                serde_json::to_string(&entry.source_event_ids)?,
                entry.opened_by.label(),
                entry.status.label(),
                entry.opened_at,
                entry.last_touched_at,
                entry.last_touched_distance as i64,
                entry.resolution,
            ],
        )?;

        Ok(entry)
    }

    /// Mark an open loop as being worked on again: resets its age and puts it
    /// back at the top of the list.
    ///
    /// This is what an explicit focus flag used to be, without the flag. Only
    /// open loops can be touched — picking a closed one back up means opening a
    /// new loop, so that the decision to reopen is visible rather than implicit.
    pub fn touch(&self, id: &str, distance: u64) -> Result<std::result::Result<OpenLoop, String>> {
        let Some(entry) = self.get(id)? else {
            return Ok(Err(format!("no loop with id '{id}' (see loop_list)")));
        };
        if entry.status != LoopStatus::Open {
            return Ok(Err(format!(
                "{id} is {} — open a new loop rather than reviving a closed one",
                entry.status.label()
            )));
        }

        let db = self.connect()?;
        db.execute(
            "UPDATE loops SET last_touched_at=?2, last_touched_distance=?3 WHERE id=?1",
            params![id, Local::now().to_rfc3339(), distance as i64],
        )?;
        Ok(Ok(self.get(id)?.unwrap_or(entry)))
    }

    /// Close a loop. The only way a loop ever leaves the open set: "I think that's
    /// done" is exactly the judgment the loop exists to keep visible, so nothing
    /// closes automatically and nothing closes on elapsed time or distance.
    pub fn close(
        &self,
        id: &str,
        status: LoopStatus,
        resolution: &str,
        distance: u64,
    ) -> Result<std::result::Result<OpenLoop, String>> {
        if status == LoopStatus::Open {
            return Ok(Err("close a loop as 'done' or 'dropped'".to_string()));
        }
        let Some(entry) = self.get(id)? else {
            return Ok(Err(format!("no loop with id '{id}' (see loop_list)")));
        };
        if entry.status != LoopStatus::Open {
            return Ok(Err(format!("{id} is already {}", entry.status.label())));
        }

        let db = self.connect()?;
        db.execute(
            "UPDATE loops SET status=?2, resolution=?3, last_touched_at=?4, \
             last_touched_distance=?5 WHERE id=?1",
            params![
                id,
                status.label(),
                resolution.trim(),
                Local::now().to_rfc3339(),
                distance as i64,
            ],
        )?;

        Ok(Ok(self.get(id)?.unwrap_or(entry)))
    }
}

// ------------------------------------------------------------------ rendering

fn truncate(text: &str, chars: usize) -> String {
    let mut out: String = text.chars().take(chars).collect();
    if text.chars().count() > chars {
        out.push('…');
    }
    out
}

/// "opened Wed 21 Oct 2026 09:10", for the block. Only when it opened, which
/// never changes: the block moves only when its text changes, and a touch
/// that changed the text would move it (and re-prefill everything after where
/// it sat) every time the agent said it was still on something. When it was last
/// touched is in loop_list and /loops.
fn opened(entry: &OpenLoop) -> Option<String> {
    crate::clock::parse(&entry.opened_at).map(|at| format!("opened {}", crate::clock::short(&at)))
}

/// When it was last touched, for loop_list and /loops.
fn touched(entry: &OpenLoop) -> Option<String> {
    crate::clock::parse(&entry.last_touched_at).map(|at| crate::clock::short(&at))
}

fn full_entry(entry: &OpenLoop, now: u64) -> String {
    let mut line = format!(
        "· {} [{}] {}",
        entry.id,
        entry.kind.label(),
        truncate(entry.text.trim(), MAX_TEXT_CHARS),
    );
    if let Some(opened) = opened(entry) {
        line.push_str(&format!(" ({opened})"));
    }
    if entry.opened_by == OpenedBy::Fold {
        line.push_str(" (noticed while folding)");
    }
    if !entry.source_event_ids.is_empty() {
        let ids: Vec<String> = entry.source_event_ids.iter().map(|id| format!("#{id}")).collect();
        line.push_str(&format!(" [from {}]", ids.join(", ")));
    }
    if entry.is_stale(now) {
        line.push_str(" — still open?");
    }
    line
}

fn compact_entry(entry: &OpenLoop, now: u64) -> String {
    let mut line = format!("· {} {}", entry.id, truncate(entry.first_line(), 120));
    if let Some(opened) = opened(entry) {
        line.push_str(&format!(" ({opened})"));
    }
    if entry.is_stale(now) {
        line.push_str(" — still open?");
    }
    line
}

/// Render the loops block.
///
/// One rule decides the shape: detail degrades before anything disappears. Over
/// the cap, the least recently touched loops drop to an ID and a first line, and
/// none is ever hidden. Since the list already arrives most-recently-touched
/// first, that also means whatever the agent is currently on keeps its full text
/// longest — which is what an explicit focus flag was for.
pub fn render(loops: &[OpenLoop], now: u64) -> String {
    let open: Vec<&OpenLoop> =
        loops.iter().filter(|entry| entry.status == LoopStatus::Open).collect();

    let header = "OPEN LOOPS — things I started and haven't finished. \
                  Nothing here closes on its own; I close it when it's actually done or \
                  I've decided to drop it.";

    if open.is_empty() {
        return format!("{header}\nNothing open.");
    }

    let budget_chars = RENDER_BUDGET_TOKENS * BYTES_PER_TOKEN;
    let mut lines: Vec<String> =
        open.iter().map(|entry| full_entry(entry, now)).collect();

    let measure = |lines: &[String]| -> usize {
        header.chars().count() + lines.iter().map(|line| line.chars().count() + 1).sum::<usize>()
    };

    // Over budget: demote from the *oldest* end, one loop at a time, so the
    // entries the agent touched most recently keep their detail longest.
    let mut demote = open.len();
    while measure(&lines) > budget_chars && demote > 0 {
        demote -= 1;
        lines[demote] = compact_entry(open[demote], now);
    }

    format!("{header}\n{}", lines.join("\n"))
}

// ------------------------------------------------------------------ context block

/// First line of the appended loops block. Identifies it for stripping, and marks
/// it to the agent as harness-rendered state rather than something anyone said.
pub const BLOCK_MARKER: &str = "[open loops — current state, rendered by the harness]";

/// The loops block as it appears in context, or `None` when nothing is open.
pub fn context_block(store: &LoopStore) -> Result<Option<Message>> {
    let loops = store.open_loops()?;
    if loops.is_empty() {
        return Ok(None);
    }
    let now = store.distance()?;
    Ok(Some(Message::user(format!("{BLOCK_MARKER}\n{}", render(&loops, now)))))
}

/// Whether this message is the harness-rendered block, not a conversation turn.
pub fn is_block(message: &Message) -> bool {
    message.content.starts_with(BLOCK_MARKER)
}

/// Take the loops block out of the context.
///
/// A fold plans over the live turns, and the block is harness-rendered state
/// rather than anything anyone said: folding it into a gist would put the
/// harness's own words into the agent's memory of the conversation. So it comes out
/// before a fold and goes back afterwards, via `refresh_context`.
///
/// Returns whether a block was there.
pub fn strip_block(messages: &mut Vec<Message>) -> bool {
    match messages.iter().position(is_block) {
        Some(index) => {
            messages.remove(index);
            true
        }
        None => false,
    }
}

/// Keep the loops block current, at the end of the context.
///
/// Appended rather than held in the memory message, because volatility should
/// decide position, not importance. Editing a message invalidates the KV cache
/// for everything after it, so the most-frequently-changing layer is the worst
/// possible thing to put at position two: a loop opened at 200k of context would
/// cost a re-prefill of the whole live layer. At the end, a refresh costs only
/// what has arrived since the last one.
///
/// So this moves the block *only when its text changes*. An unchanged block is
/// left exactly where it is — still accurate, costing nothing — and drifts
/// backwards through the context until something about the loops actually
/// differs. Cost per fold cycle is therefore bounded by the size of the cycle
/// rather than by how often the agent touches its loops.
///
/// Returns whether the context changed.
pub fn refresh_context(messages: &mut Vec<Message>, store: &LoopStore) -> Result<bool> {
    let fresh = context_block(store)?;
    let existing = messages.iter().position(is_block);

    match (existing, fresh) {
        // Unchanged and already present: leave it alone. This is the common case
        // and the whole reason the scheme is affordable.
        (Some(index), Some(block)) if messages[index].content == block.content => Ok(false),

        (Some(index), Some(block)) => {
            messages.remove(index);
            messages.push(block);
            Ok(true)
        }
        (Some(index), None) => {
            messages.remove(index);
            Ok(true)
        }
        (None, Some(block)) => {
            messages.push(block);
            Ok(true)
        }
        (None, None) => Ok(false),
    }
}

// ------------------------------------------------------------------ tools

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::function(
            "loop_open",
            "Open a loop for something unfinished: a request from the operator you haven't answered, a \
             task you've started, a promise to check something later, a question you've left open \
             for yourself. Not for goals in general (those are threads) and not for facts (those \
             are the archive and the self-model). Your open loops stay in front of you no matter \
             how much happens in between, so this is how you make sure you come back to \
             something.",
            json!({"type": "object", "properties": {
                "text": {"type": "string",
                         "description": "What is open, in your own words, one or two sentences."},
                "kind": {"type": "string", "enum": ["request", "task", "promise", "question"]},
                "source_event_ids": {
                    "type": "array", "items": {"type": "integer", "minimum": 1},
                    "maxItems": MAX_SOURCE_IDS,
                    "description": "Archive events where this arose. Checked: the call is rejected if any don't exist."
                }
            }, "required": ["text", "kind"]}),
        ),
        ToolDefinition::function(
            "loop_touch",
            "Say you're back on a loop, or still on it. It moves to the top of your list and stops \
             looking stale. Use it when you pick something up again after a while, so the list \
             reflects what you're actually working on.",
            json!({"type": "object", "properties": {
                "id": {"type": "string", "description": "Loop ID, e.g. loop-0003."}
            }, "required": ["id"]}),
        ),
        ToolDefinition::function(
            "loop_close",
            "Close a loop, as 'done' if it's finished or 'dropped' if it isn't going to happen, \
             with one line on how it resolved. Nothing closes on its own, because deciding something \
             is finished is exactly the judgment worth making explicitly; the operator may also close one \
             from his memory tab, and you are told when he does. Don't close a loop on the strength of expecting it to work: close \
             it when you've seen that it did.",
            json!({"type": "object", "properties": {
                "id": {"type": "string", "description": "Loop ID, e.g. loop-0003."},
                "status": {"type": "string", "enum": ["done", "dropped"]},
                "resolution": {"type": "string",
                               "description": "One line on how it closed, with event IDs where useful."}
            }, "required": ["id", "status", "resolution"]}),
        ),
        ToolDefinition::function(
            "loop_list",
            "List your loops with their IDs, kind, status and how long since you last touched \
             them. Open loops are already in your context; use this for closed ones, or for the \
             exact IDs.",
            json!({"type": "object", "properties": {
                "status": {"type": "string", "enum": ["open", "done", "dropped", "all"],
                           "description": "Default open."}
            }}),
        ),
    ]
}

fn rejected(error: impl Into<String>) -> String {
    json!({"ok": false, "error": error.into(), "recorded": false}).to_string()
}

fn describe(entry: &OpenLoop, now: u64) -> Value {
    json!({
        "id": entry.id,
        "text": entry.text,
        "kind": entry.kind.label(),
        "status": entry.status.label(),
        "opened_by": entry.opened_by.label(),
        "source_event_ids": entry.source_event_ids,
        "opened": crate::clock::stored(&entry.opened_at),
        "touched": crate::clock::stored(&entry.last_touched_at),
        "distance_since_touched": entry.distance_since(now),
        "stale": entry.is_stale(now),
        "resolution": entry.resolution,
    })
}

pub fn execute_with(
    store: &LoopStore,
    source: &dyn EvidenceSource,
    call: &ToolCall,
) -> Result<String> {
    let now = store.distance()?;

    match call.function.name.as_str() {
        "loop_open" => {
            #[derive(Deserialize)]
            struct Args {
                text: String,
                kind: String,
                #[serde(default)]
                source_event_ids: Vec<u64>,
            }
            let args: Args = match serde_json::from_str(&call.function.arguments) {
                Ok(args) => args,
                Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
            };

            let text = args.text.trim();
            let chars = text.chars().count();
            if chars < MIN_TEXT_CHARS {
                return Ok(rejected(format!(
                    "text is too short (at least {MIN_TEXT_CHARS} characters)"
                )));
            }
            if chars > MAX_TEXT_CHARS {
                return Ok(rejected(format!(
                    "text is too long (at most {MAX_TEXT_CHARS} characters)"
                )));
            }

            let kind = match LoopKind::parse(&args.kind) {
                Ok(kind) => kind,
                Err(error) => return Ok(rejected(error)),
            };

            if args.source_event_ids.len() > MAX_SOURCE_IDS {
                return Ok(rejected(format!("at most {MAX_SOURCE_IDS} source event IDs")));
            }

            // Same discipline as all cited evidence: a citation that doesn't
            // resolve is rejected rather than stored, so a loop can never point
            // at an event that never existed.
            if !args.source_event_ids.is_empty() {
                let found = source.events(&args.source_event_ids)?;
                let missing: Vec<u64> = args
                    .source_event_ids
                    .iter()
                    .copied()
                    .filter(|id| !found.iter().any(|event| event.id == *id))
                    .collect();
                if !missing.is_empty() {
                    return Ok(rejected(format!(
                        "these event IDs don't exist: {missing:?} (find real ones with archive_search)"
                    )));
                }
            }

            let entry = store.open(text, kind, &args.source_event_ids, OpenedBy::S0ma, now)?;
            Ok(json!({
                "ok": true,
                "loop": describe(&entry, now),
                "open_loops": store.open_loops()?.len(),
            })
            .to_string())
        }

        "loop_touch" => {
            #[derive(Deserialize)]
            struct Args {
                id: String,
            }
            let args: Args = match serde_json::from_str(&call.function.arguments) {
                Ok(args) => args,
                Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
            };

            match store.touch(args.id.trim(), now)? {
                Ok(entry) => Ok(json!({"ok": true, "loop": describe(&entry, now)}).to_string()),
                Err(error) => Ok(rejected(error)),
            }
        }

        "loop_close" => {
            #[derive(Deserialize)]
            struct Args {
                id: String,
                status: String,
                resolution: String,
            }
            let args: Args = match serde_json::from_str(&call.function.arguments) {
                Ok(args) => args,
                Err(error) => return Ok(rejected(format!("bad arguments: {error}"))),
            };

            let status = match LoopStatus::parse(&args.status) {
                Ok(status) => status,
                Err(error) => return Ok(rejected(error)),
            };
            let resolution = args.resolution.trim();
            if resolution.chars().count() < MIN_TEXT_CHARS {
                return Ok(rejected("say in one line how it resolved"));
            }

            match store.close(args.id.trim(), status, resolution, now)? {
                Ok(entry) => Ok(json!({
                    "ok": true,
                    "loop": describe(&entry, now),
                    "open_loops": store.open_loops()?.len(),
                })
                .to_string()),
                Err(error) => Ok(rejected(error)),
            }
        }

        "loop_list" => {
            #[derive(Deserialize)]
            struct Args {
                #[serde(default)]
                status: Option<String>,
            }
            let args: Args =
                serde_json::from_str(&call.function.arguments).unwrap_or(Args { status: None });

            let wanted = args.status.as_deref().unwrap_or("open");
            let entries: Vec<Value> = store
                .all()?
                .into_iter()
                .filter(|entry| match wanted {
                    "all" => true,
                    other => entry.status.label() == other,
                })
                .map(|entry| describe(&entry, now))
                .collect();

            Ok(json!({"ok": true, "loops": entries}).to_string())
        }

        other => Err(format!("unknown open-loop tool '{other}'").into()),
    }
}

// ------------------------------------------------------------------ The operator closing a loop

/// Loops the operator closed from the memory tab, for his next message to the agent.
static CLOSED_BY_OPERATOR: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The operator closes a loop: done, or dropped, with what he said (or that it was
/// him). The agent is told with his next message (`take_note`); the loop leaves
/// the open-loops block in its context at the next refresh.
pub fn close_by_operator(store: &LoopStore, id: &str, done: bool, said: &str) -> Result<std::result::Result<OpenLoop, String>> {
    let status = if done { LoopStatus::Done } else { LoopStatus::Dropped };
    let said = said.trim();
    let resolution = if said.is_empty() { "closed by the operator".to_string() } else { format!("{said} (closed by the operator)") };
    let distance = store.distance()?;
    let closed = store.close(id, status, &resolution, distance)?;
    if let Ok(entry) = &closed {
        let line = format!(
            "- {} {} ({}){}",
            if done { "closed as done:" } else { "dropped:" },
            entry.id,
            truncate(entry.first_line(), 120),
            if said.is_empty() { String::new() } else { format!(" — he said: {}", truncate(said, 400)) }
        );
        CLOSED_BY_OPERATOR.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(line);
    }
    Ok(closed)
}

/// The note for the operator's next message: the loops he closed since the agent was
/// last told.
pub fn take_note() -> Option<String> {
    let lines = std::mem::take(&mut *CLOSED_BY_OPERATOR.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
    (!lines.is_empty()).then(|| format!("\n\n[OPEN LOOPS]\nThe operator, in his memory tab:\n{}", lines.join("\n")))
}

// ------------------------------------------------------------------ operator command

pub const OPERATOR_HELP: &str = "usage: /loops [open|done|dropped|all]";

/// `/loops [status]`. Returns the lines to print.
pub fn handle_operator_command(store: &LoopStore, command: &str) -> Result<String> {
    let parts: Vec<&str> = command.split_whitespace().collect();
    let wanted = match parts.as_slice() {
        ["/loops"] => "open",
        ["/loops", status @ ("open" | "done" | "dropped" | "all")] => status,
        _ => return Ok(OPERATOR_HELP.to_string()),
    };

    let now = store.distance()?;
    let entries: Vec<OpenLoop> = store
        .all()?
        .into_iter()
        .filter(|entry| wanted == "all" || entry.status.label() == wanted)
        .collect();

    if entries.is_empty() {
        return Ok(format!("[loops] nothing {wanted}"));
    }

    let mut out = format!("[loops] {} {wanted}, at {now} tokens of distance", entries.len());
    for entry in entries {
        out.push_str(&format!(
            "\n  {} [{}/{}] {}{}{}",
            entry.id,
            entry.kind.label(),
            entry.status.label(),
            truncate(entry.first_line(), 100),
            touched(&entry).map(|at| format!(" · touched {at}")).unwrap_or_default(),
            if entry.is_stale(now) { " — stale" } else { "" },
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;



    /// A store in its own directory. Each call gets a fresh path so tests never
    /// share a database.
    fn store() -> LoopStore {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let root = std::env::temp_dir().join(format!(
            "s0ma-open-loops-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst),
        ));
        std::fs::create_dir_all(&root).expect("test directory should be created");
        LoopStore::beside_archive(&root.join("hippocampus.jsonl"))
    }

    #[test]
    fn operator_closes_a_loop_and_s0ma_is_told_once() {
        let store = store();
        let entry = store.open("Check the flaky upload test again tomorrow", LoopKind::Promise, &[12], OpenedBy::S0ma, 0).unwrap();
        let _ = take_note();
        let closed = close_by_operator(&store, &entry.id, true, "it passed 50 runs").unwrap().unwrap();
        assert_eq!(closed.status, LoopStatus::Done);
        assert_eq!(closed.resolution.as_deref(), Some("it passed 50 runs (closed by the operator)"));
        assert!(store.open_loops().unwrap().is_empty());
        // Closed already: said so, and no second note.
        assert!(close_by_operator(&store, &entry.id, false, "").unwrap().is_err());
        let other = store.open("Ask about the deploy key", LoopKind::Question, &[], OpenedBy::Fold, 0).unwrap();
        close_by_operator(&store, &other.id, false, "").unwrap().unwrap();
        assert_eq!(store.get(&other.id).unwrap().unwrap().resolution.as_deref(), Some("closed by the operator"));
        let note = take_note().unwrap();
        assert!(note.contains("[OPEN LOOPS]") && note.contains("closed as done:") && note.contains("it passed 50 runs"), "{note}");
        assert!(note.contains("dropped:") && note.contains("Ask about the deploy key"), "{note}");
        assert!(take_note().is_none(), "told once");
    }

    #[test]
    fn a_loop_opens_and_only_closes_explicitly() {
        let store = store();
        let entry =
            store.open("check the fold timings", LoopKind::Task, &[], OpenedBy::S0ma, 0).unwrap();

        assert_eq!(store.open_loops().unwrap().len(), 1);

        // No amount of distance closes it. This is the whole point: a loop is the
        // one thing in working memory that does not age away.
        store.advance_distance(5_000_000).unwrap();
        assert_eq!(store.open_loops().unwrap().len(), 1);

        store
            .close(&entry.id, LoopStatus::Done, "measured, 9s per fold", 5_000_000)
            .unwrap()
            .unwrap();
        assert!(store.open_loops().unwrap().is_empty());
        assert_eq!(store.get(&entry.id).unwrap().unwrap().status, LoopStatus::Done);
    }

    #[test]
    fn a_closed_loop_cannot_be_touched() {
        let store = store();
        let entry = store.open("a finished thing", LoopKind::Task, &[], OpenedBy::S0ma, 0).unwrap();
        store.close(&entry.id, LoopStatus::Done, "done and verified", 0).unwrap().unwrap();
        assert!(store.touch(&entry.id, 0).unwrap().is_err());
    }

    #[test]
    fn a_changed_block_moves_to_the_end_and_never_duplicates() {
        let store = store();
        let first = store.open("the first thing", LoopKind::Task, &[], OpenedBy::S0ma, 0).unwrap();

        let mut messages = vec![Message::system("prefix".into())];
        refresh_context(&mut messages, &store).unwrap();
        messages.push(Message::user("and another thing".into()));

        store.open("the second thing", LoopKind::Request, &[], OpenedBy::S0ma, 0).unwrap();
        assert!(refresh_context(&mut messages, &store).unwrap());

        assert_eq!(messages.iter().filter(|m| is_block(m)).count(), 1, "exactly one block");
        assert!(is_block(messages.last().unwrap()), "a changed block moves to the end");
        assert!(messages.last().unwrap().content.contains("the second thing"));
        assert!(messages.last().unwrap().content.contains(&first.id));
    }
}
