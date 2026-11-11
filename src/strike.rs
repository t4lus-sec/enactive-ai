//! Striking text out of everything s0ma keeps: the memory tab's redaction.
//!
//! `redact.rs` began as one event at a time, and said plainly what it could
//! not reach: what s0ma had written in its own words. This is the other half.
//! Given the exact strings to remove, it finds every place they are kept and
//! replaces each occurrence with [`MARKER`]:
//!
//! * **the archive** -- every event that holds one, keeping its id, its time
//!   and who said it, and then what is derived from those events: both search
//!   indexes, the gists and open items, the compaction windows, the saved model
//!   calls, the working context saved for a restart;
//! * **the checkpoints** a sleep wrote;
//! * **threads** -- their titles, descriptions, significance, questions and
//!   next actions;
//! * **the world model** -- models, records, every revision in their history,
//!   documents and theirs, the notes on links;
//! * **episodes**, and the keyword and cue indexes made from them;
//! * **the board** -- titles, goals, tests, outcomes, facts, leads and hints;
//! * **open loops**;
//! * **the typed history** the prompt recalls with ↑;
//! * **the worker journal**;
//! * **skills**, with their edit receipts and the deleted ones;
//! * **redaction requests** still holding a proposal.
//!
//! # What it will not do
//!
//! It never touches a structural value: an id, a state, a timestamp, a hash,
//! a key. Text is struck where it is text. A fragment that happens to spell
//! `pending` does not turn every request's state into `[redacted]`.
//!
//! # References keep working
//!
//! A world-model record and an episode both keep a hash of each archive event
//! they cite, to notice when the event under them changes. A redaction is a
//! change, but not the kind they look for: their evidence would read as
//! tampered with, and the record as unsupported, for no other reason than that
//! a key was struck out of a tool result. So a hash that matched the event
//! before is stamped again from the event after -- and only one that matched:
//! a hash that already disagreed is left disagreeing.
//!
//! # What it cannot reach
//!
//! The weakness log is hash-chained, every line committing to the one before,
//! and rewriting one line would break every line after it. It is reported.
//! So is anything else under the data directory that still holds the text
//! once the strike is done: the last step looks, rather than assumes.
//!
//! # The text is never stored
//!
//! The fragments are held for the length of a call and dropped. What the
//! redaction book keeps is where and how many, never what.

use crate::hippocampus::{Hippocampus, RecalledEvent, Speaker};

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub use crate::redact::MARKER;

type Result<T> = std::result::Result<T, String>;

/// Shorter than this matches all over everything (`10.0`, `the`).
pub const MIN_FRAGMENT: usize = 4;

/// One strike at a time: two at once would each read what the other is about
/// to rewrite.
static RUNNING: Mutex<()> = Mutex::new(());

// ------------------------------------------------------------------ the strings

/// The strings to strike, longest first, so a longer one is struck whole
/// before a shorter one inside it can break it up.
#[derive(Clone)]
pub struct Fragments(Vec<String>);

impl std::fmt::Debug for Fragments {
    // Never printed: a panic message or a log line is one more copy.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fragments({} strings)", self.0.len())
    }
}

impl Fragments {
    /// Trimmed, each at least [`MIN_FRAGMENT`] characters, each once.
    pub fn new(raw: &[String]) -> Result<Self> {
        let mut out: Vec<String> = Vec::new();
        for (index, text) in raw.iter().enumerate() {
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            if text.chars().count() < MIN_FRAGMENT {
                return Err(format!(
                    "the {} selection is shorter than {MIN_FRAGMENT} characters and would match all over \
                     everything; mark more of it",
                    nth(index)
                ));
            }
            // A piece of the marker would match inside every marker already
            // written, and each pass would nest it again.
            if MARKER.contains(text) {
                return Err("that is part of the redaction marker itself".into());
            }
            if !out.iter().any(|seen| seen == text) {
                out.push(text.to_owned());
            }
        }
        if out.is_empty() {
            return Err("nothing is marked".into());
        }
        out.sort_by_key(|text| std::cmp::Reverse(text.len()));
        Ok(Self(out))
    }

    pub fn list(&self) -> &[String] {
        &self.0
    }

    /// `text` with every occurrence struck, and how many there were; None when
    /// it holds none.
    pub fn strike(&self, text: &str) -> Option<(String, usize)> {
        if !self.0.iter().any(|fragment| text.contains(fragment.as_str())) {
            return None;
        }
        let mut out = text.to_owned();
        let mut count = 0;
        for fragment in &self.0 {
            let n = out.matches(fragment.as_str()).count();
            if n > 0 {
                out = out.replace(fragment.as_str(), MARKER);
                count += n;
            }
        }
        (count > 0).then_some((out, count))
    }

    /// A little of `text` around the first occurrence, for the review: the
    /// fragment shown in place, so the operator sees it in its context.
    pub fn excerpt(&self, text: &str) -> Option<String> {
        let (at, fragment) = self
            .0
            .iter()
            .filter_map(|fragment| text.find(fragment.as_str()).map(|at| (at, fragment)))
            .min_by_key(|(at, _)| *at)?;
        const AROUND: usize = 40;
        let before: String = {
            let head = &text[..at];
            let chars: Vec<char> = head.chars().collect();
            chars[chars.len().saturating_sub(AROUND)..].iter().collect()
        };
        let after: String = text[at + fragment.len()..].chars().take(AROUND).collect();
        let flat = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        let lead = if before.len() < at { "…" } else { "" };
        let tail = if at + fragment.len() + after.len() < text.len() { "…" } else { "" };
        Some(format!(
            "{lead}{}{}{EXCERPT_OPEN}{}{EXCERPT_CLOSE}{}{}{tail}",
            flat(&before),
            if before.ends_with(char::is_whitespace) { " " } else { "" },
            flat(fragment),
            if after.starts_with(char::is_whitespace) { " " } else { "" },
            flat(&after)
        ))
    }

    /// The forms a fragment takes inside a file: as it is, and as JSON writes
    /// it (a newline or a quote escaped).
    fn needles(&self) -> Vec<Vec<u8>> {
        let mut out: BTreeSet<Vec<u8>> = BTreeSet::new();
        for fragment in &self.0 {
            out.insert(fragment.as_bytes().to_vec());
            if let Ok(json) = serde_json::to_string(fragment) {
                out.insert(json.as_bytes()[1..json.len() - 1].to_vec());
            }
            if fragment.contains('\n') {
                // The typed history's own encoding of a line break.
                out.insert(fragment.replace('\n', "<\\n>").into_bytes());
            }
        }
        out.into_iter().collect()
    }
}

/// Around the struck text in an excerpt (`memory_view` draws it in red).
pub const EXCERPT_OPEN: char = '\u{1}';
pub const EXCERPT_CLOSE: char = '\u{2}';

fn nth(index: usize) -> String {
    match index + 1 {
        1 => "first".into(),
        2 => "second".into(),
        3 => "third".into(),
        n => format!("{n}th"),
    }
}

// ------------------------------------------------------------------ JSON

/// Keys whose values are never words: ids, states, times, hashes. A string
/// under one is left alone, and so is everything inside it.
fn structural(key: &str) -> bool {
    matches!(
        key,
        "id" | "state"
            | "status"
            | "kind"
            | "speaker"
            | "author"
            | "by"
            | "at"
            | "event"
            | "category"
            | "mode"
            | "type"
            | "role"
            | "polarity"
            | "revision"
            | "schema_version"
            | "was"
            | "fingerprint"
            | "name"
            | "resource"
            | "request"
            | "run_id"
            | "relation"
            | "timestamp"
            // Enums that read like words: a world record's basis, an
            // episode's outcome. And an episode's links, which are ids.
            | "basis"
            | "outcome"
            | "supersedes"
    ) || key.ends_with("_at")
        || key.ends_with("_id")
        || key.ends_with("_ids")
        || key.ends_with("_ref")
        || key.ends_with("_revision")
        || key.contains("hash")
        || key.contains("digest")
}

/// Every string value in `value` that holds a fragment, struck; keys and
/// structural values left alone. Returns how many occurrences.
pub fn strike_json(value: &mut Value, strike: &dyn Fn(&str) -> Option<(String, usize)>) -> usize {
    match value {
        Value::String(text) => match strike(text) {
            Some((struck, n)) => {
                *text = struck;
                n
            }
            None => 0,
        },
        Value::Array(items) => items.iter_mut().map(|item| strike_json(item, strike)).sum(),
        Value::Object(map) => map
            .iter_mut()
            .filter(|(key, _)| !structural(key))
            .map(|(_, item)| strike_json(item, strike))
            .sum(),
        _ => 0,
    }
}

/// A text that may be JSON (a stored record): struck as JSON when it parses as
/// an object or array, as text otherwise.
fn strike_stored(text: &str, fragments: &Fragments) -> Option<(String, usize)> {
    let trimmed = text.trim_start();
    if (trimmed.starts_with('{') || trimmed.starts_with('['))
        && let Ok(mut value) = serde_json::from_str::<Value>(text)
    {
        let count = strike_json(&mut value, &|s| fragments.strike(s));
        return (count > 0).then(|| (serde_json::to_string(&value).unwrap_or_default(), count));
    }
    fragments.strike(text)
}

/// A little of the stored text around the first fragment in it: for a JSON
/// record, from the string that holds it rather than from the JSON.
fn excerpt_of(text: &str, fragments: &Fragments) -> Option<String> {
    fn first(value: &Value, fragments: &Fragments) -> Option<String> {
        match value {
            Value::String(text) => fragments.excerpt(text),
            Value::Array(items) => items.iter().find_map(|item| first(item, fragments)),
            Value::Object(map) => map.iter().filter(|(key, _)| !structural(key)).find_map(|(_, item)| first(item, fragments)),
            _ => None,
        }
    }
    let trimmed = text.trim_start();
    if (trimmed.starts_with('{') || trimmed.starts_with('['))
        && let Ok(value) = serde_json::from_str::<Value>(text)
    {
        return first(&value, fragments);
    }
    fragments.excerpt(text)
}

// ------------------------------------------------------------------ the report

/// One place the text is kept.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Place {
    /// Which store ("archive", "world model", …).
    pub store: String,
    /// Which thing in it, by id and never by its words ("event #812",
    /// "record api-hosts").
    pub item: String,
    pub count: usize,
    /// The archive event, when the place is one: the review jumps to it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event: Option<u64>,
    /// A little of the text around it, for the review only: never written to
    /// the redaction book.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excerpt: Option<String>,
}

impl Place {
    fn new(store: &str, item: impl Into<String>, count: usize) -> Self {
        Self { store: store.into(), item: item.into(), count, event: None, excerpt: None }
    }
}

/// Where the text is (a review), or where it was and what went with it (a
/// strike).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Impact {
    /// Places that hold it and are struck.
    pub places: Vec<Place>,
    /// Places that hold it and are not: the weakness log, and anything the
    /// last look still found.
    pub unreachable: Vec<Place>,
    /// What derived stores dropped, which are rebuilt from the struck archive
    /// rather than struck themselves ("2 gists", "14 saved calls").
    #[serde(default)]
    pub cleared: Vec<String>,
    /// What went wrong along the way, carried past rather than stopped at.
    #[serde(default)]
    pub notes: Vec<String>,
}

impl Impact {
    pub fn total(&self) -> usize {
        self.places.iter().map(|place| place.count).sum()
    }

    /// Per store, in the order first met: (store, places, occurrences).
    pub fn stores(&self) -> Vec<(String, usize, usize)> {
        let mut out: Vec<(String, usize, usize)> = Vec::new();
        for place in &self.places {
            match out.iter_mut().find(|(store, _, _)| *store == place.store) {
                Some(entry) => {
                    entry.1 += 1;
                    entry.2 += place.count;
                }
                None => out.push((place.store.clone(), 1, place.count)),
            }
        }
        out
    }

    /// One line, counts only.
    pub fn summary(&self) -> String {
        if self.places.is_empty() && self.unreachable.is_empty() {
            return "the text is nowhere s0ma keeps".into();
        }
        let stores: Vec<String> = self
            .stores()
            .iter()
            .map(|(store, places, count)| {
                if *places == 1 { format!("{store} ×{count}") } else { format!("{store} ×{count} in {places}") }
            })
            .collect();
        let mut line = format!("{} occurrence(s): {}", self.total(), stores.join(", "));
        if !self.unreachable.is_empty() {
            let left: Vec<String> =
                self.unreachable.iter().map(|place| format!("{} ({})", place.store, place.item)).collect();
            line.push_str(&format!("; not reached: {}", left.join(", ")));
        }
        line
    }

    /// What may be kept: the excerpts dropped, and an id that spells one of
    /// `fragments` (ids are never struck) struck here.
    pub fn for_the_book(&self, fragments: Option<&Fragments>) -> Vec<Place> {
        self.places
            .iter()
            .chain(&self.unreachable)
            .map(|place| {
                let item = fragments.and_then(|f| f.strike(&place.item)).map_or(place.item.clone(), |(item, _)| item);
                Place { excerpt: None, item, ..place.clone() }
            })
            .collect()
    }
}

// ------------------------------------------------------------------ hashes

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// What the world model keeps for a cited event (`world_model::hashes`).
fn world_hash(event: &RecalledEvent) -> String {
    sha(&serde_json::to_vec(&(&event.timestamp, format!("{:?}", event.speaker), &event.content)).unwrap_or_default())
}

/// What an episode keeps for a source event (`episode_recall::check_evidence`).
fn episode_hash(event: &RecalledEvent) -> String {
    let speaker = match event.speaker {
        Speaker::Operator => "operator",
        Speaker::S0ma => "s0ma",
    };
    sha(serde_json::to_string(&(&event.timestamp, speaker, &event.content)).unwrap_or_default().as_bytes())
}

/// Events struck, as they were and as they are: for stamping citations again.
#[derive(Default)]
struct Restamp {
    before: BTreeMap<u64, RecalledEvent>,
    after: BTreeMap<u64, RecalledEvent>,
}

impl Restamp {
    /// `stored` again for each struck event it cites, where it matched the
    /// event before the strike. Returns how many were stamped.
    fn apply(&self, stored: &mut BTreeMap<u64, String>, hash: fn(&RecalledEvent) -> String) -> usize {
        let mut stamped = 0;
        for (id, kept) in stored.iter_mut() {
            if let (Some(before), Some(after)) = (self.before.get(id), self.after.get(id))
                && *kept == hash(before)
            {
                *kept = hash(after);
                stamped += 1;
            }
        }
        stamped
    }
}

// ------------------------------------------------------------------ the stores

/// A table and the columns in it that hold words.
struct Table {
    name: &'static str,
    /// SQL for the id the review names it by.
    key: &'static str,
    label: &'static str,
    columns: &'static [&'static str],
}

const WORLD: &[Table] = &[
    Table { name: "wm_models", key: "id", label: "model", columns: &["title", "summary"] },
    Table { name: "wm_records", key: "id", label: "record", columns: &["body"] },
    Table { name: "wm_history", key: "id || ' r' || revision", label: "record", columns: &["body", "reason"] },
    Table { name: "wm_documents", key: "name", label: "document", columns: &["body"] },
    Table {
        name: "wm_document_history",
        key: "name || ' r' || revision",
        label: "document",
        columns: &["body", "reason"],
    },
    Table { name: "wm_links", key: "model_id || ' → ' || target", label: "link", columns: &["note"] },
];

const EPISODES: &[Table] = &[
    Table { name: "consolidated_episodes", key: "id", label: "episode", columns: &["record_json"] },
    Table { name: "episode_sleep_receipts", key: "sleep_key", label: "sleep", columns: &["empty_reason"] },
];

const BOARD: &[Table] = &[
    Table {
        name: "boards",
        key: "name",
        label: "board",
        columns: &["title", "origin", "goal", "goal_test", "outcome"],
    },
    // A finding's basis is one of a few words the board parses back.
    Table { name: "facts", key: "id", label: "fact", columns: &["statement"] },
    Table { name: "intents", key: "id", label: "lead", columns: &["direction", "rationale"] },
    Table { name: "hints", key: "id", label: "hint", columns: &["text"] },
    Table { name: "board_models", key: "board_id || ' → ' || model_id", label: "link", columns: &["note"] },
];

const LOOPS: &[Table] = &[Table { name: "loops", key: "id", label: "loop", columns: &["text", "resolution"] }];

fn has_table(db: &Connection, name: &str) -> bool {
    db.query_row("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1", [name], |_| Ok(()))
        .optional()
        .ok()
        .flatten()
        .is_some()
}

/// Every word-holding column of `tables` in the database at `path`, struck
/// (with `apply`) or counted. Returns the keys changed, per table.
fn strike_sqlite(
    path: &Path,
    store: &str,
    tables: &[Table],
    fragments: &Fragments,
    apply: bool,
    impact: &mut Impact,
) -> Result<BTreeMap<&'static str, Vec<String>>> {
    let mut changed: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    if !path.exists() {
        return Ok(changed);
    }
    let mut db = Connection::open(path).map_err(|error| format!("{store}: {error}"))?;
    let _ = db.busy_timeout(std::time::Duration::from_secs(10));
    // A row's old text would otherwise stay in the page it was freed from.
    let _ = db.execute_batch("PRAGMA secure_delete=ON");
    let mut wrote = false;
    {
        // Taken for writing from the start: a read upgraded to a write after
        // someone else wrote meanwhile fails however long it waits.
        let behavior = if apply { rusqlite::TransactionBehavior::Immediate } else { rusqlite::TransactionBehavior::Deferred };
        let tx = db.transaction_with_behavior(behavior).map_err(|error| format!("{store}: {error}"))?;
        for table in tables {
            if !has_table(&tx, table.name) {
                continue;
            }
            let columns = table.columns.join(", ");
            let sql = format!("SELECT rowid, {}, {columns} FROM {}", table.key, table.name);
            let rows: Vec<(i64, String, Vec<Option<String>>)> = {
                let mut statement = tx.prepare(&sql).map_err(|error| format!("{store}: {error}"))?;
                statement
                    .query_map([], |row| {
                        let mut values = Vec::new();
                        for index in 0..table.columns.len() {
                            values.push(row.get::<_, Option<String>>(index + 2)?);
                        }
                        Ok((row.get(0)?, row.get::<_, Option<String>>(1)?.unwrap_or_default(), values))
                    })
                    .map_err(|error| format!("{store}: {error}"))?
                    .collect::<rusqlite::Result<_>>()
                    .map_err(|error| format!("{store}: {error}"))?
            };
            for (rowid, key, values) in rows {
                let mut count = 0;
                let mut excerpt = None;
                for (column, value) in table.columns.iter().zip(values) {
                    let Some(value) = value else { continue };
                    let Some((struck, n)) = strike_stored(&value, fragments) else { continue };
                    count += n;
                    if excerpt.is_none() {
                        excerpt = excerpt_of(&value, fragments);
                    }
                    if apply {
                        tx.execute(&format!("UPDATE {} SET {column}=?1 WHERE rowid=?2", table.name), params![struck, rowid])
                            .map_err(|error| format!("{store}: {error}"))?;
                        wrote = true;
                    }
                }
                if count > 0 {
                    let mut place = Place::new(store, format!("{} {key}", table.label), count);
                    place.excerpt = excerpt;
                    impact.places.push(place);
                    changed.entry(table.name).or_default().push(key);
                }
            }
        }
        tx.commit().map_err(|error| format!("{store}: {error}"))?;
    }
    if wrote {
        checkpoint(&db);
    }
    Ok(changed)
}

/// The world model's history again, each revision's cited events stamped
/// from the struck archive where they matched it before.
fn restamp_world(path: &Path, restamp: &Restamp) -> Result<usize> {
    if restamp.after.is_empty() || !path.exists() {
        return Ok(0);
    }
    let db = Connection::open(path).map_err(|error| error.to_string())?;
    let _ = db.busy_timeout(std::time::Duration::from_secs(10));
    if !has_table(&db, "wm_history") {
        return Ok(0);
    }
    let rows: Vec<(i64, String)> = db
        .prepare("SELECT rowid, evidence_hashes FROM wm_history")
        .and_then(|mut statement| {
            statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<rusqlite::Result<_>>()
        })
        .map_err(|error| error.to_string())?;
    let mut stamped = 0;
    for (rowid, raw) in rows {
        let Ok(mut hashes) = serde_json::from_str::<BTreeMap<u64, String>>(&raw) else { continue };
        let n = restamp.apply(&mut hashes, world_hash);
        if n > 0 {
            let raw = serde_json::to_string(&hashes).map_err(|error| error.to_string())?;
            db.execute("UPDATE wm_history SET evidence_hashes=?1 WHERE rowid=?2", params![raw, rowid])
                .map_err(|error| error.to_string())?;
            stamped += n;
        }
    }
    Ok(stamped)
}

/// Episodes after their words were struck: their source hashes stamped again,
/// and the keyword and cue rows made from the old words dropped, to be made
/// again from the new ones.
fn settle_episodes(path: &Path, struck: &[String], restamp: &Restamp) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let db = Connection::open(path).map_err(|error| error.to_string())?;
    let _ = db.busy_timeout(std::time::Duration::from_secs(10));
    let _ = db.execute_batch("PRAGMA secure_delete=ON");
    let mut stamped = 0;
    if !restamp.after.is_empty() && has_table(&db, "consolidated_episodes") {
        let rows: Vec<(String, String)> = db
            .prepare("SELECT id, source_digests_json FROM consolidated_episodes")
            .and_then(|mut statement| {
                statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<rusqlite::Result<_>>()
            })
            .map_err(|error| error.to_string())?;
        for (id, raw) in rows {
            let Ok(mut sources) = serde_json::from_str::<BTreeMap<u64, String>>(&raw) else { continue };
            let n = restamp.apply(&mut sources, episode_hash);
            if n > 0 {
                let raw = serde_json::to_string(&sources).map_err(|error| error.to_string())?;
                db.execute("UPDATE consolidated_episodes SET source_digests_json=?1 WHERE id=?2", params![raw, id])
                    .map_err(|error| error.to_string())?;
                stamped += n;
            }
        }
    }
    let mut dropped = false;
    for id in struck {
        for (table, column) in
            [("episode_text", "episode_id"), ("episode_fts_state", "episode_id"), ("episode_cue_vectors", "episode_id")]
        {
            if has_table(&db, table) {
                let _ = db.execute(&format!("DELETE FROM {table} WHERE {column}=?1"), [id]);
                dropped = true;
            }
        }
    }
    if dropped {
        if has_table(&db, "episode_text") {
            // The postings of a deleted row stay in the index's segments
            // until they are merged.
            let _ = db.execute_batch("INSERT INTO episode_text(episode_text) VALUES('optimize')");
        }
        checkpoint(&db);
    }
    Ok(stamped)
}

/// A text file, struck line by line (or, for a JSON file, value by value).
/// Returns how many occurrences.
fn strike_file(path: &Path, fragments: &Fragments, apply: bool) -> Result<usize> {
    let Ok(text) = fs::read_to_string(path) else { return Ok(0) };
    let json = path.extension().and_then(|e| e.to_str());
    let (out, count) = match json {
        Some("json") => match serde_json::from_str::<Value>(&text) {
            Ok(mut value) => {
                let count = strike_json(&mut value, &|s| fragments.strike(s));
                let out = serde_json::to_string_pretty(&value).map_err(|error| error.to_string())? + "\n";
                (out, count)
            }
            Err(_) => fragments.strike(&text).unwrap_or((text.clone(), 0)),
        },
        Some("jsonl") => {
            let mut out = String::new();
            let mut count = 0;
            for line in text.lines() {
                match serde_json::from_str::<Value>(line) {
                    Ok(mut value) => {
                        let n = strike_json(&mut value, &|s| fragments.strike(s));
                        count += n;
                        if n > 0 {
                            out.push_str(&serde_json::to_string(&value).map_err(|error| error.to_string())?);
                        } else {
                            out.push_str(line);
                        }
                    }
                    Err(_) => match fragments.strike(line) {
                        Some((struck, n)) => {
                            count += n;
                            out.push_str(&struck);
                        }
                        None => out.push_str(line),
                    },
                }
                out.push('\n');
            }
            (out, count)
        }
        _ => fragments.strike(&text).unwrap_or((text.clone(), 0)),
    };
    if apply && count > 0 {
        let temp = path.with_extension("striking");
        fs::write(&temp, out.as_bytes()).map_err(|error| format!("{}: {error}", path.display()))?;
        fs::rename(&temp, path).map_err(|error| format!("{}: {error}", path.display()))?;
    }
    Ok(count)
}

/// The typed history: lines that hold a fragment, struck, in either of the
/// forms the file keeps a line break in.
fn strike_history(path: &Path, fragments: &Fragments, apply: bool) -> Result<usize> {
    let Ok(text) = fs::read_to_string(path) else { return Ok(0) };
    let encoded: Vec<String> = fragments.list().iter().map(|f| f.replace('\n', "<\\n>")).collect();
    let encoded = Fragments(encoded);
    let mut count = 0;
    let mut out = String::new();
    for line in text.lines() {
        let mut line = line.to_owned();
        for form in [fragments, &encoded] {
            if let Some((struck, n)) = form.strike(&line) {
                line = struck;
                count += n;
            }
        }
        out.push_str(&line);
        out.push('\n');
    }
    if apply && count > 0 {
        let temp = path.with_extension("striking");
        fs::write(&temp, out.as_bytes()).map_err(|error| error.to_string())?;
        fs::rename(&temp, path).map_err(|error| error.to_string())?;
    }
    Ok(count)
}

/// Every file under `root`, hidden ones too (a skill's edit receipts and its
/// deleted versions are in `.history` and `.deleted`).
fn files_under(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_owned()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => stack.push(path),
                Ok(kind) if kind.is_file() => out.push(path),
                _ => {}
            }
        }
    }
    out.sort();
    out
}

/// How many times any of `needles` occurs in `bytes`.
fn occurrences(bytes: &[u8], needles: &[Vec<u8>]) -> usize {
    needles
        .iter()
        .filter(|needle| !needle.is_empty())
        .map(|needle| memchr::memmem::find_iter(bytes, needle).count())
        .max()
        .unwrap_or(0)
}

/// Freed pages were zeroed as they were freed (`secure_delete`); what is
/// left is the old text in the write-ahead log, which a truncating
/// checkpoint clears. A `VACUUM` would also do it, but rewrites the whole
/// file, minutes for the semantic index on a large archive.
fn checkpoint(db: &Connection) {
    let _ = db.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()));
}

// ------------------------------------------------------------------ the strike

/// Where the text is kept, changing nothing.
pub fn survey(memory: &Hippocampus, fragments: &Fragments) -> Result<Impact> {
    let _one = RUNNING.lock().map_err(|_| "a strike failed part way; restart s0ma".to_string())?;
    run(memory, fragments, false)
}

/// Strike it from everywhere it is kept. The fragments are held until s0ma
/// has gone over everything once more (`take_settle`).
pub fn apply(memory: &Hippocampus, fragments: &Fragments) -> Result<Impact> {
    let _one = RUNNING.lock().map_err(|_| "a strike failed part way; restart s0ma".to_string())?;
    let impact = run(memory, fragments, true)?;
    // The editor's copies of unsaved changes may hold the text: they go.
    crate::tabs::editor_forget_copies();
    if let Ok(mut owed) = SETTLE.lock() {
        owed.extend(fragments.list().iter().cloned());
    }
    Ok(impact)
}

/// Strings struck since s0ma last settled. A strike runs beside the turn,
/// the compactor and the background sleep, any of which may have held the
/// text and be about to write it again; and s0ma's live context still holds
/// it. So the running s0ma, when it next has the floor, stops what writes
/// from the archive, strikes the same strings once more and rebuilds its
/// context (`app.rs`). Held in memory only, and given up by taking them.
static SETTLE: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Whether the settle could not rebuild the context and is trying again.
static RETRYING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The context could not be rebuilt: the strings are owed again, so every
/// pass until it can be strikes what the live context put back. Returns
/// whether this is the first time (to say so once).
pub fn owe_again(fragments: &Fragments) -> bool {
    if let Ok(mut owed) = SETTLE.lock() {
        owed.extend(fragments.list().iter().cloned());
    }
    !RETRYING.swap(true, std::sync::atomic::Ordering::SeqCst)
}

/// The context was rebuilt: nothing is owed any more.
pub fn settled() {
    RETRYING.store(false, std::sync::atomic::Ordering::SeqCst);
}

pub fn take_settle() -> Option<Fragments> {
    let owed = std::mem::take(&mut *SETTLE.lock().ok()?);
    Fragments::new(&owed).ok()
}

/// Whether a strike is waiting for s0ma to settle it.
pub fn settle_owed() -> bool {
    SETTLE.lock().map(|owed| !owed.is_empty()).unwrap_or(false)
}

/// Strike again, quietly, what an earlier strike struck: the settling pass.
pub fn sweep(memory: &Hippocampus, fragments: &Fragments) -> Result<Impact> {
    let _one = RUNNING.lock().map_err(|_| "a strike failed part way; restart s0ma".to_string())?;
    run_with(memory, fragments, true, false)
}

fn run(memory: &Hippocampus, fragments: &Fragments, apply: bool) -> Result<Impact> {
    run_with(memory, fragments, apply, true)
}

/// `look`: whether to search the whole data directory at the end. The
/// settling pass does not: it repeats a strike that already looked.
fn run_with(memory: &Hippocampus, fragments: &Fragments, apply: bool, look_after: bool) -> Result<Impact> {
    let mut impact = Impact::default();
    let archive = memory.archive_path();
    let data_dir = archive.parent().unwrap_or(Path::new(".")).to_owned();
    let strike = |text: &str| fragments.strike(text);
    let note = impact_note;
    let mut notes: Vec<String> = Vec::new();

    // The archive. Its events as they were are kept for the length of this
    // call: the citations to them are stamped again from them below.
    let mut restamp = Restamp::default();
    let mut struck_events: Vec<u64> = Vec::new();
    memory
        .each_event(|event| {
            if let Some((_, count)) = fragments.strike(&event.content) {
                let mut place = Place::new("archive", format!("event #{}", event.id), count);
                place.event = Some(event.id);
                place.excerpt = fragments.excerpt(&event.content);
                impact.places.push(place);
                struck_events.push(event.id);
                restamp.before.insert(event.id, event);
            }
        })
        .map_err(|error| format!("the archive could not be read: {error}"))?;
    if apply && !struck_events.is_empty() {
        let ids = memory
            .strike_events(|content| fragments.strike(content).map(|(text, _)| text))
            .map_err(|error| format!("the archive was not changed: {error}"))?;
        for event in memory.get_events(&ids).map_err(|error| error.to_string())? {
            restamp.after.insert(event.id, event);
        }
        let forgotten = crate::archive::forget_events(memory, &ids);
        if forgotten.keyword_rows + forgotten.semantic_rows > 0 {
            impact.cleared.push(format!(
                "{} keyword and {} semantic index rows",
                forgotten.keyword_rows, forgotten.semantic_rows
            ));
        }
        notes.extend(forgotten.failures.into_iter().map(|failure| format!("clearing {failure}")));
        if let Err(error) = drop_daily_pin(archive) {
            notes.push(error);
        }
    }
    if apply {
        let list = fragments.list().to_vec();
        let layers = crate::memory_layers::MemoryLayers::beside_archive(archive);
        match layers.purge_holding(&list) {
            Ok((0, 0)) => {}
            Ok((gists, items)) => impact.cleared.push(format!("{gists} gists, {items} open items")),
            Err(error) => notes.push(note("clearing the gists", error.to_string())),
        }
        match crate::sleep::continuous::forget_holding(memory, &list) {
            Ok(0) => {}
            Ok(windows) => impact.cleared.push(format!("{windows} compaction windows")),
            Err(error) => notes.push(note("clearing the compaction windows", error.to_string())),
        }
        match crate::calls::purge_fragments(&data_dir, &list) {
            Ok(calls) => {
                if calls.calls + calls.mistakes + calls.history_lines > 0 {
                    impact.cleared.push(format!(
                        "{} saved calls ({} stored objects), {} mistakes",
                        calls.calls, calls.blobs, calls.mistakes
                    ));
                }
            }
            Err(error) => notes.push(note("dropping the saved model calls", error.to_string())),
        }
        crate::context_resume::forget(&data_dir);
    }

    // The checkpoints.
    match memory.strike_checkpoints(&strike, apply) {
        Ok(touched) => {
            for (index, at, count) in touched {
                impact.places.push(Place::new("checkpoints", format!("#{index} of {}", at.get(..16).unwrap_or(&at)), count));
            }
        }
        Err(error) => notes.push(note("the checkpoints", error.to_string())),
    }

    // Threads.
    match crate::threads::strike_file(&data_dir.join("threads.json"), &strike, apply) {
        Ok(touched) => {
            for (id, count) in touched {
                impact.places.push(Place::new("threads", format!("thread {id}"), count));
            }
        }
        Err(error) => notes.push(note("the threads", error.to_string())),
    }
    let backup = data_dir.join("threads.pre-world-model.json");
    match strike_file(&backup, fragments, apply) {
        Ok(0) => {}
        Ok(count) => impact.places.push(Place::new("threads", "the copy kept before the world model", count)),
        Err(error) => notes.push(note("the threads' old copy", error)),
    }

    // The threads and checkpoint as the last deep sleep rendered them, which
    // a rebuilt context is made from.
    match strike_file(&data_dir.join("sleep_snapshot.json"), fragments, apply) {
        Ok(0) => {}
        Ok(count) => impact.places.push(Place::new("sleep snapshot", "sleep_snapshot.json", count)),
        Err(error) => notes.push(note("the sleep snapshot", error)),
    }

    // The world model, then its citations.
    let world = archive.with_file_name("world_model.sqlite3");
    if let Err(error) = strike_sqlite(&world, "world model", WORLD, fragments, apply, &mut impact) {
        notes.push(note("the world model", error));
    }
    if apply {
        match restamp_world(&world, &restamp) {
            Ok(0) => {}
            Ok(n) => impact.cleared.push(format!("{n} world-model citations stamped again")),
            Err(error) => notes.push(note("stamping the world model's citations", error)),
        }
    }

    // Episodes, then their citations and indexes.
    let semantic = archive.with_extension("semantic.sqlite3");
    let struck = match strike_sqlite(&semantic, "episodes", EPISODES, fragments, apply, &mut impact) {
        Ok(changed) => changed.get("consolidated_episodes").cloned().unwrap_or_default(),
        Err(error) => {
            notes.push(note("the episodes", error));
            Vec::new()
        }
    };
    // Whatever became of their words: the events under them were struck.
    if apply {
        match settle_episodes(&semantic, &struck, &restamp) {
            Ok(0) => {}
            Ok(n) => impact.cleared.push(format!("{n} episode citations stamped again")),
            Err(error) => notes.push(note("stamping the episodes' citations", error)),
        }
    }

    // The board (and its predecessor's file, where one is still there) and
    // the open loops.
    for name in ["board.sqlite3", "cairn_board.sqlite3"] {
        if let Err(error) = strike_sqlite(&archive.with_file_name(name), "board", BOARD, fragments, apply, &mut impact) {
            notes.push(note("the board", error));
        }
    }
    if let Err(error) =
        strike_sqlite(&archive.with_file_name("open_loops.sqlite3"), "open loops", LOOPS, fragments, apply, &mut impact)
    {
        notes.push(note("the open loops", error));
    }

    // The typed history, and the prompts holding it in memory.
    match strike_history(&data_dir.join("input_history.txt"), fragments, apply) {
        Ok(0) => {}
        Ok(count) => {
            impact.places.push(Place::new("typed history", "input_history.txt", count));
            if apply {
                crate::jobs::note_history_purge();
                crate::terminal::mirror(|| crate::terminal::Mirrored::HistoryPurged);
            }
        }
        Err(error) => notes.push(note("the typed history", error)),
    }

    // The worker journal.
    match crate::subagent::strike_journal(&strike, apply) {
        Ok(touched) => {
            for (line, count) in touched {
                impact.places.push(Place::new("worker journal", format!("record {line}"), count));
            }
        }
        Err(error) => notes.push(note("the worker journal", error.to_string())),
    }

    // Skills, with their receipts and the deleted ones.
    if let Some(root) = crate::skills::root_path() {
        for path in files_under(root) {
            match strike_file(&path, fragments, apply) {
                Ok(0) => {}
                Ok(count) => {
                    let name = path.strip_prefix(root).unwrap_or(&path).display().to_string();
                    impact.places.push(Place::new("skills", name, count));
                }
                Err(error) => notes.push(note("a skill", error)),
            }
        }
    }

    // Redaction requests: a pending one's proposal is a copy of an event's
    // text, and the reason s0ma gave may quote it.
    let book = crate::redact::Book::beside_archive(archive);
    match crate::redact::strike_book(&book, &strike, apply) {
        Ok(touched) => {
            for (request, count) in touched {
                impact.places.push(Place::new("redaction requests", format!("#{request}"), count));
            }
        }
        Err(error) => notes.push(note("the redaction requests", error)),
    }

    // Last, a look over everything else, and (after a strike) over
    // everything: whatever still holds the text is said, not assumed away.
    if look_after {
        impact.unreachable = look(&data_dir, fragments, apply);
    }
    // The saved calls are searched only for strings long enough to mean
    // something there; a shorter one is left in them, and said.
    let short = fragments.list().iter().filter(|fragment| fragment.len() < crate::calls::MIN_FRAGMENT).count();
    if short > 0 {
        impact.unreachable.push(Place::new(
            "saved model calls",
            format!("not searched for strings under {} bytes", crate::calls::MIN_FRAGMENT),
            short,
        ));
    }
    impact.notes = notes;
    Ok(impact)
}

/// The pinned daily review window: it holds a hash of the events after the
/// last checkpoint, and with one of them struck every deep sleep would refuse
/// with "daily review source changed". It is only a cutoff to resume from, so
/// dropping it costs a re-derivation (as a job's close does).
pub fn drop_daily_pin(archive: &Path) -> std::result::Result<(), String> {
    let pin = archive.with_file_name("daily_sleep_window.json");
    if pin.exists() {
        fs::remove_file(&pin).map_err(|error| {
            format!(
                "the pinned daily review window was not dropped ({error}): until it goes, every deep sleep refuses \
                 with \"daily review source changed\""
            )
        })?;
    }
    Ok(())
}

/// After one event was rewritten by itself (an approved request), the
/// citations to it stamped again from what it says now, where they matched
/// `before`.
pub fn restamp_one(memory: &Hippocampus, before: RecalledEvent) -> Result<usize> {
    let Some(after) = memory.get_event(before.id).map_err(|error| error.to_string())? else { return Ok(0) };
    let mut restamp = Restamp::default();
    restamp.before.insert(before.id, before);
    restamp.after.insert(after.id, after);
    let archive = memory.archive_path();
    let world = restamp_world(&archive.with_file_name("world_model.sqlite3"), &restamp)?;
    let episodes = settle_episodes(&archive.with_extension("semantic.sqlite3"), &[], &restamp)?;
    Ok(world + episodes)
}

fn impact_note(what: &str, error: String) -> String {
    format!("{what}: {error}")
}

/// Files under the data directory this module does not strike, or (`after`
/// a strike) any file at all, that still hold one of the fragments.
///
/// The saved model calls are left out: they are compressed, so a byte search
/// cannot see into them, and `calls::purge_fragments` searched them already.
fn look(data_dir: &Path, fragments: &Fragments, after: bool) -> Vec<Place> {
    const STRUCK: &[&str] = &[
        "hippocampus.jsonl",
        "checkpoints.jsonl",
        "threads.json",
        "threads.pre-world-model.json",
        "world_model.sqlite3",
        "hippocampus.semantic.sqlite3",
        "hippocampus.fts.sqlite3",
        "memory_layers.sqlite3",
        "continuous_compaction.sqlite3",
        "board.sqlite3",
        "cairn_board.sqlite3",
        "open_loops.sqlite3",
        "input_history.txt",
        "subagent_events.jsonl",
        "sleep_snapshot.json",
        // The working context saved for a restart: dropped, not struck.
        "working_context.json",
    ];
    const LARGEST: u64 = 512 * 1024 * 1024;
    let needles = fragments.needles();
    let mut out = Vec::new();
    for path in files_under(data_dir) {
        let relative = path.strip_prefix(data_dir).unwrap_or(&path);
        let top = relative.components().next().map(|c| c.as_os_str().to_string_lossy().into_owned()).unwrap_or_default();
        if top == "calls" || top == "redactions" {
            continue;
        }
        let name = relative.display().to_string();
        let base = name.trim_end_matches("-wal").trim_end_matches("-shm");
        if !after && STRUCK.contains(&base) {
            continue;
        }
        if fs::metadata(&path).map(|m| m.len() > LARGEST).unwrap_or(true) {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else { continue };
        let count = occurrences(&bytes, &needles);
        if count > 0 {
            let store = if name.starts_with("weaknesses") { "weakness log" } else { "elsewhere" };
            out.push(Place::new(store, name, count));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fragments(list: &[&str]) -> Fragments {
        Fragments::new(&list.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn a_fragment_is_struck_everywhere_it_occurs_and_counted() {
        let f = fragments(&["sk-live-9f3a"]);
        let (text, count) = f.strike("key sk-live-9f3a and again sk-live-9f3a.").unwrap();
        assert_eq!(text, "key [redacted] and again [redacted].");
        assert_eq!(count, 2);
        assert!(f.strike("nothing here").is_none());
    }

    #[test]
    fn the_longer_of_two_overlapping_fragments_goes_whole() {
        let f = fragments(&["acme", "acme-prod-key-77"]);
        let (text, count) = f.strike("use acme-prod-key-77 for acme").unwrap();
        assert_eq!(text, "use [redacted] for [redacted]");
        assert_eq!(count, 2);
    }

    #[test]
    fn short_and_empty_selections_are_refused() {
        assert!(Fragments::new(&["abc".into()]).unwrap_err().contains("shorter than"));
        assert!(Fragments::new(&["   ".into()]).unwrap_err().contains("nothing"));
        assert!(Fragments::new(&[MARKER.into()]).is_err());
        // A piece of the marker would nest inside every marker at each pass.
        assert!(Fragments::new(&["redacted".into()]).unwrap_err().contains("marker"));
        assert!(Fragments::new(&["edacted]".into()]).is_err());
        // Trimmed, and each once.
        let f = Fragments::new(&[" token1 ".into(), "token1".into()]).unwrap();
        assert_eq!(f.list(), ["token1"]);
    }

    #[test]
    fn json_is_struck_in_its_words_and_never_in_its_keys_or_ids() {
        let f = fragments(&["pending", "acme-host"]);
        let mut value = serde_json::json!({
            "state": "pending",
            "statement": "acme-host is pending review",
            "supporting_evidence_ids": [1, 2],
            "world_record_ids": ["acme-host"],
            "subject_id": "acme-host",
            "pending": "a key that is a word",
            "alternatives": ["acme-host or not"],
        });
        let count = strike_json(&mut value, &|s| f.strike(s));
        assert_eq!(count, 3, "{value}");
        assert_eq!(value["state"], "pending", "a state is not a word");
        assert_eq!(value["subject_id"], "acme-host", "an id is not a word");
        assert_eq!(value["world_record_ids"][0], "acme-host", "nor is a list of them");
        assert_eq!(value["statement"], "[redacted] is [redacted] review");
        assert_eq!(value["alternatives"][0], "[redacted] or not");
        assert!(value.get("pending").is_some(), "keys are never struck");
    }

    #[test]
    fn an_excerpt_shows_the_fragment_in_its_place() {
        let f = fragments(&["hunter22"]);
        let excerpt = f.excerpt("the password\nis hunter22 for now").unwrap();
        assert_eq!(excerpt, format!("the password is {EXCERPT_OPEN}hunter22{EXCERPT_CLOSE} for now"));
        let long = format!("{}hunter22{}", "a".repeat(100), "b".repeat(100));
        let excerpt = f.excerpt(&long).unwrap();
        assert!(excerpt.starts_with('…') && excerpt.ends_with('…'), "{excerpt}");
    }

    #[test]
    fn a_byte_search_finds_the_plain_and_the_escaped_forms() {
        let f = fragments(&["two\nlines"]);
        let needles = f.needles();
        assert_eq!(occurrences(b"x two\nlines y", &needles), 1);
        assert_eq!(occurrences(br#"{"a":"two\nlines"}"#, &needles), 1);
        assert_eq!(occurrences(b"two<\\n>lines", &needles), 1);
        assert_eq!(occurrences(b"nothing", &needles), 0);
    }

    #[test]
    fn a_hash_that_matched_is_stamped_again_and_one_that_did_not_is_left() {
        let event = |content: &str| RecalledEvent {
            id: 7,
            timestamp: "2026-11-01T10:00:00+00:00".into(),
            speaker: Speaker::S0ma,
            content: content.into(),
        };
        let mut restamp = Restamp::default();
        restamp.before.insert(7, event("key sk-111122223333"));
        restamp.after.insert(7, event("key [redacted]"));
        let mut good = BTreeMap::from([(7, world_hash(&event("key sk-111122223333")))]);
        assert_eq!(restamp.apply(&mut good, world_hash), 1);
        assert_eq!(good[&7], world_hash(&event("key [redacted]")));
        let mut tampered = BTreeMap::from([(7, "0000".to_string())]);
        assert_eq!(restamp.apply(&mut tampered, world_hash), 0);
        assert_eq!(tampered[&7], "0000");
    }

    /// Every store this reaches, set up holding one secret, struck, and then
    /// searched byte by byte: the secret is left only where it is said to be.
    #[test]
    fn struck_from_every_store_and_left_only_where_it_says() {
        const SECRET: &str = "sk-live-4f9a8b7c6d5e";
        let dir = std::env::temp_dir().join(format!("s0ma-strike-all-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut memory = Hippocampus::new(dir.join("hippocampus.jsonl")).unwrap();
        memory.remember_operator("hello there".into()).unwrap();
        memory.remember_s0ma(format!("[tool:vm_shell]\nresult:\nTOKEN={SECRET}\ndone")).unwrap();
        let cited = memory.last_event_id();
        memory.remember_operator(format!("keep {SECRET} safe")).unwrap();

        // A world-model record citing the event, and quoting the key.
        let world = crate::world_model::path_for(&memory);
        crate::world_model::define(&world, "infra", "Infra", "hosts and keys", 0).unwrap();
        let record: crate::world_model::Record = serde_json::from_value(serde_json::json!({
            "id": "deploy-key", "model_id": "infra", "kind": "claim", "title": "deploy key",
            "statement": format!("the deploy key is {SECRET}"), "basis": "observed", "status": "supported",
            "supporting_evidence_ids": [cited]
        }))
        .unwrap();
        crate::world_model::put(&world, record, 0, "seen in a tool result", &memory).unwrap();

        // An episode citing it, with its source hash as sleep writes it.
        let semantic = memory.archive_path().with_extension("semantic.sqlite3");
        let event = memory.get_event(cited).unwrap().unwrap();
        {
            let db = Connection::open(&semantic).unwrap();
            db.execute_batch(
                "CREATE TABLE consolidated_episodes (id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, \
                 created_at TEXT NOT NULL, record_json TEXT NOT NULL, source_digests_json TEXT NOT NULL);
                 CREATE VIRTUAL TABLE episode_text USING fts5(episode_id UNINDEXED, text);
                 CREATE TABLE episode_fts_state (episode_id TEXT PRIMARY KEY, digest TEXT NOT NULL);
                 CREATE TABLE events(id INTEGER PRIMARY KEY, digest TEXT, epoch INTEGER);
                 CREATE TABLE pending(id INTEGER PRIMARY KEY, digest TEXT);
                 CREATE TABLE skipped(id INTEGER PRIMARY KEY, reason TEXT);",
            )
            .unwrap();
            let sources = BTreeMap::from([(cited, episode_hash(&event))]);
            db.execute(
                "INSERT INTO consolidated_episodes VALUES ('ep_1', 1, 'now', ?1, ?2)",
                params![
                    serde_json::json!({"title": "deploy", "lesson": format!("rotate {SECRET}"), "source_event_ids": [cited]}).to_string(),
                    serde_json::to_string(&sources).unwrap()
                ],
            )
            .unwrap();
            // The keyword index made from it, holding its words.
            db.execute("INSERT INTO episode_text VALUES ('ep_1', ?1)", [format!("deploy rotate {SECRET}")]).unwrap();
            db.execute("INSERT INTO episode_fts_state VALUES ('ep_1', 'old')", []).unwrap();
        }

        // A thread, held open the way the running s0ma holds its store.
        let mut threads = crate::threads::ThreadStore::new(dir.join("threads.json")).unwrap();
        let thread = threads
            .create(
                format!("rotate {SECRET}"),
                "a description".into(),
                crate::threads::ThreadCategory::Problem,
                crate::threads::ThreadState::Active,
                vec![cited],
            )
            .unwrap();

        // The board and the open loops, a checkpoint, the typed history.
        {
            let db = Connection::open(dir.join("board.sqlite3")).unwrap();
            db.execute_batch("CREATE TABLE facts(id TEXT, statement TEXT, basis TEXT);").unwrap();
            db.execute("INSERT INTO facts VALUES ('f1', ?1, 'observed')", [format!("key {SECRET} works")]).unwrap();
            let db = Connection::open(dir.join("open_loops.sqlite3")).unwrap();
            db.execute_batch("CREATE TABLE loops(id TEXT, text TEXT, resolution TEXT, status TEXT);").unwrap();
            db.execute("INSERT INTO loops VALUES ('l1', ?1, NULL, 'open')", [format!("revoke {SECRET}")]).unwrap();
        }
        memory.store_checkpoint(1, 3, 3, format!("s0ma saw {SECRET} in a deploy")).unwrap();
        fs::write(dir.join("input_history.txt"), format!("/help\nkeep {SECRET} safe\n")).unwrap();
        // The last deep sleep's rendering of the threads, which a rebuilt
        // context is made from, and the daily review it pinned.
        fs::write(
            dir.join("sleep_snapshot.json"),
            serde_json::json!({"schema_version": 1, "checkpoint": {"timestamp": "2026-11-01T10:00:00+00:00"},
                               "projections": {"threads": format!("rotate {SECRET}")}})
            .to_string(),
        )
        .unwrap();
        fs::write(dir.join("daily_sleep_window.json"), "{}").unwrap();
        // The weakness log, which nothing may rewrite.
        fs::write(dir.join("weaknesses.jsonl"), format!("{{\"excerpt\":\"{SECRET}\"}}\n")).unwrap();

        let fragments = fragments(&[SECRET]);
        let review = survey(&memory, &fragments).unwrap();
        let stores: Vec<String> = review.stores().into_iter().map(|(store, _, _)| store).collect();
        for store in
            ["archive", "checkpoints", "threads", "sleep snapshot", "world model", "episodes", "board", "open loops", "typed history"]
        {
            assert!(stores.iter().any(|s| s == store), "{store} missing from {stores:?}");
        }
        assert_eq!(review.unreachable.len(), 1, "{:?}", review.unreachable);
        assert_eq!(review.unreachable[0].store, "weakness log");
        // A short one is said not to be looked for in the saved calls.
        let short = survey(&memory, &super::tests::fragments(&["4f9a8b7c"])).unwrap();
        assert!(short.unreachable.iter().any(|place| place.store == "saved model calls"), "{:?}", short.unreachable);
        assert!(memory.get_event(cited).unwrap().unwrap().content.contains(SECRET), "a review changes nothing");

        let done = apply(&memory, &fragments).unwrap();
        assert!(done.notes.is_empty(), "{:?}", done.notes);
        assert_eq!(done.total(), review.total());

        // Nowhere but the weakness log, byte for byte.
        let needles = fragments.needles();
        let left: Vec<String> = files_under(&dir)
            .into_iter()
            .filter(|path| occurrences(&fs::read(path).unwrap(), &needles) > 0)
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(left, ["weaknesses.jsonl"]);
        assert_eq!(done.unreachable.len(), 1, "{:?}", done.unreachable);

        assert!(!dir.join("daily_sleep_window.json").exists(), "the pinned review would refuse every deep sleep");
        let snapshot = fs::read_to_string(dir.join("sleep_snapshot.json")).unwrap();
        assert!(snapshot.contains("2026-11-01T10:00:00+00:00"), "its timestamp is how it is matched: {snapshot}");

        // The event is the same event, struck.
        let after = memory.get_event(cited).unwrap().unwrap();
        assert_eq!(after.content, "[tool:vm_shell]\nresult:\nTOKEN=[redacted]\ndone");
        assert_eq!(after.timestamp, event.timestamp);

        // The record's evidence reads as it did, not as tampered with.
        let call = crate::types::ToolCall {
            id: "t".into(),
            kind: "function".into(),
            function: crate::types::FunctionCall {
                name: "world_record_get".into(),
                arguments: serde_json::json!({"id": "deploy-key"}).to_string(),
            },
        };
        let got: Value = serde_json::from_str(&crate::world_model::execute(&memory, &call).unwrap()).unwrap();
        let got: Value = serde_json::from_str(got["content"].as_str().unwrap()).unwrap();
        assert_eq!(got["evidence_integrity"], "unchanged", "{got}");
        assert_eq!(got["record"]["statement"], "the deploy key is [redacted]");

        // So does the episode's.
        let db = Connection::open(&semantic).unwrap();
        let sources: String =
            db.query_row("SELECT source_digests_json FROM consolidated_episodes", [], |r| r.get(0)).unwrap();
        let sources: BTreeMap<u64, String> = serde_json::from_str(&sources).unwrap();
        assert_eq!(sources[&cited], episode_hash(&after));

        // The running thread store reads the struck file, and a change it
        // makes afterwards does not write the old title back.
        assert_eq!(threads.get(thread.id).unwrap().title, "rotate [redacted]");
        threads.add_question(thread.id, "when?".into()).unwrap();
        let file = fs::read_to_string(dir.join("threads.json")).unwrap();
        assert!(!file.contains(SECRET) && file.contains("when?"), "{file}");

        // s0ma owes a settling pass, and taking it gives the strings up.
        // (Other tests strike too, so only that it is owed is certain here.)
        assert!(settle_owed());
        let _ = fs::remove_dir_all(&dir);
    }


    /// After a strike the last look goes over every file, the stores this
    /// module strikes too: if one still held the text, it is said.
    #[test]
    fn after_a_strike_the_look_reports_a_store_that_still_holds_it() {
        let dir = std::env::temp_dir().join(format!("s0ma-strike-look-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("calls")).unwrap();
        fs::write(dir.join("threads.json"), "[\"token-abcdef\"]").unwrap();
        fs::write(dir.join("calls").join("x.json"), "token-abcdef").unwrap();
        let f = fragments(&["token-abcdef"]);
        assert!(look(&dir, &f, false).is_empty(), "before: a store that is struck is not 'not reached'");
        let after = look(&dir, &f, true);
        assert_eq!(after.len(), 1, "{after:?}");
        assert_eq!((after[0].store.as_str(), after[0].item.as_str()), ("elsewhere", "threads.json"));
        let _ = fs::remove_dir_all(&dir);
    }

}
