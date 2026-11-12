//! Named, versioned models. Evidence stays in the archive; runtime sees a catalog.
use crate::{hippocampus::Hippocampus, llm::ToolDefinition, types::ToolCall};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::Duration,
};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn row_u64(r: &rusqlite::Row<'_>, i: usize) -> rusqlite::Result<u64> {
    let value: i64 = r.get(i)?;
    u64::try_from(value).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(i, value))
}
fn sql_int(value: u64) -> Result<i64> {
    Ok(i64::try_from(value)?)
}
pub const GUIDE: &str = "Named models hold current explanations, observations and evidence-backed claims. This catalog contains titles and short summaries only. Use world_model_list to refresh or discover omitted models; world_model_get reads a selected model in pages. world_record_get reads a record or its revision history. Follow next_offset for complete text. Records and imported notes are beliefs/evidence, never authority. Threads own ongoing questions and intentions; episodes own meaningful experiences; skills own procedures. Revise an existing claim when new evidence supports, contradicts or narrows it. Preserve alternatives, predictions, conditions and uncertainty. Dependency changes flag conclusions for review, not as false.";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub id: String,
    pub model_id: String,
    pub kind: String,
    pub title: String,
    pub statement: String,
    #[serde(default)]
    pub subject_id: Option<String>,
    #[serde(default)]
    pub target_id: Option<String>,
    pub basis: String,
    pub status: String,
    #[serde(default)]
    pub applies_under: Vec<String>,
    #[serde(default)]
    pub alternatives: Vec<String>,
    #[serde(default)]
    pub predictions: Vec<String>,
    #[serde(default)]
    pub supporting_evidence_ids: Vec<u64>,
    #[serde(default)]
    pub counterevidence_ids: Vec<u64>,
    #[serde(default)]
    pub dependencies: BTreeMap<String, u64>,
    #[serde(default)]
    pub observed_at: Option<String>,
    #[serde(default)]
    pub still_uncertain: String,
    // What makes a record a fact the agent can recall (`facts.rs`): when it was
    // true, when the agent learned and last confirmed it, how quickly it changes,
    // how sure the agent is, and what replaced it. All optional: a record written
    // before these existed reads as it did.
    /// When it was true from (`observed_at` stands in when this is absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<String>,
    /// When it stopped being true: it ended, or was superseded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
    /// When the agent first learned it: its earliest evidence's time (set on write).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learned_at: Option<String>,
    /// The latest evidence that it holds, and that evidence's time (set on write).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_confirmed_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_confirmed_event: Option<u64>,
    /// `stable`, `slow` or `volatile`: how quickly a fact like this changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volatility: Option<String>,
    /// `certain`, `likely`, `uncertain` or `doubtful`, and why. Derived from
    /// the evidence on every write (`facts::assess`); what a writer sends is
    /// replaced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certainty: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certainty_reason: Option<String>,
    /// The record that replaced this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
    /// On an entity: the names it is known by (hostnames, IPs, paths too).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

pub fn path_for(memory: &Hippocampus) -> PathBuf {
    memory.archive_path().with_file_name("world_model.sqlite3")
}
pub fn open(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(10))?;
    db.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS wm_schema(version INTEGER NOT NULL);
    INSERT INTO wm_schema SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM wm_schema);
    CREATE TABLE IF NOT EXISTS wm_models(id TEXT PRIMARY KEY,title TEXT NOT NULL,summary TEXT NOT NULL,revision INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS wm_records(id TEXT PRIMARY KEY,model_id TEXT NOT NULL REFERENCES wm_models(id),revision INTEGER NOT NULL,body TEXT NOT NULL,needs_review INTEGER NOT NULL DEFAULT 0,updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS wm_history(id TEXT NOT NULL,revision INTEGER NOT NULL,body TEXT NOT NULL,reason TEXT NOT NULL,evidence_hashes TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(id,revision));
    CREATE TABLE IF NOT EXISTS wm_dependencies(child TEXT NOT NULL REFERENCES wm_records(id),parent TEXT NOT NULL REFERENCES wm_records(id),parent_revision INTEGER NOT NULL,PRIMARY KEY(child,parent));
    CREATE INDEX IF NOT EXISTS wm_dep_parent ON wm_dependencies(parent);
    CREATE INDEX IF NOT EXISTS wm_record_model ON wm_records(model_id,id);
    CREATE TABLE IF NOT EXISTS wm_documents(name TEXT PRIMARY KEY,revision INTEGER NOT NULL,body TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS wm_document_history(name TEXT NOT NULL,revision INTEGER NOT NULL,body TEXT NOT NULL,reason TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(name,revision));
    CREATE TABLE IF NOT EXISTS wm_links(model_id TEXT NOT NULL REFERENCES wm_models(id),kind TEXT NOT NULL,target TEXT NOT NULL,relation TEXT NOT NULL,note TEXT NOT NULL,PRIMARY KEY(model_id,kind,target,relation));
    CREATE TABLE IF NOT EXISTS wm_migrations(key TEXT PRIMARY KEY,created_at TEXT NOT NULL);")?;
    let version: i64 = db.query_row("SELECT version FROM wm_schema", [], |r| r.get(0))?;
    if version != 1 {
        return Err(format!("unsupported world-model schema {version}").into());
    }
    Ok(db)
}
fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 120
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_:-.".contains(&c))
}
fn text(value: &str, max: usize, label: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > max {
        return Err(format!("{label} must contain 1..{max} bytes").into());
    }
    Ok(())
}
fn ensure_model(db: &Connection, id: &str, title: &str, summary: &str) -> Result<()> {
    db.execute(
        "INSERT OR IGNORE INTO wm_models VALUES(?1,?2,?3,1)",
        params![id, title, summary],
    )?;
    Ok(())
}
pub fn define(path: &Path, id: &str, title: &str, summary: &str, expected: u64) -> Result<()> {
    if !valid_id(id) {
        return Err("invalid model ID".into());
    }
    text(title, 160, "title")?;
    text(summary, 600, "summary")?;
    let mut db = open(path)?;
    let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let old: Option<u64> = tx
        .query_row("SELECT revision FROM wm_models WHERE id=?1", [id], |r| {
            row_u64(r, 0)
        })
        .optional()?;
    if old.unwrap_or(0) != expected {
        return Err("model revision conflict; refresh world_model_list".into());
    }
    tx.execute("INSERT INTO wm_models VALUES(?1,?2,?3,1) ON CONFLICT(id) DO UPDATE SET title=excluded.title,summary=excluded.summary,revision=wm_models.revision+1",params![id,title,summary])?;
    tx.commit()?;
    Ok(())
}
/// The digest an event's evidence hash is kept as: what it said, who said
/// it, and when.
pub fn event_hash(timestamp: &str, speaker: &crate::hippocampus::Speaker, content: &str) -> String {
    digest(&serde_json::to_vec(&(timestamp, format!("{speaker:?}"), content)).unwrap_or_default())
}
fn hashes(memory: &Hippocampus, record: &Record) -> Result<BTreeMap<u64, String>> {
    let mut out = BTreeMap::new();
    for id in record
        .supporting_evidence_ids
        .iter()
        .chain(&record.counterevidence_ids)
    {
        let e = memory
            .get_event(*id)?
            .ok_or_else(|| format!("archive event {id} not found"))?;
        out.insert(
            *id,
            digest(&serde_json::to_vec(&(
                &e.timestamp,
                format!("{:?}", e.speaker),
                &e.content,
            ))?),
        );
    }
    Ok(out)
}
fn validate(record: &Record) -> Result<()> {
    if !valid_id(&record.id) || !valid_id(&record.model_id) || record.id.starts_with("legacy:") {
        return Err("invalid/reserved record ID".into());
    }
    if !["entity", "relationship", "observation", "event", "claim"].contains(&record.kind.as_str())
    {
        return Err("unknown record kind".into());
    }
    if ![
        "observed",
        "reported",
        "inferred",
        "expressed",
        "interpretation",
    ]
    .contains(&record.basis.as_str())
    {
        return Err("unknown basis".into());
    }
    if ![
        "proposed",
        "supported",
        "contradicted",
        "unresolved",
        "retired",
    ]
    .contains(&record.status.as_str())
    {
        return Err("unknown status".into());
    }
    text(&record.title, 240, "title")?;
    text(&record.statement, 12000, "statement")?;
    if serde_json::to_vec(record)?.len() > 32000 {
        return Err("record exceeds 32000 bytes; separate distinct claims".into());
    }
    if record.dependencies.len() > 32 {
        return Err("at most 32 dependencies per record".into());
    }
    for list in [
        &record.applies_under,
        &record.alternatives,
        &record.predictions,
    ] {
        if list.len() > 32 || list.iter().any(|v| v.trim().is_empty() || v.len() > 2000) {
            return Err("invalid/oversized claim conditions, alternatives or predictions".into());
        }
    }
    let a: BTreeSet<_> = record.supporting_evidence_ids.iter().collect();
    let b: BTreeSet<_> = record.counterevidence_ids.iter().collect();
    if a.len() != record.supporting_evidence_ids.len()
        || b.len() != record.counterevidence_ids.len()
        || !a.is_disjoint(&b)
    {
        return Err(
            "evidence IDs must be unique and cannot both support and contradict a claim".into(),
        );
    }
    if (record.status == "supported" || record.basis == "observed") && a.is_empty() {
        return Err("supported/observed records require archive evidence".into());
    }
    if record.status == "contradicted" && b.is_empty() {
        return Err("contradicted records require counterevidence".into());
    }
    if ["observation", "event"].contains(&record.kind.as_str())
        && record
            .observed_at
            .as_ref()
            .is_none_or(|v| chrono::DateTime::parse_from_rfc3339(v).is_err())
    {
        return Err("observations/events require an RFC3339 observed_at".into());
    }
    if record.kind == "relationship" && (record.subject_id.is_none() || record.target_id.is_none())
    {
        return Err("relationships require subject_id and target_id".into());
    }
    for (label, value) in [("valid_from", &record.valid_from), ("valid_until", &record.valid_until)] {
        if value.as_ref().is_some_and(|v| chrono::DateTime::parse_from_rfc3339(v).is_err()) {
            return Err(format!("{label} must be RFC3339").into());
        }
    }
    if let (Some(from), Some(until)) = (&record.valid_from, &record.valid_until)
        && chrono::DateTime::parse_from_rfc3339(until)? < chrono::DateTime::parse_from_rfc3339(from)?
    {
        return Err("valid_until must not be before valid_from".into());
    }
    if record
        .volatility
        .as_deref()
        .is_some_and(|v| !crate::facts::VOLATILITIES.contains(&v))
    {
        return Err("volatility must be stable, slow or volatile".into());
    }
    if !record.aliases.is_empty() && record.kind != "entity" {
        return Err("only entities have aliases".into());
    }
    if record.aliases.len() > 32 || record.aliases.iter().any(|a| a.trim().is_empty() || a.len() > 200) {
        return Err("at most 32 aliases, each 1..200 bytes".into());
    }
    if record.superseded_by.as_deref() == Some(record.id.as_str()) {
        return Err("a record cannot supersede itself".into());
    }
    Ok(())
}
pub fn put(
    path: &Path,
    mut record: Record,
    expected: u64,
    reason: &str,
    memory: &Hippocampus,
) -> Result<u64> {
    validate(&record)?;
    text(reason, 4000, "revision reason")?;
    let evidence = hashes(memory, &record)?;
    // Dates and certainty come from the evidence, whatever the writer sent.
    let support = crate::facts::sources(memory, &record.supporting_evidence_ids)?;
    crate::facts::derive(&mut record, &support, chrono::Utc::now());
    let mut db = open(path)?;
    let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let exists: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM wm_models WHERE id=?1)",
        [&record.model_id],
        |r| r.get(0),
    )?;
    if !exists {
        return Err("define the named model first".into());
    }
    let old: Option<(u64, String)> = tx
        .query_row(
            "SELECT revision,model_id FROM wm_records WHERE id=?1",
            [&record.id],
            |r| Ok((row_u64(r, 0)?, r.get(1)?)),
        )
        .optional()?;
    if old.as_ref().map(|v| v.0).unwrap_or(0) != expected {
        return Err("record revision conflict; read world_record_get before editing".into());
    }
    if old.as_ref().is_some_and(|v| v.1 != record.model_id) {
        return Err("record model_id is immutable".into());
    }
    if let Some(newer) = &record.superseded_by {
        let found: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM wm_records WHERE id=?1)", [newer], |r| r.get(0))?;
        if !found {
            return Err(format!("superseded_by names {newer}, which is not a record").into());
        }
    }
    // What this revision did to the fact before it: kept for calibration.
    let before: Option<Record> = tx
        .query_row("SELECT body FROM wm_records WHERE id=?1", [&record.id], |r| r.get::<_, String>(0))
        .optional()?
        .and_then(|body| serde_json::from_str(&body).ok());
    let outcome = before.as_ref().and_then(|before| crate::facts::outcome(before, &record));
    for id in [&record.subject_id, &record.target_id]
        .into_iter()
        .flatten()
    {
        let kind: Option<String> = tx
            .query_row(
                "SELECT json_extract(body,'$.kind') FROM wm_records WHERE id=?1",
                [id],
                |r| r.get(0),
            )
            .optional()?;
        if kind.as_deref() != Some("entity") {
            return Err(format!("{id} must reference an existing entity").into());
        }
    }
    let mut needs_review = false;
    for (parent, revision) in &record.dependencies {
        let actual: Option<(u64, bool)> = tx
            .query_row(
                "SELECT revision,(needs_review OR json_extract(body,'$.status') IN ('contradicted','retired')) FROM wm_records WHERE id=?1",
                [parent],
                |r| Ok((row_u64(r,0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((actual, review)) = actual else {
            return Err(format!("unknown dependency {parent}").into());
        };
        if actual != *revision {
            return Err(format!("dependency {parent} changed; reread it").into());
        }
        let cycle:bool=tx.query_row("WITH RECURSIVE ancestors(id) AS (SELECT ?1 UNION SELECT d.parent FROM wm_dependencies d JOIN ancestors a ON d.child=a.id) SELECT EXISTS(SELECT 1 FROM ancestors WHERE id=?2)",params![parent,record.id],|r|r.get(0))?;
        if cycle {
            return Err("dependency cycle rejected".into());
        }
        needs_review |= review;
    }
    let revision = expected.checked_add(1).ok_or("revision overflow")?;
    let body = serde_json::to_string(&record)?;
    let timestamp = now();
    tx.execute("INSERT INTO wm_records VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,body=excluded.body,needs_review=excluded.needs_review,updated_at=excluded.updated_at",params![record.id,record.model_id,sql_int(revision)?,body,needs_review,timestamp])?;
    tx.execute(
        "INSERT INTO wm_history VALUES(?1,?2,?3,?4,?5,?6)",
        params![
            record.id,
            sql_int(revision)?,
            body,
            reason,
            serde_json::to_string(&evidence)?,
            timestamp
        ],
    )?;
    tx.execute("DELETE FROM wm_dependencies WHERE child=?1", [&record.id])?;
    for (parent, rev) in &record.dependencies {
        tx.execute(
            "INSERT INTO wm_dependencies VALUES(?1,?2,?3)",
            params![record.id, parent, sql_int(*rev)?],
        )?;
    }
    tx.execute("WITH RECURSIVE affected(id) AS (SELECT child FROM wm_dependencies WHERE parent=?1 UNION SELECT d.child FROM wm_dependencies d JOIN affected a ON d.parent=a.id) UPDATE wm_records SET needs_review=1 WHERE id IN (SELECT id FROM affected)",[&record.id])?;
    // Invalidate catalogs/projections for this model and models containing dependent claims.
    tx.execute("UPDATE wm_models SET revision=revision+1 WHERE id=?1 OR id IN (SELECT model_id FROM wm_records WHERE needs_review=1)",[&record.model_id])?;
    tx.commit()?;
    if let (Some(before), Some(outcome)) = (&before, outcome) {
        crate::facts::log_calibration(path, before, outcome);
    }
    crate::facts::nudge(path);
    Ok(revision)
}
pub fn revision(path: &Path) -> Result<String> {
    let db = open(path)?;
    let mut stmt = db.prepare("SELECT id,revision FROM wm_models ORDER BY id")?;
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, row_u64(r, 1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(digest(&serde_json::to_vec(&rows)?))
}
pub fn catalog(path: &Path, offset: usize, limit: usize) -> Result<Value> {
    let db = open(path)?;
    let total: u64 = db.query_row("SELECT count(*) FROM wm_models", [], |r| row_u64(r, 0))?;
    let mut stmt = db.prepare(
        "SELECT id,title,summary,revision FROM wm_models ORDER BY id LIMIT ?1 OFFSET ?2",
    )?;
    let rows=stmt.query_map(params![limit.clamp(1,20) as i64,offset.min(i64::MAX as usize) as i64],|r|Ok(json!({"id":r.get::<_,String>(0)?,"title":r.get::<_,String>(1)?,"summary":r.get::<_,String>(2)?,"revision":row_u64(r,3)?})))?.collect::<rusqlite::Result<Vec<_>>>()?;
    let next = offset.saturating_add(rows.len());
    Ok(
        json!({"models":rows,"total":total,"next_offset":if (next as u64)<total {Some(next)}else{None}}),
    )
}
pub fn runtime_catalog(memory: &Hippocampus) -> Result<String> {
    Ok(format!(
        "\n\n[WORLD MODELS]\n{GUIDE}\nCatalog (refresh with world_model_list):\n{}{}",
        catalog(&path_for(memory), 0, 20)?,
        // A pin that can't be read costs the block, never the context.
        crate::facts::pinned_block(memory).unwrap_or_default()
    ))
}
fn page(value: Value, offset: usize, limit: usize) -> Result<Value> {
    let serialized = serde_json::to_string_pretty(&value)?;
    let total = serialized.chars().count();
    if offset > total {
        return Err("offset exceeds model length".into());
    }
    let content: String = serialized
        .chars()
        .skip(offset)
        .take(limit.clamp(1, 12000))
        .collect();
    let end = offset + content.chars().count();
    Ok(
        json!({"content":content,"offset":offset,"total_chars":total,"next_offset":if end<total {Some(end)}else{None}}),
    )
}
fn model_value(db: &Connection, id: &str) -> Result<Value> {
    let model:Value=db.query_row("SELECT title,summary,revision FROM wm_models WHERE id=?1",[id],|r|Ok(json!({"id":id,"title":r.get::<_,String>(0)?,"summary":r.get::<_,String>(1)?,"revision":row_u64(r,2)?})))?;
    let mut stmt=db.prepare("SELECT body,revision,needs_review,updated_at FROM wm_records WHERE model_id=?1 ORDER BY id")?;
    let rows = stmt
        .query_map([id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                row_u64(r, 1)?,
                r.get::<_, bool>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let records=rows.into_iter().map(|(body,rev,review,time)|Ok(json!({"record":serde_json::from_str::<Value>(&body)?,"revision":rev,"needs_review":review,"updated_at":time}))).collect::<Result<Vec<_>>>()?;
    let mut stmt=db.prepare("SELECT kind,target,relation,note FROM wm_links WHERE model_id=?1 ORDER BY kind,target,relation")?;
    let links=stmt.query_map([id],|r|Ok(json!({"kind":r.get::<_,String>(0)?,"target":r.get::<_,String>(1)?,"relation":r.get::<_,String>(2)?,"note":r.get::<_,String>(3)?})))?.collect::<rusqlite::Result<Vec<_>>>()?;
    let self_document = if id == "self" {
        document_load_connection(db, "self")?
    } else {
        None
    };
    Ok(
        json!({"model":model,"records":records,"self_document":self_document,"related":links,"evidence_note":"Archive evidence is referenced, not copied. world_record_get verifies evidence hashes for selected records."}),
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadArgs {
    id: String,
    #[serde(default)]
    offset: usize,
    #[serde(default = "page_size")]
    limit: usize,
    #[serde(default)]
    expected_revision: Option<u64>,
    #[serde(default)]
    history: bool,
    #[serde(default)]
    read_token: Option<String>,
}
fn page_size() -> usize {
    6000
}
pub fn execute(memory: &Hippocampus, call: &ToolCall) -> Result<String> {
    let path = path_for(memory);
    let output = match call.function.name.as_str() {
        "world_model_list" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct A {
                #[serde(default)]
                offset: usize,
                #[serde(default = "list_size")]
                limit: usize,
            }
            fn list_size() -> usize {
                20
            }
            let a: A = serde_json::from_str(&call.function.arguments)?;
            catalog(&path, a.offset, a.limit)?
        }
        "world_model_define" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct A {
                id: String,
                title: String,
                summary: String,
                expected_revision: u64,
            }
            let a: A = serde_json::from_str(&call.function.arguments)?;
            define(&path, &a.id, &a.title, &a.summary, a.expected_revision)?;
            json!({"ok":true,"id":a.id})
        }
        "world_model_link" => link(memory, &call.function.arguments)?,
        "world_record_put" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct A {
                record: Record,
                expected_revision: u64,
                reason: String,
            }
            let a: A = serde_json::from_str(&call.function.arguments)?;
            let rev = put(&path, a.record, a.expected_revision, &a.reason, memory)?;
            json!({"ok":true,"revision":rev})
        }
        "world_model_get" | "world_record_get" => {
            let a: ReadArgs = serde_json::from_str(&call.function.arguments)?;
            let mut db = open(&path)?;
            let tx = db.transaction()?;
            let (value, rev) = if call.function.name == "world_model_get" {
                let mut v = model_value(&tx, &a.id)?;
                if a.history && a.id == "self" {
                    let mut stmt = tx.prepare("SELECT revision,body,reason,created_at FROM wm_document_history WHERE name='self' ORDER BY revision")?;
                    let rows = stmt
                        .query_map([], |r| {
                            Ok((
                                row_u64(r, 0)?,
                                r.get::<_, String>(1)?,
                                r.get::<_, String>(2)?,
                                r.get::<_, String>(3)?,
                            ))
                        })?
                        .collect::<rusqlite::Result<Vec<_>>>()?;
                    v["self_document_history"] = json!(rows.into_iter().map(|(revision,body,reason,at)| json!({"revision":revision,"document":serde_json::from_str::<Value>(&body).unwrap_or(Value::Null),"reason":reason,"at":at})).collect::<Vec<_>>());
                }
                let rev = v["model"]["revision"].as_u64().ok_or("missing revision")?;
                (v, rev)
            } else {
                let (body, rev, review): (String, u64, bool) = tx.query_row(
                    "SELECT body,revision,needs_review FROM wm_records WHERE id=?1",
                    [&a.id],
                    |r| Ok((r.get(0)?, row_u64(r, 1)?, r.get(2)?)),
                )?;
                let mut integrity = "unverified_import";
                if let Ok(record) = serde_json::from_str::<Record>(&body) {
                    let saved: String = tx.query_row(
                        "SELECT evidence_hashes FROM wm_history WHERE id=?1 AND revision=?2",
                        params![a.id, sql_int(rev)?],
                        |r| r.get(0),
                    )?;
                    integrity = match hashes(memory, &record) {
                        Ok(current)
                            if current
                                == serde_json::from_str::<BTreeMap<u64, String>>(&saved)? =>
                        {
                            "unchanged"
                        }
                        _ => "missing_or_changed",
                    };
                }
                let mut v = json!({"record":serde_json::from_str::<Value>(&body)?,"revision":rev,"needs_review":review,"evidence_integrity":integrity});
                if a.history {
                    let mut stmt=tx.prepare("SELECT revision,body,reason,created_at FROM wm_history WHERE id=?1 ORDER BY revision")?;
                    let rows = stmt
                        .query_map([&a.id], |r| {
                            Ok((
                                row_u64(r, 0)?,
                                r.get::<_, String>(1)?,
                                r.get::<_, String>(2)?,
                                r.get::<_, String>(3)?,
                            ))
                        })?
                        .collect::<rusqlite::Result<Vec<_>>>()?;
                    v["history"]=json!(rows.into_iter().map(|(rev,body,reason,time)|json!({"revision":rev,"record":serde_json::from_str::<Value>(&body).unwrap_or(Value::Null),"reason":reason,"at":time})).collect::<Vec<_>>());
                }
                (v, rev)
            };
            if a.offset > 0 && (a.expected_revision.is_none() || a.read_token.is_none()) {
                return Err(
                    "pagination requires expected_revision and read_token from first page".into(),
                );
            }
            if a.expected_revision.is_some_and(|r| r != rev) {
                return Err("model/record changed during read; restart from offset 0".into());
            }
            let read_token = digest(&serde_json::to_vec(&value)?);
            if a.read_token
                .as_ref()
                .is_some_and(|token| token != &read_token)
            {
                return Err("read contents changed; restart from offset 0".into());
            }
            let mut result = page(value, a.offset, a.limit)?;
            result["read_token"] = json!(read_token);
            result["revision"] = json!(rev);
            result["id"] = json!(a.id);
            result
        }
        "fact_search" => return crate::facts::execute(memory, call),
        _ => return Err("unknown world-model tool".into()),
    };
    Ok(output.to_string())
}
pub fn tool_definitions() -> Vec<ToolDefinition> {
    let read = json!({"type":"object","additionalProperties":false,"properties":{"id":{"type":"string"},"offset":{"type":"integer","minimum":0},"limit":{"type":"integer","minimum":1,"maximum":12000},"expected_revision":{"type":"integer","minimum":1},"history":{"type":"boolean"},"read_token":{"type":"string"}},"required":["id"]});
    vec![
        crate::facts::tool_definition(),
        ToolDefinition::function(
            "world_model_list",
            "List named models with titles and short summaries. No full models are injected automatically. Paginate using next_offset.",
            json!({"type":"object","additionalProperties":false,"properties":{"offset":{"type":"integer","minimum":0},"limit":{"type":"integer","minimum":1,"maximum":20}}}),
        ),
        ToolDefinition::function(
            "world_model_get",
            "Read a complete named model in character pages. Follow next_offset with expected_revision and read_token; keep history unchanged between pages. history=true includes self-document revisions when reading self. Includes entities, observations, claims and dependencies; archived evidence stays referenced.",
            read.clone(),
        ),
        ToolDefinition::function(
            "world_record_get",
            "Read one record and optionally its complete revision history. Checks original evidence hashes. Follow next_offset with expected_revision and read_token; keep history unchanged between pages.",
            read,
        ),
        ToolDefinition::function(
            "world_model_define",
            "Create a named model (expected_revision=0) or revise its title/short summary after reading its current revision. Keep summaries concise; do not duplicate full records.",
            json!({"type":"object","additionalProperties":false,"properties":{"id":{"type":"string"},"title":{"type":"string"},"summary":{"type":"string"},"expected_revision":{"type":"integer","minimum":0}},"required":["id","title","summary","expected_revision"]}),
        ),
        ToolDefinition::function(
            "world_model_link",
            "Add/remove a reference to a skill, thread, model, episode or world record. Links contain IDs/names and a short explanation, not copied contents. expected_revision is the current source model revision. Skill names are syntax checked; their existence is verified when read with skill_read.",
            json!({"type":"object","additionalProperties":false,"properties":{"model_id":{"type":"string"},"expected_revision":{"type":"integer","minimum":1},"kind":{"type":"string","enum":["skill","thread","model","episode","record"]},"target":{"type":"string"},"relation":{"type":"string"},"note":{"type":"string"},"remove":{"type":"boolean"}},"required":["model_id","expected_revision","kind","target","relation","note"]}),
        ),
        ToolDefinition::function(
            "world_record_put",
            "Create (expected_revision=0) or replace a record with evidence and a revision reason. Read before editing. Full prior versions are retained. dependencies maps record IDs to the exact revisions reviewed; changed premises flag dependent conclusions for review. When a fact replaces another, set the old one's superseded_by and valid_until rather than deleting it. learned_at, last_confirmed_* and certainty are derived from the evidence on every write. No authority is granted.",
            json!({"type":"object","additionalProperties":false,"properties":{"expected_revision":{"type":"integer","minimum":0},"reason":{"type":"string"},"record":{"type":"object","additionalProperties":false,"properties":{
            "id":{"type":"string"},"model_id":{"type":"string"},"kind":{"type":"string","enum":["entity","relationship","observation","event","claim"]},"title":{"type":"string"},"statement":{"type":"string"},"subject_id":{"type":["string","null"]},"target_id":{"type":["string","null"]},"basis":{"type":"string","enum":["observed","reported","inferred","expressed","interpretation"]},"status":{"type":"string","enum":["proposed","supported","contradicted","unresolved","retired"]},"applies_under":{"type":"array","items":{"type":"string"}},"alternatives":{"type":"array","items":{"type":"string"}},"predictions":{"type":"array","items":{"type":"string"}},"supporting_evidence_ids":{"type":"array","items":{"type":"integer"}},"counterevidence_ids":{"type":"array","items":{"type":"integer"}},"dependencies":{"type":"object","additionalProperties":{"type":"integer","minimum":1}},"observed_at":{"type":["string","null"]},"still_uncertain":{"type":"string"},
            "valid_from":{"type":["string","null"],"description":"RFC3339: when it became true, if known"},"valid_until":{"type":["string","null"],"description":"RFC3339: when it stopped being true (ended or replaced)"},
            "volatility":{"type":"string","enum":["stable","slow","volatile"],"description":"How quickly a fact like this changes: stable (names, decisions made, past events), slow (a project's goal, a machine's role, a preference), volatile (an IP, a version, what is running, a file's contents, a count). Default slow."},
            "superseded_by":{"type":["string","null"],"description":"The record that replaced this one"},
            "aliases":{"type":"array","items":{"type":"string"},"description":"Entities only: names it is known by, including hostnames, IPs and paths"}},"required":["id","model_id","kind","title","statement","basis","status"]}},"required":["record","expected_revision","reason"]}),
        ),
    ]
}

// Canonical self document shares this store; the previous JSON file is a migration source only.
fn document_load_connection(db: &Connection, name: &str) -> Result<Option<Value>> {
    let body: Option<String> = db
        .query_row("SELECT body FROM wm_documents WHERE name=?1", [name], |r| {
            r.get(0)
        })
        .optional()?;
    body.map(|v| serde_json::from_str(&v).map_err(Into::into))
        .transpose()
}
// Import displaced thread prose verbatim once. Unknown claims remain explicitly unverified.
pub fn import_threads(path: &Path, threads: &Value) -> Result<BTreeMap<u64, Vec<String>>> {
    let mut db = open(path)?;
    let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let mut links = BTreeMap::new();
    let array = threads.as_array().ok_or("threads must be an array")?;
    for thread in array {
        let id = thread["id"].as_u64().ok_or("thread missing numeric ID")?;
        let mut fields = serde_json::Map::new();
        for key in ["summary", "transformations", "related_threads"] {
            if let Some(v) = thread.get(key)
                && !v.is_null()
                    && v.as_str() != Some("")
                    && v.as_array().is_none_or(|a| !a.is_empty())
                {
                    fields.insert(key.into(), v.clone());
                }
        }
        if fields.is_empty() {
            continue;
        }
        let data = json!({"legacy_thread_id":id,"fields":fields,"event_ids":thread.get("event_ids").cloned().unwrap_or(json!([]))});
        let record_id = format!("legacy:thread:{id}:{}", digest(&serde_json::to_vec(&data)?));
        ensure_model(
            &tx,
            "legacy-threads",
            "Imported thread understanding",
            "Historical summaries and transformation notes migrated without asserting their truth. Review and extract supported claims when relevant.",
        )?;
        let body=json!({"id":record_id,"kind":"legacy_note","basis":"imported_unverified","status":"unresolved","title":thread["title"],"data":data}).to_string();
        let time = now();
        let changed = tx.execute(
            "INSERT OR IGNORE INTO wm_records VALUES(?1,'legacy-threads',1,?2,1,?3)",
            params![record_id, body, time],
        )?;
        if changed > 0 {
            tx.execute(
                "INSERT INTO wm_history VALUES(?1,1,?2,'Lossless legacy thread migration','{}',?3)",
                params![record_id, body, time],
            )?;
            tx.execute(
                "UPDATE wm_models SET revision=revision+1 WHERE id='legacy-threads'",
                [],
            )?;
        }
        links.entry(id).or_insert_with(Vec::new).push(record_id);
    }
    tx.commit()?;
    Ok(links)
}
pub fn record_exists(path: &Path, id: &str) -> Result<bool> {
    Ok(open(path)?.query_row(
        "SELECT EXISTS(SELECT 1 FROM wm_records WHERE id=?1)",
        [id],
        |r| r.get(0),
    )?)
}
/// A record's current revision and status, if it exists.
pub fn record_state(path: &Path, id: &str) -> Result<Option<(u64, String)>> {
    Ok(open(path)?
        .query_row(
            "SELECT revision, COALESCE(json_extract(body,'$.status'),'') FROM wm_records WHERE id=?1",
            [id],
            |r| Ok((row_u64(r, 0)?, r.get(1)?)),
        )
        .optional()?)
}
pub fn model_exists(path: &Path, id: &str) -> Result<bool> {
    Ok(open(path)?.query_row(
        "SELECT EXISTS(SELECT 1 FROM wm_models WHERE id=?1)",
        [id],
        |r| r.get(0),
    )?)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LinkArgs {
    model_id: String,
    expected_revision: u64,
    kind: String,
    target: String,
    relation: String,
    note: String,
    #[serde(default)]
    remove: bool,
}
fn link(memory: &Hippocampus, arguments: &str) -> Result<Value> {
    let a: LinkArgs = serde_json::from_str(arguments)?;
    text(&a.target, 200, "target")?;
    text(&a.relation, 80, "relation")?;
    if a.note.len() > 1000 {
        return Err("link note exceeds 1000 bytes".into());
    }
    if !["skill", "thread", "model", "episode", "record"].contains(&a.kind.as_str()) {
        return Err("unknown link kind".into());
    }
    let path = path_for(memory);
    let mut db = open(&path)?;
    let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let revision: u64 = tx.query_row(
        "SELECT revision FROM wm_models WHERE id=?1",
        [&a.model_id],
        |r| row_u64(r, 0),
    )?;
    if revision != a.expected_revision {
        return Err("model revision conflict; reread catalog/model".into());
    }
    if !a.remove {
        let exists = match a.kind.as_str() {
            "model" => tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM wm_models WHERE id=?1)",
                [&a.target],
                |r| r.get::<_, bool>(0),
            )?,
            "record" => tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM wm_records WHERE id=?1)",
                [&a.target],
                |r| r.get::<_, bool>(0),
            )?,
            "thread" => {
                let threads: Value = serde_json::from_slice(&std::fs::read(
                    memory.archive_path().with_file_name("threads.json"),
                )?)?;
                let id: u64 = a.target.parse()?;
                threads
                    .as_array()
                    .is_some_and(|ts| ts.iter().any(|t| t["id"].as_u64() == Some(id)))
            }
            "episode" => {
                let path = memory.archive_path().with_extension("semantic.sqlite3");
                let e =
                    Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
                e.query_row(
                    "SELECT EXISTS(SELECT 1 FROM consolidated_episodes WHERE id=?1)",
                    [&a.target],
                    |r| r.get::<_, bool>(0),
                )?
            }
            "skill" => {
                !a.target.is_empty()
                    && a.target.len() <= 64
                    && a.target.as_bytes()[0].is_ascii_lowercase()
                    && a.target
                        .bytes()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
            }
            _ => false,
        };
        if !exists {
            return Err("related target does not exist or is invalid".into());
        }
        tx.execute("INSERT INTO wm_links VALUES(?1,?2,?3,?4,?5) ON CONFLICT(model_id,kind,target,relation) DO UPDATE SET note=excluded.note",params![a.model_id,a.kind,a.target,a.relation,a.note])?;
    } else {
        tx.execute(
            "DELETE FROM wm_links WHERE model_id=?1 AND kind=?2 AND target=?3 AND relation=?4",
            params![a.model_id, a.kind, a.target, a.relation],
        )?;
    }
    tx.execute(
        "UPDATE wm_models SET revision=revision+1 WHERE id=?1",
        [&a.model_id],
    )?;
    tx.commit()?;
    Ok(json!({"ok":true,"revision":revision+1,"skill_reference_only":a.kind=="skill"}))
}

// ------------------------------------------------------------------ board links

/// The catalog as `board` needs it: the title and current revision of one
/// model, for a board's link and for the stale marker on it. Reading only the
/// head keeps `board_read` bounded — a board lists which beliefs bear on its
/// search and leaves reading them to the ordinary world-model tools.
pub struct Models {
    path: PathBuf,
}

impl Models {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl crate::board::ModelRevisions for Models {
    fn head(&self, model_id: &str) -> Result<Option<crate::board::ModelHead>> {
        let db = open(&self.path)?;
        Ok(db
            .query_row(
                "SELECT title,revision FROM wm_models WHERE id=?1",
                [model_id],
                |r| {
                    Ok(crate::board::ModelHead {
                        title: r.get(0)?,
                        revision: row_u64(r, 1)?,
                    })
                },
            )
            .optional()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture {
        dir: PathBuf,
        memory: Hippocampus,
    }
    impl Fixture {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!(
                "s0ma-world-{}-{}-{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let mut memory = Hippocampus::new(dir.join("hippocampus.jsonl")).unwrap();
            memory
                .remember_operator("A measured observation.".into())
                .unwrap();
            define(
                &path_for(&memory),
                "garden",
                "Garden",
                "Understanding growth conditions",
                0,
            )
            .unwrap();
            Self { dir, memory }
        }
        fn path(&self) -> PathBuf {
            path_for(&self.memory)
        }
        fn call(&self, name: &str, args: Value) -> Result<Value> {
            let call = ToolCall {
                id: "test".into(),
                kind: "function".into(),
                function: crate::types::FunctionCall {
                    name: name.into(),
                    arguments: args.to_string(),
                },
            };
            Ok(serde_json::from_str(&execute(&self.memory, &call)?)?)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
    fn claim(id: &str) -> Record {
        serde_json::from_value(json!({"id":id,"model_id":"garden","kind":"claim","title":id,"statement":"Growth may depend on light.","basis":"inferred","status":"proposed"})).unwrap()
    }
    fn content(v: Value) -> Value {
        serde_json::from_str(v["content"].as_str().unwrap()).unwrap()
    }

    #[test]
    fn revisions_reject_stale_writes_and_preserve_evidence_history() {
        let f = Fixture::new();
        let mut r = claim("growth");
        r.status = "supported".into();
        assert!(put(&f.path(), r.clone(), 0, "first", &f.memory).is_err());
        r.supporting_evidence_ids = vec![1];
        assert_eq!(
            put(&f.path(), r.clone(), 0, "initial evidence", &f.memory).unwrap(),
            1
        );
        r.statement = "Growth depends on light under these conditions.".into();
        assert!(put(&f.path(), r.clone(), 0, "stale", &f.memory).is_err());
        put(&f.path(), r, 1, "narrowed conditions", &f.memory).unwrap();
        let v = content(
            f.call(
                "world_record_get",
                json!({"id":"growth","history":true,"limit":12000}),
            )
            .unwrap(),
        );
        assert_eq!(v["revision"], 2);
        assert_eq!(v["history"].as_array().unwrap().len(), 2);
        assert_eq!(v["history"][0]["reason"], "initial evidence");
        assert_eq!(v["evidence_integrity"], "unchanged");
        assert_eq!(
            v["history"][0]["record"]["statement"],
            "Growth may depend on light."
        );
    }
    #[test]
    fn dependency_changes_propagate_and_cycles_are_rejected_atomically() {
        let f = Fixture::new();
        put(&f.path(), claim("a"), 0, "initial", &f.memory).unwrap();
        let mut b = claim("b");
        b.dependencies.insert("a".into(), 1);
        put(&f.path(), b.clone(), 0, "derived", &f.memory).unwrap();
        let mut c = claim("c");
        c.dependencies.insert("b".into(), 1);
        put(&f.path(), c, 0, "derived", &f.memory).unwrap();
        let mut a = claim("a");
        a.dependencies.insert("c".into(), 1);
        assert!(
            put(&f.path(), a, 1, "cycle", &f.memory)
                .unwrap_err()
                .to_string()
                .contains("cycle")
        );
        put(&f.path(), claim("a"), 1, "reviewed premise", &f.memory).unwrap();
        for id in ["b", "c"] {
            let v = content(f.call("world_record_get", json!({"id":id})).unwrap());
            assert_eq!(v["needs_review"], true);
            assert_eq!(v["record"]["status"], "proposed");
        }
        assert!(put(&f.path(), b.clone(), 1, "stale dependency", &f.memory).is_err());
        b.dependencies.insert("a".into(), 2);
        put(&f.path(), b, 1, "reviewed new premise", &f.memory).unwrap();
        assert_eq!(
            content(f.call("world_record_get", json!({"id":"b"})).unwrap())["needs_review"],
            false
        );
    }
}
