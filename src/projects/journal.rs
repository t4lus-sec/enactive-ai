//! The stores' databases as an index of the project files. Every row a
//! board's or an audit's store changes is written, once its transaction
//! commits, as a line in that board's `.board` or that audit's `.audit`
//! file: the row as it now stands, or its key when it is gone. Replaying a
//! file's lines rebuilds its rows, so the files are the record and the
//! databases can always be made again from them.
//!
//! The stores open their connections through [`wrap`]: temporary triggers
//! on each table note every row changed (new values, or the old key for a
//! delete) in a temporary table of the connection's. A rollback takes the
//! notes with it, so what is there when the connection closes is what
//! committed, and it goes to the files. (Triggers, not SQLite's pre-update
//! hook: that needs bindings generated at build time, with clang.)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::types::{Value as SqlValue, ValueRef};
use rusqlite::{Connection, params_from_iter};
use serde_json::{Map, Value};

use super::Kind;

/// Which database a connection is to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Store {
    Board,
    Flow,
    Audit,
}

/// The tables kept in files: the store, the table, the column naming the
/// board or audit a row belongs to, which of the two, and the word its
/// lines carry.
pub const TABLES: &[(Store, &str, &str, Kind, &str)] = &[
    (Store::Board, "boards", "id", Kind::Board, "board"),
    (Store::Board, "intents", "board_id", Kind::Board, "lead"),
    (Store::Board, "facts", "board_id", Kind::Board, "finding"),
    (Store::Board, "hints", "board_id", Kind::Board, "hint"),
    (Store::Board, "comments", "board_id", Kind::Board, "comment"),
    (Store::Board, "board_models", "board_id", Kind::Board, "model"),
    (Store::Flow, "flow", "board_id", Kind::Board, "flow"),
    (Store::Flow, "notes", "board_id", Kind::Board, "note"),
    (Store::Audit, "audits", "name", Kind::Audit, "audit"),
    (Store::Audit, "claims", "audit", Kind::Audit, "claim"),
    (Store::Audit, "checks", "audit", Kind::Audit, "check"),
    (Store::Audit, "evidence", "audit", Kind::Audit, "evidence"),
    (Store::Audit, "links", "audit", Kind::Audit, "link"),
    (Store::Audit, "log", "audit", Kind::Audit, "log"),
    (Store::Audit, "runs", "audit", Kind::Audit, "run"),
];

fn table(store: Store, name: &str) -> Option<&'static (Store, &'static str, &'static str, Kind, &'static str)> {
    TABLES.iter().find(|t| t.0 == store && t.1 == name)
}

fn table_by_name(name: &str) -> Option<&'static (Store, &'static str, &'static str, Kind, &'static str)> {
    TABLES.iter().find(|t| t.1 == name)
}

/// A table's columns and its key, read once per database.
#[derive(Clone, Debug)]
struct Columns {
    names: Vec<String>,
    key: Vec<String>,
}

static COLUMNS: Mutex<Option<HashMap<(PathBuf, String), Columns>>> = Mutex::new(None);

fn columns_of(db: &Connection, path: &Path, name: &str) -> Option<Columns> {
    let mut cache = crate::util::locked(&COLUMNS);
    let cache = cache.get_or_insert_with(HashMap::new);
    let key = (path.to_path_buf(), name.to_string());
    if let Some(found) = cache.get(&key) {
        return Some(found.clone());
    }
    let mut statement = db.prepare(&format!("PRAGMA table_info({name})")).ok()?;
    let rows: Vec<(String, i64)> = statement.query_map([], |row| Ok((row.get::<_, String>(1)?, row.get::<_, i64>(5)?))).ok()?.filter_map(|r| r.ok()).collect();
    if rows.is_empty() {
        return None;
    }
    let mut keyed: Vec<(i64, String)> = rows.iter().filter(|(_, pk)| *pk > 0).map(|(name, pk)| (*pk, name.clone())).collect();
    keyed.sort();
    let found = Columns { names: rows.into_iter().map(|(name, _)| name).collect(), key: keyed.into_iter().map(|(_, name)| name).collect() };
    cache.insert(key, found.clone());
    Some(found)
}

/// One row changed: its table, the board or audit it belongs to, and the
/// row as it now is (None: gone, with its key).
#[derive(Clone, Debug)]
pub struct Change {
    pub table: &'static str,
    pub kind: Kind,
    pub owner: String,
    pub row: Option<Map<String, Value>>,
    pub key: Map<String, Value>,
}

/// A store's connection: the connection itself, and the store whose
/// changes it notes (none: not journaled), to be written to the files once
/// it closes.
pub struct Db {
    conn: Option<Connection>,
    noted: Option<(Store, HashMap<&'static str, Columns>)>,
}

/// The temporary table the triggers note changes in.
const NOTES: &str = "s0ma_changes";

impl std::ops::Deref for Db {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        self.conn.as_ref().expect("open")
    }
}

impl std::ops::DerefMut for Db {
    fn deref_mut(&mut self) -> &mut Connection {
        self.conn.as_mut().expect("open")
    }
}

impl Drop for Db {
    fn drop(&mut self) {
        let Some(conn) = self.conn.take() else { return };
        let changes = match &self.noted {
            Some((store, columns)) => noted(&conn, *store, columns),
            None => Vec::new(),
        };
        drop(conn);
        if !changes.is_empty() {
            write(changes);
        }
    }
}

/// What the triggers noted and committed, in order.
fn noted(conn: &Connection, store: Store, columns: &HashMap<&'static str, Columns>) -> Vec<Change> {
    // A transaction still open is rolled back as the connection closes:
    // what it noted goes with it, what committed before it stays.
    if !conn.is_autocommit() && conn.execute_batch("ROLLBACK").is_err() {
        return Vec::new();
    }
    let Ok(mut statement) = conn.prepare(&format!("SELECT tbl, gone, row FROM temp.{NOTES} ORDER BY seq")) else { return Vec::new() };
    let rows: Vec<(String, bool, String)> = match statement.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))) {
        Ok(rows) => rows.filter_map(Result::ok).collect(),
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for (name, gone, text) in rows {
        let Some(&(_, table_name, owner_column, kind, _)) = table(store, &name) else { continue };
        let Some(cols) = columns.get(table_name) else { continue };
        let Ok(Value::Object(mut row)) = serde_json::from_str::<Value>(&text) else { continue };
        // As the columns are ordered, with blobs as `blob:<hex>`.
        let mut ordered = Map::new();
        for column in &cols.names {
            let value = row.remove(column).unwrap_or(Value::Null);
            ordered.insert(column.clone(), value);
        }
        let owner = match ordered.get(owner_column) {
            Some(Value::String(owner)) => owner.clone(),
            _ => continue,
        };
        let key: Map<String, Value> = cols.key.iter().map(|k| (k.clone(), ordered.get(k).cloned().unwrap_or(Value::Null))).collect();
        out.push(Change { table: table_name, kind, owner, row: (!gone).then_some(ordered), key });
    }
    out
}

/// A column's value for the notes: a blob as `blob:<hex>` (JSON holds none).
fn noted_value(side: &str, column: &str) -> String {
    format!("CASE WHEN typeof({side}.\"{column}\")='blob' THEN 'blob:'||lower(hex({side}.\"{column}\")) ELSE {side}.\"{column}\" END")
}

/// The triggers on `name` that note its changes.
fn triggers(name: &str, cols: &Columns) -> String {
    let object = |side: &str, columns: &[String]| {
        let pairs: Vec<String> = columns.iter().map(|c| format!("'{c}', {}", noted_value(side, c))).collect();
        format!("json_object({})", pairs.join(", "))
    };
    let all_new = object("NEW", &cols.names);
    // A delete notes the row as it was: its key, and the owner (what finds
    // its file).
    let gone = object("OLD", &cols.names);
    format!(
        "CREATE TEMP TRIGGER IF NOT EXISTS s0ma_{name}_insert AFTER INSERT ON main.{name} BEGIN \
           INSERT INTO temp.{NOTES}(tbl, gone, row) VALUES('{name}', 0, {all_new}); END;\n\
         CREATE TEMP TRIGGER IF NOT EXISTS s0ma_{name}_update AFTER UPDATE ON main.{name} BEGIN \
           INSERT INTO temp.{NOTES}(tbl, gone, row) VALUES('{name}', 0, {all_new}); END;\n\
         CREATE TEMP TRIGGER IF NOT EXISTS s0ma_{name}_delete AFTER DELETE ON main.{name} BEGIN \
           INSERT INTO temp.{NOTES}(tbl, gone, row) VALUES('{name}', 1, {gone}); END;\n"
    )
}

thread_local! {
    /// Replaying files into the index: nothing goes back to the files.
    static QUIET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A store's connection, its changes going to the files (when there is a
/// projects folder, and not while files are replayed into it).
pub fn wrap(conn: Connection, store: Store, path: &Path) -> Db {
    if super::root().is_none() || QUIET.with(|q| q.get()) {
        return Db { conn: Some(conn), noted: None };
    }
    let mut columns: HashMap<&'static str, Columns> = HashMap::new();
    let mut sql = format!("CREATE TEMP TABLE IF NOT EXISTS {NOTES}(seq INTEGER PRIMARY KEY, tbl TEXT NOT NULL, gone INTEGER NOT NULL, row TEXT NOT NULL);\n");
    for (s, name, ..) in TABLES {
        if *s == store
            && let Some(found) = columns_of(&conn, path, name)
        {
            sql.push_str(&triggers(name, &found));
            columns.insert(name, found);
        }
    }
    if let Err(error) = conn.execute_batch(&sql) {
        super::warn_once(&format!("[projects] changes can't be noted for the files: {error}"));
        return Db { conn: Some(conn), noted: None };
    }
    Db { conn: Some(conn), noted: Some((store, columns)) }
}

fn json_of(value: ValueRef<'_>) -> Value {
    match value {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => Value::from(i),
        ValueRef::Real(f) => serde_json::Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null),
        ValueRef::Text(t) => Value::String(String::from_utf8_lossy(t).into_owned()),
        ValueRef::Blob(b) => Value::String(format!("blob:{}", b.iter().map(|x| format!("{x:02x}")).collect::<String>())),
    }
}

fn sql_of(value: &Value) -> SqlValue {
    match value {
        Value::Null => SqlValue::Null,
        Value::Bool(b) => SqlValue::Integer(i64::from(*b)),
        Value::Number(n) => n.as_i64().map(SqlValue::Integer).or_else(|| n.as_f64().map(SqlValue::Real)).unwrap_or(SqlValue::Null),
        Value::String(s) => match s.strip_prefix("blob:") {
            Some(hex) if hex.len() % 2 == 0 && hex.chars().all(|c| c.is_ascii_hexdigit()) => {
                SqlValue::Blob((0..hex.len()).step_by(2).filter_map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok()).collect())
            }
            _ => SqlValue::Text(s.clone()),
        },
        other => SqlValue::Text(other.to_string()),
    }
}

/// The line a change is written as.
pub fn line_of(change: &Change) -> Map<String, Value> {
    let word = table_by_name(change.table).map(|t| t.4).unwrap_or("row");
    let mut line = Map::new();
    match &change.row {
        Some(row) => {
            line.insert("op".into(), Value::String(word.into()));
            line.insert("table".into(), Value::String(change.table.into()));
            line.insert("row".into(), Value::Object(row.clone()));
        }
        None => {
            line.insert("op".into(), Value::String(format!("{word}-gone")));
            line.insert("table".into(), Value::String(change.table.into()));
            line.insert("key".into(), Value::Object(change.key.clone()));
        }
    }
    line
}

/// Committed changes to the files, a file at a time.
fn write(changes: Vec<Change>) {
    let mut by_owner: Vec<((Kind, String), Vec<Change>)> = Vec::new();
    for change in changes {
        let key = (change.kind, change.owner.clone());
        match by_owner.iter_mut().find(|(k, _)| *k == key) {
            Some((_, list)) => list.push(change),
            None => by_owner.push((key, vec![change])),
        }
    }
    for ((kind, owner), changes) in by_owner {
        if let Err(error) = super::record(kind, &owner, changes.iter().map(line_of).collect()) {
            super::warn_once(&format!("[projects] {} {owner} could not be written to its file: {error}", kind.word()));
        }
    }
}

/// A board's or audit's rows, read from the index: every table's, as the
/// lines a snapshot holds (`{"table":…, "row":…}`), boards' parents first.
pub fn rows_of(kind: Kind, owner: &str) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for (store, name, owner_column, k, _) in TABLES {
        if *k != kind {
            continue;
        }
        let Some(path) = super::store_path(*store) else { continue };
        if !path.exists() {
            continue;
        }
        let db = crate::db::open(&path).map_err(|e| e.to_string())?;
        let Some(cols) = columns_of(&db, &path, name) else { continue };
        let sql = format!("SELECT {} FROM {name} WHERE {owner_column}=?1", cols.names.join(","));
        let mut statement = db.prepare(&sql).map_err(|e| e.to_string())?;
        let rows = statement
            .query_map([owner], |row| {
                let mut map = Map::new();
                for (i, column) in cols.names.iter().enumerate() {
                    map.insert(column.clone(), json_of(row.get_ref(i)?));
                }
                Ok(map)
            })
            .map_err(|e| e.to_string())?;
        for row in rows {
            out.push(serde_json::json!({"table": name, "row": Value::Object(row.map_err(|e| e.to_string())?)}));
        }
    }
    Ok(out)
}

/// A board's or audit's rows taken out of the index (archived, trashed,
/// or gone from the files), children first.
pub fn remove(kind: Kind, owner: &str) -> Result<(), String> {
    QUIET.with(|q| q.set(true));
    let result = (|| {
        for store in [Store::Board, Store::Flow, Store::Audit] {
            let Some(path) = super::store_path(store) else { continue };
            if !path.exists() {
                continue;
            }
            let mut db = crate::db::open(&path).map_err(|e| e.to_string())?;
            let _ = db.execute_batch("PRAGMA foreign_keys=ON;");
            let tx = db.transaction().map_err(|e| e.to_string())?;
            for (s, name, owner_column, k, _) in TABLES.iter().rev() {
                if *s == store && *k == kind && has_table(&tx, name) {
                    tx.execute(&format!("DELETE FROM {name} WHERE {owner_column}=?1"), [owner]).map_err(|e| e.to_string())?;
                }
            }
            tx.commit().map_err(|e| e.to_string())?;
        }
        Ok(())
    })();
    QUIET.with(|q| q.set(false));
    result
}

/// A file's lines into the index: its rows there replaced by what the
/// file says now. A row whose key another board or audit holds in the index
/// is left out (never taken from it); how many were, returned.
pub fn replay(kind: Kind, owner: &str, log: &super::log::Log) -> Result<usize, String> {
    remove(kind, owner)?;
    QUIET.with(|q| q.set(true));
    let result = (|| {
        let mut left_out = 0;
        // Every row's latest state, in the order first seen (parents first).
        let mut order: Vec<(String, String)> = Vec::new();
        let mut state: HashMap<(String, String), Option<Map<String, Value>>> = HashMap::new();
        let mut apply = |name: &str, row: Option<&Map<String, Value>>, key: &Map<String, Value>| {
            let key_text = serde_json::to_string(key).unwrap_or_default();
            let id = (name.to_string(), key_text);
            if !state.contains_key(&id) {
                order.push(id.clone());
            }
            state.insert(id, row.cloned());
        };
        for line in log.current() {
            match super::log::op(line) {
                "snapshot" => {
                    for item in line.get("rows").and_then(Value::as_array).cloned().unwrap_or_default() {
                        let (Some(name), Some(row)) = (item.get("table").and_then(Value::as_str), item.get("row").and_then(Value::as_object)) else { continue };
                        let key = key_of(kind, name, row);
                        apply(name, Some(row), &key);
                    }
                }
                _ => {
                    let Some(name) = line.get("table").and_then(Value::as_str) else { continue };
                    if let Some(row) = line.get("row").and_then(Value::as_object) {
                        let key = key_of(kind, name, row);
                        apply(name, Some(row), &key);
                    } else if let Some(key) = line.get("key").and_then(Value::as_object) {
                        let key = key.clone();
                        apply(name, None, &key);
                    }
                }
            }
        }
        for store in [Store::Board, Store::Flow, Store::Audit] {
            let Some(path) = super::store_path(store) else { continue };
            if !path.exists() {
                continue;
            }
            let mut db = crate::db::open(&path).map_err(|e| e.to_string())?;
            let _ = db.execute_batch("PRAGMA foreign_keys=ON;");
            let tx = db.transaction().map_err(|e| e.to_string())?;
            // Parents before children: the tables' own order.
            for (s, name, owner_column, k, _) in TABLES {
                if *s != store || *k != kind {
                    continue;
                }
                if !has_table(&tx, name) {
                    continue;
                }
                let Some(cols) = columns_of(&tx, &path, name) else { continue };
                for id in order.iter().filter(|(t, _)| t == name) {
                    let Some(Some(row)) = state.get(id) else { continue };
                    // Always this file's own board or audit.
                    if row.get(*owner_column).and_then(Value::as_str) != Some(owner) {
                        continue;
                    }
                    let present: Vec<&String> = cols.names.iter().filter(|c| row.contains_key(c.as_str())).collect();
                    let names: Vec<&str> = present.iter().map(|c| c.as_str()).collect();
                    let marks: Vec<String> = (1..=names.len()).map(|i| format!("?{i}")).collect();
                    let update: Vec<String> = names.iter().filter(|c| !cols.key.iter().any(|k| k == *c)).map(|c| format!("{c}=excluded.{c}")).collect();
                    // Its own rows were just removed, so a key still held
                    // is another's: that row stays as it is.
                    let conflict = match (cols.key.is_empty(), update.is_empty()) {
                        (true, _) => String::new(),
                        (false, true) => format!(" ON CONFLICT({}) DO NOTHING", cols.key.join(",")),
                        (false, false) => {
                            format!(" ON CONFLICT({}) DO UPDATE SET {} WHERE {name}.{owner_column}=excluded.{owner_column}", cols.key.join(","), update.join(","))
                        }
                    };
                    let sql = format!("INSERT INTO {name}({}) VALUES({}){conflict}", names.join(","), marks.join(","));
                    let values: Vec<SqlValue> = names.iter().map(|c| sql_of(&row[*c])).collect();
                    if tx.execute(&sql, params_from_iter(values)).map_err(|e| format!("{name}: {e}"))? == 0 {
                        left_out += 1;
                    }
                }
            }
            tx.commit().map_err(|e| e.to_string())?;
        }
        Ok(left_out)
    })();
    QUIET.with(|q| q.set(false));
    result
}

fn has_table(db: &Connection, name: &str) -> bool {
    db.query_row("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1", [name], |_| Ok(())).is_ok()
}

fn key_of(kind: Kind, name: &str, row: &Map<String, Value>) -> Map<String, Value> {
    let keys: Vec<String> = TABLES
        .iter()
        .find(|t| t.3 == kind && t.1 == name)
        .and_then(|(store, ..)| super::store_path(*store))
        .and_then(|path| crate::db::open(&path).ok().and_then(|db| columns_of(&db, &path, name)))
        .map(|cols| cols.key)
        .unwrap_or_default();
    if keys.is_empty() {
        return row.clone();
    }
    keys.iter().map(|k| (k.clone(), row.get(k).cloned().unwrap_or(Value::Null))).collect()
}
