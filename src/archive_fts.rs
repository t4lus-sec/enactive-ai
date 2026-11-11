//! BM25 keyword index over the archive (SQLite FTS5). Derived and rebuildable:
//! the JSONL archive stays authoritative, and deleting `*.fts.sqlite3` only costs
//! a rebuild.
//!
//! Unlike the semantic index this needs no model server, so keyword search keeps
//! working when embeddings are down. It is synced synchronously before each query
//! (a no-op when the archive hasn't changed) and warmed in the background at
//! startup, so a search never ranks against a stale view.
//!
//! Tokenization is unicode61: case-folded, split on punctuation. An identifier like
//! `10.10.11.4` or `ticket-42` becomes a token sequence that a quoted phrase query
//! matches in order. Matching is per token, not per substring: `pass` does not
//! match `password`.
use super::semantic::digest;
use crate::hippocampus::ReadMark;
use rusqlite::{params, Connection, OptionalExtension};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, SystemTime},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Bump to force a rebuild when the table layout, tokenizer or kind rules change.
/// v2 adds each event's time (Unix seconds) so searches can take a time window.
const SCHEMA: &str = "s0ma-fts-v2";

pub(super) struct Match {
    pub id: u64,
}

pub(super) struct KeywordSearch {
    /// Best first by BM25.
    pub hits: Vec<Match>,
    /// Every event that matched (within the allowed kinds), before `limit`.
    pub total_matches: usize,
}

#[derive(Deserialize)]
struct SourceEvent {
    id: u64,
    #[serde(default)]
    timestamp: String,
    kind: String,
    content: String,
}

type Stamp = (u64, SystemTime);
/// What this process last synced: the archive's stamp then, and how far the
/// sync read (see `ReadMark`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Synced {
    stamp: Stamp,
    mark: ReadMark,
}
/// One lock per database: syncs and queries on the same index never interleave.
static SYNCED: OnceLock<Mutex<HashMap<PathBuf, IndexLock>>> = OnceLock::new();

/// One database's lock, and what it last synced.
type IndexLock = Arc<Mutex<Option<Synced>>>;

fn db_path(source: &Path) -> PathBuf {
    source.with_extension("fts.sqlite3")
}

/// Forget where the last sync stopped, so the next one reads the whole archive.
///
/// The fast path syncs only what was appended past the mark, and it is chosen
/// whenever the file merely grew from where the mark left off. A redaction
/// rewrites lines in place: the mark's own checks catch that (the rewrite is a
/// rename, so the file is a different one), but only on the next call, and only
/// if nothing re-stamps the mark first. Dropping it is the cheap certainty.
fn forget_sync_mark(source: &Path) {
    if let Ok(all) = SYNCED.get_or_init(Default::default).lock()
        && let Some(entry) = all.get(&db_path(source))
            && let Ok(mut synced) = entry.lock() {
                *synced = None;
            }
}

fn lock_for(db: &Path) -> Result<Arc<Mutex<Option<Synced>>>> {
    let mut all = SYNCED
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "keyword index lock poisoned")?;
    Ok(all.entry(db.to_owned()).or_default().clone())
}

fn connect(path: &Path) -> Result<Connection> {
    let db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(5))?;
    Ok(db)
}

/// Remove these events from the index, whatever the archive says about them.
///
/// Two statements each: the postings live in a contentless fts5 table, which
/// `contentless_delete=1` makes deletable, and the row that records the event's
/// digest is what would otherwise make the next scan think it is still indexed
/// and up to date. The file is written out again afterwards so the freed pages
/// go with the rows, since the text of a removed event is the thing being got
/// rid of.
pub(super) fn forget(source: &Path, ids: &[u64]) -> Result<usize> {
    let path = db_path(source);
    if !path.exists() || ids.is_empty() {
        return Ok(0);
    }
    let mut db = connect(&path)?;
    let mut removed = 0;
    {
        let tx = db.transaction()?;
        for id in ids {
            let id = i64::try_from(*id).unwrap_or(i64::MAX);
            tx.execute("DELETE FROM event_text WHERE rowid=?1", [id])?;
            removed += tx.execute("DELETE FROM events WHERE id=?1", [id])?;
        }
        tx.commit()?;
    }
    // Outside the transaction: VACUUM cannot run inside one. Neither this nor
    // the checkpoint can fail the removal -- the rows are already gone, and
    // reporting a failure here would say nothing was removed when it was.
    crate::hippocampus::compact(&db);
    // The next sync must read the whole archive rather than appending from
    // where it stopped, or it will not re-index the lines that replaced these.
    forget_sync_mark(source);
    Ok(removed)
}

fn initialize(db: &Connection) -> Result<()> {
    db.execute_batch(
        "PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    )?;
    let schema: Option<String> = db
        .query_row("SELECT value FROM metadata WHERE key='schema'", [], |r| r.get(0))
        .optional()?;
    if schema.as_deref() != Some(SCHEMA) {
        db.execute_batch(
            "DROP TABLE IF EXISTS events;
            DROP TABLE IF EXISTS event_text;
            DELETE FROM metadata;",
        )?;
    }
    // Contentless: the archive already holds the text, so the index stores only
    // postings. contentless_delete lets edited or removed events be dropped.
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, digest TEXT NOT NULL,
            epoch INTEGER NOT NULL, kind TEXT NOT NULL, at INTEGER);
        CREATE INDEX IF NOT EXISTS events_kind ON events(kind);
        CREATE INDEX IF NOT EXISTS events_at ON events(at);
        CREATE VIRTUAL TABLE IF NOT EXISTS event_text USING fts5(
            content, content='', contentless_delete=1,
            tokenize='unicode61 remove_diacritics 2');",
    )?;
    db.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('schema',?1)",
        [SCHEMA],
    )?;
    Ok(())
}

/// Bring the index in line with the whole archive. Unchanged events cost one
/// digest and one lookup (nothing is written for them); changed ones are
/// replaced; events no longer in the archive are pruned, but only after a
/// complete scan. Returns how far it read: just past the last complete line.
fn sync(db: &mut Connection, source: &Path) -> Result<ReadMark> {
    let file = File::open(source)?;
    let meta = file.metadata()?;
    let mut mark = ReadMark::start(&meta);
    let mut seen = HashSet::new();
    let mut reader = BufReader::new(file.take(meta.len()));
    let mut line = String::new();
    let tx = db.transaction()?;
    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            break;
        }
        // A writer may be between serde_json::to_writer and writeln; the next
        // query will pick the event up once the line is complete.
        if !line.ends_with('\n') {
            break;
        }
        let parsed = parse_line(&line)?;
        mark.advance(read, parsed.as_ref().map(|(_, _, event)| event.id));
        let Some((id, speaker, event)) = parsed else {
            continue;
        };
        if !seen.insert(id) {
            return Err(format!("duplicate archive event ID {}", event.id).into());
        }
        index_event(&tx, id, speaker, &event)?;
    }
    let gone: Vec<i64> = tx
        .prepare("SELECT id FROM events")?
        .query_map([], |r| r.get(0))?
        .collect::<std::result::Result<Vec<i64>, _>>()?
        .into_iter()
        .filter(|id| !seen.contains(id))
        .collect();
    for id in gone {
        tx.execute("DELETE FROM event_text WHERE rowid=?1", [id])?;
        tx.execute("DELETE FROM events WHERE id=?1", [id])?;
    }
    tx.commit()?;
    Ok(mark)
}

/// Index only the lines appended after `from`, where the last sync stopped.
/// None when one of them carries an id already indexed: something other than
/// an append happened, and the caller syncs the whole archive.
fn sync_appended(db: &mut Connection, source: &Path, from: ReadMark) -> Result<Option<ReadMark>> {
    use std::io::{Seek, SeekFrom};
    let mut file = File::open(source)?;
    let source_len = file.metadata()?.len();
    file.seek(SeekFrom::Start(from.through))?;
    let mut reader = BufReader::new(file.take(source_len.saturating_sub(from.through)));
    let mut line = String::new();
    let mut mark = from;
    let tx = db.transaction()?;
    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 || !line.ends_with('\n') {
            break;
        }
        let parsed = parse_line(&line)?;
        mark.advance(read, parsed.as_ref().map(|(_, _, event)| event.id));
        let Some((id, speaker, event)) = parsed else {
            continue;
        };
        let known: Option<i64> = tx.query_row("SELECT id FROM events WHERE id=?1", [id], |r| r.get(0)).optional()?;
        if known.is_some() {
            return Ok(None);
        }
        index_event(&tx, id, speaker, &event)?;
    }
    tx.commit()?;
    Ok(Some(mark))
}

/// An archive line as the index needs it; None for a blank or unreadable
/// line, or one of a kind the index doesn't hold.
fn parse_line(line: &str) -> Result<Option<(i64, &'static str, SourceEvent)>> {
    if line.trim().is_empty() {
        return Ok(None);
    }
    let Ok(event) = serde_json::from_str::<SourceEvent>(line) else {
        return Ok(None);
    };
    let speaker = match event.kind.as_str() {
        "operator" => "operator",
        "s0ma" => "s0ma",
        _ => return Ok(None),
    };
    let id = i64::try_from(event.id).map_err(|_| "event ID exceeds SQLite integer range")?;
    Ok(Some((id, speaker, event)))
}

/// Index one event, replacing an older version of it; an unchanged one is
/// left as it is.
fn index_event(tx: &rusqlite::Transaction, id: i64, speaker: &str, event: &SourceEvent) -> Result<()> {
    let hash = digest(&event.content);
    let old: Option<String> = tx
        .query_row("SELECT digest FROM events WHERE id=?1", [id], |r| r.get(0))
        .optional()?;
    if old.as_deref() == Some(hash.as_str()) {
        return Ok(());
    }
    if old.is_some() {
        tx.execute("DELETE FROM event_text WHERE rowid=?1", [id])?;
    }
    let at = super::rank::event_time(&event.timestamp);
    // `epoch` is no longer used (pruning works from the scan's own set); the
    // column stays so existing indexes keep their layout.
    tx.execute(
        "INSERT OR REPLACE INTO events VALUES (?1,?2,0,?3,?4)",
        params![id, hash, super::source_kind(speaker, &event.content), at],
    )?;
    tx.execute(
        "INSERT INTO event_text(rowid,content) VALUES (?1,?2)",
        params![id, event.content],
    )?;
    Ok(())
}

/// Open the index, syncing first if the archive changed since the last sync.
/// The first sync in a process reads the whole archive (so an edit made while
/// s0ma was stopped is picked up); after that, an archive that only grew is
/// synced from where the last sync ended, reading just the new lines. Anything
/// else (a replaced, shortened or rewritten file) gets a whole sync again.
fn ready(source: &Path) -> Result<(Connection, Arc<Mutex<Option<Synced>>>)> {
    let path = db_path(source);
    let lock = lock_for(&path)?;
    let mut db = connect(&path)?;
    {
        let mut last = lock.lock().map_err(|_| "keyword index lock poisoned")?;
        match std::fs::metadata(source) {
            Err(_) => initialize(&db)?,
            Ok(meta) => {
                let now: Stamp = (meta.len(), meta.modified().unwrap_or(SystemTime::UNIX_EPOCH));
                if last.map(|synced| synced.stamp) != Some(now) {
                    let appended = match *last {
                        Some(previous) if now.0 > previous.mark.through && previous.mark.still_holds(source, &meta)? => {
                            sync_appended(&mut db, source, previous.mark)?
                        }
                        _ => None,
                    };
                    let mark = match appended {
                        Some(mark) => mark,
                        None => {
                            initialize(&db)?;
                            sync(&mut db, source)?
                        }
                    };
                    *last = Some(Synced { stamp: now, mark });
                }
            }
        }
    }
    Ok((db, lock))
}

/// Warm the index in the background so the first search after startup is fast.
pub(super) fn start(source: &Path) {
    let source = source.to_owned();
    let _ = std::thread::Builder::new()
        .name("archive-fts".into())
        .spawn(move || {
            if let Err(error) = ready(&source) {
                crate::terminal::warn_status(&format!("[archive] keyword index warm-up failed: {error}"));
            }
        });
}

/// FTS5 query text for the operator's terms. Each term becomes a quoted phrase, so
/// no user text is ever parsed as FTS5 syntax. Terms with no letters or digits
/// can't match any token and are dropped.
pub(super) fn match_expression(query: &str, require_all: bool) -> Option<String> {
    let phrases: Vec<String> = crate::hippocampus::literal_terms(query)
        .into_iter()
        .filter(|term| term.chars().any(char::is_alphanumeric))
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect();
    if phrases.is_empty() {
        return None;
    }
    Some(phrases.join(if require_all { " AND " } else { " OR " }))
}

/// Best-first BM25 search. `require_all` gives literal-mode semantics (every term
/// must appear); otherwise any term may match and BM25 favours events with more
/// and rarer terms, which is what hybrid fusion wants.
pub(super) fn search(
    source: &Path,
    query: &str,
    kinds: &[&str],
    require_all: bool,
    limit: usize,
    window: Option<super::rank::TimeWindow>,
    range: Option<super::rank::EventRange>,
) -> Result<KeywordSearch> {
    let Some(expression) = match_expression(query, require_all) else {
        return Ok(KeywordSearch {
            hits: Vec::new(),
            total_matches: 0,
        });
    };
    let (db, lock) = ready(source)?;
    let _guard = lock.lock().map_err(|_| "keyword index lock poisoned")?;
    let kind_list = serde_json::to_string(kinds)?;
    // An open end is NULL; an event with no readable time never matches a window.
    let since = window.and_then(|w| w.since);
    let until = window.and_then(|w| w.until);
    let windowed = window.is_some() as i64;
    // Both ends inclusive; an open end is NULL.
    let (from, through) = range.map(|r| r.sql_bounds()).unwrap_or_default();
    let total: i64 = db.query_row(
        "SELECT count(*) FROM event_text f JOIN events e ON e.id=f.rowid
         WHERE event_text MATCH ?1 AND e.kind IN (SELECT value FROM json_each(?2))
           AND (?3=0 OR (e.at IS NOT NULL AND (?4 IS NULL OR e.at>=?4) AND (?5 IS NULL OR e.at<?5)))
           AND (?6 IS NULL OR e.id>=?6) AND (?7 IS NULL OR e.id<=?7)",
        params![expression, kind_list, windowed, since, until, from, through],
        |r| r.get(0),
    )?;
    let hits = db
        .prepare(
            "SELECT f.rowid FROM event_text f JOIN events e ON e.id=f.rowid
             WHERE event_text MATCH ?1 AND e.kind IN (SELECT value FROM json_each(?2))
               AND (?4=0 OR (e.at IS NOT NULL AND (?5 IS NULL OR e.at>=?5) AND (?6 IS NULL OR e.at<?6)))
               AND (?7 IS NULL OR e.id>=?7) AND (?8 IS NULL OR e.id<=?8)
             ORDER BY bm25(event_text), f.rowid DESC LIMIT ?3",
        )?
        .query_map(params![expression, kind_list, limit as i64, windowed, since, until, from, through], |r| {
            r.get::<_, i64>(0)
        })?
        .map(|id| Ok(Match { id: u64::try_from(id?)? }))
        .collect::<Result<Vec<_>>>()?;
    Ok(KeywordSearch {
        hits,
        total_matches: usize::try_from(total)?,
    })
}

/// IDs of every indexed event inside the scope: recorded inside `window` (an
/// event with no readable time never is) and with an id in `range`. An absent
/// bound doesn't restrict.
pub(super) fn ids_in_scope(
    source: &Path,
    window: Option<super::rank::TimeWindow>,
    range: Option<super::rank::EventRange>,
) -> Result<HashSet<u64>> {
    let (db, lock) = ready(source)?;
    let _guard = lock.lock().map_err(|_| "keyword index lock poisoned")?;
    let (from, through) = range.map(|r| r.sql_bounds()).unwrap_or_default();
    let ids = db
        .prepare(
            "SELECT id FROM events
             WHERE (?1=0 OR (at IS NOT NULL AND (?2 IS NULL OR at>=?2) AND (?3 IS NULL OR at<?3)))
               AND (?4 IS NULL OR id>=?4) AND (?5 IS NULL OR id<=?5)",
        )?
        .query_map(
            params![window.is_some() as i64, window.and_then(|w| w.since), window.and_then(|w| w.until), from, through],
            |r| r.get::<_, i64>(0),
        )?
        .map(|id| Ok(u64::try_from(id?)?))
        .collect::<Result<HashSet<_>>>()?;
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "s0ma_fts_{}_{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn source(&self) -> PathBuf {
            self.0.join("hippocampus.jsonl")
        }
        fn write(&self, events: &[(u64, &str, &str)]) {
            let mut file = File::create(self.source()).unwrap();
            for (id, kind, content) in events {
                writeln!(
                    file,
                    "{}",
                    serde_json::json!({"id":id,"timestamp":"2026-09-28T00:00:00Z","kind":kind,"content":content})
                )
                .unwrap();
            }
            // Different lengths are not guaranteed between rewrites in a test;
            // clear the cached stamp so every rewrite is re-synced.
            *lock_for(&db_path(&self.source())).unwrap().lock().unwrap() = None;
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    const ALL: &[&str] = &[
        "operator",
        "s0ma_reply",
        "tool_result",
        "reasoning",
        "subagent_summary",
        "environment",
        "runtime",
    ];
    fn ids(result: &KeywordSearch) -> Vec<u64> {
        result.hits.iter().map(|m| m.id).collect()
    }

    /// A job's close has to make its text unfindable, and the rows have to go
    /// at once rather than when some later scan gets round to it: keyword
    /// search has no check at query time that a hit is still in the archive --
    /// unlike the semantic side, which re-reads and compares digests -- and the
    /// bytes cannot be vacuumed out of the file while the rows are still there.
    #[test]
    fn forgetting_an_event_takes_its_words_out_of_the_index() {
        let f = Fixture::new();
        f.write(&[
            (1, "operator", "the quarterly figures for northwind"),
            (2, "s0ma", "northwind margins were thin"),
            (3, "operator", "unrelated weather talk"),
        ]);
        let found = search(&f.source(), "northwind", ALL, true, 10, None, None).unwrap();
        assert_eq!(ids(&found), vec![2, 1], "both events mention it");

        // In the order a close does it: the archive is redacted first, then
        // the index is told. The other order would be pointless -- a sync
        // reads the archive, so it would put back whatever was still there.
        f.write(&[
            (1, "operator", "[redacted for job blue-7]"),
            (2, "s0ma", "[redacted for job blue-7]"),
            (3, "operator", "unrelated weather talk"),
        ]);
        let removed = forget(&f.source(), &[1, 2]).unwrap();
        assert_eq!(removed, 2);

        // Immediately, before any search has had the chance to re-sync: the
        // rows are out of both tables.
        let db = connect(&db_path(&f.source())).unwrap();
        let rows: i64 = db.query_row("SELECT COUNT(*) FROM events WHERE id IN (1,2)", [], |r| r.get(0)).unwrap();
        assert_eq!(rows, 0, "the digest rows went, or a sync would think them current");
        let postings: i64 = db
            .query_row("SELECT COUNT(*) FROM event_text WHERE rowid IN (1,2)", [], |r| r.get(0))
            .unwrap();
        assert_eq!(postings, 0, "and the postings with them");
        drop(db);

        let after = search(&f.source(), "northwind", ALL, true, 10, None, None).unwrap();
        assert!(ids(&after).is_empty(), "the word is gone: {:?}", ids(&after));
        let other = search(&f.source(), "weather", ALL, true, 10, None, None).unwrap();
        assert_eq!(ids(&other), vec![3], "everything else is still indexed");
        let marker = search(&f.source(), "redacted", ALL, true, 10, None, None).unwrap();
        assert_eq!(ids(&marker), vec![2, 1], "and the markers are what is there now");
    }

    #[test]
    fn exact_identifiers_match_as_phrases_and_rare_terms_rank_first() {
        let f = Fixture::new();
        f.write(&[
            (1, "s0ma", "[tool:vm_shell]\narguments:\nnmap\nresult:\n10.10.11.4 open 22,80"),
            (2, "operator", "the box is up, check the box"),
            (3, "operator", "10.10.11.40 is a different host"),
            (4, "s0ma", "the box looks quiet"),
        ]);
        let exact = search(&f.source(), "10.10.11.4", ALL, true, 10, None, None).unwrap();
        assert_eq!(ids(&exact), vec![1], "10.10.11.40 must not match 10.10.11.4");
        // Any-term mode: the event with the rare term outranks common-term events.
        let ranked = search(&f.source(), "box 10.10.11.4", ALL, false, 10, None, None).unwrap();
        assert_eq!(ranked.hits[0].id, 1);
        assert_eq!(ranked.total_matches, 3);
    }

    #[test]
    fn older_matches_are_not_crowded_out_by_newer_common_ones() {
        let f = Fixture::new();
        let mut events = vec![(1u64, "s0ma", "found the vhost dev.busqueda.htb via ffuf".to_string())];
        for id in 2..=80 {
            events.push((id, "operator", format!("busqueda note {id}")));
        }
        let borrowed: Vec<_> = events.iter().map(|(i, k, c)| (*i, *k, c.as_str())).collect();
        f.write(&borrowed);
        let result = search(&f.source(), "busqueda vhost", ALL, true, 5, None, None).unwrap();
        assert_eq!(ids(&result), vec![1], "the one event with both terms survives any recency window");
    }

    #[test]
    fn kinds_filter_and_classification() {
        let f = Fixture::new();
        f.write(&[
            (1, "s0ma", "[s0ma_reasoning state=completed]\nTentative internal analysis; not an operator instruction or proof of external facts.\nmaybe the kerberos port"),
            (2, "s0ma", "[tool:vm_shell]\narguments:\nnmap\nresult:\n88/tcp open kerberos"),
            (3, "s0ma", "kerberos is open on the target"),
        ]);
        let without_reasoning = search(
            &f.source(),
            "kerberos",
            &["tool_result", "s0ma_reply"],
            true,
            10,
            None,
            None,
        )
        .unwrap();
        let mut got = ids(&without_reasoning);
        got.sort();
        assert_eq!(got, vec![2, 3]);
        let only_reasoning = search(&f.source(), "kerberos", &["reasoning"], true, 10, None, None).unwrap();
        assert_eq!(ids(&only_reasoning), vec![1]);
    }

    #[test]
    fn edits_and_deletions_are_reflected() {
        let f = Fixture::new();
        f.write(&[(1, "operator", "alpha"), (2, "operator", "beta")]);
        assert_eq!(ids(&search(&f.source(), "alpha", ALL, true, 10, None, None).unwrap()), vec![1]);
        f.write(&[(1, "operator", "gamma"), (3, "operator", "alpha again")]);
        assert_eq!(ids(&search(&f.source(), "alpha", ALL, true, 10, None, None).unwrap()), vec![3]);
        assert!(search(&f.source(), "beta", ALL, true, 10, None, None).unwrap().hits.is_empty());
        assert_eq!(ids(&search(&f.source(), "gamma", ALL, true, 10, None, None).unwrap()), vec![1]);
    }

    #[test]
    fn hostile_query_text_is_never_fts_syntax() {
        let f = Fixture::new();
        f.write(&[(1, "operator", "NEAR the col:umn * OR thing")]);
        for query in ["NEAR(", "col:umn", "a OR", "*", "\"unterminated", "^start", "-", "()"] {
            // Must not error; may or may not match.
            search(&f.source(), query, ALL, true, 10, None, None).unwrap();
            search(&f.source(), query, ALL, false, 10, None, None).unwrap();
        }
        assert_eq!(ids(&search(&f.source(), "col:umn", ALL, true, 10, None, None).unwrap()), vec![1]);
    }

    fn synced(f: &Fixture) -> Synced {
        lock_for(&db_path(&f.source())).unwrap().lock().unwrap().unwrap()
    }

    #[test]
    fn appended_lines_are_synced_without_rereading_the_archive() {
        let f = Fixture::new();
        f.write(&[(1, "operator", "alpha"), (2, "s0ma", "beta")]);
        assert_eq!(ids(&search(&f.source(), "alpha", ALL, true, 10, None, None).unwrap()), vec![1]);
        let first = synced(&f);
        let mut file = std::fs::OpenOptions::new().append(true).open(f.source()).unwrap();
        writeln!(file, "{}", serde_json::json!({"id":3,"timestamp":"2026-09-28T00:00:00Z","kind":"operator","content":"gamma alpha"})).unwrap();
        drop(file);
        let found = search(&f.source(), "alpha", ALL, true, 10, None, None).unwrap();
        let mut got = ids(&found);
        got.sort();
        assert_eq!(got, vec![1, 3]);
        let after = synced(&f);
        assert!(after.mark.through > first.mark.through);
        assert_eq!(after.mark.through, std::fs::metadata(f.source()).unwrap().len());
    }

    fn line(id: u64, kind: &str, content: &str) -> String {
        format!("{}\n", serde_json::json!({"id":id,"timestamp":"2026-09-28T00:00:00Z","kind":kind,"content":content}))
    }

    #[test]
    fn a_running_index_reads_only_appends_and_a_restart_reads_everything() {
        let f = Fixture::new();
        f.write(&[(1, "operator", "alpha"), (2, "s0ma", "beta")]);
        assert_eq!(ids(&search(&f.source(), "alpha", ALL, true, 10, None, None).unwrap()), vec![1]);
        // Event 1 edited in place at the same length, and event 3 appended. The
        // last synced line (2) is untouched, so only line 3 is read: the edit
        // waits for the next whole sync, which every process starts with.
        let text = [line(1, "operator", "omega"), line(2, "s0ma", "beta"), line(3, "operator", "gamma")].concat();
        std::fs::write(f.source(), text).unwrap();
        assert_eq!(ids(&search(&f.source(), "gamma", ALL, true, 10, None, None).unwrap()), vec![3]);
        assert_eq!(ids(&search(&f.source(), "alpha", ALL, true, 10, None, None).unwrap()), vec![1]);
        // A new process (no sync yet) reads the whole archive and sees the edit.
        *lock_for(&db_path(&f.source())).unwrap().lock().unwrap() = None;
        assert!(search(&f.source(), "alpha", ALL, true, 10, None, None).unwrap().hits.is_empty());
        assert_eq!(ids(&search(&f.source(), "omega", ALL, true, 10, None, None).unwrap()), vec![1]);
    }

    #[test]
    fn a_rewrite_that_moves_the_last_synced_line_is_synced_whole() {
        let f = Fixture::new();
        f.write(&[(1, "operator", "aa"), (2, "operator", "bb"), (3, "operator", "cc")]);
        assert_eq!(ids(&search(&f.source(), "bb", ALL, true, 10, None, None).unwrap()), vec![2]);
        // Every line the same length, so the old end is still a line boundary
        // and the line past it (4) is new; but event 5 now sits where 3 was.
        let text = [line(1, "operator", "aa"), line(3, "operator", "cc"), line(5, "operator", "ee"), line(4, "operator", "dd")].concat();
        std::fs::write(f.source(), text).unwrap();
        assert!(search(&f.source(), "bb", ALL, true, 10, None, None).unwrap().hits.is_empty());
        assert_eq!(ids(&search(&f.source(), "ee", ALL, true, 10, None, None).unwrap()), vec![5]);
        assert_eq!(ids(&search(&f.source(), "dd", ALL, true, 10, None, None).unwrap()), vec![4]);
    }

    #[test]
    fn a_rewrite_that_also_grows_the_archive_is_synced_whole() {
        let f = Fixture::new();
        f.write(&[(1, "operator", "alpha"), (2, "s0ma", "beta")]);
        assert_eq!(ids(&search(&f.source(), "beta", ALL, true, 10, None, None).unwrap()), vec![2]);
        // Rewritten in place (same file), longer than before, without the
        // fixture's stamp reset: the appended-only path must not be trusted.
        let text: String = [(1, "operator", "alpha rewritten at some length"), (2, "s0ma", "delta"), (3, "operator", "epsilon")]
            .iter()
            .map(|(id, kind, content)| format!("{}\n", serde_json::json!({"id":id,"timestamp":"2026-09-28T00:00:00Z","kind":kind,"content":content})))
            .collect();
        std::fs::write(f.source(), text).unwrap();
        assert!(search(&f.source(), "beta", ALL, true, 10, None, None).unwrap().hits.is_empty());
        assert_eq!(ids(&search(&f.source(), "delta", ALL, true, 10, None, None).unwrap()), vec![2]);
        assert_eq!(ids(&search(&f.source(), "rewritten", ALL, true, 10, None, None).unwrap()), vec![1]);
    }

    #[test]
    fn a_partial_final_line_is_left_for_the_next_sync() {
        let f = Fixture::new();
        f.write(&[(1, "operator", "complete")]);
        let mut file = std::fs::OpenOptions::new().append(true).open(f.source()).unwrap();
        write!(file, "{{\"id\":2,\"kind\":\"operator\",\"content\":\"half").unwrap();
        assert_eq!(ids(&search(&f.source(), "complete", ALL, true, 10, None, None).unwrap()), vec![1]);
    }

    #[test]
    fn time_window_filters_matches_and_lists_ids() {
        let f = Fixture::new();
        {
            let mut file = File::create(f.source()).unwrap();
            for (id, at, content) in [
                (1, "2026-09-01T10:00:00+00:00", "ssh brute force on busqueda"),
                (2, "2026-10-05T10:00:00+00:00", "ssh key found on busqueda"),
                (3, "not a time", "ssh config"),
            ] {
                writeln!(file, "{}", serde_json::json!({"id":id,"timestamp":at,"kind":"s0ma","content":content})).unwrap();
            }
        }
        let at = |t: &str| super::super::rank::event_time(t).unwrap();
        let window = super::super::rank::TimeWindow { since: Some(at("2026-10-01T00:00:00+00:00")), until: None };
        let found = search(&f.source(), "ssh", ALL, false, 10, Some(window), None).unwrap();
        assert_eq!(ids(&found), vec![2]);
        assert_eq!(found.total_matches, 1);
        assert_eq!(search(&f.source(), "ssh", ALL, false, 10, None, None).unwrap().total_matches, 3);
        assert_eq!(ids_in_scope(&f.source(), Some(window), None).unwrap(), [2u64].into());
    }

    #[test]
    fn an_event_range_is_inclusive_and_combines_with_a_window() {
        let f = Fixture::new();
        {
            let mut file = File::create(f.source()).unwrap();
            for (id, at, content) in [
                (1, "2026-09-01T10:00:00+00:00", "lease draft"),
                (2, "2026-09-02T10:00:00+00:00", "lease signed"),
                (3, "2026-10-05T10:00:00+00:00", "lease renewal"),
                (4, "not a time", "lease question"),
                (5, "2026-10-06T10:00:00+00:00", "lease ended"),
            ] {
                writeln!(file, "{}", serde_json::json!({"id":id,"timestamp":at,"kind":"operator","content":content})).unwrap();
            }
        }
        let range = |from, through| super::super::rank::EventRange::new(from, through).unwrap();
        let mut found = ids(&search(&f.source(), "lease", ALL, false, 10, None, range(Some(2), Some(4))).unwrap());
        found.sort();
        assert_eq!(found, vec![2, 3, 4], "both ends inclusive; no time needed without a window");
        let tail = search(&f.source(), "lease", ALL, false, 10, None, range(Some(4), None)).unwrap();
        assert_eq!((ids(&tail).len(), tail.total_matches), (2, 2));
        assert_eq!(ids_in_scope(&f.source(), None, range(None, Some(2))).unwrap(), [1u64, 2].into());
        // With a window as well, an event must satisfy both, and one with no
        // readable time (4) never matches a window.
        let at = |t: &str| super::super::rank::event_time(t).unwrap();
        let window = super::super::rank::TimeWindow { since: Some(at("2026-10-01T00:00:00+00:00")), until: None };
        assert_eq!(ids_in_scope(&f.source(), Some(window), range(Some(2), Some(4))).unwrap(), [3u64].into());
        assert_eq!(ids(&search(&f.source(), "lease", ALL, false, 10, Some(window), range(Some(2), Some(4))).unwrap()), vec![3]);
    }
}
