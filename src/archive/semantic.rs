//! Local, rebuildable derived index. The JSONL archive remains authoritative.
//!
//! A pass has two stages, and the split is what makes the order right.
//! **Reading** settles what each event needs — nothing, a place in the
//! `pending` queue, or a row in `skipped` — and touches no model server, so it
//! always finishes; with a mark from an earlier pass it reads only what was
//! appended. **Embedding** then drains the queue newest first. Before the
//! split, a pass embedded as it read, so a backlog was worked oldest first and
//! the events most likely to be searched for became findable last: on a
//! 68,000-event backlog that meant days. Now a rebuild makes the recent past
//! searchable within minutes and works backwards.
//!
//! Because of that ordering there is no longer a boundary event below which
//! everything is embedded, so coverage is a set: `settled` is what a search
//! can see, and `archive::assess_recall` subtracts it from the scope.
use reqwest::blocking::Client;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use crate::hippocampus::ReadMark;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant, SystemTime},
};

const CHUNK_BYTES: usize = 1800;
const OVERLAP_BYTES: usize = 256;
/// The most of a chunk's budget its header may take, so header plus text
/// always fits what the embedding server will accept.
const MAX_HEADER_BYTES: usize = 160;
/// However long a header is, a chunk keeps at least this much text.
const MIN_CHUNK_BYTES: usize = 1_200;
const MAX_DIMENSIONS: usize = 8192;
/// Text (≤ CHUNK_BYTES) plus prefix must fit the embedding server's 2048-token
/// context even at one token per byte.
const MAX_PREFIX_BYTES: usize = 100;
/// Nemotron-3-Embed-1B's own prompts (config_sentence_transformers.json):
/// "query: " on searches, "passage: " on everything stored.
const QUERY_PREFIX: &str = "query: ";
const DOCUMENT_PREFIX: &str = "passage: ";
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Debug)]
struct Config {
    url: String,
    model: String,
    query_prefix: String,
    document_prefix: String,
    /// Query side of episode recall, for models whose queries carry a task
    /// instruction; the same "query: " for Nemotron. Not in the fingerprint: it
    /// never touches a stored vector, so changing it needs no rebuild.
    episode_query_prefix: String,
    revision: String,
}
impl Config {
    fn load() -> Result<Self> {
        // Off in the overview (or S0MA_EMBEDDINGS=off): nothing is embedded,
        // and every user of an embedder says so and does without.
        if !crate::model_config::embeddings_on() {
            return Err(crate::model_config::EMBEDDINGS_OFF.into());
        }
        fn env(key: &str, default: &str) -> String {
            std::env::var(key).unwrap_or_else(|_| default.into())
        }
        let config = Self {
            url: env(
                "S0MA_ARCHIVE_EMBEDDING_URL",
                "http://127.0.0.1:8081/v1/embeddings",
            ),
            model: env("S0MA_ARCHIVE_EMBEDDING_MODEL", "nemotron-3-embed-1b"),
            query_prefix: env("S0MA_ARCHIVE_QUERY_PREFIX", QUERY_PREFIX),
            document_prefix: env("S0MA_ARCHIVE_DOCUMENT_PREFIX", DOCUMENT_PREFIX),
            episode_query_prefix: env("S0MA_EPISODE_QUERY_PREFIX", QUERY_PREFIX),
            // The same default as start-s0ma.sh ("2 = rebuilt for
            // Nemotron-3-Embed-1B"). A start that skips its run_s0ma (after
            // --servers-only, or the binary from any shell) used to get "1" here,
            // and each switch between the two emptied the whole index.
            revision: env("S0MA_ARCHIVE_EMBEDDING_REVISION", "2"),
        };
        let url = reqwest::Url::parse(&config.url)?;
        if !matches!(url.scheme(), "http" | "https")
            || !matches!(
                url.host_str(),
                Some("127.0.0.1" | "localhost" | "[::1]" | "::1")
            )
        {
            return Err("archive embeddings must use a loopback endpoint".into());
        }
        if [&config.query_prefix, &config.document_prefix, &config.episode_query_prefix]
            .iter()
            .any(|prefix| prefix.len() > MAX_PREFIX_BYTES)
        {
            return Err(format!("embedding prefixes must be at most {MAX_PREFIX_BYTES} bytes").into());
        }
        Ok(config)
    }
    fn fingerprint(&self) -> String {
        digest(&format!(
            "s0ma-semantic-v2-headers-i16|{CHUNK_BYTES}|{OVERLAP_BYTES}|{MAX_HEADER_BYTES}|{}",
            serde_json::json!([
                &self.url,
                &self.model,
                &self.query_prefix,
                &self.document_prefix,
                &self.revision
            ])
        ))
    }
}

/// The settings the fingerprint covers, as stored beside it.
const SETTINGS: [(&str, &str); 5] = [
    ("url", "S0MA_ARCHIVE_EMBEDDING_URL"),
    ("model", "S0MA_ARCHIVE_EMBEDDING_MODEL"),
    ("query_prefix", "S0MA_ARCHIVE_QUERY_PREFIX"),
    ("document_prefix", "S0MA_ARCHIVE_DOCUMENT_PREFIX"),
    ("revision", "S0MA_ARCHIVE_EMBEDDING_REVISION"),
];
impl Config {
    fn settings(&self) -> String {
        serde_json::json!({
            "url": self.url, "model": self.model, "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix, "revision": self.revision,
        })
        .to_string()
    }
}

/// Said when a changed setting has just emptied the index. Every setting in
/// the fingerprint counts, the URL's spelling included, and the harness takes
/// them from its environment: starting it from another shell or launcher is
/// enough to cause a full re-embedding, which on a CPU embedder takes hours.
fn rebuild_notice(before: Option<&str>, config: &Config, dropped: i64) -> String {
    let now: serde_json::Value = serde_json::from_str(&config.settings()).unwrap_or_default();
    let before: Option<serde_json::Value> = before.and_then(|text| serde_json::from_str(text).ok());
    let changes: Vec<String> = SETTINGS
        .iter()
        .filter_map(|(key, variable)| {
            let then = before.as_ref()?.get(*key)?;
            (then != &now[*key]).then(|| format!("{variable}: {then} → {}", now[*key]))
        })
        .collect();
    let what = if changes.is_empty() {
        format!(
            "they are now {}",
            SETTINGS.iter().map(|(key, variable)| format!("{variable}={}", now[*key])).collect::<Vec<_>>().join(", ")
        )
    } else {
        changes.join("; ")
    };
    format!(
        "[archive] the embedding settings differ from the ones the semantic index was built with ({what}), \
         so it was emptied and {dropped} event(s) will be embedded again in the background, which takes hours \
         on a CPU embedder. The harness reads these from the environment it starts in: set them the same way \
         every time (in the launcher, not one shell), since each change empties it again."
    )
}

/// The longest background embedding waits for the model at one stretch. A
/// model that never pauses (loop mode, a row of workers, a sleep) would
/// otherwise stall the index for good; after this, one piece is embedded and
/// the wait starts again, so the index still moves, slowly. One piece is tens
/// of milliseconds on a GPU, so a minute of decoding hardly notices it.
const GENERATION_YIELD_MAX: Duration = Duration::from_secs(60);
/// How often a waiting embedder looks again.
const GENERATION_YIELD_POLL: Duration = Duration::from_millis(100);

/// For background embedding only (the archive index, episode indexing after a
/// sleep): wait while the model is generating, so the embedder doesn't take GPU
/// or CPU time from prefill and decoding. Embeddings a turn is waiting on (the
/// skill match, archive_search, episode recall) don't call this.
/// `S0MA_EMBED_YIELD=0` turns it off. Returns how long it waited.
pub(crate) fn yield_to_generation() -> Duration {
    if !yield_enabled() {
        return Duration::ZERO;
    }
    wait_while(crate::llm::generating, GENERATION_YIELD_MAX, GENERATION_YIELD_POLL)
}

fn yield_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("S0MA_EMBED_YIELD").map(|value| value.trim().to_ascii_lowercase()).as_deref(),
            Ok("0" | "off" | "false" | "no")
        )
    })
}

/// Wait while `busy`, looking every `poll`, for at most `longest`.
fn wait_while(busy: impl Fn() -> bool, longest: Duration, poll: Duration) -> Duration {
    let started = Instant::now();
    while busy() {
        let waited = started.elapsed();
        if waited >= longest {
            break;
        }
        std::thread::sleep(poll.min(longest - waited));
    }
    started.elapsed()
}

pub(super) fn digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct IndexStatus {
    pub state: String,
    pub indexed_events: usize,
    /// Events read and queued for embedding but not embedded yet. The queue is
    /// drained newest first, so a backlog is missing its oldest events.
    pub pending_events: usize,
    /// Events this index holds no vectors for on purpose (see `skip_reason`).
    pub skipped_events: usize,
    pub source_bytes: u64,
    pub indexed_source_bytes: u64,
    pub last_error: Option<String>,
    /// The highest event id with vectors. Not a boundary: the queue is drained
    /// newest first, so events below this one may still be waiting. What a
    /// search can and cannot see comes from `settled`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newest_indexed_event: Option<u64>,
    /// Text the embedding server refused even in small pieces, since the
    /// harness started: skipped, so the index could go on past it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_refusal: Option<String>,
}

/// How far the index reaches, reported as a pass advances.
#[derive(Debug, Clone, Default)]
pub(super) struct Coverage {
    pub indexed_events: usize,
    pub skipped_events: usize,
    pub pending_events: usize,
    pub newest_indexed: Option<u64>,
    /// Pieces of this event the server refused and the index skipped.
    pub refused: Option<String>,
}

/// Which events a search can see: embedded, or settled as never embedded, and
/// not waiting to be embedded again. A stale row whose re-embedding is queued
/// is not among them, because `validate_candidates` would reject its hits.
pub(super) fn settled(source: &Path) -> Result<HashSet<u64>> {
    let engine = engine(source)?;
    let db = connect(&engine.db)?;
    let mut ids: HashSet<u64> = db
        .prepare("SELECT id FROM events UNION SELECT id FROM skipped")?
        .query_map([], |r| r.get::<_, i64>(0))?
        .collect::<std::result::Result<Vec<i64>, _>>()?
        .into_iter()
        .filter_map(|id| u64::try_from(id).ok())
        .collect();
    for queued in db
        .prepare("SELECT id FROM pending")?
        .query_map([], |r| r.get::<_, i64>(0))?
        .collect::<std::result::Result<Vec<i64>, _>>()?
    {
        if let Ok(id) = u64::try_from(queued) {
            ids.remove(&id);
        }
    }
    Ok(ids)
}

/// The highest event id with vectors (see `IndexStatus::newest_indexed_event`).
fn newest_indexed(db: &Connection) -> Result<Option<u64>> {
    let newest: Option<i64> = db.query_row("SELECT max(id) FROM events", [], |r| r.get(0))?;
    Ok(newest.and_then(|id| u64::try_from(id).ok()))
}
struct Engine {
    db: PathBuf,
    config: Config,
    status: Mutex<IndexStatus>,
}
static ENGINES: OnceLock<Mutex<HashMap<PathBuf, Arc<Engine>>>> = OnceLock::new();

fn connect(path: &Path) -> Result<Connection> {
    let db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(5))?;
    db.execute_batch("PRAGMA foreign_keys=ON;")?;
    Ok(db)
}
/// Remove these events from the index, whatever the archive says about them.
///
/// One statement each: `chunks` and `chunk_vectors` reference `events` with
/// `ON DELETE CASCADE`, and `PRAGMA foreign_keys=ON` is set on every connection
/// here, so deleting the event takes its text and its vectors with it. `pending`
/// and `skipped` are cleared too, or a later pass would embed what it finds at
/// those ids and put the index back out of step with what it says it holds.
///
/// `consolidated_episodes` shares this file and is **not** touched: it is the
/// durable episode store, not a derived index. Deep sleep is held while a job
/// is open, so there should be nothing in it covering redacted events; a caller
/// that needs to be sure checks separately rather than dropping the file.
///
/// Vacuumed afterwards, for the reason given on `archive::forget_events`.
pub(super) fn forget(source: &Path, ids: &[u64]) -> Result<usize> {
    let path = source.with_extension("semantic.sqlite3");
    if !path.exists() || ids.is_empty() {
        return Ok(0);
    }
    let mut db = connect(&path)?;
    let mut removed = 0;
    {
        let tx = db.transaction()?;
        for id in ids {
            let id = i64::try_from(*id).unwrap_or(i64::MAX);
            removed += tx.execute("DELETE FROM events WHERE id=?1", [id])?;
            tx.execute("DELETE FROM pending WHERE id=?1", [id])?;
            tx.execute("DELETE FROM skipped WHERE id=?1", [id])?;
        }
        tx.commit()?;
    }
    // Outside the transaction: VACUUM cannot run inside one. Neither this nor
    // the checkpoint can fail the removal -- the rows are already gone, and
    // reporting a failure here would say nothing was removed when it was.
    crate::hippocampus::compact(&db);
    Ok(removed)
}

fn initialize(path: &Path, config: &Config) -> Result<usize> {
    let mut db = connect(path)?;
    db.execute_batch(
        "PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    )?;
    let layout: Option<String> = db
        .query_row("SELECT value FROM metadata WHERE key='layout'", [], |r| r.get(0))
        .optional()?;
    if layout.as_deref() != Some(LAYOUT) {
        db.execute_batch(
            "DROP TABLE IF EXISTS chunk_vectors;
             DROP TABLE IF EXISTS chunks;
             DROP TABLE IF EXISTS events;
             DROP TABLE IF EXISTS pending;
             DROP TABLE IF EXISTS skipped;
             DELETE FROM metadata WHERE key='dimensions';",
        )?;
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('layout',?1)", [LAYOUT])?;
    }
    db.execute_batch("
        CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, digest TEXT NOT NULL, epoch INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks (event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
          ordinal INTEGER NOT NULL, start_byte INTEGER NOT NULL, end_byte INTEGER NOT NULL,
          text TEXT NOT NULL, PRIMARY KEY(event_id,ordinal));
        -- Vectors live apart from the text they belong to, because a search
        -- reads every vector and almost none of the text. In one table the
        -- vector sits last in a row that overflows its page, so reaching it
        -- means walking past 1800 bytes of text whether or not the text is
        -- selected: measured, a scan over 30k chunks took 116 ms that way and
        -- 73 ms over a table of nothing but vectors.
        CREATE TABLE IF NOT EXISTS chunk_vectors (event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
          ordinal INTEGER NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(event_id,ordinal));
        -- Events read but not yet embedded, drained newest first.
        CREATE TABLE IF NOT EXISTS pending (id INTEGER PRIMARY KEY, digest TEXT NOT NULL);
        -- Events this index holds no vectors for on purpose, so that a search
        -- can tell them from a backlog.
        CREATE TABLE IF NOT EXISTS skipped (id INTEGER PRIMARY KEY, reason TEXT NOT NULL);")?;
    let existing: Option<String> = db
        .query_row(
            "SELECT value FROM metadata WHERE key='fingerprint'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if existing.as_deref() != Some(config.fingerprint().as_str()) {
        let tx = db.transaction()?;
        let dropped: i64 = tx.query_row("SELECT count(*) FROM events", [], |r| r.get(0))?;
        let before: Option<String> = tx
            .query_row("SELECT value FROM metadata WHERE key='settings'", [], |r| r.get(0))
            .optional()?;
        tx.execute("DELETE FROM events", [])?;
        // A rebuild re-decides every event, so neither queue survives it.
        tx.execute("DELETE FROM pending", [])?;
        tx.execute("DELETE FROM skipped", [])?;
        // By key, not the lot: the layout row must survive, or every start
        // would read it as a layout change and drop the index again.
        tx.execute("DELETE FROM metadata WHERE key='dimensions'", [])?;
        // Replacing, not inserting: only the `dimensions` key was cleared
        // above, so these rows may still be the previous run's.
        tx.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('fingerprint',?1)",
            [config.fingerprint()],
        )?;
        tx.execute("INSERT OR REPLACE INTO metadata VALUES ('settings',?1)", [config.settings()])?;
        tx.commit()?;
        if existing.is_some() && dropped > 0 {
            crate::terminal::warn(&rebuild_notice(before.as_deref(), config, dropped));
        }
    } else {
        // Indexes from before settings were recorded: record them now, so a
        // later change can say what changed.
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('settings',?1)", [config.settings()])?;
    }
    Ok(usize::try_from(db.query_row(
        "SELECT count(*) FROM events",
        [],
        |r| r.get::<_, i64>(0),
    )?)?)
}
fn engine(source: &Path) -> Result<Arc<Engine>> {
    // Turned off while an index runs: searches use keywords only, and the
    // indexer waits (`Engine::run`). Turned on again, it carries on.
    if !crate::model_config::embeddings_on() {
        return Err(crate::model_config::EMBEDDINGS_OFF.into());
    }
    let source = if source.is_absolute() {
        source.to_owned()
    } else {
        std::env::current_dir()?.join(source)
    };
    let mut engines = ENGINES
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "archive manager lock poisoned")?;
    if let Some(engine) = engines.get(&source) {
        return Ok(engine.clone());
    }
    let config = Config::load()?;
    let db = source.with_extension("semantic.sqlite3");
    let count = initialize(&db, &config)?;
    let through = newest_indexed(&connect(&db)?)?;
    let result = Arc::new(Engine {
        db,
        config,
        status: Mutex::new(IndexStatus {
            state: "starting".into(),
            indexed_events: count,
            pending_events: 0,
            skipped_events: 0,
            source_bytes: 0,
            indexed_source_bytes: 0,
            last_error: None,
            newest_indexed_event: through,
            last_refusal: None,
        }),
    });
    let background = result.clone();
    let worker_source = source.clone();
    std::thread::Builder::new()
        .name("archive-index".into())
        .spawn(move || {
            // Recover the status even if an unexpected indexer panic occurs.
            let worker = background.clone();
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker.run(&worker_source)))
                .is_err()
                && let Ok(mut status) = background.status.lock() {
                    status.state = "failed".into();
                    status.last_error = Some("indexer panicked; restart harness to retry".into());
                }
        })?;
    engines.insert(source, result.clone());
    Ok(result)
}

pub(super) fn start(source: &Path) -> Result<()> {
    engine(source).map(|_| ())
}
pub(super) fn status(source: &Path) -> Result<IndexStatus> {
    let engine = engine(source)?;
    let mut status = engine
        .status
        .lock()
        .map_err(|_| "archive status lock poisoned")?
        .clone();
    if let Ok(meta) = std::fs::metadata(source) {
        status.source_bytes = meta.len();
        if status.state == "ready" && status.source_bytes != status.indexed_source_bytes {
            status.state = "catching_up".into();
        }
    }
    Ok(status)
}

/// Passes in a row that failed → seconds until the next try: 15 s while
/// healthy, then 30, 60, 120, 240 and at most 300 s. A server that is down or
/// overloaded isn't asked again every 15 s, and the archive being mid-write
/// (`WriterBusy`) doesn't count.
fn retry_after(failures: u32) -> u64 {
    if failures == 0 {
        15
    } else {
        (15u64 << failures.min(5)).min(300)
    }
}

/// The archive's last line is still being written; the pass tries again soon.
#[derive(Debug)]
struct WriterBusy;
impl std::fmt::Display for WriterBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("archive has an incomplete final line; retrying after writer finishes")
    }
}
impl Error for WriterBusy {}

impl Engine {
    fn run(&self, source: &Path) {
        let mut last_stamp = None;
        // Where the last pass stopped, so the next one reads on from there; and
        // whether the pruning a whole pass owes is still to do (see `sync_from`).
        // A new process starts with a whole pass.
        let mut mark: Option<ReadMark> = None;
        let mut prune_owed = true;
        let mut failures = 0u32;
        // What the operator was last told, so a failure is said once, not every pass.
        let mut announced: Option<String> = None;
        loop {
            if !crate::model_config::embeddings_on() {
                if let Ok(mut status) = self.status.lock() {
                    status.state = "off".into();
                }
                // Whatever came in meanwhile is read on the first pass after.
                last_stamp = None;
                std::thread::sleep(Duration::from_secs(15));
                continue;
            }
            let stamp = std::fs::metadata(source)
                .ok()
                .map(|m| (m.len(), m.modified().unwrap_or(SystemTime::UNIX_EPOCH)));
            if let Some((source_bytes, _)) = stamp.filter(|_| stamp != last_stamp) {
                if let Ok(mut status) = self.status.lock() {
                    status.state = "indexing".into();
                    status.source_bytes = source_bytes;
                }
                let outcome = (|| -> Result<()> {
                    let client = embedding_client()?;
                    let mut db = connect(&self.db)?;
                    sync_from(
                        &mut db,
                        source,
                        &mut mark,
                        &mut prune_owed,
                        // Each piece waits for any generation in flight first.
                        &mut |text| {
                            yield_to_generation();
                            embed(&client, &self.config, text, false)
                        },
                        &mut |coverage: Coverage| {
                            if let Ok(mut s) = self.status.lock() {
                                s.indexed_events = coverage.indexed_events;
                                s.pending_events = coverage.pending_events;
                                s.skipped_events = coverage.skipped_events;
                                s.newest_indexed_event = coverage.newest_indexed;
                                if let Some(refused) = &coverage.refused {
                                    s.last_refusal = Some(refused.clone());
                                }
                            }
                            if let Some(refused) = coverage.refused {
                                crate::terminal::warn_status(&format!("[archive] {refused}"));
                            }
                        },
                    )?;
                    Ok(())
                })();
                let busy = matches!(&outcome, Err(error) if error.is::<WriterBusy>());
                if let Ok(mut status) = self.status.lock() {
                    match &outcome {
                        Ok(()) => {
                            status.state = "ready".into();
                            status.last_error = None;
                            status.indexed_source_bytes = source_bytes;
                            last_stamp = stamp;
                        }
                        Err(error) => {
                            status.state = "retrying".into();
                            status.last_error = Some(error.to_string());
                        }
                    }
                }
                match outcome {
                    Ok(()) => {
                        if failures > 0 && announced.take().is_some() {
                            crate::terminal::say_status("[archive] semantic indexing is working again");
                        }
                        failures = 0;
                    }
                    Err(_) if busy => {}
                    Err(error) => {
                        failures += 1;
                        let error = error.to_string();
                        // Said once per kind of failure; a blip that clears on
                        // the next pass isn't worth a line.
                        if failures >= 2 && announced.as_deref() != Some(error.as_str()) {
                            crate::terminal::warn_status(&format!(
                                "[archive] semantic indexing is stalled: {error}. It retries with growing pauses, up to {} minutes apart; keyword search is unaffected.",
                                retry_after(u32::MAX) / 60
                            ));
                            announced = Some(error);
                        }
                    }
                }
            } else if stamp.is_none() {
                if let Ok(mut status) = self.status.lock() {
                    status.state = "waiting_for_archive".into();
                }
                last_stamp = None;
                mark = None;
            }
            std::thread::sleep(Duration::from_secs(retry_after(failures)));
        }
    }
}

fn embedding_client() -> Result<Client> {
    Ok(Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()?)
}
#[derive(Deserialize)]
struct Embeddings {
    data: Vec<Embedding>,
}
#[derive(Deserialize)]
struct Embedding {
    embedding: Vec<f32>,
}
fn normalize(mut values: Vec<f32>) -> Result<Vec<f32>> {
    if values.is_empty() || values.len() > MAX_DIMENSIONS || values.iter().any(|v| !v.is_finite()) {
        return Err("invalid embedding dimensions or non-finite values".into());
    }
    let norm = values
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm <= 0.0 {
        return Err("zero embedding vector".into());
    }
    for value in &mut values {
        *value = (*value as f64 / norm) as f32;
    }
    Ok(values)
}
fn embed(client: &Client, config: &Config, text: &str, query: bool) -> Result<Vec<f32>> {
    let prefix = if query {
        &config.query_prefix
    } else {
        &config.document_prefix
    };
    embed_prefixed(client, config, prefix, text)
}
fn embed_prefixed(client: &Client, config: &Config, prefix: &str, text: &str) -> Result<Vec<f32>> {
    if !crate::model_config::embeddings_on() {
        return Err(crate::model_config::EMBEDDINGS_OFF.into());
    }
    if text.len() > CHUNK_BYTES {
        return Err(
            "semantic query exceeds 1800 UTF-8 bytes; use a shorter description or literal mode"
                .into(),
        );
    }
    let response = client
        .post(&config.url)
        .json(&serde_json::json!({
            "model":config.model, "input":format!("{prefix}{text}"), "encoding_format":"float"
        }))
        .send()
        .map_err(|error| unreachable_server(&config.url, &error))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        return Err(Box::new(Refused { status: status.as_u16(), reason: server_reason(&body) }));
    }
    let response: Embeddings = response
        .json()
        .map_err(|error| format!("the embedding server's reply is not an embeddings response: {error}"))?;
    if response.data.len() != 1 {
        return Err("embedding endpoint must return one pooled vector per input".into());
    }
    normalize(response.data.into_iter().next().unwrap().embedding).map_err(|error| -> Box<dyn Error> {
        Box::new(Refused::unusable(error))
    })
}

/// The embedding server answered and said no (or answered with a vector that
/// can't be used). It is up, so the text may be what it can't take: llama.cpp,
/// for one, refuses an input longer than its batch ("input is too large to
/// process. increase the physical batch size"), and 1800 bytes of code or JSON
/// can be more tokens than its default 512.
#[derive(Debug)]
pub(super) struct Refused {
    status: u16,
    reason: String,
}
impl Refused {
    /// It answered (HTTP 2xx), but with a vector that can't be used: empty,
    /// too long, all zeros, or not finite.
    fn unusable(error: impl std::fmt::Display) -> Self {
        Self { status: 200, reason: error.to_string() }
    }
    fn answered(&self) -> bool {
        (200..300).contains(&self.status)
    }
    /// For a server that refuses even a two-letter text: the reason, once.
    fn everything(&self) -> String {
        if self.answered() {
            format!("the embedding server answers everything, even a two-letter text, with an unusable vector ({})", self.reason)
        } else {
            format!("the embedding server refuses everything, even a two-letter text (HTTP {}): {}", self.status, self.reason)
        }
    }
}
impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.answered() {
            write!(f, "the embedding server answered with an unusable vector ({})", self.reason)
        } else {
            write!(f, "the embedding server refused it (HTTP {}): {}", self.status, self.reason)
        }
    }
}
impl Error for Refused {}

/// The server's own words from an error body: OpenAI and llama.cpp put them in
/// `error.message`, TEI in `error`, vLLM in `message`; otherwise the body.
fn server_reason(body: &str) -> String {
    let parsed: Option<serde_json::Value> = serde_json::from_str(body).ok();
    let said = parsed.as_ref().and_then(|value| {
        [&value["error"]["message"], &value["error"], &value["message"], &value["detail"]]
            .into_iter()
            .find_map(|field| field.as_str().map(str::to_owned))
    });
    let text = said.unwrap_or_else(|| body.to_owned());
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match text.char_indices().nth(300) {
        _ if text.is_empty() => "no reason given".to_owned(),
        Some((cut, _)) => format!("{}…", &text[..cut]),
        None => text,
    }
}

/// A request that never got an answer, in words: where, and what happened.
/// reqwest's own text ("error sending request for url (…)") hides the cause.
fn unreachable_server(url: &str, error: &reqwest::Error) -> String {
    let at = reqwest::Url::parse(url)
        .ok()
        .and_then(|url| Some(format!("{}:{}", url.host_str()?, url.port_or_known_default()?)))
        .unwrap_or_else(|| url.to_owned());
    let mut cause: &dyn Error = error;
    while let Some(inner) = cause.source() {
        cause = inner;
    }
    if error.is_timeout() {
        format!("the embedding server at {at} did not answer in time ({cause}); it may be busy with other embeddings")
    } else if error.is_connect() {
        format!("could not connect to the embedding server at {at} ({cause}); is it running?")
    } else {
        format!("the request to the embedding server at {at} failed ({cause})")
    }
}

/// Chunks of one event that are embedded: all of them up to
/// `MAX_EMBEDDED_CHUNKS`; past that, the head and the tail. A VM result can now
/// be kept whole up to a million characters a stream, and embedding all of it
/// would be about 650 calls a stream; its head and tail are where a search
/// usually finds it, and the keyword index still has every word of it.
const MAX_EMBEDDED_CHUNKS: usize = 48;
const EMBEDDED_TAIL_CHUNKS: usize = 16;

fn embedded_chunks(mut parts: Vec<(usize, usize, &str)>) -> Vec<(usize, usize, &str)> {
    if parts.len() > MAX_EMBEDDED_CHUNKS {
        let tail = parts.split_off(parts.len() - EMBEDDED_TAIL_CHUNKS);
        parts.truncate(MAX_EMBEDDED_CHUNKS - EMBEDDED_TAIL_CHUNKS);
        parts.extend(tail);
    }
    parts
}

/// A line naming where a chunk came from: its kind, the tool call that
/// produced it, and when it was recorded.
///
/// Embedded with the chunk but not stored with it. A chunk is a byte window,
/// so from the second one of a long tool result the text no longer says which
/// tool ran or when — "the df output from Tuesday" had nothing to match on.
/// The header gives every chunk that much. Keeping it out of the stored text
/// means an excerpt stays verbatim archive text and its byte offsets still
/// locate it exactly.
///
/// A query carries no header, which is the usual asymmetry for a model with
/// separate query and passage prefixes; the header is short beside 1800 bytes
/// of text, so it tilts a passage rather than dominating it.
fn chunk_header(event: &SourceEvent) -> String {
    let kind = super::source_kind(&event.kind, &event.content);
    let at = super::rank::short_time(&event.timestamp);
    let called = tool_call(&event.content)
        .map(|(tool, arguments)| format!(" {tool}({arguments})"))
        .unwrap_or_default();
    let mut header = format!("[{kind}{called} · {at}]\n");
    if header.len() > MAX_HEADER_BYTES {
        let mut end = MAX_HEADER_BYTES - 2;
        while !header.is_char_boundary(end) {
            end -= 1;
        }
        header = format!("{}]\n", &header[..end]);
    }
    header
}

/// The tool a result came from and a short form of its arguments, from the
/// record `agent.rs` writes: `[tool:<name>]\narguments:\n…\nresult:\n…`.
fn tool_call(content: &str) -> Option<(String, String)> {
    let (name, rest) = content.strip_prefix("[tool:")?.split_once(']')?;
    let arguments = rest
        .split_once("arguments:")
        .map(|(_, after)| after.split("result:").next().unwrap_or(""))
        .unwrap_or("");
    Some((name.to_owned(), super::one_line(arguments, 60)))
}

/// Byte-bounded UTF-8 chunks avoid silently truncating long tool/ reasoning
/// records. `budget` is the room left for text once a header is counted.
fn chunks(text: &str, budget: usize) -> Vec<(usize, usize, &str)> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let budget = budget.max(MIN_CHUNK_BYTES);
    let mut result = Vec::new();
    let mut start = 0;
    loop {
        let mut end = (start + budget).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        result.push((start, end, &text[start..end]));
        if end == text.len() {
            break;
        }
        let mut next = end.saturating_sub(OVERLAP_BYTES.min(budget / 2));
        while !text.is_char_boundary(next) {
            next += 1;
        }
        start = next.max(start + 1);
    }
    result
}
/// Stored vectors are two bytes a component rather than four, because a
/// search reads every one of them: at 2048 dimensions that is 4 KB a passage
/// instead of 8 KB, and on a complete index of the operator's archive about 550 MB a
/// search instead of 1.1 GB.
///
/// Fixed point rather than IEEE half precision. Every stored vector is unit
/// length, so each component is in [-1, 1], and over that range a scaled i16
/// resolves to 3.1e-5 everywhere — finer than f16, which spends its precision
/// on an exponent this data never uses, and without hand-rolling subnormal
/// rounding. The error it adds to a cosine is some four orders of magnitude
/// below the gaps that decide a ranking; `quantising_a_vector_does_not_change_
/// what_ranks_first` holds it to that.
/// The shape of the derived tables. A change here cannot be migrated, so they
/// are dropped and rebuilt; it is kept apart from the fingerprint, which is
/// about what the vectors mean rather than where they live.
const LAYOUT: &str = "chunks-and-vectors-v1";

const VECTOR_SCALE: f32 = 32_767.0;
/// Multiplying by this beats dividing by `VECTOR_SCALE` once a scan is doing
/// it tens of millions of times.
const VECTOR_RESCALE: f32 = 1.0 / VECTOR_SCALE;

fn pack(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|v| ((v.clamp(-1.0, 1.0) * VECTOR_SCALE).round() as i16).to_le_bytes())
        .collect()
}

fn unpack(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err("corrupt vector byte length; rebuild the derived index".into());
    }
    let result: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|v| f32::from(i16::from_le_bytes(v.try_into().unwrap())) * VECTOR_RESCALE)
        .collect();
    if result.is_empty() || result.len() > MAX_DIMENSIONS || result.iter().any(|v| !v.is_finite()) {
        return Err("corrupt vector; rebuild the derived index".into());
    }
    Ok(result)
}
#[derive(Deserialize)]
struct SourceEvent {
    id: u64,
    #[serde(default)]
    timestamp: String,
    kind: String,
    content: String,
}

/// A whole pass over the archive (see `sync_from`).
#[cfg(test)]
fn sync_archive(
    db: &mut Connection,
    source: &Path,
    embed_document: &mut impl FnMut(&str) -> Result<Vec<f32>>,
    progress: &mut impl FnMut(Coverage),
) -> Result<()> {
    sync_from(db, source, &mut None, &mut true, embed_document, progress)
}

/// Bring the index in line with the archive, in two stages.
///
/// **Reading** walks the archive and settles what each event needs: nothing
/// (its vectors are current), a place in the `pending` queue (new or changed),
/// or a row in `skipped` (a kind this index does not embed). It touches no
/// model server, so it always finishes, and with a `mark` from an earlier pass
/// over a file that still holds what that pass read (`ReadMark::still_holds`)
/// it reads only what was appended. An "append" carrying an id already known
/// means something other than an append happened, so the whole archive is read
/// instead, as it is on the first pass of a process.
///
/// **Embedding** then drains the queue newest first. That ordering is the point
/// of the split: a backlog used to be worked oldest first, so during the hours
/// or days of a rebuild the events most likely to be searched for — the recent
/// ones — were the last to become findable. Whatever stops the drain (an
/// embedder outage, a refusal) leaves the queue to the next pass, so progress
/// is never re-read or re-hashed.
///
/// Events gone from the archive are pruned when a pass reaches the end after a
/// whole read began (`prune_owed`), so a half-written archive never loses
/// records and a whole pass interrupted mid-drain still owes its pruning to
/// whichever later pass finishes.
fn sync_from(
    db: &mut Connection,
    source: &Path,
    mark: &mut Option<ReadMark>,
    prune_owed: &mut bool,
    embed_document: &mut impl FnMut(&str) -> Result<Vec<f32>>,
    progress: &mut impl FnMut(Coverage),
) -> Result<()> {
    let meta = std::fs::metadata(source)?;
    let mut appended = false;
    if let Some(from) = *mark
        && from.still_holds(source, &meta)? {
            let mut read = from;
            let outcome = read_lines(db, source, &mut read, false, progress);
            *mark = Some(read);
            appended = matches!(outcome?, Lines::End);
        }
    if !appended {
        *prune_owed = true;
        let mut read = ReadMark::start(&meta);
        let outcome = read_lines(db, source, &mut read, true, progress);
        *mark = Some(read);
        if matches!(outcome?, Lines::NotAppended) {
            return Err("a whole read cannot find an append out of place".into());
        }
    }
    drain(db, source, embed_document, progress)?;
    finish_pass(db, source, prune_owed, progress)
}

/// How a run of lines ended.
enum Lines {
    /// Every complete line was read.
    End,
    /// A line after the mark carries an id the index already knows, so the
    /// file did not only grow: the caller reads the whole archive.
    NotAppended,
}

/// Why an event is in the index without vectors.
const SKIPPED_KIND: &str = "kind";
/// The agent's own tentative analysis. `archive_search` leaves reasoning out of its
/// default kinds, so embedding it spent the shared embedder on passages the
/// search would not offer anyway; on the operator's archive tool results and reasoning
/// together were 91% of all chunks. Keyword matching still finds every word of
/// it, which is what `kinds: ["reasoning"]` searches now rely on.
const SKIPPED_REASONING: &str = "reasoning";

/// Read the lines after `mark`, settling each event, and advance the mark past
/// each one read. `whole` means the mark is the start of the file: every event
/// is compared with what the index holds, and an id seen twice is an error.
fn read_lines(
    db: &mut Connection,
    source: &Path,
    mark: &mut ReadMark,
    whole: bool,
    progress: &mut impl FnMut(Coverage),
) -> Result<Lines> {
    use std::io::{Seek, SeekFrom};
    let mut file = File::open(source)?;
    let source_len = file.metadata()?.len();
    file.seek(SeekFrom::Start(mark.through))?;
    let mut seen = HashSet::new();
    let mut reader = BufReader::new(file.take(source_len.saturating_sub(mark.through)));
    let mut line = String::new();
    let tx = db.transaction()?;
    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            break;
        }
        // A writer may be between serde_json::to_writer and writeln. Retry later.
        if !line.ends_with('\n') {
            return Err(Box::new(WriterBusy));
        }
        if line.trim().is_empty() {
            mark.advance(read, None);
            continue;
        }
        let event: SourceEvent = serde_json::from_str(&line)?;
        let id = i64::try_from(event.id).map_err(|_| "event ID exceeds SQLite integer range")?;
        if !seen.insert(id) {
            return Err(format!("duplicate archive event ID {}", event.id).into());
        }
        let known = |table: &str| -> Result<Option<String>> {
            Ok(tx
                .query_row(&format!("SELECT digest FROM {table} WHERE id=?1"), [id], |r| r.get(0))
                .optional()?)
        };
        if let Some(reason) = skip_reason(&event) {
            if !whole {
                // Already settled, so this is not an append.
                let settled: bool =
                    tx.query_row("SELECT EXISTS(SELECT 1 FROM skipped WHERE id=?1)", [id], |r| r.get(0))?;
                if settled {
                    tx.commit()?;
                    return Ok(Lines::NotAppended);
                }
            }
            tx.execute("INSERT OR REPLACE INTO skipped VALUES (?1,?2)", params![id, reason])?;
            tx.execute("DELETE FROM pending WHERE id=?1", [id])?;
            // Vectors a previous policy left behind go now, so a search can
            // never match what this index says it does not hold.
            tx.execute("DELETE FROM events WHERE id=?1", [id])?;
            mark.advance(read, Some(event.id));
            continue;
        }
        let hash = digest(&event.content);
        let indexed = known("events")?;
        let queued = known("pending")?;
        if !whole && (indexed.is_some() || queued.is_some()) {
            tx.commit()?;
            return Ok(Lines::NotAppended);
        }
        // Current vectors: nothing to write. Writing here used to stamp the
        // pass's epoch on the row, one autocommit per archive event on every
        // pass — tens of thousands of back-to-back write transactions each
        // time the archive changed. The write lock was almost never free, and
        // the episode indexer sharing this file timed out with "database is
        // locked".
        if indexed.as_deref() == Some(&hash) {
            if queued.is_some() {
                tx.execute("DELETE FROM pending WHERE id=?1", [id])?;
            }
            mark.advance(read, Some(event.id));
            continue;
        }
        if queued.as_deref() != Some(&hash) {
            tx.execute("INSERT OR REPLACE INTO pending VALUES (?1,?2)", params![id, hash])?;
        }
        tx.execute("DELETE FROM skipped WHERE id=?1", [id])?;
        mark.advance(read, Some(event.id));
    }
    tx.commit()?;
    progress(count_coverage(db)?);
    Ok(Lines::End)
}

/// Why this index holds no vectors for an event, or `None` when it embeds it.
///
/// Settling an event this way is not the same as leaving it out: `settled`
/// counts it, so a search is never told it is waiting for vectors that are
/// never coming, and a reasoning record that stops being skipped (see
/// `embed_reasoning`) is queued by the next whole read.
fn skip_reason(event: &SourceEvent) -> Option<&'static str> {
    if !matches!(event.kind.as_str(), "operator" | "s0ma") {
        return Some(SKIPPED_KIND);
    }
    (!embed_reasoning() && super::source_kind(&event.kind, &event.content) == "reasoning")
        .then_some(SKIPPED_REASONING)
}

/// `S0MA_ARCHIVE_EMBED_REASONING=1` puts reasoning back in the index. Off by
/// default. Not part of the fingerprint: turning it on queues the reasoning
/// events on the next whole read, and turning it off drops their rows, so
/// neither direction costs a rebuild of everything else.
fn embed_reasoning() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| embed_reasoning_from(|key| std::env::var(key).ok()))
}

fn embed_reasoning_from(get: impl Fn(&str) -> Option<String>) -> bool {
    matches!(
        get("S0MA_ARCHIVE_EMBED_REASONING").map(|value| value.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}

/// Embed the queue, newest first, until it is empty.
///
/// Each event is read from the archive by the line index rather than carried
/// over from the read stage, so what is embedded is what the file holds now;
/// an event edited between the two stages is embedded as it currently reads,
/// under its current digest. A complete event and its vectors commit together,
/// a failed embedding leaves the previous version in place, and source
/// validation keeps that stale copy from being served.
fn drain(
    db: &mut Connection,
    source: &Path,
    embed_document: &mut impl FnMut(&str) -> Result<Vec<f32>>,
    progress: &mut impl FnMut(Coverage),
) -> Result<()> {
    // Counted once and kept by hand: re-counting three tables after every
    // event would be tens of thousands of full scans while a backlog drains.
    let mut coverage = count_coverage(db)?;
    loop {
        let next: Option<i64> = db
            .query_row("SELECT id FROM pending ORDER BY id DESC LIMIT 1", [], |r| r.get(0))
            .optional()?;
        let Some(id) = next else { return Ok(()) };
        let event = read_event(source, u64::try_from(id)?)?;
        let Some(event) = event else {
            // Gone from the archive since the read; `finish_pass` prunes the rest.
            db.execute("DELETE FROM pending WHERE id=?1", [id])?;
            coverage.pending_events = coverage.pending_events.saturating_sub(1);
            continue;
        };
        if let Some(reason) = skip_reason(&event) {
            let tx = db.transaction()?;
            tx.execute("INSERT OR REPLACE INTO skipped VALUES (?1,?2)", params![id, reason])?;
            tx.execute("DELETE FROM pending WHERE id=?1", [id])?;
            tx.execute("DELETE FROM events WHERE id=?1", [id])?;
            tx.commit()?;
            coverage.pending_events = coverage.pending_events.saturating_sub(1);
            coverage.skipped_events += 1;
            continue;
        }
        let hash = digest(&event.content);
        let header = chunk_header(&event);
        let mut records = Vec::new();
        let mut refused = Vec::new();
        for (start, _end, text) in embedded_chunks(chunks(&event.content, CHUNK_BYTES - header.len())) {
            embed_piece(embed_document, &header, start, text, &mut records, &mut refused)?;
        }
        // Embedding took a model call, and the archive may have moved under it:
        // a job's close redacts events and then clears this index, and an
        // insert landing after that would put the text and its vector back --
        // after the file had been vacuumed, so the bytes would stay in it.
        // Re-read and compare before writing. The window is now the few
        // microseconds between this check and the commit, where before it was
        // the whole embedding call.
        match read_event(source, u64::try_from(id).unwrap_or(0)) {
            Ok(Some(current)) if digest(&current.content) == hash => {}
            Ok(_) => {
                // Gone or changed: drop the work and let a later pass index
                // whatever is there now.
                db.execute("DELETE FROM pending WHERE id=?1", [id])?;
                continue;
            }
            Err(error) => return Err(error),
        }
        let tx = db.transaction()?;
        let dimensions: Option<usize> = tx
            .query_row("SELECT value FROM metadata WHERE key='dimensions'", [], |r| r.get::<_, String>(0))
            .optional()?
            .map(|v| v.parse())
            .transpose()?;
        let expected = dimensions.or_else(|| records.first().map(|r| r.3.len()));
        if records.iter().any(|r| Some(r.3.len()) != expected) {
            return Err("embedding dimensions changed; set S0MA_ARCHIVE_EMBEDDING_REVISION and restart to rebuild".into());
        }
        if let Some(n) = expected {
            tx.execute("INSERT OR REPLACE INTO metadata VALUES ('dimensions',?1)", [n.to_string()])?;
        }
        let replaced = tx.execute("DELETE FROM events WHERE id=?1", [id])?;
        tx.execute("INSERT INTO events VALUES (?1,?2,0)", params![id, hash])?;
        for (ordinal, (start, end, text, vector)) in records.into_iter().enumerate() {
            tx.execute(
                "INSERT INTO chunks VALUES (?1,?2,?3,?4,?5)",
                params![id, ordinal as i64, start as i64, end as i64, text],
            )?;
            tx.execute(
                "INSERT INTO chunk_vectors VALUES (?1,?2,?3)",
                params![id, ordinal as i64, pack(&vector)],
            )?;
        }
        tx.execute("DELETE FROM pending WHERE id=?1", [id])?;
        tx.commit()?;
        coverage.pending_events = coverage.pending_events.saturating_sub(1);
        if replaced == 0 {
            coverage.indexed_events += 1;
        }
        coverage.newest_indexed = coverage.newest_indexed.max(Some(event.id));
        coverage.refused = (!refused.is_empty()).then(|| Refusal::describe(event.id, &refused));
        progress(coverage.clone());
    }
}

/// One event from the archive, by the line index.
fn read_event(source: &Path, id: u64) -> Result<Option<SourceEvent>> {
    let Some(offset) = crate::hippocampus::event_offset(source, id)? else {
        return Ok(None);
    };
    let mut file = File::open(source)?;
    let Some(line) = crate::hippocampus::line_at(&mut file, offset)? else {
        return Ok(None);
    };
    let event: SourceEvent = serde_json::from_slice(&line)?;
    // The line index is checked on every read, but an id that moved under it
    // is still worth refusing rather than embedding as another event.
    Ok((event.id == id).then_some(event))
}

/// What the index holds right now.
fn count_coverage(db: &Connection) -> Result<Coverage> {
    let count = |table: &str| -> Result<usize> {
        Ok(usize::try_from(db.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| {
            r.get::<_, i64>(0)
        })?)?)
    };
    Ok(Coverage {
        indexed_events: count("events")?,
        skipped_events: count("skipped")?,
        pending_events: count("pending")?,
        newest_indexed: newest_indexed(db)?,
        refused: None,
    })
}

/// The end of a pass: prune what a whole read owes, then report coverage.
/// Usually nothing is stale, so usually nothing is written; when something is,
/// it goes in one short transaction.
fn finish_pass(
    db: &mut Connection,
    source: &Path,
    prune_owed: &mut bool,
    progress: &mut impl FnMut(Coverage),
) -> Result<()> {
    if *prune_owed {
        let live = crate::hippocampus::event_ids(source)?;
        let mut stale: Vec<(&'static str, i64)> = Vec::new();
        for table in ["events", "skipped", "pending"] {
            let ids: Vec<i64> = db
                .prepare(&format!("SELECT id FROM {table}"))?
                .query_map([], |r| r.get(0))?
                .collect::<std::result::Result<Vec<i64>, _>>()?;
            stale.extend(
                ids.into_iter()
                    .filter(|id| u64::try_from(*id).map_or(true, |id| !live.contains(&id)))
                    .map(|id| (table, id)),
            );
        }
        if !stale.is_empty() {
            let tx = db.transaction()?;
            for (table, id) in stale {
                tx.execute(&format!("DELETE FROM {table} WHERE id=?1"), [id])?;
            }
            tx.commit()?;
        }
        *prune_owed = false;
    }
    progress(count_coverage(db)?);
    Ok(())
}

/// A refused piece is split in two and each half tried; one still refused at
/// under twice this is skipped. 1800 → 900 → 450 → 225 bytes: at most 15
/// requests for a chunk refused at every size.
const MIN_PIECE_BYTES: usize = 200;

/// Text the embedding server would not take, even in small pieces.
struct Refusal {
    bytes: usize,
    reason: String,
}
impl Refusal {
    /// One line for the operator and for `archive_search`'s status.
    fn describe(event: u64, refused: &[Refusal]) -> String {
        let bytes: usize = refused.iter().map(|r| r.bytes).sum();
        format!(
            "skipped {} piece(s), {bytes} bytes, of event #{event} that the embedding server refused even in pieces of {MIN_PIECE_BYTES}–{} bytes ({}); keyword search still finds that text",
            refused.len(),
            2 * MIN_PIECE_BYTES - 1,
            refused[0].reason
        )
    }
}

/// Embed one chunk. If the server refuses it, embed its halves instead, and so
/// on down to `MIN_PIECE_BYTES`; a piece still refused there is skipped, so one
/// text the server can't take doesn't stop the index at that event forever
/// (before: every pass stopped there, 15 s later tried again, re-embedding the
/// event's other chunks each time, and nothing after it was ever indexed).
/// Skipping needs proof the server takes anything at all: if a two-letter probe
/// is refused too, the server is the problem, and the pass stops as for an
/// outage, keeping the backlog. Anything that isn't a refusal (no connection,
/// a timeout) stops the pass as before.
/// One embedding, unit length; an unusable vector counts as a refusal.
fn checked(embed_document: &mut impl FnMut(&str) -> Result<Vec<f32>>, text: &str) -> Result<Vec<f32>> {
    embed_document(text).and_then(|vector| normalize(vector).map_err(|error| -> Box<dyn Error> { Box::new(Refused::unusable(error)) }))
}

fn embed_piece<'a>(
    embed_document: &mut impl FnMut(&str) -> Result<Vec<f32>>,
    header: &str,
    start: usize,
    text: &'a str,
    records: &mut Vec<(usize, usize, &'a str, Vec<f32>)>,
    refused: &mut Vec<Refusal>,
) -> Result<()> {
    // The header goes to the server with every piece, halves included, and
    // into no record: what is stored is the archive's own text.
    let error = match checked(embed_document, &format!("{header}{text}")) {
        Ok(vector) => {
            records.push((start, start + text.len(), text, vector));
            return Ok(());
        }
        Err(error) => error,
    };
    let Some(refusal) = error.downcast_ref::<Refused>() else {
        return Err(error);
    };
    if text.len() >= 2 * MIN_PIECE_BYTES {
        let mut middle = text.len() / 2;
        while !text.is_char_boundary(middle) {
            middle -= 1;
        }
        embed_piece(embed_document, header, start, &text[..middle], records, refused)?;
        return embed_piece(embed_document, header, start + middle, &text[middle..], records, refused);
    }
    if let Err(probe) = checked(embed_document, "ok") {
        return Err(match probe.downcast_ref::<Refused>() {
            // The usual case (the server is misconfigured): say the reason once.
            Some(same) if same.status == refusal.status && same.reason == refusal.reason => same.everything(),
            _ => format!("{refusal}; a two-letter text failed too ({probe}), so the server is refusing everything"),
        }
        .into());
    }
    refused.push(Refusal { bytes: text.len(), reason: refusal.reason.clone() });
    Ok(())
}

#[derive(Debug, Clone)]
pub(super) struct Candidate {
    pub id: u64,
    pub digest: String,
    pub text: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub similarity: f32,
}
/// Best passage per event, best first. With `allow`, only those events compete,
/// so a narrow time window is searched exhaustively instead of being whatever
/// survived a global top-N.
/// Best passage per event, best first. With `allow`, only those events
/// compete, so a narrow time window or event range is searched exhaustively
/// instead of being whatever survived a global top-N.
///
/// Two stages, because the scan is the whole cost: the first reads vectors and
/// nothing else, and only the winners' text is fetched. Reading the text in
/// the scan meant carrying up to 1800 bytes a passage past the dot product
/// that discards almost all of them — on a complete index of the operator's archive,
/// some 240 MB of text a search, to use a few kilobytes of it.
fn nearest(
    db: &Connection,
    query: &[f32],
    limit: usize,
    allow: Option<&std::collections::HashSet<u64>>,
) -> Result<Vec<Candidate>> {
    let mut best: HashMap<u64, (i64, f32)> = HashMap::new();
    {
        let mut stmt = db.prepare("SELECT event_id,ordinal,vector FROM chunk_vectors")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id = u64::try_from(row.get::<_, i64>(0)?)?;
            if allow.is_some_and(|allow| !allow.contains(&id)) {
                continue;
            }
            let vector = unpack(&row.get::<_, Vec<u8>>(2)?)?;
            if vector.len() != query.len() {
                return Err(
                    "query/index embedding dimensions differ; check embedding model and rebuild".into(),
                );
            }
            let score: f32 = vector.iter().zip(query).map(|(a, b)| a * b).sum();
            if best.get(&id).is_none_or(|(_, old)| score > *old) {
                best.insert(id, (row.get(1)?, score));
            }
        }
    }
    let mut ranked: Vec<(u64, i64, f32)> =
        best.into_iter().map(|(id, (ordinal, score))| (id, ordinal, score)).collect();
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| b.0.cmp(&a.0)));
    ranked.truncate(limit);

    let mut fetch = db.prepare(
        "SELECT c.text,c.start_byte,c.end_byte,e.digest FROM chunks c JOIN events e ON e.id=c.event_id
         WHERE c.event_id=?1 AND c.ordinal=?2",
    )?;
    let mut candidates = Vec::with_capacity(ranked.len());
    for (id, ordinal, score) in ranked {
        // Gone since the scan: dropped rather than served without provenance.
        let Some((text, start_byte, end_byte, digest)) = fetch
            .query_row(params![i64::try_from(id)?, ordinal], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?))
            })
            .optional()?
        else {
            continue;
        };
        candidates.push(Candidate {
            id,
            digest,
            text,
            start_byte: usize::try_from(start_byte)?,
            end_byte: usize::try_from(end_byte)?,
            // Quantising leaves a dot product a hair outside [-1, 1].
            similarity: score.clamp(-1.0, 1.0),
        });
    }
    Ok(candidates)
}

pub(super) fn search(
    source: &Path,
    query: &str,
    limit: usize,
    allow: Option<&std::collections::HashSet<u64>>,
) -> Result<Vec<Candidate>> {
    let engine = engine(source)?;
    let db = connect(&engine.db)?;
    let count: i64 = db.query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let vector = embed(&embedding_client()?, &engine.config, query, true)?;
    nearest(&db, &vector, limit, allow)
}

/// The archive's embedding model for other harness features (the per-turn
/// skill match): stored text gets the document prefix, searches the query
/// prefix, and vectors come back unit length. The timeout only guards against a
/// hung server; callers wait for the answer otherwise.
pub(crate) struct Embedder {
    config: Config,
    client: Client,
}
const EMBEDDER_TIMEOUT: Duration = Duration::from_secs(120);
impl Embedder {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            config: Config::load()?,
            client: Client::builder()
                .connect_timeout(Duration::from_secs(2))
                .timeout(EMBEDDER_TIMEOUT)
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy()
                .build()?,
        })
    }
    /// Changes whenever stored vectors would: endpoint, model, prefixes, revision.
    pub(crate) fn fingerprint(&self) -> String {
        self.config.fingerprint()
    }
    /// At most `CHUNK_BYTES` (1800) bytes; see `truncate_for_embedding`.
    pub(crate) fn query(&self, text: &str) -> Result<Vec<f32>> {
        embed(&self.client, &self.config, text, true)
    }
    pub(crate) fn document(&self, text: &str) -> Result<Vec<f32>> {
        embed(&self.client, &self.config, text, false)
    }
}
/// The longest prefix of `text` an embedding call accepts, cut on a character
/// boundary.
pub(crate) fn truncate_for_embedding(text: &str) -> &str {
    let mut end = text.len().min(CHUNK_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Reuse the same local embedding model, document prefix and version
/// fingerprint; queries get the episode-recall prefix.
pub(super) struct EpisodeEmbedder {
    config: Config,
    client: Client,
}
/// Recall runs off the turn path, so this only bounds a stuck request. A full
/// 1800-byte query on a 1B CPU model takes seconds, and the single-slot server
/// may first finish an archive chunk it is indexing.
const EPISODE_EMBED_TIMEOUT: Duration = Duration::from_secs(20);
impl EpisodeEmbedder {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            config: Config::load()?,
            client: Client::builder()
                .connect_timeout(Duration::from_secs(2))
                .timeout(EPISODE_EMBED_TIMEOUT)
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy()
                .build()?,
        })
    }
    pub(super) fn fingerprint(&self) -> String {
        self.config.fingerprint()
    }
    pub(super) fn encode(&self, text: &str, query: bool) -> Result<Vec<f32>> {
        let prefix = if query {
            &self.config.episode_query_prefix
        } else {
            &self.config.document_prefix
        };
        embed_prefixed(&self.client, &self.config, prefix, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "s0ma_semantic_{}_{}",
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
        fn db(&self) -> PathBuf {
            self.0.join("index.sqlite3")
        }
        fn write(&self, events: &[(u64, &str)]) {
            let text:String=events.iter().map(|(id,content)|format!("{}\n",serde_json::json!({
                "id":id,"timestamp":"2026-09-13T00:00:00Z","kind":"s0ma","content":content,"intermediate":false
            }))).collect();
            std::fs::write(self.source(), text).unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn config() -> Config {
        Config {
            url: "http://127.0.0.1:8081/v1/embeddings".into(),
            model: "test".into(),
            query_prefix: "query: ".into(),
            document_prefix: "passage: ".into(),
            // Distinct from query_prefix so the tests can tell which one was used.
            episode_query_prefix: "episode query: ".into(),
            revision: "1".into(),
        }
    }
    /// What the index has settled, as `settled` reports it to a search.
    fn settled_ids(db: &Connection) -> Vec<i64> {
        let mut ids: Vec<i64> = db
            .prepare("SELECT id FROM events UNION SELECT id FROM skipped EXCEPT SELECT id FROM pending")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn an_event_this_index_never_embeds_is_settled_not_queued() {
        let f = Fixture::new();
        initialize(&f.db(), &config()).unwrap();
        let mut db = connect(&f.db()).unwrap();
        // A kind the index does not hold vectors for, between two it does.
        let text: String = [
            serde_json::json!({"id":1,"timestamp":"2026-09-13T00:00:00Z","kind":"s0ma","content":"first","intermediate":false}),
            serde_json::json!({"id":2,"timestamp":"2026-09-13T00:00:00Z","kind":"checkpoint","content":"not embedded"}),
            serde_json::json!({"id":3,"timestamp":"2026-09-13T00:00:00Z","kind":"operator","content":"third","intermediate":false}),
        ]
        .iter()
        .map(|line| format!("{line}\n"))
        .collect();
        std::fs::write(f.source(), text).unwrap();
        sync_archive(&mut db, &f.source(), &mut |_| Ok(vec![1.0, 0.0]), &mut |_| {}).unwrap();
        // Settled, so a search is not told it is missing something; and not in
        // the index proper, so it cannot be matched.
        assert_eq!(settled_ids(&db), [1, 2, 3]);
        assert_eq!(count(&db), 2);
        let reason: String = db
            .query_row("SELECT reason FROM skipped WHERE id=2", [], |r| r.get(0))
            .unwrap();
        assert_eq!(reason, SKIPPED_KIND);
    }

    fn count(db: &Connection) -> usize {
        usize::try_from(
            db.query_row("SELECT count(*) FROM events", [], |r| r.get::<_, i64>(0))
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn nearest_uses_vectors_and_deduplicates_passages() {
        let f = Fixture::new();
        initialize(&f.db(), &config()).unwrap();
        let mut db = connect(&f.db()).unwrap();
        let long = format!("{} lease agreement", "background ".repeat(400));
        f.write(&[(1, &long), (2, "GPU fan curve")]);
        sync_archive(
            &mut db,
            &f.source(),
            &mut |text| {
                Ok(if text.contains("lease") {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                })
            },
            &mut |_| {},
        )
        .unwrap();
        // Controlled vectors isolate retrieval plumbing, not model quality.
        let found = nearest(&db, &[1.0, 0.0], 10, None).unwrap();
        let only_two: std::collections::HashSet<u64> = [2].into();
        let filtered = nearest(&db, &[1.0, 0.0], 10, Some(&only_two)).unwrap();
        assert_eq!(filtered.iter().map(|c| c.id).collect::<Vec<_>>(), vec![2]);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].id, 1);
        assert!(found[0].text.contains("lease"));
        assert!(found[0].start_byte > 0);
        assert_eq!(found[0].similarity, 1.0);
    }
    #[test]
    fn failed_reembedding_preserves_previous_complete_event() {
        let f = Fixture::new();
        initialize(&f.db(), &config()).unwrap();
        let mut db = connect(&f.db()).unwrap();
        f.write(&[(1, "original")]);
        sync_archive(
            &mut db,
            &f.source(),
            &mut |_| Ok(vec![1.0, 0.0]),
            &mut |_| {},
        )
        .unwrap();
        f.write(&[(1, "replacement")]);
        assert!(sync_archive(
            &mut db,
            &f.source(),
            &mut |_| Err("offline".into()),
            &mut |_| {}
        )
        .is_err());
        let hit = nearest(&db, &[1.0, 0.0], 1, None).unwrap().remove(0);
        assert_eq!(hit.digest, digest("original"));
        assert_ne!(hit.digest, digest("replacement")); // source validation rejects this candidate
    }
    #[test]
    fn changed_model_rebuilds_and_dimension_drift_is_rejected() {
        let f = Fixture::new();
        let mut cfg = config();
        initialize(&f.db(), &cfg).unwrap();
        let mut db = connect(&f.db()).unwrap();
        f.write(&[(1, "one")]);
        sync_archive(
            &mut db,
            &f.source(),
            &mut |_| Ok(vec![1.0, 0.0]),
            &mut |_| {},
        )
        .unwrap();
        f.write(&[(1, "one"), (2, "two")]);
        assert!(sync_archive(
            &mut db,
            &f.source(),
            &mut |_| Ok(vec![1.0, 0.0, 0.0]),
            &mut |_| {}
        )
        .is_err());
        assert_eq!(count(&db), 1);
        drop(db);
        cfg.model = "different-model".into();
        assert_eq!(initialize(&f.db(), &cfg).unwrap(), 0);
    }

    fn refuse(reason: &str) -> Box<dyn Error> {
        Box::new(Refused { status: 500, reason: reason.into() })
    }

    /// Text that is many tokens a byte, like tool output: what llama.cpp's
    /// 512-token batch refuses in one 1800-byte chunk.
    fn dense(bytes: usize) -> String {
        "{\"k\":\"0x9f3a\",\"p\":[1,2]}".repeat(bytes / 24 + 1)[..bytes].to_owned()
    }

    #[test]
    fn a_chunk_the_server_refuses_is_embedded_in_halves_and_the_index_goes_on() {
        let f = Fixture::new();
        initialize(&f.db(), &config()).unwrap();
        let mut db = connect(&f.db()).unwrap();
        let big = dense(1800);
        f.write(&[(1, "before"), (2, &big), (3, "after")]);
        // A server whose batch holds 900 bytes of this text, header included.
        let mut inputs = Vec::new();
        let mut refused = Vec::new();
        sync_archive(
            &mut db,
            &f.source(),
            &mut |text| {
                inputs.push(text.to_owned());
                if text.len() > 900 { Err(refuse("input is too large to process. increase the physical batch size")) } else { Ok(vec![1.0, 0.0]) }
            },
            &mut |c| refused.extend(c.refused),
        )
        .unwrap();
        assert_eq!(count(&db), 3, "every event indexed, including the one after");
        assert!(refused.is_empty(), "nothing skipped: the halves fit");
        // Every input carried its header, halves included.
        assert!(inputs.iter().all(|input| input.starts_with("[s0ma_reply · ")), "{inputs:?}");
        // Halving covers the whole chunk and nothing else: the stored pieces
        // run from the start of the event to its end, each inside the budget.
        let pieces: Vec<(i64, i64)> = db
            .prepare("SELECT start_byte,end_byte FROM chunks WHERE event_id=2 ORDER BY ordinal")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert_eq!(pieces.first().unwrap().0, 0);
        assert_eq!(pieces.last().unwrap().1, big.len() as i64);
        for pair in pieces.windows(2) {
            assert!(pair[1].0 <= pair[0].1, "a gap between {:?} and {:?}", pair[0], pair[1]);
        }
        let stored: Vec<i64> = pieces.iter().map(|(start, end)| end - start).collect();
        assert!(stored.iter().all(|len| *len <= CHUNK_BYTES as i64), "{stored:?}");
    }

    #[test]
    fn a_piece_refused_at_every_size_is_skipped_and_said() {
        let f = Fixture::new();
        initialize(&f.db(), &config()).unwrap();
        let mut db = connect(&f.db()).unwrap();
        let poisoned = format!("{}POISON{}", "a".repeat(1000), "b".repeat(700));
        f.write(&[(1, &poisoned), (2, "after")]);
        let mut notes = Vec::new();
        let mut calls = 0;
        sync_archive(
            &mut db,
            &f.source(),
            &mut |text| {
                calls += 1;
                if text.contains("POISON") { Err(refuse("unsupported token")) } else { Ok(vec![1.0, 0.0]) }
            },
            &mut |c| notes.extend(c.refused),
        )
        .unwrap();
        assert_eq!(count(&db), 2, "the event after it is indexed");
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("event #1") && notes[0].contains("unsupported token"), "{}", notes[0]);
        // The rest of that event is still searchable.
        let kept: i64 = db.query_row("SELECT count(*) FROM chunks WHERE event_id=1", [], |r| r.get(0)).unwrap();
        assert_eq!(kept, 3, "1706 bytes: halves of 853, the poisoned one halved again");
        assert!(calls <= 15 + 1, "bounded work for one chunk: {calls}");
        // A later pass doesn't try it again.
        sync_archive(&mut db, &f.source(), &mut |_| panic!("unchanged"), &mut |_| {}).unwrap();
    }

    #[test]
    fn a_server_that_refuses_everything_stops_the_pass_and_skips_nothing() {
        let f = Fixture::new();
        initialize(&f.db(), &config()).unwrap();
        let mut db = connect(&f.db()).unwrap();
        f.write(&[(1, &dense(1800)), (2, "two")]);
        let error = sync_archive(&mut db, &f.source(), &mut |_| Err(refuse("model not loaded")), &mut |_| {}).unwrap_err();
        assert_eq!(
            error.to_string(),
            "the embedding server refuses everything, even a two-letter text (HTTP 500): model not loaded",
            "the reason once, not twice"
        );
        assert_eq!(count(&db), 0, "nothing recorded as done, so it's all retried once the server works");
        // And a server returning unusable vectors for everything is the same.
        let error = sync_archive(&mut db, &f.source(), &mut |_| Ok(vec![f32::NAN]), &mut |_| {}).unwrap_err();
        assert_eq!(
            error.to_string(),
            "the embedding server answers everything, even a two-letter text, with an unusable vector \
             (invalid embedding dimensions or non-finite values)"
        );
        assert_eq!(count(&db), 0);
    }

    /// One HTTP exchange: answers `status` with `body`, returns the address.
    fn serve_once(status: &'static str, body: &'static str) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut size = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                    size = v.trim().parse().unwrap();
                }
            }
            let mut request = vec![0; size];
            reader.read_exact(&mut request).unwrap();
            write!(stream, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        });
        (address, worker)
    }

    #[test]
    fn a_refusal_over_http_carries_the_servers_words_and_an_outage_says_where() {
        let (address, worker) = serve_once(
            "500 Internal Server Error",
            r#"{"error":{"code":500,"message":"input is too large to process. increase the physical batch size","type":"server_error"}}"#,
        );
        let mut cfg = config();
        cfg.url = format!("http://{address}/v1/embeddings");
        let error = embed(&embedding_client().unwrap(), &cfg, "text", false).unwrap_err();
        worker.join().unwrap();
        assert!(error.is::<Refused>());
        assert_eq!(
            error.to_string(),
            "the embedding server refused it (HTTP 500): input is too large to process. increase the physical batch size"
        );

        // Nothing listening: not a refusal, so a pass stops and keeps the backlog.
        let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        cfg.url = format!("http://{closed}/v1/embeddings");
        let error = embed(&embedding_client().unwrap(), &cfg, "text", false).unwrap_err();
        assert!(!error.is::<Refused>());
        assert!(error.to_string().starts_with(&format!("could not connect to the embedding server at {closed}")), "{error}");
    }
}

