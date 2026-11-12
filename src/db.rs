//! Opening the SQLite stores: one way, so their settings can't drift apart.

use std::path::Path;
use std::time::Duration;

use rusqlite::Connection;

/// A store's connection: waits up to 10 s for a lock, writes ahead to a log
/// (readers never block the writer) and syncs fully, so a crash loses
/// nothing it reported written.
pub fn open(path: impl AsRef<Path>) -> rusqlite::Result<Connection> {
    let db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(10))?;
    db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")?;
    Ok(db)
}
