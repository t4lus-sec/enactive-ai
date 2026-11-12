//! What happens when a thread panics (`install`).
//!
//! - The main thread: the agent ends. The terminal is put back first, whatever was
//!   held when it happened, so the message is printed where it can be read
//!   and the shell gets a working terminal.
//! - The keyboard's thread during a turn: caught there and reported in the
//!   conversation (`input::note_key_panic`).
//! - Any other thread (a connection, a sync, a subagent's worker): its
//!   message and where it happened are appended to `panics.log` beside the
//!   archive, and a line on the page says so. It is not printed: the screen
//!   is the agent's, and a message written over it would be lost in the next
//!   frame and garble this one. The hook takes no lock that a panicking
//!   thread might hold; the line is said by a watcher thread of its own.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

static LOG: OnceLock<PathBuf> = OnceLock::new();
/// Panics of other threads not yet said on the page.
static UNSAID: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The panic hook, once, with the log beside the archive.
pub fn install(archive: &Path) {
    if LOG.set(archive.with_file_name("panics.log")).is_err() {
        return;
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| match std::thread::current().name() {
        Some("main") => {
            crate::terminal::emergency_restore();
            previous(info);
        }
        Some(crate::input::KEYS_THREAD) => crate::input::note_key_panic(info),
        // Under test the harness reads a failing test's panic as printed.
        _ if cfg!(test) => previous(info),
        name => record(name.unwrap_or("unnamed"), &describe(info)),
    }));
    let _ = std::thread::Builder::new().name("panic-reporter".into()).spawn(|| loop {
        std::thread::sleep(Duration::from_millis(500));
        let unsaid = std::mem::take(&mut *crate::util::locked(&UNSAID));
        for line in unsaid {
            crate::terminal::warn_status(&line);
        }
    });
}

/// The panic's message, and where it happened.
fn describe(info: &std::panic::PanicHookInfo<'_>) -> String {
    let what = info
        .payload()
        .downcast_ref::<&str>()
        .map(|text| text.to_string())
        .or_else(|| info.payload().downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".into());
    match info.location() {
        Some(at) => format!("{what}, at {}:{}", at.file(), at.line()),
        None => what,
    }
}

/// Another thread's panic: logged, and queued to be said. Never waits on a
/// lock: one held by the panicking thread would never come free.
fn record(thread: &str, what: &str) {
    let at = chrono::Local::now().to_rfc3339();
    let log = LOG.get();
    if let Some(path) = log
        && let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path)
    {
        let backtrace = std::backtrace::Backtrace::capture();
        let _ = writeln!(file, "{at} thread {thread}: {what}\n{backtrace}\n");
    }
    let place = log.map(|path| format!(" (details in {})", path.display())).unwrap_or_default();
    if let Ok(mut unsaid) = UNSAID.try_lock() {
        unsaid.push(format!("[panic] the {thread} thread stopped: {what}{place}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn another_threads_panic_is_logged_and_queued_not_printed() {
        let dir = std::env::temp_dir().join(format!("s0ma-crash-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let log = dir.join("panics.log");
        let _ = LOG.set(log.clone());
        if LOG.get() != Some(&log) {
            return; // another test installed it first: nothing to check here
        }
        record("share-sync", "it broke, at src/share_sync.rs:12");
        let text = std::fs::read_to_string(&log).unwrap();
        assert!(text.contains("thread share-sync: it broke, at src/share_sync.rs:12"), "{text}");
        let unsaid = std::mem::take(&mut *crate::util::locked(&UNSAID));
        assert!(unsaid.iter().any(|line| line.starts_with("[panic] the share-sync thread stopped: it broke") && line.contains("panics.log")), "{unsaid:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
