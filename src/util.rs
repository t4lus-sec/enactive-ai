//! Small helpers several modules need, each written once: a lock that
//! survives a panic, one shell word, a text clipped, a quick hash, and a file
//! replaced whole.

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

/// `mutex`, locked even when a thread panicked holding it: what it guards is
/// still the best there is, and one panic should not take every later caller
/// down with it.
pub fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `text` as one word for a POSIX shell: in single quotes, each single quote
/// closed, escaped and reopened (`'it'\''s'`). Nothing inside is expanded.
pub fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
}

/// `text` cut to its first `chars` characters, with `…` after when anything
/// was cut. (Not a display fit: the result can be one longer than `chars`;
/// `tabs` fits text into columns its own way.)
pub fn clip(text: &str, chars: usize) -> String {
    match text.char_indices().nth(chars) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_owned(),
    }
}

/// FNV-1a, 64 bits: a quick fingerprint, not a cryptographic one.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3))
}

/// `path` replaced whole by `contents`, never left half written: written to a
/// file of its own beside it (named for this process and this write, so two
/// writers at once never share one), flushed to the disk, renamed over
/// `path`, and the folder flushed so the rename lasts too. On an error the
/// file beside it is removed and `path` is as it was.
pub fn write_atomic(path: &Path, contents: impl AsRef<[u8]>) -> std::io::Result<()> {
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    let beside = path.with_file_name(format!(".{name}.{}-{}.tmp", std::process::id(), WRITES.fetch_add(1, Ordering::SeqCst)));
    let written = (|| {
        let mut file = std::fs::File::create(&beside)?;
        file.write_all(contents.as_ref())?;
        file.sync_all()?;
        std::fs::rename(&beside, path)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&beside);
        return written;
    }
    if let Some(folder) = path.parent().filter(|folder| !folder.as_os_str().is_empty()) {
        let _ = std::fs::File::open(folder).and_then(|folder| folder.sync_all());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shell_word_is_one_word_whatever_is_in_it() {
        assert_eq!(shell_quote("plain"), "'plain'");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
        assert_eq!(shell_quote(""), "''");
        // Run through a real shell: what comes out is what went in.
        for text in ["it's", "$(echo no)", "`echo no`", "a b\tc", "line\nbreak", "\\", "'", "*", "-n"] {
            let out = std::process::Command::new("sh").arg("-c").arg(format!("printf %s {}", shell_quote(text))).output().unwrap();
            assert_eq!(String::from_utf8(out.stdout).unwrap(), text, "{text:?}");
        }
    }

    #[test]
    fn clipping_counts_characters_and_says_when_it_cut() {
        assert_eq!(clip("short", 10), "short");
        assert_eq!(clip("exactly", 7), "exactly");
        assert_eq!(clip("longer text", 6), "longer…");
        assert_eq!(clip("日本語のテキスト", 3), "日本語…", "characters, not bytes");
        assert_eq!(clip("", 0), "");
    }

    #[test]
    fn the_hash_is_fnv1a() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn a_file_is_replaced_whole_and_nothing_is_left_beside_it() {
        let dir = std::env::temp_dir().join(format!("s0ma-util-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");
        write_atomic(&path, "one").unwrap();
        write_atomic(&path, b"two").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "two");
        let left: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|entry| entry.file_name()).collect();
        assert_eq!(left, vec![std::ffi::OsString::from("state.json")], "nothing beside it");
        // A folder that is not there: an error, and nothing made.
        assert!(write_atomic(&dir.join("missing/state.json"), "x").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_lock_a_panic_left_poisoned_still_opens() {
        let mutex = std::sync::Arc::new(Mutex::new(1));
        let held = std::sync::Arc::clone(&mutex);
        let _ = std::thread::spawn(move || {
            let _guard = held.lock().unwrap();
            panic!("poisoning it on purpose");
        })
        .join();
        assert!(mutex.is_poisoned());
        *locked(&mutex) += 1;
        assert_eq!(*locked(&mutex), 2);
    }
}
