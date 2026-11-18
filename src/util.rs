//! Small helpers several modules need, each written once: a lock that
//! survives a panic, one shell word, a text clipped, a quick hash, a file
//! replaced whole, and a temporary folder of this process's own.

use std::io::Write;
use std::path::{Path, PathBuf};
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
    write_whole(path, contents.as_ref(), None)
}

/// `write_atomic` for a file only its owner may read (mode 0600), such as one
/// holding a key: the file beside is made with that mode, so the contents are
/// never readable by anyone else, even for a moment.
pub fn write_private(path: &Path, contents: impl AsRef<[u8]>) -> std::io::Result<()> {
    write_whole(path, contents.as_ref(), Some(0o600))
}

fn write_whole(path: &Path, contents: &[u8], mode: Option<u32>) -> std::io::Result<()> {
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    let beside = path.with_file_name(format!(".{name}.{}-{}.tmp", std::process::id(), WRITES.fetch_add(1, Ordering::SeqCst)));
    let written = (|| {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        if let Some(mode) = mode {
            std::os::unix::fs::OpenOptionsExt::mode(&mut options, mode);
        }
        let mut file = options.open(&beside)?;
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

static PRIVATE_TEMP: Mutex<Option<PathBuf>> = Mutex::new(None);

/// A folder in the temporary directory for what the operator is writing or
/// moving (a draft in the editor, a shared file being edited or carried):
/// made once, under a name nobody can guess, for its owner alone (0700), so
/// nothing in it is read or put in its place by another user of the machine.
pub fn private_temp() -> std::io::Result<PathBuf> {
    use std::os::unix::fs::DirBuilderExt;
    let mut made = locked(&PRIVATE_TEMP);
    if let Some(dir) = made.as_ref().filter(|dir| dir.is_dir()) {
        return Ok(dir.clone());
    }
    let mut last = None;
    for _ in 0..8 {
        let mut bytes = [0u8; 8];
        std::fs::File::open("/dev/urandom").and_then(|mut file| std::io::Read::read_exact(&mut file, &mut bytes))?;
        let tag: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
        let dir = std::env::temp_dir().join(format!("s0ma-{}-{tag}", std::process::id()));
        // Made new, never an existing folder taken: that one could be another's.
        match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => {
                *made = Some(dir.clone());
                return Ok(dir);
            }
            Err(error) => last = Some(error),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("no temporary folder")))
}

/// `private_temp`'s folder removed, as the process ends.
pub struct PrivateTemp;

impl Drop for PrivateTemp {
    fn drop(&mut self) {
        if let Some(dir) = locked(&PRIVATE_TEMP).take() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
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
}
