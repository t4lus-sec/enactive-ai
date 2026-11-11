//! `@path` mentions in operator input.
//!
//! The transcript keeps the message as typed; the model receives it with
//! each mentioned file's contents (or a directory's listing) appended.
//! Anything that isn't an existing path — `@someone`, an email address —
//! is left alone.

use crate::completion::{resolve_path, unescape_path};

use std::fmt::Write as _;
use std::path::Path;

const MAX_FILE_BYTES: usize = 256 * 1024;
const MAX_TOTAL_BYTES: usize = 1024 * 1024;
const MAX_DIR_ENTRIES: usize = 200;
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "gif", "bmp", "webp", "tga"];

pub struct Expanded {
    /// What the model receives.
    pub content: String,
    /// One line per mention, for the operator.
    pub notes: Vec<String>,
}

pub fn expand(text: &str) -> Expanded {
    let trimmed = text.trim_start();
    // Commands are matched literally; appending to them would break them.
    // `/img <path> <caption>` is the exception: its caption is a message.
    if trimmed.starts_with('/') && !trimmed.starts_with("/img ") {
        return Expanded {
            content: text.to_string(),
            notes: Vec::new(),
        };
    }

    let mut blocks = String::new();
    let mut notes = Vec::new();
    let mut seen = Vec::new();
    let mut total = 0;

    for mention in mentions(text) {
        if seen.contains(&mention) {
            continue;
        }
        seen.push(mention.clone());
        let path = resolve_path(&mention);

        if path.is_dir() {
            let listing = list_dir(&path);
            notes.push(format!("[attach] {mention} (directory, {} entries)", listing.1));
            let _ = write!(
                blocks,
                "\n\n<attached_directory path=\"{mention}\">\n{}</attached_directory>",
                listing.0
            );
            continue;
        }

        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                notes.push(format!("[attach] {mention}: not attached ({error})"));
                continue;
            }
        };

        let is_image = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()));
        let text = (!bytes[..bytes.len().min(8192)].contains(&0))
            .then(|| String::from_utf8(bytes.clone()).ok())
            .flatten();
        let Some(mut text) = text else {
            notes.push(if is_image {
                format!("[attach] {mention} is an image; send it with /img {mention} <message>")
            } else {
                format!("[attach] {mention}: binary file, not attached")
            });
            continue;
        };

        if total >= MAX_TOTAL_BYTES {
            notes.push(format!("[attach] {mention}: skipped, attachments are over 1 MiB"));
            continue;
        }
        let budget = MAX_FILE_BYTES.min(MAX_TOTAL_BYTES - total);
        let truncated = text.len() > budget;
        if truncated {
            let mut end = budget;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
        }
        total += text.len();

        let lines = text.lines().count();
        notes.push(format!(
            "[attach] {mention} ({lines} lines{})",
            if truncated { ", truncated" } else { "" }
        ));
        let _ = write!(
            blocks,
            "\n\n<attached_file path=\"{mention}\"{}>\n{text}{}</attached_file>",
            if truncated { " truncated=\"true\"" } else { "" },
            if text.ends_with('\n') { "" } else { "\n" },
        );
    }

    Expanded {
        content: format!("{text}{blocks}"),
        notes,
    }
}

/// Paths mentioned as `@path` (start of input or after whitespace) that
/// exist. Trailing punctuation is dropped if the path only exists without
/// it, so "see @notes.md." works.
fn mentions(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut found = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let at_boundary = i == 0 || chars[i - 1].is_whitespace();
        if chars[i] != '@' || !at_boundary {
            i += 1;
            continue;
        }
        let mut token = String::new();
        let mut j = i + 1;
        while j < chars.len() {
            match chars[j] {
                '\\' if chars.get(j + 1) == Some(&' ') => {
                    token.push_str("\\ ");
                    j += 2;
                }
                c if c.is_whitespace() => break,
                c => {
                    token.push(c);
                    j += 1;
                }
            }
        }
        i = j;

        let mut candidate = unescape_path(&token);
        loop {
            if !candidate.is_empty() && resolve_path(&candidate).exists() {
                found.push(candidate);
                break;
            }
            match candidate.chars().last() {
                Some(c) if ".,;:!?)]}'\"".contains(c) => {
                    candidate.pop();
                }
                _ => break,
            }
        }
    }
    found
}

fn list_dir(path: &Path) -> (String, usize) {
    let Ok(entries) = std::fs::read_dir(path) else {
        return ("(unreadable)\n".into(), 0);
    };
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .map(|entry| {
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if entry.path().is_dir() {
                name.push('/');
            }
            name
        })
        .collect();
    names.sort();
    let count = names.len();
    let mut out = String::new();
    for name in names.iter().take(MAX_DIR_ENTRIES) {
        out.push_str(name);
        out.push('\n');
    }
    if count > MAX_DIR_ENTRIES {
        let _ = writeln!(out, "... and {} more", count - MAX_DIR_ENTRIES);
    }
    (out, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("s0ma-attach-{}-{name}", std::process::id()));
        std::fs::create_dir_all(dir.join("a dir")).unwrap();
        std::fs::write(dir.join("notes.md"), "line one\nline two\n").unwrap();
        std::fs::write(dir.join("pic.png"), [0x89, b'P', b'N', b'G', 0, 0]).unwrap();
        dir
    }

    #[test]
    fn mentioned_files_are_appended() {
        let dir = fixture("files");
        let base = dir.to_str().unwrap();
        let input = format!("summarize @{base}/notes.md. thanks, mail me@example.com");
        let out = expand(&input);
        assert!(out.content.starts_with(&input));
        assert!(out.content.contains("line one\nline two\n</attached_file>"));
        assert_eq!(out.notes.len(), 1);
        assert!(out.notes[0].contains("(2 lines)"));
    }

    #[test]
    fn directories_images_and_non_paths() {
        let dir = fixture("mixed");
        let base = dir.to_str().unwrap();
        let out = expand(&format!("@{base}/a\\ dir @{base}/pic.png @nobody"));
        assert!(out.content.contains("<attached_directory"));
        assert!(out.notes.iter().any(|n| n.contains("send it with /img")));
        assert_eq!(out.notes.len(), 2);
    }

    #[test]
    fn commands_are_left_alone() {
        let dir = fixture("commands");
        let line = format!("/board show @{}/notes.md", dir.to_str().unwrap());
        assert_eq!(expand(&line).content, line);
    }
}
