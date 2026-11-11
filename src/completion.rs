//! Tab completion shared by the idle Reedline prompt and the steering pane:
//! slash commands, their subcommands, `@file` mentions and `/img` paths.

use std::path::{Path, PathBuf};

pub struct Command {
    pub name: &'static str,
    pub description: &'static str,
    /// Second-word completions.
    pub args: &'static [&'static str],
    /// The command expects more input after its name.
    pub takes_args: bool,
}

const fn cmd(
    name: &'static str,
    description: &'static str,
    args: &'static [&'static str],
    takes_args: bool,
) -> Command {
    Command {
        name,
        description,
        args,
        takes_args,
    }
}

const WEAKNESS_ARGS: &[&str] = &[
    "help", "list", "show", "confirm", "resolve", "reject", "merge", "reopen", "verify",
    "export",
];

/// Every operator command `App::handle_command` (and the modules it
/// delegates to) understands. Keep in step when adding commands.
pub const COMMANDS: &[Command] = &[
    cmd("/approval", "every command in every shell, and every subagent, waits for you (alt+a twice approves, alt+x declines)", &["on", "off", "status", "list"], true),
    cmd("/board", "list or manage boards", &["list", "all", "show", "hint", "unhint", "extend", "confirm", "reject", "exhaust", "abandon", "reuse"], true),
    cmd("/compact", "rotate the working context now", &[], false),
    cmd("/copy", "copy the last reply (or its last code block) to the clipboard", &["code"], false),
    cmd("/download", "copy files from the shared folder to this computer's ~/Downloads: /download <name> …", &[], true),
    cmd("/goal", "an objective s0ma works toward turn after turn until it is done", &["status", "pause", "resume", "clear", "help"], true),
    cmd("/help", "list these commands and the keyboard shortcuts", &[], false),
    cmd("/job", "work whose record is removed when it ends", &["start", "pause", "resume", "close", "list", "show", "help"], true),
    cmd("/redact", "the memory tab at its redactions: what s0ma asked to remove", &["view", "list", "help"], true),
    cmd("/hint", "leave a note on a board: it stays, and s0ma sees it on every read", &[], true),
    cmd("/identity", "show the identity text in context", &[], false),
    cmd("/img", "send an image: /img <path> <message>", &[], true),
    cmd("/kill", "stop the running tool call, or a subagent (list, all, subagents, <slot>)", &["list", "all", "subagents"], true),
    cmd("/loop", "autonomous looping mode", &["on", "off", "status"], true),
    cmd("/loops", "list open loops", &["open", "done", "dropped", "all"], true),
    cmd("/mcp", "MCP servers: status, tools, reload", &["status", "tools", "reload"], true),
    cmd("/mode", "reasoning profile; /mode workers <level> for subagents", &["auto", "instruct", "low", "medium", "xhigh", "workers"], true),
    cmd("/shell", "shells: list, new, close, mode", &["list", "new", "close", "mode"], true),
    cmd("/now", "while s0ma works: send this now, not after its current step (or ctrl+s)", &[], true),
    cmd("/pair", "confirm s0ma's commands everywhere", &["on", "off"], true),
    cmd("/quit", "stop s0ma", &[], false),
    cmd("/quit!", "stop s0ma even with terminals attached", &[], false),
    cmd("/share", "copy files to /share on s0ma's machine: /share <path> …", &[], true),
    cmd("/sleep", "consolidate memory and reset context; /sleep set 0200 runs one daily at 2 am (off to stop)", &["set"], false),
    cmd("/tokens", "show context usage", &[], false),
    cmd("/unhint", "take a note off a board: /unhint <board> h2", &[], true),
    cmd("/tz", "your time zone, shown beside the host's: /tz UTC-6, /tz America/Denver, /tz off", &["off"], true),
    cmd("/upload", "copy files from this computer into the shared folder: /upload <path> … [--to <folder>]", &[], true),
    cmd("/voice", "voice input and output", &["on", "off", "status"], true),
    cmd("/backup", "back up the shared folder now (it is backed up every 6h to ~/s0ma/backups)", &["now", "status"], true),
    cmd("/vm", "s0ma's access to its machine (off: its commands, shells, desktop, browser and /share files are refused)", &["on", "off", "status"], true),
    cmd("/weakness", "review weakness reports", WEAKNESS_ARGS, true),
    cmd("/weaknesses", "review weakness reports", WEAKNESS_ARGS, true),
];

/// The keyboard shortcuts, for `/help`, grouped by when they apply. They used
/// to sit in the bottom-right corner, which now shows the rates. Keep in step
/// with input.rs: the idle prompt's bindings and the steering pane's keys.
pub const SHORTCUTS: &[(&str, &[(&str, &str)])] = &[
    (
        "while s0ma works",
        &[
            ("ctrl+c", "stop what it is doing"),
            ("ctrl+s, ctrl+enter", "send the draft now, not after its current step (as /now)"),
            ("ctrl+d", "stop, and leave"),
        ],
    ),
    (
        "at the prompt",
        &[
            ("ctrl+c", "clear the line"),
            ("ctrl+d", "leave: quit, or detach from a running s0ma"),
            ("alt+enter", "new line"),
        ],
    ),
    (
        "either way",
        &[
            ("ctrl+o", "collapse or expand reasoning and tool calls; in a shell, each command to one line"),
            ("pgup, wheel", "the conversation further back; esc comes back"),
            ("ctrl+l", "draw the page again"),
            ("alt+a twice, alt+x", "approval mode: approve or decline the oldest request waiting, here or in any tab (/approval)"),
            ("shift+drag", "select text: s0ma has the mouse for the wheel"),
            ("ctrl+g", "edit the draft in $VISUAL or $EDITOR"),
            ("tab", "complete commands, their arguments and @file paths"),
            ("ctrl+a, ctrl+e", "start, end of the line"),
            ("ctrl+u, ctrl+k", "delete to the start, to the end"),
            ("ctrl+w", "delete the word before"),
            ("alt+left, alt+right", "back, forward a word (alt+f forward too); with two panes, the pane"),
        ],
    ),
    (
        "tabs",
        &[
            ("alt+2-9, alt+0", "a subagent's tab: what it is doing, live"),
            ("alt+p", "projects: open, review, verification, complete; tab and shift+tab switch them, ←→ a project's sections (board, reports, files, notes; ⏎ folds the board), ctrl+n and ctrl+p the boards"),
            ("alt+b", "back to the tab before, as it was left"),
            ("alt+m", "memory: search the archive, threads, world model, redactions; mark text to strike"),
            ("alt+o", "overview: s0ma's settings, goal, jobs, where its commands run, what it left running, and MCP servers; change them with keys"),
            ("alt+s", "share: /share on s0ma's machine, the folder you both work in; view, edit, make files (n) and folders (f), add, save a copy, delete (x twice; a folder with all in it)"),
            ("alt+e", "the editor: files of /share opened with e in the share tab; ^s save, ^f find, ^g go to line, ^p markdown: live preview, then split, then text; alt+z wrap, ^q close"),
            ("alt+shift+1-9, 0", "shells"),
            ("alt+n", "new shell (in a shell tab)"),
            ("ctrl+\\", "take or hand back control of a shell"),
            ("ctrl+q", "close a shell, or a finished subagent's tab"),
            ("alt+\\", "two panes side by side, each any tab (s0ma's page among them); again for one (ctrl+shift+p too)"),
            ("alt+← alt+→", "with two panes: the left one or the right one has the keyboard (a click in it, or ctrl+1, ctrl+2, too)"),
            ("alt+1, esc", "back to s0ma"),
            ("enter", "in a subagent's tab, a direction to it (/wrap [text] and /kill too); on the board, a hint"),
            ("ctrl+t", "on the board, what enter adds: a hint, a lead of yours (under the selected finding), or a finding (supports, rules out) on the selected lead or finding"),
            ("ctrl+c", "on the board, grey a row out to the bottom of its siblings, or back; for your reading only, and a hint on it brings its branch back"),
            ("pgup, pgdn", "scroll a tab (up, down by a row); end follows it again"),
        ],
    ),
];

/// What `/help` prints: every command, then the keys.
pub fn help_text() -> String {
    let width = COMMANDS.iter().map(|command| command.name.len()).max().unwrap_or(0);
    let mut text = String::from("[help] commands\n");
    for command in COMMANDS {
        text.push_str(&format!("  {:width$}  {}\n", command.name, command.description));
    }
    let key_width = SHORTCUTS
        .iter()
        .flat_map(|(_, keys)| keys.iter())
        .map(|(key, _)| key.len())
        .max()
        .unwrap_or(0);
    text.push_str("\n[help] keys");
    for (when, keys) in SHORTCUTS {
        text.push_str(&format!("\n  {when}\n"));
        for (key, what) in *keys {
            text.push_str(&format!("    {key:key_width$}  {what}\n"));
        }
    }
    text.truncate(text.trim_end().len());
    text
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub value: String,
    pub description: Option<String>,
    /// Add a space after inserting (false for directories and partial paths).
    pub append_space: bool,
}

/// Replace `line[start..end]` with one of `candidates`.
#[derive(Debug, Default)]
pub struct Completion {
    pub start: usize,
    pub end: usize,
    pub candidates: Vec<Candidate>,
}

const MAX_PATH_CANDIDATES: usize = 200;

pub fn complete(line: &str, pos: usize) -> Completion {
    let pos = pos.min(line.len());
    let before = &line[..pos];
    let start = token_start(before);
    let token = &before[start..];
    let earlier: Vec<&str> = before[..start].split_whitespace().collect();

    let candidates = if let Some(path) = token.strip_prefix('@') {
        paths(path)
            .into_iter()
            .map(|mut c| {
                c.value.insert(0, '@');
                c
            })
            .collect()
    } else if earlier.is_empty() && token.starts_with('/') {
        COMMANDS
            .iter()
            .filter(|c| c.name.starts_with(token))
            .map(|c| Candidate {
                value: c.name.to_string(),
                description: Some(c.description.to_string()),
                append_space: c.takes_args,
            })
            .collect()
    } else if earlier.len() == 1 && earlier[0] == "/img" {
        paths(token)
    } else if matches!(earlier.first(), Some(&"/share") | Some(&"/upload")) {
        // Every argument of /share and /upload is a path here.
        paths(token)
    } else if earlier.len() == 1 {
        COMMANDS
            .iter()
            .find(|c| c.name == earlier[0])
            .map(|c| {
                c.args
                    .iter()
                    .filter(|a| a.starts_with(token))
                    .map(|a| Candidate {
                        value: a.to_string(),
                        description: None,
                        append_space: true,
                    })
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    Completion {
        start,
        end: pos,
        candidates,
    }
}

/// Start of the whitespace-delimited token ending at the cursor. A
/// backslash-escaped space (`my\ file.txt`) stays inside the token.
pub(crate) fn token_start(before: &str) -> usize {
    let chars: Vec<(usize, char)> = before.char_indices().collect();
    let mut i = chars.len();
    while i > 0 {
        let (index, c) = chars[i - 1];
        if c.is_whitespace() && !(i >= 2 && chars[i - 2].1 == '\\') {
            return index + c.len_utf8();
        }
        i -= 1;
    }
    0
}

pub fn longest_common_prefix(values: &[&str]) -> String {
    let Some(first) = values.first() else {
        return String::new();
    };
    let mut end = first.len();
    for value in &values[1..] {
        end = first
            .char_indices()
            .zip(value.chars())
            .take_while(|((_, a), b)| a == b)
            .last()
            .map_or(0, |((i, a), _)| i + a.len_utf8())
            .min(end);
    }
    first[..end].to_string()
}

/// Escape spaces so a completed path stays one token.
pub fn escape_path(path: &str) -> String {
    path.replace(' ', "\\ ")
}

pub fn unescape_path(path: &str) -> String {
    path.replace("\\ ", " ")
}

/// Expand a leading `~/` and resolve against the working directory.
pub fn resolve_path(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME") {
            return Path::new(&home).join(rest);
        }
    PathBuf::from(path)
}

/// Directory entries completing `typed` (as typed, possibly escaped).
pub(crate) fn paths(typed: &str) -> Vec<Candidate> {
    let unescaped = unescape_path(typed);
    let (dir_text, file_prefix) = match unescaped.rfind('/') {
        Some(i) => (&unescaped[..=i], &unescaped[i + 1..]),
        None => ("", unescaped.as_str()),
    };
    let dir = if dir_text.is_empty() {
        PathBuf::from(".")
    } else {
        resolve_path(dir_text)
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };

    let mut found: Vec<(String, bool)> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            if !name.starts_with(file_prefix) {
                return None;
            }
            if name.starts_with('.') && !file_prefix.starts_with('.') {
                return None;
            }
            let is_dir = entry.path().is_dir();
            Some((name, is_dir))
        })
        .collect();
    found.sort();
    found.truncate(MAX_PATH_CANDIDATES);

    found
        .into_iter()
        .map(|(name, is_dir)| {
            let mut value = escape_path(&format!("{dir_text}{name}"));
            if is_dir {
                value.push('/');
            }
            Candidate {
                value,
                description: None,
                append_space: !is_dir,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_argument_of_send_completes_as_a_path() {
        let dir = std::env::temp_dir().join(format!("s0ma-send-complete-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("report.pdf"), b"x").unwrap();
        let prefix = format!("{}/rep", dir.display());
        for line in [format!("/share {prefix}"), format!("/share first.txt {prefix}")] {
            let completion = super::complete(&line, line.len());
            assert!(
                completion.candidates.iter().any(|candidate| candidate.value.ends_with("report.pdf")),
                "{line}: {:?}",
                completion.candidates
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn help_lists_every_command_and_the_keys_that_left_the_corner() {
        let help = super::help_text();
        for command in super::COMMANDS {
            assert!(help.contains(command.name), "{} missing from /help", command.name);
        }
        assert!(help.contains("/help"), "it lists itself, so completion offers it");
        for key in ["ctrl+c", "ctrl+s", "ctrl+o", "ctrl+g", "ctrl+d", "tab", "alt+1", "alt+p", "alt+b", "alt+m", "alt+s", "alt+o"] {
            assert!(help.contains(key), "{key} missing from /help");
        }
        assert!(!help.ends_with('\n'));
    }

    use super::*;

    fn values(c: &Completion) -> Vec<&str> {
        c.candidates.iter().map(|c| c.value.as_str()).collect()
    }

    #[test]
    fn commands_complete_from_the_start_of_the_line() {
        let c = complete("/lo", 3);
        assert_eq!(values(&c), ["/loop", "/loops"]);
        assert_eq!((c.start, c.end), (0, 3));
        // Not a command when it isn't the first word.
        assert!(complete("see /lo", 7).candidates.is_empty());
    }

    #[test]
    fn subcommands_complete_after_the_command() {
        let c = complete("/mode m", 7);
        assert_eq!(values(&c), ["medium"]);
        assert_eq!((c.start, c.end), (6, 7));
        assert_eq!(values(&complete("/loop ", 6)), ["on", "off", "status"]);
    }

    #[test]
    fn file_mentions_complete_paths() {
        let dir = std::env::temp_dir().join(format!("s0ma-complete-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub dir")).unwrap();
        std::fs::write(dir.join("notes.md"), "x").unwrap();
        std::fs::write(dir.join(".hidden"), "x").unwrap();
        let base = dir.to_str().unwrap();

        let line = format!("look at @{base}/");
        let c = complete(&line, line.len());
        assert_eq!(
            values(&c),
            [format!("@{base}/notes.md"), format!("@{base}/sub\\ dir/")]
        );
        assert!(c.candidates[0].append_space);
        assert!(!c.candidates[1].append_space);
        assert_eq!(c.start, 8);

        // Escaped spaces keep the token together.
        let line = format!("@{base}/sub\\ d");
        assert_eq!(values(&complete(&line, line.len())), [format!("@{base}/sub\\ dir/")]);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn common_prefix() {
        assert_eq!(longest_common_prefix(&["/loop", "/loops"]), "/loop");
        assert_eq!(longest_common_prefix(&["/board", "/loop"]), "/");
        assert_eq!(longest_common_prefix(&["é1", "é2"]), "é");
    }
}
