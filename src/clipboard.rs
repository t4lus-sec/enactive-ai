//! The operator's clipboard: `/copy`, and the editor's Ctrl+C, Ctrl+X and
//! Ctrl+V.
//!
//! Copying goes every way that might reach it, since none is sure to:
//! - **OSC 52** asks the terminal itself to set the clipboard, which also
//!   works over SSH. Not every terminal honours it (VTE-based ones such as
//!   GNOME Terminal don't). Inside tmux it is sent plain, for tmux to take
//!   (`set-clipboard on`), and wrapped, for a terminal past it
//!   (`allow-passthrough on`).
//! - **tmux's own buffer** (`load-buffer -w`, which tmux 3.2+ also hands to
//!   the outer terminal's clipboard).
//! - **The desktop's clipboard tool** on a local session (`wl-copy`, `xclip`,
//!   `xsel`, `pbcopy`). A session started before the desktop, or kept in
//!   tmux, can lack `DISPLAY` and `WAYLAND_DISPLAY`: the display is then
//!   found from its socket.
//!
//! Reading back (Ctrl+V) takes the desktop's clipboard on a local session,
//! else tmux's buffer; a terminal's own paste works anywhere.

use base64::Engine;

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Past this, some terminals drop an OSC 52 copy.
const OSC52_SAFE_BYTES: usize = 100_000;

/// Copy `text` every way there is; what was tried, for the operator.
pub fn copy(text: &str) -> String {
    let mut methods: Vec<&str> = Vec::new();

    // Tests never touch the clipboard of whoever runs them.
    if !cfg!(test)
        && let Some(display) = local_display()
        && let Some(tool) = copy_with_tool(text, &display)
    {
        methods.push(tool);
    }
    if !cfg!(test) && std::env::var_os("TMUX").is_some() && tmux_load(text) {
        methods.push("tmux");
    }

    let encoded = base64::engine::general_purpose::STANDARD.encode(text);
    let plain = format!("\x1b]52;c;{encoded}\x07");
    let mut sequence = plain.clone();
    if std::env::var_os("TMUX").is_some() {
        sequence.push_str(&crate::notify::passthrough(&plain));
    }
    let mut stdout = std::io::stdout().lock();
    if stdout.write_all(sequence.as_bytes()).and_then(|_| stdout.flush()).is_ok() {
        methods.push("OSC 52");
    }

    let chars = text.chars().count();
    let lines = text.lines().count().max(1);
    let mut said = format!(
        "[copy] {chars} characters, {lines} line{} via {}",
        if lines == 1 { "" } else { "s" },
        methods.join(" + ")
    );
    if methods == ["OSC 52"] {
        if std::env::var_os("VTE_VERSION").is_some() {
            said.push_str(" (this terminal ignores OSC 52: select with the mouse, or use /copy from a terminal that takes it)");
        } else if text.len() > OSC52_SAFE_BYTES {
            said.push_str(" (large: some terminals drop an OSC 52 copy past ~100 KB)");
        }
    }
    said
}

/// What the clipboard holds, and where it came from: the desktop's, on a
/// local session, else tmux's buffer. `None` when neither can be read.
pub fn read() -> Option<(String, &'static str)> {
    if cfg!(test) {
        return None;
    }
    if let Some(display) = local_display() {
        let tools: &[(&str, &[&str])] = if cfg!(target_os = "macos") {
            &[("pbpaste", &[])]
        } else {
            &[
                ("wl-paste", &["--no-newline"]),
                ("xclip", &["-selection", "clipboard", "-o"]),
                ("xsel", &["--clipboard", "--output"]),
            ]
        };
        for (program, args) in tools {
            if program.starts_with("wl-") && !display.iter().any(|(k, _)| *k == "WAYLAND_DISPLAY") && std::env::var_os("WAYLAND_DISPLAY").is_none() {
                continue;
            }
            if let Some(text) = run_reading(program, args, &display) {
                return Some((text, program));
            }
        }
    }
    if std::env::var_os("TMUX").is_some()
        && let Some(text) = run_reading("tmux", &["save-buffer", "-"], &[])
    {
        return Some((text, "tmux"));
    }
    None
}

/// A local desktop session, and what the clipboard tools need set to reach
/// it (empty when the environment has it already). `None` over SSH, or with
/// no desktop to be found.
fn local_display() -> Option<Vec<(&'static str, String)>> {
    if std::env::var_os("SSH_CONNECTION").is_some() || std::env::var_os("SSH_TTY").is_some() {
        return None;
    }
    if cfg!(target_os = "macos")
        || std::env::var_os("WAYLAND_DISPLAY").is_some()
        || std::env::var_os("DISPLAY").is_some()
    {
        return Some(Vec::new());
    }
    let mut found = Vec::new();
    // Wayland: a socket in the user's runtime folder.
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(format!("/run/user/{}", unsafe { libc::getuid() })));
    if let Ok(entries) = std::fs::read_dir(&runtime) {
        let mut sockets: Vec<String> = entries
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with("wayland-") && !name.ends_with(".lock"))
            .collect();
        sockets.sort();
        if let Some(socket) = sockets.into_iter().next() {
            found.push(("WAYLAND_DISPLAY", socket));
            found.push(("XDG_RUNTIME_DIR", runtime.to_string_lossy().into_owned()));
        }
    }
    // X: the lowest display with a socket.
    if let Ok(entries) = std::fs::read_dir("/tmp/.X11-unix") {
        let mut displays: Vec<u32> = entries
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().to_string_lossy().strip_prefix('X').and_then(|n| n.parse().ok()))
            .collect();
        displays.sort_unstable();
        if let Some(n) = displays.first() {
            found.push(("DISPLAY", format!(":{n}")));
        }
    }
    (!found.is_empty()).then_some(found)
}

/// Put `text` in tmux's buffer, and (tmux 3.2+) the outer clipboard.
fn tmux_load(text: &str) -> bool {
    run_writing("tmux", &["load-buffer", "-w", "-"], &[], text) || run_writing("tmux", &["load-buffer", "-"], &[], text)
}

fn run_writing(program: &str, args: &[&str], env: &[(&str, String)], text: &str) -> bool {
    let Ok(mut child) = Command::new(program)
        .args(args)
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    let wrote = child.stdin.take().is_some_and(|mut stdin| stdin.write_all(text.as_bytes()).is_ok());
    // wl-copy and xclip fork a background owner and exit promptly.
    wrote && wait_for(&mut child, Duration::from_secs(2)).is_some_and(|status| status.success())
}

/// What `program` prints, if it succeeds within a second.
fn run_reading(program: &str, args: &[&str], env: &[(&str, String)]) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stdout.read_to_end(&mut bytes);
        bytes
    });
    let status = wait_for(&mut child, Duration::from_secs(1))?;
    let bytes = reader.join().ok()?;
    (status.success() && !bytes.is_empty()).then(|| String::from_utf8_lossy(&bytes).into_owned())
}

/// The child's exit, or `None` (and the child killed) past `limit`.
fn wait_for(child: &mut std::process::Child, limit: Duration) -> Option<std::process::ExitStatus> {
    let began = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) if began.elapsed() < limit => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

/// The last fenced code block in a markdown reply.
pub fn last_code_block(markdown: &str) -> Option<String> {
    let mut blocks = Vec::new();
    let mut current: Option<(String, String)> = None; // (fence, body)
    for line in markdown.lines() {
        let trimmed = line.trim_start();
        match current.take() {
            None => {
                let fence: String = trimmed.chars().take_while(|&c| c == '`' || c == '~').collect();
                let uniform = fence.chars().all(|c| c == fence.chars().next().unwrap_or('`'));
                if fence.len() >= 3 && uniform {
                    current = Some((fence, String::new()));
                }
            }
            Some((fence, mut body)) => {
                if trimmed.starts_with(&fence) && trimmed.trim_end().chars().all(|c| fence.contains(c)) {
                    blocks.push(body);
                } else {
                    body.push_str(line);
                    body.push('\n');
                    current = Some((fence, body));
                }
            }
        }
    }
    if let Some((_, body)) = current {
        blocks.push(body);
    }
    blocks.pop().map(|mut body| {
        if body.ends_with('\n') {
            body.pop();
        }
        body
    })
}

fn copy_with_tool(text: &str, display: &[(&'static str, String)]) -> Option<&'static str> {
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some() || display.iter().any(|(k, _)| *k == "WAYLAND_DISPLAY");
    let tools: &[(&'static str, &[&str])] = if cfg!(target_os = "macos") {
        &[("pbcopy", &[])]
    } else if wayland {
        &[("wl-copy", &[]), ("xclip", &["-selection", "clipboard"]), ("xsel", &["--clipboard", "--input"])]
    } else {
        &[("xclip", &["-selection", "clipboard"]), ("xsel", &["--clipboard", "--input"])]
    };
    tools.iter().find(|(program, args)| run_writing(program, args, display, text)).map(|(program, _)| *program)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_last_code_block() {
        let reply = "Try:\n```rust\nfn a() {}\n```\nor\n~~~\nb\n  c\n~~~\ndone";
        assert_eq!(last_code_block(reply).as_deref(), Some("b\n  c"));
        assert_eq!(last_code_block("```\nunterminated").as_deref(), Some("unterminated"));
        assert_eq!(last_code_block("no code"), None);
    }

    #[test]
    fn a_clipboard_program_is_read_and_one_that_hangs_is_given_up_on() {
        assert_eq!(run_reading("printf", &["copied text"], &[]).as_deref(), Some("copied text"));
        assert_eq!(run_reading("sh", &["-c", "printf \"$S0MA_TEST_VAR\""], &[("S0MA_TEST_VAR", "from env".into())]).as_deref(), Some("from env"));
        assert_eq!(run_reading("false", &[], &[]), None);
        let began = Instant::now();
        assert_eq!(run_reading("sleep", &["30"], &[]), None);
        assert!(began.elapsed() < Duration::from_secs(5), "waited {:?}", began.elapsed());
        assert!(run_writing("sh", &["-c", "cat > /dev/null"], &[], "x"));
        assert!(!run_writing("no-such-clipboard-tool", &[], &[], "x"));
    }
}
