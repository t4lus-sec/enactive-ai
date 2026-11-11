//! Tell the operator a long turn has finished.
//!
//! Always rings the terminal bell (most terminals flag the tab/window). On a
//! local desktop with `notify-send` that also gives a desktop notification;
//! otherwise (e.g. over SSH) the OSC 9 and OSC 777 notification sequences are
//! sent, which iTerm2, kitty, WezTerm, Ghostty, foot and others turn into a
//! desktop notification and everything else ignores.
//!
//! `S0MA_NOTIFY=off` disables this; `S0MA_NOTIFY=bell` keeps only the bell.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Without focus reporting, only turns at least this long notify.
const UNKNOWN_FOCUS_THRESHOLD: Duration = Duration::from_secs(30);
/// Even when the terminal is unfocused, don't notify for near-instant turns.
const MIN_TURN: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Focus {
    /// The terminal has never reported focus changes.
    Unknown,
    Focused,
    Unfocused,
}

pub fn should_notify(elapsed: Duration, focus: Focus) -> bool {
    match focus {
        Focus::Focused => false,
        Focus::Unfocused => elapsed >= MIN_TURN,
        Focus::Unknown => elapsed >= UNKNOWN_FOCUS_THRESHOLD,
    }
}

pub fn turn_finished(reply: &str) {
    let mode = std::env::var("S0MA_NOTIFY").unwrap_or_default();
    if mode == "off" {
        return;
    }

    let mut out = String::from("\x07");
    if mode != "bell" {
        let body = summary(reply);
        if !desktop_notification(&body) {
            out.push_str(&passthrough(&format!("\x1b]9;s0ma: {body}\x07")));
            out.push_str(&passthrough(&format!("\x1b]777;notify;s0ma;{body}\x07")));
        }
    }

    let mut stdout = std::io::stdout().lock();
    let _ = stdout.write_all(out.as_bytes());
    let _ = stdout.flush();
}

/// tmux swallows OSC sequences unless they are wrapped for passthrough.
pub fn passthrough(sequence: &str) -> String {
    if std::env::var_os("TMUX").is_some() {
        format!("\x1bPtmux;{}\x1b\\", sequence.replace('\x1b', "\x1b\x1b"))
    } else {
        sequence.to_string()
    }
}

/// First line of the reply, minus markdown markers, as plain safe text.
fn summary(reply: &str) -> String {
    let line = reply
        .lines()
        .map(|l| l.trim().trim_start_matches(['#', '>', '-', '*', ' ']))
        .find(|l| !l.is_empty())
        .unwrap_or("finished");
    let clean: String = line
        .chars()
        .filter(|c| !c.is_control() && !matches!(c, '*' | '`' | '_' | '~'))
        .map(|c| if c == ';' { ',' } else { c })
        .collect();
    let mut out: String = clean.chars().take(100).collect();
    if clean.chars().count() > 100 {
        out.push('…');
    }
    if out.is_empty() { "finished".into() } else { out }
}

fn desktop_notification(body: &str) -> bool {
    let local = std::env::var_os("SSH_CONNECTION").is_none()
        && std::env::var_os("SSH_TTY").is_none()
        && (std::env::var_os("WAYLAND_DISPLAY").is_some() || std::env::var_os("DISPLAY").is_some());
    if !local {
        return false;
    }
    let Ok(mut child) = Command::new("notify-send")
        .args(["--app-name=s0ma", "s0ma", body])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    // Reap it in the background rather than leave a zombie.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notifies_only_when_away_or_long() {
        let s = Duration::from_secs;
        assert!(!should_notify(s(600), Focus::Focused));
        assert!(should_notify(s(5), Focus::Unfocused));
        assert!(!should_notify(s(1), Focus::Unfocused));
        assert!(!should_notify(s(10), Focus::Unknown));
        assert!(should_notify(s(45), Focus::Unknown));
    }

    #[test]
    fn summary_is_plain_one_line() {
        assert_eq!(summary("## Done: **all** `5` tests\nmore"), "Done: all 5 tests");
        assert_eq!(summary("a;b\x1b[2J"), "a,b[2J");
        assert_eq!(summary(""), "finished");
    }
}
