//! s0ma's own page: the terminal's alternate screen, with the tab bar and a
//! rule under it on the two top rows, and the conversation under them.
//!
//! The conversation scrolls in the rows below the bar (a scroll region that
//! starts under it), so the bar stays put as text goes by. The terminal keeps
//! no scrollback of its own here: what went off the top is drawn again from
//! the record (`terminal::TranscriptRecord`) when the page is (a resize, a
//! tab closing, Ctrl+O), and PgUp or the wheel scroll back through it in a
//! view of its own (`tabs::View::History`).

use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// The columns `c` adds after `previous`: a zero-width selector can make the
/// character before it wide (an emoji's VS16), as the renderer counts it.
fn added_width(previous: Option<char>, c: char) -> usize {
    match (c.width(), previous) {
        (Some(0), Some(previous)) => {
            let mut pair = String::from(previous);
            pair.push(c);
            UnicodeWidthStr::width(pair.as_str()).saturating_sub(previous.width().unwrap_or(0))
        }
        (width, _) => width.unwrap_or(0),
    }
}

/// How many rows the tab bar takes now (`tabs::main_bar` says, each time it
/// is made): it wraps onto more when a half runs out of room.
#[cfg(not(test))]
static BAR_ROWS: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(1);

// Tests draw bars at once on many threads: each keeps its own.
#[cfg(test)]
thread_local! {
    static BAR_ROWS: std::cell::Cell<u16> = const { std::cell::Cell::new(1) };
}

pub fn set_bar_rows(rows: usize) {
    let rows = rows.clamp(1, 3) as u16;
    #[cfg(not(test))]
    BAR_ROWS.store(rows, Ordering::SeqCst);
    #[cfg(test)]
    BAR_ROWS.with(|kept| kept.set(rows));
}

/// The rows atop the page: the tab bar's, and the rule under them.
pub fn top_rows() -> u16 {
    #[cfg(not(test))]
    let rows = BAR_ROWS.load(Ordering::SeqCst);
    #[cfg(test)]
    let rows = BAR_ROWS.with(|kept| kept.get());
    rows + 1
}

/// The page has the screen (s0ma is drawing on the alternate screen).
static ON: AtomicBool = AtomicBool::new(false);

/// Something else drew over the page (a review window, an editor): drawn
/// again before the idle prompt is.
static STALE: AtomicBool = AtomicBool::new(false);

/// Whether s0ma is drawing on its own page.
pub fn on() -> bool {
    ON.load(Ordering::SeqCst)
}

/// Take the alternate screen for the page, cleared, when the output is a
/// terminal. The returned guard gives the shell its screen back.
pub fn enter() -> Option<Guard> {
    if !io::stdout().is_terminal() {
        return None;
    }
    let rows = crossterm::terminal::size().map_or(24, |(_, rows)| rows);
    let mut stdout = io::stdout().lock();
    // The conversation's rows scroll from the start, under the top two
    // (drawn with the first prompt).
    let _ = write!(stdout, "\x1b[?1049h\x1b[r\x1b[H\x1b[2J{}", region(rows));
    let _ = write!(stdout, "\x1b[{};1H", top_rows() + 1);
    let _ = stdout.flush();
    ON.store(true, Ordering::SeqCst);
    Some(Guard)
}

/// The shell's screen, for a program that draws on its own (an editor):
/// `back` takes the page again, to be drawn afresh.
pub fn step_out() {
    if on() {
        let mut stdout = io::stdout().lock();
        let _ = write!(stdout, "\x1b[r\x1b[?1049l");
        let _ = stdout.flush();
    }
}

/// Back on the page after `step_out` or another screen's use of it: cleared,
/// and marked to be drawn again.
pub fn back() {
    if on() && !crate::terminal::torn_down() {
        let mut stdout = io::stdout().lock();
        let _ = write!(stdout, "\x1b[?1049h\x1b[r\x1b[H\x1b[2J");
        let _ = stdout.flush();
        mark_stale();
    }
}

/// Something drew over the page; it is drawn again before the idle prompt.
pub fn mark_stale() {
    STALE.store(true, Ordering::SeqCst);
}

/// Whether the page has to be drawn again, clearing the mark.
pub fn take_stale() -> bool {
    STALE.swap(false, Ordering::SeqCst)
}

/// What gives the shell its screen back, on the way out.
pub struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        leave();
    }
}

/// The shell's screen back, for good (s0ma is ending, or this terminal is
/// detaching), with `last` under what the shell had: the end of the
/// conversation as the page showed it, so the last words (and any warning
/// on the way out) stay readable there.
pub fn leave_with(last: &[String]) {
    if ON.swap(false, Ordering::SeqCst) && !crate::terminal::torn_down() {
        let mut stdout = io::stdout().lock();
        let _ = write!(stdout, "\x1b[r\x1b[?1049l");
        if !last.is_empty() {
            let _ = write!(stdout, "{}\x1b[0m\r\n", last.join("\r\n"));
        }
        let _ = stdout.flush();
    }
}

/// The shell's screen back, for good, with nothing more.
pub fn leave() {
    leave_with(&[]);
}

/// Whether the page was up, cleared (the terminal is being put back after a
/// panic, which leaves the alternate screen itself).
pub fn take_on() -> bool {
    ON.swap(false, Ordering::SeqCst)
}

/// The scroll region for the conversation: from under the rule down to row
/// `bottom` (1-based, inclusive; the screen's height when nothing is under it).
pub fn region(bottom: u16) -> String {
    format!("\x1b[{};{}r", top_rows() + 1, bottom.max(top_rows() + 2))
}

/// The top rows drawn: `bar` (with its colours, a row to a line,
/// `tabs::main_bar`) and a dim rule `columns` wide under it. Moves the
/// cursor; the caller puts it back.
pub fn top(bar: &str, columns: u16) -> String {
    let rule = "─".repeat(columns as usize);
    let mut out = String::new();
    let mut row = 1;
    for line in bar.split('\n') {
        out.push_str(&format!("\x1b[{row};1H\x1b[2K{line}\x1b[0m"));
        row += 1;
    }
    out.push_str(&format!("\x1b[{row};1H\x1b[2K\x1b[2;90m{rule}\x1b[0m"));
    out
}

/// How many columns `row` (a row of `rows`) takes: its codes take none.
pub fn width(row: &str) -> usize {
    let mut width = 0;
    let mut previous = None;
    let mut chars = row.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.next() {
                Some('[') => {
                    for next in chars.by_ref() {
                        if ('\x40'..='\x7e').contains(&next) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    let mut escape = false;
                    for next in chars.by_ref() {
                        if next == '\x07' || (escape && next == '\\') {
                            break;
                        }
                        escape = next == '\x1b';
                    }
                }
                _ => {}
            }
        } else if !c.is_control() {
            width += added_width(previous, c);
            previous = Some(c);
        }
    }
    width
}

/// Rendered transcript text cut into the rows a terminal `columns` wide
/// shows it in, each row standing on its own: it starts with the colours in
/// force there and ends with them reset (all but the last, which is left
/// open, as the text left it).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Rows {
    pub rows: Vec<String>,
    /// The column the text ends at, on the last row.
    pub column: usize,
}

pub fn rows(text: &str, columns: usize) -> Rows {
    let columns = columns.max(1);
    let mut rows = Vec::new();
    // The colour codes in force: every one since the last reset.
    let mut colours: Vec<String> = Vec::new();
    let mut row = String::new();
    let mut column = 0;
    // The last character drawn, for a selector after it.
    let mut previous: Option<char> = None;
    let finish = |row: &mut String, colours: &[String], rows: &mut Vec<String>| {
        if !colours.is_empty() {
            row.push_str("\x1b[0m");
        }
        rows.push(std::mem::replace(row, colours.concat()));
    };
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' => {
                let mut sequence = String::from(c);
                match chars.peek() {
                    Some('[') => {
                        sequence.push(chars.next().unwrap_or('['));
                        for next in chars.by_ref() {
                            sequence.push(next);
                            if ('\x40'..='\x7e').contains(&next) {
                                break;
                            }
                        }
                        if sequence.len() < 3 {
                            continue;
                        }
                        let parameters = &sequence[2..sequence.len() - 1];
                        match sequence.chars().last() {
                            Some('m') if parameters.is_empty() || parameters == "0" => {
                                colours.clear();
                                row.push_str(&sequence);
                            }
                            Some('m') => {
                                colours.push(sequence.clone());
                                row.push_str(&sequence);
                            }
                            // The line is rewritten (a work line that counts).
                            Some('K') if parameters == "2" => {
                                row = colours.concat();
                                column = 0;
                            }
                            _ => {}
                        }
                    }
                    Some(']') => {
                        // A link or a title, kept as it is: up to BEL or ST.
                        for next in chars.by_ref() {
                            sequence.push(next);
                            if next == '\x07' || sequence.ends_with("\x1b\\") {
                                break;
                            }
                        }
                        row.push_str(&sequence);
                    }
                    _ => {
                        chars.next();
                    }
                }
            }
            '\n' => {
                finish(&mut row, &colours, &mut rows);
                column = 0;
            }
            '\r' => {}
            '\t' => {
                let to = (column / 8 + 1) * 8;
                while column < to.min(columns) {
                    row.push(' ');
                    column += 1;
                }
            }
            c if c.is_control() => {}
            c => {
                let width = added_width(previous, c);
                // A selector stays with its character, wherever that is.
                if column + width > columns && c.width().unwrap_or(0) > 0 {
                    finish(&mut row, &colours, &mut rows);
                    column = 0;
                }
                row.push(c);
                column += width;
                previous = Some(c);
            }
        }
    }
    rows.push(row);
    Rows { rows, column }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_split_at_newlines_and_wrap_at_the_width() {
        let shown = rows("one\r\ntwo three\r\n", 5);
        assert_eq!(shown.rows, vec!["one", "two t", "hree", ""]);
        assert_eq!(shown.column, 0);
        let shown = rows("abcde", 5);
        assert_eq!(shown.rows, vec!["abcde"]);
        assert_eq!(shown.column, 5, "a full row waits to wrap, as a terminal does");
    }

    #[test]
    fn every_row_carries_the_colours_in_force() {
        let shown = rows("\x1b[1mbold\r\nstill\x1b[0m\r\nplain", 20);
        assert_eq!(shown.rows, vec!["\x1b[1mbold\x1b[0m", "\x1b[1mstill\x1b[0m", "plain"]);
        // The last row is left as the text left it.
        let shown = rows("\x1b[32mgreen\r\nmore", 20);
        assert_eq!(shown.rows[1], "\x1b[32mmore");
    }

    #[test]
    fn a_rewritten_line_shows_only_what_it_says_last() {
        let shown = rows("  \x1b[3m⠋ 1 call\x1b[0m\r\x1b[2K  \x1b[3m✓ 2 calls\x1b[0m\r\nnext", 40);
        assert_eq!(shown.rows, vec!["  \x1b[3m✓ 2 calls\x1b[0m", "next"]);
    }

    #[test]
    fn an_emoji_made_wide_by_its_selector_counts_two() {
        assert_eq!(width("⚠\u{FE0F} x"), UnicodeWidthStr::width("⚠\u{FE0F} x"));
        assert_eq!(rows("ab⚠\u{FE0F}", 3).rows, vec!["ab⚠\u{FE0F}"]);
        assert_eq!(rows("ab⚠\u{FE0F}", 3).column, 4);
        // A cut-off code at the very end is dropped, not a panic.
        assert_eq!(rows("ok\x1b[", 10).rows, vec!["ok"]);
    }

    #[test]
    fn wide_characters_move_whole_to_the_next_row() {
        let shown = rows("ab日本", 5);
        assert_eq!(shown.rows, vec!["ab日", "本"]);
        assert_eq!(shown.column, 2);
    }

    #[test]
    fn the_region_starts_under_the_rule() {
        assert_eq!(region(24), "\x1b[3;24r");
        // Never less than two rows, which is all a terminal takes.
        assert_eq!(region(2), "\x1b[3;4r");
    }
}
