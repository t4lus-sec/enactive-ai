//! The operator's steering draft while s0ma is active.
//!
//! Reedline owns the keyboard only while s0ma is idle. During a turn, a
//! lightweight crossterm reader edits this draft instead, so it needs its own
//! cursor, motions and history. The draft round-trips through Reedline at the
//! start and end of each turn, so nothing typed is ever lost.

use crate::terminal::safe_text;

use unicode_width::UnicodeWidthChar;

/// Multiline text plus a cursor (a byte offset that is always on a char
/// boundary).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Draft {
    text: String,
    cursor: usize,
}

impl Draft {
    pub fn new(text: &str, cursor: usize) -> Self {
        let text = safe_text(text);
        let mut cursor = cursor.min(text.len());
        while !text.is_char_boundary(cursor) {
            cursor -= 1;
        }
        Self { text, cursor }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn take(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.text)
    }

    /// Replace the whole draft, cursor at the end.
    pub fn set(&mut self, text: &str) {
        self.text = safe_text(text);
        self.cursor = self.text.len();
    }

    // ---- editing --------------------------------------------------------

    pub fn insert_str(&mut self, s: &str) {
        // Same sanitizing as the transcript: no escape sequences in the pane.
        let s = safe_text(s);
        self.text.insert_str(self.cursor, &s);
        self.cursor += s.len();
    }

    /// Replace `start..end` with `s`, leaving the cursor after it.
    pub fn replace(&mut self, start: usize, end: usize, s: &str) {
        let s = safe_text(s);
        let end = end.min(self.text.len());
        let start = start.min(end);
        if !self.text.is_char_boundary(start) || !self.text.is_char_boundary(end) {
            return;
        }
        self.text.replace_range(start..end, &s);
        self.cursor = start + s.len();
    }

    pub fn insert(&mut self, c: char) {
        self.insert_str(c.encode_utf8(&mut [0; 4]));
    }

    pub fn backspace(&mut self) {
        if let Some(prev) = self.prev_boundary(self.cursor) {
            self.text.replace_range(prev..self.cursor, "");
            self.cursor = prev;
        }
    }

    pub fn delete(&mut self) {
        if let Some(next) = self.next_boundary(self.cursor) {
            self.text.replace_range(self.cursor..next, "");
        }
    }

    /// Ctrl+W / Alt+Backspace.
    pub fn delete_word_before(&mut self) {
        let start = self.word_start_before(self.cursor);
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Alt+D.
    pub fn delete_word_after(&mut self) {
        let end = self.word_end_after(self.cursor);
        self.text.replace_range(self.cursor..end, "");
    }

    /// Ctrl+U: delete from the start of the current line to the cursor.
    pub fn kill_line_before(&mut self) {
        let start = self.line_start(self.cursor);
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Ctrl+K: delete from the cursor to the end of the current line.
    pub fn kill_line_after(&mut self) {
        let end = self.line_end(self.cursor);
        self.text.replace_range(self.cursor..end, "");
    }

    // ---- motion ---------------------------------------------------------

    pub fn left(&mut self) {
        if let Some(prev) = self.prev_boundary(self.cursor) {
            self.cursor = prev;
        }
    }

    pub fn right(&mut self) {
        if let Some(next) = self.next_boundary(self.cursor) {
            self.cursor = next;
        }
    }

    pub fn word_left(&mut self) {
        self.cursor = self.word_start_before(self.cursor);
    }

    pub fn word_right(&mut self) {
        self.cursor = self.word_end_after(self.cursor);
    }

    pub fn home(&mut self) {
        self.cursor = self.line_start(self.cursor);
    }

    pub fn end(&mut self) {
        self.cursor = self.line_end(self.cursor);
    }

    /// Move to the previous line of a multiline draft, keeping the column.
    /// Returns false on the first line, so the caller can fall back to
    /// history.
    pub fn up(&mut self) -> bool {
        let start = self.line_start(self.cursor);
        if start == 0 {
            return false;
        }
        let column = display_width(&self.text[start..self.cursor]);
        let prev_start = self.line_start(start - 1);
        self.cursor = self.at_column(prev_start, start - 1, column);
        true
    }

    /// Move to the next line, keeping the column. Returns false on the last
    /// line.
    pub fn down(&mut self) -> bool {
        let end = self.line_end(self.cursor);
        if end == self.text.len() {
            return false;
        }
        let column = display_width(&self.text[self.line_start(self.cursor)..self.cursor]);
        let next_start = end + 1;
        let next_end = self.line_end(next_start);
        self.cursor = self.at_column(next_start, next_end, column);
        true
    }

    // ---- helpers --------------------------------------------------------

    fn prev_boundary(&self, at: usize) -> Option<usize> {
        self.text[..at].char_indices().next_back().map(|(i, _)| i)
    }

    fn next_boundary(&self, at: usize) -> Option<usize> {
        self.text[at..].chars().next().map(|c| at + c.len_utf8())
    }

    fn line_start(&self, at: usize) -> usize {
        self.text[..at].rfind('\n').map_or(0, |i| i + 1)
    }

    fn line_end(&self, at: usize) -> usize {
        self.text[at..].find('\n').map_or(self.text.len(), |i| at + i)
    }

    fn word_start_before(&self, at: usize) -> usize {
        let before = &self.text[..at];
        let trimmed = before.trim_end_matches(char::is_whitespace);
        trimmed
            .char_indices()
            .rev()
            .find(|(_, c)| c.is_whitespace())
            .map_or(0, |(i, c)| i + c.len_utf8())
    }

    fn word_end_after(&self, at: usize) -> usize {
        let after = &self.text[at..];
        let skipped = after.len() - after.trim_start_matches(char::is_whitespace).len();
        let rest = &after[skipped..];
        let word = rest.find(char::is_whitespace).unwrap_or(rest.len());
        at + skipped + word
    }

    /// The byte offset in `start..end` closest to display column `column`.
    fn at_column(&self, start: usize, end: usize, column: usize) -> usize {
        let mut cells = 0;
        for (i, c) in self.text[start..end].char_indices() {
            let width = UnicodeWidthChar::width(c).unwrap_or(0);
            if cells + width > column {
                return start + i;
            }
            cells += width;
        }
        end
    }
}

fn display_width(s: &str) -> usize {
    s.chars().map(|c| UnicodeWidthChar::width(c).unwrap_or(0)).sum()
}

/// Up/Down recall of earlier submissions while s0ma is active.
///
/// Entries typed at the idle Reedline prompt are added here too, and steers
/// typed during a turn are handed back to Reedline's history when the turn
/// ends, so Up behaves the same in both modes.
#[derive(Default)]
pub struct SteerHistory {
    entries: Vec<String>,
    browsing: Option<usize>,
    stash: Option<Draft>,
    unsynced: Vec<String>,
}

const HISTORY_LIMIT: usize = 500;

impl SteerHistory {
    /// Forget every line held, so Up recalls nothing from before now.
    ///
    /// For a job's close: this list is the steering pane's own copy of what was
    /// typed, including lines whose content has just been taken out of the
    /// archive and the history file. Unsynced lines go too -- they are the ones
    /// not yet written anywhere, so dropping them is what keeps them out.
    pub fn forget(&mut self) {
        self.entries.clear();
        self.unsynced.clear();
        self.browsing = None;
        self.stash = None;
    }

    /// Record a submission. `from_active` entries still need to be copied
    /// into Reedline's history.
    pub fn push(&mut self, entry: &str, from_active: bool) {
        self.stop_browsing();
        if entry.trim().is_empty() || self.entries.last().is_some_and(|last| last == entry) {
            return;
        }
        self.entries.push(entry.to_string());
        if self.entries.len() > HISTORY_LIMIT {
            self.entries.remove(0);
        }
        if from_active {
            self.unsynced.push(entry.to_string());
        }
    }

    pub fn take_unsynced(&mut self) -> Vec<String> {
        std::mem::take(&mut self.unsynced)
    }

    /// Any edit ends history browsing; the edited entry becomes the draft.
    pub fn stop_browsing(&mut self) {
        self.browsing = None;
        self.stash = None;
    }

    pub fn previous(&mut self, draft: &mut Draft) -> bool {
        let index = match self.browsing {
            None if self.entries.is_empty() => return false,
            None => {
                self.stash = Some(draft.clone());
                self.entries.len() - 1
            }
            Some(0) => return false,
            Some(i) => i - 1,
        };
        self.browsing = Some(index);
        draft.set(&self.entries[index]);
        true
    }

    pub fn next(&mut self, draft: &mut Draft) -> bool {
        match self.browsing {
            None => false,
            Some(i) if i + 1 < self.entries.len() => {
                self.browsing = Some(i + 1);
                draft.set(&self.entries[i + 1]);
                true
            }
            Some(_) => {
                // Past the newest entry: back to what was being typed.
                *draft = self.stash.take().unwrap_or_default();
                self.browsing = None;
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(draft: &Draft) -> String {
        let mut s = draft.text().to_string();
        s.insert(draft.cursor(), '|');
        s
    }

    #[test]
    fn insert_and_delete_at_the_cursor() {
        let mut d = Draft::new("helo world", 3);
        d.insert('l');
        assert_eq!(at(&d), "hell|o world");
        d.end();
        d.backspace();
        assert_eq!(at(&d), "hello worl|");
        d.home();
        d.delete();
        assert_eq!(at(&d), "|ello worl");
    }

    #[test]
    fn unicode_is_never_split() {
        let mut d = Draft::new("a🌱é", usize::MAX);
        d.left();
        assert_eq!(at(&d), "a🌱|é");
        d.left();
        d.backspace();
        assert_eq!(at(&d), "|🌱é");
        // A cursor inside a multi-byte char is pulled back to a boundary.
        assert_eq!(Draft::new("🌱", 2).cursor(), 0);
    }

    #[test]
    fn word_motion_and_deletion() {
        let mut d = Draft::new("one two  three", usize::MAX);
        d.word_left();
        assert_eq!(at(&d), "one two  |three");
        d.word_left();
        assert_eq!(at(&d), "one |two  three");
        d.word_right();
        assert_eq!(at(&d), "one two|  three");
        d.delete_word_before();
        assert_eq!(at(&d), "one |  three");
        d.delete_word_after();
        assert_eq!(at(&d), "one |");
    }

    #[test]
    fn line_kills_stay_on_their_line() {
        let mut d = Draft::new("first\nsecond line", 9);
        d.kill_line_before();
        assert_eq!(at(&d), "first\n|ond line");
        d.kill_line_after();
        assert_eq!(at(&d), "first\n|");
    }

    #[test]
    fn up_and_down_keep_the_column() {
        let mut d = Draft::new("abcdef\nxy\nlonger", usize::MAX);
        assert!(d.up());
        assert_eq!(at(&d), "abcdef\nxy|\nlonger");
        assert!(d.up());
        assert_eq!(at(&d), "ab|cdef\nxy\nlonger");
        assert!(!d.up());
        assert!(d.down());
        assert!(d.down());
        assert!(!d.down());
        assert_eq!(at(&d), "abcdef\nxy\nlo|nger");
    }

    #[test]
    fn pasted_escapes_are_neutralised() {
        let mut d = Draft::default();
        d.insert_str("a\x1b[2Jb\r\nc\td");
        assert_eq!(d.text(), "a␛[2Jb\nc    d");
    }

    #[test]
    fn history_browses_and_restores_the_draft() {
        let mut h = SteerHistory::default();
        h.push("first", false);
        h.push("second", true);
        h.push("second", true); // consecutive duplicate ignored
        let mut d = Draft::new("typing", usize::MAX);

        assert!(h.previous(&mut d));
        assert_eq!(d.text(), "second");
        assert!(h.previous(&mut d));
        assert_eq!(d.text(), "first");
        assert!(!h.previous(&mut d));
        assert!(h.next(&mut d));
        assert!(h.next(&mut d));
        assert_eq!(d.text(), "typing");
        assert!(!h.next(&mut d));
        assert_eq!(h.take_unsynced(), vec!["second".to_string()]);
    }
}
