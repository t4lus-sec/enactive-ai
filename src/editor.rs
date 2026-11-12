//! The editor's text: lines, the cursor and selection, edits, undo, find,
//! and the three-way merge used when a file changed while it was open
//! (`claude/editor-spec.md`). Nothing here draws or does I/O: the tab
//! (`tabs/editor_view.rs`) does.
//!
//! The text is a `Vec<String>` of lines without their endings. A position is
//! (line, byte offset), always on a char boundary. Line endings (LF or CRLF)
//! and whether the file ended in a newline are kept as found, so saving an
//! unchanged file writes the same bytes.

pub(crate) mod highlight;
pub(crate) mod live;

use std::time::{Duration, Instant};
use unicode_width::UnicodeWidthChar;

/// Typing more within this long of the last key goes in the same undo step.
const GROUP_PAUSE: Duration = Duration::from_secs(1);
/// Undo steps kept.
const UNDO_KEPT: usize = 2_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Pos {
    pub line: usize,
    pub col: usize,
}

impl Pos {
    pub fn new(line: usize, col: usize) -> Self {
        Self { line, col }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    Lf,
    Crlf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Indent {
    Tabs,
    Spaces(usize),
}

impl Indent {
    pub fn unit(&self) -> String {
        match self {
            Indent::Tabs => "\t".into(),
            Indent::Spaces(n) => " ".repeat(*n),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Indent::Tabs => "tabs".into(),
            Indent::Spaces(n) => format!("{n} spaces"),
        }
    }
}

/// One change: `removed` at `at` became `inserted`.
#[derive(Clone, Debug)]
struct Edit {
    at: Pos,
    removed: String,
    inserted: String,
}

/// What undo takes back in one go, and where the cursor was either side.
#[derive(Clone, Debug)]
struct Step {
    edits: Vec<Edit>,
    before: (Pos, Option<Pos>),
    after: (Pos, Option<Pos>),
}

/// Which kind of edit the open undo step is, for grouping typing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Typing,
    Deleting,
    Other,
}

#[derive(Clone, Debug)]
pub struct Buffer {
    lines: Vec<String>,
    pub ending: Ending,
    pub final_newline: bool,
    pub indent: Indent,
    pub tab_width: usize,
    pub cursor: Pos,
    /// The selection's other end, when there is a selection.
    pub anchor: Option<Pos>,
    /// The display column up and down keep to.
    goal: Option<usize>,
    undo: Vec<Step>,
    redo: Vec<Step>,
    open: Option<(Kind, Instant)>,
    /// Bumped by every change of the text.
    pub revision: u64,
    /// The first line changed since `take_changed_from` was last asked (for
    /// the highlighter).
    changed_from: Option<usize>,
    /// Lines someone else changed (the agent, by a reload or a merge), kept in
    /// step with the lines as they move.
    pub theirs: Vec<bool>,
}

impl Buffer {
    pub fn from_text(text: &str, tab_width: usize, default_indent: Indent) -> Self {
        let crlf = text.matches("\r\n").count();
        let lf = text.matches('\n').count();
        let ending = if crlf > 0 && crlf * 2 >= lf { Ending::Crlf } else { Ending::Lf };
        let final_newline = text.ends_with('\n');
        let lines = split_lines(text);
        let indent = detect_indent(&lines).unwrap_or(default_indent);
        let theirs = vec![false; lines.len()];
        Self {
            lines,
            ending,
            final_newline,
            indent,
            tab_width: tab_width.max(1),
            cursor: Pos::default(),
            anchor: None,
            goal: None,
            undo: Vec::new(),
            redo: Vec::new(),
            open: None,
            revision: 0,
            changed_from: Some(0),
            theirs,
        }
    }

    /// The whole text as it is written: its line endings, and a final newline
    /// when the file had one.
    pub fn text(&self) -> String {
        let newline = match self.ending {
            Ending::Lf => "\n",
            Ending::Crlf => "\r\n",
        };
        let mut text = self.lines.join(newline);
        if self.final_newline {
            text.push_str(newline);
        }
        text
    }

    /// The lines, without endings.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    pub fn line(&self, index: usize) -> &str {
        self.lines.get(index).map_or("", String::as_str)
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// The first line changed since this was last asked, if any.
    pub fn take_changed_from(&mut self) -> Option<usize> {
        self.changed_from.take()
    }

    // -------------------------------------------------------------- positions

    pub fn clamp(&self, pos: Pos) -> Pos {
        let line = pos.line.min(self.lines.len().saturating_sub(1));
        let text = &self.lines[line];
        let mut col = pos.col.min(text.len());
        while !text.is_char_boundary(col) {
            col -= 1;
        }
        Pos { line, col }
    }

    pub fn end_pos(&self) -> Pos {
        let line = self.lines.len() - 1;
        Pos { line, col: self.lines[line].len() }
    }

    /// The selection, start first, when it is not empty.
    pub fn selection(&self) -> Option<(Pos, Pos)> {
        let anchor = self.anchor?;
        (anchor != self.cursor).then(|| if anchor < self.cursor { (anchor, self.cursor) } else { (self.cursor, anchor) })
    }

    pub fn selected_text(&self) -> Option<String> {
        self.selection().map(|(start, end)| self.slice(start, end))
    }

    /// The text between two positions, lines joined by `\n`.
    pub fn slice(&self, start: Pos, end: Pos) -> String {
        if start.line == end.line {
            return self.lines[start.line][start.col..end.col].to_string();
        }
        let mut out = self.lines[start.line][start.col..].to_string();
        for line in &self.lines[start.line + 1..end.line] {
            out.push('\n');
            out.push_str(line);
        }
        out.push('\n');
        out.push_str(&self.lines[end.line][..end.col]);
        out
    }

    /// The column `col` (bytes) is drawn at, tabs to their stops.
    pub fn display_col(&self, line: usize, col: usize) -> usize {
        display_width(&self.line(line)[..col.min(self.line(line).len())], self.tab_width)
    }

    /// The byte offset drawn at display column `target` (or before it, inside
    /// a wide character or a tab).
    pub fn col_at(&self, line: usize, target: usize) -> usize {
        let text = self.line(line);
        let mut width = 0;
        for (index, c) in text.char_indices() {
            let w = char_width(c, width, self.tab_width);
            if width + w > target {
                return index;
            }
            width += w;
        }
        text.len()
    }

    // -------------------------------------------------------------- moving

    fn set_cursor(&mut self, pos: Pos, extend: bool, keep_goal: bool) {
        if extend {
            if self.anchor.is_none() {
                self.anchor = Some(self.cursor);
            }
        } else {
            self.anchor = None;
        }
        self.cursor = self.clamp(pos);
        if !keep_goal {
            self.goal = None;
        }
        self.open = None;
    }

    /// Put the cursor at `pos` (a click, go to line).
    pub fn go(&mut self, pos: Pos, extend: bool) {
        self.set_cursor(pos, extend, false);
    }

    pub fn left(&mut self, extend: bool) {
        if !extend && let Some((start, _)) = self.selection() {
            return self.set_cursor(start, false, false);
        }
        let Pos { line, col } = self.cursor;
        let pos = if col > 0 {
            Pos::new(line, prev_boundary(&self.lines[line], col))
        } else if line > 0 {
            Pos::new(line - 1, self.lines[line - 1].len())
        } else {
            self.cursor
        };
        self.set_cursor(pos, extend, false);
    }

    pub fn right(&mut self, extend: bool) {
        if !extend && let Some((_, end)) = self.selection() {
            return self.set_cursor(end, false, false);
        }
        let Pos { line, col } = self.cursor;
        let pos = if col < self.lines[line].len() {
            Pos::new(line, next_boundary(&self.lines[line], col))
        } else if line + 1 < self.lines.len() {
            Pos::new(line + 1, 0)
        } else {
            self.cursor
        };
        self.set_cursor(pos, extend, false);
    }

    /// Up or down `rows` lines, keeping to the display column it started at.
    pub fn vertical(&mut self, rows: isize, extend: bool) {
        let goal = self.goal.unwrap_or_else(|| self.display_col(self.cursor.line, self.cursor.col));
        let last = self.lines.len() as isize - 1;
        let target = self.cursor.line as isize + rows;
        let pos = if target < 0 {
            Pos::new(0, 0)
        } else if target > last {
            self.end_pos()
        } else {
            let line = target as usize;
            Pos::new(line, self.col_at(line, goal))
        };
        self.set_cursor(pos, extend, true);
        self.goal = Some(goal);
    }

    /// Home: the first non-blank, then the line's start.
    pub fn home(&mut self, extend: bool) {
        let text = &self.lines[self.cursor.line];
        let first = text.len() - text.trim_start().len();
        let col = if self.cursor.col == first { 0 } else { first };
        self.set_cursor(Pos::new(self.cursor.line, col), extend, false);
    }

    pub fn end(&mut self, extend: bool) {
        let col = self.lines[self.cursor.line].len();
        self.set_cursor(Pos::new(self.cursor.line, col), extend, false);
    }

    pub fn start_of_file(&mut self, extend: bool) {
        self.set_cursor(Pos::default(), extend, false);
    }

    pub fn end_of_file(&mut self, extend: bool) {
        self.set_cursor(self.end_pos(), extend, false);
    }

    pub fn word_left(&mut self, extend: bool) {
        let pos = self.word_boundary_before(self.cursor);
        self.set_cursor(pos, extend, false);
    }

    pub fn word_right(&mut self, extend: bool) {
        let Pos { line, col } = self.cursor;
        let text = &self.lines[line];
        let pos = if col >= text.len() {
            if line + 1 < self.lines.len() { Pos::new(line + 1, 0) } else { self.cursor }
        } else {
            let rest: Vec<(usize, char)> = text[col..].char_indices().collect();
            let mut i = 0;
            while i < rest.len() && rest[i].1.is_whitespace() {
                i += 1;
            }
            if i < rest.len() {
                let word = is_word(rest[i].1);
                while i < rest.len() && !rest[i].1.is_whitespace() && is_word(rest[i].1) == word {
                    i += 1;
                }
            }
            Pos::new(line, rest.get(i).map_or(text.len(), |(offset, _)| col + offset))
        };
        self.set_cursor(pos, extend, false);
    }

    fn word_boundary_before(&self, pos: Pos) -> Pos {
        let Pos { line, col } = pos;
        if col == 0 {
            return if line > 0 { Pos::new(line - 1, self.lines[line - 1].len()) } else { pos };
        }
        let before: Vec<(usize, char)> = self.lines[line][..col].char_indices().collect();
        let mut i = before.len();
        while i > 0 && before[i - 1].1.is_whitespace() {
            i -= 1;
        }
        if i > 0 {
            let word = is_word(before[i - 1].1);
            while i > 0 && !before[i - 1].1.is_whitespace() && is_word(before[i - 1].1) == word {
                i -= 1;
            }
        }
        Pos::new(line, before.get(i).map_or(0, |(offset, _)| *offset))
    }

    pub fn select_all(&mut self) {
        self.anchor = Some(Pos::default());
        self.cursor = self.end_pos();
        self.open = None;
    }

    pub fn clear_selection(&mut self) {
        self.anchor = None;
    }

    // -------------------------------------------------------------- editing

    /// Replace `start..end` with `text` (`\n` between lines), recorded for
    /// undo in the open step of `kind` or a new one. The cursor goes to the
    /// end of what was inserted.
    fn replace(&mut self, start: Pos, end: Pos, text: &str, kind: Kind) -> Pos {
        let before = (self.cursor, self.anchor);
        let removed = self.slice(start, end);
        let after_pos = self.apply(start, end, text);
        let edit = Edit { at: start, removed, inserted: text.to_string() };
        let now = Instant::now();
        let joins = matches!(self.open, Some((open, at)) if open == kind && kind != Kind::Other && now.duration_since(at) < GROUP_PAUSE)
            && self.undo.last().is_some_and(|step| step.after.0 == before.0 && before.1.is_none());
        self.cursor = after_pos;
        self.anchor = None;
        self.goal = None;
        if joins && let Some(step) = self.undo.last_mut() {
            step.edits.push(edit);
            step.after = (self.cursor, None);
        } else {
            self.undo.push(Step { edits: vec![edit], before, after: (self.cursor, None) });
            if self.undo.len() > UNDO_KEPT {
                self.undo.remove(0);
            }
        }
        // A space typed ends the word: the next word is its own step.
        let ends_word = kind == Kind::Typing && text.chars().all(char::is_whitespace);
        self.open = (kind != Kind::Other && !ends_word).then_some((kind, now));
        if ends_word {
            self.open = None;
        }
        self.redo.clear();
        after_pos
    }

    /// The text change itself, with the marks kept in step: no undo record.
    fn apply(&mut self, start: Pos, end: Pos, text: &str) -> Pos {
        let head = self.lines[start.line][..start.col].to_string();
        let tail = self.lines[end.line][end.col..].to_string();
        let mut new: Vec<String> = text.split('\n').map(str::to_string).collect();
        let last = new.len() - 1;
        let end_col = new[last].len() + if last == 0 { head.len() } else { 0 };
        new[0] = format!("{head}{}", new[0]);
        new[last].push_str(&tail);
        let added = new.len();
        let first_mark = self.theirs.get(start.line).copied().unwrap_or(false);
        let mut marks = vec![false; added];
        // The first line keeps its mark when only its end changed.
        marks[0] = first_mark && start.col > 0;
        self.lines.splice(start.line..=end.line, new);
        let theirs_end = (end.line + 1).min(self.theirs.len());
        if start.line <= theirs_end {
            self.theirs.splice(start.line..theirs_end, marks);
        }
        self.theirs.resize(self.lines.len(), false);
        self.revision += 1;
        self.changed_from = Some(self.changed_from.map_or(start.line, |from| from.min(start.line)));
        Pos::new(start.line + last, end_col)
    }

    /// Type `text` over the selection, or at the cursor.
    pub fn insert(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        let (start, end) = self.selection().unwrap_or((self.cursor, self.cursor));
        let kind = if text.contains('\n') || text.chars().count() > 1 || self.selection().is_some() { Kind::Other } else { Kind::Typing };
        self.replace(start, end, &text, kind);
    }

    /// A paste: one undo step, whatever its size.
    pub fn paste(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        let (start, end) = self.selection().unwrap_or((self.cursor, self.cursor));
        self.replace(start, end, &text, Kind::Other);
    }

    /// Enter: a new line, indented as this one is.
    pub fn newline(&mut self) {
        let line = &self.lines[self.cursor.line];
        let indent: String = line.chars().take_while(|c| *c == ' ' || *c == '\t').collect();
        let indent = &indent[..indent.len().min(self.selection().map_or(self.cursor.col, |(start, _)| if start.line == self.cursor.line { start.col } else { usize::MAX }))];
        let text = format!("\n{indent}");
        let (start, end) = self.selection().unwrap_or((self.cursor, self.cursor));
        self.replace(start, end, &text, Kind::Other);
    }

    pub fn backspace(&mut self) {
        if let Some((start, end)) = self.selection() {
            self.replace(start, end, "", Kind::Other);
            return;
        }
        let Pos { line, col } = self.cursor;
        if col == 0 && line == 0 {
            return;
        }
        // In leading spaces, back to the previous indent stop.
        let text = &self.lines[line];
        let start = if col > 0 && self.indent != Indent::Tabs && text[..col].chars().all(|c| c == ' ') {
            let Indent::Spaces(unit) = self.indent else { unreachable!() };
            let stop = (col - 1) / unit * unit;
            Pos::new(line, stop)
        } else if col > 0 {
            Pos::new(line, prev_boundary(text, col))
        } else {
            Pos::new(line - 1, self.lines[line - 1].len())
        };
        self.replace(start, self.cursor, "", Kind::Deleting);
    }

    pub fn delete(&mut self) {
        if let Some((start, end)) = self.selection() {
            self.replace(start, end, "", Kind::Other);
            return;
        }
        let Pos { line, col } = self.cursor;
        let end = if col < self.lines[line].len() {
            Pos::new(line, next_boundary(&self.lines[line], col))
        } else if line + 1 < self.lines.len() {
            Pos::new(line + 1, 0)
        } else {
            return;
        };
        self.replace(self.cursor, end, "", Kind::Deleting);
    }

    pub fn delete_word_before(&mut self) {
        if let Some((start, end)) = self.selection() {
            self.replace(start, end, "", Kind::Other);
            return;
        }
        let start = self.word_boundary_before(self.cursor);
        if start != self.cursor {
            self.replace(start, self.cursor, "", Kind::Other);
        }
    }

    /// The selected lines, or the cursor's: (first, last).
    fn line_span(&self) -> (usize, usize) {
        match self.selection() {
            Some((start, end)) => {
                // A selection ending at a line's start does not take that line.
                let last = if end.col == 0 && end.line > start.line { end.line - 1 } else { end.line };
                (start.line, last)
            }
            None => (self.cursor.line, self.cursor.line),
        }
    }

    /// Ctrl+K: the line (or the selected lines) gone.
    pub fn delete_lines(&mut self) {
        let (first, last) = self.line_span();
        let (start, end) = if last + 1 < self.lines.len() {
            (Pos::new(first, 0), Pos::new(last + 1, 0))
        } else if first > 0 {
            (Pos::new(first - 1, self.lines[first - 1].len()), Pos::new(last, self.lines[last].len()))
        } else {
            (Pos::new(0, 0), Pos::new(last, self.lines[last].len()))
        };
        self.replace(start, end, "", Kind::Other);
        let line = self.cursor.line.min(self.lines.len() - 1);
        self.cursor = Pos::new(line, 0);
    }

    /// The text Ctrl+C or Ctrl+X take: the selection, or the line with its
    /// newline.
    pub fn copy_text(&self) -> String {
        self.selected_text().unwrap_or_else(|| format!("{}\n", self.lines[self.cursor.line]))
    }

    pub fn cut(&mut self) -> String {
        let text = self.copy_text();
        if let Some((start, end)) = self.selection() {
            self.replace(start, end, "", Kind::Other);
        } else {
            self.delete_lines();
        }
        text
    }

    /// Tab with a selection over lines, or Shift+Tab: the lines in or out by
    /// one unit, as one step; the selection kept over them.
    pub fn indent_lines(&mut self, out: bool) {
        let (first, last) = self.line_span();
        let unit = self.indent.unit();
        let width = match self.indent {
            Indent::Tabs => self.tab_width,
            Indent::Spaces(n) => n,
        };
        let before = (self.cursor, self.anchor);
        let mut edits = Vec::new();
        let mut shifts: Vec<isize> = Vec::new();
        for line in first..=last {
            let text = self.lines[line].clone();
            if out {
                let remove = if text.starts_with('\t') {
                    1
                } else {
                    text.chars().take(width).take_while(|c| *c == ' ').count()
                };
                if remove > 0 {
                    let at = Pos::new(line, 0);
                    self.apply(at, Pos::new(line, remove), "");
                    edits.push(Edit { at, removed: text[..remove].to_string(), inserted: String::new() });
                }
                shifts.push(-(remove as isize));
            } else if !text.trim().is_empty() || first == last {
                let at = Pos::new(line, 0);
                self.apply(at, at, &unit);
                edits.push(Edit { at, removed: String::new(), inserted: unit.clone() });
                shifts.push(unit.len() as isize);
            } else {
                shifts.push(0);
            }
        }
        if edits.is_empty() {
            return;
        }
        let moved = |pos: Pos| {
            if pos.line < first || pos.line > last {
                return pos;
            }
            let shift = shifts[pos.line - first];
            Pos::new(pos.line, (pos.col as isize + shift).max(0) as usize)
        };
        self.cursor = self.clamp(moved(self.cursor));
        self.anchor = self.anchor.map(|anchor| self.clamp(moved(anchor)));
        self.undo.push(Step { edits, before, after: (self.cursor, self.anchor) });
        self.redo.clear();
        self.open = None;
    }

    /// Ctrl+/: each line commented with `prefix` (`// `), or all of them
    /// uncommented when every non-blank one already is.
    pub fn toggle_comment(&mut self, prefix: &str) {
        let (first, last) = self.line_span();
        let bare = prefix.trim_end();
        let lines: Vec<usize> = (first..=last).filter(|line| !self.lines[*line].trim().is_empty()).collect();
        if lines.is_empty() {
            return;
        }
        let commented = lines.iter().all(|line| self.lines[*line].trim_start().starts_with(bare));
        let column = lines.iter().map(|line| self.lines[*line].len() - self.lines[*line].trim_start().len()).min().unwrap_or(0);
        let before = (self.cursor, self.anchor);
        let mut edits = Vec::new();
        for line in lines {
            let text = self.lines[line].clone();
            if commented {
                let at = text.len() - text.trim_start().len();
                let length = if text[at..].starts_with(prefix) { prefix.len() } else { bare.len() };
                let pos = Pos::new(line, at);
                self.apply(pos, Pos::new(line, at + length), "");
                edits.push(Edit { at: pos, removed: text[at..at + length].to_string(), inserted: String::new() });
            } else {
                let pos = Pos::new(line, column);
                self.apply(pos, pos, prefix);
                edits.push(Edit { at: pos, removed: String::new(), inserted: prefix.to_string() });
            }
        }
        self.cursor = self.clamp(self.cursor);
        self.anchor = self.anchor.map(|anchor| self.clamp(anchor));
        self.undo.push(Step { edits, before, after: (self.cursor, self.anchor) });
        self.redo.clear();
        self.open = None;
    }

    /// Alt+↑/↓: the line (or the selected lines) moved one up or down.
    pub fn move_lines(&mut self, down: bool) {
        let (first, last) = self.line_span();
        if (down && last + 1 >= self.lines.len()) || (!down && first == 0) {
            return;
        }
        let before = (self.cursor, self.anchor);
        let block = self.lines[first..=last].join("\n");
        let (start, end, text, delta): (Pos, Pos, String, isize) = if down {
            let next = self.lines[last + 1].clone();
            (Pos::new(first, 0), Pos::new(last + 1, next.len()), format!("{next}\n{block}"), 1)
        } else {
            let previous = self.lines[first - 1].clone();
            (Pos::new(first - 1, 0), Pos::new(last, self.lines[last].len()), format!("{block}\n{previous}"), -1)
        };
        let removed = self.slice(start, end);
        self.apply(start, end, &text);
        let shift = |pos: Pos| Pos::new((pos.line as isize + delta) as usize, pos.col);
        self.cursor = self.clamp(shift(self.cursor));
        self.anchor = self.anchor.map(|anchor| self.clamp(shift(anchor)));
        self.undo.push(Step { edits: vec![Edit { at: start, removed, inserted: text }], before, after: (self.cursor, self.anchor) });
        self.redo.clear();
        self.open = None;
    }

    // -------------------------------------------------------------- undo

    pub fn undo(&mut self) -> bool {
        let Some(step) = self.undo.pop() else { return false };
        for edit in step.edits.iter().rev() {
            let end = end_of(edit.at, &edit.inserted);
            self.apply(edit.at, end, &edit.removed);
        }
        (self.cursor, self.anchor) = (self.clamp(step.before.0), step.before.1.map(|anchor| self.clamp(anchor)));
        self.goal = None;
        self.open = None;
        self.redo.push(step);
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(step) = self.redo.pop() else { return false };
        for edit in &step.edits {
            let end = end_of(edit.at, &edit.removed);
            self.apply(edit.at, end, &edit.inserted);
        }
        (self.cursor, self.anchor) = (self.clamp(step.after.0), step.after.1.map(|anchor| self.clamp(anchor)));
        self.goal = None;
        self.open = None;
        self.undo.push(step);
        true
    }

    /// The whole text replaced from outside (a reload, a merge), as one undo
    /// step; the cursor stays on the same line where it can. `theirs`: which
    /// of the new lines to mark as someone else's.
    pub fn replace_all(&mut self, text: &str, theirs: Vec<bool>) {
        let line = self.cursor.line;
        let col = self.cursor.col;
        let mut new = split_lines(text);
        if new.is_empty() {
            new.push(String::new());
        }
        let before = (self.cursor, self.anchor);
        let removed = self.lines.join("\n");
        let inserted = new.join("\n");
        let start = Pos::default();
        let end = self.end_pos();
        self.apply(start, end, &inserted);
        self.final_newline = text.ends_with('\n') || (text.is_empty() && self.final_newline);
        self.theirs = theirs;
        self.theirs.resize(self.lines.len(), false);
        self.cursor = self.clamp(Pos::new(line, col));
        self.anchor = None;
        self.undo.push(Step { edits: vec![Edit { at: start, removed, inserted }], before, after: (self.cursor, None) });
        self.redo.clear();
        self.open = None;
    }

    /// Nothing typed since is joined to the step before (a save, a move).
    pub fn seal(&mut self) {
        self.open = None;
    }

    // -------------------------------------------------------------- find

    /// The next match of `query` after `from` (or at it, `inclusive`),
    /// before it going back, wrapping round; case-insensitive unless the
    /// query has a capital.
    pub fn find(&self, query: &str, from: Pos, forward: bool, inclusive: bool) -> Option<(Pos, Pos)> {
        if query.is_empty() || query.contains('\n') {
            return None;
        }
        let fold = !query.chars().any(char::is_uppercase);
        let needle = if fold { query.to_lowercase() } else { query.to_string() };
        let in_line = |line: usize| -> Vec<usize> {
            let text = &self.lines[line];
            let hay = if fold { lower_same_len(text) } else { text.clone() };
            hay.match_indices(&needle).map(|(at, _)| at).filter(|at| text.is_char_boundary(*at) && text.is_char_boundary(at + needle.len())).collect()
        };
        let count = self.lines.len();
        let found = |line: usize, at: usize| (Pos::new(line, at), Pos::new(line, at + needle.len()));
        for step in 0..=count {
            let (line, wrapped) = if forward {
                ((from.line + step) % count, step == count)
            } else {
                ((from.line + count - step % count) % count, step == count)
            };
            let all = in_line(line);
            let col = from.col;
            let pick = |at: &usize| -> bool {
                match (step == 0, wrapped, forward) {
                    (true, _, true) => *at > col || (inclusive && *at == col),
                    (true, _, false) => *at < col,
                    (false, true, true) => *at <= col,
                    (false, true, false) => *at >= col,
                    _ => true,
                }
            };
            let hit = if forward { all.iter().find(|at| pick(at)) } else { all.iter().rev().find(|at| pick(at)) };
            if let Some(at) = hit {
                return Some(found(line, *at));
            }
        }
        None
    }

    /// Every match of `query`, in order (for drawing them and counting).
    pub fn matches(&self, query: &str, limit: usize) -> Vec<(Pos, Pos)> {
        if query.is_empty() || query.contains('\n') {
            return Vec::new();
        }
        let fold = !query.chars().any(char::is_uppercase);
        let needle = if fold { query.to_lowercase() } else { query.to_string() };
        let mut out = Vec::new();
        for (line, text) in self.lines.iter().enumerate() {
            let hay = if fold { lower_same_len(text) } else { text.clone() };
            for (at, _) in hay.match_indices(&needle) {
                if text.is_char_boundary(at) && text.is_char_boundary(at + needle.len()) {
                    out.push((Pos::new(line, at), Pos::new(line, at + needle.len())));
                    if out.len() >= limit {
                        return out;
                    }
                }
            }
        }
        out
    }

    /// Every match replaced, as one step; how many.
    pub fn replace_matches(&mut self, query: &str, with: &str) -> usize {
        let found = self.matches(query, usize::MAX);
        if found.is_empty() {
            return 0;
        }
        let before = (self.cursor, self.anchor);
        let mut edits = Vec::new();
        for (start, end) in found.iter().rev() {
            let removed = self.slice(*start, *end);
            self.apply(*start, *end, with);
            edits.push(Edit { at: *start, removed, inserted: with.to_string() });
        }
        self.cursor = self.clamp(self.cursor);
        self.anchor = None;
        self.undo.push(Step { edits, before, after: (self.cursor, None) });
        self.redo.clear();
        self.open = None;
        found.len()
    }
}

/// Lowercased with the same byte offsets (characters whose lowercase has
/// another length are left as they are).
fn lower_same_len(text: &str) -> String {
    text.chars()
        .map(|c| {
            let mut lower = c.to_lowercase();
            match (lower.next(), lower.next()) {
                (Some(l), None) if l.len_utf8() == c.len_utf8() => l,
                _ => c,
            }
        })
        .collect()
}

/// Where `text` inserted at `at` ends.
fn end_of(at: Pos, text: &str) -> Pos {
    match text.rsplit_once('\n') {
        Some((before, last)) => Pos::new(at.line + before.matches('\n').count() + 1, last.len()),
        None => Pos::new(at.line, at.col + text.len()),
    }
}

/// The text's lines, without endings (a final newline adds no line).
pub fn split_lines(text: &str) -> Vec<String> {
    let body = text.strip_suffix('\n').unwrap_or(text);
    let body = if text.ends_with('\n') { body.strip_suffix('\r').unwrap_or(body) } else { body };
    let mut lines: Vec<String> = body.split('\n').map(|line| line.strip_suffix('\r').unwrap_or(line).to_string()).collect();
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Tabs, or the most common step of leading spaces, when the file says.
fn detect_indent(lines: &[String]) -> Option<Indent> {
    let mut tabs = 0;
    let mut steps = [0usize; 9];
    let mut previous = 0;
    for line in lines.iter().filter(|line| !line.trim().is_empty()).take(2_000) {
        if line.starts_with('\t') {
            tabs += 1;
            continue;
        }
        let spaces = line.len() - line.trim_start_matches(' ').len();
        let step = spaces.abs_diff(previous);
        if (2..=8).contains(&step) {
            steps[step] += 1;
        }
        previous = spaces;
    }
    let (step, count) = steps.iter().enumerate().max_by_key(|(_, count)| **count)?;
    if tabs > *count && tabs > 0 {
        return Some(Indent::Tabs);
    }
    (*count > 0).then_some(Indent::Spaces(step))
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn prev_boundary(text: &str, col: usize) -> usize {
    text[..col].char_indices().next_back().map_or(0, |(index, _)| index)
}

fn next_boundary(text: &str, col: usize) -> usize {
    text[col..].chars().next().map_or(col, |c| col + c.len_utf8())
}

/// How wide `c` is drawn at display column `at`: a tab to the next stop, a
/// control character as `^X`.
pub fn char_width(c: char, at: usize, tab_width: usize) -> usize {
    match c {
        '\t' => tab_width - at % tab_width,
        c if c.is_control() => 2,
        c => c.width().unwrap_or(0),
    }
}

pub fn display_width(text: &str, tab_width: usize) -> usize {
    let mut width = 0;
    for c in text.chars() {
        width += char_width(c, width, tab_width);
    }
    width
}

// ------------------------------------------------------------------ the three-way merge

/// A line of a merge: the same in all three or taken from one side, or a
/// stretch where the two sides changed the same lines differently.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Merged {
    Line { text: String, theirs: bool },
    Conflict { theirs: Vec<String>, mine: Vec<String> },
}

/// `base` changed into `mine` here and `theirs` there: both sets of changes,
/// line by line. Changes that touch the same lines of `base` differently
/// are a conflict.
pub fn merge3(base: &[String], mine: &[String], theirs: &[String]) -> Vec<Merged> {
    let ours = hunks(base, mine);
    let other = hunks(base, theirs);
    let mut out = Vec::new();
    let (mut i, mut a, mut b) = (0usize, 0usize, 0usize);
    loop {
        let next_a = ours.get(a);
        let next_b = other.get(b);
        let start = match (next_a, next_b) {
            (None, None) => base.len(),
            (Some(h), None) => h.start,
            (None, Some(h)) => h.start,
            (Some(x), Some(y)) => x.start.min(y.start),
        };
        // The unchanged stretch before the next hunk.
        while i < start {
            out.push(Merged::Line { text: base[i].clone(), theirs: false });
            i += 1;
        }
        if next_a.is_none() && next_b.is_none() {
            break;
        }
        // The first hunk, then every hunk of either side that overlaps the
        // region it makes.
        let (mut take_a, mut take_b) = (a, b);
        let mut end = match (next_a, next_b) {
            (Some(h), Some(other)) if h.start <= other.start => {
                take_a += 1;
                h.end
            }
            (Some(h), None) => {
                take_a += 1;
                h.end
            }
            (_, Some(h)) => {
                take_b += 1;
                h.end
            }
            (None, None) => unreachable!(),
        };
        loop {
            let mut grew = false;
            while let Some(h) = ours.get(take_a).filter(|h| overlaps(h, start, end)) {
                end = end.max(h.end);
                take_a += 1;
                grew = true;
            }
            while let Some(h) = other.get(take_b).filter(|h| overlaps(h, start, end)) {
                end = end.max(h.end);
                take_b += 1;
                grew = true;
            }
            if !grew {
                break;
            }
        }
        let mine_part = rebuild(base, &ours[a..take_a], start, end);
        let theirs_part = rebuild(base, &other[b..take_b], start, end);
        if take_b == b {
            out.extend(mine_part.into_iter().map(|text| Merged::Line { text, theirs: false }));
        } else if take_a == a || mine_part == theirs_part {
            out.extend(theirs_part.into_iter().map(|text| Merged::Line { text, theirs: true }));
        } else {
            out.push(Merged::Conflict { theirs: theirs_part, mine: mine_part });
        }
        i = end;
        a = take_a;
        b = take_b;
    }
    out
}

/// A change of `base`'s lines `start..end` into `lines`.
#[derive(Clone, Debug)]
struct Hunk {
    start: usize,
    end: usize,
    lines: Vec<String>,
}

/// Whether hunk `h` overlaps the region `start..end` of `base`. Changes of
/// neighbouring lines don't (the agent editing the line under yours is no clash);
/// an insertion clashes with another at the same place, or with a change of
/// the lines it would sit among.
fn overlaps(h: &Hunk, start: usize, end: usize) -> bool {
    match (h.start == h.end, start == end) {
        (true, true) => h.start == start,
        (true, false) => h.start >= start && h.start < end,
        (false, true) => h.start <= start && h.end > start,
        (false, false) => h.start < end && h.end > start,
    }
}

fn hunks(base: &[String], changed: &[String]) -> Vec<Hunk> {
    let ops = similar::capture_diff_slices(similar::Algorithm::Myers, base, changed);
    let mut out: Vec<Hunk> = Vec::new();
    for op in &ops {
        let (old, new) = (op.old_range(), op.new_range());
        if matches!(op.tag(), similar::DiffTag::Equal) {
            continue;
        }
        let lines = changed[new].to_vec();
        match out.last_mut() {
            Some(last) if last.end == old.start => {
                last.end = old.end;
                last.lines.extend(lines);
            }
            _ => out.push(Hunk { start: old.start, end: old.end, lines }),
        }
    }
    out
}

/// `base[start..end]` with these hunks applied.
fn rebuild(base: &[String], hunks: &[Hunk], start: usize, end: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = start;
    for h in hunks {
        while i < h.start {
            out.push(base[i].clone());
            i += 1;
        }
        out.extend(h.lines.iter().cloned());
        i = h.end;
    }
    while i < end {
        out.push(base[i].clone());
        i += 1;
    }
    out
}

/// Which lines of `after` are new or changed against `before` (for the
/// gutter), and where lines were taken out (the line after the gap).
pub fn changed_lines(before: &[String], after: &[String]) -> (Vec<bool>, Vec<bool>) {
    let mut changed = vec![false; after.len()];
    let mut removed = vec![false; after.len() + 1];
    let ops = similar::capture_diff_slices(similar::Algorithm::Myers, before, after);
    for op in &ops {
        match op.tag() {
            similar::DiffTag::Equal => {}
            similar::DiffTag::Delete => removed[op.new_range().start] = true,
            similar::DiffTag::Insert | similar::DiffTag::Replace => {
                for line in op.new_range() {
                    changed[line] = true;
                }
            }
        }
    }
    (changed, removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(text: &str) -> Buffer {
        Buffer::from_text(text, 4, Indent::Spaces(4))
    }

    fn lines(text: &str) -> Vec<String> {
        split_lines(text)
    }

    #[test]
    fn undo_groups_typing_and_redo_puts_it_back() {
        let mut b = buffer("");
        for c in "hello there".chars() {
            b.insert(&c.to_string());
        }
        assert_eq!(b.text(), "hello there");
        b.undo();
        assert_eq!(b.text(), "hello ", "a word at a time");
        b.undo();
        assert_eq!(b.text(), "");
        b.redo();
        b.redo();
        assert_eq!(b.text(), "hello there");
        b.home(false);
        b.insert("X");
        assert!(b.redo.is_empty(), "a new edit drops the redo");
        b.undo();
        assert_eq!(b.text(), "hello there");
        // A paste is one step.
        b.end(false);
        b.paste("\nline\nmore");
        assert_eq!(b.text(), "hello there\nline\nmore");
        b.undo();
        assert_eq!(b.text(), "hello there");
        assert_eq!(b.cursor, Pos::new(0, 11), "the cursor where it was");
    }

    #[test]
    fn a_merge_keeps_both_sides_and_finds_conflicts() {
        let base = lines("a\nb\nc\nd\ne\n");
        let mine = lines("a\nB\nc\nd\ne\n");
        let theirs = lines("a\nb\nc\nd\nE\nf\n");
        let merged = merge3(&base, &mine, &theirs);
        let text: Vec<String> = merged.iter().map(|m| match m { Merged::Line { text, .. } => text.clone(), _ => "!".into() }).collect();
        assert_eq!(text, ["a", "B", "c", "d", "E", "f"]);
        let theirs_marks: Vec<bool> = merged.iter().map(|m| matches!(m, Merged::Line { theirs: true, .. })).collect();
        assert_eq!(theirs_marks, [false, false, false, false, true, true]);

        let clash = merge3(&base, &lines("a\nX\nc\nd\ne\n"), &lines("a\nY\nc\nd\ne\n"));
        assert!(clash.contains(&Merged::Conflict { theirs: vec!["Y".into()], mine: vec!["X".into()] }), "{clash:?}");
        let same = merge3(&base, &lines("a\nZ\nc\nd\ne\n"), &lines("a\nZ\nc\nd\ne\n"));
        assert!(same.iter().all(|m| matches!(m, Merged::Line { .. })), "the same change on both sides is no conflict");
        // Insertions at different places, and at the end.
        let inserts = merge3(&lines("a\nb\n"), &lines("x\na\nb\n"), &lines("a\nb\ny\n"));
        let text: Vec<String> = inserts.iter().filter_map(|m| match m { Merged::Line { text, .. } => Some(text.clone()), _ => None }).collect();
        assert_eq!(text, ["x", "a", "b", "y"]);
        // Both inserting at the same place differently: a conflict.
        let both = merge3(&lines("a\nb\n"), &lines("a\nm\nb\n"), &lines("a\nt\nb\n"));
        assert!(both.iter().any(|m| matches!(m, Merged::Conflict { .. })), "{both:?}");
        // One side deleting a line the other left alone.
        let deleted = merge3(&base, &lines("a\nc\nd\ne\n"), &lines("a\nb\nc\nd\ne\nf\n"));
        let text: Vec<String> = deleted.iter().filter_map(|m| match m { Merged::Line { text, .. } => Some(text.clone()), _ => None }).collect();
        assert_eq!(text, ["a", "c", "d", "e", "f"]);
        // Neighbouring lines changed on each side: no clash.
        let next = merge3(&base, &lines("a\nB\nc\nd\ne\n"), &lines("a\nb\nC\nd\ne\n"));
        let text: Vec<String> = next.iter().filter_map(|m| match m { Merged::Line { text, .. } => Some(text.clone()), _ => None }).collect();
        assert_eq!(text, ["a", "B", "C", "d", "e"], "{next:?}");
    }
}
