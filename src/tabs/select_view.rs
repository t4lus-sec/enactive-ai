//! Selecting text with the mouse in any tab but the editor (which has its
//! own), a submodule of `tabs.rs`, and copying it when the button is let go.
//!
//! A drag selects, from where the button went down. Dragged onto the top or
//! the bottom row of the tab, the tab scrolls that way a few rows a move, and
//! the wheel while the button is held scrolls it too: the selection goes on
//! past the screen, as far as the tab scrolls. Let go, and what is selected
//! goes to the clipboard (`clipboard::copy`), and the footer says how.
//!
//! What the tab shows scrolls by its own rules (the conversation and the
//! shells by rows, the lists by their selection), so the selection keeps the
//! text it has seen rather than a place in the tab: every frame drawn during
//! the drag is lined up with the text kept so far by where its rows match,
//! and adds what scrolled into view above or below it.

use crate::style::{Line, Style};
use unicode_width::UnicodeWidthChar;

/// A place in the kept text: its line, and the display column in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct At {
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Default)]
pub(super) struct Selection {
    /// The text seen so far, top to bottom, one line a row.
    lines: Vec<String>,
    /// The line in `lines` the tab's first body row shows.
    top: usize,
    /// Where it began, and where the mouse is now, in `lines`; the anchor is
    /// placed when the first frame is lined up.
    anchor: Option<At>,
    cursor: Option<At>,
    /// Where the button went down, and where the mouse is, on the screen:
    /// body row and column.
    pub pressed: (usize, usize),
    pub mouse: (usize, usize),
    /// Rows the tab was just asked to scroll (negative: up, to earlier
    /// text): how the next frame is expected to sit, when its rows don't say.
    expected: isize,
    /// The button is still down.
    pub dragging: bool,
}

impl Selection {
    pub fn new(pressed: (usize, usize)) -> Self {
        Self { pressed, mouse: pressed, dragging: true, ..Self::default() }
    }

    /// The tab was asked to scroll `rows` (negative: up).
    pub fn scrolled(&mut self, rows: isize) {
        self.expected += rows;
    }

    /// A frame's body rows, as text: lined up with what was kept, merged in,
    /// and the anchor and cursor put where the screen says.
    pub fn take_frame(&mut self, rows: &[String]) {
        if rows.is_empty() {
            return;
        }
        if self.lines.is_empty() {
            self.lines = rows.to_vec();
            self.top = 0;
        } else {
            let shift = self.offset(rows);
            let mut top = self.top as isize + shift;
            if top < 0 {
                // Earlier text came into view: it goes in front, and
                // everything kept moves down by as much.
                let added = (-top) as usize;
                let mut front: Vec<String> = rows[..added.min(rows.len())].to_vec();
                front.resize(added, String::new());
                self.lines.splice(0..0, front);
                for at in [&mut self.anchor, &mut self.cursor].into_iter().flatten() {
                    at.line += added;
                }
                top = 0;
            }
            self.top = top as usize;
            for (i, row) in rows.iter().enumerate() {
                let line = self.top + i;
                if line < self.lines.len() {
                    self.lines[line] = row.clone();
                } else {
                    self.lines.push(row.clone());
                }
            }
        }
        self.expected = 0;
        let at = |(row, column): (usize, usize), top: usize| At { line: top + row.min(rows.len() - 1), column };
        if self.anchor.is_none() {
            self.anchor = Some(at(self.pressed, self.top));
        }
        if self.dragging {
            self.cursor = Some(at(self.mouse, self.top));
        }
    }

    /// How far `rows` sit from where the last frame did, in lines: where the
    /// most non-blank rows match, nearest the scroll asked for first; the
    /// scroll asked for when none do.
    fn offset(&self, rows: &[String]) -> isize {
        let n = rows.len() as isize;
        let mut best: Option<(usize, isize)> = None;
        let mut shifts: Vec<isize> = (-n - self.expected.abs()..=n + self.expected.abs()).collect();
        shifts.sort_by_key(|shift| ((shift - self.expected).abs(), *shift));
        for shift in shifts {
            let mut score = 0;
            for (i, row) in rows.iter().enumerate() {
                let line = self.top as isize + shift + i as isize;
                if row.trim().is_empty() || line < 0 {
                    continue;
                }
                if self.lines.get(line as usize) == Some(row) {
                    score += 1;
                }
            }
            if score > 0 && best.is_none_or(|(most, _)| score > most) {
                best = Some((score, shift));
            }
        }
        let shown = rows.iter().filter(|row| !row.trim().is_empty()).count();
        match best {
            // Enough of the frame agrees to trust it.
            Some((score, shift)) if score * 2 >= shown.clamp(1, 4) => shift,
            _ => self.expected,
        }
    }

    /// The selected span, in order.
    fn span(&self) -> Option<(At, At)> {
        let (a, b) = (self.anchor?, self.cursor?);
        (a != b).then(|| if a <= b { (a, b) } else { (b, a) })
    }

    /// The selected text: whole lines between the ends, the first from its
    /// column, the last to its (the column under the mouse included), each
    /// without trailing spaces.
    pub fn text(&self) -> Option<String> {
        let (start, end) = self.span()?;
        let mut out = Vec::new();
        for line in start.line..=end.line.min(self.lines.len().saturating_sub(1)) {
            let text = &self.lines[line];
            let from = if line == start.line { start.column } else { 0 };
            let to = if line == end.line { end.column + 1 } else { usize::MAX };
            out.push(columns(text, from, to).trim_end().to_string());
        }
        // Begun past the end of its first line: that line holds nothing.
        if out.len() > 1 && out[0].is_empty() {
            out.remove(0);
        }
        let text = out.join("\n");
        (!text.trim().is_empty()).then_some(text)
    }

    /// The body rows on screen with the selection shown inverted. A row it
    /// touches is drawn as plain text (its colours give way to the mark).
    pub fn mark(&self, body: &mut [Line]) {
        let Some((start, end)) = self.span() else { return };
        for (i, row) in body.iter_mut().enumerate() {
            let line = self.top + i;
            if line < start.line || line > end.line {
                continue;
            }
            let text: String = row.iter().map(|(_, text)| super::plain_text(text)).collect();
            let from = if line == start.line { start.column } else { 0 };
            let to = if line == end.line { end.column + 1 } else { usize::MAX };
            let width: usize = text.chars().map(|c| c.width().unwrap_or(0)).sum();
            let before = columns(&text, 0, from);
            let mut marked = columns(&text, from, to);
            // A row selected to its end shows it to the edge it reaches.
            if to == usize::MAX && marked.is_empty() && width <= from {
                marked = " ".into();
            }
            let after = columns(&text, to, usize::MAX);
            *row = vec![(Style::Plain, before), (Style::Caret, marked), (Style::Plain, after)];
        }
    }
}

/// The part of `text` from display column `from` up to (not including) `to`.
pub(super) fn columns(text: &str, from: usize, to: usize) -> String {
    let mut out = String::new();
    let mut at = 0;
    for c in text.chars() {
        let width = c.width().unwrap_or(0);
        if at >= from && at < to {
            out.push(c);
        }
        at += width;
        if at >= to {
            break;
        }
    }
    out
}

