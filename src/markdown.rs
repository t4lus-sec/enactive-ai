//! Streaming Markdown → ANSI renderer for the transcript.
//!
//! Model output arrives a few tokens at a time and is written straight to the
//! terminal at s0ma's saved output cursor, so this renderer can never go back
//! and restyle text it has already emitted. Instead it holds back only the
//! characters that are still ambiguous:
//!
//! - the first few characters of a line, until it's clear whether the line is
//!   a heading, bullet, quote, rule, code fence or table row;
//! - a run of `*`, `_` or `~`, until the next character says whether it
//!   opens, closes or is literal;
//! - a `[link](url)`, until the `)` arrives (or it turns out not to be a link);
//! - table rows, until the table ends, because columns can't be aligned
//!   before every row is known.
//!
//! Tables are laid out to fit the terminal width the caller sets (see
//! `set_width`): columns shrink, widest first, and long cells wrap within
//! their column, so a wide table never spills past the edge and breaks its
//! borders. On a screen too narrow for that, each row is shown as a record.
//!
//! Everything else streams through immediately.
//!
//! Input MUST already have passed through `terminal::safe_text`. This module
//! only ever adds SGR colour/style sequences of its own, so the guarantee that
//! model text cannot move the cursor or erase output still holds.

use unicode_width::UnicodeWidthChar;

const RESET: &str = "\x1b[0m";

// Styles, as SGR parameter lists (without the leading "0;").
const BODY: &str = "37";
// CRT phosphor green (#33FF33) as 24-bit colour, so terminal themes can't
// remap it the way they remap the 16 basic colours.
const HEADING: &str = "1;38;2;51;255;51";
const HEADING_1: &str = "1;4;38;2;51;255;51";
const QUOTE: &str = "3;90";
const CODE: &str = "36";
const DECO: &str = "90";
const BULLET: &str = "1;96";
const LINK: &str = "4;94";

const MAX_MARKER: usize = 16;
const MAX_LINK_TEXT: usize = 256;
const MAX_LINK_URL: usize = 1024;
const RULE_WIDTH: usize = 40;
/// A column is never narrower than this (or its content, if shorter).
const MIN_COLUMN: usize = 5;
/// A column gets room for its longest word, so wrapping never splits words,
/// unless that word is longer than this (a URL, a hash), which is broken.
/// A table that cannot give every column that much is shown as records.
const MAX_UNBROKEN_WORD: usize = 15;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LineKind {
    Body,
    Heading(u8),
    Quote,
    FenceInfo,
    Code,
    /// The remainder of a closing-fence line; dropped.
    Hidden,
}

enum Link {
    None,
    Text(String),
    /// Saw `]`, waiting to see whether `(` follows.
    Bracket(String),
    Url(String, String),
}

pub struct MarkdownStream {
    out: String,
    current_style: String,

    inline_only: bool,
    finishing: bool,

    // Line-start classification buffer.
    at_line_start: bool,
    indent: String,
    marker: String,

    line_kind: LineKind,
    fence: Option<(char, usize)>,

    // Inline state.
    bold: bool,
    italic: bool,
    strike: bool,
    code: bool,
    escape: bool,
    delim: Option<(char, usize)>,
    prev: Option<char>,
    link: Link,

    // Table buffering.
    table: Vec<String>,
    table_row: Option<String>,

    /// Terminal columns tables must fit in; `None` lays them out at their
    /// natural width.
    width: Option<usize>,
    /// Whether anything has been output since the stream (re)started.
    emitted: bool,
}

impl Default for MarkdownStream {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownStream {
    pub fn new() -> Self {
        Self::with_mode(false)
    }

    /// No block-level parsing: used for table cells.
    fn inline_only() -> Self {
        Self::with_mode(true)
    }

    fn with_mode(inline_only: bool) -> Self {
        Self {
            out: String::new(),
            current_style: String::new(),
            inline_only,
            finishing: false,
            at_line_start: !inline_only,
            indent: String::new(),
            marker: String::new(),
            line_kind: LineKind::Body,
            fence: None,
            bold: false,
            italic: false,
            strike: false,
            code: false,
            escape: false,
            delim: None,
            prev: None,
            link: Link::None,
            table: Vec::new(),
            table_row: None,
            width: None,
            emitted: false,
        }
    }

    /// The terminal's width in columns, so tables can be laid out to fit.
    /// Takes effect from the next table, and survives `finish`.
    pub fn set_width(&mut self, columns: Option<usize>) {
        self.width = columns.filter(|columns| *columns > 0);
    }

    /// Feed a sanitized chunk; returns styled text ready to write (possibly
    /// empty if everything in the chunk is still being held back).
    pub fn push(&mut self, text: &str) -> String {
        for c in text.chars() {
            self.feed(c);
        }
        self.take_output()
    }

    /// End of a reply (or a tool boundary / interruption): emit anything still
    /// held back and reset to a fresh state.
    pub fn finish(&mut self) -> String {
        self.finishing = true;
        if let Some(row) = self.table_row.take() {
            self.table.push(row);
        } else if self.at_line_start {
            if !self.marker.is_empty() || !self.indent.is_empty() {
                self.classify('\n');
            }
        } else {
            self.end_line();
        }
        self.flush_table();
        let out = self.take_output();
        let width = self.width;
        *self = Self::with_mode(self.inline_only);
        self.width = width;
        out
    }

    fn take_output(&mut self) -> String {
        let mut out = std::mem::take(&mut self.out);
        if !out.is_empty() {
            out.push_str(RESET);
            self.emitted = true;
        }
        // The caller may interleave other output between chunks, so the next
        // chunk must re-assert its style.
        self.current_style.clear();
        out
    }

    // ------------------------------------------------------------------
    // Character dispatch
    // ------------------------------------------------------------------

    fn feed(&mut self, c: char) {
        if let Some(row) = self.table_row.as_mut() {
            if c == '\n' {
                let row = self.table_row.take().unwrap();
                self.table.push(row);
                self.at_line_start = true;
            } else {
                row.push(c);
            }
            return;
        }

        if self.at_line_start {
            if c == '\n' {
                self.classify(c);
            } else if c == ' ' && self.marker.is_empty() {
                self.indent.push(c);
            } else if self.is_marker_char(c) && self.marker.len() < MAX_MARKER {
                self.marker.push(c);
            } else {
                self.classify(c);
            }
            return;
        }

        if c == '\n' {
            self.end_line();
        } else {
            self.inline(c);
        }
    }

    fn is_marker_char(&self, c: char) -> bool {
        if self.fence.is_some() {
            matches!(c, '`' | '~')
        } else {
            matches!(
                c,
                '#' | '-' | '*' | '+' | '>' | '`' | '~' | '_' | '|' | '.' | ')' | '0'..='9'
            )
        }
    }

    // ------------------------------------------------------------------
    // Block-level classification
    // ------------------------------------------------------------------

    /// Called with the first character that settles what kind of line this is
    /// (`next` may be '\n').
    fn classify(&mut self, next: char) {
        let indent = std::mem::take(&mut self.indent);
        let marker = std::mem::take(&mut self.marker);
        self.at_line_start = false;

        // Inside a fenced code block, the only thing that matters is whether
        // this line closes it.
        if let Some((fence_char, fence_len)) = self.fence {
            let closes = marker.chars().count() >= fence_len
                && marker.chars().all(|c| c == fence_char)
                && next.is_whitespace();
            if closes {
                self.fence = None;
                self.emit_deco(DECO, &format!("{indent}└─"));
                if next == '\n' {
                    self.end_line();
                } else {
                    self.line_kind = LineKind::Hidden;
                }
                return;
            }
            // No gutter on code lines: the header and footer mark the block,
            // and the colour sets the code apart, so selecting lines with the
            // mouse copies exactly the code and nothing of the frame.
            self.line_kind = LineKind::Code;
            self.emit_str(&indent);
            self.emit_str(&marker);
            self.after_marker(next);
            return;
        }

        if marker.starts_with('|') {
            let mut row = indent;
            row.push_str(&marker);
            if next == '\n' {
                self.table.push(row);
                self.at_line_start = true;
            } else {
                row.push(next);
                self.table_row = Some(row);
            }
            return;
        }
        self.flush_table();

        let count = marker.chars().count();
        let first = marker.chars().next();
        let uniform = first.is_some_and(|f| marker.chars().all(|c| c == f));

        // Horizontal rule: a whole line of ---, *** or ___
        if next == '\n' && count >= 3 && uniform && matches!(first, Some('-' | '*' | '_')) {
            self.emit_deco(DECO, &format!("{indent}{}", "─".repeat(RULE_WIDTH)));
            self.end_line();
            return;
        }

        // Code fence: ``` or ~~~ (optionally followed by a language)
        if count >= 3 && uniform && matches!(first, Some('`' | '~')) {
            self.fence = Some((first.unwrap(), count));
            self.emit_deco(DECO, &format!("{indent}┌─"));
            if next == '\n' {
                self.end_line();
            } else {
                self.line_kind = LineKind::FenceInfo;
                self.emit_str(" ");
                if next != ' ' {
                    self.emit_char(next);
                }
            }
            return;
        }

        if next == ' ' {
            // Heading
            if (1..=6).contains(&count) && uniform && first == Some('#') {
                self.line_kind = LineKind::Heading(count as u8);
                self.emit_str(&indent);
                self.prev = None;
                return;
            }
            // Bullet
            if matches!(marker.as_str(), "-" | "*" | "+") {
                self.emit_str(&indent);
                self.emit_deco(BULLET, "•");
                self.emit_str(" ");
                self.prev = None;
                return;
            }
            // Numbered item: 1. or 1)
            if is_ordered_marker(&marker) {
                self.emit_str(&indent);
                self.emit_deco(BULLET, &marker);
                self.emit_str(" ");
                self.prev = None;
                return;
            }
        }

        // Block quote (one or more '>')
        if count >= 1 && uniform && first == Some('>') {
            self.line_kind = LineKind::Quote;
            self.emit_str(&indent);
            self.emit_deco(DECO, &"│ ".repeat(count));
            self.prev = None;
            if next != ' ' {
                self.after_marker(next);
            }
            return;
        }

        // Ordinary paragraph text: the buffered marker was just text.
        self.line_kind = LineKind::Body;
        self.emit_str(&indent);
        for c in marker.chars() {
            self.inline(c);
        }
        self.after_marker(next);
    }

    fn after_marker(&mut self, next: char) {
        if next == '\n' {
            self.end_line();
        } else {
            self.inline(next);
        }
    }

    fn end_line(&mut self) {
        self.resolve_delim('\n');
        self.abort_link();
        if self.escape {
            self.escape = false;
            self.emit_char('\\');
        }
        // Inline styles never leak past a line: an unmatched `*` must not turn
        // the rest of the reply italic.
        self.bold = false;
        self.italic = false;
        self.strike = false;
        self.code = false;
        self.prev = None;

        if !self.finishing {
            self.out.push_str(RESET);
            self.current_style.clear();
            self.out.push('\n');
        }

        self.line_kind = if self.fence.is_some() {
            LineKind::Code
        } else {
            LineKind::Body
        };
        self.at_line_start = !self.inline_only;
    }

    // ------------------------------------------------------------------
    // Inline parsing
    // ------------------------------------------------------------------

    fn inline(&mut self, c: char) {
        match self.line_kind {
            LineKind::Hidden => return,
            LineKind::Code | LineKind::FenceInfo => {
                self.emit_char(c);
                return;
            }
            _ => {}
        }

        if self.code {
            if c == '`' {
                self.code = false;
                self.prev = Some(c);
            } else {
                self.emit_char(c);
            }
            return;
        }

        if self.capture_link(c) {
            return;
        }

        if let Some((d, n)) = self.delim {
            if c == d {
                self.delim = Some((d, n + 1));
                return;
            }
            self.resolve_delim(c);
        }

        if self.escape {
            self.escape = false;
            if !c.is_ascii_punctuation() {
                // Not an escape (e.g. a Windows path); keep the backslash.
                self.emit_char('\\');
            }
            self.emit_char(c);
            return;
        }

        match c {
            '\\' => self.escape = true,
            '*' | '_' | '~' => self.delim = Some((c, 1)),
            '`' => {
                self.code = true;
                self.prev = Some(c);
            }
            '[' => self.link = Link::Text(String::new()),
            _ => self.emit_char(c),
        }
    }

    /// Returns true if `c` was consumed by link capture.
    fn capture_link(&mut self, c: char) -> bool {
        match std::mem::replace(&mut self.link, Link::None) {
            Link::None => false,
            Link::Text(mut text) => {
                if c == ']' {
                    self.link = Link::Bracket(text);
                } else if text.len() >= MAX_LINK_TEXT || c == '[' {
                    self.link = Link::Text(text);
                    self.abort_link();
                    self.inline(c);
                } else {
                    text.push(c);
                    self.link = Link::Text(text);
                }
                true
            }
            Link::Bracket(text) => {
                if c == '(' {
                    self.link = Link::Url(text, String::new());
                } else {
                    self.link = Link::Bracket(text);
                    self.abort_link();
                    self.inline(c);
                }
                true
            }
            Link::Url(text, mut url) => {
                if c == ')' {
                    self.render_link(&text, &url);
                } else if c == ' ' || url.len() >= MAX_LINK_URL {
                    self.link = Link::Url(text, url);
                    self.abort_link();
                    self.inline(c);
                } else {
                    url.push(c);
                    self.link = Link::Url(text, url);
                }
                true
            }
        }
    }

    fn render_link(&mut self, text: &str, url: &str) {
        if text.is_empty() || text == url {
            self.emit_deco(LINK, url);
        } else {
            self.emit_deco(LINK, text);
            self.emit_deco(DECO, &format!(" ({url})"));
        }
        self.prev = Some(')');
    }

    /// The bracketed text turned out not to be a link; print it as it was.
    fn abort_link(&mut self) {
        let raw = match std::mem::replace(&mut self.link, Link::None) {
            Link::None => return,
            Link::Text(text) => format!("[{text}"),
            Link::Bracket(text) => format!("[{text}]"),
            Link::Url(text, url) => format!("[{text}]({url}"),
        };
        self.emit_str(&raw);
    }

    /// Decide what a run of `*`, `_` or `~` means, now that the following
    /// character is known.
    fn resolve_delim(&mut self, next: char) {
        let Some((d, n)) = self.delim.take() else {
            return;
        };
        let boundary = |c: Option<char>| {
            c.is_none_or(|c| c.is_whitespace() || c.is_ascii_punctuation())
        };
        let can_open = boundary(self.prev) && !next.is_whitespace();
        let can_close = self.prev.is_some_and(|p| !p.is_whitespace())
            && (d != '_' || boundary(Some(next)));

        // Strikethrough is exactly `~~`. A single `~` stays literal because
        // it usually means "approximately" (~5 min) or a home path (~/src).
        if d == '~' {
            if n == 2 && self.strike && can_close {
                self.strike = false;
                self.prev = Some(d);
            } else if n == 2 && !self.strike && can_open {
                self.strike = true;
                self.prev = Some(d);
            } else {
                self.emit_str(&"~".repeat(n));
            }
            return;
        }

        let (want_bold, want_italic) = match n {
            1 => (false, true),
            2 => (true, false),
            3 => (true, true),
            _ => {
                self.emit_str(&d.to_string().repeat(n));
                return;
            }
        };
        let all_on = (!want_bold || self.bold) && (!want_italic || self.italic);
        let all_off = (!want_bold || !self.bold) && (!want_italic || !self.italic);

        if all_on && can_close {
            if want_bold {
                self.bold = false;
            }
            if want_italic {
                self.italic = false;
            }
            self.prev = Some(d);
        } else if all_off && can_open {
            if want_bold {
                self.bold = true;
            }
            if want_italic {
                self.italic = true;
            }
            self.prev = Some(d);
        } else {
            self.emit_str(&d.to_string().repeat(n));
        }
    }

    // ------------------------------------------------------------------
    // Tables
    // ------------------------------------------------------------------

    fn flush_table(&mut self) {
        if self.table.is_empty() {
            return;
        }
        let lines = std::mem::take(&mut self.table);

        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut aligns: Vec<Align> = Vec::new();
        let mut header_rows = 0;
        for line in &lines {
            let cells = split_row(line);
            if header_rows == 0 && !rows.is_empty() && is_separator(&cells) {
                header_rows = rows.len();
                aligns = cells.iter().map(|c| alignment(c)).collect();
                continue;
            }
            rows.push(cells);
        }

        // Without a |---| separator this isn't really a table: show it raw.
        if header_rows == 0 {
            for line in &lines {
                let mut plain = Self::inline_only();
                let text = plain.push(line.trim_start()) + &plain.finish();
                self.out.push_str(&text);
                self.out.push('\n');
            }
            self.current_style.clear();
            return;
        }

        let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
        let rendered: Vec<Vec<String>> = rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                (0..columns)
                    .map(|col| {
                        let text = row.get(col).map(String::as_str).unwrap_or("");
                        let mut cell = Self::inline_only();
                        if i < header_rows {
                            cell.line_kind = LineKind::Heading(2);
                        }
                        cell.push(text) + &cell.finish()
                    })
                    .collect()
            })
            .collect();

        let mut natural = vec![0usize; columns];
        let mut longest_word = vec![0usize; columns];
        for row in &rendered {
            for (col, cell) in row.iter().enumerate() {
                natural[col] = natural[col].max(visible_width(cell));
                longest_word[col] = longest_word[col].max(widest_word(cell));
            }
        }
        // Each column's least comfortable width: its longest word.
        let minimum: Vec<usize> = natural
            .iter()
            .zip(&longest_word)
            .map(|(&natural, &word)| natural.min(word.clamp(MIN_COLUMN, MAX_UNBROKEN_WORD)))
            .collect();

        // A table that opens the reply starts on its own line, clear of the
        // "s0ma>" prefix the caller writes in front of the first output.
        let mut out = String::new();
        if !self.emitted && self.out.is_empty() {
            out.push('\n');
        }

        // The last column stays free: a line that exactly fills the terminal
        // makes some terminals wrap early.
        let limit = self.width.map(|columns| columns.saturating_sub(1));
        let widths = match limit {
            None => natural,
            Some(limit) => match fit_columns(&natural, &minimum, limit) {
                Some(widths) => widths,
                None => {
                    out.push_str(&records(&rendered, header_rows, limit));
                    self.out.push_str(&out);
                    self.current_style.clear();
                    return;
                }
            },
        };

        let border = |left: &str, mid: &str, right: &str| -> String {
            let segments: Vec<String> = widths.iter().map(|w| "─".repeat(w + 2)).collect();
            format!("\x1b[0;{DECO}m{left}{}{right}{RESET}\n", segments.join(mid))
        };

        out.push_str(&border("┌", "┬", "┐"));
        for (i, row) in rendered.iter().enumerate() {
            // Each cell wrapped to its column; the row is as tall as its
            // tallest cell.
            let cells: Vec<Vec<String>> =
                row.iter().enumerate().map(|(col, cell)| wrap_styled(cell, widths[col])).collect();
            let height = cells.iter().map(Vec::len).max().unwrap_or(1).max(1);
            for line in 0..height {
                out.push_str(&format!("\x1b[0;{DECO}m│{RESET}"));
                for (col, lines) in cells.iter().enumerate() {
                    let cell = lines.get(line).map(String::as_str).unwrap_or("");
                    let pad = widths[col].saturating_sub(visible_width(cell));
                    let (left, right) = match aligns.get(col).copied().unwrap_or(Align::Left) {
                        Align::Left => (0, pad),
                        Align::Right => (pad, 0),
                        Align::Center => (pad / 2, pad - pad / 2),
                    };
                    out.push_str(&format!(
                        " {}{cell}{RESET}{} \x1b[0;{DECO}m│{RESET}",
                        " ".repeat(left),
                        " ".repeat(right)
                    ));
                }
                out.push('\n');
            }
            if i + 1 == header_rows && rendered.len() > header_rows {
                out.push_str(&border("├", "┼", "┤"));
            }
        }
        out.push_str(&border("└", "┴", "┘"));

        self.out.push_str(&out);
        self.current_style.clear();
    }

    // ------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------

    fn style(&self) -> String {
        let mut s = String::from("\x1b[0;");
        s.push_str(match self.line_kind {
            LineKind::Body | LineKind::Hidden => BODY,
            LineKind::Heading(1) => HEADING_1,
            LineKind::Heading(_) => HEADING,
            LineKind::Quote => QUOTE,
            LineKind::Code => CODE,
            LineKind::FenceInfo => DECO,
        });
        if self.code {
            s.push(';');
            s.push_str(CODE);
        }
        if self.bold {
            s.push_str(";1");
        }
        if self.italic {
            s.push_str(";3");
        }
        if self.strike {
            s.push_str(";9");
        }
        s.push('m');
        s
    }

    fn emit_char(&mut self, c: char) {
        let style = self.style();
        if style != self.current_style {
            self.out.push_str(&style);
            self.current_style = style;
        }
        self.out.push(c);
        self.prev = Some(c);
    }

    fn emit_str(&mut self, s: &str) {
        for c in s.chars() {
            self.emit_char(c);
        }
    }

    /// Decoration added by the renderer (bullets, gutters, rules), drawn in
    /// its own style.
    fn emit_deco(&mut self, sgr: &str, s: &str) {
        let style = format!("\x1b[0;{sgr}m");
        self.out.push_str(&style);
        self.out.push_str(s);
        self.current_style = style;
    }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Align {
    Left,
    Right,
    Center,
}

fn is_ordered_marker(marker: &str) -> bool {
    let Some(digits) = marker.strip_suffix('.').or_else(|| marker.strip_suffix(')')) else {
        return false;
    };
    !digits.is_empty() && digits.len() <= 9 && digits.chars().all(|c| c.is_ascii_digit())
}

fn split_row(line: &str) -> Vec<String> {
    let line = line.trim();
    let line = line.strip_prefix('|').unwrap_or(line);
    let line = match line.strip_suffix('|') {
        Some(rest) if !rest.ends_with('\\') => rest,
        _ => line,
    };
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek() == Some(&'|') => {
                cell.push('|');
                chars.next();
            }
            '|' => cells.push(std::mem::take(&mut cell).trim().to_string()),
            _ => cell.push(c),
        }
    }
    cells.push(cell.trim().to_string());
    cells
}

fn is_separator(cells: &[String]) -> bool {
    cells.iter().all(|c| {
        let c = c.trim();
        !c.is_empty() && c.contains('-') && c.chars().all(|ch| matches!(ch, '-' | ':' | ' '))
    })
}

fn alignment(cell: &str) -> Align {
    let cell = cell.trim();
    match (cell.starts_with(':'), cell.ends_with(':')) {
        (true, true) => Align::Center,
        (false, true) => Align::Right,
        _ => Align::Left,
    }
}

/// Column widths for a table no wider than `limit`. Natural widths when they
/// fit. Otherwise each column keeps at least its `minimum` (its longest word)
/// and the rest of the room is shared out under a common cap, so the widest
/// columns give way first and narrow ones stay whole. `None` when even the
/// minimums do not fit: the table is better shown as records.
fn fit_columns(natural: &[usize], minimum: &[usize], limit: usize) -> Option<Vec<usize>> {
    // "│" plus " cell │" for each column.
    let budget = limit.checked_sub(1 + 3 * natural.len())?;
    if natural.iter().sum::<usize>() <= budget {
        return Some(natural.to_vec());
    }
    let capped = |cap: usize| -> usize {
        natural.iter().zip(minimum).map(|(&want, &least)| cap.clamp(least, want.max(least))).sum()
    };
    if capped(0) > budget {
        return None;
    }
    // The largest cap that fits.
    let (mut low, mut high) = (0, natural.iter().copied().max().unwrap_or(0));
    while low < high {
        let mid = (low + high).div_ceil(2);
        if capped(mid) <= budget { low = mid } else { high = mid - 1 }
    }
    let mut widths: Vec<usize> =
        natural.iter().zip(minimum).map(|(&want, &least)| low.clamp(least, want.max(least))).collect();
    // Fewer columns are held at the cap than there are spare characters'
    // worth of a one-higher cap, so one each never overshoots.
    let mut spare = budget - widths.iter().sum::<usize>();
    for (width, &want) in widths.iter_mut().zip(natural) {
        if spare == 0 {
            break;
        }
        if want > *width {
            *width += 1;
            spare -= 1;
        }
    }
    Some(widths)
}

/// The widest run of non-space characters, ignoring styling.
fn widest_word(s: &str) -> usize {
    let (mut widest, mut current) = (0, 0);
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' => {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            }
            ' ' => current = 0,
            c => {
                current += UnicodeWidthChar::width(c).unwrap_or(0);
                widest = widest.max(current);
            }
        }
    }
    widest
}

/// A table too wide for the screen even with wrapped columns: each row as a
/// record, every cell under its column's header.
fn records(rendered: &[Vec<String>], header_rows: usize, limit: usize) -> String {
    let limit = limit.max(MIN_COLUMN);
    let rule = format!("\x1b[0;{DECO}m{}{RESET}\n", "─".repeat(limit.min(RULE_WIDTH)));
    let headers: &[String] = rendered.first().map(Vec::as_slice).unwrap_or(&[]);
    let body = rendered.get(header_rows..).unwrap_or(&[]);
    let mut out = String::new();
    let emit = |text: &str, out: &mut String| {
        // Continuation lines are indented under the first.
        for (n, line) in wrap_styled(text, limit.saturating_sub(2).max(1)).into_iter().enumerate() {
            out.push_str(if n == 0 { "" } else { "  " });
            out.push_str(&line);
            out.push_str(RESET);
            out.push('\n');
        }
    };
    if body.is_empty() {
        for header in headers {
            emit(header, &mut out);
        }
        return out;
    }
    for row in body {
        out.push_str(&rule);
        for (col, cell) in row.iter().enumerate() {
            let label = headers.get(col).map(String::as_str).unwrap_or("");
            emit(&format!("{label}{RESET}\x1b[0;{DECO}m:{RESET} {cell}"), &mut out);
        }
    }
    out.push_str(&rule);
    out
}

/// A whole document rendered as the page renders s0ma's replies, in lines
/// fitted to `width` columns (each line closes its own styles), for a tab
/// that shows one: the projects tab's write-ups and reports. `text` is made
/// safe here.
pub fn render_lines(text: &str, width: usize) -> Vec<String> {
    let mut stream = MarkdownStream::new();
    stream.set_width(Some(width));
    let mut out = stream.push(&crate::terminal::safe_text(text));
    out.push_str(&stream.finish());
    let mut lines = Vec::new();
    let mut style = String::new();
    for line in out.trim_end_matches('\n').split('\n') {
        // A style left open at the end of a line goes on into the next.
        let carried = format!("{style}{line}");
        let mut at = 0;
        while let Some(start) = carried[at..].find('\x1b') {
            let start = at + start;
            let Some(end) = carried[start..].find('m') else { break };
            style = carried[start..=start + end].to_string();
            at = start + end + 1;
        }
        if style == RESET || style.starts_with("\x1b[0m") {
            style.clear();
        }
        let wrapped = wrap_styled(&carried, width);
        if wrapped.is_empty() {
            lines.push(String::new());
        }
        lines.extend(wrapped.into_iter().map(|line| format!("{line}{RESET}")));
    }
    lines
}

/// Word-wrap text carrying this module's SGR styling to `width` columns.
/// Words longer than a line are broken; a line that starts mid-span reopens
/// the style in force there, since the caller resets styles at line ends.
fn wrap_styled(text: &str, width: usize) -> Vec<String> {
    let mut wrap = StyledWrap { width: width.max(1), ..StyledWrap::default() };
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' => {
                let mut sequence = String::from(c);
                for c in chars.by_ref() {
                    sequence.push(c);
                    if c == 'm' {
                        break;
                    }
                }
                wrap.style(sequence);
            }
            ' ' => wrap.space(),
            c => wrap.char(c),
        }
    }
    wrap.finish()
}

#[derive(Default)]
struct StyledWrap {
    width: usize,
    lines: Vec<String>,
    line: String,
    line_width: usize,
    /// The last style sequence seen: what is in force at this point.
    style: String,
    word: String,
    word_width: usize,
    /// The style in force where `word` began.
    word_style: String,
    /// Spaces before `word`.
    gap: usize,
}

impl StyledWrap {
    fn begin_word(&mut self) {
        if self.word.is_empty() {
            self.word_style = self.style.clone();
        }
    }

    fn style(&mut self, sequence: String) {
        self.begin_word();
        self.word.push_str(&sequence);
        self.style = sequence;
    }

    fn char(&mut self, c: char) {
        let width = UnicodeWidthChar::width(c).unwrap_or(0);
        // A word as wide as the line is broken where it fills it.
        if self.word_width > 0 && self.word_width + width > self.width {
            self.place();
        }
        self.begin_word();
        self.word.push(c);
        self.word_width += width;
    }

    fn space(&mut self) {
        self.place();
        self.gap += 1;
    }

    fn place(&mut self) {
        if self.word_width == 0 {
            // Styling only: it applies to whatever comes next.
            self.line.push_str(&self.word);
            self.word.clear();
            return;
        }
        if self.line_width > 0 && self.line_width + self.gap + self.word_width > self.width {
            self.lines.push(std::mem::take(&mut self.line));
            self.line = self.word_style.clone();
            self.line_width = 0;
        } else if self.line_width > 0 {
            self.line.extend(std::iter::repeat_n(' ', self.gap));
            self.line_width += self.gap;
        }
        self.line.push_str(&self.word);
        self.line_width += self.word_width;
        self.word.clear();
        self.word_width = 0;
        self.gap = 0;
    }

    fn finish(mut self) -> Vec<String> {
        self.place();
        if self.line_width > 0 || self.lines.is_empty() {
            self.lines.push(self.line);
        }
        self.lines
    }
}

/// Display width ignoring the SGR sequences this module inserts.
fn visible_width(s: &str) -> usize {
    let mut width = 0;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for c in chars.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            width += UnicodeWidthChar::width(c).unwrap_or(0);
        }
    }
    width
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render in tiny chunks to exercise the streaming paths, then strip the
    /// SGR codes so assertions can look at the visible text.
    fn render(input: &str, chunk: usize) -> (String, String) {
        let mut md = MarkdownStream::new();
        let mut raw = String::new();
        let chars: Vec<char> = input.chars().collect();
        for piece in chars.chunks(chunk) {
            raw.push_str(&md.push(&piece.iter().collect::<String>()));
        }
        raw.push_str(&md.finish());
        (raw.clone(), strip(&raw))
    }

    fn strip(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    /// Visual check in your own terminal:
    /// `cargo test markdown_preview -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn markdown_preview() {
        let sample = "# Heading\n\nSome **bold**, *italic*, ~~struck~~ and `code`.\n\n- ~~done~~ item\n- [link](https://example.com)\n\n| a | b |\n|---|---|\n| ~~x~~ | **y** |\n";
        let mut md = MarkdownStream::new();
        let mut out = md.push(sample);
        out.push_str(&md.finish());
        println!("\n{out}");
    }

    #[test]
    fn chunking_never_changes_the_result() {
        let input = "# Title\n\nSome **bold** and *italic* and ~~gone~~ and `code`.\n\n- one\n- two [link](https://x.y)\n\n```rust\nfn main() {}\n```\n\n| a | b |\n|---|--:|\n| 1 | 22 |\n\n> quoted\n---\n";
        let whole = render(input, usize::MAX).1;
        for chunk in [1, 2, 3, 5, 7] {
            assert_eq!(render(input, chunk).1, whole, "chunk size {chunk}");
        }
    }

    #[test]
    fn markers_are_replaced_by_styling() {
        let (raw, text) = render("Some **bold** and *it* and `x`\n", 1);
        assert_eq!(text, "Some bold and it and x\n");
        assert!(raw.contains(";1m"));
        assert!(raw.contains(";3m"));
    }

    #[test]
    fn strikethrough() {
        for chunk in [1, 2, usize::MAX] {
            let (raw, text) = render("was ~~$40~~ now $30\n", chunk);
            assert_eq!(text, "was $40 now $30\n");
            assert!(raw.contains(";9m"));
        }
        // At the start of a line, `~~` must not be mistaken for a code fence.
        let (raw, text) = render("~~gone~~ kept\n", 1);
        assert_eq!(text, "gone kept\n");
        assert!(raw.contains(";9m"));
        // Inside table cells too.
        let (raw, _) = render("| a |\n|---|\n| ~~x~~ |\n", 1);
        assert!(raw.contains(";9m"));
    }

    #[test]
    fn single_tildes_stay_literal() {
        let (raw, text) = render("takes ~5 min in ~/src, ~~~ and ~~ alone\n", 1);
        assert_eq!(text, "takes ~5 min in ~/src, ~~~ and ~~ alone\n");
        assert!(!raw.contains(";9"));
        // An unclosed ~~ is struck only to the end of its line.
        let (raw, _) = render("~~open\nnext\n", usize::MAX);
        assert!(!raw[raw.find('\n').unwrap()..].contains(";9"));
    }

    #[test]
    fn identifiers_and_arithmetic_stay_literal() {
        let (_, text) = render("snake_case_name and 2 * 3 = 6\n", 1);
        assert_eq!(text, "snake_case_name and 2 * 3 = 6\n");
    }

    #[test]
    fn unclosed_emphasis_does_not_leak_past_the_line() {
        for chunk in [1, usize::MAX] {
            let (raw, text) = render("an *open\nnext line\n", chunk);
            assert_eq!(text, "an open\nnext line\n");
            // Every character on the second line is drawn in plain body style.
            let second = &raw[raw.find('\n').unwrap()..];
            assert!(!second.contains(";3m"), "italic leaked: {second:?}");
        }
    }

    #[test]
    fn block_elements() {
        let (_, text) = render("## Head\n- item\n3. third\n> q\n***\n", 1);
        assert!(text.contains("Head\n"));
        assert!(!text.contains('#'));
        assert!(text.contains("• item"));
        assert!(text.contains("3. third"));
        assert!(text.contains("│ q"));
        assert!(text.contains("────"));
    }

    #[test]
    fn code_blocks_are_not_parsed() {
        let (_, text) = render("```py\nx = a*b*c # not a heading\n```\nafter\n", 1);
        assert!(text.contains("┌─ py"));
        assert!(text.contains("\nx = a*b*c # not a heading\n"));
        assert!(text.contains("└─"));
        assert!(text.contains("after"));
    }

    #[test]
    fn code_lines_copy_as_exactly_the_code() {
        let code = "def f(x):\n    if x:\n        return x | 1\n\n    return 0";
        let (raw, text) = render(&format!("```python\n{code}\n```\n"), 2);
        // Between the header and the footer, the visible lines are the code
        // itself: indentation kept, nothing added in front.
        let lines: Vec<&str> = text.lines().collect();
        let start = lines.iter().position(|line| line.starts_with("┌─")).unwrap();
        let end = lines.iter().position(|line| line.starts_with("└─")).unwrap();
        assert_eq!(lines[start + 1..end].join("\n"), code);
        // Still styled as code.
        assert!(raw.contains("\x1b[0;36m"));
    }

    #[test]
    fn tables_are_aligned() {
        let (_, text) = render("| name | n |\n|:--|--:|\n| **a** | 1 |\n| bbbb | 22 |\nafter\n", 3);
        assert!(text.contains("│ name │  n │"));
        assert!(text.contains("│ a    │  1 │"));
        assert!(text.contains("│ bbbb │ 22 │"));
        assert!(text.ends_with("after\n"));
    }

    fn render_at(input: &str, width: usize) -> (String, String) {
        let mut md = MarkdownStream::new();
        md.set_width(Some(width));
        let raw = md.push(input) + &md.finish();
        (raw.clone(), strip(&raw))
    }

    const SERVICES: &str = "| Port | Service | Notes |\n|---|---|---|\n\
        | 22 | ssh | Password auth enabled; root login permitted, worth checking |\n\
        | 8080 | http-proxy | Manager app at /manager/html |\n";

    #[test]
    fn wide_tables_fit_the_terminal_and_keep_their_borders() {
        for width in [60, 50, 40] {
            let (_, text) = render_at(SERVICES, width);
            let lines: Vec<&str> = text.lines().filter(|line| !line.is_empty()).collect();
            assert!(lines.len() > 6, "rows wrapped onto extra lines at {width}:\n{text}");
            for line in &lines {
                assert!(unicode_width::UnicodeWidthStr::width(*line) < width, "too wide at {width}: {line}");
                assert!(matches!(line.chars().next(), Some('┌' | '│' | '├' | '└')), "{line}");
                assert!(matches!(line.chars().last(), Some('┐' | '│' | '┤' | '┘')), "{line}");
            }
            // Wrapping happened between words, never inside one.
            for word in ["Password", "permitted,", "checking", "http-proxy", "/manager/html"] {
                assert!(text.contains(word), "{word} split at {width}:\n{text}");
            }
        }
    }

    #[test]
    fn tables_that_fit_are_laid_out_as_before() {
        let table = "| name | n |\n|:--|--:|\n| **a** | 1 |\n| bbbb | 22 |\n";
        assert_eq!(render_at(table, 200).1, render(table, 3).1);
    }

    #[test]
    fn narrow_screens_get_one_record_per_row() {
        let (_, text) = render_at(SERVICES, 30);
        assert!(!text.contains('│'), "no grid:\n{text}");
        assert!(text.contains("Port: 22\nService: ssh\nNotes: Password auth"), "{text}");
        assert!(text.contains("Port: 8080\nService: http-proxy\n"), "{text}");
        for line in text.lines() {
            assert!(unicode_width::UnicodeWidthStr::width(line) < 30, "{line}");
        }
    }

    #[test]
    fn styling_carries_onto_wrapped_lines() {
        let table = "| a | b |\n|---|---|\n| x | **one two three four five six seven eight nine ten** |\n";
        let (raw, text) = render_at(table, 30);
        let bold = "\x1b[0;37;1m";
        let wrapped: Vec<&str> = raw.lines().filter(|line| line.contains("│") && !strip(line).contains(" a ")).collect();
        assert!(wrapped.len() > 1, "{text}");
        for line in wrapped {
            if strip(line).split('│').nth(2).is_some_and(|cell| !cell.trim().is_empty()) {
                assert!(line.contains(bold), "bold lost on a wrapped line: {line:?}");
            }
        }
    }

    #[test]
    fn a_reply_that_opens_with_a_table_starts_it_on_its_own_line() {
        let (_, text) = render_at("| a |\n|---|\n| b |\n", 80);
        assert!(text.starts_with("\n┌"), "{text:?}");
        let (_, text) = render_at("Results:\n| a |\n|---|\n| b |\n", 80);
        assert!(text.starts_with("Results:\n┌"), "{text:?}");
        // The width survives finish(), for the next reply.
        let mut md = MarkdownStream::new();
        md.set_width(Some(30));
        let _ = md.push("x") + &md.finish();
        let second = strip(&(md.push(SERVICES) + &md.finish()));
        assert!(second.contains("Port: 22"), "{second}");
    }

    #[test]
    fn column_fitting_shares_room_widest_first() {
        // Fits: unchanged.
        assert_eq!(fit_columns(&[4, 10], &[4, 5], 80), Some(vec![4, 10]));
        // Too wide: the narrow column stays whole, the wide one gives way.
        let widths = fit_columns(&[4, 10, 80], &[4, 10, 13], 60).unwrap();
        assert_eq!(widths[..2], [4, 10]);
        // Exactly the limit (the caller has already kept the last terminal column free).
        assert_eq!(1 + 3 * 3 + widths.iter().sum::<usize>(), 60);
        // Not even the longest words fit: records.
        assert_eq!(fit_columns(&[20, 20, 20], &[15, 15, 15], 40), None);
    }

    #[test]
    fn non_links_and_paths_survive() {
        let (_, text) = render("arr[i] and - [ ] todo and C:\\Users\\me and \\*x\\*\n", 1);
        assert_eq!(text, "arr[i] and - [ ] todo and C:\\Users\\me and *x*\n");
    }

    #[test]
    fn held_back_text_is_flushed_on_finish() {
        let (_, text) = render("ends with **", 1);
        assert_eq!(text, "ends with **");
        let (_, text) = render("| a |", 1);
        assert_eq!(text, "| a |\n");
    }

    #[test]
    fn only_sgr_sequences_are_emitted() {
        let (raw, _) = render("# h\n- **b** [l](u)\n```\nc\n```\n| a |\n|-|\n| b |\n", 1);
        let mut rest = raw.as_str();
        while let Some(i) = rest.find('\x1b') {
            rest = &rest[i + 1..];
            let end = rest.find('m').unwrap();
            assert!(rest[..end].starts_with('['));
            assert!(rest[1..end].chars().all(|c| c.is_ascii_digit() || c == ';'));
            rest = &rest[end + 1..];
        }
    }

    #[test]
    fn a_document_renders_in_lines_that_fit() {
        let lines = super::render_lines("# Title\n\nSome **bold** words that run on for a while past the edge.\n\n- one\n- two\n\n```\ncode\n```", 24);
        let plain: Vec<String> = lines.iter().map(|line| strip(line)).collect();
        assert!(plain.iter().all(|line| unicode_width::UnicodeWidthStr::width(line.as_str()) <= 24), "{plain:?}");
        assert!(plain.iter().any(|line| line.contains("Title")) && plain.iter().any(|line| line.contains("bold")), "{plain:?}");
        assert!(plain.iter().any(|line| line.contains("one")) && plain.iter().any(|line| line.contains("code")), "{plain:?}");
        assert!(!plain.iter().any(|line| line.contains("**")), "rendered, not raw: {plain:?}");
        assert!(lines.iter().all(|line| line.ends_with(super::RESET)), "each line closes its styles");
    }
}
