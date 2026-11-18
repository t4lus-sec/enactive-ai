//! The editor's live preview of markdown (Ctrl+P, as Obsidian's): every line
//! but the one being edited is shown rendered in place: its markers hidden
//! (`#`, `**`, backticks, a link's address), its text styled. The cursor's
//! line, and any the selection touches, stay as written. One source line is
//! one line on screen, wrapped, so the cursor and the mouse keep their places.
//!
//! What it renders, a line at a time: headings, bullets and numbered lists,
//! task boxes, quotes, rules, fenced code (the fences as a frame, the code as
//! code), tables (their columns lined up); inline, `**bold**`, `*italic*`,
//! `` `code` ``, `~~struck~~`, `[links](…)` and images. Anything it does not
//! know is shown as written. Colours are the page's (`markdown.rs`).

/// Where a line stands with fenced code: outside it, a fence marker line
/// (the ``` itself), or inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fence {
    Outside,
    /// The marker that opens the code.
    Marker,
    /// The one that closes it.
    Closing,
    Inside,
}

/// Where a line stands in the text: with fenced code, and in a table (its
/// columns' widths, the same for every row of it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    pub fence: Fence,
    pub columns: Option<std::sync::Arc<Vec<usize>>>,
}

impl Default for Place {
    fn default() -> Self {
        Self { fence: Fence::Outside, columns: None }
    }
}

/// Each line's place, from the top of the text.
pub fn places<'a>(lines: impl Iterator<Item = &'a str> + Clone) -> Vec<Place> {
    let mut out: Vec<Place> = fences(lines.clone()).into_iter().map(|fence| Place { fence, columns: None }).collect();
    let lines: Vec<&str> = lines.collect();
    let mut at = 0;
    while at < lines.len() {
        let row = |i: usize| out[i].fence == Fence::Outside && table_row(lines[i]).is_some();
        if !row(at) {
            at += 1;
            continue;
        }
        let mut end = at;
        while end < lines.len() && row(end) {
            end += 1;
        }
        // Each column as wide as its widest cell, as shown.
        let mut widths: Vec<usize> = Vec::new();
        for line in &lines[at..end] {
            let Some(cells) = table_row(line).filter(|_| !separator(line)) else { continue };
            for (column, cell) in cells.iter().enumerate() {
                let shown = width(&inline(cell, Look::default()));
                if column >= widths.len() {
                    widths.push(shown);
                } else {
                    widths[column] = widths[column].max(shown);
                }
            }
        }
        let widths = std::sync::Arc::new(widths);
        for place in &mut out[at..end] {
            place.columns = Some(std::sync::Arc::clone(&widths));
        }
        at = end;
    }
    out
}

/// A table row's cells, trimmed: a line that starts and ends with a pipe.
fn table_row(line: &str) -> Option<Vec<&str>> {
    let inner = line.trim().strip_prefix('|')?.strip_suffix('|')?;
    Some(inner.split('|').map(str::trim).collect())
}

/// The row under a table's head: dashes, colons and pipes only.
fn separator(line: &str) -> bool {
    line.contains('-') && line.trim().chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

/// Each line's place with fenced code, from the top of the text.
pub fn fences<'a>(lines: impl Iterator<Item = &'a str>) -> Vec<Fence> {
    let mut open: Option<(char, usize)> = None;
    let mut out = Vec::new();
    for line in lines {
        let trimmed = line.trim_start();
        let run = |c: char| trimmed.chars().take_while(|&x| x == c).count();
        let marker = ['`', '~'].into_iter().find(|&c| run(c) >= 3).map(|c| (c, run(c)));
        match (open, marker) {
            (None, Some(found)) => {
                open = Some(found);
                out.push(Fence::Marker);
            }
            // Closed by a run of the same character, at least as long, alone.
            (Some((c, length)), Some((d, found))) if c == d && found >= length && trimmed[found..].trim().is_empty() => {
                open = None;
                out.push(Fence::Closing);
            }
            (Some(_), _) => out.push(Fence::Inside),
            (None, None) => out.push(Fence::Outside),
        }
    }
    out
}

/// How a piece of a line is drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Look {
    pub bold: bool,
    pub italic: bool,
    pub code: bool,
    pub struck: bool,
    pub link: bool,
    pub dim: bool,
    /// A list's marker.
    pub bullet: bool,
    /// A heading's level, 1 to 6.
    pub heading: u8,
}

impl Look {
    /// The look as SGR codes, in the agent's palette.
    pub fn sgr(self) -> String {
        let mut codes = vec!["0"];
        match self.heading {
            1 => codes.extend(["1", "4", "38;2;51;255;51"]),
            2..=6 => codes.extend(["1", "38;2;51;255;51"]),
            _ => {}
        }
        if self.bold {
            codes.push("1");
        }
        if self.italic {
            codes.push("3");
        }
        if self.struck {
            codes.push("9");
        }
        if self.code {
            codes.push("36");
        } else if self.link {
            codes.extend(["4", "94"]);
        } else if self.bullet {
            codes.extend(["1", "96"]);
        } else if self.dim {
            codes.push("90");
        } else if self.heading == 0 {
            codes.push("37");
        }
        format!("\x1b[{}m", codes.join(";"))
    }
}

/// A line rendered: its pieces, each with its look. `width` is for a rule,
/// drawn across it.
pub fn render(line: &str, place: &Place, width: usize) -> Vec<(Look, String)> {
    let dim = Look { dim: true, ..Look::default() };
    match place.fence {
        // The frame the page draws around code, with its language.
        Fence::Marker => {
            let label = line.trim_start().trim_start_matches(['`', '~']).trim();
            return vec![(dim, if label.is_empty() { "┌─".to_string() } else { format!("┌─ {label}") })];
        }
        Fence::Closing => return vec![(dim, "└─".into())],
        Fence::Inside => return vec![(Look { code: true, ..Look::default() }, line.to_string())],
        Fence::Outside => {}
    }
    let indent: String = line.chars().take_while(|c| *c == ' ' || *c == '\t').collect();
    let rest = &line[indent.len()..];
    let mut out = Vec::new();
    if !indent.is_empty() {
        out.push((Look::default(), indent.clone()));
    }

    // A rule: --- *** ___ (three or more, spaces allowed between).
    let bare: String = rest.chars().filter(|c| !c.is_whitespace()).collect();
    if bare.len() >= 3 && ["-", "*", "_"].iter().any(|c| bare.chars().all(|x| x.to_string() == *c)) && indent.len() < 4 {
        return vec![(dim, "─".repeat(width.max(3)))];
    }
    // A heading.
    let hashes = rest.chars().take_while(|c| *c == '#').count();
    if (1..=6).contains(&hashes) && (rest.len() == hashes || rest[hashes..].starts_with(' ')) {
        let text = rest[hashes..].trim().trim_end_matches('#').trim_end();
        let look = Look { heading: hashes as u8, ..Look::default() };
        out.extend(inline(text, look));
        return out;
    }
    // A quote, a bar for each level.
    let mut rest = rest;
    let mut quoted = false;
    while let Some(after) = rest.strip_prefix('>') {
        out.push((dim, "│ ".into()));
        rest = after.strip_prefix(' ').unwrap_or(after);
        quoted = true;
    }
    // A table row: its cells as inline text, each as wide as its column.
    if let Some(cells) = table_row(rest) {
        let columns = place.columns.as_deref().map(Vec::as_slice).unwrap_or_default();
        let wide = |column: usize| columns.get(column).copied().unwrap_or(0);
        if separator(rest) {
            let bars: Vec<String> = (0..cells.len()).map(|column| "─".repeat(wide(column).max(3) + 2)).collect();
            out.push((dim, format!("├{}┤", bars.join("┼"))));
            return out;
        }
        for (column, cell) in cells.iter().enumerate() {
            out.push((dim, "│ ".into()));
            let pieces = inline(cell, Look::default());
            let pad = wide(column).max(3).saturating_sub(self::width(&pieces));
            out.extend(pieces);
            out.push((Look::default(), " ".repeat(pad + 1)));
        }
        out.push((dim, "│".into()));
        return out;
    }
    // A bullet, and a task's box.
    let mut done = false;
    if let Some(after) = ["- ", "* ", "+ "].iter().find_map(|marker| rest.strip_prefix(marker)) {
        if let Some(task) = after.strip_prefix("[ ] ") {
            out.push((Look::default(), "☐ ".into()));
            rest = task;
        } else if let Some(task) = after.strip_prefix("[x] ").or_else(|| after.strip_prefix("[X] ")) {
            out.push((dim, "☑ ".into()));
            rest = task;
            done = true;
        } else {
            out.push((Look { bullet: true, ..Look::default() }, "• ".into()));
            rest = after;
        }
    } else {
        // A numbered item: its number as the page shows it.
        let digits = rest.chars().take_while(char::is_ascii_digit).count();
        if digits > 0 && digits <= 9 && rest[digits..].starts_with(". ") {
            out.push((Look { bullet: true, ..Look::default() }, rest[..digits + 1].to_string()));
            rest = &rest[digits + 1..];
        }
    }
    let base = match (done, quoted) {
        (true, _) => Look { struck: true, dim: true, ..Look::default() },
        (false, true) => Look { italic: true, dim: true, ..Look::default() },
        (false, false) => Look::default(),
    };
    out.extend(inline(rest, base));
    out
}

/// The width a rendered line takes (no tabs: they are spaces by then).
pub fn width(pieces: &[(Look, String)]) -> usize {
    pieces.iter().map(|(_, text)| unicode_width::UnicodeWidthStr::width(text.as_str())).sum()
}

/// Inline markup in `text`, on top of `base`. A marker without its
/// partner is shown as written.
fn inline(text: &str, base: Look) -> Vec<(Look, String)> {
    let mut out: Vec<(Look, String)> = Vec::new();
    let mut plain = String::new();
    let flush = |out: &mut Vec<(Look, String)>, plain: &mut String| {
        if !plain.is_empty() {
            out.push((base, std::mem::take(plain)));
        }
    };
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        // An escaped character, as itself.
        if c == '\\'
            && let Some(next) = rest[1..].chars().next().filter(|next| next.is_ascii_punctuation())
        {
            plain.push(next);
            rest = &rest[1 + next.len_utf8()..];
            continue;
        }
        // Code: as written inside, nothing else parsed.
        if c == '`' {
            let ticks = rest.chars().take_while(|x| *x == '`').count();
            let fence = &rest[..ticks];
            if let Some(end) = rest[ticks..].find(fence) {
                flush(&mut out, &mut plain);
                let inside = &rest[ticks..ticks + end];
                let inside = inside.strip_prefix(' ').and_then(|s| s.strip_suffix(' ')).filter(|s| !s.trim().is_empty()).unwrap_or(inside);
                out.push((Look { code: true, ..base }, inside.to_string()));
                rest = &rest[ticks + end + ticks..];
                continue;
            }
        }
        // An image, then a link: the text shown, the address hidden.
        let image = rest.starts_with("![");
        if (image || c == '[')
            && let Some((label, after)) = link(&rest[usize::from(image)..])
        {
            flush(&mut out, &mut plain);
            if image {
                out.push((Look { dim: true, ..base }, format!("[image: {label}]")));
            } else {
                out.extend(inline(label, Look { link: true, ..base }));
            }
            rest = after;
            continue;
        }
        // Emphasis: ** and __ bold, * and _ italic, ~~ struck.
        let pair = ["**", "__", "~~", "*", "_"].into_iter().find(|marker| rest.starts_with(marker));
        if let Some(marker) = pair {
            let inner = &rest[marker.len()..];
            // `_` only between words' edges, so snake_case stays as it is.
            let word_edge = !marker.starts_with('_') || plain.chars().last().is_none_or(|last| !last.is_alphanumeric());
            let opens = inner.chars().next().is_some_and(|next| !next.is_whitespace());
            if word_edge
                && opens
                && let Some(end) = closing(inner, marker)
            {
                flush(&mut out, &mut plain);
                let look = match marker {
                    "**" | "__" => Look { bold: true, ..base },
                    "~~" => Look { struck: true, ..base },
                    _ => Look { italic: true, ..base },
                };
                out.extend(inline(&inner[..end], look));
                rest = &inner[end + marker.len()..];
                continue;
            }
        }
        plain.push(c);
        rest = &rest[c.len_utf8()..];
    }
    flush(&mut out, &mut plain);
    out
}

/// `[label](address)` at the start of `text`: the label and what follows.
fn link(text: &str) -> Option<(&str, &str)> {
    let inner = text.strip_prefix('[')?;
    let close = inner.find("](")?;
    let label = &inner[..close];
    if label.is_empty() || label.contains('[') {
        return None;
    }
    let after = &inner[close + 2..];
    let end = after.find(')')?;
    Some((label, &after[end + 1..]))
}

/// Where `marker` closes `inner`: not right after a space, and for `_` not
/// inside a word.
fn closing(inner: &str, marker: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(found) = inner[from..].find(marker) {
        let at = from + found;
        let before = inner[..at].chars().last();
        let after = inner[at + marker.len()..].chars().next();
        let single = marker.len() == 1;
        // A single `*` is not half of a `**`.
        let doubled = single && (after == marker.chars().next() || inner[..at].ends_with(marker));
        let in_word = marker.starts_with('_') && after.is_some_and(char::is_alphanumeric);
        if at > 0 && before.is_some_and(|before| !before.is_whitespace()) && !doubled && !in_word {
            return Some(at);
        }
        from = at + marker.len();
    }
    None
}


#[cfg(test)]
mod tests {
    use super::*;

    fn shown(text: &str) -> Vec<String> {
        let lines: Vec<&str> = text.lines().collect();
        let places = places(lines.iter().copied());
        lines.iter().zip(&places).map(|(line, place)| render(line, place, 40).into_iter().map(|(_, text)| text).collect()).collect()
    }

    #[test]
    fn a_tables_columns_line_up() {
        let rows = shown("| name | value |\n|---|--:|\n| alpha | 1 |\n| b | **22** |\n\nafter | this\n");
        assert_eq!(rows[0], "│ name  │ value │");
        assert_eq!(rows[1], "├───────┼───────┤");
        assert_eq!(rows[2], "│ alpha │ 1     │");
        assert_eq!(rows[3], "│ b     │ 22    │");
        assert_eq!(rows[5], "after | this");
    }

    #[test]
    fn fences_are_a_frame_and_pipes_inside_are_code() {
        let rows = shown("```rust\n| not | a table |\n```\n```\nplain\n```\n");
        assert_eq!(rows, ["┌─ rust", "| not | a table |", "└─", "┌─", "plain", "└─"]);
    }

    #[test]
    fn lists_and_quotes_keep_their_marks() {
        let rows = shown("1. one\n   - nested\n> said\n- [x] done\n");
        assert_eq!(rows, ["1. one", "   • nested", "│ said", "☑ done"]);
    }
}
