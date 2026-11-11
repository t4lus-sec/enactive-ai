//! The editor's live preview of markdown (Ctrl+P, as Obsidian's): every line
//! but the one being edited is shown rendered in place: its markers hidden
//! (`#`, `**`, backticks, a link's address), its text styled. The cursor's
//! line, and any the selection touches, stay as written. One source line is
//! one line on screen, wrapped, so the cursor and the mouse keep their places.
//!
//! What it renders, a line at a time: headings, bullets and numbered lists,
//! task boxes, quotes, rules, fenced code (the fences dimmed, the code as
//! code), table rows (their pipes dimmed); inline, `**bold**`, `*italic*`,
//! `` `code` ``, `~~struck~~`, `[links](…)` and images. Anything it does not
//! know is shown as written.

/// Where a line stands with fenced code: outside it, a fence marker line
/// (the ``` itself), or inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fence {
    Outside,
    Marker,
    Inside,
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
                out.push(Fence::Marker);
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
    /// A heading's level, 1 to 6.
    pub heading: u8,
}

impl Look {
    /// The look as SGR codes, in s0ma's palette.
    pub fn sgr(self) -> String {
        let mut codes = vec!["0"];
        match self.heading {
            1 => codes.extend(["1", "4", "37"]),
            2 => codes.extend(["1", "37"]),
            3..=6 => codes.extend(["1", "36"]),
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
            codes.push("32");
        } else if self.link {
            codes.extend(["4", "36"]);
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
pub fn render(line: &str, fence: Fence, width: usize) -> Vec<(Look, String)> {
    let dim = Look { dim: true, ..Look::default() };
    match fence {
        Fence::Marker => return vec![(dim, line.to_string())],
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
    while let Some(after) = rest.strip_prefix('>') {
        out.push((dim, "▎ ".into()));
        rest = after.strip_prefix(' ').unwrap_or(after);
    }
    // A table row: pipes dimmed, cells as inline text.
    if rest.starts_with('|') && rest.trim_end().ends_with('|') && rest.trim().len() > 1 {
        let separator = rest.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '));
        if separator {
            out.push((dim, rest.replace('-', "─")));
            return out;
        }
        for (at, cell) in rest.split('|').enumerate() {
            if at > 0 {
                out.push((dim, "│".into()));
            }
            out.extend(inline(cell, Look::default()));
        }
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
            out.push((Look::default(), "• ".into()));
            rest = after;
        }
    }
    let base = if done { Look { struck: true, dim: true, ..Look::default() } } else { Look::default() };
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

    fn shown(line: &str) -> String {
        render(line, Fence::Outside, 10).into_iter().map(|(_, text)| text).collect()
    }

    fn look_of(line: &str, part: &str) -> Look {
        render(line, Fence::Outside, 10).into_iter().find(|(_, text)| text == part).map(|(look, _)| look).unwrap_or_else(|| panic!("{part} in {line}"))
    }

    #[test]
    fn markers_are_hidden_and_the_text_styled() {
        assert_eq!(shown("## Plan for **today**"), "Plan for today");
        assert_eq!(look_of("## Plan for **today**", "today"), Look { heading: 2, bold: true, ..Look::default() });
        assert_eq!(shown("- [ ] call the *bank*"), "☐ call the bank");
        assert!(look_of("- [ ] call the *bank*", "bank").italic);
        assert_eq!(shown("- [x] done"), "☑ done");
        assert!(look_of("- [x] done", "done").struck);
        assert_eq!(shown("  * nested"), "  • nested");
        assert_eq!(shown("> quoted `code`"), "▎ quoted code");
        assert!(look_of("> quoted `code`", "code").code);
        assert_eq!(shown("see [the spec](https://example.com/a_b) now"), "see the spec now");
        assert!(look_of("see [the spec](https://example.com/a_b) now", "the spec").link);
        assert_eq!(shown("![chart](c.png)"), "[image: chart]");
        assert_eq!(shown("---"), "─".repeat(10));
        assert_eq!(shown("~~old~~ new"), "old new");
        assert_eq!(shown("| a | **b** |"), "│ a │ b │");
    }

    #[test]
    fn what_is_not_markup_is_left_as_written() {
        for line in ["snake_case_name stays", "2 * 3 * 4", "a ** b", "an *unclosed marker", "#hashtag", "\\*not italic\\*", "[not a link]"] {
            let expected = line.replace("\\*", "*");
            assert_eq!(shown(line), expected, "{line}");
        }
        assert_eq!(shown("1. numbered"), "1. numbered");
    }

    #[test]
    fn fenced_code_is_code_and_its_fences_dimmed() {
        let text = ["intro", "```rust", "let a = **b**;", "```", "after **bold**"];
        let fences = fences(text.iter().copied());
        assert_eq!(fences, [Fence::Outside, Fence::Marker, Fence::Inside, Fence::Marker, Fence::Outside]);
        let inside = render(text[2], fences[2], 10);
        assert_eq!(inside, vec![(Look { code: true, ..Look::default() }, "let a = **b**;".to_string())]);
        assert_eq!(width(&render("## Plan", Fence::Outside, 10)), 4);
    }
}
