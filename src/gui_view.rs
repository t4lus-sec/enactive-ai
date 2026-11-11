//! The tabs as a GUI, a submodule of `tabs.rs`: what can be clicked, and
//! what a click does. The keys all still work; nothing here needs them shown.
//!
//! - **Clickable text.** A renderer makes any span clickable with `hot`, a
//!   button with `button`, a whole row with `pick_row`: the span's style
//!   becomes `Style::Hot(id, look)`, drawn as `look`, and `id` is the
//!   click registered for this frame (`begin` starts a frame's, `hits` reads
//!   where each one landed once the frame is composed).
//! - **Buttons for keys.** A view's footer says its keys as `(key, label)`
//!   pairs (`Footer`); `footer` turns each into a button named for what it
//!   does, which presses that key (`part`). Keys that only move (arrows,
//!   Tab between subtabs, Alt+1) have no button: a click or the wheel does
//!   that. Keys with nothing to click for them (Shift+arrows marking text,
//!   Tab completing) stay as they were, a key and what it does.
//! - **Rows.** A click on a row selects it (`Click::Pick`, which the view
//!   puts into effect); a second click on the selected row, soon after,
//!   presses Enter: a double-click opens.
//! - **Menus.** A right-click selects the row under it and opens a menu of
//!   what the footer offers for it (`Menu`): ↑↓ and Enter or a click to
//!   choose, Esc or a click elsewhere to close.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::cell::RefCell;
use std::time::{Duration, Instant};
use unicode_width::UnicodeWidthChar;

use crate::mistake_view::{Line, Style};

use super::View;

/// A second click on the same row within this is a double-click.
pub(super) const DOUBLE: Duration = Duration::from_millis(450);

/// What a click does.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Click {
    /// Press these keys in the tab, one after another.
    Keys(Vec<KeyEvent>),
    /// Send this command from the tab's prompt, as typed (what was being
    /// typed is kept).
    Command(&'static str),
    /// The approval waiting: one press of Alt+A.
    Approve,
    /// Select this row of the view (`Pick` says which, the view knows how).
    Pick(Pick),
}

/// A row a view can select by a click.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Pick {
    /// The share tab's entry at this path.
    Share(String),
    /// The memory tab's row, by its index in the section's rows.
    Memory(usize),
    /// The overview's row, by its index.
    Overview(usize),
    /// A projects subtab's board, by its index in the list.
    Project(usize),
    /// The board's node, by its index in the rows shown.
    Board(usize),
    /// The open board's section (files, notes, reports…), by its index.
    Section(usize),
    /// A row in the open board's section (a file, a note, a report).
    Item(usize),
    /// A memory section, by its index.
    MemorySection(usize),
    /// A projects subtab.
    Subtab(View),
}

/// Where a click lands: a frame's row, its columns `from..to`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Hit {
    pub row: usize,
    pub from: usize,
    pub to: usize,
    pub click: Click,
    /// What it says (a button's name), for a menu.
    pub label: String,
}

thread_local! {
    /// The clicks registered for the frame being composed, by id.
    static CLICKS: RefCell<Vec<Click>> = const { RefCell::new(Vec::new()) };
}

/// A new frame: no clicks registered yet.
pub(super) fn begin() {
    CLICKS.with(|clicks| clicks.borrow_mut().clear());
}

/// `look` as a `'static` reference, as `Style::Hot` keeps it.
fn kept(look: Style) -> &'static Style {
    match look {
        Style::Plain => &Style::Plain,
        Style::Dim => &Style::Dim,
        Style::Bold => &Style::Bold,
        Style::Heading => &Style::Heading,
        Style::Removed => &Style::Removed,
        Style::Added => &Style::Added,
        Style::RemovedEmph => &Style::RemovedEmph,
        Style::AddedEmph => &Style::AddedEmph,
        Style::Banner => &Style::Banner,
        Style::Problem => &Style::Problem,
        Style::Key => &Style::Key,
        Style::Accent => &Style::Accent,
        Style::Operator => &Style::Operator,
        Style::Status => &Style::Status,
        Style::Raw => &Style::Raw,
        Style::Warn => &Style::Warn,
        Style::Danger => &Style::Danger,
        Style::Success => &Style::Success,
        Style::Cursor => &Style::Cursor,
        Style::Work => &Style::Work,
        Style::Tab => &Style::Tab,
        Style::Marked => &Style::Marked,
        Style::Caret => &Style::Caret,
        Style::Hot(_, look) => look,
    }
}

fn register(click: Click) -> u16 {
    CLICKS.with(|clicks| {
        let mut clicks = clicks.borrow_mut();
        clicks.push(click);
        (clicks.len() - 1) as u16
    })
}

/// `text`, drawn as `look`, that does `click` when clicked.
pub(super) fn hot(look: Style, text: impl Into<String>, click: Click) -> (Style, String) {
    (Style::Hot(register(click), kept(look)), text.into())
}

/// A button: `label`, inverted, that does `click`.
pub(super) fn button(label: &str, click: Click) -> (Style, String) {
    hot(Style::Key, format!(" {label} "), click)
}

/// Every span of `line` clickable, all doing `click`, each drawn as it was;
/// a button in it keeps its own click.
pub(super) fn pick_row(line: Line, click: Click) -> Line {
    let id = register(click);
    line.into_iter()
        .map(|(style, text)| match style {
            Style::Hot(..) => (style, text),
            style => (Style::Hot(id, kept(style)), text),
        })
        .collect()
}

/// A row's own keys, as a row writes them (`⏎ turn on`, `e edit · a add a
/// server`, `←→ change`), as buttons; what can't be clicked as it was.
pub(super) fn row_buttons(action: &str) -> Line {
    let mut line: Line = Vec::new();
    for (index, part) in action.split(" · ").enumerate() {
        if index > 0 {
            line.push((Style::Plain, " ".into()));
        }
        let (spec, label) = part.split_once(' ').unwrap_or((part, ""));
        if spec == "←→" {
            let press = |code| Click::Keys(vec![KeyEvent::new(code, KeyModifiers::NONE)]);
            line.push(button("‹", press(KeyCode::Left)));
            line.push((Style::Dim, format!(" {label} ")));
            line.push(button("›", press(KeyCode::Right)));
            continue;
        }
        match key(spec) {
            Some(key) if !label.is_empty() && !spec.starts_with('/') => line.push(button(&named(label), Click::Keys(vec![key]))),
            _ => line.push((Style::Dim, part.to_string())),
        }
    }
    line
}

/// The style a span is drawn in: its look, when it is clickable.
#[cfg(test)]
pub(super) fn look(style: Style) -> Style {
    match style {
        Style::Hot(_, look) => *look,
        other => other,
    }
}

fn width(text: &str) -> usize {
    text.chars().map(|c| c.width().unwrap_or(0)).sum()
}

/// Where each clickable span of `frame` landed, with its click. Spans
/// beside each other with the same click make one hit.
pub(super) fn hits(frame: &[Line]) -> Vec<Hit> {
    let clicks = CLICKS.with(|clicks| clicks.borrow().clone());
    let mut out: Vec<Hit> = Vec::new();
    for (row, line) in frame.iter().enumerate() {
        let mut column = 0;
        for (style, text) in line {
            let span = width(&super::plain_text(text));
            if let Style::Hot(id, _) = style
                && let Some(click) = clicks.get(*id as usize)
                && span > 0
            {
                let text = super::plain_text(text);
                match out.last_mut() {
                    Some(last) if last.row == row && last.to == column && last.click == *click => {
                        last.to += span;
                        last.label.push_str(&text);
                    }
                    _ => out.push(Hit { row, from: column, to: column + span, click: click.clone(), label: text }),
                }
            }
            column += span;
        }
    }
    out
}

/// The hit at `(row, column)` of a frame, if any.
pub(super) fn at(hits: &[Hit], row: usize, column: usize) -> Option<&Hit> {
    hits.iter().find(|hit| hit.row == row && (hit.from..hit.to).contains(&column))
}

// ------------------------------------------------------------- keys as buttons

/// What a footer's `(key, label)` becomes.
#[derive(Debug, PartialEq)]
pub(super) enum Part {
    /// A button: its name, and what it does.
    Button(String, Click),
    /// A key with nothing to click for it: shown as it was.
    Keep(&'static str, &'static str),
    /// A key that only moves about: a click or the wheel does it.
    Drop,
}

/// One key as a footer writes it (`esc`, `⏎`, `^o`, `ctrl+q`, `alt n`, `x`…).
pub(super) fn key(spec: &str) -> Option<KeyEvent> {
    let spec = spec.trim();
    let (modifiers, rest) = if let Some(rest) = spec.strip_prefix("ctrl+").or_else(|| spec.strip_prefix('^').filter(|rest| !rest.is_empty())) {
        (KeyModifiers::CONTROL, rest)
    } else if let Some(rest) = spec.strip_prefix("alt+").or_else(|| spec.strip_prefix("alt ")) {
        (KeyModifiers::ALT, rest)
    } else if let Some(rest) = spec.strip_prefix('⇧').filter(|rest| !rest.is_empty()) {
        (KeyModifiers::SHIFT, rest)
    } else {
        (KeyModifiers::NONE, spec)
    };
    let code = match rest {
        "esc" => KeyCode::Esc,
        "⏎" | "enter" => KeyCode::Enter,
        "tab" => KeyCode::Tab,
        "del" | "delete" => KeyCode::Delete,
        "end" => KeyCode::End,
        "home" => KeyCode::Home,
        "space" => KeyCode::Char(' '),
        one if one.chars().count() == 1 => KeyCode::Char(one.chars().next()?),
        _ => return None,
    };
    Some(KeyEvent::new(code, modifiers))
}

/// `label` as a button says it: its first letter capitalised.
fn named(label: &str) -> String {
    let mut chars = label.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// What the footer's `(key, label)` becomes: one button or two (a pair of
/// keys, `[ ]` or `^n ^p`), the key kept as it was, or nothing.
pub(super) fn part(spec: &'static str, label: &'static str) -> Vec<Part> {
    // Only moving: a click, the wheel, the tab bar and the subtab and
    // section names do it.
    let moving = matches!(spec, "↑↓" | "←→" | "⇧pgup" | "⇧↑↓" | "alt ←→" | "alt 1" | "pgup" | "pgdn")
        || (spec == "tab" && matches!(label, "subtab" | "section" | "next"));
    if moving {
        return vec![Part::Drop];
    }
    // Nothing to click for these: marking text, moving the caret, completing.
    if matches!(spec, "⇧+arrows" | "←→↑↓") || (spec == "tab" && label == "complete") {
        return vec![Part::Keep(spec, label)];
    }
    // A command typed from the tab's prompt.
    if spec.starts_with('/') && spec.len() > 1 && label.is_empty() {
        return vec![Part::Button(named(&spec[1..]), Click::Command(spec))];
    }
    // A pair: the one before, the one after.
    let pair = match spec {
        "[ ]" => Some(("[", "]")),
        "^n ^p" => Some(("^p", "^n")),
        _ => None,
    };
    if let Some((before, after)) = pair {
        return [(before, format!("‹ {}", named(label))), (after, format!("{} ›", named(label)))]
            .into_iter()
            .filter_map(|(spec, name)| key(spec).map(|key| Part::Button(name, Click::Keys(vec![key]))))
            .collect();
    }
    match key(spec) {
        Some(key) if !label.is_empty() => vec![Part::Button(named(label), Click::Keys(vec![key]))],
        _ => vec![Part::Keep(spec, label)],
    }
}

/// The footer: `left` at the left edge, then the buttons for the keys
/// (`part`), the last ones left out when they don't fit (the menu still has
/// them). With the buttons, as `(name, click)`, for the menu.
pub(super) fn footer(left: Line, hints: &[(&'static str, &'static str)], room: usize) -> (Line, Vec<(String, Click)>) {
    let mut actions: Vec<(String, Click)> = Vec::new();
    let mut kept: Vec<(&str, &str)> = Vec::new();
    for (spec, label) in hints {
        for each in part(spec, label) {
            match each {
                Part::Button(name, click) if !actions.iter().any(|(had, _)| *had == name) => actions.push((name, click)),
                Part::Keep(spec, label) => kept.push((spec, label)),
                _ => {}
            }
        }
    }
    // What is on the left keeps up to half the row; the buttons have the
    // rest, and what is on the left gives way to them past that.
    let keep_left = width(&left.iter().map(|(_, text)| super::plain_text(text)).collect::<String>()).min(room / 2);
    let budget = room.saturating_sub(2 + keep_left);
    let mut right: Line = Vec::new();
    let mut right_width = 0;
    for (spec, label) in &kept {
        let piece = 2 + width(spec) + if label.is_empty() { 0 } else { 1 + width(label) };
        if right_width + piece > budget {
            break;
        }
        right.push((Style::Plain, format!("  {spec}")));
        if !label.is_empty() {
            right.push((Style::Dim, format!(" {label}")));
        }
        right_width += piece;
    }
    for (name, click) in &actions {
        let piece = 1 + width(name) + 2;
        if right_width + piece > budget {
            break;
        }
        right.push((Style::Plain, " ".into()));
        right.push(button(name, click.clone()));
        right_width += piece;
    }
    let mut line: Line = vec![(Style::Plain, " ".into())];
    let left_room = room.saturating_sub(2 + right_width);
    let mut used = 0;
    for (style, text) in left {
        let text = super::plain_text(&text);
        let fits = super::select_view::columns(&text, 0, left_room.saturating_sub(used));
        used += width(&fits);
        if !fits.is_empty() {
            line.push((style, fits));
        }
    }
    if !right.is_empty() {
        line.push((Style::Plain, " ".repeat(room.saturating_sub(1 + used + right_width + 1))));
        line.extend(right);
    }
    (line, actions)
}

/// A confirmation that names the key to press again (`ctrl+q again closes
/// …`, `x again clears the goal`): what it does, and that doing it again
/// confirms, by the key or by the button.
pub(super) fn again(text: &str) -> String {
    match text.split_once(" again ") {
        Some((key, rest)) if !key.is_empty() && !key.contains(' ') => format!("{rest} · again to confirm"),
        _ => text.to_string(),
    }
}

// ------------------------------------------------------------------ menus

/// A menu opened by a right-click: where, what it offers, which is chosen.
#[derive(Clone, Debug)]
pub(super) struct Menu {
    pub row: usize,
    pub column: usize,
    pub items: Vec<(String, Click)>,
    pub at: usize,
    /// How wide the frame it was drawn in is (a pane's, split): clicks on it
    /// are found where it was drawn.
    pub columns: usize,
}

impl Menu {
    /// Its box, laid out in a `columns` by `rows` frame: top row, left
    /// column, and its lines.
    fn laid(&self, columns: usize, rows: usize) -> (usize, usize, Vec<Line>) {
        let inner = self.items.iter().map(|(name, _)| width(name)).max().unwrap_or(0) + 2;
        let mut lines: Vec<Line> = vec![vec![(Style::Dim, format!("┌{}┐", "─".repeat(inner)))]];
        for (index, (name, _)) in self.items.iter().enumerate() {
            let pad = " ".repeat(inner - 1 - width(name));
            let style = if index == self.at { Style::Key } else { Style::Plain };
            lines.push(vec![(Style::Dim, "│".into()), (style, format!(" {name}{pad}")), (Style::Dim, "│".into())]);
        }
        lines.push(vec![(Style::Dim, format!("└{}┘", "─".repeat(inner)))]);
        let box_width = inner + 2;
        let left = self.column.min(columns.saturating_sub(box_width));
        let top = if self.row + 1 + lines.len() <= rows { self.row + 1 } else { self.row.saturating_sub(lines.len()) };
        (top, left, lines)
    }

    /// Drawn over a frame `rows` high; the item under `(row, column)`, when
    /// one is.
    pub fn item_at(&self, rows: usize, row: usize, column: usize) -> Option<usize> {
        let (top, left, lines) = self.laid(self.columns, rows);
        let box_width = lines.first().map(|line| width(&line[0].1)).unwrap_or(0);
        (row > top && row < top + lines.len() - 1 && (left..left + box_width).contains(&column)).then(|| row - top - 1)
    }

    /// The menu drawn over `frame` (`columns` wide).
    pub fn draw(&mut self, frame: &mut [Line], columns: usize) {
        self.columns = columns;
        let (top, left, lines) = self.laid(columns, frame.len());
        for (offset, piece) in lines.into_iter().enumerate() {
            if let Some(row) = frame.get_mut(top + offset) {
                *row = overlay(row, left, piece);
            }
        }
    }
}

/// `line` with `piece` drawn over it from column `left`: what was under
/// it is gone, the rest kept as it was drawn.
pub(super) fn overlay(line: &Line, left: usize, piece: Line) -> Line {
    let piece_width: usize = piece.iter().map(|(_, text)| width(text)).sum();
    let mut out = cut(line, 0, left);
    out.push((Style::Plain, " ".repeat(left.saturating_sub(column_of(&out)))));
    out.extend(piece);
    out.extend(cut(line, left + piece_width, usize::MAX));
    out
}

/// The part of `line` in columns `from..to`, each span in its own style
/// (one with its own colour codes as plain text).
fn cut(line: &Line, from: usize, to: usize) -> Line {
    let mut out: Line = Vec::new();
    let mut column = 0;
    for (style, text) in line {
        let text = super::plain_text(text);
        let span = width(&text);
        let (start, end) = (column, column + span);
        column = end;
        if end <= from || start >= to {
            continue;
        }
        let piece = super::select_view::columns(&text, from.saturating_sub(start), to.saturating_sub(start));
        if !piece.is_empty() {
            out.push((if *style == Style::Raw { Style::Plain } else { *style }, piece));
        }
    }
    out
}

fn column_of(line: &Line) -> usize {
    line.iter().map(|(_, text)| width(text)).sum()
}

/// Clicks on rows, told apart: whether this one is the second of a
/// double-click on the same thing. By what was clicked, not where: the first
/// click can move the rows (a board's strip grows, a list scrolls), and the
/// second must not open whatever moved under the mouse.
#[derive(Clone, Debug, Default)]
pub(super) struct Clicks {
    last: Option<(Instant, Pick)>,
}

impl Clicks {
    pub fn double(&mut self, pick: &Pick, now: Instant) -> bool {
        let double = self.last.as_ref().is_some_and(|(at, last)| last == pick && now.duration_since(*at) <= DOUBLE);
        self.last = if double { None } else { Some((now, pick.clone())) };
        double
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_become_buttons_named_for_what_they_do_and_moving_keys_go() {
        let button = |name: &str, code: KeyCode, modifiers: KeyModifiers| {
            Part::Button(name.into(), Click::Keys(vec![KeyEvent::new(code, modifiers)]))
        };
        assert_eq!(part("esc", "back"), vec![button("Back", KeyCode::Esc, KeyModifiers::NONE)]);
        assert_eq!(part("^q", "close"), vec![button("Close", KeyCode::Char('q'), KeyModifiers::CONTROL)]);
        assert_eq!(part("ctrl+q", "close"), vec![button("Close", KeyCode::Char('q'), KeyModifiers::CONTROL)]);
        assert_eq!(part("alt n", "new"), vec![button("New", KeyCode::Char('n'), KeyModifiers::ALT)]);
        assert_eq!(part("⇧⏎", "previous"), vec![button("Previous", KeyCode::Enter, KeyModifiers::SHIFT)]);
        assert_eq!(part("⏎", "open"), vec![button("Open", KeyCode::Enter, KeyModifiers::NONE)]);
        assert_eq!(part("del", "discard"), vec![button("Discard", KeyCode::Delete, KeyModifiers::NONE)]);
        assert_eq!(part("/", "search"), vec![button("Search", KeyCode::Char('/'), KeyModifiers::NONE)]);
        assert_eq!(part("[ ]", "beside").len(), 2);
        assert_eq!(part("^n ^p", "board")[0], button("‹ Board", KeyCode::Char('p'), KeyModifiers::CONTROL));
        assert_eq!(part("/kill", ""), vec![Part::Button("Kill".into(), Click::Command("/kill"))]);
        for moving in [("↑↓", "move"), ("←→", "section"), ("tab", "subtab"), ("alt 1", "back"), ("⇧pgup", "scroll")] {
            assert_eq!(part(moving.0, moving.1), vec![Part::Drop], "{moving:?}");
        }
        assert_eq!(part("⇧+arrows", "mark"), vec![Part::Keep("⇧+arrows", "mark")]);
        assert_eq!(part("tab", "complete"), vec![Part::Keep("tab", "complete")]);
    }

    #[test]
    fn a_footer_has_buttons_where_they_landed_and_no_key_names() {
        begin();
        let (line, actions) = footer(vec![(Style::Status, "notes".into())], &[("↑↓", "move"), ("n", "new"), ("esc", "back")], 60);
        let text: String = line.iter().map(|(_, text)| text.as_str()).collect();
        assert!(text.contains(" New ") && text.contains(" Back ") && !text.contains("esc") && !text.contains("↑↓"), "{text}");
        assert_eq!(actions.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>(), ["New", "Back"]);
        let found = hits(&[line]);
        assert_eq!(found.len(), 2);
        let back = &found[1];
        assert_eq!(back.click, Click::Keys(vec![KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)]));
        assert_eq!(back.to, 60 - 1, "flush with the right edge, one column in");
        assert!(at(&found, 0, back.from).is_some() && at(&found, 0, back.from - 1).is_none());
    }

    #[test]
    fn a_whole_row_picks_and_keeps_its_looks() {
        begin();
        let row = pick_row(vec![(Style::Cursor, "❯ ".into()), (Style::Bold, "notes.md".into())], Click::Pick(Pick::Share("notes.md".into())));
        assert_eq!(look(row[0].0), Style::Cursor);
        let found = hits(&[Vec::new(), row]);
        assert_eq!(found, vec![Hit { row: 1, from: 0, to: 10, click: Click::Pick(Pick::Share("notes.md".into())), label: "❯ notes.md".into() }]);
        // A row's own buttons keep their clicks inside a picked row.
        let mut line = vec![(Style::Plain, "loop  off  ".to_string())];
        line.extend(row_buttons("⏎ turn on · x clear"));
        let row = pick_row(line, Click::Pick(Pick::Overview(1)));
        let found: Vec<Hit> = hits(&[row]).into_iter().filter(|hit| matches!(hit.click, Click::Keys(_))).collect();
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(found[0].label.trim(), "Turn on");
        assert_eq!(found[0].click, Click::Keys(vec![KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)]));
        assert_eq!(found[1].label.trim(), "Clear");
    }

    #[test]
    fn a_menu_is_drawn_over_the_frame_and_knows_its_items() {
        let mut menu = Menu { row: 1, column: 4, items: vec![("Open".into(), Click::Approve), ("Delete".into(), Click::Approve)], at: 0, columns: 0 };
        let mut frame: Vec<Line> = (0..8).map(|i| vec![(Style::Plain, format!("row {i} {}", "x".repeat(20)))]).collect();
        menu.draw(&mut frame, 30);
        let text = |line: &Line| line.iter().map(|(_, text)| text.as_str()).collect::<String>();
        assert!(text(&frame[2]).starts_with("row ┌"), "{}", text(&frame[2]));
        assert!(text(&frame[3]).contains("│ Open   │") && text(&frame[3]).ends_with('x'), "{}", text(&frame[3]));
        assert_eq!(menu.item_at(8, 4, 6), Some(1));
        assert_eq!(menu.item_at(8, 2, 6), None, "its border");
        assert_eq!(menu.item_at(8, 4, 1), None, "beside it");
    }

    #[test]
    fn confirmations_say_again_without_the_key_and_double_clicks_are_told_apart() {
        assert_eq!(again("ctrl+q again closes w's tab"), "closes w's tab · again to confirm");
        assert_eq!(again("x again clears the goal"), "clears the goal · again to confirm");
        assert_eq!(again("delete again discards it"), "discards it · again to confirm");
        assert_eq!(again("nothing to confirm"), "nothing to confirm");
        let mut clicks = Clicks::default();
        let now = Instant::now();
        let (a, b) = (Pick::Share("a".into()), Pick::Share("b".into()));
        assert!(!clicks.double(&a, now));
        assert!(clicks.double(&a, now + Duration::from_millis(200)));
        assert!(!clicks.double(&a, now + Duration::from_millis(300)), "a third starts again");
        assert!(!clicks.double(&b, now + Duration::from_millis(350)), "another row: not a double-click on it");
        assert!(!clicks.double(&b, now + Duration::from_secs(2)));
    }
}
