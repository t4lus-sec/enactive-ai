//! The agent's page further back: the history and the page itself.

use super::*;

/// The agent's conversation, as the page draws it, for looking back through.
pub(super) fn history_body(tab: &mut Open, display: &ActiveDisplay, width: usize) -> (Vec<Line>, Footer) {
    if let Err(error) = history_rows(tab, display, width) {
        return (empty("The conversation can't be shown.", &error.to_string(), width), Footer::new(Vec::new(), vec![("esc", "back")]));
    }
    let rows = tab.history.as_ref().map(|history| history.rows.clone()).unwrap_or_default();
    let detail = if tab.expanded { "less" } else { "more" };
    (rows, Footer::new(vec![(Style::Dim, "s0ma, further back".into())], vec![("esc", "back"), ("^o", detail)]))
}

/// The agent's page in a pane: its conversation as the page draws it, following
/// the end (the wheel and PgUp scroll it back), and what the agent is doing.
pub(super) fn page_body(tab: &mut Open, display: &ActiveDisplay, width: usize) -> (Vec<Line>, Footer) {
    if let Err(error) = history_rows(tab, display, width) {
        return (empty("The conversation can't be shown.", &error.to_string(), width), Footer::new(Vec::new(), Vec::new()));
    }
    let mut rows = tab.history.as_ref().map(|history| history.rows.clone()).unwrap_or_default();
    let working = display.working();
    if let Some(working) = &working {
        rows.push(vec![(Style::Work, format!("  [*] {working}"))]);
    }
    let left = match working {
        Some(_) => vec![(Style::Status, "s0ma · what is sent steers it".to_string())],
        None => vec![(Style::Status, "s0ma".to_string())],
    };
    let detail = if tab.expanded { "less" } else { "more" };
    let mut hints = vec![("⏎", "send")];
    if working.is_some() {
        hints.push(("^s", "send now"));
    }
    hints.extend([("^o", detail), ("⇧pgup", "scroll")]);
    if is_split() {
        hints.extend([("alt ←→", "pane"), ("alt \\", "one pane")]);
    }
    (rows, Footer::new(left, hints))
}

/// Bring `View::History`'s rows up to date; how many there are. While
/// scrolled back, what is being read stays put as rows are added at the end
/// (and, in a long session, dropped from the front: the redraw starts from a
/// later point), found again by its rows (`reanchor`).
pub(super) fn history_rows(tab: &mut Open, display: &ActiveDisplay, width: usize) -> io::Result<usize> {
    let from = (display.record_revision(), width, tab.expanded);
    // Laid out again at most once a second while the record grows (a turn
    // streaming): each is the whole conversation.
    if let Some(history) = &tab.history
        && (history.from == from
            || ((history.from.1, history.from.2) == (width, tab.expanded) && history.built.elapsed() < HISTORY_EVERY))
    {
        return Ok(history.rows.len());
    }
    let rows: Vec<Line> =
        display.conversation_rows(width, tab.expanded)?.into_iter().map(|row| vec![(Style::Raw, row)]).collect();
    if let Some(history) = &tab.history
        && (history.from.1, history.from.2) == (width, tab.expanded)
        && tab.back > 0
    {
        tab.back = reanchor(&history.rows, &rows, tab.back);
    }
    let count = rows.len();
    tab.history = Some(HistoryRows { from, rows, built: Instant::now() });
    Ok(count)
}

/// Where the view `back` rows from the end of `old` is in `new`, as rows back
/// from its end: found by the rows ending there (eight, or fewer), the nearest to
/// where they would be with nothing dropped from the front; failing that,
/// moved back by however many rows were added.
pub(super) fn reanchor(old: &[Line], new: &[Line], back: usize) -> usize {
    let grown = new.len().saturating_sub(old.len());
    let Some(at) = old.len().checked_sub(1 + back) else {
        return back + grown;
    };
    let expected = (at + grown).min(new.len().saturating_sub(1));
    // Fewer rows when those before were dropped from the front too.
    for size in [8, 4, 2, 1] {
        let window = &old[(at + 1).saturating_sub(size)..=at];
        for end in (window.len().saturating_sub(1)..=expected).rev() {
            if new[end + 1 - window.len()..=end] == *window {
                return new.len() - 1 - end;
            }
        }
    }
    back + grown
}
