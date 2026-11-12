//! The board's tab: the tree, the strip, its keys and clicks.

use super::*;

/// The board's tab: its head, the search as a tree to walk (`board_view`),
/// and the selected node in full in a strip under it. Laid out to fill
/// `rows` exactly, the tree scrolled to keep the selection in sight.
pub(super) fn board_body(tab: &mut Open, width: usize, rows: usize, now: Instant) -> (Vec<Line>, Footer) {
    board_refresh(tab, now);
    let Some(read) = tab.board_read.take() else {
        return (empty("Reading the board…", "", width), Footer::new(Vec::new(), vec![("esc", "back")]));
    };
    // Out of the tab while it is laid out (the tree's folds and selection are
    // the tab's to change), and back after. Under the subtab line and the
    // section bar.
    let laid_out = board_layout(tab, &read, width, rows, 2, false);
    tab.board_read = Some(read);
    laid_out
}

/// The open board, read again on a thread when the last read is stale.
pub(super) fn board_refresh(tab: &mut Open, now: Instant) {
    // A read that came in: kept, unless it is for another board than the
    // one up now. One that failed keeps what was read before on screen (a
    // worker mid-write), until the board has failed for a while.
    if let Some((index, slot)) = &tab.board_reading
        && let Some(read) = slot.lock().ok().and_then(|mut slot| slot.take())
    {
        let index = *index;
        tab.board_reading = None;
        if index == tab.board {
            if read.snapshot.is_ok() {
                tab.board_failing = None;
                tab.board_read = Some(read);
            } else {
                let since = *tab.board_failing.get_or_insert(now);
                let keep_old = now.duration_since(since) < BOARD_GIVE_UP
                    && tab.board_read.as_ref().is_some_and(|old| old.snapshot.is_ok());
                if keep_old {
                    // What was read before stays up; asked again at the next poll.
                    if let Some(old) = tab.board_read.as_mut() {
                        old.at = now;
                    }
                } else {
                    tab.board_read = Some(read);
                }
            }
        }
    }
    // Read again, on a thread: a write-locked store (a worker writing the
    // board) never holds the tab up.
    let stale = tab.board_read.as_ref().is_none_or(|read| now.duration_since(read.at) >= BOARD_POLL);
    if stale && tab.board_reading.is_none() {
        let index = tab.board;
        let archive = ARCHIVE.get().cloned();
        let slot = std::sync::Arc::new(Mutex::new(None));
        let answer = slot.clone();
        let spawned = std::thread::Builder::new().name("board-tab".into()).spawn(move || {
            let read = match archive {
                Some(archive) => read_board(&crate::board::BoardStore::beside_archive(&archive), index)
                    .unwrap_or_else(|error| BoardRead { at: Instant::now(), snapshot: Err(error.to_string()), name: None, open: 0, waiting: 0 }),
                None => BoardRead {
                    at: Instant::now(),
                    snapshot: Err("the board is not available in this process".into()),
                    name: None,
                    open: 0,
                    waiting: 0,
                },
            };
            if let Ok(mut answer) = answer.lock() {
                *answer = Some(read);
            }
            poke();
        });
        if spawned.is_ok() {
            tab.board_reading = Some((index, slot));
        }
    }
}

/// The open board as the open subtab's other sections show it: its files,
/// notes and reports.
pub(super) fn open_item(tab: &Open) -> Option<projects_view::Item> {
    let snapshot = tab.board_read.as_ref()?.snapshot.as_ref().ok()?.as_ref()?;
    let mut flow = snapshot.flow.clone().unwrap_or_default();
    if flow.board_id.is_empty() {
        flow.board_id = snapshot.board.id.clone();
        flow.name = snapshot.board.name.clone();
    }
    Some(projects_view::Item { board: snapshot.board.clone(), flow, notes: snapshot.notes.clone(), snapshot: None })
}

/// A board as a section of a later subtab (review, verification,
/// complete): its tree and the strip under it, in `rows` rows from row
/// `above` of the tab's body, walked as on the board's own tab. Its keys.
pub(super) fn embedded_board(tab: &mut Open, snapshot: &board_view::Snapshot, width: usize, rows: usize, above: usize) -> (Vec<Line>, Vec<(&'static str, &'static str)>) {
    // Another board: its tree as new.
    if tab.projects.board_for.as_deref() != Some(snapshot.board.id.as_str()) {
        tab.projects.board_for = Some(snapshot.board.id.clone());
        tab.folds = board_view::Folds::default();
        tab.board_selected = None;
        tab.board_top = 0;
        tab.strip_scroll = 0;
        tab.strip_for = None;
    }
    let read = BoardRead { at: Instant::now(), snapshot: Ok(Some(snapshot.clone())), name: Some(snapshot.board.name.clone()), open: 1, waiting: 0 };
    let (lines, footer) = board_layout(tab, &read, width, rows, above, true);
    // Kept for its keys (`board_section_key`).
    tab.board_read = Some(read);
    (lines, footer.hints)
}

/// A key in a board section of a later subtab: Shift+↑↓ the strip, the
/// tree's own walking and folding keys. Whether it was one.
pub(super) fn board_section_key(tab: &mut Open, key: &KeyEvent, page: usize) -> bool {
    strip_key(tab, key, page) || board_walk(tab, key, page)
}

/// On the board, Shift scrolls the strip with the selected node in full;
/// whether `key` did.
pub(super) fn strip_key(tab: &mut Open, key: &KeyEvent, page: usize) -> bool {
    if key.modifiers != KeyModifiers::SHIFT {
        return false;
    }
    let scroll = match key.code {
        KeyCode::Up => tab.strip_scroll.saturating_sub(1),
        KeyCode::Down => tab.strip_scroll.saturating_add(1),
        KeyCode::PageUp => tab.strip_scroll.saturating_sub(page / 2),
        KeyCode::PageDown => tab.strip_scroll.saturating_add(page / 2),
        KeyCode::Home => 0,
        KeyCode::End => usize::MAX,
        _ => return false,
    };
    // Kept within the strip when it is laid out.
    tab.strip_scroll = scroll;
    true
}

/// A turn of the wheel at `row` of the tab's body over a board: below the
/// strip's rule it scrolls the strip; over the tree, the tree scrolls
/// first, and the selection moves with it only when it would leave the rows
/// shown, and on its own once the tree is at its end.
pub(super) fn board_wheel(tab: &mut Open, up: bool, row: usize, page: usize) {
    if row >= tab.strip_at {
        tab.strip_scroll = if up { tab.strip_scroll.saturating_sub(WHEEL_ROWS) } else { tab.strip_scroll.saturating_add(WHEEL_ROWS) };
        return;
    }
    let len = tab.board_shown.len();
    let rows = tab.board_tree_rows.max(1);
    let last_top = len.saturating_sub(rows);
    let at = tab.board_selected.as_ref().and_then(|id| tab.board_shown.iter().position(|shown| shown == id));
    let scrolls = if up { tab.board_top > 0 } else { tab.board_top < last_top };
    if scrolls {
        tab.board_top = if up { tab.board_top.saturating_sub(WHEEL_ROWS) } else { (tab.board_top + WHEEL_ROWS).min(last_top) };
        if let Some(at) = at {
            let kept = at.clamp(tab.board_top, (tab.board_top + rows).saturating_sub(1).min(len.saturating_sub(1)));
            if kept != at {
                tab.board_selected = tab.board_shown.get(kept).cloned();
                tab.strip_scroll = 0;
            }
        }
    } else {
        let code = if up { KeyCode::Up } else { KeyCode::Down };
        board_walk(tab, &KeyEvent::new(code, KeyModifiers::NONE), page);
    }
}

/// The board, laid out in `rows` rows from row `at_row` of the tab's body:
/// its head (on its own tab), the tree, the strip with the selected node in
/// full. `embedded`: a section of a later subtab, under its list of boards
/// (no head, and only the tree's keys).
pub(super) fn board_layout(tab: &mut Open, read: &BoardRead, width: usize, rows: usize, at_row: usize, embedded: bool) -> (Vec<Line>, Footer) {
    let footer = |left: Line, hints: Vec<(&'static str, &'static str)>| Footer::new(left, hints);
    let snapshot = match &read.snapshot {
        Err(error) => {
            return (
                vec![Vec::new(), vec![(Style::Problem, format!("  the board could not be read: {error}"))]],
                footer(Vec::new(), vec![("esc", "back")]),
            );
        }
        Ok(None) => {
            return (
                empty("No open board.", "", width),
                footer(Vec::new(), vec![("esc", "back")]),
            );
        }
        Ok(Some(snapshot)) => snapshot,
    };
    let clock = chrono::Local::now();
    let tree = grey_tree(snapshot, clock);
    tab.folds.update(&tree, snapshot);
    let shown = board_view::rows(&tree, &tab.folds);
    let selected = selected_row(tab, &tree, &shown);

    let mut out = if embedded { Vec::new() } else { board_view::header(snapshot, width, clock) };
    // Finished: the agent's report is due, then it goes to the operator's review.
    let proposed_exhausted = snapshot.board.status == crate::board::BoardStatus::ProposedExhausted;
    if embedded {
    } else if matches!(snapshot.board.status, crate::board::BoardStatus::ProposedSolved | crate::board::BoardStatus::Solved) {
        out.extend(board_view::write_up_due(snapshot, width));
    } else if proposed_exhausted {
        out.extend(board_view::exhaustion_proposed(snapshot, width));
    }
    let left_rows = rows.saturating_sub(out.len());
    // The strip under the tree: the selected node in full, as many rows as it
    // needs up to two fifths of what is left (at least six when there is
    // room, three when short, none when very short) or all the tree leaves
    // over; past that it scrolls.
    let strip_cap = match left_rows {
        12.. => (left_rows * 2 / 5).max(6),
        7..12 => 3,
        _ => 0,
    };
    // And whatever the tree leaves over (a short tree, a long text).
    let strip_cap = if strip_cap == 0 { 0 } else { strip_cap.max(left_rows.saturating_sub(shown.len().max(1) + 1)) };
    let full = match selected {
        Some(index) if strip_cap > 0 => board_view::detail(&tree, snapshot, shown[index].node, width, clock),
        _ => Vec::new(),
    };
    if tab.strip_for != tab.board_selected {
        tab.strip_for = tab.board_selected.clone();
        tab.strip_scroll = 0;
    }
    let shown_rows = full.len().min(strip_cap);
    tab.strip_scroll = tab.strip_scroll.min(full.len() - shown_rows);
    let (above, below) = (tab.strip_scroll, full.len() - shown_rows - tab.strip_scroll);
    let strip: Vec<Line> = full.into_iter().skip(above).take(shown_rows).collect();
    let strip_rows = if strip.is_empty() { 0 } else { strip.len() + 1 };
    let tree_rows = left_rows.saturating_sub(strip_rows).max(1);
    if let Some(index) = selected {
        if index < tab.board_top {
            tab.board_top = index;
        } else if index >= tab.board_top + tree_rows {
            tab.board_top = index + 1 - tree_rows;
        }
    }
    tab.board_top = tab.board_top.min(shown.len().saturating_sub(tree_rows));
    tab.board_shown = shown.iter().map(|row| tree.nodes[row.node].id.clone()).collect();
    tab.board_tree_rows = tree_rows;
    if shown.is_empty() {
        out.push(vec![(Style::Dim, "  No leads yet.".into())]);
    }
    for (index, row) in shown.iter().enumerate().skip(tab.board_top).take(tree_rows) {
        let line = board_view::row_line(&tree, snapshot, &tab.folds, row, Some(index) == selected, width, clock);
        out.push(gui_view::pick_row(line, gui_view::Click::Pick(gui_view::Pick::Board(index))));
    }
    // Where the strip starts in the tab's body (under the subtab line).
    tab.strip_at = if strip.is_empty() { usize::MAX } else { at_row + rows.saturating_sub(strip_rows) };
    if !strip.is_empty() {
        out.resize(rows.saturating_sub(strip_rows), Vec::new());
        // The rule says when there is more of the strip than it shows.
        let more = match (above, below) {
            (0, 0) => String::new(),
            (0, below) => format!(" {below} more "),
            (above, 0) => format!(" {above} above "),
            (above, below) => format!(" {above} above · {below} more "),
        };
        let rule = width.saturating_sub(4 + text_width(&more));
        out.push(vec![(Style::Dim, format!("  {}", "─".repeat(rule))), (Style::Cursor, more)]);
        out.extend(strip);
    }
    out.truncate(rows);

    // ←→ step the sections, as in every other section, and ⏎ folds (on the
    // open board, while no hint is being typed).
    let mut hints: Vec<(&'static str, &'static str)> = vec![("↑↓", "move")];
    if embedded || tab.draft.is_empty() {
        hints.push(("⏎", "fold"));
    }
    if above + below > 0 {
        hints.push(("⇧↑↓", "detail"));
    }
    if !tab.draft.is_empty() {
        hints.push(("⏎", "send"));
    }
    if embedded {
        if let Some(index) = selected.filter(|&index| board_view::can_grey(&tree, shown[index].node)) {
            hints.push(("^c", if tree.grey[shown[index].node] { "ungrey" } else { "grey" }));
        }
        return (out, footer(Vec::new(), hints));
    }
    // One of the operator's notes: Delete takes it off (twice).
    if tab.draft.is_empty() && selected.is_some_and(|index| matches!(tree.nodes[shown[index].node].kind, board_view::Kind::Hint(_))) {
        hints.push(("del", "remove"));
    }
    if tab.draft.is_empty() && selected.is_some_and(|index| matches!(tree.nodes[shown[index].node].kind, board_view::Kind::Board)) {
        hints.push(("del", "discard"));
    }
    if tab.draft.is_empty() && selected.is_some_and(|index| board_view::can_grey(&tree, shown[index].node)) {
        hints.push(("^c", if tree.grey[shown[selected.unwrap()].node] { "ungrey" } else { "grey" }));
    }
    if read.open > 1 {
        hints.push(("^n ^p", "board"));
    }
    hints.push(("^q", "close"));
    if tab.draft.is_empty() {
        hints.push(("←→", "section"));
    }
    hints.push(("tab", "subtab"));
    hints.push(("esc", "back"));
    let mut left = match (&read.name, read.open) {
        (Some(_), open) if open > 1 => vec![(Style::Status, format!("board {} of {open}", tab.board % open + 1))],
        (Some(name), _) => vec![(Style::Status, name.clone())],
        (None, _) => Vec::new(),
    };
    if read.waiting > 0 {
        left.push((Style::Warn, format!("  [!] {} in review", read.waiting)));
    }
    (out, footer(left, hints))
}

/// The selected row's index among `shown`: the node selected before, by id,
/// or the first row. Records it, so it stays put as rows come and go.
pub(super) fn selected_row(tab: &mut Open, tree: &board_view::Tree, shown: &[board_view::Row]) -> Option<usize> {
    let wanted = tab.board_selected.as_deref().and_then(|id| tree.find(id));
    // A node folded away under its parent: the nearest ancestor shown.
    let mut node = wanted;
    let index = loop {
        match node {
            Some(at) => match shown.iter().position(|row| row.node == at) {
                Some(index) => break Some(index),
                None => node = tree.nodes[at].parent,
            },
            // At first, the first lead or finding: the row after the board's
            // own and the hints under it.
            None => {
                let first = shown.iter().skip(1).position(|row| row.guides.is_empty()).map(|index| index + 1);
                break (!shown.is_empty()).then(|| first.unwrap_or(if shown.len() > 1 { 1 } else { 0 }));
            }
        }
    };
    tab.board_selected = index.map(|index| tree.nodes[shown[index].node].id.clone());
    index
}

/// The board as the tab has it now: its tree and the rows shown.
pub(super) fn board_tree(tab: &mut Open) -> Option<(board_view::Tree, Vec<board_view::Row>, Option<usize>)> {
    let read = tab.board_read.take()?;
    let found = read.snapshot.as_ref().ok().and_then(Option::as_ref).map(|snapshot| {
        let tree = grey_tree(snapshot, chrono::Local::now());
        tab.folds.update(&tree, snapshot);
        let shown = board_view::rows(&tree, &tab.folds);
        let selected = selected_row(tab, &tree, &shown);
        (tree, shown, selected)
    });
    tab.board_read = Some(read);
    found
}

/// Ctrl+C, with nothing typed: the selected row greyed out, or back. For
/// the operator's reading only; the agent's board is as it was. Whether there
/// was a row to grey.
pub(super) fn board_grey(tab: &mut Open) -> bool {
    if !tab.draft.is_empty() {
        return false;
    }
    let board = tab.board_read.as_ref().and_then(|read| read.snapshot.as_ref().ok()?.as_ref()).map(|snapshot| snapshot.board.id.clone());
    let (Some((tree, shown, Some(index))), Some(board)) = (board_tree(tab), board) else {
        return false;
    };
    let node = shown[index].node;
    if !board_view::can_grey(&tree, node) {
        return false;
    }
    tab.board_unhint = None;
    with_greyed(&board, |greyed| board_view::toggle_grey(&tree, greyed, node));
    // The chevron stays where it was, on the row that comes up into the
    // place, so the next Ctrl+C takes the next one.
    if let Some((tree, shown, _)) = board_tree(tab) {
        tab.board_selected = shown.get(index.min(shown.len().saturating_sub(1))).map(|row| tree.nodes[row.node].id.clone());
    }
    true
}

/// A key on the board's tab. The arrows walk the tree, unless a hint is being
/// typed and they are the draft's: ↑↓ move, → unfolds (or steps into an
/// unfolded node), ← folds (or steps out to the parent), ⏎ with nothing typed
/// folds or unfolds, Home and End go to the first and last rows, Ctrl+O
/// unfolds or folds everything. The rest is the prompt's, as on any chat.
pub(super) fn board_key(tab: &mut Open, key: &KeyEvent, display: &ActiveDisplay, page: usize) -> io::Result<Outcome> {
    let plain = !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
    // Ctrl+Q: once to ask, again to close the board (discarded: closed,
    // abandoned, out of the projects tab). Any other key keeps it.
    let ctrl_only = key.modifiers.contains(KeyModifiers::CONTROL) && !key.modifiers.contains(KeyModifiers::ALT);
    if ctrl_only && matches!(key.code, KeyCode::Char('q' | 'Q')) {
        let board = tab.board_read.as_ref().and_then(|read| read.snapshot.as_ref().ok()?.as_ref()).map(|snapshot| (snapshot.board.id.clone(), snapshot.board.name.clone()));
        let armed = tab.board_unhint.take();
        if let Some((id, name)) = board {
            let mark = format!("close:{id}");
            if armed.as_deref() == Some(mark.as_str()) {
                match act(Action::Project { board: id, done: projects_view::Move::Discard }) {
                    Ok(reply) => tab.note(reply, false),
                    Err(reply) => tab.note(reply, true),
                }
                tab.board_read = None;
                tab.board_selected = None;
            } else {
                tab.ask(format!("ctrl+q again closes {name}: abandoned, and out of the projects tab"));
                tab.board_unhint = Some(mark);
            }
        }
        return Ok(Outcome::Stay);
    }
    if ctrl_only && matches!(key.code, KeyCode::Char('c' | 'C')) && board_grey(tab) {
        return Ok(Outcome::Stay);
    }
    // Ctrl+T: what the prompt adds, the next of hint, lead, finding.
    if ctrl_only && matches!(key.code, KeyCode::Char('t' | 'T')) {
        tab.board_unhint = None;
        let audits = tab.board_read.as_ref().and_then(|read| read.name.as_deref()).map(|name| crate::audit::ops::for_board(name).len()).unwrap_or(0);
        tab.board_adding = tab.board_adding.next(audits);
        return Ok(Outcome::Stay);
    }
    // Delete, with nothing typed, on one of the operator's notes: once to ask,
    // again to take it off the board. Any other key keeps it.
    let armed = tab.board_unhint.take();
    // Delete on a linked audit's row: once to ask, again to unlink it;
    // what is typed goes with it as the reason, which the agent hears.
    if plain && key.code == KeyCode::Delete {
        let linked = board_tree(tab).and_then(|(tree, shown, selected)| {
            let node = &tree.nodes[shown[selected?].node];
            let board_view::Kind::Lead(i) = node.kind else { return None };
            let snapshot = tab.board_read.as_ref()?.snapshot.as_ref().ok()?.as_ref()?;
            snapshot.intents[i].direction.strip_prefix("linked audit ").map(|audit| (audit.trim().to_string(), snapshot.board.name.clone()))
        });
        if let Some((audit, board)) = linked {
            let mark = format!("unlink:{audit}:{board}");
            if armed.as_deref() == Some(mark.as_str()) {
                let reason = tab.draft.text().trim().to_string();
                let result = crate::audit::store()
                    .ok_or_else(|| "the audit store is not ready".to_string())
                    .and_then(|store| crate::audit::ops::unlink(&store, &audit, &board, None, crate::audit::Who::Operator, &reason));
                match result {
                    Ok(reply) => {
                        tab.draft = Draft::default();
                        tab.note(reply.trim_start_matches("[audit] ").to_string(), false);
                    }
                    Err(error) => tab.note(error, true),
                }
                tab.board_read = None;
            } else {
                tab.ask(format!("del again unlinks {audit} from {board}{}", if tab.draft.is_empty() { "" } else { ", with what you typed as the reason" }));
                tab.board_unhint = Some(mark);
            }
            return Ok(Outcome::Stay);
        }
    }
    if plain && key.code == KeyCode::Delete && tab.draft.is_empty() {
        let picked = board_tree(tab).and_then(|(tree, shown, selected)| {
            let node = &tree.nodes[shown[selected?].node];
            match node.kind {
                board_view::Kind::Hint(_) => Some((false, node.id.clone())),
                board_view::Kind::Board => Some((true, node.id.clone())),
                _ => None,
            }
        });
        // On the board's own row: once to ask, again to discard the board.
        if let Some((true, row)) = &picked {
            let id = tab.board_read.as_ref().and_then(|read| read.snapshot.as_ref().ok()?.as_ref()).map(|snapshot| snapshot.board.id.clone());
            let name = tab.board_read.as_ref().and_then(|read| read.name.clone()).unwrap_or_default();
            if let Some(id) = id {
                let mark = format!("{row}:{id}");
                if armed.as_deref() == Some(mark.as_str()) {
                    match act(Action::Project { board: id, done: projects_view::Move::Discard }) {
                        Ok(reply) => tab.note(reply, false),
                        Err(reply) => tab.note(reply, true),
                    }
                    tab.board_read = None;
                    tab.board_selected = None;
                } else {
                    tab.ask(format!("delete again discards {name}: it is closed, and leaves the projects tab"));
                    tab.board_unhint = Some(mark);
                }
            }
            return Ok(Outcome::Stay);
        }
        let note = picked.map(|(_, id)| id);
        let board = tab.board_read.as_ref().and_then(|read| read.name.clone());
        match (note, board) {
            (Some(id), Some(board)) if armed.as_deref() == Some(id.as_str()) => {
                match act(Action::Unhint { board, id }) {
                    Ok(reply) => tab.note(reply, false),
                    Err(reply) => tab.note(reply, true),
                }
                // Read again at once, to show it gone.
                if let Some(read) = tab.board_read.as_mut() {
                    read.at = Instant::now().checked_sub(BOARD_POLL).unwrap_or(read.at);
                }
            }
            (Some(id), Some(_)) => {
                tab.ask(format!("delete again takes {} off the board", board_view::short(&id)));
                tab.board_unhint = Some(id);
            }
            _ => {}
        }
        return Ok(Outcome::Stay);
    }
    if board_walk(tab, key, page) {
        return Ok(Outcome::Stay);
    }
    chat_key(tab, key, display, page)
}

/// The keys that walk the board's tree (↑↓ PgUp PgDn, and with nothing
/// typed Home End, ⏎ to fold) and Ctrl+O; whether `key` was one of them.
/// ←→ step the sections before a key comes here.
/// A click on a node of the board's tree: it is selected (its row as last
/// drawn, `board_shown`).
pub(super) fn board_pick(tab: &mut Open, index: usize) {
    if let Some(id) = tab.board_shown.get(index).cloned()
        && tab.board_selected.as_ref() != Some(&id)
    {
        tab.board_selected = Some(id);
        tab.strip_scroll = 0;
    }
}

pub(super) fn board_walk(tab: &mut Open, key: &KeyEvent, page: usize) -> bool {
    let plain = !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SHIFT);
    let ctrl_o = key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('o' | 'O'));
    let walks = plain
        && match key.code {
            KeyCode::Up | KeyCode::Down | KeyCode::PageUp | KeyCode::PageDown => true,
            KeyCode::Home | KeyCode::End | KeyCode::Enter => tab.draft.is_empty(),
            _ => false,
        };
    if !(walks || ctrl_o) {
        return false;
    }
    let Some((tree, shown, selected)) = board_tree(tab) else {
        return ctrl_o;
    };
    if ctrl_o {
        tab.folds.toggle_all(&tree);
        return true;
    }
    let Some(index) = selected else {
        return true;
    };
    let node = shown[index].node;
    let has_children = !tree.nodes[node].children.is_empty();
    let open = tab.folds.is_open(&tree, node);
    let go = |tab: &mut Open, index: usize| {
        let index = index.min(shown.len() - 1);
        tab.board_selected = Some(tree.nodes[shown[index].node].id.clone());
    };
    match key.code {
        KeyCode::Up => go(tab, index.saturating_sub(1)),
        KeyCode::Down => go(tab, index + 1),
        KeyCode::PageUp => go(tab, index.saturating_sub(page)),
        KeyCode::PageDown => go(tab, index + page),
        KeyCode::Home => go(tab, 0),
        KeyCode::End => go(tab, shown.len() - 1),
        KeyCode::Enter if has_children => tab.folds.set(&tree, node, !open),
        _ => {}
    }
    true
}

/// The open board at `index` (wrapping) and what is on it, or `None` in the
/// snapshot when no board is open.
pub(super) fn read_board(store: &crate::board::BoardStore, index: usize) -> Result<BoardRead, Box<dyn std::error::Error>> {
    let boards = shown_boards(store)?;
    let waiting = waiting_for_operator(store);
    WAITING.store(waiting, std::sync::atomic::Ordering::SeqCst);
    if boards.is_empty() {
        return Ok(BoardRead { at: Instant::now(), snapshot: Ok(None), name: None, open: 0, waiting });
    }
    let board = boards[index % boards.len()].clone();
    let flows = crate::board::flow::FlowStore::beside(store);
    let snapshot = board_view::Snapshot {
        intents: store.intents(&board.id)?,
        facts: store.facts(&board.id)?,
        hints: store.hints(&board.id, false)?,
        flow: flows.get(&board.id)?,
        notes: flows.notes(&board.id)?,
        board,
    };
    Ok(BoardRead {
        at: Instant::now(),
        name: Some(snapshot.board.name.clone()),
        snapshot: Ok(Some(snapshot)),
        open: boards.len(),
        waiting,
    })
}

/// The boards on the board subtab, newest first: the open ones, and those
/// the agent finished whose write-up is due (still working, in `board_flow`; a
/// proposal from before the workflow too). Those written up are in review
/// and on (`tabs/projects_view.rs`).
pub(super) fn shown_boards(store: &crate::board::BoardStore) -> Result<Vec<crate::board::Board>, Box<dyn std::error::Error>> {
    use crate::board::BoardStatus;
    use crate::board::flow::Stage;
    let flows: std::collections::HashMap<String, crate::board::flow::Flow> =
        crate::board::flow::FlowStore::beside(store).all()?.into_iter().map(|flow| (flow.board_id.clone(), flow)).collect();
    Ok(store
        .boards(false)?
        .into_iter()
        .filter(|board| match (board.closed_at.is_none(), board.status, flows.get(&board.id)) {
            (true, _, _) => true,
            // Finished, or the agent's word that nothing is left: here until it
            // is written up, then in the operator's review.
            (false, BoardStatus::Solved | BoardStatus::ProposedSolved | BoardStatus::ProposedExhausted, Some(flow)) => flow.stage() == Stage::Working,
            (false, BoardStatus::ProposedSolved | BoardStatus::ProposedExhausted, None) => true,
            _ => false,
        })
        .collect())
}

/// Boards waiting for the operator: in review, or verified.
pub(super) fn waiting_for_operator(store: &crate::board::BoardStore) -> usize {
    crate::board::flow::waiting(&crate::board::flow::FlowStore::beside(store))
}

/// How many boards wait for the operator (in review, verified, proposed exhausted), as last counted.
pub(super) static WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// When they were last counted for the tab bar (at most every `WAITING_EVERY`).
pub(super) static WAITING_COUNTED: Mutex<Option<Instant>> = Mutex::new(None);
pub(super) const WAITING_EVERY: Duration = Duration::from_secs(10);

/// Boards waiting for the operator, for the tab bar's mark: the last count, and a
/// new one taken on a thread now and then (the store may be busy), with the
/// projects subtabs' counts.
/// The agent's redaction requests waiting for the operator, as last counted (with
/// the boards, `boards_waiting`).
pub(super) static REDACTIONS_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Audits waiting for the operator (`u audit [!]`), counted at most every
/// `WAITING_EVERY` on a thread of its own.
pub(super) fn audits_waiting() -> usize {
    static COUNTED: Mutex<Option<Instant>> = Mutex::new(None);
    static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let due = {
        let mut counted = COUNTED.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let due = counted.is_none_or(|at| at.elapsed() >= WAITING_EVERY);
        if due {
            *counted = Some(Instant::now());
        }
        due
    };
    if due && !cfg!(test) {
        let _ = std::thread::Builder::new().name("audits-waiting".into()).spawn(|| {
            let n = crate::audit::ops::waiting();
            if COUNT.swap(n, std::sync::atomic::Ordering::SeqCst) != n {
                poke();
            }
        });
    }
    COUNT.load(std::sync::atomic::Ordering::SeqCst)
}

pub(super) fn redactions_waiting() -> usize {
    REDACTIONS_WAITING.load(std::sync::atomic::Ordering::SeqCst)
}

pub(super) fn boards_waiting() -> usize {
    let due = {
        let mut counted = WAITING_COUNTED.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let due = counted.is_none_or(|at| at.elapsed() >= WAITING_EVERY);
        if due {
            *counted = Some(Instant::now());
        }
        due
    };
    if due && let Some(archive) = ARCHIVE.get().cloned() {
        let _ = std::thread::Builder::new().name("boards-waiting".into()).spawn(move || {
            let store = crate::board::BoardStore::beside_archive(&archive);
            let waiting = waiting_for_operator(&store);
            // And the agent's redaction requests, for the memory tab's mark.
            let asked = crate::redact::Book::beside_archive(&archive).pending().map(|pending| pending.len()).unwrap_or(0);
            if REDACTIONS_WAITING.swap(asked, std::sync::atomic::Ordering::SeqCst) != asked {
                poke();
            }
            if let Ok((_, counts)) = projects_view::read(&store, crate::board::flow::Stage::Deliverable, None) {
                projects_view::set_counts(counts);
            }
            if WAITING.swap(waiting, std::sync::atomic::Ordering::SeqCst) != waiting {
                poke();
            }
        });
    }
    WAITING.load(std::sync::atomic::Ordering::SeqCst)
}
