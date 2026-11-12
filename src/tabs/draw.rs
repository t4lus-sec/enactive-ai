//! Drawing a tab: the bar, the body and the footer, fitted to the screen.

use super::*;

pub(super) fn draw(open: &mut Option<Open>, display: &ActiveDisplay) -> io::Result<Outcome> {
    let Some(tab) = open.as_mut() else {
        return Ok(Outcome::Stay);
    };
    let size = crossterm::terminal::size()?;
    // The tab bar lists the shells, whichever tab is up.
    crate::vm::terms::touch();
    // What this view shows changed: its own feed only, so a subagent's
    // tokens do not lay out the commands, nor a shell's output a subagent.
    // The bar (shells, workers, running commands) follows at the second's
    // redraw.
    let feed = match &tab.view {
        View::Subagent(_) => crate::subagent::feed::version(),
        View::Shell { .. } => crate::vm::terms::version(),
        View::History | View::Page => display.record_revision(),
        View::Board | View::Memory | View::Share | View::Overview | View::Audit | View::Edit => 0,
        View::Review | View::Verification | View::Deliverables => 0,
    };
    let version = feed.wrapping_add(POKED.load(std::sync::atomic::Ordering::SeqCst));
    let now = Instant::now();
    if let Some((drawn_version, drawn_size, at)) = tab.painted
        && !tab.dirty
    {
        let since = now.duration_since(at);
        let changed = drawn_version != version || drawn_size != size;
        let board_due = (tab.view == View::Board && tab.board_read.as_ref().is_none_or(|read| now.duration_since(read.at) >= BOARD_POLL))
            || matches!(tab.view, View::Review | View::Verification | View::Deliverables);
        if since < FRAME || (!changed && !board_due && since < CLOCK) {
            return Ok(Outcome::Stay);
        }
    }
    if size.0 < 20 || size.1 < 6 {
        // Written once per size, not again every second.
        if tab.painted.is_none_or(|(_, drawn, _)| drawn != size) || !tab.frame.is_empty() {
            let mut stdout = io::stdout().lock();
            queue!(stdout, Clear(ClearType::All), MoveTo(0, 0), Print("too small · alt 1 back"))?;
            stdout.flush()?;
        }
        tab.frame.clear();
        tab.painted = Some((version, size, now));
        tab.dirty = false;
        return Ok(Outcome::Stay);
    }
    let (frame, cursor) = compose(tab, display, size, now);
    paint(&tab.frame, &frame, cursor, size.0 as usize)?;
    tab.frame = frame;
    tab.painted = Some((version, size, now));
    tab.dirty = false;
    Ok(Outcome::Stay)
}

/// The screen row a tab's body starts on, as last drawn: under the tab bar's
/// rows and its rule.
pub(super) static BODY_TOP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2);
/// How many rows the body had, as last drawn.
pub(super) static BODY_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(20);

pub(super) fn body_top() -> usize {
    BODY_TOP.load(std::sync::atomic::Ordering::Relaxed)
}

/// Every row of the screen, and where the cursor goes: the tab bar, a rule
/// under it, the view, its footer, and on a chat its prompt.
pub(super) fn compose(tab: &mut Open, display: &ActiveDisplay, (columns, rows): (u16, u16), now: Instant) -> (Vec<Line>, Option<(u16, u16)>) {
    // What can be clicked in it is registered as it is drawn.
    gui_view::begin();
    let (frame, cursor) = compose_frame(tab, display, (columns, rows), now);
    let mut frame = frame;
    if let Some(menu) = tab.menu.as_mut() {
        menu.draw(&mut frame, columns as usize);
    }
    tab.hits = gui_view::hits(&frame);
    tab.size = (columns, rows);
    (frame, cursor)
}

pub(super) fn compose_frame(tab: &mut Open, display: &ActiveDisplay, (columns, rows): (u16, u16), now: Instant) -> (Vec<Line>, Option<(u16, u16)>) {
    let width = columns as usize;
    let rows = rows as usize;
    // The open subtab's prompt is its board section's.
    let prompt = tab.view.has_prompt() && !(tab.view == View::Board && projects_view::section(tab) != projects_view::Section::Board);
    // What the agent is doing, beside its key while another tab covers it.
    let working = if matches!(tab.view, View::History | View::Page) { None } else { display.working() };
    let bar = tab_bar(Some(&tab.view), working, width);
    // The body starts under the bar's rows and the rule.
    let top = (bar.len() + 1).min(rows.saturating_sub(4));
    BODY_TOP.store(top, std::sync::atomic::Ordering::Relaxed);
    let body_rows = rows.saturating_sub(top + 1 + usize::from(prompt)).max(1);
    BODY_ROWS.store(body_rows, std::sync::atomic::Ordering::Relaxed);
    if let View::Subagent(number) = tab.view
        && tab.draft.is_empty()
    {
        tab.pinned = subagent_at(number).map(|worker| worker.run_id);
    }
    // A shell's key with no shell on it yet takes the first that comes.
    if !server_only(&tab.view)
        && let View::Shell { number, session: None } = tab.view
    {
        let session = crate::vm::terms::sessions().get(number - 1).cloned();
        tab.view = View::Shell { number, session };
    }
    let mut cursor_in_body = None;
    let mut history_loading = false;
    let (body, footer) = match &tab.view {
        view if server_only(view) => remote_body(view, width),
        View::Subagent(number) => subagent_body(*number, tab.expanded, width, now),
        // The projects tab's first subtab: its line on top, then its
        // sections (the board, its files, notes and reports).
        View::Board if projects_view::section(tab) == projects_view::Section::Board => {
            let (mut body, footer) = board_body(tab, width, body_rows.saturating_sub(2), now);
            let bar = open_item(tab).map(|item| projects_view::section_bar(tab, &item, width)).unwrap_or_default();
            body.insert(0, projects_view::subtab_line(&View::Board, width));
            body.insert(1, bar);
            (body, footer)
        }
        View::Board => {
            board_refresh(tab, now);
            let item = open_item(tab);
            projects_view::open_body(tab, item.as_ref(), width, body_rows)
        }
        View::Review | View::Verification | View::Deliverables => projects_view::body(tab, width, body_rows),
        View::History => history_body(tab, display, width),
        View::Page => page_body(tab, display, width),
        View::Memory => memory_view::body(tab, width, body_rows),
        View::Share => share_view::body(tab, width, body_rows),
        View::Overview => overview_view::body(tab, width, body_rows),
        View::Audit => audit_view::body(tab, width, body_rows),
        View::Edit => {
            tab.back = 0;
            let (body, footer, cursor) = editor_view::body(tab, width, body_rows);
            cursor_in_body = cursor;
            (body, footer)
        }
        View::Shell { number, session } => {
            let (body, footer, cursor, loading) = shell_body(*number, session.as_deref(), tab.back, (width, body_rows));
            cursor_in_body = cursor;
            history_loading = loading;
            (body, footer)
        }
    };
    let mut frame = Vec::with_capacity(rows);
    frame.extend(bar.into_iter().take(top.saturating_sub(1)));
    frame.push(vec![(Style::Dim, "─".repeat(width))]);
    let longest = body.len().saturating_sub(body_rows);
    // A shell's history is on its way: keep the place asked for until it is.
    if !history_loading {
        tab.back = tab.back.min(longest);
    }
    let start = longest.saturating_sub(tab.back);
    frame.extend(body.into_iter().skip(start).take(body_rows));
    frame.resize(top + body_rows, Vec::new());
    // Text selected with the mouse: lined up with what this frame shows,
    // and shown.
    if let Some(selection) = tab.selection.as_mut() {
        let shown: Vec<String> =
            frame[top..top + body_rows].iter().map(|row| row.iter().map(|(_, text)| plain_text(text)).collect()).collect();
        selection.take_frame(&shown);
        selection.mark(&mut frame[top..top + body_rows]);
    }
    let Footer { mut left, mut hints } = footer;
    if tab.back > 0 {
        // First, so a narrow footer keeps it.
        left.insert(0, (Style::Dim, format!("↑ {} rows  ", tab.back)));
        if tab.view != View::History {
            hints.insert(0, (if prompt { "end" } else { "⇧end" }, "follow"));
        }
    }
    // What a key or a click just did takes the left of the footer for a
    // while; the buttons stay, to be clicked again.
    let left = match &tab.note {
        Some((text, problem, at)) if now.duration_since(*at) < if text.starts_with("[?] ") { ARM_FOR } else { NOTE_FOR } => {
            let style = match (problem, text.starts_with("[?] ")) {
                (true, _) => Style::Problem,
                (false, true) => Style::Warn,
                (false, false) => Style::Plain,
            };
            vec![(style, text.clone())]
        }
        _ => left,
    };
    let (line, actions) = gui_view::footer(left, &hints, width);
    tab.actions = actions;
    frame.push(line);
    if prompt {
        let (prompt, cursor) = prompt_line(tab, width);
        frame.push(prompt);
        return (frame, Some((cursor as u16, (rows - 1) as u16)));
    }
    // A shell's cursor where the shell has it, when that row is on screen;
    // otherwise none: a cursor parked in a corner reads as one stuck there.
    let cursor = cursor_in_body
        .filter(|_| tab.back == 0)
        .and_then(|(column, row): (usize, usize)| row.checked_sub(start).filter(|row| *row < body_rows).map(|row| (column, row)))
        .map(|(column, row)| (column.min(width.saturating_sub(1)) as u16, (row + top) as u16));
    (frame, cursor)
}

/// A view's footer: what it shows on the left, its keys on the right.
pub(super) struct Footer {
    pub(super) left: Line,
    pub(super) hints: Vec<(&'static str, &'static str)>,
}

impl Footer {
    pub(super) fn new(left: Line, hints: Vec<(&'static str, &'static str)>) -> Self {
        Self { left, hints }
    }
}

/// A view with nothing to show yet: what it is, and (when it helps) how it fills.
pub(super) fn empty(title: &str, detail: &str, width: usize) -> Vec<Line> {
    let mut out = vec![Vec::new(), vec![(Style::Plain, format!("  {title}"))]];
    if !detail.is_empty() {
        out.extend(flow(&[(Style::Dim, format!("  {detail}"))], width, 2));
    }
    out
}

/// `segments` wrapped at spaces to `width`, every row after the first
/// indented `indent` columns (the first is too, when it starts a paragraph:
/// pass the indent in its text). A word longer than a row is broken.
pub(super) fn flow(segments: &[(Style, String)], width: usize, indent: usize) -> Vec<Line> {
    let width = width.max(indent + 8);
    let mut lines: Vec<Line> = vec![Vec::new()];
    let mut used = 0;
    let push = |lines: &mut Vec<Line>, style: Style, text: &str| {
        let line = lines.last_mut().expect("one line");
        match line.last_mut() {
            Some((last, piece)) if *last == style => piece.push_str(text),
            _ => line.push((style, text.to_string())),
        }
    };
    for (style, text) in segments {
        // Words with the spaces after them, so a row never starts with one.
        let mut words: Vec<String> = Vec::new();
        for c in text.chars() {
            let c = if c == '\t' || c == '\n' { ' ' } else { c };
            if c.is_control() {
                continue;
            }
            match words.last_mut() {
                Some(word) if !(c != ' ' && word.ends_with(' ')) => word.push(c),
                _ => words.push(c.to_string()),
            }
        }
        for word in words {
            let bare = text_width(word.trim_end());
            if used > indent && used + bare > width {
                trim_end(lines.last_mut().expect("one line"));
                lines.push(vec![(Style::Plain, " ".repeat(indent))]);
                used = indent;
                if word.trim().is_empty() {
                    continue;
                }
            }
            let mut piece = String::new();
            for c in word.chars() {
                let w = c.width().unwrap_or(0);
                if used + w > width {
                    if c == ' ' {
                        break;
                    }
                    push(&mut lines, *style, &piece);
                    piece.clear();
                    trim_end(lines.last_mut().expect("one line"));
                    lines.push(vec![(Style::Plain, " ".repeat(indent))]);
                    used = indent;
                }
                piece.push(c);
                used += w;
            }
            push(&mut lines, *style, &piece);
        }
    }
    if let Some(last) = lines.last_mut() {
        trim_end(last);
    }
    lines
}

/// A row without the spaces at its end (a wrapped word's).
pub(super) fn trim_end(line: &mut Line) {
    while let Some((_, text)) = line.last_mut() {
        let kept = text.trim_end_matches(' ').len();
        if kept > 0 {
            text.truncate(kept);
            return;
        }
        line.pop();
    }
}

/// Write the rows that changed since `previous`, then put the cursor on the
/// prompt.
/// One write per frame, as one synchronized update where the terminal
/// supports it (`?2026`): it shows whole frames, never one half drawn.
pub(super) fn paint(previous: &[Line], frame: &[Line], cursor: Option<(u16, u16)>, width: usize) -> io::Result<()> {
    let mut bytes: Vec<u8> = Vec::with_capacity(16 * 1024);
    queue!(bytes, crossterm::terminal::BeginSynchronizedUpdate)?;
    paint_to(&mut bytes, previous, frame, cursor, width)?;
    queue!(bytes, crossterm::terminal::EndSynchronizedUpdate)?;
    let mut stdout = io::stdout().lock();
    stdout.write_all(&bytes)?;
    stdout.flush()
}

/// `cursor`: where the cursor shows, or `None` to leave it hidden (a view
/// with no text cursor, or a shell whose program hid its own).
pub(super) fn paint_to(stdout: &mut impl Write, previous: &[Line], frame: &[Line], cursor: Option<(u16, u16)>, width: usize) -> io::Result<()> {
    queue!(stdout, Hide)?;
    for (index, line) in frame.iter().enumerate() {
        if previous.get(index) == Some(line) {
            continue;
        }
        queue!(stdout, MoveTo(0, index as u16))?;
        let mut used = 0;
        'segments: for (style, text) in line {
            apply_style(stdout, *style)?;
            if *style == Style::Raw {
                // Cut to the width already (`page::rows`), codes and all.
                queue!(stdout, Print(text))?;
                used += crate::page::width(text);
                continue;
            }
            let mut piece = String::new();
            for c in text.chars() {
                let c = if c.is_control() { ' ' } else { c };
                let w = c.width().unwrap_or(0);
                if used + w > width {
                    queue!(stdout, Print(piece))?;
                    break 'segments;
                }
                used += w;
                piece.push(c);
            }
            queue!(stdout, Print(piece))?;
        }
        apply_style(stdout, Style::Plain)?;
        // A row that reaches the last column leaves the cursor waiting to
        // wrap, on that column: erasing to the end of the line from there
        // would take its last character with it. Full, it has nothing to
        // erase anyway.
        if used < width {
            queue!(stdout, Clear(ClearType::UntilNewLine))?;
        }
    }
    if let Some((column, row)) = cursor {
        queue!(stdout, MoveTo(column, row), Show)?;
    }
    stdout.flush()
}

/// The tab bar: the agent's own tabs (alt) on the left half, the shells (alt⇧)
/// on the right, the `│` between them pinned at the middle; a half that
/// runs out of room carries on, on the next row. The tab that is up is in
/// the chevron's cyan.
///
/// ```text
///  alt  1 the agent  2 check the logs [+]  p projects   │  alt⇧  1 main  2 build
///       m memory  s share  o overview              │
/// ```
///
/// `view`: the tab that is up, or none for the agent's own page. `working`: what
/// the agent is doing, shown beside its tab while another covers it.
pub(super) fn tab_bar(view: Option<&View>, working: Option<String>, width: usize) -> Vec<Line> {
    // Labels as long as they can be on two rows, then shorter, then closer;
    // never more than three rows (a narrow terminal keeps room for the tab).
    let mut best: Vec<Line> = Vec::new();
    for (limit, gap) in [(18, 2), (12, 2), (8, 2), (8, 1), (5, 1)] {
        let bar = tab_bar_with(view, working.as_deref(), limit, gap, width);
        let fewer = best.is_empty() || bar.len() < best.len();
        if fewer {
            best = bar;
        }
        if best.len() <= 2 {
            break;
        }
    }
    best.truncate(MAX_BAR_ROWS);
    best
}

pub(super) const MAX_BAR_ROWS: usize = 3;

/// The tab bar for the agent's own page, where `1 s0ma` is the one that is up:
/// written out with its colours, a row to a line, and how many columns its
/// widest row takes. Above the idle prompt and atop the steering pane; the
/// page keeps as many rows for it (`page::set_bar_rows`).
pub fn main_bar(width: usize) -> (String, usize) {
    // The VM's shells are read once at start for it (shells from before a
    // restart); after that, when the agent opens or closes one and while a tab is up.
    static FIRST_LIST: std::sync::Once = std::sync::Once::new();
    FIRST_LIST.call_once(|| {
        if !cfg!(test) {
            crate::vm::terms::touch();
        }
    });
    gui_view::begin();
    let rows = tab_bar(None, None, width);
    crate::page::set_bar_rows(rows.len());
    *page_hits() = gui_view::hits(&rows);
    let cells = rows.iter().map(|line| line.iter().map(|(_, text)| text_width(text)).sum::<usize>()).max().unwrap_or(0);
    let mut written: Vec<String> = Vec::new();
    for line in &rows {
        let mut out: Vec<u8> = Vec::new();
        for (style, text) in line {
            let _ = apply_style(&mut out, *style);
            out.extend_from_slice(text.as_bytes());
        }
        let _ = apply_style(&mut out, Style::Plain);
        written.push(String::from_utf8_lossy(&out).into_owned());
    }
    (written.join("\n"), cells)
}

/// Where the page's tab bar (`main_bar`) can be clicked, as last made.
pub(super) static PAGE_HITS: Mutex<Vec<gui_view::Hit>> = Mutex::new(Vec::new());

pub(super) fn page_hits() -> MutexGuard<'static, Vec<gui_view::Hit>> {
    PAGE_HITS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A click on the agent's own page at `(column, row)`: on its tab bar, the key
/// that opens the tab clicked (Alt+A for the approval mark), as the page
/// takes keys.
pub fn page_click(column: u16, row: u16) -> Option<KeyEvent> {
    let hits = page_hits();
    match &gui_view::at(&hits, row as usize, column as usize)?.click {
        gui_view::Click::Keys(keys) => keys.first().copied(),
        gui_view::Click::Approve => Some(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::ALT)),
        _ => None,
    }
}

/// One tab in the bar: its label, and a mark after it; a click on it opens
/// the tab, as its key does (`key`, Alt's; `terminal`, Alt+Shift's).
pub(super) fn bar_item(up: bool, key: char, label: String, style: Style, mark: Option<(Style, &str)>, gap: usize) -> Line {
    bar_item_in(Group::Chat, up, key, label, style, mark, gap)
}

pub(super) fn bar_item_in(group: Group, up: bool, key: char, label: String, style: Style, mark: Option<(Style, &str)>, gap: usize) -> Line {
    let modifiers = if group == Group::Terminal { KeyModifiers::ALT | KeyModifiers::SHIFT } else { KeyModifiers::ALT };
    let click = gui_view::Click::Keys(vec![KeyEvent::new(KeyCode::Char(key), modifiers)]);
    // The approval mark is its own button: a click is one press of Alt+A.
    let approve = |mark: &str| gui_view::hot(Style::Warn, format!(" {mark}"), gui_view::Click::Approve);
    let mut line = Line::new();
    if up {
        line.push((Style::Plain, " ".repeat(gap - 1)));
        match mark {
            Some((_, text)) if text.contains("approve") => {
                line.push(gui_view::hot(Style::Tab, format!(" {label}"), click));
                line.push(approve(text));
                line.push((Style::Tab, " ".into()));
            }
            _ => {
                let mark = mark.map(|(_, mark)| format!(" {mark}")).unwrap_or_default();
                line.push(gui_view::hot(Style::Tab, format!(" {label}{mark} "), click));
            }
        }
    } else {
        line.push((Style::Plain, " ".repeat(gap)));
        line.push(gui_view::hot(style, label, click.clone()));
        match mark {
            Some((_, text)) if text.contains("approve") => line.push(approve(text)),
            Some((style, text)) => line.push(gui_view::hot(style, format!(" {text}"), click)),
            None => {}
        }
        line.push((Style::Plain, " ".into()));
    }
    line
}

/// `items` after `lead`, flowed into rows `room` wide; a row after the
/// first starts under the first item.
pub(super) fn flow_items(lead: &str, items: Vec<Line>, room: usize) -> Vec<Line> {
    let width = |line: &Line| line.iter().map(|(_, text)| text_width(text)).sum::<usize>();
    let indent = text_width(lead);
    let mut rows: Vec<Line> = vec![vec![(Style::Dim, lead.to_string())]];
    for item in items {
        let used = rows.last().map(width).unwrap_or(0);
        if used + width(&item) > room && used > indent {
            rows.push(vec![(Style::Plain, " ".repeat(indent))]);
        }
        rows.last_mut().expect("one row at least").extend(item);
    }
    rows
}

pub(super) fn tab_bar_with(view: Option<&View>, working: Option<&str>, limit: usize, gap: usize, width: usize) -> Vec<Line> {
    let mut left: Vec<Line> = Vec::new();
    // [!]: something waits for the operator's approval (alt+a twice).
    let approve = (!crate::approval::pending().is_empty()).then_some((Style::Warn, "[!] approve"));
    let mut first = bar_item(view.is_none_or(|view| matches!(view, View::History | View::Page)), '1', "s0ma".into(), Style::Plain, approve, gap);
    if let Some(working) = working {
        first.pop();
        first.push((Style::Status, format!(" · {} ", clip(working, limit.max(12)))));
    }
    left.push(first);
    let key_of = |number: usize| if number == 10 { '0' } else { char::from(b'0' + number as u8) };
    let workers = crate::subagent::feed::tabs();
    for (index, worker) in workers.iter().enumerate().take(9) {
        let number = index + 2;
        let mark = match worker.status.as_deref() {
            None => None,
            Some("completed") => Some((Style::Added, "[+]")),
            Some(_) => Some((Style::Removed, "[-]")),
        };
        let label = clip(&worker.label, limit);
        let style = if worker.running { Style::Plain } else { Style::Dim };
        left.push(bar_item(view == Some(&View::Subagent(number)), key_of(number), label, style, mark, gap));
    }
    // A key with nothing on it is shown only while it is the one up, so
    // the bar says where you are.
    if let Some(View::Subagent(number)) = view
        && *number >= workers.len().min(9) + 2
    {
        left.push(bar_item(true, key_of(*number), "empty".into(), Style::Plain, None, gap));
    }
    // [!]: a board waits for the operator: in review, or verified.
    let waiting = (boards_waiting() > 0).then_some((Style::Warn, "[!]"));
    let projects = view.is_some_and(projects_view::is_projects);
    left.push(bar_item(projects, 'p', "projects".into(), Style::Plain, waiting, gap));
    // [!]: the agent asks to redact something (the memory tab's redactions).
    let asked = (redactions_waiting() > 0).then_some((Style::Warn, "[!]"));
    // [!]: an audit waits for the operator (a proposal, a weaker check, a
    // model-only reason, a result to post, an adapter).
    let audits = (audits_waiting() > 0).then_some((Style::Warn, "[!]"));
    left.push(bar_item(view == Some(&View::Audit), 'u', "audit".into(), Style::Plain, audits, gap));
    left.push(bar_item(view == Some(&View::Memory), 'm', "memory".into(), Style::Plain, asked, gap));
    left.push(bar_item(view == Some(&View::Share), 's', "share".into(), Style::Plain, None, gap));
    // The editor, while a file is open in it (or it is up): the file, how
    // many are open, and [+] for unsaved changes.
    match editor_view::label() {
        Some((name, count, dirty)) => {
            let label = if count > 1 { format!("{} ·{count}", clip(&name, limit)) } else { clip(&name, limit) };
            left.push(bar_item(view == Some(&View::Edit), 'e', label, Style::Plain, dirty.then_some((Style::Warn, "[+]")), gap));
        }
        None if view == Some(&View::Edit) => left.push(bar_item(true, 'e', "edit".into(), Style::Plain, None, gap)),
        None => {}
    }
    left.push(bar_item(view == Some(&View::Overview), 'o', "overview".into(), Style::Plain, None, gap));
    let mut right: Vec<Line> = Vec::new();
    let sessions = crate::vm::terms::sessions();
    let netboxes = crate::vm::terms::netboxes();
    for (index, session) in sessions.iter().enumerate().take(10) {
        let number = index + 1;
        let up = matches!(view, Some(View::Shell { session: Some(shown), .. }) if shown == session);
        right.push(bar_item_in(Group::Terminal, up, key_of(number), clip(&shell_label(session), limit.max(8)), Style::Plain, netbox_mark(&netboxes, session), gap));
    }
    if let Some(View::Shell { number, session: None }) = view {
        right.push(bar_item_in(Group::Terminal, true, key_of(*number), "empty".into(), Style::Plain, None, gap));
    }
    // The halves, and the pipe between them on every row.
    let half = width.saturating_sub(1) / 2;
    let left = flow_items("", left, half);
    let right = flow_items("", right, width.saturating_sub(half + 1));
    let rows = left.len().max(right.len());
    (0..rows)
        .map(|at| {
            let mut line: Line = left.get(at).cloned().unwrap_or_default();
            let used: usize = line.iter().map(|(_, text)| text_width(text)).sum();
            line.push((Style::Plain, " ".repeat(half.saturating_sub(used))));
            line.push((Style::Dim, "│".into()));
            line.extend(right.get(at).cloned().unwrap_or_default());
            line
        })
        .collect()
}

/// A netbox's mark in the tab bar: `[!]` the operator has the wheel, `[*]` a
/// command of the agent's is running in it.
pub(super) fn netbox_mark(netboxes: &[crate::vm::netboxes::Netbox], session: &str) -> Option<(Style, &'static str)> {
    let name = crate::vm::netboxes::name_of(session)?;
    let netbox = netboxes.iter().find(|netbox| netbox.name == name)?;
    if netbox.wheel {
        Some((Style::Warn, "[!]"))
    } else if !netbox.running.is_empty() {
        Some((Style::Work, "[*]"))
    } else {
        None
    }
}

/// A shell as the tab bar names it: by its name (`w·b3-1` a subagent's).
pub(super) fn shell_label(session: &str) -> String {
    match crate::vm::netboxes::name_of(session) {
        Some(name) => crate::vm::netboxes::display(name),
        None => clip(session, 16),
    }
}

/// What the operator greyed out on each board, by board id: for his
/// reading only (the agent never sees it). Kept beside the board's database, so it
/// stays as he left it.
pub(super) static GREYED: Mutex<Option<Greyed>> = Mutex::new(None);

/// When the greyed file was read, and the rows greyed on each board.
pub(super) type Greyed = (Option<std::time::SystemTime>, HashMap<String, HashSet<String>>);

pub(super) fn grey_path() -> Option<PathBuf> {
    ARCHIVE.get().map(|archive| archive.with_file_name("board-grey.json"))
}

pub(super) fn grey_stamp(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|metadata| metadata.modified()).ok()
}

/// `change` made to what is greyed on `board`, and kept. The file is read
/// again whenever another agent (an attached terminal) wrote it since, so
/// neither writes over the other's; a file that does not read is left as it
/// is and what this one had is kept; it is written whole, beside, and then
/// renamed over the old.
pub(super) fn with_greyed<T>(board: &str, change: impl FnOnce(&mut HashSet<String>) -> T) -> T {
    let mut kept = GREYED.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let path = grey_path();
    let stamp = path.as_deref().and_then(grey_stamp);
    let stale = kept.as_ref().is_none_or(|(read_at, _)| *read_at != stamp);
    if stale {
        let read = path.as_deref().and_then(|path| std::fs::read_to_string(path).ok()).and_then(|json| serde_json::from_str(&json).ok());
        match (read, kept.as_mut()) {
            (Some(all), _) => *kept = Some((stamp, all)),
            (None, Some((read_at, _))) => *read_at = stamp,
            (None, None) => *kept = Some((stamp, HashMap::new())),
        }
    }
    let (read_at, all) = kept.as_mut().expect("read above");
    let before = all.get(board).cloned().unwrap_or_default();
    let set = all.entry(board.to_string()).or_default();
    let out = change(set);
    let changed = *set != before;
    if set.is_empty() {
        all.remove(board);
    }
    if changed && let (Some(path), Ok(json)) = (path, serde_json::to_string(&*all))
        && crate::util::write_atomic(&path, json).is_ok() {
            *read_at = grey_stamp(&path);
        }
    out
}

/// The board's tree, greyed out as the operator left it.
pub(super) fn grey_tree(snapshot: &board_view::Snapshot, now: chrono::DateTime<chrono::Local>) -> board_view::Tree {
    let mut tree = board_view::Tree::build(snapshot, now);
    with_greyed(&snapshot.board.id, |greyed| tree.sink(greyed));
    tree
}

/// The selected lead or finding's short id, for the prompt.
pub(super) fn board_target(tab: &Open) -> Option<String> {
    let snapshot = tab.board_read.as_ref()?.snapshot.as_ref().ok()?.as_ref()?;
    let tree = grey_tree(snapshot, chrono::Local::now());
    let node = tree.find(tab.board_selected.as_deref()?)?;
    board_view::prompt_target(&tree, Some(node))
}

pub(super) fn prompt_line(tab: &Open, width: usize) -> (Line, usize) {
    let (label, live) = match &tab.view {
        View::Page => (" you ›".to_string(), true),
        View::Shell { .. } | View::History | View::Memory | View::Share | View::Overview | View::Audit | View::Edit => return (Vec::new(), 0),
        View::Review | View::Verification | View::Deliverables => return (Vec::new(), 0),
        View::Subagent(number) => match subagent_at(*number) {
            // Named for the one the draft is for, which `submit` checks.
            Some(worker) if worker.running => {
                let to = tab.pinned.as_deref().filter(|_| !tab.draft.is_empty()).unwrap_or(&worker.run_id);
                (format!(" {} ›", short_id(to)), true)
            }
            Some(_) => (" finished".to_string(), false),
            None => (String::new(), false),
        },
        View::Board => match tab.board_read.as_ref().and_then(|read| read.name.as_deref()) {
            // Named for what it adds, and the lead or finding it will be
            // about (a lead hangs from a finding only).
            Some(_) => {
                let target = board_target(tab);
                let label = match (tab.board_adding, target) {
                    (Adding::Hint, Some(target)) => format!(" hint on {target} ›"),
                    (Adding::Hint, None) => " hint ›".to_string(),
                    (Adding::Lead, Some(target)) if target.starts_with('f') => format!(" lead from {target} ›"),
                    (Adding::Lead, _) => " lead ›".to_string(),
                    (Adding::Supports, Some(target)) => format!(" finding on {target}, supports ›"),
                    (Adding::Supports, None) => " finding, supports ›".to_string(),
                    (Adding::RulesOut, Some(target)) => format!(" finding on {target}, rules out ›"),
                    (Adding::RulesOut, None) => " finding, rules out ›".to_string(),
                    (Adding::Claim(i), target) => {
                        let audits = tab.board_read.as_ref().and_then(|read| read.name.as_deref()).map(crate::audit::ops::for_board).unwrap_or_default();
                        let to = audits.get(i).map(|a| a.name.clone()).unwrap_or_default();
                        let from = target.filter(|t| t.starts_with('f')).map(|t| format!(" from {t}")).unwrap_or_default();
                        if audits.len() > 1 { format!(" claim{from} to {to} ›") } else { format!(" claim{from} ›") }
                    }
                };
                (label, true)
            }
            None => (String::new(), false),
        },
    };
    if !live {
        return (vec![(Style::Dim, label)], 0);
    }
    let label_width = text_width(&label) + 1;
    let mut line: Line = vec![(Style::Operator, label), (Style::Plain, " ".into())];
    // One row: a newline shows as ⏎, and the text scrolls to keep the cursor
    // in sight.
    let shown: String = tab.draft.text().chars().map(|c| if c == '\n' { '⏎' } else { c }).collect();
    let before: String = tab.draft.text()[..tab.draft.cursor()].chars().map(|c| if c == '\n' { '⏎' } else { c }).collect();
    let room = width.saturating_sub(label_width + 1).max(1);
    let cursor_at = text_width(&before);
    let skip = cursor_at.saturating_sub(room.saturating_sub(1));
    let mut visible = String::new();
    let mut at = 0;
    for c in shown.chars() {
        let w = c.width().unwrap_or(0);
        if at >= skip && at + w <= skip + room {
            visible.push(c);
        }
        at += w;
    }
    if visible.is_empty() {
        let hint = match (&tab.view, tab.board_adding) {
            (View::Board, Adding::Hint) => "hint for s0ma · ^t a lead or finding",
            (View::Board, Adding::Lead) => "a lead of yours to explore (p1–p10 first: its priority)",
            (View::Board, Adding::Claim(_)) => "a claim for the audit: yours, accepted",
            (View::Board, _) => "what you found",
            (View::Page, _) => "to s0ma",
            _ => "a direction, taken before its next step",
        };
        line.push((Style::Work, hint.to_string()));
        return (line, label_width.min(width.saturating_sub(1)));
    }
    line.push((Style::Plain, visible));
    (line, (label_width + cursor_at - skip).min(width.saturating_sub(1)))
}
