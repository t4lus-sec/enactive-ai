//! The audit tab (Alt+U), a submodule of `tabs.rs`: every audit (attached
//! to projects, and standing), one at a time with Tab, Ctrl+N and Ctrl+P;
//! its head (status, scope, levels, settings), a row of filters, and one
//! tree (`audit_tree.rs`): each claim once, its checks under it, their
//! evidence under them, in plan order or by file (f). The selected row in
//! full under it; at full width (one pane), the code beside it. Every key
//! is also a footer button and a right-click menu item.
//!
//! The operator acts here; the harness decides. Nothing in this tab writes
//! a verdict: it accepts, rejects, rewords, takes, waives, posts, pauses,
//! changes settings and starts, and the harness checks (`audit/run.rs`).

use std::collections::BTreeSet;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::audit::{self, Audit, Claim, Level, Posted, Repro, Review, Status, Verdict, Who};
use crate::style::{Line, Style};

use super::audit_tree::{self, FILTERS, Filter, Node, Row};
use super::gui_view::{self, Click, Pick};
use super::{Footer, Open, Outcome};

/// The code beside the tree from this many columns, in one pane.
const CODE_BESIDE: usize = 140;

/// What is being typed at the tab's prompt.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Typing {
    /// A note to the agent about a claim (a board hint).
    Note,
    /// The claim reworded: the operator's wording is final.
    Edit,
    /// Why its check goes back.
    Back,
    /// His own claim.
    Add,
    /// `key=value` settings.
    Settings,
    /// Why a proposal is declined.
    Decline,
    /// A direction to the worker formalizing the claim.
    Steer,
    /// `board [lead]` to link the claim (or the audit) to.
    Link,
    /// The tree's search.
    Filter,
    /// The standing audit to merge this one into, at complete.
    Merge,
    /// A plan row's priority, 1 to 10.
    Priority,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Input {
    pub typing: Typing,
    pub claim: Option<String>,
    pub text: Vec<char>,
    pub cursor: usize,
}

#[derive(Default)]
pub(super) struct State {
    /// Which audit: None is every audit (the portfolio head).
    pub audit: Option<String>,
    /// Which claims the tree shows, and how: in plan order or by file.
    pub show: Filter,
    pub by_file: bool,
    /// Rows folded and unfolded by the operator, by key (the rest as built).
    pub open: BTreeSet<String>,
    pub closed: BTreeSet<String>,
    /// The selected row, by key, so it stays put as the tree changes.
    pub selected_key: Option<String>,
    pub selected: usize,
    pub top: usize,
    pub search: String,
    pub input: Option<Input>,
    /// A key pressed once on a claim, to be pressed again.
    pub armed: Option<(char, String)>,
    pub strip_scroll: usize,
    /// The claim view, open on a claim (`tabs/claim_view.rs`).
    pub view: Option<super::claim_view::State>,
    /// The portfolio's selected row.
    pub portfolio: usize,
    /// Embedded in a project's audit section: its audits, the only ones
    /// stepped through (`[` `]`), and no portfolio.
    pub scope: Option<Vec<String>>,
    /// The code beside the tree, for the selected row (by its key). Its
    /// `pane` says what has the keys: 0 the tree, 1 the code, 2 the check.
    pub preview: Option<(String, super::claim_view::State)>,
    /// The column the code beside the tree starts at (None: not shown).
    pub code_left: Option<usize>,
    /// The portfolio shows the archive too: archived audits, and those
    /// stopped, closed or declined.
    pub show_archive: bool,
    /// The settings window, open at its selected row.
    pub settings: Option<usize>,
}

impl State {
    /// What of the code beside the tree has the keys: 0 none (the tree),
    /// 1 the code, 2 the check.
    fn code_focus(&self) -> usize {
        self.preview.as_ref().map_or(0, |(_, view)| view.pane)
    }
}

/// Everything one frame shows, read from the store.
struct Read {
    audits: Vec<Audit>,
    audit: Option<Audit>,
    claims: Vec<Claim>,
}

fn read(state: &State) -> Read {
    let Some(store) = audit::store() else { return Read { audits: Vec::new(), audit: None, claims: Vec::new() } };
    // The live ones first; the archive after them, when it is shown.
    let all = store.audits().unwrap_or_default();
    let mut audits: Vec<Audit> = all.iter().filter(|a| a.shown() || state.audit.as_deref() == Some(a.name.as_str())).cloned().collect();
    if state.show_archive {
        audits.extend(all.iter().filter(|a| !a.shown() && state.audit.as_deref() != Some(a.name.as_str())).cloned());
    }
    let audit = state.audit.as_ref().and_then(|name| audits.iter().find(|a| &a.name == name).cloned());
    let claims = audit.as_ref().map(|a| store.claims(&a.name).unwrap_or_default()).unwrap_or_default();
    Read { audits, audit, claims }
}

/// A level's colour on the bar: dim (documented) to green (proven).
fn level_style(level: Level) -> Style {
    match level {
        Level::Documented => Style::Dim,
        Level::Tested => Style::Work,
        Level::Property => Style::Status,
        Level::Bounded => Style::Accent,
        Level::Model | Level::Proven => Style::Success,
        Level::Assumed => Style::Operator,
    }
}

/// The levels bar: every claim, coloured by level; unreviewed results
/// shown apart (`[?] N unreviewed`), never counted as assurance.
fn levels_bar(claims: &[Claim], width: usize) -> Line {
    let mut out: Line = vec![(Style::Status, "  levels  ".into())];
    let shown: Vec<&Claim> = claims.iter().filter(|c| !matches!(c.review, Review::Rejected | Review::Dropped)).collect();
    if shown.is_empty() {
        out.push((Style::Dim, "no claims yet".into()));
        return out;
    }
    let bar = 24.min(width.saturating_sub(60)).max(8);
    let mut counts: Vec<(Level, usize)> = Vec::new();
    let mut unreviewed = 0;
    for claim in &shown {
        if claim.review != Review::Accepted && !claim.is_assumption() && claim.verdict != Verdict::Unchecked {
            unreviewed += 1;
            continue;
        }
        let level = if claim.is_assumption() {
            Level::Assumed
        } else if claim.counts() || (claim.verdict == Verdict::Holds && claim.review == Review::Accepted && !claim.waived && claim.unsound.is_none()) {
            claim.level
        } else {
            Level::Documented
        };
        match counts.iter_mut().find(|(l, _)| *l == level) {
            Some((_, n)) => *n += 1,
            None => counts.push((level, 1)),
        }
    }
    counts.sort_by_key(|(level, _)| *level);
    let total = shown.len();
    let mut used = 0;
    for (level, n) in &counts {
        let cells = (n * bar).div_ceil(total).min(bar - used);
        used += cells;
        out.push((level_style(*level), "█".repeat(cells)));
    }
    let hatched = bar.saturating_sub(used).min((unreviewed * bar).div_ceil(total));
    out.push((Style::Plain, "░".repeat(hatched)));
    out.push((Style::Dim, "░".repeat(bar.saturating_sub(used + hatched))));
    let words: Vec<String> = counts.iter().map(|(level, n)| format!("{} {n}", if *level == Level::Documented { "doc" } else { level.label() })).collect();
    out.push((Style::Dim, format!("  {}", words.join(" · "))));
    if unreviewed > 0 {
        out.push((Style::Dim, format!(" · [?] {unreviewed} unreviewed")));
    }
    out
}

// ------------------------------------------------------------------ body

/// The audit section of a project (`tabs/projects_view.rs`): the tab embedded,
/// on the project's audits.
pub(super) fn embedded_body(tab: &mut Open, board: &str, width: usize, rows: usize) -> (Vec<Line>, Footer) {
    embed(tab, board);
    let (lines, mut footer) = body(tab, width, rows);
    footer.hints.retain(|(key, _)| !matches!(*key, "tab" | "esc"));
    (lines, footer)
}

fn embed(tab: &mut Open, board: &str) {
    let names: Vec<String> = crate::audit::ops::for_board(board).into_iter().map(|a| a.name).collect();
    if tab.audit.audit.as_ref().is_none_or(|name| !names.contains(name)) {
        tab.audit.audit = names.first().cloned();
        tab.audit.show = Filter::All;
        tab.audit.selected = 0;
        tab.audit.selected_key = None;
        tab.audit.top = 0;
        tab.audit.view = None;
    }
    tab.audit.scope = Some(names);
}

/// A key in a project's audit section.
pub(super) fn embedded_key(tab: &mut Open, board: &str, key: &KeyEvent) -> Outcome {
    embed(tab, board);
    self::key(tab, key)
}

/// Whether the audit tab takes every key now: a prompt, or the claim view.
pub(super) fn holds_keys(tab: &Open) -> bool {
    tab.audit.input.is_some() || tab.audit.view.is_some() || tab.audit.settings.is_some()
}

pub(super) fn body(tab: &mut Open, width: usize, rows_left: usize) -> (Vec<Line>, Footer) {
    if tab.view == super::View::Audit {
        tab.audit.scope = None;
    }
    if tab.audit.view.is_some() {
        return super::claim_view::body(tab, width, rows_left);
    }
    let data = read(&tab.audit);
    if tab.audit.settings.is_some()
        && let Some(audit) = data.audit.clone()
    {
        return super::audit_settings::body(tab, &audit, width, rows_left);
    }
    let state = &mut tab.audit;
    // Stepping keeps to audits that exist.
    if state.audit.as_ref().is_some_and(|name| !data.audits.iter().any(|a| &a.name == name)) {
        state.audit = None;
    }
    let Some(audit) = data.audit.clone() else { return portfolio(state, &data, width, rows_left) };
    let mut out: Vec<Line> = Vec::new();
    // Head.
    let (mark, status) = audit.status_mark();
    let status_style = match mark {
        "[!]" => Style::Warn,
        "[+]" => Style::Success,
        _ => Style::Warn,
    };
    let mut head: Line = vec![(Style::Bold, format!("  {}  ", audit.name)), (status_style, format!("{mark} {status}"))];
    head.push((Style::Dim, format!("  {}", audit.board.as_ref().map(|b| format!("for {b}")).unwrap_or_else(|| "standing".into()))));
    let target = audit.targets.first().cloned().unwrap_or_default();
    head.push((Style::Dim, format!(
        "   {}{}{}",
        target,
        if audit.branch.is_empty() { String::new() } else { format!(" @ {}", audit.branch) },
        if audit.commit.len() >= 7 { format!(" {}", &audit.commit[..7]) } else { String::new() }
    )));
    if audit.step_cap > 0 {
        head.push((Style::Dim, format!("   steps {}/{}", audit.steps_used, audit.step_cap)));
    }
    if crate::audit::run::running(&audit.name) {
        head.push((Style::Warn, "   [*] running".into()));
    }
    if audit.paused {
        head.push((Style::Warn, "   paused".into()));
    }
    out.push(head);
    if let Some(problem) = &audit.problem {
        out.push(vec![(Style::Problem, format!("  [-] {problem}"))]);
    }
    if !audit.scope.is_empty() {
        let assumed = data.claims.iter().filter(|c| c.is_assumption()).count();
        let slice = audit.slice.as_ref().map(|s| s.line(assumed)).unwrap_or_else(|| "mapping…".into());
        out.push(vec![(Style::Status, "  scope   ".into()), (Style::Plain, super::clip(&format!("{} · {slice}", audit.scope.join(", ")), width.saturating_sub(12)))]);
    }
    if !audit.languages.is_empty() {
        out.push(vec![(Style::Status, "  langs   ".into()), (Style::Dim, super::clip(&audit.languages.join(" · "), width.saturating_sub(12)))]);
    }
    out.push(levels_bar(&data.claims, width));
    // A proposal: what the agent asks to start.
    if audit.status == Status::Proposed {
        out.push(Vec::new());
        out.push(vec![(Style::Warn, "  [!] audit proposed".into())]);
        for line in [
            format!("why       {}", audit.why),
            format!("targets   {}", audit.targets.join(", ")),
            format!("scope     {}", if audit.scope.is_empty() { "the whole target".into() } else { audit.scope.join(", ") }),
            format!("estimate  {}", audit.estimate),
            format!("gate      {} · post {}", gate_word(&audit), post_word(&audit)),
        ] {
            out.extend(super::flow(&[(Style::Plain, format!("    {line}"))], width, 14));
        }
    }
    if let Some(why) = &audit.rerun_proposed {
        out.push(vec![(Style::Warn, "  [!] s0ma proposes a re-run".into()), (Style::Dim, format!(": {}", super::clip(why, width.saturating_sub(40))))]);
    }
    if let Some((board, why)) = &audit.attach_proposed {
        out.push(vec![(Style::Warn, format!("  [!] s0ma proposes attaching it to {board}")), (Style::Dim, format!(": {}", super::clip(why, width.saturating_sub(50))))]);
    }
    // The filters, each with its count, and the order.
    out.push(filter_row(state, &audit, &data.claims, width));
    // The tree, the code beside it at full width in one pane, the selected
    // row in full under them.
    let rows = tree(state, &audit, &data.claims);
    select(state, &rows);
    let beside = !super::in_split() && width >= CODE_BESIDE;
    let detail_lines: Vec<(Style, String)> = match (rows.get(state.selected), audit::store()) {
        (Some(row), Some(store)) => audit_tree::detail(row, &audit, &store),
        _ => Vec::new(),
    };
    let tree_width = if beside { (width * 45 / 100).max(64) } else { width };
    let mut strip: Vec<Line> = Vec::new();
    for (n, (style, text)) in detail_lines.iter().enumerate() {
        if let Some(rest) = text.strip_prefix("board    ") {
            strip.push(vec![(Style::Plain, "  board    ".into()), gui_view::hot(Style::Accent, super::clip(rest, width.saturating_sub(13)), Click::Pick(Pick::ClaimBoard))]);
            continue;
        }
        let style = if n == 0 { *style } else { Style::Plain };
        strip.extend(super::flow(&[(style, format!("  {text}"))], width, 11));
    }
    let strip_room = strip.len().min(rows_left.saturating_sub(out.len() + 6) / 2).max(3);
    let list_room = rows_left.saturating_sub(out.len() + strip_room + 2).max(3);
    if state.selected < state.top {
        state.top = state.selected;
    }
    if state.selected >= state.top + list_room {
        state.top = state.selected + 1 - list_room;
    }
    let mut list: Vec<Line> = Vec::new();
    if rows.is_empty() {
        let hint = if !state.search.is_empty() {
            "no matches".to_string()
        } else if state.show != Filter::All {
            format!("nothing {}", state.show.name())
        } else {
            "no claims yet".into()
        };
        list.extend(super::flow(&[(Style::Dim, format!("  {hint}"))], tree_width, 2));
    }
    if audit.gone_at.is_none() && audit.settings.gate == audit::Gate::Plan && !data.claims.is_empty() {
        list.push(vec![(Style::Dim, "  gate plan · waits for go".into())]);
    }
    for (i, row) in rows.iter().enumerate().skip(state.top).take(list_room.saturating_sub(list.len())) {
        let line = audit_tree::row_line(row, &audit, i == state.selected, tree_width);
        list.push(gui_view::pick_row(line, Click::Pick(Pick::AuditRow(i))));
    }
    if rows.len() > state.top + list_room {
        list.push(vec![(Style::Dim, format!("  ↓ {} more", rows.len() - state.top - list_room))]);
    }
    state.code_left = beside.then_some(tree_width + 3);
    if beside {
        let code_width = width.saturating_sub(tree_width + 3);
        let code = preview(state, &audit, rows.get(state.selected), code_width, list_room.max(list.len()));
        for r in 0..list_room.max(list.len()) {
            let mut line = fit(list.get(r).cloned().unwrap_or_default(), tree_width);
            line.push((if state.code_focus() > 0 { Style::Cursor } else { Style::Dim }, " │ ".into()));
            line.extend(code.get(r).cloned().unwrap_or_default());
            out.push(line);
        }
    } else {
        state.preview = None;
        out.extend(list);
    }
    // The strip.
    let left = rows_left.saturating_sub(out.len() + 1);
    out.push(vec![(Style::Dim, format!("  {}", "─".repeat(width.saturating_sub(4))))]);
    state.strip_scroll = state.strip_scroll.min(strip.len().saturating_sub(1));
    out.extend(strip.into_iter().skip(state.strip_scroll).take(left.saturating_sub(1)));
    // The prompt.
    if let Some(input) = &state.input {
        out.truncate(rows_left.saturating_sub(1));
        out.push(prompt_line(input));
    }
    out.truncate(rows_left);
    let left: Line = match &state.scope {
        Some(names) => vec![(Style::Status, "audit".into()), (Style::Dim, format!("  {} of {}", names.iter().position(|n| *n == audit.name).map_or(0, |i| i + 1), names.len()))],
        None => vec![(Style::Status, "audit".into()), (Style::Dim, format!("  {} of {}", position(&data.audits, &audit.name), data.audits.len()))],
    };
    let selected = rows.get(state.selected);
    (out, Footer::new(left, hints(state, &audit, selected)))
}

/// A line cut or padded to `width` columns.
fn fit(line: Line, width: usize) -> Line {
    let mut out: Line = Vec::new();
    let mut used = 0;
    for (style, text) in line {
        if used >= width {
            break;
        }
        let piece = super::clip(&text, width - used);
        used += super::text_width(&piece);
        out.push((style, piece));
    }
    out.push((Style::Plain, " ".repeat(width.saturating_sub(used))));
    out
}

/// The tree as the state shapes it.
fn tree(state: &State, audit: &Audit, claims: &[Claim]) -> Vec<Row> {
    let Some(store) = audit::store() else { return Vec::new() };
    let shape = audit_tree::Shape { filter: state.show, search: &state.search, by_file: state.by_file, open: &state.open, closed: &state.closed };
    audit_tree::rows(&store, audit, claims, &shape)
}

/// The selected row found again by its key (the tree may have moved under
/// it), or kept in range.
fn select(state: &mut State, rows: &[Row]) {
    if let Some(key) = &state.selected_key
        && let Some(i) = rows.iter().position(|r| &r.key == key)
    {
        state.selected = i;
    }
    state.selected = state.selected.min(rows.len().saturating_sub(1));
    state.selected_key = rows.get(state.selected).map(|r| r.key.clone());
}

/// The filters over the tree, each with its count, the selected one lit;
/// the order on the right.
fn filter_row(state: &State, audit: &Audit, claims: &[Claim], width: usize) -> Line {
    let mut bar: Line = vec![(Style::Dim, " ─ ".into())];
    for (i, filter) in FILTERS.iter().enumerate() {
        let count = claims.iter().filter(|c| filter.matches(c, audit)).count();
        if count == 0 && *filter != Filter::All && *filter != state.show {
            continue;
        }
        if bar.len() > 1 {
            bar.push((Style::Dim, " ─ ".into()));
        }
        let look = if *filter == state.show {
            Style::Tab
        } else if *filter == Filter::Waiting {
            Style::Warn
        } else if *filter == Filter::Failing {
            Style::Danger
        } else {
            Style::Plain
        };
        bar.push(gui_view::hot(look, format!(" {} {count} ", filter.name()), Click::Pick(Pick::AuditSection(i))));
    }
    if !state.search.is_empty() {
        bar.push((Style::Dim, format!("   search: {}", state.search)));
    }
    let order = if state.by_file { "by file" } else { "plan order" };
    let used = bar.iter().map(|(_, t)| super::text_width(t)).sum::<usize>();
    bar.push((Style::Dim, format!(" {} ", "─".repeat(width.saturating_sub(used + super::text_width(order) + 6)))));
    bar.push(gui_view::hot(Style::Dim, order.to_string(), Click::Keys(vec![KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE)])));
    bar.push((Style::Dim, " ─".into()));
    bar
}

/// The code beside the tree for the selected row: its claim's code and the
/// check's own files (`claim_view::preview`), read once per row and kept.
fn preview(state: &mut State, audit: &Audit, row: Option<&Row>, width: usize, rows: usize) -> Vec<Line> {
    let Some(row) = row else { return Vec::new() };
    let want = match (&row.node, row.node.claim_id()) {
        (_, Some(claim)) => Some(format!("{}/{claim}/{}", audit.name, row.node.slot().unwrap_or(""))),
        (Node::File { .. } | Node::Unclaimed { .. } | Node::CallOut { .. }, None) => row.node.file().map(|f| format!("{}/file:{f}", audit.name)),
        _ => None,
    };
    let Some(want) = want else {
        state.preview = None;
        return Vec::new();
    };
    if state.preview.as_ref().is_none_or(|(key, _)| *key != want) {
        let store = audit::store();
        let view = match row.node.claim_id().and_then(|id| store.as_ref()?.claim(&audit.name, id).ok().flatten()) {
            Some(claim) => super::claim_view::State::open_check(audit, &claim, row.node.slot()),
            None => super::claim_view::State::open_file(audit, &row.node.file().unwrap_or_default()),
        };
        // The tree has the keys until the code is clicked.
        let mut view = view;
        view.pane = 0;
        state.preview = Some((want, view));
    }
    let Some((_, view)) = state.preview.as_mut() else { return Vec::new() };
    super::claim_view::preview(view, width, rows)
}

/// Whether the audit's project is complete or discarded and it is still
/// attached: the operator decides keep, let go or merge.
fn at_complete(audit: &Audit) -> bool {
    let Some(board) = &audit.board else { return false };
    if matches!(audit.status, Status::Closed | Status::Standing) {
        return false;
    }
    let Some(boards) = audit::boards() else { return false };
    let Ok(Some(found)) = boards.board(board) else { return false };
    let flows = crate::board::flow::FlowStore::beside(&boards);
    matches!(flows.get(&found.id).ok().flatten().and_then(|f| f.stage), Some(crate::board::flow::Stage::Deliverable | crate::board::flow::Stage::Discarded))
}

fn gate_word(audit: &Audit) -> &'static str {
    match audit.settings.gate {
        audit::Gate::Plan => "plan",
        audit::Gate::Reviewed => "reviewed",
        audit::Gate::None => "none",
    }
}

fn post_word(audit: &Audit) -> &'static str {
    match audit.settings.post {
        audit::Post::Auto => "auto",
        audit::Post::Failures => "failures",
        audit::Post::Review => "review",
    }
}

fn position(audits: &[Audit], name: &str) -> usize {
    audits.iter().position(|a| a.name == name).map_or(0, |i| i + 1)
}

fn prompt_line(input: &Input) -> Line {
    let label = match (&input.typing, &input.claim) {
        (Typing::Note, Some(c)) => format!("note on {c} › "),
        (Typing::Edit, Some(c)) => format!("{c}, your wording › "),
        (Typing::Back, Some(c)) => format!("back on {c} › "),
        (Typing::Add, Some(source)) => format!("your claim about {source} › "),
        (Typing::Add, None) => "your claim › ".into(),
        (Typing::Settings, _) => "settings › ".into(),
        (Typing::Decline, _) => "decline, why › ".into(),
        (Typing::Steer, Some(c)) => format!("direction for {c}'s worker › "),
        (Typing::Link, Some(c)) => format!("link {c} to <board> <lead> › "),
        (Typing::Link, None) => "link to <board> › ".into(),
        (Typing::Filter, _) => "search › ".into(),
        (Typing::Merge, _) => "merge into the standing audit › ".into(),
        (Typing::Priority, Some(c)) => format!("{c} priority p"),
        _ => "› ".into(),
    };
    let before: String = input.text[..input.cursor].iter().collect();
    let at: String = input.text.get(input.cursor).map_or(" ".to_string(), |c| c.to_string());
    let after: String = input.text.get(input.cursor + 1..).map(|rest| rest.iter().collect()).unwrap_or_default();
    vec![(Style::Cursor, "  ".into()), (Style::Cursor, label), (Style::Plain, before), (Style::Caret, at), (Style::Plain, after)]
}

fn hints(state: &State, audit: &Audit, row: Option<&Row>) -> Vec<(&'static str, &'static str)> {
    if state.input.is_some() {
        return vec![("⏎", "send"), ("esc", "cancel")];
    }
    if state.code_focus() > 0 {
        let has_check = state.preview.as_ref().is_some_and(|(_, view)| super::claim_view::preview_has_check(view));
        let mut out = vec![("↑↓", "scroll"), ("↑↓ pgup pgdn", "scroll")];
        if has_check {
            out.push(("tab", if state.code_focus() == 1 { "the check" } else { "the code" }));
        }
        out.push(("esc", "the tree"));
        return out;
    }
    let mut out: Vec<(&'static str, &'static str)> = vec![("↑↓", "move"), ("⇧←→", "filter")];
    if audit.status == Status::Proposed {
        out.extend([("a", "start"), ("e", "settings"), ("x", "decline")]);
        return out;
    }
    // At complete: what happens to it (keep, let go, merge).
    if at_complete(audit) {
        out.extend([("k", "keep"), ("v", "let go"), ("M", "merge")]);
    }
    if let Some(row) = row {
        if row.foldable {
            out.push(("⏎", if row.open { "fold" } else { "unfold" }));
        }
        let store = audit::store();
        let claim = row.node.claim_id().and_then(|id| store.as_ref()?.claim(&audit.name, id).ok().flatten());
        match &row.node {
            Node::Claim(_) => {
                let claim = claim.unwrap_or_default();
                out.push(("o", "open"));
                if claim.review == Review::Proposed {
                    out.extend([("y", "accept"), ("n", "reject")]);
                }
                if !claim.readings.is_empty() && claim.picked.is_none() {
                    out.extend([("1", "reading 1"), ("2", "reading 2")]);
                }
                out.push(("e", "reword"));
                if claim.weaker.is_some() || (claim.repro == Repro::ModelOnly && !claim.model_only_accepted) {
                    out.push(("a", if claim.weaker.is_some() { "accept weaker" } else { "accept reason" }));
                }
                if claim.verdict != Verdict::Unchecked {
                    out.push(("b", "send back"));
                }
                if claim.verdict == Verdict::Fails || claim.waived {
                    out.push(("w", if claim.waived { "lift waiver" } else { "waive" }));
                }
                if claim.posted == Posted::Ready {
                    out.extend([("p", "post"), ("x", "discard")]);
                }
                if claim.repro == Repro::Fails && audit.standing() {
                    out.push(("F", "fix project"));
                }
                out.extend([("t", if claim.taken { "give back" } else { "take" }), ("r", "run now"), ("d", "deeper check"), ("m", "note"), ("l", "link")]);
                out.push(("s", if claim.held_by.is_some() { "steer" } else if claim.skipped { "unskip" } else { "skip" }));
                out.push(("!", if claim.pinned { "unpin" } else { "pin" }));
                if claim.posted != Posted::Ready {
                    out.push(("p", "priority"));
                }
                if claim.review == Review::Accepted && claim.posted != Posted::Ready {
                    out.push(("x", "drop"));
                }
            }
            Node::Check { check, .. } => {
                out.extend([("o", "open"), ("r", "run it now"), ("b", "send back"), ("d", "deeper check"), ("m", "note")]);
                if check.held_weaker {
                    out.push(("a", "accept weaker"));
                }
            }
            Node::Writing { worker, .. } => {
                out.push(("o", "open"));
                if worker.is_some() {
                    out.push(("s", "steer"));
                }
            }
            Node::Evidence { .. } => out.push(("o", "open")),
            Node::File { .. } | Node::Unclaimed { .. } | Node::CallOut { .. } => out.extend([("o", "open file"), ("+", "add claim")]),
            _ => {}
        }
    }
    if audit.gone_at.is_none() || audit.settings.gate == audit::Gate::Plan {
        out.push(("g", "go"));
    }
    if row.is_none_or(|r| !matches!(r.node, Node::File { .. } | Node::Unclaimed { .. } | Node::CallOut { .. })) {
        out.push(("+", "add claim"));
    }
    out.extend([("/", "search"), ("f", if state.by_file { "by plan" } else { "by file" }), ("c", "settings"), ("z", if audit.paused { "resume" } else { "pause" })]);
    match &state.scope {
        Some(names) if names.len() > 1 => out.push(("[ ]", "audit")),
        Some(_) => {}
        None => out.push(("tab", "next")),
    }
    out.push(("esc", "back"));
    out
}

/// No audit selected: every audit, one line each, with its levels bar, its
/// `[!]` count and its ceiling; adapters waiting; Enter opens one.
fn portfolio(state: &mut State, data: &Read, width: usize, rows_left: usize) -> (Vec<Line>, Footer) {
    let mut out: Vec<Line> = vec![vec![(Style::Status, "  audits".into())]];
    let store = audit::store();
    let adapters = crate::audit::adapters::proposed();
    let total = data.audits.len() + adapters.len();
    state.portfolio = state.portfolio.min(total.saturating_sub(1));
    if total == 0 {
        out.push(vec![(Style::Dim, "  no audits · /audit start <name> <path>".into())]);
    }
    let mut group = String::new();
    for (i, audit) in data.audits.iter().enumerate() {
        let owner = if !audit.shown() { "archive".to_string() } else { audit.board.clone().unwrap_or_else(|| "standing".into()) };
        if owner != group {
            out.push(vec![(Style::Status, format!("  {owner}"))]);
            group = owner;
        }
        let claims = store.as_ref().map(|s| s.claims(&audit.name).unwrap_or_default()).unwrap_or_default();
        let waiting = claims.iter().filter(|c| c.waits_on(audit)).count() + usize::from(audit.status == Status::Proposed || audit.rerun_proposed.is_some());
        let (mark, status) = audit.status_mark();
        let (mark, status) = if audit.archived { ("[~]", format!("archived · {status}")) } else { (mark, status.to_string()) };
        let here = i == state.portfolio;
        let mut line: Line = vec![
            (if here { Style::Cursor } else { Style::Plain }, if here { "❯ ".into() } else { "  ".into() }),
            (if here { Style::Bold } else { Style::Plain }, format!("{:<18}", audit.name)),
            (if mark == "[!]" { Style::Warn } else { Style::Dim }, format!("{mark} {:<28}", super::clip(&status, 26))),
        ];
        let mut bar = levels_bar(&claims, width.min(90));
        bar.remove(0);
        line.extend(bar.into_iter().take(4));
        if waiting > 0 {
            line.push((Style::Warn, format!("  [!] {waiting}")));
        }
        line.push((Style::Dim, format!("  ceiling {}", crate::audit::adapters::ceiling_for(&audit.languages).label())));
        out.push(gui_view::pick_row(line, Click::Pick(Pick::AuditRow(i))));
    }
    for (j, lang) in adapters.iter().enumerate() {
        let i = data.audits.len() + j;
        let here = i == state.portfolio;
        let line: Line = vec![
            (if here { Style::Cursor } else { Style::Plain }, if here { "❯ ".into() } else { "  ".into() }),
            (Style::Warn, format!("[!] adapter {lang}")),
            (Style::Dim, " · proposed".into()),
        ];
        out.push(gui_view::pick_row(line, Click::Pick(Pick::AuditRow(i))));
    }
    // What the archive holds, when it is not shown.
    let hidden = store.as_ref().map(|s| s.audits().unwrap_or_default().iter().filter(|a| !a.shown()).count()).unwrap_or(0);
    if !state.show_archive && hidden > 0 {
        out.push(Vec::new());
        out.push(vec![gui_view::hot(Style::Dim, format!("  ▸ archive  {hidden}"), Click::Keys(vec![KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE)]))]);
    }
    if let Some(input) = &state.input {
        out.push(prompt_line(input));
    }
    out.truncate(rows_left);
    let chosen = data.audits.get(state.portfolio);
    let hints = if state.input.is_some() {
        vec![("⏎", "send"), ("esc", "cancel")]
    } else {
        let mut hints = vec![("↑↓", "move"), ("⏎", "open"), ("a", "start"), ("r", "re-run")];
        if let Some(audit) = chosen {
            hints.push(("A", if audit.archived { "unarchive" } else { "archive" }));
            hints.push(("del", "delete"));
        }
        if hidden > 0 || state.show_archive {
            hints.push(("h", if state.show_archive { "hide archive" } else { "show archive" }));
        }
        hints.push(("esc", "back"));
        hints
    };
    (out, Footer::new(vec![(Style::Status, "audit".into()), (Style::Dim, format!("  {} audit{}", data.audits.len(), if data.audits.len() == 1 { "" } else { "s" }))], hints))
}

// ------------------------------------------------------------------ keys

pub(super) fn note(tab: &mut Open, result: std::result::Result<String, String>) {
    match result {
        Ok(text) => tab.tell(plain(&text)),
        Err(error) => tab.note(plain(&error), true),
    }
}

/// A reply as the operator reads it: a tool's JSON by its note.
fn plain(text: &str) -> String {
    let text = text.trim_start_matches("[audit] ");
    if text.starts_with('{')
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(text)
    {
        if let Some(note) = value.get("note").and_then(|v| v.as_str()) {
            return note.to_string();
        }
        if let Some(error) = value.get("error").and_then(|v| v.as_str()) {
            return error.to_string();
        }
    }
    text.to_string()
}

/// `key` pressed on `id` once arms it; again does it.
fn twice(tab: &mut Open, key: char, id: &str, say: String) -> bool {
    if tab.audit.armed.as_ref().is_some_and(|(was, on)| *was == key && on == id) {
        tab.audit.armed = None;
        return true;
    }
    tab.ask(say);
    tab.audit.armed = Some((key, id.to_string()));
    false
}

/// A value typed into the settings window's row `key`.
pub(super) fn type_setting(tab: &mut Open, key: &str, now: &str) {
    start_input(tab, Typing::Settings, Some(key.to_string()), now);
}

fn start_input(tab: &mut Open, typing: Typing, claim: Option<String>, text: &str) {
    let text: Vec<char> = text.chars().collect();
    tab.audit.input = Some(Input { typing, claim, cursor: text.len(), text });
}

pub(super) fn paste(tab: &mut Open, text: &str) {
    if let Some(input) = tab.audit.input.as_mut() {
        for c in text.chars().filter(|c| !c.is_control()) {
            input.text.insert(input.cursor, c);
            input.cursor += 1;
        }
    }
}

/// A click on a row.
pub(super) fn pick(tab: &mut Open, index: usize) {
    tab.audit.armed = None;
    if tab.audit.settings.is_some() && tab.audit.audit.is_some() {
        super::audit_settings::pick(tab, index);
        return;
    }
    if tab.audit.audit.is_none() {
        tab.audit.portfolio = index;
    } else {
        tab.audit.selected = index;
        tab.audit.selected_key = None;
        tab.audit.strip_scroll = 0;
    }
}

/// The board reference in the claim view or the strip: the board's tab at
/// the finding (or the lead the claim's results land under).
pub(super) fn to_board(tab: &mut Open) {
    let Some(store) = audit::store() else { return };
    let (audit, id) = match &tab.audit.view {
        Some(view) => (view.audit.clone(), view.claim.clone()),
        None => (tab.audit.audit.clone().unwrap_or_default(), None),
    };
    let claim = id.and_then(|id| store.claim(&audit, &id).ok().flatten());
    let target = claim.as_ref().and_then(|c| {
        c.fact.as_ref().and_then(|f| f.split_once('/').map(|(b, n)| (b.to_string(), Some(n.to_string())))).or_else(|| c.lead.clone().map(|(b, l)| (b, Some(l))))
    });
    let target = target.or_else(|| store.audit(&audit).ok().flatten().and_then(|a| a.board.clone().map(|b| (b, a.lead.clone()))));
    if let Some((board, node)) = target {
        super::show_board(tab, &board, node.as_deref());
    }
}

/// A click on a filter over the tree.
pub(super) fn pick_section(tab: &mut Open, index: usize) {
    if let Some(filter) = FILTERS.get(index) {
        tab.audit.show = *filter;
        tab.audit.selected = 0;
        tab.audit.selected_key = None;
        tab.audit.top = 0;
    }
}

/// The audit tab opened at an audit (a board's audit lead, the projects tab).
pub(super) fn open_at(tab: &mut Open, audit: &str, claim: Option<&str>) {
    tab.audit.audit = Some(audit.to_string());
    tab.audit.show = Filter::All;
    tab.audit.selected = 0;
    tab.audit.selected_key = claim.map(|c| format!("c:{c}"));
}

/// A click on the code beside the tree: the keys and the wheel go there
/// (`part` 0 the code, at its line `at`; 1 the check).
pub(super) fn pick_preview(tab: &mut Open, part: usize, at: usize) {
    tab.audit.armed = None;
    let Some((_, view)) = tab.audit.preview.as_mut() else { return };
    if part == 1 && super::claim_view::preview_has_check(view) {
        view.pane = 2;
    } else {
        view.pane = 1;
        if part == 0 && at != usize::MAX {
            view.cursor = at;
        }
    }
}

/// The wheel: over the code beside the tree (or with the code holding the
/// keys) it scrolls the code; elsewhere it walks the tree.
pub(super) fn wheel(tab: &mut Open, up: bool, column: usize) {
    let over_code = tab.audit.code_left.is_some_and(|left| column >= left);
    if let Some((_, view)) = tab.audit.preview.as_mut()
        && (over_code || view.pane > 0)
    {
        super::claim_view::preview_scroll(view, if up { -3 } else { 3 });
        return;
    }
    if tab.audit.audit.is_none() {
        tab.audit.portfolio = if up { tab.audit.portfolio.saturating_sub(1) } else { tab.audit.portfolio + 1 };
    } else {
        tab.audit.selected = if up { tab.audit.selected.saturating_sub(1) } else { tab.audit.selected + 1 };
        tab.audit.selected_key = None;
    }
}

fn typing(tab: &mut Open, key: &KeyEvent) {
    let Some(input) = tab.audit.input.as_mut() else { return };
    match key.code {
        KeyCode::Esc => tab.audit.input = None,
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            input.text.insert(input.cursor, c);
            input.cursor += 1;
        }
        KeyCode::Backspace if input.cursor > 0 => {
            input.cursor -= 1;
            input.text.remove(input.cursor);
        }
        KeyCode::Delete if input.cursor < input.text.len() => {
            input.text.remove(input.cursor);
        }
        KeyCode::Left => input.cursor = input.cursor.saturating_sub(1),
        KeyCode::Right => input.cursor = (input.cursor + 1).min(input.text.len()),
        KeyCode::Home => input.cursor = 0,
        KeyCode::End => input.cursor = input.text.len(),
        KeyCode::Enter => {
            let input = tab.audit.input.take().expect("typing");
            submit(tab, input);
        }
        _ => {}
    }
}

fn submit(tab: &mut Open, input: Input) {
    let text: String = input.text.iter().collect::<String>().trim().to_string();
    let Some(store) = audit::store() else { return };
    let Some(name) = tab.audit.audit.clone() else {
        // The portfolio: a decline's reason.
        if input.typing == Typing::Decline
            && let Some(audit) = input.claim
        {
            note(tab, crate::audit::ops::decline(&store, &audit, &text));
        }
        return;
    };
    let claim = input.claim.clone().unwrap_or_default();
    let result = match input.typing {
        Typing::Filter => {
            tab.audit.search = text;
            tab.audit.selected = 0;
            tab.audit.selected_key = None;
            return;
        }
        _ if text.is_empty() && input.typing != Typing::Decline => return,
        // In review, a note on a claim is the project's note (`on c9`), which
        // the agent answers with board_answer.
        Typing::Note if tab.view == super::View::Review => {
            let board = store.audit(&name).ok().flatten().and_then(|a| a.board);
            let id = board.as_deref().and_then(|b| audit::boards()?.board(b).ok().flatten()).map(|b| b.id);
            match id {
                Some(id) => super::act(super::Action::Project { board: id, done: super::projects_view::Move::Note { text: format!("on {name}/{claim}: {text}") } }),
                None => Err("this audit's project is not on a board".into()),
            }
        }
        Typing::Note => {
            // A note about a claim reaches the agent as a board hint does.
            let audit = store.audit(&name).ok().flatten();
            let to_board = audit.as_ref().and_then(|a| a.board.clone());
            let _ = store.log(&name, Who::Operator, &format!("note on {claim}"), &text);
            crate::audit::ops::needs_s0ma(format!("· {name}/{claim}: a note from the operator: {text}"));
            if let (Some(board), Some(boards)) = (to_board, audit::boards())
                && let Ok(Some(found)) = boards.board(&board)
            {
                let _ = boards.add_hint(&found.id, &format!("on {name}/{claim}: {text}"));
            }
            Ok(format!("note on {claim} sent to s0ma"))
        }
        Typing::Edit => crate::audit::ops::operator_claim(&store, &name, &claim, "edit", &text),
        Typing::Back => crate::audit::ops::operator_claim(&store, &name, &claim, "back", &text),
        Typing::Add => crate::audit::ops::operator_claim_new(&store, &name, &text, input.claim.as_deref().unwrap_or("")),
        Typing::Settings => match &input.claim {
            // A row of the settings window.
            Some(key) => crate::audit::ops::set(&store, &name, &super::audit_settings::pair(key, &text)),
            None => crate::audit::ops::set(&store, &name, &text),
        },
        Typing::Decline => crate::audit::ops::decline(&store, &name, &text),
        Typing::Steer => {
            let held = store.claim(&name, &claim).ok().flatten().and_then(|c| c.held_by);
            let sent = held.as_deref().and_then(|label| {
                crate::subagent::running().into_iter().find(|job| crate::vm::netboxes::agent_label(&job.id) == label).map(|job| crate::subagent::steer_from_operator(&job.id, &text, false))
            });
            match sent {
                Some(result) => {
                    let _ = store.log(&name, Who::Operator, &format!("steered {claim}'s worker"), &text);
                    result
                }
                None => crate::audit::ops::operator_claim(&store, &name, &claim, "steer", &text),
            }
        }
        Typing::Link => {
            let words: Vec<&str> = text.split_whitespace().collect();
            match (input.claim.as_deref(), words.as_slice()) {
                (Some(c), [board, lead]) => crate::audit::ops::link(&store, &name, board, Some(c), Some(lead), Who::Operator, ""),
                (_, [board]) => crate::audit::ops::link(&store, &name, board, None, None, Who::Operator, ""),
                _ => Err("<board> <lead> for a claim, or <board> for the audit".into()),
            }
        }
        Typing::Priority => crate::audit::ops::operator_claim(&store, &name, &claim, "priority", &text),
        Typing::Merge => crate::audit::run::after_project(&store, &name, "merge", Some(&text)),
    };
    note(tab, result);
}

/// Esc here would leave the audit (nothing open over it).
pub(super) fn at_top(tab: &Open) -> bool {
    tab.audit.view.is_none() && tab.audit.input.is_none() && tab.audit.settings.is_none() && tab.audit.search.is_empty() && tab.audit.code_focus() == 0
}

pub(super) fn key(tab: &mut Open, key: &KeyEvent) -> Outcome {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    match key.code {
        KeyCode::Char('c' | 'C') if ctrl => return Outcome::Interrupt,
        KeyCode::Char('d' | 'D') if ctrl => return Outcome::Eof,
        _ => {}
    }
    if tab.audit.view.is_some() {
        if super::claim_view::key(tab, key) {
            tab.audit.view = None;
        }
        return Outcome::Stay;
    }
    if tab.audit.input.is_some() {
        typing(tab, key);
        return Outcome::Stay;
    }
    let data = read(&tab.audit);
    if tab.audit.settings.is_some() {
        match data.audit.clone() {
            Some(audit) => return super::audit_settings::key(tab, &audit, key),
            None => tab.audit.settings = None,
        }
    }
    let Some(store) = audit::store() else { return Outcome::Closed };
    // Stepping through audits.
    let step = |tab: &mut Open, by: isize| {
        let scoped = tab.audit.scope.clone();
        let names: Vec<String> = scoped.clone().unwrap_or_else(|| data.audits.iter().map(|a| a.name.clone()).collect());
        if names.is_empty() {
            return;
        }
        let at = tab.audit.audit.as_ref().and_then(|n| names.iter().position(|x| x == n)).map_or(-1, |i| i as isize);
        // Standalone, one past the last is every audit (the portfolio).
        let wrap = if scoped.is_some() { names.len() } else { names.len() + 1 } as isize;
        let next = (at + by).rem_euclid(wrap);
        tab.audit.audit = names.get(next as usize).cloned();
        tab.audit.selected = 0;
        tab.audit.selected_key = None;
        tab.audit.top = 0;
        tab.audit.armed = None;
    };
    match key.code {
        KeyCode::Char(']') if tab.audit.scope.is_some() => {
            step(tab, 1);
            return Outcome::Stay;
        }
        KeyCode::Char('[') if tab.audit.scope.is_some() => {
            step(tab, -1);
            return Outcome::Stay;
        }
        // With the code beside the tree holding the keys, Tab goes between
        // its code and its check instead.
        KeyCode::Tab if tab.audit.code_focus() == 0 => {
            step(tab, 1);
            return Outcome::Stay;
        }
        KeyCode::BackTab if tab.audit.code_focus() == 0 => {
            step(tab, -1);
            return Outcome::Stay;
        }
        KeyCode::Char('n') if ctrl => {
            step(tab, 1);
            return Outcome::Stay;
        }
        KeyCode::Char('p') if ctrl => {
            step(tab, -1);
            return Outcome::Stay;
        }
        _ => {}
    }
    // The portfolio.
    let Some(audit) = data.audit.clone() else {
        let adapters = crate::audit::adapters::proposed();
        let at = tab.audit.portfolio;
        let chosen = data.audits.get(at).cloned();
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return Outcome::Closed,
            KeyCode::Up => tab.audit.portfolio = at.saturating_sub(1),
            KeyCode::Down => tab.audit.portfolio = at + 1,
            KeyCode::Enter => {
                if let Some(a) = chosen {
                    tab.audit.audit = Some(a.name);
                    tab.audit.show = Filter::All;
                    tab.audit.selected = 0;
                    tab.audit.selected_key = None;
                }
            }
            KeyCode::Char('a') => {
                if let Some(a) = &chosen
                    && a.status == Status::Proposed
                {
                    if twice(tab, 'a', &a.name, format!("a again starts {}", a.name)) {
                        note(tab, crate::audit::ops::start(&store, &a.name, Who::Operator));
                    }
                } else if let Some(lang) = adapters.get(at.saturating_sub(data.audits.len()))
                    && at >= data.audits.len()
                    && twice(tab, 'a', lang, format!("a again approves {lang}"))
                {
                    note(tab, crate::audit::adapters::approve(lang));
                }
            }
            KeyCode::Char('x') => {
                if let Some(a) = &chosen
                    && a.status == Status::Proposed
                    && twice(tab, 'x', &a.name, format!("x again declines {}", a.name))
                {
                    start_input(tab, Typing::Decline, Some(a.name.clone()), "");
                }
            }
            KeyCode::Char('r') => {
                if let Some(a) = &chosen {
                    note(tab, crate::audit::run::rerun(&store, &a.name, Who::Operator));
                }
            }
            // Archive, or back from it: reversible, so once.
            KeyCode::Char('A') => {
                if let Some(a) = &chosen {
                    note(tab, crate::audit::ops::archive(&store, &a.name, !a.archived));
                }
            }
            KeyCode::Char('h') => {
                tab.audit.show_archive = !tab.audit.show_archive;
            }
            // Delete: for good, so twice.
            KeyCode::Delete | KeyCode::Char('D') => {
                if let Some(a) = &chosen {
                    let what = if a.work.is_empty() { String::new() } else if a.branch.is_empty() { " and work tree".to_string() } else { format!(", work tree and branch {}", a.branch) };
                    if twice(tab, 'D', &a.name, format!("del again deletes {}{what}", a.name)) {
                        note(tab, crate::audit::ops::delete(&store, &a.name));
                    }
                }
            }
            _ => {}
        }
        return Outcome::Stay;
    };
    let rows = tree(&tab.audit, &audit, &data.claims);
    select(&mut tab.audit, &rows);
    let at = tab.audit.selected;
    let row = rows.get(at);
    let node = row.map(|r| r.node.clone());
    let claim: Option<Claim> = row.and_then(|r| r.node.claim_id()).and_then(|id| store.claim(&audit.name, id).ok().flatten());
    let id = claim.as_ref().map(|c| c.id.clone()).unwrap_or_default();
    let slot = row.and_then(|r| r.node.slot()).map(str::to_string);
    let on_claim = matches!(node, Some(Node::Claim(_)));
    let name = audit.name.clone();
    let act = |tab: &mut Open, action: &str, text: &str| {
        let result = crate::audit::ops::operator_claim(&store, &name, &id, action, text);
        note(tab, result);
    };
    // A proposal: start, change, decline.
    if audit.status == Status::Proposed {
        match key.code {
            KeyCode::Char('a') if twice(tab, 'a', &name, format!("a again starts {name}")) => note(tab, crate::audit::ops::start(&store, &name, Who::Operator)),
            KeyCode::Char('e' | 'c') => tab.audit.settings = Some(0),
            KeyCode::Char('x') if twice(tab, 'x', &name, format!("x again declines {name}")) => start_input(tab, Typing::Decline, None, ""),
            KeyCode::Esc if tab.audit.scope.is_some() => return Outcome::Closed,
            KeyCode::Esc => tab.audit.audit = None,
            KeyCode::Char('q') => return Outcome::Closed,
            _ => {}
        }
        return Outcome::Stay;
    }
    // The code beside the tree has the keys: they scroll it, Tab goes
    // between the code and its check, Esc (or ←) gives them back.
    if tab.audit.code_focus() > 0
        && let Some((_, view)) = tab.audit.preview.as_mut()
    {
        let page = 10;
        let handled = match key.code {
            KeyCode::Up => Some(-1),
            KeyCode::Down => Some(1),
            KeyCode::PageUp => Some(-page),
            KeyCode::PageDown => Some(page),
            KeyCode::Home => Some(-1_000_000),
            KeyCode::End => Some(1_000_000),
            KeyCode::Tab | KeyCode::BackTab => {
                if super::claim_view::preview_has_check(view) {
                    view.pane = if view.pane == 1 { 2 } else { 1 };
                }
                Some(0)
            }
            KeyCode::Esc | KeyCode::Left => {
                view.pane = 0;
                Some(0)
            }
            // ⏎ (or a double-click): the claim view, at the same line.
            // It is the claim view's own state, read already: handed over.
            KeyCode::Enter => {
                if let Some((_, mut full)) = tab.audit.preview.take() {
                    full.pane = 1;
                    tab.audit.view = Some(full);
                }
                return Outcome::Stay;
            }
            _ => None,
        };
        if let Some(by) = handled {
            if by != 0 {
                super::claim_view::preview_move(view, by);
            }
            return Outcome::Stay;
        }
    }
    let fold = |tab: &mut Open, open: bool| {
        if let Some(row) = rows.get(at).filter(|r| r.foldable) {
            if open {
                tab.audit.closed.remove(&row.key);
                tab.audit.open.insert(row.key.clone());
            } else {
                tab.audit.open.remove(&row.key);
                tab.audit.closed.insert(row.key.clone());
            }
        }
    };
    // The claim view at the row: a claim, one of its checks, a file.
    let open_view = |tab: &mut Open| match (&node, &claim) {
        (Some(Node::File { .. } | Node::Unclaimed { .. } | Node::CallOut { .. }), _) => {
            if let Some(file) = node.as_ref().and_then(Node::file) {
                tab.audit.view = Some(super::claim_view::State::open_file(&audit, &file));
            }
        }
        (_, Some(c)) => tab.audit.view = Some(super::claim_view::State::open_check(&audit, c, slot.as_deref())),
        _ => {}
    };
    let move_to = |tab: &mut Open, to: usize| {
        tab.audit.selected = to.min(rows.len().saturating_sub(1));
        tab.audit.selected_key = rows.get(tab.audit.selected).map(|r| r.key.clone());
        tab.audit.strip_scroll = 0;
        tab.audit.armed = None;
    };
    match key.code {
        KeyCode::Esc => {
            if !tab.audit.search.is_empty() {
                tab.audit.search.clear();
            } else if tab.audit.scope.is_some() {
                return Outcome::Closed;
            } else {
                tab.audit.audit = None;
            }
        }
        KeyCode::Char('q') => return Outcome::Closed,
        KeyCode::Left if shift => {
            let i = FILTERS.iter().position(|f| *f == tab.audit.show).unwrap_or(0);
            pick_section(tab, (i + FILTERS.len() - 1) % FILTERS.len());
        }
        KeyCode::Right if shift => {
            let i = FILTERS.iter().position(|f| *f == tab.audit.show).unwrap_or(0);
            pick_section(tab, (i + 1) % FILTERS.len());
        }
        // ←→ fold and unfold (← on a folded row goes to its parent).
        KeyCode::Left => match rows.get(at) {
            Some(r) if r.foldable && r.open => fold(tab, false),
            Some(r) if !r.guides.is_empty() => {
                let depth = r.guides.chars().count();
                if let Some(parent) = (0..at).rev().find(|&i| rows[i].guides.chars().count() < depth) {
                    move_to(tab, parent);
                }
            }
            _ => {}
        },
        KeyCode::Right => fold(tab, true),
        // In plan order, Shift+↑↓ moves a claim (its priority past its neighbour's).
        KeyCode::Up if shift && on_claim && !tab.audit.by_file => {
            let above = (0..at).rev().find_map(|i| match &rows[i].node {
                Node::Claim(c) => Some(c.priority),
                _ => None,
            });
            if let Some(p) = above {
                let _ = crate::audit::ops::operator_claim(&store, &name, &id, "priority", &(p + 1).min(10).to_string());
            }
        }
        KeyCode::Down if shift && on_claim && !tab.audit.by_file => {
            let below = (at + 1..rows.len()).find_map(|i| match &rows[i].node {
                Node::Claim(c) => Some(c.priority),
                _ => None,
            });
            if let Some(p) = below {
                let _ = crate::audit::ops::operator_claim(&store, &name, &id, "priority", &(p - 1).max(1).to_string());
            }
        }
        KeyCode::Up if shift => tab.audit.strip_scroll = tab.audit.strip_scroll.saturating_sub(1),
        KeyCode::Down if shift => tab.audit.strip_scroll += 1,
        KeyCode::Up => move_to(tab, at.saturating_sub(1)),
        KeyCode::Down => move_to(tab, at + 1),
        KeyCode::PageUp => move_to(tab, at.saturating_sub(10)),
        KeyCode::PageDown => move_to(tab, at + 10),
        KeyCode::Home => move_to(tab, 0),
        KeyCode::End => move_to(tab, rows.len().saturating_sub(1)),
        KeyCode::Enter => match rows.get(at) {
            Some(r) if r.foldable => fold(tab, !r.open),
            Some(_) => open_view(tab),
            None => {}
        },
        KeyCode::Char('o') => open_view(tab),
        KeyCode::Char('k') if at_complete(&audit) => {
            if twice(tab, 'k', &format!("keep {name}"), format!("k again keeps {name}")) {
                note(tab, crate::audit::run::after_project(&store, &name, "keep", None));
            }
        }
        KeyCode::Char('v') if at_complete(&audit) => {
            if twice(tab, 'v', &format!("letgo {name}"), format!("v again lets {name} go")) {
                note(tab, crate::audit::run::after_project(&store, &name, "letgo", None));
            }
        }
        KeyCode::Char('M') if at_complete(&audit) => start_input(tab, Typing::Merge, None, ""),
        KeyCode::Char('g') => note(tab, crate::audit::ops::go(&store, &name)),
        KeyCode::Char('z') => note(tab, crate::audit::ops::pause(&store, &name)),
        KeyCode::Char('c') => tab.audit.settings = Some(0),
        KeyCode::Char('f') => {
            tab.audit.by_file = !tab.audit.by_file;
            tab.audit.top = 0;
        }
        KeyCode::Char('+') => match &node {
            Some(Node::Unclaimed { file, symbol }) => start_input(tab, Typing::Add, Some(format!("{file} ({symbol})")), &format!("{symbol} ")),
            Some(Node::CallOut { symbol, .. }) => start_input(tab, Typing::Add, Some(format!("sym:{symbol}")), &format!("assumes: {symbol} ")),
            Some(Node::File { path, .. }) => start_input(tab, Typing::Add, Some(path.clone()), ""),
            _ => start_input(tab, Typing::Add, None, ""),
        },
        KeyCode::Char('/') => start_input(tab, Typing::Filter, None, &tab.audit.search.clone()),
        KeyCode::Char('r') if claim.is_none() => note(tab, crate::audit::run::rerun(&store, &name, Who::Operator)),
        KeyCode::Char('r') => note(tab, crate::audit::run::run_check_now(&store, &name, Some(&id), if on_claim { None } else { slot.as_deref() })),
        KeyCode::Char('l') => start_input(tab, Typing::Link, claim.as_ref().map(|c| c.id.clone()), ""),
        KeyCode::Char('a') if audit.attach_proposed.is_some() && claim.as_ref().is_none_or(|c| c.weaker.is_none() && c.repro != Repro::ModelOnly) => {
            let (board, why) = audit.attach_proposed.clone().unwrap_or_default();
            if twice(tab, 'a', &format!("attach {name}"), format!("a again attaches {name} to {board}")) {
                note(tab, crate::audit::ops::attach(&store, &name, &board, Who::Operator, &why));
            }
        }
        _ if claim.is_none() => {}
        // Steering the worker writing a check.
        KeyCode::Char('s') if matches!(node, Some(Node::Writing { worker: Some(_), .. })) || (on_claim && claim.as_ref().is_some_and(|c| c.held_by.is_some())) => {
            start_input(tab, Typing::Steer, Some(id.clone()), "")
        }
        KeyCode::Char('b') if !on_claim => {
            // One check sent back: `c3/k2`.
            let target = format!("{id}/{}", slot.clone().unwrap_or_else(|| "k1".into()));
            start_input(tab, Typing::Back, Some(target), "")
        }
        KeyCode::Char('d') => act(tab, "depth", ""),
        KeyCode::Char('m') => start_input(tab, Typing::Note, Some(id.clone()), ""),
        KeyCode::Char('a') if matches!(&node, Some(Node::Check { check, .. }) if check.held_weaker) => {
            if twice(tab, 'a', &id, format!("a again accepts {id}'s weaker check")) {
                note(tab, crate::audit::run::accept_weaker(&store, &name, &id));
            }
        }
        _ if !on_claim => {}
        KeyCode::Char('y') => act(tab, "accept", ""),
        KeyCode::Char('n') => act(tab, "reject", ""),
        KeyCode::Char(digit @ '1'..='4') => act(tab, "pick", &digit.to_string()),
        KeyCode::Char('e') => start_input(tab, Typing::Edit, Some(id.clone()), &claim.as_ref().map(|c| c.text.clone()).unwrap_or_default()),
        KeyCode::Char('b') => start_input(tab, Typing::Back, Some(id.clone()), ""),
        KeyCode::Char('x') if claim.as_ref().is_some_and(|c| c.posted == Posted::Ready) => {
            if twice(tab, 'x', &id, format!("x again discards {id}'s result")) {
                act(tab, "discard", "");
            }
        }
        KeyCode::Char('x') if twice(tab, 'x', &id, format!("x again drops {id}")) => act(tab, "drop", ""),
        KeyCode::Char('a') if twice(tab, 'a', &id, format!("a again accepts what {id} waits on")) => {
            if claim.as_ref().is_some_and(|c| c.weaker.is_some()) {
                note(tab, crate::audit::run::accept_weaker(&store, &name, &id));
            } else {
                act(tab, "accept-weaker", "");
            }
        }
        KeyCode::Char('t') => act(tab, "take", ""),
        KeyCode::Char('w') => act(tab, "waive", ""),
        KeyCode::Char('p') if claim.as_ref().is_some_and(|c| c.posted == Posted::Ready) => act(tab, "post", ""),
        KeyCode::Char('p') => start_input(tab, Typing::Priority, Some(id.clone()), ""),
        KeyCode::Char('!') => act(tab, "pin", ""),
        KeyCode::Char('s') => act(tab, "skip", ""),
        KeyCode::Char('F') => note(tab, crate::audit::ops::fix_project(&store, &name, &id)),
        _ => {}
    }
    Outcome::Stay
}

