//! The board's tab as a tree to walk (a submodule of `tabs.rs`).
//!
//! A board is already a tree, written as it was searched: a lead (an Intent,
//! a direction worth trying) is explored into findings (Facts, each naming
//! the lead it came from), and a finding opens new leads (each naming the
//! findings it came from). So the tab draws goal → leads → what each found →
//! the leads that opened, one row each, folded and unfolded with the arrow
//! keys, with the selected row's full text in a strip at the bottom.
//!
//! - A node with more than one parent sits under the first, and says where
//!   else it came from ("also f2").
//! - Leads being explored, open and explored start unfolded; ruled-out ones
//!   start folded, so they don't pile up. What is folded by hand stays so
//!   across the board's re-reads (every second), by id.
//! - The selection is kept by id too, so rows arriving or moving don't move it.
//! - Hints (the operator's notes, which stand on the board until he takes one off)
//!   stand under the lead or finding they are about, after its own children;
//!   one about nothing in particular stands under the board's row.
//! - The first row is the board itself (how it started): selected, the strip
//!   has the board in full, title, origin, goal and test, which the head can
//!   only show cut.
//! - Every row is one row, cut to the width; the strip has the selected
//!   node's text in full, and scrolls (Shift+↑↓, Shift+PgUp/PgDn) when it
//!   runs longer than it has rows.

use super::{clip, flow, short_id, tab_of, text_width};
use crate::board::{Basis, Board, BoardStatus, Fact, FactAuthor, Hint, Intent, IntentState, Polarity};
use crate::style::{Line, Style};

use chrono::{DateTime, Local};
use std::collections::{HashMap, HashSet};

/// A piece of a row, in its style.
type Segment = (Style, String);

/// The board as last read.
#[derive(Clone)]
pub struct Snapshot {
    pub board: Board,
    pub intents: Vec<Intent>,
    pub facts: Vec<Fact>,
    pub hints: Vec<Hint>,
    /// Where it is in the workflow, and the operator's review notes, when
    /// read for the open subtab's other sections.
    pub flow: Option<crate::board::flow::Flow>,
    pub notes: Vec<crate::board::flow::Note>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The board itself: its first row.
    Board,
    /// An intent, by index in `Snapshot::intents`.
    Lead(usize),
    /// A fact, by index in `Snapshot::facts`.
    Finding(usize),
    /// A hint, by index in `Snapshot::hints`: under the lead or finding it
    /// is about, or else under the board's row.
    Hint(usize),
}

pub struct Node {
    pub id: String,
    pub kind: Kind,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Other parents it came from, by id (short).
    pub also: Vec<String>,
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub roots: Vec<usize>,
    /// Greyed out by the operator, by node: set aside, for reading only
    /// (`sink`). The agent never sees it.
    pub grey: Vec<bool>,
}

/// One visible row: a node, how deep, and for each ancestor level whether a
/// line runs on below it (a later sibling), for the guides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub node: usize,
    pub guides: Vec<bool>,
    pub last: bool,
}

/// What is folded, by node id; and which ids have been seen, so a node's
/// default is applied once and a fold by hand is kept.
#[derive(Default, Debug)]
pub struct Folds {
    open: HashSet<String>,
    seen: HashSet<String>,
}

/// The board's own row (never an intent, fact or hint id).
pub const BOARD_ID: &str = "board";

/// `intent-0007` → `i7`, `fact-0003` → `f3`, `hint-0002` → `h2`.
pub fn short(id: &str) -> String {
    let Some((prefix, number)) = id.split_once('-') else {
        return id.to_string();
    };
    let number = number.trim_start_matches('0');
    let number = if number.is_empty() { "0" } else { number };
    match prefix {
        "intent" => format!("i{number}"),
        "fact" => format!("f{number}"),
        "hint" => format!("h{number}"),
        _ => id.to_string(),
    }
}

/// How a lead is drawn and ordered: exploring, open, explored, ruled out.
fn lead_rank(intent: &Intent, now: DateTime<Local>) -> u8 {
    match intent.state {
        IntentState::Claimed if !intent.claim_expired(now) => 0,
        IntentState::Open | IntentState::Claimed => 1,
        IntentState::Explored => 2,
        IntentState::Dead => 3,
    }
}

impl Tree {
    pub fn build(snapshot: &Snapshot, now: DateTime<Local>) -> Tree {
        let mut nodes: Vec<Node> = Vec::new();
        let mut index: HashMap<&str, usize> = HashMap::new();
        for (i, intent) in snapshot.intents.iter().enumerate() {
            index.insert(&intent.id, nodes.len());
            nodes.push(Node { id: intent.id.clone(), kind: Kind::Lead(i), parent: None, children: Vec::new(), also: Vec::new() });
        }
        for (i, fact) in snapshot.facts.iter().enumerate() {
            index.insert(&fact.id, nodes.len());
            nodes.push(Node { id: fact.id.clone(), kind: Kind::Finding(i), parent: None, children: Vec::new(), also: Vec::new() });
        }
        // Parents: a lead's first finding it came from; a finding's lead, or
        // else its first finding.
        let mut wanted: Vec<(usize, Vec<String>)> = Vec::new();
        for (i, intent) in snapshot.intents.iter().enumerate() {
            wanted.push((i, intent.from_fact_ids.clone()));
        }
        let leads = snapshot.intents.len();
        for (i, fact) in snapshot.facts.iter().enumerate() {
            let mut from: Vec<String> = fact.from_intent.iter().cloned().collect();
            from.extend(fact.from_fact_ids.iter().cloned());
            wanted.push((leads + i, from));
        }
        for (node, from) in wanted {
            let known: Vec<&String> = from.iter().filter(|id| index.contains_key(id.as_str())).collect();
            let Some(first) = known.first() else {
                continue;
            };
            let parent = index[first.as_str()];
            // Never under itself, however the records point.
            let mut up = Some(parent);
            let mut steps = 0;
            let mut cycle = false;
            while let Some(at) = up {
                if at == node || steps > nodes.len() {
                    cycle = true;
                    break;
                }
                up = nodes[at].parent;
                steps += 1;
            }
            if cycle {
                continue;
            }
            nodes[node].parent = Some(parent);
            nodes[node].also = known[1..].iter().map(|id| short(id)).collect();
        }
        // Children, in reading order.
        let order = |node: &Node| -> (u8, u8, i64, String) {
            match node.kind {
                Kind::Lead(i) => {
                    let intent = &snapshot.intents[i];
                    (1, lead_rank(intent, now), -intent.priority, intent.created_at.clone())
                }
                Kind::Finding(i) => (0, 0, 0, snapshot.facts[i].created_at.clone()),
                Kind::Hint(i) => (2, 0, 0, snapshot.hints[i].created_at.clone()),
                Kind::Board => (3, 0, 0, String::new()),
            }
        };
        // The board's own row, and the operator's hints: each under what it is
        // about, or else under the board.
        let board = nodes.len();
        nodes.push(Node { id: BOARD_ID.into(), kind: Kind::Board, parent: None, children: Vec::new(), also: Vec::new() });
        for (i, hint) in snapshot.hints.iter().enumerate() {
            let about = hint_parts(&hint.text).0;
            let parent = about
                .and_then(|about| nodes.iter().position(|node| matches!(node.kind, Kind::Lead(_) | Kind::Finding(_)) && short(&node.id) == about))
                .unwrap_or(board);
            nodes.push(Node { id: hint.id.clone(), kind: Kind::Hint(i), parent: Some(parent), children: Vec::new(), also: Vec::new() });
        }
        let mut roots: Vec<usize> = Vec::new();
        for i in 0..nodes.len() {
            match nodes[i].parent {
                Some(parent) => nodes[parent].children.push(i),
                None if i != board => roots.push(i),
                None => {}
            }
        }
        for i in 0..nodes.len() {
            let mut children = std::mem::take(&mut nodes[i].children);
            children.sort_by_key(|&child| order(&nodes[child]));
            nodes[i].children = children;
        }
        // At the root, leads first (findings with no lead after them).
        roots.sort_by_key(|&root| {
            let (kind, rank, priority, created) = order(&nodes[root]);
            (if kind == 1 { 0 } else { 1 }, rank, priority, created)
        });
        // The board itself, first.
        roots.insert(0, board);
        let grey = vec![false; nodes.len()];
        Tree { nodes, roots, grey }
    }

    /// Greys out the nodes in `greyed` (by id), and everything under them,
    /// and moves each greyed node to the end of its siblings, its children
    /// with it: a root to the bottom of the board.
    pub fn sink(&mut self, greyed: &HashSet<String>) {
        fn mark(tree: &mut Tree, greyed: &HashSet<String>, node: usize, above: bool) {
            let grey = above || greyed.contains(&tree.nodes[node].id);
            tree.grey[node] = grey;
            for child in tree.nodes[node].children.clone() {
                mark(tree, greyed, child, grey);
            }
        }
        for root in self.roots.clone() {
            mark(self, greyed, root, false);
        }
        let grey = std::mem::take(&mut self.grey);
        // Stable: the order within each part is kept.
        self.roots.sort_by_key(|&node| grey[node]);
        for node in &mut self.nodes {
            node.children.sort_by_key(|&child| grey[child]);
        }
        self.grey = grey;
    }

    /// The nodes from the root down to `node`.
    fn path(&self, node: usize) -> Vec<usize> {
        let mut path = vec![node];
        while let Some(parent) = self.nodes[*path.last().unwrap()].parent {
            path.push(parent);
        }
        path.reverse();
        path
    }

    /// Every node under `node`, at any depth.
    pub fn descendants(&self, node: usize) -> usize {
        self.nodes[node].children.iter().map(|&child| 1 + self.descendants(child)).sum()
    }

    pub fn find(&self, id: &str) -> Option<usize> {
        self.nodes.iter().position(|node| node.id == id)
    }
}

/// What can be greyed out: a lead, a finding, a hint; not the board's row.
pub fn can_grey(tree: &Tree, node: usize) -> bool {
    tree.nodes[node].kind != Kind::Board
}

/// Ctrl+C on `node`: greyed out, with what is under it; or, when it is grey,
/// back (`revive`).
pub fn toggle_grey(tree: &Tree, greyed: &mut HashSet<String>, node: usize) {
    if !can_grey(tree, node) {
        return;
    }
    if tree.grey[node] {
        bring_back(tree, greyed, node, false);
    } else {
        greyed.insert(tree.nodes[node].id.clone());
    }
}

/// A hint on `node`: it and the nodes above it out of the grey, and only
/// those. The other branches under a node brought back, and what is under
/// `node`, stay grey.
pub fn revive(tree: &Tree, greyed: &mut HashSet<String>, node: usize) {
    bring_back(tree, greyed, node, true);
}

/// `node` and the nodes above it out of the grey; the other branches under
/// them stay grey, and so does what is under `node` when `below` says so.
fn bring_back(tree: &Tree, greyed: &mut HashSet<String>, node: usize, below: bool) {
    if !tree.grey[node] {
        return;
    }
    let path = tree.path(node);
    for (at, &on) in path.iter().enumerate() {
        let next = path.get(at + 1).copied();
        let pushed = greyed.remove(&tree.nodes[on].id) || (below && next.is_none());
        if !pushed || (next.is_none() && !below) {
            continue;
        }
        // It greyed out what was under it: those off the path stay grey.
        for &child in &tree.nodes[on].children {
            if Some(child) != next {
                greyed.insert(tree.nodes[child].id.clone());
            }
        }
    }
}

impl Folds {
    /// Unfolded at first: everything but ruled-out leads.
    fn default_open(tree: &Tree, snapshot: &Snapshot, node: usize) -> bool {
        match tree.nodes[node].kind {
            Kind::Lead(i) => snapshot.intents[i].state != IntentState::Dead,
            Kind::Board | Kind::Finding(_) | Kind::Hint(_) => true,
        }
    }

    /// New nodes take their default; the rest stay as they were left.
    pub fn update(&mut self, tree: &Tree, snapshot: &Snapshot) {
        for (i, node) in tree.nodes.iter().enumerate() {
            if self.seen.insert(node.id.clone()) && Self::default_open(tree, snapshot, i) {
                self.open.insert(node.id.clone());
            }
        }
    }

    pub fn is_open(&self, tree: &Tree, node: usize) -> bool {
        self.open.contains(&tree.nodes[node].id)
    }

    pub fn set(&mut self, tree: &Tree, node: usize, open: bool) {
        let id = tree.nodes[node].id.clone();
        if open {
            self.open.insert(id);
        } else {
            self.open.remove(&id);
        }
    }

    /// Everything unfolded, or (when it all is) everything folded.
    pub fn toggle_all(&mut self, tree: &Tree) {
        let branches: Vec<usize> = (0..tree.nodes.len()).filter(|&i| !tree.nodes[i].children.is_empty()).collect();
        let all_open = branches.iter().all(|&i| self.is_open(tree, i));
        for i in branches {
            self.set(tree, i, !all_open);
        }
    }
}

/// The rows to draw, top to bottom: every root, and under each unfolded node
/// its children.
pub fn rows(tree: &Tree, folds: &Folds) -> Vec<Row> {
    fn walk(tree: &Tree, folds: &Folds, nodes: &[usize], guides: &mut Vec<bool>, out: &mut Vec<Row>) {
        for (position, &node) in nodes.iter().enumerate() {
            let last = position + 1 == nodes.len();
            out.push(Row { node, guides: guides.clone(), last });
            if folds.is_open(tree, node) && !tree.nodes[node].children.is_empty() {
                guides.push(!last);
                walk(tree, folds, &tree.nodes[node].children, guides, out);
                guides.pop();
            }
        }
    }
    let mut out = Vec::new();
    walk(tree, folds, &tree.roots, &mut Vec::new(), &mut out);
    out
}

// ------------------------------------------------------------------ drawing

/// `4m`, `2h`, `3d`: how long ago, short, for a row's tail.
fn ago(stamp: &str, now: DateTime<Local>) -> String {
    let Ok(then) = DateTime::parse_from_rfc3339(stamp) else {
        return String::new();
    };
    let seconds = (now.fixed_offset() - then).num_seconds().max(0);
    match seconds {
        0..60 => format!("{seconds}s"),
        60..3_600 => format!("{}m", seconds / 60),
        3_600..86_400 => format!("{}h", seconds / 3_600),
        _ => format!("{}d", seconds / 86_400),
    }
}

/// Who holds a claim: the subagent's tab and short id.
fn holder(run_id: &str) -> String {
    if let Some(audit) = run_id.strip_prefix("audit:") {
        return format!("audit·{audit}");
    }
    match tab_of(run_id) {
        Some(number) => format!("alt {number} · {}", short_id(run_id)),
        None => short_id(run_id),
    }
}

/// A lead's icon, its colour, and how its text is drawn.
/// The markers, as the agent's page uses them: `[+]` done or for, `[*]` under
/// way, `[-]` ruled out or ended, `[?]` open (not tried yet), `[!]` wants
/// attention (the operator's notes, a spent budget).
pub fn status_marker(status: BoardStatus) -> &'static str {
    match status {
        BoardStatus::Solved => "[+]",
        BoardStatus::Open | BoardStatus::ProposedSolved => "[*]",
        BoardStatus::ProposedExhausted => "[*]",
        BoardStatus::Suspended => "[!]",
        BoardStatus::Exhausted | BoardStatus::Abandoned => "[-]",
    }
}

fn lead_look(intent: &Intent, now: DateTime<Local>) -> (&'static str, Style, Style, &'static str) {
    match lead_rank(intent, now) {
        0 => ("[*]", Style::Warn, Style::Bold, "exploring"),
        1 => ("[?]", Style::Bold, Style::Bold, "open"),
        2 => ("[+]", Style::Success, Style::Plain, "explored"),
        _ => ("[-]", Style::Danger, Style::Dim, "ruled out"),
    }
}

fn finding_look(fact: &Fact) -> (&'static str, Style, &'static str) {
    match fact.polarity {
        Polarity::Supports => ("[+]", Style::Success, "supports"),
        Polarity::RulesOut => ("[-]", Style::Danger, "rules out"),
    }
}

/// One row, cut to `width`: the cursor, the guides, the fold, the icon, the
/// text and, dim, what it is.
pub fn row_line(tree: &Tree, snapshot: &Snapshot, folds: &Folds, row: &Row, selected: bool, width: usize, now: DateTime<Local>) -> Line {
    let node = &tree.nodes[row.node];
    let mut line: Line = vec![if selected { (Style::Cursor, " ❯ ".into()) } else { (Style::Plain, "   ".into()) }];
    // Under each ancestor below the roots, a line where it has more after it;
    // then this row's own branch. The roots stand on their own, so their
    // level draws nothing.
    let mut lead_in = String::new();
    if !row.guides.is_empty() {
        for &more in row.guides.iter().skip(1) {
            lead_in.push_str(if more { "│  " } else { "   " });
        }
        lead_in.push_str(if row.last { "└─ " } else { "├─ " });
    }
    line.push((Style::Dim, lead_in));
    let folded = !node.children.is_empty() && !folds.is_open(tree, row.node);
    line.push(match (node.children.is_empty(), folded) {
        (true, _) => (Style::Plain, "  ".into()),
        (false, true) => (Style::Cursor, "▸ ".into()),
        (false, false) => (Style::Status, "▾ ".into()),
    });

    let (icon, text, meta): (Segment, Segment, Vec<Segment>) = match node.kind {
        Kind::Board => (
            (Style::Heading, format!("{} origin  ", status_marker(snapshot.board.status))),
            (Style::Plain, snapshot.board.origin.clone()),
            vec![(Style::Dim, "  full below".into())],
        ),
        Kind::Lead(i) => {
            let intent = &snapshot.intents[i];
            let (icon, icon_style, text_style, _) = lead_look(intent, now);
            let mut meta = vec![(if intent.priority >= 8 { Style::Bold } else { Style::Dim }, format!("  p{}", intent.priority))];
            if lead_rank(intent, now) == 0 {
                let who = intent.claimed_by.as_deref().map(holder).unwrap_or_else(|| "someone".into());
                let since = intent.claimed_at.as_deref().map(|at| format!(" {}", ago(at, now))).unwrap_or_default();
                meta.push((Style::Warn, format!(" · {who}{since}")));
            }
            meta.push((Style::Dim, format!("  {}", short(&intent.id))));
            ((icon_style, format!("{icon} ")), (text_style, intent.direction.clone()), meta)
        }
        Kind::Finding(i) => {
            let fact = &snapshot.facts[i];
            let (icon, icon_style, _) = finding_look(fact);
            let mut meta = Vec::new();
            match fact.author {
                FactAuthor::Operator => meta.push((Style::Operator, "  you".to_string())),
                FactAuthor::Harness => meta.push((Style::Status, "  harness".to_string())),
                FactAuthor::Model => {}
            }
            if fact.basis == Basis::Inferred {
                meta.push((Style::Work, "  inferred".to_string()));
            }
            meta.push((Style::Dim, format!("  {}", short(&fact.id))));
            ((icon_style, format!("{icon} ")), (Style::Plain, fact.statement.clone()), meta)
        }
        Kind::Hint(i) => {
            let hint = &snapshot.hints[i];
            let mut meta = vec![match hint.consumed_at {
                Some(_) => (Style::Dim, "  read".to_string()),
                None => (Style::Problem, "  unread".to_string()),
            }];
            let (about, text) = hint_parts(&hint.text);
            // Said only when it could not stand under what it is about.
            let under_board = node.parent.is_some_and(|parent| tree.nodes[parent].kind == Kind::Board);
            if let Some(about) = about.filter(|_| under_board) {
                meta.push((Style::Status, format!("  on {about}")));
            }
            meta.push((Style::Dim, format!("  {}", short(&hint.id))));
            ((Style::Operator, if hint.consumed_at.is_none() { "[!] " } else { "[*] " }.into()), (Style::Plain, text.to_string()), meta)
        }
    };
    let mut meta = meta;
    // Greyed out: the row in a work line's italic grey, all but the
    // selection and the fold marks.
    let grey = tree.grey.get(row.node).copied().unwrap_or(false);
    let (icon, text) = if grey { ((Style::Work, icon.1), (Style::Work, text.1)) } else { (icon, text) };
    if grey {
        for piece in &mut meta {
            piece.0 = Style::Work;
        }
    }
    if folded {
        let hidden = tree.descendants(row.node);
        meta.push((Style::Cursor, format!("  +{hidden}")));
    }
    if !node.also.is_empty() {
        meta.push((Style::Dim, format!("  also {}", node.also.join(", "))));
    }

    // The text gives way, then the tail, so the row is one row.
    let used: usize = line.iter().map(|(_, t)| text_width(t)).sum::<usize>() + text_width(&icon.1);
    let tail: usize = meta.iter().map(|(_, t)| text_width(t)).sum();
    let room = width.saturating_sub(used + 1);
    let text_room = room.saturating_sub(tail).max(room.min(12));
    let shown = clip(&text.1.split_whitespace().collect::<Vec<_>>().join(" "), text_room);
    let mut left = room.saturating_sub(text_width(&shown));
    line.push(icon);
    line.push((text.0, shown));
    for (style, piece) in meta {
        if left == 0 {
            break;
        }
        let piece = if text_width(&piece) > left { clip(&piece, left) } else { piece };
        left = left.saturating_sub(text_width(&piece));
        line.push((style, piece));
    }
    line
}

/// Under the head of a board the agent proposes exhausted: its argument, and
/// the operator's keys (he closes it exhausted, or opens it again).
pub fn exhaustion_proposed(snapshot: &Snapshot, width: usize) -> Vec<Line> {
    let case = snapshot.board.outcome.as_deref().map(str::trim).filter(|case| !case.is_empty()).unwrap_or("(no argument given)");
    let mut lines = vec![Vec::new()];
    let mut said = flow(&[(Style::Work, "  [*] s0ma says nothing is left to try: ".into()), (Style::Bold, case.to_string())], width, 6);
    if said.len() > 3 {
        said.truncate(3);
        if let Some(last) = said.last_mut() {
            last.push((Style::Dim, " … (the board's row has it all)".into()));
        }
    }
    lines.extend(said);
    lines.push(vec![(Style::Dim, "      s0ma reports what it tried (board_report), then it comes to your review".into())]);
    lines
}

/// Under the head of a board the agent finished: its write-up is due, and then
/// it goes to the operator's review (the projects tab's next subtab).
pub fn write_up_due(snapshot: &Snapshot, width: usize) -> Vec<Line> {
    let case = snapshot.board.outcome.as_deref().map(str::trim).filter(|case| !case.is_empty()).unwrap_or("(no case given)");
    let mut lines = vec![Vec::new()];
    let mut said = flow(&[(Style::Work, "  [*] finished: ".into()), (Style::Bold, case.to_string())], width, 6);
    if said.len() > 3 {
        said.truncate(3);
        if let Some(last) = said.last_mut() {
            last.push((Style::Dim, " … (the board's row has it all)".into()));
        }
    }
    lines.extend(said);
    lines.push(vec![(Style::Dim, "      s0ma writes its report (board_report), then it comes to your review".into())]);
    lines
}

/// The board's head: its name, status and title; the goal and its test; the
/// steps as a bar, and how the leads stand.
pub fn header(snapshot: &Snapshot, width: usize, now: DateTime<Local>) -> Vec<Line> {
    let board = &snapshot.board;
    let status = match board.status {
        BoardStatus::Open => (Style::Work, format!("{} open", status_marker(board.status))),
        BoardStatus::Solved => (Style::Success, format!("{} solved", status_marker(board.status))),
        BoardStatus::ProposedSolved => (Style::Work, format!("{} proposed solved", status_marker(board.status))),
        BoardStatus::ProposedExhausted => (Style::Warn, format!("{} proposed exhausted", status_marker(board.status))),
        BoardStatus::Suspended => (Style::Warn, format!("{} suspended", status_marker(board.status))),
        BoardStatus::Exhausted => (Style::Danger, format!("{} exhausted", status_marker(board.status))),
        BoardStatus::Abandoned => (Style::Danger, format!("{} abandoned", status_marker(board.status))),
    };
    let mut lines = vec![Vec::new()];
    let mut title = vec![(Style::Accent, format!("  {}", board.name)), (Style::Plain, "  ".into()), status, (Style::Plain, "  ".into())];
    let used: usize = title.iter().map(|(_, t)| text_width(t)).sum();
    title.push((Style::Bold, clip(&board.title, width.saturating_sub(used + 1))));
    lines.push(title);
    let field = |label: &str, value: Vec<(Style, String)>| -> Vec<Line> {
        let mut segments = vec![(Style::Dim, format!("  {label:<7}"))];
        segments.extend(value);
        let mut rows = flow(&segments, width, 9);
        // Two rows at most: the strip below has the rest.
        if rows.len() > 2 {
            rows.truncate(2);
            if let Some(last) = rows.last_mut() {
                // Room for the mark: a character off the end if the row is full.
                while last.iter().map(|(_, t)| text_width(t)).sum::<usize>() + 1 > width {
                    match last.last_mut() {
                        Some((_, piece)) if !piece.is_empty() => {
                            piece.pop();
                        }
                        Some(_) => {
                            last.pop();
                        }
                        None => break,
                    }
                }
                last.push((Style::Dim, "…".into()));
            }
        }
        rows
    };
    lines.extend(field("goal", vec![(Style::Plain, board.goal.clone())]));
    lines.extend(field("test", vec![(Style::Status, board.goal_test.clone()), (Style::Dim, format!("  {}", board.goal_test_kind.label()))]));

    // Steps as a bar that warms as it fills, and the leads by state.
    let budget = board.step_budget.max(1);
    let used_steps = board.steps_used.clamp(0, budget);
    let cells = 12;
    let filled = ((used_steps * cells + budget - 1) / budget) as usize;
    let fill = match used_steps * 100 / budget {
        0..60 => Style::Success,
        60..85 => Style::Warn,
        _ => Style::Danger,
    };
    let mut steps = vec![
        (Style::Dim, "  steps  ".to_string()),
        (fill, "█".repeat(filled)),
        (Style::Dim, "░".repeat(cells as usize - filled)),
        (Style::Plain, format!("  {} of {}", board.steps_used, board.step_budget)),
    ];
    let mut counts = [0usize; 4];
    for intent in &snapshot.intents {
        counts[lead_rank(intent, now) as usize] += 1;
    }
    let mut tally: Vec<(Style, String)> = counts
        .iter()
        .zip([(Style::Warn, "exploring"), (Style::Bold, "open"), (Style::Success, "explored"), (Style::Danger, "ruled out")])
        .filter(|(count, _)| **count > 0)
        .map(|(count, (style, label))| (style, format!("   {count} {label}")))
        .collect();
    let findings = snapshot.facts.len();
    if findings > 0 {
        tally.push((Style::Dim, format!("   {findings} finding{}", if findings == 1 { "" } else { "s" })));
    }
    // As many of the counts as fit, in order.
    let mut used: usize = steps.iter().map(|(_, t)| text_width(t)).sum();
    for (style, piece) in tally {
        used += text_width(&piece);
        if used >= width {
            break;
        }
        steps.push((style, piece));
    }
    lines.push(steps);
    // Its audits and its audit policy: only on a project that has one, or a
    // policy of its own (one without looks as it always did).
    if let Some(value) = audits_row(&board.name) {
        lines.extend(field("audits", value));
    }
    // A proposed solution's case is under the head, with what the operator can say
    // (`proposal`).
    if let Some(outcome) = board.outcome.as_ref().filter(|_| board.status != BoardStatus::ProposedSolved) {
        lines.extend(field("result", vec![(Style::Bold, outcome.clone())]));
    }
    lines.push(Vec::new());
    lines
}

/// The board head's audits row: its audits (a proposal marked `[!]`), each
/// opening the audit tab, and its policy. None on a board with no audit and
/// no policy of its own. Read at most every two seconds per board; the
/// clickable spans are made anew each frame.
fn audits_row(board_name: &str) -> Option<Vec<(Style, String)>> {
    type Row = Option<(Vec<(String, Style, String, bool)>, String)>;
    thread_local! {
        static SEEN: std::cell::RefCell<HashMap<String, (Row, std::time::Instant)>> = std::cell::RefCell::new(HashMap::new());
    }
    let cached = SEEN.with(|seen| seen.borrow().get(board_name).filter(|(_, at)| at.elapsed() < std::time::Duration::from_secs(2)).map(|(row, _)| row.clone()));
    let row: Row = match cached {
        Some(row) => row,
        None => {
            let row = (|| -> Row {
                let store = crate::audit::store()?;
                let audits = crate::audit::ops::for_board(board_name);
                if audits.is_empty() && !store.has_policy(board_name).unwrap_or(false) {
                    return None;
                }
                let items = audits
                    .iter()
                    .map(|audit| {
                        let (mark, status) = audit.status_mark();
                        let (look, text) = if audit.status == crate::audit::Status::Proposed {
                            (Style::Warn, format!("[!] audit proposed {}", audit.name))
                        } else {
                            (Style::Plain, format!("{} {mark} {status}", audit.name))
                        };
                        (audit.name.clone(), look, text, audit.board.as_deref() != Some(board_name))
                    })
                    .collect();
                Some((items, store.policy(board_name).unwrap_or_default().label().to_string()))
            })();
            SEEN.with(|seen| seen.borrow_mut().insert(board_name.to_string(), (row.clone(), std::time::Instant::now())));
            row
        }
    };
    let (items, policy) = row?;
    let mut value: Vec<(Style, String)> = Vec::new();
    for (name, look, text, linked) in items {
        let click = super::gui_view::Click::Pick(super::gui_view::Pick::OpenAudit(name, None));
        value.push(super::gui_view::hot(look, text, click));
        if linked {
            value.push((Style::Dim, " (linked)".into()));
        }
        value.push((Style::Dim, "  ".into()));
    }
    value.push((Style::Dim, format!("policy {policy}")));
    Some(value)
}

/// An audit result's reference in a finding: `audit:<name>/c<N>`.
fn audit_ref(statement: &str) -> Option<(String, String)> {
    let at = statement.find("audit:")?;
    let rest = &statement[at + 6..];
    let end = rest.find(|c: char| c.is_whitespace() || matches!(c, ')' | ',' | ']')).unwrap_or(rest.len());
    let (audit, claim) = rest[..end].split_once('/')?;
    Some((audit.to_string(), claim.to_string()))
}

/// The audit holding a lead, in one line: its status and counts, its name
/// opening the audit tab.
fn audit_head(name: &str) -> Vec<(Style, String)> {
    let click = super::gui_view::Click::Pick(super::gui_view::Pick::OpenAudit(name.to_string(), None));
    let mut out = vec![super::gui_view::hot(Style::Accent, name.to_string(), click)];
    if let Some(store) = crate::audit::store()
        && let Ok(Some(audit)) = store.audit(name)
    {
        let claims = store.claims(name).unwrap_or_default();
        let (mark, status) = audit.status_mark();
        let failing = claims.iter().filter(|c| c.verdict == crate::audit::Verdict::Fails && !c.waived).count();
        let holding = claims.iter().filter(|c| c.verdict == crate::audit::Verdict::Holds).count();
        let waiting = claims.iter().filter(|c| c.mark().0 == "[!]").count();
        out.push((Style::Warn, format!("  {mark} {status}")));
        out.push((Style::Dim, format!(" · {} claims", claims.len())));
        if failing > 0 {
            out.push((Style::Danger, format!(" · [-] {failing} failing")));
        }
        if holding > 0 {
            out.push((Style::Success, format!(" · [+] {holding} hold")));
        }
        if waiting > 0 {
            out.push((Style::Warn, format!(" · [!] {waiting}")));
        }
    }
    out.push((Style::Dim, " · click opens it (Alt+U)".into()));
    out
}

/// The selected node in full, for the strip under the tree.
pub fn detail(tree: &Tree, snapshot: &Snapshot, node: usize, width: usize, now: DateTime<Local>) -> Vec<Line> {
    // Labels as wide as the widest ("exploring", "closed by"), and a space.
    let label = |text: &str| (Style::Dim, format!("  {text:<10}"));
    let field = |name: &str, value: Vec<(Style, String)>| -> Vec<Line> {
        let mut segments = vec![label(name)];
        segments.extend(value);
        flow(&segments, width, 12)
    };
    let node_ref = &tree.nodes[node];
    let mut out: Vec<Line> = Vec::new();
    match node_ref.kind {
        Kind::Lead(i) => {
            let intent = &snapshot.intents[i];
            let (icon, icon_style, _, state) = lead_look(intent, now);
            out.push(vec![
                (icon_style, format!("  {icon} ")),
                (Style::Heading, format!("lead {}", short(&intent.id))),
                (icon_style, format!("  {state}")),
                (Style::Dim, format!("  · priority {} · {} ago", intent.priority, ago(&intent.created_at, now))),
            ]);
            out.extend(flow(&[(Style::Bold, format!("  {}", intent.direction))], width, 2));
            if !intent.rationale.trim().is_empty() {
                out.extend(field("why", vec![(Style::Plain, intent.rationale.clone())]));
            }
            if lead_rank(intent, now) == 0 {
                let who = intent.claimed_by.as_deref().map(holder).unwrap_or_else(|| "someone".into());
                let since = intent.claimed_at.as_deref().map(|at| format!(" for {}", ago(at, now))).unwrap_or_default();
                out.extend(field("exploring", vec![(Style::Warn, format!("{who}{since}"))]));
            }
            let from: Vec<String> = intent.from_fact_ids.iter().map(|id| short(id)).collect();
            if !from.is_empty() {
                out.extend(field("from", vec![(Style::Status, from.join(", "))]));
            }
            if let Some(fact) = &intent.closed_by_fact {
                out.extend(field("closed by", vec![(Style::Status, short(fact))]));
            }
            if let Some(audit) = intent.claimed_by.as_deref().and_then(|by| by.strip_prefix("audit:")) {
                out.extend(field("audit", audit_head(audit)));
            }
        }
        Kind::Finding(i) => {
            let fact = &snapshot.facts[i];
            let (icon, icon_style, polarity) = finding_look(fact);
            let who = match fact.author {
                FactAuthor::Operator => "you",
                FactAuthor::Harness => "the harness",
                FactAuthor::Model => "s0ma",
            };
            out.push(vec![
                (icon_style, format!("  {icon} ")),
                (Style::Heading, format!("finding {}", short(&fact.id))),
                (icon_style, format!("  {polarity}")),
                (Style::Dim, format!("  · {} · by {who} · {} ago", fact.basis.label(), ago(&fact.created_at, now))),
            ]);
            out.extend(flow(&[(Style::Bold, format!("  {}", fact.statement))], width, 2));
            if let Some((audit, claim)) = audit_ref(&fact.statement) {
                let click = super::gui_view::Click::Pick(super::gui_view::Pick::OpenAudit(audit.clone(), Some(claim.clone())));
                out.extend(field("audit", vec![super::gui_view::hot(Style::Accent, format!("{audit}/{claim} · open the claim in Alt+U"), click)]));
            }
            let mut from: Vec<String> = fact.from_intent.iter().map(|id| short(id)).collect();
            from.extend(fact.from_fact_ids.iter().map(|id| short(id)));
            if !from.is_empty() {
                out.extend(field("from", vec![(Style::Status, from.join(", "))]));
            }
            if !fact.evidence_event_ids.is_empty() {
                let ids: Vec<String> = fact.evidence_event_ids.iter().map(|id| format!("#{id}")).collect();
                out.extend(field("evidence", vec![(Style::Status, ids.join(" ")), (Style::Dim, "  archive_get reads one".into())]));
            }
        }
        Kind::Board => {
            let board = &snapshot.board;
            let mut head = vec![(Style::Heading, format!("  {} board {}", status_marker(board.status), board.name)), (Style::Dim, format!("  · {}", board.status.label()))];
            head.push((Style::Dim, format!(" · opened {} ago", ago(&board.created_at, now))));
            if let Some(closed) = &board.closed_at {
                head.push((Style::Dim, format!(" · closed {} ago", ago(closed, now))));
            }
            out.push(head);
            out.extend(flow(&[(Style::Bold, format!("  {}", board.title))], width, 2));
            out.extend(field("origin", vec![(Style::Plain, board.origin.clone())]));
            out.extend(field("goal", vec![(Style::Plain, board.goal.clone())]));
            out.extend(field("test", vec![(Style::Status, board.goal_test.clone()), (Style::Dim, format!("  {}", board.goal_test_kind.label()))]));
            out.extend(field("steps", vec![(Style::Plain, format!("{} of {} used", board.steps_used, board.step_budget))]));
            if let Some(outcome) = &board.outcome {
                out.extend(field("outcome", vec![(Style::Bold, outcome.clone())]));
            }
        }
        Kind::Hint(i) => {
            let hint = &snapshot.hints[i];
            let read = match &hint.consumed_at {
                Some(at) => (Style::Dim, format!("  read {} ago", ago(at, now))),
                None => (Style::Problem, "  unread".to_string()),
            };
            let (about, text) = hint_parts(&hint.text);
            let mark = if hint.consumed_at.is_none() { "[!]" } else { "[*]" };
            let mut head = vec![(Style::Operator, format!("  {mark} hint {}", short(&hint.id))), read];
            if let Some(about) = about {
                head.push((Style::Status, format!("  on {about}")));
            }
            head.push((Style::Dim, format!("  · {} ago", ago(&hint.created_at, now))));
            out.push(head);
            out.extend(flow(&[(Style::Plain, format!("  {text}"))], width, 2));
            out.push(vec![(Style::Dim, "  stays on the board: s0ma sees it on every read · delete twice takes it off".into())]);
        }
    }
    out
}

/// The hint as sent from a selected lead or finding: it says which, so the agent
/// knows where the operator is pointing.
pub fn hint_about(tree: &Tree, snapshot: &Snapshot, node: Option<usize>, text: &str) -> String {
    let about = node.and_then(|node| match tree.nodes[node].kind {
        Kind::Lead(i) => Some((&snapshot.intents[i].id, &snapshot.intents[i].direction)),
        Kind::Finding(i) => Some((&snapshot.facts[i].id, &snapshot.facts[i].statement)),
        Kind::Board | Kind::Hint(_) => None,
    });
    match about {
        Some((id, what)) => format!("On {id} ({}): {text}", clip(&what.split_whitespace().collect::<Vec<_>>().join(" "), 60)),
        None => text.to_string(),
    }
}

/// A hint's text without the `On intent-0007 (…): ` that `hint_about` put in
/// front, and what it is about (`i7`), when it has one.
fn hint_parts(text: &str) -> (Option<String>, &str) {
    let Some(rest) = text.strip_prefix("On ") else {
        return (None, text);
    };
    let Some((id, _)) = rest.split_once(' ') else {
        return (None, text);
    };
    let known = ["intent-", "fact-"].iter().any(|prefix| id.starts_with(prefix));
    let after = &rest[id.len()..];
    if !known || !after.starts_with(" (") {
        return (None, text);
    }
    // The quoted wording ends at the `)` that closes its `(`, so a lead
    // worded `foo(): bar` is not cut inside; one whose brackets don't
    // balance (cut short) ends at the first `): `.
    let quoted = &after[1..];
    let mut depth = 0usize;
    let mut close = None;
    for (at, c) in quoted.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    close = Some(at);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = close.filter(|at| quoted[at + 1..].starts_with(": ")).map(|at| id.len() + 1 + at + 1).or_else(|| rest.find("): ").map(|at| at + 1));
    match end {
        Some(end) => (Some(short(id)), &rest[end + 2..]),
        None => (None, text),
    }
}

/// What the prompt calls the selected node, if a hint would be about it.
pub fn prompt_target(tree: &Tree, node: Option<usize>) -> Option<String> {
    let node = &tree.nodes[node?];
    matches!(node.kind, Kind::Lead(_) | Kind::Finding(_)).then(|| short(&node.id))
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;


    #[test]
    fn a_hint_on_a_lead_worded_with_brackets_keeps_its_text() {
        assert_eq!(hint_parts("On intent-0007 (Does fsync(): wait for the disk?): check the mount"), (Some("i7".into()), "check the mount"));
        assert_eq!(hint_parts("On fact-0003 (plain words): look again"), (Some("f3".into()), "look again"));
        assert_eq!(hint_parts("On intent-0002 (cut short (…): still read"), (Some("i2".into()), "still read"));
        assert_eq!(hint_parts("just a note"), (None, "just a note"));
    }
}
