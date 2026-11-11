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
use crate::mistake_view::{Line, Style};

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
    pub flow: Option<crate::board_flow::Flow>,
    pub notes: Vec<crate::board_flow::Note>,
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
    /// (`sink`). s0ma never sees it.
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
    match tab_of(run_id) {
        Some(number) => format!("alt {number} · {}", short_id(run_id)),
        None => short_id(run_id),
    }
}

/// A lead's icon, its colour, and how its text is drawn.
/// The markers, as s0ma's page uses them: `[+]` done or for, `[*]` under
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

/// Under the head of a board s0ma proposes exhausted: its argument, and
/// The operator's keys (he closes it exhausted, or opens it again).
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

/// Under the head of a board s0ma finished: its write-up is due, and then
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
    // A proposed solution's case is under the head, with what the operator can say
    // (`proposal`).
    if let Some(outcome) = board.outcome.as_ref().filter(|_| board.status != BoardStatus::ProposedSolved) {
        lines.extend(field("result", vec![(Style::Bold, outcome.clone())]));
    }
    lines.push(Vec::new());
    lines
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

/// The hint as sent from a selected lead or finding: it says which, so s0ma
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
    use crate::board::GoalTestKind;

    fn at(minutes_ago: i64) -> String {
        (Local::now() - chrono::Duration::minutes(minutes_ago)).to_rfc3339()
    }

    pub fn lead(id: &str, direction: &str, state: IntentState, priority: i64, from: &[&str], minutes_ago: i64) -> Intent {
        Intent {
            id: id.into(),
            direction: direction.into(),
            rationale: format!("because {direction}"),
            from_fact_ids: from.iter().map(|s| s.to_string()).collect(),
            state,
            priority,
            claimed_by: (state == IntentState::Claimed).then(|| "subagent-1-2-3-1".into()),
            claimed_at: (state == IntentState::Claimed).then(|| at(4)),
            prior_claims: Vec::new(),
            closed_by_fact: None,
            created_at: at(minutes_ago),
        }
    }

    pub fn finding(id: &str, statement: &str, polarity: Polarity, intent: Option<&str>, from: &[&str], minutes_ago: i64) -> Fact {
        Fact {
            id: id.into(),
            statement: statement.into(),
            polarity,
            basis: Basis::Observed,
            author: FactAuthor::Model,
            from_intent: intent.map(str::to_string),
            evidence_event_ids: vec![1423, 1430],
            from_fact_ids: from.iter().map(|s| s.to_string()).collect(),
            created_at: at(minutes_ago),
        }
    }

    /// The board from the design: a lead being explored with two findings, one
    /// of which opened a lead; an open lead; a ruled-out one; a hint.
    pub fn snapshot() -> Snapshot {
        Snapshot {
            board: Board {
                id: "board-0001".into(),
                name: "flaky-upload".into(),
                title: "Find why test_upload flakes on CI".into(),
                origin: "1 in 8 on CI".into(),
                goal: "test_upload passes 50 runs in a row on CI".into(),
                goal_test: "for i in $(seq 50); do pytest -q tests/test_upload.py || exit 1; done".into(),
                goal_test_kind: GoalTestKind::Command,
                status: BoardStatus::Open,
                step_budget: 12,
                steps_used: 7,
                created_at: at(60),
                closed_at: None,
                outcome: None,
            },
            intents: vec![
                lead("intent-0004", "Pin the S3 client to the local version", IntentState::Dead, 4, &[], 50),
                lead("intent-0007", "Does the CI runner's /tmp fill up mid-run?", IntentState::Open, 7, &[], 45),
                lead("intent-0008", "Race between the upload and the cleanup fixture", IntentState::Claimed, 8, &[], 40),
                lead("intent-0009", "Hold the fixture until the handle closes", IntentState::Open, 6, &["fact-0003"], 10),
            ],
            facts: vec![
                finding("fact-0004", "no race with one CPU core", Polarity::RulesOut, Some("intent-0008"), &[], 5),
                finding("fact-0003", "cleanup ran before the upload's fsync, 2 runs of 8", Polarity::Supports, Some("intent-0008"), &[], 20),
                finding("fact-0002", "the S3 client is the same version", Polarity::RulesOut, Some("intent-0004"), &[], 48),
            ],
            hints: vec![Hint { id: "hint-0001".into(), text: "look at the fixture's teardown".into(), created_at: at(3), consumed_at: None }],
            flow: None,
            notes: Vec::new(),
        }
    }

    fn ids(tree: &Tree, rows: &[Row]) -> Vec<String> {
        rows.iter().map(|row| short(&tree.nodes[row.node].id)).collect()
    }

    fn text(line: &Line) -> String {
        line.iter().map(|(_, t)| t.as_str()).collect()
    }

    #[test]
    fn a_hint_on_a_lead_worded_with_brackets_keeps_its_text() {
        assert_eq!(hint_parts("On intent-0007 (Does fsync(): wait for the disk?): check the mount"), (Some("i7".into()), "check the mount"));
        assert_eq!(hint_parts("On fact-0003 (plain words): look again"), (Some("f3".into()), "look again"));
        assert_eq!(hint_parts("On intent-0002 (cut short (…): still read"), (Some("i2".into()), "still read"));
        assert_eq!(hint_parts("just a note"), (None, "just a note"));
    }

    #[test]
    fn greyed_out_rows_sink_to_the_end_of_their_siblings_and_come_back() {
        let snapshot = snapshot();
        let now = Local::now();
        let built = |greyed: &HashSet<String>| {
            let mut tree = Tree::build(&snapshot, now);
            tree.sink(greyed);
            let mut folds = Folds::default();
            folds.update(&tree, &snapshot);
            let shown = rows(&tree, &folds);
            (tree, folds, shown)
        };
        let grey_ids = |tree: &Tree| -> Vec<String> {
            let mut out: Vec<String> = (0..tree.nodes.len()).filter(|&i| tree.grey[i]).map(|i| short(&tree.nodes[i].id)).collect();
            out.sort();
            out
        };
        let mut greyed = HashSet::new();
        let (tree, _, shown) = built(&greyed);
        assert_eq!(ids(&tree, &shown), ["board", "h1", "i8", "f3", "i9", "f4", "i7", "i4"]);

        // A lead at the root: to the bottom, with what is under it.
        toggle_grey(&tree, &mut greyed, tree.find("intent-0008").unwrap());
        let (tree, folds, shown) = built(&greyed);
        assert_eq!(ids(&tree, &shown), ["board", "h1", "i7", "i4", "i8", "f3", "i9", "f4"]);
        assert_eq!(grey_ids(&tree), ["f3", "f4", "i8", "i9"]);
        let line = row_line(&tree, &snapshot, &folds, &shown[4], false, 100, now);
        assert!(line.iter().any(|(style, t)| *style == Style::Work && t.starts_with("Race between")), "grey and italic");
        assert!(line.iter().all(|(style, t)| *style != Style::Warn || t.trim().is_empty()), "no colour left: {line:?}");
        let selected = row_line(&tree, &snapshot, &folds, &shown[4], true, 100, now);
        assert_eq!(selected[0], (Style::Cursor, " ❯ ".into()), "still where the chevron is");

        // Again: back, and back in its place.
        toggle_grey(&tree, &mut greyed, tree.find("intent-0008").unwrap());
        let (tree, _, shown) = built(&greyed);
        assert_eq!(ids(&tree, &shown), ["board", "h1", "i8", "f3", "i9", "f4", "i7", "i4"]);
        assert!(greyed.is_empty() && grey_ids(&tree).is_empty());

        // Deeper: to the end of its own siblings only.
        toggle_grey(&tree, &mut greyed, tree.find("fact-0003").unwrap());
        let (tree, _, shown) = built(&greyed);
        assert_eq!(ids(&tree, &shown), ["board", "h1", "i8", "f4", "f3", "i9", "i7", "i4"]);
        assert_eq!(grey_ids(&tree), ["f3", "i9"]);
        greyed.clear();

        // Brought back from deep in a greyed branch (a hint on i9): the branch
        // from the root down to it; its sibling branch stays grey.
        greyed.insert("intent-0008".into());
        let (tree_grey, _, _) = built(&greyed);
        revive(&tree_grey, &mut greyed, tree_grey.find("intent-0009").unwrap());
        let (tree, _, shown) = built(&greyed);
        assert_eq!(grey_ids(&tree), ["f4"]);
        assert_eq!(ids(&tree, &shown), ["board", "h1", "i8", "f3", "i9", "f4", "i7", "i4"]);

        // Brought back part way down: what is under it stays grey.
        greyed.clear();
        greyed.insert("intent-0008".into());
        let (tree, _, _) = built(&greyed);
        revive(&tree, &mut greyed, tree.find("fact-0003").unwrap());
        let (tree, _, shown) = built(&greyed);
        assert_eq!(grey_ids(&tree), ["f4", "i9"]);
        // Ctrl+C on it brings back what is under it too.
        greyed.clear();
        greyed.insert("intent-0008".into());
        let (tree, _, _) = built(&greyed);
        toggle_grey(&tree, &mut greyed, tree.find("fact-0003").unwrap());
        let (tree, _, _) = built(&greyed);
        assert_eq!(grey_ids(&tree), ["f4"]);
        assert_eq!(ids(&tree, &shown), ["board", "h1", "i8", "f3", "i9", "f4", "i7", "i4"]);

        // The board's own row is never grey.
        toggle_grey(&tree, &mut greyed, tree.find(BOARD_ID).unwrap());
        assert!(!greyed.contains(BOARD_ID));
    }

    #[test]
    fn the_board_reads_as_a_tree_of_leads_and_what_they_found() {
        // Claims are named by the subagent's tab: none here.
        let _guard = crate::worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        crate::worker_feed::clear_for_test();
        let snapshot = snapshot();
        let tree = Tree::build(&snapshot, Local::now());
        let mut folds = Folds::default();
        folds.update(&tree, &snapshot);
        let rows = rows(&tree, &folds);
        // Exploring first, then open, then ruled out (folded: its finding is
        // hidden); under the lead being explored, its findings oldest first,
        // and under the finding the lead it opened. A hint about nothing in
        // particular stands under the board's row.
        assert_eq!(ids(&tree, &rows), ["board", "h1", "i8", "f3", "i9", "f4", "i7", "i4"]);
        let rows = &rows[2..];
        assert_eq!(rows[1].guides, [true]);
        assert!(!rows[1].last && rows[3].last, "f3 has a sibling after it; f4 is the last");
        assert_eq!(rows[2].guides, [true, true]);

        let now = Local::now();
        let line = |index: usize, selected: bool| row_line(&tree, &snapshot, &folds, &rows[index], selected, 100, now);
        let explored = text(&line(0, true));
        assert!(explored.starts_with(" ❯ ▾ [*] Race between the upload and the cleanup fixture  p8 · b3-1 4m  i8"), "{explored}");
        assert_eq!(line(0, true)[0], (Style::Cursor, " ❯ ".into()));
        assert!(line(0, false).iter().any(|(style, t)| *style == Style::Warn && t == "[*] "));
        assert_eq!(text(&line(1, false)), "   ├─ ▾ [+] cleanup ran before the upload's fsync, 2 runs of 8  f3");
        assert_eq!(text(&line(2, false)), "   │  └─   [?] Hold the fixture until the handle closes  p6  i9");
        assert_eq!(text(&line(3, false)), "   └─   [-] no race with one CPU core  f4");
        let dead = line(5, false);
        assert!(text(&dead).starts_with("   ▸ [-] Pin the S3 client") && text(&dead).ends_with("  +1"), "{}", text(&dead));
        assert!(dead.iter().any(|(style, t)| *style == Style::Dim && t.starts_with("Pin the")), "ruled out reads quiet");
        let hint = row_line(&tree, &snapshot, &folds, &self::rows(&tree, &folds)[1], false, 100, now);
        assert_eq!(text(&hint), "   └─   [!] look at the fixture's teardown  unread  h1");

        // A row never runs past the width; the text gives way first.
        for width in [40, 60, 100] {
            for (index, _) in rows.iter().enumerate() {
                let drawn = text(&line_at(&tree, &snapshot, &folds, &rows[index], width));
                assert!(text_width(&drawn) <= width, "{width}: {drawn}");
            }
        }
    }

    fn line_at(tree: &Tree, snapshot: &Snapshot, folds: &Folds, row: &Row, width: usize) -> Line {
        row_line(tree, snapshot, folds, row, false, width, Local::now())
    }

    #[test]
    fn folds_are_kept_by_id_and_new_nodes_take_their_default() {
        let mut snapshot = snapshot();
        let tree = Tree::build(&snapshot, Local::now());
        let mut folds = Folds::default();
        folds.update(&tree, &snapshot);
        let i8 = tree.find("intent-0008").unwrap();
        folds.set(&tree, i8, false);
        assert_eq!(ids(&tree, &rows(&tree, &folds)), ["board", "h1", "i8", "i7", "i4"]);
        // A re-read with a new finding under it: still folded, the new one
        // counted in.
        snapshot.facts.insert(0, finding("fact-0005", "teardown deletes the upload dir", Polarity::Supports, Some("intent-0008"), &[], 1));
        let tree = Tree::build(&snapshot, Local::now());
        folds.update(&tree, &snapshot);
        let shown = rows(&tree, &folds);
        assert_eq!(ids(&tree, &shown), ["board", "h1", "i8", "i7", "i4"]);
        assert!(text(&line_at(&tree, &snapshot, &folds, &shown[2], 120)).ends_with("+4"));
        // Everything open, then everything folded.
        folds.toggle_all(&tree);
        assert_eq!(rows(&tree, &folds).len(), tree.nodes.len());
        folds.toggle_all(&tree);
        assert_eq!(ids(&tree, &rows(&tree, &folds)), ["board", "i8", "i7", "i4"]);
    }

    #[test]
    fn odd_records_still_make_a_tree() {
        let mut snapshot = snapshot();
        // Two parents: under the first, saying the other. A loop: broken. A
        // parent that isn't there: at the root.
        snapshot.intents.push(lead("intent-0010", "Both findings point here", IntentState::Open, 5, &["fact-0003", "fact-0004"], 2));
        snapshot.facts.push(finding("fact-0006", "points at a ghost", Polarity::Supports, Some("intent-0099"), &[], 1));
        snapshot.facts.push(finding("fact-0007", "loops one way", Polarity::Supports, None, &["fact-0008"], 1));
        snapshot.facts.push(finding("fact-0008", "loops the other", Polarity::Supports, None, &["fact-0007"], 1));
        let tree = Tree::build(&snapshot, Local::now());
        let both = tree.find("intent-0010").unwrap();
        assert_eq!(tree.nodes[tree.nodes[both].parent.unwrap()].id, "fact-0003");
        assert_eq!(tree.nodes[both].also, ["f4"]);
        let ghost = tree.find("fact-0006").unwrap();
        assert!(tree.nodes[ghost].parent.is_none() && tree.roots.contains(&ghost));
        let (seven, eight) = (tree.find("fact-0007").unwrap(), tree.find("fact-0008").unwrap());
        assert!(tree.nodes[seven].parent.is_none() || tree.nodes[eight].parent.is_none(), "the loop is cut somewhere");
        // Every node is reachable from the roots exactly once.
        let mut folds = Folds::default();
        folds.update(&tree, &snapshot);
        folds.toggle_all(&tree);
        if rows(&tree, &folds).len() != tree.nodes.len() {
            folds.toggle_all(&tree);
        }
        assert_eq!(rows(&tree, &folds).len(), tree.nodes.len());
    }

    #[test]
    fn the_head_and_the_strip() {
        // Claims are named by the subagent's tab: none here.
        let _guard = crate::worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        crate::worker_feed::clear_for_test();
        let snapshot = snapshot();
        let now = Local::now();
        let head = header(&snapshot, 100, now);
        let all: Vec<String> = head.iter().map(text).collect();
        assert_eq!(all[1], "  flaky-upload  [*] open  Find why test_upload flakes on CI");
        assert!(all[2].starts_with("  goal   test_upload passes"));
        assert!(all[3].starts_with("  test   for i in") && all[3].ends_with("  command"), "{}", all[3]);
        // 7 of 12: 7 cells of 12 filled, still green (under 60%).
        assert!(all[4].starts_with("  steps  ███████░░░░░  7 of 12   1 exploring   2 open   1 ruled out   3 findings"), "{}", all[4]);
        assert!(head[4].iter().any(|(style, t)| *style == Style::Success && t == "███████"));
        for width in [40, 72, 100] {
            for line in header(&snapshot, width, now) {
                assert!(text_width(&text(&line)) <= width, "{width}: {}", text(&line));
            }
        }
        let mut late = super::tests::snapshot();
        late.board.steps_used = 11;
        assert!(header(&late, 100, now)[4].iter().any(|(style, t)| *style == Style::Danger && t == "███████████"));

        let tree = Tree::build(&snapshot, now);
        let lead = detail(&tree, &snapshot, tree.find("intent-0008").unwrap(), 100, now);
        let lines: Vec<String> = lead.iter().map(text).collect();
        assert!(lines[0].starts_with("  [*] lead i8  exploring  · priority 8 ·"), "{lines:?}");
        assert_eq!(lines[1], "  Race between the upload and the cleanup fixture");
        assert!(lines.iter().any(|l| l.starts_with("  why       because Race")));
        assert!(lines.iter().any(|l| l == "  exploring b3-1 for 4m"), "{lines:?}");
        // The board's own row: what the head cuts, in full.
        let board = detail(&tree, &snapshot, tree.find(BOARD_ID).unwrap(), 60, now);
        let lines: Vec<String> = board.iter().map(text).collect();
        assert!(lines[0].starts_with("  [*] board flaky-upload  · open · opened"), "{lines:?}");
        assert!(lines.iter().any(|l| l.starts_with("  origin    1 in 8 on CI")), "{lines:?}");
        let test_rows = lines.iter().skip_while(|l| !l.starts_with("  test")).take_while(|l| !l.starts_with("  steps")).count();
        assert!(test_rows >= 2, "the test wraps whole rather than being cut: {lines:?}");
        assert!(lines.iter().all(|l| !l.contains('…')), "{lines:?}");
        assert!(lines.iter().any(|l| l == "  steps     7 of 12 used"), "{lines:?}");
        let row = Row { node: tree.find(BOARD_ID).unwrap(), guides: Vec::new(), last: false };
        let mut folds = Folds::default();
        folds.update(&tree, &snapshot);
        assert_eq!(text(&row_line(&tree, &snapshot, &folds, &row, false, 100, now)), "   ▾ [*] origin  1 in 8 on CI  full below");
        let fact = detail(&tree, &snapshot, tree.find("fact-0003").unwrap(), 100, now);
        let lines: Vec<String> = fact.iter().map(text).collect();
        assert!(lines[0].starts_with("  [+] finding f3  supports  · observed · by s0ma"), "{lines:?}");
        assert!(lines.iter().any(|l| l.starts_with("  from      i8")));
        assert!(lines.iter().any(|l| l.starts_with("  evidence  #1423 #1430")));
    }

    #[test]
    fn a_hint_says_what_it_is_about() {
        let snapshot = snapshot();
        let tree = Tree::build(&snapshot, Local::now());
        let lead = tree.find("intent-0007");
        assert_eq!(
            hint_about(&tree, &snapshot, lead, "check df on the runner"),
            "On intent-0007 (Does the CI runner's /tmp fill up mid-run?): check df on the runner"
        );
        assert_eq!(prompt_target(&tree, lead).as_deref(), Some("i7"));
        // Drawn back short: the text, and what it is about.
        let sent = hint_about(&tree, &snapshot, lead, "check df: on the runner");
        assert_eq!(hint_parts(&sent), (Some("i7".to_string()), "check df: on the runner"));
        assert_eq!(hint_parts("On reflection (maybe): no"), (None, "On reflection (maybe): no"));
        let mut with_hint = super::tests::snapshot();
        with_hint.hints[0].text = sent;
        let tree = Tree::build(&with_hint, Local::now());
        let mut folds = Folds::default();
        folds.update(&tree, &with_hint);
        let shown = rows(&tree, &folds);
        // It stands under i7, so it needn't say so.
        assert_eq!(ids(&tree, &shown), ["board", "i8", "f3", "i9", "f4", "i7", "h1", "i4"]);
        let row = shown.iter().find(|row| matches!(tree.nodes[row.node].kind, Kind::Hint(_))).unwrap();
        assert_eq!(text(&row_line(&tree, &with_hint, &folds, row, false, 120, Local::now())), "   └─   [!] check df: on the runner  unread  h1");
        let hint = tree.find("hint-0001");
        assert_eq!(tree.nodes[tree.nodes[hint.unwrap()].parent.unwrap()].id, "intent-0007");
        assert_eq!(hint_about(&tree, &snapshot, hint, "plain"), "plain");
        assert_eq!(prompt_target(&tree, hint), None);
        // About a lead no longer on the board: under the board's row, saying which.
        with_hint.hints[0].text = "On intent-0042 (gone): still worth a look".into();
        let tree = Tree::build(&with_hint, Local::now());
        let mut folds = Folds::default();
        folds.update(&tree, &with_hint);
        let shown = rows(&tree, &folds);
        assert_eq!(ids(&tree, &shown)[..2], ["board", "h1"]);
        assert_eq!(text(&row_line(&tree, &with_hint, &folds, &shown[1], false, 120, Local::now())), "   └─   [!] still worth a look  unread  on i42  h1");
        assert_eq!(short("intent-0010"), "i10");
        assert_eq!(short("fact-0000"), "f0");
        assert_eq!(short("custom"), "custom");
    }
}
