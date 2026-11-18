//! The audit tab's tree, a submodule of `tabs.rs` drawn by `audit_view`:
//! each claim once, its checks under it and each check's evidence under
//! that (mutants, the counterexample, the repro, what it says). In plan
//! order, or by file: the files in scope with what they define and have no
//! claim about, the calls out of the slice with their assumptions, and the
//! callers in. Activity (the log) and the runs come last.

use std::collections::BTreeSet;

use crate::audit::{self, Audit, AuditStore, Check, Claim, Level, LogEntry, Posted, Repro, Review, Run, RunCommand, Tone, Verdict};
use crate::style::{Line, Style};

/// Which claims the tree shows: the row of filters over it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(super) enum Filter {
    #[default]
    All,
    Waiting,
    Failing,
    Holds,
    Unchecked,
    Aside,
}

pub(super) const FILTERS: [Filter; 6] = [Filter::All, Filter::Waiting, Filter::Failing, Filter::Holds, Filter::Unchecked, Filter::Aside];

impl Filter {
    pub(super) fn name(self) -> &'static str {
        match self {
            Filter::All => "all",
            Filter::Waiting => "waiting on you",
            Filter::Failing => "failing",
            Filter::Holds => "holds",
            Filter::Unchecked => "unchecked",
            Filter::Aside => "set aside",
        }
    }

    pub(super) fn matches(self, claim: &Claim, audit: &Audit) -> bool {
        let aside = matches!(claim.review, Review::Rejected | Review::Dropped);
        match self {
            Filter::All => !aside,
            Filter::Waiting => !aside && claim.waits_on(audit),
            Filter::Failing => !aside && claim.verdict == Verdict::Fails && !claim.waived,
            Filter::Holds => !aside && claim.verdict == Verdict::Holds,
            Filter::Unchecked => !aside && claim.verdict == Verdict::Unchecked && claim.review != Review::Proposed,
            Filter::Aside => aside,
        }
    }
}

/// One piece of a check's evidence.
#[derive(Clone, Debug)]
pub(super) enum Proof {
    Mutant { name: String, caught: bool },
    Trace(String),
    Repro { test: String, repro: Repro, output: String },
    Says(String),
}

#[derive(Clone, Debug)]
pub(super) enum Node {
    File { path: String, weakest: Option<Level>, defines: Vec<String>, claims: usize },
    Group { title: &'static str, note: String },
    Claim(Box<Claim>),
    Check { claim: Box<Claim>, check: Box<Check>, rewriting: Option<String> },
    Writing { claim: String, slot: String, worker: Option<String>, why: Option<String> },
    Evidence { claim: String, slot: String, proof: Proof },
    Unclaimed { file: String, symbol: String },
    CallOut { symbol: String, file: String },
    Caller { text: String },
    Activity { count: usize, latest: String },
    Log(LogEntry),
    Runs { count: usize },
    Batch(Box<Run>),
    Command(RunCommand),
}

impl Node {
    /// The claim a row is about, if any.
    pub(super) fn claim_id(&self) -> Option<&str> {
        match self {
            Node::Claim(c) => Some(&c.id),
            Node::Check { claim, .. } => Some(&claim.id),
            Node::Writing { claim, .. } | Node::Evidence { claim, .. } => Some(claim),
            _ => None,
        }
    }

    /// The check a row is about (`k2`), if any.
    pub(super) fn slot(&self) -> Option<&str> {
        match self {
            Node::Check { check, .. } => Some(check.slot()),
            Node::Evidence { slot, .. } | Node::Writing { slot, .. } => Some(slot),
            _ => None,
        }
    }

    /// The file a row is about: its own, or its claim's.
    pub(super) fn file(&self) -> Option<String> {
        match self {
            Node::File { path, .. } | Node::Unclaimed { file: path, .. } => Some(path.clone()),
            Node::CallOut { file, .. } => Some(file.clone()),
            Node::Claim(c) => file_of(c),
            Node::Check { claim, .. } => file_of(claim),
            _ => None,
        }
    }
}

pub(super) struct Row {
    pub key: String,
    pub node: Node,
    /// The tree's guides before it: `│  ├─ `.
    pub guides: String,
    pub foldable: bool,
    pub open: bool,
}

/// What the tree's builder needs from the tab's state.
pub(super) struct Shape<'a> {
    pub filter: Filter,
    pub search: &'a str,
    pub by_file: bool,
    pub open: &'a BTreeSet<String>,
    pub closed: &'a BTreeSet<String>,
}

impl Shape<'_> {
    fn is_open(&self, key: &str, default: bool) -> bool {
        if self.closed.contains(key) {
            false
        } else {
            self.open.contains(key) || default
        }
    }
}

/// The file a claim is about: its source's path.
pub(super) fn file_of(claim: &Claim) -> Option<String> {
    let source = claim.source.trim();
    let path = source.split([':', ' ']).next().unwrap_or(source);
    (path.contains('.') || path.contains('/')).then(|| path.to_string())
}

/// A node before it is laid out: its key, itself, its children.
struct Item {
    key: String,
    node: Node,
    default_open: bool,
    children: Vec<Item>,
}

fn leaf(key: String, node: Node) -> Item {
    Item { key, node, default_open: false, children: Vec::new() }
}

/// The tree's rows, top to bottom, as folded.
pub(super) fn rows(store: &AuditStore, audit: &Audit, claims: &[Claim], shape: &Shape) -> Vec<Row> {
    let search = shape.search.to_lowercase();
    let shown = |c: &Claim| {
        shape.filter.matches(c, audit)
            && (search.is_empty() || c.text.to_lowercase().contains(&search) || c.source.to_lowercase().contains(&search) || c.id == search)
    };
    let writing = crate::audit::run::writing(&audit.name);
    let mut ordered: Vec<&Claim> = claims.iter().filter(|c| shown(c)).collect();
    ordered.sort_by_key(|c| (c.is_assumption(), crate::audit::run::plan_key(c)));
    let mut items: Vec<Item> = Vec::new();
    if shape.by_file {
        items.extend(by_file(store, audit, claims, &ordered, &writing));
    } else {
        items.extend(ordered.iter().map(|c| claim_item(store, audit, c, &writing)));
    }
    // Activity and the runs, last, folded.
    let log = store.log_entries(&audit.name).unwrap_or_default();
    if !log.is_empty() {
        let latest = log.last().map(|e| format!("{} {}", e.at.get(11..16).unwrap_or(""), e.what)).unwrap_or_default();
        let children = log.iter().rev().take(200).enumerate().map(|(i, e)| leaf(format!("log:{i}"), Node::Log(e.clone()))).collect();
        items.push(Item { key: "activity".into(), node: Node::Activity { count: log.len(), latest }, default_open: false, children });
    }
    let runs = store.runs(&audit.name).unwrap_or_default();
    if !runs.is_empty() {
        let children = runs
            .iter()
            .rev()
            .take(40)
            .map(|run| Item {
                key: format!("run:{}", run.n),
                node: Node::Batch(Box::new(run.clone())),
                default_open: false,
                children: run.commands.iter().enumerate().map(|(i, c)| leaf(format!("run:{}:{i}", run.n), Node::Command(c.clone()))).collect(),
            })
            .collect();
        items.push(Item { key: "runs".into(), node: Node::Runs { count: runs.len() }, default_open: false, children });
    }
    let mut out = Vec::new();
    lay_out(&items, shape, "", true, &mut out);
    out
}

fn lay_out(items: &[Item], shape: &Shape, prefix: &str, top: bool, out: &mut Vec<Row>) {
    for (i, item) in items.iter().enumerate() {
        let last = i + 1 == items.len();
        let foldable = !item.children.is_empty();
        let open = foldable && shape.is_open(&item.key, item.default_open);
        let guides = if top { String::new() } else { format!("{prefix}{}", if last { "└─ " } else { "├─ " }) };
        out.push(Row { key: item.key.clone(), node: item.node.clone(), guides, foldable, open });
        if open {
            let next = if top { "  ".to_string() } else { format!("{prefix}{}", if last { "   " } else { "│  " }) };
            lay_out(&item.children, shape, &next, false, out);
        }
    }
}

/// A claim with its checks under it, each with its evidence.
fn claim_item(store: &AuditStore, audit: &Audit, claim: &Claim, writing: &[(String, String, Option<String>)]) -> Item {
    let mut children = Vec::new();
    let checks = store.current_checks(&audit.name, &claim.id).unwrap_or_default();
    let held: Vec<Check> = store.checks(&audit.name, &claim.id).unwrap_or_default().into_iter().filter(|k| k.held_weaker && k.superseded_by.is_none()).collect();
    let evidence = store.evidence(&audit.name, Some(&claim.id)).unwrap_or_default();
    let mine: Vec<&(String, String, Option<String>)> = writing.iter().filter(|(c, _, _)| *c == claim.id).collect();
    let failing_slot = checks.iter().find(|k| k.verdict == Verdict::Fails).map(|k| k.slot().to_string());
    let mut check_items: Vec<Item> = Vec::new();
    for check in checks.iter().chain(held.iter()) {
        let slot = check.slot().to_string();
        let key = format!("k:{}/{slot}/{}", claim.id, check.version);
        let mut proofs = Vec::new();
        if check.verdict == Verdict::Fails
            && let Some(trace) = evidence.iter().rev().find(|e| e.kind == "trace" && (e.check == slot || e.check.is_empty()) && Some(&e.path) == check.trace.as_ref())
        {
            proofs.push(Proof::Trace(trace.result.clone()));
        }
        if failing_slot.as_deref() == Some(slot.as_str())
            && let Some(test) = &claim.repro_test
        {
            proofs.push(Proof::Repro { test: test.split(" :: ").next().unwrap_or(test).to_string(), repro: claim.repro, output: claim.repro_output.clone().unwrap_or_default() });
        }
        for name in &check.killed {
            proofs.push(Proof::Mutant { name: name.clone(), caught: true });
        }
        for name in &check.survived {
            proofs.push(Proof::Mutant { name: name.clone(), caught: false });
        }
        if let Some(says) = &check.back_translation {
            proofs.push(Proof::Says(says.clone()));
        }
        let rewriting = mine.iter().find(|(_, s, _)| *s == slot).map(|(_, _, w)| w.clone().unwrap_or_else(|| "a worker, starting".into()));
        let children = proofs
            .into_iter()
            .enumerate()
            .map(|(i, proof)| leaf(format!("{key}/{i}"), Node::Evidence { claim: claim.id.clone(), slot: slot.clone(), proof }))
            .collect();
        let failing = check.verdict == Verdict::Fails;
        check_items.push(Item {
            key,
            node: Node::Check { claim: Box::new(claim.clone()), check: Box::new(check.clone()), rewriting },
            default_open: failing,
            children,
        });
    }
    children.extend(check_items);
    // Checks being written that it doesn't have yet, or asked for.
    for (_, slot, worker) in mine.iter().filter(|(_, s, _)| !checks.iter().any(|k| k.slot() == s.as_str())) {
        children.push(leaf(format!("w:{}/{slot}", claim.id), Node::Writing { claim: claim.id.clone(), slot: slot.clone(), worker: worker.clone(), why: None }));
    }
    if let Some(why) = &claim.steer
        && mine.is_empty()
    {
        let slot = claim.steer_check.clone().unwrap_or_else(|| "k1".into());
        children.push(leaf(format!("w:{}/{slot}", claim.id), Node::Writing { claim: claim.id.clone(), slot, worker: None, why: Some(why.clone()) }));
    }
    let default_open = !children.is_empty();
    Item { key: format!("c:{}", claim.id), node: Node::Claim(Box::new(claim.clone())), default_open, children }
}


/// By file: each file in scope (and any other a claim names) with its
/// claims, then what it defines with no claim; the calls out of the slice
/// with their assumptions; the callers in.
fn by_file(store: &AuditStore, audit: &Audit, all: &[Claim], shown: &[&Claim], writing: &[(String, String, Option<String>)]) -> Vec<Item> {
    let slice = audit.slice.clone().unwrap_or_default();
    let mut files: Vec<String> = slice.inside_files.clone();
    for claim in shown.iter().filter(|c| !c.is_assumption()) {
        if let Some(file) = file_of(claim)
            && !files.contains(&file)
        {
            files.push(file);
        }
    }
    let mut out = Vec::new();
    for file in &files {
        let here: Vec<&&Claim> = shown.iter().filter(|c| !c.is_assumption() && file_of(c).as_deref() == Some(file.as_str())).collect();
        let defines = slice.by_file.get(file).cloned().unwrap_or_default();
        // Defined here, and named by no claim (its text, source or checks).
        let named = |symbol: &str| {
            all.iter().any(|c| {
                !matches!(c.review, Review::Rejected | Review::Dropped)
                    && (c.text.contains(symbol)
                        || c.source.contains(symbol)
                        || store.current_checks(&audit.name, &c.id).unwrap_or_default().iter().any(|k| k.targets.iter().any(|t| t.ends_with(&format!(":{symbol}")) || t.ends_with(&format!("::{symbol}")))))
            })
        };
        let unclaimed: Vec<String> = defines.iter().filter(|s| s.len() > 1 && !named(s)).cloned().collect();
        if here.is_empty() && unclaimed.is_empty() {
            continue;
        }
        let weakest = here.iter().map(|c| if c.verdict == Verdict::Holds { c.level } else { Level::Documented }).min();
        let mut children: Vec<Item> = here.iter().map(|c| claim_item(store, audit, c, writing)).collect();
        children.extend(unclaimed.iter().map(|symbol| leaf(format!("u:{file}:{symbol}"), Node::Unclaimed { file: file.clone(), symbol: symbol.clone() })));
        out.push(Item {
            key: format!("f:{file}"),
            node: Node::File { path: file.clone(), weakest: if here.is_empty() { None } else { weakest }, defines, claims: here.len() },
            default_open: true,
            children,
        });
    }
    let no_file: Vec<&&Claim> = shown.iter().filter(|c| !c.is_assumption() && file_of(c).is_none()).collect();
    if !no_file.is_empty() {
        out.push(Item {
            key: "f:".into(),
            node: Node::File { path: "(no file)".into(), weakest: None, defines: Vec::new(), claims: no_file.len() },
            default_open: true,
            children: no_file.iter().map(|c| claim_item(store, audit, c, writing)).collect(),
        });
    }
    // The boundary: calls out (with the assumption made about each), callers in.
    let assumptions: Vec<&&Claim> = shown.iter().filter(|c| c.is_assumption()).collect();
    let mut calls: Vec<Item> = assumptions.iter().map(|c| claim_item(store, audit, c, writing)).collect();
    for call in &slice.calls_out {
        let symbol = call.split(' ').next().unwrap_or(call).to_string();
        if all.iter().any(|c| c.is_assumption() && (c.source == format!("sym:{symbol}") || c.text.contains(&symbol))) {
            continue;
        }
        let file = call.split_once('(').map(|(_, f)| f.trim_end_matches(')').to_string()).unwrap_or_default();
        calls.push(leaf(format!("o:{symbol}"), Node::CallOut { symbol, file }));
    }
    if !calls.is_empty() {
        out.push(Item { key: "g:out".into(), node: Node::Group { title: "calls out of the slice", note: String::new() }, default_open: true, children: calls });
    }
    if !slice.called_from.is_empty() {
        let children = slice.called_from.iter().take(60).enumerate().map(|(i, c)| leaf(format!("i:{i}"), Node::Caller { text: c.replace(" <- ", "  ← ") })).collect();
        out.push(Item { key: "g:in".into(), node: Node::Group { title: "called from outside", note: String::new() }, default_open: false, children });
    }
    out
}

// ------------------------------------------------------------------ drawing

pub(super) fn tone(tone: Tone) -> Style {
    match tone {
        Tone::Plain => Style::Plain,
        Tone::Dim => Style::Dim,
        Tone::Warn => Style::Warn,
        Tone::Danger => Style::Danger,
        Tone::Success => Style::Success,
    }
}

/// A check's mark and how it reads.
fn check_mark(check: &Check) -> (&'static str, Style, String) {
    if check.held_weaker {
        return ("[!]", Style::Warn, "held: weaker".into());
    }
    match check.verdict {
        Verdict::Fails => ("[-]", Style::Danger, "fails".into()),
        Verdict::Holds => (
            "[+]",
            Style::Success,
            match check.kills {
                Some((k, t)) => format!("holds · {} · {k} of {t} mutants caught", check.level.label()),
                None => format!("holds · {}", check.level.label()),
            },
        ),
        Verdict::Vacuous => ("[~]", Style::Dim, "vacuous: no mutant caught".into()),
        Verdict::Error => ("[-]", Style::Dim, "error".into()),
        Verdict::Unchecked => ("[ ]", Style::Dim, "not run yet".into()),
    }
}

/// What a claim's row says on its right.
fn claim_right(claim: &Claim, audit: &Audit) -> (Style, String) {
    let mut parts: Vec<String> = Vec::new();
    if claim.pinned {
        parts.push("pinned".into());
    }
    if !claim.is_assumption() {
        parts.push(format!("p{}", claim.priority.max(1)));
    }
    // Not reviewed where the audit checks it anyway: said, not waited on.
    let unreviewed = claim.review == Review::Proposed;
    let (style, state) = if unreviewed && claim.waits_on(audit) && !claim.is_assumption() {
        (Style::Warn, "proposed · waits on you".to_string())
    } else if let Some(weaker) = &claim.weaker {
        (Style::Warn, format!("weaker check waits: {}", super::clip(weaker, 30)))
    } else if claim.repro == Repro::ModelOnly && !claim.model_only_accepted {
        (Style::Warn, "model-only: waits on you".into())
    } else if claim.posted == Posted::Ready {
        (Style::Warn, "result waits to be posted".into())
    } else if !claim.readings.is_empty() && claim.picked.is_none() {
        (Style::Warn, "question · waits on you".into())
    } else if claim.is_assumption() {
        (Style::Dim, if unreviewed { "assumption · unreviewed".into() } else { "assumption".into() })
    } else if unreviewed && claim.verdict == Verdict::Unchecked {
        (Style::Dim, "proposed".into())
    } else if unreviewed {
        // A result on a claim nobody reviewed: shown, counting for nothing.
        (Style::Dim, format!("{} · unreviewed", claim.verdict.label()))
    } else {
        match claim.verdict {
            Verdict::Fails if claim.waived => (Style::Dim, "fails · waived".into()),
            Verdict::Fails if claim.repro == Repro::Fails => (Style::Danger, "fails · reproduced".into()),
            Verdict::Fails => (Style::Danger, "fails".into()),
            Verdict::Holds => (Style::Success, format!("holds · {}", claim.level.label())),
            Verdict::Vacuous => (Style::Dim, "vacuous".into()),
            Verdict::Error => (Style::Dim, "error".into()),
            Verdict::Unchecked if claim.blocked.is_some() => (Style::Warn, format!("blocked: {}", claim.blocked.clone().unwrap_or_default())),
            Verdict::Unchecked if claim.skipped => (Style::Dim, "skipped".into()),
            Verdict::Unchecked if claim.taken => (Style::Dim, "yours to write".into()),
            Verdict::Unchecked if claim.new && audit.gone_at.is_some() => (Style::Dim, "new: waits for go".into()),
            Verdict::Unchecked => (Style::Dim, "no result yet".into()),
        }
    };
    if let Some(fact) = &claim.fact {
        parts.push(format!("{state} · {fact}"));
    } else {
        parts.push(state);
    }
    (style, parts.join(" · "))
}

/// One row, as drawn: cursor, guides, fold, mark, text and, right-aligned,
/// its state.
pub(super) fn row_line(row: &Row, audit: &Audit, selected: bool, width: usize) -> Line {
    let mut line: Line = vec![(if selected { Style::Cursor } else { Style::Plain }, if selected { "❯ ".into() } else { "  ".into() })];
    line.push((Style::Dim, row.guides.clone()));
    let fold = if row.foldable { if row.open { "▾ " } else { "▸ " } } else { "  " };
    line.push((Style::Plain, fold.into()));
    let label = |style: Style| if selected { Style::Bold } else { style };
    let (left, right): (Line, (Style, String)) = match &row.node {
        Node::File { path, weakest, defines, claims } => (
            vec![(label(Style::Bold), path.clone())],
            (
                Style::Dim,
                format!(
                    "{} claim{} · weakest {}{}",
                    claims,
                    if *claims == 1 { "" } else { "s" },
                    weakest.map_or("nothing checked", Level::label),
                    if defines.is_empty() { String::new() } else { format!(" · {}", defines.join(" · ")) }
                ),
            ),
        ),
        Node::Group { title, note } => (vec![(label(Style::Bold), title.to_string())], (Style::Dim, note.clone())),
        Node::Claim(claim) => {
            let (mark, t) = claim.mark();
            (vec![(tone(t), format!("{mark} ")), (label(Style::Bold), format!("{:<4}", claim.id)), (Style::Plain, format!(" {}", claim.text))], claim_right(claim, audit))
        }
        Node::Check { check, rewriting, .. } => {
            let (mark, style, said) = check_mark(check);
            let what = format!(" {}{}", check.kind, if check.run.is_empty() { String::new() } else { format!(" · {}", check.dir) });
            let right = match rewriting {
                Some(worker) => (Style::Warn, format!("{worker} rewriting it")),
                None => (style, said),
            };
            (vec![(style, format!("{mark} ")), (label(Style::Plain), check.slot().to_string()), (Style::Dim, what)], right)
        }
        Node::Writing { slot, worker, why, .. } => match worker {
            Some(worker) => (vec![(Style::Warn, "[*] ".into()), (label(Style::Plain), slot.clone()), (Style::Dim, " being written".into())], (Style::Warn, format!("{worker} writing it"))),
            None => (
                vec![(Style::Dim, "[ ] ".into()), (label(Style::Plain), slot.clone()), (Style::Dim, why.as_ref().map(|w| format!(" {w}")).unwrap_or_default())],
                (Style::Dim, "queued".into()),
            ),
        },
        Node::Evidence { proof, .. } => match proof {
            Proof::Mutant { name, caught } => (
                vec![(if *caught { Style::Success } else { Style::Danger }, if *caught { "[+] ".into() } else { "[-] ".into() }), (label(Style::Plain), "mutant ".into()), (Style::Plain, name.clone())],
                (Style::Dim, if *caught { "caught".into() } else { "survived: the check missed it".into() }),
            ),
            Proof::Trace(text) => (
                vec![(Style::Danger, "[-] ".into()), (label(Style::Plain), "counterexample ".into()), (Style::Plain, text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string())],
                (Style::Dim, String::new()),
            ),
            Proof::Repro { test, repro, .. } => {
                let (mark, style, said) = match repro {
                    Repro::Fails => ("[+]", Style::Success, "reproduced"),
                    Repro::Passes => ("[-]", Style::Dim, "passes: not a repro"),
                    Repro::Pending => ("[*]", Style::Warn, "waits to run"),
                    _ => ("[ ]", Style::Dim, ""),
                };
                (vec![(style, format!("{mark} ")), (label(Style::Plain), "repro ".into()), (Style::Plain, test.clone())], (Style::Dim, said.into()))
            }
            Proof::Says(text) => {
                let (read, says) = read_back(text);
                let (mark, style) = match read {
                    Some((true, _)) => ("[+] ", Style::Dim),
                    Some((false, _)) => ("[~] ", Style::Warn),
                    None => ("[~] ", Style::Dim),
                };
                let state = match read {
                    Some((false, _)) => (Style::Warn, "read in part".to_string()),
                    _ => (Style::Dim, String::new()),
                };
                (vec![(style, mark.into()), (label(Style::Plain), "says ".into()), (Style::Dim, says.replace('\n', " "))], state)
            }
        },
        Node::Unclaimed { symbol, .. } => (
            vec![(Style::Warn, "[!] ".into()), (label(Style::Bold), symbol.clone()), (Style::Warn, "  no claim".into())],
            (Style::Dim, "+ adds one of yours".into()),
        ),
        Node::CallOut { symbol, file } => (vec![(Style::Warn, "[!] ".into()), (label(Style::Bold), symbol.clone()), (Style::Dim, format!("  in {file}"))], (Style::Warn, "no assumption about it".into())),
        Node::Caller { text } => (vec![(label(Style::Plain), text.clone())], (Style::Dim, String::new())),
        Node::Activity { count, latest } => (vec![(label(Style::Status), "activity  ".into()), (Style::Dim, latest.clone())], (Style::Dim, format!("{count} entries"))),
        Node::Log(e) => {
            let who = match e.who.as_str() {
                "operator" => Style::Operator,
                "s0ma" => Style::Accent,
                _ => Style::Warn,
            };
            (vec![(Style::Dim, format!("{} ", e.at.get(11..16).unwrap_or(""))), (who, format!("{:<8} ", e.who)), (label(Style::Plain), e.what.clone())], (Style::Dim, String::new()))
        }
        Node::Runs { count } => (vec![(label(Style::Status), "runs".into())], (Style::Dim, format!("{count} batches"))),
        Node::Batch(run) => (
            vec![(label(Style::Plain), format!("batch {} · {} · {}", run.n, run.started_at.get(11..16).unwrap_or(""), run.claims.join(" ")))],
            (Style::Dim, format!("{} commands · {}", run.commands.len(), run.approval)),
        ),
        Node::Command(c) => {
            let look = match c.verdict.as_str() {
                "holds" | "killed" | "reproduced" => Style::Success,
                "fails" => Style::Danger,
                v if v.starts_with("error") => Style::Problem,
                _ => Style::Dim,
            };
            (vec![(label(Style::Plain), format!("{} {} ", c.claim, c.what)), (Style::Dim, c.command.clone())], (look, format!("exit {} · {}", c.exit.map_or("-".into(), |e| e.to_string()), c.verdict)))
        }
    };
    let used: usize = line.iter().map(|(_, t)| super::text_width(t)).sum();
    let right_text = if right.1.is_empty() { String::new() } else { format!("  {}", right.1) };
    let right_width = super::text_width(&right_text).min(width.saturating_sub(used + 12) / 2 + 12);
    let right_text = super::clip(&right_text, right_width);
    let room = width.saturating_sub(used + super::text_width(&right_text) + 1);
    let mut left_width = 0;
    for (style, text) in left {
        let piece = super::clip(&text, room.saturating_sub(left_width));
        left_width += super::text_width(&piece);
        if !piece.is_empty() {
            line.push((style, piece));
        }
    }
    line.push((Style::Plain, " ".repeat(room.saturating_sub(left_width))));
    line.push((right.0, right_text));
    line
}

// ------------------------------------------------------------------ the details

/// The selected row in full, for the strip under the tree.
pub(super) fn detail(row: &Row, audit: &Audit, store: &AuditStore) -> Vec<(Style, String)> {
    let mut out: Vec<(Style, String)> = Vec::new();
    let mut line = |style: Style, text: String| out.push((style, text));
    match &row.node {
        Node::Claim(claim) => {
            let (mark, t) = claim.mark();
            line(
                tone(t),
                format!(
                    "{}  {mark} {} · {} · p{} · {} · {}",
                    claim.id,
                    claim.verdict.label(),
                    claim.level.label(),
                    claim.priority.max(1),
                    format!("{:?}", claim.review).to_lowercase(),
                    if claim.author == "operator" { "his wording" } else { "s0ma's wording" }
                ),
            );
            line(Style::Plain, format!("claim    {}", claim.text));
            let checks = store.current_checks(&audit.name, &claim.id).unwrap_or_default();
            let clauses = audit::clauses(&claim.text);
            if clauses.len() > 1 || !checks.is_empty() {
                for (i, clause) in clauses.iter().enumerate() {
                    let k = i + 1;
                    let by: Vec<String> = checks
                        .iter()
                        .filter(|c| c.anchors.contains(&k) || (clauses.len() == 1))
                        .map(|c| format!("{} {}", check_mark(c).0, c.slot()))
                        .collect();
                    line(Style::Plain, format!("{}[{k}] {clause}{}", if i == 0 { "clauses  " } else { "         " }, if by.is_empty() { String::new() } else { format!("    {}", by.join(" · ")) }));
                }
            }
            if !claim.source.is_empty() {
                line(Style::Plain, format!("source   {}", claim.source));
            }
            if !claim.why.is_empty() {
                line(Style::Plain, format!("why      {}", claim.why));
            }
            for (i, reading) in claim.readings.iter().enumerate() {
                let picked = claim.picked == Some(i);
                line(Style::Plain, format!("{}  [{}]{} {reading}", if i == 0 { "question" } else { "        " }, i + 1, if picked { " picked" } else { "" }));
            }
            if let Some(test) = &claim.repro_test {
                let said = match claim.repro {
                    Repro::Fails => format!("fails: {}", crate::util::clip(claim.repro_output.as_deref().unwrap_or("").lines().last().unwrap_or(""), 80)),
                    Repro::Passes => "passes: not a repro".into(),
                    Repro::Pending => "waits to run".into(),
                    _ => String::new(),
                };
                line(Style::Plain, format!("repro    {}   {said}", test.split(" :: ").next().unwrap_or(test)));
            }
            if let Some(why) = &claim.model_only_why {
                line(Style::Plain, format!("model    only: {why}{}", if claim.model_only_accepted { " (accepted)" } else { "" }));
            }
            if claim.is_assumption() {
                line(Style::Plain, format!("status   {}", crate::audit::scope::assumption_status(store, claim)));
            }
            if !claim.rests_on.is_empty() {
                line(Style::Plain, format!("assumes  {}", claim.rests_on.join(", ")));
            }
            if let Some(fact) = &claim.fact {
                line(Style::Plain, format!("board    {fact}{}", claim.lead.as_ref().map(|(b, l)| format!(" · under {b} {l}")).unwrap_or_default()));
            } else if let Some((board, lead)) = &claim.lead {
                line(Style::Plain, format!("board    {board} {lead}"));
            }
            if let Some(stale) = &claim.stale {
                line(Style::Warn, format!("stale    {stale}"));
            }
            if let Some(unsound) = &claim.unsound {
                line(Style::Warn, format!("unsound  {unsound}"));
            }
            if !claim.cited.is_empty() {
                line(Style::Plain, format!("cited    {}", claim.cited.join(", ")));
            }
        }
        Node::Check { claim, check, rewriting } => {
            let (mark, style, said) = check_mark(check);
            line(style, format!("{} · {}  {} {mark} {said}", claim.id, check.slot(), check.kind));
            line(Style::Plain, format!("run      {}", check.run));
            line(Style::Plain, format!("files    {} · v{} by {}{}", check.dir, check.version, check.by, check.checked_at.as_ref().map(|t| format!(" · ran {}", t.get(11..16).unwrap_or(""))).unwrap_or_default()));
            // Its last run: the exit and what it printed, an error's why.
            if check.verdict != Verdict::Unchecked
                && let Some((n, ran)) = last_run(store, &audit.name, &claim.id, check.slot())
            {
                let look = if check.verdict == Verdict::Error { Style::Problem } else { Style::Plain };
                line(look, format!("exit     {} · batch {n} · {} ms", ran.exit.map_or("none".into(), |e| e.to_string()), ran.ms));
                output_lines(&ran.output, &mut line);
            }
            if !check.targets.is_empty() {
                line(Style::Plain, format!("targets  {}", check.targets.join(", ")));
            }
            if !check.anchors.is_empty() {
                let clauses = audit::clauses(&claim.text);
                let anchored: Vec<String> = check.anchors.iter().map(|k| format!("[{k}] {}", clauses.get(k - 1).cloned().unwrap_or_default())).collect();
                line(Style::Plain, format!("clauses  {}", anchored.join(" · ")));
            }
            if let Some((k, t)) = check.kills {
                line(Style::Plain, format!("mutants  {k} of {t} caught{}", if check.survived.is_empty() { String::new() } else { format!(" · survived: {}", check.survived.join(", ")) }));
            }
            if let Some(text) = &check.back_translation {
                let (read, says) = read_back(text);
                line(Style::Plain, format!("says     {}", says.replace('\n', " ")));
                if let Some((whole, note)) = read {
                    line(if whole { Style::Dim } else { Style::Warn }, format!("read     {note}"));
                }
            }
            if !check.by_job.is_empty() || !check.back_by.is_empty() {
                let job = |id: &str| if id.is_empty() { "not recorded".to_string() } else { id.to_string() };
                line(Style::Dim, format!("jobs     written by {} · read back by {}", job(&check.by_job), job(&check.back_by)));
            }
            if let Some(worker) = rewriting {
                line(Style::Warn, format!("now      {worker} is writing it again"));
            }
        }
        Node::Writing { claim, slot, worker, why } => {
            line(Style::Warn, format!("{claim} · {slot}  {}", worker.as_ref().map_or("queued".to_string(), |w| format!("{w} writing"))));
            if let Some(why) = why {
                line(Style::Plain, format!("asked    {why}"));
            }
        }
        Node::Evidence { claim, slot, proof } => match proof {
            Proof::Mutant { name, caught } => {
                line(if *caught { Style::Success } else { Style::Danger }, format!("{claim} · {slot}  mutant {name} {}", if *caught { "caught" } else { "survived" }));
                line(Style::Plain, format!("patch    checks of {claim}/{slot}/mutants/{name}"));
            }
            Proof::Trace(text) => {
                line(Style::Danger, format!("{claim} · {slot}  counterexample"));
                for l in text.lines().filter(|l| !l.trim().is_empty()).take(12) {
                    line(Style::Plain, format!("  {l}"));
                }
            }
            Proof::Repro { test, repro, output } => {
                line(Style::Plain, format!("{claim}  repro {test} · {:?}", repro).to_lowercase());
                for l in output.lines().filter(|l| !l.trim().is_empty()).take(10) {
                    line(Style::Plain, format!("  {l}"));
                }
            }
            Proof::Says(text) => {
                let (read, says) = read_back(text);
                line(Style::Plain, format!("{claim} · {slot}  read-back"));
                line(Style::Plain, says);
                if let Some((whole, note)) = read {
                    line(if whole { Style::Dim } else { Style::Warn }, note);
                }
            }
        },
        Node::File { path, weakest, defines, claims } => {
            line(Style::Bold, format!("{path}  {claims} claim{} · weakest {}", if *claims == 1 { "" } else { "s" }, weakest.map_or("nothing checked", Level::label)));
            if !defines.is_empty() {
                line(Style::Plain, format!("defines  {}", defines.join(" · ")));
            }
        }
        Node::Unclaimed { file, symbol } => {
            line(Style::Warn, format!("{symbol}  {file} · no claim"));
            if let Some(slice) = &audit.slice {
                let callers: Vec<&String> = slice.called_from.iter().filter(|c| c.starts_with(&format!("{symbol} "))).collect();
                if !callers.is_empty() {
                    line(Style::Plain, format!("callers  {}", callers.iter().map(|c| c.split(" <- ").nth(1).unwrap_or(c)).collect::<Vec<_>>().join(", ")));
                }
            }
        }
        Node::CallOut { symbol, file } => {
            line(Style::Warn, format!("{symbol}  {file} · called out"));
            line(Style::Plain, "no assumption recorded".into());
        }
        Node::Group { title, note } => line(Style::Bold, if note.is_empty() { title.to_string() } else { format!("{title} · {note}") }),
        Node::Caller { text } => line(Style::Plain, text.clone()),
        Node::Activity { count, .. } => line(Style::Plain, format!("{count} entries")),
        Node::Log(e) => {
            line(Style::Plain, format!("{} {}  {}", e.at.get(0..16).unwrap_or(""), e.who, e.what));
            if !e.why.is_empty() {
                line(Style::Plain, e.why.clone());
            }
        }
        Node::Runs { count } => line(Style::Plain, format!("{count} batches · /share/audits/{}/runs/", audit.name)),
        Node::Batch(run) => {
            line(Style::Plain, format!("batch {} · {} → {} · {}", run.n, run.started_at.get(0..16).unwrap_or(""), run.finished_at.as_deref().and_then(|t| t.get(11..16)).unwrap_or("running"), run.approval));
            if !run.note.is_empty() {
                line(if run.note.starts_with("stopped") { Style::Problem } else { Style::Plain }, format!("note     {}", run.note));
            }
            if !run.transcript.is_empty() {
                line(Style::Plain, format!("log      {}", run.transcript));
            }
        }
        Node::Command(c) => {
            line(Style::Plain, format!("{} {} · exit {} · {} · {} ms", c.claim, c.what, c.exit.map_or("-".into(), |e| e.to_string()), c.verdict, c.ms));
            line(Style::Plain, format!("$ {}", c.command));
            output_lines(&c.output, &mut line);
        }
    }
    out
}

/// The last time `claim`'s check `slot` ran: its batch and its command.
fn last_run(store: &AuditStore, audit: &str, claim: &str, slot: &str) -> Option<(i64, crate::audit::RunCommand)> {
    let what = format!("check {slot}");
    store.runs(audit).ok()?.into_iter().rev().find_map(|run| {
        let n = run.n;
        run.commands.into_iter().rev().find(|c| c.claim == claim && (c.what == what || (slot == "k1" && c.what == "check"))).map(|c| (n, c))
    })
}

/// A command's output in the strip: its last 30 lines, as printed.
fn output_lines(output: &str, line: &mut impl FnMut(Style, String)) {
    let lines: Vec<&str> = output.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        line(Style::Dim, "output   none".into());
        return;
    }
    let skip = lines.len().saturating_sub(30);
    if skip > 0 {
        line(Style::Dim, format!("output   … {skip} lines above (the run's log has them)"));
    } else {
        line(Style::Dim, "output".into());
    }
    for text in &lines[skip..] {
        line(Style::Plain, format!("  {text}"));
    }
}

/// A back-translation as kept: whether its reader read every file whole
/// (and the harness's line saying so), and its sentences.
pub(super) fn read_back(text: &str) -> (Option<(bool, String)>, String) {
    let text = text.trim();
    if let Some(rest) = text.strip_prefix("[~] partial review: ") {
        let (note, says) = rest.split_once('\n').unwrap_or((rest, ""));
        return (Some((false, format!("[~] in part: {note}"))), says.trim().to_string());
    }
    if let Some((says, last)) = text.rsplit_once('\n')
        && last.starts_with("[+] read whole")
    {
        return (Some((true, last.to_string())), says.trim().to_string());
    }
    (None, text.to_string())
}
