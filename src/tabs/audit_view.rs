//! The audit tab (Alt+U), a submodule of `tabs.rs`: every audit (attached
//! to projects, and standing), one at a time with Tab, Ctrl+N and Ctrl+P;
//! its head (status, scope, levels, counts, settings), its sections (plan,
//! claims, map, findings, log, runs), the strip for the selected claim, and
//! every key also a footer button and a right-click menu item.
//!
//! The operator acts here; the harness decides. Nothing in this tab writes
//! a verdict: it accepts, rejects, rewords, takes, waives, posts, pauses,
//! changes settings and starts, and the harness checks (`audit/run.rs`).

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::audit::{self, Audit, Claim, Level, Posted, Repro, Review, Status, Tone, Verdict, Who};
use crate::style::{Line, Style};

use super::gui_view::{self, Click, Pick};
use super::{Footer, Open, Outcome};

/// The sections, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(super) enum Section {
    Plan,
    #[default]
    Claims,
    Map,
    Findings,
    Log,
    Runs,
}

const SECTIONS: [Section; 6] = [Section::Plan, Section::Claims, Section::Map, Section::Findings, Section::Log, Section::Runs];

impl Section {
    fn name(self) -> &'static str {
        match self {
            Section::Plan => "plan",
            Section::Claims => "claims",
            Section::Map => "map",
            Section::Findings => "findings",
            Section::Log => "log",
            Section::Runs => "runs",
        }
    }
}

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
    /// The claims section's filter.
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
    pub section: Section,
    pub selected: usize,
    pub top: usize,
    pub filter: String,
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
}

/// Everything one frame shows, read from the store.
struct Read {
    audits: Vec<Audit>,
    audit: Option<Audit>,
    claims: Vec<Claim>,
}

fn read(state: &State) -> Read {
    let Some(store) = audit::store() else { return Read { audits: Vec::new(), audit: None, claims: Vec::new() } };
    let audits: Vec<Audit> = store.audits().unwrap_or_default().into_iter().filter(|a| a.status.live() || state.audit.as_deref() == Some(a.name.as_str())).collect();
    let audit = state.audit.as_ref().and_then(|name| audits.iter().find(|a| &a.name == name).cloned());
    let claims = audit.as_ref().map(|a| store.claims(&a.name).unwrap_or_default()).unwrap_or_default();
    Read { audits, audit, claims }
}

/// The rows a section lists, as claim ids (or indexes, for log and runs).
fn section_rows(state: &State, audit: &Audit, claims: &[Claim]) -> Vec<Claim> {
    let filter = state.filter.to_lowercase();
    let wanted = |c: &&Claim| {
        filter.is_empty()
            || c.text.to_lowercase().contains(&filter)
            || c.source.to_lowercase().contains(&filter)
            || c.id == filter
            || c.mark().0 == filter
    };
    match state.section {
        Section::Plan => {
            let mut plan: Vec<Claim> =
                claims.iter().filter(|c| !matches!(c.review, Review::Rejected | Review::Dropped) && !c.is_assumption()).cloned().collect();
            plan.sort_by_key(crate::audit::run::plan_key);
            plan
        }
        Section::Claims => {
            let mut all: Vec<Claim> = claims.iter().filter(wanted).cloned().collect();
            all.sort_by_key(Claim::rank);
            all
        }
        Section::Map => {
            let mut all: Vec<Claim> = claims.to_vec();
            all.sort_by_key(|c| (file_of(c), c.id.clone()));
            all
        }
        Section::Findings => claims.iter().filter(|c| c.repro == Repro::Fails || c.posted == Posted::Ready || c.verdict == Verdict::Fails).cloned().collect(),
        Section::Log | Section::Runs => {
            let _ = audit;
            Vec::new()
        }
    }
}

/// The file a claim is about, for the map: its source's path, or its check's
/// first target.
fn file_of(claim: &Claim) -> String {
    let source = claim.source.trim();
    let path = source.split([':', ' ']).next().unwrap_or(source);
    if path.contains('.') || path.contains('/') { path.to_string() } else { "(no file)".into() }
}

fn tone(tone: Tone) -> Style {
    match tone {
        Tone::Plain => Style::Plain,
        Tone::Dim => Style::Dim,
        Tone::Warn => Style::Warn,
        Tone::Danger => Style::Danger,
        Tone::Success => Style::Success,
    }
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

/// The claims line: counts by state, `[!]` first.
fn counts_line(claims: &[Claim]) -> Line {
    let n = |f: &dyn Fn(&Claim) -> bool| claims.iter().filter(|c| f(c)).count();
    let mut out: Line = vec![(Style::Status, "  claims  ".into()), (Style::Plain, format!("{:<5}", claims.len()))];
    let groups: [(&str, &str, Style, usize); 6] = [
        ("[!]", "wanting you", Style::Warn, n(&|c| c.mark().0 == "[!]")),
        ("[?]", "not reviewed", Style::Plain, n(&|c| c.review == Review::Proposed)),
        ("[-]", "failing", Style::Danger, n(&|c| c.verdict == Verdict::Fails && !c.waived)),
        ("[*]", "checking", Style::Warn, n(&|c| c.running || c.held_by.is_some())),
        ("[+]", "hold", Style::Success, n(&|c| c.verdict == Verdict::Holds)),
        ("[~]", "vacuous", Style::Dim, n(&|c| c.verdict == Verdict::Vacuous)),
    ];
    for (mark, word, style, count) in groups {
        if count > 0 {
            out.push((style, format!(" {mark}")));
            out.push((Style::Dim, format!(" {count} {word}  ")));
        }
    }
    out
}

/// One claim's row in a list.
fn claim_row(claim: &Claim, audit: &Audit, selected: bool, width: usize, plan: bool) -> Line {
    let (mark, t) = claim.mark();
    let lead = if selected { (Style::Cursor, "❯ ".to_string()) } else { (Style::Plain, "  ".to_string()) };
    let mut line: Line = vec![lead, (tone(t), format!("{mark:<4}")), (if selected { Style::Bold } else { Style::Plain }, format!("{:<5}", claim.id))];
    let right = if plan {
        let mut parts = Vec::new();
        if claim.pinned {
            parts.push("! pinned".to_string());
        }
        parts.push(format!("p{}", claim.priority.max(1)));
        parts.push(claim.depth.map_or_else(|| if claim.tool.is_empty() { "—".into() } else { claim.tool.clone() }, |d| d.label().to_string()));
        if claim.estimate_secs > 0 {
            parts.push(format!("~{}", audit::show_secs(claim.estimate_secs)));
        }
        if claim.skipped {
            parts.push("skipped".into());
        }
        if claim.taken {
            parts.push("yours".into());
        }
        if claim.new && audit.gone_at.is_some() {
            parts.push("new".into());
        }
        parts.join(" · ")
    } else {
        let mut parts = Vec::new();
        if let Some(held) = &claim.held_by {
            parts.push(held.clone());
        }
        if claim.running {
            parts.push("running".into());
        }
        if claim.verdict != Verdict::Unchecked {
            parts.push(if claim.verdict == Verdict::Holds { claim.level.label().to_string() } else { claim.verdict.label().to_string() });
        } else if !claim.tool.is_empty() {
            parts.push(claim.tool.clone());
        }
        if claim.repro == Repro::Fails {
            parts.push("reproduced".into());
        }
        if let Some(fact) = &claim.fact {
            parts.push(format!("on {fact}"));
        }
        if claim.posted == Posted::Ready {
            parts.push("ready".into());
        }
        if let Some(weaker) = &claim.weaker {
            parts.push(weaker.clone());
        }
        if claim.repro == Repro::ModelOnly && !claim.model_only_accepted {
            parts.push("model-only".into());
        }
        if claim.waived {
            parts.push("waived".into());
        }
        if let Some(stale) = &claim.stale {
            parts.push(format!("stale: {}", crate::util::clip(stale, 30)));
        }
        if let Some(unsound) = &claim.unsound {
            parts.push(format!("unsound: {unsound}"));
        }
        if claim.taken {
            parts.push("yours".into());
        }
        if !claim.readings.is_empty() && claim.picked.is_none() {
            parts.push(format!("? {} readings", claim.readings.len()));
        }
        parts.join(" · ")
    };
    let right_width = super::text_width(&right);
    let room = width.saturating_sub(2 + 4 + 5 + right_width + 3);
    let text = super::clip(&claim.text, room);
    let pad = room.saturating_sub(super::text_width(&text));
    line.push((if claim.review == Review::Rejected { Style::Dim } else { Style::Plain }, text));
    line.push((Style::Plain, " ".repeat(pad + 2)));
    line.push((tone(t).clone_or_dim(), right));
    line
}

trait DimRight {
    fn clone_or_dim(self) -> Style;
}
impl DimRight for Style {
    fn clone_or_dim(self) -> Style {
        match self {
            Style::Danger | Style::Warn | Style::Success => self,
            _ => Style::Dim,
        }
    }
}

/// The strip: the selected claim in short.
fn strip(claim: &Claim, audit: &Audit) -> Vec<String> {
    let store = audit::store();
    let mut lines = Vec::new();
    lines.push(format!(
        "{}   {} {} · {}{}",
        claim.id,
        claim.mark().0,
        claim.verdict.label(),
        claim.level.label(),
        if claim.author == "operator" { "  · his wording" } else { "  · s0ma's wording" }
    ));
    lines.push(format!("claim    {}", claim.text));
    if !claim.source.is_empty() {
        lines.push(format!("source   {}", claim.source));
    }
    if !claim.why.is_empty() {
        lines.push(format!("why      {}", claim.why));
    }
    for (i, reading) in claim.readings.iter().enumerate() {
        let picked = claim.picked == Some(i);
        lines.push(format!("{}  {}  {reading}", if i == 0 { "question" } else { "        " }, if picked { format!("[{}] picked", i + 1) } else { format!("[{}] ", i + 1) }));
    }
    if let Some(store) = &store {
        if let Ok(Some(check)) = store.current_check(&audit.name, &claim.id) {
            lines.push(format!(
                "check    {} · {} · {}{}",
                check.kind,
                check.run,
                check.kills.map(|(k, t)| format!("{k}/{t} mutants killed")).unwrap_or_else(|| "not vetted yet".into()),
                if check.survived.is_empty() { String::new() } else { format!(" · survived: {}", check.survived.join(", ")) }
            ));
            if let Some(says) = &check.back_translation {
                lines.push(format!("says     (back-translation) {says}"));
            }
        }
        if claim.is_assumption() {
            lines.push(format!("status   {}", crate::audit::scope::assumption_status(store, claim)));
        }
    }
    if let Some(test) = &claim.repro_test {
        lines.push(format!(
            "repro    {}   {}",
            test.split(" :: ").next().unwrap_or(test),
            match claim.repro {
                Repro::Fails => format!("fails: {}", crate::util::clip(claim.repro_output.as_deref().unwrap_or("").lines().last().unwrap_or(""), 80)),
                Repro::Passes => "passes on the audited code: not a repro".into(),
                Repro::Pending => "waiting to run".into(),
                _ => String::new(),
            }
        ));
    }
    if let Some(why) = &claim.model_only_why {
        lines.push(format!("model    only: {why}{}", if claim.model_only_accepted { " (accepted)" } else { "" }));
    }
    if !claim.rests_on.is_empty() {
        lines.push(format!("assumes  {}", claim.rests_on.join(", ")));
    }
    if let Some(fact) = &claim.fact {
        lines.push(format!("board    {fact}{}", claim.lead.as_ref().map(|(b, l)| format!(" · under {b} {l}")).unwrap_or_default()));
    } else if let Some((board, lead)) = &claim.lead {
        lines.push(format!("board    results land under {board} {lead}"));
    }
    if !claim.cited.is_empty() {
        lines.push(format!("cited    {}", claim.cited.join(", ")));
    }
    lines
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
        tab.audit.section = Section::Claims;
        tab.audit.selected = 0;
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
    tab.audit.input.is_some() || tab.audit.view.is_some()
}

pub(super) fn body(tab: &mut Open, width: usize, rows_left: usize) -> (Vec<Line>, Footer) {
    if tab.view == super::View::Audit {
        tab.audit.scope = None;
    }
    if tab.audit.view.is_some() {
        return super::claim_view::body(tab, width, rows_left);
    }
    let data = read(&tab.audit);
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
        head.push((Style::Warn, "   [*] a batch is running".into()));
    }
    if audit.paused {
        head.push((Style::Warn, "   paused (z resumes)".into()));
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
    out.push(counts_line(&data.claims));
    // The settings row, changed in place.
    let mut settings: Line = vec![(Style::Status, "  set     ".into())];
    let row = super::clip(&audit.settings.row(), width.saturating_sub(24));
    settings.push((Style::Dim, row));
    settings.push((Style::Plain, "  ".into()));
    settings.push(gui_view::button("‹ change ›", Click::Keys(vec![KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE)])));
    out.push(settings);
    // A proposal: what the agent asks to start.
    if audit.status == Status::Proposed {
        out.push(Vec::new());
        out.push(vec![(Style::Warn, "  [!] audit proposed".into()), (Style::Dim, " · a twice starts it · e changes its settings first · x twice declines".into())]);
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
        out.push(vec![(Style::Warn, "  [!] s0ma proposes a re-run".into()), (Style::Dim, format!(": {} · r starts it", super::clip(why, width.saturating_sub(40))))]);
    }
    if let Some((board, why)) = &audit.attach_proposed {
        out.push(vec![(Style::Warn, format!("  [!] s0ma proposes attaching it to {board}")), (Style::Dim, format!(": {} · a twice attaches", super::clip(why, width.saturating_sub(50))))]);
    }
    // Sections.
    let mut bar: Line = vec![(Style::Dim, " ─ ".into())];
    for (i, section) in SECTIONS.iter().enumerate() {
        if i > 0 {
            bar.push((Style::Dim, " · ".into()));
        }
        let look = if *section == state.section { Style::Tab } else { Style::Plain };
        bar.push(gui_view::hot(look, format!(" {} ", section.name()), Click::Pick(Pick::AuditSection(i))));
    }
    if !state.filter.is_empty() && state.section == Section::Claims {
        bar.push((Style::Dim, format!("   filter: {}", state.filter)));
    }
    let used = bar.iter().map(|(_, t)| super::text_width(t)).sum::<usize>();
    bar.push((Style::Dim, format!(" {}", "─".repeat(width.saturating_sub(used + 2)))));
    out.push(bar);
    // The list.
    let list_room = rows_left.saturating_sub(out.len() + 10).max(3);
    let mut strip_lines: Vec<String> = Vec::new();
    match state.section {
        Section::Log | Section::Runs => {
            let store = audit::store();
            let lines: Vec<(Style, String)> = match (state.section, &store) {
                (Section::Log, Some(store)) => store
                    .log_entries(&audit.name)
                    .unwrap_or_default()
                    .iter()
                    .rev()
                    .map(|e| {
                        let who = match e.who.as_str() {
                            "operator" => Style::Operator,
                            "s0ma" => Style::Accent,
                            _ => Style::Warn,
                        };
                        (who, format!("{} {:<8} {}{}", e.at.get(11..16).unwrap_or(""), e.who, e.what, if e.why.is_empty() { String::new() } else { format!(" — {}", e.why) }))
                    })
                    .collect(),
                (Section::Runs, Some(store)) => store
                    .runs(&audit.name)
                    .unwrap_or_default()
                    .iter()
                    .rev()
                    .flat_map(|run| {
                        let mut rows = vec![(Style::Bold, format!("batch {} · {} · {} · {}", run.n, run.started_at.get(11..16).unwrap_or(""), run.claims.join(" "), run.approval))];
                        for command in &run.commands {
                            let look = match command.verdict.as_str() {
                                "holds" | "killed" | "reproduced" => Style::Success,
                                "fails" => Style::Danger,
                                v if v.starts_with("error") => Style::Problem,
                                _ => Style::Dim,
                            };
                            rows.push((look, format!("  {} {:<10} exit {:<4} {:<11} {}", command.claim, command.what, command.exit.map_or("-".into(), |e| e.to_string()), command.verdict, command.command)));
                        }
                        rows
                    })
                    .collect(),
                _ => Vec::new(),
            };
            state.selected = state.selected.min(lines.len().saturating_sub(1));
            if state.selected < state.top {
                state.top = state.selected;
            }
            if state.selected >= state.top + list_room {
                state.top = state.selected + 1 - list_room;
            }
            if lines.is_empty() {
                out.push(vec![(Style::Dim, format!("  nothing in the {} yet", state.section.name()))]);
            }
            for (i, (style, text)) in lines.iter().enumerate().skip(state.top).take(list_room) {
                let here = i == state.selected;
                let row = vec![(if here { Style::Cursor } else { Style::Plain }, if here { "❯ ".into() } else { "  ".into() }), (*style, super::clip(text, width.saturating_sub(4)))];
                out.push(gui_view::pick_row(row, Click::Pick(Pick::AuditRow(i))));
            }
            if let Some((_, text)) = lines.get(state.selected) {
                strip_lines.push(text.clone());
            }
        }
        _ => {
            let rows = section_rows(state, &audit, &data.claims);
            state.selected = state.selected.min(rows.len().saturating_sub(1));
            if state.selected < state.top {
                state.top = state.selected;
            }
            if state.selected >= state.top + list_room {
                state.top = state.selected + 1 - list_room;
            }
            if rows.is_empty() {
                let hint = match state.section {
                    Section::Plan => "no plan yet: s0ma proposes claims once the audit is set up (harvest). + adds one of yours.",
                    Section::Findings => "no findings: a failure is a finding only once its repro fails on the real code.",
                    _ if !state.filter.is_empty() => "nothing matches the filter (/ to change it)",
                    _ => "no claims yet: s0ma proposes them as it harvests. + adds one of yours.",
                };
                out.extend(super::flow(&[(Style::Dim, format!("  {hint}"))], width, 2));
            }
            if state.section == Section::Plan {
                let gone = audit.gone_at.is_some();
                out.push(vec![(Style::Dim, format!(
                    "  gate {} · {}",
                    gate_word(&audit),
                    if gone { "go pressed: rows added since are new and wait for another g" } else { "nothing is checked until g (go)" }
                ))]);
            }
            let mut last_file = String::new();
            for (i, claim) in rows.iter().enumerate().skip(state.top).take(list_room) {
                if state.section == Section::Map {
                    let file = file_of(claim);
                    if file != last_file {
                        let weakest = rows.iter().filter(|c| file_of(c) == file).map(|c| if c.verdict == Verdict::Holds { c.level } else { Level::Documented }).min().unwrap_or_default();
                        let header = vec![(Style::Status, format!("  {file}")), (Style::Dim, format!("  weakest: {}", weakest.label()))];
                        out.push(gui_view::pick_row(header, Click::Pick(Pick::AuditFile(file.clone()))));
                        last_file = file;
                    }
                }
                let row = claim_row(claim, &audit, i == state.selected, width, state.section == Section::Plan);
                out.push(gui_view::pick_row(row, Click::Pick(Pick::AuditRow(i))));
            }
            if rows.len() > state.top + list_room {
                out.push(vec![(Style::Dim, format!("  ⇧↓ {} more", rows.len() - state.top - list_room))]);
            }
            if let Some(claim) = rows.get(state.selected) {
                strip_lines = strip(claim, &audit);
            }
        }
    }
    // The strip.
    let left = rows_left.saturating_sub(out.len() + 1);
    out.push(vec![(Style::Dim, format!("  {}", "─".repeat(width.saturating_sub(4))))]);
    let mut wrapped: Vec<Line> = Vec::new();
    for (n, line) in strip_lines.iter().enumerate() {
        let style = if n == 0 { Style::Bold } else { Style::Plain };
        if let Some(rest) = line.strip_prefix("board    ") {
            wrapped.push(vec![(Style::Plain, "  board    ".into()), gui_view::hot(Style::Accent, super::clip(rest, width.saturating_sub(13)), Click::Pick(Pick::ClaimBoard))]);
            continue;
        }
        wrapped.extend(super::flow(&[(style, format!("  {line}"))], width, 11));
    }
    state.strip_scroll = state.strip_scroll.min(wrapped.len().saturating_sub(1));
    out.extend(wrapped.into_iter().skip(state.strip_scroll).take(left.saturating_sub(1)));
    // The prompt.
    if let Some(input) = &state.input {
        out.truncate(rows_left.saturating_sub(1));
        out.push(prompt_line(input));
    }
    out.truncate(rows_left);
    let left: Line = match &state.scope {
        Some(names) => vec![(Style::Status, "audit".into()), (Style::Dim, format!("  {} of {} for this project", names.iter().position(|n| *n == audit.name).map_or(0, |i| i + 1), names.len()))],
        None => vec![(Style::Status, "audit".into()), (Style::Dim, format!("  {} of {}", position(&data.audits, &audit.name), data.audits.len()))],
    };
    (out, Footer::new(left, hints(state, &audit, &data.claims)))
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
        (Typing::Add, _) => "your claim › ".into(),
        (Typing::Settings, _) => "settings › ".into(),
        (Typing::Decline, _) => "decline, why › ".into(),
        (Typing::Steer, Some(c)) => format!("direction for {c}'s worker › "),
        (Typing::Link, Some(c)) => format!("link {c} to <board> <lead> › "),
        (Typing::Link, None) => "link to <board> › ".into(),
        (Typing::Filter, _) => "filter › ".into(),
        (Typing::Merge, _) => "merge into the standing audit › ".into(),
        (Typing::Priority, Some(c)) => format!("{c} priority p"),
        _ => "› ".into(),
    };
    let before: String = input.text[..input.cursor].iter().collect();
    let at: String = input.text.get(input.cursor).map_or(" ".to_string(), |c| c.to_string());
    let after: String = input.text.get(input.cursor + 1..).map(|rest| rest.iter().collect()).unwrap_or_default();
    vec![(Style::Cursor, "  ".into()), (Style::Cursor, label), (Style::Plain, before), (Style::Caret, at), (Style::Plain, after)]
}

fn hints(state: &State, audit: &Audit, claims: &[Claim]) -> Vec<(&'static str, &'static str)> {
    if state.input.is_some() {
        return vec![("⏎", "send"), ("esc", "cancel")];
    }
    let mut out: Vec<(&'static str, &'static str)> = vec![("↑↓", "move"), ("⇧←→", "section")];
    if audit.status == Status::Proposed {
        out.extend([("a", "start"), ("e", "settings"), ("x", "decline")]);
        return out;
    }
    // At complete: what happens to it (keep, let go, merge).
    if at_complete(audit) {
        out.extend([("k", "keep"), ("v", "let go"), ("M", "merge")]);
    }
    let rows = section_rows(state, audit, claims);
    let claim = rows.get(state.selected);
    if let Some(claim) = claim {
        out.push(("⏎", "open"));
        if claim.review == Review::Proposed {
            out.extend([("y", "accept"), ("n", "reject")]);
        }
        if !claim.readings.is_empty() && claim.picked.is_none() {
            out.push(("1", "reading 1"));
            out.push(("2", "reading 2"));
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
        if claim.posted == Posted::Ready && state.section != Section::Plan {
            out.extend([("p", "post"), ("x", "discard")]);
        }
        if claim.repro == Repro::Fails && audit.standing() {
            out.push(("f", "fix project"));
        }
        out.extend([("t", if claim.taken { "give back" } else { "take" }), ("r", "run now"), ("m", "note"), ("l", "link")]);
        if claim.held_by.is_some() && state.section != Section::Plan {
            out.push(("s", "steer"));
        }
        if state.section == Section::Plan {
            out.extend([("!", "pin"), ("s", "skip"), ("d", "depth"), ("p", "priority")]);
        }
        if state.section == Section::Map {
            out.push(("o", "open file"));
        }
        if claim.review == Review::Accepted && claim.posted != Posted::Ready {
            out.push(("x", "drop"));
        }
    }
    if state.section == Section::Plan || audit.gone_at.is_none() {
        out.push(("g", "go"));
    }
    out.extend([("+", "add claim"), ("/", "filter"), ("c", "settings"), ("z", if audit.paused { "resume" } else { "pause" })]);
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
    let mut out: Vec<Line> = vec![vec![(Style::Status, "  audits".into()), (Style::Dim, "  · attached to projects, and standing · ⏎ opens one".into())]];
    let store = audit::store();
    let adapters = crate::audit::adapters::proposed();
    let total = data.audits.len() + adapters.len();
    state.portfolio = state.portfolio.min(total.saturating_sub(1));
    if total == 0 {
        out.extend(super::flow(
            &[(Style::Dim, "  No audits yet. s0ma proposes one for a project when checking what its code claims would help (it waits for you here), \
                or start one: /audit start <name> <path>… [--for <board>] [--scope path:src/x.rs]".to_string())],
            width,
            2,
        ));
    }
    let mut group = String::new();
    for (i, audit) in data.audits.iter().enumerate() {
        let owner = audit.board.clone().unwrap_or_else(|| "standing".into());
        if owner != group {
            out.push(vec![(Style::Status, format!("  {owner}"))]);
            group = owner;
        }
        let claims = store.as_ref().map(|s| s.claims(&audit.name).unwrap_or_default()).unwrap_or_default();
        let waiting = claims.iter().filter(|c| c.mark().0 == "[!]").count() + usize::from(audit.status == Status::Proposed || audit.rerun_proposed.is_some());
        let (mark, status) = audit.status_mark();
        let here = i == state.portfolio;
        let mut line: Line = vec![
            (if here { Style::Cursor } else { Style::Plain }, if here { "❯ ".into() } else { "  ".into() }),
            (if here { Style::Bold } else { Style::Plain }, format!("{:<18}", audit.name)),
            (if mark == "[!]" { Style::Warn } else { Style::Dim }, format!("{mark} {:<28}", super::clip(status, 26))),
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
            (Style::Dim, " · s0ma proposes it: a twice approves it and runs its doctor".into()),
        ];
        out.push(gui_view::pick_row(line, Click::Pick(Pick::AuditRow(i))));
    }
    if let Some(input) = &state.input {
        out.push(prompt_line(input));
    }
    out.truncate(rows_left);
    let hints = if state.input.is_some() { vec![("⏎", "send"), ("esc", "cancel")] } else { vec![("↑↓", "move"), ("⏎", "open"), ("a", "start"), ("r", "re-run"), ("esc", "back")] };
    (out, Footer::new(vec![(Style::Status, "audit".into()), (Style::Dim, format!("  {} audits", data.audits.len()))], hints))
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
    if tab.audit.audit.is_none() {
        tab.audit.portfolio = index;
    } else {
        tab.audit.selected = index;
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

/// A click on a file's row in the map: the file, code first.
pub(super) fn pick_file(tab: &mut Open, file: &str) {
    if let Some(audit) = read(&tab.audit).audit {
        tab.audit.view = Some(super::claim_view::State::open_file(&audit, file));
    }
}

pub(super) fn pick_section(tab: &mut Open, index: usize) {
    if let Some(section) = SECTIONS.get(index) {
        tab.audit.section = *section;
        tab.audit.selected = 0;
        tab.audit.top = 0;
    }
}

/// The audit tab opened at an audit (a board's audit lead, the projects tab).
pub(super) fn open_at(tab: &mut Open, audit: &str, claim: Option<&str>) {
    tab.audit.audit = Some(audit.to_string());
    tab.audit.section = Section::Claims;
    if let Some(claim) = claim {
        let data = read(&tab.audit);
        if let Some(a) = data.audit {
            let rows = section_rows(&tab.audit, &a, &data.claims);
            tab.audit.selected = rows.iter().position(|c| c.id == claim).unwrap_or(0);
        }
    }
}

pub(super) fn wheel(tab: &mut Open, up: bool) {
    if tab.audit.audit.is_none() {
        tab.audit.portfolio = if up { tab.audit.portfolio.saturating_sub(1) } else { tab.audit.portfolio + 1 };
    } else if up {
        tab.audit.selected = tab.audit.selected.saturating_sub(1);
    } else {
        tab.audit.selected += 1;
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
            tab.audit.filter = text;
            tab.audit.selected = 0;
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
        Typing::Add => crate::audit::ops::operator_claim_new(&store, &name, &text, ""),
        Typing::Settings => crate::audit::ops::set(&store, &name, &text),
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
        KeyCode::Tab => {
            step(tab, 1);
            return Outcome::Stay;
        }
        KeyCode::BackTab => {
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
                    tab.audit.section = if a.gone_at.is_none() { Section::Plan } else { Section::Claims };
                    tab.audit.selected = 0;
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
                    && twice(tab, 'a', lang, format!("a again approves the {lang} adapter and runs its doctor"))
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
            _ => {}
        }
        return Outcome::Stay;
    };
    let rows = section_rows(&tab.audit, &audit, &data.claims);
    let at = tab.audit.selected;
    let claim = rows.get(at).cloned();
    let id = claim.as_ref().map(|c| c.id.clone()).unwrap_or_default();
    let name = audit.name.clone();
    let act = |tab: &mut Open, action: &str, text: &str| {
        let result = crate::audit::ops::operator_claim(&store, &name, &id, action, text);
        note(tab, result);
    };
    // A proposal: start, change, decline.
    if audit.status == Status::Proposed {
        match key.code {
            KeyCode::Char('a') if twice(tab, 'a', &name, format!("a again starts {name} as proposed")) => note(tab, crate::audit::ops::start(&store, &name, Who::Operator)),
            KeyCode::Char('e' | 'c') => start_input(tab, Typing::Settings, None, ""),
            KeyCode::Char('x') if twice(tab, 'x', &name, format!("x again declines {name}")) => start_input(tab, Typing::Decline, None, ""),
            KeyCode::Esc if tab.audit.scope.is_some() => return Outcome::Closed,
            KeyCode::Esc => tab.audit.audit = None,
            KeyCode::Char('q') => return Outcome::Closed,
            _ => {}
        }
        return Outcome::Stay;
    }
    match key.code {
        KeyCode::Esc => {
            if !tab.audit.filter.is_empty() {
                tab.audit.filter.clear();
            } else if tab.audit.scope.is_some() {
                return Outcome::Closed;
            } else {
                tab.audit.audit = None;
            }
        }
        KeyCode::Char('q') => return Outcome::Closed,
        KeyCode::Left if shift => {
            let i = SECTIONS.iter().position(|s| *s == tab.audit.section).unwrap_or(0);
            pick_section(tab, (i + SECTIONS.len() - 1) % SECTIONS.len());
        }
        KeyCode::Right if shift => {
            let i = SECTIONS.iter().position(|s| *s == tab.audit.section).unwrap_or(0);
            pick_section(tab, (i + 1) % SECTIONS.len());
        }
        // In the plan, Shift+↑↓ moves a row (its priority past its neighbour's).
        KeyCode::Up if shift && tab.audit.section == Section::Plan => {
            if let (Some(c), Some(above)) = (&claim, at.checked_sub(1).and_then(|i| rows.get(i))) {
                let p = (above.priority + 1).min(10);
                let _ = crate::audit::ops::operator_claim(&store, &name, &c.id, "priority", &p.to_string());
                tab.audit.selected = at.saturating_sub(1);
            }
        }
        KeyCode::Down if shift && tab.audit.section == Section::Plan => {
            if let (Some(c), Some(below)) = (&claim, rows.get(at + 1)) {
                let p = (below.priority - 1).max(1);
                let _ = crate::audit::ops::operator_claim(&store, &name, &c.id, "priority", &p.to_string());
                tab.audit.selected = at + 1;
            }
        }
        KeyCode::Up if shift => tab.audit.strip_scroll = tab.audit.strip_scroll.saturating_sub(1),
        KeyCode::Down if shift => tab.audit.strip_scroll += 1,
        KeyCode::Up => {
            tab.audit.selected = at.saturating_sub(1);
            tab.audit.strip_scroll = 0;
            tab.audit.armed = None;
        }
        KeyCode::Down => {
            tab.audit.selected = at + 1;
            tab.audit.strip_scroll = 0;
            tab.audit.armed = None;
        }
        KeyCode::PageUp => tab.audit.selected = at.saturating_sub(10),
        KeyCode::PageDown => tab.audit.selected = at + 10,
        KeyCode::Home => tab.audit.selected = 0,
        KeyCode::End => tab.audit.selected = rows.len().saturating_sub(1),
        KeyCode::Enter => {
            if let Some(c) = &claim {
                tab.audit.view = Some(super::claim_view::State::open(&audit, c));
            }
        }
        KeyCode::Char('k') if at_complete(&audit) => {
            if twice(tab, 'k', &format!("keep {name}"), format!("k again keeps {name} as a standing audit: unreviewed claims go to triage, counting for nothing")) {
                note(tab, crate::audit::run::after_project(&store, &name, "keep", None));
            }
        }
        KeyCode::Char('v') if at_complete(&audit) => {
            if twice(tab, 'v', &format!("letgo {name}"), format!("v again lets {name} go: it closes, its files stay in /share/audits/{name}/")) {
                note(tab, crate::audit::run::after_project(&store, &name, "letgo", None));
            }
        }
        KeyCode::Char('M') if at_complete(&audit) => start_input(tab, Typing::Merge, None, ""),
        KeyCode::Char('g') => note(tab, crate::audit::ops::go(&store, &name)),
        KeyCode::Char('z') => note(tab, crate::audit::ops::pause(&store, &name)),
        KeyCode::Char('c') => start_input(tab, Typing::Settings, None, ""),
        KeyCode::Char('+') => start_input(tab, Typing::Add, None, ""),
        KeyCode::Char('/') => start_input(tab, Typing::Filter, None, &tab.audit.filter.clone()),
        KeyCode::Char('r') if claim.is_none() => note(tab, crate::audit::run::rerun(&store, &name, Who::Operator)),
        KeyCode::Char('r') => note(tab, crate::audit::run::run_now(&store, &name, Some(&id))),
        KeyCode::Char('l') => start_input(tab, Typing::Link, claim.as_ref().map(|c| c.id.clone()), ""),
        KeyCode::Char('a') if audit.attach_proposed.is_some() && claim.as_ref().is_none_or(|c| c.weaker.is_none() && c.repro != Repro::ModelOnly) => {
            let (board, why) = audit.attach_proposed.clone().unwrap_or_default();
            if twice(tab, 'a', &format!("attach {name}"), format!("a again attaches {name} to {board}")) {
                note(tab, crate::audit::ops::attach(&store, &name, &board, Who::Operator, &why));
            }
        }
        _ if claim.is_none() => {}
        KeyCode::Char('y') => act(tab, "accept", ""),
        KeyCode::Char('n') => act(tab, "reject", ""),
        KeyCode::Char(digit @ '1'..='4') => act(tab, "pick", &digit.to_string()),
        KeyCode::Char('e') => start_input(tab, Typing::Edit, Some(id.clone()), &claim.as_ref().map(|c| c.text.clone()).unwrap_or_default()),
        KeyCode::Char('m') => start_input(tab, Typing::Note, Some(id.clone()), ""),
        KeyCode::Char('b') => start_input(tab, Typing::Back, Some(id.clone()), ""),
        KeyCode::Char('x') if claim.as_ref().is_some_and(|c| c.posted == Posted::Ready) => {
            if twice(tab, 'x', &id, format!("x again discards {id}'s result")) {
                act(tab, "discard", "");
            }
        }
        KeyCode::Char('x') if twice(tab, 'x', &id, format!("x again drops {id}")) => act(tab, "drop", ""),
        KeyCode::Char('o') if tab.audit.section == Section::Map => {
            if let Some(c) = &claim {
                tab.audit.view = Some(super::claim_view::State::open_file(&audit, &file_of(c)));
            }
        }
        KeyCode::Char('a') if twice(tab, 'a', &id, format!("a again accepts what {id} waits on")) => {
            if claim.as_ref().is_some_and(|c| c.weaker.is_some()) {
                note(tab, crate::audit::run::accept_weaker(&store, &name, &id));
            } else {
                act(tab, "accept-weaker", "");
            }
        }
        KeyCode::Char('t') => act(tab, "take", ""),
        KeyCode::Char('w') => act(tab, "waive", ""),
        KeyCode::Char('p') if tab.audit.section == Section::Plan => start_input(tab, Typing::Priority, Some(id.clone()), ""),
        KeyCode::Char('p') => act(tab, "post", ""),
        KeyCode::Char('!') => act(tab, "pin", ""),
        KeyCode::Char('s') if tab.audit.section == Section::Plan => act(tab, "skip", ""),
        KeyCode::Char('s') => start_input(tab, Typing::Steer, Some(id.clone()), ""),
        KeyCode::Char('d') => act(tab, "depth", ""),
        KeyCode::Char('f') => note(tab, crate::audit::ops::fix_project(&store, &name, &id)),
        _ => {}
    }
    let _ = (Level::Tested, Tone::Plain);
    Outcome::Stay
}

