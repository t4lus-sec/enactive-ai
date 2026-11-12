//! The claim view (⏎ on a claim in the audit tab), a submodule of `tabs.rs`:
//! the claim, its code and its actual spec side by side at 160 columns or
//! more, as tabs (`1` `2` `3`) narrower. Clause anchors light their lines in
//! all three; the code's gutter carries the audit's marks (scope, coverage,
//! killed and survived mutants, calls out with their assumptions, the
//! counterexample's step); `[` `]` replay a counterexample; `m` steps through
//! survived mutants. Code first: a file with every claim about it in the
//! gutter.
//!
//! Read-only: the code is the project's to change, and the spec is the
//! harness's to judge. Only a claim the operator took has a spec he edits
//! (`e`, in the editor), and submits (`s`) to the harness like any other.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::audit::{self, Audit, Claim, Repro, Verdict};
use crate::editor::highlight::{Class, Highlighter};
use crate::style::{Line, Style};

use super::gui_view::{self, Click, Pick};
use super::{Footer, Open};

/// Three panes side by side from this many columns.
pub(super) const WIDE: usize = 160;

const PANES: [&str; 3] = ["claim", "code", "spec"];

/// A mutant patch, read: where it changes the code, and to what.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Mutant {
    pub name: String,
    pub file: String,
    /// 0-based line in the audited file.
    pub line: usize,
    pub change: String,
    pub killed: bool,
}

/// One step of a counterexample.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Step {
    pub label: String,
    /// The state after it (`var = value`), for a model's trace.
    pub state: Vec<String>,
    /// The lines that changed since the step before.
    pub changed: Vec<usize>,
    /// Where it ran: a real source location (Kani, tests).
    pub at: Option<(String, usize)>,
    /// A model's action, mapped to code by the agent's `@code` annotation.
    pub action: Option<String>,
}

/// What was read from the audit's work tree.
#[derive(Clone, Debug, Default)]
pub(super) struct Loaded {
    pub files: Vec<(String, Vec<String>)>,
    pub spec: Vec<(String, Vec<String>)>,
    pub mutants: Vec<Mutant>,
    /// Lines the check's runs reached, by file (`checks/c<N>/coverage.lines`).
    pub coverage: HashMap<String, Vec<usize>>,
}

type Slot<T> = Arc<Mutex<Option<Result<T, String>>>>;

pub(super) struct State {
    pub audit: String,
    /// The claim (None: code first, a file with its claims).
    pub claim: Option<String>,
    pub pane: usize,
    pub scroll: [usize; 3],
    /// The code pane's line, and which file.
    pub cursor: usize,
    pub file: Option<String>,
    pub clause: Option<usize>,
    pub step: Option<usize>,
    pub mutant: Option<usize>,
    pub loaded: Option<Loaded>,
    loading: Option<Slot<Loaded>>,
    jumping: Option<Slot<(String, Vec<String>, usize)>>,
    /// Where ⏎ on a call came from: Ctrl+O goes back.
    pub back: Vec<(String, usize)>,
    highlight: HashMap<String, Highlighter>,
    /// The code file code first opened at.
    first_file: Option<String>,
    /// The claim's trace, parsed.
    pub steps: Vec<Step>,
}

impl State {
    /// The view of `claim`: its code and spec read from the work tree on a
    /// thread of their own.
    pub(super) fn open(audit: &Audit, claim: &Claim) -> State {
        let mut state = State::blank(&audit.name, Some(claim.id.clone()));
        state.steps = audit::store().map(|store| trace_steps(&store, &audit.name, claim)).unwrap_or_default();
        state.load(audit, claim_files(audit, claim));
        state
    }

    /// Code first: `file` with every claim about it in the gutter.
    pub(super) fn open_file(audit: &Audit, file: &str) -> State {
        let mut state = State::blank(&audit.name, None);
        state.pane = 1;
        state.first_file = Some(file.to_string());
        state.load(audit, vec![file.to_string()]);
        state
    }

    fn blank(audit: &str, claim: Option<String>) -> State {
        State {
            audit: audit.to_string(),
            claim,
            pane: 0,
            scroll: [0; 3],
            cursor: 0,
            file: None,
            clause: None,
            step: None,
            mutant: None,
            loaded: None,
            loading: None,
            jumping: None,
            back: Vec::new(),
            highlight: HashMap::new(),
            first_file: None,
            steps: Vec::new(),
        }
    }

    fn load(&mut self, audit: &Audit, files: Vec<String>) {
        if cfg!(test) {
            return;
        }
        let slot: Slot<Loaded> = Arc::new(Mutex::new(None));
        self.loading = Some(slot.clone());
        let work = audit.work.clone();
        let claim = self.claim.clone();
        let survived: Vec<String> = claim
            .as_deref()
            .and_then(|c| audit::store().and_then(|s| s.current_check(&audit.name, c).ok().flatten()))
            .map(|check| check.survived)
            .unwrap_or_default();
        let _ = std::thread::Builder::new().name("claim-view".into()).spawn(move || {
            let read = read_tree(&work, claim.as_deref(), &files).map(|mut loaded| {
                for mutant in &mut loaded.mutants {
                    mutant.killed = !survived.contains(&mutant.name);
                }
                loaded
            });
            *slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(read);
        });
    }

    /// What was read arrived: open at the claim's code.
    fn arrive(&mut self, loaded: Loaded) {
        let file = self.first_file.clone().filter(|f| loaded.files.iter().any(|(path, _)| path == f)).or_else(|| loaded.files.first().map(|(f, _)| f.clone()));
        self.file = file;
        self.loaded = Some(loaded);
        self.cursor = self.start_line().unwrap_or(0);
        self.scroll[1] = self.cursor.saturating_sub(3);
    }

    /// Whether a read on its thread has come back; true when it changed the
    /// view.
    fn poll(&mut self) -> Option<String> {
        let mut problem = None;
        if let Some(slot) = &self.loading {
            let done = slot.lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(result) = done {
                self.loading = None;
                match result {
                    Ok(loaded) => self.arrive(loaded),
                    Err(error) => {
                        self.loaded = Some(Loaded::default());
                        problem = Some(error);
                    }
                }
            }
        }
        if let Some(slot) = &self.jumping {
            let done = slot.lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(result) = done {
                self.jumping = None;
                match result {
                    Ok((file, lines, line)) => {
                        if let Some(loaded) = self.loaded.as_mut()
                            && !loaded.files.iter().any(|(f, _)| *f == file)
                        {
                            loaded.files.push((file.clone(), lines));
                        }
                        self.go(&file, line);
                    }
                    Err(error) => problem = Some(error),
                }
            }
        }
        problem
    }

    fn lines(&self) -> &[String] {
        let Some(loaded) = &self.loaded else { return &[] };
        let Some(file) = &self.file else { return &[] };
        loaded.files.iter().find(|(f, _)| f == file).map(|(_, lines)| lines.as_slice()).unwrap_or(&[])
    }

    /// Where the claim's code starts: its target's definition, or its
    /// source's line.
    fn start_line(&self) -> Option<usize> {
        let claim = self.claim_now()?;
        let file = self.file.clone()?;
        let lines = self.lines();
        let check = audit::store().and_then(|s| s.current_check(&self.audit, &claim.id).ok().flatten());
        let targets = check.map(|c| c.targets).unwrap_or_default();
        for target in targets {
            if let Some((path, symbol)) = target.split_once(':')
                && path == file
                && let Some(line) = definition(lines, symbol.rsplit("::").next().unwrap_or(symbol))
            {
                return Some(line);
            }
        }
        let source = claim.source.trim();
        let (path, line) = source.split_once(':')?;
        (path == file).then(|| line.split(|c: char| !c.is_ascii_digit()).next()?.parse::<usize>().ok().map(|n| n.saturating_sub(1)))?
    }

    fn claim_now(&self) -> Option<Claim> {
        let id = self.claim.as_deref()?;
        audit::store()?.claim(&self.audit, id).ok().flatten()
    }

    fn go(&mut self, file: &str, line: usize) {
        self.file = Some(file.to_string());
        self.cursor = line;
        self.scroll[1] = line.saturating_sub(3);
        self.pane = 1;
    }
}

// ------------------------------------------------------------------ reading

/// The files a claim's code is in: its check's targets, the files its spec
/// maps with `@code`, its source, the scope's paths.
fn claim_files(audit: &Audit, claim: &Claim) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    let mut add = |file: &str| {
        let file = file.trim().trim_start_matches("./").to_string();
        if !file.is_empty() && !files.contains(&file) && (file.contains('.') || file.contains('/')) && !file.contains(' ') {
            files.push(file);
        }
    };
    if let Some(store) = audit::store()
        && let Ok(Some(check)) = store.current_check(&audit.name, &claim.id)
    {
        for target in &check.targets {
            add(target.split(':').next().unwrap_or(target));
        }
    }
    let source = claim.source.trim();
    if !source.starts_with("board:") && !source.contains(".md") {
        add(source.split([':', ' ']).next().unwrap_or(source));
    }
    for spec in &audit.scope {
        if let Some(path) = spec.strip_prefix("path:")
            && !path.contains('*')
        {
            add(path);
        }
    }
    files
}

/// The files, the claim's spec, its mutants and coverage, from the work tree
/// on the agent's machine, in one command.
fn read_tree(work: &str, claim: Option<&str>, files: &[String]) -> Result<Loaded, String> {
    let q = crate::util::shell_quote;
    let mut script = format!(
        "cd {} || exit 3\nemit(){{ printf '\\n@@@S0MA %s %s\\n' \"$1\" \"$2\"; head -c 400000 \"$3\"; }}\n",
        q(work)
    );
    for file in files {
        script.push_str(&format!("[ -f {f} ] && emit code {f} {f}\n", f = q(file)));
    }
    if let Some(claim) = claim {
        let dir = q(&format!("checks/{claim}"));
        script.push_str(&format!(
            "if [ -d {dir} ]; then\n\
             for f in $(find {dir} -type f ! -path '*/mutants/*' ! -name coverage.lines | sort | head -20); do emit spec \"${{f#checks/{claim}/}}\" \"$f\"; done\n\
             for f in {dir}/mutants/*; do [ -f \"$f\" ] && emit mutant \"$(basename \"$f\")\" \"$f\"; done\n\
             [ -f {dir}/coverage.lines ] && emit coverage x {dir}/coverage.lines\n\
             for f in $(grep -rhoE '@code [^: ]+' {dir} 2>/dev/null | awk '{{print $2}}' | sort -u | head -8); do [ -f \"$f\" ] && emit code \"$f\" \"$f\"; done\n\
             fi\n"
        ));
    }
    let ran = crate::vm::netboxes::machine(&script, 60)?;
    if ran.exit_code == Some(3) {
        return Err("the audit's work tree is not there yet (it is set up as harvest starts)".into());
    }
    Ok(parse_tree(&ran.stdout))
}

/// `read_tree`'s output, read back.
pub(super) fn parse_tree(text: &str) -> Loaded {
    let mut loaded = Loaded::default();
    let mut current: Option<(String, String, Vec<String>)> = None;
    let finish = |current: Option<(String, String, Vec<String>)>, loaded: &mut Loaded| {
        let Some((kind, name, lines)) = current else { return };
        match kind.as_str() {
            "code" if !loaded.files.iter().any(|(f, _)| *f == name) => loaded.files.push((name, lines)),
            "spec" => loaded.spec.push((name, lines)),
            "mutant" => loaded.mutants.extend(parse_patch(&name, &lines.join("\n"))),
            "coverage" => {
                for line in lines {
                    if let Some((file, n)) = line.trim().rsplit_once(':')
                        && let Ok(n) = n.parse::<usize>()
                    {
                        loaded.coverage.entry(file.trim_start_matches("./").to_string()).or_default().push(n.saturating_sub(1));
                    }
                }
            }
            _ => {}
        }
    };
    for line in text.lines() {
        if let Some(head) = line.strip_prefix("@@@S0MA ") {
            finish(current.take(), &mut loaded);
            let (kind, name) = head.split_once(' ').unwrap_or((head, ""));
            current = Some((kind.to_string(), name.to_string(), Vec::new()));
        } else if let Some((_, _, lines)) = current.as_mut() {
            lines.push(line.to_string());
        }
    }
    finish(current, &mut loaded);
    loaded
}

/// A unified diff's first change: the file, the line it changes (0-based,
/// in the original), and what the line becomes.
pub(super) fn parse_patch(name: &str, patch: &str) -> Option<Mutant> {
    let mut file = String::new();
    let mut old = 0usize;
    let mut in_hunk = false;
    for line in patch.lines() {
        if let Some(path) = line.strip_prefix("+++ ") {
            file = path.trim().trim_start_matches("b/").split('\t').next().unwrap_or("").to_string();
            in_hunk = false;
        } else if let Some(rest) = line.strip_prefix("@@ -") {
            old = rest.split([',', ' ']).next().and_then(|n| n.parse::<usize>().ok()).unwrap_or(1).saturating_sub(1);
            in_hunk = true;
        } else if in_hunk && line.starts_with("---") {
            in_hunk = false;
        } else if in_hunk {
            match line.chars().next() {
                Some('-') => {
                    let change = patch
                        .lines()
                        .skip_while(|l| *l != line)
                        .skip(1)
                        .find(|l| l.starts_with('+') && !l.starts_with("+++"))
                        .map(|l| l[1..].trim().to_string())
                        .unwrap_or_else(|| "(line removed)".into());
                    return Some(Mutant { name: name.to_string(), file, line: old, change, killed: true });
                }
                Some('+') => return Some(Mutant { name: name.to_string(), file, line: old, change: line[1..].trim().to_string(), killed: true }),
                _ => old += 1,
            }
        }
    }
    None
}

/// The claim's counterexample, from its trace evidence, as steps: a TLC
/// trace's states, or the source locations of a Kani trace or a failing
/// test's output, or a shrunk case.
fn trace_steps(store: &audit::AuditStore, audit: &str, claim: &Claim) -> Vec<Step> {
    let Some(trace) = &claim.trace else { return Vec::new() };
    let text = store.evidence(audit, Some(&claim.id)).unwrap_or_default().into_iter().rev().find(|e| e.kind == "trace" && e.path == *trace).map(|e| e.result).unwrap_or_default();
    parse_trace(&text)
}

pub(super) fn parse_trace(text: &str) -> Vec<Step> {
    let mut steps: Vec<Step> = Vec::new();
    // TLC: "State 3: <Download line 40, col 5 to …>" then "/\ var = value".
    if text.lines().any(|l| l.trim_start().starts_with("State ") && l.contains(':')) {
        for line in text.lines() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("State ")
                && let Some((_, label)) = rest.split_once(':')
            {
                let label = label.trim().trim_start_matches('<').trim_end_matches('>').to_string();
                let action = label.split_whitespace().next().filter(|a| !a.eq_ignore_ascii_case("initial")).map(str::to_string);
                steps.push(Step { label, action, ..Step::default() });
            } else if let Some(step) = steps.last_mut()
                && (t.starts_with("/\\") || (t.contains(" = ") && !t.is_empty()))
            {
                step.state.push(t.trim_start_matches("/\\").trim().to_string());
            }
        }
        for i in 1..steps.len() {
            let before = steps[i - 1].state.clone();
            let changed: Vec<usize> = steps[i].state.iter().enumerate().filter(|(_, s)| !before.contains(s)).map(|(n, _)| n).collect();
            steps[i].changed = changed;
        }
        return steps;
    }
    // Source locations: Kani's and a test's own.
    for line in text.lines() {
        if let Some((file, n)) = location(line)
            && !steps.iter().any(|s| s.at.as_ref() == Some(&(file.clone(), n)))
        {
            steps.push(Step { label: crate::util::clip(line.trim(), 100), at: Some((file, n)), ..Step::default() });
        }
    }
    // A shrunk case.
    if steps.is_empty()
        && let Some(line) = text.lines().find(|l| l.contains("minimal failing input") || l.contains("Falsifying example"))
    {
        steps.push(Step { label: line.trim().to_string(), ..Step::default() });
    }
    steps
}

/// The first `path.ext:line` in a line of output.
fn location(line: &str) -> Option<(String, usize)> {
    for word in line.split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | ',' | '\'' | '"')) {
        let mut parts = word.split(':');
        let (Some(path), Some(n)) = (parts.next(), parts.next()) else { continue };
        if path.contains('.') && !path.starts_with("http") && path.len() > 2 && let Ok(n) = n.parse::<usize>() {
            return Some((path.trim_start_matches("./").to_string(), n.saturating_sub(1)));
        }
    }
    None
}

/// The line where `name` is defined, by the usual ways languages write it.
pub(super) fn definition(lines: &[String], name: &str) -> Option<usize> {
    if name.is_empty() {
        return None;
    }
    let ends = |rest: &str| rest.chars().next().is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
    for keyword in ["fn ", "def ", "function ", "func ", "class ", "struct ", "enum ", "trait ", "type ", "interface "] {
        for (n, line) in lines.iter().enumerate() {
            let mut from = 0;
            while let Some(at) = line[from..].find(keyword) {
                let start = from + at;
                let before_ok = start == 0 || !line[..start].chars().last().is_some_and(|c| c.is_alphanumeric() || c == '_');
                let rest = &line[start + keyword.len()..];
                if before_ok && rest.starts_with(name) && ends(&rest[name.len()..]) {
                    return Some(n);
                }
                from = start + keyword.len();
            }
        }
    }
    // TLA+: `Name ==`; C-like: a line at the margin naming it with `(`.
    lines.iter().position(|line| {
        let t = line.trim_start();
        (t.starts_with(name) && t[name.len()..].trim_start().starts_with("=="))
            || (!line.starts_with([' ', '\t']) && !line.trim_end().ends_with(';') && line.contains(&format!("{name}(")) && !line.trim_start().starts_with("//"))
    })
}

const KEYWORDS: [&str; 16] = ["if", "for", "while", "match", "return", "fn", "def", "Some", "Ok", "Err", "assert", "assert_eq", "println", "format", "vec", "switch"];

/// The calls on a line: names followed by `(`.
pub(super) fn calls(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let name: String = chars[start..i].iter().collect();
            let mut j = i;
            if j < chars.len() && chars[j] == '!' {
                j += 1;
            }
            if j < chars.len() && chars[j] == '(' && !KEYWORDS.contains(&name.as_str()) && !out.contains(&name) {
                out.push(name);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// The clause numbers anchored on a line: `// [1]`, `\* [2]`, `# [3]`.
pub(super) fn anchors(line: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(at) = rest.find('[') {
        let after = &rest[at + 1..];
        let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
        if !digits.is_empty() && after[digits.len()..].starts_with(']') {
            let before = rest[..at].trim_end();
            if before.ends_with("//") || before.ends_with("\\*") || before.ends_with('#') || before.ends_with("--") || before.ends_with(']') || before.ends_with("(*") {
                out.push(digits.parse().unwrap_or(0));
            }
        }
        rest = after;
    }
    out
}

/// A spec's `@code <file>:<fn>` annotation on a line.
pub(super) fn code_mapping(line: &str) -> Option<(String, String)> {
    let rest = &line[line.find("@code ")? + 6..];
    let word = rest.split_whitespace().next()?;
    let (file, symbol) = word.split_once(':')?;
    Some((file.to_string(), symbol.to_string()))
}

// ------------------------------------------------------------------ drawing

fn class_style(class: Class) -> Style {
    match class {
        Class::Plain => Style::Plain,
        Class::Keyword | Class::Heading => Style::Bold,
        Class::String | Class::Raw => Style::Accent,
        Class::Constant => Style::Status,
        Class::Comment => Style::Work,
        Class::Error => Style::Problem,
    }
}

/// A line clipped or padded to exactly `width` columns, its clicks kept.
fn fit(line: &Line, width: usize) -> Line {
    let mut out: Line = Vec::new();
    let mut used = 0;
    for (style, text) in line {
        if used >= width {
            break;
        }
        let piece = super::clip(text, width - used);
        used += super::text_width(&piece);
        out.push((*style, piece));
    }
    if used < width {
        out.push((Style::Plain, " ".repeat(width - used)));
    }
    out
}

/// Spec lines (across its files, as drawn) that state clause `k`.
fn anchored(loaded: &Loaded, k: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (f, (_, lines)) in loaded.spec.iter().enumerate() {
        for (n, line) in lines.iter().enumerate() {
            if anchors(line).contains(&k) {
                out.push((f, n));
            }
        }
    }
    out
}

/// The code lines a clause lights: the definitions of what its anchored spec
/// lines call, and the functions their `@code` annotations map to.
fn clause_code_lines(state: &State, k: usize) -> Vec<usize> {
    let Some(loaded) = &state.loaded else { return Vec::new() };
    let lines = state.lines();
    let mut out = Vec::new();
    for (f, n) in anchored(loaded, k) {
        let spec_lines = &loaded.spec[f].1;
        let line = &spec_lines[n];
        let mut names = calls(line);
        // The annotation nearest above, within its block.
        for above in spec_lines[..=n].iter().rev().take(12) {
            if let Some((file, symbol)) = code_mapping(above) {
                if Some(&file) == state.file.as_ref() {
                    names.push(symbol);
                }
                break;
            }
        }
        for name in names {
            if let Some(at) = definition(lines, &name)
                && !out.contains(&at)
            {
                out.push(at);
            }
        }
    }
    out
}

/// The code line a counterexample's step is on: its own location, or the
/// function the agent's `@code` maps its action to (marked as the agent's mapping).
fn step_line(state: &State) -> Option<(usize, bool)> {
    let step = state.steps.get(state.step?)?;
    if let Some((file, line)) = &step.at {
        return (Some(file) == state.file.as_ref()).then_some((*line, false));
    }
    let action = step.action.as_ref()?;
    let loaded = state.loaded.as_ref()?;
    for (_, lines) in &loaded.spec {
        let Some(at) = definition(lines, action) else { continue };
        for line in lines[at.saturating_sub(3)..lines.len().min(at + 15)].iter() {
            if let Some((file, symbol)) = code_mapping(line)
                && Some(&file) == state.file.as_ref()
            {
                return definition(state.lines(), &symbol).map(|n| (n, true));
            }
        }
    }
    None
}

/// The claim pane, whole.
fn claim_pane(state: &State, claim: &Claim, width: usize) -> Vec<Line> {
    let store = audit::store();
    let check = store.as_ref().and_then(|s| s.current_check(&state.audit, &claim.id).ok().flatten());
    let mut out: Vec<Line> = Vec::new();
    let (mark, tone) = claim.mark();
    let tone = match tone {
        audit::Tone::Danger => Style::Danger,
        audit::Tone::Success => Style::Success,
        audit::Tone::Warn => Style::Warn,
        audit::Tone::Dim => Style::Dim,
        audit::Tone::Plain => Style::Plain,
    };
    out.push(vec![
        (Style::Bold, format!("{} ", claim.id)),
        (tone, format!("{mark} {}", claim.verdict.label())),
        (Style::Dim, format!(" · {} · {}", claim.level.label(), if claim.author == "operator" { "his wording" } else { "s0ma's wording" })),
    ]);
    let text = |out: &mut Vec<Line>, style: Style, s: String| out.extend(super::flow(&[(style, s)], width, 2));
    text(&mut out, Style::Plain, claim.text.clone());
    if !claim.source.is_empty() {
        text(&mut out, Style::Dim, format!("source  {}", claim.source));
    }
    out.push(Vec::new());
    // The clauses, each with what the spec says about it.
    out.push(vec![(Style::Status, "clauses".into()), (Style::Dim, "  ←→ selects one: its lines light in every pane".into())]);
    let back = check.as_ref().and_then(|c| c.back_translation.clone());
    let sentences: Vec<String> = back.as_deref().map(|b| b.split_inclusive(['.', ';']).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).unwrap_or_default();
    let mut placed = vec![false; sentences.len()];
    for (i, clause) in audit::clauses(&claim.text).iter().enumerate() {
        let k = i + 1;
        let here = state.clause == Some(k);
        let anchored = check.as_ref().is_some_and(|c| c.anchors.contains(&k));
        let mut line: Line = vec![
            gui_view::hot(if here { Style::Cursor } else { Style::Plain }, format!("{}[{k}] ", if here { "❯" } else { " " }), Click::Pick(Pick::Clause(k))),
            gui_view::hot(if here { Style::Bold } else { Style::Plain }, clause.clone(), Click::Pick(Pick::Clause(k))),
        ];
        if check.is_some() && !anchored {
            line.push((Style::Problem, "  no anchor".into()));
        }
        out.extend(super::flow(&line, width, 5));
        for (n, s) in sentences.iter().enumerate() {
            if s.contains(&format!("[{k}]")) {
                placed[n] = true;
                out.extend(super::flow(&[(Style::Work, format!("     says: {s}"))], width, 11));
            }
        }
    }
    let unplaced: Vec<&String> = sentences.iter().zip(&placed).filter(|(_, p)| !**p).map(|(s, _)| s).collect();
    if !unplaced.is_empty() {
        out.extend(super::flow(&[(Style::Work, format!("says (back-translation)  {}", unplaced.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ")))], width, 2));
    } else if check.is_some() && back.is_none() {
        out.push(vec![(Style::Dim, "says    (back-translation not back yet)".into())]);
    }
    out.push(Vec::new());
    // The result.
    match &check {
        None => out.push(vec![(Style::Dim, "check   no spec yet".into())]),
        Some(check) => {
            out.push(vec![(Style::Status, "check   ".into()), (Style::Plain, format!("{} · {} · v{}{}", check.kind, check.lang, check.version, if check.by == "operator" { " · yours" } else { "" }))]);
            match check.kills {
                Some((k, t)) => out.push(vec![(Style::Status, "mutants ".into()), (if k == 0 { Style::Danger } else { Style::Plain }, format!("{k}/{t} killed")), (Style::Dim, if k == 0 { " · vacuous: counts as documented".into() } else { String::new() })]),
                None => out.push(vec![(Style::Dim, "mutants not run yet".into())]),
            }
            let readings: Vec<audit::Evidence> = store.as_ref().map(|s| s.evidence(&state.audit, Some(&claim.id)).unwrap_or_default().into_iter().filter(|e| e.kind == "reading").collect()).unwrap_or_default();
            for (i, name) in check.survived.iter().enumerate() {
                let here = state.mutant == Some(i);
                out.push(vec![(if here { Style::Cursor } else { Style::Danger }, format!("  {}[-] {name}", if here { "❯" } else { " " }))]);
                match readings.iter().rev().find(|r| r.name == *name) {
                    Some(reading) => out.extend(super::flow(&[(Style::Work, format!("      s0ma's reading: {}", reading.result))], width, 6)),
                    None => out.push(vec![(Style::Dim, "      no reading from s0ma yet: a gap, or outside the claim?".into())]),
                }
            }
            if check.kind == "tla" && claim.repro != Repro::Fails && claim.verdict == Verdict::Fails {
                out.push(vec![(Style::Danger, "model only: the code pane follows s0ma's mapping, not checked".into())]);
            }
        }
    }
    if !claim.rests_on.is_empty() {
        let status = store.as_ref().map(|s| claim.rests_on.iter().map(|a| s.claim(&state.audit, a).ok().flatten().map(|c| format!("{a} {}", audit_scope_status(s, &c))).unwrap_or_else(|| a.clone())).collect::<Vec<_>>().join(" · "));
        text(&mut out, Style::Plain, format!("assumes {}", status.unwrap_or_default()));
    }
    if let Some(test) = &claim.repro_test {
        let said = match claim.repro {
            Repro::Fails => format!("fails on the real code: {}", crate::util::clip(claim.repro_output.as_deref().unwrap_or("").lines().last().unwrap_or(""), 60)),
            Repro::Passes => "passes on the real code: not a repro".into(),
            _ => "waiting to run".into(),
        };
        text(&mut out, if claim.repro == Repro::Fails { Style::Danger } else { Style::Plain }, format!("repro   {} · {said}", test.split(" :: ").next().unwrap_or(test)));
    }
    if let Some(why) = &claim.model_only_why {
        text(&mut out, Style::Danger, format!("model-only: {why}"));
    }
    if let Some(fact) = &claim.fact {
        let (board, lead) = claim.lead.clone().unwrap_or_default();
        out.push(vec![(Style::Status, "board   ".into()), gui_view::hot(Style::Accent, format!("{fact}{}", if board.is_empty() { String::new() } else { format!(" on {board} {lead}") }), Click::Pick(Pick::ClaimBoard))]);
    }
    // The counterexample.
    if !state.steps.is_empty() {
        out.push(Vec::new());
        out.push(vec![(Style::Status, "counterexample".into()), (Style::Dim, format!("  {} steps · [ ] step through", state.steps.len()))]);
        for (i, step) in state.steps.iter().enumerate() {
            let here = state.step == Some(i);
            out.push(vec![(if here { Style::Cursor } else { Style::Plain }, format!("{}{:>2} ", if here { "[*]" } else { "   " }, i + 1)), (if here { Style::Bold } else { Style::Plain }, crate::util::clip(&step.label, width.saturating_sub(6)))]);
            if here {
                for (n, var) in step.state.iter().enumerate() {
                    out.push(vec![(if step.changed.contains(&n) { Style::Bold } else { Style::Dim }, format!("     {var}"))]);
                }
            }
        }
    }
    out
}

fn audit_scope_status(store: &audit::AuditStore, claim: &Claim) -> String {
    crate::audit::scope::assumption_status(store, claim)
}

/// Code first: the claims about `file`, by the line they cover.
fn claims_by_line(state: &State) -> HashMap<usize, Vec<String>> {
    let mut out: HashMap<usize, Vec<String>> = HashMap::new();
    let Some(store) = audit::store() else { return out };
    let Some(file) = &state.file else { return out };
    let lines = state.lines();
    for claim in store.claims(&state.audit).unwrap_or_default() {
        let mut at: Vec<usize> = Vec::new();
        if let Ok(Some(check)) = store.current_check(&state.audit, &claim.id) {
            for target in &check.targets {
                if let Some((path, symbol)) = target.split_once(':')
                    && path == file
                    && let Some(n) = definition(lines, symbol.rsplit("::").next().unwrap_or(symbol))
                {
                    at.push(n);
                }
            }
        }
        if let Some((path, n)) = claim.source.split_once(':')
            && path == file
            && let Some(n) = n.split(|c: char| !c.is_ascii_digit()).next().and_then(|n| n.parse::<usize>().ok())
        {
            at.push(n.saturating_sub(1));
        }
        for n in at {
            out.entry(n).or_default().push(claim.id.clone());
        }
    }
    out
}

/// The code pane: `rows` lines from its scroll, with the gutter.
fn code_pane(state: &mut State, width: usize, rows: usize) -> Vec<Line> {
    let mut out: Vec<Line> = Vec::new();
    let Some(file) = state.file.clone() else {
        out.push(vec![(Style::Dim, if state.loaded.is_none() { "reading the work tree…".into() } else { "no code file for this claim (its check names its targets)".to_string() })]);
        return out;
    };
    let lines: Vec<String> = state.lines().to_vec();
    let loaded = state.loaded.clone().unwrap_or_default();
    let audit = audit::store().and_then(|s| s.audit(&state.audit).ok().flatten());
    let in_scope = audit.as_ref().is_none_or(|a| {
        let paths: Vec<&str> = a.scope.iter().filter_map(|s| s.strip_prefix("path:")).collect();
        paths.is_empty() || paths.iter().any(|p| glob_match(p, &file))
    });
    let covered = loaded.coverage.get(&file);
    let lit: Vec<usize> = state.clause.map(|k| clause_code_lines(state, k)).unwrap_or_default();
    let step = step_line(state);
    let gutter_claims = if state.claim.is_none() { claims_by_line(state) } else { HashMap::new() };
    // Calls out of the slice, with the assumption made about each.
    let assumptions: Vec<(String, String)> = audit::store()
        .map(|s| s.claims(&state.audit).unwrap_or_default().into_iter().filter(|c| c.is_assumption()).filter_map(|c| assumed_symbol(&c.text).map(|sym| (sym, c.id))).collect())
        .unwrap_or_default();
    let mapped = step.is_some_and(|(_, mapped)| mapped);
    let mut head: Line = vec![(Style::Status, file.to_string()), (Style::Dim, format!("  {} lines", lines.len()))];
    if !in_scope {
        head.push((Style::Dim, " · outside the scope".into()));
    }
    if mapped {
        head.push((Style::Danger, " · [*] s0ma's mapping, not checked".into()));
    }
    if loaded.files.len() > 1 {
        head.push((Style::Dim, format!(" · f: {} files", loaded.files.len())));
    }
    out.push(head);
    let rows = rows.saturating_sub(1);
    if state.cursor < state.scroll[1] {
        state.scroll[1] = state.cursor;
    }
    if state.cursor >= state.scroll[1] + rows {
        state.scroll[1] = state.cursor + 1 - rows;
    }
    let number_width = lines.len().to_string().len().max(3);
    let language = crate::editor::highlight::language(&file);
    let shown_mutant = current_mutant(state).map(str::to_string);
    let has_mutant = state.mutant.is_some();
    let highlighter = state.highlight.entry(file.clone()).or_insert_with(|| Highlighter::new(&language, lines.iter().map(String::len).sum()));
    for n in state.scroll[1]..lines.len().min(state.scroll[1] + rows) {
        let here = n == state.cursor && state.pane == 1;
        let mut line: Line = vec![(if here || lit.contains(&n) { Style::Cursor } else { Style::Dim }, format!("{:>number_width$} ", n + 1))];
        // The gutter: one mark, the strongest.
        let survived = loaded.mutants.iter().position(|m| m.file == file && m.line == n && !m.killed);
        let killed = loaded.mutants.iter().any(|m| m.file == file && m.line == n && m.killed);
        let call_out = assumptions.iter().find(|(sym, _)| lines[n].contains(&format!("{sym}(")));
        let (mark, style) = if step.is_some_and(|(at, _)| at == n) {
            ("[*]".to_string(), Style::Warn)
        } else if let Some(ids) = gutter_claims.get(&n) {
            (format!("{} ", ids.join(",")), Style::Accent)
        } else if survived.is_some() {
            ("[-]".into(), Style::Danger)
        } else if let Some((_, id)) = call_out {
            (format!("{id} "), Style::Operator)
        } else if killed {
            ("[+]".into(), Style::Success)
        } else if lit.contains(&n) {
            (format!("[{}]", state.clause.unwrap_or(0)), Style::Cursor)
        } else if let Some(covered) = covered {
            ("│ ".into(), if covered.contains(&n) { Style::Success } else { Style::Dim })
        } else if in_scope {
            ("│ ".into(), Style::Dim)
        } else {
            ("  ".into(), Style::Dim)
        };
        let mark = format!("{mark:<5}");
        line.push(gui_view::hot(style, mark, Click::Pick(Pick::CodeLine(n))));
        let text = &lines[n];
        if !in_scope {
            line.push((Style::Dim, text.clone()));
        } else {
            let spans = highlighter.line(&lines, n).to_vec();
            if spans.is_empty() {
                line.push((Style::Plain, text.clone()));
            }
            for (start, end, class) in spans {
                if let Some(piece) = text.get(start..end) {
                    line.push((if here { Style::Bold } else { class_style(class) }, piece.to_string()));
                }
            }
        }
        if let Some(i) = survived {
            line.push((Style::Danger, format!("   [-] {}", loaded.mutants[i].change)));
        } else if has_mutant && loaded.mutants.iter().any(|m| m.file == file && m.line == n && m.killed && Some(&m.name) == shown_mutant.as_ref()) {
            line.push((Style::Success, "   [+] killed".into()));
        }
        out.push(fit(&line, width));
    }
    out
}

fn current_mutant(state: &State) -> Option<&str> {
    let i = state.mutant?;
    state.loaded.as_ref()?.mutants.iter().filter(|m| !m.killed).nth(i).map(|m| m.name.as_str())
}

/// What an assumption is about: its first `a::b` or `name(` symbol, by its
/// last segment (what a call names).
fn assumed_symbol(text: &str) -> Option<String> {
    let word = text.split(|c: char| c.is_whitespace() || c == '`').find(|w| w.contains("::") || w.ends_with("()"))?;
    let last = word.trim_end_matches("()").rsplit("::").next()?.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
    (!last.is_empty()).then(|| last.to_string())
}

fn glob_match(pattern: &str, path: &str) -> bool {
    if !pattern.contains('*') {
        return path == pattern || path.starts_with(&format!("{}/", pattern.trim_end_matches('/')));
    }
    let prefix = pattern.split('*').next().unwrap_or("");
    path.starts_with(prefix)
}

/// The spec pane, whole: every file of the check, its anchors lit.
fn spec_pane(state: &mut State, claim: Option<&Claim>, width: usize) -> Vec<Line> {
    let mut out: Vec<Line> = Vec::new();
    let Some(loaded) = state.loaded.clone() else {
        out.push(vec![(Style::Dim, "reading the work tree…".into())]);
        return out;
    };
    if loaded.spec.is_empty() {
        out.push(vec![(Style::Dim, "no spec yet".into())]);
        if claim.is_some_and(|c| c.taken) {
            out.extend(super::flow(&[(Style::Dim, "Yours: e writes its check.toml in the editor, s submits it to the harness.".to_string())], width, 0));
        }
        return out;
    }
    let taken = claim.is_some_and(|c| c.taken);
    out.push(vec![(Style::Dim, if taken { "yours · e edits it, s submits it".into() } else { "read-only · t takes the claim to edit it".to_string() })]);
    let step_action = state.step.and_then(|i| state.steps.get(i)).and_then(|s| s.action.clone());
    let step_def = |lines: &[String]| step_action.as_deref().and_then(|a| definition(lines, a));
    for (name, lines) in &loaded.spec {
        out.push(vec![(Style::Status, format!("── {name}"))]);
        let at = step_def(lines);
        let language = crate::editor::highlight::language(name);
        let key = format!("spec:{name}");
        let highlighter = state.highlight.entry(key).or_insert_with(|| Highlighter::new(&language, lines.iter().map(String::len).sum()));
        for (n, text) in lines.iter().enumerate() {
            let anchors = anchors(text);
            let lit = state.clause.is_some_and(|k| anchors.contains(&k));
            let stepped = at == Some(n);
            let mark = if stepped {
                ("[*]".to_string(), Style::Warn)
            } else if lit {
                (format!("[{}]", state.clause.unwrap_or(0)), Style::Cursor)
            } else if let Some(k) = anchors.first() {
                (format!("{k} "), Style::Dim)
            } else {
                ("  ".into(), Style::Dim)
            };
            let mut line: Line = vec![(mark.1, format!(" {:<4}", mark.0.trim_end()))];
            let spans = highlighter.line(lines, n).to_vec();
            if spans.is_empty() || lit {
                line.push((if lit { Style::Bold } else { Style::Plain }, text.clone()));
            } else {
                for (start, end, class) in spans {
                    if let Some(piece) = text.get(start..end) {
                        line.push((class_style(class), piece.to_string()));
                    }
                }
            }
            if code_mapping(text).is_some() {
                line.push((Style::Dim, "  (s0ma's mapping)".into()));
            }
            out.push(line);
        }
    }
    out
}

pub(super) fn body(tab: &mut Open, width: usize, rows: usize) -> (Vec<Line>, Footer) {
    let Some(state) = tab.audit.view.as_mut() else { return (Vec::new(), Footer::new(Vec::new(), Vec::new())) };
    if let Some(problem) = state.poll() {
        tab.note(problem, true);
    }
    let state = tab.audit.view.as_mut().expect("open");
    let claim = state.claim_now();
    let wide = width >= WIDE;
    let mut out: Vec<Line> = Vec::new();
    // The bar: the panes, by number, clickable.
    let mut bar: Line = vec![(Style::Status, format!("  {}", state.audit)), (Style::Dim, format!("/{}  ", state.claim.as_deref().unwrap_or("code")))];
    for (i, name) in PANES.iter().enumerate() {
        let look = if state.pane == i { Style::Tab } else { Style::Plain };
        bar.push(gui_view::hot(look, format!(" {} {name} ", i + 1), Click::Pick(Pick::ClaimPane(i))));
        bar.push((Style::Plain, " ".into()));
    }
    if state.loading.is_some() || state.jumping.is_some() {
        bar.push((Style::Warn, " [*] reading".into()));
    }
    out.push(bar);
    let room = rows.saturating_sub(1);
    if wide {
        let w0 = width * 3 / 10;
        let w1 = width * 4 / 10;
        let w2 = width.saturating_sub(w0 + w1 + 6);
        let left = claim.as_ref().map(|c| claim_pane(state, c, w0)).unwrap_or_else(|| code_first_list(state));
        let middle = code_pane(state, w1, room);
        let right = spec_pane(state, claim.as_ref(), w2);
        state.scroll[0] = state.scroll[0].min(left.len().saturating_sub(1));
        state.scroll[2] = state.scroll[2].min(right.len().saturating_sub(1));
        for r in 0..room {
            let mut line: Line = Vec::new();
            line.push((Style::Plain, " ".into()));
            line.extend(fit(left.get(state.scroll[0] + r).unwrap_or(&Vec::new()), w0));
            line.push((if state.pane == 1 { Style::Cursor } else { Style::Dim }, " │ ".into()));
            line.extend(fit(middle.get(r).unwrap_or(&Vec::new()), w1));
            line.push((if state.pane == 2 { Style::Cursor } else { Style::Dim }, " │ ".into()));
            line.extend(fit(right.get(state.scroll[2] + r).unwrap_or(&Vec::new()), w2));
            out.push(line);
        }
    } else {
        let inner = width.saturating_sub(2);
        let pane: Vec<Line> = match state.pane {
            0 => claim.as_ref().map(|c| claim_pane(state, c, inner)).unwrap_or_else(|| code_first_list(state)),
            1 => code_pane(state, inner, room),
            _ => spec_pane(state, claim.as_ref(), inner),
        };
        let scroll = if state.pane == 1 { 0 } else { state.scroll[state.pane].min(pane.len().saturating_sub(1)) };
        if state.pane != 1 {
            state.scroll[state.pane] = scroll;
        }
        for line in pane.into_iter().skip(scroll).take(room) {
            let mut row: Line = vec![(Style::Plain, " ".into())];
            row.extend(line);
            out.push(row);
        }
    }
    out.truncate(rows);
    let state = tab.audit.view.as_ref().expect("open");
    let mut hints: Vec<(&'static str, &'static str)> = vec![("1", "claim"), ("2", "code"), ("3", "spec"), ("↑↓", "scroll")];
    if claim.is_some() {
        hints.push(("←→", "clause"));
    }
    if state.pane == 1 {
        hints.push(("⏎", "definition"));
        if !state.back.is_empty() {
            hints.push(("^o", "back"));
        }
        if state.loaded.as_ref().is_some_and(|l| l.files.len() > 1) {
            hints.push(("f", "next file"));
        }
        if state.claim.is_none() {
            hints.push(("o", "open claim"));
        }
    }
    if !state.steps.is_empty() {
        hints.push(("[ ]", "step"));
    }
    if state.loaded.as_ref().is_some_and(|l| l.mutants.iter().any(|m| !m.killed)) {
        hints.push(("m", "survived mutant"));
    }
    if let Some(c) = &claim {
        if c.taken {
            hints.extend([("e", "edit spec"), ("s", "submit")]);
        } else {
            hints.push(("t", "take"));
        }
    }
    hints.push(("esc", "back"));
    let left: Line = vec![(Style::Status, "claim view".into())];
    (out, Footer::new(left, hints))
}

/// Code first, the left pane: the claims about the file, by line.
fn code_first_list(state: &State) -> Vec<Line> {
    let mut out: Vec<Line> = vec![(vec![(Style::Status, "claims about this file".into())])];
    let mut by_line: Vec<(usize, Vec<String>)> = claims_by_line(state).into_iter().collect();
    by_line.sort();
    if by_line.is_empty() {
        out.push(vec![(Style::Dim, "none yet".into())]);
    }
    let store = audit::store();
    for (n, ids) in by_line {
        for id in ids {
            let text = store.as_ref().and_then(|s| s.claim(&state.audit, &id).ok().flatten()).map(|c| format!("{} {}", c.mark().0, c.text)).unwrap_or_default();
            out.push(vec![gui_view::hot(Style::Accent, format!("{id:<5}"), Click::Pick(Pick::CodeLine(n))), (Style::Dim, format!("l{:<5} ", n + 1)), (Style::Plain, text)]);
        }
    }
    out
}

// ------------------------------------------------------------------ keys

fn scroll(state: &mut State, by: isize) {
    if state.pane == 1 {
        let len = state.lines().len();
        state.cursor = (state.cursor as isize + by).clamp(0, len.saturating_sub(1) as isize) as usize;
    } else {
        state.scroll[state.pane] = (state.scroll[state.pane] as isize + by).max(0) as usize;
    }
}

/// ⏎ on a call in the code pane: to its definition, here or found in the
/// work tree.
fn jump(state: &mut State) -> Option<String> {
    let lines = state.lines().to_vec();
    let line = lines.get(state.cursor)?.clone();
    let file = state.file.clone()?;
    for name in calls(&line) {
        if let Some(at) = definition(&lines, &name)
            && at != state.cursor
        {
            state.back.push((file, state.cursor));
            state.go(&state.file.clone().unwrap_or_default(), at);
            return None;
        }
        if let Some(loaded) = &state.loaded {
            for (other, other_lines) in &loaded.files {
                if *other != file
                    && let Some(at) = definition(other_lines, &name)
                {
                    state.back.push((file, state.cursor));
                    let other = other.clone();
                    state.go(&other, at);
                    return None;
                }
            }
        }
    }
    let name = calls(&line).into_iter().next()?;
    if cfg!(test) {
        return Some(format!("{name} is not defined in the files read"));
    }
    let work = audit::store()?.audit(&state.audit).ok().flatten()?.work;
    let slot: Slot<(String, Vec<String>, usize)> = Arc::new(Mutex::new(None));
    state.jumping = Some(slot.clone());
    state.back.push((file, state.cursor));
    let _ = std::thread::Builder::new().name("claim-view-def".into()).spawn(move || {
        let q = crate::util::shell_quote;
        let script = format!(
            "cd {} || exit 3; f=$(grep -rlE '(fn|def|function|func|class|struct|enum|trait|type) {n}\\b' --exclude-dir=.git --exclude-dir=target --exclude-dir=node_modules --exclude-dir=checks . 2>/dev/null | head -1); [ -n \"$f\" ] || exit 4; printf '%s\\n' \"${{f#./}}\"; head -c 400000 \"$f\"",
            q(&work),
            n = name
        );
        let result = crate::vm::netboxes::machine(&script, 30).and_then(|ran| {
            if ran.exit_code != Some(0) {
                return Err(format!("no definition of {name} found in the work tree"));
            }
            let mut lines = ran.stdout.lines();
            let file = lines.next().unwrap_or("").to_string();
            let body: Vec<String> = lines.map(str::to_string).collect();
            let at = definition(&body, &name).unwrap_or(0);
            Ok((file, body, at))
        });
        *slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(result);
    });
    None
}

/// A key in the claim view; true closes it (back to the list).
pub(super) fn key(tab: &mut Open, key: &KeyEvent) -> bool {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let Some(state) = tab.audit.view.as_mut() else { return true };
    let claim = state.claim_now();
    let clauses = claim.as_ref().map(|c| audit::clauses(&c.text).len()).unwrap_or(0);
    let mut said: Option<std::result::Result<String, String>> = None;
    match key.code {
        KeyCode::Esc => return true,
        KeyCode::Tab => state.pane = (state.pane + 1) % 3,
        KeyCode::BackTab => state.pane = (state.pane + 2) % 3,
        KeyCode::Char(d @ '1'..='3') => state.pane = d as usize - '1' as usize,
        KeyCode::Up => scroll(state, -1),
        KeyCode::Down => scroll(state, 1),
        KeyCode::PageUp => scroll(state, -15),
        KeyCode::PageDown => scroll(state, 15),
        KeyCode::Home if state.pane == 1 => state.cursor = 0,
        KeyCode::End if state.pane == 1 => state.cursor = state.lines().len().saturating_sub(1),
        KeyCode::Left if clauses > 0 => state.clause = Some(state.clause.map_or(clauses, |k| if k <= 1 { clauses } else { k - 1 })),
        KeyCode::Right if clauses > 0 => state.clause = Some(state.clause.map_or(1, |k| if k >= clauses { 1 } else { k + 1 })),
        KeyCode::Char('o') if ctrl => {
            if let Some((file, line)) = state.back.pop() {
                state.go(&file, line);
            }
        }
        KeyCode::Enter if state.pane == 1 => {
            if let Some(problem) = jump(state) {
                said = Some(Err(problem));
            }
        }
        KeyCode::Char('o') if state.claim.is_none() => {
            let at = state.cursor;
            if let Some(id) = claims_by_line(state).get(&at).and_then(|ids| ids.first()).cloned()
                && let Some(store) = audit::store()
                && let (Ok(Some(audit)), Ok(Some(claim))) = (store.audit(&state.audit), store.claim(&state.audit, &id))
            {
                *state = State::open(&audit, &claim);
            }
        }
        KeyCode::Char('f') => {
            if let Some(loaded) = &state.loaded
                && !loaded.files.is_empty()
            {
                let at = loaded.files.iter().position(|(f, _)| Some(f) == state.file.as_ref()).map_or(0, |i| (i + 1) % loaded.files.len());
                let file = loaded.files[at].0.clone();
                state.go(&file, 0);
            }
        }
        KeyCode::Char('[') | KeyCode::Char(']') if !state.steps.is_empty() => {
            let last = state.steps.len() - 1;
            state.step = Some(match (key.code, state.step) {
                (KeyCode::Char(']'), None) => 0,
                (KeyCode::Char('['), None) => last,
                (KeyCode::Char(']'), Some(s)) => (s + 1).min(last),
                (_, Some(s)) => s.saturating_sub(1),
                _ => 0,
            });
            if let Some(step) = state.steps.get(state.step.unwrap_or(0)).cloned() {
                if let Some((file, line)) = step.at {
                    if state.loaded.as_ref().is_some_and(|l| l.files.iter().any(|(f, _)| *f == file)) {
                        state.go(&file, line);
                    }
                } else if let Some((line, _)) = step_line(state) {
                    state.cursor = line;
                }
            }
        }
        KeyCode::Char('m') => {
            let survived: Vec<Mutant> = state.loaded.as_ref().map(|l| l.mutants.iter().filter(|m| !m.killed).cloned().collect()).unwrap_or_default();
            if survived.is_empty() {
                said = Some(Err("no mutant survived".into()));
            } else {
                let next = state.mutant.map_or(0, |i| (i + 1) % survived.len());
                state.mutant = Some(next);
                let m = &survived[next];
                if state.loaded.as_ref().is_some_and(|l| l.files.iter().any(|(f, _)| *f == m.file)) {
                    let (file, line) = (m.file.clone(), m.line);
                    state.go(&file, line);
                } else {
                    said = Some(Err(format!("{} changes {}, which was not read", m.name, m.file)));
                }
            }
        }
        KeyCode::Char('t') => {
            if let (Some(c), Some(store)) = (&claim, audit::store()) {
                said = Some(crate::audit::ops::operator_claim(&store, &state.audit, &c.id, "take", ""));
            }
        }
        KeyCode::Char('e') if claim.as_ref().is_some_and(|c| c.taken) => {
            let c = claim.clone().expect("claim");
            if let Some(audit) = audit::store().and_then(|s| s.audit(&state.audit).ok().flatten()) {
                let path = state.loaded.as_ref().and_then(|l| l.spec.iter().find(|(n, _)| n == "check.toml").or(l.spec.first())).map(|(n, _)| format!("checks/{}/{n}", c.id)).unwrap_or_else(|| format!("checks/{}/check.toml", c.id));
                let text = state
                    .loaded
                    .as_ref()
                    .and_then(|l| l.spec.iter().find(|(n, _)| path.ends_with(&format!("/{n}"))))
                    .map(|(_, lines)| lines.join("\n") + "\n")
                    .unwrap_or_else(|| format!("claim   = \"{}\"\nlang    = \"\"\nkind    = \"test\"\nrun     = \"\"\ntimeout = 600\ntargets = []\n", c.id));
                if !cfg!(test) {
                    let _ = crate::vm::netboxes::machine(&format!("mkdir -p {}", crate::util::shell_quote(&format!("{}/checks/{}", audit.work, c.id))), 20);
                }
                super::editor_view::open(&audit.work, &path, &text, None);
                let _ = super::go_to(tab, super::Target::View(super::View::Edit));
                return false;
            }
        }
        KeyCode::Char('s') if claim.as_ref().is_some_and(|c| c.taken) => {
            if let (Some(c), Some(store)) = (&claim, audit::store()) {
                said = Some(crate::audit::run::submit_check(&store, &state.audit, &c.id, &format!("checks/{}", c.id), "operator"));
            }
        }
        _ => {}
    }
    if let Some(result) = said {
        super::audit_view::note(tab, result);
    }
    false
}

/// A click on a pane's name, a clause, or a code line.
pub(super) fn pick_pane(tab: &mut Open, pane: usize) {
    if let Some(state) = tab.audit.view.as_mut() {
        state.pane = pane.min(2);
    }
}

pub(super) fn pick_clause(tab: &mut Open, k: usize) {
    if let Some(state) = tab.audit.view.as_mut() {
        state.clause = if state.clause == Some(k) { None } else { Some(k) };
    }
}

pub(super) fn pick_line(tab: &mut Open, line: usize) {
    if let Some(state) = tab.audit.view.as_mut() {
        state.cursor = line;
        state.pane = 1;
    }
}

pub(super) fn wheel(tab: &mut Open, up: bool) {
    if let Some(state) = tab.audit.view.as_mut() {
        scroll(state, if up { -3 } else { 3 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tlc_trace_becomes_steps_with_what_changed() {
        let trace = "State 1: <Initial predicate>\n/\\ file = \"v1\"\n/\\ pending = FALSE\nState 2: <Download line 40, col 5>\n/\\ file = \"v1\"\n/\\ pending = TRUE\n";
        let steps = parse_trace(trace);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[1].action.as_deref(), Some("Download"));
        assert_eq!(steps[1].changed, vec![1], "pending changed");
        assert_eq!(steps[0].action, None, "the initial state is no action");
    }
}
