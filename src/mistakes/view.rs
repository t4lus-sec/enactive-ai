//! The review window (`claude/mistake-capture-spec.md` §5): `/mistake view [n]`
//! at the idle prompt, or `s0ma --mistake-view [n]` in a terminal of its own.
//!
//! It shows a mistake's original response beside the version up for approval,
//! one section per field (reasoning, reply, each tool call paired by id), only
//! the changed hunks, with word-level highlighting. The diff is computed here
//! from the two responses, never from a proposal's edit list, so a manual
//! version is shown the same way. Keys: `a` approve, `e` edit by hand, `d`
//! discard, `r` regenerate, `c` the context, `n`/`p` next/previous, `q` close.
//!
//! Nothing here runs a tool call: a corrected call is text (R2).

use crate::style::{apply_style, width_of, wrap, Line, Style};
use crate::mistakes::{self, Book, Record, Response, Split, State, Version};
use crate::types::{FunctionCall, ToolCall};
use serde_json::Value;
use similar::{ChangeTag, TextDiff};
use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;
use unicode_width::UnicodeWidthChar;

/// Side by side from this many columns; stacked below it.
const SIDE_BY_SIDE_COLUMNS: usize = 160;
const CONTEXT_LINES: usize = 3;
const CONTEXT_MESSAGES: usize = 6;

// ------------------------------------------------------------------ the editable form

const SECTION_REASONING: &str = "reasoning";
const SECTION_REPLY: &str = "reply";

fn marker(nonce: &str, inner: &str) -> String {
    if nonce.is_empty() {
        format!("=== {inner} ===")
    } else {
        format!("=== [{nonce}] {inner} ===")
    }
}

fn looks_like_marker(line: &str) -> bool {
    line.starts_with("===") && line.trim_end().ends_with("===")
}

/// The response as plain text, one marked section per field, so nothing has to
/// be escaped. When a field itself holds a line that looks like a marker, the
/// markers carry a nonce that the text cannot contain.
pub fn to_form(response: &Response) -> String {
    let texts = std::iter::once(response.reasoning_content.as_str())
        .chain(std::iter::once(response.content.as_str()))
        .chain(response.tool_calls.iter().map(|c| c.function.arguments.as_str()))
        .collect::<Vec<_>>();
    let mut nonce = String::new();
    if texts.iter().any(|t| t.lines().any(looks_like_marker)) {
        let mut seed = 0u64;
        loop {
            let candidate = format!("{:08x}", fold_hash(&format!("{seed}{}", response.content.len())));
            if !texts.iter().any(|t| t.contains(&candidate)) {
                nonce = candidate;
                break;
            }
            seed += 1;
        }
    }
    let mut form = String::new();
    let mut section = |inner: String, text: &str| {
        form.push_str(&marker(&nonce, &inner));
        form.push('\n');
        form.push_str(text);
        form.push('\n');
    };
    section(SECTION_REASONING.into(), &response.reasoning_content);
    section(SECTION_REPLY.into(), &response.content);
    for (index, call) in response.tool_calls.iter().enumerate() {
        section(format!("tool_call {}: {} (id {})", index + 1, call.function.name, call.id), &call.function.arguments);
    }
    form
}

fn fold_hash(text: &str) -> u32 {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(text.as_bytes());
    u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]])
}

#[derive(Debug, PartialEq, Eq)]
pub struct FormError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for FormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

enum Section {
    Reasoning,
    Reply,
    ToolCall { name: String, id: Option<String> },
}

/// Parse the form back. Unchanged, it reproduces the response exactly. A
/// tool-call section added with `(id new)` gets `fresh_id(index)`; removing a
/// section removes the call. Every tool call's arguments must be JSON.
pub fn from_form(form: &str, mut fresh_id: impl FnMut(usize) -> String) -> Result<Response, FormError> {
    let lines: Vec<&str> = form.split('\n').collect();
    let first = lines.first().copied().unwrap_or("");
    let nonce = first
        .strip_prefix("=== [")
        .and_then(|rest| rest.split_once(']'))
        .map(|(nonce, _)| nonce.to_owned())
        .filter(|n| n.len() == 8 && n.bytes().all(|b| b.is_ascii_hexdigit()))
        .unwrap_or_default();
    let prefix = if nonce.is_empty() { "=== ".to_owned() } else { format!("=== [{nonce}] ") };
    let parse_marker = |line: &str| -> Option<String> {
        let inner = line.strip_prefix(prefix.as_str())?.strip_suffix(" ===")?;
        if nonce.is_empty() && inner.starts_with('[') {
            return None;
        }
        Some(inner.to_owned())
    };
    let mut sections: Vec<(usize, Section, Vec<&str>)> = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if let Some(inner) = parse_marker(line) {
            let section = if inner == SECTION_REASONING {
                Section::Reasoning
            } else if inner == SECTION_REPLY {
                Section::Reply
            } else if let Some(rest) = inner.strip_prefix("tool_call ") {
                let (_, rest) = rest.split_once(": ").ok_or_else(|| FormError {
                    line: index + 1,
                    message: format!("expected `tool_call N: name (id …)`, got {inner:?}"),
                })?;
                let (name, id) = match rest.rsplit_once(" (id ") {
                    Some((name, id)) => (name.trim(), id.strip_suffix(')').map(str::trim)),
                    None => (rest.trim(), None),
                };
                let id = match id {
                    None | Some("new") | Some("") => None,
                    Some(id) => Some(id.to_owned()),
                };
                if name.is_empty() {
                    return Err(FormError { line: index + 1, message: "a tool call needs a name".into() });
                }
                Section::ToolCall { name: name.to_owned(), id }
            } else {
                return Err(FormError {
                    line: index + 1,
                    message: format!("unknown section {inner:?} (reasoning, reply or tool_call N: name (id …))"),
                });
            };
            sections.push((index + 1, section, Vec::new()));
        } else {
            match sections.last_mut() {
                Some((_, _, body)) => body.push(line),
                None if line.trim().is_empty() => {}
                None => {
                    return Err(FormError { line: index + 1, message: "text before the first section marker".into() })
                }
            }
        }
    }
    // Each section's text is followed by one newline: before the next marker
    // that newline is the split's, and at the end of the file it leaves one
    // empty element in the last section, which is the form's, not the field's.
    // (A file whose final newline an editor removed has none.)
    if form.ends_with('\n')
        && let Some((_, _, body)) = sections.last_mut() {
            body.pop();
        }
    let mut reasoning = None;
    let mut reply = None;
    let mut tool_calls = Vec::new();
    for (line, section, body) in sections {
        let text = body.join("\n");
        match section {
            Section::Reasoning => {
                if reasoning.replace(text).is_some() {
                    return Err(FormError { line, message: "a second reasoning section".into() });
                }
            }
            Section::Reply => {
                if reply.replace(text).is_some() {
                    return Err(FormError { line, message: "a second reply section".into() });
                }
            }
            Section::ToolCall { name, id } => {
                if let Err(error) = serde_json::from_str::<Value>(&text) {
                    return Err(FormError {
                        line: line + error.line(),
                        message: format!("the arguments of tool call {} ({name}) are not JSON: {error}", tool_calls.len() + 1),
                    });
                }
                let id = id.unwrap_or_else(|| fresh_id(tool_calls.len()));
                tool_calls.push(ToolCall {
                    id,
                    kind: "function".into(),
                    function: FunctionCall { name, arguments: text },
                });
            }
        }
    }
    Ok(Response {
        reasoning_content: reasoning.ok_or(FormError { line: 1, message: "the reasoning section is missing".into() })?,
        content: reply.ok_or(FormError { line: 1, message: "the reply section is missing".into() })?,
        tool_calls,
    })
}

// ------------------------------------------------------------------ the diff

/// One line of a field: its text in segments, `true` where the word-level diff
/// marks a change.
pub type Segments = Vec<(bool, String)>;

#[derive(Debug, Clone)]
pub enum Row {
    Same(String),
    Changed { old: Option<Segments>, new: Option<Segments> },
    Gap,
}

#[derive(Debug, Clone)]
pub struct FieldDiff {
    pub title: String,
    pub unchanged_chars: Option<usize>,
    pub rows: Vec<Row>,
    pub hunks: usize,
}

pub(crate) fn text_diff(title: String, old: &str, new: &str) -> FieldDiff {
    if old == new {
        return FieldDiff { title, unchanged_chars: Some(old.chars().count()), rows: Vec::new(), hunks: 0 };
    }
    let diff = TextDiff::from_lines(old, new);
    let mut rows = Vec::new();
    let groups = diff.grouped_ops(CONTEXT_LINES);
    let hunks = groups.len();
    for (index, group) in groups.iter().enumerate() {
        if index > 0 {
            rows.push(Row::Gap);
        }
        let mut deleted: Vec<Segments> = Vec::new();
        let mut inserted: Vec<Segments> = Vec::new();
        let flush = |rows: &mut Vec<Row>, deleted: &mut Vec<Segments>, inserted: &mut Vec<Segments>| {
            let pairs = deleted.len().max(inserted.len());
            let mut d = deleted.drain(..);
            let mut i = inserted.drain(..);
            for _ in 0..pairs {
                rows.push(Row::Changed { old: d.next(), new: i.next() });
            }
        };
        for op in group {
            for change in diff.iter_inline_changes(op) {
                let segments: Segments = change
                    .iter_strings_lossy()
                    .map(|(emph, text)| (emph, text.trim_end_matches('\n').to_owned()))
                    .filter(|(_, text)| !text.is_empty())
                    .collect();
                match change.tag() {
                    ChangeTag::Equal => {
                        flush(&mut rows, &mut deleted, &mut inserted);
                        rows.push(Row::Same(segments.into_iter().map(|(_, t)| t).collect()));
                    }
                    ChangeTag::Delete => {
                        if !inserted.is_empty() {
                            flush(&mut rows, &mut deleted, &mut inserted);
                        }
                        deleted.push(segments);
                    }
                    ChangeTag::Insert => inserted.push(segments),
                }
            }
        }
        flush(&mut rows, &mut deleted, &mut inserted);
    }
    FieldDiff { title, unchanged_chars: None, rows, hunks }
}

/// One section per field: reasoning, reply, then tool calls paired by id
/// (removed ones, then added ones).
pub fn field_diffs(original: &Response, shown: &Response) -> Vec<FieldDiff> {
    let mut out = vec![
        text_diff("reasoning".into(), &original.reasoning_content, &shown.reasoning_content),
        text_diff("reply".into(), &original.content, &shown.content),
    ];
    for call in &original.tool_calls {
        match shown.tool_calls.iter().find(|c| c.id == call.id) {
            Some(new) => {
                let title = if new.function.name == call.function.name {
                    format!("tool call {} ({})", call.function.name, call.id)
                } else {
                    format!("tool call {} → {} ({})", call.function.name, new.function.name, call.id)
                };
                let mut diff = text_diff(title, &call.function.arguments, &new.function.arguments);
                if new.function.name != call.function.name && diff.unchanged_chars.is_some() {
                    diff.unchanged_chars = None;
                    diff.hunks = 1;
                    diff.rows = vec![Row::Changed {
                        old: Some(vec![(true, call.function.name.clone())]),
                        new: Some(vec![(true, new.function.name.clone())]),
                    }];
                }
                out.push(diff);
            }
            None => {
                let mut diff = text_diff(format!("tool call {} ({}) removed", call.function.name, call.id), &call.function.arguments, "");
                diff.hunks = diff.hunks.max(1);
                out.push(diff);
            }
        }
    }
    for call in shown.tool_calls.iter().filter(|c| !original.tool_calls.iter().any(|o| o.id == c.id)) {
        let mut diff = text_diff(format!("tool call {} ({}) added", call.function.name, call.id), "", &call.function.arguments);
        diff.hunks = diff.hunks.max(1);
        out.push(diff);
    }
    out
}

// ------------------------------------------------------------------ hindsight check

fn fact_like(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() < 2 {
        return false;
    }
    let inner_dot = chars.windows(3).any(|w| w[1] == '.' && w[0].is_alphanumeric() && w[2].is_alphanumeric());
    token.chars().any(|c| c.is_ascii_digit())
        || token.contains('/')
        || token.contains('\\')
        || token.contains('_')
        || token.contains("::")
        || token.starts_with('~')
        || (token.starts_with('-') && chars.len() > 2 && chars[1].is_alphanumeric() || token.starts_with("--"))
        || inner_dot
}

/// Numbers, paths and identifiers the version shown adds that occur nowhere in
/// what the agent saw (the request) or wrote (the original): candidates for
/// hindsight carried over from the note. A correction should use only what was
/// in front of the agent; where that was not enough, the right behaviour is to look
/// it up or say it is unknown.
pub fn unseen_facts(context: &str, original: &Response, shown: &Response) -> Vec<String> {
    let mut seen_text = String::from(context);
    seen_text.push_str(&original.reasoning_content);
    seen_text.push_str(&original.content);
    for call in &original.tool_calls {
        seen_text.push_str(&call.function.arguments);
    }
    let mut added = String::new();
    for (old, new) in [
        (original.reasoning_content.as_str(), shown.reasoning_content.as_str()),
        (original.content.as_str(), shown.content.as_str()),
    ] {
        let diff = TextDiff::from_words(old, new);
        for change in diff.iter_all_changes() {
            if change.tag() == ChangeTag::Insert {
                added.push_str(change.value());
                added.push(' ');
            }
        }
    }
    for call in &shown.tool_calls {
        let before = original.tool_calls.iter().find(|c| c.id == call.id).map_or("", |c| c.function.arguments.as_str());
        let diff = TextDiff::from_words(before, call.function.arguments.as_str());
        for change in diff.iter_all_changes() {
            if change.tag() == ChangeTag::Insert {
                added.push_str(change.value());
                added.push(' ');
            }
        }
    }
    let mut found = BTreeSet::new();
    for token in added.split_whitespace() {
        let token = token.trim_matches(|c: char| "\"'`,;()[]{}<>“”‘’!?".contains(c));
        let token = token.trim_end_matches(['.', ':']);
        if fact_like(token) && !seen_text.contains(token) {
            found.insert(token.to_owned());
        }
    }
    found.into_iter().take(12).collect()
}

/// Everything in the request that the agent read, as text.
fn request_text(messages: &[Value]) -> String {
    let mut text = String::new();
    for message in messages {
        match message.get("content") {
            Some(Value::String(content)) => text.push_str(content),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if let Some(t) = part.get("text").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                }
            }
            _ => {}
        }
        text.push('\n');
        if let Some(r) = message.get("reasoning_content").and_then(Value::as_str) {
            text.push_str(r);
        }
        for call in message.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
            if let Some(a) = call.pointer("/function/arguments").and_then(Value::as_str) {
                text.push_str(a);
            }
        }
    }
    text
}

// ------------------------------------------------------------------ rendering

fn pad(line: &mut Line, width: usize) {
    let used: usize = line.iter().map(|(_, t)| width_of(t)).sum();
    if used < width {
        line.push((Style::Plain, " ".repeat(width - used)));
    }
}

fn styled(segments: &Segments, plain: Style, emph: Style) -> Vec<(Style, String)> {
    segments.iter().map(|(e, t)| (if *e { emph } else { plain }, t.clone())).collect()
}

/// The diff sections as lines for a window `width` columns wide.
pub fn render_diffs(diffs: &[FieldDiff], width: usize) -> Vec<Line> {
    let side = width >= SIDE_BY_SIDE_COLUMNS;
    let half = width.saturating_sub(3) / 2;
    let mut out: Vec<Line> = Vec::new();
    for diff in diffs {
        match diff.unchanged_chars {
            Some(chars) => out.push(vec![(Style::Dim, format!("── {}: unchanged, {} chars", diff.title, group_digits(chars)))]),
            None => {
                out.push(vec![(
                    Style::Heading,
                    format!("── {}: {} hunk{}", diff.title, diff.hunks, if diff.hunks == 1 { "" } else { "s" }),
                )]);
                for row in &diff.rows {
                    match row {
                        Row::Gap => out.push(vec![(Style::Dim, "   ⋯".into())]),
                        Row::Same(text) => {
                            if side {
                                let left = wrap(&[(Style::Dim, text.clone())], half);
                                let right = left.clone();
                                out.extend(join_columns(left, right, half));
                            } else {
                                for line in wrap(&[(Style::Dim, format!("  {text}"))], width) {
                                    out.push(line);
                                }
                            }
                        }
                        Row::Changed { old, new } => {
                            if side {
                                let left = old.as_ref().map(|s| wrap(&styled(s, Style::Removed, Style::RemovedEmph), half)).unwrap_or_default();
                                let right = new.as_ref().map(|s| wrap(&styled(s, Style::Added, Style::AddedEmph), half)).unwrap_or_default();
                                out.extend(join_columns(left, right, half));
                            } else {
                                if let Some(old) = old {
                                    let mut segments = vec![(Style::Removed, "- ".to_owned())];
                                    segments.extend(styled(old, Style::Removed, Style::RemovedEmph));
                                    out.extend(wrap(&segments, width));
                                }
                                if let Some(new) = new {
                                    let mut segments = vec![(Style::Added, "+ ".to_owned())];
                                    segments.extend(styled(new, Style::Added, Style::AddedEmph));
                                    out.extend(wrap(&segments, width));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn join_columns(left: Vec<Line>, right: Vec<Line>, half: usize) -> Vec<Line> {
    let rows = left.len().max(right.len()).max(1);
    let mut out = Vec::with_capacity(rows);
    let mut left = left.into_iter();
    let mut right = right.into_iter();
    for _ in 0..rows {
        let mut line = left.next().unwrap_or_default();
        pad(&mut line, half);
        line.push((Style::Dim, " │ ".into()));
        line.extend(right.next().unwrap_or_default());
        out.push(line);
    }
    out
}

fn group_digits(n: usize) -> String {
    let digits: Vec<char> = n.to_string().chars().collect();
    let mut groups: Vec<String> = digits.rchunks(3).map(|chunk| chunk.iter().collect()).collect();
    groups.reverse();
    groups.join(",")
}

// ------------------------------------------------------------------ one mistake, loaded

struct Loaded {
    record: Record,
    original: Option<Response>,
    shown: Option<Version>,
    problem: Option<String>,
    summary: Option<String>,
    counts: Option<String>,
    unseen: Vec<String>,
    messages: Vec<Value>,
}

fn load(book: &Book, n: u64) -> Result<Loaded, String> {
    let record = book.load(n)?;
    let mut problem = record.error.clone();
    let original = match book.original(&record) {
        Ok(original) => Some(original),
        Err(error) => {
            problem = Some(error);
            None
        }
    };
    let shown = match book.shown(&record) {
        Ok(shown) => Some(shown),
        Err(error) => {
            problem.get_or_insert(error);
            None
        }
    };
    let (summary, counts) = match record.proposal.as_deref().map(|rel| book.read_proposal_file(rel)) {
        Some(Ok(p)) => (
            p.summary.clone(),
            match (p.edits, p.changed_tokens, p.total_tokens) {
                (Some(e), Some(c), Some(t)) => Some(format!("{e} span{} changed, {c} of {t} tokens", if e == 1 { "" } else { "s" })),
                _ => None,
            },
        ),
        _ => (None, None),
    };
    // The context of the step being corrected: what the agent saw when it took it.
    let store = book.store();
    let messages = store
        .find(record.target_id())
        .ok()
        .flatten()
        .and_then(|line| store.messages(&line).ok())
        .unwrap_or_default();
    let unseen = match (&original, &shown) {
        (Some(original), Some(shown)) if shown.file != "original" => {
            unseen_facts(&request_text(&messages), original, &shown.response)
        }
        _ => Vec::new(),
    };
    Ok(Loaded { record, original, shown, problem, summary, counts, unseen, messages })
}

fn header(loaded: &Loaded, width: usize) -> Vec<Line> {
    let r = &loaded.record;
    let time = chrono::DateTime::parse_from_rfc3339(&r.logged_at).map(|t| t.format("%H:%M").to_string()).unwrap_or_default();
    let mut out = vec![vec![(
        Style::Bold,
        format!(
            "mistake #{} · {time}{} · call {} of turn {} · state: {} · attempt {}",
            r.mistake,
            r.flagged_by.as_deref().map(|by| format!(" · flagged by {by}")).unwrap_or_default(),
            r.call_in_turn,
            r.turn_id,
            r.state.label(),
            r.attempts
        ),
    )]];
    let label = match r.flagged_by.as_deref() {
        Some(by) => format!("{by}'s reason"),
        None => "note".to_owned(),
    };
    out.extend(wrap(&[(Style::Plain, format!("{label}: {}", r.note))], width));
    // Which step of the stretch is being corrected.
    if let Some((k, of)) = r.step_of(r.target_id()).filter(|(_, of)| *of > 1) {
        let mut text = format!("correcting step {k} of the {of} this mistake covers");
        if k < of {
            text.push_str(&format!(
                "; step{} {} after it {} not trained, as {} context holds the mistake",
                if of - k == 1 { "" } else { "s" },
                if of - k == 1 { format!("{of}") } else { format!("{}–{of}", k + 1) },
                if of - k == 1 { "is" } else { "are" },
                if of - k == 1 { "its" } else { "their" },
            ));
        }
        text.push_str(" · t picks another step");
        out.extend(wrap(&[(Style::Plain, text)], width));
    }
    if r.split == Split::Heldout {
        out.push(vec![(Style::Banner, " held out: used to measure whether this mistake recurs; never trained ".into())]);
    }
    if r.state == State::Pending {
        let since = r
            .rewrite
            .as_ref()
            .and_then(|w| chrono::DateTime::parse_from_rfc3339(&w.launched_at).ok())
            .map(|t| (chrono::Local::now().fixed_offset() - t).num_minutes())
            .map(|m| format!(" (started {m} min ago)"))
            .unwrap_or_default();
        out.push(vec![(Style::Dim, format!("rewrite running{since}"))]);
    }
    match &loaded.shown {
        Some(shown) if shown.manual => out.push(vec![(Style::Plain, format!("showing the manual version {}", shown.file))]),
        Some(shown) if shown.file == "original" => {
            out.push(vec![(Style::Dim, "showing the original: no version to approve yet (e edits it, r regenerates)".into())])
        }
        Some(shown) => {
            let mut text = format!("proposal {}", shown.file);
            if let Some(summary) = &loaded.summary {
                text.push_str(&format!(": {summary}"));
            }
            if let Some(counts) = &loaded.counts {
                text.push_str(&format!(" · {counts}"));
            }
            out.extend(wrap(&[(Style::Plain, text)], width));
        }
        None => {}
    }
    if let Some(problem) = &loaded.problem {
        for line in problem.lines().take(8) {
            out.extend(wrap(&[(Style::Problem, line.to_owned())], width));
        }
    }
    if !loaded.unseen.is_empty() {
        out.extend(wrap(
            &[
                (Style::Problem, "not in what s0ma saw: ".into()),
                (Style::Plain, loaded.unseen.join(", ")),
                (Style::Dim, " (hindsight from the note? a correction may only use the context)".into()),
            ],
            width,
        ));
    }
    out.push(Vec::new());
    out
}

fn body(loaded: &Loaded, width: usize, context: bool) -> Vec<Line> {
    if context {
        return context_lines(loaded, width);
    }
    match (&loaded.original, &loaded.shown) {
        (Some(original), Some(shown)) => render_diffs(&field_diffs(original, &shown.response), width),
        _ => vec![vec![(Style::Problem, "the original response could not be loaded".into())]],
    }
}

fn context_lines(loaded: &Loaded, width: usize) -> Vec<Line> {
    let mut out = vec![vec![(
        Style::Heading,
        format!("── the last {CONTEXT_MESSAGES} messages before the call (read-only; c returns to the diff)"),
    )]];
    if loaded.messages.is_empty() {
        out.push(vec![(Style::Problem, "the call's messages could not be loaded".into())]);
        return out;
    }
    if let Some(system) = loaded.messages.first().filter(|m| m.get("role").and_then(Value::as_str) == Some("system")) {
        let chars = system.get("content").and_then(Value::as_str).map_or(0, |c| c.chars().count());
        out.push(vec![(Style::Dim, format!("system: {} chars", group_digits(chars)))]);
    }
    let start = loaded.messages.len().saturating_sub(CONTEXT_MESSAGES).max(1);
    for message in &loaded.messages[start..] {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("?");
        out.push(vec![(Style::Bold, format!("{role}:"))]);
        let content = match message.get("content") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .map(|p| p.get("text").and_then(Value::as_str).unwrap_or("[image]").to_owned())
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        let lines: Vec<&str> = content.lines().collect();
        for line in lines.iter().take(40) {
            out.extend(wrap(&[(Style::Plain, format!("  {line}"))], width));
        }
        if lines.len() > 40 {
            out.push(vec![(Style::Dim, format!("  … {} more lines", lines.len() - 40))]);
        }
        for call in message.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
            let name = call.pointer("/function/name").and_then(Value::as_str).unwrap_or("?");
            let args = call.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("");
            out.extend(wrap(&[(Style::Dim, format!("  → {name} {}", mistakes::one_line(args, 300)))], width));
        }
    }
    out
}

const KEYS: &str = "a approve · e edit · d discard · r regenerate · t step · c context · n/p mistake · j/k · q close";

// ------------------------------------------------------------------ the window

pub(crate) struct Screen;

impl Screen {
    pub(crate) fn enter() -> io::Result<Self> {
        // The window reads no mouse; an editor opened from it must not get
        // the wheel's reports as text.
        crate::terminal::set_mouse(false);
        crossterm::terminal::enable_raw_mode()?;
        // On the agent's page the alternate screen is the agent's already: the window
        // draws on it, and the page is drawn again after.
        if !crate::page::on() {
            crossterm::execute!(io::stdout(), crossterm::terminal::EnterAlternateScreen)?;
            crate::terminal::note_view_screen(true);
        }
        crossterm::execute!(io::stdout(), crossterm::cursor::Hide)?;
        Ok(Self)
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        // After a panic put the terminal back, the view's screen is already
        // gone: leaving it again would put the cursor back over the message.
        if crate::page::on() {
            let _ = crossterm::execute!(io::stdout(), crossterm::cursor::Show);
            crate::page::back();
        } else if !crate::terminal::torn_down() {
            crate::terminal::note_view_screen(false);
            let _ = crossterm::execute!(io::stdout(), crossterm::cursor::Show, crossterm::terminal::LeaveAlternateScreen);
        }
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

/// The window, every row written and cleared to its end, as one
/// synchronized write: no clear first, so a key never flashes the screen
/// blank.
pub(crate) fn draw(lines: &[Line], scroll: usize, footer: &[Line], columns: usize, rows: usize) -> io::Result<()> {
    use crossterm::{
        cursor::MoveTo,
        style::Print,
        terminal::{BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate},
    };
    let mut out: Vec<u8> = Vec::with_capacity(16 * 1024);
    crossterm::queue!(out, BeginSynchronizedUpdate)?;
    let body_rows = rows.saturating_sub(footer.len());
    let shown: Vec<&Line> = lines.iter().skip(scroll).take(body_rows).collect();
    for row in 0..body_rows {
        crossterm::queue!(out, MoveTo(0, row as u16))?;
        let mut used = 0;
        if let Some(line) = shown.get(row) {
            for (style, text) in line.iter() {
                apply_style(&mut out, *style)?;
                let mut piece = String::new();
                for c in text.chars() {
                    let w = c.width().unwrap_or(0);
                    if used + w > columns {
                        break;
                    }
                    used += w;
                    piece.push(c);
                }
                crossterm::queue!(out, Print(piece))?;
            }
        }
        apply_style(&mut out, Style::Plain)?;
        if used < columns {
            crossterm::queue!(out, Clear(ClearType::UntilNewLine))?;
        }
    }
    for (index, line) in footer.iter().enumerate() {
        crossterm::queue!(out, MoveTo(0, (body_rows + index) as u16))?;
        let mut used = 0;
        for (style, text) in line {
            apply_style(&mut out, *style)?;
            let piece: String = text.chars().take(columns.saturating_sub(used)).collect();
            used += piece.chars().count();
            crossterm::queue!(out, Print(piece))?;
        }
        apply_style(&mut out, Style::Plain)?;
        if used < columns {
            crossterm::queue!(out, Clear(ClearType::UntilNewLine))?;
        }
    }
    apply_style(&mut out, Style::Plain)?;
    crossterm::queue!(out, EndSynchronizedUpdate)?;
    let mut stdout = io::stdout().lock();
    stdout.write_all(&out)?;
    stdout.flush()
}

enum Prompt {
    Discard,
    Regenerate,
}

/// Read a line at the bottom of the window. `None` on Esc.
/// Ask for a line of text under `lines`, which ↑↓ and PgUp/PgDn scroll
/// (`scroll` is where they start; `usize::MAX` is the bottom).
pub(crate) fn prompt_line(label: &str, initial: &str, lines: &[Line], scroll: usize) -> io::Result<Option<String>> {
    use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
    let mut text = initial.to_owned();
    let mut scroll = scroll;
    loop {
        let (columns, rows) = crossterm::terminal::size()?;
        let (columns, rows) = (columns as usize, rows as usize);
        let footer = vec![
            vec![(Style::Key, format!(" {label} "))],
            vec![(Style::Plain, format!("> {text}▏")), (Style::Dim, "   enter to confirm · esc to cancel".into())],
        ];
        let max_scroll = lines.len().saturating_sub(rows.saturating_sub(footer.len()));
        scroll = scroll.min(max_scroll);
        let page = rows.saturating_sub(4).max(1);
        draw(lines, scroll, &footer, columns, rows)?;
        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Up => scroll = scroll.saturating_sub(1),
                KeyCode::Down => scroll = (scroll + 1).min(max_scroll),
                KeyCode::PageUp => scroll = scroll.saturating_sub(page),
                KeyCode::PageDown => scroll = (scroll + page).min(max_scroll),
                KeyCode::Enter => return Ok(Some(text)),
                KeyCode::Esc => return Ok(None),
                KeyCode::Backspace => {
                    text.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => text.clear(),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(None),
                KeyCode::Char(c) => text.push(c),
                _ => {}
            }
        }
    }
}

/// Open `$VISUAL` / `$EDITOR` (default `vi`) on `text`; the edited text, or
/// `None` if the editor failed. `tag` only names the temporary file.
/// Whether `program` is on the PATH.
fn on_path(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| std::env::split_paths(&path).any(|dir| dir.join(program).is_file()))
}

pub(crate) fn edit_text(text: &str, tag: &str) -> io::Result<Option<String>> {
    let path = std::env::temp_dir().join(format!("s0ma-{tag}-{}.txt", std::process::id()));
    std::fs::write(&path, text)?;
    let saved = edit_path(&path);
    let edited = match saved {
        Ok(true) => Some(std::fs::read_to_string(&path)),
        Ok(false) => None,
        Err(error) => Some(Err(error)),
    };
    let _ = std::fs::remove_file(&path);
    edited.transpose()
}

/// The operator's editor on `path`, the terminal handed over to it and taken
/// back: whether it exited cleanly.
pub(crate) fn edit_path(path: &std::path::Path) -> io::Result<bool> {
    if crate::page::on() {
        // The editor on the shell's screen, the page's scroll region reset.
        crossterm::execute!(io::stdout(), crossterm::cursor::Show)?;
        crate::page::step_out();
    } else {
        crate::terminal::note_view_screen(false);
        crossterm::execute!(io::stdout(), crossterm::cursor::Show, crossterm::terminal::LeaveAlternateScreen)?;
    }
    crossterm::terminal::disable_raw_mode()?;
    let editor = std::env::var("VISUAL")
        .ok()
        .filter(|e| !e.trim().is_empty())
        .or_else(|| std::env::var("EDITOR").ok().filter(|e| !e.trim().is_empty()))
        // With none set, nano before vi: it says on screen how to save and
        // leave (^O, ^X), where vi leaves someone who does not know it stuck
        // on a screen that is not the agent's.
        .unwrap_or_else(|| if on_path("nano") { "nano".into() } else { "vi".into() });
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!("{editor} \"$1\""))
        .arg("s0ma-editor")
        .arg(path)
        .status();
    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(io::stdout(), crossterm::terminal::EnterAlternateScreen, crossterm::cursor::Hide)?;
    if !crate::page::on() {
        crate::terminal::note_view_screen(true);
    }
    Ok(status.is_ok_and(|status| status.success()))
}

/// The stretch, one line per step: when, what it did, and which is being
/// corrected and which was flagged.
fn step_lines(book: &Book, record: &Record, width: usize) -> Vec<Line> {
    let store = book.store();
    let mut out = wrap(
        &[(
            Style::Heading,
            "── the steps this mistake covers: the turn's calls since your last message (or it began), and before that \
             message if the flag came right after it. Correct the earliest wrong one."
                .to_owned(),
        )],
        width,
    );
    for (index, step) in record.steps().iter().enumerate() {
        let what = store
            .find(&step.call_id)
            .ok()
            .flatten()
            .and_then(|line| store.response(&line).ok())
            .map(|response| mistakes::gist(&response))
            .unwrap_or_else(|| "(not in the store)".into());
        let time = chrono::DateTime::parse_from_rfc3339(&step.at).map(|t| t.format("%H:%M:%S").to_string()).unwrap_or_default();
        let mut marks = Vec::new();
        if step.call_id == record.target_id() {
            marks.push("correcting");
        }
        if step.call_id == record.call_id {
            marks.push("flagged");
        }
        let marks = if marks.is_empty() { String::new() } else { format!("  [{}]", marks.join(", ")) };
        out.extend(wrap(
            &[(Style::Bold, format!("{:>3}  ", index + 1)), (Style::Plain, format!("{time}  {what}")), (Style::Dim, marks)],
            width,
        ));
    }
    out
}

fn fresh_id(n: u64, index: usize) -> String {
    let seed = format!("{n}-{index}-{:?}-{}", std::time::SystemTime::now(), std::process::id());
    format!("call_m{n}_{:08x}{:08x}", fold_hash(&seed), fold_hash(&format!("{seed}x")))
}

/// Run the window on mistake `start` (or the first waiting). Returns a line to
/// print when it closes.
pub fn run(book: &Book, start: Option<u64>) -> io::Result<String> {
    use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
    let waiting = |book: &Book| -> Vec<u64> {
        book.waiting().map(|w| w.into_iter().map(|r| r.mistake).collect()).unwrap_or_default()
    };
    let mut current = match start.or_else(|| waiting(book).first().copied()) {
        Some(n) => n,
        None => return Ok("[mistake] no mistakes waiting for review".into()),
    };
    if let Err(error) = book.load(current) {
        return Ok(format!("[mistake] {error}"));
    }
    let _screen = Screen::enter()?;
    let mut scroll = 0usize;
    let mut context = false;
    let mut message: Option<(Style, String)> = None;
    // An unsaved edit, kept for the next `e`: by mistake and step, so a draft
    // written against one step never opens (or is saved) as another's.
    let mut drafts: std::collections::HashMap<(u64, String), String> = std::collections::HashMap::new();
    let mut decided = 0usize;
    'outer: loop {
        let loaded = match load(book, current) {
            Ok(loaded) => loaded,
            Err(error) => return Ok(format!("[mistake] {error}")),
        };
        loop {
            let (columns, rows) = crossterm::terminal::size()?;
            let (columns, rows) = (columns as usize, rows as usize);
            let mut lines = header(&loaded, columns);
            lines.extend(body(&loaded, columns, context));
            let footer = vec![
                message.clone().map(|(s, t)| vec![(s, t)]).unwrap_or_default(),
                vec![(Style::Key, format!(" {KEYS} "))],
            ];
            let max_scroll = lines.len().saturating_sub(rows.saturating_sub(footer.len()));
            scroll = scroll.min(max_scroll);
            draw(&lines, scroll, &footer, columns, rows)?;
            // Poll while a rewrite runs, so the result appears without a key.
            if !event::poll(Duration::from_millis(if loaded.record.state == State::Pending { 1000 } else { 60_000 }))? {
                if loaded.record.state == State::Pending
                    && book.load(current).is_ok_and(|r| r.state != State::Pending || r.attempts != loaded.record.attempts)
                {
                    continue 'outer;
                }
                continue;
            }
            let Event::Key(key) = event::read()? else { continue };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            let page = rows.saturating_sub(4).max(1);
            message = None;
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break 'outer,
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break 'outer,
                KeyCode::Char('j') | KeyCode::Down => scroll = (scroll + 1).min(max_scroll),
                KeyCode::Char('k') | KeyCode::Up => scroll = scroll.saturating_sub(1),
                KeyCode::PageDown | KeyCode::Char(' ') => scroll = (scroll + page).min(max_scroll),
                KeyCode::PageUp => scroll = scroll.saturating_sub(page),
                KeyCode::Char('g') | KeyCode::Home => scroll = 0,
                KeyCode::Char('G') | KeyCode::End => scroll = max_scroll,
                KeyCode::Char('c') => {
                    context = !context;
                    scroll = 0;
                }
                KeyCode::Char('t') => {
                    let steps = loaded.record.steps();
                    if steps.len() < 2 {
                        message = Some((Style::Dim, "this mistake has one step: the call flagged".into()));
                        continue;
                    }
                    if !loaded.record.state.waiting() {
                        message = Some((Style::Problem, match loaded.record.state {
                            State::Pending => "the rewrite is still running; wait for it".into(),
                            state => format!("it is {}; reopen it to pick another step", state.label()),
                        }));
                        continue;
                    }
                    let listing = step_lines(book, &loaded.record, columns);
                    // From the bottom: the flagged step and the latest are last.
                    let Some(answer) =
                        prompt_line(&format!("correct which step? 1–{} (↑↓ scroll)", steps.len()), "", &listing, usize::MAX)?
                    else {
                        continue;
                    };
                    let chosen = answer.trim().parse::<usize>().ok().filter(|k| (1..=steps.len()).contains(k));
                    let Some(k) = chosen else {
                        message = Some((Style::Problem, format!("not a step: {answer:?}")));
                        continue;
                    };
                    match book.retarget(current, &steps[k - 1].call_id) {
                        Ok(_) => {
                            message = Some((Style::Added, format!("correcting step {k}; e edits it")));
                            scroll = 0;
                            context = false;
                            continue 'outer;
                        }
                        Err(error) => message = Some((Style::Problem, error)),
                    }
                }
                KeyCode::Char('n') | KeyCode::Char('p') => {
                    let list = waiting(book);
                    let next = if key.code == KeyCode::Char('n') {
                        list.iter().copied().find(|m| *m > current).or(list.first().copied())
                    } else {
                        list.iter().rev().copied().find(|m| *m < current).or(list.last().copied())
                    };
                    match next {
                        Some(next) if next != current => {
                            current = next;
                            scroll = 0;
                            context = false;
                            continue 'outer;
                        }
                        _ => message = Some((Style::Dim, "no other mistake is waiting for review".into())),
                    }
                }
                KeyCode::Char('a') => match book.approve(current, loaded.shown.as_ref().map(|v| v.file.as_str())) {
                    Ok(record) => {
                        decided += 1;
                        message = Some((
                            Style::Added,
                            format!("#{current} {} — n for the next waiting", record.state.label()),
                        ));
                        continue 'outer;
                    }
                    Err(error) => {
                        message = Some((Style::Problem, error));
                        // Whatever changed, show it.
                        continue 'outer;
                    }
                },
                KeyCode::Char('e') => {
                    if loaded.record.state == State::Pending {
                        message = Some((Style::Problem, "the rewrite is still running; wait, or regenerate".into()));
                        continue;
                    }
                    if !loaded.record.state.waiting() {
                        message = Some((Style::Problem, format!("it is {}; reopen it to edit", loaded.record.state.label())));
                        continue;
                    }
                    let Some(shown) = &loaded.shown else {
                        message = Some((Style::Problem, "nothing to edit: the original could not be loaded".into()));
                        continue;
                    };
                    let draft_key = (current, shown.target.clone());
                    let form = drafts.remove(&draft_key).unwrap_or_else(|| to_form(&shown.response));
                    let Some(edited) = edit_text(&form, &format!("mistake-{current}"))? else {
                        drafts.insert(draft_key, form);
                        message = Some((Style::Problem, "the editor did not exit cleanly; nothing saved".into()));
                        continue;
                    };
                    if edited == to_form(&shown.response) {
                        message = Some((Style::Dim, "no changes".into()));
                        continue;
                    }
                    let n = current;
                    match from_form(&edited, |i| fresh_id(n, i)) {
                        Ok(response) => match book.save_edit(current, &response, &shown.file, &shown.target) {
                            Ok(rel) => {
                                message = Some((Style::Added, format!("saved {rel}; a approves it")));
                                continue 'outer;
                            }
                            Err(error) => {
                                drafts.insert(draft_key, edited);
                                message = Some((Style::Problem, error));
                            }
                        },
                        Err(error) => {
                            // Kept: the next `e` starts from what was typed.
                            drafts.insert(draft_key, edited);
                            message = Some((Style::Problem, format!("not saved: {error} (e to fix it)")));
                        }
                    }
                }
                KeyCode::Char('d') | KeyCode::Char('r') => {
                    let kind = if key.code == KeyCode::Char('d') { Prompt::Discard } else { Prompt::Regenerate };
                    let (label, initial) = match kind {
                        Prompt::Discard => ("discard: a reason (optional)", String::new()),
                        Prompt::Regenerate => ("regenerate with this note", loaded.record.note.clone()),
                    };
                    let Some(text) = prompt_line(label, &initial, &lines, scroll)? else { continue };
                    let result = match kind {
                        Prompt::Discard => book.discard(current, Some(&text)).map(|r| {
                            decided += 1;
                            format!("#{current} {}", r.state.label())
                        }),
                        Prompt::Regenerate => book.regenerate(current, Some(&text)).map(|(r, launch)| match launch {
                            mistakes::Launch::Started => format!("#{current} rewriting (attempt {})", r.attempts),
                            mistakes::Launch::Failed(error) => format!("#{current}: the rewriter could not start: {error}"),
                        }),
                    };
                    match result {
                        Ok(text) => {
                            message = Some((Style::Added, text));
                            continue 'outer;
                        }
                        Err(error) => message = Some((Style::Problem, error)),
                    }
                }
                _ => {}
            }
        }
    }
    Ok(if decided > 0 {
        format!("[mistake] review closed: {decided} decided this time")
    } else {
        "[mistake] review closed".into()
    })
}

/// `s0ma --mistake-view [n]`: the window in a terminal of its own, beside a
/// running agent or without one.
pub fn run_standalone(root: &Path, start: Option<u64>) -> Result<(), Box<dyn std::error::Error>> {
    let book = Book::open(&root.join("data"));
    let closing = run(&book, start)?;
    println!("{closing}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall { id: id.into(), kind: "function".into(), function: FunctionCall { name: name.into(), arguments: arguments.into() } }
    }

    fn response(reasoning: &str, reply: &str, calls: Vec<ToolCall>) -> Response {
        Response { reasoning_content: reasoning.into(), content: reply.into(), tool_calls: calls }
    }

    fn no_fresh(_: usize) -> String {
        panic!("no new tool call expected")
    }

    #[test]
    fn the_form_round_trips_exactly() {
        for r in [
            response("thinking\n", "Reply — ✓\n\nwith a blank line", vec![call("call_8f2", "run_command", "{\"command\": \"cargo test -p parser\"}")]),
            response("", "", Vec::new()),
            response("ends with two newlines\n\n", "\n", vec![call("a", "x", "{}"), call("b", "y", "[1,\n 2]\n")]),
            // A reply about this very feature: marker-like lines inside a field.
            response("=== reply ===\n=== tool_call 1: x (id y) ===", "see:\n=== reasoning ===\n", vec![call("c", "z", "{\"t\": \"=== reply ===\"}")]),
        ] {
            let form = to_form(&r);
            assert_eq!(from_form(&form, no_fresh).unwrap(), r, "form:\n{form}");
            // An editor that drops the final newline loses nothing that matters.
            if !r.tool_calls.last().is_some_and(|c| c.function.arguments.ends_with('\n')) && !r.content.ends_with('\n') {
                assert_eq!(from_form(form.strip_suffix('\n').unwrap(), no_fresh).unwrap(), r);
            }
        }
        let tricky = response("=== reply ===", "x", Vec::new());
        assert!(to_form(&tricky).starts_with("=== ["), "a nonce is used when a field holds a marker line");
    }
}
