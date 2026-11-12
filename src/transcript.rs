//! How the conversation is drawn in the transcript, and the record that lets
//! Ctrl+O draw all of it again the other way.
//!
//! **Collapsed** (the default) reads as a conversation: your messages, the agent's
//! replies, and between them one dim line for each stretch of work. While the agent
//! works, that line shows what it is doing now (`[*] thinking · ~240 tokens`,
//! `[*] vm_exec(df -h)`); when the reply starts, it settles into a summary:
//! `[+] 14s · ran vm_exec ×2, web_search · ~1.2k tokens of thought · 1 failed`.
//! Harness status lines (folds, subagent progress, masking) are not shown.
//!
//! Each of the agent's messages hangs under one `s0ma>`: its later lines, the text
//! of later passes and the work lines between them sit in a gutter as wide as
//! the name, so only a new speaker (`you>`, the agent's next message, a harness
//! line) starts at the left edge:
//!
//! ```text
//! you> check the disk
//!       [+] 4s · ran vm_exec · ~300 tokens of thought
//! the agent> I'll look at /var too.
//!       [+] 2s · ran vm_exec
//!       The root filesystem is 82% full, mostly logs.
//! ```
//!
//! **Expanded** is the full stream, as it always was: `[Thinking]` with the
//! reasoning text, `[tool]` lines and every status line.
//!
//! In both, a reply starts right after `s0ma>`: blank lines a model opens its
//! answer with are dropped, and the prefix waits for the first visible text.
//!
//! Everything the display writes is kept in a [`TranscriptRecord`] for the
//! whole session (bounded), so switching modes clears the screen and draws the
//! conversation again through a fresh [`Renderer`]: the redraw ends in the same
//! state live output would have reached, and a turn in progress carries on
//! from there. Set `S0MA_DETAIL=expanded` to start expanded.

use crate::terminal::{
    NAME_STYLE, OPERATOR_STYLE, OPERATOR_TEXT_STYLE, RESET, STATUS_STYLE, TEXT_STYLE,
    THINKING_STYLE, WordWrap, format_elapsed, safe_text, truncate_to_width,
};

/// Collapsed work lines: quieter than the reply text around them.
pub(crate) const WORK_STYLE: &str = "\x1b[3;90m";

/// Collapsed: how far the agent's message hangs in under its name (`s0ma> `).
pub const GUTTER: usize = 6;
/// Narrower than this, a gutter would cost too much of the line.
const GUTTER_MIN_COLUMNS: usize = 40;

/// The gutter at a terminal `columns` wide (0 when the width is unknown).
/// Reply text is laid out for what is left of the line (see `StreamPrinter`).
pub fn gutter_width(columns: usize) -> usize {
    if columns >= GUTTER_MIN_COLUMNS { GUTTER } else { 0 }
}

use serde_json::Value;
use unicode_width::UnicodeWidthStr;

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

/// A finished tool call.
const OK_MARK: &str = "\x1b[92m";
/// A tool call that failed.
const FAILED_MARK: &str = "\x1b[91m";
/// A tool call the operator stopped.
const KILLED_MARK: &str = "\x1b[91m";

/// The live work line is redrawn at most this often.
const THINKING_REDRAW: Duration = Duration::from_millis(150);
/// Reasoning characters per token until usage reports calibrate it.
pub const DEFAULT_CHARS_PER_TOKEN: f64 = 3.8;

/// The most the session's record keeps; the oldest entries go first.
const RECORD_LIMIT: usize = 8 << 20;
/// Consecutive reasoning or reply chunks (or plain output) are merged up to
/// this size, so a long pass is still several entries a partial redraw can
/// start between.
const MERGE_LIMIT: usize = 32 << 10;
/// What an entry costs the record beyond its text, counted against
/// `RECORD_LIMIT` so many small entries stay bounded too.
const ENTRY_OVERHEAD: usize = 64;
/// How much Ctrl+O redraws, counted in what the new mode writes.
const REPLAY_BUDGET: usize = 1 << 20;
/// Distinct tool names a work summary lists before `+N more`.
const SUMMARY_TOOLS: usize = 4;

/// How much of reasoning, tool calls and status the transcript shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    Collapsed,
    Expanded,
}

impl Detail {
    /// `S0MA_DETAIL=expanded` starts expanded; anything else, collapsed.
    pub fn from_env() -> Self {
        Self::parse(std::env::var("S0MA_DETAIL").ok().as_deref())
    }

    fn parse(value: Option<&str>) -> Self {
        match value.map(|value| value.trim().to_ascii_lowercase()) {
            Some(value) if matches!(value.as_str(), "expanded" | "expand" | "full" | "verbose") => {
                Detail::Expanded
            }
            _ => Detail::Collapsed,
        }
    }

    pub fn toggled(self) -> Self {
        match self {
            Detail::Collapsed => Detail::Expanded,
            Detail::Expanded => Detail::Collapsed,
        }
    }
}

/// The dim line a redraw starts with when the record no longer reaches back
/// to the start of the session.
pub const EARLIER_NOT_SHOWN: &str = "── earlier in this session is not shown here; the archive has all of it ──";

/// How a notice reads collapsed: `[*]` something happening, `[+]` a step that
/// finished, `[-]` a problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Mark {
    Work,
    Done,
    Problem,
}

/// One thing written to the transcript.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Entry {
    /// A turn begins (the operator's or an autonomous one): the agent's next words
    /// start a new message, under its name again.
    Turn,
    /// A display pass begins (the steering pane opened): a fresh line, and the
    /// next reasoning and reply get their headers again.
    Pass,
    /// Reasoning text, already made safe for the terminal.
    Reasoning(String),
    /// Reply text as the markdown renderer styled it. The first of a pass gets
    /// the `s0ma>` prefix.
    Answer(String),
    /// Reply text the markdown renderer released late (at a status line or
    /// after an operator message): written as it is, never with a prefix.
    AnswerTail(String),
    /// A harness status line inside a turn, already made safe.
    Status(String),
    /// A harness note between display passes (fold progress, context
    /// notices): a plain line expanded, not shown collapsed.
    Note(String),
    /// A harness line for the operator (a warning, a command's result), shown
    /// in both modes on a line of its own, even in the middle of a reply.
    Line(String),
    /// A harness notice with a quiet collapsed form: there, `brief` as a dim
    /// `[*]` line in the gutter, like a work line (a green `[+]` for a step
    /// that finished, `[-]` and brighter for a problem); expanded, `full` as a
    /// `Line` is shown. Both made safe already.
    Notice { brief: String, full: String, mark: Mark },
    /// Written as it is in both modes. `operator`: the operator's own lines,
    /// after which reasoning and reply get their headers again.
    Raw { text: String, operator: bool },
    /// A message from the operator (made safe): `you>  ` and the text, its
    /// later lines hanging under the first, in both modes.
    Operator(String),
}

impl Entry {
    fn size(&self) -> usize {
        match self {
            Entry::Turn | Entry::Pass => 0,
            Entry::Reasoning(text)
            | Entry::Answer(text)
            | Entry::AnswerTail(text)
            | Entry::Status(text)
            | Entry::Note(text)
            | Entry::Line(text)
            | Entry::Operator(text)
            | Entry::Raw { text, .. } => text.len(),
            Entry::Notice { brief, full, .. } => brief.len() + full.len(),
        }
    }

    /// Roughly what drawing this entry in `detail` writes.
    fn cost(&self, detail: Detail) -> usize {
        match (detail, self) {
            (Detail::Collapsed, Entry::Reasoning(_) | Entry::Status(_) | Entry::Note(_)) => 0,
            _ => self.size(),
        }
    }
}

/// An entry with when it began and when it was last extended.
#[derive(Clone, Debug)]
pub struct Recorded {
    pub first: Instant,
    pub last: Instant,
    pub entry: Entry,
}

/// What the display wrote this session, bounded.
#[derive(Default)]
pub struct TranscriptRecord {
    entries: VecDeque<Recorded>,
    bytes: usize,
    /// Entries were dropped from the front to stay under the limit.
    dropped: bool,
    /// Counts every change.
    revision: u64,
}

impl TranscriptRecord {
    /// Every entry kept, oldest first, and whether earlier ones were dropped:
    /// what a terminal attaching to this agent starts from (`session.rs`).
    pub fn all(&self) -> (impl Iterator<Item = &Recorded>, bool) {
        (self.entries.iter(), self.dropped)
    }

    /// An entry as another terminal recorded it, with its own times.
    pub fn push_recorded(&mut self, recorded: Recorded) {
        self.revision = self.revision.wrapping_add(1);
        self.bytes += recorded.entry.size() + ENTRY_OVERHEAD;
        self.entries.push_back(recorded);
        self.trim();
    }

    pub fn set_dropped(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.dropped = true;
    }

    /// Changes whenever the record does.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|recorded| matches!(recorded.entry, Entry::Turn | Entry::Pass))
    }

    pub fn push(&mut self, at: Instant, entry: Entry) {
        self.revision = self.revision.wrapping_add(1);
        let size = entry.size();
        if let Some(last) = self.entries.back_mut() {
            let merged = match (&mut last.entry, &entry) {
                (Entry::Reasoning(held), Entry::Reasoning(more))
                | (Entry::Answer(held), Entry::Answer(more))
                | (Entry::Raw { text: held, operator: false }, Entry::Raw { text: more, operator: false })
                    if held.len() < MERGE_LIMIT =>
                {
                    held.push_str(more);
                    true
                }
                _ => false,
            };
            if merged {
                last.last = at;
                self.bytes += size;
                self.trim();
                return;
            }
        }
        self.entries.push_back(Recorded { first: at, last: at, entry });
        self.bytes += size + ENTRY_OVERHEAD;
        self.trim();
    }

    fn trim(&mut self) {
        while self.bytes > RECORD_LIMIT && self.entries.len() > 1 {
            if let Some(dropped) = self.entries.pop_front() {
                self.bytes -= dropped.entry.size() + ENTRY_OVERHEAD;
                self.dropped = true;
            }
        }
    }

    /// The entries a redraw in `detail` shows, oldest first, and whether
    /// earlier ones are left out.
    pub fn replay(&self, detail: Detail) -> (Vec<&Recorded>, bool) {
        let mut start = self.entries.len();
        let mut spent = 0;
        while start > 0 {
            let cost = self.entries[start - 1].entry.cost(detail);
            if spent + cost > REPLAY_BUDGET && spent > 0 {
                break;
            }
            spent += cost;
            start -= 1;
        }
        let partial = start > 0 || self.dropped;
        (self.entries.range(start..).collect(), partial)
    }
}

/// A `[tool]` status line the collapsed view folds into its work line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolStatus {
    Running { name: String, arguments: String },
    /// `time` as the agent wrote it, such as `1.2s`.
    Finished { name: String, time: String },
    Failed { name: String, error: String },
    Killed { name: String, detail: Option<String> },
}

/// Read the agent's tool status lines (see `agent::run`):
/// `[tool] running: NAME, arguments: ARGS`, `[tool] finished: NAME (1.2s)`,
/// `[tool] failed: NAME: ERROR` and `[tool] killed: NAME[: DETAIL]`.
pub fn parse_tool_status(message: &str) -> Option<ToolStatus> {
    let rest = message.strip_prefix("[tool] ")?;
    if let Some(call) = rest.strip_prefix("running: ") {
        let (name, arguments) = call.split_once(", arguments: ").unwrap_or((call, ""));
        let name = name.trim();
        return valid_name(name).then(|| ToolStatus::Running {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    if let Some(done) = rest.strip_prefix("finished: ") {
        let (name, time) = done.rsplit_once(" (")?;
        let time = time.strip_suffix(')')?;
        return valid_name(name).then(|| ToolStatus::Finished { name: name.to_string(), time: time.to_string() });
    }
    if let Some(failure) = rest.strip_prefix("failed: ") {
        let (name, error) = failure.split_once(": ")?;
        return valid_name(name).then(|| ToolStatus::Failed { name: name.to_string(), error: error.to_string() });
    }
    if let Some(killed) = rest.strip_prefix("killed: ") {
        let (name, detail) = match killed.split_once(": ") {
            Some((name, detail)) => (name, Some(detail.to_string())),
            None => (killed, None),
        };
        return valid_name(name).then(|| ToolStatus::Killed { name: name.to_string(), detail });
    }
    None
}

/// Tool names are identifiers; anything else is a status line about something
/// other than one call.
fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'))
}

/// Argument names whose value says what a call does; shown first, unlabelled.
const PRIMARY_ARGUMENTS: &[&str] = &[
    "command", "query", "url", "path", "file", "keys", "text", "code", "script", "prompt", "task",
    "goal", "message", "name", "id",
];

/// A tool call's arguments as one short line: the value alone when there is
/// one argument; otherwise the main one first (`command`, `query`, `path` …)
/// and the rest as `key: value`.
pub fn compact_arguments(arguments: &str) -> String {
    let arguments = arguments.trim();
    let value_text = |value: &Value| match value {
        Value::String(text) => flatten(text, 300),
        other => flatten(&other.to_string(), 300),
    };
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) if map.len() == 1 => map.values().next().map(value_text).unwrap_or_default(),
        Ok(Value::Object(map)) => {
            let primary = PRIMARY_ARGUMENTS.iter().find(|key| map.contains_key(**key)).copied();
            let mut parts: Vec<String> = primary.and_then(|key| map.get(key)).map(value_text).into_iter().collect();
            parts.extend(
                map.iter()
                    .filter(|(key, _)| Some(key.as_str()) != primary)
                    .take(8)
                    .map(|(key, value)| format!("{}: {}", flatten(key, 60), value_text(value))),
            );
            parts.join(", ")
        }
        Ok(other) => value_text(&other),
        Err(_) => flatten(arguments, 300),
    }
}

/// One line of at most `limit` characters, safe for the terminal: line breaks
/// and runs of spaces become single spaces. Decoding JSON can turn `\u001b`
/// back into a real escape, so this sanitizes again.
fn flatten(text: &str, limit: usize) -> String {
    let text = safe_text(text);
    let mut out = String::new();
    for word in text.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
        if out.chars().count() >= limit {
            return out.chars().take(limit).collect::<String>() + "…";
        }
    }
    out
}

/// `~240 tokens`, `~1.2k tokens`, `~38k tokens`.
pub fn format_tokens(tokens: usize) -> String {
    match tokens {
        1 => "~1 token".to_string(),
        0..1_000 => format!("~{tokens} tokens"),
        1_000..10_000 => {
            let tenths = (tokens as f64 / 100.0).round() / 10.0;
            if tenths.fract() == 0.0 {
                format!("~{tenths:.0}k tokens")
            } else {
                format!("~{tenths:.1}k tokens")
            }
        }
        _ => format!("~{}k tokens", (tokens as f64 / 1000.0).round() as usize),
    }
}

/// A span of time: `<1s`, `8s`, `1m 04s`.
fn format_span(duration: Duration) -> String {
    if duration < Duration::from_secs(1) {
        "<1s".to_string()
    } else {
        format_elapsed(duration)
    }
}

/// Reply text with what comes before its first visible character dropped:
/// blank lines and spaces go, styling stays. Returns it and whether anything
/// visible was found.
fn strip_leading_space(text: &str) -> (String, bool) {
    let mut out = String::with_capacity(text.len());
    let mut visible = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if visible {
            out.push(c);
            continue;
        }
        match c {
            '\x1b' => {
                out.push(c);
                if chars.peek() == Some(&'[') {
                    out.push(chars.next().unwrap());
                    for c in chars.by_ref() {
                        out.push(c);
                        if ('\x40'..='\x7e').contains(&c) {
                            break;
                        }
                    }
                }
            }
            c if c.is_whitespace() => {}
            c => {
                visible = true;
                out.push(c);
            }
        }
    }
    (out, visible)
}

/// Where `text`'s last visible character ends (0 if it has none): after it
/// come only whitespace and styling.
fn visible_end(text: &str) -> usize {
    let mut end = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        if c == '\x1b' {
            if chars.peek().map(|(_, next)| *next) == Some('[') {
                chars.next();
                for (_, c) in chars.by_ref() {
                    if ('\x40'..='\x7e').contains(&c) {
                        break;
                    }
                }
            }
        } else if !c.is_whitespace() {
            end = at + c.len_utf8();
        }
    }
    end
}

/// The styling sequences in `text`, without the rest.
fn escapes_only(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            out.push(c);
            out.push(chars.next().unwrap());
            for c in chars.by_ref() {
                out.push(c);
                if ('\x40'..='\x7e').contains(&c) {
                    break;
                }
            }
        }
    }
    out
}

/// Where rendered text goes: the terminal's width, and whether to word-wrap
/// (inside the active display, and in redraws) or pass text through (live
/// output outside the display, as before).
pub struct Out<'a> {
    pub text: &'a mut String,
    pub columns: usize,
    pub wrapping: bool,
}

impl Out<'_> {
    /// The widest a rewritable line may be: the last column stays empty so
    /// drawing it can't wrap.
    fn line_width(&self) -> usize {
        let columns = if self.columns == 0 { 80 } else { self.columns };
        columns.saturating_sub(1).max(1)
    }
}

/// What the live work line shows (collapsed).
#[derive(Clone, Debug)]
enum Now {
    Thinking,
    Tool {
        name: String,
        arguments: String,
        /// Set once it finished, as the agent wrote it (`1.2s`).
        time: Option<String>,
        outcome: Option<Outcome>,
    },
}

#[derive(Clone, Debug)]
enum Outcome {
    Failed(String),
    Killed,
}

/// One stretch of work between replies (collapsed): drawn as one line, live
/// while it lasts and summarized when it ends.
struct Stretch {
    started: Instant,
    last: Instant,
    thought_chars: usize,
    /// Calls by name, in the order first seen.
    tools: Vec<(String, usize)>,
    failed: usize,
    stopped: usize,
    folded: bool,
    now: Now,
}

impl Stretch {
    fn new(at: Instant, folded: bool) -> Self {
        Self {
            started: at,
            last: at,
            thought_chars: 0,
            tools: Vec::new(),
            failed: 0,
            stopped: 0,
            folded,
            now: Now::Thinking,
        }
    }
}

/// Turns entries into terminal text for one mode.
pub struct Renderer {
    pub detail: Detail,
    /// Whose replies these are (`s0ma>`; a subagent's tab uses its own), and
    /// the name's style.
    name: String,
    name_style: &'static str,
    pub wrap: WordWrap,
    pub chars_per_token: f64,
    thinking_started: bool,
    answer_started: bool,
    /// Reply styling that arrived before any visible reply text this pass:
    /// the `s0ma>` prefix waits for text.
    answer_held: Option<String>,
    /// Collapsed: the agent's name is on screen for the message in progress, so
    /// more of it (a later pass's text) goes on in the gutter, without it.
    message_open: bool,
    /// Collapsed: notices that came while a reply was mid-line, drawn once
    /// that pass of the reply is over rather than in the middle of a sentence.
    deferred: Vec<(String, Mark)>,
    /// Collapsed: the line breaks (and styling among them) the reply's latest
    /// text ended with, held until more of it comes. A pass that ends there
    /// leaves no blank lines above what follows it.
    trailing: String,
    /// Collapsed: the stretch of work in progress.
    stretch: Option<Stretch>,
    /// Collapsed: a fold happened since the last stretch ended.
    fold_since: bool,
    /// Collapsed: the work line can still be rewritten in place.
    open: bool,
    /// The open line changed since it was last drawn.
    dirty: bool,
    drawn_at: Option<Instant>,
    /// Display width of the open line as last drawn.
    open_width: usize,
}

impl Renderer {
    pub fn new(detail: Detail, chars_per_token: f64) -> Self {
        Self {
            detail,
            name: "s0ma".to_string(),
            name_style: NAME_STYLE,
            wrap: WordWrap::default(),
            chars_per_token,
            thinking_started: false,
            answer_started: false,
            answer_held: None,
            message_open: false,
            deferred: Vec::new(),
            trailing: String::new(),
            stretch: None,
            fold_since: false,
            open: false,
            dirty: false,
            drawn_at: None,
            open_width: 0,
        }
    }

    /// The same, for another agent's replies: `name>` in `style`.
    pub fn named(mut self, name: &str, style: &'static str) -> Self {
        self.name = name.to_string();
        self.name_style = style;
        self
    }

    /// Write `entry`, which began at `first` and was last extended at `last`.
    pub fn apply(&mut self, out: &mut Out, first: Instant, last: Instant, entry: &Entry) {
        match entry {
            // Some servers send an empty reasoning field with every reply chunk.
            Entry::Reasoning(text) if text.is_empty() => return,
            // What the operator sent at the idle prompt, which reedline always
            // put at the start of a line (steering lines bring their own break).
            Entry::Raw { text, operator: true } if !text.starts_with(['\r', '\n']) => {
                self.settle(out);
                self.ensure_line_start(out);
            }
            _ => {}
        }
        match self.detail {
            Detail::Expanded => self.apply_expanded(out, entry),
            Detail::Collapsed => self.apply_collapsed(out, first, last, entry),
        }
    }

    /// The full stream, as `StreamPrinter` always wrote it.
    fn apply_expanded(&mut self, out: &mut Out, entry: &Entry) {
        match entry {
            Entry::Turn => {}
            Entry::Pass => {
                self.new_answer();
                self.thinking_started = false;
                self.ensure_line_start(out);
            }
            Entry::Reasoning(text) => {
                if !self.thinking_started {
                    self.push_text(out, &format!("\r\n{THINKING_STYLE}[Thinking]{RESET}\r\n"));
                    self.thinking_started = true;
                }
                self.push_text(out, &format!("{THINKING_STYLE}{text}{RESET}"));
            }
            Entry::Answer(text) => self.answer(out, text, false),
            Entry::AnswerTail(text) => self.answer(out, text, true),
            Entry::Status(message) => {
                self.push_text(out, &format!("\r\n{STATUS_STYLE}{message}{RESET}\r\n"));
                // A tool boundary starts a new inference pass: its reasoning and
                // reply get fresh headers.
                self.thinking_started = false;
                self.new_answer();
            }
            Entry::Note(text) => {
                self.ensure_line_start(out);
                self.push_text(out, &format!("{text}\r\n"));
            }
            Entry::Line(text) => self.harness_line(out, text),
            Entry::Notice { full, .. } => self.harness_line(out, full),
            Entry::Operator(text) => self.operator_line(out, text),
            Entry::Raw { text, operator } => {
                self.push_text(out, text);
                if *operator {
                    self.thinking_started = false;
                    self.new_answer();
                }
            }
        }
    }

    fn apply_collapsed(&mut self, out: &mut Out, first: Instant, last: Instant, entry: &Entry) {
        // Notices held back for a reply go out before whatever ends its line.
        if !self.deferred.is_empty() && self.ends_reply_line(entry) {
            self.release_deferred(out);
        }
        match entry {
            Entry::Turn => {
                self.settle(out);
                self.ensure_line_start(out);
                self.close_message();
                self.new_answer();
            }
            Entry::Pass => {
                self.settle(out);
                self.new_answer();
                self.ensure_line_start(out);
            }
            Entry::Reasoning(text) => {
                let chars = text.chars().count();
                let stretch = self.stretch_at(first);
                stretch.thought_chars += chars;
                stretch.last = last;
                stretch.now = Now::Thinking;
                self.show_work(out, false);
            }
            Entry::Answer(text) | Entry::AnswerTail(text) => {
                // Only visible reply text ends the stretch: a model can send an
                // empty or whitespace chunk before it thinks (and the markdown
                // renderer can release styled whitespace).
                let mut tail = matches!(entry, Entry::AnswerTail(_));
                if self.answer_started || strip_leading_space(text).1 {
                    let resumed = self.stretch.is_some() && self.answer_started;
                    self.settle(out);
                    if resumed {
                        // Thinking came in the middle of the reply: the rest
                        // goes under the summary, after its own prefix.
                        self.new_answer();
                        tail = false;
                    }
                }
                self.answer(out, text, tail);
            }
            Entry::Status(message) => {
                match parse_tool_status(message) {
                    Some(ToolStatus::Running { name, arguments }) => {
                        let stretch = self.stretch_at(first);
                        stretch.last = last;
                        match stretch.tools.iter_mut().find(|(seen, _)| *seen == name) {
                            Some((_, count)) => *count += 1,
                            None => stretch.tools.push((name.clone(), 1)),
                        }
                        stretch.now = Now::Tool { name, arguments, time: None, outcome: None };
                        // A new step shows at once.
                        self.show_work(out, true);
                    }
                    Some(ToolStatus::Failed { name, error }) => self.tool_outcome(&name, last, |stretch| {
                        stretch.failed += 1;
                        Some(Outcome::Failed(error))
                    }),
                    Some(ToolStatus::Killed { name, .. }) => self.tool_outcome(&name, last, |stretch| {
                        stretch.stopped += 1;
                        Some(Outcome::Killed)
                    }),
                    Some(ToolStatus::Finished { name, time }) => {
                        if let Some(stretch) = self.stretch.as_mut() {
                            stretch.last = last;
                            match &mut stretch.now {
                                Now::Tool { name: current, time: done, .. } if *current == name => *done = Some(time),
                                _ => {}
                            }
                            self.dirty = true;
                        }
                    }
                    // Harness status lines are for the expanded view, and
                    // don't interrupt a reply (subagent notices arrive
                    // mid-stream). Other `[tool]` lines (a rejected batch)
                    // still end the pass: the model is asked again.
                    // A subagent that stopped (paused, failed) and the board's
                    // workflow: drawn collapsed too, as they want the
                    // operator's eye; a subagent's progress stays expanded.
                    None if (message.starts_with("[subagent") && (message.contains("paused") || message.contains("failed")))
                        || message.starts_with("[board]") =>
                    {
                        let brief = crate::terminal::tagged_brief(message);
                        if self.mid_reply() {
                            self.deferred.push((brief, Mark::Work));
                        } else {
                            self.draw_notice(out, &brief, Mark::Work);
                        }
                        return;
                    }
                    None if !message.starts_with("[tool]") => return,
                    None => {}
                }
                self.new_answer();
            }
            Entry::Note(text) => {
                if text.starts_with("[context] folded") {
                    match self.stretch.as_mut() {
                        Some(stretch) => stretch.folded = true,
                        None => self.fold_since = true,
                    }
                    self.dirty = true;
                }
            }
            Entry::Line(text) => {
                self.settle(out);
                self.harness_line(out, text);
            }
            Entry::Operator(text) => self.operator_line(out, text),
            Entry::Notice { brief, mark, .. } => {
                if self.mid_reply() {
                    self.deferred.push((brief.clone(), *mark));
                } else {
                    self.draw_notice(out, brief, *mark);
                }
            }
            Entry::Raw { text, operator } => {
                self.settle(out);
                if *operator {
                    // The operator speaks at the left edge; the agent's answer is
                    // a new message, under its name.
                    self.flush_word(out);
                    self.close_message();
                }
                if !*operator && text == "\r\n" {
                    // A pass that ended early (a steer, a fold) ends its line;
                    // one already ended gets no blank line after it.
                    self.ensure_line_start(out);
                } else {
                    self.drop_trailing(out);
                    self.push_text(out, text);
                }
                if *operator {
                    self.new_answer();
                }
            }
        }
    }

    /// The operator's message: `you>  `, so its text starts in the column
    /// the agent's does (after `s0ma> `), and the lines after the first (wrapped,
    /// or after a line break in it) hang under that column. The agent's answer is
    /// a new message, under its name.
    fn operator_line(&mut self, out: &mut Out, text: &str) {
        self.speaker_line(out, "you", OPERATOR_STYLE, OPERATOR_TEXT_STYLE, text);
    }

    /// Someone's message, as the operator's is drawn: `label>` in
    /// `label_style`, then the text in `text_style`, its later lines hanging
    /// in the gutter. A subagent's tab draws the agent's directions this way.
    pub fn speaker_line(&mut self, out: &mut Out, label: &str, label_style: &str, text_style: &str, text: &str) {
        self.settle(out);
        self.flush_word(out);
        self.close_message();
        self.ensure_line_start(out);
        let lead = format!("{label}>");
        let pad = GUTTER.saturating_sub(UnicodeWidthStr::width(lead.as_str())).max(1);
        out.text.push_str(&format!("{label_style}{lead}{RESET}{}", " ".repeat(pad)));
        let body = format!("{text_style}{text}{RESET}\r\n");
        // Laid out even outside the display: the line starts at the left
        // edge, so the width is all it takes.
        if out.columns >= GUTTER + 10 {
            self.wrap.col = GUTTER;
            self.wrap.indent = GUTTER;
            self.wrap.push(&body, out.columns, out.text);
            self.sync_indent(out);
        } else {
            out.text.push_str(&body);
        }
        self.wrap.col = 0;
        self.thinking_started = false;
        self.new_answer();
    }

    /// A reply is being written and its line isn't finished.
    fn mid_reply(&self) -> bool {
        self.answer_started && !self.open && (self.wrap.col > 0 || self.wrap.holding() || !self.trailing.is_empty())
    }

    /// Whether drawing `entry` ends the reply's current line (collapsed).
    fn ends_reply_line(&self, entry: &Entry) -> bool {
        match entry {
            Entry::Answer(_) | Entry::AnswerTail(_) | Entry::Note(_) => false,
            Entry::Reasoning(text) => !text.is_empty(),
            // Hidden status lines don't interrupt a reply; tool lines end its pass.
            Entry::Status(message) => message.starts_with("[tool]"),
            Entry::Notice { .. } => !self.mid_reply(),
            Entry::Turn | Entry::Pass | Entry::Line(_) | Entry::Operator(_) | Entry::Raw { .. } => true,
        }
    }

    fn release_deferred(&mut self, out: &mut Out) {
        for (brief, mark) in std::mem::take(&mut self.deferred) {
            self.draw_notice(out, &brief, mark);
        }
    }

    /// A notice as a quiet `[*]` line in the gutter, like a work line, a green
    /// `[+]` one for a step that finished, or a `[-]` one for a problem. A long
    /// one wraps under its own text. It leaves the agent's message open: more of it
    /// goes on below without the name.
    fn draw_notice(&mut self, out: &mut Out, brief: &str, mark: Mark) {
        self.settle(out);
        self.ensure_line_start(out);
        let gutter = self.work_gutter(out);
        let (mark_style, mark, text_style) = match mark {
            Mark::Problem => (FAILED_MARK, "[-]", TEXT_STYLE),
            Mark::Done => (OK_MARK, "[+]", WORK_STYLE),
            Mark::Work => (WORK_STYLE, "[*]", WORK_STYLE),
        };
        out.text.push_str(&" ".repeat(gutter));
        out.text.push_str(&format!("{mark_style}{mark}{RESET} "));
        let text = format!("{text_style}{brief}{RESET}\r\n");
        // Wrapped even outside the display: the line starts at the left edge,
        // so the width it was last measured at is enough to go on.
        let hang = gutter + 4;
        if out.columns > hang + 10 {
            self.wrap.col = hang;
            self.wrap.indent = hang;
            self.wrap.push(&text, out.columns, out.text);
            self.sync_indent(out);
        } else {
            out.text.push_str(&text);
        }
        self.wrap.col = 0;
    }

    /// A harness line on a line of its own. A reply it interrupts goes on
    /// below it after its own prefix.
    fn harness_line(&mut self, out: &mut Out, text: &str) {
        self.ensure_line_start(out);
        // At the left edge, like a speaker of its own; what the agent says next
        // comes under its name again.
        self.close_message();
        self.push_text(out, &format!("{text}\r\n"));
        if self.answer_started {
            self.new_answer();
            // Late text too (a held table) starts again after the prefix.
            self.answer_held = Some(String::new());
        }
        // Reasoning that goes on below it gets its header again (expanded).
        self.thinking_started = false;
    }

    /// The stretch in progress, begun at `at` if there is none.
    fn stretch_at(&mut self, at: Instant) -> &mut Stretch {
        let folded = std::mem::take(&mut self.fold_since);
        self.stretch.get_or_insert_with(|| Stretch::new(at, folded))
    }

    /// A failed or killed line for the current tool.
    fn tool_outcome(&mut self, name: &str, last: Instant, mark: impl FnOnce(&mut Stretch) -> Option<Outcome>) {
        let Some(stretch) = self.stretch.as_mut() else { return };
        stretch.last = last;
        let current = matches!(&stretch.now, Now::Tool { name: current, .. } if current == name);
        let outcome = mark(stretch);
        match &mut stretch.now {
            Now::Tool { outcome: slot, .. } if current => *slot = outcome,
            _ => {}
        }
        self.dirty = true;
    }

    /// Bring the work line up to date: open it if it isn't, redraw it now when
    /// `now`, otherwise mark it for the next due redraw.
    fn show_work(&mut self, out: &mut Out, now: bool) {
        if !self.open {
            self.ensure_line_start(out);
            self.open = true;
            self.draw_open(out, Instant::now());
        } else if now {
            self.draw_open(out, Instant::now());
        } else {
            self.dirty = true;
        }
    }

    /// A new reply may follow: its first visible text gets the prefix again.
    fn new_answer(&mut self) {
        self.answer_started = false;
        self.answer_held = None;
    }

    /// The agent's message is over (someone else speaks, or a turn begins): its
    /// next text starts at the left edge under its name.
    fn close_message(&mut self) {
        self.message_open = false;
        self.wrap.indent = 0;
    }

    /// Collapsed, the lines of an open message hang in the gutter.
    fn sync_indent(&mut self, out: &Out) {
        self.wrap.indent = match self.detail {
            Detail::Collapsed if self.message_open => gutter_width(out.columns),
            _ => 0,
        };
    }

    /// The gutter collapsed work lines are drawn in.
    fn work_gutter(&self, out: &Out) -> usize {
        match self.detail {
            Detail::Collapsed => gutter_width(out.columns),
            Detail::Expanded => 0,
        }
    }

    /// Write out the word the wrapper is holding back, where the message
    /// in progress would have put it. Line breaks a reply ended on are
    /// dropped: what comes next starts a line of its own anyway.
    fn flush_word(&mut self, out: &mut Out) {
        self.drop_trailing(out);
        self.sync_indent(out);
        out.text.push_str(&self.wrap.take_pending(out.columns));
    }

    /// Reply text. Collapsed, the line breaks it ends on wait for more of
    /// the reply (see `trailing`).
    fn push_answer(&mut self, out: &mut Out, text: &str) {
        if self.detail == Detail::Expanded {
            self.push_text(out, text);
            return;
        }
        let end = visible_end(text);
        if end == 0 {
            self.trailing.push_str(text);
            return;
        }
        let mut chunk = std::mem::take(&mut self.trailing);
        chunk.push_str(&text[..end]);
        self.push_text(out, &chunk);
        self.trailing.push_str(&text[end..]);
    }

    /// Give up the held line breaks, keeping the styling among them.
    fn drop_trailing(&mut self, out: &mut Out) {
        if self.trailing.is_empty() {
            return;
        }
        let styling = escapes_only(&std::mem::take(&mut self.trailing));
        if !styling.is_empty() {
            self.push_text(out, &styling);
        }
    }

    /// Reply text. `tail`: released late by the markdown renderer, which only
    /// gets the prefix if this pass's reply hasn't shown any text yet.
    fn answer(&mut self, out: &mut Out, text: &str, tail: bool) {
        if self.answer_started {
            if !text.is_empty() {
                self.push_answer(out, text);
            }
            return;
        }
        if tail && self.answer_held.is_none() {
            // Late text for a reply whose prefix went out before an operator
            // message: it continues as it is.
            if !text.is_empty() {
                self.push_answer(out, text);
            }
            return;
        }
        let mut held = self.answer_held.take().unwrap_or_default();
        held.push_str(text);
        let (stripped, visible) = strip_leading_space(&held);
        if !visible {
            self.answer_held = Some(stripped);
            return;
        }
        let prefix = format!("{}{}>{RESET} ", self.name_style, self.name);
        match self.detail {
            Detail::Expanded => {
                if self.thinking_started {
                    self.push_text(out, "\r\n");
                }
                self.push_text(out, &format!("{prefix}{stripped}"));
            }
            // More of the message in progress: on in the gutter, no name.
            Detail::Collapsed if self.message_open => {
                self.ensure_line_start(out);
                self.push_answer(out, &stripped);
            }
            Detail::Collapsed => {
                self.ensure_line_start(out);
                // The name at the left edge, then everything after it in
                // the gutter, from the first line break on.
                self.push_text(out, &prefix);
                self.message_open = true;
                self.push_answer(out, &stripped);
            }
        }
        self.answer_started = true;
        // A fold with no work after it before the reply isn't news for a
        // later stretch.
        self.fold_since = false;
    }

    /// Text through the word wrapper (inside the active display), or straight
    /// through with the wrapper emptied first (outside it).
    pub fn push_text(&mut self, out: &mut Out, text: &str) {
        self.sync_indent(out);
        if out.wrapping {
            self.wrap.push(text, out.columns.max(1), out.text);
        } else {
            out.text.push_str(&self.wrap.take_pending(out.columns));
            out.text.push_str(text);
            // Only whether the line is still open matters out here (for
            // `ensure_line_start`), so the width is rough: styles count.
            self.wrap.col = match text.rfind(['\n', '\r']) {
                Some(at) => UnicodeWidthStr::width(&text[at + 1..]),
                None => self.wrap.col + UnicodeWidthStr::width(text),
            };
        }
    }

    /// Start a fresh line unless already at the start of one.
    fn ensure_line_start(&mut self, out: &mut Out) {
        self.flush_word(out);
        if self.wrap.col > 0 {
            out.text.push_str("\r\n");
            self.wrap.col = 0;
        }
    }

    /// End the stretch of work with its summary line, in place if the line is
    /// still open. Nothing to do in expanded mode.
    pub fn settle(&mut self, out: &mut Out) {
        let Some(stretch) = self.stretch.take() else {
            self.detach();
            return;
        };
        let gutter = self.work_gutter(out);
        let (line, width) = summary_line(&stretch, self.chars_per_token, out.line_width().saturating_sub(gutter));
        if self.open {
            out.text.push_str("\r\x1b[2K");
        } else {
            self.ensure_line_start(out);
        }
        out.text.push_str(&" ".repeat(gutter));
        out.text.push_str(&line);
        self.open = false;
        self.dirty = false;
        self.wrap.col = gutter + width;
    }

    /// Leave the open line as drawn; whatever comes next starts below it.
    fn detach(&mut self) {
        if self.open {
            self.open = false;
            self.wrap.col = self.open_width;
            self.dirty = false;
        }
    }

    /// The cursor is no longer at the end of the open line (a new layout, a
    /// resize): stop rewriting it, without drawing anything.
    pub fn forget_open_line(&mut self) {
        self.open = false;
        self.dirty = false;
    }

    /// The open line has changed and is due a redraw at `now`.
    pub fn wants_refresh(&self, now: Instant) -> bool {
        self.open && self.dirty && self.drawn_at.is_none_or(|drawn| now.duration_since(drawn) >= THINKING_REDRAW)
    }

    /// Redraw the open line if it changed: when due, or always with `force`.
    pub fn refresh(&mut self, out: &mut Out, now: Instant, force: bool) {
        if self.open && self.dirty && (force || self.wants_refresh(now)) {
            self.draw_open(out, now);
        }
    }

    fn draw_open(&mut self, out: &mut Out, now: Instant) {
        let Some(stretch) = self.stretch.as_ref() else { return };
        let gutter = self.work_gutter(out);
        let (line, line_width) = live_line(stretch, self.chars_per_token, out.line_width().saturating_sub(gutter));
        out.text.push_str("\r\x1b[2K");
        out.text.push_str(&" ".repeat(gutter));
        out.text.push_str(&line);
        self.open_width = gutter + line_width;
        self.dirty = false;
        self.drawn_at = Some(now);
    }

    /// End of output for now (a redraw at the idle prompt): settle, and leave
    /// the cursor at the start of a line.
    pub fn finish(&mut self, out: &mut Out) {
        self.release_deferred(out);
        self.settle(out);
        self.ensure_line_start(out);
    }
}

fn tokens(chars: usize, chars_per_token: f64) -> usize {
    if chars == 0 { 0 } else { ((chars as f64 / chars_per_token.max(0.5)).round() as usize).max(1) }
}

/// The work line while it lasts: `[*] thinking · ~240 tokens`, or the current
/// call, `[*] vm_exec(df -h)`, then `[+] … · 1.2s` once it's done, or
/// `[-] … · failed: …` / `[-] … · stopped`.
fn live_line(stretch: &Stretch, chars_per_token: f64, width: usize) -> (String, usize) {
    match &stretch.now {
        Now::Thinking => fit_pieces(
            &[(WORK_STYLE, format!("[*] thinking · {}", format_tokens(tokens(stretch.thought_chars, chars_per_token))))],
            width,
        ),
        Now::Tool { name, arguments, time, outcome } => {
            let (mark, symbol, suffix) = match (time, outcome) {
                (_, Some(Outcome::Failed(error))) => (FAILED_MARK, "[-] ", format!(" · failed: {}", flatten(error, 80))),
                (_, Some(Outcome::Killed)) => (KILLED_MARK, "[-] ", " · stopped".to_string()),
                (Some(time), None) => (OK_MARK, "[+] ", format!(" · {time}")),
                (None, None) => (STATUS_STYLE, "[*] ", String::new()),
            };
            let arguments = compact_arguments(arguments);
            let fixed = UnicodeWidthStr::width(symbol) + UnicodeWidthStr::width("()") + UnicodeWidthStr::width(name.as_str())
                + UnicodeWidthStr::width(suffix.as_str());
            let room = width.saturating_sub(fixed);
            let arguments = if arguments.is_empty() {
                String::new()
            } else if room >= 2 {
                truncate_to_width(&arguments, room)
            } else {
                "…".to_string()
            };
            fit_pieces(
                &[
                    (mark, symbol.to_string()),
                    (TEXT_STYLE, name.clone()),
                    (STATUS_STYLE, format!("({arguments}){suffix}")),
                ],
                width,
            )
        }
    }
}

/// The work line once the stretch is over:
/// `[*] thought for 8s · ~1.2k tokens`, or with calls,
/// `[+] 14s · ran vm_exec ×2, web_search · ~1.2k tokens of thought · 1 failed`.
fn summary_line(stretch: &Stretch, chars_per_token: f64, width: usize) -> (String, usize) {
    let span = format_span(stretch.last.saturating_duration_since(stretch.started));
    let thought = tokens(stretch.thought_chars, chars_per_token);
    let mut main = if stretch.tools.is_empty() {
        format!("[*] thought for {span} · {}", format_tokens(thought))
    } else {
        let mut names: Vec<String> = stretch
            .tools
            .iter()
            .take(SUMMARY_TOOLS)
            .map(|(name, count)| if *count > 1 { format!("{name} ×{count}") } else { name.clone() })
            .collect();
        if stretch.tools.len() > SUMMARY_TOOLS {
            names.push(format!("+{} more", stretch.tools.len() - SUMMARY_TOOLS));
        }
        let mut text = format!("[+] {span} · ran {}", names.join(", "));
        if thought > 0 {
            text.push_str(&format!(" · {} of thought", format_tokens(thought)));
        }
        text
    };
    if stretch.folded {
        main.push_str(" · context folded");
    }
    let mut pieces = vec![(WORK_STYLE, main)];
    if stretch.failed > 0 {
        pieces.push((FAILED_MARK, format!(" · {} failed", stretch.failed)));
    }
    if stretch.stopped > 0 {
        pieces.push((KILLED_MARK, format!(" · {} stopped", stretch.stopped)));
    }
    fit_pieces(&pieces, width)
}

/// Styled pieces cut to `width` columns, with `…` where the cut falls.
/// Returns the text and its display width.
fn fit_pieces(pieces: &[(&str, String)], width: usize) -> (String, usize) {
    let mut out = String::new();
    let mut used = 0;
    for (style, text) in pieces {
        let text_width = UnicodeWidthStr::width(text.as_str());
        if used + text_width <= width {
            out.push_str(&format!("{style}{text}{RESET}"));
            used += text_width;
            continue;
        }
        let cut = truncate_to_width(text, width - used);
        used += UnicodeWidthStr::width(cut.as_str());
        if !cut.is_empty() {
            out.push_str(&format!("{style}{cut}{RESET}"));
        }
        break;
    }
    (out, used)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny terminal: enough of one to show what a stream of renderer output
    /// leaves on screen (line feeds, carriage returns, erase-line, SGR).
    fn screen(bytes: &str) -> Vec<String> {
        let mut lines = vec![Vec::<char>::new()];
        let mut col = 0;
        let mut chars = bytes.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '\r' => col = 0,
                '\n' => {
                    lines.push(Vec::new());
                    col = 0;
                }
                '\x1b' => {
                    let mut sequence = String::new();
                    if chars.peek() == Some(&'[') {
                        chars.next();
                        for c in chars.by_ref() {
                            sequence.push(c);
                            if ('\x40'..='\x7e').contains(&c) {
                                break;
                            }
                        }
                    }
                    if sequence == "2K" {
                        lines.last_mut().unwrap().clear();
                    }
                }
                c => {
                    let line = lines.last_mut().unwrap();
                    while line.len() < col {
                        line.push(' ');
                    }
                    if col < line.len() {
                        line[col] = c;
                    } else {
                        line.push(c);
                    }
                    col += 1;
                }
            }
        }
        let mut lines: Vec<String> =
            lines.into_iter().map(|line| line.into_iter().collect::<String>().trim_end().to_string()).collect();
        while lines.last().is_some_and(String::is_empty) {
            lines.pop();
        }
        lines
    }

    fn render(detail: Detail, entries: &[(u64, Entry)]) -> String {
        let base = Instant::now();
        let mut renderer = Renderer::new(detail, 4.0);
        let mut text = String::new();
        for (millis, entry) in entries {
            let at = base + Duration::from_millis(*millis);
            let mut out = Out { text: &mut text, columns: 80, wrapping: true };
            renderer.apply(&mut out, at, at, entry);
        }
        let mut out = Out { text: &mut text, columns: 80, wrapping: true };
        renderer.finish(&mut out);
        text
    }

    fn reasoning(text: &str) -> Entry {
        Entry::Reasoning(text.to_string())
    }

    fn status(text: &str) -> Entry {
        Entry::Status(text.to_string())
    }

    fn answer(text: &str) -> Entry {
        Entry::Answer(text.to_string())
    }

    #[test]
    fn the_record_is_bounded_and_a_long_session_replays_its_end() {
        let base = Instant::now();
        let mut record = TranscriptRecord::default();
        assert!(record.is_empty());
        record.push(base, Entry::Pass);
        assert!(record.is_empty(), "a pass marker alone is nothing to show");
        for index in 0..500 {
            record.push(base, Entry::Raw { text: format!("{index:>20}\n").repeat(1_000), operator: false });
        }
        assert!(record.bytes <= RECORD_LIMIT && record.dropped);
        let (shown, partial) = record.replay(Detail::Expanded);
        assert!(partial && !shown.is_empty());
        let size: usize = shown.iter().map(|recorded| recorded.entry.size()).sum();
        assert!(size <= REPLAY_BUDGET, "{size}");

        // Reasoning and status cost nothing collapsed, so a long pass replays whole.
        let mut record = TranscriptRecord::default();
        for _ in 0..40 {
            record.push(base, reasoning(&"r".repeat(MERGE_LIMIT)));
        }
        record.push(base, answer("ok"));
        let (shown, partial) = record.replay(Detail::Collapsed);
        assert!(!partial && shown.len() == record.entries.len());
        assert!(record.replay(Detail::Expanded).1);
    }

    #[test]
    fn a_rejected_batch_ends_the_pass_and_the_retry_goes_on_in_the_same_message() {
        let text = render(Detail::Collapsed, &[
            (0, Entry::Pass),
            (0, answer("I'll run both.")),
            (100, status("[tool] batch rejected: invalid JSON in the arguments of vm_exec")),
            (200, answer("Retrying with valid JSON.")),
        ]);
        let lines = screen(&text);
        assert_eq!(lines, ["s0ma> I'll run both.", "      Retrying with valid JSON."]);
    }
}
