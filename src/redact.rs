//! Redacting part of one archive event, at the agent's request and with approval.
//!
//! The agent sometimes sees something in its own past that should not be kept: a
//! credential that came back in a tool result, a detail a client asked not to
//! be held, something pasted by accident. `/job` is for a whole stretch of such
//! work. This is for one event, and usually for part of one -- a key struck out
//! of a command's output, with the rest of the output left as it was.
//!
//! # Nothing is removed without approval
//!
//! The tool **proposes**. It writes a request, tells the operator, and changes
//! nothing; the memory tab (Alt+M, or `/redact`) shows the proposal as a diff
//! against what the event says now, and only `a` there writes anything --
//! once to see everywhere the text is kept, and again to strike it. The reason
//! the agent gives is part of the request, because the operator is being asked to
//! destroy a piece of the record and has to know what for.
//!
//! The operator's own redactions are made in the same tab, by marking text
//! anywhere the agent keeps it (`strike_by_operator`). He approves them by making
//! them; they are recorded here with him as their author.
//!
//! # Why the harness does the striking
//!
//! The agent names the exact strings to remove and this module replaces each one
//! with [`MARKER`]. No model is asked to rewrite the event:
//!
//! * a rewriter would have to be shown the thing being removed, and its output
//!   kept, which is two more copies of it in two more places;
//! * a rewriter can paraphrase, and an event is the exact record -- the one
//!   thing it may not become is a plausible version of itself;
//! * replacement is decidable. Either the string was there and is gone, or it
//!   was not there and the request is refused, and both are checkable.
//!
//! The operator can still edit the proposal by hand (`e`), which is where
//! judgment belongs.
//!
//! # What approving reaches
//!
//! The event is rewritten in place -- same id, same timestamp, same speaker,
//! the approved text as its content. Then the pieces that were actually struck
//! are worked out by diffing what the event said against what was approved,
//! and struck from everywhere else they are kept (`strike.rs`): the other
//! events, what is derived from them, the checkpoints, threads, the world
//! model and its history, episodes, the board, open loops, the typed history,
//! the worker journal, skills, other requests. A piece shorter than
//! `strike::MIN_FRAGMENT` is struck from the event alone.
//!
//! # What it cannot reach
//!
//! * **The request itself, in the saved calls.** the agent had to name the strings
//!   to have them removed, so they are in that tool call's arguments, which is
//!   saved like any other. The call is dropped with the others when the
//!   fragments are long enough to search for, but a short one is not searched
//!   for at all and stays.
//! * **The weakness log**, which is hash-chained and holds a 500-character
//!   excerpt of anything it cites. The review lists it among what is not
//!   reached.

use crate::hippocampus::{content_digest, Hippocampus};
use crate::llm::ToolDefinition;
use crate::types::ToolCall;

use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::json;

use std::fs;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

/// What replaces a struck fragment. Short, so a diff of a long tool result
/// stays readable, and fixed, so it can be searched for afterwards.
pub const MARKER: &str = "[redacted]";

pub const TOOL: &str = "archive_redact";

/// A string shorter than this is refused: `sk-` or `10.0` occurs all over an
/// event and striking every occurrence would take the event with it.
const MIN_FRAGMENT: usize = 4;
const MAX_FRAGMENTS: usize = 20;
const MIN_REASON: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// Waiting for the operator.
    Pending,
    /// Written into the archive.
    Approved,
    /// Turned down; the event is untouched.
    Rejected,
    /// The event moved after the request was made, so the proposal is against
    /// text that is no longer there and cannot be applied.
    Stale,
}

pub fn state_word(state: State) -> &'static str {
    match state {
        State::Pending => "pending",
        State::Approved => "approved",
        State::Rejected => "rejected",
        State::Stale => "stale",
    }
}

/// One version of the event's content as someone would have it: the agent's
/// proposal, then whatever the operator edited it into.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    pub by: String,
    pub at: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub request: u64,
    pub event_id: u64,
    /// The turn the agent asked from, when it asked during one. Provenance for the
    /// review: what it was doing when it noticed is often the thing that makes
    /// the request make sense.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn: Option<String>,
    /// Who said the event, for the window's header.
    pub speaker: String,
    pub at: String,
    /// Why the agent asked. Shown to the operator and kept with the decision.
    pub reason: String,
    pub state: State,
    /// The digest of the event's content when the request was made. The
    /// approval is written only if the event still matches: another redaction,
    /// or a job's close, may have moved it since.
    pub was: String,
    /// How many occurrences of how many strings the proposal struck, so the
    /// window can say so without keeping the strings.
    pub struck: usize,
    /// Oldest first; the last is what approving would write.
    pub versions: Vec<Version>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decided_at: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
    /// Who asked: the agent, through the tool, or the operator, striking text
    /// from the memory tab -- which goes straight through, and is recorded
    /// here so the book is the whole account of what was removed.
    #[serde(default = "s0ma")]
    pub author: String,
    /// Where the text was struck, and where it could not be, once decided:
    /// ids and counts, never the text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub places: Vec<crate::strike::Place>,
}

fn s0ma() -> String {
    "s0ma".into()
}

impl Request {
    /// What approving would write.
    pub fn current(&self) -> &str {
        self.versions.last().map(|version| version.text.as_str()).unwrap_or_default()
    }

    pub fn edited(&self) -> bool {
        self.versions.len() > 1
    }
}

// ------------------------------------------------------------------ the store

/// `data/redactions`, beside the archive.
#[derive(Debug, Clone)]
pub struct Book {
    dir: PathBuf,
}

impl Book {
    pub fn beside_archive(archive: &Path) -> Self {
        Self { dir: archive.with_file_name("redactions") }
    }

    fn path(&self, n: u64) -> PathBuf {
        self.dir.join(format!("{n}.json"))
    }

    pub fn load(&self, n: u64) -> Result<Request> {
        let path = self.path(n);
        let bytes = fs::read(&path).map_err(|error| format!("redaction {n} could not be read: {error}"))?;
        serde_json::from_slice(&bytes).map_err(|error| format!("redaction {n} is unreadable: {error}"))
    }

    pub fn save(&self, request: &Request) -> Result<()> {
        fs::create_dir_all(&self.dir).map_err(|error| format!("cannot create {}: {error}", self.dir.display()))?;
        let bytes = serde_json::to_vec_pretty(request).map_err(|error| error.to_string())?;
        let path = self.path(request.request);
        let temp = path.with_extension("tmp");
        fs::write(&temp, &bytes).map_err(|error| format!("cannot write {}: {error}", temp.display()))?;
        fs::rename(&temp, &path).map_err(|error| format!("cannot replace {}: {error}", path.display()))?;
        if let Ok(dir) = fs::File::open(&self.dir) {
            let _ = dir.sync_all();
        }
        Ok(())
    }

    /// Every request, oldest first.
    pub fn all(&self) -> Result<Vec<Request>> {
        let Ok(entries) = fs::read_dir(&self.dir) else { return Ok(Vec::new()) };
        let mut numbers = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Some(n) = path.file_stem().and_then(|n| n.to_str()).and_then(|n| n.parse::<u64>().ok()) {
                numbers.push(n);
            }
        }
        numbers.sort_unstable();
        numbers.iter().map(|n| self.load(*n)).collect()
    }

    /// The requests still waiting for the operator (the tab bar's mark).
    pub fn pending(&self) -> Result<Vec<Request>> {
        Ok(self.all()?.into_iter().filter(|request| request.state == State::Pending).collect())
    }

    /// Hold `NUMBERING` from this to the `save` of the request it numbers.
    fn next_number(&self) -> Result<u64> {
        Ok(self.all()?.iter().map(|request| request.request).max().unwrap_or(0) + 1)
    }
}

/// Held from taking a request's number to writing it: the agent asks on the
/// app's thread, the operator strikes from the tab's.
static NUMBERING: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ------------------------------------------------------------------ the tool

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![ToolDefinition::function(
        TOOL,
        format!(
            "Ask the operator to remove something from one of your own archive events: a credential that came back in a \
             tool result, a detail a client asked you not to keep, something pasted by accident. Name the exact \
             strings to strike with `remove` and each one is replaced with \"{MARKER}\", leaving the rest of the \
             event as it is; leave `remove` out to propose removing the whole event's content. \
             This changes nothing by itself — it asks. The operator sees your reason beside a diff of what would change, \
             can edit it, and approves or turns it down; the event stays exactly as it is until he does. \
             Use it when something is in the archive that should not be, not for something you merely regret \
             saying, and not to tidy up: every request costs him a review of something he cannot undo. \
             The event keeps its number, its time and who said it either way, so your own past stays accounted \
             for. Approving also strikes the same strings from everywhere else you keep them — other events, the \
             world model, the board, threads, episodes, skills — so you need not ask once per place. The strings \
             you give here are part of this tool call, which is saved like any other, so give what is needed and \
             no more — a short one cannot be searched for and will stay."
        ),
        json!({"type": "object", "properties": {
            "event_id": {"type": "integer", "minimum": 1,
                "description": "The archive event to change (archive_search or archive_get finds it)."},
            "reason": {"type": "string",
                "description": "Why this should not be kept, for the operator to judge. Say what kind of thing it is, not the thing itself."},
            "remove": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_FRAGMENTS,
                "description": format!("The exact strings to strike, each at least {MIN_FRAGMENT} characters, copied from the event. Every occurrence of each goes. Leave this out to propose the whole event's content."),
            }
        }, "required": ["event_id", "reason"]}),
    )]
}

fn rejected(error: impl Into<String>) -> String {
    json!({"ok": false, "requested": false, "error": error.into()}).to_string()
}

pub fn execute(memory: &Hippocampus, call: &ToolCall) -> std::result::Result<String, Box<dyn std::error::Error>> {
    Ok(execute_with(&Book::beside_archive(memory.archive_path()), memory, call))
}

pub fn execute_with(book: &Book, memory: &Hippocampus, call: &ToolCall) -> String {
    #[derive(Deserialize)]
    struct Args {
        event_id: u64,
        reason: String,
        #[serde(default)]
        remove: Vec<String>,
    }
    let args: Args = match serde_json::from_str(&call.function.arguments) {
        Ok(args) => args,
        Err(error) => return rejected(format!("the arguments did not parse: {error}")),
    };
    match request(book, memory, args.event_id, &args.reason, &args.remove) {
        Ok(message) => message,
        Err(error) => rejected(error),
    }
}

fn request(book: &Book, memory: &Hippocampus, event_id: u64, reason: &str, remove: &[String]) -> Result<String> {
    let reason = reason.trim();
    if reason.chars().count() < MIN_REASON {
        return Err(format!("say why, in at least {MIN_REASON} characters: the operator decides on the reason"));
    }
    if remove.len() > MAX_FRAGMENTS {
        return Err(format!("at most {MAX_FRAGMENTS} strings at a time"));
    }

    let existing = book.all()?;
    if let Some(open) = existing
        .iter()
        .find(|request| request.event_id == event_id && request.state == State::Pending)
    {
        return Err(format!(
            "event {event_id} is already waiting for the operator as redaction #{}. Leave it with him rather than \
             asking twice",
            open.request
        ));
    }
    if let Some(name) = crate::jobs::covering_event(memory.last_event_id(), event_id) {
        return Err(format!(
            "event {event_id} is inside job \"{name}\", so its content goes when that job closes and there is \
             nothing to ask for here"
        ));
    }
    let turn = crate::calls::current_turn();

    let Some(event) = memory.get_event(event_id).map_err(|error| error.to_string())? else {
        return Err(format!("there is no event {event_id} in the archive"));
    };
    let speaker = match event.speaker {
        crate::hippocampus::Speaker::Operator => "operator",
        crate::hippocampus::Speaker::S0ma => "s0ma",
    };
    if event.content.trim() == MARKER || event.content.starts_with("[redacted") {
        return Err(format!("event {event_id} has already been redacted"));
    }

    // The proposal, made here rather than by a model: every occurrence of each
    // string, replaced. A string that is not in the event is the interesting
    // case -- it means the agent is remembering the event rather than reading it,
    // and striking nothing while reporting success would be the worst outcome.
    let (proposed, struck) = if remove.is_empty() {
        (MARKER.to_owned(), 1)
    } else {
        let mut text = event.content.clone();
        let mut struck = 0;
        for (index, fragment) in remove.iter().enumerate() {
            if fragment.chars().count() < MIN_FRAGMENT {
                return Err(format!(
                    "the {} string is shorter than {MIN_FRAGMENT} characters; it would match all over the \
                     event. Give enough of it to be unambiguous",
                    nth(index)
                ));
            }
            let occurrences = text.matches(fragment.as_str()).count();
            if occurrences == 0 {
                return Err(format!(
                    "the {} string does not occur in event {event_id}. Read it with archive_get and copy \
                     exactly what is there — nothing was requested",
                    nth(index)
                ));
            }
            text = text.replace(fragment.as_str(), MARKER);
            struck += occurrences;
        }
        (text, struck)
    };

    let now = Local::now().to_rfc3339();
    // A number, and the file under it, one request at a time.
    let _numbering = NUMBERING.lock().map_err(|_| "the redaction book's lock was poisoned".to_string())?;
    let request = Request {
        request: book.next_number()?,
        event_id,
        turn,
        speaker: speaker.to_owned(),
        at: now.clone(),
        reason: reason.to_owned(),
        state: State::Pending,
        was: content_digest(&event.content),
        struck,
        versions: vec![Version { by: "s0ma".into(), at: now, text: proposed }],
        decided_at: None,
        note: String::new(),
        author: s0ma(),
        places: Vec::new(),
    };
    book.save(&request)?;

    // A notice, drawn collapsed too: a request waits for the operator.
    crate::terminal::warn_status(&format!(
        "[redact] s0ma asks to redact event #{event_id} ({speaker}){}: \"{}\" — /redact view {}",
        if remove.is_empty() { ", all of it".into() } else { format!(", {struck} occurrence(s)") },
        brief(reason),
        request.request
    ));

    Ok(json!({
        "ok": true, "requested": true, "request": request.request, "state": "pending",
        "event_id": event_id, "struck": struck,
        "note": "Nothing has changed: the event is as it was until the operator approves this. Carry on with what you \
                 were doing, and don't ask again for this event."
    })
    .to_string())
}

/// "first", "second", … for naming which of the given strings was the problem
/// without quoting it back.
fn nth(index: usize) -> String {
    match index + 1 {
        1 => "first".into(),
        2 => "second".into(),
        3 => "third".into(),
        n => format!("{n}th"),
    }
}

fn brief(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= 80 {
        flat
    } else {
        format!("{}…", flat.chars().take(79).collect::<String>())
    }
}

// ------------------------------------------------------------------ approving

/// What this approval would take out of the archive, worked out by comparing
/// what the event says now against what is about to replace it.
///
/// The struck text is never stored: it is the one thing being got rid of, and
/// keeping a copy to know what to search for would defeat the point. It is
/// recovered when it is needed and dropped again, and because it comes from a
/// diff it covers the operator's own edits as well as the agent's proposal.
pub fn struck_fragments(was: &str, now: &str) -> Vec<String> {
    use similar::{ChangeTag, TextDiff};
    // Each run of the old text that went, whole: a block taken out is one
    // fragment, exactly as it stood, never the words it was made of. Split into
    // words, a block of tool output would strike `[tool:vm_exec]`, `result:`
    // and the like from every event that ever had them.
    let mut runs: Vec<String> = Vec::new();
    let mut run = String::new();
    let mut open = false;
    for change in TextDiff::from_words(was, now).iter_all_changes() {
        match change.tag() {
            ChangeTag::Delete => {
                run.push_str(change.value());
                open = true;
            }
            // What comes in where the run went does not end it.
            ChangeTag::Insert => {}
            ChangeTag::Equal => {
                // Whitespace kept between two deleted pieces is part of the run.
                if open && change.value().trim().is_empty() {
                    run.push_str(change.value());
                    continue;
                }
                if open {
                    runs.push(std::mem::take(&mut run));
                    open = false;
                }
            }
        }
    }
    if open {
        runs.push(run);
    }
    let mut out: Vec<String> = runs
        .into_iter()
        .map(|run| run.trim().to_owned())
        .filter(|run| !run.is_empty() && was.contains(run.as_str()))
        .collect();
    out.sort_by_key(|fragment| std::cmp::Reverse(fragment.len()));
    out.dedup();
    out
}

/// What approving request `n` reaches beyond its own event: every other place
/// the struck text is kept (`strike::survey`). Changes nothing.
///
/// The pieces shorter than `strike::MIN_FRAGMENT` are struck from the event
/// alone, since searching everything for them would match all over it; the
/// returned count says how many there are.
pub fn review(memory: &Hippocampus, book: &Book, n: u64) -> Result<(crate::strike::Impact, usize)> {
    let request = book.load(n)?;
    let Some(event) = memory.get_event(request.event_id).map_err(|error| error.to_string())? else {
        return Err(format!("event {} is no longer in the archive", request.event_id));
    };
    let (fragments, short) = searchable(&event.content, request.current());
    let impact = match fragments {
        Some(fragments) => crate::strike::survey(memory, &fragments)?,
        None => crate::strike::Impact::default(),
    };
    Ok((impact, short))
}

/// The struck pieces long enough to look for everywhere, and how many are not.
fn searchable(was: &str, now: &str) -> (Option<crate::strike::Fragments>, usize) {
    let all = struck_fragments(was, now);
    let (long, short): (Vec<String>, Vec<String>) =
        all.into_iter().partition(|fragment| fragment.chars().count() >= crate::strike::MIN_FRAGMENT);
    (crate::strike::Fragments::new(&long).ok(), short.len())
}

/// Write the approved text into the event, then strike what it took out from
/// everywhere else it is kept. The request is marked either way, so nothing
/// is approved twice.
pub fn approve(memory: &Hippocampus, book: &Book, n: u64) -> Result<(Request, crate::strike::Impact)> {
    let mut request = book.load(n)?;
    if request.state != State::Pending {
        return Err(format!("redaction #{n} is {} already", state_word(request.state)));
    }
    let Some(event) = memory.get_event(request.event_id).map_err(|error| error.to_string())? else {
        return Err(format!("event {} is no longer in the archive", request.event_id));
    };
    if content_digest(&event.content) != request.was {
        request.state = State::Stale;
        request.decided_at = Some(Local::now().to_rfc3339());
        request.note = "the event changed after the request was made".into();
        request.versions.clear();
        book.save(&request)?;
        return Err(format!(
            "event {} has changed since this was proposed, so the proposal is against text that is no longer \
             there. #{n} is marked stale and nothing was written",
            request.event_id
        ));
    }

    let approved = request.current().to_owned();
    let (fragments, short) = searchable(&event.content, &approved);

    // The event first, exactly as approved. If it fails the request stays
    // pending and can be approved again; everything after it is reported and
    // carried past, since stopping half way would leave the event rewritten
    // and no way to tell which of the stores still held it.
    memory
        .replace_event_content(request.event_id, &approved, &request.was)
        .map_err(|error| format!("the archive was not changed ({error}); #{n} is still pending"))?;
    let forgotten = crate::archive::forget_events(memory, &[request.event_id]);
    // What cites the event reads it as redacted, not as tampered with; and
    // the daily review pinned on it is dropped, or deep sleep would refuse.
    let restamped = crate::strike::restamp_one(memory, event.clone());
    let pin = crate::strike::drop_daily_pin(memory.archive_path());

    let mut impact = match &fragments {
        Some(fragments) => crate::strike::apply(memory, fragments).unwrap_or_else(|error| crate::strike::Impact {
            notes: vec![format!("striking it elsewhere failed: {error}")],
            ..Default::default()
        }),
        None => {
            // Nothing long enough to look for elsewhere; what holds this one
            // event still goes.
            crate::context::resume::forget(memory.archive_path().parent().unwrap_or(Path::new(".")));
            crate::strike::Impact::default()
        }
    };
    impact.places.insert(
        0,
        crate::strike::Place {
            store: "archive".into(),
            item: format!("event #{}", request.event_id),
            count: request.struck.max(1),
            event: Some(request.event_id),
            excerpt: None,
        },
    );
    impact.notes.extend(forgotten.failures.into_iter().map(|failure| format!("clearing {failure}")));
    if let Err(error) = restamped {
        impact.notes.push(format!("stamping what cites the event again: {error}"));
    }
    if let Err(error) = pin {
        impact.notes.push(error);
    }
    if short > 0 {
        impact.notes.push(format!(
            "{short} piece(s) under {} characters were struck from this event only",
            crate::strike::MIN_FRAGMENT
        ));
    }

    // Read again: the strike went through the book too, and the agent's reason
    // may have quoted what it was asking to remove.
    let mut request = book.load(n).unwrap_or(request);
    request.state = State::Approved;
    request.decided_at = Some(Local::now().to_rfc3339());
    // The versions held the text that was about to be written; once it is in
    // the archive there is no reason for a second copy of it in here.
    request.versions.clear();
    request.places = impact.for_the_book(fragments.as_ref());
    book.save(&request)?;
    Ok((request, impact))
}

/// The operator's own redaction, from the memory tab: struck from everywhere
/// at once (he is the one who approves), and recorded in the book with him as
/// its author -- where, and how many, and never what.
pub fn strike_by_operator(
    memory: &Hippocampus,
    book: &Book,
    fragments: &crate::strike::Fragments,
    note: &str,
) -> Result<(Request, crate::strike::Impact)> {
    // The number first, held until the record is written: a book that cannot
    // be read is found out before anything is struck, not after, and a
    // request the agent makes meanwhile cannot take the same number.
    let _numbering = NUMBERING.lock().map_err(|_| "the redaction book's lock was poisoned".to_string())?;
    let number = book.next_number()?;
    let impact = crate::strike::apply(memory, fragments)?;
    let now = Local::now().to_rfc3339();
    let request = Request {
        request: number,
        event_id: impact.places.iter().find_map(|place| place.event).unwrap_or(0),
        turn: None,
        speaker: String::new(),
        at: now.clone(),
        reason: if note.trim().is_empty() { "struck from the memory tab".into() } else { note.trim().to_owned() },
        state: State::Approved,
        was: String::new(),
        struck: impact.total(),
        versions: Vec::new(),
        decided_at: Some(now),
        note: String::new(),
        author: "operator".into(),
        places: impact.for_the_book(Some(fragments)),
    };
    book.save(&request)?;
    Ok((request, impact))
}

/// Every request's words -- the agent's reason, a proposal still waiting, a note --
/// given what `strike` returns for them. With `apply`, saved. Returns, for
/// each request that held something, its number and how many occurrences.
///
/// A waiting proposal is a copy of its event's text, so it is as much a place
/// the text is kept as the event is.
pub fn strike_book(
    book: &Book,
    strike: &dyn Fn(&str) -> Option<(String, usize)>,
    apply: bool,
) -> Result<Vec<(u64, usize)>> {
    let mut touched = Vec::new();
    for mut request in book.all()? {
        let mut count = 0;
        let mut one = |text: &mut String| {
            if let Some((struck, n)) = strike(text) {
                *text = struck;
                count += n;
            }
        };
        one(&mut request.reason);
        one(&mut request.note);
        for version in &mut request.versions {
            one(&mut version.text);
        }
        if count > 0 {
            touched.push((request.request, count));
            if apply {
                book.save(&request)?;
            }
        }
    }
    Ok(touched)
}

pub fn reject(book: &Book, n: u64, note: &str) -> Result<Request> {
    let mut request = book.load(n)?;
    if request.state != State::Pending {
        return Err(format!("redaction #{n} is {} already", state_word(request.state)));
    }
    request.state = State::Rejected;
    request.decided_at = Some(Local::now().to_rfc3339());
    request.note = note.trim().to_owned();
    // Nothing was written, and the proposal is of no further use: keeping it
    // would leave a copy of the event's text in a second file for no reason.
    request.versions.clear();
    book.save(&request)?;
    Ok(request)
}

/// Record the operator's own version of the text. Only while it is pending.
pub fn save_edit(book: &Book, n: u64, text: &str) -> Result<Request> {
    let mut request = book.load(n)?;
    if request.state != State::Pending {
        return Err(format!("redaction #{n} is {} already", state_word(request.state)));
    }
    request.versions.push(Version {
        by: "operator".into(),
        at: Local::now().to_rfc3339(),
        text: text.to_owned(),
    });
    book.save(&request)?;
    Ok(request)
}

// ------------------------------------------------------------------ the operator's command

const HELP: &str = "\
Redactions: s0ma asks, you decide, and nothing changes until you do. Both are
in the memory tab (Alt+M), which /redact opens at its redactions.

/redact                 the memory tab, at the redactions
/redact view [n]        the same, at request n
/redact list [all]      one line each; default is what is waiting
/redact help            this text (also -h, --help, ?)

There: a shows everywhere the text is kept, a again approves · e edit · d decline.
Anywhere in the tab, ⏎ puts a cursor in the text; Shift+arrows mark, ⏎ keeps the
marked text, and a / a strikes it from every store at once.

Approving rewrites that one event, keeping its number, its time and who said
it, and strikes what it took out from everywhere else s0ma keeps it: the other
events and what is derived from them, the checkpoints, threads, the world model
and its history, episodes, the board, open loops, the typed history, the worker
journal and skills. The weakness log is hash-chained and cannot be rewritten:
the review says when it holds the text.";

/// `/redact` or `/redact view [n]`: the memory tab, at the redactions (and
/// at request n). For a terminal that opens it itself (`session.rs`, the
/// steering pane), where `handle_command` would open it on the agent's own.
pub fn tab_request(command: &str) -> Option<Option<u64>> {
    let mut words = command.split_whitespace();
    if words.next()? != "/redact" {
        return None;
    }
    match (words.next(), words.next(), words.next()) {
        (None, _, _) | (Some("view"), None, _) => Some(None),
        (Some("view"), Some(n), None) => n.trim_start_matches('#').parse().ok().map(Some),
        _ => None,
    }
}

/// Returns None if `command` isn't a /redact command, and `Reply::View` when
/// the window has to be opened, which only the caller can do.
pub fn handle_command(memory: &Hippocampus, command: &str) -> Option<Reply> {
    let mut words = command.split_whitespace();
    if words.next()? != "/redact" {
        return None;
    }
    let args: Vec<&str> = words.collect();
    let book = Book::beside_archive(memory.archive_path());
    Some(match args.first().copied() {
        None => Reply::View(None),
        Some("help" | "-h" | "--help" | "?") => Reply::Text(HELP.to_owned()),
        Some("view") => match args.get(1) {
            None => Reply::View(None),
            Some(n) => match n.trim_start_matches('#').parse::<u64>() {
                Ok(n) => Reply::View(Some(n)),
                Err(_) => Reply::Error(format!("[redact] {n} is not a request number")),
            },
        },
        Some("list") => match list(&book, args.get(1).copied() == Some("all")) {
            Ok(text) => Reply::Text(text),
            Err(error) => Reply::Error(format!("[redact] {error}")),
        },
        Some(other) => Reply::Error(format!("[redact] unknown subcommand '{other}'\n{HELP}")),
    })
}

pub enum Reply {
    Text(String),
    Error(String),
    /// Open the review window, at this request if given.
    View(Option<u64>),
}

fn list(book: &Book, all: bool) -> Result<String> {
    let requests = book.all()?;
    let shown: Vec<&Request> =
        requests.iter().filter(|request| all || request.state == State::Pending).collect();
    if shown.is_empty() {
        return Ok(if all {
            "[redact] none yet".into()
        } else {
            "[redact] nothing waiting (/redact list all shows the decided ones)".into()
        });
    }
    Ok(shown
        .iter()
        .map(|request| {
            format!(
                "#{:<4} event #{:<7} {:<9} {}",
                request.request,
                request.event_id,
                state_word(request.state),
                brief(&request.reason)
            )
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionCall, ToolCall};

    struct Fixture {
        dir: PathBuf,
        memory: Hippocampus,
        book: Book,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("s0ma-redact-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            let memory = Hippocampus::new(dir.join("hippocampus.jsonl")).unwrap();
            let book = Book::beside_archive(memory.archive_path());
            Self { dir, memory, book }
        }

        fn say(&mut self, text: &str) -> u64 {
            self.memory.remember_s0ma(text.to_string()).unwrap();
            self.memory.last_event_id()
        }

        fn ask(&self, arguments: serde_json::Value) -> serde_json::Value {
            let call = ToolCall {
                id: "c1".into(),
                kind: "function".into(),
                function: FunctionCall { name: TOOL.into(), arguments: arguments.to_string() },
            };
            serde_json::from_str(&execute_with(&self.book, &self.memory, &call)).unwrap()
        }

        fn content(&self, id: u64) -> String {
            self.memory.get_event(id).unwrap().expect("the event is there").content
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    const REASON: &str = "a live credential came back in this command's output and should not be kept";

    #[test]
    fn a_request_changes_nothing_until_it_is_approved() {
        let mut fixture = Fixture::new("gate");
        let said = "[tool:vm_shell]\nresult:\nexport TOKEN=sk-live-9f3a2b7c\ndeploy finished";
        let id = fixture.say(said);

        let out = fixture.ask(json!({"event_id": id, "reason": REASON, "remove": ["sk-live-9f3a2b7c"]}));
        assert_eq!(out["requested"], true, "{out}");
        assert_eq!(out["state"], "pending");
        assert_eq!(out["struck"], 1);
        // The archive is untouched: asking is not doing.
        assert_eq!(fixture.content(id), said);

        let request = fixture.book.load(1).unwrap();
        assert_eq!(request.state, State::Pending);
        assert_eq!(request.current(), "[tool:vm_shell]\nresult:\nexport TOKEN=[redacted]\ndeploy finished");
        assert!(!request.current().contains("sk-live"), "the proposal is what would be written");

        let (_, applied) = approve(&fixture.memory, &fixture.book, 1).unwrap();
        assert_eq!(
            fixture.content(id),
            "[tool:vm_shell]\nresult:\nexport TOKEN=[redacted]\ndeploy finished",
            "only the named string went"
        );
        assert_eq!(applied.places[0].item, format!("event #{id}"), "{applied:?}");
        assert_eq!(fixture.book.load(1).unwrap().state, State::Approved);
    }

    #[test]
    fn the_same_event_is_not_queued_twice_and_a_redacted_one_is_refused() {
        let mut fixture = Fixture::new("twice");
        let id = fixture.say("a key sk-dup111222 sits in here somewhere");
        fixture.ask(json!({"event_id": id, "reason": REASON, "remove": ["sk-dup111222"]}));

        let out = fixture.ask(json!({"event_id": id, "reason": REASON, "remove": ["sk-dup111222"]}));
        assert_eq!(out["requested"], false, "{out}");
        assert!(out["error"].as_str().unwrap().contains("already waiting"), "{out}");

        approve(&fixture.memory, &fixture.book, 1).unwrap();
        let out = fixture.ask(json!({"event_id": id, "reason": REASON, "remove": ["somewhere here"]}));
        assert_eq!(out["requested"], false, "{out}");
    }

    #[test]
    fn an_event_that_moved_since_the_request_is_refused_and_marked_stale() {
        let mut fixture = Fixture::new("stale");
        let id = fixture.say("key sk-stale11122 and some other text");
        fixture.ask(json!({"event_id": id, "reason": REASON, "remove": ["sk-stale11122"]}));

        // Something else changed the event meanwhile — another redaction, or a
        // job's close.
        fixture.memory.redact_ranges(&[(id, id)], "[redacted for job acme]").unwrap();

        let error = approve(&fixture.memory, &fixture.book, 1).unwrap_err();
        assert!(error.contains("changed since"), "{error}");
        assert_eq!(fixture.book.load(1).unwrap().state, State::Stale);
        assert_eq!(fixture.content(id), "[redacted for job acme]", "the approval did not overwrite it");
    }

    /// The event approving rewrote is cited by a world record: the record's
    /// evidence still reads as unchanged, not as tampered with.
    #[test]
    fn approving_keeps_what_cites_the_event_reading_unchanged() {
        let mut fixture = Fixture::new("cited");
        let id = fixture.say("deploy key sk-cited-4455667 in the output");
        let world = crate::world_model::path_for(&fixture.memory);
        crate::world_model::define(&world, "infra", "Infra", "keys", 0).unwrap();
        let record: crate::world_model::Record = serde_json::from_value(json!({
            "id": "deploy", "model_id": "infra", "kind": "claim", "title": "deploy",
            "statement": "a deploy key was printed", "basis": "observed", "status": "supported",
            "supporting_evidence_ids": [id]
        }))
        .unwrap();
        crate::world_model::put(&world, record, 0, "seen it", &fixture.memory).unwrap();
        fs::write(fixture.dir.join("daily_sleep_window.json"), "{}").unwrap();
        fixture.ask(json!({"event_id": id, "reason": REASON, "remove": ["sk-cited-4455667"]}));
        approve(&fixture.memory, &fixture.book, 1).unwrap();
        let call = ToolCall {
            id: "g".into(),
            kind: "function".into(),
            function: FunctionCall { name: "world_record_get".into(), arguments: json!({"id": "deploy"}).to_string() },
        };
        let got: serde_json::Value = serde_json::from_str(&crate::world_model::execute(&fixture.memory, &call).unwrap()).unwrap();
        let got: serde_json::Value = serde_json::from_str(got["content"].as_str().unwrap()).unwrap();
        assert_eq!(got["evidence_integrity"], "unchanged", "{got}");
        assert!(!fixture.dir.join("daily_sleep_window.json").exists(), "the pinned review would refuse every deep sleep");
    }
}
