//! Folding: turning the oldest live turns into first-person gists.
//!
//! `memory_layers` owns the storage, the planning and the merge cascade, and
//! stays testable without a server. This module is the part that needs one: it
//! writes the gists, and it drives a fold from end to end.
//!
//! The shape of a fold:
//!
//! 1. Plan a cut over the oldest live turns at a legal boundary.
//! 2. Write one gist per episode, on a maintenance worker. Past the schedule's
//!    `prepare` point (`context_guard::FoldSchedule`) this happens ahead of time
//!    on a background thread (`FoldPrep`), so the fold itself only has to store
//!    what was written. On a long window the fold waits for those gists rather
//!    than stopping a turn to write them, up to the schedule's `ceiling`.
//! 3. Store them with the archive range each episode actually covers, and
//!    store what each stretch left unsettled as NOT SETTLED items, apart from
//!    the gists, where no merge can drop or harden them.
//! 4. Run the merge cascade, drop the folded turns, rebuild the memory message,
//!    and measure. If the result is still over target, plan a bigger cut and go
//!    again.
//!
//! Nothing is cleared. Everything newer than the cut stays exactly where it was,
//! as real conversation turns, which is what makes consecutive moments share
//! almost all of their context.
//!
//! Epistemic status is carried two ways. Inside a gist, claims are tagged with
//! how they were known ([observed #N], [operator said], [inferred], [guess]); that
//! is best effort, because a merge rewrites prose. What a stretch left open is
//! carried structurally instead, as items that leave only when a later gist
//! settles them on evidence from its own stretch, calls them moot, or deep sleep
//! takes them over into the world model.

use crate::context_guard::{
    self, ContextUsage, FOLD_RESERVE_TOKENS, FoldNeed, FoldSchedule, MIN_LIVE_TURNS_TOKENS, PREFIX_BUDGET_TOKENS,
};
use crate::hippocampus::Hippocampus;
use crate::llm::{self, ChatOptions, ToolDefinition};
use crate::memory_layers::{
    self, FoldPlan, Gist, ItemState, Level, MemoryLayers, NewGist, NewOpenItem, OpenItem,
    Sources, Summarizer, cited_observations, estimate_tokens, keep_citations, mechanical_gist,
};
use crate::open_loops::LoopStore;
use crate::types::Message;

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Output allowance for one gist. An episode is ~40k of turns at about 10:1, so
/// 4k of room is generous for a 4k answer.
const GIST_OUTPUT: u32 = 4_096;

/// Output allowance for a merge. Shape entries are smaller than gists.
const MERGE_OUTPUT: u32 = 2_048;

/// How many times a fold will re-plan with a bigger cut before giving up on
/// reaching the target. Each attempt is a real model call per episode, so this
/// is deliberately small.
const MAX_FOLD_ATTEMPTS: usize = 3;

/// How long a fold will wait for a worker lane before giving up and writing a
/// mechanical stub instead.
///
/// The design called for carving `FOLD_RESERVE_TOKENS` permanently out of the
/// subagent pool so a fold never waits. Measured against the real numbers that
/// is a bad trade: the pool is 377,856 tokens, and a static 75k reserve drops
/// concurrent workers from nine to seven — a permanent cost, to avoid waiting for
/// something that runs roughly every 145k tokens.
///
/// So the lane is borrowed rather than reserved, and the starvation the reserve
/// was for is handled directly: a fold waits for a lane instead of immediately
/// falling back. Falling back is what actually hurts — it puts a hole in memory
/// because a subagent batch happened to be running.
const LEASE_WAIT: std::time::Duration = std::time::Duration::from_secs(120);
const LEASE_POLL: std::time::Duration = std::time::Duration::from_millis(500);

/// Borrow a maintenance lane, waiting for one if they are all busy. A set
/// `cancel` flag ends the wait at once.
fn wait_for_lease(cancel: Option<&AtomicBool>) -> Result<crate::subagent::MaintenanceLease> {
    let deadline = std::time::Instant::now() + LEASE_WAIT;
    loop {
        if cancel.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
            return Err("cancelled".into());
        }
        if let Some(lease) = crate::subagent::reserve_maintenance(FOLD_RESERVE_TOKENS, 16_384)? {
            return Ok(lease);
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "no worker lane came free within {}s; every lane is held by subagents or sleep",
                LEASE_WAIT.as_secs(),
            )
            .into());
        }
        std::thread::sleep(LEASE_POLL);
    }
}

/// Tool results are capped at 8k tokens by `limit_tool_output`, but a whole
/// episode of them still shouldn't be pasted verbatim into a gist prompt. This
/// bounds each one in the rendered transcript.
const TRANSCRIPT_TOOL_CHARS: usize = 2_000;

/// The same for reasoning: enough of each step's thinking for the gist to keep
/// what s0ma meant to do and why, its head and its conclusion.
const TRANSCRIPT_REASONING_CHARS: usize = 1_500;

// ------------------------------------------------------------------ summarizer

/// Writes gists and merges with a model call on a maintenance worker.
pub struct ModelSummarizer {
    model: String,
    /// Set to stop a background writer before its next lane is borrowed.
    cancel: Option<Arc<AtomicBool>>,
}

impl ModelSummarizer {
    pub fn new() -> Self {
        Self { model: llm::model_name(), cancel: None }
    }

    fn cancellable(cancel: Arc<AtomicBool>) -> Self {
        Self { model: llm::model_name(), cancel: Some(cancel) }
    }

    /// One call on a borrowed maintenance slot.
    ///
    /// Reasoning is off. A gist is an account of what just happened, written from
    /// a transcript that is right there — it is recall and compression, not
    /// analysis, and the same reasoning budget that exhausted the digest rollup
    /// would exhaust this.
    fn generate(&self, system: &str, user: String, max_tokens: u32) -> Result<String> {
        let mut lease = wait_for_lease(self.cancel.as_deref())?;

        let mut options = ChatOptions::sleep();
        options.max_tokens = max_tokens;
        options.reasoning_effort = "none";

        let messages = vec![Message::system(system.to_string()), Message::user(user)];
        let outcome = llm::chat_sleep_cancel(&messages, &options, lease.slot, self.cancel.as_deref());

        // Release the slot whether or not the call worked; a lost lease would
        // starve every later fold.
        let released = lease.release();
        let response = outcome?;
        released?;

        let content = response.content.trim();
        if content.is_empty() {
            return Err("the model returned nothing".into());
        }
        if response.finish_reason.as_deref() == Some("length") {
            return Err(format!(
                "ran out of its {max_tokens}-token allowance ({} characters written)",
                content.chars().count()
            )
            .into());
        }
        Ok(content.to_string())
    }
}

impl Default for ModelSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

/// What a gist is for, said to the model writing one.
///
/// Three things carry the weight. It is written in the first person, because it
/// becomes s0ma's own memory of that stretch rather than a report about it. It
/// says how each thing was known, because a summary under a word budget drops
/// qualifiers first, and a conclusion that loses its "I think" reads as a fact
/// from then on. And it is honest about being coarse: the archive holds the exact
/// record, and a gist that reads as complete invites filling in the details it
/// dropped — the hard case of confabulation.
///
/// The sections at the end are parsed by `parse_gist`; their headings must
/// match it.
const GIST_SYSTEM: &str = "You are s0ma, writing your own memory of a stretch of conversation \
    that is about to leave your working context.\n\n\
    Write a first-person account of what happened: what you were doing, what you found, what \
    was decided, what changed your mind. Keep every specific that would be expensive to recover \
    — names, IDs, file paths, numbers, error messages, the outcome of anything you tried. Keep \
    the order things happened in.\n\n\
    Say how you knew each thing that matters, with a tag after the claim:\n\
    - [observed #N] you saw it in a tool result. N is the archive_event_id printed at the end of \
    that result; copy the real number, never a guessed one. If the result shows no event id, \
    write [observed] alone.\n\
    - [operator said] the operator told you and you did not check it.\n\
    - [inferred] your own conclusion from the evidence, not checked directly. Write \
    [inferred, untested] when it was meant to be tested and wasn't.\n\
    - [guess] a hunch with little behind it.\n\
    In the transcript, \"tool result\" lines are observations, \"The operator\" lines are the operator's own words, \
    and \"me\" lines are your own claims: a claim of yours is an inference unless a tool result in \
    the transcript backs it. \"me (thinking)\" lines are your private reasoning at the time, cut \
    short: use them for what you meant to do and why, and for plans still in progress, never as \
    evidence that something is so. \"runtime\" lines are the harness talking (notices, subagent results, \
    environment events, autonomous prompts). They are never [operator said], and a subagent's \
    conclusion is its inference, not your observation.\n\n\
    Keep the confidence you had at the time. Do not turn a hypothesis into a finding, a plan into \
    something done, or a report into a fact. Where you were unsure, say so. Where you changed \
    your mind, keep the earlier view and what changed it, not just the final one. A tool result \
    that says it was truncated, or shows \"characters omitted\", is a preview: never conclude from \
    one that something was absent, complete or failed.\n\n\
    This is your outline of that stretch, not the record; the archive has the exact text. Do not \
    write it as though it were complete, and do not invent detail to make it read well. If you \
    dropped something, say that it happened without its specifics rather than reconstructing \
    specifics you no longer have.\n\n\
    After the account, list what this stretch left unsettled, under this heading on a line of \
    its own. Don't repeat items already listed as not settled: they stay open on their own.\n\n\
    NOT SETTLED\n\
    - one line per open question, untested conclusion or unchecked report, with its tag and what \
    would settle it. Don't give new items a U-number; numbers are assigned for you.\n\n\
    If this stretch settled any of the items listed as currently not settled (U-numbers), add:\n\n\
    SETTLED\n\
    - U12: what settled it, citing the [observed #N] of the tool result that shows it, or \
    [operator said]\n\n\
    If one of them stopped mattering (the plan changed, the question went away), add:\n\n\
    MOOT\n\
    - U12: why it no longer matters\n\n\
    Settling needs evidence from this stretch: a tool result shown in it ([observed #N]) or the operator \
    saying so in it ([operator said]). Your own reasoning is not enough, and an item without that \
    evidence stays open. Settle items one U-number at a time. Leave out any section that would be \
    empty. No preamble and no other headings.";

/// What a merge is for.
///
/// The second paragraph is the point: under a halved word budget, qualifiers
/// read as detail and go first. `merge_into` checks the one failure it can see
/// mechanically — every tentative tag gone — and marks the entry if it happens.
const MERGE_SYSTEM: &str = "You are s0ma, compressing your own older memories further to make \
    room.\n\n\
    You are given accounts of consecutive stretches, oldest first. Combine them into one \
    shorter account of the whole span, in the first person. Keep the shape of what happened and \
    any specific that would be expensive to recover. Drop detail evenly rather than dropping \
    whole stretches — the beginning of the span matters as much as the end.\n\n\
    Qualifiers are not detail. Every claim you keep keeps its tag ([observed #N], [operator said], \
    [inferred], [guess]); if there is no room for the tag, drop the claim, never the tag. A merge \
    never raises confidence: what one account suspected stays suspected unless a later account \
    says what confirmed it, and then keep what confirmed it. Keep corrections and changes of mind \
    rather than smoothing them into one consistent story.\n\n\
    Open questions are tracked separately and are not your job here: do not answer, settle or \
    drop them, and do not present anything as resolved that the accounts leave open.\n\n\
    This is a coarser memory than what you started from, and the archive still holds the exact \
    record. Do not present it as complete, and do not invent detail to bridge the pieces.\n\n\
    Write only the combined account.";

impl Summarizer for ModelSummarizer {
    fn gist(&self, transcript: &str, open_items: &str) -> Result<String> {
        let open = if open_items.trim().is_empty() {
            "Nothing is currently listed as not settled.".to_string()
        } else {
            format!(
                "Currently not settled, from earlier in this session. Settle or moot one only on \
                 evidence from the stretch below:\n{open_items}"
            )
        };
        self.generate(
            GIST_SYSTEM,
            format!("{open}\n\nThe stretch of conversation leaving your context:\n\n{transcript}"),
            GIST_OUTPUT,
        )
    }

    fn merge(&self, level: Level, inputs: &[Gist]) -> Result<String> {
        let mut body = String::new();
        // Each headed by its dates (merge_into puts them in `time_label`), or
        // by its range alone when the archive couldn't say.
        for gist in inputs {
            let when = match gist.time_label.trim() {
                "" => String::new(),
                when => format!("{when} | "),
            };
            body.push_str(&format!(
                "\n\n--- {when}archive events {} ---\n{}",
                gist.range_label(),
                gist.text.trim(),
            ));
        }
        let _ = level;
        self.generate(MERGE_SYSTEM, format!("Accounts to combine, oldest first:{body}"), MERGE_OUTPUT)
    }

    fn time_label(&self, inputs: &[Gist]) -> String {
        // Only when the archive couldn't date the merge (merge_into dates it
        // otherwise): the oldest input's stored label. History only: what
        // s0ma sees is dated from the archive at render time.
        inputs.first().map(|gist| gist.time_label.clone()).unwrap_or_else(|| "earlier".to_string())
    }

    fn model_name(&self) -> String {
        self.model.clone()
    }
}

// ------------------------------------------------------------------ transcript

fn render_message(message: &Message) -> String {
    let speaker = match message.role.as_str() {
        // Autonomous prompts, subagent results and harness notices are user
        // messages too. Only what the operator wrote is the operator.
        "user" if message.is_operator_input() => "The operator",
        "user" => "runtime",
        "assistant" => "me",
        "tool" => "tool result",
        other => other,
    };

    let mut out = String::new();
    // Thinking comes first, as it did: what s0ma meant to do, then the call.
    let reasoning = message.reasoning_content.as_deref().map(str::trim);
    if let Some(reasoning) = reasoning.filter(|r| message.role == "assistant" && !r.is_empty()) {
        out.push_str(&format!("me (thinking): {}\n", shorten(reasoning, TRANSCRIPT_REASONING_CHARS)));
    }
    if let Some(calls) = &message.tool_calls {
        for call in calls {
            out.push_str(&format!(
                "me (tool call): {}({})\n",
                call.function.name,
                shorten(&call.function.arguments, TRANSCRIPT_TOOL_CHARS),
            ));
        }
    }

    let content = message.content.trim();
    if !content.is_empty() {
        let text = if message.role == "tool" {
            shorten_tool_result(content, TRANSCRIPT_TOOL_CHARS)
        } else {
            content.to_string()
        };
        out.push_str(&format!("{speaker}: {text}\n"));
    }
    out
}

/// Keep the head and the tail of a long text and say how much of the middle
/// went. The end of a result is often where its outcome is (an exit status, the
/// last error), so cutting only from the end loses the part that matters most.
fn shorten(text: &str, chars: usize) -> String {
    let total = text.chars().count();
    if total <= chars {
        return text.to_string();
    }
    let head_chars = chars * 3 / 4;
    let tail_chars = chars - head_chars;
    let head: String = text.chars().take(head_chars).collect();
    let tail: String = text.chars().skip(total - tail_chars).collect();
    format!(
        "{head}\n…[{} characters omitted from the middle; the archive has all of it]…\n{tail}",
        total - head_chars - tail_chars,
    )
}

/// Shorten a tool result's body and keep its trailer whole.
///
/// The trailer names the archive event that holds the result, which is the only
/// thing that lets a gist cite `[observed #N]`. Cutting from the end used to take
/// it off every result longer than the limit.
fn shorten_tool_result(content: &str, chars: usize) -> String {
    let (body, trailer) = crate::observation_mask::split_trailer(content);
    format!("{}{trailer}", shorten(body, chars))
}

fn transcript(messages: &[Message]) -> String {
    messages.iter().map(render_message).collect()
}

/// One episode as a gist writer sees it.
struct EpisodeText {
    transcript: String,
    /// Whether the operator said anything in it, which is what lets `[operator said]`
    /// settle an item.
    operator_spoke: bool,
}

fn operator_spoke(messages: &[Message]) -> bool {
    messages.iter().any(Message::is_operator_input)
}

fn episode_texts(live: &[Message], plan: &FoldPlan) -> Vec<EpisodeText> {
    plan.episodes
        .iter()
        .map(|episode| {
            let slice = &live[episode.start..episode.end];
            EpisodeText { transcript: transcript(slice), operator_spoke: operator_spoke(slice) }
        })
        .collect()
}

// ------------------------------------------------------------------ reading a gist

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Narrative,
    NotSettled,
    Settled,
    Moot,
}

/// A NOT SETTLED entry. `id` is the U-number it leads with, if any.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Opened {
    pub id: Option<i64>,
    pub text: String,
}

/// A gist as the model wrote it, split into its account and its sections.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ParsedGist {
    pub narrative: String,
    /// What this stretch left open, as written; see `new_items`.
    pub opened: Vec<Opened>,
    /// Earlier items this stretch says it settled: (id, what settled it).
    pub settled: Vec<(i64, String)>,
    /// Earlier items this stretch says stopped mattering: (id, why).
    pub moot: Vec<(i64, String)>,
}

/// A section heading, leniently: models decorate headings with `#`, `**` or a
/// trailing colon even when asked not to, and sometimes gloss the first one
/// ("Not settled (open questions):"). SETTLED and MOOT must stand alone, so a
/// sentence that begins "Settled on…" stays part of the account.
fn section_heading(line: &str) -> Option<Section> {
    let bare = line
        .trim()
        .trim_start_matches(['#', '*', ' '])
        .trim_end_matches(['*', ':', ' ']);
    let upper = bare.to_ascii_uppercase();
    match upper.as_str() {
        "NOT SETTLED" => return Some(Section::NotSettled),
        "SETTLED" => return Some(Section::Settled),
        "MOOT" => return Some(Section::Moot),
        _ => {}
    }
    // A gloss has to look like one — in brackets or after a dash — and the
    // line can't read as a sentence, or "Not settled whether it was the lock."
    // in the account would turn the rest of the account into items.
    let glossed = upper.strip_prefix("NOT SETTLED").is_some_and(|rest| {
        bare.chars().count() <= 48
            && rest.trim_start().starts_with(['(', '-', '—', '–'])
            && !bare.ends_with(['.', '!', '?'])
    });
    glossed.then_some(Section::NotSettled)
}

/// An entry that says there is nothing to list: "None.", "None — all resolved",
/// "No open questions remain.", "Nothing else is unsettled."
///
/// Careful on purpose, because dropping a real item is the failure that
/// matters: an entry has to open with none / nothing / no, carry no tag, and
/// either be very short, go on after a dash or colon, or be a short sentence
/// about there being nothing open. "Nothing confirms the retry fix works" is
/// an item, and stays.
fn says_none(entry: &str) -> bool {
    let lower = entry.trim().trim_end_matches(['.', '!']).to_lowercase();
    if lower.contains('[') {
        return false;
    }
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '/' && c != '\'')
        .filter(|word| !word.is_empty())
        .collect();
    let Some(first) = words.first() else {
        return true;
    };
    if !matches!(*first, "none" | "nothing" | "no" | "n/a") {
        return false;
    }
    if words.len() <= 2 || lower.trim_start_matches(first).trim_start().starts_with([':', '—', '–', '-']) {
        return true;
    }
    const EMPTY: [&str; 12] = [
        "open", "unsettled", "remain", "remains", "left", "else", "further", "new", "outstanding",
        "pending", "know", "unresolved",
    ];
    words.len() <= 6 && words.iter().any(|word| EMPTY.contains(word))
}

/// A list bullet: `-`, `*`, `•`, or a number (`1.` / `1)`).
fn strip_bullet(line: &str) -> Option<&str> {
    if let Some(rest) = ["- ", "* ", "• "].iter().find_map(|mark| line.strip_prefix(mark)) {
        return Some(rest);
    }
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 || digits > 3 {
        return None;
    }
    line[digits..].strip_prefix(". ").or_else(|| line[digits..].strip_prefix(") "))
}

/// The U-numbers an entry starts with, and what it says about them:
/// "U12: what settled it", "**U12** — why", "U3, U4: both dropped".
fn item_references(entry: &str) -> Option<(Vec<i64>, String)> {
    let mut rest = entry.trim().trim_start_matches(['*', '[', '(', ' ']);
    let mut ids = Vec::new();
    while let Some(after) = rest.strip_prefix(['U', 'u']) {
        let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
        let Ok(id) = digits.parse::<i64>() else { break };
        ids.push(id);
        rest = after[digits.len()..].trim_start_matches(['*', ']', ')']);
        let joined = rest.trim_start();
        rest = if let Some(next) = joined.strip_prefix(',').or_else(|| joined.strip_prefix('/')) {
            let next = next.trim_start();
            next.strip_prefix("and ").map_or(next, str::trim_start)
        } else if let Some(next) = joined.strip_prefix("and ") {
            next.trim_start()
        } else {
            break;
        };
    }
    if ids.is_empty() {
        return None;
    }
    let reason = rest
        .trim_start_matches(|c: char| c == ':' || c == '-' || c == '—' || c == '–' || c.is_whitespace())
        .trim()
        .to_string();
    (!reason.is_empty()).then_some((ids, reason))
}

impl ParsedGist {
    /// The items this gist opens, given the id the store will issue next, and
    /// how many entries were dropped as restatements.
    ///
    /// An entry that leads with a U-number below `next_id` names an item that
    /// already exists: storing it again would duplicate it. One at or above
    /// `next_id` is the model numbering its own new item after the format of the
    /// list it was shown; it is kept, without the number, since the store issues
    /// ids. Writing and storing both call this at the same point, with the same
    /// `next_id`, so they agree.
    fn new_items(&self, next_id: i64) -> (Vec<String>, usize) {
        let mut items = Vec::new();
        let mut restated = 0;
        for entry in &self.opened {
            match entry.id {
                Some(id) if id < next_id => restated += 1,
                _ => items.push(entry.text.clone()),
            }
        }
        (items, restated)
    }
}

pub(crate) fn parse_gist(text: &str) -> ParsedGist {
    let mut section = Section::Narrative;
    let mut narrative = Vec::new();
    let mut entries: Vec<(Section, String)> = Vec::new();

    for line in text.lines() {
        if let Some(heading) = section_heading(line) {
            section = heading;
            continue;
        }
        if section == Section::Narrative {
            narrative.push(line);
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match strip_bullet(trimmed) {
            Some(entry) => entries.push((section, entry.trim().to_string())),
            // A wrapped line continues the entry above it.
            None => match entries.last_mut().filter(|(owner, _)| *owner == section) {
                Some((_, entry)) => {
                    entry.push(' ');
                    entry.push_str(trimmed);
                }
                None => entries.push((section, trimmed.to_string())),
            },
        }
    }

    let mut parsed = ParsedGist {
        narrative: narrative.join("\n").trim().to_string(),
        ..ParsedGist::default()
    };
    for (section, entry) in entries {
        match section {
            Section::NotSettled if !says_none(&entry) => {
                let opened = match item_references(&entry) {
                    Some((ids, text)) => Opened { id: ids.first().copied(), text },
                    None => Opened { id: None, text: entry },
                };
                parsed.opened.push(opened);
            }
            Section::Settled | Section::Moot => {
                let Some((ids, reason)) = item_references(&entry) else { continue };
                let target = if section == Section::Settled { &mut parsed.settled } else { &mut parsed.moot };
                target.extend(ids.into_iter().map(|id| (id, reason.clone())));
            }
            _ => {}
        }
    }
    parsed
}

/// Whether a SETTLED line carries evidence from the stretch that settled it: an
/// observation that `observed_here` accepts (a tool result inside that
/// stretch), or the operator's word in a stretch where the operator actually said something.
///
/// Reasoning alone can't settle anything. That is exactly the path by which a
/// hunch turns into a fact: a later stretch "concludes" it and the question
/// quietly closes. Neither can a number that merely falls in the stretch's
/// range: it has to be an observation there.
fn settled_on_evidence(reason: &str, observed_here: &dyn Fn(u64) -> bool, operator_spoke: bool) -> bool {
    (operator_spoke && said_tag(reason).is_some()) || cited_observations(reason).into_iter().any(observed_here)
}

/// The same test without the range, for use while gists are still being
/// written: whether a settlement could be accepted at all.
fn looks_evidenced(reason: &str, operator_spoke: bool) -> bool {
    (operator_spoke && said_tag(reason).is_some()) || !cited_observations(reason).is_empty()
}

/// Where a "said" tag starts in `text`: `[operator said]`, or one written
/// under an older name (`[<name> said]`), which memory kept from before.
fn said_tag(text: &str) -> Option<usize> {
    let lower = text.to_lowercase();
    lower.match_indices('[').map(|(at, _)| at).find(|&at| {
        let rest = &lower[at + 1..];
        rest.find(']').filter(|&end| end <= 40).is_some_and(|end| {
            let inside = rest[..end].trim();
            inside.ends_with(" said") && inside[..inside.len() - 5].chars().all(|c| c.is_alphabetic() || c == ' ')
        })
    })
}

/// The start of an item, for quoting it elsewhere.
fn headline(text: &str) -> String {
    let text = text.trim();
    if text.chars().count() <= 80 {
        return text.to_string();
    }
    let head: String = text.chars().take(80).collect();
    format!("{head}…")
}

// ------------------------------------------------------------------ writing gists

/// One result per planned episode, in order: the gist text, or why none came.
type Written = Vec<std::result::Result<String, String>>;

/// What counts as a written gist. Writing and storing must agree on it, or the
/// item numbers the writer handed out would drift from the ones stored.
fn is_written(text: &str) -> bool {
    !text.trim().is_empty()
}

/// How much of the unsettled list a gist writer is shown. The newest items are
/// kept when it runs over; older ones still stand, behind their pointer, and go
/// to the world model at deep sleep.
const PROMPT_ITEMS_TOKENS: usize = 4_000;

/// What a gist writer is shown as not settled: every unsettled item by its own
/// number, set-aside ones included, oldest first.
fn open_snapshot(layers: &MemoryLayers) -> Result<Vec<(i64, String)>> {
    let mut shown = Vec::new();
    let mut tokens = 0;
    for item in layers.unsettled_items()?.into_iter().rev() {
        tokens += item.tokens;
        if tokens > PROMPT_ITEMS_TOKENS && !shown.is_empty() {
            break;
        }
        shown.push((item.id, item.text));
    }
    shown.reverse();
    Ok(shown)
}

fn render_open_for_prompt(open: &[(i64, String)]) -> String {
    open.iter().map(|(id, text)| format!("- U{id}: {}\n", text.trim())).collect()
}

/// Write the gists for a plan's episodes in order, each one seeing what is not
/// settled — including what earlier episodes of this same fold left open, so a
/// question raised and answered inside one fold can be settled there instead of
/// staying open until deep sleep.
///
/// New items are numbered from `first_new_id`, the id the store will issue
/// next. `apply_written` stores them in the same order, so the numbers a later
/// episode's gist used are the numbers the items land with.
fn write_gists(
    summarizer: &dyn Summarizer,
    episodes: &[EpisodeText],
    mut open: Vec<(i64, String)>,
    first_new_id: i64,
    cancel: Option<&AtomicBool>,
) -> Written {
    let mut next_id = first_new_id;
    let mut written = Vec::with_capacity(episodes.len());
    for episode in episodes {
        if cancel.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
            written.push(Err("cancelled".to_string()));
            continue;
        }
        let result = summarizer
            .gist(&episode.transcript, &render_open_for_prompt(&open))
            .map_err(|error| error.to_string());
        if let Some(text) = result.as_ref().ok().filter(|text| is_written(text)) {
            let parsed = parse_gist(text);
            // Hidden from later episodes only if storing will accept it; a
            // settlement that will be refused leaves the item visible, so a
            // later episode holding the real evidence can still settle it.
            let closed: HashSet<i64> = parsed
                .settled
                .iter()
                .filter(|(_, reason)| looks_evidenced(reason, episode.operator_spoke))
                .chain(&parsed.moot)
                .map(|(id, _)| *id)
                .collect();
            open.retain(|(id, _)| !closed.contains(id));
            for item in parsed.new_items(next_id).0 {
                open.push((next_id, item));
                next_id += 1;
            }
        }
        written.push(result);
    }
    written
}

/// Store what was written for a plan, then drop the turns it covers.
///
/// Ranges and positions are worked out here, from the live layer as it is now,
/// not when the gists were written: for a prepared fold the context has moved
/// on since, and the anchors after the cut are what bound the last episode.
///
/// What each gist cites is checked against the archive here too, before it is
/// stored: an `[observed #N]` stands only if N is a tool result inside the
/// range the gist covers, which is what the writer was shown. The check runs
/// on the whole reply, so the account, its NOT SETTLED items (which become
/// world-model evidence at deep sleep) and its SETTLED reasons all meet the
/// same rule. If the archive can't be read, nothing can be checked, so every
/// numbered citation is taken out rather than stored unchecked; the fold still
/// goes ahead, because a fold that can't happen is the worse failure.
#[allow(clippy::too_many_arguments)]
fn apply_written(
    messages: &mut Vec<Message>,
    layers: &MemoryLayers,
    sources: &dyn Sources,
    plan: &FoldPlan,
    written: Written,
    floor: u64,
    latest: u64,
    now_distance: u64,
    written_by: &str,
    outcome: &mut FoldOutcome,
    on_status: &mut dyn FnMut(String),
) -> Result<()> {
    // The job open as this is stored, which is what the rows are tagged with.
    // Read once, so every row of one fold is tagged alike.
    //
    // An episode's range can straddle the moment a job opened, and then the
    // tag covers a little more than the job did. That is the safe direction:
    // the close deletes slightly more of its own memory than it had to, rather
    // than leaving a row that holds part of the job.
    let job = crate::jobs::gist_tag();
    let live = memory_layers::live_turns(messages);
    let ranges = memory_layers::episode_ranges(live, plan, floor, latest);
    // Read before anything is stored, so a failure here changes nothing.
    let spans = match sources.spans(&ranges) {
        Ok(spans) if spans.len() == ranges.len() => Some(spans),
        Ok(spans) => {
            on_status(format!("[fold] the archive answered for {} of {} stretches; citations can't be checked", spans.len(), ranges.len()));
            None
        }
        Err(error) => {
            on_status(format!("[fold] archive unreadable ({error}); citations can't be checked, so numbered ones are kept as [observed] only"));
            None
        }
    };
    let positions = memory_layers::episode_positions(live, plan, now_distance);
    // When each stretch happened, for its stored label.
    let times = sources.times(&memory_layers::range_ends(ranges.iter().copied()));
    let folded_anchor = memory_layers::highest_anchor(&live[..plan.messages_folded]);
    let spoke: Vec<bool> = plan
        .episodes
        .iter()
        .map(|episode| operator_spoke(&live[episode.start..episode.end]))
        .collect();

    for (index, result) in written.into_iter().enumerate() {
        let (from, through) = ranges[index];
        let position = positions[index];
        let label = memory_layers::dated(from, through, &times).unwrap_or_else(|| {
            memory_layers::distance_label(now_distance.saturating_sub(position)).to_string()
        });
        let span = spans.as_ref().map(|spans| &spans[index]);
        let source_hash = span.map(|span| span.source_hash.clone()).unwrap_or_default();
        let observed_here = |id: u64| span.is_some_and(|span| span.tool_results.contains(&id));

        let text = match result {
            Ok(text) if is_written(&text) => text,
            other => {
                let reason = match other {
                    Err(error) => error,
                    Ok(_) => "the model returned nothing".to_string(),
                };
                on_status(format!("[fold] no gist written: {reason}"));
                outcome.mechanical += 1;
                layers.append(&mechanical_gist(
                    Level::Episode,
                    from,
                    through,
                    source_hash,
                    label,
                    position,
                    &reason,
                    job.clone(),
                ))?;
                continue;
            }
        };

        let (text, removed) = keep_citations(&text, &observed_here);
        if !removed.is_empty() {
            let ids = removed.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(", ");
            on_status(if spans.is_some() {
                format!("[fold] took {ids} out of the gist for events #{from}..#{through}: not tool results in that stretch")
            } else {
                format!("[fold] took {ids} out of the gist for events #{from}..#{through}: unchecked")
            });
            outcome.uncited += removed.len();
        }
        let parsed = parse_gist(&text);
        let mut footer = Vec::new();
        // Its claims seen or heard here become candidate facts, searchable
        // at once and consolidated by deep sleep. Citations are checked by now.
        let staged = sources.stage_facts(&candidate_facts(&parsed.narrative, from, through, spoke[index]));
        outcome.staged_facts += staged;

        for (id, reason) in &parsed.settled {
            let Some(item) = unsettled_item(layers, *id)? else {
                on_status(format!("[fold] U{id} was marked settled but isn't an unsettled item; ignored"));
                outcome.refused += 1;
                continue;
            };
            if !settled_on_evidence(reason, &observed_here, spoke[index]) {
                on_status(format!(
                    "[fold] U{id} was called settled without evidence from events \
                     #{from}..#{through}; it stays open"
                ));
                outcome.refused += 1;
                continue;
            }
            layers.resolve_item(*id, ItemState::Settled, reason)?;
            footer.push(format!("{}U{id} ({}) — {reason}", memory_layers::FOOTER_PREFIXES[0], headline(&item.text)));
            outcome.settled += 1;
        }

        for (id, reason) in &parsed.moot {
            let Some(item) = unsettled_item(layers, *id)? else {
                on_status(format!("[fold] U{id} was marked moot but isn't an unsettled item; ignored"));
                outcome.refused += 1;
                continue;
            };
            layers.resolve_item(*id, ItemState::Moot, reason)?;
            on_status(format!("[fold] U{id} called moot: {}", headline(reason)));
            footer.push(format!("{}U{id} ({}) — {reason}", memory_layers::FOOTER_PREFIXES[1], headline(&item.text)));
            outcome.moot += 1;
        }

        let (new_items, restated) = parsed.new_items(layers.next_item_id()?);
        if restated > 0 {
            on_status(format!("[fold] {restated} NOT SETTLED line(s) restated items already open; not duplicated"));
        }
        let mut opened = Vec::new();
        for item in &new_items {
            let id = layers.next_item_id()?;
            layers.add_item(
                id,
                &NewOpenItem {
                    from_event_id: from,
                    through_event_id: through,
                    text: item.clone(),
                    distance: position,
                    job: job.clone(),
                },
            )?;
            opened.push(format!("U{id}"));
            outcome.opened += 1;
        }
        if !opened.is_empty() {
            footer.push(format!("{}{} (see NOT SETTLED).", memory_layers::FOOTER_PREFIXES[2], opened.join(", ")));
        }

        let mut body = if parsed.narrative.is_empty() {
            "I kept no account of this stretch beyond what it left open.".to_string()
        } else {
            parsed.narrative
        };
        if !footer.is_empty() {
            body.push_str("\n\n");
            body.push_str(&footer.join("\n"));
        }

        outcome.episodes += 1;
        layers.append(&NewGist {
            level: Level::Episode,
            from_event_id: from,
            through_event_id: through,
            source_hash,
            tokens: estimate_tokens(&body),
            time_label: label,
            written_by: written_by.to_string(),
            text: body,
            mechanical: false,
            distance: position,
            job: job.clone(),
        })?;
    }

    // After every episode, never between them: a pointer takes an id, and the
    // numbering a prepared fold used must match the order items land in.
    let collapsed = layers.enforce_item_budget()?;
    if collapsed > 0 {
        on_status(format!(
            "[fold] NOT SETTLED over budget; set the oldest items aside behind {collapsed} pointer(s)"
        ));
    }

    memory_layers::drop_folded(messages, plan.messages_folded);
    // With no anchor in what was folded, the floor this fold used is still a
    // true bound; recording it stops a later fold falling back to a gist's
    // over-broad end.
    layers.raise_fold_floor(folded_anchor.unwrap_or(floor))?;
    outcome.messages_folded += plan.messages_folded;
    Ok(())
}

/// An item that can still be settled: open or set aside, never a pointer.
fn unsettled_item(layers: &MemoryLayers, id: i64) -> Result<Option<OpenItem>> {
    Ok(layers.item(id)?.filter(|item| {
        item.kind == memory_layers::ItemKind::Item
            && matches!(item.state, ItemState::Open | ItemState::SetAside)
    }))
}

fn cascade_and_rebuild(
    messages: &mut Vec<Message>,
    layers: &MemoryLayers,
    summarizer: &dyn Summarizer,
    sources: &dyn Sources,
    schedule: &FoldSchedule,
    outcome: &mut FoldOutcome,
    on_status: &mut dyn FnMut(String),
) -> Result<()> {
    let report =
        memory_layers::enforce_budgets_within(layers, summarizer, sources, schedule.gist_budget, schedule.shape_budget)?;
    if !report.citations_removed.is_empty() {
        let ids = report.citations_removed.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(", ");
        on_status(format!("[fold] took {ids} out of merged memory: not cited by what was merged, or not a tool result in its range"));
        outcome.uncited += report.citations_removed.len();
    }
    for unread in &report.unread {
        on_status(format!("[fold] merged without a source hash; archive unreadable for {unread}"));
    }
    if report.episode_merges + report.shape_merges > 0 {
        on_status(format!(
            "[fold] merged {} episodes and {} shape entries; layers now {}+{} tokens",
            report.episode_merges,
            report.shape_merges,
            report.episode_tokens,
            report.shape_tokens,
        ));
    }
    if report.flattened_merges > 0 {
        on_status(format!(
            "[fold] {} merge(s) kept no tentative tags from their inputs; marked as unconfirmed",
            report.flattened_merges,
        ));
    }
    outcome.mechanical += report.mechanical_merges;
    outcome.flattened += report.flattened_merges;

    // No loops here: they render as an appended block at the end of the
    // context, not inside the memory message. See open_loops::refresh_context
    // for why the most volatile layer sits where a rewrite is cheapest.
    let rebuilt = memory_layers::build_memory_message(layers, sources)?;
    memory_layers::install_memory_message(messages, rebuilt);
    Ok(())
}

// ------------------------------------------------------------------ preparation

/// Harness-rendered blocks are taken out before a fold plans, so preparation
/// has to plan over the same view.
fn is_harness_block(message: &Message) -> bool {
    crate::open_loops::is_block(message) || crate::board::is_block(message)
}

/// Every block, not just the first: a block can be duplicated when it moves
/// in the same agent iteration that carries messages across a fold.
fn strip_harness_blocks(messages: &mut Vec<Message>) {
    while crate::open_loops::strip_block(messages) {}
    while crate::board::strip_block(messages) {}
}

/// The live turns as a fold will see them, or `None` if the context doesn't have
/// its memory message yet.
fn fold_view(messages: &[Message]) -> Option<Vec<Message>> {
    let laid_out = messages
        .get(memory_layers::MEMORY_INDEX)
        .is_some_and(|message| message.content.starts_with(memory_layers::MEMORY_MARKER));
    laid_out.then(|| {
        memory_layers::live_turns(messages)
            .iter()
            .filter(|message| !is_harness_block(message))
            .cloned()
            .collect()
    })
}

/// Identifies the turns a prepared fold covers, so it is applied only to the
/// turns it was written from.
///
/// Tool results are identified by their call ID alone: observation masking
/// rewrites older ones in place, which changes their text but not what they
/// were, and a gist written from the unmasked text is if anything better.
fn fingerprint(messages: &[Message]) -> Vec<u64> {
    messages
        .iter()
        .map(|message| {
            let mut hasher = DefaultHasher::new();
            message.role.hash(&mut hasher);
            message.tool_call_id.hash(&mut hasher);
            for call in message.tool_calls.iter().flatten() {
                call.id.hash(&mut hasher);
            }
            if message.role != "tool" {
                message.content.hash(&mut hasher);
            }
            hasher.finish()
        })
        .collect()
}

/// A fold written ahead of time.
///
/// Past `FOLD_PREPARE_TOKENS`, the gists for the oldest live turns are written
/// on a background thread while s0ma keeps working; the fold at
/// `FOLD_AT_TOKENS` then only stores them, merges and drops the turns, instead
/// of stopping to write ~145k tokens of gists. The prepared fold is used only if
/// the turns it covers are still exactly the turns at the front of the live
/// layer and its NOT SETTLED numbering is still the next to be issued; otherwise
/// it is discarded and the fold is written on the spot, as it would have been
/// without preparation.
///
/// The thread borrows worker lanes like any fold. Before anything that needs
/// every lane (deep sleep) or replaces the context (`/compact`), `quiesce` stops
/// it and waits for the lane it holds to come back.
#[derive(Default)]
pub struct FoldPrep {
    job: Option<PrepJob>,
    /// Jobs cancelled for being stale, still finishing their current call.
    /// `quiesce` waits for these too, so no lane is left borrowed.
    draining: Vec<std::thread::JoinHandle<Written>>,
}

struct PrepJob {
    handle: std::thread::JoinHandle<Written>,
    plan: FoldPlan,
    fingerprint: Vec<u64>,
    first_new_id: i64,
    cancel: Arc<AtomicBool>,
    /// The job open when these gists were prepared. Gists are written from the
    /// live context by a model call on another thread, and a job opening or
    /// closing in the meantime changes which of them the prose may hold: a
    /// prepared gist stored afterwards would carry the wrong tag, and prose
    /// mixing a job's work with the rest cannot be separated again. Checked
    /// alongside the other two staleness checks, and the gists are rewritten
    /// if it moved.
    open_job: Option<String>,
    /// The fold came due while these were still being written, and that was said.
    announced: bool,
    /// Set by the thread when it is done: how many of the gists, from the
    /// first, came back usable (`usable_count`).
    usable: Arc<AtomicUsize>,
}

/// How many results, from the first, are gists that can be stored: a fold uses
/// those and writes the rest itself (`take_valid`).
fn usable_count(written: &Written) -> usize {
    written.iter().take_while(|result| result.as_ref().is_ok_and(|text| is_written(text))).count()
}

impl PrepJob {
    /// Done, with every gist usable: a fold from these needs no model call
    /// to finish (a lane taken by subagents, or an empty reply, would
    /// otherwise make "done" mean writing them all now).
    fn all_written(&self) -> bool {
        self.handle.is_finished() && self.usable.load(Ordering::SeqCst) >= self.plan.episodes.len()
    }

    /// Why these gists can't make the fold: done but not all usable, or no
    /// longer for these turns. Only looked at while a job is held, so the live
    /// layer is copied only then.
    fn unusable(&self, messages: &[Message], layers: &MemoryLayers) -> Option<String> {
        if self.handle.is_finished() && !self.all_written() {
            return Some(format!(
                "only {} of the {} gists written in the background came back usable",
                self.usable.load(Ordering::SeqCst),
                self.plan.episodes.len()
            ));
        }
        let live = fold_view(messages)?;
        self.stale(&live, layers).map(str::to_string)
    }

    /// Why these gists no longer fit `live` and `layers`, if they don't.
    fn stale(&self, live: &[Message], layers: &MemoryLayers) -> Option<&'static str> {
        let covered = self.plan.messages_folded;
        if live.len() < covered || fingerprint(&live[..covered]) != self.fingerprint {
            return Some("the turns the prepared gists cover have changed");
        }
        if layers.next_item_id().map_or(true, |next| next != self.first_new_id) {
            return Some("NOT SETTLED changed since the gists were prepared");
        }
        if crate::jobs::gist_tag() != self.open_job {
            return Some("a job opened or closed since the gists were prepared");
        }
        None
    }
}

impl FoldPrep {
    #[cfg(test)]
    fn is_active(&self) -> bool {
        self.job.is_some()
    }

    /// Stop any background writing and wait for its thread, so the lane it
    /// borrowed is back before the caller goes on. Waits at most for the one
    /// model call in flight; a lane wait or the next episode sees the flag.
    pub fn quiesce(&mut self) {
        if let Some(job) = self.job.take() {
            job.cancel.store(true, Ordering::SeqCst);
            self.draining.push(job.handle);
        }
        if self.draining.iter().any(|handle| !handle.is_finished()) {
            crate::terminal::say_note("[fold] stopping background gist writing");
        }
        for handle in self.draining.drain(..) {
            let _ = handle.join();
        }
    }

    /// Cancel a job and keep its thread where `quiesce` can wait for it.
    fn abandon(&mut self, job: PrepJob) {
        job.cancel.store(true, Ordering::SeqCst);
        self.draining.retain(|handle| !handle.is_finished());
        self.draining.push(job.handle);
    }

    /// Start preparing if the context is in the prepare band and nothing is
    /// prepared yet, and say whether a prepared fold is ready to apply: written,
    /// and still for these turns. A prepared fold that no longer fits them is
    /// dropped here and written again while there is time (a long window's fold
    /// waits for it). Never fails the caller: a problem is reported, and the
    /// fold is simply written on the spot later.
    pub fn maybe_start(&mut self, messages: &[Message], usage: &ContextUsage, layers: &MemoryLayers) -> bool {
        let need = usage.fold_need();
        let schedule = usage.schedule();
        // Only a window whose fold waits for the gists (a long one) checks them
        // ahead of time; a short window's fold is due at once and sorts them
        // out then (`take_valid`), as it always has.
        // Once the fold is due it takes what there is (`take_valid` keeps the
        // usable gists before a failed one), so a job is only dropped before.
        let waits = schedule.ceiling > schedule.fold_at;
        if waits
            && need < FoldNeed::Due
            && let Some(job) = self.job.as_ref()
            && let Some(why) = job.unusable(messages, layers)
        {
            llm::background_status(format_args!("[fold] {why}; writing the gists again"));
            let job = self.job.take().expect("checked above");
            self.abandon(job);
        }
        // A fold that is due writes its own gists; preparing then would only
        // make it wait for a thread started a moment earlier.
        if self.job.is_none() && need.should_prepare() && need < FoldNeed::Due {
            let cancel = Arc::new(AtomicBool::new(false));
            let summarizer = ModelSummarizer::cancellable(Arc::clone(&cancel));
            match self.start(messages, layers, &schedule, summarizer, cancel) {
                Ok(Some(episodes)) => llm::background_status(format_args!(
                    "[fold] writing {episodes} gist(s) in the background at {} input tokens",
                    usage.input_tokens,
                )),
                Ok(None) => {}
                Err(error) => llm::background_status(format_args!("[fold] preparation skipped: {error}")),
            }
        }
        let ready = self.job.as_ref().is_some_and(PrepJob::all_written);
        if need == FoldNeed::Fold
            && !ready
            && let Some(job) = self.job.as_mut()
            && !job.announced
        {
            job.announced = true;
            llm::background_status(format_args!(
                "[fold] past {}: folding when the gists being written are ready (at {} at the latest)",
                schedule.fold_at, schedule.ceiling,
            ));
        }
        ready
    }

    fn start<S: Summarizer + Send + 'static>(
        &mut self,
        messages: &[Message],
        layers: &MemoryLayers,
        schedule: &FoldSchedule,
        summarizer: S,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<usize>> {
        let Some(live) = fold_view(messages) else {
            return Ok(None);
        };
        // What the fold at `fold_at` will have to remove to land on target (as
        // much as there is, when that is less: the turns that come while the
        // gists are written are kept).
        let plan = memory_layers::plan_fold(&live, schedule.fold_at - schedule.fold_to);
        if plan.is_empty() {
            return Ok(None);
        }
        let episodes = episode_texts(&live, &plan);
        let fingerprint = fingerprint(&live[..plan.messages_folded]);
        let open = open_snapshot(layers)?;
        let first_new_id = layers.next_item_id()?;
        let count = plan.episodes.len();

        let flag = Arc::clone(&cancel);
        let usable = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&usable);
        let handle = std::thread::Builder::new().name("fold-prep".to_string()).spawn(move || {
            let written = write_gists(&summarizer, &episodes, open, first_new_id, Some(&flag));
            counted.store(usable_count(&written), Ordering::SeqCst);
            written
        })?;

        self.job = Some(PrepJob {
            handle,
            plan,
            fingerprint,
            first_new_id,
            cancel,
            open_job: crate::jobs::gist_tag(),
            announced: false,
            usable,
        });
        Ok(Some(count))
    }

    /// Hand over the prepared fold if it still fits `live`, waiting for it to
    /// finish if it hasn't.
    ///
    /// Whether it still fits is checked first, so a stale job is cancelled
    /// rather than waited for. If a later episode failed in the background (a
    /// lane didn't come free in time, say), the episodes before it are still
    /// used and the rest is left for the fold to write now, when it is blocking
    /// anyway and more likely to get a lane.
    fn take_valid(
        &mut self,
        live: &[Message],
        layers: &MemoryLayers,
        on_status: &mut dyn FnMut(String),
    ) -> Result<Option<(FoldPlan, Written)>> {
        let Some(job) = self.job.take() else {
            return Ok(None);
        };
        if let Some(why) = job.stale(live, layers) {
            self.abandon(job);
            on_status(format!("[fold] {why}; writing them now"));
            return Ok(None);
        }
        if !job.handle.is_finished() {
            on_status("[fold] waiting for the gists being written in the background".to_string());
        }
        let Ok(mut written) = job.handle.join() else {
            on_status("[fold] background gist writing failed; writing them now".to_string());
            return Ok(None);
        };

        let usable = usable_count(&written);
        let mut plan = job.plan;
        if usable < written.len() {
            let reason = match &written[usable] {
                Err(error) => error.clone(),
                Ok(_) => "the model returned nothing".to_string(),
            };
            if usable == 0 {
                on_status(format!("[fold] background gists unusable ({reason}); writing them now"));
                return Ok(None);
            }
            on_status(format!(
                "[fold] background gist {} of {} failed ({reason}); using the {usable} before it",
                usable + 1,
                written.len(),
            ));
            plan.episodes.truncate(usable);
            plan.messages_folded = plan.episodes.last().map_or(0, |episode| episode.end);
            plan.est_tokens_removed = plan.episodes.iter().map(|episode| episode.est_tokens).sum();
            written.truncate(usable);
        }
        Ok(Some((plan, written)))
    }
}

// ------------------------------------------------------------------ the fold

/// Warn if the prefix and memory message have grown into the space a fold needs.
///
/// A fold measures the *whole* context against `FOLD_TO_TOKENS`, so these two
/// are a floor it can never get under. If the floor grows past the target, the
/// fold loop drops live turns, measures, misses, and drops more — tightening
/// forever without ever landing. That failure looks like a hang, not a large
/// context, which is why it is worth a round trip per fold to see coming.
///
/// A warning rather than an error: neither the prefix nor the memory message is
/// something a fold can shrink, so there is nothing to do about it here except
/// say so loudly enough that the budgets get re-measured.
fn check_floor(
    messages: &[Message],
    options: &ChatOptions,
    tools: &[ToolDefinition],
    schedule: &FoldSchedule,
    on_status: &mut dyn FnMut(String),
) -> Result<()> {
    // `ensure_layout` leaves an empty list empty, so there may be no floor yet.
    if messages.is_empty() {
        return Ok(());
    }
    let end = memory_layers::MEMORY_INDEX.min(messages.len() - 1);
    let floor = context_guard::measure(&messages[..=end], options, tools)?.input_tokens;
    let layers_budget = schedule.memory_message_budget();
    let budget = PREFIX_BUDGET_TOKENS + layers_budget;
    let target = schedule.fold_to;

    if floor + MIN_LIVE_TURNS_TOKENS > target {
        on_status(format!(
            "[fold] WARNING: prefix plus memory message is {floor} tokens, leaving under \
             {MIN_LIVE_TURNS_TOKENS} for live turns at the {target} target. Folding \
             cannot get under this floor — raise the fold target or shrink the prefix.",
        ));
    } else if floor > budget {
        on_status(format!(
            "[fold] prefix plus memory message is {floor} tokens, over the {budget} budget \
             (prefix {PREFIX_BUDGET_TOKENS} + layers {layers_budget}); \
             still folding, but re-measure the prefix.",
        ));
    }
    Ok(())
}

#[derive(Debug, Default)]
pub struct FoldOutcome {
    pub episodes: usize,
    pub mechanical: usize,
    pub messages_folded: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
    /// Whether gists written in the background were used.
    pub prepared: bool,
    /// NOT SETTLED items this fold added, settled, and called moot.
    pub opened: usize,
    pub settled: usize,
    pub moot: usize,
    /// SETTLED or MOOT lines refused: no evidence from the stretch, or naming
    /// nothing open.
    pub refused: usize,
    /// Merges that dropped every tentative tag their inputs carried.
    pub flattened: usize,
    /// Event ids taken out of `[observed #N]` tags: ones that weren't tool
    /// results in the stretch the gist covers (or, in a merge, that its inputs
    /// never cited), and every one when the archive couldn't be read.
    pub uncited: usize,
    /// Candidate facts this fold's gists staged for deep sleep (`facts.rs`).
    pub staged_facts: usize,
}

/// Fold the oldest live turns down until the whole context fits the window's
/// fold target (`FoldSchedule::fold_to`). A fold made from gists written in the
/// background stops at `land_within` instead: the turns since they were
/// planned stay verbatim rather than being written now, mid-turn.
///
/// `rebuilt_through` is the checkpoint the running context was rebuilt from: no
/// live turn is older than it, so it bounds the first episode's archive range
/// from below. The distance odometer in `loops` must already be current (the
/// caller advances it to the context's present size), because each episode's
/// position is counted back from it.
///
/// Returns `None` when nothing could usefully be folded — which is a real
/// outcome, not a failure: it means the live layer has no legal cut big enough to
/// be worth its re-prefill, and the caller has to decide what to do instead.
#[allow(clippy::too_many_arguments)]
pub fn fold_context(
    messages: &mut Vec<Message>,
    layers: &MemoryLayers,
    loops: &LoopStore,
    memory: &Hippocampus,
    summarizer: &dyn Summarizer,
    options: &ChatOptions,
    tools: &[ToolDefinition],
    rebuilt_through: u64,
    prep: &mut FoldPrep,
    on_status: &mut dyn FnMut(String),
) -> Result<Option<FoldOutcome>> {
    // The loops and board blocks are harness state, not conversation. Take them
    // out before planning so they can never be folded into a gist, and let the
    // caller's refresh put them back at the end afterwards.
    strip_harness_blocks(messages);

    memory_layers::ensure_layout(messages);
    let before = context_guard::measure(messages, options, tools)?;
    let tokens_before = before.input_tokens;
    let schedule = before.schedule();
    let now_distance = loops.distance()?;
    let written_by = summarizer.model_name();
    let mut outcome = FoldOutcome { tokens_before, tokens_after: tokens_before, ..FoldOutcome::default() };

    // Memory kept on a longer window than this one (the same data run against
    // a smaller one) is merged down to this window's budgets first, so the
    // floor is measured, and the cut planned, against what the fold will
    // really leave.
    if layers.tokens_at(Level::Episode)? > schedule.gist_budget || layers.tokens_at(Level::Shape)? > schedule.shape_budget {
        cascade_and_rebuild(messages, layers, summarizer, memory, &schedule, &mut outcome, on_status)?;
    }

    check_floor(messages, options, tools, &schedule, on_status)?;

    // A fold written ahead of time, if there is one and it still fits.
    let prepared = prep.take_valid(memory_layers::live_turns(messages), layers, on_status)?;
    if let Some((plan, written)) = prepared {
        on_status(format!(
            "[fold] using {} gist(s) written in the background over {} messages",
            plan.episodes.len(),
            plan.messages_folded,
        ));
        let floor = layers.fold_floor()?.max(rebuilt_through);
        apply_written(
            messages,
            layers,
            memory,
            &plan,
            written,
            floor,
            memory.last_event_id(),
            now_distance,
            &written_by,
            &mut outcome,
            on_status,
        )?;
        cascade_and_rebuild(messages, layers, summarizer, memory, &schedule, &mut outcome, on_status)?;
        outcome.prepared = true;
    }

    let land_within = schedule.lands_within(outcome.prepared);
    for attempt in 1..=MAX_FOLD_ATTEMPTS {
        let current = context_guard::measure(messages, options, tools)?.input_tokens;
        if current <= land_within {
            break;
        }

        // Ask for more than the shortfall on later attempts: landing just under
        // target and immediately needing another fold is the expensive schedule.
        // After a prepared fold only down to where it may land: this is written
        // mid-turn.
        let shortfall = current - land_within;
        let target = shortfall.saturating_mul(attempt);

        let (plan, episodes) = {
            let live = memory_layers::live_turns(messages);
            let plan = memory_layers::plan_fold(live, target);
            if plan.is_empty() {
                on_status(format!(
                    "[fold] attempt {attempt}: no cut worth making in {} live messages; \
                     context stays at {current}",
                    live.len(),
                ));
                break;
            }
            let episodes = episode_texts(live, &plan);
            (plan, episodes)
        };

        on_status(format!(
            "[fold] attempt {attempt}: {} episodes over {} messages, ~{} tokens",
            plan.episodes.len(),
            plan.messages_folded,
            plan.est_tokens_removed,
        ));

        let written = write_gists(
            summarizer,
            &episodes,
            open_snapshot(layers)?,
            layers.next_item_id()?,
            None,
        );
        let floor = layers.fold_floor()?.max(rebuilt_through);
        apply_written(
            messages,
            layers,
            memory,
            &plan,
            written,
            floor,
            memory.last_event_id(),
            now_distance,
            &written_by,
            &mut outcome,
            on_status,
        )?;
        cascade_and_rebuild(messages, layers, summarizer, memory, &schedule, &mut outcome, on_status)?;
    }

    outcome.tokens_after = context_guard::measure(messages, options, tools)?.input_tokens;

    if outcome.messages_folded == 0 {
        return Ok(None);
    }

    if outcome.opened + outcome.settled + outcome.moot + outcome.refused > 0 {
        on_status(format!(
            "[fold] not settled: {} opened, {} settled, {} moot, {} refused for want of evidence",
            outcome.opened, outcome.settled, outcome.moot, outcome.refused,
        ));
    }

    if outcome.tokens_after > land_within {
        on_status(format!(
            "[fold] landed at {} tokens, above the {land_within} target; \
             the floor is the prefix plus the memory message",
            outcome.tokens_after,
        ));
    }

    Ok(Some(outcome))
}

// ------------------------------------------------------------------ deep sleep handoff

/// The world model that unsettled items move to when deep sleep takes over the
/// stretch they came from.
pub const UNSETTLED_MODEL: &str = "unsettled";

#[derive(Debug, Default)]
pub struct Handoff {
    pub written: usize,
    /// Records that already existed from an earlier, interrupted sleep.
    pub already: usize,
    /// Records written or revised as `retired`: moot items, and items settled
    /// after an earlier sleep handed them over.
    pub retired: usize,
    /// Unsettled items that couldn't be written; they stay open.
    pub failed: Vec<String>,
    /// Closed items whose record couldn't be written or retired; they stay in
    /// working memory, closed, until a later handoff manages it.
    pub failed_closed: Vec<String>,
}

/// How an item is to be written into the world model.
struct UnsettledRecord<'a> {
    status: &'a str,
    /// For a retired record, what closed it.
    closing: Option<&'a str>,
    still_uncertain: &'a str,
}

/// Before deep sleep covers a window, move what is still not settled from that
/// window into the world model as `unresolved` claims. The integration pass can
/// then see them (and settle them from the archive), and they outlive the gists
/// `clear_through` is about to remove.
///
/// Two kinds of closed item go too, so nothing leaves working memory unseen:
/// an item called moot becomes a `retired` record carrying the reason (moot
/// needs no evidence, so it shouldn't vanish silently), and an item settled
/// after an earlier sleep had already handed it over has that record retired
/// with the settlement, rather than left saying `unresolved`. A record deep
/// sleep has already ruled on (`supported`, `contradicted`) is never
/// overwritten: its evidence outranks working memory's.
///
/// Every item dealt with is marked, and `clear_through` removes only marked
/// items, so anything whose record couldn't be written, open or closed, stays
/// in working memory. Safe to repeat: a retried or failed sleep finds the
/// records already there.
pub fn hand_off_unsettled(layers: &MemoryLayers, memory: &Hippocampus, through: u64) -> Result<Handoff> {
    let unsettled = layers.unsettled_through(through)?;
    let closed = layers.closed_through(through)?;
    let mut report = Handoff::default();
    if unsettled.is_empty() && closed.is_empty() {
        return Ok(report);
    }

    let path = crate::world_model::path_for(memory);
    if !crate::world_model::model_exists(&path, UNSETTLED_MODEL)? {
        crate::world_model::define(
            &path,
            UNSETTLED_MODEL,
            "Unsettled from working memory",
            "Open questions, untested conclusions and unchecked reports that folding carried out \
             of working memory before anything settled them. Each is unresolved until evidence \
             settles it; none of them is established. Retired records were closed in working \
             memory; their statement says how.",
            0,
        )?;
    }

    for item in unsettled {
        let record_id = format!("{UNSETTLED_MODEL}:U{}", item.id);
        if crate::world_model::record_exists(&path, &record_id)? {
            layers.mark_handed_off(item.id, &record_id)?;
            report.already += 1;
            continue;
        }
        let entry = UnsettledRecord {
            status: "unresolved",
            closing: None,
            still_uncertain: "Nothing had settled this when it left working memory.",
        };
        match write_unsettled_record(&path, memory, &record_id, &item, &entry, 0) {
            Ok(()) => {
                layers.mark_handed_off(item.id, &record_id)?;
                report.written += 1;
            }
            Err(error) => report.failed.push(format!("U{}: {error}", item.id)),
        }
    }

    for item in closed {
        let record_id = format!("{UNSETTLED_MODEL}:U{}", item.id);
        let existing = crate::world_model::record_state(&path, &record_id)?;
        match (&existing, item.state) {
            // Already ruled on, or already retired: leave the record as it is.
            (Some((_, status)), _) if status != "unresolved" => {
                layers.mark_handed_off(item.id, &record_id)?;
                continue;
            }
            // Settled on evidence and recorded in its gist; no record to correct.
            (None, ItemState::Settled) => {
                layers.mark_handed_off(item.id, NOTHING_HANDED_OFF)?;
                continue;
            }
            _ => {}
        }
        let (closing, still_uncertain) = match item.state {
            ItemState::Moot => (
                format!("Called moot in working memory: {}", item.resolution),
                "Closed as no longer relevant, without evidence either way.",
            ),
            _ => (
                format!("Settled in working memory after it was handed over: {}", item.resolution),
                "Settled in working memory; see the statement for what settled it.",
            ),
        };
        let entry = UnsettledRecord { status: "retired", closing: Some(&closing), still_uncertain };
        let expected = existing.map_or(0, |(revision, _)| revision);
        match write_unsettled_record(&path, memory, &record_id, &item, &entry, expected) {
            Ok(()) => {
                layers.mark_handed_off(item.id, &record_id)?;
                report.retired += 1;
            }
            Err(error) => report.failed_closed.push(format!("U{}: {error}", item.id)),
        }
    }
    Ok(report)
}

/// What a settled item with no world-model record is marked with: there was
/// nothing to hand over, and `clear_through` may remove it.
const NOTHING_HANDED_OFF: &str = "settled in working memory; nothing to hand off";

/// The candidate facts in a gist's account: each sentence that carries an
/// `[observed #N]` (with the events it cites) or, when the operator spoke in
/// the stretch, an `[operator said]`, with its tags taken out. Inferences
/// and guesses are not facts; they stay in the gist.
fn candidate_facts(narrative: &str, from: u64, through: u64, operator_spoke: bool) -> Vec<crate::facts::Candidate> {
    let mut out = Vec::new();
    for sentence in sentences(narrative) {
        let (basis, evidence) = basis_of(sentence);
        let heard = basis == "reported" && operator_spoke;
        if !(basis == "observed" || heard) {
            continue;
        }
        let statement = without_tags(sentence);
        if statement.chars().count() < 12 || statement.len() > 1_200 {
            continue;
        }
        out.push(crate::facts::Candidate { statement, basis, evidence, from_event: from, through_event: through });
    }
    out
}

/// `text` cut into sentences: at a line break, or at `.`, `;`, `!` or `?`
/// followed by a space or the end (so `10.0.0.1` and `v1.2` stay whole).
fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for (at, c) in text.char_indices() {
        let ends = c == '\n' || (matches!(c, '.' | ';' | '!' | '?') && bytes.get(at + 1).is_none_or(|next| next.is_ascii_whitespace()));
        if ends {
            let piece = text[start..=at].trim();
            if !piece.is_empty() {
                out.push(piece);
            }
            start = at + c.len_utf8();
        }
    }
    let rest = text[start..].trim();
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

/// A claim without its knowledge tags (`[observed #12]`, `[operator said]`,
/// `[inferred]`, `[guess]`), spaces tidied, ending with a full stop.
fn without_tags(sentence: &str) -> String {
    let mut out = String::new();
    let mut rest = sentence;
    while let Some(open) = rest.find('[') {
        let close = rest[open..].find(']').map(|c| open + c);
        let inside = close.map(|c| rest[open + 1..c].to_lowercase()).unwrap_or_default();
        let is_tag = close.is_some()
            && (inside.starts_with("observed") || inside.starts_with("inferred") || inside.starts_with("guess") || inside.trim_end().ends_with(" said"));
        if is_tag {
            out.push_str(&rest[..open]);
            rest = &rest[close.unwrap() + 1..];
        } else {
            out.push_str(&rest[..=open]);
            rest = &rest[open + 1..];
        }
    }
    out.push_str(rest);
    let mut tidy = out.split_whitespace().collect::<Vec<_>>().join(" ");
    for (from, to) in [(" .", "."), (" ,", ","), (" ;", ";")] {
        tidy = tidy.replace(from, to);
    }
    let tidy = tidy.trim_end_matches([';', ',', ' ']).to_string();
    if tidy.ends_with(['.', '!', '?']) { tidy } else { format!("{tidy}.") }
}

/// The world model's basis for an item, from the first tag in its text.
fn basis_of(text: &str) -> (&'static str, Vec<u64>) {
    let lower = text.to_lowercase();
    let first = [("[observed", "observed"), ("[inferred", "inferred"), ("[guess", "interpretation")]
        .into_iter()
        .filter_map(|(tag, basis)| lower.find(tag).map(|at| (at, basis)))
        .chain(said_tag(text).map(|at| (at, "reported")))
        .min_by_key(|(at, _)| *at)
    .map_or("inferred", |(_, basis)| basis);

    match first {
        "observed" => {
            let cited = cited_observations(text);
            if cited.is_empty() { ("inferred", Vec::new()) } else { ("observed", cited) }
        }
        other => (other, Vec::new()),
    }
}

/// Cut text to at most `bytes` bytes on a character boundary.
fn clip_bytes(text: &str, bytes: usize) -> String {
    if text.len() <= bytes {
        return text.to_string();
    }
    let mut end = bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

fn write_unsettled_record(
    path: &std::path::Path,
    memory: &Hippocampus,
    record_id: &str,
    item: &OpenItem,
    entry: &UnsettledRecord<'_>,
    expected_revision: u64,
) -> Result<()> {
    let mut statement = format!(
        "{}\n\nCarried out of working memory unsettled. It came from archive events #{}..#{}.",
        clip_bytes(item.text.trim(), 9_000),
        item.from_event_id,
        item.through_event_id,
    );
    if let Some(closing) = entry.closing {
        statement.push_str("\n\n");
        statement.push_str(&clip_bytes(closing, 2_000));
    }
    let record = |basis: &str, evidence: Vec<u64>| crate::world_model::Record {
        id: record_id.to_string(),
        model_id: UNSETTLED_MODEL.to_string(),
        kind: "claim".to_string(),
        title: clip_bytes(item.text.trim(), 200),
        statement: statement.clone(),
        subject_id: None,
        target_id: None,
        basis: basis.to_string(),
        status: entry.status.to_string(),
        applies_under: Vec::new(),
        alternatives: Vec::new(),
        predictions: Vec::new(),
        supporting_evidence_ids: evidence,
        counterevidence_ids: Vec::new(),
        dependencies: BTreeMap::new(),
        observed_at: None,
        still_uncertain: entry.still_uncertain.to_string(),
        ..Default::default()
    };
    let reason = "Handed over from working memory before deep sleep covered the stretch it came from.";

    let (basis, evidence) = basis_of(&item.text);
    match crate::world_model::put(path, record(basis, evidence), expected_revision, reason, memory) {
        Ok(_) => Ok(()),
        // A cited event that doesn't resolve shouldn't cost the item: keep it as
        // an inference, which claims less.
        Err(_) if basis == "observed" => crate::world_model::put(
            path,
            record("inferred", Vec::new()),
            expected_revision,
            reason,
            memory,
        )
        .map(|_| ()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionCall, ToolCall};

    #[test]
    fn a_gist_s_seen_and_heard_claims_become_candidate_facts_and_its_guesses_do_not() {
        let narrative = "I checked the Legion's address: it answers at 192.168.10.106 [observed #48390]. \
                         The operator moved it to the hallway rack [operator said]; it has a 4 TB disk [observed #48391, #48392].\n\
                         The slowness is probably the disk [inferred]. Maybe the fan [guess].";
        let facts = candidate_facts(narrative, 48380, 48400, true);
        let statements: Vec<&str> = facts.iter().map(|f| f.statement.as_str()).collect();
        assert_eq!(
            statements,
            [
                "I checked the Legion's address: it answers at 192.168.10.106.",
                "The operator moved it to the hallway rack.",
                "it has a 4 TB disk.",
            ]
        );
        assert_eq!((facts[0].basis, facts[0].evidence.as_slice()), ("observed", &[48390][..]));
        assert_eq!((facts[1].basis, facts[1].evidence.len()), ("reported", 0));
        assert_eq!(facts[2].evidence, [48391, 48392]);
        assert!(facts.iter().all(|f| (f.from_event, f.through_event) == (48380, 48400)));
        // The operator said nothing in the stretch: a "said" tag is not his.
        let facts = candidate_facts(narrative, 48380, 48400, false);
        assert!(facts.iter().all(|f| f.basis == "observed"));
        assert_eq!(sentences("v1.2 is out. Next"), ["v1.2 is out.", "Next"]);
    }
    use std::sync::Mutex;

    fn layers() -> MemoryLayers {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "s0ma-memory-fold-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst),
        ));
        std::fs::create_dir_all(&root).expect("test directory should be created");
        MemoryLayers::beside_archive(&root.join("hippocampus.jsonl"))
    }

    fn trailer(event: u64) -> String {
        format!("\n[completed Mon 5 Oct 2026 18:42 local, archive_event_id {event}]")
    }

    fn call(id: &str) -> Message {
        Message::assistant_with_tools(
            String::new(),
            vec![ToolCall {
                id: id.to_string(),
                kind: "function".to_string(),
                function: FunctionCall { name: "vm_exec".to_string(), arguments: "{}".to_string() },
            }],
        )
    }

    fn tool(id: &str, body: &str, event: u64) -> Message {
        let mut message = Message::tool(id.to_string(), body.to_string());
        message.content.push_str(&trailer(event));
        message
    }

    /// Answers each gist call with the next scripted reply and records the open
    /// list it was shown.
    struct Scripted {
        replies: Mutex<Vec<String>>,
        shown: Mutex<Vec<String>>,
    }

    impl Scripted {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: Mutex::new(replies.iter().rev().map(|reply| reply.to_string()).collect()),
                shown: Mutex::new(Vec::new()),
            }
        }
    }

    impl Summarizer for Scripted {
        fn gist(&self, _transcript: &str, open_items: &str) -> Result<String> {
            self.shown.lock().unwrap().push(open_items.to_string());
            self.replies.lock().unwrap().pop().ok_or_else(|| "no more replies".into())
        }
        fn merge(&self, _level: Level, inputs: &[Gist]) -> Result<String> {
            Ok(inputs.iter().map(|gist| gist.text.clone()).collect::<Vec<_>>().join(" "))
        }
        fn model_name(&self) -> String {
            "scripted".to_string()
        }
    }

    // ---- transcript ----

    #[test]
    fn a_long_tool_result_keeps_its_event_id_in_the_transcript() {
        let long = format!("{}THE END", "x".repeat(10_000));
        let message = tool("a", &long, 4812);
        let rendered = render_message(&message);
        assert!(rendered.contains("archive_event_id 4812]"), "the trailer was cut: {rendered}");
        assert!(rendered.contains("THE END"), "the tail of the body survives too");
        assert!(rendered.contains("characters omitted from the middle"));
        assert!(rendered.chars().count() < 2_400, "still bounded: {}", rendered.chars().count());
    }

    #[test]
    fn a_gist_sees_the_reasoning_behind_each_step_bounded() {
        let mut step = Message::assistant_with_tools(
            "Checking the disk.".into(),
            vec![crate::types::ToolCall {
                id: "a".into(),
                kind: "function".into(),
                function: crate::types::FunctionCall { name: "vm_exec".into(), arguments: "{\"command\":\"df\"}".into() },
            }],
        );
        step = step.with_reasoning(format!("PLAN: df first, then du on /var. {}SO: run df.", "x".repeat(10_000)));
        let rendered = render_message(&step);
        assert!(rendered.starts_with("me (thinking): PLAN: df first"), "{rendered}");
        // Its conclusion survives the cut, and the call and the words follow it.
        let (thinking, rest) = rendered.split_once("SO: run df.\n").expect("the conclusion survives too");
        assert!(thinking.contains("characters omitted"));
        assert!(thinking.chars().count() < TRANSCRIPT_REASONING_CHARS + 120, "bounded");
        assert_eq!(rest, "me (tool call): vm_exec({\"command\":\"df\"})\nme: Checking the disk.\n");
        // Nobody else's text is ever labelled as thinking.
        assert!(!render_message(&Message::user("hi".into())).contains("thinking"));
    }

    #[test]
    fn a_short_tool_result_is_left_alone() {
        let rendered = render_message(&tool("a", "ok", 7));
        assert!(rendered.starts_with("tool result: ok\n[completed"), "{rendered}");
    }

    // ---- parsing ----

    #[test]
    fn a_gist_splits_into_its_account_and_sections() {
        let text = "I traced the stall to the lease wait [observed #4812].\n\
                    The operator thinks it's the pool [operator said].\n\n\
                    **NOT SETTLED:**\n\
                    - whether the pool holds the lanes [operator said]; the reserve logs\n  \
                      would show it\n\
                    - the retry fix [inferred, untested]\n\n\
                    SETTLED\n\
                    - U3: confirmed by the rerun [observed #4820]\n\n\
                    ## MOOT\n\
                    - U7 — we dropped that plan";
        let parsed = parse_gist(text);
        assert_eq!(
            parsed.narrative,
            "I traced the stall to the lease wait [observed #4812].\nThe operator thinks it's the pool [operator said]."
        );
        assert_eq!(parsed.opened.len(), 2);
        assert!(parsed.opened[0].text.ends_with("the reserve logs would show it"), "{:?}", parsed.opened);
        assert_eq!(parsed.settled, vec![(3, "confirmed by the rerun [observed #4820]".to_string())]);
        assert_eq!(parsed.moot, vec![(7, "we dropped that plan".to_string())]);
    }

    #[test]
    fn an_empty_section_and_a_plain_account_parse_cleanly() {
        let parsed = parse_gist("Just an account.\n\nNOT SETTLED\n- none");
        assert_eq!(parsed.narrative, "Just an account.");
        assert!(parsed.opened.is_empty());

        let plain = parse_gist("Nothing but prose, with the word settled in it.");
        assert_eq!(plain.narrative, "Nothing but prose, with the word settled in it.");
        assert_eq!(plain, ParsedGist { narrative: plain.narrative.clone(), ..ParsedGist::default() });
    }

    #[test]
    fn settling_needs_evidence_from_the_stretch_itself() {
        // The stretch's tool results are #50 and #55; #45 is in its range but
        // is the operator's message, and #12 is an older result.
        let here = |id: u64| [50, 55].contains(&id);
        assert!(settled_on_evidence("the rerun passed [observed #50]", &here, false));
        assert!(settled_on_evidence("both runs [observed #12, #55]", &here, false));
        assert!(settled_on_evidence("he confirmed it [operator said]", &here, true));
        assert!(!settled_on_evidence("he confirmed it [operator said]", &here, false), "The operator said nothing here");
        assert!(!settled_on_evidence("an old result [observed #12]", &here, true), "outside the stretch");
        assert!(!settled_on_evidence("in range, but no observation [observed #45]", &here, true), "not a tool result");
        assert!(!settled_on_evidence("I worked it out [inferred]", &here, true), "reasoning isn't evidence");
        assert!(!settled_on_evidence("it's fine now", &here, true));
    }

    #[test]
    fn the_parser_copes_with_how_models_actually_format_lists() {
        let parsed = parse_gist(
            "Settled on the second approach after the rerun [observed #9].\n\n\
             Not settled (open questions):\n\
             1. whether the lock matters [inferred]\n\
             2) the cache [guess]\n\
             - U3: still unclear whether the pool holds lanes [inferred]\n\
             - None — the rest was resolved.\n\
             - Nothing confirms the retry fix works [inferred, untested]\n\n\
             SETTLED\n\
             - **U4**: the rerun passed [observed #9]\n\
             - U5, U6: both covered by the same run [observed #9]\n\n\
             MOOT\n\
             - none",
        );
        assert!(parsed.narrative.starts_with("Settled on the second approach"), "a sentence, not a heading");
        assert_eq!(
            parsed.new_items(10),
            (
                vec![
                    "whether the lock matters [inferred]".to_string(),
                    "the cache [guess]".to_string(),
                    "Nothing confirms the retry fix works [inferred, untested]".to_string(),
                ],
                1,
            ),
            "numbered items count, a restated U3 and a 'None —' line don't, a real 'Nothing…' item stays",
        );
        let settled: Vec<i64> = parsed.settled.iter().map(|(id, _)| *id).collect();
        assert_eq!(settled, vec![4, 5, 6]);
        assert!(parsed.moot.is_empty());
    }

    #[test]
    fn a_sentence_that_starts_not_settled_stays_in_the_account() {
        let parsed = parse_gist(
            "I ran the test [observed #9].\nNot settled whether it was the lock.\n\
             Then I fixed the config [observed #12].",
        );
        assert!(parsed.opened.is_empty(), "{:?}", parsed.opened);
        assert!(parsed.narrative.contains("Then I fixed the config"));
        assert!(parsed.narrative.contains("Not settled whether it was the lock."));
    }

    #[test]
    fn items_the_model_numbers_itself_are_kept_and_restatements_dropped() {
        let parsed = parse_gist(
            "Account.\n\nNOT SETTLED\n- U12: whether the pool holds lanes [inferred]\n\
             - U13: whether the cache is warm [guess]\n- U14, U15: two at once [guess]",
        );
        // U12 exists (next id is 13); U13 and U14 are the model's own new items.
        let (items, restated) = parsed.new_items(13);
        assert_eq!(restated, 1);
        assert_eq!(items, vec!["whether the cache is warm [guess]".to_string(), "two at once [guess]".to_string()]);
    }

    #[test]
    fn empty_phrasings_are_recognised_and_real_items_kept() {
        for empty in [
            "None.", "none", "(none)", "N/A", "Nothing.", "None — the rest was resolved.",
            "No open questions remain.", "Nothing remains open.", "Nothing else is unsettled.",
            "None that I know of.", "No unresolved items.",
        ] {
            assert!(says_none(empty), "{empty:?} should read as empty");
        }
        for item in [
            "Nothing confirms the retry fix works [inferred, untested]",
            "Nothing confirms the retry fix works",
            "No one has checked whether the cache is warm",
            "Nothing new in the logs explains the crash",
            "None of the three runs reproduced it [observed #4]",
            "nonexistent cache entries [guess]",
        ] {
            assert!(!says_none(item), "{item:?} is an item");
        }
    }

    #[test]
    fn a_list_of_ids_with_a_final_and_is_read_whole() {
        let parsed = parse_gist("x\n\nMOOT\n- U3, U4, and U5: the plan was dropped");
        let ids: Vec<i64> = parsed.moot.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![3, 4, 5]);
        assert!(parsed.moot.iter().all(|(_, reason)| reason == "the plan was dropped"));
    }

    #[test]
    fn runtime_messages_are_not_attributed_to_operator() {
        let notice = Message::user("[SUBAGENT CONTEXT EXHAUSTED — TOOL OBSERVATION, NOT OPERATOR INPUT]".into());
        assert!(render_message(&notice).starts_with("runtime: "));
        let operator = Message::user("please check the pool".into()).by_operator();
        assert!(render_message(&operator).starts_with("The operator: "));
    }

    // ---- writing and storing ----

    fn conversation() -> Vec<Message> {
        // prefix, memory message, then two episodes' worth of turns and a tail.
        let mut messages = vec![Message::system("prefix".into())];
        memory_layers::ensure_layout(&mut messages);
        messages.extend([
            Message::user("look at the stall".into()),
            call("a"),
            tool("a", "lease wait 120s", 10),
            Message::assistant("probably the pool".into()),
            Message::user("test it".into()),
            call("b"),
            tool("b", "reserve log: sleep holds slot 3", 20),
            Message::assistant("it's sleep, not the pool".into()),
            Message::user("ok, next thing".into()),
            call("c"),
            tool("c", "later", 30),
        ]);
        messages
    }

    fn plan_of_two() -> FoldPlan {
        FoldPlan {
            episodes: vec![
                memory_layers::Episode { start: 0, end: 4, est_tokens: 1 },
                memory_layers::Episode { start: 4, end: 8, est_tokens: 1 },
            ],
            messages_folded: 8,
            est_tokens_removed: 2,
        }
    }

    #[test]
    fn a_later_episode_can_settle_what_an_earlier_one_left_open() {
        let layers = layers();
        let first = layers.next_item_id().unwrap();
        let scripted = Scripted::new(&[
            "I saw a 120s lease wait [observed #10]. I think it's the pool [inferred].\n\n\
             NOT SETTLED\n- whether the pool holds the lanes [inferred, untested]",
            &format!(
                "I checked the reserve log [observed #20].\n\nSETTLED\n- U{first}: sleep holds \
                 slot 3, not the pool [observed #20]"
            ),
        ]);

        let mut messages = conversation();
        let plan = plan_of_two();
        let transcripts = episode_texts(memory_layers::live_turns(&messages), &plan);
        let written = write_gists(&scripted, &transcripts, Vec::new(), first, None);

        let shown = scripted.shown.lock().unwrap().clone();
        assert!(shown[0].is_empty());
        assert!(shown[1].contains(&format!("U{first}: whether the pool")), "{shown:?}");

        let mut outcome = FoldOutcome::default();
        let mut statuses = Vec::new();
        apply_written(
            &mut messages, &layers, &memory_layers::FakeArchive(vec![10, 20, 30]), &plan, written, 5, 99, 500_000, "scripted", &mut outcome,
            &mut |status| statuses.push(status),
        )
        .unwrap();

        assert_eq!((outcome.opened, outcome.settled, outcome.refused), (1, 1, 0), "{statuses:?}");
        let item = layers.item(first).unwrap().unwrap();
        assert_eq!(item.state, ItemState::Settled);
        assert!(layers.open_items().unwrap().is_empty());

        let episodes = layers.level(Level::Episode).unwrap();
        assert_eq!(episodes.len(), 2);
        assert_eq!((episodes[0].from_event_id, episodes[0].through_event_id), (6, 19));
        assert_eq!((episodes[1].from_event_id, episodes[1].through_event_id), (11, 29));
        assert!(episodes[0].text.contains(&format!("Left open here: U{first}")));
        assert!(episodes[1].text.contains(&format!("Settled here: U{first}")));

        // The folded turns are gone, the tail is untouched, and the floor moved.
        let live = memory_layers::live_turns(&messages);
        assert_eq!(live.len(), 3);
        assert_eq!(live[0].content, "ok, next thing");
        assert_eq!(layers.fold_floor().unwrap(), 20);
    }

    #[test]
    fn a_settlement_without_evidence_leaves_the_item_open() {
        let layers = layers();
        let id = layers.next_item_id().unwrap();
        layers
            .add_item(id, &NewOpenItem {
                job: None,
                from_event_id: 1,
                through_event_id: 5,
                text: "is the cache warm [guess]".into(),
                distance: 0,
            })
            .unwrap();
        let scripted = Scripted::new(&[
            &format!("I thought about it.\n\nSETTLED\n- U{id}: it must be warm [inferred]"),
            &format!("More.\n\nSETTLED\n- U{id}: an old run said so [observed #3]\n- U999: nothing"),
        ]);

        let mut messages = conversation();
        let plan = plan_of_two();
        let transcripts = episode_texts(memory_layers::live_turns(&messages), &plan);
        let written = write_gists(&scripted, &transcripts, open_snapshot(&layers).unwrap(), id + 1, None);
        let mut outcome = FoldOutcome::default();
        apply_written(
            &mut messages, &layers, &memory_layers::FakeArchive(vec![10, 20, 30]), &plan, written, 5, 99, 0, "scripted", &mut outcome,
            &mut |_| {},
        )
        .unwrap();

        assert_eq!(outcome.settled, 0);
        assert_eq!(outcome.refused, 3, "inference, out-of-stretch evidence, unknown item");
        assert_eq!(layers.item(id).unwrap().unwrap().state, ItemState::Open);
    }

    #[test]
    fn a_gist_keeps_only_citations_of_tool_results_in_its_own_stretch() {
        let layers = layers();
        let id = layers.next_item_id().unwrap();
        layers
            .add_item(id, &NewOpenItem { from_event_id: 1, through_event_id: 5, text: "is it the lock [guess]".into(), distance: 0 , job: None})
            .unwrap();
        // Episode one covers #6..#19 and holds result #10; episode two covers
        // #11..#29 and holds #20. #12 is in two's range but is no tool result.
        let scripted = Scripted::new(&[
            "The lease waited [observed #10]. The reserve log said sleep [observed #20]. \
             And something else [observed #9999].\n\nNOT SETTLED\n- whether it recurs [observed #9999]",
            &format!("I checked again [observed #20].\n\nSETTLED\n- U{id}: it's the lock [observed #12]"),
        ]);
        let mut messages = conversation();
        let plan = plan_of_two();
        let written = write_gists(&scripted, &episode_texts(memory_layers::live_turns(&messages), &plan),
            open_snapshot(&layers).unwrap(), id + 1, None);
        let mut outcome = FoldOutcome::default();
        let mut statuses = Vec::new();
        apply_written(&mut messages, &layers, &memory_layers::FakeArchive(vec![10, 20, 30]), &plan, written, 5, 99, 0,
            "scripted", &mut outcome, &mut |status| statuses.push(status)).unwrap();

        let episodes = layers.level(Level::Episode).unwrap();
        assert_eq!(
            episodes[0].text.lines().next().unwrap(),
            "The lease waited [observed #10]. The reserve log said sleep [observed]. And something else [observed]."
        );
        assert!(episodes[1].text.starts_with("I checked again [observed #20]."), "{}", episodes[1].text);
        // The item opened carries no citation it can't back into deep sleep.
        let opened = layers.open_items().unwrap().into_iter().find(|item| item.id != id).unwrap();
        assert_eq!(opened.text, "whether it recurs [observed]");
        // A number in range that is no observation settles nothing.
        assert_eq!(layers.item(id).unwrap().unwrap().state, ItemState::Open);
        assert_eq!((outcome.uncited, outcome.settled, outcome.refused), (4, 0, 1), "{statuses:?}");
        assert!(statuses.iter().any(|s| s.contains("took #20, #9999, #9999 out of the gist for events #6..#19")), "{statuses:?}");
        assert_eq!((episodes[0].source_hash.as_str(), episodes[1].source_hash.as_str()), ("hash-6-19", "hash-11-29"));
    }

    #[test]
    fn with_the_archive_unreadable_a_fold_stores_no_numbered_citation() {
        let layers = layers();
        let scripted = Scripted::new(&["Saw it [observed #10].", "Saw that [observed #20]."]);
        let mut messages = conversation();
        let plan = plan_of_two();
        let written = write_gists(&scripted, &episode_texts(memory_layers::live_turns(&messages), &plan), Vec::new(),
            layers.next_item_id().unwrap(), None);
        let mut outcome = FoldOutcome::default();
        let mut statuses = Vec::new();
        apply_written(&mut messages, &layers, &memory_layers::UnreadableArchive, &plan, written, 5, 99, 0, "scripted",
            &mut outcome, &mut |status| statuses.push(status)).unwrap();

        let episodes = layers.level(Level::Episode).unwrap();
        assert_eq!(episodes.len(), 2, "the fold still happens");
        assert!(episodes[0].text.starts_with("Saw it [observed]."), "{}", episodes[0].text);
        assert!(episodes.iter().all(|gist| gist.source_hash.is_empty()));
        assert_eq!(outcome.uncited, 2);
        assert!(statuses[0].contains("archive unreadable"), "{statuses:?}");
        assert_eq!(memory_layers::live_turns(&messages).len(), 3, "the folded turns left the context");
    }

    #[test]
    fn a_failed_gist_is_a_stub_and_opens_nothing() {
        let layers = layers();
        let scripted = Scripted::new(&["Fine.\n\nNOT SETTLED\n- one thing [guess]"]);
        let mut messages = conversation();
        let plan = plan_of_two();
        let transcripts = episode_texts(memory_layers::live_turns(&messages), &plan);
        // The second call has no reply scripted, so it fails.
        let first = layers.next_item_id().unwrap();
        let written = write_gists(&scripted, &transcripts, Vec::new(), first, None);
        assert!(written[1].is_err());

        let mut outcome = FoldOutcome::default();
        apply_written(
            &mut messages, &layers, &memory_layers::FakeArchive(vec![10, 20, 30]), &plan, written, 5, 99, 0, "scripted", &mut outcome,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!((outcome.episodes, outcome.mechanical, outcome.opened), (1, 1, 1));
        assert_eq!(layers.open_items().unwrap()[0].id, first, "numbered as the writer saw it");
    }

    // ---- preparation ----

    #[test]
    fn masking_a_tool_result_does_not_invalidate_a_prepared_fold() {
        let original = conversation();
        let mut masked = original.clone();
        masked[4].content = format!("[MASKED TOOL RESULT] pointer{}", trailer(10));
        assert_eq!(fingerprint(&original), fingerprint(&masked));

        let mut edited = original.clone();
        edited[2].content = "a different request".into();
        assert_ne!(fingerprint(&original), fingerprint(&edited));
    }

    #[test]
    fn the_fold_view_leaves_out_harness_blocks() {
        let mut messages = conversation();
        messages.insert(5, Message::user(format!("{}\nnothing open", crate::open_loops::BLOCK_MARKER)));
        let view = fold_view(&messages).unwrap();
        assert_eq!(view.len(), memory_layers::live_turns(&conversation()).len());
        assert!(!view.iter().any(is_harness_block));

        let bare = vec![Message::system("prefix".into()), Message::user("hi".into())];
        assert!(fold_view(&bare).is_none(), "no memory message yet");
    }

    fn usage_at(input_tokens: usize) -> ContextUsage {
        ContextUsage { input_tokens, projected_tokens: input_tokens + 32_768, capacity: 1_048_576 }
    }

    fn job_for(messages: &[Message], layers: &MemoryLayers, handle: std::thread::JoinHandle<Written>) -> PrepJob {
        let live = memory_layers::live_turns(messages).to_vec();
        let plan = plan_of_two();
        PrepJob {
            usable: Arc::new(AtomicUsize::new(2)),
            announced: false,
            open_job: crate::jobs::gist_tag(),
            handle,
            fingerprint: fingerprint(&live[..plan.messages_folded]),
            plan,
            first_new_id: layers.next_item_id().unwrap(),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn a_long_window_is_told_when_its_prepared_fold_is_ready() {
        let layers = layers();
        let messages = conversation();
        let (release, hold) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::spawn(move || -> Written {
            let _ = hold.recv();
            vec![Ok("a".into()), Ok("b".into())]
        });
        let mut prep = FoldPrep::default();
        prep.job = Some(job_for(&messages, &layers, handle));
        // Past 500k, still writing: not ready, and the turn goes on.
        let due = usage_at(context_guard::LONG_FOLD_AT_TOKENS + 10_000);
        assert!(!prep.maybe_start(&messages, &due, &layers));
        assert!(!due.fold_need().should_fold(false));
        assert!(prep.job.as_ref().unwrap().announced, "said once that the fold waits for them");
        release.send(()).unwrap();
        while !prep.job.as_ref().unwrap().handle.is_finished() {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(prep.maybe_start(&messages, &due, &layers), "written and still for these turns: ready");
        assert!(due.fold_need().should_fold(true));
    }

    #[test]
    fn background_gists_that_failed_are_not_ready_and_are_written_again() {
        let layers = layers();
        let messages = conversation();
        // Done, but the lane never came free: nothing usable.
        let handle = std::thread::spawn(|| -> Written { vec![Err("no lane".into()), Err("no lane".into())] });
        let mut job = job_for(&messages, &layers, handle);
        job.usable = Arc::new(AtomicUsize::new(0));
        while !job.handle.is_finished() {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!job.all_written());
        let mut prep = FoldPrep::default();
        prep.job = Some(job);
        // Due: kept, for the fold to take what it can.
        assert!(!prep.maybe_start(&messages, &usage_at(context_guard::LONG_FOLD_CEILING_TOKENS), &layers));
        assert!(prep.is_active());
        // Below the prepare point, so nothing new is started in its place.
        assert!(!prep.maybe_start(&messages, &usage_at(100_000), &layers), "a fold from these would stop the turn");
        assert!(!prep.is_active(), "dropped, to be written again");
        assert_eq!(usable_count(&vec![Ok("a".into()), Ok("  ".into()), Ok("c".into())]), 1, "counted from the first");
    }

    #[test]
    fn a_short_window_keeps_its_prepared_fold_for_the_fold_to_sort_out() {
        let layers = layers();
        let messages = conversation();
        let handle = std::thread::spawn(|| -> Written { vec![Ok("a".into()), Ok("b".into())] });
        let mut prep = FoldPrep::default();
        prep.job = Some(job_for(&messages, &layers, handle));
        let mut changed = messages.clone();
        changed[memory_layers::MEMORY_INDEX + 1].content = "something else".into();
        let short = ContextUsage { input_tokens: 100_000, projected_tokens: 132_768, capacity: 262_144 };
        prep.maybe_start(&changed, &short, &layers);
        assert!(prep.is_active(), "as before: take_valid drops it when the fold comes");
    }

    #[test]
    fn a_stale_prepared_fold_is_dropped_before_the_fold_waits_on_it() {
        let layers = layers();
        let messages = conversation();
        let handle = std::thread::spawn(|| -> Written { vec![Ok("a".into()), Ok("b".into())] });
        let mut prep = FoldPrep::default();
        prep.job = Some(job_for(&messages, &layers, handle));
        let mut changed = messages.clone();
        changed[memory_layers::MEMORY_INDEX + 1].content = "something else".into();
        // Below the prepare point, so nothing new is started in its place.
        assert!(!prep.maybe_start(&changed, &usage_at(100_000), &layers));
        assert!(!prep.is_active(), "dropped, to be written again");
    }

    #[test]
    fn a_prepared_fold_is_refused_when_its_turns_changed() {
        let layers = layers();
        let messages = conversation();
        let live = memory_layers::live_turns(&messages).to_vec();
        let plan = plan_of_two();
        let handle = std::thread::spawn(|| -> Written { vec![Ok("a".into()), Ok("b".into())] });
        let mut prep = FoldPrep {
            job: Some(PrepJob {
                usable: Arc::new(AtomicUsize::new(2)),
                announced: false,
                open_job: None,
                handle,
                fingerprint: fingerprint(&live[..plan.messages_folded]),
                plan,
                first_new_id: layers.next_item_id().unwrap(),
                cancel: Arc::new(AtomicBool::new(false)),
            }),
            draining: Vec::new(),
        };

        let mut changed = live.clone();
        changed[0].content = "something else".into();
        assert!(prep.take_valid(&changed, &layers, &mut |_| {}).unwrap().is_none());
        assert!(!prep.is_active(), "a refused job is gone, not retried");
    }

    #[test]
    fn preparation_writes_on_a_thread_and_the_fold_stores_it() {
        let layers = layers();
        let mut messages = vec![Message::system("prefix".into())];
        memory_layers::ensure_layout(&mut messages);
        for n in 0..8u64 {
            messages.push(Message::user(format!("step {n}")));
            messages.push(call(&format!("c{n}")));
            messages.push(tool(&format!("c{n}"), "ok", 100 + n));
            messages.push(Message::assistant(format!("{n}: {}", "w".repeat(100_000))));
        }
        let replies: Vec<String> =
            (0..8).map(|n| format!("Episode {n}.\n\nNOT SETTLED\n- open {n} [guess]")).collect();
        let scripted = Scripted::new(&replies.iter().map(String::as_str).collect::<Vec<_>>());

        let mut prep = FoldPrep::default();
        let episodes = prep.start(&messages, &layers, &FoldSchedule::SHORT, scripted, Arc::new(AtomicBool::new(false))).unwrap().expect("a plan");
        assert!(episodes >= 2);

        let live = memory_layers::live_turns(&messages).to_vec();
        let (plan, written) = prep.take_valid(&live, &layers, &mut |_| {}).unwrap().expect("still valid");
        assert_eq!(written.len(), plan.episodes.len());

        let mut outcome = FoldOutcome::default();
        apply_written(
            &mut messages, &layers, &memory_layers::FakeArchive(vec![10, 20, 30]), &plan, written, 0, 999, 0, "scripted", &mut outcome, &mut |_| {},
        )
        .unwrap();
        assert_eq!(outcome.episodes, plan.episodes.len());
        assert_eq!(outcome.opened, plan.episodes.len(), "one open item per episode, numbered in order");
        let ids: Vec<i64> = layers.open_items().unwrap().iter().map(|item| item.id).collect();
        assert!(ids.windows(2).all(|pair| pair[1] == pair[0] + 1), "{ids:?}");
    }

    #[test]
    fn a_prepared_fold_is_used_when_nothing_changed() {
        let layers = layers();
        let messages = conversation();
        let live = memory_layers::live_turns(&messages).to_vec();
        let plan = plan_of_two();
        let mut prep = FoldPrep::default();
        prep.job = Some(PrepJob {
            usable: Arc::new(AtomicUsize::new(2)),
            announced: false,
            open_job: None,
            handle: std::thread::spawn(|| -> Written { vec![Ok("a".into()), Ok("b".into())] }),
            fingerprint: fingerprint(&live[..plan.messages_folded]),
            plan: plan.clone(),
            first_new_id: layers.next_item_id().unwrap(),
            cancel: Arc::new(AtomicBool::new(false)),
        });
        let (used, written) = prep.take_valid(&live, &layers, &mut |_| {}).unwrap().unwrap();
        assert_eq!(used, plan);
        assert_eq!(written.len(), 2);
    }

    #[test]
    fn a_refused_settlement_leaves_the_item_for_a_later_episode() {
        let layers = layers();
        let id = layers.next_item_id().unwrap();
        layers
            .add_item(id, &NewOpenItem { from_event_id: 1, through_event_id: 5, text: "is it the lock [guess]".into(), distance: 0 , job: None})
            .unwrap();
        let scripted = Scripted::new(&[
            &format!("Thought about it.\n\nSETTLED\n- U{id}: must be the lock [inferred]"),
            &format!("Ran it.\n\nSETTLED\n- U{id}: the trace shows the lock [observed #20]"),
        ]);
        let mut messages = conversation();
        let plan = plan_of_two();
        let episodes = episode_texts(memory_layers::live_turns(&messages), &plan);
        let written = write_gists(&scripted, &episodes, open_snapshot(&layers).unwrap(), id + 1, None);
        assert!(scripted.shown.lock().unwrap()[1].contains(&format!("U{id}:")), "still shown to episode 2");

        let mut outcome = FoldOutcome::default();
        apply_written(&mut messages, &layers, &memory_layers::FakeArchive(vec![10, 20, 30]), &plan, written, 5, 99, 0, "scripted", &mut outcome, &mut |_| {}).unwrap();
        assert_eq!((outcome.refused, outcome.settled), (1, 1));
        assert_eq!(layers.item(id).unwrap().unwrap().state, ItemState::Settled);
    }

    #[test]
    fn a_pointer_named_in_a_gist_is_refused() {
        let layers = layers();
        for n in 0..60u64 {
            let id = layers.next_item_id().unwrap();
            layers
                .add_item(id, &NewOpenItem { from_event_id: n + 1, through_event_id: n + 1, text: format!("item {n} {}", "q".repeat(180)), distance: 0, job: None })
                .unwrap();
        }
        layers.enforce_item_budget().unwrap();
        let pointer = layers.open_items().unwrap().into_iter().find(|item| item.kind == memory_layers::ItemKind::Pointer).unwrap();
        assert!(!open_snapshot(&layers).unwrap().iter().any(|(id, _)| *id == pointer.id), "writers never see pointers");

        let scripted = Scripted::new(&[
            &format!("Ran it.\n\nSETTLED\n- U{}: all answered [observed #10]", pointer.id),
            "Nothing else.",
        ]);
        let mut messages = conversation();
        let plan = plan_of_two();
        let episodes = episode_texts(memory_layers::live_turns(&messages), &plan);
        let written = write_gists(&scripted, &episodes, Vec::new(), layers.next_item_id().unwrap(), None);
        let mut outcome = FoldOutcome::default();
        apply_written(&mut messages, &layers, &memory_layers::FakeArchive(vec![10, 20, 30]), &plan, written, 5, 99, 0, "scripted", &mut outcome, &mut |_| {}).unwrap();
        assert_eq!((outcome.settled, outcome.refused), (0, 1));
        assert_eq!(layers.unsettled_items().unwrap().len(), 60, "nothing closed wholesale");
    }

    #[test]
    fn a_cancelled_writer_stops_without_calling_the_model() {
        let scripted = Scripted::new(&["never used"]);
        let cancel = AtomicBool::new(true);
        let episodes = vec![
            EpisodeText { transcript: "a".into(), operator_spoke: false },
            EpisodeText { transcript: "b".into(), operator_spoke: false },
        ];
        let written = write_gists(&scripted, &episodes, Vec::new(), 1, Some(&cancel));
        assert!(written.iter().all(|result| result.as_ref().err().map(String::as_str) == Some("cancelled")));
        assert!(scripted.shown.lock().unwrap().is_empty());
    }

    #[test]
    fn quiesce_stops_and_joins_the_background_writer() {
        let layers = layers();
        let messages = conversation();
        let live = memory_layers::live_turns(&messages).to_vec();
        let plan = plan_of_two();
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        let handle = std::thread::spawn(move || -> Written {
            while !flag.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            vec![Err("cancelled".into())]
        });
        let mut prep = FoldPrep {
            job: Some(PrepJob { usable: Arc::new(AtomicUsize::new(2)), announced: false, open_job: None, handle, fingerprint: fingerprint(&live[..8]), plan, first_new_id: layers.next_item_id().unwrap(), cancel }),
            draining: Vec::new(),
        };
        prep.quiesce();
        assert!(!prep.is_active());
    }

    #[test]
    fn a_stale_job_is_cancelled_and_still_waited_for_by_quiesce() {
        let layers = layers();
        let messages = conversation();
        let live = memory_layers::live_turns(&messages).to_vec();
        let plan = plan_of_two();
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        let finished = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&finished);
        let handle = std::thread::spawn(move || -> Written {
            while !flag.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            done.store(true, Ordering::SeqCst);
            vec![Err("cancelled".into())]
        });
        let mut prep = FoldPrep::default();
        prep.job = Some(PrepJob {
            usable: Arc::new(AtomicUsize::new(2)),
            announced: false,
            open_job: None,
            handle,
            fingerprint: vec![0; plan.messages_folded],
            plan,
            first_new_id: layers.next_item_id().unwrap(),
            cancel: Arc::clone(&cancel),
        });

        assert!(prep.take_valid(&live, &layers, &mut |_| {}).unwrap().is_none(), "stale");
        assert!(cancel.load(Ordering::SeqCst), "cancelled, not left running");
        prep.quiesce();
        assert!(finished.load(Ordering::SeqCst), "quiesce waited for the abandoned thread");
    }

    #[test]
    fn a_prepared_fold_keeps_what_was_written_before_a_failure() {
        let layers = layers();
        let messages = conversation();
        let live = memory_layers::live_turns(&messages).to_vec();
        let plan = plan_of_two();
        let mut prep = FoldPrep {
            job: Some(PrepJob {
                usable: Arc::new(AtomicUsize::new(2)),
                announced: false,
                open_job: None,
                handle: std::thread::spawn(|| -> Written { vec![Ok("first".into()), Err("no lane".into())] }),
                fingerprint: fingerprint(&live[..plan.messages_folded]),
                plan: plan.clone(),
                first_new_id: layers.next_item_id().unwrap(),
                cancel: Arc::new(AtomicBool::new(false)),
            }),
            draining: Vec::new(),
        };
        let mut statuses = Vec::new();
        let (kept, written) = prep.take_valid(&live, &layers, &mut |status| statuses.push(status)).unwrap().unwrap();
        assert_eq!(kept.episodes.len(), 1);
        assert_eq!(kept.messages_folded, plan.episodes[0].end);
        assert_eq!(written.len(), 1);
        assert!(statuses.iter().any(|status| status.contains("failed (no lane)")), "{statuses:?}");
    }

    // ---- handoff ----

    #[test]
    fn deep_sleep_moves_unsettled_items_into_the_world_model_once() {
        let root = std::env::temp_dir().join(format!(
            "s0ma-fold-handoff-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap(),
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut memory = Hippocampus::new(root.join("hippocampus.jsonl")).unwrap();
        memory.remember_operator("the reserve log says slot 3".into()).unwrap();
        let real_event = memory.last_event_id();
        let layers = MemoryLayers::beside_archive(memory.archive_path());

        let add = |text: &str, through: u64| {
            let id = layers.next_item_id().unwrap();
            layers
                .add_item(id, &NewOpenItem {
                    job: None,
                    from_event_id: 1,
                    through_event_id: through,
                    text: text.to_string(),
                    distance: 0,
                })
                .unwrap();
            id
        };
        let observed = add(&format!("slot 3 is held, by what? [observed #{real_event}]"), 10);
        let bad_cite = add("an exit code of 7 [observed #999999]", 10);
        let reported = add("The operator says the pool is the cause [operator said]", 10);
        let settled = add("is the cache warm [guess]", 10);
        let later = add("beyond this sleep's window [inferred]", 500);
        layers.resolve_item(settled, ItemState::Settled, "yes [operator said]").unwrap();

        let report = hand_off_unsettled(&layers, &memory, 100).unwrap();
        assert_eq!((report.written, report.already), (3, 0), "{:?}", report.failed);
        assert!(report.failed.is_empty());

        let path = crate::world_model::path_for(&memory);
        let body = |id: i64| -> serde_json::Value {
            let db = crate::world_model::open(&path).unwrap();
            let raw: String = db
                .query_row(
                    "SELECT body FROM wm_records WHERE id=?1",
                    [format!("{UNSETTLED_MODEL}:U{id}")],
                    |row| row.get(0),
                )
                .unwrap();
            serde_json::from_str(&raw).unwrap()
        };
        assert_eq!(body(observed)["basis"], "observed");
        assert_eq!(body(observed)["supporting_evidence_ids"][0], real_event);
        assert_eq!(body(bad_cite)["basis"], "inferred", "a citation that doesn't resolve claims less");
        assert_eq!(body(reported)["basis"], "reported");
        assert_eq!(body(reported)["status"], "unresolved");
        assert!(!crate::world_model::record_exists(&path, &format!("{UNSETTLED_MODEL}:U{settled}")).unwrap());
        assert!(!crate::world_model::record_exists(&path, &format!("{UNSETTLED_MODEL}:U{later}")).unwrap());

        // A retried sleep finds nothing left to hand over.
        let again = hand_off_unsettled(&layers, &memory, 100).unwrap();
        assert_eq!((again.written, again.already, again.retired), (0, 0, 0));

        // And clearing the window leaves only what is still open beyond it.
        layers.clear_through(100).unwrap();
        let open: Vec<i64> = layers.open_items().unwrap().iter().map(|item| item.id).collect();
        assert_eq!(open, vec![later]);
    }

    #[test]
    fn moot_and_late_settled_items_are_retired_in_the_world_model() {
        let root = std::env::temp_dir().join(format!(
            "s0ma-fold-retire-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap(),
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut memory = Hippocampus::new(root.join("hippocampus.jsonl")).unwrap();
        memory.remember_operator("anything".into()).unwrap();
        let layers = MemoryLayers::beside_archive(memory.archive_path());
        let add = |text: &str| {
            let id = layers.next_item_id().unwrap();
            layers
                .add_item(id, &NewOpenItem { from_event_id: 1, through_event_id: 10, text: text.into(), distance: 0 , job: None})
                .unwrap();
            id
        };
        let mooted = add("whether to port the cache [inferred]");
        let late = add("whether the pool holds the lanes [operator said]");

        // A sleep that hands `late` over and then fails.
        hand_off_unsettled(&layers, &memory, 100).unwrap();

        // Working memory carries on: one item goes moot, the other is settled.
        layers.resolve_item(mooted, ItemState::Moot, "the port was cancelled").unwrap();
        layers.resolve_item(late, ItemState::Settled, "sleep holds them [observed #1]").unwrap();

        let report = hand_off_unsettled(&layers, &memory, 100).unwrap();
        assert_eq!(report.retired, 2, "{:?}", report.failed);

        let path = crate::world_model::path_for(&memory);
        for (id, says) in [(mooted, "Called moot"), (late, "Settled in working memory")] {
            let (revision, status) = crate::world_model::record_state(&path, &format!("{UNSETTLED_MODEL}:U{id}")).unwrap().unwrap();
            assert_eq!(status, "retired");
            assert!(revision >= 1);
            let db = crate::world_model::open(&path).unwrap();
            let body: String = db
                .query_row("SELECT body FROM wm_records WHERE id=?1", [format!("{UNSETTLED_MODEL}:U{id}")], |row| row.get(0))
                .unwrap();
            assert!(body.contains(says), "{body}");
        }

        // Once retired, a later sleep leaves them alone.
        assert_eq!(hand_off_unsettled(&layers, &memory, 100).unwrap().retired, 0);
    }

    #[test]
    fn a_verdict_deep_sleep_already_wrote_is_never_overwritten() {
        let root = std::env::temp_dir().join(format!(
            "s0ma-fold-verdict-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap(),
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut memory = Hippocampus::new(root.join("hippocampus.jsonl")).unwrap();
        memory.remember_operator("the reserve log".into()).unwrap();
        let event = memory.last_event_id();
        let layers = MemoryLayers::beside_archive(memory.archive_path());
        let id = layers.next_item_id().unwrap();
        layers
            .add_item(id, &NewOpenItem { from_event_id: 1, through_event_id: 10, text: "who holds slot 3 [guess]".into(), distance: 0 , job: None})
            .unwrap();

        // Sleep 1 hands it over; integration rules on it; then sleep 1 fails.
        hand_off_unsettled(&layers, &memory, 100).unwrap();
        let path = crate::world_model::path_for(&memory);
        let record_id = format!("{UNSETTLED_MODEL}:U{id}");
        let (revision, _) = crate::world_model::record_state(&path, &record_id).unwrap().unwrap();
        let verdict = crate::world_model::Record {
            id: record_id.clone(),
            model_id: UNSETTLED_MODEL.into(),
            kind: "claim".into(),
            title: "who holds slot 3".into(),
            statement: "Sleep holds slot 3.".into(),
            subject_id: None,
            target_id: None,
            basis: "observed".into(),
            status: "supported".into(),
            applies_under: vec![],
            alternatives: vec![],
            predictions: vec![],
            supporting_evidence_ids: vec![event],
            counterevidence_ids: vec![],
            dependencies: BTreeMap::new(),
            observed_at: None,
            still_uncertain: String::new(),
            ..Default::default()
        };
        crate::world_model::put(&path, verdict, revision, "integration", &memory).unwrap();

        // Working memory settles it too, and sleep 2 runs.
        layers.resolve_item(id, ItemState::Settled, "sleep holds it [observed #1]").unwrap();
        let report = hand_off_unsettled(&layers, &memory, 100).unwrap();
        assert_eq!(report.retired, 0);
        assert_eq!(
            crate::world_model::record_state(&path, &record_id).unwrap().unwrap().1,
            "supported",
            "the evidence-backed verdict stands",
        );
        layers.clear_through(100).unwrap();
        assert!(layers.item(id).unwrap().is_none(), "dealt with, so cleared");
    }

    /// Provenance: every gist a fold or merge stores names the archive events
    /// it was written from. Checked against a real archive written the way
    /// agent.rs and app.rs write it, through the same calls `fold_context`
    /// makes (plan, write, store, merge cascade, rebuild).
    mod provenance {
        use super::*;
        use crate::hippocampus::Hippocampus;

        /// A session as it is recorded: each context message beside the
        /// archive events it came from.
        struct Session {
            memory: Hippocampus,
            layers: MemoryLayers,
            messages: Vec<Message>,
            /// Parallel to `messages`.
            sources: Vec<Vec<u64>>,
            calls: usize,
        }

        impl Session {
            fn new() -> Self {
                static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let root = std::env::temp_dir().join(format!(
                    "s0ma-provenance-{}-{}",
                    std::process::id(),
                    COUNTER.fetch_add(1, Ordering::SeqCst)
                ));
                let _ = std::fs::remove_dir_all(&root);
                std::fs::create_dir_all(&root).unwrap();
                let archive = root.join("hippocampus.jsonl");
                let mut messages = vec![Message::system("prefix".into())];
                memory_layers::ensure_layout(&mut messages);
                let sources = vec![Vec::new(); messages.len()];
                Self { memory: Hippocampus::new(&archive).unwrap(), layers: MemoryLayers::beside_archive(&archive), messages, sources, calls: 0 }
            }

            fn push(&mut self, message: Message, sources: Vec<u64>) {
                self.messages.push(message);
                self.sources.push(sources);
            }

            fn operator(&mut self, text: &str) {
                self.memory.remember_operator(text.into()).unwrap();
                let id = self.memory.last_event_id();
                self.push(Message::user(text.into()).by_operator(), vec![id]);
            }

            fn reply(&mut self, text: String) {
                self.memory.remember_s0ma(text.clone()).unwrap();
                let id = self.memory.last_event_id();
                self.push(Message::assistant(text), vec![id]);
            }

            /// A tool call and its result, archived and trailed as agent.rs
            /// does. `arrives_mid_call` is a subagent completion archived while
            /// the call runs and seen in context after its result.
            fn tool(&mut self, body: &str, arrives_mid_call: bool) -> u64 {
                self.calls += 1;
                let call_id = format!("c{}", self.calls);
                let completion = arrives_mid_call.then(|| {
                    self.memory.remember_s0ma_intermediate("[subagent_summary]\n{\"event\":\"worker_finished\"}".into()).unwrap();
                    self.memory.last_event_id()
                });
                let event = self.memory.remember_tool_result(format!("[tool:vm_exec]\narguments:\n{{}}\nresult:\n{body}")).unwrap();
                self.push(call(&call_id), vec![event]);
                self.push(tool(&call_id, body, event), vec![event]);
                if let Some(completion) = completion {
                    self.push(Message::user("[SUBAGENT COMPLETION — TOOL OBSERVATION, NOT OPERATOR INPUT]\n{}".into()), vec![completion]);
                }
                event
            }

            /// One step of work: the operator asks, s0ma runs something, s0ma answers.
            fn step(&mut self, n: usize, body: &str, arrives_mid_call: bool) {
                self.operator(&format!("step {n}"));
                self.tool(body, arrives_mid_call);
                self.reply(format!("step {n} done: {}", "w".repeat(60_000)));
            }

            /// Fold the way `fold_context` does; returns, for each stored
            /// episode, the archive events its turns came from.
            fn fold(&mut self, summarizer: &dyn Summarizer, target: usize) -> Vec<Vec<u64>> {
                let live = memory_layers::live_turns(&self.messages);
                let plan = memory_layers::plan_fold(live, target);
                assert!(plan.episodes.len() >= 2, "a fold of several episodes: {plan:?}");
                let base = memory_layers::MEMORY_INDEX + 1;
                let expected: Vec<Vec<u64>> = plan.episodes.iter()
                    .map(|e| self.sources[base + e.start..base + e.end].iter().flatten().copied().collect())
                    .collect();
                let episodes = episode_texts(live, &plan);
                // The writer is shown every tool result's archive id, so it can
                // cite each one as [observed #N].
                for (episode, text) in plan.episodes.iter().zip(&episodes) {
                    let shown = trailer_ids(&text.transcript);
                    for (offset, message) in live[episode.start..episode.end].iter().enumerate() {
                        if message.role == "tool" {
                            let id = self.sources[base + episode.start + offset][0];
                            assert!(shown.contains(&id), "result #{id} reaches the gist writer without its id");
                        }
                    }
                }
                let written = write_gists(summarizer, &episodes, open_snapshot(&self.layers).unwrap(),
                    self.layers.next_item_id().unwrap(), None);
                let floor = self.layers.fold_floor().unwrap();
                let mut outcome = FoldOutcome::default();
                apply_written(&mut self.messages, &self.layers, &self.memory, &plan, written, floor, self.memory.last_event_id(),
                    0, "honest", &mut outcome, &mut |_| {}).unwrap();
                self.sources.drain(base..base + plan.messages_folded);
                cascade_and_rebuild(&mut self.messages, &self.layers, summarizer, &self.memory, &FoldSchedule::SHORT, &mut outcome, &mut |_| {}).unwrap();
                expected
            }

            fn gists(&self) -> Vec<Gist> {
                let mut all = self.layers.level(Level::Shape).unwrap();
                all.extend(self.layers.level(Level::Episode).unwrap());
                all
            }

            /// What every stored gist cites, checked against the archive.
            fn check_citations(&self) {
                let last = self.memory.last_event_id();
                let memory_message = &self.messages[memory_layers::MEMORY_INDEX].content;
                for gist in self.gists() {
                    let (from, through) = (gist.from_event_id, gist.through_event_id);
                    assert!(1 <= from && from <= through && through <= last, "range #{from}..#{through} outside the archive (1..#{last})");
                    assert!(memory_message.contains(&format!("events #{from}..#{through}]")), "s0ma is shown the range #{from}..#{through}");
                    if gist.mechanical {
                        assert!(gist.text.contains(&format!("#{from}..#{through}")), "a stub names its range: {}", gist.text);
                    }
                    // The hash is the stretch's own, read back from the archive.
                    let span = self.memory.spans(&[(from, through)]).unwrap().remove(0);
                    assert_eq!(gist.source_hash, span.source_hash, "source hash of #{from}..#{through}");
                    let cited = cited_observations(&gist.text);
                    let found = self.memory.get_events(&cited).unwrap();
                    for id in &cited {
                        let event = found.iter().find(|e| e.id == *id).unwrap_or_else(|| panic!("[observed #{id}] is not in the archive"));
                        assert!(event.content.starts_with("[tool:"), "[observed #{id}] is a tool result");
                        assert!((from..=through).contains(id), "[observed #{id}] is outside its gist's range #{from}..#{through}");
                    }
                }
                // NOT SETTLED items carry their citations to the world model.
                for item in self.layers.unsettled_items().unwrap() {
                    let (from, through) = (item.from_event_id, item.through_event_id);
                    let cited = cited_observations(&item.text);
                    let found = self.memory.get_events(&cited).unwrap();
                    for id in &cited {
                        let event = found.iter().find(|e| e.id == *id).unwrap_or_else(|| panic!("item cites #{id}, not in the archive"));
                        assert!(event.content.starts_with("[tool:") && (from..=through).contains(id), "item cites #{id} outside #{from}..#{through}");
                    }
                }
            }
        }

        /// Writes what a careful model would: an [observed #N] for each tool
        /// result it was shown, with N copied from that result's trailer.
        struct Honest {
            calls: Mutex<usize>,
            fail: Vec<usize>,
        }

        impl Honest {
            fn new(fail: &[usize]) -> Self {
                Self { calls: Mutex::new(0), fail: fail.to_vec() }
            }
        }

        fn trailer_ids(transcript: &str) -> Vec<u64> {
            transcript.match_indices("archive_event_id ").filter_map(|(at, marker)| {
                let digits: String = transcript[at + marker.len()..].chars().take_while(char::is_ascii_digit).collect();
                digits.parse().ok()
            }).collect()
        }

        impl Summarizer for Honest {
            fn gist(&self, transcript: &str, _open_items: &str) -> Result<String> {
                let mut calls = self.calls.lock().unwrap();
                *calls += 1;
                if self.fail.contains(&*calls) {
                    return Err("the lane was busy".into());
                }
                let mut text: String = trailer_ids(transcript).iter().map(|id| format!("I ran a step and saw its result [observed #{id}]. ")).collect();
                text.push_str(&"p".repeat(16_000));
                Ok(text)
            }
            fn merge(&self, _level: Level, inputs: &[Gist]) -> Result<String> {
                Ok(inputs.iter().map(|gist| gist.text.clone()).collect::<Vec<_>>().join(" "))
            }
            fn model_name(&self) -> String {
                "honest".into()
            }
        }

        /// Cites the first result it was shown, then one from far away and one
        /// that doesn't exist; its merges cite something nobody did.
        struct Careless;

        impl Summarizer for Careless {
            fn gist(&self, transcript: &str, _open_items: &str) -> Result<String> {
                let first = trailer_ids(transcript)[0];
                Ok(format!(
                    "Saw it [observed #{first}]. Long ago [observed #1]. Never [observed #9999].\n\n\
                     NOT SETTLED\n- whether it holds [observed #{first}, #9999]\n"
                ))
            }
            fn merge(&self, _level: Level, inputs: &[Gist]) -> Result<String> {
                Ok(format!("Merged {} [observed #4242].", inputs.len()))
            }
            fn model_name(&self) -> String {
                "careless".into()
            }
        }

        #[test]
        fn folds_and_gists_cite_the_archive_events_they_were_written_from() {
            let mut session = Session::new();
            for n in 0..6 {
                // A long result (shortened in the transcript), and a subagent
                // completion that arrives during a call.
                let body = if n == 1 { format!("{}EXIT 0", "x".repeat(9_000)) } else { format!("result {n}") };
                session.step(n, &body, n == 3);
            }
            // A masked result keeps its trailer, so it still anchors.
            let masked = session.messages.iter().position(|m| m.role == "tool" && m.content.starts_with("result 2")).unwrap();
            let id = crate::observation_mask::event_id(&session.messages[masked].content).unwrap();
            session.messages[masked].content = format!("[MASKED TOOL RESULT] pointer{}", trailer(id));
            for n in 6..8 {
                session.step(n, &format!("result {n}"), false);
            }

            // Fold one: the second episode's gist can't be written (a stub).
            let expected = session.fold(&Honest::new(&[2]), 60_000);
            let stored = session.layers.level(Level::Episode).unwrap();
            assert_eq!(stored.len(), expected.len());
            assert!(stored.iter().any(|gist| gist.mechanical));
            for (gist, sources) in stored.iter().zip(&expected) {
                for id in sources {
                    assert!((gist.from_event_id..=gist.through_event_id).contains(id),
                        "event #{id} the episode was written from is outside its range {}", gist.range_label());
                }
            }
            session.check_citations();

            // More work, a second fold, and enough gist text that the cascade
            // merges episodes into shape.
            for n in 8..16 {
                session.step(n, &format!("result {n}"), false);
            }
            let before: Vec<i64> = session.gists().iter().map(|g| g.id).collect();
            let expected = session.fold(&Honest::new(&[]), 120_000);
            let fresh: Vec<Gist> = session.layers.level(Level::Episode).unwrap().into_iter().filter(|g| !before.contains(&g.id)).collect();
            assert_eq!(fresh.len(), expected.len());
            for (gist, sources) in fresh.iter().zip(&expected) {
                for id in sources {
                    assert!((gist.from_event_id..=gist.through_event_id).contains(id), "#{id} outside {}", gist.range_label());
                }
            }
            assert!(!session.layers.level(Level::Shape).unwrap().is_empty(), "the cascade merged");
            session.check_citations();

            // A careless writer: ids from other stretches, ids that don't exist.
            // None of them is stored.
            for n in 16..24 {
                session.step(n, &format!("result {n}"), false);
            }
            session.fold(&Careless, 120_000);
            session.check_citations();

            // Nothing folded is left without a gist that names it: every event
            // up to the newest anchor folded sits inside some stored range.
            let floor = session.layers.fold_floor().unwrap();
            let gists = session.gists();
            let uncovered: Vec<u64> = (1..=floor)
                .filter(|id| !gists.iter().any(|g| (g.from_event_id..=g.through_event_id).contains(id)))
                .collect();
            assert!(uncovered.is_empty(), "archived events no gist covers: {uncovered:?}");
        }
    }

    #[test]
    fn an_items_basis_comes_from_its_first_tag() {
        assert_eq!(basis_of("the pool [operator said], maybe [guess]"), ("reported", vec![]));
        assert_eq!(basis_of("the pool [Alex said] earlier"), ("reported", vec![]), "a tag from before, under another name");
        assert_eq!(basis_of("the pool [not a tag said by me]"), ("inferred", vec![]));
        assert_eq!(basis_of("odd exit code [observed #44]"), ("observed", vec![44]));
        assert_eq!(basis_of("odd exit code [observed]"), ("inferred", vec![]));
        assert_eq!(basis_of("probably a race [guess]"), ("interpretation", vec![]));
        assert_eq!(basis_of("no tag at all"), ("inferred", vec![]));
        assert_eq!(clip_bytes("ééé", 3), "é…");
    }
}
