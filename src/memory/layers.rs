//! Graded working memory: the layers between the live turns and the checkpoint.
//!
//! The old shape had two resolutions and a gap between them — the last couple of
//! turns verbatim, fixed-ratio digests for everything else — and the task in
//! progress fell straight into the gap. This module is the graded middle: episode
//! gists at roughly 10:1, Shape entries at roughly 5:1 on those, each layer
//! budget-bounded, with overflow always moving down a level rather than being
//! dropped.
//!
//! Three things make it behave like memory rather than like truncation:
//!
//! - **Age is intervening context, not elapsed time.** A quiet hour costs nothing;
//!   an hour of heavy tool output ages memory fast. Everything here is triggered
//!   by token counts and ordered by a distance odometer.
//! - **Continuity by overlap.** A fold changes a small, old slice. Everything
//!   newer stays exactly where it was, as real conversation turns.
//! - **Shape merges into itself.** A long day compresses further instead of losing
//!   its beginning, so the memory message can't grow with the length of the day.
//!
//! The archive stays the exact record underneath. Every gist ends saying so, which
//! is what turns a vague memory into a reason to look something up rather than a
//! licence to fill in the details (the hard case of confabulation).
//!
//! Uncertainty gets its own layer. A rewrite under a word budget drops qualifiers
//! first, and Shape rewrites itself over and over, so a caveat that lives only in
//! gist prose eventually disappears and its conclusion is left looking settled.
//! What a stretch leaves open is therefore stored apart from the gists, as NOT
//! SETTLED items that no merge reads or writes. An item leaves only when a later
//! gist settles it on evidence from its own stretch, declares it moot, or deep
//! sleep takes it over into the world model.
//!
//! What lives elsewhere: the gist and merge *prompts* and their model calls are in
//! `memory/fold.rs`, behind the `Summarizer` trait below, so this module stays
//! testable without a server; open loops are in `open_loops.rs`.

use crate::context::guard::{GIST_EPISODE_TOKENS, MIN_LIVE_TURNS_TOKENS, OPEN_ITEMS_BUDGET_TOKENS};
// The 262,144 window's budgets: the cascade takes each window's
// (`enforce_budgets_within`); the tests hold the short one to these.
#[cfg(test)]
use crate::context::guard::{GIST_BUDGET_TOKENS, SHAPE_BUDGET_TOKENS};
use crate::types::Message;

use chrono::{DateTime, Local};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Bytes per token for planning estimates. Exact counts come from the server —
/// `usage.input_tokens` for the live layer, `/tokenize` once per stored gist — so
/// this ratio only decides where a chunk boundary falls, never a budget.
const BYTES_PER_TOKEN: usize = 4;

/// A fold re-prefills everything that survives it, so one that frees almost
/// nothing costs a full re-prefill and buys no headroom. Below this, planning
/// reports "nothing foldable" and lets the caller decide, rather than spending
/// ~9 s to remove a handful of tokens.
const MIN_USEFUL_FOLD_TOKENS: usize = 8_000;

/// The most recent stretch of the live layer a fold never takes: at least this
/// much, and always the latest model output whole, with its tool results.
///
/// Whatever a fold removes, the agent keeps its current working set verbatim: its
/// last calls, their results and the reasoning around them. A fold in the
/// middle of a turn then carries on from where the turn was, not from a
/// summary of it. Without this, a fold that missed its target on the first
/// attempt asked for more on the next and took the turn up to its last tool
/// result, leaving the agent nothing of the turn but gists.
pub const KEEP_RECENT_TOKENS: usize = MIN_LIVE_TURNS_TOKENS;

/// Appended to every gist. The single most load-bearing sentence in the module:
/// a coarse gist is adjacent-but-not-identical memory, and without this line the
/// natural thing to do with it is to fill in the specifics.
pub const ARCHIVE_LINE: &str =
    "Exact details of this stretch are in the archive under the event range above \
     (archive_search with from_event/through_event set to that range searches just \
     that stretch; archive_get reads an event). This is my outline of it, not the \
     record — if you need a specific value from here, look it up rather than \
     recalling it.";

// ------------------------------------------------------------------ records

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    /// A first-person account of one ~40k episode, about 10:1 on its source.
    Episode,
    /// Merged gists: the outline of a longer stretch, about 5:1 on the gists.
    Shape,
}

impl Level {
    pub fn label(self) -> &'static str {
        match self {
            Self::Episode => "episode",
            Self::Shape => "shape",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "shape" => Self::Shape,
            _ => Self::Episode,
        }
    }

    /// The budget this layer is held to by the merge cascade on the 262,144
    /// window (`crate::context::guard::FoldSchedule` has each window's).
    #[cfg(test)]
    pub fn budget(self) -> usize {
        match self {
            Self::Episode => GIST_BUDGET_TOKENS,
            Self::Shape => SHAPE_BUDGET_TOKENS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Gist {
    pub id: i64,
    pub level: Level,
    pub from_event_id: u64,
    pub through_event_id: u64,
    /// Hash of the covered archive events, in the same form continuous
    /// compaction uses (`hippocampus::hash_event`), so a gist can be told
    /// whether its source still matches. Empty when the archive couldn't be
    /// read as it was stored, and on gists stored before it was filled in.
    pub source_hash: String,
    /// Measured once, when written, and never re-measured: the budgets are all
    /// comparisons between stored gists.
    pub tokens: usize,
    /// When the stretch happened as it was worked out at writing time: its
    /// dates from the archive (`dated`), or, where the archive couldn't be
    /// read, the distance label older entries used. History only. Rendering
    /// and the merge prompt date each entry from the archive afresh, and show
    /// no time rather than a distance dressed up as one.
    pub time_label: String,
    /// Which model wrote it, so an adapter change is visible in memory's history.
    pub written_by: String,
    pub text: String,
    /// True when no gist could be written and a stub stands in its place.
    pub mechanical: bool,
    pub created_at: String,
    /// Where the stretch this covers ends on the distance odometer (cumulative
    /// context volume, not the clock). It orders and budgets memory; it is not
    /// shown as a time. A merged entry takes its oldest input's position.
    pub distance: u64,
    /// The job that was open when this was written, if there was one
    /// (`jobs::gist_tag`). It decides two things: a gist is never merged with
    /// one carrying a different tag, so no stored row ever holds prose mixing
    /// a job's work with anything else; and a job's close deletes exactly the
    /// rows carrying its name. Absent on everything written outside a job and
    /// on everything written before jobs existed -- both count as "no job" and
    /// merge freely with each other.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job: Option<String>,
}

impl Gist {
    pub fn range_label(&self) -> String {
        format!("#{}..#{}", self.from_event_id, self.through_event_id)
    }

    /// Headed by when the stretch happened, from the archive's timestamps for
    /// its first and last events (`times`), or by its range alone when those
    /// can't be read: an honest gap rather than a distance worded as a time.
    fn render(&self, times: &HashMap<u64, DateTime<Local>>) -> String {
        let mut out = match dated(self.from_event_id, self.through_event_id, times) {
            Some(when) => format!("[{when} | events {}]", self.range_label()),
            None => format!("[events {}]", self.range_label()),
        };
        if self.mechanical {
            out.push_str(" (no gist was written for this stretch — the archive is all there is)");
        }
        out.push('\n');
        out.push_str(self.text.trim());
        out
    }
}

/// A gist as it is about to be stored, before the store assigns it an id.
#[derive(Debug, Clone, PartialEq)]
pub struct NewGist {
    pub level: Level,
    pub from_event_id: u64,
    pub through_event_id: u64,
    pub source_hash: String,
    pub tokens: usize,
    pub time_label: String,
    pub written_by: String,
    pub text: String,
    pub mechanical: bool,
    pub distance: u64,
    pub job: Option<String>,
}

/// The stub that stands in when a gist can't be written.
///
/// It is deliberately not silent. The agent finding out later that a stretch is thin
/// is fine; the agent not knowing there's a hole is not — the stub names the range so
/// the archive lookup is obvious, and every one of these is listed for the next
/// deep sleep.
pub fn mechanical_gist(
    level: Level,
    from_event_id: u64,
    through_event_id: u64,
    source_hash: String,
    time_label: String,
    distance: u64,
    reason: &str,
    job: Option<String>,
) -> NewGist {
    let text = format!(
        "I couldn't write an account of events #{from_event_id}..#{through_event_id} \
         ({reason}). I don't have the detail of this stretch in memory — the archive \
         has all of it (archive_search with from_event {from_event_id} and through_event \
         {through_event_id}, then archive_get). Treat this as a gap I know about, not as \
         a stretch where nothing happened.",
    );
    let tokens = estimate_tokens(&text);
    NewGist {
        level,
        from_event_id,
        through_event_id,
        source_hash,
        tokens,
        time_label,
        written_by: "runtime".to_string(),
        text,
        mechanical: true,
        distance,
        job,
    }
}

/// An event id as sqlite stores it. Clamped rather than cast: `u64::MAX as i64`
/// is -1, which would turn "everything up to the end" into "nothing".
fn sql_id(id: u64) -> i64 {
    id.min(i64::MAX as u64) as i64
}

pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(BYTES_PER_TOKEN)
}

/// How far back something is, in words, from how much context has passed since.
///
/// Distance, not the clock: a quiet hour costs nothing and an hour of heavy tool
/// output ages memory fast. That made it wrong as a label: "earlier today" read
/// as a time, so after a quiet night yesterday's work was "a little earlier".
/// What the agent sees is `dated`; this is only the stored fallback when the archive
/// can't be read at writing time.
pub fn distance_label(back: u64) -> &'static str {
    match back {
        0..=50_000 => "just now",
        50_001..=150_000 => "a little earlier",
        150_001..=400_000 => "earlier today",
        _ => "a while back",
    }
}

/// When events `from`..=`through` happened, from the archive's own timestamps
/// (`Sources::times`): "Wed 21 Oct 2026 14:05–16:40". None unless both ends
/// are known.
pub fn dated(from: u64, through: u64, times: &HashMap<u64, DateTime<Local>>) -> Option<String> {
    Some(crate::clock::range(times.get(&from)?, times.get(&through)?))
}

/// Rough size of a message as it will be sent, for planning only.
pub fn estimate_message_tokens(message: &Message) -> usize {
    let mut bytes = message.role.len() + message.content.len();
    if let Some(reasoning) = &message.reasoning_content {
        bytes += reasoning.len();
    }
    if let Some(calls) = &message.tool_calls {
        for call in calls {
            bytes += call.function.name.len() + call.function.arguments.len() + 16;
        }
    }
    bytes.div_ceil(BYTES_PER_TOKEN)
}

// ------------------------------------------------------------------ fold planning

/// One episode's worth of live turns, as message indices into the live slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Episode {
    /// Half-open range over the live slice.
    pub start: usize,
    pub end: usize,
    pub est_tokens: usize,
}

/// What a fold will do: which live messages leave the context, cut into episodes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FoldPlan {
    pub episodes: Vec<Episode>,
    /// Messages folded in total — always an episode boundary, never mid-turn.
    pub messages_folded: usize,
    pub est_tokens_removed: usize,
}

impl FoldPlan {
    pub fn is_empty(&self) -> bool {
        self.messages_folded == 0
    }
}

/// Cut points where the context may legally be divided.
///
/// Two things must never happen: an assistant's tool call separated from its
/// results (the model would see a call it never got an answer to), and a tool
/// result left without the call that produced it. Both are the same rule from
/// different sides — a cut is legal only where no tool call is outstanding and
/// the next message isn't a tool result.
///
/// Within a *completed* turn that is the turn boundary. Inside a running turn,
/// each finished tool-call-and-results group is also a legal cut, which is what
/// stops one very long turn from pinning the whole context.
pub fn fold_boundaries(live: &[Message]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut outstanding: HashSet<String> = HashSet::new();

    for (index, message) in live.iter().enumerate() {
        match message.role.as_str() {
            "assistant" => {
                if let Some(calls) = &message.tool_calls {
                    for call in calls {
                        outstanding.insert(call.id.clone());
                    }
                }
            }
            "tool" => {
                if let Some(id) = &message.tool_call_id {
                    outstanding.remove(id);
                }
            }
            _ => {}
        }

        let cut = index + 1;
        let next_is_tool_result =
            live.get(cut).is_some_and(|next| next.role == "tool");

        if outstanding.is_empty() && !next_is_tool_result {
            boundaries.push(cut);
        }
    }

    boundaries
}

/// Whether a cut leaves a completed turn behind it: an assistant message that
/// said something rather than one that only called tools.
fn ends_completed_turn(live: &[Message], cut: usize) -> bool {
    cut.checked_sub(1)
        .and_then(|index| live.get(index))
        .is_some_and(|message| {
            message.role == "assistant"
                && message.tool_calls.as_ref().is_none_or(|calls| calls.is_empty())
        })
}

/// Plan a fold that removes at least `target_tokens` from the live layer, cut
/// into episodes of about `GIST_EPISODE_TOKENS` each.
///
/// A fold covers ~145k of turns, so this writes one gist per episode rather than
/// one enormous gist. One very large tool result can make an episode bigger than
/// the target; that's fine — it still folds as a unit, because splitting it would
/// mean splitting a call from its result.
pub fn plan_fold(live: &[Message], target_tokens: usize) -> FoldPlan {
    let limit = fold_limit(live);
    let boundaries: Vec<usize> = fold_boundaries(live).into_iter().filter(|cut| *cut <= limit).collect();
    if boundaries.is_empty() || target_tokens == 0 {
        return FoldPlan::default();
    }

    let mut plan = FoldPlan::default();
    let mut episode_start = 0usize;
    let mut episode_tokens = 0usize;
    let mut running = 0usize;
    let mut cursor = 0usize;
    // (cut, tokens before it), for a short final episode.
    let mut reached: Vec<(usize, usize)> = Vec::with_capacity(boundaries.len());

    for &cut in &boundaries {
        // Everything between the last cut we considered and this one.
        for message in &live[cursor..cut] {
            let tokens = estimate_message_tokens(message);
            episode_tokens += tokens;
            running += tokens;
        }
        cursor = cut;
        reached.push((cut, running));

        // Close an episode once it's big enough, but keep the last boundary of a
        // completed turn where one is available: a gist that ends mid-turn reads
        // as an interruption rather than an account of something.
        let big_enough = episode_tokens >= GIST_EPISODE_TOKENS;
        let finished_turn = ends_completed_turn(live, cut);

        if big_enough && (finished_turn || episode_tokens >= GIST_EPISODE_TOKENS * 2) {
            plan.episodes.push(Episode { start: episode_start, end: cut, est_tokens: episode_tokens });
            plan.messages_folded = cut;
            plan.est_tokens_removed = running;
            episode_start = cut;
            episode_tokens = 0;

            if running >= target_tokens {
                return plan;
            }
        }
    }

    // Target not reached on episode-sized cuts alone. End on a short final
    // episode rather than folding too little: at the first legal cut that
    // reaches the target, or the last one there is.
    if plan.est_tokens_removed < target_tokens && episode_tokens > 0 {
        let end = reached
            .iter()
            .find(|(cut, before)| *cut > episode_start && *before >= target_tokens)
            .or_else(|| reached.last())
            .copied();
        if let Some((cut, before)) = end.filter(|(cut, _)| *cut > episode_start) {
            plan.episodes.push(Episode {
                start: episode_start,
                end: cut,
                est_tokens: before - plan.est_tokens_removed,
            });
            plan.messages_folded = cut;
            plan.est_tokens_removed = before;
        }
    }

    // A fold this small isn't worth its re-prefill. Returning nothing is the
    // honest answer: the context is one nearly-indivisible unit, and the caller
    // has better options (wait for the running turn to finish, or escalate) than
    // paying for a fold that frees nothing.
    if plan.est_tokens_removed < MIN_USEFUL_FOLD_TOKENS.min(target_tokens) {
        return FoldPlan::default();
    }

    plan
}

/// The furthest a fold may cut: no later than the latest model output, and
/// leaving at least `KEEP_RECENT_TOKENS` of the live layer after the cut.
/// Zero when the live layer is too short to fold anything.
fn fold_limit(live: &[Message]) -> usize {
    let latest_output = live.iter().rposition(|message| message.role == "assistant").unwrap_or(live.len());
    let mut kept = 0usize;
    let mut limit = live.len();
    while limit > 0 && (kept < KEEP_RECENT_TOKENS || limit > latest_output) {
        limit -= 1;
        kept += estimate_message_tokens(&live[limit]);
    }
    limit
}

// ------------------------------------------------------------------ provenance

/// The archive event each live message is anchored to, where it carries one.
///
/// Only tool results carry an ID (the trailer agent.rs appends, which masking
/// keeps); every other message sits between its neighbours' anchors.
fn anchors(live: &[Message]) -> Vec<Option<u64>> {
    live.iter()
        .map(|message| {
            if message.role == "tool" {
                crate::observation_mask::event_id(&message.content)
            } else {
                None
            }
        })
        .collect()
}

/// The highest archive anchor among `messages`, if any carries one.
pub fn highest_anchor(messages: &[Message]) -> Option<u64> {
    anchors(messages).into_iter().flatten().max()
}

/// The archive range each planned episode covers.
///
/// Messages don't carry event IDs, but tool results do, and the archive is
/// append-only, so those anchors bound everything around them: an episode starts
/// after the last anchor before it and ends before the first anchor after it.
/// That is over-broad by a message or two at each edge (an operator message
/// archived just before the next episode's first tool result, say).
///
/// One case can come out too narrow: a subagent result is archived when it
/// arrives, mid-call, but only enters the context after that call's tool
/// results, so its event can sit just below the range of the episode holding
/// it. It is still inside the previous episode's range, and the stub and the
/// gist both point at the archive, so a lookup one range back finds it.
///
/// `floor` is the highest event known to precede every live message: the last
/// anchor an earlier fold removed, or the checkpoint the context was rebuilt
/// from. `latest` is the archive's newest event, the bound for an episode with
/// no anchor anywhere after it.
pub fn episode_ranges(
    live: &[Message],
    plan: &FoldPlan,
    floor: u64,
    latest: u64,
) -> Vec<(u64, u64)> {
    let anchors = anchors(live);
    plan.episodes
        .iter()
        .map(|episode| {
            let inside = || anchors[episode.start..episode.end].iter().flatten().copied();
            let before = anchors[..episode.start].iter().flatten().copied().max().unwrap_or(0);
            let after = anchors[episode.end..].iter().flatten().copied().min();

            let mut from = before.max(floor).saturating_add(1);
            if let Some(first) = inside().min() {
                from = from.min(first);
            }
            let mut through = after.map_or(latest, |next| next.saturating_sub(1));
            if let Some(last) = inside().max() {
                through = through.max(last);
            }
            (from, through.max(from))
        })
        .collect()
}

/// Where each planned episode ends on the distance odometer.
///
/// `now_distance` is the odometer at the end of `live`; an episode ends that much
/// less whatever live context came after it. Estimated, like all planning, which
/// is fine for labels this coarse.
pub fn episode_positions(live: &[Message], plan: &FoldPlan, now_distance: u64) -> Vec<u64> {
    plan.episodes
        .iter()
        .map(|episode| {
            let after: usize = live[episode.end..].iter().map(estimate_message_tokens).sum();
            now_distance.saturating_sub(after as u64)
        })
        .collect()
}

// ------------------------------------------------------------------ not settled

/// Where an item stands. Only `Open` ones render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemState {
    Open,
    /// Folded into a pointer for space. Still unsettled: handed to the world
    /// model at deep sleep like any open item.
    SetAside,
    Settled,
    Moot,
}

impl ItemState {
    fn label(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::SetAside => "set_aside",
            Self::Settled => "settled",
            Self::Moot => "moot",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "set_aside" => Self::SetAside,
            "settled" => Self::Settled,
            "moot" => Self::Moot,
            _ => Self::Open,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    /// One open question, untested conclusion or unchecked report.
    Item,
    /// Stands in for older items set aside when the list ran over budget.
    Pointer,
}

/// Something a stretch left unsettled, held apart from the gists so no merge can
/// drop it or quietly turn it into a conclusion.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenItem {
    pub id: i64,
    pub kind: ItemKind,
    pub state: ItemState,
    pub from_event_id: u64,
    pub through_event_id: u64,
    pub text: String,
    pub tokens: usize,
    /// How many items a pointer stands for; 1 for an item.
    pub members: usize,
    /// What settled it, or why it went moot.
    pub resolution: String,
    /// The world-model record deep sleep moved it to, once it has.
    pub handed_off_to: Option<String>,
    pub distance: u64,
    /// The job that was open when this was raised. As on a gist: a pointer
    /// never stands for items from more than one job, and a close takes these
    /// with it.
    pub job: Option<String>,
}

impl OpenItem {
    pub fn label(&self) -> String {
        format!("U{}", self.id)
    }

    fn render(&self, times: &HashMap<u64, DateTime<Local>>) -> String {
        let when = dated(self.from_event_id, self.through_event_id, times)
            .map(|when| format!("{when} | "))
            .unwrap_or_default();
        format!(
            "- {} [{when}events #{}..#{}] {}",
            self.label(),
            self.from_event_id,
            self.through_event_id,
            self.text.trim(),
        )
    }
}

/// An item as a fold is about to store it.
#[derive(Debug, Clone, PartialEq)]
pub struct NewOpenItem {
    pub from_event_id: u64,
    pub through_event_id: u64,
    pub text: String,
    pub distance: u64,
    pub job: Option<String>,
}

/// A collapse has to free real space, or the loop that enforces the budget could
/// trade two short items for one longer pointer and never converge. Each one
/// takes at least this many tokens of the oldest entries.
const COLLAPSE_AT_LEAST_TOKENS: usize = OPEN_ITEMS_BUDGET_TOKENS / 4;

/// Ceiling on a pointer's text, so a pointer is always much smaller than what it
/// replaced.
const POINTER_TEXT_CHARS: usize = 400;

/// How much of each set-aside item a pointer quotes.
const HEADLINE_CHARS: usize = 60;

fn pointer_text(members: &[String]) -> String {
    let mut out = format!(
        "{} older open items set aside for space. Still unsettled, none of them answered: ",
        members.len()
    );
    let mut shown = 0;
    for member in members {
        let headline = truncate(member.trim(), HEADLINE_CHARS);
        if out.chars().count() + headline.chars().count() + 2 > POINTER_TEXT_CHARS {
            break;
        }
        if shown > 0 {
            out.push_str("; ");
        }
        out.push_str(&headline);
        shown += 1;
    }
    if shown < members.len() {
        out.push_str(&format!("; and {} more", members.len() - shown));
    }
    out
}

// ------------------------------------------------------------------ store

pub struct MemoryLayers {
    path: PathBuf,
}

impl MemoryLayers {
    pub fn beside_archive(archive: &Path) -> Self {
        Self { path: archive.with_file_name("memory_layers.sqlite3") }
    }

    fn connect(&self) -> Result<Connection> {
        let db = crate::db::open(&self.path)?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS gists(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                from_event_id INTEGER NOT NULL,
                through_event_id INTEGER NOT NULL,
                source_hash TEXT NOT NULL,
                tokens INTEGER NOT NULL,
                time_label TEXT NOT NULL,
                written_by TEXT NOT NULL,
                text TEXT NOT NULL,
                mechanical INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                distance INTEGER NOT NULL);
             CREATE INDEX IF NOT EXISTS gists_level_order
                ON gists(level, from_event_id, id);
             CREATE TABLE IF NOT EXISTS open_items(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                state TEXT NOT NULL,
                from_event_id INTEGER NOT NULL,
                through_event_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                tokens INTEGER NOT NULL,
                members INTEGER NOT NULL,
                pointer_id INTEGER,
                resolution TEXT NOT NULL,
                handed_off_to TEXT,
                created_at TEXT NOT NULL,
                distance INTEGER NOT NULL);
             CREATE INDEX IF NOT EXISTS open_items_order
                ON open_items(state, from_event_id, id);
             CREATE TABLE IF NOT EXISTS layer_state(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL);",
        )?;
        Self::add_job_column(&db);
        Ok(db)
    }

    /// Add the `job` column to a database made before it existed.
    ///
    /// `CREATE TABLE IF NOT EXISTS` does nothing to a table already there, so
    /// a new column has to be added on its own. This crate has no migration
    /// machinery and no schema version to hang one on, so this runs on every
    /// connection and is written to be a no-op once the column exists: the
    /// error sqlite gives for a duplicate column is the ordinary case from the
    /// second connection onwards. A genuine failure leaves the column absent,
    /// which reads back as "no job" -- what every row written before this is.
    ///
    /// Both row readers take their columns by position, so the new column is
    /// appended to the end of each select list rather than put in order.
    fn add_job_column(db: &Connection) {
        for table in ["gists", "open_items"] {
            let _ = db.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN job TEXT"));
        }
    }

    fn row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Gist> {
        Ok(Gist {
            id: row.get(0)?,
            level: Level::parse(&row.get::<_, String>(1)?),
            from_event_id: row.get::<_, i64>(2)?.max(0) as u64,
            through_event_id: row.get::<_, i64>(3)?.max(0) as u64,
            source_hash: row.get(4)?,
            tokens: row.get::<_, i64>(5)?.max(0) as usize,
            time_label: row.get(6)?,
            written_by: row.get(7)?,
            text: row.get(8)?,
            mechanical: row.get::<_, i64>(9)? != 0,
            created_at: row.get(10)?,
            distance: row.get::<_, i64>(11)?.max(0) as u64,
            job: row.get(12)?,
        })
    }

    const SELECT: &'static str = "SELECT id,level,from_event_id,through_event_id,source_hash,\
        tokens,time_label,written_by,text,mechanical,created_at,distance,job FROM gists";

    /// Everything at one level, oldest first — which is also reading order.
    pub fn level(&self, level: Level) -> Result<Vec<Gist>> {
        let db = self.connect()?;
        let sql = format!("{} WHERE level=?1 ORDER BY from_event_id, id", Self::SELECT);
        let mut statement = db.prepare(&sql)?;
        let rows = statement.query_map(params![level.label()], Self::row)?;
        let mut gists = Vec::new();
        for gist in rows {
            gists.push(gist?);
        }
        Ok(gists)
    }

    pub fn append(&self, gist: &NewGist) -> Result<Gist> {
        let db = self.connect()?;
        let created_at = Local::now().to_rfc3339();
        db.execute(
            "INSERT INTO gists(level,from_event_id,through_event_id,source_hash,tokens,\
             time_label,written_by,text,mechanical,created_at,distance,job) \
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
            params![
                gist.level.label(),
                gist.from_event_id as i64,
                gist.through_event_id as i64,
                gist.source_hash,
                gist.tokens as i64,
                gist.time_label,
                gist.written_by,
                gist.text,
                i64::from(gist.mechanical),
                created_at,
                gist.distance as i64,
                gist.job,
            ],
        )?;
        let id = db.last_insert_rowid();
        let sql = format!("{} WHERE id=?1", Self::SELECT);
        Ok(db.query_row(&sql, params![id], Self::row)?)
    }

    pub fn remove(&self, ids: &[i64]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let mut db = self.connect()?;
        let mut removed = 0;
        let transaction = db.transaction()?;
        for id in ids {
            removed += transaction.execute("DELETE FROM gists WHERE id=?1", params![id])?;
        }
        transaction.commit()?;
        Ok(removed)
    }

    /// Total tokens held at one level.
    pub fn tokens_at(&self, level: Level) -> Result<usize> {
        let db = self.connect()?;
        let total: i64 = db
            .query_row(
                "SELECT COALESCE(SUM(tokens),0) FROM gists WHERE level=?1",
                params![level.label()],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or(0);
        Ok(total.max(0) as usize)
    }

    /// Delete every gist and open item whose own text holds one of
    /// `fragments`. Returns (gists, items).
    ///
    /// For a redaction of part of an event: what matters is whether a row
    /// actually quotes what was struck, not whether it covers the event it was
    /// struck from. A gist of events #100..#140 is a summary, and deleting it
    /// because #120 had a key taken out of it would cost the memory of forty
    /// events to remove something that was almost certainly never in it. The
    /// test is the text.
    ///
    /// Matched in SQL rather than by reading every row out, so the fragments
    /// are compared inside the database and never collected anywhere.
    pub fn purge_holding(&self, fragments: &[String]) -> Result<(usize, usize)> {
        if fragments.is_empty() {
            return Ok((0, 0));
        }
        let mut db = self.connect()?;
        let transaction = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut gists = 0;
        let mut items = 0;
        for fragment in fragments {
            // `instr` rather than LIKE: no escaping, and no pattern characters
            // in a fragment to go wrong.
            gists += transaction.execute("DELETE FROM gists WHERE instr(text, ?1) > 0", params![fragment])?;
            items +=
                transaction.execute("DELETE FROM open_items WHERE instr(text, ?1) > 0", params![fragment])?;
        }
        let pointers: Vec<i64> = {
            let mut statement = transaction.prepare("SELECT id FROM open_items WHERE kind='pointer'")?;
            let rows = statement.query_map([], |row| row.get::<_, i64>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        for pointer in pointers {
            Self::refresh_pointer(&transaction, pointer)?;
        }
        transaction.commit()?;
        if gists + items > 0 {
            crate::hippocampus::compact(&db);
        }
        Ok((gists, items))
    }

    /// Everything a job left in the layers, gone: the rows it wrote and the
    /// rows overlapping the events it covered. Pointers are rewritten from
    /// whatever is left behind them. Returns (gists, items).
    ///
    /// Two conditions rather than one, because neither is enough on its own.
    /// The tag is exact for anything written while the job was open, and is the
    /// only way to tell a merged entry's provenance -- a merge carries its
    /// inputs' range forward, so ranges alone would spare an entry whose prose
    /// came from the job. The range catches what the tag cannot: a row written
    /// before jobs existed, one a prepared fold stored with the tag it had at
    /// preparation time, and anything a future path forgets to stamp. A gist's
    /// range is a bounding interval, deliberately over-broad at the edges, so
    /// overlap can take a neighbour's row with it. That is the direction to err
    /// in: a little unrelated memory lost, rather than part of a job kept.
    ///
    /// One transaction, as `clear_through` is one, and for the same reason: a
    /// pointer quotes its members, so deleting members and rewriting the
    /// pointer are the same change.
    pub fn purge_job(&self, job: &str, ranges: &[(u64, u64)]) -> Result<(usize, usize)> {
        let mut db = self.connect()?;
        let transaction = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut gists = 0;
        let mut items = 0;
        gists += transaction.execute("DELETE FROM gists WHERE job=?1", params![job])?;
        items += transaction.execute("DELETE FROM open_items WHERE job=?1", params![job])?;
        for (from, through) in ranges {
            let (from, through) = (sql_id(*from), sql_id(*through));
            // Overlap, not containment: a row covering part of the span holds
            // part of the job.
            gists += transaction.execute(
                "DELETE FROM gists WHERE through_event_id >= ?1 AND from_event_id <= ?2",
                params![from, through],
            )?;
            items += transaction.execute(
                "DELETE FROM open_items WHERE kind='item' AND through_event_id >= ?1 AND from_event_id <= ?2",
                params![from, through],
            )?;
        }
        // A set-aside item is reachable only through its pointer, so one whose
        // pointer was the job's would be left in the table invisible to every
        // reader and to `refresh_pointer`, which walks surviving pointers only.
        items += transaction.execute(
            "DELETE FROM open_items WHERE state='set_aside' AND pointer_id IS NOT NULL \
             AND pointer_id NOT IN (SELECT id FROM open_items WHERE kind='pointer')",
            [],
        )?;
        let pointers: Vec<i64> = {
            let mut statement = transaction.prepare("SELECT id FROM open_items WHERE kind='pointer'")?;
            let rows = statement.query_map([], |row| row.get::<_, i64>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        for pointer in pointers {
            Self::refresh_pointer(&transaction, pointer)?;
        }
        transaction.commit()?;
        // The rows are gone from the tables; this takes their bytes out of the
        // file rather than leaving them in pages marked free.
        crate::hippocampus::compact(&db);
        Ok((gists, items))
    }

    /// Everything deep sleep has now covered leaves the layers: the checkpoint is
    /// the account of that span from here on, and keeping both would be the same
    /// day remembered twice at two resolutions. Returns the gists removed.
    ///
    /// NOT SETTLED items in that span leave only once the deep-sleep handoff
    /// has dealt with them (`crate::memory::fold::hand_off_unsettled` marks each one it
    /// wrote, retired or found nothing to do for). One whose handoff failed
    /// stays here, open or closed: an item outliving its gist is the safe
    /// direction. Pointers are then rewritten from what is left behind them.
    ///
    /// All of it is one transaction. A pointer quotes its members, so deleting
    /// members and rewriting the pointer are the same change: committed apart,
    /// a failure between them leaves a pointer quoting items that are gone,
    /// and the memory message renders it. On error nothing has been retired.
    pub fn clear_through(&self, through_event_id: u64) -> Result<usize> {
        let mut db = self.connect()?;
        let transaction = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let through = sql_id(through_event_id);
        let cleared =
            transaction.execute("DELETE FROM gists WHERE through_event_id <= ?1", params![through])?;
        transaction.execute(
            "DELETE FROM open_items WHERE kind='item' AND through_event_id <= ?1 \
             AND handed_off_to IS NOT NULL",
            params![through],
        )?;
        let pointers: Vec<i64> = {
            let mut statement = transaction.prepare("SELECT id FROM open_items WHERE kind='pointer'")?;
            let rows = statement.query_map([], |row| row.get::<_, i64>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        for pointer in pointers {
            Self::refresh_pointer(&transaction, pointer)?;
        }
        transaction.commit()?;
        Ok(cleared)
    }

    /// The highest archive event known to precede every live message: the last
    /// anchor a fold removed. The next fold's first episode starts after it.
    ///
    /// Before the first fold that records one (a store written by an earlier
    /// version), the newest gist's end stands in. That bound can be a little
    /// high, but `episode_ranges` never lets a range start after its own first
    /// anchor, and it keeps new gists ordered after the old ones.
    pub fn fold_floor(&self) -> Result<u64> {
        if let Some(floor) = self.stored_fold_floor()? {
            return Ok(floor);
        }
        let db = self.connect()?;
        let newest: i64 =
            db.query_row("SELECT COALESCE(MAX(through_event_id),0) FROM gists", [], |row| row.get(0))?;
        Ok(newest.max(0) as u64)
    }

    fn stored_fold_floor(&self) -> Result<Option<u64>> {
        let db = self.connect()?;
        let value: Option<String> = db
            .query_row("SELECT value FROM layer_state WHERE key='fold_floor'", [], |row| row.get(0))
            .optional()?;
        Ok(value.and_then(|value| value.parse().ok()))
    }

    /// Record the fold floor; once recorded it never moves down. The first
    /// value recorded replaces the fallback in `fold_floor` outright, since
    /// a newest gist's end is only a stand-in for a store that has none.
    pub fn raise_fold_floor(&self, to: u64) -> Result<()> {
        if self.stored_fold_floor()?.is_some_and(|stored| to <= stored) {
            return Ok(());
        }
        let db = self.connect()?;
        db.execute(
            "INSERT INTO layer_state(key,value) VALUES('fold_floor',?1) \
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![to.to_string()],
        )?;
        Ok(())
    }

    /// Mechanical folds since a given point, for deep sleep to review.
    pub fn mechanical_since(&self, from_event_id: u64) -> Result<Vec<Gist>> {
        let db = self.connect()?;
        let sql = format!(
            "{} WHERE mechanical=1 AND through_event_id >= ?1 ORDER BY from_event_id, id",
            Self::SELECT
        );
        let mut statement = db.prepare(&sql)?;
        let rows = statement.query_map(params![from_event_id as i64], Self::row)?;
        let mut gists = Vec::new();
        for gist in rows {
            gists.push(gist?);
        }
        Ok(gists)
    }

    // ---- not settled ----

    const ITEM_SELECT: &'static str = "SELECT id,kind,state,from_event_id,through_event_id,text,\
        tokens,members,resolution,handed_off_to,distance,job FROM open_items";

    fn item_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<OpenItem> {
        Ok(OpenItem {
            id: row.get(0)?,
            kind: if row.get::<_, String>(1)? == "pointer" { ItemKind::Pointer } else { ItemKind::Item },
            state: ItemState::parse(&row.get::<_, String>(2)?),
            from_event_id: row.get::<_, i64>(3)?.max(0) as u64,
            through_event_id: row.get::<_, i64>(4)?.max(0) as u64,
            text: row.get(5)?,
            tokens: row.get::<_, i64>(6)?.max(0) as usize,
            members: row.get::<_, i64>(7)?.max(1) as usize,
            resolution: row.get(8)?,
            handed_off_to: row.get(9)?,
            distance: row.get::<_, i64>(10)?.max(0) as u64,
            job: row.get(11)?,
        })
    }

    fn items_where(&self, condition: &str, args: impl rusqlite::Params) -> Result<Vec<OpenItem>> {
        let db = self.connect()?;
        let sql = format!("{} WHERE {condition} ORDER BY from_event_id, id", Self::ITEM_SELECT);
        let mut statement = db.prepare(&sql)?;
        let rows = statement.query_map(args, Self::item_row)?;
        let mut items = Vec::new();
        for item in rows {
            items.push(item?);
        }
        Ok(items)
    }

    /// What the memory message lists as not settled, oldest first: open items
    /// and the pointers standing in for set-aside ones.
    pub fn open_items(&self) -> Result<Vec<OpenItem>> {
        self.items_where("state='open'", params![])
    }

    /// Every item still unsettled, set-aside ones included, oldest first. This
    /// is what a gist writer is shown: a pointer saves room in the memory
    /// message, but settling has to name one item at a time.
    pub fn unsettled_items(&self) -> Result<Vec<OpenItem>> {
        self.items_where("kind='item' AND state IN ('open','set_aside')", params![])
    }

    pub fn item(&self, id: i64) -> Result<Option<OpenItem>> {
        Ok(self.items_where("id=?1", params![id])?.into_iter().next())
    }

    /// Tokens the open entries take in the memory message.
    pub fn item_tokens(&self) -> Result<usize> {
        let db = self.connect()?;
        let total: i64 = db.query_row(
            "SELECT COALESCE(SUM(tokens),0) FROM open_items WHERE state='open'",
            [],
            |row| row.get(0),
        )?;
        Ok(total.max(0) as usize)
    }

    /// The id the next stored item will get.
    ///
    /// Ids are issued here rather than left to sqlite so a fold written ahead of
    /// time can number its new items before storing them: the gist for a later
    /// episode refers to an item an earlier episode opened, and the number it
    /// used has to be the number the item lands with. AUTOINCREMENT keeps a
    /// number from ever being reused after deep sleep clears old items.
    pub fn next_item_id(&self) -> Result<i64> {
        let db = self.connect()?;
        let next: i64 = db.query_row(
            "SELECT MAX(COALESCE((SELECT seq FROM sqlite_sequence WHERE name='open_items'),0), \
                        COALESCE((SELECT MAX(id) FROM open_items),0)) + 1",
            [],
            |row| row.get(0),
        )?;
        Ok(next)
    }

    /// Store an item under the id `next_item_id` gave for it.
    pub fn add_item(&self, id: i64, item: &NewOpenItem) -> Result<OpenItem> {
        let db = self.connect()?;
        db.execute(
            "INSERT INTO open_items(id,kind,state,from_event_id,through_event_id,text,tokens,\
             members,pointer_id,resolution,handed_off_to,created_at,distance,job) \
             VALUES(?1,'item','open',?2,?3,?4,?5,1,NULL,'',NULL,?6,?7,?8)",
            params![
                id,
                sql_id(item.from_event_id),
                sql_id(item.through_event_id),
                item.text,
                estimate_tokens(&item.text) as i64,
                Local::now().to_rfc3339(),
                item.distance as i64,
                item.job,
            ],
        )?;
        self.item(id)?.ok_or_else(|| "stored item vanished".into())
    }

    /// Close one unsettled item, open or set aside, as settled or moot. Returns
    /// false when there is no such unsettled item.
    ///
    /// Pointers can't be closed: one piece of evidence settles one question,
    /// and a pointer stands for several. Closing a set-aside item rewrites its
    /// pointer to match what is left behind it, in the same transaction: on
    /// error the item is still unsettled and its pointer still counts it.
    pub fn resolve_item(&self, id: i64, state: ItemState, resolution: &str) -> Result<bool> {
        if !matches!(state, ItemState::Settled | ItemState::Moot) {
            return Err("an item can only be resolved as settled or moot".into());
        }
        let mut db = self.connect()?;
        let transaction = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current: Option<(String, String, Option<i64>)> = transaction
            .query_row("SELECT kind, state, pointer_id FROM open_items WHERE id=?1", params![id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .optional()?;
        let Some((kind, was, pointer)) = current else {
            return Ok(false);
        };
        let was = ItemState::parse(&was);
        if kind != "item" || !matches!(was, ItemState::Open | ItemState::SetAside) {
            return Ok(false);
        }
        // Closing clears any earlier handoff: the record it pointed at still
        // says unresolved, and only the next handoff's retire marks it done.
        transaction.execute(
            "UPDATE open_items SET state=?2, resolution=?3, handed_off_to=NULL WHERE id=?1",
            params![id, state.label(), resolution],
        )?;
        if let Some(pointer) = pointer.filter(|_| was == ItemState::SetAside) {
            Self::refresh_pointer(&transaction, pointer)?;
        }
        transaction.commit()?;
        Ok(true)
    }

    /// Rewrite a pointer from the items still set aside behind it (text, count
    /// and range), or remove it once none are. It runs inside the caller's
    /// transaction: whatever changed the members commits with the rewrite.
    fn refresh_pointer(transaction: &rusqlite::Transaction<'_>, pointer: i64) -> Result<()> {
        let members: Vec<(String, i64, i64)> = {
            let mut statement = transaction.prepare(
                "SELECT text, from_event_id, through_event_id FROM open_items \
                 WHERE pointer_id=?1 AND state='set_aside' ORDER BY from_event_id, id",
            )?;
            let rows = statement.query_map(params![pointer], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?))
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        if members.is_empty() {
            transaction.execute("DELETE FROM open_items WHERE id=?1 AND kind='pointer'", params![pointer])?;
        } else {
            let texts: Vec<String> = members.iter().map(|(text, _, _)| text.clone()).collect();
            let text = pointer_text(&texts);
            let from = members.iter().map(|(_, from, _)| *from).min().unwrap_or(0);
            let through = members.iter().map(|(_, _, through)| *through).max().unwrap_or(0);
            transaction.execute(
                "UPDATE open_items SET text=?2, tokens=?3, members=?4, from_event_id=?5, \
                 through_event_id=?6 WHERE id=?1 AND kind='pointer'",
                params![pointer, text, estimate_tokens(&text) as i64, members.len() as i64, from, through],
            )?;
        }
        Ok(())
    }

    /// Hold the NOT SETTLED list to its budget by setting the oldest entries
    /// aside behind a pointer. Nothing is dropped: set-aside items stay
    /// unsettled, the pointer says how many there are and quotes them, and deep
    /// sleep hands each one to the world model. Returns how many collapses ran.
    pub fn enforce_item_budget(&self) -> Result<usize> {
        let mut collapses = 0;
        while self.item_tokens()? > OPEN_ITEMS_BUDGET_TOKENS {
            let open = self.open_items()?;
            // A pointer quotes the entries it stands for, so it may only take
            // entries of one job -- the same rule as a gist merge, for the same
            // reason. The run stops at the first entry carrying another tag.
            let job = open.first().and_then(|entry| entry.job.clone());
            let mut take = 0;
            let mut freed = 0;
            for entry in &open {
                if entry.job != job {
                    break;
                }
                take += 1;
                freed += entry.tokens;
                if take >= 2 && freed >= COLLAPSE_AT_LEAST_TOKENS {
                    break;
                }
            }
            if take < 2 || freed < COLLAPSE_AT_LEAST_TOKENS.min(OPEN_ITEMS_BUDGET_TOKENS) {
                break;
            }
            self.collapse(&open[..take])?;
            collapses += 1;
            if collapses > 64 {
                break;
            }
        }
        Ok(collapses)
    }

    fn collapse(&self, entries: &[OpenItem]) -> Result<()> {
        let pointer = self.next_item_id()?;
        let mut db = self.connect()?;
        let transaction = db.transaction()?;
        for entry in entries {
            match entry.kind {
                ItemKind::Item => {
                    transaction.execute(
                        "UPDATE open_items SET state='set_aside', pointer_id=?2 WHERE id=?1",
                        params![entry.id, pointer],
                    )?;
                }
                // A pointer being set aside hands its members to the new one.
                ItemKind::Pointer => {
                    transaction.execute(
                        "UPDATE open_items SET pointer_id=?2 WHERE pointer_id=?1",
                        params![entry.id, pointer],
                    )?;
                    transaction.execute("DELETE FROM open_items WHERE id=?1", params![entry.id])?;
                }
            }
        }

        let members: Vec<String> = {
            let mut statement = transaction.prepare(
                "SELECT text FROM open_items WHERE pointer_id=?1 AND state='set_aside' \
                 ORDER BY from_event_id, id",
            )?;
            let rows = statement.query_map(params![pointer], |row| row.get::<_, String>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        let text = pointer_text(&members);
        let from = entries.iter().map(|entry| entry.from_event_id).min().unwrap_or(0);
        let through = entries.iter().map(|entry| entry.through_event_id).max().unwrap_or(0);
        let distance = entries.first().map_or(0, |entry| entry.distance);

        transaction.execute(
            "INSERT INTO open_items(id,kind,state,from_event_id,through_event_id,text,tokens,\
             members,pointer_id,resolution,handed_off_to,created_at,distance,job) \
             VALUES(?1,'pointer','open',?2,?3,?4,?5,?6,NULL,'',NULL,?7,?8,?9)",
            params![
                pointer,
                sql_id(from),
                sql_id(through),
                text,
                estimate_tokens(&text) as i64,
                members.len().max(1) as i64,
                Local::now().to_rfc3339(),
                distance as i64,
                // Every entry shares one job: `enforce_item_budget` only ever
                // hands over entries that do.
                entries.first().and_then(|entry| entry.job.clone()),
            ],
        )?;
        transaction.commit()?;
        Ok(())
    }

    /// Items deep sleep should take over: unsettled (open or set aside), from a
    /// stretch it is covering, and not handed over already.
    pub fn unsettled_through(&self, through_event_id: u64) -> Result<Vec<OpenItem>> {
        self.items_where(
            "kind='item' AND state IN ('open','set_aside') AND handed_off_to IS NULL \
             AND through_event_id <= ?1",
            params![sql_id(through_event_id)],
        )
    }

    /// Items closed in working memory (settled or moot) from a stretch deep
    /// sleep is covering, that the handoff hasn't dealt with yet.
    pub fn closed_through(&self, through_event_id: u64) -> Result<Vec<OpenItem>> {
        self.items_where(
            "kind='item' AND state IN ('settled','moot') AND handed_off_to IS NULL \
             AND through_event_id <= ?1",
            params![sql_id(through_event_id)],
        )
    }

    pub fn mark_handed_off(&self, id: i64, record_id: &str) -> Result<()> {
        let db = self.connect()?;
        db.execute(
            "UPDATE open_items SET handed_off_to=?2 WHERE id=?1",
            params![id, record_id],
        )?;
        Ok(())
    }
}

// ------------------------------------------------------------------ merge cascade

/// Where merge text comes from. The live implementation makes a model call with
/// the merge prompt; tests use a deterministic stub.
pub trait Summarizer {
    /// Write a first-person account of one episode from its transcript.
    ///
    /// `open_items` lists what is currently not settled, one `- U<id>: text`
    /// line each (or nothing), so the gist can say which of them its stretch
    /// settled. The reply ends with NOT SETTLED / SETTLED / MOOT sections that
    /// `crate::memory::fold::parse_gist` reads.
    ///
    /// Errors are not fatal: the fold stores a mechanical stub in place of the
    /// gist, because a stretch the agent knows it has no account of is recoverable,
    /// and a fold that refuses to happen is not.
    fn gist(&self, transcript: &str, open_items: &str) -> Result<String>;

    /// Merge several entries into one at `level`, returning its text.
    ///
    /// Errors are not fatal: the cascade falls back to a mechanical stub, because
    /// a memory message that is one entry over budget is a much smaller problem
    /// than one that refuses to be built.
    fn merge(&self, level: Level, inputs: &[Gist]) -> Result<String>;

    /// Time label for a merged entry, given the entries it covers.
    fn time_label(&self, inputs: &[Gist]) -> String {
        inputs.first().map(|gist| gist.time_label.clone()).unwrap_or_else(|| "earlier".to_string())
    }

    fn model_name(&self) -> String {
        "unknown".to_string()
    }
}

/// What a cascade run did, for logging.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CascadeReport {
    pub episode_merges: usize,
    pub shape_merges: usize,
    pub mechanical_merges: usize,
    /// Merges whose inputs had tentative tags and whose output kept none.
    pub flattened_merges: usize,
    /// Event ids taken out of merges: ones none of their inputs cited, or that
    /// aren't tool results in the merged range.
    pub citations_removed: Vec<u64>,
    /// Merges whose range couldn't be read from the archive: stored without a
    /// source hash, keeping only citations their inputs made inside the range.
    pub unread: Vec<String>,
    pub episode_tokens: usize,
    pub shape_tokens: usize,
}

/// Tags that mark a claim as less than established. Matched lowercase.
const TENTATIVE_TAGS: [&str; 2] = ["[inferred", "[guess"];

/// Lines the fold writes under a gist's account to record what happened to
/// NOT SETTLED items. They quote items, tags and all, so they don't count as
/// the account's own claims.
pub const FOOTER_PREFIXES: [&str; 3] = ["Settled here: ", "No longer relevant: ", "Left open here: "];

// ------------------------------------------------------------------ citations

/// An archive for tests: the listed ids are its tool results, and each span's
/// source hash names the span.
#[cfg(test)]
pub(crate) struct FakeArchive(pub Vec<u64>);

#[cfg(test)]
impl Sources for FakeArchive {
    fn spans(&self, ranges: &[(u64, u64)]) -> Result<Vec<crate::hippocampus::SpanDigest>> {
        Ok(ranges
            .iter()
            .map(|(from, through)| crate::hippocampus::SpanDigest {
                source_hash: format!("hash-{from}-{through}"),
                tool_results: self.0.iter().copied().filter(|id| (*from..=*through).contains(id)).collect(),
            })
            .collect())
    }
}

/// An archive that can't be read.
#[cfg(test)]
pub(crate) struct UnreadableArchive;

#[cfg(test)]
impl Sources for UnreadableArchive {
    fn spans(&self, _ranges: &[(u64, u64)]) -> Result<Vec<crate::hippocampus::SpanDigest>> {
        Err("disk error".into())
    }
}

/// The archive as folding reads it: where a gist's source hash comes from,
/// what its `[observed #N]` tags are checked against, and when its stretch
/// happened.
pub trait Sources {
    /// For each inclusive range, its source hash and the tool results in it.
    fn spans(&self, ranges: &[(u64, u64)]) -> Result<Vec<crate::hippocampus::SpanDigest>>;

    /// When the events with these ids happened (host time). Ids it doesn't
    /// have are left out, and so is everything when it can't be read: memory
    /// then shows its ranges undated.
    fn times(&self, _ids: &[u64]) -> HashMap<u64, DateTime<Local>> {
        HashMap::new()
    }

    /// Keep the candidate facts a gist's claims make, for deep sleep to
    /// consolidate (`facts::stage`); how many were new. Nowhere to keep
    /// them by default.
    fn stage_facts(&self, _facts: &[crate::facts::Candidate]) -> usize {
        0
    }
}

impl Sources for crate::hippocampus::Hippocampus {
    fn stage_facts(&self, facts: &[crate::facts::Candidate]) -> usize {
        crate::facts::stage(&crate::world_model::path_for(self), facts).unwrap_or(0)
    }

    fn spans(&self, ranges: &[(u64, u64)]) -> Result<Vec<crate::hippocampus::SpanDigest>> {
        crate::hippocampus::Hippocampus::spans(self, ranges)
    }

    fn times(&self, ids: &[u64]) -> HashMap<u64, DateTime<Local>> {
        let mut ids = ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        self.get_events(&ids)
            .map(|events| {
                events
                    .into_iter()
                    .filter_map(|event| crate::clock::parse(&event.timestamp).map(|at| (event.id, at)))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Both ends of every range, for one `Sources::times` lookup.
pub fn range_ends<'a>(ranges: impl IntoIterator<Item = (u64, u64)> + 'a) -> Vec<u64> {
    ranges.into_iter().flat_map(|(from, through)| [from, through]).collect()
}

/// Every `[observed …]` tag in `text`: its byte range and the event ids it
/// names (`[observed #12]`, `[observed #12, #15]`). A tag ends at the first
/// `]`; one that meets a `[` or a line break first isn't closed, and isn't a
/// citation. Matched case-insensitively.
fn observed_tags(text: &str) -> Vec<(std::ops::Range<usize>, Vec<u64>)> {
    const OPEN: &str = "[observed";
    let lower = text.to_ascii_lowercase();
    let mut tags = Vec::new();
    let mut at = 0;
    while let Some(found) = lower[at..].find(OPEN) {
        let start = at + found;
        let inner = start + OPEN.len();
        match text[inner..].find([']', '[', '\n']) {
            Some(offset) if text[inner + offset..].starts_with(']') => {
                let end = inner + offset + 1;
                let ids = text[inner..end - 1]
                    .split('#')
                    .skip(1)
                    .filter_map(|part| part.chars().take_while(char::is_ascii_digit).collect::<String>().parse().ok())
                    .collect();
                tags.push((start..end, ids));
                at = end;
            }
            _ => at = inner,
        }
    }
    tags
}

/// Archive events cited as `[observed #N]` (or `[observed #N, #M]`), in order.
pub(crate) fn cited_observations(text: &str) -> Vec<u64> {
    observed_tags(text).into_iter().flat_map(|(_, ids)| ids).collect()
}

/// `text` with every cited event that fails `valid` taken out of its tag, and
/// the ids taken out. A tag whose ids all pass is left exactly as written; one
/// with any that fail keeps only those that pass (`[observed #12]`), or becomes
/// `[observed]` when none do: the claim was still reported as seen, but no
/// longer points at an event that doesn't back it.
pub(crate) fn keep_citations(text: &str, valid: &dyn Fn(u64) -> bool) -> (String, Vec<u64>) {
    let mut out = String::with_capacity(text.len());
    let mut removed = Vec::new();
    let mut at = 0;
    for (range, ids) in observed_tags(text) {
        let (kept, dropped): (Vec<u64>, Vec<u64>) = ids.into_iter().partition(|id| valid(*id));
        if dropped.is_empty() {
            continue;
        }
        out.push_str(&text[at..range.start]);
        out.push_str(&text[range.start..range.start + "[observed".len()]);
        if !kept.is_empty() {
            out.push(' ');
            out.push_str(&kept.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(", "));
        }
        out.push(']');
        removed.extend(dropped);
        at = range.end;
    }
    out.push_str(&text[at..]);
    (out, removed)
}

fn tentative_tags(text: &str) -> usize {
    text.lines()
        .filter(|line| !FOOTER_PREFIXES.iter().any(|prefix| line.starts_with(prefix)))
        .map(|line| {
            let lower = line.to_lowercase();
            TENTATIVE_TAGS.iter().map(|tag| lower.matches(tag).count()).sum::<usize>()
        })
        .sum()
}

/// Appended to a merge that dropped every tentative tag its inputs carried.
///
/// The merge prompt asks for tags to survive, but nothing guarantees it, and a
/// merge that kept the claims and lost their tags has done exactly the thing
/// this system exists to stop. It can't be undone mechanically, because the
/// merged prose can't be aligned claim by claim with its inputs, but it can be
/// made visible, so the entry reads as unconfirmed rather than settled.
pub const FLATTENED_NOTE: &str = "[Some claims merged into this entry were tentative in the \
    accounts it came from, and this merge kept none of their tags. Treat its conclusions as \
    unconfirmed unless they cite [observed #N]; the archive has the originals.]";

/// How many tentative tags in the inputs it takes before losing all of them
/// counts as flattening rather than as the merge simply dropping a claim.
const FLATTENED_AT: usize = 2;

/// How many entries a single merge consumes. Two at a time keeps each merge a
/// small, local operation, which is what keeps drift-through-merges bounded.
const MERGE_ARITY: usize = 2;

/// Hold every layer to its budget, merging the oldest entries downward.
///
/// The oldest run of `arity` neighbours that may be merged with one another:
/// all carrying the same job tag, or all carrying none.
///
/// Merging is what turns several gists into one piece of prose, and once that
/// has happened there is no way back to the parts. A job's close has to be able
/// to delete its memory without taking unrelated memory with it, so a row is
/// never allowed to hold both. Neighbours, because a merge takes the whole span
/// from its first input to its last: merging across a gap would swallow
/// everything in between.
///
/// None when no such run exists -- the level stays over budget until one does,
/// which is the next gist written inside the same job. The caller must treat
/// that as a reason to stop rather than to try again: the budget loops would
/// otherwise spin until their guard.
fn mergeable_run(gists: &[Gist], arity: usize) -> Option<Vec<Gist>> {
    gists
        .windows(arity)
        .find(|run| run.iter().all(|gist| gist.job == run[0].job))
        .map(<[Gist]>::to_vec)
}

/// Overflow always moves down one level; nothing is dropped without something
/// standing in its place. Shape merges into *itself* when full, which is the
/// property that stops the memory message growing with the length of the day.
#[cfg(test)]
pub fn enforce_budgets(layers: &MemoryLayers, summarizer: &dyn Summarizer, sources: &dyn Sources) -> Result<CascadeReport> {
    enforce_budgets_within(layers, summarizer, sources, Level::Episode.budget(), Level::Shape.budget())
}

/// `enforce_budgets` against a window's own budgets for episode gists and
/// Shape entries (`crate::context::guard::FoldSchedule`): a long window keeps more of
/// both before merging.
pub fn enforce_budgets_within(
    layers: &MemoryLayers,
    summarizer: &dyn Summarizer,
    sources: &dyn Sources,
    gist_budget: usize,
    shape_budget: usize,
) -> Result<CascadeReport> {
    let mut report = CascadeReport::default();

    // Episodes first: they overflow into Shape, so doing them in the other order
    // could leave Shape over budget again immediately.
    let mut guard = 0;
    while layers.tokens_at(Level::Episode)? > gist_budget {
        let episodes = layers.level(Level::Episode)?;
        if episodes.len() < MERGE_ARITY {
            // A single episode over budget on its own can't be merged with
            // anything. Leaving it is correct: it will merge with the next one.
            break;
        }
        let Some(oldest) = mergeable_run(&episodes, MERGE_ARITY) else {
            // Only neighbours from different jobs are left. Leaving the level
            // over budget is the lesser cost: a merge here would write one
            // entry holding a job's work and other work together.
            break;
        };
        let merged = merge_into(layers, summarizer, sources, Level::Shape, &oldest, &mut report)?;
        report.episode_merges += 1;
        report.mechanical_merges += usize::from(merged.mechanical);
        report.flattened_merges += usize::from(merged.flattened);

        guard += 1;
        if guard > 64 {
            break;
        }
    }

    let mut guard = 0;
    while layers.tokens_at(Level::Shape)? > shape_budget {
        let shapes = layers.level(Level::Shape)?;
        if shapes.len() < MERGE_ARITY {
            break;
        }
        let Some(oldest) = mergeable_run(&shapes, MERGE_ARITY) else {
            break;
        };
        let merged = merge_into(layers, summarizer, sources, Level::Shape, &oldest, &mut report)?;
        report.shape_merges += 1;
        report.mechanical_merges += usize::from(merged.mechanical);
        report.flattened_merges += usize::from(merged.flattened);

        guard += 1;
        if guard > 64 {
            break;
        }
    }

    report.episode_tokens = layers.tokens_at(Level::Episode)?;
    report.shape_tokens = layers.tokens_at(Level::Shape)?;
    Ok(report)
}

struct Merged {
    mechanical: bool,
    flattened: bool,
}

/// Replace `inputs` with one merged entry at `level`.
fn merge_into(
    layers: &MemoryLayers,
    summarizer: &dyn Summarizer,
    sources: &dyn Sources,
    level: Level,
    inputs: &[Gist],
    report: &mut CascadeReport,
) -> Result<Merged> {
    let from = inputs.iter().map(|gist| gist.from_event_id).min().unwrap_or(0);
    let through = inputs.iter().map(|gist| gist.through_event_id).max().unwrap_or(0);

    // Merges carry their event range forward, so however many times a stretch has
    // been merged, the way back to the originals is unchanged. That is what bounds
    // telephone-style drift: deep sleep re-derives the day from the originals, so
    // a misreading lives at most until the next sleep. The source hash is the
    // merged range's own, so it can be checked against the archive like any gist's.
    let span = match sources.spans(&[(from, through)]) {
        Ok(mut spans) if spans.len() == 1 => Some(spans.remove(0)),
        Ok(_) => None,
        Err(error) => {
            report.unread.push(format!("#{from}..#{through}: {error}"));
            None
        }
    };
    let source_hash = span.as_ref().map(|span| span.source_hash.clone()).unwrap_or_default();

    // The merge prompt heads each input with its dates from the archive, never
    // a stored distance label: a label like "earlier today" there could be
    // copied into the merged prose as if it were a time.
    let times = sources.times(&range_ends(
        inputs.iter().map(|gist| (gist.from_event_id, gist.through_event_id)).chain([(from, through)]),
    ));
    let dated_inputs: Vec<Gist> = inputs
        .iter()
        .cloned()
        .map(|mut gist| {
            gist.time_label = dated(gist.from_event_id, gist.through_event_id, &times).unwrap_or_default();
            gist
        })
        .collect();

    let (text, mechanical) = match summarizer.merge(level, &dated_inputs) {
        Ok(text) if !text.trim().is_empty() => (text, false),
        Ok(_) => (
            mechanical_merge_text(inputs, "the merge came back empty"),
            true,
        ),
        Err(error) => (
            mechanical_merge_text(inputs, &format!("the merge failed: {error}")),
            true,
        ),
    };

    // A merge rewrites prose, and can rewrite a citation with it. It may only
    // keep events its inputs cited, and only ones that are tool results in its
    // range (in its range, when the archive couldn't be read).
    let cited_in: HashSet<u64> = inputs.iter().flat_map(|gist| cited_observations(&gist.text)).collect();
    let (mut text, removed) = keep_citations(&text, &|id| {
        cited_in.contains(&id)
            && match &span {
                Some(span) => span.tool_results.contains(&id),
                None => (from..=through).contains(&id),
            }
    });
    report.citations_removed.extend(removed);

    let tentative_in: usize = inputs.iter().map(|gist| tentative_tags(&gist.text)).sum();
    let flattened =
        !mechanical && tentative_in >= FLATTENED_AT && tentative_tags(&text) == 0;
    if flattened {
        text.push_str("\n\n");
        text.push_str(FLATTENED_NOTE);
    }

    let merged = NewGist {
        level,
        from_event_id: from,
        through_event_id: through,
        source_hash,
        tokens: estimate_tokens(&text),
        time_label: dated(from, through, &times).unwrap_or_else(|| summarizer.time_label(inputs)),
        written_by: if mechanical { "runtime".to_string() } else { summarizer.model_name() },
        text,
        mechanical,
        // The oldest input's position, so the label says where the span starts.
        distance: inputs.first().map_or(0, |gist| gist.distance),
        // Every input carries the same tag (`mergeable_run`), so the merged
        // entry carries it too and stays deletable with its job.
        job: inputs.first().and_then(|gist| gist.job.clone()),
    };

    layers.append(&merged)?;
    layers.remove(&inputs.iter().map(|gist| gist.id).collect::<Vec<_>>())?;

    Ok(Merged { mechanical, flattened })
}

/// A merge that couldn't be written keeps the inputs' first lines rather than
/// losing them: coarser than a real merge, but never a silent gap.
fn mechanical_merge_text(inputs: &[Gist], reason: &str) -> String {
    let mut out = format!(
        "[merged mechanically — {reason}. This is an index of the stretch, not an account of it.]",
    );
    for gist in inputs {
        let first = gist.text.lines().find(|line| !line.trim().is_empty()).unwrap_or("").trim();
        out.push_str(&format!("\n- events {}: {}", gist.range_label(), truncate(first, 200)));
    }
    out
}

fn truncate(text: &str, chars: usize) -> String {
    let mut out: String = text.chars().take(chars).collect();
    if text.chars().count() > chars {
        out.push('…');
    }
    out
}

// ------------------------------------------------------------------ rendering

/// Where the fixed prefix sits.
pub const PREFIX_INDEX: usize = 0;

/// Where the memory message sits: directly under the prefix, above every live
/// turn. A fold edits exactly this one message and removes some old turns.
pub const MEMORY_INDEX: usize = 1;

/// Make sure the three-part layout exists, inserting an empty memory message if
/// the context predates folding (a restart, or a context built before this
/// module existed).
pub fn ensure_layout(messages: &mut Vec<Message>) {
    // Nothing to do without a prefix to sit under.
    if messages.get(PREFIX_INDEX).is_none() {
        return;
    }
    let present = messages
        .get(MEMORY_INDEX)
        .is_some_and(|message| message.content.starts_with(MEMORY_MARKER));
    if !present {
        messages.insert(MEMORY_INDEX, Message::user(empty_memory_message()));
    }
}

/// The live turns: everything below the memory message.
pub fn live_turns(messages: &[Message]) -> &[Message] {
    messages.get(MEMORY_INDEX + 1..).unwrap_or(&[])
}

/// Replace the memory message in place.
pub fn install_memory_message(messages: &mut Vec<Message>, text: String) {
    ensure_layout(messages);
    if let Some(slot) = messages.get_mut(MEMORY_INDEX) {
        *slot = Message::user(text);
    }
}

/// Remove the oldest `count` live turns, having gisted them.
pub fn drop_folded(messages: &mut Vec<Message>, count: usize) {
    let first = MEMORY_INDEX + 1;
    let last = (first + count).min(messages.len());
    if last > first {
        messages.drain(first..last);
    }
}

/// Opening line of the memory message, used to recognise it in a message list.
///
/// The memory message is a `user` message, not a `system` one. The chat template
/// accepts exactly one system message and only at position 0, so a second one at
/// index 1 is rejected outright ("System message must be at the beginning").
/// Merging it into the prefix instead would work, but then every fold re-prefills
/// the prefix too — about 24k of tokens the agent already has cached, for nothing.
///
/// So it is marked rather than roled: this line says plainly that the harness
/// rendered it, and the text under it is first-person throughout, so it reads as
/// the agent's own record rather than as something the operator said.
pub const MEMORY_MARKER: &str = "[memory — earlier in this session, rendered by the harness]";

fn empty_memory_message() -> String {
    render_memory_message(&[], &[], &[], &HashMap::new())
}

/// What the tags in the memory message mean, said once at the top of it.
pub const TAG_LEGEND: &str = "Tags say how I knew each thing at the time: [observed #N] I saw it \
    in a tool result (archive event N), [operator said] the operator told me and I didn't check it, \
    [inferred] my own conclusion, not checked, [guess] a hunch. A claim without a tag is not \
    thereby established.";

/// Build the memory message: Shape, then episode gists, oldest first, then what
/// is not settled.
///
/// Each entry is headed by when its stretch happened, from the archive's
/// timestamps (`times`). Dates, not "how long ago": they stay true however
/// long the message sits in context, so its bytes only change when a fold,
/// sleep or restart rebuilds it, and the cache holds in between. The intervals
/// the agent needs come fresh in each turn header instead.
///
/// This is the one message a fold edits. Everything below it in the context is
/// the agent's own turns, and everything above is the stable prefix, so a fold changes
/// exactly one message and removes some old turns — which is what "continuity by
/// overlap" amounts to in practice.
///
/// Open loops are deliberately *not* here. They change far more often than folds
/// do, and editing a message invalidates the cache for everything after it, so
/// they render as an appended block at the end of the context instead.
pub fn render_memory_message(
    shape: &[Gist],
    episodes: &[Gist],
    open: &[OpenItem],
    times: &HashMap<u64, DateTime<Local>>,
) -> String {
    let mut out = format!(
        "{MEMORY_MARKER}\n\
         My own memory of earlier in this session, at falling resolution. The further back \
         something is, the less detail I kept. The archive has all of it exactly.\n",
    );
    let anything = !(shape.is_empty() && episodes.is_empty() && open.is_empty());
    if anything {
        out.push_str(TAG_LEGEND);
        out.push('\n');
    }

    if !shape.is_empty() {
        out.push_str("\nSHAPE — the outline of a longer stretch, oldest first.\n");
        for gist in shape {
            out.push('\n');
            out.push_str(&gist.render(times));
            out.push('\n');
        }
    }

    if !episodes.is_empty() {
        out.push_str("\nEPISODES — what I was doing, in more detail, oldest first.\n");
        for gist in episodes {
            out.push('\n');
            out.push_str(&gist.render(times));
            out.push('\n');
        }
    }

    if !open.is_empty() {
        out.push_str(
            "\nNOT SETTLED — open questions, untested conclusions and unchecked reports, oldest \
             first. Each stays here until evidence settles it; none of them is established.\n",
        );
        for item in open {
            out.push_str(&item.render(times));
            out.push('\n');
        }
    }

    if !anything {
        out.push_str("\nNothing folded yet this session: everything so far is still in front of me.\n");
    } else {
        out.push('\n');
        out.push_str(ARCHIVE_LINE);
        out.push('\n');
    }

    out
}

/// Assemble the memory message from storage, dated from the archive
/// (`sources`, the Hippocampus).
pub fn build_memory_message(layers: &MemoryLayers, sources: &dyn Sources) -> Result<String> {
    let shape = layers.level(Level::Shape)?;
    let episodes = layers.level(Level::Episode)?;
    let open = layers.open_items()?;
    let times = sources.times(&range_ends(
        shape
            .iter()
            .chain(&episodes)
            .map(|gist| (gist.from_event_id, gist.through_event_id))
            .chain(open.iter().map(|item| (item.from_event_id, item.through_event_id))),
    ));
    Ok(render_memory_message(&shape, &episodes, &open, &times))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionCall, ToolCall};
    use std::cell::RefCell;

    fn layers() -> MemoryLayers {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let root = std::env::temp_dir().join(format!(
            "s0ma-memory-layers-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst),
        ));
        std::fs::create_dir_all(&root).expect("test directory should be created");
        MemoryLayers::beside_archive(&root.join("hippocampus.jsonl"))
    }

    fn gist(level: Level, from: u64, through: u64, tokens: usize) -> NewGist {
        NewGist {
            level,
            from_event_id: from,
            through_event_id: through,
            source_hash: format!("hash-{from}-{through}"),
            tokens,
            time_label: "about an hour ago".to_string(),
            written_by: "s0ma".to_string(),
            text: format!("I worked on events {from}..{through}.\nIt went as expected."),
            mechanical: false,
            distance: through * 100,
            job: None,
        }
    }

    /// The same, tagged as a job's.
    fn job_gist(level: Level, from: u64, through: u64, tokens: usize, job: &str) -> NewGist {
        NewGist { job: Some(job.to_string()), ..gist(level, from, through, tokens) }
    }

    struct Stub {
        calls: RefCell<usize>,
        fail: bool,
        /// The time labels each merge was shown, in order.
        merged: RefCell<Vec<Vec<String>>>,
    }

    impl Stub {
        fn new() -> Self {
            Self { calls: RefCell::new(0), fail: false, merged: RefCell::default() }
        }
    }

    impl Summarizer for Stub {
        fn gist(&self, transcript: &str, _open_items: &str) -> Result<String> {
            *self.calls.borrow_mut() += 1;
            if self.fail {
                return Err("no server".into());
            }
            Ok(format!("I worked through {} characters of that.", transcript.len()))
        }

        fn merge(&self, _level: Level, inputs: &[Gist]) -> Result<String> {
            *self.calls.borrow_mut() += 1;
            self.merged.borrow_mut().push(inputs.iter().map(|gist| gist.time_label.clone()).collect());
            if self.fail {
                return Err("no server".into());
            }
            // ~5:1, the Shape ratio, so budgets actually come down.
            let total: usize = inputs.iter().map(|gist| gist.tokens).sum();
            Ok("m".repeat(total * BYTES_PER_TOKEN / 5))
        }
        fn model_name(&self) -> String {
            "stub".to_string()
        }
    }

    // ---- message helpers ----

    fn user(text: &str) -> Message {
        Message::user(text.to_string())
    }
    fn said(text: &str) -> Message {
        Message::assistant(text.to_string())
    }
    fn called(id: &str, size: usize) -> Message {
        Message::assistant_with_tools(
            String::new(),
            vec![ToolCall {
                id: id.to_string(),
                kind: "function".to_string(),
                function: FunctionCall {
                    name: "vm_exec".to_string(),
                    arguments: "x".repeat(size),
                },
            }],
        )
    }
    /// A tool result. Keep `size` at or under 7,500 bytes in tests that care
    /// about sizes: `Message::tool` caps larger content via `limit_tool_output`,
    /// which probes the tokenizer, so the resulting size would depend on whether
    /// a server happens to be running.
    fn result(id: &str, size: usize) -> Message {
        Message::tool(id.to_string(), "y".repeat(size))
    }
    /// A tool result built directly: `Message::tool` would ask the tokenizer.
    fn raw_result(id: &str, size: usize) -> Message {
        let mut message = Message::user("y".repeat(size));
        message.role = "tool".into();
        message.tool_call_id = Some(id.into());
        message
    }

    // ---- boundaries ----

    #[test]
    fn a_tool_call_is_never_split_from_its_result() {
        let live = vec![user("go"), called("a", 10), result("a", 10), said("done")];
        let cuts = fold_boundaries(&live);

        // After the call (index 1) the result is outstanding, so cut 2 is illegal.
        assert!(!cuts.contains(&2), "cut 2 orphans the result: {cuts:?}");
        assert!(cuts.contains(&3), "after the result is a legal cut: {cuts:?}");
        assert!(cuts.contains(&4), "after the reply is a legal cut: {cuts:?}");
    }

    // ---- planning ----

    #[test]
    fn a_fold_never_takes_the_running_turns_latest_steps() {
        // A long running turn, and a target bigger than all of it: the shape a
        // second or third fold attempt asks for. The latest steps stay.
        let mut live = vec![user("go")];
        for n in 0..30 {
            live.push(called(&format!("c{n}"), 2_000).with_reasoning("r".repeat(4_000)));
            live.push(raw_result(&format!("c{n}"), 7_000));
        }
        let total: usize = live.iter().map(estimate_message_tokens).sum();

        let plan = plan_fold(&live, total * 2);
        assert!(!plan.is_empty());
        let kept: usize = live[plan.messages_folded..].iter().map(estimate_message_tokens).sum();
        assert!(kept >= KEEP_RECENT_TOKENS, "kept only {kept} tokens of the turn");
        assert!(fold_boundaries(&live).contains(&plan.messages_folded));
        assert_eq!(live[plan.messages_folded].role, "assistant", "the kept stretch starts with a step of its own");

        // However big the latest step, it stays whole with its results.
        let live = vec![user("go"), called("a", 200_000), raw_result("a", 10), called("b", 200_000), raw_result("b", 10)];
        let plan = plan_fold(&live, 1_000_000);
        assert!(plan.messages_folded <= 3, "the latest call and its result stay: {plan:?}");

        // Worth folding by size (over 8k), but all of it is the latest step:
        // nothing is folded.
        let live = vec![user("go"), called("a", 40_000), raw_result("a", 100)];
        assert!(estimate_message_tokens(&live[1]) > MIN_USEFUL_FOLD_TOKENS);
        assert!(plan_fold(&live, 50_000).is_empty());
    }

    // ---- store and cascade ----

    #[test]
    fn overflowing_episodes_become_shape_and_nothing_is_lost() {
        let layers = layers();
        let stub = Stub::new();

        // Five episodes at 5k each: 25k against a 16k budget.
        for n in 0..5u64 {
            layers.append(&gist(Level::Episode, n * 40 + 1, n * 40 + 40, 5_000)).unwrap();
        }

        let report = enforce_budgets(&layers, &stub, &FakeArchive(vec![])).unwrap();

        assert!(report.episode_merges > 0);
        assert!(layers.tokens_at(Level::Episode).unwrap() <= GIST_BUDGET_TOKENS);
        assert!(layers.tokens_at(Level::Shape).unwrap() <= SHAPE_BUDGET_TOKENS);

        // The whole span is still covered: the oldest event any layer knows about
        // is still event 1, it is just held more coarsely now.
        let shape = layers.level(Level::Shape).unwrap();
        let episodes = layers.level(Level::Episode).unwrap();
        let earliest = shape
            .iter()
            .chain(episodes.iter())
            .map(|gist| gist.from_event_id)
            .min()
            .unwrap();
        assert_eq!(earliest, 1, "the beginning of the day survived, coarsened");
    }

    #[test]
    fn shape_merges_into_itself_so_a_long_day_cannot_grow_the_message() {
        let layers = layers();
        let stub = Stub::new();

        // Simulate a long day: many folds, each adding episodes.
        for round in 0..40u64 {
            for n in 0..4u64 {
                let base = round * 400 + n * 40 + 1;
                layers.append(&gist(Level::Episode, base, base + 39, 5_000)).unwrap();
            }
            enforce_budgets(&layers, &stub, &FakeArchive(vec![])).unwrap();

            assert!(
                layers.tokens_at(Level::Episode).unwrap() <= GIST_BUDGET_TOKENS,
                "episodes over budget in round {round}",
            );
            assert!(
                layers.tokens_at(Level::Shape).unwrap() <= SHAPE_BUDGET_TOKENS,
                "shape over budget in round {round}",
            );
        }

        // And the start of the day is still represented.
        let shape = layers.level(Level::Shape).unwrap();
        assert!(!shape.is_empty());
        assert_eq!(shape.iter().map(|gist| gist.from_event_id).min().unwrap(), 1);
    }

    // ---- rendering ----

    #[test]
    fn the_layout_survives_a_context_that_predates_folding() {
        // A restart, or a context built before this module existed, has a prefix
        // and turns but no memory message. Folding must be able to start there
        // rather than mistaking a real turn for the memory slot and overwriting
        // it.
        let mut messages =
            vec![Message::system("prefix".into()), user("hello"), said("hi")];
        ensure_layout(&mut messages);

        assert_eq!(messages.len(), 4);
        assert!(messages[MEMORY_INDEX].content.starts_with(MEMORY_MARKER));
        assert_eq!(live_turns(&messages).len(), 2, "both turns are still live");
        assert_eq!(live_turns(&messages)[0].content, "hello");

        // Idempotent: running it again does not insert a second one.
        ensure_layout(&mut messages);
        assert_eq!(messages.len(), 4);
    }

    // ---- dates ----

    // ---- provenance ----

    // ---- not settled ----

    fn new_item(from: u64, through: u64, text: &str) -> NewOpenItem {
        NewOpenItem {
            from_event_id: from,
            through_event_id: through,
            text: text.to_string(),
            distance: through,
            job: None,
        }
    }

    #[test]
    fn deep_sleep_clears_items_only_once_they_are_closed_or_handed_off() {
        let layers = layers();
        let ids: Vec<i64> = (0..4)
            .map(|n| {
                let id = layers.next_item_id().unwrap();
                layers.add_item(id, &new_item(n * 10 + 1, n * 10 + 10, &format!("item {n}"))).unwrap();
                id
            })
            .collect();
        layers.resolve_item(ids[0], ItemState::Settled, "done [operator said]").unwrap();
        // What the handoff does: a settled item with no record needs nothing,
        // an open one gets a record.
        layers.mark_handed_off(ids[0], "nothing to hand off").unwrap();
        layers.mark_handed_off(ids[1], "unsettled:U2").unwrap();
        // ids[2]: open, handoff failed. ids[3]: beyond the sleep window.

        layers.clear_through(30).unwrap();

        let left: Vec<i64> = layers.items_where("1=1", params![]).unwrap().iter().map(|i| i.id).collect();
        assert_eq!(left, vec![ids[2], ids[3]], "unhanded and uncovered items stay");
        assert!(layers.next_item_id().unwrap() > ids[3], "numbers are never reused");
    }

    // ---- flattening ----

    struct Echo(&'static str);

    impl Summarizer for Echo {
        fn gist(&self, _transcript: &str, _open_items: &str) -> Result<String> {
            Ok(self.0.to_string())
        }
        fn merge(&self, _level: Level, _inputs: &[Gist]) -> Result<String> {
            Ok(self.0.to_string())
        }
    }

    /// The loops that enforce the budgets are `while over budget`, so a merge
    /// they refuse has to stop them rather than be retried: before the guard
    /// existed they would spin to their 64-iteration backstop every fold.
    #[test]
    fn a_refused_merge_stops_the_budget_loop_instead_of_spinning() {
        let layers = layers();
        let over = Level::Shape.budget();
        layers.append(&gist(Level::Shape, 1, 40, over)).unwrap();
        layers.append(&job_gist(Level::Shape, 41, 80, over, "acme")).unwrap();
        let stub = Stub::new();
        enforce_budgets(&layers, &stub, &FakeArchive(vec![])).unwrap();
        assert_eq!(*stub.calls.borrow(), 0, "the summarizer was not called once");
    }

    // ---- citations ----

    fn citing(from: u64, through: u64, cited: u64) -> NewGist {
        let mut entry = gist(Level::Episode, from, through, 9_000);
        entry.text = format!("I saw it happen [observed #{cited}]. {}", "p".repeat(36_000));
        entry
    }

    #[test]
    fn a_merge_whose_range_cant_be_read_still_keeps_only_what_its_inputs_cited() {
        let layers = layers();
        layers.append(&citing(1, 10, 2)).unwrap();
        layers.append(&citing(11, 20, 12)).unwrap();
        let merged = "Kept [observed #2]. Invented [observed #4242]. Out of range [observed #12, #30].";
        let report = enforce_budgets(&layers, &Echo(merged), &UnreadableArchive).unwrap();
        assert_eq!(report.unread.len(), 1, "{:?}", report.unread);
        assert_eq!(report.citations_removed, vec![4242, 30]);
        let shape = &layers.level(Level::Shape).unwrap()[0];
        assert_eq!(shape.text, "Kept [observed #2]. Invented [observed]. Out of range [observed #12].");
        assert_eq!(shape.source_hash, "", "no hash rather than a wrong one");
    }
}
