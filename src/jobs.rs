//! Jobs: a named stretch of work whose record is removed when it ends.
//!
//! Some work is done in confidence for a client, and when it is over the record
//! of what was said has to go, while everything built on top of the archive has
//! to keep working. `/job start <name>` opens such a stretch, `/job close
//! <name>` ends it and takes the content out of every store that kept a copy.
//!
//! # Why redaction rather than deletion
//!
//! An event's id is its line number: `Hippocampus::count_events` derives the
//! next id from the number of lines in the archive. Remove a line and every
//! event after it is renumbered, which silently repoints every reference held
//! anywhere else -- the call store's `archive_events`, the keyword and semantic
//! indexes, the weakness log's cited evidence, a mistake's `pointed_at`. So the
//! archive keeps one line per id and the content is replaced with [`MARKER`].
//! What an event *was* is gone; that there was an event, when, and who spoke is
//! kept, which is what lets the agent see a gap in its own past rather than a
//! discontinuity it cannot account for.
//!
//! # What a span is
//!
//! A job covers **everything appended while it was open**, not everything about
//! a topic. Nothing here guesses which events were sensitive: an event's place
//! in time is the only thing that can be judged reliably, and a guess that got
//! it wrong would leave content behind. Unrelated work in the middle of a job
//! is what `/job pause` is for -- it closes the current span, and `/job resume`
//! opens the next one, so the gap between them is never covered.
//!
//! This is also why only one job is open at a time: two would make an event's
//! span ambiguous, and closing one would have to decide what the other's
//! redaction left of it.
//!
//! # What a close reaches
//!
//! Content lands in more places than the archive, and a close that stopped
//! there would leave the text in several of them:
//!
//! * the archive -- redacted in place (above);
//! * the keyword and semantic indexes, which hold event text and embedded
//!   chunks of it;
//! * the call store, where each saved model call holds the whole conversation
//!   as it was sent;
//! * graded memory, whose gists are written from the events;
//! * the continuous-compaction windows, likewise;
//! * `/mistake` records, which point at calls that are going;
//! * the input history file, which keeps what was typed at the prompt, and the
//!   prompt's own memory of this session;
//! * the worker journal, a second verbatim archive: a subagent is given its
//!   task in the agent's words and journals its whole transcript.
//!
//! Deep sleep is held while a job is open, because what it writes -- the
//! checkpoint file and the world model -- is merged prose with no event ranges
//! to delete by, so content that reached it could not be taken out again
//! without destroying unrelated memory with it. A scheduled sleep falling
//! inside a job is **skipped, not queued**: a job can outlast the grace a held
//! sleep gets, and holding one would pause the workers and loop mode for the
//! rest of the job and then skip it anyway. Closing lifts the hold, so the next
//! scheduled sleep runs over the redacted archive, or `/sleep` does one at
//! once.
//!
//! # What a close cannot reach
//!
//! Said plainly because a half-kept promise is worse than a known limit:
//!
//! * **What the agent wrote down deliberately.** The world model, the board,
//!   continuity threads, skills, open loops, its attention note, and a weakness
//!   it reported (that log is hash-chained and cannot be rewritten at all) are
//!   written by tool calls, in the agent's own words, with no record of which event
//!   they came from. A job being open is in every turn header, which is where
//!   the agent is told to keep client specifics out of them.
//! * **Blocks on the disk.** A rewrite leaves the old bytes wherever the
//!   filesystem had them. The SQLite stores are vacuumed, which rebuilds them
//!   without the freed pages, but neither that nor a rename is an erase.
//! * **Anything already copied out** -- a `/share`d file, a transcript, a
//!   terminal scrollback.

use crate::hippocampus::Hippocampus;

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

type Result<T> = std::result::Result<T, String>;

/// What replaces a redacted event's content. Fixed and dull on purpose: it is
/// matched in tests, read by a person scrolling the archive, and indexed by
/// both indexes once the redacted line is re-read.
///
/// **The name is in it, and the marker stays for good.** So a job named after
/// a client writes that client's name permanently into the archive and into
/// both search indexes -- which can be the one thing the job was keeping back.
/// A code name avoids that, and nothing here depends on the name meaning
/// anything. `/job start` says so.
pub fn marker(job: &str) -> String {
    format!("[redacted for job {job}]")
}

const FILE: &str = "jobs.json";
const MAX_NAME: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// Open: everything appended now is covered.
    Open,
    /// Set aside. The spans so far stay; nothing new is covered.
    Paused,
    /// Ended and purged.
    Closed,
}

fn state_word(state: State) -> &'static str {
    match state {
        State::Open => "open",
        State::Paused => "paused",
        State::Closed => "closed",
    }
}

/// One stretch of the archive a job covers: the events appended between the
/// moment it opened (or resumed) and the moment it paused (or closed).
///
/// `through_event_id` is absent while the span is open; the live end is the
/// archive's newest event, read when it is needed rather than stored, so a
/// crash cannot leave a span claiming to end before events it covers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub from_event_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub through_event_id: Option<u64>,
    pub from: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub name: String,
    pub state: State,
    pub opened_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed_at: Option<String>,
    #[serde(default)]
    pub spans: Vec<Span>,
    /// What the close removed, kept so that `/job list` can still account for
    /// a job whose content is gone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purged: Option<Purged>,
    /// Lines typed during the job that the archive never got, so that its
    /// redaction cannot hand them to the typed history's purge: a `/goal`
    /// that was refused, with the job's content after it. Emptied by the close.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub typed_elsewhere: Vec<String>,
}

/// Counts from a close. Numbers only: naming what was removed would put some of
/// it back.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Purged {
    pub events: usize,
    /// The events redacted, and what the operator's held, carried from the
    /// redaction to the stores that need them. Not part of the record: both
    /// are taken out before it is written, so nothing of a closed job's text
    /// or shape is kept in `jobs.json`.
    #[serde(skip)]
    ids: Vec<u64>,
    #[serde(skip)]
    typed: Vec<String>,
    pub calls: usize,
    pub blobs: usize,
    pub gists: usize,
    pub open_items: usize,
    pub compaction_windows: usize,
    pub mistakes: usize,
    /// Lines taken out of the mistake history.
    pub history_lines: usize,
    /// Lines taken out of the prompt's typed history.
    pub typed_lines: usize,
    /// Records taken out of the worker journal.
    pub worker_records: usize,
    pub keyword_rows: usize,
    pub semantic_rows: usize,
}

impl Job {
    /// The event ranges this job covers, with any open span ended at `latest`.
    /// Empty ranges (a span opened and closed with nothing appended) are left
    /// out, so a caller never sees a reversed range.
    pub fn ranges(&self, latest: u64) -> Vec<(u64, u64)> {
        self.spans
            .iter()
            .filter_map(|span| {
                let through = span.through_event_id.unwrap_or(latest);
                (through >= span.from_event_id).then_some((span.from_event_id, through))
            })
            .collect()
    }

    /// The wall-clock windows this job covers, for selecting saved calls, with
    /// any open window ended now. A call is judged by when it started.
    fn windows(&self) -> Vec<(DateTime<Local>, DateTime<Local>)> {
        let now = Local::now();
        self.spans
            .iter()
            .filter_map(|span| {
                let from = DateTime::parse_from_rfc3339(&span.from).ok()?.with_timezone(&Local);
                let to = match &span.to {
                    Some(to) => DateTime::parse_from_rfc3339(to).ok()?.with_timezone(&Local),
                    None => now,
                };
                Some((from, to))
            })
            .collect()
    }

    fn open_span(&mut self) -> Option<&mut Span> {
        self.spans.iter_mut().find(|span| span.through_event_id.is_none())
    }

    fn end_span(&mut self, latest: u64, at: &str) {
        if let Some(span) = self.open_span() {
            span.through_event_id = Some(latest);
            span.to = Some(at.to_owned());
        }
    }

    fn events(&self, latest: u64) -> u64 {
        self.ranges(latest).iter().map(|(from, through)| through - from + 1).sum()
    }
}

// ------------------------------------------------------------------ the file

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Jobs {
    #[serde(default)]
    jobs: Vec<Job>,
}

impl Jobs {
    fn get(&self, name: &str) -> Option<&Job> {
        self.jobs.iter().find(|job| job.name == name)
    }

    fn get_mut(&mut self, name: &str) -> Option<&mut Job> {
        self.jobs.iter_mut().find(|job| job.name == name)
    }

    fn open(&self) -> Option<&Job> {
        self.jobs.iter().find(|job| job.state == State::Open)
    }
}

/// Where the jobs are kept, and what is in them.
///
/// A global, because the questions asked of it come from everywhere -- the turn
/// header, a gist being stored, an event about to be cited -- and threading a
/// path through all of those would say nothing except that jobs exist.
static DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
static LOADED: Mutex<Option<Jobs>> = Mutex::new(None);

/// The data folder, so the commands and the header note read the same file.
/// Called once at start-up; without it nothing here does anything, which is
/// what a mode that never calls it (`--print-system-prompt`) wants.
pub fn init(data_dir: &Path) {
    *lock(&DIR) = Some(data_dir.to_path_buf());
    *lock(&LOADED) = None;
    // Load now, so a job left open by the last run is in the first turn header
    // rather than appearing once something else asks.
    with(|_| ());
}

fn lock<T>(cell: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    cell.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn path() -> Option<PathBuf> {
    Some(lock(&DIR).as_ref()?.join(FILE))
}

/// Read the jobs, loading them from disk the first time.
fn with<T>(read: impl FnOnce(&Jobs) -> T) -> Option<T> {
    let mut guard = lock(&LOADED);
    if guard.is_none() {
        *guard = Some(load(&path()?).unwrap_or_else(|error| {
            crate::terminal::warn(&format!(
                "[job] {FILE} could not be read ({error}); nothing here can say whether a job is open, so deep \
                 sleep is held and a weakness can cite no event until it is fixed by hand."
            ));
            UNREADABLE.store(true, std::sync::atomic::Ordering::SeqCst);
            Jobs::default()
        }));
    }
    Some(read(guard.as_ref()?))
}

/// Change the jobs and write them back. The cache is replaced only once the
/// file is on disk, so a failed write never leaves this process believing in a
/// job the next one will not find.
fn change<T>(edit: impl FnOnce(&mut Jobs) -> Result<T>) -> Result<T> {
    let path = path().ok_or("no data folder: jobs are unavailable in this mode")?;
    let mut guard = lock(&LOADED);
    let mut jobs = match guard.take() {
        Some(jobs) => jobs,
        None => load(&path).map_err(|error| format!("{FILE} could not be read: {error}"))?,
    };
    let result = edit(&mut jobs);
    if result.is_ok()
        && let Err(error) = save(&path, &jobs) {
            // The edit is dropped: the cache goes back to the file's contents.
            *guard = load(&path).ok();
            return Err(format!("{FILE} could not be written ({error}); nothing changed"));
        }
    *guard = Some(jobs);
    result
}

fn load(path: &Path) -> std::result::Result<Jobs, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(Jobs::default());
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn save(path: &Path, jobs: &Jobs) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(jobs).map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    crate::util::write_atomic(path, &bytes)?;
    Ok(())
}

// ------------------------------------------------------------------ what the rest of the agent asks

/// The open job's name, or nothing. Every gate below is this question.
pub fn open_job() -> Option<String> {
    with(|jobs| jobs.open().map(|job| job.name.clone())).flatten()
}

/// The job that is open or paused, by name: what was written while it was
/// either may quote it (a paused job's conversation is still in the live
/// context). Unlike `sleep_held`, nothing when the store cannot be read, since
/// the answer is used as a name.
pub fn held_job() -> Option<String> {
    with(|jobs| {
        jobs.jobs
            .iter()
            .find(|job| matches!(job.state, State::Open | State::Paused))
            .map(|job| job.name.clone())
    })
    .flatten()
}

/// A line typed while a job is open that the archive will not hold, to be
/// taken out of the typed history when the job closes. Best effort: a store
/// that cannot be written says so.
pub fn keep_for_close(line: &str) {
    let Some(name) = open_job() else { return };
    let kept = change(|jobs| {
        let job = jobs.get_mut(&name).ok_or(format!("no job {name}"))?;
        if !job.typed_elsewhere.iter().any(|kept| kept == line) {
            job.typed_elsewhere.push(line.to_string());
        }
        Ok(())
    });
    if let Err(error) = kept {
        crate::terminal::warn(&format!(
            "[job] a line typed during \"{name}\" could not be noted for its close ({error}); it may stay in the \
             typed history afterwards"
        ));
    }
}

/// Whether the store would not parse, in which case every question here is
/// answered the careful way round rather than as "no job".
///
/// The file is the only thing that knows a job is open, and what it gates is
/// what keeps a job's content out of the stores a close cannot reach. A missing
/// file is not this -- that is simply no jobs -- only one that would not read.
fn unreadable() -> bool {
    // The flag is raised by the load, so the load has to have happened: asked
    // before anything else has read the file, this would otherwise answer for
    // a file it has not looked at.
    with(|_| ());
    UNREADABLE.load(std::sync::atomic::Ordering::SeqCst)
}

static UNREADABLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The job holding deep sleep back, if any: one that is open **or paused**.
///
/// Paused counts. Pausing says "what I am doing now is not part of the job",
/// not "the job's record no longer needs protecting" -- its earlier spans are
/// still there and still to be redacted. A deep sleep during a pause would
/// fold those spans into the checkpoint and the world model, which a close
/// cannot reach, so the hold lasts until the job is closed.
pub fn sleep_held() -> Option<String> {
    if unreadable() {
        return Some(format!("unknown — {FILE} could not be read"));
    }
    with(|jobs| {
        jobs.jobs
            .iter()
            .find(|job| matches!(job.state, State::Open | State::Paused))
            .map(|job| job.name.clone())
    })
    .flatten()
}

/// Whether a close has left the agent's live context holding text that is no longer
/// in the archive, so the caller owes it a rotation. Cleared by reading.
///
/// A close redacts the archive and the stores, but the conversation is also in
/// the live context, which nothing here can edit: the messages are the agent's
/// working memory, held by the running session. Left alone, the next fold would
/// write gists from those messages and put the content back into the store the
/// close just cleared. Rotating rebuilds the context from the archive, which by
/// then says `[redacted for job …]`.
pub fn take_rotation_owed() -> bool {
    ROTATION_OWED.swap(false, std::sync::atomic::Ordering::SeqCst)
}

static ROTATION_OWED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Bumped by a close. The prompt reads it before each line and forgets what it
/// is holding when it has moved.
///
/// The typed history is not only the file. The prompt keeps this session's
/// lines in memory and writes them out when it syncs -- on a steer, and on the
/// way out -- so a close that rewrote the file alone would be undone at exit,
/// putting the job's lines back. The steering pane keeps its own list too.
/// Neither is reachable from here: they belong to the thread reading the
/// keyboard, which is why this is a counter it can poll rather than a call.
/// An attached terminal hears that this agent's history was purged: its
/// prompt forgets its own lines the same way (`session.rs`).
pub fn note_history_purge() {
    HISTORY_PURGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

pub fn history_purges() -> u64 {
    HISTORY_PURGES.load(std::sync::atomic::Ordering::SeqCst)
}

static HISTORY_PURGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The tag a gist written now carries, so a close can find it again. A gist
/// tagged this way is never merged with one carrying another tag or none.
///
/// This follows the **hold**, not what is being covered: a pause stops new
/// events belonging to the job, but it does not take the job's conversation out
/// of the live context, and a gist written during a pause is written from that
/// context. Untagged, such a gist would survive the close holding part of the
/// job; tagged, it goes with it. The cost is that unrelated work during a pause
/// loses its gists at the close too -- the safe way round, and a reason to
/// close rather than pause when the work is finished.
pub fn gist_tag() -> Option<String> {
    sleep_held()
}

/// The job covering `id`, if one does and is not closed yet. `latest` is the
/// archive's newest event, which ends a span that is still open.
///
/// This exists for the weakness log. That log is append-only and hash-chained,
/// and each cited event is stored with a 500-character excerpt of it -- so an
/// instance citing a job event would keep a copy of it that a close cannot
/// take out without breaking the chain. Citing one is refused instead.
pub fn covering_event(latest: u64, id: u64) -> Option<String> {
    if unreadable() {
        return Some(format!("unknown — {FILE} could not be read"));
    }
    with(|jobs| {
        jobs.jobs
            .iter()
            .find(|job| {
                job.state != State::Closed
                    && job.ranges(latest).iter().any(|(from, through)| (*from..=*through).contains(&id))
            })
            .map(|job| job.name.clone())
    })
    .flatten()
}

/// The line about the open job for a turn header, appended to `header`.
///
/// It goes in the header rather than the system context because a job opens and
/// closes mid-session, and the system context is rebuilt only at start-up,
/// after sleep and on rotation -- a job noted there would appear hours late and
/// linger after its close. The header is written fresh for every input.
pub fn note_for_header(header: String) -> String {
    let Some(job) = sleep_held() else { return header };
    let state = if open_job().is_some() {
        "is open: this is work held in confidence. Everything in the archive from here until it closes will \
         have its content removed then, and the saved model calls, the memory of it and the workers' records \
         with it. Work normally, and say what you have to say."
    } else {
        "is paused, so what is said now is not part of it -- but the job's own conversation is still in your \
         context, and it is still to be removed when the job closes."
    };
    format!(
        "{header}\nJob \"{job}\" {state} One thing to keep in mind: what you write into the world model, the \
         board, a thread, a skill, an open loop, your attention note or a weakness report is in your own \
         words with no record of where it came from, so a close cannot take it out. Keep client specifics in \
         the conversation, where they are covered, and out of those. Deep sleep is held until the job closes."
    )
}

// ------------------------------------------------------------------ the command

const HELP: &str = "\
Jobs: a named stretch of work whose record is removed when it ends.

/job                       list the jobs
/job help                  this text (also -h, --help, ?)
/job start <name>          open a job: everything appended from now is covered
/job pause <name>          stop covering new events (unrelated work in between)
/job resume <name>         cover new events again, in a new span
/job close <name>          what closing would remove, and nothing else
/job close <name> --yes    remove it: archive content, both indexes, the saved
                           calls, the gists, the compaction windows, the
                           mistakes on those calls, the typed history and the
                           worker journal
/job show <name>           one job's spans and counts

A job covers everything appended while it was open, not everything about a
topic: time is the only thing that can be judged reliably. Pause it for
unrelated work in the middle. One job is open at a time.

The name goes into the archive in place of what was redacted and stays there,
so a code name is safer than a client's own.

A close cannot reach what s0ma wrote down in its own words -- the world model,
the board, threads, skills, open loops, its attention note, or a weakness it
reported -- so a job being open is in every turn header.

Deep sleep is held until the job is closed, a pause included. A scheduled one
falling inside a job is skipped rather than queued; closing lifts the hold, and
/sleep then runs one.";

/// Whether this command is about to remove something, so a caller can stop
/// anything that writes from what it is about to redact.
///
/// Continuous compaction is the case this exists for: `tick` hands a frozen
/// copy of the events to a worker, which then takes a model call per batch and
/// saves each one. A digest of the unredacted events could land in the windows
/// table after they were deleted, and the next rotation would put it straight
/// back into the live context. Quiescing it first is the only ordering that
/// works, and only the caller holds it.
pub fn will_remove(command: &str) -> bool {
    let mut words = command.split_whitespace();
    words.next() == Some("/job")
        && words.next() == Some("close")
        && command.split_whitespace().any(|word| word == "--yes")
}

/// Returns None if `command` isn't a /job command.
pub fn handle_operator_command(
    memory: &Hippocampus,
    command: &str,
) -> Option<std::result::Result<String, String>> {
    let mut words = command.split_whitespace();
    if words.next()? != "/job" {
        return None;
    }
    let args: Vec<&str> = words.collect();
    Some(run(memory, &args).map_err(|error| format!("[job] {error}")))
}

fn name_of(args: &[&str], sub: &str) -> Result<String> {
    let name = args.get(1).copied().unwrap_or_default();
    if name.is_empty() || name.starts_with("--") {
        return Err(format!("usage: /job {sub} <name>"));
    }
    // A name holds no spaces, so anything after it was meant to be part of one
    // and would name a different job than the one typed. `--yes` is the only
    // word allowed after a name, and only where it means something.
    let extra: Vec<&str> = args[2..].iter().copied().filter(|word| *word != "--yes").collect();
    if !extra.is_empty() {
        return Err(format!(
            "a job name is one word, so \"{}\" is not part of it: /job {sub} <name>",
            extra.join(" ")
        ));
    }
    if args.contains(&"--yes") && sub != "close" {
        return Err(format!("--yes means nothing to /job {sub}"));
    }
    if name.chars().count() > MAX_NAME {
        return Err(format!("that name is longer than {MAX_NAME} characters"));
    }
    if !name.chars().all(|c| c.is_alphanumeric() || "-_.".contains(c)) {
        return Err("a job name holds letters, digits, '-', '_' and '.' only: it goes into the archive in place \
                    of what was redacted"
            .into());
    }
    Ok(name.to_owned())
}

fn run(memory: &Hippocampus, args: &[&str]) -> Result<String> {
    let sub = args.first().copied().unwrap_or("list");
    match sub {
        "help" | "-h" | "--help" | "?" => Ok(HELP.to_owned()),
        "list" => list(memory),
        "show" => show(memory, &name_of(args, "show")?),
        "start" => start(memory, &name_of(args, "start")?),
        "pause" => pause(memory, &name_of(args, "pause")?),
        "resume" => resume(memory, &name_of(args, "resume")?),
        "close" => close(memory, &name_of(args, "close")?, args.contains(&"--yes")),
        other => Err(format!("unknown subcommand '{other}'\n{HELP}")),
    }
}

fn list(memory: &Hippocampus) -> Result<String> {
    let latest = memory.last_event_id();
    let lines = with(|jobs| {
        jobs.jobs
            .iter()
            .map(|job| {
                let events = match &job.purged {
                    Some(purged) => format!("{} events removed", purged.events),
                    None => format!("{} events", job.events(latest)),
                };
                format!("{:<20} {:<7} {events}", job.name, state_word(job.state))
            })
            .collect::<Vec<_>>()
    })
    .unwrap_or_default();
    if lines.is_empty() {
        Ok("[job] none yet. /job start <name> opens one.".into())
    } else {
        Ok(lines.join("\n"))
    }
}

fn show(memory: &Hippocampus, name: &str) -> Result<String> {
    let latest = memory.last_event_id();
    with(|jobs| match jobs.get(name) {
        None => Err(format!("no job {name}")),
        Some(job) => {
            let mut lines = vec![format!(
                "{} — {}, opened {}{}",
                job.name,
                state_word(job.state),
                job.opened_at,
                job.closed_at.as_ref().map(|at| format!(", closed {at}")).unwrap_or_default()
            )];
            for (index, span) in job.spans.iter().enumerate() {
                lines.push(match span.through_event_id {
                    Some(through) => format!("  span {}: events #{}..#{through}", index + 1, span.from_event_id),
                    None => format!("  span {}: events #{}.. (open)", index + 1, span.from_event_id),
                });
            }
            if let Some(purged) = &job.purged {
                lines.push(format!(
                    "  removed: {} events, {} calls, {} blobs, {} gists, {} open items, {} compaction windows, \
                     {} mistakes ({} history lines), {} typed lines, {} worker records, {} keyword rows, \
                     {} semantic rows",
                    purged.events,
                    purged.calls,
                    purged.blobs,
                    purged.gists,
                    purged.open_items,
                    purged.compaction_windows,
                    purged.mistakes,
                    purged.history_lines,
                    purged.typed_lines,
                    purged.worker_records,
                    purged.keyword_rows,
                    purged.semantic_rows
                ));
            } else {
                lines.push(format!("  covers {} events so far", job.events(latest)));
            }
            Ok(lines.join("\n"))
        }
    })
    .unwrap_or_else(|| Err("no data folder: jobs are unavailable in this mode".into()))
}

fn start(memory: &Hippocampus, name: &str) -> Result<String> {
    let from = memory.last_event_id() + 1;
    let at = Local::now().to_rfc3339();
    change(|jobs| {
        if let Some(existing) = jobs.get(name) {
            return Err(format!(
                "there is already a job called {name} ({}). Names are kept after a close, so its record stays \
                 accountable; pick another",
                state_word(existing.state)
            ));
        }
        if let Some(open) = jobs.open() {
            return Err(format!(
                "job \"{}\" is open: pause or close it first. One job is open at a time, so that every event \
                 belongs to one of them",
                open.name
            ));
        }
        jobs.jobs.push(Job {
            name: name.to_owned(),
            state: State::Open,
            opened_at: at.clone(),
            closed_at: None,
            spans: vec![Span { from_event_id: from, through_event_id: None, from: at.clone(), to: None }],
            purged: None,
            typed_elsewhere: Vec::new(),
        });
        Ok(())
    })?;
    Ok(format!(
        "[job] {name} open from event #{from}. Everything appended until it closes is covered, so pause it for \
         unrelated work. Deep sleep is held until the close.\n        The redaction reads \"{}\" and stays in \
         the archive for good, so if the name itself is confidential, a code name is the safer choice: close \
         this and start one under another name.",
        marker(name)
    ))
}

fn pause(memory: &Hippocampus, name: &str) -> Result<String> {
    let latest = memory.last_event_id();
    let at = Local::now().to_rfc3339();
    change(|jobs| {
        let job = jobs.get_mut(name).ok_or(format!("no job {name}"))?;
        match job.state {
            State::Closed => Err(format!("{name} is closed")),
            State::Paused => Err(format!("{name} is already paused")),
            State::Open => {
                job.end_span(latest, &at);
                job.state = State::Paused;
                Ok(())
            }
        }
    })?;
    Ok(format!(
        "[job] {name} paused at event #{latest}. What is appended now is not covered by it; /job resume {name} \
         starts covering again."
    ))
}

fn resume(memory: &Hippocampus, name: &str) -> Result<String> {
    let from = memory.last_event_id() + 1;
    let at = Local::now().to_rfc3339();
    change(|jobs| {
        if let Some(open) = jobs.open() {
            let open = open.name.clone();
            if open != name {
                return Err(format!("job \"{open}\" is open: pause or close it first"));
            }
        }
        let job = jobs.get_mut(name).ok_or(format!("no job {name}"))?;
        match job.state {
            State::Closed => Err(format!("{name} is closed: its record has been removed, and it cannot reopen")),
            State::Open => Err(format!("{name} is already open")),
            State::Paused => {
                job.spans.push(Span {
                    from_event_id: from,
                    through_event_id: None,
                    from: at.clone(),
                    to: None,
                });
                job.state = State::Open;
                Ok(())
            }
        }
    })?;
    Ok(format!("[job] {name} open again from event #{from}."))
}

/// What a close would remove, counted before anything is touched.
struct Plan {
    ranges: Vec<(u64, u64)>,
    events: u64,
    mid_turn: bool,
}

fn plan(memory: &Hippocampus, name: &str) -> Result<Plan> {
    let latest = memory.last_event_id();
    let job = with(|jobs| jobs.get(name).cloned())
        .flatten()
        .ok_or_else(|| format!("no job {name}"))?;
    if job.state == State::Closed {
        return Err(format!("{name} is already closed"));
    }
    let ranges = job.ranges(latest);
    Ok(Plan {
        events: ranges.iter().map(|(from, through)| through - from + 1).sum(),
        ranges,
        mid_turn: crate::calls::current_turn().is_some(),
    })
}

fn close(memory: &Hippocampus, name: &str, confirmed: bool) -> Result<String> {
    let plan = plan(memory, name)?;
    if !confirmed {
        let mut lines = vec![format!(
            "[job] closing {name} would remove the content of {} archive events, in {} stretch(es): {}",
            plan.events,
            plan.ranges.len(),
            plan.ranges
                .iter()
                .map(|(from, through)| format!("#{from}..#{through}"))
                .collect::<Vec<_>>()
                .join(", ")
        )];
        lines.push(
            "        Each of those events keeps its number, its time and who spoke; its content becomes \
             \"[redacted for job <name>]\". Also removed: those events' rows in the keyword and semantic \
             indexes, every saved model call from the job (each holds the whole conversation as it was sent), \
             the gists and NOT SETTLED items of the job, the compaction windows covering it, the /mistake \
             records on those calls, the matching lines of the typed history, and a /goal set during it."
                .into(),
        );
        lines.push(
            "        Not removed, because nothing records where it came from: what s0ma wrote into the world \
             model, the board, threads, skills, open loops or its attention note, and any weakness it \
             reported — that log is hash-chained, so it cannot be rewritten at all."
                .into(),
        );
        if plan.mid_turn {
            lines.push(
                "        s0ma is working now. Anything it appends after the close is not covered by the job, \
                 so pause the job instead if this stretch is not finished."
                    .into(),
            );
        }
        lines.push(format!("        This cannot be undone. /job close {name} --yes does it."));
        return Ok(lines.join("\n"));
    }

    // Refused rather than attempted, because the call the agent is making now
    // cannot be reached. A saved call is written when the next one begins, so
    // the one in flight has no line in the store yet and no archive events
    // listed against it -- neither the time window nor the event ranges can
    // select it -- while its request holds the conversation and its response
    // becomes events inside the job. Its blobs would be written after the
    // purge and stay for good.
    if plan.mid_turn {
        return Err(format!(
            "s0ma is in the middle of a turn, and the model call it is making now cannot be removed cleanly: \
             it is not in the call store yet, so nothing here can find it, and its request holds this \
             conversation. Let the turn finish (or interrupt it) and run this again. \"{name}\" stays open \
             until you do, so nothing stops being covered meanwhile."
        ));
    }

    let at = Local::now().to_rfc3339();
    let mut purged = Purged::default();
    let mut notes: Vec<String> = Vec::new();

    // The span stays open until the archive is redacted. The other order
    // leaves a failure halfway: the span closed, so nothing new is covered,
    // while the job reads as open and the header still says so.
    //
    // It is also why this loops. `redact_ranges` holds the append lock for its
    // own pass, but an event can land between reading the newest id and that
    // pass starting -- the agent is often working when a job is closed -- and such
    // an event would be inside the open span and outside the range just
    // redacted. So the newest id is read again afterwards and the remainder
    // redacted too, until a pass sees nothing new.
    let mut latest;
    let mut passes = 0;
    loop {
        latest = memory.last_event_id();
        let ranges = with(|jobs| jobs.get(name).map(|job| job.ranges(latest)))
            .flatten()
            .unwrap_or_default();
        let redacted = memory
            .redact_ranges(&ranges, &marker(name))
            .map_err(|error| format!("the archive was not redacted ({error}); the job is still open"))?;
        purged.events = redacted.events;
        purged.ids = redacted.ids;
        // Extended, never replaced. A second pass re-reads lines the first one
        // already redacted, so what it hands back for those is the marker; the
        // original text is only ever seen once, on the pass that replaced it.
        purged.typed.extend(redacted.typed);
        if redacted.unreadable > 0 {
            notes.push(format!(
                "{} archive line(s) would not parse and were replaced whole rather than edited",
                redacted.unreadable
            ));
        }
        passes += 1;
        if memory.last_event_id() == latest {
            break;
        }
        if passes >= 3 {
            // Left open deliberately. Closing here would record the job as
            // finished while known content of it is still in the archive, and
            // `plan` refuses a closed job -- so the advice to try again would
            // have been impossible to follow.
            return Err(format!(
                "events kept being appended through {passes} passes of the redaction, so some of \"{name}\" \
                 may still hold its content and the job is left open. What has been redacted stays redacted. \
                 Find what is still writing to the archive -- a loop, or a worker -- stop it, and close again."
            ));
        }
    }
    let ids = std::mem::take(&mut purged.ids);
    let mut typed = std::mem::take(&mut purged.typed);
    typed.extend(with(|jobs| jobs.get(name).map(|job| job.typed_elsewhere.clone())).flatten().unwrap_or_default());

    // Now the span can close: everything it covered has been redacted.
    change(|jobs| {
        let job = jobs.get_mut(name).ok_or(format!("no job {name}"))?;
        job.end_span(latest, &at);
        Ok(())
    })?;
    let ranges = with(|jobs| jobs.get(name).map(|job| job.ranges(latest)))
        .flatten()
        .unwrap_or_default();
    let windows = with(|jobs| jobs.get(name).map(Job::windows)).flatten().unwrap_or_default();

    // Each of the rest is reported and carried past rather than returned: the
    // archive is already redacted, so stopping here would leave the job open
    // with half its content gone and no way to tell which half.
    let mut failed = |what: &str, error: String| notes.push(format!("{what} failed: {error}"));

    let forgotten = crate::archive::forget_events(memory, &ids);
    purged.keyword_rows = forgotten.keyword_rows;
    purged.semantic_rows = forgotten.semantic_rows;
    for failure in forgotten.failures {
        failed("clearing", failure);
    }
    match crate::calls::purge_window(data_dir(memory), &windows, &ranges) {
        Ok(calls) => {
            purged.calls = calls.calls;
            purged.blobs = calls.blobs;
            purged.mistakes = calls.mistakes;
            purged.history_lines = calls.history_lines;
        }
        Err(error) => failed("dropping the saved model calls", error.to_string()),
    }
    // The working context saved for a restart holds the job's turns too.
    crate::context::resume::forget(data_dir(memory));
    // And the engine's kept prompts of the job's turns (and the side calls',
    // which saw them).
    crate::conversations::job_closed(name);
    // And the editor's copies of unsaved changes.
    crate::tabs::editor_forget_copies();
    let layers = crate::memory::layers::MemoryLayers::beside_archive(memory.archive_path());
    match layers.purge_job(name, &ranges) {
        Ok((gists, items)) => {
            purged.gists = gists;
            purged.open_items = items;
        }
        Err(error) => failed("clearing the memory of it", error.to_string()),
    }
    match crate::sleep::continuous::forget_ranges(memory, &ranges) {
        Ok(windows) => purged.compaction_windows = windows,
        Err(error) => failed("dropping the compaction windows", error.to_string()),
    }
    // A goal set during the job is the job's: it goes, and so does the line
    // that set it. One set before keeps its objective but not a note written
    // while the job was open.
    let mut goal_line = None;
    match crate::goal::forget_job(name) {
        Ok(forgotten) => {
            if let Some(line) = forgotten.typed {
                typed.push(line);
            }
            goal_line = match (forgotten.removed, forgotten.scrubbed) {
                (true, _) => Some("The goal set during it was cleared."),
                (false, true) => Some("The goal's note from during it was cleared; the goal itself was set before it."),
                (false, false) => None,
            };
        }
        Err(error) => failed("clearing the goal set during it", error),
    }
    match purge_typed_history(data_dir(memory), &typed) {
        Ok(lines) => purged.typed_lines = lines,
        Err(error) => failed("clearing the typed history", error.to_string()),
    }
    match crate::subagent::forget_journal_windows(&windows) {
        Ok(records) => purged.worker_records = records,
        Err(error) => failed("clearing the worker journal", error.to_string()),
    }
    // The pending daily review pins a hash of the events after the last
    // checkpoint. Redacting any of them makes that hash wrong, and every deep
    // sleep from then on refuses with "daily review source changed" until the
    // pin is removed by hand, with nothing to say a job caused it. The pin is
    // only a cutoff to resume from, so dropping it costs a re-derivation.
    let pin = memory.archive_path().with_file_name("daily_sleep_window.json");
    if pin.exists()
        && let Err(error) = fs::remove_file(&pin) {
            failed("dropping the pinned daily review window", error.to_string());
            notes.push(
                "until that file goes, every deep sleep will refuse with \"daily review source changed\": what \
                 it pinned is what was just redacted"
                    .into(),
            );
        }

    ROTATION_OWED.store(true, std::sync::atomic::Ordering::SeqCst);
    let counts = purged.clone();
    change(|jobs| {
        let job = jobs.get_mut(name).ok_or(format!("no job {name}"))?;
        job.state = State::Closed;
        job.closed_at = Some(at.clone());
        job.purged = Some(counts);
        job.typed_elsewhere.clear();
        Ok(())
    })?;

    let mut report = format!(
        "[job] {name} closed. Removed: {} archive events' content, {} saved calls ({} stored objects), \
         {} gists, {} open items, {} compaction windows, {} mistakes, {} typed lines, \
         {} worker records, {} keyword rows, {} semantic rows.",
        purged.events,
        purged.calls,
        purged.blobs,
        purged.gists,
        purged.open_items,
        purged.compaction_windows,
        purged.mistakes,
        purged.typed_lines,
        purged.worker_records,
        purged.keyword_rows,
        purged.semantic_rows
    );
    if let Some(line) = goal_line {
        report.push(' ');
        report.push_str(line);
    }
    if !notes.is_empty() {
        report.push_str("\n        ");
        report.push_str(&notes.join("\n        "));
        report.push_str("\n        What is listed above did go. Run /job show ");
        report.push_str(name);
        report.push_str(" for the counts.");
    }
    report.push_str(
        "\n        Deep sleep is no longer held: the next scheduled one runs over the redacted archive, or \
         /sleep does one now.",
    );
    Ok(report)
}

/// The data folder, from the archive beside it.
fn data_dir(memory: &Hippocampus) -> &Path {
    memory.archive_path().parent().unwrap_or(Path::new("."))
}

/// Take the job's own lines out of the prompt's typed history.
///
/// The history is what ↑ walks back through, and it keeps a plain list of lines
/// with no times and no ids -- so the only thing to match on is the text, which
/// is why the redaction hands it back. Every line equal to one of the operator's
/// redacted events goes; one of them is exactly what was typed.
///
/// Two kinds of line are left alone. A command is not client content and is
/// most of what the history is for -- `/job pause acme` would otherwise go,
/// having been typed during the job and archived like anything else. And a very
/// short line (`y`, `ok`) carries nothing worth removing while matching all
/// over the file.
fn purge_typed_history(data_dir: &Path, typed: &[String]) -> Result<usize> {
    const KEEP_SHORTER_THAN: usize = 12;
    // `/goal <objective>` is the exception among commands: the objective is
    // what was asked, as much the job's as a message.
    let keep = |line: &str| {
        (line.starts_with('/') && !crate::goal::carries_objective(line)) || line.chars().count() < KEEP_SHORTER_THAN
    };
    if typed.is_empty() {
        return Ok(0);
    }
    // Both forms of each line. The history file escapes a newline as `<\n>`
    // (reedline's own encoding), so a pasted multi-line message is one line
    // there and several events here -- and a paste is the case most worth
    // removing.
    let mut doomed: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in typed.iter().map(|line| line.trim()).filter(|line| !keep(line)) {
        doomed.insert(line.to_owned());
        if line.contains('\n') {
            doomed.insert(line.replace('\n', "<\\n>"));
        }
    }
    remove_from_history(data_dir, &doomed)
}

/// One line typed at the prompt, taken out of its history: the file, the
/// prompt's own memory of this session, and attached terminals'. For a line
/// that carried a secret (`/model key …`).
pub fn forget_typed_line(data_dir: &Path, line: &str) -> Result<usize> {
    remove_from_history(data_dir, &std::iter::once(line.trim().to_owned()).collect())
}

/// `doomed` lines out of `input_history.txt`, and the prompt told to forget
/// what it holds.
fn remove_from_history(data_dir: &Path, doomed: &std::collections::HashSet<String>) -> Result<usize> {
    // Asked for first, and whatever the file turns out to hold. The prompt
    // keeps this session's lines in memory until something syncs, and those are
    // the ones most likely to be the job's -- a file that does not exist yet
    // means exactly that nothing has been written out, not that there is
    // nothing to forget.
    HISTORY_PURGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    // Attached terminals keep their own prompt history, from the same file:
    // told once the file is purged, however this returns, so one that
    // attaches before then hears of it, and one after reads the purged file.
    struct TellAttached;
    impl Drop for TellAttached {
        fn drop(&mut self) {
            crate::terminal::mirror(|| crate::terminal::Mirrored::HistoryPurged);
        }
    }
    let _tell = TellAttached;
    let path = data_dir.join("input_history.txt");
    if !path.exists() {
        return Ok(0);
    }
    let text = fs::read_to_string(&path).map_err(|error| format!("cannot read it: {error}"))?;
    let mut kept = String::new();
    let mut removed = 0;
    for line in text.lines() {
        if doomed.contains(line.trim()) {
            removed += 1;
            continue;
        }
        kept.push_str(line);
        kept.push('\n');
    }
    if removed > 0 {
        let temp = path.with_extension("tmp");
        fs::write(&temp, kept.as_bytes()).map_err(|error| format!("cannot write it: {error}"))?;
        fs::rename(&temp, &path).map_err(|error| format!("cannot replace it: {error}"))?;
    }
    Ok(removed)
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    /// The store is a global, so the tests that use it run one at a time.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    struct Fixture {
        dir: PathBuf,
        memory: Hippocampus,
        _guard: std::sync::MutexGuard<'static, ()>,
        // A close consults the goal, a global too.
        _goal: std::sync::MutexGuard<'static, ()>,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let guard = ONE_AT_A_TIME.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let goal = crate::goal::TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let dir = std::env::temp_dir().join(format!("s0ma-job-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            let memory = Hippocampus::new(dir.join("hippocampus.jsonl")).unwrap();
            init(&dir);
            crate::goal::init(&dir);
            // Every global this module keeps, so one test cannot see another's.
            take_rotation_owed();
            UNREADABLE.store(false, std::sync::atomic::Ordering::SeqCst);
            Self { dir, memory, _guard: guard, _goal: goal }
        }

        /// One event from the operator, one from the agent, as a turn would leave.
        fn turn(&mut self, said: &str) {
            self.memory.remember_operator(said.to_string()).unwrap();
            self.memory.remember_s0ma(format!("about {said}")).unwrap();
        }

        fn run(&self, command: &str) -> std::result::Result<String, String> {
            handle_operator_command(&self.memory, command).expect("a /job command")
        }

        fn ok(&self, command: &str) -> String {
            self.run(command).unwrap_or_else(|error| panic!("{command}: {error}"))
        }

        fn content(&self, id: u64) -> String {
            self.memory.get_event(id).unwrap().expect("the event is still there").content
        }

        fn lines(&self) -> usize {
            fs::read_to_string(self.dir.join("hippocampus.jsonl")).unwrap().lines().count()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            *lock(&DIR) = None;
            *lock(&LOADED) = None;
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn a_close_redacts_the_events_and_keeps_every_id() {
        let mut fixture = Fixture::new("close");
        fixture.turn("ordinary work");
        fixture.ok("/job start acme");
        fixture.turn("the client's secret");
        let before = fixture.lines();
        let next_id = fixture.memory.last_event_id() + 1;

        let report = fixture.ok("/job close acme --yes");
        assert!(report.contains("closed"), "{report}");

        // The job's events: content gone, everything else kept.
        assert_eq!(fixture.content(3), "[redacted for job acme]");
        assert_eq!(fixture.content(4), "[redacted for job acme]");
        // Outside the job: untouched.
        assert_eq!(fixture.content(1), "ordinary work");
        assert_eq!(fixture.content(2), "about ordinary work");
        // An id is a line number, so the count may not move and the next event
        // must still take the id it would have.
        assert_eq!(fixture.lines(), before, "a redaction adds and removes no lines");
        fixture.turn("after the close");
        assert_eq!(fixture.content(next_id), "after the close", "the next id is unchanged");

        // Closed for good.
        assert_eq!(open_job(), None);
        assert_eq!(sleep_held(), None);
        assert!(fixture.run("/job resume acme").unwrap_err().contains("closed"));
        assert!(fixture.run("/job close acme").unwrap_err().contains("already closed"));
        let shown = fixture.ok("/job show acme");
        assert!(shown.contains("removed: 2 events"), "{shown}");
    }

    #[test]
    fn a_close_rotates_the_context_because_it_cannot_edit_it() {
        let fixture = Fixture::new("rotation");
        assert!(!take_rotation_owed());
        fixture.ok("/job start acme");
        fixture.ok("/job close acme --yes");
        assert!(take_rotation_owed(), "the live context still holds what the archive no longer does");
        assert!(!take_rotation_owed(), "and the caller is told once");
    }

    #[test]
    fn a_refused_goal_line_typed_during_the_job_goes_at_its_close() {
        let mut fixture = Fixture::new("refused-goal");
        fixture.ok("/job start acme");
        let line = "/goal --turn 5 fix the acme deploy pipeline";
        // As the prompt does, the line is in the history whether or not /goal took it.
        fs::write(fixture.dir.join("input_history.txt"), format!("{line}\n/goal pause\n")).unwrap();
        assert!(crate::goal::handle_command(line, None).unwrap().reply.is_err(), "--turn is not a flag");
        fixture.turn("the deploy key is in the vault");
        fixture.ok("/job close acme --yes");
        let left = fs::read_to_string(fixture.dir.join("input_history.txt")).unwrap();
        assert!(!left.contains("acme"), "{left}");
        assert!(left.contains("/goal pause"));
        assert!(!fs::read_to_string(fixture.dir.join("jobs.json")).unwrap().contains("acme deploy"), "not kept after");
    }

    #[test]
    fn a_name_that_would_not_survive_being_written_into_the_archive_is_refused() {
        let fixture = Fixture::new("names");
        for bad in ["/job start ../etc", "/job start with space", "/job start a\"quote"] {
            assert!(fixture.run(bad).is_err(), "{bad} was accepted");
        }
        assert!(fixture.run(&format!("/job start {}", "x".repeat(MAX_NAME + 1))).is_err());
        assert!(fixture.run("/job start").unwrap_err().contains("usage"));
        fixture.ok("/job start a-client_1.2");
    }

    /// The weakness log is append-only and hash-chained, and it stores an
    /// excerpt of every event it cites. Citing a job event would leave the only
    /// surviving copy of it in the one file a close cannot edit.
    #[test]
    fn a_weakness_cannot_cite_an_event_the_close_will_redact() {
        let mut fixture = Fixture::new("weakness");
        fixture.turn("ordinary work");
        fixture.ok("/job start acme");
        fixture.turn("the client's secret");

        let log = crate::weaknesses::WeaknessLog::beside_archive(fixture.memory.archive_path());
        let report = |events: &str| crate::types::ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: crate::types::FunctionCall {
                name: "weakness_report".into(),
                arguments: format!(
                    r#"{{"title":"stated something unsourced","behavior":"said it as fact",
                        "expected":"I say I do not know.",
                        "justification":"the reply states it with nothing in context behind it at all",
                        "event_ids":{events}}}"#
                ),
            },
        };

        let refused: serde_json::Value = serde_json::from_str(
            &crate::weaknesses::execute_with(&log, &fixture.memory, "m", &report("[4]")).unwrap(),
        )
        .unwrap();
        assert_eq!(refused["recorded"], false, "{refused}");
        let error = refused["error"].as_str().unwrap();
        assert!(error.contains("job acme"), "{error}");
        assert!(error.contains("excerpt"), "it says why, not just no: {error}");
        assert!(log.entries().unwrap().is_empty(), "nothing was written");

        // An event from outside the job is fine.
        let accepted: serde_json::Value = serde_json::from_str(
            &crate::weaknesses::execute_with(&log, &fixture.memory, "m", &report("[2]")).unwrap(),
        )
        .unwrap();
        assert_eq!(accepted["recorded"], true, "{accepted}");
    }
}
