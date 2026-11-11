//! Wall-clock time in a form the model can use without arithmetic.
//!
//! s0ma used to see the time only as `Utc::now().to_rfc3339()`: UTC, with
//! nanoseconds, no weekday and no local time. Converting that to a local date
//! and working out the weekday are exactly the steps models get wrong, and a
//! hard-to-read timestamp loses to the training-data prior about what year it
//! is. Everything here is pre-computed from the host clock: weekday, local time
//! with its UTC offset, and UTC.
//!
//! The header of every turn and the trailer on every tool result
//! (`observation_mask::trailer`) carry the current time, so it stays in view
//! through long tool loops and autonomous cycles instead of aging with the turn
//! header.
//!
//! # The harness does the arithmetic
//!
//! Turning two timestamps into "how long ago" is the step models are worst at
//! (duration questions score lowest in temporal-reasoning benchmarks, with
//! off-by-one-day the commonest error), so s0ma is never left to do it. Every
//! interval it needs is computed here and written in words: since the previous
//! input, since its last turn ended, since it last ran before a restart, how
//! long a tool took, how long a queued message waited. The rule for where each
//! kind goes:
//!
//! - **Absolute** times (`short`, `range`, `stored`) on everything that stays
//!   in context: gists, memory lines, threads, loops, recalled episodes. They
//!   stay true, and their bytes never change, so the prefix cache holds.
//! - **Intervals** (`span`) only in text written fresh: the turn header
//!   (`Arrival::header`) and tool trailers. Each is anchored to the time on
//!   the same line or message ("before this input", "after"), so it stays
//!   true when read later.
//!
//! # The operator's time zone
//!
//! The host's zone is not necessarily the operator's: travelling, his evening can be
//! the host's next morning, and "today" and "tomorrow" are his. `/tz` sets his
//! zone (an offset like `UTC-6`, or a name like `America/Denver` that follows
//! daylight saving) and saves it in `data/operator_tz`; without that,
//! `S0MA_OPERATOR_TZ` gives it. The turn header then shows his local time
//! beside the host's.
use chrono::{DateTime, FixedOffset, Local, Offset, TimeZone, Timelike, Utc};
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

pub fn now() -> DateTime<Local> {
    Local::now()
}

/// "Monday 5 October 2026, 11:05 local (UTC-07:00); 2026-10-05 18:05 UTC"
pub fn describe<Tz: TimeZone>(at: &DateTime<Tz>) -> String
where
    Tz::Offset: Display,
{
    format!(
        "{} local (UTC{}); {} UTC",
        at.format("%A %-d %B %Y, %H:%M"),
        at.format("%:z"),
        at.with_timezone(&Utc).format("%Y-%m-%d %H:%M"),
    )
}

/// "Mon 5 Oct 2026 11:05", for trailers and anything dated in context.
pub fn short<Tz: TimeZone>(at: &DateTime<Tz>) -> String
where
    Tz::Offset: Display,
{
    at.format("%a %-d %b %Y %H:%M").to_string()
}

/// An interval in words: "under 0.1 s", "4.2 s", "42 s", "4 min 12 s",
/// "34 min", "3 h 12 min", "2 days 3 h", "16 days". Coarser the longer it
/// is: seconds stop mattering past ten minutes, minutes past a day.
pub fn span(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    let plural = |n: u64, one: &str, many: &str| if n == 1 { format!("1 {one}") } else { format!("{n} {many}") };
    match seconds {
        0..=9 => {
            let tenths = elapsed.as_millis() / 100;
            if tenths == 0 {
                "under 0.1 s".to_owned()
            } else {
                format!("{}.{} s", tenths / 10, tenths % 10)
            }
        }
        10..=59 => format!("{seconds} s"),
        60..=599 => match seconds % 60 {
            0 => format!("{} min", seconds / 60),
            s => format!("{} min {s} s", seconds / 60),
        },
        600..=3_599 => format!("{} min", seconds / 60),
        3_600..=86_399 => match (seconds % 3_600) / 60 {
            0 => format!("{} h", seconds / 3_600),
            m => format!("{} h {m} min", seconds / 3_600),
        },
        _ => {
            let days = seconds / 86_400;
            let hours = (seconds % 86_400) / 3_600;
            if days >= 7 || hours == 0 {
                plural(days, "day", "days")
            } else {
                format!("{} {hours} h", plural(days, "day", "days"))
            }
        }
    }
}

/// From `earlier` to `later`; zero if `later` isn't later.
pub fn between<A: TimeZone, B: TimeZone>(earlier: &DateTime<A>, later: &DateTime<B>) -> Duration {
    (later.with_timezone(&Utc) - earlier.with_timezone(&Utc)).to_std().unwrap_or_default()
}

/// The interval between two times as `short` shows them: from their minutes
/// once it is a minute or more, so "10:53" and "14:05" are always 3 h 12 min
/// apart, never 3 h 11 min because of the seconds nobody sees. Under a
/// minute, exact.
pub fn between_shown<A: TimeZone, B: TimeZone>(earlier: &DateTime<A>, later: &DateTime<B>) -> Duration {
    let exact = between(earlier, later);
    if exact < Duration::from_secs(60) {
        return exact;
    }
    let minute = |at: DateTime<Utc>| at.with_second(0).and_then(|at| at.with_nanosecond(0)).unwrap_or(at);
    between(&minute(earlier.with_timezone(&Utc)), &minute(later.with_timezone(&Utc)))
}

/// A stored RFC 3339 timestamp, in the host's zone.
pub fn parse(raw: &str) -> Option<DateTime<Local>> {
    DateTime::parse_from_rfc3339(raw.trim()).ok().map(|at| at.with_timezone(&Local))
}

/// A stored timestamp as `short` writes it ("Tue 21 Oct 2026 21:42", host
/// time), or the raw text if it doesn't parse.
pub fn stored(raw: &str) -> String {
    parse(raw).map_or_else(|| raw.to_owned(), |at| short(&at))
}

/// A stretch of time: one moment if both ends fall in the same minute,
/// "Tue 21 Oct 2026 14:05–16:40" within a day, both ends in full otherwise.
pub fn range<Tz: TimeZone>(from: &DateTime<Tz>, through: &DateTime<Tz>) -> String
where
    Tz::Offset: Display,
{
    let (first, last) = if through < from { (through, from) } else { (from, through) };
    if first.format("%Y-%m-%d %H:%M").to_string() == last.format("%Y-%m-%d %H:%M").to_string() {
        short(first)
    } else if first.date_naive() == last.date_naive() {
        format!("{}–{}", short(first), last.format("%H:%M"))
    } else {
        format!("{} – {}", short(first), short(last))
    }
}

// ------------------------------------------------------------------ The operator's zone

/// Who the operator is, as the rest of the harness names him.
pub const OPERATOR: &str = "The operator";
/// Read at start when `/tz` hasn't saved a zone.
pub const ZONE_ENV: &str = "S0MA_OPERATOR_TZ";
/// In the data directory: what `/tz` last set ("off" when turned off).
const ZONE_FILE: &str = "operator_tz";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Zone {
    /// A fixed offset: "UTC-6". Doesn't follow daylight saving.
    Fixed(FixedOffset),
    /// A named zone: "America/Denver". Follows daylight saving.
    Named(chrono_tz::Tz),
}

impl Zone {
    /// An offset ("UTC-6", "utc - 6", "GMT+5:30", "-06:00", "+0530",
    /// "UTC") or a zone name ("America/Denver", any case).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let spec = spec.trim();
        let compact: String = spec.chars().filter(|c| !c.is_whitespace()).collect();
        if compact.is_empty() {
            return Err("no zone given".to_owned());
        }
        if let Some(offset) = parse_offset(&compact)? {
            return Ok(Self::Fixed(offset));
        }
        if let Ok(zone) = spec.parse::<chrono_tz::Tz>() {
            return Ok(Self::Named(zone));
        }
        let wanted = spec.to_ascii_lowercase();
        chrono_tz::TZ_VARIANTS
            .iter()
            .find(|zone| zone.name().to_ascii_lowercase() == wanted)
            .map(|zone| Self::Named(*zone))
            .ok_or_else(|| {
                format!(
                    "'{spec}' is neither an offset (UTC-6, -06:00, UTC+5:30) nor a zone name \
                     (America/Denver, Europe/London)"
                )
            })
    }

    fn offset_at(&self, at: &DateTime<Utc>) -> FixedOffset {
        match self {
            Self::Fixed(offset) => *offset,
            Self::Named(zone) => zone.offset_from_utc_datetime(&at.naive_utc()).fix(),
        }
    }

    /// How `/tz` saves it: "UTC-06:00", or the zone's name.
    fn spec(&self) -> String {
        match self {
            Self::Fixed(offset) => utc_offset(*offset),
            Self::Named(zone) => zone.name().to_owned(),
        }
    }

    /// "UTC-06:00", or "America/Denver, UTC-06:00" (its offset at `at`).
    fn label_at(&self, at: &DateTime<Utc>) -> String {
        let offset = utc_offset(self.offset_at(at));
        match self {
            Self::Fixed(_) => offset,
            Self::Named(zone) => format!("{}, {offset}", zone.name()),
        }
    }

    /// "Wednesday 21 October 2026, 21:05 (UTC-06:00)": `at` in this zone.
    pub fn describe_at<Tz: TimeZone>(&self, at: &DateTime<Tz>) -> String {
        let utc = at.with_timezone(&Utc);
        let there = utc.with_timezone(&self.offset_at(&utc));
        format!("{} ({})", there.format("%A %-d %B %Y, %H:%M"), self.label_at(&utc))
    }
}

/// "UTC+11:00", "UTC-06:00", "UTC+05:30".
fn utc_offset(offset: FixedOffset) -> String {
    let seconds = offset.local_minus_utc();
    let sign = if seconds < 0 { '-' } else { '+' };
    let minutes = seconds.unsigned_abs() / 60;
    format!("UTC{sign}{:02}:{:02}", minutes / 60, minutes % 60)
}

/// `Ok(None)`: not an offset (try it as a name). `Err`: an offset, but not a
/// valid one.
fn parse_offset(compact: &str) -> Result<Option<FixedOffset>, String> {
    let upper = compact.to_ascii_uppercase();
    if matches!(upper.as_str(), "UTC" | "GMT" | "Z") {
        return Ok(FixedOffset::east_opt(0));
    }
    let rest = upper.strip_prefix("UTC").or_else(|| upper.strip_prefix("GMT")).unwrap_or(&upper);
    let (sign, digits) = if let Some(digits) = rest.strip_prefix('+') {
        (1, digits)
    } else if let Some(digits) = rest.strip_prefix('-').or_else(|| rest.strip_prefix('\u{2212}')) {
        (-1, digits)
    } else {
        return Ok(None);
    };
    let bad = || format!("'{compact}' isn't an offset: use hours, or hours and minutes (UTC-6, UTC+5:30, -0600)");
    // Byte positions below are only char boundaries in ASCII.
    if !digits.is_ascii() {
        return Err(bad());
    }
    let (hours, minutes) = match digits.split_once(':') {
        Some((hours, minutes)) => (hours, minutes),
        None if digits.len() <= 2 => (digits, "0"),
        None if digits.len() <= 4 => digits.split_at(digits.len() - 2),
        None => return Err(bad()),
    };
    if hours.is_empty() || !hours.chars().chain(minutes.chars()).all(|c| c.is_ascii_digit()) {
        return Err(bad());
    }
    let (hours, minutes): (i32, i32) = (hours.parse().map_err(|_| bad())?, minutes.parse().map_err(|_| bad())?);
    if hours > 14 || minutes >= 60 || (hours == 14 && minutes > 0) {
        return Err(format!("'{compact}' is out of range: offsets run from UTC-14:00 to UTC+14:00"));
    }
    Ok(FixedOffset::east_opt(sign * (hours * 3_600 + minutes * 60)))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ZoneSource {
    /// Set with `/tz`, saved in the data directory.
    Command,
    /// From `S0MA_OPERATOR_TZ`.
    Env,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OperatorZone {
    zone: Zone,
    from: ZoneSource,
}

static OPERATOR_ZONE: Mutex<Option<OperatorZone>> = Mutex::new(None);
static ZONE_PATH: OnceLock<PathBuf> = OnceLock::new();

fn operator_zone() -> Option<OperatorZone> {
    OPERATOR_ZONE.lock().map_or_else(|poisoned| *poisoned.into_inner(), |zone| *zone)
}

/// The operator's time zone as `/tz` takes it (`UTC-6`, `America/Denver`), if set.
pub fn operator_zone_spec() -> Option<String> {
    operator_zone().map(|operator| operator.zone.spec())
}

fn set_operator_zone(zone: Option<OperatorZone>) {
    match OPERATOR_ZONE.lock() {
        Ok(mut current) => *current = zone,
        Err(poisoned) => *poisoned.into_inner() = zone,
    }
}

/// At start: the zone `/tz` saved, or else `S0MA_OPERATOR_TZ`. Returns a
/// warning when one is set but doesn't parse.
pub fn init_operator_zone(data_dir: &Path) -> Option<String> {
    let path = data_dir.join(ZONE_FILE);
    let saved = std::fs::read_to_string(&path).ok();
    let _ = ZONE_PATH.set(path);
    let (zone, warning) = choose_zone(saved.as_deref(), std::env::var(ZONE_ENV).ok().as_deref());
    set_operator_zone(zone);
    warning
}

/// What `/tz` saved wins over the environment, including "off".
fn choose_zone(saved: Option<&str>, env: Option<&str>) -> (Option<OperatorZone>, Option<String>) {
    let mut warning = None;
    if let Some(saved) = saved.map(str::trim).filter(|saved| !saved.is_empty()) {
        if saved == "off" {
            return (None, None);
        }
        match Zone::parse(saved) {
            Ok(zone) => return (Some(OperatorZone { zone, from: ZoneSource::Command }), None),
            Err(error) => {
                warning = Some(format!("[tz] the zone /tz saved doesn't read back ({error}); /tz sets it again"));
            }
        }
    }
    match env.map(str::trim).filter(|env| !env.is_empty()) {
        Some(env) => match Zone::parse(env) {
            Ok(zone) => (Some(OperatorZone { zone, from: ZoneSource::Env }), warning),
            Err(error) => (None, Some(format!("[tz] {ZONE_ENV} ignored: {error}"))),
        },
        None => (None, warning),
    }
}

/// "The operator's local time: Wednesday 21 October 2026, 21:05 (UTC-06:00)", once
/// his zone is set.
pub fn operator_line<Tz: TimeZone>(at: &DateTime<Tz>) -> Option<String> {
    operator_zone().map(|operator| format!("{OPERATOR}'s local time: {}", operator.zone.describe_at(at)))
}

/// `/tz`: show the operator's zone; `/tz <zone>` sets it; `/tz off` turns it off.
pub fn handle_command(command: &str) -> Option<Result<String, String>> {
    let mut words = command.split_whitespace();
    if words.next()? != "/tz" {
        return None;
    }
    let rest = words.collect::<Vec<_>>().join(" ");
    Some(tz_command(&rest, &now()))
}

fn tz_command(rest: &str, at: &DateTime<Local>) -> Result<String, String> {
    let host = utc_offset(at.offset().fix());
    match rest.trim() {
        "" => Ok(match operator_zone() {
            Some(operator) => format!(
                "[tz] Your time zone is {} ({}): it is {} there; the host is {host}.\n\
                 /tz <zone> changes it, /tz off turns it off.",
                operator.zone.spec(),
                match operator.from {
                    ZoneSource::Command => "set with /tz",
                    ZoneSource::Env => "from S0MA_OPERATOR_TZ",
                },
                operator.zone.describe_at(at),
            ),
            None => format!(
                "[tz] Your time zone is not set, so s0ma sees only the host's ({host}).\n\
                 /tz UTC-6 sets an offset; /tz America/Denver sets a zone that follows daylight saving."
            ),
        }),
        "off" | "none" => {
            save_zone("off")?;
            set_operator_zone(None);
            Ok(format!("[tz] Your time zone is off: s0ma sees only the host's time ({host})."))
        }
        spec => {
            let zone = Zone::parse(spec).map_err(|error| format!("[tz] {error}"))?;
            save_zone(&zone.spec())?;
            set_operator_zone(Some(OperatorZone { zone, from: ZoneSource::Command }));
            Ok(format!(
                "[tz] Your time zone is now {}: it is {} there. s0ma sees it from its next input.",
                zone.spec(),
                zone.describe_at(at),
            ))
        }
    }
}

fn save_zone(text: &str) -> Result<(), String> {
    match ZONE_PATH.get() {
        Some(path) => save_zone_to(path, text),
        // Not started from App::load (a test): kept for this process only.
        None => Ok(()),
    }
}

/// Whole or not at all: written beside it, then renamed over it.
fn save_zone_to(path: &Path, text: &str) -> Result<(), String> {
    crate::util::write_atomic(path, format!("{text}\n"))
        .map_err(|error| format!("[tz] not saved to {}: {error}", path.display()))
}

// ------------------------------------------------------------------ arrivals

/// The last thing the archive recorded before this process started.
#[derive(Clone, Debug, PartialEq)]
pub struct Recorded {
    pub at: DateTime<Local>,
    /// It was a deep sleep's manifest: s0ma's last act was consolidating.
    pub was_sleep: bool,
}

/// Why the working context was rebuilt between two inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rebuild {
    /// A deep sleep (`/sleep`, or the daily one).
    Sleep,
    /// `/compact`: rotated onto the compacted history.
    Compaction,
}

/// What came before the next input, for its header. Only inputs s0ma
/// actually takes in (`accept`) move it: one refused before reaching the
/// model (a router failure, a full context) leaves it as it was, so the
/// message sent again still says what the first would have.
struct Timeline {
    started: Option<DateTime<Local>>,
    before_start: Option<Recorded>,
    previous_input: Option<DateTime<Local>>,
    /// The last input that was the operator's own.
    last_operator: Option<DateTime<Local>>,
    turn_ended: Option<DateTime<Local>>,
    /// A turn was accepted and hasn't ended: only then does an end count.
    turn_open: bool,
    /// The context was rebuilt since the previous input: when, and why.
    rebuilt: Option<(DateTime<Local>, Rebuild)>,
    /// The working context was resumed at start as it was saved before the
    /// restart (`context_resume`), not rebuilt.
    resumed: Option<Resumed>,
}

/// The working context came back from before the restart: when it was saved,
/// and the archive events that came after the save (not in it), if any.
#[derive(Clone, Debug, PartialEq)]
pub struct Resumed {
    pub saved_at: DateTime<Local>,
    pub missing: Option<(u64, u64)>,
}

static TIMELINE: Mutex<Timeline> = Mutex::new(Timeline {
    started: None,
    before_start: None,
    previous_input: None,
    last_operator: None,
    turn_ended: None,
    turn_open: false,
    rebuilt: None,
    resumed: None,
});

fn timeline<T>(act: impl FnOnce(&mut Timeline) -> T) -> T {
    match TIMELINE.lock() {
        Ok(mut timeline) => act(&mut timeline),
        Err(poisoned) => act(&mut poisoned.into_inner()),
    }
}

/// At start, with the last event the archive holds (None if it is empty).
pub fn mark_start(before: Option<Recorded>) {
    timeline(|timeline| {
        timeline.started = Some(now());
        timeline.before_start = before;
    });
}

/// At start, when the working context was resumed rather than rebuilt.
pub fn mark_resumed(resumed: Resumed) {
    timeline(|timeline| timeline.resumed = Some(resumed));
}

/// After every turn, however it ended. Counts only for a turn that was
/// accepted: one refused before reaching the model, or a bad `/img`, wasn't
/// a turn s0ma took.
pub fn mark_turn_end() {
    timeline(|timeline| {
        if std::mem::take(&mut timeline.turn_open) {
            timeline.turn_ended = Some(now());
        }
    });
}

/// After the working context was rebuilt: a deep sleep, or `/compact`.
pub fn mark_rebuilt(why: Rebuild) {
    timeline(|timeline| {
        timeline.rebuilt = Some((now(), why));
        // Whatever was resumed at start is not what is in front of s0ma now.
        timeline.resumed = None;
    });
}

/// Whether an input starts a turn or joins the one running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Joins {
    NewTurn,
    RunningTurn,
}

/// Whose input it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Who {
    /// The operator's own message (typed, spoken, or from an attached client).
    Operator,
    /// Anything else: loop continuations, worker results, VM wake events.
    Harness,
}

/// One input's arrival, with what its header says about the time before it.
#[derive(Clone, Debug)]
pub struct Arrival {
    pub at: DateTime<Local>,
    joins: Joins,
    from: Who,
    /// When it was sent, if it waited in the queue before s0ma took it.
    sent_at: Option<DateTime<Local>>,
    previous_input: Option<DateTime<Local>>,
    /// The operator's last message before this one.
    last_operator: Option<DateTime<Local>>,
    /// When s0ma's last turn ended (a new turn only).
    turn_ended: Option<DateTime<Local>>,
    /// The first input since this process started: when it started, and the
    /// archive's last event before that.
    first: Option<(Option<DateTime<Local>>, Option<Recorded>)>,
    rebuilt: Option<(DateTime<Local>, Rebuild)>,
    /// With the first input: the context was resumed from before the restart.
    resumed: Option<Resumed>,
    operator: Option<String>,
}

/// An input arriving now, and what its header says. `sent_at`: when it was
/// typed or sent, if that was before s0ma took it (a queued message).
///
/// Changes nothing: `accept` records it once s0ma takes it in, so an input
/// refused on the way (and sent again) doesn't use up what the next header
/// has to say.
pub fn arrive(joins: Joins, from: Who, sent_at: Option<DateTime<Local>>) -> Arrival {
    let at = now();
    timeline(|timeline| Arrival {
        at,
        joins,
        from,
        sent_at,
        previous_input: timeline.previous_input,
        last_operator: timeline.last_operator,
        turn_ended: if joins == Joins::NewTurn { timeline.turn_ended } else { None },
        first: timeline
            .previous_input
            .is_none()
            .then(|| (timeline.started, timeline.before_start.clone())),
        rebuilt: timeline.rebuilt,
        resumed: if timeline.previous_input.is_none() { timeline.resumed.clone() } else { None },
        operator: operator_line(&at),
    })
}

/// The input is in s0ma's context now: the next header counts from it, and
/// what this one said about a rebuild isn't said again.
pub fn accept(arrival: &Arrival) {
    timeline(|timeline| {
        timeline.previous_input = Some(arrival.at);
        if arrival.from == Who::Operator {
            timeline.last_operator = Some(arrival.at);
        }
        if arrival.rebuilt.is_some() && timeline.rebuilt == arrival.rebuilt {
            timeline.rebuilt = None;
        }
        if arrival.joins == Joins::NewTurn {
            timeline.turn_open = true;
        }
    });
}

/// Below this, a queued message's wait isn't worth a line.
const SHOW_WAIT_AFTER: Duration = Duration::from_secs(60);
/// A turn that ended this soon after the previous input adds nothing to it.
const SHOW_TURN_END_AFTER_INPUT: Duration = Duration::from_secs(120);
/// Nor one that ended this recently (loop mode's next cycle).
const SHOW_TURN_END_IDLE: Duration = Duration::from_secs(60);

impl Arrival {
    /// The header's time lines: the current time, the operator's, and the intervals
    /// since what came before, each computed here.
    pub fn header(&self) -> String {
        let at = &self.at;
        let before = |then: &DateTime<Local>| span(between_shown(then, at));
        let mut lines = vec![format!("Current time: {}", describe(at))];
        if let Some(operator) = &self.operator {
            lines.push(operator.clone());
        }
        if let Some(sent) = self.sent_at.filter(|sent| between_shown(sent, at) >= SHOW_WAIT_AFTER) {
            lines.push(format!(
                "Sent at {}: it waited {} in the queue while you worked.",
                short(&sent),
                before(&sent)
            ));
        }
        match (&self.previous_input, &self.first) {
            (Some(previous), _) => {
                let mut line = format!("Previous input: {}, {} before this one.", short(previous), before(previous));
                if let Some(ended) = self.turn_ended.filter(|ended| {
                    between_shown(previous, ended) >= SHOW_TURN_END_AFTER_INPUT
                        && between_shown(ended, at) >= SHOW_TURN_END_IDLE
                }) {
                    line.push_str(&format!(
                        " Your last turn ended {}, {} before this input.",
                        short(&ended),
                        before(&ended)
                    ));
                }
                lines.push(line);
            }
            (None, Some((started, recorded))) => {
                let mut line = match started {
                    Some(started) => format!(
                        "This is the first input since s0ma started at {}, {} before it.",
                        short(started),
                        before(started)
                    ),
                    None => "This is the first input since s0ma started.".to_owned(),
                };
                if let Some(recorded) = recorded {
                    line.push_str(&format!(
                        " Before that, the archive's last record {}{}, {} before this input.",
                        if recorded.was_sleep { "was a deep sleep that finished " } else { "is from " },
                        short(&recorded.at),
                        before(&recorded.at),
                    ));
                }
                if let Some(resumed) = &self.resumed {
                    line.push_str(&format!(
                        " Your working context is the one saved at {}, {} before this input: what was in front \
                         of you before the restart still is.",
                        short(&resumed.saved_at),
                        before(&resumed.saved_at),
                    ));
                    if let Some((from, through)) = resumed.missing {
                        let (events, are, them) = if from == through {
                            (format!("event #{from}"), "is", "it")
                        } else {
                            (format!("events #{from}–#{through}"), "are", "them")
                        };
                        line.push_str(&format!(
                            " Archive {events} came after that save and {are} not in it (a turn the restart cut \
                             short, or work in the background); archive_get reads {them}."
                        ));
                    }
                }
                lines.push(line);
            }
            (None, None) => {}
        }
        // The operator's last message, when other inputs came after it (loop cycles,
        // worker results, VM events): how long since he wrote is its own fact.
        if let Some(wrote) = self.last_operator.filter(|wrote| Some(*wrote) != self.previous_input) {
            lines.push(format!(
                "{OPERATOR}'s last message before this was at {}, {} before this input.",
                short(&wrote),
                before(&wrote)
            ));
        }
        if let Some((rebuilt, why)) = &self.rebuilt {
            lines.push(format!(
                "{} at {}, {} before this input: your context was rebuilt then, so what came before it is in \
                 memory now, not in front of you.",
                match why {
                    Rebuild::Sleep => "A deep sleep finished",
                    Rebuild::Compaction => "Your context was compacted",
                },
                short(rebuilt),
                before(rebuilt)
            ));
        }
        let mut instruction = String::from(
            "This is when this input arrived, from the host clock. Use it, not your training data, for today's \
             date and year, and use the intervals given here rather than working them out; tool results show \
             their own completion time and duration as the turn goes on.",
        );
        if self.operator.is_some() {
            instruction.push_str(&format!(
                " Every other time in your context (tool results, memory, loops, threads) is host time, like the \
                 current time above. {OPERATOR}'s local time is his: when he says today, tonight or tomorrow, he \
                 means his date."
            ));
        }
        lines.push(instruction);
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    fn pacific(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<FixedOffset> {
        FixedOffset::west_opt(7 * 3600).unwrap().with_ymd_and_hms(y, mo, d, h, mi, 30).unwrap()
    }

    fn local(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Local> {
        Local.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
    }

    fn arrival(at: DateTime<Local>) -> Arrival {
        Arrival {
            at,
            joins: Joins::NewTurn,
            from: Who::Operator,
            sent_at: None,
            previous_input: None,
            last_operator: None,
            turn_ended: None,
            first: None,
            rebuilt: None,
            resumed: None,
            operator: None,
        }
    }

    /// Tests that use the process-wide zone or timeline take turns.
    static GLOBALS: Mutex<()> = Mutex::new(());

    fn globals() -> std::sync::MutexGuard<'static, ()> {
        GLOBALS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn reset_timeline() {
        timeline(|timeline| {
            *timeline = Timeline {
                started: None,
                before_start: None,
                previous_input: None,
                last_operator: None,
                turn_ended: None,
                turn_open: false,
                rebuilt: None,
                resumed: None,
            }
        });
    }

    #[test]
    fn describe_gives_weekday_local_offset_and_utc() {
        assert_eq!(
            describe(&pacific(2026, 10, 5, 11, 5)),
            "Monday 5 October 2026, 11:05 local (UTC-07:00); 2026-10-05 18:05 UTC"
        );
    }

    #[test]
    fn utc_date_can_differ_from_local_date() {
        // Late evening local is already tomorrow in UTC; both are shown.
        assert_eq!(
            describe(&pacific(2026, 12, 31, 20, 0)),
            "Thursday 31 December 2026, 20:00 local (UTC-07:00); 2027-01-01 03:00 UTC"
        );
    }

    #[test]
    fn short_form_keeps_weekday_date_and_year() {
        assert_eq!(short(&pacific(2026, 10, 5, 18, 42)), "Mon 5 Oct 2026 18:42");
    }

    #[test]
    fn local_now_formats() {
        let line = describe(&now());
        assert!(line.contains(" local (UTC") && line.ends_with(" UTC"), "{line}");
    }

    #[test]
    fn spans_get_coarser_as_they_grow() {
        let s = Duration::from_secs;
        assert_eq!(span(Duration::from_millis(40)), "under 0.1 s");
        assert_eq!(span(Duration::from_millis(4_260)), "4.2 s");
        assert_eq!(span(s(42)), "42 s");
        assert_eq!(span(s(252)), "4 min 12 s");
        assert_eq!(span(s(240)), "4 min");
        assert_eq!(span(s(34 * 60 + 59)), "34 min");
        assert_eq!(span(s(3 * 3600 + 12 * 60 + 9)), "3 h 12 min");
        assert_eq!(span(s(5 * 3600)), "5 h");
        assert_eq!(span(s(86_400 + 3 * 3600)), "1 day 3 h");
        assert_eq!(span(s(2 * 86_400 + 3 * 3600 + 59 * 60)), "2 days 3 h");
        assert_eq!(span(s(2 * 86_400)), "2 days");
        assert_eq!(span(s(16 * 86_400 + 5 * 3600)), "16 days");
    }

    #[test]
    fn an_interval_never_goes_negative() {
        let (early, late) = (pacific(2026, 10, 5, 9, 0), pacific(2026, 10, 5, 9, 30));
        assert_eq!(between(&early, &late), Duration::from_secs(30 * 60));
        assert_eq!(between(&late, &early), Duration::ZERO);
    }

    #[test]
    fn ranges_say_as_little_as_they_need_to() {
        assert_eq!(range(&pacific(2026, 10, 21, 14, 5), &pacific(2026, 10, 21, 14, 5)), "Wed 21 Oct 2026 14:05");
        assert_eq!(
            range(&pacific(2026, 10, 21, 14, 5), &pacific(2026, 10, 21, 16, 40)),
            "Wed 21 Oct 2026 14:05–16:40"
        );
        assert_eq!(
            range(&pacific(2026, 10, 21, 22, 5), &pacific(2026, 10, 22, 9, 10)),
            "Wed 21 Oct 2026 22:05 – Thu 22 Oct 2026 09:10"
        );
        // Ends given the wrong way round still read forwards.
        assert_eq!(
            range(&pacific(2026, 10, 21, 16, 40), &pacific(2026, 10, 21, 14, 5)),
            "Wed 21 Oct 2026 14:05–16:40"
        );
    }

    #[test]
    fn stored_stamps_read_like_the_rest_and_bad_ones_pass_through() {
        let at = parse("2026-10-21T21:42:26.479444669+11:00").unwrap();
        assert_eq!(stored("2026-10-21T21:42:26.479444669+11:00"), short(&at));
        assert!(!stored("2026-10-21T21:42:26.479444669+11:00").contains(":26."), "no seconds, no RFC 3339");
        assert_eq!(stored("t"), "t");
    }

    #[test]
    fn zones_parse_as_people_type_them() {
        let fixed = |hours: i32, minutes: i32| Zone::Fixed(FixedOffset::east_opt(hours * 3600 + minutes * 60).unwrap());
        assert_eq!(Zone::parse("UTC-6"), Ok(fixed(-6, 0)));
        assert_eq!(Zone::parse("utc - 6"), Ok(fixed(-6, 0)), "as the operator wrote it");
        assert_eq!(Zone::parse("-06:00"), Ok(fixed(-6, 0)));
        assert_eq!(Zone::parse("GMT+5:30"), Ok(fixed(5, 30)));
        assert_eq!(Zone::parse("+0530"), Ok(fixed(5, 30)));
        assert_eq!(Zone::parse("UTC+11"), Ok(fixed(11, 0)));
        assert_eq!(Zone::parse("UTC"), Ok(fixed(0, 0)));
        assert_eq!(Zone::parse("America/Denver"), Ok(Zone::Named(chrono_tz::America::Denver)));
        assert_eq!(Zone::parse("america/mexico_city"), Ok(Zone::Named(chrono_tz::America::Mexico_City)));
        // POSIX names keep their inverted sign: Etc/GMT+6 is six hours behind.
        assert_eq!(Zone::parse("Etc/GMT+6"), Ok(Zone::Named(chrono_tz::Etc::GMTPlus6)));
        assert!(Zone::parse("UTC-15").is_err());
        assert!(Zone::parse("UTC-6:75").is_err());
        assert!(Zone::parse("Mars/Olympus").is_err());
        assert!(Zone::parse("  ").is_err());
        // Not ASCII: refused, not a panic on a byte index.
        assert!(Zone::parse("+é1").is_err());
        assert!(Zone::parse("UTC+1€").is_err());
        assert!(Zone::parse("−6").is_ok(), "a typographic minus is still a minus");
    }

    #[test]
    fn a_zone_shows_its_own_date_which_can_differ_from_the_hosts() {
        // 14:05 in Sydney (UTC+11) is still the evening before at UTC-6.
        let sydney = FixedOffset::east_opt(11 * 3600).unwrap().with_ymd_and_hms(2026, 10, 22, 14, 5, 0).unwrap();
        assert_eq!(
            Zone::parse("UTC-6").unwrap().describe_at(&sydney),
            "Wednesday 21 October 2026, 21:05 (UTC-06:00)"
        );
        // A named zone follows daylight saving: Denver is UTC-6 in October and
        // UTC-7 after the first Sunday of November.
        let denver = Zone::parse("America/Denver").unwrap();
        assert_eq!(denver.describe_at(&sydney), "Wednesday 21 October 2026, 21:05 (America/Denver, UTC-06:00)");
        let december = Utc.with_ymd_and_hms(2026, 12, 1, 3, 5, 0).unwrap();
        assert!(denver.describe_at(&december).ends_with("(America/Denver, UTC-07:00)"));
    }

    #[test]
    fn what_tz_saved_wins_over_the_environment() {
        let zone = |spec: &str| Zone::parse(spec).unwrap();
        assert_eq!(
            choose_zone(Some("UTC-06:00\n"), Some("Europe/London")),
            (Some(OperatorZone { zone: zone("UTC-6"), from: ZoneSource::Command }), None)
        );
        assert_eq!(choose_zone(Some("off"), Some("Europe/London")), (None, None), "off stays off");
        assert_eq!(
            choose_zone(None, Some("Europe/London")),
            (Some(OperatorZone { zone: zone("Europe/London"), from: ZoneSource::Env }), None)
        );
        assert_eq!(choose_zone(None, None), (None, None));
        let (fallback, warning) = choose_zone(Some("garbage"), Some("UTC-6"));
        assert_eq!(fallback, Some(OperatorZone { zone: zone("UTC-6"), from: ZoneSource::Env }));
        assert!(warning.unwrap().contains("doesn't read back"));
        let (none, warning) = choose_zone(None, Some("Mars/Olympus"));
        assert_eq!(none, None);
        assert!(warning.unwrap().contains("S0MA_OPERATOR_TZ ignored"));
    }

    #[test]
    fn a_saved_zone_reads_back_as_it_was_set() {
        let dir = std::env::temp_dir().join(format!("s0ma-tz-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(ZONE_FILE);
        let zone = Zone::parse("America/Denver").unwrap();
        save_zone_to(&path, &zone.spec()).unwrap();
        let saved = std::fs::read_to_string(&path).unwrap();
        assert_eq!(choose_zone(Some(&saved), Some("UTC+3")).0, Some(OperatorZone { zone, from: ZoneSource::Command }));
        save_zone_to(&path, "off").unwrap();
        let saved = std::fs::read_to_string(&path).unwrap();
        assert_eq!(choose_zone(Some(&saved), Some("UTC+3")), (None, None));
        assert!(!path.with_extension("tmp").exists(), "renamed into place");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tz_sets_shows_and_turns_off_the_zone() {
        let _globals = globals();
        let at = local(2026, 10, 22, 14, 5);
        assert!(handle_command("/tzx").is_none());
        let set = tz_command("utc - 6", &at).unwrap();
        assert!(set.contains("is now UTC-06:00"), "{set}");
        assert!(operator_line(&at).unwrap().starts_with("The operator's local time: "));
        let shown = tz_command("", &at).unwrap();
        assert!(shown.contains("UTC-06:00 (set with /tz)"), "{shown}");
        assert!(tz_command("Mars/Olympus", &at).unwrap_err().starts_with("[tz] "));
        assert!(operator_line(&at).is_some(), "a bad zone changes nothing");
        assert!(tz_command("off", &at).unwrap().contains("is off"));
        assert!(operator_line(&at).is_none());
        assert!(tz_command("", &at).unwrap().starts_with("[tz] Your time zone is not set"));
    }

    #[test]
    fn the_header_gives_the_intervals_instead_of_the_arithmetic() {
        let at = local(2026, 10, 22, 14, 5);
        let header = Arrival {
            previous_input: Some(local(2026, 10, 22, 10, 53)),
            turn_ended: Some(local(2026, 10, 22, 11, 40)),
            ..arrival(at)
        }
        .header();
        assert!(header.starts_with(&format!("Current time: {}\n", describe(&at))), "{header}");
        assert!(
            header.contains(
                "Previous input: Thu 22 Oct 2026 10:53, 3 h 12 min before this one. \
                 Your last turn ended Thu 22 Oct 2026 11:40, 2 h 25 min before this input."
            ),
            "{header}"
        );
        assert!(header.ends_with("as the turn goes on."), "{header}");
        assert!(header.contains("not your training data"));
        assert!(!header.contains("local time is his"), "no zone set, no note about it");
    }

    #[test]
    fn a_turn_end_that_adds_nothing_is_left_out() {
        let at = local(2026, 10, 22, 14, 5);
        // A quick reply: the previous input says it all.
        let quick = Arrival {
            previous_input: Some(local(2026, 10, 22, 10, 53)),
            turn_ended: Some(local(2026, 10, 22, 10, 54)),
            ..arrival(at)
        };
        assert!(!quick.header().contains("last turn ended"));
        // Loop mode: the next cycle starts as the last one ends.
        let looping = Arrival {
            previous_input: Some(local(2026, 10, 22, 13, 50)),
            turn_ended: Some(at),
            ..arrival(at)
        };
        assert!(!looping.header().contains("last turn ended"));
    }

    #[test]
    fn the_first_input_after_a_start_says_how_long_s0ma_was_away() {
        let at = local(2026, 10, 22, 14, 5);
        let header = Arrival {
            first: Some((
                Some(local(2026, 10, 22, 13, 58)),
                Some(Recorded { at: local(2026, 10, 21, 18, 40), was_sleep: false }),
            )),
            ..arrival(at)
        }
        .header();
        assert!(
            header.contains(
                "This is the first input since s0ma started at Thu 22 Oct 2026 13:58, 7 min before it. \
                 Before that, the archive's last record is from Wed 21 Oct 2026 18:40, 19 h 25 min before this input."
            ),
            "{header}"
        );
        let after_sleep = Arrival {
            first: Some((None, Some(Recorded { at: local(2026, 10, 22, 3, 0), was_sleep: true }))),
            ..arrival(at)
        }
        .header();
        assert!(!after_sleep.contains("last message before this"), "nothing known of the operator yet");
        assert!(after_sleep.contains("last record was a deep sleep that finished Thu 22 Oct 2026 03:00, 11 h 5 min"), "{after_sleep}");
    }

    #[test]
    fn a_queued_message_says_how_long_it_waited() {
        let at = local(2026, 10, 22, 12, 9);
        let waited = Arrival { sent_at: Some(local(2026, 10, 22, 12, 4)), ..arrival(at) }.header();
        assert!(waited.contains("Sent at Thu 22 Oct 2026 12:04: it waited 5 min in the queue while you worked."), "{waited}");
        let prompt = Arrival { sent_at: Some(at), ..arrival(at) }.header();
        assert!(!prompt.contains("Sent at"), "taken at once: nothing to say");
    }

    #[test]
    fn a_resumed_context_is_said_with_the_first_input() {
        let at = local(2026, 10, 22, 9, 0);
        let first = Some((Some(local(2026, 10, 22, 8, 58)), None));
        let header = Arrival {
            first: first.clone(),
            resumed: Some(Resumed { saved_at: local(2026, 10, 22, 8, 40), missing: None }),
            ..arrival(at)
        }
        .header();
        assert!(header.contains("Your working context is the one saved at Thu 22 Oct 2026 08:40, 20 min before this input"), "{header}");
        assert!(!header.contains("archive_get"), "{header}");
        let cut_short = Arrival {
            first,
            resumed: Some(Resumed { saved_at: local(2026, 10, 22, 8, 40), missing: Some((120, 131)) }),
            ..arrival(at)
        }
        .header();
        assert!(cut_short.contains("Archive events #120–#131 came after that save and are not in it"), "{cut_short}");
        let one = Arrival {
            first: Some((Some(local(2026, 10, 22, 8, 58)), None)),
            resumed: Some(Resumed { saved_at: local(2026, 10, 22, 8, 40), missing: Some((11, 11)) }),
            ..arrival(at)
        }
        .header();
        assert!(one.contains("Archive event #11 came after that save and is not in it") && one.contains("archive_get reads it."), "{one}");
    }

    #[test]
    fn a_rebuild_since_the_previous_input_is_said() {
        let at = local(2026, 10, 22, 9, 0);
        let header = Arrival {
            previous_input: Some(local(2026, 10, 21, 23, 0)),
            rebuilt: Some((local(2026, 10, 22, 2, 30), Rebuild::Sleep)),
            ..arrival(at)
        }
        .header();
        assert!(header.contains("A deep sleep finished at Thu 22 Oct 2026 02:30, 6 h 30 min before this input"), "{header}");
        let compacted = Arrival {
            previous_input: Some(local(2026, 10, 22, 8, 50)),
            rebuilt: Some((local(2026, 10, 22, 8, 55), Rebuild::Compaction)),
            ..arrival(at)
        }
        .header();
        assert!(compacted.contains("Your context was compacted at Thu 22 Oct 2026 08:55, 5 min before"), "{compacted}");
    }

    #[test]
    fn intervals_match_the_minutes_shown() {
        use chrono::Duration as Delta;
        // 10:53:59 and 14:05:00 show as 10:53 and 14:05: 3 h 12 min apart.
        let at = local(2026, 10, 22, 14, 5);
        let previous = local(2026, 10, 22, 10, 53) + Delta::seconds(59);
        let header = Arrival { previous_input: Some(previous), ..arrival(at) }.header();
        assert!(header.contains("Previous input: Thu 22 Oct 2026 10:53, 3 h 12 min before this one."), "{header}");
        // Under a minute it is exact.
        assert_eq!(between_shown(&(at - Delta::seconds(42)), &at), Duration::from_secs(42));
    }

    #[test]
    fn operators_last_message_is_given_when_other_inputs_came_after_it() {
        let at = local(2026, 10, 22, 14, 5);
        let looping = Arrival {
            from: Who::Harness,
            previous_input: Some(local(2026, 10, 22, 14, 0)),
            last_operator: Some(local(2026, 10, 22, 10, 0)),
            ..arrival(at)
        }
        .header();
        assert!(looping.contains("The operator's last message before this was at Thu 22 Oct 2026 10:00, 4 h 5 min before this input."), "{looping}");
        let conversation = Arrival {
            previous_input: Some(local(2026, 10, 22, 10, 0)),
            last_operator: Some(local(2026, 10, 22, 10, 0)),
            ..arrival(at)
        }
        .header();
        assert!(!conversation.contains("last message before this"), "the previous input already says it: {conversation}");
    }

    #[test]
    fn with_operators_zone_shown_the_hosts_times_are_named_as_such() {
        let at = local(2026, 10, 22, 14, 5);
        let header = Arrival { operator: Some("The operator's local time: x".into()), ..arrival(at) }.header();
        assert!(header.contains("is host time"), "{header}");
        assert!(!arrival(at).header().contains("host time"), "one zone, nothing to tell apart");
    }

    #[test]
    fn operators_time_comes_with_a_note_on_whose_today_it_is() {
        let at = local(2026, 10, 22, 14, 5);
        let header = Arrival {
            operator: Some("The operator's local time: Wednesday 21 October 2026, 21:05 (UTC-06:00)".into()),
            ..arrival(at)
        }
        .header();
        let lines: Vec<&str> = header.lines().collect();
        assert_eq!(lines[1], "The operator's local time: Wednesday 21 October 2026, 21:05 (UTC-06:00)");
        assert!(header.contains("he means his date"));
    }

    #[test]
    fn arrivals_chain_and_a_running_turn_gets_no_turn_end() {
        let _globals = globals();
        reset_timeline();
        // The process-wide timeline: arrivals in order, as App sees them.
        mark_start(None);
        let first = arrive(Joins::NewTurn, Who::Operator, None);
        assert!(first.header().contains("first input since s0ma started"), "{}", first.header());
        accept(&first);
        mark_turn_end();
        let second = arrive(Joins::NewTurn, Who::Harness, None);
        assert_eq!(second.previous_input, Some(first.at));
        assert!(second.turn_ended.is_some());
        accept(&second);
        let steer = arrive(Joins::RunningTurn, Who::Operator, None);
        assert_eq!(steer.turn_ended, None);
        assert_eq!(steer.first, None);
        assert_eq!(steer.last_operator, Some(first.at), "the loop cycle wasn't the operator's");
        accept(&steer);
        mark_rebuilt(Rebuild::Sleep);
        let woken = arrive(Joins::NewTurn, Who::Operator, None);
        assert!(woken.rebuilt.is_some());
        accept(&woken);
        assert!(arrive(Joins::NewTurn, Who::Operator, None).rebuilt.is_none(), "said once");
        reset_timeline();
    }

    #[test]
    fn a_resumed_context_is_said_once_and_not_after_a_rebuild() {
        let _globals = globals();
        reset_timeline();
        mark_start(None);
        mark_resumed(Resumed { saved_at: now(), missing: Some((5, 9)) });
        let first = arrive(Joins::NewTurn, Who::Operator, None);
        assert!(first.resumed.is_some());
        accept(&first);
        assert!(arrive(Joins::NewTurn, Who::Operator, None).resumed.is_none(), "only with the first input");
        reset_timeline();
        // A /compact or a sleep before the first input: the context in front
        // of s0ma is the rebuilt one, and only that is said.
        mark_start(None);
        mark_resumed(Resumed { saved_at: now(), missing: Some((5, 9)) });
        mark_rebuilt(Rebuild::Compaction);
        let after = arrive(Joins::NewTurn, Who::Operator, None);
        assert!(after.resumed.is_none() && after.rebuilt.is_some());
        reset_timeline();
    }

    #[test]
    fn an_input_refused_on_the_way_uses_nothing_up() {
        let _globals = globals();
        reset_timeline();
        mark_start(Some(Recorded { at: now(), was_sleep: true }));
        mark_rebuilt(Rebuild::Sleep);
        // The router fails: the header was built, the turn never accepted.
        let refused = arrive(Joins::NewTurn, Who::Operator, None);
        mark_turn_end();
        let again = arrive(Joins::NewTurn, Who::Operator, None);
        assert!(again.first.is_some(), "still the first input s0ma sees");
        assert!(again.rebuilt.is_some(), "the sleep is still news");
        assert_eq!(again.previous_input, None, "nothing s0ma never saw is 'previous'");
        assert_eq!(timeline(|timeline| timeline.turn_ended), None, "a refused turn didn't end anything");
        let _ = refused;
        // A rebuild after the header was built isn't cleared by accepting it.
        mark_rebuilt(Rebuild::Compaction);
        accept(&again);
        assert!(arrive(Joins::NewTurn, Who::Operator, None).rebuilt.is_some());
        reset_timeline();
    }
}
