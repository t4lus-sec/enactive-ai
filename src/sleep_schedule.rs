//! A daily sleep at a set time: `/sleep set 0200` runs a sleep cycle every day
//! at 2 am, local time.
//!
//! - The time is saved beside the archive (`data/sleep_schedule.json`), so it
//!   survives restarts. `/sleep set off` turns it off; `/sleep set` alone says
//!   what is set and when the next one runs.
//! - A timer thread wakes the idle prompt at the time. The sleep itself runs
//!   from `App::next_input`, so it only ever starts between turns: a turn still
//!   running at 2 am finishes first, and the sleep follows it. Subagents still
//!   working are waited for, never cancelled.
//! - A sleep that can't start within `GRACE_HOURS` of its time (a turn or
//!   subagent that ran for hours, or a machine suspended overnight) is skipped
//!   until the next day, with a note, rather than resetting the context in the
//!   middle of someone's morning.
//! - A sleep that fails is tried once more `RETRY_MINUTES` later, inside the
//!   same grace. Nothing else changes: loop mode and VM wake events carry on.
//! - Times are wall-clock local. On a daylight-saving day a time that doesn't
//!   exist (02:30 when clocks jump from 02:00 to 03:00) runs at the first
//!   minute that does; one that happens twice runs the first time.
//! - It only fires while s0ma is running. A 2 am that passed while it was
//!   stopped is not made up on startup.

use chrono::{
    DateTime, Duration as ChronoDuration, Local, LocalResult, NaiveDateTime, NaiveTime, TimeZone, Timelike,
};
use serde::{Deserialize, Serialize};

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, Thread},
    time::Duration,
};

const FILE: &str = "sleep_schedule.json";

/// How late a scheduled sleep may still start.
pub const GRACE_HOURS: i64 = 3;

/// A failed scheduled sleep is tried once more this much later.
pub const RETRY_MINUTES: i64 = 15;

/// The timer re-reads the clock at least this often, so a suspended machine
/// or a clock change is noticed promptly (thread sleeps don't count suspend).
const TICK: Duration = Duration::from_secs(30);

const USAGE: &str = "usage: /sleep (sleep now), /sleep set HHMM (every day at that time, 24-hour, e.g. 0200), \
     /sleep set off, /sleep set (what is set)";

#[derive(Serialize, Deserialize)]
struct Saved {
    /// `HH:MM`, local.
    daily_at: String,
}

/// A scheduled sleep whose time came.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DueSleep {
    /// When it was due: the grace counts from here, retries included.
    pub at: DateTime<Local>,
    /// The schedule it belongs to, for notes (`at` can differ from it on a
    /// daylight-saving day).
    pub daily_at: NaiveTime,
    /// Already tried once and failed.
    pub retried: bool,
    /// The operator has been told it is waiting for subagents.
    pub waiting_noted: bool,
    /// Handed back to wait (`put_back`): while it waits, no new subagents
    /// start and loop mode pauses.
    pub held: bool,
}

impl DueSleep {
    /// `02:00`
    pub fn label(&self) -> String {
        self.daily_at.format("%H:%M").to_string()
    }
}

#[derive(Default)]
struct State {
    daily_at: Option<NaiveTime>,
    /// When the timer fires next for the schedule.
    next: Option<DateTime<Local>>,
    /// A time that came and hasn't been acted on yet.
    due: Option<DueSleep>,
    /// A failed sleep's second try, and when.
    retry: Option<(DateTime<Local>, DueSleep)>,
}

struct Shared {
    path: PathBuf,
    state: Mutex<State>,
    timer: Mutex<Option<Thread>>,
}

/// Cheap to clone; every clone is the same schedule.
#[derive(Clone)]
pub struct SleepSchedule {
    shared: Arc<Shared>,
}

/// What to do with a scheduled sleep whose time came.
#[derive(Debug, PartialEq, Eq)]
pub enum Due {
    Run,
    /// More than `GRACE_HOURS` late: skipped until the next day.
    Skip,
}

impl SleepSchedule {
    /// The saved schedule, or none. A damaged file is reported and ignored.
    pub fn load(data_dir: &Path) -> Self {
        let path = data_dir.join(FILE);
        let daily_at = match fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<Saved>(&text)
                .map_err(|e| e.to_string())
                .and_then(|saved| parse_time(&saved.daily_at))
            {
                Ok(time) => Some(time),
                Err(error) => {
                    crate::terminal::warn_status(&format!("[sleep] ignoring {}: {error}", path.display()));
                    None
                }
            },
            Err(_) => None,
        };
        let next = daily_at.map(|at| next_after(&Local::now(), at));
        Self {
            shared: Arc::new(Shared {
                path,
                state: Mutex::new(State { daily_at, next, ..State::default() }),
                timer: Mutex::new(None),
            }),
        }
    }

    /// Start the timer. `wake` is the idle prompt's external break: set when
    /// the time comes, so a prompt waiting for input lets the sleep run.
    pub fn start(&self, wake: Arc<AtomicBool>) {
        let shared = Arc::clone(&self.shared);
        let handle = thread::Builder::new().name("sleep-schedule".into()).spawn(move || {
            loop {
                let wait = {
                    let state = lock(&shared.state);
                    let now = Local::now();
                    // A held sleep's grace running out ends the pause it causes.
                    let grace_end = state.due.as_ref().filter(|due| due.held).map(grace_end).filter(|end| *end > now);
                    let soonest = [state.next, state.retry.as_ref().map(|(at, _)| *at), grace_end].into_iter().flatten().min();
                    match soonest {
                        Some(at) => (at - now).to_std().unwrap_or(Duration::ZERO).min(TICK),
                        None => TICK,
                    }
                };
                thread::park_timeout(wait);
                let mut state = lock(&shared.state);
                let now = Local::now();
                let fired = fire_if_due(&mut state, now);
                let expired = state.due.as_ref().is_some_and(|due| due.held && now >= grace_end(due));
                if fired || expired {
                    wake.store(true, Ordering::SeqCst);
                }
            }
        });
        match handle {
            Ok(handle) => *lock(&self.shared.timer) = Some(handle.thread().clone()),
            Err(error) => crate::terminal::warn_status(&format!("[sleep] the daily sleep timer could not start: {error}")),
        }
    }

    /// A scheduled sleep whose time came, if there is one; taking it clears it.
    pub fn take_due(&self) -> Option<DueSleep> {
        lock(&self.shared.state).due.take()
    }

    /// Hand back a sleep that has to wait (subagents still working, or their
    /// results still to be read). Kept only while the schedule it came from
    /// is still set.
    pub fn put_back(&self, due: DueSleep) {
        let mut state = lock(&self.shared.state);
        if state.daily_at == Some(due.daily_at) && state.due.is_none() {
            state.due = Some(DueSleep { held: true, ..due });
        }
    }

    /// A due sleep is waiting its turn (see `put_back`).
    pub fn waiting(&self) -> bool {
        lock(&self.shared.state).due.as_ref().is_some_and(|due| due.held)
    }

    /// The operator slept by hand: nothing scheduled is still owed tonight.
    pub fn clear_pending(&self) {
        let mut state = lock(&self.shared.state);
        state.due = None;
        state.retry = None;
    }

    /// A scheduled sleep failed: try it once more `RETRY_MINUTES` from `now`,
    /// if that is still inside its grace and the schedule is unchanged.
    /// Returns when, or None when it won't be retried.
    pub fn retry(&self, due: &DueSleep, now: DateTime<Local>) -> Option<DateTime<Local>> {
        let at = now + ChronoDuration::minutes(RETRY_MINUTES);
        if due.retried || decide(&due.at, &at) == Due::Skip {
            return None;
        }
        {
            let mut state = lock(&self.shared.state);
            if state.daily_at != Some(due.daily_at) {
                return None;
            }
            state.retry = Some((at, DueSleep { retried: true, waiting_noted: false, held: false, ..due.clone() }));
        }
        self.nudge();
        Some(at)
    }

    /// `/sleep set …` and `/sleep <anything else>`; None for plain `/sleep`
    /// (sleep now) and for other commands.
    pub fn handle_command(&self, command: &str) -> Option<Result<String, String>> {
        let mut words = command.split_whitespace();
        if words.next() != Some("/sleep") {
            return None;
        }
        match words.next() {
            None => None,
            Some("set") => {
                let argument = words.collect::<Vec<_>>().join(" ");
                Some(self.command_at(&argument, Local::now()))
            }
            // `/sleep 0200`, `/sleep off`: say how, rather than letting it go
            // to s0ma as a message.
            Some(_) => Some(Err(format!("[sleep] {USAGE}"))),
        }
    }

    fn command_at(&self, argument: &str, now: DateTime<Local>) -> Result<String, String> {
        let argument = argument.trim();
        if argument.is_empty() {
            return Ok(self.status(&now));
        }

        if matches!(argument.to_ascii_lowercase().as_str(), "off" | "none" | "clear") {
            let was = lock(&self.shared.state).daily_at;
            self.save(None).map_err(|error| format!("[sleep] could not save the schedule: {error}"))?;
            *lock(&self.shared.state) = State::default();
            self.nudge();
            return Ok(match was {
                Some(at) => format!("[sleep] the daily sleep at {} is off", at.format("%H:%M")),
                None => "[sleep] no daily sleep was set".to_string(),
            });
        }

        let at = parse_time(argument).map_err(|error| format!("[sleep] {error}. {USAGE}"))?;
        self.save(Some(at)).map_err(|error| format!("[sleep] could not save the schedule: {error}"))?;
        let next = next_after(&now, at);
        {
            let mut state = lock(&self.shared.state);
            if state.daily_at != Some(at) {
                // Whatever was pending belonged to the old time.
                state.due = None;
                state.retry = None;
            }
            state.daily_at = Some(at);
            state.next = Some(next);
        }
        self.nudge();
        Ok(format!(
            "[sleep] a sleep cycle will run every day at {}; the next is {}. /sleep set off turns it off.",
            at.format("%H:%M"),
            describe_next(&now, &next),
        ))
    }

    /// The daily sleep's time, if one is set, and the schedule as `/sleep
    /// set` says it (for the overview).
    pub fn summary(&self) -> (Option<String>, String) {
        let at = lock(&self.shared.state).daily_at.map(|at| at.format("%H:%M").to_string());
        (at, self.status(&Local::now()))
    }

    fn status(&self, now: &DateTime<Local>) -> String {
        let state = lock(&self.shared.state);
        let Some(at) = state.daily_at else {
            return "[sleep] no daily sleep is set. /sleep set 0200 runs one every day at 2 am.".to_string();
        };
        let mut text = format!("[sleep] a sleep cycle runs every day at {}.", at.format("%H:%M"));
        if let Some(due) = &state.due {
            let what = if decide(&due.at, now) == Due::Skip {
                "it is past its grace and will be skipped"
            } else if due.held {
                "it waits for subagents to finish and for their results to be read"
            } else {
                "it runs once s0ma is between turns"
            };
            text.push_str(&format!(" The {} sleep is due: {what}.", due.label()));
        }
        if let Some((retry, _)) = &state.retry {
            text.push_str(&format!(" A failed one is tried again at {}.", retry.format("%H:%M")));
        }
        if let Some(next) = state.next {
            text.push_str(&format!(" The next is {}.", describe_next(now, &next)));
        }
        text.push_str(" /sleep set off turns it off.");
        text
    }

    /// Have the timer look at the new schedule now rather than at its next tick.
    fn nudge(&self) {
        if let Some(timer) = lock(&self.shared.timer).as_ref() {
            timer.unpark();
        }
    }

    fn save(&self, at: Option<NaiveTime>) -> std::io::Result<()> {
        let path = &self.shared.path;
        let Some(at) = at else {
            return match fs::remove_file(path) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
                _ => Ok(()),
            };
        };
        let text = serde_json::to_string(&Saved { daily_at: at.format("%H:%M").to_string() })?;
        crate::util::write_atomic(path, text + "\n")
    }
}

/// When a due sleep stops being worth running.
fn grace_end(due: &DueSleep) -> DateTime<Local> {
    due.at + ChronoDuration::hours(GRACE_HOURS)
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The timer's step: a retry or the schedule's time that has come becomes
/// due, and the schedule moves on to the following day. Returns whether
/// anything came due.
fn fire_if_due(state: &mut State, now: DateTime<Local>) -> bool {
    let mut fired = false;
    if let Some((_, due)) = state.retry.take_if(|(at, _)| now >= *at) {
        state.due.get_or_insert(due);
        fired = true;
    }
    if let (Some(at), Some(next)) = (state.daily_at, state.next.filter(|next| now >= *next)) {
        // A newer time supersedes anything still pending from an older one.
        state.due = Some(DueSleep { at: next, daily_at: at, retried: false, waiting_noted: false, held: false });
        state.retry = None;
        state.next = Some(next_after(&now, at));
        fired = true;
    }
    fired
}

/// Whether a sleep due at `due` still runs at `now`.
pub fn decide<Tz: TimeZone>(due: &DateTime<Tz>, now: &DateTime<Tz>) -> Due {
    if now.clone().signed_duration_since(due.clone()) <= ChronoDuration::hours(GRACE_HOURS) {
        Due::Run
    } else {
        Due::Skip
    }
}

/// `0200`, `200`, `02:00`, `2:00`, `2am`, `2:30 pm`, `14`.
pub fn parse_time(text: &str) -> Result<NaiveTime, String> {
    let lower = text.trim().to_ascii_lowercase().replace(' ', "");
    let (clock, meridiem) = if let Some(rest) = lower.strip_suffix("am") {
        (rest, Some(false))
    } else if let Some(rest) = lower.strip_suffix("pm") {
        (rest, Some(true))
    } else {
        (lower.as_str(), None)
    };
    let bad = || format!("'{}' isn't a time", text.trim());
    let (hour, minute): (u32, u32) = match clock.split_once(':') {
        Some((hour, minute)) if !hour.is_empty() && minute.len() == 2 => {
            (hour.parse().map_err(|_| bad())?, minute.parse().map_err(|_| bad())?)
        }
        Some(_) => return Err(bad()),
        None if !clock.is_empty() && clock.len() <= 4 && clock.chars().all(|c| c.is_ascii_digit()) => {
            let number: u32 = clock.parse().map_err(|_| bad())?;
            if clock.len() <= 2 { (number, 0) } else { (number / 100, number % 100) }
        }
        None => return Err(bad()),
    };
    let hour = match meridiem {
        Some(_) if !(1..=12).contains(&hour) => return Err(bad()),
        Some(false) => hour % 12,
        Some(true) => hour % 12 + 12,
        None => hour,
    };
    NaiveTime::from_hms_opt(hour, minute, 0).ok_or_else(bad)
}

/// The first moment after `now` that the local clock reads `at`.
pub fn next_after<Tz: TimeZone>(now: &DateTime<Tz>, at: NaiveTime) -> DateTime<Tz> {
    let zone = now.timezone();
    let mut date = now.date_naive();
    loop {
        if let Some(moment) = resolve(&zone, date.and_time(at)).filter(|moment| moment > now) {
            return moment;
        }
        date = date.succ_opt().expect("a date after today exists");
    }
}

/// A local wall-clock time as a moment: the first of two when the clocks go
/// back, the first minute after the gap when they go forward.
///
/// Every candidate the zone offers is checked against the clock: some zone
/// implementations offer a moment that reads differently once converted back
/// (02:00 PDT on the night 02:00 becomes 01:00 PST), and they don't promise
/// which of two comes first.
fn resolve<Tz: TimeZone>(zone: &Tz, local: NaiveDateTime) -> Option<DateTime<Tz>> {
    (0..=180).find_map(|minutes| {
        let wanted = local + ChronoDuration::minutes(minutes);
        let candidates = match zone.from_local_datetime(&wanted) {
            LocalResult::Single(moment) => vec![moment],
            LocalResult::Ambiguous(first, second) => vec![first, second],
            LocalResult::None => Vec::new(),
        };
        candidates.into_iter().filter(|moment| moment.with_timezone(zone).naive_local() == wanted).min()
    })
}

/// `tonight at 02:00 (in 3h 12m)`, `tomorrow at 14:00 (in 20h 05m)`,
/// `Monday 12 October at 02:00 (in 1d 3h)`.
fn describe_next<Tz: TimeZone>(now: &DateTime<Tz>, next: &DateTime<Tz>) -> String
where
    Tz::Offset: std::fmt::Display,
{
    let days = (next.date_naive() - now.date_naive()).num_days();
    let when = match days {
        0 if next.hour() >= 18 || now.hour() >= 18 => "tonight".to_string(),
        0 => "today".to_string(),
        1 if next.hour() < 6 && now.hour() >= 18 => "tonight".to_string(),
        1 => "tomorrow".to_string(),
        _ => next.format("%A %-d %B").to_string(),
    };
    let left = next.clone().signed_duration_since(now.clone());
    let minutes = left.num_minutes().max(0);
    let wait = if minutes >= 24 * 60 {
        format!("{}d {}h", minutes / (24 * 60), minutes % (24 * 60) / 60)
    } else if minutes >= 60 {
        format!("{}h {:02}m", minutes / 60, minutes % 60)
    } else {
        format!("{minutes}m")
    };
    format!("{when} at {} (in {wait})", next.format("%H:%M"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{FixedOffset, NaiveDate, Offset};

    fn at(hour: u32, minute: u32) -> NaiveTime {
        NaiveTime::from_hms_opt(hour, minute, 0).unwrap()
    }

    fn pacific(day: u32, hour: u32, minute: u32) -> DateTime<FixedOffset> {
        FixedOffset::west_opt(7 * 3600).unwrap().with_ymd_and_hms(2026, 10, day, hour, minute, 0).unwrap()
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "s0ma-sleep-schedule-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// US Pacific around its 2026 changes, built by hand (no tz database here):
    /// clocks go 02:00 → 03:00 on 8 March and 02:00 → 01:00 on 1 November.
    /// Ambiguous times come back later-first, and 02:00 on 1 November comes
    /// back ambiguous with a PDT reading that is really 01:00 PST, the way
    /// chrono's `Local` was seen to.
    #[derive(Clone, Copy, Debug)]
    struct Pacific;

    const PST: i32 = -8 * 3600;
    const PDT: i32 = -7 * 3600;

    fn offset(seconds: i32) -> FixedOffset {
        FixedOffset::east_opt(seconds).unwrap()
    }

    fn summer_utc(utc: &NaiveDateTime) -> bool {
        let start = NaiveDate::from_ymd_opt(2026, 3, 8).unwrap().and_hms_opt(10, 0, 0).unwrap();
        let end = NaiveDate::from_ymd_opt(2026, 11, 1).unwrap().and_hms_opt(9, 0, 0).unwrap();
        *utc >= start && *utc < end
    }

    impl TimeZone for Pacific {
        type Offset = FixedOffset;
        fn from_offset(_: &FixedOffset) -> Self {
            Pacific
        }
        fn offset_from_local_date(&self, local: &NaiveDate) -> LocalResult<FixedOffset> {
            self.offset_from_local_datetime(&local.and_hms_opt(12, 0, 0).unwrap())
        }
        fn offset_from_local_datetime(&self, local: &NaiveDateTime) -> LocalResult<FixedOffset> {
            let spring = NaiveDate::from_ymd_opt(2026, 3, 8).unwrap();
            let fall = NaiveDate::from_ymd_opt(2026, 11, 1).unwrap();
            let hour = local.hour();
            if local.date() == spring && hour == 2 {
                return LocalResult::None;
            }
            if local.date() == fall && (hour == 1 || (hour == 2 && local.minute() == 0)) {
                return LocalResult::Ambiguous(offset(PST), offset(PDT));
            }
            // Read as daylight time; in summer that reading is the one.
            let as_daylight = *local - ChronoDuration::seconds(PDT as i64);
            LocalResult::Single(offset(if summer_utc(&as_daylight) { PDT } else { PST }))
        }
        fn offset_from_utc_date(&self, utc: &NaiveDate) -> FixedOffset {
            self.offset_from_utc_datetime(&utc.and_hms_opt(12, 0, 0).unwrap())
        }
        fn offset_from_utc_datetime(&self, utc: &NaiveDateTime) -> FixedOffset {
            offset(if summer_utc(utc) { PDT } else { PST })
        }
    }

    fn pacific_tz(month: u32, day: u32, hour: u32, minute: u32) -> DateTime<Pacific> {
        let local = NaiveDate::from_ymd_opt(2026, month, day).unwrap().and_hms_opt(hour, minute, 0).unwrap();
        resolve(&Pacific, local).unwrap()
    }

    #[test]
    fn daylight_saving_days_run_once_at_the_right_wall_time() {
        // Clocks go back: 01:30 happens twice, and the first one is taken.
        let first = next_after(&pacific_tz(10, 31, 23, 0), at(1, 30));
        assert_eq!(first.offset().fix(), offset(PDT), "the first 01:30, still daylight time");
        // 02:00 happens once, after the repeated hour, never an hour early.
        let two = next_after(&pacific_tz(10, 31, 23, 0), at(2, 0));
        assert_eq!(two.with_timezone(&Pacific).naive_local().hour(), 2);
        assert_eq!(two.offset().fix(), offset(PST));
        // Clocks go forward: 02:30 doesn't exist, so the first minute that does.
        let gap = next_after(&pacific_tz(3, 7, 23, 0), at(2, 30));
        assert_eq!(gap.with_timezone(&Pacific).naive_local().time(), at(3, 0));
        // Ordinary days are untouched.
        assert_eq!(next_after(&pacific_tz(6, 1, 1, 0), at(2, 0)).offset().fix(), offset(PDT));
    }

    #[test]
    fn times_are_read_the_ways_people_write_them() {
        assert_eq!(parse_time("0200"), Ok(at(2, 0)));
        assert_eq!(parse_time("200"), Ok(at(2, 0)));
        assert_eq!(parse_time("02:00"), Ok(at(2, 0)));
        assert_eq!(parse_time("2:30"), Ok(at(2, 30)));
        assert_eq!(parse_time("2am"), Ok(at(2, 0)));
        assert_eq!(parse_time("12am"), Ok(at(0, 0)));
        assert_eq!(parse_time("12pm"), Ok(at(12, 0)));
        assert_eq!(parse_time("2:30 PM"), Ok(at(14, 30)));
        assert_eq!(parse_time("14"), Ok(at(14, 0)));
        assert_eq!(parse_time("2359"), Ok(at(23, 59)));
        for wrong in ["2400", "0260", "13pm", "0am", "2:3", "abc", "", "12345", ":30", "-100"] {
            assert!(parse_time(wrong).is_err(), "{wrong:?} was accepted");
        }
    }

    #[test]
    fn the_next_run_is_the_next_time_the_clock_reads_it() {
        // Before 2 am: tonight. At or after it: tomorrow night.
        assert_eq!(next_after(&pacific(10, 1, 59), at(2, 0)), pacific(10, 2, 0));
        assert_eq!(next_after(&pacific(10, 2, 0), at(2, 0)), pacific(11, 2, 0));
        assert_eq!(next_after(&pacific(10, 23, 30), at(2, 0)), pacific(11, 2, 0));
        assert_eq!(next_after(&pacific(31, 23, 30), at(2, 0)).date_naive().to_string(), "2026-11-01");
    }

    #[test]
    fn the_note_says_when_in_words() {
        assert_eq!(describe_next(&pacific(10, 22, 48), &pacific(11, 2, 0)), "tonight at 02:00 (in 3h 12m)");
        assert_eq!(describe_next(&pacific(10, 9, 0), &pacific(11, 2, 0)), "tomorrow at 02:00 (in 17h 00m)");
        assert_eq!(describe_next(&pacific(10, 1, 15), &pacific(10, 2, 0)), "today at 02:00 (in 45m)");
    }

    #[test]
    fn a_late_sleep_runs_within_the_grace_and_is_skipped_after_it() {
        let due = pacific(11, 2, 0);
        assert_eq!(decide(&due, &pacific(11, 2, 0)), Due::Run);
        assert_eq!(decide(&due, &pacific(11, 4, 59)), Due::Run, "a turn that ran on");
        assert_eq!(decide(&due, &pacific(11, 9, 30)), Due::Skip, "a machine asleep all night");
    }

    #[test]
    fn the_timer_marks_the_time_due_once_and_moves_to_the_next_day() {
        let now = Local::now();
        let mut state = State { daily_at: Some(at(2, 0)), next: Some(now + ChronoDuration::seconds(5)), ..State::default() };
        assert!(!fire_if_due(&mut state, now), "not yet");
        let later = now + ChronoDuration::seconds(6);
        assert!(fire_if_due(&mut state, later));
        assert_eq!(state.due.as_ref().unwrap().at, now + ChronoDuration::seconds(5));
        assert!(state.next.unwrap() > later, "the next is still to come");
        assert!(!fire_if_due(&mut state, later), "once");
        assert!(!fire_if_due(&mut State::default(), later), "nothing set, nothing fires");
    }

    #[test]
    fn set_show_and_off_survive_a_restart() {
        let dir = temp_dir("roundtrip");
        let schedule = SleepSchedule::load(&dir);
        let now = Local::now();
        assert!(schedule.command_at("", now).unwrap().contains("no daily sleep is set"));

        let set = schedule.command_at("0200", now).unwrap();
        assert!(set.contains("every day at 02:00"), "{set}");
        assert!(fs::read_to_string(dir.join(FILE)).unwrap().contains("\"02:00\""));

        // A restart reads it back.
        let reloaded = SleepSchedule::load(&dir);
        let shown = reloaded.command_at("", now).unwrap();
        assert!(shown.contains("every day at 02:00") && shown.contains("The next is"), "{shown}");
        assert!(lock(&reloaded.shared.state).next.unwrap() > now);

        assert!(reloaded.command_at("tomorrow", now).unwrap_err().contains("/sleep set HHMM"));
        assert!(reloaded.command_at("off", now).unwrap().contains("02:00 is off"));
        assert!(!dir.join(FILE).exists());
        assert!(SleepSchedule::load(&dir).command_at("", now).unwrap().contains("no daily sleep"));
        assert!(reloaded.command_at("off", now).unwrap().contains("no daily sleep was set"));

        // A damaged file is ignored, not fatal.
        fs::write(dir.join(FILE), "{not json").unwrap();
        assert!(lock(&SleepSchedule::load(&dir).shared.state).daily_at.is_none());
    }

    fn due_now(daily_at: NaiveTime) -> DueSleep {
        DueSleep { at: Local::now(), daily_at, retried: false, waiting_noted: false, held: false }
    }

    #[test]
    fn a_pending_sleep_shows_and_setting_the_same_time_keeps_it() {
        let schedule = SleepSchedule::load(&temp_dir("pending"));
        let now = Local::now();
        schedule.command_at("0200", now).unwrap();
        lock(&schedule.shared.state).due = Some(due_now(at(2, 0)));
        assert!(schedule.command_at("", now).unwrap().contains("sleep is due: it runs once s0ma is between turns"));
        schedule.command_at("2:00", now).unwrap();
        assert!(schedule.take_due().is_some(), "re-entering the same time keeps tonight's");
        // A different time drops it.
        lock(&schedule.shared.state).due = Some(due_now(at(2, 0)));
        schedule.command_at("0300", now).unwrap();
        assert!(schedule.take_due().is_none());
    }

    #[test]
    fn a_waiting_sleep_goes_back_only_while_its_schedule_stands() {
        let schedule = SleepSchedule::load(&temp_dir("putback"));
        let now = Local::now();
        schedule.command_at("0200", now).unwrap();
        assert!(!schedule.waiting());
        schedule.put_back(due_now(at(2, 0)));
        assert!(schedule.waiting(), "held back: spawns and loop mode wait");
        assert!(schedule.command_at("", now).unwrap().contains("waits for subagents to finish"));
        assert!(schedule.take_due().unwrap().held);
        assert!(!schedule.waiting());
        // A manual /sleep settles whatever was owed.
        schedule.put_back(due_now(at(2, 0)));
        schedule.retry(&due_now(at(2, 0)), now);
        schedule.clear_pending();
        assert!(!schedule.waiting() && schedule.take_due().is_none());
        assert!(lock(&schedule.shared.state).retry.is_none());
        schedule.command_at("off", now).unwrap();
        schedule.put_back(due_now(at(2, 0)));
        assert!(schedule.take_due().is_none(), "turned off meanwhile");
    }

    #[test]
    fn a_failed_sleep_is_tried_once_more_inside_its_grace() {
        let schedule = SleepSchedule::load(&temp_dir("retry"));
        let now = Local::now();
        schedule.command_at("0200", now).unwrap();
        let due = due_now(at(2, 0));
        let retry_at = schedule.retry(&due, now).expect("a first failure is retried");
        assert_eq!(retry_at, now + ChronoDuration::minutes(RETRY_MINUTES));
        assert!(schedule.command_at("", now).unwrap().contains("tried again at"));

        // It comes due at its time, still counting the grace from the original.
        let mut state = lock(&schedule.shared.state);
        assert!(!fire_if_due(&mut state, now));
        assert!(fire_if_due(&mut state, retry_at));
        let again = state.due.take().unwrap();
        drop(state);
        assert!(again.retried && again.at == due.at);
        assert!(schedule.retry(&again, retry_at).is_none(), "only once");

        // Too late in the grace for another try: none.
        let old = DueSleep { at: now - ChronoDuration::hours(GRACE_HOURS), ..due_now(at(2, 0)) };
        assert!(schedule.retry(&old, now).is_none());
    }

    #[test]
    fn sleep_with_anything_but_set_is_answered_not_sent_to_s0ma() {
        let schedule = SleepSchedule::load(&temp_dir("commands"));
        assert!(schedule.handle_command("/sleep").is_none(), "plain /sleep still sleeps now");
        assert!(schedule.handle_command("  /sleep  ").is_none());
        assert!(schedule.handle_command("/sleeping set 0200").is_none());
        for slip in ["/sleep 0200", "/sleep off", "/sleep settle"] {
            assert!(schedule.handle_command(slip).unwrap().unwrap_err().contains("usage"), "{slip}");
        }
        assert!(schedule.handle_command("/sleep set").unwrap().is_ok());
        assert!(schedule.handle_command("  /sleep   set   2:00 am ").unwrap().unwrap().contains("02:00"));
    }

    #[test]
    fn the_timer_wakes_the_prompt_when_a_held_sleeps_grace_runs_out() {
        let schedule = SleepSchedule::load(&temp_dir("grace"));
        let wake = Arc::new(AtomicBool::new(false));
        schedule.command_at("0200", Local::now()).unwrap();
        let almost_over = Local::now() - ChronoDuration::hours(GRACE_HOURS) + ChronoDuration::milliseconds(150);
        schedule.put_back(DueSleep { at: almost_over, ..due_now(at(2, 0)) });
        schedule.start(Arc::clone(&wake));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !wake.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(wake.load(Ordering::SeqCst), "woken so the pause ends with the skip");
        assert_eq!(decide(&schedule.take_due().unwrap().at, &Local::now()), Due::Skip);
    }

    #[test]
    fn the_timer_wakes_the_prompt_when_the_time_comes() {
        let schedule = SleepSchedule::load(&temp_dir("timer"));
        let wake = Arc::new(AtomicBool::new(false));
        {
            let mut state = lock(&schedule.shared.state);
            state.daily_at = Some(at(2, 0));
            state.next = Some(Local::now() + ChronoDuration::milliseconds(150));
        }
        schedule.start(Arc::clone(&wake));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !wake.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(wake.load(Ordering::SeqCst), "the prompt was woken");
        assert!(schedule.take_due().is_some());
        assert!(schedule.take_due().is_none(), "taken once");
    }
}
