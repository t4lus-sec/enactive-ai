//! Scopes for archive search: time windows (parsing `since`/`until`, and
//! reading event times) and event ranges (`from_event`/`through_event`). Kept
//! separate so other memory searches can reuse the same rules.
use chrono::{DateTime, Duration as ChronoDuration, Local, NaiveDate, NaiveDateTime, TimeZone};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    error::Error,
    io::{BufRead, BufReader},
    path::Path,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// ---------------------------------------------------------------------------
// Time windows
// ---------------------------------------------------------------------------

/// A half-open time window `[since, until)` in Unix seconds. Either end may be open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TimeWindow {
    pub since: Option<i64>,
    pub until: Option<i64>,
}

pub(crate) const WINDOW_HELP: &str = "since/until accept: a relative age (\"90m\", \"24h\", \"7d\", \"2w\" = that long before now), \"today\", \"yesterday\", a local date \"2026-10-01\" (a date as `until` includes that whole day), a local time \"2026-10-01 14:30\", or RFC 3339 with an offset.";

impl TimeWindow {
    /// `None` when neither bound is given. Errors explain the accepted forms, so
    /// the model can correct itself instead of silently searching everything.
    pub(crate) fn parse(
        since: Option<&str>,
        until: Option<&str>,
        now: DateTime<Local>,
    ) -> std::result::Result<Option<Self>, String> {
        let since = since.map(str::trim).filter(|s| !s.is_empty());
        let until = until.map(str::trim).filter(|s| !s.is_empty());
        if since.is_none() && until.is_none() {
            return Ok(None);
        }
        let window = Self {
            since: since.map(|s| parse_bound(s, now, false)).transpose()?,
            until: until.map(|s| parse_bound(s, now, true)).transpose()?,
        };
        if let (Some(a), Some(b)) = (window.since, window.until)
            && b <= a {
                return Err(format!(
                    "until ({}) must be later than since ({})",
                    describe(b),
                    describe(a)
                ));
            }
        Ok(Some(window))
    }

    pub(crate) fn contains(&self, at: i64) -> bool {
        self.since.is_none_or(|s| at >= s) && self.until.is_none_or(|u| at < u)
    }

    /// The resolved window, echoed back so the caller sees what was searched.
    pub(crate) fn to_json(self) -> Value {
        json!({
            "since": self.since.map(describe),
            "until_exclusive": self.until.map(describe),
        })
    }
}

// ---------------------------------------------------------------------------
// Event ranges
// ---------------------------------------------------------------------------

/// An inclusive range of event IDs, `[from, through]`. Either end may be open.
/// A memory gist names the stretch it covers as `events #A..#B`, and masked
/// results and episodes cite event IDs; this is how a search is held to them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EventRange {
    pub from: Option<u64>,
    pub through: Option<u64>,
}

impl EventRange {
    /// `None` when neither end is given.
    pub(crate) fn new(from: Option<u64>, through: Option<u64>) -> std::result::Result<Option<Self>, String> {
        if let (Some(from), Some(through)) = (from, through)
            && through < from {
                return Err(format!(
                    "through_event (#{through}) is before from_event (#{from}); both ends are inclusive, so for one event give the same ID twice"
                ));
            }
        Ok((from.is_some() || through.is_some()).then_some(Self { from, through }))
    }

    pub(crate) fn contains(&self, id: u64) -> bool {
        self.from.is_none_or(|from| id >= from) && self.through.is_none_or(|through| id <= through)
    }

    /// The ends as SQLite integers. An end past SQLite's range matches as
    /// anything that large would: a `from` there excludes every event, a
    /// `through` there excludes none.
    pub(crate) fn sql_bounds(&self) -> (Option<i64>, Option<i64>) {
        (
            self.from.map(|from| i64::try_from(from).unwrap_or(i64::MAX)),
            self.through.and_then(|through| i64::try_from(through).ok()),
        )
    }

    /// The range, echoed back so the caller sees what was searched.
    pub(crate) fn to_json(self) -> Value {
        json!({"from_event": self.from, "through_event": self.through})
    }
}

/// An event's time in a compact local form, for a one-line reference.
pub(crate) fn short_time(timestamp: &str) -> String {
    event_time(timestamp)
        .and_then(|at| Local.timestamp_opt(at, 0).single())
        .map(|at| at.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| timestamp.chars().take(16).collect())
}

fn describe(at: i64) -> String {
    Local
        .timestamp_opt(at, 0)
        .single()
        .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Secs, false))
        .unwrap_or_else(|| at.to_string())
}

fn local_midnight(date: NaiveDate) -> std::result::Result<i64, String> {
    let midnight = date.and_hms_opt(0, 0, 0).ok_or("invalid date")?;
    Local
        .from_local_datetime(&midnight)
        .earliest()
        .map(|t| t.timestamp())
        .ok_or_else(|| format!("{date} has no local midnight"))
}

/// One bound. A whole day as `until` means "through the end of that day".
fn parse_bound(text: &str, now: DateTime<Local>, is_until: bool) -> std::result::Result<i64, String> {
    let lower = text.to_ascii_lowercase();
    let day = |date: NaiveDate| -> std::result::Result<i64, String> {
        if is_until {
            local_midnight(date.succ_opt().ok_or("date out of range")?)
        } else {
            local_midnight(date)
        }
    };
    match lower.as_str() {
        "now" => return Ok(now.timestamp()),
        "today" => return day(now.date_naive()),
        "yesterday" => return day(now.date_naive().pred_opt().ok_or("date out of range")?),
        _ => {}
    }
    if let Some(age) = relative_age(&lower) {
        return Ok((now - age).timestamp());
    }
    if let Ok(date) = NaiveDate::parse_from_str(text, "%Y-%m-%d") {
        return day(date);
    }
    for format in ["%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(text, format) {
            return Local
                .from_local_datetime(&naive)
                .earliest()
                .map(|t| t.timestamp())
                .ok_or_else(|| format!("{text} does not exist in local time"));
        }
    }
    if let Ok(at) = DateTime::parse_from_rfc3339(text) {
        return Ok(at.timestamp());
    }
    Err(format!("could not read time {text:?}. {WINDOW_HELP}"))
}

/// "7d", "24h", "90m", "2w", optionally "7 days ago".
fn relative_age(text: &str) -> Option<ChronoDuration> {
    let text = text.strip_suffix(" ago").unwrap_or(text).trim();
    let split = text.find(|c: char| !c.is_ascii_digit())?;
    if split == 0 {
        return None;
    }
    let amount: i64 = text[..split].parse().ok()?;
    let unit = text[split..].trim();
    let duration = match unit {
        "m" | "min" | "mins" | "minute" | "minutes" => ChronoDuration::try_minutes(amount),
        "h" | "hr" | "hrs" | "hour" | "hours" => ChronoDuration::try_hours(amount),
        "d" | "day" | "days" => ChronoDuration::try_days(amount),
        "w" | "wk" | "week" | "weeks" => ChronoDuration::try_weeks(amount),
        _ => None,
    }?;
    (amount <= 100 * 366).then_some(duration)
}

/// Unix seconds for an archive timestamp, `None` when it can't be read.
pub(crate) fn event_time(timestamp: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(timestamp.trim()).ok().map(|t| t.timestamp())
}

/// Every archive event's time, by scanning the JSONL directly. The fallback when
/// the keyword index (which stores times) can't be used.
pub(crate) fn scan_event_times(archive: &Path) -> Result<HashMap<u64, i64>> {
    #[derive(serde::Deserialize)]
    struct Line {
        id: u64,
        #[serde(default)]
        timestamp: String,
    }
    let mut times = HashMap::new();
    if !archive.exists() {
        return Ok(times);
    }
    for line in BufReader::new(std::fs::File::open(archive)?).lines() {
        let line = line?;
        let Ok(event) = serde_json::from_str::<Line>(&line) else { continue };
        if let Some(at) = event_time(&event.timestamp) {
            times.insert(event.id, at);
        }
    }
    Ok(times)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> DateTime<Local> {
        Local.with_ymd_and_hms(2026, 10, 7, 15, 30, 0).single().unwrap()
    }

    #[test]
    fn relative_and_keyword_bounds() {
        let w = TimeWindow::parse(Some("7d"), None, now()).unwrap().unwrap();
        assert_eq!(w.since, Some((now() - ChronoDuration::days(7)).timestamp()));
        assert_eq!(w.until, None);
        let w = TimeWindow::parse(Some("3 days ago"), Some("90m"), now()).unwrap().unwrap();
        assert_eq!(w.until, Some((now() - ChronoDuration::minutes(90)).timestamp()));
        let w = TimeWindow::parse(Some("yesterday"), Some("yesterday"), now()).unwrap().unwrap();
        assert_eq!(w.until.unwrap() - w.since.unwrap(), 86_400);
        let today = local_midnight(now().date_naive()).unwrap();
        assert_eq!(w.until, Some(today));
    }

    #[test]
    fn event_times_parse_rfc3339_with_offsets() {
        assert_eq!(event_time("2026-10-07T00:00:00+02:00"), Some(DateTime::parse_from_rfc3339("2026-10-06T22:00:00Z").unwrap().timestamp()));
        assert_eq!(event_time("not a time"), None);
    }
}
