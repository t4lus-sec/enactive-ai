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
    fn a_short_time_label_falls_back_to_the_raw_stamp() {
        assert_eq!(short_time("2026-10-16T14:02:11.123456+00:00").len(), 16);
        assert_eq!(short_time("not a time"), "not a time");
        assert_eq!(short_time(""), "");
    }

    #[test]
    fn event_ranges_are_inclusive_and_either_end_may_be_open() {
        assert_eq!(EventRange::new(None, None).unwrap(), None);
        let range = EventRange::new(Some(10), Some(20)).unwrap().unwrap();
        assert!(range.contains(10) && range.contains(20) && !range.contains(9) && !range.contains(21));
        let one = EventRange::new(Some(7), Some(7)).unwrap().unwrap();
        assert!(one.contains(7) && !one.contains(8));
        let open = EventRange::new(Some(5), None).unwrap().unwrap();
        assert!(open.contains(u64::MAX) && !open.contains(4));
        assert!(EventRange::new(None, Some(3)).unwrap().unwrap().contains(0));
        let error = EventRange::new(Some(20), Some(10)).unwrap_err();
        assert!(error.contains("through_event (#10) is before from_event (#20)"), "{error}");
        assert_eq!(EventRange::new(Some(u64::MAX), Some(u64::MAX)).unwrap().unwrap().sql_bounds(), (Some(i64::MAX), None));
        assert_eq!(range.to_json(), json!({"from_event": 10, "through_event": 20}));
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
    fn a_date_as_until_includes_that_day() {
        let w = TimeWindow::parse(Some("2026-10-01"), Some("2026-10-01"), now()).unwrap().unwrap();
        let noon = Local.with_ymd_and_hms(2026, 10, 1, 12, 0, 0).single().unwrap().timestamp();
        let next = Local.with_ymd_and_hms(2026, 10, 2, 0, 0, 0).single().unwrap().timestamp();
        assert!(w.contains(noon));
        assert!(!w.contains(next));
    }

    #[test]
    fn explicit_times_and_rfc3339() {
        let w = TimeWindow::parse(Some("2026-10-01 14:30"), Some("2026-10-02T00:00:00+00:00"), now())
            .unwrap()
            .unwrap();
        assert_eq!(w.until, Some(DateTime::parse_from_rfc3339("2026-10-02T00:00:00+00:00").unwrap().timestamp()));
        assert!(w.since.is_some());
    }

    #[test]
    fn bad_input_explains_itself() {
        assert_eq!(TimeWindow::parse(None, Some("  "), now()).unwrap(), None);
        let err = TimeWindow::parse(Some("last spring"), None, now()).unwrap_err();
        assert!(err.contains("relative age"));
        assert!(TimeWindow::parse(Some("today"), Some("yesterday"), now()).unwrap_err().contains("later than"));
        assert!(TimeWindow::parse(Some("7x"), None, now()).is_err());
        assert!(TimeWindow::parse(Some("99999999d"), None, now()).is_err());
    }

    #[test]
    fn open_ended_windows() {
        let w = TimeWindow { since: Some(100), until: Some(200) };
        assert!(w.contains(100));
        assert!(!w.contains(200));
        assert!(TimeWindow { since: None, until: None }.contains(0));
    }

    #[test]
    fn event_times_parse_rfc3339_with_offsets() {
        assert_eq!(event_time("2026-10-07T00:00:00+02:00"), Some(DateTime::parse_from_rfc3339("2026-10-06T22:00:00Z").unwrap().timestamp()));
        assert_eq!(event_time("not a time"), None);
    }


}
