//! Observation masking: older tool results become pointers to the archive.
//!
//! A long tool loop fills the context with output that has already done its
//! job. Masking replaces older results with a short pointer to their archive
//! event and leaves the tool calls themselves in place. It is the simplest
//! strategy measured (Lindenbauer et al., NeurIPS 2025): in SWE-agent it halved
//! cost against no context management while matching LLM summarization's solve
//! rate. Here it is also lossless. Every masked result names its archive event,
//! and `archive_get` returns the full text, or matching lines with `pattern`.
//! Unlike a gist, a pointer is evidence, not a model's retelling of it.
//!
//! # Off by default
//!
//! Masking trades a prefix-cache miss for room: it paid when prefill was fast
//! (fne, ~8k tokens/s) and the window was 262,144. On ds41 a miss costs minutes
//! of prefill and a 1M window has room to spare, so it is off unless
//! `S0MA_OBSERVATION_MASK=1`. What was masked before stays masked; nothing new is.
//!
//! # When
//!
//! Rewriting an old message invalidates vLLM's prefix cache from that point on,
//! so masking happens rarely and all at once: only when the maskable older
//! output adds up to `TRIGGER_CHARS`. Then every eligible result is masked in
//! one batch, and the next batch waits for another `TRIGGER_CHARS` to build up.
//!
//! # What is eligible
//!
//! A tool result that is not among the `KEEP_RECENT` newest, has a body of at
//! least `MIN_CHARS`, carries an archive event id in its trailer (so it can be
//! restored), is not already masked, and did not come from a tool whose output
//! is standing instructions or identity (`NEVER_MASK`).
//!
//! # The trailer
//!
//! agent.rs ends every tool result with one harness-written line:
//! `[completed Mon 5 Oct 2026 18:42 local after 4 min 12 s, archive_event_id 123]`.
//! It keeps the current time in view through long loops (see `clock`), says
//! how long the call took so the agent never has to subtract, tells the agent which
//! archive event holds the full result, and is how this module finds that event
//! again. Subagent results are archived elsewhere, carry no id, and are never
//! masked.
use crate::types::Message;
use chrono::{DateTime, Local};
use std::collections::HashMap;
use std::time::Duration;

const KEEP_RECENT: usize = 8;
const MIN_CHARS: usize = 1_500;
const TRIGGER_CHARS: usize = 100_000;
const EXCERPT_CHARS: usize = 200;
/// Output that is instructions or identity rather than observation.
const NEVER_MASK: &[&str] = &["skill_read"];

const MASKED_HEADER: &str = "[MASKED TOOL RESULT]";
const TRAILER_START: &str = "[completed ";
const EVENT_MARKER: &str = ", archive_event_id ";

/// The line agent.rs appends to every tool result: when the call finished,
/// how long it took (`took`), and the archive event holding the result.
pub fn trailer(at: &DateTime<Local>, took: Option<Duration>, event_id: Option<u64>) -> String {
    let when = crate::clock::short(at);
    match took {
        Some(took) => trailer_text(&format!("{when} local after {}", crate::clock::span(took)), event_id),
        None => trailer_text(&format!("{when} local"), event_id),
    }
}

fn trailer_text(when: &str, event_id: Option<u64>) -> String {
    match event_id {
        Some(id) => format!("\n{TRAILER_START}{when}{EVENT_MARKER}{id}]"),
        None => format!("\n{TRAILER_START}{when}]"),
    }
}

/// The archive event a tool result's trailer names, if it names one.
///
/// Folding uses these as anchors: they are the only archive IDs a message
/// carries, so they are what an episode's event range is measured from.
pub fn event_id(content: &str) -> Option<u64> {
    parse_trailer(content).map(|(_, id)| id)
}

/// Split a tool result into its body and its trailer line, with or without an
/// event id. The trailer is empty when there isn't one.
///
/// Anything that shortens a tool result for display must keep the trailer
/// whole: it is the only place the result says which archive event holds it.
pub fn split_trailer(content: &str) -> (&str, &str) {
    match content.rfind('\n') {
        Some(start) => {
            let last = &content[start + 1..];
            if last.starts_with(TRAILER_START) && last.ends_with(']') {
                content.split_at(start)
            } else {
                (content, "")
            }
        }
        None => (content, ""),
    }
}

/// Byte index where the trailer starts (its leading newline) and the archive
/// event it names. Only a trailer with an event id counts.
fn parse_trailer(content: &str) -> Option<(usize, u64)> {
    let start = content.rfind('\n')?;
    let last = &content[start + 1..];
    if !last.starts_with(TRAILER_START) || !last.ends_with(']') {
        return None;
    }
    let marker = last.rfind(EVENT_MARKER)?;
    let id = last[marker + EVENT_MARKER.len()..last.len() - 1].parse().ok()?;
    Some((start, id))
}

#[derive(Debug, PartialEq, Eq)]
pub struct MaskReport {
    pub masked: usize,
    pub chars_before: usize,
    pub chars_after: usize,
}

/// Whether masking is on: `S0MA_OBSERVATION_MASK=1`. Off by default.
pub fn enabled() -> bool {
    std::env::var("S0MA_OBSERVATION_MASK").is_ok_and(|value| value.trim() == "1")
}

/// Mask every eligible older tool result in one batch, but only once they add
/// up to `TRIGGER_CHARS`; otherwise change nothing.
pub fn mask_if_due(messages: &mut [Message]) -> Option<MaskReport> {
    let tool_names: HashMap<&str, &str> = messages
        .iter()
        .filter_map(|message| message.tool_calls.as_ref())
        .flatten()
        .map(|call| (call.id.as_str(), call.function.name.as_str()))
        .collect();
    let tool_positions: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.role == "tool")
        .map(|(index, _)| index)
        .collect();
    let older = tool_positions.len().saturating_sub(KEEP_RECENT);

    let mut eligible = Vec::new();
    let mut total = 0usize;
    for &index in &tool_positions[..older] {
        let message = &messages[index];
        if message.content.starts_with(MASKED_HEADER) {
            continue;
        }
        let protected = message
            .tool_call_id
            .as_deref()
            .and_then(|id| tool_names.get(id))
            .is_some_and(|name| NEVER_MASK.contains(name));
        if protected {
            continue;
        }
        let Some((trailer_at, event_id)) = parse_trailer(&message.content) else { continue };
        let body_chars = message.content[..trailer_at].chars().count();
        if body_chars >= MIN_CHARS {
            eligible.push((index, trailer_at, event_id));
            total += body_chars;
        }
    }
    if total < TRIGGER_CHARS {
        return None;
    }

    let mut report = MaskReport { masked: 0, chars_before: 0, chars_after: 0 };
    for (index, trailer_at, event_id) in eligible {
        let message = &mut messages[index];
        report.chars_before += message.content.chars().count();
        let (body, trailer) = message.content.split_at(trailer_at);
        message.content = masked(body, trailer, event_id);
        report.chars_after += message.content.chars().count();
        report.masked += 1;
    }
    Some(report)
}

fn masked(body: &str, trailer: &str, event_id: u64) -> String {
    let one_line = body.split_whitespace().collect::<Vec<_>>().join(" ");
    let excerpt: String = one_line.chars().take(EXCERPT_CHARS).collect();
    let more = if one_line.chars().count() > EXCERPT_CHARS { "…" } else { "" };
    format!(
        "{MASKED_HEADER} Older output, replaced by this pointer; nothing was summarized. \
         Full result in archive event {event_id}: archive_get(id={event_id}), or archive_get(id={event_id}, pattern=...) to find lines.\n\
         Begins: {excerpt}{more}{trailer}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn trailer_round_trips() {
        let with_id = trailer_text("Mon 5 Oct 2026 18:42 local", Some(123));
        assert_eq!(with_id, "\n[completed Mon 5 Oct 2026 18:42 local, archive_event_id 123]");
        assert_eq!(parse_trailer(&format!("body{with_id}")), Some((4, 123)));
        assert_eq!(parse_trailer(&format!("body{}", trailer_text("Mon 5 Oct 2026 18:42 local", None))), None);
        assert_eq!(parse_trailer("no trailer"), None);
        assert_eq!(parse_trailer("x\n[completed soon, archive_event_id abc]"), None);
    }

    #[test]
    fn the_trailer_says_how_long_the_call_took_and_still_parses() {
        use chrono::TimeZone;
        let at = Local.with_ymd_and_hms(2026, 10, 5, 18, 42, 0).unwrap();
        let line = trailer(&at, Some(Duration::from_secs(252)), Some(123));
        assert_eq!(line, "\n[completed Mon 5 Oct 2026 18:42 local after 4 min 12 s, archive_event_id 123]");
        assert_eq!(event_id(&format!("body{line}")), Some(123), "masking still finds the event");
        assert_eq!(trailer(&at, None, None), "\n[completed Mon 5 Oct 2026 18:42 local]");
    }
}
