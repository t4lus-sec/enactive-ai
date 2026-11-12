//! Bounds newly created model-visible tool messages; durable evidence is logged upstream.
use crate::{llm::{self, ChatOptions}, types::Message};
const LIMIT: usize = 8_000;
// A conservative UTF-8 byte bound avoids tokenization probes for small messages
// and bounds results if the local byte-level tokenizer service is unavailable.
const SAFE_BYTES: usize = 7_500;

pub(crate) fn limit_tool_output(content: String) -> String {
    limit_with(content, count_tokens)
}

fn count_tokens(text: &str) -> Result<usize, String> {
    llm::count_input_tokens(&[Message::user(text.to_owned())], &ChatOptions::normal(), None).map_err(|e| e.to_string())
}

/// Whether the result itself is incomplete: the host cut a stream when it
/// captured it (a VM result's `stdout_truncated`/`stderr_truncated`). Then the
/// kept result ends where the cut was, and what came after was never kept, so
/// no handle can bring it back. Said in so many words wherever the result is.
pub(crate) fn source_truncation(content: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    let flagged = |key: &str| value.get(key) == Some(&serde_json::Value::Bool(true));
    let mut cut: Vec<&str> = ["stdout", "stderr"]
        .into_iter()
        .filter(|stream| flagged(&format!("{stream}_truncated")))
        .collect();
    // A shell snapshot (shell_read) says it its own way.
    if flagged("output_truncated_by_host") {
        cut.push("the shell output");
    }
    (!cut.is_empty()).then(|| {
        format!(
            "{} was cut by the host when it was captured: the kept result ends at the cut, and what the command \
             printed after it was never kept, so result_read can't return it. Re-run narrower (grep, tail) to see it.",
            cut.join(" and ")
        )
    })
}

/// How much of one of its tool results a subagent sees, in characters. Its
/// context is small, so this is tighter than the token cap.
pub(crate) const WORKER_VISIBLE_CHARS: usize = 8_000;

/// A subagent's view of one of its tool results. Within the cap it is the
/// result exactly. Over it (too many characters, or too many tokens for dense
/// text), the head and the tail with the cut points labelled in characters of
/// the result, and `result_ref`, which reads the omitted middle back with
/// `result_read`: the loss is queryable instead of silent.
pub(crate) fn limit_worker_output(content: String, result_ref: &str) -> String {
    limit_worker_with(content, WORKER_VISIBLE_CHARS, result_ref, count_tokens)
}

/// Whether `text` fits the cap; if not, roughly what share of it would.
fn fit<F>(text: &str, count: &mut F) -> Result<(), f64>
where F: FnMut(&str) -> Result<usize, String> {
    if text.len() <= SAFE_BYTES {
        return Ok(());
    }
    match count(text) {
        Ok(tokens) if tokens <= LIMIT => Ok(()),
        Ok(tokens) => Err(LIMIT as f64 * 0.95 / tokens as f64),
        // No tokenizer: the byte bound `fallback` uses.
        Err(_) => Err(SAFE_BYTES as f64 * 0.95 / text.len() as f64),
    }
}

/// `amount` cut down by `share`, and always by something (never to nothing).
fn shrink(amount: usize, share: f64) -> usize {
    ((amount as f64 * share) as usize).min(amount.saturating_sub(amount / 10).saturating_sub(1)).max(1)
}

fn limit_worker_with<F>(content: String, max_chars: usize, result_ref: &str, mut count: F) -> String
where F: FnMut(&str) -> Result<usize, String> {
    let total = content.chars().count();
    let incomplete = source_truncation(&content);
    let mut shown = if total > max_chars {
        max_chars
    } else {
        match fit(&content, &mut count) {
            Ok(()) => return content,
            Err(share) => shrink(total, share),
        }
    };
    loop {
        let candidate = worker_preview(&content, total, shown, result_ref, incomplete.as_deref());
        match fit(&candidate, &mut count) {
            Err(share) if shown > 64 => shown = shrink(shown, share),
            _ => return candidate,
        }
    }
}

/// A worker's view of a result over its cap. Fields serialize in this order:
/// that it was cut, the ref and the cut first, then the text.
#[derive(serde::Serialize)]
struct WorkerPreview<'a> {
    truncated: bool,
    result_ref: &'a str,
    total_chars: usize,
    omitted_range: [usize; 2],
    /// The result itself is incomplete (see `source_truncation`).
    #[serde(skip_serializing_if = "Option::is_none")]
    source_truncated: Option<&'a str>,
    note: String,
    head_range: [usize; 2],
    head: String,
    tail_range: [usize; 2],
    tail: String,
}

/// `shown` characters of `content` (three quarters head, one quarter tail),
/// and where everything is, in characters of the result.
fn worker_preview(content: &str, total: usize, shown: usize, result_ref: &str, incomplete: Option<&str>) -> String {
    let shown = shown.min(total);
    let from = shown * 3 / 4;
    let to = total - (shown - from);
    serde_json::to_string(&WorkerPreview {
        truncated: true,
        result_ref,
        total_chars: total,
        omitted_range: [from, to],
        source_truncated: incomplete,
        note: format!(
            "Characters {from}..{to} of {total} are omitted here, not summarized. They are kept: \
             result_read(ref=\"{result_ref}\", pattern=...) finds lines in them, and \
             result_read(ref=\"{result_ref}\", offset={from}) reads on from the cut. A narrower command \
             works too when running it again is cheap and safe. Do not infer absence from this preview."
        ),
        head_range: [0, from],
        head: content.chars().take(from).collect(),
        tail_range: [to, total],
        tail: content.chars().skip(to).collect(),
    })
    .expect("a preview serializes")
}

/// What replaces a result elided from a subagent's context to make room (P2):
/// the P1 envelope's grammar with nothing shown, one line.
#[derive(serde::Serialize)]
struct WorkerStub<'a> {
    truncated: bool,
    elided: bool,
    result_ref: &'a str,
    total_chars: usize,
    omitted_range: [usize; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    source_truncated: Option<&'a str>,
    note: String,
}

/// The stub for a kept VM result (`total_chars` long) elided from context.
pub(crate) fn worker_stub(result_ref: &str, total_chars: usize, incomplete: Option<&str>) -> String {
    serde_json::to_string(&WorkerStub {
        truncated: true,
        elided: true,
        result_ref,
        total_chars,
        omitted_range: [0, total_chars],
        source_truncated: incomplete,
        note: format!(
            "Elided from your context to make room, not summarized. result_read(ref=\"{result_ref}\", pattern=...) \
             finds lines in it; result_read(ref=\"{result_ref}\", offset=0) reads it back."
        ),
    })
    .expect("a stub serializes")
}

/// The stub for a `result_read` page elided from context: which part of
/// which result it was, to read again.
pub(crate) fn page_stub(result_ref: &str, from: usize, to: usize) -> String {
    serde_json::json!({
        "elided": true,
        "result_ref": result_ref,
        "omitted_range": [from, to],
        "note": format!(
            "A page of {result_ref} you read earlier (characters {from}..{to}), elided from your context to make \
             room. result_read(ref=\"{result_ref}\", offset={from}) reads it again."
        ),
    })
    .to_string()
}

/// The stub for a read-only call's result elided from context (a skill or
/// board read): the call, to make again.
pub(crate) fn reread_stub(tool: &str, arguments: &str) -> String {
    let arguments: String = arguments.chars().take(400).collect();
    serde_json::json!({
        "elided": true,
        "tool": tool,
        "arguments": arguments,
        "note": "Elided from your context to make room. The call only read; make it again if you need the result \
                 (a board read shows the board as it is now).",
    })
    .to_string()
}

/// Whether a tool message is already one of these stubs.
pub(crate) fn is_stub(content: &str) -> bool {
    content.starts_with('{') && content.len() < 1_024 && content.contains("\"elided\":true")
}

/// One page of a kept result for `result_read`: the characters from `offset`,
/// at most `limit` of them, fewer if that many would pass the token cap.
pub(crate) fn result_page(text: &str, result_ref: &str, offset: usize, limit: usize) -> String {
    result_page_with(text, result_ref, offset, limit, count_tokens)
}

fn result_page_with<F>(text: &str, result_ref: &str, offset: usize, limit: usize, mut count: F) -> String
where F: FnMut(&str) -> Result<usize, String> {
    let total = text.chars().count();
    let incomplete = source_truncation(text);
    // Past the end is an answer, not an empty success: an empty page would
    // read as "nothing there" when the result simply ends (or was cut).
    if offset >= total && total > 0 {
        let error = match &incomplete {
            Some(why) => format!("the kept result {result_ref} ends at character {total}. {why}"),
            None => format!("{result_ref} has {total} characters; offset {offset} is past its end"),
        };
        return serde_json::json!({"ok": false, "result_ref": result_ref, "total_chars": total, "error": error}).to_string();
    }
    let mut take = limit.max(1).min(total - offset);
    loop {
        let end = offset + take;
        let mut page = serde_json::json!({
            "ok": true,
            "result_ref": result_ref,
            "total_chars": total,
            "range": [offset, end],
            "text": text.chars().skip(offset).take(take).collect::<String>(),
        });
        if end < total {
            page["next_offset"] = end.into();
        } else if let Some(why) = &incomplete {
            page["source_truncated"] = why.as_str().into();
        }
        let page = page.to_string();
        match fit(&page, &mut count) {
            Err(share) if take > 64 => take = shrink(take, share),
            _ => return page,
        }
    }
}

const MAX_MATCHES: usize = 20;
const MAX_MATCH_CHARS: usize = 300;

/// `result_read` with a pattern: the lines of a kept result that contain it
/// (case-insensitive, literal), from character `start` on (a line `start`
/// falls inside counts), each with the offset its line starts at. A VM result
/// is JSON, so an escaped `\n` inside it ends a line too, as it did in the
/// command's own output, and the pattern is looked for as written and as JSON
/// escapes it (quotes, backslashes, tabs). The list is cut to fit the token cap,
/// with where to resume.
pub(crate) fn result_matches(text: &str, result_ref: &str, pattern: &str, start: usize) -> String {
    result_matches_with(text, result_ref, pattern, start, count_tokens)
}

fn result_matches_with<F>(text: &str, result_ref: &str, pattern: &str, start: usize, mut count: F) -> String
where F: FnMut(&str) -> Result<usize, String> {
    let incomplete = source_truncation(text);
    let plain = pattern.to_lowercase();
    let escaped = serde_json::to_string(pattern).map_or_else(|_| plain.clone(), |quoted| {
        quoted[1..quoted.len() - 1].to_lowercase()
    });
    let mut matches = Vec::new();
    let mut more_from = None;
    for (line_start, line) in lines_with_offsets(text) {
        let chars = line.chars().count();
        if line_start + chars <= start && line_start < start {
            continue;
        }
        let lower = line.to_lowercase();
        let Some(byte_at) = lower.find(&plain).or_else(|| lower.find(&escaped)) else { continue };
        if matches.len() == MAX_MATCHES {
            more_from = Some(line_start);
            break;
        }
        let shown = if chars <= MAX_MATCH_CHARS {
            line.to_string()
        } else {
            let hit = lower[..byte_at].chars().count().min(chars);
            let from = hit.saturating_sub(MAX_MATCH_CHARS / 3);
            let window: String = line.chars().skip(from).take(MAX_MATCH_CHARS).collect();
            format!("{}{window}{}", if from > 0 { "…" } else { "" }, if from + MAX_MATCH_CHARS < chars { "…" } else { "" })
        };
        matches.push((line_start, shown));
    }
    let total = text.chars().count();
    let mut keep = matches.len();
    loop {
        let listed: Vec<serde_json::Value> = matches[..keep]
            .iter()
            .map(|(offset, line)| serde_json::json!({"offset": offset, "line": line}))
            .collect();
        let resume = if keep < matches.len() { Some(matches[keep].0) } else { more_from };
        let mut found = serde_json::json!({"ok": true, "result_ref": result_ref, "total_chars": total, "pattern": pattern,
            "matches": listed, "more_from_offset": resume});
        if let Some(why) = &incomplete {
            // No match here is not "not in the output": only what was kept was searched.
            found["source_truncated"] = why.as_str().into();
        }
        let found = found.to_string();
        match fit(&found, &mut count) {
            Err(share) if keep > 1 => keep = shrink(keep, share),
            _ => return found,
        }
    }
}

/// The lines of `text` with the character offset each starts at, breaking at
/// newlines and at JSON-escaped `\n` (an escaped backslash is not one).
fn lines_with_offsets(text: &str) -> Vec<(usize, String)> {
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut line_start = 0;
    let mut offset = 0;
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        offset += 1;
        match character {
            '\n' => {
                lines.push((line_start, std::mem::take(&mut line)));
                line_start = offset;
            }
            '\\' => match characters.peek() {
                Some('n') => {
                    characters.next();
                    offset += 1;
                    lines.push((line_start, std::mem::take(&mut line)));
                    line_start = offset;
                }
                Some('\\') => {
                    characters.next();
                    offset += 1;
                    line.push_str("\\\\");
                }
                _ => line.push('\\'),
            },
            other => line.push(other),
        }
    }
    lines.push((line_start, line));
    lines
}

fn preview(content: &str, budget: usize) -> String {
    let mut head_end = (budget * 3 / 4).min(content.len());
    while !content.is_char_boundary(head_end) { head_end -= 1; }
    let mut tail_start = content.len().saturating_sub(budget / 4).max(head_end);
    while !content.is_char_boundary(tail_start) { tail_start += 1; }
    // Characters too, so shown and omitted can be added up (bytes and tokens can't).
    let (head, tail) = (&content[..head_end], &content[tail_start..]);
    let total_chars = content.chars().count();
    serde_json::json!({
        "truncated": true,
        "tool_output_token_limit": LIMIT,
        "original_bytes": content.len(),
        "total_chars": total_chars,
        "omitted_chars": total_chars - head.chars().count() - tail.chars().count(),
        "head": head,
        "tail": tail,
        "note": "Middle omitted, not summarized. Full captured result is in the tool archive/log. Request a narrower range or query for missing detail; do not infer absence from this preview."
    }).to_string()
}

fn fallback(content: &str) -> String {
    let mut budget = SAFE_BYTES / 2;
    loop {
        let candidate = preview(content, budget);
        // Include JSON escaping and the omission marker in the byte budget.
        if candidate.len() <= SAFE_BYTES { return candidate; }
        budget /= 2;
    }
}

fn limit_with<F>(content: String, mut count: F) -> String
where F: FnMut(&str) -> Result<usize, String> {
    if content.len() <= SAFE_BYTES { return content; }
    // Avoid sending arbitrarily large tool dumps to the tokenizer endpoint.
    if content.len() <= 256_000 {
        match count(&content) {
            Ok(tokens) if tokens <= LIMIT => return content,
            Err(_) => return fallback(&content),
            _ => {}
        }
    }
    let mut budget = content.len().min(24_000);
    for _ in 0..3 {
        let candidate = preview(&content, budget);
        match count(&candidate) {
            Ok(tokens) if tokens <= LIMIT => return candidate,
            Ok(tokens) => budget = budget.saturating_mul(7_000) / tokens.max(1),
            Err(_) => break,
        }
    }
    fallback(&content)
}

/// When `limited` is a truncation preview of `original`, say exactly where the
/// rest is: the archive event, and the character range the preview left out,
/// in `archive_get`'s coordinates. `result_offset` is where the result starts
/// inside the archived record (after its tool name and arguments). Anything
/// that isn't one of this module's previews comes back unchanged.
pub(crate) fn locate_in_archive(limited: String, original: &str, event_id: u64, result_offset: usize) -> String {
    if limited == original {
        return limited;
    }
    let Ok(serde_json::Value::Object(mut preview)) = serde_json::from_str::<serde_json::Value>(&limited) else {
        return limited;
    };
    if preview.get("truncated") != Some(&serde_json::Value::Bool(true)) {
        return limited;
    }
    let chars = |key: &str| preview.get(key).and_then(|v| v.as_str()).map_or(0, |s| s.chars().count());
    let (head, tail) = (chars("head"), chars("tail"));
    let total = original.chars().count();
    let from = result_offset + head;
    let to = result_offset + total.saturating_sub(tail).max(head);
    preview.insert("archive_event_id".into(), event_id.into());
    preview.insert("omitted_range".into(), serde_json::json!([from, to]));
    preview.insert(
        "note".into(),
        format!(
            "Middle omitted, not summarized. The full result is archived as event {event_id}: \
             archive_get(id={event_id}, offset={from}) pages through the omitted part, and \
             archive_get(id={event_id}, pattern=...) finds lines in it. Do not infer absence from this preview."
        )
        .into(),
    );
    serde_json::Value::Object(preview).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_pages_read_back_in_order_and_fit_the_cap() {
        let original: String = (0..2_000).map(|i| format!("{i}\n")).collect();
        let total = original.chars().count();
        let mut offset = 0;
        let mut read = String::new();
        loop {
            let page: serde_json::Value =
                serde_json::from_str(&result_page_with(&original, "w1.1", offset, 1_500, |s| Ok(s.len() / 4))).unwrap();
            assert_eq!(page["range"][0], offset);
            read.push_str(page["text"].as_str().unwrap());
            match page["next_offset"].as_u64() {
                Some(next) => offset = next as usize,
                None => break,
            }
        }
        assert_eq!(read, original);
        // Too many tokens for one page: a shorter page, and where to go on.
        let page: serde_json::Value =
            serde_json::from_str(&result_page_with(&"界".repeat(9_000), "w1.1", 0, 8_000, |s| Ok(s.chars().count() * 2))).unwrap();
        assert!(page["next_offset"].as_u64().unwrap() < 8_000);
        // Past the end: said so, not an empty success.
        let page: serde_json::Value = serde_json::from_str(&result_page_with(&original, "w1.1", total + 5, 100, |_| Ok(1))).unwrap();
        assert_eq!(page["ok"], false);
        assert!(page["error"].as_str().unwrap().contains("past its end"));
    }

    #[test]
    fn measured_output_under_limit_remains_complete() {
        let text = "a".repeat(12_000);
        assert_eq!(limit_with(text.clone(), |_| Ok(3_000)), text);
    }
    #[test]
    fn unicode_preview_and_marker_share_the_budget() {
        let result = limit_with("界\n\"".repeat(20_000), |s| Ok(s.len()));
        assert!(result.len() <= LIMIT);
        let value: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(value["truncated"], true);
        assert!(!value["tail"].as_str().unwrap().is_empty());
    }
    #[test]
    fn unavailable_tokenizer_preserves_valid_bounded_json() {
        let result = limit_with("\u{0001}界".repeat(20_000), |_| Err("offline".into()));
        assert!(result.len() <= SAFE_BYTES);
        assert_eq!(serde_json::from_str::<serde_json::Value>(&result).unwrap()["truncated"], true);
    }
}
