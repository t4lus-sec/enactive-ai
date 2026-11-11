//! P2: keep a subagent's transcript inside its window.
//!
//! P1 bounds one tool result; nothing bounded the sum. Every turn re-sends the
//! whole transcript, every old result included, so a worker's reservation is a
//! wall it reaches at the speed its own old output echoes. This is a transform
//! on the worker's message list before each request, with no new tool and no
//! instruction to the worker, in three steps of rising cost:
//!
//! 1. **Elide** (M1). Past `ELIDE_AT` of the usable window, old bulky results
//!    become one-line stubs in P1's grammar (`result_ref`, `omitted_range`),
//!    oldest first, down to `ELIDE_TO`. Past `HARD_AT`, smaller ones go too
//!    and only the latest turn is kept whole, down to `HARD_TO`. The brief
//!    (system prompt and task) is never touched.
//! 2. **Recall** (M2). A stub carries its handle, so `result_read` brings
//!    anything elided back. Elision is only safe because of that.
//! 3. **Summarize** (M3). Still past `HARD_AT` after eliding, the oldest middle
//!    stretch (after the brief, before the latest turn) is replaced by a
//!    summary the worker writes in one model call.
//!
//! Elision is in place and only ever goes one way, so the request's prefix
//! changes once per elision rather than every turn (the server's prefix cache
//! keeps working between them).

use crate::types::Message;
use std::error::Error;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Shares of the usable window (the reservation less the output cap).
pub(crate) const ELIDE_AT: f64 = 0.60;
const ELIDE_TO: f64 = 0.50;
pub(crate) const HARD_AT: f64 = 0.85;
const HARD_TO: f64 = 0.70;
/// Results smaller than this aren't worth a stub (a stub is about 330 chars).
const SOFT_MIN_CHARS: usize = 1_200;
const HARD_MIN_CHARS: usize = 450;
/// Assistant turns kept whole, with their results.
const SOFT_KEEP_TURNS: usize = 2;
const HARD_KEEP_TURNS: usize = 1;
/// The system prompt and the task: never elided or summarized.
const BRIEF: usize = 2;
/// Fewer messages than this in the middle isn't worth a model call.
const MIN_SUMMARY_REGION: usize = 3;
/// Nor fewer tokens: a summary is up to 2,048, so a smaller stretch (mostly
/// stubs already, or a summary and one turn) would come back no smaller.
const MIN_SUMMARY_TOKENS: usize = 6_144;
/// After a summary fails or saves nothing, the transcript must grow this much
/// before another is tried, so a stuck one isn't retried every turn.
const SUMMARY_RETRY_GROWTH: usize = 2_048;

pub(crate) const SUMMARY_INSTRUCTION: &str = "[HARNESS — CONTEXT SUMMARY, NOT A NEW TASK]\n\
Your context is nearly full. The part of your transcript after your task will be replaced by the summary you \
write now; your latest step and anything after it stay as they are. Write it as plain text and call no tools. \
Include: what you have established, with the specific evidence (exact paths, hostnames, versions, findings); \
commands already run and what they showed, with the result_ref of any result you may need again (result_read \
still works on it); the state you left on the machine; dead ends and why; and what you were about to do. Be dense: \
facts over narrative. Do not invent results.";

const SUMMARY_HEADER: &str = "[HARNESS — SUMMARY OF YOUR EARLIER WORK, written by you when your context filled. \
The transcript it replaces is kept in the harness's journal; the result_refs below still work with result_read.]\n";

/// Which steps run: `S0MA_WORKER_CONTEXT` = `off`, `elide` (M1 and M2) or
/// `full` (the default: all three). `off` is the unmanaged baseline, for
/// comparing runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    Elide,
    Full,
}

impl Mode {
    pub(crate) fn from_env() -> Self {
        Self::parse(std::env::var("S0MA_WORKER_CONTEXT").ok().as_deref())
    }

    fn parse(value: Option<&str>) -> Self {
        match value.map(|value| value.trim().to_ascii_lowercase()).as_deref() {
            Some("off" | "none" | "0") => Self::Off,
            Some("elide") => Self::Elide,
            _ => Self::Full,
        }
    }
}

/// The stub a tool message can be replaced by, if it can be brought back.
pub(crate) trait Stubs {
    fn stub(&self, message: &Message) -> Option<String>;
}

/// What carries over between passes: when a summary may be tried again.
#[derive(Debug, Default)]
pub(crate) struct State {
    /// The transcript's size when a summary last failed, saved nothing, or had
    /// too small a stretch to work on. Cleared when the window changes or the
    /// transcript shrinks below it (a compaction, a resize).
    summary_failed_at: Option<usize>,
    usable: usize,
}

/// What a pass did.
#[derive(Debug, Default)]
pub(crate) struct Managed {
    /// Input tokens after it.
    pub tokens: usize,
    /// The tool calls whose results were elided.
    pub elided: Vec<String>,
    /// Messages replaced by a summary, if one was written.
    pub summarized: Option<usize>,
    /// Why a summary that was due wasn't written.
    pub summary_failed: Option<String>,
}

/// Where the last `keep_turns` assistant turns start (never inside the brief).
pub(crate) fn recent_start(messages: &[Message], keep_turns: usize) -> usize {
    let mut seen = 0;
    for index in (BRIEF..messages.len()).rev() {
        if messages[index].role == "assistant" {
            seen += 1;
            if seen == keep_turns {
                return index;
            }
        }
    }
    BRIEF
}

/// M1: replace old tool results of at least `min_chars` with their stubs,
/// oldest first, until the transcript measures at most `target` tokens.
/// `tokens` is its size now. Returns its size after, and the calls elided.
fn elide(
    messages: &mut [Message],
    stubs: &dyn Stubs,
    keep_turns: usize,
    min_chars: usize,
    target: usize,
    mut tokens: usize,
    count: &mut dyn FnMut(&[Message]) -> Result<usize>,
) -> Result<(usize, Vec<String>)> {
    let start = recent_start(messages, keep_turns);
    let candidates: Vec<usize> = (BRIEF.min(start)..start)
        .filter(|&index| {
            let message = &messages[index];
            message.role == "tool" && message.content.chars().count() >= min_chars && !crate::llm::is_stub(&message.content)
        })
        .collect();
    let mut elided = Vec::new();
    let mut next = 0;
    // Estimate with the transcript's own chars-per-token, then measure; a few
    // rounds at most, since each measurement is a tokenizer call.
    for _ in 0..4 {
        if tokens <= target || next >= candidates.len() {
            break;
        }
        // Replayed reasoning counts too, or the estimate would run high.
        let chars: usize = messages
            .iter()
            .map(|message| message.content.chars().count() + message.reasoning_content.as_ref().map_or(0, |r| r.chars().count()))
            .sum();
        let per_char = tokens as f64 / chars.max(1) as f64;
        let mut estimate = tokens as f64;
        let mut changed = false;
        while estimate > target as f64 && next < candidates.len() {
            let index = candidates[next];
            next += 1;
            let Some(stub) = stubs.stub(&messages[index]) else { continue };
            let size = messages[index].content.chars().count();
            // A stub that would save little isn't worth the handle hop.
            if stub.chars().count() * 10 > size * 7 {
                continue;
            }
            estimate -= (size - stub.chars().count()) as f64 * per_char;
            messages[index].content = stub;
            changed = true;
            if let Some(id) = &messages[index].tool_call_id {
                elided.push(id.clone());
            }
        }
        if !changed {
            break;
        }
        tokens = count(messages)?;
    }
    Ok((tokens, elided))
}

/// M1 alone, soft then hard as needed. For the middle of a batch of calls,
/// where a summary would cut the batch's own results.
pub(crate) fn elide_to_fit(
    messages: &mut [Message],
    stubs: &dyn Stubs,
    mode: Mode,
    usable: usize,
    tokens: usize,
    count: &mut dyn FnMut(&[Message]) -> Result<usize>,
) -> Result<Managed> {
    let mut managed = Managed { tokens, ..Managed::default() };
    if mode == Mode::Off || (tokens as f64) <= ELIDE_AT * usable as f64 {
        return Ok(managed);
    }
    let soft = (ELIDE_TO * usable as f64) as usize;
    let (tokens, mut elided) = elide(messages, stubs, SOFT_KEEP_TURNS, SOFT_MIN_CHARS, soft, tokens, count)?;
    managed.tokens = tokens;
    if (tokens as f64) > HARD_AT * usable as f64 {
        let hard = (HARD_TO * usable as f64) as usize;
        let (tokens, more) = elide(messages, stubs, HARD_KEEP_TURNS, HARD_MIN_CHARS, hard, tokens, count)?;
        managed.tokens = tokens;
        elided.extend(more);
    }
    managed.elided = elided;
    Ok(managed)
}

/// Everything before a request: M1, then M3 if still past `HARD_AT` (in
/// `Full` mode). `summarize` gets the request for the summary (the transcript
/// up to the latest turn, then `SUMMARY_INSTRUCTION`) and returns its text.
/// A summary that fails leaves the transcript as it was.
#[allow(clippy::too_many_arguments)]
pub(crate) fn manage(
    messages: &mut Vec<Message>,
    state: &mut State,
    stubs: &dyn Stubs,
    mode: Mode,
    usable: usize,
    tokens: usize,
    count: &mut dyn FnMut(&[Message]) -> Result<usize>,
    summarize: &mut dyn FnMut(&[Message]) -> Result<String>,
) -> Result<Managed> {
    let mut managed = elide_to_fit(messages, stubs, mode, usable, tokens, count)?;
    if mode != Mode::Full || (managed.tokens as f64) <= HARD_AT * usable as f64 {
        return Ok(managed);
    }
    // Not again until the transcript has grown, after one that didn't help.
    if state.usable != usable || state.summary_failed_at.is_some_and(|at| managed.tokens < at) {
        state.summary_failed_at = None;
    }
    state.usable = usable;
    if state.summary_failed_at.is_some_and(|at| managed.tokens < at + SUMMARY_RETRY_GROWTH) {
        return Ok(managed);
    }
    let start = recent_start(messages, HARD_KEEP_TURNS);
    let earlier_summary = messages.get(BRIEF).is_some_and(|message| message.content.starts_with(SUMMARY_HEADER));
    if start.saturating_sub(BRIEF) < MIN_SUMMARY_REGION + usize::from(earlier_summary) {
        return Ok(managed);
    }
    // Only a stretch big enough to come back smaller.
    let region = count(&messages[..start])?.saturating_sub(count(&messages[..BRIEF])?);
    if region < MIN_SUMMARY_TOKENS {
        // Measured once; not again until it could have grown enough.
        state.summary_failed_at = Some(managed.tokens);
        return Ok(managed);
    }
    let mut request = messages[..start].to_vec();
    request.push(Message::user(SUMMARY_INSTRUCTION.to_owned()));
    match summarize(&request) {
        Ok(summary) => {
            let replaced: Vec<Message> =
                messages.splice(BRIEF..start, [Message::user(format!("{SUMMARY_HEADER}{}", summary.trim()))]).collect();
            let after = count(messages)?;
            if after < managed.tokens {
                managed.summarized = Some(replaced.len());
                managed.tokens = after;
                state.summary_failed_at = None;
            } else {
                // It saved nothing: put the stretch back.
                messages.splice(BRIEF..BRIEF + 1, replaced);
                managed.summary_failed = Some(format!("the summary measured {after} tokens against {} before; kept the transcript", managed.tokens));
                state.summary_failed_at = Some(managed.tokens);
            }
        }
        Err(error) => {
            managed.summary_failed = Some(error.to_string());
            state.summary_failed_at = Some(managed.tokens);
        }
    }
    Ok(managed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionCall, ToolCall};

    /// Any tool message whose call is in `known` can be stubbed.
    struct Known(Vec<String>);

    impl Stubs for Known {
        fn stub(&self, message: &Message) -> Option<String> {
            let id = message.tool_call_id.as_deref()?;
            self.0.iter().any(|known| known == id).then(|| crate::llm::worker_stub(id, message.content.chars().count(), None))
        }
    }

    fn count(messages: &[Message]) -> Result<usize> {
        Ok(messages.iter().map(|message| message.content.chars().count() / 4 + 4).sum())
    }

    /// A brief, then `turns` assistant turns, each with one result of `size`.
    fn transcript(turns: usize, size: usize) -> (Vec<Message>, Known) {
        let mut messages = vec![Message::system("you are a worker".into()), Message::user("{\"task\":\"find it\"}".into())];
        let mut known = Vec::new();
        for turn in 0..turns {
            let id = format!("call-{turn}");
            let call = ToolCall {
                id: id.clone(),
                kind: "function".into(),
                function: FunctionCall { name: "vm_exec".into(), arguments: "{}".into() },
            };
            messages.push(Message::assistant_with_tools(format!("step {turn}"), vec![call]));
            messages.push(Message::tool_bounded(id.clone(), format!("{turn}:{}", "x".repeat(size))));
            known.push(id);
        }
        (messages, Known(known))
    }

    fn never(_: &[Message]) -> Result<String> {
        panic!("no summary expected")
    }

    #[test]
    fn below_the_threshold_nothing_changes() {
        let (mut messages, stubs) = transcript(4, 2_000);
        let before: Vec<String> = messages.iter().map(|message| message.content.clone()).collect();
        let tokens = count(&messages).unwrap();
        let managed = manage(&mut messages, &mut State::default(), &stubs, Mode::Full, tokens * 2, tokens, &mut count, &mut never).unwrap();
        assert!(managed.elided.is_empty());
        assert_eq!(messages.iter().map(|message| message.content.clone()).collect::<Vec<_>>(), before);
    }

    #[test]
    fn old_results_become_stubs_oldest_first_and_the_brief_and_recent_turns_stay() {
        let (mut messages, stubs) = transcript(8, 4_000);
        let tokens = count(&messages).unwrap();
        // Past 0.6 of the window, short of 0.85.
        let usable = (tokens as f64 / 0.7) as usize;
        let managed = manage(&mut messages, &mut State::default(), &stubs, Mode::Full, usable, tokens, &mut count, &mut never).unwrap();
        assert!(managed.tokens as f64 <= ELIDE_TO * usable as f64, "{} of {usable}", managed.tokens);
        assert_eq!(managed.elided.first().map(String::as_str), Some("call-0"));
        // The brief is untouched, and the last two turns are whole.
        assert_eq!(messages[1].content, "{\"task\":\"find it\"}");
        let tail: Vec<&Message> = messages.iter().filter(|message| message.role == "tool").rev().take(2).collect();
        assert!(tail.iter().all(|message| !crate::llm::is_stub(&message.content)));
        // A stub carries the handle in P1's grammar.
        let stub: serde_json::Value = serde_json::from_str(&messages[3].content).unwrap();
        assert_eq!((stub["elided"].as_bool(), stub["result_ref"].as_str()), (Some(true), Some("call-0")));
        assert!(stub["note"].as_str().unwrap().contains("result_read(ref=\"call-0\""));
        // A second pass changes nothing more (stubs stay stubs).
        let again = manage(&mut messages, &mut State::default(), &stubs, Mode::Full, usable, managed.tokens, &mut count, &mut never).unwrap();
        assert!(again.elided.is_empty());
    }

    #[test]
    fn past_the_hard_mark_only_the_latest_turn_stays_whole() {
        let (mut messages, stubs) = transcript(4, 6_000);
        let tokens = count(&messages).unwrap();
        // A window the latest two turns alone fill past 0.85.
        let two_turns = count(&[&messages[..2], &messages[messages.len() - 4..]].concat()).unwrap();
        let usable = (two_turns as f64 * 1.05) as usize;
        let managed = elide_to_fit(&mut messages, &stubs, Mode::Elide, usable, tokens, &mut count).unwrap();
        assert!(managed.tokens as f64 <= HARD_TO * usable as f64);
        assert_eq!(managed.elided, ["call-0", "call-1", "call-2"]);
        assert!(!crate::llm::is_stub(&messages.last().unwrap().content));
        // Off is the unmanaged baseline.
        let (mut messages, stubs) = transcript(4, 6_000);
        assert!(elide_to_fit(&mut messages, &stubs, Mode::Off, usable, tokens, &mut count).unwrap().elided.is_empty());
    }

    #[test]
    fn still_too_full_after_eliding_the_middle_is_summarized_once() {
        // Turns whose bulk isn't in results (long assistant text), so eliding can't help.
        let (mut messages, stubs) = transcript(6, 10);
        for message in messages.iter_mut().filter(|message| message.role == "assistant") {
            message.content = "thinking out loud ".repeat(300);
        }
        let tokens = count(&messages).unwrap();
        let usable = (tokens as f64 / 0.9) as usize;
        let mut asked = Vec::new();
        let mut summarize = |request: &[Message]| -> Result<String> {
            asked.push(request.to_vec());
            Ok("Established: it is in /var/log. Next: read w1.3 again.".into())
        };
        let managed = manage(&mut messages, &mut State::default(), &stubs, Mode::Full, usable, tokens, &mut count, &mut summarize).unwrap();
        assert_eq!(asked.len(), 1);
        // The request is the transcript up to the latest turn, then the instruction.
        let request = &asked[0];
        assert_eq!(request.last().unwrap().content, SUMMARY_INSTRUCTION);
        assert_eq!(request.len(), 2 + 10 + 1);
        // Brief, summary, latest turn.
        assert_eq!(managed.summarized, Some(10));
        assert_eq!(messages.len(), 2 + 1 + 2);
        assert!(messages[2].content.starts_with(SUMMARY_HEADER) && messages[2].content.contains("w1.3"));
        assert_eq!(messages[3].role, "assistant");
        assert!(managed.tokens < tokens);
        // In elide mode there is no summary; a failed one leaves things as they were.
        let (mut messages, stubs) = transcript(6, 10);
        for message in messages.iter_mut().filter(|message| message.role == "assistant") {
            message.content = "thinking out loud ".repeat(300);
        }
        let before = messages.len();
        assert!(manage(&mut messages, &mut State::default(), &stubs, Mode::Elide, usable, tokens, &mut count, &mut never).unwrap().summarized.is_none());
        let failed = manage(&mut messages, &mut State::default(), &stubs, Mode::Full, usable, tokens, &mut count, &mut |_: &[Message]| -> Result<String> {
            Err("the summary didn't fit".into())
        })
        .unwrap();
        assert_eq!((messages.len(), failed.summary_failed.as_deref()), (before, Some("the summary didn't fit")));
    }

    /// A long middle that elision can't shrink (assistant text, not results).
    fn chatty(turns: usize) -> (Vec<Message>, Known) {
        let (mut messages, stubs) = transcript(turns, 10);
        for message in messages.iter_mut().filter(|message| message.role == "assistant") {
            message.content = "thinking out loud ".repeat(300);
        }
        (messages, stubs)
    }

    #[test]
    fn a_summary_that_saves_nothing_is_undone_and_not_retried_every_turn() {
        let (mut messages, stubs) = chatty(10);
        let before = messages.clone();
        let tokens = count(&messages).unwrap();
        let usable = (tokens as f64 / 0.9) as usize;
        let mut state = State::default();
        let mut calls = 0;
        let mut bloated = |_: &[Message]| -> Result<String> {
            calls += 1;
            Ok("word ".repeat(20_000))
        };
        let managed = manage(&mut messages, &mut state, &stubs, Mode::Full, usable, tokens, &mut count, &mut bloated).unwrap();
        assert!(managed.summarized.is_none() && managed.summary_failed.is_some());
        assert_eq!(messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>(), before.iter().map(|m| m.content.clone()).collect::<Vec<_>>());
        // Next turn, no growth: not tried again.
        manage(&mut messages, &mut state, &stubs, Mode::Full, usable, tokens, &mut count, &mut bloated).unwrap();
        assert_eq!(calls, 1);
        // After a failure too.
        let mut state = State::default();
        let mut failing = 0;
        for _ in 0..3 {
            manage(&mut messages, &mut state, &stubs, Mode::Full, usable, tokens, &mut count, &mut |_: &[Message]| -> Result<String> {
                failing += 1;
                Err("cut off".into())
            })
            .unwrap();
        }
        assert_eq!(failing, 1);
    }

    #[test]
    fn a_small_middle_is_not_summarized() {
        // Past the hard mark, but the stretch before the latest turn is small.
        let (mut messages, stubs) = transcript(4, 300);
        messages.last_mut().unwrap().content = "x".repeat(40_000);
        let tokens = count(&messages).unwrap();
        let usable = (tokens as f64 / 0.95) as usize;
        let mut state = State::default();
        let counted = std::cell::Cell::new(0);
        let mut counting = |messages: &[Message]| -> Result<usize> {
            counted.set(counted.get() + 1);
            count(messages)
        };
        let managed = manage(&mut messages, &mut state, &stubs, Mode::Full, usable, tokens, &mut counting, &mut never).unwrap();
        assert!(managed.summarized.is_none() && managed.summary_failed.is_none());
        // Next turn it isn't measured again.
        let before = counted.get();
        manage(&mut messages, &mut state, &stubs, Mode::Full, usable, managed.tokens, &mut counting, &mut never).unwrap();
        assert_eq!(counted.get(), before);
        // A different window (a resize) clears the mark.
        manage(&mut messages, &mut state, &stubs, Mode::Full, usable - 1, managed.tokens, &mut counting, &mut never).unwrap();
        assert!(counted.get() > before);
    }

    #[test]
    fn the_mode_comes_from_the_environment_setting() {
        assert_eq!(Mode::parse(None), Mode::Full);
        assert_eq!(Mode::parse(Some(" OFF ")), Mode::Off);
        assert_eq!(Mode::parse(Some("elide")), Mode::Elide);
        assert_eq!(Mode::parse(Some("full")), Mode::Full);
    }
}
