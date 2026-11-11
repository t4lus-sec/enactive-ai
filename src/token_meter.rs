//! Tokens per second for the bottom-right corner.
//!
//! While a reply streams, the corner shows the rate over the last two seconds.
//! With vLLM's continuous usage stats, every chunk carries the server's running
//! count of generated tokens, so the rate is measured. A server that only
//! reports totals at the end gives an estimate, `≈57 tok/s`, from characters
//! streamed, divided by a characters-per-token ratio that each finished request
//! calibrates. Between requests (a tool running, the idle prompt), the corner
//! shows the last request's generation rate: its tokens over the time from its
//! first to its last output.
//!
//! Beside it, the prefill rate: the prompt tokens the server actually had to
//! compute, over the time from the generation starting (`StreamEvent::Started`)
//! to its first output. With prefix caching most of an agent's prompt is
//! already cached, so dividing the whole prompt by that time would report
//! hundreds of thousands of tokens a second. When the server says how much was
//! cached (vLLM's `prompt_tokens_details.cached_tokens`: opt-in with
//! `--enable-prompt-tokens-details`, needs prefix caching on to be non-zero,
//! and left null by some vLLM releases even then; taken from whichever usage
//! chunk carries it), the rate is exact. Otherwise it is an
//! estimate, `≈`, of what was new since the previous request: this prompt less
//! that request's prompt and completion, which holds while the conversation
//! only grows. When neither can be known (the first request without cache
//! counts, or a prompt that shrank because the context was folded), there is no
//! prefill rate rather than a wrong one. A prefill under `MIN_PREFILL_TOKENS` is
//! mostly fixed overhead, so it leaves the last rate standing.

use crate::transcript::DEFAULT_CHARS_PER_TOKEN;

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

/// The live rate is taken over this much of the stream.
const WINDOW: Duration = Duration::from_secs(2);
/// Less than this much stream is too little to give a rate.
const MIN_SPAN: Duration = Duration::from_millis(250);
/// Fewer new prompt tokens than this measures the server's fixed overhead per
/// request more than its prefill.
const MIN_PREFILL_TOKENS: u64 = 512;
/// A time to first output shorter than this is too short to divide by.
const MIN_PREFILL_SPAN: Duration = Duration::from_millis(10);

/// A generation rate, and whether it was counted by the server or estimated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rate {
    pub per_second: f64,
    pub counted: bool,
}

impl Rate {
    /// `57 tok/s`, `≈57 tok/s` when estimated, `7.5 tok/s` under ten.
    pub fn label(self) -> String {
        let mark = if self.counted { "" } else { "≈" };
        if self.per_second < 9.95 {
            format!("{mark}{:.1} tok/s", self.per_second)
        } else {
            format!("{mark}{:.0} tok/s", self.per_second)
        }
    }

    /// Prefill runs in thousands: `850 tok/s`, `3.1k tok/s`, `≈42k tok/s`.
    pub fn compact_label(self) -> String {
        let mark = if self.counted { "" } else { "≈" };
        let rate = self.per_second;
        if rate < 999.5 {
            format!("{mark}{rate:.0} tok/s")
        } else if rate < 9_950.0 {
            format!("{mark}{:.1}k tok/s", rate / 1_000.0)
        } else {
            format!("{mark}{:.0}k tok/s", rate / 1_000.0)
        }
    }
}

/// The request being streamed.
struct Request {
    /// When the generation started on the wire (`StreamEvent::Started`): the
    /// prefill runs from here to `first`.
    started: Option<Instant>,
    /// The prompt as the server counted it, and how much of it its prefix
    /// cache already held, as the latest report had them.
    prompt_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    /// First and latest generated output.
    first: Option<Instant>,
    latest: Option<Instant>,
    /// What had been generated when `first` arrived, as the server's count
    /// where there is one and otherwise as characters. A rate is decode only,
    /// so the first output is the baseline the rest is measured against: the
    /// time it took to arrive is prefill, which is not decode.
    first_tokens: Option<f64>,
    first_chars: usize,
    /// Characters of reasoning, reply and tool-call arguments.
    chars: usize,
    /// The server's running count, and whether it sends one with every chunk.
    reported: Option<u64>,
    continuous: bool,
    /// (when, tokens so far), oldest first, for the window.
    samples: VecDeque<(Instant, f64)>,
}

impl Request {
    fn new() -> Self {
        Self {
            started: None,
            prompt_tokens: None,
            cached_tokens: None,
            first: None,
            latest: None,
            first_tokens: None,
            first_chars: 0,
            chars: 0,
            reported: None,
            continuous: false,
            samples: VecDeque::new(),
        }
    }

    fn tokens(&self, chars_per_token: f64) -> (f64, bool) {
        match self.reported {
            Some(reported) if self.continuous => (reported as f64, true),
            _ => (self.chars as f64 / chars_per_token, false),
        }
    }

    /// Output arrived at `now`. `sample`: it moves the count the window
    /// reads (text without the server's counts, or a count).
    fn output(&mut self, now: Instant, chars_per_token: f64, sample: bool) {
        if self.first.is_none() {
            self.first = Some(now);
            self.first_chars = self.chars;
            if self.continuous {
                self.first_tokens = self.reported.map(|reported| reported as f64);
            }
        }
        self.latest = Some(now);
        if !sample {
            return;
        }
        let (tokens, _) = self.tokens(chars_per_token);
        self.samples.push_back((now, tokens));
        while self.samples.front().is_some_and(|(at, _)| now.duration_since(*at) > WINDOW) && self.samples.len() > 2 {
            self.samples.pop_front();
        }
    }
}

pub struct TokenMeter {
    chars_per_token: f64,
    current: Option<Request>,
    last: Option<Rate>,
    last_prefill: Option<Rate>,
    /// The previous finished request's prompt plus completion, as the server
    /// counted them: what the next prompt starts with while the conversation
    /// only grows, so what its prefix cache should already hold.
    previous_context: Option<u64>,
}

impl Default for TokenMeter {
    fn default() -> Self {
        Self {
            chars_per_token: DEFAULT_CHARS_PER_TOKEN,
            current: None,
            last: None,
            last_prefill: None,
            previous_context: None,
        }
    }
}

impl TokenMeter {
    /// Characters per token as last calibrated; the thinking line's token
    /// count uses it too.
    pub fn chars_per_token(&self) -> f64 {
        self.chars_per_token
    }

    /// A model request was sent: the one before it is over.
    pub fn request(&mut self) {
        self.finish(None);
        self.current = Some(Request::new());
    }

    /// A generation started streaming at `now` (`StreamEvent::Started`): a
    /// fresh request whose prefill is timed from here. A retry of the same
    /// call starts over, so its prefill is measured on its own.
    pub fn started(&mut self, now: Instant) {
        self.request();
        if let Some(request) = self.current.as_mut() {
            request.started = Some(now);
        }
    }

    /// The server's count of the prompt, and of the part its prefix cache
    /// already held, when it sends one. Reports without a cache count leave an
    /// earlier one standing.
    pub fn prompt(&mut self, prompt_tokens: Option<u64>, cached_tokens: Option<u64>) {
        let request = self.current.get_or_insert_with(Request::new);
        if prompt_tokens.is_some() {
            request.prompt_tokens = prompt_tokens;
        }
        if cached_tokens.is_some() {
            request.cached_tokens = cached_tokens;
        }
    }

    /// Generated text arrived: reasoning, reply or tool-call arguments.
    pub fn text(&mut self, chars: usize, now: Instant) {
        if chars == 0 {
            return;
        }
        let chars_per_token = self.chars_per_token;
        let request = self.current.get_or_insert_with(Request::new);
        request.chars += chars;
        // With counts on every chunk, the count that follows this text is
        // what moves the rate.
        let sample = !request.continuous;
        request.output(now, chars_per_token, sample);
    }

    /// The server's count of tokens generated so far; `last`, its closing
    /// report, which ends the request.
    pub fn usage(&mut self, completion_tokens: u64, last: bool, now: Instant) {
        let chars_per_token = self.chars_per_token;
        let request = self.current.get_or_insert_with(Request::new);
        request.reported = Some(completion_tokens);
        if last {
            self.finish(Some(completion_tokens));
            return;
        }
        // A count mid-stream: the server reports as it goes.
        if !request.continuous {
            request.continuous = true;
            // Earlier samples were estimates; start the window, and the decode
            // baseline, from the server's first count instead.
            request.samples.clear();
            request.first = Some(now);
            request.first_chars = request.chars;
            request.first_tokens = Some(completion_tokens as f64);
        }
        request.output(now, chars_per_token, true);
    }

    /// The request is over (a tool runs, the pass ended).
    pub fn end_request(&mut self) {
        self.finish(None);
    }

    /// Close the current request: remember its rate, and learn the
    /// characters-per-token ratio from the server's total.
    fn finish(&mut self, total: Option<u64>) {
        let Some(request) = self.current.take() else { return };
        if let Some(prefill) = self.prefill_of(&request) {
            self.last_prefill = Some(prefill);
        }
        // Only a request the server counted in full says what the next prompt
        // will start with; anything else leaves the old figure, which the next
        // prompt being smaller than it will show to be stale.
        if let (Some(prompt), Some(completion)) = (request.prompt_tokens, total.or(request.reported)) {
            self.previous_context = Some(prompt + completion);
        }
        let (tokens, counted) = match total.or(request.reported) {
            Some(total) => (total as f64, true),
            None => request.tokens(self.chars_per_token),
        };
        // Enough of both to be worth learning from.
        if let Some(total) = total.filter(|total| request.chars >= 200 && *total >= 50) {
            let measured = request.chars as f64 / total as f64;
            self.chars_per_token = (0.5 * self.chars_per_token + 0.5 * measured).clamp(1.5, 8.0);
        }
        if let (Some(first), Some(latest)) = (request.first, request.latest) {
            let span = latest.duration_since(first);
            // Decode only: the tokens that arrived after the first output, over
            // the time between the first and the last. Where the server counted
            // the baseline, subtract it; otherwise take out the first output's
            // share of the characters, which is the same thing to the accuracy
            // an estimate has anyway.
            let decoded = match request.first_tokens {
                Some(first_tokens) => tokens - first_tokens,
                None if request.chars > 0 => {
                    tokens * (1.0 - request.first_chars as f64 / request.chars as f64)
                }
                None => tokens,
            }
            .max(0.0);
            if span >= MIN_SPAN && decoded >= 2.0 {
                self.last = Some(Rate { per_second: decoded / span.as_secs_f64(), counted });
            }
        }
    }

    /// The rate over the last two seconds of the request streaming now.
    pub fn live(&self, now: Instant) -> Option<Rate> {
        let request = self.current.as_ref()?;
        let latest = request.latest?;
        // Nothing for a while: not streaming (waiting on the first token of a
        // retry, or a stall). Fall back to the last request's rate.
        if now.duration_since(latest) > WINDOW {
            return None;
        }
        let start_index = request.samples.iter().position(|(at, _)| now.duration_since(*at) <= WINDOW)?;
        // One sample in the window measures no interval. Reporting the
        // difference against itself showed a confident "0.0 tok/s" through
        // every slow or stalling stream; the last known rate is the honest
        // answer until a second sample arrives.
        if start_index + 1 >= request.samples.len() {
            return None;
        }
        let (start, start_tokens) = request.samples[start_index];
        let (end, tokens_now) = *request.samples.back()?;
        // Between the samples, not up to now: time since the last chunk is
        // waiting, not decoding. A stream that has stopped is caught above.
        let span = end.duration_since(start);
        if span < MIN_SPAN {
            return None;
        }
        let (_, counted) = request.tokens(self.chars_per_token);
        Some(Rate { per_second: (tokens_now - start_tokens).max(0.0) / span.as_secs_f64(), counted })
    }

    /// What the corner shows now: the live rate while a reply streams, else
    /// the last request's.
    pub fn label(&self, now: Instant) -> Option<String> {
        self.live(now).or(self.last).map(Rate::label)
    }

    /// The last finished request's rate, for the idle prompt.
    pub fn last_label(&self) -> Option<String> {
        self.last.map(Rate::label)
    }

    /// `request`'s prefill rate, once its first output has arrived and its
    /// prompt has been counted. See the module notes for what is exact, what
    /// is estimated and what is left unsaid.
    fn prefill_of(&self, request: &Request) -> Option<Rate> {
        let span = request.first?.checked_duration_since(request.started?)?;
        let prompt = request.prompt_tokens?;
        let (computed, counted) = match request.cached_tokens {
            Some(cached) => (prompt.saturating_sub(cached), true),
            // Without the server's word, only growth since the last request
            // is known to be new. A prompt smaller than that was folded or
            // reset, and how much of it is cached cannot be told.
            None => match self.previous_context {
                Some(previous) if prompt >= previous => (prompt - previous, false),
                _ => return None,
            },
        };
        if computed < MIN_PREFILL_TOKENS || span < MIN_PREFILL_SPAN {
            return None;
        }
        Some(Rate { per_second: computed as f64 / span.as_secs_f64(), counted })
    }

    /// The prefill rate to show: the request streaming now once it has one,
    /// else the last request that gave one.
    pub fn prefill_label(&self) -> Option<String> {
        self.current
            .as_ref()
            .and_then(|request| self.prefill_of(request))
            .or(self.last_prefill)
            .map(Rate::compact_label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(base: Instant, millis: u64) -> Instant {
        base + Duration::from_millis(millis)
    }

    #[test]
    fn counted_streams_give_the_measured_rate() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        meter.request();
        assert_eq!(meter.label(base), None, "nothing streamed yet");
        // 50 tokens a second, reported with every chunk.
        for step in 0..=40u64 {
            let now = at(base, step * 100);
            meter.text(17, now);
            meter.usage(5 * step, false, now);
        }
        let live = meter.live(at(base, 4_000)).unwrap();
        assert!(live.counted && (live.per_second - 50.0).abs() < 1.0, "{live:?}");
        assert_eq!(live.label(), "50 tok/s");

        meter.usage(200, true, at(base, 4_000));
        let last = meter.last.unwrap();
        assert!(last.counted && (last.per_second - 50.0).abs() < 1.0, "{last:?}");
        // 41 chunks of 17 characters in 200 tokens, averaged with the starting guess.
        let measured = 41.0 * 17.0 / 200.0;
        assert!((meter.chars_per_token() - (0.5 * DEFAULT_CHARS_PER_TOKEN + 0.5 * measured)).abs() < 0.01);
        assert_eq!(meter.label(at(base, 9_000)).as_deref(), Some("50 tok/s"));
    }

    #[test]
    fn without_counts_the_rate_is_an_estimate_until_the_total_arrives() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        meter.request();
        for step in 0..=20u64 {
            meter.text(38, at(base, step * 100));
        }
        let live = meter.live(at(base, 2_000)).unwrap();
        assert!(!live.counted && (live.per_second - 100.0).abs() < 2.0, "{live:?}");
        assert!(live.label().starts_with('≈'));

        // The closing report gives the true count: 120 tokens, of which the
        // first chunk's share arrived before the measured span began, so the
        // rate is the 20 intervals' worth over 2 s, not all 120 over 2 s.
        meter.usage(120, true, at(base, 2_000));
        let last = meter.last.unwrap();
        assert!(last.counted && (last.per_second - 57.1).abs() < 1.0, "{last:?}");
        assert!(meter.chars_per_token() > DEFAULT_CHARS_PER_TOKEN, "798 chars in 120 tokens is 6.65 a token");
    }

    #[test]
    fn a_server_that_reports_nothing_still_leaves_an_estimate() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        meter.request();
        for step in 0..=10u64 {
            meter.text(38, at(base, step * 100));
        }
        meter.end_request();
        let last = meter.last.unwrap();
        assert!(!last.counted);
        // Eleven chunks, ten measured intervals in one second.
        assert_eq!(meter.last_label().as_deref(), Some("≈100 tok/s"));
        assert_eq!(meter.chars_per_token(), DEFAULT_CHARS_PER_TOKEN, "nothing to calibrate from");
    }

    #[test]
    fn a_stalled_stream_falls_back_to_the_last_rate() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        meter.request();
        meter.text(38, base);
        meter.text(38, at(base, 500));
        meter.end_request();
        let last = meter.last_label();
        meter.request();
        meter.text(38, at(base, 1_000));
        assert_eq!(meter.label(at(base, 1_100)), last, "too little of the new stream yet");
        meter.text(38, at(base, 1_500));
        assert!(meter.live(at(base, 1_600)).is_some());
        assert_eq!(meter.label(at(base, 5_000)), last, "nothing for over two seconds");
    }

    /// One generation: started at `base + start`, `prompt`/`cached` reported
    /// with the first output `ttft` later, then `tokens` more at 50 tok/s.
    fn generation(meter: &mut TokenMeter, base: Instant, start: u64, ttft: u64, prompt: u64, cached: Option<u64>, tokens: u64) {
        meter.started(at(base, start));
        meter.prompt(Some(prompt), cached);
        let steps = tokens / 5;
        for step in 0..=steps {
            let now = at(base, start + ttft + step * 100);
            meter.text(17, now);
            meter.usage(5 * step, false, now);
        }
        meter.usage(tokens, true, at(base, start + ttft + steps * 100));
    }

    #[test]
    fn with_the_servers_cache_count_prefill_is_the_uncached_part_over_the_time_to_first_output() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        // 40,000 prompt tokens, 36,000 of them cached: 4,000 computed in 0.5 s.
        generation(&mut meter, base, 0, 500, 40_000, Some(36_000), 100);
        assert_eq!(meter.prefill_label().as_deref(), Some("8.0k tok/s"));
        // The whole prompt over that time would have claimed 80k tok/s.
        assert_eq!(meter.last_label().as_deref(), Some("50 tok/s"), "decode is untouched");
    }

    #[test]
    fn without_it_prefill_is_estimated_from_what_the_conversation_grew_by() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        // First request, no cache count: nothing honest to say yet.
        generation(&mut meter, base, 0, 300, 10_000, None, 300);
        assert_eq!(meter.prefill_label(), None, "how much of the first prompt was cached is unknown");
        // The next prompt is the last one, its 300 generated tokens, and 2,000
        // new: 2,000 in 0.25 s, marked as an estimate.
        generation(&mut meter, base, 10_000, 250, 12_300, None, 100);
        assert_eq!(meter.prefill_label().as_deref(), Some("≈8.0k tok/s"));
    }

    #[test]
    fn a_folded_prompt_or_a_tiny_one_leaves_the_last_prefill_standing() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        generation(&mut meter, base, 0, 500, 40_000, Some(36_000), 100);
        assert_eq!(meter.prefill_label().as_deref(), Some("8.0k tok/s"));
        // The context was folded: smaller than before, with no cache count.
        generation(&mut meter, base, 10_000, 900, 9_000, None, 100);
        assert_eq!(meter.prefill_label().as_deref(), Some("8.0k tok/s"), "cannot tell, so says nothing new");
        // A short tool result: 200 new tokens is mostly the per-request overhead.
        generation(&mut meter, base, 20_000, 40, 9_300, Some(9_100), 100);
        assert_eq!(meter.prefill_label().as_deref(), Some("8.0k tok/s"));
    }

    #[test]
    fn prefill_shows_as_soon_as_the_first_output_arrives_and_a_retry_starts_its_own() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        meter.started(base);
        meter.prompt(Some(20_000), Some(18_000));
        assert_eq!(meter.prefill_label(), None, "still prefilling");
        // A retry of the same call: its own generation, timed from its own start.
        meter.started(at(base, 3_000));
        meter.prompt(Some(20_000), Some(18_000));
        meter.text(17, at(base, 3_400));
        assert_eq!(meter.prefill_label().as_deref(), Some("5.0k tok/s"), "2,000 in 0.4 s, not in 3.4 s");
    }

    #[test]
    fn prefill_labels_stay_short_in_the_thousands() {
        assert_eq!(Rate { per_second: 850.4, counted: true }.compact_label(), "850 tok/s");
        assert_eq!(Rate { per_second: 3_140.0, counted: true }.compact_label(), "3.1k tok/s");
        assert_eq!(Rate { per_second: 999.7, counted: true }.compact_label(), "1.0k tok/s");
        assert_eq!(Rate { per_second: 42_400.0, counted: false }.compact_label(), "≈42k tok/s");
    }

    #[test]
    fn labels_are_short() {
        assert_eq!(Rate { per_second: 57.4, counted: true }.label(), "57 tok/s");
        assert_eq!(Rate { per_second: 7.46, counted: false }.label(), "≈7.5 tok/s");
        assert_eq!(Rate { per_second: 9.96, counted: true }.label(), "10 tok/s");
    }
}

#[cfg(test)]
mod decode_only {
    use super::*;

    /// One generation: a prefill gap, then `chunks` chunks `chunk_ms` apart
    /// carrying `per_chunk` tokens each (and their text), then the closing
    /// report. Returns the rate the corner would keep afterwards.
    fn generation(
        meter: &mut TokenMeter,
        base: Instant,
        start_ms: u64,
        prefill_ms: u64,
        chunks: u64,
        chunk_ms: u64,
        per_chunk: u64,
        counted: bool,
    ) -> Option<Rate> {
        let mut total = 0;
        let first = start_ms + prefill_ms;
        for index in 0..chunks {
            let now = base + Duration::from_millis(first + index * chunk_ms);
            total += per_chunk;
            meter.text(4 * per_chunk as usize, now);
            if counted {
                meter.usage(total, false, now);
            }
        }
        meter.usage(total, true, base + Duration::from_millis(first + (chunks - 1) * chunk_ms));
        meter.last
    }

    /// The whole point of the number: it measures decode, so the time spent
    /// prefilling the prompt must not appear in it. Same stream, same decode
    /// rate, 8 s of prefill between them.
    #[test]
    fn prefill_is_not_in_the_rate() {
        let base = Instant::now();
        for prefill_ms in [0, 8_000] {
            let mut meter = TokenMeter::default();
            meter.request();
            let rate = generation(&mut meter, base, 0, prefill_ms, 40, 50, 3, true).unwrap();
            // 39 intervals of 50 ms carry 117 of the 120 tokens: 60 tok/s.
            assert!(rate.counted && (rate.per_second - 60.0).abs() < 0.1, "{prefill_ms}ms: {rate:?}");
        }
    }

    /// A call can stream more than once: `sample_until_answered` retries a
    /// broken stream and resamples an empty answer. Each generation pays its
    /// own prefill, and the second one's token counter starts again from zero.
    /// Measured as one stream, the abandoned attempt, the gap and the second
    /// prefill all landed in the denominator while only the last attempt's
    /// tokens were in the numerator.
    #[test]
    fn a_retried_call_is_measured_one_generation_at_a_time() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        meter.request();
        // A first attempt that decodes for two seconds and then breaks off,
        // with no closing report (that is what makes it malformed).
        let mut total = 0;
        for index in 0..40u64 {
            let now = base + Duration::from_millis(1_000 + index * 50);
            total += 3;
            meter.text(12, now);
            meter.usage(total, false, now);
        }
        // The retry announces itself (StreamEvent::Started -> request()).
        meter.request();
        let rate = generation(&mut meter, base, 3_000, 8_000, 40, 50, 3, true).unwrap();
        assert!(
            rate.counted && (rate.per_second - 60.0).abs() < 0.1,
            "the retry's own decode rate, not dragged down by the first attempt: {rate:?}"
        );
    }

    /// A stream whose chunks are seconds apart used to read a confident
    /// "0.0 tok/s": the window held one sample, so the rate was measured
    /// against itself. The last known rate is the honest answer instead.
    #[test]
    fn a_sparse_stream_never_reads_zero() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        meter.request();
        let earlier = generation(&mut meter, base, 0, 0, 40, 50, 3, true).unwrap().label();

        meter.request();
        for (index, offset) in [0u64, 3_000, 6_000].iter().enumerate() {
            let now = base + Duration::from_millis(5_000 + offset);
            meter.text(40, now);
            meter.usage(10 * (index as u64 + 1), false, now);
            for probe in [100u64, 500, 1_500, 2_500] {
                let shown = meter.label(now + Duration::from_millis(probe));
                assert_eq!(shown.as_deref(), Some(earlier.as_str()), "{offset}ms + {probe}ms");
            }
        }
    }

    /// Waiting is not decoding: a pause inside a stream must not bend the live
    /// rate downwards while the chunks around it are arriving at full speed.
    #[test]
    fn the_live_rate_measures_between_chunks_not_up_to_now() {
        let base = Instant::now();
        let mut meter = TokenMeter::default();
        meter.request();
        for index in 0..10u64 {
            let now = base + Duration::from_millis(index * 50);
            meter.text(12, now);
            meter.usage(3 * (index + 1), false, now);
        }
        let streaming = meter.live(base + Duration::from_millis(500)).unwrap();
        assert!((streaming.per_second - 60.0).abs() < 0.1, "{streaming:?}");
        // Half a second later, with nothing new, the number is the same rather
        // than 500 ms of silence averaged into it.
        let paused = meter.live(base + Duration::from_millis(1_000)).unwrap();
        assert_eq!(paused, streaming);
    }
}
