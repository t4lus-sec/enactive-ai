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
}
