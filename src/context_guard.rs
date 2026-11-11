use crate::llm::{self, ChatOptions, ToolDefinition};
use crate::types::Message;

// ============================================================
// GRADED WORKING MEMORY — FOLD THRESHOLDS
// ============================================================
//
// The old model was clear-and-rebuild: at 240k tokens the whole
// message list was dropped and one fresh system message took its place, which
// replaced about 80% of the working context in a single step.
//
// The fold model never clears anything. The oldest live turns are replaced by a
// first-person gist while everything newer stays exactly where it was, so
// consecutive moments share nearly all of their context.

/// Start writing gists for the oldest live turns on a background thread
/// (`memory_fold::FoldPrep`). Nothing in the active context changes here; this
/// only buys a prepared fold, so the fold at `FOLD_AT_TOKENS` doesn't have to
/// stop and write them.
pub const FOLD_PREPARE_TOKENS: usize = 200_000;

/// Apply the fold before the next model call: the prepared one if it still
/// matches the context, otherwise one written on the spot.
pub const FOLD_AT_TOKENS: usize = 220_000;

/// The measured size a fold brings the *whole* context down to — prefix, memory
/// message and remaining live turns together.
///
/// Set by prefill speed, not by taste. Every fold re-prefills everything that
/// survives it (dropping the oldest turns invalidates the cache for all of it),
/// so folding little and often is the expensive schedule: 220k → 180k costs ~22 s
/// every ~40k of activity (~56 s per 100k), where 220k → 75k costs ~9 s every
/// ~145k (~6.5 s per 100k). Revisit when the n-gram table moves to system RAM and
/// prefill goes from ~8k to ~28k tokens/s.
pub const FOLD_TO_TOKENS: usize = 75_000;

/// The most KV a fold asks for when it borrows a maintenance lane for one gist or
/// merge call (`memory_fold::wait_for_lease`).
///
/// Borrowed per call, not reserved: nothing is carved out of the subagent pool
/// permanently, so a fold can wait for a lane when subagents hold them all. See
/// `memory_fold::LEASE_WAIT` for why that trade was made.
pub const FOLD_RESERVE_TOKENS: usize = 75_000;

/// A fold covers ~145k of turns. Rather than one enormous gist, it writes one per
/// episode of about this size.
pub const GIST_EPISODE_TOKENS: usize = 40_000;

// ---- memory-message layer budgets (enforced by the merge cascade) ----

/// Open loops, always present, never folded.
pub const LOOPS_BUDGET_TOKENS: usize = 2_000;

/// Episode gists (~10:1). Over budget, the oldest two merge into one Shape entry.
pub const GIST_BUDGET_TOKENS: usize = 16_000;

/// Shape entries (~5:1 on gists). Over budget, the oldest two merge into one, so a
/// long day compresses further instead of losing its beginning.
pub const SHAPE_BUDGET_TOKENS: usize = 6_000;

/// The NOT SETTLED list: open questions, untested conclusions and unchecked
/// reports, held apart from the gists so no merge can drop or harden them.
/// Over budget, the oldest entries collapse into a pointer that is still marked
/// open (see `MemoryLayers::enforce_item_budget`).
pub const OPEN_ITEMS_BUDGET_TOKENS: usize = 2_000;

/// Ceiling for the fixed prefix (system prompt, self-model pointer, threads, skill
/// and world-model catalogs). Checked at startup and at every fold; exceeding it is
/// a warning, not a failure, because the prefix is not something a fold can shrink.
///
/// This is a ceiling, not a measurement: re-measure the real prefix and lower it.
/// It is load-bearing arithmetic — prefix plus memory message is the floor a fold
/// cannot get under, so every token here is a token the live layer doesn't get.
pub const PREFIX_BUDGET_TOKENS: usize = 24_000;

/// Everything a fold can't shrink apart from the prefix: the memory message
/// (gists, Shape, the NOT SETTLED list) and the loops block appended at the end.
/// A fold that lands on target leaves the live layer roughly `FOLD_TO_TOKENS`
/// minus the prefix and this.
/// (The 262,144 window's; `FoldSchedule::memory_message_budget` has each
/// window's, and the tests hold the short one to this.)
#[allow(dead_code)]
pub const MEMORY_MESSAGE_BUDGET_TOKENS: usize =
    LOOPS_BUDGET_TOKENS + GIST_BUDGET_TOKENS + SHAPE_BUDGET_TOKENS + OPEN_ITEMS_BUDGET_TOKENS;

/// The live layer a fold must leave behind in the worst case, where both the
/// prefix and the memory message are at their ceilings.
///
/// A fold exists to preserve continuity: if it can land on target while leaving
/// only a couple of turns verbatim, it has reinvented the clear-and-rebuild it
/// replaced. `fold_budget_is_coherent` holds the three constants to this.
pub const MIN_LIVE_TURNS_TOKENS: usize = 16_000;

// ---- a long window (ds41 at 1,048,576) ----
//
// On a window with room for it the context grows to 500k before it folds, and
// keeps more memory when it does. Folding then is about time, not room: every
// fold prefills what survives it again (~1,500 tokens/s on ds41, so ~85 s for
// the 128k a fold lands on), and folding at 220k meant one of those every
// ~145k of work. At 500k it is one every ~370k. The 262,144-token window keeps
// the schedule above unchanged.

/// Gists for a long window's next fold start being written here.
pub const LONG_FOLD_PREPARE_TOKENS: usize = 450_000;

/// A long window folds from here on, as soon as the gists written in the
/// background are ready: it never stops a turn to wait for them before
/// `LONG_FOLD_CEILING_TOKENS`. ds41 serves one sequence at a time, so gists
/// written beside a busy turn can take a while.
pub const LONG_FOLD_AT_TOKENS: usize = 500_000;

/// The most a long window waits for background gists: here the fold happens
/// whatever is ready, writing the rest on the spot.
pub const LONG_FOLD_CEILING_TOKENS: usize = 650_000;

/// What a long window's fold lands on. Higher than `FOLD_TO_TOKENS` because the
/// memory message it keeps is bigger (`LONG_GIST_BUDGET_TOKENS`,
/// `LONG_SHAPE_BUDGET_TOKENS`); the live turns still get `MIN_LIVE_TURNS_TOKENS`.
pub const LONG_FOLD_TO_TOKENS: usize = 128_000;

/// Episode gists on a long window: three times as many before the oldest merge.
pub const LONG_GIST_BUDGET_TOKENS: usize = 48_000;

/// Shape entries on a long window, in the same proportion to the gists.
pub const LONG_SHAPE_BUDGET_TOKENS: usize = 16_000;

/// How much one step of a turn can add after the last check before a model
/// call (a large tool result, the reply): room left above the ceiling for it.
pub const LONG_STEP_GROWTH_TOKENS: usize = 64_000;

/// The smallest window the long schedule is used on: the ceiling, one step's
/// growth past it, and a full output reservation.
pub const LONG_WINDOW_MIN_TOKENS: usize =
    LONG_FOLD_CEILING_TOKENS + LONG_STEP_GROWTH_TOKENS + OUTPUT_RESERVATION_TOKENS + TURN_HEADROOM_TOKENS;

/// When the context folds and how much memory it keeps, for one window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldSchedule {
    /// Gists for the next fold start being written in the background.
    pub prepare: usize,
    /// Fold once the background gists are ready.
    pub fold_at: usize,
    /// Fold whatever is ready, writing the rest on the spot. The same as
    /// `fold_at` on a short window: the fold is due there, as it always was.
    pub ceiling: usize,
    /// What a fold aims to bring the whole context down to.
    pub fold_to: usize,
    /// A fold made from gists written in the background may land up to here
    /// without writing more on the spot (the turns since the gists were
    /// planned stay verbatim): writing them would stop the turn, which is
    /// what preparing was for.
    pub land_within: usize,
    /// The merge cascade's budgets for episode gists and Shape entries.
    pub gist_budget: usize,
    pub shape_budget: usize,
}

impl FoldSchedule {
    /// The 262,144-token window's schedule, unchanged.
    pub const SHORT: Self = Self {
        prepare: FOLD_PREPARE_TOKENS,
        fold_at: FOLD_AT_TOKENS,
        ceiling: FOLD_AT_TOKENS,
        fold_to: FOLD_TO_TOKENS,
        land_within: FOLD_TO_TOKENS,
        gist_budget: GIST_BUDGET_TOKENS,
        shape_budget: SHAPE_BUDGET_TOKENS,
    };

    pub const LONG: Self = Self {
        prepare: LONG_FOLD_PREPARE_TOKENS,
        fold_at: LONG_FOLD_AT_TOKENS,
        ceiling: LONG_FOLD_CEILING_TOKENS,
        fold_to: LONG_FOLD_TO_TOKENS,
        land_within: LONG_FOLD_TO_TOKENS + (LONG_FOLD_AT_TOKENS - LONG_FOLD_TO_TOKENS) / 2,
        gist_budget: LONG_GIST_BUDGET_TOKENS,
        shape_budget: LONG_SHAPE_BUDGET_TOKENS,
    };

    pub fn for_window(window: usize) -> Self {
        if window >= LONG_WINDOW_MIN_TOKENS { Self::LONG } else { Self::SHORT }
    }

    /// Where a fold stops: `land_within` when it was made from gists written in
    /// the background, the target when it was written on the spot.
    pub fn lands_within(&self, prepared: bool) -> usize {
        if prepared { self.land_within } else { self.fold_to }
    }

    /// The memory message's budget: everything but the gists and Shape is the
    /// same on every window.
    pub fn memory_message_budget(&self) -> usize {
        LOOPS_BUDGET_TOKENS + self.gist_budget + self.shape_budget + OPEN_ITEMS_BUDGET_TOKENS
    }

    /// How much of the KV pool the main turn can come to hold on `window`:
    /// the whole window when it is the short one (as it always was), the
    /// ceiling with a step past it and an output on a long one.
    pub fn main_reserve(&self, window: usize) -> usize {
        if *self == Self::LONG {
            LONG_WINDOW_MIN_TOKENS.min(window)
        } else {
            crate::subagent::MAIN_CONTEXT_RESERVE
        }
    }
}

/// The window the server actually enforces. The live value comes from
/// `llm::slot_context_capacity`; this is the nominal figure used for planning and
/// logging when no server round trip is wanted.
#[allow(dead_code)]
pub const NOMINAL_WINDOW_TOKENS: usize = 262_144;

/// The largest output reservation any profile asks for (`ChatOptions::xhigh`).
#[allow(dead_code)]
pub const OUTPUT_RESERVATION_TOKENS: usize = 32_768;

/// Planning ceiling for prompt tokens: the window less a full output reservation
/// and the headroom. A model call projected past this waits for a fold to land.
/// Replaces the 240k emergency swap — nothing is ever swapped out wholesale.
/// Test-only in practice, and that is the point: the live check reads the
/// window from the server (`ContextUsage::capacity`) rather than trusting a
/// constant, so this exists to pin the invariant that `FOLD_AT_TOKENS` leaves
/// room to land before calls start waiting. `NOMINAL_WINDOW_TOKENS` and
/// `OUTPUT_RESERVATION_TOKENS` are dead for the same reason — they feed this.
#[allow(dead_code)]
pub const HARD_LIMIT_TOKENS: usize =
    NOMINAL_WINDOW_TOKENS - OUTPUT_RESERVATION_TOKENS - TURN_HEADROOM_TOKENS;

/// Slack between a projected request and the server's limit, so an estimate that
/// is off by a few tokens still lands inside the window.
pub const TURN_HEADROOM_TOKENS: usize = 2_048;

/// What the context wants done before the next model call.
///
/// Ordered by urgency: each variant implies everything milder than it, so callers
/// can match on the strongest condition and ignore the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FoldNeed {
    /// Below the prepare threshold. Carry on.
    None,
    /// Past `FOLD_PREPARE_TOKENS`: start writing gists in the background. The
    /// active context is untouched.
    Prepare,
    /// Past the schedule's `fold_at`: fold at the next boundary once the gists
    /// written in the background are ready. A short window has no such band
    /// (its fold is `Due` there).
    Fold,
    /// Past the schedule's `ceiling`: fold at the next boundary, waiting for
    /// the background gists or writing them now. There is still room to finish
    /// what's in flight first.
    Due,
    /// The call as projected would not fit. It waits here: the prepared fold is
    /// applied, or one starts immediately and s0ma pauses until it lands.
    Wait,
}

impl FoldNeed {
    /// Whether a fold has to happen before this call can be sent at all.
    ///
    /// Not called yet: both call sites use `should_fold`, which `Wait` also
    /// satisfies, and the failed-fold path already degrades output budget rather
    /// than sending. Kept because the distinction is real — `Fold` can finish the
    /// turn first, `Wait` cannot — and a caller that needs it will need it badly.
    #[allow(dead_code)]
    pub fn must_wait(self) -> bool {
        matches!(self, Self::Wait)
    }

    /// Whether a fold should be applied now, given the chance. `prepared`:
    /// the gists written in the background are ready (`FoldPrep::maybe_start`).
    pub fn should_fold(self, prepared: bool) -> bool {
        self >= Self::Due || (self == Self::Fold && prepared)
    }

    /// Whether gist preparation should be running: past `FOLD_PREPARE_TOKENS`,
    /// gists for the oldest turns are written in the background so the fold at
    /// `FOLD_AT_TOKENS` can be applied without stopping to write them.
    pub fn should_prepare(self) -> bool {
        self >= Self::Prepare
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ContextUsage {
    pub input_tokens: usize,
    /// Prompt plus the full output reservation (`max_tokens`).
    pub projected_tokens: usize,
    /// The server's window (`max_model_len`). There used to be a hard-coded
    /// 300,000 here, larger than the real 262,144 window, so the in-turn output
    /// clamp never engaged and vLLM rejected the request outright.
    pub capacity: usize,
}

impl ContextUsage {
    /// vLLM rejects any request whose prompt PLUS max_tokens exceeds the window
    /// (llama.cpp never enforced that), so the whole output reservation has to
    /// fit, not just the prompt.
    pub fn exceeds_hard_limit(&self) -> bool {
        self.projected_tokens.saturating_add(TURN_HEADROOM_TOKENS) >= self.capacity
    }

    /// What this context needs before the call is sent.
    ///
    /// Unlike the swap predicate it replaces, this never says "give up": the two
    /// upper states both end in a fold, and the difference between them is only
    /// whether s0ma can keep working while it happens.
    pub fn fold_need(&self) -> FoldNeed {
        let schedule = self.schedule();
        if self.exceeds_hard_limit() {
            FoldNeed::Wait
        } else if self.input_tokens >= schedule.ceiling {
            FoldNeed::Due
        } else if self.input_tokens >= schedule.fold_at {
            FoldNeed::Fold
        } else if self.input_tokens >= schedule.prepare {
            FoldNeed::Prepare
        } else {
            FoldNeed::None
        }
    }

    /// The fold schedule for the window this was measured against.
    pub fn schedule(&self) -> FoldSchedule {
        FoldSchedule::for_window(self.capacity)
    }

    /// The largest output budget that fits beside this prompt.
    pub fn output_room(&self) -> usize {
        self.capacity
            .saturating_sub(self.input_tokens.saturating_add(TURN_HEADROOM_TOKENS))
    }

    pub fn utilization_percent(&self) -> f32 {
        self.projected_tokens as f32 / self.capacity.max(1) as f32 * 100.0
    }
}

/// Measure messages exactly as they would be sent, against the live window.
pub fn measure(
    messages: &[Message],
    options: &ChatOptions,
    tools: &[ToolDefinition],
) -> Result<ContextUsage, Box<dyn std::error::Error>> {
    let capacity = llm::slot_context_capacity(0)?;
    let input_tokens = llm::count_input_tokens(messages, options, Some(tools))?;

    Ok(ContextUsage {
        input_tokens,
        projected_tokens: input_tokens + options.max_tokens as usize,
        capacity,
    })
}

pub fn preflight(
    messages: &[Message],
    pending_message: &Message,
    options: &ChatOptions,
    tools: &[ToolDefinition],
) -> Result<ContextUsage, Box<dyn std::error::Error>> {
    let mut prospective = messages.to_vec();

    prospective.push(pending_message.clone());

    measure(&prospective, options, tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: usize = 262_144;

    fn usage(input_tokens: usize, max_tokens: usize) -> ContextUsage {
        ContextUsage { input_tokens, projected_tokens: input_tokens + max_tokens, capacity: WINDOW }
    }

    #[test]
    fn the_reported_request_is_caught_before_it_is_sent() {
        // vLLM: "you requested 32768 output tokens and your prompt contains at
        // least 229377 input tokens, for a total of at least 262145 tokens."
        let reported = usage(229_377, 32_768);
        assert!(reported.exceeds_hard_limit());
        assert!(reported.fold_need().must_wait(), "the output reservation cannot fit");
        assert!(reported.output_room() < 32_768);
    }

    #[test]
    fn the_fold_threshold_applies_even_when_output_would_fit() {
        let small_output = usage(FOLD_AT_TOKENS, 1_024);
        assert!(!small_output.exceeds_hard_limit());
        // A short window's fold is due at its threshold, prepared or not.
        assert_eq!(small_output.fold_need(), FoldNeed::Due);
        assert!(small_output.fold_need().should_fold(false));
        assert!(!small_output.fold_need().must_wait(), "there is still room to finish the turn");
    }

    const LONG_WINDOW: usize = 1_048_576;

    fn long(input_tokens: usize) -> ContextUsage {
        ContextUsage { input_tokens, projected_tokens: input_tokens + 32_768, capacity: LONG_WINDOW }
    }

    #[test]
    fn a_long_window_folds_at_500k_when_the_gists_are_ready() {
        assert_eq!(long(220_000).fold_need(), FoldNeed::None, "220k is nothing on a 1M window");
        assert_eq!(long(LONG_FOLD_PREPARE_TOKENS).fold_need(), FoldNeed::Prepare);
        let at = long(LONG_FOLD_AT_TOKENS).fold_need();
        assert_eq!(at, FoldNeed::Fold);
        assert!(!at.should_fold(false), "not ready: the turn goes on");
        assert!(at.should_fold(true));
        let ceiling = long(LONG_FOLD_CEILING_TOKENS).fold_need();
        assert_eq!(ceiling, FoldNeed::Due);
        assert!(ceiling.should_fold(false), "at the ceiling it folds, ready or not");
        assert_eq!(long(LONG_WINDOW - 32_768).fold_need(), FoldNeed::Wait);
    }

    #[test]
    fn the_schedule_follows_the_window() {
        assert_eq!(FoldSchedule::for_window(262_144), FoldSchedule::SHORT);
        assert_eq!(FoldSchedule::for_window(LONG_WINDOW_MIN_TOKENS - 1), FoldSchedule::SHORT);
        assert_eq!(FoldSchedule::for_window(LONG_WINDOW_MIN_TOKENS), FoldSchedule::LONG);
        assert_eq!(FoldSchedule::for_window(LONG_WINDOW), FoldSchedule::LONG);
        assert_eq!(FoldSchedule::SHORT.main_reserve(262_144), 262_144);
        assert_eq!(FoldSchedule::LONG.main_reserve(LONG_WINDOW), LONG_WINDOW_MIN_TOKENS);
        assert!(FoldSchedule::LONG.gist_budget > GIST_BUDGET_TOKENS && FoldSchedule::LONG.shape_budget > SHAPE_BUDGET_TOKENS);
    }

    #[test]
    fn a_long_schedule_is_coherent() {
        let long = FoldSchedule::LONG;
        assert!(long.fold_to < long.prepare && long.prepare < long.fold_at && long.fold_at < long.ceiling);
        assert!(long.fold_to <= long.land_within && long.land_within < long.prepare);
        // Prefix and the bigger memory message still leave the live turns room.
        assert!(PREFIX_BUDGET_TOKENS + long.memory_message_budget() + MIN_LIVE_TURNS_TOKENS <= long.fold_to);
        // ds41's 1M window takes the long schedule, with room past the ceiling.
        assert!(LONG_WINDOW_MIN_TOKENS <= LONG_WINDOW);
        let past = ContextUsage {
            input_tokens: LONG_FOLD_CEILING_TOKENS + LONG_STEP_GROWTH_TOKENS,
            projected_tokens: LONG_FOLD_CEILING_TOKENS + LONG_STEP_GROWTH_TOKENS + OUTPUT_RESERVATION_TOKENS,
            capacity: LONG_WINDOW,
        };
        assert_eq!(past.fold_need(), FoldNeed::Due, "a step past the ceiling: due, not yet waiting");
        assert_eq!(long.lands_within(true), long.land_within);
        assert_eq!(long.lands_within(false), long.fold_to);
        let short = FoldSchedule::SHORT;
        assert_eq!(short.lands_within(true), FOLD_TO_TOKENS, "a short window tops a prepared fold up, as before");
        assert_eq!((short.fold_at, short.ceiling, short.land_within), (short.ceiling, FOLD_AT_TOKENS, FOLD_TO_TOKENS));
        assert_eq!(short.memory_message_budget(), MEMORY_MESSAGE_BUDGET_TOKENS);
    }

    #[test]
    fn preparation_starts_before_the_fold_does() {
        let preparing = usage(FOLD_PREPARE_TOKENS, 32_768);
        assert_eq!(preparing.fold_need(), FoldNeed::Prepare);
        assert!(preparing.fold_need().should_prepare());
        assert!(!preparing.fold_need().should_fold(true));
    }

    #[test]
    fn ordinary_turns_are_left_alone() {
        let normal = usage(180_000, 32_768);
        assert!(!normal.exceeds_hard_limit());
        assert_eq!(normal.fold_need(), FoldNeed::None);
        assert_eq!(normal.output_room(), WINDOW - 180_000 - TURN_HEADROOM_TOKENS);
    }

    #[test]
    fn full_output_reservation_fits_exactly_up_to_the_headroom() {
        let edge = WINDOW - TURN_HEADROOM_TOKENS - 32_768;
        assert!(!usage(edge - 1, 32_768).exceeds_hard_limit());
        assert!(usage(edge, 32_768).exceeds_hard_limit());
    }

    #[test]
    fn room_never_underflows() {
        assert_eq!(usage(WINDOW, 0).output_room(), 0);
        assert_eq!(usage(WINDOW + 10_000, 0).output_room(), 0);
    }

    #[test]
    fn urgency_is_ordered() {
        assert!(FoldNeed::Wait > FoldNeed::Due);
        assert!(FoldNeed::Due > FoldNeed::Fold);
        assert!(FoldNeed::Fold > FoldNeed::Prepare);
        assert!(FoldNeed::Prepare > FoldNeed::None);
        assert!(FoldNeed::Wait.should_fold(false) && FoldNeed::Wait.should_prepare());
        assert!(!FoldNeed::Prepare.should_fold(true));
    }

    #[test]
    fn fold_budget_is_coherent() {
        // A fold measures the WHOLE context against FOLD_TO_TOKENS, so the prefix
        // and the memory message are a floor it can never get under. If that floor
        // ever grows past the target, the fold loop would drop live turns forever
        // and still miss — the failure mode is an infinite tightening loop, not a
        // large context, so it's worth an assertion rather than a runtime warning.
        let floor = PREFIX_BUDGET_TOKENS + MEMORY_MESSAGE_BUDGET_TOKENS;
        assert!(
            floor + MIN_LIVE_TURNS_TOKENS <= FOLD_TO_TOKENS,
            "prefix ({PREFIX_BUDGET_TOKENS}) + memory message ({MEMORY_MESSAGE_BUDGET_TOKENS}) \
             leaves under {MIN_LIVE_TURNS_TOKENS} tokens of live turns at the {FOLD_TO_TOKENS} target",
        );
    }

    #[test]
    fn a_fold_covers_more_than_one_episode() {
        // FOLD_AT - FOLD_TO is the span one fold removes; at ~40k per gist that is
        // several episodes, which is why folding writes a gist per episode rather
        // than one gist for the whole chunk.
        let span = FOLD_AT_TOKENS - FOLD_TO_TOKENS;
        assert!(span / GIST_EPISODE_TOKENS >= 2, "a fold should write more than one gist");
    }

    #[test]
    fn the_thresholds_are_ordered_and_leave_room_to_land() {
        assert!(FOLD_PREPARE_TOKENS < FOLD_AT_TOKENS);
        assert!(FOLD_AT_TOKENS < HARD_LIMIT_TOKENS, "a fold must be applied before calls start waiting");
        assert!(FOLD_TO_TOKENS < FOLD_PREPARE_TOKENS);
    }
}
