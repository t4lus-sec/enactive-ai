//! Knowledge pass: on every operator message (the one that starts a turn, and
//! any typed while a turn runs) and every autonomous turn, ask the base model
//! Jev-style nouls, one per information source asked, in one parallel pass
//! with a 1000 ms budget (audited, not enforced), and tell the agent which sources
//! to check before answering.
//!
//! # Which are asked
//!
//! By default only the sources of truth outside the agent's memory: `internet`
//! and `vm` (`S0MA_KNOW_SOURCES`, a comma list of the names below, or `all`).
//! On ds41d each noul is a prefill of its own, and four run at once, so six
//! took two rounds; two take one. The agent's memory is searched on every turn
//! instead of when a noul says so: the archive lookahead and the facts
//! (`memory/recall_push.rs`) start with the pass, and offer only good matches (an
//! archive hit needs both words and meaning, a fact's words, or a cosine of
//! `S0MA_KNOW_LOOKAHEAD_MIN_COSINE`, 0.50), saying nothing when none are.
//! Its threads and self-knowledge are in its context already. A noul asked
//! again (`S0MA_KNOW_SOURCES=all`) gates its search as before.
//!
//! # Sources
//!
//! | source        | noul (true / false)                                     | looks up on |
//! |---------------|---------------------------------------------------------|-------------|
//! | `self`        | can you answer from your own knowledge?                 | false       |
//! | `internet`    | does a good reply need the internet?                    | true        |
//! | `archive`     | does it need your record of past conversations?         | true        |
//! | `thread`      | does it need your threads (projects, commitments)?      | true        |
//! | `vm`          | does it need the VM's state (files, output, sessions)?  | true        |
//! | `world_model` | is what it needs likely recorded in your world model?   | true        |
//!
//! `p_lookup` is the probability of the looking-up answer: P(no) for `self`,
//! P(yes) for the rest. A source is flagged when its `p_lookup` is above the
//! threshold (0.50), so the self noul is exactly "look up when P(no) > 0.50".
//!
//! `internet` is then scaled up: `p_lookup = min(1, 1.5 × P(yes) + 0.1)`, so
//! at the default threshold it flags from P(yes) > 0.27. The base model
//! under-calls web lookups, and an unneeded search costs less than a stale
//! answer. The archive record keeps the raw noul next to the scaled value.
//!
//! # Skill match
//!
//! Alongside the six, the pass names specific skills; there is no generic
//! "does it need a skill?" noul, these replace it. skill_match.rs embeds
//! the message and shortlists the closest skills by name and description (3 by
//! default, skipping any already read or pointed to in context).
//!
//! **By cosine (the default).** A shortlisted skill whose cosine is at least
//! `S0MA_SKILL_MATCH_MIN_COSINE` (0.50) goes into the note as a pointer to read
//! with skill_read, and the note says it was matched by description, not
//! checked. No request reaches the model for it: on an engine that prefills
//! every prompt in full (ds41d), each skill noul was one more prefill the turn
//! waited for, three a turn by default.
//!
//! **Judged (`S0MA_SKILL_MATCH_JUDGE=1`).** As before: one more noul per
//! candidate on the same state prefix, as soon as the shortlist is ready (does
//! this skill's procedure apply to the message?), and a skill whose P(true) is
//! above `S0MA_KNOW_THRESHOLD_SKILL_MATCH` is named. `s0ma --skill-cutoff`
//! reads the judged passes in `data/knowledge_audit.jsonl` and says which
//! cosine cutoff agrees best with the judge, for setting the default mode's.
//!
//! Skill text is never injected. The pass waits for the shortlist like any
//! other noul; an error there (embedding server down) only costs the turn its
//! skill pointers. With the match off (`S0MA_SKILL_MATCH=0`) the pass says
//! nothing about skills.
//!
//! # The noul structure
//!
//! TypeSafe AI's Jev answers several named questions about one state in one
//! parallel pass; a noul is a yes/no question answered with one number, the
//! probability of true. As jevper implements it for OpenAI-compatible servers:
//!
//! - Each question carries instructions and true/false criteria, rendered as an
//!   options block with single-letter labels: A is true, B is false.
//! - The answer is forced: vLLM's `structured_outputs.choice` constrains
//!   decoding to exactly "A" or "B".
//! - The readout is one position: `max_tokens: 1`, `logprobs: true`,
//!   `top_logprobs: 20`, then the label logprobs softmaxed over the labels only.
//!   A label the server didn't report counts as 0. Nothing is parsed from text.
//!
//! The state goes first and the question last, so every request in a pass
//! shares everything up to the question (system prompt and state) and vLLM's
//! prefix cache can serve it to all of them. Thinking is off so the first position is the
//! answer; temperature stays 1 so reported logprobs are not sharpened.
//!
//! # Archive lookahead
//!
//! Naming the archive as worth checking still left the agent to spend a decode step
//! and a tool round trip on a search the harness could have run while the pass
//! was waiting on the model. So it runs it: when the archive noul flags, the
//! pass searches the archive on the operator's own message and ends its note
//! with the best few hits as one-line references — event id, time, kind and a
//! bounded preview.
//!
//! What it attaches are pointers, not content. The agent still reads an event with
//! `archive_get`, and the references say so; they also say they rank by
//! resemblance rather than truth, and sit under a header marking them as
//! historical data rather than instructions, because a preview is archived
//! text and archived text can read like a command.
//!
//! The search starts when the archive noul answers, so most of it runs beside
//! the other nouls, and it is on a thread of its own: past
//! `S0MA_KNOW_LOOKAHEAD_MS` after the nouls are done it is abandoned, the note
//! carries the plain hint instead, and the overrun is audited. A noul that
//! does not flag, or that failed, searches nothing and costs nothing. Finding
//! nothing is reported too, with the search's own words that a miss is not
//! evidence, so an empty lookahead doesn't read as proof.
//!
//! # Budget and failure
//!
//! The six source requests run concurrently on one pooled client at the main lane's
//! priority. The pass fails open: it never costs a turn. Every noul runs to
//! completion (reqwest's 30 s default is the only timeout), so a slow pass is
//! slow but still answers. A noul that errors is left out and the others still
//! count; if all six fail, the turn goes ahead with no note. A pass that runs
//! over its 1000 ms budget, or has a failed noul, prints a warning with every
//! noul's time and appends a line to `data/knowledge_audit.jsonl`, so an
//! overrun can be audited later.
//!
//! # Environment
//!
//! - `S0MA_KNOW=0` — disable the pass.
//! - `S0MA_KNOW_SOURCES` — the nouls asked (`internet,vm`; `all`, or any of
//!   self, internet, archive, thread, vm, world_model).
//! - `S0MA_KNOW_LOOKAHEAD_MIN_COSINE` — what an archive hit searched unasked
//!   needs to be offered (0.50).
//! - `S0MA_KNOW_THRESHOLD` — `p_lookup` above which a source is flagged (0.50).
//! - `S0MA_KNOW_THRESHOLD_<SOURCE>` — per-source override, e.g.
//!   `S0MA_KNOW_THRESHOLD_ARCHIVE=0.5` or `S0MA_KNOW_THRESHOLD_WORLD_MODEL`, for
//!   calibrating one noul. It applies to the scaled `p_lookup`.
//! - `S0MA_SKILL_MATCH_MIN_COSINE` — the cosine a shortlisted skill needs to be
//!   named (0.50), unless judged.
//! - `S0MA_SKILL_MATCH_JUDGE=1` — ask the model of each shortlisted skill
//!   instead (one noul each).
//! - `S0MA_KNOW_THRESHOLD_SKILL_MATCH` — judged, the P(true) above which a
//!   shortlisted skill is named (default: the shared threshold). Shortlist
//!   settings are in skill_match.rs. (`S0MA_KNOW_THRESHOLD_SKILLS`, the removed generic noul's
//!   threshold, is refused.)
//! - `S0MA_KNOW_BUDGET_MS` — the pass time above which it is audited (1000).
//! - `S0MA_KNOW_LOOKAHEAD_MS` — how long the pass waits for the archive
//!   lookahead once its nouls are done (1500); `0` turns the lookahead off.
//! - `S0MA_KNOW_LOOKAHEAD_HITS` — references it offers, 1 to 5 (default 3).
//!
//! The readout `judge` exposes is used in two more places, both of which turn
//! off with `S0MA_KNOW=0`: episode reminders rank their candidates with it
//! (`episode_recall`), and `archive_search` reorders its hits with it
//! (`S0MA_ARCHIVE_RERANK=0`, `S0MA_ARCHIVE_RERANK_POOL`).
//! - `S0MA_KNOW_MODEL` — served model to ask (default `S0MA_LLM_MODEL`). Set it to
//!   the base model's served name if that alias is ever not the base.
use crate::{
    hippocampus::Hippocampus,
    llm,
    skill_match::{self, Candidate},
    types::Message,
};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::{
    sync::{mpsc, OnceLock},
    thread,
    time::{Duration, Instant},
};

const DEFAULT_THRESHOLD: f64 = 0.50;
const DEFAULT_BUDGET: Duration = Duration::from_millis(1_000);
/// How long the pass waits for the archive lookahead once its nouls are done.
/// The search starts when the archive noul answers, so most of it runs beside
/// the other nouls and this is only the tail.
const DEFAULT_LOOKAHEAD_BUDGET: Duration = Duration::from_millis(1_500);
/// References the lookahead offers.
const DEFAULT_LOOKAHEAD_HITS: usize = 3;
/// The questions asked by default: the sources of truth outside the agent's
/// memory. Its memory (archive, facts) is searched on every turn instead,
/// and its threads and self-knowledge are in its context already.
const DEFAULT_SOURCES: [Source; 2] = [Source::Internet, Source::Vm];
/// What an archive hit searched unasked needs: found by both words and
/// meaning, through a fact's words, or by meaning at least this close.
const DEFAULT_LOOKAHEAD_MIN_COSINE: f32 = 0.50;
/// The cosine a shortlisted skill needs to be named when it is not judged.
/// Nemotron-3-Embed-1B scores a clear query/passage match around 0.65 and an
/// unrelated pair near 0 (the published smoke test), and a skill is only a
/// pointer, so the bar sits a little below a clear match. `s0ma --skill-cutoff`
/// measures it against the judge's past answers.
const DEFAULT_SKILL_MIN_COSINE: f32 = 0.50;
/// Noul labels in criteria order: A is true, B is false.
const TRUE_LABEL: &str = "A";
const FALSE_LABEL: &str = "B";
/// vLLM's default --max-logprobs.
const TOP_LOGPROBS: u32 = 20;
/// One noul per source.
const SOURCES: usize = 6;

const MAX_PROBE_INPUT_CHARS: usize = 4_000;
const MAX_PROBE_CONTEXT_CHARS: usize = 1_500;
/// A skill's description in its pointer, at most.
const MAX_POINTER_DESCRIPTION_CHARS: usize = 300;

/// The noul asked of each shortlisted skill; the instructions name the skill
/// and quote its description.
const SKILL_WHEN_TRUE: &str = "The message asks for something this procedure covers, so reading and following it would change or improve the reply.";
const SKILL_WHEN_FALSE: &str = "The procedure is unrelated, or shares only the topic without covering what the message asks for.";
fn skill_instructions(candidate: &Candidate) -> String {
    format!(
        "Does a good reply to the message need your skill \"{}\", a written procedure described as: {}",
        candidate.name, candidate.description
    )
}

/// Replaces the archive noul's hint when the lookahead has already searched.
const ARCHIVE_SEARCHED: &str = "the harness has already run that search for you; its hits are at the end of this note, and archive_get reads one in full";
/// Replaces the world-model noul's hint when the facts have been searched.
const FACTS_SEARCHED: &str = "the harness has already searched your facts for this message; what matched well is at the end of this note, and fact_search finds more";

/// Shared by every request, the skill nouls' included: who is asking and what
/// each source is.
const PROBE_SYSTEM: &str = "You are s0ma, a persistent agent running locally. Besides your own knowledge you can consult: \
the internet, through a browser; your archive, the full record of your past conversations and actions; \
your threads, the projects, questions and commitments you keep track of; your skills, procedures and references your operator wrote; \
your machine, the Parrot OS laptop where you run commands and keep working files; \
and your world model, named models where you record explanations, observations and evidence-backed claims about how things are. \
You answer one typed question about the state you are given. Reply with exactly one option label and nothing else.";

// ------------------------------------------------------------------
// The six source nouls
// ------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The model's own knowledge ("self" in output and configuration).
    SelfKnowledge,
    Internet,
    Archive,
    Thread,
    Vm,
    WorldModel,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Self::SelfKnowledge => "self",
            Self::Internet => "internet",
            Self::Archive => "archive",
            Self::Thread => "thread",
            Self::Vm => "vm",
            Self::WorldModel => "world_model",
        }
    }
}

struct Noul {
    source: Source,
    instructions: &'static str,
    when_true: &'static str,
    when_false: &'static str,
    /// The answer that means "look it up": false for self, true otherwise.
    lookup_on: bool,
    /// `p_lookup = min(1, scale × P(lookup answer) + offset)`; (1, 0) for
    /// most sources.
    scale: f64,
    offset: f64,
    /// What the agent is told to do when this source is flagged.
    hint: &'static str,
}

impl Noul {
    /// Probability of the looking-up answer, after this source's scaling,
    /// kept within [0, 1].
    fn p_lookup(&self, p_true: f64) -> f64 {
        let p = if self.lookup_on { p_true } else { 1.0 - p_true };
        (self.scale * p + self.offset).clamp(0.0, 1.0)
    }
}

const NOULS: [Noul; SOURCES] = [
    Noul {
        source: Source::SelfKnowledge,
        instructions: "Can you answer the message accurately from your own knowledge, without looking anything up?",
        when_true: "You know the answer reliably, or the message does not ask for facts: conversation, opinions, tasks, commands, or code to write.",
        when_false: "A good answer depends on facts you do not reliably know: recent or time-sensitive events, current versions, prices or status, exact figures or quotes, or niche specifics.",
        lookup_on: false,
        scale: 1.0,
        offset: 0.0,
        hint: "your own knowledge is probably not enough here: don't answer from recall alone; check the source that fits, or say what you don't know",
    },
    Noul {
        source: Source::Internet,
        instructions: "Does a good reply to the message need information from the internet?",
        when_true: "It depends on public facts best checked on the web: current events, recent releases or versions, documentation, prices, status, or specifics worth verifying against a source.",
        when_false: "Nothing it needs is on the public web: conversation, opinions, your own past or work, or facts you know reliably.",
        lookup_on: true,
        // Biased toward looking things up: min(1, 1.5p + 0.1).
        scale: 1.5,
        offset: 0.1,
        hint: "browser_navigate to a search engine, browser_read the most relevant results, and name your sources",
    },
    Noul {
        source: Source::Archive,
        instructions: "Does a good reply to the message need something from your archive, the record of your past conversations and actions?",
        when_true: "It refers to or depends on your own past: something said, decided, done or found in an earlier conversation or session that is not shown in the state.",
        when_false: "It does not depend on your past beyond what the state shows.",
        lookup_on: true,
        scale: 1.0,
        offset: 0.0,
        hint: "archive_search for the earlier conversation or event, then archive_get to read it in full",
    },
    Noul {
        source: Source::Thread,
        instructions: "Does a good reply to the message need your threads, the projects, questions and commitments you keep track of?",
        when_true: "It concerns ongoing work or a standing question or commitment: its status, open questions, next actions, or history.",
        when_false: "It is self-contained and touches no ongoing project, question or commitment.",
        lookup_on: true,
        scale: 1.0,
        offset: 0.0,
        hint: "thread_list to find the relevant thread, then thread_get for its state, questions and actions",
    },
    Noul {
        source: Source::Vm,
        instructions: "Does a good reply to the message need information from your machine, the Parrot OS laptop where you run commands and keep working files?",
        when_true: "It depends on the machine's current state: files, command output, scan results, running sessions, or what is installed there.",
        when_false: "Nothing on your machine is relevant to it.",
        lookup_on: true,
        scale: 1.0,
        offset: 0.0,
        hint: "shell_read for a running shell's output, or vm_exec to inspect files and state",
    },
    Noul {
        source: Source::WorldModel,
        instructions: "Is the information a good reply to the message needs likely to be in your world model, the named models where you record explanations, observations and evidence-backed claims about how things are?",
        when_true: "It concerns something you may already have modelled: how a system, person, place or situation works or currently stands, an explanation or claim you recorded, or evidence for or against one.",
        when_false: "Nothing it needs is a recorded explanation or claim: conversation, general knowledge, or your past, ongoing work or procedures rather than your understanding of how things are.",
        lookup_on: true,
        scale: 1.0,
        offset: 0.0,
        hint: "world_model_list to find the relevant model, then world_model_get to read it (world_record_get for one record and its revision history)",
    },
];

// ------------------------------------------------------------------
// Entry point
// ------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Readout {
    /// The noul answer: probability of true (A), softmaxed over the two labels.
    pub noul: f64,
    /// How much of the reported distribution fell on the labels before the
    /// constraint; low means the model would rather have said something else.
    pub label_mass: f64,
}

/// One answered noul.
pub struct Answer {
    source: Source,
    readout: Readout,
    /// Probability of the looking-up answer, after the source's scaling.
    p_lookup: f64,
    threshold: f64,
}

impl Answer {
    fn flagged(&self) -> bool {
        self.p_lookup > self.threshold
    }
}

/// One shortlisted skill, with its noul when judged.
struct SkillResult {
    candidate: Candidate,
    /// When it finished, from the start of the pass.
    at: Duration,
    /// The judge's readout; `Ok(None)` when the skill is matched by cosine.
    outcome: Result<Option<Readout>, String>,
}

/// The skill match: the embedding shortlist, then by cosine or one noul per
/// candidate.
#[derive(Default)]
struct SkillStage {
    /// How long the shortlist took; `None` if it never ran.
    shortlist_took: Option<Duration>,
    shortlist_error: Option<String>,
    /// In shortlist order (closest first).
    results: Vec<SkillResult>,
    threshold: f64,
    /// Matched by cosine, against this bar, rather than judged.
    min_cosine: Option<f32>,
}

impl SkillStage {
    /// Named skills, likeliest first: by the judge's P(true), or by cosine.
    fn flagged(&self) -> Vec<(&Candidate, f64)> {
        let mut flagged: Vec<(&Candidate, f64)> = match self.min_cosine {
            Some(bar) => self
                .results
                .iter()
                .filter(|r| r.candidate.cosine >= bar)
                .map(|r| (&r.candidate, f64::from(r.candidate.cosine)))
                .collect(),
            None => self
                .results
                .iter()
                .filter_map(|r| match &r.outcome {
                    Ok(Some(readout)) => Some((&r.candidate, readout.noul)),
                    _ => None,
                })
                .filter(|(_, p)| *p > self.threshold)
                .collect(),
        };
        flagged.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.name.cmp(&b.0.name)));
        flagged
    }

    fn has_failures(&self) -> bool {
        self.shortlist_error.is_some() || self.results.iter().any(|r| r.outcome.is_err())
    }

    /// For the status line; `None` when there was nothing to match.
    fn summary(&self) -> Option<String> {
        if self.shortlist_error.is_some() {
            return Some("error".into());
        }
        if self.results.is_empty() {
            return None;
        }
        let parts: Vec<String> = self
            .results
            .iter()
            .map(|r| match &r.outcome {
                Ok(Some(readout)) => format!(
                    "{} {:.2}{} (cos {:.2})",
                    r.candidate.name,
                    readout.noul,
                    if readout.noul > self.threshold { "*" } else { "" },
                    r.candidate.cosine
                ),
                Ok(None) => format!(
                    "{} cos {:.2}{}",
                    r.candidate.name,
                    r.candidate.cosine,
                    if self.min_cosine.is_some_and(|bar| r.candidate.cosine >= bar) { "*" } else { "" }
                ),
                Err(_) => format!("{} error", r.candidate.name),
            })
            .collect();
        Some(parts.join(" · "))
    }

    /// Timings and errors for the audit line.
    fn detail(&self) -> Vec<String> {
        let mut parts = Vec::new();
        if let Some(took) = self.shortlist_took {
            match &self.shortlist_error {
                Some(error) => parts.push(format!("skill shortlist {}ms: {error}", took.as_millis())),
                None => parts.push(format!("skill shortlist {}ms", took.as_millis())),
            }
        }
        for r in self.results.iter().filter(|r| !matches!(r.outcome, Ok(None))) {
            match &r.outcome {
                Ok(_) => parts.push(format!("skill {} {}ms", r.candidate.name, r.at.as_millis())),
                Err(error) => parts.push(format!("skill {} {}ms: {error}", r.candidate.name, r.at.as_millis())),
            }
        }
        parts
    }
}

/// The archive search for one pass, ready to run on a thread of its own.
/// Boxed and `Send` so the pass can start it and abandon it if it overruns.
pub(crate) type Lookahead = Box<dyn FnOnce() -> Result<crate::archive::Lookahead, String> + Send>;

/// What a lookahead did, for the note, the status line and the audit.
struct LookaheadOutcome<T = crate::archive::Lookahead> {
    /// When it finished, from the start of the pass.
    at: Duration,
    result: Result<T, String>,
}

/// A completed pass: the answered nouls in source order, and any that failed.
pub struct Pass {
    answers: Vec<Answer>,
    failures: Vec<(Source, String)>,
    /// When each noul finished, in source order, for the audit.
    timings: Vec<(Source, Duration)>,
    took: Duration,
    budget: Duration,
    skills: SkillStage,
    /// The archive lookahead, when the archive noul flagged and there was one.
    lookahead: Option<LookaheadOutcome>,
    /// The facts pushed, when the fact lookahead ran (`memory/recall_push.rs`).
    facts: Option<LookaheadOutcome<crate::memory::recall_push::FactsFound>>,
    /// The query rewrite both lookaheads searched with, when it ran.
    rewrite: Option<std::sync::Arc<crate::memory::recall_push::PlanSlot>>,
}

/// Run the pass on the turn's input. `None` means there is nothing to add:
/// disabled or misconfigured. Never blocks a turn: problems are printed and
/// audited, and the turn goes ahead.
pub fn check(memory: &Hippocampus, conversation: &[Message], input: &str) -> Option<Pass> {
    // The pass reads a base model's log-probabilities under vLLM's structured
    // outputs: a hosted API has neither.
    if crate::model_config::hosted() {
        return None;
    }
    let config = match Config::from_vars(|key| std::env::var(key).ok()) {
        Ok(Some(config)) => config,
        Ok(None) => return None,
        Err(error) => {
            crate::terminal::warn_status(&format!("[know] misconfigured, pass skipped: {error}"));
            return None;
        }
    };
    let shortlist = || skill_match::shortlist(conversation, input);
    let url = llm::endpoint("/v1/chat/completions");
    // The query both lookaheads search with, rewritten beside the nouls.
    let push = config.push;
    let searching = !config.lookahead_budget.is_zero() || push.mode != crate::memory::recall_push::PushMode::Off;
    let plan = match push.rewrite {
        _ if !searching => crate::memory::recall_push::PlanSlot::none(),
        crate::memory::recall_push::Rewrite::Off => crate::memory::recall_push::PlanSlot::none(),
        // Made here, no model asked: the names the last reply mentioned.
        crate::memory::recall_push::Rewrite::Local => {
            let previous = conversation
                .iter()
                .rev()
                .find(|message| message.role == "assistant" && !message.content.trim().is_empty())
                .map(|message| truncate_chars(&message.content, MAX_PROBE_CONTEXT_CHARS))
                .unwrap_or_default();
            crate::memory::recall_push::local_slot(input, &crate::facts::names_in(memory, &previous))
        }
        crate::memory::recall_push::Rewrite::Model => {
            let body = crate::memory::recall_push::rewrite_body(&config.model, &state_block(conversation, input), config.priority);
            crate::memory::recall_push::start_rewrite(client(), url.clone(), body, push.rewrite_wait)
        }
    };
    let lookahead = archive_lookahead(memory, &config, input, std::sync::Arc::clone(&plan));
    let facts = (push.mode != crate::memory::recall_push::PushMode::Off)
        .then(|| crate::memory::recall_push::fact_lookahead(memory, input, conversation, std::sync::Arc::clone(&plan), push));
    let mut pass = run_with(&url, &config, conversation, input, &shortlist, lookahead, facts);
    pass.rewrite = Some(plan);
    crate::terminal::say_note(&pass.status_line());
    pass.audit(input);
    Some(pass)
}

/// One yes/no question for `judge`, in the noul's shape: what to decide, and
/// what makes it true or false.
pub(crate) struct Question {
    pub instructions: String,
    pub when_true: &'static str,
    pub when_false: &'static str,
}

/// The pass's forced-choice readout for other callers (episode recall's
/// rerank): one question per candidate, all asked at once on the same `state`
/// prefix, P(true) for each, in order. `Err` when the probes are off or
/// misconfigured; a question that fails comes back as its own `Err`.
///
/// Scheduled behind the workers: unlike the pass, nothing waits on it.
pub(crate) fn judge(state: &str, questions: &[Question]) -> Result<Vec<Result<f64, String>>, String> {
    let config = Config::from_vars(|key| std::env::var(key).ok())?.ok_or("the knowledge probes are off (S0MA_KNOW=0)")?;
    let url = llm::endpoint("/v1/chat/completions");
    // Behind the workers, with sleep: nothing waits on it.
    let priority = crate::conversations::priority_enabled().then_some(crate::conversations::RERANK_PRIORITY);
    Ok(judge_with(client(), &url, &config.model, priority, state, questions))
}

fn judge_with(
    client: &Client,
    url: &str,
    model: &str,
    priority: Option<i32>,
    state: &str,
    questions: &[Question],
) -> Vec<Result<f64, String>> {
    thread::scope(|scope| {
        let handles: Vec<_> = questions
            .iter()
            .map(|question| {
                let messages = question_messages(state, &question.instructions, question.when_true, question.when_false);
                let body = probe_body(model, &messages, priority);
                scope.spawn(move || ask(client, url, &body).map(|readout| readout.noul))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or_else(|_| Err("judge thread panicked".into())))
            .collect()
    })
}

/// No skill match, for the warm-up and for callers that don't want one.
fn no_skills() -> Result<Vec<Candidate>, String> {
    Ok(Vec::new())
}

/// The archive search this pass would run, or `None` with the lookahead off.
/// It owns everything it needs, because the pass may abandon it mid-search.
fn archive_lookahead(
    memory: &Hippocampus,
    config: &Config,
    input: &str,
    plan: std::sync::Arc<crate::memory::recall_push::PlanSlot>,
) -> Option<Lookahead> {
    // Found by cosine: with embeddings off there is nothing for it to find
    // that the fact lookahead's keywords don't.
    if config.lookahead_budget.is_zero() || !crate::model_config::embeddings_on() {
        return None;
    }
    let memory = memory.clone();
    // The semantic arm takes at most CHUNK_BYTES; the keyword arm is happy
    // with the same text.
    let query = crate::archive::truncate_for_embedding(input).to_owned();
    let hits = config.lookahead_hits;
    // Run on every turn when its question is off: only good matches, then.
    let bar = (!config.sources.contains(&Source::Archive)).then_some(config.lookahead_min_cosine);
    Some(Box::new(move || {
        // The rewritten query when it is ready in time, else the message.
        let query = plan.plan().map(|plan| crate::archive::truncate_for_embedding(&plan.query).to_owned()).unwrap_or(query);
        crate::archive::lookahead(&memory, &query, hits, bar).map_err(|error| error.to_string())
    }))
}

/// One pass at startup: compiles the constraint grammar and opens the pooled
/// connections, so the first real turn is not the cold one, and reports how
/// long the pass takes.
pub fn warm_up() {
    if crate::model_config::hosted() {
        crate::terminal::say_note("[know] off: the model server is a hosted API, with no base model to probe");
        return;
    }
    match Config::from_vars(|key| std::env::var(key).ok()) {
        Ok(None) => crate::terminal::say_note("[know] disabled (S0MA_KNOW=0)"),
        Err(error) => crate::terminal::warn_status(&format!("[know] misconfigured, the pass will be skipped: {error}")),
        Ok(Some(config)) => {
            let pass = run(&llm::endpoint("/v1/chat/completions"), &config, &[], "hello", &no_skills, None);
            crate::terminal::say_note(&format!("[know] warm-up: {}", pass.status_line()));
            pass.audit("(startup warm-up)");
        }
    }
}

impl Pass {
    fn flagged(&self) -> Vec<&Answer> {
        let mut flagged: Vec<&Answer> = self.answers.iter().filter(|a| a.flagged()).collect();
        flagged.sort_by(|a, b| b.p_lookup.total_cmp(&a.p_lookup));
        flagged
    }

    fn over_budget(&self) -> bool {
        self.took > self.budget
    }

    /// What the lookahead found, when it ran and finished in time.
    fn found(&self) -> Option<&crate::archive::Lookahead> {
        self.lookahead.as_ref().and_then(|outcome| outcome.result.as_ref().ok())
    }

    /// Printed every turn: the time against the budget, and each source's
    /// p_lookup, flagged ones marked and failed ones shown as errors.
    pub fn status_line(&self) -> String {
        let scores: Vec<String> = self
            .timings
            .iter()
            .map(|(source, _)| match self.answers.iter().find(|a| a.source == *source) {
                Some(a) => format!("{} {:.2}{}", a.source.name(), a.p_lookup, if a.flagged() { "*" } else { "" }),
                None => format!("{} error", source.name()),
            })
            .collect();
        let over = if self.over_budget() { " OVER BUDGET" } else { "" };
        let skills = self.skills.summary().map(|s| format!(" | skill match: {s}")).unwrap_or_default();
        let lookahead = self
            .lookahead
            .as_ref()
            .map(|outcome| match &outcome.result {
                Ok(found) => format!(" | archive lookahead: {} at {}ms", found.summary(), outcome.at.as_millis()),
                Err(error) => format!(" | archive lookahead: {error}"),
            })
            .unwrap_or_default();
        let facts = self
            .facts
            .as_ref()
            .map(|outcome| match &outcome.result {
                Ok(found) => format!(" | facts: {} at {}ms", found.summary(), outcome.at.as_millis()),
                Err(error) => format!(" | facts: {error}"),
            })
            .unwrap_or_default();
        let rewrite = self.rewrite.as_ref().and_then(|slot| slot.summary()).map(|s| format!(" | {s}")).unwrap_or_default();
        let mut line = format!(
            "[know {}ms/{}ms{over} p(lookup): {}{skills}{lookahead}{facts}{rewrite}]",
            self.took.as_millis(),
            self.budget.as_millis(),
            scores.join(" · ")
        );
        if self.needs_audit() {
            line.push_str(&format!("\n[know] audit: {}", self.detail()));
        }
        line
    }

    fn needs_audit(&self) -> bool {
        self.over_budget()
            || !self.failures.is_empty()
            || self.skills.has_failures()
            || self.lookahead.as_ref().is_some_and(|outcome| outcome.result.is_err())
            || self.facts.as_ref().is_some_and(|outcome| outcome.result.is_err())
    }

    /// Every noul's finishing time, with its error if it failed.
    fn detail(&self) -> String {
        let mut parts: Vec<String> = self
            .timings
            .iter()
            .map(|(source, at)| match self.failures.iter().find(|(s, _)| s == source) {
                Some((_, error)) => format!("{} {}ms: {error}", source.name(), at.as_millis()),
                None => format!("{} {}ms", source.name(), at.as_millis()),
            })
            .collect();
        parts.extend(self.skills.detail());
        parts.join("; ")
    }

    /// The audit record for an overrun or a failed noul; `None` for a clean pass.
    fn audit_record(&self, input: &str) -> Option<Value> {
        if !self.needs_audit() {
            return None;
        }
        let nouls: Vec<Value> = self
            .timings
            .iter()
            .map(|(source, at)| {
                let mut entry = json!({"source": source.name(), "ms": at.as_millis() as u64});
                if let Some(answer) = self.answers.iter().find(|a| a.source == *source) {
                    entry["p_lookup"] = answer.p_lookup.into();
                }
                if let Some((_, error)) = self.failures.iter().find(|(s, _)| s == source) {
                    entry["error"] = error.as_str().into();
                }
                entry
            })
            .collect();
        let mut record = json!({
            "at": chrono::Local::now().to_rfc3339(),
            "took_ms": self.took.as_millis() as u64,
            "budget_ms": self.budget.as_millis() as u64,
            "over_budget": self.over_budget(),
            "input_chars": input.chars().count(),
            "nouls": nouls,
        });
        if let Some(outcome) = &self.lookahead {
            let mut entry = json!({"ms": outcome.at.as_millis() as u64});
            match &outcome.result {
                Ok(found) => {
                    entry["hits"] = found.ids().into();
                    entry["summary"] = found.summary().into();
                }
                Err(error) => entry["error"] = error.as_str().into(),
            }
            record["archive_lookahead"] = entry;
        }
        if let Some(outcome) = &self.facts {
            let mut entry = json!({"ms": outcome.at.as_millis() as u64});
            match &outcome.result {
                Ok(found) => {
                    if let Some(query) = &found.rewritten {
                        entry["query"] = query.as_str().into();
                    }
                    entry["pushed"] = found.hits.iter().map(|h| h.id.clone()).collect::<Vec<_>>().into();
                    entry["weak"] = found.weak.into();
                    entry["already_shown"] = found.shown.into();
                }
                Err(error) => entry["error"] = error.as_str().into(),
            }
            record["fact_push"] = entry;
        }
        if let Some(summary) = self.rewrite.as_ref().and_then(|slot| slot.summary()) {
            record["rewrite"] = summary.into();
        }
        if let Some(took) = self.skills.shortlist_took {
            let skills: Vec<Value> = self
                .skills
                .results
                .iter()
                .map(|r| {
                    let mut entry = json!({"name": r.candidate.name, "cosine": r.candidate.cosine, "ms": r.at.as_millis() as u64});
                    match &r.outcome {
                        Ok(Some(readout)) => entry["noul"] = readout.noul.into(),
                        Ok(None) => {}
                        Err(error) => entry["error"] = error.as_str().into(),
                    }
                    entry
                })
                .collect();
            record["skill_match"] = json!({"shortlist_ms": took.as_millis() as u64, "skills": skills});
            if let Some(bar) = self.skills.min_cosine {
                record["skill_match"]["min_cosine"] = bar.into();
            }
            if let Some(error) = &self.skills.shortlist_error {
                record["skill_match"]["error"] = error.as_str().into();
            }
        }
        Some(record)
    }

    /// Append the audit record, if any, to data/knowledge_audit.jsonl. A write
    /// failure is printed and otherwise ignored.
    fn audit(&self, input: &str) {
        let Some(record) = self.audit_record(input) else {
            return;
        };
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("knowledge_audit.jsonl");
        let written = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut file| std::io::Write::write_all(&mut file, format!("{record}\n").as_bytes()));
        if let Err(error) = written {
            crate::terminal::warn_status(&format!("[know] audit not written to {}: {error}", path.display()));
        }
    }

    /// Appended to the turn's message when any source is flagged or any skill
    /// is confirmed, before the context preflight so its tokens are counted.
    pub fn context_block(&self) -> Option<String> {
        let sources = self.flagged();
        let skills = self.skills.flagged();
        let facts = self.facts_found().filter(|found| !found.hits.is_empty());
        if sources.is_empty() && skills.is_empty() {
            // Searches that ran unasked (their question is off, or facts are
            // pushed always) come with or without a flagged source, when
            // they found something good enough.
            let archive = self.found().map(|found| found.render()).unwrap_or_default();
            let facts = facts.map(|found| found.render()).unwrap_or_default();
            if archive.is_empty() && facts.is_empty() {
                return None;
            }
            return Some(format!(
                "\n\n[HARNESS KNOWLEDGE CHECK]\nFrom the harness, not the operator: what your memory holds that matches this input.{archive}{facts}"
            ));
        }
        let mut block = String::from(
            "\n\n[HARNESS KNOWLEDGE CHECK]\n\
             From the harness, not the operator: a probe of your base model asked where what this input needs is likely to be.",
        );
        let searched = self.found().is_some();
        let facts_searched = self.facts_found().is_some();
        if !sources.is_empty() {
            block.push_str(" Before answering, check the flagged sources, likeliest first:");
            for answer in sources {
                // The archive's hint stands only while the harness has not
                // already searched it for this message.
                let hint = match answer.source {
                    Source::Archive if searched => ARCHIVE_SEARCHED,
                    Source::WorldModel if facts_searched => FACTS_SEARCHED,
                    source => noul(source).hint,
                };
                block.push_str(&format!("\n- {} ({:.2}): {}", answer.source.name(), answer.p_lookup, hint));
            }
        }
        if !skills.is_empty() {
            block.push_str(if self.skills.min_cosine.is_some() {
                "\nSkills whose descriptions sit closest to this input, closest first (matched by description, not \
                 checked, so one may not apply); read one with skill_read before applying it:"
            } else {
                "\nSkills whose procedures likely apply, likeliest first; read each with skill_read before applying it:"
            });
            let score = if self.skills.min_cosine.is_some() { "cos " } else { "" };
            for (candidate, p) in skills {
                block.push_str(&format!(
                    "\n{}{}` ({score}{p:.2}): {}",
                    skill_match::POINTER_PREFIX,
                    candidate.name,
                    truncate_chars(&candidate.description, MAX_POINTER_DESCRIPTION_CHARS)
                ));
            }
        }
        block.push_str(
            "\nThe scores are estimates. If a source turns out to have nothing relevant, say so and move on; \
             if the sources don't settle it, say what you still don't know rather than filling the gap.",
        );
        // Last, under a header of its own: it is archived text rather than
        // part of the harness's instruction to look.
        if let Some(found) = self.found() {
            block.push_str(&found.render());
        }
        if let Some(found) = facts {
            block.push_str(&found.render());
        }
        Some(block)
    }

    /// What the fact lookahead found, when it ran and finished in time.
    fn facts_found(&self) -> Option<&crate::memory::recall_push::FactsFound> {
        self.facts.as_ref().and_then(|outcome| outcome.result.as_ref().ok())
    }

    /// Archived with the turn when the note was attached, so memory shows why
    /// the agent went looking.
    pub fn archive_record(&self) -> Option<String> {
        let sources = self.flagged();
        let skills = self.skills.flagged();
        if sources.is_empty() && skills.is_empty() {
            return None;
        }
        let mut record = format!("[harness:knowledge_check]\npass: {} ms of {} ms", self.took.as_millis(), self.budget.as_millis());
        for a in &self.answers {
            record.push_str(&format!(
                "\n{}: noul {:.3}, p_lookup {:.3} (threshold {:.2}, label mass {:.2}){}",
                a.source.name(),
                a.readout.noul,
                a.p_lookup,
                a.threshold,
                a.readout.label_mass,
                if a.flagged() { " flagged" } else { "" }
            ));
        }
        for r in &self.skills.results {
            match &r.outcome {
                Ok(None) => record.push_str(&format!(
                    "\nskill {}: cosine {:.3} (cutoff {:.2}){}",
                    r.candidate.name,
                    r.candidate.cosine,
                    self.skills.min_cosine.unwrap_or_default(),
                    if self.skills.min_cosine.is_some_and(|bar| r.candidate.cosine >= bar) { " flagged" } else { "" }
                )),
                Ok(Some(readout)) => record.push_str(&format!(
                    "\nskill {}: noul {:.3} (threshold {:.2}, label mass {:.2}, cosine {:.3}){}",
                    r.candidate.name,
                    readout.noul,
                    self.skills.threshold,
                    readout.label_mass,
                    r.candidate.cosine,
                    if readout.noul > self.skills.threshold { " flagged" } else { "" }
                )),
                Err(error) => record.push_str(&format!("\nskill {}: error ({error})", r.candidate.name)),
            }
        }
        if let Some(outcome) = &self.lookahead {
            record.push_str(&match &outcome.result {
                Ok(found) => format!("\narchive lookahead: {} at {} ms", found.summary(), outcome.at.as_millis()),
                Err(error) => format!("\narchive lookahead: {error}"),
            });
        }
        if let Some(outcome) = &self.facts {
            record.push_str(&match &outcome.result {
                Ok(found) => format!("\nfacts: {} at {} ms", found.summary(), outcome.at.as_millis()),
                Err(error) => format!("\nfacts: {error}"),
            });
        }
        let mut steps = Vec::new();
        if !sources.is_empty() {
            let names: Vec<&str> = sources.iter().map(|a| a.source.name()).collect();
            steps.push(format!("check {}", names.join(", ")));
        }
        if !skills.is_empty() {
            let names: Vec<&str> = skills.iter().map(|(c, _)| c.name.as_str()).collect();
            steps.push(format!("read skill {}", names.join(", ")));
        }
        record.push_str(&format!("\ninstruction: {} before answering", steps.join("; ")));
        Some(record)
    }
}

fn noul(source: Source) -> &'static Noul {
    NOULS.iter().find(|n| n.source == source).expect("every source has a noul")
}

// ------------------------------------------------------------------
// Configuration
// ------------------------------------------------------------------

struct Config {
    thresholds: [f64; SOURCES],
    /// P(true) above which a shortlisted skill is named, when judged.
    skill_threshold: f64,
    /// Ask the model of each shortlisted skill, rather than going by cosine.
    skill_judge: bool,
    /// The cosine a shortlisted skill needs when not judged.
    skill_min_cosine: f32,
    budget: Duration,
    /// How long the pass waits for the archive lookahead after its nouls;
    /// zero turns the lookahead off.
    lookahead_budget: Duration,
    /// References the lookahead offers.
    lookahead_hits: usize,
    model: String,
    priority: Option<i32>,
    /// The fact push and the query rewrite (`memory/recall_push.rs`).
    push: crate::memory::recall_push::PushConfig,
    /// The questions asked (`S0MA_KNOW_SOURCES`), in `NOULS` order.
    sources: Vec<Source>,
    /// What an archive hit needs to be offered when the archive's question
    /// is off and it is searched unasked (`S0MA_KNOW_LOOKAHEAD_MIN_COSINE`).
    lookahead_min_cosine: f32,
}

impl Config {
    /// `Ok(None)` when disabled.
    fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<Option<Self>, String> {
        let get = |key: &str| get(key).map(|value| value.trim().to_owned()).filter(|value| !value.is_empty());
        if matches!(get("S0MA_KNOW").map(|v| v.to_ascii_lowercase()).as_deref(), Some("0" | "off" | "false" | "no")) {
            return Ok(None);
        }
        if get("S0MA_KNOW_THRESHOLD_MODEL").is_some() {
            return Err("S0MA_KNOW_THRESHOLD_MODEL was renamed: the source is now `self`; use S0MA_KNOW_THRESHOLD_SELF".into());
        }
        if get("S0MA_KNOW_THRESHOLD_SKILLS").is_some() {
            return Err("S0MA_KNOW_THRESHOLD_SKILLS was removed with the generic skills noul: skills are matched one by one, and S0MA_KNOW_THRESHOLD_SKILL_MATCH sets their bar; unset it".into());
        }
        if get("S0MA_KNOW_CONSTRAINT").is_some() {
            return Err("S0MA_KNOW_CONSTRAINT was removed: the pass forces its answer with vLLM's structured_outputs only; unset it".into());
        }
        let threshold = |key: &str, default: f64| -> Result<f64, String> {
            match get(key) {
                Some(raw) => raw
                    .parse::<f64>()
                    .ok()
                    .filter(|t| (0.0..=1.0).contains(t))
                    .ok_or_else(|| format!("{key} must be a number from 0 to 1")),
                None => Ok(default),
            }
        };
        let shared = threshold("S0MA_KNOW_THRESHOLD", DEFAULT_THRESHOLD)?;
        let mut thresholds = [shared; SOURCES];
        for (slot, noul) in thresholds.iter_mut().zip(&NOULS) {
            *slot = threshold(&format!("S0MA_KNOW_THRESHOLD_{}", noul.source.name().to_ascii_uppercase()), shared)?;
        }
        let skill_threshold = threshold("S0MA_KNOW_THRESHOLD_SKILL_MATCH", shared)?;
        let skill_judge = match get("S0MA_SKILL_MATCH_JUDGE").map(|v| v.to_ascii_lowercase()).as_deref() {
            None | Some("0" | "off" | "false" | "no") => false,
            Some("1" | "on" | "true" | "yes") => true,
            Some(_) => return Err("S0MA_SKILL_MATCH_JUDGE must be 1 (ask the model of each skill) or 0 (by cosine)".into()),
        };
        let skill_min_cosine = match get("S0MA_SKILL_MATCH_MIN_COSINE") {
            Some(raw) => raw
                .parse::<f32>()
                .ok()
                .filter(|c| (-1.0..=1.0).contains(c))
                .ok_or("S0MA_SKILL_MATCH_MIN_COSINE must be a number from -1 to 1")?,
            None => DEFAULT_SKILL_MIN_COSINE,
        };
        let budget = match get("S0MA_KNOW_BUDGET_MS") {
            Some(raw) => raw
                .parse::<u64>()
                .ok()
                .filter(|ms| (1..=60_000).contains(ms))
                .map(Duration::from_millis)
                .ok_or("S0MA_KNOW_BUDGET_MS must be a whole number of milliseconds from 1 to 60000")?,
            None => DEFAULT_BUDGET,
        };
        let lookahead_budget = match get("S0MA_KNOW_LOOKAHEAD_MS") {
            Some(raw) => raw
                .parse::<u64>()
                .ok()
                .filter(|ms| *ms <= 60_000)
                .map(Duration::from_millis)
                .ok_or("S0MA_KNOW_LOOKAHEAD_MS must be a whole number of milliseconds from 0 to 60000 (0 turns the archive lookahead off)")?,
            None => DEFAULT_LOOKAHEAD_BUDGET,
        };
        let lookahead_hits = match get("S0MA_KNOW_LOOKAHEAD_HITS") {
            Some(raw) => raw
                .parse::<usize>()
                .ok()
                .filter(|hits| (1..=crate::archive::MAX_REFERENCES).contains(hits))
                .ok_or_else(|| format!("S0MA_KNOW_LOOKAHEAD_HITS must be a whole number from 1 to {}", crate::archive::MAX_REFERENCES))?,
            None => DEFAULT_LOOKAHEAD_HITS,
        };
        Ok(Some(Self {
            thresholds,
            skill_threshold,
            skill_judge,
            skill_min_cosine,
            budget,
            lookahead_budget,
            lookahead_hits,
            model: get("S0MA_KNOW_MODEL").unwrap_or_else(llm::model_name),
            // The pass gates the main turn, so it is scheduled like the main turn.
            priority: llm::lane_priority(0),
            push: crate::memory::recall_push::PushConfig::from_vars(get)?,
            sources: match get("S0MA_KNOW_SOURCES") {
                None => DEFAULT_SOURCES.to_vec(),
                Some(list) if list.eq_ignore_ascii_case("all") => NOULS.iter().map(|noul| noul.source).collect(),
                Some(list) => {
                    let mut chosen = Vec::new();
                    for name in list.split(',').map(str::trim).filter(|name| !name.is_empty()) {
                        let Some(noul) = NOULS.iter().find(|noul| noul.source.name() == name) else {
                            return Err(format!(
                                "S0MA_KNOW_SOURCES: unknown source {name:?}; the sources are {} (or all)",
                                NOULS.iter().map(|noul| noul.source.name()).collect::<Vec<_>>().join(", ")
                            ));
                        };
                        if !chosen.contains(&noul.source) {
                            chosen.push(noul.source);
                        }
                    }
                    // In NOULS order, whatever order they were named in.
                    NOULS.iter().map(|noul| noul.source).filter(|source| chosen.contains(source)).collect()
                }
            },
            lookahead_min_cosine: match get("S0MA_KNOW_LOOKAHEAD_MIN_COSINE") {
                Some(raw) => raw
                    .parse::<f32>()
                    .ok()
                    .filter(|c| (-1.0..=1.0).contains(c))
                    .ok_or("S0MA_KNOW_LOOKAHEAD_MIN_COSINE must be a number from -1 to 1")?,
                None => DEFAULT_LOOKAHEAD_MIN_COSINE,
            },
        }))
    }
}

// ------------------------------------------------------------------
// Probe
// ------------------------------------------------------------------

/// The state block, identical for every noul of a pass.
fn state_block(conversation: &[Message], input: &str) -> String {
    let mut state = String::from("State:\n");
    if let Some(previous) = conversation
        .iter()
        .rev()
        .find(|message| message.role == "assistant" && !message.content.trim().is_empty())
    {
        state.push_str("Your previous reply:\n");
        state.push_str(&truncate_chars(&previous.content, MAX_PROBE_CONTEXT_CHARS));
        state.push_str("\n\n");
    }
    state.push_str("Message:\n");
    state.push_str(&truncate_chars(input, MAX_PROBE_INPUT_CHARS));
    state
}

/// One noul in Jev's shape (instructions, then an options block of labelled
/// criteria), after the shared state.
fn probe_messages(state: &str, noul: &Noul) -> Vec<Message> {
    question_messages(state, noul.instructions, noul.when_true, noul.when_false)
}

/// The same shape for a shortlisted skill.
fn skill_messages(state: &str, candidate: &Candidate) -> Vec<Message> {
    question_messages(state, &skill_instructions(candidate), SKILL_WHEN_TRUE, SKILL_WHEN_FALSE)
}

fn question_messages(state: &str, instructions: &str, when_true: &str, when_false: &str) -> Vec<Message> {
    let prompt = format!(
        "{state}\n\nQuestion (noul: true or false)\n{instructions}\n\nOptions:\n{TRUE_LABEL}: true. {when_true}\n{FALSE_LABEL}: false. {when_false}"
    );
    vec![Message::system(PROBE_SYSTEM.to_owned()), Message::user(prompt)]
}

fn probe_body(model: &str, messages: &[Message], priority: Option<i32>) -> Value {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "logprobs": true,
        "top_logprobs": TOP_LOGPROBS,
        "chat_template_kwargs": {"enable_thinking": false},
        "structured_outputs": {"choice": [TRUE_LABEL, FALSE_LABEL]},
    });
    if let Some(priority) = priority {
        body["priority"] = priority.into();
    }
    // The knowledge side call's conversation, for ds41d: never saved (one
    // token), and never erasing the main turn's save.
    if crate::conversations::enabled() {
        body["conversation_id"] = crate::conversations::Conversation::side("knowledge").id().into();
    }
    body
}

/// One pooled client for every pass: no connection setup on the hot path.
fn client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        Client::builder()
            .tcp_nodelay(true)
            // Six source nouls plus up to ten skill nouls at once.
            .pool_max_idle_per_host(NOULS.len() + 10)
            .build()
            .expect("the probe's HTTP client builds with default TLS settings")
    })
}

/// The skill shortlist for one pass (skill_match::shortlist in production).
type Shortlist<'a> = &'a (dyn Fn() -> Result<Vec<Candidate>, String> + Sync);
/// When a source noul finished, and what it said.
type NoulResult = (Duration, Result<Readout, String>);

/// Ask the six source nouls concurrently and, alongside them, run the skill match:
/// the shortlist, then one noul per candidate. Wait for every one: no
/// deadline. The budget is only compared afterwards, for the audit.
fn run(
    url: &str,
    config: &Config,
    conversation: &[Message],
    input: &str,
    shortlist: Shortlist,
    lookahead: Option<Lookahead>,
) -> Pass {
    run_with(url, config, conversation, input, shortlist, lookahead, None)
}

/// `run`, with the fact lookahead too: it searches once the world-model or
/// the archive noul flags (or at once, pushing always), beside the rest.
fn run_with(
    url: &str,
    config: &Config,
    conversation: &[Message],
    input: &str,
    shortlist: Shortlist,
    lookahead: Option<Lookahead>,
    facts: Option<crate::memory::recall_push::FactLookahead>,
) -> Pass {
    let started = Instant::now();
    let state = state_block(conversation, input);
    let client = client();

    // The lookahead waits for the archive noul and then searches, so most of
    // it runs beside the other nouls. It is on a thread of its own rather than
    // in the scope below because a search that overruns is abandoned: the
    // embedding server is shared and can be slow, and the pass gates the turn.
    // A noul that does not flag (or fails) drops the sender, which ends the
    // pass's wait at once.
    let (flag_tx, flag_rx) = mpsc::channel::<bool>();
    let found = lookahead.map(|search| {
        let (found_tx, found_rx) = mpsc::channel();
        std::thread::spawn(move || {
            if flag_rx.recv().unwrap_or(false) {
                let _ = found_tx.send((started.elapsed(), search()));
            }
        });
        found_rx
    });
    let mut flag = found.is_some().then_some(flag_tx);
    // The questions asked (`S0MA_KNOW_SOURCES`): a search whose question is
    // not asked runs at once, beside the ones that are.
    let active: Vec<(&Noul, f64)> = NOULS
        .iter()
        .zip(config.thresholds)
        .filter(|(noul, _)| config.sources.contains(&noul.source))
        .collect();
    let asked = |source: Source| active.iter().any(|(noul, _)| noul.source == source);
    if !asked(Source::Archive)
        && let Some(flag) = flag.take()
    {
        let _ = flag.send(true);
    }
    // The facts wait for either noul that would send the agent looking for one.
    let always = config.push.mode == crate::memory::recall_push::PushMode::Always || !(asked(Source::Archive) || asked(Source::WorldModel));
    let (facts_tx, facts_rx) = mpsc::channel::<bool>();
    let facts_found = facts.map(|search| {
        let (found_tx, found_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let go = always
                || loop {
                    match facts_rx.recv() {
                        Ok(true) => break true,
                        Ok(false) => continue,
                        Err(_) => break false,
                    }
                };
            if go {
                let _ = found_tx.send((started.elapsed(), search()));
            }
        });
        found_rx
    });
    let facts_flag = facts_found.is_some().then_some(facts_tx);
    let senders: Vec<Vec<mpsc::Sender<bool>>> = active
        .iter()
        .map(|(noul, _)| {
            let mut senders = Vec::new();
            if noul.source == Source::Archive {
                senders.extend(flag.take());
            }
            if matches!(noul.source, Source::Archive | Source::WorldModel) {
                senders.extend(facts_flag.clone());
            }
            senders
        })
        .collect();
    // Only the nouls hold the facts' senders now: when both are done, the
    // wait ends.
    drop(facts_flag);

    let (results, skills): (Vec<NoulResult>, SkillStage) = thread::scope(|scope| {
        let handles: Vec<_> = active
            .iter()
            .copied()
            .zip(senders)
            .map(|((noul, threshold), flags)| {
                let body = probe_body(&config.model, &probe_messages(&state, noul), config.priority);
                scope.spawn(move || {
                    let result = ask(client, url, &body);
                    // Worth searching by the same test `Answer::flagged` applies.
                    let worth = result.as_ref().is_ok_and(|readout| noul.p_lookup(readout.noul) > threshold);
                    for flag in flags {
                        let _ = flag.send(worth);
                    }
                    (started.elapsed(), result)
                })
            })
            .collect();
        let state = &state;
        let skill_handle = scope.spawn(move || skill_stage(client, url, config, state, started, shortlist));
        let results = handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or_else(|_| (started.elapsed(), Err("probe thread panicked".into()))))
            .collect();
        let skills = skill_handle.join().unwrap_or_else(|_| SkillStage {
            shortlist_took: Some(started.elapsed()),
            shortlist_error: Some("skill match thread panicked".into()),
            threshold: config.skill_threshold,
            ..SkillStage::default()
        });
        (results, skills)
    });
    let took = started.elapsed();
    // Both lookaheads finish within one wait after the nouls.
    let wait = if config.lookahead_budget.is_zero() { DEFAULT_LOOKAHEAD_BUDGET } else { config.lookahead_budget };
    let deadline = Instant::now() + wait;
    let lookahead = found.and_then(|rx| collect_lookahead(rx, config.lookahead_budget, started));
    let facts = facts_found.and_then(|rx| collect_lookahead(rx, deadline.saturating_duration_since(Instant::now()), started));

    let mut answers = Vec::new();
    let mut failures = Vec::new();
    let mut timings = Vec::new();
    for ((noul, threshold), (at, result)) in active.iter().copied().zip(results) {
        timings.push((noul.source, at));
        match result {
            Ok(readout) => {
                let p_lookup = noul.p_lookup(readout.noul);
                answers.push(Answer { source: noul.source, readout, p_lookup, threshold });
            }
            Err(error) => failures.push((noul.source, error)),
        }
    }
    Pass { answers, failures, timings, took, budget: config.budget, skills, lookahead, facts, rewrite: None }
}

/// What the lookahead found, waiting at most `budget` after the nouls are
/// done. `None` when the archive noul did not flag, so nothing was searched
/// (its thread dropped the sender without searching).
fn collect_lookahead<T>(
    rx: mpsc::Receiver<(Duration, Result<T, String>)>,
    budget: Duration,
    started: Instant,
) -> Option<LookaheadOutcome<T>> {
    match rx.recv_timeout(budget) {
        Ok((at, result)) => Some(LookaheadOutcome { at, result }),
        // Still searching. It is left to finish on its own and its result
        // dropped; the turn goes ahead with the plain hint instead.
        Err(mpsc::RecvTimeoutError::Timeout) => Some(LookaheadOutcome {
            at: started.elapsed(),
            result: Err(format!(
                "unfinished {} ms after the nouls; abandoned, so search yourself",
                budget.as_millis()
            )),
        }),
        Err(mpsc::RecvTimeoutError::Disconnected) => None,
    }
}

/// The skill match, on its own thread inside the pass: shortlist, then the
/// candidates' nouls in parallel on the shared state prefix.
fn skill_stage(client: &Client, url: &str, config: &Config, state: &str, started: Instant, shortlist: Shortlist) -> SkillStage {
    let mut stage = SkillStage { threshold: config.skill_threshold, ..SkillStage::default() };
    let began = Instant::now();
    let candidates = shortlist();
    stage.shortlist_took = Some(began.elapsed());
    let candidates = match candidates {
        Ok(candidates) => candidates,
        Err(error) => {
            stage.shortlist_error = Some(error);
            return stage;
        }
    };
    if !config.skill_judge {
        // By cosine: nothing more to ask.
        stage.min_cosine = Some(config.skill_min_cosine);
        let at = started.elapsed();
        stage.results = candidates.into_iter().map(|candidate| SkillResult { candidate, at, outcome: Ok(None) }).collect();
        return stage;
    }
    stage.results = thread::scope(|scope| {
        let handles: Vec<_> = candidates
            .into_iter()
            .map(|candidate| {
                let body = probe_body(&config.model, &skill_messages(state, &candidate), config.priority);
                let fallback = candidate.clone();
                let handle = scope.spawn(move || {
                    let outcome = ask(client, url, &body).map(Some);
                    SkillResult { candidate, at: started.elapsed(), outcome }
                });
                (fallback, handle)
            })
            .collect();
        handles
            .into_iter()
            .map(|(fallback, handle)| {
                handle.join().unwrap_or_else(|_| SkillResult {
                    candidate: fallback,
                    at: started.elapsed(),
                    outcome: Err("probe thread panicked".into()),
                })
            })
            .collect()
    });
    stage
}

fn ask(client: &Client, url: &str, body: &Value) -> Result<Readout, String> {
    // A probe is a (short) generation: background embedding waits for it too.
    let _generating = crate::llm::Generating::start();
    let response = crate::model_config::authorize(client.post(url).json(body)).send().map_err(|error| format!("request failed: {error}"))?;
    let status = response.status();
    let text = response.text().map_err(|error| format!("response unreadable: {error}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {}", truncate_chars(text.trim(), 300)));
    }
    let value: Value = serde_json::from_str(&text).map_err(|e| format!("malformed response: {e}"))?;
    readout(&value)
}

/// jevper's label readout: the answer token must be a label; each label's
/// logprob comes from the answer token or its top_logprobs (0 if unreported);
/// softmax over the labels only.
fn readout(response: &Value) -> Result<Readout, String> {
    let position = response
        .pointer("/choices/0/logprobs/content/0")
        .ok_or("no logprobs came back; the server must support logprobs and top_logprobs on chat completions")?;
    let answer = position.get("token").and_then(Value::as_str).unwrap_or_default();
    let Some(answer_label) = label(answer) else {
        return Err(format!(
            "answer token {answer:?} is not {TRUE_LABEL} or {FALSE_LABEL}; the server did not apply structured_outputs.choice"
        ));
    };
    let answer_logprob = position
        .get("logprob")
        .and_then(Value::as_f64)
        .filter(|lp| lp.is_finite())
        .ok_or("the answer token has no usable logprob")?;
    let top = position.get("top_logprobs").and_then(Value::as_array).cloned().unwrap_or_default();
    if top.len() < 2 {
        return Err("no alternatives came back (top_logprobs); one logprob is not a distribution".into());
    }

    // Best logprob seen per label: the answer token itself, then alternatives.
    let mut best = [None::<f64>; 2];
    let slot = |is_true: bool| usize::from(!is_true);
    best[slot(answer_label)] = Some(answer_logprob);
    for entry in &top {
        let token = entry.get("token").and_then(Value::as_str).unwrap_or_default();
        let (Some(is_true), Some(logprob)) = (label(token), entry.get("logprob").and_then(Value::as_f64).filter(|lp| lp.is_finite())) else {
            continue;
        };
        let current = &mut best[slot(is_true)];
        *current = Some(current.map_or(logprob, |seen: f64| seen.max(logprob)));
    }
    let p_true = best[0].map_or(0.0, f64::exp);
    let p_false = best[1].map_or(0.0, f64::exp);
    let label_mass = p_true + p_false;
    if label_mass <= 0.0 {
        return Err("both labels underflowed to zero probability".into());
    }
    Ok(Readout { noul: p_true / label_mass, label_mass })
}

/// A label token, compared case-insensitively over ASCII after trimming
/// whitespace: " A" and "a" are A; "A:" and "AB" are not labels.
fn label(token: &str) -> Option<bool> {
    let token = token.trim();
    if token.eq_ignore_ascii_case(TRUE_LABEL) {
        Some(true)
    } else if token.eq_ignore_ascii_case(FALSE_LABEL) {
        Some(false)
    } else {
        None
    }
}

fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    let mut out: String = text.chars().take(limit.saturating_sub(1)).collect();
    out.truncate(out.trim_end().len());
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;


    /// OpenAI-compatible chat response for one position, as vLLM returns it:
    /// the sampled token, its logprob, and top_logprobs.
    fn response(sampled: (&str, f64), top: &[(&str, f64)]) -> Value {
        let top: Vec<Value> = top.iter().map(|(token, logprob)| json!({"token": token, "logprob": logprob, "bytes": token.as_bytes()})).collect();
        json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": sampled.0},
               "logprobs": {"content": [{"token": sampled.0, "logprob": sampled.1, "top_logprobs": top}]},
               "finish_reason": "length"}]})
    }
    fn ln(p: f64) -> f64 {
        p.ln()
    }

    // -------------------------------------------------------------- mock vLLM

    // -------------------------------------------------------------- skill match

    // -------------------------------------------------------------- readout

    #[test]
    fn readout_refuses_what_is_not_a_noul_answer() {
        let unconstrained = readout(&response(("I", ln(0.9)), &[("I", ln(0.9)), ("A", ln(0.05)), ("B", ln(0.01))])).unwrap_err();
        assert!(unconstrained.contains("did not apply structured_outputs.choice") && unconstrained.contains("\"I\""), "{unconstrained}");
        assert!(readout(&json!({"choices": [{"message": {"content": "A"}}]})).unwrap_err().contains("no logprobs"));
        assert!(readout(&response(("A", ln(0.9)), &[("A", ln(0.9))])).unwrap_err().contains("no alternatives"));
        let null_logprob = json!({"choices": [{"logprobs": {"content": [{"token": "A", "logprob": null, "top_logprobs": [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.0}]}]}}]});
        assert!(readout(&null_logprob).unwrap_err().contains("no usable logprob"));
    }

    // -------------------------------------------------------------- request

    // -------------------------------------------------------------- config

    // -------------------------------------------------------------- output
}
