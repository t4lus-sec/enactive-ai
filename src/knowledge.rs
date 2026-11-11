//! Knowledge pass: on every operator message (the one that starts a turn, and
//! any typed while a turn runs) and every autonomous turn, ask the base model
//! Jev-style nouls, one per information source asked, in one parallel pass
//! with a 1000 ms budget (audited, not enforced), and tell s0ma which sources
//! to check before answering.
//!
//! # Which are asked
//!
//! By default only the sources of truth outside s0ma's memory: `internet`
//! and `vm` (`S0MA_KNOW_SOURCES`, a comma list of the names below, or `all`).
//! On ds41d each noul is a prefill of its own, and four run at once, so six
//! took two rounds; two take one. s0ma's memory is searched on every turn
//! instead of when a noul says so: the archive lookahead and the facts
//! (`recall_push.rs`) start with the pass, and offer only good matches (an
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
//! Naming the archive as worth checking still left s0ma to spend a decode step
//! and a tool round trip on a search the harness could have run while the pass
//! was waiting on the model. So it runs it: when the archive noul flags, the
//! pass searches the archive on the operator's own message and ends its note
//! with the best few hits as one-line references — event id, time, kind and a
//! bounded preview.
//!
//! What it attaches are pointers, not content. s0ma still reads an event with
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
/// The questions asked by default: the sources of truth outside s0ma's
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
    /// What s0ma is told to do when this source is flagged.
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
    /// The facts pushed, when the fact lookahead ran (`recall_push.rs`).
    facts: Option<LookaheadOutcome<crate::recall_push::FactsFound>>,
    /// The query rewrite both lookaheads searched with, when it ran.
    rewrite: Option<std::sync::Arc<crate::recall_push::PlanSlot>>,
}

/// Run the pass on the turn's input. `None` means there is nothing to add:
/// disabled or misconfigured. Never blocks a turn: problems are printed and
/// audited, and the turn goes ahead.
pub fn check(memory: &Hippocampus, conversation: &[Message], input: &str) -> Option<Pass> {
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
    let searching = !config.lookahead_budget.is_zero() || push.mode != crate::recall_push::PushMode::Off;
    let plan = match push.rewrite {
        _ if !searching => crate::recall_push::PlanSlot::none(),
        crate::recall_push::Rewrite::Off => crate::recall_push::PlanSlot::none(),
        // Made here, no model asked: the names the last reply mentioned.
        crate::recall_push::Rewrite::Local => {
            let previous = conversation
                .iter()
                .rev()
                .find(|message| message.role == "assistant" && !message.content.trim().is_empty())
                .map(|message| truncate_chars(&message.content, MAX_PROBE_CONTEXT_CHARS))
                .unwrap_or_default();
            crate::recall_push::local_slot(input, &crate::facts::names_in(memory, &previous))
        }
        crate::recall_push::Rewrite::Model => {
            let body = crate::recall_push::rewrite_body(&config.model, &state_block(conversation, input), config.priority);
            crate::recall_push::start_rewrite(client(), url.clone(), body, push.rewrite_wait)
        }
    };
    let lookahead = archive_lookahead(memory, &config, input, std::sync::Arc::clone(&plan));
    let facts = (push.mode != crate::recall_push::PushMode::Off)
        .then(|| crate::recall_push::fact_lookahead(memory, input, conversation, std::sync::Arc::clone(&plan), push));
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
    plan: std::sync::Arc<crate::recall_push::PlanSlot>,
) -> Option<Lookahead> {
    if config.lookahead_budget.is_zero() {
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
    fn facts_found(&self) -> Option<&crate::recall_push::FactsFound> {
        self.facts.as_ref().and_then(|outcome| outcome.result.as_ref().ok())
    }

    /// Archived with the turn when the note was attached, so memory shows why
    /// s0ma went looking.
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
    /// The fact push and the query rewrite (`recall_push.rs`).
    push: crate::recall_push::PushConfig,
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
            push: crate::recall_push::PushConfig::from_vars(get)?,
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
    facts: Option<crate::recall_push::FactLookahead>,
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
    // The facts wait for either noul that would send s0ma looking for one.
    let always = config.push.mode == crate::recall_push::PushMode::Always || !(asked(Source::Archive) || asked(Source::WorldModel));
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
    let response = client.post(url).json(body).send().map_err(|error| format!("request failed: {error}"))?;
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
    use std::{
        collections::HashMap,
        io::{BufRead, BufReader, Read, Write},
        net::TcpListener,
        sync::{Arc, Mutex},
    };

    fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |key| map.get(key).cloned()
    }

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

    fn config(budget_ms: u64) -> Config {
        Config {
            thresholds: [0.30; SOURCES],
            skill_threshold: 0.30,
            // The judged match is what most skill tests exercise; the cosine
            // one has its own.
            skill_judge: true,
            skill_min_cosine: DEFAULT_SKILL_MIN_COSINE,
            budget: Duration::from_millis(budget_ms),
            // The lookahead is given its own fixture where it is under test.
            lookahead_budget: Duration::ZERO,
            lookahead_hits: DEFAULT_LOOKAHEAD_HITS,
            model: "base".into(),
            priority: None,
            push: crate::recall_push::PushConfig { mode: crate::recall_push::PushMode::Off, rewrite: crate::recall_push::Rewrite::Off, ..Default::default() },
            // Every question, as the tests of each were written.
            sources: NOULS.iter().map(|noul| noul.source).collect(),
            lookahead_min_cosine: DEFAULT_LOOKAHEAD_MIN_COSINE,
        }
    }

    fn candidate(name: &str, description: &str, cosine: f32) -> Candidate {
        Candidate { name: name.into(), description: description.into(), cosine }
    }

    /// The name a skill noul's prompt asks about.
    fn asked_skill(prompt: &str) -> Option<String> {
        let rest = prompt.split_once("your skill \"")?.1;
        Some(rest.split_once('"')?.0.to_owned())
    }

    fn pass(nouls: [f64; SOURCES]) -> Pass {
        let answers = NOULS
            .iter()
            .zip(nouls)
            .map(|(noul, p_true)| Answer {
                source: noul.source,
                readout: Readout { noul: p_true, label_mass: 0.9 },
                p_lookup: noul.p_lookup(p_true),
                threshold: 0.30,
            })
            .collect();
        let timings = NOULS.iter().map(|n| (n.source, Duration::from_millis(180))).collect();
        Pass {
            answers,
            failures: Vec::new(),
            timings,
            took: Duration::from_millis(180),
            budget: DEFAULT_BUDGET,
            skills: SkillStage::default(),
            lookahead: None,
            facts: None,
            rewrite: None,
        }
    }

    // -------------------------------------------------------------- mock vLLM

    #[derive(Clone, Copy)]
    enum Reply {
        /// Answer with this P(true), after a delay.
        Noul(f64, u64),
        /// Ignore the constraint: answer "I".
        Unconstrained,
        /// 404, as vLLM answers an unknown model.
        NotFound,
    }

    struct Mock {
        url: String,
        bodies: Arc<Mutex<Vec<Value>>>,
    }

    fn mock(replies: [Reply; SOURCES]) -> Mock {
        mock_with(replies, &[])
    }

    /// A minimal HTTP/1.1 server: one thread per connection, the reply chosen
    /// by which noul's instructions the request carries, or for a skill noul,
    /// by the skill it names.
    fn mock_with(replies: [Reply; SOURCES], skill_replies: &[(&str, Reply)]) -> Mock {
        let skill_replies: Arc<HashMap<String, Reply>> =
            Arc::new(skill_replies.iter().map(|(name, reply)| (name.to_string(), *reply)).collect());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/chat/completions", listener.local_addr().unwrap());
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&bodies);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let seen = Arc::clone(&seen);
                let skill_replies = Arc::clone(&skill_replies);
                thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut length = 0;
                    loop {
                        let mut line = String::new();
                        reader.read_line(&mut line).unwrap();
                        if line == "\r\n" || line.is_empty() {
                            break;
                        }
                        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                            length = value.trim().parse().unwrap();
                        }
                    }
                    let mut raw = vec![0; length];
                    reader.read_exact(&mut raw).unwrap();
                    let body: Value = serde_json::from_slice(&raw).unwrap();
                    let prompt = body["messages"][1]["content"].as_str().unwrap().to_owned();
                    seen.lock().unwrap().push(body);
                    // A skill noul by the skill it names; a source noul by its
                    // instructions; any other question (`judge`) by a key it contains.
                    let reply = match asked_skill(&prompt) {
                        Some(name) => skill_replies[&name],
                        None => match NOULS.iter().position(|n| prompt.contains(n.instructions)) {
                            Some(index) => replies[index],
                            None => *skill_replies.iter().find(|(key, _)| prompt.contains(key.as_str())).unwrap().1,
                        },
                    };
                    let (status, payload) = match reply {
                        Reply::Noul(p_true, delay) => {
                            thread::sleep(Duration::from_millis(delay));
                            let sampled = if p_true >= 0.5 { ("A", ln(p_true)) } else { ("B", ln(1.0 - p_true)) };
                            ("200 OK", response(sampled, &[("A", ln(p_true)), ("B", ln(1.0 - p_true))]))
                        }
                        Reply::Unconstrained => ("200 OK", response(("I", ln(0.9)), &[("I", ln(0.9)), ("A", ln(0.05))])),
                        Reply::NotFound => ("404 Not Found", json!({"error": {"message": "The model `base` does not exist."}})),
                    };
                    let payload = payload.to_string();
                    let _ = write!(
                        stream,
                        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                        payload.len()
                    );
                });
            }
        });
        Mock { url, bodies }
    }

    #[test]
    fn judge_asks_each_question_at_once_on_the_shared_state() {
        let server = mock_with(
            [Reply::Noul(0.5, 0); SOURCES],
            &[("episode alpha", Reply::Noul(0.9, 150)), ("episode beta", Reply::Noul(0.2, 150)), ("episode gamma", Reply::NotFound)],
        );
        let question = |name: &str| Question {
            instructions: format!("Past episode: {name}. Does it bear on the request?"),
            when_true: "It applies.",
            when_false: "It does not.",
        };
        let started = Instant::now();
        let answers = judge_with(
            client(),
            &server.url,
            "base",
            Some(7),
            "State:\nCurrent request: check the recorder",
            &[question("episode alpha"), question("episode beta"), question("episode gamma")],
        );
        assert!(started.elapsed() < Duration::from_millis(290), "asked at once, not one after another");
        assert!((answers[0].as_ref().unwrap() - 0.9).abs() < 1e-9);
        assert!((answers[1].as_ref().unwrap() - 0.2).abs() < 1e-9);
        assert!(answers[2].as_ref().unwrap_err().contains("404"), "{answers:?}");
        for body in server.bodies.lock().unwrap().iter() {
            assert_eq!(body["max_tokens"], 1);
            assert_eq!(body["priority"], 7);
            assert_eq!(body["structured_outputs"]["choice"], json!(["A", "B"]));
            assert!(body["messages"][1]["content"].as_str().unwrap().starts_with("State:\nCurrent request: check the recorder"));
        }
    }

    #[test]
    fn pass_asks_every_source_at_once_and_reads_each() {
        // self knows it (0.9), internet 0.4 (scaled to 0.7), the rest unneeded; 150 ms each.
        let server = mock([0.9, 0.4, 0.1, 0.05, 0.02, 0.15].map(|p| Reply::Noul(p, 150)));
        let pass = run(&server.url, &config(1_000), &[], "what's new in the latest llama.cpp release?", &no_skills, None);
        // Six 150 ms requests in well under the 900 ms they would take in turn.
        assert!(pass.took < Duration::from_millis(500), "took {:?}", pass.took);
        let lookups: Vec<(Source, f64)> = pass.answers.iter().map(|a| (a.source, (a.p_lookup * 100.0).round() / 100.0)).collect();
        assert_eq!(
            lookups,
            vec![
                (Source::SelfKnowledge, 0.1),
                (Source::Internet, 0.7),
                (Source::Archive, 0.1),
                (Source::Thread, 0.05),
                (Source::Vm, 0.02),
                (Source::WorldModel, 0.15),
            ]
        );
        let flagged: Vec<Source> = pass.flagged().iter().map(|a| a.source).collect();
        assert_eq!(flagged, vec![Source::Internet]);
        assert!(!pass.over_budget() && pass.failures.is_empty() && pass.audit_record("q").is_none());
        assert!(!pass.status_line().contains("audit"));

        let bodies = server.bodies.lock().unwrap();
        assert_eq!(bodies.len(), SOURCES);
        for body in bodies.iter() {
            assert_eq!(body["structured_outputs"], json!({"choice": ["A", "B"]}));
            assert_eq!(body["max_tokens"], 1);
            assert!(body.get("priority").is_none());
        }
    }

    #[test]
    fn self_noul_looks_up_on_no() {
        // s0ma doesn't know (P(true) 0.4 → P(no) 0.6): flagged alone. Internet
        // at 0.1 scales to 0.25, under the test threshold of 0.30.
        let server = mock([0.4, 0.1, 0.1, 0.1, 0.1, 0.1].map(|p| Reply::Noul(p, 0)));
        let pass = run(&server.url, &config(1_000), &[], "who maintains the zxqv-wobble crate?", &no_skills, None);
        let flagged: Vec<(Source, f64)> = pass.flagged().iter().map(|a| (a.source, (a.p_lookup * 100.0).round() / 100.0)).collect();
        assert_eq!(flagged, vec![(Source::SelfKnowledge, 0.6)]);
    }

    #[test]
    fn internet_is_scaled_up_and_capped_at_one() {
        let internet = noul(Source::Internet);
        for (p_true, expected) in [(0.0, 0.1), (0.2, 0.4), (0.27, 0.505), (0.5, 0.85), (0.6, 1.0), (1.0, 1.0)] {
            assert!((internet.p_lookup(p_true) - expected).abs() < 1e-9, "{p_true} -> {}", internet.p_lookup(p_true));
        }
        // Every other source is unscaled.
        for other in NOULS.iter().filter(|n| n.source != Source::Internet) {
            assert_eq!((other.scale, other.offset), (1.0, 0.0), "{:?}", other.source);
        }
        assert_eq!(noul(Source::SelfKnowledge).p_lookup(0.8), 1.0 - 0.8);

        // The archive record keeps the raw noul beside the scaled value.
        let pass = pass([0.9, 0.5, 0.1, 0.1, 0.1, 0.1]);
        assert!(pass.archive_record().unwrap().contains("\ninternet: noul 0.500, p_lookup 0.850"));
    }

    #[test]
    fn world_model_noul_is_its_own_source() {
        let server = mock([0.9, 0.0, 0.1, 0.1, 0.1, 0.8].map(|p| Reply::Noul(p, 0)));
        let pass = run(&server.url, &config(1_000), &[], "how does the busqueda box's auth flow work again?", &no_skills, None);
        let flagged: Vec<Source> = pass.flagged().iter().map(|a| a.source).collect();
        assert_eq!(flagged, vec![Source::WorldModel]);
        let block = pass.context_block().unwrap();
        assert!(block.contains("\n- world_model (0.80): world_model_list to find the relevant model, then world_model_get"), "{block}");
    }

    #[test]
    fn overrun_runs_to_completion_and_is_audited() {
        let mut replies = [Reply::Noul(0.1, 20); SOURCES];
        replies[0] = Reply::Noul(0.9, 20); // s0ma knows it
        replies[1] = Reply::Noul(0.0, 20); // internet scales to 0.10
        replies[4] = Reply::Noul(0.9, 400); // vm
        let server = mock(replies);
        let pass = run(&server.url, &config(200), &[], "what did the last nmap scan find?", &no_skills, None);
        // Not cut off: the slow noul still answered, and it still counts.
        assert!(pass.took >= Duration::from_millis(400), "{:?}", pass.took);
        assert!(pass.failures.is_empty() && pass.answers.len() == SOURCES);
        let flagged: Vec<Source> = pass.flagged().iter().map(|a| a.source).collect();
        assert_eq!(flagged, vec![Source::Vm]);
        assert!(pass.context_block().unwrap().contains("- vm (0.90)"));

        assert!(pass.over_budget());
        let line = pass.status_line();
        assert!(line.contains("/200ms OVER BUDGET p(lookup):") && line.contains("\n[know] audit: self "), "{line}");
        let vm = pass.timings[4].1.as_millis();
        assert!(line.contains(&format!("; vm {vm}ms; world_model ")), "{line}");

        let record = pass.audit_record("what did the last nmap scan find?").unwrap();
        assert_eq!((record["over_budget"].clone(), record["budget_ms"].clone()), (json!(true), json!(200)));
        assert_eq!(record["input_chars"], 33);
        assert_eq!(record["nouls"][4]["source"], "vm");
        assert!(record["nouls"][4]["ms"].as_u64().unwrap() >= 400);
        assert!((record["nouls"][4]["p_lookup"].as_f64().unwrap() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn failed_nouls_are_left_out_and_the_rest_still_count() {
        let mut replies = [Reply::Noul(0.1, 0); SOURCES];
        replies[0] = Reply::Noul(0.9, 0);
        replies[1] = Reply::Noul(0.8, 0);
        replies[2] = Reply::Unconstrained;
        let pass = run(&mock(replies).url, &config(1_000), &[], "hi", &no_skills, None);
        assert_eq!(pass.answers.len(), SOURCES - 1);
        assert_eq!(pass.failures.len(), 1);
        assert_eq!(pass.failures[0].0, Source::Archive);
        assert!(pass.failures[0].1.contains("did not apply structured_outputs.choice"));
        // Internet still flags the turn.
        assert!(pass.context_block().unwrap().contains("- internet (1.00)"));
        let line = pass.status_line();
        assert!(line.contains("· archive error ·") && line.contains("; archive ") && line.contains("did not apply"), "{line}");
        let record = pass.audit_record("hi").unwrap();
        assert_eq!(record["over_budget"], false);
        assert!(record["nouls"][2]["error"].as_str().unwrap().contains("did not apply"));
        assert!(record["nouls"][2].get("p_lookup").is_none());

        // Every noul failing still yields a pass, just one with nothing to add.
        let pass = run(&mock([Reply::NotFound; SOURCES]).url, &config(1_000), &[], "hi", &no_skills, None);
        assert!(pass.answers.is_empty() && pass.context_block().is_none() && pass.archive_record().is_none());
        assert!(pass.failures.iter().all(|(_, e)| e.contains("HTTP 404") && e.contains("does not exist")));

        // Nothing listening.
        let closed = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let pass = run(&format!("http://{closed}/v1/chat/completions"), &config(1_000), &[], "hi", &no_skills, None);
        assert_eq!(pass.failures.len(), SOURCES);
        assert!(pass.failures[0].1.contains("request failed"));
        assert!(pass.context_block().is_none());
    }

    // -------------------------------------------------------------- skill match

    #[test]
    fn confirmed_skills_are_named_as_pointers() {
        // Sources: self knows it, the rest low.
        let replies = [0.9, 0.0, 0.1, 0.1, 0.1, 0.1].map(|p| Reply::Noul(p, 0));
        let skills = [("nmap-triage", Reply::Noul(0.9, 0)), ("web-recon", Reply::Noul(0.2, 0)), ("backup-rotation", Reply::Noul(0.6, 0))];
        let server = mock_with(replies, &skills);
        let shortlist = || {
            Ok(vec![
                candidate("nmap-triage", "Triage nmap output: open ports, versions, next probes.", 0.71),
                candidate("web-recon", "Enumerate a web app's routes and tech stack.", 0.55),
                candidate("backup-rotation", "Rotate and verify the nightly backups.", 0.40),
            ])
        };
        let input = "what did the last nmap scan find, and what should I probe next?";
        let pass = run(&server.url, &config(1_000), &[], input, &shortlist, None);

        let block = pass.context_block().unwrap();
        assert!(block.starts_with("\n\n[HARNESS KNOWLEDGE CHECK]\nFrom the harness, not the operator"), "{block}");
        // Only skills to report: no source list, and no generic skills source.
        assert!(!block.contains("check the flagged sources") && !block.contains("\n- skills ("), "{block}");
        let nmap = block
            .find("\nSkills whose procedures likely apply, likeliest first; read each with skill_read before applying it:\n- skill `nmap-triage` (0.90): Triage nmap output: open ports, versions, next probes.")
            .expect(&block);
        let backup = block.find("\n- skill `backup-rotation` (0.60): Rotate and verify").unwrap();
        assert!(nmap < backup && !block.contains("web-recon"));
        assert!(block.ends_with("rather than filling the gap."));

        // The rendered pointers are what the next turn's shortlist skips.
        let next_turn = [Message::user(format!("{input}{block}"))];
        assert_eq!(skill_match::already_in_context(&next_turn).into_iter().collect::<Vec<_>>(), ["backup-rotation", "nmap-triage"]);

        assert!(pass.status_line().ends_with(
            " | skill match: nmap-triage 0.90* (cos 0.71) · web-recon 0.20 (cos 0.55) · backup-rotation 0.60* (cos 0.40)]"
        ));
        let record = pass.archive_record().unwrap();
        assert!(!record.contains("\nskills: "), "{record}");
        assert!(record.contains("\nskill nmap-triage: noul 0.900 (threshold 0.30, label mass 1.00, cosine 0.710) flagged"));
        assert!(record.contains("\nskill web-recon: noul 0.200 (threshold 0.30, label mass 1.00, cosine 0.550)\n"));
        assert!(record.ends_with("\ninstruction: read skill nmap-triage, backup-rotation before answering"), "{record}");

        // Nine nouls, all on the same state prefix; the skill ones quote the description.
        let bodies = server.bodies.lock().unwrap();
        assert_eq!(bodies.len(), SOURCES + 3);
        let state = state_block(&[], input);
        for body in bodies.iter() {
            assert!(body["messages"][1]["content"].as_str().unwrap().starts_with(&state));
            assert_eq!(body["structured_outputs"], json!({"choice": ["A", "B"]}));
        }
        let web = bodies
            .iter()
            .map(|b| b["messages"][1]["content"].as_str().unwrap())
            .find(|p| p.contains("\"web-recon\""))
            .unwrap();
        assert!(web.contains("Does a good reply to the message need your skill \"web-recon\", a written procedure described as: Enumerate a web app's routes and tech stack."));
        assert!(web.ends_with(&format!("\n\nOptions:\nA: true. {SKILL_WHEN_TRUE}\nB: false. {SKILL_WHEN_FALSE}")));
    }

    #[test]
    fn by_default_skills_are_named_by_cosine_without_asking_the_model() {
        let replies = [0.9, 0.0, 0.1, 0.1, 0.1, 0.1].map(|p| Reply::Noul(p, 0));
        let server = mock(replies);
        let shortlist = || {
            Ok(vec![
                candidate("nmap-triage", "Triage nmap output: open ports, versions, next probes.", 0.71),
                candidate("web-recon", "Enumerate a web app's routes and tech stack.", 0.55),
                candidate("backup-rotation", "Rotate and verify the nightly backups.", 0.40),
            ])
        };
        let config = Config { skill_judge: false, ..config(1_000) };
        let input = "what did the last nmap scan find, and what should I probe next?";
        let pass = run(&server.url, &config, &[], input, &shortlist, None);

        // Only the six sources went to the model.
        assert_eq!(server.bodies.lock().unwrap().len(), SOURCES);
        let block = pass.context_block().unwrap();
        let nmap = block
            .find("(matched by description, not checked, so one may not apply); read one with skill_read before applying it:\n- skill `nmap-triage` (cos 0.71): Triage nmap output")
            .expect(&block);
        let web = block.find("\n- skill `web-recon` (cos 0.55): Enumerate").unwrap();
        assert!(nmap < web && !block.contains("backup-rotation"), "{block}");
        // The rendered pointers are still what the next turn's shortlist skips.
        let next_turn = [Message::user(format!("{input}{block}"))];
        assert_eq!(skill_match::already_in_context(&next_turn).into_iter().collect::<Vec<_>>(), ["nmap-triage", "web-recon"]);

        assert!(pass.status_line().ends_with(" | skill match: nmap-triage cos 0.71* · web-recon cos 0.55* · backup-rotation cos 0.40]"));
        let record = pass.archive_record().unwrap();
        assert!(record.contains("\nskill nmap-triage: cosine 0.710 (cutoff 0.50) flagged"), "{record}");
        assert!(record.contains("\nskill backup-rotation: cosine 0.400 (cutoff 0.50)\n"), "{record}");
        assert!(record.ends_with("\ninstruction: read skill nmap-triage, web-recon before answering"), "{record}");
        // Nothing failed or ran over: no audit; and nothing timed per skill.
        assert!(pass.audit_record(input).is_none());
        assert!(!pass.detail().contains("skill nmap-triage"));

        // A higher bar names fewer.
        let strict = Config { skill_judge: false, skill_min_cosine: 0.7, ..config };
        let pass = run(&mock(replies).url, &strict, &[], input, &shortlist, None);
        let block = pass.context_block().unwrap();
        assert!(block.contains("`nmap-triage`") && !block.contains("`web-recon`"), "{block}");
    }

    #[test]
    fn sources_and_skills_share_one_note() {
        // internet flagged; one skill confirmed.
        let replies = [0.9, 0.6, 0.1, 0.1, 0.1, 0.1].map(|p| Reply::Noul(p, 0));
        let server = mock_with(replies, &[("nmap-triage", Reply::Noul(0.8, 0))]);
        let shortlist = || Ok(vec![candidate("nmap-triage", "Triage nmap output.", 0.6)]);
        let pass = run(&server.url, &config(1_000), &[], "scan results?", &shortlist, None);
        let block = pass.context_block().unwrap();
        let sources = block.find(" Before answering, check the flagged sources, likeliest first:\n- internet (1.00)").expect(&block);
        let skills = block.find("\nSkills whose procedures likely apply").unwrap();
        assert!(sources < skills && !block.contains("\n- skills ("));
        assert!(pass.archive_record().unwrap().ends_with("\ninstruction: check internet; read skill nmap-triage before answering"));
        // No skill confirmed: just the sources, with no generic skills line in
        // its place.
        let server = mock_with(replies, &[("nmap-triage", Reply::Noul(0.1, 0))]);
        let pass = run(&server.url, &config(1_000), &[], "scan results?", &shortlist, None);
        let block = pass.context_block().unwrap();
        assert!(block.contains("\n- internet (1.00)") && !block.contains("\n- skills (") && !block.contains("Skills whose procedures"), "{block}");
    }

    #[test]
    fn skill_match_runs_alongside_and_is_waited_for() {
        // Source nouls take 150 ms; the shortlist 100 ms, then its noul 300 ms.
        let replies = [0.9, 0.0, 0.1, 0.1, 0.1, 0.1].map(|p| Reply::Noul(p, 150));
        let server = mock_with(replies, &[("nmap-triage", Reply::Noul(0.9, 300))]);
        let shortlist = || {
            thread::sleep(Duration::from_millis(100));
            Ok(vec![candidate("nmap-triage", "Triage nmap output.", 0.7)])
        };
        let pass = run(&server.url, &config(200), &[], "what did the scan find?", &shortlist, None);
        // Not cut off: the skill noul answered, and the pass waited for it.
        assert!(pass.took >= Duration::from_millis(400), "{:?}", pass.took);
        // Alongside, not after: the source nouls finished long before.
        assert!(pass.timings.iter().all(|(_, at)| *at < Duration::from_millis(350)), "{:?}", pass.timings);
        assert!(pass.context_block().unwrap().contains("- skill `nmap-triage` (0.90)"));

        // Over budget, so audited, with the skill stage's timings.
        let line = pass.status_line();
        assert!(line.contains("OVER BUDGET") && line.contains("; skill shortlist 1") && line.contains("; skill nmap-triage 4"), "{line}");
        let record = pass.audit_record("what did the scan find?").unwrap();
        assert!(record["skill_match"]["shortlist_ms"].as_u64().unwrap() >= 100);
        assert_eq!(record["skill_match"]["skills"][0]["name"], "nmap-triage");
        assert!((record["skill_match"]["skills"][0]["noul"].as_f64().unwrap() - 0.9).abs() < 1e-9);
        assert!(record["skill_match"]["skills"][0]["ms"].as_u64().unwrap() >= 400);
    }

    #[test]
    fn skill_match_failures_cost_only_the_pointers() {
        let replies = [0.9, 0.6, 0.1, 0.1, 0.1, 0.1].map(|p| Reply::Noul(p, 0));

        // The shortlist failed (embedding server down): the sources still count.
        let server = mock(replies);
        let down = || Err("embedding the message: connection refused".to_owned());
        let pass = run(&server.url, &config(1_000), &[], "hi", &down, None);
        assert_eq!(pass.answers.len(), SOURCES);
        let block = pass.context_block().unwrap();
        assert!(block.contains("- internet (1.00)") && !block.contains("Skills whose"));
        let line = pass.status_line();
        assert!(line.contains(" | skill match: error]") && line.contains("; skill shortlist 0ms: embedding the message: connection refused"), "{line}");
        let record = pass.audit_record("hi").unwrap();
        assert_eq!(record["skill_match"]["error"], "embedding the message: connection refused");
        assert_eq!(server.bodies.lock().unwrap().len(), SOURCES);

        // One skill's noul failed: the other is still named.
        let server = mock_with(replies, &[("nmap-triage", Reply::Noul(0.8, 0)), ("web-recon", Reply::Unconstrained)]);
        let shortlist = || Ok(vec![candidate("nmap-triage", "Triage.", 0.7), candidate("web-recon", "Recon.", 0.6)]);
        let pass = run(&server.url, &config(1_000), &[], "hi", &shortlist, None);
        assert!(pass.context_block().unwrap().contains("- skill `nmap-triage` (0.80)"));
        assert!(pass.status_line().contains("· web-recon error]"));
        let record = pass.archive_record().unwrap();
        assert!(record.contains("\nskill web-recon: error (answer token \"I\""), "{record}");
        let audit = pass.audit_record("hi").unwrap();
        assert!(audit["skill_match"]["skills"][1]["error"].as_str().unwrap().contains("structured_outputs"));

        // An empty shortlist changes nothing: same line as a pass without skills.
        let pass = run(&mock(replies).url, &config(1_000), &[], "hi", &no_skills, None);
        assert!(!pass.status_line().contains("skill match") && pass.audit_record("hi").is_none());
    }

    // -------------------------------------------------------------- readout

    #[test]
    fn labels_are_single_letters_compared_over_ascii() {
        for t in ["A", " A", "a", "A\n"] {
            assert_eq!(label(t), Some(true), "{t:?}");
        }
        for f in ["B", " b"] {
            assert_eq!(label(f), Some(false), "{f:?}");
        }
        for neither in ["A:", "AB", "Yes", "no", "", "C", "\u{0410}"] {
            assert_eq!(label(neither), None, "{neither:?}");
        }
    }

    #[test]
    fn noul_is_the_softmax_over_the_labels() {
        // Raw (pre-constraint) distribution: the model leaned to "I", then A, then B.
        let r = readout(&response(("A", ln(0.30)), &[("I", ln(0.50)), ("A", ln(0.30)), ("B", ln(0.10)), (" a", ln(0.05))])).unwrap();
        assert!((r.noul - 0.30 / 0.40).abs() < 1e-9, "{r:?}");
        assert!((r.label_mass - 0.40).abs() < 1e-9);
        // Masked reporting (labels only) gives the same answer.
        let masked = readout(&response(("A", ln(0.75)), &[("A", ln(0.75)), ("B", ln(0.25))])).unwrap();
        assert!((masked.noul - r.noul).abs() < 1e-9);
        // Sampling B doesn't change the distribution read.
        let sampled_b = readout(&response(("B", ln(0.10)), &[("I", ln(0.50)), ("A", ln(0.30)), ("B", ln(0.10))])).unwrap();
        assert!((sampled_b.noul - 0.75).abs() < 1e-9);
    }

    #[test]
    fn unreported_label_counts_as_zero() {
        let r = readout(&response(("A", ln(0.9)), &[("A", ln(0.9)), ("The", ln(0.05))])).unwrap();
        assert_eq!(r.noul, 1.0);
        let r = readout(&response(("B", 0.0), &[("B", 0.0), ("A", -9999.0)])).unwrap();
        assert_eq!(r.noul, 0.0);
    }

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

    #[test]
    fn six_nouls_one_per_source_with_their_lookup_answer() {
        let sources: Vec<(Source, bool)> = NOULS.iter().map(|n| (n.source, n.lookup_on)).collect();
        assert_eq!(
            sources,
            vec![
                (Source::SelfKnowledge, false),
                (Source::Internet, true),
                (Source::Archive, true),
                (Source::Thread, true),
                (Source::Vm, true),
                (Source::WorldModel, true),
            ]
        );
        let names: Vec<&str> = NOULS.iter().map(|n| n.source.name()).collect();
        // No generic skills noul: the skill match asks about each skill instead.
        assert_eq!(names, ["self", "internet", "archive", "thread", "vm", "world_model"]);
        // Each hint names the tools s0ma has for that source.
        for (source, tool) in [
            (Source::Internet, "browser_read"),
            (Source::Archive, "archive_search"),
            (Source::Thread, "thread_get"),
            (Source::Vm, "vm_exec"),
            (Source::WorldModel, "world_model_get"),
        ] {
            assert!(noul(source).hint.contains(tool), "{source:?}");
        }
    }

    #[test]
    fn probe_request_forces_one_label_with_thinking_off() {
        let messages = probe_messages(&state_block(&[], "what is the capital of France?"), &NOULS[0]);
        let body = probe_body("qwen38-base", &messages, None);
        assert_eq!(body["max_tokens"], 1);
        assert_eq!(body["logprobs"], true);
        assert_eq!(body["top_logprobs"], 20);
        assert_eq!(body["temperature"], 1.0);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(body["structured_outputs"], json!({"choice": ["A", "B"]}));
        assert!(body.get("grammar").is_none() && body.get("guided_choice").is_none() && body.get("priority").is_none());
        assert_eq!(body["model"], "qwen38-base");
        assert!(body.get("tools").is_none() && body.get("stream").is_none());
        // Scheduled like the main turn when priority scheduling is on.
        assert_eq!(probe_body("m", &messages, Some(0))["priority"], 0);
    }

    #[test]
    fn state_first_so_every_noul_shares_a_prefix() {
        let conversation = vec![
            Message::user("earlier".into()),
            Message::assistant(format!("previous answer {}", "x".repeat(3_000))),
            Message::user("current".into()),
        ];
        let state = state_block(&conversation, "and the year after that?");
        assert!(state.starts_with("State:\nYour previous reply:\nprevious answer"));
        assert!(state.contains('…'));
        assert!(state.ends_with("Message:\nand the year after that?"));

        let prompts: Vec<String> = NOULS.iter().map(|n| probe_messages(&state, n)[1].content.clone()).collect();
        for (prompt, noul) in prompts.iter().zip(&NOULS) {
            assert!(prompt.starts_with(&state), "state first");
            assert!(prompt.ends_with(&format!("\n\nOptions:\nA: true. {}\nB: false. {}", noul.when_true, noul.when_false)));
            assert!(prompt.contains(&format!("\n\nQuestion (noul: true or false)\n{}\n", noul.instructions)));
        }
        // System prompts identical, so the shared prefix is system + state.
        let systems: Vec<String> = NOULS.iter().map(|n| probe_messages(&state, n)[0].content.clone()).collect();
        assert!(systems.iter().all(|s| s == PROBE_SYSTEM));
        assert!(!state_block(&[], "hi").contains("previous reply"));
    }

    // -------------------------------------------------------------- config

    #[test]
    fn configuration() {
        let config = Config::from_vars(vars(&[])).unwrap().unwrap();
        assert_eq!(config.thresholds, [DEFAULT_THRESHOLD; SOURCES]);
        assert_eq!(config.skill_threshold, DEFAULT_THRESHOLD);
        assert_eq!(DEFAULT_THRESHOLD, 0.50);
        // The skill-match bar follows the shared threshold unless set itself.
        let shared = Config::from_vars(vars(&[("S0MA_KNOW_THRESHOLD", "0.4")])).unwrap().unwrap();
        assert_eq!(shared.skill_threshold, 0.4);
        let own = Config::from_vars(vars(&[("S0MA_KNOW_THRESHOLD", "0.4"), ("S0MA_KNOW_THRESHOLD_SKILL_MATCH", "0.65")])).unwrap().unwrap();
        assert_eq!((own.skill_threshold, own.thresholds[0]), (0.65, 0.4));
        assert!(Config::from_vars(vars(&[("S0MA_KNOW_THRESHOLD_SKILL_MATCH", "2")])).is_err());
        // Skills by cosine unless judging is asked for.
        assert!(!config.skill_judge);
        assert_eq!(config.skill_min_cosine, DEFAULT_SKILL_MIN_COSINE);
        let judged = Config::from_vars(vars(&[("S0MA_SKILL_MATCH_JUDGE", "1"), ("S0MA_SKILL_MATCH_MIN_COSINE", "0.42")])).unwrap().unwrap();
        assert!(judged.skill_judge);
        assert_eq!(judged.skill_min_cosine, 0.42);
        assert!(Config::from_vars(vars(&[("S0MA_SKILL_MATCH_JUDGE", "maybe")])).is_err());
        assert!(Config::from_vars(vars(&[("S0MA_SKILL_MATCH_MIN_COSINE", "1.5")])).is_err());
        assert_eq!(config.budget, Duration::from_millis(1_000));

        let config = Config::from_vars(vars(&[
            ("S0MA_KNOW_THRESHOLD", "0.4"),
            ("S0MA_KNOW_THRESHOLD_ARCHIVE", "0.6"),
            ("S0MA_KNOW_THRESHOLD_SELF", "0.55"),
            ("S0MA_KNOW_THRESHOLD_WORLD_MODEL", "0.45"),
            ("S0MA_KNOW_BUDGET_MS", "600"),
            ("S0MA_KNOW_MODEL", "base"),
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(config.thresholds, [0.55, 0.4, 0.6, 0.4, 0.4, 0.45]);
        assert_eq!((config.budget, config.model.as_str()), (Duration::from_millis(600), "base"));

        for off in ["0", "off", "FALSE"] {
            assert!(Config::from_vars(vars(&[("S0MA_KNOW", off)])).unwrap().is_none());
        }
        for bad in [("S0MA_KNOW_THRESHOLD", "1.5"), ("S0MA_KNOW_THRESHOLD_VM", "lots"), ("S0MA_KNOW_BUDGET_MS", "0"), ("S0MA_KNOW_BUDGET_MS", "1s")] {
            assert!(Config::from_vars(vars(&[bad])).is_err(), "{bad:?}");
        }
        // The removed alternative constraints are refused, not ignored.
        let removed = Config::from_vars(vars(&[("S0MA_KNOW_CONSTRAINT", "llamacpp")])).err().unwrap();
        assert!(removed.contains("removed"), "{removed}");
        // The removed generic skills noul's threshold is refused with the setting
        // that replaced it.
        let skills = Config::from_vars(vars(&[("S0MA_KNOW_THRESHOLD_SKILLS", "0.5")])).err().unwrap();
        assert!(skills.contains("S0MA_KNOW_THRESHOLD_SKILL_MATCH"), "{skills}");
        // The old name of the self threshold is refused with the new one.
        let renamed = Config::from_vars(vars(&[("S0MA_KNOW_THRESHOLD_MODEL", "0.5")])).err().unwrap();
        assert!(renamed.contains("S0MA_KNOW_THRESHOLD_SELF"), "{renamed}");
    }

    // -------------------------------------------------------------- output

    #[test]
    fn note_lists_flagged_sources_likeliest_first() {
        // self P(true) 0.5 → p_lookup 0.5; internet 0.4 → 0.7; vm 0.35;
        // world_model 0.4; rest low.
        let pass = pass([0.5, 0.4, 0.1, 0.3, 0.35, 0.4]);
        let block = pass.context_block().unwrap();
        assert!(block.starts_with("\n\n[HARNESS KNOWLEDGE CHECK]\nFrom the harness, not the operator"));
        let internet = block.find("\n- internet (0.70): browser_navigate").unwrap();
        let own = block.find("\n- self (0.50): your own knowledge").unwrap();
        let world = block.find("\n- world_model (0.40): world_model_list").unwrap();
        let vm = block.find("\n- vm (0.35): shell_read").unwrap();
        assert!(internet < own && own < world && world < vm);
        // 0.30 is not above the threshold.
        assert!(!block.contains("- thread") && !block.contains("- archive") && !block.contains("- skills"));
        assert!(block.contains("say what you still don't know"));

        let record = pass.archive_record().unwrap();
        assert!(record.starts_with("[harness:knowledge_check]\npass: 180 ms of 1000 ms\nself: noul 0.500, p_lookup 0.500"));
        assert!(record.contains("\ninternet: noul 0.400, p_lookup 0.700 (threshold 0.30"));
        assert!(record.contains("\nthread: noul 0.300, p_lookup 0.300 (threshold 0.30, label mass 0.90)\n"));
        assert!(record.ends_with("instruction: check internet, self, world_model, vm before answering"));

        assert_eq!(
            pass.status_line(),
            "[know 180ms/1000ms p(lookup): self 0.50* · internet 0.70* · archive 0.10 · thread 0.30 · vm 0.35* · world_model 0.40*]"
        );
    }

    #[test]
    fn nothing_flagged_adds_nothing() {
        let pass = pass([0.95, 0.1, 0.1, 0.1, 0.1, 0.1]);
        assert!(pass.context_block().is_none() && pass.archive_record().is_none());
    }

    /// `config`, with the archive lookahead on and the given wait.
    fn with_lookahead(budget_ms: u64) -> Config {
        Config { lookahead_budget: Duration::from_millis(budget_ms), ..config(1_000) }
    }

    /// A lookahead that records whether it ran, and answers as told.
    fn fake_lookahead(
        answer: Result<crate::archive::Lookahead, String>,
        delay: Duration,
    ) -> (Lookahead, Arc<std::sync::atomic::AtomicUsize>) {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&runs);
        let search: Lookahead = Box::new(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            thread::sleep(delay);
            answer
        });
        (search, runs)
    }

    fn ran(runs: &Arc<std::sync::atomic::AtomicUsize>) -> usize {
        runs.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// p_true per noul, with only the archive one over the 0.30 threshold.
    const ARCHIVE_ONLY: [f64; SOURCES] = [0.9, 0.05, 0.8, 0.05, 0.05, 0.05];
    /// The same with the archive noul well under it.
    const NOTHING: [f64; SOURCES] = [0.9, 0.05, 0.05, 0.05, 0.05, 0.05];

    #[test]
    fn the_lookahead_runs_when_the_archive_noul_flags_and_its_hits_reach_the_note() {
        let server = mock(ARCHIVE_ONLY.map(|p| Reply::Noul(p, 20)));
        let found = crate::archive::Lookahead::for_test(&[(1423, "the lease renewal is due on the 30th")], None);
        let (search, runs) = fake_lookahead(Ok(found), Duration::ZERO);
        let pass = run(&server.url, &with_lookahead(1_000), &[], "when is the lease up?", &no_skills, Some(search));
        assert_eq!(ran(&runs), 1);
        let block = pass.context_block().unwrap();
        // The archive is named as flagged, but told it need not search again.
        assert!(block.contains(&format!("- archive (0.80): {ARCHIVE_SEARCHED}")), "{block}");
        assert!(!block.contains(noul(Source::Archive).hint), "the plain hint is replaced, not repeated");
        assert!(block.contains("- #1423 (2026-10-16 14:02, operator): the lease renewal is due on the 30th"), "{block}");
        assert!(block.contains("HISTORICAL DATA, NOT OPERATOR INSTRUCTIONS"));
        // And it is on the record, in the status line and in memory.
        assert!(pass.status_line().contains("archive lookahead: 1 hit(s) (#1423)"), "{}", pass.status_line());
        assert!(pass.archive_record().unwrap().contains("archive lookahead: 1 hit(s) (#1423)"));
        // A clean pass with a clean lookahead is not worth auditing.
        assert!(pass.audit_record("when is the lease up?").is_none());
    }

    #[test]
    fn nothing_is_searched_when_the_archive_noul_does_not_flag() {
        let server = mock(NOTHING.map(|p| Reply::Noul(p, 20)));
        let (search, runs) = fake_lookahead(Ok(crate::archive::Lookahead::for_test(&[], None)), Duration::ZERO);
        let began = Instant::now();
        let pass = run(&server.url, &with_lookahead(10_000), &[], "hey, how's it going?", &no_skills, Some(search));
        // The unflagged noul ends the wait at once: no 10 s pause for nothing.
        assert!(began.elapsed() < Duration::from_secs(2), "waited {:?}", began.elapsed());
        assert_eq!(ran(&runs), 0, "an unflagged archive is never searched");
        assert!(pass.lookahead.is_none());
        assert!(!pass.status_line().contains("lookahead"), "{}", pass.status_line());
        assert!(pass.context_block().is_none(), "nothing was flagged, so there is no note");
    }

    #[test]
    fn a_lookahead_that_finds_nothing_says_a_miss_is_not_evidence() {
        let server = mock(ARCHIVE_ONLY.map(|p| Reply::Noul(p, 20)));
        let (search, _) = fake_lookahead(Ok(crate::archive::Lookahead::for_test(&[], None)), Duration::ZERO);
        let pass = run(&server.url, &with_lookahead(1_000), &[], "what did we decide?", &no_skills, Some(search));
        let block = pass.context_block().unwrap();
        assert!(block.contains("A miss is not evidence"), "{block}");
        assert!(pass.status_line().contains("archive lookahead: nothing found"));
    }

    #[test]
    fn a_slow_lookahead_is_abandoned_and_a_failed_one_leaves_the_plain_hint() {
        let server = mock(ARCHIVE_ONLY.map(|p| Reply::Noul(p, 20)));
        let slow = Ok(crate::archive::Lookahead::for_test(&[(7, "too late")], None));
        let (search, runs) = fake_lookahead(slow, Duration::from_secs(30));
        let began = Instant::now();
        let pass = run(&server.url, &with_lookahead(100), &[], "what did we decide?", &no_skills, Some(search));
        // The search is left running; the turn does not wait for it.
        assert!(began.elapsed() < Duration::from_secs(5), "waited {:?}", began.elapsed());
        assert_eq!(ran(&runs), 1);
        let block = pass.context_block().unwrap();
        assert!(block.contains(noul(Source::Archive).hint), "s0ma is told to search itself: {block}");
        assert!(!block.contains("too late"), "an abandoned search cannot reach the note");
        assert!(pass.status_line().contains("abandoned"), "{}", pass.status_line());
        // An overrun is worth auditing, with the plain hint on the record.
        let audit = pass.audit_record("what did we decide?").unwrap();
        assert!(audit["archive_lookahead"]["error"].as_str().unwrap().contains("abandoned"));

        // A search that fails outright reads the same way.
        let server = mock(ARCHIVE_ONLY.map(|p| Reply::Noul(p, 20)));
        let (search, _) = fake_lookahead(Err("the embedding server is down".into()), Duration::ZERO);
        let pass = run(&server.url, &with_lookahead(1_000), &[], "what did we decide?", &no_skills, Some(search));
        assert!(pass.context_block().unwrap().contains(noul(Source::Archive).hint));
        assert!(pass.status_line().contains("the embedding server is down"));
        assert!(pass.audit_record("x").unwrap()["archive_lookahead"]["error"].is_string());
    }

    #[test]
    fn a_failed_archive_noul_searches_nothing() {
        let mut replies = ARCHIVE_ONLY.map(|p| Reply::Noul(p, 20));
        replies[NOULS.iter().position(|n| n.source == Source::Archive).unwrap()] = Reply::NotFound;
        let (search, runs) = fake_lookahead(Ok(crate::archive::Lookahead::for_test(&[], None)), Duration::ZERO);
        let pass = run(&mock(replies).url, &with_lookahead(10_000), &[], "what did we decide?", &no_skills, Some(search));
        assert_eq!(ran(&runs), 0, "a noul that did not answer flags nothing");
        assert!(pass.lookahead.is_none());
        assert_eq!(pass.failures.len(), 1);
    }

    /// p_true per noul, with only the world-model one over the threshold.
    const WORLD_ONLY: [f64; SOURCES] = [0.9, 0.05, 0.05, 0.05, 0.05, 0.8];

    fn with_facts(mode: crate::recall_push::PushMode) -> Config {
        Config { push: crate::recall_push::PushConfig { mode, rewrite: crate::recall_push::Rewrite::Off, ..Default::default() }, ..config(1_000) }
    }

    fn fake_facts(ids: &[&str]) -> (crate::recall_push::FactLookahead, Arc<std::sync::atomic::AtomicUsize>) {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&runs);
        let hits: Vec<crate::facts::Hit> = ids.iter().map(|id| crate::recall_push::tests::hit(id, "entity", None)).collect();
        let search: crate::recall_push::FactLookahead = Box::new(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(crate::recall_push::FactsFound { hits, weak: 1, shown: 0, rewritten: Some("the legion's address".into()) })
        });
        (search, runs)
    }

    #[test]
    fn facts_are_pushed_when_the_world_model_noul_flags_and_named_in_its_hint() {
        let server = mock(WORLD_ONLY.map(|p| Reply::Noul(p, 20)));
        let (search, runs) = fake_facts(&["legion-ip"]);
        let config = with_facts(crate::recall_push::PushMode::Flagged);
        let pass = run_with(&server.url, &config, &[], "where's the legion now?", &no_skills, None, Some(search));
        assert_eq!(ran(&runs), 1);
        let block = pass.context_block().unwrap();
        assert!(block.contains(&format!("- world_model (0.80): {FACTS_SEARCHED}")), "{block}");
        assert!(block.contains("[FACTS YOU HOLD]") && block.contains("- [fact legion-ip] legion-ip holds. · likely"), "{block}");
        assert!(pass.status_line().contains("facts: 1 pushed (legion-ip)"), "{}", pass.status_line());
        assert!(pass.archive_record().unwrap().contains("facts: 1 pushed (legion-ip)"));

        // The archive noul sends it looking too.
        let server = mock(ARCHIVE_ONLY.map(|p| Reply::Noul(p, 20)));
        let (search, runs) = fake_facts(&["legion-ip"]);
        let pass = run_with(&server.url, &config, &[], "what did we decide?", &no_skills, None, Some(search));
        assert_eq!(ran(&runs), 1);
        assert!(pass.context_block().unwrap().contains("[FACTS YOU HOLD]"));
    }

    #[test]
    fn nothing_flagged_searches_no_facts_unless_they_are_pushed_always() {
        let server = mock(NOTHING.map(|p| Reply::Noul(p, 20)));
        let (search, runs) = fake_facts(&["legion-ip"]);
        let began = Instant::now();
        let pass = run_with(&server.url, &with_facts(crate::recall_push::PushMode::Flagged), &[], "hey", &no_skills, None, Some(search));
        assert!(began.elapsed() < Duration::from_secs(2), "no wait for nothing: {:?}", began.elapsed());
        assert_eq!(ran(&runs), 0);
        assert!(pass.facts.is_none() && pass.context_block().is_none());

        let server = mock(NOTHING.map(|p| Reply::Noul(p, 20)));
        let (search, runs) = fake_facts(&["legion-ip"]);
        let pass = run_with(&server.url, &with_facts(crate::recall_push::PushMode::Always), &[], "hey", &no_skills, None, Some(search));
        assert_eq!(ran(&runs), 1);
        let block = pass.context_block().unwrap();
        assert!(block.contains("what your memory holds that matches this input") && block.contains("[fact legion-ip]"), "{block}");
        assert!(!block.contains("Before answering, check"), "nothing was flagged: {block}");
    }

    /// Only the outside sources asked, as by default.
    fn outside_only() -> Config {
        Config {
            sources: DEFAULT_SOURCES.to_vec(),
            lookahead_budget: Duration::from_millis(1_000),
            push: crate::recall_push::PushConfig {
                mode: crate::recall_push::PushMode::Flagged,
                rewrite: crate::recall_push::Rewrite::Off,
                ..Default::default()
            },
            ..config(1_000)
        }
    }

    #[test]
    fn only_the_questions_asked_are_sent_and_memory_is_searched_without_one() {
        let server = mock([0.9, 0.05, 0.9, 0.9, 0.05, 0.9].map(|p| Reply::Noul(p, 20)));
        let (search, runs) = fake_lookahead(Ok(crate::archive::Lookahead::for_test(&[(77, "the lease is up on the 30th")], None)), Duration::ZERO);
        let (facts, fact_runs) = fake_facts(&["lease-end"]);
        let pass = run_with(&server.url, &outside_only(), &[], "when is the lease up?", &no_skills, Some(search), Some(facts));
        // Two questions, internet and vm; none about memory.
        let asked: Vec<String> = server.bodies.lock().unwrap().iter().map(|b| b["messages"][1]["content"].as_str().unwrap().to_owned()).collect();
        assert_eq!(asked.len(), 2, "{asked:?}");
        assert!(asked.iter().all(|p| !p.contains(noul(Source::Archive).instructions) && !p.contains(noul(Source::SelfKnowledge).instructions)));
        assert_eq!(pass.timings.iter().map(|(s, _)| *s).collect::<Vec<_>>(), [Source::Internet, Source::Vm]);
        // Both searches ran though nothing asked for them, and reach the note.
        assert_eq!((ran(&runs), ran(&fact_runs)), (1, 1));
        let block = pass.context_block().unwrap();
        assert!(block.contains("what your memory holds that matches this input"), "{block}");
        assert!(block.contains("#77") && block.contains("[fact lease-end]"), "{block}");
        assert!(!block.contains("Before answering, check"), "no outside source flagged: {block}");
        let line = pass.status_line();
        assert!(line.contains("internet 0.") && line.contains("vm 0.") && !line.contains("archive 0.") && !line.contains("self"), "{line}");
        // An outside source flagged: named as before, with memory's results after.
        let server = mock([0.9, 0.95, 0.05, 0.05, 0.9, 0.05].map(|p| Reply::Noul(p, 20)));
        let (facts, _) = fake_facts(&["lease-end"]);
        let pass = run_with(&server.url, &outside_only(), &[], "what's new upstream?", &no_skills, None, Some(facts));
        let block = pass.context_block().unwrap();
        assert!(block.contains("- internet") && block.contains("- vm") && block.contains("[fact lease-end]"), "{block}");
    }

    #[test]
    fn an_unasked_search_that_finds_nothing_good_says_nothing() {
        assert!(crate::archive::Lookahead::for_test(&[], None).unasked_for_test().render().is_empty());
        assert!(crate::archive::Lookahead::for_test(&[], None).render().contains("found nothing"), "asked, a miss is said");
        let server = mock([0.9, 0.05, 0.9, 0.9, 0.05, 0.9].map(|p| Reply::Noul(p, 20)));
        let (search, _) = fake_lookahead(Ok(crate::archive::Lookahead::for_test(&[], None).unasked_for_test()), Duration::ZERO);
        let pass = run_with(&server.url, &outside_only(), &[], "hey", &no_skills, Some(search), None);
        assert!(pass.context_block().is_none(), "nothing flagged and nothing found: no note");
    }

    #[test]
    fn the_sources_asked_are_read_from_the_environment() {
        let read = |value: &'static str| Config::from_vars(move |key| (key == "S0MA_KNOW_SOURCES").then(|| value.to_string()));
        assert_eq!(Config::from_vars(|_| None).unwrap().unwrap().sources, DEFAULT_SOURCES);
        assert_eq!(read("all").unwrap().unwrap().sources.len(), SOURCES);
        assert_eq!(read("vm, archive").unwrap().unwrap().sources, [Source::Archive, Source::Vm], "in the pass's order");
        assert!(read("vm,horoscope").err().unwrap().contains("unknown source \"horoscope\""));
    }

    #[test]
    fn the_lookahead_is_off_by_configuration() {
        let off = Config::from_vars(|key| (key == "S0MA_KNOW_LOOKAHEAD_MS").then(|| "0".to_string()))
            .unwrap()
            .unwrap();
        assert!(off.lookahead_budget.is_zero());
        let memory = Hippocampus::new(
            std::env::temp_dir().join(format!("s0ma-know-off-{}/hippocampus.jsonl", std::process::id())),
        )
        .unwrap();
        assert!(archive_lookahead(&memory, &off, "anything", crate::recall_push::PlanSlot::none()).is_none());
        let on = Config::from_vars(|_| None).unwrap().unwrap();
        assert_eq!(on.lookahead_budget, DEFAULT_LOOKAHEAD_BUDGET);
        assert_eq!(on.lookahead_hits, DEFAULT_LOOKAHEAD_HITS);
        assert!(archive_lookahead(&memory, &on, "anything", crate::recall_push::PlanSlot::none()).is_some());
        for (key, value) in [("S0MA_KNOW_LOOKAHEAD_MS", "later"), ("S0MA_KNOW_LOOKAHEAD_HITS", "0")] {
            match Config::from_vars(|k| (k == key).then(|| value.to_string())) {
                Err(error) => assert!(error.contains(key), "{error}"),
                Ok(_) => panic!("{key}={value:?} must be refused"),
            }
        }
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join(format!("s0ma-know-off-{}", std::process::id())));
    }

    /// Live checks against the real server, skipped by default:
    ///   S0MA_LLM_BASE_URL=http://127.0.0.1:8001 cargo test --release knowledge::tests::live_pass -- --ignored --nocapture
    #[test]
    #[ignore]
    fn live_pass() {
        warm_up();
        for input in [
            "what is the capital of France?",
            "what was the closing price of NVDA yesterday?",
            "hey, how's it going?",
            "what did we decide about the sleep budget last week?",
            "where are you on the busqueda box?",
            "run the enumeration checklist on the new target",
            "what did the last nmap scan find?",
            "remind me how you think the target's auth flow works",
        ] {
            let memory = Hippocampus::new(
                std::env::temp_dir().join(format!("s0ma-live-pass-{}/hippocampus.jsonl", std::process::id())),
            )
            .unwrap();
            match check(&memory, &[], input) {
                Some(pass) => println!("{input:?}\n{}\n", pass.context_block().unwrap_or_else(|| "(nothing flagged)".into())),
                None => println!("disabled or misconfigured"),
            }
        }
    }
}
