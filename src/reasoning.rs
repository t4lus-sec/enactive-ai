use crate::{
    llm::{self, ChatOptions},
    types::Message,
};

const ROUTER_PROMPT: &str = r#"
You are selecting the reasoning depth for s0ma's next response.

Return exactly one of these words:

instruct
low
medium
xhigh

Choose the least expensive profile that is adequate.

INSTRUCT:
Use when deliberate internal reasoning is not materially useful.
Examples include ordinary conversation, banter, simple factual questions,
straightforward explanations, brief reactions, and simple commands.

LOW:
Use for light reasoning.
Examples include routine technical questions, small coding issues,
simple debugging, or tasks requiring a few connected reasoning steps.

MEDIUM:
Use for substantial multi-step reasoning.
Examples include architecture decisions, nontrivial debugging, technical
analysis, ambiguous evidence, philosophical analysis, or problems where
careful comparison materially improves the answer.

XHIGH:
Reserve for unusually difficult problems where extensive reasoning is
likely to materially improve correctness.
Examples include deeply conflicting evidence, difficult synthesis,
complex investigations, or major decisions with many interacting constraints.

Do not select a deeper mode merely because the topic is technical,
security-related, philosophical, or important.

Judge the reasoning actually required by the current turn.

Return only the profile name. No punctuation or explanation.
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningProfile {
    Instruct,
    Low,
    Medium,
    XHigh,
}

impl ReasoningProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "instruct" => Some(Self::Instruct),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "xhigh" => Some(Self::XHigh),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Instruct => "instruct",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::XHigh => "xhigh",
        }
    }

    pub fn options(self) -> ChatOptions {
        match self {
            Self::Instruct => ChatOptions::instruct(),
            Self::Low => ChatOptions::low(),
            Self::Medium => ChatOptions::medium(),
            Self::XHigh => ChatOptions::xhigh(),
        }
    }
}

// ------------------------------------------------------------------ subagents

/// How hard subagents think: one of s0ma's own profiles (the same sampling
/// as s0ma's turns at that level), set with `/mode workers <level>` or in the
/// overview, and kept across restarts in `data/worker_reasoning`. Medium
/// until set: thinking, without spending a small worker's budget the way
/// xhigh would.
static WORKER_PROFILE: std::sync::Mutex<Option<ReasoningProfile>> = std::sync::Mutex::new(None);
static WORKER_FILE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
pub const WORKER_DEFAULT: ReasoningProfile = ReasoningProfile::Medium;

/// At start: the level kept from before, if any.
pub fn init_workers(data_dir: &std::path::Path) {
    let file = data_dir.join("worker_reasoning");
    if let Some(profile) = std::fs::read_to_string(&file).ok().and_then(|text| ReasoningProfile::parse(&text)) {
        *WORKER_PROFILE.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(profile);
    }
    let _ = WORKER_FILE.set(file);
}

/// The subagents' level now.
pub fn worker_profile() -> ReasoningProfile {
    WORKER_PROFILE.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).unwrap_or(WORKER_DEFAULT)
}

fn set_worker_profile(profile: ReasoningProfile) {
    *WORKER_PROFILE.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(profile);
    if let Some(file) = WORKER_FILE.get() {
        let _ = std::fs::write(file, profile.label());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningPolicy {
    Auto,
    Forced(ReasoningProfile),
}

pub struct ReasoningState {
    policy: ReasoningPolicy,
}

impl ReasoningState {
    pub fn new() -> Self {
        Self {
            policy: ReasoningPolicy::Forced(ReasoningProfile::XHigh),
        }
    }

    pub fn handle_mode_command(&mut self, input: &str) -> Option<Result<String, String>> {
        let mut parts = input.split_whitespace();

        if parts.next()? != "/mode" {
            return None;
        }

        let argument = parts.next();

        // `/mode workers [level]`: the subagents' level (new workers take it).
        if argument == Some("workers") {
            let level = parts.next();
            if parts.next().is_some() {
                return Some(Err("usage: /mode workers [instruct|low|medium|xhigh]".to_string()));
            }
            return Some(match level {
                None => Ok(format!("[mode] workers {}", worker_profile().label())),
                Some(value) => match ReasoningProfile::parse(value) {
                    Some(profile) => {
                        set_worker_profile(profile);
                        Ok(format!("[mode] workers {} (subagents started from now on)", profile.label()))
                    }
                    None => Err("usage: /mode workers [instruct|low|medium|xhigh]".to_string()),
                },
            });
        }

        if parts.next().is_some() {
            return Some(Err(
                "usage: /mode [auto|instruct|low|medium|xhigh] · /mode workers [instruct|low|medium|xhigh]".to_string()
            ));
        }

        let result = match argument {
            None => Ok(self.status()),

            Some("auto") => {
                self.policy = ReasoningPolicy::Auto;
                Ok("[mode] auto".to_string())
            }

            Some(value) => {
                let Some(profile) = ReasoningProfile::parse(value) else {
                    return Some(Err(
                        "usage: /mode [auto|instruct|low|medium|xhigh]".to_string()
                    ));
                };

                self.policy = ReasoningPolicy::Forced(profile);

                Ok(format!("[mode] forced {}", profile.label()))
            }
        };

        Some(result)
    }

    pub fn policy(&self) -> ReasoningPolicy {
        self.policy
    }

    pub fn status(&self) -> String {
        match self.policy {
            ReasoningPolicy::Auto => "[mode] auto".to_string(),

            ReasoningPolicy::Forced(profile) => {
                format!("[mode] forced {}", profile.label())
            }
        }
    }
}

pub fn choose_profile(
    state: &ReasoningState,
    conversation: &[Message],
    input: &str,
) -> Result<ReasoningProfile, Box<dyn std::error::Error>> {
    match state.policy() {
        ReasoningPolicy::Forced(profile) => Ok(profile),

        ReasoningPolicy::Auto => route(conversation, input),
    }
}

fn route(
    conversation: &[Message],
    input: &str,
) -> Result<ReasoningProfile, Box<dyn std::error::Error>> {
    let previous_response = previous_s0ma_response(conversation);

    let mut routing_input = String::new();

    if let Some(previous) = previous_response {
        routing_input.push_str("Previous s0ma response:\n");

        routing_input.push_str(&truncate_chars(previous, 4000));

        routing_input.push_str("\n\n");
    }

    routing_input.push_str("Current input:\n");
    routing_input.push_str(input);

    let messages = vec![
        Message::system(ROUTER_PROMPT.to_string()),
        Message::user(routing_input),
    ];

    let options = ChatOptions::router();

    let response = llm::chat_quiet(&messages, &options)?;

    let output = response.content.trim();

    ReasoningProfile::parse(output)
        .ok_or_else(|| format!("router returned invalid profile: {output:?}").into())
}

fn previous_s0ma_response(conversation: &[Message]) -> Option<&str> {
    conversation
        .iter()
        .rev()
        .find(|message| message.role == "assistant" && !message.content.trim().is_empty())
        .map(|message| message.content.as_str())
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    let mut chars = input.chars();

    let truncated: String = chars.by_ref().take(max_chars).collect();

    if chars.next().is_some() {
        format!("{truncated}\n[previous response truncated]")
    } else {
        truncated
    }
}

#[cfg(test)]
mod default_tests {
    #[test]
    fn startup_forces_xhigh_until_operator_changes_mode() {
        assert_eq!(
            super::ReasoningState::new().policy,
            super::ReasoningPolicy::Forced(super::ReasoningProfile::XHigh)
        );
    }
}

#[cfg(test)]
mod worker_tests {
    use super::*;

    #[test]
    fn the_subagents_level_is_set_kept_and_used() {
        let dir = std::env::temp_dir().join(format!("s0ma-worker-reasoning-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        init_workers(&dir);
        let mut state = ReasoningState::new();
        assert!(state.handle_mode_command("/mode workers").unwrap().unwrap().starts_with("[mode] workers "));
        assert!(state.handle_mode_command("/mode workers sideways").unwrap().is_err());
        let said = state.handle_mode_command("/mode workers low").unwrap().unwrap();
        assert!(said.contains("workers low"), "{said}");
        assert_eq!(worker_profile(), ReasoningProfile::Low);
        assert_eq!(state.policy(), ReasoningPolicy::Forced(ReasoningProfile::XHigh), "s0ma's own is untouched");
        // s0ma's sampling at that level: thinking on, no penalties.
        let options = worker_profile().options();
        assert_eq!((options.reasoning_effort, options.temperature, options.presence_penalty), ("low", 1.0, 0.0));
        // Kept for the next start (where WORKER_FILE points, in this run).
        if WORKER_FILE.get() == Some(&dir.join("worker_reasoning")) {
            assert_eq!(std::fs::read_to_string(dir.join("worker_reasoning")).unwrap(), "low");
        }
        state.handle_mode_command("/mode workers medium").unwrap().unwrap();
        assert_eq!(worker_profile(), WORKER_DEFAULT);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
