#[path = "tool_output.rs"]
mod tool_output;
pub(crate) use tool_output::{
    is_stub, limit_tool_output, limit_worker_output, locate_in_archive, page_stub, reread_stub, result_matches,
    result_page, source_truncation, worker_stub, WORKER_VISIBLE_CHARS,
};
#[cfg(test)]
pub(crate) use tool_output::limit_worker_output_offline;
#[path = "background_status.rs"]
mod background;
pub(crate) use background::{background_log, background_scope, background_status, configure_background_log, report_background_status, sleep_status, status};
macro_rules! diagnostic { ($($args:tt)*) => { crate::llm::status(format_args!($($args)*)) }; }
use crate::conversations::Conversation;
use crate::types::{FunctionCall, Message, ToolCall};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::RefCell;
use std::io::{BufRead, BufReader, Read};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Generation requests in flight right now: the main turn, workers, sleep and
/// the knowledge probes, from the moment one is sent until its reply has been
/// read to the end. Background embedding waits while there are any
/// (`archive_semantic::yield_to_generation`), so the embedder doesn't take GPU or CPU
/// time from prefill and decoding.
static GENERATING: AtomicUsize = AtomicUsize::new(0);

/// Counts one generation request for as long as it lives: made before the
/// request is sent, kept until its reply is fully read. However the request
/// ends (an answer, an error, a panic), dropping it ends the count.
pub(crate) struct Generating(&'static AtomicUsize);

impl Generating {
    pub(crate) fn start() -> Self {
        Self::on(&GENERATING)
    }

    fn on(counter: &'static AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self(counter)
    }
}

impl Drop for Generating {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Whether any generation request is in flight (see `GENERATING`).
pub(crate) fn generating() -> bool {
    GENERATING.load(Ordering::SeqCst) > 0
}

// Model failures remain distinguishable after crossing string-returning workers.
const FATAL_MODEL: &str = "[fatal-model]";
const RETRY_EXHAUSTED: &str = "[retry-exhausted]";
const MODEL_RESPONSE: &str = "[model-response]";
pub fn is_fatal_message(message: &str) -> bool {
    message.starts_with(FATAL_MODEL)
}
pub fn is_fatal_error(error: &(dyn std::error::Error + 'static)) -> bool {
    is_fatal_message(&error.to_string())
}
pub fn fatal_model(message: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    format!("{FATAL_MODEL} {message}").into()
}
fn fatal_http(status: reqwest::StatusCode, detail: &str) -> bool {
    let detail = detail.to_ascii_lowercase();
    status.as_u16() == 413
        || (status.is_client_error() && !matches!(status.as_u16(), 408 | 409 | 425 | 429))
        || status.as_u16() == 501
        || detail.contains("out of memory")
        || detail.contains("cuda_error_out_of_memory")
        || detail.contains("exceeds the available context")
        || detail.contains("maximum context length")
        || detail.contains("context size exceeded")
}
/// Retry read-only setup probes; never use this around a tool or user turn.
pub fn retry_read<T, F>(label: &str, mut operation: F) -> Result<T, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<T, Box<dyn std::error::Error>>,
{
    for attempt in 0..3 {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) => {
                let retryable = error.downcast_ref::<reqwest::Error>().is_some_and(|e| {
                    e.is_connect()
                        || e.is_timeout()
                        || e.status().is_some_and(|s| {
                            s.is_server_error() || matches!(s.as_u16(), 408 | 409 | 425 | 429)
                        })
                });
                if !retryable || attempt == 2 {
                    return Err(error);
                }
                diagnostic!(
                    "[{label}] retrying transient read failure ({}/3): {error}",
                    attempt + 2
                );
                std::thread::sleep(Duration::from_secs(attempt + 1));
            }
        }
    }
    unreachable!()
}

fn send_model_request(
    request: reqwest::blocking::RequestBuilder,
) -> Result<reqwest::blocking::Response, Box<dyn std::error::Error>> {
    let mut last = String::new();
    for attempt in 0..3 {
        if attempt > 0 {
            diagnostic!("[llm] retry {}/3 after: {}", attempt + 1, last);
            std::thread::sleep(Duration::from_secs(attempt));
        }
        let request = request
            .try_clone()
            .ok_or_else(|| fatal_model("model request body cannot be retried"))?;
        match request.send() {
            Ok(response) if response.status().is_success() => return Ok(response),
            Ok(response) => {
                let status = response.status();
                let detail = response
                    .text()
                    .unwrap_or_else(|e| format!("could not read error body: {e}"));
                let detail: String = detail.chars().take(4000).collect();
                if fatal_http(status, &detail) {
                    return Err(fatal_model(format!("HTTP {status}: {detail}")));
                }
                last = format!("HTTP {status}: {detail}");
            }
            Err(error) if error.is_builder() => return Err(fatal_model(error)),
            Err(error) => last = error.to_string(),
        }
    }
    Err(format!("{RETRY_EXHAUSTED} model transport: {last}").into())
}
/// For tool-free generation/validation only. Never replay side-effecting tools.
/// The operation gets the exact prior error, so malformed output can be repaired.
pub(crate) fn retry_text<F>(label: &str, mut operation: F) -> Result<String, String>
where
    F: FnMut(Option<&str>) -> Result<String, String>,
{
    let mut last: Option<String> = None;
    for attempt in 0..3 {
        match operation(last.as_deref()) {
            Ok(value) => return Ok(value),
            Err(error) if is_fatal_message(&error) || error.starts_with(RETRY_EXHAUSTED) => {
                return Err(error);
            }
            Err(error) => {
                diagnostic!("[{label}] attempt {}/3 failed: {error}", attempt + 1);
                last = Some(error);
            }
        }
    }
    Err(format!(
        "{RETRY_EXHAUSTED} {label}: {}",
        last.unwrap_or_default()
    ))
}

// ---- vLLM endpoint ----
//
// The backend is vLLM's OpenAI-compatible server. Both values can be overridden
// without a rebuild; the defaults match the single-card NVFP4 recipe.
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8001";
const DEFAULT_MODEL: &str = "qwen38-nvfp4";

/// Logical request lanes. Lane 0 is the main conversation and sleep integration;
/// 1..=WORKER_LANES are subagents, sleep batch workers and compaction leases.
///
/// vLLM has no slots. It batches every request continuously over one shared,
/// paged KV pool, so lanes exist only for s0ma's own admission control. Sized to
/// the server's `--max-num-seqs` (16 in the recipe) minus the main lane: more
/// lanes than that would only queue inside vLLM.
pub const WORKER_LANES: usize = 15;

/// vLLM priority scheduling: lower values are scheduled first and can preempt
/// higher ones when the KV pool runs short. Opt-in via S0MA_LLM_PRIORITY=1,
/// because a server started without `--scheduling-policy priority` rejects any
/// non-zero priority with a 400, which s0ma would treat as fatal; sent anyway
/// to ds41d, which says it takes it (and keeps prompts by it). Model calls
/// take theirs from their conversation (`conversations.rs`); these are for
/// the knowledge probes, which build their own requests.
const MAIN_PRIORITY: i32 = crate::conversations::MAIN_PRIORITY;
const WORKER_PRIORITY: i32 = crate::conversations::WORKER_PRIORITY;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_owned())
}

pub(crate) fn endpoint(path: &str) -> String {
    format!("{}{path}", env_or("S0MA_LLM_BASE_URL", DEFAULT_BASE_URL).trim_end_matches('/'))
}

pub(crate) fn model_name() -> String {
    env_or("S0MA_LLM_MODEL", DEFAULT_MODEL)
}

pub(crate) fn lane_priority(lane: usize) -> Option<i32> {
    crate::conversations::priority_enabled().then_some(if lane == 0 { MAIN_PRIORITY } else { WORKER_PRIORITY })
}

fn check_lane(lane: usize) -> Result<(), Box<dyn std::error::Error>> {
    if lane > WORKER_LANES {
        return Err(format!("invalid request lane {lane}; lanes are 0..={WORKER_LANES}").into());
    }
    Ok(())
}

/// vLLM renders the chat template server-side and counts it, which is what the
/// llama.cpp input_tokens endpoint used to do. The response also carries the
/// token ids themselves; serde skips them.
/// Guarantee no unparseable tool-call arguments reach the server.
///
/// Every request goes through here, so this covers messages however they got
/// into the list — freshly generated, restored from a sleep journal, or carried
/// across a resume. That matters because the damage outlives the generation that
/// caused it: a call truncated mid-arguments gets persisted, and the server
/// parses every `arguments` string it is handed, so each later request fails with
/// a 400 describing the fragment. Repairing only at generation time leaves any
/// journal already on disk permanently poisoned.
///
/// Borrows unless something actually needs fixing, which is almost always.
fn sanitized(messages: &[Message]) -> std::borrow::Cow<'_, [Message]> {
    let broken = messages.iter().any(|message| {
        message.tool_calls.as_ref().is_some_and(|calls| {
            calls.iter().any(|call| {
                let arguments = call.function.arguments.trim();
                !arguments.is_empty()
                    && serde_json::from_str::<Value>(arguments).is_err()
            })
        })
    });

    if !broken {
        return std::borrow::Cow::Borrowed(messages);
    }

    let mut repaired = messages.to_vec();
    let mut fixed = Vec::new();
    for message in &mut repaired {
        if let Some(calls) = message.tool_calls.as_mut() {
            fixed.extend(crate::types::repair_tool_call_arguments(calls));
        }
    }
    // Only this copy is repaired: the calls stay broken in history and come
    // back with every request (and every context measurement) until a fold
    // takes them. Saying so once per call is enough.
    let new = first_reports(&fixed);
    if new > 0 {
        crate::terminal::warn_status(&format!("[llm] repaired {new} truncated tool-call argument(s) before sending"));
    }
    std::borrow::Cow::Owned(repaired)
}

/// How many of these repaired calls haven't been reported yet (and now are).
fn first_reports(ids: &[String]) -> usize {
    static REPORTED: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
    let mut reported = REPORTED.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let reported = reported.get_or_insert_with(Default::default);
    ids.iter().filter(|id| reported.insert((*id).clone())).count()
}

#[derive(Debug, Serialize)]
struct TokenizeRequest<'a> {
    model: String,
    messages: std::borrow::Cow<'a, [Message]>,
    add_generation_prompt: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDefinition]>,
    chat_template_kwargs: TemplateOptions<'a>,
}

#[derive(Debug, Deserialize)]
struct TokenizeResponse {
    count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,

    pub function: ToolFunctionDefinition,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolFunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            kind: "function".to_string(),

            function: ToolFunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

// No cache_prompt or id_slot: vLLM's automatic prefix caching shares any common
// prefix across requests, so pinning a conversation to a slot has nothing to buy.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
    messages: std::borrow::Cow<'a, [Message]>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    presence_penalty: f32,
    // vLLM applies this over the ENTIRE output so far, reasoning included, with
    // no window. llama.cpp applied it over the last 64 tokens only. Over a long
    // thinking trace the unwindowed form compounds on structural tokens (`"`,
    // `,`, `}`) until they become unsampleable, which broke tool-call JSON and
    // produced runaway text that could not close its string. Keep it at 0 on
    // vLLM; the field stays so a windowed logits processor can use it later.
    frequency_penalty: f32,
    stream: bool,
    /// Token counts in the stream, for the live tok/s (see `stream_options`).
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    chat_template_kwargs: TemplateOptions<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    priority: Option<i32>,
    /// Which conversation it is, for ds41d's kept prompts
    /// (`conversations.rs`); only to an engine that takes it.
    #[serde(skip_serializing_if = "Option::is_none")]
    conversation_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDefinition]>,

    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a str>,
}

/// Asks the server to count generated tokens as it streams: `include_usage`
/// (OpenAI's) adds a final chunk with the totals, and `continuous_usage_stats`
/// (vLLM's) puts the running count on every chunk. A server that knows neither
/// ignores them, and the tok/s falls back to an estimate from characters.
#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
    continuous_usage_stats: bool,
}

/// `S0MA_STREAM_USAGE=off` sends no `stream_options`, for a server that
/// rejects them.
fn stream_options(stream: bool) -> Option<StreamOptions> {
    stream_options_for(stream, std::env::var("S0MA_STREAM_USAGE").ok().as_deref())
}

fn stream_options_for(stream: bool, setting: Option<&str>) -> Option<StreamOptions> {
    let off = setting.is_some_and(|value| value.trim().eq_ignore_ascii_case("off"));
    (stream && !off).then_some(StreamOptions { include_usage: true, continuous_usage_stats: true })
}

#[derive(Debug, Serialize)]
struct TemplateOptions<'a> {
    enable_thinking: bool,
    preserve_thinking: bool,
    /// Only for /tokenize, which has no top-level reasoning_effort field. Chat
    /// requests send it top-level, the way the model card does.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
}

/// The model card lists xhigh, medium and low; s0ma's instruct and router modes
/// use "none", which vLLM would reject as an effort level. Express it the way the
/// card does instead: thinking disabled, no effort sent.
fn thinking(options: &ChatOptions) -> (bool, Option<&'static str>) {
    match options.reasoning_effort {
        "none" => (false, None),
        effort => (true, Some(effort)),
    }
}

fn chat_request<'a>(
    messages: &'a [Message],
    options: &ChatOptions,
    tools: Option<&'a [ToolDefinition]>,
    conversation: &Conversation,
    stream: bool,
    response_format: Option<Value>,
) -> ChatRequest<'a> {
    let (enable_thinking, reasoning_effort) = thinking(options);
    let (conversation_id, priority) = crate::conversations::request_fields(conversation);
    // An empty tool list is sent as no tools rather than `[]`.
    let tools = tools.filter(|tools| !tools.is_empty());

    ChatRequest {
        model: model_name(),
        response_format,
        messages: sanitized(messages),
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        presence_penalty: options.presence_penalty,
        frequency_penalty: options.frequency_penalty,
        stream,
        stream_options: stream_options(stream),
        reasoning_effort,
        chat_template_kwargs: TemplateOptions {
            enable_thinking,
            preserve_thinking: true,
            reasoning_effort: None,
        },
        priority,
        conversation_id,
        tool_choice: tools.map(|_| "auto"),
        tools,
    }
}

/// Where ds41d found the part of a prompt it didn't prefill again
/// (`usage.prompt_tokens_details.cache_source`): nowhere, the request's own
/// live slot, a kept prompt in RAM, or one on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheSource {
    None,
    Live,
    Ram,
    Disk,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,

    /// Present when `stream_options` asked for it: on every chunk with
    /// vLLM's continuous stats, and on a final chunk with no choices. Read
    /// loosely: a usage block of an unexpected shape is ignored, never an error.
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Delta,

    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    // vLLM has emitted the reasoning channel as `reasoning_content` and, in newer
    // releases, as `reasoning`. Separate fields rather than a serde alias, because
    // a release that sends both would make an alias fail as a duplicate field.
    #[serde(default)]
    reasoning_content: Option<String>,

    #[serde(default)]
    reasoning: Option<String>,

    #[serde(default)]
    content: Option<String>,

    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: usize,

    #[serde(default)]
    id: Option<String>,

    #[serde(rename = "type", default)]
    kind: Option<String>,

    #[serde(default)]
    function: Option<FunctionCallDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionCallDelta {
    #[serde(default)]
    name: Option<String>,

    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct ToolCallBuilder {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}

#[derive(Debug)]
pub enum StreamEvent {
    /// A generation has begun on the wire, before any of its output.
    ///
    /// One logical model call can stream more than once: `sample_until_answered`
    /// retries a malformed stream and resamples an empty answer, through the
    /// same callback. Each of those is a separate generation with its own
    /// prefill, so anything measuring a rate starts over here instead of
    /// folding two generations, the gap and the second prefill into one.
    Started,
    Reasoning(String),
    Content(String),
    /// A piece of a tool call as it is generated: the function name, then its
    /// arguments. Not shown; counted towards the tok/s, because the model
    /// generated it.
    ToolArguments(String),
    /// The server's count of tokens generated so far in this request. `last`:
    /// the closing report, after the final choice. `prompt_tokens` and
    /// `cached_tokens` (the part of the prompt its prefix cache already held,
    /// vLLM's `prompt_tokens_details.cached_tokens`) when the server sends them,
    /// for the prefill rate.
    Usage { completion_tokens: u64, prompt_tokens: Option<u64>, cached_tokens: Option<u64>, cache_source: Option<CacheSource>, last: bool },
    /// Nothing has arrived for `QUIET_POLL`: the server is still taking the
    /// request in or prefilling it, or pausing between tokens. Shows nothing.
    /// It is there so a caller can stop a request that has not started to
    /// stream: an error returned for it, as for any event, ends the call and
    /// drops the connection, and vLLM aborts a request whose client has gone,
    /// prefill included. Without it the first chance to stop came with the
    /// first token, after the whole prefill.
    Waiting,
}

#[derive(Debug)]
pub struct S0maResponse {
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ChatOptions {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub reasoning_effort: &'static str,
}

impl ChatOptions {
    pub fn instruct() -> Self {
        Self {
            max_tokens: 16384,
            temperature: 0.7,
            top_p: 0.80,
            top_k: 20,
            presence_penalty: 1.5,
            frequency_penalty: 0.0,
            reasoning_effort: "none",
        }
    }

    pub fn low() -> Self {
        Self {
            max_tokens: 16384,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            // Qwen's recommended thinking-mode settings: no presence or
            // frequency penalty. The 0.5 frequency penalty used on llama.cpp
            // was a 64-token-window loop breaker for the 27B; on vLLM it is
            // unwindowed and corrupts long outputs (see ChatRequest).
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            reasoning_effort: "low",
        }
    }

    pub fn medium() -> Self {
        Self {
            max_tokens: 16384,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            reasoning_effort: "medium",
        }
    }

    pub fn xhigh() -> Self {
        Self {
            max_tokens: 32768,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            reasoning_effort: "xhigh",
        }
    }

    pub fn router() -> Self {
        Self {
            max_tokens: 8,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            reasoning_effort: "none",
        }
    }

    pub fn normal() -> Self {
        Self::xhigh()
    }

    pub fn sleep() -> Self {
        let mut options = Self::xhigh();
        options.max_tokens = 32768;
        options
    }

    /// An integration pass before the last: `sleep()` at a lower reasoning
    /// effort, since such a pass reconciles one stretch of batches and the last
    /// pass, at xhigh, writes the checkpoint. medium by default, the step below
    /// xhigh on the model card; S0MA_SLEEP_PASS_EFFORT sets low, medium, high
    /// or xhigh.
    pub fn sleep_pass() -> Self {
        let mut options = Self::sleep();
        options.reasoning_effort = match std::env::var("S0MA_SLEEP_PASS_EFFORT")
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Ok("low") => "low",
            Ok("high") => "high",
            Ok("xhigh") => "xhigh",
            _ => "medium",
        };
        options
    }

}

pub fn count_input_tokens(
    messages: &[Message],
    options: &ChatOptions,
    tools: Option<&[ToolDefinition]>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(60))
        .build()?;

    let (enable_thinking, reasoning_effort) = thinking(options);

    let request = TokenizeRequest {
        model: model_name(),
        messages: sanitized(messages),
        add_generation_prompt: true,
        tools: tools.filter(|tools| !tools.is_empty()),
        chat_template_kwargs: TemplateOptions {
            enable_thinking,
            preserve_thinking: true,
            reasoning_effort,
        },
    };

    let response = send_model_request(client.post(endpoint("/tokenize")).json(&request))?
        .json::<TokenizeResponse>()?;

    Ok(response.count)
}

#[allow(clippy::too_many_arguments)]
fn chat_inner<F>(
    messages: &[Message],
    options: &ChatOptions,
    tools: Option<&[ToolDefinition]>,
    id_slot: usize,
    conversation: &Conversation,
    timeout: Option<Duration>,
    on_event: F,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    chat_config(
        messages, options, tools, id_slot, conversation, timeout, None, 3, on_event,
    )
}

#[allow(clippy::too_many_arguments)]
fn chat_config<F>(
    messages: &[Message],
    options: &ChatOptions,
    tools: Option<&[ToolDefinition]>,
    id_slot: usize,
    conversation: &Conversation,
    timeout: Option<Duration>,
    response_format: Option<Value>,
    attempts: usize,
    on_event: F,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    chat_config_capturing(messages, options, tools, id_slot, conversation, timeout, response_format, attempts, on_event, None, None)
}

/// `chat_config`, also handing back the exact bytes POSTed (`body`), for the
/// call store. Set before the first attempt is sent, so it is there however the
/// call ends.
#[allow(clippy::too_many_arguments)]
fn chat_config_capturing<F>(
    messages: &[Message],
    options: &ChatOptions,
    tools: Option<&[ToolDefinition]>,
    id_slot: usize,
    conversation: &Conversation,
    timeout: Option<Duration>,
    response_format: Option<Value>,
    attempts: usize,
    mut on_event: F,
    body_out: Option<&mut Option<Vec<u8>>>,
    base_url: Option<&str>,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    check_lane(id_slot)?;
    let request = chat_request(messages, options, tools, conversation, true, response_format);
    let url = match base_url {
        Some(base) => format!("{}/v1/chat/completions", base.trim_end_matches('/')),
        None => endpoint("/v1/chat/completions"),
    };
    // Serialised once: these bytes are what every attempt sends, and what the
    // call store keeps (the same serialisation `.json()` performed).
    let body = serde_json::to_vec(&request)?;
    if let Some(out) = body_out {
        *out = Some(body.clone());
    }

    let stream = Stream::new()?;
    // Counted as running until it returns: its conversation is not ended
    // under it, and one whose prompt broke off is ended before it.
    let _running = crate::conversations::begin_request(conversation, messages, tools.filter(|tools| !tools.is_empty()));
    sample_until_answered(attempts, options.temperature > 0.0, || {
        generate_once(&stream, &url, &body, timeout, &mut on_event)
    })
}

/// One generation attempt: send it, then read its stream to the end, counted
/// as generating throughout (prefill included), so background embedding
/// waits for all of it.
///
/// `quiet_limit`: how long to wait with nothing arriving, for the response
/// and then for each piece of it, before giving up (None: no limit). While it
/// waits, `on_event` gets `StreamEvent::Waiting` every `QUIET_POLL`, and an
/// error it returns stops the call there, prefill or not.
fn generate_once<F>(
    stream: &Stream,
    url: &str,
    body: &[u8],
    quiet_limit: Option<Duration>,
    on_event: &mut F,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    let _generating = Generating::start();
    // Two callers of one callback, never at the same time: the waits, and
    // `parse_stream` with what arrives.
    let on_event = RefCell::new(on_event);
    let mut waiting = || (*on_event.borrow_mut())(StreamEvent::Waiting);
    let response = stream.send(url, body, quiet_limit, &mut waiting)?;
    let stopped = RefCell::new(None);
    let reader = BufReader::new(Body {
        stream,
        response,
        chunk: Vec::new(),
        at: 0,
        quiet_limit,
        waiting: &mut waiting,
        stopped: &stopped,
    });
    let parsed = parse_stream(reader, |event| (*on_event.borrow_mut())(event));
    // Stopped by the caller mid-wait: its own error, not the read error
    // `parse_stream` made of it (which would be retried as a broken stream).
    match stopped.into_inner() {
        Some(error) => Err(error),
        None => parsed,
    }
}

/// How long a streamed call goes with nothing arriving before its caller is
/// asked whether it still wants it (`StreamEvent::Waiting`). Ctrl+C during a
/// prefill is felt within this.
const QUIET_POLL: Duration = Duration::from_millis(100);

/// The connection of one streamed model call: reqwest's async client on a
/// runtime of its own, driven only while this thread waits on it.
///
/// The blocking client could not be given up on. Its waits park the thread
/// until data or the timeout comes, and a prefill sends no data, so ctrl+C
/// was seen only at the first token. Here every wait is polled in
/// `QUIET_POLL` steps, and returning between them drops the request and its
/// connection, which is what makes vLLM abort the work.
struct Stream {
    runtime: tokio::runtime::Runtime,
    client: reqwest::Client,
}

type Waiter<'a> = dyn FnMut() -> Result<(), Box<dyn std::error::Error>> + 'a;

impl Stream {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            runtime: tokio::runtime::Builder::new_current_thread().enable_all().build()?,
            client: reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).build()?,
        })
    }

    /// `future`'s output, asking `waiting` whether to go on at every quiet
    /// poll. None: `limit` passed first. An error from `waiting` drops the
    /// future, and with it whatever it was doing.
    fn wait<T>(
        &self,
        future: impl std::future::Future<Output = T>,
        limit: Option<Duration>,
        waiting: &mut Waiter<'_>,
    ) -> Result<Option<T>, Box<dyn std::error::Error>> {
        let mut future = std::pin::pin!(future);
        let started = Instant::now();
        loop {
            // The timer is made inside the runtime, which it needs.
            let polled = self.runtime.block_on(async { tokio::time::timeout(QUIET_POLL, &mut future).await });
            if let Ok(value) = polled {
                return Ok(Some(value));
            }
            waiting()?;
            if limit.is_some_and(|limit| started.elapsed() >= limit) {
                return Ok(None);
            }
        }
    }

    /// `send_model_request` for a streamed call: the same retries and the
    /// same verdicts on a failed status, with every wait interruptible.
    fn send(
        &self,
        url: &str,
        body: &[u8],
        limit: Option<Duration>,
        waiting: &mut Waiter<'_>,
    ) -> Result<reqwest::Response, Box<dyn std::error::Error>> {
        let mut last = String::new();
        for attempt in 0..3u64 {
            if attempt > 0 {
                diagnostic!("[llm] retry {}/3 after: {}", attempt + 1, last);
                self.wait(async move { tokio::time::sleep(Duration::from_secs(attempt)).await }, None, waiting)?;
            }
            let request = self
                .client
                .post(url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body.to_vec());
            // Sent from inside the runtime, which whatever it sets up may need.
            match self.wait(async move { request.send().await }, limit, waiting)? {
                None => last = format!("no response within {:?}", limit.unwrap_or_default()),
                Some(Ok(response)) if response.status().is_success() => return Ok(response),
                Some(Ok(response)) => {
                    let status = response.status();
                    let detail = match self.wait(response.text(), Some(Duration::from_secs(30)), waiting)? {
                        Some(Ok(text)) => text,
                        Some(Err(error)) => format!("could not read error body: {error}"),
                        None => "could not read error body: it did not arrive".to_string(),
                    };
                    let detail: String = detail.chars().take(4000).collect();
                    if fatal_http(status, &detail) {
                        return Err(fatal_model(format!("HTTP {status}: {detail}")));
                    }
                    last = format!("HTTP {status}: {detail}");
                }
                Some(Err(error)) if error.is_builder() => return Err(fatal_model(error)),
                Some(Err(error)) => last = error.to_string(),
            }
        }
        Err(format!("{RETRY_EXHAUSTED} model transport: {last}").into())
    }
}

/// A streamed response's body, read by `parse_stream` as lines.
struct Body<'a, 'w> {
    stream: &'a Stream,
    response: reqwest::Response,
    chunk: Vec<u8>,
    at: usize,
    quiet_limit: Option<Duration>,
    waiting: &'a mut Waiter<'w>,
    /// Why the caller stopped it, if it did: handed back by `generate_once`.
    stopped: &'a RefCell<Option<Box<dyn std::error::Error>>>,
}

impl Read for Body<'_, '_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let Body { stream, response, chunk, at, quiet_limit, waiting, stopped } = self;
        while *at == chunk.len() {
            match stream.wait(response.chunk(), *quiet_limit, &mut **waiting) {
                Err(error) => {
                    *stopped.borrow_mut() = Some(error);
                    return Err(std::io::Error::other("stopped by its caller"));
                }
                Ok(None) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("nothing arrived for {:?}", quiet_limit.unwrap_or_default()),
                    ));
                }
                Ok(Some(Ok(Some(bytes)))) => {
                    *chunk = bytes.to_vec();
                    *at = 0;
                }
                Ok(Some(Ok(None))) => return Ok(0),
                Ok(Some(Err(error))) => return Err(std::io::Error::other(error)),
            }
        }
        let n = buf.len().min(chunk.len() - *at);
        buf[..n].copy_from_slice(&chunk[*at..*at + n]);
        *at += n;
        Ok(n)
    }
}

/// Extra samples taken when a turn ends with nothing to say. Qwen3.8 at xhigh
/// sometimes emits end-of-sequence while still inside <think>, which vLLM reports
/// as a clean `stop` with an empty answer. Each resample is a fresh draw at the
/// same effort, so this costs latency on the affected turns and nothing in
/// quality. Two extras take even the worst reported per-call rate (~38%) to about
/// a 5% chance of a still-empty turn.
const EMPTY_ANSWER_RESAMPLES: usize = 2;

/// A turn that stopped normally but said nothing and called nothing. A length
/// stop is excluded: resampling at the same budget would just run out again.
fn is_empty_answer(response: &S0maResponse) -> bool {
    response.finish_reason.as_deref() == Some("stop")
        && response.content.trim().is_empty()
        && response.tool_calls.is_empty()
}

/// Sample until the model produces something, within two separate budgets:
/// `attempts` for malformed or truncated streams (unchanged from before), and
/// EMPTY_ANSWER_RESAMPLES for clean-but-empty answers. Empty resampling is
/// side-effect free — this layer never executes tools — and is skipped for
/// greedy requests, where a resample would reproduce the same empty turn.
/// If every sample is empty, the empty response is returned exactly as before
/// rather than turned into an error, so callers' own handling still applies.
fn sample_until_answered<F>(
    attempts: usize,
    resample_empty: bool,
    mut sample: F,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<S0maResponse, Box<dyn std::error::Error>>,
{
    let mut malformed = 0;
    let mut empty = 0;

    loop {
        match sample() {
            Ok(response) if is_empty_answer(&response) && resample_empty && empty < EMPTY_ANSWER_RESAMPLES => {
                empty += 1;
                diagnostic!(
                    "[llm] model stopped with an empty answer ({} reasoning chars); resampling ({}/{EMPTY_ANSWER_RESAMPLES}). Displayed reasoning will repeat.",
                    response.reasoning_content.chars().count(),
                    empty
                );
            }
            Ok(response) => {
                if empty > 0 && is_empty_answer(&response) {
                    diagnostic!("[llm] answer still empty after {empty} resamples; returning it unchanged");
                }
                return Ok(response);
            }
            Err(error) if error.to_string().starts_with(MODEL_RESPONSE) => {
                malformed += 1;
                if malformed >= attempts {
                    return Err(
                        format!("{RETRY_EXHAUSTED} incomplete model response: {error}").into(),
                    );
                }
                diagnostic!(
                    "[llm] incomplete/malformed response; regenerating ({}/3). Partial displayed text may repeat. {error}",
                    malformed + 1
                );
            }
            // Operator steering/interruption and callback failures must pass through.
            Err(error) => return Err(error),
        }
    }
}

fn parse_stream<R, F>(
    reader: R,
    mut on_event: F,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    R: BufRead,
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    let mut content = String::new();
    let mut reasoning_content = String::new();

    let mut tool_calls: Vec<ToolCallBuilder> = Vec::new();
    let mut finish_reason = None;
    let mut saw_done = false;

    // This response is one generation, whether it is the call's first attempt
    // or a retry of a broken one.
    on_event(StreamEvent::Started)?;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("{MODEL_RESPONSE} stream read failed: {e}"))?;

        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };

        let data = data.trim();

        if data == "[DONE]" {
            saw_done = true;
            break;
        }

        if data.is_empty() {
            continue;
        }

        let value: Value = serde_json::from_str(data)
            .map_err(|e| format!("{MODEL_RESPONSE} invalid stream JSON: {e}"))?;
        if let Some(error) = value.get("error") {
            let detail = error.to_string();
            if fatal_http(reqwest::StatusCode::INTERNAL_SERVER_ERROR, &detail) {
                return Err(fatal_model(detail));
            }
            return Err(format!("{MODEL_RESPONSE} server stream error: {detail}").into());
        }
        let chunk: StreamChunk = serde_json::from_value(value)
            .map_err(|e| format!("{MODEL_RESPONSE} invalid stream structure: {e}"))?;

        // Reported after this chunk's text, so a steer taken at it never
        // drops text the operator didn't see.
        let usage = chunk.usage.as_ref().and_then(|usage| {
            let completion_tokens = usage.get("completion_tokens")?.as_u64()?;
            let prompt_tokens = usage.get("prompt_tokens").and_then(Value::as_u64);
            // Sent only by a server asked for it (vLLM:
            // --enable-prompt-tokens-details), and null otherwise.
            let details = usage.get("prompt_tokens_details");
            let cached_tokens = details.and_then(|details| details.get("cached_tokens")).and_then(Value::as_u64);
            // ds41d's: where the cached part came from.
            let cache_source = details
                .and_then(|details| details.get("cache_source"))
                .and_then(|source| serde_json::from_value::<CacheSource>(source.clone()).ok());
            Some(StreamEvent::Usage { completion_tokens, prompt_tokens, cached_tokens, cache_source, last: chunk.choices.is_empty() })
        });

        let Some(choice) = chunk.choices.first() else {
            if let Some(usage) = usage {
                on_event(usage)?;
            }
            continue;
        };

        if let Some(reason) = &choice.finish_reason {
            finish_reason = Some(reason.clone());
        }

        // Retain the exact trace as well as streaming it to the terminal.
        if let Some(reasoning) = choice.delta.reasoning_content.as_ref().or(choice.delta.reasoning.as_ref()) {
            reasoning_content.push_str(reasoning);
            on_event(StreamEvent::Reasoning(reasoning.clone()))?;
        }

        // Assistant text is valid alongside tool calls, so expose each
        // content delta immediately instead of holding the entire response
        // until the completion marker arrives.
        if let Some(text) = &choice.delta.content {
            content.push_str(text);

            if !text.is_empty() {
                on_event(StreamEvent::Content(text.clone()))?;
            }
        }

        // Collect structured tool calls.
        for tool_delta in &choice.delta.tool_calls {
            while tool_calls.len() <= tool_delta.index {
                tool_calls.push(ToolCallBuilder::default());
            }

            let builder = &mut tool_calls[tool_delta.index];

            if let Some(id) = &tool_delta.id {
                builder.id.push_str(id);
            }

            if let Some(kind) = &tool_delta.kind {
                builder.kind.push_str(kind);
            }

            if let Some(function) = &tool_delta.function {
                if let Some(name) = &function.name {
                    builder.name.push_str(name);
                    if !name.is_empty() {
                        on_event(StreamEvent::ToolArguments(name.clone()))?;
                    }
                }

                if let Some(arguments) = &function.arguments {
                    builder.arguments.push_str(arguments);
                    if !arguments.is_empty() {
                        on_event(StreamEvent::ToolArguments(arguments.clone()))?;
                    }
                }
            }
        }

        if let Some(usage) = usage {
            on_event(usage)?;
        }
    }

    if !saw_done {
        return Err(
            format!("{MODEL_RESPONSE} LLM stream ended before the completion marker").into(),
        );
    }

    let tool_calls: Vec<ToolCall> = tool_calls
        .into_iter()
        .filter(|call| !call.name.is_empty())
        .map(|call| ToolCall {
            id: call.id,
            kind: if call.kind.is_empty() {
                "function".to_string()
            } else {
                call.kind
            },
            function: FunctionCall {
                name: call.name,
                arguments: call.arguments,
            },
        })
        .collect();

    Ok(S0maResponse {
        content,
        reasoning_content,
        tool_calls,
        finish_reason,
    })
}

/// The thread's conversation, else `default` as a one-off, ended once its
/// call returns (a sleep pass or a worker scope its caller didn't open).
fn one_off<T>(default: impl FnOnce() -> Conversation, call: impl FnOnce(&Conversation) -> T) -> T {
    let (conversation, one_off) = crate::conversations::current_or(default);
    let result = call(&conversation);
    if one_off {
        conversation.end_later();
    }
    result
}

/// The thread's conversation, else `default` (one that lives on: the main
/// turn's, a side call's).
fn standing(default: impl FnOnce() -> Conversation) -> Conversation {
    crate::conversations::current_or(default).0
}

/// A side call on the main lane (the reasoning router's).
pub fn chat<F>(
    messages: &[Message],
    options: &ChatOptions,
    on_event: F,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    let conversation = standing(|| Conversation::side("router"));
    chat_inner(messages, options, None, 0, &conversation, None, on_event)
}

/// Sleep compactions share model weights, with explicit slot routing.
pub fn chat_sleep(
    messages: &[Message],
    options: &ChatOptions,
    slot: usize,
) -> Result<S0maResponse, Box<dyn std::error::Error>> {
    if slot > WORKER_LANES {
        return Err("invalid sleep lane".into());
    }
    one_off(|| Conversation::sleep(&format!("lane-{slot}")), |conversation| {
        chat_inner(messages, options, None, slot, conversation, Some(Duration::from_secs(7200)), |_| Ok(()))
    })
}

/// `chat_sleep` that stops at the next stream event once `cancel` is set, so a
/// background writer can be stopped mid-answer instead of waited for.
pub fn chat_sleep_cancel(
    messages: &[Message],
    options: &ChatOptions,
    slot: usize,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<S0maResponse, Box<dyn std::error::Error>> {
    if slot > WORKER_LANES {
        return Err("invalid sleep lane".into());
    }
    one_off(|| Conversation::sleep(&format!("lane-{slot}")), |conversation| {
        chat_config(messages, options, None, slot, conversation, Some(Duration::from_secs(7200)), None, 3, |_| {
            if cancel.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::SeqCst)) {
                Err("cancelled at a stream boundary".into())
            } else {
                Ok(())
            }
        })
    })
}

/// Strict batch decoding; application validation still checks evidence and bounds.
pub fn chat_sleep_structured(
    messages: &[Message],
    options: &ChatOptions,
    slot: usize,
    schema: Value,
) -> Result<S0maResponse, Box<dyn std::error::Error>> {
    chat_sleep_structured_cancel(messages,options,slot,schema,None)
}
pub fn chat_sleep_structured_cancel(messages:&[Message],options:&ChatOptions,slot:usize,schema:Value,
    cancel:Option<&std::sync::atomic::AtomicBool>)->Result<S0maResponse,Box<dyn std::error::Error>> {
    if !(1..=WORKER_LANES).contains(&slot) {
        return Err("invalid sleep worker lane".into());
    }
    one_off(|| Conversation::sleep(&format!("batch-{slot}")), |conversation| {
        chat_config(
            messages,
            options,
            None,
            slot,
            conversation,
            Some(Duration::from_secs(1800)),
            Some(serde_json::json!({"type":"json_schema", "json_schema":{
                "name":"sleep_batch", "strict":true, "schema":schema}})),
            1,
            |_| {if cancel.is_some_and(|flag|flag.load(std::sync::atomic::Ordering::SeqCst)) {
                Err("background compaction cancelled at a stream boundary".into())
            } else {Ok(())}},
        )
    })
}

/// A judgement with a strict schema and no tools, made on the main lane while
/// the main agent waits for it (the goal audit runs inside a tool call). Stops
/// at the next stream event once `interrupt` is set.
pub fn chat_judge(
    messages: &[Message],
    options: &ChatOptions,
    name: &str,
    schema: Value,
    interrupt: &std::sync::atomic::AtomicBool,
) -> Result<S0maResponse, Box<dyn std::error::Error>> {
    let conversation = standing(|| Conversation::side("judge"));
    chat_config(
        messages,
        options,
        None,
        0,
        &conversation,
        Some(Duration::from_secs(900)),
        Some(serde_json::json!({"type": "json_schema", "json_schema": {"name": name, "strict": true, "schema": schema}})),
        2,
        |_| {
            if interrupt.load(std::sync::atomic::Ordering::SeqCst) {
                Err("interrupted".into())
            } else {
                Ok(())
            }
        },
    )
}

/// Final integration keeps an append-only prefix on slot zero, with bounded transport time.
pub fn chat_sleep_tools<F>(
    messages: &[Message],
    options: &ChatOptions,
    tools: &[ToolDefinition],
    on_event: F,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    one_off(|| Conversation::sleep("integration"), |conversation| {
        chat_config(messages, options, Some(tools), 0, conversation, Some(Duration::from_secs(3600)), None, 1, on_event)
    })
}

/// Every worker lane, paired with the server's per-request ceiling.
///
/// There is no slot state to poll. vLLM aborts a request when its client
/// disconnects, so a lane whose call has returned carries no server-side work;
/// which lanes are free is s0ma's own accounting (the subagent manager and
/// maintenance leases). Deep sleep drains both before it borrows these.
pub fn sleep_worker_slots() -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
    let capacity = model_context_capacity()?;
    Ok((1..=WORKER_LANES).map(|lane| (lane, capacity)).collect())
}

pub fn chat_quiet(
    messages: &[Message],
    options: &ChatOptions,
) -> Result<S0maResponse, Box<dyn std::error::Error>> {
    chat(messages, options, |_| Ok(()))
}

/// The bytes the main agent's request would POST, for tests of the call store.
#[cfg(test)]
pub(crate) fn main_request_body_for_test(messages: &[Message], tools: &[ToolDefinition]) -> Vec<u8> {
    serde_json::to_vec(&chat_request(messages, &ChatOptions::xhigh(), Some(tools), &Conversation::main(), true, None)).unwrap()
}

/// The main agent's call: `body` receives the exact bytes POSTed, so the call
/// store saves what was sent rather than a re-rendering.
pub fn chat_with_tools_capturing<F>(
    messages: &[Message],
    options: &ChatOptions,
    tools: &[ToolDefinition],
    on_event: F,
    body: &mut Option<Vec<u8>>,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    let conversation = standing(Conversation::main);
    chat_config_capturing(messages, options, Some(tools), 0, &conversation, None, None, 3, on_event, Some(body), None)
}

/// `chat_with_tools_capturing` against another server, for tests.
#[cfg(test)]
pub(crate) fn chat_with_tools_capturing_at<F>(
    base_url: &str,
    messages: &[Message],
    tools: &[ToolDefinition],
    on_event: F,
    body: &mut Option<Vec<u8>>,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    chat_config_capturing(messages, &ChatOptions::xhigh(), Some(tools), 0, &Conversation::main(), None, None, 3, on_event, Some(body), Some(base_url))
}

/// Task-scoped worker request: explicitly pinned, VM tools, finite lifetime.
pub fn chat_worker<F>(
    messages: &[Message],
    options: &ChatOptions,
    tools: &[ToolDefinition],
    slot: usize,
    on_event: F,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    one_off(|| Conversation::worker("unscoped", &crate::conversations::new_uuid()), |conversation| {
        chat_inner(messages, options, Some(tools), slot, conversation, Some(Duration::from_secs(300)), on_event)
    })
}

/// Worker request with no tools offered: used for compaction, where the turn
/// must be a written summary rather than more VM activity. History may still
/// contain earlier tool calls and results.
pub fn chat_worker_text<F>(
    messages: &[Message],
    options: &ChatOptions,
    slot: usize,
    on_event: F,
) -> Result<S0maResponse, Box<dyn std::error::Error>>
where
    F: FnMut(StreamEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    one_off(|| Conversation::worker("unscoped", &crate::conversations::new_uuid()), |conversation| {
        chat_inner(messages, options, None, slot, conversation, Some(Duration::from_secs(300)), on_event)
    })
}

/// The server's per-request context ceiling, from `/v1/models`. vLLM applies one
/// `max_model_len` to every request, so there is no per-slot partition to read.
/// Also verifies the configured model is the one being served.
pub fn model_context_capacity() -> Result<usize, Box<dyn std::error::Error>> {
    let response: Value = Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .build()?
        .get(endpoint("/v1/models"))
        .send()?
        .error_for_status()?
        .json()?;
    served_capacity(&response, &model_name())
}

fn served_capacity(response: &Value, model: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let data = response
        .get("data")
        .and_then(Value::as_array)
        .ok_or("unexpected /v1/models response")?;
    let entry = data
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model))
        .ok_or_else(|| {
            let served: Vec<&str> = data.iter().filter_map(|e| e.get("id").and_then(Value::as_str)).collect();
            format!("model '{model}' is not served (serving: {}); set S0MA_LLM_MODEL", served.join(", "))
        })?;
    let limit = entry
        .get("max_model_len")
        .and_then(Value::as_u64)
        .filter(|limit| *limit > 0)
        .ok_or_else(|| format!("model '{model}' advertises no max_model_len; cannot budget context"))?;
    Ok(usize::try_from(limit)?)
}

/// Kept under its old name so callers are unchanged: every lane has the same
/// ceiling now.
pub fn slot_context_capacity(lane: usize) -> Result<usize, Box<dyn std::error::Error>> {
    check_lane(lane)?;
    model_context_capacity()
}

/// Fail closed if a lane is out of range or a reservation exceeds what one
/// request may hold.
pub fn check_worker_slot_ids(
    ids: &[usize],
    context_limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let capacity = model_context_capacity()?;
    for &id in ids {
        if !(1..=WORKER_LANES).contains(&id) {
            return Err(format!("worker lane {id} is outside 1..={WORKER_LANES}").into());
        }
    }
    if context_limit > capacity {
        return Err(format!("worker reservation of {context_limit} tokens exceeds the server's {capacity}").into());
    }
    Ok(())
}

/// Nothing to erase: vLLM frees a request's KV blocks when it finishes and evicts
/// cached prefixes itself. Kept, with its lane check, so every release path and
/// its error handling stays intact.
pub fn release_worker_slot(slot: usize) -> Result<(), Box<dyn std::error::Error>> {
    if !(1..=WORKER_LANES).contains(&slot) {
        return Err("invalid worker lane".into());
    }
    Ok(())
}

#[cfg(test)]
mod reasoning_retention_tests {
    use super::*;

    #[test]
    fn capacity_comes_from_the_configured_served_model() {
        let models = serde_json::json!({"data":[
            {"id":"qwen38-nvfp4","max_model_len":262144},
            {"id":"other","max_model_len":8192}
        ]});
        assert_eq!(served_capacity(&models, "qwen38-nvfp4").unwrap(), 262144);
        let missing = served_capacity(&models, "qwen").unwrap_err().to_string();
        assert!(missing.contains("qwen38-nvfp4"), "error should list what is served: {missing}");
        assert!(served_capacity(&serde_json::json!({"data":[{"id":"m","max_model_len":0}]}), "m").is_err());
        assert!(served_capacity(&serde_json::json!([]), "m").is_err());
    }

    #[test]
    fn requests_carry_no_llama_cpp_fields_and_map_none_to_no_thinking() {
        let messages = [Message::user("hi".into())];
        let value = serde_json::to_value(chat_request(&messages, &ChatOptions::instruct(), None, &Conversation::side("test"), true, None)).unwrap();
        assert!(value.get("cache_prompt").is_none());
        assert!(value.get("id_slot").is_none());
        assert!(value.get("reasoning_effort").is_none());
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], false);
        assert!(value["chat_template_kwargs"].get("reasoning_effort").is_none());

        let no_tools: &[ToolDefinition] = &[];
        let value = serde_json::to_value(chat_request(&messages, &ChatOptions::xhigh(), Some(no_tools), &Conversation::side("test"), true, None)).unwrap();
        assert_eq!(value["reasoning_effort"], "xhigh");
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);
        assert!(value.get("tools").is_none() && value.get("tool_choice").is_none());
    }

    #[test]
    fn streaming_requests_ask_for_token_counts() {
        let options = serde_json::to_value(stream_options_for(true, None)).unwrap();
        assert_eq!(options["include_usage"], true);
        assert_eq!(options["continuous_usage_stats"], true);
        assert!(stream_options_for(false, None).is_none(), "only streaming requests may carry it");
        assert!(stream_options_for(true, Some(" OFF ")).is_none());
        assert!(stream_options_for(true, Some("on")).is_some());
        if std::env::var("S0MA_STREAM_USAGE").is_err() {
            let messages = [Message::user("hi".into())];
            let value = serde_json::to_value(chat_request(&messages, &ChatOptions::instruct(), None, &Conversation::side("test"), true, None)).unwrap();
            assert_eq!(value["stream_options"]["include_usage"], true);
        }
    }

    #[test]
    fn the_prompt_and_its_cached_part_ride_along_with_usage_for_the_prefill_rate() {
        // vLLM with --enable-prompt-tokens-details on the closing report, and
        // its null prompt_tokens_details without the flag mid-stream.
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}],\"usage\":{\"prompt_tokens\":40000,\"completion_tokens\":1,\"prompt_tokens_details\":null}}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":2}}\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":40000,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":36000}}}\n",
            "data: [DONE]\n",
        );
        let mut seen = Vec::new();
        parse_stream(stream.as_bytes(), |event| {
            if let StreamEvent::Usage { prompt_tokens, cached_tokens, last, .. } = event {
                seen.push((prompt_tokens, cached_tokens, last));
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(
            seen,
            [(Some(40_000), None, false), (None, None, false), (Some(40_000), Some(36_000), true)],
            "null details and a missing prompt count are both just absent"
        );
    }

    #[test]
    fn usage_in_the_stream_is_reported_after_each_chunks_text() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"hm\"}}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":1}}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\",\"tool_calls\":[{\"index\":0,\"id\":\"c\",\"function\":{\"name\":\"vm_exec\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":4}}\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4,\"total_tokens\":13}}\n",
            "data: [DONE]\n",
        );
        let mut seen = Vec::new();
        let response = parse_stream(stream.as_bytes(), |event| {
            seen.push(match event {
                StreamEvent::Started => "start".to_string(),
                StreamEvent::Reasoning(text) => format!("r:{text}"),
                StreamEvent::Content(text) => format!("c:{text}"),
                StreamEvent::ToolArguments(text) => format!("a:{text}"),
                StreamEvent::Usage { completion_tokens, last, .. } => format!("u:{completion_tokens}:{last}"),
                StreamEvent::Waiting => "wait".to_string(),
            });
            Ok(())
        })
        .unwrap();
        // "start" opens every generation, so a retry of this call is measured
        // on its own; the tool's name counts as generated output beside its
        // arguments.
        assert_eq!(seen, ["start", "r:hm", "u:1:false", "c:ok", "a:vm_exec", "a:{}", "u:4:false", "u:4:true"]);
        assert_eq!(response.content, "ok");
        assert_eq!(response.tool_calls.len(), 1);

        // Usage of an unexpected shape is ignored rather than failing the stream.
        let odd = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}],\"usage\":{\"completion_tokens\":1.5}}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}],\"usage\":\"n/a\"}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"c\"},\"finish_reason\":\"stop\"}],\"usage\":null}\n",
            "data: [DONE]\n",
        );
        let mut usages = 0;
        let response = parse_stream(odd.as_bytes(), |event| {
            usages += usize::from(matches!(event, StreamEvent::Usage { .. }));
            Ok(())
        })
        .unwrap();
        assert_eq!((response.content.as_str(), usages), ("abc", 0));

        // A server that sends no usage: the stream reads as before.
        let plain = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\ndata: [DONE]\n";
        let mut count = 0;
        let response = parse_stream(plain.as_bytes(), |_| { count += 1; Ok(()) }).unwrap();
        // The generation's own start, then its one content delta.
        assert_eq!((response.content.as_str(), count), ("hi", 2));
    }

    /// The bug this guards: a broken stream is retried through the same
    /// callback, so without a per-generation marker the abandoned attempt, the
    /// gap and the retry's prefill were all measured as one generation while
    /// only the retry's tokens were counted. The rate read far too low.
    #[test]
    fn every_attempt_of_a_retried_call_opens_its_own_generation() {
        // No [DONE]: an incomplete stream, which is what earns a retry.
        let broken = "data: {\"choices\":[{\"delta\":{\"content\":\"part\"}}]}\n";
        let good = "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\ndata: [DONE]\n";
        let mut bodies = [broken, good].into_iter();
        let mut starts = 0;
        let response = sample_until_answered(3, false, || {
            let body = bodies.next().expect("one body per attempt");
            parse_stream(body.as_bytes(), |event| {
                starts += usize::from(matches!(event, StreamEvent::Started));
                Ok(())
            })
        })
        .unwrap();
        assert_eq!((response.content.as_str(), starts), ("done", 2));
    }

    #[test]
    fn requests_name_their_conversation_only_to_an_engine_that_takes_it() {
        let _lock = crate::conversations::TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let messages = [Message::user("hi".into())];
        let worker = Conversation::worker("b1-2", &crate::conversations::new_uuid());
        crate::conversations::set_features_for_test(None);
        let plain = serde_json::to_value(chat_request(&messages, &ChatOptions::instruct(), None, &worker, true, None)).unwrap();
        assert!(plain.get("conversation_id").is_none(), "vLLM and fne: as before");
        if std::env::var("S0MA_LLM_PRIORITY").is_err() {
            assert!(plain.get("priority").is_none());
        }
        crate::conversations::set_features_for_test(Some(crate::conversations::ds41()));
        let named = serde_json::to_value(chat_request(&messages, &ChatOptions::instruct(), None, &worker, true, None)).unwrap();
        assert_eq!(named["conversation_id"], worker.id());
        assert_eq!(named["priority"], crate::conversations::WORKER_PRIORITY);
        let main = serde_json::to_value(chat_request(&messages, &ChatOptions::instruct(), None, &Conversation::main(), true, None)).unwrap();
        assert!(main["conversation_id"].as_str().unwrap().starts_with("main:"));
        assert_eq!(main["priority"], 0, "the main turn keeps the lowest number");
        crate::conversations::set_features_for_test(None);
    }

    #[test]
    fn where_the_cache_came_from_rides_along() {
        let stream = concat!(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":800,\"cache_source\":\"disk\"}}}\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":800,\"cache_source\":\"elsewhere\"}}}\n",
            "data: [DONE]\n",
        );
        let mut sources = Vec::new();
        parse_stream(stream.as_bytes(), |event| {
            if let StreamEvent::Usage { cache_source, .. } = event {
                sources.push(cache_source);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(sources, [Some(CacheSource::Disk), None], "an unknown source is left out");
    }

    #[test]
    fn priority_is_omitted_unless_enabled() {
        if std::env::var("S0MA_LLM_PRIORITY").is_ok() { return; }
        assert_eq!(lane_priority(0), None);
        assert_eq!(lane_priority(5), None);
    }

    #[test]
    fn lanes_are_bounded() {
        assert!(check_lane(0).is_ok());
        assert!(check_lane(WORKER_LANES).is_ok());
        assert!(check_lane(WORKER_LANES + 1).is_err());
        assert!(release_worker_slot(0).is_err());
        assert!(release_worker_slot(WORKER_LANES).is_ok());
    }

    #[test]
    fn newer_vllm_reasoning_field_is_read() {
        let stream = "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think\"}}]}\n\ndata: [DONE]\n";
        let response = parse_stream(std::io::Cursor::new(stream), |_| Ok(())).unwrap();
        assert_eq!(response.reasoning_content, "think");
    }

    #[test]
    fn streaming_trace_is_returned_with_tool_call_and_replayed_separately() {
        let chunks = [
            serde_json::json!({"choices":[{"delta":{"reasoning_content":"Check "}}]}),
            serde_json::json!({"choices":[{"delta":{"reasoning_content":"the control.\n", "content":"Checking now."}}]}),
            serde_json::json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"vm_exec","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}),
        ];
        let mut stream = String::new();
        for chunk in chunks {
            stream.push_str(&format!("data: {chunk}\n\n"));
        }
        stream.push_str("data: [DONE]\n\n");
        let mut displayed_reasoning = String::new();
        let response = parse_stream(std::io::Cursor::new(stream), |event| {
            if let StreamEvent::Reasoning(text) = event {
                displayed_reasoning.push_str(&text);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(response.reasoning_content, "Check the control.\n");
        assert_eq!(response.reasoning_content, displayed_reasoning);
        assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
        let message = Message::assistant_with_tools(response.content, response.tool_calls)
            .with_reasoning(response.reasoning_content);
        let value = serde_json::to_value(message).unwrap();
        assert_eq!(value["reasoning_content"], displayed_reasoning);
        assert_eq!(value["content"], "Checking now.");
        assert_eq!(value["tool_calls"][0]["function"]["name"], "vm_exec");
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;
    #[test]
    fn only_unrecoverable_model_failures_are_fatal() {
        for code in [400, 401, 403, 404, 413, 422, 501] {
            assert!(fatal_http(
                reqwest::StatusCode::from_u16(code).unwrap(),
                "bad request"
            ));
        }
        for code in [408, 409, 425, 429, 500, 502, 503, 504] {
            assert!(!fatal_http(
                reqwest::StatusCode::from_u16(code).unwrap(),
                "temporarily unavailable"
            ));
        }
        assert!(fatal_http(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "CUDA out of memory"
        ));
        assert!(fatal_http(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "request exceeds the available context size"
        ));
        assert!(!is_fatal_message(
            "episode applicability must contain 1..12000 bytes"
        ));
        assert!(is_fatal_error(fatal_model("oversized request").as_ref()));
    }
    #[test]
    fn retry_passes_validation_feedback_and_stops_after_success() {
        let mut calls = 0;
        let value = retry_text("test", |feedback| {
            calls += 1;
            if calls == 1 {
                assert!(feedback.is_none());
                Err("episode applicability must contain 1..12000 bytes".into())
            } else {
                assert!(feedback.unwrap().contains("applicability"));
                Ok("repaired".into())
            }
        })
        .unwrap();
        assert_eq!(value, "repaired");
        assert_eq!(calls, 2);
    }
    #[test]
    fn validation_exhaustion_is_nonfatal_and_bounded() {
        let mut calls = 0;
        let error = retry_text("test", |_| {
            calls += 1;
            Err("invalid JSON".into())
        })
        .unwrap_err();
        assert_eq!(calls, 3);
        assert!(error.starts_with(RETRY_EXHAUSTED));
        assert!(!is_fatal_message(&error));
    }
    #[test]
    fn fatal_and_already_exhausted_failures_do_not_multiply_retries() {
        for message in [
            "[fatal-model] oversized request",
            "[retry-exhausted] transport unavailable",
        ] {
            let mut calls = 0;
            assert_eq!(
                retry_text("test", |_| {
                    calls += 1;
                    Err(message.into())
                })
                .unwrap_err(),
                message
            );
            assert_eq!(calls, 1);
        }
    }
    #[test]
    fn truncated_and_malformed_streams_are_recoverable_but_callbacks_pass_through() {
        for text in ["data: {\n\n", "data: {\"choices\":[]}\n\n"] {
            let error = parse_stream(std::io::Cursor::new(text), |_| Ok(())).unwrap_err();
            assert!(error.to_string().starts_with(MODEL_RESPONSE));
        }
        let error = parse_stream(
            std::io::Cursor::new(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n",
            ),
            |_| Err("operator interrupted".into()),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "operator interrupted");
    }
    #[test]
    fn stream_context_error_remains_fatal() {
        let error=parse_stream(std::io::Cursor::new("data: {\"error\":{\"message\":\"request exceeds the available context size\"}}\n\n"), |_|Ok(())).unwrap_err();
        assert!(is_fatal_error(error.as_ref()));
    }
    #[test]
    fn transient_http_failure_retries_then_succeeds() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for (status, body) in [("503 Service Unavailable", "loading"), ("200 OK", "ready")] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let _ = stream.read(&mut request).unwrap();
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            }
        });
        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        assert_eq!(
            send_model_request(client.get(format!("http://{address}")))
                .unwrap()
                .text()
                .unwrap(),
            "ready"
        );
        server.join().unwrap();
    }
    #[test]
    fn http_fatal_body_is_preserved_without_retry() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            let body = "request (300000 tokens) exceeds the available context size (230000 tokens)";
            write!(
                stream,
                "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });
        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let error = send_model_request(client.get(format!("http://{address}"))).unwrap_err();
        assert!(is_fatal_error(error.as_ref()));
        assert!(error.to_string().contains("300000 tokens"));
        server.join().unwrap();
    }
}

#[cfg(test)]
mod generating_tests {
    use super::*;

    #[test]
    fn a_generation_counts_while_it_lives_and_not_after_however_it_ends() {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let count = || COUNTER.load(Ordering::SeqCst);
        let first = Generating::on(&COUNTER);
        let second = Generating::on(&COUNTER);
        assert_eq!(count(), 2, "concurrent requests (a turn and its workers) each count");
        drop(first);
        assert_eq!(count(), 1);
        drop(second);
        assert_eq!(count(), 0);
        // A request that panics mid-stream still stops counting.
        let _ = std::panic::catch_unwind(|| {
            let _held = Generating::on(&COUNTER);
            panic!("stream broke");
        });
        assert_eq!(count(), 0);
    }

    #[test]
    fn a_request_counts_from_its_send_to_the_end_of_its_stream() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 8192];
            let _ = stream.read(&mut request).unwrap();
            // While the server has the request, as during a prefill.
            let during_prefill = generating();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
            write!(stream, "data: {{\"choices\":[{{\"delta\":{{\"content\":\"half\"}}}}]}}\n\n").unwrap();
            stream.flush().unwrap();
            std::thread::sleep(Duration::from_millis(200));
            write!(stream, "data: {{\"choices\":[{{\"delta\":{{\"content\":\" done\"}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n").unwrap();
            during_prefill
        });
        let stream = Stream::new().unwrap();
        let mut while_streaming = Vec::new();
        let response = generate_once(&stream, &format!("http://{address}/v1/chat/completions"), &serde_json::to_vec(&serde_json::json!({"model":"m"})).unwrap(), Some(Duration::from_secs(5)), &mut |event| {
            if matches!(event, StreamEvent::Content(_)) {
                while_streaming.push(generating());
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(response.content, "half done");
        assert!(server.join().unwrap(), "counted while the server held the request");
        assert_eq!(while_streaming, [true, true], "and while the reply streamed");
    }
}

#[cfg(test)]
mod interim_fix_tests {
    use super::*;
    use std::cell::RefCell;

    fn response(content: &str, reasoning: &str, finish: &str, tools: bool) -> S0maResponse {
        S0maResponse {
            content: content.into(),
            reasoning_content: reasoning.into(),
            tool_calls: if tools {
                vec![ToolCall { id: "c".into(), kind: "function".into(),
                    function: FunctionCall { name: "vm_exec".into(), arguments: "{}".into() } }]
            } else { vec![] },
            finish_reason: Some(finish.into()),
        }
    }

    /// Scripted samples; counts how many were drawn.
    fn script(items: Vec<Result<S0maResponse, Box<dyn std::error::Error>>>)
        -> (RefCell<std::vec::IntoIter<Result<S0maResponse, Box<dyn std::error::Error>>>>, RefCell<usize>) {
        (RefCell::new(items.into_iter()), RefCell::new(0))
    }

    #[test]
    fn thinking_modes_send_no_frequency_penalty() {
        for options in [ChatOptions::low(), ChatOptions::medium(), ChatOptions::xhigh(), ChatOptions::sleep()] {
            assert_eq!(options.frequency_penalty, 0.0,
                "vLLM applies frequency_penalty unwindowed; a non-zero value corrupts long outputs");
            assert_eq!(options.presence_penalty, 0.0);
            assert_ne!(options.reasoning_effort, "none", "thinking modes must keep thinking on");
        }
        let messages = [Message::user("hi".into())];
        let value = serde_json::to_value(chat_request(&messages, &ChatOptions::normal(), None, &Conversation::side("test"), true, None)).unwrap();
        assert_eq!(value["frequency_penalty"], 0.0);
        assert_eq!(value["reasoning_effort"], "xhigh");
    }

    #[test]
    fn empty_answer_is_resampled_until_the_model_speaks() {
        let (items, drawn) = script(vec![
            Ok(response("", "thinking...", "stop", false)),
            Ok(response("  ", "more thinking", "stop", false)),
            Ok(response("Here it is.", "done", "stop", false)),
        ]);
        let result = sample_until_answered(1, true, || { *drawn.borrow_mut() += 1; items.borrow_mut().next().unwrap() }).unwrap();
        assert_eq!(result.content, "Here it is.");
        assert_eq!(*drawn.borrow(), 3);
    }

    #[test]
    fn persistent_empty_answer_is_returned_not_turned_into_an_error() {
        let (items, drawn) = script((0..5).map(|_| Ok(response("", "t", "stop", false))).collect());
        let result = sample_until_answered(3, true, || { *drawn.borrow_mut() += 1; items.borrow_mut().next().unwrap() }).unwrap();
        assert!(result.content.is_empty());
        assert_eq!(*drawn.borrow(), 1 + EMPTY_ANSWER_RESAMPLES);
    }

    #[test]
    fn greedy_requests_are_not_resampled() {
        let (items, drawn) = script(vec![Ok(response("", "t", "stop", false)), Ok(response("x", "", "stop", false))]);
        let result = sample_until_answered(3, false, || { *drawn.borrow_mut() += 1; items.borrow_mut().next().unwrap() }).unwrap();
        assert!(result.content.is_empty());
        assert_eq!(*drawn.borrow(), 1);
    }

    #[test]
    fn tool_calls_and_length_stops_are_not_empty_answers() {
        assert!(!is_empty_answer(&response("", "t", "tool_calls", true)));
        assert!(!is_empty_answer(&response("", "t", "stop", true)));
        assert!(!is_empty_answer(&response("", "t", "length", false)));
        assert!(is_empty_answer(&response("\n", "t", "stop", false)));
    }

    #[test]
    fn malformed_stream_budget_is_unchanged_and_separate() {
        let malformed = || -> Result<S0maResponse, Box<dyn std::error::Error>> {
            Err(format!("{MODEL_RESPONSE} truncated").into())
        };
        // attempts = 1: first malformed stream is terminal, as before.
        let (items, drawn) = script(vec![malformed(), Ok(response("x", "", "stop", false))]);
        let error = sample_until_answered(1, true, || { *drawn.borrow_mut() += 1; items.borrow_mut().next().unwrap() }).unwrap_err();
        assert!(error.to_string().starts_with(RETRY_EXHAUSTED));
        assert_eq!(*drawn.borrow(), 1);
        // attempts = 3: empties don't consume the malformed budget, and vice versa.
        let (items, drawn) = script(vec![
            Ok(response("", "t", "stop", false)), malformed(), Ok(response("", "t", "stop", false)),
            malformed(), Ok(response("answer", "", "stop", false)),
        ]);
        let result = sample_until_answered(3, true, || { *drawn.borrow_mut() += 1; items.borrow_mut().next().unwrap() }).unwrap();
        assert_eq!(result.content, "answer");
        assert_eq!(*drawn.borrow(), 5);
    }

    #[test]
    fn interrupts_pass_through_without_resampling() {
        let (items, drawn) = script(vec![Err("operator interrupted".into()), Ok(response("x", "", "stop", false))]);
        let error = sample_until_answered(3, true, || { *drawn.borrow_mut() += 1; items.borrow_mut().next().unwrap() }).unwrap_err();
        assert_eq!(error.to_string(), "operator interrupted");
        assert_eq!(*drawn.borrow(), 1);
    }
}

#[cfg(test)]
mod default_effort_tests {
    #[test]
    fn requested_modes_use_xhigh() {
        for options in [
            super::ChatOptions::normal(),
            super::ChatOptions::sleep(),
        ] {
            assert_eq!(options.reasoning_effort, "xhigh");
        }
    }
}

#[cfg(test)]
mod repair_report_tests {
    #[test]
    fn a_repaired_call_is_reported_once() {
        let ids = vec!["call-report-once-a".to_string(), "call-report-once-b".to_string()];
        assert_eq!(super::first_reports(&ids), 2);
        // The same broken calls on the next request: nothing new to say.
        assert_eq!(super::first_reports(&ids), 0);
        assert_eq!(super::first_reports(&["call-report-once-c".to_string(), ids[0].clone()]), 1);
    }
}

#[cfg(test)]
mod stop_tests {
    //! Ctrl+C (any error from the callback) stops a call that has not started
    //! streaming, and the server sees the connection go: what makes vLLM abort
    //! the prefill.
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::mpsc;

    /// What the server does once it has read the request.
    #[derive(Clone, Copy)]
    enum Engine {
        /// Says nothing at all: still taking the request in.
        Silent,
        /// Sends the headers, then nothing: prefilling.
        Prefilling,
        /// Sends the headers, waits, then streams an answer.
        SlowFirstToken(Duration),
    }

    /// An engine on a free port. The channel gets how long after the request
    /// the client went away, if it did within 10 s.
    fn engine(behaviour: Engine) -> (String, mpsc::Receiver<Option<Duration>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            // The request: headers, then Content-Length bytes of body.
            let mut seen = Vec::new();
            let mut byte = [0u8; 1];
            while !seen.ends_with(b"\r\n\r\n") {
                socket.read_exact(&mut byte).unwrap();
                seen.push(byte[0]);
            }
            let head = String::from_utf8_lossy(&seen).to_ascii_lowercase();
            let length: usize = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .map(|value| value.trim().parse().unwrap())
                .unwrap_or(0);
            let mut body = vec![0; length];
            socket.read_exact(&mut body).unwrap();
            let asked = Instant::now();
            let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
            match behaviour {
                Engine::Silent => {}
                Engine::Prefilling => socket.write_all(headers.as_bytes()).unwrap(),
                Engine::SlowFirstToken(delay) => {
                    socket.write_all(headers.as_bytes()).unwrap();
                    std::thread::sleep(delay);
                    let answer = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                    let _ = socket.write_all(answer.as_bytes());
                    let _ = tx.send(None);
                    return;
                }
            }
            // Gone when a read sees the end of the stream (or the reset).
            let mut rest = [0u8; 64];
            let gone = loop {
                match socket.read(&mut rest) {
                    Ok(0) => break Some(asked.elapsed()),
                    Ok(_) => continue,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock || error.kind() == std::io::ErrorKind::TimedOut => break None,
                    Err(_) => break Some(asked.elapsed()),
                }
            };
            let _ = tx.send(gone);
        });
        (format!("http://{address}/v1/chat/completions"), rx)
    }

    fn call(url: &str, quiet_limit: Option<Duration>, stop_after: Option<usize>) -> (Result<S0maResponse, Box<dyn std::error::Error>>, Vec<String>, Duration) {
        let stream = Stream::new().unwrap();
        let mut events = Vec::new();
        let mut waits = 0;
        let started = Instant::now();
        let result = generate_once(&stream, url, b"{}", quiet_limit, &mut |event: StreamEvent| {
            if let StreamEvent::Waiting = event {
                waits += 1;
                if stop_after.is_some_and(|n| waits >= n) {
                    return Err("stopped: ctrl+c".into());
                }
            }
            events.push(format!("{event:?}"));
            Ok(())
        });
        (result, events, started.elapsed())
    }

    #[test]
    fn ctrl_c_during_a_prefill_stops_the_call_and_drops_the_connection() {
        let (url, gone) = engine(Engine::Prefilling);
        let (result, events, took) = call(&url, None, Some(3));
        let error = result.err().expect("stopped");
        assert_eq!(error.to_string(), "stopped: ctrl+c", "the caller's own error, not a broken stream to retry");
        assert!(took < Duration::from_secs(2), "within a few polls, not at the first token: {took:?}");
        assert_eq!(events.first().map(String::as_str), Some("Started"), "the headers had come");
        let gone = gone.recv_timeout(Duration::from_secs(15)).unwrap();
        assert!(gone.is_some_and(|after| after < Duration::from_secs(3)), "the server saw the client go: {gone:?}");
    }

    #[test]
    fn ctrl_c_before_the_server_has_answered_stops_it_too() {
        let (url, gone) = engine(Engine::Silent);
        let (result, events, took) = call(&url, None, Some(3));
        assert_eq!(result.err().expect("stopped").to_string(), "stopped: ctrl+c");
        assert!(took < Duration::from_secs(2), "{took:?}");
        assert!(events.iter().all(|event| event == "Waiting"), "nothing but waits: {events:?}");
        let gone = gone.recv_timeout(Duration::from_secs(15)).unwrap();
        assert!(gone.is_some_and(|after| after < Duration::from_secs(3)), "{gone:?}");
    }

    #[test]
    fn a_slow_first_token_is_waited_for_and_read_whole() {
        let (url, _gone) = engine(Engine::SlowFirstToken(Duration::from_millis(450)));
        let (result, events, _) = call(&url, None, None);
        let response = result.unwrap();
        assert_eq!(response.content, "ok");
        assert_eq!(events.first().map(String::as_str), Some("Started"));
        assert!(events.iter().filter(|event| *event == "Waiting").count() >= 3, "{events:?}");
        assert!(events.contains(&"Content(\"ok\")".to_string()));
    }

    #[test]
    fn the_quiet_limit_still_ends_a_stream_that_stops_arriving() {
        let (url, _gone) = engine(Engine::Prefilling);
        let (result, _, took) = call(&url, Some(Duration::from_millis(300)), None);
        let error = result.err().expect("timed out").to_string();
        assert!(error.starts_with(MODEL_RESPONSE) && error.contains("nothing arrived for 300ms"), "{error}");
        assert!(took < Duration::from_secs(2), "{took:?}");
    }
}
