mod tool_output;
pub(crate) use tool_output::{
    is_stub, limit_tool_output, limit_worker_output, locate_in_archive, page_stub, reread_stub, result_matches,
    result_page, source_truncation, worker_stub, WORKER_VISIBLE_CHARS,
};
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
    let request = crate::model_config::authorize(request);
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

// ---- the model server ----
//
// A local engine (ds41d or vLLM) by default, or a hosted OpenAI-compatible API:
// where it is, which model, its key and which it is are `model_config`'s.

/// Logical request lanes. Lane 0 is the main conversation and sleep integration;
/// 1..=WORKER_LANES are subagents, sleep batch workers and compaction leases.
///
/// vLLM has no slots. It batches every request continuously over one shared,
/// paged KV pool, so lanes exist only for the agent's own admission control. Sized to
/// the server's `--max-num-seqs` (16 in the recipe) minus the main lane: more
/// lanes than that would only queue inside vLLM.
pub const WORKER_LANES: usize = 15;

/// vLLM priority scheduling: lower values are scheduled first and can preempt
/// higher ones when the KV pool runs short. Opt-in via S0MA_LLM_PRIORITY=1,
/// because a server started without `--scheduling-policy priority` rejects any
/// non-zero priority with a 400, which the agent would treat as fatal; sent anyway
/// to ds41d, which says it takes it (and keeps prompts by it). Model calls
/// take theirs from their conversation (`conversations.rs`); these are for
/// the knowledge probes, which build their own requests.
const MAIN_PRIORITY: i32 = crate::conversations::MAIN_PRIORITY;
const WORKER_PRIORITY: i32 = crate::conversations::WORKER_PRIORITY;

pub(crate) fn endpoint(path: &str) -> String {
    format!("{}{path}", crate::model_config::base_url())
}

pub(crate) fn model_name() -> String {
    crate::model_config::model()
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
    /// Not an OpenAI field: left out for a hosted API.
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
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
    /// The engine's chat template's options: left out for a hosted API.
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<TemplateOptions<'a>>,
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
    /// vLLM's: left out for a hosted API, which may reject it.
    #[serde(skip_serializing_if = "Option::is_none")]
    continuous_usage_stats: Option<bool>,
}

/// `S0MA_STREAM_USAGE=off` sends no `stream_options`, for a server that
/// rejects them.
fn stream_options_for(stream: bool, setting: Option<&str>, hosted: bool) -> Option<StreamOptions> {
    let off = setting.is_some_and(|value| value.trim().eq_ignore_ascii_case("off"));
    (stream && !off).then_some(StreamOptions { include_usage: true, continuous_usage_stats: (!hosted).then_some(true) })
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

/// The model card lists xhigh, medium and low; the agent's instruct and router modes
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
    chat_request_for(messages, options, tools, conversation, stream, response_format, crate::model_config::hosted())
}

/// `chat_request`, for a local engine or (`hosted`) an OpenAI-compatible API.
/// A hosted API gets only the standard fields: no `top_k`, no template
/// options, no reasoning effort (its levels are the engine's model card's,
/// not an API's), no priority or conversation id, which a strict API rejects.
fn chat_request_for<'a>(
    messages: &'a [Message],
    options: &ChatOptions,
    tools: Option<&'a [ToolDefinition]>,
    conversation: &Conversation,
    stream: bool,
    response_format: Option<Value>,
    hosted: bool,
) -> ChatRequest<'a> {
    let (enable_thinking, reasoning_effort) = thinking(options);
    let (conversation_id, priority) = if hosted { (None, None) } else { crate::conversations::request_fields(conversation) };
    // An empty tool list is sent as no tools rather than `[]`.
    let tools = tools.filter(|tools| !tools.is_empty());

    ChatRequest {
        model: model_name(),
        response_format,
        messages: sanitized(messages),
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: (!hosted).then_some(options.top_k),
        presence_penalty: options.presence_penalty,
        frequency_penalty: options.frequency_penalty,
        stream,
        stream_options: stream_options_for(stream, std::env::var("S0MA_STREAM_USAGE").ok().as_deref(), hosted),
        reasoning_effort: reasoning_effort.filter(|_| !hosted),
        chat_template_kwargs: (!hosted).then_some(TemplateOptions {
            enable_thinking,
            preserve_thinking: true,
            reasoning_effort: None,
        }),
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
    if crate::model_config::hosted() {
        return Ok(estimated_tokens(messages, tools));
    }
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

/// A hosted API's count, estimated: it has no `/tokenize`, and its
/// tokenizer is not this one's to run. Generous on purpose (three bytes of
/// the request's JSON to a token, where English runs nearer four, and a
/// little for each message's framing), so the context is budgeted on the safe
/// side: a fold comes a little early rather than a request too large.
fn estimated_tokens(messages: &[Message], tools: Option<&[ToolDefinition]>) -> usize {
    let bytes = serde_json::to_vec(messages).map_or(0, |bytes| bytes.len())
        + tools.filter(|tools| !tools.is_empty()).and_then(|tools| serde_json::to_vec(tools).ok()).map_or(0, |bytes| bytes.len());
    bytes.div_ceil(3) + messages.len() * 8 + 16
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
            let request = crate::model_config::authorize_async(
                self.client.post(url).header(reqwest::header::CONTENT_TYPE, "application/json").body(body.to_vec()),
            );
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
/// which lanes are free is the agent's own accounting (the subagent manager and
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
///
/// A window set in the overview (`/model context`) wins. A hosted API's is
/// read from its model list where it gives one (`hosted_capacity`).
pub fn model_context_capacity() -> Result<usize, Box<dyn std::error::Error>> {
    if let Some(window) = crate::model_config::context_window() {
        return Ok(window);
    }
    if crate::model_config::hosted() {
        return Ok(hosted_capacity());
    }
    let response: Value = crate::model_config::authorize(
        Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(10))
            .build()?
            .get(endpoint("/v1/models")),
    )
    .send()?
    .error_for_status()?
    .json()?;
    served_capacity(&response, &model_name())
}

/// A hosted model's window: its entry in the API's `/v1/models`, where the
/// API gives one (`context_length`, OpenRouter's; `max_model_len`;
/// `context_window`), else `ASSUMED_HOSTED_CONTEXT`. Asked at most every ten
/// minutes per server and model: an API's list can run to hundreds of models.
fn hosted_capacity() -> usize {
    static KNOWN: std::sync::Mutex<Option<(String, usize, Instant, bool)>> = std::sync::Mutex::new(None);
    // A new server, model or key is asked again at once: the list a key
    // couldn't read (or got refused) may read with the next.
    let key = format!(
        "{} {} {:x}",
        crate::model_config::base_url(),
        model_name(),
        crate::util::fnv1a64(crate::model_config::api_key().unwrap_or_default().as_bytes())
    );
    if let Some((known, window, at, read)) = &*crate::util::locked(&KNOWN)
        && *known == key
        // A list that was read holds for ten minutes; one that wasn't is
        // tried again after one.
        && at.elapsed() < Duration::from_secs(if *read { 600 } else { 60 })
    {
        return *window;
    }
    let listed = (|| -> Option<Value> {
        let client = Client::builder().connect_timeout(Duration::from_secs(5)).timeout(Duration::from_secs(15)).build().ok()?;
        crate::model_config::authorize(client.get(endpoint("/v1/models"))).send().ok()?.error_for_status().ok()?.json().ok()
    })();
    let window = listed.as_ref().and_then(|list| listed_window(list, &model_name()));
    *crate::util::locked(&KNOWN) = Some((key, window.unwrap_or(crate::model_config::ASSUMED_HOSTED_CONTEXT), Instant::now(), listed.is_some()));
    window.unwrap_or(crate::model_config::ASSUMED_HOSTED_CONTEXT)
}

/// The window `model`'s entry in a `/v1/models` list gives, if it gives one.
fn listed_window(list: &Value, model: &str) -> Option<usize> {
    let entry = list.get("data")?.as_array()?.iter().find(|entry| entry.get("id").and_then(Value::as_str) == Some(model))?;
    ["context_length", "max_model_len", "context_window"]
        .iter()
        .find_map(|field| entry.get(field).and_then(Value::as_u64))
        .or_else(|| entry.pointer("/top_provider/context_length").and_then(Value::as_u64))
        .filter(|window| *window > 0)
        .and_then(|window| usize::try_from(window).ok())
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
    fn a_hosted_count_is_estimated_on_the_safe_side_and_its_window_read_from_the_list() {
        let text = "The operator asked for the build log and a summary of what failed. ".repeat(40);
        let messages = [Message::system("You are a careful assistant.".into()), Message::user(text.clone())];
        let estimate = estimated_tokens(&messages, None);
        // English runs about four characters a token: the estimate is above that.
        assert!(estimate > text.len() / 4 && estimate < text.len(), "{estimate} for {} chars", text.len());
        let tools = [ToolDefinition::function("vm_exec", "run a command", serde_json::json!({"type": "object"}))];
        assert!(estimated_tokens(&messages, Some(&tools)) > estimate, "tools count too");

        let list = serde_json::json!({"data": [
            {"id": "vendor/a", "context_length": 200000},
            {"id": "vendor/b", "top_provider": {"context_length": 65536}},
            {"id": "vendor/c", "max_model_len": 32768},
            {"id": "vendor/d"},
        ]});
        assert_eq!(listed_window(&list, "vendor/a"), Some(200_000));
        assert_eq!(listed_window(&list, "vendor/b"), Some(65_536));
        assert_eq!(listed_window(&list, "vendor/c"), Some(32_768));
        assert_eq!(listed_window(&list, "vendor/d"), None, "not given: assumed by the caller");
        assert_eq!(listed_window(&list, "vendor/e"), None);
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
}

