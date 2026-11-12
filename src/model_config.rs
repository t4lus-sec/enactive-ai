//! The model server: where it is, which model, its API key, whether it is a
//! local engine or a hosted API, and whether embeddings are used.
//!
//! Set in the overview (Alt+O, "model") or with `/model` and `/embeddings`,
//! and kept in `data/model.json`, a file only its owner can read (it may hold
//! the key). An environment variable, when set, wins over what is kept, and
//! the overview says so:
//!
//! | setting | command | environment |
//! |---|---|---|
//! | server | `/model server <url>` | `S0MA_LLM_BASE_URL` |
//! | model | `/model name <name>` | `S0MA_LLM_MODEL` |
//! | API key | the overview's api key row | `S0MA_LLM_API_KEY` |
//! | type | `/model type local\|hosted` | `S0MA_LLM_SERVER` |
//! | context window | `/model context <tokens>\|auto` | `S0MA_LLM_CONTEXT` |
//! | embeddings | `/embeddings on\|off` | `S0MA_EMBEDDINGS` |
//!
//! **The key** goes with every request to the model server as
//! `Authorization: Bearer <key>`, and nowhere else: not to the embedding
//! server, which is always local, and never into the transcript, the archive
//! or a reply. It is typed in the overview, where it is never shown. Typed at
//! the prompt instead (`/model key <key>`), it is kept too, and its line is
//! taken out of the prompt's history (`carries_key`).
//!
//! **Local** (the default) is an engine on this machine or network: ds41d or
//! vLLM, with their extensions (template options, `top_k`, priorities, kept
//! prompts, `/tokenize`, `max_model_len`). **Hosted** is an OpenAI-compatible
//! API (OpenRouter, OpenAI, Together and the like): requests carry only the
//! standard fields, token counts are estimated, the context window is read
//! from `/v1/models` where the API lists it (or set), and the knowledge pass,
//! which needs a base model's log-probabilities, is off.
//!
//! **Embeddings off** leaves keyword search (SQLite FTS5) for the archive and
//! facts; semantic search, episode reminders and the per-turn skill match are
//! off, and say so once rather than failing every turn.

use serde::{Deserialize, Serialize};

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The local engine, as the recipe serves it.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8001";
pub const DEFAULT_MODEL: &str = "qwen38-nvfp4";
/// A hosted model's window when neither the setting nor the API says.
pub const ASSUMED_HOSTED_CONTEXT: usize = 131_072;

/// What `data/model.json` holds. Each is None until set.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Saved {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    server: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hosted: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embeddings: Option<bool>,
}

#[cfg(not(test))]
static SAVED: std::sync::RwLock<Saved> = std::sync::RwLock::new(Saved { server: None, model: None, api_key: None, hosted: None, context: None, embeddings: None });
// Under test each test's thread has its own, so one that sets a hosted API
// never changes the requests another test builds meanwhile.
#[cfg(test)]
thread_local! {
    static SAVED: std::cell::RefCell<Saved> = std::cell::RefCell::new(Saved::default());
}
static PATH: OnceLock<PathBuf> = OnceLock::new();

#[cfg(not(test))]
fn saved() -> Saved {
    SAVED.read().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
}
#[cfg(test)]
fn saved() -> Saved {
    SAVED.with(|saved| saved.borrow().clone())
}
#[cfg(not(test))]
fn replace(next: Saved) {
    *SAVED.write().unwrap_or_else(|poisoned| poisoned.into_inner()) = next;
}
#[cfg(test)]
fn replace(next: Saved) {
    SAVED.with(|saved| *saved.borrow_mut() = next);
}
/// One change at a time: two at once would each write over the other's.
static CHANGING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Where a setting's value came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    #[default]
    Default,
    Saved,
    Env,
}

impl Source {
    fn words(self) -> &'static str {
        match self {
            Source::Default => "default",
            Source::Saved => "set here",
            Source::Env => "from the environment",
        }
    }
}

/// Every setting as it stands, and where each came from.
#[derive(Clone, Debug, PartialEq)]
struct Resolved {
    server: (String, Source),
    model: (String, Source),
    api_key: (Option<String>, Source),
    hosted: (bool, Source),
    context: (Option<usize>, Source),
    embeddings: (bool, Source),
}

fn env_value(get: &dyn Fn(&str) -> Option<String>, key: &str) -> Option<String> {
    get(key).map(|value| value.trim().to_string()).filter(|value| !value.is_empty())
}

fn switch(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some(true),
        "0" | "off" | "false" | "no" => Some(false),
        _ => None,
    }
}

fn resolve(get: &dyn Fn(&str) -> Option<String>, saved: &Saved) -> Resolved {
    fn pick<T>(env: Option<T>, saved: Option<T>, default: T) -> (T, Source) {
        match (env, saved) {
            (Some(value), _) => (value, Source::Env),
            (None, Some(value)) => (value, Source::Saved),
            (None, None) => (default, Source::Default),
        }
    }
    let server = pick(env_value(get, "S0MA_LLM_BASE_URL"), saved.server.clone(), DEFAULT_BASE_URL.to_string());
    let hosted_env = env_value(get, "S0MA_LLM_SERVER").and_then(|value| match value.to_ascii_lowercase().as_str() {
        "hosted" => Some(true),
        "local" => Some(false),
        _ => None,
    });
    let context_env = env_value(get, "S0MA_LLM_CONTEXT").and_then(|value| value.replace(['_', ','], "").parse::<usize>().ok()).filter(|n| *n > 0);
    let api_key = match (env_value(get, "S0MA_LLM_API_KEY"), saved.api_key.clone()) {
        (Some(key), _) => (Some(key), Source::Env),
        (None, Some(key)) => (Some(key), Source::Saved),
        (None, None) => (None, Source::Default),
    };
    let context = match (context_env, saved.context) {
        (Some(n), _) => (Some(n), Source::Env),
        (None, Some(n)) => (Some(n), Source::Saved),
        (None, None) => (None, Source::Default),
    };
    Resolved {
        server: (normalize_server(&server.0), server.1),
        model: pick(env_value(get, "S0MA_LLM_MODEL"), saved.model.clone(), DEFAULT_MODEL.to_string()),
        api_key,
        hosted: pick(hosted_env, saved.hosted, false),
        context,
        embeddings: pick(env_value(get, "S0MA_EMBEDDINGS").and_then(|value| switch(&value)), saved.embeddings, true),
    }
}

/// The server's base, without a trailing `/` or `/v1`: requests add
/// `/v1/chat/completions` themselves, and an API's documented base often
/// ends in `/v1` (`https://api.openai.com/v1`).
fn normalize_server(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed).trim_end_matches('/').to_string()
}

fn now() -> Resolved {
    resolve(&|key| std::env::var(key).ok(), &saved())
}

/// Read what was kept, from `data/model.json`. A file others can read is
/// closed to them: it may hold the key.
pub fn init(data_dir: &Path) {
    let path = data_dir.join("model.json");
    if let Ok(text) = std::fs::read_to_string(&path) {
        match serde_json::from_str::<Saved>(&text) {
            Ok(saved) => replace(saved),
            Err(error) => crate::terminal::warn_status(&format!(
                "[model] {} does not read ({error}); the defaults and the environment apply until it is set again",
                path.display()
            )),
        }
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(&path)
            && metadata.permissions().mode() & 0o077 != 0
        {
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
    }
    let _ = PATH.set(path);
}

/// The model server's base URL (`http://127.0.0.1:8001`).
pub fn base_url() -> String {
    now().server.0
}

pub fn model() -> String {
    now().model.0
}

pub fn api_key() -> Option<String> {
    now().api_key.0
}

/// A hosted, OpenAI-compatible API rather than a local engine.
pub fn hosted() -> bool {
    now().hosted.0
}

/// The context window, when it was set.
pub fn context_window() -> Option<usize> {
    now().context.0
}

/// Whether embeddings are used (semantic search, episode reminders, the
/// skill match).
pub fn embeddings_on() -> bool {
    now().embeddings.0
}

/// What an embedding user's setup says when embeddings are off.
pub const EMBEDDINGS_OFF: &str = "embeddings are off";

/// `request` with the key, when there is one.
pub fn authorize(request: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
    match api_key() {
        Some(key) => request.bearer_auth(key),
        None => request,
    }
}

/// `authorize`, for the streamed calls' async client.
pub fn authorize_async(request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match api_key() {
        Some(key) => request.bearer_auth(key),
        None => request,
    }
}

/// The key for a process the agent starts that talks to the same server (the
/// mistake rewriter), as the environment variable it reads.
pub fn key_env() -> Option<(&'static str, String)> {
    api_key().map(|key| ("S0MA_LLM_API_KEY", key))
}

/// A key as it may be shown: never more than its last four characters, and
/// only those of a long one.
pub fn mask(key: &str) -> String {
    let count = key.chars().count();
    if count >= 16 {
        let tail: String = key.chars().skip(count - 4).collect();
        format!("set, ends …{tail}")
    } else {
        "set".to_string()
    }
}

/// Plain http to anywhere but this machine: a key would cross the network
/// readable.
fn exposed(server: &str, has_key: bool) -> bool {
    has_key
        && reqwest::Url::parse(server).is_ok_and(|url| {
            url.scheme() == "http" && !matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]" | "::1"))
        })
}

/// What the overview shows of the settings.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Shown {
    pub server: String,
    pub server_from: Source,
    pub model: String,
    pub model_from: Source,
    /// `set, ends …abcd`, or None.
    pub key: Option<String>,
    pub key_from: Source,
    pub hosted: bool,
    pub hosted_from: Source,
    pub context: Option<usize>,
    pub context_from: Source,
    pub embeddings: bool,
    pub embeddings_from: Source,
    /// Said when the key would cross the network unencrypted.
    pub warning: Option<String>,
}

pub fn shown() -> Shown {
    let now = now();
    Shown {
        warning: exposed(&now.server.0, now.api_key.0.is_some())
            .then(|| "the API key would be sent over plain http to another machine: use https".to_string()),
        server: now.server.0,
        server_from: now.server.1,
        model: now.model.0,
        model_from: now.model.1,
        key: now.api_key.0.as_deref().map(mask),
        key_from: now.api_key.1,
        hosted: now.hosted.0,
        hosted_from: now.hosted.1,
        context: now.context.0,
        context_from: now.context.1,
        embeddings: now.embeddings.0,
        embeddings_from: now.embeddings.1,
    }
}

/// `/model` and `/embeddings`, as the overview shows them.
pub fn status() -> String {
    let shown = shown();
    let from = |from: Source| format!(" ({})", from.words());
    let mut lines = vec![
        format!("[model] server {}{} · {}", shown.server, from(shown.server_from), if shown.hosted { "hosted API" } else { "local engine" }),
        format!("model {}{}", shown.model, from(shown.model_from)),
        format!("api key {}", shown.key.as_ref().map_or("not set".to_string(), |key| format!("{key}{}", from(shown.key_from)))),
        match shown.context {
            Some(n) => format!("context window {n} tokens{}", from(shown.context_from)),
            None if shown.hosted => format!("context window from the API's model list, or {ASSUMED_HOSTED_CONTEXT} tokens; /model context <tokens> sets it"),
            None => "context window from the engine".to_string(),
        },
        format!(
            "embeddings {}{}",
            if shown.embeddings { "on" } else { "off: keyword search only, no episode reminders or skill match" },
            from(shown.embeddings_from)
        ),
    ];
    if let Some(warning) = shown.warning {
        lines.push(format!("[!] {warning}"));
    }
    lines.join("\n        ")
}

/// A command whose line holds a key: `/model key <key>`. Its line is taken
/// out of the prompt's history once it is handled.
pub fn carries_key(command: &str) -> bool {
    matches!(command.split_whitespace().collect::<Vec<_>>().as_slice(), ["/model", "key", key] if *key != "clear")
}

/// The line as the page shows it: a key typed at the prompt as its last four
/// characters at most.
pub fn hide_key(line: &str) -> std::borrow::Cow<'_, str> {
    if !carries_key(line) {
        return std::borrow::Cow::Borrowed(line);
    }
    let key = line.split_whitespace().nth(2).unwrap_or_default();
    std::borrow::Cow::Owned(format!("/model key ({})", mask(key)))
}

fn save(change: impl FnOnce(&mut Saved)) -> Result<(), String> {
    let _one = crate::util::locked(&CHANGING);
    let mut next = saved();
    change(&mut next);
    if let Some(path) = PATH.get() {
        let text = serde_json::to_string_pretty(&next).map_err(|error| error.to_string())?;
        crate::util::write_private(path, text).map_err(|error| format!("could not keep it in {}: {error}", path.display()))?;
    }
    replace(next);
    Ok(())
}

/// What a saved change says, with the environment variable that still wins
/// over it, if one is set.
fn said(what: String, variable: &str) -> String {
    match std::env::var(variable).ok().filter(|value| !value.trim().is_empty()) {
        Some(_) => format!("{what}; {variable} is set and wins until it is unset"),
        None => what,
    }
}

/// The key, kept. From the overview's api key row, which never shows it.
pub fn set_key(key: &str) -> Result<String, String> {
    let key = key.trim();
    if key.is_empty() || key.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err("[model] not saved: a key is one word, with no spaces".into());
    }
    save(|saved| saved.api_key = Some(key.to_string()))?;
    let mut reply = said(format!("[model] api key saved ({})", mask(key)), "S0MA_LLM_API_KEY");
    if exposed(&base_url(), true) {
        reply.push_str(". [!] The server is plain http on another machine: the key crosses the network readable; use https");
    }
    Ok(reply)
}

/// A key on its way from the overview to the app (over the attach socket
/// from an attached terminal): never printed, even in a debug line.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Secret(pub String);

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(hidden)")
    }
}

/// `/model …` and `/embeddings …`; None for anything else.
pub fn handle_command(command: &str) -> Option<Result<String, String>> {
    let words: Vec<&str> = command.split_whitespace().collect();
    Some(match words.as_slice() {
        ["/model"] | ["/model", "status"] | ["/embeddings"] | ["/embeddings", "status"] => Ok(status()),
        ["/model", "server", "default"] => save(|saved| saved.server = None)
            .map(|()| said(format!("[model] server back to {DEFAULT_BASE_URL}"), "S0MA_LLM_BASE_URL")),
        ["/model", "server", url] => {
            let url = normalize_server(url);
            match reqwest::Url::parse(&url) {
                Ok(parsed) if matches!(parsed.scheme(), "http" | "https") && parsed.host_str().is_some() => {
                    save(|saved| saved.server = Some(url.clone())).map(|()| {
                        let mut reply = said(format!("[model] server {url}: from the next request"), "S0MA_LLM_BASE_URL");
                        if exposed(&url, api_key().is_some()) {
                            reply.push_str(". [!] Plain http to another machine: the API key would cross the network readable; use https");
                        }
                        reply
                    })
                }
                _ => Err(format!("[model] not a server address: {url} (an http or https URL, like https://openrouter.ai/api/v1)")),
            }
        }
        ["/model", "name", "default"] => {
            save(|saved| saved.model = None).map(|()| said(format!("[model] model back to {DEFAULT_MODEL}"), "S0MA_LLM_MODEL"))
        }
        ["/model", "name", name] => {
            save(|saved| saved.model = Some(name.to_string())).map(|()| said(format!("[model] model {name}: from the next request"), "S0MA_LLM_MODEL"))
        }
        ["/model", "type", kind @ ("local" | "hosted")] => {
            let hosted = *kind == "hosted";
            save(|saved| saved.hosted = Some(hosted)).map(|()| {
                said(
                    if hosted {
                        "[model] hosted API: requests carry only the standard fields, token counts are estimated, and the knowledge pass is off".to_string()
                    } else {
                        "[model] local engine: its extensions are used again, and the knowledge pass is back".to_string()
                    },
                    "S0MA_LLM_SERVER",
                )
            })
        }
        ["/model", "type", ..] => Err("[model] /model type local or /model type hosted".into()),
        ["/model", "key", "clear"] => save(|saved| saved.api_key = None).map(|()| said("[model] api key cleared".into(), "S0MA_LLM_API_KEY")),
        ["/model", "key", key] => set_key(key).map(|reply| format!("{reply}. Its line is out of your input history; the overview's api key row (Alt+O) never shows it")),
        ["/model", "key", ..] => Err("[model] set the key in the overview (Alt+O, api key), or /model key clear".into()),
        ["/model", "context", "auto" | "default"] => save(|saved| saved.context = None)
            .map(|()| said("[model] context window from the server again".into(), "S0MA_LLM_CONTEXT")),
        ["/model", "context", tokens] => match tokens.replace(['_', ','], "").parse::<usize>() {
            Ok(n) if n >= 4_096 => save(|saved| saved.context = Some(n)).map(|()| said(format!("[model] context window {n} tokens"), "S0MA_LLM_CONTEXT")),
            _ => Err("[model] a context window is a number of tokens, at least 4096 (or auto)".into()),
        },
        ["/embeddings", setting @ ("on" | "off")] => {
            let on = *setting == "on";
            save(|saved| saved.embeddings = Some(on)).map(|()| {
                said(
                    if on {
                        "[model] embeddings on: semantic search, episode reminders and the skill match use the local embedding server".to_string()
                    } else {
                        "[model] embeddings off: archive and fact search are keyword only; episode reminders and the skill match are off".to_string()
                    },
                    "S0MA_EMBEDDINGS",
                )
            })
        }
        ["/model", ..] => Err("[model] /model, /model server <url>|default, /model name <name>|default, /model type local|hosted, /model context <tokens>|auto, /model key clear".into()),
        ["/embeddings", ..] => Err("[model] /embeddings on or /embeddings off".into()),
        _ => return None,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |key| pairs.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_string())
    }

    #[test]
    fn the_environment_wins_over_what_is_kept_which_wins_over_the_default() {
        let none = env(&[]);
        let defaults = resolve(&none, &Saved::default());
        assert_eq!(defaults.server, (DEFAULT_BASE_URL.to_string(), Source::Default));
        assert_eq!(defaults.model, (DEFAULT_MODEL.to_string(), Source::Default));
        assert_eq!(defaults.api_key, (None, Source::Default));
        assert_eq!(defaults.hosted, (false, Source::Default));
        assert_eq!(defaults.embeddings, (true, Source::Default));

        let saved = Saved {
            server: Some("https://openrouter.ai/api/v1/".into()),
            model: Some("vendor/model".into()),
            api_key: Some("sk-saved".into()),
            hosted: Some(true),
            context: Some(200_000),
            embeddings: Some(false),
        };
        let kept = resolve(&none, &saved);
        assert_eq!(kept.server, ("https://openrouter.ai/api".to_string(), Source::Saved), "a documented /v1 base is taken as given");
        assert_eq!(kept.api_key, (Some("sk-saved".into()), Source::Saved));
        assert_eq!((kept.hosted, kept.context, kept.embeddings), ((true, Source::Saved), (Some(200_000), Source::Saved), (false, Source::Saved)));

        let over = env(&[
            ("S0MA_LLM_BASE_URL", "http://10.0.0.5:8001"),
            ("S0MA_LLM_API_KEY", " sk-env "),
            ("S0MA_LLM_SERVER", "local"),
            ("S0MA_LLM_CONTEXT", "32_768"),
            ("S0MA_EMBEDDINGS", "on"),
            ("S0MA_LLM_MODEL", "  "),
        ]);
        let wins = resolve(&over, &saved);
        assert_eq!(wins.server, ("http://10.0.0.5:8001".to_string(), Source::Env));
        assert_eq!(wins.api_key, (Some("sk-env".into()), Source::Env));
        assert_eq!(wins.hosted, (false, Source::Env));
        assert_eq!(wins.context, (Some(32_768), Source::Env));
        assert_eq!(wins.embeddings, (true, Source::Env));
        assert_eq!(wins.model, ("vendor/model".to_string(), Source::Saved), "a blank variable is not set");
        // A value the variable can't mean is not taken for one.
        assert_eq!(resolve(&env(&[("S0MA_LLM_SERVER", "cloud"), ("S0MA_EMBEDDINGS", "maybe")]), &saved).hosted, (true, Source::Saved));
    }
}
