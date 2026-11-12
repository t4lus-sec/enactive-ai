//! What the overview tab (Alt+O, `tabs/overview_view.rs`) shows and asks.
//!
//! The agent's settings and state live in the app (`app.rs`): the reasoning
//! mode, looping, voice, the daily sleep, the operator's time zone, the goal, the
//! jobs, and the MCP servers. The tab may run in another process (an
//! attached terminal), so it never reaches them directly:
//!
//! - **Reading.** The tab asks (`want`), which wakes the prompt; the app,
//!   at the top of its input loop, builds a `Snapshot` and `publish`es it.
//!   The tab draws the latest one, with how old it is.
//! - **Changing.** The tab queues the slash command the operator would have typed
//!   (`queue`, e.g. `/mode low`), which wakes the prompt the same way; the
//!   app runs each through `handle_command`, as typed, and publishes again.
//!   During a turn they wait for it to end.
//! - **Always there.** The app also publishes as each turn starts and ends,
//!   so the tab has the state as of the last turn's edge while the agent works,
//!   and every snapshot is kept in `data/overview.json` (`init`), so a
//!   restarted the agent shows the last one at once, marked as from before.

use serde::{Deserialize, Serialize};

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Server {
    pub name: String,
    /// connected, disconnected, failed, disabled, config error, closed.
    pub state: String,
    pub detail: String,
    /// Each exposed tool, with its size when loaded.
    pub tools: Vec<String>,
}

/// The agent's context: its tokens and the model's window, or why they could not
/// be measured.
pub type Context = Result<(usize, usize), String>;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    /// When it was taken (the app's clock), for the tab to say how old.
    pub taken: String,
    /// The agent is in a turn: changes wait for it.
    pub busy: bool,
    /// `auto` or `forced xhigh`.
    pub reasoning: String,
    /// The subagents' level: `medium`.
    #[serde(default)]
    pub worker_reasoning: String,
    pub looping: bool,
    pub voice_on: bool,
    pub voice: String,
    /// The daily sleep's time (`02:00`), and what the schedule says.
    pub sleep_at: Option<String>,
    pub sleep: String,
    /// The operator's time zone, as `/tz` says it, and the zone itself if set.
    pub tz: String,
    pub tz_set: bool,
    pub tz_zone: Option<String>,
    /// As `/goal status` says it; whether there is one.
    pub goal: String,
    pub goal_set: bool,
    /// As `/job list` says it.
    pub jobs: String,
    pub mcp: Vec<Server>,
    /// The MCP config's place and its warnings.
    pub mcp_notes: Vec<String>,
    /// The agent's context: its tokens and the model's window, or why they
    /// could not be measured (`None`: not measured yet).
    #[serde(default)]
    pub context: Option<Context>,
    /// `mcp.json`'s path, for the tab to edit (`None`: MCP not set up).
    #[serde(default)]
    pub mcp_config: Option<String>,
    /// What the app said to the overview's last command (kept across
    /// snapshots; `seq` tells a new one).
    pub reply: Option<Reply>,
    /// Read back from `data/overview.json`: from before the agent restarted.
    #[serde(default)]
    pub restored: bool,
    /// Whether the agent may use its machine (`/vm on|off`).
    #[serde(default = "in_shells")]
    pub vm_access: bool,
    /// Where the agent's commands run: in shells the operator sees, or (his own
    /// setting) outside them.
    #[serde(default = "in_shells")]
    pub in_shells: bool,
    /// What runs on the agent's machine without a turn: its watches and
    /// background commands, as last listed (`watches`).
    #[serde(default)]
    pub watches: Watches,
    /// The model server, its key (masked) and the embeddings switch.
    #[serde(default)]
    pub model: crate::model_config::Shown,
    /// The agent works on this computer (host mode), not in its VM.
    #[serde(default)]
    pub machine_host: bool,
    /// What is set for the next start, when it differs.
    #[serde(default)]
    pub machine_next: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Reply {
    pub seq: u64,
    pub command: String,
    pub text: String,
    pub problem: bool,
}

struct Published {
    snapshot: Snapshot,
    at: Instant,
}

static PUBLISHED: Mutex<Option<Published>> = Mutex::new(None);
/// Where each snapshot is kept (`data/overview.json`).
static STORE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// The app's store for snapshots, and the last one kept there, shown until
/// the app publishes (asked again at once).
pub fn init(data_dir: &std::path::Path) {
    let path = data_dir.join("overview.json");
    let kept = std::fs::read_to_string(&path).ok().and_then(|text| serde_json::from_str::<Snapshot>(&text).ok());
    let _ = STORE.set(path);
    let Some(mut snapshot) = kept else { return };
    let mut published = lock(&PUBLISHED);
    if published.is_none() {
        // Its replies were numbered by the agent before; it was not in a turn
        // of this one.
        snapshot.reply = None;
        snapshot.busy = false;
        snapshot.restored = true;
        let at = Instant::now().checked_sub(FRESH_FOR).unwrap_or_else(Instant::now);
        *published = Some(Published { snapshot, at });
    }
}

/// Kept whole: written beside, then renamed over.
fn store(snapshot: &Snapshot) {
    let Some(path) = STORE.get() else { return };
    let Ok(text) = serde_json::to_string_pretty(snapshot) else { return };
    let _ = crate::util::write_atomic(path, text);
}
static WANTED: AtomicBool = AtomicBool::new(false);
static QUEUE: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Built again at most this often while the tab asks.
pub const FRESH_FOR: Duration = Duration::from_millis(800);

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The app's latest snapshot (the last reply kept when it has none).
pub fn publish(mut snapshot: Snapshot) {
    let mut published = lock(&PUBLISHED);
    if snapshot.reply.is_none() {
        snapshot.reply = published.as_ref().and_then(|published| published.snapshot.reply.clone());
    }
    store(&snapshot);
    *published = Some(Published { snapshot, at: Instant::now() });
    WANTED.store(false, Ordering::SeqCst);
    drop(published);
    // An overview open here draws it now.
    crate::tabs::poke();
}

static HEARD: Mutex<Option<Vec<(String, bool)>>> = Mutex::new(None);
static REPLIES: AtomicU64 = AtomicU64::new(0);

/// While the app runs one of the overview's commands: what it says is kept
/// for the tab (`heard`), as well as shown on the agent's page.
pub fn listen() {
    *lock(&HEARD) = Some(Vec::new());
}

/// What the app said (a no-op unless `listen`ing).
pub fn heard(text: &str, problem: bool) {
    if let Some(heard) = lock(&HEARD).as_mut() {
        heard.push((text.trim().to_string(), problem));
    }
}

/// What was said to `command`, as the reply the tab shows.
pub fn reply_to(command: &str) -> Reply {
    let heard = lock(&HEARD).take().unwrap_or_default();
    let problem = heard.iter().any(|(_, problem)| *problem);
    let text = heard.into_iter().map(|(text, _)| bare(&text)).collect::<Vec<_>>().join(" · ");
    Reply {
        seq: REPLIES.fetch_add(1, Ordering::SeqCst) + 1,
        command: command.to_string(),
        text: if text.is_empty() { "done".into() } else { text },
        problem,
    }
}

/// The tab wants a fresh snapshot: the app builds one when it next comes
/// round (the prompt wakes for it). What it has now, and how old that is.
pub fn want() -> Option<(Snapshot, Duration)> {
    let published = lock(&PUBLISHED);
    let stale = published.as_ref().is_none_or(|published| published.at.elapsed() >= FRESH_FOR);
    if stale {
        WANTED.store(true, Ordering::SeqCst);
    }
    published.as_ref().map(|published| (published.snapshot.clone(), published.at.elapsed()))
}

/// The tab asked for a snapshot, or queued a command: the prompt wakes, and
/// the app comes round.
pub fn pending() -> bool {
    WANTED.load(Ordering::SeqCst) || !lock(&QUEUE).is_empty()
}

pub fn wanted() -> bool {
    WANTED.load(Ordering::SeqCst)
}

/// A command for the app to run as typed (only the overview's own).
pub fn queue(command: &str) -> Result<(), String> {
    if !allowed(command) {
        return Err(format!("not a command the overview sends: {command}"));
    }
    // The machine's switch takes effect at once, mid-turn too (the app runs
    // queued commands only between turns); the app still says so.
    if command.split_whitespace().next() == Some("/vm") {
        crate::vm::access::handle_command(command);
    }
    lock(&QUEUE).push(command.to_string());
    Ok(())
}

pub fn take_commands() -> Vec<String> {
    std::mem::take(&mut *lock(&QUEUE))
}

/// What the overview may send: its settings, the goal's controls and MCP's
/// reload, nothing else (it reaches the app from attached terminals too).
pub fn allowed(command: &str) -> bool {
    let words: Vec<&str> = command.split_whitespace().collect();
    matches!(
        words.as_slice(),
        ["/mode", "auto" | "instruct" | "low" | "medium" | "xhigh"]
            | ["/mode", "workers", "instruct" | "low" | "medium" | "xhigh"]
            | ["/loop", "on" | "off"]
            | ["/voice", "on" | "off"]
            | ["/vm", "on" | "off"]
            | ["/sleep", "set", _]
            | ["/tz", _]
            | ["/goal", "pause" | "resume" | "clear"]
            | ["/mcp", "reload"]
            | ["/mcp", "reload", _]
            | ["/model", "server" | "name" | "context", _]
            | ["/model", "type", "local" | "hosted"]
            | ["/model", "key", "clear"]
            | ["/embeddings", "on" | "off"]
            | ["/machine", "host" | "vm"]
    )
}

/// `[mode] forced xhigh` → `forced xhigh`.
fn in_shells() -> bool {
    true
}

/// The agent's watches and background commands, as `vm_wake_list` last said.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Watches {
    /// Running ones: name, whether it repeats, its command (cut short).
    pub running: Vec<(String, bool, String)>,
    /// How many have ended (listed until cleared).
    pub ended: usize,
    /// Why they could not be listed; `None` when they were.
    pub error: Option<String>,
    /// `false` until a list has come back.
    pub listed: bool,
}

static WATCHES: Mutex<Option<(Instant, Watches)>> = Mutex::new(None);
static LISTING: AtomicBool = AtomicBool::new(false);

/// How often the watches are listed again while the overview asks.
const LIST_WATCHES_EVERY: Duration = Duration::from_secs(20);

/// The watches as last listed, and a fresh list asked for on a thread of its
/// own when that one is old (a guest command: never on the turn's thread).
pub fn watches() -> Watches {
    let kept = WATCHES.lock().ok().and_then(|kept| kept.clone());
    let stale = kept.as_ref().is_none_or(|(at, _)| at.elapsed() >= LIST_WATCHES_EVERY);
    if stale && crate::vm::wake::enabled() && !LISTING.swap(true, Ordering::SeqCst) {
        let spawned = std::thread::Builder::new().name("overview-watches".into()).spawn(|| {
            let listed = list_watches();
            if let Ok(mut kept) = WATCHES.lock() {
                *kept = Some((Instant::now(), listed));
            }
            LISTING.store(false, Ordering::SeqCst);
        });
        if spawned.is_err() {
            LISTING.store(false, Ordering::SeqCst);
        }
    }
    if !crate::vm::wake::enabled() {
        return Watches { error: Some("the wake channel is off: nothing can wake s0ma from its machine".into()), listed: true, ..Watches::default() };
    }
    kept.map(|(_, watches)| watches).unwrap_or_default()
}

/// `vm_wake_list`, as the tool runs it, read into `Watches`.
fn list_watches() -> Watches {
    let call = crate::types::ToolCall {
        id: "overview".into(),
        kind: "function".into(),
        function: crate::types::FunctionCall { name: "vm_wake_list".into(), arguments: "{}".into() },
    };
    let reply = match crate::vm::wake::execute(&call) {
        Ok(reply) => reply,
        Err(error) => return Watches { error: Some(error.to_string()), listed: true, ..Watches::default() },
    };
    let value: serde_json::Value = serde_json::from_str(&reply).unwrap_or_default();
    let mut watches = Watches { listed: true, ..Watches::default() };
    for watch in value["watches"].as_array().into_iter().flatten() {
        if watch["running"].as_bool() == Some(true) {
            watches.running.push((
                watch["name"].as_str().unwrap_or("?").to_string(),
                watch["repeat"].as_bool().unwrap_or(false),
                watch["command"].as_str().unwrap_or("").to_string(),
            ));
        } else {
            watches.ended += 1;
        }
    }
    watches
}

pub fn bare(text: &str) -> String {
    let text = text.trim();
    match text.strip_prefix('[').and_then(|rest| rest.split_once("] ")) {
        Some((_, rest)) => rest.to_string(),
        None => text.to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Held by every test that uses the app's queue and snapshot.
    pub(crate) static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn the_last_snapshot_is_kept_and_shown_at_once_after_a_restart() {
        let _lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = std::env::temp_dir().join(format!("s0ma-overview-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let before = Snapshot {
            taken: "09:15:00".into(),
            busy: true,
            reasoning: "forced low".into(),
            reply: Some(Reply { seq: 7, command: "/mode low".into(), text: "forced low".into(), problem: false }),
            ..Snapshot::default()
        };
        std::fs::write(dir.join("overview.json"), serde_json::to_string(&before).unwrap()).unwrap();
        // A restart: nothing published yet in this agent.
        *lock(&PUBLISHED) = None;
        init(&dir);
        let (shown, age) = want().expect("shown at once");
        assert_eq!((shown.reasoning.as_str(), shown.restored, shown.busy, shown.reply), ("forced low", true, false, None));
        assert!(age >= FRESH_FOR && wanted(), "and the app is asked for a fresh one");
        // Each one the app publishes is kept.
        publish(Snapshot { taken: "09:20:00".into(), reasoning: "auto".into(), ..Snapshot::default() });
        let kept: Snapshot = serde_json::from_str(&std::fs::read_to_string(dir.join("overview.json")).unwrap()).unwrap();
        assert_eq!((kept.reasoning.as_str(), kept.restored), ("auto", false));
        assert!(!dir.join("overview.json.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
