//! Which conversation each model request belongs to, for ds41d's kept prompts.
//!
//! ds41d keeps every complete prompt's state, in RAM and on disk behind it, and
//! a request whose prompt starts with a kept one prefills only the rest
//! (`claude/ds41-subagent-kv-cache.md`, ds41d change 109). A request names its
//! conversation (`conversation_id`), and that decides two things only:
//! - **what a save erases:** a newer prompt of the same conversation that
//!   extends a kept one replaces it, so a subagent or a side call never erases
//!   the main turn's save;
//! - **what is ended together:** `POST /v1/conversations/end` drops a
//!   conversation's states, so a finished worker's don't sit in RAM and on
//!   disk until they age out.
//!
//! Lookups still match by tokens across conversations.
//!
//! The ids, `<role>:<name>:<uuid>`:
//!
//! | who | id |
//! |---|---|
//! | the main turn | `main:<session>` (`job:<job>:<uuid>` while a /job is open) |
//! | a subagent | `worker:<b3-1>:<uuid>`, a uuid per spawned worker |
//! | sleep and other maintenance | `sleep:<name>:<uuid>` |
//! | side calls on the main turn | `knowledge:<session>`, `router:<session>`, `judge:<session>` |
//!
//! The session's uuid is kept in the data folder, so a restart that takes the
//! working context back up (`context_resume`) keeps its id. A job's uuid is
//! kept there too while the job is open.
//!
//! Retention follows `priority` (lower is kept longer, and runs first), so the
//! main turn has the lowest number of everything that is saved: main 0,
//! saved side calls 1, workers 10, sleep and reranks 20. Knowledge probes keep
//! the main turn's 0: they are never saved, and the main turn waits for them.
//!
//! A conversation is ended (in the background, never while one of its requests
//! is running):
//! - a subagent, when its thread is done for good (finished, failed, killed);
//!   a pause is inside the thread, so it never ends one;
//! - a sleep call or pass, when it is over;
//! - any append-only conversation whose next prompt no longer starts with its
//!   last one (a fold, a compaction, a rebuilt runtime context, a new tool
//!   list), just before that request: nothing will match the old states again;
//! - a job's and the side calls', when the job closes;
//! - at exit: the side calls', any left, and the main turn's unless the working
//!   context was saved to be taken back up;
//! - at start: every `worker:` and `sleep:` left by an earlier run (one s0ma
//!   per ds41d).
//!
//! All of it only when the engine says it can (`ds41_features` in
//! `/v1/models` lists `conversation_end`); fne and vLLM say nothing, and get
//! what they got before.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::llm::ToolDefinition;
use crate::types::Message;

/// Retention and scheduling, lower first.
pub const MAIN_PRIORITY: i32 = 0;
pub const SIDE_PRIORITY: i32 = 1;
pub const WORKER_PRIORITY: i32 = 10;
pub const SLEEP_PRIORITY: i32 = 20;
/// The knowledge reranks (lane-1 traffic): with sleep, behind the workers.
pub const RERANK_PRIORITY: i32 = 20;

/// How long an end may take (the engine runs it between steps, and a long
/// prefill is one step).
const END_TIMEOUT: Duration = Duration::from_secs(600);
/// How long a request whose conversation was just ended waits for the end to
/// be sent first.
const END_FIRST: Duration = Duration::from_secs(2);
/// How long exit waits for ends still on their way.
const EXIT_WAIT: Duration = Duration::from_secs(5);
/// The side calls on the main turn, each one conversation per session.
const SIDE_ROLES: [&str; 3] = ["knowledge", "router", "judge"];
const FILE: &str = "conversations.json";

// ------------------------------------------------------------------ ids

/// What a conversation is, which sets its priority and whether its prompts
/// only ever grow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Main,
    Side,
    Worker,
    Sleep,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conversation {
    id: String,
    kind: Kind,
}

impl Conversation {
    /// The main turn: the session's, or the open job's.
    pub fn main() -> Self {
        let state = state();
        let id = match crate::jobs::open_job() {
            Some(job) => job_id(&job, &job_uuid(state, &job)),
            None => format!("main:{}", state.session),
        };
        Self { id, kind: Kind::Main }
    }

    /// A side call on the main turn (`knowledge`, `router`, `judge`).
    pub fn side(role: &str) -> Self {
        Self { id: format!("{}:{}", slug(role, 24), state().session), kind: Kind::Side }
    }

    /// A subagent: its shown name (`b3-1`) and its own uuid, made when it was
    /// spawned.
    pub fn worker(name: &str, uuid: &str) -> Self {
        Self { id: format!("worker:{}:{uuid}", slug(name, 40)), kind: Kind::Worker }
    }

    /// A sleep or maintenance call, with a uuid of its own.
    pub fn sleep(name: &str) -> Self {
        Self { id: format!("sleep:{}:{}", slug(name, 40), new_uuid()), kind: Kind::Sleep }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn priority(&self) -> i32 {
        match self.kind {
            Kind::Main => MAIN_PRIORITY,
            Kind::Side => SIDE_PRIORITY,
            Kind::Worker => WORKER_PRIORITY,
            Kind::Sleep => SLEEP_PRIORITY,
        }
    }

    /// Its prompts only grow, call after call: a prompt that doesn't start
    /// with the last one means the old states are of no more use.
    fn append_only(&self) -> bool {
        self.kind != Kind::Side
    }

    /// Run `body` with this as the thread's conversation: the model calls in
    /// it name it. With `end`, it is ended when the scope is left, if it sent
    /// anything.
    pub fn enter(&self, end: bool) -> Entered {
        let previous = CURRENT.with(|current| current.replace(Some(self.clone())));
        Entered { previous, conversation: self.clone(), end }
    }

    /// Ended in the background, once none of its requests is running.
    pub fn end_later(&self) {
        end_later(json!({"conversation_id": self.id}), Some(self.id.clone()));
    }
}

fn job_id(job: &str, uuid: &str) -> String {
    format!("job:{}:{uuid}", slug(job, 40))
}

/// `name` as an id's part: lowercase letters, digits and `-`, at most `max`.
pub fn slug(name: &str, max: usize) -> String {
    let mut out = String::new();
    for c in name.chars().flat_map(char::to_lowercase) {
        if out.len() >= max {
            break;
        }
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() { "x".into() } else { out }
}

/// A random (version 4) uuid.
pub fn new_uuid() -> String {
    let mut bytes = [0u8; 16];
    let read = std::fs::File::open("/dev/urandom").and_then(|mut file| std::io::Read::read_exact(&mut file, &mut bytes));
    if read.is_err() {
        // No /dev/urandom: the time, the process and a counter, mixed.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        for (index, chunk) in bytes.chunks_mut(8).enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (std::time::SystemTime::now(), std::process::id(), COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed), index)
                .hash(&mut hasher);
            chunk.copy_from_slice(&hasher.finish().to_le_bytes());
        }
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &hex[..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..])
}

// ------------------------------------------------------------------ the thread's conversation

thread_local! {
    static CURRENT: RefCell<Option<Conversation>> = const { RefCell::new(None) };
}

/// The scope `Conversation::enter` opened: left when dropped.
pub struct Entered {
    previous: Option<Conversation>,
    conversation: Conversation,
    end: bool,
}

impl Drop for Entered {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT.with(|current| *current.borrow_mut() = previous);
        if self.end && sent(&self.conversation.id) {
            self.conversation.end_later();
        }
    }
}

/// The thread's conversation, or `default()`; and whether it was the default
/// (a one-off its caller may end when its call returns).
pub fn current_or(default: impl FnOnce() -> Conversation) -> (Conversation, bool) {
    match CURRENT.with(|current| current.borrow().clone()) {
        Some(conversation) => (conversation, false),
        None => (default(), true),
    }
}

// ------------------------------------------------------------------ what the engine can do

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Features {
    pub priority: bool,
    pub conversation_id: bool,
    pub conversation_end: bool,
}

impl Features {
    /// From `/v1/models`: its `ds41_features`, wherever the server put them
    /// (the top level, or the served model's entry).
    pub fn from_models(response: &Value) -> Self {
        let models = response.get("data").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
        let listed: Vec<&str> = std::iter::once(response)
            .chain(models)
            .filter_map(|holder| holder.get("ds41_features").and_then(Value::as_array))
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        Self {
            priority: listed.contains(&"priority"),
            conversation_id: listed.contains(&"conversation_id"),
            conversation_end: listed.contains(&"conversation_end"),
        }
    }
}

/// Known for good once read; a failed read is tried again after a while
/// (s0ma may start before the engine).
static FEATURES: Mutex<Option<(Option<Features>, Instant)>> = Mutex::new(None);
const FEATURES_RETRY: Duration = Duration::from_secs(30);

pub fn features() -> Features {
    // Tests say per thread what the engine can do.
    #[cfg(test)]
    return test_features();
    #[allow(unreachable_code)]
    let mut known = FEATURES.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    match &*known {
        Some((Some(features), _)) => return features.clone(),
        Some((None, at)) if at.elapsed() < FEATURES_RETRY => return Features::default(),
        _ => {}
    }
    let read = read_features();
    *known = Some((read.clone(), Instant::now()));
    read.unwrap_or_default()
}

fn read_features() -> Option<Features> {
    let response: Value = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(10))
        .build()
        .ok()?
        .get(crate::llm::endpoint("/v1/models"))
        .send()
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .ok()?;
    Some(Features::from_models(&response))
}

/// Requests name their conversation, and ends are sent.
pub fn enabled() -> bool {
    features().conversation_end
}

/// Whether to send `priority`: the engine takes it (ds41d says so), or the
/// operator turned priority scheduling on for vLLM (`S0MA_LLM_PRIORITY=1`).
pub fn priority_enabled() -> bool {
    std::env::var("S0MA_LLM_PRIORITY").is_ok_and(|value| value.trim() == "1") || features().priority
}

/// What a request carries for `conversation`: its id when the engine takes
/// ids, and its priority when the engine takes that.
pub fn request_fields(conversation: &Conversation) -> (Option<String>, Option<i32>) {
    let id = enabled().then(|| conversation.id.clone());
    let priority = priority_enabled().then(|| conversation.priority());
    (id, priority)
}

// ------------------------------------------------------------------ requests in flight, and the prefix

#[derive(Default)]
struct Tracked {
    /// The last prompt sent: each message's hash, and the tools'.
    messages: Vec<u64>,
    tools: u64,
    running: usize,
}

struct State {
    session: String,
    /// Open jobs' uuids, by job.
    jobs: BTreeMap<String, String>,
    file: Option<PathBuf>,
    tracked: HashMap<String, Tracked>,
    /// Ends on their way, to wait for at exit.
    pending: Vec<std::thread::JoinHandle<()>>,
    /// The main turn's context was rebuilt at start: its old states go
    /// before its first request.
    end_main_first: bool,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

fn state() -> MutexGuard<'static, State> {
    STATE
        .get_or_init(|| {
            Mutex::new(State {
                session: new_uuid(),
                jobs: BTreeMap::new(),
                file: None,
                tracked: HashMap::new(),
                pending: Vec::new(),
                end_main_first: false,
            })
        })
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn job_uuid(mut state: MutexGuard<'static, State>, job: &str) -> String {
    if let Some(uuid) = state.jobs.get(job) {
        return uuid.clone();
    }
    let uuid = new_uuid();
    state.jobs.insert(job.to_string(), uuid.clone());
    save(&state);
    uuid
}

/// Whether `id` has sent a request in this run.
fn sent(id: &str) -> bool {
    state().tracked.contains_key(id)
}

fn hash_of<T: Serialize + ?Sized>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_vec(value).unwrap_or_default().hash(&mut hasher);
    hasher.finish()
}

/// A request of `conversation` is about to be sent: counted as running until
/// the guard drops. An append-only conversation whose prompt no longer starts
/// with its last one is ended first (and keeps its id).
pub fn begin_request(conversation: &Conversation, messages: &[Message], tools: Option<&[ToolDefinition]>) -> Running {
    let hashes: Vec<u64> = messages.iter().map(hash_of).collect();
    let tools = tools.map_or(0, hash_of);
    // Asked before the state is held: the first ask may wait on the engine.
    let can_end = enabled();
    let ended = {
        let mut state = state();
        let main_first = conversation.kind == Kind::Main && std::mem::take(&mut state.end_main_first);
        let tracked = state.tracked.entry(conversation.id.clone()).or_default();
        let broke = conversation.append_only()
            && !tracked.messages.is_empty()
            && (tracked.tools != tools || hashes.len() < tracked.messages.len() || hashes[..tracked.messages.len()] != tracked.messages[..]);
        tracked.messages = hashes;
        tracked.tools = tools;
        (can_end && (broke || main_first)).then(|| {
            // This request runs after the end; the old ones have returned.
            let body = json!({"conversation_id": conversation.id});
            spawn_end(&mut state, body, None)
        })
    };
    if let Some(done) = ended {
        // Sent before this request, as far as a short wait allows.
        let _ = done.recv_timeout(END_FIRST);
    }
    state().tracked.entry(conversation.id.clone()).or_default().running += 1;
    Running { id: conversation.id.clone() }
}

/// A request in flight; its end is not sent until it has returned.
pub struct Running {
    id: String,
}

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(tracked) = state().tracked.get_mut(&self.id) {
            tracked.running = tracked.running.saturating_sub(1);
        }
    }
}

fn running(id: &str) -> bool {
    state().tracked.get(id).is_some_and(|tracked| tracked.running > 0)
}

// ------------------------------------------------------------------ ends

/// `POST /v1/conversations/end` with `body`, in the background, after the
/// requests of `waits_for` (if given) have returned. Nothing when the engine
/// can't.
fn end_later(body: Value, waits_for: Option<String>) {
    if !enabled() {
        return;
    }
    let mut state = state();
    spawn_end(&mut state, body, waits_for);
}

/// The end's thread (the caller checked the engine takes ends); a receiver
/// that hears when it was sent.
fn spawn_end(held: &mut State, body: Value, waits_for: Option<String>) -> std::sync::mpsc::Receiver<()> {
    let (done, heard) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new().name("conversation-end".into()).spawn(move || {
        if let Some(id) = &waits_for {
            let since = Instant::now();
            while running(id) && since.elapsed() < END_TIMEOUT {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        post_end(&body);
        if let Some(id) = &waits_for {
            // Ended for good, or until it is used again, when it starts afresh.
            let mut state = state();
            if state.tracked.get(id).is_some_and(|tracked| tracked.running == 0) {
                state.tracked.remove(id);
            }
        }
        let _ = done.send(());
    });
    held.pending.retain(|handle| !handle.is_finished());
    if let Ok(handle) = spawned {
        held.pending.push(handle);
    }
    heard
}

#[cfg(test)]
fn post_end(body: &Value) {
    ENDS.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(body.clone());
}

#[cfg(not(test))]
fn post_end(body: &Value) {
    // Errors are ignored: a state not ended ages out as before.
    let _ = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(END_TIMEOUT)
        .build()
        .map(|client| client.post(crate::llm::endpoint("/v1/conversations/end")).json(body).send());
}

// ------------------------------------------------------------------ the session

#[derive(Debug, Default, Serialize, Deserialize)]
struct Saved {
    session: String,
    #[serde(default)]
    jobs: BTreeMap<String, String>,
}

fn save(state: &State) {
    let Some(file) = &state.file else { return };
    let saved = Saved { session: state.session.clone(), jobs: state.jobs.clone() };
    if let Ok(text) = serde_json::to_string_pretty(&saved) {
        // Lost only until the next save; s0ma goes on either way.
        let _ = crate::util::write_atomic(file, text);
    }
}

/// At start: the session's uuid (kept from before, or made now), and the
/// earlier run's leftover workers and sleep calls ended. `resumed`: the
/// working context was taken back up; otherwise it was rebuilt, and the main
/// turn's old states are ended before its first request.
pub fn init(data_dir: &Path, resumed: bool) {
    let file = data_dir.join(FILE);
    let saved: Option<Saved> = std::fs::read_to_string(&file).ok().and_then(|text| serde_json::from_str(&text).ok());
    {
        let mut state = state();
        if let Some(saved) = saved.filter(|saved| !saved.session.trim().is_empty()) {
            state.session = saved.session;
            state.jobs = saved.jobs;
        }
        // Jobs closed while s0ma was not running have nothing left to end by.
        let open = crate::jobs::open_job().into_iter().chain(crate::jobs::held_job()).collect::<Vec<_>>();
        state.jobs.retain(|job, _| open.contains(job));
        state.file = Some(file);
        state.end_main_first = !resumed;
        save(&state);
    }
    let start = || {
        for prefix in ["worker:", "sleep:"] {
            end_later(json!({"conversation_prefix": prefix}), None);
        }
    };
    if cfg!(test) {
        start();
    } else {
        // In the background: the engine may be slow to answer, or not up.
        let _ = std::thread::Builder::new().name("conversation-start".into()).spawn(start);
    }
}

/// A job closed: its conversation ended, and the side calls' (which saw it).
pub fn job_closed(job: &str) {
    let uuid = {
        let mut state = state();
        let uuid = state.jobs.remove(job);
        save(&state);
        uuid
    };
    if let Some(uuid) = uuid {
        let id = job_id(job, &uuid);
        end_later(json!({"conversation_id": id}), Some(id));
    }
    for role in SIDE_ROLES {
        Conversation::side(role).end_later();
    }
}

/// At exit: the side calls ended, and the main turn's unless the working
/// context is kept to be taken back up (`keep_main`); then a short wait for
/// the ends still on their way.
pub fn shutdown(keep_main: bool) {
    if enabled() {
        for role in SIDE_ROLES {
            Conversation::side(role).end_later();
        }
        if !keep_main {
            Conversation::main().end_later();
        }
        // Workers and sleep calls end themselves; any still known are ended.
        let left: Vec<String> = state()
            .tracked
            .keys()
            .filter(|id| id.starts_with("worker:") || id.starts_with("sleep:"))
            .cloned()
            .collect();
        for id in left {
            end_later(json!({"conversation_id": id}), Some(id));
        }
    }
    let pending = std::mem::take(&mut state().pending);
    let since = Instant::now();
    for handle in pending {
        while !handle.is_finished() && since.elapsed() < EXIT_WAIT {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

// ------------------------------------------------------------------ tests' hooks

#[cfg(test)]
static ENDS: Mutex<Vec<Value>> = Mutex::new(Vec::new());
#[cfg(test)]
thread_local! {
    static TEST_FEATURES: RefCell<Option<Features>> = const { RefCell::new(None) };
}
#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
fn test_features() -> Features {
    TEST_FEATURES.with(|features| features.borrow().clone()).unwrap_or_default()
}

/// Tests: what the engine says it can do, on this thread.
#[cfg(test)]
pub(crate) fn set_features_for_test(features: Option<Features>) {
    TEST_FEATURES.with(|known| *known.borrow_mut() = features);
}

/// Tests: the ends sent so far (waiting for those on their way), taken.
#[cfg(test)]
pub(crate) fn take_ends() -> Vec<Value> {
    let pending = std::mem::take(&mut state().pending);
    for handle in pending {
        let _ = handle.join();
    }
    std::mem::take(&mut *ENDS.lock().unwrap_or_else(|poisoned| poisoned.into_inner()))
}

#[cfg(test)]
pub(crate) fn ds41() -> Features {
    Features { priority: true, conversation_id: true, conversation_end: true }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock() -> MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn ids(ends: &[Value]) -> Vec<String> {
        ends.iter()
            .map(|end| end.get("conversation_id").or_else(|| end.get("conversation_prefix")).and_then(Value::as_str).unwrap_or("").to_string())
            .collect()
    }

    #[test]
    fn ids_have_their_form_and_a_uuid_per_worker() {
        let uuid = new_uuid();
        assert_eq!(uuid.len(), 36);
        assert_eq!(&uuid[14..15], "4", "version 4: {uuid}");
        assert!(matches!(&uuid[19..20], "8" | "9" | "a" | "b"), "the variant: {uuid}");
        assert_ne!(new_uuid(), uuid, "one each time");

        let worker = Conversation::worker("b3-1", &uuid);
        assert_eq!(worker.id(), format!("worker:b3-1:{uuid}"));
        let again = Conversation::worker("b3-1", &new_uuid());
        assert_ne!(again.id(), worker.id(), "a slot reused by a new worker gets a new id");
        assert_eq!(slug("Count the lines in the ACCESS logs!", 40), "count-the-lines-in-the-access-logs");
        assert_eq!(slug("", 40), "x");
        assert!(slug(&"long ".repeat(30), 40).len() <= 40);

        let session = state().session.clone();
        assert_eq!(Conversation::side("knowledge").id(), format!("knowledge:{session}"));
        assert_eq!(Conversation::side("router").id(), format!("router:{session}"), "side calls use the session's uuid");
        let sleep = Conversation::sleep("batch 3");
        assert!(sleep.id().starts_with("sleep:batch-3:") && sleep.id().len() < 128);
        for conversation in [&worker, &sleep, &Conversation::side("judge")] {
            let role = conversation.id().split(':').next().unwrap();
            assert!(role.len() <= 24 && role.chars().next().unwrap().is_ascii_lowercase(), "{role}");
            assert!(conversation.id().is_ascii());
        }
        assert!(Conversation::main().priority() < Conversation::side("router").priority());
        assert!(Conversation::side("router").priority() < worker.priority());
        assert!(worker.priority() < sleep.priority());
    }

    #[test]
    fn without_the_feature_nothing_is_named_or_ended() {
        let _lock = lock();
        set_features_for_test(None);
        let _ = take_ends();
        let worker = Conversation::worker("b1-1", &new_uuid());
        assert_eq!(request_fields(&worker).0, None, "no conversation_id");
        {
            let _scope = worker.enter(true);
            drop(begin_request(&worker, &[Message::user("a".into())], None));
            drop(begin_request(&worker, &[Message::user("b".into())], None));
        }
        worker.end_later();
        shutdown(false);
        assert!(take_ends().is_empty(), "no end calls");
    }

    #[test]
    fn a_worker_is_ended_once_after_its_last_request_and_a_fold_ends_it_first() {
        let _lock = lock();
        set_features_for_test(Some(ds41()));
        let _ = take_ends();
        let worker = Conversation::worker("b2-1", &new_uuid());
        assert_eq!(request_fields(&worker), (Some(worker.id().to_string()), Some(WORKER_PRIORITY)));
        let system = Message::system("you are a worker".into());
        let task = Message::user("the task".into());
        {
            let _scope = worker.enter(true);
            assert_eq!(current_or(|| Conversation::side("x")).0, worker, "the thread's conversation");
            // Growing: nothing ended.
            drop(begin_request(&worker, &[system.clone(), task.clone()], None));
            drop(begin_request(&worker, &[system.clone(), task.clone(), Message::user("result".into())], None));
            assert!(take_ends().is_empty(), "append-only: no end");
            // A compaction: the prompt no longer starts with the last one.
            let running = begin_request(&worker, &[system.clone(), Message::user("notes".into())], None);
            assert_eq!(ids(&take_ends()), [worker.id()], "one end, before the next request");
            // Still running when the scope is left: the end waits for it.
            let waiting = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                drop(running);
            });
            drop(_scope);
            assert!(ENDS.lock().unwrap().is_empty(), "not while its request runs");
            waiting.join().unwrap();
        }
        assert_eq!(ids(&take_ends()), [worker.id()], "once more when it is done for good");
        assert_eq!(current_or(|| Conversation::side("judge")).1, true, "the scope is left");

        // A paused worker is still inside its scope: nothing is ended.
        let paused = Conversation::worker("b2-2", &new_uuid());
        let scope = paused.enter(true);
        drop(begin_request(&paused, &[task.clone()], None));
        assert!(take_ends().is_empty());
        drop(scope);
        assert_eq!(ids(&take_ends()), [paused.id()]);

        // A worker that never sent anything has nothing to end.
        drop(Conversation::worker("b2-3", &new_uuid()).enter(true));
        assert!(take_ends().is_empty());
        set_features_for_test(None);
    }

    #[test]
    fn side_calls_never_end_each_other_and_start_ends_leftovers() {
        let _lock = lock();
        set_features_for_test(Some(ds41()));
        let _ = take_ends();
        let knowledge = Conversation::side("knowledge");
        // Probes differ only at the end, and run together: none ends another.
        let a = begin_request(&knowledge, &[Message::user("state, question a".into())], None);
        let b = begin_request(&knowledge, &[Message::user("state, question b".into())], None);
        drop((a, b));
        assert!(take_ends().is_empty());

        let dir = std::env::temp_dir().join(format!("s0ma-conversations-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        init(&dir, true);
        let mut started = ids(&take_ends());
        started.sort();
        assert_eq!(started, ["sleep:", "worker:"], "the earlier run's workers and sleep calls");
        let session = state().session.clone();
        let saved: Saved = serde_json::from_str(&std::fs::read_to_string(dir.join(FILE)).unwrap()).unwrap();
        assert_eq!(saved.session, session, "kept for a resumed session");
        init(&dir, true);
        assert_eq!(state().session, session, "the same after a restart");
        let _ = take_ends();

        // Rebuilt at start: the main turn's old states go before its first request.
        init(&dir, false);
        let _ = take_ends();
        let main = Conversation::main();
        assert_eq!(main.id(), format!("main:{session}"));
        drop(begin_request(&main, &[Message::user("first".into())], None));
        assert_eq!(ids(&take_ends()), [main.id()]);
        drop(begin_request(&main, &[Message::user("first".into()), Message::user("second".into())], None));
        assert!(take_ends().is_empty(), "once");

        // At exit, the working context kept: the side calls go, main stays.
        shutdown(true);
        let ended = ids(&take_ends());
        assert!(ended.contains(&format!("knowledge:{session}")) && ended.contains(&format!("judge:{session}")), "{ended:?}");
        assert!(!ended.contains(&main.id().to_string()), "kept for the restart");
        shutdown(false);
        assert!(ids(&take_ends()).contains(&main.id().to_string()));
        let _ = std::fs::remove_dir_all(&dir);
        set_features_for_test(None);
    }

    #[test]
    fn a_closed_job_ends_its_conversation_and_the_side_calls() {
        let _lock = lock();
        set_features_for_test(Some(ds41()));
        let _ = take_ends();
        let uuid = job_uuid(state(), "acme");
        assert_eq!(job_uuid(state(), "acme"), uuid, "one per job while it is open");
        let id = job_id("acme", &uuid);
        assert!(id.starts_with("job:acme:"));
        job_closed("acme");
        let ended = ids(&take_ends());
        assert!(ended.contains(&id), "{ended:?}");
        let session = state().session.clone();
        assert!(SIDE_ROLES.iter().all(|role| ended.contains(&format!("{role}:{session}"))), "they saw the job: {ended:?}");
        assert_ne!(job_uuid(state(), "acme"), uuid, "a job opened again is a new conversation");
        state().jobs.clear();
        set_features_for_test(None);
    }

    #[test]
    fn features_are_read_from_the_models_list() {
        let served = json!({"data": [{"id": "m", "max_model_len": 262144}], "ds41_features": ["priority", "conversation_id", "conversation_end"]});
        assert_eq!(Features::from_models(&served), ds41());
        let in_entry = json!({"data": [{"id": "m", "ds41_features": ["conversation_end"]}]});
        assert!(Features::from_models(&in_entry).conversation_end);
        let vllm = json!({"data": [{"id": "m", "max_model_len": 262144}]});
        assert_eq!(Features::from_models(&vllm), Features::default());
    }
}
