//! Who is running a machine call.
//!
//! A subagent's calls run with its run id set on the thread (`as_owner`), so
//! the machine tools can tell its calls from the agent's: they then run as the
//! subagents' account, in its own shell (`vm/netboxes.rs`).
//!
//! (It also kept which private shell session was whose, mirrored to attached
//! terminals. Every shell is a netbox now, named for its subagent, so nothing
//! is added to that any more; the snapshot and change are kept so terminals
//! of either version still attach.)

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex, MutexGuard, OnceLock,
    atomic::AtomicUsize,
};

#[derive(Default)]
struct State {
    /// VM shell session → the subagent it belongs to.
    sessions: BTreeMap<String, String>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn state() -> MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn with_state<T>(change: impl FnOnce(&mut State) -> T) -> T {
    let mut guard = state();
    change(guard.get_or_insert_with(State::default))
}

/// A change, as attached terminals apply it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Change {
    Session { session: String, run_id: String },
}

/// Everything, as an attached terminal starts from.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Snapshot {
    pub sessions: BTreeMap<String, String>,
}

type Sink = Box<dyn Fn(Change) + Send + Sync>;
static MIRROR: OnceLock<(Sink, Arc<AtomicUsize>)> = OnceLock::new();

/// Send every change to `sink` while `listening` is above zero. Once.
pub fn mirror_to(sink: Sink, listening: Arc<AtomicUsize>) {
    let _ = MIRROR.set((sink, listening));
}

thread_local! {
    /// The subagent whose call this thread is running, if any.
    static OWNER: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Run `call` as subagent `run_id`'s.
pub fn as_owner<T>(run_id: &str, call: impl FnOnce() -> T) -> T {
    let previous = OWNER.with(|owner| owner.replace(Some(run_id.to_string())));
    let result = call();
    OWNER.with(|owner| *owner.borrow_mut() = previous);
    result
}

/// The subagent whose call this thread is running, if any.
pub fn owner() -> Option<String> {
    OWNER.with(|owner| owner.borrow().clone())
}

/// Everything now, handed to `register` while it cannot change.
pub fn snapshot(register: impl FnOnce(Snapshot)) {
    let guard = state();
    let snapshot = guard.as_ref().map_or_else(Snapshot::default, |state| Snapshot { sessions: state.sessions.clone() });
    register(snapshot);
}

/// An attached terminal's copy, from `snapshot`.
pub fn restore(snapshot: Snapshot) {
    with_state(|state| state.sessions = snapshot.sessions);
}

/// A change the primary went through, made here.
pub fn apply(change: Change) {
    match change {
        Change::Session { session, run_id } => with_state(|state| {
            state.sessions.insert(session, run_id);
        }),
    }
}

