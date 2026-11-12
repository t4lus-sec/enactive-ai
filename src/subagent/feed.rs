//! What each subagent is doing as it does it, for the operator's tabs
//! (`tabs.rs`, Alt+2…).
//!
//! The worker journal is the durable record and is written a turn at a time;
//! this is the live one: the reasoning and text as they stream, each tool call
//! as it starts and its result as it comes back, directions as they are sent
//! and taken, pauses. Held in memory per slot, the last worker on each, so a
//! finished worker's tab stays readable until its slot is used again. Bounded
//! in size; nothing here is written anywhere. Every change is mirrored to the
//! terminals attached to this agent (`session.rs`), which keep a copy for
//! their own tabs.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

/// A feed keeps about this much text; the oldest items go first.
const FEED_CHARS: usize = 400_000;

/// Who sent a direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Sender {
    Operator,
    S0ma,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Item {
    Reasoning(String),
    Text(String),
    Call { name: String, arguments: String },
    Result(String),
    Direction { seq: u64, from: Sender, text: String, wrap_up: bool, delivered: bool },
    /// The harness's own line: a pause, a compaction, a retry.
    Note(String),
}

impl Item {
    fn size(&self) -> usize {
        match self {
            Item::Reasoning(text) | Item::Text(text) | Item::Result(text) | Item::Note(text) => text.len(),
            Item::Call { name, arguments } => name.len() + arguments.len(),
            Item::Direction { text, .. } => text.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Feed {
    pub run_id: String,
    pub slot: usize,
    pub task: String,
    pub started: Instant,
    /// When it stopped, and how ("completed", "cancelled", …).
    pub finished: Option<(Instant, String)>,
    pub items: VecDeque<Item>,
    /// When each item began and was last extended, beside `items`.
    pub times: VecDeque<(Instant, Instant)>,
    /// Directions sent and not taken in yet. They join `items` where the
    /// worker takes them (`delivered`), which is after whatever it is in the
    /// middle of: kept apart until then, they never split its text.
    pub waiting: Vec<Item>,
    /// Items dropped from the front to stay bounded.
    pub dropped: bool,
    /// Paused for a budget, and why ("paused: context"), until the agent decides.
    pub paused: Option<String>,
    chars: usize,
}

impl Feed {
    fn push(&mut self, item: Item) {
        let now = Instant::now();
        self.chars += item.size();
        self.items.push_back(item);
        self.times.push_back((now, now));
        self.trim();
    }

    fn trim(&mut self) {
        while self.chars > FEED_CHARS && self.items.len() > 1 {
            if let Some(item) = self.items.pop_front() {
                self.times.pop_front();
                self.chars -= item.size();
                self.dropped = true;
            }
        }
    }

    /// When item `index` began and was last extended (now, when unknown).
    pub fn time(&self, index: usize) -> (Instant, Instant) {
        self.times.get(index).copied().unwrap_or_else(|| (Instant::now(), Instant::now()))
    }

    /// Directions sent and not yet taken in.
    pub fn waiting_directions(&self) -> usize {
        self.waiting.len()
    }

    pub fn running(&self) -> bool {
        self.finished.is_none()
    }

    /// The task's first line, clipped, for the tab bar.
    pub fn label(&self, max: usize) -> String {
        let line = self.task.lines().map(str::trim).find(|line| !line.is_empty()).unwrap_or("");
        let mut label: String = line.chars().take(max).collect();
        if line.chars().count() > max {
            label.push('…');
        }
        label
    }
}

static FEEDS: Mutex<BTreeMap<usize, Feed>> = Mutex::new(BTreeMap::new());

/// A change to the feeds, as attached terminals apply it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Change {
    Begin { slot: usize, run_id: String, task: String },
    Stream { run_id: String, reasoning: bool, delta: String },
    Push { run_id: String, item: Item },
    Delivered { run_id: String, seqs: Vec<u64> },
    Withdraw { run_id: String, seq: u64 },
    Finish { run_id: String, status: String },
    Paused { run_id: String, paused: Option<String> },
    /// A finished worker's tab closed by the operator.
    Forget { run_id: String },
}

/// Every feed as an attached terminal starts from: ages in ms.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FeedSnapshot {
    pub slot: usize,
    pub run_id: String,
    pub task: String,
    pub started_ms: u64,
    pub finished: Option<(u64, String)>,
    pub items: Vec<Item>,
    pub waiting: Vec<Item>,
    pub dropped: bool,
    pub paused: Option<String>,
}

type Sink = Box<dyn Fn(Change) + Send + Sync>;
static MIRROR: std::sync::OnceLock<(Sink, std::sync::Arc<std::sync::atomic::AtomicUsize>)> = std::sync::OnceLock::new();

/// Send every change to `sink` while `listening` is above zero. Once.
pub fn mirror_to(sink: Sink, listening: std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    let _ = MIRROR.set((sink, listening));
}

/// Called with the feeds locked, so a snapshot and the changes after it
/// never overlap or leave a gap.
fn mirror(change: impl FnOnce() -> Change) {
    if let Some((sink, listening)) = MIRROR.get()
        && listening.load(Ordering::SeqCst) > 0
    {
        sink(change());
    }
}

/// The feeds now, handed to `register` while they cannot change; also what
/// `restore` takes.
pub fn snapshot(register: impl FnOnce(Vec<FeedSnapshot>)) {
    let feeds = feeds();
    let now = Instant::now();
    let ms = |at: Instant| now.saturating_duration_since(at).as_millis() as u64;
    register(
        feeds
            .values()
            .map(|feed| FeedSnapshot {
                slot: feed.slot,
                run_id: feed.run_id.clone(),
                task: feed.task.clone(),
                started_ms: ms(feed.started),
                finished: feed.finished.as_ref().map(|(at, status)| (ms(*at), status.clone())),
                items: feed.items.iter().cloned().collect(),
                waiting: feed.waiting.clone(),
                dropped: feed.dropped,
                paused: feed.paused.clone(),
            })
            .collect(),
    );
}

/// An attached terminal's copy of the feeds, from `snapshot`.
pub fn restore(snapshots: Vec<FeedSnapshot>) {
    let mut feeds = feeds();
    let now = Instant::now();
    let at = |ms: u64| now.checked_sub(Duration::from_millis(ms)).unwrap_or(now);
    feeds.clear();
    for snapshot in snapshots {
        let chars = snapshot.items.iter().map(Item::size).sum();
        feeds.insert(
            snapshot.slot,
            Feed {
                run_id: snapshot.run_id,
                slot: snapshot.slot,
                task: snapshot.task,
                started: at(snapshot.started_ms),
                finished: snapshot.finished.map(|(ms, status)| (at(ms), status)),
                times: snapshot.items.iter().map(|_| (now, now)).collect(),
                items: snapshot.items.into(),
                waiting: snapshot.waiting,
                dropped: snapshot.dropped,
                paused: snapshot.paused,
                chars,
            },
        );
    }
    drop(feeds);
    changed();
}

/// A change another agent's feeds went through, made here.
pub fn apply(change: Change) {
    match change {
        Change::Begin { slot, run_id, task } => begin(slot, &run_id, &task),
        Change::Stream { run_id, reasoning, delta } => stream(&run_id, reasoning, &delta),
        Change::Push { run_id, item } => push(&run_id, item),
        Change::Delivered { run_id, seqs } => delivered(&run_id, &seqs),
        Change::Withdraw { run_id, seq } => withdraw(&run_id, seq),
        Change::Finish { run_id, status } => finish(&run_id, &status),
        Change::Paused { run_id, paused: why } => paused(&run_id, why.as_deref()),
        Change::Forget { run_id } => {
            forget(&run_id);
        }
    }
}
/// Moves on with every change, so a view can tell cheaply whether to redraw.
static VERSION: AtomicU64 = AtomicU64::new(0);

fn feeds() -> std::sync::MutexGuard<'static, BTreeMap<usize, Feed>> {
    FEEDS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn changed() {
    VERSION.fetch_add(1, Ordering::SeqCst);
}

pub fn version() -> u64 {
    VERSION.load(Ordering::SeqCst)
}

/// A worker started on `slot`, replacing whatever the slot showed before.
pub fn begin(slot: usize, run_id: &str, task: &str) {
    let mut feeds = feeds();
    mirror(|| Change::Begin { slot, run_id: run_id.to_string(), task: task.to_string() });
    feeds.insert(
        slot,
        Feed {
            run_id: run_id.to_string(),
            slot,
            task: task.to_string(),
            started: Instant::now(),
            finished: None,
            items: VecDeque::new(),
            times: VecDeque::new(),
            waiting: Vec::new(),
            dropped: false,
            paused: None,
            chars: 0,
        },
    );
    drop(feeds);
    changed();
}

fn with_run(run_id: &str, change: impl FnOnce() -> Change, edit: impl FnOnce(&mut Feed)) {
    let mut feeds = feeds();
    if let Some(feed) = feeds.values_mut().find(|feed| feed.run_id == run_id) {
        mirror(change);
        edit(feed);
        drop(feeds);
        changed();
    }
}

/// More of what the worker is generating: reasoning, or its text. Joins the
/// last item when that is of the same kind.
pub fn stream(run_id: &str, reasoning: bool, delta: &str) {
    if delta.is_empty() {
        return;
    }
    let change = || Change::Stream { run_id: run_id.to_string(), reasoning, delta: delta.to_string() };
    with_run(run_id, change, |feed| {
        match (feed.items.back_mut(), reasoning) {
            (Some(Item::Reasoning(text)), true) | (Some(Item::Text(text)), false) => {
                text.push_str(delta);
                feed.chars += delta.len();
                if let Some((_, last)) = feed.times.back_mut() {
                    *last = Instant::now();
                }
                feed.trim();
            }
            (_, true) => feed.push(Item::Reasoning(delta.to_string())),
            (_, false) => feed.push(Item::Text(delta.to_string())),
        }
    });
}

/// A tool call, its result, a note; a direction waits apart until it is
/// taken (`delivered`).
pub fn push(run_id: &str, item: Item) {
    let mut feeds = feeds();
    let Some(feed) = feeds.values_mut().find(|feed| feed.run_id == run_id) else {
        return;
    };
    // Cloned only when a terminal is attached: a result can be large.
    mirror(|| Change::Push { run_id: run_id.to_string(), item: item.clone() });
    match item {
        Item::Direction { delivered: false, .. } => feed.waiting.push(item),
        item => feed.push(item),
    }
    drop(feeds);
    changed();
}

/// A direction that was refused (the worker's mailbox was full) after it was
/// put in the feed.
pub fn withdraw(run_id: &str, seq: u64) {
    with_run(run_id, || Change::Withdraw { run_id: run_id.to_string(), seq }, |feed| {
        feed.waiting.retain(|item| !matches!(item, Item::Direction { seq: waiting, .. } if *waiting == seq));
    });
}

/// The worker took these directions into its context: they go into the
/// feed here, in the order they were sent.
pub fn delivered(run_id: &str, seqs: &[u64]) {
    with_run(run_id, || Change::Delivered { run_id: run_id.to_string(), seqs: seqs.to_vec() }, |feed| {
        let (taken, still): (Vec<Item>, Vec<Item>) = std::mem::take(&mut feed.waiting)
            .into_iter()
            .partition(|item| matches!(item, Item::Direction { seq, .. } if seqs.contains(seq)));
        feed.waiting = still;
        for mut item in taken {
            if let Item::Direction { delivered, .. } = &mut item {
                *delivered = true;
            }
            feed.push(item);
        }
    });
}

pub fn finish(run_id: &str, status: &str) {
    with_run(run_id, || Change::Finish { run_id: run_id.to_string(), status: status.to_string() }, |feed| {
        if feed.finished.is_none() {
            feed.finished = Some((Instant::now(), status.to_string()));
            feed.paused = None;
        }
    });
}

/// Paused for a budget (`why`, such as "paused: context"), or not any more.
pub fn paused(run_id: &str, why: Option<&str>) {
    let change = || Change::Paused { run_id: run_id.to_string(), paused: why.map(str::to_string) };
    with_run(run_id, change, |feed| feed.paused = why.map(str::to_string));
}

/// Close a finished worker's tab: its feed is forgotten here (and on the
/// attached terminals). One still running is kept. Whether it went.
pub fn forget(run_id: &str) -> bool {
    let mut feeds = feeds();
    let Some(slot) = feeds.iter().find(|(_, feed)| feed.run_id == run_id && !feed.running()).map(|(slot, _)| *slot) else {
        return false;
    };
    mirror(|| Change::Forget { run_id: run_id.to_string() });
    feeds.remove(&slot);
    drop(feeds);
    changed();
    true
}

/// One tab's worth of summary, in slot order.
#[derive(Clone, Debug, PartialEq)]
pub struct Tab {
    pub slot: usize,
    pub run_id: String,
    pub label: String,
    pub running: bool,
    pub status: Option<String>,
}

pub fn tabs() -> Vec<Tab> {
    feeds()
        .values()
        .map(|feed| Tab {
            slot: feed.slot,
            run_id: feed.run_id.clone(),
            label: feed.label(60),
            running: feed.running(),
            status: feed.finished.as_ref().map(|(_, status)| status.clone()),
        })
        .collect()
}

/// Read `slot`'s feed without copying it.
pub fn with_feed<T>(slot: usize, read: impl FnOnce(&Feed) -> T) -> Option<T> {
    feeds().get(&slot).map(read)
}

#[cfg(test)]
pub(crate) fn clear_for_test() {
    feeds().clear();
    changed();
}

#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

