//! Liveness-based folding for a delegated worker paused for context
//! (`subagent_compact`), in place of the prose notes it used to write.
//!
//! A worker's transcript is read as a directed graph. Nodes are the task brief
//! and s0ma's directions (the active goal), the worker's notes (its assistant
//! text), and its tool calls with their results: board claims (intents), VM
//! results (by their `w<slot>.<n>` ref), board facts (committed results), and
//! the rest. Edges are "cites / derives from": a claimed intent reaches
//! everything done while it was held, up to the fact that commits it or the
//! kill that ends it; any node reaches what it names by id (a fact's evidence
//! refs, a rationale that mentions `w1.3`, a `result_read` of a ref).
//!
//! A node is HOT if it is reachable from the goal or from an intent still in
//! progress, and COLD otherwise. Reachability decides, never age or size.
//!
//! On compaction the hot region stays verbatim. Each maximal run of cold turns
//! is sent to the model once for a digest: a short conclusion of what it
//! established, in which any identifier, number, path or hash not found
//! verbatim in the run is dropped (`[dropped]`), never restated. The digest
//! replaces the run in the worker's view and keeps the exact messages it
//! replaced (they are also in the worker's event log), so:
//!
//! - `result_read` with the digest's id (`w<slot>.d<n>`) puts them back;
//! - when the worker claims an intent or commits a result, reachability is
//!   recomputed, and a digest that became reachable is replaced by its
//!   originals again: re-inflated, never re-run;
//! - digests are memoized by signature (sorted member keys + content hash), so
//!   an unchanged region is never summarized twice, and compacting twice with
//!   nothing new in between makes no model call.
//!
//! A model turn (an assistant message and its tool results) is the unit that
//! folds: a call can't be separated from its result, so a turn with any hot
//! node is kept whole.
//!
//! `S0MA_WORKER_COMPACT=notes` brings back the prose-notes compaction (the
//! rollback path, for one release). `S0MA_WORKER_FOLD_MAX_CALLS` caps the model
//! calls one compaction makes (default 8); past it, the oldest runs not yet
//! summarized stay verbatim.

use crate::types::Message;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::error::Error;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// The system prompt and the task: the goal, never folded.
const BRIEF: usize = 2;
/// A digest's output cap: a short conclusion, not a transcript.
pub(crate) const DIGEST_MAX_TOKENS: u32 = 768;
const DEFAULT_MAX_CALLS: usize = 8;
const DIGEST_MARKER: &str = "[HARNESS — FOLDED CONTEXT ";
const STEER_MARKER: &str = "[HARNESS — DIRECTION FROM s0ma";
/// What stands in for a value the digest named but its members don't hold.
pub(crate) const DROPPED: &str = "[dropped]";

pub(crate) const DIGEST_SYSTEM: &str = "You condense a finished stretch of your own earlier work into a short digest. \
It replaces that stretch in your context; the original messages stay retrievable.";

const DIGEST_INSTRUCTION: &str = "[HARNESS — FOLD DIGEST, NOT A NEW TASK]\n\
The messages below are a stretch of your earlier work that nothing you are doing now depends on. Write a short \
digest (at most a few sentences) of what it established: findings, the state it left, dead ends. Copy every \
identifier, ref (like w1.3), intent or fact id, number, path, hostname, version and hash exactly as it appears \
below; never restate a value you cannot copy from below: any value not found below is removed. Plain text, no \
tool calls, no preamble.";

/// Which compaction `subagent_compact` runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Strategy {
    /// The worker writes prose notes that replace its transcript (the old way).
    Notes,
    /// Liveness-based folding (this module).
    Liveness,
}

impl Strategy {
    pub(crate) fn from_env() -> Self {
        Self::parse(std::env::var("S0MA_WORKER_COMPACT").ok().as_deref())
    }

    fn parse(value: Option<&str>) -> Self {
        match value.map(|value| value.trim().to_ascii_lowercase()).as_deref() {
            Some("notes" | "prose" | "prose-notes") => Self::Notes,
            _ => Self::Liveness,
        }
    }
}

fn max_calls_from_env() -> usize {
    std::env::var("S0MA_WORKER_FOLD_MAX_CALLS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(DEFAULT_MAX_CALLS)
}

/// What a board call did to an intent.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Claim(String),
    Commit(String),
    Kill(String),
}

/// A digest in the worker's view, and what it replaced.
#[derive(Clone, Debug)]
struct Fold {
    id: String,
    /// The messages it replaced, exactly as they were: what revival puts back.
    originals: Vec<Message>,
    /// Ids its members can be cited by (refs, intent and fact ids).
    ids: Vec<String>,
    /// Its members' intent events, in order, so a fold never hides a commit.
    events: Vec<Event>,
    covers: String,
}

#[derive(Clone, Debug)]
struct Memo {
    id: String,
    text: String,
    dropped: Vec<String>,
}

/// One worker's folds and memo. Empty until its first compaction, and only
/// ever acts on a worker that has been folded.
#[derive(Clone, Debug)]
pub(crate) struct Folder {
    slot: usize,
    next: u64,
    max_calls: usize,
    /// Folds whose digest is in the worker's view, by digest id.
    folds: BTreeMap<String, Fold>,
    /// Digests by region signature.
    memo: HashMap<String, Memo>,
}

/// What one compaction did.
#[derive(Debug, Default)]
pub(crate) struct Report {
    pub hot_nodes: usize,
    pub cold_nodes: usize,
    /// Maximal cold runs found (each costs at most one model call).
    pub cold_chains: usize,
    pub model_calls: usize,
    pub folded: Vec<Folded>,
    /// Runs left verbatim because the call cap was reached: their member keys.
    pub kept_verbatim: Vec<Vec<String>>,
    /// Digests re-inflated because they had become reachable.
    pub revived: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct Folded {
    pub id: String,
    pub covers: String,
    pub memo_hit: bool,
    /// Values the digest named that none of its members hold.
    pub dropped: Vec<String>,
    pub messages: usize,
}

impl Folder {
    pub(crate) fn new(slot: usize) -> Self {
        Self::with_max_calls(slot, max_calls_from_env())
    }

    pub(crate) fn with_max_calls(slot: usize, max_calls: usize) -> Self {
        Self { slot, next: 0, max_calls, folds: BTreeMap::new(), memo: HashMap::new() }
    }

    /// Take the digests `from` paid for (and its id counter, so ids never
    /// repeat), without its view: what a compaction that failed afterwards
    /// keeps.
    pub(crate) fn keep_memo(&mut self, from: &Folder) {
        self.memo = from.memo.clone();
        self.next = from.next;
    }

    /// Whether any digest is in the view: the only state in which a claim,
    /// commit or `result_read` has anything to revive.
    pub(crate) fn active(&self) -> bool {
        !self.folds.is_empty()
    }

    pub(crate) fn is_digest(&self, id: &str) -> bool {
        self.folds.contains_key(id)
    }

    /// The compact handler. `refs` maps a tool call id to its result ref;
    /// `digest` makes one model call for one cold run (its rendered members)
    /// and returns the digest text.
    pub(crate) fn compact(
        &mut self,
        messages: &mut Vec<Message>,
        refs: &HashMap<String, String>,
        digest: &mut dyn FnMut(&str) -> Result<String>,
    ) -> Result<Report> {
        let mut report = Report { revived: self.revive_hot(messages, refs), ..Report::default() };
        let graph = Graph::build(messages, refs, &self.folds);
        let hot = graph.hot();
        report.hot_nodes = hot.iter().filter(|hot| **hot).count();
        report.cold_nodes = hot.len() - report.hot_nodes;
        let unit_hot: Vec<bool> = graph.units.iter().map(|unit| unit.nodes.iter().any(|n| hot[*n])).collect();

        // Maximal runs of cold turns and messages, broken by anything hot, by
        // the goal and by digests already in place (each its own region).
        let mut chains: Vec<Vec<usize>> = Vec::new();
        let mut current = Vec::new();
        for (index, unit) in graph.units.iter().enumerate() {
            let breaks = unit_hot[index] || matches!(unit.kind, UnitKind::Brief | UnitKind::Steer | UnitKind::Digest(_));
            if breaks {
                if !current.is_empty() {
                    chains.push(std::mem::take(&mut current));
                }
            } else {
                current.push(index);
            }
        }
        if !current.is_empty() {
            chains.push(current);
        }
        report.cold_chains = chains.len();

        let signatures: Vec<String> = chains.iter().map(|chain| graph.signature(messages, chain)).collect();
        let misses: Vec<usize> = (0..chains.len()).filter(|c| !self.memo.contains_key(&signatures[*c])).collect();
        // Over the cap: the oldest runs not yet summarized stay verbatim.
        let over = misses.len().saturating_sub(self.max_calls);
        let verbatim: HashSet<usize> = misses.iter().take(over).copied().collect();

        let mut folded_chains: HashMap<usize, (String, Memo, bool)> = HashMap::new();
        for (c, chain) in chains.iter().enumerate() {
            if verbatim.contains(&c) {
                report.kept_verbatim.push(chain.iter().flat_map(|u| graph.units[*u].nodes.iter().map(|n| graph.nodes[*n].key.clone())).collect());
                continue;
            }
            let signature = &signatures[c];
            let (memo, hit) = match self.memo.get(signature) {
                Some(memo) => (memo.clone(), true),
                None => {
                    let rendered = graph.render(messages, chain);
                    let raw = digest(&rendered)?;
                    report.model_calls += 1;
                    let (text, dropped) = verified(raw.trim(), &graph.source(messages, chain));
                    self.next += 1;
                    let memo = Memo { id: format!("w{}.d{}", self.slot, self.next), text, dropped };
                    self.memo.insert(signature.clone(), memo.clone());
                    (memo, false)
                }
            };
            folded_chains.insert(chain[0], (signature.clone(), memo, hit));
        }

        // The view again: digests where their runs were, everything else as it was.
        let mut chain_of: HashMap<usize, usize> = HashMap::new();
        for (c, chain) in chains.iter().enumerate() {
            if folded_chains.contains_key(&chain[0]) {
                for unit in chain {
                    chain_of.insert(*unit, c);
                }
            }
        }
        let mut view = Vec::with_capacity(messages.len());
        for (index, unit) in graph.units.iter().enumerate() {
            match chain_of.get(&index) {
                Some(c) if chains[*c][0] == index => {
                    let chain = &chains[*c];
                    let (_, memo, hit) = &folded_chains[&index];
                    let originals: Vec<Message> =
                        chain.iter().flat_map(|u| graph.units[*u].messages.iter().map(|m| messages[*m].clone())).collect();
                    let (ids, events) = graph.ids_and_events(chain);
                    let covers = graph.covers(chain);
                    view.push(Message::user(digest_message(&memo.id, &covers, &memo.text)));
                    report.folded.push(Folded {
                        id: memo.id.clone(),
                        covers: covers.clone(),
                        memo_hit: *hit,
                        dropped: memo.dropped.clone(),
                        messages: originals.len(),
                    });
                    self.folds.insert(memo.id.clone(), Fold { id: memo.id.clone(), originals, ids, events, covers });
                }
                Some(_) => {}
                None => view.extend(unit.messages.iter().map(|m| messages[*m].clone())),
            }
        }
        *messages = view;
        Ok(report)
    }

    /// After the worker claims an intent or commits a result: re-inflate any
    /// digest that has become reachable. No model call, no tool call.
    pub(crate) fn revive(&mut self, messages: &mut Vec<Message>, refs: &HashMap<String, String>) -> Vec<String> {
        if self.folds.is_empty() {
            return Vec::new();
        }
        self.revive_hot(messages, refs)
    }

    /// `result_read` on a digest id: its originals go back where it was.
    pub(crate) fn expand(&mut self, messages: &mut Vec<Message>, id: &str) -> Option<Value> {
        let at = messages.iter().position(|m| m.role == "user" && digest_id(&m.content) == Some(id))?;
        let fold = self.folds.remove(id)?;
        let restored = fold.originals.len();
        messages.splice(at..at + 1, fold.originals);
        Some(json!({"ok": true, "expanded": id, "covers": fold.covers, "restored_messages": restored,
            "note": "The original messages are back in your context where the digest was."}))
    }

    fn revive_hot(&mut self, messages: &mut Vec<Message>, refs: &HashMap<String, String>) -> Vec<String> {
        let mut revived = Vec::new();
        loop {
            let graph = Graph::build(messages, refs, &self.folds);
            let hot = graph.hot();
            let reachable: Vec<(usize, String)> = graph
                .units
                .iter()
                .filter_map(|unit| match &unit.kind {
                    UnitKind::Digest(id) if unit.nodes.iter().any(|n| hot[*n]) => Some((unit.messages[0], id.clone())),
                    _ => None,
                })
                .collect();
            if reachable.is_empty() {
                return revived;
            }
            // From the back, so earlier positions stay valid.
            for (at, id) in reachable.into_iter().rev() {
                if let Some(fold) = self.folds.remove(&id) {
                    messages.splice(at..at + 1, fold.originals);
                    revived.push(id);
                }
            }
        }
    }
}

fn digest_message(id: &str, covers: &str, text: &str) -> String {
    format!(
        "{DIGEST_MARKER}{id}, NOT A NEW TASK]\ncovers: {covers}\n{text}\n\
(This stretch is done and folded. result_read {{\"ref\": \"{id}\"}} puts its original messages back here.)"
    )
}

/// The id a digest message carries.
fn digest_id(content: &str) -> Option<&str> {
    let rest = content.strip_prefix(DIGEST_MARKER)?;
    Some(rest.split_once(',')?.0)
}

// ------------------------------------------------------------------ graph

#[derive(Debug, Clone, PartialEq, Eq)]
enum UnitKind {
    /// The system prompt or the task.
    Brief,
    /// A direction from s0ma: part of the active goal.
    Steer,
    /// A digest in place.
    Digest(String),
    /// A model turn: an assistant message and its tool results.
    Turn,
    /// Any other message (a harness notice, an earlier summary).
    Other,
}

/// What folds or stays as one: whole messages.
struct Unit {
    kind: UnitKind,
    messages: Vec<usize>,
    nodes: Vec<usize>,
}

struct Node {
    /// Stable across compactions: the call id for a call.
    key: String,
    /// Ids others cite it by.
    ids: Vec<String>,
    /// Ids it cites.
    cites: Vec<String>,
    events: Vec<Event>,
    root: bool,
}

struct Graph {
    units: Vec<Unit>,
    nodes: Vec<Node>,
}

impl Graph {
    fn build(messages: &[Message], refs: &HashMap<String, String>, folds: &BTreeMap<String, Fold>) -> Self {
        let mut graph = Graph { units: Vec::new(), nodes: Vec::new() };
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut occurrence = |prefix: &str, content: &str| {
            let base = format!("{prefix}:{}", short_hash(content));
            let n = seen.entry(base.clone()).or_default();
            *n += 1;
            format!("{base}:{n}")
        };
        let mut index = 0;
        while index < messages.len() {
            let message = &messages[index];
            let unit = graph.units.len();
            if index < BRIEF {
                graph.push_unit(UnitKind::Brief, vec![index], vec![Node {
                    key: format!("brief:{index}"), ids: Vec::new(), cites: ids_in(&message.content), events: Vec::new(), root: true,
                }]);
                index += 1;
                continue;
            }
            match message.role.as_str() {
                "assistant" => {
                    let calls = message.tool_calls.clone().unwrap_or_default();
                    let call_ids: HashSet<&str> = calls.iter().map(|call| call.id.as_str()).collect();
                    let mut members = vec![index];
                    let mut next = index + 1;
                    while next < messages.len()
                        && messages[next].role == "tool"
                        && messages[next].tool_call_id.as_deref().is_some_and(|id| call_ids.contains(id))
                    {
                        members.push(next);
                        next += 1;
                    }
                    let mut nodes = Vec::new();
                    if !message.content.trim().is_empty() {
                        nodes.push(Node {
                            key: occurrence("note", &message.content),
                            ids: Vec::new(),
                            cites: ids_in(&message.content),
                            events: Vec::new(),
                            root: false,
                        });
                    }
                    for (position, call) in calls.iter().enumerate() {
                        let result = members[1..]
                            .iter()
                            .map(|m| &messages[*m])
                            .find(|tool| tool.tool_call_id.as_deref() == Some(call.id.as_str()))
                            .map(|tool| tool.content.as_str())
                            .unwrap_or("");
                        nodes.push(call_node(call, position, unit, result, refs));
                    }
                    graph.push_unit(UnitKind::Turn, members, nodes);
                    index = next;
                }
                "user" => {
                    let (kind, node) = match digest_id(&message.content).and_then(|id| folds.get(id)) {
                        Some(fold) => {
                            let mut ids = fold.ids.clone();
                            ids.push(fold.id.clone());
                            (UnitKind::Digest(fold.id.clone()),
                                Node { key: fold.id.clone(), ids, cites: Vec::new(), events: fold.events.clone(), root: false })
                        }
                        None if message.content.starts_with(STEER_MARKER) => (UnitKind::Steer, Node {
                            key: occurrence("steer", &message.content), ids: Vec::new(),
                            cites: ids_in(&message.content), events: Vec::new(), root: true,
                        }),
                        None => (UnitKind::Other, Node {
                            key: occurrence("user", &message.content), ids: Vec::new(),
                            cites: ids_in(&message.content), events: Vec::new(), root: false,
                        }),
                    };
                    graph.push_unit(kind, vec![index], vec![node]);
                    index += 1;
                }
                _ => {
                    graph.push_unit(UnitKind::Other, vec![index], vec![Node {
                        key: occurrence("message", &message.content), ids: Vec::new(),
                        cites: Vec::new(), events: Vec::new(), root: false,
                    }]);
                    index += 1;
                }
            }
        }
        graph
    }

    fn push_unit(&mut self, kind: UnitKind, messages: Vec<usize>, nodes: Vec<Node>) {
        let first = self.nodes.len();
        let count = nodes.len();
        self.nodes.extend(nodes);
        self.units.push(Unit { kind, messages, nodes: (first..first + count).collect() });
    }

    /// Which nodes are reachable from the goal or an intent in progress.
    fn hot(&self) -> Vec<bool> {
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        // An intent reaches what was done while it was held, its commit included.
        let mut held: Option<(String, usize)> = None;
        let mut state: HashMap<String, (bool, usize)> = HashMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if let Some((_, claim)) = &held
                && *claim != index {
                    edges[*claim].push(index);
                }
            for event in &node.events {
                match event {
                    Event::Claim(intent) => {
                        held = Some((intent.clone(), index));
                        state.insert(intent.clone(), (true, index));
                    }
                    Event::Commit(intent) | Event::Kill(intent) => {
                        if held.as_ref().is_some_and(|(held, _)| held == intent) {
                            held = None;
                        }
                        state.insert(intent.clone(), (false, index));
                    }
                }
            }
        }
        // Cites: a node reaches every node that answers to an id it names.
        let mut by_id: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            for id in &node.ids {
                by_id.entry(id.as_str()).or_default().push(index);
            }
        }
        for (index, node) in self.nodes.iter().enumerate() {
            for cited in &node.cites {
                for target in by_id.get(cited.as_str()).into_iter().flatten() {
                    if *target != index {
                        edges[index].push(*target);
                    }
                }
            }
        }
        let mut hot = vec![false; self.nodes.len()];
        let mut queue: VecDeque<usize> = self.nodes.iter().enumerate().filter(|(_, n)| n.root).map(|(i, _)| i).collect();
        queue.extend(state.values().filter(|(in_progress, _)| *in_progress).map(|(_, at)| *at));
        while let Some(index) = queue.pop_front() {
            if std::mem::replace(&mut hot[index], true) {
                continue;
            }
            queue.extend(edges[index].iter().copied().filter(|n| !hot[*n]));
        }
        hot
    }

    /// Sorted member keys and a hash of the messages: an unchanged region
    /// has the same signature at every compaction.
    fn signature(&self, messages: &[Message], chain: &[usize]) -> String {
        let mut keys: Vec<&str> =
            chain.iter().flat_map(|u| self.units[*u].nodes.iter().map(|n| self.nodes[*n].key.as_str())).collect();
        keys.sort_unstable();
        let mut hasher = Sha256::new();
        for unit in chain {
            for message in &self.units[*unit].messages {
                hasher.update(serde_json::to_string(&messages[*message]).unwrap_or_default().as_bytes());
                hasher.update(b"\n");
            }
        }
        format!("{}|{:x}", keys.join(","), hasher.finalize())
    }

    /// A run's messages as the digest call reads them.
    fn render(&self, messages: &[Message], chain: &[usize]) -> String {
        let mut text = String::new();
        for unit in chain {
            for index in &self.units[*unit].messages {
                let message = &messages[*index];
                match message.role.as_str() {
                    "assistant" => {
                        if !message.content.trim().is_empty() {
                            text.push_str(&format!("[your note]\n{}\n", message.content.trim()));
                        }
                        for call in message.tool_calls.iter().flatten() {
                            text.push_str(&format!("[your call] {} {}\n", call.function.name, call.function.arguments));
                        }
                    }
                    "tool" => text.push_str(&format!("[result]\n{}\n", message.content.trim())),
                    _ => text.push_str(&format!("[harness]\n{}\n", message.content.trim())),
                }
            }
        }
        format!("{DIGEST_INSTRUCTION}\n\n{text}")
    }

    /// Everything a digest's values may be copied from.
    fn source(&self, messages: &[Message], chain: &[usize]) -> String {
        let mut text = String::new();
        for unit in chain {
            for index in &self.units[*unit].messages {
                let message = &messages[*index];
                text.push_str(&message.content);
                text.push('\n');
                for call in message.tool_calls.iter().flatten() {
                    text.push_str(&call.function.name);
                    text.push(' ');
                    text.push_str(&call.function.arguments);
                    text.push('\n');
                }
            }
        }
        for unit in chain {
            for node in &self.units[*unit].nodes {
                for id in &self.nodes[*node].ids {
                    text.push_str(id);
                    text.push('\n');
                }
            }
        }
        text
    }

    fn ids_and_events(&self, chain: &[usize]) -> (Vec<String>, Vec<Event>) {
        let mut ids = Vec::new();
        let mut events = Vec::new();
        for unit in chain {
            for node in &self.units[*unit].nodes {
                for id in &self.nodes[*node].ids {
                    if !ids.contains(id) {
                        ids.push(id.clone());
                    }
                }
                events.extend(self.nodes[*node].events.iter().cloned());
            }
        }
        (ids, events)
    }

    /// `w1.1, intent-0001, fact-0001; 2 notes, 1 other call`.
    fn covers(&self, chain: &[usize]) -> String {
        let (ids, _) = self.ids_and_events(chain);
        let (mut notes, mut calls, mut other) = (0, 0, 0);
        for unit in chain {
            for node in &self.units[*unit].nodes {
                let node = &self.nodes[*node];
                if node.key.starts_with("note:") {
                    notes += 1;
                } else if node.key.starts_with("call:") {
                    if node.ids.is_empty() {
                        calls += 1;
                    }
                } else {
                    other += 1;
                }
            }
        }
        let mut counts = Vec::new();
        for (count, one, many) in [(notes, "note", "notes"), (calls, "other call", "other calls"), (other, "other message", "other messages")] {
            if count > 0 {
                counts.push(format!("{count} {}", if count == 1 { one } else { many }));
            }
        }
        match (ids.is_empty(), counts.is_empty()) {
            (false, false) => format!("{}; {}", ids.join(", "), counts.join(", ")),
            (false, true) => ids.join(", "),
            (true, _) => counts.join(", "),
        }
    }
}

/// One tool call and its result as a node: what it can be cited by, what it
/// cites, and what it did to an intent.
fn call_node(call: &crate::types::ToolCall, position: usize, unit: usize, result: &str, refs: &HashMap<String, String>) -> Node {
    let key = if call.id.is_empty() { format!("call:{unit}:{position}") } else { format!("call:{}", call.id) };
    let arguments: Value = serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
    let parsed: Value = serde_json::from_str(result).unwrap_or(Value::Null);
    let ok = parsed["ok"].as_bool() == Some(true);
    let intent = arguments["intent_id"].as_str().map(|id| id.trim().to_string());
    let mut ids: Vec<String> = refs.get(&call.id).cloned().into_iter().collect();
    let mut cites = ids_in(&call.function.arguments);
    let mut events = Vec::new();
    match call.function.name.as_str() {
        "board_claim" => {
            // The claimed intent as described: its rationale may cite earlier work.
            cites.extend(ids_in(result));
            if let (true, Some(intent)) = (ok, intent) {
                ids.push(intent.clone());
                events.push(Event::Claim(intent));
            }
        }
        "board_fact" if ok => {
            if let Some(fact) = parsed["fact"]["id"].as_str() {
                ids.push(fact.to_string());
            }
            if let Some(intent) = intent {
                events.push(Event::Commit(intent));
            }
        }
        "board_kill" if ok => {
            if let Some(intent) = intent {
                events.push(Event::Kill(intent));
            }
        }
        "board_intent" if ok => {
            for declared in parsed["declared"].as_array().into_iter().flatten() {
                if let Some(id) = declared["id"].as_str() {
                    ids.push(id.to_string());
                }
            }
        }
        "result_read" => cites.extend(ids_in(result)),
        _ => {}
    }
    cites.retain(|cited| !ids.contains(cited));
    Node { key, ids, cites, events, root: false }
}

fn short_hash(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))[..16].to_string()
}

/// Node ids named in `text`: result refs (`w1.3`), digests (`w1.d2`), and
/// board intents and facts (`intent-0007`, `fact-0012`), as whole words.
fn ids_in(text: &str) -> Vec<String> {
    let mut ids = Vec::new();
    for word in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '-')) {
        let word = word.trim_end_matches(['.', '-']);
        if is_node_id(word) && !ids.iter().any(|id| id == word) {
            ids.push(word.to_string());
        }
    }
    ids
}

fn is_node_id(word: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if let Some(rest) = word.strip_prefix('w') {
        if let Some((slot, n)) = rest.split_once('.') {
            return digits(slot) && (digits(n) || n.strip_prefix('d').is_some_and(digits));
        }
        return false;
    }
    ["intent-", "fact-"].iter().any(|prefix| word.strip_prefix(prefix).is_some_and(digits))
}

// ------------------------------------------------------------------ fidelity

/// The digest with every value it names that `source` doesn't hold verbatim
/// replaced by `[dropped]`, and the values dropped.
pub(crate) fn verified(text: &str, source: &str) -> (String, Vec<String>) {
    let mut out = String::with_capacity(text.len());
    let mut dropped = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let space = rest.find(|c: char| !c.is_whitespace()).unwrap_or(rest.len());
        out.push_str(&rest[..space]);
        rest = &rest[space..];
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let token = &rest[..end];
        rest = &rest[end..];
        let core_start = token.len() - token.trim_start_matches(LEADING).len();
        let core_end = token.trim_end_matches(TRAILING).len().max(core_start);
        let core = &token[core_start..core_end];
        if is_value(core) && !holds_value(source, core) {
            out.push_str(&token[..core_start]);
            out.push_str(DROPPED);
            out.push_str(&token[core_end..]);
            if !dropped.iter().any(|d| d == core) {
                dropped.push(core.to_string());
            }
        } else {
            out.push_str(token);
        }
    }
    (out, dropped)
}

/// Whether `source` holds `value` as a value of its own, not as part of a
/// longer one: `w1.1` isn't in "w1.12", `4` isn't in "42" or "4.5", and `1.2`
/// isn't in "1.2.3". A value can still be a component of a larger one where a
/// separator divides them (`8080` in "localhost:8080", `/var/log` in
/// "/var/log/syslog"), since restating that is copying, not inventing.
fn holds_value(source: &str, value: &str) -> bool {
    let numeric = |c: Option<char>| c.is_some_and(|c| c.is_ascii_digit());
    source.match_indices(value).any(|(at, _)| {
        let before = source[..at].chars().rev();
        let after = source[at + value.len()..].chars();
        let (b1, b2) = { let mut b = before; (b.next(), b.next()) };
        let (a1, a2) = { let mut a = after; (a.next(), a.next()) };
        // Directly continued by a word character: part of a longer value.
        let joined = b1.is_some_and(char::is_alphanumeric) || a1.is_some_and(char::is_alphanumeric);
        // A number continued by `.digits` on either side: `1.2` in `1.2.3`.
        let more_number = (b1 == Some('.') && numeric(b2) && numeric(value.chars().next()))
            || (a1 == Some('.') && numeric(a2) && numeric(value.chars().last()));
        !joined && !more_number
    })
}

const LEADING: &[char] = &['"', '\'', '`', '(', '[', '{', '<', '*', '~', '≈'];
const TRAILING: &[char] = &['"', '\'', '`', ')', ']', '}', '>', ',', ';', ':', '!', '?', '.', '*'];

/// Whether a word is a value: an identifier, number, path, hash, address.
fn is_value(word: &str) -> bool {
    if word.is_empty() || !word.chars().any(|c| c.is_alphanumeric()) {
        return false;
    }
    if ["e.g", "i.e", "etc", "vs"].contains(&word.to_ascii_lowercase().as_str()) {
        return false;
    }
    if word.chars().any(|c| c.is_ascii_digit()) || word.contains(['/', '\\', '_', '@', '=']) {
        return true;
    }
    // `nginx.conf`, `host:port`: a dot or colon between word characters.
    let chars: Vec<char> = word.chars().collect();
    chars.windows(3).any(|w| (w[1] == '.' || w[1] == ':') && w[0].is_alphanumeric() && w[2].is_alphanumeric())
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionCall, ToolCall};

    fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall { id: id.into(), kind: "function".into(), function: FunctionCall { name: name.into(), arguments: arguments.to_string() } }
    }

    /// A model turn: a note (may be empty) and calls with their results.
    fn turn(note: &str, calls: &[(&str, &str, Value, &str)]) -> Vec<Message> {
        let tool_calls: Vec<ToolCall> = calls.iter().map(|(id, name, args, _)| call(id, name, args.clone())).collect();
        let mut messages = vec![Message::assistant_with_tools(note.into(), tool_calls)];
        messages.extend(calls.iter().map(|(id, _, _, result)| Message::tool(id.to_string(), result.to_string())));
        messages
    }

    fn brief() -> Vec<Message> {
        vec![Message::system("You are a worker.".into()), Message::user(json!({"task": "Find why the flush fails."}).to_string())]
    }

    fn refs(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(id, reference)| (id.to_string(), reference.to_string())).collect()
    }

    const JOURNAL_RESULT: &str = "4.1G\t/var/log/journal\nsha256 3f9a2c1e8b7d\n[evidence_ref: w1.1]";
    const STRACE_RESULT: &str = "fsync(3) = 0 (1729 calls in 2.04s)\n[evidence_ref: w1.2]";

    /// Intent 1 explored and committed; intent 2 claimed and in progress.
    fn board_history() -> (Vec<Message>, HashMap<String, String>) {
        let mut messages = brief();
        messages.extend(turn("Reading the board first.", &[("c1", "board_read", json!({"name": "flush"}), r#"{"ok":true,"intents":[]}"#)]));
        messages.extend(turn("", &[("c2", "board_claim", json!({"name": "flush", "intent_id": "intent-0001"}),
            r#"{"ok":true,"intent":{"id":"intent-0001","rationale":"journal growth"}}"#)]));
        messages.extend(turn("Checking the journal size.", &[("c3", "vm_exec", json!({"command": "du -sh /var/log/journal"}), JOURNAL_RESULT)]));
        messages.extend(turn("", &[("c4", "board_fact", json!({"name": "flush", "intent_id": "intent-0001",
            "statement": "The journal holds 4.1G.", "polarity": "supports", "basis": "observed", "evidence_refs": ["w1.1"]}),
            r#"{"ok":true,"fact":{"id":"fact-0001"}}"#)]));
        messages.extend(turn("", &[("c5", "board_claim", json!({"name": "flush", "intent_id": "intent-0002"}),
            r#"{"ok":true,"intent":{"id":"intent-0002","rationale":"fsync under load"}}"#)]));
        messages.extend(turn("Now the fsync path.", &[("c6", "vm_exec", json!({"command": "strace -c -e fsync ./flush"}), STRACE_RESULT)]));
        (messages, refs(&[("c3", "w1.1"), ("c6", "w1.2")]))
    }

    fn text(messages: &[Message]) -> String {
        messages
            .iter()
            .map(|m| {
                let calls: Vec<String> = m.tool_calls.iter().flatten().map(|c| format!("{} {}", c.function.name, c.function.arguments)).collect();
                format!("{}\n{}", m.content, calls.join("\n"))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn digests(messages: &[Message]) -> Vec<&str> {
        messages.iter().filter(|m| m.role == "user" && m.content.starts_with(DIGEST_MARKER)).map(|m| m.content.as_str()).collect()
    }

    /// Every tool result follows the assistant turn that called it, and every call has its result.
    fn assert_valid(messages: &[Message]) {
        let mut open: Vec<String> = Vec::new();
        for message in messages {
            match message.role.as_str() {
                "tool" => {
                    let id = message.tool_call_id.clone().unwrap();
                    assert!(open.contains(&id), "result {id} without its call: {messages:#?}");
                    open.retain(|o| *o != id);
                }
                role => {
                    assert!(open.is_empty(), "calls {open:?} without results before a {role} message");
                    open = message.tool_calls.iter().flatten().map(|c| c.id.clone()).collect();
                }
            }
        }
        assert!(open.is_empty(), "calls {open:?} without results");
    }

    /// A digest model that quotes some values right and invents one.
    fn honest_but_for<'a>(invented: &'static str, calls: &'a std::cell::Cell<usize>) -> impl FnMut(&str) -> Result<String> + 'a {
        move |rendered: &str| {
            calls.set(calls.get() + 1);
            assert!(rendered.starts_with("[HARNESS — FOLD DIGEST"), "{rendered}");
            Ok(format!(
                "Intent intent-0001 established that /var/log/journal holds 4.1G (w1.1, sha256 3f9a2c1e8b7d), \
recorded as fact-0001. The checksum {invented} was also seen."
            ))
        }
    }

    #[test]
    fn fidelity_every_value_in_the_resumed_context_matches_an_original() {
        let (mut messages, refs) = board_history();
        let original = messages.clone();
        let calls = std::cell::Cell::new(0);
        let mut folder = Folder::with_max_calls(1, 8);
        let report = folder.compact(&mut messages, &refs, &mut honest_but_for("deadbeef99", &calls)).unwrap();
        assert_eq!((calls.get(), report.model_calls, report.cold_chains), (1, 1, 1));
        assert_valid(&messages);
        // The cold region (the board read, intent 1 and its fact) is one digest;
        // the claim in progress and its work stay verbatim.
        assert_eq!(digests(&messages).len(), 1);
        assert_eq!(messages.len(), 2 + 1 + 4, "brief, digest, two hot turns");
        for (kept, at) in messages[3..].iter().zip(original.len() - 4..) {
            assert_eq!(serde_json::to_string(kept).unwrap(), serde_json::to_string(&original[at]).unwrap(), "hot node not verbatim");
        }
        // Positive control: the known values are in the resumed context, byte for byte.
        let resumed = text(&messages);
        for value in ["/var/log/journal", "4.1G", "3f9a2c1e8b7d", "w1.1", "fact-0001", "intent-0001",
            "strace -c -e fsync ./flush", "1729", "2.04s", "w1.2", "intent-0002"] {
            assert!(resumed.contains(value), "{value} missing from the resumed context:\n{resumed}");
        }
        // The invented value is gone, and every value a digest quotes is in an original.
        assert!(!resumed.contains("deadbeef99"));
        assert_eq!(report.folded[0].dropped, ["deadbeef99"]);
        let originals = text(&original);
        for digest in digests(&messages) {
            let body = digest.split_once('\n').unwrap().1;
            let body = body.rsplit_once("\n(This stretch").unwrap().0;
            for word in body.split_whitespace() {
                let core = word.trim_start_matches(LEADING).trim_end_matches(TRAILING);
                if is_value(core) {
                    assert!(originals.contains(core), "{core} in a digest but in no original");
                }
            }
        }
    }

    #[test]
    fn revival_a_new_intent_citing_a_folded_result_brings_the_original_back_without_running_anything() {
        let (mut messages, refs) = board_history();
        let journal_turn: Vec<String> = messages[6..8].iter().map(|m| serde_json::to_string(m).unwrap()).collect();
        let calls = std::cell::Cell::new(0);
        let mut folder = Folder::with_max_calls(1, 8);
        folder.compact(&mut messages, &refs, &mut honest_but_for("", &calls)).unwrap();
        assert!(!text(&messages).contains("sha256 3f9a2c1e8b7d\n"), "folded");
        // Intent 2 committed (nothing to revive), then a new intent citing w1.1, claimed.
        messages.extend(turn("", &[("c7", "board_fact", json!({"name": "flush", "intent_id": "intent-0002",
            "statement": "fsync is fine.", "polarity": "rules_out", "basis": "observed", "evidence_refs": ["w1.2"]}),
            r#"{"ok":true,"fact":{"id":"fact-0002"}}"#)]));
        assert!(folder.revive(&mut messages, &refs).is_empty(), "a commit citing nothing folded revives nothing");
        messages.extend(turn("", &[("c8", "board_intent", json!({"name": "flush", "intents": [{"direction": "Vacuum the journal",
            "rationale": "w1.1 showed 4.1G in /var/log/journal"}]}), r#"{"ok":true,"declared":[{"id":"intent-0003"}]}"#)]));
        messages.extend(turn("", &[("c9", "board_claim", json!({"name": "flush", "intent_id": "intent-0003"}),
            r#"{"ok":true,"intent":{"id":"intent-0003","rationale":"w1.1 showed 4.1G in /var/log/journal"}}"#)]));
        let revived = folder.revive(&mut messages, &refs);
        assert_eq!(revived, ["w1.d1"]);
        assert!(digests(&messages).is_empty());
        assert_valid(&messages);
        // The original turn is back, byte for byte, once: re-inflated, not re-run.
        let now: Vec<String> = messages.iter().map(|m| serde_json::to_string(m).unwrap()).collect();
        let at = now.iter().position(|m| *m == journal_turn[0]).expect("the original call is back");
        assert_eq!(now[at + 1], journal_turn[1], "its result is back as it was");
        assert_eq!(messages.iter().filter(|m| m.tool_call_id.as_deref() == Some("c3")).count(), 1);
        assert_eq!(calls.get(), 1, "revival makes no model call");
        assert!(!folder.active());
    }

    #[test]
    fn idempotence_compacting_twice_makes_no_call_and_the_same_digests() {
        let (mut messages, refs) = board_history();
        let calls = std::cell::Cell::new(0);
        let mut folder = Folder::with_max_calls(1, 8);
        folder.compact(&mut messages, &refs, &mut honest_but_for("", &calls)).unwrap();
        let once = serde_json::to_string(&messages).unwrap();
        let report = folder.compact(&mut messages, &refs, &mut honest_but_for("", &calls)).unwrap();
        assert_eq!((report.model_calls, calls.get()), (0, 1));
        assert_eq!(serde_json::to_string(&messages).unwrap(), once);
        // Expanded and folded again unchanged: the memo answers, under the same id.
        assert!(folder.expand(&mut messages, "w1.d1").is_some());
        assert!(digests(&messages).is_empty());
        let report = folder.compact(&mut messages, &refs, &mut honest_but_for("", &calls)).unwrap();
        assert_eq!((report.model_calls, calls.get()), (0, 1));
        assert!(report.folded[0].memo_hit);
        assert_eq!(serde_json::to_string(&messages).unwrap(), once);
    }

    /// Three cold stretches kept apart by directions from s0ma.
    fn three_stretches() -> (Vec<Message>, HashMap<String, String>) {
        let mut messages = brief();
        let steer = |text: &str| Message::user(format!("{STEER_MARKER}, NOT A NEW TASK AND NOT OPERATOR INPUT]\n{text}"));
        for (n, stretch) in ["one", "two", "three"].iter().enumerate() {
            let id = format!("s{n}");
            messages.extend(turn(&format!("Stretch {stretch}."), &[(&id, "vm_exec", json!({"command": format!("ls /tmp/{stretch}")}), "ok")]));
            messages.push(steer(&format!("after stretch {stretch}")));
        }
        (messages, HashMap::new())
    }

    #[test]
    fn cost_one_call_per_cold_chain_and_the_cap_keeps_the_oldest_verbatim() {
        let (mut messages, refs) = three_stretches();
        let calls = std::cell::Cell::new(0);
        let mut count = |_: &str| -> Result<String> {
            calls.set(calls.get() + 1);
            Ok("A stretch.".into())
        };
        let report = Folder::with_max_calls(1, 8).compact(&mut messages, &refs, &mut count).unwrap();
        assert_eq!((report.cold_chains, report.model_calls, calls.get()), (3, 3, 3));
        assert!(report.model_calls <= report.cold_chains);
        assert_eq!(digests(&messages).len(), 3);
        assert_eq!(messages.iter().filter(|m| m.content.starts_with(STEER_MARKER)).count(), 3, "directions are the goal: kept");

        // Capped at one call: the newest stretch is folded, the two oldest stay verbatim.
        let (mut messages, refs) = three_stretches();
        calls.set(0);
        let report = Folder::with_max_calls(1, 1).compact(&mut messages, &refs, &mut count).unwrap();
        assert_eq!((report.cold_chains, report.model_calls, report.kept_verbatim.len()), (3, 1, 2));
        let resumed = text(&messages);
        assert!(resumed.contains("ls /tmp/one") && resumed.contains("ls /tmp/two") && !resumed.contains("ls /tmp/three"), "{resumed}");
        assert_valid(&messages);
        // The next compaction gets to the rest, one at a time.
        let mut folder = Folder::with_max_calls(1, 1);
        let (mut messages, refs) = three_stretches();
        folder.compact(&mut messages, &refs, &mut count).unwrap();
        let report = folder.compact(&mut messages, &refs, &mut count).unwrap();
        assert_eq!((report.model_calls, report.kept_verbatim.len()), (1, 1));
    }

    #[test]
    fn a_worker_without_a_board_keeps_its_goal_and_folds_the_rest() {
        let mut messages = brief();
        messages.extend(turn("Looking around.", &[("a", "vm_exec", json!({"command": "uname -a"}), "Linux build 6.8.0")]));
        messages.extend(turn("Disk next.", &[("b", "vm_exec", json!({"command": "df -h /"}), "/dev/sda1 50G 41G 82%")]));
        let mut folder = Folder::with_max_calls(3, 8);
        let mut once = |_: &str| -> Result<String> { Ok("The root filesystem /dev/sda1 is 82% full (41G of 50G).".into()) };
        let report = folder.compact(&mut messages, &HashMap::new(), &mut once).unwrap();
        assert_eq!((report.cold_chains, report.model_calls), (1, 1));
        assert_eq!(messages.len(), 3);
        assert!(messages[2].content.starts_with("[HARNESS — FOLDED CONTEXT w3.d1, NOT A NEW TASK]"), "{}", messages[2].content);
        assert!(messages[2].content.contains("82% full (41G of 50G)") && report.folded[0].dropped.is_empty());
    }

    #[test]
    fn a_turn_with_anything_hot_in_it_is_kept_whole() {
        let mut messages = brief();
        // One turn: a stray look around beside the claim that is still in progress.
        messages.extend(turn("", &[
            ("x", "vm_exec", json!({"command": "uptime"}), "up 3 days"),
            ("y", "board_claim", json!({"name": "flush", "intent_id": "intent-0004"}), r#"{"ok":true,"intent":{"id":"intent-0004"}}"#),
        ]));
        let before = messages.clone();
        let mut never = |_: &str| -> Result<String> { panic!("nothing is cold") };
        let report = Folder::with_max_calls(1, 8).compact(&mut messages, &HashMap::new(), &mut never).unwrap();
        assert_eq!(report.cold_chains, 0);
        assert_eq!(serde_json::to_string(&messages).unwrap(), serde_json::to_string(&before).unwrap());
    }

    #[test]
    fn expanding_by_id_restores_the_originals_in_place() {
        let (mut messages, refs) = board_history();
        let original = serde_json::to_string(&messages).unwrap();
        let mut folder = Folder::with_max_calls(1, 8);
        let calls = std::cell::Cell::new(0);
        folder.compact(&mut messages, &refs, &mut honest_but_for("", &calls)).unwrap();
        assert!(folder.is_digest("w1.d1") && !folder.is_digest("w1.1"));
        let expanded = folder.expand(&mut messages, "w1.d1").unwrap();
        assert_eq!(expanded["restored_messages"], 8, "four turns: the board read, the claim, the journal check, the fact");
        assert_eq!(serde_json::to_string(&messages).unwrap(), original);
        assert!(folder.expand(&mut messages, "w1.d1").is_none(), "once");
    }

    #[test]
    fn values_and_ids_are_told_apart_from_words() {
        for value in ["w1.3", "/var/log", "4.1G", "3f9a2c1e8b7d", "nginx.conf", "host:8080", "a_b", "x=1", "10.0.0.5", "v2"] {
            assert!(is_value(value), "{value}");
        }
        for word in ["journal", "well-known", "e.g", "dead-end", "", "—"] {
            assert!(!is_value(word), "{word}");
        }
        let (text, dropped) = verified("Saw /etc/hosts and \"/etc/shadow\", w1.4 (sha 99aa), e.g. twice.", "/etc/hosts w1.4");
        assert_eq!(text, "Saw /etc/hosts and \"[dropped]\", w1.4 (sha [dropped]), e.g. twice.");
        assert_eq!(dropped, ["/etc/shadow", "99aa"]);
        // A value has to be in the stretch as itself, not inside a longer one.
        let (text, dropped) = verified(
            "Found in w1.1 at line 4, version 1.2, port 8080, under /var/log, ref w1.12 (line 42).",
            "w1.12 said: line 42 of /var/log/syslog, v1.2.3 on localhost:8080",
        );
        assert_eq!(text, "Found in [dropped] at line [dropped], version [dropped], port 8080, under /var/log, ref w1.12 (line 42).");
        assert_eq!(dropped, ["w1.1", "4", "1.2"]);
        assert!(holds_value("ran w1.1.", "w1.1"), "punctuation after a value ends it");
        assert!(!holds_value("got 4.5 GB", "4"));
        assert!(!holds_value("from 10.0.0.12", "0.0.12") && !holds_value("from 10.0.0.12", "10.0.0.1"));
        assert_eq!(ids_in("see w1.3, w1.d2 and intent-0007; not w1.30x or fact-12a or w.3"), ["w1.3", "w1.d2", "intent-0007"]);
        assert_eq!(Strategy::parse(Some("notes")), Strategy::Notes);
        assert_eq!(Strategy::parse(None), Strategy::Liveness);
        assert_eq!(Strategy::parse(Some("liveness")), Strategy::Liveness);
    }
}
