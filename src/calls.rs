//! The call store: every model call the main agent makes, saved at the source
//! (`claude/mistake-capture-spec.md` §1).
//!
//! The agent's context folds continuously and is not deterministic, so nothing can
//! rebuild what it saw at a past moment. `/mistake` reaches back into this
//! store instead: the exact bytes POSTed and the response as it was assembled
//! from the stream.
//!
//! Layout under the data folder:
//!
//! ```text
//! data/calls/index.jsonl            one line per call, appended (fsync per line)
//! data/calls/objects/ab/cdef….zst   content-addressed blobs: sha256 of the
//!                                   uncompressed bytes, zstd-compressed
//! ```
//!
//! **Exactness.** The request body is serialised once, here, and the same bytes
//! are both POSTed and stored. Each message is stored as the exact byte slice it
//! occupies in the body, the tools array likewise, and the rest of the body as
//! a *skeleton*: the body with the `messages` and `tools` values cut out, and
//! the byte offsets (`splice`) where they go back in. Rebuilding is pure
//! substitution: `messages` = `[` + the message blobs joined by `,` + `]`,
//! inserted at `splice.messages`, and the tools blob at `splice.tools`, the
//! larger offset first. The rebuild is checked against `body_sha256` before the
//! line is written; if it ever disagrees, the body itself is stored too
//! (`request.body`).
//!
//! **The response** is what the stream assembled into (`llm::parse_stream`):
//! the concatenated reasoning deltas, the concatenated content deltas, and the
//! tool calls with each one's id, type, name and argument fragments
//! concatenated in index order. A call the operator interrupted is saved too,
//! with what had streamed so far and `finish: "interrupted"`: the moment just
//! after stopping the agent is the most natural moment to flag it.
//!
//! **Crash safety.** A blob goes to a dot-prefixed temp file, is fsynced,
//! renamed into place, and the directory fsynced, all before the index line
//! that names it is appended (one `write`, then `fsync`). A crash can leave an
//! orphan blob, never a line pointing at nothing. A torn last line is repaired
//! on open.
//!
//! Writes happen on one background thread, so saving never delays the agent; the
//! only thing on the agent's own path is reading what the engine serves, a
//! local request with a short timeout.

use crate::llm::S0maResponse;
use crate::types::ToolCall;
use serde::de::{Deserializer, MapAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// How long the agent waits for the engine to say what it serves before the
/// request. The engine is local; this only bounds a broken one.
const SERVED_TIMEOUT_BEFORE: Duration = Duration::from_millis(250);
/// The same read on the writer thread, after the response.
const SERVED_TIMEOUT_AFTER: Duration = Duration::from_secs(2);
/// A read after the response taken later than this (the writer was behind)
/// no longer says which engine answered.
const SERVED_LATE: Duration = Duration::from_secs(2);
/// Calls whose save outcome is remembered for `/mistake` (succeeded, or why not).
const REMEMBERED_OUTCOMES: usize = 256;
/// System prompts whose spans are remembered. The system message changes only
/// when the runtime context is rebuilt (startup, sleep, compaction).
const REMEMBERED_SPANS: usize = 16;

// ------------------------------------------------------------------ ids and time

fn utc_stamp() -> String {
    chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ").to_string()
}

fn local_now() -> String {
    chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, false)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn blob_ref(hex: &str) -> String {
    format!("sha256:{hex}")
}

fn blob_hex(reference: &str) -> Option<&str> {
    reference.strip_prefix("sha256:").filter(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// The sequence number at the end of a call id: `c-…-000183` → 183.
fn call_seq(call_id: &str) -> Option<u64> {
    call_id.strip_prefix("c-")?.rsplit('-').next()?.parse().ok()
}

// ------------------------------------------------------------------ system spans

/// A labelled byte range (UTF-8) of the system message's content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub label: String,
    pub start: usize,
    pub end: usize,
}

/// Records spans as a system prompt is assembled: call `mark` after appending
/// each labelled piece.
#[derive(Debug, Default)]
pub struct SpanMarks {
    spans: Vec<Span>,
}

impl SpanMarks {
    pub fn mark(&mut self, label: &str, content: &str) {
        let start = self.spans.last().map_or(0, |span| span.end);
        self.spans.push(Span { label: label.to_owned(), start, end: content.len() });
    }

    pub fn into_spans(self) -> Vec<Span> {
        self.spans
    }
}

/// System prompts' spans, keyed by the sha256 of the prompt.
type SpanRegistry = Mutex<VecDeque<(String, Vec<Span>)>>;

fn span_registry() -> &'static SpanRegistry {
    static SPANS: OnceLock<SpanRegistry> = OnceLock::new();
    SPANS.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Remember the spans of a system prompt just assembled, keyed by its content,
/// so a call that carries it can be saved with them.
pub fn register_system_spans(content: &str, spans: Vec<Span>) {
    let key = sha256_hex(content.as_bytes());
    let mut registry = span_registry().lock().unwrap_or_else(|e| e.into_inner());
    registry.retain(|(seen, _)| seen != &key);
    if registry.len() == REMEMBERED_SPANS {
        registry.pop_front();
    }
    registry.push_back((key, spans));
}

pub fn spans_for(content: &str) -> Option<Vec<Span>> {
    let key = sha256_hex(content.as_bytes());
    let registry = span_registry().lock().unwrap_or_else(|e| e.into_inner());
    registry.iter().rev().find(|(seen, _)| seen == &key).map(|(_, spans)| spans.clone())
}

// ------------------------------------------------------------------ the blob store

/// The store's files, readable from any process (the review window opens it
/// standalone). Only the writer thread of the running agent appends.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

impl Store {
    pub fn at(data_dir: &Path) -> Self {
        Self { root: data_dir.join("calls") }
    }

    pub fn index_path(&self) -> PathBuf {
        self.root.join("index.jsonl")
    }

    fn objects(&self) -> PathBuf {
        self.root.join("objects")
    }

    fn object_path(&self, hex: &str) -> PathBuf {
        self.objects().join(&hex[..2]).join(format!("{}.zst", &hex[2..]))
    }

    /// Store `bytes` once, by the sha256 of the uncompressed bytes. Durable
    /// before it returns: temp file, fsync, rename, fsync of the directory.
    pub fn put(&self, bytes: &[u8]) -> io::Result<String> {
        let hex = sha256_hex(bytes);
        let path = self.object_path(&hex);
        if path.exists() {
            return Ok(hex);
        }
        let dir = path.parent().expect("object paths have a parent");
        fs::create_dir_all(dir)?;
        let temp = dir.join(format!(
            ".{}.tmp-{}-{}",
            &hex[2..12],
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let compressed = zstd::encode_all(bytes, 3)?;
        {
            let mut file = File::create(&temp)?;
            file.write_all(&compressed)?;
            file.sync_all()?;
        }
        fs::rename(&temp, &path)?;
        if let Ok(dir) = File::open(dir) {
            dir.sync_all()?;
        }
        Ok(hex)
    }

    /// The bytes of a blob, checked against its name.
    pub fn get(&self, reference: &str) -> io::Result<Vec<u8>> {
        let hex = blob_hex(reference)
            .or_else(|| (reference.len() == 64).then_some(reference))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("not a blob reference: {reference}")))?;
        let compressed = fs::read(self.object_path(hex))?;
        let bytes = zstd::decode_all(compressed.as_slice())?;
        if sha256_hex(&bytes) != hex {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("blob {hex} does not match its hash")));
        }
        Ok(bytes)
    }

    /// Append one line: one `write` of the line and its newline, then fsync.
    fn append_line(&self, line: &[u8]) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        let mut file = OpenOptions::new().create(true).append(true).open(self.index_path())?;
        let mut bytes = Vec::with_capacity(line.len() + 1);
        bytes.extend_from_slice(line);
        bytes.push(b'\n');
        let before = file.metadata()?.len();
        if let Err(error) = file.write_all(&bytes) {
            let _ = file.set_len(before);
            return Err(error);
        }
        file.sync_data()
    }

    /// A process killed mid-append leaves a partial last line. A complete one
    /// that only lacks its newline gets it; a fragment is dropped (the call it
    /// would have named was never recorded, and its blobs are orphans).
    pub fn repair_torn_tail(&self) -> io::Result<Option<usize>> {
        let path = self.index_path();
        let Ok(mut file) = OpenOptions::new().read(true).write(true).open(&path) else {
            return Ok(None);
        };
        let length = file.metadata()?.len();
        if length == 0 {
            return Ok(None);
        }
        let mut window = length.min(1 << 20);
        let tail = loop {
            file.seek(SeekFrom::Start(length - window))?;
            let mut tail = Vec::with_capacity(window as usize);
            file.read_to_end(&mut tail)?;
            if tail.last() == Some(&b'\n') {
                return Ok(None);
            }
            if tail.contains(&b'\n') || window == length {
                break tail;
            }
            window = length.min(window * 2);
        };
        let fragment_start = tail.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        let fragment = &tail[fragment_start..];
        if serde_json::from_slice::<Value>(fragment).is_ok() {
            file.seek(SeekFrom::End(0))?;
            file.write_all(b"\n")?;
            file.sync_data()?;
            return Ok(None);
        }
        file.set_len(length - fragment.len() as u64)?;
        file.sync_data()?;
        Ok(Some(fragment.len()))
    }

    /// Index lines, newest first, without reading the whole file.
    pub fn lines_newest_first(&self) -> io::Result<RevLines> {
        RevLines::open(&self.index_path())
    }

    /// The call lines of one turn, oldest first. A turn's calls are
    /// contiguous in the index (only update lines come between), so the scan
    /// stops once it is past the turn, and a line is parsed only if it names
    /// the turn.
    pub fn turn_lines(&self, turn_id: &str) -> io::Result<Vec<Value>> {
        let needle = format!("\"turn_id\":{}", serde_json::Value::String(turn_id.to_owned()));
        let mut turn = Vec::new();
        for line in self.lines_newest_first()? {
            let line = line?;
            if !contains(&line, needle.as_bytes()) {
                if !turn.is_empty() && !contains(&line, b"\"update\":true") {
                    break;
                }
                continue;
            }
            let Ok(value) = serde_json::from_slice::<Value>(&line) else { continue };
            if value.get("turn_id").and_then(Value::as_str) == Some(turn_id) && value.get("update").is_none() {
                turn.push(value);
            }
        }
        turn.reverse();
        Ok(turn)
    }

    /// The calls of one turn, oldest first, each with whether its index line
    /// records its stretch (calls saved before 2026-10-23 don't).
    pub fn turn_calls(&self, turn_id: &str) -> io::Result<Vec<(CallRef, bool)>> {
        Ok(self
            .turn_lines(turn_id)?
            .iter()
            .filter_map(|line| call_ref(line).map(|call| (call, line.get("stretch").is_some())))
            .collect())
    }

    /// The steps a mistake on `call` can correct: the calls of its turn in the
    /// same stretch (since the operator last spoke, or the turn began), up to
    /// and including it, oldest first.
    ///
    /// When `call` is the first of its stretch, the stretch before it is
    /// included too: a flag typed just after a steer ("no, the other file")
    /// lands on the first call after it, and the mistake is in the calls the
    /// steer was about. A call saved before stretches were recorded has no
    /// `stretch` to go by, so it is its only step.
    pub fn stretch_of(&self, call: &CallRef) -> io::Result<Vec<CallRef>> {
        let turn: Vec<(CallRef, bool)> =
            self.turn_calls(&call.turn_id)?.into_iter().filter(|(step, _)| step.call_in_turn <= call.call_in_turn).collect();
        if turn.iter().any(|(step, recorded)| step.call_id == call.call_id && !recorded) {
            return Ok(turn.into_iter().map(|(step, _)| step).filter(|step| step.call_id == call.call_id).collect());
        }
        let first_of_stretch = !turn.iter().any(|(step, _)| step.stretch == call.stretch && step.call_in_turn < call.call_in_turn);
        let from = if first_of_stretch && call.stretch > 0 { call.stretch - 1 } else { call.stretch };
        Ok(turn.into_iter().map(|(step, _)| step).filter(|step| (from..=call.stretch).contains(&step.stretch)).collect())
    }

    /// Up to `limit` call lines saved before `call_id`, newest first.
    pub fn calls_before(&self, call_id: &str, limit: usize) -> io::Result<Vec<Value>> {
        let mut seen = false;
        let mut found = Vec::new();
        for line in self.lines_newest_first()? {
            let line = line?;
            if contains(&line, b"\"update\":true") {
                continue;
            }
            if !seen {
                seen = line_call_id(&line) == Some(call_id.as_bytes());
                continue;
            }
            if let Ok(value) = serde_json::from_slice::<Value>(&line) {
                found.push(value);
                if found.len() == limit {
                    break;
                }
            }
        }
        Ok(found)
    }

    /// The call an archive event belongs to: one of the agent's own events (its
    /// reasoning, a tool result, its reply) recorded after that call's
    /// response, among the newest `within` calls. None for an event no saved
    /// call owns there: the operator's, a harness note, one older than that,
    /// or one whose call is still open (its events reach the index when the
    /// next call begins). Archive ids rise with time, so the scan stops once
    /// it is older than the event.
    pub fn call_with_event(&self, event_id: u64, within: usize) -> io::Result<Option<CallRef>> {
        let mut owner: Option<String> = None;
        let mut calls = 0;
        for line in self.lines_newest_first()? {
            let line = line?;
            let is_update = contains(&line, b"\"update\":true");
            if let Some(owner) = &owner {
                if !is_update
                    && line_call_id(&line) == Some(owner.as_bytes())
                    && let Ok(value) = serde_json::from_slice::<Value>(&line)
                {
                    return Ok(call_ref(&value));
                }
                continue;
            }
            if !is_update {
                calls += 1;
                if calls > within {
                    return Ok(None);
                }
                if !contains(&line, event_id.to_string().as_bytes()) {
                    continue;
                }
            }
            let Ok(value) = serde_json::from_slice::<Value>(&line) else { continue };
            let events: Vec<u64> = value
                .get("archive_events")
                .and_then(Value::as_array)
                .map(|events| events.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            if events.contains(&event_id) {
                if !is_update {
                    return Ok(call_ref(&value));
                }
                owner = value.get("call_id").and_then(Value::as_str).map(str::to_owned);
            } else if is_update && events.iter().max().is_some_and(|newest| *newest < event_id) {
                // Every call from here back is older than the event.
                return Ok(None);
            }
        }
        Ok(None)
    }

    /// The newest `limit` calls (update lines folded in), newest first.
    pub fn recent(&self, limit: usize) -> io::Result<Vec<Value>> {
        let mut updates: HashMap<String, Value> = HashMap::new();
        let mut calls = Vec::new();
        for line in self.lines_newest_first()? {
            let Ok(value) = serde_json::from_slice::<Value>(&line?) else { continue };
            let Some(id) = value.get("call_id").and_then(Value::as_str).map(str::to_owned) else { continue };
            if value.get("update").and_then(Value::as_bool) == Some(true) {
                // Newest first, so the first update seen for a call is its last.
                updates.entry(id).or_insert(value);
                continue;
            }
            calls.push(fold_update(value, updates.remove(&id)));
            if calls.len() == limit {
                break;
            }
        }
        Ok(calls)
    }

    /// One call's line, with its latest update folded in.
    pub fn find(&self, call_id: &str) -> io::Result<Option<Value>> {
        let mut update = None;
        for line in self.lines_newest_first()? {
            let line = line?;
            // Every line this store writes starts with its call id, so a line
            // of tens of KB is skipped on its first few bytes.
            match line_call_id(&line) {
                Some(id) if id != call_id.as_bytes() => continue,
                Some(_) => {}
                None if !contains(&line, call_id.as_bytes()) => continue,
                None => {}
            }
            let Ok(value) = serde_json::from_slice::<Value>(&line) else { continue };
            if value.get("call_id").and_then(Value::as_str) != Some(call_id) {
                continue;
            }
            if value.get("update").and_then(Value::as_bool) == Some(true) {
                update.get_or_insert(value);
                continue;
            }
            return Ok(Some(fold_update(value, update)));
        }
        Ok(None)
    }

    /// The response object of a call, as stored.
    pub fn response(&self, line: &Value) -> Result<Value> {
        let reference = line.get("response").and_then(Value::as_str).ok_or("index line has no response")?;
        Ok(serde_json::from_slice(&self.get(reference)?)?)
    }

    /// The request's messages, in order, as JSON.
    pub fn messages(&self, line: &Value) -> Result<Vec<Value>> {
        let references = line
            .pointer("/request/messages")
            .and_then(Value::as_array)
            .ok_or("index line has no messages")?;
        references
            .iter()
            .map(|reference| {
                let reference = reference.as_str().ok_or("message reference is not a string")?;
                Ok(serde_json::from_slice(&self.get(reference)?)?)
            })
            .collect()
    }

    /// The exact bytes that were POSTed, rebuilt from the parts (the same
    /// substitution qref performs to check `body_sha256`).
    #[cfg(test)]
    pub fn body(&self, line: &Value) -> Result<Vec<u8>> {
        let request = line.get("request").ok_or("index line has no request")?;
        // Stored whole when the rebuild disagreed with it.
        if let Some(reference) = request.get("body").and_then(Value::as_str) {
            return Ok(self.get(reference)?);
        }
        let skeleton = request.get("skeleton").and_then(Value::as_str).ok_or("request has no skeleton")?;
        let mut messages = Vec::new();
        for reference in request.get("messages").and_then(Value::as_array).ok_or("request has no messages")? {
            messages.push(self.get(reference.as_str().ok_or("message reference is not a string")?)?);
        }
        let tools = match request.get("tools").and_then(Value::as_str) {
            Some(reference) => Some(self.get(reference)?),
            None => None,
        };
        let splice = request.get("splice").ok_or("request has no splice")?;
        let at = |key: &str| splice.get(key).and_then(Value::as_u64).map(|n| n as usize);
        rebuild(skeleton, &messages, at("messages"), tools.as_deref(), at("tools"))
    }
}

/// The call id a line opens with (`{"call_id":"…"`), if it opens that way.
fn line_call_id(line: &[u8]) -> Option<&[u8]> {
    let rest = line.strip_prefix(b"{\"call_id\":\"")?;
    let end = rest.iter().take(64).position(|b| *b == b'"')?;
    Some(&rest[..end])
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|window| window == needle)
}

fn fold_update(mut line: Value, update: Option<Value>) -> Value {
    if let (Some(update), Some(object)) = (update, line.as_object_mut())
        && let Some(events) = update.get("archive_events") {
            object.insert("archive_events".into(), events.clone());
        }
    line
}

/// Lines of a file from the last to the first, read backwards in chunks. An
/// unterminated last line (an append in progress, or debris) is left out.
pub struct RevLines {
    file: Option<File>,
    pos: u64,
    buf: Vec<u8>,
}

impl RevLines {
    fn open(path: &Path) -> io::Result<Self> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(Self { file: None, pos: 0, buf: Vec::new() })
            }
            Err(error) => return Err(error),
        };
        let pos = file.metadata()?.len();
        let mut lines = Self { file: Some(file), pos, buf: Vec::new() };
        // Drop an unterminated tail.
        loop {
            if let Some(newline) = lines.buf.iter().rposition(|b| *b == b'\n') {
                lines.buf.truncate(newline + 1);
                break;
            }
            if !lines.read_more()? {
                lines.buf.clear();
                break;
            }
        }
        Ok(lines)
    }

    fn read_more(&mut self) -> io::Result<bool> {
        let Some(file) = self.file.as_mut() else { return Ok(false) };
        if self.pos == 0 {
            return Ok(false);
        }
        let chunk = self.pos.min(256 * 1024);
        self.pos -= chunk;
        file.seek(SeekFrom::Start(self.pos))?;
        let mut bytes = vec![0; chunk as usize];
        file.read_exact(&mut bytes)?;
        bytes.extend_from_slice(&self.buf);
        self.buf = bytes;
        Ok(true)
    }
}

impl Iterator for RevLines {
    type Item = io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // `buf` ends with a newline (or is empty): the line before it is next.
            if self.buf.is_empty() {
                return None;
            }
            let end = self.buf.len() - 1;
            match self.buf[..end].iter().rposition(|b| *b == b'\n') {
                Some(start) => {
                    let line = self.buf[start + 1..end].to_vec();
                    self.buf.truncate(start + 1);
                    if line.iter().all(u8::is_ascii_whitespace) {
                        continue;
                    }
                    return Some(Ok(line));
                }
                None => match self.read_more() {
                    Ok(true) => continue,
                    Ok(false) => {
                        let line = self.buf[..end].to_vec();
                        self.buf.clear();
                        if line.iter().all(u8::is_ascii_whitespace) {
                            return None;
                        }
                        return Some(Ok(line));
                    }
                    Err(error) => return Some(Err(error)),
                },
            }
        }
    }
}

// ------------------------------------------------------------------ the body, split

/// A JSON object's members in their order, each value as the exact text it had.
struct Members(Vec<(String, Box<RawValue>)>);

impl<'de> Deserialize<'de> for Members {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct MembersVisitor;
        impl<'de> Visitor<'de> for MembersVisitor {
            type Value = Members;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Members, A::Error> {
                let mut members = Vec::new();
                while let Some(member) = map.next_entry::<String, Box<RawValue>>()? {
                    members.push(member);
                }
                Ok(Members(members))
            }
        }
        deserializer.deserialize_map(MembersVisitor)
    }
}

/// A request body taken apart for storage.
#[derive(Debug)]
pub struct SplitBody {
    /// Each message exactly as it appears in the body.
    pub messages: Vec<Vec<u8>>,
    pub tools: Option<Vec<u8>>,
    /// Every other member, in order, as a JSON object.
    pub params: String,
    pub skeleton: String,
    pub splice_messages: Option<usize>,
    pub splice_tools: Option<usize>,
}

pub fn split_body(body: &[u8]) -> Result<SplitBody> {
    let Members(members) = serde_json::from_slice(body)?;
    let mut skeleton = String::from("{");
    let mut params = String::from("{");
    let mut split = SplitBody {
        messages: Vec::new(),
        tools: None,
        params: String::new(),
        skeleton: String::new(),
        splice_messages: None,
        splice_tools: None,
    };
    for (index, (key, value)) in members.iter().enumerate() {
        if index > 0 {
            skeleton.push(',');
        }
        skeleton.push_str(&serde_json::to_string(key)?);
        skeleton.push(':');
        match key.as_str() {
            "messages" => {
                let messages: Vec<Box<RawValue>> = serde_json::from_str(value.get())?;
                split.messages = messages.iter().map(|m| m.get().as_bytes().to_vec()).collect();
                split.splice_messages = Some(skeleton.len());
            }
            "tools" => {
                split.tools = Some(value.get().as_bytes().to_vec());
                split.splice_tools = Some(skeleton.len());
            }
            _ => {
                skeleton.push_str(value.get());
                if params.len() > 1 {
                    params.push(',');
                }
                params.push_str(&serde_json::to_string(key)?);
                params.push(':');
                params.push_str(value.get());
            }
        }
    }
    skeleton.push('}');
    params.push('}');
    split.skeleton = skeleton;
    split.params = params;
    Ok(split)
}

/// The inverse of `split_body`: pure substitution, larger offset first.
pub fn rebuild(
    skeleton: &str,
    messages: &[Vec<u8>],
    splice_messages: Option<usize>,
    tools: Option<&[u8]>,
    splice_tools: Option<usize>,
) -> Result<Vec<u8>> {
    let mut joined = vec![b'['];
    for (index, message) in messages.iter().enumerate() {
        if index > 0 {
            joined.push(b',');
        }
        joined.extend_from_slice(message);
    }
    joined.push(b']');
    let mut inserts: Vec<(usize, &[u8])> = Vec::new();
    if let Some(at) = splice_messages {
        inserts.push((at, &joined));
    }
    if let (Some(at), Some(tools)) = (splice_tools, tools) {
        inserts.push((at, tools));
    }
    inserts.sort_by_key(|(at, _)| std::cmp::Reverse(*at));
    let mut body = skeleton.as_bytes().to_vec();
    for (at, bytes) in inserts {
        if at > body.len() {
            return Err(format!("splice offset {at} is past the skeleton's end").into());
        }
        body.splice(at..at, bytes.iter().copied());
    }
    Ok(body)
}

// ------------------------------------------------------------------ what the engine serves

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Served {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slice_tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_started: Option<u64>,
    /// Where `model_dir` came from when the engine did not say: `serve.pid`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// The reads before and after the call disagree: the engine changed while
    /// the call ran, so which weights answered is not known.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub uncertain: bool,
}

/// What the engine serves, from `/v1/models` (`model_dir`, `slice_tag` and
/// `created` on the model's entry, or at the top level). When the engine does
/// not report `model_dir`, the folder is read from the `--model` argument of
/// the process in the engine's pid file, as the capability eval does.
pub fn served_now(client: &reqwest::blocking::Client, base_url: &str, model: &str) -> Option<Served> {
    // A hosted API serves what it serves: there is no folder or engine to name.
    if crate::model_config::hosted() {
        return Some(Served { model: model.to_owned(), model_dir: None, slice_tag: None, engine_started: None, source: Some("hosted API".into()), uncertain: false });
    }
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let response: Value = crate::model_config::authorize(client.get(url)).send().ok()?.error_for_status().ok()?.json().ok()?;
    let entry = response
        .get("data")
        .and_then(Value::as_array)
        .and_then(|data| data.iter().find(|e| e.get("id").and_then(Value::as_str) == Some(model)).cloned());
    let field = |key: &str| {
        entry
            .as_ref()
            .and_then(|e| e.get(key))
            .or_else(|| response.get(key))
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    let mut served = Served {
        model: model.to_owned(),
        model_dir: field("model_dir"),
        slice_tag: field("slice_tag"),
        engine_started: entry.as_ref().and_then(|e| e.get("created")).and_then(Value::as_u64),
        source: None,
        uncertain: false,
    };
    if served.model_dir.is_none()
        && let Some(dir) = model_dir_from_pid_file() {
            served.model_dir = Some(dir);
            served.source = Some("serve.pid".into());
        }
    Some(served)
}

/// One client per timeout, built once: building a blocking client starts a
/// thread, which is not something to do on the agent's path per call.
fn served_client(timeout: Duration) -> Option<&'static reqwest::blocking::Client> {
    static BEFORE: OnceLock<Option<reqwest::blocking::Client>> = OnceLock::new();
    static AFTER: OnceLock<Option<reqwest::blocking::Client>> = OnceLock::new();
    let cell = if timeout == SERVED_TIMEOUT_BEFORE { &BEFORE } else { &AFTER };
    cell.get_or_init(|| {
        reqwest::blocking::Client::builder().connect_timeout(timeout).timeout(timeout).build().ok()
    })
    .as_ref()
}

fn served_with(timeout: Duration, base_url: &str, model: &str) -> Option<Served> {
    served_now(served_client(timeout)?, base_url, model)
}

fn model_dir_from_pid_file() -> Option<String> {
    let path = std::env::var("S0MA_ENGINE_PIDFILE")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var("HOME").ok().map(|home| Path::new(&home).join("s0ma-engine/.serve.pid")))?;
    let pid: u32 = fs::read_to_string(path).ok()?.trim().parse().ok()?;
    let cmdline = fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    model_arg(&cmdline)
}

fn model_arg(cmdline: &[u8]) -> Option<String> {
    let args: Vec<String> = cmdline
        .split(|b| *b == 0)
        .filter(|a| !a.is_empty())
        .map(|a| String::from_utf8_lossy(a).into_owned())
        .collect();
    args.iter().enumerate().find_map(|(i, arg)| {
        if arg == "--model" {
            args.get(i + 1).cloned()
        } else {
            arg.strip_prefix("--model=").map(str::to_owned)
        }
    })
}

/// The stamp a call is saved with, and `uncertain` when the reads before the
/// request and after the response disagree (the engine changed around it).
/// The read after the response saw the engine that answered, unless it
/// restarted since; so it is the stamp when it was taken promptly, and the
/// read before the request is when it was not (`late`: the writer was behind).
pub fn merge_served(before: Option<Served>, after: Option<Served>, model: &str, late: bool) -> Served {
    match (before, after) {
        (Some(before), Some(after)) => {
            let uncertain = before.model_dir != after.model_dir
                || before.slice_tag != after.slice_tag
                || before.engine_started != after.engine_started;
            let mut stamp = if late { before } else { after };
            stamp.uncertain = uncertain;
            stamp
        }
        (None, Some(mut after)) if late => {
            // Nothing from before the call, and a read long after it.
            after.uncertain = true;
            after
        }
        (None, Some(after)) => after,
        (Some(before), None) => before,
        (None, None) => Served {
            model: model.to_owned(),
            model_dir: None,
            slice_tag: None,
            engine_started: None,
            source: None,
            uncertain: false,
        },
    }
}

// ------------------------------------------------------------------ turns (the agent's thread)

struct Turn {
    id: String,
    calls: u32,
    /// Which stretch of the turn its calls belong to: 0 from its start, one
    /// more each time an operator message joins it (`operator_joined`). A
    /// mistake covers its call's stretch: everything since the operator last
    /// spoke.
    stretch: u32,
    /// The call whose reply is being archived: events appended after its
    /// response and before the next call (or the turn's end) are what it
    /// became.
    open: Option<(String, Vec<u64>)>,
}

thread_local! {
    static TURN: RefCell<Option<Turn>> = const { RefCell::new(None) };
}

/// `t-<UTC milliseconds>`, unique in this process. Within the same
/// millisecond, or after the clock steps back, it counts on from the last one
/// (`t-…Z-1`, `-2`, …) rather than waiting for the clock.
fn new_turn_id() -> String {
    static LAST: Mutex<(String, u32)> = Mutex::new((String::new(), 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let id = format!("t-{}", utc_stamp());
    if id > last.0 {
        *last = (id.clone(), 0);
        return id;
    }
    last.1 += 1;
    format!("{}-{}", last.0, last.1)
}

/// Ends the turn when dropped, however the turn ends.
pub struct TurnGuard {
    _private: (),
}

impl Drop for TurnGuard {
    fn drop(&mut self) {
        match RECORDER.get() {
            Some(recorder) => recorder.end_turn(),
            None => TURN.with(|turn| *turn.borrow_mut() = None),
        }
    }
}

/// A turn begins on this thread: archive events appended on it carry the
/// turn's id until the guard is dropped.
pub fn begin_turn() -> TurnGuard {
    let id = new_turn_id();
    TURN.with(|turn| *turn.borrow_mut() = Some(Turn { id, calls: 0, stretch: 0, open: None }));
    TurnGuard { _private: () }
}

/// An operator message joined the turn running on this thread (sent while
/// the agent worked): the calls after it start a new stretch.
pub fn operator_joined() {
    TURN.with(|turn| {
        if let Some(turn) = turn.borrow_mut().as_mut() {
            turn.stretch += 1;
        }
    });
}

/// The call on this thread whose reply is being archived, with the events
/// archived for it so far: the store has none of them until the next call
/// begins or the turn ends.
pub fn open_call_events() -> Option<(String, Vec<u64>)> {
    TURN.with(|turn| turn.borrow().as_ref().and_then(|t| t.open.clone()))
}

/// The turn running on this thread, for the archive events it appends.
pub fn current_turn() -> Option<String> {
    TURN.with(|turn| turn.borrow().as_ref().map(|t| t.id.clone()))
}

/// An archive event was appended on this thread. The agent's own events (its
/// reasoning, tool results, reply) are what the open call became; the
/// operator's carry the turn but belong to no call.
pub fn archived(event_id: u64, own: bool) {
    if !own {
        return;
    }
    TURN.with(|turn| {
        if let Some(Turn { open: Some((_, events)), .. }) = turn.borrow_mut().as_mut() {
            events.push(event_id);
        }
    });
}

// ------------------------------------------------------------------ recording

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// ds41d's: where the cached part came from (`none`, `live`, `ram`, `disk`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_source: Option<crate::llm::CacheSource>,
}

/// What the generation streaming now has produced, for saving a call that
/// is interrupted: it starts over when a retry or a resample starts a new
/// generation (`StreamEvent::Started`), so two generations are never glued
/// into one response, and keeps that generation's token counts.
#[derive(Debug, Default)]
pub struct Streamed {
    pub content: String,
    pub reasoning: String,
    pub usage: Usage,
}

impl Streamed {
    pub fn observe(&mut self, event: &crate::llm::StreamEvent) {
        use crate::llm::StreamEvent;
        match event {
            StreamEvent::Started => *self = Self::default(),
            StreamEvent::Reasoning(text) => self.reasoning.push_str(text),
            StreamEvent::Content(text) => self.content.push_str(text),
            StreamEvent::ToolArguments(_) | StreamEvent::Waiting => {}
            StreamEvent::Usage { completion_tokens, prompt_tokens, cached_tokens, cache_source, .. } => {
                self.usage.completion_tokens = Some(*completion_tokens);
                self.usage.prompt_tokens = prompt_tokens.or(self.usage.prompt_tokens);
                self.usage.cached_tokens = cached_tokens.or(self.usage.cached_tokens);
                self.usage.cache_source = cache_source.or(self.usage.cache_source);
            }
        }
    }
}

/// A call about to be sent.
pub struct Ticket {
    call_id: String,
    turn_id: String,
    call_in_turn: u32,
    stretch: u32,
    at: String,
    served_before: Option<Served>,
}

/// How a call ended.
pub enum Outcome<'a> {
    Response(&'a S0maResponse),
    /// The operator stopped it mid-stream: what had streamed so far.
    Interrupted { content: &'a str, reasoning: &'a str },
}

/// A saved (or being saved) call, as `/mistake` refers to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallRef {
    pub call_id: String,
    pub turn_id: String,
    pub call_in_turn: u32,
    /// The stretch of the turn it belongs to (`Turn::stretch`); 0 on calls
    /// saved before stretches were recorded.
    pub stretch: u32,
    pub at: String,
}

/// A `/mistake` typed while a call was in flight: applied to that call when it
/// is saved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingFlag {
    pub note: String,
    pub weakness: Option<String>,
}

pub enum FlagEvent {
    /// The call is in the store; log the mistake.
    Logged(CallRef, PendingFlag),
    /// The call in flight ended without being saved.
    Dropped(PendingFlag, String),
}

pub type FlagSink = Box<dyn Fn(FlagEvent) + Send + Sync>;

/// What `/mistake` without `@k` refers to.
pub enum Latest {
    /// The newest saved call.
    Call(CallRef),
    /// A call is streaming now; the flag waits for it.
    Deferred { turn_id: String, call_in_turn: u32 },
    None,
}

#[derive(Default)]
struct Live {
    in_flight: Option<(String, u32)>,
    pending: Vec<PendingFlag>,
    latest: Option<CallRef>,
    /// Save outcomes of recent calls: `Ok` stored, `Err` why not.
    outcomes: VecDeque<(String, std::result::Result<(), String>)>,
}

enum Job {
    Call(Box<CallJob>),
    Update { call_id: String, events: Vec<u64> },
    Barrier(Sender<()>),
}

struct CallJob {
    call: CallRef,
    finished: std::time::Instant,
    served_before: Option<Served>,
    body: Vec<u8>,
    response: Vec<u8>,
    finish: String,
    usage: Usage,
    flags: Vec<PendingFlag>,
}

pub struct Recorder {
    store: Store,
    sink: Arc<FlagSink>,
    base_url: String,
    model: String,
    session: String,
    seq: AtomicU64,
    jobs: Mutex<Sender<Job>>,
    live: Arc<Mutex<Live>>,
}

static RECORDER: OnceLock<Recorder> = OnceLock::new();

#[derive(Serialize)]
struct ResponseBlob<'a> {
    content: &'a str,
    reasoning_content: &'a str,
    tool_calls: &'a [ToolCall],
}

#[derive(Serialize)]
struct Splice {
    messages: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<usize>,
}

#[derive(Serialize)]
struct RequestRef {
    messages: Vec<String>,
    tools: Option<String>,
    params: Box<RawValue>,
    skeleton: String,
    splice: Splice,
    body_sha256: String,
    body_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<String>,
}

#[derive(Serialize)]
struct IndexLine<'a> {
    call_id: &'a str,
    at: &'a str,
    harness_version: &'a str,
    session: &'a str,
    turn_id: &'a str,
    call_in_turn: u32,
    /// The stretch of the turn (`Turn::stretch`): calls with the same turn
    /// and stretch came after the same operator message.
    stretch: u32,
    purpose: &'static str,
    served: Served,
    request: RequestRef,
    system_spans: Option<Vec<Span>>,
    response: String,
    finish: &'a str,
    usage: Usage,
    archive_events: Vec<u64>,
}

#[derive(Serialize)]
struct UpdateLine<'a> {
    call_id: &'a str,
    update: bool,
    archive_events: &'a [u64],
}

/// `s0ma <version>+<12 hex of the binary's sha256>`: which build saved a call.
fn harness_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        let build = std::env::current_exe()
            .ok()
            .and_then(|exe| fs::read(exe).ok())
            .map(|bytes| sha256_hex(&bytes)[..12].to_owned())
            .unwrap_or_else(|| "unknown".into());
        format!("s0ma {}+{build}", env!("CARGO_PKG_VERSION"))
    })
}

impl Recorder {
    /// Open the store under `data_dir` and start its writer thread.
    pub fn start(data_dir: &Path, base_url: String, model: String, sink: FlagSink) -> io::Result<Self> {
        let (recorder, receiver) = Self::open(data_dir, base_url, model, sink)?;
        recorder.spawn_writer(receiver, data_dir.to_path_buf());
        Ok(recorder)
    }

    fn open(data_dir: &Path, base_url: String, model: String, sink: FlagSink) -> io::Result<(Self, Receiver<Job>)> {
        let store = Store::at(data_dir);
        fs::create_dir_all(&store.root)?;
        if let Some(dropped) = store.repair_torn_tail()? {
            crate::llm::background_status(format_args!(
                "[calls] index.jsonl ended with a {dropped}-byte partial line (a process killed mid-write); dropped it"
            ));
        }
        let mut latest = None;
        let mut seq = 0;
        for line in store.lines_newest_first()? {
            let Ok(value) = serde_json::from_slice::<Value>(&line?) else { continue };
            let Some(id) = value.get("call_id").and_then(Value::as_str) else { continue };
            seq = seq.max(call_seq(id).unwrap_or(0));
            if value.get("update").is_none() && latest.is_none() {
                latest = call_ref(&value);
            }
            if latest.is_some() {
                break;
            }
        }
        let (sender, receiver) = mpsc::channel();
        let recorder = Self {
            store,
            sink: Arc::new(sink),
            base_url,
            model,
            session: format!("s-{}-{}", utc_stamp(), std::process::id()),
            seq: AtomicU64::new(seq),
            jobs: Mutex::new(sender),
            live: Arc::new(Mutex::new(Live { latest, ..Live::default() })),
        };
        Ok((recorder, receiver))
    }

    fn send(&self, job: Job) {
        let _ = self.jobs.lock().unwrap_or_else(|e| e.into_inner()).send(job);
    }

    fn live(&self) -> std::sync::MutexGuard<'_, Live> {
        self.live.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn archive_events(&self, call_id: String, events: Vec<u64>) {
        if !events.is_empty() {
            self.send(Job::Update { call_id, events });
        }
    }

    /// A main-agent call is about to be sent. Reads what the engine serves
    /// (a local request, bounded by a short timeout).
    pub fn begin(&self) -> Ticket {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        let call_id = format!("c-{}-{seq:06}", utc_stamp());
        // The previous call's reply is complete once the next call begins.
        let (turn_id, call_in_turn, stretch, finished) = TURN.with(|turn| {
            let mut turn = turn.borrow_mut();
            match turn.as_mut() {
                Some(turn) => {
                    turn.calls += 1;
                    (turn.id.clone(), turn.calls, turn.stretch, turn.open.take())
                }
                // A call outside any turn (none today) still gets one.
                None => (new_turn_id(), 1, 0, None),
            }
        });
        if let Some((previous, events)) = finished {
            self.archive_events(previous, events);
        }
        self.live().in_flight = Some((turn_id.clone(), call_in_turn));
        let served_before = served_with(SERVED_TIMEOUT_BEFORE, &self.base_url, &self.model);
        Ticket { call_id, turn_id, call_in_turn, stretch, at: local_now(), served_before }
    }

    /// The call ended. `body` is what was POSTed; `None` when nothing was sent.
    pub fn finish(&self, ticket: Ticket, body: Option<Vec<u8>>, outcome: Outcome<'_>, usage: Usage) {
        let call = CallRef {
            call_id: ticket.call_id,
            turn_id: ticket.turn_id,
            call_in_turn: ticket.call_in_turn,
            stretch: ticket.stretch,
            at: ticket.at,
        };
        let (response, finish) = match outcome {
            Outcome::Response(response) => (
                ResponseBlob {
                    content: &response.content,
                    reasoning_content: &response.reasoning_content,
                    tool_calls: &response.tool_calls,
                },
                response.finish_reason.clone().unwrap_or_else(|| "unknown".into()),
            ),
            Outcome::Interrupted { content, reasoning } => (
                ResponseBlob { content, reasoning_content: reasoning, tool_calls: &[] },
                "interrupted".to_owned(),
            ),
        };
        let response = serde_json::to_vec(&response).unwrap_or_default();
        let flags = {
            let mut live = self.live();
            live.in_flight = None;
            std::mem::take(&mut live.pending)
        };
        let Some(body) = body.filter(|_| !response.is_empty()) else {
            self.abandon_flags(flags, "the call in progress was never sent".into());
            return;
        };
        self.live().latest = Some(call.clone());
        TURN.with(|turn| {
            if let Some(turn) = turn.borrow_mut().as_mut()
                && turn.id == call.turn_id {
                    turn.open = Some((call.call_id.clone(), Vec::new()));
                }
        });
        self.send(Job::Call(Box::new(CallJob {
            call,
            finished: std::time::Instant::now(),
            served_before: ticket.served_before,
            body,
            response,
            finish,
            usage,
            flags,
        })));
    }

    /// The call failed without a response worth keeping (a transport error).
    pub fn abandon(&self, _ticket: Ticket, reason: &str) {
        let flags = {
            let mut live = self.live();
            live.in_flight = None;
            std::mem::take(&mut live.pending)
        };
        self.abandon_flags(flags, format!("the call in progress failed ({reason}) and was not saved"));
    }

    fn abandon_flags(&self, flags: Vec<PendingFlag>, reason: String) {
        for flag in flags {
            (self.sink)(FlagEvent::Dropped(flag, reason.clone()));
        }
    }

    /// End the turn on this thread: what its last call became is recorded.
    /// (`TurnGuard` does this for the global recorder.)
    pub fn end_turn(&self) {
        let ended = TURN.with(|turn| turn.borrow_mut().take());
        if let Some(Turn { open: Some((call_id, events)), .. }) = ended {
            self.archive_events(call_id, events);
        }
    }

    /// The newest call that finished (saved, or on its way to disk). While
    /// the agent's tools run, that is the call that asked for them.
    pub fn latest(&self) -> Option<CallRef> {
        self.live().latest.clone()
    }

    /// What `/mistake` refers to: the call in flight (the flag waits for it)
    /// or the newest saved call.
    pub fn latest_or_defer(&self, flag: PendingFlag) -> (Latest, Option<PendingFlag>) {
        let mut live = self.live();
        if let Some((turn_id, call_in_turn)) = live.in_flight.clone() {
            live.pending.push(flag);
            return (Latest::Deferred { turn_id, call_in_turn }, None);
        }
        match live.latest.clone() {
            Some(call) => (Latest::Call(call), Some(flag)),
            None => (Latest::None, Some(flag)),
        }
    }

    /// Wait until every call handed to the writer so far is written.
    pub fn sync(&self, timeout: Duration) -> bool {
        let (sender, receiver) = mpsc::channel();
        self.send(Job::Barrier(sender));
        receiver.recv_timeout(timeout).is_ok()
    }

    /// Why a recent call is not in the store, if it failed to save.
    pub fn save_error(&self, call_id: &str) -> Option<String> {
        self.live().outcomes.iter().rev().find(|(id, _)| id == call_id).and_then(|(_, r)| r.clone().err())
    }

    fn spawn_writer(&self, receiver: Receiver<Job>, data_dir: PathBuf) {
        let sink = Arc::clone(&self.sink);
        let store = self.store.clone();
        let live = Arc::clone(&self.live);
        let base_url = self.base_url.clone();
        let model = self.model.clone();
        let session = self.session.clone();
        std::thread::Builder::new()
            .name("s0ma-calls".into())
            .spawn(move || {
                crate::llm::background_scope(|| {
                    if let Some(days) = keep_days() {
                        match prune(&store, &data_dir, days) {
                            Ok(Some(report)) => crate::llm::background_status(format_args!("[calls] {report}")),
                            Ok(None) => {}
                            Err(error) => crate::llm::background_status(format_args!("[calls] retention failed: {error}")),
                        }
                    }
                    for job in receiver {
                        match job {
                            Job::Call(job) => {
                                let call = job.call.clone();
                                let flags = job.flags.clone();
                                let result = write_call(&store, &base_url, &model, &session, *job);
                                {
                                    let mut live = live.lock().unwrap_or_else(|e| e.into_inner());
                                    if live.outcomes.len() == REMEMBERED_OUTCOMES {
                                        live.outcomes.pop_front();
                                    }
                                    live.outcomes.push_back((
                                        call.call_id.clone(),
                                        result.as_ref().map(|_| ()).map_err(|e| e.to_string()),
                                    ));
                                }
                                if let Err(error) = &result {
                                    crate::llm::background_status(format_args!(
                                        "[calls] call {} failed to save: {error}",
                                        call.call_id
                                    ));
                                }
                                for flag in flags {
                                    match &result {
                                        Ok(()) => sink(FlagEvent::Logged(call.clone(), flag)),
                                        Err(error) => sink(FlagEvent::Dropped(
                                            flag,
                                            format!("call {} failed to save: {error}", call.call_id),
                                        )),
                                    }
                                }
                            }
                            Job::Update { call_id, events } => {
                                let line = UpdateLine { call_id: &call_id, update: true, archive_events: &events };
                                let written = serde_json::to_vec(&line)
                                    .map_err(io::Error::other)
                                    .and_then(|line| store.append_line(&line));
                                if let Err(error) = written {
                                    crate::llm::background_status(format_args!(
                                        "[calls] archive events of {call_id} not recorded: {error}"
                                    ));
                                }
                            }
                            Job::Barrier(done) => {
                                let _ = done.send(());
                            }
                        }
                    }
                })
            })
            .expect("spawn the call store's writer");
    }
}

pub fn call_ref(line: &Value) -> Option<CallRef> {
    Some(CallRef {
        call_id: line.get("call_id")?.as_str()?.to_owned(),
        turn_id: line.get("turn_id")?.as_str()?.to_owned(),
        call_in_turn: line.get("call_in_turn")?.as_u64()? as u32,
        stretch: line.get("stretch").and_then(Value::as_u64).unwrap_or(0) as u32,
        at: line.get("at")?.as_str()?.to_owned(),
    })
}

fn write_call(store: &Store, base_url: &str, model: &str, session: &str, job: CallJob) -> Result<()> {
    let late = job.finished.elapsed() > SERVED_LATE;
    let served_after = served_with(SERVED_TIMEOUT_AFTER, base_url, model);
    let served = merge_served(job.served_before, served_after, model, late);

    let split = split_body(&job.body)?;
    let body_sha256 = sha256_hex(&job.body);
    let rebuilt = rebuild(
        &split.skeleton,
        &split.messages,
        split.splice_messages,
        split.tools.as_deref(),
        split.splice_tools,
    )?;
    let mut messages = Vec::with_capacity(split.messages.len());
    for message in &split.messages {
        messages.push(blob_ref(&store.put(message)?));
    }
    let tools = match &split.tools {
        Some(tools) => Some(blob_ref(&store.put(tools)?)),
        None => None,
    };
    // Never trust the rebuild on faith: if it disagrees, keep the body too.
    let body = if sha256_hex(&rebuilt) == body_sha256 { None } else { Some(blob_ref(&store.put(&job.body)?)) };
    let response = blob_ref(&store.put(&job.response)?);

    let system_spans = split
        .messages
        .first()
        .and_then(|first| serde_json::from_slice::<Value>(first).ok())
        .filter(|first| first.get("role").and_then(Value::as_str) == Some("system"))
        .and_then(|first| first.get("content").and_then(Value::as_str).and_then(spans_for));

    let line = IndexLine {
        call_id: &job.call.call_id,
        at: &job.call.at,
        harness_version: harness_version(),
        session,
        turn_id: &job.call.turn_id,
        call_in_turn: job.call.call_in_turn,
        stretch: job.call.stretch,
        purpose: "agent",
        served,
        request: RequestRef {
            messages,
            tools,
            params: RawValue::from_string(split.params)?,
            skeleton: split.skeleton,
            splice: Splice { messages: split.splice_messages, tools: split.splice_tools },
            body_sha256,
            body_bytes: job.body.len(),
            body,
        },
        system_spans,
        response,
        finish: &job.finish,
        usage: job.usage,
        archive_events: Vec::new(),
    };
    store.append_line(&serde_json::to_vec(&line)?)?;
    Ok(())
}

// ------------------------------------------------------------------ retention

/// Drop every saved call made inside `windows`, or holding an archive event in
/// `ranges`, and the `/mistake` records that point at them. Returns what went.
///
/// A saved call holds the whole conversation as it was sent, so this is where
/// most of a redacted stretch actually lives: the archive keeps one line per
/// event, a call keeps every message of the request. Leaving these would make
/// the redaction cosmetic.
///
/// **Blobs are mark-and-sweep, exactly as in `prune`, and for the same
/// reason.** Message blobs are content-addressed, so consecutive calls share
/// almost all of them: deleting the blobs a dying line names would take out
/// text a surviving neighbour still needs. So the survivors are read first and
/// what they name is marked, and only unmarked objects are deleted.
///
/// Two selectors, because neither alone is complete. `at` is on the call line
/// and says when the call started, which catches a call whose events never
/// arrived (one interrupted, or the newest, whose events are still owed).
/// `archive_events` catches a call that started before the job opened and
/// archived into it. A call line carries an empty `archive_events` until an
/// update line fills it in, so the events are folded first.
///
/// The mistakes lock is held throughout, as `prune` holds it, so a flag cannot
/// land on a call while it is being dropped. Mistake records go before the
/// index is rewritten: a record outliving its call would point at nothing, and
/// every path that reads one expects the call to be there.
pub fn purge_window(
    data_dir: &Path,
    windows: &[(chrono::DateTime<chrono::Local>, chrono::DateTime<chrono::Local>)],
    ranges: &[(u64, u64)],
) -> Result<PurgedCalls> {
    use std::io::BufRead;
    let store = Store::at(data_dir);
    let index = store.index_path();
    let mut report = PurgedCalls::default();
    if !index.exists() || (windows.is_empty() && ranges.is_empty()) {
        return Ok(report);
    }
    let _lock = crate::mistakes::lock_mistakes(data_dir, Duration::from_secs(600))?;
    // Calls are written by a background thread off a queue. One already handed
    // to it but not yet on disk would be written *after* the index was read and
    // the sweep ran: its message blobs -- the whole conversation as it was sent
    // -- would survive the purge, and its own blobs could be swept out from
    // under the line that was about to name them. The barrier the recorder
    // already has for this waits for the queue to drain.
    if let Some(recorder) = RECORDER.get() && !recorder.sync(Duration::from_secs(30)) {
        return Err("the call store's writer did not finish what was queued within 30 s, so a call of the job \
                    could still be written after this; nothing was removed"
            .into());
    }
    let lines = || -> io::Result<_> {
        Ok(io::BufReader::new(File::open(&index)?).split(b'\n').filter(|line| {
            line.as_ref().map_or(true, |l| !l.iter().all(u8::is_ascii_whitespace))
        }))
    };
    let in_window = |at: &str| {
        chrono::DateTime::parse_from_rfc3339(at).is_ok_and(|at| {
            windows.iter().any(|(from, to)| at >= *from && at <= *to)
        })
    };
    let in_ranges = |events: &[u64]| {
        events.iter().any(|id| ranges.iter().any(|(from, through)| (*from..=*through).contains(id)))
    };

    // Pass 0: the events each call ended up with, from its update lines.
    let mut events: HashMap<String, Vec<u64>> = HashMap::new();
    for line in lines()? {
        let Ok(value) = serde_json::from_slice::<Value>(&line?) else { continue };
        if value.get("update").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        let Some(id) = value.get("call_id").and_then(Value::as_str) else { continue };
        let ids: Vec<u64> = value
            .get("archive_events")
            .and_then(Value::as_array)
            .map(|events| events.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        events.insert(id.to_owned(), ids);
    }

    // Pass 1: which calls go, and every blob the rest name.
    let mut doomed: HashSet<String> = HashSet::new();
    let mut marked: HashSet<String> = HashSet::new();
    for line in lines()? {
        let Ok(value) = serde_json::from_slice::<Value>(&line?) else { continue };
        if value.get("update").is_some() {
            continue;
        }
        let Some(id) = value.get("call_id").and_then(Value::as_str) else { continue };
        let own = events.get(id).map(Vec::as_slice).unwrap_or_default();
        let started = value.get("at").and_then(Value::as_str);
        if started.is_some_and(in_window) || in_ranges(own) {
            doomed.insert(id.to_owned());
            continue;
        }
        for pointer in ["/request/tools", "/request/body", "/response"] {
            if let Some(hex) = value.pointer(pointer).and_then(Value::as_str).and_then(blob_hex) {
                marked.insert(hex.to_owned());
            }
        }
        for reference in value.pointer("/request/messages").and_then(Value::as_array).into_iter().flatten() {
            if let Some(hex) = reference.as_str().and_then(blob_hex) {
                marked.insert(hex.to_owned());
            }
        }
    }
    if doomed.is_empty() {
        return Ok(report);
    }
    report.calls = doomed.len();

    // The mistakes on those calls, while the lock is still held.
    let dropped = crate::mistakes::drop_on_calls(data_dir, &doomed)?;
    report.mistakes = dropped.0;
    report.history_lines = dropped.1;

    // Pass 2: the lines that stay, byte for byte, in order.
    let temp = store.root.join(format!(".index.jsonl.purge-{}", std::process::id()));
    {
        let mut out = io::BufWriter::new(File::create(&temp)?);
        for line in lines()? {
            let line = line?;
            let drop_it = match line_call_id(&line) {
                Some(id) => std::str::from_utf8(id).is_ok_and(|id| doomed.contains(id)),
                None => serde_json::from_slice::<Value>(&line)
                    .ok()
                    .and_then(|v| v.get("call_id").and_then(Value::as_str).map(|id| doomed.contains(id)))
                    .unwrap_or(false),
            };
            if !drop_it {
                out.write_all(&line)?;
                out.write_all(b"\n")?;
            }
        }
        out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    }
    fs::rename(&temp, &index)?;
    if let Ok(dir) = File::open(&store.root) {
        dir.sync_all()?;
    }

    // Sweep: every object no surviving line names, which is the dropped calls'
    // own blobs and any orphan left by an earlier crash.
    if let Ok(prefixes) = fs::read_dir(store.objects()) {
        for prefix in prefixes {
            let prefix = prefix?;
            let head = prefix.file_name().to_string_lossy().into_owned();
            for object in fs::read_dir(prefix.path())? {
                let object = object?;
                let name = object.file_name().to_string_lossy().into_owned();
                let Some(rest) = name.strip_suffix(".zst") else { continue };
                if !marked.contains(&format!("{head}{rest}")) {
                    fs::remove_file(object.path())?;
                    report.blobs += 1;
                }
            }
        }
    }
    Ok(report)
}

/// Drop every saved call whose stored request or response holds any of
/// `fragments`, and the blobs the survivors do not name. Returns what went.
///
/// This is for a redaction of part of an event rather than all of it. The
/// archive keeps one copy of an event, but a saved call keeps every message of
/// the request it sent -- so the text of event #84119 is in the call that
/// produced it *and* in every later call of that conversation, as a message.
/// Selecting by `archive_events` finds only the first. The text itself is the
/// only thing that finds the rest.
///
/// So the objects are read, not the index: each message blob is decompressed
/// once and searched, and a call naming a blob that matched goes. Blobs are
/// content-addressed and deduplicated, so the shared message is one object
/// however many calls carried it, and the scan is over the store's distinct
/// text rather than over every call's copy of it.
///
/// A fragment has to be long enough to mean something -- `MIN_FRAGMENT` -- or
/// it would match everywhere and take the store with it. Shorter ones are
/// ignored and counted, so the caller can say what was skipped rather than
/// quietly leaving the text.
pub fn purge_fragments(data_dir: &Path, fragments: &[String]) -> Result<PurgedCalls> {
    use std::io::BufRead;
    let store = Store::at(data_dir);
    let index = store.index_path();
    let mut report = PurgedCalls::default();
    let wanted: Vec<&String> = fragments.iter().filter(|text| text.len() >= MIN_FRAGMENT).collect();
    report.fragments_too_short = fragments.len() - wanted.len();
    if !index.exists() || wanted.is_empty() {
        return Ok(report);
    }
    let _lock = crate::mistakes::lock_mistakes(data_dir, Duration::from_secs(600))?;
    if let Some(recorder) = RECORDER.get() && !recorder.sync(Duration::from_secs(30)) {
        return Err("the call store's writer did not finish what was queued within 30 s, so a call holding the \
                    text could still be written after this; nothing was removed"
            .into());
    }

    // Which objects hold it. Read once each, whatever names them.
    let mut tainted: HashSet<String> = HashSet::new();
    if let Ok(prefixes) = fs::read_dir(store.objects()) {
        for prefix in prefixes {
            let prefix = prefix?;
            let head = prefix.file_name().to_string_lossy().into_owned();
            for object in fs::read_dir(prefix.path())? {
                let object = object?;
                let name = object.file_name().to_string_lossy().into_owned();
                let Some(rest) = name.strip_suffix(".zst") else { continue };
                let hex = format!("{head}{rest}");
                let Ok(bytes) = store.get(&hex) else { continue };
                // The message is JSON, so the text is in it escaped. Both forms
                // are looked for: the raw one for a plain message, the escaped
                // one for anything with a newline or a quote in it.
                let haystack = String::from_utf8_lossy(&bytes).into_owned();
                if wanted.iter().any(|text| {
                    haystack.contains(text.as_str())
                        || serde_json::to_string(text)
                            .ok()
                            .and_then(|escaped| {
                                escaped.strip_prefix('"').and_then(|e| e.strip_suffix('"')).map(str::to_owned)
                            })
                            .is_some_and(|escaped| haystack.contains(&escaped))
                }) {
                    tainted.insert(hex);
                }
            }
        }
    }
    if tainted.is_empty() {
        return Ok(report);
    }

    let lines = || -> io::Result<_> {
        Ok(io::BufReader::new(File::open(&index)?).split(b'\n').filter(|line| {
            line.as_ref().map_or(true, |l| !l.iter().all(u8::is_ascii_whitespace))
        }))
    };
    let names_tainted = |value: &Value| {
        let mut named = Vec::new();
        for pointer in ["/request/tools", "/request/body", "/response"] {
            if let Some(hex) = value.pointer(pointer).and_then(Value::as_str).and_then(blob_hex) {
                named.push(hex.to_owned());
            }
        }
        for reference in value.pointer("/request/messages").and_then(Value::as_array).into_iter().flatten() {
            if let Some(hex) = reference.as_str().and_then(blob_hex) {
                named.push(hex.to_owned());
            }
        }
        named.iter().any(|hex| tainted.contains(hex))
    };

    // The same two passes as `purge_window`, and the same reason for them: what
    // the survivors name is marked first, so a blob shared with one of them is
    // not swept out from under it.
    let mut doomed: HashSet<String> = HashSet::new();
    let mut marked: HashSet<String> = HashSet::new();
    for line in lines()? {
        let Ok(value) = serde_json::from_slice::<Value>(&line?) else { continue };
        if value.get("update").is_some() {
            continue;
        }
        let Some(id) = value.get("call_id").and_then(Value::as_str) else { continue };
        if names_tainted(&value) {
            doomed.insert(id.to_owned());
            continue;
        }
        for pointer in ["/request/tools", "/request/body", "/response"] {
            if let Some(hex) = value.pointer(pointer).and_then(Value::as_str).and_then(blob_hex) {
                marked.insert(hex.to_owned());
            }
        }
        for reference in value.pointer("/request/messages").and_then(Value::as_array).into_iter().flatten() {
            if let Some(hex) = reference.as_str().and_then(blob_hex) {
                marked.insert(hex.to_owned());
            }
        }
    }
    if doomed.is_empty() {
        return Ok(report);
    }
    report.calls = doomed.len();
    let dropped = crate::mistakes::drop_on_calls(data_dir, &doomed)?;
    report.mistakes = dropped.0;
    report.history_lines = dropped.1;
    report.blobs = rewrite_without(&store, &doomed, &marked)?;
    Ok(report)
}

/// The shortest fragment worth searching the store for. Below this a match
/// says nothing: every call holds "the" and most hold a bare hostname.
pub const MIN_FRAGMENT: usize = 12;

/// Write the index again without `doomed`'s lines, then delete every object no
/// surviving line names. Shared by the two purges, since getting the order
/// wrong is what loses a survivor's blob.
fn rewrite_without(store: &Store, doomed: &HashSet<String>, marked: &HashSet<String>) -> Result<usize> {
    use std::io::BufRead;
    let index = store.index_path();
    let temp = store.root.join(format!(".index.jsonl.purge-{}", std::process::id()));
    {
        let mut out = io::BufWriter::new(File::create(&temp)?);
        for line in io::BufReader::new(File::open(&index)?).split(b'\n') {
            let line = line?;
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let drop_it = match line_call_id(&line) {
                Some(id) => std::str::from_utf8(id).is_ok_and(|id| doomed.contains(id)),
                None => serde_json::from_slice::<Value>(&line)
                    .ok()
                    .and_then(|v| v.get("call_id").and_then(Value::as_str).map(|id| doomed.contains(id)))
                    .unwrap_or(false),
            };
            if !drop_it {
                out.write_all(&line)?;
                out.write_all(b"\n")?;
            }
        }
        out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    }
    fs::rename(&temp, &index)?;
    if let Ok(dir) = File::open(&store.root) {
        dir.sync_all()?;
    }
    let mut blobs = 0;
    if let Ok(prefixes) = fs::read_dir(store.objects()) {
        for prefix in prefixes {
            let prefix = prefix?;
            let head = prefix.file_name().to_string_lossy().into_owned();
            for object in fs::read_dir(prefix.path())? {
                let object = object?;
                let name = object.file_name().to_string_lossy().into_owned();
                let Some(rest) = name.strip_suffix(".zst") else { continue };
                if !marked.contains(&format!("{head}{rest}")) {
                    // Counted and carried past rather than returned: the index
                    // is already written, and one object that will not unlink
                    // must not make a finished removal report as a failure.
                    match fs::remove_file(object.path()) {
                        Ok(()) => blobs += 1,
                        Err(error) => crate::llm::background_status(format_args!(
                            "[calls] a stored object was not deleted, so its bytes stay in the store: {error}"
                        )),
                    }
                }
            }
        }
    }
    Ok(blobs)
}

/// What `purge_window` removed.
#[derive(Debug, Default, Clone, Copy)]
pub struct PurgedCalls {
    pub calls: usize,
    pub blobs: usize,
    pub mistakes: usize,
    pub history_lines: usize,
    /// Fragments too short to search the store for (`MIN_FRAGMENT`), so the
    /// caller can say what it did not look for rather than implying it did.
    pub fragments_too_short: usize,
}

fn keep_days() -> Option<u64> {
    std::env::var("S0MA_CALLS_KEEP_DAYS").ok()?.trim().parse().ok().filter(|days| *days > 0)
}

fn referenced_calls(data_dir: &Path) -> io::Result<HashSet<String>> {
    let mut calls = HashSet::new();
    let Ok(entries) = fs::read_dir(data_dir.join("mistakes")) else { return Ok(calls) };
    for entry in entries {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        // The flagged call, and every step of its stretch: a correction can
        // target any of them, and trains on that step's own request.
        if let Ok(value) = serde_json::from_slice::<Value>(&fs::read(&path)?) {
            if let Some(id) = value.get("call_id").and_then(Value::as_str) {
                calls.insert(id.to_owned());
            }
            for step in value.get("stretch").and_then(Value::as_array).into_iter().flatten() {
                if let Some(id) = step.get("call_id").and_then(Value::as_str) {
                    calls.insert(id.to_owned());
                }
            }
        }
    }
    Ok(calls)
}

/// Mark and sweep: keep the lines of calls newer than `days` or referenced by
/// a mistake (and their update lines), rewrite the index atomically, then
/// delete the blobs no kept line names. Runs on the writer thread before it
/// takes any call, so nothing is appended meanwhile, and holds the mistakes
/// lock throughout, so no mistake can be logged on a call it is dropping.
/// Streams the index twice rather than holding it in memory.
pub fn prune(store: &Store, data_dir: &Path, days: u64) -> Result<Option<String>> {
    use std::io::BufRead;
    let index = store.index_path();
    if !index.exists() {
        return Ok(None);
    }
    let _lock = crate::mistakes::lock_mistakes(data_dir, Duration::from_secs(600))?;
    let referenced = referenced_calls(data_dir)?;
    let cutoff = chrono::Utc::now() - chrono::Duration::days(days as i64);
    let lines = || -> io::Result<_> {
        Ok(io::BufReader::new(File::open(&index)?).split(b'\n').filter(|line| {
            line.as_ref().map_or(true, |l| !l.iter().all(u8::is_ascii_whitespace))
        }))
    };

    // Pass 1: which calls stay, and every blob they name.
    let mut kept: HashSet<String> = HashSet::new();
    let mut marked: HashSet<String> = HashSet::new();
    let mut total = 0;
    for line in lines()? {
        let Ok(value) = serde_json::from_slice::<Value>(&line?) else { continue };
        if value.get("update").is_some() {
            continue;
        }
        let Some(id) = value.get("call_id").and_then(Value::as_str) else { continue };
        total += 1;
        let recent = value
            .get("at")
            .and_then(Value::as_str)
            .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
            .is_none_or(|at| at >= cutoff);
        if !(recent || referenced.contains(id)) {
            continue;
        }
        kept.insert(id.to_owned());
        for pointer in ["/request/tools", "/request/body", "/response"] {
            if let Some(hex) = value.pointer(pointer).and_then(Value::as_str).and_then(blob_hex) {
                marked.insert(hex.to_owned());
            }
        }
        for reference in value.pointer("/request/messages").and_then(Value::as_array).into_iter().flatten() {
            if let Some(hex) = reference.as_str().and_then(blob_hex) {
                marked.insert(hex.to_owned());
            }
        }
    }
    if kept.len() == total {
        return Ok(None);
    }

    // Pass 2: the kept lines, in order, to a temp file renamed over the index.
    let temp = store.root.join(format!(".index.jsonl.tmp-{}", std::process::id()));
    {
        let mut out = io::BufWriter::new(File::create(&temp)?);
        for line in lines()? {
            let line = line?;
            let keep = match line_call_id(&line) {
                Some(id) => std::str::from_utf8(id).is_ok_and(|id| kept.contains(id)),
                None => serde_json::from_slice::<Value>(&line)
                    .ok()
                    .and_then(|v| v.get("call_id").and_then(Value::as_str).map(|id| kept.contains(id)))
                    .unwrap_or(false),
            };
            if keep {
                out.write_all(&line)?;
                out.write_all(b"\n")?;
            }
        }
        out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    }
    fs::rename(&temp, &index)?;
    if let Ok(dir) = File::open(&store.root) {
        dir.sync_all()?;
    }

    // Sweep: blobs no kept line names.
    let mut deleted = 0;
    if let Ok(prefixes) = fs::read_dir(store.objects()) {
        for prefix in prefixes {
            let prefix = prefix?;
            let head = prefix.file_name().to_string_lossy().into_owned();
            for object in fs::read_dir(prefix.path())? {
                let object = object?;
                let name = object.file_name().to_string_lossy().into_owned();
                let Some(rest) = name.strip_suffix(".zst") else { continue };
                if !marked.contains(&format!("{head}{rest}")) {
                    fs::remove_file(object.path())?;
                    deleted += 1;
                }
            }
        }
    }
    Ok(Some(format!(
        "retention ({days} days): kept {} of {total} calls ({} for mistakes), deleted {deleted} blobs",
        kept.len(),
        kept.iter().filter(|id| referenced.contains(*id)).count()
    )))
}

// ------------------------------------------------------------------ the global recorder

/// Start the call store for this agent. Calls made before this are not saved.
pub fn init(data_dir: &Path, sink: FlagSink) -> io::Result<()> {
    if RECORDER.get().is_some() {
        return Ok(());
    }
    let recorder = Recorder::start(data_dir, crate::llm::endpoint(""), crate::llm::model_name(), sink)?;
    let _ = RECORDER.set(recorder);
    Ok(())
}

pub fn recorder() -> Option<&'static Recorder> {
    RECORDER.get()
}

pub fn begin_call() -> Option<Ticket> {
    RECORDER.get().map(Recorder::begin)
}

pub fn finish_call(ticket: Option<Ticket>, body: Option<Vec<u8>>, outcome: Outcome<'_>, usage: Usage) {
    if let (Some(recorder), Some(ticket)) = (RECORDER.get(), ticket) {
        recorder.finish(ticket, body, outcome, usage);
    }
}

pub fn abandon_call(ticket: Option<Ticket>, reason: &str) {
    if let (Some(recorder), Some(ticket)) = (RECORDER.get(), ticket) {
        recorder.abandon(ticket, reason);
    }
}

/// Wait (briefly) for a recorder's writer to finish what was queued, e.g. in
/// tests.
#[cfg(test)]
pub(crate) fn wait_written(recorder: &Recorder) {
    assert!(recorder.sync(Duration::from_secs(10)), "writer did not drain");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolDefinition;
    use crate::types::{FunctionCall, Message, ToolCall};
    use serde_json::json;


    use std::sync::atomic::AtomicUsize;

    fn temp_dir(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "s0ma-calls-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Nothing listens here: the engine reads fail fast.
    fn no_engine() -> String {
        "http://127.0.0.1:9".into()
    }

    fn quiet() -> FlagSink {
        Box::new(|_| {})
    }

    fn tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition::function(
            "run_command",
            "Run a command — carefully.",
            json!({"type": "object", "properties": {"command": {"type": "string"}}}),
        )]
    }

    /// Escapes, control characters, non-ASCII, a tool call and its result, an image.
    fn tricky_messages() -> Vec<Message> {
        let call = ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: FunctionCall {
                name: "run_command".into(),
                arguments: r#"{"command":"echo \"hi\" \\ \u001b[0m é"}"#.into(),
            },
        };
        vec![
            Message::system("SYSTEM ⚙ base \u{1b}[31m red \"quoted\" \\ back </script>\n[KNOWN WEAKNESSES]\n- wk-0001\n".into()),
            Message::user("hello 👋\r\nline two\u{0}nul\u{7f}".into()),
            Message::assistant_with_tools("Running it — “now”.".into(), vec![call]).with_reasoning("thinking…\n\ttab".into()),
            Message::tool("call_1".into(), "{\"ok\":true}".into()),
            Message::user_with_image("look at this".into(), "data:image/png;base64,iVBORw0KGgo=".into()),
        ]
    }

    fn response() -> S0maResponse {
        S0maResponse {
            content: "Done — ✓ \"quoted\"".into(),
            reasoning_content: "it started at 09:02\n".into(),
            tool_calls: vec![ToolCall {
                id: "call_2".into(),
                kind: "function".into(),
                function: FunctionCall { name: "run_command".into(), arguments: "{\"command\":\"ls\"}".into() },
            }],
            finish_reason: Some("tool_calls".into()),
        }
    }

    fn record_one(recorder: &Recorder, body: &[u8], response: &S0maResponse) -> String {
        let ticket = recorder.begin();
        let id = ticket.call_id.clone();
        recorder.finish(ticket, Some(body.to_vec()), Outcome::Response(response), Usage::default());
        wait_written(recorder);
        id
    }

    #[test]
    fn a_blob_that_cannot_be_written_leaves_no_line() {
        let dir = temp_dir("noblob");
        let recorder = Recorder::start(&dir, no_engine(), "m".into(), quiet()).unwrap();
        // The objects folder can't be created: every blob write fails.
        fs::write(dir.join("calls/objects"), b"not a folder").unwrap();
        let body = crate::llm::main_request_body_for_test(&tricky_messages(), &tools());
        let id = record_one(&recorder, &body, &response());
        assert!(recorder.save_error(&id).is_some());
        assert!(!Store::at(&dir).index_path().exists() || Store::at(&dir).recent(1).unwrap().is_empty());
    }

    #[test]
    fn a_body_the_rebuild_cannot_reproduce_is_stored_whole() {
        let dir = temp_dir("whole");
        let recorder = Recorder::start(&dir, no_engine(), "m".into(), quiet()).unwrap();
        let body = br#"{ "model": "m", "messages": [ {"role": "user", "content": "x"} ], "stream": true }"#;
        let id = record_one(&recorder, body, &response());
        let store = Store::at(&dir);
        let line = store.find(&id).unwrap().unwrap();
        assert!(line["request"]["body"].is_string(), "{line}");
        assert_eq!(store.body(&line).unwrap(), body);
        assert_eq!(store.messages(&line).unwrap()[0]["content"], "x");
    }

    #[test]
    fn an_index_line_never_names_a_missing_blob() {
        let dir = temp_dir("blobs");
        let recorder = Recorder::start(&dir, no_engine(), "m".into(), quiet()).unwrap();
        let body = crate::llm::main_request_body_for_test(&tricky_messages(), &tools());
        let first = record_one(&recorder, &body, &response());
        // The index can't be appended to: the blobs are written first, so the
        // failure leaves orphans and no line naming them.
        let index = Store::at(&dir).index_path();
        let saved = fs::read(&index).unwrap();
        fs::remove_file(&index).unwrap();
        fs::create_dir(&index).unwrap();
        let mut other = response();
        other.content = "a different reply".into();
        let failed = record_one(&recorder, &body, &other);
        assert!(recorder.save_error(&failed).is_some());
        fs::remove_dir(&index).unwrap();
        fs::write(&index, saved).unwrap();
        record_one(&recorder, &body, &response());

        let store = Store::at(&dir);
        let calls = store.recent(10).unwrap();
        assert_eq!(calls.len(), 2);
        assert!(store.find(&failed).unwrap().is_none());
        assert!(store.find(&first).unwrap().is_some());
        for line in &calls {
            store.body(line).unwrap();
            store.response(line).unwrap();
            assert_eq!(store.messages(line).unwrap().len(), 5);
        }
        // Blobs are shared: the five messages and the tools are stored once.
        let objects = walk(&store.objects());
        assert!(objects.len() <= 5 + 1 + 2, "{} objects", objects.len());
        // No temp files survive a successful write.
        assert!(objects.iter().all(|p| !p.file_name().unwrap().to_string_lossy().starts_with('.')));
    }

    fn walk(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
            if entry.path().is_dir() {
                out.extend(walk(&entry.path()));
            } else {
                out.push(entry.path());
            }
        }
        out
    }

    #[test]
    fn lines_read_backwards_across_chunk_boundaries() {
        let dir = temp_dir("rev");
        let path = dir.join("f.jsonl");
        let long = "a".repeat(300_000);
        let longer = "b".repeat(600_000);
        fs::write(&path, format!("{long}\nshort\n\n{longer}\nunterminated")).unwrap();
        let lines: Vec<Vec<u8>> = RevLines::open(&path).unwrap().map(|l| l.unwrap()).collect();
        assert_eq!(lines, vec![longer.into_bytes(), b"short".to_vec(), long.into_bytes()]);
        assert_eq!(RevLines::open(&dir.join("missing")).unwrap().count(), 0);
    }

    #[test]
    fn retention_keeps_recent_calls_and_every_call_a_mistake_names() {
        let dir = temp_dir("prune");
        let recorder = Recorder::start(&dir, no_engine(), "m".into(), quiet()).unwrap();
        let body = crate::llm::main_request_body_for_test(&tricky_messages(), &tools());
        let mut replies = Vec::new();
        for reply in ["old and unreferenced", "old but flagged", "recent"] {
            let mut r = response();
            r.content = reply.into();
            replies.push(record_one(&recorder, &body, &r));
        }
        let store = Store::at(&dir);
        let old = |id: &str| store.find(id).unwrap().unwrap()["at"].as_str().unwrap().to_owned();
        let mut index = fs::read_to_string(store.index_path()).unwrap();
        for id in &replies[..2] {
            index = index.replace(&old(id), "2020-01-01T00:00:00.000+00:00");
        }
        fs::write(store.index_path(), index).unwrap();
        fs::create_dir_all(dir.join("mistakes")).unwrap();
        fs::write(dir.join("mistakes/1.json"), json!({"mistake": 1, "call_id": replies[1]}).to_string()).unwrap();
        let unique = store.find(&replies[0]).unwrap().unwrap()["response"].as_str().unwrap().to_owned();

        let report = prune(&store, &dir, 30).unwrap().expect("something to prune");
        assert!(report.contains("kept 2 of 3 calls (1 for mistakes)"), "{report}");
        assert!(store.find(&replies[0]).unwrap().is_none());
        assert!(store.get(&unique).is_err(), "its own blob is gone");
        for id in &replies[1..] {
            let line = store.find(id).unwrap().unwrap();
            assert_eq!(store.body(&line).unwrap(), body, "shared blobs of kept calls survive");
            store.response(&line).unwrap();
        }
        assert_eq!(prune(&store, &dir, 30).unwrap(), None, "nothing more to do");
    }
}
