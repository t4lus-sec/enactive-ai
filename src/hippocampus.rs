use std::collections::{HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::Local;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum MemoryEventKind {
    Operator {
        content: String,
    },

    S0ma {
        content: String,

        #[serde(default)]
        intermediate: bool,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct MemoryEvent {
    id: u64,
    timestamp: String,

    /// The turn this event belongs to, when it was appended during one (the
    /// operator's message, the agent's reasoning, tool results and reply): how a
    /// saved model call (calls.rs) joins the archive without guessing. Absent
    /// on events from before it existed and on background writers'.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    turn_id: Option<String>,

    #[serde(flatten)]
    event: MemoryEventKind,
}

impl MemoryEvent {
    fn operator(id: u64, content: String) -> Self {
        Self {
            id,
            timestamp: Local::now().to_rfc3339(),
            turn_id: None,
            event: MemoryEventKind::Operator { content },
        }
    }

    fn s0ma(id: u64, content: String) -> Self {
        Self {
            id,
            timestamp: Local::now().to_rfc3339(),
            turn_id: None,
            event: MemoryEventKind::S0ma {
                content,
                intermediate: false,
            },
        }
    }

    fn s0ma_intermediate(id: u64, content: String) -> Self {
        Self {
            id,
            timestamp: Local::now().to_rfc3339(),
            turn_id: None,
            event: MemoryEventKind::S0ma {
                content,
                intermediate: true,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    Operator,
    S0ma,
}

#[derive(Debug, Clone)]
pub struct RecalledEvent {
    pub id: u64,
    pub timestamp: String,
    pub speaker: Speaker,
    pub content: String,
}

/// What a redaction replaced: one entry per event whose content was removed.
#[derive(Debug, Default, Clone)]
pub struct Redacted {
    pub events: usize,
    pub ids: Vec<u64>,
    /// Lines that would not parse and were replaced whole rather than edited.
    pub unreadable: usize,
    /// What the operator's redacted events held, for a caller that has to find
    /// the same text somewhere else and take it out too -- the prompt's typed
    /// history keeps a copy with nothing but the text to match on. Held only
    /// for the length of that call and never written anywhere.
    pub typed: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub operator_event_id: u64,
    pub s0ma_event_id: u64,
    pub operator: String,
    pub s0ma: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCheckpoint {
    pub timestamp: String,
    pub from_event_id: u64,
    pub through_event_id: u64,
    pub source_event_count: usize,
    pub summary: String,
}

/// The archive. A clone is another handle on the same archive: the next event
/// ID is shared and taken under one lock, together with the append, so events
/// from several threads (the agent's, and the evidence a subagent cites in a board
/// fact) land in ID order, one line per ID.
/// Held across every rewrite of the checkpoints file: sleep appends to it on
/// the app's thread, and a redaction can rewrite it from the tab's.
static CHECKPOINTS: Mutex<()> = Mutex::new(());

#[derive(Clone)]
pub struct Hippocampus {
    path: PathBuf,
    checkpoint_path: PathBuf,
    next_id: Arc<Mutex<u64>>,
}

/// A line struck: its number, the text it had, how many marks it took.
type Struck = (usize, String, usize);

impl Hippocampus {
    /// Byte-aware eligibility cutoff.
    ///
    /// `compaction_boundary` measures protection in *spoken* turns, but tool
    /// results, reasoning records and subagent summaries are all intermediate
    /// events and do not advance that count. One agentic reply can therefore drag
    /// tens of thousands of tokens behind it, and a context can fill long before
    /// nine completed turns exist — at which point the turn rule protects the whole
    /// post-checkpoint range and nothing is ever eligible.
    ///
    /// This keeps the turn rule as the default and treats `keep_bytes` as a ceiling
    /// on the protected tail: when the tail is over budget the cutoff advances
    /// toward the present one completed turn at a time, never protecting fewer than
    /// `min_keep_turns` replies. The cutoff always lands on a completed turn, so a
    /// turn is never split across the boundary.
    ///
    /// Sizes are measured on the serialized archive line, which slightly over-counts
    /// against replay cost. That errs toward compacting more, never less.
    pub fn compaction_boundary_budgeted(
        &self,
        keep_turns: usize,
        keep_bytes: usize,
        min_keep_turns: usize,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if !self.path.exists() {
            return Ok(0);
        }

        // (event id, serialized size, is a completed-turn end)
        let mut events: Vec<(u64, usize, bool)> = Vec::new();

        for line in complete_lines(File::open(&self.path)?) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let event: MemoryEvent = serde_json::from_str(&line)?;
            let end = matches!(
                event.event,
                MemoryEventKind::S0ma {
                    intermediate: false,
                    ..
                }
            );
            events.push((event.id, line.len(), end));
        }

        let ends: Vec<usize> = events
            .iter()
            .enumerate()
            .filter(|(_, event)| event.2)
            .map(|(index, _)| index)
            .collect();

        // Index of the completed turn that a given protection window would cut at.
        let cut_for = |protected: usize| -> Option<usize> {
            ends.len()
                .checked_sub(protected)?
                .checked_sub(1)
                .map(|slot| ends[slot])
        };

        let trailing = |cut: Option<usize>| -> usize {
            let from = cut.map_or(0, |index| index + 1);
            events[from..].iter().map(|event| event.1).sum()
        };

        // The turn rule alone. `None` means fewer completed turns exist than the
        // window protects, which is the degenerate case where the turn-only rule
        // yields nothing eligible at all.
        let mut chosen = cut_for(keep_turns);

        if trailing(chosen) > keep_bytes {
            // The floor: never advance past this, so recent dialogue survives
            // verbatim as the rotated context's only continuity.
            if let Some(floor) = cut_for(min_keep_turns.min(keep_turns)) {
                for &candidate in &ends {
                    if candidate > floor {
                        break;
                    }
                    if chosen.is_some_and(|cut| candidate <= cut) {
                        continue;
                    }
                    chosen = Some(candidate);
                    if trailing(chosen) <= keep_bytes {
                        break;
                    }
                }
            }
        }

        Ok(chosen.map_or(0, |index| events[index].0))
    }

    pub fn archive_path(&self) -> &std::path::Path {
        &self.path
    }

    /// ID of the newest event written so far (0 if the archive is empty).
    pub fn last_event_id(&self) -> u64 {
        self.next_id.lock().map_or_else(|poisoned| *poisoned.into_inner(), |next| *next).saturating_sub(1)
    }

    /// Append the event `make` builds for the next ID, and return that ID.
    fn append(&self, make: impl FnOnce(u64) -> MemoryEvent) -> Result<u64, Box<dyn std::error::Error>> {
        let mut next = self.next_id.lock().map_err(|_| "archive lock poisoned")?;
        let id = *next;
        let mut event = make(id);
        event.turn_id = crate::calls::current_turn();
        self.write_event(&event)?;
        *next += 1;
        drop(next);
        crate::calls::archived(id, matches!(event.event, MemoryEventKind::S0ma { .. }));
        Ok(id)
    }

    /// Replace the content of every event whose id falls in `ranges`, keeping
    /// each event's id, timestamp and speaker. Returns what it changed.
    ///
    /// **No line is ever removed.** An event's id is its line number
    /// (`count_events` derives the next id from the line count), so dropping a
    /// line would renumber every event after it and silently repoint every
    /// reference held elsewhere -- the call store's `archive_events`, both
    /// indexes, the weakness log's cited evidence. One line in, one line out,
    /// every id still its own.
    ///
    /// The rewrite goes to a file beside the archive and is renamed over it.
    /// That is atomic for readers, and it changes the inode, which is one of
    /// the things `ReadMark::still_holds` checks: the line index and both
    /// derived indexes notice and start over rather than reading on from
    /// offsets into a file that no longer exists.
    ///
    /// The append lock is held throughout, so an event cannot land between the
    /// read and the rename and be dropped by it.
    pub fn redact_ranges(
        &self,
        ranges: &[(u64, u64)],
        marker: &str,
    ) -> Result<Redacted, Box<dyn std::error::Error>> {
        if ranges.is_empty() {
            return Ok(Redacted::default());
        }
        let wanted = |id: u64| ranges.iter().any(|(from, through)| (*from..=*through).contains(&id));
        self.rewrite(|id, _| wanted(id).then(|| marker.to_owned()))
    }

    /// Replace one event's content with `text`, keeping its id, timestamp and
    /// speaker, and only if it still holds what `was` hashes.
    ///
    /// For a redaction of part of an event rather than all of it: the content
    /// is whatever a person approved, so this takes the text rather than
    /// composing a marker. `was` is the digest of the content the approval was
    /// made against (`content_digest`), so an event that has moved since --
    /// another redaction, a job's close -- is refused instead of having the
    /// approved text written over whatever is there now.
    pub fn replace_event_content(
        &self,
        id: u64,
        text: &str,
        was: &str,
    ) -> Result<Redacted, Box<dyn std::error::Error>> {
        match self.get_event(id)? {
            None => return Err(format!("event {id} is not in the archive").into()),
            Some(event) if content_digest(&event.content) != was => {
                return Err(format!(
                    "event {id} is not what this was decided against any more, so nothing was written; look at \
                     it again"
                )
                .into())
            }
            Some(_) => {}
        }
        self.rewrite(|at, _| (at == id).then(|| text.to_owned()))
    }

    /// Give every event the content `strike` returns for it (None leaves it
    /// as it is), in one rewrite. Returns the ids changed. For the memory
    /// tab's redactions, which go by what an event holds rather than by id.
    pub fn strike_events(
        &self,
        strike: impl Fn(&str) -> Option<String>,
    ) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
        let report = self.rewrite(|_, kind| match kind {
            MemoryEventKind::Operator { content } | MemoryEventKind::S0ma { content, .. } => strike(content),
        })?;
        Ok(report.ids)
    }

    /// Every event whose line `keep` passes (checked before it is parsed),
    /// in order.
    pub fn each_event_where(
        &self,
        keep: impl Fn(&str) -> bool,
        mut each: impl FnMut(RecalledEvent),
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.path.exists() {
            return Ok(());
        }
        for line in complete_lines(File::open(&self.path)?) {
            let line = line?;
            if line.trim().is_empty() || !keep(&line) {
                continue;
            }
            if let Ok(event) = serde_json::from_str::<MemoryEvent>(&line) {
                each(Self::to_recalled_event(event));
            }
        }
        Ok(())
    }

    /// Every event, in order, read from the file as it is now.
    pub fn each_event(
        &self,
        mut each: impl FnMut(RecalledEvent),
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.path.exists() {
            return Ok(());
        }
        for line in complete_lines(File::open(&self.path)?) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(event) = serde_json::from_str::<MemoryEvent>(&line) {
                each(Self::to_recalled_event(event));
            }
        }
        Ok(())
    }

    /// The checkpoints' summaries, each given what `strike` returns for it
    /// (the text, and how many occurrences it struck). With `apply` false
    /// nothing is written. Returns, for each checkpoint that holds something,
    /// its position, its time and the count.
    pub fn strike_checkpoints(
        &self,
        strike: &dyn Fn(&str) -> Option<(String, usize)>,
        apply: bool,
    ) -> Result<Vec<Struck>, Box<dyn std::error::Error>> {
        let _file = CHECKPOINTS.lock().map_err(|_| "checkpoints lock poisoned")?;
        if !self.checkpoint_path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&self.checkpoint_path)?;
        let mut out = String::new();
        let mut touched = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let changed = serde_json::from_str::<MemoryCheckpoint>(line)
                .ok()
                .and_then(|mut checkpoint| {
                    let (summary, count) = strike(&checkpoint.summary)?;
                    checkpoint.summary = summary;
                    Some((checkpoint, count))
                });
            match changed {
                Some((checkpoint, count)) => {
                    touched.push((index + 1, checkpoint.timestamp.clone(), count));
                    out.push_str(&serde_json::to_string(&checkpoint)?);
                }
                None => out.push_str(line),
            }
            out.push('\n');
        }
        if apply && !touched.is_empty() {
            let temp = self.checkpoint_path.with_extension("striking");
            {
                let mut file = File::create(&temp)?;
                file.write_all(out.as_bytes())?;
                file.sync_all()?;
            }
            fs::rename(temp, &self.checkpoint_path)?;
        }
        Ok(touched)
    }

    /// One pass over the archive, giving each event the content `content`
    /// returns for it and leaving the rest exactly as they were.
    ///
    /// **No line is ever removed**, which is the whole reason this is a rewrite
    /// rather than a filter: an event's id is its line number (`count_events`
    /// derives the next id from the line count), so dropping a line would
    /// renumber every event after it and silently repoint every reference held
    /// elsewhere -- the call store's `archive_events`, both indexes, the
    /// weakness log's cited evidence. One line in, one line out, every id still
    /// its own, and a count that disagrees is refused before anything is
    /// written.
    ///
    /// The rewrite goes to a file beside the archive and is renamed over it.
    /// That is atomic for readers, and it changes the inode, which is one of
    /// the things `ReadMark::still_holds` checks: the line index and both
    /// derived indexes notice and start over rather than reading on from
    /// offsets into a file that no longer exists.
    ///
    /// The append lock is held throughout, so an event cannot land between the
    /// read and the rename and be dropped by it.
    fn rewrite(
        &self,
        content: impl Fn(u64, &MemoryEventKind) -> Option<String>,
    ) -> Result<Redacted, Box<dyn std::error::Error>> {
        let next = self.next_id.lock().map_err(|_| "archive lock poisoned")?;
        if !self.path.exists() {
            return Ok(Redacted::default());
        }

        let mut report = Redacted::default();
        let mut out: Vec<u8> = Vec::new();
        let mut lines = 0_u64;
        for line in BufReader::new(File::open(&self.path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                // Written back rather than dropped: no event is numbered by it
                // (`count_events` skips blanks too), and a rewrite that is not
                // supposed to change the file should not quietly reflow it.
                out.push(b'\n');
                continue;
            }
            lines += 1;
            // An id comes from the line when it can be read, and otherwise from
            // where the line is: an event's id *is* its line number, so a line
            // too damaged to parse still has one, and a damaged line being
            // rewritten is exactly the one that must not keep its content.
            let id = leading_id(&line).or_else(|| line_id(line.as_bytes())).unwrap_or(lines);
            let parsed = serde_json::from_str::<MemoryEvent>(&line);
            let replacement = match &parsed {
                Ok(event) => content(id, &event.event),
                // Nothing to judge it by but its id. A caller replacing one
                // event by id still gets its answer; one asking about the kind
                // is told it is an operator event, the shape that carries the
                // least.
                Err(_) => content(id, &MemoryEventKind::Operator { content: String::new() }),
            };
            match replacement {
                None => out.extend_from_slice(line.as_bytes()),
                Some(text) => {
                    let rewritten = match parsed {
                        Ok(event) => MemoryEvent {
                            id: event.id,
                            timestamp: event.timestamp,
                            turn_id: event.turn_id,
                            event: match event.event {
                                MemoryEventKind::Operator { content } => {
                                    report.typed.push(content);
                                    MemoryEventKind::Operator { content: text }
                                }
                                MemoryEventKind::S0ma { intermediate, .. } => {
                                    MemoryEventKind::S0ma { content: text, intermediate }
                                }
                            },
                        },
                        Err(_) => {
                            // An unreadable line is replaced wholesale rather
                            // than left as it is: it is still readable to a
                            // person, and leaving content behind is the one
                            // outcome a redaction may not have.
                            report.unreadable += 1;
                            MemoryEvent {
                                id,
                                timestamp: Local::now().to_rfc3339(),
                                turn_id: None,
                                event: MemoryEventKind::S0ma { content: text, intermediate: false },
                            }
                        }
                    };
                    out.extend_from_slice(&serde_json::to_vec(&rewritten)?);
                    report.events += 1;
                    report.ids.push(id);
                }
            }
            out.push(b'\n');
        }

        // The id space is the line count, so a rewrite that changed it is a
        // bug in this function and must not reach the archive.
        if lines + 1 != *next {
            return Err(format!(
                "the rewrite would leave {lines} events where the archive has {}; nothing written",
                next.saturating_sub(1)
            )
            .into());
        }

        let temp = self.path.with_extension("redacting");
        {
            let mut file = File::create(&temp)?;
            file.write_all(&out)?;
            file.sync_all()?;
        }
        fs::rename(&temp, &self.path)?;
        if let Some(parent) = self.path.parent() {
            let _ = File::open(parent).and_then(|dir| dir.sync_all());
        }
        // Every cached byte offset past the first rewritten line is now wrong.
        // The mark's own checks would catch it -- the rename changed the file
        // this index was built from -- but there is no reason to leave a reader
        // one stale read away from it when dropping the index costs a lock.
        forget_line_index(&self.path);
        drop(next);
        Ok(report)
    }

    /// For each inclusive range of ids, the hash of its events and the tool
    /// results among them, in one pass over the archive. Lines outside every
    /// range are skipped on their id alone, without being parsed.
    pub fn spans(&self, ranges: &[(u64, u64)]) -> Result<Vec<SpanDigest>, Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let mut digests: Vec<(Sha256, Option<u64>, Vec<u64>)> = ranges.iter().map(|_| (Sha256::new(), None, Vec::new())).collect();
        if !ranges.is_empty() && self.path.exists() {
            let wanted = |id: u64| ranges.iter().any(|(from, through)| (*from..=*through).contains(&id));
            for line in complete_lines(File::open(&self.path)?) {
                let line = line?;
                if line.trim().is_empty() || leading_id(&line).is_some_and(|id| !wanted(id)) {
                    continue;
                }
                let event = Self::to_recalled_event(serde_json::from_str::<MemoryEvent>(&line)?);
                for ((from, through), (digest, previous, tools)) in ranges.iter().zip(digests.iter_mut()) {
                    if !(*from..=*through).contains(&event.id) {
                        continue;
                    }
                    if previous.is_some_and(|id| event.id <= id) {
                        return Err(format!("archive ids out of order at #{} in #{from}..#{through}", event.id).into());
                    }
                    *previous = Some(event.id);
                    hash_event(digest, &event)?;
                    if is_tool_result(&event) {
                        tools.push(event.id);
                    }
                }
            }
        }
        Ok(digests.into_iter().map(|(digest, _, tool_results)| SpanDigest { source_hash: format!("{:x}", digest.finalize()), tool_results }).collect())
    }

    /// Resolve a candidate set, in archive order, from the current file
    /// (freshly repaired records included): a seek per event through the line
    /// index, or one pass over the archive if the index can't be relied on.
    pub fn get_events(
        &self,
        ids: &[u64],
    ) -> Result<Vec<RecalledEvent>, Box<dyn std::error::Error>> {
        if ids.is_empty() || !self.path.exists() {
            return Ok(Vec::new());
        }
        if let Some(events) = self.indexed_events(ids)? {
            return Ok(events.into_iter().map(Self::to_recalled_event).collect());
        }
        self.scan_events(ids)
    }

    /// The events through the line index; None if it stayed stale after a
    /// rebuild (the caller scans instead).
    fn indexed_events(&self, ids: &[u64]) -> Result<Option<Vec<MemoryEvent>>, Box<dyn std::error::Error>> {
        for _ in 0..2 {
            let located = locate(&self.path, ids)?;
            if let Some(events) = read_located(&self.path, &located)? {
                return Ok(Some(events));
            }
            forget_line_index(&self.path);
        }
        Ok(None)
    }

    /// One pass over the archive: what lookups did before the line index.
    fn scan_events(&self, ids: &[u64]) -> Result<Vec<RecalledEvent>, Box<dyn std::error::Error>> {
        let wanted: std::collections::HashSet<u64> = ids.iter().copied().collect();
        let mut events = Vec::new();
        for line in complete_lines(File::open(&self.path)?) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let event: MemoryEvent = serde_json::from_str(&line)?;
            if wanted.contains(&event.id) {
                events.push(Self::to_recalled_event(event));
            }
        }
        Ok(events)
    }
    /// Drop a half-written final line from an append-only JSONL file.
    ///
    /// The archive is appended one line at a time, so a process killed
    /// mid-append — an SSH session dropping, a crash, a machine losing power —
    /// leaves a partial line at the end. Every reader here parses line by line
    /// with `?`, so that one fragment makes the whole archive unreadable and
    /// the agent will not start: the failure reads as
    /// `Error("EOF while parsing a string", line: 1, column: 25)`, naming
    /// neither the file nor the line.
    ///
    /// Repaired here rather than tolerated at each read, because there are seven
    /// such loops and because tolerating it would leave the fragment on disk to
    /// be re-skipped forever. Truncation is safe precisely because the file is
    /// append-only: a partial line is by definition the one that was being
    /// written when the process died, so the event it describes never completed.
    ///
    /// Only the tail is examined, and only a trailing fragment is removed. A
    /// parse failure anywhere earlier is real corruption and is left alone to be
    /// reported by the reader that meets it.
    fn repair_trailing_partial_line(path: &Path) -> std::io::Result<()> {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = fs::OpenOptions::new().read(true).write(true).open(path)?;
        let length = file.metadata()?.len();
        if length == 0 {
            return Ok(());
        }

        // Read back from the end, a megabyte at a time, until the window
        // holds the whole final line (a large tool result can be several).
        let mut window = length.min(1_048_576);
        let tail = loop {
            file.seek(SeekFrom::Start(length - window))?;
            let mut tail = Vec::with_capacity(window as usize);
            file.read_to_end(&mut tail)?;
            let found = tail[..tail.len().saturating_sub(1)].contains(&b'\n');
            if found || window == length || tail.last() == Some(&b'\n') {
                break tail;
            }
            window = length.min(window.saturating_mul(2));
        };

        // Where the final line starts, and whether the file ends on a boundary.
        let last_newline = tail.iter().rposition(|byte| *byte == b'\n');
        let final_line = match last_newline {
            Some(index) if index + 1 == tail.len() => return Ok(()), // ends cleanly
            Some(index) => &tail[index + 1..],
            // No newline anywhere: the window is the whole file, one
            // unterminated line.
            None => &tail[..],
        };

        if final_line.iter().all(|byte| byte.is_ascii_whitespace()) {
            return Ok(());
        }

        // A complete line would parse. If it does, the file simply lacks its
        // closing newline: add it, or readers (which leave out an unfinished
        // last line) would miss the event, and the next append would join it.
        if serde_json::from_slice::<serde_json::Value>(final_line).is_ok() {
            file.seek(SeekFrom::End(0))?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            return Ok(());
        }

        let keep = length - final_line.len() as u64;
        crate::terminal::warn(&format!(
            "[archive] {} ends with a {}-byte partial line, left by a process that was killed \
             mid-write. Dropping it; the event it would have recorded never completed. \
             Everything before it is intact.",
            path.display(),
            final_line.len()
        ));
        file.set_len(keep)?;
        file.sync_all()?;
        Ok(())
    }

    pub fn new(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();

        let checkpoint_path = path.with_file_name("checkpoints.jsonl");

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Before anything reads either file.
        for jsonl in [&path, &checkpoint_path] {
            if jsonl.exists() {
                Self::repair_trailing_partial_line(jsonl)?;
            }
        }

        let next_id = Self::count_events(&path)?;

        Ok(Self {
            path,
            checkpoint_path,
            next_id: Arc::new(Mutex::new(next_id)),
        })
    }

    /// The id the next appended event would take: one past the number of
    /// non-empty lines already in the archive.
    fn count_events(path: &Path) -> std::io::Result<u64> {
        if !path.exists() {
            return Ok(1);
        }
        let reader = BufReader::new(std::fs::File::open(path)?);
        let mut count = 0_u64;
        for line in reader.lines() {
            if !line?.trim().is_empty() {
                count += 1;
            }
        }
        Ok(count + 1)
    }

    /// Open the archive for reading only: no directory creation, and no
    /// trailing-partial-line repair.
    ///
    /// `new` repairs a partial final line, which means opening both JSONL
    /// files for writing and possibly truncating the tail. That is correct at
    /// start-up, where this process is the only the agent and a partial line can
    /// only be debris from a crash. It is wrong for a reader running beside a
    /// live agent -- `--print-system-prompt` is one, and an eval probe may well
    /// run while the operator's own session is up -- because the "partial"
    /// line it would truncate is more likely an append in progress than
    /// debris.
    ///
    /// Nothing is lost by skipping it: readers already leave an unfinished
    /// final line out, so the only event this view misses is the one still
    /// being written. A handle opened this way must not be used to append;
    /// nothing here does.
    pub fn open_read_only(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let checkpoint_path = path.with_file_name("checkpoints.jsonl");
        let next_id = Self::count_events(&path)?;
        Ok(Self {
            path,
            checkpoint_path,
            next_id: Arc::new(Mutex::new(next_id)),
        })
    }

    pub fn remember_operator(&mut self, content: String) -> Result<(), Box<dyn std::error::Error>> {
        self.append(|id| MemoryEvent::operator(id, content))?;
        Ok(())
    }

    pub fn remember_s0ma(&mut self, content: String) -> Result<(), Box<dyn std::error::Error>> {
        self.append(|id| MemoryEvent::s0ma(id, content))?;
        Ok(())
    }
    /// Intermediate records are included by recall_after(), hence by both sleep
    /// source builders, but are not replayed as ordinary conversation replies.
    pub fn remember_reasoning(
        &mut self,
        reasoning: &str,
        interrupted: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if reasoning.trim().is_empty() {
            return Ok(());
        }
        let state = if interrupted {
            "interrupted"
        } else {
            "completed"
        };
        self.remember_s0ma_intermediate(format!(
            "[s0ma_reasoning state={state}]\nTentative internal analysis; not an operator instruction or proof of external facts.\n{reasoning}"
        ))
    }

    pub fn remember_s0ma_intermediate(
        &mut self,
        content: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.append(|id| MemoryEvent::s0ma_intermediate(id, content))?;
        Ok(())
    }

    /// Persist exact captured evidence before returning a bounded model preview.
    pub fn remember_tool_result(
        &mut self,
        content: String,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        self.append(|id| MemoryEvent::s0ma_intermediate(id, content))
    }

    fn write_event(&self, event: &MemoryEvent) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        // One write, newline last: a reader on another thread sees the line
        // whole or not at all (`complete_lines` leaves an unfinished one out).
        // A write that fails partway (a full disk) is taken back, so the next
        // event doesn't join a fragment and readers never stop at one.
        let mut line = serde_json::to_vec(event)?;
        line.push(b'\n');
        let before = file.metadata()?.len();
        if let Err(error) = file.write_all(&line) {
            let _ = file.set_len(before);
            return Err(error.into());
        }

        Ok(())
    }

    fn to_recalled_event(event: MemoryEvent) -> RecalledEvent {
        let MemoryEvent {
            id,
            timestamp,
            event,
            ..
        } = event;

        match event {
            MemoryEventKind::Operator { content } => RecalledEvent {
                id,
                timestamp,
                speaker: Speaker::Operator,
                content,
            },

            MemoryEventKind::S0ma { content, .. } => RecalledEvent {
                id,
                timestamp,
                speaker: Speaker::S0ma,
                content,
            },
        }
    }

    /*
        pub fn recall_recent(
            &self,
            limit: usize,
        ) -> Result<Vec<RecalledEvent>, Box<dyn std::error::Error>> {
            if !self.path.exists() || limit == 0 {
                return Ok(Vec::new());
            }

            let file = std::fs::File::open(&self.path)?;
            let reader = BufReader::new(file);

            let mut recent = VecDeque::with_capacity(limit);

            for line in reader.lines() {
                let line = line?;

                if line.trim().is_empty() {
                    continue;
                }

                let event: MemoryEvent = serde_json::from_str(&line)?;

                if recent.len() == limit {
                    recent.pop_front();
                }

                recent.push_back(event);
            }

            let recalled = recent
                .into_iter()
                .map(|event| {
                    let MemoryEvent {
                        id,
                        timestamp,
                        event,
                    } = event;

                    match event {
                        MemoryEventKind::Operator { content } => RecalledEvent {
                            id,
                            timestamp,
                            speaker: Speaker::Operator,
                            content,
                        },

                        MemoryEventKind::S0ma { content } => RecalledEvent {
                            id,
                            timestamp,
                            speaker: Speaker::S0ma,
                            content,
                        },
                    }
                })
                .collect();

            Ok(recalled)
        }
    */
    pub fn recall_after(
        &self,
        after_id: u64,
    ) -> Result<Vec<RecalledEvent>, Box<dyn std::error::Error>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let mut file = std::fs::File::open(&self.path)?;
        // Ids ascend in the archive, so everything after `after_id` starts at
        // one offset: read from there instead of parsing the whole file.
        match offset_after(&self.path, after_id)? {
            Some(None) => return Ok(Vec::new()),
            Some(Some(offset)) => {
                use std::io::{Seek, SeekFrom};
                file.seek(SeekFrom::Start(offset))?;
            }
            None => {}
        }

        let mut recalled = Vec::new();

        for line in complete_lines(file) {
            let line = line?;

            if line.trim().is_empty() {
                continue;
            }

            let event: MemoryEvent = serde_json::from_str(&line)?;

            if event.id <= after_id {
                continue;
            }

            recalled.push(Self::to_recalled_event(event));
        }

        Ok(recalled)
    }

    #[cfg(test)]
    pub fn recent_conversation_turns(
        &self,
        limit: usize,
    ) -> Result<Vec<ConversationTurn>, Box<dyn std::error::Error>> {
        if limit == 0 || !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.path)?;

        let mut turns = VecDeque::new();

        let mut pending_operator: Option<(u64, String)> = None;

        for line in complete_lines(file) {
            let line = line?;

            if line.trim().is_empty() {
                continue;
            }

            let event: MemoryEvent = serde_json::from_str(&line)?;

            let event_id = event.id;

            match event.event {
                MemoryEventKind::Operator { content } => {
                    pending_operator = Some((event_id, content));
                }

                MemoryEventKind::S0ma {
                    content,
                    intermediate,
                } => {
                    if intermediate {
                        continue;
                    }

                    if let Some((operator_event_id, operator)) = pending_operator.take() {
                        turns.push_back(ConversationTurn {
                            operator_event_id,
                            s0ma_event_id: event_id,
                            operator,
                            s0ma: content,
                        });

                        while turns.len() > limit {
                            turns.pop_front();
                        }
                    }
                }
            }
        }

        Ok(turns.into_iter().collect())
    }

    pub fn latest_checkpoint(
        &self,
    ) -> Result<Option<MemoryCheckpoint>, Box<dyn std::error::Error>> {
        if !self.checkpoint_path.exists() {
            return Ok(None);
        }

        let file = std::fs::File::open(&self.checkpoint_path)?;
        let reader = BufReader::new(file);

        let mut latest = None;

        for line in reader.lines() {
            let line = line?;

            if line.trim().is_empty() {
                continue;
            }

            latest = Some(serde_json::from_str::<MemoryCheckpoint>(&line)?);
        }

        Ok(latest)
    }

    pub fn append_checkpoint(
        &self,
        checkpoint: &MemoryCheckpoint,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // A redaction from the memory tab rewrites the file too.
        let _file = CHECKPOINTS.lock().map_err(|_| "checkpoints lock poisoned")?;
        // The main thread is the sole writer. Replace atomically to avoid torn JSONL appends.
        let mut bytes=if self.checkpoint_path.exists(){fs::read(&self.checkpoint_path)?}else{Vec::new()};
        if !bytes.is_empty() && !bytes.ends_with(b"\n"){bytes.push(b'\n');}
        serde_json::to_writer(&mut bytes,checkpoint)?;bytes.push(b'\n');
        let temp=self.checkpoint_path.with_extension("tmp");
        {let mut file=File::create(&temp)?;file.write_all(&bytes)?;file.sync_all()?;}
        fs::rename(temp,&self.checkpoint_path)?;
        Ok(())
    }

    pub fn get_event(&self, id: u64) -> Result<Option<RecalledEvent>, Box<dyn std::error::Error>> {
        if !self.path.exists() {
            return Ok(None);
        }
        if let Some(mut events) = self.indexed_events(&[id])? {
            return Ok(events.pop().map(Self::to_recalled_event));
        }

        let file = std::fs::File::open(&self.path)?;

        for line in complete_lines(file) {
            let line = line?;

            if line.trim().is_empty() {
                continue;
            }

            let event: MemoryEvent = serde_json::from_str(&line)?;

            if event.id == id {
                return Ok(Some(Self::to_recalled_event(event)));
            }
        }

        Ok(None)
    }

    /// Literal archive search. Every term must appear in an event's content
    /// (case-insensitive, any order); a double-quoted phrase is one term and must
    /// appear verbatim. Returns the newest `limit` matches, plus how many matched
    /// in total and how many unreadable archive lines were skipped, so callers can
    /// say what the window left out instead of silently truncating.
    ///
    /// Requiring every term is a strict superset of the old single-substring
    /// test: any event containing the whole query verbatim contains every term.
    pub fn search_events(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<LiteralSearch, Box<dyn std::error::Error>> {
        let mut result = LiteralSearch::default();
        let needles: Vec<String> = literal_terms(query).iter().map(|t| t.to_lowercase()).collect();

        if !self.path.exists() || needles.is_empty() || limit == 0 {
            return Ok(result);
        }

        let file = std::fs::File::open(&self.path)?;

        let mut matches = VecDeque::with_capacity(limit);

        for line in complete_lines(file) {
            let line = line?;

            if line.trim().is_empty() {
                continue;
            }

            // One corrupt line used to abort the whole search with an error.
            // Skip it and report the count instead.
            let Ok(event) = serde_json::from_str::<MemoryEvent>(&line) else {
                result.skipped_lines += 1;
                continue;
            };

            let content = match &event.event {
                MemoryEventKind::Operator { content } => content,
                MemoryEventKind::S0ma { content, .. } => content,
            };

            let haystack = content.to_lowercase();
            if needles.iter().all(|needle| haystack.contains(needle.as_str())) {
                result.total_matches += 1;
                if matches.len() == limit {
                    matches.pop_front();
                }

                matches.push_back(Self::to_recalled_event(event));
            }
        }

        result.events = matches.into_iter().collect();
        Ok(result)
    }
}

/// Result of a literal archive search.
#[derive(Debug, Default)]
pub struct LiteralSearch {
    /// The newest matches, oldest first, at most the requested limit.
    pub events: Vec<RecalledEvent>,
    /// Every event that matched, before the newest-`limit` window was applied.
    pub total_matches: usize,
    /// Archive lines that could not be parsed and were skipped.
    pub skipped_lines: usize,
}

/// Split a literal query into terms: double-quoted phrases stay whole, and
/// everything else splits on whitespace. Original case is kept for previews;
/// matching lowercases both sides.
pub fn literal_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for (index, part) in query.split('"').enumerate() {
        if index % 2 == 1 {
            // Inside quotes: the whole phrase is one term.
            let phrase = part.trim();
            if !phrase.is_empty() {
                terms.push(phrase.to_string());
            }
        } else {
            terms.extend(part.split_whitespace().map(str::to_string));
        }
    }
    terms
}

/// The archive's complete lines, as `lines()` gives them. Another thread may be
/// appending an event at this moment (a subagent's cited evidence), and its
/// line is not complete until its newline lands: the part already written is
/// left out rather than failing the read. (At startup, `new` repairs a partial
/// line a crash left.)
/// One stretch of the archive as folding reads it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpanDigest {
    /// The stretch's source hash, in the form continuous compaction uses.
    pub source_hash: String,
    /// The tool results in it, oldest first: what `[observed #N]` may name.
    pub tool_results: Vec<u64>,
}

/// Add one event to a source hash. Continuous compaction and gists hash their
/// archive spans with this, so the two can be compared.
/// The digest of one event's content, for deciding later whether it still says
/// what it said. The same function as the derived indexes use to tell a changed
/// event from an unchanged one, so a redaction decided against a digest is
/// decided against exactly what they would read.
pub fn content_digest(content: &str) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(content.as_bytes()))
}

pub fn hash_event(digest: &mut sha2::Sha256, event: &RecalledEvent) -> Result<(), Box<dyn std::error::Error>> {
    use sha2::Digest;
    digest.update(serde_json::to_vec(&(event.id, &event.timestamp, matches!(event.speaker, Speaker::Operator), &event.content))?);
    Ok(())
}

/// Whether an archived event is a tool result: the record agent.rs writes for
/// a call, `[tool:<name>]` and what it returned. That is what the trailer of a
/// result in context names, so it is the only kind of event `[observed #N]`
/// can stand for. (Harness records archived beside them, such as a knowledge
/// check, are not observations.)
pub fn is_tool_result(event: &RecalledEvent) -> bool {
    matches!(event.speaker, Speaker::S0ma) && event.content.starts_with("[tool:")
}

/// The id at the head of an archive line as `write_event` writes it
/// (`{"id":N,…`), without parsing the rest. None for any other shape.
fn leading_id(line: &str) -> Option<u64> {
    let rest = line.strip_prefix("{\"id\":")?;
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    rest[..digits].parse().ok()
}

// ------------------------------------------------------------------
// Line index: where each event's line starts
// ------------------------------------------------------------------

/// Write a SQLite store out again without the pages its deleted rows freed,
/// and truncate the write-ahead log beside it.
///
/// Deleting a row does not take its bytes out of the file: they stay in pages
/// marked free, and a store that held redacted text would keep holding it.
/// `VACUUM` rebuilds the file without them. The WAL needs saying separately --
/// these stores run in WAL mode, and a vacuum does not truncate the sidecar,
/// so frames holding the old text can remain beside a clean database.
///
/// Neither is allowed to fail the caller. By the time this runs the rows are
/// gone, which is the part that matters; this is about the bytes, and a busy
/// database is a reason to try again later, not to report that nothing was
/// removed. Both are no-ops inside a transaction, so callers run it after
/// committing.
pub(crate) fn compact(db: &rusqlite::Connection) {
    if let Err(error) = db.execute_batch("VACUUM") {
        crate::terminal::warn(&format!(
            "[archive] a store was not rewritten after its rows went, so their bytes stay in free pages \
             until it is: {error}"
        ));
    }
    let _ = db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)");
}

/// Where each event's line starts in an archive file, kept per path for the
/// life of the process, so reading an event by id is one seek and one line
/// instead of a parse of the whole file.
///
/// The archive only grows (cutting a torn final line is the one exception), so
/// the index only ever reads what was appended since it last looked, and it
/// takes the id from the head of each line without parsing the rest. It is
/// never trusted blindly: a replaced or shortened file, or one whose indexed
/// part no longer ends on a line boundary, starts it over, and every read
/// checks that the line at the offset carries the id asked for. A stale read
/// rebuilds the index once; if that still fails, the caller does the full
/// scan this replaced.
#[derive(Default)]
struct LineIndex {
    /// How far the index has read, and what tells a grown file from a
    /// rewritten one.
    mark: ReadMark,
    offsets: HashMap<u64, u64>,
    /// Ids in file order. In an archive `append` wrote they ascend; once one
    /// doesn't, `ascending` is false and range reads fall back to a scan.
    order: Vec<u64>,
    ascending: bool,
}

/// How far an incremental reader of the archive has got, and enough to tell
/// later whether the file has only grown since. The line index, the keyword
/// index and the semantic index each keep one, so each reads only what was
/// appended since it last looked.
///
/// The archive only grows: the one exception, cutting a failed write or a
/// torn final line, removes only an unfinished line, which no reader counts as
/// read. Anything else is a hand edit, and these are cheap ways to catch one:
/// the same file (device and inode), still ending a line where the mark does,
/// and the last event line read still where it was with the id it had. None
/// of them reads more than a few bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ReadMark {
    /// Bytes read so far; always just past a newline (0: nothing read yet).
    pub through: u64,
    /// Device and inode of the file read.
    pub identity: Option<(u64, u64)>,
    /// Where the last event line read starts, and its id.
    pub last: Option<(u64, u64)>,
}

impl ReadMark {
    /// A mark at the start of the file `meta` describes.
    pub(crate) fn start(meta: &fs::Metadata) -> Self {
        Self { identity: file_identity(meta), ..Self::default() }
    }

    /// One more complete line read: `len` bytes with its newline, carrying
    /// `id` when it is an event line.
    pub(crate) fn advance(&mut self, len: usize, id: Option<u64>) {
        if let Some(id) = id {
            self.last = Some((self.through, id));
        }
        self.through += len as u64;
    }

    /// Whether the file at `path` (described by `meta`) still holds what this
    /// mark read, as far as the cheap checks can tell, so that reading on from
    /// `through` sees exactly what was appended since.
    pub(crate) fn still_holds(&self, path: &Path, meta: &fs::Metadata) -> std::io::Result<bool> {
        if file_identity(meta) != self.identity || meta.len() < self.through || !ends_line_at(path, self.through)? {
            return Ok(false);
        }
        match self.last {
            Some((offset, id)) => line_has_id(path, offset, id),
            None => Ok(true),
        }
    }
}

/// Whether the line at `offset` is event `id`'s. A line `write_event` wrote is
/// judged from its first bytes; any other shape is read whole and parsed.
pub(crate) fn line_has_id(path: &Path, offset: u64, id: u64) -> std::io::Result<bool> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut head = [0u8; 32];
    let mut filled = 0;
    while filled < head.len() {
        match file.read(&mut head[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    if let Some(found) = leading_id(utf8_prefix(&head[..filled])) {
        return Ok(found == id);
    }
    Ok(match line_at(&mut file, offset)? {
        Some(line) => line_id(&line) == Some(id),
        None => false,
    })
}

/// The longest valid UTF-8 prefix: a fixed-size head can end inside a
/// character.
fn utf8_prefix(bytes: &[u8]) -> &str {
    match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => std::str::from_utf8(&bytes[..error.valid_up_to()]).unwrap_or_default(),
    }
}

/// Whether `offset` is a line boundary of the file at `path`: the start, or
/// just past a newline. An index that synced up to `offset` can carry on from
/// there only if it still is.
pub(crate) fn ends_line_at(path: &Path, offset: u64) -> std::io::Result<bool> {
    use std::io::{Read, Seek, SeekFrom};
    if offset == 0 {
        return Ok(true);
    }
    let mut file = File::open(path)?;
    if file.metadata()?.len() < offset {
        return Ok(false);
    }
    file.seek(SeekFrom::Start(offset - 1))?;
    let mut byte = [0u8];
    Ok(file.read(&mut byte)? == 1 && byte[0] == b'\n')
}

/// Device and inode, so an index can tell the same file grown from a file
/// replaced under it.
pub(crate) fn file_identity(meta: &fs::Metadata) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some((meta.dev(), meta.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = meta;
        None
    }
}

/// The id of an archive line: from its head when `write_event` wrote it,
/// otherwise from a parse of the line. None for a blank or unreadable line.
fn line_id(line: &[u8]) -> Option<u64> {
    if let Some(id) = leading_id(utf8_prefix(&line[..line.len().min(32)])) {
        return Some(id);
    }
    #[derive(Deserialize)]
    struct IdOnly {
        id: u64,
    }
    serde_json::from_slice::<IdOnly>(line).ok().map(|line| line.id)
}

impl LineIndex {
    fn fresh(meta: Option<&fs::Metadata>) -> Self {
        Self { mark: meta.map(ReadMark::start).unwrap_or_default(), ascending: true, ..Self::default() }
    }

    /// Index whatever was appended since the last call; start over when the
    /// file no longer holds what was indexed (see `ReadMark::still_holds`).
    fn refresh(&mut self, path: &Path) -> std::io::Result<()> {
        let meta = match fs::metadata(path) {
            Ok(meta) => meta,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                *self = Self::fresh(None);
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        if self.mark.through == 0 || !self.mark.still_holds(path, &meta)? {
            *self = Self::fresh(Some(&meta));
        }
        if meta.len() == self.mark.through {
            return Ok(());
        }
        if !self.extend(path)? {
            // An id seen twice while extending: the indexed part was rewritten.
            *self = Self::fresh(Some(&meta));
            self.extend(path)?;
        }
        Ok(())
    }

    /// Index complete lines from the mark to the end. False when an id that
    /// is already indexed turns up while extending (the file was rewritten);
    /// in a build from the start a repeated id keeps its first line, which is
    /// the one a scan finds first.
    fn extend(&mut self, path: &Path) -> std::io::Result<bool> {
        use std::io::{Seek, SeekFrom};
        let extending = self.mark.through > 0;
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(self.mark.through))?;
        let mut reader = BufReader::with_capacity(1 << 20, file);
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = reader.read_until(b'\n', &mut line)?;
            // An unfinished final line is being written: index it next time.
            if read == 0 || line.last() != Some(&b'\n') {
                return Ok(true);
            }
            let start = self.mark.through;
            let id = line_id(&line);
            self.mark.advance(read, id);
            let Some(id) = id else { continue };
            if self.offsets.contains_key(&id) {
                if extending {
                    return Ok(false);
                }
                continue;
            }
            self.offsets.insert(id, start);
            if self.order.last().is_some_and(|last| *last >= id) {
                self.ascending = false;
            }
            self.order.push(id);
        }
    }
}

fn line_indexes() -> &'static Mutex<HashMap<PathBuf, LineIndex>> {
    static INDEXES: std::sync::OnceLock<Mutex<HashMap<PathBuf, LineIndex>>> = std::sync::OnceLock::new();
    INDEXES.get_or_init(Default::default)
}

fn with_line_index<T>(path: &Path, f: impl FnOnce(&LineIndex) -> T) -> std::io::Result<T> {
    let mut all = line_indexes()
        .lock()
        .map_err(|_| std::io::Error::other("archive line index lock poisoned"))?;
    let index = all.entry(path.to_owned()).or_insert_with(|| LineIndex::fresh(None));
    index.refresh(path)?;
    Ok(f(index))
}

fn forget_line_index(path: &Path) {
    if let Ok(mut all) = line_indexes().lock() {
        all.remove(path);
    }
}

/// Where the lines of `ids` start, in file order; ids the archive doesn't
/// hold are left out.
fn locate(path: &Path, ids: &[u64]) -> std::io::Result<Vec<(u64, u64)>> {
    with_line_index(path, |index| {
        let mut found: Vec<(u64, u64)> = ids
            .iter()
            .filter_map(|id| index.offsets.get(id).map(|offset| (*id, *offset)))
            .collect();
        found.sort_by_key(|(_, offset)| *offset);
        found.dedup();
        found
    })
}

/// The offset of the first line whose id is above `after`, when ids ascend:
/// `Some(None)` if there is none, `None` if the order can't be relied on.
fn offset_after(path: &Path, after: u64) -> std::io::Result<Option<Option<u64>>> {
    with_line_index(path, |index| {
        if !index.ascending {
            return None;
        }
        let at = index.order.partition_point(|id| *id <= after);
        Some(index.order.get(at).map(|id| index.offsets[id]))
    })
}

/// The archive's byte offset of event `id`'s line, for an index that reads the
/// archive itself. None when the archive has no such event.
pub(crate) fn event_offset(path: &Path, id: u64) -> std::io::Result<Option<u64>> {
    Ok(locate(path, &[id])?.first().map(|(_, offset)| *offset))
}

/// Every event id in the archive, from the line index: what an index that
/// mirrors the archive may keep when it prunes.
pub(crate) fn event_ids(path: &Path) -> std::io::Result<std::collections::HashSet<u64>> {
    with_line_index(path, |index| index.order.iter().copied().collect())
}

/// Build the line index in the background, so the first lookup of a session
/// doesn't read the whole archive on the turn path.
pub fn warm_line_index(path: &Path) {
    let path = path.to_owned();
    let _ = std::thread::Builder::new().name("archive-lines".into()).spawn(move || {
        if let Err(error) = with_line_index(&path, |_| ()) {
            crate::terminal::warn_status(&format!("[archive] line index warm-up failed: {error}"));
        }
    });
}

/// One complete line at `offset`, without its newline; None if the line there
/// is unfinished.
pub(crate) fn line_at(file: &mut File, offset: u64) -> std::io::Result<Option<Vec<u8>>> {
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    let mut line = Vec::new();
    BufReader::new(&mut *file).read_until(b'\n', &mut line)?;
    if line.last() != Some(&b'\n') {
        return Ok(None);
    }
    line.pop();
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    Ok(Some(line))
}

/// The events at these offsets, or None if any line there is not the event
/// the index said (it is stale).
fn read_located(path: &Path, located: &[(u64, u64)]) -> Result<Option<Vec<MemoryEvent>>, Box<dyn std::error::Error>> {
    if located.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let mut file = File::open(path)?;
    let mut events = Vec::with_capacity(located.len());
    for (id, offset) in located {
        let Some(line) = line_at(&mut file, *offset)? else { return Ok(None) };
        match serde_json::from_slice::<MemoryEvent>(&line) {
            Ok(event) if event.id == *id => events.push(event),
            _ => return Ok(None),
        }
    }
    Ok(Some(events))
}

fn complete_lines(file: File) -> impl Iterator<Item = std::io::Result<String>> {
    let mut reader = BufReader::new(file);
    std::iter::from_fn(move || {
        let mut line = Vec::new();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => None,
            Ok(_) if line.last() != Some(&b'\n') => None,
            Ok(_) => {
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                Some(String::from_utf8(line).map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error)))
            }
            Err(error) => Some(Err(error)),
        }
    })
}

/// Bounded head/tail preview; the immutable archive remains authoritative.
/// Explicit archive reads should be paginated, never passed through this helper.
pub fn tool_preview(content: &str, id: u64, limit: usize) -> String {
    let count = content.chars().count();
    if count <= limit {
        return content.to_owned();
    }
    let head: String = content.chars().take(limit * 3 / 4).collect();
    let tail: String = content.chars().skip(count - limit / 4).collect();
    let metadata = serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|v| v.as_object().cloned())
        .map(|mut fields| {
            fields.retain(|key, value| {
                !matches!(key.as_str(), "stdout" | "stderr" | "output")
                    && value.to_string().len() <= 512
            });
            fields
        });
    serde_json::json!({"preview_only":true,"archive_event_id":id,"metadata":metadata,
        "total_chars":count,"head":head,"tail":tail,
        "instruction":"Middle omitted, not summarized. Do not infer absence or completeness. Retrieve decisive evidence with archive_get(id, offset, limit); offsets are character positions in the archived record."}).to_string()
}

#[cfg(test)]
mod preview_tests {
    #[test]
    fn preview_preserves_small_results_and_marks_unicode_omissions() {
        assert_eq!(super::tool_preview("ok", 4, 8), "ok");
        let result = super::tool_preview(&"界".repeat(100), 42, 8);
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["archive_event_id"], 42);
        assert_eq!(v["head"].as_str().unwrap().chars().count(), 6);
        assert_eq!(v["tail"].as_str().unwrap().chars().count(), 2);
        assert_eq!(v["total_chars"], 100);
    }
}

#[cfg(test)]
mod shared_handle_tests {
    use super::*;

    fn line_index_archive(tag: &str) -> (PathBuf, Hippocampus) {
        let dir = std::env::temp_dir().join(format!("s0ma-line-index-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let memory = Hippocampus::new(dir.join("hippocampus.jsonl")).unwrap();
        (dir, memory)
    }

    fn contents(events: Vec<RecalledEvent>) -> Vec<(u64, String)> {
        events.into_iter().map(|event| (event.id, event.content)).collect()
    }

    #[test]
    fn a_rewrite_that_keeps_the_old_end_on_a_line_boundary_is_still_noticed() {
        let (dir, memory) = line_index_archive("boundary");
        let line = |id: u64, content: &str| {
            format!("{{\"id\":{id},\"timestamp\":\"t\",\"kind\":\"operator\",\"content\":\"{content}\"}}\n")
        };
        fs::write(memory.archive_path(), [line(1, "aa"), line(2, "bb"), line(3, "cc")].concat()).unwrap();
        assert_eq!(memory.get_event(3).unwrap().unwrap().content, "cc");
        // Same inode, every line the same length: the old end is still a line
        // boundary, and the only line past it (4) is an id the index never saw.
        // Event 5 now sits where 3 was.
        fs::write(memory.archive_path(), [line(1, "aa"), line(3, "cc"), line(5, "ee"), line(4, "dd")].concat()).unwrap();
        assert_eq!(memory.get_event(5).unwrap().unwrap().content, "ee");
        assert_eq!(memory.get_event(3).unwrap().unwrap().content, "cc");
        assert!(memory.get_event(2).unwrap().is_none());
        assert_eq!(contents(memory.get_events(&[4, 1]).unwrap()), vec![(1, "aa".into()), (4, "dd".into())]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_read_while_another_handle_appends_never_sees_half_a_line() {
        let dir = std::env::temp_dir().join(format!("s0ma-torn-read-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut memory = Hippocampus::new(dir.join("hippocampus.jsonl")).unwrap();
        memory.remember_operator("hello".into()).unwrap();
        let mut writer = memory.clone();
        let appending = std::thread::spawn(move || {
            for n in 0..300 {
                // Big enough to take several pages per line.
                writer.remember_tool_result(format!("{n} {}", "x".repeat(20_000))).unwrap();
            }
        });
        let mut reads = 0;
        while !appending.is_finished() {
            memory.recall_after(0).unwrap();
            memory.get_events(&[1, 2, 3]).unwrap();
            reads += 1;
        }
        appending.join().unwrap();
        assert!(reads > 0);
        assert_eq!(memory.recall_after(0).unwrap().len(), 301);
        let _ = fs::remove_dir_all(&dir);
    }
}

