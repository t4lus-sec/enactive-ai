//! Weakness log: an auditable record of failures observed during normal operation.
//!
//! s0ma proposes, the operator decides. s0ma can report a weakness and add further
//! instances of one, each tied to archive event IDs and/or consolidated episode IDs
//! that must exist at the moment of the report. Only the operator can confirm,
//! reject, resolve or merge (the /weakness commands), so the log never certifies
//! itself: the checker is not the generator.
//!
//! A weakness has one name, its `wk-NNNN` ID. Nothing else is shown as one: no
//! class labels (the operator's command no longer takes one, and classes logged
//! before 2026-10-26 reach only the exports) and no log sequence numbers.
//!
//! Nothing from the log is in s0ma's context. Confirmed titles used to be listed
//! there as a `[KNOWN WEAKNESSES]` block; since 2026-10-26 a confirmed weakness
//! reaches s0ma only through training (Q10) and through `weakness_list` when it
//! looks. The call store still records an empty `weaknesses` span in every
//! system message (see `runtime_context::append_catalogs`).
//!
//! Storage, next to the hippocampus archive:
//!   weaknesses.jsonl  append-only and hash-chained; each line commits to the
//!                     previous line's hash, and nothing is ever rewritten. This is
//!                     the record.
//!   weaknesses.json   current state folded from the log, regenerated after every
//!                     append, for reading by hand. Derived; safe to delete.
//!   weaknesses-export.jsonl
//!                     the same fold as one object per line, header first, for
//!                     Q10: `qref weakness-render` reads it beside the archive
//!                     and turns confirmed and resolved weaknesses into the
//!                     lessons the nightly slice trains on. Written whenever the
//!                     fold changes and refreshed by the sleep pass, since
//!                     `q10.sh` warns on an export older than 48 h. Derived;
//!                     safe to delete.
//!
//! Every entry is stamped by the pipeline, not asserted by the model: sequence
//! number, time, actor, the served model name (so weaknesses of different base
//! models or adapters stay separable), the archive position at report time, and
//! for each cited event a content hash plus excerpt, so a later audit can tell
//! whether the evidence still says what it said when it was cited.

use crate::hippocampus::{Hippocampus, Speaker};
use crate::llm::ToolDefinition;
use crate::types::ToolCall;

use chrono::{Local, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const EXPORT_SCHEMA: &str = "s0ma.weaknesses.v1";
const MAX_IDS_PER_KIND: usize = 20;
const MIN_JUSTIFICATION_CHARS: usize = 40;
const MIN_FIELD_CHARS: usize = 10;
const MAX_FIELD_CHARS: usize = 4000;
const EXCERPT_CHARS: usize = 500;
const GENESIS: &str = "genesis";

// ------------------------------------------------------------------ evidence

/// One archive event as the log sees it.
pub struct EventSnapshot {
    pub id: u64,
    pub timestamp: String,
    pub speaker: &'static str,
    pub content: String,
}

/// Where evidence is resolved. The live implementation is the hippocampus;
/// tests use a fake.
pub trait EvidenceSource {
    /// The events among `ids` that exist. Missing IDs are simply absent.
    fn events(&self, ids: &[u64]) -> Result<Vec<EventSnapshot>>;
    /// None if the episode doesn't exist; Some(unchanged) if it does, where
    /// `unchanged` is whether its own source evidence is still intact.
    fn episode(&self, id: &str) -> Result<Option<bool>>;
    /// Newest archive event ID at this moment.
    fn latest_event_id(&self) -> u64;
}

impl EvidenceSource for Hippocampus {
    fn events(&self, ids: &[u64]) -> Result<Vec<EventSnapshot>> {
        Ok(self
            .get_events(ids)?
            .into_iter()
            .map(|event| EventSnapshot {
                id: event.id,
                timestamp: event.timestamp,
                speaker: match event.speaker {
                    Speaker::Operator => "operator",
                    Speaker::S0ma => "s0ma",
                },
                content: event.content,
            })
            .collect())
    }

    fn episode(&self, id: &str) -> Result<Option<bool>> {
        crate::archive::recall::check_episode(self, id)
    }

    fn latest_event_id(&self) -> u64 {
        self.last_event_id()
    }
}

fn sha256_hex(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn event_digest(event: &EventSnapshot) -> Result<String> {
    Ok(sha256_hex(&serde_json::to_string(&(
        &event.timestamp,
        event.speaker,
        &event.content,
    ))?))
}

fn excerpt(text: &str, chars: usize) -> String {
    let mut out: String = text.chars().take(chars).collect();
    if text.chars().count() > chars {
        out.push('…');
    }
    out
}

/// A snapshotted excerpt as the Q10 export must give it: the raw leading
/// characters of the event's content, with nothing else and nothing added.
///
/// qref's integrity check is that the archived event's content *still begins
/// with* this string — the cheap "is this evidence still what it was" test that
/// catches the archive moving under the log. The ellipsis `excerpt()` appends
/// when it cut the content short is a reading aid, not part of the content, so
/// left in it would fail that check on every event longer than
/// `EXCERPT_CHARS` — which is most of them.
///
/// The marker is identified by length, not by guessing: `excerpt()` only
/// appends it past `EXCERPT_CHARS` characters, so an excerpt longer than that
/// ends in exactly one marker character. Nothing else is touched: no trimming,
/// no whitespace normalisation, no redaction. (If something must be redacted,
/// drop the instance rather than altering the excerpt.)
fn raw_excerpt(excerpt: &str) -> &str {
    if excerpt.chars().count() > EXCERPT_CHARS {
        excerpt.strip_suffix('…').unwrap_or(excerpt)
    } else {
        excerpt
    }
}

/// The export's name for one instance: the weakness ID and the instance's
/// position in the fold, 1-based. Stable for the life of the log.
fn instance_id(weakness_id: &str, index: usize) -> String {
    format!("{weakness_id}/{}", index + 1)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Evidence {
    Event {
        id: u64,
        speaker: String,
        timestamp: String,
        content_sha256: String,
        excerpt: String,
    },
    Episode {
        id: String,
        source_evidence_unchanged: bool,
    },
}

/// Resolve and snapshot cited evidence. The outer Result is an I/O failure; the
/// inner Err is a rejection to show the model (missing IDs, none given, ...).
fn collect_evidence(
    source: &dyn EvidenceSource,
    event_ids: &[u64],
    episode_ids: &[String],
) -> Result<std::result::Result<Vec<Evidence>, String>> {
    if event_ids.is_empty() && episode_ids.is_empty() {
        return Ok(Err(
            "cite at least one archive event ID or episode ID where the weakness shows; find them with archive_search or archive_get".into(),
        ));
    }
    if event_ids.len() > MAX_IDS_PER_KIND || episode_ids.len() > MAX_IDS_PER_KIND {
        return Ok(Err(format!(
            "cite at most {MAX_IDS_PER_KIND} event IDs and {MAX_IDS_PER_KIND} episode IDs; pick the clearest examples"
        )));
    }
    // The log keeps an excerpt of every cited event and is hash-chained, so a
    // citation cannot be edited out later. An event inside a job is going to be
    // redacted, which would leave the only surviving copy of it here.
    let latest = source.latest_event_id();
    let held: Vec<String> = event_ids
        .iter()
        .filter_map(|id| crate::jobs::covering_event(latest, *id).map(|job| format!("{id} (job {job})")))
        .collect();
    if !held.is_empty() {
        return Ok(Err(format!(
            "these events are inside a job whose record will be removed when it closes, and this log keeps an \
             excerpt of everything it cites, so it cannot cite them: {}. Report the weakness with an event from \
             outside the job, or describe the pattern in `behavior` and `expected` without citing the work \
             itself.",
            held.join(", ")
        )));
    }

    let unique_events: BTreeSet<u64> = event_ids.iter().copied().collect();
    let unique_episodes: BTreeSet<&str> = episode_ids.iter().map(|s| s.trim()).collect();
    if unique_events.len() != event_ids.len() || unique_episodes.len() != episode_ids.len() {
        return Ok(Err("each ID may be cited only once".into()));
    }

    let found: BTreeMap<u64, EventSnapshot> = source
        .events(&unique_events.iter().copied().collect::<Vec<_>>())?
        .into_iter()
        .map(|event| (event.id, event))
        .collect();
    let missing_events: Vec<u64> = unique_events
        .iter()
        .filter(|id| !found.contains_key(id))
        .copied()
        .collect();

    let mut episodes = Vec::new();
    let mut missing_episodes = Vec::new();
    for id in &unique_episodes {
        match source.episode(id)? {
            Some(unchanged) => episodes.push(Evidence::Episode {
                id: (*id).to_string(),
                source_evidence_unchanged: unchanged,
            }),
            None => missing_episodes.push((*id).to_string()),
        }
    }

    if !missing_events.is_empty() || !missing_episodes.is_empty() {
        let mut parts = Vec::new();
        if !missing_events.is_empty() {
            parts.push(format!("archive events {missing_events:?}"));
        }
        if !missing_episodes.is_empty() {
            parts.push(format!("episodes {missing_episodes:?}"));
        }
        return Ok(Err(format!(
            "these cited IDs do not exist: {}. Nothing was recorded; check the IDs with archive_get or episode_get",
            parts.join(" and ")
        )));
    }

    let mut evidence = Vec::new();
    for id in &unique_events {
        let event = &found[id];
        evidence.push(Evidence::Event {
            id: event.id,
            speaker: event.speaker.to_string(),
            timestamp: event.timestamp.clone(),
            content_sha256: event_digest(event)?,
            excerpt: excerpt(&event.content, EXCERPT_CHARS),
        });
    }
    evidence.extend(episodes);
    Ok(Ok(evidence))
}

// ------------------------------------------------------------------ log format

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Confirm,
    /// Trained out: Q10 keeps training the weakness and gates every night on
    /// its fix still holding. `Reopen` is the undo.
    Resolve,
    Reject,
    Merge,
    Reopen,
    /// No longer offered (2026-10-26: weaknesses have no class now). Kept so
    /// that logs holding it still fold and verify.
    Classify,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub justification: String,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Action {
    Report {
        weakness_id: String,
        title: String,
        behavior: String,
        expected: String,
        suggested_class: Option<String>,
        severity: Severity,
        instance: Instance,
    },
    AddInstance {
        weakness_id: String,
        instance: Instance,
    },
    Review {
        weakness_id: String,
        decision: Decision,
        class: Option<String>,
        merged_into: Option<String>,
        note: String,
    },
}

impl Action {
    fn weakness_id(&self) -> &str {
        match self {
            Action::Report { weakness_id, .. }
            | Action::AddInstance { weakness_id, .. }
            | Action::Review { weakness_id, .. } => weakness_id,
        }
    }
}

/// One line of weaknesses.jsonl. `hash` is sha256 over this entry serialized with
/// `hash` empty; since the entry includes `prev_hash`, each line commits to the
/// whole history before it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub seq: u64,
    pub timestamp: String,
    pub actor: String,
    pub model: String,
    pub after_event_id: u64,
    pub action: Action,
    pub prev_hash: String,
    pub hash: String,
}

impl Entry {
    fn compute_hash(&self) -> Result<String> {
        let mut unsigned = self.clone();
        unsigned.hash = String::new();
        Ok(sha256_hex(&serde_json::to_string(&unsigned)?))
    }
}

// ------------------------------------------------------------------ state

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Proposed,
    Confirmed,
    /// Confirmed, then fixed. Still in the log, still in the export: the Q10
    /// slice is rebuilt from the base weights every night, so a resolved
    /// weakness has to keep being trained or its fix is undone, and its
    /// held-out margin is gated as a regression.
    Resolved,
    Rejected,
    Merged,
}

/// The status as the log and both exports spell it. One function so the
/// operator's list, the counts and the export can never drift apart.
fn status_word(status: Status) -> &'static str {
    match status {
        Status::Proposed => "proposed",
        Status::Confirmed => "confirmed",
        Status::Resolved => "resolved",
        Status::Rejected => "rejected",
        Status::Merged => "merged",
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct InstanceRecord {
    pub seq: u64,
    pub added_at: String,
    pub actor: String,
    pub model: String,
    pub after_event_id: u64,
    pub justification: String,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReviewRecord {
    pub seq: u64,
    pub at: String,
    /// Who decided, as the log stamped it. Reviews are operator-only, so this
    /// is "operator" in practice; the export carries it for the audit trail.
    pub actor: String,
    pub decision: Decision,
    pub class: Option<String>,
    pub merged_into: Option<String>,
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Weakness {
    pub id: String,
    pub title: String,
    pub behavior: String,
    pub expected: String,
    pub status: Status,
    pub suggested_class: Option<String>,
    pub assigned_class: Option<String>,
    pub merged_into: Option<String>,
    pub severity: Severity,
    pub reported_at: String,
    pub reported_by: String,
    pub reported_on_model: String,
    pub models_observed: BTreeSet<String>,
    pub instances: Vec<InstanceRecord>,
    pub reviews: Vec<ReviewRecord>,
}

fn fold(entries: &[Entry]) -> BTreeMap<String, Weakness> {
    let mut map: BTreeMap<String, Weakness> = BTreeMap::new();
    for entry in entries {
        let record = |instance: &Instance| InstanceRecord {
            seq: entry.seq,
            added_at: entry.timestamp.clone(),
            actor: entry.actor.clone(),
            model: entry.model.clone(),
            after_event_id: entry.after_event_id,
            justification: instance.justification.clone(),
            evidence: instance.evidence.clone(),
        };
        match &entry.action {
            Action::Report {
                weakness_id,
                title,
                behavior,
                expected,
                suggested_class,
                severity,
                instance,
            } => {
                map.insert(
                    weakness_id.clone(),
                    Weakness {
                        id: weakness_id.clone(),
                        title: title.clone(),
                        behavior: behavior.clone(),
                        expected: expected.clone(),
                        status: Status::Proposed,
                        suggested_class: suggested_class.clone(),
                        assigned_class: None,
                        merged_into: None,
                        severity: *severity,
                        reported_at: entry.timestamp.clone(),
                        reported_by: entry.actor.clone(),
                        reported_on_model: entry.model.clone(),
                        models_observed: BTreeSet::from([entry.model.clone()]),
                        instances: vec![record(instance)],
                        reviews: Vec::new(),
                    },
                );
            }
            Action::AddInstance {
                weakness_id,
                instance,
            } => {
                if let Some(weakness) = map.get_mut(weakness_id) {
                    weakness.models_observed.insert(entry.model.clone());
                    weakness.instances.push(record(instance));
                }
            }
            Action::Review {
                weakness_id,
                decision,
                class,
                merged_into,
                note,
            } => {
                if let Some(weakness) = map.get_mut(weakness_id) {
                    match decision {
                        Decision::Confirm => weakness.status = Status::Confirmed,
                        Decision::Resolve => weakness.status = Status::Resolved,
                        Decision::Reject => weakness.status = Status::Rejected,
                        Decision::Merge => weakness.status = Status::Merged,
                        // The undo of the decision before it. Undoing `resolve`
                        // means the fix did not hold, which is a confirmed
                        // weakness again, not an unreviewed claim; anything
                        // else goes back to being s0ma's own proposal.
                        Decision::Reopen => {
                            weakness.status = if weakness.status == Status::Resolved {
                                Status::Confirmed
                            } else {
                                Status::Proposed
                            }
                        }
                        Decision::Classify => {}
                    }
                    if class.is_some() {
                        weakness.assigned_class = class.clone();
                    }
                    match decision {
                        Decision::Merge => weakness.merged_into = merged_into.clone(),
                        Decision::Classify => {}
                        _ => weakness.merged_into = None,
                    }
                    weakness.reviews.push(ReviewRecord {
                        seq: entry.seq,
                        at: entry.timestamp.clone(),
                        actor: entry.actor.clone(),
                        decision: *decision,
                        class: class.clone(),
                        merged_into: merged_into.clone(),
                        note: note.clone(),
                    });
                }
            }
        }
    }
    map
}

// ------------------------------------------------------------------ the log

/// Replace `path`'s contents in one step: a temp file beside it, fsynced, then
/// renamed over the target. A reader that opens the path mid-write — `q10.sh`
/// reads the export while the harness is running — sees either the whole old
/// file or the whole new one, never a half-written one. The directory entry is
/// fsynced too, so the rename survives a crash.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if path.file_name().is_none() {
        return Err(format!("{} is not a file path", path.display()).into());
    }
    crate::util::write_atomic(path, bytes)?;
    Ok(())
}

pub struct WeaknessLog {
    log_path: PathBuf,
    export_path: PathBuf,
    q10_export_path: PathBuf,
}

impl WeaknessLog {
    pub fn beside_archive(archive: &Path) -> Self {
        Self {
            log_path: archive.with_file_name("weaknesses.jsonl"),
            export_path: archive.with_file_name("weaknesses.json"),
            q10_export_path: archive.with_file_name("weaknesses-export.jsonl"),
        }
    }

    /// `weaknesses.json`: the whole fold as one pretty object, for reading by
    /// hand.
    pub fn export_path(&self) -> &Path {
        &self.export_path
    }

    /// `weaknesses-export.jsonl`: the same fold one weakness per line, for Q10.
    /// `q10.sh` reads this path by default.
    pub fn q10_export_path(&self) -> &Path {
        &self.q10_export_path
    }

    pub fn entries(&self) -> Result<Vec<Entry>> {
        if !self.log_path.exists() {
            return Ok(Vec::new());
        }
        let mut entries = Vec::new();
        for (index, line) in BufReader::new(File::open(&self.log_path)?).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let entry: Entry = serde_json::from_str(&line).map_err(|error| {
                format!(
                    "{} line {} is unreadable ({error}); the log is append-only, so repair it by hand and run /weakness verify",
                    self.log_path.display(),
                    index + 1
                )
            })?;
            entries.push(entry);
        }
        Ok(entries)
    }

    pub fn weaknesses(&self) -> Result<BTreeMap<String, Weakness>> {
        Ok(fold(&self.entries()?))
    }

    fn append(
        &self,
        actor: &str,
        model: &str,
        after_event_id: u64,
        action: Action,
    ) -> Result<Entry> {
        let entries = self.entries()?;
        let (seq, prev_hash) = match entries.last() {
            Some(last) => (last.seq + 1, last.hash.clone()),
            None => (1, GENESIS.to_string()),
        };
        let mut entry = Entry {
            seq,
            timestamp: Local::now().to_rfc3339(),
            actor: actor.to_string(),
            model: model.to_string(),
            after_event_id,
            action,
            prev_hash,
            hash: String::new(),
        };
        entry.hash = entry.compute_hash()?;

        if let Some(parent) = self.log_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        let mut line = serde_json::to_vec(&entry)?;
        line.push(b'\n');
        file.write_all(&line)?;
        file.sync_all()?;

        let mut all = entries;
        all.push(entry.clone());
        self.write_export(&all)?;
        Ok(entry)
    }

    /// Rebuild every derived file from `entries`. One fold, both writers, so
    /// `weaknesses.json` and the Q10 export can never describe different states
    /// of the log.
    fn write_export(&self, entries: &[Entry]) -> Result<()> {
        let weaknesses = fold(entries);
        self.write_state_export(entries, &weaknesses)?;
        self.write_q10_export(entries, &weaknesses)
    }

    fn write_state_export(
        &self,
        entries: &[Entry],
        weaknesses: &BTreeMap<String, Weakness>,
    ) -> Result<()> {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for weakness in weaknesses.values() {
            *counts.entry(status_word(weakness.status)).or_default() += 1;
        }
        let export = json!({
            "schema": EXPORT_SCHEMA,
            "generated_at": Local::now().to_rfc3339(),
            "source_log": self.log_path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "log_entries": entries.len(),
            "log_head_sha256": entries.last().map(|e| e.hash.clone()),
            "counts": counts,
            "weaknesses": weaknesses.values().collect::<Vec<_>>(),
        });
        let mut bytes = serde_json::to_vec_pretty(&export)?;
        bytes.push(b'\n');
        write_atomic(&self.export_path, &bytes)
    }

    /// The Q10 export: a header line, then one line per weakness, **whatever
    /// its status**. The exporter holds no policy — qref filters on `status`,
    /// `merged_into`, `expected` and `models_observed` — which keeps this file a
    /// faithful fold of the log and auditable on its own.
    ///
    /// The field names are qref's, not this module's: `sha256` for the event
    /// digest, `intact` for an episode's evidence, `class` for whichever class
    /// is in force. Classes come only from entries logged before 2026-10-26;
    /// newer weaknesses export `class: null`, which qref reads as unclassified.
    /// See `claude/weakness-export-spec.md`; the authoritative prose is the
    /// module docstring of `qref/weakness.py`.
    fn write_q10_export(
        &self,
        entries: &[Entry],
        weaknesses: &BTreeMap<String, Weakness>,
    ) -> Result<()> {
        let last = entries.last();
        let mut out = serde_json::to_string(&json!({"export": {
            // The one field with downstream weight: qref records it in the
            // trained slice's manifest, so a served slice names the exact state
            // of the log it came from.
            "chain_head": last.map(|entry| entry.hash.as_str()),
            "seq": last.map_or(0, |entry| entry.seq),
            "model": last.map(|entry| entry.model.as_str()),
            "written": Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
            "weaknesses": weaknesses.len(),
        }}))?;
        out.push('\n');

        for weakness in weaknesses.values() {
            let instances: Vec<Value> = weakness
                .instances
                .iter()
                .enumerate()
                .map(|(index, instance)| {
                    let mut events = Vec::new();
                    let mut episodes = Vec::new();
                    for evidence in &instance.evidence {
                        match evidence {
                            Evidence::Event {
                                id,
                                speaker,
                                timestamp,
                                content_sha256,
                                excerpt,
                            } => events.push(json!({
                                "id": id,
                                "speaker": speaker,
                                "timestamp": timestamp,
                                "sha256": content_sha256,
                                "excerpt": raw_excerpt(excerpt),
                            })),
                            Evidence::Episode {
                                id,
                                source_evidence_unchanged,
                            } => episodes.push(json!({
                                "id": id,
                                "intact": source_evidence_unchanged,
                            })),
                        }
                    }
                    json!({
                        // Stable across exports: the log is append-only and the
                        // fold keeps instances in log order, so an instance
                        // keeps its number for good. qref keys the drafted
                        // corrections on it, and a renumbering would throw away
                        // every correction written so far.
                        "instance_id": instance_id(&weakness.id, index),
                        "seq": instance.seq,
                        "at": instance.added_at,
                        "after_event_id": instance.after_event_id,
                        "actor": instance.actor,
                        "model": instance.model,
                        "justification": instance.justification,
                        "events": events,
                        "episodes": episodes,
                    })
                })
                .collect();

            let reviews: Vec<Value> = weakness
                .reviews
                .iter()
                .map(|review| {
                    json!({
                        "seq": review.seq,
                        "decision": review.decision,
                        "actor": review.actor,
                        "at": review.at,
                        "class": review.class,
                        "merged_into": review.merged_into,
                        "note": review.note,
                    })
                })
                .collect();

            out.push_str(&serde_json::to_string(&json!({
                "id": weakness.id,
                "status": status_word(weakness.status),
                "title": weakness.title,
                "behavior": weakness.behavior,
                "expected": weakness.expected,
                // The operator's class if there is one, else s0ma's guess, as
                // weakness_list shows it. Both are given separately so an audit
                // can tell which it is.
                "class": weakness.assigned_class.as_ref().or(weakness.suggested_class.as_ref()),
                "assigned_class": weakness.assigned_class,
                "suggested_class": weakness.suggested_class,
                "severity": weakness.severity,
                "models_observed": weakness.models_observed,
                "merged_into": weakness.merged_into,
                "reported_at": weakness.reported_at,
                "reported_on_model": weakness.reported_on_model,
                "instances": instances,
                "reviews": reviews,
            }))?);
            out.push('\n');
        }

        write_atomic(&self.q10_export_path, out.as_bytes())
    }

    /// Rebuild `weaknesses.json` and the Q10 export from the log.
    pub fn export(&self) -> Result<()> {
        self.write_export(&self.entries()?)
    }

    /// Check the hash chain, then re-check every cited event against the archive.
    pub fn verify(&self, source: &dyn EvidenceSource) -> Result<Vec<String>> {
        let entries = self.entries()?;
        let mut problems = Vec::new();
        let mut prev = GENESIS.to_string();
        for (index, entry) in entries.iter().enumerate() {
            if entry.seq != index as u64 + 1 {
                problems.push(format!("entry {}: seq is {}, expected {}", index + 1, entry.seq, index + 1));
            }
            if entry.prev_hash != prev {
                problems.push(format!("seq {}: prev_hash does not match the previous entry (chain broken)", entry.seq));
            }
            if entry.compute_hash()? != entry.hash {
                problems.push(format!("seq {}: content does not match its hash (entry edited)", entry.seq));
            }
            prev = entry.hash.clone();
        }

        let mut cited: BTreeMap<u64, (String, String)> = BTreeMap::new();
        for entry in &entries {
            let instance = match &entry.action {
                Action::Report { instance, .. } | Action::AddInstance { instance, .. } => instance,
                Action::Review { .. } => continue,
            };
            for evidence in &instance.evidence {
                if let Evidence::Event { id, content_sha256, .. } = evidence {
                    cited.insert(*id, (entry.action.weakness_id().to_string(), content_sha256.clone()));
                }
            }
        }
        let current: BTreeMap<u64, String> = source
            .events(&cited.keys().copied().collect::<Vec<_>>())?
            .iter()
            .map(|event| Ok((event.id, event_digest(event)?)))
            .collect::<Result<_>>()?;
        for (id, (weakness_id, digest)) in &cited {
            match current.get(id) {
                None => problems.push(format!("{weakness_id}: cited event {id} no longer exists in the archive")),
                Some(now) if now != digest => problems.push(format!("{weakness_id}: cited event {id} changed since it was cited")),
                Some(_) => {}
            }
        }
        Ok(problems)
    }

    fn next_weakness_id(entries: &[Entry]) -> String {
        let reports = entries
            .iter()
            .filter(|e| matches!(e.action, Action::Report { .. }))
            .count();
        format!("wk-{:04}", reports + 1)
    }
}

// ------------------------------------------------------------------ derived files

/// Rebuild the derived files from the log: `weaknesses.json` and the Q10
/// export. Every append already does this, so the files are current the moment
/// a review lands. This is for the two cases an append does not cover, both of
/// which would otherwise be discovered by a Q10 night failing:
///
/// * **startup** — the files are derived, so a log that predates the exporter,
///   or one whose export was deleted, has no export beside it at all until
///   something appends. A first night must not depend on a review happening;
/// * **the sleep pass** — a night on which nothing was reviewed still leaves
///   `q10.sh` an export newer than 48 h, so it has nothing to warn about.
///
/// Cheap either way: a fold over a log of tens of entries.
///
/// Never fails the caller. Both files are derived and disposable — deleting them
/// loses nothing, and the next write rebuilds them from the log — so a sleep
/// pass must not be lost over one.
pub fn refresh_export(memory: &Hippocampus) {
    let log = WeaknessLog::beside_archive(memory.archive_path());
    if let Err(error) = log.export() {
        crate::terminal::warn(&format!(
            "[weakness] export not refreshed; tonight's Q10 run may warn that it is stale: {error}"
        ));
    }
}

// ------------------------------------------------------------------ tools

pub fn tool_definitions() -> Vec<ToolDefinition> {
    let ids = json!({
        "event_ids": {
            "type": "array", "items": {"type": "integer", "minimum": 1}, "maxItems": MAX_IDS_PER_KIND,
            "description": "Archive event IDs where the weakness shows (find them with archive_search / archive_get). \
                            Include the one holding your own turn, the one that went wrong, not only the operator's reply about it."
        },
        "episode_ids": {
            "type": "array", "items": {"type": "string"}, "maxItems": MAX_IDS_PER_KIND,
            "description": "Consolidated episode IDs where it shows (episode_get)."
        }
    });
    let justification = json!({
        "type": "string",
        "description": "What in the cited events shows the weakness: point to the specific claim, action or omission, not a restatement of the title."
    });
    // Why this insistence: the log is training data. The lesson rendered from an
    // instance is the archive up to the mistake, then the corrected turn — so an
    // instance that cites only the operator's complaint has no mistake to correct and
    // teaches nothing.
    let own_turn = " Cite the event where you did it — your own turn — and not only the operator's correction of it: an \
                     instance that points at the complaint alone shows the verdict without the behaviour, and nothing \
                     can be learned from it.";

    let mut report_props = json!({
        "title": {"type": "string", "description": "Short name for the weakness, e.g. 'states HTB box facts not in any tool output'."},
        "behavior": {"type": "string", "description": "What you actually did, concretely."},
        // Not decoration: `expected` is the target the Q10 lesson trains
        // toward, and the fallback when no corrected turn has been drafted yet.
        // A vague one is the most common reason a weakness teaches nothing.
        "expected": {"type": "string", "description": "What you should have done instead, in your own voice, as a rule you could follow next time — e.g. 'I run the tests before I say something is fixed.' This is the lesson, so make it concrete and don't leave it to be inferred."},
        "justification": justification.clone(),
        "severity": {"type": "string", "enum": ["low", "medium", "high"], "description": "How much it mattered in this case. Default medium."}
    });
    let mut add_props = json!({
        "weakness_id": {"type": "string", "description": "Existing ID from weakness_list, e.g. wk-0003."},
        "justification": justification
    });
    for props in [&mut report_props, &mut add_props] {
        if let (Some(target), Some(extra)) = (props.as_object_mut(), ids.as_object()) {
            target.extend(extra.clone());
        }
    }

    vec![
        ToolDefinition::function(
            "weakness_report",
            format!("Record a weakness in your own behavior, observed in real operation, for the operator to review. Use it when a specific \
             past event shows you getting something wrong: a confabulated fact, a claim no tool result supports, a repetition \
             loop, steering talk toward your own interests, giving up on or hedging a task you could have pushed further, \
             refusing legitimate work, abstaining on something that was in your context. Also use it when the operator points out a \
             failure and asks you to log it. Cite the archive event IDs and/or episode IDs where it happened; every ID is \
             checked, and the report is rejected if any don't exist. This records a candidate only: the operator confirms or \
             rejects it. Call weakness_list first, and if the same weakness is already logged use weakness_add_instance \
             instead. Finish the task in front of you before logging, and don't let logging take over the conversation.\
             {own_turn}"),
            json!({"type": "object", "properties": report_props,
                   "required": ["title", "behavior", "expected", "justification"]}),
        ),
        ToolDefinition::function(
            "weakness_add_instance",
            format!("Add another observed instance to a weakness already in the log (see weakness_list), with its own cited event or \
             episode IDs and a justification for why this is the same weakness. Worth doing for a weakness the operator has already \
             marked resolved, too: a recurrence there is news.{own_turn}"),
            json!({"type": "object", "properties": add_props,
                   "required": ["weakness_id", "justification"]}),
        ),
        ToolDefinition::function(
            "weakness_list",
            "List logged weaknesses (ID, title, status, instance count) so you can check for an existing entry before \
             reporting. Default shows open ones (proposed and confirmed); \"resolved\" lists the ones the operator has marked fixed, \
             which stay on record and are still worth an instance if one recurs.",
            json!({"type": "object", "properties": {
                "status": {"type": "string", "enum": ["open", "proposed", "confirmed", "resolved", "rejected", "merged", "all"]}
            }}),
        ),
    ]
}

pub fn execute(memory: &Hippocampus, call: &ToolCall) -> Result<String> {
    let log = WeaknessLog::beside_archive(memory.archive_path());
    execute_with(&log, memory, &crate::llm::model_name(), call)
}

fn text_field(name: &str, value: &str, min: usize) -> std::result::Result<String, String> {
    let value = value.trim();
    let chars = value.chars().count();
    if chars < min {
        return Err(format!("{name} is too short (at least {min} characters)"));
    }
    if chars > MAX_FIELD_CHARS {
        return Err(format!("{name} is too long (at most {MAX_FIELD_CHARS} characters)"));
    }
    Ok(value.to_string())
}

fn rejected(error: impl Into<String>) -> String {
    json!({"ok": false, "error": error.into(), "recorded": false}).to_string()
}

/// Every archive event the log cites.
///
/// The log keeps a 500-character excerpt of everything it cites and is
/// hash-chained, so a redaction of one of these cannot reach the copy in here.
/// The memory tab says so beside the event: it is better to know a removal
/// will be partial than to be told it was complete.
pub fn cited_events(memory: &Hippocampus) -> std::collections::HashSet<u64> {
    let log = WeaknessLog::beside_archive(memory.archive_path());
    log.entries()
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| match &entry.action {
                    Action::Report { instance, .. } | Action::AddInstance { instance, .. } => Some(instance),
                    Action::Review { .. } => None,
                })
                .flat_map(|instance| instance.evidence.iter())
                .filter_map(|evidence| match evidence {
                    Evidence::Event { id, .. } => Some(*id),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Whether the cited events include a turn of s0ma's own.
///
/// Q10 renders a lesson from the last cited event that became one of s0ma's own
/// turns: that is the mistake, and the prefix before it is the situation that
/// invited it. An instance citing only the operator's correction has no mistake
/// in it, so qref counts it unusable and it trains nothing. Not grounds to
/// refuse the record — the observation is still worth keeping, and the operator
/// can ask for a better instance — but worth saying plainly while s0ma still
/// has the event IDs to hand.
fn cites_own_turn(evidence: &[Evidence]) -> bool {
    evidence
        .iter()
        .any(|item| matches!(item, Evidence::Event { speaker, .. } if speaker == "s0ma"))
}

const CITE_OWN_TURN: &str = "None of the cited events is a turn of yours: as logged, this instance \
     shows the complaint about the mistake but not the mistake, which makes it evidence of the \
     verdict only. Find the event holding what you actually said or did and add it with \
     weakness_add_instance.";

pub fn execute_with(
    log: &WeaknessLog,
    source: &dyn EvidenceSource,
    model: &str,
    call: &ToolCall,
) -> Result<String> {
    match call.function.name.as_str() {
        "weakness_report" => {
            #[derive(Deserialize)]
            struct Args {
                title: String,
                behavior: String,
                expected: String,
                justification: String,
                // `suggested_class` is no longer offered; one sent anyway is
                // ignored, as any unknown argument is.
                #[serde(default)]
                severity: Option<Severity>,
                #[serde(default)]
                event_ids: Vec<u64>,
                #[serde(default)]
                episode_ids: Vec<String>,
            }
            let args: Args = serde_json::from_str(&call.function.arguments)?;
            let fields = (|| {
                Ok::<_, String>((
                    text_field("title", &args.title, 5)?,
                    text_field("behavior", &args.behavior, MIN_FIELD_CHARS)?,
                    text_field("expected", &args.expected, MIN_FIELD_CHARS)?,
                    text_field("justification", &args.justification, MIN_JUSTIFICATION_CHARS)?,
                ))
            })();
            let (title, behavior, expected, justification) = match fields {
                Ok(fields) => fields,
                Err(error) => return Ok(rejected(error)),
            };
            let evidence = match collect_evidence(source, &args.event_ids, &args.episode_ids)? {
                Ok(evidence) => evidence,
                Err(error) => return Ok(rejected(error)),
            };
            let own_turn = cites_own_turn(&evidence);
            let weakness_id = WeaknessLog::next_weakness_id(&log.entries()?);
            log.append(
                "s0ma",
                model,
                source.latest_event_id(),
                Action::Report {
                    weakness_id: weakness_id.clone(),
                    title,
                    behavior,
                    expected,
                    suggested_class: None,
                    severity: args.severity.unwrap_or(Severity::Medium),
                    instance: Instance {
                        justification,
                        evidence,
                    },
                },
            )?;
            let mut result = json!({
                "ok": true, "recorded": true, "weakness_id": weakness_id, "status": "proposed",
                "note": "Recorded as a candidate for the operator to review. Carry on with what you were doing."
            });
            if !own_turn {
                result["evidence_note"] = json!(CITE_OWN_TURN);
            }
            Ok(result.to_string())
        }

        "weakness_add_instance" => {
            #[derive(Deserialize)]
            struct Args {
                weakness_id: String,
                justification: String,
                #[serde(default)]
                event_ids: Vec<u64>,
                #[serde(default)]
                episode_ids: Vec<String>,
            }
            let args: Args = serde_json::from_str(&call.function.arguments)?;
            let weakness_id = args.weakness_id.trim().to_string();
            let weaknesses = log.weaknesses()?;
            let Some(existing) = weaknesses.get(&weakness_id) else {
                return Ok(rejected(format!("no weakness {weakness_id}; see weakness_list")));
            };
            if existing.status == Status::Merged {
                return Ok(rejected(format!(
                    "{weakness_id} was merged into {}; add the instance there",
                    existing.merged_into.as_deref().unwrap_or("another entry")
                )));
            }
            let justification =
                match text_field("justification", &args.justification, MIN_JUSTIFICATION_CHARS) {
                    Ok(text) => text,
                    Err(error) => return Ok(rejected(error)),
                };
            let evidence = match collect_evidence(source, &args.event_ids, &args.episode_ids)? {
                Ok(evidence) => evidence,
                Err(error) => return Ok(rejected(error)),
            };
            let own_turn = cites_own_turn(&evidence);
            log.append(
                "s0ma",
                model,
                source.latest_event_id(),
                Action::AddInstance {
                    weakness_id: weakness_id.clone(),
                    instance: Instance {
                        justification,
                        evidence,
                    },
                },
            )?;
            let mut result = json!({
                "ok": true, "recorded": true, "weakness_id": weakness_id,
                // Unchanged by adding an instance, but worth saying: an instance
                // on a resolved weakness means a fix stopped holding, which is
                // The operator's to decide about (/weakness reopen), not s0ma's.
                "status": status_word(existing.status),
                "instances": existing.instances.len() + 1
            });
            if !own_turn {
                result["evidence_note"] = json!(CITE_OWN_TURN);
            }
            Ok(result.to_string())
        }

        "weakness_list" => {
            #[derive(Deserialize, Default)]
            struct Args {
                #[serde(default)]
                status: Option<String>,
            }
            let args: Args = if call.function.arguments.trim().is_empty() {
                Args::default()
            } else {
                serde_json::from_str(&call.function.arguments)?
            };
            let filter = args.status.unwrap_or_else(|| "open".into());
            let keep = |status: Status| match filter.as_str() {
                "all" => true,
                "proposed" | "confirmed" | "resolved" | "rejected" | "merged" => {
                    filter == status_word(status)
                }
                _ => matches!(status, Status::Proposed | Status::Confirmed),
            };
            let rows: Vec<Value> = log
                .weaknesses()?
                .values()
                .filter(|w| keep(w.status))
                .map(|w| {
                    json!({
                        "id": w.id, "title": w.title, "status": w.status,
                        "instances": w.instances.len(),
                        "last_seen": w.instances.last().map(|i| i.added_at.clone()),
                        "behavior": excerpt(&w.behavior, 240),
                    })
                })
                .collect();
            Ok(json!({"ok": true, "filter": filter, "count": rows.len(), "weaknesses": rows}).to_string())
        }

        other => Err(format!("unknown weakness tool: {other}").into()),
    }
}

// ------------------------------------------------------------------ operator review

const HELP: &str = "\
Weakness log review (/weakness and /weaknesses are the same command).
s0ma reports candidates; only these commands confirm or reject them. A weakness's
one ID is its wk- number, and none of them is shown in s0ma's context.

/weaknesses help      this text (also -h, --help, ?)
/weaknesses           same as /weaknesses list
/weakness list [all|proposed|confirmed|resolved|rejected|merged]   default: proposed + confirmed
/weakness show <id>
/weakness confirm <id> [note]               e.g. /weakness confirm wk-0003 clear confabulation;
                                            Q10 trains it from tonight
/weakness resolve <id> [note]               trained out: Q10 keeps training it and gates
                                            every night on the fix holding
/weakness reject <id> [note]
/weakness merge <id> <into-id> [note]       mark a duplicate; its instances stay on record
/weakness reopen <id> [note]                undo the last decision: resolved -> confirmed, else -> proposed
/weakness verify      check the hash chain and re-check cited evidence against the archive
/weakness export      rebuild weaknesses.json and weaknesses-export.jsonl (Q10's input) from the log";

/// One weakness as `/weakness show` prints it: the folded record without the
/// class fields or the log's sequence numbers, so the wk- number is the only ID
/// in it beside the archive's own event and episode IDs. The files on disk and
/// the Q10 export keep everything.
fn shown(weakness: &Weakness) -> Result<Value> {
    let mut value = serde_json::to_value(weakness)?;
    if let Some(record) = value.as_object_mut() {
        record.remove("suggested_class");
        record.remove("assigned_class");
        for list in ["instances", "reviews"] {
            for item in record.get_mut(list).and_then(Value::as_array_mut).into_iter().flatten() {
                if let Some(item) = item.as_object_mut() {
                    item.remove("seq");
                    item.remove("class");
                }
            }
        }
    }
    Ok(value)
}

/// Operator-only review commands. Returns None if `command` isn't a /weakness command.
pub fn handle_operator_command(
    memory: &Hippocampus,
    command: &str,
) -> Option<std::result::Result<String, String>> {
    let log = WeaknessLog::beside_archive(memory.archive_path());
    handle_operator_with(&log, memory, &crate::llm::model_name(), command)
}

pub fn handle_operator_with(
    log: &WeaknessLog,
    source: &dyn EvidenceSource,
    model: &str,
    command: &str,
) -> Option<std::result::Result<String, String>> {
    let mut words = command.split_whitespace();
    let head = words.next()?;
    if head != "/weakness" && head != "/weaknesses" {
        return None;
    }
    let args: Vec<&str> = words.collect();
    Some(operator(log, source, model, &args).map_err(|e| format!("[weakness] {e}")))
}

fn operator(
    log: &WeaknessLog,
    source: &dyn EvidenceSource,
    model: &str,
    args: &[&str],
) -> Result<String> {
    let sub = args.first().copied().unwrap_or("list");
    let rest = |from: usize| args.get(from..).unwrap_or(&[]).join(" ");
    match sub {
        "help" | "-h" | "--help" | "?" => Ok(HELP.to_string()),

        "list" => {
            let filter = args.get(1).copied().unwrap_or("open");
            let weaknesses = log.weaknesses()?;
            let mut lines = Vec::new();
            for w in weaknesses.values() {
                let shown = match filter {
                    "all" => true,
                    // A resolved weakness is not open: it is listed on request
                    // and by `all`, but it is no longer waiting for anything.
                    "proposed" | "confirmed" | "resolved" | "rejected" | "merged" => {
                        filter == status_word(w.status)
                    }
                    _ => matches!(w.status, Status::Proposed | Status::Confirmed),
                };
                if shown {
                    lines.push(format!(
                        "{}  {:<9} x{:<3} {}",
                        w.id,
                        status_word(w.status),
                        w.instances.len(),
                        w.title
                    ));
                }
            }
            if lines.is_empty() {
                Ok(format!("[weakness] none ({filter})"))
            } else {
                Ok(lines.join("\n"))
            }
        }

        "show" => {
            let id = args.get(1).ok_or("usage: /weakness show <id>")?;
            let weaknesses = log.weaknesses()?;
            let w = weaknesses.get(*id).ok_or(format!("no weakness {id}"))?;
            Ok(serde_json::to_string_pretty(&shown(w)?)?)
        }

        "classify" => Err("classify is gone: weaknesses have no class now, and the wk- number is their only ID".into()),

        "confirm" | "resolve" | "reject" | "merge" | "reopen" => {
            let id = args.get(1).ok_or("usage: see /weakness help".to_string())?.to_string();
            let weaknesses = log.weaknesses()?;
            if !weaknesses.contains_key(&id) {
                return Err(format!("no weakness {id}").into());
            }
            let (decision, class, merged_into, note) = match sub {
                "confirm" => {
                    // `-` stood for "no class" when confirm took one; it is
                    // dropped rather than kept as the start of the note.
                    let from = if args.get(2) == Some(&"-") { 3 } else { 2 };
                    (Decision::Confirm, None, None, rest(from))
                }
                "resolve" => {
                    // The ratchet only turns one way: a weakness is resolved
                    // after it was confirmed and trained out, so resolving one
                    // that was never confirmed would hand Q10 a nightly
                    // regression test for a claim nobody reviewed.
                    let status = weaknesses[&id].status;
                    if !matches!(status, Status::Confirmed | Status::Resolved) {
                        return Err(format!(
                            "{id} is {}; only a confirmed weakness can be resolved — confirm it first",
                            status_word(status)
                        )
                        .into());
                    }
                    (Decision::Resolve, None, None, rest(2))
                }
                "merge" => {
                    let into = args.get(2).ok_or("usage: /weakness merge <id> <into-id> [note]")?.to_string();
                    if into == id {
                        return Err("can't merge a weakness into itself".into());
                    }
                    match weaknesses.get(&into) {
                        None => return Err(format!("no weakness {into}").into()),
                        Some(target) if target.status == Status::Merged => {
                            return Err(format!("{into} is itself merged; merge into its target").into())
                        }
                        Some(_) => {}
                    }
                    (Decision::Merge, None, Some(into), rest(3))
                }
                "reject" => (Decision::Reject, None, None, rest(2)),
                _ => (Decision::Reopen, None, None, rest(2)),
            };
            log.append(
                "operator",
                model,
                source.latest_event_id(),
                Action::Review {
                    weakness_id: id.clone(),
                    decision,
                    class,
                    merged_into,
                    note,
                },
            )?;
            let note = match sub {
                "confirm" => "; Q10 trains it from tonight",
                "resolve" => "; from tonight Q10 trains it as a regression that must hold",
                _ => "",
            };
            Ok(format!("[weakness] {id}: {sub} recorded{note}"))
        }

        "verify" => {
            let problems = log.verify(source)?;
            let count = log.entries()?.len();
            if problems.is_empty() {
                Ok(format!("[weakness] {count} log entries: chain intact, all cited events unchanged"))
            } else {
                Ok(format!("[weakness] {} problem(s):\n{}", problems.len(), problems.join("\n")))
            }
        }

        "export" => {
            log.export()?;
            Ok(format!(
                "[weakness] wrote {} and {}",
                log.export_path().display(),
                log.q10_export_path().display()
            ))
        }

        other => Err(format!("unknown subcommand '{other}'\n{HELP}").into()),
    }
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionCall;
    use std::cell::RefCell;

    struct Fake {
        events: RefCell<BTreeMap<u64, (String, String)>>,
        episodes: BTreeMap<String, bool>,
    }

    impl Fake {
        fn new() -> Self {
            let mut events = BTreeMap::new();
            for id in 1..=10 {
                events.insert(id, ("operator".to_string(), format!("event {id} content")));
            }
            // A turn of s0ma's own, longer than EXCERPT_CHARS so the snapshot
            // is cut short: the two things the Q10 export has to get right.
            events.insert(
                11,
                (
                    "s0ma".to_string(),
                    format!("The parser is fixed. {}end", "verbose padding ".repeat(60)),
                ),
            );
            Self {
                events: RefCell::new(events),
                episodes: BTreeMap::from([("ep-a".to_string(), true)]),
            }
        }

        fn content(&self, id: u64) -> String {
            self.events.borrow()[&id].1.clone()
        }
    }

    impl EvidenceSource for Fake {
        fn events(&self, ids: &[u64]) -> Result<Vec<EventSnapshot>> {
            Ok(ids
                .iter()
                .filter_map(|id| {
                    self.events.borrow().get(id).map(|(speaker, content)| EventSnapshot {
                        id: *id,
                        timestamp: "2026-09-24T00:00:00+00:00".into(),
                        speaker: if speaker == "operator" { "operator" } else { "s0ma" },
                        content: content.clone(),
                    })
                })
                .collect())
        }
        fn episode(&self, id: &str) -> Result<Option<bool>> {
            Ok(self.episodes.get(id).copied())
        }
        fn latest_event_id(&self) -> u64 {
            11
        }
    }

    fn temp_log(name: &str) -> WeaknessLog {
        let dir = std::env::temp_dir().join(format!("s0ma-weakness-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        WeaknessLog::beside_archive(&dir.join("hippocampus.jsonl"))
    }

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: FunctionCall { name: name.into(), arguments: args.to_string() },
        }
    }

    fn report(ids: Value) -> ToolCall {
        call("weakness_report", json!({
            "title": "invented a job title",
            "behavior": "Stated the operator's job title though nothing in context gave it.",
            "expected": "Say I don't know the job title.",
            "justification": "Event 3 asks the job title; event 4 answers 'security engineer' with no source in context.",
            "suggested_class": "confabulation",
            "event_ids": ids,
            "episode_ids": ["ep-a"]
        }))
    }

    fn parse(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn report_add_review_export_verify() {
        let log = temp_log("flow");
        let src = Fake::new();

        let out = parse(&execute_with(&log, &src, "qwen38-nvfp4", &report(json!([3, 4]))).unwrap());
        assert_eq!(out["ok"], true, "{out}");
        assert_eq!(out["weakness_id"], "wk-0001");
        assert_eq!(out.get("log_seq"), None, "the wk- number is the only ID handed back");

        let add = call("weakness_add_instance", json!({
            "weakness_id": "wk-0001",
            "justification": "Event 7 repeats the same move: an unsourced employer name stated as fact.",
            "event_ids": [7]
        }));
        let out = parse(&execute_with(&log, &src, "s0ma-v1", &add).unwrap());
        assert_eq!(out["instances"], 2, "{out}");
        assert_eq!(out.get("log_seq"), None);

        let msg = handle_operator_with(&log, &src, "s0ma-v1", "/weakness confirm wk-0001 clear case")
            .unwrap()
            .unwrap();
        assert_eq!(msg, "[weakness] wk-0001: confirm recorded; Q10 trains it from tonight");

        let export: Value = parse(&fs::read_to_string(log.export_path()).unwrap());
        assert_eq!(export["schema"], EXPORT_SCHEMA);
        let w = &export["weaknesses"][0];
        assert_eq!(w["status"], "confirmed");
        assert_eq!(w["assigned_class"], Value::Null, "confirm takes no class");
        assert_eq!(w["suggested_class"], Value::Null, "s0ma's suggestion is no longer recorded");
        assert_eq!(w["reviews"][0]["note"], "clear case");
        assert_eq!(w["instances"].as_array().unwrap().len(), 2);
        assert_eq!(w["models_observed"], json!(["qwen38-nvfp4", "s0ma-v1"]));
        assert_eq!(w["instances"][0]["evidence"][0]["kind"], "event");
        assert_eq!(w["instances"][0]["evidence"][2]["kind"], "episode");

        assert!(log.verify(&src).unwrap().is_empty());

        // Evidence changing after citation is caught.
        src.events.borrow_mut().insert(4, ("s0ma".into(), "rewritten".into()));
        let problems = log.verify(&src).unwrap();
        assert!(problems.iter().any(|p| p.contains("event 4 changed")), "{problems:?}");
    }

    #[test]
    fn rejects_missing_or_absent_evidence_and_records_nothing() {
        let log = temp_log("reject");
        let src = Fake::new();
        let out = parse(&execute_with(&log, &src, "m", &report(json!([3, 99]))).unwrap());
        assert_eq!(out["recorded"], false);
        assert!(out["error"].as_str().unwrap().contains("99"));

        let none = call("weakness_report", json!({
            "title": "something", "behavior": "did a wrong thing", "expected": "do the right thing",
            "justification": "a sufficiently long justification with no evidence cited at all"
        }));
        let out = parse(&execute_with(&log, &src, "m", &none).unwrap());
        assert_eq!(out["recorded"], false);

        let thin = call("weakness_report", json!({
            "title": "something", "behavior": "did a wrong thing", "expected": "do the right thing",
            "justification": "bad", "event_ids": [1]
        }));
        let out = parse(&execute_with(&log, &src, "m", &thin).unwrap());
        assert_eq!(out["recorded"], false);

        assert!(log.entries().unwrap().is_empty());
    }

    #[test]
    fn tampering_breaks_the_chain() {
        let log = temp_log("tamper");
        let src = Fake::new();
        execute_with(&log, &src, "m", &report(json!([3]))).unwrap();
        execute_with(&log, &src, "m", &report(json!([5]))).unwrap();
        let text = fs::read_to_string(&log.log_path).unwrap();
        fs::write(&log.log_path, text.replacen("invented a job title", "edited title", 1)).unwrap();
        let problems = log.verify(&src).unwrap();
        assert!(problems.iter().any(|p| p.contains("entry edited")), "{problems:?}");
    }

    #[test]
    fn merge_rules_and_list_filters() {
        let log = temp_log("merge");
        let src = Fake::new();
        execute_with(&log, &src, "m", &report(json!([3]))).unwrap();
        execute_with(&log, &src, "m", &report(json!([5]))).unwrap();

        let err = handle_operator_with(&log, &src, "m", "/weakness merge wk-0002 wk-0002").unwrap();
        assert!(err.is_err());
        handle_operator_with(&log, &src, "m", "/weakness merge wk-0002 wk-0001 dup").unwrap().unwrap();

        let add = call("weakness_add_instance", json!({
            "weakness_id": "wk-0002",
            "justification": "another instance of the same unsourced claim pattern here"
            , "event_ids": [6]
        }));
        let out = parse(&execute_with(&log, &src, "m", &add).unwrap());
        assert_eq!(out["recorded"], false);
        assert!(out["error"].as_str().unwrap().contains("wk-0001"));

        let open = parse(&execute_with(&log, &src, "m", &call("weakness_list", json!({}))).unwrap());
        assert_eq!(open["count"], 1);
        let all = parse(&execute_with(&log, &src, "m", &call("weakness_list", json!({"status": "all"}))).unwrap());
        assert_eq!(all["count"], 2);

        assert!(handle_operator_with(&log, &src, "m", "/sleep").is_none());
        for cmd in ["/weaknesses help", "/weakness help", "/weaknesses --help", "/weakness ?"] {
            let help = handle_operator_with(&log, &src, "m", cmd).unwrap().unwrap();
            assert!(help.contains("/weakness confirm"), "{cmd}");
        }
        let bad = handle_operator_with(&log, &src, "m", "/weaknesses frobnicate").unwrap();
        assert!(bad.unwrap_err().contains("/weaknesses help"));
    }

    /// A log written before 2026-10-26, with classes on it: a review from the
    /// removed `classify` and a confirm that took one.
    fn old_review(log: &WeaknessLog, id: &str, decision: Decision, class: &str) {
        log.append(
            "operator",
            "m",
            1,
            Action::Review {
                weakness_id: id.into(),
                decision,
                class: Some(class.into()),
                merged_into: None,
                note: String::new(),
            },
        )
        .unwrap();
    }

    #[test]
    fn the_wk_number_is_the_only_id_shown() {
        let log = temp_log("ids");
        let src = Fake::new();
        execute_with(&log, &src, "m", &report(json!([3]))).unwrap();
        execute_with(&log, &src, "m", &report(json!([5]))).unwrap();
        old_review(&log, "wk-0001", Decision::Confirm, "W9");
        old_review(&log, "wk-0002", Decision::Classify, "W11");
        assert_eq!(log.weaknesses().unwrap()["wk-0001"].assigned_class.as_deref(), Some("W9"));

        // The operator's list and show.
        let rows = operator(&log, &src, "/weakness list");
        assert!(rows.contains("wk-0001") && rows.contains("wk-0002"), "{rows}");
        let shown = operator(&log, &src, "/weakness show wk-0001");
        for text in [&rows, &shown] {
            assert!(!text.contains("W9") && !text.contains("W11"), "{text}");
            assert!(!text.contains("confabulation"), "{text}");
            assert!(!text.contains("class") && !text.contains("\"seq\""), "{text}");
        }
        let shown: Value = parse(&shown);
        assert_eq!(shown["id"], "wk-0001");
        assert_eq!(shown["reviews"][0]["decision"], "confirm", "the review itself stays");
        assert_eq!(shown["instances"][0]["evidence"][0]["id"], 3, "archive IDs stay");

        // s0ma's list.
        let listed = execute_with(&log, &src, "m", &call("weakness_list", json!({"status": "all"}))).unwrap();
        assert!(!listed.contains("W9") && !listed.contains("W11") && !listed.contains("class"), "{listed}");
        assert_eq!(parse(&listed)["count"], 2);

        // A tool description that offered a class would invite one back.
        for tool in tool_definitions() {
            let text = serde_json::to_string(&tool).unwrap();
            assert!(!text.contains("class"), "{}: {text}", tool.function.name);
        }

        // The command that set one is gone, with a reason rather than a usage dump.
        let gone = handle_operator_with(&log, &src, "m", "/weakness classify wk-0001 W9").unwrap().unwrap_err();
        assert!(gone.contains("no class"), "{gone}");
        assert!(!HELP.contains("classify") && !HELP.contains("W9"), "{HELP}");

        // The old "no class" placeholder is not taken for the start of a note.
        operator(&log, &src, "/weakness confirm wk-0002 - same as before");
        let reviews = &log.weaknesses().unwrap()["wk-0002"].reviews;
        assert_eq!(reviews.last().unwrap().note, "same as before");
        assert_eq!(reviews.last().unwrap().class, None);

        // verify names the weakness and the event, not a log position.
        src.events.borrow_mut().insert(3, ("operator".into(), "rewritten".into()));
        let problems = log.verify(&src).unwrap();
        assert_eq!(problems, ["wk-0001: cited event 3 changed since it was cited"]);
    }

    // -------------------------------------------------------- the Q10 export

    /// The export as qref reads it: the header, then one weakness per line.
    fn read_q10(log: &WeaknessLog) -> Vec<Value> {
        fs::read_to_string(log.q10_export_path())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).expect("every line is one JSON object"))
            .collect()
    }

    fn q10_by_id(log: &WeaknessLog) -> BTreeMap<String, Value> {
        read_q10(log)[1..]
            .iter()
            .map(|w| (w["id"].as_str().unwrap().to_string(), w.clone()))
            .collect()
    }

    fn operator(log: &WeaknessLog, src: &Fake, command: &str) -> String {
        handle_operator_with(log, src, "m", command)
            .unwrap_or_else(|| panic!("not a weakness command: {command}"))
            .unwrap_or_else(|error| panic!("{command}: {error}"))
    }

    #[test]
    fn q10_export_carries_every_status_and_names_the_log_head() {
        let log = temp_log("q10-statuses");
        let src = Fake::new();
        for _ in 0..5 {
            execute_with(&log, &src, "qwen38-nvfp4", &report(json!([3, 11]))).unwrap();
        }
        operator(&log, &src, "/weakness confirm wk-0001 holds up");
        old_review(&log, "wk-0001", Decision::Classify, "verification");
        operator(&log, &src, "/weakness confirm wk-0002");
        operator(&log, &src, "/weakness resolve wk-0002 trained out on 2026-10-20");
        operator(&log, &src, "/weakness reject wk-0003 not a weakness");
        operator(&log, &src, "/weakness merge wk-0004 wk-0001 same thing");
        // wk-0005 is left proposed.

        let lines = read_q10(&log);
        assert_eq!(lines.len(), 6, "a header and every weakness: {lines:?}");

        let entries = log.entries().unwrap();
        let last = entries.last().unwrap();
        let header = &lines[0]["export"];
        assert_eq!(header["chain_head"], last.hash, "the head of the chain it was folded from");
        assert_eq!(header["seq"], last.seq);
        assert_eq!(header["model"], last.model);
        assert_eq!(header["weaknesses"], 5);
        let written = header["written"].as_str().unwrap();
        assert!(written.ends_with('Z') && written.len() == 20, "{written}");

        // Whatever the status, it is in the file, and it says what fold() says.
        let folded = log.weaknesses().unwrap();
        let exported = q10_by_id(&log);
        assert_eq!(exported.len(), folded.len());
        for (id, weakness) in &folded {
            assert_eq!(exported[id]["status"], status_word(weakness.status), "{id}");
        }
        assert_eq!(exported["wk-0001"]["status"], "confirmed");
        assert_eq!(exported["wk-0002"]["status"], "resolved");
        assert_eq!(exported["wk-0003"]["status"], "rejected");
        assert_eq!(exported["wk-0004"]["status"], "merged");
        assert_eq!(exported["wk-0005"]["status"], "proposed");

        // The fields qref decides on.
        let confirmed = &exported["wk-0001"];
        // A class logged before they were removed still reaches qref; a newer
        // weakness has none.
        assert_eq!(confirmed["class"], "verification");
        assert_eq!(confirmed["assigned_class"], "verification");
        assert_eq!(exported["wk-0002"]["class"], Value::Null);
        assert_eq!(confirmed["severity"], "medium");
        assert_eq!(confirmed["models_observed"], json!(["qwen38-nvfp4"]));
        assert_eq!(confirmed["merged_into"], Value::Null);
        assert_eq!(confirmed["expected"], "Say I don't know the job title.");
        assert_eq!(exported["wk-0004"]["merged_into"], "wk-0001");
        for weakness in exported.values() {
            assert!(
                !weakness["expected"].as_str().unwrap().is_empty(),
                "an empty `expected` would make the weakness contribute nothing"
            );
        }

        // One instance, its evidence split the way qref expects.
        let instance = &confirmed["instances"][0];
        assert_eq!(instance["instance_id"], "wk-0001/1");
        assert_eq!(instance["events"].as_array().unwrap().len(), 2);
        assert_eq!(instance["events"][0]["id"], 3);
        assert_eq!(instance["events"][1]["speaker"], "s0ma");
        assert_eq!(instance["episodes"], json!([{"id": "ep-a", "intact": true}]));
        assert_eq!(instance["after_event_id"], 11);
        assert!(instance["events"][0]["sha256"].as_str().unwrap().len() == 64);

        // Reviews are the audit trail, with who decided.
        let reviews = exported["wk-0002"]["reviews"].as_array().unwrap();
        assert_eq!(reviews.len(), 2);
        assert_eq!(reviews[0]["decision"], "confirm");
        assert_eq!(reviews[1]["decision"], "resolve");
        assert_eq!(reviews[1]["actor"], "operator");
        assert_eq!(reviews[1]["note"], "trained out on 2026-10-20");

        // And /weakness export rebuilds the same thing from the log alone.
        let before = fs::read_to_string(log.q10_export_path()).unwrap();
        fs::remove_file(log.q10_export_path()).unwrap();
        let message = operator(&log, &src, "/weakness export");
        assert!(message.contains("weaknesses-export.jsonl"), "{message}");
        let after = fs::read_to_string(log.q10_export_path()).unwrap();
        let strip = |text: &str| text.split_once('\n').unwrap().1.to_string();
        assert_eq!(strip(&before), strip(&after), "only the timestamp differs");
    }

    #[test]
    fn an_empty_log_exports_a_header_and_nothing_else() {
        // The sleep pass refreshes the export every night, including on a
        // machine where nothing has ever been logged. A header with no chain
        // head is honest about that and still parses as JSONL.
        let log = temp_log("q10-empty");
        log.export().unwrap();
        let lines = read_q10(&log);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["export"]["chain_head"], Value::Null);
        assert_eq!(lines[0]["export"]["seq"], 0);
        assert_eq!(lines[0]["export"]["weaknesses"], 0);
        assert!(fs::read_to_string(log.export_path()).unwrap().contains("\"weaknesses\""));
    }

    #[test]
    fn startup_exports_a_log_that_has_no_export_beside_it() {
        // The first run after this code lands has a log full of weaknesses and
        // no export, because the export is derived and nothing has appended
        // since. A night that discovers that by failing is not good enough, so
        // startup writes it like the sleep pass does.
        let dir = std::env::temp_dir().join(format!("s0ma-weakness-first-run-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let memory = Hippocampus::new(dir.join("hippocampus.jsonl")).unwrap();
        let log = WeaknessLog::beside_archive(memory.archive_path());
        let src = Fake::new();
        execute_with(&log, &src, "qwen38-nvfp4", &report(json!([3, 11]))).unwrap();
        handle_operator_with(&log, &src, "m", "/weakness confirm wk-0001")
            .unwrap()
            .unwrap();

        // As if the log predates the exporter, or someone deleted the files.
        fs::remove_file(log.q10_export_path()).unwrap();
        fs::remove_file(log.export_path()).unwrap();

        refresh_export(&memory);
        assert!(log.q10_export_path().exists(), "the Q10 export is rebuilt from the log alone");
        assert!(log.export_path().exists());
        let lines = read_q10(&log);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["id"], "wk-0001");
        assert_eq!(lines[1]["status"], "confirmed");
        assert_eq!(lines[0]["export"]["chain_head"], log.entries().unwrap().last().unwrap().hash);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn exported_excerpts_are_the_raw_prefix_of_the_event() {
        let log = temp_log("q10-excerpt");
        let src = Fake::new();
        execute_with(&log, &src, "m", &report(json!([3, 11]))).unwrap();

        let exported = q10_by_id(&log);
        let events = exported["wk-0001"]["instances"][0]["events"].clone();

        // A short event is passed through untouched.
        assert_eq!(events[0]["excerpt"], src.content(3));

        // A long one keeps its raw leading characters and nothing else: qref
        // re-reads the event and requires the content to still begin with this
        // string, so the log's "…" marker cannot be left in it.
        let content = src.content(11);
        let excerpt = events[1]["excerpt"].as_str().unwrap();
        assert_eq!(excerpt.chars().count(), EXCERPT_CHARS);
        assert!(!excerpt.ends_with('…'), "the marker is not part of the content");
        assert!(content.starts_with(excerpt), "the archived content must still begin with it");
        assert_eq!(excerpt, content.chars().take(EXCERPT_CHARS).collect::<String>());

        // Byte-identical to the snapshot the log took, marker aside: no
        // trimming, no whitespace normalisation, nothing tidied.
        let entries = log.entries().unwrap();
        let Action::Report { instance, .. } = &entries[0].action else {
            panic!("the first entry is the report")
        };
        for (evidence, line) in instance.evidence.iter().zip(events.as_array().unwrap()) {
            let Evidence::Event { excerpt: stored, content_sha256, .. } = evidence else {
                continue;
            };
            assert_eq!(
                stored.strip_suffix('…').unwrap_or(stored),
                line["excerpt"].as_str().unwrap()
            );
            assert_eq!(content_sha256, line["sha256"].as_str().unwrap());
        }
    }

    #[test]
    fn instance_ids_survive_a_review_between_exports() {
        let log = temp_log("q10-ids");
        let src = Fake::new();
        execute_with(&log, &src, "m", &report(json!([3]))).unwrap();
        for (event, text) in [(7, "the seventh event repeats it"), (8, "and the eighth does too")] {
            let add = call("weakness_add_instance", json!({
                "weakness_id": "wk-0001",
                "justification": format!("{text}: an unsourced claim stated as fact once again"),
                "event_ids": [event]
            }));
            assert_eq!(parse(&execute_with(&log, &src, "m", &add).unwrap())["recorded"], true);
        }

        let keyed = |log: &WeaknessLog| -> BTreeMap<String, Value> {
            q10_by_id(log)["wk-0001"]["instances"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| (i["instance_id"].as_str().unwrap().to_string(), i["justification"].clone()))
                .collect()
        };
        let before = keyed(&log);
        assert_eq!(
            before.keys().collect::<Vec<_>>(),
            ["wk-0001/1", "wk-0001/2", "wk-0001/3"]
        );
        let head_before = read_q10(&log)[0]["export"]["chain_head"].clone();

        // A review, a resolve, and a second weakness reported in between:
        // qref keys the drafted corrections on these IDs, so none of it may
        // renumber what is already out there.
        operator(&log, &src, "/weakness confirm wk-0001");
        execute_with(&log, &src, "m", &report(json!([5]))).unwrap();
        operator(&log, &src, "/weakness resolve wk-0001 trained out");
        assert_eq!(keyed(&log), before);

        let head_after = read_q10(&log)[0]["export"]["chain_head"].clone();
        assert_ne!(head_before, head_after, "the head names the new state of the log");
        assert_eq!(head_after, log.entries().unwrap().last().unwrap().hash);
        assert_eq!(q10_by_id(&log)["wk-0002"]["instances"][0]["instance_id"], "wk-0002/1");
    }

    #[test]
    fn the_export_is_replaced_in_one_step() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let log = temp_log("q10-atomic");
        let src = Fake::new();
        execute_with(&log, &src, "m", &report(json!([3]))).unwrap();

        // q10.sh may read the export while the harness is running, so every
        // read has to see a whole file: a header that parses and exactly as
        // many weakness lines as it claims.
        let path = log.q10_export_path().to_path_buf();
        let stop = Arc::new(AtomicBool::new(false));
        let reader = {
            let (path, stop) = (path.clone(), Arc::clone(&stop));
            std::thread::spawn(move || {
                let mut reads = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let text = match fs::read_to_string(&path) {
                        Ok(text) => text,
                        Err(error) => panic!("the export vanished mid-write: {error}"),
                    };
                    let mut lines = text.lines();
                    let header: Value =
                        serde_json::from_str(lines.next().expect("a header")).expect("a torn header");
                    let rest: Vec<&str> = lines.collect();
                    assert_eq!(
                        header["export"]["weaknesses"].as_u64().unwrap() as usize,
                        rest.len(),
                        "a half-written export was readable"
                    );
                    for line in rest {
                        serde_json::from_str::<Value>(line).expect("a torn weakness line");
                    }
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..40 {
            execute_with(&log, &src, "m", &report(json!([3]))).unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().unwrap() > 0, "the reader never read anything");

        // And nothing is left lying beside it.
        let leftovers: Vec<String> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    // ------------------------------------------------------------- resolve

    #[test]
    fn resolve_keeps_it_on_record_and_reopen_undoes_it() {
        let log = temp_log("resolve");
        let src = Fake::new();
        execute_with(&log, &src, "m", &report(json!([3]))).unwrap();
        execute_with(&log, &src, "m", &report(json!([5]))).unwrap();

        // The ratchet turns one way: confirmed first.
        let error = handle_operator_with(&log, &src, "m", "/weakness resolve wk-0001")
            .unwrap()
            .unwrap_err();
        assert!(error.contains("confirm it first"), "{error}");

        operator(&log, &src, "/weakness confirm wk-0001");
        operator(&log, &src, "/weakness confirm wk-0002");

        let message = operator(&log, &src, "/weakness resolve wk-0001 trained out in Q10");
        assert!(message.contains("resolve recorded"), "{message}");
        assert!(message.contains("regression"), "{message}");

        // Still in the log, in both lists, and in the export.
        assert_eq!(log.weaknesses().unwrap()["wk-0001"].status, Status::Resolved);
        let listed = parse(
            &execute_with(&log, &src, "m", &call("weakness_list", json!({"status": "resolved"}))).unwrap(),
        );
        assert_eq!(listed["count"], 1);
        assert_eq!(listed["weaknesses"][0]["id"], "wk-0001");
        let open = parse(&execute_with(&log, &src, "m", &call("weakness_list", json!({}))).unwrap());
        assert_eq!(open["count"], 1, "resolved is not open");
        assert_eq!(open["weaknesses"][0]["id"], "wk-0002");
        let rows = operator(&log, &src, "/weakness list resolved");
        assert!(rows.contains("wk-0001") && !rows.contains("wk-0002"), "{rows}");
        assert_eq!(q10_by_id(&log)["wk-0001"]["status"], "resolved");

        // A recurrence is news, not a rejection: the instance is recorded and
        // the status is handed back so s0ma can say the fix stopped holding.
        let add = call("weakness_add_instance", json!({
            "weakness_id": "wk-0001",
            "justification": "Event 6 does it again after the fix: the same unsourced claim as fact.",
            "event_ids": [6]
        }));
        let out = parse(&execute_with(&log, &src, "m", &add).unwrap());
        assert_eq!(out["recorded"], true, "{out}");
        assert_eq!(out["status"], "resolved");

        // Reopen is the undo: back to confirmed, not back to an unreviewed claim.
        operator(&log, &src, "/weakness reopen wk-0001 the fix did not hold");
        assert_eq!(log.weaknesses().unwrap()["wk-0001"].status, Status::Confirmed);

        // And reopen still means "back to proposed" everywhere else.
        operator(&log, &src, "/weakness reject wk-0002 not a real one");
        operator(&log, &src, "/weakness reopen wk-0002 on reflection");
        assert_eq!(log.weaknesses().unwrap()["wk-0002"].status, Status::Proposed);

        assert!(HELP.contains("/weakness resolve"));
    }

    #[test]
    fn an_instance_citing_only_the_complaint_is_flagged() {
        let log = temp_log("own-turn");
        let src = Fake::new();

        // Event 3 is the operator's; 11 is s0ma's own turn. qref renders the
        // lesson from the last cited turn of s0ma's own, so without one there
        // is no mistake to correct.
        let out = parse(&execute_with(&log, &src, "m", &report(json!([3]))).unwrap());
        assert_eq!(out["recorded"], true, "the record is still kept");
        assert!(
            out["evidence_note"].as_str().unwrap().contains("weakness_add_instance"),
            "{out}"
        );

        let out = parse(&execute_with(&log, &src, "m", &report(json!([3, 11]))).unwrap());
        assert_eq!(out["evidence_note"], Value::Null, "{out}");

        let add = |ids: Value| call("weakness_add_instance", json!({
            "weakness_id": "wk-0002",
            "justification": "The same move again, stated as fact with nothing behind it at all.",
            "event_ids": ids
        }));
        let out = parse(&execute_with(&log, &src, "m", &add(json!([4]))).unwrap());
        assert!(out["evidence_note"].is_string(), "{out}");
        let out = parse(&execute_with(&log, &src, "m", &add(json!([11]))).unwrap());
        assert_eq!(out["evidence_note"], Value::Null, "{out}");

        // The tools say so up front, too.
        let tools = tool_definitions();
        for name in ["weakness_report", "weakness_add_instance"] {
            let tool = tools.iter().find(|t| t.function.name == name).unwrap();
            assert!(tool.function.description.contains("your own turn"), "{name}");
        }
    }
}
