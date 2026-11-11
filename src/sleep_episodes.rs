//! Durable structured episodes. Only sleep integration can stage them; no runtime recall.
//!
//! Integration works on a set. When a pass opens, the set holds every episode
//! proposal its batches made (handles `b<batch>.<n>`) and every episode earlier
//! passes kept, and each stays as it is unless the model changes it. Changes
//! come as small operations (edit some fields, replace several with one, add,
//! drop), each checked and applied on its own: a bad one is named in the result
//! with its problem and changes nothing, and the others stand. So the model
//! writes only what it changes, and a mistake costs one operation, not the set.
use super::*;
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Episode {
    pub title: String,
    pub situation: String,
    pub cues: Vec<String>,
    pub uncertainty: String,
    pub outcome: Outcome,
    pub resolution: String,
    pub lesson: Option<String>,
    pub applicability: String,
    pub limitations: String,
    pub source_event_ids: Vec<u64>,
    #[serde(default)]
    pub supersedes: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Outcome {
    Resolved,
    Unresolved,
    FalseLead,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn episode_digest(episode: &Episode) -> String {
    digest(&serde_json::to_vec(episode).expect("episode serialization"))
}
pub(super) fn schema() -> serde_json::Value {
    json!({"type":"object","additionalProperties":false,"properties":{
        "title":{"type":"string"},"situation":{"type":"string"},
        "cues":{"type":"array","items":{"type":"string"}},
        "uncertainty":{"type":"string"},
        "outcome":{"type":"string","enum":["resolved","unresolved","false_lead"]},
        "resolution":{"type":"string"},"lesson":{"type":["string","null"]},
        "applicability":{"type":"string"},"limitations":{"type":"string"},
        "source_event_ids":{"type":"array","items":{"type":"integer","minimum":0}},
        "supersedes":{"type":"array","items":{"type":"string"}}
    },"required":["title","situation","cues","uncertainty","outcome","resolution","lesson","applicability","limitations","source_event_ids","supersedes"]})
}

/// The operations episode_consolidate takes.
fn operations_schema() -> serde_json::Value {
    let mut fields = schema();
    fields.as_object_mut().expect("episode schema").remove("required");
    json!({"type":"object","additionalProperties":false,"properties":{
        "operations":{"type":"array","items":{"type":"object","additionalProperties":false,"properties":{
            "episode":schema(),
            "replace":{"type":"array","items":{"type":"string"},
                "description":"With episode: the handles it merges or rewrites; they leave the set."},
            "edit":{"type":"string","description":"With fields: the handle of the episode to change."},
            "fields":fields,
            "drop":{"type":"array","items":{"type":"string"},"description":"With reason: handles to remove."},
            "reason":{"type":"string"}
        }}}
    },"required":["operations"]})
}

const OPERATIONS: &str = "Each operation is one of: {\"edit\": handle, \"fields\": {only the fields that change}}; \
    {\"replace\": [handles], \"episode\": one full episode} to merge or rewrite them as one; {\"episode\": a full \
    episode} to add one the proposals missed; {\"drop\": [handles], \"reason\": why} for duplicates and unsupported \
    ones. Operations are checked and applied one by one, in order. The result lists each by index: a failed one \
    changed nothing and only it needs resending, corrected. Calls add up; nothing is written until integration \
    finishes. Source event IDs must come from this window; supersedes names only an existing durable episode ID \
    (ep_…), never a handle.";

pub(super) fn tool() -> llm::ToolDefinition {
    llm::ToolDefinition::function(
        "episode_consolidate",
        format!(
            "Change this sleep window's set of durable episodes. Every episode proposal in the batch digests \
             (handle b<batch>.<n>) is already in the set and is kept unchanged unless an operation here changes \
             it, so send only changes, never correct proposals again. {OPERATIONS} This is storage only, not \
             recall; after a committed retry the original receipt is retained."
        ),
        operations_schema(),
    )
}

/// The tool in an integration done in passes: the set also holds what earlier
/// passes kept.
pub(super) fn pass_tool() -> llm::ToolDefinition {
    llm::ToolDefinition::function(
        "episode_consolidate",
        format!(
            "Change the set of durable episodes. The set holds every episode proposal in THIS pass's batch \
             digests (handle b<batch>.<n>) and every episode earlier passes kept (handles listed in the prompt); \
             each is kept unchanged unless an operation here changes it, so send only changes, never correct \
             ones again. {OPERATIONS} After the last pass, every pass's set is committed together."
        ),
        operations_schema(),
    )
}

/// Reads one episode of the set, or lists this pass's part of it.
pub(super) fn staged_get_tool() -> llm::ToolDefinition {
    llm::ToolDefinition::function(
        "sleep_staged_get",
        "Read one episode of the current set in full by its handle (a proposal b<batch>.<n>, a staged_N, or an \
         earlier pass's handle), before editing or merging it. Without a handle, list this pass's part of the \
         set: its proposals and the episodes added, edited or dropped so far.",
        json!({"type":"object","additionalProperties":false,"properties":{
            "handle":{"type":"string"}}}),
    )
}

/// What is wrong with `episode`, said so that it can be fixed; None if nothing.
fn problem(episode: &Episode, allowed: Option<&BTreeSet<u64>>) -> Option<String> {
    for (name, value) in [
        ("title", &episode.title),
        ("situation", &episode.situation),
        ("uncertainty", &episode.uncertainty),
        ("resolution", &episode.resolution),
        ("applicability", &episode.applicability),
        ("limitations", &episode.limitations),
    ] {
        if value.trim().is_empty() {
            return Some(format!("{name} is empty"));
        }
        if value.len() > 12_000 {
            return Some(format!("{name} is {} bytes; at most 12000", value.len()));
        }
    }
    if episode.title.len() > 240 {
        return Some(format!("title is {} bytes; at most 240", episode.title.len()));
    }
    if episode.cues.is_empty() {
        return Some("cues is empty; give 1 to 16".into());
    }
    if episode.cues.len() > 16 {
        return Some(format!("cues has {}; at most 16", episode.cues.len()));
    }
    for (index, cue) in episode.cues.iter().enumerate() {
        if cue.trim().is_empty() {
            return Some(format!("cue {index} is empty"));
        }
        if cue.len() > 1000 {
            return Some(format!("cue {index} is {} bytes; at most 1000", cue.len()));
        }
    }
    match &episode.lesson {
        Some(lesson) if lesson.trim().is_empty() => return Some("lesson is empty; use null for none".into()),
        Some(lesson) if lesson.len() > 12_000 => {
            return Some(format!("lesson is {} bytes; at most 12000", lesson.len()));
        }
        _ => {}
    }
    if episode.source_event_ids.is_empty() {
        return Some("source_event_ids is empty".into());
    }
    let mut unique = BTreeSet::new();
    for id in &episode.source_event_ids {
        if !unique.insert(*id) {
            return Some(format!("source_event_ids repeats #{id}"));
        }
    }
    if let Some(allowed) = allowed {
        let outside: Vec<String> = unique.difference(allowed).map(|id| format!("#{id}")).collect();
        if !outside.is_empty() {
            let shown = outside.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
            let more = if outside.len() > 5 { format!(" and {} more", outside.len() - 5) } else { String::new() };
            return Some(format!("source_event_ids {shown}{more} are not in this sleep window"));
        }
    }
    let mut links = BTreeSet::new();
    for link in &episode.supersedes {
        if !links.insert(link) {
            return Some(format!("supersedes names {link} twice"));
        }
    }
    None
}

pub(super) fn validate(episodes: &[Episode], allowed: Option<&BTreeSet<u64>>) -> Result<()> {
    for (index, episode) in episodes.iter().enumerate() {
        if let Some(problem) = problem(episode, allowed) {
            return Err(format!("episode {index}: {problem}").into());
        }
    }
    Ok(())
}
fn connect(path: &Path) -> Result<Connection> {
    let db = Connection::open(path)?;
    db.busy_timeout(std::time::Duration::from_secs(10))?;
    // These are durable tables, not part of the rebuildable semantic index.
    db.execute_batch(
        "PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS consolidated_episodes (
          id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, created_at TEXT NOT NULL,
          record_json TEXT NOT NULL, source_digests_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS episode_supersessions (
          older_id TEXT NOT NULL REFERENCES consolidated_episodes(id),
          newer_id TEXT NOT NULL REFERENCES consolidated_episodes(id),
          PRIMARY KEY(older_id,newer_id));
        CREATE TABLE IF NOT EXISTS episode_sleep_receipts (
          sleep_key TEXT PRIMARY KEY, created_at TEXT NOT NULL, episode_ids_json TEXT NOT NULL,
          empty_reason TEXT);",
    )?;
    Ok(db)
}

/// What earlier integration passes kept, carried to the next pass and saved
/// with each pass's receipt so a resumed sleep picks up where it stopped.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct Kept {
    /// (handle, episode), in set order.
    pub episodes: Vec<(String, Episode)>,
    /// Why passes that kept nothing of their own kept nothing.
    pub empty_reasons: Vec<String>,
    /// Passes closed so far.
    pub passes: usize,
    next_handle: usize,
}

/// One batch worker's episode proposal.
struct Proposal {
    batch: usize,
    handle: String,
    episode: Episode,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Origin {
    /// Kept by an earlier pass.
    Earlier,
    /// A proposal of this pass's batches.
    Proposal,
    /// Added or merged by this pass.
    New,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    handle: String,
    origin: Origin,
    /// Edited in this pass.
    changed: bool,
    episode: Episode,
}

/// The set while a pass is open. Serialized into the integration journal after
/// every change, so a sleep stopped mid-pass resumes with it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Pool {
    /// The batches whose proposals it opened with: `start..end`.
    batches: (usize, usize),
    entries: Vec<Entry>,
    /// A proposal identical to an earlier entry, and that entry's handle.
    aliases: BTreeMap<String, String>,
    /// "handles: reason", in order.
    drops: Vec<String>,
    next_handle: usize,
}

impl Pool {
    fn position(&self, handle: &str) -> std::result::Result<usize, String> {
        if let Some(index) = self.entries.iter().position(|entry| entry.handle == handle) {
            return Ok(index);
        }
        if let Some(same) = self.aliases.get(handle) {
            return Err(if self.entries.iter().any(|entry| entry.handle == *same) {
                format!("{handle} is identical to {same}, which stands for both: use {same}")
            } else {
                format!("{handle} is identical to {same}, which is no longer in the set")
            });
        }
        if self.drops.iter().any(|drop| drop.split(':').next().is_some_and(|handles| handles.split(", ").any(|h| h == handle))) {
            return Err(format!("{handle} is no longer in the set (dropped earlier)"));
        }
        Err(format!("{handle} is not in the set (gone after a replace, or never there)"))
    }

    /// Positions of `handles`, all in the set and each named once.
    fn positions(&self, handles: &[String]) -> std::result::Result<Vec<usize>, String> {
        let mut seen = BTreeSet::new();
        let mut positions = Vec::new();
        for handle in handles {
            if !seen.insert(handle) {
                return Err(format!("{handle} is named twice"));
            }
            positions.push(self.position(handle)?);
        }
        Ok(positions)
    }

    fn summary(&self) -> serde_json::Value {
        let count = |test: &dyn Fn(&Entry) -> bool| self.entries.iter().filter(|entry| test(entry)).count();
        json!({
            "episodes": self.entries.len(),
            "from_earlier_passes": count(&|entry| entry.origin == Origin::Earlier),
            "proposals_unchanged": count(&|entry| entry.origin == Origin::Proposal && !entry.changed),
            "edited": count(&|entry| entry.changed),
            "added_or_merged": count(&|entry| entry.origin == Origin::New),
            "drops": self.drops.len(),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Operation {
    episode: Option<Episode>,
    replace: Option<Vec<String>>,
    edit: Option<String>,
    fields: Option<serde_json::Map<String, serde_json::Value>>,
    drop: Option<Vec<String>>,
    reason: Option<String>,
}

const SHAPES: &str = "an operation is exactly one of {\"episode\"} (add), {\"replace\", \"episode\"}, \
    {\"edit\", \"fields\"} or {\"drop\", \"reason\"}";

/// The checks one episode must pass to enter the set.
struct Checks<'a> {
    allowed: &'a BTreeSet<u64>,
    path: &'a Path,
    db: Option<Connection>,
}

impl Checks<'_> {
    fn check(&mut self, episode: &Episode) -> std::result::Result<(), String> {
        if let Some(problem) = problem(episode, Some(self.allowed)) {
            return Err(problem);
        }
        for id in &episode.supersedes {
            if !id.starts_with("ep_") {
                return Err(format!(
                    "supersedes names {id}, which is not a durable episode ID; supersedes takes only an existing ep_… ID, never a handle"
                ));
            }
            if self.db.is_none() {
                self.db = Some(connect(self.path).map_err(|e| e.to_string())?);
            }
            let exists: bool = self
                .db
                .as_ref()
                .expect("connected above")
                .query_row("SELECT EXISTS(SELECT 1 FROM consolidated_episodes WHERE id=?1)", [id], |r| r.get(0))
                .map_err(|e| e.to_string())?;
            if !exists {
                return Err(format!("supersedes names {id}, which is not an existing durable episode"));
            }
        }
        Ok(())
    }
}

/// Apply one operation to `pool`; on an error nothing changed.
fn apply(pool: &mut Pool, value: serde_json::Value, checks: &mut Checks) -> std::result::Result<serde_json::Value, String> {
    let operation: Operation = serde_json::from_value(value).map_err(|e| format!("{e}; {SHAPES}"))?;
    match operation {
        // A reason given with a replace or an edit is welcome and ignored.
        Operation { episode: Some(episode), replace, edit: None, fields: None, drop: None, reason: _ } => {
            let replace = replace.unwrap_or_default();
            let mut positions = pool.positions(&replace)?;
            checks.check(&episode)?;
            pool.next_handle += 1;
            let handle = format!("staged_{}", pool.next_handle);
            // The merged episode takes the place of the first it replaces.
            let at = positions.iter().min().copied().unwrap_or(pool.entries.len());
            positions.sort_unstable();
            for position in positions.iter().rev() {
                pool.entries.remove(*position);
            }
            pool.entries.insert(at.min(pool.entries.len()), Entry { handle: handle.clone(), origin: Origin::New, changed: false, episode });
            Ok(if replace.is_empty() { json!({"added":handle}) } else { json!({"handle":handle,"replaced":replace}) })
        }
        Operation { episode: None, replace: None, edit: Some(handle), fields: Some(fields), drop: None, reason: _ } => {
            if fields.is_empty() {
                return Err("fields is empty: name the fields that change".into());
            }
            let position = pool.position(&handle)?;
            let mut record = serde_json::to_value(&pool.entries[position].episode).map_err(|e| e.to_string())?;
            let record_fields = record.as_object_mut().expect("an episode is an object");
            for (name, value) in &fields {
                if !record_fields.contains_key(name) {
                    return Err(format!("fields: {name} is not an episode field"));
                }
                record_fields.insert(name.clone(), value.clone());
            }
            let episode: Episode = serde_json::from_value(record).map_err(|e| format!("fields: {e}"))?;
            checks.check(&episode)?;
            let entry = &mut pool.entries[position];
            entry.episode = episode;
            entry.changed = true;
            Ok(json!({"edited":handle,"fields":fields.keys().collect::<Vec<_>>()}))
        }
        Operation { episode: None, replace: None, edit: None, fields: None, drop: Some(drop), reason: Some(reason) } => {
            if reason.trim().is_empty() {
                return Err("reason is empty".into());
            }
            if drop.is_empty() {
                return Err("drop names no handles".into());
            }
            let mut positions = pool.positions(&drop)?;
            positions.sort_unstable();
            for position in positions.iter().rev() {
                pool.entries.remove(*position);
            }
            let reason: String = reason.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(240).collect();
            pool.drops.push(format!("{}: {reason}", drop.join(", ")));
            Ok(json!({"dropped":drop}))
        }
        other => {
            let given: Vec<&str> = [
                ("episode", other.episode.is_some()),
                ("replace", other.replace.is_some()),
                ("edit", other.edit.is_some()),
                ("fields", other.fields.is_some()),
                ("drop", other.drop.is_some()),
                ("reason", other.reason.is_some()),
            ]
            .into_iter()
            .filter_map(|(key, present)| present.then_some(key))
            .collect();
            Err(format!("{SHAPES}; this one has {}", if given.is_empty() { "no keys".to_owned() } else { given.join(", ") }))
        }
    }
}

/// One catalog line.
fn catalog_line(handle: &str, episode: &Episode) -> String {
    let first = episode.source_event_ids.iter().min().copied().unwrap_or(0);
    let last = episode.source_event_ids.iter().max().copied().unwrap_or(0);
    let outcome = match episode.outcome {
        Outcome::Resolved => "resolved",
        Outcome::Unresolved => "unresolved",
        Outcome::FalseLead => "false_lead",
    };
    let cue: String = episode.cues.first().map(|c| c.chars().take(120).collect()).unwrap_or_default();
    format!("{handle} [{outcome}] {} (events #{first}–#{last}; cue: {cue})", episode.title)
}

/// Most groups `likely_duplicates` lists.
const DUPLICATE_GROUPS: usize = 60;

pub(super) struct Session {
    path: PathBuf,
    key: String,
    sources: BTreeMap<u64, String>,
    proposals: Vec<Proposal>,
    /// The set while a pass is open.
    pool: Option<Pool>,
    committed: Option<Vec<String>>,
    /// What the closed passes kept. A single-pass integration is one pass.
    kept: Kept,
}
impl Session {
    pub fn new(memory: &Hippocampus, input: &SleepInput) -> Result<Self> {
        let sources = input
            .source_events
            .iter()
            .map(|event| {
                Ok((
                    event.id,
                    digest(&serde_json::to_vec(&(
                        &event.timestamp,
                        match event.speaker {
                            Speaker::Operator => "operator",
                            Speaker::S0ma => "s0ma",
                        },
                        &event.content,
                    ))?),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        // Use the existing SQLite archive sidecar, with independent durable tables.
        let path = memory.archive_path().with_extension("semantic.sqlite3");
        // A prompt/schema-plan change may rerun batches, but must not duplicate
        // a previously committed source window. Actual source edits change this key.
        let key = digest(&serde_json::to_vec(&("episode-window-v1", &sources))?);
        Self::open(path, &key, sources)
    }
    fn open(path: PathBuf, key: &str, sources: BTreeMap<u64, String>) -> Result<Self> {
        let db = connect(&path)?;
        let receipt: Option<String> = db
            .query_row(
                "SELECT episode_ids_json FROM episode_sleep_receipts WHERE sleep_key=?1",
                [key],
                |r| r.get(0),
            )
            .optional()?;
        Ok(Self {
            path,
            key: key.into(),
            sources,
            proposals: Vec::new(),
            pool: None,
            committed: receipt.map(|s| serde_json::from_str(&s)).transpose()?,
            kept: Kept::default(),
        })
    }

    /// Register batch `batch`'s (0-based) next proposal; returns its handle,
    /// `b<batch + 1>.<n>`.
    pub fn register_proposal(&mut self, batch: usize, episode: Episode) -> Result<String> {
        validate(
            std::slice::from_ref(&episode),
            Some(&self.sources.keys().copied().collect()),
        )?;
        let n = self.proposals.iter().filter(|proposal| proposal.batch == batch).count() + 1;
        let handle = format!("b{}.{n}", batch + 1);
        self.proposals.push(Proposal { batch, handle: handle.clone(), episode });
        Ok(handle)
    }

    /// Whether this window's episodes were committed, by this session or an
    /// earlier one.
    pub fn committed(&self) -> bool {
        self.committed.is_some()
    }

    /// Open a pass over batches `batches` (0-based): its set is what earlier
    /// passes kept, then those batches' proposals. A proposal identical to an
    /// entry already there is not added twice.
    ///
    /// Opened the same way on a window already committed, so the pass's prompt
    /// (which lists likely duplicates from it) is the same as before the
    /// commit and its journal still matches; operations there change nothing.
    pub fn open_pass(&mut self, batches: std::ops::Range<usize>) {
        let mut seen = BTreeSet::new();
        let mut first: BTreeMap<String, String> = BTreeMap::new();
        let mut entries = Vec::new();
        let mut aliases = BTreeMap::new();
        let earlier = self.kept.episodes.iter().map(|(handle, episode)| (handle, episode, Origin::Earlier));
        let proposals = self
            .proposals
            .iter()
            .filter(|proposal| batches.contains(&proposal.batch))
            .map(|proposal| (&proposal.handle, &proposal.episode, Origin::Proposal));
        for (handle, episode, origin) in earlier.chain(proposals) {
            let digest = episode_digest(episode);
            if seen.insert(digest.clone()) {
                first.insert(digest, handle.clone());
                entries.push(Entry { handle: handle.clone(), origin, changed: false, episode: episode.clone() });
            } else {
                aliases.insert(handle.clone(), first[&digest].clone());
            }
        }
        self.pool = Some(Pool {
            batches: (batches.start, batches.end),
            entries,
            aliases,
            drops: Vec::new(),
            next_handle: self.kept.next_handle,
        });
    }

    /// episode_consolidate: apply each operation on its own and say how each went.
    pub fn operate(&mut self, arguments: &str) -> Result<String> {
        if let Some(ids) = &self.committed {
            return Ok(json!({"ok":true,"already_committed":true,"episode_ids":ids}).to_string());
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Call {
            operations: Vec<serde_json::Value>,
        }
        let call: Call = serde_json::from_str(arguments)
            .map_err(|e| format!("the arguments must be {{\"operations\": [...]}}: {e}"))?;
        let pool = self.pool.as_mut().ok_or("no integration pass is open")?;
        let allowed: BTreeSet<u64> = self.sources.keys().copied().collect();
        let mut checks = Checks { allowed: &allowed, path: &self.path, db: None };
        let mut results = Vec::new();
        let mut failed = 0;
        for (index, value) in call.operations.into_iter().enumerate() {
            match apply(pool, value, &mut checks) {
                Ok(mut result) => {
                    result["index"] = index.into();
                    result["ok"] = true.into();
                    results.push(result);
                }
                Err(error) => {
                    failed += 1;
                    results.push(json!({"index":index,"ok":false,"error":error}));
                }
            }
        }
        let note = if failed > 0 {
            "The operations marked ok are applied. Resend only the failed ones, corrected."
        } else if results.is_empty() {
            "No operations: the set is unchanged."
        } else {
            "Applied. Proposals no operation names stay as they are; nothing is written until integration finishes."
        };
        Ok(json!({"ok":failed == 0,"applied":results.len() - failed,"failed":failed,"results":results,
            "set":pool.summary(),"note":note})
        .to_string())
    }

    /// The open pass's set, for the integration journal.
    pub fn pass_state(&self) -> Option<String> {
        self.pool.as_ref().map(|pool| serde_json::to_string(pool).expect("pool serialization"))
    }

    /// Resume the open pass with the set its journal saved.
    pub fn resume_pass(&mut self, state: &str) -> Result<()> {
        if self.committed.is_some() {
            return Ok(());
        }
        let saved: Pool = serde_json::from_str(state)?;
        match &self.pool {
            Some(open) if open.batches == saved.batches => {
                self.pool = Some(saved);
                Ok(())
            }
            Some(_) => Err("the journaled episode set belongs to another integration pass".into()),
            None => Err("no integration pass is open".into()),
        }
    }

    /// The episode IDs this session committed, for the sleep manifest. Empty
    /// until `commit` has run.
    pub fn committed_ids(&self) -> Vec<String> {
        self.committed.clone().unwrap_or_default()
    }

    /// End the open pass: its set is what is kept now. Returns that, for the
    /// pass's receipt. Nothing is written.
    pub fn close_pass(&mut self) -> Result<Kept> {
        if self.committed.is_some() {
            self.pool = None;
            self.kept.passes += 1;
            return Ok(self.kept.clone());
        }
        let pool = self.pool.take().ok_or("no integration pass is open")?;
        if !pool.entries.iter().any(|entry| entry.origin != Origin::Earlier) {
            let reason = if pool.drops.is_empty() {
                "no episode proposals".to_owned()
            } else {
                pool.drops.join("; ").chars().take(600).collect()
            };
            self.kept.empty_reasons.push(format!("pass {}: {reason}", self.kept.passes + 1));
        }
        let mut seen = BTreeSet::new();
        self.kept.episodes = pool
            .entries
            .into_iter()
            .filter(|entry| seen.insert(episode_digest(&entry.episode)))
            .map(|entry| (entry.handle, entry.episode))
            .collect();
        self.kept.next_handle = pool.next_handle;
        self.kept.passes += 1;
        Ok(self.kept.clone())
    }

    /// Resume after closed passes: what they kept, from the last one's receipt.
    pub fn restore(&mut self, kept: Kept) {
        self.kept = kept;
        self.pool = None;
    }

    /// One line per episode earlier passes kept, for the next pass's prompt.
    pub fn kept_catalog(&self) -> String {
        self.kept
            .episodes
            .iter()
            .map(|(handle, episode)| catalog_line(handle, episode))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Groups in the open set that likely tell one experience twice: entries
    /// that cite largely the same source events (at least half of the smaller
    /// one's), at least one of each pair from this pass. Found by code, so the
    /// model does not have to compare every pair itself. Empty when none.
    pub fn likely_duplicates(&self) -> String {
        let Some(pool) = &self.pool else {
            return String::new();
        };
        let entries = &pool.entries;
        let mut by_event: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (index, entry) in entries.iter().enumerate() {
            for id in &entry.episode.source_event_ids {
                by_event.entry(*id).or_default().push(index);
            }
        }
        let mut shared: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for list in by_event.values() {
            for (i, &a) in list.iter().enumerate() {
                for &b in &list[i + 1..] {
                    if entries[a].origin != Origin::Earlier || entries[b].origin != Origin::Earlier {
                        *shared.entry((a, b)).or_default() += 1;
                    }
                }
            }
        }
        let mut parent: Vec<usize> = (0..entries.len()).collect();
        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        for (&(a, b), &count) in &shared {
            let smaller = entries[a].episode.source_event_ids.len().min(entries[b].episode.source_event_ids.len());
            if count * 2 >= smaller.max(1) {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for index in 0..entries.len() {
            let r = root(&mut parent, index);
            groups.entry(r).or_default().push(index);
        }
        let mut lines = Vec::new();
        let groups: Vec<Vec<usize>> = groups.into_values().filter(|members| members.len() > 1).collect();
        for members in groups.iter().take(DUPLICATE_GROUPS) {
            let mut cited: BTreeMap<u64, usize> = BTreeMap::new();
            for &member in members {
                for id in &entries[member].episode.source_event_ids {
                    *cited.entry(*id).or_default() += 1;
                }
            }
            let common: Vec<String> = cited.iter().filter(|(_, n)| **n > 1).map(|(id, _)| format!("#{id}")).collect();
            let more = if common.len() > 6 { format!(" and {} more", common.len() - 6) } else { String::new() };
            let handles: Vec<&str> = members.iter().map(|&member| entries[member].handle.as_str()).collect();
            lines.push(format!(
                "- {}: share events {}{more}",
                handles.join(", "),
                common.iter().take(6).cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        if groups.len() > DUPLICATE_GROUPS {
            lines.push(format!("- … and {} more groups", groups.len() - DUPLICATE_GROUPS));
        }
        for (alias, same) in &pool.aliases {
            lines.push(format!("- {alias} is identical to {same}; the set holds it once, as {same}"));
        }
        lines.join("\n")
    }

    /// sleep_staged_get: one episode of the set in full, or this pass's part
    /// of the set.
    pub fn staged_get(&self, arguments: &str) -> Result<String> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Args {
            handle: Option<String>,
        }
        let args: Args = serde_json::from_str(if arguments.trim().is_empty() { "{}" } else { arguments })?;
        let Some(pool) = &self.pool else {
            let handle = args.handle.ok_or("no integration pass is open")?;
            let (_, episode) = self
                .kept
                .episodes
                .iter()
                .find(|(kept, _)| *kept == handle)
                .ok_or_else(|| format!("{handle} is not in the set"))?;
            return Ok(json!({"handle":handle,"episode":episode}).to_string());
        };
        if let Some(handle) = args.handle {
            let entry = &pool.entries[pool.position(&handle)?];
            return Ok(json!({"handle":handle,"origin":entry.origin,"edited":entry.changed,"episode":entry.episode}).to_string());
        }
        let mut lines = Vec::new();
        for entry in &pool.entries {
            if entry.origin == Origin::Earlier && !entry.changed {
                continue;
            }
            let mark = match (entry.origin, entry.changed) {
                (Origin::New, _) => " (added or merged in this pass)",
                (_, true) => " (edited)",
                _ => "",
            };
            lines.push(format!("{}{mark}", catalog_line(&entry.handle, &entry.episode)));
        }
        let unchanged = pool.entries.iter().filter(|entry| entry.origin == Origin::Earlier && !entry.changed).count();
        Ok(json!({"this_pass":lines,"earlier_passes_unchanged":unchanged,"drops":pool.drops,"set":pool.summary()}).to_string())
    }

    pub fn commit(&mut self) -> Result<Vec<String>> {
        if let Some(ids) = &self.committed {
            return Ok(ids.clone());
        }
        if self.pool.is_some() {
            return Err("an integration pass is still open".into());
        }
        if self.kept.passes == 0 {
            return Err("episode consolidation was not staged".into());
        }
        let episodes: Vec<Episode> = self.kept.episodes.iter().map(|(_, episode)| episode.clone()).collect();
        let empty_reason = episodes.is_empty().then(|| {
            if self.kept.empty_reasons.is_empty() {
                "no episode proposals".to_owned()
            } else {
                self.kept.empty_reasons.join("; ")
            }
        });
        let mut db = connect(&self.path)?;
        let tx = db.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT episode_ids_json FROM episode_sleep_receipts WHERE sleep_key=?1",
                [&self.key],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(existing) = existing {
            let ids = serde_json::from_str(&existing)?;
            self.committed = Some(ids);
            return Ok(self.committed.clone().unwrap());
        }
        let mut ids = BTreeSet::new();
        let now = Local::now().to_rfc3339();
        for episode in &episodes {
            let mut episode = episode.clone();
            episode.source_event_ids.sort_unstable();
            episode.supersedes.sort();
            let sources: BTreeMap<_, _> = episode
                .source_event_ids
                .iter()
                .map(|id| (*id, self.sources[id].clone()))
                .collect();
            let record = serde_json::to_string(&episode)?;
            let provenance = serde_json::to_string(&sources)?;
            let id = format!(
                "ep_{}",
                digest(&serde_json::to_vec(&(1, &record, &provenance))?)
            );
            tx.execute(
                "INSERT OR IGNORE INTO consolidated_episodes VALUES (?1,1,?2,?3,?4)",
                params![id, now, record, provenance],
            )?;
            for old in &episode.supersedes {
                if old == &id {
                    return Err("episode cannot supersede itself".into());
                }
                // Existing targets only: corrections form an append-only chronological graph.
                tx.execute(
                    "INSERT OR IGNORE INTO episode_supersessions VALUES (?1,?2)",
                    params![old, id],
                )?;
            }
            ids.insert(id);
        }
        let ids: Vec<_> = ids.into_iter().collect();
        tx.execute(
            "INSERT INTO episode_sleep_receipts VALUES (?1,?2,?3,?4)",
            params![
                self.key,
                now,
                serde_json::to_string(&ids)?,
                empty_reason
            ],
        )?;
        tx.commit()?;
        self.committed = Some(ids.clone());
        crate::terminal::say_note(&format!("[sleep] committed {} durable episodes", ids.len()));
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "s0ma-episodes-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn session(&self, key: &str) -> Session {
            Session::open(
                self.0.join("archive.sqlite3"),
                key,
                BTreeMap::from([(4, "source hash".into()), (5, "other source hash".into()), (6, "third".into())]),
            )
            .unwrap()
        }
        fn count(&self, table: &str) -> i64 {
            connect(&self.0.join("archive.sqlite3"))
                .unwrap()
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
                .unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn episode() -> Episode {
        Episode {
            title: "Empty capture".into(),
            situation: "The capture was empty".into(),
            cues: vec!["No recorded requests".into()],
            uncertainty: "Recorder status unknown".into(),
            outcome: Outcome::Unresolved,
            resolution: "Recorder has not been checked".into(),
            lesson: None,
            applicability: "Capture-based negative claims".into(),
            limitations: "No positive control yet".into(),
            source_event_ids: vec![4],
            supersedes: vec![],
        }
    }
    fn titled(title: &str, sources: &[u64]) -> Episode {
        Episode { title: title.into(), source_event_ids: sources.to_vec(), ..episode() }
    }
    fn ops(operations: serde_json::Value) -> String {
        json!({"operations":operations}).to_string()
    }
    fn run(session: &mut Session, operations: serde_json::Value) -> serde_json::Value {
        serde_json::from_str(&session.operate(&ops(operations)).unwrap()).unwrap()
    }
    fn titles(session: &Session) -> Vec<String> {
        session.pool.as_ref().unwrap().entries.iter().map(|e| format!("{} {}", e.handle, e.episode.title)).collect()
    }
    /// A single-pass window: the proposals, nothing changed, committed.
    fn commit_as_proposed(session: &mut Session, episodes: Vec<Episode>) -> Vec<String> {
        for episode in episodes {
            session.register_proposal(0, episode).unwrap();
        }
        session.open_pass(0..1);
        session.close_pass().unwrap();
        session.commit().unwrap()
    }

    #[test]
    fn proposals_are_kept_unless_changed_and_unmentioned_ones_commit_as_they_are() {
        let f = Fixture::new();
        let mut session = f.session("keep");
        assert_eq!(session.register_proposal(0, titled("first", &[4])).unwrap(), "b1.1");
        assert_eq!(session.register_proposal(0, titled("second", &[5])).unwrap(), "b1.2");
        assert_eq!(session.register_proposal(2, titled("third", &[6])).unwrap(), "b3.1");
        let mut wrong = episode();
        wrong.source_event_ids = vec![999];
        assert!(session.register_proposal(0, wrong).is_err());
        session.open_pass(0..3);
        assert_eq!(titles(&session), ["b1.1 first", "b1.2 second", "b3.1 third"]);
        // No call at all: everything stands.
        session.close_pass().unwrap();
        assert_eq!(session.commit().unwrap().len(), 3);
        assert_eq!(f.count("consolidated_episodes"), 3);
    }

    #[test]
    fn each_operation_stands_or_fails_on_its_own_and_names_its_problem() {
        let f = Fixture::new();
        let mut s = f.session("ops");
        for (title, sources) in [("a", &[4][..]), ("b", &[4, 5]), ("c", &[6]), ("d", &[5])] {
            s.register_proposal(0, titled(title, sources)).unwrap();
        }
        s.open_pass(0..1);
        let mut bad = titled("bad", &[4]);
        bad.cues = (0..17).map(|i| format!("cue {i}")).collect();
        let result = run(&mut s, json!([
            {"edit":"b1.1","fields":{"resolution":"Checked: the recorder was off"}},
            {"replace":["b1.2","b1.4"],"episode":titled("b and d", &[4, 5])},
            {"episode":bad},
            {"drop":["b1.3"],"reason":"unsupported"},
            {"edit":"b1.9","fields":{"title":"x"}},
            {"edit":"b1.1","fields":{"colour":"red"}},
            {"drop":["b1.2"],"reason":"already merged"},
            {"replace":["b1.1"]},
            {"episode":titled("outside", &[4, 999])},
            {"edit":"b1.1","fields":{"source_event_ids":[4, 4]}},
            {"episode":{"title":"incomplete"}},
            {"edit":"b1.1","fields":{"supersedes":["staged_1"]}},
        ]));
        assert_eq!(result["ok"], false);
        assert_eq!(result["applied"], 3);
        assert_eq!(result["failed"], 9);
        let error = |index: usize| result["results"][index]["error"].as_str().unwrap_or("").to_owned();
        assert_eq!(result["results"][0]["ok"], true);
        assert_eq!(result["results"][1]["handle"], "staged_1");
        assert_eq!(error(2), "cues has 17; at most 16");
        assert_eq!(result["results"][3]["dropped"], json!(["b1.3"]));
        assert!(error(4).contains("b1.9 is not in the set"), "{}", error(4));
        assert_eq!(error(5), "fields: colour is not an episode field");
        assert!(error(6).contains("b1.2 is not in the set"), "replaced by op 1: {}", error(6));
        assert!(error(7).starts_with("an operation is exactly one of") && error(7).ends_with("this one has replace"), "{}", error(7));
        assert_eq!(error(8), "source_event_ids #999 are not in this sleep window");
        assert_eq!(error(9), "source_event_ids repeats #4");
        assert!(error(10).contains("missing field"), "{}", error(10));
        assert!(error(11).contains("never a handle"), "{}", error(11));
        assert_eq!(titles(&s), ["b1.1 a", "staged_1 b and d"], "the merged episode takes the first replaced one's place");
        assert_eq!(s.pool.as_ref().unwrap().entries[0].episode.resolution, "Checked: the recorder was off");
        // Only the failed one, corrected, is resent; the set builds on.
        let mut fixed = titled("bad, fixed", &[4]);
        fixed.cues = vec!["one cue".into()];
        let result = run(&mut s, json!([{"episode":fixed}]));
        assert_eq!(result["ok"], true);
        assert_eq!(result["results"][0]["added"], "staged_2");
        assert!(s.operate(r#"{"episodes":[]}"#).is_err(), "the old shape is refused whole, with the right one named");
        let result = run(&mut s, json!([{"edit":"b1.1","fields":{"title":"a, again"},"reason":"clearer"}]));
        assert_eq!(result["ok"], true, "a reason with an edit is fine: {result}");
        s.close_pass().unwrap();
        assert_eq!(s.commit().unwrap().len(), 3);
    }

    #[test]
    fn a_dropped_handle_says_so_and_a_drop_needs_a_reason() {
        let f = Fixture::new();
        let mut s = f.session("drops");
        s.register_proposal(0, titled("a", &[4])).unwrap();
        s.register_proposal(0, titled("b", &[5])).unwrap();
        s.open_pass(0..1);
        let result = run(&mut s, json!([{"drop":["b1.1"],"reason":"  "},{"drop":[],"reason":"x"},{"drop":["b1.1","b1.1"],"reason":"x"},
            {"drop":["b1.1"],"reason":"duplicate of b1.2"},{"edit":"b1.1","fields":{"title":"y"}}]));
        assert_eq!(result["results"][0]["error"], "reason is empty");
        assert_eq!(result["results"][1]["error"], "drop names no handles");
        assert_eq!(result["results"][2]["error"], "b1.1 is named twice");
        assert_eq!(result["results"][3]["ok"], true);
        assert_eq!(result["results"][4]["error"], "b1.1 is no longer in the set (dropped earlier)");
        assert_eq!(titles(&s), ["b1.2 b"]);
    }

    #[test]
    fn identical_proposals_enter_once_and_the_second_handle_points_at_the_first() {
        let f = Fixture::new();
        let mut s = f.session("same");
        s.register_proposal(0, episode()).unwrap();
        s.register_proposal(1, episode()).unwrap();
        s.open_pass(0..2);
        assert_eq!(titles(&s), ["b1.1 Empty capture"]);
        assert!(s.likely_duplicates().contains("b2.1 is identical to b1.1"));
        let result = run(&mut s, json!([{"edit":"b2.1","fields":{"title":"x"}}]));
        assert!(result["results"][0]["error"].as_str().unwrap().contains("use b1.1"));
        run(&mut s, json!([{"drop":["b1.1"],"reason":"gone"}]));
        let result = run(&mut s, json!([{"edit":"b2.1","fields":{"title":"x"}}]));
        assert!(result["results"][0]["error"].as_str().unwrap().ends_with("which is no longer in the set"), "{result}");
        assert!(s.close_pass().unwrap().episodes.is_empty());
    }

    #[test]
    fn likely_duplicates_group_entries_citing_mostly_the_same_events() {
        let f = Fixture::new();
        let mut s = f.session("dups");
        s.register_proposal(0, titled("first", &[4, 5])).unwrap();
        s.register_proposal(0, titled("unrelated", &[6])).unwrap();
        s.register_proposal(1, titled("retold", &[5])).unwrap();
        s.open_pass(0..2);
        assert_eq!(s.likely_duplicates(), "- b1.1, b2.1: share events #5");
        // Two earlier passes' episodes that overlap were already reconciled:
        // only pairs with one from this pass are flagged.
        let mut s = f.session("dups-earlier");
        s.register_proposal(0, titled("kept one", &[4, 5])).unwrap();
        s.register_proposal(0, titled("kept two", &[5])).unwrap();
        s.open_pass(0..1);
        run(&mut s, json!([]));
        s.close_pass().unwrap();
        s.register_proposal(1, titled("new", &[6])).unwrap();
        s.open_pass(1..2);
        assert_eq!(s.likely_duplicates(), "");
        // One shared event of four on each side is not enough.
        let mut s = f.session("dups-loose");
        s.sources.extend((7..=12).map(|id| (id, format!("h{id}"))));
        s.register_proposal(0, titled("long", &[4, 7, 8, 9])).unwrap();
        s.register_proposal(0, titled("other", &[4, 10, 11, 12])).unwrap();
        s.open_pass(0..1);
        assert_eq!(s.likely_duplicates(), "");
    }

    #[test]
    fn source_window_receipt_survives_replanning_but_not_source_edits() {
        let f = Fixture::new();
        let memory = Hippocampus::new(f.0.join("hippocampus.jsonl")).unwrap();
        let mut input = SleepInput {
            previous_checkpoint: None,
            source_events: vec![RecalledEvent {
                id: 4,
                timestamp: "2026-09-15T00:00:00Z".into(),
                speaker: Speaker::S0ma,
                content: "Empty capture".into(),
            }],
        };
        let mut first = Session::new(&memory, &input).unwrap();
        commit_as_proposed(&mut first, vec![episode()]);
        assert!(Session::new(&memory, &input).unwrap().committed());
        input.source_events[0].content = "Recorder was active".into();
        assert!(!Session::new(&memory, &input).unwrap().committed());
    }
    #[test]
    fn staging_does_not_write_and_drop_discards_unfinished_integration() {
        let f = Fixture::new();
        {
            let mut s = f.session("window");
            s.register_proposal(0, episode()).unwrap();
            s.open_pass(0..1);
            run(&mut s, json!([{"edit":"b1.1","fields":{"title":"changed"}}]));
            assert_eq!(f.count("consolidated_episodes"), 0);
        }
        assert!(!f.session("window").committed());
        assert_eq!(f.count("episode_sleep_receipts"), 0);
    }
    #[test]
    fn committed_retry_is_idempotent_and_preserves_unresolved_evidence() {
        let f = Fixture::new();
        let mut s = f.session("window");
        let ids = commit_as_proposed(&mut s, vec![episode()]);
        assert_eq!(ids.len(), 1);
        assert_eq!(s.commit().unwrap(), ids);
        let mut resumed = f.session("window");
        assert!(resumed.committed());
        resumed.open_pass(0..1);
        assert!(resumed.operate("{}").unwrap().contains("already_committed"));
        resumed.close_pass().unwrap();
        assert_eq!(resumed.commit().unwrap(), ids);
        assert_eq!(f.count("consolidated_episodes"), 1);
        let db = connect(&f.0.join("archive.sqlite3")).unwrap();
        let (record, provenance): (String, String) = db
            .query_row(
                "SELECT record_json,source_digests_json FROM consolidated_episodes",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert!(record.contains("unresolved"));
        assert!(record.contains("\"lesson\":null"));
        assert!(provenance.contains("source hash"));
    }
    #[test]
    fn unknown_corrections_are_refused_and_known_ones_link() {
        let f = Fixture::new();
        let mut first = f.session("first");
        let old = commit_as_proposed(&mut first, vec![episode()])[0].clone();
        let mut second = f.session("second");
        second.register_proposal(0, titled("later", &[5])).unwrap();
        second.open_pass(0..1);
        let result = run(&mut second, json!([
            {"edit":"b1.1","fields":{"supersedes":["ep_nonexistent"]}},
            {"edit":"b1.1","fields":{"outcome":"false_lead","resolution":"Positive control showed recorder was inactive","supersedes":[old.clone()]}},
        ]));
        assert_eq!(result["results"][0]["error"], "supersedes names ep_nonexistent, which is not an existing durable episode");
        assert_eq!(result["results"][1]["ok"], true);
        second.close_pass().unwrap();
        let new = second.commit().unwrap()[0].clone();
        assert_ne!(old, new);
        assert_eq!(f.count("consolidated_episodes"), 2);
        assert_eq!(f.count("episode_supersessions"), 1);
    }
    #[test]
    fn an_empty_window_commits_an_empty_receipt_with_the_reasons() {
        let f = Fixture::new();
        let mut s = f.session("window");
        s.open_pass(0..1);
        s.close_pass().unwrap();
        assert!(s.commit().unwrap().is_empty());
        assert!(f.session("window").committed());
        let reason: String = connect(&f.0.join("archive.sqlite3")).unwrap()
            .query_row("SELECT empty_reason FROM episode_sleep_receipts", [], |r| r.get(0)).unwrap();
        assert_eq!(reason, "pass 1: no episode proposals");
    }
    #[test]
    fn transaction_rolls_back_if_superseded_record_disappears() {
        let f = Fixture::new();
        let mut first = f.session("first");
        let old = commit_as_proposed(&mut first, vec![episode()])[0].clone();
        let mut correction = titled("correction", &[5]);
        correction.supersedes = vec![old.clone()];
        let mut next = f.session("second");
        next.open_pass(0..1);
        assert_eq!(run(&mut next, json!([{"episode":correction}]))["ok"], true);
        next.close_pass().unwrap();
        connect(&f.0.join("archive.sqlite3"))
            .unwrap()
            .execute("DELETE FROM consolidated_episodes WHERE id=?1", [old])
            .unwrap();
        assert!(next.commit().is_err());
        assert_eq!(f.count("consolidated_episodes"), 0);
        assert_eq!(f.count("episode_sleep_receipts"), 1);
    }
    fn handles(kept: &Kept) -> Vec<&str> {
        kept.episodes.iter().map(|(handle, _)| handle.as_str()).collect()
    }
    #[test]
    fn passes_keep_each_others_episodes_merge_across_the_boundary_and_commit_together() {
        let f = Fixture::new();
        let mut s = f.session("passes");
        s.register_proposal(0, titled("first half", &[4])).unwrap();
        s.register_proposal(1, titled("second half", &[5])).unwrap();
        s.register_proposal(1, titled("other", &[6])).unwrap();
        // Pass 1: batch 1, as proposed.
        s.open_pass(0..1);
        let kept = s.close_pass().unwrap();
        assert_eq!(handles(&kept), ["b1.1"]);
        assert_eq!(s.kept_catalog(), "b1.1 [unresolved] first half (events #4–#4; cue: No recorded requests)");
        assert!(s.staged_get(r#"{"handle":"b1.1"}"#).unwrap().contains("Recorder has not been checked"));
        // Pass 2: batch 2's proposals and pass 1's episode, which it merges.
        s.open_pass(1..2);
        assert_eq!(titles(&s), ["b1.1 first half", "b2.1 second half", "b2.2 other"]);
        assert!(s.staged_get(r#"{"handle":"b2.1"}"#).unwrap().contains("\"origin\":\"proposal\""), "a proposal reads in full");
        let mut merged = titled("whole story", &[4, 5]);
        merged.outcome = Outcome::Resolved;
        assert_eq!(run(&mut s, json!([{"replace":["b1.1","b2.1"],"episode":merged}]))["ok"], true);
        let listing = s.staged_get("{}").unwrap();
        assert!(listing.contains("staged_1 [resolved] whole story") && listing.contains("(added or merged in this pass)"), "{listing}");
        assert!(s.commit().is_err(), "nothing commits while a pass is open");
        let kept = s.close_pass().unwrap();
        assert_eq!(handles(&kept), ["staged_1", "b2.2"]);
        // Pass 3 has no proposals of its own.
        s.open_pass(2..3);
        s.close_pass().unwrap();
        assert_eq!(f.count("consolidated_episodes"), 0, "staging across passes writes nothing");
        let ids = s.commit().unwrap();
        assert_eq!(ids.len(), 2, "the merged episode and the other; the halves are gone");
        assert_eq!(f.count("episode_sleep_receipts"), 1, "one receipt for the window");
        assert_eq!(s.commit().unwrap(), ids);
    }
    #[test]
    fn a_resumed_sleep_restores_what_closed_passes_kept_and_a_journaled_set_mid_pass() {
        let f = Fixture::new();
        let setup = |s: &mut Session| {
            s.register_proposal(0, titled("one", &[4])).unwrap();
            s.register_proposal(1, titled("two", &[5])).unwrap();
        };
        let (kept, state) = {
            let mut first = f.session("resume");
            setup(&mut first);
            first.open_pass(0..1);
            run(&mut first, json!([{"episode":titled("added", &[6])}]));
            let kept = first.close_pass().unwrap();
            first.open_pass(1..2);
            run(&mut first, json!([{"edit":"b2.1","fields":{"title":"two, edited"}}]));
            (kept, first.pass_state().unwrap())
        };
        // Saved with the pass's receipt and read back after a restart.
        let saved: Kept = serde_json::from_str(&serde_json::to_string(&kept).unwrap()).unwrap();
        let mut resumed = f.session("resume");
        setup(&mut resumed);
        resumed.restore(saved);
        assert!(resumed.kept_catalog().starts_with("b1.1 "));
        resumed.open_pass(0..1);
        assert!(resumed.resume_pass(&state).is_err(), "another pass's set");
        resumed.open_pass(1..2);
        resumed.resume_pass(&state).unwrap();
        assert_eq!(titles(&resumed), ["b1.1 one", "staged_1 added", "b2.1 two, edited"]);
        let result = run(&mut resumed, json!([{"episode":titled("later", &[5])}]));
        assert_eq!(result["results"][0]["added"], "staged_2", "handles continue, never reused");
        resumed.close_pass().unwrap();
        assert_eq!(resumed.commit().unwrap().len(), 4);
    }
    #[test]
    fn passes_that_keep_nothing_commit_an_empty_receipt_with_their_reasons() {
        let f = Fixture::new();
        let mut s = f.session("quiet");
        s.register_proposal(0, titled("chatter", &[4])).unwrap();
        s.open_pass(0..1);
        run(&mut s, json!([{"drop":["b1.1"],"reason":"setup chatter"}]));
        s.close_pass().unwrap();
        s.open_pass(1..2);
        s.close_pass().unwrap();
        assert!(s.commit().unwrap().is_empty());
        let reason: String = connect(&f.0.join("archive.sqlite3")).unwrap()
            .query_row("SELECT empty_reason FROM episode_sleep_receipts", [], |r| r.get(0)).unwrap();
        assert_eq!(reason, "pass 1: b1.1: setup chatter; pass 2: no episode proposals");
    }
    #[test]
    fn an_identical_episode_from_two_passes_is_kept_once() {
        let f = Fixture::new();
        let mut s = f.session("twice");
        for pass in 0..2 {
            s.open_pass(pass..pass + 1);
            run(&mut s, json!([{"episode":episode()}]));
            let kept = s.close_pass().unwrap();
            assert_eq!(handles(&kept), ["staged_1"], "the second copy is not kept");
        }
        assert_eq!(s.commit().unwrap().len(), 1);
    }
    #[test]
    fn semantic_index_rebuild_does_not_delete_episode_tables() {
        let f = Fixture::new();
        let mut s = f.session("window");
        commit_as_proposed(&mut s, vec![episode()]);
        let db = connect(&f.0.join("archive.sqlite3")).unwrap();
        db.execute_batch("CREATE TABLE events(id INTEGER PRIMARY KEY); CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT); DELETE FROM events; DELETE FROM metadata;").unwrap();
        assert_eq!(f.count("consolidated_episodes"), 1);
        assert!(f.session("window").committed());
    }
}
