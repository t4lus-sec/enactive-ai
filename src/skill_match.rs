//! Per-turn skill match, stage 1: shortlist the skills whose name and
//! description sit closest to the operator's message in embedding space.
//!
//! Stage 2 is in knowledge.rs: the knowledge pass asks the base model one noul
//! per shortlisted skill ("does this procedure apply?"), alongside its source
//! nouls, and only the skills it confirms reach the agent, as pointers to read with
//! skill_read. Skill text itself is never injected.
//!
//! - Every skill in the catalog is considered, not only the 20 in the startup
//!   catalog. Each is embedded once as "name: description" with the document
//!   prefix and cached by content and embedding fingerprint, so a new or edited
//!   skill is picked up on the next turn, and a deleted one is gone from it
//!   (skill_delete also drops its vectors, `forget`).
//! - The cache is saved to `<data>/skill_vectors.json`, so a restart embeds
//!   only skills that are new or changed since the last run, or everything once
//!   after the embedding model, endpoint or prefixes change (the fingerprint is
//!   part of every key). `warm_up` brings it up to date at startup and drops
//!   vectors no current skill uses, so the file stays the catalog's size.
//! - The query is the operator's message (its first 1800 bytes), with the query
//!   prefix, on the same local embedding server as archive search.
//! - Skills already read in the current context (a skill_read call), or already
//!   pointed to by a knowledge-check note still in context, are skipped, so a
//!   pointer isn't repeated every turn. After a fold drops them, they can match
//!   again.
//! - No deadline: the turn waits for the shortlist. An error (embedding server
//!   down) leaves the turn without skill matches and is reported in the
//!   knowledge status line.
//!
//! Environment:
//! - `S0MA_SKILL_MATCH=0` — off.
//! - `S0MA_SKILL_MATCH_CANDIDATES` — shortlist size, 1 to 10 (default 3).
//! - `S0MA_KNOW_THRESHOLD_SKILL_MATCH` — stage 2's bar (knowledge.rs).
use crate::{
    archive::{truncate_for_embedding, Embedder},
    types::Message,
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::Instant,
};

/// How knowledge.rs renders a confirmed skill in its note; also how earlier
/// pointers are recognised in context.
pub const POINTER_PREFIX: &str = "- skill `";
/// The knowledge-check note that carries pointers.
const NOTE_MARKER: &str = "[HARNESS KNOWLEDGE CHECK]";
const DEFAULT_CANDIDATES: usize = 3;
const MAX_CANDIDATES: usize = 10;

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub name: String,
    pub description: String,
    pub cosine: f32,
}

#[derive(Debug, PartialEq)]
struct Settings {
    enabled: bool,
    candidates: usize,
}

impl Settings {
    fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let get = |key: &str| get(key).map(|v| v.trim().to_owned()).filter(|v| !v.is_empty());
        let enabled = !matches!(
            get("S0MA_SKILL_MATCH").map(|v| v.to_ascii_lowercase()).as_deref(),
            Some("0" | "off" | "false" | "no")
        );
        let candidates = match get("S0MA_SKILL_MATCH_CANDIDATES") {
            Some(raw) => raw
                .parse::<usize>()
                .ok()
                .filter(|n| (1..=MAX_CANDIDATES).contains(n))
                .ok_or_else(|| format!("S0MA_SKILL_MATCH_CANDIDATES must be a whole number from 1 to {MAX_CANDIDATES}"))?,
            None => DEFAULT_CANDIDATES,
        };
        Ok(Self { enabled, candidates })
    }

    /// As set, and off while embeddings are off: the shortlist is by
    /// embedding.
    fn from_env() -> Result<Self, String> {
        let mut settings = Self::from_vars(|key| std::env::var(key).ok())?;
        settings.enabled &= crate::model_config::embeddings_on();
        Ok(settings)
    }
}

/// Cache key (embedding fingerprint and text) → the skill it belongs to (for
/// `forget`) and its vector.
type VectorCache = HashMap<String, (String, Vec<f32>)>;

/// The file format version; anything else is ignored and rebuilt.
const STORE_VERSION: u32 = 1;

/// Embedded skill descriptions, in memory and, once `init` has named a file,
/// on disk.
#[derive(Default)]
struct VectorStore {
    vectors: VectorCache,
    path: Option<PathBuf>,
}

#[derive(Serialize, Deserialize)]
struct SavedStore {
    version: u32,
    vectors: Vec<SavedVector>,
}

#[derive(Serialize, Deserialize)]
struct SavedVector {
    key: String,
    skill: String,
    /// Little-endian f32s, base64: a quarter the size of JSON numbers, and
    /// exact.
    vector: String,
}

fn encode_vector(vector: &[f32]) -> String {
    let bytes: Vec<u8> = vector.iter().flat_map(|value| value.to_le_bytes()).collect();
    STANDARD.encode(bytes)
}

fn decode_vector(text: &str) -> Option<Vec<f32>> {
    let bytes = STANDARD.decode(text).ok()?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return None;
    }
    let vector: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    vector.iter().all(|value| value.is_finite()).then_some(vector)
}

impl VectorStore {
    /// The store saved at `path`. A missing file is an empty store; an
    /// unreadable or invalid one is reported and rebuilt, never trusted in
    /// part: an entry that fails its checks is dropped.
    fn load(path: PathBuf) -> Self {
        let mut store = Self { vectors: VectorCache::new(), path: Some(path.clone()) };
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return store,
            Err(error) => {
                crate::terminal::warn_status(&format!("[skills] {} could not be read ({error}); skills will be embedded again", path.display()));
                return store;
            }
        };
        let saved: SavedStore = match serde_json::from_str(&text) {
            Ok(saved) => saved,
            Err(error) => {
                crate::terminal::warn_status(&format!("[skills] {} is not valid ({error}); skills will be embedded again", path.display()));
                return store;
            }
        };
        if saved.version != STORE_VERSION {
            crate::terminal::warn_status(&format!("[skills] {} is format version {}, not {STORE_VERSION}; skills will be embedded again", path.display(), saved.version));
            return store;
        }
        let total = saved.vectors.len();
        for entry in saved.vectors {
            let key_ok = entry.key.len() == 64 && entry.key.bytes().all(|b| b.is_ascii_hexdigit());
            if let (true, Some(vector)) = (key_ok, decode_vector(&entry.vector)) {
                store.vectors.insert(entry.key, (entry.skill, vector));
            }
        }
        if store.vectors.len() < total {
            crate::terminal::warn(&format!(
                "[skills] {} had {} damaged entr(ies); those skills will be embedded again",
                path.display(),
                total - store.vectors.len()
            ));
        }
        store
    }

    /// Write the store to its file, replacing the old one atomically. A
    /// failure is reported and costs only re-embedding on the next start.
    fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let mut vectors: Vec<SavedVector> = self
            .vectors
            .iter()
            .map(|(key, (skill, vector))| SavedVector { key: key.clone(), skill: skill.clone(), vector: encode_vector(vector) })
            .collect();
        vectors.sort_by(|a, b| a.skill.cmp(&b.skill).then_with(|| a.key.cmp(&b.key)));
        let result = (|| -> std::io::Result<()> {
            crate::util::write_atomic(path, serde_json::to_vec(&SavedStore { version: STORE_VERSION, vectors })?)
        })();
        if let Err(error) = result {
            crate::terminal::warn_status(&format!("[skills] saving {} failed ({error}); skills will be embedded again next start", path.display()));
        }
    }

    /// Vectors for `entries`, embedding only those not stored yet, and saving
    /// what was added (even if a later one failed).
    fn vectors(
        &mut self,
        fingerprint: &str,
        entries: &[(String, String)],
        mut embed: impl FnMut(&str) -> Result<Vec<f32>, String>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut added = false;
        let result = entries
            .iter()
            .map(|(name, description)| {
                let text = document_text(name, description);
                let key = cache_key(fingerprint, &text);
                if let Some((_, vector)) = self.vectors.get(&key) {
                    return Ok(vector.clone());
                }
                let vector = embed(&text)?;
                self.vectors.insert(key, (name.clone(), vector.clone()));
                added = true;
                Ok(vector)
            })
            .collect();
        if added {
            self.save();
        }
        result
    }

    /// Keep only the vectors `entries` use under `fingerprint`: edited
    /// descriptions, deleted skills and other models' vectors go. Returns how
    /// many went.
    fn prune(&mut self, fingerprint: &str, entries: &[(String, String)]) -> usize {
        let keep: HashSet<String> = entries
            .iter()
            .map(|(name, description)| cache_key(fingerprint, &document_text(name, description)))
            .collect();
        let before = self.vectors.len();
        self.vectors.retain(|key, _| keep.contains(key));
        let dropped = before - self.vectors.len();
        if dropped > 0 {
            self.save();
        }
        dropped
    }

    fn forget(&mut self, name: &str) {
        let before = self.vectors.len();
        self.vectors.retain(|_, (skill, _)| skill != name);
        if self.vectors.len() < before {
            self.save();
        }
    }
}

static VECTORS: OnceLock<Mutex<VectorStore>> = OnceLock::new();

fn store() -> Result<std::sync::MutexGuard<'static, VectorStore>, String> {
    VECTORS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "skill vector cache lock poisoned".to_owned())
}

/// Load the saved vectors from `<data_dir>/skill_vectors.json`, and save there
/// from now on. Called once at startup, before `warm_up`.
pub fn init(data_dir: &Path) {
    let loaded = VectorStore::load(data_dir.join("skill_vectors.json"));
    if VECTORS.set(Mutex::new(loaded)).is_err() {
        crate::terminal::warn_status("[skills] vector cache was already in use; not loading the saved one");
    }
}

/// Drops every cached vector of skill `name`; skill_delete calls this. The
/// shortlist already considers only skills in the catalog, so this frees the
/// memory and makes a later skill of the same name start fresh.
pub fn forget(name: &str) {
    if let Ok(mut store) = store() {
        store.forget(name);
    }
}

fn document_text(name: &str, description: &str) -> String {
    truncate_for_embedding(&format!("{name}: {description}")).to_owned()
}

fn cache_key(fingerprint: &str, text: &str) -> String {
    format!("{:x}", Sha256::digest(format!("{fingerprint}\n{text}").as_bytes()))
}

/// Vectors for `entries`, embedding only those not cached yet. The cache stays
/// locked meanwhile, so a startup warm-up and a turn never embed the same text
/// twice; a turn that arrives during the warm-up waits for it.
fn document_vectors(
    fingerprint: &str,
    entries: &[(String, String)],
    embed: impl FnMut(&str) -> Result<Vec<f32>, String>,
) -> Result<Vec<Vec<f32>>, String> {
    store()?.vectors(fingerprint, entries, embed)
}

/// Skills already in the agent's current context: read with skill_read, or pointed
/// to by an earlier knowledge-check note.
pub fn already_in_context(messages: &[Message]) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    for message in messages {
        for call in message.tool_calls.iter().flatten() {
            if call.function.name == "skill_read"
                && let Some(name) = serde_json::from_str::<Value>(&call.function.arguments)
                    .ok()
                    .and_then(|args| args["name"].as_str().map(str::to_owned))
                {
                    seen.insert(name);
                }
        }
        if message.role == "user" && message.content.contains(NOTE_MARKER) {
            for line in message.content.lines() {
                if let Some((name, _)) = line.strip_prefix(POINTER_PREFIX).and_then(|rest| rest.split_once('`')) {
                    seen.insert(name.to_owned());
                }
            }
        }
    }
    seen
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The `k` closest skills to `query` (both sides unit vectors), best first;
/// ties go to the name that sorts first.
fn rank(entries: &[(String, String)], vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<Candidate> {
    let mut scored: Vec<Candidate> = entries
        .iter()
        .zip(vectors)
        .filter(|(_, vector)| vector.len() == query.len())
        .map(|((name, description), vector)| Candidate {
            name: name.clone(),
            description: description.clone(),
            cosine: cosine(vector, query),
        })
        .collect();
    scored.sort_by(|a, b| b.cosine.total_cmp(&a.cosine).then_with(|| a.name.cmp(&b.name)));
    scored.truncate(k);
    scored
}

/// Stage 1 for one turn. `Ok(empty)` when off, when there are no skills, or
/// when every skill is already in context.
pub fn shortlist(messages: &[Message], input: &str) -> Result<Vec<Candidate>, String> {
    let settings = Settings::from_env()?;
    if !settings.enabled {
        return Ok(Vec::new());
    }
    let skip = already_in_context(messages);
    let entries: Vec<(String, String)> = crate::skills::catalog_entries()?
        .into_iter()
        .filter(|(name, _)| !skip.contains(name))
        .collect();
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    let embedder = Embedder::new().map_err(|e| format!("embedding setup: {e}"))?;
    let vectors = document_vectors(&embedder.fingerprint(), &entries, |text| {
        embedder.document(text).map_err(|e| format!("embedding a skill: {e}"))
    })?;
    let query = embedder
        .query(truncate_for_embedding(input))
        .map_err(|e| format!("embedding the message: {e}"))?;
    Ok(rank(&entries, &vectors, &query, settings.candidates))
}

/// `s0ma --skill-cutoff`: which cosine cutoff, for naming shortlisted skills
/// without asking the model, agrees best with the model's own answers. The
/// pairs are the judged skills in the knowledge audit (`audit`): their cosine,
/// and whether the judge's P(true) was above `bar`.
pub fn cutoff_report(audit: &Path, bar: f64) -> Result<String, String> {
    let text = std::fs::read_to_string(audit).map_err(|e| format!("{}: {e}", audit.display()))?;
    let pairs: Vec<(f64, bool)> = text
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .flat_map(|record| record.pointer("/skill_match/skills").and_then(Value::as_array).cloned().unwrap_or_default())
        .filter_map(|skill| Some((skill.get("cosine")?.as_f64()?, skill.get("noul")?.as_f64()? > bar)))
        .collect();
    Ok(cutoff_table(&pairs, bar, &audit.display().to_string()))
}

fn cutoff_table(pairs: &[(f64, bool)], bar: f64, from: &str) -> String {
    let named = pairs.iter().filter(|(_, yes)| *yes).count();
    let mut out = format!(
        "{} judged skills in {from}; the judge named {named} (P(true) > {bar:.2}).\n",
        pairs.len()
    );
    if pairs.len() < 20 || named == 0 || named == pairs.len() {
        out.push_str(
            "Too few, or all one way, to set a cutoff from. The judge answers only while S0MA_SKILL_MATCH_JUDGE=1, \
             and a pass is in the audit only when it ran over its budget or something failed.\n",
        );
        return out;
    }
    out.push_str("cutoff  agree  named  of the judge's  of these, the judge's\n");
    let mut best: Option<(f64, f64)> = None;
    for step in 0..=30 {
        let cutoff = 0.20 + 0.02 * step as f64;
        let by_cosine = |cosine: f64| cosine >= cutoff;
        let agree = pairs.iter().filter(|(cosine, yes)| by_cosine(*cosine) == *yes).count();
        let both = pairs.iter().filter(|(cosine, yes)| by_cosine(*cosine) && *yes).count();
        let ours = pairs.iter().filter(|(cosine, _)| by_cosine(*cosine)).count();
        let recall = both as f64 / named as f64;
        let precision = if ours == 0 { 0.0 } else { both as f64 / ours as f64 };
        let f1 = if both == 0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
        if best.is_none_or(|(_, score)| f1 > score) {
            best = Some((cutoff, f1));
        }
        out.push_str(&format!(
            "  {cutoff:.2}  {:>5.0}%  {ours:>5}  {:>14.0}%  {:>21.0}%\n",
            100.0 * agree as f64 / pairs.len() as f64,
            100.0 * recall,
            100.0 * precision
        ));
    }
    if let Some((cutoff, f1)) = best {
        out.push_str(&format!(
            "Best match to the judge (F1 {f1:.2}): S0MA_SKILL_MATCH_MIN_COSINE={cutoff:.2}\n"
        ));
    }
    out
}

/// At startup: embed every skill now, so the first turn doesn't pay for it.
/// A failure is printed and the turn retries.
pub fn warm_up() {
    match Settings::from_env() {
        Ok(settings) if !settings.enabled => {
            crate::terminal::say_note(if crate::model_config::embeddings_on() {
                "[skills] match off (S0MA_SKILL_MATCH=0)"
            } else {
                "[skills] match off: embeddings are off"
            });
            return;
        }
        Err(error) => {
            crate::terminal::warn_status(&format!("[skills] match misconfigured: {error}"));
            return;
        }
        Ok(_) => {}
    }
    let started = Instant::now();
    let result = (|| -> Result<(usize, usize, usize), String> {
        let entries = crate::skills::catalog_entries()?;
        let embedder = Embedder::new().map_err(|e| format!("embedding setup: {e}"))?;
        let fingerprint = embedder.fingerprint();
        let mut store = store()?;
        let mut embedded = 0;
        store.vectors(&fingerprint, &entries, |text| {
            embedded += 1;
            embedder.document(text).map_err(|e| e.to_string())
        })?;
        let dropped = store.prune(&fingerprint, &entries);
        Ok((entries.len(), embedded, dropped))
    })();
    match result {
        Ok((count, embedded, dropped)) => crate::terminal::say_note(&format!(
            "[skills] match index: {count} skills ready in {} ms ({} saved, {embedded} embedded now{})",
            started.elapsed().as_millis(),
            count - embedded,
            if dropped > 0 { format!(", {dropped} stale dropped") } else { String::new() }
        )),
        Err(error) => crate::terminal::warn_status(&format!("[skills] match index not built at startup ({error}); turns will retry")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    fn entries(names: &[&str]) -> Vec<(String, String)> {
        names.iter().map(|n| (n.to_string(), format!("how to {n}"))).collect()
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn rank_keeps_the_closest_k_best_first() {
        let skills = entries(&["nmap-triage", "web-recon", "backup-rotation", "report-writing"]);
        let vectors = vec![unit(&[1.0, 0.1, 0.0]), unit(&[0.7, 0.7, 0.0]), unit(&[0.0, 0.0, 1.0]), unit(&[0.0, 1.0, 0.1])];
        let query = unit(&[1.0, 0.2, 0.0]);
        let ranked = rank(&skills, &vectors, &query, 2);
        let names: Vec<&str> = ranked.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["nmap-triage", "web-recon"]);
        assert!(ranked[0].cosine > ranked[1].cosine && ranked[0].cosine <= 1.0);
        assert_eq!(ranked[0].description, "how to nmap-triage");
        // Fewer skills than k: all of them. A vector of the wrong size is left out.
        assert_eq!(rank(&skills[..2], &vectors[..2], &query, 3).len(), 2);
        assert!(rank(&skills[..1], &[vec![1.0, 0.0]], &query, 3).is_empty());
        // Equal scores: by name.
        let tied = rank(&entries(&["b-skill", "a-skill"]), &[query.clone(), query.clone()], &query, 2);
        assert_eq!(tied[0].name, "a-skill");
    }

    fn temp_store(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("s0ma-skill-vectors-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("skill_vectors.json")
    }

    #[test]
    fn saved_vectors_survive_a_restart_exactly() {
        let path = temp_store("restart");
        let skills = entries(&["nmap-triage", "web-recon"]);
        let vector = |text: &str| vec![text.len() as f32, 0.1, -3.25e-7, f32::MIN_POSITIVE];
        let mut before = VectorStore::load(path.clone());
        let first = before.vectors("fp", &skills, |text| Ok(vector(text))).unwrap();
        assert!(path.exists());

        // A restart: nothing is embedded again, and the vectors are bit-identical.
        let mut after = VectorStore::load(path.clone());
        let again = after.vectors("fp", &skills, |_| panic!("saved vectors must not be re-embedded")).unwrap();
        assert_eq!(first, again);

        // A new skill is embedded and saved alongside; the old ones aren't touched.
        let more = entries(&["nmap-triage", "web-recon", "report-writing"]);
        let mut embedded = Vec::new();
        after.vectors("fp", &more, |text| {
            embedded.push(text.to_owned());
            Ok(vector(text))
        })
        .unwrap();
        assert_eq!(embedded, ["report-writing: how to report-writing"]);
        assert_eq!(VectorStore::load(path).vectors.len(), 3);
    }

    #[test]
    fn edits_model_changes_and_deletions_never_serve_a_stale_vector() {
        let path = temp_store("stale");
        let skills = entries(&["nmap-triage", "web-recon"]);
        let mut store = VectorStore::load(path.clone());
        store.vectors("fp-old", &skills, |_| Ok(vec![1.0, 0.0])).unwrap();

        let mut store = VectorStore::load(path.clone());
        let mut embedded = Vec::new();
        // Another embedding model: every skill again.
        store.vectors("fp-new", &skills, |text| {
            embedded.push(format!("new {text}"));
            Ok(vec![0.0, 1.0])
        })
        .unwrap();
        // An edited description: that skill again.
        let edited = vec![("nmap-triage".to_string(), "Triage nmap XML output".to_string()), skills[1].clone()];
        store.vectors("fp-new", &edited, |text| {
            embedded.push(format!("edit {text}"));
            Ok(vec![0.5, 0.5])
        })
        .unwrap();
        assert_eq!(embedded, [
            "new nmap-triage: how to nmap-triage",
            "new web-recon: how to web-recon",
            "edit nmap-triage: Triage nmap XML output",
        ]);

        // Pruning to the current catalog drops the old model's and the old description's vectors.
        assert_eq!(store.prune("fp-new", &edited), 3);
        assert_eq!(VectorStore::load(path.clone()).vectors.len(), 2);

        // A deleted skill's vectors leave the file too.
        store.forget("web-recon");
        let reloaded = VectorStore::load(path);
        assert_eq!(reloaded.vectors.values().map(|(skill, _)| skill.as_str()).collect::<Vec<_>>(), ["nmap-triage"]);
    }
}
