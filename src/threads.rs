use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use std::{
    error::Error,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::llm::ToolDefinition;
use crate::types::ToolCall;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThreadState {
    Candidate,
    Active,
    Dormant,
    Transformed,
    Resolved,
    Abandoned,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThreadCategory {
    Relationship,
    Question,
    Project,
    Problem,
    Commitment,
    Curiosity,
    SelfUnderstanding,
    Experiment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thread {
    pub id: u64,

    pub created_at: String,

    pub updated_at: String,

    // Identity
    pub title: String,

    pub description: String,

    pub category: ThreadCategory,

    // Lifecycle
    pub state: ThreadState,

    // Hippocampus references
    #[serde(default)]
    pub event_ids: Vec<u64>,

    // References to canonical understanding, never a duplicate summary.
    #[serde(default)]
    pub world_record_ids: Vec<String>,
    #[serde(default)]
    pub model_ids: Vec<String>,

    // Why this continues
    pub significance: String,

    // Open questions
    #[serde(default)]
    pub questions: Vec<String>,

    // Concrete continuation
    #[serde(default)]
    pub next_actions: Vec<String>,
}

pub struct ThreadStore {
    path: PathBuf,

    threads: Vec<Thread>,

    next_id: u64,

    /// The `REWRITES` count this copy was read at: a redaction rewrites the
    /// file under the running store, which reads it again before it is used.
    seen: u64,
}

/// Bumped by a redaction that rewrote a threads file (`strike_file`).
static REWRITES: AtomicU64 = AtomicU64::new(0);

/// Held across every write of a threads file, and across a store's change
/// from the moment it reads the file again: a redaction cannot land between
/// the store's read and its save and be written over.
static FILE: Mutex<()> = Mutex::new(());

fn read_threads(path: &Path) -> Result<Vec<Thread>, Box<dyn Error>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let contents = fs::read_to_string(path)?;
    if contents.trim().is_empty() {
        return Ok(Vec::new());
    }
    Ok(serde_json::from_str(&contents)?)
}

fn write_threads(path: &Path, threads: &[Thread]) -> Result<(), Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(threads)?;
    bytes.push(b'\n');
    crate::util::write_atomic(path, bytes)?;
    Ok(())
}

/// Each thread's own words (title, description, significance, questions and
/// next actions) given what `strike` returns for them; with `apply`, written.
/// Returns, for each thread that held something, its id and how many
/// occurrences. Its ids and states are not words and are left alone.
pub fn strike_file(
    path: &Path,
    strike: &dyn Fn(&str) -> Option<(String, usize)>,
    apply: bool,
) -> Result<Vec<(u64, usize)>, Box<dyn Error>> {
    let _file = FILE.lock().map_err(|_| "threads lock poisoned")?;
    let mut threads = read_threads(path)?;
    let mut touched = Vec::new();
    for thread in &mut threads {
        let mut count = 0;
        let mut one = |text: &mut String| {
            if let Some((struck, n)) = strike(text) {
                *text = struck;
                count += n;
            }
        };
        one(&mut thread.title);
        one(&mut thread.description);
        one(&mut thread.significance);
        thread.questions.iter_mut().for_each(&mut one);
        thread.next_actions.iter_mut().for_each(&mut one);
        if count > 0 {
            touched.push((thread.id, count));
        }
    }
    if apply && !touched.is_empty() {
        write_threads(path, &threads)?;
        REWRITES.fetch_add(1, Ordering::SeqCst);
    }
    Ok(touched)
}

impl ThreadStore {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, Box<dyn Error>> {
        let path = path.into();
        // Before the file is read: a redaction landing while it is makes
        // this copy older than the count it carries, and it reads again.
        let seen = REWRITES.load(Ordering::SeqCst);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let threads = if path.exists() {
            let mut contents = String::new();

            File::open(&path)?.read_to_string(&mut contents)?;

            if contents.trim().is_empty() {
                Vec::new()
            } else {
                let raw: Value = serde_json::from_str(&contents)?;
                let links = crate::world_model::import_threads(
                    &path.with_file_name("world_model.sqlite3"),
                    &raw,
                )?;
                let migrated = raw.as_array().is_some_and(|items| {
                    items.iter().any(|t| {
                        ["summary", "transformations", "related_threads"]
                            .iter()
                            .any(|k| t.get(k).is_some())
                    })
                });
                if migrated {
                    let backup = path.with_file_name("threads.pre-world-model.json");
                    match std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&backup)
                    {
                        Ok(mut file) => {
                            file.write_all(contents.as_bytes())?;
                            file.sync_all()?;
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                        Err(e) => return Err(e.into()),
                    }
                }
                let mut threads: Vec<Thread> = serde_json::from_value(raw)?;
                for thread in &mut threads {
                    if let Some(ids) = links.get(&thread.id) {
                        for id in ids {
                            if !thread.world_record_ids.contains(id) {
                                thread.world_record_ids.push(id.clone());
                            }
                        }
                    }
                }
                if migrated {
                    let temp = path.with_extension("migration.tmp");
                    let mut file = File::create(&temp)?;
                    serde_json::to_writer_pretty(&mut file, &threads)?;
                    file.sync_all()?;
                    fs::rename(temp, &path)?;
                }
                threads
            }
        } else {
            Vec::new()
        };

        let next_id = threads
            .iter()
            .map(|thread: &Thread| thread.id)
            .max()
            .unwrap_or(0)
            + 1;

        Ok(Self {
            path,
            threads,
            next_id,
            seen,
        })
    }

    /// The threads as the file has them, when a redaction rewrote it since
    /// this copy was read; this copy otherwise.
    fn current(&self) -> std::borrow::Cow<'_, [Thread]> {
        if REWRITES.load(Ordering::SeqCst) != self.seen
            && let Ok(threads) = read_threads(&self.path)
        {
            return std::borrow::Cow::Owned(threads);
        }
        std::borrow::Cow::Borrowed(&self.threads)
    }

    /// Before a change: the file read again if a redaction rewrote it, under
    /// the lock the returned guard holds until the change is saved.
    fn fresh(&mut self) -> Result<std::sync::MutexGuard<'static, ()>, Box<dyn Error>> {
        let guard = FILE.lock().map_err(|_| "threads lock poisoned")?;
        let now = REWRITES.load(Ordering::SeqCst);
        if now != self.seen {
            self.threads = read_threads(&self.path)?;
            self.next_id = self.next_id.max(self.threads.iter().map(|t| t.id).max().unwrap_or(0) + 1);
            self.seen = now;
        }
        Ok(guard)
    }

    pub fn create(
        &mut self,
        title: String,
        description: String,
        category: ThreadCategory,
        state: ThreadState,
        event_ids: Vec<u64>,
    ) -> Result<Thread, Box<dyn Error>> {
        let _file = self.fresh()?;
        let now = Local::now().to_rfc3339();

        let thread = Thread {
            id: self.next_id,

            created_at: now.clone(),

            updated_at: now,

            title,

            description,

            category,

            state,

            event_ids,

            world_record_ids: Vec::new(),
            model_ids: Vec::new(),

            significance: String::new(),

            questions: Vec::new(),

            next_actions: Vec::new(),
        };

        self.next_id += 1;

        self.threads.push(thread.clone());

        self.save()?;

        Ok(thread)
    }

    pub fn list(&self) -> Vec<Thread> {
        self.current().into_owned()
    }

    pub fn revision(&self) -> Result<String, Box<dyn Error>> {
        // Stable FNV-1a over the canonical serialized store is sufficient for
        // invalidating a derived context projection. It is not an integrity or
        // authority mechanism.
        let serialized = serde_json::to_vec(&*self.current())?;
        let mut hash = 0xcbf29ce484222325_u64;

        for byte in serialized {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }

        Ok(format!("{hash:016x}"))
    }

    pub fn active(&self) -> Vec<Thread> {
        self.current()
            .iter()
            .filter(|thread| {
                matches!(
                    thread.state,
                    ThreadState::Candidate | ThreadState::Active | ThreadState::Transformed
                )
            })
            .cloned()
            .collect()
    }

    pub fn get(&self, id: u64) -> Option<Thread> {
        self.current().iter().find(|thread| thread.id == id).cloned()
    }

    pub fn update(
        &mut self,
        id: u64,

        category: Option<ThreadCategory>,

        title: Option<String>,

        description: Option<String>,

        significance: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        let _file = self.fresh()?;
        let thread = self
            .threads
            .iter_mut()
            .find(|thread| thread.id == id)
            .ok_or("thread not found")?;

        if let Some(value) = category {
            thread.category = value;
        }

        if let Some(value) = title {
            thread.title = value;
        }

        if let Some(value) = description {
            thread.description = value;
        }

        if let Some(value) = significance {
            thread.significance = value;
        }

        thread.updated_at = Local::now().to_rfc3339();

        self.save()
    }

    pub fn set_state(&mut self, id: u64, state: ThreadState) -> Result<(), Box<dyn Error>> {
        let _file = self.fresh()?;
        let thread = self
            .threads
            .iter_mut()
            .find(|thread| thread.id == id)
            .ok_or("thread not found")?;

        thread.state = state;

        thread.updated_at = Local::now().to_rfc3339();

        self.save()
    }

    pub fn add_event(&mut self, id: u64, event_id: u64) -> Result<(), Box<dyn Error>> {
        let _file = self.fresh()?;
        let thread = self
            .threads
            .iter_mut()
            .find(|thread| thread.id == id)
            .ok_or("thread not found")?;

        if !thread.event_ids.contains(&event_id) {
            thread.event_ids.push(event_id);
        }

        thread.updated_at = Local::now().to_rfc3339();

        self.save()
    }

    pub fn remove_event(&mut self, id: u64, event_id: u64) -> Result<(), Box<dyn Error>> {
        let _file = self.fresh()?;
        let thread = self
            .threads
            .iter_mut()
            .find(|thread| thread.id == id)
            .ok_or("thread not found")?;

        thread.event_ids.retain(|id| *id != event_id);

        thread.updated_at = Local::now().to_rfc3339();

        self.save()
    }

    pub fn add_question(&mut self, id: u64, question: String) -> Result<(), Box<dyn Error>> {
        let _file = self.fresh()?;
        let thread = self
            .threads
            .iter_mut()
            .find(|thread| thread.id == id)
            .ok_or("thread not found")?;

        if !thread.questions.contains(&question) {
            thread.questions.push(question);
        }

        thread.updated_at = Local::now().to_rfc3339();

        self.save()
    }

    pub fn remove_question(&mut self, id: u64, question: String) -> Result<(), Box<dyn Error>> {
        let _file = self.fresh()?;
        let thread = self
            .threads
            .iter_mut()
            .find(|thread| thread.id == id)
            .ok_or("thread not found")?;

        thread.questions.retain(|q| q != &question);

        thread.updated_at = Local::now().to_rfc3339();

        self.save()
    }

    pub fn add_action(&mut self, id: u64, action: String) -> Result<(), Box<dyn Error>> {
        let _file = self.fresh()?;
        let thread = self
            .threads
            .iter_mut()
            .find(|thread| thread.id == id)
            .ok_or("thread not found")?;

        if !thread.next_actions.contains(&action) {
            thread.next_actions.push(action);
        }

        thread.updated_at = Local::now().to_rfc3339();

        self.save()
    }

    /// Attach a world-model record or named model to a thread by reference.
    /// Returns true when the reference was newly added, false when the thread
    /// already carried it (idempotent; nothing is rewritten in that case).
    pub fn link_world(
        &mut self,
        id: u64,
        target: String,
        kind: &str,
    ) -> Result<bool, Box<dyn Error>> {
        let _file = self.fresh()?;
        if kind != "record" && kind != "model" {
            return Err(format!("kind must be \"model\" or \"record\", got {kind:?}").into());
        }
        let target = target.trim().to_string();
        if target.is_empty() {
            return Err("target must be a non-empty world-model id".into());
        }
        // Check the thread before touching the world-model store, so a wrong
        // thread id is reported as such rather than as a missing world reference.
        if !self.threads.iter().any(|t| t.id == id) {
            return Err(format!("thread {id} not found; use thread_list for valid ids").into());
        }

        let dbpath = self.path.with_file_name("world_model.sqlite3");
        let exists = if kind == "record" {
            crate::world_model::record_exists(&dbpath, &target)?
        } else {
            crate::world_model::model_exists(&dbpath, &target)?
        };
        if !exists {
            let hint = if kind == "record" {
                "use world_record_get or world_model_get to find a record id, or create it with world_record_put first"
            } else {
                "use world_model_list to find a model id, or create it with world_model_define first"
            };
            return Err(format!("world {kind} {target:?} not found; {hint}").into());
        }

        let thread = self
            .threads
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or("thread not found")?;
        let refs = if kind == "record" {
            &mut thread.world_record_ids
        } else {
            &mut thread.model_ids
        };
        if refs.contains(&target) {
            return Ok(false);
        }
        refs.push(target);
        thread.updated_at = Local::now().to_rfc3339();
        self.save()?;
        Ok(true)
    }

    fn save(&self) -> Result<(), Box<dyn Error>> {
        write_threads(&self.path, &self.threads)
    }
}

// ======================================================
// TOOL INTERFACE
// ======================================================

/// Every tool this module executes. Dispatchers route with `handles` instead of
/// keeping their own copy of the list: a stale copy in agent.rs (still naming the
/// retired `thread_link`) is what left `thread_link_world` unreachable from the
/// main loop while sleep could call it.
pub const TOOL_NAMES: &[&str] = &[
    "thread_link_world",
    "thread_create",
    "thread_list",
    "thread_get",
    "thread_update",
    "thread_add_event",
    "thread_remove_event",
    "thread_add_question",
    "thread_remove_question",
    "thread_add_action",
    "thread_set_state",
];

pub fn handles(name: &str) -> bool {
    TOOL_NAMES.contains(&name)
}

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::function(
            "thread_link_world",
            "Attach an existing world-model reference to a thread. kind \"model\" takes a model id (world_model_list); kind \"record\" takes a record id (world_record_get / world_record_put). The target must already exist; linking twice is a no-op.",
            json!({"type":"object","additionalProperties":false,"properties":{"id":{"type":"integer","description":"Thread id."},"kind":{"type":"string","enum":["model","record"]},"target":{"type":"string","minLength":1,"description":"Existing world model id or world record id."}},"required":["id","kind","target"]}),
        ),
        ToolDefinition::function(
            "thread_create",
            "Create ongoing attention: a question, intention, commitment or experiment. Description states the purpose, not factual understanding; store claims in named world models and link them.",
            json!({
                "type":"object",
                "properties":{
                    "title":{
                        "type":"string"
                    },
                    "description":{
                        "type":"string"
                    },
                    "category":{
                        "type":"string",
                        "enum":[
                            "relationship",
                            "question",
                            "project",
                            "problem",
                            "commitment",
                            "curiosity",
                            "self_understanding",
                            "experiment"
                        ]
                    },
                    "state":{
                        "type":"string",
                        "enum":[
                            "candidate",
                            "active",
                            "dormant",
                            "transformed",
                            "resolved",
                            "abandoned"
                        ]
                    },
                    "event_ids":{
                        "type":"array",
                        "items":{
                            "type":"integer"
                        }
                    }
                },
                "required":[
                    "title",
                    "description",
                    "category"
                ],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thread_list",
            "List a compact continuity-thread index. Use thread_get for full details.",
            json!({
                "type":"object",
                "properties":{},
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thread_get",
            "Retrieve a continuity thread by id.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    }
                },
                "required":[
                    "id"
                ],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thread_update",
            "Update a thread's purpose, category or significance. Factual understanding belongs in world records, not thread summaries.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    },
                    "category":{
                        "type":"string",
                        "enum":[
                            "relationship",
                            "question",
                            "project",
                            "problem",
                            "commitment",
                            "curiosity",
                            "self_understanding",
                            "experiment"
                        ]
                    },
                    "title":{
                        "type":"string"
                    },
                    "description":{
                        "type":"string"
                    },
                    "significance":{
                        "type":"string"
                    }
                },
                "required":[
                    "id"
                ],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thread_add_event",
            "Attach a hippocampus event to a thread.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    },
                    "event_id":{
                        "type":"integer"
                    }
                },
                "required":[
                    "id",
                    "event_id"
                ],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thread_remove_event",
            "Remove a hippocampus event from a thread.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    },
                    "event_id":{
                        "type":"integer"
                    }
                },
                "required":[
                    "id",
                    "event_id"
                ],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thread_add_question",
            "Add an unresolved question to a thread.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    },
                    "question":{
                        "type":"string"
                    }
                },
                "required":[
                    "id",
                    "question"
                ],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thread_remove_question",
            "Remove a resolved question from a thread.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    },
                    "question":{
                        "type":"string"
                    }
                },
                "required":[
                    "id",
                    "question"
                ],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thread_add_action",
            "Add a possible next action to a thread.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    },
                    "action":{
                        "type":"string"
                    }
                },
                "required":[
                    "id",
                    "action"
                ],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thread_set_state",
            "Change the lifecycle state of a thread.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    },
                    "state":{
                        "type":"string"
                    }
                },
                "required":[
                    "id",
                    "state"
                ],
                "additionalProperties":false
            }),
        ),
    ]
}

fn parse_state(value: &str) -> ThreadState {
    match value {
        "candidate" => ThreadState::Candidate,
        "dormant" => ThreadState::Dormant,
        "transformed" => ThreadState::Transformed,
        "resolved" => ThreadState::Resolved,
        "abandoned" => ThreadState::Abandoned,
        _ => ThreadState::Active,
    }
}

fn parse_category(value: &str) -> ThreadCategory {
    match value {
        "relationship" => ThreadCategory::Relationship,
        "project" => ThreadCategory::Project,
        "problem" => ThreadCategory::Problem,
        "commitment" => ThreadCategory::Commitment,
        "curiosity" => ThreadCategory::Curiosity,
        "self_understanding" => ThreadCategory::SelfUnderstanding,
        "experiment" => ThreadCategory::Experiment,
        _ => ThreadCategory::Question,
    }
}

fn string_arg(args: &Value, name: &str) -> Result<String, Box<dyn Error>> {
    Ok(args
        .get(name)
        .and_then(Value::as_str)
        .ok_or(format!("missing {name}"))?
        .to_string())
}

fn u64_arg(args: &Value, name: &str) -> Result<u64, Box<dyn Error>> {
    let value = args.get(name).ok_or(format!("missing {name}"))?;
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()))
        .ok_or_else(|| format!("{name} must be a non-negative integer").into())
}

fn optional_string(args: &Value, name: &str) -> Option<String> {
    args.get(name)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn array_u64(args: &Value, name: &str) -> Vec<u64> {
    args.get(name)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default()
}

pub fn execute(store: &mut ThreadStore, call: &ToolCall) -> Result<String, Box<dyn Error>> {
    let args: Value = serde_json::from_str(&call.function.arguments)?;

    if args.get("summary").is_some() {
        return Err("thread summary removed; use world_record_put and thread_link_world".into());
    }
    match call.function.name.as_str() {
        "thread_link_world" => {
            let id = u64_arg(&args, "id")?;
            let kind = string_arg(&args, "kind")?;
            let target = string_arg(&args, "target")?;
            let added = store.link_world(id, target.clone(), &kind)?;

            Ok(json!({
                "ok": true,
                "thread_id": id,
                "kind": kind,
                "target": target.trim(),
                "already_linked": !added
            })
            .to_string())
        }
        "thread_create" => {
            let thread = store.create(
                string_arg(&args, "title")?,
                string_arg(&args, "description")?,
                args.get("category")
                    .and_then(Value::as_str)
                    .map(parse_category)
                    .unwrap_or(ThreadCategory::Question),
                args.get("state")
                    .and_then(Value::as_str)
                    .map(parse_state)
                    .unwrap_or(ThreadState::Candidate),
                array_u64(&args, "event_ids"),
            )?;

            Ok(json!({
                "ok": true,
                "thread": thread
            })
            .to_string())
        }

        "thread_list" => Ok(json!({
            "ok": true,
            "threads": store.list().iter().map(|t| json!({"id":t.id,"title":t.title,"category":t.category,"state":t.state,
                "updated":crate::clock::stored(&t.updated_at)})).collect::<Vec<_>>(),
            "detail_tool":"thread_get"
        })
        .to_string()),

        "thread_get" => {
            let id = u64_arg(&args, "id")?;

            Ok(json!({
                "ok": true,
                "thread": store.get(id)
            })
            .to_string())
        }

        "thread_update" => {
            store.update(
                u64_arg(&args, "id")?,
                args.get("category")
                    .and_then(Value::as_str)
                    .map(parse_category),
                optional_string(&args, "title"),
                optional_string(&args, "description"),
                optional_string(&args, "significance"),
            )?;

            Ok(json!({
                "ok": true
            })
            .to_string())
        }

        "thread_add_event" => {
            store.add_event(u64_arg(&args, "id")?, u64_arg(&args, "event_id")?)?;

            Ok(json!({
                "ok": true
            })
            .to_string())
        }

        "thread_remove_event" => {
            store.remove_event(u64_arg(&args, "id")?, u64_arg(&args, "event_id")?)?;

            Ok(json!({
                "ok": true
            })
            .to_string())
        }

        "thread_add_question" => {
            store.add_question(u64_arg(&args, "id")?, string_arg(&args, "question")?)?;

            Ok(json!({
                "ok": true
            })
            .to_string())
        }

        "thread_remove_question" => {
            store.remove_question(u64_arg(&args, "id")?, string_arg(&args, "question")?)?;

            Ok(json!({
                "ok": true
            })
            .to_string())
        }

        "thread_add_action" => {
            store.add_action(u64_arg(&args, "id")?, string_arg(&args, "action")?)?;

            Ok(json!({
                "ok": true
            })
            .to_string())
        }

        "thread_set_state" => {
            store.set_state(
                u64_arg(&args, "id")?,
                parse_state(&string_arg(&args, "state")?),
            )?;

            Ok(json!({
                "ok": true
            })
            .to_string())
        }

        name => Ok(json!({
            "ok": false,
            "error": format!("unknown thread tool: {name}")
        })
        .to_string()),
    }
}

#[cfg(test)]
mod link_world_tests {
    use super::*;
    use crate::types::FunctionCall;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn store() -> (PathBuf, ThreadStore) {
        let dir = std::env::temp_dir().join(format!(
            "s0ma-threads-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let db = crate::world_model::open(&dir.join("world_model.sqlite3")).unwrap();
        db.execute_batch(
            "INSERT INTO wm_models VALUES('htb','HTB','boxes',1);
             INSERT INTO wm_records VALUES('htb:busqueda','htb',1,'{}',0,'now');",
        )
        .unwrap();
        let mut store = ThreadStore::new(dir.join("threads.json")).unwrap();
        store
            .create("t".into(), "d".into(), ThreadCategory::Project, ThreadState::Active, vec![])
            .unwrap();
        (dir, store)
    }

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: "c".into(),
            kind: "function".into(),
            function: FunctionCall { name: name.into(), arguments: args.to_string() },
        }
    }

    #[test]
    fn every_defined_tool_is_routed_here() {
        let defined: Vec<String> =
            tool_definitions().into_iter().map(|t| t.function.name).collect();
        assert_eq!(defined.len(), TOOL_NAMES.len());
        let (_dir, mut store) = store();
        for name in &defined {
            assert!(handles(name), "{name} defined but not in TOOL_NAMES");
            let out = execute(&mut store, &call(name, json!({})));
            if let Ok(text) = out {
                assert!(!text.contains("unknown thread tool"), "{name} has no execute arm");
            }
        }
        assert!(!handles("thread_link"), "retired name must not route");
    }

    #[test]
    fn links_record_and_model_and_persists() {
        let (dir, mut store) = store();
        let out = execute(&mut store, &call("thread_link_world", json!({"id":1,"kind":"record","target":"htb:busqueda"}))).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap()["already_linked"], false);
        execute(&mut store, &call("thread_link_world", json!({"id":"1","kind":"model","target":" htb "}))).unwrap();
        let again = execute(&mut store, &call("thread_link_world", json!({"id":1,"kind":"model","target":"htb"}))).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&again).unwrap()["already_linked"], true);

        let reloaded = ThreadStore::new(dir.join("threads.json")).unwrap();
        let thread = reloaded.get(1).unwrap();
        assert_eq!(thread.world_record_ids, vec!["htb:busqueda".to_string()]);
        assert_eq!(thread.model_ids, vec!["htb".to_string()]);
    }

    #[test]
    fn rejects_bad_kind_thread_and_target() {
        let (_dir, mut store) = store();
        let mut err = |args: Value| execute(&mut store, &call("thread_link_world", args)).unwrap_err().to_string();
        assert!(err(json!({"id":1,"kind":"episode","target":"htb"})).contains("kind must be"));
        assert!(err(json!({"id":99,"kind":"model","target":"htb"})).contains("thread 99 not found"));
        assert!(err(json!({"id":1,"kind":"record","target":"htb"})).contains("world record"));
        assert!(err(json!({"id":1,"kind":"model","target":"htb:busqueda"})).contains("world model"));
        assert!(err(json!({"id":1,"kind":"model","target":"  "})).contains("non-empty"));
    }
}
