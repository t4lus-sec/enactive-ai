use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    error::Error,
    fs::{self, File},
    io::{Read, Write},
    path::PathBuf,
};

use crate::llm::ToolDefinition;
use crate::types::ToolCall;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThoughtState {
    Active,
    Dormant,
    Integrated,
    Discarded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    pub id: u64,

    pub created_at: String,
    pub updated_at: String,

    pub content: String,

    pub state: ThoughtState,

    #[serde(default)]
    pub related_threads: Vec<u64>,

    #[serde(default)]
    pub source_events: Vec<u64>,

    #[serde(default)]
    pub notes: String,
}

pub struct ThoughtStore {
    path: PathBuf,
    thoughts: Vec<Thought>,
    next_id: u64,
}

impl ThoughtStore {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, Box<dyn Error>> {
        let path = path.into();

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let thoughts = if path.exists() {
            let mut contents = String::new();

            File::open(&path)?.read_to_string(&mut contents)?;

            if contents.trim().is_empty() {
                Vec::new()
            } else {
                serde_json::from_str::<Vec<Thought>>(&contents)?
            }
        } else {
            Vec::new()
        };

        let next_id = thoughts.iter().map(|thought| thought.id).max().unwrap_or(0) + 1;

        Ok(Self {
            path,
            thoughts,
            next_id,
        })
    }

    pub fn create(&mut self, content: String) -> Result<Thought, Box<dyn Error>> {
        let now = Local::now().to_rfc3339();

        let thought = Thought {
            id: self.next_id,

            created_at: now.clone(),
            updated_at: now,

            content,

            state: ThoughtState::Active,

            related_threads: Vec::new(),
            source_events: Vec::new(),

            notes: String::new(),
        };

        self.next_id += 1;

        self.thoughts.push(thought.clone());

        self.save()?;

        Ok(thought)
    }

    pub fn active(&self) -> Vec<Thought> {
        self.thoughts
            .iter()
            .filter(|thought| matches!(thought.state, ThoughtState::Active))
            .cloned()
            .collect()
    }

    pub fn all(&self) -> Vec<Thought> {
        self.thoughts.clone()
    }

    pub fn get(&self, id: u64) -> Option<Thought> {
        self.thoughts
            .iter()
            .find(|thought| thought.id == id)
            .cloned()
    }

    pub fn update_content(&mut self, id: u64, content: String) -> Result<(), Box<dyn Error>> {
        let thought = self
            .thoughts
            .iter_mut()
            .find(|thought| thought.id == id)
            .ok_or("thought not found")?;

        thought.content = content;
        thought.updated_at = Local::now().to_rfc3339();

        self.save()
    }

    pub fn update_state(&mut self, id: u64, state: ThoughtState) -> Result<(), Box<dyn Error>> {
        let thought = self
            .thoughts
            .iter_mut()
            .find(|thought| thought.id == id)
            .ok_or("thought not found")?;

        thought.state = state;
        thought.updated_at = Local::now().to_rfc3339();

        self.save()
    }

    pub fn attach_thread(&mut self, id: u64, thread_id: u64) -> Result<(), Box<dyn Error>> {
        let thought = self
            .thoughts
            .iter_mut()
            .find(|thought| thought.id == id)
            .ok_or("thought not found")?;

        if !thought.related_threads.contains(&thread_id) {
            thought.related_threads.push(thread_id);
        }

        thought.updated_at = Local::now().to_rfc3339();

        self.save()
    }

    pub fn cleanup(&mut self) -> Result<(), Box<dyn Error>> {
        self.thoughts
            .retain(|thought| !matches!(thought.state, ThoughtState::Discarded));

        self.save()
    }

    fn save(&self) -> Result<(), Box<dyn Error>> {
        let temp = self.path.with_extension("tmp");

        {
            let mut file = File::create(&temp)?;

            serde_json::to_writer_pretty(&mut file, &self.thoughts)?;

            file.write_all(b"\n")?;
            file.sync_all()?;
        }

        fs::rename(temp, &self.path)?;

        Ok(())
    }
}

// -------------------------
// LLM TOOL INTERFACE
// -------------------------

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::function(
            "thought_create",
            "Create a temporary active thought in cognitive workspace.",
            json!({
                "type":"object",
                "properties":{
                    "content":{
                        "type":"string"
                    }
                },
                "required":["content"],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thought_list",
            "List active thoughts.",
            json!({
                "type":"object",
                "properties":{},
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thought_get",
            "Get a thought by id.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    }
                },
                "required":["id"],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thought_update",
            "Update an existing thought.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    },
                    "content":{
                        "type":"string"
                    }
                },
                "required":["id","content"],
                "additionalProperties":false
            }),
        ),
        ToolDefinition::function(
            "thought_discard",
            "Remove a thought from active cognition.",
            json!({
                "type":"object",
                "properties":{
                    "id":{
                        "type":"integer"
                    }
                },
                "required":["id"],
                "additionalProperties":false
            }),
        ),
    ]
}

pub fn execute(store: &mut ThoughtStore, call: &ToolCall) -> Result<String, Box<dyn Error>> {
    let args: Value = serde_json::from_str(&call.function.arguments)?;

    match call.function.name.as_str() {
        "thought_create" => {
            let content = args
                .get("content")
                .and_then(Value::as_str)
                .ok_or("missing content")?;

            Ok(json!({
                "ok": true,
                "thought": store.create(content.to_string())?
            })
            .to_string())
        }

        "thought_list" => Ok(json!({
            "ok": true,
            "thoughts": store.active()
        })
        .to_string()),

        "thought_get" => {
            let id = args.get("id").and_then(Value::as_u64).ok_or("missing id")?;

            Ok(match store.get(id) {
                Some(thought) => json!({
                    "ok": true,
                    "thought": thought
                }),

                None => json!({
                    "ok": false,
                    "error": "thought not found"
                }),
            }
            .to_string())
        }

        "thought_update" => {
            let id = args.get("id").and_then(Value::as_u64).ok_or("missing id")?;

            let content = args
                .get("content")
                .and_then(Value::as_str)
                .ok_or("missing content")?;

            store.update_content(id, content.to_string())?;

            Ok(json!({
                "ok": true
            })
            .to_string())
        }

        "thought_discard" => {
            let id = args.get("id").and_then(Value::as_u64).ok_or("missing id")?;

            store.update_state(id, ThoughtState::Discarded)?;

            Ok(json!({
                "ok": true
            })
            .to_string())
        }

        name => Ok(json!({
            "ok": false,
            "error": format!("unknown thought tool: {name}")
        })
        .to_string()),
    }
}
