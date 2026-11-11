//! Durable integration transcript. Writes with uncertain outcomes are never replayed.
use super::*;
use crate::types::ToolCall;
use sha2::{Digest, Sha256};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Serialize, Deserialize)]
struct StoredMessage {
    role: String,
    content: String,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    tool_call_id: Option<String>,
}
impl StoredMessage {
    fn from_message(m: &Message) -> Self {
        Self {
            role: m.role.clone(),
            content: m.content.clone(),
            reasoning_content: m.reasoning_content.clone(),
            tool_calls: m.tool_calls.clone(),
            tool_call_id: m.tool_call_id.clone(),
        }
    }
    fn restore(&self) -> Message {
        let mut m = Message::system(self.content.clone());
        m.role = self.role.clone();
        m.reasoning_content = self.reasoning_content.clone();
        m.tool_calls = self.tool_calls.clone();
        m.tool_call_id = self.tool_call_id.clone();
        m
    }
}
#[derive(Serialize, Deserialize)]
pub(super) struct Journal {
    signature: String,
    messages: Vec<StoredMessage>,
    revisions: String,
    pub staged: Option<String>,
    pub complete: Option<String>,
    #[serde(skip)]
    path: Option<PathBuf>,
}
impl Journal {
    pub fn open(
        path: Option<&Path>,
        messages: &mut Vec<Message>,
        tools: &[llm::ToolDefinition],
        revisions: String,
    ) -> Result<Self> {
        let signature = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(&messages, tools))?)
        );
        let saved: Option<Self> = match path {
            Some(p) => SleepStateStore::read_json(p)?,
            None => None,
        };
        let mut journal = match saved {
            Some(mut saved) if saved.signature == signature => {
                *messages = saved.messages.iter().map(StoredMessage::restore).collect();
                let missing = fill_uncertain(messages);
                if missing || saved.revisions != revisions {
                    saved.complete = None;
                    messages.push(Message::user("Integration resumed with changed state or an uncertain tool outcome. Read affected records before further edits. Do not repeat a write merely because its result is absent; reconcile with current revisions.".into()));
                }
                saved
            }
            _ => Self {
                signature,
                messages: Vec::new(),
                revisions: revisions.clone(),
                staged: None,
                complete: None,
                path: None,
            },
        };
        journal.path = path.map(Path::to_owned);
        journal.persist(messages, revisions)?;
        Ok(journal)
    }
    pub fn persist(&mut self, messages: &[Message], revisions: String) -> Result<()> {
        self.messages = messages.iter().map(StoredMessage::from_message).collect();
        self.revisions = revisions;
        if let Some(path) = &self.path {
            SleepStateStore::write_json(path, self)?;
        }
        Ok(())
    }
}
fn fill_uncertain(messages: &mut Vec<Message>) -> bool {
    let Some(index) = messages
        .iter()
        .rposition(|m| m.tool_calls.as_ref().is_some_and(|v| !v.is_empty()))
    else {
        return false;
    };
    let calls = messages[index].tool_calls.clone().unwrap();
    let mut changed = false;
    for call in calls {
        if !messages[index + 1..]
            .iter()
            .any(|m| m.tool_call_id.as_deref() == Some(call.id.as_str()))
        {
            messages.push(Message::tool(call.id,json!({"ok":false,"status":"unknown_after_interruption",
                "instruction":"Read affected state and reconcile; this call was not automatically replayed."}).to_string()));
            changed = true;
        }
    }
    changed
}
pub(super) fn revisions(memory: &Hippocampus, threads: &ThreadStore) -> Result<String> {
    Ok(serde_json::to_string(&(
        threads.revision()?,
        crate::world_model::revision(&crate::world_model::path_for(memory))?,
    ))?)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uncertain_write_gets_a_result_without_replay() {
        let call = ToolCall {
            id: "write".into(),
            kind: "function".into(),
            function: crate::types::FunctionCall {
                name: "thread_create".into(),
                arguments: "{}".into(),
            },
        };
        let mut messages = vec![
            Message::assistant_with_tools("".into(), vec![call]).with_reasoning("thought".into()),
        ];
        assert!(fill_uncertain(&mut messages));
        assert_eq!(messages.len(), 2);
        assert!(!fill_uncertain(&mut messages));
        let restored = StoredMessage::from_message(&messages[0]).restore();
        assert_eq!(restored.reasoning_content.as_deref(), Some("thought"));
    }
    #[test]
    fn journal_resumes_completed_prefix_and_detects_state_drift() {
        let path = std::env::temp_dir().join(format!("sleep-journal-{}.json", std::process::id()));
        let _ = fs::remove_file(&path);
        let base = vec![Message::system("fixed".into())];
        let mut messages = base.clone();
        let mut journal = Journal::open(Some(&path), &mut messages, &[], "v1".into()).unwrap();
        messages.push(Message::assistant("checkpoint".into()));
        journal.complete = Some("checkpoint".into());
        journal.persist(&messages, "v1".into()).unwrap();
        let mut restored = base.clone();
        let saved = Journal::open(Some(&path), &mut restored, &[], "v1".into()).unwrap();
        assert_eq!(saved.complete.as_deref(), Some("checkpoint"));
        assert_eq!(restored.len(), 2);
        let mut restored = base;
        let saved = Journal::open(Some(&path), &mut restored, &[], "v2".into()).unwrap();
        assert!(saved.complete.is_none());
        assert_eq!(restored.len(), 3);
        fs::remove_file(path).unwrap();
    }
}
