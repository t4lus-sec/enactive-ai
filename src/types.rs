use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,

    #[serde(rename = "type")]
    pub kind: String,

    pub function: FunctionCall,
}

/// Repair tool calls whose arguments aren't valid JSON, returning the IDs of the
/// ones that needed it.
///
/// A generation that runs out of room mid-tool-call emits a fragment — `{"episode_id"`
/// and nothing else. Executing it fails, which is fine and recoverable. The part
/// that is not recoverable is that the fragment goes into the transcript, and the
/// transcript gets replayed on the next request: the server parses every
/// `arguments` string it is handed, so one truncated call turns into a 400 on
/// that request and every request after it. The conversation is poisoned, and the
/// error it reports ("Expecting value: line 1 column 14") describes the fragment
/// rather than anything the caller did.
///
/// So the fragment never enters the transcript. The arguments become an empty
/// object, which is always parseable, and the caller turns the repaired IDs into
/// tool results saying what happened — the model is told its call was cut off,
/// which is what it needs to know to reissue a shorter one.
pub fn repair_tool_call_arguments(calls: &mut [ToolCall]) -> Vec<String> {
    let mut repaired = Vec::new();
    for call in calls.iter_mut() {
        let arguments = call.function.arguments.trim();
        // An empty argument list is legitimate for a no-parameter tool; the
        // server accepts it, so leave it alone.
        if arguments.is_empty() {
            continue;
        }
        if serde_json::from_str::<serde_json::Value>(arguments).is_err() {
            call.function.arguments = "{}".to_string();
            repaired.push(call.id.clone());
        }
    }
    repaired
}

/// How many malformed tool calls one agent pass tolerates before giving up.
pub const MAX_MALFORMED_TOOL_CALL_RETRIES: usize = 2;

impl ToolCall {
    /// Why these arguments can't be stored in history, if they can't.
    ///
    /// vLLM parses the arguments of every tool call in the conversation as JSON
    /// when it renders the chat template. One malformed call in history therefore
    /// makes EVERY later request fail with an HTTP 400 before generation starts,
    /// and retrying cannot help because the same history is re-sent. Arguments are
    /// rebuilt by concatenating streamed fragments, so a cut-off or mangled call
    /// can arrive even when the response was not flagged as length-truncated.
    pub fn argument_error(&self) -> Option<String> {
        let arguments = self.function.arguments.trim();
        if arguments.is_empty() {
            return None; // normalized to {} below
        }
        match serde_json::from_str::<serde_json::Value>(arguments) {
            Ok(serde_json::Value::Object(_)) => None,
            Ok(_) => Some("arguments are JSON but not an object".into()),
            Err(error) => Some(error.to_string()),
        }
    }

    /// An argument-free call may stream no argument text at all; an empty string
    /// is not valid JSON, so store it as an empty object.
    pub fn normalize_empty_arguments(&mut self) {
        if self.function.arguments.trim().is_empty() {
            self.function.arguments = "{}".into();
        }
    }
}

/// Check a response's tool calls before any of them is executed or stored.
/// Returns a runtime notice for the model when the batch must be rejected.
///
/// The whole batch is rejected, not just the bad call: a batch is committed to
/// history as one assistant message, and every call in it needs a result.
pub fn reject_malformed_tool_calls(
    calls: &mut [ToolCall],
    truncated: bool,
) -> Option<String> {
    for call in calls.iter_mut() {
        call.normalize_empty_arguments();
    }

    let problems: Vec<String> = calls
        .iter()
        .filter_map(|call| {
            call.argument_error().map(|error| {
                format!(
                    "{} ({} characters of arguments): {error}",
                    call.function.name,
                    call.function.arguments.chars().count()
                )
            })
        })
        .collect();

    if problems.is_empty() && !truncated {
        return None;
    }

    let cause = if truncated {
        "your response hit its output limit partway through, so the call was cut off".to_string()
    } else {
        format!("the arguments were not valid JSON: {}", problems.join("; "))
    };

    Some(format!(
        "[runtime] Your last tool call batch was NOT executed and nothing ran: {cause}. \
         It has been left out of the conversation so it cannot break later requests. \
         Retry with smaller arguments; split a large file write into several calls."
    ))
}

#[cfg(test)]
mod tool_call_validation_tests {
    use super::*;
    fn call(args: &str) -> ToolCall {
        ToolCall { id: "c".into(), kind: "function".into(),
            function: FunctionCall { name: "vm_exec".into(), arguments: args.into() } }
    }
    #[test]
    fn the_reported_truncation_is_rejected() {
        // vLLM 400: Unterminated string starting at char 12, i.e. `{"command": "`
        assert!(call("{\"command\": \"cd /workspace && cat big").argument_error().is_some());
    }
    #[test]
    fn valid_and_empty_arguments_pass() {
        assert!(call("{\"command\":\"ls\"}").argument_error().is_none());
        assert!(call("  ").argument_error().is_none()); // normalized to {}
        let mut c = call(""); c.normalize_empty_arguments();
        assert_eq!(c.function.arguments, "{}");
    }
    #[test]
    fn non_object_arguments_are_rejected() {
        assert!(call("[1,2,3]").argument_error().is_some());
        assert!(call("\"just a string\"").argument_error().is_some());
    }
    #[test]
    fn batch_rejection_reports_and_a_clean_truncation_still_rejects() {
        let mut bad = [call("{\"command\": \"cat")];
        assert!(reject_malformed_tool_calls(&mut bad, false).is_some());
        // Valid JSON, but the response was length-truncated: reject anyway.
        let mut cut = [call("{\"command\":\"ls\"}")];
        assert!(reject_malformed_tool_calls(&mut cut, true).is_some());
        // Valid and complete: allowed, and empty args normalized.
        let mut good = [call("{\"command\":\"ls\"}"), call("")];
        assert!(reject_malformed_tool_calls(&mut good, false).is_none());
        assert_eq!(good[1].function.arguments, "{}");
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,

    // Present only while the image-bearing turn is active. Custom
    // serialization emits OpenAI-compatible typed content and never exposes
    // this as an extra request field.
    image_data_url: Option<String>,
    image_scope: Option<ImageScope>,

    // Whether the operator wrote this, as opposed to the harness speaking in a user
    // message (autonomous prompts, subagent results, reminders, notices).
    // Never serialized. Folding reads it, so a gist can't attribute runtime
    // text to the operator.
    operator_input: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum ImageScope {
    Turn,
    Desktop,
}

/// A message as kept on disk (`context_resume`): every field, the ones the
/// request never carries too, so it comes back exactly as it was and is sent
/// exactly as it would have been.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StoredMessage {
    role: String,
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    image_data_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    image_scope: Option<ImageScope>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    operator_input: bool,
}

impl Message {
    /// Every field into `state`, the ones the request never carries too, by
    /// reference: what `StoredMessage` would hash, without the copy.
    pub fn hash_all<H: std::hash::Hasher>(&self, state: &mut H) {
        use std::hash::Hash;
        self.role.hash(state);
        self.content.hash(state);
        self.reasoning_content.hash(state);
        self.tool_calls.hash(state);
        self.tool_call_id.hash(state);
        self.image_data_url.hash(state);
        self.image_scope.hash(state);
        self.operator_input.hash(state);
    }
}

impl From<&Message> for StoredMessage {
    fn from(message: &Message) -> Self {
        Self {
            role: message.role.clone(),
            content: message.content.clone(),
            reasoning_content: message.reasoning_content.clone(),
            tool_calls: message.tool_calls.clone(),
            tool_call_id: message.tool_call_id.clone(),
            image_data_url: message.image_data_url.clone(),
            image_scope: message.image_scope,
            operator_input: message.operator_input,
        }
    }
}

impl From<StoredMessage> for Message {
    fn from(stored: StoredMessage) -> Self {
        Self {
            role: stored.role,
            content: stored.content,
            reasoning_content: stored.reasoning_content,
            tool_calls: stored.tool_calls,
            tool_call_id: stored.tool_call_id,
            image_data_url: stored.image_data_url,
            image_scope: stored.image_scope,
            operator_input: stored.operator_input,
        }
    }
}

#[derive(Serialize)]
struct ImageUrl<'a> {
    url: &'a str,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart<'a> {
    Text { text: &'a str },
    ImageUrl { image_url: ImageUrl<'a> },
}

impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut fields = 2;
        if self.reasoning_content.is_some() { fields += 2; }

        if self.tool_calls.is_some() {
            fields += 1;
        }

        if self.tool_call_id.is_some() {
            fields += 1;
        }

        let mut map = serializer.serialize_map(Some(fields))?;

        map.serialize_entry("role", &self.role)?;
        if let Some(reasoning) = &self.reasoning_content {
            // Replayed under both names, as the model card's own example does:
            // vLLM releases have read preserved thinking from one or the other.
            // The template consumes one, so this never renders twice.
            map.serialize_entry("reasoning_content", reasoning)?;
            map.serialize_entry("reasoning", reasoning)?;
        }

        if let Some(data_url) = self.image_data_url.as_deref() {
            let parts = [
                ContentPart::ImageUrl {
                    image_url: ImageUrl { url: data_url },
                },
                ContentPart::Text {
                    text: &self.content,
                },
            ];

            map.serialize_entry("content", &parts)?;
        } else {
            map.serialize_entry("content", &self.content)?;
        }

        if let Some(tool_calls) = &self.tool_calls {
            map.serialize_entry("tool_calls", tool_calls)?;
        }

        if let Some(tool_call_id) = &self.tool_call_id {
            map.serialize_entry("tool_call_id", tool_call_id)?;
        }

        map.end()
    }
}

impl Message {
    /// Preserve the server's reasoning separately from visible assistant text.
    pub fn with_reasoning(mut self, reasoning: String) -> Self {
        if self.role == "assistant" && !reasoning.is_empty() {
            self.reasoning_content = Some(reasoning);
        }
        self
    }

    pub fn system(content: String) -> Self {
        Self {
            role: "system".to_string(),
            content,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            image_data_url: None,
            image_scope: None,
            operator_input: false,
        }
    }

    pub fn user(content: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            image_data_url: None,
            image_scope: None,
            operator_input: false,
        }
    }

    pub fn user_with_image(content: String, image_data_url: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            image_data_url: Some(image_data_url),
            image_scope: Some(ImageScope::Turn),
            operator_input: false,
        }
    }

    pub fn desktop_observation(content: String, image_data_url: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            image_data_url: Some(image_data_url),
            image_scope: Some(ImageScope::Desktop),
            operator_input: false,
        }
    }

    pub fn assistant(content: String) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            image_data_url: None,
            image_scope: None,
            operator_input: false,
        }
    }

    pub fn assistant_with_tools(content: String, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            reasoning_content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            image_data_url: None,
            image_scope: None,
            operator_input: false,
        }
    }

    pub fn tool(tool_call_id: String, content: String) -> Self {
        Self::tool_bounded(tool_call_id, crate::llm::limit_tool_output(content))
    }

    /// A tool message whose content the caller has already bounded (a
    /// subagent's view of its own result). `tool` would count it again, and if
    /// that count failed, wrap it in a second preview that could cut out the
    /// labels saying where the rest is.
    pub fn tool_bounded(tool_call_id: String, content: String) -> Self {
        Self {
            role: "tool".to_string(),
            content,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
            image_data_url: None,
            image_scope: None,
            operator_input: false,
        }
    }

    /// Remove binary image transport after the active agent cycle. The text
    /// and attachment metadata remain in working and episodic history.
    /// Mark this as the operator's own input.
    pub fn by_operator(mut self) -> Self {
        self.operator_input = true;
        self
    }

    /// Whether the operator wrote this (see `by_operator`).
    pub fn is_operator_input(&self) -> bool {
        self.operator_input
    }

    pub fn release_image(&mut self) -> bool {
        self.image_scope = None;
        self.image_data_url.take().is_some()
    }

    /// Remove an earlier desktop frame immediately before a newer one is
    /// attached. Operator-provided `/img` content remains available for the
    /// rest of its active agent cycle.
    pub fn release_desktop_image(&mut self) -> bool {
        if self.image_scope == Some(ImageScope::Desktop) {
            self.image_scope = None;
            self.image_data_url.take().is_some()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Message;
    use serde_json::json;

    #[test]
    fn reasoning_survives_tool_result_and_new_user_turn() {
        use super::{FunctionCall, ToolCall};
        let trace = "Check the owner control before interpreting a null result.\n";
        let messages = vec![
            Message::user("Investigate".into()),
            Message::assistant_with_tools("Checking".into(), vec![ToolCall {
                id: "call_1".into(), kind: "function".into(),
                function: FunctionCall { name: "vm_exec".into(), arguments: "{}".into() },
            }]).with_reasoning(trace.into()),
            Message::tool("call_1".into(), "owner read succeeded".into()),
            Message::assistant("Result".into()).with_reasoning("The control is valid.".into()),
            Message::user("Continue".into()),
        ];
        let encoded = serde_json::to_value(&messages).unwrap();
        assert_eq!(encoded[1]["reasoning_content"], trace);
        assert_eq!(encoded[1]["content"], "Checking");
        assert_eq!(encoded[1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(encoded[2]["tool_call_id"], "call_1");
        assert!(encoded[2].get("reasoning_content").is_none());
        assert_eq!(encoded[3]["reasoning_content"], "The control is valid.");
        assert!(encoded[4].get("reasoning_content").is_none());
    }

    #[test]
    fn empty_trace_does_not_change_existing_message_format() {
        let encoded = serde_json::to_value(Message::assistant("hello".into())
            .with_reasoning(String::new())).unwrap();
        assert_eq!(encoded, json!({"role": "assistant", "content": "hello"}));
    }

    #[test]
    fn ordinary_message_keeps_string_content() {
        let message = Message::user("hello".to_string());
        let value = serde_json::to_value(message).expect("message should serialize");

        assert_eq!(value, json!({"role": "user", "content": "hello"}));
    }

    #[test]
    fn image_message_uses_openai_content_parts() {
        let message = Message::user_with_image(
            "inspect this".to_string(),
            "data:image/png;base64,AAAA".to_string(),
        );
        let value = serde_json::to_value(message).expect("message should serialize");

        assert_eq!(
            value,
            json!({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"}
                    },
                    {"type": "text", "text": "inspect this"}
                ]
            })
        );
    }

    #[test]
    fn releasing_image_restores_string_content() {
        let mut message = Message::user_with_image(
            "inspect this".to_string(),
            "data:image/png;base64,AAAA".to_string(),
        );

        assert!(message.release_image());
        assert!(!message.release_image());

        let value = serde_json::to_value(message).expect("message should serialize");
        assert_eq!(value, json!({"role": "user", "content": "inspect this"}));
    }

    #[test]
    fn replacing_desktop_frame_does_not_release_operator_image() {
        let mut operator = Message::user_with_image(
            "operator image".to_string(),
            "data:image/png;base64,AAAA".to_string(),
        );
        let mut desktop = Message::desktop_observation(
            "desktop frame".to_string(),
            "data:image/png;base64,BBBB".to_string(),
        );

        assert!(!operator.release_desktop_image());
        assert!(desktop.release_desktop_image());
        assert!(serde_json::to_value(operator).unwrap()["content"].is_array());
        assert_eq!(
            serde_json::to_value(desktop).unwrap()["content"],
            "desktop frame"
        );
    }
}
