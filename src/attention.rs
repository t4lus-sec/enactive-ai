use crate::{llm::ToolDefinition, types::ToolCall};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttentionMode {
    Engaged,
    Waiting,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConditionKind {
    OperatorMessage,
    /// An event through the VM wake channel (`vm_wake.rs`): a watch firing,
    /// or a script sending one.
    VmEvent,
}

/// Condition kinds s0ma no longer has: a Signal contact's message, and the
/// Minecraft events. A saved wait for one loads as a wait for operator input.
const RETIRED_KINDS: &[&str] = &[
    "contact_message",
    "minecraft_player_chat",
    "minecraft_action_completed",
    "minecraft_action_failed",
    "minecraft_event",
];

/// Condition fields from kinds s0ma no longer has: Minecraft's filters, and
/// the Signal contact a `contact_message` wait named.
const RETIRED_FIELDS: &[&str] = &["event_kind", "action_id", "username", "from"];

/// Unknown fields are refused, so a filter s0ma no longer has fails loudly
/// rather than being ignored (a saved one is dropped on load first).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WaitCondition {
    pub kind: ConditionKind,
    /// With `vm_event`: only events with this name end the wait.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watch: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionState {
    pub schema_version: u32,
    pub mode: AttentionMode,
    pub activity: String,
    pub orientation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<WaitCondition>,
    pub updated_at: String,
}

/// The activity of a wait the harness sets after a failure.
const RECOVERY_ACTIVITY: &str = "recovering a harness-level failure";

pub struct AttentionStore {
    state: AttentionState,
    path: PathBuf,
}

#[derive(Deserialize)]
struct OrientArgs {
    activity: String,
    orientation: String,
}

#[derive(Deserialize)]
struct WaitArgs {
    activity: String,
    orientation: String,
    condition: WaitCondition,
}

impl AttentionStore {
    pub fn load(data_dir: &Path) -> Result<Self, Box<dyn Error>> {
        let path = data_dir.join("attention.json");
        let unreadable = |error: &dyn std::fmt::Display| format!("{} can't be read: {error}", path.display());
        let state = match fs::read_to_string(&path) {
            Ok(text) => {
                let mut saved: serde_json::Value = serde_json::from_str(&text).map_err(|error| unreadable(&error))?;
                if let Some(kind) = retire_old_condition(&mut saved) {
                    crate::terminal::warn(&format!(
                        "[attention] the saved wait was for {kind}, which s0ma no longer has; it now waits \
                         for operator input"
                    ));
                }
                serde_json::from_value(saved).map_err(|error| unreadable(&error))?
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => AttentionState {
                schema_version: 1,
                mode: AttentionMode::Engaged,
                activity: "orienting to the current situation".into(),
                orientation: "notice what calls for attention".into(),
                condition: None,
                updated_at: Utc::now().to_rfc3339(),
            },
            Err(error) => return Err(error.into()),
        };
        validate_state(&state)?;
        Ok(Self { state, path })
    }

    pub fn is_engaged(&self) -> bool {
        self.state.mode == AttentionMode::Engaged
    }

    /// Waiting for a VM wake event (any, or one watch's). Only this kind of
    /// wait holds back the events it isn't for; a wait for operator input is
    /// idle as far as the VM is concerned.
    pub fn waits_for_vm_event(&self) -> bool {
        self.state.mode == AttentionMode::Waiting
            && self.state.condition.as_ref().is_some_and(|condition| condition.kind == ConditionKind::VmEvent)
    }

    /// The harness suspended s0ma after a failure (`wait_for_operator_recovery`)
    /// and only operator input should end it. Read at startup, so a restart
    /// doesn't let VM events retry what failed.
    pub fn is_recovery_wait(&self) -> bool {
        self.state.mode == AttentionMode::Waiting && self.state.activity == RECOVERY_ACTIVITY
    }
    pub fn state(&self) -> &AttentionState {
        &self.state
    }

    pub fn resume(&mut self, orientation: impl Into<String>) -> Result<(), Box<dyn Error>> {
        let mut next = self.state.clone();
        next.mode = AttentionMode::Engaged;
        next.orientation = orientation.into();
        next.condition = None;
        next.updated_at = Utc::now().to_rfc3339();
        self.replace(next)
    }

    pub fn wait_for_operator_recovery(
        &mut self,
        orientation: impl Into<String>,
    ) -> Result<(), Box<dyn Error>> {
        self.set_wait(WaitArgs {
            activity: RECOVERY_ACTIVITY.into(),
            orientation: orientation.into(),
            condition: WaitCondition { kind: ConditionKind::OperatorMessage, watch: None },
        })
    }

    /// Whether a VM wake event named `name` ends the current wait.
    pub fn matches_vm(&self, name: &str) -> bool {
        match (&self.state.mode, &self.state.condition) {
            (AttentionMode::Waiting, Some(condition)) => {
                condition.kind == ConditionKind::VmEvent
                    && condition.watch.as_deref().is_none_or(|watch| watch == name)
            }
            _ => false,
        }
    }

    pub fn continuation_prompt(&self) -> String {
        format!(
            "[CONTINUING ACTIVITY]\n\nCurrent activity: {}\nCurrent orientation: {}\n\n\
             No external event is required before continuing. Continue thinking, acting, \
             observing, or speaking according to what the situation calls for. Speaking is \
             an action, not a required heartbeat. Use attention_orient when your ongoing \
             activity or orientation materially changes. Use attention_wait only for a \
             concrete observable condition whose completion can be detected by the harness.",
            self.state.activity, self.state.orientation,
        )
    }

    fn set_orient(&mut self, args: OrientArgs) -> Result<(), Box<dyn Error>> {
        require_text("activity", &args.activity)?;
        require_text("orientation", &args.orientation)?;
        let mut next = self.state.clone();
        next.mode = AttentionMode::Engaged;
        next.activity = args.activity;
        next.orientation = args.orientation;
        next.condition = None;
        next.updated_at = Utc::now().to_rfc3339();
        self.replace(next)
    }

    fn set_wait(&mut self, args: WaitArgs) -> Result<(), Box<dyn Error>> {
        require_text("activity", &args.activity)?;
        require_text("orientation", &args.orientation)?;
        validate_condition(&args.condition)?;
        let mut next = self.state.clone();
        next.mode = AttentionMode::Waiting;
        next.activity = args.activity;
        next.orientation = args.orientation;
        next.condition = Some(args.condition);
        next.updated_at = Utc::now().to_rfc3339();
        self.replace(next)
    }

    fn replace(&mut self, next: AttentionState) -> Result<(), Box<dyn Error>> {
        validate_state(&next)?;
        self.persist(&next)?;
        self.state = next;
        Ok(())
    }

    fn persist(&self, state: &AttentionState) -> Result<(), Box<dyn Error>> {
        let parent = self
            .path
            .parent()
            .ok_or("attention state path has no parent")?;
        fs::create_dir_all(parent)?;
        let mut bytes = serde_json::to_vec_pretty(state)?;
        bytes.push(b'\n');
        crate::util::write_atomic(&self.path, bytes)?;
        Ok(())
    }
}

fn require_text(name: &str, value: &str) -> Result<(), Box<dyn Error>> {
    let length = value.trim().chars().count();
    if !(1..=500).contains(&length) {
        return Err(format!("{name} must contain 1..=500 characters").into());
    }
    Ok(())
}

fn validate_condition(condition: &WaitCondition) -> Result<(), Box<dyn Error>> {
    if condition.kind == ConditionKind::VmEvent {
        if condition.watch.as_deref().is_some_and(|watch| !crate::vm_wake::is_name(watch)) {
            return Err("condition.watch must be 1-32 letters, digits, _ or -".into());
        }
    } else if condition.watch.is_some() {
        return Err("condition.watch is allowed only with vm_event".into());
    }
    Ok(())
}

/// A saved wait for a retired kind becomes a wait for operator input (the
/// operator's first input after a start ends any wait anyway), and fields only
/// retired kinds had are dropped. Returns the retired kind. Anything else that
/// isn't a kind s0ma has is left to fail loading, as it always did.
fn retire_old_condition(saved: &mut serde_json::Value) -> Option<String> {
    let condition = saved.get_mut("condition")?.as_object_mut()?;
    for field in RETIRED_FIELDS {
        condition.remove(*field);
    }
    let kind = condition.get("kind")?.as_str()?.to_string();
    if !RETIRED_KINDS.contains(&kind.as_str()) {
        return None;
    }
    condition.clear();
    condition.insert("kind".into(), json!("operator_message"));
    Some(kind)
}

fn validate_state(state: &AttentionState) -> Result<(), Box<dyn Error>> {
    if state.schema_version != 1 {
        return Err("unsupported attention.json schema_version".into());
    }
    require_text("activity", &state.activity)?;
    require_text("orientation", &state.orientation)?;
    match (&state.mode, &state.condition) {
        (AttentionMode::Engaged, None) => Ok(()),
        (AttentionMode::Waiting, Some(condition)) => validate_condition(condition),
        _ => {
            Err("engaged attention cannot have a condition; waiting attention requires one".into())
        }
    }
}

pub fn tool_definitions() -> Vec<ToolDefinition> {
    // vm_event is offered only while the VM wake channel runs; a persisted
    // vm_event wait still loads without it, and operator input still ends it.
    let vm_wake = crate::vm_wake::enabled();
    let mut kinds = vec!["operator_message"];
    if vm_wake {
        kinds.push("vm_event");
    }
    let wait_description = if vm_wake {
        "Suspend the continuous inference loop only until one concrete observable condition is complete. For anything on your machine (a process finishing, a file appearing, a log line, a port opening, or a length of time) start a watch with vm_wake_watch and wait with condition kind vm_event; a vm_event wait also ends a response in conversation mode. To wait for the operator, use condition kind operator_message; wake events still start turns during that wait, as they do whenever you're idle, so it doesn't silence a watch or script (cancel those instead). Generic idleness and unspecified environmental change are unsupported. Operator input can always interrupt a wait."
    } else {
        "Suspend the continuous inference loop until the operator's next message (condition kind operator_message). Time, duration, generic idleness, and unspecified environmental change are unsupported."
    };
    let condition = json!({
        "type":"object", "additionalProperties":false,
        "properties":{
            "kind":{"type":"string","enum":kinds},
            "watch":{"type":"string","pattern":"^[A-Za-z0-9_-]{1,32}$","description":"vm_event only: the watch or event name to wait for; any wake event from your machine if omitted"}
        }, "required":["kind"]
    });
    vec![
        ToolDefinition::function(
            "attention_orient",
            "Persist the activity and orientation that should organize the next immediate cognitive cycle. This keeps inference engaged; it is not necessary on every cycle.",
            json!({"type":"object","additionalProperties":false,"properties":{
                "activity":{"type":"string","minLength":1,"maxLength":500},
                "orientation":{"type":"string","minLength":1,"maxLength":500}},
                "required":["activity","orientation"]}),
        ),
        ToolDefinition::function(
            "attention_wait",
            wait_description,
            json!({"type":"object","additionalProperties":false,"properties":{
                "activity":{"type":"string","minLength":1,"maxLength":500},
                "orientation":{"type":"string","minLength":1,"maxLength":500},
                "condition":condition},"required":["activity","orientation","condition"]}),
        ),
    ]
}

pub fn execute(store: &mut AttentionStore, call: &ToolCall) -> Result<String, Box<dyn Error>> {
    match call.function.name.as_str() {
        "attention_orient" => {
            store.set_orient(serde_json::from_str(&call.function.arguments)?)?;
            Ok(json!({"ok":true,"mode":"engaged","state":store.state()}).to_string())
        }
        "attention_wait" => {
            let args: WaitArgs = serde_json::from_str(&call.function.arguments)?;
            // Checked here, not in validation: a saved vm_event wait must still
            // load after the channel is turned off (operator input ends it).
            if args.condition.kind == ConditionKind::VmEvent && !crate::vm_wake::enabled() {
                return Err("the wake channel is off, so a vm_event wait would never end".into());
            }
            let already = match args.condition.kind {
                ConditionKind::VmEvent => crate::vm_wake::take_reached(args.condition.watch.as_deref()),
                ConditionKind::OperatorMessage => Vec::new(),
            };
            store.set_wait(args)?;
            Ok(wait_reply(store.state().condition.as_ref(), &already).to_string())
        }
        name => Err(format!("unknown attention tool: {name}").into()),
    }
}

/// What attention_wait answers. `already`: events that joined a turn as a
/// note since s0ma last waited and that the wait would have ended on; it ends
/// only on a new one now, which s0ma is told, since it may have meant to wait
/// for one.
fn wait_reply(condition: Option<&WaitCondition>, already: &[String]) -> serde_json::Value {
    let mut reply = json!({"ok":true,"mode":"waiting","condition":condition,
        "instruction":"Suspension is registered. Finish the current response without beginning another action."});
    if !already.is_empty() {
        let names = already.iter().map(|name| format!("\"{name}\"")).collect::<Vec<_>>().join(", ");
        reply["already_joined_your_turn"] = json!(already);
        reply["instruction"] = json!(format!(
            "Suspension is registered, but wake event {names} already joined your turn in a [VM WAKE \
             EVENT] note since you last waited, so this wait ends only on a new one. If that event is what you \
             were waiting for, act on it now (attention_orient keeps you engaged); otherwise finish the current \
             response without beginning another action."
        ));
    }
    reply
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    fn store() -> AttentionStore {
        let name = format!(
            "s0ma-attention-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        AttentionStore::load(&std::env::temp_dir().join(name)).unwrap()
    }
    #[test]
    fn only_a_wait_for_a_vm_event_holds_vm_events() {
        let mut store = store();
        assert!(!store.waits_for_vm_event(), "engaged");
        let wait = |kind, watch: Option<&str>| WaitArgs {
            activity: "waiting".into(),
            orientation: "until it happens".into(),
            condition: WaitCondition { kind, watch: watch.map(Into::into) },
        };
        store.set_wait(wait(ConditionKind::OperatorMessage, None)).unwrap();
        assert!(!store.waits_for_vm_event() && !store.is_recovery_wait(), "idle, as a loop stops");
        store.set_wait(wait(ConditionKind::VmEvent, Some("build"))).unwrap();
        assert!(store.waits_for_vm_event());
        store.set_wait(wait(ConditionKind::VmEvent, None)).unwrap();
        assert!(store.waits_for_vm_event());
    }
    #[test]
    fn a_failure_pause_is_recognised_across_a_restart() {
        let mut store = store();
        store.wait_for_operator_recovery("the router failed").unwrap();
        assert!(store.is_recovery_wait() && !store.waits_for_vm_event());
        let dir = store.path.parent().unwrap().to_owned();
        assert!(AttentionStore::load(&dir).unwrap().is_recovery_wait());
        store.resume("operator input").unwrap();
        assert!(!AttentionStore::load(&dir).unwrap().is_recovery_wait());
    }
    #[test]
    fn time_is_not_a_wait_condition() {
        // The invariant is that no attention tool accepts a *time* parameter.
        //
        // Check the parameter schemas, not the whole serialized definition. The
        // attention_wait description names duration precisely in order to say it
        // is unsupported, so scanning descriptions made the tool's own disclaimer
        // trip the test that exists to enforce it.
        for tool in tool_definitions() {
            let parameters = serde_json::to_string(&tool.function.parameters).unwrap();
            for forbidden in ["timestamp", "duration", "wake_at", "seconds", "timeout", "delay"] {
                assert!(
                    !parameters.contains(forbidden),
                    "{} takes a '{forbidden}' parameter; time is not a wait condition",
                    tool.function.name,
                );
            }
        }
    }
    #[test]
    fn wait_requires_a_condition() {
        let mut store = store();
        let call = ToolCall {
            id: "1".into(),
            kind: "function".into(),
            function: crate::types::FunctionCall {
                name: "attention_wait".into(),
                arguments: "{\"activity\":\"watching\",\"orientation\":\"notice completion\"}"
                    .into(),
            },
        };
        assert!(execute(&mut store, &call).is_err());
        assert!(store.is_engaged());
    }
    fn wait_call(condition: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "1".into(),
            kind: "function".into(),
            function: crate::types::FunctionCall {
                name: "attention_wait".into(),
                arguments: json!({"activity": "building", "orientation": "wait for the build", "condition": condition})
                    .to_string(),
            },
        }
    }
    #[test]
    fn a_vm_event_wait_matches_its_watch_or_any_event() {
        let mut store = store();
        assert!(!store.matches_vm("build"), "engaged attention waits for nothing");

        // Through the tool, vm_event needs the channel on (it is off in tests).
        let refused = execute(&mut store, &wait_call(json!({"kind": "vm_event"}))).unwrap_err().to_string();
        assert!(refused.contains("channel is off"), "{refused}");
        assert!(store.is_engaged());

        let vm_wait = |watch: Option<&str>| WaitArgs {
            activity: "building".into(),
            orientation: "wait for the build".into(),
            condition: WaitCondition { kind: ConditionKind::VmEvent, watch: watch.map(str::to_string) },
        };
        store.set_wait(vm_wait(Some("build"))).unwrap();
        assert!(store.matches_vm("build"));
        assert!(!store.matches_vm("tests"));

        store.set_wait(vm_wait(None)).unwrap();
        assert!(store.matches_vm("anything"));

        // Saved like any wait, and loadable whether or not the channel is on.
        // (After a restart, the operator's first input ends it.)
        let reloaded = AttentionStore::load(store.path.parent().unwrap()).unwrap();
        assert!(reloaded.matches_vm("anything"));

        store.resume("operator input").unwrap();
        assert!(!store.matches_vm("build"));
    }
    /// Writes a saved state with this condition and loads it.
    fn saved_with(condition: serde_json::Value) -> AttentionStore {
        let store = store();
        std::fs::create_dir_all(store.path.parent().unwrap()).unwrap();
        std::fs::write(
            &store.path,
            json!({"schema_version": 1, "mode": "waiting", "activity": "dinner", "orientation": "wait for it",
                   "condition": condition, "updated_at": "2026-10-10T00:00:00Z"}).to_string(),
        )
        .unwrap();
        AttentionStore::load(store.path.parent().unwrap()).unwrap()
    }
    #[test]
    fn a_saved_wait_of_a_retired_kind_loads_as_a_wait_for_operator_input() {
        for retired in [
            // A Signal contact's message, from before the bridge was removed.
            json!({"kind": "contact_message", "from": "Alex"}),
            // Minecraft waits, from before Minecraft events stopped waking s0ma.
            json!({"kind": "minecraft_player_chat", "username": "steve"}),
            json!({"kind": "minecraft_action_completed", "action_id": "walk-7"}),
            json!({"kind": "minecraft_action_failed"}),
            json!({"kind": "minecraft_event", "event_kind": "death"}),
        ] {
            let mut saved = saved_with(retired.clone());
            assert!(!saved.is_engaged(), "{retired}");
            assert_eq!(
                saved.state().condition,
                Some(WaitCondition { kind: ConditionKind::OperatorMessage, watch: None }),
                "{retired}"
            );
            assert_eq!(saved.state().activity, "dinner", "the rest of the state is kept");
            saved.resume("operator input").unwrap();
            assert!(saved.is_engaged());
        }
        // A stray retired field on a kind s0ma still has is dropped, not refused.
        let kept = saved_with(json!({"kind": "vm_event", "watch": "build", "from": "Alex", "username": "steve"}));
        assert!(kept.matches_vm("build"));
        let operator = saved_with(json!({"kind": "operator_message", "action_id": "walk-7"}));
        assert!(!operator.is_engaged());
    }
    #[test]
    fn a_wait_for_an_event_that_already_joined_the_turn_says_so() {
        let condition = WaitCondition { kind: ConditionKind::VmEvent, watch: Some("build".into()) };
        let plain = wait_reply(Some(&condition), &[]);
        assert!(plain.get("already_joined_your_turn").is_none());
        let warned = wait_reply(Some(&condition), &["build".to_string()]);
        assert_eq!(warned["already_joined_your_turn"], json!(["build"]));
        let instruction = warned["instruction"].as_str().unwrap();
        assert!(instruction.contains("\"build\" already joined your turn") && instruction.contains("ends only on a new one"), "{instruction}");
    }
    #[test]
    fn retired_kinds_are_not_kinds_s0ma_has_and_anything_else_unknown_still_fails() {
        for retired in RETIRED_KINDS {
            assert!(serde_json::from_value::<ConditionKind>(json!(retired)).is_err(), "{retired}");
        }
        for kept in ["operator_message", "vm_event"] {
            assert!(serde_json::from_value::<ConditionKind>(json!(kept)).is_ok(), "{kept}");
        }
        // An unknown kind or field fails loading, and says which file.
        for broken in [json!({"kind": "lunch"}), json!({"kind": "operator_message", "note": "x"})] {
            let store = store();
            std::fs::create_dir_all(store.path.parent().unwrap()).unwrap();
            std::fs::write(
                &store.path,
                json!({"schema_version": 1, "mode": "waiting", "activity": "a", "orientation": "b",
                       "condition": broken, "updated_at": "2026-10-10T00:00:00Z"}).to_string(),
            )
            .unwrap();
            let error = AttentionStore::load(store.path.parent().unwrap()).err().expect("refused").to_string();
            assert!(error.contains("attention.json"), "{error}");
        }
    }
    #[test]
    fn retired_kinds_and_filters_are_refused_as_new_waits() {
        let mut store = store();
        for gone in [
            json!({"kind": "minecraft_player_chat"}),
            json!({"kind": "minecraft_event", "event_kind": "death"}),
            json!({"kind": "contact_message"}),
            json!({"kind": "operator_message", "username": "steve"}),
            json!({"kind": "operator_message", "from": "Alex"}),
        ] {
            assert!(execute(&mut store, &wait_call(gone.clone())).is_err(), "{gone}");
            assert!(store.is_engaged(), "{gone}");
        }
        let schema = serde_json::to_string(&tool_definitions()).unwrap();
        assert!(!schema.contains("minecraft") && !schema.contains("contact"), "{schema}");
    }
    #[test]
    fn vm_event_filters_are_checked() {
        let mut store = store();
        for bad in [json!({"kind": "vm_event", "watch": "no spaces"}), json!({"kind": "operator_message", "watch": "build"})] {
            let condition: WaitCondition = serde_json::from_value(bad.clone()).unwrap();
            assert!(validate_condition(&condition).is_err(), "{bad}");
            assert!(execute(&mut store, &wait_call(bad.clone())).is_err(), "{bad}");
            assert!(store.is_engaged(), "{bad}");
        }
        // A Minecraft filter is no field at all now.
        assert!(serde_json::from_value::<WaitCondition>(json!({"kind": "vm_event", "username": "steve"})).is_err());
    }
}
