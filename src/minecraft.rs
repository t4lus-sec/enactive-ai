use crate::desktop::{DesktopExecution, DesktopObservation};
use crate::llm::ToolDefinition;
use crate::media;
use crate::types::ToolCall;
use crate::vm::{self, GuestCommandOutcome};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::error::Error;
use std::path::Path;

const DEFAULT_API_URL: &str = "http://127.0.0.1:31337/command";
const DEFAULT_VIEW_URL: &str = "http://127.0.0.1:3007";
const VIEW_WIDTH: u32 = 1280;
const VIEW_HEIGHT: u32 = 720;

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        function("minecraft_status", "Check whether s0ma's Minecraft body is connected and report its current action.", object(json!({}))),
        function("minecraft_observe", "Perceive a bounded semantic view around s0ma's Minecraft body: condition, position, inventory, nearby entities, salient blocks, recent events, and current action.", object(json!({"radius":{"type":"integer","minimum":2,"maximum":16,"default":10}}))),
        function("minecraft_go_to", "Begin navigating s0ma's Minecraft body near a world coordinate. The autonomic pathfinder continues after this call returns.", object(json!({"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"},"range":{"type":"integer","minimum":1,"maximum":8,"default":1}})).with_required(&["x","y","z"])),
        function("minecraft_look_at", "Turn s0ma's Minecraft body toward a coordinate.", object(json!({"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}})).with_required(&["x","y","z"])),
        function("minecraft_collect", "Begin finding and collecting a bounded quantity of one block type. Navigation, tool selection, digging, and pickup continue autonomically.", object(json!({"block":{"type":"string","maxLength":80},"count":{"type":"integer","minimum":1,"maximum":64}})).with_required(&["block","count"])),
        function("minecraft_break", "Break one exact loaded block within six blocks. Reports held/best tool and refuses a no-drop tool unless allow_wrong_tool is explicit.", object(json!({"x":{"type":"integer"},"y":{"type":"integer"},"z":{"type":"integer"},"allow_wrong_tool":{"type":"boolean","default":false}})).with_required(&["x","y","z"])),
        function("minecraft_break_many", "Break an ordered batch of up to 64 exact loaded blocks within reach. Returns tool assessments and records partial progress if the batch fails.", object(json!({"blocks":{"type":"array","minItems":1,"maxItems":64,"items":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"},"z":{"type":"integer"}},"required":["x","y","z"],"additionalProperties":false}},"allow_wrong_tool":{"type":"boolean","default":false}})).with_required(&["blocks"])),
        function("minecraft_craft", "Craft a bounded quantity of an item using the body's inventory and a nearby crafting table when required.", object(json!({"item":{"type":"string","maxLength":80},"count":{"type":"integer","minimum":1,"maximum":64}})).with_required(&["item","count"])),
        function("minecraft_equip", "Equip an inventory item in the requested destination.", object(json!({"item":{"type":"string","maxLength":80},"destination":{"type":"string","enum":["hand","off-hand","head","torso","legs","feet"],"default":"hand"}})).with_required(&["item"])),
        function("minecraft_place", "Place the held block against a referenced neighboring block face.", object(json!({"reference":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"},"z":{"type":"integer"}},"required":["x","y","z"],"additionalProperties":false},"face":{"type":"string","enum":["up","down","north","south","east","west"]}})).with_required(&["reference","face"])),
        function("minecraft_place_many", "Place the held blocks as an ordered batch of up to 64 reference-block and face pairs. Records partial progress if placement fails.", object(json!({"placements":{"type":"array","minItems":1,"maxItems":64,"items":{"type":"object","properties":{"reference":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"},"z":{"type":"integer"}},"required":["x","y","z"],"additionalProperties":false},"face":{"type":"string","enum":["up","down","north","south","east","west"]}},"required":["reference","face"],"additionalProperties":false}}})).with_required(&["placements"])),
        function("minecraft_interact", "Activate a block at a coordinate, such as a door, bed, button, chest, or crafting table.", object(json!({"x":{"type":"integer"},"y":{"type":"integer"},"z":{"type":"integer"}})).with_required(&["x","y","z"])),
        function("minecraft_chat", "Speak through Minecraft chat. This does not grant authority outside the local Minecraft environment.", object(json!({"message":{"type":"string","minLength":1,"maxLength":256}})).with_required(&["message"])),
        function("minecraft_stop", "Cancel the current autonomous Minecraft action and stop pathfinding.", object(json!({}))),
        function("minecraft_view", "Take an occasional transient first-person visual glimpse from s0ma's Minecraft body. Use for curiosity, orientation, aesthetics, surprises, and visual confirmation; semantic observation is cheaper for routine control.", object(json!({}))),
    ]
}

pub fn execute(call: &ToolCall) -> Result<DesktopExecution, Box<dyn Error>> {
    run(call).map_err(|error| -> Box<dyn Error> {
        if error.is::<crate::kill::Killed>() && call.function.name != "minecraft_view" {
            Box::new(crate::kill::Killed::new(
                "the bridge request was stopped; the bot may still be carrying out the action (minecraft_stop halts it)",
            ))
        } else {
            error
        }
    })
}

fn run(call: &ToolCall) -> Result<DesktopExecution, Box<dyn Error>> {
    if call.function.name == "minecraft_view" {
        return view();
    }

    let action = match call.function.name.as_str() {
        "minecraft_status" => "status",
        "minecraft_observe" => "observe",
        "minecraft_go_to" => "go_to",
        "minecraft_look_at" => "look_at",
        "minecraft_collect" => "collect",
        "minecraft_break" => "break",
        "minecraft_break_many" => "break_many",
        "minecraft_craft" => "craft",
        "minecraft_equip" => "equip",
        "minecraft_place" => "place",
        "minecraft_place_many" => "place_many",
        "minecraft_interact" => "interact",
        "minecraft_chat" => "chat",
        "minecraft_stop" => "stop",
        name => return Err(format!("unknown Minecraft tool: {name}").into()),
    };

    let mut request: Value = serde_json::from_str(&call.function.arguments)?;
    let object = request
        .as_object_mut()
        .ok_or("Minecraft tool arguments must be a JSON object")?;
    object.insert("action".to_string(), Value::String(action.to_string()));
    let payload = STANDARD.encode(serde_json::to_vec(&request)?);
    let api_url =
        std::env::var("S0MA_MINECRAFT_API_URL").unwrap_or_else(|_| DEFAULT_API_URL.to_string());
    let command = format!(
        "printf %s {} | base64 -d | curl --silent --show-error --fail-with-body --max-time 30 -H 'Content-Type: application/json' --data-binary @- {}",
        crate::util::shell_quote(&payload),
        crate::util::shell_quote(&api_url),
    );
    let result = run_vm(&command, 35)?;
    Ok(DesktopExecution {
        result: result.stdout.trim().to_string(),
        observation: None,
    })
}

fn view() -> Result<DesktopExecution, Box<dyn Error>> {
    let view_url =
        std::env::var("S0MA_MINECRAFT_VIEW_URL").unwrap_or_else(|_| DEFAULT_VIEW_URL.to_string());
    let display = std::env::var("S0MA_DESKTOP_DISPLAY").unwrap_or_else(|_| ":0".to_string());
    let xauthority = std::env::var("S0MA_DESKTOP_XAUTHORITY")
        .unwrap_or_else(|_| "/home/s0ma/.Xauthority".to_string());
    let command = format!(
        "set -eu; export DISPLAY={} XAUTHORITY={} HOME=/home/s0ma XDG_RUNTIME_DIR=/run/user/1001; if ! pgrep -f -- 's0ma-minecraft-viewer-profile' >/dev/null; then setsid -f chromium --no-sandbox --no-first-run --disable-dev-shm-usage --ignore-gpu-blocklist --enable-webgl --enable-unsafe-swiftshader --use-gl=angle --use-angle=swiftshader-webgl --disable-features=VaapiVideoDecoder,VaapiVideoEncoder --user-data-dir=/tmp/s0ma-minecraft-viewer-profile --app={} --start-maximized >/tmp/s0ma-minecraft-viewer.log 2>&1; sleep 5; fi; set -o pipefail; import -window root png:- | base64 -w0",
        crate::util::shell_quote(&display),
        crate::util::shell_quote(&xauthority),
        crate::util::shell_quote(&view_url),
    );
    let result = run_vm(&command, 15)?;
    if result.stdout_truncated {
        return Err("Minecraft view exceeded the capture limit".into());
    }
    let bytes = STANDARD.decode(result.stdout.trim())?;
    if bytes.is_empty() {
        return Err("Minecraft view returned an empty image".into());
    }
    let fingerprint = crate::util::fnv1a64(&bytes);
    let data_url = media::image_data_url(Path::new("minecraft-view.png"), &bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    Ok(DesktopExecution {
        result: json!({
            "ok": true,
            "width": VIEW_WIDTH,
            "height": VIEW_HEIGHT,
            "fingerprint": format!("fnv1a64:{fingerprint:016x}"),
            "image_attached": true,
            "transient": true,
        })
        .to_string(),
        observation: Some(DesktopObservation {
            content: format!(
                "[MINECRAFT VISUAL GLIMPSE]\nfirst_person=true\ndimensions={VIEW_WIDTH}x{VIEW_HEIGHT}\nframe_fingerprint=fnv1a64:{fingerprint:016x}\nThis transient image is a situated visual view from s0ma's Minecraft body. It is not a complete semantic account of the surrounding world."
            ),
            data_url,
        }),
    })
}

fn run_vm(command: &str, timeout_secs: u64) -> Result<vm::GuestCommandResult, Box<dyn Error>> {
    match vm::run_desktop_command(command, timeout_secs)? {
        GuestCommandOutcome::Exited(result) if result.exit_code == Some(0) => Ok(result),
        GuestCommandOutcome::Exited(result) => Err(format!(
            "Minecraft bridge command failed (exit={:?}, signal={:?}): {}{}",
            result.exit_code,
            result.signal,
            result.stderr.trim(),
            if result.stdout.trim().is_empty() {
                String::new()
            } else {
                format!("; response: {}", result.stdout.trim())
            }
        )
        .into()),
        GuestCommandOutcome::HostTimeout {
            pid, timeout_secs, ..
        } => Err(format!("Minecraft bridge command {pid} exceeded {timeout_secs} seconds").into()),
    }
}

fn function(name: &str, description: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition::function(name, description, parameters)
}

fn object(properties: Value) -> Value {
    json!({"type":"object","properties":properties,"additionalProperties":false})
}

trait RequiredSchema {
    fn with_required(self, fields: &[&str]) -> Value;
}

impl RequiredSchema for Value {
    fn with_required(mut self, fields: &[&str]) -> Value {
        self["required"] = json!(fields);
        self
    }
}


