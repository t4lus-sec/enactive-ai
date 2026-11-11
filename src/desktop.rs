use crate::llm::ToolDefinition;
use crate::media;
use crate::types::ToolCall;
use crate::vm::{self, GuestCommandOutcome};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Duration;

const DEFAULT_DISPLAY: &str = ":100";
const DEFAULT_VM_DISPLAY: &str = ":0";
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const MAX_TYPE_CHARS: usize = 8_192;
const MAX_WAIT_MS: u64 = 30_000;
const MAX_RELATIVE_MOUSE_DELTA: i32 = 4_096;

pub struct DesktopController {
    backend: DesktopBackend,
    display: String,
    xauthority: PathBuf,
    width: u32,
    height: u32,
    last_frame: Option<u64>,
    held_keys: BTreeSet<String>,
    held_buttons: BTreeSet<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DesktopBackend {
    Vm,
    Xephyr,
}

pub struct DesktopExecution {
    pub result: String,
    pub observation: Option<DesktopObservation>,
}

pub struct DesktopObservation {
    pub content: String,
    pub data_url: String,
}

impl DesktopController {
    pub fn new(data_dir: &Path) -> Result<Self, Box<dyn Error>> {
        let state_dir = data_dir.join("desktop");
        fs::create_dir_all(&state_dir)?;

        let backend = match std::env::var("S0MA_DESKTOP_BACKEND")
            .unwrap_or_else(|_| "vm".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "vm" => DesktopBackend::Vm,
            "xephyr" | "host" => DesktopBackend::Xephyr,
            value => return Err(format!("unsupported S0MA_DESKTOP_BACKEND {value:?}; use vm or xephyr").into()),
        };
        let default_display = match backend {
            DesktopBackend::Vm => DEFAULT_VM_DISPLAY,
            DesktopBackend::Xephyr => DEFAULT_DISPLAY,
        };
        let display = std::env::var("S0MA_DESKTOP_DISPLAY")
            .unwrap_or_else(|_| default_display.to_string());
        let xauthority = std::env::var_os("S0MA_DESKTOP_XAUTHORITY")
            .map(PathBuf::from)
            .unwrap_or_else(|| match backend {
                DesktopBackend::Vm => PathBuf::from("/home/s0ma/.Xauthority"),
                DesktopBackend::Xephyr => state_dir.join("Xauthority"),
            });
        let width = env_dimension("S0MA_DESKTOP_WIDTH", DEFAULT_WIDTH);
        let height = env_dimension("S0MA_DESKTOP_HEIGHT", DEFAULT_HEIGHT);

        Ok(Self {
            backend,
            display,
            xauthority,
            width,
            height,
            last_frame: None,
            held_keys: BTreeSet::new(),
            held_buttons: BTreeSet::new(),
        })
    }

    pub fn execute(&mut self, call: &ToolCall) -> Result<DesktopExecution, Box<dyn Error>> {
        match call.function.name.as_str() {
            "desktop_view" => self.view(),
            "desktop_click" => self.click(&call.function.arguments),
            "desktop_button" => self.button(&call.function.arguments),
            "desktop_move" => self.move_pointer(&call.function.arguments),
            "desktop_drag" => self.drag(&call.function.arguments),
            "desktop_look" => self.look(&call.function.arguments),
            "desktop_key" => self.key(&call.function.arguments),
            "desktop_type" => self.type_text(&call.function.arguments),
            "desktop_wait" => self.wait(&call.function.arguments),
            "desktop_release_all" => self.release_all(),
            "desktop_status" => self.status(),
            "desktop_launch" => self.launch(&call.function.arguments),
            name => Ok(text_result(json!({"ok": false, "error": format!("unknown desktop tool: {name}")}))),
        }
    }

    fn view(&mut self) -> Result<DesktopExecution, Box<dyn Error>> {
        let bytes = match self.backend {
            DesktopBackend::Xephyr => {
                let mut command = self.host_command("import");
                command.args(["-window", "root", "png:-"]);
                let output = crate::kill::output(command)?;
                require_success("capture desktop", &output)?;
                output.stdout
            }
            DesktopBackend::Vm => {
                let command = format!(
                    "set -o pipefail; env DISPLAY={} XAUTHORITY={} import -window root png:- | base64 -w0",
                    crate::util::shell_quote(&self.display),
                    crate::util::shell_quote(&self.xauthority.to_string_lossy()),
                );
                let result = self.run_vm(&command, 30)?;
                if result.stdout_truncated {
                    return Err("desktop frame exceeded the capture limit".into());
                }
                STANDARD.decode(result.stdout.trim())?
            }
        };

        if bytes.is_empty() {
            return Err("desktop capture returned an empty image".into());
        }

        if let Some((width, height)) = png_dimensions(&bytes) {
            self.width = width;
            self.height = height;
        }

        let fingerprint = crate::util::fnv1a64(&bytes);
        let unchanged = self.last_frame == Some(fingerprint);
        self.last_frame = Some(fingerprint);
        let data_url = media::image_data_url(Path::new("desktop.png"), &bytes)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let content = format!(
            "[DESKTOP OBSERVATION]\n\
             display={}\n\
             dimensions={}x{}\n\
             frame_fingerprint=fnv1a64:{fingerprint:016x}\n\
             unchanged_from_previous={}\n\
             This is a transient view of s0ma's dedicated desktop. Pixel coordinates use this image's dimensions.",
            self.display, self.width, self.height, unchanged,
        );

        Ok(DesktopExecution {
            result: json!({
                "ok": true,
                "backend": self.backend_label(),
                "display": self.display,
                "width": self.width,
                "height": self.height,
                "fingerprint": format!("fnv1a64:{fingerprint:016x}"),
                "unchanged": unchanged,
                "image_attached": true
            }).to_string(),
            observation: Some(DesktopObservation { content, data_url }),
        })
    }

    fn click(&mut self, arguments: &str) -> Result<DesktopExecution, Box<dyn Error>> {
        let args: PointButtonArgs = serde_json::from_str(arguments)?;
        self.validate_point(args.x, args.y)?;
        if !(1..=5).contains(&args.button) {
            return Err("button must be between 1 and 5".into());
        }
        self.run_xdotool(&["mousemove", &args.x.to_string(), &args.y.to_string()])?;
        self.run_xdotool(&["click", &args.button.to_string()])?;
        Ok(ok_result(json!({"x": args.x, "y": args.y, "button": args.button})))
    }

    fn button(&mut self, arguments: &str) -> Result<DesktopExecution, Box<dyn Error>> {
        let args: ButtonArgs = serde_json::from_str(arguments)?;
        if !(1..=3).contains(&args.button) {
            return Err("button must be between 1 and 3".into());
        }
        let button = args.button.to_string();
        match args.action.as_str() {
            "down" => {
                if self.held_buttons.insert(args.button)
                    && let Err(error) = self.run_xdotool(&["mousedown", &button]) {
                        self.held_buttons.remove(&args.button);
                        return Err(error);
                    }
            }
            "up" => {
                self.run_xdotool(&["mouseup", &button])?;
                self.held_buttons.remove(&args.button);
            }
            _ => return Err("action must be down or up".into()),
        }
        Ok(ok_result(json!({
            "button": args.button,
            "action": args.action,
            "held_buttons": self.held_buttons,
        })))
    }

    fn move_pointer(&mut self, arguments: &str) -> Result<DesktopExecution, Box<dyn Error>> {
        let args: PointArgs = serde_json::from_str(arguments)?;
        self.validate_point(args.x, args.y)?;
        self.run_xdotool(&["mousemove", &args.x.to_string(), &args.y.to_string()])?;
        Ok(ok_result(json!({"x": args.x, "y": args.y})))
    }

    fn drag(&mut self, arguments: &str) -> Result<DesktopExecution, Box<dyn Error>> {
        let args: DragArgs = serde_json::from_str(arguments)?;
        self.validate_point(args.from_x, args.from_y)?;
        self.validate_point(args.to_x, args.to_y)?;
        if !(1..=3).contains(&args.button) {
            return Err("drag button must be between 1 and 3".into());
        }
        let from_x = args.from_x.to_string();
        let from_y = args.from_y.to_string();
        let to_x = args.to_x.to_string();
        let to_y = args.to_y.to_string();
        let button = args.button.to_string();
        self.run_xdotool(&["mousemove", &from_x, &from_y])?;
        self.run_xdotool(&["mousedown", &button])?;
        self.held_buttons.insert(args.button);
        let movement = self.run_xdotool(&["mousemove", "--sync", &to_x, &to_y]);
        // Even when the move was stopped, the button must come back up.
        let release = {
            let _cleanup = crate::kill::suspend();
            self.run_xdotool(&["mouseup", &button])
        };
        self.held_buttons.remove(&args.button);
        movement?;
        release?;

        Ok(ok_result(json!({
            "from": {"x": args.from_x, "y": args.from_y},
            "to": {"x": args.to_x, "y": args.to_y},
            "button": args.button,
        })))
    }

    fn look(&mut self, arguments: &str) -> Result<DesktopExecution, Box<dyn Error>> {
        let args: LookArgs = serde_json::from_str(arguments)?;
        if args.dx.unsigned_abs() > MAX_RELATIVE_MOUSE_DELTA as u32
            || args.dy.unsigned_abs() > MAX_RELATIVE_MOUSE_DELTA as u32
        {
            return Err(format!(
                "relative mouse deltas must be between -{MAX_RELATIVE_MOUSE_DELTA} and {MAX_RELATIVE_MOUSE_DELTA}"
            )
            .into());
        }
        let dx = args.dx.to_string();
        let dy = args.dy.to_string();
        // Pointer-locked applications consume relative input events. X11 cursor
        // warps fight their recentering logic and can produce runaway rotation,
        // so use the uinput-backed virtual mouse for embodied camera movement.
        self.run_program("ydotool", &["mousemove", "--", &dx, &dy])?;
        Ok(ok_result(json!({"dx": args.dx, "dy": args.dy})))
    }

    fn key(&mut self, arguments: &str) -> Result<DesktopExecution, Box<dyn Error>> {
        let args: KeyArgs = serde_json::from_str(arguments)?;
        validate_key(&args.key)?;
        match args.action.as_str() {
            "tap" => self.run_xdotool(&["key", &args.key])?,
            "down" => {
                self.run_xdotool(&["keydown", &args.key])?;
                self.held_keys.insert(args.key.clone());
            }
            "up" => {
                self.run_xdotool(&["keyup", &args.key])?;
                self.held_keys.remove(&args.key);
            }
            _ => return Err("action must be tap, down, or up".into()),
        }
        Ok(ok_result(json!({"key": args.key, "action": args.action})))
    }

    fn type_text(&mut self, arguments: &str) -> Result<DesktopExecution, Box<dyn Error>> {
        let args: TypeArgs = serde_json::from_str(arguments)?;
        if args.text.chars().count() > MAX_TYPE_CHARS {
            return Err(format!("text exceeds {MAX_TYPE_CHARS} characters").into());
        }
        if args.delay_ms > 1_000 {
            return Err("delay_ms must be at most 1000".into());
        }
        let delay = args.delay_ms.to_string();
        self.run_xdotool(&["type", "--clearmodifiers", "--delay", &delay, "--", &args.text])?;
        Ok(ok_result(json!({"characters": args.text.chars().count()})))
    }

    fn wait(&mut self, arguments: &str) -> Result<DesktopExecution, Box<dyn Error>> {
        let args: WaitArgs = serde_json::from_str(arguments)?;
        if args.milliseconds > MAX_WAIT_MS {
            return Err(format!("milliseconds must be at most {MAX_WAIT_MS}").into());
        }
        crate::kill::sleep(Duration::from_millis(args.milliseconds))?;
        Ok(ok_result(json!({"waited_ms": args.milliseconds})))
    }

    fn release_all(&mut self) -> Result<DesktopExecution, Box<dyn Error>> {
        let keys: Vec<String> = self.held_keys.iter().cloned().collect();
        let mut failures = Vec::new();
        for key in &keys {
            if let Err(error) = self.run_xdotool(&["keyup", key]) {
                failures.push(format!("{key}: {error}"));
            }
        }
        self.held_keys.clear();
        let buttons: Vec<u8> = self.held_buttons.iter().copied().collect();
        for button in &buttons {
            if let Err(error) = self.run_xdotool(&["mouseup", &button.to_string()]) {
                failures.push(format!("mouse button {button}: {error}"));
            }
        }
        self.held_buttons.clear();
        if failures.is_empty() {
            Ok(ok_result(json!({"released_keys": keys, "released_buttons": buttons})))
        } else {
            Err(format!("failed to release some keys: {}", failures.join(", ")).into())
        }
    }

    fn status(&mut self) -> Result<DesktopExecution, Box<dyn Error>> {
        let display_info = self.run_program("xdpyinfo", &[]);
        let available = display_info.is_ok();
        if let Ok(info) = &display_info
            && let Some((width, height)) = parse_xdpyinfo_dimensions(info) {
                self.width = width;
                self.height = height;
            }
        let active_window = if available {
            self.run_program("xdotool", &["getactivewindow", "getwindowname"])
                .ok()
                .map(|output| output.trim().to_string())
        } else {
            None
        };

        Ok(text_result(json!({
            "ok": true,
            "backend": self.backend_label(),
            "available": available,
            "display": self.display,
            "width": self.width,
            "height": self.height,
            "xauthority": self.xauthority,
            "active_window": active_window,
            "held_keys": self.held_keys,
            "held_buttons": self.held_buttons,
        })))
    }

    fn launch(&mut self, arguments: &str) -> Result<DesktopExecution, Box<dyn Error>> {
        let args: LaunchArgs = serde_json::from_str(arguments)?;
        let program = match args.application.as_str() {
            "terminal" => "xterm",
            "minecraft" => "prismlauncher",
            _ => return Err("application must be terminal or minecraft".into()),
        };
        let pid = match self.backend {
            DesktopBackend::Xephyr => self.host_command(program).spawn()?.id(),
            DesktopBackend::Vm => {
                let command = format!(
                    "env DISPLAY={} XAUTHORITY={} nohup {} </dev/null >/dev/null 2>&1 & printf %s \"$!\"",
                    crate::util::shell_quote(&self.display),
                    crate::util::shell_quote(&self.xauthority.to_string_lossy()),
                    crate::util::shell_quote(program),
                );
                self.run_vm(&command, 10)?.stdout.trim().parse::<u32>()?
            }
        };
        Ok(ok_result(json!({"application": args.application, "pid": pid})))
    }

    fn validate_point(&self, x: u32, y: u32) -> Result<(), Box<dyn Error>> {
        if x >= self.width || y >= self.height {
            return Err(format!("point ({x}, {y}) is outside {}x{} desktop", self.width, self.height).into());
        }
        Ok(())
    }

    fn run_xdotool(&self, args: &[&str]) -> Result<(), Box<dyn Error>> {
        self.run_program("xdotool", args).map(|_| ())
    }

    fn run_program(&self, program: &str, args: &[&str]) -> Result<String, Box<dyn Error>> {
        match self.backend {
            DesktopBackend::Xephyr => {
                let mut command = self.host_command(program);
                command.args(args);
                let output = crate::kill::output(command)?;
                require_success(program, &output)?;
                Ok(String::from_utf8_lossy(&output.stdout).into_owned())
            }
            DesktopBackend::Vm => {
                let mut words = vec![
                    "env".to_string(),
                    format!("DISPLAY={}", self.display),
                    format!("XAUTHORITY={}", self.xauthority.display()),
                    program.to_string(),
                ];
                words.extend(args.iter().map(|argument| (*argument).to_string()));
                let command = words.iter().map(|word| crate::util::shell_quote(word)).collect::<Vec<_>>().join(" ");
                Ok(self.run_vm(&command, 30)?.stdout)
            }
        }
    }

    fn run_vm(&self, command: &str, timeout_secs: u64) -> Result<vm::GuestCommandResult, Box<dyn Error>> {
        match vm::run_desktop_command(command, timeout_secs)? {
            GuestCommandOutcome::Exited(result) if result.exit_code == Some(0) => Ok(result),
            GuestCommandOutcome::Exited(result) => Err(format!(
                "desktop command failed (exit={:?}, signal={:?}): {}",
                result.exit_code, result.signal, result.stderr.trim()
            ).into()),
            GuestCommandOutcome::HostTimeout { pid, timeout_secs, .. } => Err(format!(
                "desktop command {pid} exceeded {timeout_secs} seconds"
            ).into()),
        }
    }

    fn backend_label(&self) -> &'static str {
        match self.backend {
            DesktopBackend::Vm => "vm",
            DesktopBackend::Xephyr => "xephyr",
        }
    }

    fn host_command(&self, program: &str) -> Command {
        let mut command = Command::new(program);
        command.env("DISPLAY", &self.display);
        command.env("XAUTHORITY", &self.xauthority);
        command
    }
}

impl Drop for DesktopController {
    fn drop(&mut self) {
        let keys: Vec<String> = self.held_keys.iter().cloned().collect();
        for key in keys {
            let _ = self.run_xdotool(&["keyup", &key]);
        }
        let buttons: Vec<u8> = self.held_buttons.iter().copied().collect();
        for button in buttons {
            let _ = self.run_xdotool(&["mouseup", &button.to_string()]);
        }
    }
}

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::function("desktop_view", "See s0ma's dedicated desktop. Returns geometry and attaches one transient screenshot to the next model pass. Call this after actions when visual confirmation matters.", object(json!({}))),
        ToolDefinition::function("desktop_click", "Move the pointer and click at absolute screenshot coordinates on s0ma's dedicated desktop.", object(json!({"x":{"type":"integer","minimum":0},"y":{"type":"integer","minimum":0},"button":{"type":"integer","minimum":1,"maximum":5,"description":"1 left, 2 middle, 3 right, 4 wheel up, 5 wheel down"}})).with_required(&["x","y","button"])),
        ToolDefinition::function("desktop_button", "Hold or release a mouse button at the current pointer position. Use down before movement or looking and up afterward. Held buttons are tracked and desktop_release_all releases them after interrupted actions.", object(json!({"button":{"type":"integer","minimum":1,"maximum":3,"description":"1 left, 2 middle, 3 right"},"action":{"type":"string","enum":["down","up"]}})).with_required(&["button","action"])),
        ToolDefinition::function("desktop_move", "Move the pointer to absolute screenshot coordinates without clicking.", object(json!({"x":{"type":"integer","minimum":0},"y":{"type":"integer","minimum":0}})).with_required(&["x","y"])),
        ToolDefinition::function("desktop_drag", "Drag between two absolute screenshot coordinates while holding a mouse button. Use for sliders, scrollbars, files, and other desktop UI elements; do not use this for Minecraft camera movement.", object(json!({"from_x":{"type":"integer","minimum":0},"from_y":{"type":"integer","minimum":0},"to_x":{"type":"integer","minimum":0},"to_y":{"type":"integer","minimum":0},"button":{"type":"integer","minimum":1,"maximum":3,"default":1}})).with_required(&["from_x","from_y","to_x","to_y","button"])),
        ToolDefinition::function("desktop_look", "Send relative motion through the virtual mouse. Use while an application such as Minecraft has captured the pointer. Positive dx turns right and positive dy looks down. Start with small changes and take a new desktop view afterward.", object(json!({"dx":{"type":"integer","minimum":-4096,"maximum":4096},"dy":{"type":"integer","minimum":-4096,"maximum":4096}})).with_required(&["dx","dy"])),
        ToolDefinition::function("desktop_key", "Tap, hold, or release one keyboard key on s0ma's desktop. Held keys are tracked and can be cleared with desktop_release_all.", object(json!({"key":{"type":"string","description":"X11 key name, such as Return, Escape, w, or ctrl+l"},"action":{"type":"string","enum":["tap","down","up"]}})).with_required(&["key","action"])),
        ToolDefinition::function("desktop_type", "Type literal text into the focused desktop application.", object(json!({"text":{"type":"string","maxLength":8192},"delay_ms":{"type":"integer","minimum":0,"maximum":1000,"default":10}})).with_required(&["text"])),
        ToolDefinition::function("desktop_wait", "Wait briefly for a desktop application or animation. This does not take a screenshot.", object(json!({"milliseconds":{"type":"integer","minimum":0,"maximum":30000}})).with_required(&["milliseconds"])),
        ToolDefinition::function("desktop_release_all", "Release every keyboard key held through desktop_key. Use after interrupted movement or when input appears stuck.", object(json!({}))),
        ToolDefinition::function("desktop_status", "Check whether s0ma's dedicated desktop is reachable and report its active window and geometry.", object(json!({}))),
        ToolDefinition::function("desktop_launch", "Launch an allowlisted application on s0ma's dedicated desktop. This grants no shell access and accepts no arbitrary command.", object(json!({"application":{"type":"string","enum":["terminal","minecraft"]}})).with_required(&["application"])),
    ]
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

fn object(properties: Value) -> Value {
    json!({"type":"object","properties":properties,"additionalProperties":false})
}

fn ok_result(extra: Value) -> DesktopExecution {
    let mut value = json!({"ok": true});
    if let (Some(target), Some(source)) = (value.as_object_mut(), extra.as_object()) {
        target.extend(source.clone());
    }
    text_result(value)
}

fn text_result(value: Value) -> DesktopExecution {
    DesktopExecution { result: value.to_string(), observation: None }
}

fn require_success(operation: &str, output: &Output) -> Result<(), Box<dyn Error>> {
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(format!("{operation} failed: {}", stderr.trim()).into())
}

fn validate_key(key: &str) -> Result<(), Box<dyn Error>> {
    if key.is_empty() || key.len() > 64 || !key.chars().all(|c| c.is_ascii_alphanumeric() || "_+-:".contains(c)) {
        return Err("key contains unsupported characters".into());
    }
    Ok(())
}

fn env_dimension(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).filter(|value| *value > 0).unwrap_or(default)
}



fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return None;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    (width > 0 && height > 0).then_some((width, height))
}

fn parse_xdpyinfo_dimensions(output: &str) -> Option<(u32, u32)> {
    let dimensions = output
        .lines()
        .find_map(|line| line.trim().strip_prefix("dimensions:"))?
        .split_whitespace()
        .next()?;
    let (width, height) = dimensions.split_once('x')?;
    Some((width.parse().ok()?, height.parse().ok()?))
}

#[derive(Deserialize)]
struct PointArgs { x: u32, y: u32 }

#[derive(Deserialize)]
struct PointButtonArgs { x: u32, y: u32, #[serde(default = "left_button")] button: u8 }

fn left_button() -> u8 { 1 }

#[derive(Deserialize)]
struct ButtonArgs { button: u8, action: String }

#[derive(Deserialize)]
struct DragArgs {
    from_x: u32,
    from_y: u32,
    to_x: u32,
    to_y: u32,
    #[serde(default = "left_button")]
    button: u8,
}

#[derive(Deserialize)]
struct LookArgs { dx: i32, dy: i32 }

#[derive(Deserialize)]
struct KeyArgs { key: String, action: String }

#[derive(Deserialize)]
struct TypeArgs { text: String, #[serde(default = "default_type_delay")] delay_ms: u64 }

fn default_type_delay() -> u64 { 10 }

#[derive(Deserialize)]
struct WaitArgs { milliseconds: u64 }

#[derive(Deserialize)]
struct LaunchArgs { application: String }

#[cfg(test)]
mod tests {
    use super::{parse_xdpyinfo_dimensions, png_dimensions};

    #[test]
    fn reads_png_dimensions_from_ihdr() {
        let mut bytes = b"\x89PNG\r\n\x1a\n00000000".to_vec();
        bytes.extend_from_slice(&1280_u32.to_be_bytes());
        bytes.extend_from_slice(&720_u32.to_be_bytes());
        assert_eq!(png_dimensions(&bytes), Some((1280, 720)));
    }

    #[test]
    fn reads_xdpyinfo_dimensions() {
        let output = "screen #0:\n  dimensions:    1920x1080 pixels (508x285 millimeters)\n";
        assert_eq!(parse_xdpyinfo_dimensions(output), Some((1920, 1080)));
    }

    #[test]
    fn shell_quote_handles_apostrophes() {
        // One word, the apostrophe closed, escaped and reopened (util.rs).
        assert_eq!(crate::util::shell_quote("The operator's desktop"), r"'The operator'\''s desktop'");
    }
}
