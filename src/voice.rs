use crate::{input::ExternalSubmitter, llm::ToolDefinition, types::ToolCall};

use serde::Deserialize;
use serde_json::json;

use std::{
    env,
    error::Error,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
};

const VOICE_INPUT_PREFIX: &str = "[VOICE INPUT]\n\n";
const MAX_SPEECH_CHARS: usize = 4_000;

#[derive(Debug, Deserialize)]
struct BridgeEvent {
    #[serde(rename = "type")]
    kind: String,

    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpeakArgs {
    text: String,
}

#[derive(Clone)]
pub struct VoiceSpeaker {
    shared: Arc<VoiceShared>,
}

struct VoiceShared {
    bridge_stdin: Mutex<Option<ChildStdin>>,
    enabled: AtomicBool,
    generation: AtomicU64,
    tts_configured: bool,
}

pub struct VoiceRuntime {
    python: String,
    bridge_path: PathBuf,
    child: Option<Child>,
    operator_submitter: ExternalSubmitter,
    shared: Arc<VoiceShared>,
}

impl VoiceRuntime {
    pub fn new(root: &Path, operator_submitter: ExternalSubmitter) -> Result<Self, Box<dyn Error>> {
        let python = env::var("S0MA_VOICE_PYTHON").unwrap_or_else(|_| "python3".to_string());

        let bridge_path = env::var_os("S0MA_VOICE_BRIDGE")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("scripts/voice_bridge.py"));

        let shared = Arc::new(VoiceShared {
            bridge_stdin: Mutex::new(None),
            enabled: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            tts_configured: env::var_os("S0MA_PIPER_MODEL").is_some(),
        });

        let mut runtime = Self {
            python,
            bridge_path,
            child: None,
            operator_submitter,
            shared,
        };

        if env_flag("S0MA_VOICE_ENABLED") {
            runtime.enable()?;
        }

        Ok(runtime)
    }

    pub fn speaker(&self) -> VoiceSpeaker {
        VoiceSpeaker {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn handle_command(&mut self, input: &str) -> Option<Result<String, String>> {
        let mut parts = input.split_whitespace();

        if parts.next()? != "/voice" {
            return None;
        }

        let argument = parts.next().unwrap_or("status");

        if parts.next().is_some() {
            return Some(Err("usage: /voice [on|off|status]".to_string()));
        }

        let result = match argument {
            "on" => self
                .enable()
                .map(|_| "[voice] listening".to_string())
                .map_err(|error| format!("[voice] failed to start: {error}")),

            "off" => self
                .disable()
                .map(|_| "[voice] disabled".to_string())
                .map_err(|error| format!("[voice] failed to stop: {error}")),

            "status" => Ok(self.status()),

            _ => Err("usage: /voice [on|off|status]".to_string()),
        };

        Some(result)
    }

    fn enable(&mut self) -> Result<(), Box<dyn Error>> {
        self.refresh_child_state()?;

        if self.shared.enabled.load(Ordering::SeqCst) {
            return Ok(());
        }

        if !self.bridge_path.is_file() {
            return Err(format!(
                "voice bridge does not exist: {}",
                self.bridge_path.display()
            )
            .into());
        }

        let mut child = Command::new(&self.python)
            .arg("-u")
            .arg(&self.bridge_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or("voice bridge did not expose stdin")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("voice bridge did not expose stdout")?;

        let generation = self.shared.generation.fetch_add(1, Ordering::SeqCst) + 1;

        {
            let mut bridge_stdin = self
                .shared
                .bridge_stdin
                .lock()
                .map_err(|_| "voice bridge stdin lock poisoned")?;
            *bridge_stdin = Some(stdin);
        }

        self.shared.enabled.store(true, Ordering::SeqCst);

        let operator_submitter = self.operator_submitter.clone();
        let shared = Arc::clone(&self.shared);

        thread::spawn(move || {
            read_bridge_events(stdout, operator_submitter);

            // A previous bridge reader may reach EOF just after a replacement
            // bridge has started. Only the reader that owns the current
            // generation may clear its connection.
            if shared.generation.load(Ordering::SeqCst) != generation {
                return;
            }

            shared.enabled.store(false, Ordering::SeqCst);

            if let Ok(mut stdin) = shared.bridge_stdin.lock() {
                *stdin = None;
            }
        });

        self.child = Some(child);

        Ok(())
    }

    fn disable(&mut self) -> Result<(), Box<dyn Error>> {
        if self.shared.enabled.load(Ordering::SeqCst) {
            let _ = send_bridge_command(&self.shared, json!({ "type": "shutdown" }));
        }

        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        self.shared.enabled.store(false, Ordering::SeqCst);

        if let Ok(mut stdin) = self.shared.bridge_stdin.lock() {
            *stdin = None;
        }

        if let Some(mut child) = self.child.take() {
            if child.try_wait()?.is_none() {
                // The bridge may finish between try_wait and kill after reading the
                // shutdown command. Treat that race as an already-clean shutdown.
                let _ = child.kill();
            }
            let _ = child.wait();
        }

        Ok(())
    }

    fn refresh_child_state(&mut self) -> Result<(), Box<dyn Error>> {
        let finished = match self.child.as_mut() {
            Some(child) => child.try_wait()?.is_some(),
            None => false,
        };

        if finished {
            self.child = None;
            self.shared.generation.fetch_add(1, Ordering::SeqCst);
            self.shared.enabled.store(false, Ordering::SeqCst);

            if let Ok(mut stdin) = self.shared.bridge_stdin.lock() {
                *stdin = None;
            }
        }

        Ok(())
    }

    /// Listening now (for the overview, without asking the bridge).
    pub fn listening(&self) -> bool {
        self.shared.enabled.load(Ordering::SeqCst)
    }

    pub fn status(&mut self) -> String {
        let _ = self.refresh_child_state();

        let listening = self.shared.enabled.load(Ordering::SeqCst);
        let tts = if self.shared.tts_configured {
            "configured"
        } else {
            "missing S0MA_PIPER_MODEL"
        };

        format!(
            "[voice] listening={} tts={} bridge={}",
            listening,
            tts,
            self.bridge_path.display()
        )
    }
}

impl Drop for VoiceRuntime {
    fn drop(&mut self) {
        let _ = self.disable();
    }
}

impl VoiceSpeaker {
    pub fn speak(&self, text: &str) -> Result<(), Box<dyn Error>> {
        let text = text.trim();
        let length = text.chars().count();

        if !(1..=MAX_SPEECH_CHARS).contains(&length) {
            return Err(
                format!("voice_speak text must contain 1..={MAX_SPEECH_CHARS} characters").into(),
            );
        }

        if !self.shared.enabled.load(Ordering::SeqCst) {
            return Err("voice is disabled; the operator can enable it with /voice on".into());
        }

        if !self.shared.tts_configured {
            return Err("voice output requires S0MA_PIPER_MODEL".into());
        }

        send_bridge_command(
            &self.shared,
            json!({
                "type": "speak",
                "text": text,
            }),
        )
    }
}

pub fn command_text(input: &str) -> &str {
    input.strip_prefix(VOICE_INPUT_PREFIX).unwrap_or(input)
}

pub fn is_voice_input(input: &str) -> bool {
    input.starts_with(VOICE_INPUT_PREFIX)
}

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![ToolDefinition::function(
        "voice_speak",
        "Speak text aloud through s0ma's local voice bridge. Use this intentionally when audible speech belongs in the interaction, especially when replying to input marked [VOICE INPUT]. Do not narrate thinking or tool status, and do not repeat the same wording in the final assistant response.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SPEECH_CHARS,
                    "description": "The exact natural-language utterance to speak aloud"
                }
            },
            "required": ["text"]
        }),
    )]
}

pub fn execute(speaker: &VoiceSpeaker, call: &ToolCall) -> Result<String, Box<dyn Error>> {
    match call.function.name.as_str() {
        "voice_speak" => {
            let args: SpeakArgs = serde_json::from_str(&call.function.arguments)?;
            speaker.speak(&args.text)?;

            Ok(json!({
                "ok": true,
                "queued": true,
                "characters": args.text.chars().count()
            })
            .to_string())
        }

        name => Err(format!("unknown voice tool: {name}").into()),
    }
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn tag_voice_input(text: String) -> String {
    format!("{VOICE_INPUT_PREFIX}{}", text.trim())
}

fn send_bridge_command(
    shared: &VoiceShared,
    command: serde_json::Value,
) -> Result<(), Box<dyn Error>> {
    let mut bridge_stdin = shared
        .bridge_stdin
        .lock()
        .map_err(|_| "voice bridge stdin lock poisoned")?;

    let stdin = bridge_stdin
        .as_mut()
        .ok_or("voice bridge is not connected")?;

    serde_json::to_writer(&mut *stdin, &command)?;
    stdin.write_all(b"\n")?;
    stdin.flush()?;

    Ok(())
}

fn read_bridge_events(stdout: impl std::io::Read, operator_submitter: ExternalSubmitter) {
    for line in BufReader::new(stdout).lines() {
        let Ok(line) = line else {
            break;
        };

        let Ok(event) = serde_json::from_str::<BridgeEvent>(&line) else {
            continue;
        };

        if event.kind != "transcript" {
            continue;
        }

        let Some(text) = event.text else {
            continue;
        };

        let text = text.trim();

        if text.is_empty() {
            continue;
        }

        let visible_text = text.to_string();
        let content = tag_voice_input(visible_text.clone());

        if operator_submitter.submit(content, visible_text).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_input_tag_round_trips_for_command_dispatch() {
        let tagged = tag_voice_input("/voice off".to_string());

        assert_eq!(command_text(&tagged), "/voice off");
    }
}
