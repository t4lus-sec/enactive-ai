//! The operator's switch on s0ma's access to its machine: `/vm off` (or the
//! overview's "vm access" row) and every tool that acts on the VM is refused,
//! for s0ma and its subagents alike, until `/vm on`.
//!
//! The gates are where those tools run, not in the model's tool list, so a
//! call can't slip past one: `vm::execute` (commands and every shell), the
//! board's goal tests (`vm::GoalTest`), the agent's dispatch for the desktop,
//! the browser, images, `/share` files, `share`, `board_report`, Minecraft
//! and the wake tools, and MCP servers that run in the VM (`mcp.rs`). The
//! harness's own work on the machine (the share sync, file previews) carries
//! on: the switch is on what s0ma does.
//!
//! Kept in `data/vm-access.json`, so it holds across a restart. A file that
//! can't be read keeps access off: when in doubt, the machine stays closed.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

static OFF: AtomicBool = AtomicBool::new(false);
static PATH: OnceLock<PathBuf> = OnceLock::new();

/// Read the switch as it was left, from beside the archive.
pub fn init(archive: &Path) {
    let path = archive.with_file_name("vm-access.json");
    OFF.store(kept_off(&path), Ordering::SeqCst);
    let _ = PATH.set(path);
}

/// Whether the file at `path` says off: no file, on; one that does not read
/// (or says anything but off: false), off.
fn kept_off(path: &Path) -> bool {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str::<serde_json::Value>(&text).map_or(true, |value| value["off"].as_bool() != Some(false)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => true,
    }
}

/// Whether s0ma may use its machine now.
pub fn allowed() -> bool {
    !OFF.load(Ordering::SeqCst)
}

/// Turn access on or off, and keep it. What to say.
pub fn set(on: bool) -> Result<String, String> {
    let was = allowed();
    OFF.store(!on, Ordering::SeqCst);
    if let Some(path) = PATH.get() {
        let body = serde_json::json!({"off": !on, "at": chrono::Local::now().to_rfc3339()}).to_string();
        crate::util::write_atomic(path, body).map_err(|error| {
            format!("[vm] access is {} for now, but could not be kept for a restart: {error}", if on { "on" } else { "off" })
        })?;
    }
    Ok(match (was, on) {
        (true, true) => "[vm] s0ma's access to its machine was already on".into(),
        (false, false) => "[vm] s0ma's access to its machine was already off".into(),
        (_, true) => "[vm] s0ma's access to its machine is on again".into(),
        (_, false) => "[vm] s0ma's access to its machine is off: its commands, shells, desktop, browser, /share files, \
                       board tests and VM MCP servers are refused until /vm on"
            .into(),
    })
}

/// `/vm`, `/vm status`, `/vm on`, `/vm off`; `None` for anything else.
pub fn handle_command(command: &str) -> Option<Result<String, String>> {
    let words: Vec<&str> = command.split_whitespace().collect();
    match words.as_slice() {
        ["/vm"] | ["/vm", "status"] => Some(Ok(format!(
            "[vm] s0ma's access to its machine is {}",
            if allowed() { "on (/vm off turns it off)" } else { "off (/vm on turns it back on)" }
        ))),
        ["/vm", "on"] => Some(set(true)),
        ["/vm", "off"] => Some(set(false)),
        ["/vm", ..] => Some(Err("usage: /vm [on|off|status]".into())),
        _ => None,
    }
}

/// What a refused tool gives back.
pub fn refusal(tool: &str) -> String {
    serde_json::json!({
        "ok": false,
        "error": format!(
            "not run: the operator has turned off your access to your machine (/vm off), so {tool} and every \
             other tool that acts on it are refused until he turns it back on. Nothing ran. Work from what you \
             have, and tell him if you need the machine."
        ),
    })
    .to_string()
}

/// Tools the agent's dispatch refuses while access is off (the rest of the
/// machine's tools are refused where they run: `vm::execute`, `GoalTest`).
pub fn gated(name: &str) -> bool {
    name.starts_with("desktop_")
        || name.starts_with("minecraft_")
        || name == "board_report"
        || crate::vm_image::handles(name)
        || crate::share::handles(name)
        || crate::shared_files::handles(name)
        || crate::browser::handles(name)
        || crate::vm_wake::handles(name)
}

/// The turn header's line while access is off, after the rest of it.
pub fn note_for_header(header: String) -> String {
    if allowed() {
        return header;
    }
    format!(
        "{header}\nYour machine is off limits: the operator has turned off your access to it (/vm off). Commands, \
         shells, the desktop and browser, /share files, board tests and reports, MCP servers in the VM and wake \
         watches are refused until he turns it back on, for your subagents too. Work from what you have."
    )
}

#[cfg(test)]
pub(crate) fn set_for_test(on: bool) {
    OFF.store(!on, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_command_turns_it_off_and_on_and_keeps_it() {
        let _lock = crate::worker_feed::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        assert!(handle_command("/share x").is_none());
        set_for_test(true);
        assert!(handle_command("/vm off").unwrap().unwrap().contains("is off"));
        assert!(!allowed());
        assert!(handle_command("/vm").unwrap().unwrap().contains("off (/vm on"));
        assert!(note_for_header("time".into()).contains("off limits"));
        assert!(refusal("vm_exec").contains("Nothing ran"));
        assert!(gated("desktop_click") && gated("shared_write") && gated("board_report") && !gated("archive_search"));
        // Where commands run: refused before anything reaches the machine.
        let call = crate::types::ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: crate::types::FunctionCall { name: "vm_exec".into(), arguments: r#"{"command":"ls"}"#.into() },
        };
        assert!(crate::vm::execute(&call).unwrap().contains("Nothing ran"));
        let test = crate::board::TestRunner::run(&crate::vm::GoalTest, "true", 5).unwrap_err();
        assert!(test.to_string().contains("/vm off"), "{test}");
        assert!(handle_command("/vm on").unwrap().unwrap().contains("on again"));
        assert!(allowed());
        assert_eq!(note_for_header("time".into()), "time");
        assert!(handle_command("/vm sideways").unwrap().is_err());
    }

    #[test]
    fn a_file_that_does_not_read_keeps_the_machine_closed() {
        let dir = std::env::temp_dir().join(format!("s0ma-vm-access-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vm-access.json");
        assert!(!kept_off(&path), "no file: on");
        std::fs::write(&path, "{not json").unwrap();
        assert!(kept_off(&path), "unreadable: off");
        std::fs::write(&path, r#"{"off": false}"#).unwrap();
        assert!(!kept_off(&path));
        std::fs::write(&path, r#"{"off": true}"#).unwrap();
        assert!(kept_off(&path));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
