//! Where the agent works: its own virtual machine (the default), or this
//! computer, the one the harness runs on ("host mode"), for a setup with no
//! VM.
//!
//! Set with `/machine host` or `/machine vm` (the overview's machine row), or
//! `S0MA_MACHINE=host|vm`, which wins. Kept in `data/machine.json`, and taken
//! up when the harness starts: shells, the shared folder and wake watches are
//! set up once, so a change waits for the next start.
//!
//! **In host mode** every command the harness would have run in the VM
//! through the QEMU guest agent (`virsh qemu-agent-command`, `guest-exec`
//! and `guest-exec-status`) runs here instead, in a process of its own, as
//! this user (`agent_command`). The rest follows from that one change:
//! - commands and shells (the netboxes, in tmux on a server of their own, its
//!   sockets in `data/host-tmux`) run on this computer, starting in
//!   `S0MA_HOST_WORKSPACE` or the home folder;
//! - the shared folder is `~/s0ma-share` itself, and nothing syncs it with
//!   anything;
//! - subagents work as this user too (no account is made for them unless
//!   `S0MA_WORKER_USER` names one);
//! - the browser and desktop tools are off: they drive the VM's display;
//! - wake watches poll, with no VM doorbell to ring.
//!
//! The agent's commands touch the operator's own files, so the first time
//! host mode is taken up, approval mode is turned on (`approval.rs`): every
//! command waits for Alt+A until the operator turns it off.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Saved {
    /// The machine set: host or not.
    #[serde(default)]
    host: bool,
    /// Approval mode was turned on for host mode once; the operator's own
    /// choice is left alone after that.
    #[serde(default)]
    approval_given: bool,
}

/// What this run took up when it started: never changes while it runs.
static HOST: AtomicBool = AtomicBool::new(false);
static PATH: OnceLock<PathBuf> = OnceLock::new();
static TMUX_DIR: OnceLock<PathBuf> = OnceLock::new();

fn env_choice(get: impl Fn(&str) -> Option<String>) -> Option<bool> {
    match get("S0MA_MACHINE")?.trim().to_ascii_lowercase().as_str() {
        "host" | "local" | "this" => Some(true),
        "vm" => Some(false),
        _ => None,
    }
}

fn read(path: &Path) -> Saved {
    std::fs::read_to_string(path).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default()
}

fn write(path: &Path, saved: &Saved) -> Result<(), String> {
    let text = serde_json::to_string_pretty(saved).map_err(|error| error.to_string())?;
    crate::util::write_atomic(path, text).map_err(|error| format!("could not keep it in {}: {error}", path.display()))
}

/// Read the machine as kept (or as `S0MA_MACHINE` says), for this run:
/// before anything reaches the machine.
pub fn init(data_dir: &Path) {
    let path = data_dir.join("machine.json");
    let host = env_choice(|key| std::env::var(key).ok()).unwrap_or(read(&path).host);
    HOST.store(host, Ordering::SeqCst);
    let _ = TMUX_DIR.set(data_dir.join("host-tmux"));
    let _ = PATH.set(path);
}

/// In host mode, what to tell the operator at the start; and the first time,
/// approval mode turned on. Once approval mode can be set (`approval::init`).
pub fn announce() -> Option<String> {
    if !on_host() {
        return None;
    }
    let mut said = format!(
        "[machine] host mode: commands and shells run on this computer, from {}; the shared folder is {}; the browser and desktop tools are off",
        crate::share::sync::home_short(&workspace()),
        crate::share::sync::home_short(&crate::share::receive_dir()),
    );
    let path = PATH.get()?;
    let mut saved = read(path);
    if !saved.approval_given {
        match crate::approval::set(true) {
            Ok(_) => said.push_str(". Approval mode is on: every command waits for you (alt+a twice approves; /approval off ends it)"),
            Err(error) => said.push_str(&format!(". [!] Approval mode could not be turned on: {error}")),
        }
        saved.approval_given = true;
        if let Err(error) = write(path, &saved) {
            said.push_str(&format!(" ({error})"));
        }
    }
    Some(said)
}

/// The agent works on this computer, not in a VM (as this run started).
pub fn on_host() -> bool {
    HOST.load(Ordering::SeqCst)
}

/// What is set for the next start, when it differs from this run.
pub fn pending() -> Option<bool> {
    let path = PATH.get()?;
    let next = env_choice(|key| std::env::var(key).ok()).unwrap_or(read(path).host);
    (next != on_host()).then_some(next)
}

/// Where host commands start: `S0MA_HOST_WORKSPACE`, or the home folder.
pub fn workspace() -> PathBuf {
    std::env::var_os("S0MA_HOST_WORKSPACE")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// `/machine`, `/machine host|vm`; None for anything else.
pub fn handle_command(command: &str) -> Option<Result<String, String>> {
    let words: Vec<&str> = command.split_whitespace().collect();
    Some(match words.as_slice() {
        ["/machine"] | ["/machine", "status"] => Ok(status()),
        ["/machine", choice @ ("host" | "vm")] => {
            let host = *choice == "host";
            let Some(path) = PATH.get() else { return Some(Err("[machine] not available in this process".into())) };
            let mut saved = read(path);
            saved.host = host;
            if let Err(error) = write(path, &saved) {
                return Some(Err(format!("[machine] {error}")));
            }
            let mut said = if host == on_host() {
                format!("[machine] {} (as now)", if host { "this computer" } else { "its VM" })
            } else if host {
                "[machine] this computer from the next start: commands and shells will run here, among your files, \
                 and the browser and desktop tools will be off"
                    .to_string()
            } else {
                "[machine] its VM from the next start".to_string()
            };
            if host && !saved.approval_given && !crate::approval::enabled() {
                said.push_str(". Approval mode will be on when it starts");
            }
            if std::env::var("S0MA_MACHINE").is_ok_and(|value| !value.trim().is_empty()) {
                said.push_str("; S0MA_MACHINE is set and wins until it is unset");
            }
            Ok(said)
        }
        ["/machine", ..] => Err("[machine] /machine host (this computer) or /machine vm".into()),
        _ => return None,
    })
}

pub fn status() -> String {
    let now = if on_host() {
        format!("this computer (host mode), commands from {}", crate::share::sync::home_short(&workspace()))
    } else {
        "its VM".to_string()
    };
    match pending() {
        Some(next) => format!("[machine] {now}; {} from the next start", if next { "this computer" } else { "its VM" }),
        None => format!("[machine] {now}"),
    }
}

/// A browser or desktop tool, in host mode: what it says instead.
pub fn off_here(tool: &str) -> String {
    format!(
        "{tool} is off: the agent works on this computer (host mode), and the browser and desktop tools drive a VM's display. \
         Use web_search and commands instead."
    )
}

/// What the system prompt says of the machine in host mode.
pub fn prompt_block() -> Option<String> {
    on_host().then(|| {
        format!(
            "[YOUR MACHINE — THIS COMPUTER]\n\
             You work on the operator's own computer, the one the harness runs on, not on a separate machine or a VM: \
             this supersedes what tool descriptions say about your machine. Your commands run as the operator's user, \
             among their files, starting in {}. The shared folder is {} (where a tool says /share, it means this folder). \
             The browser and desktop tools are off. Be careful with anything that changes or deletes what is already here; \
             when in doubt, ask first.",
            workspace().display(),
            crate::share::receive_dir().display()
        )
    })
}

// ------------------------------------------------------------------ the guest agent, here

struct Finished {
    code: i64,
    out: Vec<u8>,
    err: Vec<u8>,
}

type Slot = Arc<Mutex<Option<Finished>>>;

fn running() -> &'static Mutex<HashMap<i64, Slot>> {
    static RUNNING: OnceLock<Mutex<HashMap<i64, Slot>>> = OnceLock::new();
    RUNNING.get_or_init(Default::default)
}

/// The most of a stream kept: past the harness's own capture limits.
const KEEP_BYTES: usize = 32 * 1024 * 1024;

fn read_all(mut from: impl Read + Send + 'static) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            match from.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let room = KEEP_BYTES.saturating_sub(kept.len());
                    kept.extend_from_slice(&buffer[..n.min(room)]);
                }
            }
        }
        kept
    })
}

/// A QEMU guest agent request, answered on this computer as the agent in the
/// VM would: `guest-exec` of the runner (a timeout and a base64 command)
/// starts the command and gives its pid; `guest-exec-status` says whether it
/// has exited, and with what.
pub fn agent_command(request: &Value) -> Result<Value, Box<dyn std::error::Error>> {
    match request.get("execute").and_then(Value::as_str) {
        Some("guest-exec") => {
            let args = request.pointer("/arguments/arg").and_then(Value::as_array).ok_or("guest-exec without arguments")?;
            let timeout = args.first().and_then(Value::as_str).ok_or("guest-exec without a timeout")?;
            if timeout.is_empty() || !timeout.bytes().all(|b| b.is_ascii_digit()) {
                return Err(format!("guest-exec timeout is not a number of seconds: {timeout:?}").into());
            }
            let command = String::from_utf8(STANDARD.decode(args.get(1).and_then(Value::as_str).ok_or("guest-exec without a command")?)?)?;
            let start = workspace();
            let mut process = Command::new("timeout");
            process
                .args([timeout, "bash", "-c", &command])
                .current_dir(if start.is_dir() { start.clone() } else { PathBuf::from("/") })
                .env_remove("TMUX")
                // Where a new shell starts (`crate::vm::netboxes::setup_script`).
                .env("S0MA_WORKDIR", &start)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            // Shells on a tmux server of their own, not the operator's.
            if let Some(dir) = TMUX_DIR.get() {
                let _ = std::fs::create_dir_all(dir);
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
                process.env("TMUX_TMPDIR", dir);
            }
            // Its own process group: a key at the harness's terminal never
            // reaches it, and it can be stopped as a tree.
            std::os::unix::process::CommandExt::process_group(&mut process, 0);
            let mut child = process.spawn()?;
            let pid = i64::from(child.id());
            let out = read_all(child.stdout.take().ok_or("no stdout")?);
            let err = read_all(child.stderr.take().ok_or("no stderr")?);
            let slot: Slot = Arc::default();
            crate::util::locked(running()).insert(pid, Arc::clone(&slot));
            std::thread::Builder::new().name("host-exec".into()).spawn(move || {
                let code = match child.wait() {
                    Ok(status) => status.code().map(i64::from).unwrap_or_else(|| {
                        128 + i64::from(std::os::unix::process::ExitStatusExt::signal(&status).unwrap_or(0))
                    }),
                    Err(_) => -1,
                };
                let finished = Finished { code, out: out.join().unwrap_or_default(), err: err.join().unwrap_or_default() };
                *crate::util::locked(&slot) = Some(finished);
            })?;
            Ok(json!({"return": {"pid": pid}}))
        }
        Some("guest-exec-status") => {
            let pid = request.pointer("/arguments/pid").and_then(Value::as_i64).ok_or("guest-exec-status without a pid")?;
            let slot = crate::util::locked(running()).get(&pid).cloned().ok_or_else(|| format!("no process {pid} was started here"))?;
            let finished = crate::util::locked(&slot).take();
            match finished {
                None => Ok(json!({"return": {"exited": false}})),
                Some(finished) => {
                    crate::util::locked(running()).remove(&pid);
                    Ok(json!({"return": {
                        "exited": true,
                        "exitcode": finished.code,
                        "out-data": STANDARD.encode(&finished.out),
                        "err-data": STANDARD.encode(&finished.err),
                    }}))
                }
            }
        }
        other => Err(format!("host mode has no guest agent command {other:?}").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn run(command: &str, timeout: u64) -> Value {
        let pid = agent_command(&json!({"execute": "guest-exec", "arguments": {
            "path": "/usr/local/bin/s0ma-run", "arg": [timeout.to_string(), STANDARD.encode(command)], "capture-output": true}}))
        .unwrap()["return"]["pid"]
            .as_i64()
            .unwrap();
        let until = Instant::now() + Duration::from_secs(20);
        loop {
            let status = agent_command(&json!({"execute": "guest-exec-status", "arguments": {"pid": pid}})).unwrap();
            if status["return"]["exited"] == true {
                return status["return"].clone();
            }
            assert!(Instant::now() < until, "{command} never finished");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn text(status: &Value, field: &str) -> String {
        String::from_utf8(STANDARD.decode(status[field].as_str().unwrap()).unwrap()).unwrap()
    }

    #[test]
    fn a_command_runs_here_as_the_guest_agent_would_run_it() {
        let status = run("echo out; echo err >&2; exit 3", 10);
        assert_eq!(status["exitcode"], 3);
        assert_eq!((text(&status, "out-data"), text(&status, "err-data")), ("out\n".to_string(), "err\n".to_string()));
        // Its own tmux server, never the operator's.
        let status = run("echo \"[$TMUX]\"", 10);
        assert_eq!(text(&status, "out-data"), "[]\n");
        // The runner's timeout ends it.
        let started = Instant::now();
        let status = run("sleep 30", 1);
        assert_eq!(status["exitcode"], 124, "timeout's own code");
        assert!(started.elapsed() < Duration::from_secs(10));
        // A status asked of a pid never started here: an error, not a hang.
        assert!(agent_command(&json!({"execute": "guest-exec-status", "arguments": {"pid": 1}})).is_err());
        assert!(agent_command(&json!({"execute": "guest-file-open", "arguments": {}})).is_err());
        assert!(agent_command(&json!({"execute": "guest-exec", "arguments": {"arg": ["1; rm", "eA=="]}})).is_err(), "a timeout is digits");
    }
}
