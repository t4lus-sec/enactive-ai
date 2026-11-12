pub(crate) mod access;
pub(crate) mod image;
pub(crate) mod machine;
pub(crate) mod netbox_run;
pub(crate) mod netboxes;
pub(crate) mod terms;
pub(crate) mod wake;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    error::Error,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    process::Command,
    sync::OnceLock,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::llm::ToolDefinition;
use crate::types::ToolCall;

/// The libvirt connection, overridable with `S0MA_LIBVIRT_URI` so the VM can
/// live on another machine (`qemu+ssh://operator@legion/system`) without a rebuild.
const DEFAULT_LIBVIRT_URI: &str = "qemu:///system";

/// The libvirt domain the agent works in, as `virsh list` names it.
const DEFAULT_VM_NAME: &str = "s0ma";

/// `DEFAULT_VM_NAME`, or `S0MA_VM_NAME` when the operator sets it.
///
/// Read once and kept, so every audit record of a run names the same domain
/// even if the environment changes under it. An override exists because a
/// rebuilt or renamed VM should not need a rebuilt harness: the name reaches
/// `virsh` (dumpxml, qemu-agent-command) and the audit log, nothing else.
fn libvirt_uri() -> &'static str {
    static URI: OnceLock<String> = OnceLock::new();
    URI.get_or_init(|| chosen_setting("S0MA_LIBVIRT_URI", DEFAULT_LIBVIRT_URI)).as_str()
}

fn vm_name() -> &'static str {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| chosen_setting("S0MA_VM_NAME", DEFAULT_VM_NAME)).as_str()
}

/// One of these settings as configured, or `fallback`. An unset, blank or
/// whitespace-only override leaves the default in place rather than handing
/// virsh an empty domain name or connection string.
fn chosen_setting(variable: &str, fallback: &str) -> String {
    chosen(std::env::var(variable).ok(), fallback)
}

/// `value` trimmed, or `fallback` when there is none or it is blank.
fn chosen(value: Option<String>, fallback: &str) -> String {
    value.map(|value| value.trim().to_owned()).filter(|value| !value.is_empty()).unwrap_or_else(|| fallback.to_owned())
}
const GUEST_RUNNER: &str = "/usr/local/bin/s0ma-run";

const DEFAULT_TIMEOUT_SECS: u64 = 600;
const MAX_TIMEOUT_SECS: u64 = 3600;

const SHELL_HELPER_TIMEOUT_SECS: u64 = 30;
const DEFAULT_SHELL_SESSION: &str = "main";
const DEFAULT_SHELL_CAPTURE_LINES: usize = 200;
const MAX_SHELL_CAPTURE_LINES: usize = 2_000;
const MAX_SHELL_INPUT_CHARS: usize = 32_000;

/// What the host keeps of a command's stdout and of its stderr, in characters.
/// Everything kept is journaled and archived whole; only what a model is shown
/// is cut down (to a preview with a handle on the rest). It was 64,000, which
/// cut a 70k-character result before it was kept, so a handle to it could
/// only ever return the first 64k. Past this bound the cut is still made here,
/// and said so: `*_truncated`, and the marker says how much was dropped. (A
/// model's context is protected by those previews, not by this. It does not
/// restrict execution inside the VM.)
const MAX_CAPTURE_CHARS: usize = 1_000_000;
const MAX_DESKTOP_CAPTURE_CHARS: usize = 16 * 1024 * 1024;

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::function(
            "vm_exec",
            "Execute a single shell command as the s0ma user on your machine, the Parrot OS laptop you work on. It runs in a shell, a terminal the operator sees (see SHELLS): the one you name, or the one you last worked in, in a subshell of it (its folder and exported variables), and returns after the command exits with its output (stdout and stderr together, as the terminal showed them) and exit code. The first shell, main, starts in /workspace. When the command wrote a file the harness can check (.py, .sh, .json, .yaml, .toml, .js, .rs, or a #! script), the result also carries diagnostics: the harness's own read-only syntax checks of what was just written. For one long command — a build, a long scan, a download — set background=true: the call returns at once with the job's name, the whole output goes to a log you can tail while it runs, and when it finishes a wake event named after the job carries its exit code and the end of its output. Don't poll the log in a loop; do something else and read the event. For an interactive program or a long-running foreground process, use shell_type, shell_key and shell_read in a shell instead. Never run sleep (alone or in a loop) to stay idle or let time pass: it holds the call and the shell and does nothing. When there is nothing to do, end your turn or use attention_wait; to wait for something on your machine, run it with background=true (or start a wake watch) and act on the event when it comes. This tool does not grant network authorization, expand engagement scope, or reach the operator's own workstation, which is a separate computer.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute on your machine"
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run it detached and return at once, with the output kept in a log and a wake event when it ends; the operator sees it in a shell of its own. timeout_secs does not apply. Default false."
                    },
                    "why": {
                        "type": "string",
                        "description": "Why you are running it, in a few words: the operator sees it above the command"
                    },
                    "shell": {
                        "type": "string",
                        "description": "The shell to run it in (default: the one you last worked in, at first main)"
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Run it in the shell itself rather than a subshell, so a cd, export, source or venv stays there for you and the operator. Its output then goes through a pipe, not a terminal. Default false."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3600,
                        "description": "Maximum runtime in seconds before the command is terminated. Defaults to 600 seconds"
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        ),
    ]
}

// ------------------------------------------------------------------ the netboxes

/// The first netbox, the shell the agent and the operator share (`vm/netboxes.rs` has the rest).
pub const NETBOX: &str = crate::vm::netboxes::DEFAULT;

/// What the agent is told about the netboxes, in its system context.
pub const NETBOX_CONTEXT: &str = crate::vm::netboxes::CONTEXT;

/// The netboxes' tools: the agent's only, never a worker's.
pub fn netbox_tool_definitions() -> Vec<ToolDefinition> {
    crate::vm::netboxes::tool_definitions()
}

/// A guest script that opens the first netbox when it is not open (exit 0
/// either way), or restarts an old one in place when it is idle.
pub(crate) fn netbox_open_script() -> String {
    crate::vm::netboxes::open_script(&crate::vm::netboxes::Spec::default_netbox())
}

fn execute_netbox(call: &ToolCall) -> Result<String, Box<dyn Error>> {
    let mut args: Value = serde_json::from_str(&call.function.arguments)?;
    let object = args.as_object_mut().ok_or("tool arguments must be an object")?;
    let named = object.remove("shell").or_else(|| object.remove("netbox"));
    object.remove("session");
    let worker = crate::exec_feed::owner();
    // A subagent's are its own shell's, whatever it names.
    let name = match &worker {
        Some(run_id) => crate::vm::netboxes::worker_name(run_id),
        None => crate::vm::netboxes::canonical(named.as_ref().and_then(Value::as_str).unwrap_or(NETBOX)),
    };
    if let Err(error) = crate::vm::netboxes::valid_name(&name) {
        return Ok(json!({"ok": false, "error": error}).to_string());
    }
    let session = crate::vm::netboxes::session_of(&name);
    let tmux_session = tmux_session_name(&session)?;
    if name == NETBOX {
        let opened = run_shell_helper(&netbox_open_script())?;
        if opened.exit_code != Some(0) {
            return Ok(shell_helper_error(NETBOX, &opened));
        }
        crate::vm::terms::touch();
    }
    if let Some(run_id) = &worker {
        let user = crate::vm::netboxes::acting_worker();
        if let Some(user) = &user
            && let Err(error) = crate::vm::netboxes::ensure_worker_account(&mut crate::vm::netboxes::machine, user)
        {
            return Ok(json!({"ok": false, "error": error}).to_string());
        }
        let spec = crate::vm::netbox_run::worker_spec(run_id, user);
        if let Err(error) = crate::vm::netboxes::open(&mut crate::vm::netboxes::machine, &spec) {
            return Ok(json!({"ok": false, "error": format!("your shell could not be opened: {error}")}).to_string());
        }
        crate::vm::terms::touch();
    }
    let read = matches!(call.function.name.as_str(), "shell_read" | "netbox_read");
    // Typing: not while the operator has taken control.
    if !read {
        let wheel = run_shell_helper(&format!(
            "tmux has-session -t {tmux_session} 2>/dev/null || exit 44; tmux show-options -v -t {tmux_session} @s0ma_wheel 2>/dev/null; true"
        ))?;
        if wheel.exit_code == Some(44) {
            return Ok(json!({"ok": false, "shell": name, "error": format!("no shell {name} is open; shell_open opens one")}).to_string());
        }
        if wheel.stdout.trim() == "1" {
            let after = if worker.is_some() { "" } else { " vm_exec still works: it runs in your lane beside it." };
            return Ok(json!({"ok": false, "shell": name, "error": format!(
                "The operator has taken control of {}: you cannot type there until it is handed back.{after}", crate::vm::netboxes::display(&name))}).to_string());
        }
    }
    // Approval mode: what would be typed or pressed waits for the operator.
    if !read {
        let shown = crate::vm::netboxes::display(&name);
        let (what, detail) = match call.function.name.as_str() {
            "shell_key" | "netbox_key" => {
                let key = object.get("key").and_then(Value::as_str).unwrap_or("a key");
                (format!("press {key} in {shown}"), String::new())
            }
            _ => {
                let text = object.get("text").and_then(Value::as_str).unwrap_or_default();
                // Enter after it unless the call says not: then it runs.
                let enter = object.get("enter").and_then(Value::as_bool).unwrap_or(true);
                let then = if enter { ", then press enter" } else { " (no enter)" };
                (format!("type into {shown}{then}: {}", crate::approval::one_line(text)), text.to_string())
            }
        };
        if let Err(why) = crate::approval::ask(&what, &detail) {
            return Ok(json!({"ok": false, "shell": name, "error": why}).to_string());
        }
    }
    if worker.is_none() {
        crate::vm::netboxes::set_working(&name);
    }
    object.insert("session".into(), json!(session));
    if read && !object.contains_key("lines") {
        object.insert("lines".into(), json!(80));
    }
    let mut scoped = call.clone();
    scoped.function.arguments = args.to_string();
    let result = match call.function.name.as_str() {
        "shell_type" | "netbox_type" => execute_shell_input(&scoped)?,
        "shell_key" | "netbox_key" if object_key(&args) == Some("ctrl_l") => {
            let command = format!("tmux has-session -t {tmux_session} 2>/dev/null || exit 44; tmux send-keys -t {tmux_session}:0.0 C-l");
            let done = run_shell_helper(&command)?;
            if done.exit_code != Some(0) {
                return Ok(shell_helper_error(&name, &done));
            }
            json!({"ok": true, "session": session, "key": "ctrl_l"}).to_string()
        }
        "shell_key" | "netbox_key" => execute_shell_key(&scoped)?,
        _ => execute_shell_read(&scoped)?,
    };
    // Whose is whose: what the agent typed is marked, and a read says what the operator
    // typed since the last one.
    let mut value: Value = serde_json::from_str(&result).unwrap_or(json!({"output": result}));
    if let Some(object) = value.as_object_mut() {
        object.remove("session");
        object.insert("shell".into(), json!(name));
        if object.get("status").and_then(Value::as_str) == Some("not_open") {
            object.insert("error".into(), json!(format!("no shell {name} is open; shell_open opens one")));
        }
        object.insert("shared_with".into(), json!("The operator sees this terminal live and can type into it too"));
        // The pane's blank rows under the prompt say nothing.
        if let Some(Value::String(output)) = object.get_mut("output") {
            let kept = output.trim_end_matches(['\n', ' ']).len();
            output.truncate(kept);
        }
        if read {
            let typed = crate::vm::terms::take_netbox_typing(&session);
            object.insert(
                "operator_typed_since_you_last_looked".into(),
                if typed.is_empty() { json!("nothing") } else { json!(typed) },
            );
        }
    }
    Ok(value.to_string())
}

fn object_key(args: &Value) -> Option<&str> {
    args.get("key").and_then(Value::as_str)
}

#[derive(Debug, Serialize)]
struct VmAuditEvent<'a> {
    timestamp: DateTime<Utc>,
    command_id: &'a str,
    event: &'a str,
    vm: &'a str,

    command: &'a str,
    timeout_secs: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    exit_code: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    signal: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u128>,

    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,

    /// The background job this command became, on the record that says it was
    /// detached. Its own field rather than a note in `error`, so a reader
    /// filtering the log for failures does not find a command that started
    /// perfectly well.
    #[serde(skip_serializing_if = "Option::is_none")]
    job: Option<&'a str>,

    /// The netbox it ran in, when it ran in one.
    #[serde(skip_serializing_if = "Option::is_none")]
    netbox: Option<&'a str>,

    /// The account it ran as, when not the agent's (a subagent's).
    #[serde(skip_serializing_if = "Option::is_none")]
    account: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct VmExecArgs {
    command: String,

    #[serde(default = "default_timeout_secs")]
    timeout_secs: u64,

    /// Run it detached and return at once. `timeout_secs` then means nothing:
    /// the point of a background command is to outlive the call.
    #[serde(default)]
    background: bool,

    /// Why, for the operator: shown above the command in the netbox.
    #[serde(default)]
    why: Option<String>,

    /// The shell (netbox) to run it in; `netbox` as calls made before the
    /// rename have it.
    #[serde(default, rename = "shell", alias = "netbox")]
    netbox: Option<String>,

    /// In the netbox's own shell, not a subshell.
    #[serde(default)]
    persist: bool,
}

#[derive(Debug, Deserialize)]
struct ShellInputArgs {
    #[serde(default = "default_shell_session")]
    session: String,

    text: String,

    #[serde(default = "default_true")]
    enter: bool,
}

#[derive(Debug, Deserialize)]
struct ShellKeyArgs {
    #[serde(default = "default_shell_session")]
    session: String,

    key: String,
}

#[derive(Debug, Deserialize)]
struct ShellReadArgs {
    #[serde(default = "default_shell_session")]
    session: String,

    #[serde(default = "default_shell_capture_lines")]
    lines: usize,
}

#[derive(Debug)]
pub(crate) struct GuestCommandResult {
    pub(crate) exit_code: Option<i64>,
    pub(crate) signal: Option<i64>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) stdout_truncated: bool,
    pub(crate) stderr_truncated: bool,
    pub(crate) duration_ms: u128,
}

#[derive(Debug)]
pub(crate) enum GuestCommandOutcome {
    Exited(GuestCommandResult),
    HostTimeout {
        pid: i64,
        timeout_secs: u64,
        duration_ms: u128,
    },
}

fn default_timeout_secs() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

fn default_shell_session() -> String {
    DEFAULT_SHELL_SESSION.to_string()
}

fn default_shell_capture_lines() -> usize {
    DEFAULT_SHELL_CAPTURE_LINES
}

fn default_true() -> bool {
    true
}

pub fn execute(call: &ToolCall) -> Result<String, Box<dyn Error>> {
    // Every command and shell, the agent's and its subagents', stops here while
    // the operator has the machine off limits.
    if !crate::vm::access::allowed() {
        return Ok(crate::vm::access::refusal(&call.function.name));
    }
    match call.function.name.as_str() {
        "vm_exec" => execute_vm_exec(call),
        // `netbox_*`: the same tools, as calls made before the rename name them.
        "shell_type" | "shell_key" | "shell_read" | "netbox_type" | "netbox_key" | "netbox_read" => execute_netbox(call),
        "shell_open" | "shell_close" | "netbox_open" | "netbox_close" => Ok(crate::vm::netboxes::execute(call)),
        // The private shells there were before every shell was shared.
        "vm_shell_open" | "vm_shell_input" | "vm_shell_key" | "vm_shell_read" | "vm_shell_close" => Ok(json!({
            "ok": false,
            "error": format!("{} is gone: every shell is one the operator sees. Use vm_exec, or shell_type, shell_key and shell_read (shell_open for one of your own).", call.function.name)
        })
        .to_string()),

        other => Ok(json!({
            "ok": false,
            "error": format!("unknown machine tool: {other}")
        })
        .to_string()),
    }
}

/// Start a command detached and return at once.
///
/// Audited like any other: a `started` record before anything runs, so a
/// command that outlives the call is on the record from the moment it is
/// launched rather than only when it ends, and a `background` record naming
/// the job once it has. Nothing here waits for it; what it did arrives later
/// as a wake event, or sits in its log.
///
/// The result deliberately carries **no `exit_code`**. The one way this could
/// mislead is by looking like a command that finished, so it does not: it
/// names the job, says where the log is, and says whether an event is coming.
fn start_background_command(command_id: &str, command: &str) -> Result<String, Box<dyn Error>> {
    let name = crate::vm::wake::background_name();

    // Fail closed, as the foreground path does: no audit record, nothing runs.
    append_audit(&VmAuditEvent {
        timestamp: Utc::now(),
        command_id,
        event: "started",
        vm: vm_name(),
        command,
        timeout_secs: 0,
        exit_code: None,
        signal: None,
        duration_ms: None,
        error: None,
        job: None,
        netbox: None,
        account: None,
    })?;

    let job = match crate::vm::wake::start_background(&name, command) {
        Ok(job) => job,
        Err(error) => {
            let error_message = error.to_string();
            append_audit(&VmAuditEvent {
                timestamp: Utc::now(),
                command_id,
                event: "failed",
                vm: vm_name(),
                command,
                timeout_secs: 0,
                exit_code: None,
                signal: None,
                duration_ms: None,
                error: Some(&error_message),
                job: None,
                netbox: None,
                account: None,
            })?;
            return Err(error);
        }
    };

    append_audit(&VmAuditEvent {
        timestamp: Utc::now(),
        command_id,
        event: "background",
        vm: vm_name(),
        command,
        timeout_secs: 0,
        exit_code: None,
        signal: None,
        duration_ms: None,
        error: None,
        job: Some(&job.name),
        netbox: None,
        account: None,
    })?;

    let log = format!("{}/{}.log", job.dir, job.name);
    let mut value = json!({
        "background": true,
        "job": job.name,
        "pid": job.pid,
        "log": log,
        "event_when_done": job.notified,
    });
    value["note"] = json!(if job.notified {
        format!(
            "Started, and this call is done — it tells you nothing about how the command went. When it ends, \
             a wake event named \"{}\" will carry its exit code and the end of its output. Until then the \
             whole output is in {log}, which you can read while it runs. Get on with something else rather \
             than waiting on it.",
            job.name
        )
    } else {
        format!(
            "Started, and this call is done — it tells you nothing about how the command went. The wake \
             channel is not on in this process, so nothing will announce the end of it: the whole output, \
             and its exit code on the last line, are in {log}. Read it when you expect it to be finished."
        )
    });
    Ok(value.to_string())
}

/// Whether vm_exec runs in a netbox (`S0MA_VM_EXEC_IN_NETBOX=off`: by the
/// guest runner alone, as before netboxes).
pub(crate) fn in_netbox() -> bool {
    // Approval mode: every command where the operator sees it.
    crate::approval::enabled() || !matches!(std::env::var("S0MA_VM_EXEC_IN_NETBOX").ok().as_deref().map(str::trim), Some("off" | "0" | "false"))
}

/// One audit record of a vm_exec: the event, and what is known of it.
struct Audit<'a> {
    command_id: &'a str,
    command: &'a str,
    timeout_secs: u64,
    netbox: Option<&'a str>,
    account: Option<&'a str>,
}

impl Audit<'_> {
    fn record(&self, event: &str, exit_code: Option<i64>, duration_ms: Option<u128>, error: Option<&str>) -> Result<(), Box<dyn Error>> {
        append_audit(&VmAuditEvent {
            timestamp: Utc::now(),
            command_id: self.command_id,
            event,
            vm: vm_name(),
            command: self.command,
            timeout_secs: self.timeout_secs,
            exit_code,
            signal: None,
            duration_ms,
            error,
            job: None,
            netbox: self.netbox,
            account: self.account,
        })
    }
}

fn execute_vm_exec(call: &ToolCall) -> Result<String, Box<dyn Error>> {
    let args: VmExecArgs = serde_json::from_str(&call.function.arguments)?;

    if args.command.trim().is_empty() {
        return Err("vm_exec command cannot be empty".into());
    }

    if !(1..=MAX_TIMEOUT_SECS).contains(&args.timeout_secs) {
        return Err(
            format!("vm_exec timeout_secs must be between 1 and {MAX_TIMEOUT_SECS}").into(),
        );
    }

    // Approval mode: the command waits for the operator before anything else.
    let worker_run = crate::exec_feed::owner();
    let approved = crate::approval::enabled();
    {
        let place = if args.background {
            "in the background".to_string()
        } else if worker_run.is_some() {
            "its shell".to_string()
        } else {
            crate::vm::netboxes::display(&args.netbox.clone().map(|name| crate::vm::netboxes::canonical(&name)).unwrap_or_else(crate::vm::netboxes::working))
        };
        let why = args.why.as_deref().map(str::trim).filter(|why| !why.is_empty()).map(|why| format!("# {why}\n")).unwrap_or_default();
        if let Err(refused) = crate::approval::ask(&format!("run in {place}: {}", crate::approval::one_line(&args.command)), &format!("{why}{}", args.command)) {
            return Ok(json!({"ok": false, "error": refused}).to_string());
        }
    }
    let command_id = new_command_id();
    let started = Instant::now();
    // A subagent's call runs as the subagents' account (when there is one).
    let account = crate::vm::netboxes::acting_worker();
    if let Some(user) = &account
        && let Err(error) = crate::vm::netboxes::ensure_worker_account(&mut crate::vm::netboxes::machine, user)
    {
        return Ok(json!({"ok": false, "error": error}).to_string());
    }

    if args.background {
        if worker_run.is_some() {
            return Ok(json!({"ok": false, "error": "background commands are s0ma's: run it in the foreground, within your time limit"}).to_string());
        }
        let started = start_background_command(&command_id, &args.command)?;
        let mut value: Value = serde_json::from_str(&started).unwrap_or_default();
        // The operator sees it in a netbox of its own.
        if let Some(job) = value["job"].as_str().map(str::to_string) {
            match open_job_netbox(&job, &args.command, args.why.as_deref().unwrap_or("")) {
                Ok(name) => value["shell"] = json!(name),
                Err(error) => value["shell_error"] = json!(error),
            }
            return Ok(value.to_string());
        }
        return Ok(started);
    }

    let requested = args.netbox.clone().unwrap_or_else(|| {
        if worker_run.is_some() { "own".into() } else { crate::vm::netboxes::working() }
    });
    // Fail closed: if we cannot create the audit record, do not execute
    // anything in the VM.
    Audit { command_id: &command_id, command: &args.command, timeout_secs: args.timeout_secs, netbox: Some(&requested), account: account.as_deref() }
        .record("started", None, None, None)?;

    // In a shell or not at all: when no shell the operator sees can take the
    // command, it is refused, never run where he cannot see it. Only the
    // operator's own setting (`S0MA_VM_EXEC_IN_NETBOX=off`) runs commands
    // outside a shell, and the overview says when it is set.
    let mut elsewhere: Option<String> = None;
    if in_netbox() {
        let ask = crate::vm::netbox_run::Ask {
            command: args.command.clone(),
            why: args.why.clone().unwrap_or_default(),
            persist: args.persist,
            timeout_secs: args.timeout_secs,
            netbox: if worker_run.is_some() { None } else { args.netbox.clone() },
            worker: worker_run.clone().map(|run_id| (run_id, account.clone())),
            approved,
            label: None,
            no_lane: false,
        };
        match crate::vm::netbox_run::start(&ask, &mut crate::vm::netboxes::machine) {
            Ok(crate::vm::netbox_run::Started::Typed(typed)) => {
                if worker_run.is_none() {
                    crate::vm::netboxes::set_working(if crate::vm::netboxes::is_lane(&typed.netbox) { &requested } else { &typed.netbox });
                }
                return in_netbox_finish(&args.command, &command_id, typed, worker_run.is_some(), account.as_deref());
            }
            Ok(crate::vm::netbox_run::Started::Elsewhere(why)) => elsewhere = Some(why),
            Err(error) if args.netbox.is_some() => {
                Audit { command_id: &command_id, command: &args.command, timeout_secs: args.timeout_secs, netbox: Some(&requested), account: account.as_deref() }
                    .record("failed", None, None, Some(&error))?;
                return Ok(json!({"ok": false, "error": error}).to_string());
            }
            Err(error) => elsewhere = Some(format!("the shell could not be reached: {error}")),
        }
    }
    if let Some(why) = elsewhere.take() {
        let refused = if worker_run.is_some() {
            format!("not run: {why}. A command runs only in a shell the operator sees; wait for yours to be free and try again.")
        } else {
            format!(
                "not run: {why}. A command runs only in a shell the operator sees; wait for it to be free, name another \
                 shell (netbox), open one with shell_open, or run it with background=true (a shell of its own)."
            )
        };
        Audit { command_id: &command_id, command: &args.command, timeout_secs: args.timeout_secs, netbox: Some(&requested), account: account.as_deref() }
            .record("refused", None, None, Some(&refused))?;
        return Ok(json!({"ok": false, "error": refused}).to_string());
    }
    let audit = Audit { command_id: &command_id, command: &args.command, timeout_secs: args.timeout_secs, netbox: None, account: account.as_deref() };

    let guest_command = match &account {
        Some(user) => crate::vm::netboxes::as_user(user, &format!("cd /workspace 2>/dev/null || cd; {}", args.command)),
        None => args.command.clone(),
    };
    let outcome = match run_guest_command(&guest_command, args.timeout_secs) {
        Ok(outcome) => outcome,

        Err(error) => {
            let error_message = error.to_string();
            let killed = error.downcast_ref::<crate::kill::Killed>().is_some();
            audit.record(if killed { "killed" } else { "failed" }, None, Some(started.elapsed().as_millis()), Some(&error_message))?;
            return Err(error);
        }
    };

    match outcome {
        GuestCommandOutcome::Exited(result) => {
            append_audit(&VmAuditEvent {
                timestamp: Utc::now(),
                command_id: &command_id,
                event: "finished",
                vm: vm_name(),
                command: &args.command,
                timeout_secs: args.timeout_secs,
                exit_code: result.exit_code,
                signal: result.signal,
                duration_ms: Some(result.duration_ms),
                error: None,
                job: None,
                netbox: None,
                account: account.as_deref(),
            })?;

            let mut value: Value = serde_json::from_str(&format_guest_result(&result))?;
            // P4: the files this command wrote, checked, in the same result.
            if crate::edit_check::enabled() {
                match post_edit_diagnostics(&command_id, &args.command, result.duration_ms) {
                    Ok(Some(diagnostics)) => value["diagnostics"] = diagnostics,
                    Ok(None) => {}
                    Err(error) => value["diagnostics_error"] = json!(error),
                }
            }
            Ok(value.to_string())
        }

        GuestCommandOutcome::HostTimeout {
            pid,
            timeout_secs,
            duration_ms,
        } => {
            let error_message = format!(
                "lost contact with guest process {pid} after {} seconds",
                timeout_secs + 5
            );
            audit.record("timeout", None, Some(duration_ms), Some(&error_message))?;

            Ok(json!({
                "ok": false,
                "status": "timeout",
                "command_id": command_id,
                "pid": pid,
                "timeout_seconds": timeout_secs,
                "duration_ms": duration_ms,
                "error": error_message
            })
            .to_string())
        }
    }
}

/// The result of a command that ran in a netbox, as vm_exec gives it: the
/// guest runner's shape, its output as the terminal showed it.
fn netbox_result(typed: &crate::vm::netbox_run::Typed, done: &crate::vm::netbox_run::Finished, command_id: &str) -> Value {
    let mut value = json!({
        "exit_code": done.exit_code,
        "signal": null,
        "stdout": done.output,
        "stderr": "",
        "stdout_truncated": done.truncated,
        "stderr_truncated": false,
        "shell": typed.netbox,
        "duration_ms": done.duration_ms,
        "command_id": command_id,
        "note": "stdout and stderr together, as the operator saw them in the shell"
    });
    if let Some(moved) = &typed.moved {
        value["moved"] = json!(moved);
    }
    if let Some(edited) = &done.edited {
        value["edited_by_operator"] = json!(edited);
        value["note"] = json!(format!("The operator edited your command before he ran it: what ran was `{edited}`. stdout and stderr together, as he saw them"));
    }
    if done.declined {
        value = json!({
            "ok": false, "status": "declined_by_operator", "shell": typed.netbox, "command_id": command_id,
            "note": "The operator declined it at his prompt (Ctrl+C): it did not run. Ask him, or do something else."
        });
    } else if done.interrupted {
        value["note"] = json!("stopped by Ctrl+C in the shell before it finished");
    }
    value
}

/// Wait for a command typed into a netbox and give its result: at once when
/// it ends; early when it waits for input or for the operator's Enter (the agent's: the
/// rest comes as an event; a subagent's is stopped instead, it cannot be
/// answered); stopped by a timeout or /kill.
fn in_netbox_finish(
    command: &str,
    command_id: &str,
    typed: crate::vm::netbox_run::Typed,
    worker: bool,
    account: Option<&str>,
) -> Result<String, Box<dyn Error>> {
    use crate::vm::netbox_run::Waited;
    let audit = Audit { command_id, command, timeout_secs: typed.timeout_secs, netbox: Some(&typed.netbox), account };
    let waited = crate::vm::netbox_run::wait(&typed, &mut crate::vm::netboxes::machine, true, crate::vm::netbox_run::CONFIRM_SECS);
    let waited = match waited {
        Err(error) if crate::kill::requested() => {
            let _cleanup = crate::kill::suspend();
            let stopped = crate::vm::netbox_run::interrupt(&typed, &mut crate::vm::netboxes::machine);
            let mut killed = crate::kill::Killed::new(format!("the command in {} was stopped", crate::vm::netboxes::display(&typed.netbox)));
            if let Ok(Waited::Done(done)) = &stopped {
                killed = killed.with_partial(netbox_result(&typed, done, command_id).to_string());
            }
            audit.record("killed", None, None, Some(&error))?;
            return Err(Box::new(killed));
        }
        Err(error) => {
            audit.record("failed", None, None, Some(&error))?;
            return Err(error.into());
        }
        Ok(waited) => waited,
    };
    match waited {
        Waited::Done(done) => {
            audit.record(if done.declined { "declined" } else { "finished" }, done.exit_code, Some(done.duration_ms), None)?;
            let mut value = netbox_result(&typed, &done, command_id);
            if !done.declined && crate::edit_check::enabled() {
                let ran = done.edited.as_deref().unwrap_or(command);
                match post_edit_diagnostics(command_id, ran, done.duration_ms) {
                    Ok(Some(diagnostics)) => value["diagnostics"] = diagnostics,
                    Ok(None) => {}
                    Err(error) => value["diagnostics_error"] = json!(error),
                }
            }
            Ok(value.to_string())
        }
        Waited::TimedOut => {
            let stopped = crate::vm::netbox_run::interrupt(&typed, &mut crate::vm::netboxes::machine);
            let mut value = json!({"ok": false, "status": "timeout", "shell": typed.netbox, "command_id": command_id,
                "timeout_seconds": typed.timeout_secs, "error": format!("stopped after {} seconds", typed.timeout_secs)});
            if let Ok(Waited::Done(done)) = stopped {
                value["stdout"] = json!(done.output);
                value["exit_code"] = json!(done.exit_code);
            }
            audit.record("timeout", None, None, Some("timed out in the shell"))?;
            Ok(value.to_string())
        }
        Waited::Gone => {
            audit.record("failed", None, None, Some("the shell closed"))?;
            Ok(json!({"ok": false, "status": "shell_closed", "shell": typed.netbox,
                "error": format!("{}'s shell went away (closed, or exited) before the command finished; its output is lost", crate::vm::netboxes::display(&typed.netbox))}).to_string())
        }
        Waited::Input(so_far) if worker => {
            let _ = crate::vm::netbox_run::interrupt(&typed, &mut crate::vm::netboxes::machine);
            audit.record("stopped", None, None, Some("waited for keyboard input"))?;
            Ok(json!({"ok": false, "status": "waiting_for_input", "shell": typed.netbox, "output_so_far": so_far,
                "note": "It sat waiting for keyboard input, so it was stopped: run it so it needs none (a flag, a file, yes |, </dev/null)."}).to_string())
        }
        Waited::Input(so_far) => {
            let value = json!({"ok": true, "status": "waiting_for_input", "shell": typed.netbox, "command_id": command_id, "output_so_far": so_far,
                "note": format!("It is waiting for keyboard input (a prompt, a password, a y/n). The operator sees it in {0} and may answer; \
                    you can with shell_type (shell={1}). Its result comes as an event when it ends: don't wait for it here.",
                    crate::vm::netboxes::display(&typed.netbox), typed.netbox)});
            follow_later(command.to_string(), command_id.to_string(), typed, account.map(str::to_string));
            Ok(value.to_string())
        }
        Waited::StillWaiting => {
            let value = json!({"ok": true, "status": "still_waiting", "shell": typed.netbox, "command_id": command_id,
                "note": format!("The operator has not answered in {} minutes: your command still waits at his prompt in {} for his Enter. \
                    Its result comes as an event if he runs it or declines it.", crate::vm::netbox_run::CONFIRM_SECS / 60, crate::vm::netboxes::display(&typed.netbox))});
            follow_later(command.to_string(), command_id.to_string(), typed, account.map(str::to_string));
            Ok(value.to_string())
        }
    }
}

/// A command whose call returned before it ended: waited for away from the
/// turn, and its result made an event for the agent.
fn follow_later(command: String, command_id: String, typed: crate::vm::netbox_run::Typed, account: Option<String>) {
    let spawned = thread::Builder::new().name("netbox-follow".into()).spawn(move || {
        use crate::vm::netbox_run::Waited;
        let audit = Audit { command_id: &command_id, command: &command, timeout_secs: typed.timeout_secs, netbox: Some(&typed.netbox), account: account.as_deref() };
        let mut failures = 0;
        let waited = loop {
            match crate::vm::netbox_run::wait(&typed, &mut crate::vm::netboxes::machine, false, 12 * 3600) {
                Ok(Waited::TimedOut) => break crate::vm::netbox_run::interrupt(&typed, &mut crate::vm::netboxes::machine).map(|waited| (waited, true)),
                Ok(waited) => break Ok((waited, false)),
                Err(_) if failures < 20 => {
                    failures += 1;
                    thread::sleep(Duration::from_secs(15));
                }
                Err(error) => break Err(error),
            }
        };
        let label = crate::mistakes::one_line(&command, 120);
        let place = crate::vm::netboxes::display(&typed.netbox);
        let text = match waited {
            Ok((Waited::Done(done), timed_out)) => {
                let _ = audit.record(if done.declined { "declined" } else if timed_out { "timeout" } else { "finished" }, done.exit_code, Some(done.duration_ms), None);
                if done.declined {
                    format!("- {place}: `{label}` (command {command_id}) was declined by the operator at his prompt; it did not run.")
                } else {
                    let tail: String = {
                        let chars: Vec<char> = done.output.chars().collect();
                        chars[chars.len().saturating_sub(3000)..].iter().collect()
                    };
                    let mut text = format!(
                        "- {place}: `{label}` (command {command_id}) {} exit {} after {:.1}s.",
                        if timed_out { "ran past its timeout and was stopped:" } else { "finished:" },
                        done.exit_code.map_or("?".into(), |code| code.to_string()),
                        done.duration_ms as f64 / 1000.0
                    );
                    if let Some(edited) = &done.edited {
                        text.push_str(&format!(" the operator edited it first; what ran was `{edited}`."));
                    }
                    text.push_str(&format!("\n  output (the end of it):\n{tail}"));
                    text
                }
            }
            Ok((Waited::Gone, _)) => {
                let _ = audit.record("failed", None, None, Some("the shell closed"));
                format!("- {place}: `{label}` (command {command_id}) ended with its shell (closed, or exited); its output is lost.")
            }
            Ok((_, _)) => {
                format!("- {place}: `{label}` (command {command_id}) was still not run after 12 hours; it may still be at the operator's prompt.")
            }
            Err(error) => {
                format!("- {place}: `{label}` (command {command_id}): its end could not be read ({error}); shell_read shell={} shows where it is.", typed.netbox)
            }
        };
        crate::vm::netboxes::push_event(text);
    });
    if spawned.is_err() {
        crate::vm::netboxes::push_event("- a command that went on after its call returned could not be followed: read its shell to see how it went".into());
    }
}

/// A background job's netbox: its output as it comes, then an hour's
/// grace (Enter opens a shell there) before it closes.
fn open_job_netbox(job: &str, command: &str, why: &str) -> Result<String, String> {
    let name = crate::vm::netboxes::job_name(job);
    let label = crate::mistakes::one_line(command, 200);
    let shown = crate::vm::netboxes::display(&name);
    let program = format!(
        "f=\"$HOME/.s0ma-bg/{job}\"\n\
         [ -n {why} ] && printf '\\033[3;90m# \\033[0;38;2;51;255;51ms0ma\\033[0;3;90m: %s\\033[0m\\n' {why}\n\
         printf '\\033[90m$\\033[0m \\033[38;2;51;255;51m%s\\033[0m\\n' {label}\n\
         printf '\\033[3;90m# background job {job} · ctrl+c closes this view, not the job\\033[0m\\n'\n\
         i=0; while [ ! -s \"$f.pid\" ] && [ $i -lt 50 ]; do sleep 0.1; i=$((i + 1)); done\n\
         tail -n +1 -F --pid=\"$(cat \"$f.pid\" 2>/dev/null || echo 1)\" \"$f.log\" 2>/dev/null\n\
         code=$(cat \"$f.exit\" 2>/dev/null || echo '?')\n\
         if [ \"$code\" = 0 ]; then printf '\\n\\033[38;2;51;255;51m[+]\\033[90m job {job} ended\\033[0m\\n'; \
         else printf '\\n\\033[38;2;255;95;95m[-] job {job} ended: exit %s\\033[0m\\n' \"$code\"; fi\n\
         printf '\\033[3;90menter: a shell · closes in 1 h\\033[0m\\n\\n'\n\
         read -r -t 3600 _ && exec env S0MA_NETBOX={shown_q} bash --rcfile \"$HOME/.s0ma-netbox-rc\" -i\n",
        why = crate::util::shell_quote(why.trim()),
        label = crate::util::shell_quote(&label),
        shown_q = crate::util::shell_quote(&shown),
    );
    let spec = crate::vm::netboxes::Spec { name: name.clone(), purpose: label, by: format!("job:{job}"), program: Some(program), ..Default::default() };
    crate::vm::netboxes::open(&mut crate::vm::netboxes::machine, &spec)?;
    crate::vm::terms::touch();
    Ok(name)
}

/// Check the files `command` wrote (see `edit_check`): a separate, read-only
/// guest command right after it. `None` when it named nothing checkable that
/// it modified. A check that couldn't run is an error to report, so its
/// absence is never taken for a pass; a stopped one (`/kill`) is just skipped.
fn post_edit_diagnostics(command_id: &str, command: &str, duration_ms: u128) -> Result<Option<Value>, String> {
    let (paths, dirs) = crate::edit_check::candidates(command);
    if paths.is_empty() {
        return Ok(None);
    }
    let script = crate::edit_check::script(&paths, &dirs, crate::edit_check::WORKDIR, crate::edit_check::window(duration_ms));
    let audited = format!("[harness post-edit check after {command_id}] {}", paths.join(" "));
    let audit = |event: &str, error: Option<&str>| {
        append_audit(&VmAuditEvent {
            timestamp: Utc::now(),
            command_id,
            event,
            vm: vm_name(),
            command: &audited,
            timeout_secs: EDIT_CHECK_TIMEOUT_SECS,
            exit_code: None,
            signal: None,
            duration_ms: None,
            error,
            job: None,
            netbox: None,
            account: None,
        })
    };
    audit("edit_check", None).map_err(|error| error.to_string())?;
    match run_guest_command(&script, EDIT_CHECK_TIMEOUT_SECS) {
        Ok(GuestCommandOutcome::Exited(check)) => {
            let found = crate::edit_check::parse(&check.stdout);
            if found.is_empty() {
                return Ok(None);
            }
            serde_json::to_value(found).map(Some).map_err(|error| error.to_string())
        }
        Ok(GuestCommandOutcome::HostTimeout { .. }) => {
            let _ = audit("edit_check_timeout", None);
            Err("the post-edit check of the files this command wrote did not finish; they were not checked".into())
        }
        Err(error) if error.downcast_ref::<crate::kill::Killed>().is_some() => Ok(None),
        Err(error) => {
            let message = error.to_string();
            let _ = audit("edit_check_failed", Some(&message));
            Err(format!("the post-edit check could not run ({message}); the files this command wrote were not checked"))
        }
    }
}

/// How long the post-edit check may take (each tool in it gets 8 seconds).
const EDIT_CHECK_TIMEOUT_SECS: u64 = 30;

fn execute_shell_input(call: &ToolCall) -> Result<String, Box<dyn Error>> {
    let args: ShellInputArgs = serde_json::from_str(&call.function.arguments)?;
    let tmux_session = tmux_session_name(&args.session)?;

    if args.text.chars().count() > MAX_SHELL_INPUT_CHARS {
        return Ok(json!({
            "ok": false,
            "session": args.session,
            "error": format!(
                "shell_type text exceeds the {MAX_SHELL_INPUT_CHARS}-character limit"
            )
        })
        .to_string());
    }

    let encoded = STANDARD.encode(args.text.as_bytes());
    let enter = if args.enter {
        format!("; tmux send-keys -t {tmux_session}:0.0 Enter")
    } else {
        String::new()
    };

    let command = format!(
        "tmux has-session -t {tmux_session} 2>/dev/null || exit 44; \
         printf '%s' '{encoded}' | base64 -d | tmux load-buffer -b {tmux_session}-input -; \
         tmux paste-buffer -d -b {tmux_session}-input -t {tmux_session}:0.0{enter}"
    );

    let result = run_shell_helper(&command)?;

    if result.exit_code == Some(44) {
        return Ok(shell_not_open(&args.session));
    }

    if result.exit_code != Some(0) {
        return Ok(shell_helper_error(&args.session, &result));
    }

    Ok(json!({
        "ok": true,
        "session": args.session,
        "written_chars": args.text.chars().count(),
        "written_bytes": args.text.len(),
        "enter_sent": args.enter
    })
    .to_string())
}

fn execute_shell_key(call: &ToolCall) -> Result<String, Box<dyn Error>> {
    let args: ShellKeyArgs = serde_json::from_str(&call.function.arguments)?;
    let tmux_session = tmux_session_name(&args.session)?;

    let tmux_key = match args.key.as_str() {
        "ctrl_c" => "C-c",
        "ctrl_d" => "C-d",
        "ctrl_z" => "C-z",
        "enter" => "Enter",
        "tab" => "Tab",
        "escape" => "Escape",
        "up" => "Up",
        "down" => "Down",
        "left" => "Left",
        "right" => "Right",
        _ => {
            return Ok(json!({
                "ok": false,
                "session": args.session,
                "error": format!("unsupported terminal key: {}", args.key)
            })
            .to_string());
        }
    };

    let command = format!(
        "tmux has-session -t {tmux_session} 2>/dev/null || exit 44; \
         tmux send-keys -t {tmux_session}:0.0 {tmux_key}"
    );

    let result = run_shell_helper(&command)?;

    if result.exit_code == Some(44) {
        return Ok(shell_not_open(&args.session));
    }

    if result.exit_code != Some(0) {
        return Ok(shell_helper_error(&args.session, &result));
    }

    Ok(json!({
        "ok": true,
        "session": args.session,
        "key": args.key
    })
    .to_string())
}

fn execute_shell_read(call: &ToolCall) -> Result<String, Box<dyn Error>> {
    let args: ShellReadArgs = serde_json::from_str(&call.function.arguments)?;
    let tmux_session = tmux_session_name(&args.session)?;

    if !(1..=MAX_SHELL_CAPTURE_LINES).contains(&args.lines) {
        return Ok(json!({
            "ok": false,
            "session": args.session,
            "error": format!(
                "shell_read lines must be between 1 and {MAX_SHELL_CAPTURE_LINES}"
            )
        })
        .to_string());
    }

    const OUTPUT_MARKER: &str = "__S0MA_SHELL_OUTPUT_7F3C1A__";

    let command = format!(
        "tmux has-session -t {tmux_session} 2>/dev/null || exit 44; \
         tmux display-message -p -t {tmux_session}:0.0 \
             '#{{pane_current_command}}|#{{pane_dead}}|#{{pane_dead_status}}|#{{pane_pid}}|#{{pane_current_path}}'; \
         printf '\n{OUTPUT_MARKER}\n'; \
         tmux capture-pane -p -J -t {tmux_session}:0.0 -S -{}",
        args.lines
    );

    let result = run_shell_helper(&command)?;

    if result.exit_code == Some(44) {
        return Ok(shell_not_open(&args.session));
    }

    if result.exit_code != Some(0) {
        return Ok(shell_helper_error(&args.session, &result));
    }

    let marker = format!("\n{OUTPUT_MARKER}\n");
    let (metadata, terminal) = match result.stdout.split_once(&marker) {
        Some(parts) => parts,
        None => {
            return Ok(json!({
                "ok": false,
                "session": args.session,
                "error": "could not parse shell terminal snapshot"
            })
            .to_string());
        }
    };

    // `|` between fields (tmux 3.5 writes a tab as `_`); the folder last.
    let mut fields = metadata.trim().splitn(5, '|');
    let foreground_process = fields.next().unwrap_or("");
    let pane_dead = fields.next().unwrap_or("0") == "1";
    let dead_status = fields.next().filter(|value| !value.is_empty());
    let pane_pid = fields.next().and_then(|value| value.parse::<i64>().ok());
    let cwd = fields.next().unwrap_or("");

    Ok(json!({
        "ok": true,
        "session": args.session,
        "foreground_process": foreground_process,
        "cwd": cwd,
        "pane_dead": pane_dead,
        "dead_status": dead_status,
        "pane_pid": pane_pid,
        "requested_history_lines": args.lines,
        "output": terminal,
        "output_truncated_by_host": result.stdout_truncated
    })
    .to_string())
}

pub(crate) fn run_shell_helper(command: &str) -> Result<GuestCommandResult, Box<dyn Error>> {
    match run_guest_command(command, SHELL_HELPER_TIMEOUT_SECS)? {
        GuestCommandOutcome::Exited(result) => Ok(result),

        GuestCommandOutcome::HostTimeout {
            pid, timeout_secs, ..
        } => Err(format!(
            "the shell helper lost contact with process {pid} on your machine after {} seconds",
            timeout_secs + 5
        )
        .into()),
    }
}

fn run_guest_command(
    command: &str,
    timeout_secs: u64,
) -> Result<GuestCommandOutcome, Box<dyn Error>> {
    run_guest_command_with_capture_limit(command, timeout_secs, MAX_CAPTURE_CHARS)
}

/// Run a short harness-owned helper in the guest (not a model tool call), such
/// as starting an MCP server that lives in the VM.
pub(crate) fn run_helper_command(
    command: &str,
    timeout_secs: u64,
) -> Result<GuestCommandOutcome, Box<dyn Error>> {
    run_guest_command(command, timeout_secs)
}

/// Run a harness-owned helper through the agent runner (so as the agent user)
/// and wait for it the way a long poll needs: status checked every
/// `status_poll` instead of every 100 ms, since each check spawns a `virsh`,
/// and a failed check retried rather than ending the wait, since the helper
/// may already have done its work (taken events off the VM's wake channel,
/// say). A `status_poll` under half a second is for a helper expected back at
/// once, and backs off (`next_status_pause`). Not for model tool calls, which
/// have `run_guest_command`.
pub(crate) fn run_helper_waiting(
    command: &str,
    timeout_secs: u64,
    status_poll: Duration,
) -> Result<GuestCommandOutcome, Box<dyn Error>> {
    if !(1..=MAX_TIMEOUT_SECS).contains(&timeout_secs) {
        return Err(format!("guest timeout must be between 1 and {MAX_TIMEOUT_SECS}").into());
    }
    if crate::kill::requested() {
        return Err(Box::new(crate::kill::Killed::new("not started: the call had already been stopped")));
    }
    let started = Instant::now();
    let request = json!({
        "execute": "guest-exec",
        "arguments": {
            "path": GUEST_RUNNER,
            "arg": [timeout_secs.to_string(), STANDARD.encode(command.as_bytes())],
            "capture-output": true
        }
    });
    let response = qga_command(&request)?;
    let pid = response
        .pointer("/return/pid")
        .and_then(Value::as_i64)
        .ok_or("QEMU guest agent did not return a PID")?;
    // The runner's own timeout ends the helper; give the agent a few seconds
    // past it to report, as `run_guest_command` does.
    wait_retrying(pid, started, Duration::from_secs(timeout_secs + 5), status_poll)
}

/// Wait up to `wait` for guest process `pid`, checking every `status_poll`
/// (see `next_status_pause`) and retrying failed checks. `/kill` ends the
/// wait, not the process.
fn wait_retrying(
    pid: i64,
    started: Instant,
    wait: Duration,
    status_poll: Duration,
) -> Result<GuestCommandOutcome, Box<dyn Error>> {
    let deadline = Instant::now() + wait;
    let mut pause = status_poll;
    loop {
        // Only the latest check counts: one that errors after many that
        // didn't still means the process can't be found right now.
        let last_error = match guest_exec_status(pid) {
            Ok(Some(status)) => {
                return Ok(GuestCommandOutcome::Exited(parse_guest_result(
                    &status,
                    started.elapsed().as_millis(),
                    MAX_CAPTURE_CHARS,
                )?));
            }
            Ok(None) => None,
            Err(error) => Some(error.to_string()),
        };
        if crate::kill::requested() {
            return Err(Box::new(crate::kill::Killed::new(format!(
                "stopped waiting for process {pid}; it may still finish on your machine"
            ))));
        }
        if Instant::now() >= deadline {
            if let Some(error) = last_error {
                return Err(format!("lost track of guest process {pid}: {error}").into());
            }
            return Ok(GuestCommandOutcome::HostTimeout {
                pid,
                timeout_secs: wait.as_secs(),
                duration_ms: started.elapsed().as_millis(),
            });
        }
        thread::sleep(pause);
        pause = next_status_pause(pause, status_poll);
    }
}

/// The pause before the next status check. A poll shorter than half a second
/// (for a command expected to finish at once, like a quick read of the VM wake
/// channel) doubles after each check up to half a second, so one that doesn't
/// finish quickly isn't checked twenty times a second. Longer polls stay as
/// they are.
fn next_status_pause(pause: Duration, status_poll: Duration) -> Duration {
    (pause * 2).min(status_poll.max(Duration::from_millis(500)))
}

/// Execute a bounded desktop operation through the existing guest runner.
/// Desktop frames are base64 text and need a larger host capture allowance;
/// this does not alter the ordinary VM tool's context-output limit.
pub(crate) fn run_desktop_command(
    command: &str,
    timeout_secs: u64,
) -> Result<GuestCommandOutcome, Box<dyn Error>> {
    run_guest_command_with_capture_limit(command, timeout_secs, MAX_DESKTOP_CAPTURE_CHARS)
}

fn run_guest_command_with_capture_limit(
    command: &str,
    timeout_secs: u64,
    max_capture_chars: usize,
) -> Result<GuestCommandOutcome, Box<dyn Error>> {
    if !(1..=MAX_TIMEOUT_SECS).contains(&timeout_secs) {
        return Err(format!("guest timeout must be between 1 and {MAX_TIMEOUT_SECS}").into());
    }

    // Once the call is stopped, nothing more starts in the VM for it; cleanup
    // that must still run does so under kill::suspend().
    if crate::kill::requested() {
        return Err(Box::new(crate::kill::Killed::new("not started: the call had already been stopped")));
    }

    let started = Instant::now();
    let timeout_arg = timeout_secs.to_string();
    let command_b64 = STANDARD.encode(command.as_bytes());

    let request = json!({
        "execute": "guest-exec",
        "arguments": {
            "path": GUEST_RUNNER,
            "arg": [
                timeout_arg,
                command_b64
            ],
            "capture-output": true
        }
    });

    let response = qga_command(&request)?;

    let pid = response
        .pointer("/return/pid")
        .and_then(Value::as_i64)
        .ok_or("QEMU guest agent did not return a PID")?;

    // The guest-side GNU timeout is the primary execution timeout.
    // Give QGA a few extra seconds for cleanup/status collection.
    let deadline = Instant::now() + Duration::from_secs(timeout_secs + 5);

    loop {
        if let Some(status) = guest_exec_status(pid)? {
            return Ok(GuestCommandOutcome::Exited(parse_guest_result(
                &status,
                started.elapsed().as_millis(),
                max_capture_chars,
            )?));
        }

        // The operator's /kill, or a cancelled subagent: stop the command in
        // the VM, not just the wait for it.
        if crate::kill::requested() {
            return Err(Box::new(stop_guest_process(pid, started, max_capture_chars)));
        }

        if Instant::now() >= deadline {
            return Ok(GuestCommandOutcome::HostTimeout {
                pid,
                timeout_secs,
                duration_ms: started.elapsed().as_millis(),
            });
        }

        thread::sleep(Duration::from_millis(100));
    }
}

/// The finished status of a guest process, or `None` while it runs.
fn guest_exec_status(pid: i64) -> Result<Option<Value>, Box<dyn Error>> {
    let response = qga_command(&json!({
        "execute": "guest-exec-status",
        "arguments": { "pid": pid }
    }))?;
    let status = response
        .get("return")
        .ok_or("QEMU guest agent returned no status object")?;
    let exited = status.get("exited").and_then(Value::as_bool).unwrap_or(false);
    Ok(exited.then(|| status.clone()))
}

/// How long a stopped command gets to exit before the agent stops waiting for it.
const STOP_GRACE: Duration = Duration::from_secs(4);

/// The shell run (through the same runner, so as the same user) to stop guest
/// process `pid` and everything under it: TERM for the whole tree, collected
/// first because children of an exited parent are reparented out of reach,
/// then KILL for whatever is still there a second later. A subagent's
/// command runs as its own account (under sudo), which the agent stops with sudo.
pub(crate) fn stop_tree_script(pid: i64) -> String {
    format!(
        "tree() {{ echo \"$1\"; for child in $(ps -o pid= --ppid \"$1\" 2>/dev/null); do tree \"$child\"; done; }}\n\
         pids=$(tree {pid})\n\
         kill -TERM $pids 2>/dev/null || sudo -n kill -TERM $pids 2>/dev/null\n\
         sleep 1\n\
         kill -KILL $pids 2>/dev/null || sudo -n kill -KILL $pids 2>/dev/null\n\
         exit 0\n"
    )
}

/// Stop a guest command for a kill, and say how that went. The stop helper
/// runs to completion whatever the kill token says; only then is the command
/// given a few seconds to exit, so its output so far can be returned.
fn stop_guest_process(pid: i64, started: Instant, max_capture_chars: usize) -> crate::kill::Killed {
    let ran_for = started.elapsed().as_secs_f64();
    let stopped = {
        let _quiet = crate::kill::suspend();
        run_guest_command(&stop_tree_script(pid), 15)
    };
    if let Err(error) = &stopped {
        return crate::kill::Killed::new(format!(
            "could not stop process {pid} after {ran_for:.1}s ({error}); it may still be running on your machine"
        ));
    }
    let until = Instant::now() + STOP_GRACE;
    while Instant::now() < until {
        match guest_exec_status(pid) {
            Ok(Some(status)) => {
                let killed = crate::kill::Killed::new(format!("the command was stopped on your machine after {ran_for:.1}s"));
                return match parse_guest_result(&status, started.elapsed().as_millis(), max_capture_chars) {
                    Ok(result) => killed.with_partial(format_guest_result(&result)),
                    Err(_) => killed,
                };
            }
            Ok(None) => thread::sleep(Duration::from_millis(100)),
            Err(_) => break,
        }
    }
    crate::kill::Killed::new(format!(
        "sent TERM and KILL to guest process {pid} and its children after {ran_for:.1}s, but it had not exited \
         {}s later; it may still be running on your machine",
        STOP_GRACE.as_secs()
    ))
}

fn qga_command(request: &Value) -> Result<Value, Box<dyn Error>> {
    // Host mode: the agent works on this computer, and this one answers.
    if crate::vm::machine::on_host() {
        return crate::vm::machine::agent_command(request);
    }
    let request = serde_json::to_string(request)?;

    let output = Command::new("virsh")
        .args([
            "-c",
            libvirt_uri(),
            "qemu-agent-command",
            vm_name(),
            "--timeout",
            "30",
            &request,
        ])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);

        return Err(format!("virsh qemu-agent-command failed: {}", stderr.trim()).into());
    }

    let stdout = String::from_utf8(output.stdout)?;
    let response: Value = serde_json::from_str(stdout.trim())?;

    if let Some(error) = response.get("error") {
        return Err(format!("QEMU guest agent error: {}", serde_json::to_string(error)?).into());
    }

    Ok(response)
}

/// The VM's libvirt definition (`virsh dumpxml`): the live one while it runs,
/// so paths libvirt fills in at start are there.
pub(crate) fn domain_xml() -> Result<String, Box<dyn Error>> {
    if crate::vm::machine::on_host() {
        return Err("host mode: the agent works on this computer, with no VM".into());
    }
    let output = Command::new("virsh").args(["-c", libvirt_uri(), "dumpxml", vm_name()]).output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("virsh dumpxml failed: {}", stderr.trim()).into());
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn parse_guest_result(
    status: &Value,
    duration_ms: u128,
    max_capture_chars: usize,
) -> Result<GuestCommandResult, Box<dyn Error>> {
    let stdout = decode_capture(status, "out-data")?;
    let stderr = decode_capture(status, "err-data")?;

    let exit_code = status.get("exitcode").and_then(Value::as_i64);
    let signal = status.get("signal").and_then(Value::as_i64);

    let qga_stdout_truncated = status
        .get("out-truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let qga_stderr_truncated = status
        .get("err-truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let (stdout, stdout_host_truncated) = truncate_capture(stdout, max_capture_chars, qga_stdout_truncated);
    let (stderr, stderr_host_truncated) = truncate_capture(stderr, MAX_CAPTURE_CHARS, qga_stderr_truncated);

    Ok(GuestCommandResult {
        exit_code,
        signal,
        stdout,
        stderr,
        stdout_truncated: qga_stdout_truncated || stdout_host_truncated,
        stderr_truncated: qga_stderr_truncated || stderr_host_truncated,
        duration_ms,
    })
}

fn format_guest_result(result: &GuestCommandResult) -> String {
    json!({
        "exit_code": result.exit_code,
        "signal": result.signal,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated
    })
    .to_string()
}

pub(crate) fn tmux_session_name(session: &str) -> Result<String, Box<dyn Error>> {
    if session.is_empty() || session.len() > 32 {
        return Err("shell session name must contain 1-32 characters".into());
    }

    if !session
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        return Err("shell session name may contain only A-Z, a-z, 0-9, _ and -".into());
    }

    Ok(format!("s0ma-{session}"))
}

fn shell_not_open(session: &str) -> String {
    json!({
        "ok": false,
        "session": session,
        "status": "not_open",
        "error": "the shell is not open; shell_open opens one"
    })
    .to_string()
}

fn shell_helper_error(session: &str, result: &GuestCommandResult) -> String {
    json!({
        "ok": false,
        "session": session,
        "exit_code": result.exit_code,
        "signal": result.signal,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": "shell control operation failed"
    })
    .to_string()
}

fn decode_capture(status: &Value, field: &str) -> Result<String, Box<dyn Error>> {
    let Some(encoded) = status.get(field).and_then(Value::as_str) else {
        return Ok(String::new());
    };

    let bytes = STANDARD.decode(encoded)?;

    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// `already_cut`: the guest agent stopped capturing first, so `text` isn't
/// all the command printed, and its length is only a lower bound.
fn truncate_capture(text: String, max_chars: usize, already_cut: bool) -> (String, bool) {
    let total = text.chars().count();
    if total <= max_chars {
        return (text, false);
    }

    let truncated: String = text.chars().take(max_chars).collect();
    let of = if already_cut { "at least " } else { "" };

    (
        format!("{truncated}\n\n[output truncated by the harness: kept the first {max_chars} of {of}{total} characters; the rest was not captured]"),
        true,
    )
}

fn audit_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("vm_audit.jsonl")
}

fn append_audit(event: &VmAuditEvent<'_>) -> Result<(), Box<dyn Error>> {
    let path = audit_path();

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;

    serde_json::to_writer(&mut file, event)?;
    file.write_all(b"\n")?;
    file.flush()?;

    Ok(())
}

/// Record a model-chosen command that runs in the VM outside `vm_exec`: a
/// wake watch, which runs detached, so there is no finish to record. Fails
/// closed as `vm_exec` does: no record, no command.
pub(crate) fn audit_detached(event: &str, command: &str) -> Result<(), Box<dyn Error>> {
    append_audit(&VmAuditEvent {
        timestamp: Utc::now(),
        command_id: &new_command_id(),
        event,
        vm: vm_name(),
        command,
        timeout_secs: 0,
        exit_code: None,
        signal: None,
        duration_ms: None,
        error: None,
        job: None,
        netbox: None,
        account: None,
    })
}

fn new_command_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    format!("{}-{nanos}", std::process::id())
}

// ------------------------------------------------------------------ goal tests

/// Where a board's `command` goal test runs: inside the isolated analysis
/// VM, as the unprivileged the agent user, through the same guest runner `vm_exec`
/// uses. The call comes from the harness, never from the model — the model can
/// only ask for the test, and the exit status is the verdict it does not get to
/// author. A timeout is a failure, not an inconclusive result, because a goal
/// test that cannot finish has not been passed.
pub(crate) struct GoalTest;

impl crate::board::TestRunner for GoalTest {
    fn run(
        &self,
        command: &str,
        timeout_secs: u64,
    ) -> Result<crate::board::TestRun, Box<dyn Error>> {
        if !crate::vm::access::allowed() {
            return Err("not run: the operator has turned off your access to your machine (/vm off), so the goal test cannot run until he turns it back on".into());
        }
        // `audit-run …` is the harness's own command (`crate::audit::run::command`).
        if let Some(run) = crate::audit::run::command(command) {
            return Ok(run);
        }
        crate::approval::ask(&format!("run the goal test: {}", crate::approval::one_line(command)), command)?;
        match run_guest_command(command, timeout_secs)? {
            GuestCommandOutcome::Exited(result) => Ok(crate::board::TestRun {
                exit_code: result.exit_code,
                stdout: result.stdout,
                stderr: result.stderr,
                duration_ms: result.duration_ms,
                timed_out: false,
            }),

            GuestCommandOutcome::HostTimeout { duration_ms, .. } => {
                Ok(crate::board::TestRun {
                    exit_code: None,
                    stdout: String::new(),
                    stderr: format!(
                        "the goal test did not return within {timeout_secs} seconds"
                    ),
                    duration_ms,
                    timed_out: true,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// An old netbox, set up by typing into it, comes back as a clean shell in
    /// the same folder with no leftover lines; a busy one is left alone.
    #[test]
    fn an_old_netbox_restarts_clean_in_its_folder() {
        if std::process::Command::new("tmux").arg("-V").output().is_err() {
            return;
        }
        let root = std::env::temp_dir().join(format!("s0ma-netbox-{}", std::process::id()));
        let home = root.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let sh = |script: &str| {
            std::process::Command::new("bash")
                .arg("-c")
                .arg(script)
                .env("TMUX_TMPDIR", &root)
                .env("HOME", &home)
                .output()
                .unwrap()
        };
        let pane = "s0ma-netbox:0.0";
        sh("tmux new-session -d -s s0ma-netbox -x 80 -y 20 'bash --norc -i' && tmux set-option -t s0ma-netbox @s0ma_netbox 1");
        sh(&format!("sleep 0.3; tmux send-keys -t {pane} 'cd /tmp; echo leftover-line' Enter; sleep 0.3"));
        let ran = sh(&super::netbox_open_script());
        assert!(ran.status.success(), "{}", String::from_utf8_lossy(&ran.stderr));
        let seen = sh(&format!("sleep 0.6; tmux capture-pane -p -S -100 -t {pane}; tmux display-message -p -t {pane} '#{{pane_current_path}} #{{@s0ma_netbox}}'"));
        let seen = String::from_utf8_lossy(&seen.stdout).to_string();
        assert!(!seen.contains("leftover-line"), "{seen}");
        assert!(!seen.contains(".s0ma-netbox-rc"), "nothing typed into it: {seen}");
        assert!(seen.contains("main"), "the first shell's prompt: {seen}");
        assert!(seen.trim_end().ends_with(&format!("/tmp {}", crate::vm::netboxes::VERSION)), "same folder, marked: {seen}");

        // A netbox running something is not restarted under it.
        sh(&format!("tmux set-option -t s0ma-netbox @s0ma_netbox 1; tmux send-keys -t {pane} 'sleep 30' Enter; sleep 0.4"));
        sh(&super::netbox_open_script());
        let busy = sh(&format!("tmux display-message -p -t {pane} '#{{pane_current_command}} #{{@s0ma_netbox}}'; tmux kill-server"));
        assert_eq!(String::from_utf8_lossy(&busy.stdout).trim(), "sleep 1");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The one way a background start could mislead is by reading as a command
    /// that finished, so the result says nothing that could be taken for an
    /// outcome -- no exit code, no stdout -- and says plainly that it is not
    /// one. The schema offers it, and the description says where the output
    /// goes, or the agent would have no way to find it.
    #[test]
    fn a_background_result_cannot_be_read_as_a_finished_command() {
        let tools = super::tool_definitions();
        let exec = tools.iter().find(|tool| tool.function.name == "vm_exec").expect("vm_exec is offered");
        let properties = &exec.function.parameters["properties"];
        assert!(properties.get("background").is_some(), "the option is in the schema");
        assert_eq!(properties["background"]["type"], "boolean");
        let described = &exec.function.description;
        assert!(described.contains("background=true"), "{described}");
        assert!(described.contains("log"), "it says where the output is: {described}");
        assert!(described.contains("wake event"), "and how the end of it arrives: {described}");

        // The shape itself. Built here rather than by starting a job, which
        // would need the machine; what matters is what the fields are.
        let value = serde_json::json!({
            "background": true,
            "job": "bg-1a2b3c4d",
            "pid": 4242,
            "log": "/home/s0ma/.s0ma-bg/bg-1a2b3c4d.log",
            "event_when_done": true,
        });
        assert!(value.get("exit_code").is_none(), "an exit code here would be a lie: {value}");
        assert!(value.get("stdout").is_none(), "{value}");
        assert_eq!(value["background"], true, "and it says what it is");
    }
}
