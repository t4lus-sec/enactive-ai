//! vm_exec in a netbox: the agent's commands where the operator sees them.
//!
//! A command goes to a netbox (`vm/netboxes.rs`): the one agent named, or the one
//! it last worked in, or, for a subagent, its own. It is never typed over
//! the operator: when he is typing there, has a half-typed line or the wheel, or
//! something runs in it, the agent's command waits a little (for his typing) and
//! then goes to the agent's lane beside it (`netbox-s0ma`), and when that is busy
//! too, it runs the old way, by the guest runner, flagged as such.
//!
//! The harness leaves the command in a file and types `__s0ma_run ID ROWS`
//! (the netbox's shim, in `crate::vm::netboxes::rc`): the operator sees the command and its
//! output live, the harness reads the exact output, exit code and time from
//! the shim's files. A command that sits waiting for keyboard input with no
//! output for `QUIET_SECS` comes back early (the agent's result says so, and the
//! rest arrives later as an event); so does one operator has not confirmed in
//! `CONFIRM_SECS`.

use base64::{Engine as _, engine::general_purpose::STANDARD};

use crate::vm::netboxes::{Netbox, Run, Spec};

/// No output for this long while reading the keyboard: waiting for input.
pub const QUIET_SECS: u64 = 5;
/// The operator's Enter, waited for this long before the call returns.
pub const CONFIRM_SECS: u64 = 600;
/// For the operator to stop typing in a netbox before the agent goes to its lane.
const OPERATOR_WAIT: std::time::Duration = std::time::Duration::from_secs(10);
/// A command typed into a netbox at a prompt starts within this.
const START_SECS: u64 = 20;
/// The longest one guest call waits; a longer wait is several.
const CALL_SECS: u64 = 1800;
/// Of a command's output, kept for the agent (as the guest runner keeps).
const MAX_OUTPUT: usize = 1_000_000;

/// What the agent asked for.
#[derive(Clone, Debug, Default)]
pub struct Ask {
    pub command: String,
    pub why: String,
    pub persist: bool,
    pub timeout_secs: u64,
    /// The netbox the agent named.
    pub netbox: Option<String>,
    /// A subagent's call: its run id, and the account it works as.
    pub worker: Option<(String, Option<String>)>,
    /// The operator approved it already (`approval.rs`): no confirm prompt
    /// in the shell as well.
    pub approved: bool,
    /// Whose it is, above it in the shell, when not the agent's or a subagent's
    /// (`audit c9`: an audit's check, drawn in the harness's teal).
    pub label: Option<String>,
    /// Never moved to a lane when the shell is busy: an audit's commands run
    /// in its own shell or wait.
    pub no_lane: bool,
}

/// A command typed into a netbox, being waited for.
#[derive(Clone, Debug)]
pub struct Typed {
    pub id: String,
    pub netbox: String,
    pub user: Option<String>,
    pub confirm: bool,
    /// Why it is not in the netbox the agent meant (it went to the lane).
    pub moved: Option<String>,
    pub timeout_secs: u64,
}

/// What came of starting it.
#[derive(Debug)]
pub enum Started {
    Typed(Typed),
    /// It could not go to a netbox: run it by the guest runner, and say why.
    Elsewhere(String),
}

/// How a wait ended.
#[derive(Debug, PartialEq)]
pub enum Waited {
    Done(Finished),
    /// It sits reading the keyboard with nothing new on screen.
    Input(String),
    /// The operator has not answered the confirm prompt.
    StillWaiting,
    TimedOut,
    /// The netbox's shell is gone (closed, or exited), and the command with it.
    Gone,
}

#[derive(Debug, Default, PartialEq)]
pub struct Finished {
    pub exit_code: Option<i64>,
    pub duration_ms: u128,
    /// stdout and stderr together, as the terminal showed them.
    pub output: String,
    pub truncated: bool,
    pub declined: bool,
    /// Stopped by Ctrl+C in the netbox's own shell (persist).
    pub interrupted: bool,
    /// What the operator changed the command to before he ran it.
    pub edited: Option<String>,
}

fn new_id() -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
    let mixed = nanos ^ u64::from(NEXT.fetch_add(1, Ordering::Relaxed)).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ u64::from(std::process::id());
    format!("{:08x}", (mixed ^ (mixed >> 32)) as u32)
}

/// A script run as `user` when there is one (a subagent's), else as the agent.
fn as_account(user: Option<&str>, script: &str) -> String {
    match user {
        Some(user) => crate::vm::netboxes::as_user(user, script),
        None => script.to_string(),
    }
}

/// A subagent's own shell, and how to open it (as its account, `user`).
pub fn worker_spec(run_id: &str, user: Option<String>) -> Spec {
    Spec {
        name: crate::vm::netboxes::worker_name(run_id),
        purpose: format!("subagent {}'s", crate::vm::netboxes::agent_label(run_id)),
        by: format!("worker:{run_id}"),
        user,
        ..Spec::default()
    }
}

/// The netbox a call goes to first, and how to open it.
fn first_choice(ask: &Ask) -> Spec {
    match &ask.worker {
        Some((run_id, user)) => worker_spec(run_id, user.clone()),
        None => {
            let name = ask.netbox.clone().unwrap_or_else(crate::vm::netboxes::working);
            Spec { name, ..Spec::default() }
        }
    }
}

/// Start `ask` in a netbox: typed there, or why it has to run elsewhere.
pub fn start(ask: &Ask, run: Run) -> Result<Started, String> {
    let mut first = first_choice(ask);
    crate::vm::netboxes::valid_name(&first.name)?;
    // A netbox the agent names must be open, but for the first, which is always
    // there to open. One it worked in that has gone sends it to the first.
    if ask.worker.is_none() && first.name != crate::vm::netboxes::DEFAULT {
        let existing: Vec<Netbox> = crate::vm::netboxes::list(run)?;
        if !existing.iter().any(|netbox| netbox.name == first.name && netbox.open) {
            if ask.netbox.is_some() {
                return Err(format!(
                    "no shell {} is open (open: {}); shell_open opens one",
                    first.name,
                    existing.iter().filter(|netbox| netbox.open).map(|netbox| netbox.name.as_str()).collect::<Vec<_>>().join(", ")
                ));
            }
            crate::vm::netboxes::set_working(crate::vm::netboxes::DEFAULT);
            first = Spec::default_netbox();
        }
    }
    let id = new_id();
    let session = crate::vm::netboxes::session_of(&first.name);
    let theirs = ask.worker.is_none() && !crate::vm::netboxes::is_lane(&first.name);
    let until = std::time::Instant::now() + OPERATOR_WAIT;
    let busy = loop {
        // The operator typing here, or a line he left: wait for him, a while.
        if theirs && let Some(reason) = crate::vm::terms::operator_busy(&session) {
            if std::time::Instant::now() < until && !crate::kill::requested() {
                std::thread::sleep(std::time::Duration::from_millis(400));
                continue;
            }
            break reason.to_string();
        }
        match prepare(ask, &first, &id, run)? {
            Prepared::Typed { confirm } => return Ok(Started::Typed(typed(ask, &first, id, confirm, None))),
            Prepared::Busy(reason) if reason == "line" && theirs && std::time::Instant::now() < until && !crate::kill::requested() => {
                std::thread::sleep(std::time::Duration::from_millis(800));
            }
            Prepared::Busy(reason) => break describe(&reason),
            Prepared::Failed(error) => return Ok(Started::Elsewhere(format!("{} could not take it: {error}", crate::vm::netboxes::display(&first.name)))),
        }
    };
    // A subagent's netbox, a lane, or a shell that keeps its own (an
    // audit's), has no lane of its own.
    if ask.worker.is_some() || ask.no_lane || crate::vm::netboxes::is_lane(&first.name) {
        return Ok(Started::Elsewhere(format!("{} was busy ({busy})", crate::vm::netboxes::display(&first.name))));
    }
    let lane = Spec {
        name: crate::vm::netboxes::lane_of(&first.name),
        purpose: format!("s0ma's own, beside {}", crate::vm::netboxes::display(&first.name)),
        by: "s0ma".into(),
        ..Spec::default()
    };
    let moved = format!("{} was busy ({busy}), so it ran in your lane {}", crate::vm::netboxes::display(&first.name), crate::vm::netboxes::display(&lane.name));
    let opened = crate::vm::netboxes::open(run, &lane);
    if let Err(error) = opened {
        return Ok(Started::Elsewhere(format!("{} was busy ({busy}) and your lane could not open: {error}", crate::vm::netboxes::display(&first.name))));
    }
    match prepare(ask, &lane, &id, run)? {
        Prepared::Typed { confirm } => Ok(Started::Typed(typed(ask, &lane, id, confirm, Some(moved)))),
        Prepared::Busy(reason) => Ok(Started::Elsewhere(format!(
            "{} was busy ({busy}) and so was your lane {} ({})",
            crate::vm::netboxes::display(&first.name),
            crate::vm::netboxes::display(&lane.name),
            describe(&reason)
        ))),
        Prepared::Failed(error) => Ok(Started::Elsewhere(format!("your lane could not take it: {error}"))),
    }
}

fn typed(ask: &Ask, spec: &Spec, id: String, confirm: bool, moved: Option<String>) -> Typed {
    Typed { id, netbox: spec.name.clone(), user: spec.user.clone(), confirm, moved, timeout_secs: ask.timeout_secs }
}

fn describe(reason: &str) -> String {
    match reason {
        "wheel" => "The operator has control".into(),
        "running" => "another command of yours is running in it".into(),
        "line" => "The operator has unfinished input in it".into(),
        reason => match reason.strip_prefix("program ") {
            Some(program) => format!("{program} is running in it"),
            None => reason.to_string(),
        },
    }
}

enum Prepared {
    Typed { confirm: bool },
    Busy(String),
    Failed(String),
}

/// One guest call: the netbox opened if need be, checked free (at its
/// prompt, nothing half-typed, no wheel, no command of the agent's running), the
/// command's files written, and the shim typed. A netbox just made gets a
/// few seconds for its prompt.
fn prepare(ask: &Ask, spec: &Spec, id: &str, run: Run) -> Result<Prepared, String> {
    let session = format!("s0ma-{}", crate::vm::netboxes::session_of(&spec.name));
    let pane = format!("{session}:0.0");
    let files = |confirm: bool| {
        let mut script = format!(
            "d=\"$HOME/.s0ma-netbox/cmd\"; mkdir -p \"$d\" && f=\"$d/{id}\" && \
             printf %s '{}' | base64 -d > \"$f.cmd\" && printf %s '{}' | base64 -d > \"$f.why\"",
            STANDARD.encode(ask.command.as_bytes()),
            STANDARD.encode(ask.why.trim().as_bytes()),
        );
        if ask.persist {
            script.push_str(" && : > \"$f.persist\"");
        }
        // Whose command it is, for the comment above it: a subagent's by its name.
        if let Some((run_id, _)) = &ask.worker {
            script.push_str(&format!(" && printf %s {} > \"$f.who\"", crate::util::shell_quote(&crate::vm::netboxes::agent_label(run_id))));
        } else if let Some(label) = &ask.label {
            script.push_str(&format!(" && printf %s {} > \"$f.who\"", crate::util::shell_quote(label)));
        }
        if confirm {
            script.push_str(" && : > \"$f.confirm\"");
        }
        as_account(spec.user.as_deref(), &script)
    };
    // Pairing holds the agent's own commands, not a subagent's.
    let pairs = if ask.worker.is_none() { "1" } else { "never" };
    let script = format!(
        "S={session}; P={pane}\n\
         opened=$({open}) || {{ echo \"FAILED $opened\"; exit 0; }}\n\
         busy() {{\n\
           [ \"$(tmux show-options -v -t $S @s0ma_wheel 2>/dev/null)\" = 1 ] && {{ echo wheel; return; }}\n\
           r=$(tmux show-options -v -t $S @s0ma_running 2>/dev/null)\n\
           c=$(tmux display-message -p -t $P '#{{pane_current_command}}')\n\
           case \"$c\" in bash|sh|dash|zsh|sudo) ;; *) [ -n \"$r\" ] && echo running || echo \"program $c\"; return ;; esac\n\
           y=$(tmux display-message -p -t $P '#{{cursor_y}}')\n\
           l=$(tmux capture-pane -p -t $P -S \"$y\" -E \"$y\" | sed 's/[[:space:]]*$//')\n\
           case \"$l\" in *'$') ;; *) [ -n \"$r\" ] && echo running || echo line; return ;; esac\n\
           [ -n \"$r\" ] && tmux set-option -u -t $S @s0ma_running\n\
         }}\n\
         b=$(busy); i=0\n\
         case \"$opened\" in *created*) while [ \"$b\" = line ] && [ $i -lt 50 ]; do sleep 0.1; i=$((i + 1)); b=$(busy); done ;; esac\n\
         [ -n \"$b\" ] && {{ echo \"BUSY $b\"; exit 0; }}\n\
         m=$(tmux show-options -v -t $S @s0ma_mode 2>/dev/null); p=$(tmux show-options -gv @s0ma_pair 2>/dev/null)\n\
         if [ {approved} = 0 ] && {{ [ \"$m\" = confirm ] || [ \"$p\" = {pairs} ]; }}; then confirm=1; else confirm=0; fi\n\
         if [ $confirm = 1 ]; then {files_confirm}; else {files_auto}; fi || {{ echo 'FAILED the command could not be left for the netbox'; exit 0; }}\n\
         tmux set-option -t $S @s0ma_running {id}\n\
         x=$(tmux display-message -p -t $P '#{{cursor_x}}'); w=$(tmux display-message -p -t $P '#{{pane_width}}')\n\
         r=$(( (x + {typed_len} + w - 1) / w )); [ $r -gt 9 ] && r=9\n\
         tmux send-keys -t $P -l ' __s0ma_run {id} '\"$r\" && tmux send-keys -t $P Enter || {{ tmux set-option -u -t $S @s0ma_running; echo 'FAILED the keys could not be sent'; exit 0; }}\n\
         echo \"TYPED $confirm\"\n",
        open = crate::vm::netboxes::open_script(spec),
        files_confirm = files(true),
        files_auto = files(false),
        approved = u8::from(ask.approved),
        typed_len = 14 + id.len(),
    );
    let ran = run(&script, 90)?;
    let out = ran.stdout.trim();
    let last = out.lines().last().unwrap_or("");
    if let Some(confirm) = last.strip_prefix("TYPED ") {
        if out.contains("created") {
            crate::vm::netboxes::changed();
            crate::vm::terms::touch();
        }
        return Ok(Prepared::Typed { confirm: confirm == "1" });
    }
    if let Some(reason) = last.strip_prefix("BUSY ") {
        return Ok(Prepared::Busy(reason.to_string()));
    }
    Ok(Prepared::Failed(
        out.strip_prefix("FAILED ").map(str::to_string).unwrap_or_else(|| format!("{}{}", out, ran.stderr.trim())).chars().take(300).collect(),
    ))
}

/// The guest script that waits for command `typed`, as its account, up to
/// `limit` seconds this call: done, reading the keyboard quietly (when
/// `early`), not yet confirmed after `confirm_secs`, timed out, or gone.
/// Done, it reads the files and clears them, and the netbox is free again.
fn wait_script(typed: &Typed, early: bool, confirm_secs: u64, limit: u64) -> String {
    let id = &typed.id;
    let session = format!("s0ma-{}", crate::vm::netboxes::session_of(&typed.netbox));
    let start = if typed.confirm { confirm_secs } else { START_SECS };
    let quiet = if early { QUIET_SECS } else { 0 };
    let inner = format!(
        "f=\"$HOME/.s0ma-netbox/cmd/{id}\"\n\
         reading() {{\n\
           sid=$(cat \"$f.pid\" 2>/dev/null) || return 1\n\
           for p in $(ps -o pid= -s \"$sid\" 2>/dev/null); do\n\
             case \"$(cut -d' ' -f1,2 /proc/$p/syscall 2>/dev/null)\" in\n\
               '0 0x0'|'63 0x0') case \"$(readlink /proc/$p/fd/0 2>/dev/null)\" in /dev/pts/*) return 0 ;; esac ;;\n\
             esac\n\
           done\n\
           return 1\n\
         }}\n\
         begin=$(date +%s)\n\
         while [ ! -f \"$f.status\" ]; do\n\
           now=$(date +%s)\n\
           if [ -f \"$f.shell\" ] && ! kill -0 \"$(cat \"$f.shell\")\" 2>/dev/null && [ ! -f \"$f.status\" ]; then echo __GONE__; rm -f \"$f\".*; exit 0; fi\n\
           if [ -f \"$f.started\" ]; then\n\
             s=$(cat \"$f.started\" 2>/dev/null); s=${{s:-$now}}\n\
             [ $((now - s)) -ge {timeout} ] && {{ echo __TIMEOUT__; exit 0; }}\n\
             if [ {quiet} -gt 0 ] && [ -f \"$f.pid\" ]; then\n\
               m=$(stat -c %Y \"$f.out\" 2>/dev/null || echo \"$now\")\n\
               if [ $((now - m)) -ge {quiet} ] && reading; then echo __WAITING__; tail -c 3000 \"$f.out\" 2>/dev/null; exit 0; fi\n\
             fi\n\
           elif [ ! -f \"$f.cmd\" ]; then echo __GONE__; exit 0\n\
           elif [ $((now - begin)) -ge {start} ]; then echo __STILL__; exit 0\n\
           fi\n\
           [ $((now - begin)) -ge {limit} ] && {{ echo __AGAIN__; exit 0; }}\n\
           sleep 0.2\n\
         done\n\
         echo __DONE__\n\
         cat \"$f.status\"\n\
         echo __EDITED__; [ -f \"$f.edited\" ] && base64 -w0 < \"$f.edited\"; echo\n\
         echo __SIZE__; stat -c %s \"$f.out\" 2>/dev/null || echo 0\n\
         echo __OUT__; head -c {max} \"$f.out\" 2>/dev/null\n\
         rm -f \"$f\".*\n",
        timeout = typed.timeout_secs.max(1),
        max = MAX_OUTPUT + 4096,
    );
    format!(
        "out=$({inner_run}); printf '%s' \"$out\"\n\
         case \"$out\" in __DONE__*|__GONE__*) tmux set-option -u -t {session} @s0ma_running 2>/dev/null ;; esac; true",
        inner_run = as_account(typed.user.as_deref(), &inner),
    )
}

/// The line the harness drew from a command's end (an audit's verdict),
/// for the shell to print under it in the harness's teal.
pub fn verdict(typed: &Typed, text: &str, run: Run) -> Result<(), String> {
    let script = format!(
        "f=\"$HOME/.s0ma-netbox/cmd/{}\"; printf %s '{}' | base64 -d > \"$f.verdict.new\" && mv -f \"$f.verdict.new\" \"$f.verdict\"",
        typed.id,
        STANDARD.encode(text.replace('\n', " ").as_bytes())
    );
    run(&as_account(typed.user.as_deref(), &script), 30).map(|_| ())
}

/// Wait for `typed` (see `wait_script`): how it ended.
pub fn wait(typed: &Typed, run: Run, early: bool, confirm_secs: u64) -> Result<Waited, String> {
    loop {
        let limit = CALL_SECS.min(typed.timeout_secs.max(confirm_secs) + 30);
        let ran = run(&wait_script(typed, early, confirm_secs, limit), limit + 30)?;
        let out = ran.stdout;
        if out.starts_with("__AGAIN__") {
            continue;
        }
        if let Some(rest) = out.strip_prefix("__WAITING__") {
            return Ok(Waited::Input(clean(rest.trim_start_matches('\n'))));
        }
        if out.starts_with("__STILL__") {
            return Ok(Waited::StillWaiting);
        }
        if out.starts_with("__TIMEOUT__") {
            return Ok(Waited::TimedOut);
        }
        if out.starts_with("__GONE__") {
            return Ok(Waited::Gone);
        }
        return match out.strip_prefix("__DONE__\n") {
            Some(rest) => Ok(Waited::Done(parse_done(rest))),
            None => Err(format!("the shell's answer could not be read: {}", format!("{}{}", out, ran.stderr).chars().take(200).collect::<String>())),
        };
    }
}

fn parse_done(rest: &str) -> Finished {
    let (status, rest) = rest.split_once("\n__EDITED__\n").unwrap_or((rest, ""));
    let (edited, rest) = rest.split_once("\n__SIZE__\n").unwrap_or(("", rest));
    let (size, out) = rest.split_once("\n__OUT__\n").unwrap_or(("0", ""));
    let mut finished = Finished::default();
    let mut fields = status.split_whitespace();
    match fields.next() {
        Some("declined") => finished.declined = true,
        Some(code) => {
            finished.exit_code = code.parse().ok();
            finished.duration_ms = fields.next().and_then(|ms| ms.parse().ok()).unwrap_or(0);
            finished.interrupted = fields.next() == Some("interrupted");
        }
        None => {}
    }
    let edited = edited.trim();
    if !edited.is_empty() {
        finished.edited = STANDARD.decode(edited).ok().and_then(|bytes| String::from_utf8(bytes).ok());
    }
    let size: usize = size.trim().parse().unwrap_or(0);
    // What was read is the first `MAX_OUTPUT + 4096` bytes (less the
    // newlines a shell takes off the end).
    finished.truncated = size > MAX_OUTPUT + 4096;
    let mut output = clean(out);
    if output.chars().count() > MAX_OUTPUT {
        output = output.chars().take(MAX_OUTPUT).collect();
        finished.truncated = true;
    }
    if finished.truncated {
        output.push_str(&format!("\n\n[output truncated by the harness: the command wrote {size} bytes; the first part is kept]"));
    }
    finished.output = output;
    finished
}

/// Stop `typed` (a timeout, or /kill): Ctrl+C in its netbox (which also
/// declines one waiting for the operator's Enter), then TERM and KILL for what is
/// left of it; then what it wrote.
pub fn interrupt(typed: &Typed, run: Run) -> Result<Waited, String> {
    let id = &typed.id;
    let pane = format!("s0ma-{}:0.0", crate::vm::netboxes::session_of(&typed.netbox));
    let inner = format!(
        "f=\"$HOME/.s0ma-netbox/cmd/{id}\"\n\
         gone() {{ i=0; while [ ! -f \"$f.status\" ] && [ $i -lt 20 ]; do sleep 0.1; i=$((i + 1)); done; [ -f \"$f.status\" ]; }}\n\
         if ! gone && [ -f \"$f.pid\" ]; then\n\
           sid=$(cat \"$f.pid\"); pkill -TERM -s \"$sid\" 2>/dev/null\n\
           gone || {{ pkill -KILL -s \"$sid\" 2>/dev/null; sleep 0.3; }}\n\
         fi\n\
         true\n"
    );
    run(&format!("tmux send-keys -t {pane} C-c 2>/dev/null\n{}", as_account(typed.user.as_deref(), &inner)), 30)?;
    let short = Typed { timeout_secs: 1, confirm: false, ..typed.clone() };
    match wait(&short, run, false, 0)? {
        Waited::Done(finished) => Ok(Waited::Done(finished)),
        other => Ok(other),
    }
}

/// The terminal's record as text: `script`'s own first and last lines off,
/// colour and cursor codes out, a line overwritten by `\r` as it ended up.
pub fn clean(raw: &str) -> String {
    let mut text = raw;
    if text.starts_with("Script started on ") {
        text = text.split_once('\n').map_or("", |(_, rest)| rest);
    }
    if let Some(at) = text.rfind("Script done on ")
        && (at == 0 || text[..at].ends_with('\n'))
    {
        text = &text[..at];
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut line = String::new();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' => match chars.next() {
                Some('[') => {
                    for c in chars.by_ref() {
                        if ('\x40'..='\x7e').contains(&c) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    while let Some(c) = chars.next() {
                        if c == '\x07' || (c == '\x1b' && chars.peek() == Some(&'\\')) {
                            if c == '\x1b' {
                                chars.next();
                            }
                            break;
                        }
                    }
                }
                _ => {}
            },
            '\r' if chars.peek() == Some(&'\n') => {}
            '\r' => line.clear(),
            '\x08' => {
                line.pop();
            }
            '\n' => {
                out.push_str(&line);
                out.push('\n');
                line.clear();
            }
            c if c < ' ' && c != '\t' => {}
            c => line.push(c),
        }
    }
    out.push_str(&line);
    let kept = out.trim_end_matches('\n');
    if kept.is_empty() { String::new() } else { format!("{kept}\n") }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn the_terminals_record_reads_as_text() {
        let raw = "Script started on 2026-11-05 [COMMAND=\"x\"]\nhello\r\n\x1b[31mred\x1b[0m\r\n10%\r50%\r100%\r\n\x1b]0;title\x07ok\x08K\r\n\r\nScript done on 2026-11-05 [COMMAND_EXIT_CODE=\"0\"]\n";
        assert_eq!(clean(raw), "hello\nred\n100%\noK\n");
        assert_eq!(clean(""), "");
        assert_eq!(clean("no newline"), "no newline\n");
    }
}
