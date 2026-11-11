//! vm_exec in a netbox: s0ma's commands where the operator sees them.
//!
//! A command goes to a netbox (`netboxes.rs`): the one s0ma named, or the one
//! it last worked in, or, for a subagent, its own. It is never typed over
//! The operator: when he is typing there, has a half-typed line or the wheel, or
//! something runs in it, s0ma's command waits a little (for his typing) and
//! then goes to s0ma's lane beside it (`netbox-s0ma`), and when that is busy
//! too, it runs the old way, by the guest runner, flagged as such.
//!
//! The harness leaves the command in a file and types `__s0ma_run ID ROWS`
//! (the netbox's shim, in `netboxes::rc`): the operator sees the command and its
//! output live, the harness reads the exact output, exit code and time from
//! the shim's files. A command that sits waiting for keyboard input with no
//! output for `QUIET_SECS` comes back early (s0ma's result says so, and the
//! rest arrives later as an event); so does one the operator has not confirmed in
//! `CONFIRM_SECS`.

use base64::{Engine as _, engine::general_purpose::STANDARD};

use crate::netboxes::{self, Netbox, Run, Spec};

/// No output for this long while reading the keyboard: waiting for input.
pub const QUIET_SECS: u64 = 5;
/// The operator's Enter, waited for this long before the call returns.
pub const CONFIRM_SECS: u64 = 600;
/// For the operator to stop typing in a netbox before s0ma goes to its lane.
const OPERATOR_WAIT: std::time::Duration = std::time::Duration::from_secs(10);
/// A command typed into a netbox at a prompt starts within this.
const START_SECS: u64 = 20;
/// The longest one guest call waits; a longer wait is several.
const CALL_SECS: u64 = 1800;
/// Of a command's output, kept for s0ma (as the guest runner keeps).
const MAX_OUTPUT: usize = 1_000_000;

/// What s0ma asked for.
#[derive(Clone, Debug, Default)]
pub struct Ask {
    pub command: String,
    pub why: String,
    pub persist: bool,
    pub timeout_secs: u64,
    /// The netbox s0ma named.
    pub netbox: Option<String>,
    /// A subagent's call: its run id, and the account it works as.
    pub worker: Option<(String, Option<String>)>,
    /// The operator approved it already (`approval.rs`): no confirm prompt
    /// in the shell as well.
    pub approved: bool,
}

/// A command typed into a netbox, being waited for.
#[derive(Clone, Debug)]
pub struct Typed {
    pub id: String,
    pub netbox: String,
    pub user: Option<String>,
    pub confirm: bool,
    /// Why it is not in the netbox s0ma meant (it went to the lane).
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

/// A script run as `user` when there is one (a subagent's), else as s0ma.
fn as_account(user: Option<&str>, script: &str) -> String {
    match user {
        Some(user) => netboxes::as_user(user, script),
        None => script.to_string(),
    }
}

/// A subagent's own shell, and how to open it (as its account, `user`).
pub fn worker_spec(run_id: &str, user: Option<String>) -> Spec {
    Spec {
        name: netboxes::worker_name(run_id),
        purpose: format!("subagent {}'s", netboxes::agent_label(run_id)),
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
            let name = ask.netbox.clone().unwrap_or_else(netboxes::working);
            Spec { name, ..Spec::default() }
        }
    }
}

/// Start `ask` in a netbox: typed there, or why it has to run elsewhere.
pub fn start(ask: &Ask, run: Run) -> Result<Started, String> {
    let mut first = first_choice(ask);
    netboxes::valid_name(&first.name)?;
    // A netbox s0ma names must be open, but for the first, which is always
    // there to open. One it worked in that has gone sends it to the first.
    if ask.worker.is_none() && first.name != netboxes::DEFAULT {
        let existing: Vec<Netbox> = netboxes::list(run)?;
        if !existing.iter().any(|netbox| netbox.name == first.name && netbox.open) {
            if ask.netbox.is_some() {
                return Err(format!(
                    "no shell {} is open (open: {}); shell_open opens one",
                    first.name,
                    existing.iter().filter(|netbox| netbox.open).map(|netbox| netbox.name.as_str()).collect::<Vec<_>>().join(", ")
                ));
            }
            netboxes::set_working(netboxes::DEFAULT);
            first = Spec::default_netbox();
        }
    }
    let id = new_id();
    let session = netboxes::session_of(&first.name);
    let theirs = ask.worker.is_none() && !netboxes::is_lane(&first.name);
    let until = std::time::Instant::now() + OPERATOR_WAIT;
    let busy = loop {
        // The operator typing here, or a line he left: wait for him, a while.
        if theirs && let Some(reason) = crate::vm_terms::operator_busy(&session) {
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
            Prepared::Failed(error) => return Ok(Started::Elsewhere(format!("{} could not take it: {error}", netboxes::display(&first.name)))),
        }
    };
    // A subagent's netbox, or a lane, has no lane of its own.
    if ask.worker.is_some() || netboxes::is_lane(&first.name) {
        return Ok(Started::Elsewhere(format!("{} was busy ({busy})", netboxes::display(&first.name))));
    }
    let lane = Spec {
        name: netboxes::lane_of(&first.name),
        purpose: format!("s0ma's own, beside {}", netboxes::display(&first.name)),
        by: "s0ma".into(),
        ..Spec::default()
    };
    let moved = format!("{} was busy ({busy}), so it ran in your lane {}", netboxes::display(&first.name), netboxes::display(&lane.name));
    let opened = netboxes::open(run, &lane);
    if let Err(error) = opened {
        return Ok(Started::Elsewhere(format!("{} was busy ({busy}) and your lane could not open: {error}", netboxes::display(&first.name))));
    }
    match prepare(ask, &lane, &id, run)? {
        Prepared::Typed { confirm } => Ok(Started::Typed(typed(ask, &lane, id, confirm, Some(moved)))),
        Prepared::Busy(reason) => Ok(Started::Elsewhere(format!(
            "{} was busy ({busy}) and so was your lane {} ({})",
            netboxes::display(&first.name),
            netboxes::display(&lane.name),
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
/// prompt, nothing half-typed, no wheel, no command of s0ma's running), the
/// command's files written, and the shim typed. A netbox just made gets a
/// few seconds for its prompt.
fn prepare(ask: &Ask, spec: &Spec, id: &str, run: Run) -> Result<Prepared, String> {
    let session = format!("s0ma-{}", netboxes::session_of(&spec.name));
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
            script.push_str(&format!(" && printf %s {} > \"$f.who\"", crate::util::shell_quote(&netboxes::agent_label(run_id))));
        }
        if confirm {
            script.push_str(" && : > \"$f.confirm\"");
        }
        as_account(spec.user.as_deref(), &script)
    };
    // Pairing holds s0ma's own commands, not a subagent's.
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
        open = netboxes::open_script(spec),
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
            netboxes::changed();
            crate::vm_terms::touch();
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
    let session = format!("s0ma-{}", netboxes::session_of(&typed.netbox));
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
    let pane = format!("s0ma-{}:0.0", netboxes::session_of(&typed.netbox));
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
    use crate::netboxes::tests::Machine;

    #[test]
    fn the_terminals_record_reads_as_text() {
        let raw = "Script started on 2026-11-05 [COMMAND=\"x\"]\nhello\r\n\x1b[31mred\x1b[0m\r\n10%\r50%\r100%\r\n\x1b]0;title\x07ok\x08K\r\n\r\nScript done on 2026-11-05 [COMMAND_EXIT_CODE=\"0\"]\n";
        assert_eq!(clean(raw), "hello\nred\n100%\noK\n");
        assert_eq!(clean(""), "");
        assert_eq!(clean("no newline"), "no newline\n");
    }

    #[test]
    fn a_finished_commands_files_are_read() {
        let edited = STANDARD.encode("make test -j4");
        let done = parse_done(&format!("0 1234\n__EDITED__\n{edited}\n__SIZE__\n6\n__OUT__\nhello\n"));
        assert_eq!(done, Finished { exit_code: Some(0), duration_ms: 1234, output: "hello\n".into(), edited: Some("make test -j4".into()), ..Finished::default() });
        assert!(parse_done("declined\n__EDITED__\n\n__SIZE__\n0\n__OUT__\n").declined);
        assert!(parse_done("130 0 interrupted\n__EDITED__\n\n__SIZE__\n0\n__OUT__\n").interrupted);
        let cut = parse_done("0 5\n__EDITED__\n\n__SIZE__\n9000000\n__OUT__\nabc");
        assert!(cut.truncated && cut.output.contains("9000000 bytes"), "{}", cut.output);
    }

    fn ask(command: &str) -> Ask {
        Ask { command: command.into(), why: "check things".into(), timeout_secs: 30, ..Ask::default() }
    }

    fn open_netbox(run: Run, name: &str) {
        let spec = if name == netboxes::DEFAULT { Spec::default_netbox() } else { Spec { name: name.into(), by: "s0ma".into(), ..Spec::default() } };
        netboxes::open(run, &spec).unwrap();
    }

    fn screen(machine: &Machine, session: &str) -> String {
        machine.run(&format!("tmux capture-pane -p -t s0ma-{session}:0.0"), 5).unwrap().stdout
    }

    fn started(outcome: Started) -> Typed {
        match outcome {
            Started::Typed(typed) => typed,
            Started::Elsewhere(why) => panic!("not in a netbox: {why}"),
        }
    }

    /// The whole of it on a real tmux and bash: a command typed into the
    /// netbox, seen there as if typed, its output, code and time read back.
    fn quiet() -> std::sync::MutexGuard<'static, ()> {
        let guard = crate::vm_terms::TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        crate::vm_terms::forget_typing_for_test();
        guard
    }

    #[test]
    fn a_command_runs_in_the_netbox_where_operator_sees_it() {
        let _quiet = quiet();
        let Some(machine) = Machine::new("run") else { return };
        let mut run = machine.runner();
        let typed = started(start(&ask("echo hello; ls /nonexistent-x; exit 3"), &mut run).unwrap());
        assert_eq!(typed.netbox, netboxes::DEFAULT);
        assert!(!typed.confirm && typed.moved.is_none());
        let Waited::Done(done) = wait(&typed, &mut run, true, CONFIRM_SECS).unwrap() else { panic!("not done") };
        assert_eq!(done.exit_code, Some(3));
        assert!(done.output.starts_with("hello\n") && done.output.contains("No such file"), "{:?}", done.output);
        assert!(!done.output.contains("Script started"), "{:?}", done.output);
        std::thread::sleep(std::time::Duration::from_millis(300));
        let seen = screen(&machine, "netbox");
        assert!(seen.contains("# s0ma: check things"), "the why above it: {seen}");
        assert!(seen.contains("$ echo hello; ls /nonexistent-x; exit 3"), "the command as if typed: {seen}");
        assert!(!seen.contains("__s0ma_run"), "the shim's line is gone: {seen}");
        // The next prompt: a blank line, then the account, the folder and, on
        // the right, how it went; then `$ `.
        let lines: Vec<&str> = seen.lines().collect();
        let after = lines.iter().position(|line| line.trim_end().ends_with("[-] exit 3 · 0.0s") || line.contains("[-] exit 3 · 0.")).expect(&seen);
        let me = String::from_utf8_lossy(&std::process::Command::new("id").arg("-un").output().unwrap().stdout).trim().to_string();
        assert!(lines[after].starts_with(&format!("{me}@main  ")), "{seen}");
        assert!(lines[after - 1].trim().is_empty() && lines[after + 1].starts_with('$'), "{seen}");
        let command = lines.iter().position(|line| line.starts_with("$ echo hello;")).expect(&seen);
        assert!(lines[command - 1] == "# s0ma: check things", "the why between the two lines: {seen}");
        // Who typed what, by colour: s0ma's command green, the operator's blue, the
        // output the terminal's own.
        machine.run("tmux send-keys -t s0ma-netbox:0.0 -l 'echo operator' && tmux send-keys -t s0ma-netbox:0.0 Enter; sleep 0.4", 5).unwrap();
        let coloured = machine.run("tmux capture-pane -e -p -t s0ma-netbox:0.0", 5).unwrap().stdout;
        const GREEN: &str = "\x1b[38;2;51;255;51m";
        const BLUE: &str = "\x1b[38;2;95;175;255m";
        assert!(coloured.contains(&format!("{GREEN}echo hello;")), "s0ma's in green: {coloured:?}");
        assert!(coloured.contains(&format!("{BLUE}echo operator")), "The operator's in blue: {coloured:?}");
        let output = coloured.lines().find(|line| line.ends_with("operator") && !line.contains("echo")).unwrap_or_default();
        assert!(!output.contains(BLUE) && !output.contains(GREEN), "output in neither: {output:?}");
        assert!(coloured.lines().any(|line| line.contains(&format!("{GREEN}s0ma")) && line.contains(": check things")), "the why, s0ma named in green: {coloured:?}");
        assert!(screen(&machine, "netbox").contains("[+] 0."), "the operator's command says how it went too");
        assert!(machine.run("ls ~/.s0ma-netbox/cmd", 5).unwrap().stdout.trim().is_empty(), "its files are cleared");

        // In a subshell: a cd does not stay. With persist, it does.
        let typed = started(start(&ask("cd /tmp"), &mut run).unwrap());
        wait(&typed, &mut run, true, CONFIRM_SECS).unwrap();
        let typed = started(start(&Ask { persist: true, ..ask("cd /tmp && export S0MA_X=kept") }, &mut run).unwrap());
        wait(&typed, &mut run, true, CONFIRM_SECS).unwrap();
        let typed = started(start(&ask("pwd; echo $S0MA_X"), &mut run).unwrap());
        let Waited::Done(done) = wait(&typed, &mut run, true, CONFIRM_SECS).unwrap() else { panic!() };
        assert_eq!(done.output, "/tmp\nkept\n");

        // Waiting for the keyboard: back early, with what it said so far;
        // answered, it finishes.
        let typed = started(start(&ask("printf 'name? '; read n; echo hi $n"), &mut run).unwrap());
        let Waited::Input(so_far) = wait(&typed, &mut run, true, CONFIRM_SECS).unwrap() else { panic!("not waiting") };
        assert!(so_far.contains("name?"), "{so_far:?}");
        machine.run("tmux send-keys -t s0ma-netbox:0.0 operator Enter", 5).unwrap();
        let Waited::Done(done) = wait(&typed, &mut run, false, CONFIRM_SECS).unwrap() else { panic!() };
        assert!(done.output.contains("hi operator"), "{:?}", done.output);

        // Quiet but not reading the keyboard: waited for, not returned early.
        let typed = started(start(&ask("sleep 6; echo woke"), &mut run).unwrap());
        let Waited::Done(done) = wait(&typed, &mut run, true, CONFIRM_SECS).unwrap() else { panic!("returned early") };
        assert_eq!(done.output, "woke\n");

        // Too long: stopped, and its output so far kept.
        let typed = started(start(&Ask { timeout_secs: 1, ..ask("echo before; sleep 60") }, &mut run).unwrap());
        assert_eq!(wait(&typed, &mut run, true, CONFIRM_SECS).unwrap(), Waited::TimedOut);
        let Waited::Done(done) = interrupt(&typed, &mut run).unwrap() else { panic!() };
        assert_eq!(done.exit_code, Some(130));
        assert!(done.output.contains("before"));
        let listed = netboxes::list(&mut run).unwrap();
        assert!(listed.iter().all(|netbox| netbox.running.is_empty()), "free again: {listed:?}");
    }

    /// The operator's netbox busy: s0ma's command goes to its lane, never over him;
    /// set to confirm, it waits for his Enter, his edit is reported, his
    /// Ctrl+C declines it.
    #[test]
    fn busy_netboxes_send_it_to_the_lane_and_confirm_waits_for_operator() {
        let _quiet = quiet();
        let Some(machine) = Machine::new("lane") else { return };
        let mut run = machine.runner();
        open_netbox(&mut run, netboxes::DEFAULT);
        std::thread::sleep(std::time::Duration::from_millis(500));
        // A program running in it.
        machine.run("tmux send-keys -t s0ma-netbox:0.0 'sleep 30' Enter; sleep 0.4", 5).unwrap();
        let typed = started(start(&ask("echo in-the-lane"), &mut run).unwrap());
        assert_eq!(typed.netbox, "main-s0ma");
        assert!(typed.moved.as_deref().unwrap().contains("sleep is running"), "{:?}", typed.moved);
        let Waited::Done(done) = wait(&typed, &mut run, true, CONFIRM_SECS).unwrap() else { panic!() };
        assert_eq!(done.output, "in-the-lane\n");
        assert!(!screen(&machine, "netbox").contains("in-the-lane"), "nothing typed over the operator's");
        machine.run("tmux send-keys -t s0ma-netbox:0.0 C-c; sleep 0.3", 5).unwrap();

        // A line the operator left half-typed (typed straight into the pane, so only
        // the screen shows it): s0ma waits, then goes to the lane.
        machine.run("tmux send-keys -t s0ma-netbox:0.0 -l 'echo half'; sleep 0.3", 5).unwrap();
        let typed = started(start(&ask("echo beside"), &mut run).unwrap());
        assert!(typed.moved.as_deref().unwrap().contains("unfinished input"), "{:?}", typed.moved);
        wait(&typed, &mut run, true, CONFIRM_SECS).unwrap();
        assert!(screen(&machine, "netbox").contains("$ echo half"), "his line untouched");
        machine.run("tmux send-keys -t s0ma-netbox:0.0 C-u; sleep 0.3", 5).unwrap();

        // The wheel.
        machine.run(&netboxes::option_script(netboxes::DEFAULT, "@s0ma_wheel", Some("1")), 5).unwrap();
        let typed = started(start(&ask("true"), &mut run).unwrap());
        assert!(typed.moved.as_deref().unwrap().contains("The operator has control"));
        wait(&typed, &mut run, true, CONFIRM_SECS).unwrap();
        machine.run(&netboxes::option_script(netboxes::DEFAULT, "@s0ma_wheel", None), 5).unwrap();

        // Confirm: typed for the operator, not run until his Enter; his edit reported.
        machine.run(&netboxes::option_script(netboxes::DEFAULT, "@s0ma_mode", Some("confirm")), 5).unwrap();
        let typed = started(start(&ask("echo one"), &mut run).unwrap());
        assert!(typed.confirm && typed.netbox == netboxes::DEFAULT);
        assert_eq!(wait(&typed, &mut run, true, 1).unwrap(), Waited::StillWaiting);
        assert!(screen(&machine, "netbox").contains("enter runs it"));
        machine.run("tmux send-keys -t s0ma-netbox:0.0 ' two' Enter", 5).unwrap();
        let Waited::Done(done) = wait(&typed, &mut run, true, CONFIRM_SECS).unwrap() else { panic!() };
        assert_eq!(done.edited.as_deref(), Some("echo one two"));
        assert_eq!(done.output, "one two\n");
        // Declined.
        let typed = started(start(&ask("echo never"), &mut run).unwrap());
        std::thread::sleep(std::time::Duration::from_millis(400));
        machine.run("tmux send-keys -t s0ma-netbox:0.0 C-c", 5).unwrap();
        let Waited::Done(done) = wait(&typed, &mut run, true, CONFIRM_SECS).unwrap() else { panic!() };
        assert!(done.declined);
        assert!(!screen(&machine, "netbox").contains("\nnever"), "it did not run");

        // Approved already (approval mode): it runs without the shell's own
        // confirm prompt.
        let typed = started(start(&Ask { approved: true, ..ask("echo approved") }, &mut run).unwrap());
        assert!(!typed.confirm);
        let Waited::Done(done) = wait(&typed, &mut run, true, CONFIRM_SECS).unwrap() else { panic!() };
        assert_eq!(done.output, "approved\n");

        // A named netbox must be open.
        let missing = start(&Ask { netbox: Some("nope".into()), ..ask("true") }, &mut run);
        assert!(missing.unwrap_err().contains("shell_open"));
    }
}
