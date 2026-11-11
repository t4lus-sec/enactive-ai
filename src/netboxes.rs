//! Netboxes: the terminals on s0ma's machine that it shares with the operator.
//!
//! Each is a tmux session (`s0ma-netbox` for the first, `s0ma-nb-<name>` for
//! the rest) running bash with the netbox's own settings (`rc`): a prompt in
//! s0ma's colours with the netbox's name, and `__s0ma_run`, the shim s0ma's
//! commands run through when vm_exec runs them in a netbox (`netbox_run.rs`).
//! What a netbox is for, who opened it, whether s0ma asks before running in
//! it, and whether the operator has taken control are kept on the tmux session as user
//! options, so every terminal (an attached one too) and a restarted s0ma see
//! the same thing without a file of their own.
//!
//! Up to `MAX` netboxes opened by the operator or s0ma; a subagent's own netbox and a
//! background job's are not counted, and close themselves.
//!
//! Subagents work as their own account on s0ma's machine (`worker_user`):
//! their netboxes and shells run bash as that account, and their commands
//! run through `sudo -n -u <account>`, so a subagent cannot touch s0ma's
//! home, keys or memory. /workspace is shared between the two (ACLs, set up
//! once by `ensure_worker_account`).

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::share::Ran;

/// The first netbox, always there for the operator (looking at it opens it).
/// Called a shell where anyone sees it, this one `main`.
pub const DEFAULT: &str = "main";
/// The first netbox's tmux session, as it has been since before the rename.
pub const DEFAULT_SESSION: &str = "netbox";
/// Netboxes the operator and s0ma may have open at once.
pub const MAX: usize = 8;
/// Bumped when an open netbox needs restarting into a newer setup (the
/// default netbox was 2 before there were several).
pub const VERSION: u32 = 8;
/// What a lane is called after the netbox it is beside: `netbox-s0ma`.
const LANE: &str = "-s0ma";
/// A netbox name's longest.
const NAME_MAX: usize = 24;

/// A runner of guest commands: the machine, or the local one in tests.
pub type Run<'a> = &'a mut dyn FnMut(&str, u64) -> Result<Ran, String>;

/// s0ma's machine, through the guest runner (as the s0ma user).
pub fn machine(command: &str, timeout_secs: u64) -> Result<Ran, String> {
    crate::share::run_on_machine(command, timeout_secs)
}

// ------------------------------------------------------------------ names

/// A usable netbox name: lowercase letters, digits, `-` and `_`.
pub fn valid_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > NAME_MAX {
        return Err(format!("a shell name has 1 to {NAME_MAX} characters"));
    }
    if name.starts_with('-') || !name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_') {
        return Err("a shell name has lowercase letters, digits, - and _ (not first)".into());
    }
    Ok(())
}

/// The shell session (as vm_terms and vm_shell_* name them, without
/// `s0ma-`) a netbox is.
pub fn session_of(name: &str) -> String {
    if name == DEFAULT { DEFAULT_SESSION.to_string() } else { format!("nb-{name}") }
}

/// The netbox a shell session is, if it is one.
pub fn name_of(session: &str) -> Option<&str> {
    if session == DEFAULT_SESSION { Some(DEFAULT) } else { session.strip_prefix("nb-").filter(|name| !name.is_empty()) }
}

/// A name as a call gives it: `netbox`, the first one's name before the
/// rename, is `main`.
pub fn canonical(name: &str) -> String {
    let name = name.trim();
    if name == DEFAULT_SESSION { DEFAULT.to_string() } else { name.to_string() }
}

/// A netbox's name as the operator sees it: a subagent's `w·b3-1`, a lane `netbox·s0ma`.
pub fn display(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("w-") {
        return format!("w·{rest}");
    }
    match name.strip_suffix(LANE) {
        Some(base) if !base.is_empty() => format!("{base}·s0ma"),
        _ => name.to_string(),
    }
}

/// The lane beside `name`: s0ma's own netbox for when that one is busy.
pub fn lane_of(name: &str) -> String {
    let mut base = name.to_string();
    base.truncate(NAME_MAX - LANE.len());
    format!("{}{LANE}", base.trim_end_matches('-'))
}

pub fn is_lane(name: &str) -> bool {
    name.ends_with(LANE)
}

/// A subagent's netbox, by its run id: `w-b3-1` for `subagent-…-3-1`.
pub fn worker_name(run_id: &str) -> String {
    let parts: Vec<&str> = run_id.split('-').collect();
    let short = match parts.as_slice() {
        ["subagent", _, _, batch, slot] => format!("b{batch}-{slot}"),
        _ => run_id.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').collect::<String>().to_ascii_lowercase(),
    };
    let mut name = format!("w-{}", short.trim_matches('-'));
    name.truncate(NAME_MAX);
    name
}

/// A subagent as the operator knows it, by its run id: `b3-1` for
/// `subagent-…-3-1` (its tab's name, and its name above its commands).
pub fn agent_label(run_id: &str) -> String {
    match worker_name(run_id).strip_prefix("w-") {
        Some(short) if !short.is_empty() => short.to_string(),
        _ => run_id.to_string(),
    }
}

/// A background job's netbox: `job-1a2b3c4d` for `bg-1a2b3c4d`.
pub fn job_name(job: &str) -> String {
    let mut name = format!("job-{}", job.strip_prefix("bg-").unwrap_or(job).to_ascii_lowercase());
    name.retain(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    name.truncate(NAME_MAX);
    name
}

// ------------------------------------------------------------------ the list

/// A netbox as the machine has it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Netbox {
    pub name: String,
    /// What it is for, in the words of whoever opened it.
    pub purpose: String,
    /// `operator`, `s0ma`, `worker:<run id>`, `job:<name>`; empty for the
    /// default netbox.
    pub opened_by: String,
    /// s0ma's commands wait for the operator's Enter here.
    pub confirm: bool,
    /// Its shell is the subagents' account's.
    pub worker: bool,
    /// The s0ma command running in it, by id; empty when none.
    pub running: String,
    /// The operator has taken control: s0ma does not type into it.
    pub wheel: bool,
    /// The program in front in it (`bash` at a prompt).
    pub command: String,
    /// /pair is on: s0ma's commands wait for the operator's Enter everywhere.
    pub pair: bool,
    /// Open on the machine (the default netbox is listed either way).
    pub open: bool,
    /// The account its shell runs as (`@s0ma_user`); empty for one opened
    /// before it was kept.
    pub user: String,
}

impl Netbox {
    /// Opened by the operator or s0ma, so counted against `MAX`.
    pub fn counted(&self) -> bool {
        !self.opened_by.starts_with("worker:") && !self.opened_by.starts_with("job:")
    }

    pub fn is_workers(&self) -> bool {
        self.opened_by.starts_with("worker:")
    }

    /// The account its shell runs as, as far as is known.
    pub fn account(&self) -> String {
        if !self.user.is_empty() {
            return self.user.clone();
        }
        if self.worker { worker_user().unwrap_or_else(|| "the subagents' account".into()) } else { "s0ma's account".into() }
    }

    /// Why closing it now would cut something short, if it would.
    pub fn busy(&self) -> Option<String> {
        busy_reason(self)
    }

    /// s0ma's commands here wait for the operator's Enter.
    pub fn asks_first(&self) -> bool {
        self.confirm || (self.pair && !self.worker && !self.opened_by.starts_with("job:"))
    }
}

/// The guest command that lists the shells, a netbox's settings beside it,
/// split by `|` (tmux writes a tab as `_`; nothing listed holds a `|`: the
/// purpose is base64).
pub const LIST: &str = "tmux list-sessions -F '#{session_name}|#{@s0ma_netbox}|#{@s0ma_purpose}|#{@s0ma_by}|#{@s0ma_mode}|#{@s0ma_account}|#{@s0ma_running}|#{@s0ma_wheel}|#{pane_current_command}|#{@s0ma_pair}|#{@s0ma_user}' 2>/dev/null; true";

/// `LIST`'s answer: every shell session (without `s0ma-`), and the netboxes
/// among them, the default one first and listed even when it is not open.
pub fn parse(out: &str) -> (Vec<String>, Vec<Netbox>) {
    let mut sessions = Vec::new();
    let mut netboxes = Vec::new();
    let mut pair = false;
    for line in out.lines() {
        let fields: Vec<&str> = line.split('|').collect();
        let Some(session) = fields.first().and_then(|name| name.trim().strip_prefix("s0ma-")) else { continue };
        if session.is_empty() {
            continue;
        }
        sessions.push(session.to_string());
        let field = |index: usize| fields.get(index).map_or("", |field| field.trim());
        pair = pair || field(9) == "1";
        let Some(name) = name_of(session) else { continue };
        netboxes.push(Netbox {
            name: name.to_string(),
            purpose: STANDARD.decode(field(2)).ok().and_then(|bytes| String::from_utf8(bytes).ok()).unwrap_or_default(),
            opened_by: field(3).to_string(),
            confirm: field(4) == "confirm",
            worker: field(5) == "worker",
            running: field(6).to_string(),
            wheel: field(7) == "1",
            command: field(8).to_string(),
            pair: field(9) == "1",
            open: true,
            user: field(10).to_string(),
        });
    }
    if !netboxes.iter().any(|netbox| netbox.name == DEFAULT) {
        netboxes.push(Netbox { name: DEFAULT.into(), open: false, ..Netbox::default() });
    }
    for netbox in &mut netboxes {
        netbox.pair = pair;
    }
    netboxes.sort_by_key(|netbox| (rank(netbox), netbox.name.clone()));
    (sessions, netboxes)
}

/// The default netbox, the operator's and s0ma's, lanes, jobs, then subagents'.
fn rank(netbox: &Netbox) -> u8 {
    if netbox.name == DEFAULT {
        0
    } else if netbox.is_workers() {
        4
    } else if netbox.opened_by.starts_with("job:") {
        3
    } else if is_lane(&netbox.name) {
        2
    } else {
        1
    }
}

pub fn list(run: Run) -> Result<Vec<Netbox>, String> {
    let ran = run(LIST, 30)?;
    Ok(parse(&ran.stdout).1)
}

// ------------------------------------------------------------------ the subagents' account

/// The account subagents work as on s0ma's machine: `S0MA_WORKER_USER`, or
/// `s0ma-worker`. `off` (or `none`) has them work as s0ma, as before.
pub fn worker_user() -> Option<String> {
    let chosen = std::env::var("S0MA_WORKER_USER").ok().map(|value| value.trim().to_string()).filter(|value| !value.is_empty());
    match chosen.as_deref() {
        Some("off" | "none") => None,
        Some(user) if user.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') => Some(user.to_string()),
        Some(_) | None => Some("s0ma-worker".to_string()),
    }
}

/// The subagents' account, when this thread is running a subagent's call
/// and they have one.
pub fn acting_worker() -> Option<String> {
    crate::exec_feed::owner().and(worker_user())
}

/// `script`, run as `user` by bash in the user's home settings, from s0ma's
/// runner (s0ma may act as that account without a password).
/// The script goes in on stdin, so its size is not an argument's; what it
/// runs reads /dev/null, as a guest command does.
pub fn as_user(user: &str, script: &str) -> String {
    format!("printf %s '{}' | sudo -n -u {user} -H -- bash -c 's=$(base64 -d); exec </dev/null; eval \"$s\"'", STANDARD.encode(script))
}

/// Set up the subagents' account once a run: made if missing (s0ma has
/// passwordless sudo on its machine), /workspace shared both ways and /share
/// readable, and s0ma's own home closed to it. Each step only when needed.
pub fn ensure_worker_account(run: Run, user: &str) -> Result<(), String> {
    static READY: AtomicBool = AtomicBool::new(false);
    if READY.load(Ordering::SeqCst) {
        return Ok(());
    }
    let ran = run(&provision_script(user), 600)?;
    match ran.exit_code {
        Some(0) => {
            READY.store(true, Ordering::SeqCst);
            Ok(())
        }
        _ => Err(format!(
            "the subagents' account {user} is not ready on s0ma's machine: {}. Subagents work as that account, \
             so their commands do not run until it is (S0MA_WORKER_USER=off has them work as s0ma)",
            first_line(&format!("{}{}", ran.stderr, ran.stdout))
        )),
    }
}

fn provision_script(user: &str) -> String {
    format!(
        r#"u={user}
me=$(id -un)
if ! id -u "$u" >/dev/null 2>&1; then
  sudo -n true 2>/dev/null || {{ echo "it does not exist, and $me cannot make it (sudo -n fails)" >&2; exit 45; }}
  sudo -n useradd --create-home --shell /bin/bash --comment "s0ma's subagents" "$u" || {{ echo "useradd $u failed" >&2; exit 46; }}
fi
sudo -n -u "$u" true 2>/dev/null || {{ echo "$me cannot act as $u (sudo -n -u $u fails)" >&2; exit 45; }}
share() {{
  [ -d "$1" ] || return 0
  getfacl -p "$1" 2>/dev/null | grep -q "^default:user:$u:$3" && return 0
  command -v setfacl >/dev/null 2>&1 || {{ echo "setfacl is missing (the acl package): $1 cannot be shared with $u" >&2; exit 47; }}
  sudo -n setfacl -R -m "u:$u:$2,d:u:$u:$2,d:u:$me:rwX" "$1" || {{ echo "setfacl on $1 failed" >&2; exit 48; }}
}}
share /workspace rwX rwx
share /share rX r-x
case "$(stat -c %a "$HOME")" in *0) ;; *) chmod o-rwx "$HOME" ;; esac
echo ready
"#
    )
}

fn first_line(text: &str) -> String {
    text.lines().map(str::trim).find(|line| !line.is_empty()).unwrap_or("no reason given").chars().take(240).collect()
}

// ------------------------------------------------------------------ the shell's settings

/// The netbox's settings, sourced by its bash: the operator's ~/.bashrc, a
/// prompt of two lines, colour for ls and grep, and the shim s0ma's commands
/// run through. Who typed what is in its colour, s0ma's page's own: what the
/// operator types at the prompt is blue (the prompt ends in blue, and PS0
/// resets it before the command runs), s0ma's commands green, output the
/// terminal's own.
///
/// The prompt (`claude/netbox.md`, "How a shell reads"): after a command, a
/// blank line; then `account@name  folder` (bold, dim cyan) with, on the
/// right, how the last command went, `[+] 41.2s` or, in red, `[-] exit 101 ·
/// 3.1s`; then `$ ` and the command. A command's further lines (PS2, and the
/// shim's) are set in under it by the `$ `'s two columns. s0ma's reason has a
/// line of its own between the two. PS0 notes when a command starts (an
/// empty line, or one abandoned with Ctrl+C, runs nothing and says nothing);
/// the prompt hook lays the first line out. The shell's tab can fold it back
/// to one line (`compact` in `tabs.rs`): it knows the lines by these shapes.
///
/// `__s0ma_run ID ROWS` runs the command s0ma left in `$d/ID.cmd`. The line
/// that called it is taken off the screen (ROWS rows: the harness measured
/// them before typing it), and in its place the operator sees `# s0ma: why` dimmed
/// and the prompt with the command, as if typed. The command runs under
/// `script`, so it has a terminal and the operator sees it live, while all it writes
/// is kept in `ID.out`; its exit code and how long it took go to
/// `ID.status`, which says it is done. With `ID.confirm` the command waits at
/// a prompt of its own for the operator: Enter runs it (edited, if he edited it,
/// noted in `ID.edited`), Ctrl+C declines it. With `ID.persist` it runs in
/// the netbox's own shell, so a cd or an export stays (its output then goes
/// through a pipe, not a terminal). A command cut short by Ctrl+C is noted
/// by the prompt hook, which runs at the next prompt.
const RC: &str = r#"[ -f ~/.bashrc ] && . ~/.bashrc
case "$TERM" in screen|tmux|"") export TERM=screen-256color ;; esac
export CLICOLOR=1
alias ls='ls --color=auto' grep='grep --color=auto' diff='diff --color=auto' 2>/dev/null
HISTCONTROL="${HISTCONTROL:+$HISTCONTROL:}ignorespace"
__s0ma_name=${S0MA_NETBOX:-main}
__s0ma_green=$'\033[38;2;51;255;51m' __s0ma_red=$'\033[38;2;255;95;95m'
__s0ma_user=$(id -un 2>/dev/null || echo "$USER")
__s0ma_last='\[\033[0;90m\]$\[\033[0m\] \[\033[38;2;95;175;255m\]'
__s0ma_meta=
PS1='${__s0ma_meta}'"$__s0ma_last"
PS2='  '
PS0=$'\033[0m''${__s0ma_at[__s0ma_t0=${EPOCHREALTIME/./}]}'
__s0ma_d="$HOME/.s0ma-netbox/cmd"
__s0ma_done() { printf '%s\n' "$2" > "$__s0ma_d/$1.status.new" && mv -f "$__s0ma_d/$1.status.new" "$__s0ma_d/$1.status"; }
# The prompt's first line: account@name and the folder, and on the right how
# the last command went (when one ran): [+] 41.2s, or in red [-] exit 101 · 3.1s.
# The pane's width as it is now (COLUMNS lags behind a resize until a command runs).
__s0ma_cols() { local size; size=$(stty size 2>/dev/null) && [ -n "${size#* }" ] && COLUMNS=${size#* }; }
# A command as s0ma's lines show it: each line cut to the width less the
# `$ `'s two columns (at a space where there is one), the rest set in by two.
__s0ma_fold() {
  local line out= first=1 room=$(( ${2:-80} > 12 ? ${2:-80} : 12 )) chunk cut
  while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$first" ] && out+=$'\n  '
    first=
    while [ ${#line} -gt $room ]; do
      chunk=${line:0:$room}; cut=${chunk% *}
      [ ${#cut} -lt $(( room / 2 )) ] && cut=$chunk
      out+="$cut"$'\n  '; line=${line:${#cut}}; line=${line# }
    done
    out+="$line"
  done <<< "$1"
  printf '%s' "$out"
}
__s0ma_line() {
  __s0ma_cols
  local s=$1 ms=$2 took= said= shown where=${PWD/#$HOME/\~} pad w=${COLUMNS:-80}
  # How it went sits at the right, no further out than 80 columns, so a
  # narrower pane later rarely wraps it.
  [ "$w" -gt 81 ] && w=81
  if [ -n "$ms" ]; then
    if [ "$ms" -lt 60000 ]; then took="$((ms / 1000)).$(( ms % 1000 / 100 ))s"
    elif [ "$ms" -lt 3600000 ]; then printf -v took '%dm %02ds' $((ms / 60000)) $((ms / 1000 % 60))
    else printf -v took '%dh %02dm' $((ms / 3600000)) $((ms / 60000 % 60)); fi
    if [ -n "$__s0ma_note" ]; then said="[-] $__s0ma_note"
    elif [ "$s" -eq 0 ]; then said="[+] $took"
    else said="[-] exit $s · $took"; fi
  fi
  __s0ma_note=
  shown="$__s0ma_user@$__s0ma_name  $where"
  __s0ma_meta=$'\001\033[0;1m\002'"$__s0ma_user@$__s0ma_name"$'\001\033[0;2;36m\002'"  $where"$'\001\033[0m\002'
  if [ -n "$said" ]; then
    pad=$(( w - ${#shown} - ${#said} - 1 )); [ $pad -lt 2 ] && pad=2
    printf -v pad '%*s' $pad ''
    if [ "${said:1:1}" = + ]; then __s0ma_meta+="$pad"$'\001'"$__s0ma_green"$'\002[+]\001\033[0;90m\002'"${said:3}"$'\001\033[0m\002'
    else __s0ma_meta+="$pad"$'\001'"$__s0ma_red"$'\002'"$said"$'\001\033[0m\002'; fi
    __s0ma_meta=$'\n'"$__s0ma_meta"
  fi
  __s0ma_meta+=$'\n'
}
__s0ma_after() {
  local s=$?
  if [ -n "$__s0ma_pending" ]; then
    if [ "$__s0ma_stage" = confirm ]; then __s0ma_done "$__s0ma_pending" declined; __s0ma_note=declined
    else __s0ma_done "$__s0ma_pending" "130 0 interrupted"; fi
    __s0ma_pending=
  fi
  if [ -n "$__s0ma_t0" ]; then
    __s0ma_line "$s" $(( (${EPOCHREALTIME/./} - __s0ma_t0) / 1000 ))
    __s0ma_t0=
  else
    __s0ma_line "$s" ""
  fi
  return $s
}
case ";${PROMPT_COMMAND};" in *";__s0ma_after;"*) ;; *) PROMPT_COMMAND="__s0ma_after${PROMPT_COMMAND:+;$PROMPT_COMMAND}" ;; esac
__s0ma_run() {
  local s=$? id=$1 r=${2:-1} f="$__s0ma_d/$1" cmd why who line code start pr pd
  [ -r "$f.cmd" ] || return $s
  printf '%s' "$$" > "$f.shell"
  printf '\r'; while [ "$r" -gt 0 ]; do printf '\033[1A\033[2K'; r=$((r - 1)); done
  cmd=$(cat "$f.cmd")
  why=$(cat "$f.why" 2>/dev/null)
  who=$(cat "$f.who" 2>/dev/null)
  [ -n "$why" ] && printf '\033[0;3;90m# \033[0m%s%s\033[0;3;90m: %s\033[0m\n' "$__s0ma_green" "${who:-s0ma}" "$why"
  pr="$( (exit $s); printf '%s' "${__s0ma_last@P}" )"
  pd=${pr//$'\001'/}; pd=${pd//$'\002'/}
  __s0ma_pending=$id
  if [ -e "$f.confirm" ]; then
    __s0ma_stage=confirm
    printf '\033[0;3;90m# %s asks · enter runs it · ctrl+c declines\033[0m\n' "${who:-s0ma}"
    IFS= read -r -e -i "$cmd" -p "$pr"$'\001'"$__s0ma_green"$'\002' line || { printf '\033[0m'; __s0ma_pending=; __s0ma_done "$id" declined; __s0ma_note=declined; return 130; }
    printf '\033[0m'
    if [ "$line" != "$cmd" ]; then printf '%s' "$line" > "$f.edited"; cmd=$line; printf '%s\n' "$cmd" > "$f.cmd"; fi
  else
    __s0ma_cols
    printf '%s%s%s\033[0m\n' "$pd" "$__s0ma_green" "$(__s0ma_fold "$cmd" $(( ${COLUMNS:-80} - 2 )))"
  fi
  __s0ma_stage=run
  history -s "$cmd"
  date +%s > "$f.started"
  start=${EPOCHREALTIME/./}
  __s0ma_t0=$start
  if [ -e "$f.persist" ]; then
    eval "$cmd" > >(tee "$f.out") 2>&1
    code=$?
    wait $! 2>/dev/null
  else
    SHELL=/bin/bash script -q -e -f -c "echo \$\$ > '$f.pid'; exec bash '$f.cmd'" "$f.out"
    code=$?
  fi
  __s0ma_pending=
  __s0ma_done "$id" "$code $(( (${EPOCHREALTIME/./} - start) / 1000 ))"
  return $code
}
"#;

/// What opening a netbox asks for.
#[derive(Clone, Debug, Default)]
pub struct Spec {
    pub name: String,
    pub purpose: String,
    /// `operator`, `s0ma`, `worker:<run id>`, `job:<name>`, or empty.
    pub by: String,
    /// The account its shell is, when not s0ma's.
    pub user: Option<String>,
    pub confirm: bool,
    /// Run this in it instead of the shell (a job's follow script), as s0ma.
    pub program: Option<String>,
}

impl Spec {
    pub fn default_netbox() -> Spec {
        Spec { name: DEFAULT.into(), ..Spec::default() }
    }
}


/// The guest script that opens netbox `spec` when it is not open (exit 0
/// either way; `created` on stdout when it made it). One set up by an older
/// s0ma is restarted in place the first time it sits at an idle shell: same
/// folder, a clean screen, the new settings. Nothing is typed into it. A
/// busy one is left alone until it is idle.
pub fn open_script(spec: &Spec) -> String {
    let rc = STANDARD.encode(RC.as_bytes());
    let session = format!("s0ma-{}", session_of(&spec.name));
    let pane = format!("{session}:0.0");
    let label = crate::util::shell_quote(&display(&spec.name));
    let (install, shell) = match &spec.user {
        None => (
            format!(
                "printf '%s' '{rc}' | base64 -d > \"$HOME/.s0ma-netbox-rc\" && mkdir -p \"$HOME/.s0ma-netbox/cmd\" && chmod 700 \"$HOME/.s0ma-netbox\" && \
                 find \"$HOME/.s0ma-netbox/cmd\" -type f -mmin +1440 -delete 2>/dev/null; "
            ),
            format!("env S0MA_NETBOX={label} bash --rcfile \"$HOME/.s0ma-netbox-rc\" -i"),
        ),
        Some(user) => (
            format!(
                "printf '%s' '{rc}' | sudo -n -u {user} -H -- sh -c 'base64 -d > \"$HOME/.s0ma-netbox-rc\" && mkdir -p \"$HOME/.s0ma-netbox/cmd\" && chmod 700 \"$HOME/.s0ma-netbox\"' || {{ echo 'cannot set up a netbox as {user}' >&2; exit 45; }}; "
            ),
            // `~user` is expanded by the shell tmux runs this with.
            format!("sudo -n -u {user} -H -- env S0MA_NETBOX={label} bash --rcfile ~{user}/.s0ma-netbox-rc -i"),
        ),
    };
    let shell = match &spec.program {
        Some(program) => format!("bash -c {}", crate::util::shell_quote(program)),
        None => shell,
    };
    let set = |key: &str, value: &str| format!("tmux set-option -t {session} {key} {} >/dev/null; ", crate::util::shell_quote(value));
    let version = set("@s0ma_netbox", &VERSION.to_string());
    let mut options = version.clone();
    if !spec.purpose.is_empty() {
        options.push_str(&set("@s0ma_purpose", &STANDARD.encode(spec.purpose.as_bytes())));
    }
    if !spec.by.is_empty() {
        options.push_str(&set("@s0ma_by", &spec.by));
    }
    options.push_str(&set("@s0ma_mode", if spec.confirm { "confirm" } else { "auto" }));
    options.push_str(&set("@s0ma_account", if spec.user.is_some() { "worker" } else { "s0ma" }));
    // The account by name: the subagents', or whoever s0ma runs as.
    let user = match &spec.user {
        Some(user) => crate::util::shell_quote(user),
        None => "\"$(id -un)\"".to_string(),
    };
    options.push_str(&format!("tmux set-option -t {session} @s0ma_user {user} >/dev/null; "));
    format!(
        "command -v tmux >/dev/null 2>&1 || {{ echo 'tmux is not installed on the machine' >&2; exit 127; }}; \
         {install}\
         if ! tmux has-session -t {session} 2>/dev/null; then \
             [ -d /workspace ] && d=/workspace || d=\"$HOME\"; \
             tmux new-session -d -s {session} -c \"$d\" -x 120 -y 40 {shell} && {{ {options}}} && echo created; \
         elif [ \"$(tmux show-options -v -t {session} @s0ma_netbox 2>/dev/null)\" != {VERSION} ]; then \
             case \"$(tmux display-message -p -t {pane} '#{{pane_current_command}}')\" in \
             bash|sh|dash|zsh|fish|sudo) \
                 dir=\"$(tmux display-message -p -t {pane} '#{{pane_current_path}}')\"; \
                 tmux respawn-pane -k -t {pane} -c \"${{dir:-/workspace}}\" {shell} && \
                 tmux clear-history -t {pane} && {version} && tmux set-option -t {session} @s0ma_user {user} >/dev/null ;; \
             esac; \
         fi",
        shell = crate::util::shell_quote(&shell),
        version = version.trim_end().trim_end_matches(';'),
    )
}

/// The guest script that brings an open netbox set up by an older s0ma to
/// these settings, when it sits idle at its shell: the settings file written
/// again (for s0ma's own account), and the pane restarted in its folder with
/// the command it was made with. A background job's netbox, a subagent's,
/// and a busy one are left as they are.
pub fn upgrade_script(name: &str) -> String {
    let session = format!("s0ma-{}", session_of(name));
    let pane = format!("{session}:0.0");
    let rc = STANDARD.encode(RC.as_bytes());
    format!(
        "if [ \"$(tmux show-options -v -t {session} @s0ma_netbox 2>/dev/null)\" != {VERSION} ]; then \
             case \"$(tmux show-options -v -t {session} @s0ma_by 2>/dev/null)|$(tmux show-options -v -t {session} @s0ma_account 2>/dev/null)\" in \
             job:*|*'|worker') ;; \
             *) case \"$(tmux display-message -p -t {pane} '#{{pane_current_command}}')\" in \
                bash|sh|dash|zsh|fish) \
                    printf '%s' '{rc}' | base64 -d > \"$HOME/.s0ma-netbox-rc\"; \
                    dir=\"$(tmux display-message -p -t {pane} '#{{pane_current_path}}')\"; \
                    tmux respawn-pane -k -t {pane} -c \"${{dir:-/workspace}}\" && tmux clear-history -t {pane} && \
                    tmux set-option -t {session} @s0ma_netbox {VERSION} >/dev/null && \
                    tmux set-option -t {session} @s0ma_user \"$(id -un)\" >/dev/null ;; \
                esac ;; \
             esac; \
         fi; true"
    )
}

/// The guest script that closes netbox `name`.
pub fn close_script(name: &str) -> String {
    let session = format!("s0ma-{}", session_of(name));
    format!("if tmux has-session -t {session} 2>/dev/null; then tmux kill-session -t {session}; echo closed; fi; true")
}

/// The guest script that sets (`Some`) or unsets a user option on a netbox.
pub fn option_script(name: &str, key: &str, value: Option<&str>) -> String {
    let session = format!("s0ma-{}", session_of(name));
    match value {
        Some(value) => format!("tmux has-session -t {session} 2>/dev/null || exit 44; tmux set-option -t {session} {key} {} >/dev/null", crate::util::shell_quote(value)),
        None => format!("tmux set-option -u -t {session} {key} >/dev/null 2>&1; true"),
    }
}

/// Why netbox `netbox` is too busy to close, if it is.
fn busy_reason(netbox: &Netbox) -> Option<String> {
    if !netbox.running.is_empty() {
        return Some("a command of s0ma's is running in it".into());
    }
    match netbox.command.as_str() {
        "" | "bash" | "sh" | "dash" | "zsh" | "fish" | "sudo" => None,
        program => Some(format!("{program} is running in it")),
    }
}

/// Open netbox `spec` unless it is open: whether it was made now. Refused
/// past `MAX` of the operator's and s0ma's.
pub fn open(run: Run, spec: &Spec) -> Result<bool, String> {
    valid_name(&spec.name)?;
    let counted = !spec.by.starts_with("worker:") && !spec.by.starts_with("job:");
    if spec.name != DEFAULT && counted {
        let open = list(run)?;
        let exists = open.iter().any(|netbox| netbox.name == spec.name && netbox.open);
        let count = open.iter().filter(|netbox| netbox.open && netbox.counted()).count();
        if !exists && count >= MAX {
            return Err(format!("{MAX} shells are open already; close one first"));
        }
    }
    let ran = run(&open_script(spec), 60)?;
    if ran.exit_code != Some(0) {
        return Err(format!("the shell could not be opened: {}", first_line(&format!("{}{}", ran.stderr, ran.stdout))));
    }
    let created = ran.stdout.contains("created");
    if created {
        changed();
    }
    Ok(created)
}

/// Close netbox `name`: refused while something runs in it unless `force`.
pub fn close(run: Run, name: &str, force: bool) -> Result<(), String> {
    let netboxes = list(run)?;
    let Some(netbox) = netboxes.iter().find(|netbox| netbox.name == name && netbox.open) else {
        return Err(format!("no shell {name} is open"));
    };
    if !force && let Some(reason) = busy_reason(netbox) {
        return Err(format!("{reason}; /shell close {name} --force closes it anyway"));
    }
    run(&close_script(name), 30)?;
    changed();
    Ok(())
}

// ------------------------------------------------------------------ The operator's commands

/// `/netbox …` and `/pair …`, the operator's: a line for him, or what went wrong.
pub fn handle_operator_command(command: &str) -> Result<String, String> {
    handle_with(command, &mut machine)
}

fn handle_with(command: &str, run: Run) -> Result<String, String> {
    let mut words: Vec<&str> = command.split_whitespace().collect();
    // `/netbox`, as it was called before the rename.
    if words.first() == Some(&"/netbox") {
        words[0] = "/shell";
    }
    match words.as_slice() {
        ["/shell"] | ["/shell", "list"] => Ok(describe_for_operator(&list(run)?)),
        ["/shell", "new", rest @ ..] => {
            let (name, purpose) = match rest {
                [] => (free_name(&list(run)?), String::new()),
                [name, purpose @ ..] => (name.to_string(), purpose.join(" ")),
            };
            valid_name(&name)?;
            let created = open(run, &Spec { name: name.clone(), purpose, by: "operator".into(), ..Spec::default() })?;
            crate::vm_terms::touch();
            Ok(if created { format!("[shell] {} opened", display(&name)) } else { format!("[shell] {} is open already", display(&name)) })
        }
        ["/shell", "close", name] | ["/shell", "close", name, "--force"] => {
            let name = canonical(name);
            close(run, &name, words.last() == Some(&"--force"))?;
            crate::vm_terms::touch();
            Ok(if name == DEFAULT {
                "[shell] main closed; a fresh one opens when it is next used".to_string()
            } else {
                format!("[shell] {} closed", display(&name))
            })
        }
        ["/shell", "mode", name, mode @ ("auto" | "confirm")] => {
            let name = canonical(name);
            let name = name.as_str();
            valid_name(name)?;
            let ran = run(&option_script(name, "@s0ma_mode", Some(mode)), 30)?;
            if ran.exit_code == Some(44) {
                return Err(format!("no shell {name} is open"));
            }
            changed();
            crate::vm_terms::touch();
            Ok(format!("[shell] {}: {mode}", display(name)))
        }
        ["/pair"] => {
            let on = list(run)?.first().is_some_and(|netbox| netbox.pair);
            Ok(format!("[pair] {}", if on { "on" } else { "off" }))
        }
        ["/pair", on @ ("on" | "off")] => {
            let script = if *on == "on" { "tmux set-option -g @s0ma_pair 1 2>/dev/null || { tmux start-server \\; set-option -g @s0ma_pair 1; }" } else { "tmux set-option -gu @s0ma_pair 2>/dev/null; true" };
            run(script, 30)?;
            changed();
            crate::vm_terms::touch();
            Ok(format!("[pair] {on}"))
        }
        _ => Err("usage: /shell [list] · /shell new [name] [purpose] · /shell close <name> [--force] · /shell mode <name> auto|confirm · /pair [on|off]".into()),
    }
}

/// A name for a netbox the operator opens without naming it: `shell-2`, `shell-3`…
pub fn free_name(open: &[Netbox]) -> String {
    (2..)
        .map(|n| format!("shell-{n}"))
        .find(|name| !open.iter().any(|netbox| &netbox.name == name && netbox.open))
        .unwrap_or_else(|| "shell-x".to_string())
}

/// Ctrl+Q in a shell's tab: the shell `session` closed, away from the keys,
/// whatever runs in it (the tab asked first when something did).
pub fn close_for_operator(session: &str) -> Result<String, String> {
    let name = name_of(session).ok_or("not a shell")?.to_string();
    let spawned = std::thread::Builder::new().name("netbox-close".into()).spawn(move || {
        if let Err(error) = close(&mut machine, &name, true) {
            crate::vm_terms::set_error(format!("shell: {error}"));
        }
        if working() == name {
            set_working(DEFAULT);
        }
        crate::vm_terms::touch();
        crate::tabs::poke();
    });
    spawned.map(|_| "closing".to_string()).map_err(|_| "could not start closing it".to_string())
}

/// Alt+N in a shell's tab: a netbox of the operator's, opened away from the keys;
/// the shell session it will be.
pub fn open_for_operator() -> String {
    let name = free_name(&crate::vm_terms::netboxes());
    let session = session_of(&name);
    let spec = Spec { name, by: "operator".into(), ..Spec::default() };
    let spawned = std::thread::Builder::new().name("netbox-open".into()).spawn(move || {
        let opened = open(&mut machine, &spec);
        if let Err(error) = opened {
            crate::vm_terms::set_error(format!("shell: {error}"));
        }
        crate::vm_terms::touch();
        crate::tabs::poke();
    });
    if spawned.is_err() {
        crate::vm_terms::set_error("shell: could not start opening it".into());
    }
    session
}

fn describe_for_operator(netboxes: &[Netbox]) -> String {
    let mut out = String::from("[shell]");
    for netbox in netboxes.iter().filter(|netbox| netbox.open || netbox.name == DEFAULT) {
        out.push_str(&format!("\n  {}", line_for(netbox, true)));
    }
    if netboxes.first().is_some_and(|netbox| netbox.pair) {
        out.push_str("\n  /pair on");
    }
    out
}

/// One netbox, in a line, as s0ma is told of it or (`for_operator`) as the operator is.
fn line_for(netbox: &Netbox, for_operator: bool) -> String {
    // Who is who, from the reader's side.
    let (me, other) = if for_operator { ("you", "s0ma") } else { ("you", "the operator") };
    let s0ma = if for_operator { "s0ma" } else { "you" };
    let operator = if for_operator { "you" } else { "the operator" };
    let mut line = display(&netbox.name);
    let lane = is_lane(&netbox.name) && netbox.opened_by == "s0ma";
    if !netbox.purpose.is_empty() && !lane {
        line.push_str(&format!(" — {}", netbox.purpose));
    }
    let by = match netbox.opened_by.as_str() {
        "" => format!("shared by {me} and {other}"),
        "operator" => format!("opened by {operator}"),
        "s0ma" if lane => format!("{} lane", if for_operator { "s0ma's" } else { "your" }),
        "s0ma" => format!("opened by {s0ma}"),
        by if by.starts_with("worker:") => format!("subagent {}'s", &by[7..]),
        by if by.starts_with("job:") => format!("background job {}", &by[4..]),
        by => by.to_string(),
    };
    line.push_str(&format!(" ({by}"));
    if netbox.open && !netbox.user.is_empty() {
        line.push_str(&format!("; as {}", netbox.user));
    }
    if netbox.confirm {
        line.push_str(if for_operator { "; s0ma's commands wait for your Enter" } else { "; your commands wait for the operator's Enter" });
    }
    if netbox.wheel {
        line.push_str(if for_operator { "; you have control" } else { "; the operator has control" });
    }
    if !netbox.open {
        line.push_str("; opens when used");
    }
    line.push(')');
    line
}

// ------------------------------------------------------------------ s0ma's working netbox

static WORKING: Mutex<Option<String>> = Mutex::new(None);

/// The netbox s0ma last worked in: its commands go there unless it names one.
pub fn working() -> String {
    WORKING.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone().unwrap_or_else(|| DEFAULT.to_string())
}

pub fn set_working(name: &str) {
    *WORKING.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(name.to_string());
}

// ------------------------------------------------------------------ s0ma's tools

/// What s0ma is told about the netboxes, in its system context.
pub const CONTEXT: &str = "\n\nSHELLS\n\
Every shell on your machine is one you share with the operator: a bash terminal, in /workspace at first, that he \
watches live and can type into himself (Alt+Shift and its number on his side). There can be several, each with a \
name and a purpose; the first is called main. Your vm_exec commands run in a shell, where the operator sees them: in \
the one you name (shell=...), or else the one you last worked in. Give a short why: it is shown above the command. A \
command runs in a subshell of the shell (its folder and exported variables, nothing it changes stays); persist=true \
runs it in the shell itself, for a cd, a venv or an export you both want to keep. If the operator is typing there, \
has unfinished input there, has taken control of it (Ctrl+\\), or something is running, your command goes to your own \
lane beside it (main-s0ma) instead: you never type over him. A shell the operator set to confirm (or /pair on) shows \
him your command first: he runs it with Enter, may edit it (the result says so), or declines it. A command that waits \
for keyboard input returns early with waiting_for_input: the operator can answer it, or you can with shell_type; its \
result arrives later as an event, as does one still waiting for the operator's Enter after ten minutes. Use \
shell_type, shell_key and shell_read for interactive programs and long-running foreground processes; shell_read says \
what the operator typed since you last looked. Open a shell of your own with shell_open for a separate piece of \
work, and close it when done. Subagents each work in a shell of their own. The shells and their purposes are in the \
turn header when they change.";

/// The `shell` argument's schema.
fn shell_parameter() -> Value {
    json!({"type": "string", "description": "Which shell (default: main, the first)"})
}

pub fn tool_definitions() -> Vec<crate::llm::ToolDefinition> {
    let shell = shell_parameter();
    vec![
        crate::llm::ToolDefinition::function(
            "shell_open",
            "Open a shell of your own: a terminal the operator sees, for a separate piece of work (a server, a build, a \
             long investigation). Up to 8 shells are open at once. Your vm_exec commands go to it once you name it \
             (shell=...) or work in it.",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]{0,23}$", "description": "Its name, short: lowercase letters, digits, - and _"},
                    "purpose": {"type": "string", "description": "What it is for, in a few words the operator sees"}
                },
                "required": ["name", "purpose"],
                "additionalProperties": false
            }),
        ),
        crate::llm::ToolDefinition::function(
            "shell_close",
            "Close a shell you opened, or your lane, when you are done with it. Refused while something runs in it. \
             main and the operator's shells are his to close.",
            json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": false
            }),
        ),
        crate::llm::ToolDefinition::function(
            "shell_type",
            "Type into a shell, a terminal you share with the operator: he sees it as you type. Opens main if it is not \
             open. Enter is sent after the text unless enter=false. Read it first if he may be using it; refused while \
             he has taken control of it. For a command whose result you want, use vm_exec instead. Don't type sleep to stay \
             idle or let time pass: when there is nothing to do, end your turn or use attention_wait.",
            json!({
                "type": "object",
                "properties": {
                    "shell": shell,
                    "text": {"type": "string", "description": "Literal text to type"},
                    "enter": {"type": "boolean", "description": "Send Enter after the text. Defaults to true"}
                },
                "required": ["text"],
                "additionalProperties": false
            }),
        ),
        crate::llm::ToolDefinition::function(
            "shell_key",
            "Send a key to a shell, a terminal you share with the operator. Refused while he has taken control of it.",
            json!({
                "type": "object",
                "properties": {
                    "shell": shell,
                    "key": {
                        "type": "string",
                        "enum": ["ctrl_c", "ctrl_d", "ctrl_z", "ctrl_l", "enter", "tab", "escape", "up", "down", "left", "right"],
                        "description": "Terminal key to send"
                    }
                },
                "required": ["key"],
                "additionalProperties": false
            }),
        ),
        crate::llm::ToolDefinition::function(
            "shell_read",
            "Look at a shell, a terminal you share with the operator: what it shows, what is running in it, where it is, \
             and what the operator typed into it since you last looked. Opens main if it is not open.",
            json!({
                "type": "object",
                "properties": {
                    "shell": shell,
                    "lines": {
                        "type": "integer", "minimum": 1, "maximum": 2000,
                        "description": "Lines of history to include. Defaults to 80"
                    }
                },
                "additionalProperties": false
            }),
        ),
    ]
}

/// A subagent's shell tools: shell_type, shell_key and shell_read, on its own
/// shell, which it does not name.
pub fn worker_tool_definitions() -> Vec<crate::llm::ToolDefinition> {
    let mut tools: Vec<_> = tool_definitions().into_iter().filter(|tool| matches!(tool.function.name.as_str(), "shell_type" | "shell_key" | "shell_read")).collect();
    for tool in &mut tools {
        if let Some(properties) = tool.function.parameters["properties"].as_object_mut() {
            properties.remove("shell");
        }
        tool.function.description = match tool.function.name.as_str() {
            "shell_type" => "Type into your shell, the terminal your vm_exec commands run in, which the operator can watch. \
                Enter is sent after the text unless enter=false. For an interactive program or a long-running foreground \
                process; for a command whose result you want, use vm_exec. Refused while the operator has taken control of it.",
            "shell_key" => "Send a key to your shell (ctrl_c stops what runs in the foreground). Refused while the operator has taken control of it.",
            _ => "Look at your shell: what it shows, what is running in it, where it is, and what the operator typed into it since you last looked.",
        }
        .to_string();
    }
    tools
}

/// shell_open and shell_close, s0ma's.
pub fn execute(call: &crate::types::ToolCall) -> String {
    execute_with(call, &mut machine)
}

fn execute_with(call: &crate::types::ToolCall, run: Run) -> String {
    let args: Value = serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
    let name = canonical(args["name"].as_str().unwrap_or(""));
    let result = match call.function.name.as_str() {
        "shell_open" | "netbox_open" => {
            let purpose = args["purpose"].as_str().unwrap_or("").trim().chars().take(120).collect::<String>();
            valid_name(&name).and_then(|()| open(run, &Spec { name: name.clone(), purpose, by: "s0ma".into(), ..Spec::default() })).map(|created| {
                set_working(&name);
                crate::vm_terms::touch();
                json!({"ok": true, "shell": name, "created": created, "note": format!(
                    "{} is open and is now where your vm_exec commands run; the operator sees it in his tab bar.", display(&name))})
            })
        }
        "shell_close" | "netbox_close" => (|| {
            valid_name(&name)?;
            let netboxes = list(run)?;
            let netbox = netboxes.iter().find(|netbox| netbox.name == name && netbox.open).ok_or_else(|| format!("no shell {name} is open"))?;
            if netbox.opened_by != "s0ma" {
                return Err(format!("{} is the operator's to close (it is {}); ask him if it should go", display(&name), line_for(netbox, false)));
            }
            close(run, &name, false)?;
            if working() == name {
                set_working(DEFAULT);
            }
            crate::vm_terms::touch();
            Ok(json!({"ok": true, "shell": name, "closed": true}))
        })(),
        other => Err(format!("unknown shell tool: {other}")),
    };
    result.unwrap_or_else(|error| json!({"ok": false, "error": error})).to_string()
}

// ------------------------------------------------------------------ what s0ma hears

static CHANGED: AtomicBool = AtomicBool::new(true);
static REPORTED: Mutex<Option<String>> = Mutex::new(None);

/// The list changed (opened, closed, a mode): read again for the next turn.
pub fn changed() {
    CHANGED.store(true, Ordering::SeqCst);
}

/// The netboxes, for the turn header, when they differ from what s0ma was
/// last told (subagents' left out: they come and go with their work).
pub fn take_note() -> Option<String> {
    if !CHANGED.swap(false, Ordering::SeqCst) {
        return None;
    }
    let netboxes = match list(&mut machine) {
        Ok(netboxes) => netboxes,
        Err(_) => {
            CHANGED.store(true, Ordering::SeqCst);
            return None;
        }
    };
    note_for(&netboxes)
}

fn note_for(netboxes: &[Netbox]) -> Option<String> {
    let shown: Vec<&Netbox> = netboxes.iter().filter(|netbox| !netbox.is_workers() && (netbox.open || netbox.name == DEFAULT)).collect();
    let mut text = String::from("\n\n[SHELLS]");
    for netbox in &shown {
        text.push_str(&format!("\n- {}", line_for(netbox, false)));
    }
    if netboxes.first().is_some_and(|netbox| netbox.pair) {
        text.push_str("\n- /pair is on: your commands wait for the operator's Enter in every shell");
    }
    let approval = crate::approval::enabled();
    if approval {
        text.push_str(
            "\n- approval mode is on: each command you or a subagent run (vm_exec, shell_type, shell_key, goal tests) and each \
             subagent you start waits for the operator to approve it; a declined one is not run, and its result says so",
        );
    }
    let mut reported = REPORTED.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // Only the first netbox, as ever: nothing to say until there is more.
    let first_only = shown.len() == 1 && shown[0].name == DEFAULT && !shown[0].confirm && !shown[0].pair && !approval;
    if reported.as_deref() == Some(text.as_str()) || (reported.is_none() && first_only) {
        *reported = Some(text);
        return None;
    }
    *reported = Some(text.clone());
    Some(text)
}

static EVENTS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// A result that came after its call returned (a command that waited for
/// input or for the operator's Enter): s0ma's next turn, or the one after its current.
pub fn push_event(text: String) {
    EVENTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(text);
    crate::subagent::wake_prompt();
}

/// The waiting results as a turn of their own, taken once.
pub fn take_turn() -> Option<String> {
    let events = std::mem::take(&mut *EVENTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
    (!events.is_empty()).then(|| {
        format!("[SHELLS — COMMANDS THAT FINISHED AFTER THEIR CALL RETURNED, RENDERED BY THE HARNESS]\n{}", events.join("\n\n"))
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A private tmux server and home on this machine standing in for s0ma's.
    pub(crate) struct Machine {
        pub root: std::path::PathBuf,
        pub home: std::path::PathBuf,
    }

    impl Machine {
        pub fn new(tag: &str) -> Option<Machine> {
            std::process::Command::new("tmux").arg("-V").output().ok()?;
            let root = std::env::temp_dir().join(format!("s0ma-nb-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            let home = root.join("home");
            std::fs::create_dir_all(&home).ok()?;
            Some(Machine { root, home })
        }

        pub fn run(&self, command: &str, _timeout: u64) -> Result<Ran, String> {
            let output = std::process::Command::new("bash")
                .arg("-c")
                .arg(command)
                .env("TMUX_TMPDIR", &self.root)
                .env("HOME", &self.home)
                .env_remove("TMUX")
                .stdin(std::process::Stdio::null())
                .output()
                .map_err(|error| error.to_string())?;
            Ok(Ran {
                exit_code: output.status.code().map(i64::from),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            })
        }

        pub fn runner(&self) -> impl FnMut(&str, u64) -> Result<Ran, String> + '_ {
            move |command: &str, timeout: u64| self.run(command, timeout)
        }
    }

    impl Drop for Machine {
        fn drop(&mut self) {
            let _ = self.run("tmux kill-server 2>/dev/null; true", 5);
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn names_sessions_and_how_operator_sees_them() {
        // The first is main, in the session it has always had.
        assert_eq!(session_of("main"), "netbox");
        assert_eq!(session_of("build"), "nb-build");
        assert_eq!(name_of("nb-build"), Some("build"));
        assert_eq!(name_of("netbox"), Some("main"));
        assert_eq!(name_of("main"), None, "a private shell from before is not one");
        assert_eq!((canonical("netbox"), canonical(" build ")), ("main".to_string(), "build".to_string()));
        assert_eq!(display("main-s0ma"), "main·s0ma");
        assert_eq!(name_of("nb-"), None);
        assert_eq!(display("w-b3-1"), "w·b3-1");
        assert_eq!(display("netbox-s0ma"), "netbox·s0ma");
        assert_eq!(display("build"), "build");
        assert_eq!(lane_of("netbox"), "netbox-s0ma");
        assert!(lane_of("a-very-long-netbox-name-x").len() <= NAME_MAX);
        assert!(is_lane("netbox-s0ma") && !is_lane("netbox"));
        assert_eq!(worker_name("subagent-1700000000000-42-3-1"), "w-b3-1");
        assert_eq!(job_name("bg-1a2b3c4d"), "job-1a2b3c4d");
        assert!(valid_name("build_2").is_ok());
        for bad in ["", "Build", "-x", "a b", "x;y", &"a".repeat(25)] {
            assert!(valid_name(bad).is_err(), "{bad}");
        }
        assert!(crate::vm::tmux_session_name(&session_of(&"a".repeat(NAME_MAX))).is_ok(), "every netbox is a usable session");
    }

    #[test]
    fn the_list_reads_each_netboxs_settings_and_puts_the_first_first() {
        let purpose = STANDARD.encode("watch the build");
        let out = format!(
            "s0ma-main||||||||bash|\n\
             s0ma-nb-w-b3-1|3||worker:subagent-1-2-3-1|auto|worker|||bash|1\n\
             s0ma-nb-build|3|{purpose}|operator|confirm|s0ma|ab12cd34|1|make|1\n\
             s0ma-netbox|3|||auto|s0ma|||bash|1\n\
             other|||||||||\n"
        );
        let (sessions, netboxes) = parse(&out);
        assert_eq!(sessions, ["main", "nb-w-b3-1", "nb-build", "netbox"]);
        let names: Vec<&str> = netboxes.iter().map(|netbox| netbox.name.as_str()).collect();
        assert_eq!(names, ["main", "build", "w-b3-1"]);
        // The account: by name once kept, else from the kind of shell.
        let (_, with_user) = parse("s0ma-nb-w-b3-1|6||worker:r|auto|worker|||bash||s0ma-worker\ns0ma-netbox|6|||auto|s0ma|||bash||s0ma\n");
        assert_eq!(with_user.iter().map(Netbox::account).collect::<Vec<_>>(), ["s0ma", "s0ma-worker"]);
        assert_eq!(netboxes[0].account(), "s0ma's account", "kept before the account was");
        assert!(line_for(&with_user[1], true).contains("; as s0ma-worker"), "{}", line_for(&with_user[1], true));
        let build = &netboxes[1];
        assert_eq!(build.purpose, "watch the build");
        assert_eq!(build.opened_by, "operator");
        assert!(build.confirm && build.wheel && build.pair && build.open);
        assert_eq!(build.running, "ab12cd34");
        assert_eq!(build.command, "make");
        assert!(netboxes[2].worker && netboxes[2].is_workers() && !netboxes[2].counted());
        assert!(!netboxes[2].asks_first(), "pairing never holds up a subagent");
        assert!(netboxes[0].asks_first(), "pairing holds s0ma's commands in the first netbox");

        // Not open yet, the first netbox is listed all the same.
        let (_, only) = parse("s0ma-main||||||||bash|\n");
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].name, DEFAULT);
        assert!(!only[0].open);
    }

    #[test]
    fn a_subagents_commands_run_as_its_account() {
        let wrapped = as_user("s0ma-worker", "cd /workspace; echo 'hi'");
        assert!(wrapped.starts_with("printf %s '") && wrapped.contains("| sudo -n -u s0ma-worker -H -- bash -c"), "{wrapped}");
        let encoded = wrapped.split('\'').nth(1).unwrap();
        assert_eq!(String::from_utf8(STANDARD.decode(encoded).unwrap()).unwrap(), "cd /workspace; echo 'hi'");
        let script = provision_script("s0ma-worker");
        assert!(script.contains("useradd --create-home") && script.contains("share /workspace rwX rwx") && script.contains("share /share rX r-x"));
        assert!(script.contains("chmod o-rwx \"$HOME\""), "s0ma's home is closed to the account");
        let open = open_script(&Spec { name: "w-b3-1".into(), by: "worker:x".into(), user: Some("s0ma-worker".into()), ..Spec::default() });
        assert!(open.contains("sudo -n -u s0ma-worker -H -- env S0MA_NETBOX=") && open.contains("--rcfile ~s0ma-worker/.s0ma-netbox-rc"), "{open}");
        assert!(open.contains("@s0ma_account 'worker'"), "{open}");
    }

    /// A netbox opened, listed with its settings, set to confirm, refused
    /// past the limit, closed (refused while busy), all on a real tmux.
    #[test]
    fn netboxes_open_list_and_close_on_a_real_tmux() {
        let _lock = crate::vm_terms::TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(machine) = Machine::new("open") else { return };
        let mut run = machine.runner();
        let spec = Spec { name: "build".into(), purpose: "the release build".into(), by: "operator".into(), ..Spec::default() };
        assert_eq!(open(&mut run, &spec), Ok(true));
        assert_eq!(open(&mut run, &spec), Ok(false), "open already");
        assert!(machine.home.join(".s0ma-netbox-rc").exists());
        assert!(machine.home.join(".s0ma-netbox/cmd").is_dir());
        std::thread::sleep(std::time::Duration::from_millis(600));
        let seen = machine.run("tmux capture-pane -p -t s0ma-nb-build:0.0", 5).unwrap().stdout;
        let me = machine.run("id -un", 5).unwrap().stdout.trim().to_string();
        assert!(seen.contains(&format!("{me}@build")) && seen.trim_end().ends_with('$'), "its own prompt, with the account: {seen}");

        let listed = list(&mut run).unwrap();
        let build = listed.iter().find(|netbox| netbox.name == "build").unwrap();
        assert_eq!((build.purpose.as_str(), build.opened_by.as_str(), build.confirm), ("the release build", "operator", false));
        assert_eq!(build.account(), me, "the account kept by name");
        assert_eq!(listed[0].name, DEFAULT, "first, though not open");

        assert_eq!(handle_with("/shell mode build confirm", &mut run).unwrap(), "[shell] build: confirm");
        assert_eq!(handle_with("/netbox mode build confirm", &mut run).unwrap(), "[shell] build: confirm", "as it was called before");
        assert!(list(&mut run).unwrap().iter().any(|netbox| netbox.name == "build" && netbox.confirm));
        assert!(handle_with("/netbox mode nothing confirm", &mut run).is_err());
        assert_eq!(handle_with("/pair on", &mut run).unwrap(), "[pair] on");
        assert!(list(&mut run).unwrap().iter().all(|netbox| netbox.pair));
        handle_with("/pair off", &mut run).unwrap();
        assert!(list(&mut run).unwrap().iter().all(|netbox| !netbox.pair));

        // Seven more is too many.
        for n in 2..=MAX {
            open(&mut run, &Spec { name: format!("b{n}"), by: "s0ma".into(), ..Spec::default() }).unwrap();
        }
        let refused = open(&mut run, &Spec { name: "one-more".into(), by: "s0ma".into(), ..Spec::default() });
        assert!(refused.unwrap_err().contains("close one first"));
        // A subagent's does not count.
        assert_eq!(open(&mut run, &Spec { name: "w-b1-1".into(), by: "worker:x".into(), ..Spec::default() }), Ok(true));

        // Busy: refused, unless forced.
        machine.run("tmux send-keys -t s0ma-nb-build:0.0 'sleep 30' Enter; sleep 0.5", 5).unwrap();
        let busy = close(&mut run, "build", false).unwrap_err();
        assert!(busy.contains("sleep is running") && busy.contains("--force"), "{busy}");
        assert!(handle_with("/netbox close build --force", &mut run).unwrap().contains("closed"));
        assert!(!list(&mut run).unwrap().iter().any(|netbox| netbox.name == "build"));
        assert!(handle_with("/netbox close netbox", &mut run).is_err(), "the first stays");

        // s0ma closes its own, not the operator's.
        open(&mut run, &Spec { name: "theirs".into(), by: "operator".into(), ..Spec::default() }).unwrap();
        let call = |name: &str, arguments: Value| crate::types::ToolCall {
            id: "1".into(),
            kind: "function".into(),
            function: crate::types::FunctionCall { name: name.into(), arguments: arguments.to_string() },
        };
        let theirs: Value = serde_json::from_str(&execute_with(&call("netbox_close", json!({"name": "theirs"})), &mut run)).unwrap();
        assert_eq!(theirs["ok"], json!(false), "The operator's is his to close: {theirs}");
        let mine: Value = serde_json::from_str(&execute_with(&call("netbox_close", json!({"name": "b2"})), &mut run)).unwrap();
        assert_eq!(mine["ok"], json!(true), "{mine}");
        let opened: Value = serde_json::from_str(&execute_with(&call("netbox_open", json!({"name": "probe", "purpose": "x"})), &mut run)).unwrap();
        assert_eq!(opened["ok"], json!(true), "{opened}");
        assert_eq!(working(), "probe", "where its commands go now");
        set_working(DEFAULT);
    }

    /// A netbox an older s0ma set up is restarted into these settings when
    /// it is looked at, idle; a busy one, or a job's, is left alone.
    #[test]
    fn an_older_netbox_is_brought_up_to_date_when_idle() {
        let Some(machine) = Machine::new("upgrade") else { return };
        let mut run = machine.runner();
        open(&mut run, &Spec { name: "old".into(), by: "operator".into(), ..Spec::default() }).unwrap();
        // As an older s0ma left it: its own prompt, an older version.
        machine.run("printf 'PS1=\"OLD> \"\n' > ~/.s0ma-netbox-rc; tmux set-option -t s0ma-nb-old @s0ma_netbox 3; \
                     tmux respawn-pane -k -t s0ma-nb-old:0.0; sleep 0.5", 5).unwrap();
        let screen = |machine: &Machine| machine.run("tmux capture-pane -p -t s0ma-nb-old:0.0", 5).unwrap().stdout;
        assert!(screen(&machine).contains("OLD>"), "{}", screen(&machine));
        machine.run("tmux send-keys -t s0ma-nb-old:0.0 'sleep 30' Enter; sleep 0.3", 5).unwrap();
        machine.run(&upgrade_script("old"), 5).unwrap();
        let version = |machine: &Machine| machine.run("tmux show-options -v -t s0ma-nb-old @s0ma_netbox", 5).unwrap().stdout.trim().to_string();
        assert_eq!(version(&machine), "3", "busy: left alone");
        machine.run("tmux send-keys -t s0ma-nb-old:0.0 C-c; sleep 0.3", 5).unwrap();
        machine.run(&upgrade_script("old"), 5).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(600));
        assert_eq!(version(&machine), VERSION.to_string());
        let now = screen(&machine);
        assert!(!now.contains("OLD>") && now.trim_end().ends_with('$'), "the new prompt, a clean screen: {now}");
    }

    #[test]
    fn operator_opens_netboxes_by_free_names_and_the_header_says_when_they_change() {
        let open = |name: &str| Netbox { name: name.into(), open: true, ..Netbox::default() };
        assert_eq!(free_name(&[open("main")]), "shell-2");
        assert_eq!(free_name(&[open("main"), open("shell-2")]), "shell-3");

        *REPORTED.lock().unwrap() = None;
        assert_eq!(note_for(&[open("main")]), None, "the first netbox alone says nothing new");
        let mut build = open("build");
        build.purpose = "the build".into();
        build.opened_by = "operator".into();
        let note = note_for(&[open("main"), build.clone()]).unwrap();
        assert!(note.contains("[SHELLS]") && note.contains("build — the build (opened by the operator)"), "{note}");
        let lane = Netbox { opened_by: "s0ma".into(), purpose: "s0ma's own".into(), ..open("main-s0ma") };
        assert_eq!(line_for(&lane, false), "main·s0ma (your lane)");
        assert_eq!(line_for(&lane, true), "main·s0ma (s0ma's lane)");
        assert_eq!(line_for(&open("main"), true), "main (shared by you and s0ma)");
        assert_eq!(line_for(&open("main"), false), "main (shared by you and the operator)");
        assert_eq!(note_for(&[open("main"), build.clone()]), None, "said once");
        let worker = Netbox { opened_by: "worker:run".into(), ..open("w-b1-1") };
        assert_eq!(note_for(&[open("main"), build.clone(), worker]), None, "subagents' are left out");
        assert!(note_for(&[open("main")]).is_some(), "and closing one is news");
    }

    #[test]
    fn late_results_make_a_turn_of_their_own() {
        let _ = take_turn();
        push_event("- main: `make` finished".into());
        let turn = take_turn().unwrap();
        assert!(turn.starts_with("[SHELLS — ") && turn.contains("`make` finished"), "{turn}");
        assert_eq!(take_turn(), None, "taken once");
    }
}
