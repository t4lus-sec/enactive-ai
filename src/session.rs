//! One the agent, many terminals.
//!
//! The problem this solves is narrow and specific: an SSH session drops while
//! the agent is halfway through a turn. The process keeps running — it is not
//! attached to the tty for anything but display — but the operator has no way
//! back in, and starting `s0ma` again would create a *second* instance writing
//! to the same archive, the same self-model and the same sqlite stores. Two
//! Agents sharing one memory is worse than no agent.
//!
//! So the socket is the lock. A second process cannot become a second agent; it
//! becomes a terminal onto the first one.
//!
//! - The primary binds `data/s0ma.sock`. Binding is the single-instance claim.
//! - Anything started afterwards connects instead and becomes a terminal of
//!   that the agent, which looks and works the way the primary's own does: the
//!   primary sends it what its display records (`terminal::Mirrored`) and
//!   what its subagents do (`crate::subagent::feed::Change`), from the conversation so
//!   far on, and the attached terminal draws them with the same display,
//!   collapsed or expanded with its own Ctrl+O, at its own width, with the
//!   same prompt, steering pane, tabs and keys. What is typed there goes to
//!   the primary, which handles it as it handles its own keyboard's.
//! - Ctrl+D on an attached terminal detaches it and leaves the agent running.
//!
//! The protocol is JSON lines, after a greeting line (`HELLO`). A primary
//! from before this greeting speaks the older, simpler stream, which an
//! attached terminal still shows (`attach_legacy`).
//!
//! # From another machine
//!
//! `s0ma connect <ssh target>` runs `s0ma --bridge` there over SSH: the
//! bridge joins that the agent's socket and passes the protocol through SSH's
//! input and output, so a terminal on the operator's own machine attaches
//! as one on the server would, through whatever carries SSH (the LAN, a
//! VPN, a tailnet). Nothing listens on the network that SSH doesn't. Over
//! such a connection the terminal also moves files: `/download <file in
//! the share>` saves it on this machine, `/upload <path>` puts a file of
//! this machine into the share (`transfer`). The tabs that read the
//! server's own files say they aren't there yet (`is_remote`).

use crate::input::{ExternalSubmitter, InputEvent, InputReader, Steered};
use crate::markdown::MarkdownStream;
use crate::terminal::{
    ActiveDisplay, DisplaySnapshot, Mirrored, NAME_STYLE, OPERATOR_STYLE, RESET, STATUS_STYLE, THINKING_STYLE,
    safe_text,
};
use crate::subagent::feed::{Change, FeedSnapshot};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// The primary's first line to a terminal that connects.
const HELLO: &str = "s0ma-attach 2";

/// How long an attached terminal waits for the primary to answer a request.
const ANSWER_WITHIN: Duration = Duration::from_secs(15);
/// How long an attached terminal waits for the greeting and the snapshot (a
/// primary whose own terminal is wedged may not manage them).
const GREETING_WITHIN: Duration = Duration::from_secs(20);
/// A terminal that takes longer than this to accept a line has stopped
/// reading (suspended, hung): it is dropped, so the others never wait on it
/// and nothing piles up for it.
const WRITE_WITHIN: Duration = Duration::from_secs(5);

/// The primary to an attached terminal.
#[derive(Serialize, Deserialize)]
enum ToTerminal {
    /// First: everything so far.
    Snapshot {
        display: DisplaySnapshot,
        feeds: Vec<FeedSnapshot>,
        /// Whose shells are whose; absent from an older primary.
        #[serde(default)]
        execs: crate::exec_feed::Snapshot,
        #[serde(default)]
        features: Vec<String>,
        /// What waits for the operator's approval (`approval.rs`).
        #[serde(default)]
        approvals: Vec<crate::approval::Pending>,
    },
    Display(Mirrored),
    /// The requests waiting for approval, whenever they change.
    Approvals(Vec<crate::approval::Pending>),
    Feed(Change),
    Exec(crate::exec_feed::Change),
    Answer { id: u64, answer: Answer },
}

/// What this primary can do beyond the base protocol, told in its snapshot
/// (an older terminal ignores the field, an older primary leaves it out).
const FEATURES: &[&str] = &["transfer"];

/// An attached terminal to the primary.
#[derive(Serialize, Deserialize)]
enum ToPrimary {
    /// Submitted at the idle prompt.
    Say { id: u64, text: String },
    /// Sent from the steering pane; `now`: Ctrl+S / Ctrl+Enter.
    Steer { id: u64, text: String, now: bool },
    /// Enter in a tab.
    Act { id: u64, action: crate::tabs::Action },
    /// Ctrl+C during a turn.
    Interrupt,
    /// Part of a file in the share, for `/download`.
    Fetch { id: u64, path: String, offset: u64 },
    /// Part of a file for the share, for `/upload`: written beside its
    /// name under `upload` (one per file), and put in place at `done`.
    Put { id: u64, upload: String, name: String, offset: u64, data: String, done: bool },
}

#[derive(Serialize, Deserialize)]
enum Answer {
    Steered(Steered),
    Acted(std::result::Result<String, String>),
    /// For the terminal that asked only.
    Note(String),
    /// Part of a file: base64 `data` from the offset asked, the file's size,
    /// and whether this is its end.
    Data(std::result::Result<crate::transfer::Part, String>),
}

// Legacy framing, for a primary older than `HELLO` (`attach_legacy`).
const TEXT_FRAME: char = 't';
const REASONING_FRAME: char = 'r';
const STATUS_FRAME: char = 's';
const TURN_END_FRAME: char = 'e';

fn unescape(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars();
    while let Some(character) = chars.next() {
        if character != '\\' {
            out.push(character);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

pub fn socket_path(data_dir: &Path) -> PathBuf {
    data_dir.join("s0ma.sock")
}

fn pid_path(data_dir: &Path) -> PathBuf {
    data_dir.join("s0ma.pid")
}

/// Whether a process with this id is still alive.
///
/// The socket alone cannot distinguish "the owner is dead" from "the owner is
/// alive but did not answer", and those need opposite responses: the first wants
/// the stale file removed, the second must never start a second agent against the
/// same archive. So the pid is the tie-breaker, and an unanswered socket with a
/// living owner refuses to start rather than guessing.
fn process_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

fn recorded_owner(data_dir: &Path) -> Option<u32> {
    std::fs::read_to_string(pid_path(data_dir)).ok()?.trim().parse().ok()
}

/// What is at the socket path.
enum Occupant {
    /// A live instance answered, and this is the connection it answered on.
    ///
    /// The stream is carried out rather than dropped because dropping it *is* an
    /// attach as far as the primary is concerned: it accepts the connection,
    /// greets it, and finds the peer already gone. Probing and then connecting
    /// again would make every attach cost one spurious client and one
    /// "Broken pipe" on the primary's terminal.
    Running(UnixStream),
    /// A socket file with nobody behind it — a previous run that was killed.
    Stale,
    /// The socket did not answer, but the process that made it is still alive.
    /// Refuse to do anything: removing its socket would orphan a running agent,
    /// and binding a new one would put two of them on the same archive.
    Unresponsive(u32),
    /// Nothing there.
    Absent,
}

fn probe(data_dir: &Path) -> Occupant {
    let path = socket_path(data_dir);
    if !path.exists() {
        return Occupant::Absent;
    }
    // The only reliable liveness test for a Unix socket is to connect to it. A
    // leftover file refuses the connection; a live listener accepts it.
    match UnixStream::connect(&path) {
        Ok(stream) => Occupant::Running(stream),
        Err(_) => match recorded_owner(data_dir) {
            Some(pid) if process_alive(pid) => Occupant::Unresponsive(pid),
            _ => Occupant::Stale,
        },
    }
}

// ------------------------------------------------------------------ primary

/// What the sending thread does next.
enum Outgoing {
    /// To every attached terminal.
    All(String),
    /// A terminal that just attached, with its snapshot.
    Join(u64, UnixStream, String),
    /// To one terminal.
    One(u64, String),
}

pub struct Server {
    clients: Arc<AtomicUsize>,
    path: PathBuf,
    pid_path: PathBuf,
}

fn line<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

impl Server {
    /// Claim the instance. `Ok(None)` means another agent already holds it.
    /// `interrupt`: what Ctrl+C sets, for an attached terminal's.
    pub fn bind(
        data_dir: &Path,
        submitter: ExternalSubmitter,
        display: ActiveDisplay,
        interrupt: Arc<AtomicBool>,
    ) -> Result<Option<Self>> {
        let path = socket_path(data_dir);

        match probe(data_dir) {
            Occupant::Running(_) => return Ok(None),
            Occupant::Unresponsive(pid) => {
                return Err(format!(
                    "process {pid} still holds this archive but its socket is not answering. \
                     Not starting a second s0ma. Check it with `ps -p {pid}`; if it is wedged, \
                     stop it, then start again."
                )
                .into());
            }
            Occupant::Stale => {
                crate::terminal::warn_status("[session] the previous run left a socket behind without exiting cleanly \
                     (killed, most likely by SIGHUP when its terminal closed); removing it");
                std::fs::remove_file(&path)?;
            }
            Occupant::Absent => {}
        }

        let listener = UnixListener::bind(&path)?;
        // The operator's alone: a terminal attached here can do whatever he
        // can, approve the agent's commands among it (`approval.rs`).
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::write(pid_path(data_dir), std::process::id().to_string())?;

        // Terminals attached or attaching: nothing is mirrored while there
        // are none, so a primary on its own pays nothing.
        let clients = Arc::new(AtomicUsize::new(0));
        let (outgoing, queue) = mpsc::channel::<Outgoing>();
        {
            let outgoing = Mutex::new(outgoing.clone());
            crate::terminal::mirror_to(
                Box::new(move |event| {
                    let _ = crate::util::locked(&outgoing).send(Outgoing::All(line(&ToTerminal::Display(event))));
                }),
                Arc::clone(&clients),
            );
        }
        {
            let outgoing = Mutex::new(outgoing.clone());
            crate::subagent::feed::mirror_to(
                Box::new(move |change| {
                    let _ = crate::util::locked(&outgoing).send(Outgoing::All(line(&ToTerminal::Feed(change))));
                }),
                Arc::clone(&clients),
            );
        }
        {
            let outgoing = Mutex::new(outgoing.clone());
            crate::approval::mirror_to(Box::new(move |pending| {
                let _ = crate::util::locked(&outgoing).send(Outgoing::All(line(&ToTerminal::Approvals(pending))));
            }));
        }
        {
            let outgoing = Mutex::new(outgoing.clone());
            crate::exec_feed::mirror_to(
                Box::new(move |change| {
                    let _ = crate::util::locked(&outgoing).send(Outgoing::All(line(&ToTerminal::Exec(change))));
                }),
                Arc::clone(&clients),
            );
        }

        // One thread writes to the terminals, in the order things happened;
        // a slow one never holds up the display.
        let counted = Arc::clone(&clients);
        thread::spawn(move || {
            let mut attached: Vec<(u64, UnixStream)> = Vec::new();
            for message in queue {
                let before = attached.len();
                match message {
                    Outgoing::All(text) => attached.retain_mut(|(_, stream)| send_line(stream, &text)),
                    Outgoing::Join(id, mut stream, snapshot) => {
                        let _ = stream.set_write_timeout(Some(WRITE_WITHIN));
                        if send_line(&mut stream, &snapshot) {
                            attached.push((id, stream));
                        } else {
                            counted.fetch_sub(1, Ordering::SeqCst);
                        }
                    }
                    Outgoing::One(id, text) => {
                        attached.retain_mut(|(client, stream)| *client != id || send_line(stream, &text))
                    }
                }
                let gone = before.saturating_sub(attached.len());
                if gone > 0 {
                    counted.fetch_sub(gone, Ordering::SeqCst);
                }
            }
        });

        let accepting = Arc::clone(&clients);
        thread::spawn(move || {
            let next = AtomicU64::new(1);
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let id = next.fetch_add(1, Ordering::SeqCst);
                // On a thread of its own: the snapshot waits for the display,
                // which waits for this terminal's own screen, and one stuck
                // there must not keep the next terminal from attaching.
                let (accepting, outgoing, display, submitter, interrupt) =
                    (Arc::clone(&accepting), outgoing.clone(), display.clone(), submitter.clone(), Arc::clone(&interrupt));
                thread::spawn(move || {
                    // A failed greeting means the peer is already gone, which
                    // is ordinary: anything testing whether this agent is alive
                    // connects and leaves. Nobody to tell, so it is dropped.
                    if let Err(error) = welcome(id, stream, &accepting, &outgoing, &display, &submitter, &interrupt)
                        && std::env::var_os("S0MA_SESSION_DEBUG").is_some()
                    {
                        crate::terminal::warn_status(&format!("[session] a client went away before attaching ({error})"));
                    }
                });
            }
        });

        crate::terminal::say_note(&format!("[session] listening on {}", path.display()));
        Ok(Some(Self { clients, path, pid_path: pid_path(data_dir) }))
    }

    /// How many terminals are attached besides the primary.
    pub fn attached(&self) -> usize {
        self.clients.load(Ordering::SeqCst)
    }
}

fn send_line(stream: &mut UnixStream, text: &str) -> bool {
    writeln!(stream, "{text}").and_then(|()| stream.flush()).is_ok()
}

/// A terminal that connected: the greeting, everything so far, then what
/// it sends, handled as the primary's own keyboard's would be.
fn welcome(
    id: u64,
    stream: UnixStream,
    clients: &Arc<AtomicUsize>,
    outgoing: &mpsc::Sender<Outgoing>,
    display: &ActiveDisplay,
    submitter: &ExternalSubmitter,
    interrupt: &Arc<AtomicBool>,
) -> std::io::Result<()> {
    let mut writer = stream.try_clone()?;
    writeln!(writer, "{HELLO}")?;
    writer.flush()?;

    // Taken with the display and the feeds held, and the terminal counted
    // before they are let go: everything after it is mirrored to it, and
    // nothing in it is mirrored again.
    display.snapshot(|display| {
        crate::subagent::feed::snapshot(|feeds| {
            crate::exec_feed::snapshot(|execs| {
                clients.fetch_add(1, Ordering::SeqCst);
                let features = FEATURES.iter().map(|f| f.to_string()).collect();
                let approvals = crate::approval::pending();
                let snapshot = line(&ToTerminal::Snapshot { display, feeds, execs, features, approvals });
                if outgoing.send(Outgoing::Join(id, writer, snapshot)).is_err() {
                    clients.fetch_sub(1, Ordering::SeqCst);
                }
            })
        })
    })?;

    let display = display.clone();
    let submitter = submitter.clone();
    let interrupt = Arc::clone(interrupt);
    let outgoing = outgoing.clone();
    thread::spawn(move || {
        for text in BufReader::new(stream).lines() {
            let Ok(text) = text else { break };
            let Ok(message) = serde_json::from_str::<ToPrimary>(&text) else { continue };
            let answer = |request: u64, answer: Answer| {
                let _ = outgoing.send(Outgoing::One(id, line(&ToTerminal::Answer { id: request, answer })));
            };
            match message {
                // Only during a turn, as this keyboard's Ctrl+C: set between
                // turns, it would stop the next one as it began.
                ToPrimary::Interrupt if display.working().is_some() => interrupt.store(true, Ordering::SeqCst),
                ToPrimary::Interrupt => {}
                ToPrimary::Steer { id: request, text, now } => {
                    let steered = submitter.steer(text, now).unwrap_or(Steered::Done);
                    answer(request, Answer::Steered(steered));
                }
                ToPrimary::Say { id: request, text } => answer(request, say(&display, &submitter, text)),
                ToPrimary::Act { id: request, action } => answer(request, Answer::Acted(crate::tabs::perform(action))),
                ToPrimary::Fetch { id: request, path, offset } => {
                    answer(request, Answer::Data(crate::transfer::read_part(&crate::share::receive_dir(), &path, offset)))
                }
                ToPrimary::Put { id: request, upload, name, offset, data, done } => answer(
                    request,
                    Answer::Acted(crate::transfer::write_part(&crate::share::receive_dir(), &upload, &name, offset, &data, done)),
                ),
            }
        }
    });
    Ok(())
}

/// A line submitted at an attached terminal's idle prompt: as if typed at
/// this one. `/mistake` is answered here, and its review window opens on the
/// terminal it was typed on.
fn say(display: &ActiveDisplay, submitter: &ExternalSubmitter, text: String) -> Answer {
    let command = crate::input::strip_send_now(&text).unwrap_or(&text).to_string();
    // The memory tab opens on the terminal it was asked for on; what it
    // writes comes back here (`tabs::Action`).
    if let Some(start) = crate::redact::tab_request(&command) {
        return Answer::Steered(Steered::MemoryTab(start));
    }
    let place = if display.working().is_some() { crate::mistakes::Place::Steering } else { crate::mistakes::Place::Idle };
    if let Some(reply) = crate::mistakes::handle_command(&command, place) {
        let _ = display.show_operator(&command);
        let (report, problem) = match reply {
            crate::mistakes::Reply::View(start) => return Answer::Steered(Steered::MistakeView(start)),
            crate::mistakes::Reply::Text(text) => (text, false),
            crate::mistakes::Reply::Error(text) => (text, true),
        };
        let brief = report.lines().next().unwrap_or_default().to_owned();
        let _ = display.notice(&brief, &report, problem);
        return Answer::Steered(Steered::Done);
    }
    let (content, notes) = crate::input::expand_attachments(&text);
    for note in notes {
        crate::terminal::say(&format!("{STATUS_STYLE}{note}{RESET}"));
    }
    let _ = submitter.submit(content, text);
    Answer::Steered(Steered::Done)
}

impl Drop for Server {
    fn drop(&mut self) {
        // Leave nothing behind for the next run to have to recognise as stale.
        // This only runs on a clean exit; a killed process leaves both files, and
        // `probe` is what has to make sense of them.
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(&self.pid_path);
    }
}

// ------------------------------------------------------------------ attaching

/// If an agent is already running, become a terminal of it and block until
/// the operator detaches. Returns whether that happened.
///
/// This is what makes reconnecting after a dropped SSH session do the right
/// thing: the running turn continues where it is, and the new terminal joins it
/// rather than starting a rival instance against the same memory.
pub fn attach_if_running(data_dir: &Path) -> Result<bool> {
    let stream = match probe(data_dir) {
        // Reuse the connection the probe already made.
        Occupant::Running(stream) => stream,
        Occupant::Unresponsive(pid) => {
            return Err(format!(
                "process {pid} still holds this archive but its socket is not answering, so \
                 there is nothing to attach to and starting a second s0ma is not safe. \
                 Check it with `ps -p {pid}`."
            )
            .into());
        }
        _ => return Ok(false),
    };
    stream.set_read_timeout(Some(GREETING_WITHIN))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut greeting = String::new();
    reader.read_line(&mut greeting).map_err(|error| format!("the running s0ma did not greet this terminal ({error})"))?;
    if greeting.trim_end() != HELLO {
        stream.set_read_timeout(None)?;
        attach_legacy(stream, reader, greeting.trim_end_matches(['\r', '\n']).to_string())?;
        return Ok(true);
    }
    // Back to blocking reads once the snapshot is in: the reading thread
    // waits as long as the session lasts.
    let reader = TimedReader::new(reader, Box::new({
        let stream = stream.try_clone()?;
        move |within| {
            let _ = stream.set_read_timeout(within);
        }
    }));
    let closer = stream.try_clone()?;
    Terminal::run(
        data_dir,
        Connection {
            reader,
            writer: Box::new(stream),
            close: Box::new(move || {
                let _ = closer.shutdown(std::net::Shutdown::Both);
            }),
        },
    )?;
    Ok(true)
}

/// Whether this terminal is attached from another machine (`connect`): the
/// tabs that read the server's own files can't, here.
static REMOTE: AtomicBool = AtomicBool::new(false);

pub fn is_remote() -> bool {
    REMOTE.load(Ordering::Relaxed)
}

/// A connection to the primary, whatever carries it: the socket, or SSH.
struct Connection {
    reader: TimedReader,
    writer: Box<dyn Write + Send>,
    /// Ends the connection (shuts the socket, or stops ssh).
    close: Box<dyn FnOnce() + Send>,
}

/// The reading side, with a way to bound one read (the snapshot) when the
/// transport can: a socket can; SSH's output is bounded by `connect`'s own
/// wait instead.
struct TimedReader {
    reader: Box<dyn BufRead + Send>,
    within: Box<dyn Fn(Option<Duration>) + Send>,
}

impl TimedReader {
    fn new(reader: impl BufRead + Send + 'static, within: Box<dyn Fn(Option<Duration>) + Send>) -> Self {
        Self { reader: Box::new(reader), within }
    }
}

/// A connection to an agent on another machine, greeted, before its page.
pub struct Remote {
    child: std::process::Child,
    reader: BufReader<std::process::ChildStdout>,
    stdin: std::process::ChildStdin,
}

/// `s0ma connect <target>`, first half: SSH to the server, run the bridge
/// there, and take its greeting. Before the page, so ssh can ask for a
/// password or a host key on the shell's own screen. `remote`: the command
/// that runs the agent there (`s0ma` on its PATH by default).
pub fn open_remote(target: &str, remote: &str) -> Result<Remote> {
    use std::process::{Command, Stdio};
    eprintln!("[connect] {target}: starting {remote} --bridge over ssh…");
    let mut child = Command::new("ssh")
        .args(["-T", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "--", target])
        .arg(format!("{remote} --bridge"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // Its own messages (a host key to accept, a password) reach the
        // operator as they would from ssh itself.
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("ssh could not start: {error}"))?;
    let stdin = child.stdin.take().ok_or("ssh gave no input")?;
    let stdout = child.stdout.take().ok_or("ssh gave no output")?;
    let mut reader = BufReader::new(stdout);
    let mut greeting = String::new();
    // ssh may ask for a password first: no time limit here, ssh has its own.
    let read = reader.read_line(&mut greeting);
    if read.is_err() || greeting.trim_end() != HELLO {
        let _ = child.kill();
        let status = child.wait().ok();
        let said = greeting.trim();
        return Err(match (read, said.is_empty()) {
            (Err(error), _) => format!("nothing came back from {target} ({error})"),
            (Ok(_), true) => format!(
                "{target} closed the connection{}: is s0ma running there, and is `{remote}` the command that runs it? \
                 (`--remote <command>` names another)",
                status.and_then(|s| s.code()).map(|c| format!(" (ssh exit {c})")).unwrap_or_default()
            ),
            (Ok(_), false) => format!("{target} answered {said:?}, not s0ma: is `{remote}` the command that runs it?"),
        }
        .into());
    }
    Ok(Remote { child, reader, stdin })
}

/// `s0ma connect <target>`, second half: the terminal, until detached.
pub fn attach_remote(data_dir: &Path, remote: Remote) -> Result<()> {
    let Remote { child, reader, stdin } = remote;
    REMOTE.store(true, Ordering::Relaxed);
    let child = Arc::new(Mutex::new(child));
    let stopper = Arc::clone(&child);
    let result = Terminal::run(
        data_dir,
        Connection {
            reader: TimedReader::new(reader, Box::new(|_| {})),
            writer: Box::new(stdin),
            close: Box::new(move || {
                let mut child = crate::util::locked(&stopper);
                let _ = child.kill();
                let _ = child.wait();
            }),
        },
    );
    let mut child = crate::util::locked(&child);
    let _ = child.kill();
    let _ = child.wait();
    result
}

/// `s0ma --bridge`: the far end of `connect`. Joins the agent running here
/// and passes its protocol through this process's input and output.
pub fn bridge(data_dir: &Path) -> Result<()> {
    let stream = match probe(data_dir) {
        Occupant::Running(stream) => stream,
        Occupant::Unresponsive(pid) => return Err(format!("s0ma (process {pid}) is not answering on its socket").into()),
        _ => return Err(format!("no s0ma is running here ({})", socket_path(data_dir).display()).into()),
    };
    let mut from_s0ma = stream.try_clone()?;
    let mut to_s0ma = stream.try_clone()?;
    let out = thread::spawn(move || {
        let mut stdout = std::io::stdout().lock();
        let _ = std::io::copy(&mut from_s0ma, &mut StdoutFlushing(&mut stdout));
    });
    let _ = std::io::copy(&mut std::io::stdin().lock(), &mut to_s0ma);
    // The client went (Ctrl+D, or ssh dropped): leave the agent's socket closed.
    let _ = stream.shutdown(std::net::Shutdown::Both);
    let _ = out.join();
    Ok(())
}

/// Stdout that flushes every write: a line must reach ssh as it is sent,
/// not when a buffer fills.
struct StdoutFlushing<'a, W: Write>(&'a mut W);

impl<W: Write> Write for StdoutFlushing<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.0.write(bytes)?;
        self.0.flush()?;
        Ok(written)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

/// What the reading thread hands the terminal's own.
enum Incoming {
    Message(ToTerminal),
    Closed,
}

/// An attached terminal: its own display and keyboard, drawing what the
/// primary sends and sending it what is typed.
struct Terminal {
    input: InputReader,
    display: ActiveDisplay,
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    /// Ends the connection, on leaving.
    close: Option<Box<dyn FnOnce() + Send>>,
    /// What the primary can do beyond the base protocol (`FEATURES`).
    features: Vec<String>,
    requests: Arc<Requests>,
    incoming: mpsc::Receiver<Incoming>,
    interrupt: Arc<AtomicBool>,
}

/// Requests waiting for the primary's answer, by id.
#[derive(Default)]
struct Requests {
    next: AtomicU64,
    waiting: Mutex<HashMap<u64, mpsc::Sender<Answer>>>,
    /// The connection is gone: nothing will be answered.
    closed: AtomicBool,
}

impl Requests {
    /// Send `make(id)` and wait for its answer.
    fn ask(&self, writer: &Mutex<Box<dyn Write + Send>>, make: impl FnOnce(u64) -> ToPrimary) -> std::io::Result<Answer> {
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let (tell, told) = mpsc::channel();
        if self.closed.load(Ordering::SeqCst) {
            return Err(std::io::Error::other("the connection to s0ma is closed"));
        }
        crate::util::locked(&self.waiting).insert(id, tell);
        // Closed between the look and the insert: `close` cleared the others
        // and not this one, which would wait out the whole timeout.
        if self.closed.load(Ordering::SeqCst) {
            crate::util::locked(&self.waiting).remove(&id);
            return Err(std::io::Error::other("the connection to s0ma is closed"));
        }
        if let Err(error) = send(writer, &make(id)) {
            crate::util::locked(&self.waiting).remove(&id);
            return Err(error);
        }
        let answer = told.recv_timeout(ANSWER_WITHIN);
        crate::util::locked(&self.waiting).remove(&id);
        answer.map_err(|_| std::io::Error::other("s0ma did not answer"))
    }

    /// No answer is coming: whoever waits is told now, not after a timeout.
    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        crate::util::locked(&self.waiting).clear();
    }

    fn answered(&self, id: u64, answer: Answer) {
        if let Some(tell) = crate::util::locked(&self.waiting).remove(&id) {
            let _ = tell.send(answer);
        }
    }
}

fn send(writer: &Mutex<Box<dyn Write + Send>>, message: &ToPrimary) -> std::io::Result<()> {
    let mut writer = crate::util::locked(writer);
    writeln!(writer, "{}", line(message))?;
    writer.flush()
}

impl Terminal {
    fn run(data_dir: &Path, connection: Connection) -> Result<()> {
        let Connection { reader: TimedReader { mut reader, within }, writer: stream, close } = connection;
        // Everything so far first, before this terminal reads the prompt
        // history: a job's close that purged it before the snapshot is in
        // the file it reads, and one after is mirrored to it.
        let mut first = String::new();
        reader.read_line(&mut first).map_err(|error| format!("the running s0ma sent nothing to start from ({error})"))?;
        let Ok(ToTerminal::Snapshot { display: snapshot, feeds, execs, features, approvals }) = serde_json::from_str::<ToTerminal>(&first) else {
            return Err("the running s0ma sent nothing to start from".into());
        };
        within(None);
        crate::subagent::feed::restore(feeds);
        crate::exec_feed::restore(execs);
        crate::approval::apply(approvals);

        // Ctrl+C is a key here, as on the primary; outside raw mode it must
        // not end this terminal either.
        let interrupt = Arc::new(AtomicBool::new(false));
        let signalled = Arc::clone(&interrupt);
        let _ = ctrlc::set_handler(move || signalled.store(true, Ordering::SeqCst));

        let input = InputReader::spawn(Arc::clone(&interrupt), data_dir.join("input_history.txt"))?;
        let display = input.active_display();
        // From another machine, the server's files aren't here to read.
        if !is_remote() {
            crate::tabs::init(&data_dir.join("hippocampus.jsonl"));
            crate::mistakes::open_for_view(data_dir);
        }

        let writer = Arc::new(Mutex::new(stream));
        let requests = Arc::new(Requests::default());
        {
            let (writer, requests) = (Arc::clone(&writer), Arc::clone(&requests));
            crate::input::steer_remotely(Box::new(move |text, now| {
                match requests.ask(&writer, |id| ToPrimary::Steer { id, text, now })? {
                    Answer::Steered(steered) => Ok(steered),
                    Answer::Acted(_) | Answer::Note(_) | Answer::Data(_) => Ok(Steered::Done),
                }
            }));
        }
        {
            let (writer, requests) = (Arc::clone(&writer), Arc::clone(&requests));
            crate::tabs::act_remotely(Box::new(move |action| {
                match requests.ask(&writer, |id| ToPrimary::Act { id, action }) {
                    Ok(Answer::Acted(result)) => result,
                    Ok(Answer::Steered(_) | Answer::Note(_) | Answer::Data(_)) => Ok(String::new()),
                    Err(error) => Err(format!("s0ma did not answer: {error}")),
                }
            }));
        }

        // The reading thread: answers go to whoever asked; everything else to
        // this terminal's own thread, which is woken from its prompt for it.
        let (deliver, incoming) = mpsc::channel();
        {
            let requests = Arc::clone(&requests);
            let wake = input.external_break_handle();
            thread::spawn(move || {
                for text in reader.lines() {
                    let Ok(text) = text else { break };
                    let Ok(message) = serde_json::from_str::<ToTerminal>(&text) else { continue };
                    match message {
                        ToTerminal::Answer { id, answer } => requests.answered(id, answer),
                        // Nothing else to wake the prompt for.
                        ToTerminal::Snapshot { .. } => {}
                        // Only a tab draws these, and it follows them itself:
                        // nothing to wake the prompt for.
                        ToTerminal::Feed(change) => crate::subagent::feed::apply(change),
                        ToTerminal::Exec(change) => crate::exec_feed::apply(change),
                        ToTerminal::Approvals(pending) => crate::approval::apply(pending),
                        message => {
                            if deliver.send(Incoming::Message(message)).is_err() {
                                return;
                            }
                            wake.store(true, Ordering::Relaxed);
                        }
                    }
                }
                requests.close();
                let _ = deliver.send(Incoming::Closed);
                wake.store(true, Ordering::Relaxed);
            });
        }

        let mut terminal = Terminal { input, display, writer, close: Some(close), features, requests, incoming, interrupt };
        let ended = terminal.start(snapshot).and_then(|()| terminal.serve(data_dir));
        if terminal.input.is_active() {
            let _ = terminal.input.end_active();
        }
        crate::tabs::close();
        // The message below goes on the shell's screen, under the
        // conversation's last rows.
        crate::terminal::leave_page();
        // A purge heard last thing is acted on before the history is saved.
        terminal.input.forget_history_if_asked();
        let closed = ended?;
        if let Some(close) = terminal.close.take() {
            close();
        }
        drop(terminal);
        println!(
            "{}",
            if closed { "\n[session] the s0ma process closed the connection." } else { "\n[session] detached; s0ma is still running." }
        );
        Ok(())
    }

    /// The conversation so far, drawn; a turn in progress, joined.
    fn start(&mut self, snapshot: DisplaySnapshot) -> Result<()> {
        let active = snapshot.active;
        self.display.load(snapshot)?;
        if active {
            self.input.begin_active()?;
        }
        self.display.redraw()?;
        Ok(())
    }

    /// Until Ctrl+D (false) or the primary goes away (true).
    fn serve(&mut self, data_dir: &Path) -> Result<bool> {
        loop {
            if self.input.is_eof() {
                return Ok(false);
            }
            loop {
                match self.incoming.try_recv() {
                    Ok(Incoming::Message(message)) => self.apply(message, data_dir)?,
                    Ok(Incoming::Closed) | Err(mpsc::TryRecvError::Disconnected) => return Ok(true),
                    Err(mpsc::TryRecvError::Empty) => break,
                }
            }
            if self.input.is_active() {
                // Ctrl+C in the pane (or a tab) stops the turn, as there.
                if self.interrupt.swap(false, Ordering::SeqCst) && !self.input.is_eof() {
                    send(&self.writer, &ToPrimary::Interrupt)?;
                }
                match self.incoming.recv_timeout(Duration::from_millis(25)) {
                    Ok(Incoming::Message(message)) => self.apply(message, data_dir)?,
                    Ok(Incoming::Closed) | Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(true),
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
                continue;
            }
            self.interrupt.store(false, Ordering::SeqCst);
            let event = self.input.recv()?;
            if matches!(event, InputEvent::Timeout) && !crate::tabs::is_open() {
                // Woken for what the primary sent: the prompt was erased, and
                // its first row is where the cursor is now. Whatever is drawn
                // there, the prompt comes back below it.
                let (draft, cursor) = self.input.idle_draft();
                let start = crate::terminal::cursor_row();
                let columns = crossterm::terminal::size().map_or(80, |(columns, _)| columns);
                let (_, prompt_rows) = crate::terminal::prompt_geometry(&draft, cursor, columns);
                loop {
                    match self.incoming.try_recv() {
                        Ok(Incoming::Message(message)) => self.apply(message, data_dir)?,
                        Ok(Incoming::Closed) | Err(mpsc::TryRecvError::Disconnected) => return Ok(true),
                        Err(mpsc::TryRecvError::Empty) => break,
                    }
                    if self.input.is_active() {
                        break;
                    }
                }
                if let Some(start) = start
                    && !self.input.is_active()
                {
                    crate::terminal::clear_of_prompt(start, prompt_rows)?;
                }
                continue;
            }
            match event {
                InputEvent::Submit(text) if crate::transfer::is_command(&text) => {
                    let start = crate::terminal::cursor_row();
                    let _defer = ClearOfPrompt(start);
                    self.display.apply(Mirrored::Entry(crate::terminal::operator_entry(&text)))?;
                    let said = if self.features.iter().any(|f| f == "transfer") {
                        let (writer, requests) = (Arc::clone(&self.writer), Arc::clone(&self.requests));
                        let display = self.display.clone();
                        crate::transfer::run_command(&text, &mut |request| match request {
                            crate::transfer::Request::Fetch { path, offset } => {
                                match requests.ask(&writer, |id| ToPrimary::Fetch { id, path, offset }) {
                                    Ok(Answer::Data(part)) => part,
                                    Ok(_) => Err("s0ma answered something else".into()),
                                    Err(error) => Err(format!("s0ma did not answer: {error}")),
                                }
                                .map(crate::transfer::Reply::Part)
                            }
                            crate::transfer::Request::Put { upload, name, offset, data, done } => {
                                match requests.ask(&writer, |id| ToPrimary::Put { id, upload, name, offset, data, done }) {
                                    Ok(Answer::Acted(result)) => result.map(crate::transfer::Reply::Saved),
                                    Ok(_) => Err("s0ma answered something else".into()),
                                    Err(error) => Err(format!("s0ma did not answer: {error}")),
                                }
                            }
                            crate::transfer::Request::Progress(text) => {
                                let _ = display.transient(Some(&text));
                                Ok(crate::transfer::Reply::Noted)
                            }
                        })
                    } else {
                        Err("the s0ma running there is older and can't move files; update it".into())
                    };
                    let _ = self.display.transient(None);
                    // A notice a file: what came through, and what didn't.
                    let (text, failed) = match said {
                        Ok(text) => (text, false),
                        Err(text) => (text, true),
                    };
                    for line in text.lines() {
                        self.display.notice(line, line, failed || line.starts_with("not moved"))?;
                    }
                }
                InputEvent::Submit(text) => {
                    // What is written here in answer, the prompt comes back below.
                    let start = crate::terminal::cursor_row();
                    let answer = match self.requests.ask(&self.writer, |id| ToPrimary::Say { id, text: text.clone() }) {
                        Ok(answer) => answer,
                        Err(_) if self.requests.closed.load(Ordering::SeqCst) => return Ok(true),
                        Err(error) => return Err(error.into()),
                    };
                    let _defer = ClearOfPrompt(start);
                    // What the primary drew before answering comes first.
                    loop {
                        match self.incoming.try_recv() {
                            Ok(Incoming::Message(message)) => self.apply(message, data_dir)?,
                            Ok(Incoming::Closed) | Err(mpsc::TryRecvError::Disconnected) => return Ok(true),
                            Err(mpsc::TryRecvError::Empty) => break,
                        }
                    }
                    match answer {
                        Answer::Steered(Steered::MistakeView(start)) => self.mistake_view(start),
                        Answer::Steered(Steered::MemoryTab(start)) => {
                            crate::tabs::memory_start_at(crate::tabs::MemorySection::Redactions, start);
                            crate::tabs::request(crate::tabs::Target::View(crate::tabs::View::Memory));
                        }
                        Answer::Note(note) => {
                            // Answered here only, so the line it answers is shown here too.
                            self.display.apply(Mirrored::Entry(crate::terminal::operator_entry(&text)))?;
                            let brief = note.lines().next().unwrap_or_default().to_owned();
                            self.display.notice(&brief, &note, true)?;
                        }
                        _ => {}
                    }
                }
                InputEvent::Eof => return Ok(false),
                InputEvent::Interrupt | InputEvent::Timeout => {}
            }
        }
    }

    fn apply(&mut self, message: ToTerminal, _data_dir: &Path) -> Result<()> {
        match message {
            ToTerminal::Display(Mirrored::Begin) => self.input.begin_active()?,
            ToTerminal::Display(Mirrored::End) => self.input.end_active()?,
            // At once: this terminal may be left before its prompt is read
            // again, and leaving saves the history.
            ToTerminal::Display(Mirrored::HistoryPurged) => {
                crate::jobs::note_history_purge();
                self.input.forget_history_if_asked();
            }
            ToTerminal::Display(event) => self.display.apply(event)?,
            ToTerminal::Feed(change) => crate::subagent::feed::apply(change),
            ToTerminal::Exec(change) => crate::exec_feed::apply(change),
            ToTerminal::Approvals(pending) => crate::approval::apply(pending),
            ToTerminal::Snapshot { .. } | ToTerminal::Answer { .. } => {}
        }
        Ok(())
    }

    /// `/mistake view` typed at this terminal's prompt: the window opens here.
    fn mistake_view(&self, start: Option<u64>) {
        let Some(book) = crate::mistakes::book() else { return };
        let closing = crate::mistakes::view::run(book, start)
            .unwrap_or_else(|error| format!("[mistake] the review window failed: {error}"));
        let _ = self.display.transcript_line(&safe_text(&closing));
    }
}

/// On drop: the prompt about to be drawn again goes below whatever was
/// written since the cursor was on `0` (its first row).
struct ClearOfPrompt(Option<u16>);

impl Drop for ClearOfPrompt {
    fn drop(&mut self) {
        if let Some(start) = self.0 {
            let _ = crate::terminal::clear_of_prompt(start, 1);
        }
    }
}

/// A primary from before `HELLO`: its stream of text, reasoning and status
/// frames, shown as it was then.
fn attach_legacy(stream: UnixStream, reader: BufReader<UnixStream>, first: String) -> Result<()> {
    println!("[session] an s0ma is already running (an older build); attaching to it.");
    println!("[session] type to steer it. Ctrl-D detaches and leaves it running.");
    print!("\n{OPERATOR_STYLE}you>  {RESET}");
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let printer = thread::spawn(move || {
        let mut out = std::io::stdout();
        let mut thinking_started = false;
        let mut answer_started = false;
        let mut markdown = MarkdownStream::new();

        let mut render = |tag: char, body: &str| -> std::io::Result<()> {
            markdown.set_width(crossterm::terminal::size().ok().map(|(columns, _)| columns as usize));
            let body = safe_text(body);
            match tag {
                REASONING_FRAME => {
                    if !thinking_started {
                        write!(out, "\n{THINKING_STYLE}[Thinking]{RESET}\n")?;
                        thinking_started = true;
                    }
                    write!(out, "{THINKING_STYLE}{body}{RESET}")?;
                }
                TEXT_FRAME => {
                    let text = markdown.push(&body);
                    if !answer_started {
                        if thinking_started {
                            writeln!(out)?;
                        }
                        write!(out, "{NAME_STYLE}s0ma>{RESET} {text}")?;
                        answer_started = true;
                    } else {
                        write!(out, "{text}")?;
                    }
                }
                STATUS_FRAME => {
                    write!(out, "{}", markdown.finish())?;
                    write!(out, "\n{STATUS_STYLE}{body}{RESET}\n")?;
                    thinking_started = false;
                    answer_started = false;
                }
                TURN_END_FRAME => {
                    write!(out, "{}", markdown.finish())?;
                    write!(out, "\n\n{OPERATOR_STYLE}you>  {RESET}")?;
                    thinking_started = false;
                    answer_started = false;
                }
                _ => write!(out, "{body}")?,
            }
            out.flush()
        };

        let frames = std::iter::once(Ok(first)).chain(reader.lines());
        for frame in frames {
            let Ok(frame) = frame else { break };
            let Some(tag) = frame.chars().next() else { continue };
            let body = unescape(&frame[tag.len_utf8()..]);
            if render(tag, &body).is_err() {
                break;
            }
        }
        println!("\n[session] the s0ma process closed the connection.");
    });

    let mut writer = stream;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if writeln!(writer, "{line}").and_then(|()| writer.flush()).is_err() {
            break;
        }
    }
    let _ = writer.shutdown(std::net::Shutdown::Both);
    let _ = printer.join();
    println!("[session] detached; s0ma is still running.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "s0ma-session-{tag}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst),
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn a_leftover_socket_file_is_recognised_as_stale() {
        // A process killed without running its Drop leaves the file behind. If
        // that read as "already running", a crash would lock the agent out of its own
        // socket permanently and every later start would refuse to run.
        let data = dir("stale");
        std::fs::write(socket_path(&data), b"not a socket").unwrap();
        // No pid file: nobody claims to own it.
        assert!(matches!(probe(&data), Occupant::Stale));

        // A pid file naming a process that has since exited is still stale.
        // PID 1 always exists, so use something that cannot be running.
        std::fs::write(pid_path(&data), u32::MAX.to_string()).unwrap();
        assert!(matches!(probe(&data), Occupant::Stale));
    }

    #[test]
    fn a_live_owner_is_never_treated_as_stale() {
        // The dangerous case. If an unanswered socket whose owner is alive read
        // as stale, starting the agent again would delete a running instance's socket
        // and then bind its own — two agents writing one archive, which is worse
        // than no agent at all.
        let data = dir("unresponsive");
        std::fs::write(socket_path(&data), b"not a socket").unwrap();
        std::fs::write(pid_path(&data), std::process::id().to_string()).unwrap();

        match probe(&data) {
            Occupant::Unresponsive(pid) => assert_eq!(pid, std::process::id()),
            _ => panic!("a socket owned by a living process must not be removed"),
        }
    }
}
