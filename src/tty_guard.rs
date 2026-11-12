//! Keys that would stop or freeze the agent from outside, made harmless.
//!
//! In raw mode every key reaches the agent as a key. Between raw stretches (a
//! moment between the prompt and a turn, a command being handled, a strike
//! being settled), the terminal is in its ordinary mode for a moment, and
//! two keys act on the process from outside:
//!
//! - **Ctrl+Z** stopped the agent (SIGTSTP). The shell took the screen back,
//!   The agent's page and all, and since bash does not put raw mode back on
//!   `fg`, what came back read keys a line at a time and echoed them: a soft
//!   lock that only a restart got out of. SIGTSTP is now ignored, in the agent
//!   and in what it runs.
//! - **Ctrl+S** froze the terminal's output (XOFF) until Ctrl+Q, which looks
//!   the same as a hang. Output flow control (IXON, IXOFF) is off while the agent
//!   runs, and put back as it was when it ends.
//!
//! Something can still stop the agent: SIGSTOP cannot be ignored (nano's ^T^Z
//! sends it to its whole process group, the agent included). On being continued,
//! the agent puts its raw mode back if it had it, and draws everything again: a
//! SIGWINCH to itself, which every screen already answers with a full
//! redraw.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Mutex;

/// The terminal's flow control as it was: IXON and IXOFF bits.
static ORIGINAL_FLOW: Mutex<Option<libc::tcflag_t>> = Mutex::new(None);
/// The write end of the pipe SIGCONT wakes the resume thread through.
static WAKE: AtomicI32 = AtomicI32::new(-1);

const FLOW: libc::tcflag_t = libc::IXON | libc::IXOFF;

/// Once, early in `main`, before anything takes raw mode.
pub fn install() {
    // SAFETY: setting a signal's disposition to ignore has no preconditions.
    unsafe {
        libc::signal(libc::SIGTSTP, libc::SIG_IGN);
    }
    // SAFETY: isatty only reads the descriptor's state.
    if unsafe { libc::isatty(0) } != 1 {
        return;
    }
    if let Some(flow) = flow_off()
        && let Ok(mut original) = ORIGINAL_FLOW.lock()
    {
        *original = Some(flow);
        // SAFETY: `put_back` is a plain extern "C" function with no arguments.
        unsafe {
            libc::atexit(put_back);
        }
    }
    watch_continue();
}

/// Flow control off on stdin's terminal; what it was.
fn flow_off() -> Option<libc::tcflag_t> {
    // SAFETY: a zeroed termios is a valid out-parameter for tcgetattr.
    let mut modes: libc::termios = unsafe { std::mem::zeroed() };
    // SAFETY: fd 0 is a terminal (checked) and `modes` is a valid pointer.
    if unsafe { libc::tcgetattr(0, &mut modes) } != 0 {
        return None;
    }
    let was = modes.c_iflag & FLOW;
    modes.c_iflag &= !FLOW;
    // SAFETY: as above; the modes are the terminal's own, less two bits.
    (unsafe { libc::tcsetattr(0, libc::TCSANOW, &modes) } == 0).then_some(was)
}

/// At exit: flow control as it was before the agent.
extern "C" fn put_back() {
    let Some(flow) = ORIGINAL_FLOW.lock().ok().and_then(|original| *original) else { return };
    // SAFETY: as in `flow_off`.
    let mut modes: libc::termios = unsafe { std::mem::zeroed() };
    unsafe {
        if libc::tcgetattr(0, &mut modes) == 0 {
            modes.c_iflag = (modes.c_iflag & !FLOW) | flow;
            libc::tcsetattr(0, libc::TCSANOW, &modes);
        }
    }
}

extern "C" fn on_continue(_: libc::c_int) {
    let fd = WAKE.load(Ordering::SeqCst);
    if fd >= 0 {
        // SAFETY: write is async-signal-safe; one byte from a static.
        unsafe {
            libc::write(fd, b"c".as_ptr().cast(), 1);
        }
    }
}

/// A thread that puts the terminal right each time the agent is continued after
/// a stop.
fn watch_continue() {
    let mut fds = [0 as libc::c_int; 2];
    // SAFETY: `fds` has room for the two descriptors pipe writes.
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return;
    }
    let [read, write] = fds;
    WAKE.store(write, Ordering::SeqCst);
    let spawned = std::thread::Builder::new().name("tty-resume".into()).spawn(move || {
        let mut byte = 0u8;
        loop {
            // SAFETY: `read` is this pipe's read end, open for the process's life.
            let got = unsafe { libc::read(read, (&mut byte as *mut u8).cast(), 1) };
            if got <= 0 {
                if got < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return;
            }
            resumed();
        }
    });
    if spawned.is_ok() {
        // SAFETY: `on_continue` only writes a byte to a pipe.
        unsafe {
            libc::signal(libc::SIGCONT, on_continue as extern "C" fn(libc::c_int) as libc::sighandler_t);
        }
    }
}

/// Continued after a stop: the shell left the terminal in its ordinary mode
/// and its own text on the agent's screen.
fn resumed() {
    let _ = flow_off();
    if crossterm::terminal::is_raw_mode_enabled().unwrap_or(false) {
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::terminal::enable_raw_mode();
    }
    // SAFETY: raising a signal at this process has no preconditions.
    unsafe {
        libc::raise(libc::SIGWINCH);
    }
}
