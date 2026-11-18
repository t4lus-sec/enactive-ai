//! A watch on the keys: while a tab or the steering pane is up, the thread
//! reading keys passes `beat` many times a second. When none passes for
//! `STALL`, the screen has stopped answering: what every thread was doing is
//! written to `freeze-<time>.txt` beside the archive (each thread's stack,
//! from gdb when the machine has it; else each thread's name and what it
//! waits on), and once keys are read again a line on the page says where.
//!
//! `at` leaves a mark of where the key reader (or the agent's loop, between
//! turns) last was, for the file to name even without gdb.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// No key pass for this long is a freeze.
const STALL: Duration = Duration::from_secs(6);

static DIR: OnceLock<PathBuf> = OnceLock::new();
static START: OnceLock<Instant> = OnceLock::new();
/// Milliseconds since `START` of the last pass.
static BEAT: AtomicU64 = AtomicU64::new(0);
/// Keys are meant to be read now (a tab or the steering pane is up).
static WATCHING: AtomicBool = AtomicBool::new(false);
static MARK: Mutex<(&str, u64)> = Mutex::new(("", 0));
/// A freeze written, to be said once keys are read again.
static WRITTEN: Mutex<Option<(PathBuf, u64)>> = Mutex::new(None);

fn now_ms() -> u64 {
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Start watching, writing what is found beside `archive`. Once.
pub fn init(archive: &std::path::Path) {
    if cfg!(test) || DIR.set(archive.parent().map(PathBuf::from).unwrap_or_default()).is_err() {
        return;
    }
    now_ms();
    let _ = std::thread::Builder::new().name("freeze-watch".into()).spawn(|| {
        let mut reported_for = u64::MAX;
        loop {
            std::thread::sleep(Duration::from_millis(500));
            let beat = BEAT.load(Ordering::SeqCst);
            let stalled = now_ms().saturating_sub(beat);
            if !WATCHING.load(Ordering::SeqCst) || stalled < STALL.as_millis() as u64 || beat == reported_for {
                continue;
            }
            reported_for = beat;
            if let Some(path) = write(stalled) {
                *WRITTEN.lock().unwrap_or_else(|p| p.into_inner()) = Some((path, beat));
            }
        }
    });
}

/// Keys are read now, or no longer need to be (the idle prompt has them).
pub fn watch(on: bool) {
    beat();
    WATCHING.store(on, Ordering::SeqCst);
}

/// An editor of the operator's has the terminal: keys are its, not ours.
pub struct Paused(bool);

pub fn pause() -> Paused {
    Paused(WATCHING.swap(false, Ordering::SeqCst))
}

impl Drop for Paused {
    fn drop(&mut self) {
        beat();
        WATCHING.store(self.0, Ordering::SeqCst);
    }
}

/// A pass of the key reader.
pub fn beat() {
    BEAT.store(now_ms(), Ordering::SeqCst);
}

/// Where the key reader, or the agent's loop, is now.
pub fn at(place: &'static str) {
    *MARK.lock().unwrap_or_else(|p| p.into_inner()) = (place, now_ms());
}

/// A freeze since recovered from: the line for the page, once.
pub fn take_note() -> Option<String> {
    let mut written = WRITTEN.lock().unwrap_or_else(|p| p.into_inner());
    let (_, beat) = written.as_ref()?;
    if BEAT.load(Ordering::SeqCst) == *beat {
        return None;
    }
    let (path, _) = written.take()?;
    Some(format!("[freeze] the screen stopped answering for a while; what each thread was doing is in {}", path.display()))
}

fn write(stalled: u64) -> Option<PathBuf> {
    use std::fmt::Write as _;
    let dir = DIR.get()?;
    let path = dir.join(format!("freeze-{}.txt", chrono::Local::now().format("%Y%m%d-%H%M%S")));
    let (place, at) = *MARK.lock().unwrap_or_else(|p| p.into_inner());
    let mut text = String::new();
    let _ = writeln!(text, "no keys read for {:.1}s", stalled as f64 / 1000.0);
    let _ = writeln!(text, "last mark: {place} ({:.1}s ago)\n", now_ms().saturating_sub(at) as f64 / 1000.0);
    let _ = writeln!(text, "threads (name · kernel wait):");
    if let Ok(tasks) = std::fs::read_dir("/proc/self/task") {
        for task in tasks.flatten() {
            let read = |file: &str| std::fs::read_to_string(task.path().join(file)).unwrap_or_default().trim().to_string();
            let _ = writeln!(text, "  {:>8} {:<18} {}", task.file_name().to_string_lossy(), read("comm"), read("wchan"));
        }
    }
    text.push_str("\nstacks:\n");
    text.push_str(&stacks().unwrap_or_else(|why| format!("  (none: {why})\n")));
    std::fs::write(&path, text).ok()?;
    Some(path)
}

/// Every thread's stack, by gdb attached for a moment.
fn stacks() -> Result<String, String> {
    let pid = std::process::id().to_string();
    let dir = DIR.get().ok_or("no folder")?;
    let out_path = dir.join(".freeze-gdb.out");
    let out = std::fs::File::create(&out_path).map_err(|e| e.to_string())?;
    let err = out.try_clone().map_err(|e| e.to_string())?;
    // Where only a parent may trace its children (Yama), this process lets
    // its own gdb in, and only while it looks.
    #[cfg(target_os = "linux")]
    unsafe {
        libc::prctl(libc::PR_SET_PTRACER, -1i64 as libc::c_ulong, 0, 0, 0);
    }
    // The program's symbols read first, so this process is stopped only
    // for the walk of its stacks (a second or so), not while gdb reads
    // them; the output to a file, as a pipe left unread would fill and hold
    // gdb, and this process with it.
    let child = std::process::Command::new("gdb")
        .args(["-batch", "-nx", &format!("/proc/{pid}/exe"), "-p", &pid, "-ex", "thread apply all bt 40"])
        .stdin(std::process::Stdio::null())
        .stdout(out)
        .stderr(err)
        .spawn();
    let ran = match child {
        Ok(mut child) => {
            let deadline = Instant::now() + Duration::from_secs(60);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break Ok(()),
                    Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(100)),
                    _ => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break Err("gdb took too long".to_string());
                    }
                }
            }
        }
        Err(error) => Err(format!("gdb could not run ({error}); with gdb installed, each thread's stack is here")),
    };
    #[cfg(target_os = "linux")]
    unsafe {
        libc::prctl(libc::PR_SET_PTRACER, 0 as libc::c_ulong, 0, 0, 0);
    }
    ran?;
    let text = std::fs::read_to_string(&out_path).unwrap_or_default();
    let _ = std::fs::remove_file(&out_path);
    if text.contains("Thread ") { Ok(text) } else { Err(text.trim().to_string()) }
}
