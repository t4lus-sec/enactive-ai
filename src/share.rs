//! Files between the two machines, both ways.
//!
//! - `/share` (the operator): copy files from the operator's machine into
//!   `/share` on the agent's.
//! - `send_to_operator` (a tool, the agent): copy a file from the agent's machine into the
//!   operator's receive folder, `~/s0ma-share` unless `S0MA_RECEIVE_DIR` says
//!   otherwise.
//!
//! ## /share
//!
//! `@path` puts a file's text into a message; this puts the file itself where
//! the agent can run, edit or keep it. The only way onto the agent's machine is the guest
//! command path (exec and status, through `virsh`), which has no file
//! transfer, so the file goes over in pieces: each piece base64-encoded inside
//! a command and appended to a hidden temp file beside its destination. Every
//! guest command crosses two single-argument limits on its way (virsh's argv,
//! then `s0ma-run`'s), each 128 KiB on Linux, and is base64-encoded twice on
//! the way, so a piece is 48 KiB of file. Then the temp file's SHA-256 is
//! checked against the original's, and it is renamed into place without
//! replacing anything: a second `notes.md` arrives as `notes (2).md`.
//!
//! Nothing here starts a turn. The next message the operator sends tells the agent
//! what arrived (`arrival_note`, appended the way the knowledge check's note
//! is), so a file landing is news exactly when the operator has something to
//! say about it.

pub(crate) mod backup;
pub(crate) mod files;
pub(crate) mod sync;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use sha2::{Digest, Sha256};

use std::path::{Path, PathBuf};

/// Where sent files land, unless `S0MA_SHARE_DIR` names another directory.
/// Created, owned by the agent, by `setup-legion.sh` and `bootstrap-guest.sh`.
pub const DEFAULT_SHARE_DIR: &str = "/share";

/// File bytes per guest command. Base64 twice makes it ~87 KiB of argument,
/// under the 128 KiB a single argument may be.
pub(crate) const CHUNK_BYTES: usize = 48 * 1024;

/// At a few hundred KB/s through the guest command path, bigger than this is a
/// job for scp.
pub const MAX_SEND_BYTES: u64 = 32 * 1024 * 1024;

/// Seconds any one guest command may take.
pub(crate) const STEP_TIMEOUT_SECS: u64 = 60;

/// The first status check of a step; `run_helper_waiting` backs off from it.
#[cfg_attr(test, allow(dead_code))]
const QUICK_STATUS_POLL: std::time::Duration = std::time::Duration::from_millis(10);

/// What a guest command did.
pub struct Ran {
    pub exit_code: Option<i64>,
    pub stdout: String,
    pub stderr: String,
}

/// A file that arrived.
#[derive(Debug)]
pub struct Sent {
    pub destination: String,
    pub bytes: u64,
    pub executable: bool,
}

impl Sent {
    /// One line for the operator and for the agent: `/share/run.sh (2.1 KB, executable)`.
    pub fn describe(&self) -> String {
        let executable = if self.executable { ", executable" } else { "" };
        format!("{} ({}{executable})", self.destination, human_size(self.bytes))
    }
}

pub fn share_dir() -> String {
    std::env::var("S0MA_SHARE_DIR")
        .ok()
        .map(|dir| dir.trim().trim_end_matches('/').to_string())
        .filter(|dir| !dir.is_empty())
        .unwrap_or_else(|| {
            // Host mode: the folder on this computer is the shared one itself.
            if crate::vm::machine::on_host() {
                receive_dir().display().to_string().trim_end_matches('/').to_string()
            } else {
                DEFAULT_SHARE_DIR.to_string()
            }
        })
}

/// The paths after `/share`, separated by spaces. A backslash or quotes keep a
/// space inside a path (Tab completion writes `\ `), as with `/img`, and a
/// leading `~/` is the home directory.
pub fn parse_paths(rest: &str) -> Result<Vec<PathBuf>, String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = rest.chars();
    while let Some(character) = chars.next() {
        match (quote, character) {
            (Some(open), close) if close == open => quote = None,
            (_, '\\') => {
                if let Some(escaped) = chars.next() {
                    word.push(escaped);
                    in_word = true;
                }
            }
            (Some(_), other) => word.push(other),
            (None, '\'' | '"') => {
                quote = Some(character);
                in_word = true;
            }
            (None, space) if space.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            (None, other) => {
                word.push(other);
                in_word = true;
            }
        }
    }
    if quote.is_some() {
        return Err("a quote in the paths is not closed".into());
    }
    if in_word {
        words.push(word);
    }
    if words.is_empty() {
        return Err(format!("usage: /share <path> [<path> …], to copy files to {} on s0ma's machine", share_dir()));
    }
    Ok(words.iter().map(|word| crate::completion::resolve_path(word)).collect())
}

/// Copy `path` into `dir` on the agent's machine, through `run` (a guest command
/// and its timeout). Nothing is left behind when it fails.
pub fn send(path: &Path, dir: &str, run: &mut dyn FnMut(&str, u64) -> Result<Ran, String>) -> Result<Sent, String> {
    let metadata = std::fs::metadata(path).map_err(|error| format!("cannot open it: {error}"))?;
    if metadata.is_dir() {
        return Err("is a directory; send the files in it, or a tar of it".into());
    }
    if !metadata.is_file() {
        return Err("is not a regular file".into());
    }
    if metadata.len() > MAX_SEND_BYTES {
        return Err(format!(
            "is {}, over the {} /share carries; copy it with scp instead",
            human_size(metadata.len()),
            human_size(MAX_SEND_BYTES)
        ));
    }
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or("has no file name")?;
    let bytes = std::fs::read(path).map_err(|error| format!("cannot read it: {error}"))?;
    let executable = is_executable(&metadata);
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let (stem, extension) = split_name(&name);

    // A hidden temp file beside where it will go, so the last step is a rename
    // within one directory.
    let prepare = format!(
        "d={dir}\n\
         [ -d \"$d\" ] || {{ echo \"$d does not exist on s0ma's machine; setup-legion.sh creates it\" >&2; exit 3; }}\n\
         [ -w \"$d\" ] || {{ echo \"$d is not writable by $(id -un)\" >&2; exit 4; }}\n\
         mktemp \"$d/.send.XXXXXXXX\"",
        dir = crate::util::shell_quote(dir)
    );
    let temp = step(run, &prepare)?.stdout.trim().to_string();
    if temp.is_empty() {
        return Err("s0ma's machine gave no temp file to write into".into());
    }

    let result = (|| {
        for piece in bytes.chunks(CHUNK_BYTES) {
            step(run, &format!("printf %s '{}' | base64 -d >> {}", STANDARD.encode(piece), crate::util::shell_quote(&temp)))?;
        }
        let finish = format!(
            "t={temp}; d={dir}\n\
             got=$(sha256sum \"$t\" | cut -d' ' -f1)\n\
             if [ \"$got\" != {digest} ]; then rm -f \"$t\"; echo \"arrived damaged (sha256 $got)\" >&2; exit 5; fi\n\
             chmod {mode} \"$t\"\n\
             f=\"$d/\"{name}; n=1\n\
             while [ -e \"$f\" ]; do n=$((n+1)); f=\"$d/\"{stem}\" ($n)\"{extension}; done\n\
             mv -n \"$t\" \"$f\"\n\
             if [ -e \"$t\" ]; then rm -f \"$t\"; echo \"could not move it into place\" >&2; exit 6; fi\n\
             printf '%s' \"$f\"",
            temp = crate::util::shell_quote(&temp),
            dir = crate::util::shell_quote(dir),
            digest = crate::util::shell_quote(&digest),
            mode = if executable { "755" } else { "644" },
            name = crate::util::shell_quote(&name),
            stem = crate::util::shell_quote(stem),
            extension = crate::util::shell_quote(extension),
        );
        step(run, &finish)
    })();

    match result {
        Ok(ran) => Ok(Sent {
            destination: ran.stdout.trim().to_string(),
            bytes: bytes.len() as u64,
            executable,
        }),
        Err(error) => {
            // Best effort: the failure is the news, not the cleanup.
            let _ = run(&format!("rm -f {}", crate::util::shell_quote(&temp)), STEP_TIMEOUT_SECS);
            Err(error)
        }
    }
}

// ------------------------------------------------------------------
// send_to_operator: the agent's machine to the operator's
// ------------------------------------------------------------------
//
// The other direction has a much larger budget: file bytes come back as a
// command's output, which never passes through an argument list. The limit is
// the harness's capture of a helper's output (1,000,000 characters) and the
// exec agent's (24 MiB), so a piece is 512 KiB, base64 699,052 characters.
//
// The file is first copied to a temp file on the agent's machine, so a file still
// being written arrives as one consistent version that its checksum describes.
// On the operator's machine it is written beside its destination and hard-linked
// into place, which fails rather than replaces when the name is taken, so no
// race can overwrite anything. It never arrives executable: a file the model
// chose to put on the operator's machine should not run just by being opened.

pub const TOOL_NAME: &str = "send_to_operator";

/// File bytes per guest command in this direction.
pub(crate) const RECEIVE_CHUNK_BYTES: usize = 512 * 1024;
const MAX_PATH_CHARS: usize = 4096;
const MAX_NAME_BYTES: usize = 255;

/// Where files the agent sends land on the operator's machine.
pub fn receive_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("S0MA_RECEIVE_DIR").filter(|dir| !dir.is_empty()) {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join("s0ma-share")
}

pub fn handles(name: &str) -> bool {
    name == TOOL_NAME
}

pub fn tool_definitions() -> Vec<crate::llm::ToolDefinition> {
    vec![crate::llm::ToolDefinition::function(
        TOOL_NAME,
        "Copy a file from your machine to the operator's, into their receive folder. Use it to hand \
         over something the operator should open or keep themselves: a report, an export, an image, a \
         build. It travels byte for byte, checked by SHA-256, up to 32 MB; send a directory as a tar or \
         zip you make first. Nothing there is ever replaced (a second report.pdf arrives as \
         report (2).pdf), it arrives without execute permission, and the operator sees where it landed. \
         Only for files outside /share: /share is mirrored to his machine already, so a file to work on \
         together, or for him to read or review, goes in /share (shared_write), and new versions go over it there. \
         Relative paths are from /workspace. Files the operator sends you arrive in /share.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "maxLength": MAX_PATH_CHARS, "description": "The file on your machine."},
                "name": {
                    "type": "string", "maxLength": MAX_NAME_BYTES,
                    "description": "A file name to give it on arrival, if not its own. A name only, no directories."
                }
            },
            "required": ["path"],
            "additionalProperties": false
        }),
    )]
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiveArgs {
    path: String,
    name: Option<String>,
}

/// The tool, on the production guest command path.
pub fn execute(call: &crate::types::ToolCall) -> Result<String, Box<dyn std::error::Error>> {
    let args: ReceiveArgs = serde_json::from_str(&call.function.arguments)?;
    let received = receive(&args.path, args.name.as_deref(), &receive_dir(), &mut run_on_machine)?;
    crate::llm::background_status(format_args!(
        "[share] s0ma sent {} → {}",
        received.source,
        received.describe()
    ));
    Ok(serde_json::json!({
        "ok": true,
        "sent": received.source,
        "to": received.destination,
        "bytes": received.bytes,
        "sha256": received.sha256,
    })
    .to_string())
}

/// A file that reached the operator.
#[derive(Debug)]
pub struct Received {
    /// The path on the agent's machine, as resolved there.
    pub source: String,
    pub destination: String,
    pub bytes: u64,
    pub sha256: String,
}

impl Received {
    pub fn describe(&self) -> String {
        format!("{} ({})", self.destination, human_size(self.bytes))
    }
}

/// The name a file arrives under: a single path component, printable.
fn arrival_name(name: &str) -> Result<String, String> {
    let cleaned: String = name.chars().map(|c| if c.is_control() { '_' } else { c }).collect();
    let cleaned = cleaned.trim();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        return Err(format!("{name:?} cannot be a file name"));
    }
    if cleaned.contains('/') || cleaned.contains('\\') {
        return Err(format!("{name:?} is a path; give a file name only"));
    }
    if cleaned.len() > MAX_NAME_BYTES {
        return Err(format!("the name is over {MAX_NAME_BYTES} bytes"));
    }
    Ok(cleaned.to_string())
}

/// Copy `path` on the agent's machine into `dir` on this one, through `run`.
pub fn receive(
    path: &str,
    name: Option<&str>,
    dir: &Path,
    run: &mut dyn FnMut(&str, u64) -> Result<Ran, String>,
) -> Result<Received, String> {
    if path.trim().is_empty() || path.len() > MAX_PATH_CHARS {
        return Err("give the path of one file on your machine".into());
    }
    // Checked before anything moves, so a bad name costs nothing.
    let chosen = name.map(arrival_name).transpose()?;

    // The path never reaches the shell as text: it travels base64 and is
    // decoded into a variable, as vm_image_view does.
    let snapshot = format!(
        "p=$(printf %s '{encoded}' | base64 -d)\n\
         case $p in \"~/\"*) p=\"$HOME/${{p#\"~/\"}}\";; esac\n\
         [ -e \"$p\" ] || {{ echo \"no such file: $p\" >&2; exit 3; }}\n\
         [ -d \"$p\" ] && {{ echo \"$p is a directory; make a tar or zip of it and send that\" >&2; exit 4; }}\n\
         [ -f \"$p\" ] || {{ echo \"$p is not a regular file\" >&2; exit 5; }}\n\
         [ -r \"$p\" ] || {{ echo \"$p is not readable by $(id -un)\" >&2; exit 6; }}\n\
         size=$(stat -Lc %s -- \"$p\")\n\
         if [ \"$size\" -gt {max} ]; then echo \"$p is $size bytes, over the {max} this carries\" >&2; exit 7; fi\n\
         t=$(mktemp) && cp -- \"$p\" \"$t\" || exit 8\n\
         printf '%s\\n%s\\n%s\\n' \"$t\" \"$(stat -c %s -- \"$t\")\" \"$(sha256sum < \"$t\" | cut -d' ' -f1)\"\n\
         printf '%s' \"$(cd \"$(dirname -- \"$p\")\" && pwd)/$(basename -- \"$p\")\" | base64 -w0",
        encoded = STANDARD.encode(path.as_bytes()),
        max = MAX_SEND_BYTES,
    );
    let ran = step(run, &snapshot)?;
    let mut lines = ran.stdout.lines();
    let (Some(temp), Some(size), Some(digest), Some(resolved)) = (lines.next(), lines.next(), lines.next(), lines.next()) else {
        return Err(format!("s0ma's machine described the file unreadably: {:?}", ran.stdout));
    };
    let temp = temp.to_string();
    let forget = |run: &mut dyn FnMut(&str, u64) -> Result<Ran, String>| {
        let _ = run(&format!("rm -f {}", crate::util::shell_quote(&temp)), STEP_TIMEOUT_SECS);
    };
    let (Ok(size), Some(source)) = (size.parse::<u64>(), STANDARD.decode(resolved).ok().and_then(|bytes| String::from_utf8(bytes).ok())) else {
        forget(run);
        return Err(format!("s0ma's machine described the file unreadably: {:?}", ran.stdout));
    };
    let digest = digest.to_string();
    let own_name = Path::new(&source).file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    let name = match chosen {
        Some(name) => name,
        None => match arrival_name(&own_name) {
            Ok(name) => name,
            Err(error) => {
                forget(run);
                return Err(error);
            }
        },
    };

    let result = fetch(run, &temp, size, &digest, &name, dir);
    forget(run);
    let destination = result?;
    Ok(Received { source, destination: destination.display().to_string(), bytes: size, sha256: digest })
}

/// Pull the snapshot over piece by piece, check it, and put it in place.
fn fetch(
    run: &mut dyn FnMut(&str, u64) -> Result<Ran, String>,
    temp: &str,
    size: u64,
    digest: &str,
    name: &str,
    dir: &Path,
) -> Result<PathBuf, String> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt as _;

    std::fs::create_dir_all(dir).map_err(|error| format!("cannot make {}: {error}", dir.display()))?;
    // Beside its destination, created fresh (never through an existing name or
    // link), readable but not executable.
    let (partial, mut file) = (0..100)
        .find_map(|attempt| {
            let partial = dir.join(format!(".receiving-{}-{attempt}", std::process::id()));
            std::fs::OpenOptions::new().write(true).create_new(true).mode(0o644).open(&partial).ok().map(|file| (partial, file))
        })
        .ok_or_else(|| format!("cannot write in {}", dir.display()))?;

    let outcome = (|| {
        let mut hasher = Sha256::new();
        let mut offset = 0u64;
        while offset < size {
            let length = (size - offset).min(RECEIVE_CHUNK_BYTES as u64);
            let ran = step(
                run,
                &format!(
                    "dd if={} iflag=skip_bytes,count_bytes skip={offset} count={length} status=none | base64 -w0",
                    crate::util::shell_quote(temp)
                ),
            )?;
            let piece = STANDARD.decode(ran.stdout.trim()).map_err(|error| format!("a piece arrived unreadable: {error}"))?;
            if piece.len() as u64 != length {
                return Err(format!("a piece arrived short: {} of {length} bytes at {offset}", piece.len()));
            }
            hasher.update(&piece);
            file.write_all(&piece).map_err(|error| format!("cannot write it: {error}"))?;
            offset += length;
        }
        file.sync_all().map_err(|error| format!("cannot write it: {error}"))?;
        let got = format!("{:x}", hasher.finalize());
        if got != digest {
            return Err(format!("arrived damaged (sha256 {got}, expected {digest})"));
        }
        // A hard link fails when the name is taken, so nothing is ever replaced,
        // even by a file that appears while this runs.
        let (stem, extension) = split_name(name);
        for n in 1..10_000 {
            let candidate = if n == 1 { dir.join(name) } else { dir.join(format!("{stem} ({n}){extension}")) };
            match std::fs::hard_link(&partial, &candidate) {
                Ok(()) => return Ok(candidate),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("cannot put it in place: {error}")),
            }
        }
        Err(format!("{} already holds too many files named {name}", dir.display()))
    })();
    let _ = std::fs::remove_file(&partial);
    outcome
}

/// The note the agent gets with the operator's next message./// The note the agent gets with the operator's next message.
pub fn arrival_note(arrived: &[String]) -> Option<String> {
    if arrived.is_empty() {
        return None;
    }
    let mut note = String::from("\n\n[FILES SENT]\nThe operator copied these onto your machine:\n");
    for line in arrived {
        note.push_str(&format!("- {line}\n"));
    }
    note.truncate(note.trim_end().len());
    Some(note)
}

/// Production: the guest command path, as the agent user. Each step takes
/// milliseconds, so its status is asked for after 10 ms and then less and less
/// often, rather than every 100 ms: measured through the shim, a 145 KB file
/// took 1.04 s at the default pace.
#[cfg(not(test))]
pub fn run_on_machine(command: &str, timeout_secs: u64) -> Result<Ran, String> {
    match crate::vm::run_helper_waiting(command, timeout_secs, QUICK_STATUS_POLL) {
        Ok(crate::vm::GuestCommandOutcome::Exited(result)) => {
            Ok(Ran { exit_code: result.exit_code, stdout: result.stdout, stderr: result.stderr })
        }
        Ok(crate::vm::GuestCommandOutcome::HostTimeout { timeout_secs, .. }) => {
            Err(format!("s0ma's machine did not finish a step within {timeout_secs}s"))
        }
        Err(error) => Err(format!("cannot reach s0ma's machine: {error}")),
    }
}

/// Tests stand the local machine in for the agent's, as `mcp_guest` does.
#[cfg(test)]
pub fn run_on_machine(command: &str, timeout_secs: u64) -> Result<Ran, String> {
    tests::local(command, timeout_secs)
}

pub(crate) fn step(run: &mut dyn FnMut(&str, u64) -> Result<Ran, String>, command: &str) -> Result<Ran, String> {
    let ran = run(command, STEP_TIMEOUT_SECS)?;
    if ran.exit_code == Some(0) {
        return Ok(ran);
    }
    let detail = ran.stderr.trim();
    Err(if detail.is_empty() {
        format!("a step failed (exit {})", ran.exit_code.map_or("none".into(), |code| code.to_string()))
    } else {
        detail.to_string()
    })
}

/// `notes.tar.gz` → (`notes.tar`, `.gz`); `.bashrc` and `Makefile` have no
/// extension, so a second one becomes `.bashrc (2)`.
pub(crate) fn split_name(name: &str) -> (&str, &str) {
    match name.rfind('.') {
        Some(dot) if dot > 0 => (&name[..dot], &name[dot..]),
        _ => (name, ""),
    }
}

#[cfg(unix)]
fn is_executable(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn is_executable(_: &std::fs::Metadata) -> bool {
    false
}

pub fn human_size(bytes: u64) -> String {
    let bytes = bytes as f64;
    if bytes < 1024.0 {
        format!("{bytes:.0} B")
    } else if bytes < 1024.0 * 1024.0 {
        format!("{:.1} KB", bytes / 1024.0)
    } else {
        format!("{:.1} MB", bytes / (1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The local machine standing in for the agent's, the way other guest modules'
    /// tests do it: each command run by bash, as this user.
    pub(super) fn local(command: &str, _timeout: u64) -> Result<Ran, String> {
        let output = std::process::Command::new("bash")
            .args(["-c", command])
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|error| error.to_string())?;
        Ok(Ran {
            exit_code: output.status.code().map(i64::from),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!("s0ma-share-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(path.join("share")).unwrap();
            Scratch(path)
        }
        fn share(&self) -> String {
            self.0.join("share").display().to_string()
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Bytes that exercise base64 and the shell: every byte value, zeros
    /// included, over several pieces.
    fn awkward(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    #[test]
    fn refusals_happen_before_anything_is_sent() {
        let scratch = Scratch::new("refuse");
        let mut calls = 0;
        let mut counting = |command: &str, timeout: u64| {
            calls += 1;
            local(command, timeout)
        };
        assert!(send(&scratch.0, &scratch.share(), &mut counting).unwrap_err().contains("is a directory"));
        assert!(send(&scratch.0.join("missing"), &scratch.share(), &mut counting).unwrap_err().contains("cannot open"));
        let huge = scratch.0.join("huge.img");
        std::fs::File::create(&huge).unwrap().set_len(MAX_SEND_BYTES + 1).unwrap();
        assert!(send(&huge, &scratch.share(), &mut counting).unwrap_err().contains("scp"));
        assert_eq!(calls, 0, "not one guest command for a file that was never going");
    }

    // ---- send_to_operator ----

    /// The agent's machine, standing in: bash with its own temp directory, home and
    /// working directory, so what the snapshot leaves behind can be checked.
    fn machine(scratch: &Scratch) -> impl FnMut(&str, u64) -> Result<Ran, String> + '_ {
        for dir in ["tmp", "home", "workspace", "received"] {
            std::fs::create_dir_all(scratch.0.join(dir)).unwrap();
        }
        move |command: &str, _timeout: u64| {
            let output = std::process::Command::new("bash")
                .args(["-c", command])
                .env("TMPDIR", scratch.0.join("tmp"))
                .env("HOME", scratch.0.join("home"))
                .current_dir(scratch.0.join("workspace"))
                .stdin(std::process::Stdio::null())
                .output()
                .map_err(|error| error.to_string())?;
            Ok(Ran {
                exit_code: output.status.code().map(i64::from),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            })
        }
    }

    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .map(|entries| entries.map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned()).collect())
            .unwrap_or_default();
        names.sort();
        names
    }

    #[test]
    fn a_file_comes_back_byte_for_byte_never_executable_and_tidily() {
        use std::os::unix::fs::PermissionsExt as _;
        let scratch = Scratch::new("back");
        let mut run = machine(&scratch);
        let original = awkward(RECEIVE_CHUNK_BYTES * 2 + 999);
        let source = scratch.0.join("workspace/build.bin");
        std::fs::write(&source, &original).unwrap();
        std::fs::set_permissions(&source, std::fs::Permissions::from_mode(0o755)).unwrap();
        let received_dir = scratch.0.join("received");

        let mut pieces = 0;
        let mut counting = |command: &str, timeout: u64| {
            pieces += usize::from(command.starts_with("dd if="));
            run(command, timeout)
        };
        let received = receive("build.bin", None, &received_dir, &mut counting).unwrap();

        assert_eq!(received.source, source.display().to_string(), "resolved from /workspace");
        assert_eq!(std::fs::read(&received.destination).unwrap(), original);
        assert_eq!(pieces, 3, "512 KiB pieces");
        assert_eq!(received.sha256, format!("{:x}", Sha256::digest(&original)));
        let mode = std::fs::metadata(&received.destination).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o644, "an executable on s0ma's machine is not one on the operator's");
        assert_eq!(entries(&received_dir), ["build.bin"], "no partial file left");
        assert!(entries(&scratch.0.join("tmp")).is_empty(), "the snapshot is gone from s0ma's machine");
    }

    #[test]
    fn a_file_still_being_written_arrives_as_one_consistent_version() {
        let scratch = Scratch::new("moving");
        let mut run = machine(&scratch);
        let source = scratch.0.join("workspace/log.txt");
        let first_version = awkward(RECEIVE_CHUNK_BYTES + 5);
        std::fs::write(&source, &first_version).unwrap();
        // Something keeps appending to it after the transfer has started.
        let mut growing = |command: &str, timeout: u64| {
            let ran = run(command, timeout);
            let mut file = std::fs::OpenOptions::new().append(true).open(&source).unwrap();
            std::io::Write::write_all(&mut file, b"more lines\n").unwrap();
            ran
        };
        let received = receive("log.txt", None, &scratch.0.join("received"), &mut growing).unwrap();
        assert_eq!(std::fs::read(&received.destination).unwrap(), first_version);
    }

    #[test]
    fn refusals_on_s0mas_machine_come_back_as_reasons() {
        let scratch = Scratch::new("refuse-back");
        let mut run = machine(&scratch);
        let received_dir = scratch.0.join("received");
        std::fs::create_dir_all(scratch.0.join("workspace/results")).unwrap();
        let huge = scratch.0.join("workspace/huge.img");
        std::fs::File::create(&huge).unwrap().set_len(MAX_SEND_BYTES + 1).unwrap();
        std::fs::write(scratch.0.join("home/note.md"), b"from home").unwrap();

        assert!(receive("results", None, &received_dir, &mut run).unwrap_err().contains("make a tar or zip"));
        assert!(receive("missing.txt", None, &received_dir, &mut run).unwrap_err().contains("no such file"));
        assert!(receive("huge.img", None, &received_dir, &mut run).unwrap_err().contains("over the"));
        assert!(entries(&received_dir).is_empty() && entries(&scratch.0.join("tmp")).is_empty());
        // And ~/ is the agent's home.
        let home = receive("~/note.md", None, &received_dir, &mut run).unwrap();
        assert_eq!(std::fs::read(&home.destination).unwrap(), b"from home");
    }
}
