//! The operator's own computer, as seen from a terminal attached with
//! `s0ma connect`.
//!
//! `s0ma connect <target>` runs s0ma's own terminal on the server over
//! `ssh -t` (so every tab is the one a terminal there shows: the same code,
//! reading the same files), and forwards a socket from the server back to
//! itself (`ssh -R`): this file link. Over it, the attached terminal on the
//! server asks the operator's computer for what only it has: a folder's
//! listing and a file's bytes (to upload into the share), and a place in its
//! Downloads (to download from the share), and its clipboard, to put copied
//! text on (never to read it). Nothing else is served.
//!
//! - The computer's side (`serve`, in `s0ma connect`) reads only regular
//!   files and folders under the home folder, never a hidden name on the way
//!   (`~/.ssh`, `~/.config` and the like are out), and writes only into
//!   Downloads (`S0MA_DOWNLOAD_DIR`), under free names, never over a file.
//! - The server's side (`set`, from `s0ma --laptop <socket>`) asks per call;
//!   the socket is the user's alone (sshd makes it 0600).
//!
//! The protocol is a JSON line each way per call.

use base64::Engine;
use serde_json::{Value, json};

use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use crate::transfer::PART_BYTES;

/// Files and folders a folder's transfer carries at most.
pub const MAX_FILES: usize = 2000;

/// Text copied to the computer's clipboard in one go, at most (bytes).
const MAX_COPY: usize = 1 << 20;

/// How long a call waits for the computer.
const ANSWER_WITHIN: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------- the computer's side

fn home() -> PathBuf {
    std::env::var_os("S0MA_LAPTOP_HOME")
        .or_else(|| std::env::var_os("HOME"))
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

fn downloads() -> PathBuf {
    if let Some(dir) = std::env::var_os("S0MA_DOWNLOAD_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    home().join("Downloads")
}

/// A path the server may read: under the home folder, by its own names, no
/// hidden one, no `..`.
fn readable(path: &str) -> Result<PathBuf, String> {
    let home = home();
    let trimmed = path.trim();
    let full = match trimmed.strip_prefix("~") {
        Some(rest) => home.join(rest.trim_start_matches('/')),
        None if trimmed.is_empty() => home.clone(),
        None => PathBuf::from(trimmed),
    };
    let plain = |under: &Path, full: &Path| -> Result<PathBuf, String> {
        let relative = full.strip_prefix(under).map_err(|_| format!("{path}: only files in your home folder"))?;
        let mut at = under.to_path_buf();
        for part in relative.components() {
            let name = part.as_os_str().to_string_lossy();
            if name == ".." || name == "." || name.starts_with('.') {
                return Err(format!("{path}: no hidden names or `..`"));
            }
            at.push(part);
        }
        Ok(at)
    };
    let at = plain(&home, &full)?;
    // Where its links lead, held to the same: a link out of the home folder
    // or into a hidden one is not followed.
    let real_home = std::fs::canonicalize(&home).map_err(|e| format!("{}: {e}", home.display()))?;
    let real = std::fs::canonicalize(&at).map_err(|e| format!("{path}: {e}"))?;
    plain(&real_home, &real)?;
    Ok(at)
}

fn handle(request: &Value) -> Result<Value, String> {
    let text = |key: &str| request.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    match request.get("op").and_then(Value::as_str).unwrap_or("") {
        "hello" => Ok(json!({
            "home": home().display().to_string(),
            "downloads": downloads().display().to_string(),
            "host": std::fs::read_to_string("/etc/hostname").map(|h| h.trim().to_string()).unwrap_or_default(),
        })),
        "list" => {
            let dir = readable(&text("path"))?;
            let mut entries = Vec::new();
            for entry in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with('.') {
                    continue;
                }
                let Ok(metadata) = std::fs::metadata(entry.path()) else { continue };
                if !metadata.is_dir() && !metadata.is_file() {
                    continue;
                }
                // A link that leads where nothing is served: not listed.
                if entry.path().to_str().is_none_or(|path| readable(path).is_err()) {
                    continue;
                }
                entries.push(json!({"name": name, "dir": metadata.is_dir(), "size": metadata.len()}));
            }
            entries.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
            Ok(json!({"path": dir.display().to_string(), "entries": entries}))
        }
        "stat" => {
            let path = readable(&text("path"))?;
            let metadata = std::fs::metadata(&path).map_err(|e| format!("{}: {e}", text("path")))?;
            Ok(json!({"path": path.display().to_string(), "dir": metadata.is_dir(), "size": metadata.len()}))
        }
        "read" => {
            let path = readable(&text("path"))?;
            let offset = request.get("offset").and_then(Value::as_u64).unwrap_or(0);
            let metadata = std::fs::metadata(&path).map_err(|e| format!("{}: {e}", text("path")))?;
            if !metadata.is_file() {
                return Err(format!("{}: not a file", text("path")));
            }
            let size = metadata.len();
            let mut file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
            file.seek(SeekFrom::Start(offset.min(size))).map_err(|e| e.to_string())?;
            let mut bytes = Vec::new();
            file.take(PART_BYTES as u64).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
            Ok(json!({"data": base64::engine::general_purpose::STANDARD.encode(&bytes), "size": size, "eof": offset + bytes.len() as u64 >= size}))
        }
        // A new folder in Downloads, under a free name.
        "folder" => {
            let name = text("name");
            if name.is_empty() || name.contains('/') || name.starts_with('.') {
                return Err(format!("{name}: not a folder name"));
            }
            let root = downloads();
            std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
            let made = (1..1000)
                .map(|n| if n == 1 { name.clone() } else { format!("{name} ({n})") })
                .find(|candidate| std::fs::create_dir(root.join(candidate)).is_ok())
                .ok_or("no free name")?;
            Ok(json!({"name": made}))
        }
        // Part of a file into Downloads: its folders made, put in place whole
        // under a free name.
        "write" => {
            let root = downloads();
            let name = text("name");
            let target = root.join(&name);
            if name.split('/').any(|part| part.is_empty() || part == ".." || part.starts_with('.')) {
                return Err(format!("{name}: not a name in Downloads"));
            }
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let offset = request.get("offset").and_then(Value::as_u64).unwrap_or(0);
            let done = request.get("done").and_then(Value::as_bool).unwrap_or(false);
            let landed = crate::transfer::write_part_saying(&root, &text("upload"), &name, offset, &text("data"), done, false)?;
            Ok(json!({"landed": landed, "downloads": root.display().to_string()}))
        }
        // Text copied in the attached terminal: onto this computer's
        // clipboard. Never read back.
        "copy" => {
            let copied = text("text");
            if copied.len() > MAX_COPY {
                return Err("too much to copy".into());
            }
            crate::clipboard::copy_here(&copied).map(|tool| json!({"tool": tool}))
        }
        other => Err(format!("not something this computer does: {other:?}")),
    }
}

/// `s0ma connect`'s side: answer the server at `socket` until the process
/// ends.
pub fn serve(socket: &Path) -> std::io::Result<()> {
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket)?;
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600))?;
    }
    std::thread::Builder::new().name("laptop".into()).spawn(move || {
        for stream in listener.incoming().flatten() {
            let _ = std::thread::Builder::new().name("laptop-call".into()).spawn(move || answer(stream));
        }
    })?;
    Ok(())
}

fn answer(stream: UnixStream) {
    let Ok(mut writer) = stream.try_clone() else { return };
    for line in BufReader::new(stream).lines() {
        let Ok(line) = line else { return };
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(request) => match handle(&request) {
                Ok(value) => json!({"ok": true, "value": value}),
                Err(error) => json!({"ok": false, "error": error}),
            },
            Err(error) => json!({"ok": false, "error": error.to_string()}),
        };
        if writeln!(writer, "{reply}").and_then(|()| writer.flush()).is_err() {
            return;
        }
    }
}

// ---------------------------------------------------------------- the server's side

static LINK: OnceLock<PathBuf> = OnceLock::new();

/// This terminal was started by `s0ma connect`: its computer is behind
/// `socket`.
pub fn set(socket: PathBuf) {
    let _ = LINK.set(socket);
}

/// Started by `s0ma connect` (the link may still be down: `ask` says so).
pub fn linked() -> bool {
    LINK.get().is_some()
}

/// The socket, gone when this terminal ends.
pub fn forget() {
    if let Some(socket) = LINK.get() {
        let _ = std::fs::remove_file(socket);
    }
}

/// `forget` however the terminal ends.
pub struct Forget;

impl Drop for Forget {
    fn drop(&mut self) {
        forget();
    }
}

fn ask(request: Value) -> Result<Value, String> {
    ask_within(request, ANSWER_WITHIN)
}

fn ask_within(request: Value, within: Duration) -> Result<Value, String> {
    let socket = LINK.get().ok_or("not connected with s0ma connect")?;
    let stream = UnixStream::connect(socket)
        .map_err(|_| "no file link to your computer (ssh could not forward it; is AllowStreamLocalForwarding off on the server?)".to_string())?;
    stream.set_read_timeout(Some(within)).map_err(|e| e.to_string())?;
    stream.set_write_timeout(Some(within)).map_err(|e| e.to_string())?;
    let mut writer = stream.try_clone().map_err(|e| e.to_string())?;
    writeln!(writer, "{request}").map_err(|e| format!("your computer: {e}"))?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).map_err(|e| format!("your computer did not answer: {e}"))?;
    let reply: Value = serde_json::from_str(&line).map_err(|_| "your computer answered something else".to_string())?;
    if reply["ok"] == true {
        Ok(reply["value"].clone())
    } else {
        Err(reply["error"].as_str().unwrap_or("failed").to_string())
    }
}

/// `text` onto the computer's clipboard. A short wait: a copy is at the keys.
pub fn copy(text: &str) -> Result<(), String> {
    if text.len() > MAX_COPY {
        return Err("too much to copy".into());
    }
    ask_within(json!({"op": "copy", "text": text}), Duration::from_secs(3)).map(|_| ())
}

/// What a listing shows of one entry.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub name: String,
    pub dir: bool,
    pub size: u64,
}

/// A folder on the computer (`~/…` or a full path), and its full path.
pub fn list(path: &str) -> Result<(String, Vec<Entry>), String> {
    // Asked while a key waits (Tab): never long.
    let value = ask_within(json!({"op": "list", "path": path}), Duration::from_secs(5))?;
    let entries = value["entries"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|e| Entry { name: e["name"].as_str().unwrap_or("").to_string(), dir: e["dir"] == true, size: e["size"].as_u64().unwrap_or(0) })
        .collect();
    Ok((value["path"].as_str().unwrap_or("").to_string(), entries))
}

/// Where downloads land on the computer, as it says it.
pub fn downloads_shown() -> String {
    ask(json!({"op": "hello"}))
        .ok()
        .and_then(|v| {
            let (home, downloads) = (v["home"].as_str()?.to_string(), v["downloads"].as_str()?.to_string());
            Some(match downloads.strip_prefix(&home) {
                Some(rest) => format!("~{rest}"),
                None => downloads,
            })
        })
        .unwrap_or_else(|| "~/Downloads".into())
}

/// A file on the computer, fetched whole into `into` (a file here).
fn fetch(path: &str, into: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<u64, String> {
    let mut file = std::fs::File::create(into).map_err(|e| e.to_string())?;
    let mut offset = 0u64;
    loop {
        let part = ask(json!({"op": "read", "path": path, "offset": offset}))?;
        let bytes = base64::engine::general_purpose::STANDARD.decode(part["data"].as_str().unwrap_or("")).map_err(|_| "a part that isn't base64".to_string())?;
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        offset += bytes.len() as u64;
        let size = part["size"].as_u64().unwrap_or(offset);
        progress(offset, size);
        if part["eof"] == true || bytes.is_empty() {
            file.sync_all().map_err(|e| e.to_string())?;
            return Ok(offset);
        }
    }
}

/// A file here sent whole to the computer's Downloads as `name` (folders in
/// it made). Where it landed, inside Downloads.
fn push(file: &Path, name: &str, progress: &mut dyn FnMut(u64, u64)) -> Result<String, String> {
    let mut opened = std::fs::File::open(file).map_err(|e| e.to_string())?;
    let size = opened.metadata().map(|m| m.len()).unwrap_or(0);
    let upload = {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        format!("{:x}{:x}", now, std::process::id())
    };
    let mut buffer = vec![0u8; PART_BYTES];
    let mut offset = 0u64;
    loop {
        let mut read = 0;
        while read < buffer.len() {
            match opened.read(&mut buffer[read..]) {
                Ok(0) => break,
                Ok(n) => read += n,
                Err(error) => return Err(error.to_string()),
            }
        }
        let done = offset + read as u64 >= size || read < buffer.len();
        let data = base64::engine::general_purpose::STANDARD.encode(&buffer[..read]);
        let value = ask(json!({"op": "write", "upload": upload, "name": name, "offset": offset, "data": data, "done": done}))?;
        offset += read as u64;
        progress(offset, size);
        if done {
            return Ok(value["landed"].as_str().unwrap_or(name).to_string());
        }
    }
}

/// A staging folder here, removed when dropped.
struct Staging(PathBuf);

impl Staging {
    fn new() -> Result<Self, String> {
        // The operator's files pass through it: theirs alone to read.
        static MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = crate::util::private_temp().map_err(|e| e.to_string())?.join(format!("laptop-{}", MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        Ok(Staging(dir))
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// What a transfer did, said in one line.
pub struct Moved {
    pub files: usize,
    pub bytes: u64,
    /// Where: in the share, or on the computer.
    pub to: String,
}

/// From the computer into the share: a file or a folder (its files, in
/// folders as they are) at `path` on the computer, into `dir`'s `folder`
/// (the share tab's), through `run` (the agent's machine).
pub fn upload(path: &str, dir: &str, folder: &str, progress: &mut dyn FnMut(String)) -> Result<Moved, String> {
    let stat = ask(json!({"op": "stat", "path": path}))?;
    let full = stat["path"].as_str().unwrap_or(path).to_string();
    let name = full.trim_end_matches('/').rsplit('/').next().unwrap_or("").to_string();
    if name.is_empty() {
        return Err(format!("{path}: no name"));
    }
    let staging = Staging::new()?;
    let into = |sub: &str| {
        let parts: Vec<&str> = [folder, sub].into_iter().filter(|p| !p.is_empty()).collect();
        if parts.is_empty() { dir.to_string() } else { format!("{dir}/{}", parts.join("/")) }
    };
    let mut run = crate::share::run_on_machine;
    // One file.
    if stat["dir"] != true {
        let local = staging.0.join(&name);
        fetch(&full, &local, &mut |done, size| progress(format!("uploading {name}: {} of {}", crate::share::human_size(done), crate::share::human_size(size))))?;
        let sent = crate::share::send(&local, &into(""), &mut run)?;
        return Ok(Moved { files: 1, bytes: stat["size"].as_u64().unwrap_or(0), to: sent.destination });
    }
    // A folder: every file in it, in its folders.
    let mut files: Vec<(String, u64)> = Vec::new();
    let mut folders: Vec<String> = vec![String::new()];
    let mut queue = vec![String::new()];
    while let Some(sub) = queue.pop() {
        let (_, entries) = list(&if sub.is_empty() { full.clone() } else { format!("{full}/{sub}") })?;
        for entry in entries {
            let rel = if sub.is_empty() { entry.name.clone() } else { format!("{sub}/{}", entry.name) };
            if entry.dir {
                folders.push(rel.clone());
                queue.push(rel);
            } else {
                files.push((rel, entry.size));
            }
            if files.len() + folders.len() > MAX_FILES {
                return Err(format!("{path}: over {MAX_FILES} files and folders; send a .tar.gz of it"));
            }
        }
    }
    let top = into(&name);
    folders.sort();
    for sub in &folders {
        let made = if sub.is_empty() { top.clone() } else { format!("{top}/{sub}") };
        let relative = crate::share::files::relative(dir, &made).unwrap_or(made.clone());
        match crate::share::files::make_dir(dir, &relative, &mut run) {
            Ok(()) | Err(crate::share::files::WriteError::Changed) => {}
            Err(crate::share::files::WriteError::Failed(error)) => return Err(error),
        }
    }
    let mut bytes = 0;
    for (index, (rel, size)) in files.iter().enumerate() {
        let file_name = rel.rsplit('/').next().unwrap_or(rel);
        let local = staging.0.join(format!("{index}")).join(file_name);
        std::fs::create_dir_all(local.parent().unwrap_or(&staging.0)).map_err(|e| e.to_string())?;
        progress(format!("uploading {name}: {} of {} files", index + 1, files.len()));
        fetch(&format!("{full}/{rel}"), &local, &mut |_, _| {})?;
        let parent = rel.rsplit_once('/').map(|(p, _)| format!("{top}/{p}")).unwrap_or(top.clone());
        crate::share::send(&local, &parent, &mut run).map_err(|e| format!("{rel}: {e}"))?;
        let _ = std::fs::remove_file(&local);
        bytes += size;
    }
    Ok(Moved { files: files.len(), bytes, to: top })
}

/// From the share to the computer's Downloads: `paths` are the share's files
/// (`/share/notes/a.md`), each `(source, its path under the folder taken)`;
/// `folder`: the name of the folder they go in there (a free one is made),
/// none for one file.
pub fn download(paths: &[(String, String)], folder: Option<&str>, progress: &mut dyn FnMut(String)) -> Result<Moved, String> {
    if paths.len() > MAX_FILES {
        return Err(format!("over {MAX_FILES} files; download a .tar.gz of it"));
    }
    let top = match folder {
        Some(name) => Some(ask(json!({"op": "folder", "name": name}))?["name"].as_str().unwrap_or(name).to_string()),
        None => None,
    };
    let staging = Staging::new()?;
    let mut run = crate::share::run_on_machine;
    let (mut bytes, mut landed) = (0u64, String::new());
    for (index, (source, rel)) in paths.iter().enumerate() {
        let here = staging.0.join(format!("{index}"));
        std::fs::create_dir_all(&here).map_err(|e| e.to_string())?;
        let received = crate::share::receive(source, None, &here, &mut run).map_err(|e| format!("{source}: {e}"))?;
        let file = PathBuf::from(&received.destination);
        bytes += std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0);
        let name = match &top {
            Some(top) => format!("{top}/{rel}"),
            None => rel.clone(),
        };
        let shown = rel.clone();
        landed = push(&file, &name, &mut |done, size| {
            progress(if paths.len() > 1 {
                format!("downloading {}: {} of {} files", top.as_deref().unwrap_or(""), index + 1, paths.len())
            } else {
                format!("downloading {shown}: {} of {}", crate::share::human_size(done), crate::share::human_size(size))
            })
        })?;
        let _ = std::fs::remove_dir_all(&here);
    }
    let to = match &top {
        Some(top) => format!("{}/{top}", downloads_shown()),
        None => format!("{}/{landed}", downloads_shown()),
    };
    Ok(Moved { files: paths.len(), bytes, to })
}

/// `/download` and `/upload` typed on a terminal `s0ma connect` started:
/// between the share and the operator's computer.
pub fn command(text: &str, progress: &mut dyn FnMut(String)) -> Result<String, String> {
    // Paths as typed: `~` is the computer's home, not this one's.
    let words: Vec<String> = crate::transfer::words(text).into_iter().skip(1).collect();
    let dir = crate::share::share_dir();
    match text.split_whitespace().next() {
        Some("/download") if !words.is_empty() => {
            let mut said = Vec::new();
            for path in &words {
                let relative = crate::share::files::relative(&dir, path).unwrap_or(path.clone());
                let name = relative.rsplit('/').next().unwrap_or(&relative).to_string();
                let source = crate::share::files::shown(&dir, &relative);
                said.push(match download(&[(source, name.clone())], None, progress) {
                    Ok(moved) => format!("downloaded {name} ({}) to {}", crate::share::human_size(moved.bytes), moved.to),
                    Err(error) => format!("not moved: {error}"),
                });
            }
            Ok(said.join("\n"))
        }
        Some("/upload") if !words.is_empty() => {
            let (paths, to) = match words.iter().position(|w| w == "--to") {
                Some(at) => (&words[..at], words.get(at + 1).cloned().unwrap_or_default()),
                None => (&words[..], String::new()),
            };
            let folder = crate::share::files::relative(&dir, &to).unwrap_or(to);
            let mut said = Vec::new();
            for path in paths {
                said.push(match upload(path, &dir, &folder, progress) {
                    Ok(moved) if moved.files == 1 => format!("uploaded {path} ({}) to {}", crate::share::human_size(moved.bytes), moved.to),
                    Ok(moved) => format!("uploaded {path}: {} files ({}) to {}", moved.files, crate::share::human_size(moved.bytes), moved.to),
                    Err(error) => format!("not moved: {error}"),
                });
            }
            Ok(said.join("\n"))
        }
        _ => Err("/download <file in the share> … · /upload <file or folder on your computer> … [--to <folder in the share>]".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_computer_serves_only_its_home_and_writes_only_downloads() {
        let base = std::env::temp_dir().join(format!("s0ma-laptop-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("home/docs/sub")).unwrap();
        std::fs::create_dir_all(base.join("home/.ssh")).unwrap();
        std::fs::write(base.join("home/docs/a.txt"), "a").unwrap();
        std::fs::write(base.join("home/.ssh/id"), "secret").unwrap();
        // SAFETY: only this test reads these two.
        unsafe {
            std::env::set_var("S0MA_LAPTOP_HOME", base.join("home"));
            std::env::set_var("S0MA_DOWNLOAD_DIR", base.join("home/Downloads"));
        }
        let socket = base.join("link.sock");
        serve(&socket).unwrap();
        set(socket.clone());
        let (_, entries) = list("~/docs").unwrap();
        assert_eq!(entries.iter().map(|e| (e.name.as_str(), e.dir)).collect::<Vec<_>>(), [("a.txt", false), ("sub", true)]);
        assert!(list("~").unwrap().1.iter().all(|e| !e.name.starts_with('.')), "hidden left out");
        assert!(ask(json!({"op": "read", "path": "~/.ssh/id"})).is_err(), "no hidden names");
        assert!(ask(json!({"op": "read", "path": "/etc/hostname"})).is_err(), "home only");
        assert!(ask(json!({"op": "read", "path": "~/docs/../.ssh/id"})).is_err());
        std::os::unix::fs::symlink(base.join("home/.ssh"), base.join("home/docs/keys")).unwrap();
        std::os::unix::fs::symlink("/etc", base.join("home/docs/etc")).unwrap();
        assert!(ask(json!({"op": "read", "path": "~/docs/keys/id"})).is_err(), "no link into a hidden folder");
        assert!(ask(json!({"op": "list", "path": "~/docs/etc"})).is_err(), "no link out of the home folder");
        // Fetched whole, and pushed into Downloads under free names.
        let here = base.join("fetched.txt");
        fetch("~/docs/a.txt", &here, &mut |_, _| {}).unwrap();
        assert_eq!(std::fs::read_to_string(&here).unwrap(), "a");
        assert_eq!(push(&here, "a.txt", &mut |_, _| {}).unwrap(), "a.txt");
        assert_eq!(push(&here, "a.txt", &mut |_, _| {}).unwrap(), "a (2).txt");
        let folder = ask(json!({"op": "folder", "name": "notes"})).unwrap()["name"].as_str().unwrap().to_string();
        assert_eq!(push(&here, &format!("{folder}/deep/a.txt"), &mut |_, _| {}).unwrap(), "notes/deep/a.txt");
        assert!(ask(json!({"op": "write", "upload": "abcdefgh1", "name": "../x", "offset": 0, "data": "", "done": true})).is_err());
        // The clipboard is written to, within a size, and never read.
        let too_much = handle(&json!({"op": "copy", "text": "x".repeat(MAX_COPY + 1)})).unwrap_err();
        assert_eq!(too_much, "too much to copy");
        assert!(handle(&json!({"op": "paste"})).unwrap_err().contains("not something"));
        assert_eq!(ask(json!({"op": "folder", "name": "notes"})).unwrap()["name"], "notes (2)");
        let _ = std::fs::remove_dir_all(&base);
    }
}
