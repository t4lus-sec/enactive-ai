//! Moving JSON-RPC messages between s0ma and one MCP server.
//!
//! Three transports share one `Router`, which matches responses to waiting
//! requests by id and handles what the server sends unprompted:
//!
//! - **line streams** — one JSON message per line, over either a host child
//!   process's stdin/stdout or a TCP connection to a server in the VM (relayed
//!   by `socat` there). A host child gets its own process group so the
//!   operator's Ctrl+C (which interrupts s0ma's turn) does not also kill it,
//!   and a scrubbed environment unless `inherit_env`. Its stderr goes to
//!   `data/mcp/<server>.log`, never the terminal.
//! - **Streamable HTTP** — one POST per message; the reply is a JSON body or an
//!   SSE stream, read on a worker thread so the caller can time out or be
//!   interrupted while it waits.
//! - **HTTP+SSE** — the deprecated 2024-11-05 transport: a long-lived GET stream
//!   carries every server message; requests are POSTed to the endpoint it names.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use reqwest::{
    StatusCode, Url,
    blocking::{Client as HttpClient, RequestBuilder, Response},
    header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fmt,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio as Piped},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::{Duration, Instant},
};

/// One JSON-RPC message, whichever transport carries it.
pub const MAX_MESSAGE_BYTES: usize = 32 * 1024 * 1024;
const MAX_ERROR_BODY_BYTES: usize = 16 * 1024;
const MAX_STDERR_LINE_BYTES: usize = 8 * 1024;
const STDERR_TAIL_LINES: usize = 12;
const LOG_ROTATE_BYTES: u64 = 2 * 1024 * 1024;
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);
const POST_TIMEOUT: Duration = Duration::from_secs(30);

/// What a stdio server inherits when `inherit_env` is off: the variables the
/// reference MCP SDKs pass by default, plus locale and temp dir.
const ENV_PASSTHROUGH: &[&str] = &[
    "HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// A non-success HTTP status, with the start of its body.
    Http { status: u16, body: String },
    /// The process exited or the stream ended; nothing more will arrive.
    Closed(String),
    /// A legacy Streamable HTTP server forgot our session (HTTP 404).
    SessionExpired,
    Io(String),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http { status, body } if body.trim().is_empty() => write!(f, "HTTP {status}"),
            Self::Http { status, body } => write!(f, "HTTP {status}: {}", clip(body.trim(), 300)),
            Self::Closed(why) => write!(f, "connection closed: {why}"),
            Self::SessionExpired => write!(f, "the server ended the MCP session"),
            Self::Io(why) => write!(f, "{why}"),
        }
    }
}

/// The full JSON-RPC response object, or why none will come.
pub type Reply = Result<Value, TransportError>;

pub trait Transport: Send + Sync {
    fn kind(&self) -> &'static str;
    /// Send a request whose response the router will deliver to `id`'s
    /// receiver. `timeout` bounds how long the transport may keep the exchange
    /// open; the caller enforces its own deadline separately.
    fn send_request(&self, id: u64, method: &str, message: &Value, timeout: Duration)
    -> Result<(), TransportError>;
    fn send_notification(&self, message: &Value) -> Result<(), TransportError>;
    /// The protocol version to stamp on HTTP headers, and whether the
    /// 2026-07-28 per-request headers apply. Header-less transports ignore it.
    fn configure(&self, _version: Option<&str>, _modern: bool) {}
    fn alive(&self) -> bool;
    fn close(&self);
}

// ------------------------------------------------------------------
// Router
// ------------------------------------------------------------------

pub struct Router {
    pending: Mutex<HashMap<u64, Sender<Reply>>>,
    /// Set when the server says its tool list changed. The exposed list does
    /// not follow on its own: changing tools mid-session re-renders the whole
    /// prompt prefix, so it waits for the operator's `/mcp reload`.
    stale: AtomicBool,
    stderr_tail: Mutex<VecDeque<String>>,
    log: Mutex<Option<File>>,
}

impl Router {
    pub fn new(log_path: Option<PathBuf>) -> Arc<Self> {
        let log = log_path.as_deref().and_then(open_log);
        Arc::new(Self {
            pending: Mutex::new(HashMap::new()),
            stale: AtomicBool::new(false),
            stderr_tail: Mutex::new(VecDeque::new()),
            log: Mutex::new(log),
        })
    }

    pub fn register(&self, id: u64) -> Receiver<Reply> {
        let (sender, receiver) = mpsc::channel();
        lock(&self.pending).insert(id, sender);
        receiver
    }

    pub fn forget(&self, id: u64) {
        lock(&self.pending).remove(&id);
    }

    /// Deliver the reply for `id` if anyone is still waiting for it.
    pub fn complete(&self, id: u64, reply: Reply) -> bool {
        let sender = lock(&self.pending).remove(&id);
        sender.is_some_and(|sender| sender.send(reply).is_ok())
    }

    pub fn fail_all(&self, error: TransportError) {
        for (_, sender) in lock(&self.pending).drain() {
            let _ = sender.send(Err(error.clone()));
        }
    }

    pub fn stale(&self) -> bool {
        self.stale.load(Ordering::SeqCst)
    }

    /// A fresh `tools/list` is about to be taken. A change announced before
    /// it (servers often register tools right after connecting) is covered by
    /// it; one announced after it sets the flag again.
    pub fn clear_stale(&self) {
        self.stale.store(false, Ordering::SeqCst);
    }

    pub fn log(&self, line: &str) {
        let mut log = lock(&self.log);
        if let Some(file) = log.as_mut() {
            let _ = writeln!(file, "{} {line}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
        }
    }

    fn note_stderr(&self, line: &str) {
        {
            let mut tail = lock(&self.stderr_tail);
            if tail.len() == STDERR_TAIL_LINES {
                tail.pop_front();
            }
            tail.push_back(clip(line, 300));
        }
        self.log(&format!("stderr: {line}"));
    }

    /// The last few stderr lines, for saying why a server failed.
    pub fn stderr_tail(&self) -> String {
        let tail = lock(&self.stderr_tail);
        tail.iter().rev().take(4).rev().cloned().collect::<Vec<_>>().join(" | ")
    }

    /// Route one incoming message (or a legacy batch). Returns the responses
    /// owed to server-initiated requests; the transport sends them back.
    pub fn dispatch(&self, message: Value) -> Vec<Value> {
        match message {
            Value::Array(batch) => batch.into_iter().flat_map(|m| self.dispatch(m)).collect(),
            Value::Object(_) => self.dispatch_one(message).into_iter().collect(),
            other => {
                self.log(&format!("ignored non-object message: {}", clip(&other.to_string(), 200)));
                Vec::new()
            }
        }
    }

    fn dispatch_one(&self, message: Value) -> Option<Value> {
        let method = message.get("method").and_then(Value::as_str).map(str::to_owned);
        let id = message.get("id").cloned().filter(|id| !id.is_null());
        match (method, id) {
            // A request from the server. s0ma offers no sampling, roots or
            // elicitation, so everything but ping is politely refused.
            (Some(method), Some(id)) => {
                self.log(&format!("server request: {method}"));
                Some(if method == "ping" {
                    json!({"jsonrpc": "2.0", "id": id, "result": {}})
                } else {
                    json!({"jsonrpc": "2.0", "id": id, "error": {
                        "code": -32601,
                        "message": format!("s0ma's MCP client does not support {method}")
                    }})
                })
            }
            (Some(method), None) => {
                match method.as_str() {
                    "notifications/tools/list_changed" => {
                        self.stale.store(true, Ordering::SeqCst);
                        self.log("server reports its tool list changed");
                    }
                    "notifications/message" => {
                        let params = message.get("params").cloned().unwrap_or(Value::Null);
                        let level = params.get("level").and_then(Value::as_str).unwrap_or("info");
                        let data = params.get("data").map(|d| match d {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        });
                        self.log(&format!("server log [{level}]: {}", clip(&data.unwrap_or_default(), 2000)));
                    }
                    "notifications/progress" | "notifications/cancelled" => {}
                    other => self.log(&format!("notification: {other}")),
                }
                None
            }
            (None, Some(id)) => {
                match id.as_u64().or_else(|| id.as_str().and_then(|s| s.parse().ok())) {
                    Some(id) => {
                        self.complete(id, Ok(message));
                    }
                    None => self.log(&format!("response with unexpected id {id}")),
                }
                None
            }
            (None, None) => {
                self.log(&format!("ignored message: {}", clip(&message.to_string(), 200)));
                None
            }
        }
    }
}

fn open_log(path: &Path) -> Option<File> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if fs::metadata(path).is_ok_and(|m| m.len() > LOG_ROTATE_BYTES) {
        let _ = fs::rename(path, path.with_extension("log.1"));
    }
    OpenOptions::new().create(true).append(true).open(path).ok()
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub use crate::util::clip;

/// Read one `\n`-terminated line into `out` without letting it grow past
/// `cap`. `None` at end of stream; `Some(false)` when the line was too long
/// (it is consumed and discarded); `Some(true)` otherwise.
pub fn read_line_capped<R: BufRead>(
    reader: &mut R,
    out: &mut Vec<u8>,
    cap: usize,
) -> std::io::Result<Option<bool>> {
    out.clear();
    let mut overflow = false;
    let mut read_any = false;
    loop {
        let available = match reader.fill_buf() {
            Ok(buffer) => buffer,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if available.is_empty() {
            return Ok(read_any.then_some(!overflow));
        }
        read_any = true;
        let newline = available.iter().position(|&b| b == b'\n');
        let chunk = &available[..newline.unwrap_or(available.len())];
        if !overflow {
            if out.len() + chunk.len() > cap {
                overflow = true;
                out.clear();
            } else {
                out.extend_from_slice(chunk);
            }
        }
        let used = newline.map_or(available.len(), |i| i + 1);
        reader.consume(used);
        if newline.is_some() {
            if out.last() == Some(&b'\r') {
                out.pop();
            }
            return Ok(Some(!overflow));
        }
    }
}

// ------------------------------------------------------------------
// Line streams: stdio on the host, or TCP to a server in the VM
// ------------------------------------------------------------------
//
// Both carry the same framing, one JSON-RPC message per line; the 2026-07-28
// stdio binding says it works unchanged over any reliable byte stream. A VM
// server's stdin/stdout are relayed onto a TCP connection by `socat` in the
// guest (see mcp_guest.rs), so from here it is just a socket.

pub struct StdioSpec<'a> {
    pub command: &'a str,
    pub args: &'a [String],
    pub env: &'a BTreeMap<String, String>,
    pub cwd: Option<&'a Path>,
    pub inherit_env: bool,
}

type Writer = Arc<Mutex<Option<Box<dyn Write + Send>>>>;

enum Owner {
    Child(Child),
    Socket(TcpStream),
}

pub struct LineStream {
    kind: &'static str,
    writer: Writer,
    alive: Arc<AtomicBool>,
    owner: Mutex<Option<Owner>>,
}

impl LineStream {
    /// Start a host process and speak over its stdin/stdout.
    pub fn spawn(spec: StdioSpec<'_>, router: Arc<Router>) -> Result<Self, String> {
        let mut command = Command::new(spec.command);
        command
            .args(spec.args)
            .stdin(Piped::piped())
            .stdout(Piped::piped())
            .stderr(Piped::piped());
        if !spec.inherit_env {
            command.env_clear();
            for key in ENV_PASSTHROUGH {
                if let Some(value) = std::env::var_os(key) {
                    command.env(key, value);
                }
            }
        }
        command.envs(spec.env);
        if let Some(cwd) = spec.cwd {
            command.current_dir(cwd);
        }
        // Ctrl+C goes to the terminal's foreground process group. It means
        // "interrupt s0ma's turn", which cancels the call politely; it should
        // not kill every server s0ma started.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }

        let mut child = command
            .spawn()
            .map_err(|error| format!("cannot start '{}': {error}", spec.command))?;
        router.log(&format!("started pid {}: {} {}", child.id(), spec.command, spec.args.join(" ")));

        let stdout = child.stdout.take().ok_or("child stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("child stderr unavailable")?;
        let stdin: Box<dyn Write + Send> = Box::new(child.stdin.take().ok_or("child stdin unavailable")?);
        {
            let router = Arc::clone(&router);
            thread::Builder::new()
                .name("mcp-stderr".into())
                .spawn(move || {
                    let mut reader = BufReader::new(stderr);
                    let mut line = Vec::new();
                    while let Ok(Some(_)) = read_line_capped(&mut reader, &mut line, MAX_STDERR_LINE_BYTES) {
                        let text = String::from_utf8_lossy(&line);
                        if !text.trim().is_empty() {
                            router.note_stderr(text.trim_end());
                        }
                    }
                })
                .map_err(|e| e.to_string())?;
        }
        Self::start("stdio", stdout, stdin, Owner::Child(child), router, |router| {
            // Give the stderr reader a moment so the reason is in the tail.
            thread::sleep(Duration::from_millis(50));
            let tail = router.stderr_tail();
            if tail.is_empty() {
                "the server process exited".into()
            } else {
                format!("the server process exited; stderr: {tail}")
            }
        })
    }

    /// Connect to a server relayed onto TCP. The relay may still be starting,
    /// so a refused connection is retried until `deadline`.
    pub fn connect(
        address: SocketAddr,
        deadline: Instant,
        router: Arc<Router>,
        ended: String,
    ) -> Result<Self, String> {
        let stream = loop {
            let wait = deadline.saturating_duration_since(Instant::now()).clamp(Duration::from_millis(100), Duration::from_secs(3));
            match TcpStream::connect_timeout(&address, wait) {
                Ok(stream) => break stream,
                Err(error) if Instant::now() >= deadline => {
                    return Err(format!("cannot connect to {address}: {error}"));
                }
                Err(_) => thread::sleep(Duration::from_millis(100)),
            }
        };
        let _ = stream.set_nodelay(true);
        router.log(&format!("connected to {address}"));
        let reader = stream.try_clone().map_err(|e| e.to_string())?;
        let writer: Box<dyn Write + Send> = Box::new(stream.try_clone().map_err(|e| e.to_string())?);
        Self::start("vm", reader, writer, Owner::Socket(stream), router, move |_| ended.clone())
    }

    fn start(
        kind: &'static str,
        reader: impl Read + Send + 'static,
        writer: Box<dyn Write + Send>,
        owner: Owner,
        router: Arc<Router>,
        ended: impl Fn(&Router) -> String + Send + 'static,
    ) -> Result<Self, String> {
        let writer: Writer = Arc::new(Mutex::new(Some(writer)));
        let alive = Arc::new(AtomicBool::new(true));
        {
            let writer = Arc::clone(&writer);
            let alive = Arc::clone(&alive);
            thread::Builder::new()
                .name(format!("mcp-{kind}-in"))
                .spawn(move || {
                    let mut reader = BufReader::new(reader);
                    let mut line = Vec::new();
                    loop {
                        match read_line_capped(&mut reader, &mut line, MAX_MESSAGE_BYTES) {
                            Ok(None) | Err(_) => break,
                            Ok(Some(false)) => {
                                router.log("dropped a message larger than the size limit");
                                continue;
                            }
                            Ok(Some(true)) => {}
                        }
                        if line.iter().all(u8::is_ascii_whitespace) {
                            continue;
                        }
                        match serde_json::from_slice::<Value>(&line) {
                            Ok(message) => {
                                for reply in router.dispatch(message) {
                                    let _ = write_line(&writer, &reply);
                                }
                            }
                            // A server that logs to stdout breaks the protocol;
                            // keep the line for diagnosis and carry on.
                            Err(_) => router.log(&format!(
                                "stdout (not JSON-RPC): {}",
                                clip(&String::from_utf8_lossy(&line), 300)
                            )),
                        }
                    }
                    alive.store(false, Ordering::SeqCst);
                    let reason = ended(&router);
                    router.log(&format!("stream closed: {reason}"));
                    router.fail_all(TransportError::Closed(reason));
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(Self { kind, writer, alive, owner: Mutex::new(Some(owner)) })
    }
}

fn write_line(writer: &Mutex<Option<Box<dyn Write + Send>>>, message: &Value) -> Result<(), TransportError> {
    // serde_json escapes newlines inside strings, so this is one line.
    let mut line = serde_json::to_vec(message).map_err(|e| TransportError::Io(e.to_string()))?;
    line.push(b'\n');
    let mut writer = lock(writer);
    let pipe = writer
        .as_mut()
        .ok_or_else(|| TransportError::Closed("the stream is already closed".into()))?;
    pipe.write_all(&line)
        .and_then(|()| pipe.flush())
        .map_err(|e| TransportError::Closed(format!("cannot write to the server: {e}")))
}

impl Transport for LineStream {
    fn kind(&self) -> &'static str {
        self.kind
    }

    fn send_request(&self, _id: u64, _method: &str, message: &Value, _timeout: Duration) -> Result<(), TransportError> {
        if !self.alive() {
            return Err(TransportError::Closed("the server is not running".into()));
        }
        // A failed write means the server is gone even if the reader has not
        // seen EOF yet; mark it now so the next call reconnects.
        write_line(&self.writer, message).inspect_err(|_| self.alive.store(false, Ordering::SeqCst))
    }

    fn send_notification(&self, message: &Value) -> Result<(), TransportError> {
        write_line(&self.writer, message).inspect_err(|_| self.alive.store(false, Ordering::SeqCst))
    }

    fn alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Closing the input is the spec's shutdown signal. A host process that
    /// ignores it gets SIGTERM, then SIGKILL, for its whole process group (an
    /// `npx` wrapper's node child included). A VM server's relay ends with the
    /// connection, and the server sees EOF on its stdin.
    fn close(&self) {
        drop(lock(&self.writer).take());
        match lock(&self.owner).take() {
            None => {}
            Some(Owner::Socket(stream)) => {
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
            Some(Owner::Child(mut child)) => {
                let deadline = Instant::now() + SHUTDOWN_GRACE;
                let mut signalled = false;
                loop {
                    match child.try_wait() {
                        Ok(Some(_)) | Err(_) => break,
                        Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(25)),
                        Ok(None) if !signalled => {
                            signalled = true;
                            signal_group(child.id(), "TERM");
                            thread::sleep(Duration::from_millis(500));
                        }
                        Ok(None) => {
                            signal_group(child.id(), "KILL");
                            let _ = child.kill();
                            let _ = child.wait();
                            break;
                        }
                    }
                }
            }
        }
        self.alive.store(false, Ordering::SeqCst);
    }
}

impl Drop for LineStream {
    fn drop(&mut self) {
        self.close();
    }
}

/// Signal a whole process group without a libc dependency.
fn signal_group(pid: u32, signal: &str) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args([format!("-{signal}"), "--".into(), format!("-{pid}")])
            .stdin(Piped::null())
            .stdout(Piped::null())
            .stderr(Piped::null())
            .status();
    }
    #[cfg(not(unix))]
    let _ = (pid, signal);
}

// ------------------------------------------------------------------
// HTTP plumbing shared by Streamable HTTP and legacy SSE
// ------------------------------------------------------------------

/// No client-wide timeout: SSE streams stay open indefinitely. Every one-shot
/// request sets its own. Redirects are refused, as elsewhere in s0ma, so
/// configured credentials never follow a redirect to another host.
pub fn http_client() -> Result<HttpClient, String> {
    HttpClient::builder()
        .timeout(None::<Duration>)
        .connect_timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(concat!("s0ma-mcp/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("cannot build HTTP client: {e}"))
}

pub fn header_map(headers: &BTreeMap<String, String>) -> Result<HeaderMap, String> {
    let mut map = HeaderMap::new();
    for (name, value) in headers {
        let header = HeaderName::from_bytes(name.as_bytes()).map_err(|_| format!("invalid header name '{name}'"))?;
        let mut value = HeaderValue::from_str(value).map_err(|_| format!("invalid value for header '{name}'"))?;
        value.set_sensitive(true);
        map.insert(header, value);
    }
    Ok(map)
}

/// Error text never repeats the URL: it may carry a token in its query string,
/// and these messages reach the model and the archive.
fn describe_http_error(error: &reqwest::Error) -> String {
    let mut text = error.to_string();
    if let Some(url) = error.url() {
        text = text.replace(url.as_str(), "<url>");
    }
    if error.is_timeout() {
        "the HTTP request timed out".into()
    } else if error.is_connect() {
        format!("cannot connect: {text}")
    } else {
        text
    }
}

fn read_capped(response: Response, cap: usize) -> std::io::Result<(Vec<u8>, bool)> {
    let mut bytes = Vec::new();
    response.take(cap as u64 + 1).read_to_end(&mut bytes)?;
    let over = bytes.len() > cap;
    bytes.truncate(cap);
    Ok((bytes, over))
}

fn error_body(response: Response) -> String {
    read_capped(response, MAX_ERROR_BODY_BYTES)
        .map(|(bytes, _)| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

fn is_event_stream(response: &Response) -> bool {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.trim_start().to_ascii_lowercase().starts_with("text/event-stream"))
}

/// `Mcp-Name` carries a tool name or resource URI; anything that is not plain
/// printable ASCII goes in the spec's base64 sentinel form.
pub fn mcp_name_header(name: &str) -> String {
    let plain = !name.is_empty()
        && name.bytes().all(|b| (0x20..=0x7e).contains(&b))
        && !name.starts_with(' ')
        && !name.ends_with(' ')
        && !name.starts_with("=?");
    if plain { name.to_owned() } else { format!("=?base64?{}?=", STANDARD.encode(name)) }
}

pub struct SseEvent {
    pub event: String,
    pub data: String,
}

/// Parse an SSE stream, calling `on_event` per event until it returns false,
/// `stop` says so, or the stream ends.
pub fn read_sse<R: BufRead>(
    reader: &mut R,
    stop: &dyn Fn() -> bool,
    mut on_event: impl FnMut(SseEvent) -> bool,
) -> std::io::Result<()> {
    let mut event = String::new();
    let mut data = String::new();
    let mut has_data = false;
    let mut line = Vec::new();
    loop {
        if stop() {
            return Ok(());
        }
        match read_line_capped(reader, &mut line, MAX_MESSAGE_BYTES)? {
            None => return Ok(()),
            Some(false) => {
                // An over-long line poisons the event it belongs to.
                data.clear();
                has_data = false;
                continue;
            }
            Some(true) => {}
        }
        if line.is_empty() {
            if has_data {
                let complete = SseEvent { event: std::mem::take(&mut event), data: std::mem::take(&mut data) };
                if !on_event(complete) {
                    return Ok(());
                }
            }
            event.clear();
            has_data = false;
            continue;
        }
        let text = String::from_utf8_lossy(&line);
        if text.starts_with(':') {
            continue; // comment / keepalive
        }
        let (field, value) = match text.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (text.as_ref(), ""),
        };
        match field {
            "event" => event = value.to_owned(),
            "data" => {
                if data.len() + value.len() > MAX_MESSAGE_BYTES {
                    data.clear();
                    has_data = false;
                    continue;
                }
                if has_data {
                    data.push('\n');
                }
                data.push_str(value);
                has_data = true;
            }
            _ => {}
        }
    }
}

fn is_response_to(message: &Value, id: u64) -> bool {
    match message {
        Value::Array(batch) => batch.iter().any(|m| is_response_to(m, id)),
        Value::Object(fields) => {
            !fields.contains_key("method")
                && fields.get("id").is_some_and(|v| {
                    v.as_u64() == Some(id) || v.as_str().is_some_and(|s| s == id.to_string())
                })
        }
        _ => false,
    }
}

// ------------------------------------------------------------------
// Streamable HTTP
// ------------------------------------------------------------------

pub struct StreamableHttp {
    inner: Arc<HttpInner>,
}

struct HttpInner {
    client: HttpClient,
    url: Url,
    headers: HeaderMap,
    router: Arc<Router>,
    /// Legacy (2025-03-26 .. 2025-11-25) servers may assign one at initialize.
    session: Mutex<Option<String>>,
    version: Mutex<Option<String>>,
    modern: AtomicBool,
    alive: AtomicBool,
}

impl StreamableHttp {
    pub fn new(url: &str, headers: &BTreeMap<String, String>, router: Arc<Router>) -> Result<Self, String> {
        Ok(Self {
            inner: Arc::new(HttpInner {
                client: http_client()?,
                url: Url::parse(url).map_err(|e| format!("invalid url: {e}"))?,
                headers: header_map(headers)?,
                router,
                session: Mutex::new(None),
                version: Mutex::new(None),
                modern: AtomicBool::new(false),
                alive: AtomicBool::new(true),
            }),
        })
    }
}

impl HttpInner {
    fn post(&self, message: &Value, method: Option<&str>, timeout: Duration) -> Result<RequestBuilder, TransportError> {
        let body = serde_json::to_vec(message).map_err(|e| TransportError::Io(e.to_string()))?;
        let mut request = self
            .client
            .post(self.url.clone())
            .headers(self.headers.clone())
            .header(ACCEPT, "application/json, text/event-stream")
            .header(CONTENT_TYPE, "application/json")
            .timeout(timeout)
            .body(body);
        if let Some(version) = lock(&self.version).as_deref() {
            request = request.header("MCP-Protocol-Version", version);
        }
        if let Some(session) = lock(&self.session).as_deref() {
            request = request.header("Mcp-Session-Id", session);
        }
        if self.modern.load(Ordering::SeqCst)
            && let Some(method) = method {
                request = request.header("Mcp-Method", method);
                let params = message.get("params");
                let name = match method {
                    "tools/call" | "prompts/get" => params.and_then(|p| p.get("name")),
                    "resources/read" => params.and_then(|p| p.get("uri")),
                    _ => None,
                };
                if let Some(name) = name.and_then(Value::as_str) {
                    request = request.header("Mcp-Name", mcp_name_header(name));
                }
            }
        Ok(request)
    }

    /// Runs on its own thread: send one request and route everything that
    /// comes back on its response, delivering the reply for `id`.
    fn exchange(self: Arc<Self>, id: u64, request: RequestBuilder) {
        let had_session = lock(&self.session).is_some();
        let response = match request.send() {
            Ok(response) => response,
            Err(error) => {
                self.router.complete(id, Err(TransportError::Io(describe_http_error(&error))));
                return;
            }
        };
        let status = response.status();
        if !self.modern.load(Ordering::SeqCst)
            && let Some(session) = response.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
                *lock(&self.session) = Some(session.to_owned());
            }
        if status == StatusCode::NOT_FOUND && had_session {
            self.alive.store(false, Ordering::SeqCst);
            self.router.log("server answered 404 for our session; it has expired");
            self.router.complete(id, Err(TransportError::SessionExpired));
            return;
        }
        if !status.is_success() {
            let body = error_body(response);
            self.router.complete(id, Err(TransportError::Http { status: status.as_u16(), body }));
            return;
        }

        let mut answered = false;
        if is_event_stream(&response) {
            let mut reader = BufReader::new(response);
            let result = read_sse(&mut reader, &|| false, |event| {
                if !(event.event.is_empty() || event.event == "message") || event.data.trim().is_empty() {
                    return true;
                }
                match serde_json::from_str::<Value>(&event.data) {
                    Ok(message) => answered |= self.route(id, message),
                    Err(_) => self.router.log(&format!("non-JSON SSE data: {}", clip(&event.data, 200))),
                }
                !answered
            });
            if let Err(error) = result
                && !answered {
                    self.router.complete(id, Err(TransportError::Io(format!("response stream failed: {error}"))));
                    return;
                }
        } else {
            match read_capped(response, MAX_MESSAGE_BYTES) {
                Ok((_, true)) => {
                    self.router.complete(id, Err(TransportError::Io("response exceeded the size limit".into())));
                    return;
                }
                Ok((bytes, false)) if bytes.iter().all(u8::is_ascii_whitespace) => {}
                Ok((bytes, false)) => match serde_json::from_slice::<Value>(&bytes) {
                    Ok(message) => answered = self.route(id, message),
                    Err(error) => {
                        self.router.complete(id, Err(TransportError::Io(format!("response is not JSON: {error}"))));
                        return;
                    }
                },
                Err(error) => {
                    self.router.complete(id, Err(TransportError::Io(format!("cannot read response: {error}"))));
                    return;
                }
            }
        }
        if !answered {
            self.router.complete(
                id,
                Err(TransportError::Closed(format!("HTTP {status} carried no reply to the request"))),
            );
        }
    }

    fn route(&self, id: u64, message: Value) -> bool {
        let answers = is_response_to(&message, id);
        for reply in self.router.dispatch(message) {
            // Only legacy servers may ask the client things over HTTP.
            if !self.modern.load(Ordering::SeqCst) {
                let _ = self.post_once(&reply);
            }
        }
        answers
    }

    fn post_once(&self, message: &Value) -> Result<(), TransportError> {
        let response = self
            .post(message, None, POST_TIMEOUT)?
            .send()
            .map_err(|e| TransportError::Io(describe_http_error(&e)))?;
        let status = response.status();
        if status.is_success() {
            Ok(())
        } else {
            Err(TransportError::Http { status: status.as_u16(), body: error_body(response) })
        }
    }
}

impl Transport for StreamableHttp {
    fn kind(&self) -> &'static str {
        "http"
    }

    fn send_request(&self, id: u64, method: &str, message: &Value, timeout: Duration) -> Result<(), TransportError> {
        if !self.alive() {
            return Err(TransportError::SessionExpired);
        }
        let request = self.inner.post(message, Some(method), timeout)?;
        let inner = Arc::clone(&self.inner);
        // A 2026-07-28 request is cancelled by closing its stream. A blocking
        // read cannot be closed from outside, so an abandoned exchange runs
        // out on its own timeout instead; the caller stops waiting at once.
        thread::Builder::new()
            .name("mcp-http".into())
            .spawn(move || inner.exchange(id, request))
            .map(|_| ())
            .map_err(|e| TransportError::Io(e.to_string()))
    }

    fn send_notification(&self, message: &Value) -> Result<(), TransportError> {
        self.inner.post_once(message)
    }

    fn configure(&self, version: Option<&str>, modern: bool) {
        *lock(&self.inner.version) = version.map(str::to_owned);
        self.inner.modern.store(modern, Ordering::SeqCst);
        if modern {
            *lock(&self.inner.session) = None;
        }
    }

    fn alive(&self) -> bool {
        self.inner.alive.load(Ordering::SeqCst)
    }

    fn close(&self) {
        self.inner.alive.store(false, Ordering::SeqCst);
        let session = lock(&self.inner.session).take();
        if let Some(session) = session {
            // Best effort: tell a legacy server it can drop the session.
            let _ = self
                .inner
                .client
                .delete(self.inner.url.clone())
                .headers(self.inner.headers.clone())
                .header("Mcp-Session-Id", session)
                .timeout(Duration::from_secs(3))
                .send();
        }
    }
}

// ------------------------------------------------------------------
// HTTP+SSE (2024-11-05, deprecated)
// ------------------------------------------------------------------

pub struct LegacySse {
    inner: Arc<SseInner>,
}

struct SseInner {
    client: HttpClient,
    headers: HeaderMap,
    router: Arc<Router>,
    endpoint: Mutex<EndpointState>,
    ready: Condvar,
    alive: AtomicBool,
}

enum EndpointState {
    Waiting,
    Ready(Url),
    Failed(String),
}

impl LegacySse {
    pub fn connect(
        url: &str,
        headers: &BTreeMap<String, String>,
        router: Arc<Router>,
        timeout: Duration,
    ) -> Result<Self, String> {
        let base = Url::parse(url).map_err(|e| format!("invalid url: {e}"))?;
        let inner = Arc::new(SseInner {
            client: http_client()?,
            headers: header_map(headers)?,
            router,
            endpoint: Mutex::new(EndpointState::Waiting),
            ready: Condvar::new(),
            alive: AtomicBool::new(true),
        });
        {
            let inner = Arc::clone(&inner);
            thread::Builder::new()
                .name("mcp-sse".into())
                .spawn(move || inner.listen(base))
                .map_err(|e| e.to_string())?;
        }
        let deadline = Instant::now() + timeout;
        let mut state = lock(&inner.endpoint);
        loop {
            match &*state {
                EndpointState::Ready(_) => break,
                EndpointState::Failed(why) => return Err(why.clone()),
                EndpointState::Waiting => {
                    let now = Instant::now();
                    if now >= deadline {
                        inner.alive.store(false, Ordering::SeqCst);
                        return Err("SSE server sent no endpoint event in time".into());
                    }
                    state = inner
                        .ready
                        .wait_timeout(state, deadline - now)
                        .unwrap_or_else(|p| p.into_inner())
                        .0;
                }
            }
        }
        drop(state);
        Ok(Self { inner })
    }
}

impl SseInner {
    fn set_endpoint(&self, state: EndpointState) {
        let mut endpoint = lock(&self.endpoint);
        if matches!(*endpoint, EndpointState::Waiting) {
            *endpoint = state;
        }
        self.ready.notify_all();
    }

    fn listen(self: Arc<Self>, base: Url) {
        let response = self
            .client
            .get(base.clone())
            .headers(self.headers.clone())
            .header(ACCEPT, "text/event-stream")
            .send();
        let response = match response {
            Ok(response) if response.status().is_success() && is_event_stream(&response) => response,
            Ok(response) => {
                let status = response.status();
                self.set_endpoint(EndpointState::Failed(format!(
                    "SSE GET answered HTTP {status} without an event stream"
                )));
                self.alive.store(false, Ordering::SeqCst);
                return;
            }
            Err(error) => {
                self.set_endpoint(EndpointState::Failed(describe_http_error(&error)));
                self.alive.store(false, Ordering::SeqCst);
                return;
            }
        };
        let mut reader = BufReader::new(response);
        let closed = || !self.alive.load(Ordering::SeqCst);
        let _ = read_sse(&mut reader, &closed, |event| {
            match event.event.as_str() {
                "endpoint" => match base.join(event.data.trim()) {
                    // Only the origin s0ma was configured with gets the
                    // configured headers; an endpoint elsewhere is refused.
                    Ok(url) if url.origin() == base.origin() => self.set_endpoint(EndpointState::Ready(url)),
                    Ok(url) => self.set_endpoint(EndpointState::Failed(format!(
                        "SSE endpoint points to another origin ({}); refused",
                        url.origin().ascii_serialization()
                    ))),
                    Err(error) => self.set_endpoint(EndpointState::Failed(format!("bad SSE endpoint: {error}"))),
                },
                "" | "message" => match serde_json::from_str::<Value>(&event.data) {
                    Ok(message) => {
                        for reply in self.router.dispatch(message) {
                            let _ = self.post(&reply);
                        }
                    }
                    Err(_) => self.router.log(&format!("non-JSON SSE data: {}", clip(&event.data, 200))),
                },
                _ => {}
            }
            true
        });
        self.alive.store(false, Ordering::SeqCst);
        self.set_endpoint(EndpointState::Failed("SSE stream closed before an endpoint event".into()));
        self.router.log("SSE stream closed");
        self.router.fail_all(TransportError::Closed("the SSE stream closed".into()));
    }

    fn post(&self, message: &Value) -> Result<(), TransportError> {
        let url = match &*lock(&self.endpoint) {
            EndpointState::Ready(url) => url.clone(),
            _ => return Err(TransportError::Closed("no SSE endpoint".into())),
        };
        let body = serde_json::to_vec(message).map_err(|e| TransportError::Io(e.to_string()))?;
        let response = self
            .client
            .post(url)
            .headers(self.headers.clone())
            .header(CONTENT_TYPE, "application/json")
            .timeout(POST_TIMEOUT)
            .body(body)
            .send()
            .map_err(|e| TransportError::Io(describe_http_error(&e)))?;
        let status = response.status();
        if status.is_success() {
            Ok(())
        } else {
            Err(TransportError::Http { status: status.as_u16(), body: error_body(response) })
        }
    }
}

impl Transport for LegacySse {
    fn kind(&self) -> &'static str {
        "sse"
    }

    fn send_request(&self, _id: u64, _method: &str, message: &Value, _timeout: Duration) -> Result<(), TransportError> {
        if !self.alive() {
            return Err(TransportError::Closed("the SSE stream is closed".into()));
        }
        self.inner.post(message)
    }

    fn send_notification(&self, message: &Value) -> Result<(), TransportError> {
        self.inner.post(message)
    }

    fn alive(&self) -> bool {
        self.inner.alive.load(Ordering::SeqCst)
    }

    /// The listener notices on its next line (servers send keepalives).
    fn close(&self) {
        self.inner.alive.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn capped_lines_split_strip_cr_and_discard_overflow() {
        let mut reader = Cursor::new(b"short\r\nthis line is far too long\nok\nlast".to_vec());
        let mut line = Vec::new();
        assert_eq!(read_line_capped(&mut reader, &mut line, 10).unwrap(), Some(true));
        assert_eq!(line, b"short");
        assert_eq!(read_line_capped(&mut reader, &mut line, 10).unwrap(), Some(false));
        assert_eq!(read_line_capped(&mut reader, &mut line, 10).unwrap(), Some(true));
        assert_eq!(line, b"ok");
        assert_eq!(read_line_capped(&mut reader, &mut line, 10).unwrap(), Some(true));
        assert_eq!(line, b"last");
        assert_eq!(read_line_capped(&mut reader, &mut line, 10).unwrap(), None);
    }

    #[test]
    fn sse_parser_joins_data_lines_and_skips_comments() {
        let stream = ": keepalive\nevent: endpoint\ndata: /messages?session=1\n\n\
                      data: {\"a\":\ndata: 1}\n\nevent: message\r\ndata: {\"b\":2}\r\n\r\n";
        let mut events = Vec::new();
        read_sse(&mut Cursor::new(stream.as_bytes()), &|| false, |e| {
            events.push((e.event, e.data));
            true
        })
        .unwrap();
        assert_eq!(
            events,
            [
                ("endpoint".to_string(), "/messages?session=1".to_string()),
                (String::new(), "{\"a\":\n1}".to_string()),
                ("message".to_string(), "{\"b\":2}".to_string()),
            ]
        );
    }

    #[test]
    fn router_answers_ping_refuses_other_requests_and_tracks_list_changes() {
        let router = Router::new(None);
        let receiver = router.register(7);
        assert!(router.dispatch(json!({"jsonrpc":"2.0","id":7,"result":{"ok":true}})).is_empty());
        assert_eq!(receiver.recv().unwrap().unwrap()["result"]["ok"], true);

        let replies = router.dispatch(json!([
            {"jsonrpc":"2.0","id":"s1","method":"ping"},
            {"jsonrpc":"2.0","id":2,"method":"sampling/createMessage","params":{}},
            {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}
        ]));
        assert_eq!(replies[0], json!({"jsonrpc":"2.0","id":"s1","result":{}}));
        assert_eq!(replies[1]["error"]["code"], -32601);
        assert_eq!(replies.len(), 2);
        assert!(router.stale());

        let pending = router.register(9);
        router.fail_all(TransportError::Closed("gone".into()));
        assert_eq!(pending.recv().unwrap(), Err(TransportError::Closed("gone".into())));
    }

    #[test]
    fn response_matching_accepts_numeric_and_string_ids_only_for_responses() {
        assert!(is_response_to(&json!({"id": 3, "result": {}}), 3));
        assert!(is_response_to(&json!({"id": "3", "error": {}}), 3));
        assert!(!is_response_to(&json!({"id": 3, "method": "ping"}), 3));
        assert!(is_response_to(&json!([{"method":"x"}, {"id": 3, "result": {}}]), 3));
    }

    #[test]
    fn mcp_name_header_uses_base64_sentinel_for_unsafe_values() {
        assert_eq!(mcp_name_header("get_weather"), "get_weather");
        assert_eq!(mcp_name_header("file:///a b.txt"), "file:///a b.txt");
        assert_eq!(mcp_name_header("naïve"), format!("=?base64?{}?=", STANDARD.encode("naïve")));
        assert!(mcp_name_header(" padded").starts_with("=?base64?"));
    }

    #[cfg(unix)]
    #[test]
    fn stdio_child_gets_a_scrubbed_environment_and_exit_fails_pending_requests() {
        // CARGO_MANIFEST_DIR is in the test process's environment (cargo sets
        // it), so it stands in for a secret that must not reach the server.
        let router = Router::new(None);
        let env = BTreeMap::from([("GIVEN".to_string(), "yes".to_string())]);
        let args = vec![
            "-c".to_string(),
            // Answer the first request with what the child can see, then exit
            // after reading the second without answering it.
            r#"read line; printf '{"jsonrpc":"2.0","id":1,"result":{"cargo":"%s","given":"%s"}}\n' "${CARGO_MANIFEST_DIR:-unset}" "$GIVEN"; read line"#.to_string(),
        ];
        let stdio = LineStream::spawn(
            StdioSpec { command: "sh", args: &args, env: &env, cwd: None, inherit_env: false },
            Arc::clone(&router),
        )
        .unwrap();
        let first = router.register(1);
        stdio.send_request(1, "x", &json!({"jsonrpc":"2.0","id":1,"method":"x"}), Duration::from_secs(5)).unwrap();
        let reply = first.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        assert_eq!(reply["result"]["cargo"], "unset");
        assert_eq!(reply["result"]["given"], "yes");

        let second = router.register(2);
        stdio.send_request(2, "x", &json!({"jsonrpc":"2.0","id":2,"method":"x"}), Duration::from_secs(5)).unwrap();
        assert!(matches!(second.recv_timeout(Duration::from_secs(5)).unwrap(), Err(TransportError::Closed(_))));
        assert!(!stdio.alive());
        assert!(stdio.send_request(3, "x", &json!({}), Duration::from_secs(1)).is_err());
    }

    #[test]
    fn tcp_line_stream_waits_for_a_late_listener_and_reports_its_closing() {
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let address: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        // The relay comes up a little after the client starts trying.
        let server = thread::spawn(move || {
            thread::sleep(Duration::from_millis(300));
            let listener = std::net::TcpListener::bind(address).unwrap();
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            let mut writer = stream;
            writeln!(writer, "{}", json!({"jsonrpc":"2.0","id":request["id"],"result":{"pong":true}})).unwrap();
            // Then the server goes away.
        });
        let router = Router::new(None);
        let stream = LineStream::connect(
            address,
            Instant::now() + Duration::from_secs(5),
            Arc::clone(&router),
            "the server on the machine closed the connection".into(),
        )
        .unwrap();
        assert_eq!(stream.kind(), "vm");
        let first = router.register(1);
        stream.send_request(1, "ping", &json!({"jsonrpc":"2.0","id":1,"method":"ping"}), Duration::from_secs(5)).unwrap();
        assert_eq!(first.recv_timeout(Duration::from_secs(5)).unwrap().unwrap()["result"]["pong"], true);
        server.join().unwrap();
        // Either the write fails (the close was already seen) or the pending
        // request is failed with the stream's closing reason.
        let second = router.register(2);
        match stream.send_request(2, "ping", &json!({"jsonrpc":"2.0","id":2,"method":"ping"}), Duration::from_secs(5)) {
            Err(error) => assert!(matches!(error, TransportError::Closed(_)), "{error}"),
            Ok(()) => assert_eq!(
                second.recv_timeout(Duration::from_secs(5)).unwrap(),
                Err(TransportError::Closed("the server on the machine closed the connection".into()))
            ),
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while stream.alive() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(!stream.alive());
    }

    #[test]
    fn tcp_connect_gives_up_at_the_deadline() {
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let started = Instant::now();
        let error = LineStream::connect(
            format!("127.0.0.1:{port}").parse().unwrap(),
            Instant::now() + Duration::from_millis(400),
            Router::new(None),
            String::new(),
        )
        .err()
        .unwrap();
        assert!(error.contains("cannot connect"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(3));
    }
}
