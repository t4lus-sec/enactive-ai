//! One MCP connection: version negotiation, then requests with deadlines.
//!
//! # Negotiation
//!
//! MCP has two eras. 2026-07-28 is stateless: no handshake, and every request
//! carries its protocol version, client identity and capabilities in `_meta`.
//! Everything from 2024-11-05 through 2025-11-25 opens with an `initialize`
//! handshake instead. Servers of both kinds are in use, so `Protocol::Auto`
//! follows the spec's backward-compatibility rule: probe with
//! `server/discover`; a `DiscoverResult` means a modern server, a recognized
//! unsupported-version error means a modern server that wants another version,
//! and anything else means a legacy server, so fall back to `initialize`.
//!
//! On stdio a silent probe is ambiguous — a legacy server ignoring an unknown
//! pre-initialize request looks the same as a modern one still starting up
//! (`npx` fetching a package). So after a short wait the client sends
//! `initialize` as well and takes whichever answers first.
use super::{
    config::{Endpoint, HttpTransport, Protocol, ServerConfig},
    guest,
    transport::{LegacySse, LineStream, Reply, Router, StdioSpec, StreamableHttp, Transport, TransportError, clip},
};
use serde_json::{Map, Value, json};
use std::{
    fmt,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{Receiver, RecvTimeoutError},
    },
    thread,
    time::{Duration, Instant},
};

pub const MODERN_VERSION: &str = "2026-07-28";
/// Handshake-era versions, newest first; the first is what `initialize` asks for.
pub const LEGACY_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const META_PROTOCOL_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
const META_CLIENT_CAPABILITIES: &str = "io.modelcontextprotocol/clientCapabilities";
const META_CLIENT_INFO: &str = "io.modelcontextprotocol/clientInfo";
const META_SERVER_INFO: &str = "io.modelcontextprotocol/serverInfo";
/// UnsupportedProtocolVersionError: -32022 in 2026-07-28, -32004 in its drafts.
const UNSUPPORTED_VERSION_CODES: &[i64] = &[-32022, -32004];

/// How long a stdio probe may stay silent before `initialize` is sent too.
const PROBE_PATIENCE: Duration = if cfg!(test) { Duration::from_millis(300) } else { Duration::from_secs(5) };
const LIST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_LIST_PAGES: usize = 50;
/// Extra room the transport keeps an exchange open past the caller's deadline,
/// so a reply racing the deadline is not cut off by the HTTP layer first.
const TRANSPORT_SLACK: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Era {
    Modern,
    Legacy,
}

#[derive(Debug)]
pub enum CallError {
    Rpc { code: i64, message: String },
    /// The request never reached the server: safe to retry elsewhere.
    NotSent(TransportError),
    Transport(TransportError),
    Timeout(Duration),
    Interrupted,
}

impl CallError {
    /// The server provably did not process the request: it was never sent, or
    /// a legacy server rejected it for an unknown session before running it.
    pub fn retryable(&self) -> bool {
        matches!(self, Self::NotSent(_) | Self::Transport(TransportError::SessionExpired))
    }
}

impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rpc { code, message } => write!(f, "the server returned error {code}: {message}"),
            Self::NotSent(error) => write!(f, "could not send the request: {error}"),
            Self::Transport(error) => write!(f, "{error}"),
            Self::Timeout(after) => write!(f, "no answer within {}s", after.as_secs()),
            Self::Interrupted => write!(f, "interrupted by the operator"),
        }
    }
}

struct HandshakeError {
    message: String,
    /// The endpoint rejected Streamable HTTP outright; it may be an old
    /// HTTP+SSE server.
    try_sse: bool,
}

impl HandshakeError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), try_sse: false }
    }
}

enum Probe {
    Modern(Value),
    Unsupported(Vec<String>),
    Legacy(String),
}

pub struct Client {
    transport: Box<dyn Transport>,
    router: Arc<Router>,
    next_id: AtomicU64,
    era: Era,
    version: String,
    server_info: Option<String>,
    capabilities: Value,
    instructions: Option<String>,
}

impl Client {
    pub fn connect(config: &ServerConfig, log_dir: Option<&Path>) -> Result<Self, String> {
        let router = Router::new(log_dir.map(|dir| dir.join(format!("{}.log", config.name))));
        router.log(&format!("connecting over {}", config.transport_label()));
        let deadline = Instant::now() + config.startup_timeout;
        let result = match &config.endpoint {
            Endpoint::Host { command, args, env, cwd, inherit_env } => {
                let spec = StdioSpec { command, args, env, cwd: cwd.as_deref(), inherit_env: *inherit_env };
                LineStream::spawn(spec, Arc::clone(&router)).and_then(|stdio| {
                    Self::handshake(Box::new(stdio), &router, config.protocol, deadline).map_err(|e| e.message)
                })
            }
            Endpoint::Vm { command, args, env, cwd, address, port } => {
                let spec = guest::Launch {
                    name: &config.name,
                    command,
                    args,
                    env,
                    cwd: cwd.as_deref(),
                    address: *address,
                    port: *port,
                };
                let ended = format!(
                    "the server on the machine closed the connection (its log: {})",
                    guest::log_location(&config.name)
                );
                // A failed launch already carries the VM log's tail; a failure
                // after it (connect, handshake) fetches it.
                guest::launch(&spec, remaining(deadline)).and_then(|at| {
                    router.log(&format!("relay started on the machine at {at}"));
                    LineStream::connect(at, deadline, Arc::clone(&router), ended)
                        .and_then(|stream| {
                            Self::handshake(Box::new(stream), &router, config.protocol, deadline)
                                .map_err(|e| e.message)
                        })
                        .map_err(|error| match guest::log_tail(&config.name) {
                            Some(tail) => format!("{error}; machine log: {tail}"),
                            None => error,
                        })
                })
            }
            Endpoint::Http { url, headers, transport: HttpTransport::Sse } => {
                Self::over_sse(url, headers, &router, deadline)
            }
            Endpoint::Http { url, headers, transport } => StreamableHttp::new(url, headers, Arc::clone(&router))
                .and_then(|http| match Self::handshake(Box::new(http), &router, config.protocol, deadline) {
                    Ok(client) => Ok(client),
                    Err(error) if error.try_sse && *transport == HttpTransport::Auto => {
                        router.log(&format!("{}; trying the HTTP+SSE transport", error.message));
                        Self::over_sse(url, headers, &router, deadline)
                            .map_err(|sse| format!("{} (as HTTP+SSE: {sse})", error.message))
                    }
                    Err(error) => Err(error.message),
                }),
        };
        match result {
            Ok(client) => {
                router.log(&format!(
                    "connected: protocol {}, server {}",
                    client.version,
                    client.server_info.as_deref().unwrap_or("unidentified")
                ));
                Ok(client)
            }
            Err(mut error) => {
                let tail = router.stderr_tail();
                if !tail.is_empty() && !error.contains("stderr:") {
                    error.push_str(&format!("; stderr: {tail}"));
                }
                router.log(&format!("connect failed: {error}"));
                Err(error)
            }
        }
    }

    fn over_sse(
        url: &str,
        headers: &std::collections::BTreeMap<String, String>,
        router: &Arc<Router>,
        deadline: Instant,
    ) -> Result<Self, String> {
        let sse = LegacySse::connect(url, headers, Arc::clone(router), remaining(deadline))?;
        Self::handshake(Box::new(sse), router, Protocol::Legacy, deadline).map_err(|e| e.message)
    }

    fn handshake(
        transport: Box<dyn Transport>,
        router: &Arc<Router>,
        protocol: Protocol,
        deadline: Instant,
    ) -> Result<Self, HandshakeError> {
        let mut client = Self {
            transport,
            router: Arc::clone(router),
            next_id: AtomicU64::new(1),
            era: Era::Legacy,
            version: String::new(),
            server_info: None,
            capabilities: json!({}),
            instructions: None,
        };
        match protocol {
            Protocol::Legacy => client.initialize(deadline)?,
            Protocol::Auto => client.negotiate(deadline)?,
            Protocol::Modern => {
                let (id, receiver) = client.send_probe(deadline)?;
                let reply = recv_until(&receiver, deadline);
                client.router.forget(id);
                match reply.map(classify_probe) {
                    Some(Probe::Modern(result)) => client.adopt_modern(&result),
                    Some(Probe::Unsupported(versions)) => return Err(unsupported(&versions)),
                    Some(Probe::Legacy(why)) => {
                        return Err(HandshakeError::new(format!("server/discover failed ({why}); is this a {MODERN_VERSION} server?")));
                    }
                    None => return Err(HandshakeError::new("no answer to server/discover in time")),
                }
            }
        }
        Ok(client)
    }

    fn take_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    fn send_probe(&self, deadline: Instant) -> Result<(u64, Receiver<Reply>), HandshakeError> {
        self.transport.configure(Some(MODERN_VERSION), true);
        let id = self.take_id();
        let receiver = self.router.register(id);
        let message = json!({
            "jsonrpc": "2.0", "id": id, "method": "server/discover",
            "params": with_meta(json!({}), MODERN_VERSION),
        });
        if let Err(error) = self.transport.send_request(id, "server/discover", &message, remaining(deadline)) {
            self.router.forget(id);
            return Err(HandshakeError::new(error.to_string()));
        }
        Ok((id, receiver))
    }

    fn negotiate(&mut self, deadline: Instant) -> Result<(), HandshakeError> {
        let (probe_id, probe) = self.send_probe(deadline)?;
        // A silent probe is ambiguous only on a stream (stdio or the VM
        // relay); over HTTP every request gets a status.
        let patience = if self.transport.kind() == "http" { remaining(deadline) } else { PROBE_PATIENCE };
        match recv_until(&probe, deadline.min(Instant::now() + patience)) {
            Some(Err(TransportError::Http { status, body })) => {
                self.router.forget(probe_id);
                match classify_probe(Err(TransportError::Http { status, body })) {
                    Probe::Unsupported(versions) => self.after_unsupported(&versions, deadline),
                    Probe::Modern(_) | Probe::Legacy(_) => {
                        self.router.log(&format!("server/discover answered HTTP {status}; using initialize"));
                        self.initialize(deadline)
                    }
                }
            }
            // The connection itself failed; initialize would only fail the same way.
            Some(Err(error)) => {
                self.router.forget(probe_id);
                Err(HandshakeError::new(error.to_string()))
            }
            Some(reply) => match classify_probe(reply) {
                Probe::Modern(result) => {
                    self.adopt_modern(&result);
                    Ok(())
                }
                Probe::Unsupported(versions) => self.after_unsupported(&versions, deadline),
                Probe::Legacy(why) => {
                    self.router.log(&format!("server/discover: {why}; using initialize"));
                    self.initialize(deadline)
                }
            },
            None => self.race(probe_id, probe, deadline),
        }
    }

    /// The probe is still unanswered: send `initialize` too and keep whichever
    /// era answers first.
    fn race(&mut self, probe_id: u64, probe: Receiver<Reply>, deadline: Instant) -> Result<(), HandshakeError> {
        self.router.log("no answer to server/discover yet; sending initialize alongside it");
        self.transport.configure(None, false);
        let init_id = self.take_id();
        let init = self.router.register(init_id);
        if let Err(error) = self.transport.send_request(init_id, "initialize", &initialize_message(init_id), remaining(deadline)) {
            self.router.forget(probe_id);
            self.router.forget(init_id);
            return Err(HandshakeError::new(error.to_string()));
        }
        let mut probe_open = true;
        loop {
            if probe_open {
                match probe.try_recv() {
                    Ok(reply) => {
                        probe_open = false;
                        if let Probe::Modern(result) = classify_probe(reply) {
                            self.router.forget(init_id);
                            self.adopt_modern(&result);
                            return Ok(());
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => probe_open = false,
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                }
            }
            match init.try_recv() {
                Ok(reply) => {
                    self.router.forget(probe_id);
                    return self.finish_initialize(reply);
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.router.forget(probe_id);
                    return Err(HandshakeError::new("the connection closed during initialize"));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if Instant::now() >= deadline {
                self.router.forget(probe_id);
                self.router.forget(init_id);
                return Err(HandshakeError::new("no answer to server/discover or initialize before the startup timeout"));
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn after_unsupported(&mut self, versions: &[String], deadline: Instant) -> Result<(), HandshakeError> {
        // A modern server that also lists a handshake-era version we speak
        // still accepts `initialize`.
        if versions.iter().any(|v| LEGACY_VERSIONS.contains(&v.as_str())) {
            self.router.log(&format!("server supports {}; using initialize", versions.join(", ")));
            self.initialize(deadline)
        } else {
            Err(unsupported(versions))
        }
    }

    fn initialize(&mut self, deadline: Instant) -> Result<(), HandshakeError> {
        self.transport.configure(None, false);
        let id = self.take_id();
        let receiver = self.router.register(id);
        if let Err(error) = self.transport.send_request(id, "initialize", &initialize_message(id), remaining(deadline)) {
            self.router.forget(id);
            return Err(HandshakeError {
                try_sse: matches!(error, TransportError::Http { status: 400 | 404 | 405, .. }),
                message: format!("initialize failed: {error}"),
            });
        }
        match recv_until(&receiver, deadline) {
            Some(reply) => self.finish_initialize(reply),
            None => {
                self.router.forget(id);
                Err(HandshakeError::new("no answer to initialize before the startup timeout"))
            }
        }
    }

    fn finish_initialize(&mut self, reply: Reply) -> Result<(), HandshakeError> {
        let response = reply.map_err(|error| HandshakeError {
            try_sse: matches!(error, TransportError::Http { status: 400 | 404 | 405, .. }),
            message: format!("initialize failed: {error}"),
        })?;
        if let Some(error) = response.get("error") {
            return Err(HandshakeError::new(format!("initialize refused: {}", rpc_text(error))));
        }
        let result = response.get("result").cloned().unwrap_or(Value::Null);
        let version = result.get("protocolVersion").and_then(Value::as_str).unwrap_or_default();
        if !LEGACY_VERSIONS.contains(&version) {
            return Err(HandshakeError::new(format!(
                "server chose protocol version '{version}', which s0ma does not speak ({})",
                LEGACY_VERSIONS.join(", ")
            )));
        }
        self.era = Era::Legacy;
        self.version = version.to_owned();
        self.capabilities = result.get("capabilities").cloned().unwrap_or_else(|| json!({}));
        self.instructions = non_empty(result.get("instructions"));
        self.server_info = describe_info(result.get("serverInfo"));
        self.transport.configure(Some(version), false);
        self.transport
            .send_notification(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))
            .map_err(|error| HandshakeError::new(format!("notifications/initialized failed: {error}")))
    }

    fn adopt_modern(&mut self, result: &Value) {
        self.era = Era::Modern;
        self.version = MODERN_VERSION.to_owned();
        self.capabilities = result.get("capabilities").cloned().unwrap_or_else(|| json!({}));
        self.instructions = non_empty(result.get("instructions"));
        self.server_info = describe_info(result.get("_meta").and_then(|m| m.get(META_SERVER_INFO)));
        self.transport.configure(Some(MODERN_VERSION), true);
    }

    pub fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
        interrupt: Option<&AtomicBool>,
    ) -> Result<Value, CallError> {
        let id = self.take_id();
        let params = match self.era {
            Era::Modern => with_meta(params, &self.version),
            Era::Legacy => params,
        };
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let receiver = self.router.register(id);
        if let Err(error) = self.transport.send_request(id, method, &message, timeout + TRANSPORT_SLACK) {
            self.router.forget(id);
            return Err(CallError::NotSent(error));
        }
        let deadline = Instant::now() + timeout;
        loop {
            match receiver.recv_timeout(remaining(deadline).min(POLL)) {
                Ok(reply) => return reply.map_err(CallError::Transport).and_then(result_of),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(CallError::Transport(TransportError::Closed("the request was dropped".into())));
                }
                Err(RecvTimeoutError::Timeout) => {
                    // The operator's /kill stops this request alone; an
                    // interrupt (Ctrl+C) is on its way to ending the turn.
                    let killed = crate::kill::requested();
                    if killed || interrupt.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
                        self.cancel(id, if killed { "stopped by the operator" } else { "interrupted by the operator" });
                        return Err(CallError::Interrupted);
                    }
                    if Instant::now() >= deadline {
                        self.cancel(id, "timed out");
                        return Err(CallError::Timeout(timeout));
                    }
                }
            }
        }
    }

    /// Over HTTP in the 2026-07-28 era, cancellation is closing the response
    /// stream, which abandoning the exchange does; everywhere else the server
    /// is told.
    fn cancel(&self, id: u64, reason: &str) {
        self.router.forget(id);
        self.router.log(&format!("cancelling request {id}: {reason}"));
        if self.era == Era::Modern && self.transport.kind() == "http" {
            return;
        }
        let _ = self.transport.send_notification(&json!({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": id, "reason": reason},
        }));
    }

    /// Every tool, following `nextCursor`.
    pub fn list_tools(&self) -> Result<Vec<Value>, CallError> {
        if !self.has("tools") {
            return Ok(Vec::new());
        }
        self.router.clear_stale();
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_LIST_PAGES {
            let params = match &cursor {
                Some(cursor) => json!({"cursor": cursor}),
                None => json!({}),
            };
            let result = self.request("tools/list", params, LIST_TIMEOUT, None)?;
            if let Some(page) = result.get("tools").and_then(Value::as_array) {
                tools.extend(page.iter().cloned());
            }
            cursor = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .filter(|c| !c.is_empty())
                .map(str::to_owned);
            if cursor.is_none() {
                break;
            }
        }
        Ok(tools)
    }

    pub fn call_tool(
        &self,
        name: &str,
        arguments: Value,
        timeout: Duration,
        interrupt: &AtomicBool,
    ) -> Result<Value, CallError> {
        self.request("tools/call", json!({"name": name, "arguments": arguments}), timeout, Some(interrupt))
    }

    pub fn list_resources(&self, cursor: Option<&str>, interrupt: &AtomicBool) -> Result<Value, CallError> {
        let params = cursor.map_or_else(|| json!({}), |c| json!({"cursor": c}));
        self.request("resources/list", params, LIST_TIMEOUT, Some(interrupt))
    }

    pub fn read_resource(&self, uri: &str, timeout: Duration, interrupt: &AtomicBool) -> Result<Value, CallError> {
        self.request("resources/read", json!({"uri": uri}), timeout, Some(interrupt))
    }

    pub fn has(&self, capability: &str) -> bool {
        self.capabilities.get(capability).is_some_and(Value::is_object)
    }

    pub fn alive(&self) -> bool {
        self.transport.alive()
    }

    pub fn stale(&self) -> bool {
        self.router.stale()
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn transport_kind(&self) -> &'static str {
        self.transport.kind()
    }

    /// For people: how the connection is carried.
    pub fn transport_label(&self) -> &'static str {
        match self.transport.kind() {
            "vm" => "stdio relayed from the machine",
            "stdio" => "stdio",
            "sse" => "HTTP+SSE",
            _ => "http",
        }
    }

    pub fn server_info(&self) -> Option<&str> {
        self.server_info.as_deref()
    }

    pub fn instructions(&self) -> Option<&str> {
        self.instructions.as_deref()
    }

    pub fn log(&self, line: &str) {
        self.router.log(line);
    }

    pub fn close(&self) {
        self.router.log("closing");
        self.transport.close();
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.transport.close();
    }
}

fn remaining(deadline: Instant) -> Duration {
    deadline.saturating_duration_since(Instant::now())
}

/// The reply, or `None` at the deadline or when the call is killed.
fn recv_until(receiver: &Receiver<Reply>, deadline: Instant) -> Option<Reply> {
    loop {
        match receiver.recv_timeout(remaining(deadline).min(POLL)) {
            Ok(reply) => return Some(reply),
            Err(RecvTimeoutError::Disconnected) => {
                return Some(Err(TransportError::Closed("the connection closed".into())));
            }
            Err(RecvTimeoutError::Timeout) => {
                if Instant::now() >= deadline || crate::kill::requested() {
                    return None;
                }
            }
        }
    }
}

fn client_info() -> Value {
    json!({"name": "s0ma", "version": env!("CARGO_PKG_VERSION")})
}

fn initialize_message(id: u64) -> Value {
    json!({
        "jsonrpc": "2.0", "id": id, "method": "initialize",
        "params": {"protocolVersion": LEGACY_VERSIONS[0], "capabilities": {}, "clientInfo": client_info()},
    })
}

/// Stamp the 2026-07-28 per-request metadata, keeping any `_meta` the
/// caller already set.
fn with_meta(params: Value, version: &str) -> Value {
    let mut params = match params {
        Value::Object(map) => map,
        Value::Null => Map::new(),
        other => return other,
    };
    let meta = params.entry("_meta").or_insert_with(|| json!({}));
    if let Some(meta) = meta.as_object_mut() {
        meta.insert(META_PROTOCOL_VERSION.into(), json!(version));
        meta.insert(META_CLIENT_CAPABILITIES.into(), json!({}));
        meta.insert(META_CLIENT_INFO.into(), client_info());
    }
    Value::Object(params)
}

fn classify_probe(reply: Reply) -> Probe {
    match reply {
        Err(TransportError::Http { status, body }) => {
            // A modern server reports version trouble as a JSON-RPC error body.
            let versions = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| v.get("error").and_then(unsupported_versions));
            match versions {
                Some(versions) => Probe::Unsupported(versions),
                None => Probe::Legacy(format!("HTTP {status}")),
            }
        }
        Err(error) => Probe::Legacy(error.to_string()),
        Ok(response) => {
            if let Some(error) = response.get("error") {
                return match unsupported_versions(error) {
                    Some(versions) => Probe::Unsupported(versions),
                    None => Probe::Legacy(format!("error {}", rpc_text(error))),
                };
            }
            let result = response.get("result").cloned().unwrap_or(Value::Null);
            match result.get("supportedVersions").and_then(Value::as_array) {
                Some(versions) if versions.iter().any(|v| v.as_str() == Some(MODERN_VERSION)) => {
                    Probe::Modern(result)
                }
                Some(versions) => {
                    Probe::Unsupported(versions.iter().filter_map(Value::as_str).map(str::to_owned).collect())
                }
                None => Probe::Legacy("the reply is not a DiscoverResult".into()),
            }
        }
    }
}

fn unsupported_versions(error: &Value) -> Option<Vec<String>> {
    let code = error.get("code")?.as_i64()?;
    if !UNSUPPORTED_VERSION_CODES.contains(&code) {
        return None;
    }
    let data = error.get("data")?;
    let list = data.get("supported").or_else(|| data.get("supportedVersions"))?.as_array()?;
    Some(list.iter().filter_map(Value::as_str).map(str::to_owned).collect())
}

fn unsupported(versions: &[String]) -> HandshakeError {
    HandshakeError::new(format!(
        "server supports protocol versions [{}]; s0ma speaks {MODERN_VERSION} and {}",
        versions.join(", "),
        LEGACY_VERSIONS.join(", ")
    ))
}

fn result_of(response: Value) -> Result<Value, CallError> {
    if let Some(error) = response.get("error") {
        return Err(CallError::Rpc {
            code: error.get("code").and_then(Value::as_i64).unwrap_or(0),
            message: error.get("message").and_then(Value::as_str).map_or_else(String::new, |m| clip(m, 2000)),
        });
    }
    Ok(response.get("result").cloned().unwrap_or(Value::Null))
}

fn rpc_text(error: &Value) -> String {
    format!(
        "{} {}",
        error.get("code").and_then(Value::as_i64).unwrap_or(0),
        clip(error.get("message").and_then(Value::as_str).unwrap_or(""), 300)
    )
}

fn non_empty(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned)
}

fn describe_info(info: Option<&Value>) -> Option<String> {
    let info = info?;
    let name = info.get("name").and_then(Value::as_str)?;
    Some(match info.get("version").and_then(Value::as_str) {
        Some(version) => format!("{} {}", clip(name, 60), clip(version, 30)),
        None => clip(name, 60),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_classification_follows_the_backward_compatibility_rules() {
        let modern = json!({"jsonrpc":"2.0","id":1,"result":{"resultType":"complete",
            "supportedVersions":["2026-07-28"],"capabilities":{"tools":{}}}});
        assert!(matches!(classify_probe(Ok(modern)), Probe::Modern(_)));

        let wrong_version = json!({"jsonrpc":"2.0","id":1,"error":{"code":-32022,
            "message":"Unsupported protocol version","data":{"supported":["2027-01-01"]}}});
        assert!(matches!(classify_probe(Ok(wrong_version)), Probe::Unsupported(v) if v == ["2027-01-01"]));

        // Legacy servers answer unknown methods with whatever code they like.
        for code in [-32601, -32602, -32600] {
            let legacy = json!({"jsonrpc":"2.0","id":1,"error":{"code":code,"message":"nope"}});
            assert!(matches!(classify_probe(Ok(legacy)), Probe::Legacy(_)));
        }
        let http_modern = Err(TransportError::Http {
            status: 400,
            body: json!({"jsonrpc":"2.0","id":1,"error":{"code":-32022,"message":"x",
                "data":{"supported":["2025-11-25","2026-07-28"]}}}).to_string(),
        });
        assert!(matches!(classify_probe(http_modern), Probe::Unsupported(_)));
        let http_legacy = Err(TransportError::Http { status: 404, body: "Not Found".into() });
        assert!(matches!(classify_probe(http_legacy), Probe::Legacy(_)));
    }
}
