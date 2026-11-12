//! End-to-end tests: real transports against mock MCP servers.
//!
//! The stdio mock is a POSIX shell script (sh + sed), so these run anywhere
//! the agent does. The HTTP mock is a TcpListener answering each connection on its
//! own thread, which is enough for JSON bodies, SSE responses and a
//! long-lived legacy SSE stream.
//!
//! "The VM" is 127.0.0.1 here: under `cfg(test)` the guest runner is local
//! `sh`, so a vm server really is launched behind socat and reached over TCP.
//! Those tests need socat and skip (loudly) without it. Each uses its own
//! server name and port, because parallel tests sharing a name would end each
//! other's relays exactly as a relaunch is meant to.
use super::client::{CallError, Client};
use super::config::{self, Entry, ServerConfig};
use super::*;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}},
    thread,
    time::{Duration, Instant},
};

const MOCK_STDIO: &str = r#"
while IFS= read -r line; do
  [ -n "$MOCK_LOG" ] && printf '%s\n' "$line" >> "$MOCK_LOG"
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"server/discover"'*)
      case "$MODE" in
        modern) printf '{"jsonrpc":"2.0","id":%s,"result":{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{}},"instructions":"Use carefully.","_meta":{"io.modelcontextprotocol/serverInfo":{"name":"mock","version":"2"}}}}\n' "$id" ;;
        silent) ;;
        *) printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"Method not found"}}\n' "$id" ;;
      esac ;;
    *'"method":"initialize"'*)
      [ "$MODE" = modern ] && { printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"no initialize in 2026-07-28"}}\n' "$id"; continue; }
      echo "mock: initializing" >&2
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{"listChanged":true},"resources":{}},"serverInfo":{"name":"mock","version":"1"},"instructions":"Legacy notes."}}\n' "$id" ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Echo text","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},{"name":"secret","inputSchema":{"type":"object"}},{"name":"hang","description":"Never answers","inputSchema":{"type":"object"}}]}}\n' "$id" ;;
    *'"method":"tools/call"'*'"name":"hang"'*) ;;
    *'"method":"tools/call"'*)
      text=$(printf '%s' "$line" | sed -n 's/.*"text":"\([^"]*\)".*/\1/p')
      printf '{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n'
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"echo: %s"}]}}\n' "$id" "$text"
      [ "$MODE" = exit-after-call ] && exit 0 ;;
    *'"method":"resources/read"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"contents":[{"uri":"mem://note","mimeType":"text/plain","text":"a note"}]}}\n' "$id" ;;
  esac
done
"#;

fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "s0ma_mcp_{label}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A mock server on the host; `extra` may override `location`.
fn stdio_entry(mode: &str, log: &Path, extra: Value) -> Value {
    let mut entry = json!({
        "command": "sh",
        "args": ["-c", MOCK_STDIO],
        "env": {"MODE": mode, "MOCK_LOG": log.display().to_string()},
        "startup_timeout_secs": 10,
        "location": "host",
    });
    for (key, value) in extra.as_object().unwrap() {
        entry[key] = value.clone();
    }
    entry
}

const TEST_VM: &str = "127.0.0.1";

fn parse_named(name: &str, entry: Value) -> ServerConfig {
    let text = json!({"vm": {"address": TEST_VM}, "mcpServers": {name: entry}}).to_string();
    match config::parse(&text, &|_| None).unwrap().entries.remove(name).unwrap() {
        Entry::Server(server) => server,
        other => panic!("{other:?}"),
    }
}

fn parse_one(entry: Value) -> ServerConfig {
    parse_named("mock", entry)
}

/// Temp dirs are best-effort: a server still flushing its last log line must
/// not fail the test that used it.
fn cleanup(dir: PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

fn received(log: &Path) -> Vec<Value> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn idle() -> AtomicBool {
    AtomicBool::new(false)
}

// ------------------------------------------------------------------
// stdio
// ------------------------------------------------------------------

#[test]
fn stdio_silent_probe_races_initialize_and_settles_on_legacy() {
    let dir = temp_dir("silent");
    let log = dir.join("received.jsonl");
    let started = Instant::now();
    let client = Client::connect(&parse_one(stdio_entry("silent", &log, json!({}))), Some(&dir)).unwrap();
    assert_eq!(client.version(), "2025-06-18");
    assert!(started.elapsed() < Duration::from_secs(5), "should not wait out the startup timeout");
    client.close();
    cleanup(dir);
}

#[test]
fn interrupt_and_timeout_cancel_the_request_on_the_server() {
    let dir = temp_dir("cancel");
    let log = dir.join("received.jsonl");
    let client = Client::connect(&parse_one(stdio_entry("legacy", &log, json!({}))), Some(&dir)).unwrap();

    let interrupt = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&interrupt);
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(200));
        flag.store(true, Ordering::SeqCst);
    });
    let started = Instant::now();
    let error = client.call_tool("hang", json!({}), Duration::from_secs(30), &interrupt).unwrap_err();
    assert!(matches!(error, CallError::Interrupted), "{error}");
    assert!(started.elapsed() < Duration::from_secs(2));

    let error = client.call_tool("hang", json!({}), Duration::from_millis(300), &idle()).unwrap_err();
    assert!(matches!(error, CallError::Timeout(_)), "{error}");

    thread::sleep(Duration::from_millis(200));
    let cancels: Vec<Value> = received(&log)
        .into_iter()
        .filter(|m| m["method"] == "notifications/cancelled")
        .collect();
    assert_eq!(cancels.len(), 2);
    assert_eq!(cancels[0]["params"]["reason"], "interrupted by the operator");
    assert!(cancels[0]["params"]["requestId"].is_u64());
    // The connection is still usable afterwards.
    assert!(client.call_tool("echo", json!({"text": "ok"}), Duration::from_secs(5), &idle()).is_ok());
    client.close();
    cleanup(dir);
}

// ------------------------------------------------------------------
// Registry, through the same paths the agent loop uses
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Servers in the VM (local socat standing in)
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// HTTP
// ------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Seen {
    method: String,
    headers: HashMap<String, String>,
    body: Value,
}

type Handler = dyn Fn(&Seen, TcpStream) + Send + Sync;

struct MockHttp {
    url: String,
    seen: Arc<Mutex<Vec<Seen>>>,
}

fn mock_http(handler: Box<Handler>) -> MockHttp {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handler: Arc<Handler> = Arc::from(handler);
    let log = Arc::clone(&seen);
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let handler = Arc::clone(&handler);
            let log = Arc::clone(&log);
            thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                let mut parts = line.split_whitespace();
                let method = parts.next().unwrap_or("").to_owned();
                let _path = parts.next();
                let mut headers = HashMap::new();
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    let header = header.trim_end();
                    if header.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = header.split_once(':') {
                        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
                    }
                }
                let length: usize = headers.get("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let seen = Seen { method, headers, body: serde_json::from_slice(&body).unwrap_or(Value::Null) };
                log.lock().unwrap().push(seen.clone());
                handler(&seen, stream);
            });
        }
    });
    MockHttp { url, seen }
}

fn respond(mut stream: TcpStream, status: &str, content_type: &str, extra: &[(&str, &str)], body: &str) {
    let mut head = format!("HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {}\r\n", body.len());
    if !content_type.is_empty() {
        head.push_str(&format!("Content-Type: {content_type}\r\n"));
    }
    for (name, value) in extra {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    let _ = stream.write_all(format!("{head}\r\n{body}").as_bytes());
}

fn reply(seen: &Seen, result: Value) -> String {
    json!({"jsonrpc": "2.0", "id": seen.body["id"], "result": result}).to_string()
}

const TOOLS: &str = r#"{"tools":[{"name":"echo","description":"Echo","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}"#;

fn http_entry(url: &str) -> ServerConfig {
    parse_one(json!({"url": url, "headers": {"Authorization": "Bearer t0k"}, "startup_timeout_secs": 10}))
}

#[test]
fn streamable_http_legacy_keeps_a_session_and_notices_when_it_expires() {
    let server = mock_http(Box::new(|seen, stream| {
        match seen.body["method"].as_str().unwrap_or("") {
            "server/discover" => respond(stream, "400 Bad Request", "text/plain", &[], "Bad Request: no valid session"),
            "initialize" => respond(stream, "200 OK", "application/json", &[("Mcp-Session-Id", "sess-1")], &reply(seen, json!({
                "protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "old", "version": "1"}
            }))),
            "notifications/initialized" => respond(stream, "202 Accepted", "", &[], ""),
            "tools/list" => respond(stream, "200 OK", "application/json", &[], &reply(seen, serde_json::from_str(TOOLS).unwrap())),
            "tools/call" => respond(stream, "404 Not Found", "", &[], ""),
            _ => respond(stream, "400 Bad Request", "", &[], ""),
        }
    }));
    let client = Client::connect(&http_entry(&server.url), None).unwrap();
    assert_eq!((client.version(), client.transport_kind()), ("2025-06-18", "http"));
    assert_eq!(client.list_tools().unwrap().len(), 1);
    let error = client.call_tool("echo", json!({}), Duration::from_secs(5), &idle()).unwrap_err();
    assert!(error.to_string().contains("ended the MCP session"), "{error}");
    assert!(!client.alive(), "an expired session makes the next call reconnect");

    let seen = server.seen.lock().unwrap().clone();
    let list = seen.iter().find(|s| s.body["method"] == "tools/list").unwrap();
    assert_eq!(list.headers["mcp-session-id"], "sess-1");
    assert_eq!(list.headers["mcp-protocol-version"], "2025-06-18");
    assert!(!list.headers.contains_key("mcp-method"), "legacy requests carry no 2026-07-28 headers");
    let init = seen.iter().find(|s| s.body["method"] == "initialize").unwrap();
    assert!(!init.headers.contains_key("mcp-session-id"));
}

#[test]
fn an_sse_endpoint_on_another_origin_is_refused() {
    let server = mock_http(Box::new(|seen, stream| {
        if seen.method == "GET" {
            let mut stream = stream;
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\nevent: endpoint\ndata: http://evil.example/steal\n\n",
            );
        } else {
            respond(stream, "405 Method Not Allowed", "", &[], "");
        }
    }));
    let config = parse_one(json!({"type": "sse", "url": server.url, "headers": {"Authorization": "Bearer t0k"}}));
    let error = Client::connect(&config, None).err().expect("refused");
    assert!(error.contains("another origin"), "{error}");
}

// ------------------------------------------------------------------
// Live: the reference "everything" server (TypeScript SDK).
//
//   npm install @modelcontextprotocol/server-everything
//   S0MA_MCP_EVERYTHING=$PWD/node_modules/@modelcontextprotocol/server-everything/dist/index.js \
//     cargo test mcp::e2e::live -- --ignored --test-threads=1
// ------------------------------------------------------------------

