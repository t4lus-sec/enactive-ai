//! End-to-end tests: real transports against mock MCP servers.
//!
//! The stdio mock is a POSIX shell script (sh + sed), so these run anywhere
//! s0ma does. The HTTP mock is a TcpListener answering each connection on its
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
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}, mpsc},
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

/// A mock server in "the VM" (local socat), with its own name and port.
fn vm_entry(mode: &str, log: &Path) -> Value {
    json!({
        "command": "sh",
        "args": ["-c", MOCK_STDIO],
        "env": {"MODE": mode, "MOCK_LOG": log.display().to_string()},
        "startup_timeout_secs": 10,
        "vm_port": free_port(),
    })
}

fn relay_running(name: &str) -> bool {
    std::process::Command::new("pgrep")
        .args(["-f", &format!("^socat .*[.]s0ma-mcp/{name}[.]sh")])
        .output()
        .is_ok_and(|o| o.status.success())
}

fn need_socat() -> bool {
    let present = super::guest::socat_available();
    if !present {
        eprintln!("skipping: socat is not installed, so the VM relay cannot be exercised");
    }
    present
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

fn methods(log: &Path) -> Vec<String> {
    received(log)
        .iter()
        .filter_map(|m| m.get("method").and_then(Value::as_str).map(str::to_owned))
        .collect()
}

fn idle() -> AtomicBool {
    AtomicBool::new(false)
}

// ------------------------------------------------------------------
// stdio
// ------------------------------------------------------------------

#[test]
fn stdio_legacy_server_falls_back_to_initialize() {
    let dir = temp_dir("legacy");
    let log = dir.join("received.jsonl");
    let client = Client::connect(&parse_one(stdio_entry("legacy", &log, json!({}))), Some(&dir)).unwrap();
    assert_eq!(client.version(), "2025-06-18");
    assert_eq!(client.server_info(), Some("mock 1"));
    assert_eq!(client.instructions(), Some("Legacy notes."));
    assert!(client.has("resources"));
    assert_eq!(client.list_tools().unwrap().len(), 3);
    let result = client.call_tool("echo", json!({"text": "hi"}), Duration::from_secs(5), &idle()).unwrap();
    assert_eq!(result["content"][0]["text"], "echo: hi");
    // The server announced a tool-list change; it is recorded, not acted on.
    assert!(client.stale());
    assert_eq!(
        methods(&log),
        ["server/discover", "initialize", "notifications/initialized", "tools/list", "tools/call"]
    );
    // Legacy requests carry no 2026-07-28 metadata.
    assert!(received(&log)[3]["params"].get("_meta").is_none());
    // stderr went to the per-server log, not the terminal.
    let server_log = std::fs::read_to_string(dir.join("mock.log")).unwrap();
    assert!(server_log.contains("stderr: mock: initializing"));
    client.close();
    cleanup(dir);
}

#[test]
fn stdio_modern_server_skips_the_handshake_and_gets_meta_on_every_request() {
    let dir = temp_dir("modern");
    let log = dir.join("received.jsonl");
    let client = Client::connect(&parse_one(stdio_entry("modern", &log, json!({}))), Some(&dir)).unwrap();
    assert_eq!(client.version(), "2026-07-28");
    assert_eq!(client.server_info(), Some("mock 2"));
    assert_eq!(client.instructions(), Some("Use carefully."));
    assert_eq!(client.list_tools().unwrap().len(), 3);
    assert_eq!(methods(&log), ["server/discover", "tools/list"]);
    let list = &received(&log)[1];
    assert_eq!(list["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"], "2026-07-28");
    assert_eq!(list["params"]["_meta"]["io.modelcontextprotocol/clientInfo"]["name"], "s0ma");
    client.close();
    cleanup(dir);
}

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
fn pinned_legacy_protocol_sends_no_probe() {
    let dir = temp_dir("pinned");
    let log = dir.join("received.jsonl");
    let config = parse_one(stdio_entry("legacy", &log, json!({"protocol": "legacy"})));
    let client = Client::connect(&config, Some(&dir)).unwrap();
    assert_eq!(client.version(), "2025-06-18");
    // A request round-trip guarantees the mock has logged everything before it.
    client.list_tools().unwrap();
    assert_eq!(methods(&log), ["initialize", "notifications/initialized", "tools/list"]);
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

#[test]
fn a_missing_command_fails_cleanly_with_the_reason() {
    let config = parse_one(json!({"command": "/nonexistent/s0ma-mcp-server", "location": "host"}));
    let error = Client::connect(&config, None).err().expect("cannot connect");
    assert!(error.contains("cannot start"), "{error}");
}

// ------------------------------------------------------------------
// Registry, through the same paths the agent loop uses
// ------------------------------------------------------------------

fn registry_for(dir: &Path, servers: Value) -> Registry {
    let path = dir.join("mcp.json");
    std::fs::write(&path, json!({"vm": {"address": TEST_VM}, "mcpServers": servers}).to_string()).unwrap();
    Registry { config_path: path, log_dir: dir.join("logs"), state: Mutex::new(State::default()) }
}

fn result_of(execution: DesktopExecution) -> Value {
    serde_json::from_str(&execution.result).unwrap()
}

/// Call a tool the way the agent loop does: by the name the model called.
fn run(registry: &Registry, tool: &str, arguments: Value) -> Result<DesktopExecution, String> {
    registry.dispatch(tool, arguments.as_object().unwrap().clone(), &idle()).map_err(|e| e.to_string())
}

fn mcp_call(registry: &Registry, server: &str, tool: &str, arguments: Value) -> Result<DesktopExecution, String> {
    run(registry, CALL_TOOL, json!({"server": server, "tool": tool, "arguments": arguments}))
}

#[test]
fn registry_exposes_calls_reconnects_and_reports() {
    let dir = temp_dir("registry");
    let log = dir.join("received.jsonl");
    let registry = registry_for(
        &dir,
        json!({
            "mock": stdio_entry("exit-after-call", &log, json!({"tools": ["echo", "hang", "nope"], "timeout_secs": 2})),
            "broken": {"command": "/nonexistent/s0ma-mcp-server", "location": "host"},
            "off": {"command": "x", "disabled": true},
            "bad": {"url": "ftp://nowhere"},
        }),
    );
    let lines = registry.reload(None, false).unwrap();
    assert!(lines.iter().any(|l| l.starts_with("mock: connected over stdio (2025-06-18), 2 of 3 tools, resources; not offered by the server: nope")), "{lines:?}");
    assert!(lines.iter().any(|l| l.starts_with("broken: not connected: cannot start")), "{lines:?}");
    assert!(lines.contains(&"off: disabled".to_string()));
    assert!(lines.iter().any(|l| l.starts_with("bad: config error")));

    // Two fixed tools carry every MCP tool; only names are up front.
    let tools = definitions(&lock(&registry.state));
    let names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
    assert_eq!(names, [SEARCH_TOOL, CALL_TOOL, RESOURCES_TOOL, RESOURCE_READ_TOOL]);
    assert!(tools[0].function.description.ends_with("Tools by server:\nmock: echo, hang"), "{}", tools[0].function.description);
    assert!(!serde_json::to_string(&tools).unwrap().contains("Echo text"), "descriptions load on demand");

    // A search loads the full definition.
    let found = result_of(run(&registry, SEARCH_TOOL, json!({"names": ["echo"]})).unwrap());
    assert_eq!(found["tools"][0]["description"], "Echo text");
    assert_eq!(found["tools"][0]["input_schema"]["required"], json!(["text"]));
    let by_keyword = result_of(run(&registry, SEARCH_TOOL, json!({"query": "never answers"})).unwrap());
    assert_eq!(by_keyword["tools"][0]["tool"], "hang", "{by_keyword}");

    let args = |v: Value| v.as_object().unwrap().clone();
    let first = result_of(mcp_call(&registry, "mock", "echo", json!({"text": "one"})).unwrap());
    assert_eq!(first["content"], "echo: one");
    // The mock exited after answering. The next call reconnects (a new
    // process) instead of failing, and the tool list is unchanged. A direct
    // call by exposed name, as in history from before deferral, still works.
    thread::sleep(Duration::from_millis(100));
    let second = result_of(registry.call_tool("mcp__mock__echo", args(json!({"text": "two"})), &idle()).unwrap());
    assert_eq!(second["content"], "echo: two");
    assert_eq!(methods(&log).iter().filter(|m| *m == "initialize").count(), 2);
    assert_eq!(definitions(&lock(&registry.state)).len(), 4);

    // Arguments that break the schema are caught before anything is sent.
    let calls_before = methods(&log).iter().filter(|m| *m == "tools/call").count();
    let missing = mcp_call(&registry, "mock", "echo", json!({})).err().unwrap();
    assert!(missing.contains("mock/echo: missing required argument(s) text"), "{missing}");
    let mistyped = mcp_call(&registry, "mock", "echo", json!({"text": 5})).err().unwrap();
    assert!(mistyped.contains("text must be string, not integer"), "{mistyped}");
    let flat = run(&registry, CALL_TOOL, json!({"server": "mock", "tool": "echo", "text": "x"})).err().unwrap();
    assert!(flat.contains("inside `arguments`"), "{flat}");
    assert_eq!(methods(&log).iter().filter(|m| *m == "tools/call").count(), calls_before);

    // A hung call times out with an explicitly unknown outcome.
    let hung = result_of(mcp_call(&registry, "mock", "hang", json!({})).unwrap());
    assert_eq!(hung["ok"], false);
    assert!(hung["outcome"].as_str().unwrap().starts_with("unknown"));

    let read = result_of(
        registry
            .read_resource(args(json!({"server": "mock", "uri": "mem://note"})), &idle())
            .unwrap(),
    );
    assert_eq!(read["content"], "[resource mem://note]\na note", "{read}");

    // Unknown tools and unknown servers are reported, not panicked on. A tool
    // left out by the allowlist is as unknown as one that does not exist.
    let unknown = mcp_call(&registry, "mock", "secret", json!({})).err().unwrap();
    assert!(unknown.contains("no tool 'secret' on mock"), "{unknown}");
    let unknown = registry.call_tool("mcp__mock__secret", Map::new(), &idle()).err().unwrap().to_string();
    assert!(unknown.contains("available: mock/echo, mock/hang"), "{unknown}");
    let broken = registry.call_tool("mcp__broken__x", Map::new(), &idle()).err().unwrap().to_string();
    assert!(broken.contains("no connected MCP tool"), "{broken}");
    let broken = mcp_call(&registry, "broken", "x", json!({})).err().unwrap();
    assert!(broken.contains("no MCP server named 'broken' has tools; servers: mock"), "{broken}");

    let status = registry.status();
    assert!(status.contains("connected") && status.contains("stdio · 2025-06-18 · mock 1 · 2/3 tools · ~"), "{status}");
    assert!(status.contains("tokens up front (names)") && status.contains("on demand"), "{status}");
    // Reconnecting re-listed the tools, so nothing is stale yet.
    assert!(!status.contains("tool list changed"), "{status}");
    assert!(status.contains("allowlisted but not offered: nope"), "{status}");
    assert!(registry.tool_listing(Some("mock")).unwrap().contains("\n  hang  ~"));

    let context = context_block(&lock(&registry.state));
    assert!(context.contains("[MCP]") && context.contains("mock (host, 2 tools, resources)"), "{context}");
    assert!(context.contains(GUIDE_HOST) && !context.contains(GUIDE_VM), "only the sentences that apply");
    assert!(context.contains("Notes from mock (server-authored): Legacy notes."), "{context}");
    assert!(!context.contains("broken"));

    // Reload one server after editing the file: the change is announced.
    std::fs::write(
        &registry.config_path,
        json!({"vm": {"address": TEST_VM}, "mcpServers": {"mock": stdio_entry("legacy", &log, json!({"tools": ["echo"]}))}}).to_string(),
    )
    .unwrap();
    let lines = registry.reload(Some("mock"), true).unwrap();
    assert!(lines.iter().any(|l| l.contains("1 of 3 tools")), "{lines:?}");
    // The listed names changed, which is a change to the tool prefix.
    assert!(lines.iter().any(|l| l.contains("prompt-cache miss")), "{lines:?}");
    assert!(definitions(&lock(&registry.state))[0].function.description.ends_with("\nmock: echo"));
    // This connection persists, so a list_changed notification stays visible
    // until the operator reloads.
    mcp_call(&registry, "mock", "echo", json!({"text": "x"})).unwrap();
    assert!(registry.status().contains("! mock says its tool list changed; /mcp reload mock"));
    // A full reload drops servers no longer in the file.
    let lines = registry.reload(None, true).unwrap();
    assert!(lines.contains(&"broken: removed".to_string()), "{lines:?}");
    assert!(registry.reload(Some("ghost"), true).is_err());

    for slot in lock(&registry.state).servers.values_mut() {
        if let Some(client) = slot.client.take() {
            client.close();
        }
    }
    cleanup(dir);
}

#[test]
fn an_operator_kill_stops_one_call_and_leaves_the_connection_usable() {
    let dir = temp_dir("kill");
    let log = dir.join("received.jsonl");
    let registry = registry_for(&dir, json!({"mock": stdio_entry("legacy", &log, json!({"timeout_secs": 60}))}));
    registry.reload(None, false).unwrap();

    // The agent installs a token per call; /kill sets it from another thread.
    let flag = Arc::new(AtomicBool::new(false));
    let _token = crate::kill::install(Arc::clone(&flag));
    let killer = Arc::clone(&flag);
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(300));
        killer.store(true, Ordering::SeqCst);
    });
    let started = Instant::now();
    let error = mcp_call(&registry, "mock", "hang", json!({})).err().unwrap();
    assert!(started.elapsed() < Duration::from_secs(3), "stopped promptly, not at the 60s timeout");
    assert!(error.contains("stopped: the request to mock/hang was cancelled"), "{error}");
    thread::sleep(Duration::from_millis(200));
    let cancels: Vec<Value> = received(&log).into_iter().filter(|m| m["method"] == "notifications/cancelled").collect();
    assert_eq!(cancels.len(), 1);
    assert_eq!(cancels[0]["params"]["reason"], "stopped by the operator");

    // The kill was for that call only: the next one, with a fresh token, works.
    drop(_token);
    let _fresh = crate::kill::install(Arc::new(AtomicBool::new(false)));
    assert_eq!(result_of(mcp_call(&registry, "mock", "echo", json!({"text": "still here"})).unwrap())["content"], "echo: still here");
    for slot in lock(&registry.state).servers.values_mut() {
        if let Some(client) = slot.client.take() {
            client.close();
        }
    }
    cleanup(dir);
}

// ------------------------------------------------------------------
// Servers in the VM (local socat standing in)
// ------------------------------------------------------------------

#[test]
fn vm_server_runs_behind_the_relay_and_is_relaunched_per_connection() {
    if !need_socat() {
        return;
    }
    let dir = temp_dir("vm_relay");
    let log = dir.join("received.jsonl");
    let config = parse_named("vm-relay", vm_entry("legacy", &log));
    assert!(matches!(config.endpoint, config::Endpoint::Vm { .. }), "command servers default to the VM");

    let client = Client::connect(&config, Some(&dir)).unwrap();
    assert_eq!((client.transport_kind(), client.version()), ("vm", "2025-06-18"));
    assert_eq!(client.transport_label(), "stdio relayed from the VM");
    assert_eq!(client.list_tools().unwrap().len(), 3);
    let echo = client.call_tool("echo", json!({"text": "via vm"}), Duration::from_secs(5), &idle()).unwrap();
    assert_eq!(echo["content"][0]["text"], "echo: via vm");
    // The launcher left the server script where it said, executable only by its owner.
    let script = super::guest::test_home().join(".s0ma-mcp/vm-relay.sh");
    assert!(std::fs::read_to_string(&script).unwrap().contains("exec env 'MOCK_LOG="));
    assert!(relay_running("vm-relay"));

    // Closing the connection ends the relay and the server with it.
    client.close();
    let deadline = Instant::now() + Duration::from_secs(5);
    while relay_running("vm-relay") && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    assert!(!relay_running("vm-relay"), "relay should exit with its connection");

    // A new connection launches a fresh one.
    let again = Client::connect(&config, Some(&dir)).unwrap();
    assert!(again.call_tool("echo", json!({"text": "again"}), Duration::from_secs(5), &idle()).is_ok());
    again.close();
    assert_eq!(methods(&log).iter().filter(|m| *m == "initialize").count(), 2);
    cleanup(dir);
}

#[test]
fn vm_server_speaks_2026_07_28_over_the_relay() {
    if !need_socat() {
        return;
    }
    let dir = temp_dir("vm_modern");
    let log = dir.join("received.jsonl");
    let client = Client::connect(&parse_named("vm-modern", vm_entry("modern", &log)), Some(&dir)).unwrap();
    assert_eq!(client.version(), "2026-07-28");
    assert_eq!(client.list_tools().unwrap().len(), 3);
    client.close();
    cleanup(dir);
}

#[test]
fn vm_failures_carry_the_vm_side_log() {
    if !need_socat() {
        return;
    }
    let config = parse_named(
        "vm-broken",
        json!({"command": "/nonexistent/s0ma-mcp-server", "vm_port": free_port(), "startup_timeout_secs": 5}),
    );
    let error = Client::connect(&config, None).err().expect("cannot connect");
    assert!(error.contains("VM log:") && error.contains("s0ma-mcp-server"), "{error}");
}

#[test]
fn registry_labels_vm_servers_and_explains_them_in_context() {
    if !need_socat() {
        return;
    }
    let dir = temp_dir("vm_registry");
    let log = dir.join("received.jsonl");
    let registry = registry_for(&dir, json!({"vm-reg": vm_entry("legacy", &log)}));
    let lines = registry.reload(None, false).unwrap();
    assert!(lines[0].starts_with("vm-reg: connected over stdio relayed from the VM (2025-06-18), 3 tools"), "{lines:?}");
    let status = registry.status();
    assert!(status.contains("connected    vm · relay · 2025-06-18"), "{status}");
    let context = context_block(&lock(&registry.state));
    assert!(context.contains("vm-reg (vm, 3 tools, resources)"), "{context}");
    assert!(context.contains(GUIDE_VM) && !context.contains(GUIDE_HOST), "{context}");
    let echo = result_of(mcp_call(&registry, "vm-reg", "echo", json!({"text": "hi"})).unwrap());
    assert_eq!(echo["content"], "echo: hi");
    for slot in lock(&registry.state).servers.values_mut() {
        if let Some(client) = slot.client.take() {
            client.close();
        }
    }
    cleanup(dir);
}

#[test]
fn no_config_means_no_context_and_no_tools() {
    let state = State::default();
    assert!(context_block(&state).is_empty());
    assert!(definitions(&state).is_empty());
}

// ------------------------------------------------------------------
// HTTP
// ------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Seen {
    method: String,
    path: String,
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
                let path = parts.next().unwrap_or("").to_owned();
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
                let seen = Seen { method, path, headers, body: serde_json::from_slice(&body).unwrap_or(Value::Null) };
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
fn streamable_http_modern_uses_per_request_headers_and_reads_sse_replies() {
    let server = mock_http(Box::new(|seen, stream| {
        let method = seen.body["method"].as_str().unwrap_or("");
        match method {
            "server/discover" => respond(stream, "200 OK", "application/json", &[], &reply(seen, json!({
                "resultType": "complete", "supportedVersions": ["2026-07-28"], "capabilities": {"tools": {}},
                "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "web", "version": "3"}}
            }))),
            "tools/list" => respond(stream, "200 OK", "application/json", &[], &reply(seen, serde_json::from_str(TOOLS).unwrap())),
            "tools/call" => {
                let text = seen.body["params"]["arguments"]["text"].as_str().unwrap_or("").to_owned();
                let progress = json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}});
                let body = format!(
                    ": keepalive\n\nevent: message\ndata: {progress}\n\ndata: {}\n\n",
                    reply(seen, json!({"resultType": "complete", "content": [{"type": "text", "text": format!("web: {text}")}]}))
                );
                let mut stream = stream;
                let _ = stream.write_all(format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
                ).as_bytes());
            }
            _ => respond(stream, "400 Bad Request", "", &[], ""),
        }
    }));
    let client = Client::connect(&http_entry(&server.url), None).unwrap();
    assert_eq!((client.version(), client.transport_kind()), ("2026-07-28", "http"));
    assert_eq!(client.list_tools().unwrap().len(), 1);
    let result = client.call_tool("echo", json!({"text": "hi"}), Duration::from_secs(5), &idle()).unwrap();
    assert_eq!(result["content"][0]["text"], "web: hi");

    let seen = server.seen.lock().unwrap().clone();
    for request in &seen {
        assert_eq!(request.headers["mcp-protocol-version"], "2026-07-28");
        assert_eq!(request.headers["mcp-method"], request.body["method"].as_str().unwrap());
        assert_eq!(request.headers["authorization"], "Bearer t0k");
        assert!(request.headers["accept"].contains("text/event-stream"));
        assert!(!request.headers.contains_key("mcp-session-id"));
    }
    assert_eq!(seen.last().unwrap().headers["mcp-name"], "echo");
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
fn old_http_sse_servers_are_found_by_fallback() {
    type Outbox = Arc<Mutex<Option<mpsc::Sender<String>>>>;
    let outbox: Outbox = Arc::new(Mutex::new(None));
    let server_outbox = Arc::clone(&outbox);
    let server = mock_http(Box::new(move |seen, stream| {
        match (seen.method.as_str(), seen.path.as_str()) {
            ("GET", "/mcp") => {
                let (sender, receiver) = mpsc::channel::<String>();
                *server_outbox.lock().unwrap() = Some(sender);
                let mut stream = stream;
                let _ = stream.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\nevent: endpoint\ndata: /messages?session=9\n\n",
                );
                let _ = stream.flush();
                for message in receiver {
                    if stream.write_all(format!("event: message\ndata: {message}\n\n").as_bytes()).is_err() {
                        break;
                    }
                    let _ = stream.flush();
                }
            }
            ("POST", path) if path.starts_with("/messages") => {
                let answer = match seen.body["method"].as_str().unwrap_or("") {
                    "initialize" => Some(reply(seen, json!({
                        "protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "sse", "version": "0"}
                    }))),
                    "tools/list" => Some(reply(seen, serde_json::from_str(TOOLS).unwrap())),
                    _ => None,
                };
                respond(stream, "202 Accepted", "", &[], "Accepted");
                if let (Some(answer), Some(sender)) = (answer, server_outbox.lock().unwrap().as_ref()) {
                    let _ = sender.send(answer);
                }
            }
            _ => respond(stream, "405 Method Not Allowed", "", &[], ""),
        }
    }));
    let client = Client::connect(&http_entry(&server.url), None).unwrap();
    assert_eq!((client.version(), client.transport_kind()), ("2024-11-05", "sse"));
    assert_eq!(client.list_tools().unwrap().len(), 1);
    client.close();
    drop(outbox.lock().unwrap().take());
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

fn everything() -> Option<String> {
    std::env::var("S0MA_MCP_EVERYTHING").ok().filter(|p| Path::new(p).exists())
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

struct Spawned(std::process::Child);
impl Drop for Spawned {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn spawn_http(script: &str, mode: &str, port: u16) -> Spawned {
    let child = std::process::Command::new("node")
        .args([script, mode])
        .env("PORT", port.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(Instant::now() < deadline, "server did not start listening");
        thread::sleep(Duration::from_millis(100));
    }
    Spawned(child)
}

fn exercise(client: &Client) {
    let tools = client.list_tools().unwrap();
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    for wanted in ["echo", "get-sum", "get-tiny-image"] {
        assert!(names.contains(&wanted), "{names:?}");
    }
    let echo = client.call_tool("echo", json!({"message": "hello from s0ma"}), Duration::from_secs(20), &idle()).unwrap();
    assert!(echo["content"][0]["text"].as_str().unwrap().contains("hello from s0ma"), "{echo}");
    let sum = client.call_tool("get-sum", json!({"a": 2, "b": 3}), Duration::from_secs(20), &idle()).unwrap();
    assert!(sum["content"][0]["text"].as_str().unwrap().contains('5'), "{sum}");
}

#[test]
#[ignore = "needs node and S0MA_MCP_EVERYTHING"]
fn live_everything_over_stdio_on_the_host() {
    live_everything_through_the_registry("host", "stdio");
}

#[test]
#[ignore = "needs node, socat and S0MA_MCP_EVERYTHING"]
fn live_everything_in_the_vm_through_the_relay() {
    live_everything_through_the_registry("vm", "stdio relayed from the VM");
}

fn live_everything_through_the_registry(location: &str, label: &str) {
    let Some(script) = everything() else { return };
    let dir = temp_dir("live_stdio");
    let registry = registry_for(
        &dir,
        json!({"everything": {"command": "node", "args": [script, "stdio"], "location": location, "startup_timeout_secs": 60}}),
    );
    let lines = registry.reload(None, false).unwrap();
    println!("{}", lines.join("\n"));
    assert!(lines[0].starts_with(&format!("everything: connected over {label} (2025-11-25)")), "{lines:?}");
    let client = lock(&registry.state).servers["everything"].client.clone().unwrap();
    exercise(&client);

    // Up front: the two fixed tools. On demand: what a search loads.
    let tools = definitions(&lock(&registry.state));
    let up_front = serde_json::to_string(&tools).unwrap().len() / 4;
    let on_demand: usize = lock(&registry.state).servers["everything"].tools.iter().map(RemoteTool::loaded_tokens).sum();
    println!("up front ~{up_front} tokens; all definitions ~{on_demand} tokens");
    assert!(up_front < on_demand, "names should cost less than the schemas");
    let found = result_of(run(&registry, SEARCH_TOOL, json!({"query": "add two numbers"})).unwrap());
    println!("search 'add two numbers' -> {}", clip(&found.to_string(), 300));
    assert_eq!(found["tools"][0]["tool"], "get-sum", "{found}");
    let sum = result_of(mcp_call(&registry, "everything", "get-sum", json!({"a": 2, "b": 40})).unwrap());
    assert!(sum["content"].as_str().unwrap().contains("42"), "{sum}");

    let args = |v: Value| v.as_object().unwrap().clone();
    let image = mcp_call(&registry, "everything", "get-tiny-image", json!({})).unwrap();
    assert!(image.observation.as_ref().is_some_and(|o| o.data_url.starts_with("data:image/png;base64,")));
    println!("get-tiny-image -> {}", image.result);

    let structured = result_of(mcp_call(&registry, "everything", "get-structured-content", json!({"location": "New York"})).unwrap());
    println!("get-structured-content -> {structured}");
    assert_eq!(structured["ok"], true);

    let links = result_of(mcp_call(&registry, "everything", "get-resource-links", json!({"count": 2})).unwrap());
    println!("get-resource-links -> {links}");
    assert!(links["content"].as_str().unwrap().contains("[resource link"));

    // The server asks the client for sampling mid-call; s0ma refuses, and the
    // tool must come back as a result, not hang.
    if lock(&registry.state).servers["everything"].tools.iter().any(|t| t.name == "trigger-sampling-request") {
        let sampling = result_of(mcp_call(&registry, "everything", "trigger-sampling-request", json!({"prompt": "hi"})).unwrap());
        println!("trigger-sampling-request -> {sampling}");
        assert!(sampling.get("outcome").is_none(), "must not time out: {sampling}");
    }

    if lock(&registry.state).servers["everything"].resources {
        let listed = result_of(registry.list_resources(args(json!({"server": "everything"})), &idle()).unwrap());
        let uri = listed["resources"][0]["uri"].as_str().unwrap().to_owned();
        let read = result_of(registry.read_resource(args(json!({"server": "everything", "uri": uri})), &idle()).unwrap());
        println!("resource {uri} -> {}", clip(&read.to_string(), 200));
        assert_eq!(read["ok"], true);
    }
    println!("{}", registry.status());
    println!("{}", context_block(&lock(&registry.state)));
    client.close();
    cleanup(dir);
}

#[test]
#[ignore = "needs node and S0MA_MCP_EVERYTHING"]
fn live_everything_over_streamable_http() {
    let Some(script) = everything() else { return };
    let port = free_port();
    let _server = spawn_http(&script, "streamableHttp", port);
    let client = Client::connect(&parse_one(json!({"url": format!("http://127.0.0.1:{port}/mcp")})), None).unwrap();
    assert_eq!((client.transport_kind(), client.version()), ("http", "2025-11-25"));
    exercise(&client);
    client.close();
}

#[test]
#[ignore = "needs node and S0MA_MCP_EVERYTHING"]
fn live_everything_over_legacy_sse_found_by_fallback() {
    let Some(script) = everything() else { return };
    let port = free_port();
    let _server = spawn_http(&script, "sse", port);
    let client = Client::connect(&parse_one(json!({"url": format!("http://127.0.0.1:{port}/sse")})), None).unwrap();
    assert_eq!(client.transport_kind(), "sse");
    exercise(&client);
    client.close();
}
