//! `mcp.json`: which MCP servers the agent connects to.
//!
//! Operator-owned. No tool writes it, and nothing a server says changes it. The
//! shape is the `mcpServers` object Claude Desktop and Claude Code read, so a
//! server block can be copied across unchanged; the s0ma-specific keys
//! (`tools`, `exclude_tools`, `timeout_secs`, ...) are optional.
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "notes": {
//!       "command": "npx",
//!       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/operator/notes"],
//!       "tools": ["read_text_file", "list_directory", "search_files"]
//!     },
//!     "remote": {
//!       "type": "http",
//!       "url": "https://example.com/mcp",
//!       "headers": { "Authorization": "Bearer ${EXAMPLE_TOKEN}" }
//!     }
//!   }
//! }
//! ```
//!
//! `${VAR}` and `${VAR:-default}` expand from the agent's environment in `command`,
//! `args`, `env`, `cwd`, `url` and `headers`, so secrets need not sit in the file.
//! A variable that is unset and has no default is a configuration error for that
//! server only; the others still connect.
//!
//! # Where a server runs
//!
//! A `command` server runs **inside the agent's VM** unless it says
//! `"location": "host"`: the VM is where the agent already has authority, so that is
//! the default home for third-party code. Its paths are VM paths. The VM's
//! address defaults to 10.9.9.2 (the direct link; 192.168.10.161 is the same
//! machine over the house network); a top-level `"vm": {"address": ...}`
//! changes it. Each VM server gets a fixed port (47000-47999, derived from its
//! name; `vm_port` pins one). A `url` server is wherever its URL points; its
//! `location` (vm, host or remote) is inferred from the URL and only labels it.
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    net::Ipv4Addr,
    path::{Path, PathBuf},
    time::Duration,
};

pub const DEFAULT_CALL_TIMEOUT_SECS: u64 = 120;
pub const DEFAULT_STARTUP_TIMEOUT_SECS: u64 = 30;
const MAX_TIMEOUT_SECS: u64 = 3600;
/// Exposed tool names are `mcp__<server>__<tool>` and must fit the 64-character
/// function-name limit with room left for the tool.
pub const MAX_SERVER_NAME: usize = 32;
/// The agent VM on libvirt's default network.
pub const DEFAULT_VM_ADDRESS: Ipv4Addr = Ipv4Addr::new(10, 9, 9, 2);

/// The same machine on the house network, for when the direct link is out.
/// MCP servers are launched against one address (`vm.address`); this is here so
/// the rest of the harness can recognise either as the agent's own machine.
pub const FALLBACK_VM_ADDRESS: Ipv4Addr = Ipv4Addr::new(192, 168, 10, 161);
const VM_PORT_BASE: u16 = 47000;
const VM_PORT_SPAN: u16 = 1000;

/// Where a server's code runs, which decides what its paths mean and how much
/// its output is worth as evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Location {
    /// Inside the agent's VM, as the agent's VM user.
    Vm,
    /// On the operator's machine, outside the VM, as the operator.
    Host,
    /// Somewhere else, reached over HTTP.
    Remote,
}

impl Location {
    pub fn label(self) -> &'static str {
        match self {
            Self::Vm => "vm",
            Self::Host => "host",
            Self::Remote => "remote",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpTransport {
    /// Streamable HTTP, falling back to the deprecated HTTP+SSE transport when
    /// the endpoint turns out to be an old SSE server.
    Auto,
    Streamable,
    Sse,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Endpoint {
    /// A process on the host, spoken to over its stdin/stdout.
    Host {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        cwd: Option<PathBuf>,
        /// Pass the agent's whole environment through. Off by default: a third-party
        /// server gets a minimal environment plus what `env` names, not the
        /// search keys and endpoints in the agent's.
        inherit_env: bool,
    },
    /// A process in the VM: started by the guest agent behind `socat`, which
    /// relays one TCP connection to its stdin/stdout. `env` adds to the VM
    /// user's environment; `cwd` is a VM path.
    Vm {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        cwd: Option<PathBuf>,
        address: Ipv4Addr,
        port: u16,
    },
    Http {
        url: String,
        headers: BTreeMap<String, String>,
        transport: HttpTransport,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// Probe with `server/discover` (2026-07-28); fall back to `initialize`.
    Auto,
    /// Skip the probe and run the `initialize` handshake directly.
    Legacy,
    /// 2026-07-28 only.
    Modern,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    pub name: String,
    pub endpoint: Endpoint,
    /// Allowlist of the server's own tool names. `None` exposes every tool.
    pub include: Option<Vec<String>>,
    pub exclude: Vec<String>,
    pub call_timeout: Duration,
    pub startup_timeout: Duration,
    pub protocol: Protocol,
    /// Put the server's `instructions` (capped) in the [MCP] context block.
    pub instructions: bool,
    pub location: Location,
}

impl ServerConfig {
    pub fn exposes(&self, tool: &str) -> bool {
        self.include
            .as_ref()
            .is_none_or(|allowed| allowed.iter().any(|name| name == tool))
            && !self.exclude.iter().any(|name| name == tool)
    }

    pub fn transport_label(&self) -> &'static str {
        match &self.endpoint {
            Endpoint::Host { .. } => "stdio",
            Endpoint::Vm { .. } => "vm",
            Endpoint::Http { transport: HttpTransport::Sse, .. } => "sse",
            Endpoint::Http { .. } => "http",
        }
    }
}

/// One server entry as written, before validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    Server(ServerConfig),
    Disabled,
    Invalid(String),
}

#[derive(Debug, Default)]
pub struct Config {
    /// By server name, so the exposed tool list has a stable order.
    pub entries: BTreeMap<String, Entry>,
    pub warnings: Vec<String>,
}

#[derive(Deserialize)]
struct RawFile {
    #[serde(rename = "mcpServers", alias = "servers", default)]
    servers: BTreeMap<String, Value>,
    vm: Option<RawVm>,
    #[serde(flatten)]
    other: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVm {
    address: Option<String>,
}

#[derive(Deserialize)]
struct RawServer {
    #[serde(rename = "type", alias = "transport")]
    kind: Option<String>,
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    cwd: Option<String>,
    url: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    tools: Option<Vec<String>>,
    #[serde(default)]
    exclude_tools: Vec<String>,
    timeout_secs: Option<u64>,
    startup_timeout_secs: Option<u64>,
    #[serde(default)]
    disabled: bool,
    #[serde(default)]
    inherit_env: bool,
    protocol: Option<String>,
    instructions: Option<bool>,
    location: Option<String>,
    vm_port: Option<u16>,
    /// Keys other clients use (`autoApprove`, `alwaysAllow`, ...). Ignored with
    /// a warning rather than rejected, so a copied block still loads.
    #[serde(flatten)]
    other: BTreeMap<String, Value>,
}

/// Where the config lives: `S0MA_MCP_CONFIG`, else `<root>/mcp.json`.
pub fn path(root: &Path) -> PathBuf {
    std::env::var_os("S0MA_MCP_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("mcp.json"))
}

/// A missing file is an empty config: MCP is optional.
pub fn load(path: &Path) -> Result<Config, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse(&text, &|name| std::env::var(name).ok()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
        Err(error) => Err(format!("cannot read {}: {error}", path.display())),
    }
}

pub fn parse(text: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Result<Config, String> {
    let raw: RawFile = serde_json::from_str(text).map_err(|e| format!("mcp.json is not valid: {e}"))?;
    let vm_address = match raw.vm.and_then(|vm| vm.address) {
        None => DEFAULT_VM_ADDRESS,
        Some(address) => expand_vars(&address, lookup)?
            .parse::<Ipv4Addr>()
            .map_err(|_| format!("vm.address must be an IPv4 address, not '{address}'"))?,
    };
    let mut config = Config::default();
    for key in raw.other.keys() {
        config.warnings.push(format!("ignored unknown top-level key '{key}'"));
    }
    for (name, value) in raw.servers {
        let entry = match serde_json::from_value::<RawServer>(value) {
            Err(error) => Entry::Invalid(format!("invalid server entry: {error}")),
            Ok(server) => {
                for key in server.other.keys() {
                    config.warnings.push(format!("{name}: ignored unknown key '{key}'"));
                }
                if server.disabled {
                    Entry::Disabled
                } else {
                    match validate(&name, server, vm_address, lookup) {
                        Ok(server) => Entry::Server(server),
                        Err(error) => Entry::Invalid(error),
                    }
                }
            }
        };
        config.entries.insert(name, entry);
    }
    assign_vm_ports(&mut config.entries);
    Ok(config)
}

/// Give every VM server a distinct port: a pinned `vm_port` first, then a
/// name-derived one, probing upward on collision. Deterministic for a given
/// set of names, so a server keeps its port across restarts.
fn assign_vm_ports(entries: &mut BTreeMap<String, Entry>) {
    let mut taken = BTreeSet::new();
    let mut clash = Vec::new();
    for (name, entry) in entries.iter() {
        if let Entry::Server(ServerConfig { endpoint: Endpoint::Vm { port, .. }, .. }) = entry
            && *port != 0 && !taken.insert(*port) {
                clash.push((name.clone(), *port));
            }
    }
    for (name, port) in clash {
        entries.insert(name, Entry::Invalid(format!("vm_port {port} is already used by another server")));
    }
    for (name, entry) in entries.iter_mut() {
        if let Entry::Server(ServerConfig { endpoint: Endpoint::Vm { port, .. }, .. }) = entry
            && *port == 0 {
                let digest = Sha256::digest(name.as_bytes());
                let mut offset = u16::from_be_bytes([digest[0], digest[1]]) % VM_PORT_SPAN;
                while taken.contains(&(VM_PORT_BASE + offset)) {
                    offset = (offset + 1) % VM_PORT_SPAN;
                }
                *port = VM_PORT_BASE + offset;
                taken.insert(*port);
            }
    }
}

/// Where a URL points, for labelling only.
fn url_location(url: &reqwest::Url, vm_address: Ipv4Addr) -> Location {
    let host = url.host_str().unwrap_or("").trim_start_matches('[').trim_end_matches(']');
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) if ip == vm_address => Location::Vm,
        Ok(ip) if ip.is_loopback() => Location::Host,
        Err(_) if host.eq_ignore_ascii_case("localhost") => Location::Host,
        _ => Location::Remote,
    }
}

/// The server segment of `mcp__<server>__<tool>` must not be able to absorb a
/// `__` separator, or two servers could expose the same name.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_SERVER_NAME
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        && !name.contains("__")
        && !name.starts_with('_')
        && !name.ends_with('_')
}

fn timeout(value: Option<u64>, default: u64, key: &str) -> Result<Duration, String> {
    match value.unwrap_or(default) {
        secs @ 1..=MAX_TIMEOUT_SECS => Ok(Duration::from_secs(secs)),
        _ => Err(format!("{key} must be between 1 and {MAX_TIMEOUT_SECS}")),
    }
}

fn validate(
    name: &str,
    raw: RawServer,
    vm_address: Ipv4Addr,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<ServerConfig, String> {
    if !valid_name(name) {
        return Err(format!(
            "server name must be 1-{MAX_SERVER_NAME} characters of A-Z, a-z, 0-9, '_' or '-', \
             without '__' or a leading/trailing '_'"
        ));
    }
    let expand = |text: &str| expand_vars(text, lookup);
    let expand_map = |map: BTreeMap<String, String>| -> Result<BTreeMap<String, String>, String> {
        map.into_iter().map(|(k, v)| Ok((k, expand(&v)?))).collect()
    };

    let kind = raw.kind.as_deref().map(str::to_ascii_lowercase);
    let location = raw.location.as_deref().map(str::to_ascii_lowercase);
    let (endpoint, location) = match (kind.as_deref(), &raw.command, &raw.url) {
        (Some("stdio") | None, Some(command), None) => {
            if !raw.headers.is_empty() {
                return Err("headers apply only to http/sse servers".into());
            }
            let command = expand(command)?;
            let args = raw.args.iter().map(|a| expand(a)).collect::<Result<_, _>>()?;
            let env = expand_map(raw.env)?;
            let cwd = raw.cwd.as_deref().map(expand).transpose()?.map(PathBuf::from);
            match location.as_deref() {
                None | Some("vm") => {
                    if raw.inherit_env {
                        return Err("inherit_env applies to host servers; a vm server already runs with the machine user's environment".into());
                    }
                    if raw.vm_port.is_some_and(|port| port < 1024) {
                        return Err("vm_port must be 1024 or higher".into());
                    }
                    let port = raw.vm_port.unwrap_or(0); // 0: assigned after all entries are read
                    (Endpoint::Vm { command, args, env, cwd, address: vm_address, port }, Location::Vm)
                }
                Some("host") => {
                    if raw.vm_port.is_some() {
                        return Err("vm_port applies only to vm servers".into());
                    }
                    (Endpoint::Host { command, args, env, cwd, inherit_env: raw.inherit_env }, Location::Host)
                }
                Some(other) => return Err(format!("location for a command server must be \"vm\" or \"host\", not \"{other}\"")),
            }
        }
        (Some("http" | "streamable-http" | "streamable_http" | "sse") | None, None, Some(url)) => {
            if !raw.args.is_empty() || !raw.env.is_empty() || raw.cwd.is_some() || raw.inherit_env || raw.vm_port.is_some() {
                return Err("args, env, cwd, inherit_env and vm_port apply only to command servers".into());
            }
            let url = expand(url)?;
            let parsed = reqwest::Url::parse(&url).map_err(|e| format!("invalid url: {e}"))?;
            if !matches!(parsed.scheme(), "http" | "https") {
                return Err("url must be http or https".into());
            }
            let transport = match kind.as_deref() {
                Some("sse") => HttpTransport::Sse,
                Some(_) => HttpTransport::Streamable,
                None => HttpTransport::Auto,
            };
            let location = match location.as_deref() {
                None => url_location(&parsed, vm_address),
                Some("vm") => Location::Vm,
                Some("host") => Location::Host,
                Some("remote") => Location::Remote,
                Some(other) => return Err(format!("location must be \"vm\", \"host\" or \"remote\", not \"{other}\"")),
            };
            (Endpoint::Http { url, headers: expand_map(raw.headers)?, transport }, location)
        }
        (_, Some(_), Some(_)) => return Err("set either command (stdio) or url (http), not both".into()),
        (_, None, None) => return Err("missing command (stdio) or url (http)".into()),
        (Some(kind), _, _) => {
            return Err(format!(
                "type '{kind}' does not match the entry (stdio needs command; http/sse need url)"
            ));
        }
    };

    let protocol = match raw.protocol.as_deref() {
        None | Some("auto") => Protocol::Auto,
        Some("legacy") => Protocol::Legacy,
        Some(v) if v == super::client::MODERN_VERSION => Protocol::Modern,
        Some(other) => {
            return Err(format!(
                "protocol must be \"auto\", \"legacy\" or \"{}\", not \"{other}\"",
                super::client::MODERN_VERSION
            ));
        }
    };

    Ok(ServerConfig {
        name: name.to_owned(),
        endpoint,
        include: raw.tools,
        exclude: raw.exclude_tools,
        call_timeout: timeout(raw.timeout_secs, DEFAULT_CALL_TIMEOUT_SECS, "timeout_secs")?,
        startup_timeout: timeout(
            raw.startup_timeout_secs,
            DEFAULT_STARTUP_TIMEOUT_SECS,
            "startup_timeout_secs",
        )?,
        protocol,
        instructions: raw.instructions.unwrap_or(true),
        location,
    })
}

/// `${VAR}` and `${VAR:-default}`. A `$` not followed by `{` is literal.
pub fn expand_vars(text: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after
            .find('}')
            .ok_or_else(|| format!("unterminated ${{...}} in '{text}'"))?;
        let body = &after[..end];
        let (name, default) = match body.split_once(":-") {
            Some((name, default)) => (name, Some(default)),
            None => (body, None),
        };
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err(format!("invalid variable name '${{{body}}}'"));
        }
        match (lookup(name).filter(|v| !v.is_empty()), default) {
            (Some(value), _) => out.push_str(&value),
            (None, Some(default)) => out.push_str(default),
            (None, None) => return Err(format!("environment variable {name} is not set")),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;


    fn env(name: &str) -> Option<String> {
        match name {
            "TOKEN" => Some("t0k".into()),
            "EMPTY" => Some(String::new()),
            _ => None,
        }
    }

    #[test]
    fn claude_style_config_parses_with_inferred_transports() {
        let config = parse(
            r#"{"mcpServers": {
                "fs": {"command": "npx", "args": ["-y", "server-fs", "${NOPE:-/tmp}"],
                       "tools": ["read_file"], "autoApprove": ["read_file"]},
                "remote": {"type": "http", "url": "https://example.com/mcp",
                           "headers": {"Authorization": "Bearer ${TOKEN}"}, "timeout_secs": 30},
                "old": {"url": "http://localhost:3001/sse", "type": "sse"},
                "inferred": {"url": "http://localhost:3002/mcp"},
                "off": {"command": "x", "disabled": true}
            }}"#,
            &env,
        )
        .unwrap();
        let Entry::Server(fs) = &config.entries["fs"] else { panic!() };
        // A command server runs in the VM unless it says otherwise.
        assert!(matches!(&fs.endpoint, Endpoint::Vm { args, address, port, .. }
            if args[2] == "/tmp" && *address == DEFAULT_VM_ADDRESS && (47000..48000).contains(port)));
        assert_eq!(fs.location, Location::Vm);
        assert!(fs.exposes("read_file") && !fs.exposes("write_file"));
        assert_eq!(fs.call_timeout, Duration::from_secs(DEFAULT_CALL_TIMEOUT_SECS));
        let Entry::Server(remote) = &config.entries["remote"] else { panic!() };
        assert!(matches!(&remote.endpoint,
            Endpoint::Http { headers, transport: HttpTransport::Streamable, .. } if headers["Authorization"] == "Bearer t0k"));
        assert_eq!(remote.call_timeout, Duration::from_secs(30));
        assert_eq!(remote.location, Location::Remote);
        let Entry::Server(old) = &config.entries["old"] else { panic!() };
        assert_eq!(old.transport_label(), "sse");
        let Entry::Server(inferred) = &config.entries["inferred"] else { panic!() };
        assert!(matches!(&inferred.endpoint, Endpoint::Http { transport: HttpTransport::Auto, .. }));
        assert_eq!(inferred.location, Location::Host, "localhost URLs are on the host");
        assert_eq!(config.entries["off"], Entry::Disabled);
        assert_eq!(config.warnings, ["fs: ignored unknown key 'autoApprove'"]);
        // BTreeMap order: stable, alphabetical.
        assert_eq!(config.entries.keys().collect::<Vec<_>>(), ["fs", "inferred", "off", "old", "remote"]);
    }
}
