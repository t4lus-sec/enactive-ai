//! Starting an MCP server inside s0ma's VM.
//!
//! The QEMU guest agent, which `vm_exec` uses, can only run a command to
//! completion; it cannot hold a server's stdin/stdout open. So each connection
//! is set up in two moves:
//!
//! 1. Through the guest agent, write the server's command line to
//!    `~/.s0ma-mcp/<name>.sh` in the VM, end any relay left over from an
//!    earlier connection, and start `socat` listening on the VM's address. It
//!    relays exactly one TCP connection to the script's stdin/stdout (no
//!    `fork`), accepting it only from the host's address on the libvirt
//!    network.
//! 2. Connect from the host and speak newline-delimited JSON-RPC over the
//!    socket (see `LineStream::connect`).
//!
//! The server lives exactly as long as the connection: closing it gives the
//! server EOF on stdin, and a reconnect starts a fresh one. Its stderr, and
//! socat's, go to `~/.s0ma-mcp/<name>.log` in the VM, whose tail is attached
//! to connection errors.
//!
//! Under `cfg(test)` the "guest" is the local machine, with a private HOME, so
//! this whole path runs in tests wherever socat is installed.
use std::{
    collections::BTreeMap,
    net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket},
    path::Path,
    time::Duration,
};

use base64::{Engine as _, engine::general_purpose::STANDARD};

use super::transport::clip;

const LOG_ROTATE_BYTES: u64 = 1024 * 1024;

pub struct Launch<'a> {
    pub name: &'a str,
    pub command: &'a str,
    pub args: &'a [String],
    pub env: &'a BTreeMap<String, String>,
    pub cwd: Option<&'a Path>,
    pub address: Ipv4Addr,
    pub port: u16,
}

pub struct GuestOutput {
    pub exit_code: Option<i64>,
    pub stdout: String,
    pub stderr: String,
}

/// Start the relay in the VM; returns where to connect.
pub fn launch(spec: &Launch<'_>, timeout: Duration) -> Result<SocketAddr, String> {
    let peer = host_address_towards(spec.address)?;
    let script = launch_script(spec, peer);
    let timeout_secs = timeout.as_secs().clamp(5, 120);
    let output = run_in_guest(&script, timeout_secs)?;
    if output.exit_code != Some(0) {
        let detail = [output.stderr.trim(), output.stdout.trim()]
            .into_iter()
            .find(|text| !text.is_empty())
            .unwrap_or("no output");
        let lines: Vec<&str> = detail.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with("--- ")).collect();
        return Err(format!(
            "starting it on the machine failed (exit {}): {}",
            output.exit_code.map_or("?".into(), |code| code.to_string()),
            clip(&lines.join(" | "), 400)
        ));
    }
    Ok(SocketAddr::new(IpAddr::V4(spec.address), spec.port))
}

/// The last lines of the server's VM-side log, for error messages. Best
/// effort: `None` if the VM cannot be asked.
pub fn log_tail(name: &str) -> Option<String> {
    let command = format!("{HOME_FALLBACK}\ntail -n 4 \"$HOME/.s0ma-mcp/{name}.log\" 2>/dev/null || true");
    let output = run_in_guest(&command, 10).ok()?;
    let tail = output
        .stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("--- "))
        .collect::<Vec<_>>()
        .join(" | ");
    (!tail.is_empty()).then(|| clip(&tail, 400))
}

pub fn log_location(name: &str) -> String {
    format!("~/.s0ma-mcp/{name}.log on the machine")
}

/// The host's own address on the route to the VM: what the VM will see as
/// the connection's source, and so the only peer the relay accepts. No packet
/// is sent; connecting a UDP socket only picks the route.
fn host_address_towards(vm: Ipv4Addr) -> Result<Ipv4Addr, String> {
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).map_err(|e| e.to_string())?;
    socket
        .connect((vm, 9))
        .map_err(|e| format!("no route from the operator's workstation to the machine at {vm}: {e}"))?;
    match socket.local_addr().map_err(|e| e.to_string())?.ip() {
        IpAddr::V4(ip) => Ok(ip),
        IpAddr::V6(_) => Err("the route to the machine is not IPv4".into()),
    }
}


/// The guest agent's runner may start commands without HOME; npm and friends
/// need it, and the launcher keeps its files there.
const HOME_FALLBACK: &str = r#"[ -n "${HOME:-}" ] || HOME=$(getent passwd "$(id -un)" | cut -d: -f6); export HOME"#;

/// What socat runs for the connection: the configured command, with `env`
/// added to the VM user's environment and `cwd` (a VM path) applied.
pub fn server_script(command: &str, args: &[String], env: &BTreeMap<String, String>, cwd: Option<&Path>) -> String {
    let mut script = String::from("#!/bin/sh\n");
    if let Some(cwd) = cwd {
        script.push_str(&format!("cd {} || exit 1\n", crate::util::shell_quote(&cwd.to_string_lossy())));
    }
    script.push_str("exec ");
    if !env.is_empty() {
        script.push_str("env ");
        for (key, value) in env {
            script.push_str(&crate::util::shell_quote(&format!("{key}={value}")));
            script.push(' ');
        }
    }
    script.push_str(&crate::util::shell_quote(command));
    for arg in args {
        script.push(' ');
        script.push_str(&crate::util::shell_quote(arg));
    }
    script.push('\n');
    script
}

/// The command run through the guest agent. POSIX sh; `name` is restricted to
/// `[A-Za-z0-9_-]` by the config, so it needs no quoting. The pkill/pgrep
/// patterns are anchored to socat's own command line, so they find only this
/// server's relays: not the launcher itself, and not some other process that
/// merely mentions the script's path.
pub fn launch_script(spec: &Launch<'_>, peer: Ipv4Addr) -> String {
    let name = spec.name;
    let encoded = STANDARD.encode(server_script(spec.command, spec.args, spec.env, spec.cwd));
    format!(
        r#"set -e
{HOME_FALLBACK}
command -v socat >/dev/null 2>&1 || {{ echo "socat is not installed on the machine; install it with: sudo apt install socat" >&2; exit 127; }}
dir="$HOME/.s0ma-mcp"
mkdir -p "$dir"
chmod 700 "$dir"
pkill -f '^socat .*[.]s0ma-mcp/{name}[.]sh' 2>/dev/null || true
printf '%s' '{encoded}' | base64 -d > "$dir/{name}.sh"
chmod 700 "$dir/{name}.sh"
if [ -f "$dir/{name}.log" ] && [ "$(wc -c < "$dir/{name}.log")" -gt {LOG_ROTATE_BYTES} ]; then mv "$dir/{name}.log" "$dir/{name}.log.1"; fi
echo "--- $(date '+%F %T') launching for {peer}" >> "$dir/{name}.log"
setsid -f socat TCP4-LISTEN:{port},bind={address},range={peer}/32,reuseaddr EXEC:"$dir/{name}.sh" >> "$dir/{name}.log" 2>&1 < /dev/null
sleep 0.3
pgrep -f '^socat .*[.]s0ma-mcp/{name}[.]sh' >/dev/null || {{ echo "socat exited at once:" >&2; tail -n 5 "$dir/{name}.log" >&2; exit 1; }}
"#,
        port = spec.port,
        address = spec.address,
    )
}

#[cfg(not(test))]
fn run_in_guest(command: &str, timeout_secs: u64) -> Result<GuestOutput, String> {
    match crate::vm::run_helper_command(command, timeout_secs) {
        Ok(crate::vm::GuestCommandOutcome::Exited(result)) => Ok(GuestOutput {
            exit_code: result.exit_code,
            stdout: result.stdout,
            stderr: result.stderr,
        }),
        Ok(crate::vm::GuestCommandOutcome::HostTimeout { timeout_secs, .. }) => {
            Err(format!("the machine did not finish the launch command within {timeout_secs}s"))
        }
        Err(error) => Err(format!("cannot reach the machine: {error}")),
    }
}

/// Tests stand the local machine in for the VM, with a private HOME.
#[cfg(test)]
fn run_in_guest(command: &str, timeout_secs: u64) -> Result<GuestOutput, String> {
    let home = test_home();
    std::fs::create_dir_all(&home).map_err(|e| e.to_string())?;
    let output = std::process::Command::new("timeout")
        .arg(timeout_secs.to_string())
        .args(["sh", "-c", command])
        .env("HOME", &home)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| e.to_string())?;
    Ok(GuestOutput {
        exit_code: output.status.code().map(i64::from),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

#[cfg(test)]
pub fn test_home() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("s0ma-mcp-guest-{}", std::process::id()))
}

#[cfg(test)]
pub fn socat_available() -> bool {
    std::process::Command::new("sh")
        .args(["-c", "command -v socat"])
        .output()
        .is_ok_and(|o| o.status.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_script_quotes_everything_and_keeps_env_additive() {
        let env = BTreeMap::from([("TOKEN".to_string(), "it's secret".to_string())]);
        let args = vec!["-y".to_string(), "@scope/pkg".to_string(), "/home/s0ma/my notes".to_string()];
        let script = server_script("npx", &args, &env, Some(Path::new("/home/s0ma")));
        assert_eq!(
            script,
            "#!/bin/sh\ncd '/home/s0ma' || exit 1\nexec env 'TOKEN=it'\\''s secret' 'npx' '-y' '@scope/pkg' '/home/s0ma/my notes'\n"
        );
        let output = std::process::Command::new("sh")
            .args(["-c", &server_script("printf", &["%s|%s".into(), "a b".into(), "it's".into()], &BTreeMap::new(), None)])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&output.stdout), "a b|it's");
    }

    #[test]
    fn launch_script_binds_the_vm_address_and_accepts_only_the_host() {
        let env = BTreeMap::new();
        let spec = Launch {
            name: "notes",
            command: "npx",
            args: &[],
            env: &env,
            cwd: None,
            address: Ipv4Addr::new(192, 168, 10, 161),
            port: 47123,
        };
        let script = launch_script(&spec, Ipv4Addr::new(192, 168, 10, 24));
        assert!(script.contains("TCP4-LISTEN:47123,bind=192.168.10.161,range=192.168.10.24/32,reuseaddr EXEC:\"$dir/notes.sh\""));
        assert!(!script.contains(",fork"), "one connection per launch");
        assert!(script.contains("pkill -f '^socat .*[.]s0ma-mcp/notes[.]sh'"));
        // The pkill pattern must not match the launcher's own command line.
        let pattern = "^socat .*[.]s0ma-mcp/notes[.]sh";
        let own_line_matches = std::process::Command::new("sh")
            .args(["-c", &format!("printf '%s' {} | grep -q {}", crate::util::shell_quote(&script), crate::util::shell_quote(pattern))])
            .status()
            .unwrap()
            .success();
        assert!(!own_line_matches);
    }

    #[test]
    fn the_route_to_loopback_comes_from_loopback() {
        assert_eq!(host_address_towards(Ipv4Addr::LOCALHOST).unwrap(), Ipv4Addr::LOCALHOST);
    }
}
