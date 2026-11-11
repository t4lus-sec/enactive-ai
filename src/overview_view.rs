//! The overview tab (Alt+O), a submodule of `tabs.rs`: s0ma's settings and
//! state on one screen (`overview.rs` says how it reaches the app).
//!
//! ```text
//!   settings
//! ❯ reasoning     forced xhigh                       ←→ change
//!   loop          off                                ⏎ turn on
//!   voice         off                                ⏎ turn on
//!   daily sleep   02:00                              ⏎ set
//!   time zone     America/Denver (UTC-7)             ⏎ set
//!   s0ma
//!   goal          active · ship the share tab        p pause · r resume · x clear
//!   jobs          1 open: confidential-audit
//!   mcp
//!   github        connected · stdio · 12/12 tools    ⏎ reload
//!   mcp.json      2 servers · ~/s0ma/mcp.json        e edit · a add a server
//!   ────────── what the selected row holds, in full
//! ```
//!
//! Each change is the command the operator would have typed (`/mode low`, `/loop
//! on`, `/sleep set 0200`, `/mcp reload github`), run by the app as typed:
//! its reply shows in the footer, and the row follows once the app has said
//! what it is now. During a turn changes wait for it to end.
//!
//! `mcp.json` is edited here in the operator's editor (`e`, or `a` with a new
//! server's block put in to fill): what he saves is checked as s0ma would
//! read it, written, and MCP reloaded (`/mcp reload`). An edit that does not
//! read is not written; it is kept, and `e` opens it again.

use super::{Action, Footer, Open, Outcome, act, clip, flow, text_width};
use crate::mistake_view::{Line, Style};
use crate::overview::Snapshot;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use std::time::{Duration, Instant};

/// Asked of the app this often while the tab is up.
const ASK_EVERY: Duration = Duration::from_millis(700);

const MODES: [&str; 5] = ["auto", "instruct", "low", "medium", "xhigh"];

#[derive(Clone, Debug, PartialEq, Eq)]
enum Row {
    Context,
    Reasoning,
    WorkerReasoning,
    Loop,
    Voice,
    /// s0ma's access to its machine.
    Vm,
    Sleep,
    Tz,
    Goal,
    Jobs,
    /// Where s0ma's commands run: in shells he sees, or outside them.
    Shells,
    /// What runs on s0ma's machine without a turn: watches, background
    /// commands.
    Watches,
    Server(String),
    McpConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    Sleep,
    Tz,
}

struct Input {
    field: Field,
    text: Vec<char>,
    cursor: usize,
}

#[derive(Default)]
pub(super) struct State {
    snapshot: Option<Snapshot>,
    /// How old the app said the snapshot was, and when it was asked.
    age: Duration,
    asked: Option<Instant>,
    error: Option<String>,
    selected: usize,
    input: Option<Input>,
    /// `x` once on the goal: again clears it.
    pub(super) clearing: bool,
    /// The last reply shown, so each is said once.
    replied: u64,
    /// A change sent and not yet in a snapshot: shown as on its way.
    sent: Option<(String, Instant)>,
    /// The operator's edit of `mcp.json` that was not saved, and why.
    draft: Option<(String, String)>,
}

/// The watches row: how many run, the first named.
fn watches_value(watches: &crate::overview::Watches) -> Line {
    match (&watches.error, watches.listed, watches.running.len()) {
        (Some(error), _, _) => vec![(Style::Dim, clip(error, 60))],
        (None, false, _) => vec![(Style::Dim, "listing…".into())],
        (None, true, 0) => vec![(Style::Dim, "none running".into())],
        (None, true, count) => {
            let names: Vec<&str> = watches.running.iter().map(|(name, _, _)| name.as_str()).take(3).collect();
            let more = if count > 3 { format!(" +{}", count - 3) } else { String::new() };
            vec![(Style::Warn, format!("[*] {count} running: {}{more}", names.join(", ")))]
        }
    }
}

/// Every running watch and background command in full: what it runs.
fn watches_detail(watches: &crate::overview::Watches) -> String {
    let mut text = String::from(
        "Commands s0ma left running on its machine, outside any shell: each wakes it when it ends (vm_wake_watch, and vm_exec with background=true, whose output also has a shell of its own). Listed every 20 s.\n",
    );
    if let Some(error) = &watches.error {
        text.push_str(&format!("\n{error}"));
        return text;
    }
    for (name, repeat, command) in &watches.running {
        text.push_str(&format!("\n{name}{}: {command}", if *repeat { " (repeats)" } else { "" }));
    }
    if watches.running.is_empty() {
        text.push_str("\nNone running.");
    }
    if watches.ended > 0 {
        text.push_str(&format!("\n{} ended.", watches.ended));
    }
    text
}

fn rows(snapshot: &Snapshot) -> Vec<Row> {
    let mut rows = vec![Row::Reasoning, Row::WorkerReasoning, Row::Loop, Row::Voice, Row::Vm, Row::Sleep, Row::Tz, Row::Context, Row::Goal, Row::Jobs, Row::Shells, Row::Watches];
    rows.extend(snapshot.mcp.iter().map(|server| Row::Server(server.name.clone())));
    rows.push(Row::McpConfig);
    rows
}

/// The latest from the app, asked again when due.
fn refresh(tab: &mut Open) {
    let due = tab.overview.asked.is_none_or(|at| at.elapsed() >= ASK_EVERY);
    if !due {
        return;
    }
    tab.overview.asked = Some(Instant::now());
    match act(Action::Overview) {
        Ok(json) => match serde_json::from_str::<Option<(Snapshot, u64)>>(&json) {
            Ok(Some((snapshot, age_ms))) => {
                if let Some(reply) = snapshot.reply.as_ref().filter(|reply| reply.seq > tab.overview.replied) {
                    tab.overview.replied = reply.seq;
                    // The first reply seen when the tab opens is old news.
                    if tab.overview.snapshot.is_some() {
                        tab.overview.sent = None;
                        tab.note(format!("{}: {}", reply.command, clip(&reply.text, 200)), reply.problem);
                    }
                }
                tab.overview.snapshot = Some(snapshot);
                tab.overview.age = Duration::from_millis(age_ms);
                tab.overview.error = None;
            }
            Ok(None) => {}
            Err(error) => tab.overview.error = Some(format!("the app's answer was unreadable: {error}")),
        },
        Err(error) => tab.overview.error = Some(error),
    }
}

fn send(tab: &mut Open, command: String) {
    match act(Action::Command { text: command.clone() }) {
        Ok(_) => {
            let busy = tab.overview.snapshot.as_ref().is_some_and(|snapshot| snapshot.busy);
            tab.note(if busy { format!("{command} · after s0ma's turn") } else { format!("{command} …") }, false);
            tab.overview.sent = Some((command, Instant::now()));
            // Asked again soon, to show what it did.
            tab.overview.asked = Instant::now().checked_sub(ASK_EVERY.saturating_sub(Duration::from_millis(250)));
        }
        Err(error) => tab.note(error, true),
    }
}

fn label_row(name: &str, value: Line, action: &str, selected: bool, width: usize) -> Line {
    let lead = if selected { (Style::Cursor, "❯ ".to_string()) } else { (Style::Plain, "  ".to_string()) };
    let name = format!("{name:<14}");
    let mut line = vec![lead, (if selected { Style::Bold } else { Style::Plain }, name)];
    // The selected row's own keys, as buttons.
    let buttons = if selected { super::gui_view::row_buttons(action) } else { Vec::new() };
    let buttons_width: usize = buttons.iter().map(|(_, text)| text_width(text)).sum();
    let room = width.saturating_sub(2 + 14 + buttons_width + 3);
    let mut used = 0;
    for (style, text) in value {
        let left = room.saturating_sub(used);
        if left == 0 {
            break;
        }
        let text = clip(&text, left);
        used += text_width(&text);
        line.push((style, text));
    }
    line.push((Style::Plain, " ".repeat(room.saturating_sub(used) + 2)));
    line.extend(buttons);
    line
}

/// `84,213 of 262,144 tokens · 32%`, marked from three quarters full.
fn context_value(context: Option<&crate::overview::Context>) -> Line {
    match context {
        Some(Ok((tokens, window))) => {
            let percent = if *window == 0 { 0 } else { tokens * 100 / window };
            vec![
                (if percent >= 75 { Style::Warn } else { Style::Plain }, format!("{} of {} tokens", group_digits(*tokens), group_digits(*window))),
                (Style::Dim, format!(" · {percent}%")),
            ]
        }
        Some(Err(_)) => vec![(Style::Problem, "not measured".into())],
        None => vec![(Style::Dim, "not measured yet".into())],
    }
}

fn group_digits(n: usize) -> String {
    let digits: Vec<char> = n.to_string().chars().collect();
    let mut groups: Vec<String> = digits.rchunks(3).map(|chunk| chunk.iter().collect()).collect();
    groups.reverse();
    groups.join(",")
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or_default().trim().to_string()
}

fn on_off(on: bool) -> Line {
    if on { vec![(Style::Success, "on".into())] } else { vec![(Style::Dim, "off".into())] }
}

pub(super) fn body(tab: &mut Open, width: usize, rows_left: usize) -> (Vec<Line>, Footer) {
    refresh(tab);
    let state = &mut tab.overview;
    let Some(snapshot) = state.snapshot.clone() else {
        let text = state.error.clone().unwrap_or_else(|| "asking s0ma…".into());
        let style = if state.error.is_some() { Style::Problem } else { Style::Work };
        return (vec![Vec::new(), vec![(style, format!("  {text}"))]], Footer::new(vec![(Style::Status, "overview".into())], vec![("esc", "back")]));
    };
    let all = rows(&snapshot);
    state.selected = state.selected.min(all.len().saturating_sub(1));
    let selected = all.get(state.selected).cloned();
    let mut out: Vec<Line> = Vec::new();
    let heading = |text: &str| vec![(Style::Status, format!("  {text}"))];
    out.push(heading("settings"));
    for (index, row) in all.iter().enumerate() {
        let here = index == state.selected;
        if *row == Row::Context {
            out.push(heading("s0ma"));
        }
        if matches!(row, Row::Server(_) | Row::McpConfig) && !matches!(all.get(index.wrapping_sub(1)), Some(Row::Server(_))) {
            out.push(heading("mcp"));
        }
        let typed = state.input.as_ref().filter(|input| matches!((row, input.field), (Row::Sleep, Field::Sleep) | (Row::Tz, Field::Tz)));
        let (name, value, action): (String, Line, String) = match row {
            Row::Reasoning => ("reasoning".into(), vec![(Style::Plain, snapshot.reasoning.clone())], "←→ change".into()),
            Row::WorkerReasoning => (
                "subagents".into(),
                vec![(Style::Plain, if snapshot.worker_reasoning.is_empty() { "medium".into() } else { snapshot.worker_reasoning.clone() })],
                "←→ change".into(),
            ),
            Row::Loop => ("loop".into(), on_off(snapshot.looping), format!("⏎ turn {}", if snapshot.looping { "off" } else { "on" })),
            Row::Voice => ("voice".into(), on_off(snapshot.voice_on), format!("⏎ turn {}", if snapshot.voice_on { "off" } else { "on" })),
            Row::Vm => (
                "vm access".into(),
                if snapshot.vm_access { vec![(Style::Plain, "on".into())] } else { vec![(Style::Problem, "[!] off".into())] },
                format!("⏎ turn {}", if snapshot.vm_access { "off" } else { "on" }),
            ),
            Row::Sleep => (
                "daily sleep".into(),
                match &snapshot.sleep_at {
                    Some(at) => vec![(Style::Plain, at.clone())],
                    None => vec![(Style::Dim, "off".into())],
                },
                "⏎ set".into(),
            ),
            Row::Tz => (
                "time zone".into(),
                match &snapshot.tz_zone {
                    Some(zone) => vec![(Style::Plain, zone.clone())],
                    None => vec![(Style::Dim, "host's only".into())],
                },
                "⏎ set".into(),
            ),
            Row::Goal => (
                "goal".into(),
                if snapshot.goal_set { vec![(Style::Plain, first_line(&crate::overview::bare(&snapshot.goal)))] } else { vec![(Style::Dim, "none".into())] },
                if snapshot.goal_set { "p pause · r resume · x clear".into() } else { "/goal <objective> sets one".into() },
            ),
            Row::Context => ("context".into(), context_value(snapshot.context.as_ref()), String::new()),
            Row::Jobs => ("jobs".into(), vec![(Style::Plain, first_line(&crate::overview::bare(&snapshot.jobs)))], String::new()),
            Row::Shells => (
                "commands".into(),
                if snapshot.in_shells {
                    vec![(Style::Plain, "in shells you see, or not run".into())]
                } else {
                    vec![(Style::Problem, "[!] outside shells: S0MA_VM_EXEC_IN_NETBOX=off".into())]
                },
                String::new(),
            ),
            Row::Watches => ("watches".into(), watches_value(&snapshot.watches), String::new()),
            Row::Server(name) => {
                let server = snapshot.mcp.iter().find(|server| server.name == *name);
                let (state_style, word, detail) = match server {
                    Some(server) => (
                        match server.state.as_str() {
                            "connected" => Style::Success,
                            "disabled" | "closed" => Style::Dim,
                            _ => Style::Problem,
                        },
                        server.state.clone(),
                        server.detail.clone(),
                    ),
                    None => (Style::Dim, String::new(), String::new()),
                };
                (name.clone(), vec![(state_style, word), (Style::Dim, format!("  {detail}"))], "⏎ reload".into())
            }
            Row::McpConfig => {
                let value = match (&state.draft, &snapshot.mcp_config) {
                    (Some((_, why)), _) => vec![(Style::Problem, format!("your edit is not saved: {why}"))],
                    (None, Some(path)) => {
                        let servers = match snapshot.mcp.len() {
                            0 => "no servers yet".to_string(),
                            1 => "1 server".to_string(),
                            count => format!("{count} servers"),
                        };
                        vec![(Style::Plain, servers), (Style::Dim, format!(" · {}", crate::share_sync::home_short(std::path::Path::new(path))))]
                    }
                    (None, None) => vec![(Style::Dim, "MCP is not set up".into())],
                };
                let action = if state.draft.is_some() { "e fix it · x drop it" } else { "e edit · a add a server" };
                ("mcp.json".into(), value, action.into())
            }
        };
        match typed {
            Some(input) => {
                let label = match input.field {
                    Field::Sleep => "at (0200, or off) › ",
                    Field::Tz => "zone (UTC-6, America/Denver, or off) › ",
                };
                let before: String = input.text[..input.cursor].iter().collect();
                let at: String = input.text.get(input.cursor).map_or(" ".to_string(), |c| c.to_string());
                let after: String = input.text.get(input.cursor + 1..).map(|rest| rest.iter().collect()).unwrap_or_default();
                out.push(vec![
                    (Style::Cursor, "❯ ".into()),
                    (Style::Bold, format!("{name:<14}")),
                    (Style::Cursor, label.into()),
                    (Style::Plain, before),
                    (Style::Caret, at),
                    (Style::Plain, after),
                ]);
            }
            None => out.push(super::gui_view::pick_row(
                label_row(&name, value, &action, here, width),
                super::gui_view::Click::Pick(super::gui_view::Pick::Overview(index)),
            )),
        }
    }

    // What the selected row holds, in full.
    let detail: String = match &selected {
        Some(Row::Reasoning) => "How hard s0ma thinks: auto picks per turn; instruct, low, medium and xhigh force one (/mode).".into(),
        Some(Row::WorkerReasoning) => "How hard subagents think, with s0ma's own settings at that level: instruct (no thinking), low, medium or xhigh. Kept across restarts; subagents started from now on take it (/mode workers).".into(),
        Some(Row::Loop) => "On, s0ma keeps taking turns on its own; off, it waits for you (/loop).".into(),
        Some(Row::Voice) => snapshot.voice.clone(),
        Some(Row::Vm) if snapshot.vm_access => "On, s0ma and its subagents use its machine: commands, shells, the desktop and browser, /share files, \
            board tests and reports, MCP servers in the VM. Off, every one of those is refused at once, mid-turn too, until it is on again (/vm).".into(),
        Some(Row::Vm) => "Off: s0ma's commands, shells, desktop, browser, /share files, board tests and VM MCP servers are refused, for its subagents too, \
            and each turn tells it so. The share sync and file previews carry on. ⏎ or /vm on gives the machine back.".into(),
        Some(Row::Sleep) => snapshot.sleep.clone(),
        Some(Row::Tz) => snapshot.tz.clone(),
        Some(Row::Goal) => crate::overview::bare(&snapshot.goal),
        Some(Row::Context) => match &snapshot.context {
            Some(Err(error)) => format!("Not measured: {error}"),
            _ => "Tokens in s0ma's context now, of the model's window.".into(),
        },
        Some(Row::Jobs) => format!("{}\n/job start|pause|resume|close <name> manages them.", crate::overview::bare(&snapshot.jobs)),
        Some(Row::Shells) if snapshot.in_shells => "Every command s0ma and its subagents run with vm_exec is typed into a shell you see (Alt+Shift+1…); \
            when none can take it, it is refused, not run elsewhere. Board goal tests run outside a shell, and are shown on the page as they start and end.".into(),
        Some(Row::Shells) => "S0MA_VM_EXEC_IN_NETBOX=off is set: vm_exec runs commands through the guest runner, in no shell. \
            Each still shows on the page as a tool call; unset it to see them in shells again.".into(),
        Some(Row::Watches) => watches_detail(&snapshot.watches),
        Some(Row::McpConfig) => CONFIG_HELP.into(),
        Some(Row::Server(name)) => match snapshot.mcp.iter().find(|server| server.name == *name) {
            Some(server) => {
                let mut text = format!("{} · {}\n", server.state, server.detail);
                if server.tools.is_empty() {
                    text.push_str("no tools exposed");
                } else {
                    text.push_str(&server.tools.join("\n"));
                }
                text
            }
            None => String::new(),
        },
        None => String::new(),
    };
    let mut detail = detail;
    if matches!(selected, Some(Row::Server(_) | Row::McpConfig)) && !snapshot.mcp_notes.is_empty() {
        detail.push_str(&format!("\n\n{}", snapshot.mcp_notes.join("\n")));
    }
    let left_rows = rows_left.saturating_sub(out.len() + 1);
    out.push(vec![(Style::Dim, format!("  {}", "─".repeat(width.saturating_sub(4))))]);
    let mut strip: Vec<Line> = Vec::new();
    for line in detail.lines() {
        strip.extend(flow(&[(Style::Plain, format!("  {line}"))], width, 2));
    }
    out.extend(strip.into_iter().take(left_rows));
    out.truncate(rows_left);

    let mut left: Line = vec![(Style::Status, "overview".into())];
    if snapshot.restored {
        left.push((Style::Dim, format!("  as of {}, before s0ma restarted", snapshot.taken)));
    } else if snapshot.busy {
        left.push((Style::Work, format!("  s0ma is in a turn (as of {}): changes wait for it", snapshot.taken)));
    } else if let Some((command, at)) = &state.sent
        && at.elapsed() < Duration::from_secs(10)
    {
        left.push((Style::Work, format!("  {command} …")));
    } else if state.age > Duration::from_secs(5) {
        left.push((Style::Dim, format!("  as of {}", snapshot.taken)));
    }
    if let Some(error) = &state.error {
        left.push((Style::Problem, format!("  {error}")));
    }
    let hints = if state.input.is_some() {
        vec![("⏎", "set"), ("esc", "cancel")]
    } else if state.clearing {
        vec![("x", "clear goal"), ("esc", "keep")]
    } else {
        vec![("↑↓", "move"), ("esc", "back")]
    };
    (out, Footer::new(left, hints))
}

/// A click on a row: it is selected.
pub(super) fn pick(tab: &mut Open, index: usize) {
    if tab.overview.input.is_none() {
        tab.overview.clearing = false;
        tab.overview.selected = index;
    }
}

pub(super) fn key(tab: &mut Open, key: &KeyEvent) -> Outcome {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('c' | 'C') if ctrl => return Outcome::Interrupt,
        KeyCode::Char('d' | 'D') if ctrl => return Outcome::Eof,
        _ => {}
    }
    if tab.overview.input.is_some() {
        typing(tab, key);
        return Outcome::Stay;
    }
    let clearing = std::mem::take(&mut tab.overview.clearing);
    let Some(snapshot) = tab.overview.snapshot.clone() else {
        return if key.code == KeyCode::Esc { Outcome::Closed } else { Outcome::Stay };
    };
    let all = rows(&snapshot);
    let at = tab.overview.selected.min(all.len().saturating_sub(1));
    let row = all.get(at).cloned();
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => {
            if clearing {
                tab.note("goal kept", false);
                return Outcome::Stay;
            }
            return Outcome::Closed;
        }
        KeyCode::Up => tab.overview.selected = at.saturating_sub(1),
        KeyCode::Down => tab.overview.selected = (at + 1).min(all.len().saturating_sub(1)),
        KeyCode::Home => tab.overview.selected = 0,
        KeyCode::End => tab.overview.selected = all.len().saturating_sub(1),
        KeyCode::Left | KeyCode::Right | KeyCode::Enter if row == Some(Row::Reasoning) => {
            let now = snapshot.reasoning.rsplit(' ').next().unwrap_or("auto");
            let index = MODES.iter().position(|mode| *mode == now).unwrap_or(0);
            let next = if key.code == KeyCode::Left { (index + MODES.len() - 1) % MODES.len() } else { (index + 1) % MODES.len() };
            send(tab, format!("/mode {}", MODES[next]));
        }
        KeyCode::Enter | KeyCode::Char(' ') if row == Some(Row::Loop) => {
            send(tab, format!("/loop {}", if snapshot.looping { "off" } else { "on" }));
        }
        KeyCode::Enter | KeyCode::Char(' ') if row == Some(Row::Voice) => {
            send(tab, format!("/voice {}", if snapshot.voice_on { "off" } else { "on" }));
        }
        KeyCode::Enter | KeyCode::Char(' ') if row == Some(Row::Vm) => {
            send(tab, format!("/vm {}", if snapshot.vm_access { "off" } else { "on" }));
        }
        KeyCode::Left | KeyCode::Right | KeyCode::Enter if row == Some(Row::WorkerReasoning) => {
            let levels = &MODES[1..];
            let now = if snapshot.worker_reasoning.is_empty() { "medium" } else { snapshot.worker_reasoning.as_str() };
            let index = levels.iter().position(|level| *level == now).unwrap_or(2);
            let next = if key.code == KeyCode::Left { (index + levels.len() - 1) % levels.len() } else { (index + 1) % levels.len() };
            send(tab, format!("/mode workers {}", levels[next]));
        }
        KeyCode::Enter if row == Some(Row::Sleep) => {
            let text: Vec<char> = snapshot.sleep_at.clone().unwrap_or_default().replace(':', "").chars().collect();
            tab.overview.input = Some(Input { field: Field::Sleep, cursor: text.len(), text });
        }
        KeyCode::Enter if row == Some(Row::Tz) => {
            tab.overview.input = Some(Input { field: Field::Tz, text: Vec::new(), cursor: 0 });
        }
        KeyCode::Char('p') if row == Some(Row::Goal) && snapshot.goal_set => send(tab, "/goal pause".into()),
        KeyCode::Char('r') if row == Some(Row::Goal) && snapshot.goal_set => send(tab, "/goal resume".into()),
        KeyCode::Char('x') if row == Some(Row::Goal) && snapshot.goal_set => {
            if clearing {
                send(tab, "/goal clear".into());
            } else {
                tab.overview.clearing = true;
                tab.ask("x again clears the goal");
            }
        }
        KeyCode::Enter | KeyCode::Char('e') if row == Some(Row::McpConfig) => edit_config(tab, false),
        KeyCode::Char('a') if row == Some(Row::McpConfig) => edit_config(tab, true),
        KeyCode::Char('x') if row == Some(Row::McpConfig) && tab.overview.draft.is_some() => {
            tab.overview.draft = None;
            tab.note("your edit is dropped · mcp.json is as it was", false);
        }
        KeyCode::Enter => {
            if let Some(Row::Server(name)) = row {
                send(tab, format!("/mcp reload {name}"));
            }
        }
        _ => {}
    }
    Outcome::Stay
}

fn typing(tab: &mut Open, key: &KeyEvent) {
    let Some(input) = tab.overview.input.as_mut() else { return };
    match key.code {
        KeyCode::Esc => tab.overview.input = None,
        KeyCode::Enter => {
            let input = tab.overview.input.take().expect("typed");
            let text: String = input.text.iter().collect::<String>().trim().to_string();
            if text.is_empty() || text.contains(char::is_whitespace) {
                return tab.note("one word: 0200, UTC-6, America/Denver, or off", true);
            }
            send(
                tab,
                match input.field {
                    Field::Sleep => format!("/sleep set {text}"),
                    Field::Tz => format!("/tz {text}"),
                },
            );
        }
        KeyCode::Backspace if input.cursor > 0 => {
            input.cursor -= 1;
            input.text.remove(input.cursor);
        }
        KeyCode::Delete if input.cursor < input.text.len() => {
            input.text.remove(input.cursor);
        }
        KeyCode::Left => input.cursor = input.cursor.saturating_sub(1),
        KeyCode::Right => input.cursor = (input.cursor + 1).min(input.text.len()),
        KeyCode::Home => input.cursor = 0,
        KeyCode::End => input.cursor = input.text.len(),
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            input.text.insert(input.cursor, c);
            input.cursor += 1;
        }
        _ => {}
    }
}

pub(super) fn paste(tab: &mut Open, text: &str) {
    if let Some(input) = tab.overview.input.as_mut() {
        for c in text.chars().filter(|c| !c.is_control()) {
            input.text.insert(input.cursor, c);
            input.cursor += 1;
        }
    }
}

// ------------------------------------------------------------------ mcp.json

const CONFIG_HELP: &str = "\
mcp.json lists the MCP servers s0ma connects to, in the mcpServers shape Claude Desktop and Claude Code read: a server's block copied from either works.
  \"notes\": { \"command\": \"npx\", \"args\": [\"-y\", \"@modelcontextprotocol/server-filesystem\", \"/home/operator/notes\"] }
  \"remote\": { \"type\": \"http\", \"url\": \"https://example.com/mcp\", \"headers\": { \"Authorization\": \"Bearer ${EXAMPLE_TOKEN}\" } }
A command runs inside s0ma's VM (its paths are the VM's) unless it says \"location\": \"host\". ${VAR} reads s0ma's environment, so a secret need not sit in the file. \"tools\" or \"exclude_tools\" pick which tools s0ma sees.
e edits it, a adds a server's block to fill in: what you save is checked, written, and MCP reloaded.";

/// The block `a` puts in, for the operator to fill.
const NEW_SERVER: &str = "\"command\": \"npx\",\n      \"args\": [\"-y\", \"package-name\"]";

/// `text` with a new server's block first in `mcpServers` (made if there is
/// none), the rest of the file as the operator wrote it, and the name given.
fn with_new_server(text: &str) -> Result<(String, String), String> {
    let block = |name: &str| format!("\"{name}\": {{\n      {NEW_SERVER}\n    }}");
    if text.trim().is_empty() {
        let name = "new-server".to_string();
        return Ok((format!("{{\n  \"mcpServers\": {{\n    {}\n  }}\n}}\n", block(&name)), name));
    }
    let value: serde_json::Value = serde_json::from_str(text).map_err(|error| format!("mcp.json does not read as JSON yet ({error}): Edit opens it to fix"))?;
    let Some(top) = value.as_object() else {
        return Err("mcp.json is not a JSON object: Edit opens it to fix".into());
    };
    let servers = top.get("mcpServers").and_then(|servers| servers.as_object());
    let name = (1..)
        .map(|n| if n == 1 { "new-server".to_string() } else { format!("new-server-{n}") })
        .find(|name| servers.is_none_or(|servers| !servers.contains_key(name)))
        .expect("a free name");
    // Put in as text, so the file keeps the operator's order and layout.
    let (at, insert) = match servers {
        Some(servers) => {
            let key = text.find("\"mcpServers\"").ok_or("mcp.json's mcpServers is written oddly: e opens it")?;
            let open = key + text[key..].find('{').ok_or("mcp.json's mcpServers is written oddly: e opens it")? + 1;
            // After the others, a comma; in an empty `{}`, the brace on its
            // own line after it.
            let after = if servers.is_empty() { "\n  " } else { "," };
            (open, format!("\n    {}{after}", block(&name)))
        }
        None => {
            let open = text.find('{').expect("an object") + 1;
            let comma = if top.is_empty() { "" } else { "," };
            (open, format!("\n  \"mcpServers\": {{\n    {}\n  }}{comma}", block(&name)))
        }
    };
    let mut out = text.to_string();
    out.insert_str(at, &insert);
    Ok((out, name))
}

#[derive(Debug, PartialEq, Eq)]
enum Saved {
    Unchanged,
    /// Not as s0ma reads it: nothing written.
    Invalid(String),
    /// Changed on disk while the operator edited: nothing written.
    Moved,
    Written,
}

/// The operator's edit, `after`, of what was on disk (`on_disk`, empty when there
/// was no file) opened as `started`: checked as s0ma reads it and written
/// whole (a file beside, then renamed over), keeping the file's permissions
/// (a new one is the operator's only: it may hold secrets).
fn save_config(path: &std::path::Path, on_disk: &str, started: &str, after: &str) -> Saved {
    if super::share_view::same(started, after) && started == on_disk {
        return Saved::Unchanged;
    }
    if let Err(error) = crate::mcp::check_config(after) {
        return Saved::Invalid(error.trim_start_matches("mcp.json is not valid: ").to_string());
    }
    let now = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Saved::Invalid(format!("cannot read it again: {error}")),
    };
    if now != on_disk {
        return Saved::Moved;
    }
    let beside = path.with_extension("json.s0ma-edit");
    let written = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&beside, after)?;
        let permissions = match std::fs::metadata(path) {
            Ok(metadata) => metadata.permissions(),
            Err(_) => std::os::unix::fs::PermissionsExt::from_mode(0o600),
        };
        std::fs::set_permissions(&beside, permissions)?;
        std::fs::rename(&beside, path)
    })();
    match written {
        Ok(()) => Saved::Written,
        Err(error) => {
            let _ = std::fs::remove_file(&beside);
            Saved::Invalid(format!("could not write it: {error}"))
        }
    }
}

/// `e` (or `a`, with a new server's block put in): `mcp.json` in the operator's
/// editor, or his unsaved edit again; saved, MCP reloads.
fn edit_config(tab: &mut Open, add: bool) {
    let Some(path) = tab.overview.snapshot.as_ref().and_then(|snapshot| snapshot.mcp_config.clone()).map(std::path::PathBuf::from) else {
        return tab.note("MCP is not set up in this s0ma", true);
    };
    let on_disk = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return tab.note(format!("cannot read {}: {error}", path.display()), true),
    };
    let draft = tab.overview.draft.take().map(|(text, _)| text);
    let base = draft.clone().unwrap_or_else(|| on_disk.clone());
    let (started, added) = if add || base.trim().is_empty() {
        match with_new_server(&base) {
            Ok((text, name)) => (text, Some(name)),
            Err(error) => {
                tab.overview.draft = draft.map(|text| (text, error.clone()));
                return tab.note(error, true);
            }
        }
    } else {
        (base, None)
    };
    let after = match super::share_view::in_editor(tab, "mcp.json", &started) {
        Ok(Some(after)) => after,
        Ok(None) => {
            tab.overview.draft = draft.map(|text| (text, "the editor failed".into()));
            return tab.note("editor failed · nothing saved", true);
        }
        Err(error) => {
            tab.overview.draft = draft.map(|text| (text, error.clone()));
            return tab.note(error, true);
        }
    };
    if added.is_some() && super::share_view::same(&started, &after) && draft.is_none() {
        return tab.note("nothing added · mcp.json is as it was", false);
    }
    match save_config(&path, &on_disk, &started, &after) {
        Saved::Unchanged => tab.note("unchanged", false),
        Saved::Invalid(why) => {
            tab.note(format!("not saved: {} · e opens your edit again", clip(&why, 160)), true);
            tab.overview.draft = Some((after, clip(&why, 80)));
        }
        Saved::Moved => {
            tab.note("mcp.json changed while you edited · not saved: e opens your edit again", true);
            tab.overview.draft = Some((after, "mcp.json changed meanwhile".into()));
        }
        Saved::Written => send(tab, "/mcp reload".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overview::{Server, publish, take_commands};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn text(lines: &[Line]) -> String {
        lines.iter().map(|line| line.iter().map(|(_, text)| text.as_str()).collect::<String>()).collect::<Vec<_>>().join("\n")
    }

    use crate::overview::tests::LOCK as QUEUE;

    fn snapshot() -> Snapshot {
        Snapshot {
            taken: "12:00:00".into(),
            reasoning: "forced xhigh".into(),
            looping: false,
            voice_on: false,
            voice: "listening=false tts=configured bridge=/x".into(),
            sleep_at: Some("02:00".into()),
            sleep: "a sleep cycle runs every day at 02:00.".into(),
            tz: "The operator's time zone is UTC-6 (set with /tz)".into(),
            tz_set: true,
            tz_zone: Some("UTC-6".into()),
            goal: "[goal] active: ship the share tab\n       3 turns so far".into(),
            goal_set: true,
            jobs: "[job] no open job".into(),
            mcp: vec![Server {
                name: "github".into(),
                state: "connected".into(),
                detail: "stdio · github-mcp · 2/2 tools".into(),
                tools: vec!["search_issues  ~300 tokens".into(), "get_pr  ~200 tokens".into()],
            }],
            mcp_notes: vec!["config: /x/mcp.json".into()],
            context: Some(Ok((84_213, 262_144))),
            ..Snapshot::default()
        }
    }

    fn open() -> Open {
        publish(snapshot());
        let mut tab = Open::new(super::super::View::Overview);
        tab.overview.asked = None;
        tab
    }

    #[test]
    fn a_server_is_added_first_in_mcp_servers_and_the_rest_is_kept_as_written() {
        let servers = |text: &str| {
            crate::mcp::check_config(text).unwrap_or_else(|error| panic!("{error}: {text}"));
            let value: serde_json::Value = serde_json::from_str(text).unwrap();
            value["mcpServers"].as_object().unwrap().keys().cloned().collect::<Vec<_>>()
        };
        // No file yet: one made, with the block.
        let (made, name) = with_new_server("").unwrap();
        assert_eq!((servers(&made), name.as_str()), (vec!["new-server".to_string()], "new-server"));
        // Some servers: the block before them, the file otherwise untouched.
        let file = "{\n  \"vm\": {\"address\": \"10.9.9.2\"},\n  \"mcpServers\": {\n    \"zeta\": {\"command\": \"z\"},\n    \"alpha\": {\"command\": \"a\"}\n  }\n}\n";
        let (added, _) = with_new_server(file).unwrap();
        assert_eq!(servers(&added), ["alpha", "new-server", "zeta"]);
        assert!(added.contains("\"zeta\": {\"command\": \"z\"},\n    \"alpha\""), "The operator's order and layout kept: {added}");
        assert!(added.find("new-server").unwrap() < added.find("zeta").unwrap(), "first, where the editor opens");
        // Taken names are not reused; an empty list and no list both work.
        let (again, name) = with_new_server(&added).unwrap();
        assert_eq!((servers(&again).len(), name.as_str()), (4, "new-server-2"));
        let (into_empty, _) = with_new_server("{\n  \"mcpServers\": {}\n}\n").unwrap();
        assert_eq!(into_empty, "{\n  \"mcpServers\": {\n    \"new-server\": {\n      \"command\": \"npx\",\n      \"args\": [\"-y\", \"package-name\"]\n    }\n  }\n}\n");
        assert_eq!(servers(&with_new_server("{\"vm\": {}}").unwrap().0), ["new-server"]);
        assert_eq!(servers(&with_new_server("{}").unwrap().0), ["new-server"]);
        // A file that does not read is opened to fix, not added to.
        assert!(with_new_server("{\"mcpServers\": {").unwrap_err().contains("Edit opens it to fix"));
        assert!(with_new_server("[1]").is_err());
    }

    #[test]
    fn an_edit_is_written_only_when_it_reads_and_nobody_changed_the_file_meanwhile() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("s0ma-mcp-edit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("mcp.json");
        let good = "{\"mcpServers\": {\"notes\": {\"command\": \"npx\"}}}";
        // A new file: made, the operator's only.
        assert_eq!(save_config(&path, "", "", good), Saved::Written);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), good);
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        // An existing file keeps its permissions.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        let better = good.replace("notes", "docs");
        assert_eq!(save_config(&path, good, good, &better), Saved::Written);
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o640);
        // Saved as opened: nothing to do (an editor's final newline is no change).
        assert_eq!(save_config(&path, &better, &better, &format!("{better}\n")), Saved::Unchanged);
        // Does not read: not written.
        assert!(matches!(save_config(&path, &better, &better, "{\"mcpServers\": {"), Saved::Invalid(_)));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), better);
        // Changed on disk while the operator edited: his is not written over it.
        assert_eq!(save_config(&path, good, good, &better.replace("docs", "mine")), Saved::Moved);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), better);
        // His kept edit, opened again and saved as it was: now written.
        let mine = better.replace("docs", "mine");
        assert_eq!(save_config(&path, &better, &mine, &mine), Saved::Written);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), mine);
        assert!(!dir.join("mcp.json.s0ma-edit").exists(), "nothing left beside it");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_mcp_json_row_is_there_with_no_servers_and_an_unsaved_edit_can_be_dropped() {
        let _queue = QUEUE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        publish(Snapshot { mcp: Vec::new(), mcp_config: Some("/x/mcp.json".into()), ..snapshot() });
        let mut tab = Open::new(super::super::View::Overview);
        tab.overview.asked = None;
        let _ = body(&mut tab, 110, 40);
        let _ = super::key(&mut tab, &key(KeyCode::End));
        let (lines, _) = body(&mut tab, 110, 40);
        let shown = text(&lines);
        assert!(shown.contains("mcp.json") && shown.contains("no servers yet") && shown.contains(" Add a server "), "{shown}");
        assert!(shown.contains("mcpServers shape") && shown.contains("config: /x/mcp.json"), "the help and notes in the strip: {shown}");
        tab.overview.draft = Some(("{".into(), "EOF while parsing".into()));
        let (lines, _) = body(&mut tab, 110, 40);
        assert!(text(&lines).contains("your edit is not saved: EOF while parsing"));
        let _ = super::key(&mut tab, &key(KeyCode::Char('x')));
        assert!(tab.overview.draft.is_none());
        // No MCP in this s0ma: said, nothing opened.
        publish(Snapshot { mcp_config: None, ..snapshot() });
        tab.overview.asked = None;
        let _ = body(&mut tab, 110, 40);
        let _ = super::key(&mut tab, &key(KeyCode::Char('e')));
        assert!(tab.overview.draft.is_none());
    }

    #[test]
    fn the_overview_says_where_commands_run_and_what_is_left_running() {
        let _queue = QUEUE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let watches = crate::overview::Watches {
            running: vec![("bg-1a2b3c4d".into(), false, "cargo build --release".into()), ("log-error".into(), true, "tail -n0 -F app.log | grep -m1 ERROR".into())],
            ended: 1,
            error: None,
            listed: true,
        };
        publish(Snapshot { in_shells: true, watches: watches.clone(), ..snapshot() });
        let mut tab = Open::new(super::super::View::Overview);
        tab.overview.asked = None;
        let (lines, _) = body(&mut tab, 110, 40);
        let shown = text(&lines);
        assert!(shown.contains("commands") && shown.contains("in shells you see, or not run"), "{shown}");
        assert!(shown.contains("[*] 2 running: bg-1a2b3c4d, log-error"), "{shown}");
        let detail = watches_detail(&watches);
        assert!(detail.contains("log-error (repeats): tail -n0 -F app.log") && detail.contains("1 ended."), "{detail}");

        publish(Snapshot { in_shells: false, ..snapshot() });
        tab.overview.asked = None;
        let (lines, _) = body(&mut tab, 110, 40);
        assert!(text(&lines).contains("[!] outside shells"), "{}", text(&lines));
    }

    #[test]
    fn the_overview_shows_the_settings_s0ma_and_mcp() {
        let _queue = QUEUE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut tab = open();
        let (lines, _) = body(&mut tab, 100, 30);
        let shown = text(&lines);
        for expected in ["reasoning", "forced xhigh", "loop", "daily sleep", "02:00", "time zone", "UTC-6", "context", "84,213 of 262,144 tokens · 32%", "goal", "active: ship the share tab", "jobs", "github", "connected"] {
            assert!(shown.contains(expected), "{expected}: {shown}");
        }
        // Three quarters full: marked; not measured: said so.
        assert_eq!(context_value(Some(&Ok((200_000, 262_144))))[0].0, Style::Warn);
        assert_eq!(context_value(Some(&Ok((84_213, 262_144))))[0].0, Style::Plain);
        assert_eq!(context_value(Some(&Err("no server".into())))[0].1, "not measured");
        assert!(shown.contains("subagents") && shown.contains("medium"), "the subagents' level: {shown}");
        // Down to github: its tools in the strip.
        for _ in 0..12 {
            let _ = super::key(&mut tab, &key(KeyCode::Down));
        }
        let (lines, _) = body(&mut tab, 100, 30);
        let shown = text(&lines);
        assert!(shown.contains("search_issues") && shown.contains(" Reload "), "{shown}");
    }

    #[test]
    fn each_change_is_the_command_operator_would_have_typed() {
        let _queue = QUEUE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let _ = take_commands();
        let mut tab = open();
        let _ = body(&mut tab, 100, 30);
        let _ = super::key(&mut tab, &key(KeyCode::Right));
        // The subagents' level, from medium (not set) on to xhigh.
        let _ = super::key(&mut tab, &key(KeyCode::Down));
        let _ = super::key(&mut tab, &key(KeyCode::Right));
        let _ = super::key(&mut tab, &key(KeyCode::Down));
        let _ = super::key(&mut tab, &key(KeyCode::Enter));
        let _ = super::key(&mut tab, &key(KeyCode::Down));
        let _ = super::key(&mut tab, &key(KeyCode::Enter));
        // VM access: on, then the switch as it was (it takes effect at once).
        let _ = super::key(&mut tab, &key(KeyCode::Down));
        let _ = super::key(&mut tab, &key(KeyCode::Enter));
        assert!(crate::vm_access::allowed());
        crate::vm_access::set_for_test(true);
        // Daily sleep: typed.
        let _ = super::key(&mut tab, &key(KeyCode::Down));
        let _ = super::key(&mut tab, &key(KeyCode::Enter));
        let typed: String = tab.overview.input.as_ref().unwrap().text.iter().collect();
        assert_eq!(typed, "0200", "starts from what is set");
        for _ in 0..4 {
            let _ = super::key(&mut tab, &key(KeyCode::Backspace));
        }
        paste(&mut tab, "0330");
        let _ = super::key(&mut tab, &key(KeyCode::Enter));
        // The goal (past the time zone and the context): x twice clears;
        // once, and something else, keeps it.
        let _ = super::key(&mut tab, &key(KeyCode::Down));
        let _ = super::key(&mut tab, &key(KeyCode::Down));
        let _ = super::key(&mut tab, &key(KeyCode::Down));
        let _ = super::key(&mut tab, &key(KeyCode::Char('x')));
        let _ = super::key(&mut tab, &key(KeyCode::Esc));
        let _ = super::key(&mut tab, &key(KeyCode::Char('p')));
        let _ = super::key(&mut tab, &key(KeyCode::Char('x')));
        let _ = super::key(&mut tab, &key(KeyCode::Char('x')));
        // MCP: reload (the server is the row above mcp.json, the last).
        let _ = super::key(&mut tab, &key(KeyCode::End));
        let _ = super::key(&mut tab, &key(KeyCode::Up));
        let _ = super::key(&mut tab, &key(KeyCode::Enter));
        assert_eq!(
            take_commands(),
            ["/mode auto", "/mode workers xhigh", "/loop on", "/voice on", "/vm on", "/sleep set 0330", "/goal pause", "/goal clear", "/mcp reload github"]
        );
        assert_eq!(super::key(&mut tab, &key(KeyCode::Esc)), Outcome::Closed);
    }

    #[test]
    fn the_apps_reply_shows_once() {
        let _queue = QUEUE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut tab = open();
        let _ = body(&mut tab, 100, 30);
        let reply = crate::overview::Reply { seq: 900, command: "/tz nowhere".into(), text: "unknown zone".into(), problem: true };
        publish(Snapshot { reply: Some(reply), ..snapshot() });
        tab.overview.asked = None;
        let _ = body(&mut tab, 100, 30);
        assert!(tab.note.as_ref().is_some_and(|(text, problem, _)| text.contains("unknown zone") && *problem));
        tab.note = None;
        tab.overview.asked = None;
        let _ = body(&mut tab, 100, 30);
        assert!(tab.note.is_none(), "said once");
    }
}
