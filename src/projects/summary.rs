//! A board's or audit's page in the projects tab: made from its file (so an
//! archived one reads the same), as markdown.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use serde_json::{Map, Value};

use super::log;

/// Every row of a file as it stands, by table: the last state of each key,
/// in the order first seen.
pub fn rows(path: &Path) -> Option<HashMap<String, Vec<Map<String, Value>>>> {
    let log = log::read(path).ok()?;
    let mut order: Vec<(String, String)> = Vec::new();
    let mut state: HashMap<(String, String), Option<Map<String, Value>>> = HashMap::new();
    let mut put = |table: &str, row: Option<&Map<String, Value>>, key: String| {
        let id = (table.to_string(), key);
        if !state.contains_key(&id) {
            order.push(id.clone());
        }
        state.insert(id, row.cloned());
    };
    let key_of = |table: &str, row: &Map<String, Value>| -> String {
        let fields: &[&str] = match table {
            "boards" => &["id"],
            "facts" | "intents" | "hints" | "comments" => &["id"],
            "board_models" => &["model_id", "relation"],
            "flow" => &["board_id"],
            "notes" | "evidence" | "links" | "log" => &["seq"],
            "audits" => &["name"],
            "claims" => &["id"],
            "checks" => &["claim", "version"],
            "runs" => &["n"],
            _ => &[],
        };
        fields.iter().map(|f| row.get(*f).map(|v| v.to_string()).unwrap_or_default()).collect::<Vec<_>>().join("|")
    };
    for line in log.current() {
        if log::op(line) == "snapshot" {
            for item in line.get("rows").and_then(Value::as_array).into_iter().flatten() {
                if let (Some(table), Some(row)) = (item.get("table").and_then(Value::as_str), item.get("row").and_then(Value::as_object)) {
                    put(table, Some(row), key_of(table, row));
                }
            }
            continue;
        }
        let Some(table) = line.get("table").and_then(Value::as_str) else { continue };
        if let Some(row) = line.get("row").and_then(Value::as_object) {
            put(table, Some(row), key_of(table, row));
        } else if let Some(key) = line.get("key").and_then(Value::as_object) {
            put(table, None, key_of(table, key));
        }
    }
    let mut out: HashMap<String, Vec<Map<String, Value>>> = HashMap::new();
    for id in order {
        if let Some(Some(row)) = state.remove(&id) {
            out.entry(id.0).or_default().push(row);
        }
    }
    Some(out)
}

fn text(row: &Map<String, Value>, key: &str) -> String {
    row.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

fn int(row: &Map<String, Value>, key: &str) -> i64 {
    row.get(key).and_then(Value::as_i64).unwrap_or(0)
}

/// What a board's row in the tree and its page show.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoardInfo {
    pub id: String,
    pub name: String,
    pub title: String,
    pub status: String,
    pub stage: String,
    pub round: i64,
    pub steps_used: i64,
    pub budget: i64,
    pub goal: String,
    pub goal_test: String,
    pub open_leads: Vec<(String, String, i64)>,
    pub findings: Vec<(String, String, String)>,
    pub report: Option<(i64, String)>,
    /// Its comments, in threads.
    pub threads: Vec<crate::board::comments::Thread>,
}

/// Read once per version of the file.
type Seen<T> = HashMap<String, ((u64, i128), T)>;
static BOARDS: Mutex<Option<Seen<BoardInfo>>> = Mutex::new(None);

pub fn board(path: &Path) -> Option<BoardInfo> {
    let stamp = super::stamp(path)?;
    let key = path.to_string_lossy().into_owned();
    if let Some((seen, info)) = crate::util::locked(&BOARDS).as_ref().and_then(|m| m.get(&key)).cloned()
        && seen == stamp
    {
        return Some(info);
    }
    let rows = rows(path)?;
    let board = rows.get("boards").and_then(|r| r.last()).cloned().unwrap_or_default();
    let mut info = BoardInfo {
        id: text(&board, "id"),
        name: text(&board, "name"),
        title: text(&board, "title"),
        status: text(&board, "status"),
        steps_used: int(&board, "steps_used"),
        budget: int(&board, "step_budget"),
        goal: text(&board, "goal"),
        goal_test: text(&board, "goal_test"),
        stage: "working".into(),
        round: 1,
        ..BoardInfo::default()
    };
    if let Some(flow) = rows.get("flow").and_then(|r| r.last())
        && let Ok(flow) = serde_json::from_str::<crate::board::flow::Flow>(&text(flow, "body"))
    {
        info.stage = flow.stage().label().to_string();
        info.round = flow.round;
        info.report = flow.reports.last().map(|r| (r.round, r.text.clone())).or_else(|| flow.report.clone().map(|t| (flow.report_round, t)));
    }
    let mut leads: Vec<(String, String, i64)> = rows
        .get("intents")
        .into_iter()
        .flatten()
        .filter(|r| matches!(text(r, "state").as_str(), "open" | "claimed"))
        .map(|r| (text(r, "id"), text(r, "direction"), int(r, "priority")))
        .collect();
    leads.sort_by_key(|l| std::cmp::Reverse(l.2));
    info.open_leads = leads;
    info.findings = rows.get("facts").into_iter().flatten().rev().map(|r| (text(r, "id"), text(r, "statement"), text(r, "polarity"))).collect();
    let comments: Vec<crate::board::comments::Comment> = rows
        .get("comments")
        .into_iter()
        .flatten()
        .map(|r| crate::board::comments::Comment {
            id: text(r, "id"),
            board_id: text(r, "board_id"),
            on: text(r, "on_what"),
            round: r.get("round").and_then(Value::as_i64),
            author: text(r, "author"),
            parent: r.get("parent").and_then(Value::as_str).map(str::to_string),
            text: text(r, "text"),
            created_at: text(r, "created_at"),
            resolved_at: r.get("resolved_at").and_then(Value::as_str).map(str::to_string),
            resolved_by: r.get("resolved_by").and_then(Value::as_str).map(str::to_string),
            seen_at: None,
        })
        .collect();
    // Hints from before comments, as the operator's open comments.
    let older = rows.get("hints").into_iter().flatten().map(|r| crate::board::comments::Comment {
        id: text(r, "id"),
        board_id: text(r, "board_id"),
        on: "board".into(),
        round: None,
        author: crate::board::comments::OPERATOR.into(),
        parent: None,
        text: text(r, "text"),
        created_at: text(r, "created_at"),
        resolved_at: None,
        resolved_by: None,
        seen_at: None,
    });
    let mut all: Vec<crate::board::comments::Comment> = older.collect();
    all.extend(comments);
    info.threads = crate::board::comments::group(all);
    crate::util::locked(&BOARDS).get_or_insert_with(HashMap::new).insert(key, (stamp, info.clone()));
    Some(info)
}

/// What an audit's row and page show.
#[derive(Clone, Debug, Default)]
pub struct AuditInfo {
    pub name: String,
    pub board: Option<String>,
    pub status: String,
    pub targets: Vec<String>,
    pub claims: Vec<crate::audit::Claim>,
}

static AUDITS: Mutex<Option<Seen<AuditInfo>>> = Mutex::new(None);

pub fn audit(path: &Path) -> Option<AuditInfo> {
    let stamp = super::stamp(path)?;
    let key = path.to_string_lossy().into_owned();
    if let Some((seen, info)) = crate::util::locked(&AUDITS).as_ref().and_then(|m| m.get(&key)).cloned()
        && seen == stamp
    {
        return Some(info);
    }
    let rows = rows(path)?;
    let row = rows.get("audits").and_then(|r| r.last()).cloned().unwrap_or_default();
    let audit: Option<crate::audit::Audit> = serde_json::from_str(&text(&row, "data")).ok();
    let mut claims: Vec<(i64, crate::audit::Claim)> =
        rows.get("claims").into_iter().flatten().filter_map(|r| serde_json::from_str(&text(r, "data")).ok().map(|c| (int(r, "seq"), c))).collect();
    claims.sort_by_key(|(seq, _)| *seq);
    let claims: Vec<crate::audit::Claim> = claims.into_iter().map(|(_, c)| c).collect();
    let info = AuditInfo {
        name: text(&row, "name"),
        board: audit.as_ref().and_then(|a| a.board.clone()),
        status: audit.as_ref().map(|a| a.status.label().to_string()).unwrap_or_default(),
        targets: audit.map(|a| a.targets).unwrap_or_default(),
        claims,
    };
    crate::util::locked(&AUDITS).get_or_insert_with(HashMap::new).insert(key, (stamp, info.clone()));
    Some(info)
}

/// A board's page.
pub fn board_page(rel_path: &str, info: &BoardInfo, audits: &[(String, AuditInfo)]) -> String {
    let mut out = format!("# {}\n\n", if info.title.is_empty() { &info.name } else { &info.title });
    out.push_str(&format!("`{rel_path}` · {} · {} · round {} · {} of {} steps\n\n", info.status, info.stage, info.round, info.steps_used, info.budget));
    if !info.goal.is_empty() {
        out.push_str(&format!("**Goal** {}\n\n", info.goal));
    }
    if !info.goal_test.is_empty() {
        out.push_str(&format!("**Test** `{}`\n\n", info.goal_test));
    }
    if !info.open_leads.is_empty() {
        out.push_str(&format!("## Open leads · {}\n\n", info.open_leads.len()));
        for (id, direction, priority) in info.open_leads.iter().take(5) {
            out.push_str(&format!("- {id} p{priority} {direction}\n"));
        }
        out.push('\n');
    }
    if !info.findings.is_empty() {
        out.push_str(&format!("## Latest findings · {}\n\n", info.findings.len()));
        for (id, statement, polarity) in info.findings.iter().take(5) {
            out.push_str(&format!("- {id} ({polarity}) {statement}\n"));
        }
        out.push('\n');
    }
    if let Some((round, report)) = &info.report {
        let opening: String = report.lines().take(12).collect::<Vec<_>>().join("\n");
        out.push_str(&format!("## Report, round {round}\n\n{opening}\n\n"));
    }
    if !audits.is_empty() {
        out.push_str("## Audits\n\n");
        for (name, audit) in audits {
            out.push_str(&format!("- {name} · {} · {}\n", audit.status, counts(&audit.claims)));
        }
        out.push('\n');
    }
    out.push_str(&comments_section(&info.threads));
    out
}

/// Open comment threads as markdown: who, on what, the replies under it.
pub fn comments_section(threads: &[crate::board::comments::Thread]) -> String {
    let open: Vec<&crate::board::comments::Thread> = threads.iter().filter(|t| t.root.open()).collect();
    if open.is_empty() {
        return String::new();
    }
    let who = |author: &str| match author {
        crate::board::comments::OPERATOR => "you".to_string(),
        crate::board::comments::AGENT => "s0ma".to_string(),
        other => other.to_string(),
    };
    let waiting = open.iter().filter(|t| t.waits_on_operator()).count();
    let mut out = format!("## Comments · {}{}\n\n", open.len(), if waiting > 0 { format!(" · {waiting} for you") } else { String::new() });
    for thread in open {
        let mark = if thread.waits_on_operator() { "[!] " } else { "" };
        let round = thread.root.round.map(|r| format!(" · round {r}")).unwrap_or_default();
        let on = if thread.root.on == "board" || thread.root.on.ends_with(".md") { String::new() } else { format!(" on {}", thread.on_said()) };
        out.push_str(&format!("- {mark}**{}**{on}{round} `{}` {}\n", who(&thread.root.author), thread.root.id, thread.root.text));
        for reply in &thread.replies {
            out.push_str(&format!("  - **{}** {}\n", who(&reply.author), reply.text));
        }
    }
    out
}

fn counts(claims: &[crate::audit::Claim]) -> String {
    let mut by: Vec<(String, usize)> = Vec::new();
    for claim in claims {
        let label = claim.verdict.label().to_string();
        match by.iter_mut().find(|(l, _)| *l == label) {
            Some((_, n)) => *n += 1,
            None => by.push((label, 1)),
        }
    }
    if by.is_empty() {
        return "no claims yet".into();
    }
    by.iter().map(|(l, n)| format!("{n} {l}")).collect::<Vec<_>>().join(", ")
}

/// An audit's page.
pub fn audit_page(rel_path: &str, info: &AuditInfo) -> String {
    let mut out = format!("# Audit {}\n\n", info.name);
    out.push_str(&format!("`{rel_path}` · {}", info.status));
    if let Some(board) = &info.board {
        out.push_str(&format!(" · for {board}"));
    }
    out.push_str("\n\n");
    if !info.targets.is_empty() {
        out.push_str(&format!("**Checks** {}\n\n", info.targets.join(", ")));
    }
    if info.claims.is_empty() {
        out.push_str("No claims yet.\n");
        return out;
    }
    out.push_str(&format!("## Claims · {}\n\n", counts(&info.claims)));
    for claim in &info.claims {
        out.push_str(&format!("- {} {} · {}\n", claim.id, claim.text, claim.verdict.label()));
    }
    out
}
