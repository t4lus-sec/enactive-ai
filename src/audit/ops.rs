//! What can be done to an audit, by whom: the agent proposes and starts within
//! its project's policy; the operator starts, steers, takes, waives, posts
//! and stops; only the harness writes verdicts and posts results to boards
//! (`post_result`). Also links between audits and boards, the context block,
//! and the events that need the agent.

use crate::audit::{
    self, Audit, AuditStore, Claim, Gate, Level, Link, Mode, Policy, Post, Posted, Repro, Review, Status, Verdict, Who,
};

use chrono::Local;
use serde_json::json;

use std::sync::Mutex;

type Reply = std::result::Result<String, String>;

fn err<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

// ------------------------------------------------------------------ events for the agent

/// Events that need the agent: they start a turn (a check sent back, a claim
/// edited, a question answered, a new failure in a standing audit, an
/// unlink). Drained by the app.
static TURN: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// Events that need no work: they ride with the operator's next message.
static NOTES: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub fn needs_s0ma(text: String) {
    crate::util::locked(&TURN).push(text);
}

pub fn note(text: String) {
    crate::util::locked(&NOTES).push(text);
}

/// The events that start a turn, as one message.
pub fn take_turn() -> Option<String> {
    let events = std::mem::take(&mut *crate::util::locked(&TURN));
    (!events.is_empty()).then(|| format!("[AUDIT EVENTS — from the operator and the harness]\n{}", events.join("\n")))
}

/// The quiet ones, for the operator's next message.
pub fn take_note() -> Option<String> {
    let notes = std::mem::take(&mut *crate::util::locked(&NOTES));
    (!notes.is_empty()).then(|| format!("[audits] {}", notes.join("; ")))
}

// ------------------------------------------------------------------ starting

/// What `audit_start` and `/audit start` ask for.
#[derive(Clone, Debug, Default)]
pub struct StartAsk {
    pub name: String,
    pub board: Option<String>,
    pub targets: Vec<String>,
    pub scope: Vec<String>,
    pub settings: audit::Settings,
    pub why: String,
    pub expected_claims: Option<usize>,
}

/// The agent's `audit_start`: under the project's policy, a proposal (`ask`), a
/// start within its ceilings (`auto`), or refused (`off`).
pub fn s0ma_start(store: &AuditStore, ask: StartAsk) -> Reply {
    audit::valid_name(&ask.name)?;
    let Some(board) = ask.board.clone() else {
        return Err("an audit of yours is for a project: name its board (a standing audit is the operator's to start)".into());
    };
    let boards = audit::boards().ok_or("the board store is not ready")?;
    let Some(found) = boards.board(&board).map_err(err)? else { return Err(format!("no board named {board}")) };
    if found.status != crate::board::BoardStatus::Open {
        return Err(format!("{board} is not open ({})", found.status.label()));
    }
    let policy = store.policy(&board).map_err(err)?;
    let ceilings = store.ceilings(&board).map_err(err)?;
    let estimate = estimate(&ask, found.steps_remaining());
    let mut audit = new_audit(&ask, Mode::Diagnose);
    audit.estimate = estimate.clone();
    match policy {
        Policy::Off => Err(format!(
            "refused: the operator has audits off for {board} (/board audits {board} off). Work the board without one, or ask him."
        )),
        Policy::Auto
            if audit.settings.workers <= ceilings.workers
                && audit.settings.steps.unwrap_or(found.steps_remaining() / 3) <= ceilings.steps
                && audit.settings.depth.unwrap_or(Level::Bounded) <= ceilings.depth =>
        {
            store.create(&audit)?;
            store.log(&audit.name, Who::S0ma, "started it (policy auto, within its ceilings)", &ask.why).map_err(err)?;
            start(store, &audit.name, Who::S0ma)
        }
        _ => {
            store.create(&audit)?;
            store.log(&audit.name, Who::S0ma, "proposed it", &ask.why).map_err(err)?;
            crate::terminal::say_status(&format!(
                "[audit] s0ma proposes {} for {board}: {} · Alt+U, a twice starts it",
                audit.name,
                crate::util::clip(&ask.why, 120)
            ));
            Ok(json!({
                "ok": true,
                "proposed": audit.name,
                "note": format!("Filed as a proposal on {board}: nothing runs until the operator starts it (policy {}). \
                    It shows on the board as [!] audit proposed.", policy.label()),
                "estimate": estimate,
            })
            .to_string())
        }
    }
}

fn new_audit(ask: &StartAsk, mode: Mode) -> Audit {
    Audit {
        name: ask.name.clone(),
        board: ask.board.clone(),
        mode,
        status: Status::Proposed,
        targets: ask.targets.clone(),
        scope: ask.scope.clone(),
        settings: ask.settings.clone(),
        why: ask.why.clone(),
        created_at: audit::now(),
        ..Audit::default()
    }
}

fn estimate(ask: &StartAsk, steps_left: i64) -> String {
    let claims = ask.expected_claims.unwrap_or(20);
    let steps = ask.settings.steps.unwrap_or(steps_left / 3);
    let minutes = (claims as u64 * ask.settings.check_secs.min(600) / 60 / ask.settings.workers.max(1) as u64).max(5);
    format!(
        "about {claims} claims · {steps} steps · ~{} · {} worker slots",
        audit::show_secs((minutes * 60).min(ask.settings.total_secs)),
        ask.settings.workers
    )
}

/// The operator's `/audit start`: with `--for`, attached to that project in
/// diagnose mode; without, a project of its own whose goal is the audit, in
/// assure mode. Starts at once (he started it).
pub fn operator_start(store: &AuditStore, ask: StartAsk) -> Reply {
    audit::valid_name(&ask.name)?;
    if ask.targets.is_empty() {
        return Err("/audit start <name> <path>… [--for <board>] [--scope <spec>…]".into());
    }
    let boards = audit::boards().ok_or("the board store is not ready")?;
    let (board, mode) = match &ask.board {
        Some(board) => {
            if boards.board(board).map_err(err)?.is_none() {
                return Err(format!("no board named {board}"));
            }
            (board.clone(), Mode::Diagnose)
        }
        None => {
            let scope = if ask.scope.is_empty() { ask.targets.join(" ") } else { ask.scope.join(" ") };
            let opened = boards
                .open_board(
                    &ask.name,
                    &format!("audit {}", ask.name),
                    &format!("The operator started an audit of {}.", ask.targets.join(", ")),
                    &format!("audit {scope}"),
                    &format!("audit-run {} --all-checked", ask.name),
                    crate::board::GoalTestKind::Command,
                    crate::board::DEFAULT_STEP_BUDGET,
                )
                .map_err(err)??;
            (opened.name, Mode::Assure)
        }
    };
    let mut audit = new_audit(&StartAsk { board: Some(board), ..ask.clone() }, mode);
    audit.why = if ask.why.is_empty() { "the operator started it".into() } else { ask.why.clone() };
    store.create(&audit)?;
    store.log(&audit.name, Who::Operator, "started it", &audit.why).map_err(err)?;
    start(store, &audit.name, Who::Operator)
}

/// From proposed to harvesting: the lead on its board, its work tree, its
/// shell, and the ceiling preview. Work on the agent's machine runs on a thread.
pub fn start(store: &AuditStore, name: &str, who: Who) -> Reply {
    let boards = audit::boards().ok_or("the board store is not ready")?;
    let audit = store
        .change(name, |audit| {
            if !matches!(audit.status, Status::Proposed | Status::Standing | Status::Done | Status::Stopped) {
                return Err(format!("{name} is {} already", audit.status.label()));
            }
            audit.status = Status::Harvesting;
            audit.started_at = Some(audit::now());
            audit.gone_at = None;
            Ok(audit.clone())
        })
        .map_err(err)??;
    // Its steps: its own cap, within the project's budget.
    if let Some(board_name) = &audit.board
        && let Some(board) = boards.board(board_name).map_err(err)?
    {
        let cap = audit.settings.steps.unwrap_or((board.steps_remaining() / 3).max(1)).min(board.steps_remaining());
        let lead = ensure_lead(&boards, &board, &audit)?;
        store
            .change(name, |audit| {
                audit.step_cap = cap;
                audit.lead = Some(lead.clone());
                Ok(())
            })
            .map_err(err)??;
        store
            .add_link(&Link {
                audit: name.into(),
                board: board_name.clone(),
                kind: "owner".into(),
                lead: Some(lead),
                by: who.label().into(),
                at: audit::now(),
                ..Link::default()
            })
            .map_err(err)?;
    }
    crate::audit::run::prepare_in_background(name.to_string());
    Ok(format!(
        "[audit] {name} started: its shell is audit·{name}, and harvest begins. Nothing is checked until Go on the plan (g){}",
        if audit.settings.gate == Gate::None { " — the gate is none, so the plan is checked as it comes" } else { "" }
    ))
}

/// The audit's lead on its owner board: made and held by `audit:<name>`.
fn ensure_lead(boards: &crate::board::BoardStore, board: &crate::board::Board, audit: &Audit) -> std::result::Result<String, String> {
    if let Some(lead) = &audit.lead
        && boards.intent(&board.id, lead).map_err(err)?.is_some()
    {
        let _ = boards.renew_claim(&board.id, lead, &holder(&audit.name));
        return Ok(lead.clone());
    }
    let scope = if audit.scope.is_empty() { audit.targets.join(", ") } else { audit.scope.join(", ") };
    let direction = format!("audit {} — check what {scope} claims", audit.name);
    let batch = [(direction, format!("the harness runs audit {} for this project", audit.name), Vec::new(), 6)];
    let intents = match boards.declare_intents(&board.id, &batch, usize::MAX, Local::now()).map_err(err)? {
        Ok(intents) => intents,
        Err(_) => return Err("the lead could not be written".into()),
    };
    let lead = intents.first().ok_or("the lead could not be written")?.id.clone();
    let _ = boards.claim(&board.id, &lead, &holder(&audit.name), Local::now()).map_err(err)?;
    Ok(lead)
}

/// Who holds an audit's lead.
pub fn holder(name: &str) -> String {
    format!("audit:{name}")
}

/// The operator declines a proposal, with a reason the agent hears.
pub fn decline(store: &AuditStore, name: &str, why: &str) -> Reply {
    store
        .change(name, |audit| {
            if audit.status != Status::Proposed {
                return Err(format!("{name} is not a proposal"));
            }
            audit.status = Status::Declined;
            Ok(())
        })
        .map_err(err)??;
    store.log(name, Who::Operator, "declined it", why).map_err(err)?;
    needs_s0ma(format!("· The operator declined your proposed audit {name}{}", if why.is_empty() { ".".into() } else { format!(": {why}") }));
    Ok(format!("[audit] {name} declined"))
}

// ------------------------------------------------------------------ the operator's controls

/// The operator's change to a claim, by key. Each is logged.
pub fn operator_claim(store: &AuditStore, audit: &str, id: &str, action: &str, text: &str) -> Reply {
    let a = store.audit(audit).map_err(err)?.ok_or(format!("no audit {audit}"))?;
    let said = store
        .change_claim(audit, id, |claim| {
            Ok(match action {
                "accept" => {
                    if a.mode == Mode::Assure && !claim.readings.is_empty() && claim.picked.is_none() {
                        return Err(format!("{id} has an open question: pick a reading (1–{}) first", claim.readings.len()));
                    }
                    claim.review = Review::Accepted;
                    format!("accepted {id}")
                }
                "reject" => {
                    claim.review = Review::Rejected;
                    format!("rejected {id}")
                }
                "drop" => {
                    claim.review = Review::Dropped;
                    format!("dropped {id}")
                }
                "edit" => {
                    if text.trim().len() < 8 {
                        return Err("a claim needs a few words".into());
                    }
                    claim.text = audit::clip_text(text);
                    claim.author = "operator".into();
                    claim.review = Review::Accepted;
                    format!("reworded {id}: his wording is final")
                }
                "pick" => {
                    let n: usize = text.parse().map_err(|_| "a reading by number".to_string())?;
                    if n == 0 || n > claim.readings.len() {
                        return Err(format!("{id} has readings 1–{}", claim.readings.len()));
                    }
                    claim.picked = Some(n - 1);
                    format!("picked reading {n} of {id}: {}", claim.readings[n - 1])
                }
                "accept-weaker" => {
                    if claim.weaker.is_none() && !(claim.repro == Repro::ModelOnly && !claim.model_only_accepted) {
                        return Err(format!("{id} has nothing waiting to be accepted"));
                    }
                    if claim.weaker.is_some() {
                        claim.weaker = None;
                        "accepted the weaker check of ".to_string() + id
                    } else {
                        claim.model_only_accepted = true;
                        format!("accepted why {id} is model-only")
                    }
                }
                "back" => {
                    claim.weaker = None;
                    claim.verdict = Verdict::Unchecked;
                    claim.level = Level::Documented;
                    format!("sent {id}'s check back: {text}")
                }
                "take" => {
                    claim.taken = !claim.taken;
                    claim.held_by = None;
                    if claim.taken { format!("took {id}: he writes its check") } else { format!("gave {id} back to s0ma") }
                }
                "waive" => {
                    if !claim.waived && claim.verdict != Verdict::Fails {
                        return Err(format!("{id} is not failing: only a failing claim is waived"));
                    }
                    claim.waived = !claim.waived;
                    if claim.waived { format!("waived {id}: known, not fixing now") } else { format!("lifted the waiver on {id}") }
                }
                "pin" => {
                    claim.pinned = !claim.pinned;
                    format!("{} {id}", if claim.pinned { "pinned" } else { "unpinned" })
                }
                "skip" => {
                    claim.skipped = !claim.skipped;
                    format!("{} {id}", if claim.skipped { "skipped" } else { "unskipped" })
                }
                "priority" => {
                    let p: i64 = text.parse().ok().filter(|p| (1..=10).contains(p)).ok_or("p1 to p10")?;
                    claim.priority = p;
                    format!("{id} at p{p}")
                }
                "depth" => {
                    let ceiling = crate::audit::adapters::ceiling_for(&a.languages);
                    let now = claim.depth.unwrap_or(Level::of_kind(&claim.tool));
                    let next = now.deeper(a.settings.depth.unwrap_or(ceiling).min(ceiling), a.settings.allows("tla"));
                    claim.depth = Some(next);
                    format!("{id} at depth {}", next.label())
                }
                "post" => {
                    if claim.posted != Posted::Ready {
                        return Err(format!("{id} has no result waiting to be posted"));
                    }
                    format!("posted {id}")
                }
                "discard" => {
                    if claim.posted != Posted::Ready {
                        return Err(format!("{id} has no result waiting"));
                    }
                    claim.posted = Posted::Discarded;
                    format!("discarded {id}'s result")
                }
                "steer" => {
                    claim.steer = Some(text.to_string());
                    format!("a direction for {id}'s worker: {text}")
                }
                other => return Err(format!("unknown action {other}")),
            })
        })
        .map_err(err)??;
    store.log(audit, Who::Operator, &said, text).map_err(err)?;
    // What the agent needs to hear, now or with his next message.
    match action {
        "edit" => needs_s0ma(format!("· {audit}/{id}: the operator reworded it — re-formalize: \"{}\"", crate::util::clip(text, 200))),
        "pick" => needs_s0ma(format!("· {audit}/{id}: the operator answered its question ({said})")),
        "back" => needs_s0ma(format!("· {audit}/{id}: the operator sent its check back: {text}")),
        "steer" => note(format!("{audit}/{id}: a direction for its worker: {text}")),
        "post" => {
            post_result(store, audit, id, true).map_err(err)?;
        }
        _ => note(format!("{audit}: {said}")),
    }
    Ok(format!("[audit] {said}"))
}

/// Go on the plan: checking begins, and rows added after this wait for
/// another Go.
pub fn go(store: &AuditStore, name: &str) -> Reply {
    let now = audit::now();
    store
        .change(name, |audit| {
            if !matches!(audit.status, Status::Harvesting | Status::Checking | Status::Done | Status::Standing) {
                return Err(format!("{name} is {}: nothing to go on", audit.status.label()));
            }
            audit.gone_at = Some(now.clone());
            audit.status = Status::Checking;
            Ok(())
        })
        .map_err(err)??;
    for claim in store.claims(name).map_err(err)? {
        if claim.new {
            store.change_claim(name, &claim.id, |c| {
                c.new = false;
                Ok(())
            })
            .map_err(err)??;
        }
    }
    store.log(name, Who::Operator, "pressed go on the plan", "").map_err(err)?;
    crate::audit::run::wake();
    Ok(format!("[audit] {name}: go — checking the plan"))
}

pub fn pause(store: &AuditStore, name: &str) -> Reply {
    let paused = store
        .change(name, |audit| {
            audit.paused = !audit.paused;
            if !audit.paused {
                audit.problem = None;
            }
            Ok(audit.paused)
        })
        .map_err(err)??;
    store.log(name, Who::Operator, if paused { "paused it" } else { "resumed it" }, "").map_err(err)?;
    if !paused {
        crate::audit::run::wake();
    }
    Ok(format!("[audit] {name} {}", if paused { "paused: running checks finish, nothing new starts" } else { "resumed" }))
}

pub fn stop(store: &AuditStore, name: &str) -> Reply {
    let audit = store
        .change(name, |audit| {
            audit.status = Status::Stopped;
            audit.paused = false;
            Ok(audit.clone())
        })
        .map_err(err)??;
    store.log(name, Who::Operator, "stopped it", "").map_err(err)?;
    close_lead(&audit, "stopped by the operator");
    Ok(format!("[audit] {name} stopped; its results stay"))
}

/// Its lead on its owner board, closed.
pub fn close_lead(audit: &Audit, why: &str) {
    let (Some(board), Some(lead), Some(boards)) = (&audit.board, &audit.lead, audit::boards()) else { return };
    if let Ok(Some(found)) = boards.board(board) {
        let _ = boards.close_explored(&found.id, lead);
        let _ = boards.add_hint(&found.id, &format!("audit {}: its lead {lead} closed — {why}", audit.name));
    }
}

pub fn set(store: &AuditStore, name: &str, pairs: &str) -> Reply {
    let changed = store.change(name, |audit| audit.settings.apply(pairs)).map_err(err)??;
    store.log(name, Who::Operator, &format!("set {}", changed.join(" ")), "").map_err(err)?;
    crate::audit::run::wake();
    Ok(format!("[audit] {name}: {} — from the next check", changed.join(" ")))
}

pub fn detach(store: &AuditStore, name: &str) -> Reply {
    let audit = store.audit(name).map_err(err)?.ok_or(format!("no audit {name}"))?;
    let Some(board) = audit.board.clone() else { return Err(format!("{name} belongs to no project")) };
    close_lead(&audit, "detached");
    store
        .change(name, |audit| {
            audit.board = None;
            audit.lead = None;
            if audit.status != Status::Stopped {
                audit.status = Status::Standing;
            }
            Ok(())
        })
        .map_err(err)??;
    store.remove_links(name, &board, Some("owner"), None).map_err(err)?;
    store.log(name, Who::Operator, &format!("detached it from {board}"), "").map_err(err)?;
    Ok(format!("[audit] {name} detached from {board}; its findings stay there"))
}

/// The operator's own claim: accepted, and his.
pub fn operator_claim_new(store: &AuditStore, audit: &str, text: &str, source: &str) -> Reply {
    if text.trim().len() < 8 {
        return Err("a claim needs a few words".into());
    }
    let a = store.audit(audit).map_err(err)?.ok_or(format!("no audit {audit}"))?;
    let claim = store
        .add_claim(
            audit,
            Claim {
                text: audit::clip_text(text),
                author: "operator".into(),
                source: if source.is_empty() { "the operator".into() } else { source.into() },
                kind: "invariant".into(),
                risk: 8,
                review: Review::Accepted,
                priority: 8,
                new: a.gone_at.is_some(),
                ..Claim::default()
            },
        )
        .map_err(err)?;
    store.log(audit, Who::Operator, &format!("added his own claim {}", claim.id), text).map_err(err)?;
    note(format!("{audit}: the operator added {}: \"{}\"", claim.id, crate::util::clip(text, 160)));
    Ok(format!("[audit] {} added to {audit}, accepted", claim.id))
}

pub fn accept_all(store: &AuditStore, audit: &str, min_risk: i64) -> Reply {
    let mut n = 0;
    for claim in store.claims(audit).map_err(err)? {
        if claim.review == Review::Proposed && claim.risk >= min_risk && (claim.readings.is_empty() || claim.picked.is_some()) {
            store
                .change_claim(audit, &claim.id, |c| {
                    c.review = Review::Accepted;
                    Ok(())
                })
                .map_err(err)??;
            n += 1;
        }
    }
    store.log(audit, Who::Operator, &format!("accepted all ({n}) at risk ≥ {min_risk}"), "").map_err(err)?;
    Ok(format!("[audit] {audit}: {n} accepted"))
}

// ------------------------------------------------------------------ links

/// The agent's or the operator's link: free, logged, undoable. `owner` attaches
/// (a proposal under `ask`, `audit_attach`).
pub fn link(store: &AuditStore, audit: &str, board: &str, claim: Option<&str>, lead: Option<&str>, who: Who, reason: &str) -> Reply {
    let a = store.audit(audit).map_err(err)?.ok_or(format!("no audit {audit}"))?;
    let boards = audit::boards().ok_or("the board store is not ready")?;
    let found = boards.board(board).map_err(err)?.ok_or(format!("no board named {board}"))?;
    if let Some(claim) = claim {
        store.claim(audit, claim).map_err(err)?.ok_or(format!("no claim {claim} in {audit}"))?;
        let lead = lead.ok_or("a claim links to a lead: name it")?;
        boards.intent(&found.id, lead).map_err(err)?.ok_or(format!("no lead {lead} on {board}"))?;
        store
            .change_claim(audit, claim, |c| {
                c.lead = Some((board.to_string(), lead.to_string()));
                Ok(())
            })
            .map_err(err)??;
        store
            .add_link(&Link {
                audit: audit.into(),
                board: board.into(),
                kind: "lead".into(),
                claim: Some(claim.into()),
                lead: Some(lead.into()),
                by: who.label().into(),
                at: audit::now(),
                reason: reason.into(),
            })
            .map_err(err)?;
        store.log(audit, who, &format!("linked {claim} to {board} lead {lead}"), reason).map_err(err)?;
        return Ok(format!("[audit] {audit}/{claim}'s results land under {board} {lead}"));
    }
    if a.board.as_deref() == Some(board) {
        return Err(format!("{audit} runs for {board} already"));
    }
    if store.links(Some(audit), Some(board)).map_err(err)?.iter().any(|link| link.kind == "linked") {
        return Err(format!("{audit} is linked to {board} already"));
    }
    // Its results' row on that board.
    let batch = [(format!("linked audit {audit}"), format!("audit {audit}'s results, linked by {}", who.label()), Vec::new(), 3)];
    let row = match boards.declare_intents(&found.id, &batch, usize::MAX, Local::now()).map_err(err)? {
        Ok(intents) => intents.first().map(|intent| intent.id.clone()),
        Err(_) => None,
    };
    if let Some(row) = &row {
        let _ = boards.claim(&found.id, row, &holder(audit), Local::now());
    }
    store
        .add_link(&Link { audit: audit.into(), board: board.into(), kind: "linked".into(), lead: row, by: who.label().into(), at: audit::now(), reason: reason.into(), ..Link::default() })
        .map_err(err)?;
    store.log(audit, who, &format!("linked it to {board}"), reason).map_err(err)?;
    Ok(format!("[audit] {audit} linked to {board}{}", if who == Who::S0ma { " · by s0ma" } else { "" }))
}

pub fn unlink(store: &AuditStore, audit: &str, board: &str, claim: Option<&str>, who: Who, reason: &str) -> Reply {
    let links = store.links(Some(audit), Some(board)).map_err(err)?;
    let wanted = |link: &Link| match claim {
        Some(claim) => link.kind == "lead" && link.claim.as_deref() == Some(claim),
        None => link.kind == "linked",
    };
    let Some(link) = links.iter().find(|link| wanted(link)).cloned() else {
        return Err(format!("{audit} has no such link to {board}"));
    };
    store.remove_links(audit, board, Some(&link.kind), claim).map_err(err)?;
    if let Some(claim) = claim {
        store
            .change_claim(audit, claim, |c| {
                c.lead = None;
                Ok(())
            })
            .map_err(err)??;
    } else if let (Some(row), Some(boards)) = (&link.lead, audit::boards())
        && let Ok(Some(found)) = boards.board(board)
    {
        let _ = boards.close_explored(&found.id, row);
    }
    store.log(audit, who, &format!("unlinked it from {board}{}", claim.map(|c| format!(" ({c})")).unwrap_or_default()), reason).map_err(err)?;
    if who == Who::Operator && link.by == "s0ma" {
        needs_s0ma(format!(
            "· The operator unlinked audit {audit} from {board}, a link you made{}",
            if reason.is_empty() { ".".to_string() } else { format!(": {reason}") }
        ));
    }
    Ok(format!("[audit] {audit} unlinked from {board}"))
}

/// The agent's `audit_attach`: a standing audit runs for a project. Costs steps,
/// so it goes through the project's policy.
pub fn attach(store: &AuditStore, audit: &str, board: &str, who: Who, why: &str) -> Reply {
    let a = store.audit(audit).map_err(err)?.ok_or(format!("no audit {audit}"))?;
    if a.board.is_some() {
        return Err(format!("{audit} runs for {} already", a.board.unwrap_or_default()));
    }
    let policy = store.policy(board).map_err(err)?;
    if who == Who::S0ma && policy != Policy::Auto {
        if policy == Policy::Off {
            return Err(format!("refused: audits are off for {board}"));
        }
        store
            .change(audit, |a| {
                a.attach_proposed = Some((board.to_string(), why.to_string()));
                Ok(())
            })
            .map_err(err)??;
        store.log(audit, Who::S0ma, &format!("proposed attaching it to {board}"), why).map_err(err)?;
        return Ok(json!({"ok": true, "proposed": true, "note": format!("Attaching {audit} to {board} spends its steps: a proposal for the operator (Alt+U, a twice).")}).to_string());
    }
    store
        .change(audit, |a| {
            a.board = Some(board.to_string());
            a.mode = Mode::Diagnose;
            a.attach_proposed = None;
            if a.status == Status::Standing {
                a.status = Status::Done;
            }
            Ok(())
        })
        .map_err(err)??;
    store.log(audit, who, &format!("attached it to {board}"), why).map_err(err)?;
    start(store, audit, who)
}

/// `f` on a reproduced failure in a standing audit: a new project to fix it,
/// whose goal is the claim holding again and whose test is its repro, with
/// the standing audit attached in diagnose mode. The only way an audit opens
/// a board.
pub fn fix_project(store: &AuditStore, audit: &str, id: &str) -> Reply {
    let a = store.audit(audit).map_err(err)?.ok_or(format!("no audit {audit}"))?;
    if !a.standing() {
        return Err(format!("{audit} runs for {}: investigating a failure there is more leads on that board", a.board.unwrap_or_default()));
    }
    let claim = store.claim(audit, id).map_err(err)?.ok_or(format!("no claim {id} in {audit}"))?;
    if claim.repro != audit::Repro::Fails {
        return Err(format!("{id} has no reproduced failure: f opens a project only from one"));
    }
    let boards = audit::boards().ok_or("the board store is not ready")?;
    let mut name = format!("fix-{audit}-{id}");
    let mut n = 2;
    while boards.board(&name).map_err(err)?.is_some() {
        name = format!("fix-{audit}-{id}-{n}");
        n += 1;
    }
    let mut test = format!("audit-run {audit} --repro {id}");
    if crate::audit::run::gates(&name).iter().any(|g| g == audit) {
        test.push_str(&format!(" && audit-run {audit} --holding"));
    }
    let opened = boards
        .open_board(
            &name,
            &format!("{audit}/{id} holds again"),
            &format!("From the standing audit {audit}: {id} fails, reproduced. {}", claim.text),
            &format!("{id} holds again: {}", claim.text),
            &test,
            crate::board::GoalTestKind::Command,
            crate::board::DEFAULT_STEP_BUDGET,
        )
        .map_err(err)??;
    store
        .add_link(&Link { audit: audit.into(), board: opened.name.clone(), kind: "origin".into(), claim: Some(id.into()), by: "operator".into(), at: audit::now(), ..Link::default() })
        .map_err(err)?;
    store.log(audit, Who::Operator, &format!("opened {} to fix {id}", opened.name), "").map_err(err)?;
    attach(store, audit, &opened.name, Who::Operator, &format!("to fix {id}"))?;
    Ok(format!("[audit] {} opened to fix {id}: its test is {test}, and {audit} is attached to it", opened.name))
}

// ------------------------------------------------------------------ results to boards

/// A result onto the boards that should see it, as `post` allows: the owner
/// board under the audit's lead (or the lead the claim is linked to), and
/// each linked board under its `linked audit` row. Only the harness calls this (rule 1);
/// `operator` is his `p` on a result waiting in review.
pub fn post_result(store: &AuditStore, audit_name: &str, id: &str, operator: bool) -> audit::Result<()> {
    let Some(audit) = store.audit(audit_name)? else { return Ok(()) };
    let Some(claim) = store.claim(audit_name, id)? else { return Ok(()) };
    let claims = store.claims(audit_name)?;
    let reproduced = claim.verdict == Verdict::Fails && claim.repro == Repro::Fails;
    let holds = claim.verdict == Verdict::Holds && matches!(claim.level, Level::Bounded | Level::Model | Level::Proven);
    let finding = if reproduced {
        format!(
            "+ {id} fails: {} ({}audit:{audit_name}/{id})",
            claim.text,
            claim.repro_test.as_ref().map(|t| format!("repro {t}, ")).unwrap_or_default()
        )
    } else if holds {
        let unaccepted: Vec<&str> = claim
            .rests_on
            .iter()
            .filter(|a| claims.iter().any(|c| &c.id == *a && c.review != Review::Accepted))
            .map(String::as_str)
            .collect();
        let mut marks = Vec::new();
        if claim.review != Review::Accepted {
            marks.push("diagnostic".to_string());
        }
        if !unaccepted.is_empty() {
            marks.push(format!("if {}", unaccepted.join(", ")));
        }
        format!(
            "− {id} holds ({}): {}{} (audit:{audit_name}/{id})",
            claim.level.label(),
            claim.text,
            if marks.is_empty() { String::new() } else { format!(" [{}]", marks.join(" · ")) }
        )
    } else {
        return Ok(());
    };
    // What post allows.
    let allowed = operator
        || match audit.settings.post {
            Post::Auto => true,
            Post::Failures => reproduced,
            Post::Review => false,
        };
    if !allowed {
        if audit.settings.post == Post::Review && claim.posted == Posted::No {
            store.change_claim(audit_name, id, |c| {
                c.posted = Posted::Ready;
                Ok(())
            })??;
        }
        return Ok(());
    }
    if claim.posted == Posted::Posted {
        return Ok(());
    }
    let Some(boards) = audit::boards() else { return Ok(()) };
    let polarity = if reproduced { crate::board::Polarity::Supports } else { crate::board::Polarity::RulesOut };
    let mut first_fact = None;
    // Owner (or the lead the claim is linked to).
    let mut places: Vec<(String, Option<String>)> = Vec::new();
    match &claim.lead {
        Some((board, lead)) => places.push((board.clone(), Some(lead.clone()))),
        None => {
            if let Some(board) = &audit.board {
                places.push((board.clone(), audit.lead.clone()));
            }
        }
    }
    for link in store.links(Some(audit_name), None)? {
        if link.kind == "linked" && !places.iter().any(|(board, _)| *board == link.board) {
            places.push((link.board.clone(), link.lead.clone()));
        }
    }
    for (board, lead) in places {
        let Some(found) = boards.board(&board)? else { continue };
        let fact = boards.harness_finding(&found.id, &finding, polarity, lead.as_deref())?;
        first_fact.get_or_insert(format!("{board}/{}", fact.id));
    }
    store.change_claim(audit_name, id, |c| {
        c.posted = Posted::Posted;
        c.fact = first_fact.clone();
        Ok(())
    })??;
    store.log(audit_name, Who::Harness, &format!("posted {id}'s result"), &finding)?;
    // A new failure in a standing audit needs the agent.
    if reproduced && audit.standing() {
        needs_s0ma(format!("· Standing audit {audit_name}: a re-run found {id} failing, reproduced: {}", claim.text));
    }
    Ok(())
}

// ------------------------------------------------------------------ the context block

pub const BLOCK_MARKER: &str = "[audits — current state, rendered by the harness]";
pub const UPDATE_MARKER: &str = "[audits — update, rendered by the harness]";

/// One line per live audit, with its project; standing ones only when they
/// have something new.
pub fn lines(store: &AuditStore) -> audit::Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for audit in store.audits()? {
        if !audit.status.live() || (audit.standing() && audit.rerun_proposed.is_none() && audit.status == Status::Standing) {
            continue;
        }
        let claims = store.claims(&audit.name)?;
        let failing = claims.iter().filter(|c| c.verdict == Verdict::Fails && !c.waived).count();
        let reproduced = claims.iter().filter(|c| c.repro == Repro::Fails).count();
        let queue = claims.iter().filter(|c| c.checkable(&audit) && c.verdict == Verdict::Unchecked && !c.taken).count();
        let owner = match &audit.board {
            Some(board) => format!("for {board}, {}", if audit.mode == Mode::Diagnose { "diagnose" } else { "assure" }),
            None => "standing".to_string(),
        };
        let mut line = format!(
            "· {} ({owner}) — {}: {} claims: {failing} failing, {reproduced} reproduced; queue {queue}",
            audit.name,
            audit.status.label(),
            claims.len()
        );
        if audit.paused {
            line.push_str(" — paused by the operator");
        }
        if audit.status == Status::Proposed {
            line.push_str(" — your proposal, waiting for the operator");
        }
        out.push((audit.name.clone(), line));
    }
    Ok(out)
}

pub fn is_block(content: &str) -> bool {
    content.starts_with(BLOCK_MARKER) || content.starts_with(UPDATE_MARKER)
}

/// Keep the audits current at the end of the context, as the boards block
/// is kept: the block once, then updates naming only what changed.
pub fn refresh_context(messages: &mut Vec<crate::types::Message>) -> audit::Result<bool> {
    let Some(store) = audit::store() else { return Ok(false) };
    let fresh = lines(&store)?;
    let start = messages.iter().position(|m| m.content.starts_with(BLOCK_MARKER));
    let Some(start) = start else {
        if fresh.is_empty() {
            return Ok(false);
        }
        let body = fresh.iter().map(|(_, line)| line.clone()).collect::<Vec<_>>().join("\n");
        messages.push(crate::types::Message::user(format!(
            "{BLOCK_MARKER}\nAUDITS — claims checked by the harness for my projects. audit_read for the ledger; \
             verdicts are the harness's.\n{body}"
        )));
        return Ok(true);
    };
    let mut known: Vec<(String, String)> = Vec::new();
    for message in &messages[start..] {
        if !is_block(&message.content) {
            continue;
        }
        for line in message.content.lines().filter(|l| l.starts_with("· ")) {
            let key = line.trim_start_matches("· ").split(' ').next().unwrap_or("").to_string();
            known.retain(|(k, _)| *k != key);
            if !line.ends_with("no longer live") {
                known.push((key, line.to_string()));
            }
        }
    }
    let mut changed: Vec<String> =
        fresh.iter().filter(|(k, l)| !known.iter().any(|(kk, ll)| kk == k && ll == l)).map(|(_, l)| l.clone()).collect();
    for (key, _) in &known {
        if !fresh.iter().any(|(k, _)| k == key) {
            changed.push(format!("· {key} — no longer live"));
        }
    }
    if changed.is_empty() {
        return Ok(false);
    }
    messages.push(crate::types::Message::user(format!("{UPDATE_MARKER}\n{}", changed.join("\n"))));
    Ok(true)
}

// ------------------------------------------------------------------ the operator's commands

pub const HELP: &str = "usage: /audit [list]\n       \
    /audit start <name> <path>… [--for <board>] [--scope <spec>…]\n       \
    /audit go|pause|stop|rerun <name>\n       \
    /audit set <name> key=value…   (steps time workers depth tools mutants back-translation loophole ask-over post gate)\n       \
    /audit accept-all <name> [--risk N]\n       \
    /audit claim <name> <text>     (your own claim, accepted)\n       \
    /audit link <name> <board> [<claim> <lead>] · /audit unlink <name> <board> [<claim>] [why]\n       \
    /audit attach <name> <board> · /audit detach <name>\n       \
    /audit scope <name> add|remove <spec>\n       \
    /audit keep|letgo <name> · /audit merge <name> <into>\n       \
    /audit start-all <dir> · /audit doctor <lang>\n       \
    /audit policy off|ask|auto     (the default for new projects)\n       \
    /board audits <board> off|ask|auto [steps=N depth=L workers=N] · /board gate <board> <audit>";

/// `/audit …` and `/board audits|gate …`; None for anything else.
pub fn handle_command(command: &str) -> Option<Reply> {
    let words: Vec<&str> = command.split_whitespace().collect();
    if !matches!(words.first(), Some(&"/audit")) && !matches!(words.as_slice(), ["/board", "audits" | "gate", ..]) {
        return None;
    }
    let Some(store) = audit::store() else { return Some(Err("[audit] not ready".into())) };
    Some(match words.as_slice() {
        ["/audit"] | ["/audit", "list"] => list(&store),
        ["/audit", "start", name, rest @ ..] => {
            let mut ask = StartAsk { name: name.to_string(), ..StartAsk::default() };
            let mut i = 0;
            while i < rest.len() {
                match rest[i] {
                    "--for" if i + 1 < rest.len() => {
                        ask.board = Some(rest[i + 1].to_string());
                        i += 2;
                    }
                    "--scope" if i + 1 < rest.len() => {
                        ask.scope.push(rest[i + 1].to_string());
                        i += 2;
                    }
                    path => {
                        ask.targets.push(path.to_string());
                        i += 1;
                    }
                }
            }
            // A proposal of the agent's with that name: started as proposed.
            match store.audit(name) {
                Ok(Some(existing)) if existing.status == Status::Proposed => start(&store, name, Who::Operator),
                _ => operator_start(&store, ask),
            }
        }
        ["/audit", "go", name] => go(&store, name),
        ["/audit", "pause", name] => pause(&store, name),
        ["/audit", "stop", name] => stop(&store, name),
        ["/audit", "rerun", name] => crate::audit::run::rerun(&store, name, Who::Operator),
        ["/audit", "set", name, pairs @ ..] if !pairs.is_empty() => set(&store, name, &pairs.join(" ")),
        ["/audit", "accept-all", name] => accept_all(&store, name, 1),
        ["/audit", "accept-all", name, "--risk", n] => match n.trim_start_matches(">=").trim_start_matches('≥').parse() {
            Ok(n) => accept_all(&store, name, n),
            Err(_) => Err("--risk N".into()),
        },
        ["/audit", "claim", name, text @ ..] if !text.is_empty() => operator_claim_new(&store, name, &text.join(" "), ""),
        ["/audit", "link", name, board] => link(&store, name, board, None, None, Who::Operator, ""),
        ["/audit", "link", name, board, claim, lead] => link(&store, name, board, Some(claim), Some(lead), Who::Operator, ""),
        ["/audit", "unlink", name, board, rest @ ..] => {
            let (claim, why) = match rest.first() {
                Some(first) if first.starts_with('c') && first[1..].chars().all(|c| c.is_ascii_digit()) => (Some(*first), rest[1..].join(" ")),
                _ => (None, rest.join(" ")),
            };
            unlink(&store, name, board, claim, Who::Operator, &why)
        }
        ["/audit", "attach", name, board] => attach(&store, name, board, Who::Operator, ""),
        ["/audit", "detach", name] => detach(&store, name),
        ["/audit", "scope", name, verb @ ("add" | "remove"), spec] => crate::audit::scope::change_scope(&store, name, verb == &"add", spec),
        ["/audit", "keep", name] => crate::audit::run::after_project(&store, name, "keep", None),
        ["/audit", "letgo", name] | ["/audit", "let-go", name] => crate::audit::run::after_project(&store, name, "letgo", None),
        ["/audit", "merge", name, into] => crate::audit::run::after_project(&store, name, "merge", Some(into)),
        ["/audit", "start-all", dir] => crate::audit::run::start_all(&store, dir),
        ["/audit", "doctor", lang] => crate::audit::adapters::doctor_in_background(lang.to_string()),
        ["/audit", "policy"] => store.default_policy().map(|p| format!("[audit] new projects start with audit policy {}", p.label())).map_err(err),
        ["/audit", "policy", policy] => match Policy::parse(policy) {
            Some(policy) => {
                // Projects that exist keep the policy they started with.
                let before = store.default_policy().map_err(err);
                if let (Ok(before), Some(boards)) = (before, audit::boards()) {
                    for board in boards.boards(false).unwrap_or_default() {
                        if store.has_policy(&board.name).ok() == Some(false) {
                            let _ = store.set_policy(&board.name, before, None);
                        }
                    }
                }
                store.set_default_policy(policy).map(|()| format!("[audit] new projects start with audit policy {}", policy.label())).map_err(err)
            }
            None => Err("/audit policy off|ask|auto".into()),
        },
        ["/board", "audits", board, policy, rest @ ..] => match Policy::parse(policy) {
            Some(policy) => {
                let mut ceilings = store.ceilings(board).unwrap_or_default();
                for pair in rest {
                    match pair.split_once('=') {
                        Some(("steps", n)) => ceilings.steps = n.parse().unwrap_or(ceilings.steps),
                        Some(("workers", n)) => ceilings.workers = n.parse().unwrap_or(ceilings.workers),
                        Some(("depth", l)) => ceilings.depth = Level::parse(l).unwrap_or(ceilings.depth),
                        _ => {}
                    }
                }
                store
                    .set_policy(board, policy, Some(ceilings.clone()))
                    .map(|()| {
                        format!(
                            "[audit] {board}: audits {}{}",
                            policy.label(),
                            if policy == Policy::Auto {
                                format!(" (within {} steps, depth {}, {} workers)", ceilings.steps, ceilings.depth.label(), ceilings.workers)
                            } else {
                                String::new()
                            }
                        )
                    })
                    .map_err(err)
            }
            None => Err("/board audits <board> off|ask|auto".into()),
        },
        ["/board", "audits", board] => store.policy(board).map(|p| format!("[audit] {board}: audits {}", p.label())).map_err(err),
        ["/board", "gate", board, audit] => crate::audit::run::gate(&store, board, audit),
        _ => Err(HELP.into()),
    })
}

fn list(store: &AuditStore) -> Reply {
    let audits = store.audits().map_err(err)?;
    if audits.is_empty() {
        return Ok("[audit] none yet. /audit start <name> <path>… [--for <board>]".into());
    }
    let mut lines = vec!["[audit]".to_string()];
    for audit in audits {
        let (mark, status) = audit.status_mark();
        lines.push(format!(
            "  {mark} {} — {status}{}",
            audit.name,
            audit.board.as_ref().map(|b| format!(" · for {b}")).unwrap_or_else(|| " · standing".into())
        ));
    }
    Ok(lines.join("\n"))
}

/// What the projects tab and the board show of the audits on a board.
pub fn for_board(board: &str) -> Vec<Audit> {
    let Some(store) = audit::store() else { return Vec::new() };
    let mut out: Vec<Audit> = store.audits().unwrap_or_default().into_iter().filter(|a| a.board.as_deref() == Some(board)).collect();
    for link in store.links(None, Some(board)).unwrap_or_default() {
        if link.kind == "linked"
            && !out.iter().any(|a| a.name == link.audit)
            && let Ok(Some(audit)) = store.audit(&link.audit)
        {
            out.push(audit);
        }
    }
    out
}

/// Whether any audit is under way (workers get the audit tools then).
pub fn any_live() -> bool {
    audit::store().and_then(|s| s.audits().ok()).is_some_and(|audits| audits.iter().any(|a| matches!(a.status, Status::Harvesting | Status::Checking)))
}

/// What waits for the operator across audits: the tab bar's `[!]`.
pub fn waiting() -> usize {
    let Some(store) = audit::store() else { return 0 };
    let mut n = 0;
    for audit in store.audits().unwrap_or_default() {
        if audit.status == Status::Proposed || audit.rerun_proposed.is_some() || audit.attach_proposed.is_some() {
            n += 1;
        }
        if !audit.status.live() {
            continue;
        }
        n += store
            .claims(&audit.name)
            .unwrap_or_default()
            .iter()
            .filter(|c| c.weaker.is_some() || (c.repro == Repro::ModelOnly && !c.model_only_accepted) || c.posted == Posted::Ready)
            .count();
    }
    for lang in crate::audit::adapters::proposed() {
        let _ = lang;
        n += 1;
    }
    n
}
