//! The audit tools. None of them returns or accepts a verdict, and none
//! writes to a board: verdicts and findings are the harness's (rule 1).
//! Workers get the three that hand work to the harness, and `audit_read`.

use crate::audit::{self, Claim, Evidence, Level, Review, Settings, Who};
use crate::audit::ops::StartAsk;
use crate::llm::ToolDefinition;
use crate::types::ToolCall;

use serde_json::{Value, json};

/// The tools the agent has.
pub const S0MA_TOOLS: [&str; 12] = [
    "audit_start",
    "audit_plan_note",
    "audit_link",
    "audit_unlink",
    "audit_attach",
    "audit_rerun_propose",
    "audit_claims_propose",
    "audit_question",
    "audit_read",
    "audit_take",
    "audit_adapter_propose",
    "audit_check_submit",
];

/// The tools a worker has.
pub const WORKER_TOOLS: [&str; 4] = ["audit_read", "audit_check_submit", "audit_repro_submit", "audit_model_only"];

pub fn handles(name: &str) -> bool {
    S0MA_TOOLS.contains(&name) || WORKER_TOOLS.contains(&name)
}

fn audit_param() -> Value {
    json!({"type": "string", "description": "The audit's name."})
}

pub fn tool_definitions() -> Vec<ToolDefinition> {
    let claim = json!({"type": "string", "description": "A claim's id: c<N>, or a<N> for an assumption."});
    vec![
        ToolDefinition::function(
            "audit_start",
            "Pick up an audit for a project when checking what its code claims is a good next move: the claims its code \
and docs make about the suspect area are collected, checked mechanically by the harness, and a claim that fails with a \
reproduced counterexample points at the bug. Under the project's audit policy this is usually a PROPOSAL the operator \
starts (policy ask); under auto it starts within his ceilings; under off it is refused. The audit runs in its own shell \
(audit·<name>) the operator watches. Results are evidence for the board, never assurance, and you never score them.",
            json!({"type": "object", "required": ["board", "name", "targets", "why"], "properties": {
                "board": {"type": "string", "description": "The project (board) it is for."},
                "name": {"type": "string", "description": "A short name: lowercase letters, digits, - and _."},
                "targets": {"type": "array", "items": {"type": "string"}, "description": "Folders on your machine: a git repo (worked in a worktree on branch s0ma-audit/<name>) or any folder (worked on a copy)."},
                "scope": {"type": "array", "items": {"type": "string"}, "description": "The slice: path:<glob>, sym:<name>, diff:<range>, doc:<file>. Empty: the whole target."},
                "why": {"type": "string", "description": "Why an audit, here: shown to the operator with the proposal."},
                "expected_claims": {"type": "integer"},
                "settings": {"type": "string", "description": "Optional key=value settings: steps=N time=10m/2h workers=N depth=property tools=+tla,-kani post=failures gate=plan"}
            }}),
        ),
        ToolDefinition::function(
            "audit_claims_propose",
            "Propose claims the code and its docs make (harvest): \"never\", \"always\", \"only if\", decision tables, what \
past fixes imply, asserts, test names, and conflicts between two sources. They arrive unreviewed ([?]); the operator \
accepts, rejects or rewords them. audit_read what=sources lists what the harness's scan found. A claim from a board \
finding cites it as board:<name>/<fact>.",
            json!({"type": "object", "required": ["audit", "claims"], "properties": {
                "audit": audit_param(),
                "claims": {"type": "array", "maxItems": 30, "items": {"type": "object", "required": ["text", "source", "kind", "risk", "why"], "properties": {
                    "text": {"type": "string", "description": "One checkable sentence. Split compound claims; each 'and' is a clause a check must anchor."},
                    "source": {"type": "string", "description": "path:line, doc: \"quote\", a commit, or board:<name>/<fact>."},
                    "kind": {"type": "string", "enum": ["pure", "invariant", "protocol", "equivalence", "resource", "assumes", "conflict"]},
                    "risk": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Your guess at the cost if it is false."},
                    "why": {"type": "string", "description": "Why it is in the plan, and the tool and depth you would check it with."},
                    "tool": {"type": "string", "description": "test, property, bounded, tla, differential or command."},
                    "readings": {"type": "array", "items": {"type": "string"}, "description": "When it is ambiguous: its numbered readings."}
                }}}
            }}),
        ),
        ToolDefinition::function(
            "audit_plan_note",
            "Record why a claim is where it is in the plan, and its tool and depth: it goes to the audit's log the operator reads.",
            json!({"type": "object", "required": ["audit", "claim", "why"], "properties": {"audit": audit_param(), "claim": claim, "why": {"type": "string"}, "tool": {"type": "string"}, "depth": {"type": "string"}, "mutant": {"type": "string", "description": "A survived mutant's name: then why is your reading of it, a gap in the check (say gap) or outside the claim (say outside). Shown in the claim view, marked as your reading."}}}),
        ),
        ToolDefinition::function(
            "audit_question",
            "Ask the operator to choose between readings of an ambiguous claim. In diagnose mode each reading is also checked as \
its own claim; which reading fails is itself a clue.",
            json!({"type": "object", "required": ["audit", "claim", "options"], "properties": {"audit": audit_param(), "claim": claim, "options": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "string"}}}}),
        ),
        ToolDefinition::function(
            "audit_read",
            "Read an audit: what=ledger (every claim, its state, level and where its results went), claim (one claim with its \
checks, evidence, back-translation and board links), sources (what the harness's harvest scan found), plan, log, runs, \
or slice (the scope's boundary and assumptions).",
            json!({"type": "object", "required": ["audit"], "properties": {"audit": audit_param(), "what": {"type": "string", "enum": ["ledger", "claim", "sources", "plan", "log", "runs", "slice"]}, "claim": claim}}),
        ),
        ToolDefinition::function(
            "audit_take",
            "Hold a queued claim for a worker you are about to spawn (30 minutes, as a board lead is held). Give the worker the \
claim, the work tree and what the check must look like: a directory checks/<claim>/ with check.toml (claim, lang, kind, \
run, timeout, targets, assumes, and optionally coverage: a command writing lcov to $COV, as the language's adapter shows) whose run exits 0 when it holds, 1 when it fails, 2 on error; mutant patches in \
checks/<claim>/mutants/*.patch it must catch; and // [n] anchors for each clause. A claim the operator took is his.",
            json!({"type": "object", "required": ["audit", "claim", "worker"], "properties": {"audit": audit_param(), "claim": claim, "worker": {"type": "string", "description": "The worker's name, as its tab shows it (b3-1)."}}}),
        ),
        ToolDefinition::function(
            "audit_check_submit",
            "Hand a check to the harness: it reads check.toml, files it on the audit's branch, vets it (mutants must be caught; \
a fresh reader back-translates it) and runs it in the audit's shell when the gate allows. You get no verdict here.",
            json!({"type": "object", "required": ["audit", "claim", "dir"], "properties": {"audit": audit_param(), "claim": claim, "dir": {"type": "string", "description": "The check's directory, absolute or relative to the work tree (checks/c7)."}}}),
        ),
        ToolDefinition::function(
            "audit_repro_submit",
            "Hand a repro to the harness: a test in the repo's own framework that turns a failing check's counterexample into a \
failure on the real code (interleavings forced with hooks or barriers; the trace quoted in its doc comment). It becomes a \
finding only if it fails when the harness runs it on the audited code.",
            json!({"type": "object", "required": ["audit", "claim", "test", "run"], "properties": {"audit": audit_param(), "claim": claim, "test": {"type": "string", "description": "The test, as path::name."}, "run": {"type": "string", "description": "The command that runs just it, from the work tree."}}}),
        ),
        ToolDefinition::function(
            "audit_model_only",
            "Say a failing check's counterexample cannot be reproduced on the real code, and why. It is shown to the operator \
as [!] model-only and is never a finding.",
            json!({"type": "object", "required": ["audit", "claim", "why"], "properties": {"audit": audit_param(), "claim": claim, "why": {"type": "string"}}}),
        ),
        ToolDefinition::function(
            "audit_link",
            "Link an audit to a board so its results post there too (free, logged, the operator can undo it), or a claim to a \
lead so its results land under that lead. Never inferred from a path.",
            json!({"type": "object", "required": ["audit", "board"], "properties": {"audit": audit_param(), "board": {"type": "string"}, "claim": claim, "lead": {"type": "string"}, "why": {"type": "string"}}}),
        ),
        ToolDefinition::function(
            "audit_unlink",
            "Remove a link you made (an audit from a board, or a claim from a lead).",
            json!({"type": "object", "required": ["audit", "board"], "properties": {"audit": audit_param(), "board": {"type": "string"}, "claim": claim, "why": {"type": "string"}}}),
        ),
        ToolDefinition::function(
            "audit_attach",
            "Make a project a standing audit's owner, so the audit runs for it and its new audit starts from the standing \
one's checks. It spends that project's steps: a proposal under policy ask.",
            json!({"type": "object", "required": ["audit", "board", "why"], "properties": {"audit": audit_param(), "board": {"type": "string"}, "why": {"type": "string"}}}),
        ),
        ToolDefinition::function(
            "audit_rerun_propose",
            "Ask the operator to re-run a standing audit (the code under it changed). Nothing runs until he starts it.",
            json!({"type": "object", "required": ["audit", "why"], "properties": {"audit": audit_param(), "why": {"type": "string"}}}),
        ),
        ToolDefinition::function(
            "audit_adapter_propose",
            "Propose a language adapter (TOML: lang, detect, setup, test, property, bounded, mutate, repro_dir, ceiling, good, \
bad) for a language without one. It waits for the operator: its setup installs things and its templates are commands.",
            json!({"type": "object", "required": ["lang", "toml", "why"], "properties": {"lang": {"type": "string"}, "toml": {"type": "string"}, "why": {"type": "string"}}}),
        ),
    ]
}

/// The worker's set.
pub fn worker_tool_definitions() -> Vec<ToolDefinition> {
    tool_definitions().into_iter().filter(|tool| WORKER_TOOLS.contains(&tool.function.name.as_str())).collect()
}

fn rejected(error: impl std::fmt::Display) -> String {
    json!({"ok": false, "error": error.to_string()}).to_string()
}

/// One tool call. `worker` is who calls: `s0ma`, or a worker's run id
/// (supplied by the harness, never by the model).
pub fn execute(call: &ToolCall, worker: &str) -> String {
    let arguments: Value = match serde_json::from_str(&call.function.arguments) {
        Ok(value) => value,
        Err(error) => return rejected(format!("bad arguments: {error}")),
    };
    let Some(store) = audit::store() else { return rejected("the audit store is not ready") };
    let text = |key: &str| arguments.get(key).and_then(Value::as_str).unwrap_or_default().trim().to_string();
    let is_worker = worker != "s0ma";
    if is_worker && !WORKER_TOOLS.contains(&call.function.name.as_str()) {
        return rejected("that tool is s0ma's");
    }
    let reply = |result: std::result::Result<String, String>| match result {
        Ok(text) if text.starts_with('{') => text,
        Ok(text) => json!({"ok": true, "note": text}).to_string(),
        Err(error) => rejected(error),
    };
    // A verdict, a level, or a board write is never an argument (rule 1).
    for forbidden in ["verdict", "level", "holds", "fails", "passed", "result"] {
        if arguments.get(forbidden).is_some() {
            return rejected(format!("'{forbidden}' is the harness's to write, never an argument"));
        }
    }
    let audit_name = text("audit");
    match call.function.name.as_str() {
        "audit_start" => {
            let mut settings = Settings::default();
            if let Err(error) = settings.apply(&text("settings")) {
                return rejected(error);
            }
            let targets: Vec<String> = arguments["targets"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
            if targets.is_empty() {
                return rejected("targets: one or more folders on your machine");
            }
            let scope: Vec<String> = arguments["scope"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
            for spec in &scope {
                if let Err(error) = crate::audit::scope::valid_spec(spec) {
                    return rejected(error);
                }
            }
            reply(crate::audit::ops::s0ma_start(
                &store,
                StartAsk {
                    name: text("name"),
                    board: Some(text("board")).filter(|b| !b.is_empty()),
                    targets,
                    scope,
                    settings,
                    why: text("why"),
                    expected_claims: arguments["expected_claims"].as_u64().map(|n| n as usize),
                },
            ))
        }
        "audit_claims_propose" => {
            let Ok(Some(audit)) = store.audit(&audit_name) else { return rejected(format!("no audit {audit_name}")) };
            if !audit.status.live() || audit.status == audit::Status::Proposed {
                return rejected(format!("{audit_name} is {}: claims come once the operator started it", audit.status.label()));
            }
            let Some(claims) = arguments["claims"].as_array() else { return rejected("claims: a list") };
            let mut added = Vec::new();
            for item in claims.iter().take(30) {
                let field = |key: &str| item.get(key).and_then(Value::as_str).unwrap_or_default().trim().to_string();
                if field("text").len() < 8 {
                    return rejected("each claim needs its text");
                }
                let kind = field("kind");
                let readings: Vec<String> = item["readings"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
                let claim = Claim {
                    text: audit::clip_text(&field("text")),
                    author: "s0ma".into(),
                    source: field("source"),
                    kind: if kind.is_empty() { "invariant".into() } else { kind.clone() },
                    risk: item["risk"].as_i64().unwrap_or(5).clamp(1, 10),
                    review: Review::Proposed,
                    why: field("why"),
                    tool: field("tool"),
                    readings: readings.clone(),
                    priority: 5,
                    new: audit.gone_at.is_some(),
                    level: if kind == "assumes" { Level::Assumed } else { Level::Documented },
                    estimate_secs: 120,
                    ..Claim::default()
                };
                match store.add_claim(&audit_name, claim) {
                    Ok(claim) => {
                        let _ = store.log(&audit_name, Who::S0ma, &format!("proposed {}", claim.id), &claim.why);
                        if claim.is_assumption() {
                            crate::audit::scope::offer_for_discharge(&store, &audit_name, &claim);
                        }
                        // In diagnose mode each reading is a claim of its own.
                        if audit.mode == audit::Mode::Diagnose && readings.len() > 1 {
                            for (i, reading) in readings.iter().enumerate() {
                                let _ = store.add_claim(
                                    &audit_name,
                                    Claim { text: audit::clip_text(reading), author: "s0ma".into(), source: format!("reading {} of {}", i + 1, claim.id), kind: claim.kind.clone(), risk: claim.risk, why: format!("a reading of {}", claim.id), priority: 5, new: claim.new, ..Claim::default() },
                                );
                            }
                        }
                        added.push(claim.id);
                    }
                    Err(error) => return rejected(error),
                }
            }
            if audit.status == audit::Status::Done {
                let _ = store.change(&audit_name, |a| {
                    a.status = audit::Status::Checking;
                    Ok(())
                });
            }
            crate::audit::run::wake();
            json!({"ok": true, "added": added, "note": "Unreviewed ([?]) until the operator triages them. Checking waits for his Go on the plan (unless the gate says otherwise)."}).to_string()
        }
        "audit_plan_note" => {
            let claim_id = text("claim");
            let why = text("why");
            let mutant = text("mutant");
            if !mutant.is_empty() {
                let reading = Evidence { claim: claim_id.clone(), kind: "reading".into(), name: mutant.clone(), path: String::new(), result: why.clone(), at: audit::now() };
                return match store.add_evidence(&audit_name, &reading) {
                    Ok(()) => {
                        let _ = store.log(&audit_name, Who::S0ma, &format!("read survived mutant {mutant} of {claim_id}"), &why);
                        json!({"ok": true}).to_string()
                    }
                    Err(error) => rejected(error),
                };
            }
            let result = store.change_claim(&audit_name, &claim_id, |c| {
                c.why = why.clone();
                if !text("tool").is_empty() {
                    c.tool = text("tool");
                }
                if let Some(level) = Level::parse(&text("depth")) {
                    c.depth = Some(level);
                }
                Ok(())
            });
            match result {
                Ok(Ok(())) => {
                    let _ = store.log(&audit_name, Who::S0ma, &format!("plan note on {claim_id}"), &why);
                    json!({"ok": true}).to_string()
                }
                Ok(Err(error)) => rejected(error),
                Err(error) => rejected(error),
            }
        }
        "audit_question" => {
            let claim_id = text("claim");
            let options: Vec<String> = arguments["options"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
            if options.len() < 2 {
                return rejected("at least two readings");
            }
            match store.change_claim(&audit_name, &claim_id, |c| {
                c.readings = options.clone();
                c.picked = None;
                Ok(())
            }) {
                Ok(Ok(())) => {
                    let _ = store.log(&audit_name, Who::S0ma, &format!("asked which reading of {claim_id}"), &options.join(" | "));
                    json!({"ok": true, "note": "The operator picks a reading with 1–4 on the claim in the audit tab."}).to_string()
                }
                Ok(Err(error)) => rejected(error),
                Err(error) => rejected(error),
            }
        }
        "audit_read" => read(&store, &audit_name, &text("what"), &text("claim")),
        "audit_take" => {
            let claim_id = text("claim");
            let holder = text("worker");
            let until = (chrono::Local::now() + chrono::Duration::seconds(audit::HOLD_SECS)).to_rfc3339();
            let now = chrono::Local::now();
            match store.change_claim(&audit_name, &claim_id, |c| {
                if c.taken {
                    return Err(format!("{claim_id} is the operator's: he writes its check"));
                }
                if matches!(c.review, Review::Rejected | Review::Dropped) {
                    return Err(format!("{claim_id} was rejected"));
                }
                if c.held_by.is_some() && !audit::expired(c.held_until.as_deref(), now) {
                    return Err(format!("{claim_id} is held by {}", c.held_by.clone().unwrap_or_default()));
                }
                c.held_by = Some(if holder.is_empty() { "a worker".into() } else { holder.clone() });
                c.held_until = Some(until.clone());
                Ok(c.steer.take())
            }) {
                Ok(Ok(steer)) => {
                    let _ = store.log(&audit_name, Who::S0ma, &format!("took {claim_id} for {holder}"), "");
                    let audit = store.audit(&audit_name).ok().flatten().unwrap_or_default();
                    json!({"ok": true, "held_until": until, "work_tree": audit.work, "check_dir": format!("{}/checks/{claim_id}", audit.work),
                        "direction_from_the_operator": steer}).to_string()
                }
                Ok(Err(error)) => rejected(error),
                Err(error) => rejected(error),
            }
        }
        "audit_check_submit" => reply(crate::audit::run::submit_check(&store, &audit_name, &text("claim"), &text("dir"), if is_worker { worker } else { "s0ma" })),
        "audit_repro_submit" => reply(crate::audit::run::submit_repro(&store, &audit_name, &text("claim"), &text("test"), &text("run"))),
        "audit_model_only" => reply(crate::audit::run::model_only(&store, &audit_name, &text("claim"), &text("why"))),
        "audit_link" => {
            let claim = Some(text("claim")).filter(|c| !c.is_empty());
            let lead = Some(text("lead")).filter(|c| !c.is_empty());
            reply(crate::audit::ops::link(&store, &audit_name, &text("board"), claim.as_deref(), lead.as_deref(), Who::S0ma, &text("why")))
        }
        "audit_unlink" => {
            let claim = Some(text("claim")).filter(|c| !c.is_empty());
            reply(crate::audit::ops::unlink(&store, &audit_name, &text("board"), claim.as_deref(), Who::S0ma, &text("why")))
        }
        "audit_attach" => reply(crate::audit::ops::attach(&store, &audit_name, &text("board"), Who::S0ma, &text("why"))),
        "audit_rerun_propose" => reply(crate::audit::run::propose_rerun(&store, &audit_name, &text("why"))),
        "audit_adapter_propose" => reply(crate::audit::adapters::propose(&text("lang"), &text("toml"), &text("why"))),
        other => rejected(format!("unknown audit tool {other}")),
    }
}

fn claim_json(claim: &Claim) -> Value {
    json!({
        "id": claim.id, "text": claim.text, "author": claim.author, "source": claim.source, "kind": claim.kind,
        "risk": claim.risk, "review": format!("{:?}", claim.review).to_lowercase(), "mark": claim.mark().0,
        "verdict": claim.verdict.label(), "level": claim.level.label(), "repro": format!("{:?}", claim.repro).to_lowercase(),
        "taken_by_operator": claim.taken, "held_by": claim.held_by, "waived": claim.waived, "weaker": claim.weaker,
        "stale": claim.stale, "unsound": claim.unsound, "readings": claim.readings, "picked": claim.picked.map(|p| p + 1),
        "rests_on": claim.rests_on, "posted": format!("{:?}", claim.posted).to_lowercase(), "board_fact": claim.fact,
        "skipped": claim.skipped, "pinned": claim.pinned, "new": claim.new,
    })
}

fn read(store: &audit::AuditStore, name: &str, what: &str, claim_id: &str) -> String {
    let Ok(Some(audit)) = store.audit(name) else { return rejected(format!("no audit {name}")) };
    let claims = store.claims(name).unwrap_or_default();
    let head = json!({
        "name": audit.name, "status": audit.status.label(), "paused": audit.paused || audit.wheel_paused,
        "mode": if audit.mode == audit::Mode::Diagnose { "diagnose" } else { "assure" },
        "board": audit.board, "targets": audit.targets, "work_tree": audit.work, "branch": audit.branch, "commit": audit.commit,
        "scope": audit.scope, "languages": audit.languages, "settings": audit.settings.row(),
        "steps": format!("{}/{}", audit.steps_used, audit.step_cap), "problem": audit.problem,
    });
    match what {
        "" | "ledger" => {
            let mut sorted = claims.clone();
            sorted.sort_by_key(Claim::rank);
            json!({"audit": head, "claims": sorted.iter().map(claim_json).collect::<Vec<_>>()}).to_string()
        }
        "plan" => {
            let mut plan: Vec<&Claim> = claims.iter().filter(|c| !matches!(c.review, Review::Rejected | Review::Dropped) && !c.is_assumption()).collect();
            plan.sort_by_key(|c| crate::audit::run::plan_key(c));
            json!({"audit": head, "gate": format!("{:?}", audit.settings.gate).to_lowercase(), "gone": audit.gone_at.is_some(),
                "plan": plan.iter().map(|c| json!({"id": c.id, "text": c.text, "priority": c.priority, "pinned": c.pinned, "skipped": c.skipped,
                    "tool": c.tool, "depth": c.depth.map(Level::label), "why": c.why, "new": c.new, "taken_by_operator": c.taken})).collect::<Vec<_>>()}).to_string()
        }
        "claim" => {
            let Some(claim) = claims.iter().find(|c| c.id == claim_id) else { return rejected(format!("no claim {claim_id}")) };
            let checks = store.checks(name, claim_id).unwrap_or_default();
            let evidence = store.evidence(name, Some(claim_id)).unwrap_or_default();
            json!({"claim": claim_json(claim), "clauses": audit::clauses(&claim.text),
                "checks": checks.iter().map(|c| json!({"version": c.version, "dir": c.dir, "kind": c.kind, "run": c.run, "by": c.by,
                    "kills": c.kills, "survived": c.survived, "back_translation": c.back_translation, "held_weaker": c.held_weaker,
                    "superseded_by": c.superseded_by})).collect::<Vec<_>>(),
                "evidence": evidence.iter().map(|e| json!({"kind": e.kind, "name": e.name, "result": crate::util::clip(&e.result, 1500)})).collect::<Vec<_>>(),
                "repro_output": claim.repro_output, "model_only_why": claim.model_only_why}).to_string()
        }
        "sources" => json!({"audit": head, "sources": audit.sources}).to_string(),
        "log" => json!({"log": store.log_entries(name).unwrap_or_default().iter().rev().take(60).collect::<Vec<_>>()}).to_string(),
        "runs" => json!({"runs": store.runs(name).unwrap_or_default().iter().rev().take(10).collect::<Vec<_>>()}).to_string(),
        "slice" => json!({"slice": audit.slice, "assumptions": claims.iter().filter(|c| c.is_assumption()).map(|c| json!({
            "id": c.id, "text": c.text, "review": format!("{:?}", c.review).to_lowercase(), "status": crate::audit::scope::assumption_status(store, c)})).collect::<Vec<_>>()}).to_string(),
        other => rejected(format!("what: ledger, claim, sources, plan, log, runs or slice (not {other})")),
    }
}
