//! The harness's side of an audit: setting up its work tree and shell on
//! the agent's machine, the harvest scan, `audit-run` batches typed into the
//! audit's own shell (one approval each), vetting (mutants, back-
//! translation, strengthen-only), repros, re-runs, and what happens after
//! the project. Only this module writes verdicts and levels (rule 1).

use crate::audit::{self, Audit, AuditStore, Check, Claim, Evidence, Level, Mode, Repro, Review, Run, RunCommand, Status, Verdict, Who};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::json;

use std::collections::BTreeSet;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

type Reply = std::result::Result<String, String>;

fn err<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

/// A script on the agent's machine (its VM, or this computer in host mode).
fn machine(script: &str, timeout: u64) -> Result<crate::share::Ran, String> {
    #[cfg(test)]
    if let Some((exit, stdout)) = fake(script) {
        return Ok(crate::share::Ran { exit_code: exit.map(i64::from), stdout, stderr: String::new() });
    }
    crate::vm::netboxes::machine(script, timeout)
}

/// Tests: the agent's machine and the audit's shell, played by a closure (the
/// script or command in, its exit and output out).
#[cfg(test)]
pub(crate) type Fake = Box<dyn FnMut(&str) -> (Option<i32>, String)>;

#[cfg(test)]
thread_local! {
    pub(crate) static FAKE: std::cell::RefCell<Option<Fake>> = std::cell::RefCell::new(None);
}

#[cfg(test)]
fn fake(script: &str) -> Option<(Option<i32>, String)> {
    FAKE.with(|fake| fake.borrow_mut().as_mut().map(|fake| fake(script)))
}

fn q(text: &str) -> String {
    crate::util::shell_quote(text)
}

// ------------------------------------------------------------------ the scheduler

static WAKE: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());
/// Audits with a batch running now, so each runs one at a time.
static RUNNING: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// Look again now: a Go, a submitted check, a resumed audit.
pub fn wake() {
    let (flag, signal) = &WAKE;
    *crate::util::locked(flag) = true;
    signal.notify_all();
}

/// One thread for the life of the process. It never starts anything an
/// audit's operator has not started: only audits he started (Start, Go, r)
/// are served, and never on a schedule.
pub fn start_scheduler() {
    if cfg!(test) {
        return;
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("audit-scheduler".into()).spawn(|| loop {
            {
                let (flag, signal) = &WAKE;
                let mut woken = crate::util::locked(flag);
                if !*woken {
                    woken = signal.wait_timeout(woken, Duration::from_secs(5)).map(|(g, _)| g).unwrap_or_else(|p| p.into_inner().0);
                }
                *woken = false;
            }
            if let Some(store) = audit::store() {
                serve(&store);
            }
        });
    });
}

/// One pass: leads renewed, holds expired, the operator's wheel watched, and
/// each audit with checks ready gets a batch on a thread of its own.
fn serve(store: &AuditStore) {
    let Ok(audits) = store.audits() else { return };
    let now = chrono::Local::now();
    for audit in audits {
        if !matches!(audit.status, Status::Harvesting | Status::Checking) {
            continue;
        }
        renew_lead(&audit);
        if let Ok(claims) = store.claims(&audit.name) {
            for claim in claims.iter().filter(|c| c.held_by.is_some() && audit::expired(c.held_until.as_deref(), now)) {
                let _ = store.change_claim(&audit.name, &claim.id, |c| {
                    c.held_by = None;
                    c.held_until = None;
                    Ok(())
                });
            }
        }
        if audit.wheel_paused && !crate::vm::terms::wheel_held(&audit.shell()) {
            let _ = store.change(&audit.name, |a| {
                a.wheel_paused = false;
                Ok(())
            });
            let _ = store.log(&audit.name, Who::Operator, "handed its shell back: resumed", "");
            continue;
        }
        if audit.paused || audit.wheel_paused || crate::util::locked(&RUNNING).contains(&audit.name) {
            continue;
        }
        let ready = ready_claims(store, &audit);
        if ready.is_empty() {
            maybe_done(store, &audit);
            continue;
        }
        crate::util::locked(&RUNNING).insert(audit.name.clone());
        let name = audit.name.clone();
        let _ = std::thread::Builder::new().name(format!("audit-{name}")).spawn(move || {
            if let Some(store) = audit::store()
                && let Err(error) = run_batch(&store, &name, &ready, false)
            {
                // Paused, never retried on its own: the operator sees why and
                // resumes it (z) when it can run.
                let _ = store.change(&name, |a| {
                    a.paused = true;
                    a.problem = Some(format!("a batch could not run: {error} · z resumes it"));
                    Ok(())
                });
                let _ = store.log(&name, Who::Harness, "a batch could not run: paused", &error);
                crate::terminal::warn_status(&format!("[audit] {name}: a batch could not run, so it paused (z in Alt+U resumes it): {error}"));
            }
            crate::util::locked(&RUNNING).remove(&name);
            wake();
        });
    }
}

/// Claims with a check in force that has not run on it, or a repro waiting,
/// checkable under the audit's gate and mode, in plan order.
fn ready_claims(store: &AuditStore, audit: &Audit) -> Vec<String> {
    let Ok(mut claims) = store.claims(&audit.name) else { return Vec::new() };
    claims.sort_by_key(plan_key);
    let mut out = Vec::new();
    for claim in claims {
        let waiting_repro = claim.repro == Repro::Pending;
        let unrun = claim.verdict == Verdict::Unchecked
            && !claim.running
            && store.current_check(&audit.name, &claim.id).ok().flatten().is_some_and(|check| check.superseded_by.is_none());
        if (waiting_repro || unrun) && claim.checkable(audit) {
            out.push(claim.id);
        }
        if out.len() >= 8 {
            break;
        }
    }
    out
}

/// The plan's order: pinned first, then priority, then risk.
pub fn plan_key(claim: &Claim) -> (bool, i64, i64, String) {
    (!claim.pinned, -claim.priority, -claim.risk, claim.id.clone())
}

fn renew_lead(audit: &Audit) {
    let (Some(board), Some(lead), Some(boards)) = (&audit.board, &audit.lead, audit::boards()) else { return };
    if let Ok(Some(found)) = boards.board(board) {
        let _ = boards.renew_claim(&found.id, lead, &crate::audit::ops::holder(&audit.name));
    }
}

/// The queue empty, nothing held or running: the audit is done, and its
/// lead closes. New claims open it again.
fn maybe_done(store: &AuditStore, audit: &Audit) {
    if audit.status != Status::Checking || audit.gone_at.is_none() {
        return;
    }
    let Ok(claims) = store.claims(&audit.name) else { return };
    let open = claims.iter().any(|c| {
        c.checkable(audit) && !c.taken && (c.verdict == Verdict::Unchecked || c.held_by.is_some() || c.running || c.repro == Repro::Pending)
    });
    if open || claims.is_empty() || claims.iter().any(|c| c.new) {
        return;
    }
    let _ = store.change(&audit.name, |a| {
        a.status = Status::Done;
        Ok(())
    });
    let _ = store.log(&audit.name, Who::Harness, "the queue is empty: done", "");
    crate::audit::ops::close_lead(audit, "its queue is empty");
}

// ------------------------------------------------------------------ setting up

/// Where an audit works on the agent's machine, its shell, the harvest scan and
/// the ceiling preview: on a thread, so a start never waits on the machine.
pub fn prepare_in_background(name: String) {
    if cfg!(test) {
        return;
    }
    let _ = std::thread::Builder::new().name(format!("audit-prep-{name}")).spawn(move || {
        let Some(store) = audit::store() else { return };
        if let Err(error) = prepare(&store, &name) {
            let _ = store.change(&name, |a| {
                a.problem = Some(error.clone());
                Ok(())
            });
            let _ = store.log(&name, Who::Harness, "could not set it up", &error);
            crate::terminal::warn_status(&format!("[audit] {name}: could not set it up: {error}"));
        }
        wake();
    });
}

/// The folder audits work in on the agent's machine.
/// In host mode, under the host workspace (never the operator's own
/// folders' working trees); on its VM, /workspace/audits.
fn base() -> String {
    if crate::vm::machine::on_host() {
        return format!("b={}", q(&format!("{}/s0ma-audits", crate::vm::machine::workspace().display())));
    }
    "b=/workspace/audits; [ -d /workspace ] || b=\"$HOME/s0ma-audits\"".to_string()
}

pub fn prepare(store: &AuditStore, name: &str) -> std::result::Result<(), String> {
    let audit = store.audit(name).map_err(err)?.ok_or("gone")?;
    let first = audit.targets.first().cloned().ok_or("no target")?;
    // A git repo: a worktree on its own branch. Else a copy.
    let script = format!(
        "{base}; w=\"$b/{name}/work\"; mkdir -p \"$b/{name}\" || exit 3\n\
         t={first}\n\
         if [ ! -e \"$t\" ]; then echo \"NOTARGET\"; exit 4; fi\n\
         if [ {single} = 1 ] && top=$(git -C \"$t\" rev-parse --show-toplevel 2>/dev/null); then\n\
           if [ ! -d \"$w\" ]; then git -C \"$top\" worktree add -q -B s0ma-audit/{name} \"$w\" HEAD >/dev/null 2>&1 || {{ echo WORKTREE; exit 5; }}; fi\n\
           echo \"KIND git\"; echo \"COMMIT $(git -C \"$w\" rev-parse HEAD)\"\n\
         else\n\
           mkdir -p \"$w\"; for x in {targets}; do cp -a \"$x\" \"$w/\" 2>/dev/null || cp -a \"$x/.\" \"$w/\"; done\n\
           echo \"KIND folder\"; echo \"COMMIT -\"\n\
         fi\n\
         mkdir -p \"$w/checks\"; chmod -R a+rwX \"$b/{name}\" 2>/dev/null; echo \"WORK $w\"",
        base = base(),
        first = q(&first),
        single = u8::from(audit.targets.len() == 1),
        targets = audit.targets.iter().map(|t| q(t)).collect::<Vec<_>>().join(" "),
    );
    let ran = machine(&script, 300)?;
    let out = ran.stdout;
    let field = |key: &str| out.lines().find_map(|line| line.strip_prefix(&format!("{key} "))).map(str::to_string);
    if out.contains("NOTARGET") {
        return Err(format!("{first} is not on s0ma's machine"));
    }
    let work = field("WORK").ok_or_else(|| format!("the work tree could not be made: {}", ran.stderr.trim()))?;
    let kind = field("KIND").unwrap_or_else(|| "folder".into());
    let commit = field("COMMIT").filter(|c| c != "-").unwrap_or_default();
    store
        .change(name, |a| {
            a.work = work.clone();
            a.kind = kind.clone();
            a.commit = commit.clone();
            a.branch = if kind == "git" { format!("s0ma-audit/{name}") } else { String::new() };
            a.problem = None;
            Ok(())
        })
        .map_err(err)??;
    open_shell(&audit)?;
    // The ceiling preview, the slice map and the harvest scan.
    let languages = crate::audit::adapters::preview(&work, &audit.scope)?;
    store
        .change(name, |a| {
            a.languages = languages.clone();
            Ok(())
        })
        .map_err(err)??;
    if !audit.scope.is_empty() {
        let slice = crate::audit::scope::map(&work, &audit.scope)?;
        store
            .change(name, |a| {
                a.slice = Some(slice.clone());
                Ok(())
            })
            .map_err(err)??;
    }
    let sources = harvest_scan(&work, &audit.scope)?;
    store
        .change(name, |a| {
            a.sources = sources.clone();
            Ok(())
        })
        .map_err(err)??;
    store
        .log(
            name,
            Who::Harness,
            &format!("set it up: {kind} at {work}{}", if commit.is_empty() { String::new() } else { format!(" @ {}", &commit[..7.min(commit.len())]) }),
            &format!("languages: {} · {} sources found by the scan", languages.join(" · "), sources.len()),
        )
        .map_err(err)?;
    write_share(store, name);
    crate::audit::ops::note(format!(
        "{name} is set up ({kind}, {}). audit_read {{audit: \"{name}\", what: \"sources\"}} shows what the scan found; propose claims with audit_claims_propose",
        languages.join(" · ")
    ));
    Ok(())
}

/// Its shell: `audit·<name>`, as the workers' account, which never counts
/// toward the eight.
fn open_shell(audit: &Audit) -> std::result::Result<(), String> {
    let spec = crate::vm::netboxes::Spec {
        name: audit.shell(),
        purpose: format!("audit {}'s checks", audit.name),
        by: format!("audit:{}", audit.name),
        user: crate::vm::netboxes::worker_user(),
        confirm: false,
        program: None,
    };
    if let Some(user) = &spec.user {
        crate::vm::netboxes::ensure_worker_account(&mut crate::vm::netboxes::machine, user)?;
    }
    crate::vm::netboxes::open(&mut crate::vm::netboxes::machine, &spec)?;
    crate::vm::terms::touch();
    Ok(())
}

/// What the code says it does, found mechanically for the agent to read: comments
/// with claim words, asserts and their messages, test names, and commits
/// whose message says fix. At most 300 lines.
fn harvest_scan(work: &str, scope: &[String]) -> std::result::Result<Vec<String>, String> {
    let paths = crate::audit::scope::paths_for(work, scope);
    let script = format!(
        "cd {work} || exit 3\n\
         grep -rnI -E '(//|#|/\\*|\\*|--).*\\b(invariant|never|always|must|can.t happen|cannot happen|safe because|guarantee)' {paths} \
            --exclude-dir=.git --exclude-dir=target --exclude-dir=node_modules --exclude-dir=checks 2>/dev/null | head -150 | sed 's/^/claim-word /'\n\
         grep -rnI -E '(debug_)?assert!?\\(' {paths} --exclude-dir=.git --exclude-dir=target --exclude-dir=node_modules --exclude-dir=checks 2>/dev/null | head -80 | sed 's/^/assert /'\n\
         grep -rnI -E '(fn test_|#\\[test\\]|def test_|it\\(|test\\(|func Test)' {paths} --exclude-dir=.git --exclude-dir=target --exclude-dir=node_modules 2>/dev/null | head -60 | sed 's/^/test /'\n\
         git log -n 40 --format='%h %s' -i --grep=fix 2>/dev/null | sed 's/^/fix-commit /'",
        work = q(work),
        paths = if paths.is_empty() { ".".to_string() } else { paths.iter().map(|p| q(p)).collect::<Vec<_>>().join(" ") },
    );
    let ran = machine(&script, 120)?;
    Ok(ran.stdout.lines().take(300).map(|line| crate::util::clip(line, 240)).collect())
}

/// Everything the operator reads, as markdown in `/share/audits/<name>/`.
pub fn write_share(store: &AuditStore, name: &str) {
    let Ok(Some(audit)) = store.audit(name) else { return };
    let claims = store.claims(name).unwrap_or_default();
    let mut md = format!(
        "# Audit {name}\n\n{} · {} · {}\n\ntargets: {}\nwork: {} {}\n\n| claim | state | level | text | source |\n|---|---|---|---|---|\n",
        audit.status.label(),
        if audit.mode == Mode::Diagnose { "diagnose" } else { "assure" },
        audit.board.as_ref().map(|b| format!("for {b}")).unwrap_or_else(|| "standing".into()),
        audit.targets.join(", "),
        audit.work,
        audit.branch,
    );
    for claim in &claims {
        md.push_str(&format!(
            "| {} | {} {} | {} | {} | {} |\n",
            claim.id,
            claim.mark().0,
            claim.verdict.label(),
            claim.level.label(),
            claim.text.replace('|', "/"),
            claim.source.replace('|', "/")
        ));
    }
    let dir = format!("{}/audits/{name}", crate::share::share_dir());
    let script = format!(
        "mkdir -p {d}/runs && printf %s '{}' | base64 -d > {d}/README.md",
        STANDARD.encode(md.as_bytes()),
        d = q(&dir)
    );
    let _ = machine(&script, 30);
}

// ------------------------------------------------------------------ check submission

/// A worker's `audit_check_submit`: the harness reads `check.toml`, files it
/// under `checks/c<N>/` (committed on the audit's branch), and, for a check
/// already relied on, holds a weaker one for the operator (rule 4).
pub fn submit_check(store: &AuditStore, audit_name: &str, claim_id: &str, dir: &str, by: &str) -> Reply {
    let audit = store.audit(audit_name).map_err(err)?.ok_or(format!("no audit {audit_name}"))?;
    let claim = store.claim(audit_name, claim_id).map_err(err)?.ok_or(format!("no claim {claim_id}"))?;
    if claim.taken && by != "operator" {
        return Err(format!("{claim_id} is the operator's: he writes its check"));
    }
    if matches!(claim.review, Review::Rejected | Review::Dropped) {
        return Err(format!("{claim_id} was {}", if claim.review == Review::Rejected { "rejected" } else { "dropped" }));
    }
    let target = format!("{}/checks/{claim_id}", audit.work);
    let src = if dir.starts_with('/') { dir.to_string() } else { format!("{}/{dir}", audit.work) };
    let script = format!(
        "s={src}; t={target}; [ -f \"$s/check.toml\" ] || {{ echo NOTOML; exit 4; }}\n\
         if [ \"$(cd \"$s\" && pwd)\" != \"$(mkdir -p \"$t\" && cd \"$t\" && pwd)\" ]; then rm -rf \"$t\" && mkdir -p \"$t\" && cp -a \"$s/.\" \"$t/\"; fi\n\
         echo __TOML__; cat \"$t/check.toml\"; echo __ANCHORS__; grep -rhoE '(//|\\\\\\*|#) *\\[[0-9]+\\]' \"$t\" 2>/dev/null | grep -oE '[0-9]+' | sort -un | tr '\\n' ' '; echo\n\
         echo __MUTANTS__; ls \"$t/mutants\" 2>/dev/null | tr '\\n' ' '; echo\n\
         cd {work} && git add checks/{claim_id} >/dev/null 2>&1 && git commit -q -m 'audit {audit_name}: check for {claim_id}' >/dev/null 2>&1; echo \"__COMMIT__ $(git rev-parse HEAD 2>/dev/null)\"",
        src = q(&src),
        target = q(&target),
        work = q(&audit.work),
    );
    let ran = machine(&script, 120)?;
    if ran.stdout.contains("NOTOML") {
        return Err(format!("no check.toml in {src}"));
    }
    let toml_text = between(&ran.stdout, "__TOML__", "__ANCHORS__");
    let parsed = audit::toml_flat(&toml_text).map_err(|e| format!("check.toml does not read: {e}"))?;
    let get = |key: &str| parsed.get(key).map(|v| v.text()).unwrap_or_default();
    let list = |key: &str| parsed.get(key).map(|v| v.list()).unwrap_or_default();
    let kind = get("kind");
    if !["test", "property", "bounded", "kani", "cbmc", "crosshair", "jbmc", "tla", "differential", "command", "proof"].contains(&kind.as_str()) {
        return Err(format!("kind {kind:?}: test, property, bounded, tla, differential, command or proof"));
    }
    let run = get("run");
    if run.trim().is_empty() {
        return Err("check.toml needs run: the command that runs it (exit 0 holds, 1 fails, 2 error)".into());
    }
    if !audit.settings.allows(&kind) && !audit.settings.allows(&get("tool")) {
        return Err(format!("{kind} is barred for this audit (tools {})", audit.settings.tools.join(",")));
    }
    crate::audit::adapters::usable(&get("lang"), &kind)?;
    let anchors: Vec<usize> = between(&ran.stdout, "__ANCHORS__", "__MUTANTS__").split_whitespace().filter_map(|n| n.parse().ok()).collect();
    // Every clause has an anchor and every anchor a clause.
    let clauses = audit::clauses(&claim.text).len();
    if clauses > 1 {
        let wanted: Vec<usize> = (1..=clauses).collect();
        if anchors != wanted {
            return Err(format!(
                "the claim has {clauses} clauses: mark each in the check with an anchor (// [1] … // [{clauses}], \\* [n] in TLA+); found {:?}",
                anchors
            ));
        }
    }
    let commit = ran.stdout.lines().find_map(|l| l.strip_prefix("__COMMIT__ ")).unwrap_or("").trim().to_string();
    let check = Check {
        claim: claim_id.into(),
        dir: format!("checks/{claim_id}"),
        lang: if get("lang").is_empty() { "any".into() } else { get("lang") },
        kind: kind.clone(),
        run,
        timeout: parsed.get("timeout").and_then(|v| v.text().parse::<u64>().ok()).map(|t| t.clamp(5, 7200)).unwrap_or(audit.settings.check_secs),
        targets: list("targets"),
        breaks: list("breaks"),
        coverage: get("coverage"),
        assumes: list("assumes"),
        anchors,
        by: by.into(),
        commit,
        ..Check::default()
    };
    // Rule 4: relied on means accepted, or cited by a report.
    let relied = claim.review == Review::Accepted || !claim.cited.is_empty();
    let previous = store.current_check(audit_name, claim_id).map_err(err)?;
    let check = store.add_check(audit_name, check).map_err(err)?;
    store
        .change_claim(audit_name, claim_id, |c| {
            c.held_by = None;
            c.held_until = None;
            c.rests_on = check.assumes.clone();
            if !(relied && previous.is_some()) {
                c.verdict = Verdict::Unchecked;
                c.level = Level::Documented;
            }
            c.tool = kind.clone();
            Ok(())
        })
        .map_err(err)??;
    store
        .log(audit_name, Who::Harness, &format!("{by} submitted a {kind} check for {claim_id} (v{})", check.version), &check.run)
        .map_err(err)?;
    if audit.settings.back_translation {
        back_translate_in_background(audit_name.to_string(), claim_id.to_string(), check.version);
    }
    if relied && let Some(previous) = previous {
        strengthen_in_background(audit_name.to_string(), claim_id.to_string(), previous, check.version);
        return Ok(json!({"ok": true, "version": check.version, "note": "This claim's verdict is relied on: the harness re-runs the old evidence against the new check before it replaces the old one."}).to_string());
    }
    wake();
    Ok(json!({"ok": true, "version": check.version, "note": "Filed. The harness vets and runs it when the audit's gate allows; the verdict is the harness's."}).to_string())
}

fn between(text: &str, start: &str, end: &str) -> String {
    let after = text.split_once(start).map(|(_, rest)| rest).unwrap_or("");
    after.split_once(end).map(|(inside, _)| inside).unwrap_or(after).trim().to_string()
}

// ------------------------------------------------------------------ batches

/// `audit-run <audit> <ids>`: one batch, one approval, every command typed
/// into the audit's own shell, each verdict drawn by the harness from the
/// exit code and shown there under it.
pub fn run_batch(store: &AuditStore, name: &str, ids: &[String], forced: bool) -> std::result::Result<(), String> {
    let audit = store.audit(name).map_err(err)?.ok_or("gone")?;
    if audit.work.is_empty() {
        return Err("its work tree is not set up yet".into());
    }
    // The steps it may spend.
    if !forced && audit.step_cap > 0 && audit.steps_used >= audit.step_cap {
        store
            .change(name, |a| {
                a.paused = true;
                Ok(())
            })
            .map_err(err)??;
        crate::terminal::warn_status(&format!("[audit] {name} spent its {} steps and paused: /audit set {name} steps=N, then z", audit.step_cap));
        return Ok(());
    }
    if audit.spent_secs >= audit.settings.total_secs {
        store
            .change(name, |a| {
                a.paused = true;
                Ok(())
            })
            .map_err(err)??;
        crate::terminal::warn_status(&format!("[audit] {name} spent its {} and paused", audit::show_secs(audit.settings.total_secs)));
        return Ok(());
    }
    let mut work: Vec<(Claim, Option<Check>)> = Vec::new();
    for id in ids {
        let Some(claim) = store.claim(name, id).map_err(err)? else { continue };
        let check = store.current_check(name, id).map_err(err)?;
        work.push((claim, check));
    }
    if work.is_empty() {
        return Ok(());
    }
    // What it will run, for the one approval.
    let mut planned = Vec::new();
    let mut estimate = 0u64;
    for (claim, check) in &work {
        if claim.repro == Repro::Pending {
            if let Some(test) = &claim.repro_test {
                planned.push(format!("{} repro: {test}", claim.id));
                estimate += 120;
            }
        } else if let Some(check) = check {
            planned.push(format!("{} {}: {}", claim.id, check.kind, check.run));
            let mutants = if audit.settings.mutants { mutant_names(&audit, check).len() as u64 } else { 0 };
            estimate += check.timeout.min(audit.settings.check_secs) / 4 * (1 + mutants);
        }
    }
    let detail = planned.join("\n");
    let what = format!("run audit {name}: {} check{}", work.len(), if work.len() == 1 { "" } else { "s" });
    let always = estimate > audit.settings.ask_over_secs;
    crate::approval::ask_as(&format!("audit {name}"), &what, &detail, always)?;
    let transcript = format!("{}/audits/{name}/runs", crate::share::share_dir());
    let mut run = store
        .new_run(
            name,
            Run {
                claims: ids.to_vec(),
                started_at: audit::now(),
                approval: if crate::approval::enabled() || always { "approved by the operator".into() } else { "approval mode off".into() },
                ..Run::default()
            },
        )
        .map_err(err)?;
    run.transcript = format!("{transcript}/{}.log", run.n);
    let mut log = format!(
        "# audit {name} · batch {} · {}\n# claims: {}\n# {}\n",
        run.n,
        run.started_at,
        ids.join(" "),
        run.approval
    );
    let started = Instant::now();
    for (claim, check) in work {
        if store.audit(name).ok().flatten().is_some_and(|a| a.paused || a.wheel_paused || a.status == Status::Stopped) {
            log.push_str("# paused: the rest waits\n");
            break;
        }
        if claim.repro == Repro::Pending {
            run_repro(store, &audit, &claim, &mut run, &mut log)?;
            continue;
        }
        let Some(check) = check else { continue };
        store
            .change_claim(name, &claim.id, |c| {
                c.running = true;
                Ok(())
            })
            .map_err(err)??;
        let outcome = check_claim(store, &audit, &claim, &check, &mut run, &mut log);
        let _ = store.change_claim(name, &claim.id, |c| {
            c.running = false;
            Ok(())
        });
        outcome?;
        spend_step(store, &audit);
    }
    run.finished_at = Some(audit::now());
    store.put_run(name, &run).map_err(err)?;
    let secs = started.elapsed().as_secs();
    store
        .change(name, |a| {
            a.spent_secs += secs;
            Ok(())
        })
        .map_err(err)??;
    let _ = machine(
        &format!("mkdir -p {d} && printf %s '{}' | base64 -d > {d}/{}.log", STANDARD.encode(log.as_bytes()), run.n, d = q(&transcript)),
        30,
    );
    write_share(store, name);
    Ok(())
}

fn spend_step(store: &AuditStore, audit: &Audit) {
    let _ = store.change(&audit.name, |a| {
        a.steps_used += 1;
        Ok(())
    });
    if let (Some(board), Some(boards)) = (&audit.board, audit::boards())
        && let Ok(Some(found)) = boards.board(board)
    {
        let _ = boards.spend_steps(&found.id, 1);
    }
}

/// What one command in the shell came to.
pub struct Ran {
    pub exit: Option<i32>,
    pub output: String,
    pub ms: u64,
    pub interrupted: bool,
    pub typed: Option<crate::vm::netbox_run::Typed>,
}

/// One command typed into the audit's shell under `audit <claim>`, waited
/// for. Ctrl+\ in it pauses the audit (the operator has control).
pub fn in_shell(audit: &Audit, claim: &str, why: &str, command: &str, timeout: u64) -> std::result::Result<Ran, String> {
    let full = format!("cd {} && {command}", q(&audit.work));
    #[cfg(test)]
    if let Some((exit, output)) = fake(&full) {
        return Ok(Ran { exit, output, ms: 1, interrupted: false, typed: None });
    }
    let ask = crate::vm::netbox_run::Ask {
        command: full,
        why: why.into(),
        persist: false,
        timeout_secs: timeout,
        netbox: Some(audit.shell()),
        worker: None,
        approved: true,
        label: Some(format!("audit {claim}")),
        no_lane: true,
    };
    // Its shell finishing the command before (its verdict line drawn) is
    // busy for a moment: waited for, briefly.
    let mut tries = 0;
    let first = loop {
        match crate::vm::netbox_run::start(&ask, &mut crate::vm::netboxes::machine) {
            Ok(crate::vm::netbox_run::Started::Elsewhere(why)) if why.contains("was busy") && tries < 50 => {
                tries += 1;
                std::thread::sleep(Duration::from_millis(300));
            }
            other => break other,
        }
    };
    let started = match first {
        Ok(crate::vm::netbox_run::Started::Typed(typed)) => typed,
        Ok(crate::vm::netbox_run::Started::Elsewhere(why)) => {
            if why.contains("has control") {
                if let Some(store) = audit::store() {
                    let _ = store.change(&audit.name, |a| {
                        a.wheel_paused = true;
                        Ok(())
                    });
                    let _ = store.log(&audit.name, Who::Operator, "took control of its shell: paused", "");
                }
                return Err("paused: the operator has control of the audit's shell".into());
            }
            return Err(format!("its shell could not take the command: {why}"));
        }
        Err(error) => {
            // Its shell was closed: open it again, once.
            open_shell(audit)?;
            match crate::vm::netbox_run::start(&ask, &mut crate::vm::netboxes::machine)? {
                crate::vm::netbox_run::Started::Typed(typed) => typed,
                crate::vm::netbox_run::Started::Elsewhere(why) => return Err(format!("{error}; then {why}")),
            }
        }
    };
    let begun = Instant::now();
    loop {
        match crate::vm::netbox_run::wait(&started, &mut crate::vm::netboxes::machine, false, crate::vm::netbox_run::CONFIRM_SECS)? {
            crate::vm::netbox_run::Waited::Done(done) => {
                return Ok(Ran {
                    exit: if done.declined { None } else { done.exit_code.map(|e| e as i32) },
                    output: done.output,
                    ms: done.duration_ms as u64,
                    interrupted: done.interrupted || done.declined,
                    typed: Some(started),
                });
            }
            crate::vm::netbox_run::Waited::TimedOut => {
                let _ = crate::vm::netbox_run::interrupt(&started, &mut crate::vm::netboxes::machine);
                return Ok(Ran { exit: Some(124), output: "timed out".into(), ms: begun.elapsed().as_millis() as u64, interrupted: false, typed: Some(started) });
            }
            crate::vm::netbox_run::Waited::Gone => return Err("its shell went away mid-command".into()),
            _ => {
                if begun.elapsed().as_secs() > timeout + crate::vm::netbox_run::CONFIRM_SECS {
                    return Err("no end in sight".into());
                }
            }
        }
    }
}

/// The verdict line under a command in the shell.
fn say(ran: &Ran, text: &str) {
    if let Some(typed) = &ran.typed {
        let _ = crate::vm::netbox_run::verdict(typed, text, &mut crate::vm::netboxes::machine);
    }
}

/// The exit code contract (capability-eval's): 0 holds, 1 fails, 2 error.
pub fn verdict_of(exit: Option<i32>, interrupted: bool) -> (Verdict, &'static str) {
    if interrupted {
        return (Verdict::Error, "error · interrupted by the operator");
    }
    match exit {
        Some(0) => (Verdict::Holds, "holds"),
        Some(1) => (Verdict::Fails, "fails"),
        Some(124) => (Verdict::Error, "error · timed out"),
        _ => (Verdict::Error, "error"),
    }
}

/// One claim's check: run, then vetted (mutants must be caught), then the
/// verdict and level written, and posted as `post` allows.
fn check_claim(store: &AuditStore, audit: &Audit, claim: &Claim, check: &Check, run: &mut Run, log: &mut String) -> std::result::Result<(), String> {
    let why = if claim.why.is_empty() { crate::util::clip(&claim.text, 80) } else { crate::util::clip(&claim.why, 80) };
    let ran = in_shell(audit, &claim.id, &why, &check.run, check.timeout.min(audit.settings.check_secs))?;
    let (mut verdict, said) = verdict_of(ran.exit, ran.interrupted);
    let trace = if verdict == Verdict::Fails { Some(format!("runs/{}/{}.trace", run.n, claim.id)) } else { None };
    say(&ran, &format!("audit {}: exit {}: {said}{}", claim.id, ran.exit.map_or("-".into(), |e| e.to_string()), trace.as_ref().map(|t| format!(" · trace {t}")).unwrap_or_default()));
    log.push_str(&format!("$ {}\n{}\n# audit {}: exit {}: {said}\n", check.run, ran.output, claim.id, ran.exit.map_or("-".into(), |e| e.to_string())));
    run.commands.push(RunCommand { claim: claim.id.clone(), what: "check".into(), command: check.run.clone(), exit: ran.exit, verdict: said.into(), ms: ran.ms, by_operator: false });
    store.put_run(&audit.name, run).map_err(err)?;
    if let Some(trace) = &trace {
        // The counterexample, as the check printed it.
        let _ = store.add_evidence(
            &audit.name,
            &Evidence { claim: claim.id.clone(), kind: "trace".into(), name: trace.clone(), path: trace.clone(), result: crate::util::clip(&ran.output, 20_000), at: audit::now() },
        );
    }
    // Vetting: a check that holds must catch mutants.
    let mut level = Level::of_kind(&check.kind);
    let mut kills = None;
    let mut killed = Vec::new();
    let mut survived = Vec::new();
    if verdict == Verdict::Holds {
        let mutants = if audit.settings.mutants { mutant_names(audit, check) } else { Vec::new() };
        if !mutants.is_empty() {
            for mutant in &mutants {
                let caught = run_mutant(audit, claim, check, mutant, run, log)?;
                if caught { killed.push(mutant.clone()) } else { survived.push(mutant.clone()) }
                let _ = store.add_evidence(
                    &audit.name,
                    &Evidence {
                        claim: claim.id.clone(),
                        kind: "mutant".into(),
                        name: mutant.clone(),
                        path: format!("{}/mutants/{mutant}", check.dir),
                        result: if caught { "killed".into() } else { "survived".into() },
                        at: audit::now(),
                    },
                );
            }
            kills = Some((killed.len(), mutants.len()));
            if killed.is_empty() {
                verdict = Verdict::Vacuous;
                level = Level::Documented;
            }
        } else if audit.settings.mutants {
            // Nothing to catch: not vetted, so it counts as documented.
            verdict = Verdict::Vacuous;
            level = Level::Documented;
            log.push_str(&format!("# audit {}: no mutants to catch: vacuous until it has some\n", claim.id));
        }
    } else {
        level = Level::Documented;
    }
    // Capped by the audit's depth and the language's ceiling.
    if let Some(depth) = audit.settings.depth
        && level > depth
        && level != Level::Model
    {
        level = depth;
    }
    // Coverage, when the check says how: the lines its run reached, for the
    // claim view's gutter. It never changes the verdict.
    if !check.coverage.trim().is_empty() && !ran.interrupted {
        let command = format!(
            "COV=$(mktemp) && export COV && ( {} ) >/dev/null 2>&1; awk -F'[:,]' '/^SF:/{{f=$2}} /^DA:/{{if ($3>0) print f\":\"$2}}' \"$COV\" | sed \"s#^$PWD/##\" > {}/coverage.lines; rm -f \"$COV\"",
            check.coverage,
            q(&check.dir)
        );
        let cov = in_shell(audit, &claim.id, "coverage for the claim view (no verdict)", &command, check.timeout.min(audit.settings.check_secs))?;
        run.commands.push(RunCommand { claim: claim.id.clone(), what: "coverage".into(), command, exit: cov.exit, verdict: "no verdict".into(), ms: cov.ms, by_operator: false });
    }
    let mut stored = check.clone();
    stored.kills = kills;
    stored.killed = killed;
    stored.survived = survived;
    store.put_check(&audit.name, &stored).map_err(err)?;
    store
        .change_claim(&audit.name, &claim.id, |c| {
            c.verdict = verdict;
            c.level = level;
            c.kills = kills;
            c.checked_at = Some(audit::now());
            c.trace = trace.clone();
            if verdict == Verdict::Fails && c.repro == Repro::None {
                c.repro = Repro::None;
            }
            if verdict == Verdict::Holds && c.fixed.is_none() && c.repro == Repro::Fails {
                c.fixed = Some(audit.commit.clone());
            }
            Ok(())
        })
        .map_err(err)??;
    store
        .log(&audit.name, Who::Harness, &format!("{} {} ({})", claim.id, verdict.label(), level.label()), &kills.map(|(k, t)| format!("killed {k} of {t} mutants")).unwrap_or_default())
        .map_err(err)?;
    // An assumption that fails makes every claim resting on it unsound.
    if claim.is_assumption() && verdict == Verdict::Fails {
        crate::audit::scope::mark_unsound(store, &audit.name, &claim.id);
    }
    if verdict == Verdict::Fails {
        crate::audit::ops::note(format!(
            "{}/{} fails ({}): a worker can turn the counterexample into a repro (audit_repro_submit), or say why not (audit_model_only)",
            audit.name, claim.id, check.kind
        ));
    }
    crate::audit::ops::post_result(store, &audit.name, &claim.id, false).map_err(err)?;
    crate::audit::scope::discharge(store, &audit.name, &claim.id);
    Ok(())
}

/// The mutants a check is vetted against: patches beside it
/// (`checks/c<N>/mutants/*.patch`), which the harness applies and runs.
fn mutant_names(audit: &Audit, check: &Check) -> Vec<String> {
    let script = format!("ls {}/{}/mutants 2>/dev/null | grep -E '\\.(patch|diff)$' | head -20", q(&audit.work), check.dir);
    machine(&script, 20).map(|ran| ran.stdout.lines().map(str::to_string).collect()).unwrap_or_default()
}

/// One mutant: applied, the check run, reverted. Caught when the check no
/// longer holds.
fn run_mutant(audit: &Audit, claim: &Claim, check: &Check, mutant: &str, run: &mut Run, log: &mut String) -> std::result::Result<bool, String> {
    let patch = format!("{}/mutants/{mutant}", check.dir);
    let command = format!(
        "patch -s -p1 -f < {patch} >/dev/null && {{ {run}; s=$?; patch -s -R -p1 -f < {patch} >/dev/null; [ $s -eq 0 ] && exit 0 || exit 1; }}",
        patch = q(&patch),
        run = check.run
    );
    let ran = in_shell(audit, &claim.id, &format!("mutant {mutant}: must be caught"), &command, check.timeout.min(audit.settings.check_secs))?;
    let caught = ran.exit.is_some_and(|e| e != 0) && !ran.interrupted;
    say(&ran, &format!("audit {}: mutant {mutant} {}", claim.id, if caught { "killed" } else { "survived" }));
    log.push_str(&format!("$ {command}\n{}\n# mutant {mutant}: {}\n", ran.output, if caught { "killed" } else { "survived" }));
    run.commands.push(RunCommand { claim: claim.id.clone(), what: format!("mutant {mutant}"), command, exit: ran.exit, verdict: if caught { "killed".into() } else { "survived".into() }, ms: ran.ms, by_operator: false });
    Ok(caught)
}

// ------------------------------------------------------------------ repros

/// A worker's `audit_repro_submit`: a test in the repo's own framework. The
/// harness runs it on the audited code; it is a finding only if it fails
/// there (rule 2).
pub fn submit_repro(store: &AuditStore, audit_name: &str, claim_id: &str, test: &str, run: &str) -> Reply {
    let claim = store.claim(audit_name, claim_id).map_err(err)?.ok_or(format!("no claim {claim_id}"))?;
    if claim.verdict != Verdict::Fails {
        return Err(format!("{claim_id} has no failing check to reproduce"));
    }
    if run.trim().is_empty() {
        return Err("run: the command that runs the repro test".into());
    }
    store
        .change_claim(audit_name, claim_id, |c| {
            c.repro = Repro::Pending;
            c.repro_test = Some(format!("{test} :: {run}"));
            Ok(())
        })
        .map_err(err)??;
    store.log(audit_name, Who::S0ma, &format!("a repro for {claim_id} was submitted"), test).map_err(err)?;
    wake();
    Ok(json!({"ok": true, "note": "The harness runs it on the audited code: it becomes a finding only if it fails there."}).to_string())
}

fn run_repro(store: &AuditStore, audit: &Audit, claim: &Claim, run: &mut Run, log: &mut String) -> std::result::Result<(), String> {
    let Some(spec) = &claim.repro_test else { return Ok(()) };
    let (test, command) = spec.split_once(" :: ").unwrap_or((spec.as_str(), spec.as_str()));
    let ran = in_shell(audit, &claim.id, &format!("repro {test}: must fail on the audited code"), command, audit.settings.check_secs)?;
    let reproduced = ran.exit.is_some_and(|e| e != 0 && e != 124) && !ran.interrupted;
    say(&ran, &format!("audit {}: repro exit {}: {}", claim.id, ran.exit.map_or("-".into(), |e| e.to_string()), if reproduced { "reproduced" } else { "not reproduced" }));
    log.push_str(&format!("$ {command}\n{}\n# repro {}: {}\n", ran.output, claim.id, if reproduced { "reproduced" } else { "not reproduced" }));
    run.commands.push(RunCommand { claim: claim.id.clone(), what: "repro".into(), command: command.into(), exit: ran.exit, verdict: if reproduced { "reproduced".into() } else { "passes".into() }, ms: ran.ms, by_operator: false });
    store
        .change_claim(&audit.name, &claim.id, |c| {
            c.repro = if reproduced { Repro::Fails } else { Repro::Passes };
            c.repro_test = Some(test.to_string() + " :: " + command);
            c.repro_output = Some(crate::util::clip(ran.output.trim(), 2000));
            Ok(())
        })
        .map_err(err)??;
    let _ = store.add_evidence(
        &audit.name,
        &Evidence { claim: claim.id.clone(), kind: "repro".into(), name: test.into(), path: test.into(), result: if reproduced { "fails".into() } else { "passes".into() }, at: audit::now() },
    );
    store.log(&audit.name, Who::Harness, &format!("{}'s repro {}", claim.id, if reproduced { "fails on the audited code: reproduced" } else { "passes: not a repro" }), test).map_err(err)?;
    if reproduced {
        crate::audit::ops::post_result(store, &audit.name, &claim.id, false).map_err(err)?;
    }
    Ok(())
}

/// A worker's `audit_model_only`: no repro can be written. The operator
/// sees `[!] model-only` and accepts the reason; it is never a finding.
pub fn model_only(store: &AuditStore, audit_name: &str, claim_id: &str, why: &str) -> Reply {
    let audit = store.audit(audit_name).map_err(err)?.ok_or("gone")?;
    store
        .change_claim(audit_name, claim_id, |c| {
            if c.verdict != Verdict::Fails {
                return Err(format!("{claim_id} has no failing check"));
            }
            c.repro = Repro::ModelOnly;
            c.model_only_why = Some(why.to_string());
            Ok(())
        })
        .map_err(err)??;
    store.log(audit_name, Who::S0ma, &format!("{claim_id} is model-only"), why).map_err(err)?;
    // On the board, an open question under the audit's lead.
    if let (Some(board), Some(boards)) = (&audit.board, audit::boards())
        && let Ok(Some(found)) = boards.board(board)
    {
        let _ = boards.declare_intents(
            &found.id,
            &[(format!("is {audit_name}/{claim_id}'s model-only trace real? ({})", crate::util::clip(why, 120)), "a counterexample the harness could not reproduce".into(), Vec::new(), 4)],
            usize::MAX,
            chrono::Local::now(),
        );
    }
    Ok(json!({"ok": true, "note": "Shown to the operator as [!] model-only; it is never a finding."}).to_string())
}

// ------------------------------------------------------------------ back-translation and strengthen-only

/// A fresh model call sees only the check's files, never the claim, and says
/// in plain English what the check guarantees.
fn back_translate_in_background(audit: String, claim: String, version: i64) {
    if cfg!(test) {
        return;
    }
    let _ = std::thread::Builder::new().name("audit-back".into()).spawn(move || {
        let Some(store) = audit::store() else { return };
        let Ok(Some(a)) = store.audit(&audit) else { return };
        let script = format!(
            "cd {}/checks/{claim} && for f in $(find . -type f ! -path './mutants/*' | head -12); do echo \"==== $f\"; head -c 6000 \"$f\"; echo; done",
            q(&a.work)
        );
        let Ok(files) = machine(&script, 30) else { return };
        let messages = [
            crate::types::Message::system(
                "You read a check (a test, property, model or proof harness) and say, in two or three plain English sentences, \
                 exactly what it guarantees when it passes: what is quantified over, what is asserted, and what it leaves out. \
                 You are not told what it was meant to check."
                    .into(),
            ),
            crate::types::Message::user(crate::util::clip(&files.stdout, 30_000)),
        ];
        let options = crate::llm::ChatOptions::instruct();
        let Ok(response) = crate::llm::chat(&messages, &options, |_| Ok(())) else { return };
        if let Ok(checks) = store.checks(&audit, &claim)
            && let Some(mut check) = checks.into_iter().find(|c| c.version == version)
        {
            check.back_translation = Some(response.content.trim().to_string());
            let _ = store.put_check(&audit, &check);
        }
    });
}

/// Rule 4: a new check for a relied-on claim must still kill every mutant
/// the old one killed, and still fail every recorded repro whose bug is not
/// fixed; a `tla` check must keep the old invariant. Else it waits as
/// `[!] weaker`.
fn strengthen_in_background(audit: String, claim: String, old: Check, version: i64) {
    let work = move || -> std::result::Result<(), String> {
        let store = audit::store().ok_or("no store")?;
        let a = store.audit(&audit).map_err(err)?.ok_or("gone")?;
        let checks = store.checks(&audit, &claim).map_err(err)?;
        let mut new = checks.into_iter().find(|c| c.version == version).ok_or("gone")?;
        let c = store.claim(&audit, &claim).map_err(err)?.ok_or("gone")?;
        let mut lost = Vec::new();
        let mut run = Run { claims: vec![claim.clone()], started_at: audit::now(), note: "strengthen-only".into(), ..Run::default() };
        let mut log = String::new();
        for mutant in &old.killed {
            if !run_mutant(&a, &c, &new, mutant, &mut run, &mut log)? {
                lost.push(format!("mutant {mutant}"));
            }
        }
        if c.repro == Repro::Fails && c.fixed.is_none() {
            let ran = in_shell(&a, &claim, "the new check must still fail on the unfixed code", &new.run, new.timeout)?;
            if ran.exit != Some(1) {
                lost.push("the recorded failure".into());
            }
        }
        if old.kind == "tla" && new.kind == "tla" {
            let script = format!("cd {} && cat {}/*.cfg 2>/dev/null; echo ====; cat {}/*.cfg 2>/dev/null", q(&a.work), old.dir, new.dir);
            if let Ok(ran) = machine(&script, 20) {
                let (before, after) = ran.stdout.split_once("====").unwrap_or(("", ""));
                for line in before.lines().filter(|l| l.trim_start().starts_with("INVARIANT") || l.trim_start().starts_with("PROPERTY")) {
                    if !after.contains(line.trim()) {
                        lost.push(format!("the invariant {}", line.trim()));
                    }
                }
            }
        }
        if lost.is_empty() {
            for mut check in store.checks(&audit, &claim).map_err(err)? {
                if check.version < version && check.superseded_by.is_none() {
                    check.superseded_by = Some(version);
                    store.put_check(&audit, &check).map_err(err)?;
                }
            }
            store
                .change_claim(&audit, &claim, |c| {
                    c.verdict = Verdict::Unchecked;
                    Ok(())
                })
                .map_err(err)??;
            store.log(&audit, Who::Harness, &format!("{claim}'s new check is at least as strong: it replaces v{}", old.version), "").map_err(err)?;
        } else {
            new.held_weaker = true;
            store.put_check(&audit, &new).map_err(err)?;
            let why = format!("weaker: {} lost", lost.join(", "));
            store
                .change_claim(&audit, &claim, |c| {
                    c.weaker = Some(why.clone());
                    Ok(())
                })
                .map_err(err)??;
            store.log(&audit, Who::Harness, &format!("{claim}'s new check is held: {why}"), "").map_err(err)?;
        }
        wake();
        Ok(())
    };
    // Tests: at once, on this thread.
    if cfg!(test) {
        let _ = work();
        return;
    }
    let _ = std::thread::Builder::new().name("audit-strengthen".into()).spawn(move || {
        if let Err(error) = work() {
            crate::terminal::warn_status(&format!("[audit] strengthen-only check could not run: {error}"));
        }
    });
}

/// The operator accepts a weaker check (`a` twice): it replaces the old one.
pub fn accept_weaker(store: &AuditStore, audit: &str, claim: &str) -> Reply {
    let checks = store.checks(audit, claim).map_err(err)?;
    if let Some(held) = checks.iter().rev().find(|c| c.held_weaker).cloned() {
        for mut check in checks {
            if check.version < held.version && check.superseded_by.is_none() {
                check.superseded_by = Some(held.version);
                store.put_check(audit, &check).map_err(err)?;
            }
        }
        let mut held = held;
        held.held_weaker = false;
        store.put_check(audit, &held).map_err(err)?;
        store
            .change_claim(audit, claim, |c| {
                c.verdict = Verdict::Unchecked;
                Ok(())
            })
            .map_err(err)??;
    }
    let said = crate::audit::ops::operator_claim(store, audit, claim, "accept-weaker", "")?;
    wake();
    Ok(said)
}

// ------------------------------------------------------------------ re-runs

/// `/audit rerun` (or `r` with nothing selected): the target's current code
/// brought in, every check in scope run again, stale claims marked. Only when
/// the operator starts it.
pub fn rerun(store: &AuditStore, name: &str, who: Who) -> Reply {
    if who != Who::Operator {
        return Err("a re-run is the operator's to start: propose one with audit_rerun_propose".into());
    }
    let audit = store.audit(name).map_err(err)?.ok_or(format!("no audit {name}"))?;
    if audit.work.is_empty() {
        return Err(format!("{name} is not set up yet"));
    }
    store
        .change(name, |a| {
            a.rerun_proposed = None;
            if matches!(a.status, Status::Done | Status::Standing | Status::Stopped) {
                a.status = Status::Checking;
            }
            a.paused = false;
            if a.gone_at.is_none() {
                a.gone_at = Some(audit::now());
            }
            Ok(())
        })
        .map_err(err)??;
    let name = name.to_string();
    let _ = std::thread::Builder::new().name("audit-rerun".into()).spawn(move || {
        let Some(store) = audit::store() else { return };
        if let Err(error) = rerun_now(&store, &name) {
            crate::terminal::warn_status(&format!("[audit] {name}: re-run: {error}"));
        }
    });
    Ok(format!("[audit] re-running {}: the current code, then every check in scope", audit.name))
}

fn rerun_now(store: &AuditStore, name: &str) -> std::result::Result<(), String> {
    let audit = store.audit(name).map_err(err)?.ok_or("gone")?;
    let script = if audit.kind == "git" {
        format!(
            "cd {w} && top=$(git -C {t} rev-parse --show-toplevel) && git merge -q --no-edit \"$(git -C \"$top\" rev-parse HEAD)\" >/dev/null 2>&1; git rev-parse HEAD",
            w = q(&audit.work),
            t = q(audit.targets.first().map(String::as_str).unwrap_or("."))
        )
    } else {
        format!(
            "cd {w} && for x in {targets}; do cp -a \"$x\" ./ 2>/dev/null || cp -a \"$x/.\" ./; done; echo -",
            w = q(&audit.work),
            targets = audit.targets.iter().map(|t| q(t)).collect::<Vec<_>>().join(" ")
        )
    };
    let ran = machine(&script, 300)?;
    let commit = ran.stdout.lines().last().unwrap_or("-").trim().to_string();
    store
        .change(name, |a| {
            if commit != "-" {
                a.commit = commit.clone();
            }
            Ok(())
        })
        .map_err(err)??;
    // Stale: a source quote that no longer appears where it was.
    for claim in store.claims(name).map_err(err)? {
        if let Some(stale) = crate::audit::scope::stale(&audit, &claim) {
            store
                .change_claim(name, &claim.id, |c| {
                    c.stale = Some(stale.clone());
                    c.review = Review::Proposed;
                    Ok(())
                })
                .map_err(err)??;
        }
        // Everything with a check runs again.
        if store.current_check(name, &claim.id).map_err(err)?.is_some() {
            store
                .change_claim(name, &claim.id, |c| {
                    c.verdict = Verdict::Unchecked;
                    c.posted = audit::Posted::No;
                    Ok(())
                })
                .map_err(err)??;
        }
    }
    store.log(name, Who::Operator, "re-ran it on the current code", &commit).map_err(err)?;
    wake();
    Ok(())
}

/// The agent's `audit_rerun_propose`: a standing audit's code changed. The
/// proposal waits on `u audit [!]` until the operator starts it.
pub fn propose_rerun(store: &AuditStore, name: &str, why: &str) -> Reply {
    store
        .change(name, |a| {
            a.rerun_proposed = Some(why.to_string());
            Ok(())
        })
        .map_err(err)??;
    store.log(name, Who::S0ma, "proposed a re-run", why).map_err(err)?;
    Ok(json!({"ok": true, "note": "Waiting on the audit tab's [!] until the operator starts it (r)."}).to_string())
}

// ------------------------------------------------------------------ after the project, and gates

/// At complete or discard: let go (the default for diagnose), keep (a
/// standing audit in assure mode; unreviewed claims back to triage, counting
/// for nothing), or merge into a standing audit on the same target.
pub fn after_project(store: &AuditStore, name: &str, action: &str, into: Option<&str>) -> Reply {
    let audit = store.audit(name).map_err(err)?.ok_or(format!("no audit {name}"))?;
    match action {
        "letgo" => {
            crate::audit::ops::close_lead(&audit, "let go");
            store
                .change(name, |a| {
                    a.status = Status::Closed;
                    Ok(())
                })
                .map_err(err)??;
            store.log(name, Who::Operator, "let it go", "").map_err(err)?;
            let repros: Vec<String> = store
                .claims(name)
                .map_err(err)?
                .into_iter()
                .filter(|c| c.repro == Repro::Fails)
                .filter_map(|c| c.repro_test.map(|t| format!("{}: {t}", c.id)))
                .collect();
            Ok(format!(
                "[audit] {name} let go; its files stay in /share/audits/{name}/{}",
                if repros.is_empty() { String::new() } else { format!(". Regression tests offered: {}", repros.join("; ")) }
            ))
        }
        "keep" => {
            for claim in store.claims(name).map_err(err)? {
                if claim.review == Review::Proposed {
                    // Results shown, counted for nothing until accepted.
                    store
                        .change_claim(name, &claim.id, |c| {
                            c.review = Review::Proposed;
                            Ok(())
                        })
                        .map_err(err)??;
                }
            }
            store
                .change(name, |a| {
                    a.kept_from = a.board.take();
                    a.lead = None;
                    a.mode = Mode::Assure;
                    a.status = Status::Standing;
                    a.paused = false;
                    Ok(())
                })
                .map_err(err)??;
            store.log(name, Who::Operator, "kept it as a standing audit", "unreviewed claims went to triage").map_err(err)?;
            Ok(format!("[audit] {name} kept: a standing audit. Its unreviewed claims are [?] and count for nothing until accepted"))
        }
        "merge" => {
            let into = into.ok_or("merge into which standing audit?")?;
            let target = store.audit(into).map_err(err)?.ok_or(format!("no audit {into}"))?;
            if !target.standing() {
                return Err(format!("{into} is not a standing audit"));
            }
            let mut moved = 0;
            for claim in store.claims(name).map_err(err)? {
                let mut copy = claim.clone();
                if claim.review != Review::Accepted {
                    copy.review = Review::Proposed;
                }
                copy.source = format!("{} (from {name}/{})", claim.source, claim.id);
                store.add_claim(into, copy).map_err(err)?;
                moved += 1;
            }
            after_project(store, name, "letgo", None)?;
            store.log(into, Who::Operator, &format!("merged {name} into it ({moved} claims)"), "").map_err(err)?;
            Ok(format!("[audit] {name} merged into {into}: {moved} claims (unaccepted ones to triage)"))
        }
        _ => Err("keep, letgo or merge".into()),
    }
}

/// `/board gate <board> <audit>`: the project's verification also runs
/// `audit-run <audit> --holding`.
pub fn gate(store: &AuditStore, board: &str, audit: &str) -> Reply {
    store.audit(audit).map_err(err)?.ok_or(format!("no audit {audit}"))?;
    let existing = store.links(Some(audit), Some(board)).map_err(err)?;
    if existing.iter().any(|l| l.kind == "gate") {
        store.remove_links(audit, board, Some("gate"), None).map_err(err)?;
        return Ok(format!("[audit] {board}'s verification no longer runs audit-run {audit} --holding"));
    }
    store
        .add_link(&audit::Link { audit: audit.into(), board: board.into(), kind: "gate".into(), by: "operator".into(), at: audit::now(), ..audit::Link::default() })
        .map_err(err)?;
    Ok(format!("[audit] {board}'s verification now runs audit-run {audit} --holding: a fix can't reach complete if it breaks a claim that held"))
}

/// The gates on a board, for its verification.
pub fn gates(board: &str) -> Vec<String> {
    audit::store().and_then(|s| s.links(None, Some(board)).ok()).unwrap_or_default().into_iter().filter(|l| l.kind == "gate").map(|l| l.audit).collect()
}

/// `/audit start-all <dir>`: a proposed standing audit for each git repo
/// under `dir`, with its ceiling preview, for the operator to pick from.
pub fn start_all(store: &AuditStore, dir: &str) -> Reply {
    let script = format!("find {} -maxdepth 3 -name .git -type d 2>/dev/null | sed 's#/.git$##' | head -40", q(dir));
    let ran = machine(&script, 60)?;
    let mut proposed = Vec::new();
    for repo in ran.stdout.lines() {
        let base: String = repo.rsplit('/').next().unwrap_or("repo").to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').take(16).collect();
        if base.is_empty() || store.audit(&base).ok().flatten().is_some() {
            continue;
        }
        let languages = crate::audit::adapters::preview(repo, &[]).unwrap_or_default();
        let audit = Audit {
            name: base.clone(),
            mode: Mode::Assure,
            status: Status::Proposed,
            targets: vec![repo.to_string()],
            why: format!("start-all of {dir}"),
            estimate: languages.join(" · "),
            languages,
            created_at: audit::now(),
            ..Audit::default()
        };
        if store.create(&audit).is_ok() {
            proposed.push(format!("{base} ({})", audit.estimate));
        }
    }
    Ok(format!(
        "[audit] proposed {} standing audit{}: {} — start the ones you want in Alt+U (a twice)",
        proposed.len(),
        if proposed.len() == 1 { "" } else { "s" },
        proposed.join(", ")
    ))
}

// ------------------------------------------------------------------ audit-run as a command

/// `audit-run <audit> --all-checked | --holding | --repro c<N>`, as a board's
/// goal test or its verification runs it: the harness runs it, and its exit
/// is the answer.
pub fn command(line: &str) -> Option<crate::board::TestRun> {
    let words: Vec<&str> = line.split_whitespace().collect();
    if words.first() != Some(&"audit-run") {
        return None;
    }
    let started = Instant::now();
    let answer = (|| -> std::result::Result<(i32, String), String> {
        let store = audit::store().ok_or("no audit store")?;
        let name = words.get(1).ok_or("audit-run <audit> --all-checked | --holding | --repro c<N> | --cited <ids>")?;
        let audit = store.audit(name).map_err(err)?.ok_or(format!("no audit {name}"))?;
        let claims = store.claims(name).map_err(err)?;
        match words.get(2).copied() {
            Some("--all-checked") => {
                let open: Vec<&str> = claims
                    .iter()
                    .filter(|c| c.review == Review::Accepted && !c.waived && matches!(c.verdict, Verdict::Unchecked | Verdict::Error))
                    .map(|c| c.id.as_str())
                    .collect();
                Ok(if open.is_empty() { (0, "every accepted claim is checked".into()) } else { (1, format!("not checked yet: {}", open.join(" "))) })
            }
            Some("--holding") => {
                let held: Vec<String> = claims.iter().filter(|c| c.verdict == Verdict::Holds && !c.waived).map(|c| c.id.clone()).collect();
                let mut broke = Vec::new();
                for id in &held {
                    let Some(check) = store.current_check(name, id).map_err(err)? else { continue };
                    let ran = in_shell(&audit, id, "verification: must still hold", &check.run, check.timeout)?;
                    if ran.exit != Some(0) {
                        broke.push(id.clone());
                    }
                }
                Ok(if broke.is_empty() { (0, format!("{} claims still hold", held.len())) } else { (1, format!("no longer hold: {}", broke.join(" "))) })
            }
            Some("--repro") => {
                let id = words.get(3).ok_or("--repro c<N>")?;
                let claim = claims.iter().find(|c| c.id == *id).ok_or(format!("no claim {id}"))?;
                let check = store.current_check(name, id).map_err(err)?.ok_or(format!("{id} has no check"))?;
                let ran = in_shell(&audit, id, &format!("{id} holds again?"), &check.run, check.timeout)?;
                let mut code = if ran.exit == Some(0) { 0 } else { 1 };
                if code == 0
                    && let Some(spec) = &claim.repro_test
                {
                    let command = spec.split_once(" :: ").map(|(_, c)| c).unwrap_or(spec);
                    let repro = in_shell(&audit, id, "its repro must pass now", command, check.timeout)?;
                    if repro.exit != Some(0) {
                        code = 1;
                    }
                }
                Ok((code, format!("{id}: {}", if code == 0 { "holds again" } else { "still fails" })))
            }
            Some("--cited") => {
                let ids: Vec<&str> = words[3..].to_vec();
                crate::audit::report::rerun_cited(&store, name, &ids).map(|(ok, said)| (i32::from(!ok), said))
            }
            _ => Err("audit-run <audit> --all-checked | --holding | --repro c<N> | --cited <ids>".into()),
        }
    })();
    let (exit, said) = answer.unwrap_or_else(|error| (2, error));
    Some(crate::board::TestRun { exit_code: Some(i64::from(exit)), stdout: said, stderr: String::new(), duration_ms: started.elapsed().as_millis(), timed_out: false })
}

/// `r` on a claim: run it now, ahead of the queue, as its own batch.
pub fn run_now(store: &AuditStore, name: &str, claim: Option<&str>) -> Reply {
    let audit = store.audit(name).map_err(err)?.ok_or(format!("no audit {name}"))?;
    let Some(claim) = claim else { return rerun(store, name, Who::Operator) };
    if store.current_check(name, claim).map_err(err)?.is_none() && store.claim(name, claim).map_err(err)?.is_none_or(|c| c.repro != Repro::Pending) {
        return Err(format!("{claim} has no check yet"));
    }
    if crate::util::locked(&RUNNING).contains(name) {
        return Err(format!("a batch of {name} is running: {claim} runs after it"));
    }
    let _ = audit;
    crate::util::locked(&RUNNING).insert(name.to_string());
    let (name, claim) = (name.to_string(), claim.to_string());
    let _ = std::thread::Builder::new().name("audit-now".into()).spawn(move || {
        if let Some(store) = audit::store() {
            let _ = store.change_claim(&name, &claim, |c| {
                c.verdict = Verdict::Unchecked;
                Ok(())
            });
            if let Err(error) = run_batch(&store, &name, std::slice::from_ref(&claim), true) {
                crate::terminal::warn_status(&format!("[audit] {name}/{claim}: {error}"));
            }
        }
        crate::util::locked(&RUNNING).remove(&name);
    });
    Ok("[audit] running it now, as its own batch".into())
}

/// The operator took (or handed back) control of an audit's shell (Ctrl+\).
pub fn wheel(name: &str, taken: bool) {
    let Some(store) = audit::store() else { return };
    let _ = store.change(name, |a| {
        a.wheel_paused = taken;
        Ok(())
    });
    let _ = store.log(name, Who::Operator, if taken { "took control of its shell: paused" } else { "handed its shell back: resumed" }, "");
    if !taken {
        wake();
    }
}

pub fn running(name: &str) -> bool {
    crate::util::locked(&RUNNING).contains(name)
}

#[cfg(test)]
mod tests;
