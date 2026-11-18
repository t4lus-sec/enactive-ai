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
    // Tests never reach a real machine: without a fake, nothing runs.
    #[cfg(test)]
    {
        let _ = timeout;
        match fake(script) {
            Some((exit, stdout)) => Ok(crate::share::Ran { exit_code: exit.map(i64::from), stdout, stderr: String::new() }),
            None => Err("no machine in tests".into()),
        }
    }
    #[cfg(not(test))]
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
        let _ = std::thread::Builder::new().name("audit-scheduler".into()).spawn(|| {
            if let Some(store) = audit::store() {
                recover(&store);
            }
            schedule();
        });
    });
}

/// The harness just started, so nothing of a previous run's is running:
/// its workers and batches stopped with it. What they held is let go (it
/// would wait out its half-hour hold otherwise), and a set-up or harvest it
/// cut short starts again, in the audit's shell as before.
fn recover(store: &AuditStore) {
    let Ok(audits) = store.audits() else { return };
    for audit in audits {
        let mut freed = Vec::new();
        for claim in store.claims(&audit.name).unwrap_or_default() {
            if claim.held_by.is_some() || claim.running {
                let _ = store.change_claim(&audit.name, &claim.id, |c| {
                    c.held_by = None;
                    c.held_until = None;
                    c.running = false;
                    Ok(())
                });
                freed.push(claim.id);
            }
        }
        if !freed.is_empty() {
            let _ = store.log(&audit.name, Who::Harness, &format!("the harness restarted: let go of {}, whose workers stopped with it", freed.join(", ")), "");
        }
        if audit.harvest_open && matches!(audit.status, Status::Harvesting | Status::Checking) && !audit.paused {
            let _ = store.log(&audit.name, Who::Harness, "the harness restarted before its harvest ended: setting up and harvesting again", "");
            prepare_in_background(audit.name.clone());
        }
    }
}

fn schedule() {
    loop {
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
    }
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
        if audit.paused || audit.wheel_paused {
            continue;
        }
        // Workers for what has no check yet, or a failure with no repro.
        queue_workers(store, &audit);
        nudge_slow(store, &audit);
        if crate::util::locked(&RUNNING).contains(&audit.name) {
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
        // A worker is (re)writing its check: the old one doesn't run meanwhile.
        let held = claim.held_by.is_some() && !audit::expired(claim.held_until.as_deref(), chrono::Local::now());
        let unrun = !held
            && claim.steer.is_none()
            && !claim.running
            && store.current_checks(&audit.name, &claim.id).unwrap_or_default().iter().any(|check| check.verdict == Verdict::Unchecked);
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

/// The queue empty, nothing held, running or waiting for a worker (a failure
/// waits for its repro): the audit is done, and its
/// lead closes. New claims open it again.
fn maybe_done(store: &AuditStore, audit: &Audit) {
    if audit.status != Status::Checking || audit.gone_at.is_none() {
        return;
    }
    let Ok(claims) = store.claims(&audit.name) else { return };
    // A failure waits for its repro worker (or a reason it can't have one):
    // the worker may not have started yet, all of the audit's being busy.
    let open = claims.iter().any(|c| {
        let needs_repro = c.verdict == Verdict::Fails && c.repro == Repro::None && c.model_only_why.is_none();
        let unrun = store.current_checks(&audit.name, &c.id).unwrap_or_default().iter().any(|k| k.verdict == Verdict::Unchecked);
        c.checkable(audit) && !c.taken && (c.verdict == Verdict::Unchecked || c.held_by.is_some() || c.running || c.repro == Repro::Pending || needs_repro || unrun || c.steer.is_some())
    });
    if open || claims.is_empty() || claims.iter().any(|c| c.new) || !workers_of(&audit.name).is_empty() {
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
    // Open until its harvest worker ends: a harness stopped before then
    // starts it again (`recover`).
    store
        .change(name, |a| {
            a.harvest_open = true;
            Ok(())
        })
        .map_err(err)??;
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
    // The ceiling preview, the slice map and the harvest scan, typed into
    // its shell, where the operator sees them run.
    let mut set_up = audit.clone();
    set_up.work = work.clone();
    let shown = |step: &'static str, why: &'static str| {
        let set_up = set_up.clone();
        move |script: &str, timeout: u64| shown_in_shell(&set_up, step, why, script, timeout)
    };
    let languages = crate::audit::adapters::preview_with(&work, &audit.scope, &mut shown("languages", "its languages, and how far each can be checked"))?;
    store
        .change(name, |a| {
            a.languages = languages.clone();
            Ok(())
        })
        .map_err(err)??;
    if !audit.scope.is_empty() {
        let slice = crate::audit::scope::map_with(&work, &audit.scope, &mut shown("slice", "the slice: what it calls out to, and what calls in"))?;
        store
            .change(name, |a| {
                a.slice = Some(slice.clone());
                Ok(())
            })
            .map_err(err)??;
    }
    let sources = harvest_scan(&work, &audit.scope, &mut shown("harvest", "what the code says it does: claim words, asserts, tests, fixes"))?;
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
    // The claims themselves: a worker reads what the scan found and the code
    // around it, and proposes them, in a tab of its own.
    let fresh = store.audit(name).map_err(err)?.ok_or("gone")?;
    let proposed: Vec<Claim> = store.claims(name).map_err(err)?.into_iter().filter(|c| c.author != "operator").collect();
    want(format!("harvest:{name}"), Job::Harvest { audit: name.to_string() }, harvest_task(&fresh, &proposed));
    Ok(())
}

// ------------------------------------------------------------------ workers the harness starts

/// What a worker the harness starts is for.
#[derive(Clone, Debug)]
enum Job {
    /// Reads the scan and the code, and proposes the audit's claims.
    Harvest { audit: String },
    /// Reads a check's files, not its claim, and says what it guarantees.
    BackTranslate { audit: String, claim: String, version: i64, expect: Vec<(String, String)>, partial: Vec<String> },
    /// Writes one of a claim's checks (`k1`, or another beside it) and
    /// submits it.
    Formalize { audit: String, claim: String, slot: String },
    /// Turns a failing check's counterexample into a test that fails on the
    /// real code, or says why it can't.
    Repro { audit: String, claim: String },
}

impl Job {
    fn audit(&self) -> &str {
        match self {
            Job::Harvest { audit } | Job::BackTranslate { audit, .. } | Job::Formalize { audit, .. } | Job::Repro { audit, .. } => audit,
        }
    }

    /// A claim's own worker (formalize, repro), which counts toward the
    /// audit's `workers`.
    fn claim(&self) -> Option<&str> {
        match self {
            Job::Formalize { claim, .. } | Job::Repro { claim, .. } => Some(claim),
            _ => None,
        }
    }
}

/// Checks run as soon as they are submitted, for claims the operator asked
/// to run now before they had one.
static RUN_ON_SUBMIT: Mutex<BTreeSet<(String, String)>> = Mutex::new(BTreeSet::new());

/// The last refusal of a check for each claim, (audit, claim, why): when its
/// worker ends with nothing in, the claim is blocked for that reason rather
/// than skipped as if nothing had been tried.
static REFUSED: Mutex<Vec<(String, String, String)>> = Mutex::new(Vec::new());

fn refused(audit: &str, claim: &str) -> Option<String> {
    crate::util::locked(&REFUSED).iter().rev().find(|(a, c, _)| a == audit && c == claim).map(|(_, _, why)| why.clone())
}

/// The worker job writing `claim`'s `slot` under `label` now, by its full id
/// (labels repeat across restarts; ids don't).
fn writer_job(audit: &str, claim: &str, label: &str) -> Option<String> {
    crate::util::locked(&STARTED).iter().find_map(|(id, job)| match job {
        Job::Formalize { audit: a, claim: c, .. } | Job::Repro { audit: a, claim: c }
            if a == audit && c == claim && crate::vm::netboxes::agent_label(id) == label =>
        {
            Some(id.clone())
        }
        _ => None,
    })
}

/// An adapter passed its doctor run: claims waiting on it are tried again.
pub fn unblock(store: &AuditStore, lang: &str) {
    for audit in store.audits().unwrap_or_default() {
        for claim in store.claims(&audit.name).unwrap_or_default() {
            if claim.blocked_on.as_deref() == Some(lang) {
                let _ = store.change_claim(&audit.name, &claim.id, |c| {
                    c.blocked = None;
                    c.blocked_on = None;
                    Ok(())
                });
                let _ = store.log(&audit.name, Who::Harness, &format!("{}: the {lang} adapter passed its doctor run, so it is tried again", claim.id), "");
            }
        }
    }
    wake();
}

/// Every worker of `audit` wanted, asked for or running, by key.
fn workers_of(audit: &str) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    keys.extend(crate::util::locked(&WANTED).iter().filter(|(_, job, _)| job.audit() == audit && job.claim().is_some()).map(|(key, _, _)| key.clone()));
    keys.extend(crate::util::locked(&ASKED).iter().filter(|(_, job)| job.audit() == audit && job.claim().is_some()).map(|(key, _)| key.clone()));
    keys.extend(crate::util::locked(&STARTED).iter().filter(|(_, job)| job.audit() == audit && job.claim().is_some()).map(|(_, job)| job_key(job)));
    keys
}

fn job_key(job: &Job) -> String {
    match job {
        Job::Harvest { audit } => format!("harvest:{audit}"),
        Job::BackTranslate { audit, claim, version, .. } => format!("back:{audit}/{claim}/{version}"),
        Job::Formalize { audit, claim, slot } => format!("formalize:{audit}/{claim}/{slot}"),
        Job::Repro { audit, claim } => format!("repro:{audit}/{claim}"),
    }
}

/// After go: a worker, in a tab of its own, for each claim in plan order
/// that may be checked and has no check (nor a worker, nor the operator's
/// hand on it), and for each failure with no repro yet; at most the audit's
/// `workers` at once.
fn queue_workers(store: &AuditStore, audit: &Audit) {
    if audit.status != Status::Checking || audit.gone_at.is_none() {
        return;
    }
    let Ok(mut claims) = store.claims(&audit.name) else { return };
    claims.sort_by_key(plan_key);
    let busy = workers_of(&audit.name);
    let mut room = audit.settings.workers.saturating_sub(busy.len());
    let now = chrono::Local::now();
    for claim in claims {
        if room == 0 {
            break;
        }
        let held = claim.held_by.is_some() && !audit::expired(claim.held_until.as_deref(), now);
        if claim.taken || held || !claim.checkable(audit) {
            continue;
        }
        let checks = store.current_checks(&audit.name, &claim.id).unwrap_or_default();
        // A direction (sent back, reworded, a deeper check asked for) has the
        // check it names written, again or anew, whatever the verdict now.
        let job = if claim.steer.is_some() {
            let slot = claim.steer_check.clone().unwrap_or_else(|| audit::lead(&checks).map_or_else(|| "k1".to_string(), |c| c.slot().to_string()));
            Job::Formalize { audit: audit.name.clone(), claim: claim.id.clone(), slot }
        } else if checks.is_empty() && claim.verdict == Verdict::Unchecked {
            Job::Formalize { audit: audit.name.clone(), claim: claim.id.clone(), slot: "k1".into() }
        } else if claim.verdict == Verdict::Fails && claim.repro == Repro::None && claim.model_only_why.is_none() {
            Job::Repro { audit: audit.name.clone(), claim: claim.id.clone() }
        } else {
            continue;
        };
        if busy.contains(&job_key(&job)) {
            continue;
        }
        start_claim_worker(store, audit, &claim, job);
        room -= 1;
    }
}

/// A check-writing worker that has not submitted after `SUBMIT_BY` is told,
/// once, to submit what it has: the harness's refusal says what to fix, and
/// a check filed early lets the claim's run start.
fn nudge_slow(store: &AuditStore, audit: &Audit) {
    let due: Vec<String> = crate::util::locked(&WRITING).iter().filter(|(_, at, told)| !told && at.elapsed() >= SUBMIT_BY).map(|(id, _, _)| id.clone()).collect();
    for id in due {
        let Some((_, Job::Formalize { audit: name, claim, slot })) = crate::util::locked(&STARTED).iter().find(|(j, _)| *j == id).cloned() else { continue };
        if name != audit.name {
            continue;
        }
        if let Some((_, _, told)) = crate::util::locked(&WRITING).iter_mut().find(|(writer, _, _)| *writer == id) {
            *told = true;
        }
        // Still holding the claim: nothing submitted yet (a submission lets go).
        let label = crate::vm::netboxes::agent_label(&id);
        if store.claim(&name, &claim).ok().flatten().is_none_or(|c| c.held_by.as_deref() != Some(label.as_str())) {
            continue;
        }
        let text = format!(
            "{} minutes and no check yet: submit what you have now with audit_check_submit {{audit: \"{name}\", claim: \"{claim}\", check: \"{slot}\", dir: \"{dir}\"}}, \
             even if it is rough. The harness's refusal says what to fix.",
            SUBMIT_BY.as_secs() / 60,
            dir = audit::check_dir(&claim, &slot, &store.checks(&name, &claim).unwrap_or_default()),
        );
        if crate::subagent::steer_from_harness(&id, &text).is_ok() {
            let _ = store.log(&name, Who::Harness, &format!("told {label} to submit {claim}'s check"), "");
        }
    }
}

/// A worker for one claim: the claim held for it, so nothing else takes it.
fn start_claim_worker(store: &AuditStore, audit: &Audit, claim: &Claim, job: Job) {
    let task = match &job {
        Job::Formalize { slot, .. } => formalize_task(store, audit, claim, slot),
        _ => repro_task(store, audit, claim),
    };
    let until = (chrono::Local::now() + chrono::Duration::seconds(audit::HOLD_SECS)).to_rfc3339();
    let _ = store.change_claim(&audit.name, &claim.id, |c| {
        c.held_by = Some("a worker, starting".into());
        c.held_until = Some(until.clone());
        // A direction is in the task now.
        c.steer = None;
        c.steer_check = None;
        Ok(())
    });
    let what = match &job {
        Job::Formalize { slot, .. } => format!("to write its check {slot}"),
        _ => "to reproduce its failure".into(),
    };
    let _ = store.log(&audit.name, Who::Harness, &format!("asked for a worker for {} {what}", claim.id), "");
    want(job_key(&job), job, task);
}

/// The formalizing worker's task: the claim, where the code is, and what a
/// check must look like for the harness to take it.
fn formalize_task(store: &AuditStore, audit: &Audit, claim: &Claim, slot: &str) -> String {
    let kinds = kinds_now(store, audit);
    let all = store.checks(&audit.name, &claim.id).unwrap_or_default();
    let dir = audit::check_dir(&claim.id, slot, &all);
    // Its other checks: this one is beside them, not instead.
    let others: Vec<String> = store
        .current_checks(&audit.name, &claim.id)
        .unwrap_or_default()
        .iter()
        .filter(|c| c.slot() != slot)
        .map(|c| format!("  {} {} ({}): {}", c.slot(), c.kind, c.verdict.label(), c.dir))
        .collect();
    let others = if others.is_empty() { String::new() } else { format!("Its other checks, which stay as they are:\n{}\n", others.join("\n")) };
    let clauses = audit::clauses(&claim.text);
    let clause_lines: Vec<String> = clauses.iter().enumerate().map(|(i, c)| format!("  [{}] {c}", i + 1)).collect();
    let assumptions: Vec<String> = claim
        .rests_on
        .iter()
        .chain(store.claims(&audit.name).unwrap_or_default().iter().filter(|c| c.is_assumption() && c.review == Review::Accepted).map(|c| &c.id))
        .filter_map(|id| store.claim(&audit.name, id).ok().flatten().map(|c| format!("  {}: {}", c.id, c.text)))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let rewriting = store.current_checks(&audit.name, &claim.id).unwrap_or_default().into_iter().find(|c| c.slot() == slot);
    let depth = match &rewriting {
        Some(old) => format!("as a {} check again, replacing the one there (unless the direction says otherwise)", old.kind),
        None => depth_line(claim, audit),
    };
    let code = code_for(audit, claim);
    let reading = claim.picked.and_then(|i| claim.readings.get(i)).map(|r| format!("\nThe operator's reading: {r}")).unwrap_or_default();
    let steer = claim.steer.as_deref().map(|s| format!("\nA direction from the operator: {s}")).unwrap_or_default();
    format!(
        "Write the check {slot} for claim {id} of the audit {name}, and submit it.\n\n\
         Claim {id}: {text}\nSource: {source}\nWhy it matters: {why}{reading}{steer}\n\
         Its clauses:\n{clauses}\n{assumes}{others}\n\
         The code is in the work tree {work} on this machine ({languages}).{code}\n\
         Write the check in {work}/{dir}/ {depth}:\n\
         - check.toml: claim = \"{id}\", lang, kind (test, property, bounded, tla, differential, command or proof), run (the \
         command, run from the work tree's root), timeout (seconds), targets ([\"path:function\", ...], what the claim is about).{kinds}\n\
         - The check's own files. Mark the lines that state each clause with a comment carrying its number (// [1], # [2], \\* [3] \
         in TLA+): every clause needs one, or the harness refuses the check.\n\
         - mutants/*.patch: 5 to 10 unified diffs (paths from the work tree's root) that break the targets the way a real bug \
         would (flip a comparison, drop a guard, return early, off by one). A check that catches none counts for nothing.\n\
         - Its exit code is its verdict: 0 when the claim holds, 1 when it fails, anything else an error.\n\
         Be quick: write it, run it once in your shell, and submit it with audit_check_submit {{audit: \"{name}\", claim: \"{id}\", \
         check: \"{slot}\", dir: \"{dir}\"}} within your first few steps, without polishing it first. Read more of the code only if what is \
         above is not enough. The harness refuses what it can't take and says why: fix that and submit again. You never say \
         whether it holds: the harness runs it, with the mutants, and decides. End with a line on what the check covers and \
         what it leaves out.",
        id = claim.id,
        name = audit.name,
        text = claim.text,
        source = claim.source,
        why = claim.why,
        clauses = clause_lines.join("\n"),
        assumes = if assumptions.is_empty() {
            String::new()
        } else {
            format!("Assumptions it may rely on (stub or mock them, never check them):\n{}\n", assumptions.join("\n"))
        },
        work = audit.work,
        languages = if audit.languages.is_empty() { "languages unknown".to_string() } else { audit.languages.join(", ") },
    )
}

/// How deep the check goes. Unless the operator set a depth (`d`, or the
/// audit's `depth`), a test or property test first: quick to write and run.
/// A deeper one comes when he asks, and must then beat this one's evidence.
fn depth_line(claim: &Claim, audit: &Audit) -> String {
    match claim.depth.or(audit.settings.depth) {
        None => "as a plain test or a property test (kind test or property), not yet a bounded, TLA+ or proof check".into(),
        Some(level) => {
            let kind = match level {
                Level::Documented | Level::Tested | Level::Assumed => "test",
                Level::Property => "property",
                Level::Bounded => "bounded (the language's bounded checker)",
                Level::Model => "tla",
                Level::Proven => "proof",
            };
            format!("at the depth the operator set, {} (kind {kind})", level.label())
        }
    }
}

/// The code the claim is about, numbered, for the task: the definition
/// around its source line (or the start of its file), so the worker writes
/// the check from it instead of first searching for it; and how the repo's
/// tests run.
fn code_for(audit: &Audit, claim: &Claim) -> String {
    let spot = claim.source.split_whitespace().next().unwrap_or_default();
    let (path, line) = match spot.rsplit_once(':') {
        Some((path, line)) if line.chars().all(|c| c.is_ascii_digit()) && !line.is_empty() => (path, line.parse::<usize>().ok()),
        _ => (spot, None),
    };
    let mut out = String::new();
    if !path.is_empty() && !path.contains("..") && (path.contains('.') || path.contains('/')) {
        let script = format!("cd {} && [ -f {p} ] && head -c 400000 {p}", q(&audit.work), p = q(path));
        if let Ok(ran) = machine(&script, 30)
            && ran.exit_code == Some(0)
            && !ran.stdout.trim().is_empty()
        {
            let lines: Vec<&str> = ran.stdout.lines().collect();
            let (from, to) = around(&lines, line);
            out.push_str(&format!("\n\n{path}, lines {}-{}:\n", from + 1, to));
            for (n, text) in lines[from..to].iter().enumerate() {
                out.push_str(&format!("{:>5} | {text}\n", from + n + 1));
            }
        }
    }
    let lang = audit.languages.first().and_then(|l| l.split_whitespace().next()).unwrap_or_default();
    if let Some(test) = crate::audit::adapters::adapter(lang).and_then(|text| audit::toml_flat(&text).ok()).and_then(|t| t.get("test").map(|v| v.text()))
        && !test.is_empty()
    {
        out.push_str(&format!("\nThis repo's {lang} tests run with: {test}"));
    }
    out
}

/// The lines to show (from, to: a 0-based range): the definition holding
/// `line`, up to the next one at its indent, at most 150 lines; without a
/// line, the file's first 120.
fn around(lines: &[&str], line: Option<usize>) -> (usize, usize) {
    const MOST: usize = 150;
    let Some(line) = line.filter(|l| *l >= 1 && *l <= lines.len()) else { return (0, lines.len().min(120)) };
    let at = line - 1;
    let indent = |l: &str| l.len() - l.trim_start().len();
    let defines = |l: &str| {
        let words: Vec<&str> = l.split_whitespace().collect();
        let first = words
            .iter()
            .find(|w| !matches!(**w, "pub" | "pub(crate)" | "pub(super)" | "async" | "export" | "static" | "public" | "private" | "protected" | "unsafe" | "const" | "default" | "extern" | "inline"))
            .copied()
            .unwrap_or("");
        matches!(first, "fn" | "def" | "func" | "function" | "class" | "impl" | "struct" | "enum" | "trait" | "interface")
    };
    let start = (at.saturating_sub(200)..=at).rev().find(|&i| defines(lines[i])).unwrap_or(at.saturating_sub(20));
    let depth = indent(lines[start]);
    let end = (at + 1..lines.len())
        .find(|&i| i > start && defines(lines[i]) && indent(lines[i]) <= depth)
        .unwrap_or(lines.len());
    let end = (start + 1..end).rev().find(|&i| !lines[i - 1].trim().is_empty()).unwrap_or(end).max(at + 1);
    if end - start <= MOST {
        (start, end)
    } else {
        let from = at.saturating_sub(MOST / 3).max(start);
        (from, (from + MOST).min(end))
    }
}

/// The repro worker's task: the failing check and its counterexample.
fn repro_task(store: &AuditStore, audit: &Audit, claim: &Claim) -> String {
    let check = store.current_check(&audit.name, &claim.id).ok().flatten().unwrap_or_default();
    let trace = store
        .evidence(&audit.name, Some(&claim.id))
        .unwrap_or_default()
        .into_iter()
        .rev()
        .find(|e| e.kind == "trace")
        .map(|e| crate::util::clip(&e.result, 6000))
        .unwrap_or_else(|| "(no trace kept: run the check to see it fail)".into());
    format!(
        "Reproduce the failure of claim {id} of the audit {name} on the real code.\n\n\
         Claim {id}: {text}\nIts check ({kind}) fails: checks/{id}/, run with: {run}\nIts counterexample:\n{trace}\n\n\
         The work tree is {work}. Write a test in the repo's own framework that fails on this code for the claimed reason: \
         force any interleaving with hooks or barriers, and quote the counterexample in the test's comment. Then submit it with \
         audit_repro_submit {{audit: \"{name}\", claim: \"{id}\", test: \"<path>::<name>\", run: \"<the command that runs only it>\"}}. \
         The harness runs it: it is a finding only if it fails there. If no test on the real code can reproduce it, say why \
         with audit_model_only {{audit: \"{name}\", claim: \"{id}\", why}}. Never fix the code.",
        id = claim.id,
        name = audit.name,
        text = claim.text,
        kind = check.kind,
        run = check.run,
        work = audit.work,
    )
}

/// Workers wanted (key, what for, task), started on the main thread, where a
/// spawn can reach the archive (`app.rs`).
static WANTED: Mutex<Vec<(String, Job, String)>> = Mutex::new(Vec::new());
/// Workers started, by job id.
static STARTED: Mutex<Vec<(String, Job)>> = Mutex::new(Vec::new());
/// Check-writing workers, by job id: when each started, and whether it was
/// told to submit what it has (`nudge_slow`).
static WRITING: Mutex<Vec<(String, Instant, bool)>> = Mutex::new(Vec::new());
/// A check not submitted after this long: its worker is told to submit what
/// it has.
const SUBMIT_BY: Duration = Duration::from_secs(5 * 60);
/// Workers asked for and not yet started or refused, by key.
static ASKED: Mutex<Vec<(String, Job)>> = Mutex::new(Vec::new());

fn want(key: String, job: Job, task: String) {
    crate::util::locked(&WANTED).push((key, job, task));
    // The idle prompt comes round and starts it.
    if !cfg!(test) {
        crate::subagent::wake_prompt();
    }
}

/// Start the workers wanted, on the main thread (a spawn reaches the
/// archive): between turns, and between the agent's steps during one, so a
/// worker never waits for the agent's turn to end.
pub fn start_wanted(memory: &mut crate::hippocampus::Hippocampus, say: &mut dyn FnMut(String)) {
    for (key, task) in workers_wanted() {
        let outcome = crate::subagent::spawn_for_harness(&format!("audit-{key}"), &task, memory, say);
        worker_started(&key, outcome);
    }
}

/// The checks being written for `audit`'s claims: (claim, check, the
/// worker's label once it started).
pub fn writing(audit: &str) -> Vec<(String, String, Option<String>)> {
    let mut out = Vec::new();
    for (_, job, _) in crate::util::locked(&WANTED).iter() {
        if let Job::Formalize { audit: a, claim, slot } = job
            && a == audit
        {
            out.push((claim.clone(), slot.clone(), None));
        }
    }
    for (_, job) in crate::util::locked(&ASKED).iter() {
        if let Job::Formalize { audit: a, claim, slot } = job
            && a == audit
        {
            out.push((claim.clone(), slot.clone(), None));
        }
    }
    for (id, job) in crate::util::locked(&STARTED).iter() {
        if let Job::Formalize { audit: a, claim, slot } = job
            && a == audit
        {
            out.push((claim.clone(), slot.clone(), Some(crate::vm::netboxes::agent_label(id))));
        }
    }
    out
}

/// Workers wait to be started.
pub fn any_wanted() -> bool {
    !crate::util::locked(&WANTED).is_empty()
}

/// The workers to start now, (key, task), each once.
fn workers_wanted() -> Vec<(String, String)> {
    let wanted = std::mem::take(&mut *crate::util::locked(&WANTED));
    let mut asked = crate::util::locked(&ASKED);
    wanted
        .into_iter()
        .map(|(key, job, task)| {
            asked.push((key.clone(), job));
            (key, task)
        })
        .collect()
}

/// A wanted worker started (its job id) or could not.
pub fn worker_started(key: &str, outcome: std::result::Result<String, String>) {
    let job = {
        let mut asked = crate::util::locked(&ASKED);
        let Some(at) = asked.iter().position(|(k, _)| k == key) else { return };
        asked.remove(at).1
    };
    let Some(store) = audit::store() else { return };
    match (outcome, job) {
        (Ok(id), job) => {
            let (name, what) = match &job {
                Job::Harvest { audit } => (audit.clone(), "harvest".to_string()),
                Job::BackTranslate { audit, claim, .. } => (audit.clone(), format!("back-translation of {claim}")),
                Job::Formalize { audit, claim, slot } => (audit.clone(), format!("{claim}'s check {slot}")),
                Job::Repro { audit, claim } => (audit.clone(), format!("{claim}'s repro")),
            };
            // The claim is the worker's now: its label is how the tab
            // steers it, and how its tools are told apart.
            if let Some(claim) = job.claim() {
                let label = crate::vm::netboxes::agent_label(&id);
                let _ = store.change_claim(&name, claim, |c| {
                    c.held_by = Some(label.clone());
                    Ok(())
                });
            }
            let _ = store.log(&name, Who::Harness, &format!("started worker {id}: {what}"), "");
            if let Job::Formalize { audit, claim, .. } = &job {
                crate::util::locked(&WRITING).push((id.clone(), Instant::now(), false));
                crate::util::locked(&REFUSED).retain(|(a, c, _)| !(a == audit && c == claim));
            }
            crate::util::locked(&STARTED).push((id, job));
        }
        (Err(error), Job::Harvest { audit: name }) => {
            // No worker: the agent proposes the claims itself.
            let _ = store.change(&name, |a| {
                a.harvest_open = false;
                Ok(())
            });
            let _ = store.log(&name, Who::Harness, "the harvest worker could not start", &error);
            crate::audit::ops::needs_s0ma(format!(
                "· {name}: set up, but its harvest worker could not start ({error}). audit_read {{audit: \"{name}\", what: \"sources\"}} shows what the scan found; propose claims with audit_claims_propose"
            ));
        }
        (Err(error), Job::BackTranslate { audit: name, claim, .. }) => {
            let _ = store.log(&name, Who::Harness, &format!("the back-translation of {claim} could not start"), &error);
        }
        (Err(error), Job::Formalize { audit: name, claim, .. } | Job::Repro { audit: name, claim }) => {
            // Let go, and paused: a worker that can't start now won't on the
            // next pass either; the operator sees why and resumes it (z).
            let _ = store.change_claim(&name, &claim, |c| {
                c.held_by = None;
                c.held_until = None;
                Ok(())
            });
            let _ = store.change(&name, |a| {
                a.paused = true;
                a.problem = Some(format!("a worker for {claim} could not start: {error} · z resumes it"));
                Ok(())
            });
            let _ = store.log(&name, Who::Harness, &format!("a worker for {claim} could not start: paused"), &error);
        }
    }
}

/// A subagent finished: if it was one of the audits' workers, its result is
/// taken here, and a line for the agent is given back. None: not ours.
pub fn worker_finished(id: &str, status: &str, result: &str, error: Option<&str>) -> Option<String> {
    let job = {
        let mut started = crate::util::locked(&STARTED);
        let at = started.iter().position(|(j, _)| j == id)?;
        started.remove(at).1
    };
    crate::util::locked(&WRITING).retain(|(writer, _, _)| writer != id);
    let store = audit::store()?;
    let failed = status != "completed";
    let why = error.unwrap_or(status);
    let formal_slot = if let Job::Formalize { slot, .. } = &job { Some(slot.clone()) } else { None };
    match job {
        Job::Harvest { audit: name } => {
            let _ = store.change(&name, |a| {
                a.harvest_open = false;
                Ok(())
            });
            let proposed = store.claims(&name).unwrap_or_default().iter().filter(|c| c.review == Review::Proposed).count();
            let _ = store.log(&name, Who::Harness, &format!("the harvest worker {id} finished: {status}"), &crate::util::clip(result, 400));
            write_share(&store, &name);
            wake();
            Some(if failed {
                format!("[audit] {name}: the harvest worker {id} stopped ({why}) with {proposed} claims proposed; propose more with audit_claims_propose")
            } else {
                format!("[audit] {name}: harvested, {proposed} claims proposed for the operator's review. The worker's summary: {}", crate::util::clip(result.trim(), 600))
            })
        }
        Job::Formalize { audit: name, claim, .. } | Job::Repro { audit: name, claim } => {
            let label = crate::vm::netboxes::agent_label(id);
            let checks = store.checks(&name, &claim).unwrap_or_default();
            // This worker's own submission, or (a repro worker) any check.
            let submitted = match &formal_slot {
                Some(slot) => checks.iter().any(|c| c.slot() == slot && c.by == label),
                None => !checks.is_empty(),
            };
            if let Some(slot) = formal_slot.filter(|_| !submitted) {
                // Nothing submitted: it may have left a check in its folder,
                // or had one refused. Found out off this thread (a shell
                // command); the claim stays held meanwhile.
                let until = (chrono::Local::now() + chrono::Duration::seconds(300)).to_rfc3339();
                let _ = store.change_claim(&name, &claim, |c| {
                    c.held_by = Some(label.clone());
                    c.held_until = Some(until);
                    Ok(())
                });
                let line = format!("{claim}'s worker {id} ended ({why}) with nothing submitted: the harness looks for a check in its folder");
                let _ = store.log(&name, Who::Harness, &format!("worker {id} finished: {line}"), &crate::util::clip(result, 400));
                let (id, why) = (id.to_string(), why.to_string());
                let settle = move || settle_unsubmitted(&name, &claim, &slot, &id, &why);
                if cfg!(test) {
                    settle();
                } else {
                    let _ = std::thread::Builder::new().name("audit-settle".into()).spawn(settle);
                }
                return Some(format!("[audit] {line}"));
            }
            let others = checks.iter().any(|c| c.by != label);
            // Let go and, with nothing submitted, skipped, in one write: the
            // scheduler never sees it free and unskipped between the two and
            // starts another worker on it.
            let line = store
                .change_claim(&name, &claim, |c| {
                    if c.held_by.as_deref() == Some(&crate::vm::netboxes::agent_label(id)) {
                        c.held_by = None;
                        c.held_until = None;
                    }
                    Ok(if c.repro != Repro::None || c.model_only_why.is_some() {
                        format!("{claim}'s repro is in ({:?})", c.repro)
                    } else if submitted {
                        format!("{claim}'s check is in; the harness runs it")
                    } else if others {
                        // Its other checks stand; this one is not asked again.
                        format!("{claim}'s worker {id} ended ({why}) with nothing submitted; its other checks stand")
                    } else {
                        // Nothing submitted: not tried again on its own.
                        c.skipped = true;
                        format!("{claim}'s worker {id} ended ({why}) with nothing submitted: skipped until the operator un-skips it (s in the plan)")
                    })
                })
                .ok()
                .and_then(Result::ok)
                .unwrap_or_else(|| format!("{claim}'s worker {id} ended ({why})"));
            let _ = store.log(&name, Who::Harness, &format!("worker {id} finished: {line}"), &crate::util::clip(result, 400));
            wake();
            Some(format!("[audit] {name}: {line}"))
        }
        Job::BackTranslate { audit: name, claim, version, expect, partial } => {
            if !failed
                && let Ok(checks) = store.checks(&name, &claim)
                && let Some(mut check) = checks.into_iter().find(|c| c.version == version)
            {
                check.back_translation = Some(review::settle(result.trim(), &expect, &partial));
                check.back_by = id.to_string();
                let _ = store.put_check(&name, &check);
            }
            let _ = store.log(&name, Who::Harness, &format!("back-translation of {claim}: {status}"), "");
            Some(format!("[audit] {name}/{claim}: its back-translation is in ({status}); the operator reads it beside the claim"))
        }
    }
}

/// Which kinds the harness takes now, for the audit's languages: an
/// adapter's tools only once it passed its doctor run.
fn kinds_now(store: &AuditStore, audit: &Audit) -> String {
    let notes: Vec<String> = audit
        .languages
        .iter()
        .filter(|lang| crate::audit::adapters::adapter(lang).is_some())
        .map(|lang| {
            let passed = store.adapter_state(lang).ok().flatten().is_some_and(|s| s.doctor_passed) && !crate::audit::adapters::changed_since_doctor(lang);
            if passed {
                format!("{lang}: its adapter's tools are usable")
            } else {
                format!("{lang}: its adapter has not passed its doctor run, so only test, command, differential and tla are taken for it now")
            }
        })
        .collect();
    if notes.is_empty() { String::new() } else { format!(" Kinds taken now: {}.", notes.join("; ")) }
}

/// A check worker ended with nothing submitted. A `check.toml` it left in
/// its folder is submitted for it; else the claim is blocked by the last
/// refusal (an adapter that hasn't passed its doctor run, say), or skipped
/// when there was none, or left to its other checks. Each with why.
fn settle_unsubmitted(name: &str, claim: &str, slot: &str, id: &str, why: &str) {
    let Some(store) = audit::store() else { return };
    let Ok(Some(audit)) = store.audit(name) else { return };
    let label = crate::vm::netboxes::agent_label(id);
    let existing = store.checks(name, claim).unwrap_or_default();
    let mut dirs = vec![audit::check_dir(claim, slot, &existing), format!("checks/{claim}/{slot}"), format!("checks/{claim}-{slot}")];
    if slot == "k1" {
        dirs.push(format!("checks/{claim}"));
    }
    dirs.dedup();
    let probe = format!(
        "cd {} || exit 3; for d in {}; do [ -f \"$d/check.toml\" ] && echo \"$d\"; done; true",
        q(&audit.work),
        dirs.iter().map(|d| q(d)).collect::<Vec<_>>().join(" ")
    );
    let found: Vec<String> = machine(&probe, 30).map(|ran| ran.stdout.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect()).unwrap_or_default();
    let mut submitted = None;
    for dir in &found {
        match submit_check(&store, name, claim, dir, &label, Some(slot)) {
            Ok(_) => {
                // Its job is gone from the running list by now: named here.
                if let Some(mut check) = store.checks(name, claim).unwrap_or_default().into_iter().rev().find(|c| c.slot() == slot && c.by == label) {
                    check.by_job = id.to_string();
                    let _ = store.put_check(name, &check);
                }
                submitted = Some(dir.clone());
                break;
            }
            Err(error) => {
                let _ = store.log(name, Who::Harness, &format!("{claim}: {dir}/check.toml was left unsubmitted, and is refused"), &error);
            }
        }
    }
    let others = existing.iter().any(|c| c.slot() != slot || c.by != label);
    let refusal = refused(name, claim);
    let line = store
        .change_claim(name, claim, |c| {
            if c.held_by.as_deref() == Some(label.as_str()) {
                c.held_by = None;
                c.held_until = None;
            }
            Ok(if let Some(dir) = &submitted {
                format!("{claim}'s worker {id} ended ({why}) without submitting, but left {dir}/check.toml: submitted it for the worker")
            } else if let Some(reason) = &refusal {
                c.blocked = Some(crate::util::clip(reason, 300));
                c.blocked_on = crate::audit::adapters::blocked_lang(reason);
                format!(
                    "{claim}'s worker {id} ended ({why}) with its check refused: blocked ({}){}",
                    crate::util::clip(reason, 300),
                    if c.blocked_on.is_some() { "; tried again once the adapter passes" } else { "; s in the plan unblocks it" }
                )
            } else if others {
                format!("{claim}'s worker {id} ended ({why}) with nothing submitted for {slot}; its other checks stand")
            } else {
                c.skipped = true;
                let looked = if found.is_empty() { format!("no check.toml in {}", dirs.join(", ")) } else { format!("{} refused", found.join(", ")) };
                format!("{claim}'s worker {id} ended ({why}) with nothing submitted ({looked}): skipped until the operator un-skips it (s in the plan)")
            })
        })
        .ok()
        .and_then(Result::ok)
        .unwrap_or_else(|| format!("{claim}'s worker {id} ended ({why})"));
    let _ = store.log(name, Who::Harness, &line, "");
    crate::audit::ops::note(format!("[audit] {name}: {line}"));
    write_share(&store, name);
    wake();
}

/// The harvest worker's task (`proposed`: the claims already there, from a
/// harvest the harness's stop cut short).
fn harvest_task(audit: &Audit, proposed: &[Claim]) -> String {
    let scope = if audit.scope.is_empty() { "the whole tree".to_string() } else { audit.scope.join(", ") };
    let already = if proposed.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nAn earlier harvest stopped part way, after proposing these; don't propose them again:\n{}",
            proposed.iter().map(|c| format!("  {}: {}", c.id, c.text)).collect::<Vec<_>>().join("\n")
        )
    };
    let task = format!(
        "Harvest the audit {name}: find what its code claims to do, for the harness to check.\n\n\
         Its work tree is {work} on this machine; the scope is {scope}. Start with audit_read {{audit: \"{name}\", what: \"sources\"}}: \
         the harness's scan (comments with claim words, asserts, test names, commits that say fix). Then read the code in scope, and \
         any docs about it, with commands in your shell.\n\n\
         Propose claims with audit_claims_propose {{audit: \"{name}\", claims: [{{text, source, kind, risk, why}}]}}, in batches:\n\
         - text: one plain statement the code should make true (\"a download never replaces a file changed meanwhile\");\n\
         - source: where it comes from: path:line, a doc and its quote, or a commit;\n\
         - kind: pure, invariant, protocol, equivalence or resource; or assumes, for what the scope relies on from a call out of it;\n\
         - risk: 1 to 10, the cost if it is false; why: why it is worth checking.\n\
         Two sources that disagree are a claim too (kind as fits, why says they conflict). Propose only what you can point at.\n\n\
         You never check claims or say whether they hold: the harness does, once the operator reviews them and says go. \
         End with a few lines: what you proposed, and what you looked at and left out.",
        name = audit.name,
        work = audit.work,
    );
    task + &already
}

/// A set-up step typed into the audit's shell, where the operator sees it
/// run as one short command (`bash ../setup/harvest.sh`): the script is
/// written beside the work tree first, prints there what it found, and keeps
/// its stdout and stderr apart for the harness to read back, as
/// `netboxes::machine` would give them.
fn shown_in_shell(audit: &Audit, step: &str, why: &str, script: &str, timeout: u64) -> std::result::Result<crate::share::Ran, String> {
    let dir = format!("{}/../setup", audit.work.trim_end_matches('/'));
    let wrapped = format!("{{\n{script}\n}} > ../setup/{step}.out 2> ../setup/{step}.err\ns=$?\ncat ../setup/{step}.out\nexit $s\n");
    machine(
        &format!("mkdir -p {d} && printf %s '{}' | base64 -d > {d}/{step}.sh && chmod -R a+rwX {d}", STANDARD.encode(wrapped.as_bytes()), d = q(&dir)),
        30,
    )?;
    let ran = in_shell(audit, step, why, &format!("bash ../setup/{step}.sh"), timeout)?;
    say(&ran, &format!("{step}: exit {}", ran.exit.map_or("-".into(), |e| e.to_string())));
    let read = machine(&format!("cd {d} && cat {step}.out; printf '\\n@@@S0MA-ERR\\n'; cat {step}.err", d = q(&dir)), 30)?;
    let (stdout, stderr) = read.stdout.split_once("\n@@@S0MA-ERR\n").unwrap_or((read.stdout.as_str(), ""));
    Ok(crate::share::Ran { exit_code: ran.exit.map(i64::from), stdout: stdout.to_string(), stderr: stderr.to_string() })
}

/// Whether a batch of `audit`'s is running a command now.
pub fn batch_running(audit: &str) -> bool {
    crate::util::locked(&RUNNING).contains(audit)
}

/// Every worker of `audit`'s, wanted, asked for or running, stopped; the
/// running ones' job ids.
pub fn stop_workers(audit: &str) -> Vec<String> {
    crate::util::locked(&WANTED).retain(|(_, job, _)| job.audit() != audit);
    crate::util::locked(&ASKED).retain(|(_, job)| job.audit() != audit);
    let ids: Vec<String> = crate::util::locked(&STARTED).iter().filter(|(_, job)| job.audit() == audit).map(|(id, _)| id.clone()).collect();
    crate::util::locked(&WRITING).retain(|(id, _, _)| !ids.contains(id));
    if !cfg!(test) {
        for id in &ids {
            let _ = crate::subagent::kill(id);
        }
    }
    ids
}

/// Why an audit's shell is not closed under it.
pub const SHELL_RUNNING: &str = "the audit is running a check in it (z in the audit tab pauses the audit)";

/// Whether `shell` (a netbox's name) is the shell of an audit that is on a
/// batch of checks now.
pub fn shell_running(shell: &str) -> bool {
    crate::util::locked(&RUNNING).iter().any(|audit| crate::audit::shell_name(audit) == shell)
}

/// Its shell closed, away from the keys (it opens again when a command
/// needs it).
pub fn close_shell(audit: &Audit) {
    close_shell_named(audit.shell());
}

/// The same, by the shell's name: an audit archived, trashed or gone from
/// the projects folder.
pub fn close_shell_named(name: String) {
    if cfg!(test) {
        return;
    }
    let _ = std::thread::Builder::new().name("audit-shell-close".into()).spawn(move || {
        let _ = crate::vm::netboxes::close(&mut crate::vm::netboxes::machine, &name, true);
        crate::vm::terms::touch();
    });
}

/// An audit's files removed: its work tree (a git worktree and its branch,
/// or a copy) with its setup beside it, and its shared files. What was
/// removed, said.
pub fn remove_files(audit: &Audit) -> std::result::Result<Vec<String>, String> {
    let mut removed = Vec::new();
    let mut problems = Vec::new();
    // Only the harness's own folder: `<base>/<name>/work`.
    let suffix = format!("/{}/work", audit.name);
    if !audit.work.is_empty() && audit.work.ends_with(&suffix) {
        let folder = audit.work.trim_end_matches("/work").to_string();
        let mut script = String::new();
        if audit.kind == "git"
            && let Some(target) = audit.targets.first()
        {
            script.push_str(&format!(
                "top=$(git -C {t} rev-parse --show-toplevel 2>/dev/null) && {{ git -C \"$top\" worktree remove --force {w} 2>/dev/null; git -C \"$top\" worktree prune 2>/dev/null; {branch} }}\n",
                t = q(target),
                w = q(&audit.work),
                branch = if audit.branch.is_empty() { String::new() } else { format!("git -C \"$top\" branch -D {} >/dev/null 2>&1;", q(&audit.branch)) },
            ));
        }
        script.push_str(&format!("rm -rf -- {f} && echo REMOVED", f = q(&folder)));
        match machine(&script, 120) {
            Ok(ran) if ran.stdout.contains("REMOVED") => {
                removed.push(format!("its work tree {}", audit.work));
                if audit.kind == "git" && !audit.branch.is_empty() {
                    removed.push(format!("its branch {}", audit.branch));
                }
            }
            Ok(ran) => problems.push(format!("{folder}: {}", ran.stderr.trim())),
            Err(error) => problems.push(format!("{folder}: {error}")),
        }
    }
    let shared = std::path::Path::new(&crate::share::share_dir()).join("audits").join(&audit.name);
    if shared.exists() {
        match std::fs::remove_dir_all(&shared) {
            Ok(()) => removed.push(format!("/share/audits/{}/", audit.name)),
            Err(error) => problems.push(format!("/share/audits/{}/: {error}", audit.name)),
        }
    }
    if problems.is_empty() { Ok(removed) } else { Err(problems.join("; ")) }
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
fn harvest_scan(work: &str, scope: &[String], run: &mut dyn FnMut(&str, u64) -> std::result::Result<crate::share::Ran, String>) -> std::result::Result<Vec<String>, String> {
    let paths = crate::audit::scope::paths_for(work, scope);
    let script = format!(
        "cd {work} || exit 3\n\
         grep -rnI -E '(//|#|/\\*|\\*|--|\"|'\"'\"').*\\b(invariant|never|always|must|can.t happen|cannot happen|safe because|guarantee)' {paths} \
            --exclude-dir=.git --exclude-dir=target --exclude-dir=node_modules --exclude-dir=checks 2>/dev/null | head -150 | sed 's/^/claim-word /'\n\
         grep -rnI -E '\\b((debug_)?assert(_eq|_ne)?!|assert)\\b' {paths} --exclude-dir=.git --exclude-dir=target --exclude-dir=node_modules --exclude-dir=checks 2>/dev/null | head -80 | sed 's/^/assert /'\n\
         grep -rnI -E '(fn test_|#\\[test\\]|def test_|it\\(|test\\(|func Test)' {paths} --exclude-dir=.git --exclude-dir=target --exclude-dir=node_modules 2>/dev/null | head -60 | sed 's/^/test /'\n\
         git log -n 40 --format='%h %s' -i --grep=fix 2>/dev/null | sed 's/^/fix-commit /'",
        work = q(work),
        paths = if paths.is_empty() { ".".to_string() } else { paths.iter().map(|p| q(p)).collect::<Vec<_>>().join(" ") },
    );
    let ran = run(&script, 120)?;
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
    // Off this thread: a worker's end is taken under the subagents' lock,
    // which the tab bar needs every frame, and the machine may be slow.
    if cfg!(test) {
        let _ = machine(&script, 30);
    } else {
        let _ = std::thread::Builder::new().name("audit-share".into()).spawn(move || {
            let _ = machine(&script, 30);
        });
    }
}

// ------------------------------------------------------------------ check submission

/// A worker's `audit_check_submit`: the harness reads `check.toml`, files it
/// under `checks/c<N>/` (committed on the audit's branch), and, for a check
/// already relied on, holds a weaker one for the operator (rule 4).
/// `slot`: which of the claim's checks this is (`k2`); None reads it from
/// the directory's name (`checks/c3/k2`), else the claim's first.
pub fn submit_check(store: &AuditStore, audit_name: &str, claim_id: &str, dir: &str, by: &str, slot: Option<&str>) -> Reply {
    let outcome = submit_check_once(store, audit_name, claim_id, dir, by, slot);
    if let Err(why) = &outcome {
        let mut refusals = crate::util::locked(&REFUSED);
        refusals.retain(|(a, c, _)| !(a == audit_name && c == claim_id));
        refusals.push((audit_name.to_string(), claim_id.to_string(), why.clone()));
    }
    outcome
}

fn submit_check_once(store: &AuditStore, audit_name: &str, claim_id: &str, dir: &str, by: &str, slot: Option<&str>) -> Reply {
    let audit = store.audit(audit_name).map_err(err)?.ok_or(format!("no audit {audit_name}"))?;
    let claim = store.claim(audit_name, claim_id).map_err(err)?.ok_or(format!("no claim {claim_id}"))?;
    if claim.taken && by != "operator" {
        return Err(format!("{claim_id} is the operator's: he writes its check"));
    }
    if matches!(claim.review, Review::Rejected | Review::Dropped) {
        return Err(format!("{claim_id} was {}", if claim.review == Review::Rejected { "rejected" } else { "dropped" }));
    }
    let all = store.checks(audit_name, claim_id).map_err(err)?;
    let slot = slot
        .map(str::to_string)
        .filter(|s| !s.is_empty())
        .or_else(|| dir.trim_end_matches('/').rsplit(['/', '-']).next().filter(|last| last.len() > 1 && last.starts_with('k') && last[1..].chars().all(|c| c.is_ascii_digit())).map(str::to_string))
        .unwrap_or_else(|| "k1".into());
    if !slot.starts_with('k') || slot.len() < 2 || !slot[1..].chars().all(|c| c.is_ascii_digit()) {
        return Err(format!("check {slot:?}: k1, k2…"));
    }
    let check_dir = audit::check_dir(claim_id, &slot, &all);
    let target = format!("{}/{check_dir}", audit.work);
    let src = if dir.starts_with('/') { dir.to_string() } else { format!("{}/{dir}", audit.work) };
    let script = format!(
        "s={src}; t={target}; [ -f \"$s/check.toml\" ] || {{ echo NOTOML; exit 4; }}\n\
         if [ \"$(cd \"$s\" && pwd)\" != \"$(mkdir -p \"$t\" && cd \"$t\" && pwd)\" ]; then rm -rf \"$t\" && mkdir -p \"$t\" && cp -a \"$s/.\" \"$t/\"; fi\n\
         echo __TOML__; cat \"$t/check.toml\"; echo __ANCHORS__; grep -rhoE '(//|\\\\\\*|#) *\\[[0-9]+\\]' \"$t\" 2>/dev/null | grep -oE '[0-9]+' | sort -un | tr '\\n' ' '; echo\n\
         echo __MUTANTS__; ls \"$t/mutants\" 2>/dev/null | tr '\\n' ' '; echo\n\
         cd {work} && git add {check_dir} >/dev/null 2>&1 && git commit -q -m 'audit {audit_name}: check {slot} for {claim_id}' >/dev/null 2>&1; echo \"__COMMIT__ $(git rev-parse HEAD 2>/dev/null)\"",
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
    // A depth the operator asked for (`d`) is a floor for the check it
    // added: a shallower one doesn't answer it.
    let previous = store.current_checks(audit_name, claim_id).map_err(err)?.into_iter().find(|c| c.slot() == slot);
    if let Some(wanted) = claim.depth
        && previous.is_none()
        && Level::of_kind(&kind) < wanted
    {
        return Err(format!("the operator asked for {claim_id} at depth {}: a {kind} check is shallower", wanted.label()));
    }
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
        slot: slot.clone(),
        dir: check_dir.clone(),
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
        by_job: writer_job(audit_name, claim_id, by).unwrap_or_default(),
        commit,
        ..Check::default()
    };
    // Rule 4: relied on means accepted, or cited by a report.
    let relied = claim.review == Review::Accepted || !claim.cited.is_empty();
    let check = store.add_check(audit_name, check).map_err(err)?;
    let in_force = store.current_checks(audit_name, claim_id).map_err(err)?;
    let (verdict, level) = audit::roll_up(&in_force);
    store
        .change_claim(audit_name, claim_id, |c| {
            c.held_by = None;
            c.held_until = None;
            let mut rests: Vec<String> = in_force.iter().flat_map(|k| k.assumes.clone()).collect();
            rests.sort();
            rests.dedup();
            c.rests_on = rests;
            // A relied-on result stands until the new version proves at
            // least as strong; otherwise the claim reads from its checks.
            if !(relied && previous.is_some()) {
                c.verdict = verdict;
                c.level = level;
            }
            c.tool = audit::lead(&in_force).map(|k| k.kind.clone()).unwrap_or_else(|| kind.clone());
            Ok(())
        })
        .map_err(err)??;
    store
        .log(audit_name, Who::Harness, &format!("{by} submitted a {kind} check {slot} for {claim_id} (v{})", check.version), &check.run)
        .map_err(err)?;
    if audit.settings.back_translation {
        back_translate_in_background(audit_name.to_string(), claim_id.to_string(), check.version, check.dir.clone());
    }
    if relied && let Some(previous) = previous {
        strengthen_in_background(audit_name.to_string(), claim_id.to_string(), previous, check.version);
        return Ok(json!({"ok": true, "check": slot, "version": check.version, "note": "This claim's verdict is relied on: the harness re-runs the old evidence against the new check before it replaces the old one."}).to_string());
    }
    // The operator asked for it to run now, before it had a check.
    if crate::util::locked(&RUN_ON_SUBMIT).remove(&(audit_name.to_string(), claim_id.to_string())) {
        let _ = run_now(store, audit_name, Some(claim_id));
        return Ok(json!({"ok": true, "check": slot, "version": check.version, "note": "Filed, and running now: the operator asked for it. The verdict is the harness's."}).to_string());
    }
    wake();
    Ok(json!({"ok": true, "check": slot, "version": check.version, "note": "Filed. The harness vets and runs it when the audit's gate allows; the verdict is the harness's."}).to_string())
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
    // Each claim's checks in force that have not run: a new one beside a
    // check that holds runs alone.
    let mut work: Vec<(Claim, Vec<Check>)> = Vec::new();
    for id in ids {
        let Some(claim) = store.claim(name, id).map_err(err)? else { continue };
        let checks = store.current_checks(name, id).map_err(err)?.into_iter().filter(|c| c.verdict == Verdict::Unchecked).collect();
        work.push((claim, checks));
    }
    if work.is_empty() {
        return Ok(());
    }
    // What it will run, for the one approval.
    let mut planned = Vec::new();
    let mut estimate = 0u64;
    for (claim, checks) in &work {
        if claim.repro == Repro::Pending {
            if let Some(test) = &claim.repro_test {
                planned.push(format!("{} repro: {test}", claim.id));
                estimate += 120;
            }
        } else {
            for check in checks {
                planned.push(format!("{} {} {}: {}", claim.id, check.slot(), check.kind, check.run));
                let mutants = if audit.settings.mutants { mutant_names(&audit, check).len() as u64 } else { 0 };
                estimate += check.timeout.min(audit.settings.check_secs) / 4 * (1 + mutants);
            }
        }
    }
    let detail = planned.join("\n");
    let count = planned.len();
    let what = format!("run audit {name}: {count} check{}", if count == 1 { "" } else { "s" });
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
    store.put_run(name, &run).map_err(err)?;
    let mut log = format!(
        "# audit {name} · batch {} · {}\n# claims: {}\n# {}\n",
        run.n,
        run.started_at,
        ids.join(" "),
        run.approval
    );
    let started = Instant::now();
    let each = || -> std::result::Result<(), String> {
        for (claim, checks) in work {
            if claim.repro == Repro::Pending {
                run_repro(store, &audit, &claim, &mut run, &mut log)?;
                continue;
            }
            for check in checks {
                if store.audit(name).ok().flatten().is_some_and(|a| a.paused || a.wheel_paused || a.status == Status::Stopped) {
                    log.push_str("# paused: the rest waits\n");
                    return Ok(());
                }
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
                outcome.map_err(|e| format!("{} {}: {e}", claim.id, check.slot()))?;
                spend_step(store, &audit);
            }
        }
        Ok(())
    };
    let outcome = each();
    // Closed however it ended: when, what stopped it, the transcript.
    run.finished_at = Some(audit::now());
    if let Err(error) = &outcome {
        run.note = format!("stopped: {error}");
        log.push_str(&format!("# stopped: {error}\n"));
    }
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
    outcome.map_err(|error| format!("batch {}: {error}", run.n))
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
    let full = if audit.work.is_empty() { command.to_string() } else { format!("cd {} && {command}", q(&audit.work)) };
    #[cfg(test)]
    if let Some((exit, output)) = fake(&full) {
        if output == "__GONE__" {
            return Err("its shell is gone".into());
        }
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
            waited => {
                // No verdict from its shell long past its timeout: stopped,
                // and an error for this check alone, with what was on screen.
                if begun.elapsed().as_secs() > timeout + crate::vm::netbox_run::CONFIRM_SECS {
                    let _ = crate::vm::netbox_run::interrupt(&started, &mut crate::vm::netboxes::machine);
                    let screen = match waited {
                        crate::vm::netbox_run::Waited::Input(screen) => screen,
                        _ => String::new(),
                    };
                    return Ok(Ran {
                        exit: None,
                        output: format!("no verdict from the shell {}s after it was typed (it may be waiting for input, or its result was never written){}", begun.elapsed().as_secs(), if screen.trim().is_empty() { String::new() } else { format!("\nthe screen:\n{screen}") }),
                        ms: begun.elapsed().as_millis() as u64,
                        interrupted: false,
                        typed: Some(started),
                    });
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
    let slot = check.slot().to_string();
    let ran = in_shell(audit, &claim.id, &why, &check.run, check.timeout.min(audit.settings.check_secs))?;
    let (mut verdict, said) = verdict_of(ran.exit, ran.interrupted);
    let trace = if verdict == Verdict::Fails { Some(format!("runs/{}/{}-{slot}.trace", run.n, claim.id)) } else { None };
    say(&ran, &format!("audit {} {slot}: exit {}: {said}{}", claim.id, ran.exit.map_or("-".into(), |e| e.to_string()), trace.as_ref().map(|t| format!(" · trace {t}")).unwrap_or_default()));
    log.push_str(&format!("$ {}\n{}\n# audit {} {slot}: exit {}: {said}\n", check.run, ran.output, claim.id, ran.exit.map_or("-".into(), |e| e.to_string())));
    run.commands.push(RunCommand { claim: claim.id.clone(), what: format!("check {slot}"), command: check.run.clone(), exit: ran.exit, verdict: said.into(), ms: ran.ms, by_operator: false, output: kept(&ran.output) });
    store.put_run(&audit.name, run).map_err(err)?;
    if let Some(trace) = &trace {
        // The counterexample, as the check printed it.
        let _ = store.add_evidence(
            &audit.name,
            &Evidence { claim: claim.id.clone(), check: slot.clone(), kind: "trace".into(), name: trace.clone(), path: trace.clone(), result: crate::util::clip(&ran.output, 20_000), at: audit::now() },
        );
    }
    // Vetting: a check that holds must catch mutants.
    let mut level = Level::of_kind(&check.kind);
    let mut kills = None;
    let mut killed = Vec::new();
    let mut survived = Vec::new();
    let mut broken_mutants: Vec<String> = Vec::new();
    if verdict == Verdict::Holds {
        let mutants = if audit.settings.mutants { mutant_names(audit, check) } else { Vec::new() };
        if !mutants.is_empty() {
            let mut broken = Vec::new();
            for mutant in &mutants {
                let outcome = run_mutant(audit, claim, check, mutant, run, log)?;
                let caught = outcome == Mutated::Caught;
                match &outcome {
                    Mutated::Caught => killed.push(mutant.clone()),
                    Mutated::Survived => survived.push(mutant.clone()),
                    Mutated::Broken(why) => broken.push(format!("{mutant}: {why}")),
                }
                let _ = store.add_evidence(
                    &audit.name,
                    &Evidence {
                        claim: claim.id.clone(),
                        check: slot.clone(),
                        kind: "mutant".into(),
                        name: mutant.clone(),
                        path: format!("{}/mutants/{mutant}", check.dir),
                        result: match &outcome {
                            Mutated::Broken(why) => format!("error: {why}"),
                            _ if caught => "killed".into(),
                            _ => "survived".into(),
                        },
                        at: audit::now(),
                    },
                );
            }
            // Broken mutants count neither way: what ran is what was vetted.
            kills = Some((killed.len(), killed.len() + survived.len()));
            if killed.is_empty() {
                verdict = Verdict::Vacuous;
                level = Level::Documented;
                if survived.is_empty() {
                    log.push_str(&format!("# audit {}: no mutant could run ({}): vacuous until one can\n", claim.id, broken.join("; ")));
                }
            }
            broken_mutants = broken;
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
        run.commands.push(RunCommand { claim: claim.id.clone(), what: "coverage".into(), command, exit: cov.exit, verdict: "no verdict".into(), ms: cov.ms, by_operator: false, output: kept(&cov.output) });
    }
    let mut stored = check.clone();
    stored.kills = kills;
    stored.killed = killed;
    stored.survived = survived;
    stored.broken = broken_mutants;
    stored.verdict = verdict;
    stored.level = level;
    stored.checked_at = Some(audit::now());
    stored.trace = trace.clone();
    store.put_check(&audit.name, &stored).map_err(err)?;
    // The claim reads from all its checks in force.
    let in_force = store.current_checks(&audit.name, &claim.id).map_err(err)?;
    let (claim_verdict, claim_level) = audit::roll_up(&in_force);
    let leading = audit::lead(&in_force).cloned();
    store
        .change_claim(&audit.name, &claim.id, |c| {
            c.verdict = claim_verdict;
            c.level = claim_level;
            c.kills = leading.as_ref().and_then(|k| k.kills);
            c.checked_at = Some(audit::now());
            c.trace = in_force.iter().find(|k| k.verdict == Verdict::Fails).and_then(|k| k.trace.clone());
            c.tool = leading.as_ref().map(|k| k.kind.clone()).unwrap_or_default();
            if claim_verdict == Verdict::Holds && c.fixed.is_none() && c.repro == Repro::Fails {
                c.fixed = Some(audit.commit.clone());
            }
            Ok(())
        })
        .map_err(err)??;
    store
        .log(&audit.name, Who::Harness, &format!("{} {slot} {} ({})", claim.id, verdict.label(), level.label()), &kills.map(|(k, t)| format!("killed {k} of {t} mutants")).unwrap_or_default())
        .map_err(err)?;
    // An assumption that fails makes every claim resting on it unsound.
    if claim.is_assumption() && claim_verdict == Verdict::Fails {
        crate::audit::scope::mark_unsound(store, &audit.name, &claim.id);
    }
    if verdict == Verdict::Fails {
        crate::audit::ops::note(format!(
            "{}/{} fails ({} check {slot}): a worker can turn the counterexample into a repro (audit_repro_submit), or say why not (audit_model_only)",
            audit.name, claim.id, check.kind
        ));
    }
    crate::audit::ops::post_result(store, &audit.name, &claim.id, false).map_err(err)?;
    crate::audit::scope::discharge(store, &audit.name, &claim.id);
    Ok(())
}

/// Checks to run again: their last result set aside.
fn unrun(store: &AuditStore, audit: &str, checks: &[Check]) -> std::result::Result<(), String> {
    for check in checks {
        let mut check = check.clone();
        check.verdict = Verdict::Unchecked;
        store.put_check(audit, &check).map_err(err)?;
    }
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
fn run_mutant(audit: &Audit, claim: &Claim, check: &Check, mutant: &str, run: &mut Run, log: &mut String) -> std::result::Result<Mutated, String> {
    let patch = format!("{}/mutants/{mutant}", check.dir);
    // Applied only if it applies cleanly; reverted however the check ends.
    let command = format!(
        "command -v patch >/dev/null || {{ echo 'patch is not installed on the machine'; exit 97; }}; \
         patch -s -p1 -f --dry-run < {patch} >/dev/null || {{ echo 'the mutant does not apply to the code'; exit 98; }}; \
         patch -s -p1 -f < {patch} >/dev/null || exit 98; {{ {run}; }}; s=$?; patch -s -R -p1 -f < {patch} >/dev/null; exit $s",
        patch = q(&patch),
        run = check.run
    );
    let ran = in_shell(audit, &claim.id, &format!("mutant {mutant}: must be caught"), &command, check.timeout.min(audit.settings.check_secs))?;
    let outcome = mutated(ran.exit, ran.interrupted, &ran.output);
    let said = match &outcome {
        Mutated::Caught => "killed".to_string(),
        Mutated::Survived => "survived".to_string(),
        Mutated::Broken(why) => format!("error: {why}"),
    };
    say(&ran, &format!("audit {}: mutant {mutant} {said}", claim.id));
    log.push_str(&format!("$ {command}\n{}\n# mutant {mutant}: {said}\n", ran.output));
    run.commands.push(RunCommand {
        claim: claim.id.clone(),
        what: format!("mutant {mutant}"),
        command,
        exit: ran.exit,
        verdict: said,
        ms: ran.ms,
        by_operator: false,
        output: kept(&ran.output),
    });
    Ok(outcome)
}

/// How much of a command's output a run's record keeps (its transcript has
/// it whole).
const OUTPUT_KEPT: usize = 4000;

/// A command's output as a run keeps it: whole when short, else its start
/// and its end (where the error usually is), with how much was left out.
fn kept(output: &str) -> String {
    let chars: Vec<char> = output.chars().collect();
    if chars.len() <= OUTPUT_KEPT {
        return output.to_string();
    }
    let head: String = chars[..OUTPUT_KEPT / 4].iter().collect();
    let tail: String = chars[chars.len() - OUTPUT_KEPT * 3 / 4..].iter().collect();
    format!("{head}\n[... {} characters left out; the run's transcript has them ...]\n{tail}", chars.len() - OUTPUT_KEPT)
}

/// A mutant's run, read.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Mutated {
    /// The check stopped passing: it notices the bug the mutant plants.
    Caught,
    Survived,
    /// It never ran as a mutant: it didn't apply, the tools weren't there,
    /// the check couldn't start. Counted neither way.
    Broken(String),
}

/// 0: survived. A command that could not be found or run (126, 127), a
/// mutant that did not apply (97, 98), an interrupted run: broken, never
/// caught. Any other failure, a timeout included, is the check noticing.
pub(crate) fn mutated(exit: Option<i32>, interrupted: bool, output: &str) -> Mutated {
    let last = output.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string();
    match exit {
        _ if interrupted => Mutated::Broken("interrupted".into()),
        None => Mutated::Broken("no exit status".into()),
        Some(0) => Mutated::Survived,
        Some(97) => Mutated::Broken("patch is not installed on the machine".into()),
        Some(98) => Mutated::Broken("the mutant does not apply to the code".into()),
        Some(127) => Mutated::Broken(format!("command not found (exit 127){}", if last.is_empty() { String::new() } else { format!(": {last}") })),
        Some(126) => Mutated::Broken(format!("command could not run (exit 126){}", if last.is_empty() { String::new() } else { format!(": {last}") })),
        Some(_) => Mutated::Caught,
    }
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
    run.commands.push(RunCommand { claim: claim.id.clone(), what: "repro".into(), command: command.into(), exit: ran.exit, verdict: if reproduced { "reproduced".into() } else { "passes".into() }, ms: ran.ms, by_operator: false , output: kept(&ran.output) });
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
        &Evidence { claim: claim.id.clone(), check: String::new(), kind: "repro".into(), name: test.into(), path: test.into(), result: if reproduced { "fails".into() } else { "passes".into() }, at: audit::now() },
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

/// Back-translation: a fresh worker, in a tab of its own, reads only the
/// check's files (never the claim) and says what the check guarantees. Its
/// answer is shown beside the claim for the operator to compare.
fn back_translate_in_background(audit: String, claim: String, version: i64, dir: String) {
    if cfg!(test) {
        return;
    }
    let _ = std::thread::Builder::new().name("audit-back".into()).spawn(move || {
        let Some(store) = audit::store() else { return };
        let Ok(Some(a)) = store.audit(&audit) else { return };
        let files = match machine(&review::files_script(&a.work, &dir), 60) {
            Ok(files) if files.exit_code == Some(0) => files,
            Ok(files) => {
                let _ = store.log(&audit, Who::Harness, &format!("{claim}'s check files could not be read for its back-translation"), &files.stderr);
                return;
            }
            Err(error) => {
                let _ = store.log(&audit, Who::Harness, &format!("{claim}'s check files could not be read for its back-translation"), &error);
                return;
            }
        };
        let shown = review::parse(&files.stdout);
        let task = review::task(&shown);
        let review::Shown { expect, partial, .. } = shown;
        want(format!("back:{audit}/{claim}/{version}"), Job::BackTranslate { audit, claim, version, expect, partial }, task);
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
            match run_mutant(&a, &c, &new, mutant, &mut run, &mut log)? {
                Mutated::Caught => {}
                Mutated::Survived => lost.push(format!("mutant {mutant}")),
                Mutated::Broken(why) => lost.push(format!("mutant {mutant} ({why})")),
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
                if check.slot() == new.slot() && check.version < version && check.superseded_by.is_none() {
                    check.superseded_by = Some(version);
                    store.put_check(&audit, &check).map_err(err)?;
                }
            }
            let (verdict, level) = audit::roll_up(&store.current_checks(&audit, &claim).map_err(err)?);
            store
                .change_claim(&audit, &claim, |c| {
                    c.verdict = verdict;
                    c.level = level;
                    Ok(())
                })
                .map_err(err)??;
            store.log(&audit, Who::Harness, &format!("{claim}'s new {} is at least as strong: it replaces v{}", new.slot(), old.version), "").map_err(err)?;
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
            if check.slot() == held.slot() && check.version < held.version && check.superseded_by.is_none() {
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
            // A re-run brings it back from the archive.
            a.archived = false;
            a.archived_at = None;
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
        // Everything with a check runs again: each of its checks.
        let checks = store.current_checks(name, &claim.id).map_err(err)?;
        if !checks.is_empty() {
            unrun(store, name, &checks)?;
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
                    for check in store.current_checks(name, id).map_err(err)?.into_iter().filter(|k| k.verdict == Verdict::Holds) {
                        let ran = in_shell(&audit, id, "verification: must still hold", &check.run, check.timeout)?;
                        if ran.exit != Some(0) {
                            broke.push(format!("{id} {}", check.slot()));
                        }
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
    run_check_now(store, name, claim, None)
}

/// `r` on one check (`k2`): it alone runs again; on a claim, all of them.
pub fn run_check_now(store: &AuditStore, name: &str, claim: Option<&str>, slot: Option<&str>) -> Reply {
    let audit = store.audit(name).map_err(err)?.ok_or(format!("no audit {name}"))?;
    let Some(claim) = claim else { return rerun(store, name, Who::Operator) };
    if store.current_check(name, claim).map_err(err)?.is_none() && store.claim(name, claim).map_err(err)?.is_none_or(|c| c.repro != Repro::Pending) {
        let c = store.claim(name, claim).map_err(err)?.ok_or(format!("no claim {claim}"))?;
        if c.taken {
            return Err(format!("{claim} is yours and has no check yet: e in the claim view writes it, s submits it"));
        }
        crate::util::locked(&RUN_ON_SUBMIT).insert((name.to_string(), claim.to_string()));
        let key = job_key(&Job::Formalize { audit: name.to_string(), claim: claim.to_string(), slot: "k1".into() });
        if workers_of(name).contains(&key) {
            let started = crate::util::locked(&STARTED).iter().find(|(_, job)| job_key(job) == key).map(|(id, _)| id.clone());
            return Ok(match started {
                Some(id) => format!("[audit] {claim}'s check is being written by {} (its tab); it runs as soon as it is submitted", crate::vm::netboxes::agent_label(&id)),
                None => format!("[audit] {claim}'s worker is starting; its check runs as soon as it is submitted"),
            });
        }
        start_claim_worker(store, &audit, &c, Job::Formalize { audit: name.to_string(), claim: claim.to_string(), slot: "k1".into() });
        return Ok(format!("[audit] {claim} has no check yet: a worker is writing one (its tab opens), and it runs as soon as it is submitted"));
    }
    if crate::util::locked(&RUNNING).contains(name) {
        return Err(format!("a batch of {name} is running: {claim} runs after it"));
    }
    let _ = audit;
    crate::util::locked(&RUNNING).insert(name.to_string());
    let (name, claim, slot) = (name.to_string(), claim.to_string(), slot.map(str::to_string));
    let _ = std::thread::Builder::new().name("audit-now".into()).spawn(move || {
        if let Some(store) = audit::store() {
            let checks: Vec<Check> =
                store.current_checks(&name, &claim).unwrap_or_default().into_iter().filter(|k| slot.as_deref().is_none_or(|s| k.slot() == s)).collect();
            let _ = unrun(&store, &name, &checks);
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

mod review;
