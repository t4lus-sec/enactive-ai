//! The audit's rules, tried: each test is one thing the design protects
//! (`claude/audit-spec.md`, "The four rules" and "Checked"), with the agent's
//! machine and the audit's shell played by a closure (`FAKE`).

use super::*;
use crate::audit::{Gate, Policy, Post, Settings};
use crate::audit::ops::StartAsk;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

/// What the machine does: the check's exit, which mutants it catches, the
/// repro's exit, and what check.toml says on submission.
#[derive(Clone, Default)]
struct World {
    check_exit: i32,
    mutants: Vec<&'static str>,
    caught: Vec<&'static str>,
    repro_exit: i32,
    toml: String,
    anchors: String,
}

fn world(w: World) -> Rc<RefCell<World>> {
    let shared = Rc::new(RefCell::new(w));
    let inside = shared.clone();
    FAKE.with(|fake| {
        *fake.borrow_mut() = Some(Box::new(move |script: &str| {
            let w = inside.borrow();
            if script.contains("__TOML__") {
                return (Some(0), format!("__TOML__\n{}\n__ANCHORS__\n{}\n__MUTANTS__\n\n__COMMIT__ abc1234\n", w.toml, w.anchors));
            }
            if script.contains("/mutants 2>/dev/null | grep") {
                return (Some(0), w.mutants.join("\n"));
            }
            if script.contains("patch -s -p1") {
                let caught = w.caught.iter().any(|m| script.contains(m));
                return (Some(if caught { 1 } else { 0 }), String::new());
            }
            if script.contains("REPRO") {
                return (Some(w.repro_exit), "assertion failed: left 'v2' right 'v1'".into());
            }
            if script.contains("CHECK") {
                return (Some(w.check_exit), if w.check_exit == 1 { "State 1: <Initial predicate>\n/\\ x = 1\nState 2: <Step line 3>\n/\\ x = 2".into() } else { "ok".into() });
            }
            (Some(0), String::new())
        }));
    });
    shared
}

/// A fresh audit store and board store, this thread's own.
fn fresh() -> (AuditStore, crate::board::BoardStore) {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!("s0ma-audit-{}-{}", std::process::id(), COUNTER.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("test directory");
    audit::set_store_for_test(root.join("audit.sqlite3"));
    (audit::store().expect("store"), audit::boards().expect("boards"))
}

fn board(boards: &crate::board::BoardStore, name: &str) -> crate::board::Board {
    boards
        .open_board(name, name, "an edit sometimes disappears", "the edit stays", "true", crate::board::GoalTestKind::Command, 60)
        .expect("written")
        .expect("opened")
}

/// An audit running for `board` (diagnose), set up, gate as given.
fn running(store: &AuditStore, board: Option<&str>, mode: Mode, gate: Gate, post: Post) -> Audit {
    let audit = Audit {
        name: "share-slice".into(),
        board: board.map(str::to_string),
        mode,
        status: Status::Checking,
        targets: vec!["/src/s0ma".into()],
        work: "/w/share-slice/work".into(),
        settings: Settings { gate, post, ..Settings::default() },
        gone_at: Some(audit::now()),
        created_at: audit::now(),
        ..Audit::default()
    };
    store.create(&audit).expect("created");
    audit
}

fn claim(store: &AuditStore, audit: &str, text: &str, review: Review) -> Claim {
    store.add_claim(audit, Claim { text: text.into(), author: "s0ma".into(), source: "src/share_sync.rs:40".into(), review, risk: 8, ..Claim::default() }).expect("claim")
}

fn check(store: &AuditStore, audit: &str, claim: &str, kind: &str) -> Check {
    store
        .add_check(audit, Check { claim: claim.into(), dir: format!("checks/{claim}"), lang: "any".into(), kind: kind.into(), run: "./CHECK".into(), timeout: 60, by: "a1-2".into(), ..Check::default() })
        .expect("check")
}

fn batch(store: &AuditStore, ids: &[&str]) {
    let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
    crate::approval::by_operator(|| run_batch(store, "share-slice", &ids, false)).expect("batch ran");
}

fn findings(boards: &crate::board::BoardStore, name: &str) -> Vec<String> {
    let found = boards.board(name).unwrap().unwrap();
    boards.facts(&found.id).unwrap().into_iter().map(|f| f.statement).collect()
}

// ------------------------------------------------------------------ rule 1

#[test]
fn rule_1_no_tool_argument_writes_a_verdict_or_a_level() {
    let (store, _) = fresh();
    running(&store, None, Mode::Assure, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "A download never replaces a file changed meanwhile", Review::Proposed);
    for (tool, args) in [
        ("audit_claims_propose", json!({"audit": "share-slice", "claims": [{"text": "x holds", "source": "a", "kind": "pure", "risk": 3, "why": "w"}], "verdict": "holds"})),
        ("audit_plan_note", json!({"audit": "share-slice", "claim": c.id, "why": "w", "level": "proven"})),
        ("audit_check_submit", json!({"audit": "share-slice", "claim": c.id, "dir": "checks/c1", "result": "passed"})),
    ] {
        let call = crate::types::ToolCall { id: "1".into(), kind: "function".into(), function: crate::types::FunctionCall { name: tool.into(), arguments: args.to_string() } };
        let said = crate::audit::tools::execute(&call, "s0ma");
        assert!(said.contains("harness's to write"), "{tool}: {said}");
    }
    let after = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert_eq!((after.verdict, after.level), (Verdict::Unchecked, Level::Documented));
    // The operator's own actions change what is checked, never a verdict.
    for action in ["accept", "pin", "priority", "skip", "depth", "take", "back"] {
        let _ = crate::audit::ops::operator_claim(&store, "share-slice", &c.id, action, "3");
        assert_eq!(store.claim("share-slice", &c.id).unwrap().unwrap().verdict, Verdict::Unchecked, "{action}");
    }
}

// ------------------------------------------------------------------ rule 2

#[test]
fn rule_2_a_failure_reaches_the_board_only_once_reproduced() {
    let (store, boards) = fresh();
    board(&boards, "share-edit-lost");
    running(&store, Some("share-edit-lost"), Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "A download never replaces a file changed meanwhile", Review::Proposed);
    check(&store, "share-slice", &c.id, "tla");
    let _w = world(World { check_exit: 1, repro_exit: 1, ..World::default() });
    batch(&store, &[&c.id]);
    let failed = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert_eq!(failed.verdict, Verdict::Fails);
    assert!(failed.trace.is_some(), "the counterexample kept");
    assert!(findings(&boards, "share-edit-lost").is_empty(), "an unreproduced trace never reaches the board");
    // A repro that passes on the real code is no finding either.
    let said = submit_repro(&store, "share-slice", &c.id, "tests/race.rs::lost", "cargo test REPRO");
    assert!(said.is_ok(), "{said:?}");
    {
        let _w = world(World { check_exit: 1, repro_exit: 0, ..World::default() });
        batch(&store, &[&c.id]);
    }
    assert_eq!(store.claim("share-slice", &c.id).unwrap().unwrap().repro, Repro::Passes);
    assert!(findings(&boards, "share-edit-lost").is_empty(), "a repro that passes is not a repro");
    // One that fails there is.
    submit_repro(&store, "share-slice", &c.id, "tests/race.rs::lost", "cargo test REPRO").unwrap();
    let _w = world(World { check_exit: 1, repro_exit: 1, ..World::default() });
    batch(&store, &[&c.id]);
    let found = findings(&boards, "share-edit-lost");
    assert!(found.iter().any(|f| f.starts_with(&format!("+ {} fails", c.id)) && f.contains("audit:share-slice/")), "{found:?}");
    // Model-only is never a finding.
    let other = claim(&store, "share-slice", "No version is lost in the mirror", Review::Accepted);
    store.change_claim("share-slice", &other.id, |c| { c.verdict = Verdict::Fails; Ok(()) }).unwrap().unwrap();
    model_only(&store, "share-slice", &other.id, "the interleaving needs a kernel hook").unwrap();
    assert_eq!(findings(&boards, "share-edit-lost").len(), 1);
}

// ------------------------------------------------------------------ rule 3

#[test]
fn rule_3_an_unreviewed_hold_is_diagnostic_and_never_counts() {
    let (store, boards) = fresh();
    board(&boards, "share-edit-lost");
    running(&store, Some("share-edit-lost"), Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "Line-index lookups match a full scan", Review::Proposed);
    check(&store, "share-slice", &c.id, "bounded");
    let _w = world(World { check_exit: 0, mutants: vec!["m1.patch", "m2.patch"], caught: vec!["m1.patch", "m2.patch"], ..World::default() });
    batch(&store, &[&c.id]);
    let held = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert_eq!((held.verdict, held.level, held.kills), (Verdict::Holds, Level::Bounded, Some((2, 2))));
    assert!(!held.counts(), "a diagnose result on an unreviewed claim is never assurance");
    let found = findings(&boards, "share-edit-lost");
    assert!(found.iter().any(|f| f.starts_with(&format!("− {} holds (bounded)", c.id)) && f.contains("diagnostic")), "{found:?}");
    // In assure mode an unaccepted claim is not even checked.
    let (store, _) = fresh();
    let audit = running(&store, None, Mode::Assure, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "x never y", Review::Proposed);
    check(&store, "share-slice", &c.id, "test");
    assert!(ready_claims(&store, &audit).is_empty());
    crate::audit::ops::operator_claim(&store, "share-slice", &c.id, "accept", "").unwrap();
    assert_eq!(ready_claims(&store, &audit), vec![c.id]);
}

#[test]
fn a_hold_under_an_unaccepted_assumption_says_if_a2() {
    let (store, boards) = fresh();
    board(&boards, "share-edit-lost");
    running(&store, Some("share-edit-lost"), Mode::Diagnose, Gate::None, Post::Auto);
    let a = store.add_claim("share-slice", Claim { text: "assumes: crate::share::files::write replaces only if still listed".into(), kind: "assumes".into(), ..Claim::default() }).unwrap();
    let c = claim(&store, "share-slice", "A download never overwrites", Review::Accepted);
    store.add_check("share-slice", Check { claim: c.id.clone(), dir: format!("checks/{}", c.id), kind: "bounded".into(), run: "./CHECK".into(), assumes: vec![a.id.clone()], ..Check::default() }).unwrap();
    store.change_claim("share-slice", &c.id, |c| { c.rests_on = vec![a.id.clone()]; Ok(()) }).unwrap().unwrap();
    let _w = world(World { check_exit: 0, mutants: vec!["m1.patch"], caught: vec!["m1.patch"], ..World::default() });
    batch(&store, &[&c.id]);
    let found = findings(&boards, "share-edit-lost");
    assert!(found.iter().any(|f| f.contains(&format!("if {}", a.id))), "a guessed stub rules nothing out plainly: {found:?}");
}

#[test]
fn a_check_that_catches_no_mutant_is_vacuous_and_documented() {
    let (store, _) = fresh();
    running(&store, None, Mode::Assure, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "Redaction never changes the line count", Review::Accepted);
    check(&store, "share-slice", &c.id, "property");
    let _w = world(World { check_exit: 0, mutants: vec!["m1.patch", "m2.patch"], caught: vec![], ..World::default() });
    batch(&store, &[&c.id]);
    let after = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert_eq!((after.verdict, after.level), (Verdict::Vacuous, Level::Documented));
    assert!(!after.counts());
}

// ------------------------------------------------------------------ rule 4

#[test]
fn rule_4_a_weaker_check_for_a_relied_on_claim_waits() {
    let (store, _) = fresh();
    running(&store, None, Mode::Assure, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "Redaction never changes the line count", Review::Accepted);
    let mut first = check(&store, "share-slice", &c.id, "property");
    first.killed = vec!["m1.patch".into()];
    store.put_check("share-slice", &first).unwrap();
    store.change_claim("share-slice", &c.id, |c| { c.verdict = Verdict::Holds; c.level = Level::Property; Ok(()) }).unwrap().unwrap();
    // The new one no longer catches m1.
    let _w = world(World { toml: "claim = \"c1\"\nkind = \"property\"\nrun = \"./CHECK\"".into(), mutants: vec!["m1.patch"], caught: vec![], ..World::default() });
    submit_check(&store, "share-slice", &c.id, "checks/c1", "a1-3", None).unwrap();
    let after = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert!(after.weaker.as_deref().is_some_and(|w| w.contains("m1.patch")), "{:?}", after.weaker);
    assert_eq!(after.verdict, Verdict::Holds, "the old verdict stands meanwhile");
    assert_eq!(store.current_check("share-slice", &c.id).unwrap().unwrap().version, first.version, "the old check stays current");
    // One that still catches it replaces it.
    let _w = world(World { toml: "claim = \"c1\"\nkind = \"property\"\nrun = \"./CHECK\"".into(), mutants: vec!["m1.patch"], caught: vec!["m1.patch"], ..World::default() });
    submit_check(&store, "share-slice", &c.id, "checks/c1", "a1-3", None).unwrap();
    assert!(store.current_check("share-slice", &c.id).unwrap().unwrap().version > first.version);
}

#[test]
fn a_clause_without_an_anchor_is_refused() {
    let (store, _) = fresh();
    running(&store, None, Mode::Assure, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "Redaction never changes the line count; and it keeps every byte outside the ranges", Review::Accepted);
    let _w = world(World { toml: "kind = \"test\"\nrun = \"./CHECK\"".into(), anchors: "1".into(), ..World::default() });
    let said = submit_check(&store, "share-slice", &c.id, "checks/c1", "a1-3", None);
    assert!(said.as_ref().is_err_and(|e| e.contains("anchor")), "{said:?}");
}

// ------------------------------------------------------------------ the operator's controls

#[test]
fn a_proposal_starts_nothing_and_off_refuses() {
    let (store, boards) = fresh();
    board(&boards, "share-edit-lost");
    let ask = StartAsk { name: "share-slice".into(), board: Some("share-edit-lost".into()), targets: vec!["/src/s0ma".into()], why: "the bug is in the mirror".into(), ..StartAsk::default() };
    // Ceilings that would let it start, were it auto.
    store.set_policy("share-edit-lost", Policy::Ask, Some(crate::audit::Ceilings { steps: 1000, depth: Level::Proven, workers: 8 })).unwrap();
    crate::audit::ops::s0ma_start(&store, ask.clone()).unwrap();
    let audit = store.audit("share-slice").unwrap().unwrap();
    assert_eq!(audit.status, Status::Proposed);
    assert!(audit.lead.is_none() && store.links(Some("share-slice"), None).unwrap().is_empty(), "nothing started");
    assert!(ready_claims(&store, &audit).is_empty());
    store.set_policy("share-edit-lost", Policy::Off, None).unwrap();
    let refused = crate::audit::ops::s0ma_start(&store, StartAsk { name: "other".into(), ..ask });
    assert!(refused.is_err_and(|e| e.contains("refused")));
}

#[test]
fn nothing_is_checked_before_go_and_new_rows_wait_for_another() {
    let (store, _) = fresh();
    let mut audit = running(&store, None, Mode::Diagnose, Gate::Plan, Post::Auto);
    store.change("share-slice", |a| { a.gone_at = None; Ok(()) }).unwrap().unwrap();
    audit.gone_at = None;
    let c = claim(&store, "share-slice", "x never y", Review::Proposed);
    check(&store, "share-slice", &c.id, "test");
    assert!(ready_claims(&store, &audit).is_empty(), "the plan is checked only after g");
    crate::audit::ops::go(&store, "share-slice").unwrap();
    let audit = store.audit("share-slice").unwrap().unwrap();
    assert_eq!(ready_claims(&store, &audit), vec![c.id.clone()]);
    let late = store.add_claim("share-slice", Claim { text: "a never b".into(), new: true, ..Claim::default() }).unwrap();
    check(&store, "share-slice", &late.id, "test");
    assert!(!ready_claims(&store, &audit).contains(&late.id), "a row added after go waits for another g");
}

#[test]
fn a_claim_the_operator_took_is_never_formalized_by_s0ma() {
    let (store, _) = fresh();
    running(&store, None, Mode::Assure, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "x never y", Review::Accepted);
    crate::audit::ops::operator_claim(&store, "share-slice", &c.id, "take", "").unwrap();
    let said = submit_check(&store, "share-slice", &c.id, "checks/c1", "a1-2", None);
    assert!(said.is_err_and(|e| e.contains("operator's")));
    let call = crate::types::ToolCall { id: "1".into(), kind: "function".into(), function: crate::types::FunctionCall { name: "audit_take".into(), arguments: json!({"audit": "share-slice", "claim": c.id}).to_string() } };
    assert!(crate::audit::tools::execute(&call, "s0ma").contains("error") || crate::audit::tools::execute(&call, "s0ma").contains("operator"));
}

#[test]
fn a_waiver_is_never_a_pass() {
    let (store, _) = fresh();
    running(&store, None, Mode::Assure, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "x never y", Review::Accepted);
    store.change_claim("share-slice", &c.id, |c| { c.verdict = Verdict::Fails; Ok(()) }).unwrap().unwrap();
    crate::audit::ops::operator_claim(&store, "share-slice", &c.id, "waive", "").unwrap();
    let waived = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert!(waived.waived && waived.verdict == Verdict::Fails && !waived.counts());
    assert_eq!(waived.mark().0, "[-]");
    check(&store, "share-slice", &c.id, "test");
    let _w = world(World { check_exit: 1, ..World::default() });
    let holding = command("audit-run share-slice --holding").expect("ran");
    assert_eq!(holding.exit_code, Some(0), "left out of --holding: {}", holding.stdout);
    assert!(holding.stdout.starts_with("0 claims"), "{}", holding.stdout);
}

#[test]
fn keep_never_turns_unreviewed_claims_into_accepted_ones() {
    let (store, boards) = fresh();
    board(&boards, "share-edit-lost");
    running(&store, Some("share-edit-lost"), Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "x never y", Review::Proposed);
    store.change_claim("share-slice", &c.id, |c| { c.verdict = Verdict::Holds; c.level = Level::Bounded; Ok(()) }).unwrap().unwrap();
    after_project(&store, "share-slice", "keep", None).unwrap();
    let kept = store.audit("share-slice").unwrap().unwrap();
    assert_eq!((kept.status, kept.mode, kept.board.clone()), (Status::Standing, Mode::Assure, None));
    let after = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert_eq!(after.review, Review::Proposed);
    assert!(!after.counts());
}

#[test]
fn an_audit_never_closes_its_board() {
    let (store, boards) = fresh();
    board(&boards, "share-edit-lost");
    let audit = running(&store, Some("share-edit-lost"), Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "x never y", Review::Accepted);
    check(&store, "share-slice", &c.id, "test");
    let _w = world(World { check_exit: 0, mutants: vec!["m1.patch"], caught: vec!["m1.patch"], ..World::default() });
    batch(&store, &[&c.id]);
    maybe_done(&store, &store.audit("share-slice").unwrap().unwrap());
    assert_eq!(store.audit("share-slice").unwrap().unwrap().status, Status::Done);
    assert_eq!(boards.board("share-edit-lost").unwrap().unwrap().status, crate::board::BoardStatus::Open, "the project's test judges the bug");
    let _ = audit;
}

#[test]
fn a_report_cites_only_what_it_names() {
    let (store, boards) = fresh();
    board(&boards, "share-edit-lost");
    running(&store, Some("share-edit-lost"), Mode::Diagnose, Gate::None, Post::Auto);
    let a = claim(&store, "share-slice", "x never y", Review::Accepted);
    let b = claim(&store, "share-slice", "z never w", Review::Accepted);
    let cited = crate::audit::report::cite("share-edit-lost", &format!("The bug is share-slice/{} (reproduced).", a.id), 1);
    assert_eq!(cited, vec![("share-slice".to_string(), a.id.clone())]);
    assert!(!store.claim("share-slice", &a.id).unwrap().unwrap().cited.is_empty());
    assert!(store.claim("share-slice", &b.id).unwrap().unwrap().cited.is_empty(), "an uncited result is not re-run");
    assert_eq!(crate::audit::report::cited_ids(&store, "share-slice", "share-edit-lost"), vec![a.id.clone()], "verification re-runs exactly the cited");
}


#[test]
fn after_go_workers_write_the_checks_a_few_at_a_time() {
    let (store, _) = fresh();
    let audit = Audit {
        name: "plan-workers".into(),
        mode: Mode::Assure,
        status: Status::Checking,
        work: "/w/plan-workers/work".into(),
        settings: Settings { gate: Gate::Plan, workers: 2, ..Settings::default() },
        created_at: audit::now(),
        ..Audit::default()
    };
    store.create(&audit).expect("created");
    for text in ["a never b", "c never d", "e never f"] {
        let c = store.add_claim("plan-workers", Claim { text: text.into(), source: "x.rs:1".into(), ..Claim::default() }).unwrap();
        crate::audit::ops::operator_claim(&store, "plan-workers", &c.id, "accept", "").unwrap();
    }
    // Accepted is not enough under the plan gate, in assure mode too.
    queue_workers(&store, &store.audit("plan-workers").unwrap().unwrap());
    assert!(workers_of("plan-workers").is_empty(), "nothing before go");
    assert!(!store.claim("plan-workers", "c1").unwrap().unwrap().checkable(&store.audit("plan-workers").unwrap().unwrap()));
    crate::audit::ops::go(&store, "plan-workers").unwrap();
    queue_workers(&store, &store.audit("plan-workers").unwrap().unwrap());
    assert_eq!(workers_of("plan-workers").len(), 2, "at most the audit's workers");
    let held = store.claims("plan-workers").unwrap().iter().filter(|c| c.held_by.is_some()).count();
    assert_eq!(held, 2, "each one's claim held for it");
    let task = crate::util::locked(&WANTED).iter().find(|(k, _, _)| k == "formalize:plan-workers/c1/k1").map(|(_, _, t)| t.clone()).expect("c1's worker");
    assert!(task.contains("audit_check_submit") && task.contains("[1] a never b") && task.contains("mutants/*.patch"), "{task}");
    // A taken claim gets no worker; run now on one with no check asks for one.
    crate::audit::ops::operator_claim(&store, "plan-workers", "c3", "take", "").unwrap();
    assert!(run_now(&store, "plan-workers", Some("c3")).is_err_and(|e| e.contains("yours")));
    crate::util::locked(&WANTED).retain(|(_, job, _)| job.audit() != "plan-workers");
}

/// The tab reads the store every frame while the scheduler, the batch and
/// the workers write it: a read takes no write lock, so a writer holding it
/// never stalls the screen (it once did, for seconds, after go).
#[test]
fn a_read_waits_for_no_writer() {
    let (store, _) = fresh();
    running(&store, None, Mode::Diagnose, Gate::None, Post::Auto);
    claim(&store, "share-slice", "a download never replaces a file changed meanwhile", Review::Accepted);
    let mut writer = crate::db::open(&store.path).expect("writer");
    let held = writer.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).expect("write lock");
    let started = std::time::Instant::now();
    assert_eq!(store.claims("share-slice").expect("read").len(), 1);
    assert!(store.audit("share-slice").expect("read").is_some());
    assert!(started.elapsed() < Duration::from_secs(1), "a read waited {:?} for the writer", started.elapsed());
    drop(held);
}

/// Workers stop with the harness: on its next start, nothing they held is
/// left waiting out its half-hour hold.
#[test]
fn a_restart_lets_go_of_what_stopped_workers_held() {
    let (store, _) = fresh();
    running(&store, None, Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "a download never replaces a file changed meanwhile", Review::Accepted);
    let until = (chrono::Local::now() + chrono::Duration::seconds(audit::HOLD_SECS)).to_rfc3339();
    store
        .change_claim("share-slice", &c.id, |c| {
            c.held_by = Some("w·b1-1".into());
            c.held_until = Some(until.clone());
            c.running = true;
            Ok(())
        })
        .expect("held")
        .expect("held");
    recover(&store);
    let after = store.claim("share-slice", &c.id).expect("read").expect("claim");
    assert!(after.held_by.is_none() && after.held_until.is_none() && !after.running);
}

/// Threads opening and closing the store at once, as the tab, the scheduler
/// and the workers do: none waits on another for good. (SQLite 3.51 could
/// deadlock here: one connection's close took its file's lock and then the
/// VFS's, while an open took them the other way round.)
#[test]
fn stores_opened_and_closed_at_once_never_deadlock() {
    let (store, _) = fresh();
    running(&store, None, Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "a download never replaces a file changed meanwhile", Review::Accepted);
    let path = store.path.clone();
    let (done, finished) = std::sync::mpsc::channel();
    // Two: the race needs the closing connection to be the file's only one.
    for n in 0..2 {
        let (path, id, done) = (path.clone(), c.id.clone(), done.clone());
        std::thread::spawn(move || {
            let store = AuditStore::at(path);
            let until = std::time::Instant::now() + Duration::from_secs(3);
            while std::time::Instant::now() < until {
                let _ = store.claims("share-slice");
                if n % 2 == 0 {
                    let _ = store.change_claim("share-slice", &id, |c| {
                        c.priority = (c.priority + 1) % 10;
                        Ok(())
                    });
                }
            }
            let _ = done.send(());
        });
    }
    for _ in 0..2 {
        finished.recv_timeout(Duration::from_secs(20)).expect("a thread stuck on the store's locks");
    }
}

/// A failure is not the end of its claim: the audit stays open until a
/// worker has reproduced it or said why it can't, even when every worker was
/// busy as the check failed.
#[test]
fn a_failure_keeps_the_audit_open_for_its_repro() {
    let (store, _) = fresh();
    let audit = running(&store, None, Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "a download never replaces a file changed meanwhile", Review::Accepted);
    store
        .change_claim("share-slice", &c.id, |c| {
            c.verdict = Verdict::Fails;
            Ok(())
        })
        .expect("failed")
        .expect("failed");
    maybe_done(&store, &audit);
    assert_eq!(store.audit("share-slice").expect("read").expect("audit").status, Status::Checking);
    store
        .change_claim("share-slice", &c.id, |c| {
            c.repro = Repro::Fails;
            Ok(())
        })
        .expect("reproduced")
        .expect("reproduced");
    maybe_done(&store, &audit);
    assert_eq!(store.audit("share-slice").expect("read").expect("audit").status, Status::Done);
}

/// A claim has several checks, each with its own result: one that fails
/// makes the claim fail; the others keep theirs, and a new one beside a
/// check that holds runs alone.
#[test]
fn a_claim_reads_from_all_its_checks() {
    let (store, _) = fresh();
    running(&store, None, Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "Redaction never changes the line count", Review::Accepted);
    check(&store, "share-slice", &c.id, "test");
    let _w = world(World { check_exit: 0, mutants: vec!["m1.patch"], caught: vec!["m1.patch"], ..World::default() });
    batch(&store, &[&c.id]);
    let after = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert_eq!((after.verdict, after.level), (Verdict::Holds, Level::Tested));
    // k2, deeper, beside it: only k2 runs, and it fails.
    store
        .add_check("share-slice", Check { claim: c.id.clone(), slot: "k2".into(), dir: format!("checks/{}/k2", c.id), kind: "bounded".into(), run: "./CHECK".into(), timeout: 60, ..Check::default() })
        .unwrap();
    assert_eq!(ready_claims(&store, &store.audit("share-slice").unwrap().unwrap()), vec![c.id.clone()]);
    let _w = world(World { check_exit: 1, ..World::default() });
    batch(&store, &[&c.id]);
    let checks = store.current_checks("share-slice", &c.id).unwrap();
    assert_eq!(checks.iter().map(|k| (k.slot().to_string(), k.verdict)).collect::<Vec<_>>(), vec![("k1".into(), Verdict::Holds), ("k2".into(), Verdict::Fails)]);
    assert_eq!(store.claim("share-slice", &c.id).unwrap().unwrap().verdict, Verdict::Fails);
    assert!(ready_claims(&store, &store.audit("share-slice").unwrap().unwrap()).is_empty(), "both have run");
}

/// A batch that stops part way still closes its run: when it ended, why,
/// and every command it ran with what it printed.
#[test]
fn a_batch_that_stops_still_closes_its_run() {
    let (store, _) = fresh();
    running(&store, None, Mode::Diagnose, Gate::None, Post::Auto);
    let c1 = claim(&store, "share-slice", "Redaction never changes the line count", Review::Accepted);
    let c2 = claim(&store, "share-slice", "A download never replaces a file changed meanwhile", Review::Accepted);
    check(&store, "share-slice", &c1.id, "test");
    store
        .add_check("share-slice", Check { claim: c2.id.clone(), dir: format!("checks/{}", c2.id), kind: "test".into(), run: "./GONE".into(), timeout: 60, ..Check::default() })
        .unwrap();
    FAKE.with(|fake| {
        *fake.borrow_mut() = Some(Box::new(|script: &str| {
            if script.contains("GONE") {
                return (None, "__GONE__".into());
            }
            if script.contains("CHECK") {
                return (Some(0), "1 passed".into());
            }
            (Some(0), String::new())
        }));
    });
    let ids = vec![c1.id.clone(), c2.id.clone()];
    let outcome = crate::approval::by_operator(|| run_batch(&store, "share-slice", &ids, false));
    let error = outcome.expect_err("its shell went");
    assert!(error.starts_with("batch ") && error.contains(&c2.id), "{error}");
    let runs = store.runs("share-slice").unwrap();
    let run = runs.last().expect("a run");
    assert!(run.finished_at.is_some(), "closed");
    assert!(run.note.starts_with("stopped:"), "{}", run.note);
    let first = run.commands.iter().find(|k| k.claim == c1.id).expect("c1's command kept");
    assert_eq!((first.command.as_str(), first.output.as_str()), ("./CHECK", "1 passed"));
}

/// A worker that ends without submitting, but left its check in its
/// folder: the harness submits it, under the worker's full job id.
#[test]
fn a_check_left_on_disk_is_submitted_for_its_worker() {
    let (store, _) = fresh();
    running(&store, None, Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "Redaction never changes the line count", Review::Accepted);
    let _w = world(World { toml: "claim = \"c1\"\nkind = \"test\"\nrun = \"./CHECK\"".into(), ..World::default() });
    let probe_said = Rc::new(RefCell::new(false));
    let inside = probe_said.clone();
    let world_fake = FAKE.with(|fake| fake.borrow_mut().take()).expect("world");
    let mut world_fake = world_fake;
    FAKE.with(|fake| {
        *fake.borrow_mut() = Some(Box::new(move |script: &str| {
            if script.contains("check.toml\" ] && echo") {
                *inside.borrow_mut() = true;
                return (Some(0), "checks/c1\n".into());
            }
            world_fake(script)
        }));
    });
    let id = "bg-0011aabb".to_string();
    crate::util::locked(&STARTED).push((id.clone(), Job::Formalize { audit: "share-slice".into(), claim: c.id.clone(), slot: "k1".into() }));
    let line = worker_finished(&id, "completed", "done", None).expect("ours");
    assert!(line.contains("looks for a check"), "{line}");
    assert!(*probe_said.borrow(), "its folder was looked in");
    let checks = store.checks("share-slice", &c.id).unwrap();
    assert_eq!(checks.len(), 1, "submitted for it");
    assert_eq!(checks[0].by_job, id);
    let after = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert!(!after.skipped && after.blocked.is_none() && after.held_by.is_none());
}

/// A check refused (here: a kind the harness doesn't take) and nothing
/// else in: the claim is blocked with the reason, not skipped.
#[test]
fn a_refused_check_blocks_its_claim_rather_than_skipping_it() {
    let (store, _) = fresh();
    running(&store, None, Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "Redaction never changes the line count", Review::Accepted);
    let _w = world(World { toml: "claim = \"c1\"\nkind = \"vibes\"\nrun = \"./CHECK\"".into(), ..World::default() });
    let id = "bg-0022ccdd".to_string();
    crate::util::locked(&STARTED).push((id.clone(), Job::Formalize { audit: "share-slice".into(), claim: c.id.clone(), slot: "k1".into() }));
    let label = crate::vm::netboxes::agent_label(&id);
    assert!(submit_check(&store, "share-slice", &c.id, "checks/c1", &label, Some("k1")).is_err());
    worker_finished(&id, "completed", "gave up", None).expect("ours");
    let after = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert!(!after.skipped, "not skipped");
    assert!(after.blocked.as_deref().is_some_and(|b| b.contains("vibes")), "{:?}", after.blocked);
    assert!(!after.checkable(&store.audit("share-slice").unwrap().unwrap()), "not tried again on its own");
    crate::audit::ops::operator_claim(&store, "share-slice", &c.id, "skip", "").unwrap();
    let after = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert!(after.blocked.is_none() && !after.skipped, "s unblocks it");
}

/// Back-translation reads whole files and must quote each one's last line:
/// a reviewer that read less is marked partial.
#[test]
fn a_back_translation_must_show_it_read_every_file() {
    let stdout = "__FILE__ 120 5 0123456789ab check.py\n    1 | def helper(x):\n    2 |     return x\n    3 | \n    4 | def test_add():\n    5 |     assert helper(2) == 2\n__LAST__ check.py\t    assert helper(2) == 2\n__FILE__ 30 2 ba9876543210 check.toml\n    1 | kind = \"test\"\n    2 | run = \"pytest\"\n__LAST__ check.toml\trun = \"pytest\"\n";
    let shown = review::parse(stdout);
    assert_eq!(shown.expect.len(), 2);
    assert!(shown.partial.is_empty());
    assert!(shown.defined.contains(&"helper (check.py:1)".to_string()) && shown.defined.contains(&"test_add (check.py:4)".to_string()), "{:?}", shown.defined);
    let task = review::task(&shown);
    assert!(task.contains("check.py · 120 bytes · 5 lines · sha256 0123456789ab") && task.contains("SEEN"), "{task}");
    let whole = review::settle("It checks helper(2) is 2.\nSEEN check.py: assert helper(2) == 2\nSEEN check.toml: `run = \"pytest\"`", &shown.expect, &shown.partial);
    assert!(whole.contains("[+] read whole: 2 files") && !whole.contains("SEEN"), "{whole}");
    let short = review::settle("It checks helper.\nSEEN check.py: def helper(x):", &shown.expect, &shown.partial);
    assert!(short.starts_with("[~] partial review") && short.contains("check.py's last line") && short.contains("check.toml was not confirmed"), "{short}");
    let cut = review::parse("__FILE__ 900000 20000 aaaaaaaaaaaa big.py\n    1 | x = 1\n__CUT__\n__LAST__ big.py\tx = 20000\n");
    assert_eq!(cut.partial.len(), 1);
    assert!(review::task(&cut).contains("only its first"));
}

/// A mutant whose run couldn't start (a tool missing, a patch that doesn't
/// apply) is neither caught nor survived.
#[test]
fn a_mutant_that_never_ran_is_counted_neither_way() {
    assert!(matches!(mutated(Some(127), false, "sh: pytest: command not found"), Mutated::Broken(why) if why.contains("pytest")));
    assert!(matches!(mutated(Some(98), false, ""), Mutated::Broken(_)));
    assert!(matches!(mutated(Some(1), true, ""), Mutated::Broken(_)));
    assert!(matches!(mutated(Some(1), false, "AssertionError"), Mutated::Caught));
    assert!(matches!(mutated(Some(0), false, ""), Mutated::Survived));
}

/// Archived, an audit leaves the agent's context and its board's audits,
/// with everything kept; it comes back as it was, and a re-run brings it
/// back too.
#[test]
fn an_archived_audit_is_hidden_and_kept() {
    let (store, boards) = fresh();
    board(&boards, "share-edit-lost");
    running(&store, Some("share-edit-lost"), Mode::Diagnose, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "Redaction never changes the line count", Review::Accepted);
    assert!(crate::audit::ops::lines(&store).unwrap().iter().any(|(name, _)| name == "share-slice"));
    let said = crate::audit::ops::archive(&store, "share-slice", true).expect("archived");
    assert!(said.contains("archived and stopped"), "{said}");
    let audit = store.audit("share-slice").unwrap().unwrap();
    assert!(audit.archived && !audit.shown() && audit.status == Status::Stopped);
    assert!(crate::audit::ops::lines(&store).unwrap().iter().all(|(name, _)| name != "share-slice"), "out of the agent's context");
    assert!(crate::audit::ops::for_board("share-edit-lost").is_empty(), "out of the projects tab");
    assert!(store.claim("share-slice", &c.id).unwrap().is_some(), "its claims kept");
    crate::audit::ops::archive(&store, "share-slice", false).expect("back");
    assert!(!store.audit("share-slice").unwrap().unwrap().archived);
    crate::audit::ops::archive(&store, "share-slice", true).expect("archived again");
    rerun(&store, "share-slice", Who::Operator).expect("re-run");
    assert!(!store.audit("share-slice").unwrap().unwrap().archived, "a re-run brings it back");
}

/// Deleted, an audit is gone from the store with all it had, its work
/// tree and branch removed; another audit is untouched. Not while a batch
/// is on a command: then it is stopped, and deleted on the next try.
#[test]
fn a_deleted_audit_is_gone_with_its_work_tree() {
    let (store, _) = fresh();
    let audit = running(&store, None, Mode::Diagnose, Gate::None, Post::Auto);
    store
        .change("share-slice", |a| {
            a.kind = "git".into();
            a.branch = "s0ma-audit/share-slice".into();
            Ok(())
        })
        .unwrap()
        .unwrap();
    let c = claim(&store, "share-slice", "Redaction never changes the line count", Review::Accepted);
    check(&store, "share-slice", &c.id, "test");
    store.log("share-slice", Who::Operator, "something", "").unwrap();
    store.create(&Audit { name: "other".into(), status: Status::Standing, created_at: audit::now(), ..Audit::default() }).unwrap();
    claim(&store, "other", "A note is never lost", Review::Accepted);
    let scripts = Rc::new(RefCell::new(Vec::<String>::new()));
    let seen = scripts.clone();
    FAKE.with(|fake| {
        *fake.borrow_mut() = Some(Box::new(move |script: &str| {
            seen.borrow_mut().push(script.to_string());
            (Some(0), if script.contains("rm -rf") { "REMOVED".into() } else { String::new() })
        }));
    });
    crate::util::locked(&RUNNING).insert("share-slice".into());
    let refused = crate::audit::ops::delete(&store, "share-slice").expect_err("a batch is running");
    assert!(refused.contains("again"), "{refused}");
    assert_eq!(store.audit("share-slice").unwrap().unwrap().status, Status::Stopped);
    crate::util::locked(&RUNNING).remove("share-slice");
    let said = crate::audit::ops::delete(&store, "share-slice").expect("deleted");
    assert!(said.ends_with("deleted"), "{said}");
    assert!(store.audit("share-slice").unwrap().is_none());
    assert!(store.claims("share-slice").unwrap().is_empty() && store.checks("share-slice", &c.id).unwrap().is_empty() && store.runs("share-slice").unwrap().is_empty());
    let script = scripts.borrow().join("\n");
    assert!(script.contains("worktree remove --force") && script.contains("branch -D") && script.contains(&format!("rm -rf -- {}", q(audit.work.trim_end_matches("/work")))), "{script}");
    assert_eq!(store.claims("other").unwrap().len(), 1, "another audit untouched");
}

/// An unreviewed claim waits on the operator only where the audit needs
/// his review to check it (assure mode, or the `reviewed` gate); elsewhere
/// it is checked anyway, and nothing waits.
#[test]
fn only_a_review_the_audit_needs_waits_on_the_operator() {
    let (store, _) = fresh();
    let audit = running(&store, None, Mode::Diagnose, Gate::Plan, Post::Auto);
    let c = claim(&store, "share-slice", "Redaction never changes the line count", Review::Proposed);
    let a = store
        .add_claim("share-slice", Claim { text: "assumes: store::save writes only under the lock".into(), kind: "assumes".into(), source: "src/x.rs:1".into(), review: Review::Proposed, ..Claim::default() })
        .expect("assumption");
    assert!(!c.waits_on(&audit) && !a.waits_on(&audit), "diagnose, gate plan: checked without review");
    let reviewed = Audit { settings: Settings { gate: Gate::Reviewed, ..audit.settings.clone() }, ..audit.clone() };
    assert!(c.waits_on(&reviewed) && !a.waits_on(&reviewed), "the reviewed gate: the claim waits, an assumption never does");
    let assure = Audit { mode: Mode::Assure, ..audit.clone() };
    assert!(c.waits_on(&assure));
    let accepted = Claim { review: Review::Accepted, ..c.clone() };
    assert!(!accepted.waits_on(&assure));
    let question = Claim { readings: vec!["one".into(), "two".into()], ..accepted.clone() };
    assert!(question.waits_on(&audit), "an open question does");
    let blocked = Claim { blocked: Some("the python adapter's doctor run is going".into()), blocked_on: Some("python".into()), ..accepted.clone() };
    assert!(!blocked.waits_on(&audit) && blocked.mark().0 == "[*]", "a doctor run under way is not his");
    let refused = Claim { blocked: Some("kind vibes".into()), ..accepted };
    assert!(refused.waits_on(&audit) && refused.mark().0 == "[!]");
}

#[test]
fn a_shell_is_held_only_while_its_audit_runs_a_batch() {
    let name = "shell-held-with-a-long-name-that-is-cut";
    let shell = crate::audit::shell_name(name);
    assert!(!shell_running(&shell));
    crate::util::locked(&RUNNING).insert(name.to_string());
    assert!(shell_running(&shell), "found by the shell's (cut) name");
    assert!(!shell_running("audit-other"));
    crate::util::locked(&RUNNING).remove(name);
    assert!(!shell_running(&shell));
}
