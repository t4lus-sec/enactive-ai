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
    submit_check(&store, "share-slice", &c.id, "checks/c1", "a1-3").unwrap();
    let after = store.claim("share-slice", &c.id).unwrap().unwrap();
    assert!(after.weaker.as_deref().is_some_and(|w| w.contains("m1.patch")), "{:?}", after.weaker);
    assert_eq!(after.verdict, Verdict::Holds, "the old verdict stands meanwhile");
    assert_eq!(store.current_check("share-slice", &c.id).unwrap().unwrap().version, first.version, "the old check stays current");
    // One that still catches it replaces it.
    let _w = world(World { toml: "claim = \"c1\"\nkind = \"property\"\nrun = \"./CHECK\"".into(), mutants: vec!["m1.patch"], caught: vec!["m1.patch"], ..World::default() });
    submit_check(&store, "share-slice", &c.id, "checks/c1", "a1-3").unwrap();
    assert!(store.current_check("share-slice", &c.id).unwrap().unwrap().version > first.version);
}

#[test]
fn a_clause_without_an_anchor_is_refused() {
    let (store, _) = fresh();
    running(&store, None, Mode::Assure, Gate::None, Post::Auto);
    let c = claim(&store, "share-slice", "Redaction never changes the line count; and it keeps every byte outside the ranges", Review::Accepted);
    let _w = world(World { toml: "kind = \"test\"\nrun = \"./CHECK\"".into(), anchors: "1".into(), ..World::default() });
    let said = submit_check(&store, "share-slice", &c.id, "checks/c1", "a1-3");
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
    let said = submit_check(&store, "share-slice", &c.id, "checks/c1", "a1-2");
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

