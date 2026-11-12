//! Auditing a slice of a large codebase (assume-guarantee): what the slice
//! is (`path:`, `sym:`, `diff:`, `doc:`), its map (inside, calls out, called
//! from), the assumptions it rests on, and keeping it honest: an assumption
//! is used and never checked here, one that fails makes the claims resting on
//! it unsound, and another audit with its code in scope can discharge it.

use crate::audit::{self, AuditStore, Claim, Review, Verdict, Who};

use serde::{Deserialize, Serialize};

type Reply = std::result::Result<String, String>;

fn err<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

/// The slice map, worked out when the scope is set or changes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Slice {
    pub inside_files: Vec<String>,
    pub inside_symbols: Vec<String>,
    /// What inside code calls outside: `symbol (file)`.
    pub calls_out: Vec<String>,
    /// Outside code that calls in: `symbol ← file`.
    pub called_from: Vec<String>,
    /// Found by name matching, not a language server: a callback or a trait
    /// object can hide a call.
    pub approximate: bool,
    /// `doc:` specs, mapped by the agent, not yet confirmed by the operator.
    pub unconfirmed_docs: Vec<String>,
}

impl Slice {
    /// The head's line: `38 fns · calls out 14 · called from 6`.
    pub fn line(&self, assumed: usize) -> String {
        format!(
            "{} fns · calls out {} · called from {} · {assumed} assumed{}",
            self.inside_symbols.len(),
            self.calls_out.len(),
            self.called_from.len(),
            if self.approximate { " · approximate" } else { "" }
        )
    }
}

/// The path patterns a scope names directly (`path:` specs), for the harvest
/// scan; the whole target when it names none.
pub fn paths_for(_work: &str, scope: &[String]) -> Vec<String> {
    scope.iter().filter_map(|spec| spec.strip_prefix("path:")).map(str::to_string).collect()
}

pub fn valid_spec(spec: &str) -> std::result::Result<(), String> {
    match spec.split_once(':') {
        Some(("path" | "sym" | "diff" | "doc", rest)) if !rest.trim().is_empty() => Ok(()),
        _ => Err(format!("{spec}: path:<glob>, sym:<name>, diff:<range> or doc:<file>")),
    }
}

/// The slice map, on the agent's machine: inside files from the specs, the
/// symbols they define, and the calls across the boundary, by name matching.
pub fn map(work: &str, scope: &[String]) -> std::result::Result<Slice, String> {
    for spec in scope {
        valid_spec(spec)?;
    }
    let specs = serde_json::to_string(scope).map_err(err)?;
    let program = r#"
import json, os, re, subprocess, sys, fnmatch
work, specs = sys.argv[1], json.loads(sys.argv[2])
os.chdir(work)
skip = {'.git', 'target', 'node_modules', 'checks', '__pycache__', 'dist', 'build'}
exts = ('.rs', '.py', '.go', '.ts', '.tsx', '.js', '.c', '.h', '.cc', '.cpp', '.hpp', '.java', '.kt', '.cs', '.rb', '.cu')
files = []
for root, dirs, names in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in skip]
    for n in names:
        if n.endswith(exts):
            files.append(os.path.relpath(os.path.join(root, n)))
define = re.compile(r'\b(?:fn|def|func|function|class|struct|enum|trait|interface)\s+([A-Za-z_]\w*)')
call = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
def text(f):
    try:
        return open(f, errors='replace').read()
    except Exception:
        return ''
defs = {}
for f in files:
    for name in define.findall(text(f)):
        defs.setdefault(name, set()).add(f)
inside, docs = set(), []
for spec in specs:
    kind, _, rest = spec.partition(':')
    if kind == 'path':
        inside |= {f for f in files if fnmatch.fnmatch(f, rest) or f == rest or f.startswith(rest.rstrip('/') + '/')}
    elif kind == 'sym':
        last = rest.split('::')[-1]
        inside |= defs.get(last, set())
    elif kind == 'diff':
        out = subprocess.run(['git', 'diff', '--name-only', rest], capture_output=True, text=True).stdout.split()
        inside |= {f for f in out if f in files}
    elif kind == 'doc':
        body = text(rest)
        for f in files:
            if os.path.basename(f) in body:
                inside.add(f)
        docs.append(rest)
syms = sorted({n for f in inside for n in define.findall(text(f))})
calls_out, called_from = set(), set()
for f in inside:
    for name in set(call.findall(text(f))):
        where = defs.get(name, set())
        if where and not (where & inside) and name not in syms:
            calls_out.add('%s (%s)' % (name, sorted(where)[0]))
for f in files:
    if f in inside:
        continue
    body = text(f)
    for name in syms:
        if len(name) > 3 and re.search(r'\b' + re.escape(name) + r'\s*\(', body):
            called_from.add('%s <- %s' % (name, f))
print(json.dumps({'inside_files': sorted(inside), 'inside_symbols': syms, 'calls_out': sorted(calls_out)[:200],
                  'called_from': sorted(called_from)[:200], 'approximate': True, 'unconfirmed_docs': docs}))
"#;
    let script = format!(
        "command -v python3 >/dev/null || {{ echo 'the slice map needs python3 on the machine' >&2; exit 3; }}; python3 -c {} {} {}",
        crate::util::shell_quote(program),
        crate::util::shell_quote(work),
        crate::util::shell_quote(&specs)
    );
    let ran = crate::vm::netboxes::machine(&script, 300)?;
    if ran.exit_code != Some(0) {
        return Err(format!("the slice map: {}", ran.stderr.trim()));
    }
    serde_json::from_str(ran.stdout.trim()).map_err(|e| format!("the slice map: {e}"))
}

/// `/audit scope <name> add|remove <spec>`: the slice map again, and an
/// assumption proposed for each new call out.
pub fn change_scope(store: &AuditStore, name: &str, add: bool, spec: &str) -> Reply {
    valid_spec(spec)?;
    let audit = store
        .change(name, |a| {
            if add {
                if !a.scope.iter().any(|s| s == spec) {
                    a.scope.push(spec.to_string());
                }
            } else {
                a.scope.retain(|s| s != spec);
            }
            Ok(a.clone())
        })
        .map_err(err)??;
    store.log(name, Who::Operator, &format!("{} scope {spec}", if add { "added" } else { "removed" }), "").map_err(err)?;
    if audit.work.is_empty() {
        return Ok(format!("[audit] {name}: scope {}", audit.scope.join(" ")));
    }
    let name = name.to_string();
    let _ = std::thread::Builder::new().name("audit-scope".into()).spawn(move || {
        let Some(store) = audit::store() else { return };
        let Ok(Some(audit)) = store.audit(&name) else { return };
        match map(&audit.work, &audit.scope) {
            Ok(slice) => {
                let before: Vec<String> = audit.slice.as_ref().map(|s| s.calls_out.clone()).unwrap_or_default();
                let new_calls: Vec<String> = slice.calls_out.iter().filter(|c| !before.contains(c)).cloned().collect();
                let _ = store.change(&name, |a| {
                    a.slice = Some(slice.clone());
                    Ok(())
                });
                propose_assumptions(&store, &name, &new_calls);
            }
            Err(error) => crate::terminal::warn_status(&format!("[audit] {name}: {error}")),
        }
    });
    Ok(format!("[audit] {}: scope {} — the slice map is being worked out again", audit.name, audit.scope.join(" ")))
}

/// For each call out, an assumption proposed (`[?]`), for the agent to word and
/// the operator to triage. A symbol another live audit has in scope is
/// proposed there as a claim, which discharges it.
pub fn propose_assumptions(store: &AuditStore, name: &str, calls: &[String]) {
    let existing: Vec<String> = store.claims(name).unwrap_or_default().into_iter().filter(Claim::is_assumption).map(|c| c.source).collect();
    for call in calls.iter().take(40) {
        let symbol = call.split(' ').next().unwrap_or(call);
        let source = format!("sym:{symbol}");
        if existing.contains(&source) {
            continue;
        }
        let claim = Claim {
            text: format!("assumes: {symbol} behaves as the slice's callers rely on (to be worded)"),
            author: "harness".into(),
            source,
            kind: "assumes".into(),
            risk: 5,
            review: Review::Proposed,
            level: audit::Level::Assumed,
            ..Claim::default()
        };
        if let Ok(claim) = store.add_claim(name, claim) {
            offer_for_discharge(store, name, &claim);
        }
    }
}

/// An assumption whose symbol another live audit has in scope is proposed
/// there as a claim, and its verdict shows here.
pub fn offer_for_discharge(store: &AuditStore, name: &str, assumption: &Claim) {
    let Some(symbol) = assumption.source.strip_prefix("sym:") else { return };
    for other in store.audits().unwrap_or_default() {
        if other.name == name || !other.status.live() {
            continue;
        }
        let has = other.slice.as_ref().is_some_and(|s| s.inside_symbols.iter().any(|x| x == symbol.rsplit("::").next().unwrap_or(symbol)));
        if !has {
            continue;
        }
        let claim = Claim {
            text: assumption.text.trim_start_matches("assumes: ").to_string(),
            author: "s0ma".into(),
            source: format!("assumption:{name}/{}", assumption.id),
            kind: "invariant".into(),
            risk: assumption.risk.max(5),
            review: Review::Proposed,
            ..Claim::default()
        };
        if let Ok(there) = store.add_claim(&other.name, claim) {
            let _ = store.change_claim(name, &assumption.id, |c| {
                c.discharged_by = Some((other.name.clone(), there.id.clone()));
                Ok(())
            });
            let _ = store.log(name, Who::Harness, &format!("{} is proposed for discharge in {} as {}", assumption.id, other.name, there.id), "");
        }
        break;
    }
}

/// A claim of `audit` just got a verdict: an assumption elsewhere that it
/// discharges takes it (a failure makes the claims resting on that
/// assumption unsound; a hold clears an unsoundness it caused and re-checks).
pub fn discharge(store: &AuditStore, audit_name: &str, claim_id: &str) {
    let Ok(Some(claim)) = store.claim(audit_name, claim_id) else { return };
    for other in store.audits().unwrap_or_default() {
        for assumption in store.claims(&other.name).unwrap_or_default() {
            if assumption.discharged_by.as_ref() != Some(&(audit_name.to_string(), claim_id.to_string())) {
                continue;
            }
            match claim.verdict {
                Verdict::Fails => mark_unsound(store, &other.name, &assumption.id),
                Verdict::Holds => clear_unsound(store, &other.name, &assumption.id),
                _ => {}
            }
        }
    }
}

/// Every claim whose check used `assumption`: `[?] unsound`, until the
/// assumption is fixed or replaced.
pub fn mark_unsound(store: &AuditStore, audit_name: &str, assumption: &str) {
    for claim in store.claims(audit_name).unwrap_or_default() {
        if claim.rests_on.iter().any(|a| a == assumption) {
            let _ = store.change_claim(audit_name, &claim.id, |c| {
                c.unsound = Some(format!("{assumption} failed"));
                Ok(())
            });
        }
    }
    let _ = store.log(audit_name, Who::Harness, &format!("{assumption} failed: the claims resting on it are unsound"), "");
}

fn clear_unsound(store: &AuditStore, audit_name: &str, assumption: &str) {
    for claim in store.claims(audit_name).unwrap_or_default() {
        if claim.unsound.as_deref() == Some(&format!("{assumption} failed")) {
            let _ = store.change_claim(audit_name, &claim.id, |c| {
                c.unsound = None;
                c.verdict = Verdict::Unchecked;
                Ok(())
            });
        }
    }
    crate::audit::run::wake();
}

/// What an assumption's status is, as its row shows it: `assumed`, or
/// `assumed · holds in archive-audit (bounded)`.
pub fn assumption_status(store: &AuditStore, claim: &Claim) -> String {
    match &claim.discharged_by {
        Some((audit, id)) => match store.claim(audit, id).ok().flatten() {
            Some(other) if other.verdict == Verdict::Holds => format!("assumed · holds in {audit} ({})", other.level.label()),
            Some(other) if other.verdict == Verdict::Fails => format!("assumed · FAILS in {audit}/{id}"),
            Some(_) => format!("assumed · proposed in {audit} as {id}"),
            None => "assumed".into(),
        },
        None => "assumed".into(),
    }
}

/// A claim whose quoted source no longer appears where it came from: what
/// changed, if it is stale.
pub fn stale(audit: &audit::Audit, claim: &Claim) -> Option<String> {
    let quote = claim.source.split('"').nth(1)?.trim();
    if quote.len() < 12 {
        return None;
    }
    let script = format!("cd {} && grep -rqF -- {} . --exclude-dir=.git --exclude-dir=target", crate::util::shell_quote(&audit.work), crate::util::shell_quote(quote));
    match crate::vm::netboxes::machine(&script, 60) {
        Ok(ran) if ran.exit_code == Some(1) => Some(format!("its source quote is gone: \"{}\"", crate::util::clip(quote, 80))),
        _ => None,
    }
}
