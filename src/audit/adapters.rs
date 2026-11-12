//! Languages: the core never names one. Each language's tools live in an
//! adapter (`data/audit/langs/<lang>.toml`), shipped with defaults; a
//! language with none still gets `tested` checks, `model` (TLA+), and
//! patch mutants. An adapter is used only once its doctor run passes: a
//! known-good check must hold and a known-bad one must fail at every level
//! it claims, so a toolchain that cannot fail is caught first.

use crate::audit::{self, AdapterState, Level};

use std::path::PathBuf;

/// The shipped defaults: the most common tool at each level, where one
/// exists.
pub const SHIPPED: &[(&str, &str)] = &[
    (
        "rust",
        r#"lang      = "rust"
detect    = ["Cargo.toml", "*.rs"]
setup     = ["cargo install cargo-mutants"]
test      = "cargo test -q {test}"
property  = { lib = "proptest", run = "cargo test -q {test}" }
bounded   = { tool = "kani", run = "cargo kani --harness {test}" }
mutate    = { tool = "cargo-mutants", run = "cargo mutants --file {files} --no-shuffle" }
coverage  = "source <(cargo llvm-cov show-env --export-prefix) && cargo llvm-cov clean --workspace && {run}; cargo llvm-cov report --lcov --output-path $COV"
proof     = { tool = "verus" }
repro_dir = "tests/audit/"
ceiling   = "bounded"
good      = "exit 0"
bad       = "exit 1"
"#,
    ),
    (
        "python",
        r#"lang      = "python"
detect    = ["pyproject.toml", "setup.py", "*.py"]
setup     = ["pip install --break-system-packages hypothesis crosshair-tool mutmut"]
test      = "python3 -m pytest -q {test}"
property  = { lib = "hypothesis", run = "python3 -m pytest -q {test}" }
bounded   = { tool = "crosshair", run = "crosshair check --analysis_kind=asserts {target}" }
mutate    = { tool = "mutmut", run = "mutmut run --paths-to-mutate {files}" }
coverage  = "python3 -m coverage run -m pytest -q {test}; python3 -m coverage lcov -o $COV"
repro_dir = "tests/audit/"
ceiling   = "bounded"
good      = "python3 -c 'assert 1 + 1 == 2'"
bad       = "python3 -c 'import sys; sys.exit(1)'"
"#,
    ),
    (
        "typescript",
        r#"lang      = "typescript"
detect    = ["tsconfig.json", "*.ts", "*.tsx", "*.js"]
setup     = ["npm install --no-save fast-check @stryker-mutator/core"]
test      = "npx --no-install vitest run {test}"
property  = { lib = "fast-check", run = "npx --no-install vitest run {test}" }
mutate    = { tool = "stryker", run = "npx --no-install stryker run --mutate {files}" }
coverage  = "d=$(mktemp -d) && npx --no-install c8 --reporter=lcov --report-dir $d {run}; mv $d/lcov.info $COV"
repro_dir = "tests/audit/"
ceiling   = "property"
good      = "node -e 'process.exit(0)'"
bad       = "node -e 'process.exit(1)'"
"#,
    ),
    (
        "go",
        r#"lang      = "go"
detect    = ["go.mod", "*.go"]
setup     = ["go install github.com/go-gremlins/gremlins/cmd/gremlins@latest"]
test      = "go test -run {test} ./..."
property  = { lib = "testing/quick, rapid, native fuzzing", run = "go test -run {test} ./..." }
mutate    = { tool = "gremlins", run = "gremlins unleash" }
proof     = { tool = "gobra" }
repro_dir = "audit_test/"
ceiling   = "property"
good      = "go version >/dev/null"
bad       = "false"
"#,
    ),
    (
        "c",
        r#"lang      = "c"
detect    = ["Makefile", "CMakeLists.txt", "*.c", "*.h", "*.cpp", "*.cc", "*.cu"]
setup     = []
test      = "{test}"
property  = { lib = "rapidcheck, libFuzzer", run = "{test}" }
bounded   = { tool = "cbmc", run = "cbmc {target}" }
mutate    = { tool = "mull" }
proof     = { tool = "frama-c wp" }
repro_dir = "tests/audit/"
ceiling   = "bounded"
good      = "true"
bad       = "false"
"#,
    ),
    (
        "java",
        r#"lang      = "java"
detect    = ["pom.xml", "build.gradle", "*.java", "*.kt"]
setup     = []
test      = "mvn -q test -Dtest={test}"
property  = { lib = "jqwik", run = "mvn -q test -Dtest={test}" }
bounded   = { tool = "jbmc", run = "jbmc {target}" }
mutate    = { tool = "pit", run = "mvn -q org.pitest:pitest-maven:mutationCoverage" }
repro_dir = "src/test/java/audit/"
ceiling   = "bounded"
good      = "true"
bad       = "false"
"#,
    ),
];

/// File extensions to a language, for the ceiling preview.
const EXTENSIONS: &[(&str, &str)] = &[
    ("rs", "rust"),
    ("py", "python"),
    ("ts", "typescript"),
    ("tsx", "typescript"),
    ("js", "typescript"),
    ("go", "go"),
    ("c", "c"),
    ("h", "c"),
    ("cc", "c"),
    ("cpp", "c"),
    ("hpp", "c"),
    ("cu", "c"),
    ("java", "java"),
    ("kt", "java"),
    ("sh", "sh"),
    ("lua", "lua"),
    ("rb", "ruby"),
    ("hs", "haskell"),
    ("ex", "elixir"),
    ("cs", "csharp"),
];

pub fn langs_dir() -> Option<PathBuf> {
    audit::data_dir().map(|dir| dir.join("audit/langs"))
}

/// An adapter's text: the operator's file if there is one, else the shipped
/// default (written out for him to change).
pub fn adapter(lang: &str) -> Option<String> {
    if let Some(dir) = langs_dir() {
        let path = dir.join(format!("{lang}.toml"));
        if let Ok(text) = std::fs::read_to_string(&path) {
            return Some(text);
        }
        if let Some((_, text)) = SHIPPED.iter().find(|(name, _)| *name == lang) {
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(&path, text);
            return Some(text.to_string());
        }
    }
    SHIPPED.iter().find(|(name, _)| *name == lang).map(|(_, text)| text.to_string())
}

/// A language's ceiling: the highest level its adapter reaches, once its
/// doctor passed; `tested` without one.
pub fn ceiling(lang: &str) -> Level {
    let Some(text) = adapter(lang) else { return Level::Tested };
    let parsed = audit::toml_flat(&text).unwrap_or_default();
    parsed.get("ceiling").and_then(|v| Level::parse(&v.text())).unwrap_or(Level::Tested)
}

/// The ceiling over an audit's languages: the lowest of them (so a claim
/// stuck there reads as "as far as this language goes").
pub fn ceiling_for(languages: &[String]) -> Level {
    languages
        .iter()
        .filter_map(|entry| entry.split_whitespace().next())
        .map(ceiling)
        .min()
        .unwrap_or(Level::Tested)
}

/// The ceiling preview: the languages found in scope, how much of the code
/// each is, and the ceiling each reaches: `rust 61% ceiling bounded`.
pub fn preview(work: &str, scope: &[String]) -> std::result::Result<Vec<String>, String> {
    let paths = crate::audit::scope::paths_for(work, scope);
    let script = format!(
        "cd {} || exit 3; find {} -type f \\( -path '*/.git/*' -o -path '*/target/*' -o -path '*/node_modules/*' -o -path '*/checks/*' \\) -prune -o -type f -print 2>/dev/null \
         | sed -n 's/.*\\.\\([A-Za-z0-9]*\\)$/\\1/p' | sort | uniq -c",
        crate::util::shell_quote(work),
        if paths.is_empty() { ".".to_string() } else { paths.iter().map(|p| crate::util::shell_quote(p)).collect::<Vec<_>>().join(" ") }
    );
    let ran = crate::vm::netboxes::machine(&script, 60)?;
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for line in ran.stdout.lines() {
        let mut parts = line.split_whitespace();
        let (Some(n), Some(ext)) = (parts.next().and_then(|n| n.parse::<usize>().ok()), parts.next()) else { continue };
        if let Some((_, lang)) = EXTENSIONS.iter().find(|(e, _)| *e == ext) {
            *counts.entry(lang).or_default() += n;
        }
    }
    let total: usize = counts.values().sum::<usize>().max(1);
    let mut out: Vec<(usize, String)> = counts
        .into_iter()
        .map(|(lang, n)| {
            let state = audit::store().and_then(|s| s.adapter_state(lang).ok().flatten());
            let note = match (adapter(lang).is_some(), state.as_ref().is_some_and(|s| s.doctor_passed)) {
                (true, true) => format!("ceiling {}", ceiling(lang).label()),
                (true, false) => format!("ceiling {} (doctor not run)", ceiling(lang).label()),
                (false, _) => "tested (no adapter)".to_string(),
            };
            (n, format!("{lang} {}% {note}", n * 100 / total))
        })
        .collect();
    out.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    Ok(out.into_iter().map(|(_, line)| line).collect())
}

/// The agent's `audit_adapter_propose`: an adapter for a language without one,
/// waiting for the operator (`[!] adapter`): its setup installs things and
/// its templates are commands.
pub fn propose(lang: &str, toml: &str, why: &str) -> std::result::Result<String, String> {
    if !lang.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') || lang.is_empty() {
        return Err("lang: lowercase letters".into());
    }
    audit::toml_flat(toml)?;
    let store = audit::store().ok_or("not ready")?;
    let mut state = store.adapter_state(lang).map_err(|e| e.to_string())?.unwrap_or_default();
    state.proposed = Some(toml.to_string());
    state.proposed_why = why.to_string();
    store.set_adapter_state(lang, &state).map_err(|e| e.to_string())?;
    Ok(format!("{{\"ok\": true, \"note\": \"The adapter for {lang} waits for the operator ([!] adapter in the audit tab).\"}}"))
}

/// Adapters proposed and waiting.
pub fn proposed() -> Vec<String> {
    let Some(store) = audit::store() else { return Vec::new() };
    let mut out = Vec::new();
    for lang in ["rust", "python", "typescript", "go", "c", "java", "sh", "lua", "ruby", "haskell", "elixir", "csharp"] {
        if store.adapter_state(lang).ok().flatten().is_some_and(|s| s.proposed.is_some()) {
            out.push(lang.to_string());
        }
    }
    out
}

/// The operator approves a proposed adapter: written, and its doctor run.
pub fn approve(lang: &str) -> std::result::Result<String, String> {
    let store = audit::store().ok_or("not ready")?;
    let mut state = store.adapter_state(lang).map_err(|e| e.to_string())?.unwrap_or_default();
    let toml = state.proposed.take().ok_or(format!("no adapter for {lang} is waiting"))?;
    let dir = langs_dir().ok_or("not ready")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{lang}.toml")), &toml).map_err(|e| e.to_string())?;
    state.doctor_passed = false;
    store.set_adapter_state(lang, &state).map_err(|e| e.to_string())?;
    doctor_in_background(lang.to_string())
}

/// `audit-run --doctor <lang>`: the adapter's setup, then its known-good
/// check must hold and its known-bad check must fail, in the doctor's own
/// shell. Until it passes, the adapter isn't used.
pub fn doctor_in_background(lang: String) -> std::result::Result<String, String> {
    let text = adapter(&lang).ok_or(format!("no adapter for {lang}"))?;
    let parsed = audit::toml_flat(&text)?;
    let setup = parsed.get("setup").map(|v| v.list()).unwrap_or_default();
    let good = parsed.get("good").map(|v| v.text()).unwrap_or_else(|| "true".into());
    let bad = parsed.get("bad").map(|v| v.text()).unwrap_or_else(|| "false".into());
    let hash = format!("{:x}", crate::util::fnv1a64(text.as_bytes()));
    let detail = format!("{}\n{good}\n{bad}", setup.join("\n"));
    let thread_lang = lang.clone();
    let _ = std::thread::Builder::new().name("audit-doctor".into()).spawn(move || {
        let lang = thread_lang;
        let outcome = (|| -> std::result::Result<String, String> {
            crate::approval::ask_as(&format!("audit doctor {lang}"), &format!("install and check the {lang} adapter's tools"), &detail, !setup.is_empty())?;
            for step in &setup {
                let ran = crate::vm::netboxes::machine(step, 1800)?;
                if ran.exit_code != Some(0) {
                    return Err(format!("setup failed: {step}: {}", ran.stderr.lines().last().unwrap_or("")));
                }
            }
            let good_ran = crate::vm::netboxes::machine(&good, 300)?;
            let bad_ran = crate::vm::netboxes::machine(&bad, 300)?;
            match (good_ran.exit_code, bad_ran.exit_code) {
                (Some(0), Some(code)) if code != 0 => Ok("the known-good check holds and the known-bad one fails".into()),
                (Some(0), _) => Err("its known-bad check passed: a toolchain that can't fail is refused".into()),
                _ => Err("its known-good check failed".into()),
            }
        })();
        if let Some(store) = audit::store() {
            let mut state = store.adapter_state(&lang).ok().flatten().unwrap_or_default();
            state.doctor_at = Some(audit::now());
            state.file_hash = hash;
            match &outcome {
                Ok(note) => {
                    state.doctor_passed = true;
                    state.doctor_note = note.clone();
                    crate::terminal::say_status(&format!("[audit] the {lang} adapter passed its doctor run"));
                }
                Err(error) => {
                    state.doctor_passed = false;
                    state.doctor_note = error.clone();
                    crate::terminal::warn_status(&format!("[audit] the {lang} adapter failed its doctor run: {error}"));
                }
            }
            let _ = store.set_adapter_state(&lang, &state);
        }
    });
    Ok(format!("[audit] doctor run for {lang} started"))
}

/// Whether checks may use `lang`'s adapter tools: `any` and languages
/// without an adapter (generic `test`/`command`/`differential` checks) may;
/// one with an adapter only once its doctor run passed on its file as it is.
/// A changed adapter has its doctor run again.
pub fn usable(lang: &str, kind: &str) -> std::result::Result<(), String> {
    if lang == "any" || lang.is_empty() || matches!(kind, "test" | "command" | "differential" | "tla") || adapter(lang).is_none() {
        return Ok(());
    }
    let state = audit::store().and_then(|s| s.adapter_state(lang).ok().flatten()).unwrap_or_default();
    if changed_since_doctor(lang) {
        let _ = doctor_in_background(lang.to_string());
        return Err(format!("the {lang} adapter changed since its doctor run: it runs again now (an adapter is used only once it passes)"));
    }
    if !state.doctor_passed {
        return Err(format!("the {lang} adapter has not passed its doctor run: /audit doctor {lang} (an adapter is used only once it passes)"));
    }
    Ok(())
}

/// Whether an adapter changed since its doctor run (it runs again).
pub fn changed_since_doctor(lang: &str) -> bool {
    let (Some(text), Some(store)) = (adapter(lang), audit::store()) else { return false };
    let hash = format!("{:x}", crate::util::fnv1a64(text.as_bytes()));
    store.adapter_state(lang).ok().flatten().is_some_and(|s: AdapterState| s.doctor_at.is_some() && s.file_hash != hash)
}
