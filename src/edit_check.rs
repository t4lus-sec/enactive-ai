//! P4: post-edit diagnostics in the tool result.
//!
//! After a `vm_exec` command that wrote a file the harness knows how to check,
//! the harness itself runs a fast, read-only check on that file and puts the
//! findings into the same result: the result stops saying only "ok" and says
//! "ok, and here is what you just broke". A rule to always verify one's edits
//! is a claim about a reader under context pressure; a finding riding in the
//! observation can't be forgotten at turn 40 or reasoned around.
//!
//! s0ma has no file-editing tool: files are written by shell commands (a
//! heredoc, `sed -i`, `tee`, a redirect, a script). So "the file just touched"
//! is found from the command: the paths it names that the VM says were
//! modified while it ran. Checks, each skipped where its tool isn't installed:
//!
//! - Python: `compile()` (syntax, without running it or writing a .pyc), then
//!   `ruff check --select E9,F63,F7,F82` (errors that break at run time).
//! - Shell: `bash -n`, then `shellcheck -S error`.
//! - JSON, YAML, TOML: parsed. JavaScript: `node --check`. Rust: `rustfmt`
//!   parses it (output discarded).
//! - A file with no known extension is checked by its `#!` line.
//!
//! `S0MA_EDIT_CHECKS=off` turns it off.

use serde::Serialize;
use std::collections::BTreeSet;

/// Extensions the check script knows.
const CHECKED: &[&str] = &["py", "sh", "bash", "json", "yaml", "yml", "toml", "js", "mjs", "cjs", "rs"];
/// Paths looked at per command, and files reported.
const MAX_CANDIDATES: usize = 48;
pub(crate) const MAX_FILES: usize = 8;
/// What one file's findings may take in the result.
const MAX_FINDINGS_CHARS: usize = 1_500;
/// Where `vm_exec` commands run.
pub(crate) const WORKDIR: &str = "/workspace";

pub(crate) fn enabled() -> bool {
    !matches!(
        std::env::var("S0MA_EDIT_CHECKS").ok().map(|value| value.trim().to_ascii_lowercase()).as_deref(),
        Some("off" | "0" | "false" | "no")
    )
}

/// The paths a command may have written, for the VM to filter by what it
/// really modified: redirect and `tee` targets (any name), and every token
/// with a checkable extension. Then the directories it `cd`s into, which
/// relative paths may be relative to.
pub(crate) fn candidates(command: &str) -> (Vec<String>, Vec<String>) {
    let tokens = tokens(command);
    let mut paths = Vec::new();
    let mut seen = BTreeSet::new();
    let mut add = |path: &str, paths: &mut Vec<String>| {
        let path = path.trim();
        if !path.is_empty() && !path.starts_with('&') && !path.starts_with('-') && path.len() <= 1_024 && seen.insert(path.to_string()) {
            paths.push(path.to_string());
        }
    };
    // Written for certain: after `>`, `>>`, `tee [-a]`.
    for (index, token) in tokens.iter().enumerate() {
        let target = match token.as_str() {
            ">" | ">>" | ">|" => tokens.get(index + 1),
            "tee" => tokens[index + 1..].iter().find(|next| !next.starts_with('-')),
            _ => None,
        };
        if let Some(target) = target {
            add(target, &mut paths);
        }
    }
    for token in &tokens {
        if has_checked_extension(token) {
            add(token, &mut paths);
        }
    }
    paths.truncate(MAX_CANDIDATES);
    let mut dirs = Vec::new();
    for pair in tokens.windows(2) {
        if pair[0] == "cd" && !pair[1].starts_with('-') && !dirs.contains(&pair[1]) && dirs.len() < 8 {
            dirs.push(pair[1].clone());
        }
    }
    (paths, dirs)
}

fn has_checked_extension(token: &str) -> bool {
    let name = token.rsplit('/').next().unwrap_or(token);
    match name.rsplit_once('.') {
        Some((stem, extension)) => !stem.is_empty() && CHECKED.contains(&extension),
        None => false,
    }
}

/// The command split into words and operators, roughly as a shell would:
/// quotes removed, `>`/`>>`/`|`/`;`/`&` and parentheses as their own tokens.
/// A heredoc body is words too; the VM's mtime filter sorts out which of them
/// were written.
fn tokens(command: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut quote: Option<char> = None;
    let mut characters = command.chars().peekable();
    let flush = |word: &mut String, tokens: &mut Vec<String>| {
        if !word.is_empty() {
            tokens.push(std::mem::take(word));
        }
    };
    while let Some(character) = characters.next() {
        match quote {
            Some(open) if character == open => quote = None,
            Some(_) => word.push(character),
            None => match character {
                '\'' | '"' => quote = Some(character),
                '\\' => {
                    if let Some(next) = characters.next() {
                        word.push(next);
                    }
                }
                '>' => {
                    // `2>` and `&>` keep their fd; `>&2` is a dup, not a file.
                    let fd_redirect = word.chars().all(|c| c.is_ascii_digit() || c == '&') && !word.is_empty();
                    if !fd_redirect {
                        flush(&mut word, &mut tokens);
                    } else {
                        word.clear();
                    }
                    let mut operator = String::from(">");
                    while let Some(&next) = characters.peek() {
                        if next == '>' || next == '|' {
                            operator.push(next);
                            characters.next();
                        } else {
                            break;
                        }
                    }
                    if characters.peek() == Some(&'&') {
                        // `>&2`: not a file.
                        characters.next();
                        while characters.peek().is_some_and(|c| c.is_ascii_digit() || *c == '-') {
                            characters.next();
                        }
                        continue;
                    }
                    tokens.push(operator);
                }
                '<' | '|' | ';' | '&' | '(' | ')' | '`' | '$' | '{' | '}' => {
                    flush(&mut word, &mut tokens);
                    if character == '|' || character == ';' {
                        tokens.push(character.to_string());
                    }
                }
                c if c.is_whitespace() || c == '=' || c == ',' => flush(&mut word, &mut tokens),
                c => word.push(c),
            },
        }
    }
    flush(&mut word, &mut tokens);
    tokens
}


/// The read-only check script for the VM: of `paths` (relative ones tried
/// from `base` and each of `dirs`), the regular files modified in the last
/// `window` seconds, each checked by type. Each check prints a header line
/// (`@@s0ma-check@@ <path> TAB <check> TAB ok|fail`) and then its findings.
pub(crate) fn script(paths: &[String], dirs: &[String], base: &str, window: u64) -> String {
    let list = |items: &[String]| items.iter().map(|item| crate::util::shell_quote(item)).collect::<Vec<_>>().join(" ");
    format!(
        r#"set -u
base={base}
window={window}
now=$(date +%s)
max={max_files}
limit={max_chars}
t() {{ if command -v timeout >/dev/null 2>&1; then timeout 8 "$@"; else "$@"; fi; }}
emit() {{ printf '@@s0ma-check@@ %s\t%s\t%s\n' "$1" "$2" "$3"; printf '%s' "$4" | head -c "$limit"; printf '\n'; }}
py_compile() {{ t python3 -c 'import sys
p = sys.argv[1]
try:
    compile(open(p, "rb").read(), p, "exec", dont_inherit=True)
except SyntaxError as e:
    print("%s:%s:%s: %s" % (e.filename, e.lineno, e.offset, e.msg))
    if e.text:
        print("    " + e.text.rstrip())
    sys.exit(1)
except ValueError as e:
    print("%s: %s" % (p, e))
    sys.exit(1)' "$1" 2>&1; }}
check_python() {{
    if command -v python3 >/dev/null 2>&1; then
        out=$(py_compile "$1"); if [ $? -eq 0 ]; then emit "$1" "python compile" ok ""; else emit "$1" "python compile" fail "$out"; return; fi
    fi
    if command -v ruff >/dev/null 2>&1; then
        out=$(t ruff check --no-cache --quiet --select E9,F63,F7,F82 --output-format=concise "$1" 2>&1); rc=$?
        if [ $rc -eq 0 ]; then emit "$1" "ruff E9,F63,F7,F82" ok ""; elif [ $rc -eq 1 ]; then emit "$1" "ruff E9,F63,F7,F82" fail "$out"; fi
    fi
}}
check_shell() {{
    out=$(t bash -n "$1" 2>&1); if [ $? -eq 0 ]; then emit "$1" "bash -n" ok ""; else emit "$1" "bash -n" fail "$out"; return; fi
    if command -v shellcheck >/dev/null 2>&1; then
        out=$(t shellcheck -S error -f gcc "$1" 2>&1); rc=$?
        if [ $rc -eq 0 ]; then emit "$1" "shellcheck -S error" ok ""; elif [ $rc -eq 1 ]; then emit "$1" "shellcheck -S error" fail "$out"; fi
    fi
}}
check_parse() {{
    command -v python3 >/dev/null 2>&1 || return
    out=$(t python3 -c "$2" "$1" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then emit "$1" "$3" ok ""; elif [ $rc -eq 1 ]; then emit "$1" "$3" fail "$out"; fi
}}
check() {{
    case "$1" in
        *.py) check_python "$1" ;;
        *.sh|*.bash) check_shell "$1" ;;
        *.json) check_parse "$1" 'import json, sys
try: json.load(open(sys.argv[1], encoding="utf-8"))
except ValueError as e: print("%s: %s" % (sys.argv[1], e)); sys.exit(1)' "json parse" ;;
        *.yaml|*.yml) check_parse "$1" 'import sys
try: import yaml
except ImportError: sys.exit(3)
try: list(yaml.safe_load_all(open(sys.argv[1], encoding="utf-8")))
except yaml.YAMLError as e: print("%s: %s" % (sys.argv[1], e)); sys.exit(1)' "yaml parse" ;;
        *.toml) check_parse "$1" 'import sys
try: import tomllib
except ImportError: sys.exit(3)
try: tomllib.load(open(sys.argv[1], "rb"))
except tomllib.TOMLDecodeError as e: print("%s: %s" % (sys.argv[1], e)); sys.exit(1)' "toml parse" ;;
        *.js|*.mjs|*.cjs)
            if command -v node >/dev/null 2>&1; then
                out=$(t node --check "$1" 2>&1); if [ $? -eq 0 ]; then emit "$1" "node --check" ok ""; else emit "$1" "node --check" fail "$out"; fi
            fi ;;
        *.rs)
            if command -v rustfmt >/dev/null 2>&1; then
                out=$(t rustfmt --edition 2021 --emit stdout "$1" 2>&1 >/dev/null); rc=$?
                if [ $rc -eq 0 ]; then emit "$1" "rustfmt parse" ok ""; elif [ $rc -eq 1 ]; then emit "$1" "rustfmt parse" fail "$out"; fi
            fi ;;
        *)
            first=$(head -c 200 "$1" 2>/dev/null | head -n 1)
            case "$first" in
                '#!'*python*) check_python "$1" ;;
                '#!'*bash*|'#!'*/sh|'#!'*/sh' '*|'#!'*' sh'|'#!'*' sh '*|'#!'*dash*) check_shell "$1" ;;
            esac ;;
    esac
}}
seen=" "
count=0
# A file, if it is one the command just wrote: checked, and 0 returned.
try() {{
    [ -f "$1" ] || return 1
    r=$(readlink -f -- "$1" 2>/dev/null || printf '%s' "$1")
    case "$seen" in *" $r "*) return 0 ;; esac
    m=$(stat -c %Y -- "$1" 2>/dev/null) || return 1
    [ $((now - m)) -le "$window" ] || return 1
    seen="$seen$r "
    count=$((count + 1))
    [ "$count" -le "$max" ] || return 0
    check "$r"
    return 0
}}
for p in {paths}; do
    [ "$count" -lt "$max" ] || break
    case "$p" in "~"/*) p="$HOME/${{p#\~/}}" ;; esac
    case "$p" in
        /*) try "$p" ;;
        *)
            try "$base/$p" && continue
            for d in {dirs}; do
                [ -n "$d" ] || continue
                case "$d" in
                    /*) try "$d/$p" && break ;;
                    *) try "$base/$d/$p" && break ;;
                esac
            done ;;
    esac
done
exit 0
"#,
        base = crate::util::shell_quote(base),
        window = window,
        max_files = MAX_FILES,
        max_chars = MAX_FINDINGS_CHARS,
        paths = if paths.is_empty() { "''".into() } else { list(paths) },
        dirs = if dirs.is_empty() { "''".into() } else { list(dirs) },
    )
}

/// One checked file.
#[derive(Debug, Serialize, PartialEq)]
pub(crate) struct Diagnostic {
    pub path: String,
    pub ok: bool,
    /// The checks that ran, in order.
    pub checks: String,
    /// What the failing check said.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub findings: Option<String>,
}

/// The check script's output, by file.
pub(crate) fn parse(output: &str) -> Vec<Diagnostic> {
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    let mut current: Option<(String, String, bool, String)> = None;
    let finish = |current: &mut Option<(String, String, bool, String)>, diagnostics: &mut Vec<Diagnostic>| {
        let Some((path, check, ok, text)) = current.take() else { return };
        let text = text.trim().to_string();
        match diagnostics.iter_mut().find(|diagnostic| diagnostic.path == path) {
            Some(diagnostic) => {
                diagnostic.checks = format!("{}, {check}", diagnostic.checks);
                if !ok {
                    diagnostic.ok = false;
                    diagnostic.findings = Some(text);
                }
            }
            None => diagnostics.push(Diagnostic { path, ok, checks: check, findings: (!ok).then_some(text) }),
        }
    };
    for line in output.lines() {
        if let Some(header) = line.strip_prefix("@@s0ma-check@@ ") {
            finish(&mut current, &mut diagnostics);
            let mut parts = header.splitn(3, '\t');
            if let (Some(path), Some(check), Some(status)) = (parts.next(), parts.next(), parts.next()) {
                current = Some((path.to_string(), check.to_string(), status.trim() == "ok", String::new()));
            }
        } else if let Some((_, _, _, text)) = current.as_mut().filter(|(_, _, _, text)| text.chars().count() < MAX_FINDINGS_CHARS) {
            text.push_str(line);
            text.push('\n');
        }
    }
    finish(&mut current, &mut diagnostics);
    diagnostics.truncate(MAX_FILES);
    diagnostics
}

/// The seconds back from the end of a command that count as "while it ran".
pub(crate) fn window(duration_ms: u128) -> u64 {
    (duration_ms / 1_000) as u64 + 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_paths_are_found_in_the_ways_commands_write_them() {
        let (paths, dirs) = candidates(
            "cd tools && cat > manifest.sh <<'EOF'\necho hi\nEOF\nsed -i 's/a/b/' lib/x.py; echo ok | tee -a /tmp/notes 2>&1 >/dev/null; python3 run.py > out.json",
        );
        for expected in ["manifest.sh", "/tmp/notes", "lib/x.py", "run.py", "out.json"] {
            assert!(paths.iter().any(|path| path == expected), "{expected}: {paths:?}");
        }
        assert_eq!(dirs, ["tools"]);
        // Not files: fd duplications, options, and the null device stays a path the VM skips.
        let (paths, _) = candidates("make 2>&1 | tee build.log >&2");
        assert_eq!(paths, ["build.log"]);
        let (paths, _) = candidates("ls -la /etc; uname -a");
        assert!(paths.is_empty(), "{paths:?}");
        // Quotes and escapes come off.
        let (paths, _) = candidates("printf '%s' \"$x\" > \"my dir/a b.py\"");
        assert!(paths.iter().any(|path| path == "my dir/a b.py"), "{paths:?}");
    }

    #[test]
    fn output_is_grouped_by_file_with_the_failing_findings() {
        let output = "@@s0ma-check@@ /workspace/a.py\tpython compile\tok\n\n\
@@s0ma-check@@ /workspace/a.py\truff E9,F63,F7,F82\tfail\na.py:3:5: F821 Undefined name `r`\n\
@@s0ma-check@@ /workspace/m.sh\tbash -n\tfail\n/workspace/m.sh: line 22: syntax error near unexpected token `('\n\
@@s0ma-check@@ /workspace/c.json\tjson parse\tok\n\n";
        let diagnostics = parse(output);
        assert_eq!(diagnostics.len(), 3);
        assert_eq!(diagnostics[0], Diagnostic {
            path: "/workspace/a.py".into(),
            ok: false,
            checks: "python compile, ruff E9,F63,F7,F82".into(),
            findings: Some("a.py:3:5: F821 Undefined name `r`".into()),
        });
        assert!(!diagnostics[1].ok && diagnostics[1].findings.as_deref().unwrap().contains("line 22"));
        assert!(diagnostics[2].ok && diagnostics[2].findings.is_none());
    }

    /// The script itself, run by the local bash against files in a temp dir
    /// standing in for /workspace: the broken ones are caught, the good one
    /// passes, one modified long ago is left alone, and nothing is written.
    #[test]
    fn the_check_script_finds_what_was_just_broken() {
        if std::process::Command::new("bash").arg("--version").output().is_err() {
            return;
        }
        let base = std::env::temp_dir().join(format!("s0ma-edit-check-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("tools")).unwrap();
        std::fs::write(base.join("tools/manifest.sh"), "#!/bin/bash\necho start\nfor x in a b; do\n  echo $x\n").unwrap();
        std::fs::write(base.join("inspect.py"), "import json\nr = {}\nif r.get('cmd'),!= None:\n    pass\n").unwrap();
        std::fs::write(base.join("good.py"), "print('fine')\n").unwrap();
        std::fs::write(base.join("conf.json"), "{\"a\": 1,}").unwrap();
        std::fs::write(base.join("runner"), "#!/usr/bin/env bash\nif then\n").unwrap();
        std::fs::write(base.join("old.py"), "def broken(:\n").unwrap();
        let long_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(3_600);
        std::fs::File::options().write(true).open(base.join("old.py")).unwrap().set_modified(long_ago).unwrap();
        let (paths, dirs) = candidates(
            "cd tools && cat > manifest.sh <<'EOF'\n...\nEOF\ncd .. ; cat > inspect.py; cat > good.py; echo x > conf.json; cat > runner; python3 old.py",
        );
        let before: Vec<_> = std::fs::read_dir(&base).unwrap().map(|entry| entry.unwrap().file_name()).collect();
        // Whatever shell the guest runner uses: bash, or a POSIX sh.
        for shell in ["bash", "sh"] {
            let Ok(output) = std::process::Command::new(shell)
                .arg("-c")
                .arg(script(&paths, &dirs, base.to_str().unwrap(), 30))
                .output() else { continue };
            let diagnostics = parse(&String::from_utf8_lossy(&output.stdout));
            let find = |name: &str| diagnostics.iter().find(|diagnostic| diagnostic.path.ends_with(name));
            assert!(!find("manifest.sh").unwrap().ok, "{diagnostics:?}");
            let inspect = find("inspect.py").unwrap();
            assert!(!inspect.ok && inspect.findings.as_deref().unwrap().contains("inspect.py:3"), "{inspect:?}");
            assert!(find("good.py").unwrap().ok);
            assert!(!find("conf.json").unwrap().ok);
            assert!(!find("runner").unwrap().ok, "a #! file is checked by its interpreter");
            assert!(find("old.py").is_none(), "not modified while the command ran");
        }
        // Read-only: no new files (no __pycache__, no .ruff_cache).
        let after: Vec<_> = std::fs::read_dir(&base).unwrap().map(|entry| entry.unwrap().file_name()).collect();
        assert_eq!(before.len(), after.len(), "{after:?}");
        let _ = std::fs::remove_dir_all(&base);
    }
}
