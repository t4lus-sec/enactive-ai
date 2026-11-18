//! Back-translation's reading: a check's files shown whole, with what is
//! known about each (size, lines, hash), what they define, and a way to
//! tell afterwards whether the reviewer read every one to its end.

/// Files over this are shown in part, and the review says so.
const FILE_BYTES: usize = 80_000;
/// All files together.
const TOTAL_BYTES: usize = 240_000;
/// Lines shown of a file cut short.
const CUT_LINES: usize = 600;
/// Files shown at most.
const FILES: usize = 40;

/// The shell script that lists `dir`'s files (not its mutants, coverage or
/// other checks' folders) for `parse`.
pub(super) fn files_script(work: &str, dir: &str) -> String {
    format!(
        r#"cd {work}/{dir} || exit 3
n=0; total=0
find . -type f ! -path './mutants/*' ! -path './k[0-9]*/*' ! -name coverage.lines ! -path '*/__pycache__/*' ! -path '*/.git/*' -print0 | sort -z | while IFS= read -r -d '' f; do
  f=${{f#./}}; n=$((n+1))
  if [ $n -gt {FILES} ]; then echo "__MORE__ $f"; continue; fi
  bytes=$(wc -c <"$f")
  if [ "$bytes" -gt 0 ] && ! grep -Iq . "$f"; then echo "__BINARY__ $bytes $f"; continue; fi
  lines=$(wc -l <"$f"); sum=$(sha256sum "$f" | cut -c1-12)
  echo "__FILE__ $bytes $lines $sum $f"
  if [ "$bytes" -gt {FILE_BYTES} ] || [ $((total+bytes)) -gt {TOTAL_BYTES} ]; then
    head -n {CUT_LINES} "$f" | awk '{{printf "%5d | %s\n", NR, $0}}'; echo "__CUT__"
    total=$((total+{FILE_BYTES}/4))
  else
    awk '{{printf "%5d | %s\n", NR, $0}}' "$f"
    total=$((total+bytes))
  fi
  printf '__LAST__ %s\t%s\n' "$f" "$(grep -v '^[[:space:]]*$' "$f" | tail -n 1)"
done"#,
        work = super::q(work),
        dir = dir,
    )
}

/// What the reviewer is shown, and what its answer must quote back.
#[derive(Debug, Default, PartialEq)]
pub(super) struct Shown {
    pub text: String,
    /// Each file shown, with its last line that isn't blank.
    pub expect: Vec<(String, String)>,
    /// Files not shown whole, and why.
    pub partial: Vec<String>,
    /// Names the files define, `name (file:line)`.
    pub defined: Vec<String>,
}

pub(super) fn parse(stdout: &str) -> Shown {
    let mut shown = Shown::default();
    let mut file = String::new();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("__FILE__ ") {
            let mut parts = rest.splitn(4, ' ');
            let (bytes, lines, sum) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next().unwrap_or(""));
            file = parts.next().unwrap_or("").to_string();
            shown.text.push_str(&format!("==== {file} · {bytes} bytes · {lines} lines · sha256 {sum}\n"));
        } else if line == "__CUT__" {
            shown.text.push_str(&format!("[... {file} goes on: only its first {CUT_LINES} lines are shown ...]\n"));
            shown.partial.push(format!("{file} was too long to show whole"));
        } else if let Some(rest) = line.strip_prefix("__LAST__ ") {
            let (name, last) = rest.split_once('\t').unwrap_or((rest, ""));
            shown.expect.push((name.to_string(), last.trim().to_string()));
            shown.text.push('\n');
        } else if let Some(rest) = line.strip_prefix("__MORE__ ") {
            shown.partial.push(format!("{rest} was not shown (over {FILES} files)"));
        } else if let Some(rest) = line.strip_prefix("__BINARY__ ") {
            let (bytes, name) = rest.split_once(' ').unwrap_or(("?", rest));
            shown.text.push_str(&format!("==== {name} · binary, {bytes} bytes, not shown\n\n"));
        } else {
            if let Some((number, code)) = line.split_once(" | ")
                && let Ok(number) = number.trim().parse::<usize>()
                && let Some(name) = definition(code)
                && shown.defined.len() < 150
            {
                shown.defined.push(format!("{name} ({file}:{number})"));
            }
            shown.text.push_str(line);
            shown.text.push('\n');
        }
    }
    shown
}

/// The name a line defines, in the languages checks are written in.
fn definition(code: &str) -> Option<String> {
    let trimmed = code.trim_start();
    let mut rest = trimmed;
    for prefix in ["pub(crate) ", "pub(super) ", "pub ", "export ", "async ", "static ", "unsafe ", "const "] {
        rest = rest.strip_prefix(prefix).unwrap_or(rest);
    }
    for keyword in ["def ", "fn ", "function ", "func ", "class ", "struct ", "enum ", "trait ", "type "] {
        if let Some(after) = rest.strip_prefix(keyword) {
            let name: String = after.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            return (!name.is_empty()).then_some(name);
        }
    }
    // TLA+: `Name == …` (or `Name(x) == …`) at the start of a line.
    if trimmed.len() == code.len() {
        let name: String = code.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        let after = code[name.len()..].trim_start();
        let after = if after.starts_with('(') { after.split_once(')').map(|(_, a)| a.trim_start()).unwrap_or("") } else { after };
        if !name.is_empty() && after.starts_with("==") && !after.starts_with("===") {
            return Some(name);
        }
    }
    None
}

/// The task's own part: the files and how to answer.
pub(super) fn task(shown: &Shown) -> String {
    let defined = if shown.defined.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nDefined in these files: {}. Anything else they call comes from the code under check, a library or the language.",
            shown.defined.join(", ")
        )
    };
    let partial = if shown.partial.is_empty() {
        String::new()
    } else {
        format!("\n\nNot all of it is shown: {}. Say what that leaves you unable to tell.", shown.partial.join("; "))
    };
    format!(
        "Read this check (a test, property, model or proof harness) and say, in two or three plain English sentences, exactly \
         what it guarantees when it passes: what it ranges over, what it asserts, and what it leaves out. You are not told what \
         it was meant to check, and you need no tools. Every file is shown whole unless marked otherwise, with line numbers.\n\
         After the sentences, for each file shown, add one line `SEEN <file>: <its last line that isn't blank, copied exactly>` \
         so the harness knows each was read to its end.{defined}{partial}\n\n{}",
        shown.text
    )
}

/// The reviewer's answer, as kept on the check: its sentences, and whether
/// it read everything (each file's last line quoted back).
pub(super) fn settle(result: &str, expect: &[(String, String)], partial: &[String]) -> String {
    let mut sentences = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();
    for line in result.lines() {
        let bare = line.trim().trim_start_matches(['-', '*', ' ', '`']);
        match bare.strip_prefix("SEEN ").and_then(|rest| rest.split_once(':')) {
            Some((file, quote)) => seen.push((file.trim().trim_matches('`').to_string(), quote.to_string())),
            None => sentences.push(line),
        }
    }
    let mut gaps: Vec<String> = partial.to_vec();
    for (file, last) in expect {
        match seen.iter().find(|(f, _)| f == file) {
            Some((_, quote)) if same(quote, last) => {}
            Some(_) => gaps.push(format!("{file}'s last line was not quoted as it is")),
            None => gaps.push(format!("{file} was not confirmed read")),
        }
    }
    let body = sentences.join("\n").trim().to_string();
    if gaps.is_empty() {
        format!("{body}\n[+] read whole: {} file{}, each to its last line", expect.len(), if expect.len() == 1 { "" } else { "s" })
    } else {
        format!("[~] partial review: {}\n{body}", gaps.join("; "))
    }
}

/// Two quotes of a line agree, whatever the spacing, quoting or line number.
fn same(quote: &str, last: &str) -> bool {
    let norm = |text: &str| {
        let text = text.trim().trim_matches(['`', '"', '\'']).trim();
        let text = match text.split_once(" | ") {
            Some((number, rest)) if number.trim().parse::<usize>().is_ok() => rest,
            _ => text,
        };
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    };
    norm(quote) == norm(last)
}
