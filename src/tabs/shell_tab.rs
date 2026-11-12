//! A shell's tab: the terminal, compacted to the tab's width.

use super::*;

/// A shell in the VM as it is now (or its history, scrolled back), its footer,
/// where its cursor is among the rows, and whether the history asked for is
/// still being read.
pub(super) fn shell_body(number: usize, session: Option<&str>, back: usize, (width, rows): (usize, usize)) -> (Vec<Line>, Footer, Option<(usize, usize)>, bool) {
    let back_hints = vec![("alt 1", "back")];
    let Some(session) = session else {
        let key = if number == 10 { 0 } else { number };
        let _ = key;
        let mut body = empty("No shell here.", "", width);
        if let Some(error) = crate::vm::terms::list_error() {
            body.push(Vec::new());
            body.extend(flow(&[(Style::Problem, format!("  The VM's shells could not be listed: {error}"))], width, 2));
        } else if !crate::vm::terms::listed() {
            body.push(Vec::new());
            body.push(vec![(Style::Work, "  reading the VM's shells…".into())]);
        }
        return (body, Footer::new(Vec::new(), back_hints), None, false);
    };
    crate::vm::terms::watch(session);
    crate::vm::terms::fit(session, width, rows);
    crate::vm::terms::want_history(session, back > 0);
    let history = (back > 0).then(|| crate::vm::terms::history(session)).flatten();
    let loading = back > 0 && history.is_none() && crate::vm::terms::error(session).is_none();
    let screen = history.or_else(|| crate::vm::terms::live(session));
    let mut left: Line = vec![(Style::Status, shell_label(session))];
    let Some(screen) = screen else {
        let text = match crate::vm::terms::error(session) {
            Some(error) => format!("  The shell could not be read: {error}"),
            None => "  reading the shell…".into(),
        };
        return (vec![Vec::new(), vec![(Style::Work, text)]], Footer::new(left, back_hints), None, loading);
    };
    // A netbox says who has it, and what it is for.
    let netbox = crate::vm::netboxes::name_of(session)
        .and_then(|name| crate::vm::terms::netboxes().into_iter().find(|netbox| netbox.name == name));
    // Folded (Ctrl+O): a line a command.
    let compacted = netbox.as_ref().filter(|_| SHELL_COMPACT.load(std::sync::atomic::Ordering::Relaxed)).map(|netbox| {
        compact(&screen.lines, screen.cursor, &crate::vm::netboxes::display(&netbox.name), screen.size.0.max(width))
    });
    let (lines, cursor) = match &compacted {
        Some((lines, cursor)) => (lines.as_slice(), *cursor),
        None => (screen.lines.as_slice(), screen.cursor),
    };
    // The pane's rows, less the blank ones under the cursor and the text,
    // in the pane's own colours.
    let last_text = lines.iter().rposition(|line| !plain_text(line).trim().is_empty()).unwrap_or(0);
    let end = (last_text.max(cursor.1) + 1).min(lines.len());
    let body: Vec<Line> = lines[..end].iter().map(|line| shell_row(line, width)).collect();
    if !screen.cwd.is_empty() {
        left.push((Style::Dim, format!("  {}", screen.cwd)));
    }
    if screen.size.0 > width {
        left.push((Style::Problem, format!("  {} columns cut off", screen.size.0 - width)));
    }
    if screen.dead {
        left.push((Style::Problem, "  exited".into()));
    }
    match crate::vm::terms::error(session) {
        Some(error) => left.push((Style::Problem, format!("  {error}"))),
        None if loading => left.push((Style::Work, "  reading its history…".into())),
        None => {}
    }
    if let Some(netbox) = &netbox {
        left.insert(1, (Style::Dim, format!("  as {}", netbox.account())));
        if netbox.wheel {
            left.insert(0, (Style::Warn, "[!] you have control  ".into()));
        } else if !netbox.running.is_empty() {
            left.insert(0, (Style::Work, "[*] s0ma is running  ".into()));
        }
        if netbox.asks_first() {
            left.push((Style::Dim, "  confirm".into()));
        }
        if !netbox.purpose.is_empty() {
            left.push((Style::Dim, format!("  · {}", netbox.purpose)));
        }
    }
    let hints = if netbox.is_some() {
        let fold = if compacted.is_some() { "expand" } else { "collapse" };
        vec![("⇧pgup", "scroll"), ("ctrl+o", fold), ("ctrl+\\", "control"), ("ctrl+q", "close"), ("alt n", "new"), ("alt 1", "back")]
    } else {
        vec![("⇧pgup", "scroll"), ("alt n", "new"), ("alt 1", "back")]
    };
    // A program that hid its cursor (a pager, an editor's menu) has none.
    (body, Footer::new(left, hints), screen.cursor_shown.then_some(cursor), loading)
}

/// The shells' tabs show each prompt folded to one line, until Ctrl+O at a
/// prompt opens them out (for every shell, until pressed again).
pub(super) static SHELL_COMPACT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// A shell, not a program in it, has the pane.
pub(super) fn at_prompt(command: &str) -> bool {
    matches!(command, "bash" | "sh" | "dash" | "zsh" | "fish" | "")
}

/// A netbox's rows with each prompt folded to one line, as the shell's
/// settings draw it (`crate::vm::netboxes::RC`): `account@name  folder` (and how the
/// last command went, on the right) over `$ command` becomes `account@name
/// folder $ command`; a failure is said in it (`[-] 101`), a success not at
/// all; the agent's reason, on its own line between the two, goes after the
/// command when the row has room. `name` is the shell's, as its prompt says
/// it; `width` the pane's. The cursor moves with its row.
pub(super) fn compact(lines: &[String], cursor: (usize, usize), name: &str, width: usize) -> (Vec<String>, (usize, usize)) {
    const RED: &str = "\x1b[38;2;255;95;95m";
    // `account@name  folder…`: the prompt's first line.
    let is_meta = |plain: &str| {
        plain.split_once(' ').is_some_and(|(first, rest)| {
            first.len() > name.len() + 1 && first.ends_with(name) && first[..first.len() - name.len()].ends_with('@') && rest.starts_with(' ')
        })
    };
    let is_reason = |plain: &str| plain.starts_with("# ") && plain.contains(": ");
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let (mut column, mut row) = cursor;
    let mut i = 0;
    while i < lines.len() {
        let raw = &lines[i];
        let plain = plain_text(raw);
        if !is_meta(&plain) {
            out.push(raw.clone());
            i += 1;
            continue;
        }
        // The reason, if any, then `$ `.
        let reason = (i + 1 < lines.len() && is_reason(&plain_text(&lines[i + 1]))).then(|| i + 1);
        let at = reason.map_or(i + 1, |reason| reason + 1);
        if at >= lines.len() || !plain_text(&lines[at]).starts_with('$') {
            out.push(raw.clone());
            i += 1;
            continue;
        }
        // The first line without what it says on the right.
        let trimmed = plain.trim_end();
        let (left, said) = match trimmed.rfind("  [") {
            Some(cut) if trimmed[cut + 2..].starts_with("[+] ") || trimmed[cut + 2..].starts_with("[-] ") => {
                (trimmed[..cut].trim_end(), Some(&trimmed[cut + 2..]))
            }
            _ => (trimmed, None),
        };
        let failed = said.and_then(|said| said.strip_prefix("[-] ")).map(|rest| {
            let words: Vec<&str> = rest.split([' ', '·']).filter(|word| !word.is_empty()).collect();
            match words.as_slice() {
                ["exit", code, ..] => code.to_string(),
                [word, ..] => word.to_string(),
                [] => "failed".into(),
            }
        });
        let head = cut_at_width(raw, text_width(left));
        // `folder $`: one space between, as in a one-line prompt.
        let head = match head.rfind("  ") {
            Some(gap) if plain_text(&head[gap..]).starts_with("  ") => format!("{}{}", &head[..gap], &head[gap + 1..]),
            _ => head,
        };
        let mut merged = format!("{head}\x1b[0m ");
        let mut added = text_width(&plain_text(&head)) + 1;
        if let Some(code) = &failed {
            merged.push_str(&format!("{RED}[-] {code}\x1b[0m "));
            added += text_width(&format!("[-] {code} "));
        }
        // A command that fills its row (it wraps) is left on its own: put
        // after the first line, it would run off the edge.
        if added + text_width(plain_text(&lines[at]).trim_end()) > width {
            out.push(raw.clone());
            i += 1;
            continue;
        }
        merged.push_str(&lines[at]);
        // The reason after the command, when it fits and isn't where the cursor is.
        let mut keep_reason = None;
        if let Some(reason) = reason {
            let why = plain_text(&lines[reason]);
            let used = text_width(plain_text(&merged).trim_end());
            if at != cursor.1 && used + 2 + text_width(why.trim_end()) <= width {
                merged = format!("{}\x1b[0m  \x1b[3;90m{}\x1b[0m", cut_at_width(&merged, used), why.trim_end());
            } else {
                keep_reason = Some(lines[reason].clone());
            }
        }
        // Rows before the cursor that went: the meta line, and the reason if merged.
        let gone = 1 + usize::from(reason.is_some() && keep_reason.is_none());
        if cursor.1 >= at {
            row -= gone.min(row);
            if cursor.1 == at {
                column += added;
            }
        } else if cursor.1 > i {
            row = out.len();
        }
        if let Some(reason) = keep_reason {
            out.push(reason);
        }
        out.push(merged);
        i = at + 1;
    }
    (out, (column, row))
}

/// `raw` up to its first `width` columns of text, escape codes kept: a row
/// without the blanks tmux padded it with.
pub(super) fn cut_at_width(raw: &str, width: usize) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut used = 0;
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            out.push(c);
            out.extend(chars.next());
            for next in chars.by_ref() {
                out.push(next);
                if ('\x40'..='\x7e').contains(&next) {
                    break;
                }
            }
            continue;
        }
        let w = c.width().unwrap_or(0);
        if used + w > width {
            break;
        }
        used += w;
        out.push(c);
    }
    out
}

/// A row of a shell as tmux keeps it, its colours and all, cut to `width`.
/// The terminal's default colour is taken as the tabs' white: a reset, or a
/// `39`, would otherwise give the terminal's own (green, on a green one).
pub(super) fn shell_row(line: &str, width: usize) -> Line {
    let row = crate::page::rows(&format!("\x1b[37m{}", whiten(line)), width).rows.into_iter().next().unwrap_or_default();
    vec![(Style::Raw, format!("{row}\x1b[0m"))]
}

/// `text`'s colour codes with the default foreground made white: `0` (or an
/// empty code) is followed by `37`, and `39` becomes `37`. The numbers inside
/// a `38;5;n` or `38;2;r;g;b` are colours, not resets, and are left alone.
pub(super) fn whiten(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    let mut rest = text;
    while let Some(at) = rest.find("\x1b[") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 2..];
        let Some(end) = after.find(|c: char| ('\x40'..='\x7e').contains(&c)) else {
            out.push_str(&rest[at..]);
            return out;
        };
        let (parameters, last) = (&after[..end], &after[end..end + 1]);
        if last != "m" {
            out.push_str(&rest[at..at + 2 + end + 1]);
        } else {
            let parts: Vec<&str> = if parameters.is_empty() { vec![""] } else { parameters.split(';').collect() };
            let mut kept: Vec<String> = Vec::new();
            let mut index = 0;
            while index < parts.len() {
                let part = parts[index];
                match part {
                    "" | "0" => kept.extend(["0".to_string(), "37".to_string()]),
                    "39" => kept.push("37".into()),
                    "38" | "48" => {
                        let take = match parts.get(index + 1) {
                            Some(&"5") => 2,
                            Some(&"2") => 4,
                            _ => 0,
                        };
                        kept.extend(parts[index..(index + 1 + take).min(parts.len())].iter().map(|p| p.to_string()));
                        index += take;
                    }
                    other => kept.push(other.to_string()),
                }
                index += 1;
            }
            out.push_str(&format!("\x1b[{}m", kept.join(";")));
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    out
}

/// `text` without its escape codes.
pub(super) fn plain_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if ('\x40'..='\x7e').contains(&next) {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}
