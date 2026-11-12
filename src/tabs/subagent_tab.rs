//! A subagent's tab: its feed, as it works.

use super::*;

/// A tab that reads the agent's own files and state from the disk of the machine
/// it runs on, shown in a terminal attached from another computer (`s0ma
/// connect`, `session::is_remote`): those reads don't go over the
/// connection yet. The page, its history and the subagents' tabs do, as
/// the primary sends them.
pub(super) fn server_only(view: &View) -> bool {
    crate::session::is_remote() && !matches!(view, View::Subagent(_) | View::History | View::Page)
}

/// What a tab `server_only` shows instead.
pub(super) fn remote_body(view: &View, width: usize) -> (Vec<Line>, Footer) {
    let what = match view {
        View::Shell { .. } => "Shells",
        View::Memory => "The memory tab",
        View::Share => "The shared folder's tab",
        View::Overview => "The overview",
        View::Audit => "The audit tab",
        View::Edit => "The editor",
        _ => "The projects tab",
    };
    let text = [
        format!("{what} reads the files of the machine s0ma runs on, and a terminal attached with s0ma connect can't show it yet."),
        String::new(),
        "For it now, ssh to that machine and run s0ma there. /download and /upload move files from and to the shared folder from here.".to_string(),
    ];
    let mut body = vec![Vec::new()];
    for paragraph in text {
        if paragraph.is_empty() {
            body.push(Vec::new());
            continue;
        }
        body.extend(crate::style::wrap(&[(Style::Dim, format!("  {paragraph}"))], width));
    }
    (body, Footer::new(vec![(Style::Dim, "not available here".into())], vec![]))
}

pub(super) fn subagent_body(number: usize, expanded: bool, width: usize, now: Instant) -> (Vec<Line>, Footer) {
    let Some(worker) = subagent_at(number) else {
        let key = if number == 10 { 0 } else { number };
        let body = empty(&format!("No subagent on alt {key}."), "", width);
        return (body, Footer::new(Vec::new(), vec![("esc", "back")]));
    };
    crate::subagent::feed::with_feed(worker.slot, |feed| feed_lines(feed, expanded, width, now, feed.paused.as_deref()))
        .unwrap_or_else(|| (Vec::new(), Footer::new(Vec::new(), vec![("esc", "back")])))
}

/// The agent's label in a subagent's tab, where the subagent's own name has the
/// green the agent has on its page.
pub(super) const DIRECTOR_STYLE: &str = "\x1b[1;97m";

/// A subagent's feed as rows, and its footer: drawn by the renderer the agent's own
/// page uses, so it reads the same, with Ctrl+O its expanded form (reasoning
/// and every tool line).
pub(super) fn feed_lines(feed: &Feed, expanded: bool, width: usize, now: Instant, paused: Option<&str>) -> (Vec<Line>, Footer) {
    use crate::terminal::{NAME_STYLE, OPERATOR_STYLE, OPERATOR_TEXT_STYLE, RESET, STATUS_STYLE, TEXT_STYLE, safe_text};
    use crate::transcript::{Detail, Entry, Mark, Out, Renderer, WORK_STYLE};
    let id = short_id(&feed.run_id);
    let detail = if expanded { Detail::Expanded } else { Detail::Collapsed };
    let mut renderer = Renderer::new(detail, crate::transcript::DEFAULT_CHARS_PER_TOKEN).named(&id, NAME_STYLE);
    let mut text = String::new();
    let mut out = Out { text: &mut text, columns: width, wrapping: true };
    let start = feed.started;
    renderer.apply(&mut out, start, start, &Entry::Turn);
    // The task is the agent's first direction to it.
    let task_lines: Vec<&str> = feed.task.lines().filter(|line| !line.trim().is_empty()).collect();
    let shown = if expanded { task_lines.len() } else { TASK_LINES.min(task_lines.len()) };
    let mut task = safe_text(&task_lines[..shown].join("\n"));
    if shown < task_lines.len() {
        task.push_str(&format!("\n{WORK_STYLE}{} more lines{RESET}", task_lines.len() - shown));
    }
    renderer.speaker_line(&mut out, "s0ma", DIRECTOR_STYLE, TEXT_STYLE, &task);
    if feed.dropped {
        renderer.apply(&mut out, start, start, &Entry::Line(format!("{WORK_STYLE}earlier activity is in the worker journal{RESET}")));
    }
    let direction = |renderer: &mut Renderer, out: &mut Out, item: &Item| {
        let Item::Direction { from, text, wrap_up, delivered, .. } = item else {
            return;
        };
        let mut body = safe_text(text);
        if *wrap_up {
            body.push_str(&format!("{WORK_STYLE}  wrap up{RESET}"));
        }
        match (delivered, feed.running()) {
            (true, _) => {}
            (false, true) => body.push_str(&format!("{WORK_STYLE}  queued{RESET}")),
            (false, false) => body.push_str(&format!("{RESET}\x1b[31m  not taken: it had finished{RESET}")),
        }
        match from {
            Sender::Operator => renderer.speaker_line(out, "you", OPERATOR_STYLE, OPERATOR_TEXT_STYLE, &body),
            Sender::S0ma => renderer.speaker_line(out, "s0ma", DIRECTOR_STYLE, TEXT_STYLE, &body),
        }
        // Its next words are a new message, under its name again.
        renderer.apply(out, start, start, &Entry::Turn);
    };
    // The call in progress: its name, and when it began.
    let mut calling: Option<(String, Instant)> = None;
    for (index, item) in feed.items.iter().enumerate() {
        let (first, last) = feed.time(index);
        match item {
            Item::Reasoning(text) => renderer.apply(&mut out, first, last, &Entry::Reasoning(safe_text(text))),
            Item::Text(text) => {
                let mut markdown = crate::markdown::MarkdownStream::new();
                let mut styled = markdown.push(&safe_text(text));
                styled.push_str(&markdown.finish());
                renderer.apply(&mut out, first, last, &Entry::Answer(styled));
            }
            Item::Call { name, arguments } => {
                calling = Some((name.clone(), first));
                let status = safe_text(&format!("[tool] running: {name}, arguments: {arguments}"));
                renderer.apply(&mut out, first, last, &Entry::Status(status));
            }
            Item::Result(raw) => {
                let (name, began) = calling.take().unwrap_or_else(|| ("tool".to_string(), first));
                let (lines, problem) = result_gist(raw);
                let status = match &problem {
                    Some(problem) => format!("[tool] failed: {name}: {problem}"),
                    None => format!("[tool] finished: {name} ({:.1}s)", last.duration_since(began).as_secs_f64()),
                };
                renderer.apply(&mut out, first, last, &Entry::Status(safe_text(&status)));
                // Expanded, what it gave back, in short.
                for line in lines.iter().take(RESULT_LINES_EXPANDED) {
                    renderer.apply(&mut out, first, last, &Entry::Note(format!("{STATUS_STYLE}  {}{RESET}", safe_text(line))));
                }
                if lines.len() > RESULT_LINES_EXPANDED {
                    let more = format!("{WORK_STYLE}  {} more lines{RESET}", lines.len() - RESULT_LINES_EXPANDED);
                    renderer.apply(&mut out, first, last, &Entry::Note(more));
                }
            }
            Item::Direction { .. } => direction(&mut renderer, &mut out, item),
            Item::Note(text) => {
                let text = safe_text(text);
                renderer.apply(&mut out, first, last, &Entry::Notice { brief: text.clone(), full: text, mark: Mark::Work });
            }
        }
    }
    let mut left: Line = Vec::new();
    match &feed.finished {
        None => {
            left.push((Style::Status, format!("{id} · running {}", span(now.duration_since(feed.started)))));
            if let Some(paused) = paused {
                left.push((Style::Problem, format!(" · {paused}, waiting for s0ma")));
            }
        }
        Some((at, how)) => {
            let after = format!("{how} after {}", span(at.duration_since(feed.started)));
            let mark = if how == "completed" { Mark::Done } else { Mark::Problem };
            renderer.apply(&mut out, *at, *at, &Entry::Notice { brief: after.clone(), full: after.clone(), mark });
            left.push((Style::Status, format!("{id} · {after}")));
        }
    }
    // Sent and not taken in yet: after everything it is in the middle of.
    for item in &feed.waiting {
        direction(&mut renderer, &mut out, item);
    }
    if feed.running() {
        renderer.refresh(&mut out, now, true);
    } else {
        renderer.finish(&mut out);
    }
    let mut rows: Vec<Line> = vec![Vec::new()];
    rows.extend(crate::page::rows(&text, width).rows.into_iter().map(|row| vec![(Style::Raw, format!("{row}{RESET}"))]));
    while rows.last().is_some_and(|row| row.iter().all(|(_, text)| crate::page::width(text) == 0)) && rows.len() > 1 {
        rows.pop();
    }
    let waiting = feed.waiting_directions();
    if waiting > 0 {
        left.push((Style::Dim, format!(" · {waiting} queued")));
    }
    let detail = if expanded { "less" } else { "more" };
    let hints = if feed.running() {
        vec![("⏎", "steer"), ("/wrap", ""), ("/kill", ""), ("^o", detail), ("esc", "back")]
    } else {
        vec![("^o", detail), ("^q", "close"), ("esc", "back")]
    };
    (rows, Footer::new(left, hints))
}

/// A result in short: its output's lines, and what went wrong, if anything.
/// The envelope's bookkeeping (sizes, flags, ids) is for Ctrl+O.
pub(super) fn result_gist(raw: &str) -> (Vec<String>, Option<String>) {
    let Ok(Value::Object(map)) = serde_json::from_str::<Value>(raw) else {
        return (raw.trim_end().lines().map(str::to_string).collect(), None);
    };
    let text = |key: &str| map.get(key).and_then(Value::as_str).map(str::trim_end).filter(|text| !text.trim().is_empty());
    let mut problem = None;
    if let Some(code) = map.get("exit_code").and_then(Value::as_i64).filter(|code| *code != 0) {
        problem = Some(format!("exit {code}"));
    }
    if map.get("ok").and_then(Value::as_bool) == Some(false) || map.get("error").is_some_and(|error| !error.is_null()) {
        let why = text("error").or_else(|| text("message")).unwrap_or("failed");
        problem = Some(match problem {
            Some(exit) => format!("{exit} · {}", crate::mistakes::one_line(why, 200)),
            None => crate::mistakes::one_line(why, 200),
        });
    }
    let mut lines: Vec<String> = Vec::new();
    if let Some(body) = ["stdout", "output", "result", "content", "text", "screen"].iter().find_map(|key| text(key)) {
        lines.extend(body.lines().map(str::to_string));
    }
    if problem.is_some()
        && let Some(stderr) = text("stderr")
    {
        lines.extend(stderr.lines().map(str::to_string));
    }
    (lines, problem)
}
