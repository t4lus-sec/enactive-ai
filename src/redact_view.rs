//! The window for reviewing a redaction request.
//!
//! The same shape as the mistake review window, and the same code where it is
//! the same job: the diff, the screen, the scrolling, the prompt and the
//! `$EDITOR` handoff all come from `mistake_view`, so the two read alike and
//! there is one place to fix either of them. What is here is the part that is
//! this feature's: which event, why s0ma asked, what else holds the text, and
//! what approving writes.
//!
//! Deliberately smaller than the mistake window. A correction is a whole model
//! response in several fields; a redaction is one piece of text, so there is
//! nothing to pick between and no form to parse -- `e` opens the text and what
//! comes back is the text.

use crate::hippocampus::Hippocampus;
use crate::mistake_view::{draw, edit_text, prompt_line, render_diffs, text_diff, wrap, Line, Screen, Style};
use crate::redact::{self, state_word, Book, Request, State};

use std::io::{self};
use std::time::Duration;

const KEYS: &str = "a approve · e edit · d decline · n/p request · j/k scroll · q close";

/// Everything the window shows about one request, read fresh each pass.
struct Loaded {
    request: Request,
    /// What the event says now. Absent once it has been approved, since the
    /// request no longer holds what it would have been compared against.
    now: Option<String>,
    /// Whether the event has moved since the request was made.
    stale: bool,
    /// Whether the weakness log cites this event, which approving cannot undo.
    cited: bool,
}

fn load(memory: &Hippocampus, book: &Book, n: u64) -> Result<Loaded, String> {
    let request = book.load(n)?;
    let event = memory.get_event(request.event_id).map_err(|error| error.to_string())?;
    let now = event.as_ref().map(|event| event.content.clone());
    let stale = now
        .as_ref()
        .is_some_and(|content| crate::hippocampus::content_digest(content) != request.was);
    let cited = crate::weaknesses::cites_event(memory, request.event_id);
    Ok(Loaded { request, now, stale, cited })
}

fn header(loaded: &Loaded, width: usize) -> Vec<Line> {
    let request = &loaded.request;
    let mut lines = wrap(
        &[
            (Style::Heading, format!("redaction #{} ", request.request)),
            (
                Style::Plain,
                format!(
                    "— event #{} ({}), asked {}",
                    request.event_id,
                    request.speaker,
                    request.at.get(..16).unwrap_or(&request.at)
                ),
            ),
            (Style::Dim, format!("  ·  {}", state_word(request.state))),
        ],
        width,
    );
    lines.extend(wrap(&[(Style::Bold, "s0ma's reason: ".into()), (Style::Plain, request.reason.clone())], width));
    if request.edited() {
        lines.extend(wrap(
            &[(Style::Dim, "You have edited this; approving writes your version.".into())],
            width,
        ));
    }
    if loaded.now.is_none() {
        lines.extend(wrap(&[(Style::Problem, "That event is no longer in the archive.".into())], width));
    }
    if loaded.stale {
        lines.extend(wrap(
            &[(
                Style::Problem,
                "The event has changed since this was proposed, so the proposal is against text that is no \
                 longer there. Approving is refused; decline it and let s0ma look again."
                    .into(),
            )],
            width,
        ));
    }
    if loaded.cited {
        lines.extend(wrap(
            &[(
                Style::Problem,
                "The weakness log cites this event and keeps an excerpt of it. That log is hash-chained and \
                 cannot be rewritten, so approving will not take it out of there."
                    .into(),
            )],
            width,
        ));
    }
    lines.extend(wrap(
        &[(
            Style::Dim,
            "Approving rewrites this one event, keeping its number, its time and who said it, and then clears \
             what was struck out of the indexes, the memory of it and the saved model calls that hold it."
                .into(),
        )],
        width,
    ));
    lines.push(Vec::new());
    lines
}

fn body(loaded: &Loaded, width: usize) -> Vec<Line> {
    let Some(now) = &loaded.now else { return Vec::new() };
    if loaded.request.state != State::Pending {
        return wrap(
            &[(
                Style::Dim,
                format!(
                    "Decided {}. The event now reads:",
                    loaded.request.decided_at.as_deref().unwrap_or("—")
                ),
            )],
            width,
        )
        .into_iter()
        .chain(wrap(&[(Style::Plain, now.clone())], width))
        .collect();
    }
    let diff = text_diff("the event's content".into(), now, loaded.request.current());
    render_diffs(std::slice::from_ref(&diff), width)
}

/// Review the redactions waiting, starting at `start` or the oldest pending
/// one. Returns the line to print once the window closes.
pub fn run(memory: &Hippocampus, book: &Book, start: Option<u64>) -> io::Result<String> {
    use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
    let numbers = |book: &Book| -> Vec<u64> {
        book.all().map(|all| all.into_iter().map(|r| r.request).collect()).unwrap_or_default()
    };
    let pending = |book: &Book| -> Vec<u64> {
        book.pending().map(|p| p.into_iter().map(|r| r.request).collect()).unwrap_or_default()
    };
    let mut current = match start.or_else(|| pending(book).first().copied()).or_else(|| numbers(book).last().copied())
    {
        Some(n) => n,
        None => return Ok("[redact] nothing to review".into()),
    };
    if let Err(error) = book.load(current) {
        return Ok(format!("[redact] {error}"));
    }

    let _screen = Screen::enter()?;
    let mut scroll = 0usize;
    let mut message: Option<(Style, String)> = None;
    // An edit begun and not saved, kept per request so going away and coming
    // back does not lose it, and so one request's draft never opens as
    // another's.
    let mut drafts: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    let mut decided = 0usize;
    'outer: loop {
        let loaded = match load(memory, book, current) {
            Ok(loaded) => loaded,
            Err(error) => return Ok(format!("[redact] {error}")),
        };
        loop {
            let (columns, rows) = crossterm::terminal::size()?;
            let (columns, rows) = (columns as usize, rows as usize);
            let mut lines = header(&loaded, columns);
            lines.extend(body(&loaded, columns));
            let footer = vec![
                message.clone().map(|(style, text)| vec![(style, text)]).unwrap_or_default(),
                vec![(Style::Key, format!(" {KEYS} "))],
            ];
            let max_scroll = lines.len().saturating_sub(rows.saturating_sub(footer.len()));
            scroll = scroll.min(max_scroll);
            draw(&lines, scroll, &footer, columns, rows)?;
            if !event::poll(Duration::from_secs(60))? {
                continue;
            }
            let Event::Key(key) = event::read()? else { continue };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            let page = rows.saturating_sub(4).max(1);
            message = None;
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break 'outer,
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break 'outer,
                KeyCode::Char('j') | KeyCode::Down => scroll = (scroll + 1).min(max_scroll),
                KeyCode::Char('k') | KeyCode::Up => scroll = scroll.saturating_sub(1),
                KeyCode::PageDown | KeyCode::Char(' ') => scroll = (scroll + page).min(max_scroll),
                KeyCode::PageUp => scroll = scroll.saturating_sub(page),
                KeyCode::Char('g') | KeyCode::Home => scroll = 0,
                KeyCode::Char('G') | KeyCode::End => scroll = max_scroll,

                KeyCode::Char('n') | KeyCode::Char('p') => {
                    let all = numbers(book);
                    if let Some(at) = all.iter().position(|n| *n == current) {
                        let next = if key.code == KeyCode::Char('n') { at + 1 } else { at.wrapping_sub(1) };
                        match all.get(next) {
                            Some(n) => {
                                current = *n;
                                scroll = 0;
                                continue 'outer;
                            }
                            None => message = Some((Style::Dim, "no more".into())),
                        }
                    }
                }

                KeyCode::Char('e') => {
                    if loaded.request.state != State::Pending {
                        message = Some((Style::Problem, "decided already".into()));
                        continue;
                    }
                    let draft = drafts
                        .remove(&current)
                        .unwrap_or_else(|| loaded.request.current().to_owned());
                    let Some(edited) = edit_text(&draft, &format!("redact-{current}"))? else {
                        drafts.insert(current, draft);
                        message = Some((Style::Problem, "the editor did not exit cleanly; nothing saved".into()));
                        continue;
                    };
                    if edited == draft {
                        message = Some((Style::Dim, "unchanged".into()));
                        continue;
                    }
                    match redact::save_edit(book, current, &edited) {
                        Ok(_) => {
                            message = Some((Style::Plain, "your version is what approving would write".into()));
                            continue 'outer;
                        }
                        Err(error) => {
                            drafts.insert(current, edited);
                            message = Some((Style::Problem, error));
                        }
                    }
                }

                KeyCode::Char('a') => {
                    if loaded.request.state != State::Pending {
                        message = Some((Style::Problem, "decided already".into()));
                        continue;
                    }
                    if loaded.stale || loaded.now.is_none() {
                        message = Some((Style::Problem, "the event has moved; this cannot be applied".into()));
                        continue;
                    }
                    // Said plainly before it happens: this is the one key here
                    // that destroys something, and what it destroys is not
                    // only in the archive.
                    let label = format!(
                        "approve #{current}? this rewrites event #{} and clears what was struck from the \
                         stores — type yes",
                        loaded.request.event_id
                    );
                    let answer = prompt_line(&label, "", &lines, scroll)?;
                    if answer.as_deref().map(str::trim) != Some("yes") {
                        message = Some((Style::Dim, "left alone".into()));
                        continue;
                    }
                    match redact::approve(memory, book, current) {
                        Ok((_, applied)) => {
                            decided += 1;
                            drafts.remove(&current);
                            message = Some((Style::Plain, applied.summary()));
                            continue 'outer;
                        }
                        Err(error) => message = Some((Style::Problem, error)),
                    }
                }

                KeyCode::Char('d') => {
                    if loaded.request.state != State::Pending {
                        message = Some((Style::Problem, "decided already".into()));
                        continue;
                    }
                    let Some(note) = prompt_line("why not (enter to decline with no note)", "", &lines, scroll)?
                    else {
                        message = Some((Style::Dim, "left alone".into()));
                        continue;
                    };
                    match redact::reject(book, current, &note) {
                        Ok(_) => {
                            decided += 1;
                            drafts.remove(&current);
                            message = Some((Style::Plain, "declined; the event is untouched".into()));
                            continue 'outer;
                        }
                        Err(error) => message = Some((Style::Problem, error)),
                    }
                }
                _ => {}
            }
        }
    }

    let left = pending(book).len();
    Ok(match (decided, left) {
        (0, 0) => "[redact] nothing waiting".into(),
        (0, left) => format!("[redact] {left} still waiting"),
        (done, 0) => format!("[redact] {done} decided; nothing waiting"),
        (done, left) => format!("[redact] {done} decided; {left} still waiting"),
    })
}
