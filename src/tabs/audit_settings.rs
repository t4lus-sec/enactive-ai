//! An audit's settings, in a window of their own: one row each, ←→
//! changes the selected one, ⏎ types a value (or turns a switch), as the
//! overview tab's settings do.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::audit::{self, Audit, Gate, Level, Post, Settings};
use crate::style::{Line, Style};

use super::gui_view::{self, Click, Pick};
use super::{Footer, Open, Outcome};

/// How a row changes.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// ←→ steps through values; ⏎ types one.
    Steps,
    /// ←→ steps through a few named values.
    Choice,
    /// On or off.
    Switch,
    /// ⏎ types it.
    Typed,
}

/// The rows: the key `/audit set` takes, the name shown, how it changes,
/// and what it is, in a few words.
const ROWS: [(&str, &str, Kind, &str); 12] = [
    ("workers", "workers", Kind::Steps, "checks written at once"),
    ("steps", "steps", Kind::Steps, "the agent's step cap"),
    ("time", "per check", Kind::Steps, "a check's time limit"),
    ("time-all", "in all", Kind::Steps, "the audit's time budget"),
    ("depth", "depth", Kind::Choice, "the deepest check"),
    ("gate", "gate", Kind::Choice, "when checking starts"),
    ("post", "post", Kind::Choice, "what reaches the board"),
    ("mutants", "mutants", Kind::Switch, "checks vetted by breaking the code"),
    ("back-translation", "read-back", Kind::Switch, "a fresh reader restates each check"),
    ("loophole", "loopholes", Kind::Switch, "a hunt for gaps in passing checks"),
    ("ask-over", "ask over", Kind::Steps, "longer runs ask first"),
    ("tools", "tools", Kind::Typed, "+name allows, -name bars"),
];

const STEPS: [i64; 7] = [0, 10, 20, 40, 80, 160, 320];
const CHECK_SECS: [u64; 8] = [60, 120, 300, 600, 1200, 1800, 3600, 7200];
const TOTAL_SECS: [u64; 7] = [1800, 3600, 7200, 14400, 28800, 57600, 86400];
const ASK_SECS: [u64; 6] = [300, 900, 1800, 3600, 7200, 14400];
const DEPTHS: [Option<Level>; 6] = [None, Some(Level::Tested), Some(Level::Property), Some(Level::Bounded), Some(Level::Model), Some(Level::Proven)];

fn value(settings: &Settings, key: &str) -> Line {
    let on_off = |on: bool| if on { vec![(Style::Success, "on".to_string())] } else { vec![(Style::Dim, "off".to_string())] };
    let plain = |text: String| vec![(Style::Plain, text)];
    match key {
        "workers" => plain(settings.workers.to_string()),
        "steps" => settings.steps.map_or(vec![(Style::Dim, "auto".into())], |n| plain(n.to_string())),
        "time" => plain(audit::show_secs(settings.check_secs)),
        "time-all" => plain(audit::show_secs(settings.total_secs)),
        "depth" => settings.depth.map_or(vec![(Style::Dim, "test first".into())], |level| plain(level.label().into())),
        "gate" => plain(match settings.gate {
            Gate::Plan => "plan".into(),
            Gate::Reviewed => "reviewed".into(),
            Gate::None => "none".into(),
        }),
        "post" => plain(match settings.post {
            Post::Auto => "auto".into(),
            Post::Failures => "failures".into(),
            Post::Review => "review".into(),
        }),
        "mutants" => on_off(settings.mutants),
        "back-translation" => on_off(settings.back_translation),
        "loophole" => on_off(settings.loophole),
        "ask-over" => plain(audit::show_secs(settings.ask_over_secs)),
        "tools" => if settings.tools.is_empty() { vec![(Style::Dim, "all".into())] } else { plain(settings.tools.join(",")) },
        _ => Vec::new(),
    }
}

/// The next value one way (`up`) from `now` in `steps`.
fn next<T: PartialOrd + Copy>(steps: &[T], now: T, up: bool) -> T {
    if up {
        steps.iter().copied().find(|s| *s > now).unwrap_or(steps[steps.len() - 1])
    } else {
        steps.iter().rev().copied().find(|s| *s < now).unwrap_or(steps[0])
    }
}

/// `key=value` for the selected row moved one step (`up`: right); None
/// at either end.
fn step(settings: &Settings, key: &str, up: bool) -> Option<String> {
    let pair = moved(settings, key, up)?;
    let mut after = settings.clone();
    after.apply(&pair).ok()?;
    (value(&after, key) != value(settings, key)).then_some(pair)
}

fn moved(settings: &Settings, key: &str, up: bool) -> Option<String> {
    let cycle = |all: &[&str], now: &str| {
        let at = all.iter().position(|v| *v == now).unwrap_or(0);
        let to = if up { (at + 1) % all.len() } else { (at + all.len() - 1) % all.len() };
        all[to].to_string()
    };
    let to = match key {
        "workers" => (if up { (settings.workers + 1).min(8) } else { settings.workers.saturating_sub(1).max(1) }).to_string(),
        "steps" => match next(&STEPS, settings.steps.unwrap_or(0), up) {
            0 => "auto".into(),
            n => n.to_string(),
        },
        "time" => audit::show_secs(next(&CHECK_SECS, settings.check_secs, up)),
        "time-all" => return Some(format!("time=/{}", audit::show_secs(next(&TOTAL_SECS, settings.total_secs, up)))),
        "ask-over" => audit::show_secs(next(&ASK_SECS, settings.ask_over_secs, up)),
        "depth" => {
            let at = DEPTHS.iter().position(|d| *d == settings.depth).unwrap_or(0);
            let to = if up { (at + 1).min(DEPTHS.len() - 1) } else { at.saturating_sub(1) };
            DEPTHS[to].map_or("auto", Level::label).to_string()
        }
        "gate" => cycle(&["plan", "reviewed", "none"], &value(settings, "gate").first()?.1),
        "post" => cycle(&["auto", "failures", "review"], &value(settings, "post").first()?.1),
        "mutants" => if settings.mutants { "off" } else { "on" }.into(),
        "back-translation" => if settings.back_translation { "off" } else { "on" }.into(),
        "loophole" => if settings.loophole { "off" } else { "on" }.into(),
        _ => return None,
    };
    Some(format!("{key}={to}"))
}

fn kind(at: usize) -> Kind {
    ROWS[at.min(ROWS.len() - 1)].2
}

pub(super) fn body(tab: &mut Open, audit: &Audit, width: usize, rows_left: usize) -> (Vec<Line>, Footer) {
    let state = &mut tab.audit;
    let at = state.settings.unwrap_or(0).min(ROWS.len() - 1);
    state.settings = Some(at);
    let mut out: Vec<Line> = vec![vec![(Style::Bold, format!("  {}", audit.name)), (Style::Dim, "  settings".into())], Vec::new()];
    for (i, (key, name, kind, _)) in ROWS.iter().enumerate() {
        let here = i == at;
        let lead = if here { (Style::Cursor, "❯ ".to_string()) } else { (Style::Plain, "  ".to_string()) };
        let mut line: Line = vec![lead, (if here { Style::Bold } else { Style::Plain }, format!("{name:<12}"))];
        let typing = state.input.as_ref().filter(|input| here && input.claim.as_deref() == Some(*key));
        if let Some(input) = typing {
            let before: String = input.text[..input.cursor].iter().collect();
            let caret: String = input.text.get(input.cursor).map_or(" ".to_string(), |c| c.to_string());
            let after: String = input.text.get(input.cursor + 1..).map(|rest| rest.iter().collect()).unwrap_or_default();
            line.extend([(Style::Plain, before), (Style::Caret, caret), (Style::Plain, after)]);
            out.push(line);
            continue;
        }
        let shown = value(&audit.settings, key);
        let width_of: usize = shown.iter().map(|(_, t)| super::text_width(t)).sum();
        if *kind == Kind::Typed {
            line.push((Style::Plain, "  ".into()));
            line.extend(shown);
            line.push((Style::Plain, "  ".into()));
        } else {
            // ‹ value ›: each arrow a click that steps it.
            line.push(gui_view::hot(Style::Dim, "‹ ", Click::Keys(vec![KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)])));
            line.extend(shown);
            line.push(gui_view::hot(Style::Dim, " ›", Click::Keys(vec![KeyEvent::new(KeyCode::Right, KeyModifiers::NONE)])));
        }
        line.push((Style::Plain, " ".repeat(14usize.saturating_sub(width_of))));
        line.push((Style::Dim, ROWS[i].3.to_string()));
        out.push(gui_view::pick_row(line, Click::Pick(Pick::AuditRow(i))));
    }
    let _ = width;
    out.truncate(rows_left);
    let hints = if state.input.is_some() {
        vec![("⏎", "set"), ("esc", "cancel")]
    } else {
        match kind(at) {
            Kind::Switch => vec![("↑↓", "move"), ("⏎", "switch"), ("esc", "back")],
            Kind::Choice => vec![("↑↓", "move"), ("←→", "change"), ("esc", "back")],
            Kind::Steps => vec![("↑↓", "move"), ("←→", "change"), ("⏎", "type"), ("esc", "back")],
            Kind::Typed => vec![("↑↓", "move"), ("⏎", "type"), ("esc", "back")],
        }
    };
    (out, Footer::new(vec![(Style::Status, "audit".into()), (Style::Dim, "  settings".into())], hints))
}

/// A key in the settings window.
pub(super) fn key(tab: &mut Open, audit: &Audit, key: &KeyEvent) -> Outcome {
    let at = tab.audit.settings.unwrap_or(0).min(ROWS.len() - 1);
    let (name, _, kind, _) = ROWS[at];
    let Some(store) = audit::store() else { return Outcome::Stay };
    let set = |tab: &mut Open, pair: Option<String>| {
        if let Some(pair) = pair {
            super::audit_view::note(tab, crate::audit::ops::set(&store, &audit.name, &pair));
        }
    };
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('c') => tab.audit.settings = None,
        KeyCode::Up => tab.audit.settings = Some(at.saturating_sub(1)),
        KeyCode::Down => tab.audit.settings = Some((at + 1).min(ROWS.len() - 1)),
        KeyCode::Home => tab.audit.settings = Some(0),
        KeyCode::End => tab.audit.settings = Some(ROWS.len() - 1),
        KeyCode::Left | KeyCode::Right if kind != Kind::Typed => set(tab, step(&audit.settings, name, key.code == KeyCode::Right)),
        KeyCode::Char(' ') if kind == Kind::Switch => set(tab, step(&audit.settings, name, true)),
        KeyCode::Enter => match kind {
            Kind::Switch | Kind::Choice => set(tab, step(&audit.settings, name, true)),
            Kind::Steps | Kind::Typed => {
                let now: String = value(&audit.settings, name).into_iter().map(|(_, t)| t).collect();
                let now = if matches!(now.as_str(), "auto" | "all" | "test first") { String::new() } else { now };
                super::audit_view::type_setting(tab, name, &now);
            }
        },
        _ => {}
    }
    Outcome::Stay
}

/// A typed value for row `key`, as `/audit set` takes it.
pub(super) fn pair(key: &str, text: &str) -> String {
    match key {
        "time-all" => format!("time=/{text}"),
        _ => format!("{key}={text}"),
    }
}

/// A click on a row: it is selected.
pub(super) fn pick(tab: &mut Open, index: usize) {
    tab.audit.settings = Some(index.min(ROWS.len() - 1));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrows_step_each_setting_through_its_values() {
        let s = Settings::default();
        assert_eq!(step(&s, "workers", true).as_deref(), Some("workers=3"));
        assert_eq!(step(&s, "workers", false).as_deref(), Some("workers=1"));
        assert_eq!(step(&s, "steps", true).as_deref(), Some("steps=10"));
        assert_eq!(step(&s, "time", true).as_deref(), Some("time=20m"));
        assert_eq!(step(&s, "time-all", false).as_deref(), Some("time=/1h"));
        assert_eq!(step(&s, "depth", true).as_deref(), Some("depth=tested"));
        assert_eq!(step(&s, "gate", true).as_deref(), Some("gate=reviewed"));
        assert_eq!(step(&s, "post", false).as_deref(), Some("post=review"));
        assert_eq!(step(&s, "mutants", true).as_deref(), Some("mutants=off"));
        // At an end, nothing.
        assert_eq!(step(&s, "depth", false), None);
        assert_eq!(step(&Settings { workers: 8, ..Settings::default() }, "workers", true), None);
        // Each applies.
        for (key, ..) in ROWS.iter().filter(|r| r.2 != Kind::Typed) {
            for up in [true, false] {
                let Some(pair) = moved(&s, key, up) else { panic!("{key}") };
                Settings::default().apply(&pair).unwrap_or_else(|e| panic!("{pair}: {e}"));
            }
        }
        let mut s = Settings { steps: Some(20), depth: Some(Level::Tested), ..Settings::default() };
        s.apply(&step(&s.clone(), "steps", false).unwrap()).unwrap();
        s.apply("steps=auto depth=auto tools=+tla").unwrap();
        assert_eq!((s.steps, s.depth, s.tools.clone()), (None, None, vec!["+tla".to_string()]));
        s.apply("tools=all").unwrap();
        assert!(s.tools.is_empty());
    }
}
