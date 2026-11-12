//! The palette every screen draws with: what each `Style` means and the
//! terminal colours it is, and the styled line the screens are made of.

use std::io::{self, Write};

use unicode_width::UnicodeWidthChar;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Plain,
    Dim,
    Bold,
    Heading,
    Removed,
    Added,
    RemovedEmph,
    AddedEmph,
    Banner,
    Problem,
    Key,
    /// The agent's green (`terminal::NAME_STYLE`).
    Accent,
    /// The operator's blue (`terminal::OPERATOR_STYLE`).
    Operator,
    /// The status line's dim cyan (`terminal::STATUS_STYLE`).
    Status,
    /// Text that carries its own colour codes (the conversation as the page
    /// draws it), written as it is.
    Raw,
    /// Bold teal (`WARN_COLOR`): under way, or wants attention (a lead
    /// being explored). Beside the agent's green, the operator's blue and the
    /// status line's dim cyan, and not red, which is for problems.
    Warn,
    /// Bold red: ruled out, failed.
    Danger,
    /// Bold green: done, or for.
    Success,
    /// The chevron's bright cyan, bold: where the selection is.
    Cursor,
    /// A work line's italic grey (`transcript::WORK_STYLE`).
    Work,
    /// The tab that is up: the chevron's bright cyan, inverted.
    Tab,
    /// Text marked to be struck: white on red.
    Marked,
    /// The text cursor in the memory tab's strip: inverted.
    Caret,
    /// Text the mouse can click in a tab (`tabs::gui_view`): drawn as its
    /// look; the number is what the click does, for the frame drawn.
    Hot(u16, &'static Style),
}


pub type Line = Vec<(Style, String)>;

pub(crate) fn width_of(text: &str) -> usize {
    text.chars().map(|c| c.width().unwrap_or(0)).sum()
}


/// Wrap styled segments to `width` columns.
pub(crate) fn wrap(segments: &[(Style, String)], width: usize) -> Vec<Line> {
    let width = width.max(1);
    let mut lines = vec![Vec::new()];
    let mut used = 0;
    for (style, text) in segments {
        let mut piece = String::new();
        for c in text.chars() {
            let c = if c == '\t' { ' ' } else { c };
            if c.is_control() {
                continue;
            }
            let w = c.width().unwrap_or(0);
            if used + w > width {
                if !piece.is_empty() {
                    lines.last_mut().expect("one line").push((*style, std::mem::take(&mut piece)));
                }
                lines.push(Vec::new());
                used = 0;
            }
            piece.push(c);
            used += w;
        }
        if !piece.is_empty() {
            lines.last_mut().expect("one line").push((*style, piece));
        }
    }
    lines
}


/// Attention's colour (`Style::Warn`, and `Style::Banner` behind black): a
/// bright teal, of the cyan family the status line is in, brighter and bold.
pub(crate) const WARN_COLOR: crossterm::style::Color = crossterm::style::Color::Rgb { r: 64, g: 224, b: 208 };

pub(crate) fn apply_style(out: &mut impl Write, style: Style) -> io::Result<()> {
    use crossterm::style::{Attribute, Color, SetAttribute, SetBackgroundColor, SetForegroundColor};
    crossterm::queue!(out, SetAttribute(Attribute::Reset))?;
    match style {
        // Text in the main page's white, not the terminal's own colour (which
        // on a green-on-black terminal turned every page green); green is
        // kept for what stands out.
        Style::Plain => crossterm::queue!(out, SetForegroundColor(Color::Grey)),
        Style::Raw => Ok(()),
        Style::Warn => crossterm::queue!(out, SetAttribute(Attribute::Bold), SetForegroundColor(WARN_COLOR)),
        Style::Danger => crossterm::queue!(out, SetAttribute(Attribute::Bold), SetForegroundColor(Color::Red)),
        Style::Success => crossterm::queue!(out, SetAttribute(Attribute::Bold), SetForegroundColor(Color::Green)),
        Style::Cursor => crossterm::queue!(out, SetAttribute(Attribute::Bold), SetForegroundColor(Color::Cyan)),
        Style::Dim => crossterm::queue!(out, SetForegroundColor(Color::DarkGrey)),
        Style::Bold => crossterm::queue!(out, SetAttribute(Attribute::Bold), SetForegroundColor(Color::White)),
        Style::Heading => crossterm::queue!(out, SetAttribute(Attribute::Bold), SetForegroundColor(Color::Cyan)),
        Style::Removed => crossterm::queue!(out, SetForegroundColor(Color::Red)),
        Style::Added => crossterm::queue!(out, SetForegroundColor(Color::Green)),
        Style::RemovedEmph => crossterm::queue!(out, SetForegroundColor(Color::Black), SetBackgroundColor(Color::Red)),
        Style::AddedEmph => crossterm::queue!(out, SetForegroundColor(Color::Black), SetBackgroundColor(Color::Green)),
        Style::Banner => crossterm::queue!(out, SetForegroundColor(Color::Black), SetBackgroundColor(WARN_COLOR)),
        Style::Problem => crossterm::queue!(out, SetForegroundColor(Color::Red)),
        Style::Key => crossterm::queue!(out, SetAttribute(Attribute::Reverse)),
        Style::Accent => {
            crossterm::queue!(out, SetAttribute(Attribute::Bold), SetForegroundColor(Color::Rgb { r: 51, g: 255, b: 51 }))
        }
        Style::Operator => {
            crossterm::queue!(out, SetAttribute(Attribute::Bold), SetForegroundColor(Color::Rgb { r: 95, g: 175, b: 255 }))
        }
        Style::Status => crossterm::queue!(out, SetAttribute(Attribute::Dim), SetForegroundColor(Color::DarkCyan)),
        Style::Work => crossterm::queue!(out, SetAttribute(Attribute::Italic), SetForegroundColor(Color::DarkGrey)),
        Style::Tab => crossterm::queue!(
            out,
            SetAttribute(Attribute::Bold),
            SetForegroundColor(Color::Black),
            SetBackgroundColor(Color::Cyan)
        ),
        Style::Marked => crossterm::queue!(
            out,
            SetAttribute(Attribute::Bold),
            SetForegroundColor(Color::White),
            SetBackgroundColor(Color::DarkRed)
        ),
        Style::Caret => crossterm::queue!(out, SetAttribute(Attribute::Reverse)),
        Style::Hot(_, look) => apply_style(out, *look),
    }
}

