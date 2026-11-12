use chrono::{DateTime, Local};
use crate::agent::SteerPoint;
use crate::completion::{self, longest_common_prefix};
use crate::line_edit::{Draft, SteerHistory};
use crate::notify::Focus;
use crate::terminal::{
    ActiveDisplay, OPERATOR_STYLE, OperatorNote, RESET, STATUS_STYLE,
};

use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableFocusChange, EnableBracketedPaste, EnableFocusChange,
        Event, KeyCode as CrosstermKeyCode, KeyEventKind, KeyModifiers as CrosstermKeyModifiers,
        KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode},
};

use reedline::{
    Color, Completer, CompletionResult, DescriptionMode, EditCommand, EditMode, Emacs, ExampleHighlighter,
    FileBackedHistory, HistoryItem, IdeMenu, KeyCode, KeyModifiers, MenuBuilder, Prompt, PromptEditMode,
    PromptHistorySearch, PromptHistorySearchStatus, Reedline, ReedlineEvent, ReedlineMenu, ReedlineRawEvent,
    SearchDirection, SearchQuery, Signal, Span, Suggestion, default_emacs_keybindings,
};

use std::{
    borrow::Cow,
    collections::VecDeque,
    io,
    path::PathBuf,
    process::Command,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU8, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::Duration,
};

const HISTORY_CAPACITY: usize = 1_000;
const FOCUS_FOCUSED: u8 = 1;
const FOCUS_UNFOCUSED: u8 = 2;
/// Ctrl+O at the idle prompt, as reedline hands it back.
const TOGGLE_DETAIL: &str = "toggle-detail";
/// `tab:<key>`: Alt+<key> at the idle prompt, a chat or the board;
/// `term:<key>`: Alt+Shift+<key>, a terminal (see `tabs.rs`).
const TAB_COMMAND: &str = "tab:";
const TERMINAL_COMMAND: &str = "term:";
/// The keys that lead to a tab, with Alt (Alt+1 is the agent, Alt+2…9 and Alt+0
/// its subagents) or with Alt+Shift (the shells).
const TAB_KEYS: [char; 10] = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'];
/// A resize followed by another within this is one resize: the last.
const RESIZE_SETTLE: Duration = Duration::from_millis(120);
/// Ctrl+L at the idle prompt: the page drawn again.
const REDRAW: &str = "redraw";
/// Alt+A and Alt+X: approval mode's answer (`approval.rs`).
const APPROVE: &str = "approve";
const DECLINE: &str = "decline";
/// The terminal changed size at the idle prompt: the page drawn again for it.
const RESIZED: &str = "resized";
/// PgUp (`scroll:page`) or the wheel (`scroll:wheel`) at the idle prompt:
/// the conversation further back (`tabs::history`).
const SCROLL_BACK: &str = "scroll:";
/// Ctrl+G at the idle prompt: the draft in the operator's editor.
const EDIT_DRAFT: &str = "edit-draft";
/// Ctrl+Shift+P at the idle prompt: two panes (`tabs::split_from_page`).
const SPLIT_PANES: &str = "split-panes";

#[derive(Debug)]
pub enum InputEvent {
    Submit(String),
    Interrupt,
    Eof,
    Timeout,
}

/// The steering editor shared with the keyboard worker during a turn.
#[derive(Default)]
struct ActiveEditor {
    draft: Draft,
    history: SteerHistory,
    cycle: Option<CompletionCycle>,
}

/// Repeated Tab with several matches steps through them.
struct CompletionCycle {
    start: usize,
    values: Vec<String>,
    index: usize,
    /// The draft as the last Tab left it; any other edit ends the cycle.
    after: Draft,
}

/// Terminal focus as last reported (focus reporting is on during turns).
#[derive(Clone, Default)]
struct FocusTracker {
    seen: Arc<AtomicBool>,
    state: Arc<AtomicU8>,
}

/// Reedline adapter for the shared completion logic.
struct SomaCompleter;

impl Completer for SomaCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> CompletionResult {
        let completion = completion::complete(line, pos);
        CompletionResult::fresh(
            completion
                .candidates
                .into_iter()
                .map(|candidate| Suggestion {
                    value: candidate.value,
                    description: candidate.description,
                    span: Span::new(completion.start, completion.end),
                    append_whitespace: candidate.append_space,
                    ..Suggestion::default()
                })
                .collect::<Vec<_>>(),
        )
    }
}

/// `$VISUAL`, then `$EDITOR`, then nano / vim / vi from `PATH`.
/// The idle prompt's keys (Emacs's, with the agent's bindings), and what the page
/// needs instead of reedline's own handling: the wheel scrolls back through
/// the conversation, and a resize draws the whole page again (reedline would
/// draw only its prompt).
struct PageKeys(Emacs);

impl EditMode for PageKeys {
    fn parse_event(&mut self, event: ReedlineRawEvent) -> ReedlineEvent {
        match Event::from(event) {
            Event::Mouse(mouse) if mouse.kind == crossterm::event::MouseEventKind::ScrollUp => {
                ReedlineEvent::ExecuteHostCommand(format!("{SCROLL_BACK}wheel"))
            }
            // A click on the tab bar: the tab's key, as if pressed.
            Event::Mouse(mouse) if mouse.kind == crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
                match crate::tabs::page_click(mouse.column, mouse.row).map(|key| ReedlineRawEvent::try_from(Event::Key(key))) {
                    Some(Ok(event)) => self.0.parse_event(event),
                    _ => ReedlineEvent::None,
                }
            }
            Event::Mouse(_) => ReedlineEvent::None,
            Event::Resize(..) => ReedlineEvent::ExecuteHostCommand(RESIZED.to_string()),
            other => match ReedlineRawEvent::try_from(other) {
                Ok(event) => self.0.parse_event(event),
                Err(()) => ReedlineEvent::None,
            },
        }
    }

    fn edit_mode(&self) -> PromptEditMode {
        self.0.edit_mode()
    }
}

fn editor_command() -> Command {
    let configured = ["VISUAL", "EDITOR"]
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .find(|value| !value.trim().is_empty());
    let words: Vec<String> = match configured {
        Some(value) => value.split_whitespace().map(str::to_string).collect(),
        None => {
            let found = ["nano", "vim", "vi"].into_iter().find(|program| {
                std::env::var_os("PATH").is_some_and(|path| {
                    std::env::split_paths(&path).any(|dir| dir.join(program).is_file())
                })
            });
            vec![found.unwrap_or("vi").to_string()]
        }
    };
    let mut command = Command::new(&words[0]);
    command.args(&words[1..]);
    command
}

fn draft_file() -> PathBuf {
    std::env::temp_dir().join(format!("s0ma-draft-{}.md", std::process::id()))
}

/// Expand `@file` mentions; returns the content for the model and one note
/// per mention for the operator.
pub(crate) fn expand_attachments(text: &str) -> (String, Vec<String>) {
    let expanded = crate::attachments::expand(text);
    (expanded.content, expanded.notes)
}

type SharedEditor = Arc<Mutex<ActiveEditor>>;

/// The draft being typed. A key whose handling panicked while it was held
/// does not lose the rest of the turn's typing: the draft is taken as it was.
fn lock_editor(editor: &SharedEditor) -> io::Result<std::sync::MutexGuard<'_, ActiveEditor>> {
    Ok(editor.lock().unwrap_or_else(|poisoned| {
        editor.clear_poison();
        poisoned.into_inner()
    }))
}

struct QueuedSubmission {
    content: String,
    visible_text: Option<String>,
    /// When it was sent (Enter, or an attached client's or voice's send),
    /// so a message that waits in the queue says when it was written.
    sent_at: DateTime<Local>,
    /// Sent with "send now" (Ctrl+S, Ctrl+Enter, a `/now` prefix): taken at
    /// the next steering point instead of waiting for a safe one, and takes
    /// everything queued before it along.
    forced: bool,
    /// The message itself is in the transcript (sent now, or taken by the agent).
    shown: bool,
    /// A status line already said it is queued; it goes into the transcript
    /// when the agent takes it.
    announced: bool,
}

/// `/now <message>`: send it at once rather than after the agent's current step.
/// Returns what follows the prefix (possibly empty: send what's queued).
pub fn strip_send_now(text: &str) -> Option<&str> {
    let rest = text.trim_start().strip_prefix("/now")?;
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        Some(rest.trim_start())
    } else {
        // `/nowhere`, `/notes`: not this command.
        None
    }
}

/// Whether queued messages are handed over at this steering point: always at
/// a safe one, and anywhere once one of them was sent with "send now". Never
/// between a batch's calls, which only looks (see `sent_now_waiting`).
fn releases(point: SteerPoint, queued: &VecDeque<QueuedSubmission>) -> bool {
    !queued.is_empty()
        && match point {
            SteerPoint::Safe => true,
            SteerPoint::Streaming => queued.iter().any(|s| s.forced),
            SteerPoint::BetweenCalls => false,
        }
}

/// Whether a message sent with "send now" is waiting in the queue.
fn sent_now_waiting(queued: &VecDeque<QueuedSubmission>) -> bool {
    queued.iter().any(|s| s.forced)
}

/// An operator message handed to the agent, with when it was sent: one that
/// waited in the queue says so in its header (`clock::Arrival`).
#[derive(Clone, Debug, PartialEq)]
pub struct Submitted {
    pub content: String,
    pub sent_at: DateTime<Local>,
}

/// The messages a release hands to the agent, in order: a bare "send now" carries
/// no text of its own.
fn release_contents(released: Vec<QueuedSubmission>) -> (Vec<Submitted>, usize) {
    let contents: Vec<Submitted> = released
        .into_iter()
        .filter(|submission| !submission.content.trim().is_empty())
        .map(|submission| Submitted { content: submission.content, sent_at: submission.sent_at })
        .collect();
    let count = contents.len();
    (contents, count)
}

/// Cloneable ingress for finalized input produced outside the terminal.
/// External submissions share the same queue and wake path as keyboard
/// steering instead of maintaining a second input transport.
#[derive(Clone)]
pub struct ExternalSubmitter {
    submits: Sender<QueuedSubmission>,
    external_break_signal: Arc<AtomicBool>,
    display: ActiveDisplay,
}

impl ExternalSubmitter {
    /// While the agent works, the message waits for a safe point like a typed one;
    /// a `/now` prefix sends it at once.
    pub fn submit(&self, content: String, visible_text: String) -> Result<(), String> {
        let (content, visible_text, forced) =
            match (strip_send_now(&content), strip_send_now(&visible_text)) {
                (Some(content), Some(visible)) => (content.to_string(), visible.to_string(), true),
                _ => (content, visible_text, false),
            };
        let counted = !content.trim().is_empty();
        if counted {
            let _ = self.display.queue_steer();
        }
        let sent = self.submits.send(QueuedSubmission {
            content,
            visible_text: Some(visible_text),
            sent_at: Local::now(),
            forced,
            shown: false,
            announced: false,
        });
        if sent.is_err() {
            if counted {
                let _ = self.display.release_steers(1);
            }
            return Err("operator input queue disconnected".to_string());
        }

        self.external_break_signal.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// What an attached terminal's steering pane sent, handled as this
    /// terminal's own pane handles it (`steer_text`), leaving this pane's
    /// draft alone.
    pub fn steer(&self, text: String, force: bool) -> io::Result<Steered> {
        let steered = steer_text(&self.display, &self.submits, text, force, false);
        self.external_break_signal.store(true, Ordering::Relaxed);
        steered
    }
}

type RemoteSteer = Box<dyn Fn(String, bool) -> io::Result<Steered> + Send + Sync>;
/// On an attached terminal: where the steering pane's messages go (the text
/// as typed, and whether it was sent now), instead of this process's queue.
static REMOTE_STEER: OnceLock<RemoteSteer> = OnceLock::new();

/// Make this terminal a window onto another agent (`session.rs`): what is
/// typed goes to `send`, and is shown when that the agent takes it.
pub fn steer_remotely(send: RemoteSteer) {
    let _ = REMOTE_STEER.set(send);
}

#[cfg(test)]
impl ExternalSubmitter {
}

/// The tab bar atop the page, for the terminal's width now.
fn idle_bar() -> (String, usize) {
    let columns = crossterm::terminal::size().map_or(80, |(columns, _)| columns);
    crate::tabs::main_bar(columns.saturating_sub(1) as usize)
}

/// The agent's normal idle operator prompt.
///
/// Reedline owns the terminal only while the application is waiting
/// for an ordinary operator turn.
#[derive(Clone)]
struct SomaPrompt {
    display: ActiveDisplay,
    /// The prompt a submitted line is redrawn with: the same `you> `,
    /// without the corner's tok/s and Ctrl+O hint, which would otherwise
    /// stay in the transcript on every line the operator sent.
    submitted: bool,
}

/// The idle prompt's draft in the operator's colour (`OPERATOR_TEXT_STYLE`),
/// which it keeps on screen once sent, as the steering pane and Ctrl+O's
/// redraw show it.
fn operator_highlighter() -> ExampleHighlighter {
    let mut highlighter = ExampleHighlighter::default();
    highlighter.change_colors(Color::LightGray, Color::LightGray, Color::LightGray);
    highlighter
}

impl Prompt for SomaPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Owned(format!("{OPERATOR_STYLE}you{RESET}"))
    }

    /// The bottom-right corner, as during a turn: the last reply's tok/s
    /// and what Ctrl+O does.
    fn render_prompt_right(&self) -> Cow<'_, str> {
        if self.submitted {
            return Cow::Borrowed("");
        }
        Cow::Owned(format!("{STATUS_STYLE}{}{RESET}", self.display.idle_hint()))
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> Cow<'_, str> {
        Cow::Owned(format!("{OPERATOR_STYLE}>  {RESET}"))
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        // Later lines of the draft under the first one's text.
        Cow::Borrowed("      ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };

        Cow::Owned(format!(
            "({prefix}reverse-search: {}) ",
            history_search.term
        ))
    }
}

pub struct InputReader {
    editor: Reedline,
    prompt: SomaPrompt,

    break_signal: Arc<AtomicBool>,
    external_break_signal: Arc<AtomicBool>,

    eof_signal: Arc<AtomicBool>,

    active: bool,
    active_worker_enabled: Arc<AtomicBool>,
    active_worker_reading: Arc<AtomicBool>,
    active_editor: SharedEditor,
    submits: Receiver<QueuedSubmission>,
    /// Messages sent during a turn that wait for a safe point (or the turn's
    /// end), oldest first.
    queued: VecDeque<QueuedSubmission>,
    external_submitter: ExternalSubmitter,
    active_display: ActiveDisplay,
    focus: FocusTracker,
    /// How many times a job's close has asked the history to be forgotten, as
    /// of the last prompt. See `forget_history_if_asked`.
    history_purges: u64,
    history_file: PathBuf,
    /// The tab bar as last drawn atop the page between turns.
    drawn_bar: Arc<Mutex<String>>,
    /// A tab between turns has the terminal as it needs it (raw, bracketed
    /// paste, the keyboard's flags pushed, the mouse reported), kept so while
    /// the app answers for it (the overview's questions, a wake looked at)
    /// and the tab stays up: with them off meanwhile, a key or the mouse's
    /// release in that moment was echoed onto the screen. `leave_tab_modes`
    /// puts them back before anything else takes the terminal.
    tab_modes: bool,
    /// The size the page was last drawn for at a resize between turns, and
    /// when: a resize's later events, coming straight after for that size,
    /// draw nothing more.
    page_size: Option<((u16, u16), std::time::Instant)>,
}

impl InputReader {
    /// `history_file`: where prompt history persists across restarts.
    pub fn spawn(hard_interrupt: Arc<AtomicBool>, history_file: PathBuf) -> io::Result<Self> {

        let break_signal = Arc::new(AtomicBool::new(false));
        let external_break_signal = Arc::new(AtomicBool::new(false));

        let eof_signal = Arc::new(AtomicBool::new(false));

        let mut keybindings = default_emacs_keybindings();

        // Shift+Enter inserts a newline at the idle prompt. Most terminals only
        // tell it apart from Enter under the kitty keyboard protocol, which the
        // editor asks for below (`use_kitty_keyboard_enhancement`).
        keybindings.add_binding(
            KeyModifiers::SHIFT,
            KeyCode::Enter,
            ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
        );

        // Alt+Enter too: it works in every terminal, with or without the
        // protocol.
        keybindings.add_binding(
            KeyModifiers::ALT,
            KeyCode::Enter,
            ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
        );

        // Alt+\ (and Alt+|): two panes (`tabs::split_from_page`). Ctrl+Shift+P
        // too, where the kitty keyboard protocol tells it from Ctrl+P.
        for (modifiers, key) in [
            (KeyModifiers::ALT, '\\'),
            (KeyModifiers::ALT, '|'),
            (KeyModifiers::ALT | KeyModifiers::SHIFT, '|'),
            (KeyModifiers::CONTROL | KeyModifiers::SHIFT, 'p'),
            (KeyModifiers::CONTROL | KeyModifiers::SHIFT, 'P'),
        ] {
            keybindings.add_binding(modifiers, KeyCode::Char(key), ReedlineEvent::ExecuteHostCommand(SPLIT_PANES.to_string()));
        }

        // Ctrl+G: edit the prompt in $VISUAL/$EDITOR (replaces Emacs redo).
        // The agent's own, not reedline's: the editor gets the shell's screen, and
        // the page is drawn again after (`edit_idle_draft`).
        keybindings.add_binding(
            KeyModifiers::CONTROL,
            KeyCode::Char('g'),
            ReedlineEvent::ExecuteHostCommand(EDIT_DRAFT.to_string()),
        );

        // Alt+A (twice) and Alt+X: the oldest request waiting for approval,
        // approved or declined (`approval.rs`).
        for (key, command) in [('a', APPROVE), ('A', APPROVE), ('x', DECLINE), ('X', DECLINE)] {
            keybindings.add_binding(KeyModifiers::ALT, KeyCode::Char(key), ReedlineEvent::ExecuteHostCommand(command.to_string()));
        }

        // Ctrl+L: the page drawn again (reedline's would clear the tab bar
        // away with the rest).
        keybindings.add_binding(
            KeyModifiers::CONTROL,
            KeyCode::Char('l'),
            ReedlineEvent::ExecuteHostCommand(REDRAW.to_string()),
        );

        // PgUp: the conversation further back than the page shows.
        keybindings.add_binding(
            KeyModifiers::NONE,
            KeyCode::PageUp,
            ReedlineEvent::ExecuteHostCommand(format!("{SCROLL_BACK}page")),
        );

        // Ctrl+O: collapse or expand reasoning and tool calls, as during a turn
        // (reedline binds it to the editor too; Ctrl+G stays that).
        keybindings.add_binding(
            KeyModifiers::CONTROL,
            KeyCode::Char('o'),
            ReedlineEvent::ExecuteHostCommand(TOGGLE_DETAIL.to_string()),
        );

        // Alt+1…9, Alt+0: the chats; Alt+B: back; Alt+Shift+1…9, 0: the
        // terminals (`tabs.rs`). Alt+Shift+1 arrives as Alt+! in most
        // terminals, and as Alt+Shift+1 with the kitty keyboard protocol: both
        // are bound. Alt+B was a word left, as in Emacs; Alt+Left and
        // Ctrl+Left still are.
        for key in TAB_KEYS {
            keybindings.add_binding(
                KeyModifiers::ALT,
                KeyCode::Char(key),
                ReedlineEvent::ExecuteHostCommand(format!("{TAB_COMMAND}{key}")),
            );
            keybindings.add_binding(
                KeyModifiers::ALT | KeyModifiers::SHIFT,
                KeyCode::Char(key),
                ReedlineEvent::ExecuteHostCommand(format!("{TERMINAL_COMMAND}{key}")),
            );
        }
        for (shifted, key) in crate::tabs::SHIFTED_DIGITS {
            for modifiers in [KeyModifiers::ALT, KeyModifiers::ALT | KeyModifiers::SHIFT] {
                keybindings.add_binding(
                    modifiers,
                    KeyCode::Char(shifted),
                    ReedlineEvent::ExecuteHostCommand(format!("{TERMINAL_COMMAND}{key}")),
                );
            }
        }
        for modifiers in [KeyModifiers::ALT, KeyModifiers::ALT | KeyModifiers::SHIFT] {
            for key in ['b', 'B'] {
                keybindings.add_binding(
                    modifiers,
                    KeyCode::Char(key),
                    ReedlineEvent::ExecuteHostCommand(format!("{TAB_COMMAND}b")),
                );
            }
            for key in ['m', 'M'] {
                keybindings.add_binding(
                    modifiers,
                    KeyCode::Char(key),
                    ReedlineEvent::ExecuteHostCommand(format!("{TAB_COMMAND}m")),
                );
            }
            for key in ['s', 'S'] {
                keybindings.add_binding(
                    modifiers,
                    KeyCode::Char(key),
                    ReedlineEvent::ExecuteHostCommand(format!("{TAB_COMMAND}s")),
                );
            }
            for key in ['o', 'O'] {
                keybindings.add_binding(
                    modifiers,
                    KeyCode::Char(key),
                    ReedlineEvent::ExecuteHostCommand(format!("{TAB_COMMAND}o")),
                );
            }
            for key in ['p', 'P'] {
                keybindings.add_binding(
                    modifiers,
                    KeyCode::Char(key),
                    ReedlineEvent::ExecuteHostCommand(format!("{TAB_COMMAND}p")),
                );
            }
            for key in ['e', 'E'] {
                keybindings.add_binding(
                    modifiers,
                    KeyCode::Char(key),
                    ReedlineEvent::ExecuteHostCommand(format!("{TAB_COMMAND}e")),
                );
            }
            for key in ['u', 'U'] {
                keybindings.add_binding(
                    modifiers,
                    KeyCode::Char(key),
                    ReedlineEvent::ExecuteHostCommand(format!("{TAB_COMMAND}u")),
                );
            }
        }

        // Tab: complete commands, subcommands and @file paths.
        keybindings.add_binding(
            KeyModifiers::NONE,
            KeyCode::Tab,
            ReedlineEvent::UntilFound(vec![
                ReedlineEvent::Menu("completion_menu".to_string()),
                ReedlineEvent::MenuNext,
            ]),
        );
        keybindings.add_binding(KeyModifiers::SHIFT, KeyCode::BackTab, ReedlineEvent::MenuPrevious);

        let emacs = PageKeys(Emacs::new(keybindings));
        let active_display = ActiveDisplay::new();
        // Harness output from anywhere goes through it from now on.
        crate::terminal::share(&active_display);

        let callback_break_signal = Arc::clone(&break_signal);
        let callback_external_break_signal = Arc::clone(&external_break_signal);
        let callback_display = active_display.clone();
        let drawn_bar: Arc<Mutex<String>> = Arc::default();
        let callback_bar = Arc::clone(&drawn_bar);
        let bar_display = active_display.clone();

        let history = match FileBackedHistory::with_file(HISTORY_CAPACITY, history_file.clone()) {
            Ok(history) => history,
            Err(error) => {
                crate::terminal::warn(&format!(
                    "[input] prompt history at {} unavailable ({error}); this session's history won't be saved",
                    history_file.display()
                ));
                FileBackedHistory::new(HISTORY_CAPACITY).map_err(|e| io::Error::other(e.to_string()))?
            }
        };

        let completion_menu = IdeMenu::default()
            .with_name("completion_menu")
            .with_default_border()
            .with_description_mode(DescriptionMode::PreferRight);

        let editor = Reedline::create()
            .with_history(Box::new(history))
            .with_completer(Box::new(SomaCompleter))
            .with_menu(ReedlineMenu::EngineCompleter(Box::new(completion_menu)))
            .with_quick_completions(true)
            .with_partial_completions(true)
            .with_buffer_editor(editor_command(), draft_file())
            .use_bracketed_paste(true)
            // Shift+Enter as itself, where the terminal supports the protocol
            // (the steering pane asks for the same; see `enable_key_disambiguation`).
            .use_kitty_keyboard_enhancement(key_disambiguation_wanted())
            .with_highlighter(Box::new(operator_highlighter()))
            .with_edit_mode(Box::new(emacs))
            .with_transient_prompt(Box::new(SomaPrompt { display: active_display.clone(), submitted: true }))
            .with_break_signal(Arc::clone(&break_signal))
            .with_poll_interval(Duration::from_millis(100))
            .with_idle_callback(Box::new(move || {
                // A wake, or a harness line held while the prompt is up
                // (written above it, below).
                // Or a tab asked for by a command (`/redact`), opened once the
                // prompt is up, as Alt and a key would.
                if callback_external_break_signal.load(Ordering::Relaxed)
                    || callback_display.idle_output_waiting()
                    || crate::tabs::request_pending()
                    || crate::overview::pending()
                {
                    callback_break_signal.store(true, Ordering::Relaxed);
                }
                // The tab bar out of date (a subagent started or finished, a
                // shell opened): drawn again atop the page, here, between
                // reedline's own draws.
                let mut drawn = callback_bar.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                if !drawn.is_empty()
                    && idle_bar().0 != *drawn
                    && let Ok(bar) = bar_display.draw_idle_bar()
                {
                    *drawn = bar;
                }
            }));

        let active_worker_enabled = Arc::new(AtomicBool::new(false));
        let active_worker_reading = Arc::new(AtomicBool::new(false));
        let active_editor: SharedEditor = Arc::default();
        // Up in the steering pane reaches back into earlier sessions too.
        if let Ok(items) = editor
            .history()
            .search(SearchQuery::everything(SearchDirection::Forward, None))
        {
            let mut shared = lock_editor(&active_editor)?;
            let skip = items.len().saturating_sub(500);
            for item in items.iter().skip(skip) {
                shared.history.push(&item.command_line, false);
            }
        }
        let focus = FocusTracker::default();
        let (submit_tx, submits) = mpsc::channel();
        let external_submitter = ExternalSubmitter {
            submits: submit_tx.clone(),
            external_break_signal: Arc::clone(&external_break_signal),
            display: active_display.clone(),
        };

        spawn_active_input_worker(
            Arc::clone(&active_worker_enabled),
            Arc::clone(&active_worker_reading),
            Arc::clone(&active_editor),
            active_display.clone(),
            submit_tx,
            Arc::clone(&hard_interrupt),
            Arc::clone(&eof_signal),
            focus.clone(),
        );

        Ok(Self {
            // Starts where the counter is: a close before this prompt existed
            // has already had its file purged, and nothing is held in memory.
            history_purges: crate::jobs::history_purges(),
            history_file: history_file.clone(),
            editor,
            prompt: SomaPrompt { display: active_display.clone(), submitted: false },
            drawn_bar,
            tab_modes: false,
            page_size: None,

            break_signal,
            external_break_signal,

            eof_signal,

            active: false,
            active_worker_enabled,
            active_worker_reading,
            active_editor,
            submits,
            queued: VecDeque::new(),
            external_submitter,
            active_display,
            focus,
        })
    }

    // ========================================================
    // IDLE INPUT
    // ========================================================

    /// Wait indefinitely for ordinary operator input.
    ///
    /// Reedline owns the terminal during this call.
    pub fn recv(&mut self) -> io::Result<InputEvent> {
        if self.eof_signal.load(Ordering::SeqCst) {
            return Ok(InputEvent::Eof);
        }

        self.break_signal.store(false, Ordering::Relaxed);

        let result = self.read_idle_event();

        self.break_signal.store(false, Ordering::Relaxed);

        result
    }

    /// A background event source may set this flag to interrupt the idle
    /// Reedline prompt without manufacturing a periodic timeout. The typed
    /// edit buffer remains intact across the external break.
    pub fn external_break_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.external_break_signal)
    }

    /// Call when an operator turn starts: they just typed, so they are
    /// looking at the terminal.
    pub fn assume_focused(&self) {
        self.focus.state.store(FOCUS_FOCUSED, Ordering::SeqCst);
    }

    pub fn focus(&self) -> Focus {
        if !self.focus.seen.load(Ordering::SeqCst) {
            return Focus::Unknown;
        }
        match self.focus.state.load(Ordering::SeqCst) {
            FOCUS_UNFOCUSED => Focus::Unfocused,
            _ => Focus::Focused,
        }
    }

    /// What the idle prompt holds, and where its cursor is.
    pub fn idle_draft(&self) -> (String, usize) {
        (self.editor.current_buffer_contents().to_string(), self.editor.current_insertion_point())
    }

    /// Whether the steering pane is open (a turn's input mode).
    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn is_eof(&self) -> bool {
        self.eof_signal.load(Ordering::SeqCst)
    }

    pub fn external_submitter(&self) -> ExternalSubmitter {
        self.external_submitter.clone()
    }

    /// Retrieve a submission that is still queued when a turn ends (sent
    /// mid-turn after the last safe point, or before Ctrl+C), or that
    /// arrived while Reedline was sleeping. One per call, oldest first: the
    /// first starts the next turn, and the rest reach it at its first safe
    /// point. External input is rendered here because the normal terminal
    /// owns the screen at this point.
    pub fn take_pending_submit(&mut self) -> Option<Submitted> {
        loop {
            let submission = match self.queued.pop_front() {
                Some(submission) => submission,
                None => self.submits.try_recv().ok()?,
            };
            // A bare "send now" has nothing of its own to send.
            if submission.content.trim().is_empty() {
                continue;
            }
            let _ = self.active_display.release_steers(1);

            // Printed through the display, so Ctrl+O's redraw keeps them.
            // Not shown yet (queued, or from outside the keyboard): here,
            // where the agent takes it. Made safe: from an attached terminal or
            // voice it must not be able to move the cursor or restyle the screen.
            if !submission.shown
                && let Some(visible_text) = submission.visible_text.as_deref().filter(|text| !text.trim().is_empty()) {
                    let _ = self.active_display.show_operator(visible_text);
                }
            // It was said to be waiting: say it arrived.
            if submission.shown || submission.announced {
                let _ = self.active_display.delivered(1);
            }

            return Some(Submitted { content: submission.content, sent_at: submission.sent_at });
        }
    }

    /// Drop what the history is holding, if a job's close has asked for it.
    ///
    /// A close rewrites `input_history.txt`, but this session's lines are in
    /// memory here until something syncs, and `FileBackedHistory`'s own drop
    /// syncs on the way out -- so without this the job's typed lines would be
    /// written back after being removed. Clearing is all of it, not a
    /// selection: this side cannot tell which line was the job's, and the file
    /// has already had those taken out, so a sync after this writes back the
    /// purged file and nothing else. The cost is that ↑ forgets the rest of
    /// this session too, which is the right way round.
    ///
    /// Reedline's `clear` also deletes the file, which holds every earlier
    /// session's lines (already purged of the job's): it is put back as it
    /// was.
    pub fn forget_history_if_asked(&mut self) {
        let purges = crate::jobs::history_purges();
        if purges == self.history_purges {
            return;
        }
        self.history_purges = purges;
        let kept = std::fs::read(&self.history_file).ok();
        let cleared = self.editor.history_mut().clear();
        if let Some(kept) = kept {
            let temp = self.history_file.with_extension("keep");
            if std::fs::write(&temp, &kept).and_then(|()| std::fs::rename(&temp, &self.history_file)).is_err() {
                crate::terminal::warn_status("[input] the prompt history file could not be put back after a job closed");
            }
        }
        if let Err(error) = cleared {
            crate::terminal::warn(&format!(
                "[input] the prompt history was not cleared after a job closed ({error}); what was typed \
                 during it may still be recalled with Up, and saved on the way out"
            ));
            return;
        }
        let _ = self.editor.sync_history();
        if let Ok(mut editor) = lock_editor(&self.active_editor) {
            editor.history.forget();
        }
    }

    fn read_idle_event(&mut self) -> io::Result<InputEvent> {
        loop {
            // A tab left open while a turn ran, or while the agent looked at a
            // wake: it has the keyboard again until it is closed.
            if crate::tabs::is_open()
                && let Some(event) = self.run_idle_tab()?
            {
                return Ok(event);
            }
            // The prompt's turn: the terminal as reedline wants it.
            self.leave_tab_modes();
            self.forget_history_if_asked();
            // Lines the harness writes meanwhile are held, then written above
            // the prompt (`ExternalBreak`), rather than after what is typed.
            if self.active_display.idle_output_waiting() {
                let _ = self.active_display.flush_idle();
            }
            // Something drew over the page (a review window): drawn again.
            if crate::page::take_stale() {
                let _ = self.active_display.redraw_idle(None);
            }
            if let Ok(bar) = self.active_display.draw_idle_bar() {
                *self.drawn_bar.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = bar;
            }
            self.active_display.prompt_up(true);
            // The wheel scrolls back through the conversation.
            crate::terminal::set_mouse(true);
            let read = self.editor.read_line(&self.prompt);
            crate::terminal::set_mouse(false);
            self.active_display.prompt_up(false);
            let signal = match read {
                Ok(signal) => signal,

                Err(error)
                    if error
                        .to_string()
                        .contains("cursor position could not be read") =>
                {
                    crate::terminal::warn_status("[input] terminal cursor query timed out; retrying");

                    continue;
                }

                Err(error) => {
                    return Err(io::Error::other(error.to_string()));
                }
            };

            self.break_signal.store(false, Ordering::Relaxed);

            let event = match signal {
                Signal::Success(buffer) if REMOTE_STEER.get().is_some() => {
                    // An attached terminal: what was typed goes to the agent it
                    // is attached to as it is, and is shown when it takes it.
                    let _ = self.active_display.erase_submitted(&buffer);
                    let _ = self.active_display.flush_idle();
                    if let Ok(mut editor) = lock_editor(&self.active_editor) {
                        editor.history.push(&buffer, false);
                    }
                    if buffer.trim().is_empty() {
                        continue;
                    }
                    InputEvent::Submit(buffer)
                }

                Signal::Success(buffer) => {
                    // Reedline left it on screen; Ctrl+O's redraw needs it too.
                    // A line held as Enter was pressed goes above it.
                    let _ = self.active_display.show_submitted(&buffer);
                    // Shared with the steering editor so Up recalls it there.
                    if let Ok(mut editor) = lock_editor(&self.active_editor) {
                        editor.history.push(&buffer, false);
                    }
                    // Nothing is running, so "send now" is just sending.
                    let buffer = match strip_send_now(&buffer) {
                        Some(rest) if rest.trim().is_empty() => continue,
                        Some(rest) => rest.to_string(),
                        None => buffer,
                    };
                    let (content, notes) = expand_attachments(&buffer);
                    for note in notes {
                        crate::terminal::say(&format!("{STATUS_STYLE}{note}{RESET}"));
                    }
                    InputEvent::Submit(content)
                }

                Signal::CtrlC => InputEvent::Interrupt,

                Signal::CtrlD => {
                    self.eof_signal.store(true, Ordering::SeqCst);

                    InputEvent::Eof
                }

                Signal::ExternalBreak(_buffer) => {
                    let woken = self.external_break_signal.swap(false, Ordering::Relaxed);

                    // Reedline leaves its prompt drawn. Erase it: if a turn
                    // starts now the draft moves into the steering pane, and a
                    // frozen copy would stay above it; otherwise the prompt is
                    // simply drawn again on the same row.
                    let draft = self.editor.current_buffer_contents().to_string();
                    let cursor = self.editor.current_insertion_point();
                    let _ = crate::terminal::erase_idle_prompt(&draft, cursor);

                    // Harness lines that came while the prompt was up go where
                    // it was, and it comes back below them.
                    if self.active_display.idle_output_waiting() {
                        let start = crate::terminal::cursor_row();
                        let _ = self.active_display.flush_idle();
                        if let Some(start) = start {
                            let columns = crossterm::terminal::size().map_or(80, |(columns, _)| columns);
                            let (_, rows) = crate::terminal::prompt_geometry(&draft, cursor, columns);
                            let _ = crate::terminal::clear_of_prompt(start, rows);
                        }
                    }
                    if let Some(target) = crate::tabs::take_request() {
                        let (draft, cursor) = self.idle_draft();
                        if let Err(error) = crate::tabs::go(target, &self.active_display, Some((&draft, cursor))) {
                            crate::tabs::close();
                            let _ = self.active_display.uncover();
                            crate::terminal::warn_status(&format!("[input] the tab could not be opened: {error}"));
                        }
                    }
                    // The overview tab asked the app for something: it comes
                    // round (`App::serve_overview`).
                    if !woken && !crate::overview::pending() {
                        continue;
                    }

                    // Reedline preserves the edit buffer across
                    // the external break, so a wake does not
                    // destroy something the operator was typing.
                    InputEvent::Timeout
                }

                Signal::HostCommand(command) if command == TOGGLE_DETAIL => {
                    // The prompt keeps its draft across a host command.
                    let draft = self.editor.current_buffer_contents().to_string();
                    let cursor = self.editor.current_insertion_point();
                    if let Err(error) = self.active_display.toggle_detail_idle(&draft, cursor) {
                        crate::terminal::warn_status(&format!("[input] ctrl+o: {error}"));
                    }
                    continue;
                }

                Signal::HostCommand(command) if command == REDRAW || command == RESIZED => {
                    let (draft, cursor) = self.idle_draft();
                    if command == RESIZED {
                        // Drawn once the size stops changing; the rest of the
                        // burst, for the size just drawn, is let go.
                        let now = crossterm::terminal::size().ok();
                        if let Some((size, at)) = self.page_size
                            && Some(size) == now
                            && at.elapsed() < Duration::from_secs(2)
                        {
                            continue;
                        }
                        self.page_size = crate::terminal::settled_size().ok().map(|size| (size, std::time::Instant::now()));
                    }
                    if let Err(error) = self.active_display.redraw_idle(Some((&draft, cursor))) {
                        crate::terminal::warn_status(&format!("[input] the page could not be drawn: {error}"));
                    }
                    continue;
                }

                Signal::HostCommand(command) if command.starts_with(SCROLL_BACK) => {
                    let rows = crossterm::terminal::size().map_or(24, |(_, rows)| rows as usize);
                    let back = if command.ends_with("wheel") { crate::tabs::WHEEL_ROWS } else { rows.saturating_sub(5).max(1) };
                    let (draft, cursor) = self.idle_draft();
                    if let Err(error) = crate::tabs::history(&self.active_display, Some((&draft, cursor)), back) {
                        crate::tabs::close();
                        let _ = self.active_display.uncover();
                        crate::terminal::warn_status(&format!("[input] the conversation could not be shown: {error}"));
                    }
                    continue;
                }

                Signal::HostCommand(command) if command == SPLIT_PANES => {
                    let (draft, cursor) = self.idle_draft();
                    match crate::tabs::split_from_page(&self.active_display, Some((&draft, cursor))) {
                        // The prompt's draft went to the left pane's prompt.
                        Ok(_) => self.editor.run_edit_commands(&[EditCommand::Clear]),
                        Err(error) => {
                            crate::tabs::close();
                            let _ = self.active_display.uncover();
                            crate::terminal::warn_status(&format!("[input] the panes could not be opened: {error}"));
                        }
                    }
                    continue;
                }

                Signal::HostCommand(command) if command == APPROVE || command == DECLINE => {
                    // The prompt goes, what the key says (and anything held
                    // meanwhile) takes its place, and the prompt comes back
                    // below it with its draft, as for a line that comes
                    // while it is up (`ExternalBreak`).
                    let (draft, cursor) = self.idle_draft();
                    let _ = crate::terminal::erase_idle_prompt(&draft, cursor);
                    self.active_display.prompt_up(false);
                    let _ = self.active_display.flush_idle();
                    answer_approval(command == APPROVE);
                    continue;
                }

                Signal::HostCommand(command) if command == EDIT_DRAFT => {
                    if let Err(error) = self.edit_idle_draft() {
                        crate::terminal::warn_status(&format!("[input] editor: {error}"));
                    }
                    continue;
                }

                Signal::HostCommand(command)
                    if command.starts_with(TAB_COMMAND) || command.starts_with(TERMINAL_COMMAND) =>
                {
                    let chosen = if let Some(key) = command.strip_prefix(TAB_COMMAND) {
                        key.chars().next().map(|key| (crate::tabs::Group::Chat, key))
                    } else {
                        command[TERMINAL_COMMAND.len()..].chars().next().map(|key| (crate::tabs::Group::Terminal, key))
                    };
                    let Some(target) = chosen.and_then(|(group, key)| crate::tabs::target(group, key)) else {
                        continue;
                    };
                    if target == crate::tabs::Target::Main {
                        // Already here.
                        continue;
                    }
                    // The prompt keeps its draft across a host command; the
                    // tab covers it, and it is there again when the tab closes.
                    let draft = self.editor.current_buffer_contents().to_string();
                    let cursor = self.editor.current_insertion_point();
                    if let Err(error) = crate::tabs::go(target, &self.active_display, Some((&draft, cursor))) {
                        crate::tabs::close();
                        let _ = self.active_display.uncover();
                        crate::terminal::warn_status(&format!("[input] the tab could not be opened: {error}"));
                    }
                    continue;
                }

                Signal::HostCommand(_) => {
                    continue;
                }

                _ => {
                    continue;
                }
            };

            return Ok(event);
        }
    }

    /// The terminal modes a tab kept up across a moment with the app
    /// (`tab_modes`), put back as they were before it.
    fn leave_tab_modes(&mut self) {
        if !std::mem::take(&mut self.tab_modes) {
            return;
        }
        disable_key_disambiguation();
        let _ = execute!(io::stdout(), DisableBracketedPaste);
        crate::terminal::set_mouse(false);
        let _ = disable_raw_mode();
    }

    /// Ctrl+G at the idle prompt: the draft in the operator's editor, on the
    /// shell's screen, then the page drawn again with what was written.
    fn edit_idle_draft(&mut self) -> io::Result<()> {
        let (draft, _) = self.idle_draft();
        let path = draft_file();
        std::fs::write(&path, &draft)?;
        crate::page::step_out();
        let status = editor_command().arg(&path).status();
        crate::page::back();
        crate::page::take_stale();
        let redrawn = self.active_display.redraw_idle(None);
        let edited = match status {
            Ok(status) if status.success() => std::fs::read_to_string(&path).map(|mut text| {
                // Editors add a final newline; it isn't part of the message.
                if text.ends_with('\n') {
                    text.pop();
                }
                text
            }),
            Ok(status) => Err(io::Error::other(format!("editor exited with {status}"))),
            Err(error) => Err(error),
        };
        let _ = std::fs::remove_file(&path);
        redrawn?;
        let text = edited?;
        self.editor.run_edit_commands(&[EditCommand::Clear, EditCommand::InsertString(text)]);
        Ok(())
    }

    /// What was typed in a pane of the agent's page, between turns: as the prompt
    /// takes a line (kept in the record, recalled with Up, attachments read).
    fn submitted_from_pane(&mut self, text: String) -> Option<InputEvent> {
        let _ = self.active_display.show_submitted(&text);
        if let Ok(mut editor) = lock_editor(&self.active_editor) {
            editor.history.push(&text, false);
        }
        let text = match strip_send_now(&text) {
            Some(rest) if rest.trim().is_empty() => return None,
            Some(rest) => rest.to_string(),
            None => text,
        };
        let (content, notes) = expand_attachments(&text);
        for note in notes {
            crate::terminal::say(&format!("{STATUS_STYLE}{note}{RESET}"));
        }
        Some(InputEvent::Submit(content))
    }

    /// A tab has the screen between turns: its keys, until it is closed
    /// (`None`; the prompt comes back) or something needs the agent's loop. A wake
    /// (a subagent finishing, a VM event, a submission from outside) returns
    /// `Timeout` with the tab still open, so the agent is never held up by one: a
    /// turn may start and run behind it, and its keyboard worker carries the
    /// tab on (`handle_active_event`). Ctrl+C and Ctrl+D are what they are
    /// at the prompt.
    fn run_idle_tab(&mut self) -> io::Result<Option<InputEvent>> {
        if !self.tab_modes {
            if let Err(error) = enable_raw_mode() {
                crate::tabs::close();
                let _ = self.active_display.uncover();
                crate::terminal::warn_status(&format!("[input] the tab was closed: {error}"));
                return Ok(None);
            }
            let _ = execute!(io::stdout(), EnableBracketedPaste);
            enable_key_disambiguation();
            crate::terminal::set_mouse(true);
            self.tab_modes = true;
        }
        let outcome = loop {
            if self.external_break_signal.load(Ordering::Relaxed) {
                self.external_break_signal.store(false, Ordering::Relaxed);
                break Ok(Some(InputEvent::Timeout));
            }
            // The overview asked the app for something: back to it, the tab
            // still open.
            if crate::overview::pending() {
                break Ok(Some(InputEvent::Timeout));
            }
            let event = match event::poll(Duration::from_millis(100)) {
                Ok(true) => match event::read() {
                    Ok(event) => Some(event),
                    Err(error) => break Err(error),
                },
                Ok(false) => None,
                Err(error) => break Err(error),
            };
            if let Some(event) = event {
                match crate::tabs::event(&event, &self.active_display) {
                    Ok(crate::tabs::Outcome::Stay) => {}
                    // Typed in a pane of the agent's page: a turn, the panes kept up.
                    Ok(crate::tabs::Outcome::Submit { text, .. }) => {
                        if let Some(submitted) = self.submitted_from_pane(text) {
                            break Ok(Some(submitted));
                        }
                    }
                    Ok(crate::tabs::Outcome::Closed) => break Ok(None),
                    Ok(crate::tabs::Outcome::Interrupt) => break Ok(Some(InputEvent::Interrupt)),
                    Ok(crate::tabs::Outcome::Eof) => {
                        self.eof_signal.store(true, Ordering::SeqCst);
                        break Ok(Some(InputEvent::Eof));
                    }
                    Err(error) => break Err(error),
                }
            }
            if let Err(error) = crate::tabs::tick(&self.active_display) {
                break Err(error);
            }
        };
        if outcome.is_err() {
            crate::tabs::close();
        }
        // Back to the app for a moment, the tab still up: the terminal stays
        // as the tab has it.
        if matches!(outcome, Ok(Some(InputEvent::Timeout))) && crate::tabs::is_open() {
            return outcome;
        }
        self.tab_modes = false;
        disable_key_disambiguation();
        let _ = execute!(io::stdout(), DisableBracketedPaste);
        let uncovered = if crate::tabs::is_open() { Ok(()) } else { self.active_display.uncover() };
        crate::terminal::set_mouse(false);
        let raw = disable_raw_mode();
        match outcome {
            // A tab that cannot read or draw is closed, and the agent's prompt
            // comes back: never a reason to stop the agent.
            Err(error) => {
                crate::terminal::warn_status(&format!("[input] the tab was closed: {error}"));
                Ok(None)
            }
            outcome => {
                uncovered?;
                raw?;
                outcome
            }
        }
    }

    // ========================================================
    // ACTIVE INPUT
    // ========================================================

    /// Enter the lightweight input mode used while the agent is
    /// actively reasoning, speaking, or using tools.
    ///
    /// Reedline is not running in this mode. Crossterm gives us
    /// nonblocking access to raw key events through poll_active().
    pub fn begin_active(&mut self) -> io::Result<()> {
        if self.active {
            return Ok(());
        }
        self.leave_tab_modes();

        // Whatever is half-typed at the idle prompt (a background wake can
        // start a turn mid-sentence) moves into the steering pane, cursor and
        // all, rather than vanishing until the turn ends.
        let draft = Draft::new(
            self.editor.current_buffer_contents(),
            self.editor.current_insertion_point(),
        );
        self.editor.run_edit_commands(&[EditCommand::Clear]);

        if let Err(error) = enable_raw_mode() {
            self.restore_idle_draft(&draft);
            return Err(error);
        }

        // Anything held at the prompt goes out before the pane opens.
        let _ = self.active_display.flush_idle();
        if let Err(error) = self.active_display.begin() {
            let _ = disable_raw_mode();
            self.restore_idle_draft(&draft);
            return Err(error);
        }

        {
            let mut editor = lock_editor(&self.active_editor)?;
            editor.draft = draft.clone();
            editor.history.stop_browsing();
            editor.cycle = None;
        }
        self.active_display.set_input(draft.text(), draft.cursor())?;

        // Reedline only enables bracketed paste while its own prompt is open.
        // Without it, every newline in a paste arrives as Enter and submits a
        // fragment. Focus reports tell the done-notification whether the
        // operator looked away.
        let _ = execute!(io::stdout(), EnableBracketedPaste, EnableFocusChange);
        enable_key_disambiguation();

        self.active = true;
        self.active_worker_enabled.store(true, Ordering::SeqCst);

        Ok(())
    }

    /// Leave active steering mode and return terminal ownership
    /// to normal Reedline input.
    pub fn end_active(&mut self) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }

        // Stop the keyboard worker before returning stdin to Reedline. The
        // short handshake prevents both readers from consuming one key.
        self.active_worker_enabled.store(false, Ordering::SeqCst);

        // Waited for however long it takes: the reader holds stdin for as
        // long as the operator's editor or a /mistake view is open from the
        // turn, and a panic there lets go of it (`Reading`).
        while self.active_worker_reading.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(1));
        }
        // A resize the reader had not got to yet is laid out from here, now
        // that nothing else reads the terminal's replies.
        let _ = self.active_display.catch_up_resize();
        crate::terminal::set_mouse(false);

        disable_key_disambiguation();
        let _ = execute!(io::stdout(), DisableBracketedPaste, DisableFocusChange);

        let display_result = self.active_display.end();

        let raw_result = disable_raw_mode();

        self.active = false;

        // Hand the unfinished draft back to Reedline so the idle prompt
        // shows it, and let Up there recall steers sent during the turn.
        let (draft, steers) = {
            let mut editor = lock_editor(&self.active_editor)?;
            editor.history.stop_browsing();
            editor.cycle = None;
            (std::mem::take(&mut editor.draft), editor.history.take_unsynced())
        };
        if !steers.is_empty() {
            for steer in steers {
                let _ = self
                    .editor
                    .history_mut()
                    .save(HistoryItem::from_command_line(steer));
            }
            // Written now, not only at exit, so a crash doesn't lose them.
            let _ = self.editor.sync_history();
        }
        self.restore_idle_draft(&draft);

        display_result?;
        raw_result?;

        Ok(())
    }

    fn restore_idle_draft(&mut self, draft: &Draft) {
        if draft.is_empty() {
            return;
        }
        self.editor.run_edit_commands(&[
            EditCommand::Clear,
            EditCommand::InsertString(draft.text().to_string()),
            EditCommand::MoveToPosition {
                position: draft.cursor(),
                select: false,
            },
        ]);
    }

    /// Return the shared active terminal renderer.
    ///
    /// StreamPrinter receives a clone so streamed model output and
    /// live operator input share one terminal layout.
    pub fn active_display(&self) -> ActiveDisplay {
        self.active_display.clone()
    }

    /// Hand the agent the operator's messages, if this steering point takes them.
    ///
    /// Messages wait in a queue for a safe point (see `agent::SteerPoint`):
    /// after a batch of tool calls, never in the middle of the model's output
    /// or with tool calls decided on and not yet run. One sent with "send
    /// now" is taken at the next point of either kind, along with everything
    /// queued before it. Whatever is still queued when the turn ends starts
    /// the next one (`take_pending_submit`). Ctrl+C requests the existing
    /// hard interrupt; Ctrl+D requests EOF and interrupts the active agent so
    /// the outer runtime can exit.
    pub fn poll_active(&mut self, point: SteerPoint) -> io::Result<Vec<Submitted>> {
        if !self.active {
            return Ok(Vec::new());
        }

        let first_new = self.queued.len();
        while let Ok(submission) = self.submits.try_recv() {
            self.queued.push_back(submission);
        }
        let releasing = releases(point, &self.queued);

        // Messages from outside the keyboard (attached terminals, voice):
        // one sent now goes into the transcript at once; one that waits gets
        // a status line, and is shown when the agent takes it (below).
        if !releasing {
            for submission in self.queued.iter_mut().skip(first_new).filter(|s| !s.shown && !s.announced) {
                let Some(visible_text) = submission.visible_text.as_deref().filter(|t| !t.trim().is_empty()) else {
                    continue;
                };
                if submission.forced {
                    let typed = lock_editor(&self.active_editor)?.draft.clone();
                    self.active_display.commit_operator(visible_text, OperatorNote::SendNow)?;
                    self.active_display.set_input(typed.text(), typed.cursor())?;
                    submission.shown = true;
                } else {
                    self.active_display.queued(visible_text)?;
                    submission.announced = true;
                }
            }
        }

        if point == SteerPoint::BetweenCalls && sent_now_waiting(&self.queued) {
            // A look only: say a message was sent now, and leave it queued.
            return Ok(vec![Submitted { content: String::new(), sent_at: Local::now() }]);
        }
        if !releasing {
            return Ok(Vec::new());
        }
        // Handed over now: each message not in the transcript yet goes in
        // here, in the order the agent reads them, after the work it waited for.
        let released: Vec<QueuedSubmission> = self.queued.drain(..).collect();
        for submission in released.iter().filter(|s| !s.shown) {
            if let Some(visible_text) = submission.visible_text.as_deref().filter(|t| !t.trim().is_empty()) {
                self.active_display.show_operator(visible_text)?;
            }
        }
        let (contents, count) = release_contents(released);
        self.active_display.release_steers(count)?;
        Ok(contents)
    }

    // ========================================================
    // INTERNAL STATE
    // ========================================================
}

#[allow(clippy::too_many_arguments)]
fn spawn_active_input_worker(
    enabled: Arc<AtomicBool>,
    reading: Arc<AtomicBool>,
    editor: SharedEditor,
    display: ActiveDisplay,
    submits: Sender<QueuedSubmission>,
    hard_interrupt: Arc<AtomicBool>,
    eof_signal: Arc<AtomicBool>,
    focus: FocusTracker,
) {
    let spawned = thread::Builder::new().name(KEYS_THREAD.into()).spawn(move || {
        // Clears `reading` however a pass ends, a panic included, so
        // `end_active` never waits on a reader that is gone.
        struct Reading<'a>(&'a AtomicBool);
        impl Drop for Reading<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        // An event read while waiting out a burst of resizes, handled next.
        let mut next: Option<Event> = None;
        loop {
            if !enabled.load(Ordering::SeqCst) {
                next = None;
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            reading.store(true, Ordering::SeqCst);
            let _reading = Reading(&reading);
            // Claimed, then checked again: `end_active` turns this off and
            // then waits for `reading` to clear, so between the first check
            // and the claim it may already have let the prompt read. Only one
            // of the two reads stdin.
            if !enabled.load(Ordering::SeqCst) {
                continue;
            }

            let pass = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // The wheel scrolls back through the conversation, or a tab
                // up over the turn.
                crate::terminal::set_mouse(true);

                let event_result = match next.take() {
                    Some(event) => Ok(Some(event)),
                    None => match event::poll(Duration::from_millis(50)) {
                        Ok(true) => event::read().map(Some),
                        Ok(false) => Ok(None),
                        Err(error) => Err(error),
                    },
                };
                // Dragging a window's edge sends a resize for every step:
                // only the last of a burst is laid out.
                let event_result = match event_result {
                    Ok(Some(Event::Resize(mut columns, mut rows))) => {
                        while let Ok(true) = event::poll(RESIZE_SETTLE) {
                            match event::read() {
                                Ok(Event::Resize(c, r)) => (columns, rows) = (c, r),
                                Ok(other) => {
                                    next = Some(other);
                                    break;
                                }
                                Err(_) => break,
                            }
                        }
                        Ok(Some(Event::Resize(columns, rows)))
                    }
                    other => other,
                };

                let result = match event_result {
                    Ok(Some(event)) => handle_active_event(
                        event,
                        &editor,
                        &display,
                        &submits,
                        &hard_interrupt,
                        &eof_signal,
                        &focus,
                    ),
                    Ok(None) => Ok(()),
                    Err(error) => Err(error),
                };

                if let Err(error) = result {
                    let _ = display.write_stream(&format!("\r\n[steer input error] {error}\r\n"));
                    thread::sleep(Duration::from_millis(100));
                }

                let _ = display.tick();
                // An open tab follows what it shows (a subagent streaming).
                let _ = crate::tabs::tick(&display);
            }));
            if let Err(panic) = pass {
                // The keys are read on: one that broke is not a reason to
                // stop reading the rest. What it left half done is put back
                // first: raw mode (an editor may have been running), a tab's
                // screen with no tab, the pane put away for an editor.
                let _ = enable_raw_mode();
                let _ = display.recover_after_panic(crate::tabs::is_open());
                let why = take_key_panic().unwrap_or_else(|| {
                    panic
                        .downcast_ref::<&str>()
                        .map(|text| text.to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "no message".into())
                });
                let _ = display.write_stream(&format!("\r\n[steer input error] a key's handling failed ({why}); keys still work\r\n"));
            }
        }
    });
    if let Err(error) = spawned {
        crate::terminal::warn_status(&format!("[input] could not start reading keys during turns: {error}"));
    }
}

/// The name of the thread that reads the keys during a turn. A panic there
/// is caught and reported in the conversation (`note_key_panic`), not
/// printed over the pane.
pub(crate) const KEYS_THREAD: &str = "turn-keys";

static KEY_PANIC: Mutex<Option<String>> = Mutex::new(None);

/// From the panic hook, for a panic on the keys thread: what and where,
/// kept for the message the thread writes once it has caught it.
pub(crate) fn note_key_panic(info: &std::panic::PanicHookInfo<'_>) {
    let what = info
        .payload()
        .downcast_ref::<&str>()
        .map(|text| text.to_string())
        .or_else(|| info.payload().downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".into());
    let text = match info.location() {
        Some(at) => format!("{what}, at {}:{}", at.file(), at.line()),
        None => what,
    };
    *KEY_PANIC.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(text);
}

fn take_key_panic() -> Option<String> {
    KEY_PANIC.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take()
}

#[allow(clippy::too_many_arguments)]
fn handle_active_event(
    event: Event,
    editor: &SharedEditor,
    display: &ActiveDisplay,
    submits: &Sender<QueuedSubmission>,
    hard_interrupt: &AtomicBool,
    eof_signal: &AtomicBool,
    focus: &FocusTracker,
) -> io::Result<()> {
    match event {
        Event::FocusGained | Event::FocusLost => {
            focus.seen.store(true, Ordering::SeqCst);
            let state = if matches!(event, Event::FocusLost) {
                FOCUS_UNFOCUSED
            } else {
                FOCUS_FOCUSED
            };
            focus.state.store(state, Ordering::SeqCst);
            Ok(())
        }

        // A tab has the screen: its keys are its own (`tabs.rs`).
        _ if crate::tabs::is_open() => match crate::tabs::event(&event, display) {
            Ok(crate::tabs::Outcome::Stay) => Ok(()),
            // Typed in a pane of the agent's page: steering, as the pane's own.
            Ok(crate::tabs::Outcome::Submit { text, now }) => {
                if let Ok(mut editor) = lock_editor(editor) {
                    editor.history.push(&text, true);
                }
                send_steered(display, submits, text, now)
            }
            Ok(crate::tabs::Outcome::Closed) => switching_screens(|| display.uncover()),
            Ok(crate::tabs::Outcome::Interrupt) => {
                hard_interrupt.store(true, Ordering::SeqCst);
                Ok(())
            }
            Ok(crate::tabs::Outcome::Eof) => {
                eof_signal.store(true, Ordering::SeqCst);
                hard_interrupt.store(true, Ordering::SeqCst);
                switching_screens(|| display.uncover())
            }
            Err(error) => {
                crate::tabs::close();
                switching_screens(|| display.uncover())?;
                Err(error)
            }
        },

        Event::Key(key) if key.kind != KeyEventKind::Release => handle_active_key(
            key.code,
            key.modifiers,
            editor,
            display,
            submits,
            hard_interrupt,
            eof_signal,
        ),

        Event::Paste(text) => {
            end_completion(editor, display)?;
            edit(editor, display, |draft| draft.insert_str(&text))
        }

        Event::Resize(columns, rows) => display.resize(columns, rows),

        Event::Mouse(mouse) if mouse.kind == crossterm::event::MouseEventKind::ScrollUp => {
            scroll_back(display, crate::tabs::WHEEL_ROWS)
        }

        // A click on the tab bar: the tab's key, as if pressed.
        Event::Mouse(mouse) if mouse.kind == crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
            match crate::tabs::page_click(mouse.column, mouse.row) {
                Some(key) => handle_active_event(Event::Key(key), editor, display, submits, hard_interrupt, eof_signal, focus),
                None => Ok(()),
            }
        }

        _ => Ok(()),
    }
}

/// PgUp or the wheel during a turn: the conversation further back than the
/// page shows (`tabs::history`), over the turn.
fn scroll_back(display: &ActiveDisplay, back: usize) -> io::Result<()> {
    switching_screens(|| match crate::tabs::history(display, None, back) {
        Ok(_) => Ok(()),
        Err(error) => {
            crate::tabs::close();
            display.uncover()?;
            Err(error)
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn handle_active_key(
    code: CrosstermKeyCode,
    modifiers: CrosstermKeyModifiers,
    editor: &SharedEditor,
    display: &ActiveDisplay,
    submits: &Sender<QueuedSubmission>,
    hard_interrupt: &AtomicBool,
    eof_signal: &AtomicBool,
) -> io::Result<()> {
    use CrosstermKeyCode as K;

    let ctrl = modifiers.contains(CrosstermKeyModifiers::CONTROL);
    let alt = modifiers.contains(CrosstermKeyModifiers::ALT);

    if code == K::Tab && !ctrl && !alt {
        return complete(editor, display);
    }
    end_completion(editor, display)?;

    // Alt+digit, Alt+B, Alt+Shift+digit: a tab over the turn (`tabs.rs`).
    // Alt+1 is this one.
    let key = crossterm::event::KeyEvent::new(code, modifiers);
    // Alt+\ (or Ctrl+Shift+P): two panes over the turn, the steering pane's
    // draft in the left one's prompt.
    if crate::tabs::is_split_key(&key) {
        let draft = lock_editor(editor)?.draft.text().to_string();
        let opened = switching_screens(|| match crate::tabs::split_from_page(display, Some((&draft, draft.len()))) {
            Ok(_) => Ok(()),
            Err(error) => {
                crate::tabs::close();
                display.uncover()?;
                Err(error)
            }
        });
        if opened.is_ok() {
            lock_editor(editor)?.draft = Draft::default();
        }
        return opened;
    }
    if let Some(target) = crate::tabs::switch_key(&key).and_then(|(group, key)| crate::tabs::target(group, key)) {
        if target == crate::tabs::Target::Main {
            return Ok(());
        }
        return switching_screens(|| match crate::tabs::go(target, display, None) {
            Ok(_) => Ok(()),
            Err(error) => {
                crate::tabs::close();
                display.uncover()?;
                Err(error)
            }
        });
    }

    // AltGr arrives as Ctrl+Alt on some platforms; its characters are text.
    if ctrl && alt
        && let K::Char(character) = code {
            return edit(editor, display, |draft| draft.insert(character));
        }

    if ctrl {
        match code {
            K::Char('c' | 'C') => {
                lock_editor(editor)?.draft = Draft::default();
                display.set_input("", 0)?;
                hard_interrupt.store(true, Ordering::SeqCst);
                return Ok(());
            }
            K::Char('d' | 'D') => {
                lock_editor(editor)?.draft = Draft::default();
                display.set_input("", 0)?;
                eof_signal.store(true, Ordering::SeqCst);
                hard_interrupt.store(true, Ordering::SeqCst);
                return Ok(());
            }
            K::Char('g' | 'G') => return open_editor(editor, display),
            K::Char('o' | 'O') => return display.toggle_detail().map(|_| ()),
            // Send now: Ctrl+S everywhere, Ctrl+Enter where the terminal
            // reports it (see enable_key_disambiguation).
            K::Char('s' | 'S') | K::Enter => return submit_draft(editor, display, submits, true),
            K::Char('u' | 'U') => return edit(editor, display, Draft::kill_line_before),
            K::Char('k' | 'K') => return edit(editor, display, Draft::kill_line_after),
            K::Char('w' | 'W') | K::Backspace => {
                return edit(editor, display, Draft::delete_word_before);
            }
            K::Char('a' | 'A') => return motion(editor, display, Draft::home),
            K::Char('e' | 'E') => return motion(editor, display, Draft::end),
            K::Char('b' | 'B') => return motion(editor, display, Draft::left),
            K::Char('f' | 'F') => return motion(editor, display, Draft::right),
            K::Char('p' | 'P') => return vertical(editor, display, true),
            K::Char('n' | 'N') => return vertical(editor, display, false),
            K::Left => return motion(editor, display, Draft::word_left),
            K::Right => return motion(editor, display, Draft::word_right),
            K::Home | K::End | K::Up | K::Down | K::Delete => {}
            // Any other Ctrl chord is not text; never insert its letter.
            _ => return Ok(()),
        }
    }

    if alt {
        // Alt+B was a word left; Alt+Left and Ctrl+Left still are.
        match code {
            K::Char(key @ ('a' | 'A' | 'x' | 'X')) => {
                answer_approval(matches!(key, 'a' | 'A'));
                return Ok(());
            }
            K::Left => return motion(editor, display, Draft::word_left),
            K::Char('f' | 'F') | K::Right => return motion(editor, display, Draft::word_right),
            K::Char('d' | 'D') => return edit(editor, display, Draft::delete_word_after),
            K::Backspace => return edit(editor, display, Draft::delete_word_before),
            K::Enter => {}
            K::Char(_) => return Ok(()),
            _ => {}
        }
    }

    match code {
        K::Enter if alt || modifiers.contains(CrosstermKeyModifiers::SHIFT) => {
            edit(editor, display, |draft| draft.insert('\n'))
        }

        K::Enter => submit_draft(editor, display, submits, false),

        K::Backspace => edit(editor, display, Draft::backspace),
        K::Delete => edit(editor, display, Draft::delete),
        K::Left => motion(editor, display, Draft::left),
        K::Right => motion(editor, display, Draft::right),
        K::Home => motion(editor, display, Draft::home),
        K::End => motion(editor, display, Draft::end),
        K::Up => vertical(editor, display, true),
        K::Down => vertical(editor, display, false),
        K::PageUp => {
            let rows = crossterm::terminal::size().map_or(24, |(_, rows)| rows as usize);
            scroll_back(display, rows.saturating_sub(5).max(1))
        }
        K::Char(character) => edit(editor, display, |draft| draft.insert(character)),

        _ => Ok(()),
    }
}

/// Enter (queue it for the agent's next safe point) or send now (`force`, or a
/// `/now` prefix), into the queue `InputReader::poll_active` drains. Sent now,
/// the draft goes into the transcript at once; queued, a status line says so
/// and the message goes in where the agent takes it.
fn submit_draft(
    editor: &SharedEditor,
    display: &ActiveDisplay,
    submits: &Sender<QueuedSubmission>,
    force: bool,
) -> io::Result<()> {
    let submitted = {
        let mut editor = lock_editor(editor)?;
        let submitted = editor.draft.take();
        editor.history.push(&submitted, true);
        submitted
    };
    send_steered(display, submits, submitted, force)
}

/// What was typed during a turn, to the agent (the steering pane's, or a pane of
/// the agent's page beside a tab).
fn send_steered(display: &ActiveDisplay, submits: &Sender<QueuedSubmission>, submitted: String, force: bool) -> io::Result<()> {
    let steered = match REMOTE_STEER.get() {
        Some(send) => {
            display.set_input("", 0)?;
            if submitted.trim().is_empty() && !force {
                return Ok(());
            }
            send(submitted, force)?
        }
        None => steer_text(display, submits, submitted, force, true)?,
    };
    match steered {
        Steered::Done => Ok(()),
        Steered::MistakeView(start) => {
            let report = open_mistake_view(display, start)?;
            let brief = report.lines().next().unwrap_or_default().to_owned();
            display.notice(&brief, &report, false)
        }
        Steered::MemoryTab(start) => {
            crate::tabs::memory_start_at(crate::tabs::MemorySection::Redactions, start);
            let target = crate::tabs::Target::View(crate::tabs::View::Memory);
            switching_screens(|| match crate::tabs::go(target, display, None) {
                Ok(_) => Ok(()),
                Err(error) => {
                    crate::tabs::close();
                    display.uncover()?;
                    Err(error)
                }
            })
        }
    }
}

/// What a message sent from a steering pane came to.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Steered {
    Done,
    /// `/mistake view`: the review window opens on the terminal it was typed
    /// on, over its paused transcript.
    MistakeView(Option<u64>),
    /// `/redact [view n]`: the memory tab, at the redactions, on the
    /// terminal it was typed on.
    MemoryTab(Option<u64>),
}

/// A message from a steering pane: queued for the agent's next safe point, or
/// sent now; `/kill` and `/mistake` act at once. `local`: typed in this
/// terminal's pane, whose draft it was (one from an attached terminal
/// leaves this pane's draft as it is).
fn steer_text(
    display: &ActiveDisplay,
    submits: &Sender<QueuedSubmission>,
    submitted: String,
    force: bool,
    local: bool,
) -> io::Result<Steered> {
    let (text, forced) = match strip_send_now(&submitted) {
        Some(rest) => (rest.to_string(), true),
        None => (submitted, force),
    };
    let commit = |text: &str, note: OperatorNote| {
        if local { display.commit_operator(text, note) } else { display.commit_remote_operator(text, note) }
    };

    if text.trim().is_empty() {
        if local {
            display.set_input("", 0)?;
        }
        // Send now with nothing typed: send what's queued.
        if forced && display.queued_steers() > 0 {
            display.operator_line("sending your queued messages now")?;
            send_submission(submits, String::new(), None, true)?;
        }
        return Ok(Steered::Done);
    }

    // /kill acts here, on the keyboard thread. As a steer it would
    // wait for the agent to finish the very call it is meant to stop.
    if let Some(report) = crate::kill::handle_command(&text) {
        commit(&text, OperatorNote::Plain)?;
        display.write_stream(&format!("{STATUS_STYLE}{}{RESET}\r\n", report.replace('\n', "\r\n")))?;
        return Ok(Steered::Done);
    }

    // /vm on|off acts here too: the machine is off limits from the next tool
    // call, not after the turn.
    if let Some(result) = crate::vm::access::handle_command(&text) {
        commit(&text, OperatorNote::Plain)?;
        let line = result.unwrap_or_else(|error| error);
        display.write_stream(&format!("{STATUS_STYLE}{line}{RESET}\r\n"))?;
        return Ok(Steered::Done);
    }

    // /redact opens the memory tab over the turn, as Alt+M does.
    if let Some(start) = crate::redact::tab_request(&text) {
        return Ok(Steered::MemoryTab(start));
    }

    // So does /mistake: flagging a call never steers or delays the turn,
    // and a flag typed while a call streams waits for that call. The review
    // window opens over the paused transcript, while the agent works on.
    if let Some(reply) = crate::mistakes::handle_command(&text, crate::mistakes::Place::Steering) {
        commit(&text, OperatorNote::Plain)?;
        let (report, problem) = match reply {
            crate::mistakes::Reply::Text(text) => (text, false),
            crate::mistakes::Reply::Error(text) => (text, true),
            crate::mistakes::Reply::View(start) => return Ok(Steered::MistakeView(start)),
        };
        // A notice of its own, never inside a line the agent is still writing.
        let brief = report.lines().next().unwrap_or_default().to_owned();
        display.notice(&brief, &report, problem)?;
        return Ok(Steered::Done);
    }

    if forced {
        commit(&text, OperatorNote::SendNow)?;
    } else {
        // Typing it moved the draft into the queue.
        if local {
            display.set_input("", 0)?;
        }
        display.queued(&text)?;
    }
    let (content, notes) = expand_attachments(&text);
    for note in notes {
        display.write_stream(&format!("{STATUS_STYLE}{note}{RESET}\r\n"))?;
    }

    display.queue_steer()?;
    if let Err(error) = send_submission(submits, content, Some(text), forced) {
        display.release_steers(1)?;
        return Err(error);
    }
    Ok(Steered::Done)
}

/// `visible`: the message as typed, shown already if `forced` (sent now),
/// or announced as queued and shown when the agent takes it.
fn send_submission(
    submits: &Sender<QueuedSubmission>,
    content: String,
    visible: Option<String>,
    forced: bool,
) -> io::Result<()> {
    let typed = visible.is_some();
    submits
        .send(QueuedSubmission {
            content,
            visible_text: visible,
            sent_at: Local::now(),
            forced,
            shown: forced || !typed,
            announced: typed && !forced,
        })
        .map_err(|_| io::Error::other("active steer queue disconnected"))
}

/// A tab opening or closing during a turn: the turn's keyboard flags are
/// taken off and put back around it, so a tab that fails part way never
/// leaves them pushed twice. Between turns the idle tab loop sets them
/// itself, and takes them off before it closes.
fn switching_screens<T>(switch: impl FnOnce() -> io::Result<T>) -> io::Result<T> {
    disable_key_disambiguation();
    let switched = switch();
    enable_key_disambiguation();
    switched
}

/// Ask the terminal to report modified keys unambiguously (the kitty
/// keyboard protocol's first level) while the steering pane is open, so
/// Ctrl+Enter and Shift+Enter arrive as themselves. Terminals without the
/// protocol ignore the request. `S0MA_KEYBOARD_PROTOCOL=off` skips it, here
/// and at the idle prompt.
fn enable_key_disambiguation() {
    if key_disambiguation_wanted() {
        let _ = execute!(
            io::stdout(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );
    }
}

fn disable_key_disambiguation() {
    if key_disambiguation_wanted() {
        let _ = execute!(io::stdout(), PopKeyboardEnhancementFlags);
    }
}

fn key_disambiguation_wanted() -> bool {
    !std::env::var("S0MA_KEYBOARD_PROTOCOL").is_ok_and(|value| value.trim().eq_ignore_ascii_case("off"))
}

/// Change the draft's text. Editing a recalled history entry makes it the
/// draft, so browsing stops.
fn edit(
    editor: &SharedEditor,
    display: &ActiveDisplay,
    change: impl FnOnce(&mut Draft),
) -> io::Result<()> {
    let draft = {
        let mut editor = lock_editor(editor)?;
        editor.history.stop_browsing();
        change(&mut editor.draft);
        editor.draft.clone()
    };
    display.set_input(draft.text(), draft.cursor())
}

/// Move the cursor without changing the text.
fn motion(
    editor: &SharedEditor,
    display: &ActiveDisplay,
    step: impl FnOnce(&mut Draft),
) -> io::Result<()> {
    let draft = {
        let mut editor = lock_editor(editor)?;
        step(&mut editor.draft);
        editor.draft.clone()
    };
    display.set_input(draft.text(), draft.cursor())
}

/// Up/Down: move between lines of a multiline draft, and past the first or
/// last line, through earlier submissions.
fn vertical(editor: &SharedEditor, display: &ActiveDisplay, up: bool) -> io::Result<()> {
    let draft = {
        let mut guard = lock_editor(editor)?;
        let ActiveEditor { draft, history, .. } = &mut *guard;
        let moved = if up { draft.up() } else { draft.down() };
        if !moved {
            if up {
                history.previous(draft);
            } else {
                history.next(draft);
            }
        }
        draft.clone()
    };
    display.set_input(draft.text(), draft.cursor())
}

/// Tab in the steering pane: complete the word at the cursor. One match is
/// inserted; several extend to their common prefix, then repeated Tab steps
/// through them. The matches are listed on the progress line.
fn complete(editor: &SharedEditor, display: &ActiveDisplay) -> io::Result<()> {
    let (draft, hint) = {
        let mut guard = lock_editor(editor)?;
        let ActiveEditor {
            draft,
            history,
            cycle,
        } = &mut *guard;
        history.stop_browsing();

        if let Some(active) = cycle.as_mut().filter(|c| c.after == *draft) {
            active.index = (active.index + 1) % active.values.len();
            draft.replace(active.start, draft.cursor(), &active.values[active.index]);
            active.after = draft.clone();
            (draft.clone(), Some(list_hint(&active.values, Some(active.index))))
        } else {
            *cycle = None;
            let found = completion::complete(draft.text(), draft.cursor());
            match found.candidates.as_slice() {
                [] if found.start == found.end => {
                    draft.insert_str("    ");
                    (draft.clone(), None)
                }
                [] => (draft.clone(), Some("no completions".to_string())),
                [only] => {
                    let mut value = only.value.clone();
                    if only.append_space {
                        value.push(' ');
                    }
                    draft.replace(found.start, found.end, &value);
                    (draft.clone(), only.description.clone())
                }
                many => {
                    let values: Vec<String> = many.iter().map(|c| c.value.clone()).collect();
                    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
                    let prefix = longest_common_prefix(&refs);
                    if prefix.len() > found.end - found.start {
                        draft.replace(found.start, found.end, &prefix);
                        (draft.clone(), Some(list_hint(&values, None)))
                    } else {
                        draft.replace(found.start, found.end, &values[0]);
                        *cycle = Some(CompletionCycle {
                            start: found.start,
                            values: values.clone(),
                            index: 0,
                            after: draft.clone(),
                        });
                        (draft.clone(), Some(list_hint(&values, Some(0))))
                    }
                }
            }
        }
    };
    display.set_input(draft.text(), draft.cursor())?;
    display.set_hint(hint)
}

/// Any key but Tab ends a completion cycle and clears its hint.
fn end_completion(editor: &SharedEditor, display: &ActiveDisplay) -> io::Result<()> {
    lock_editor(editor)?.cycle = None;
    display.set_hint(None)
}

/// `/loop  [/loops]` — paths shown by their last component.
fn list_hint(values: &[String], selected: Option<usize>) -> String {
    values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let trimmed = value.trim_end_matches('/');
            let mut name = match trimmed.rfind('/') {
                Some(slash) if !trimmed.starts_with('/') || trimmed.matches('/').count() > 1 => {
                    trimmed[slash + 1..].to_string()
                }
                _ => trimmed.to_string(),
            };
            if value.ends_with('/') {
                name.push('/');
            }
            if Some(i) == selected {
                format!("[{name}]")
            } else {
                name
            }
        })
        .collect::<Vec<_>>()
        .join("  ")
}

/// `/mistake view` during a turn: the review window, with the screen and the
/// keyboard to itself, as Ctrl+G's editor has them. The agent keeps working; its
/// output is held and laid out when the window closes. A turn that ends
/// meanwhile waits for the window: this thread is still handling a key, so
/// `end_active` waits for it.
fn open_mistake_view(display: &ActiveDisplay, start: Option<u64>) -> io::Result<String> {
    let Some(book) = crate::mistakes::book() else {
        return Ok("[mistake] not available: the mistake book is not open in this process".into());
    };
    display.suspend()?;
    disable_key_disambiguation();
    let _ = execute!(io::stdout(), DisableBracketedPaste, DisableFocusChange);
    crate::terminal::set_mouse(false);

    let closed = crate::mistakes::view::run(book, start);

    // The window's screen guard leaves raw mode off on its way out; the
    // steering pane needs it back.
    let _ = enable_raw_mode();
    let _ = execute!(io::stdout(), EnableBracketedPaste, EnableFocusChange);
    enable_key_disambiguation();
    display.resume()?;
    Ok(closed.unwrap_or_else(|error| format!("[mistake] the review window failed: {error}")))
}

/// Ctrl+G during a turn: edit the draft in the operator's editor. The
/// transcript is paused (and output held) while the editor has the screen.
fn open_editor(editor: &SharedEditor, display: &ActiveDisplay) -> io::Result<()> {
    let draft = lock_editor(editor)?.draft.clone();
    let path = draft_file();
    std::fs::write(&path, draft.text())?;

    display.suspend()?;
    disable_key_disambiguation();
    let _ = execute!(io::stdout(), DisableBracketedPaste, DisableFocusChange);
    crate::terminal::set_mouse(false);
    let _ = disable_raw_mode();
    // The editor on the shell's screen; the page is drawn again after.
    crate::page::step_out();

    let status = editor_command().arg(&path).status();

    crate::page::back();
    let _ = enable_raw_mode();
    let _ = execute!(io::stdout(), EnableBracketedPaste, EnableFocusChange);
    enable_key_disambiguation();
    display.resume()?;

    let result = match status {
        Ok(status) if status.success() => std::fs::read_to_string(&path).map(|mut text| {
            // Editors add a final newline; it isn't part of the message.
            if text.ends_with('\n') {
                text.pop();
            }
            text
        }),
        Ok(status) => Err(io::Error::other(format!("editor exited with {status}"))),
        Err(error) => Err(error),
    };
    let _ = std::fs::remove_file(&path);

    match result {
        Ok(text) => {
            let draft = {
                let mut editor = lock_editor(editor)?;
                editor.history.stop_browsing();
                editor.draft.set(&text);
                editor.draft.clone()
            };
            display.set_input(draft.text(), draft.cursor())
        }
        Err(error) => display.set_hint(Some(format!("editor: {error}"))),
    }
}

impl Drop for InputReader {
    fn drop(&mut self) {
        // If the application unwinds while an active turn owns raw
        // terminal mode, make a best effort to restore the terminal --
        // unless a panic has already put it back (`emergency_restore`):
        // ending the pane now would write over the panic's message.
        if crate::terminal::torn_down() {
            return;
        }
        self.leave_tab_modes();
        let _ = self.end_active();
    }
}

/// Alt+A or Alt+X on the page: approval mode's answer, and what came of it
/// on the page.
fn answer_approval(approve: bool) {
    match crate::approval::key(approve) {
        Ok(said) if said.is_empty() => {}
        Ok(said) => crate::terminal::say_status(&said),
        Err(said) => crate::terminal::warn_status(&said),
    }
}

#[cfg(test)]
mod steer_queue_tests {
    use super::*;

    fn queued(content: &str, forced: bool) -> QueuedSubmission {
        QueuedSubmission { content: content.into(), visible_text: None, sent_at: Local::now(), forced, shown: true, announced: false }
    }

    #[test]
    fn queued_messages_wait_for_a_safe_point_unless_one_was_sent_now() {
        let mut queue = VecDeque::new();
        assert!(!releases(SteerPoint::Safe, &queue), "nothing to hand over");

        queue.push_back(queued("check the logs too", false));
        assert!(!releases(SteerPoint::Streaming, &queue), "mid-output: it waits");
        assert!(releases(SteerPoint::Safe, &queue));

        // Send now takes everything queued before it along, at any point but
        // between a batch's calls, where it is only noticed.
        queue.push_back(queued("stop, wrong host", true));
        assert!(!releases(SteerPoint::BetweenCalls, &queue));
        assert!(sent_now_waiting(&queue));
        assert!(releases(SteerPoint::Streaming, &queue));
        let (contents, count) = release_contents(queue.drain(..).collect());
        let contents: Vec<&str> = contents.iter().map(|s| s.content.as_str()).collect();
        assert_eq!(contents, vec!["check the logs too", "stop, wrong host"]);
        assert_eq!(count, 2);
    }
}
