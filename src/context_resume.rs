//! The working context across a restart.
//!
//! A restart used to rebuild the context from memory (the checkpoint, the
//! recent archive, the gists), so the live turns came back as a summary and the
//! engine read the whole prompt again. ds41d now keeps every prompt it has
//! read on disk (change 81) and picks up a prompt that extends one of them
//! without prefilling it again. That only helps if the prompt is the same one.
//! So s0ma saves its working context, exactly as it would send it, whenever it
//! is idle and the context changed, and at start takes it back up when nothing
//! it was built from has moved since. The first request after a restart is then
//! the request that would have come next without one.
//!
//! What would make the saved context wrong, and so sends a restart back to
//! rebuilding it:
//! - the base system prompt changed (a new build, a new identity, the wake
//!   channel turned on or off): the saved context carries the old one;
//! - the skills or MCP servers changed since the context was built: their
//!   catalogs in the system message are refreshed only by a rebuild, and a
//!   restart was how a new skill or server got picked up;
//! - a deep sleep finished after the save: the context it rebuilt was never saved;
//! - the gists or NOT SETTLED items moved after the save (a fold that the
//!   restart cut short): the saved live turns would sit beside their own gists;
//! - the window being served is too small for it.
//!
//! A redaction or a job's close takes content out of every store: it deletes
//! the saved context too (`forget`), and the running context, which still
//! holds the content until it is next rebuilt, is not saved again until then.
//!
//! Archive events after the context last caught up (a turn the restart cut
//! short, work finished in the background) are not a reason to rebuild: they
//! are named in the first input's header instead, where s0ma can read them
//! with archive_get.
//!
//! Like the archive, the file is the conversation itself.

use crate::calls::Span;
use crate::memory_layers::{Level, MemoryLayers};
use crate::types::{Message, StoredMessage};
use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const FILE: &str = "working_context.json";
const VERSION: u32 = 2;

/// Where it is kept, in the data folder.
pub fn path(data_dir: &Path) -> PathBuf {
    data_dir.join(FILE)
}

/// What the gists and NOT SETTLED items were when the context was saved: a
/// fold changes both, and a saved context from before one would hold the
/// turns it folded beside their gists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayersMark {
    next_item: i64,
    gists: Vec<i64>,
}

pub fn layers_mark(layers: &MemoryLayers) -> Result<LayersMark> {
    let mut gists: Vec<i64> = layers.level(Level::Episode)?.iter().map(|gist| gist.id).collect();
    gists.extend(layers.level(Level::Shape)?.iter().map(|gist| gist.id));
    Ok(LayersMark { next_item: layers.next_item_id()?, gists })
}

pub fn prompt_hash(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

/// The skills and MCP catalogs as a rebuild would put them in the system
/// message now. Recorded when a context is built (`Built::catalogs`): a
/// running context keeps the ones it was built with.
pub fn catalogs_hash() -> Result<String> {
    let skills = crate::skills::runtime_catalog_context()?;
    let mcp = crate::mcp::runtime_context();
    Ok(prompt_hash(&format!("{skills}\u{0}{mcp}")))
}

/// Bumped by `forget`: a context built before a purge still holds what it took
/// out of the archive, and is not saved again.
static GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn generation() -> u64 {
    GENERATION.load(Ordering::SeqCst)
}

/// A redaction or a job's close took content out of the archive: the saved
/// context goes, and the running one is not saved again until it is rebuilt
/// (`generation`).
pub fn forget(data_dir: &Path) {
    GENERATION.fetch_add(1, Ordering::SeqCst);
    for file in [path(data_dir), data_dir.join(format!("{FILE}.partial"))] {
        if let Err(error) = std::fs::remove_file(&file)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            crate::terminal::warn(&format!(
                "[context] the saved working context ({}) could not be deleted: {error}; delete it by hand",
                file.display()
            ));
        }
    }
}

/// What the running context was built with, kept beside it: set when it is
/// built (at start, by a sleep, by `/compact`), moved on as turns end.
#[derive(Debug, Clone, PartialEq)]
pub struct Built {
    /// `catalogs_hash` when it was built.
    pub catalogs: String,
    /// `generation` when it was built.
    pub generation: u64,
    /// The last archive event it has caught up with: the end of its last
    /// turn, or its rebuild. Events after it are named after a restart.
    pub through: u64,
}

/// What the context is built from, now: compared with what it was when the
/// context was saved.
#[derive(Debug, Clone, PartialEq)]
pub struct Basis {
    /// `prompt_hash` of the base system prompt (`app::base_system_prompt`).
    pub base_prompt: String,
    /// `catalogs_hash` now.
    pub catalogs: String,
    /// The latest deep sleep's checkpoint: through which archive event.
    pub sleep_through: Option<u64>,
    pub layers: LayersMark,
    /// The archive's last event.
    pub archive_through: u64,
    /// The window being served, if the server said.
    pub window: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Saved {
    version: u32,
    saved_at: DateTime<Local>,
    base_prompt: String,
    catalogs: String,
    sleep_through: Option<u64>,
    layers: LayersMark,
    /// `Built::through`.
    through: u64,
    /// The compaction checkpoint the context was last rebuilt from
    /// (`App::runtime_through`).
    runtime_through: u64,
    /// The window it was last measured against.
    window: Option<usize>,
    /// The system message's labelled spans (`calls::register_system_spans`),
    /// which the call store saves with every call.
    spans: Option<Vec<Span>>,
    messages: Vec<StoredMessage>,
}

/// The running context, as `save` takes it.
pub struct Running<'a> {
    pub messages: &'a [Message],
    pub runtime_through: u64,
    pub built: &'a Built,
    pub window: Option<usize>,
}

/// The context taken back up.
#[derive(Debug)]
pub struct Resumed {
    pub messages: Vec<Message>,
    pub runtime_through: u64,
    pub built: Built,
    pub window: Option<usize>,
    pub saved_at: DateTime<Local>,
    /// Archive events after it last caught up, not in it.
    pub missing: Option<(u64, u64)>,
}

#[derive(Debug)]
pub enum Outcome {
    Resumed(Resumed),
    /// A saved context that can't be used, and why.
    Rebuild(String),
    /// Nothing saved (or resuming turned off).
    Nothing,
}

/// Whether a context is small enough for `window`: one saved on a long window
/// (it folds by 650k) and taken up on a short one could be past what that
/// window can fold back down. Estimated from its size, as the fold planner does.
pub fn fits(messages: &[Message], window: usize) -> std::result::Result<(), String> {
    let estimated: usize = messages.iter().map(crate::memory_layers::estimate_message_tokens).sum();
    let ceiling = crate::context_guard::FoldSchedule::for_window(window).ceiling;
    if estimated > ceiling {
        return Err(format!(
            "it is about {estimated} tokens, past where this {window}-token window folds ({ceiling})"
        ));
    }
    Ok(())
}

/// `S0MA_RESUME_CONTEXT=0` rebuilds the context at every start, as before.
pub fn enabled() -> bool {
    std::env::var("S0MA_RESUME_CONTEXT").map_or(true, |value| value.trim() != "0")
}

/// Tells whether the context changed since it was last saved, without
/// writing it: called every time s0ma goes idle.
pub fn fingerprint(running: &Running) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hasher.write_u64(running.runtime_through);
    hasher.write_u64(running.built.through);
    hasher.write_u64(running.window.unwrap_or(0) as u64);
    for message in running.messages {
        message.hash_all(&mut hasher);
    }
    hasher.finish()
}

/// Write the context, replacing the last one in one step: written beside it,
/// flushed to disk, then renamed over it, so a crash or a full disk leaves the
/// last one whole.
pub fn save(data_dir: &Path, running: &Running, basis: &Basis) -> Result<()> {
    let saved = Saved {
        version: VERSION,
        saved_at: Local::now(),
        base_prompt: basis.base_prompt.clone(),
        catalogs: running.built.catalogs.clone(),
        sleep_through: basis.sleep_through,
        layers: basis.layers.clone(),
        through: running.built.through,
        runtime_through: running.runtime_through,
        window: running.window,
        spans: running.messages.first().and_then(|system| crate::calls::spans_for(&system.content)),
        messages: running.messages.iter().map(StoredMessage::from).collect(),
    };
    let target = path(data_dir);
    let partial = data_dir.join(format!("{FILE}.partial"));
    let written = (|| -> Result<()> {
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&partial)?);
        serde_json::to_writer(&mut writer, &saved)?;
        // The last of it written out here, where a full disk is an error,
        // not in a drop that would ignore it.
        let file = writer.into_inner().map_err(|error| error.into_error())?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&partial);
        return Err(error);
    }
    std::fs::rename(&partial, &target)?;
    Ok(())
}

/// The saved context, if there is one and nothing it was built from has
/// moved since (`basis` is what they are now).
pub fn load(data_dir: &Path, basis: &Basis) -> Outcome {
    if !enabled() {
        // Turned off: an old one is not left to come back much later.
        let _ = std::fs::remove_file(path(data_dir));
        return Outcome::Nothing;
    }
    let file = match std::fs::File::open(path(data_dir)) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Outcome::Nothing,
        Err(error) => return Outcome::Rebuild(format!("it could not be read: {error}")),
    };
    let saved: Saved = match serde_json::from_reader(std::io::BufReader::new(file)) {
        Ok(saved) => saved,
        Err(error) => return Outcome::Rebuild(format!("it could not be read: {error}")),
    };
    check(saved, basis)
}

fn check(saved: Saved, basis: &Basis) -> Outcome {
    if saved.version != VERSION {
        return Outcome::Rebuild(format!("it was saved in another format (version {})", saved.version));
    }
    if saved.base_prompt != basis.base_prompt {
        return Outcome::Rebuild("the system prompt changed since".into());
    }
    if saved.catalogs != basis.catalogs {
        return Outcome::Rebuild("the skills or MCP servers changed since it was built".into());
    }
    if saved.sleep_through != basis.sleep_through {
        return Outcome::Rebuild("a deep sleep finished after it was saved".into());
    }
    if saved.layers != basis.layers {
        return Outcome::Rebuild("the memory gists changed after it was saved (a fold the restart cut short)".into());
    }
    if saved.messages.is_empty() {
        return Outcome::Rebuild("it is empty".into());
    }
    let messages: Vec<Message> = saved.messages.into_iter().map(Message::from).collect();
    if messages[0].role != "system" {
        return Outcome::Rebuild("it does not begin with the system prompt".into());
    }
    // Only a window that changed is checked: an idle context can sit past
    // the ceiling until the next turn folds it.
    if let Some(window) = basis.window
        && saved.window.is_some_and(|before| before != window)
        && let Err(why) = fits(&messages, window)
    {
        return Outcome::Rebuild(why);
    }
    if let Some(spans) = saved.spans {
        crate::calls::register_system_spans(&messages[0].content, spans);
    }
    let missing = (basis.archive_through > saved.through).then_some((saved.through + 1, basis.archive_through));
    Outcome::Resumed(Resumed {
        messages,
        runtime_through: saved.runtime_through,
        built: Built { catalogs: saved.catalogs, generation: generation(), through: saved.through },
        window: saved.window,
        saved_at: saved.saved_at,
        missing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis() -> Basis {
        Basis {
            base_prompt: prompt_hash("You are s0ma."),
            catalogs: prompt_hash("skills\u{0}"),
            sleep_through: Some(40),
            layers: LayersMark { next_item: 3, gists: vec![1, 2] },
            archive_through: 120,
            window: Some(262_144),
        }
    }

    fn built() -> Built {
        Built { catalogs: prompt_hash("skills\u{0}"), generation: 0, through: 120 }
    }

    fn conversation() -> Vec<Message> {
        let mut answer = Message::assistant("the pool, I think".into()).with_reasoning("checking the lease".into());
        answer.tool_calls = Some(vec![crate::types::ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: crate::types::FunctionCall { name: "vm_exec".into(), arguments: "{\"command\":\"ls\"}".into() },
        }]);
        vec![
            Message::system("You are s0ma. [runtime context]".into()),
            Message::user("[memory]".into()),
            Message::user("why is it slow?".into()).by_operator(),
            answer,
            Message::tool("c1".into(), "a b c".into()),
        ]
    }

    fn dir() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "s0ma-context-resume-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn save_as(dir: &Path, messages: &[Message], runtime_through: u64, built: &Built) {
        let running = Running { messages, runtime_through, built, window: Some(262_144) };
        save(dir, &running, &basis()).unwrap();
    }

    #[test]
    fn the_context_comes_back_as_it_would_have_been_sent() {
        let dir = dir();
        let messages = conversation();
        let spans = vec![Span { label: "base".into(), start: 0, end: 12 }];
        crate::calls::register_system_spans(&messages[0].content, spans.clone());
        save_as(&dir, &messages, 77, &built());
        let Outcome::Resumed(resumed) = load(&dir, &basis()) else { panic!("not resumed") };
        assert_eq!(resumed.runtime_through, 77);
        assert_eq!(resumed.missing, None);
        assert_eq!(resumed.built.through, 120);
        // Byte for byte the request it was, and what the request never
        // carries (whose words they were) too.
        assert_eq!(serde_json::to_string(&resumed.messages).unwrap(), serde_json::to_string(&messages).unwrap());
        assert!(resumed.messages[2].is_operator_input() && !resumed.messages[1].is_operator_input());
        let again = Running { messages: &resumed.messages, runtime_through: 77, built: &resumed.built, window: resumed.window };
        let first = Running { messages: &messages, runtime_through: 77, built: &built(), window: Some(262_144) };
        assert_eq!(fingerprint(&again), fingerprint(&first));
        assert!(!dir.join(format!("{FILE}.partial")).exists());
        // The spans the call store saves come back with it.
        let saved: Saved = serde_json::from_str(&std::fs::read_to_string(path(&dir)).unwrap()).unwrap();
        assert_eq!(saved.spans, Some(spans));
    }

    #[test]
    fn events_after_it_last_caught_up_are_named_not_a_reason_to_rebuild() {
        let dir = dir();
        save_as(&dir, &conversation(), 0, &Built { through: 110, ..built() });
        let later = Basis { archive_through: 131, ..basis() };
        let Outcome::Resumed(resumed) = load(&dir, &later) else { panic!("not resumed") };
        assert_eq!(resumed.missing, Some((111, 131)));
        // Saved again before anything caught up: still named.
        save_as(&dir, &resumed.messages, 0, &resumed.built);
        let Outcome::Resumed(again) = load(&dir, &later) else { panic!("not resumed") };
        assert_eq!(again.missing, Some((111, 131)));
    }

    #[test]
    fn a_context_built_from_something_that_moved_is_rebuilt() {
        let dir = dir();
        save_as(&dir, &conversation(), 0, &built());
        let rebuilt = |basis: Basis| match load(&dir, &basis) {
            Outcome::Rebuild(why) => why,
            other => panic!("{other:?}"),
        };
        assert!(rebuilt(Basis { base_prompt: prompt_hash("You are s0ma, v2."), ..basis() }).contains("system prompt"));
        assert!(rebuilt(Basis { catalogs: prompt_hash("skills, one more\u{0}"), ..basis() }).contains("skills"));
        assert!(rebuilt(Basis { sleep_through: Some(90), ..basis() }).contains("deep sleep"));
        let folded = LayersMark { next_item: 4, gists: vec![1, 2, 3] };
        assert!(rebuilt(Basis { layers: folded, ..basis() }).contains("gists"));
        std::fs::write(path(&dir), "{ not json").unwrap();
        assert!(rebuilt(basis()).contains("could not be read"));
    }

    #[test]
    fn a_context_too_big_for_a_changed_window_is_not_taken_up() {
        let mut messages = conversation();
        assert!(fits(&messages, 262_144).is_ok());
        // Under the window, past where it folds: refused all the same.
        messages.push(Message::user("x".repeat(4 * 240_000)));
        let why = fits(&messages, 262_144).unwrap_err();
        assert!(why.contains("past where this 262144-token window folds"), "{why}");
        assert!(fits(&messages, 1_048_576).is_ok(), "a long window folds by 650k");
        // Saved on a 1M window, taken up on a 262k one: rebuilt. The same
        // window: taken up (the next turn folds it).
        let dir = dir();
        let running = Running { messages: &messages, runtime_through: 0, built: &built(), window: Some(1_048_576) };
        save(&dir, &running, &basis()).unwrap();
        assert!(matches!(load(&dir, &basis()), Outcome::Rebuild(why) if why.contains("past where")));
        assert!(matches!(load(&dir, &Basis { window: Some(1_048_576), ..basis() }), Outcome::Resumed(_)));
        assert!(matches!(load(&dir, &Basis { window: None, ..basis() }), Outcome::Resumed(_)), "server not up yet");
        // Past the ceiling on the window it was saved on: an idle context can
        // be, until the next turn folds it. Taken up.
        let running = Running { messages: &messages, runtime_through: 0, built: &built(), window: Some(262_144) };
        save(&dir, &running, &basis()).unwrap();
        assert!(matches!(load(&dir, &basis()), Outcome::Resumed(_)));
    }

    #[test]
    fn forgetting_deletes_it_and_moves_the_generation() {
        let dir = dir();
        save_as(&dir, &conversation(), 0, &built());
        let before = generation();
        forget(&dir);
        assert!(generation() > before);
        assert!(!path(&dir).exists());
        assert!(matches!(load(&dir, &basis()), Outcome::Nothing));
    }

    #[test]
    fn nothing_saved_is_nothing() {
        assert!(matches!(load(&dir(), &basis()), Outcome::Nothing));
    }

    #[test]
    fn the_fingerprint_follows_any_change() {
        let messages = conversation();
        let built = built();
        let of = |messages: &[Message], runtime_through: u64| {
            fingerprint(&Running { messages, runtime_through, built: &built, window: None })
        };
        let mut changed = messages.clone();
        changed[3].reasoning_content = Some("checking the lease again".into());
        assert_ne!(of(&messages, 0), of(&changed, 0));
        assert_ne!(of(&messages, 0), of(&messages, 1));
        let mut by_operator = messages.clone();
        by_operator[1] = by_operator[1].clone().by_operator();
        assert_ne!(of(&messages, 0), of(&by_operator, 0));
        let caught_up = Built { through: 121, ..built.clone() };
        assert_ne!(of(&messages, 0), fingerprint(&Running { messages: &messages, runtime_through: 0, built: &caught_up, window: None }));
    }
}
