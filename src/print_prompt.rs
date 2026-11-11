//! `--print-system-prompt`: assemble the system prompt and write it to stdout,
//! without contacting the model.
//!
//! # Why this exists
//!
//! The capability eval (§6) runs tasks under ablation conditions — a named
//! skill withheld (`S0MA_SKILL_DENY`) — and compares the pass rate against the
//! unablated run. (The other condition, the `[KNOWN WEAKNESSES]` block
//! withheld, went with the block on 2026-10-26: every prompt is without it.) The cell that matters is a task
//! that fails without the prompt support before training and passes without it
//! after, because that is what "the behaviour moved into the weights" means.
//!
//! All of which rests on the switch having actually taken effect. A switch
//! that is read from the wrong place, or parsed and never applied, does not
//! crash: the run completes, the numbers look ordinary, and the ablation
//! quietly compares the default prompt against the default prompt. Nothing
//! downstream can detect it, because there is no signature to detect — the
//! result is simply a measurement that was never taken, reported as one that
//! was.
//!
//! So the eval runner assembles the prompt twice through this mode, once
//! clean and once with the condition's environment set, and refuses to run
//! unless the expected block actually disappeared. This is the hook that makes
//! that check possible.
//!
//! # What it guarantees
//!
//! * **stdout is the prompt and nothing else.** Harness notices normally go to
//!   stdout; `terminal::reserve_stdout` sends them to stderr for the duration,
//!   so a caller can compare two runs byte for byte. They are diverted rather
//!   than silenced: "a skill was skipped" changes what the prompt contains and
//!   the caller needs to see it.
//! * **The same code assembles it.** The prompt comes from
//!   `app::base_system_prompt` and `runtime_context::build`, which is what a
//!   real start uses. A second implementation would drift, and a mode that
//!   prints a prompt nobody is served answers the wrong question.
//! * **Nothing is written.** The archive is opened read-only
//!   (`Hippocampus::open_read_only`), the attach socket is not bound, and no
//!   worker, watcher or reader is started. It is safe to run beside a live
//!   s0ma, which matters because an eval probe usually does.
//!
//! # What it cannot promise
//!
//! Two blocks depend on runtime state this mode does not have, and both are
//! reported on stderr rather than papered over:
//!
//! * the attention-policy and wake blocks differ depending on whether this
//!   process holds the VM wake channel, which it never does. `--assume-vm-wake`
//!   assembles the other variant.
//! * the `[MCP]` block lists servers that connected. Connecting them starts
//!   processes, so it is off unless `--with-mcp` is given; without it the block
//!   reads as it would with no servers.
//!
//! Neither affects an ablation comparison, where both sides are assembled the
//! same way and any constant difference cancels. Both matter if you are
//! reading the output to see what the model is actually told, which is the
//! other reason to run this.

use std::path::Path;

use crate::hippocampus::Hippocampus;
use crate::identity::Identity;
use crate::llm::ChatOptions;
use crate::runtime_context;

/// Which prompt to assemble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Everything a real start puts in front of the conversation, memory
    /// included. What the ablation probe compares.
    Full,
    /// Only the part that is identical in every request: no memory, no
    /// threads, no clock. What a frozen eval item set
    /// should carry in front of an archive stretch cut from any date.
    Static,
}

pub struct Options {
    pub mode: Mode,
    /// Print the tool definitions as the JSON array the request carries,
    /// instead of the prompt.
    pub tools: bool,
    /// Connect MCP servers so the `[MCP]` block is real. Off by default: it
    /// starts processes, and an eval probe runs this twice in a row.
    pub with_mcp: bool,
    /// Assemble the variant that a process holding the VM wake channel would
    /// get. This mode never holds it, so the honest default is the other one.
    pub assume_vm_wake: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { mode: Mode::Full, tools: false, with_mcp: false, assume_vm_wake: false }
    }
}

pub fn run(root: &Path, options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    // First, before anything can print: from here the only thing allowed on
    // stdout is the artefact.
    crate::terminal::reserve_stdout();

    let data_dir = root.join("data");

    // The wake channel decides three blocks of prompt text and one group of
    // tools. This process never holds it, so the caller has to say which
    // variant it wants. `init` only reads a queue file; nothing is written.
    if options.assume_vm_wake {
        crate::vm_wake::init(&data_dir);
    }
    if options.with_mcp {
        crate::mcp::init(root, &data_dir);
    }

    if options.tools {
        return print_tools(root, options);
    }

    let identity = Identity::load(&data_dir)?;
    let system_prompt = crate::app::base_system_prompt(root, &identity, options.assume_vm_wake)?;

    // The [SKILLS] block and, through the catalog, whatever S0MA_SKILL_DENY
    // withholds. `skill_match::init` is deliberately skipped: the per-turn
    // match does not contribute to this block, and it would open the vector
    // store a live s0ma may be using.
    crate::skills::init(&root.join("skills"))?;

    if !options.with_mcp {
        eprintln!(
            "[print-prompt] MCP servers not connected; the [MCP] block and the mcp_* \
             tools are absent. Pass --with-mcp to connect them."
        );
    }
    if !options.assume_vm_wake {
        eprintln!(
            "[print-prompt] assembled without the VM wake channel; the attention policy \
             and wake blocks differ from a session that holds it, and the vm_wake_* \
             tools are absent. Pass --assume-vm-wake for that variant."
        );
    }

    let memory = Hippocampus::open_read_only(data_dir.join("hippocampus.jsonl"))?;

    let text = match options.mode {
        Mode::Static => static_prompt(&system_prompt, &memory, options)?,
        Mode::Full => full_prompt(&system_prompt, &memory, &data_dir)?,
    };

    let withheld = crate::skills::denied_names();
    if !withheld.is_empty() {
        eprintln!("[print-prompt] skills withheld (S0MA_SKILL_DENY): {}", withheld.join(", "));
    }
    // An eval runner still setting it for a `no_weakness_context` condition
    // would otherwise compare two identical prompts and conclude the switch
    // did nothing; say why.
    if std::env::var_os("S0MA_WEAKNESS_CONTEXT").is_some() {
        eprintln!(
            "[print-prompt] S0MA_WEAKNESS_CONTEXT is no longer read: the [KNOWN WEAKNESSES] block was \
             removed, so every prompt is already without it"
        );
    }

    print!("{text}");
    use std::io::Write;
    std::io::stdout().flush()?;
    Ok(())
}

/// Everything a real start assembles, memory and all.
fn full_prompt(
    system_prompt: &str,
    memory: &Hippocampus,
    data_dir: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let threads = crate::threads::ThreadStore::new(data_dir.join("threads.json"))?;
    let sleep_state = crate::sleep::SleepStateStore::new(data_dir)?;
    let working_checkpoint = crate::sleep::continuous::runtime_checkpoint(memory)?;
    let checkpoint = memory.latest_checkpoint()?;
    let projections = sleep_state
        .current_projections(checkpoint.as_ref(), &threads)?
        .unwrap_or_default();

    let runtime = runtime_context::build(
        runtime_context::ContextMode::Rotated,
        system_prompt,
        memory,
        working_checkpoint.as_ref(),
        &projections,
        &ChatOptions::normal(),
    )?;
    Ok(runtime.system_context)
}

/// Only what is identical in every request.
///
/// Assembled directly rather than by building the runtime context and
/// subtracting from it. Subtraction would mean deleting blocks by pattern
/// after the fact, and a block whose header changed would start surviving the
/// filter silently — which, in a frozen item set, is a stray dynamic line
/// nobody notices until the scores are already wrong.
///
/// Left out, and why:
///
/// * `LONG-TERM MEMORY CHECKPOINT`, `PERSISTENT CONTINUITY THREADS` and
///   `UNCONSOLIDATED EPISODIC MEMORY` — all three are archive-derived text,
///   and the last carries `[memory <id> | <timestamp> | <speaker>]` headers
///   for the same events an eval stretch is cut from. A line present in both
///   is a cloze answer that is no longer unique, and nothing downstream would
///   report it.
/// * `[MCP]` — which servers answered just now is machine state. `--with-mcp`
///   includes it for anyone who wants deployment fidelity over invariance.
fn static_prompt(
    system_prompt: &str,
    memory: &Hippocampus,
    options: &Options,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut text = system_prompt.to_string();
    text.push_str(&crate::skills::runtime_catalog_context()?);
    text.push_str(&crate::world_model::runtime_catalog(memory)?);
    if options.with_mcp {
        text.push_str(&crate::mcp::runtime_context());
    }

    eprintln!(
        "[print-prompt] static: base prompt + identity, [SKILLS], [WORLD MODELS]{}",
        if options.with_mcp { ", [MCP]" } else { "" }
    );
    eprintln!(
        "[print-prompt] omitted: memory checkpoint, continuity threads, unconsolidated \
         episodic memory{}",
        if options.with_mcp { "" } else { ", [MCP]" }
    );
    eprintln!(
        "[print-prompt] note: the two catalogues listed above are stable between runs but \
         do change as skills and world models are written. Rebuild the item set if they \
         move, or delete those blocks from the file."
    );
    Ok(text)
}

/// The tool array exactly as a request carries it.
fn print_tools(root: &Path, options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    // skill_write/skill_delete are in the list, and their definitions do not
    // depend on the catalogue, but init must have run for the module to have
    // a root at all.
    crate::skills::init(&root.join("skills"))?;
    let tools = crate::app::tool_catalog();
    let wake = tools.iter().filter(|t| t.function.name.starts_with("vm_wake")).count();
    let mcp = tools.iter().filter(|t| crate::mcp::handles(&t.function.name)).count();
    eprintln!(
        "[print-prompt] {} tool definitions; vm_wake_*: {}, mcp: {}",
        tools.len(),
        if options.assume_vm_wake { wake.to_string() } else { "0 (omitted; --assume-vm-wake)".into() },
        if options.with_mcp { mcp.to_string() } else { "0 (omitted; --with-mcp)".into() },
    );
    if !options.assume_vm_wake || !options.with_mcp {
        eprintln!(
            "[print-prompt] warning: a tool the archive calls but this list omits is \
             off-distribution for anything replaying that archive."
        );
    }
    println!("{}", serde_json::to_string_pretty(&tools)?);
    Ok(())
}
