mod agent;
mod app;
mod browser;
mod calls;
mod board;
mod board_flow;
mod page;
mod clipboard;
mod clock;
mod completion;
mod conversations;
mod editor;
mod editor_highlight;
mod editor_live;
mod archive;
mod attachments;
mod attention;
mod context_guard;
mod context_resume;
mod edit_check;
mod exec_feed;
mod evidence;
mod goal;
mod desktop;
mod hippocampus;
mod knowledge;
mod line_edit;
mod identity;
mod input;
mod jobs;
mod kill;
mod llm;
mod markdown;
mod mcp;
mod media;
mod memory_fold;
mod memory_layers;
mod minecraft;
mod mistake_view;
mod mistakes;
mod notify;
mod observation_mask;
mod open_loops;
mod print_prompt;
mod reasoning;
mod redact;
mod runtime_context;
mod search;
mod session;
mod overview;
mod share;
mod share_sync;
mod share_backup;
mod facts;
mod transfer;
mod util;
mod crash;
mod approval;
mod recall_eval;
mod recall_push;
mod shared_files;
mod tty_guard;
mod sleep;
mod sleep_schedule;
mod streak;
mod skill_match;
mod skills;
mod strike;
mod subagent;
mod terminal;
mod transcript;
mod threads;
mod token_meter;
mod worker_context;
mod worker_fold;
mod world_model;
mod types;
mod vm;
mod vm_image;
mod vm_wake;
mod vm_terms;
mod netboxes;
mod netbox_run;
mod voice;
mod weaknesses;
mod worker_feed;
mod tabs;
mod vm_access;

use std::path::Path;

const USAGE: &str = "\
s0ma

    s0ma                        start (or attach to a running instance)
    s0ma --print-system-prompt         assemble the full prompt, to stdout, exit
    s0ma --print-system-prompt=static  only the part identical in every request
    s0ma --print-tools                 the tool array a request carries, as JSON
    s0ma --mistake-view [n]            review flagged mistakes in this terminal
                                       (beside a running s0ma, or without one)
    s0ma --skill-cutoff                the cosine cutoff for the per-turn skill
                                       match that agrees best with the model's
                                       past judgments (data/knowledge_audit.jsonl)
    s0ma connect <ssh target> [--remote <command>]
                                       a terminal onto the s0ma running on another
                                       machine, over ssh (it runs `s0ma --bridge`
                                       there; --remote names the command that runs
                                       s0ma, default `s0ma` or S0MA_REMOTE_COMMAND).
                                       /download and /upload move files with the share
    s0ma --fact-calibration            how often facts at each certainty level
                                       were later confirmed or contradicted
    s0ma --recall-eval generate [n]    make retrieval test cases from n facts
                                       (asks the model; data/recall_eval.jsonl)
    s0ma --recall-eval                 run them: recall and MRR per search arm

`=static` leaves out everything archive-derived or time-dependent: the memory
checkpoint, continuity threads and unconsolidated episodic memory. It is what belongs in front of an archive stretch
cut from an arbitrary date; the full form is what the ablation probe compares.

Options for all three:
    --with-mcp          connect MCP servers, so the [MCP] block and the mcp_*
                        tools are real (starts processes; off by default)
    --assume-vm-wake    assemble the variant a session holding the VM wake
                        channel would get, including the vm_wake_* tools
                        (this mode never holds the channel)

Environment read by both:
    S0MA_SKILL_DENY=a,b       withhold those skills from the catalog, the
                              per-turn match, and skill_read/write/delete
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ctrl+Z and Ctrl+S never stop or freeze s0ma (tty_guard.rs).
    tty_guard::install();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Argument handling before the attach check, deliberately. Attaching
    // would make this terminal a window onto a running s0ma, and
    // --print-system-prompt has to assemble a prompt here, in this process,
    // under this process's environment -- that is the whole point of it.
    // The review window works on the data folder's files, beside a running
    // s0ma or without one, so it never attaches.
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.first().map(String::as_str) == Some("--mistake-view") {
        let start = match arguments.get(1..).unwrap_or_default() {
            [] => None,
            [n] => match n.trim_start_matches('#').parse::<u64>() {
                Ok(n) => Some(n),
                Err(_) => {
                    eprint!("s0ma: --mistake-view takes a mistake number, not {n:?}\n\n{USAGE}");
                    std::process::exit(2);
                }
            },
            _ => {
                eprint!("s0ma: --mistake-view takes at most one mistake number\n\n{USAGE}");
                std::process::exit(2);
            }
        };
        return mistake_view::run_standalone(root, start);
    }

    if arguments.first().map(String::as_str) == Some("--recall-eval") {
        let data = root.join("data");
        let memory = hippocampus::Hippocampus::new(data.join("hippocampus.jsonl"))?;
        let rest: Vec<&str> = arguments[1..].iter().map(String::as_str).collect();
        let result = match rest.as_slice() {
            [] | ["run"] => recall_eval::run(&memory, &data),
            ["generate"] => recall_eval::generate(&memory, &data, 40).map(|n| format!("{n} case(s) added to data/recall_eval.jsonl\n")),
            ["generate", n] => match n.parse::<usize>() {
                Ok(n) if n > 0 => recall_eval::generate(&memory, &data, n).map(|n| format!("{n} case(s) added to data/recall_eval.jsonl\n")),
                _ => Err(format!("--recall-eval generate takes a number of facts, not {n:?}").into()),
            },
            _ => Err("usage: s0ma --recall-eval [run | generate [n]]".into()),
        };
        match result {
            Ok(report) => print!("{report}"),
            Err(error) => {
                eprintln!("s0ma: {error}");
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    if arguments.first().map(String::as_str) == Some("--fact-calibration") {
        match facts::calibration_report(&root.join("data").join("world_model.sqlite3")) {
            Ok(report) => print!("{report}"),
            Err(error) => {
                eprintln!("s0ma: {error}");
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    if arguments.first().map(String::as_str) == Some("--skill-cutoff") {
        let audit = root.join("data").join("knowledge_audit.jsonl");
        let bar = std::env::var("S0MA_KNOW_THRESHOLD_SKILL_MATCH")
            .or_else(|_| std::env::var("S0MA_KNOW_THRESHOLD"))
            .ok()
            .and_then(|raw| raw.trim().parse::<f64>().ok())
            .unwrap_or(0.5);
        match skill_match::cutoff_report(&audit, bar) {
            Ok(report) => print!("{report}"),
            Err(error) => {
                eprintln!("s0ma: {error}");
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    // The far end of `s0ma connect`: this process's input and output are the
    // protocol, so nothing else may be printed.
    if arguments.first().map(String::as_str) == Some("--bridge") {
        if let Err(error) = session::bridge(&root.join("data")) {
            eprintln!("s0ma: {error}");
            std::process::exit(1);
        }
        return Ok(());
    }
    // A terminal onto the s0ma running on another machine, over SSH.
    if arguments.first().map(String::as_str) == Some("connect") {
        let mut target = None;
        let mut remote = std::env::var("S0MA_REMOTE_COMMAND").ok().filter(|c| !c.trim().is_empty()).unwrap_or_else(|| "s0ma".into());
        let mut rest = arguments[1..].iter();
        while let Some(word) = rest.next() {
            match word.as_str() {
                "--remote" => match rest.next() {
                    Some(command) => remote = command.clone(),
                    None => {
                        eprint!("s0ma: --remote needs the command that runs s0ma there\n\n{USAGE}");
                        std::process::exit(2);
                    }
                },
                other if target.is_none() && !other.starts_with('-') => target = Some(other.to_string()),
                other => {
                    eprint!("s0ma connect: unrecognised argument {other:?}\n\n{USAGE}");
                    std::process::exit(2);
                }
            }
        }
        let Some(target) = target else {
            eprint!("s0ma connect: which machine? (an ssh target: host, user@host, or a Host from ~/.ssh/config)\n\n{USAGE}");
            std::process::exit(2);
        };
        let remote = match session::open_remote(&target, &remote) {
            Ok(remote) => remote,
            Err(error) => {
                eprintln!("s0ma connect: {error}");
                std::process::exit(1);
            }
        };
        let _page = page::enter();
        return session::attach_remote(&root.join("data"), remote);
    }

    let mut print_prompt = false;
    let mut options = print_prompt::Options::default();
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--print-system-prompt" => {
                print_prompt = true;
                options.mode = print_prompt::Mode::Full;
            }
            "--print-system-prompt=static" => {
                print_prompt = true;
                options.mode = print_prompt::Mode::Static;
            }
            "--print-system-prompt=full" => {
                print_prompt = true;
                options.mode = print_prompt::Mode::Full;
            }
            "--print-tools" => {
                print_prompt = true;
                options.tools = true;
            }
            "--with-mcp" => options.with_mcp = true,
            "--assume-vm-wake" => options.assume_vm_wake = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => {
                eprint!("s0ma: unrecognised argument {other:?}\n\n{USAGE}");
                std::process::exit(2);
            }
        }
    }
    if print_prompt {
        return print_prompt::run(root, &options);
    }
    if options.with_mcp || options.assume_vm_wake {
        eprint!("s0ma: those options apply to --print-system-prompt only\n\n{USAGE}");
        std::process::exit(2);
    }

    // s0ma's page (the alternate screen, the tab bar on top), for this s0ma
    // or a window onto a running one; the shell's screen comes back on the
    // way out.
    let _page = page::enter();

    // Before anything else touches the archive, the self-model or the sqlite
    // stores: if an s0ma is already running, this terminal becomes a window onto
    // it instead of a second instance sharing its memory. That is also what
    // makes reconnecting after a dropped SSH session rejoin the turn already in
    // progress rather than starting a rival one.
    if session::attach_if_running(&root.join("data"))? {
        return Ok(());
    }

    terminal::banner();

    let mut app = app::App::load(root, app::RuntimeMode::Conversation)?;
    let ran = app.run();
    // The shell's screen, with the conversation's last rows under what it
    // had; an error that ends s0ma is printed after them.
    terminal::leave_page();
    ran
}
