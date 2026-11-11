# Enactive AI

A self-hosted harness for working alongside a local language model, built for keeping the operator in the loop rather than for unattended runs.

The agent has a machine of its own: a virtual machine where it runs commands, browses and edits files. You work in that machine with it, in the same terminals and the same folder, and you can watch, step in or take over at any point. It keeps a long-term memory that it consolidates while idle, so it carries on from one day to the next instead of starting fresh each session.

Written in Rust, with a terminal interface built on crossterm.

## What it does

**Works where you can see it**
- **Shared terminals.** Each is a tmux session in the VM. The agent's commands are typed into them with a one-line reason, and you can type alongside it. Ctrl+\ takes the keyboard. In confirm mode, each command waits at an editable prompt for you to press Enter.
- **A shared folder.** It is kept in sync both ways with a copy on your machine and backed up every few hours. A built-in editor marks the lines the agent changed since you last looked.
- **Approval mode.** When it's on, every command, in every shell, the agent's and its workers', waits for you. So does every worker it starts. Pressing Alt+A twice approves; Alt+X declines. The setting persists across restarts, and if the saved setting can't be read, approval mode comes back on.
- **Steering.** Anything you type while a turn runs joins it at the next safe point. `/kill` stops a single tool call or worker without ending the turn.

**Remembers**
- **Archive.** Every event goes into an append-only JSONL archive, which is the single source of truth. Keyword (SQLite FTS5) and embedding indexes are built from it and can be rebuilt at any time.
- **Graded working memory.** Older turns are folded into first-person summaries layer by layer, so a long session stays inside the context window without a hard cut.
- **Durable knowledge:**
  - facts, each with a certainty;
  - episodes;
  - continuity threads;
  - versioned world models;
  - skills the agent writes and revises itself.
- **Knowledge pass.** Before each turn, the base model is asked where the answer is likely to be found. The facts, episodes and skills that match are pushed into the context.
- **Sleep.** A scheduled consolidation cycle, e.g. `/sleep set 0200`, turns the day's events into episodes and facts.

**Investigates**
- **Boards** for problems with no known route: leads, findings with evidence from the archive, and a workflow that ends with your review and an independent verification.
- **Workers (subagents).** Each runs in its own terminal under its own account, uses its own engine lane, and has a live feed in its own tab.
- **Goals** (`/goal`). Objectives the agent keeps working toward from turn to turn.
- **Wake channel.** Any condition in the VM can start a turn.

**Learns from corrections**
- **Call store.** Every model call is saved at the source.
- **`/mistake`.** Flags the step where a mistake started, using the exact context of that step, so the correction can be used for training. A review window lets you check and edit corrections before they're used.
- **Weakness log.** The agent can report failures it notices. You decide what happens to them.

**Interface**
- **Tabs.**
  - the conversation;
  - each worker;
  - each terminal;
  - the shared folder;
  - the editor;
  - memory;
  - projects and boards;
  - an overview of settings and state.

  Everything can be clicked, and every key still works.
- **One instance, many terminals.** Starting it again attaches to the running instance. From another machine, `connect` attaches over ssh, and `/download` and `/upload` move files through the shared folder.
- **Tools:** VM commands, the browser and desktop inside the VM, web search, viewing images, and MCP servers configured in `mcp.json`.

## Architecture

```
                                  ┌─ harness process ──────────────────────────┐        ┌─ VM (libvirt/QEMU) ───────┐
 your terminals ──attach socket──▶│ tabs ─▶ knowledge pass ─▶ context build    │        │ shared terminals (tmux)   │
 (local, or over ssh)             │            ─▶ agent loop ─▶ tools ─────────┼───────▶│ guest agent               │
                                  │                               │            │        │ browser + desktop         │
 inference server ◀──model calls──│                         approval gate      │        │ shared folder             │
   chat + base model              │                                            │        │                           │
   embeddings                     │ workers · boards · sleep · goals ◀─────────┼────────┤ wake channel              │
                                  └──────────────────────┬─────────────────────┘        └───────────────────────────┘
                                                         │ every event appended; recall reads it back
                                  ┌──────────────────────▼─────────────────────┐
                                  │ data/: archive (JSONL), FTS5, embeddings   │
                                  │ facts · episodes · threads · world models  │
                                  │ skills · call store                        │
                                  └────────────────────────────────────────────┘
```

## Requirements

- **Linux and Rust 1.85 or later** (edition 2024).
- **A libvirt/QEMU virtual machine** for the agent with:
  - the QEMU guest agent;
  - tmux;
  - a desktop session for the browser and desktop tools;
  - the guest-side helpers: the command runner and the browser reader.

  By default it uses the domain on `qemu:///system`. The domain name and connection URI can be overridden. Provisioning the guest is not covered by this repository.
- **An OpenAI-compatible inference server.**
  - Chat completions, `/v1/models` and tokenize at `http://127.0.0.1:8001` by default. A base model served next to the chat model is used for the knowledge pass.
  - Embeddings at `http://127.0.0.1:8081/v1/embeddings` by default.

  It was developed against Qwen3.8 and works best with an engine that keeps prompt state between requests and serves several lanes at once.
- **Optional:** a SearXNG instance or a Brave Search API key for web search, and MCP servers.

Endpoints, the VM, search and other settings are read from environment variables. Each is documented in the module that reads it (`src/llm.rs`, `src/vm.rs`, `src/search.rs`, `src/archive_semantic.rs` and so on).

## Building and running

```bash
cargo build --release
cargo test            # around 1,400 unit and end-to-end tests
cargo run --release
```

- **Files it needs and makes.** The harness reads its system prompt from `prompts/system.txt` next to `Cargo.toml`; this repository doesn't include it, so supply your own. Runtime state goes in `data/` in the same place. That includes an identity file you write yourself, which is created on first run if it's missing.
- **More terminals.** Running it again in another terminal attaches to the instance already running.
- **From another machine:**

  ```bash
  cargo run --release -- connect user@host
  ```

  This runs the harness with `--bridge` on the server over `ssh -T`. Use `--remote <command>` if it is installed elsewhere there.

Other entry points:

| Flag | What it does |
|---|---|
| `--print-system-prompt[=static\|full]` | Assemble the prompt and print it, without contacting the model |
| `--print-tools` | Print the tool array a request carries, as JSON |
| `--mistake-view [n]` | Review flagged mistakes in this terminal |
| `--recall-eval [generate n]` | Make, or run, a retrieval eval over the agent's own facts |
| `--fact-calibration` | How often facts at each certainty level were later confirmed or contradicted |

## A few commands

| | |
|---|---|
| `/approval on\|off\|list` | Approval mode. Alt+A twice approves the oldest request; Alt+X declines it |
| `/pair on` | Confirm mode in every shared terminal |
| `/goal <objective>` | Keep working toward an objective |
| `/sleep set 0200` | Consolidate memory every day at 2 am |
| `/kill` | Stop the running tool call or a worker |
| `/vm off` | Refuse every tool that acts on the VM until `/vm on` |
| `/mistake` | Flag a mistake for correction training |
| `/download`, `/upload` | Move files between the shared folder and this terminal's machine |

Tabs: Alt+1 is the conversation and Alt+2 onwards are workers. Alt+Shift+1 onwards are the terminals. Alt+S opens the shared folder, Alt+E the editor, Alt+M memory, Alt+P projects and Alt+O the overview. Alt+B goes back to the previous tab.

## Layout

All of the source is in `src/`, one module per concern. Each module starts with a doc comment that says what it is for. Good places to start:
- `app.rs`: the main loop and turn handling;
- `agent.rs`: the tool loop;
- `tabs.rs`: the interface;
- `hippocampus.rs` and `memory_layers.rs`: memory;
- `board.rs`: boards;
- `approval.rs`: the approval gate.
