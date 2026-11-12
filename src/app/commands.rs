//! The operator's slash commands at the idle prompt (or sent as a steer):
//! each family is handed to the module that owns it, in order, and the first
//! that recognises the command answers it.

use super::*;

impl App {
    pub(super) fn handle_command(&mut self, input: &str, options: &ChatOptions) -> AppResult<CommandOutcome> {
        let command = voice::command_text(input).trim();

        if command.is_empty() {
            return Ok(CommandOutcome::Handled);
        }

        // Usually typed mid-turn and handled by the keyboard thread; at the
        // idle prompt there is no tool call, but subagents may be running.
        if let Some(report) = crate::kill::handle_command(command) {
            self.notify(&report);
            return Ok(CommandOutcome::Handled);
        }

        if let Some(result) = self.voice.handle_command(command) {
            self.report(result);
            return Ok(CommandOutcome::Handled);
        }

        // /mode instruct|low|medium|xhigh etc.
        if let Some(result) = self.reasoning.handle_mode_command(command) {
            self.report(result);
            return Ok(CommandOutcome::Handled);
        }

        // /tz: the operator's time zone, shown in every turn header beside the host's.
        if let Some(result) = crate::clock::handle_command(command) {
            self.report(result);
            return Ok(CommandOutcome::Handled);
        }

        // /model ... and /embeddings ...: the model server, its key, and
        // whether embeddings are used. A key typed here is kept, and its line
        // taken out of the prompt's history.
        if let Some(result) = crate::model_config::handle_command(command) {
            if crate::model_config::carries_key(command)
                && let Err(error) = crate::jobs::forget_typed_line(&self.data_dir, command)
            {
                self.notify_error(&format!("[model] the key's line could not be taken out of input_history.txt: {error}"));
            }
            self.report_status(result);
            return Ok(CommandOutcome::Handled);
        }

        // /machine host|vm: where the agent works, from the next start.
        if let Some(result) = crate::vm::machine::handle_command(command) {
            self.report_status(result);
            return Ok(CommandOutcome::Handled);
        }

        // /mcp ...: MCP server status, tool listing, reload.
        if let Some(result) = crate::mcp::handle_operator_command(command) {
            // Status lines like the other commands': collapsed, `[*] mcp · …`
            // (`[-]` for an error) with the rest of the block under it.
            self.report_status(result);
            return Ok(CommandOutcome::Handled);
        }

        // /mistake ...: flag a model call the agent got wrong, and review the
        // corrections. Usually typed mid-turn and handled on the keyboard
        // thread (input.rs); here at the idle prompt, or as a steer.
        let place = if self.in_agent_cycle { crate::mistakes::Place::Turn } else { crate::mistakes::Place::Idle };
        if let Some(reply) = crate::mistakes::handle_command(command, place) {
            match reply {
                crate::mistakes::Reply::Text(message) => self.notify(&message),
                crate::mistakes::Reply::Error(message) => self.notify_error(&message),
                crate::mistakes::Reply::View(start) => self.open_mistake_view(start),
            }
            return Ok(CommandOutcome::Handled);
        }

        // /weakness ...: operator review of the agent's weakness reports.
        if let Some(result) = crate::weaknesses::handle_operator_command(&self.memory, command) {
            self.report(result);
            return Ok(CommandOutcome::Handled);
        }

        // /redact ...: the agent has asked for something to come out of an event.
        if let Some(reply) = crate::redact::handle_command(&self.memory, command) {
            match reply {
                crate::redact::Reply::Text(message) => self.notify(&message),
                crate::redact::Reply::Error(message) => self.notify_error(&message),
                crate::redact::Reply::View(start) => self.open_redact_view(start),
            }
            return Ok(CommandOutcome::Handled);
        }

        // /job ...: a stretch of work whose record is removed when it ends.
        //
        // Anything of ours that writes from the archive stops first, before the
        // redaction rather than after it: a compaction batch or a prepared gist
        // read the events as they were, and finishing afterwards would write
        // what was just redacted back into a store.
        if crate::jobs::will_remove(command) {
            self.fold_prep.quiesce();
            if let Err(error) = self.compactor.quiesce() {
                self.notify_error(&format!(
                    "[job] compaction could not be stopped ({error}); nothing was removed, since a batch it \
                     has in hand would be saved from the events as they are now"
                ));
                return Ok(CommandOutcome::Handled);
            }
            self.compaction_started = false;
        }
        if let Some(result) = crate::jobs::handle_operator_command(&self.memory, command) {
            self.report(result);
            // A close leaves the conversation in the live context, where the
            // archive no longer has it. Rotating rebuilds the context from the
            // redacted archive, so a later fold cannot write the content back
            // into memory.
            if crate::jobs::take_rotation_owed() {
                if let Err(error) = self.compactor.flush(&self.memory) {
                    self.notify_error(&format!("[job] compaction could not be flushed: {error}"));
                }
                match self.rotate_context() {
                    Ok(Some((id, tokens))) => {
                        crate::clock::mark_rebuilt(crate::clock::Rebuild::Compaction);
                        self.notify_status(
                            &format!(
                                "[job] working context rebuilt through event {id} ({tokens} input tokens), \
                                 so what was redacted is no longer in front of s0ma"
                            ),
                            false,
                        );
                        return Ok(CommandOutcome::ContextReset);
                    }
                    Ok(None) => self.notify_status(
                        "[job] nothing sealed to rebuild the context from: s0ma still holds the conversation \
                         in context until the next compaction or sleep, though the archive and the stores no \
                         longer have it",
                        true,
                    ),
                    Err(error) => self.notify_error(&format!("[job] the context was not rebuilt: {error}")),
                }
            }
            return Ok(CommandOutcome::Handled);
        }

        // /goal ...: an objective the agent works toward, turn after turn.
        let goal_waiting = self.goal_waiting_words();
        if let Some(outcome) = crate::goal::handle_command(command, goal_waiting.as_deref()) {
            // What the operator asked for goes into the archive as his words, so
            // memory has it and a job's close finds it.
            if let Some(record) = outcome.record
                && let Err(error) = self.memory.remember_operator(record)
            {
                self.notify_error(&format!("[goal] the goal was set but not recorded in the archive: {error}"));
            }
            if outcome.operator_spoke {
                // Set or resumed: the operator has said to carry on.
                self.goal_watch.operator_spoke();
            }
            match outcome.reply {
                Ok(message) => self.notify_status(&message, false),
                Err(message) => self.notify_status(&message, true),
            }
            return Ok(CommandOutcome::Handled);
        }

        // /backup: the shared folder, backed up now (share/backup.rs).
        if let Some(result) = crate::share::backup::handle_command(command) {
            self.report(result);
            return Ok(CommandOutcome::Handled);
        }

        // /approval on|off: commands and subagents wait for the operator
        // (approval.rs).
        if let Some(result) = crate::approval::handle_command(command) {
            self.report(result);
            return Ok(CommandOutcome::Handled);
        }

        // /download and /upload move files between an attached terminal's
        // computer and this one (transfer.rs); typed here, the files are
        // already on this computer.
        if crate::transfer::is_command(command) {
            self.notify(&format!(
                "/download and /upload are for terminals attached from another computer (s0ma connect); the shared folder on this one is {}",
                crate::share::receive_dir().display()
            ));
            return Ok(CommandOutcome::Handled);
        }

        // /vm on|off: the agent's access to its machine (vm/access.rs).
        if let Some(result) = crate::vm::access::handle_command(command) {
            self.report(result);
            return Ok(CommandOutcome::Handled);
        }

        let loop_parts = command.split_whitespace().collect::<Vec<_>>();
        if loop_parts.first().copied() == Some("/loop") {
            let next_mode = match loop_parts.as_slice() {
                ["/loop"] | ["/loop", "status"] => {
                    self.notify_mode(self.runtime_mode);
                    return Ok(CommandOutcome::Handled);
                }
                ["/loop", "on"] => RuntimeMode::Looping,
                ["/loop", "off"] => RuntimeMode::Conversation,
                _ => {
                    self.notify_error("usage: /loop [on|off|status]");
                    return Ok(CommandOutcome::Handled);
                }
            };

            self.runtime_mode = next_mode;
            self.notify_mode(next_mode);
            return Ok(CommandOutcome::RuntimeModeChanged(next_mode));
        }

        // /audit …, /board audits …, /board gate …: audits (`audit/ops.rs`).
        if let Some(result) = crate::audit::ops::handle_command(command) {
            self.report_status(result);
            return Ok(CommandOutcome::Handled);
        }

        if matches!(command.split_whitespace().next(), Some("/board") | Some("/hint") | Some("/unhint")) {
            match board::handle_operator_command(&self.boards, command) {
                Ok(message) => self.notify(&message),
                Err(error) => self.notify_error(&format!("[board] board store unavailable: {error}")),
            }
            return Ok(CommandOutcome::Handled);
        }

        // /sleep set HHMM | off: the daily sleep. Plain /sleep sleeps now (below).
        if let Some(result) = self.sleep_schedule.handle_command(command) {
            self.report(result);
            // Turned off or moved: a sleep that was holding spawns no longer is.
            crate::subagent::hold_spawns(self.sleep_schedule.waiting());
            return Ok(CommandOutcome::Handled);
        }

        // /shell …, /pair …: the terminals the operator shares with the agent
        // (/netbox, as /shell was called before).
        if matches!(command.split_whitespace().next(), Some("/shell") | Some("/netbox") | Some("/pair")) {
            match crate::vm::netboxes::handle_operator_command(command) {
                Ok(message) => self.notify(&message),
                Err(error) => self.notify_error(&format!("[shell] {error}")),
            }
            return Ok(CommandOutcome::Handled);
        }

        // /share <path> …: copy files into /share on the agent's machine.
        if command.split_whitespace().next() == Some("/share") {
            self.send_files(command.trim_start().trim_start_matches("/share"));
            return Ok(CommandOutcome::Handled);
        }

        if command.split_whitespace().next() == Some("/loops") {
            match open_loops::handle_operator_command(&self.loops, command) {
                Ok(message) => self.notify(&message),
                Err(error) => self.notify_error(&format!("[loops] loop store unavailable: {error}")),
            }
            return Ok(CommandOutcome::Handled);
        }

        match command {
            // `/quit` stops the agent itself, from whichever terminal typed it. On an
            // attached phone that is rarely what is meant — Ctrl-D detaches and
            // leaves it running — so say which one this is before doing it.
            // What is actually in context, not what is on disk. The invariant is
            // only auditable if it can be read back from the running process.
            "/identity" => {
                self.notify(&format!(
                    "[identity] loaded from {}\n\n{}",
                    self.identity.path().display(),
                    self.identity.text(),
                ));
                if self.identity.edited_since_load() {
                    self.notify(
                        "[identity] the file on disk has changed since startup. What is above is \
                         what is in context; the edit takes effect on restart. Identity does not \
                         change mid-session by design.",
                    );
                }
                Ok(CommandOutcome::Handled)
            }

            "/quit" => {
                if self.session.as_ref().is_some_and(|server| server.attached() > 0) {
                    self.notify(
                        "[session] /quit stops s0ma for every terminal, not just this one. \
                         Detach instead with Ctrl-D; use /quit! if stopping it is what you meant.",
                    );
                    return Ok(CommandOutcome::Handled);
                }
                Ok(CommandOutcome::Quit)
            }

            "/quit!" => Ok(CommandOutcome::Quit),

            // Every command and the keys, which no longer sit in the corner.
            "/help" => {
                self.notify(&crate::completion::help_text());
                Ok(CommandOutcome::Handled)
            }

            // The last reply as it was written (markdown, not the rendered
            // terminal text), or just its last fenced code block.
            "/copy" | "/copy code" => {
                let Some(reply) = self
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == "assistant" && !m.content.trim().is_empty())
                    .map(|m| m.content.clone())
                else {
                    self.notify_error("[copy] no reply to copy yet");
                    return Ok(CommandOutcome::Handled);
                };
                let text = if command == "/copy code" {
                    match crate::clipboard::last_code_block(&reply) {
                        Some(code) => code,
                        None => {
                            self.notify_error("[copy] the last reply has no code block; /copy copies the whole reply");
                            return Ok(CommandOutcome::Handled);
                        }
                    }
                } else {
                    reply
                };
                self.notify(&crate::clipboard::copy(&text));
                Ok(CommandOutcome::Handled)
            }

            "/tokens" => {
                let tools = self.tools();
                let pending = Message::user(String::new());
                let usage = crate::context::guard::preflight(&self.messages, &pending, options, &tools)?;

                self.notify(&format!(
                    "[tokens] {} input tokens, {} projected ({:.1}%)",
                    usage.input_tokens,
                    usage.projected_tokens,
                    usage.utilization_percent(),
                ));

                Ok(CommandOutcome::Handled)
            }

            "/compact" => {
                self.compaction_started = false;
                // Gists prepared for this context are no use once it's rotated.
                self.fold_prep.quiesce();
                self.compactor.flush(&self.memory)?;
                match self.rotate_context()? {
                    Some((id,tokens))=>{
                        // The next input's header says the context was rebuilt.
                        crate::clock::mark_rebuilt(crate::clock::Rebuild::Compaction);
                        self.notify_status(&format!("[compaction] runtime rotated through event {id}: {tokens} input tokens; daily review remains pending"),false)
                    }
                    None=>{
                        self.notify_status("[compaction] no newer completed history to activate",false);
                        match sleep::continuous::diagnostic(&self.memory) {
                            Ok(detail)=>self.notify_status(&format!("[compaction] {detail}"),false),
                            Err(error)=>self.notify_status(&format!("[compaction] diagnostic unavailable: {error}"),true),
                        }
                    }
                }
                Ok(CommandOutcome::ContextReset)
            }
            "/sleep" => {
                // Held, as the scheduled one is: the checkpoint and the world
                // model have no event ranges to purge by, so a job's content
                // must not reach them before it is redacted.
                if let Some(job) = crate::jobs::sleep_held() {
                    self.notify_error(&format!(
                        "[sleep] held while job \"{job}\" is open: a deep sleep writes the checkpoint and the \
                         world model, and a close cannot take anything out of those. Close the job and run \
                         this again, or /compact to rotate the context now."
                    ));
                    return Ok(CommandOutcome::Handled);
                }
                let slept = self.sleep_and_rebuild()?;
                // Done by hand: nothing scheduled is still owed tonight.
                self.sleep_schedule.clear_pending();
                crate::subagent::hold_spawns(false);
                let outcome = match slept {
                    Some(event_id) => {
                        self.notify(&format!(
                            "\n[sleep] memory consolidated through event {event_id}"
                        ));

                        self.notify("[sleep] working context reset");

                        CommandOutcome::ContextReset
                    }

                    None => {
                        self.notify("\n[sleep] nothing new to consolidate");

                        CommandOutcome::Handled
                    }
                };

                Ok(outcome)
            }

            _ => Ok(CommandOutcome::NotCommand),
        }
    }

    /// A command's reply, as a note (an error in red).
    fn report(&self, result: Result<String, String>) {
        match result {
            Ok(message) => self.notify(&message),
            Err(message) => self.notify_error(&message),
        }
    }

    /// A command's reply as a status line, collapsed (`[-]` for an error).
    fn report_status(&self, result: Result<String, String>) {
        match result {
            Ok(message) => self.notify_status(&message, false),
            Err(message) => self.notify_status(&message, true),
        }
    }
}
