use crate::archive;
use crate::attention::{self, AttentionStore};
use crate::desktop::{self, DesktopController, DesktopObservation};
use crate::hippocampus::Hippocampus;
use crate::llm::{self, ChatOptions, S0maResponse, StreamEvent};
use crate::minecraft;
use crate::threads::{self, ThreadStore};
use crate::types::Message;
use crate::vm;
use crate::voice::{self, VoiceSpeaker};
use std::time::Instant;

use serde_json::json;

use std::{
    error::Error,
    fmt,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Debug)]
pub enum AgentEvent {
    Stream(StreamEvent),
    Status(String),
    /// A model request is about to be sent (the activity line goes back to
    /// "waiting for the model" after a tool instead of still naming the tool).
    Requesting,
}

#[derive(Debug, Clone)]
pub struct OperatorSteer {
    /// Each with when it was sent, for its header.
    pub messages: Vec<crate::input::Submitted>,
    pub partial_content: String,
    pub partial_reasoning: String,
}

impl fmt::Display for OperatorSteer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s0ma received operator steering")
    }
}

impl Error for OperatorSteer {}

/// The next model call would cross the fold threshold, or could not reserve its
/// full output budget beside the prompt. Returned BEFORE the request is sent, so
/// the caller can fold the oldest turns and re-enter the loop instead of sending
/// a request vLLM would reject or silently shrinking the thinking budget.
///
/// `carry` holds messages this iteration had already appended (subagent
/// notifications, recall reminders). They are handed back rather than left in
/// `messages` so the fold plans over settled turns only, and the caller appends
/// them again afterwards, after the loops block the fold puts back.
#[derive(Debug)]
pub struct ContextRotationNeeded {
    pub input_tokens: usize,
    pub capacity: usize,
    pub carry: Vec<Message>,
}

impl fmt::Display for ContextRotationNeeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "context rotation needed mid-turn: {} input tokens against a {}-token window",
            self.input_tokens, self.capacity
        )
    }
}

impl Error for ContextRotationNeeded {}

#[derive(Debug)]
pub struct OperatorInterrupt;

impl fmt::Display for OperatorInterrupt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s0ma interrupted by operator")
    }
}

impl Error for OperatorInterrupt {}

fn check_interrupt(interrupt_requested: &AtomicBool) -> Result<(), Box<dyn Error>> {
    if interrupt_requested.load(Ordering::SeqCst) {
        crate::subagent::cancel_all();
        return Err(Box::new(OperatorInterrupt));
    }

    Ok(())
}

/// Where the agent loop is when it looks for operator steering.
///
/// Messages the operator sends while s0ma works wait for a safe point, the
/// way current harnesses queue them: nothing in flight is cut short, and the
/// message lands after the current batch of tool calls. A message sent with
/// "send now" (Ctrl+S, or a `/now` prefix) is taken at either kind of point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteerPoint {
    /// Mid-generation, or with a tool batch decided on but not yet run:
    /// taking a message here cuts the reply short or drops those calls.
    Streaming,
    /// Between steps, with nothing in flight: before a request, or after a
    /// batch of tool results.
    Safe,
    /// Between the calls of a tool batch. A look only: the answer is
    /// non-empty when a message was sent now, but nothing is taken; the safe
    /// point after the batch takes it (or, after Ctrl+C, the next turn does).
    BetweenCalls,
}

fn check_steering<S>(
    poll_steering: &mut S,
    point: SteerPoint,
    partial_content: &str,
    partial_reasoning: &str,
) -> Result<(), Box<dyn Error>>
where
    S: FnMut(SteerPoint) -> Result<Vec<crate::input::Submitted>, Box<dyn Error>>,
{
    let messages = poll_steering(point)?;

    if messages.is_empty() {
        return Ok(());
    }

    Err(Box::new(OperatorSteer {
        messages,
        partial_content: partial_content.to_string(),
        partial_reasoning: partial_reasoning.to_string(),
    }))
}

/// Whether queued VM wake events may join the turn at this pause:
/// - only after a batch of tool calls, never before the first request (the
///   turn's input, and anything it says about queued events, comes first);
/// - not once the turn's folds are spent, when a burst of events could leave
///   no room for the reply;
/// - not once s0ma has called attention_wait and is finishing up: the event a
///   wait is for must end the wait after the turn, not be spent inside it.
fn may_join_wake_events(after_tools: bool, allow_rotation: bool, attention: &AttentionStore) -> bool {
    after_tools && allow_rotation && attention.is_engaged()
}

/// Soft steering for VM wake events: events that arrived while s0ma works
/// join the turn at a pause after a batch of tool results and before the next
/// request, so neither the model nor a tool is cut short and nothing waits for
/// the turn to end. `take` takes queued events; `done` says whether they made
/// it in. Returns the names of the events that joined.
fn join_wake_events(
    messages: &mut Vec<Message>,
    memory: &mut Hippocampus,
    take: impl FnOnce() -> Option<crate::vm_wake::Joining>,
    done: impl FnOnce(bool),
) -> Result<Vec<String>, Box<dyn Error>> {
    let Some(joining) = take() else { return Ok(Vec::new()) };
    // Archived as an event-started turn's input is, so memory and sleep see it
    // (labelled environment, not operator).
    if let Err(error) = memory.remember_operator(joining.note.clone()) {
        done(false);
        return Err(error);
    }
    messages.push(Message::user(joining.note));
    done(true);
    Ok(joining.names)
}

#[allow(clippy::too_many_arguments)]
pub fn run<F, S, P>(
    messages: &mut Vec<Message>,
    memory: &mut Hippocampus,
    threads_store: &mut ThreadStore,
    attention_store: &mut AttentionStore,
    loops: &crate::open_loops::LoopStore,
    boards: &crate::board::BoardStore,
    desktop_controller: &mut DesktopController,
    voice_speaker: &VoiceSpeaker,
    interrupt_requested: &AtomicBool,
    options: &ChatOptions,
    allow_rotation: bool,
    mut prepare_fold: P,
    mut poll_steering: S,
    mut on_event: F,
) -> Result<S0maResponse, Box<dyn Error>>
where
    F: FnMut(AgentEvent) -> Result<(), Box<dyn Error>>,
    S: FnMut(SteerPoint) -> Result<Vec<crate::input::Submitted>, Box<dyn Error>>,
    P: FnMut(&[Message], &crate::context_guard::ContextUsage) -> bool,
{
    let mut tools = archive::tool_definitions();
    tools.extend(archive::recall::tool_definitions());
    tools.extend(crate::subagent::tool_definitions());
    tools.extend(crate::skills::tool_definitions_with_write());
    tools.extend(crate::world_model::tool_definitions());

    tools.extend(threads::tool_definitions());


    tools.extend(attention::tool_definitions());

    tools.extend(vm::tool_definitions());
    tools.extend(vm::netbox_tool_definitions());
    tools.extend(desktop::tool_definitions());
    tools.extend(crate::vm_image::tool_definitions());
    tools.extend(crate::share::tool_definitions());
    tools.extend(crate::shared_files::tool_definitions());
    tools.extend(crate::browser::tool_definitions());
    tools.extend(crate::search::tool_definitions());
    tools.extend(minecraft::tool_definitions());
    tools.extend(voice::tool_definitions());
    tools.extend(crate::weaknesses::tool_definitions());
    tools.extend(crate::redact::tool_definitions());
    // The goal the operator set with /goal: read it, or say it is done or blocked.
    tools.extend(crate::goal::tool_definitions());
    // Flag a mistake of its own: the same rewrite and review as the operator's /mistake.
    tools.extend(crate::mistakes::tool_definitions());
    tools.extend(crate::open_loops::tool_definitions());
    tools.extend(crate::board::tool_definitions());
    // A finished board's write-up, review answers and verification.
    tools.extend(crate::board_flow::tool_definitions());
    tools.extend(crate::vm_wake::tool_definitions());
    // Last, so the native tools' prefix is unchanged when MCP servers come and go.
    tools.extend(crate::mcp::tool_definitions());

    let mut malformed_batches = 0;
    // A batch of tool calls has run since this call began: queued VM wake
    // events may join (see may_join_wake_events).
    let mut after_tools = false;

    loop {
        crate::subagent::pump(memory)?;
        for status in crate::subagent::take_statuses()? {
            on_event(AgentEvent::Status(status))?;
        }
        for status in crate::mistakes::take_statuses() {
            on_event(AgentEvent::Status(status))?;
        }
        check_interrupt(interrupt_requested)?;
        check_steering(&mut poll_steering, SteerPoint::Safe, "", "")?;

        // Keep the open-loop block current at the end of the context. It moves
        // only when its text changes, so an unchanged list costs nothing and the
        // KV cache survives; see open_loops::refresh_context.
        if let Err(error) = crate::open_loops::refresh_context(messages, loops) {
            on_event(AgentEvent::Status(format!("[loops] block not refreshed: {error}")))?;
        }

        // Open boards: the block stays where it is, and a change is appended
        // after it as an update (see board::refresh_context).
        if let Err(error) = crate::board::refresh_context(messages, boards) {
            on_event(AgentEvent::Status(format!("[board] block not refreshed: {error}")))?;
        }

        // Everything appended from here until the request is sent belongs to
        // this iteration; see ContextRotationNeeded::carry. Measured after the
        // blocks above are refreshed: moving one takes it out from before this
        // point, which would shift what follows.
        let appended_from = messages.len();

        for notification in crate::subagent::take_notifications()? {
            messages.push(Message::user(notification));
        }

        if may_join_wake_events(after_tools, allow_rotation, attention_store) {
            let joined = join_wake_events(messages, memory, crate::vm_wake::take_to_join, |made_it| {
                if made_it {
                    crate::vm_wake::joined();
                } else {
                    crate::vm_wake::not_joined();
                }
            })?;
            if !joined.is_empty() {
                crate::vm_wake::note_reached(&joined);
                on_event(AgentEvent::Status(format!("[vm-wake] joined the turn: {}", joined.join(", "))))?;
            }
        }

        // Nonblocking: completed reminders append at an inference boundary.
        // Count them in the normal context budget below; never mutate old prefixes.
        if let Some(reminder) = archive::recall::poll(memory, messages) {
            messages.push(Message::user(reminder));
        }
        let mut partial_content = String::new();
        let mut partial_reasoning = String::new();

        // Older tool results become archive pointers, in rare batches, only
        // when turned on (`S0MA_OBSERVATION_MASK=1`): rewriting old turns costs
        // the engine's cached prefix, which on ds41 is worth far more than the
        // room. See observation_mask.
        if crate::observation_mask::enabled()
            && let Some(report) = crate::observation_mask::mask_if_due(messages)
        {
            on_event(AgentEvent::Status(format!(
                "[context] masked {} older tool results to archive pointers ({} -> {} chars)",
                report.masked, report.chars_before, report.chars_after
            )))?;
        }

        // Tool results and delegated answers grow a single turn, so recount
        // against the live window before EVERY inference, not just at admission.
        let usage = crate::context_guard::measure(messages, options, &tools)?;

        // Past the prepare threshold, the next fold's gists start being written
        // in the background. Checked here as well as at admission because most
        // growth happens inside turns. Never blocks and never fails the call.
        // Whether those gists are ready: a long window folds once they are.
        let prepared = prepare_fold(messages, &usage);

        // Fold before sending rather than clamp: the caller folds the oldest
        // turns and re-enters with the full output budget intact.
        if allow_rotation && usage.fold_need().should_fold(prepared) {
            let carry = messages.split_off(appended_from);
            return Err(Box::new(ContextRotationNeeded {
                input_tokens: usage.input_tokens,
                capacity: usage.capacity,
                carry,
            }));
        }

        // Only reached when folding couldn't make room (nothing foldable, or the
        // per-turn fold limit spent): the last resort is a smaller output
        // budget, announced rather than silent.
        let room = usage.output_room();
        if room < 1024 {
            // This is a harness admission failure, not something s0ma said.
            // Returning an error keeps it out of the conversation and archive.
            return Err("context admission paused: insufficient response headroom; tool results remain logged".into());
        }
        let mut request_options = *options;
        if room < options.max_tokens as usize {
            on_event(AgentEvent::Status(format!(
                "[context] folding couldn't make room; output budget reduced from {} to {room} tokens for this call",
                options.max_tokens
            )))?;
            request_options.max_tokens = room as u32;
        }

        on_event(AgentEvent::Requesting)?;
        // Every main-agent call is saved, as sent and as it came back
        // (calls.rs): `/mistake` can only flag what was saved at the source.
        let ticket = crate::calls::begin_call();
        let mut sent_body: Option<Vec<u8>> = None;
        // The generation streaming now, for the store. Unlike `partial_*`
        // (what the operator saw, across retries), it starts over with each
        // generation, so an interrupted call is never two glued together.
        let mut streamed = crate::calls::Streamed::default();
        let response = {
            let mut interruptible_event = |event: StreamEvent| -> Result<(), Box<dyn Error>> {
                crate::subagent::pump(memory)?;
                for status in crate::subagent::take_statuses()? {
                    on_event(AgentEvent::Status(status))?;
                }
                check_interrupt(interrupt_requested)?;

                // Nothing has arrived yet (a prefill, mostly): the checks
                // above and a "send now" message can still stop the call, as
                // they would mid-stream. Nothing to show or keep.
                if matches!(event, StreamEvent::Waiting) {
                    return check_steering(&mut poll_steering, SteerPoint::Streaming, &partial_content, &partial_reasoning);
                }

                // Exactly what `partial_*` takes, but for this generation only.
                streamed.observe(&event);
                match &event {
                    StreamEvent::Reasoning(text) => {
                        partial_reasoning.push_str(text);
                    }

                    StreamEvent::Content(text) => {
                        partial_content.push_str(text);
                    }

                    // Counted for the tok/s; nothing to keep.
                    StreamEvent::Started
                    | StreamEvent::ToolArguments(_)
                    | StreamEvent::Usage { .. }
                    | StreamEvent::Waiting => {}
                }

                // First make this chunk visible.
                on_event(AgentEvent::Stream(event))?;

                // Then honor a message sent with "send now"; queued ones wait
                // for a safe point. Everything the operator saw is represented
                // in the continuation state.
                check_steering(&mut poll_steering, SteerPoint::Streaming, &partial_content, &partial_reasoning)
            };

            llm::chat_with_tools_capturing(messages, &request_options, &tools, &mut interruptible_event, &mut sent_body)
        };

        // Saved before anything is archived from it, so the events it becomes
        // carry its turn and are listed against it.
        match &response {
            Ok(response) => {
                crate::calls::finish_call(ticket, sent_body, crate::calls::Outcome::Response(response), streamed.usage)
            }
            Err(error)
                if error.downcast_ref::<OperatorSteer>().is_some()
                    || error.downcast_ref::<OperatorInterrupt>().is_some() =>
            {
                crate::calls::finish_call(
                    ticket,
                    sent_body,
                    crate::calls::Outcome::Interrupted { content: &streamed.content, reasoning: &streamed.reasoning },
                    streamed.usage,
                )
            }
            Err(error) => crate::calls::abandon_call(ticket, &error.to_string()),
        }
        // Counted against a goal, if one is running, whether or not the call
        // finished: what was generated was generated.
        crate::goal::note_call(streamed.usage.completion_tokens, streamed.content.len() + streamed.reasoning.len());

        let response = match response {
            Ok(response) => response,
            Err(error) => {
                // Steering is journaled by app.rs before it dispatches /sleep.
                // Preserve partial traces on cancellation/transport failure too.
                if error.downcast_ref::<OperatorSteer>().is_none() {
                    memory.remember_reasoning(&partial_reasoning, true)?;
                }
                return Err(error);
            }
        };

        // Catch control input that arrived immediately after the last
        // streamed event but before conversation state is mutated.
        if let Err(error) = check_interrupt(interrupt_requested) {
            memory.remember_reasoning(&partial_reasoning, true)?;
            return Err(error);
        }

        // A finished reply with no tool calls ends the turn: messages still
        // queued start the next one (see InputReader::take_pending_submit), so
        // the reply isn't recorded as cut short. With tool calls decided on,
        // only a message sent with "send now" stops them running.
        if response.tool_calls.is_empty() {
            memory.remember_reasoning(&response.reasoning_content, false)?;
            return Ok(response);
        }
        check_steering(&mut poll_steering, SteerPoint::Streaming, &response.content, &partial_reasoning)?;
        memory.remember_reasoning(&response.reasoning_content, false)?;

        // Never commit a tool call vLLM can't parse: it would make every later
        // request in this conversation fail. Keep the prose, drop the calls, tell
        // the model, and let it retry a bounded number of times.
        let mut response = response;
        let truncated = response.finish_reason.as_deref() == Some("length");
        if let Some(notice) =
            crate::types::reject_malformed_tool_calls(&mut response.tool_calls, truncated)
        {
            malformed_batches += 1;
            memory.remember_s0ma_intermediate(format!("[tool_call_rejected]\n{notice}"))?;
            on_event(AgentEvent::Status(format!(
                "[tool] batch rejected before execution ({malformed_batches}/{}): {}",
                crate::types::MAX_MALFORMED_TOOL_CALL_RETRIES + 1,
                if truncated { "output limit reached mid-call" } else { "arguments were not valid JSON" }
            )))?;
            if malformed_batches > crate::types::MAX_MALFORMED_TOOL_CALL_RETRIES {
                return Err("tool calls were malformed repeatedly; nothing ran and the conversation was left intact".into());
            }
            if !response.content.trim().is_empty() || !response.reasoning_content.is_empty() {
                messages.push(
                    Message::assistant(response.content).with_reasoning(response.reasoning_content),
                );
            }
            messages.push(Message::user(notice));
            continue;
        }

        // Repair before anything else touches them. A tool call whose arguments
        // were cut off mid-generation is unparseable, and once it is in the
        // conversation the server rejects every later request with a 400 that
        // describes the fragment rather than anything the caller did.
        let mut tool_calls = response.tool_calls.clone();
        let truncated = crate::types::repair_tool_call_arguments(&mut tool_calls);

        messages.push(
            Message::assistant_with_tools(response.content, tool_calls.clone())
                .with_reasoning(response.reasoning_content),
        );
        // A goal's continuation turn that calls no tools makes no progress.
        crate::goal::note_tools(&tool_calls);

        // Once this assistant tool-call batch has been committed to the
        // conversation, finish the whole batch so every tool call receives
        // a corresponding tool result. We check Ctrl+C again afterward.
        let mut desktop_observations: Vec<DesktopObservation> = Vec::new();
        // Set once the operator kills a call: the rest of this batch is part of
        // the course they just stopped, so s0ma gets control back first.
        let mut stopped_by_operator: Option<String> = None;
        // Set once the operator sends a message with "send now" while this
        // batch runs: the calls not yet started wait behind it, and s0ma reads
        // it (at the safe point below) as soon as the call it arrived during
        // returns.
        let mut sent_now = false;

        for (index, call) in tool_calls.into_iter().enumerate() {
            let tool_started = Instant::now();

            // Between calls, only whether a message was sent now matters;
            // queued ones wait for the batch to end. Nothing is taken off the
            // queue here, and a failed look never fails the batch: every call
            // must still get a result.
            if index > 0 && !sent_now && stopped_by_operator.is_none() {
                sent_now = poll_steering(SteerPoint::BetweenCalls).is_ok_and(|waiting| !waiting.is_empty());
            }

            if sent_now {
                on_event(AgentEvent::Status(format!(
                    "[tool] not running {}: you sent a message first",
                    call.function.name,
                )))?;
                messages.push(Message::tool(
                    call.id.clone(),
                    json!({"ok": false, "executed": false, "error":
                        "not run: the operator sent a message (send now) before this call started. \
                         Read it first; reissue this call only if it still makes sense."
                    }).to_string(),
                ));
                continue;
            }

            if let Some(killed) = &stopped_by_operator {
                on_event(AgentEvent::Status(format!(
                    "[tool] not running {}: the operator stopped {killed} earlier in this batch",
                    call.function.name,
                )))?;
                messages.push(Message::tool(
                    call.id.clone(),
                    json!({"ok": false, "executed": false, "error": format!(
                        "not run: the operator stopped {killed} earlier in this batch (/kill). \
                         Reissue this call only if it still makes sense."
                    )}).to_string(),
                ));
                continue;
            }

            if truncated.contains(&call.id) {
                let error = format!(
                    "Your call to {} was cut off partway through — the arguments stopped \
                     mid-JSON, so nothing ran. This is an output-length problem, not a mistake \
                     in what you asked for. Reissue it with a shorter argument (fewer or smaller \
                     fields), or split the work across several calls.",
                    call.function.name,
                );
                on_event(AgentEvent::Status(format!(
                    "[tool] {} was cut off mid-arguments; not executing it",
                    call.function.name,
                )))?;
                messages.push(Message::tool(
                    call.id.clone(),
                    json!({"ok": false, "error": error}).to_string(),
                ));
                continue;
            }

            let tool_name = call.function.name.clone();
            let tool_arguments = call.function.arguments.clone();

            on_event(AgentEvent::Status(format!(
                "[tool] running: {tool_name}, arguments: {tool_arguments}"
            )))?;

            // While this call runs, /kill stops it (and only it): see kill.rs.
            let kill_call = crate::kill::tool_call(&tool_name);
            let execution_result: Result<String, Box<dyn Error>> = match call.function.name.as_str()
            {
                // The machine's tools, while the operator has it off limits
                // (commands and shells are refused in vm::execute itself).
                name if !crate::vm_access::allowed() && crate::vm_access::gated(name) => Ok(crate::vm_access::refusal(name)),
                // Approval mode: the desktop's input, JavaScript in the
                // browser, a wake watch and MCP tools wait for the operator
                // here; the shell tools in vm.rs, a spawn in subagent.rs.
                name if let Some(refused) = crate::approval::gate_tool(name, &call.function.arguments) => Ok(refused),
                "subagent_spawn" | "subagent_status" | "subagent_wait" | "subagent_cancel"
                | "subagent_resize" | "subagent_compact" | "subagent_steer" => {
                    crate::subagent::execute(&call, memory, interrupt_requested, |status| {
                        on_event(AgentEvent::Status(status))
                    })
                }
                "world_model_list" | "world_model_get" | "world_model_define"
                | "world_model_link" | "world_record_get" | "world_record_put" | "fact_search" => {
                    crate::world_model::execute(memory, &call)
                }
                "episode_get" | "episode_search" => archive::recall::execute(memory, &call),
                // A search has nothing worth waiting out once killed, so it runs
                // where it can be abandoned rather than stopped.
                "web_search" => {
                    let call = call.clone();
                    match crate::kill::abandonable("web_search", move || {
                        crate::search::execute(&call).map_err(|error| error.to_string())
                    }) {
                        Ok(Ok(outcome)) => outcome.map_err(Into::into),
                        Ok(Err(failure)) => Err(failure.into()),
                        Err(killed) => Err(Box::new(killed)),
                    }
                }
                "archive_search" | "archive_get" => archive::execute(memory, &call),
                crate::redact::TOOL => crate::redact::execute(memory, &call),
                // A completion claim is audited here, before s0ma goes on: the
                // verdict is this call's result.
                crate::goal::GET | crate::goal::UPDATE => {
                    crate::goal::execute(&call, messages, interrupt_requested, |status| {
                        on_event(AgentEvent::Status(status))
                    })
                }
                "skill_list" | "skill_read" | "skill_write" | "skill_delete" => {
                    crate::skills::execute_with_write(&call)
                }

                name if threads::handles(name) => threads::execute(threads_store, &call),


                "attention_orient" | "attention_wait" => attention::execute(attention_store, &call),

                // `netbox_*` and `vm_shell_*`: names from before every shell was a
                // shared one, answered there (the first as the shell tools are).
                "vm_exec" | "shell_type" | "shell_key" | "shell_read" | "shell_open" | "shell_close"
                | "netbox_type" | "netbox_key" | "netbox_read" | "netbox_open" | "netbox_close"
                | "vm_shell_open" | "vm_shell_input" | "vm_shell_key" | "vm_shell_read" | "vm_shell_close" => vm::execute(&call),

                "desktop_view"
                | "desktop_click"
                | "desktop_button"
                | "desktop_move"
                | "desktop_drag"
                | "desktop_look"
                | "desktop_key"
                | "desktop_type"
                | "desktop_wait"
                | "desktop_release_all"
                | "desktop_status"
                | "desktop_launch" => desktop_controller.execute(&call).map(|execution| {
                    if let Some(observation) = execution.observation {
                        desktop_observations.push(observation);
                    }
                    execution.result
                }),

                name if crate::vm_image::handles(name) => crate::vm_image::execute(&call).map(|execution| {
                    if let Some(observation) = execution.observation {
                        desktop_observations.push(observation);
                    }
                    execution.result
                }),

                name if crate::share::handles(name) => crate::share::execute(&call),
                name if crate::shared_files::handles(name) => crate::shared_files::execute(&call),
                name if crate::board_flow::handles(name) => {
                    let mut run = crate::share::run_on_machine;
                    crate::board_flow::execute(boards, &call, &crate::share::share_dir(), &mut run)
                }

                name if crate::browser::handles(name) => crate::browser::execute(&call).map(|execution| {
                    if let Some(observation) = execution.observation {
                        desktop_observations.push(observation);
                    }
                    execution.result
                }),

                "minecraft_status"
                | "minecraft_observe"
                | "minecraft_go_to"
                | "minecraft_look_at"
                | "minecraft_collect"
                | "minecraft_break"
                | "minecraft_break_many"
                | "minecraft_craft"
                | "minecraft_equip"
                | "minecraft_place"
                | "minecraft_place_many"
                | "minecraft_interact"
                | "minecraft_chat"
                | "minecraft_stop"
                | "minecraft_view" => minecraft::execute(&call).map(|execution| {
                    if let Some(observation) = execution.observation {
                        desktop_observations.push(observation);
                    }
                    execution.result
                }),

                "voice_speak" => voice::execute(voice_speaker, &call),
                name if crate::vm_wake::handles(name) => crate::vm_wake::execute(&call),

                "weakness_report" | "weakness_add_instance" | "weakness_list" => {
                    crate::weaknesses::execute(memory, &call)
                }

                crate::mistakes::TOOL => Ok(crate::mistakes::execute_tool(&call)),

                "loop_open" | "loop_touch" | "loop_close" | "loop_list" => {
                    crate::open_loops::execute_with(loops, &*memory, &call)
                }

                // The worker identity is the harness's to supply, not the
                // model's: a caller that could name itself could take over
                // another worker's claim.
                "board_list" | "board_open" | "board_read" | "board_link" | "board_intent"
                | "board_claim" | "board_fact" | "board_kill" | "board_test" | "board_close" => {
                    let models =
                        crate::world_model::Models::new(crate::world_model::path_for(memory));
                    let mut board_context = crate::board::Context {
                        store: boards,
                        archive: &mut *memory,
                        models: &models,
                        tests: &crate::vm::GoalTest,
                        worker: "s0ma",
                        subagent: None,
                    };
                    crate::board::execute(&mut board_context, &call)
                }

                // Tools from operator-configured MCP servers. An image result
                // becomes this batch's visual observation, like a screenshot.
                name if crate::mcp::handles(name) => {
                    crate::mcp::execute(&call, interrupt_requested).map(|execution| {
                        if let Some(observation) = execution.observation {
                            desktop_observations.push(observation);
                        }
                        execution.result
                    })
                }

                name => Ok(json!({
                    "ok": false,
                    "error": format!("unknown tool: {name}")
                })
                .to_string()),
            };

            let ending = kill_call.finish();
            if ending == crate::kill::Ending::TooLate {
                on_event(AgentEvent::Status(format!(
                    "[kill] {tool_name} finished before it could be stopped; its result stands"
                )))?;
            }

            // A malformed or invalid tool call is model-visible feedback, not
            // a harness-fatal error. Returning a tool message also preserves
            // the required one-result-per-call conversation structure so the
            // model can repair the arguments on its next turn.
            let result = match execution_result {
                // Stopped by the operator: say so plainly, keeping whatever the
                // call had produced, so s0ma neither retries blindly nor
                // mistakes it for an ordinary failure.
                outcome if ending == crate::kill::Ending::Killed => {
                    let (killed, own) = match &outcome {
                        Ok(result) => (None, Some(result.clone())),
                        Err(error) => match error.downcast_ref::<crate::kill::Killed>() {
                            Some(killed) => (Some(killed), None),
                            None => (None, Some(error.to_string())),
                        },
                    };
                    on_event(AgentEvent::Status(format!(
                        "[tool] killed: {tool_name}{}",
                        killed.map(|k| format!(": {}", k.detail)).unwrap_or_default()
                    )))?;
                    stopped_by_operator = Some(tool_name.clone());
                    crate::kill::tool_result(&tool_name, tool_started.elapsed(), killed, own.as_deref())
                }
                Ok(result) => result,
                Err(error) => {
                    on_event(AgentEvent::Status(format!(
                        "[tool] failed: {tool_name}: {error}"
                    )))?;

                    json!({
                        "ok": false,
                        "tool": tool_name,
                        "error": error.to_string(),
                        "instruction": "Correct the tool arguments and retry if the operation is still needed."
                    })
                    .to_string()
                }
            };
            let duration = tool_started.elapsed();

            on_event(AgentEvent::Status(format!(
                "[tool] finished: {tool_name} ({:.1}s)",
                duration.as_secs_f64()
            )))?;
            let mut model_result = result.clone();
            let mut archived = None;
            if tool_name.starts_with("subagent_") {
                // Do not persist task contexts or repeated status results in main memory.
                crate::subagent::archive_parent_call(&call, &result)?;
            } else {
                // The record's head; the result starts right after it, which is
                // its offset in archive_get's character coordinates.
                let record_head = format!("[tool:{tool_name}]\narguments:\n{tool_arguments}\nresult:\n");
                let event_id = memory.remember_tool_result(format!(
                    "{record_head}{result}\nduration_ms: {}",
                    duration.as_millis(),
                ))?;
                archived = Some((event_id, record_head.chars().count()));
                // Explicit reads and structured mutation responses remain intact.
                if matches!(
                    tool_name.as_str(),
                    "vm_exec" | "shell_read" | "shell_type" | "shell_key"
                        | "netbox_read" | "netbox_type" | "netbox_key"
                ) {
                    model_result = crate::hippocampus::tool_preview(&result, event_id, 6000);
                }
                // Pipeline-stamped provenance: the model-visible copy names the
                // archive event holding this exact result, so a world record can
                // cite the search (see search.rs, Provenance).
                if tool_name == "web_search" {
                    model_result = crate::search::stamp_event_id(&result, event_id);
                }
            }

            let mut tool_message = Message::tool(call.id, model_result.clone());
            // A result cut to a preview says which archive event holds the rest
            // and exactly which character range it left out.
            if let Some((event_id, result_offset)) = archived {
                tool_message.content = crate::llm::locate_in_archive(tool_message.content, &model_result, event_id, result_offset);
            }
            // The trailer goes on after Message::tool's size limit so it is
            // never cut: the completion time, how long the call took, and the
            // archive event that observation masking points back to.
            tool_message.content.push_str(&crate::observation_mask::trailer(
                &crate::clock::now(),
                Some(duration),
                archived.map(|(id, _)| id),
            ));
            messages.push(tool_message);
        }

        // Finish the assistant's complete tool-result batch before adding a
        // visual observation as a new user-role sensory message.
        if let Some(observation) = desktop_observations.pop() {
            for message in messages.iter_mut() {
                message.release_desktop_image();
            }

            messages.push(Message::desktop_observation(
                observation.content,
                observation.data_url,
            ));
        }
        after_tools = true;

        check_interrupt(interrupt_requested)?;

        // The safe point queued messages wait for: the batch's results are in
        // and the next request isn't sent yet. A message sent now during the
        // batch is taken here too.
        check_steering(&mut poll_steering, SteerPoint::Safe, "", "")?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm_wake::{Queue, VmWakeEvent};
    use std::sync::Mutex;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "s0ma-agent-join-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn event(name: &str) -> VmWakeEvent {
        VmWakeEvent {
            name: name.into(),
            text: format!("{name} finished"),
            clipped: false,
            received_at: chrono::Local::now(),
            instance: None,
        }
    }

    fn try_join(messages: &mut Vec<Message>, memory: &mut Hippocampus, queue: &Mutex<Queue>) -> Result<Vec<String>, Box<dyn Error>> {
        join_wake_events(messages, memory, || crate::vm_wake::take_to_join_from(queue), |made_it| {
            let mut queue = queue.lock().unwrap();
            if made_it { queue.joined() } else { queue.not_joined() }
        })
    }

    fn join(messages: &mut Vec<Message>, memory: &mut Hippocampus, queue: &Mutex<Queue>) -> Vec<String> {
        try_join(messages, memory, queue).unwrap()
    }

    #[test]
    fn queued_wake_events_join_a_running_turn_as_one_framed_note() {
        let dir = temp_dir("engaged");
        let mut memory = Hippocampus::new(dir.join("hippocampus.jsonl")).unwrap();
        let queue = Mutex::new(Queue::load(Some(dir.join("vm_wake_queue.json"))));
        queue.lock().unwrap().push(Some("1-1"), vec![event("build"), event("deploy")]);
        let mut messages = vec![Message::user("the operator's request".into()).by_operator()];

        assert_eq!(join(&mut messages, &mut memory, &queue), vec!["build", "deploy"]);
        let note = messages.last().unwrap();
        assert!(note.content.starts_with(crate::vm_wake::VM_WAKE_PREFIX), "{}", note.content);
        assert!(note.content.contains("2 events came through") && note.content.contains("> build finished"));
        assert!(!note.is_operator_input(), "environmental evidence, never the operator's");
        // Archived (as environment), and gone from the saved queue.
        let archived = std::fs::read_to_string(dir.join("hippocampus.jsonl")).unwrap();
        assert!(archived.contains("2 events came through"));
        assert!(Queue::load(Some(dir.join("vm_wake_queue.json"))).take_to_join().0.is_empty());

        // Nothing queued: nothing joins.
        assert!(join(&mut messages, &mut memory, &queue).is_empty());
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn events_join_only_after_tools_with_room_and_while_attention_is_engaged() {
        let dir = temp_dir("when");
        let engaged = AttentionStore::load(&dir).unwrap();
        assert!(may_join_wake_events(true, true, &engaged));
        assert!(!may_join_wake_events(false, true, &engaged), "never before the first request");
        assert!(!may_join_wake_events(true, false, &engaged), "not once the turn's folds are spent");

        // Finishing up to wait for a VM event: the event must end the wait.
        let waiting_dir = temp_dir("waiting");
        std::fs::write(
            waiting_dir.join("attention.json"),
            json!({"schema_version": 1, "mode": "waiting", "activity": "building", "orientation": "wait for it",
                   "condition": {"kind": "vm_event", "watch": "build"}, "updated_at": "2026-10-10T00:00:00Z"}).to_string(),
        )
        .unwrap();
        let waiting = AttentionStore::load(&waiting_dir).unwrap();
        assert!(!may_join_wake_events(true, true, &waiting));
    }

    #[test]
    fn events_that_cannot_be_archived_stay_queued() {
        let dir = temp_dir("unarchivable");
        let path = dir.join("hippocampus.jsonl");
        let mut memory = Hippocampus::new(&path).unwrap();
        // The archive can't be written to: a directory where its file was.
        let _ = std::fs::remove_file(&path);
        std::fs::create_dir(&path).unwrap();
        let queue = Mutex::new(Queue::load(None));
        queue.lock().unwrap().push(None, vec![event("build"), event("deploy")]);
        let mut messages = Vec::new();
        assert!(try_join(&mut messages, &mut memory, &queue).is_err());
        assert!(messages.is_empty());
        let (back, _) = queue.lock().unwrap().take_to_join();
        assert_eq!(back.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), vec!["build", "deploy"], "first in line, in order");
    }
}
