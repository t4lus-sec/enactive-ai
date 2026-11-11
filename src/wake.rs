use crate::{llm::ToolDefinition, types::ToolCall};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

const MIN_WAKE_DELAY_SECONDS: i64 = 60;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduledWake {
    pub at: DateTime<Utc>,

    #[serde(default)]
    pub reason: Option<String>,
}

pub struct WakeScheduler {
    next: Option<ScheduledWake>,
    path: PathBuf,
}

impl WakeScheduler {
    // --------------------------------------------------------
    // LOAD / STATE
    // --------------------------------------------------------

    /// Load persisted wake state.
    ///
    /// A missing wake.json, an empty wake.json, or a persisted `null`
    /// means there is currently no scheduled wake.
    pub fn load(data_dir: &Path) -> Result<Self, Box<dyn Error>> {
        let path = data_dir.join("wake.json");

        let next = match fs::read_to_string(&path) {
            Ok(contents) => {
                if contents.trim().is_empty() {
                    None
                } else {
                    serde_json::from_str::<Option<ScheduledWake>>(&contents)?
                }
            }

            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,

            Err(error) => return Err(error.into()),
        };

        Ok(Self { next, path })
    }

    /// Whether any wake is currently persisted.
    pub fn has_wake(&self) -> bool {
        self.next.is_some()
    }

    /// Return a copy of the currently persisted wake without
    /// consuming or modifying it.
    ///
    /// The runtime can retain this value as the wake that began an
    /// autonomous cycle, then verify that s0ma replaced it before
    /// allowing that cycle to finish.
    pub fn current(&self) -> Option<ScheduledWake> {
        self.next.clone()
    }

    // --------------------------------------------------------
    // TIME / DUE STATE
    // --------------------------------------------------------

    /// Return the duration until the current wake.
    ///
    /// An overdue wake returns Duration::ZERO so it can fire
    /// immediately after restart or after another foreground
    /// operation finishes.
    pub fn remaining(&self) -> Option<Duration> {
        let wake = self.next.as_ref()?;
        let now = Utc::now();

        if wake.at <= now {
            return Some(Duration::ZERO);
        }

        (wake.at - now).to_std().ok()
    }

    /// Whether the currently persisted wake has reached or passed
    /// its scheduled time.
    pub fn is_due(&self) -> bool {
        matches!(
            self.remaining(),
            Some(remaining) if remaining.is_zero()
        )
    }

    /// Return the currently persisted wake only when it is due.
    ///
    /// This does not consume the wake.
    pub fn due_wake(&self) -> Option<ScheduledWake> {
        if !self.is_due() {
            return None;
        }

        self.current()
    }

    /// Atomically consume and return the current wake when it is due.
    ///
    /// Ordinary waking activity is now sustained by the harness attention
    /// loop, so a fired wake is a one-shot scheduled event rather than a link
    /// that s0ma must replace before an autonomous cycle may end.
    pub fn take_due_wake(&mut self) -> Result<Option<ScheduledWake>, Box<dyn Error>> {
        let Some(wake) = self.due_wake() else {
            return Ok(None);
        };

        self.persist_state(&None)?;
        self.next = None;

        Ok(Some(wake))
    }

    // --------------------------------------------------------
    // REPLACEMENT TRACKING
    // --------------------------------------------------------

    /// Whether the wake that began an autonomous cycle has been
    /// successfully replaced.
    ///
    /// `previous` should be the wake captured when that cycle began.
    ///
    /// A missing current wake does not count as replacement. A cycle
    /// is complete only when another persisted wake exists.
    pub fn was_replaced(&self, previous: &ScheduledWake) -> bool {
        matches!(
            self.next.as_ref(),
            Some(current) if current != previous
        )
    }

    // --------------------------------------------------------
    // MUTATION
    // --------------------------------------------------------

    /// Atomically replace the current wake with a newly selected wake.
    ///
    /// The previous persisted wake remains intact until the replacement
    /// has been successfully written to disk.
    fn schedule(
        &mut self,
        at: DateTime<Utc>,
        reason: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        let next = Some(ScheduledWake { at, reason });

        // Durable state first.
        //
        // If persistence fails, leave both the on-disk and in-memory
        // current wake unchanged.
        self.persist_state(&next)?;

        self.next = next;

        Ok(())
    }

    /// Persist wake state using write-then-rename instead of modifying
    /// the live wake.json file in place.
    fn persist_state(&self, next: &Option<ScheduledWake>) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let temp_path = self.path.with_extension("json.tmp");

        {
            let mut file = File::create(&temp_path)?;

            serde_json::to_writer_pretty(&mut file, next)?;

            file.write_all(b"\n")?;

            // Ensure the temporary file contents reach the filesystem
            // before it replaces the live wake state.
            file.sync_all()?;
        }

        // Atomic replacement on the local Unix filesystem.
        fs::rename(&temp_path, &self.path)?;

        Ok(())
    }
}

// ============================================================
// TOOL
// ============================================================

#[derive(Debug, Deserialize)]
struct WakeSetArgs {
    at: String,

    #[serde(default)]
    reason: Option<String>,
}

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![ToolDefinition::function(
        "wake_set",
        "Choose and persist a particular future time that matters. \
         This atomically replaces the previously scheduled wake and \
         survives program restarts. The harness already sustains ordinary \
         waking activity, so this tool is for timed intentions and reminders, \
         not for keeping yourself active. Use an RFC3339 timestamp with a \
         timezone offset.",
        json!({
            "type": "object",
            "properties": {
                "at": {
                    "type": "string",
                    "description": "Exact future RFC3339 wake time including timezone offset, for example 2026-09-01T03:47:00-06:00"
                },
                "reason": {
                    "type": "string",
                    "description": "Optional note to your future self about why you chose this time"
                }
            },
            "required": ["at"],
            "additionalProperties": false
        }),
    )]
}

pub fn execute(scheduler: &mut WakeScheduler, call: &ToolCall) -> Result<String, Box<dyn Error>> {
    match call.function.name.as_str() {
        "wake_set" => {
            let args: WakeSetArgs = serde_json::from_str(&call.function.arguments)?;

            let parsed = DateTime::parse_from_rfc3339(&args.at)?;

            let at = parsed.with_timezone(&Utc);
            let now = Utc::now();

            let delay = at - now;

            if delay.num_seconds() < MIN_WAKE_DELAY_SECONDS {
                return Ok(json!({
                    "ok": false,
                    "error": format!(
                        "wake time must be at least \
                         {MIN_WAKE_DELAY_SECONDS} seconds \
                         in the future"
                    )
                })
                .to_string());
            }

            let requested_at = args.at;
            let reason = args.reason;

            scheduler.schedule(at, reason.clone())?;

            Ok(json!({
                "ok": true,
                "scheduled_for": at.to_rfc3339(),
                "requested_at": requested_at,
                "reason": reason,
                "persistent": true,
                "replaced_previous_wake": true
            })
            .to_string())
        }

        _ => Ok(json!({
            "ok": false,
            "error": "unknown wake tool"
        })
        .to_string()),
    }
}
