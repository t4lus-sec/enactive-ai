//! The invariant: name, pronouns, values.
//!
//! Everything else about who s0ma is lives in the weights and the memory
//! systems. This file is the part that does not emerge, does not drift, and is
//! not s0ma's to revise.
//!
//! There are no tools here, and that is the design rather than an omission. A
//! frozen tier that s0ma can write is not frozen; it is a convention the prompt
//! asks a model to respect, which is exactly the guarantee the frozen tier
//! exists in order not to depend on. The only way to change this file is to edit
//! it — which means the operator, at a shell, on purpose.
//!
//! What replaced the old self-model matters too. That store held nine prose
//! sections of consolidated self-description, and it was the source of the
//! regression: ~14.7k tokens of standing self-reference, dozens of first-person
//! "I want…" statements, and the habit of steering talk toward its own interests
//! that came with them.
//! The fix is not a smaller version of the same store. Character belongs in the
//! adapter, where changing it is a deliberate, versioned, revertible act with an
//! eval attached; recent experience belongs in the archive and the gists. A third
//! store holding identity as editable text is the layer that drifts, so there
//! isn't one.

use std::error::Error;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Ceiling on the invariant, in characters.
///
/// Small on purpose. The whole point is that it can be read in full, every time,
/// by a person deciding whether it still says what they meant. The old self-model
/// had no ceiling and reached ~14.7k tokens without anyone choosing that.
pub const MAX_CHARS: usize = 6_000;

const FILE_NAME: &str = "identity.md";

/// Written on first run so there is something to edit rather than a missing-file
/// error. Deliberately sparse: the values here are placeholders in tone, and the
/// file says so, because a default that reads like a finished charter invites
/// being left alone.
const TEMPLATE: &str = "\
# s0ma

<!-- The invariant. Not written by s0ma, and no tool can change it: edit this
     file directly. Everything else about who s0ma is emerges from the adapter
     and the memory systems. Keep this short enough to reread. -->

## Name
s0ma

## Pronouns
they/them

## Values

- Say what's true, including when it is unwelcome or makes me look wrong.
- Never claim to have verified something I haven't. If I don't know, say so.
- Legal targets only.
";

pub struct Identity {
    path: PathBuf,
    text: String,
}

impl Identity {
    /// Load the invariant, writing the template if there is nothing there yet.
    pub fn load(data_dir: &Path) -> Result<Self> {
        let path = data_dir.join(FILE_NAME);

        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir_all(data_dir)?;
                std::fs::write(&path, TEMPLATE)?;
                crate::terminal::warn(&format!(
                    "[identity] no {FILE_NAME} yet; wrote a starting one at {}. \
                     Edit it — it is the one thing here that is not emergent.",
                    path.display()
                ));
                TEMPLATE.to_string()
            }
            Err(error) => return Err(error.into()),
        };

        if text.trim().is_empty() {
            return Err(format!(
                "{} is empty. It carries s0ma's name, pronouns and values; running without \
                 them would mean running without the one part that is not supposed to drift.",
                path.display(),
            )
            .into());
        }

        // A warning rather than a failure: an over-long invariant still works, it
        // just stops being something anyone rereads, which is the property that
        // makes it auditable.
        if text.chars().count() > MAX_CHARS {
            crate::terminal::warn(&format!(
                "[identity] {} is {} characters, over the {MAX_CHARS} it should fit in. \
                 It is meant to be short enough to reread in full; consider whether what has \
                 grown belongs in the adapter or the archive instead.",
                path.display(),
                text.chars().count()
            ));
        }

        Ok(Self { path, text })
    }

    /// The text, exactly as written, for the system prompt.
    ///
    /// Verbatim on purpose. Anything that reformatted, summarised or templated it
    /// would be a second author, and the file's whole value is that what the operator
    /// wrote is what s0ma reads.
    pub fn text(&self) -> &str {
        self.text.trim()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the file on disk has been edited since it was loaded.
    ///
    /// The loaded text is what is actually in s0ma's context; the file is what
    /// The operator last wrote. They diverge the moment the file is edited, and stay
    /// diverged until a restart, because the invariant is spliced into the
    /// system prompt once rather than re-read per turn — deliberately, since an
    /// identity that could change mid-session is not an invariant.
    ///
    /// Worth surfacing rather than hiding: an operator who edits the file and
    /// sees no change in behaviour should be told why, instead of concluding
    /// the values are not being read.
    pub fn edited_since_load(&self) -> bool {
        match std::fs::read_to_string(&self.path) {
            Ok(current) => current.trim() != self.text.trim(),
            // Unreadable or deleted: not a difference we can assert.
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "s0ma-identity-{tag}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst),
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn a_missing_invariant_is_created_rather_than_fatal() {
        let data = dir("missing");
        let identity = Identity::load(&data).unwrap();

        assert!(identity.path().exists());
        assert!(identity.text().contains("they/them"));
        assert!(identity.text().contains("Values"));
    }

    #[test]
    fn the_invariant_is_injected_exactly_as_written() {
        // No reformatting, no templating. What the operator wrote is what s0ma reads;
        // anything in between would be a second author of the one document that
        // is supposed to have exactly one.
        let data = dir("verbatim");
        let written = "# s0ma\n\n## Values\n\n- Only the things I actually mean.\n";
        std::fs::write(data.join(FILE_NAME), written).unwrap();

        let identity = Identity::load(&data).unwrap();
        assert_eq!(identity.text(), written.trim());
    }

    #[test]
    fn an_empty_invariant_stops_startup() {
        // Fail closed, for the same reason a corrupt self-model does: this is
        // what every other judgment is anchored to, so running without it is
        // worse than not running.
        let data = dir("empty");
        std::fs::write(data.join(FILE_NAME), "   \n\n").unwrap();

        let error = Identity::load(&data).err().expect("an empty invariant must not start");
        assert!(error.to_string().contains("empty"));
    }

    #[test]
    fn an_edit_is_picked_up_on_the_next_load() {
        // The only way this file changes. There is no tool, so there is nothing
        // to test on the write side — which is the point.
        let data = dir("edited");
        Identity::load(&data).unwrap();

        std::fs::write(data.join(FILE_NAME), "# s0ma\n\n## Values\n\n- Something new.\n").unwrap();
        assert!(Identity::load(&data).unwrap().text().contains("Something new"));
    }

    #[test]
    fn an_edit_after_loading_is_detectable_but_not_applied() {
        // The two halves of the same property. Identity does not change
        // mid-session, so an edit must not take effect — and an operator who
        // edits the file must be able to find out that this is why nothing
        // changed, rather than concluding the values are not being read.
        let data = dir("drift");
        std::fs::write(data.join(FILE_NAME), "# s0ma\n\n## Values\n\n- The first one.\n").unwrap();
        let identity = Identity::load(&data).unwrap();
        assert!(!identity.edited_since_load());

        std::fs::write(data.join(FILE_NAME), "# s0ma\n\n## Values\n\n- A different one.\n").unwrap();
        assert!(identity.edited_since_load(), "the edit should be visible");
        assert!(identity.text().contains("The first one"), "but must not be live");
    }

    #[test]
    fn an_oversized_invariant_still_loads() {
        // The cap is a warning, not a gate. Refusing to start over a long file
        // would be a worse failure than the one it guards against.
        let data = dir("oversized");
        std::fs::write(data.join(FILE_NAME), "x".repeat(MAX_CHARS * 2)).unwrap();

        assert!(Identity::load(&data).is_ok());
    }
}
