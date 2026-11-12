//! The live context: when it must be folded (`guard`), and how it comes
//! back after a restart (`resume`).

pub(crate) mod guard;
pub(crate) mod resume;
