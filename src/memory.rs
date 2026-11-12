//! Memory beyond the live context: folding turns into gists (`fold`), the
//! layers they settle into (`layers`), and recall pushed into a turn and its
//! evaluation.

pub(crate) mod fold;
pub(crate) mod layers;
pub(crate) mod recall_eval;
pub(crate) mod recall_push;
