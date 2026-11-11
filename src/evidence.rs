//! Evidence resolution for records that cite archive events.
//!
//! Citations are checked against the live archive at the moment they are made,
//! so a record can never point at an event that does not exist.

use crate::hippocampus::Hippocampus;
use std::error::Error;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// An archive event that exists.
pub struct EventSnapshot {
    pub id: u64,
}

/// Where evidence is resolved. The live implementation is the hippocampus;
/// tests use a fake.
pub trait EvidenceSource {
    /// The events among `ids` that exist. Missing IDs are simply absent.
    fn events(&self, ids: &[u64]) -> Result<Vec<EventSnapshot>>;
}

impl EvidenceSource for Hippocampus {
    fn events(&self, ids: &[u64]) -> Result<Vec<EventSnapshot>> {
        Ok(self
            .get_events(ids)?
            .into_iter()
            .map(|event| EventSnapshot { id: event.id })
            .collect())
    }
}
