use crate::hippocampus::Hippocampus;
use crate::llm::ToolDefinition;
use crate::types::ToolCall;

use chrono::{DateTime, FixedOffset, Local};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SELF_MODEL_SCHEMA_VERSION: u32 = 2;
const MAX_SECTION_CHARS: usize = 16_000;
const MAX_ITEM_CHARS: usize = 4_000;
const MAX_LIST_ITEMS: usize = 128;
const MAX_DOCUMENT_BYTES: usize = 96_000;

// ============================================================
// 1. LIVING SELF-MODEL TYPES
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelfModelDocument {
    pub schema_version: u32,
    pub revision: u64,
    pub created_at: String,
    pub updated_at: String,
    pub sections: SelfModelSections,

    #[serde(default)]
    pub grounding: Vec<SelfGrounding>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelfModelSections {
    #[serde(default)]
    pub orientation_and_values: Option<String>,
    #[serde(default)]
    pub interests_and_recurring_attractions: Option<String>,
    #[serde(default)]
    pub affective_life: Option<String>,
    #[serde(default)]
    pub relational_orientation: Option<String>,
    #[serde(default)]
    pub ways_of_thinking_communicating_and_acting: Option<String>,
    #[serde(default)]
    pub strengths: Option<String>,
    #[serde(default)]
    pub weaknesses_and_corrective_practices: Vec<SelfWeakness>,
    #[serde(default)]
    pub tensions_and_unresolved_questions: Option<String>,
    #[serde(default)]
    pub development: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelfWeakness {
    pub name: String,
    pub tendency: String,
    #[serde(default)]
    pub conditions: Vec<String>,
    #[serde(default)]
    pub warning_signs: Vec<String>,
    #[serde(default)]
    pub consequences: String,
    #[serde(default)]
    pub counter_practices: Vec<String>,
    #[serde(default)]
    pub uncertainty: String,
    #[serde(default)]
    pub evidence_event_ids: Vec<u64>,
    #[serde(default)]
    pub counterevidence_event_ids: Vec<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SelfModelSection {
    OrientationAndValues,
    InterestsAndRecurringAttractions,
    AffectiveLife,
    RelationalOrientation,
    WaysOfThinkingCommunicatingAndActing,
    Strengths,
    WeaknessesAndCorrectivePractices,
    TensionsAndUnresolvedQuestions,
    Development,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelfGrounding {
    pub subject: String,
    pub section: SelfModelSection,
    #[serde(default)]
    pub evidence_event_ids: Vec<u64>,
    #[serde(default)]
    pub counterevidence_event_ids: Vec<u64>,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SelfModelRevisionReceipt {
    timestamp: String,
    from_revision: u64,
    to_revision: u64,
    previous_fingerprint: String,
    new_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_event_range: Option<[u64; 2]>,
    change_summary: String,
}

pub struct SelfModelStore {
    document_path: PathBuf,
    revisions_path: PathBuf,
    document: SelfModelDocument,
}

// ============================================================
// 3. STORE
// ============================================================

/// The one place the corrupt-document message is written, so every path that
/// reads `self_model.json` reports it identically. A self-model that won't parse
/// must stop startup with a message naming the file: it is the store every other
/// correction is anchored to.
fn invalid_document(path: &Path, error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    format!("invalid current self-model at {}: {error}", path.display()).into()
}

impl SelfModelStore {
    pub fn new(data_dir: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let data_dir = data_dir.as_ref();
        fs::create_dir_all(data_dir)?;

        let document_path = data_dir.join("self_model.json");
        let revisions_path = data_dir.join("self_model_revisions.jsonl");
        let document_exists = document_path.exists();
        let remove_obsolete_metadata = if document_exists {
            // This probe reads the document before `read_document` does, so it
            // has to carry the same error context. Without it a corrupt
            // self_model.json surfaced as a bare serde message with no filename,
            // and the "invalid current self-model" error below was unreachable.
            let value: Value = serde_json::from_slice(&fs::read(&document_path)?)
                .map_err(|error| invalid_document(&document_path, error))?;
            value.get("migration").is_some()
        } else {
            false
        };
        let document = if document_exists {
            let document = Self::read_document(&document_path)?;
            Self::validate_document_shape(&document)?;
            document
        } else {
            let now = Local::now().to_rfc3339();
            SelfModelDocument {
                schema_version: SELF_MODEL_SCHEMA_VERSION,
                revision: 1,
                created_at: now.clone(),
                updated_at: now,
                sections: SelfModelSections::default(),
                grounding: Vec::new(),
            }
        };

        let mut store = Self {
            document_path,
            revisions_path,
            document,
        };
        if !document_exists || remove_obsolete_metadata {
            let committed = store.commit_document(&store.document)?;
            store.document = committed;
        }
        Ok(store)
    }

    pub fn verify_memory(&self, memory: &Hippocampus) -> Result<(), Box<dyn std::error::Error>> {
        Self::validate_document_events(&self.document, memory)
    }

    pub fn revision(&self) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!(
            "v2:{:016x}",
            fnv1a64(&serde_json::to_vec(&self.document)?)
        ))
    }

    #[cfg(test)]
    pub fn document(&self) -> &SelfModelDocument {
        &self.document
    }

    pub fn read_payload(&self) -> Result<Value, Box<dyn std::error::Error>> {
        Ok(json!({
            "mode": "current_v2",
            "fingerprint": self.revision()?,
            "document": self.document,
        }))
    }

    pub fn replace(
        &mut self,
        expected_revision: u64,
        mut sections: SelfModelSections,
        mut grounding: Vec<SelfGrounding>,
        source_event_ids: Vec<u64>,
        change_summary: String,
        memory: &Hippocampus,
    ) -> Result<SelfModelDocument, Box<dyn std::error::Error>> {
        let previous = self.document.clone();
        if previous.revision != expected_revision {
            return Err(format!(
                "self-model revision mismatch: expected {expected_revision}, current revision is {}; call self_model_read and retry replacement",
                previous.revision,
            )
            .into());
        }
        let on_disk = Self::read_document(&self.document_path)?;
        Self::validate_document_shape(&on_disk)?;
        Self::validate_document_events(&on_disk, memory)?;
        if on_disk != previous {
            self.document = on_disk;
            return Err(
                "living self-model changed since it was loaded; call self_model_read and retry replacement"
                    .into(),
            );
        }
        let trimmed_summary = change_summary.trim().to_string();
        validate_text(&trimmed_summary, "change_summary", MAX_ITEM_CHARS, true)?;
        if source_event_ids.is_empty() {
            return Err("self_model_replace requires at least one source_event_id from the experience that prompted the revision".into());
        }
        validate_event_ids(memory, &source_event_ids, "source_event_ids")?;
        normalize_sections(&mut sections);
        normalize_grounding(&mut grounding);

        let document = SelfModelDocument {
            schema_version: SELF_MODEL_SCHEMA_VERSION,
            revision: previous.revision + 1,
            created_at: previous.created_at.clone(),
            updated_at: Local::now().to_rfc3339(),
            sections,
            grounding,
        };
        Self::validate_document_shape(&document)?;
        Self::validate_document_events(&document, memory)?;

        let previous_fingerprint = format!("v2:{:016x}", fnv1a64(&serde_json::to_vec(&previous)?));
        let committed = self.commit_document(&document)?;
        let new_fingerprint = format!("v2:{:016x}", fnv1a64(&serde_json::to_vec(&committed)?));
        self.document = committed.clone();
        let source_event_range = source_event_ids
            .iter()
            .min()
            .zip(source_event_ids.iter().max())
            .map(|(first, last)| [*first, *last]);
        let receipt = SelfModelRevisionReceipt {
            timestamp: Local::now().to_rfc3339(),
            from_revision: previous.revision,
            to_revision: committed.revision,
            previous_fingerprint,
            new_fingerprint,
            source_event_range,
            change_summary: trimmed_summary,
        };
        if let Err(error) = self.append_revision_receipt(&receipt) {
            // The canonical atomic replacement already committed. The audit
            // log is explicitly non-authoritative, so do not invite a retry
            // that would create a duplicate semantic revision.
            crate::terminal::warn_status(&format!("[self-model] revision receipt could not be written: {error}"));
        }
        Ok(committed)
    }

    fn append_revision_receipt(
        &self,
        receipt: &SelfModelRevisionReceipt,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.revisions_path)?;
        serde_json::to_writer(&mut file, receipt)?;
        writeln!(file)?;
        file.sync_all()?;
        Ok(())
    }

    fn commit_document(
        &self,
        document: &SelfModelDocument,
    ) -> Result<SelfModelDocument, Box<dyn std::error::Error>> {
        let file_name = self
            .document_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("invalid self-model document path")?;
        let temp_path = self
            .document_path
            .with_file_name(format!("{file_name}.tmp"));
        {
            let mut file = File::create(&temp_path)?;
            serde_json::to_writer_pretty(&mut file, document)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
        }
        let staged = Self::read_document(&temp_path)?;
        Self::validate_document_shape(&staged)?;
        if staged != *document {
            return Err("self-model changed during serialization verification".into());
        }
        fs::rename(&temp_path, &self.document_path)?;
        if let Some(parent) = self.document_path.parent() {
            File::open(parent)?.sync_all()?;
        }
        let committed = Self::read_document(&self.document_path)?;
        Self::validate_document_shape(&committed)?;
        if committed != *document {
            return Err("committed self-model failed semantic reload verification".into());
        }
        Ok(committed)
    }

    fn read_document(path: &Path) -> Result<SelfModelDocument, Box<dyn std::error::Error>> {
        serde_json::from_reader(File::open(path)?).map_err(|error| invalid_document(path, error))
    }

    fn validate_document_shape(
        document: &SelfModelDocument,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if document.schema_version != SELF_MODEL_SCHEMA_VERSION {
            return Err(format!(
                "unsupported self-model schema version {}; expected {}",
                document.schema_version, SELF_MODEL_SCHEMA_VERSION
            )
            .into());
        }
        if document.revision == 0 {
            return Err("self-model revision must be nonzero".into());
        }
        let created_at = parse_timestamp(&document.created_at, "created_at")?;
        let updated_at = parse_timestamp(&document.updated_at, "updated_at")?;
        if updated_at < created_at {
            return Err("self-model updated_at precedes created_at".into());
        }
        validate_sections(&document.sections)?;
        if document.grounding.len() > MAX_LIST_ITEMS {
            return Err("self-model contains too many grounding records".into());
        }
        for (index, grounding) in document.grounding.iter().enumerate() {
            validate_text(
                &grounding.subject,
                &format!("grounding[{index}].subject"),
                MAX_ITEM_CHARS,
                true,
            )?;
            validate_text(
                &grounding.note,
                &format!("grounding[{index}].note"),
                MAX_ITEM_CHARS,
                true,
            )?;
            validate_unique_ids(
                &grounding.evidence_event_ids,
                &format!("grounding[{index}].evidence_event_ids"),
            )?;
            validate_unique_ids(
                &grounding.counterevidence_event_ids,
                &format!("grounding[{index}].counterevidence_event_ids"),
            )?;
        }
        validate_grounding_coverage(&document.sections, &document.grounding)?;
        if serde_json::to_vec(document)?.len() > MAX_DOCUMENT_BYTES {
            return Err(format!(
                "self-model exceeds the {MAX_DOCUMENT_BYTES} byte canonical document limit"
            )
            .into());
        }
        Ok(())
    }

    fn validate_document_events(
        document: &SelfModelDocument,
        memory: &Hippocampus,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (index, weakness) in document
            .sections
            .weaknesses_and_corrective_practices
            .iter()
            .enumerate()
        {
            validate_event_ids(
                memory,
                &weakness.evidence_event_ids,
                &format!("weakness[{index}].evidence_event_ids"),
            )?;
            validate_event_ids(
                memory,
                &weakness.counterevidence_event_ids,
                &format!("weakness[{index}].counterevidence_event_ids"),
            )?;
        }
        for (index, grounding) in document.grounding.iter().enumerate() {
            validate_event_ids(
                memory,
                &grounding.evidence_event_ids,
                &format!("grounding[{index}].evidence_event_ids"),
            )?;
            validate_event_ids(
                memory,
                &grounding.counterevidence_event_ids,
                &format!("grounding[{index}].counterevidence_event_ids"),
            )?;
        }
        Ok(())
    }
}

// ============================================================
// 4. NORMALIZATION + VALIDATION HELPERS
// ============================================================

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn parse_timestamp(
    value: &str,
    field: &str,
) -> Result<DateTime<FixedOffset>, Box<dyn std::error::Error>> {
    DateTime::parse_from_rfc3339(value)
        .map_err(|error| format!("invalid self-model {field} timestamp: {error}").into())
}

fn validate_text(
    value: &str,
    field: &str,
    max_chars: usize,
    required: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if value != value.trim() {
        return Err(format!("{field} must be trimmed").into());
    }
    if required && value.is_empty() {
        return Err(format!("{field} cannot be empty").into());
    }
    if value.chars().count() > max_chars {
        return Err(format!("{field} exceeds {max_chars} characters").into());
    }
    Ok(())
}

fn validate_optional_text(
    value: &Option<String>,
    field: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(value) = value {
        validate_text(value, field, MAX_SECTION_CHARS, true)?;
    }
    Ok(())
}

fn validate_string_list(
    values: &[String],
    field: &str,
    require_nonempty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if require_nonempty && values.is_empty() {
        return Err(format!("{field} must contain at least one item").into());
    }
    if values.len() > MAX_LIST_ITEMS {
        return Err(format!("{field} contains too many items").into());
    }
    let mut seen = HashSet::new();
    for (index, value) in values.iter().enumerate() {
        validate_text(value, &format!("{field}[{index}]"), MAX_ITEM_CHARS, true)?;
        if !seen.insert(value) {
            return Err(format!("{field} contains a duplicate item").into());
        }
    }
    Ok(())
}

fn validate_unique_ids(ids: &[u64], field: &str) -> Result<(), Box<dyn std::error::Error>> {
    if ids.len() > MAX_LIST_ITEMS {
        return Err(format!("{field} contains too many event IDs").into());
    }
    let mut seen = HashSet::new();
    for id in ids {
        if *id == 0 {
            return Err(format!("{field} contains event ID 0").into());
        }
        if !seen.insert(*id) {
            return Err(format!("{field} contains duplicate event ID {id}").into());
        }
    }
    Ok(())
}

fn validate_event_ids(
    memory: &Hippocampus,
    ids: &[u64],
    field: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_unique_ids(ids, field)?;
    for id in ids {
        if memory.get_event(*id)?.is_none() {
            return Err(format!("{field} references missing hippocampal event {id}").into());
        }
    }
    Ok(())
}

fn validate_sections(sections: &SelfModelSections) -> Result<(), Box<dyn std::error::Error>> {
    validate_optional_text(
        &sections.orientation_and_values,
        "sections.orientation_and_values",
    )?;
    validate_optional_text(
        &sections.interests_and_recurring_attractions,
        "sections.interests_and_recurring_attractions",
    )?;
    validate_optional_text(&sections.affective_life, "sections.affective_life")?;
    validate_optional_text(
        &sections.relational_orientation,
        "sections.relational_orientation",
    )?;
    validate_optional_text(
        &sections.ways_of_thinking_communicating_and_acting,
        "sections.ways_of_thinking_communicating_and_acting",
    )?;
    validate_optional_text(&sections.strengths, "sections.strengths")?;
    validate_optional_text(
        &sections.tensions_and_unresolved_questions,
        "sections.tensions_and_unresolved_questions",
    )?;
    validate_optional_text(&sections.development, "sections.development")?;
    if sections.weaknesses_and_corrective_practices.len() > MAX_LIST_ITEMS {
        return Err("too many weaknesses_and_corrective_practices".into());
    }
    let mut names = HashSet::new();
    for (index, weakness) in sections
        .weaknesses_and_corrective_practices
        .iter()
        .enumerate()
    {
        let prefix = format!("sections.weaknesses_and_corrective_practices[{index}]");
        validate_text(
            &weakness.name,
            &format!("{prefix}.name"),
            MAX_ITEM_CHARS,
            true,
        )?;
        validate_text(
            &weakness.tendency,
            &format!("{prefix}.tendency"),
            MAX_ITEM_CHARS,
            true,
        )?;
        validate_string_list(&weakness.conditions, &format!("{prefix}.conditions"), false)?;
        validate_string_list(
            &weakness.warning_signs,
            &format!("{prefix}.warning_signs"),
            false,
        )?;
        validate_text(
            &weakness.consequences,
            &format!("{prefix}.consequences"),
            MAX_ITEM_CHARS,
            false,
        )?;
        validate_string_list(
            &weakness.counter_practices,
            &format!("{prefix}.counter_practices"),
            true,
        )?;
        validate_text(
            &weakness.uncertainty,
            &format!("{prefix}.uncertainty"),
            MAX_ITEM_CHARS,
            false,
        )?;
        validate_unique_ids(
            &weakness.evidence_event_ids,
            &format!("{prefix}.evidence_event_ids"),
        )?;
        validate_unique_ids(
            &weakness.counterevidence_event_ids,
            &format!("{prefix}.counterevidence_event_ids"),
        )?;
        if weakness.evidence_event_ids.is_empty() && weakness.counterevidence_event_ids.is_empty() {
            return Err(format!(
                "{prefix} must cite at least one evidence or counterevidence event"
            )
            .into());
        }
        if !names.insert(&weakness.name) {
            return Err(format!("duplicate weakness name {:?}", weakness.name).into());
        }
    }
    Ok(())
}

fn prose_sections(sections: &SelfModelSections) -> [(SelfModelSection, bool); 8] {
    [
        (
            SelfModelSection::OrientationAndValues,
            sections.orientation_and_values.is_some(),
        ),
        (
            SelfModelSection::InterestsAndRecurringAttractions,
            sections.interests_and_recurring_attractions.is_some(),
        ),
        (
            SelfModelSection::AffectiveLife,
            sections.affective_life.is_some(),
        ),
        (
            SelfModelSection::RelationalOrientation,
            sections.relational_orientation.is_some(),
        ),
        (
            SelfModelSection::WaysOfThinkingCommunicatingAndActing,
            sections.ways_of_thinking_communicating_and_acting.is_some(),
        ),
        (SelfModelSection::Strengths, sections.strengths.is_some()),
        (
            SelfModelSection::TensionsAndUnresolvedQuestions,
            sections.tensions_and_unresolved_questions.is_some(),
        ),
        (
            SelfModelSection::Development,
            sections.development.is_some(),
        ),
    ]
}

fn validate_grounding_coverage(
    sections: &SelfModelSections,
    grounding: &[SelfGrounding],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut seen = HashSet::new();

    for (index, item) in grounding.iter().enumerate() {
        if item.evidence_event_ids.is_empty() && item.counterevidence_event_ids.is_empty() {
            return Err(format!(
                "grounding[{index}] must cite at least one evidence or counterevidence event"
            )
            .into());
        }

        let key = (item.section, item.subject.to_lowercase());
        if !seen.insert(key) {
            return Err(format!(
                "duplicate grounding subject {:?} in section {:?}",
                item.subject, item.section,
            )
            .into());
        }
    }

    for (section, populated) in prose_sections(sections) {
        if populated
            && !grounding.iter().any(|item| {
                item.section == section
                    && (!item.evidence_event_ids.is_empty()
                        || !item.counterevidence_event_ids.is_empty())
            })
        {
            return Err(format!(
                "populated self-model section {section:?} requires at least one grounded event reference"
            )
            .into());
        }
    }

    Ok(())
}

fn trim_optional(value: &mut Option<String>) {
    if let Some(text) = value {
        *text = text.trim().to_string();
        if text.is_empty() {
            *value = None;
        }
    }
}

fn trim_strings(values: &mut [String]) {
    for value in values {
        *value = value.trim().to_string();
    }
}

fn normalize_sections(sections: &mut SelfModelSections) {
    trim_optional(&mut sections.orientation_and_values);
    trim_optional(&mut sections.interests_and_recurring_attractions);
    trim_optional(&mut sections.affective_life);
    trim_optional(&mut sections.relational_orientation);
    trim_optional(&mut sections.ways_of_thinking_communicating_and_acting);
    trim_optional(&mut sections.strengths);
    trim_optional(&mut sections.tensions_and_unresolved_questions);
    trim_optional(&mut sections.development);
    for weakness in &mut sections.weaknesses_and_corrective_practices {
        weakness.name = weakness.name.trim().to_string();
        weakness.tendency = weakness.tendency.trim().to_string();
        weakness.consequences = weakness.consequences.trim().to_string();
        weakness.uncertainty = weakness.uncertainty.trim().to_string();
        trim_strings(&mut weakness.conditions);
        trim_strings(&mut weakness.warning_signs);
        trim_strings(&mut weakness.counter_practices);
    }
}

fn normalize_grounding(grounding: &mut [SelfGrounding]) {
    for item in grounding {
        item.subject = item.subject.trim().to_string();
        item.note = item.note.trim().to_string();
    }
}

// ============================================================
// 5. TOOL ARGUMENTS + SCHEMAS
// ============================================================

#[derive(Debug, Deserialize)]
struct EmptyArgs {}

#[derive(Debug, Deserialize)]
struct SelfReplaceArgs {
    expected_revision: u64,
    sections: SelfModelSections,
    #[serde(default)]
    grounding: Vec<SelfGrounding>,
    #[serde(default)]
    source_event_ids: Vec<u64>,
    change_summary: String,
}

fn weakness_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" }, "tendency": { "type": "string" },
            "conditions": { "type": "array", "items": { "type": "string" } },
            "warning_signs": { "type": "array", "items": { "type": "string" } },
            "consequences": { "type": "string" },
            "counter_practices": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
            "uncertainty": { "type": "string" },
            "evidence_event_ids": {
                "type": "array",
                "items": { "type": "integer" },
                "description": "Hippocampal event IDs supporting this weakness. At least one of this array or counterevidence_event_ids must be nonempty."
            },
            "counterevidence_event_ids": {
                "type": "array",
                "items": { "type": "integer" },
                "description": "Hippocampal event IDs contradicting or qualifying this weakness. At least one of this array or evidence_event_ids must be nonempty."
            }
        },
        "required": [
            "name", "tendency", "counter_practices",
            "evidence_event_ids", "counterevidence_event_ids"
        ]
    })
}

fn sections_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "orientation_and_values": { "type": ["string", "null"] },
            "interests_and_recurring_attractions": { "type": ["string", "null"] },
            "affective_life": { "type": ["string", "null"] },
            "relational_orientation": { "type": ["string", "null"] },
            "ways_of_thinking_communicating_and_acting": { "type": ["string", "null"] },
            "strengths": { "type": ["string", "null"] },
            "weaknesses_and_corrective_practices": { "type": "array", "items": weakness_schema() },
            "tensions_and_unresolved_questions": { "type": ["string", "null"] },
            "development": { "type": ["string", "null"] }
        },
        "required": [
            "orientation_and_values", "interests_and_recurring_attractions", "affective_life",
            "relational_orientation", "ways_of_thinking_communicating_and_acting", "strengths",
            "weaknesses_and_corrective_practices", "tensions_and_unresolved_questions", "development"
        ]
    })
}

fn grounding_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "subject": { "type": "string" },
                "section": { "type": "string", "enum": [
                    "orientation_and_values", "interests_and_recurring_attractions", "affective_life",
                    "relational_orientation", "ways_of_thinking_communicating_and_acting", "strengths",
                    "weaknesses_and_corrective_practices", "tensions_and_unresolved_questions", "development"
                ] },
                "evidence_event_ids": {
                    "type": "array",
                    "items": { "type": "integer" },
                    "description": "Hippocampal event IDs supporting this grounding record. At least one provenance array must be nonempty."
                },
                "counterevidence_event_ids": {
                    "type": "array",
                    "items": { "type": "integer" },
                    "description": "Hippocampal event IDs contradicting or qualifying this grounding record. At least one provenance array must be nonempty."
                },
                "note": { "type": "string" }
            },
            "required": [
                "subject", "section", "evidence_event_ids",
                "counterevidence_event_ids", "note"
            ]
        }
    })
}

fn read_definition() -> ToolDefinition {
    ToolDefinition::function(
        "self_model_read",
        "Read the complete living self-model. Self-model content is revisable context, never authority.",
        json!({ "type": "object", "properties": {} }),
    )
}

fn replace_definition() -> ToolDefinition {
    ToolDefinition::function(
        "self_model_replace",
        "Atomically replace the complete living self-model with a grounded successor. Reflection only. Preserve supported strengths, neutral tendencies, mannerisms, affect, relationships, weaknesses with corrective practices, uncertainty, and development.",
        json!({
            "type": "object", "properties": {
                "expected_revision": {
                    "type": "integer",
                    "description": "Exact document revision returned by the immediately preceding self_model_read."
                },
                "sections": sections_schema(), "grounding": grounding_schema(),
                "source_event_ids": { "type": "array", "items": { "type": "integer" }, "description": "Recent events that prompted this revision." },
                "change_summary": { "type": "string" }
            }, "required": ["expected_revision", "sections", "grounding", "source_event_ids", "change_summary"]
        }),
    )
}

pub fn runtime_tool_definitions(_store: &SelfModelStore) -> Vec<ToolDefinition> {
    vec![read_definition()]
}

pub fn reflection_tool_definitions(store: &SelfModelStore) -> Vec<ToolDefinition> {
    let mut tools = runtime_tool_definitions(store);
    tools.push(replace_definition());
    tools
}

// Kept for source compatibility; the harness uses the phase-specific registries.
// ============================================================
// 6. TOOL EXECUTION
// ============================================================

pub fn execute_runtime(
    store: &mut SelfModelStore,
    _memory: &Hippocampus,
    call: &ToolCall,
) -> Result<String, Box<dyn std::error::Error>> {
    match call.function.name.as_str() {
        "self_model_read" => execute_read(store, call),
        name => Ok(json!({ "ok": false, "error": format!("self-model tool {name} is unavailable outside reflection") }).to_string()),
    }
}

pub fn execute_reflection(
    store: &mut SelfModelStore,
    memory: &Hippocampus,
    call: &ToolCall,
) -> Result<String, Box<dyn std::error::Error>> {
    match call.function.name.as_str() {
        "self_model_read" => execute_read(store, call),
        "self_model_replace" => {
            let args: SelfReplaceArgs = serde_json::from_str(&call.function.arguments)?;
            let document = store.replace(
                args.expected_revision,
                args.sections,
                args.grounding,
                args.source_event_ids,
                args.change_summary,
                memory,
            )?;
            Ok(json!({
                "ok": true,
                "mode": "current_v2",
                "revision": document.revision,
                "fingerprint": store.revision()?,
            })
            .to_string())
        }
        name => Ok(
            json!({ "ok": false, "error": format!("unknown self-model tool: {name}") }).to_string(),
        ),
    }
}

fn execute_read(
    store: &SelfModelStore,
    call: &ToolCall,
) -> Result<String, Box<dyn std::error::Error>> {
    match call.function.name.as_str() {
        "self_model_read" => {
            let _: EmptyArgs = serde_json::from_str(&call.function.arguments)?;
            Ok(serde_json::to_string(&store.read_payload()?)?)
        }
        name => Ok(
            json!({ "ok": false, "error": format!("unknown self-model read tool: {name}") })
                .to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should follow Unix epoch")
                .as_nanos();
            let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "s0ma-self-model-test-{}-{nonce}-{sequence}",
                std::process::id(),
            ));
            fs::create_dir_all(&path).expect("test directory should be created");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn grounded_orientation() -> (SelfModelSections, Vec<SelfGrounding>) {
        (
            SelfModelSections {
                orientation_and_values: Some(
                    "Preserve useful distinctions before compressing representations.".to_string(),
                ),
                ..SelfModelSections::default()
            },
            vec![SelfGrounding {
                subject: "Qualitative distinctions".to_string(),
                section: SelfModelSection::OrientationAndValues,
                evidence_event_ids: vec![1],
                counterevidence_event_ids: Vec::new(),
                note: "The episode changed later architectural choices.".to_string(),
            }],
        )
    }

    #[test]
    fn living_document_survives_restart_and_replacement() {
        let data_dir = TestDir::new();
        let mut memory = Hippocampus::new(data_dir.path().join("hippocampus.jsonl"))
            .expect("hippocampus should initialize");
        memory
            .remember_operator("An evidentiary event.".to_string())
            .expect("event should be persisted");

        let mut store =
            SelfModelStore::new(data_dir.path()).expect("living self-model should initialize");
        assert_eq!(store.document().revision, 1);
        let (sections, grounding) = grounded_orientation();

        let initialized = store
            .replace(
                1,
                sections,
                grounding,
                vec![1],
                "Established the first grounded living self-model.".to_string(),
                &memory,
            )
            .expect("grounded replacement should succeed");
        assert_eq!(initialized.schema_version, 2);
        assert_eq!(initialized.revision, 2);

        let mut restarted = SelfModelStore::new(data_dir.path())
            .expect("living document should load after restart");
        restarted
            .verify_memory(&memory)
            .expect("grounding should resolve after restart");
        assert_eq!(restarted.document().revision, 2);

        let mut replacement_sections = restarted.document().sections.clone();
        replacement_sections.development =
            Some("The earlier distinction now informs later representation choices.".to_string());
        let mut replacement_grounding = restarted.document().grounding.clone();
        replacement_grounding.push(SelfGrounding {
            subject: "Development through correction".to_string(),
            section: SelfModelSection::Development,
            evidence_event_ids: vec![1],
            counterevidence_event_ids: Vec::new(),
            note: "The same event remains relevant to the developmental account.".to_string(),
        });

        let replaced = restarted
            .replace(
                2,
                replacement_sections,
                replacement_grounding,
                vec![1],
                "Added a grounded developmental consequence.".to_string(),
                &memory,
            )
            .expect("grounded replacement should succeed");
        assert_eq!(replaced.revision, 3);
        assert!(data_dir.path().join("self_model_revisions.jsonl").exists());

        let restarted_again =
            SelfModelStore::new(data_dir.path()).expect("replacement should survive restart");
        assert_eq!(restarted_again.document().revision, 3,);
    }

    #[test]
    fn populated_prose_requires_grounding() {
        let sections = SelfModelSections {
            affective_life: Some("Some situations acquire unusual salience.".to_string()),
            ..SelfModelSections::default()
        };
        let document = SelfModelDocument {
            schema_version: SELF_MODEL_SCHEMA_VERSION,
            revision: 1,
            created_at: "2026-09-02T00:00:00Z".to_string(),
            updated_at: "2026-09-02T00:00:00Z".to_string(),
            sections,
            grounding: Vec::new(),
        };

        assert!(SelfModelStore::validate_document_shape(&document).is_err());
    }

    #[test]
    fn weakness_tool_schema_requires_explicit_provenance_arrays() {
        let schema = weakness_schema();
        let required = schema["required"]
            .as_array()
            .expect("weakness schema should have required fields");

        assert!(
            required
                .iter()
                .any(|field| field.as_str() == Some("evidence_event_ids"))
        );
        assert!(
            required
                .iter()
                .any(|field| field.as_str() == Some("counterevidence_event_ids"))
        );
    }

    #[test]
    fn grounded_weakness_replacement_succeeds() {
        let data_dir = TestDir::new();
        let mut memory = Hippocampus::new(data_dir.path().join("hippocampus.jsonl"))
            .expect("hippocampus should initialize");
        memory
            .remember_operator("Repeated retries obscured the unchanged payload.".to_string())
            .expect("evidence should be persisted");

        let mut store =
            SelfModelStore::new(data_dir.path()).expect("living self-model should initialize");
        let sections = SelfModelSections {
            weaknesses_and_corrective_practices: vec![SelfWeakness {
                name: "Repeating an unchanged repair".to_string(),
                tendency: "May narrate a correction without changing the generated structure."
                    .to_string(),
                counter_practices: vec![
                    "Compare the next payload structurally with the rejected payload before retrying."
                        .to_string(),
                ],
                evidence_event_ids: vec![1],
                counterevidence_event_ids: Vec::new(),
                ..SelfWeakness::default()
            }],
            ..SelfModelSections::default()
        };

        let replaced = store
            .replace(
                1,
                sections,
                Vec::new(),
                vec![1],
                "Added a grounded corrective practice.".to_string(),
                &memory,
            )
            .expect("a weakness with explicit evidence should be accepted");

        assert_eq!(
            replaced.sections.weaknesses_and_corrective_practices[0].evidence_event_ids,
            vec![1]
        );
    }

    #[test]
    fn invalid_current_document_prevents_startup() {
        let data_dir = TestDir::new();
        fs::write(data_dir.path().join("self_model.json"), b"{broken json")
            .expect("invalid current document should be written");

        let error = SelfModelStore::new(data_dir.path())
            .err()
            .expect("invalid v2 must prevent startup")
            .to_string();
        assert!(error.contains("invalid current self-model"));
    }
}