//! How well the agent finds what it knows: a retrieval eval over its own facts
//! (`claude/fact-recall-spec.md`, "Retrieval eval").
//!
//! `s0ma --recall-eval generate [n]` writes cases to `data/recall_eval.jsonl`
//! from the facts on record, asking the model for each question so the
//! wording is not the fact's own:
//!
//! - **single:** a question whose answer is one current fact.
//! - **update:** about a fact that has been replaced; the replacement must
//!   come back, and the old one must not come first.
//! - **point_in_time:** what was true before a fact was replaced; searched
//!   with `include: all` and `until`, and the old fact must come back.
//! - **abstention:** questions about things no fact covers. Nothing should
//!   match well; their best cosine, set beside the answerable cases', is
//!   the bar a pushed fact should clear.
//!
//! Cases are kept, so a later run measures the same questions; hand-written
//! ones can be added to the file in the same shape.
//!
//! `s0ma --recall-eval` (or `run`) searches each case with every arm alone
//! (entity, keyword, cosine) and fused, and reports recall at 1, 3, 5 and 10
//! and mean reciprocal rank per arm and category, how often a replaced fact
//! came first, and the cosine gap between answerable and abstention cases.
//! The report goes to stdout and `data/recall_eval_report.json`.

use crate::facts::{Arms, Args, Include};
use crate::hippocampus::Hippocampus;
use crate::world_model::Record;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Case {
    pub id: String,
    pub category: String,
    pub query: String,
    /// Any of these found counts.
    #[serde(default)]
    pub expect: Vec<String>,
    /// Must not come before what is expected (a replaced fact).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub not_first: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub until: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub include_all: bool,
}

pub const CATEGORIES: &[&str] = &["single", "update", "point_in_time", "abstention"];

fn cases_path(data: &Path) -> std::path::PathBuf {
    data.join("recall_eval.jsonl")
}

pub fn load_cases(data: &Path) -> Result<Vec<Case>> {
    let path = cases_path(data);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut cases = Vec::new();
    for (n, line) in text.lines().enumerate().filter(|(_, line)| !line.trim().is_empty()) {
        cases.push(serde_json::from_str(line).map_err(|error| format!("{} line {}: {error}", path.display(), n + 1))?);
    }
    Ok(cases)
}

// ------------------------------------------------------------------ generate

/// What the cases are made from: the records, as the world model holds them.
fn records(memory: &Hippocampus) -> Result<Vec<Record>> {
    let db = crate::world_model::open(&crate::world_model::path_for(memory))?;
    let bodies: Vec<String> = db
        .prepare("SELECT body FROM wm_records ORDER BY id")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(bodies.into_iter().filter_map(|body| serde_json::from_str(&body).ok()).collect())
}

/// A spread of `n` from `items`, the same every time for the same items.
fn spread<T: Clone>(items: &[T], n: usize) -> Vec<T> {
    if items.len() <= n {
        return items.to_vec();
    }
    (0..n).map(|i| items[i * items.len() / n].clone()).collect()
}

/// The cases to ask for, before the model words them: (category, the fact
/// to word a question about, expected IDs, IDs not to come first, until).
pub(crate) struct Plan {
    pub category: &'static str,
    pub about: String,
    pub expect: Vec<String>,
    pub not_first: Vec<String>,
    pub until: Option<String>,
}

pub(crate) fn plan(records: &[Record], n: usize) -> Vec<Plan> {
    let by_id: BTreeMap<&str, &Record> = records.iter().map(|r| (r.id.as_str(), r)).collect();
    let current = |r: &Record| r.superseded_by.is_none() && !matches!(r.status.as_str(), "retired" | "contradicted");
    let facts: Vec<&Record> =
        records.iter().filter(|r| r.kind != "entity" && current(r) && !r.supporting_evidence_ids.is_empty()).collect();
    let replaced: Vec<&Record> = records.iter().filter(|r| r.kind != "entity" && r.superseded_by.is_some()).collect();
    let mut plans = Vec::new();
    for fact in spread(&facts, n) {
        plans.push(Plan {
            category: "single",
            about: format!("{}: {}", fact.title, fact.statement),
            expect: vec![fact.id.clone()],
            not_first: Vec::new(),
            until: None,
        });
    }
    for old in spread(&replaced, (n / 4).max(1)) {
        // What replaced it, at the end of the chain.
        let mut at = old;
        for _ in 0..32 {
            match at.superseded_by.as_deref().and_then(|next| by_id.get(next)) {
                Some(next) => at = next,
                None => break,
            }
        }
        if at.id == old.id {
            continue;
        }
        plans.push(Plan {
            category: "update",
            about: format!("{} (now: {})", old.title, at.statement),
            expect: vec![at.id.clone()],
            not_first: vec![old.id.clone()],
            until: None,
        });
        if let Some(until) = &old.valid_until {
            plans.push(Plan {
                category: "point_in_time",
                about: format!("{} (before {}: {})", old.title, &until[..until.len().min(10)], old.statement),
                expect: vec![old.id.clone()],
                not_first: Vec::new(),
                until: Some(until.clone()),
            });
        }
    }
    plans
}

fn ask(prompt: String) -> Result<String> {
    let messages = [
        crate::types::Message::system(
            "You write test questions for a memory search. Answer with the question only, on one line, no quotes."
                .into(),
        ),
        crate::types::Message::user(prompt),
    ];
    let mut options = crate::llm::ChatOptions::instruct();
    options.max_tokens = 120;
    let reply = crate::llm::chat_quiet(&messages, &options)?;
    let line = reply.content.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("").trim_matches('"').to_string();
    if line.is_empty() {
        return Err("the model gave no question".into());
    }
    Ok(line)
}

/// Ask the model for each planned question and for abstention ones; add
/// them to the cases file (cases already there are kept). How many added.
pub fn generate(memory: &Hippocampus, data: &Path, n: usize) -> Result<usize> {
    let records = records(memory)?;
    let plans = plan(&records, n);
    if plans.is_empty() {
        return Err("no facts with evidence on record yet: nothing to make cases from".into());
    }
    let mut cases = load_cases(data)?;
    let known: std::collections::BTreeSet<String> = cases.iter().map(|c| c.query.clone()).collect();
    let mut added = Vec::new();
    for (i, plan) in plans.iter().enumerate() {
        let wording = match plan.category {
            "update" => "Write a question asking how this is NOW, without its exact words",
            "point_in_time" => "Write a question asking how this WAS at that earlier time, without its exact words",
            _ => "Write a question a person might ask whose answer is this fact, without copying its wording",
        };
        let query = ask(format!("{wording}:\n\n{}", plan.about))?;
        eprintln!("[{}/{}] {}: {query}", i + 1, plans.len(), plan.category);
        if known.contains(&query) {
            continue;
        }
        added.push(Case {
            id: format!("{}-{}", plan.category, cases.len() + added.len() + 1),
            category: plan.category.into(),
            query,
            expect: plan.expect.clone(),
            not_first: plan.not_first.clone(),
            until: plan.until.clone(),
            include_all: plan.category == "point_in_time",
        });
    }
    // Abstention: about the same world, but nothing on record answers it.
    let titles: Vec<String> = spread(&records, 40).iter().map(|r| r.title.clone()).collect();
    for k in 0..(n / 4).max(2) {
        let query = ask(format!(
            "These are topics a memory holds facts about:\n{}\n\nWrite question {} of several: a plausible question in the \
             same setting that NONE of these topics answer (a different machine, person, file or decision). Make it \
             differ from the others.",
            titles.join("\n"),
            k + 1
        ))?;
        eprintln!("[abstention {}] {query}", k + 1);
        if known.contains(&query) || added.iter().any(|c: &Case| c.query == query) {
            continue;
        }
        added.push(Case {
            id: format!("abstention-{}", cases.len() + added.len() + 1),
            category: "abstention".into(),
            query,
            expect: Vec::new(),
            not_first: Vec::new(),
            until: None,
            include_all: false,
        });
    }
    let count = added.len();
    cases.extend(added);
    let text: String = cases.iter().map(|c| serde_json::to_string(c).map(|l| l + "\n")).collect::<std::result::Result<_, _>>()?;
    std::fs::write(cases_path(data), text)?;
    Ok(count)
}

// ----------------------------------------------------------------------- run

/// One arm's results over a category.
#[derive(Debug, Default, Clone, Serialize, PartialEq)]
pub struct Score {
    pub cases: usize,
    pub at1: usize,
    pub at3: usize,
    pub at5: usize,
    pub at10: usize,
    pub reciprocal_rank: f64,
    /// A replaced fact came before its replacement (update cases).
    pub stale_first: usize,
    /// Something came back at all.
    pub answered: usize,
    /// The best cosine in each case (answerable: of what was expected;
    /// abstention: of anything), for the push bar.
    #[serde(skip)]
    pub cosines: Vec<f32>,
}

impl Score {
    pub fn add(&mut self, rank: Option<usize>, stale_first: bool, cosine: Option<f32>, answered: bool) {
        self.cases += 1;
        self.answered += usize::from(answered);
        if let Some(rank) = rank {
            self.at1 += usize::from(rank <= 1);
            self.at3 += usize::from(rank <= 3);
            self.at5 += usize::from(rank <= 5);
            self.at10 += usize::from(rank <= 10);
            self.reciprocal_rank += 1.0 / rank as f64;
        }
        self.stale_first += usize::from(stale_first);
        if let Some(cosine) = cosine {
            self.cosines.push(cosine);
        }
    }
    fn share(&self, n: usize) -> String {
        if self.cases == 0 {
            "-".into()
        } else {
            format!("{:.0}%", 100.0 * n as f64 / self.cases as f64)
        }
    }
    fn mrr(&self) -> String {
        if self.cases == 0 {
            "-".into()
        } else {
            format!("{:.2}", self.reciprocal_rank / self.cases as f64)
        }
    }
}

fn median(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    Some(sorted[sorted.len() / 2])
}

/// The scores of `cases`, by arm and category, searched by `search` (an
/// arm and a case to the IDs found in order, each with its cosine).
pub fn score(
    cases: &[Case],
    mut search: impl FnMut(Arms, &Case) -> Result<Vec<(String, Option<f32>)>>,
) -> Result<BTreeMap<(&'static str, String), Score>> {
    let mut scores: BTreeMap<(&'static str, String), Score> = BTreeMap::new();
    for case in cases {
        for (name, arms) in [("fused", Arms::All), ("entity", Arms::Entity), ("keyword", Arms::Literal), ("cosine", Arms::Semantic)] {
            let found = search(arms, case)?;
            let rank = found.iter().position(|(id, _)| case.expect.contains(id)).map(|i| i + 1);
            let stale_first = !case.not_first.is_empty()
                && found
                    .iter()
                    .position(|(id, _)| case.not_first.contains(id))
                    .is_some_and(|stale| rank.is_none_or(|rank| stale + 1 < rank));
            let cosine = if case.expect.is_empty() {
                found.iter().filter_map(|(_, c)| *c).reduce(f32::max)
            } else {
                rank.and_then(|r| found[r - 1].1)
            };
            for category in [case.category.clone(), "all".to_string()] {
                if category == "all" && case.category == "abstention" {
                    continue;
                }
                scores.entry((name, category)).or_default().add(rank, stale_first, cosine, !found.is_empty());
            }
        }
    }
    Ok(scores)
}

pub fn report(scores: &BTreeMap<(&'static str, String), Score>) -> String {
    let mut out = String::from("arm      category        cases   @1    @3    @5    @10   MRR   stale-first\n");
    for arm in ["fused", "entity", "keyword", "cosine"] {
        for category in CATEGORIES.iter().filter(|c| **c != "abstention").map(|c| c.to_string()).chain(["all".to_string()]) {
            let Some(score) = scores.get(&(arm, category.clone())) else { continue };
            out.push_str(&format!(
                "{:<8} {:<15} {:>5}  {:>4}  {:>4}  {:>4}  {:>4}  {:>4}  {:>5}\n",
                arm,
                category,
                score.cases,
                score.share(score.at1),
                score.share(score.at3),
                score.share(score.at5),
                score.share(score.at10),
                score.mrr(),
                if category == "update" || category == "all" { score.stale_first.to_string() } else { String::new() },
            ));
        }
    }
    let answerable = scores.get(&("cosine", "all".to_string())).and_then(|s| median(&s.cosines));
    let abstaining = scores.get(&("cosine", "abstention".to_string()));
    if let Some(abstaining) = abstaining {
        out.push_str(&format!(
            "\nAbstention: {} case(s). Fused, something came back for {} of them (a search always offers its best).\n",
            abstaining.cases,
            scores.get(&("fused", "abstention".to_string())).map_or(0, |s| s.answered)
        ));
        match (answerable, median(&abstaining.cosines)) {
            (Some(yes), Some(no)) => out.push_str(&format!(
                "Median best cosine: {yes:.3} where a fact answered, {no:.3} where none did. A push bar between the \
                 two keeps most of the first and drops most of the second.\n"
            )),
            _ => out.push_str("No cosine to compare: the cosine arm did not run (embedder down?), or no cases.\n"),
        }
    }
    out
}

/// `s0ma --recall-eval [run]`.
pub fn run(memory: &Hippocampus, data: &Path) -> Result<String> {
    let cases = load_cases(data)?;
    if cases.is_empty() {
        return Err(format!(
            "no cases in {}: make some with `s0ma --recall-eval generate`",
            cases_path(data).display()
        )
        .into());
    }
    let embedder = crate::archive::Embedder::new();
    let scores = score(&cases, |arms, case| {
        let args = Args {
            query: case.query.clone(),
            subject: None,
            since: None,
            until: case.until.clone(),
            include: if case.include_all { Include::All } else { Include::Current },
            limit: 10,
        };
        let found = match &embedder {
            Ok(embedder) => crate::facts::find_with(memory, &args, arms, Ok(embedder)),
            Err(error) => crate::facts::find_with(memory, &args, arms, Err(error.to_string())),
        }?;
        Ok(found.hits.into_iter().map(|hit| (hit.id, hit.cosine)).collect())
    })?;
    let text = report(&scores);
    let saved = json!({
        "at": chrono::Utc::now().to_rfc3339(),
        "cases": cases.len(),
        "scores": scores.iter().map(|((arm, category), score)| json!({"arm": arm, "category": category, "score": score})).collect::<Vec<Value>>(),
        "report": text,
    });
    std::fs::write(data.join("recall_eval_report.json"), serde_json::to_string_pretty(&saved)?)?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_count_ranks_stale_firsts_and_keep_the_cosines_for_the_bar() {
        let cases = vec![
            Case { id: "a".into(), category: "single".into(), query: "q1".into(), expect: vec!["x".into()], not_first: vec![], until: None, include_all: false },
            Case { id: "b".into(), category: "update".into(), query: "q2".into(), expect: vec!["new".into()], not_first: vec!["old".into()], until: None, include_all: false },
            Case { id: "c".into(), category: "abstention".into(), query: "q3".into(), expect: vec![], not_first: vec![], until: None, include_all: false },
        ];
        let scores = score(&cases, |_, case| {
            Ok(match case.query.as_str() {
                "q1" => vec![("y".into(), Some(0.5)), ("x".into(), Some(0.8))],
                "q2" => vec![("old".into(), Some(0.7)), ("new".into(), Some(0.6))],
                _ => vec![("z".into(), Some(0.3))],
            })
        })
        .unwrap();
        let fused = &scores[&("fused", "all".to_string())];
        assert_eq!((fused.cases, fused.at1, fused.at3), (2, 0, 2), "abstention left out of all");
        assert!((fused.reciprocal_rank - 1.0).abs() < 1e-9, "1/2 + 1/2");
        assert_eq!(scores[&("fused", "update".to_string())].stale_first, 1);
        assert_eq!(scores[&("cosine", "abstention".to_string())].cosines, [0.3]);
        let text = report(&scores);
        assert!(text.contains("Median best cosine: 0.800 where a fact answered, 0.300 where none did"), "{text}");
        assert!(text.lines().any(|l| l.starts_with("fused    single")), "{text}");
    }
}
