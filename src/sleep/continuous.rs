//! Descriptive compaction and daily review have independent durable boundaries.
macro_rules! compaction_status { ($($args:tt)*) => { crate::llm::background_status(format_args!($($args)*)) }; }
use super::*;
use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const WINDOW_BYTES: usize = 96_000;

/// Eligibility protection. The turn count is the default; the byte ceiling is what
/// stops one tool-heavy reply from pinning the entire post-checkpoint range, and
/// the minimum is the floor that keeps recent dialogue verbatim in rotated context.
const PROTECTED_TAIL_TURNS: usize = 8;
const PROTECTED_TAIL_BYTES: usize = 60_000;
const PROTECTED_TAIL_MIN_TURNS: usize = 2;

/// Default character budget for the compaction digests folded into a runtime
/// checkpoint. `rotate_context` tightens below this when the rebuilt context
/// misses its token target.
pub const DIGEST_BUDGET_CHARS: usize = 48_000;

/// What the compactor last did, for `diagnostic`: whether it is sealing now,
/// since when it has waited for a worker slot, and its last failure.
#[derive(Default)]
struct Activity {
    running: bool,
    waiting_since: Option<chrono::DateTime<chrono::Local>>,
    failed: Option<(chrono::DateTime<chrono::Local>, String)>,
}
static ACTIVITY: std::sync::Mutex<Activity> =
    std::sync::Mutex::new(Activity { running: false, waiting_since: None, failed: None });
fn activity(change: impl FnOnce(&mut Activity)) {
    if let Ok(mut activity) = ACTIVITY.lock() {
        change(&mut activity);
    }
}
fn ago(at: chrono::DateTime<chrono::Local>) -> String {
    let minutes = (chrono::Local::now() - at).num_minutes();
    match minutes {
        ..=0 => "just now".into(),
        1..=89 => format!("{minutes}m ago"),
        _ => format!("{}h ago", minutes / 60),
    }
}
#[derive(Serialize, Deserialize)]
pub(super) struct Window {
    pub key: String, pub ids: Vec<u64>, pub source_hash: String,
    pub batches: serde_json::Value, pub complete: bool,
}
pub(super) fn path_for(memory: &Hippocampus) -> PathBuf {
    memory.archive_path().with_file_name("continuous_compaction.sqlite3")
}
fn connect(path: &Path) -> Result<Connection> {
    let db = crate::db::open(path)?;
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS windows(key TEXT PRIMARY KEY,from_id INTEGER NOT NULL,through_id INTEGER NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);")?;
    Ok(db)
}
/// Drop every sealed window overlapping `ranges`. Returns how many went.
///
/// A window holds a model-written digest of the events it covers, so one
/// overlapping a redacted stretch holds part of it. Overlap rather than
/// containment, and a window straddling the edge of a job goes whole: its
/// payload is one piece of prose with no seam to cut along, so the choice is
/// between losing the unrelated part of it and keeping the job's part.
///
/// Compaction keeps running while a job is open -- unlike deep sleep, it writes
/// nothing long-lived and the context still has to be kept workable -- which is
/// exactly why this exists.
pub(crate) fn forget_ranges(memory: &Hippocampus, ranges: &[(u64, u64)]) -> Result<usize> {
    let path = path_for(memory);
    if !path.exists() || ranges.is_empty() {
        return Ok(0);
    }
    let mut db = connect(&path)?;
    let mut removed = 0;
    {
        let tx = db.transaction()?;
        for (from, through) in ranges {
            let from = (*from).min(i64::MAX as u64) as i64;
            let through = (*through).min(i64::MAX as u64) as i64;
            removed += tx.execute(
                "DELETE FROM windows WHERE through_id >= ?1 AND from_id <= ?2",
                rusqlite::params![from, through],
            )?;
        }
        tx.commit()?;
    }
    crate::hippocampus::compact(&db);
    Ok(removed)
}

/// Drop every sealed window whose digest holds one of `fragments`. Returns how
/// many went.
///
/// By what a window says, not what it covers: a window's payload is a summary
/// of its events, and one that never mentioned the struck text is worth
/// keeping. Compare `forget_ranges`, which a job's close uses because there the
/// whole stretch is going.
pub(crate) fn forget_holding(memory: &Hippocampus, fragments: &[String]) -> Result<usize> {
    let path = path_for(memory);
    if !path.exists() || fragments.is_empty() {
        return Ok(0);
    }
    let mut db = connect(&path)?;
    let mut removed = 0;
    {
        let tx = db.transaction()?;
        for fragment in fragments {
            removed += tx.execute("DELETE FROM windows WHERE instr(payload, ?1) > 0", rusqlite::params![fragment])?;
        }
        tx.commit()?;
    }
    if removed > 0 {
        crate::hippocampus::compact(&db);
    }
    Ok(removed)
}

pub(crate) fn source_hash(events: &[RecalledEvent]) -> Result<String> {
    let mut digest = Sha256::new(); let mut previous = None;
    for event in events {
        if previous.is_some_and(|id| event.id <= id) {return Err("archive IDs must be strictly increasing before advancing a cutoff".into());}
        previous = Some(event.id);
        crate::hippocampus::hash_event(&mut digest, event)?;
    }
    Ok(format!("{:x}",digest.finalize()))
}
pub(super) fn load(path: &Path) -> Result<Vec<Window>> {
    let db = connect(path)?;
    let mut statement = db.prepare("SELECT payload FROM windows ORDER BY from_id,through_id DESC,key")?;
    let rows = statement.query_map([],|row|row.get::<_,String>(0))?;
    let mut result=Vec::new();
    let mut skipped=0usize;
    for row in rows {
        let payload=row?;
        // Same reasoning as above: one unreadable window must not make the
        // compaction store unloadable, because startup depends on it.
        match serde_json::from_str(&payload) {
            Ok(window)=>result.push(window),
            Err(error)=>{
                skipped+=1;
                crate::terminal::warn_status(&format!("[compaction] dropping an unreadable window record ({error})"));
            }
        }
    }
    if skipped>0 { crate::terminal::warn_status(&format!("[compaction] {skipped} window record(s) unreadable; run /compact to rebuild them")); }
    Ok(result)
}
pub(super) fn save(path: &Path, window: &Window) -> Result<()> {
    let first=i64::try_from(*window.ids.first().ok_or("empty compaction window")?)?;
    let last=i64::try_from(*window.ids.last().unwrap())?;
    connect(path)?.execute("INSERT INTO windows VALUES(?1,?2,?3,?4) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
        params![window.key,first,last,serde_json::to_string(window)?])?;
    Ok(())
}
fn pending(path: &Path) -> Result<Option<usize>> {
    let value: Option<String> = connect(path)?.query_row("SELECT value FROM state WHERE key='pending_slot'",[],|r|r.get(0)).optional()?;
    value.map(|value|value.parse::<usize>().map_err(Into::into)).transpose()
}
fn set_pending(path: &Path, slot: Option<usize>) -> Result<()> {
    let db=connect(path)?;
    if let Some(slot)=slot {db.execute("INSERT INTO state VALUES('pending_slot',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[slot.to_string()])?;}
    else {db.execute("DELETE FROM state WHERE key='pending_slot'",[])?;}
    Ok(())
}
pub(super) fn prefix<'a>(events: &[RecalledEvent], windows: &'a [Window], complete_only: bool) -> Result<Vec<(&'a Window,usize)>> {
    let mut cursor=0;let mut result=Vec::new();
    while cursor<events.len() {
        let mut found=None;
        for window in windows {
            let count=window.ids.len();
            if count==0 || cursor+count>events.len() || (complete_only && !window.complete) || window.ids[0]!=events[cursor].id {continue;}
            let part=&events[cursor..cursor+count];
            if part.iter().map(|e|e.id).eq(window.ids.iter().copied()) && source_hash(part)?==window.source_hash {found=Some(window);break;}
        }
        let Some(window)=found else {break;};
        result.push((window,cursor));cursor+=window.ids.len();
    }
    Ok(result)
}
fn copy_event(event: &RecalledEvent) -> RecalledEvent {
    RecalledEvent{id:event.id,timestamp:event.timestamp.clone(),speaker:if matches!(event.speaker,Speaker::Operator){Speaker::Operator}else{Speaker::S0ma},content:event.content.clone()}
}
/// Only a runtime projection: never append it to the daily review checkpoint log.
pub fn runtime_checkpoint(memory: &Hippocampus) -> Result<Option<MemoryCheckpoint>> {
    runtime_checkpoint_budgeted(memory,DIGEST_BUDGET_CHARS)
}
/// Only a runtime projection: never append it to the daily review checkpoint log.
pub fn runtime_checkpoint_budgeted(memory: &Hippocampus,digest_chars: usize) -> Result<Option<MemoryCheckpoint>> {
    let reviewed=memory.latest_checkpoint()?;
    let after=reviewed.as_ref().map_or(0,|c|c.through_event_id);
    let events=memory.recall_after(after)?;
    let windows=load(&path_for(memory))?;
    let covered=prefix(&events,&windows,true)?;
    if covered.is_empty() {return Ok(reviewed);}
    let through=*covered.last().unwrap().0.ids.last().unwrap();
    let mut entries=Vec::new();
    // A damaged window is skipped, not fatal.
    //
    // These summaries are model output stored verbatim, so one that ran out of
    // room mid-JSON gets persisted and then fails to parse on every startup
    // afterwards — which is how a single truncated generation turned into a
    // process that would not boot. The archive underneath is lossless and these
    // windows are a cache over it, so the right response to a corrupt one is to
    // say so and carry on with less compaction, never to refuse to start.
    for (window,_) in &covered {
        let span=format!("#{}..#{}",window.ids[0],window.ids.last().unwrap());
        let Some(batches)=window.batches.as_array() else {
            crate::terminal::warn_status(&format!("[compaction] window {span}: batch list is not an array; skipping it"));
            continue;
        };
        let mut digests=Vec::new();
        let mut damaged=0usize;
        for batch in batches {
            let Some(summary)=batch["summary"].as_str() else {damaged+=1;continue;};
            let value:serde_json::Value=match serde_json::from_str(summary) {
                Ok(value)=>value,
                Err(error)=>{
                    crate::terminal::warn_status(&format!("[compaction] window {span}: batch summary did not parse ({error}); skipping that batch"));
                    damaged+=1;
                    continue;
                }
            };
            match value["memory_digest"].as_str() {
                Some(digest)=>digests.push(digest.to_owned()),
                None=>{damaged+=1;}
            }
        }
        if digests.is_empty() {
            crate::terminal::warn_status(&format!("[compaction] window {span}: no usable digest in {damaged} batch(es); the events stay in the archive and are reachable with archive_get"));
            continue;
        }
        if damaged>0 {
            crate::terminal::warn_status(&format!("[compaction] window {span}: {damaged} damaged batch(es) skipped, {} kept",digests.len()));
        }
        entries.push(format!("Events #{}..#{} (original evidence: archive_get):\n{}",window.ids[0],window.ids.last().unwrap(),digests.join("\n")));
    }
    let mut budget=digest_chars;let mut recent=Vec::new();let mut omitted=0;
    for entry in entries.into_iter().rev() {
        let size=entry.chars().count();
        if size<=budget {budget-=size;recent.push(entry);}else{omitted+=1;}
    }
    recent.reverse();
    let mut summary=reviewed.as_ref().map(|c|c.summary.clone()).unwrap_or_default();
    summary.push_str(&format!("\n\nCONTINUOUS COMPACTION — NOT YET DAILY-REVIEWED\nReviewed through #{after}; compacted through #{through}. Descriptive summaries are evidence, not canonical conclusions.\n"));
    if omitted>0 {summary.push_str(&format!("[{omitted} older digest windows omitted from this bounded runtime view. Full digests await daily review; original events remain available with archive_search/archive_get. Absence here is not evidence of absence.]\n"));}
    summary.push_str(&recent.join("\n\n"));
    Ok(Some(MemoryCheckpoint{timestamp:Local::now().to_rfc3339(),from_event_id:reviewed.as_ref().map_or(events[0].id,|c|c.from_event_id),through_event_id:through,
        source_event_count:events.iter().filter(|e|e.id<=through).count(),summary}))
}
/// Why a rotation found nothing, and which of the ways it can: nothing
/// eligible under the boundary; no window starting where the chain has to
/// go on; one there but not finished; one there that no longer matches the
/// archive under it. Then what the compactor is doing about it.
pub fn diagnostic(memory: &Hippocampus) -> Result<String> {
    let reviewed=memory.latest_checkpoint()?;
    let after=reviewed.as_ref().map_or(0,|c|c.through_event_id);
    let boundary=memory.compaction_boundary_budgeted(PROTECTED_TAIL_TURNS,PROTECTED_TAIL_BYTES,PROTECTED_TAIL_MIN_TURNS)?;
    let events=memory.recall_after(after)?;
    let eligible=events.iter().filter(|e|e.id<=boundary).count();
    let path=path_for(memory);
    let windows=load(&path)?;
    let complete=windows.iter().filter(|w|w.complete).count();
    let covered=prefix(&events,&windows,true)?;
    let through=covered.last().map(|(w,_)|*w.ids.last().unwrap());
    let stalled_at=covered.iter().map(|(w,_)|w.ids.len()).sum::<usize>();
    let mut line=format!(
        "reviewed through #{after}; eligibility cutoff #{boundary} ({eligible} of {} post-review events eligible); \
         {} windows ({complete} complete); covered prefix {}",
        events.len(),windows.len(),
        through.map_or_else(||"empty".to_owned(),|id|format!("through #{id}")),
    );
    if let Some(next)=events.get(stalled_at) {
        line.push_str(&format!("; prefix stops before #{}: {}",next.id,why_stopped(&events[stalled_at..],&windows,boundary)?));
    }
    if let Ok(activity)=ACTIVITY.lock() {
        if activity.running {
            line.push_str("; the compactor is sealing now");
        } else if let Some(since)=activity.waiting_since {
            line.push_str(&format!("; the compactor has waited for a free worker slot since {}",ago(since)));
        }
        if let Some((at,error))=&activity.failed {
            line.push_str(&format!("; its last attempt failed {}: {error}",ago(*at)));
        }
    }
    Ok(line)
}

/// Why the chain of windows does not go on at `rest[0]`.
fn why_stopped(rest: &[RecalledEvent], windows: &[Window], boundary: u64) -> Result<String> {
    let first=rest[0].id;
    if first>boundary {
        return Ok("it is past the eligibility cutoff (the protected tail)".into());
    }
    let starting:Vec<&Window>=windows.iter().filter(|w|w.ids.first()==Some(&first)).collect();
    if starting.is_empty() {
        return Ok(format!("no window starts there yet (the compactor seals from #{first})"));
    }
    let mut reasons=Vec::new();
    for window in starting.iter().take(3) {
        let span=format!("#{}..#{}",first,window.ids.last().unwrap());
        let count=window.ids.len();
        let ids_match=count<=rest.len() && rest[..count].iter().map(|e|e.id).eq(window.ids.iter().copied());
        if !ids_match {
            reasons.push(format!("window {span} covers events the archive no longer has in that order"));
        } else if source_hash(&rest[..count])?!=window.source_hash {
            reasons.push(format!("window {span} no longer matches the archive (an event in it changed after it was sealed: a redaction or a job's close); it is sealed again on the next pass"));
        } else if !window.complete {
            let batches=window.batches.as_array().map_or(0,Vec::len);
            let done=window.batches.as_array().map_or(0,|b|b.iter().filter(|b|!b["summary"].is_null()).count());
            reasons.push(format!("window {span} is not finished ({done} of {batches} batches digested)"));
        } else if count>rest.len() {
            reasons.push(format!("window {span} runs past the newest event"));
        }
    }
    if reasons.is_empty() {
        reasons.push("a window starts there but does not join the chain".into());
    }
    Ok(reasons.join("; "))
}

fn unprocessed(memory: &Hippocampus, path: &Path) -> Result<Vec<RecalledEvent>> {
    let after=memory.latest_checkpoint()?.map_or(0,|c|c.through_event_id);
    let boundary=memory.compaction_boundary_budgeted(PROTECTED_TAIL_TURNS,PROTECTED_TAIL_BYTES,PROTECTED_TAIL_MIN_TURNS)?;
    let mut events=memory.recall_after(after)?;events.retain(|e|e.id<=boundary);
    let windows=load(path)?;
    let count=prefix(&events,&windows,true)?.iter().map(|(window,_)|window.ids.len()).sum();
    Ok(events.split_off(count))
}
fn seal(events: &[RecalledEvent]) -> SleepInput {
    let mut bytes=0;let mut selected=Vec::new();
    for event in events {if bytes>=WINDOW_BYTES {break;}bytes+=event.content.len();selected.push(copy_event(event));}
    SleepInput{previous_checkpoint:None,source_events:selected}
}
fn run_reserved(input: SleepInput,path: PathBuf,mut lease: crate::subagent::MaintenanceLease,cancel: Arc<AtomicBool>) -> std::result::Result<(),String> {
    llm::background_scope(|| {
    let outcome=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Process a frozen eligible prefix. Never chase new events while the agent runs.
        let mut cursor = 0;
        while cursor < input.source_events.len() {
            if cancel.load(Ordering::SeqCst) { return Err("compaction cancelled; saved windows retained".into()); }
            let window = seal(&input.source_events[cursor..]);
            incremental::compact_window(&window,&path,lease.slot,lease.capacity,&cancel)?;
            cursor += window.source_events.len();
        }
        Ok(())
    }))
        .unwrap_or_else(|_|Err("background compaction panicked; receipts retained".into()));
    let cleanup=lease.release().map_err(|e|e.to_string());
    if cleanup.is_ok() {set_pending(&path,None).map_err(|e|e.to_string())?;}
    cleanup?;outcome
    })
}

pub struct Worker {
    path:PathBuf,handle:Option<std::thread::JoinHandle<std::result::Result<(),String>>>,
    cancel:Arc<AtomicBool>,next:Instant,failures:u32,
}
impl Worker {
    pub fn new(memory: &Hippocampus) -> Result<Self> {
        Ok(Self{path:path_for(memory),handle:None,cancel:Arc::new(AtomicBool::new(false)),next:Instant::now(),failures:0})
    }
    fn cleanup(&self) -> Result<()> {
        if let Some(slot)=pending(&self.path)? {crate::subagent::recover_maintenance(slot)?;set_pending(&self.path,None)?;}
        Ok(())
    }
    pub fn tick(&mut self,memory: &Hippocampus) -> Result<()> {
        if self.handle.as_ref().is_some_and(|h|h.is_finished()) {
            let outcome=self.handle.take().unwrap().join().unwrap_or_else(|_|Err("compactor panicked".into()));
            activity(|a|{a.running=false;a.failed=outcome.as_ref().err().map(|e|(chrono::Local::now(),e.clone()));});
            match outcome {
                Ok(())=>{self.failures=0;self.next=Instant::now();},
                Err(error)=>{self.failures=self.failures.saturating_add(1);self.next=Instant::now()+Duration::from_secs((5*(1u64<<self.failures.min(6))).min(300));
                    compaction_status!("[compaction] deferred; saved batches retained: {error}");}
            }
        }
        if self.handle.is_some() || Instant::now()<self.next {return Ok(());}
        self.next=Instant::now()+Duration::from_secs(30);self.cleanup()?;
        let events=unprocessed(memory,&self.path)?;
        if events.is_empty() {return Ok(());}
        let Some(lease)=crate::subagent::reserve_maintenance(32_000,22_528)? else {
            activity(|a|{a.waiting_since.get_or_insert_with(chrono::Local::now);});
            return Ok(());
        };
        activity(|a|{a.waiting_since=None;a.running=true;});
        set_pending(&self.path,Some(lease.slot))?;
        let slot=lease.slot;
        let input=SleepInput { previous_checkpoint: None, source_events: events };
        let path=self.path.clone();
        self.cancel=Arc::new(AtomicBool::new(false));let cancel=self.cancel.clone();
        self.handle=Some(std::thread::Builder::new().name("continuous-compaction".into()).spawn(move||run_reserved(input,path,lease,cancel))?);
        compaction_status!("[compaction] sealed completed history on worker slot {slot}");Ok(())
    }
    pub fn has_job(&self) -> bool { self.handle.is_some() }

    pub fn quiesce(&mut self) -> Result<()> {
        self.cancel.store(true,Ordering::SeqCst);
        if let Some(handle)=self.handle.take() {
            compaction_status!("[compaction] waiting for active request cleanup");
            match handle.join(){Ok(Ok(()))=>{},Ok(Err(e))=>compaction_status!("[compaction] stopped: {e}"),Err(_)=>compaction_status!("[compaction] stopped after panic")}
        }
        self.cleanup()?;self.next=Instant::now();Ok(())
    }
    pub fn compact_next(&mut self, memory: &Hippocampus) -> Result<bool> {
        self.quiesce()?;
        let events = unprocessed(memory, &self.path)?;
        if events.is_empty() { return Ok(false); }
        let Some(lease) = crate::subagent::reserve_maintenance(32_000, 22_528)? else {
            return Err("compaction needs an idle worker reservation".into());
        };
        set_pending(&self.path, Some(lease.slot))?;
        let input = seal(&events);
        crate::terminal::say_note(&format!("[compaction] events #{}..#{}; newest {PROTECTED_TAIL_TURNS} completed turns protected, \
            or fewer (never below {PROTECTED_TAIL_MIN_TURNS}) if that tail exceeds {PROTECTED_TAIL_BYTES} bytes",
            input.source_events.first().unwrap().id, input.source_events.last().unwrap().id));
        run_reserved(input, self.path.clone(), lease, Arc::new(AtomicBool::new(false)))?;
        Ok(true)
    }
    pub fn flush(&mut self, memory: &Hippocampus) -> Result<()> {
        while self.compact_next(memory)? {}
        Ok(())
    }
}
impl Drop for Worker {fn drop(&mut self){self.cancel.store(true,Ordering::SeqCst);}}
#[derive(Serialize,Deserialize)]
struct ReviewWindow {baseline:u64,through:u64,source_hash:String}
pub(super) fn prepare_review(memory: &Hippocampus) -> Result<Option<SleepInput>> {
    let previous_checkpoint=memory.latest_checkpoint()?;
    let baseline=previous_checkpoint.as_ref().map_or(0,|c|c.through_event_id);
    let path=memory.archive_path().with_file_name("daily_sleep_window.json");
    let saved:Option<ReviewWindow>=SleepStateStore::read_json(&path)?;
    let mut source_events=memory.recall_after(baseline)?;
    if let Some(saved)=saved.filter(|s|s.baseline==baseline && s.through>baseline) {
        source_events.retain(|e|e.id<=saved.through);
        if source_hash(&source_events)?!=saved.source_hash{return Err("daily review source changed; inspect the retained cutoff before retry".into());}
    } else if let Some(last)=source_events.last() {
        SleepStateStore::write_json(&path,&ReviewWindow{baseline,through:last.id,source_hash:source_hash(&source_events)?})?;
    }
    if source_events.is_empty(){return Ok(None);}
    Ok(Some(SleepInput{previous_checkpoint,source_events}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(id: u64, content: &str) -> RecalledEvent {
        RecalledEvent { id, timestamp: "2026-11-04T10:00:00+00:00".into(), speaker: Speaker::S0ma, content: content.into() }
    }

    fn window(events: &[RecalledEvent], complete: bool) -> Window {
        Window {
            key: "k".into(),
            ids: events.iter().map(|e| e.id).collect(),
            source_hash: source_hash(events).unwrap(),
            batches: serde_json::json!([{"summary": "done"}, {"summary": if complete { serde_json::json!("done") } else { serde_json::Value::Null }}]),
            complete,
        }
    }

    /// Each way the chain can stop at an event is named as itself.
    #[test]
    fn why_the_chain_stops_is_named() {
        let events = vec![event(5, "a"), event(6, "b"), event(7, "c")];
        assert!(why_stopped(&events, &[], 7).unwrap().contains("no window starts there"));
        assert!(why_stopped(&events, &[], 4).unwrap().contains("past the eligibility cutoff"));
        let unfinished = window(&events[..2], false);
        assert!(why_stopped(&events, &[unfinished], 7).unwrap().contains("not finished (1 of 2 batches"), "{}", why_stopped(&events, &[window(&events[..2], false)], 7).unwrap());
        let mut sealed = window(&events[..2], true);
        sealed.source_hash = source_hash(&[event(5, "a before a redaction"), event(6, "b")]).unwrap();
        assert!(why_stopped(&events, &[sealed], 7).unwrap().contains("no longer matches the archive"));
    }
}
