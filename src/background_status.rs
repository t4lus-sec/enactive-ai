//! Background diagnostics must never write over an active Reedline edit buffer.
use std::{cell::Cell,collections::VecDeque,fmt,fs::OpenOptions,io::Write,path::{Path,PathBuf},sync::{Mutex,OnceLock}};
thread_local! {static BACKGROUND:Cell<bool>=const {Cell::new(false)};}
static PATH:OnceLock<PathBuf>=OnceLock::new();
static QUEUE:Mutex<VecDeque<String>>=Mutex::new(VecDeque::new());
const MAX_NOTICES:usize=32;
pub(crate) fn configure_background_log(data_dir:&Path) {
    let _=PATH.set(data_dir.join("sleep-background.log"));
}
pub(crate) fn background_scope<T>(run:impl FnOnce()->T)->T {
    struct Reset(bool);impl Drop for Reset {fn drop(&mut self){BACKGROUND.with(|v|v.set(self.0));}}
    let previous=BACKGROUND.with(|v|v.replace(true));let _reset=Reset(previous);run()
}
pub(crate) fn status(args:fmt::Arguments<'_>) {
    // Failures stay in view; routine progress is harness detail, shown expanded.
    // A step that will be tried again is progress too: if the tries run out,
    // the error that ends them is reported on its own.
    if BACKGROUND.with(Cell::get) {background_status(args);}
    else {let message=args.to_string();if is_important(&message)&&!is_retry(&message) {crate::terminal::warn(&message);} else {crate::terminal::say_note(&message);}}
}
/// A deep sleep's progress line: shown as `status` shows it, and kept in the
/// sleep log too (background sleeps already write every line there).
pub(crate) fn sleep_status(args:fmt::Arguments<'_>) {
    if BACKGROUND.with(Cell::get) {background_status(args);return;}
    let message=args.to_string();
    background_log(&message);
    status(format_args!("{message}"));
}
/// A line for the sleep log only.
pub(crate) fn background_log(message:&str) {
    if let Some(path)=PATH.get() {
        let record=format!("{} {message}\n",chrono::Utc::now().to_rfc3339());
        let _=OpenOptions::new().create(true).append(true).open(path).and_then(|mut file|file.write_all(record.as_bytes()));
    }
}
/// `[llm] retrying transient read failure (2/3)`, `[router] attempt 1/3 failed`.
fn is_retry(message:&str)->bool {
    message.contains("retrying")||message.contains("regenerating")||message.contains("resampling")
        ||message.split("attempt ").skip(1).any(|rest|rest.split_once(" failed").is_some_and(|(count,_)|count.contains('/')&&count.chars().all(|c|c.is_ascii_digit()||c=='/')))
}
pub(crate) fn background_status(args:fmt::Arguments<'_>) {
    let message=args.to_string();
    // Queue only meaningful notices; routine budgets/retries stay in the log.
    let important=is_important(&message);
    let record=format!("{} {message}\n",chrono::Utc::now().to_rfc3339());
    let error=PATH.get().map(|path|{
        OpenOptions::new().create(true).append(true).open(path)
            .and_then(|mut file|file.write_all(record.as_bytes()))
    });
    if important {queue(message.chars().take(600).collect::<String>().to_string());}
    if let Some(Err(error))=error {queue(format!("[sleep log] unable to write background diagnostics: {error}"));}
}
fn is_important(message:&str)->bool {
    message.contains("failed")||message.contains("failure")||message.contains("deferred")||message.contains("reserved")||message.contains("stopped")
}
fn queue(message:String) {
    // Recover a poisoned diagnostic queue rather than panicking or printing on a worker.
    let mut queue=QUEUE.lock().unwrap_or_else(|e|e.into_inner());
    if queue.back()==Some(&message){return;}
    if queue.len()==MAX_NOTICES {queue.pop_front();}
    queue.push_back(message);
}
pub(crate) fn take_background_status()->Vec<String> {
    QUEUE.lock().unwrap_or_else(|e|e.into_inner()).drain(..).collect()
}
/// What queued up while the operator was away, shown when they next send a
/// message: one notice as it is; several as one short line saying how many and
/// from where, with each of them in the expanded view (Ctrl+O).
pub(crate) fn report_background_status() {
    let (notes,line)=background_report(take_background_status());
    for note in &notes {crate::terminal::say_note(note);}
    match line {
        // A compaction notice reads like the rest of compaction's lines.
        Some(line) if line.starts_with("[compaction]") => crate::terminal::warn_status(&line),
        Some(line) => crate::terminal::warn(&line),
        None => {}
    }
}
fn background_report(mut notices:Vec<String>)->(Vec<String>,Option<String>) {
    match notices.len() {
        0=>(notices,None),
        1=>(Vec::new(),notices.pop()),
        count=>{
            // `vm-wake ×2, compaction`: each notice's `[tag]`, in order first seen.
            let mut sources:Vec<(&str,usize)>=Vec::new();
            for notice in &notices {
                let source=notice.strip_prefix('[').and_then(|rest|rest.split_once(']')).map_or("other",|(tag,_)|tag);
                match sources.iter_mut().find(|(seen,_)|*seen==source) {Some((_,n))=>*n+=1,None=>sources.push((source,1))}
            }
            let sources:Vec<String>=sources.iter().take(4).map(|(source,n)|if *n>1 {format!("{source} ×{n}")} else {source.to_string()}).collect();
            let line=format!("[background] {count} notices ({}) · ctrl+o to read them",sources.join(", "));
            (notices,Some(line))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retries_in_progress_are_not_warnings() {
        assert!(is_retry("[llm] retrying transient read failure (2/3): timed out"));
        assert!(is_retry("[router] attempt 1/3 failed: not JSON"));
        assert!(is_retry("[llm] incomplete/malformed response; regenerating (2/3)."));
        assert!(!is_retry("[compaction] stopped: disk full"));
        assert!(!is_retry("[signal] writing state.json failed: permission denied"));
        assert!(!is_retry("[vm-wake] reading the channel again after 3 failed attempt(s)"));
    }
    #[test]
    fn several_background_notices_are_one_line() {
        assert_eq!(background_report(Vec::new()),(Vec::new(),None));
        let one=vec!["[compaction] stopped: disk full".to_string()];
        assert_eq!(background_report(one.clone()),(Vec::new(),Some(one[0].clone())));
        let several=vec![
            "[vm-wake] reading the channel failed (reset); retrying".to_string(),
            "[vm-wake] reading the channel again after 3 failed attempt(s)".to_string(),
        ];
        let (notes,line)=background_report(several.clone());
        assert_eq!(notes,several,"each still reaches the expanded view");
        assert_eq!(line.unwrap(),"[background] 2 notices (vm-wake ×2) · ctrl+o to read them");
        let (_,line)=background_report(vec!["[compaction] stopped: disk full".into(),"x".repeat(500),"[signal] writing a.json failed".into()]);
        assert_eq!(line.unwrap(),"[background] 3 notices (compaction, other, signal) · ctrl+o to read them");
    }
}
