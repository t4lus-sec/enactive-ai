use super::*;
use std::collections::HashSet;
use crate::board::{BoardStore, GoalTestKind, Polarity};

/// A projects root and a data folder of their own.
fn fresh() -> (PathBuf, PathBuf, BoardStore) {
    let base = log::tests::scratch();
    let root = base.join("projects");
    let data = base.join("data");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    set_for_test(Some(root.clone()), &data);
    let boards = BoardStore::beside_archive(&data.join("archive.jsonl"));
    (root, data, boards)
}

fn open(boards: &BoardStore, name: &str) -> crate::board::Board {
    boards.open_board(name, name, "an edit sometimes disappears", "the edit stays", "true", GoalTestKind::Command, 60).unwrap().unwrap()
}

/// Every change a board's store makes is a line in its file, and the file
/// alone rebuilds it.
#[test]
fn a_board_is_written_to_its_file_and_rebuilt_from_it() {
    let (root, _, boards) = fresh();
    let board = open(&boards, "edit-lost");
    let fact = boards.operator_fact(&board.id, "the save runs twice", Polarity::Supports, None, None).unwrap();
    let path = root.join("inbox").join("edit-lost.board");
    assert!(path.exists(), "made in inbox/");
    let file = log::read(&path).unwrap();
    assert_eq!(file.id(), board.id);
    assert!(file.lines.iter().any(|l| log::op(l) == "board"), "{:?}", file.lines);
    assert!(file.lines.iter().any(|l| log::op(l) == "finding" && l["row"]["statement"] == "the save runs twice"));
    // Gone from the index, then back from the file alone.
    journal::remove(Kind::Board, &board.id).unwrap();
    assert!(boards.board("edit-lost").unwrap().is_none());
    journal::replay(Kind::Board, &board.id, &file).unwrap();
    let back = boards.board("edit-lost").unwrap().expect("rebuilt");
    assert_eq!(back.goal, board.goal);
    assert_eq!(boards.fact(&board.id, &fact.id).unwrap().unwrap().statement, "the save runs twice");
    // A rolled-back write writes nothing.
    let before = log::read(&path).unwrap().lines.len();
    assert!(boards.remove_hint(&board.id, "hint-9999").is_ok());
    assert_eq!(log::read(&path).unwrap().lines.len(), before);
}

/// Moved, archived, trashed and restored on disk: the index follows.
#[test]
fn moves_archives_and_the_trash_are_followed() {
    let (root, _, boards) = fresh();
    let board = open(&boards, "edit-lost");
    scan();
    // A project and a folder, made as the tab makes them.
    let project = tree::create_folder("", "calc").unwrap();
    let folder = tree::create_folder(&project, "parser").unwrap();
    assert!(root.join("calc/parser/index.md").exists());
    // Moved by hand.
    std::fs::rename(root.join("inbox/edit-lost.board"), root.join("calc/parser/edit-lost.board")).unwrap();
    let said = scan();
    assert_eq!(said.moved, vec![("inbox/edit-lost.board".to_string(), "calc/parser/edit-lost.board".to_string())]);
    assert!(boards.board("edit-lost").unwrap().is_some(), "still there");
    // Writes go to its new place.
    boards.add_hint(&board.id, "look at the lock").unwrap();
    assert!(std::fs::read_to_string(root.join("calc/parser/edit-lost.board")).unwrap().contains("look at the lock"));
    assert!(!root.join("inbox/edit-lost.board").exists());
    // Archived: out of the index; back: in again.
    tree::archive("calc/parser/edit-lost.board", true).unwrap();
    assert!(boards.board("edit-lost").unwrap().is_none(), "archived: nothing works on it");
    let open_all: HashSet<String> = [project.clone(), folder.clone()].into_iter().collect();
    assert!(!tree::list(&open_all, false).iter().any(|n| n.name == "edit-lost.board"));
    assert!(tree::list(&open_all, true).iter().any(|n| n.name == "edit-lost.board" && n.archived));
    tree::archive("calc/parser/edit-lost.board", false).unwrap();
    assert!(boards.board("edit-lost").unwrap().is_some());
    // A folder archived hides what is inside it.
    tree::archive(&folder, true).unwrap();
    assert!(boards.board("edit-lost").unwrap().is_none(), "under an archived folder");
    tree::archive(&folder, false).unwrap();
    assert!(boards.board("edit-lost").unwrap().is_some());
    // The trash, and back.
    tree::trash("calc/parser/edit-lost.board").unwrap();
    assert!(boards.board("edit-lost").unwrap().is_none());
    let trashed = tree::trash_list();
    assert_eq!(trashed.len(), 1);
    assert_eq!(trashed[0].from, "calc/parser/edit-lost.board");
    let back = tree::restore(&trashed[0].slot).unwrap();
    assert_eq!(back, "calc/parser/edit-lost.board");
    assert!(boards.board("edit-lost").unwrap().is_some());
    assert!(boards.hints(&board.id, false).unwrap().iter().any(|h| h.text == "look at the lock"), "nothing lost on the way");
    // Deleted for good.
    tree::trash(&back).unwrap();
    assert_eq!(tree::empty_trash().unwrap(), 1);
    assert!(tree::trash_list().is_empty());
    assert!(boards.board("edit-lost").unwrap().is_none());
}

/// A copy of a board's file is not taken for the board.
#[test]
fn a_copy_is_set_aside() {
    let (root, _, boards) = fresh();
    open(&boards, "edit-lost");
    scan();
    std::fs::copy(root.join("inbox/edit-lost.board"), root.join("inbox/edit-lost copy.board")).unwrap();
    let said = scan();
    assert_eq!(said.copies, vec!["inbox/edit-lost copy.board".to_string()]);
    // Found again by every scan while it is there: said once.
    let mut standing = Standing::default();
    assert_eq!(standing.news(scan()).copies.len(), 1);
    assert!(standing.news(scan()).is_empty());
    assert_eq!(path_of(Kind::Board, &boards.board("edit-lost").unwrap().unwrap().id), Some(root.join("inbox/edit-lost.board")));
}

/// Boards from before projects are written out once, into imported/.
#[test]
fn boards_from_before_are_imported() {
    let (root, data, boards) = fresh();
    // Made with no projects folder.
    set_for_test(None, &data);
    let board = open(&boards, "older");
    boards.add_hint(&board.id, "from before").unwrap();
    set_for_test(Some(root.clone()), &data);
    let said = scan();
    assert_eq!(said.found, vec!["imported/older.board".to_string()]);
    assert!(root.join("imported/index.md").exists());
    // Its file alone holds it all.
    journal::remove(Kind::Board, &board.id).unwrap();
    journal::replay(Kind::Board, &board.id, &log::read(&root.join("imported/older.board")).unwrap()).unwrap();
    assert!(boards.hints(&board.id, false).unwrap().iter().any(|h| h.text == "from before"));
    // Written once.
    assert!(scan().found.is_empty());
}

/// A change made to the file outside the stores reaches the index.
#[test]
fn an_edit_outside_reaches_the_index() {
    let (root, _, boards) = fresh();
    let board = open(&boards, "edit-lost");
    scan();
    let path = root.join("inbox/edit-lost.board");
    let mut file = log::read(&path).unwrap();
    let mut row = file.lines.iter().find(|l| log::op(l) == "board").unwrap()["row"].as_object().unwrap().clone();
    row.insert("goal".into(), "a goal written by hand".into());
    let line = serde_json::json!({"op": "board", "table": "boards", "row": row}).as_object().unwrap().clone();
    log::append(&path, vec![line]).unwrap();
    file = log::read(&path).unwrap();
    assert!(file.lines.len() > 1);
    let said = scan();
    assert_eq!(said.changed, vec!["inbox/edit-lost.board".to_string()]);
    assert_eq!(boards.board_by_id(&board.id).unwrap().unwrap().goal, "a goal written by hand");
}

/// Folders, pages, renames and moves, as the tab makes them.
#[test]
fn folders_and_pages_are_made_renamed_and_moved() {
    let (root, _, _) = fresh();
    let project = tree::create_folder("", "calc").unwrap();
    let page = tree::create_page(&project, "notes").unwrap();
    assert_eq!(page, "calc/notes.md");
    assert!(tree::create_page(&project, "notes.md").is_err(), "taken");
    assert!(tree::create_folder(&project, "../x").is_err());
    let renamed = tree::rename(&page, "design.md").unwrap();
    let folder = tree::create_folder(&project, "parser").unwrap();
    let moved = tree::move_to(&renamed, &folder).unwrap();
    assert_eq!(moved, "calc/parser/design.md");
    assert!(root.join("calc/parser/design.md").exists());
    let open: HashSet<String> = ["calc".to_string(), "calc/parser".to_string()].into_iter().collect();
    let names: Vec<(String, usize)> = tree::list(&open, false).into_iter().map(|n| (n.name, n.depth)).collect();
    assert_eq!(names, vec![("calc".to_string(), 0), ("parser".to_string(), 1), ("design.md".to_string(), 2)]);
    assert!(tree::move_to("calc", "calc/parser").is_err(), "not into itself");
}

/// An audit's file is made beside its board, and the tree shows it under
/// the board wherever its file goes.
#[test]
fn an_audit_goes_beside_its_board_and_shows_under_it() {
    let (root, data, boards) = fresh();
    let board = open(&boards, "edit-lost");
    scan();
    tree::create_folder("", "calc").unwrap();
    tree::move_to("inbox/edit-lost.board", "calc").unwrap();
    crate::audit::set_store_for_test(data.join("audit.sqlite3"));
    let store = crate::audit::store().unwrap();
    store
        .create(&crate::audit::Audit { name: "slice".into(), board: Some(board.name.clone()), created_at: crate::audit::now(), ..crate::audit::Audit::default() })
        .unwrap();
    assert!(root.join("calc/slice.audit").exists(), "beside its board");
    scan();
    // Moved apart: still under its board.
    std::fs::rename(root.join("calc/slice.audit"), root.join("inbox/slice.audit")).unwrap();
    scan();
    let open: HashSet<String> = ["calc".to_string(), "inbox".to_string()].into_iter().collect();
    let rows: Vec<(String, usize, bool)> = tree::list(&open, false).into_iter().map(|n| (n.name, n.depth, n.attached)).collect();
    assert!(rows.contains(&("slice.audit".to_string(), 2, true)), "{rows:?}");
    // Archiving the board archives its audit.
    tree::archive("calc/edit-lost.board", true).unwrap();
    assert!(store.audit("slice").unwrap().is_none());
    // Its name is its id in the files: not given to another meanwhile.
    let again = store.create(&crate::audit::Audit { name: "slice".into(), created_at: crate::audit::now(), ..crate::audit::Audit::default() });
    assert!(again.unwrap_err().contains("archived"));
    tree::archive("calc/edit-lost.board", false).unwrap();
    assert!(store.audit("slice").unwrap().is_some(), "and brings it back");
}

/// Redaction reaches the files.
#[test]
fn a_strike_reaches_the_files() {
    let (root, _, boards) = fresh();
    let board = open(&boards, "edit-lost");
    boards.add_hint(&board.id, "the key is sk-123456789").unwrap();
    tree::create_folder("", "calc").unwrap();
    std::fs::write(root.join("calc/notes.md"), "the key is sk-123456789\n").unwrap();
    let fragments = crate::strike::Fragments::new(&["sk-123456789".to_string()]).unwrap();
    for path in [root.join("inbox/edit-lost.board"), root.join("calc/notes.md")] {
        assert_eq!(crate::strike::strike_project_file_for_test(&path, &fragments).unwrap(), 1, "{}", path.display());
        assert!(!std::fs::read_to_string(&path).unwrap().contains("sk-123456789"));
    }
    assert!(log::read(&root.join("inbox/edit-lost.board")).unwrap().lines.iter().all(|l| !l.is_empty()));
}

/// Moved by hand, and written before a scan saw it: the write goes to the
/// moved file, and no second file is made for the board.
#[test]
fn a_write_finds_a_file_moved_before_the_scan() {
    let (root, _, boards) = fresh();
    let board = open(&boards, "edit-lost");
    scan();
    tree::create_folder("", "calc").unwrap();
    std::fs::rename(root.join("inbox/edit-lost.board"), root.join("calc/edit-lost.board")).unwrap();
    boards.add_hint(&board.id, "written before the scan").unwrap();
    assert!(std::fs::read_to_string(root.join("calc/edit-lost.board")).unwrap().contains("written before the scan"));
    assert!(!root.join("inbox/edit-lost.board").exists(), "no second file");
    assert_eq!(path_of(Kind::Board, &board.id), Some(root.join("calc/edit-lost.board")));
    let said = scan();
    assert!(said.copies.is_empty() && said.gone.is_empty(), "{said:?}");
}

fn call(name: &str, arguments: serde_json::Value) -> serde_json::Value {
    let call = crate::types::ToolCall {
        id: "t".into(),
        kind: "function".into(),
        function: crate::types::FunctionCall { name: name.into(), arguments: arguments.to_string() },
    };
    serde_json::from_str(&tools::execute(&call).unwrap()).unwrap()
}

/// A thread: the operator's comment, the agent's reply, resolved; his
/// reply opens it again. Hints and review notes are comments.
#[test]
fn comments_are_threads_kept_in_the_board_file() {
    use crate::board::comments::{AGENT, OPERATOR};
    let (root, data, boards) = fresh();
    crate::audit::set_store_for_test(data.join("audit.sqlite3"));
    let board = open(&boards, "edit-lost");
    let hint = boards.add_hint(&board.id, "look at the lock").unwrap();
    assert_eq!(boards.hints(&board.id, true).unwrap().len(), 1, "a hint is an open comment");
    let reply = boards.reply(&board.id, &hint.id, AGENT, "it is held twice").unwrap().unwrap();
    assert_eq!(reply.parent.as_deref(), Some(hint.id.as_str()));
    assert!(boards.resolve(&board.id, &reply.id, OPERATOR).unwrap(), "by any comment in the thread");
    assert!(boards.hints(&board.id, false).unwrap().is_empty(), "resolved: off the board");
    boards.reply(&board.id, &hint.id, OPERATOR, "one more thing").unwrap().unwrap();
    assert!(boards.threads(&board.id).unwrap()[0].root.open(), "his reply opens it again");
    // In the file, and back from it.
    let file = log::read(&root.join("inbox/edit-lost.board")).unwrap();
    assert!(file.lines.iter().any(|l| log::op(l) == "comment" && l["row"]["text"] == "it is held twice"));
    journal::remove(Kind::Board, &board.id).unwrap();
    journal::replay(Kind::Board, &board.id, &file).unwrap();
    let threads = boards.threads(&board.id).unwrap();
    assert_eq!((threads.len(), threads[0].replies.len()), (1, 2));
    // A hint from before comments is moved over.
    {
        let db = crate::db::open(data.join("board.sqlite3")).unwrap();
        db.execute("INSERT INTO hints VALUES('hint-0099', 99, ?1, 'from before', 'x', NULL)", [&board.id]).unwrap();
    }
    assert!(boards.hints(&board.id, false).unwrap().iter().any(|h| h.id == "hint-0099" && h.text == "from before"));
}

/// The agent's tools: read a project, comment and reply, write a page, open
/// a board in a folder.
#[test]
fn the_agents_project_tools() {
    let (root, data, boards) = fresh();
    crate::audit::set_store_for_test(data.join("audit.sqlite3"));
    let board = open(&boards, "edit-lost");
    tree::create_folder("", "calc").unwrap();
    tree::move_to("inbox/edit-lost.board", "calc").unwrap();
    // Read.
    let all = call("project_read", serde_json::json!({}));
    assert!(all["projects"].as_array().unwrap().iter().any(|p| p["project"] == "calc" && p["boards"][0]["board"] == "calc/edit-lost"), "{all}");
    let folder = call("project_read", serde_json::json!({"path": "calc"}));
    assert_eq!(folder["text"], "# calc\n");
    // Board paths address a board.
    assert_eq!(boards.board("calc/edit-lost").unwrap().unwrap().id, board.id);
    // A question on a row, and the operator's comment answered.
    let asked = call("comment", serde_json::json!({"on": "calc/edit-lost#i1", "text": "is i1 still worth it?"}));
    assert_eq!(asked["ok"], true, "{asked}");
    let threads = boards.threads(&board.id).unwrap();
    assert!(threads[0].waits_on_operator());
    let his = boards.add_hint(&board.id, "try the lock").unwrap();
    let refused = call("comment_reply", serde_json::json!({"comment": his.id, "text": "done", "resolve": true}));
    assert_eq!(refused["ok"], false, "only he resolves his");
    let replied = call("comment_reply", serde_json::json!({"comment": his.id, "text": "done"}));
    assert_eq!(replied["ok"], true);
    let own = call("comment_reply", serde_json::json!({"comment": asked["id"], "text": "never mind", "resolve": true}));
    assert_eq!(own["resolved"], true);
    // A page: written, its old version kept; comments on it.
    let written = call("page_write", serde_json::json!({"path": "calc", "markdown": "# calc\n\nThe parser and its boards.\n"}));
    assert_eq!(written["path"], "calc/index.md", "{written}");
    assert!(root.join(written["previous_kept_at"].as_str().unwrap()).exists());
    assert_eq!(call("page_write", serde_json::json!({"path": "../x.md", "markdown": "x"}))["ok"], false);
    let on_page = call("comment", serde_json::json!({"on": "calc", "text": "a section on the lexer?"}));
    assert_eq!(on_page["ok"], true, "{on_page}");
    let page = call("project_read", serde_json::json!({"path": "calc"}));
    assert_eq!(page["comments"][0]["text"], "a section on the lexer?");
    pages::reply(on_page["id"].as_str().unwrap(), crate::board::comments::OPERATOR, "yes").unwrap();
    assert_eq!(pages::threads("calc/index.md")[0].replies.len(), 1);
    // Renamed, the page keeps its comments.
    tree::create_page("calc", "notes").unwrap();
    pages::comment("calc/notes.md", crate::board::comments::OPERATOR, "on the notes").unwrap();
    let renamed = tree::rename("calc/notes.md", "design.md").unwrap();
    assert_eq!(pages::threads(&renamed).len(), 1);
    assert!(pages::threads("calc/notes.md").is_empty());
}

/// A board archived (out of the index) keeps its id: one made after it gets
/// a new id and a file of its own, never the archived board's.
#[test]
fn ids_are_never_given_twice() {
    let (root, _, boards) = fresh();
    let older = open(&boards, "older");
    boards.operator_fact(&older.id, "found before", Polarity::Supports, None, None).unwrap();
    scan();
    tree::archive("inbox/older.board", true).unwrap();
    assert!(boards.board_by_id(&older.id).unwrap().is_none(), "out of the index");
    let newer = open(&boards, "newer");
    assert_ne!(newer.id, older.id);
    let fact = boards.operator_fact(&newer.id, "found after", Polarity::Supports, None, None).unwrap();
    assert!(root.join("inbox/newer.board").exists(), "its own file");
    let old_file = std::fs::read_to_string(root.join("inbox/older.board")).unwrap();
    assert!(!old_file.contains("newer") && !old_file.contains("found after"), "nothing in the archived board's file");
    let old_fact = log::read(&root.join("inbox/older.board")).unwrap().lines.iter().find_map(|l| (l["table"] == "facts").then(|| l["row"]["id"].as_str().unwrap().to_string())).unwrap();
    assert_ne!(fact.id, old_fact, "findings' ids too");
    // Even with the index's own marks gone (a store from before): past the files'.
    let db = crate::db::open(root.parent().unwrap().join("data/board.sqlite3")).unwrap();
    db.execute("DELETE FROM id_marks", []).unwrap();
    drop(db);
    tree::archive("inbox/newer.board", true).unwrap();
    let third = open(&boards, "third");
    assert!(third.id != older.id && third.id != newer.id, "{}", third.id);
}

/// A file a board was written into under another's reused id is split: the
/// new board gets its own file, the old board's part is set aside.
#[test]
fn a_reused_id_is_taken_apart() {
    let (root, _, boards) = fresh();
    let board = open(&boards, "edit-lost");
    scan();
    // As an older build did: another board made with the same id, its lines
    // in this file (and written out to imported/ too).
    let path = root.join("inbox/edit-lost.board");
    let mut row = log::read(&path).unwrap().lines.iter().find(|l| log::op(l) == "board").unwrap()["row"].as_object().unwrap().clone();
    row.insert("name".into(), "cache-stale".into());
    row.insert("created_at".into(), "2099-01-01T00:00:00+00:00".into());
    log::append(&path, vec![serde_json::json!({"op": "board", "table": "boards", "row": row}).as_object().unwrap().clone()]).unwrap();
    let said = repair_reused_ids(&root);
    assert_eq!(said.len(), 1, "{said:?}");
    let own = root.join("inbox/cache-stale.board");
    assert!(own.exists());
    let file = log::read(&own).unwrap();
    assert_eq!(file.id(), board.id);
    assert!(file.lines.iter().all(|l| l["row"]["name"] != "edit-lost"), "only its own lines");
    assert!(!path.exists());
    assert!(root.join(".s0ma/reused-ids/edit-lost.board").exists(), "the old board's part kept");
    assert!(repair_reused_ids(&root).is_empty(), "once");
    // The file comes back as it was (another machine's copy), with a line
    // the board's own file lacks: the board keeps its file and takes the
    // line; no second file, nothing said.
    let aside = root.join(".s0ma/reused-ids/edit-lost.board");
    let mut back = log::read(&aside).unwrap().lines;
    back.extend(log::read(&own).unwrap().lines);
    back.push(serde_json::json!({"op": "note", "n": "other.1", "at": "2099-01-02T00:00:00.000Z", "text": "made elsewhere"}).as_object().unwrap().clone());
    log::create(&path, &log::read(&own).unwrap().header).unwrap();
    log::append(&path, back).unwrap();
    assert!(repair_reused_ids(&root).is_empty(), "not said again");
    assert!(!path.exists() && !root.join("inbox/cache-stale-2.board").exists(), "no second file");
    assert!(log::read(&own).unwrap().lines.iter().any(|l| log::n(l) == "other.1"), "its own lines taken");
}

/// A file whose rows bear ids another board's rows have in the index (made
/// while ids could be given twice) never takes them.
#[test]
fn a_replay_never_takes_another_boards_rows() {
    let (root, _, boards) = fresh();
    let first = open(&boards, "edit-lost");
    let theirs = boards.operator_fact(&first.id, "the save runs twice", Polarity::Supports, None, None).unwrap();
    let second = open(&boards, "cache-stale");
    scan();
    // The second board's file claims the first's finding id.
    let path = root.join("inbox/cache-stale.board");
    let mut row = log::read(&root.join("inbox/edit-lost.board")).unwrap().lines.iter().find(|l| log::op(l) == "finding").unwrap()["row"].as_object().unwrap().clone();
    row.insert("board_id".into(), second.id.clone().into());
    row.insert("statement".into(), "not the first board's".into());
    log::append(&path, vec![serde_json::json!({"op": "finding", "table": "facts", "row": row}).as_object().unwrap().clone()]).unwrap();
    let said = scan();
    assert_eq!(said.problems.len(), 1, "{said:?}");
    assert_eq!(boards.fact(&first.id, &theirs.id).unwrap().unwrap().statement, "the save runs twice");
    assert!(boards.board("cache-stale").unwrap().is_some());
}
