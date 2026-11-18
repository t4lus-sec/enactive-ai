//! Comments: one kind of note on a board, in place of hints and review
//! notes. A comment is on the board or one of its rows (`i8`, `fact-0003`),
//! a round's report (`report:2`) or an audit's claim (`claim:slice/c4`); it
//! is the operator's, the agent's or a subagent's; it has replies, and stays
//! open until resolved. One written while the board is in review carries its
//! round: the agent replies to each before reporting again.
//!
//! The rows live in `board.sqlite3` (`comments`), journaled into the board's
//! file like every other row. A store from before is moved over once: its
//! hints as standing comments on the board, its review notes as comments of
//! their round with the agent's answer as a reply.

use super::*;
use serde::{Deserialize, Serialize};

/// Who wrote it.
pub const OPERATOR: &str = "operator";
pub const AGENT: &str = "agent";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Comment {
    pub id: String,
    pub board_id: String,
    /// `board`, a row's id, `report:<round>`, `claim:<audit>/<claim>`.
    pub on: String,
    /// The review round it was written in.
    pub round: Option<i64>,
    pub author: String,
    /// The thread's first comment, for a reply.
    pub parent: Option<String>,
    pub text: String,
    pub created_at: String,
    pub resolved_at: Option<String>,
    pub resolved_by: Option<String>,
    /// When the agent was first shown it.
    pub seen_at: Option<String>,
}

impl Comment {
    pub fn open(&self) -> bool {
        self.resolved_at.is_none()
    }

    pub fn by_operator(&self) -> bool {
        self.author == OPERATOR
    }
}

/// A comment and its replies, oldest first.
#[derive(Debug, Clone, PartialEq)]
pub struct Thread {
    pub root: Comment,
    pub replies: Vec<Comment>,
}

impl Thread {
    /// The last word was the agent's (or a subagent's) and the thread is
    /// open: it waits on the operator.
    pub fn waits_on_operator(&self) -> bool {
        self.root.open() && !self.last().by_operator()
    }

    /// Open, and the operator's word is the last: the agent owes a reply.
    pub fn waits_on_agent(&self) -> bool {
        self.root.open() && self.last().by_operator()
    }

    pub fn last(&self) -> &Comment {
        self.replies.last().unwrap_or(&self.root)
    }

    /// What it is on, said: `the board`, `i8`, `report 2`, `claim slice/c4`.
    pub fn on_said(&self) -> String {
        said_on(&self.root.on)
    }
}

pub fn said_on(on: &str) -> String {
    match on {
        "board" | "" => "the board".into(),
        on => match on.split_once(':') {
            Some(("report", round)) => format!("report {round}"),
            Some((kind, what)) => format!("{kind} {what}"),
            None => on.to_string(),
        },
    }
}

pub(super) const TABLE: &str = "
             CREATE TABLE IF NOT EXISTS comments(
                id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL,
                board_id TEXT NOT NULL REFERENCES boards(id),
                on_what TEXT NOT NULL,
                round INTEGER,
                author TEXT NOT NULL,
                parent TEXT,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                resolved_by TEXT,
                seen_at TEXT);
             CREATE INDEX IF NOT EXISTS comments_board ON comments(board_id, seq);";

const SELECT: &str = "SELECT id,board_id,on_what,round,author,parent,text,created_at,resolved_at,resolved_by,seen_at FROM comments";

fn row(row: &rusqlite::Row) -> rusqlite::Result<Comment> {
    Ok(Comment {
        id: row.get(0)?,
        board_id: row.get(1)?,
        on: row.get(2)?,
        round: row.get(3)?,
        author: row.get(4)?,
        parent: row.get(5)?,
        text: row.get(6)?,
        created_at: row.get(7)?,
        resolved_at: row.get(8)?,
        resolved_by: row.get(9)?,
        seen_at: row.get(10)?,
    })
}

impl BoardStore {
    /// A new thread on `on` (`board` when empty).
    pub fn comment(&self, board_id: &str, on: &str, round: Option<i64>, author: &str, text: &str) -> Result<Comment> {
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (id, seq) = Self::next_id(&tx, "comments", "c")?;
        let on = if on.trim().is_empty() { "board" } else { on.trim() };
        tx.execute(
            "INSERT INTO comments(id,seq,board_id,on_what,round,author,parent,text,created_at) VALUES(?1,?2,?3,?4,?5,?6,NULL,?7,?8)",
            params![id, seq, board_id, on, round, author, text, Local::now().to_rfc3339()],
        )?;
        tx.commit()?;
        self.comment_by_id(&id)?.ok_or_else(|| "the comment vanished after it was written".into())
    }

    /// A reply in `to`'s thread (a reply's reply goes to its thread). The
    /// operator's reply to a resolved thread opens it again. Err (inner):
    /// no such comment on the board.
    pub fn reply(&self, board_id: &str, to: &str, author: &str, text: &str) -> Result<std::result::Result<Comment, String>> {
        let Some(target) = self.comment_by_id(to)?.filter(|c| c.board_id == board_id) else {
            return Ok(Err(format!("no comment {to} on this board")));
        };
        let root = target.parent.clone().unwrap_or(target.id.clone());
        let Some(root_comment) = self.comment_by_id(&root)? else {
            return Ok(Err(format!("{to}'s thread is gone")));
        };
        let mut db = self.connect()?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (id, seq) = Self::next_id(&tx, "comments", "c")?;
        tx.execute(
            "INSERT INTO comments(id,seq,board_id,on_what,round,author,parent,text,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![id, seq, board_id, root_comment.on, root_comment.round, author, root, text, Local::now().to_rfc3339()],
        )?;
        if author == OPERATOR && !root_comment.open() {
            tx.execute("UPDATE comments SET resolved_at=NULL, resolved_by=NULL, seen_at=NULL WHERE id=?1", params![root])?;
        }
        tx.commit()?;
        Ok(Ok(self.comment_by_id(&id)?.ok_or("the reply vanished after it was written")?))
    }

    /// A thread resolved (by its first comment's id, or any reply's).
    /// Whether it was open.
    pub fn resolve(&self, board_id: &str, id: &str, by: &str) -> Result<bool> {
        let root = self.root_of(board_id, id)?;
        let db = self.connect()?;
        Ok(db.execute(
            "UPDATE comments SET resolved_at=?1, resolved_by=?2 WHERE board_id=?3 AND id=?4 AND resolved_at IS NULL",
            params![Local::now().to_rfc3339(), by, board_id, root],
        )? > 0)
    }

    /// A thread taken back whole, while nobody has replied. Whether it went.
    pub fn take_back(&self, board_id: &str, id: &str) -> Result<bool> {
        let db = self.connect()?;
        let replies: i64 = db.query_row("SELECT COUNT(*) FROM comments WHERE parent=?1", params![id], |r| r.get(0))?;
        if replies > 0 {
            return Ok(false);
        }
        Ok(db.execute("DELETE FROM comments WHERE board_id=?1 AND id=?2", params![board_id, id])? > 0)
    }

    fn root_of(&self, board_id: &str, id: &str) -> Result<String> {
        Ok(match self.comment_by_id(id)?.filter(|c| c.board_id == board_id) {
            Some(comment) => comment.parent.unwrap_or(comment.id),
            None => id.to_string(),
        })
    }

    pub fn comment_by_id(&self, id: &str) -> Result<Option<Comment>> {
        let db = self.connect()?;
        Ok(db.query_row(&format!("{SELECT} WHERE id=?1"), params![id], row).optional()?)
    }

    /// Every thread on the board, oldest first.
    pub fn threads(&self, board_id: &str) -> Result<Vec<Thread>> {
        self.move_notes_over()?;
        let db = self.connect()?;
        let mut statement = db.prepare(&format!("{SELECT} WHERE board_id=?1 ORDER BY seq"))?;
        let all: Vec<Comment> = statement.query_map(params![board_id], row)?.collect::<rusqlite::Result<_>>()?;
        Ok(group(all))
    }

    /// The agent was shown these.
    pub fn mark_seen(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let db = self.connect()?;
        let now = Local::now().to_rfc3339();
        for id in ids {
            db.execute("UPDATE comments SET seen_at=?1 WHERE id=?2 AND seen_at IS NULL", params![now, id])?;
        }
        Ok(())
    }

    /// The operator's comments and replies in open threads the agent has not
    /// been shown yet.
    pub fn unread_from_operator(&self, board_id: &str) -> Result<usize> {
        Ok(self
            .threads(board_id)?
            .iter()
            .filter(|t| t.root.open() && t.root.round.is_none())
            .map(|t| std::iter::once(&t.root).chain(t.replies.iter()).filter(|c| c.by_operator() && c.seen_at.is_none()).count())
            .sum())
    }

    /// Open threads waiting on the operator, on every board: the agent's
    /// questions and replies he has not answered or resolved.
    pub fn waiting_on_operator(&self) -> Result<usize> {
        let mut count = 0;
        for board in self.boards(false)? {
            count += self.threads(&board.id)?.iter().filter(|t| t.waits_on_operator()).count();
        }
        Ok(count)
    }

    /// The hints and review notes of a store from before, as comments.
    /// Done through the journaled connection, so the board's file has them.
    pub(super) fn move_notes_over(&self) -> Result<()> {
        let mut db = self.connect()?;
        let has_hints: bool = db.query_row("SELECT EXISTS(SELECT 1 FROM hints)", [], |r| r.get(0))?;
        if has_hints {
            let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let hints: Vec<(String, String, String, String, Option<String>)> = {
                let mut statement = tx.prepare("SELECT id,board_id,text,created_at,consumed_at FROM hints ORDER BY seq")?;
                statement.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?.collect::<rusqlite::Result<_>>()?
            };
            for (id, board_id, text, created_at, seen_at) in hints {
                let (on, text) = split_on(&text);
                let seq: i64 = tx.query_row("SELECT COALESCE(MAX(seq),0)+1 FROM comments", [], |r| r.get(0))?;
                tx.execute(
                    "INSERT OR IGNORE INTO comments(id,seq,board_id,on_what,round,author,parent,text,created_at,seen_at) VALUES(?1,?2,?3,?4,NULL,?5,NULL,?6,?7,?8)",
                    params![id, seq, board_id, on, OPERATOR, text, created_at, seen_at],
                )?;
            }
            tx.execute("DELETE FROM hints", [])?;
            tx.commit()?;
        }
        Ok(())
    }

    /// Review notes from the flow store, as comments of their round, the
    /// agent's answer a reply. The notes go from there once written here.
    pub(crate) fn take_notes(&self, notes: &[flow::Note]) -> Result<()> {
        for note in notes {
            if self.board_by_id(&note.board_id)?.is_none() {
                continue;
            }
            let (on, text) = split_on(&note.text);
            let root = self.comment(&note.board_id, &on, Some(note.round), OPERATOR, &text)?;
            if let Some(answer) = &note.answer {
                let _ = self.reply(&note.board_id, &root.id, AGENT, answer)?;
            }
        }
        Ok(())
    }
}

/// `on slice/c4: text` (how notes on a claim were written): what it is on,
/// and the text.
pub fn split_on(text: &str) -> (String, String) {
    if let Some(rest) = text.strip_prefix("on ")
        && let Some((what, said)) = rest.split_once(": ")
        && what.contains('/')
        && !what.contains(' ')
    {
        return (format!("claim:{what}"), said.to_string());
    }
    ("board".to_string(), text.to_string())
}

/// Comments into threads, in order.
pub fn group(all: Vec<Comment>) -> Vec<Thread> {
    let mut threads: Vec<Thread> = Vec::new();
    let mut replies: Vec<Comment> = Vec::new();
    for comment in all {
        if comment.parent.is_none() {
            threads.push(Thread { root: comment, replies: Vec::new() });
        } else {
            replies.push(comment);
        }
    }
    for reply in replies {
        if let Some(thread) = threads.iter_mut().find(|t| Some(&t.root.id) == reply.parent.as_ref()) {
            thread.replies.push(reply);
        }
    }
    threads
}
