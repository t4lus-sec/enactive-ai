//! The editor's syntax colours: `syntect`'s bundled grammars, read line by
//! line, each token sorted into one of a few classes the tab draws in s0ma's
//! palette (`claude/editor-spec.md`, "Colours"). Its themes are not used: the
//! palette is small on purpose, so the gutter's marks of who changed what
//! stay the strongest colour on screen.
//!
//! The parser's state is kept at the start of every line read, so an edit
//! reads again only from its line down to the last one drawn.

use std::sync::OnceLock;
use syntect::parsing::{ParseState, Scope, ScopeStack, SyntaxReference, SyntaxSet};

/// Past this size a file is shown without colours.
pub const MAX_BYTES: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Plain,
    Keyword,
    String,
    Constant,
    Comment,
    Heading,
    Raw,
    Error,
}

fn syntaxes() -> &'static SyntaxSet {
    static SET: OnceLock<SyntaxSet> = OnceLock::new();
    SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

/// Scope prefixes, most specific first, and their class. The innermost scope
/// of a token decides; failing that, the next one out.
fn table() -> &'static [(Scope, Class)] {
    static TABLE: OnceLock<Vec<(Scope, Class)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        [
            ("invalid", Class::Error),
            ("comment", Class::Comment),
            ("punctuation.definition.comment", Class::Comment),
            ("markup.heading", Class::Heading),
            ("entity.name.section", Class::Heading),
            ("markup.raw", Class::Raw),
            ("markup.inline.raw", Class::Raw),
            ("markup.underline.link", Class::Constant),
            ("string.other.link", Class::Constant),
            ("markup.bold", Class::Heading),
            ("string", Class::String),
            ("constant.numeric", Class::Constant),
            ("constant.language", Class::Constant),
            ("constant.character", Class::Constant),
            ("constant.other", Class::Constant),
            ("keyword", Class::Keyword),
            ("storage", Class::Keyword),
            ("variable.language", Class::Keyword),
        ]
        .into_iter()
        .filter_map(|(name, class)| Scope::new(name).ok().map(|scope| (scope, class)))
        .collect()
    })
}

fn classify(stack: &ScopeStack) -> Class {
    for scope in stack.as_slice().iter().rev() {
        if let Some((_, class)) = table().iter().find(|(prefix, _)| prefix.is_prefix_of(*scope)) {
            return *class;
        }
    }
    Class::Plain
}

/// What a file is, by its name: the grammar, a label for the footer, the
/// line comment, and whether it reads as prose (wrapped by default).
#[derive(Clone, Debug)]
pub struct Language {
    pub name: String,
    pub comment: Option<&'static str>,
    pub prose: bool,
    /// Indent with tabs (Go, makefiles), or this many spaces when the file
    /// doesn't say.
    pub tabs: bool,
    pub width: usize,
    syntax: Option<String>,
}

pub fn language(file: &str) -> Language {
    let name = file.rsplit('/').next().unwrap_or(file);
    let extension = name.rsplit_once('.').map(|(_, ext)| ext.to_ascii_lowercase()).unwrap_or_default();
    let special = match name {
        "Makefile" | "makefile" | "GNUmakefile" => Some("make"),
        "Dockerfile" => Some("Dockerfile"),
        _ => None,
    };
    let syntax = syntaxes()
        .find_syntax_by_extension(special.unwrap_or(&extension))
        .or_else(|| special.and_then(|name| syntaxes().find_syntax_by_name(name)));
    let comment = match extension.as_str() {
        "rs" | "js" | "jsx" | "ts" | "tsx" | "c" | "h" | "cc" | "cpp" | "hpp" | "go" | "java" | "kt" | "swift" | "cs" | "scala" | "dart" | "zig" => Some("// "),
        "py" | "sh" | "bash" | "zsh" | "rb" | "pl" | "yaml" | "yml" | "toml" | "conf" | "ini" | "cfg" | "r" | "nix" | "ps1" | "tf" => Some("# "),
        "sql" | "lua" | "hs" => Some("-- "),
        "el" | "lisp" | "clj" | "scm" => Some(";; "),
        "tex" => Some("% "),
        _ if special.is_some() => Some("# "),
        _ => None,
    };
    let prose = matches!(extension.as_str(), "md" | "markdown" | "mdx" | "txt" | "rst" | "org" | "adoc" | "") && special.is_none();
    let two = matches!(extension.as_str(), "yaml" | "yml" | "json" | "js" | "jsx" | "ts" | "tsx" | "md" | "markdown" | "rb" | "html" | "css" | "scss" | "nix" | "tf");
    Language {
        name: syntax.map_or_else(|| if prose { "text".into() } else { extension.clone() }, |syntax| syntax.name.to_lowercase()),
        comment,
        prose,
        tabs: special == Some("make") || extension == "go",
        width: if two { 2 } else { 4 },
        syntax: syntax.map(|syntax| syntax.name.clone()),
    }
}

/// A file's colours, as far as they have been read.
pub struct Highlighter {
    syntax: Option<&'static SyntaxReference>,
    /// The parser at the start of each line read, and the classes of each.
    states: Vec<(ParseState, ScopeStack)>,
    spans: Vec<Vec<(usize, usize, Class)>>,
    pub off: bool,
}

impl Highlighter {
    pub fn new(language: &Language, size: usize) -> Self {
        let syntax = language.syntax.as_deref().and_then(|name| syntaxes().find_syntax_by_name(name));
        let mut highlighter = Self { syntax, states: Vec::new(), spans: Vec::new(), off: size > MAX_BYTES };
        if let Some(syntax) = syntax {
            highlighter.states.push((ParseState::new(syntax), ScopeStack::new()));
        }
        highlighter
    }

    /// Lines from `line` on were changed: read again when next drawn.
    pub fn invalidate(&mut self, line: usize) {
        self.spans.truncate(line);
        self.states.truncate(line + 1);
    }

    /// The classes of line `index`, reading down to it as needed.
    pub fn line(&mut self, lines: &[String], index: usize) -> &[(usize, usize, Class)] {
        if self.off || self.syntax.is_none() || index >= lines.len() {
            return &[];
        }
        while self.spans.len() <= index {
            let at = self.spans.len();
            let Some((mut state, mut stack)) = self.states.get(at).cloned() else {
                return &[];
            };
            let mut text = lines[at].clone();
            text.push('\n');
            let mut spans = Vec::new();
            match state.parse_line(&text, syntaxes()) {
                Ok(ops) => {
                    let mut from = 0;
                    for (offset, op) in ops {
                        let offset = offset.min(lines[at].len());
                        if offset > from {
                            spans.push((from, offset, classify(&stack)));
                        }
                        from = from.max(offset);
                        if stack.apply(&op).is_err() {
                            break;
                        }
                    }
                    if from < lines[at].len() {
                        spans.push((from, lines[at].len(), classify(&stack)));
                    }
                }
                // A grammar that fails on a line: the rest plain.
                Err(_) => self.off = true,
            }
            self.spans.push(spans);
            self.states.push((state, stack));
        }
        &self.spans[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classes(file: &str, text: &str) -> Vec<Vec<(String, Class)>> {
        let lines: Vec<String> = text.lines().map(str::to_string).collect();
        let mut highlighter = Highlighter::new(&language(file), text.len());
        (0..lines.len())
            .map(|index| {
                highlighter
                    .line(&lines, index)
                    .iter()
                    .filter(|(_, _, class)| *class != Class::Plain)
                    .map(|(start, end, class)| (lines[index][*start..*end].to_string(), *class))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn tokens_take_the_palette_classes() {
        let rust = classes("main.rs", "// note\nfn main() { let x = \"hi\"; 42 }");
        assert_eq!(rust[0][0].1, Class::Comment);
        assert!(rust[1].contains(&("fn".into(), Class::Keyword)), "{:?}", rust[1]);
        assert!(rust[1].iter().any(|(text, class)| text.contains("hi") && *class == Class::String), "{:?}", rust[1]);
        assert!(rust[1].contains(&("42".into(), Class::Constant)), "{:?}", rust[1]);
        let markdown = classes("plan.md", "# Plan\nsome `code` here");
        assert!(markdown[0].iter().any(|(text, class)| text.contains("Plan") && *class == Class::Heading), "{:?}", markdown[0]);
        assert!(markdown[1].iter().any(|(text, class)| text.contains("code") && *class == Class::Raw), "{:?}", markdown[1]);
        let python = classes("x.py", "x = \"\"\"doc\nstill doc\n\"\"\"\ny = 1");
        assert!(python[1].iter().any(|(_, class)| *class == Class::String), "a string over lines: {:?}", python[1]);
    }

    #[test]
    fn an_edit_reads_again_from_its_line() {
        let mut lines: Vec<String> = vec!["x = 1".into(), "y = 2".into()];
        let mut highlighter = Highlighter::new(&language("a.py"), 10);
        assert!(!highlighter.line(&lines, 1).is_empty());
        lines[0] = "x = \"\"\"open".into();
        highlighter.invalidate(0);
        assert!(highlighter.line(&lines, 1).iter().all(|(_, _, class)| *class == Class::String), "now inside the string");
    }

    #[test]
    fn languages_by_name() {
        assert_eq!(language("a/b/plan.md").name, "markdown");
        assert!(language("plan.md").prose && !language("main.rs").prose);
        assert_eq!(language("main.rs").comment, Some("// "));
        assert_eq!(language("x.yaml").width, 2);
        assert!(language("Makefile").tabs);
        assert_eq!(language("notes").name, "text");
        let big = Highlighter::new(&language("main.rs"), MAX_BYTES + 1);
        assert!(big.off, "too big to colour");
    }
}
